"""Gate 1a, level L2: analytic-oracle integration test (D029).

The full training loop (real buffer, target network, schedule, Adam) runs on
a 3-state MDP whose optimal Q-values were derived by hand.  The learned
network must match them.  L1 tests check components; L2 checks that they
compose into correct learning.

The MDP ("quit chain")
----------------------
States S, A, B, observed as one-hot vectors, so the network can represent
any Q-table exactly.  Two actions:

    from    advance (0)                 quit (1)
    S       -> A, reward 0              terminates, reward +1
    A       -> B, reward 0              terminates, reward 0
    B       -> B, reward +1             terminates, reward 0

* After a quit the observation is the state quit from.  So if the loop
  failed to pass `terminated` to the buffer, the quit targets would pick up
  a large bootstrap term gamma * V(state).
* Episodes are truncated after 4 steps, unless the 4th step is a quit.  The
  greedy path is S -> A -> B -> B -> B: half the B -> B transitions are
  truncated, so masking the bootstrap on truncation, or storing the reset
  observation (S) as s', moves Q(B, advance) far from its true value.
* The time limit is not part of the MDP, so Q* is the infinite-horizon value.
* Reset is always S.  The env draws no random numbers.

Q* (gamma = 0.9), derived by hand by the user, 2026-10-20, and cross-checked
here by value iteration:

             advance   quit
        S      8.1      1
        A      9        0
        B     10        0

Fixed before the test was first run (D029: "tolerance fixed in the test
before it is run"): the tolerance, the training config and the seeds below.
If a correct-looking loop fails, investigate.  Do not widen the tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from dqn.config import DQNConfig
from dqn.runlog import RunLogger
from dqn.train import train

GAMMA = 0.9
TOLERANCE = 0.5
SEEDS = (1, 2, 3)
TIME_LIMIT = 4

S, A, B = 0, 1, 2
ADVANCE, QUIT = 0, 1
N_STATES, N_ACTIONS = 3, 2

# Hand-derived oracle (rows: S, A, B; columns: advance, quit).
Q_STAR = np.array([[8.1, 1.0],
                   [9.0, 0.0],
                   [10.0, 0.0]])

L2_CONFIG = dict(
    total_timesteps=30_000,
    learning_starts=1_000,
    train_frequency=1,
    batch_size=64,
    buffer_size=10_000,
    target_network_frequency=100,
    learning_rate=1e-3,
    hidden_sizes=(32,),
    gamma=GAMMA,
    start_e=1.0,
    end_e=0.1,
    exploration_fraction=0.3,
)

# (state, action) -> (next_state, reward, terminated)
TRANSITIONS = {
    (S, ADVANCE): (A, 0.0, False),
    (A, ADVANCE): (B, 0.0, False),
    (B, ADVANCE): (B, 1.0, False),
    (S, QUIT): (S, 1.0, True),
    (A, QUIT): (A, 0.0, True),
    (B, QUIT): (B, 0.0, True),
}


def _one_hot(state: int) -> np.ndarray:
    v = np.zeros(N_STATES)
    v[state] = 1.0
    return v


class QuitChainEnv:
    """The MDP above, with the reset/step protocol train() expects."""

    def __init__(self, seed=None):
        self.state = S
        self.t = 0
        self.done = True

    def reset(self, seed=None):
        self.state, self.t, self.done = S, 0, False
        return _one_hot(S), {}

    def step(self, action):
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset()")
        next_state, reward, terminated = TRANSITIONS[(self.state, int(action))]
        self.t += 1
        truncated = (not terminated) and self.t >= TIME_LIMIT
        self.state = next_state
        self.done = terminated or truncated
        return _one_hot(next_state), reward, terminated, truncated, {}


def _value_iteration(gamma: float, sweeps: int = 2_000) -> np.ndarray:
    """Infinite-horizon Q* by repeated Bellman optimality backups."""
    q = np.zeros((N_STATES, N_ACTIONS))
    for _ in range(sweeps):
        new = np.empty_like(q)
        for (s, a), (s2, r, term) in TRANSITIONS.items():
            new[s, a] = r + (0.0 if term else gamma * q[s2].max())
        q = new
    return q


def test_hand_derived_q_star_matches_value_iteration():
    np.testing.assert_allclose(_value_iteration(GAMMA), Q_STAR, atol=1e-9)


def test_env_greedy_episode_truncates_at_time_limit():
    env = QuitChainEnv()
    env.reset()
    path = []
    for _ in range(TIME_LIMIT):
        obs, r, term, trunc, _ = env.step(ADVANCE)
        path.append((int(np.argmax(obs)), r, term, trunc))
    assert path == [(A, 0.0, False, False), (B, 0.0, False, False),
                    (B, 1.0, False, False), (B, 1.0, False, True)]
    with pytest.raises(RuntimeError):
        env.step(ADVANCE)


def test_env_quit_terminates_in_place():
    env = QuitChainEnv()
    env.reset()
    env.step(ADVANCE)
    obs, r, term, trunc, _ = env.step(QUIT)
    assert int(np.argmax(obs)) == A and r == 0.0 and term and not trunc


def test_quit_on_last_step_is_terminated_not_truncated():
    env = QuitChainEnv()
    env.reset()
    for _ in range(TIME_LIMIT - 1):
        env.step(ADVANCE)
    _, _, term, trunc, _ = env.step(QUIT)
    assert term and not trunc


@pytest.mark.parametrize("seed", SEEDS)
def test_learned_q_matches_q_star(seed, tmp_path):
    cfg = DQNConfig().with_overrides({**L2_CONFIG, "seed": seed})
    with RunLogger(tmp_path / "run", cfg) as logger:
        result = train(cfg, env_fn=QuitChainEnv, obs_dim=N_STATES,
                       n_actions=N_ACTIONS, logger=logger)
    q, _ = result.online.forward(np.eye(N_STATES))
    err = np.abs(q - Q_STAR)
    table = "\n".join(f"  {name}: learned {q[i].round(3)}  Q* {Q_STAR[i]}"
                      for i, name in enumerate("SAB"))
    assert err.max() <= TOLERANCE, (
        f"seed {seed}: max |Q - Q*| = {err.max():.3f} > {TOLERANCE}\n{table}")
