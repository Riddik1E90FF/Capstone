"""Gate 1a, level L2: mutation test — does the L2 oracle actually catch bugs?

`test_l2_chain.py` asserts the trained network matches a hand-derived Q*.  That
is only evidence if a *wrong* loop would fail it.  A tolerance of +/-0.5 on
values up to 10 is not obviously tight, and "max error 0.003" means nothing
until you know what a defective implementation scores.

So this file plants the defects deliberately and asserts the oracle rejects
every one.  The logic is inverted from a normal test: each case passes when the
L2 check *fails*.

D029's Attempts table records that five planted bugs each failed by >= 4.5
against a 0.5 tolerance.  That was done once by hand in a scratch copy.  This
file makes the same claim reproducible, and guards the gate itself: if a later
refactor of `train()` silently weakens what L2 detects, this goes red.

How the defects are injected
----------------------------
`train()` takes `env_fn`, so most bugs can be simulated from outside without
editing the loop.  A lying environment produces exactly the same *stored
transitions* a buggy loop would produce, which is what the learning sees:

  * reporting a truncation as a termination  == `buffer.add(..., term or trunc)`
  * reporting a termination as a truncation  == never passing `terminated`
  * returning the reset observation as s'    == storing the post-reset obs

The remaining two are hyperparameters, so a config override is the real thing
rather than a simulation.

Runtime note: MUTANT_SEEDS is one seed by default to keep the suite fast.
Widening it to test_l2_chain.SEEDS is more rigorous and roughly triples the
cost of this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pytest

# pytest inserts tests/ on sys.path (no __init__.py here), so the chain MDP,
# its oracle and its config are reused rather than duplicated.  Keeping one
# definition means this file cannot drift from the test it validates.
from test_l2_chain import (
    L2_CONFIG,
    N_ACTIONS,
    N_STATES,
    Q_STAR,
    TOLERANCE,
    QuitChainEnv,
)

from dqn.config import DQNConfig
from dqn.runlog import RunLogger
from dqn.train import train

# One seed keeps `pytest tests/` quick.  Use test_l2_chain.SEEDS for the
# thorough version, e.g. before a gate write-up.
MUTANT_SEEDS = (1,)

EnvFn = Callable[[int | None], Any]


# --------------------------------------------------------------------------
# Environment wrappers.  Each produces the transitions a specific bug would
# store, without modifying dqn/train.py.
# --------------------------------------------------------------------------

def _truncation_as_termination(env_fn: EnvFn) -> EnvFn:
    """Bug: the loop masks the bootstrap on truncation.

    Equivalent to `buffer.add(..., terminated or truncated)`.  Half the
    B -> B transitions are truncations, so Q(B, advance) loses its bootstrap
    and collapses well below 10.
    """
    class Wrapped:
        def __init__(self, seed=None):
            self._env = env_fn(seed)

        def reset(self, seed=None):
            return self._env.reset(seed=seed)

        def step(self, action):
            obs, reward, terminated, truncated, info = self._env.step(action)
            return obs, reward, bool(terminated or truncated), truncated, info

    return Wrapped


def _termination_never_reported(env_fn: EnvFn) -> EnvFn:
    """Bug: `terminated` never reaches the buffer.

    Reported as a truncation instead, so the episode still ends but the
    transition keeps its bootstrap term.  Quit actions then pick up
    gamma * V(state) and Q(S, quit) inflates past its true value of 1.
    """
    class Wrapped:
        def __init__(self, seed=None):
            self._env = env_fn(seed)

        def reset(self, seed=None):
            return self._env.reset(seed=seed)

        def step(self, action):
            obs, reward, terminated, truncated, info = self._env.step(action)
            return obs, reward, False, bool(truncated or terminated), info

    return Wrapped


def _reset_obs_as_next_obs(env_fn: EnvFn) -> EnvFn:
    """Bug: the post-reset observation is stored as s'.

    On the last step of an episode the agent appears to transition into S
    rather than into the state it actually reached, so the bootstrap reads
    the wrong row of the Q-table.
    """
    class Wrapped:
        def __init__(self, seed=None):
            self._env = env_fn(seed)
            self._reset_obs = None

        def reset(self, seed=None):
            obs, info = self._env.reset(seed=seed)
            self._reset_obs = np.asarray(obs).copy()
            return obs, info

        def step(self, action):
            obs, reward, terminated, truncated, info = self._env.step(action)
            if terminated or truncated:
                obs = self._reset_obs.copy()
            return obs, reward, terminated, truncated, info

    return Wrapped


# --------------------------------------------------------------------------
# The mutants
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Mutant:
    name: str
    bug: str                                   # the real defect being simulated
    overrides: dict = field(default_factory=dict)
    wrap: Callable[[EnvFn], EnvFn] | None = None


MUTANTS = [
    Mutant(
        name="wrong_gamma",
        bug="gamma is 0.99 instead of the MDP's 0.9",
        overrides={"gamma": 0.99},
    ),
    Mutant(
        name="target_never_syncs",
        bug="hard_sync never runs, so the bootstrap comes from the init net",
        overrides={"target_network_frequency": 10**9},
    ),
    Mutant(
        name="truncation_masked",
        bug="truncation is treated as termination; bootstrap wrongly dropped",
        wrap=_truncation_as_termination,
    ),
    Mutant(
        name="terminated_not_stored",
        bug="terminated never reaches the buffer; terminal states bootstrap",
        wrap=_termination_never_reported,
    ),
    Mutant(
        name="reset_obs_as_next_obs",
        bug="the post-reset observation is stored as s'",
        wrap=_reset_obs_as_next_obs,
    ),
]


def _max_error(mutant: Mutant, seed: int, tmp_path) -> float:
    """Train under `mutant` and return max |Q - Q*| against the real oracle."""
    cfg = DQNConfig().with_overrides({**L2_CONFIG, **mutant.overrides, "seed": seed})
    env_fn = mutant.wrap(QuitChainEnv) if mutant.wrap else QuitChainEnv
    with RunLogger(tmp_path / f"mutant_{mutant.name}_{seed}", cfg) as logger:
        result = train(cfg, env_fn=env_fn, obs_dim=N_STATES,
                       n_actions=N_ACTIONS, logger=logger)
    q, _ = result.online.forward(np.eye(N_STATES))
    return float(np.abs(q - Q_STAR).max())


@pytest.mark.parametrize("seed", MUTANT_SEEDS)
@pytest.mark.parametrize("mutant", MUTANTS, ids=lambda m: m.name)
def test_l2_oracle_rejects_planted_bug(mutant: Mutant, seed: int, tmp_path):
    """Every planted defect must push max |Q - Q*| past the L2 tolerance.

    A failure here does not mean the DQN is broken.  It means the L2 oracle
    would not have noticed this defect, which is worse: the gate is weaker
    than D029 claims.
    """
    err = _max_error(mutant, seed, tmp_path)
    margin = err / TOLERANCE
    print(f"\n  {mutant.name:<24} max|Q-Q*| = {err:7.3f}  "
          f"({margin:5.1f}x tolerance)  <- {mutant.bug}")
    assert err > TOLERANCE, (
        f"{mutant.name!r} (seed {seed}) produced max |Q - Q*| = {err:.3f}, "
        f"within the L2 tolerance of {TOLERANCE}.\n"
        f"The planted bug was: {mutant.bug}\n"
        f"L2 would not have caught it. Either the MDP does not exercise this "
        f"defect, or the tolerance is too loose. Do not 'fix' this by "
        f"tightening the tolerance without re-deriving it."
    )


def test_unmutated_loop_still_passes(tmp_path):
    """Control: with no mutation applied, the same harness passes L2.

    Without this, every mutant could be failing for an unrelated reason —
    a broken harness rather than the planted bug.
    """
    control = Mutant(name="control", bug="none")
    err = _max_error(control, MUTANT_SEEDS[0], tmp_path)
    print(f"\n  {'control (no mutation)':<24} max|Q-Q*| = {err:7.3f}")
    assert err <= TOLERANCE, (
        f"The unmutated loop scored {err:.3f} > {TOLERANCE}. The mutation "
        f"harness itself is broken, so the mutant results mean nothing."
    )
