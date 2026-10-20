"""Build step 7b: greedy evaluation.  Written from the agreed interface
(dqn/evaluate.py docstring) before the implementation exists.

The oracle is a recording stub env: it logs every reset seed, every
observation it emitted and every action it received, so the tests can
recompute what a greedy policy must have done and what each return must be.

Skipped as a whole until dqn/evaluate.py exists.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

if not (Path(__file__).resolve().parents[1] / "dqn" / "evaluate.py").exists():
    pytest.skip("dqn/evaluate.py not written yet (build step 7b)", allow_module_level=True)

from dqn.evaluate import evaluate  # noqa: E402
from dqn.exploration import greedy_action  # noqa: E402
from dqn.network import MLP  # noqa: E402

OBS_DIM, N_ACT = 4, 3
SEEDS = [11, 22, 33, 44, 55]


class SeededStubEnv:
    """Episode length (2..9) and observations depend only on the reset seed.
    The reward depends on the action, so the return depends on the policy.
    Ends by termination or truncation (also from the seed).  Records
    everything into a shared log."""

    def __init__(self, seed, log: dict):
        self.log = log
        self.log["created"].append(seed)
        self.done = True

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.length = int(self.rng.integers(2, 10))
        self.ends_by_truncation = bool(self.rng.random() < 0.5)
        self.k, self.done = 0, False
        obs = self.rng.normal(size=OBS_DIM)
        self.log["resets"].append(seed)
        self.log["episodes"].append({"obs": [obs.copy()], "actions": [], "rewards": []})
        return obs, {}

    def step(self, action):
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset()")
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)) \
                or not 0 <= action < N_ACT:
            raise ValueError(f"bad action {action!r}")
        self.k += 1
        reward = [1.0, -0.5, 0.25][int(action)] + 0.1 * self.k
        end = self.k == self.length
        terminated = end and not self.ends_by_truncation
        truncated = end and self.ends_by_truncation
        self.done = end
        obs = self.rng.normal(size=OBS_DIM)
        ep = self.log["episodes"][-1]
        ep["actions"].append(int(action))
        ep["rewards"].append(reward)
        if not end:
            ep["obs"].append(obs.copy())
        return obs, reward, terminated, truncated, {}


def _log() -> dict:
    return {"created": [], "resets": [], "episodes": []}


def _env_fn(log):
    return lambda seed: SeededStubEnv(seed, log)


def _net(seed=0) -> MLP:
    return MLP((OBS_DIM, 16, N_ACT), np.random.default_rng(seed))


def test_one_return_per_seed_as_python_floats():
    returns = evaluate(_net(), _env_fn(_log()), SEEDS)
    assert isinstance(returns, list) and len(returns) == len(SEEDS)
    assert all(type(r) is float for r in returns)


def test_resets_use_the_given_seeds_in_order():
    log = _log()
    evaluate(_net(), _env_fn(log), SEEDS)
    assert log["resets"] == SEEDS


def test_every_action_is_greedy():
    net = _net()
    log = _log()
    evaluate(net, _env_fn(log), SEEDS)
    for i, ep in enumerate(log["episodes"]):
        assert len(ep["obs"]) == len(ep["actions"]), f"episode {i}"
        for t, (obs, action) in enumerate(zip(ep["obs"], ep["actions"])):
            q, _ = net.forward(obs[None])
            assert action == greedy_action(q[0]), f"episode {i} step {t}: not the greedy action"


def test_returns_are_undiscounted_sums():
    log = _log()
    returns = evaluate(_net(), _env_fn(log), SEEDS)
    for i, (ret, ep) in enumerate(zip(returns, log["episodes"])):
        assert ret == pytest.approx(sum(ep["rewards"]), abs=1e-12), f"episode {i}"


def test_episodes_run_to_their_end_and_no_further():
    # The stub raises on a step after the end, so overrunning fails the call.
    # This catches the other direction: stopping early.
    log = _log()
    evaluate(_net(), _env_fn(log), SEEDS)
    for seed, ep in zip(SEEDS, log["episodes"]):
        assert len(ep["actions"]) == _ending(seed)[0], f"seed {seed}"


def _ending(seed: int) -> tuple[int, bool]:
    """(length, ends_by_truncation) for a seed, as SeededStubEnv draws them."""
    rng = np.random.default_rng(seed)
    length = int(rng.integers(2, 10))
    return length, bool(rng.random() < 0.5)


def test_truncated_and_terminated_both_end_an_episode():
    # Stopping only on `terminated` would step a truncated episode again and
    # the stub would raise; this checks both kinds occur and run to length.
    seeds = list(range(40))
    endings = [_ending(s) for s in seeds]
    assert {trunc for _, trunc in endings} == {True, False}, "need both endings"
    log = _log()
    evaluate(_net(), _env_fn(log), seeds)
    assert [len(ep["actions"]) for ep in log["episodes"]] == [n for n, _ in endings]


def test_deterministic():
    net = _net()
    a = evaluate(net, _env_fn(_log()), SEEDS)
    b = evaluate(net, _env_fn(_log()), SEEDS)
    assert a == b


def test_different_networks_can_give_different_returns():
    a = evaluate(_net(0), _env_fn(_log()), SEEDS)
    b = evaluate(_net(1), _env_fn(_log()), SEEDS)
    assert a != b, "returns do not depend on the network: is it acting at all?"


def test_network_is_not_modified():
    net = _net()
    before = {k: v.copy() for k, v in net.params.items()}
    ids = {k: id(v) for k, v in net.params.items()}
    evaluate(net, _env_fn(_log()), SEEDS)
    for k, v in before.items():
        np.testing.assert_array_equal(net.params[k], v, err_msg=k)
        assert id(net.params[k]) == ids[k], k


def test_no_global_rng_used():
    np.random.seed(3)
    random.seed(3)
    np_before = np.random.get_state()[1].copy()
    py_before = random.getstate()
    evaluate(_net(), _env_fn(_log()), SEEDS)
    np.testing.assert_array_equal(np.random.get_state()[1], np_before)
    assert random.getstate() == py_before


def test_single_seed():
    returns = evaluate(_net(), _env_fn(_log()), [7])
    assert len(returns) == 1


def test_episodes_are_independent_of_order():
    # Each episode depends only on its own seed, so evaluating a seed alone
    # or in a list gives the same return.  Fails if state leaks between
    # episodes (e.g. an episode starting from the previous one's last obs).
    net = _net()
    together = evaluate(net, _env_fn(_log()), SEEDS)
    alone = [evaluate(net, _env_fn(_log()), [s])[0] for s in SEEDS]
    assert together == alone


@pytest.mark.parametrize("seeds", [[], [1.5], [None], [True]])
def test_rejects_bad_seeds(seeds):
    with pytest.raises(ValueError):
        evaluate(_net(), _env_fn(_log()), seeds)
