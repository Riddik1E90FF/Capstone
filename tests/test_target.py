"""Build step 4: target network and hard sync.  Written from the agreed
interface spec before the implementation exists (dqn/README.md).

Interface under test (dqn/target.py):

    def make_target(online: MLP) -> MLP
        Independent copy of `online`: same layer_sizes and dtype, bit-identical
        params in the same key order, no array shared with `online`.  Takes no
        rng: the target needs no random stream (seeding.py), and an rng
        argument is the only way it could consume `network_init`.

    def hard_sync(target: MLP, online: MLP) -> None
        Copy online's values into target's EXISTING arrays, in place.  After:
        bit-equal; target.params and every target.params[k] are the same
        objects they were before; no array shared with online; online
        unchanged.  Mismatched keys, shapes or dtype, or target being the same
        object as online, raise ValueError and leave target untouched.

    def sync_due(global_step, *, learning_starts, target_network_frequency) -> bool
        True iff global_step > learning_starts and
        global_step % target_network_frequency == 0.  global_step is 0-based
        and counts environment steps, as CleanRL's `for global_step in
        range(total_timesteps)` (dqn.py lines 163, 194, 215, commit fe8d8a0).

Ordering within one step (update first, then sync, as CleanRL does) belongs
to the training loop, step 7a, and is checked by L2, not here.

Skipped as a whole until dqn/target.py exists.  Once the file exists, an
import error inside it fails loudly instead of skipping.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

if not (Path(__file__).resolve().parents[1] / "dqn" / "target.py").exists():
    pytest.skip("dqn/target.py not written yet (build step 4)", allow_module_level=True)

from dqn.adam import Adam  # noqa: E402
from dqn.config import DQNConfig  # noqa: E402
from dqn.network import MLP  # noqa: E402
from dqn.target import hard_sync, make_target, sync_due  # noqa: E402
from dqn.td import td_loss_and_grad, td_target  # noqa: E402

CFG = DQNConfig()
HYPER = dict(lr=CFG.learning_rate, beta1=CFG.adam_beta1, beta2=CFG.adam_beta2, eps=CFG.adam_eps)
SMALL = (3, 4, 2)
CARTPOLE = (4, 120, 84, 2)


def _net(sizes=SMALL, seed: int = 0, dtype=np.float64) -> MLP:
    return MLP(sizes, np.random.default_rng(seed), dtype=dtype)


def _snapshot(net: MLP) -> dict[str, np.ndarray]:
    return {k: a.copy() for k, a in net.params.items()}


def _assert_params_equal(net: MLP, expected: dict[str, np.ndarray], label: str = "") -> None:
    assert list(net.params) == list(expected), f"{label}: keys or key order differ"
    for k, a in expected.items():
        assert net.params[k].dtype == a.dtype, f"{label}: {k} dtype"
        np.testing.assert_array_equal(net.params[k], a, err_msg=f"{label}: {k}")


def _assert_no_shared_memory(a: MLP, b: MLP) -> None:
    assert a.params is not b.params, "the two networks share one params dict"
    for ka, x in a.params.items():
        for kb, y in b.params.items():
            assert not np.shares_memory(x, y), f"{ka} and {kb} share memory"


def _adam_steps(online: MLP, opt: Adam, rng: np.random.Generator, n: int) -> None:
    """n real DQN-shaped updates: forward, TD loss, backward, Adam.

    Adam writes online's arrays in place.  That is what makes a shallow copy
    (new dict, same arrays) visible: the target moves with every update.
    """
    obs_dim, n_actions = online.layer_sizes[0], online.layer_sizes[-1]
    for _ in range(n):
        x = rng.normal(size=(8, obs_dim))
        q, cache = online.forward(x)
        actions = rng.integers(0, n_actions, size=8)
        y = td_target(rng.normal(size=8), rng.normal(size=(8, n_actions)),
                      rng.random(8) < 0.2, CFG.gamma)
        _, grad_q = td_loss_and_grad(q, actions, y)
        opt.step(online.backward(cache, grad_q))


# ----------------------------------------------------------------------
# make_target
# ----------------------------------------------------------------------

@pytest.mark.parametrize("sizes", [SMALL, CARTPOLE, (5, 2)], ids=["small", "cartpole", "linear"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_make_target_is_an_equal_copy(sizes, dtype):
    online = _net(sizes, dtype=dtype)
    target = make_target(online)
    assert isinstance(target, MLP)
    assert target is not online
    assert target.layer_sizes == online.layer_sizes
    assert target.n_layers == online.n_layers
    assert target.dtype == online.dtype
    _assert_params_equal(target, _snapshot(online), "make_target")


def test_make_target_shares_no_memory_with_online():
    online = _net(CARTPOLE)
    target = make_target(online)
    _assert_no_shared_memory(target, online)


def test_make_target_gives_identical_q_values():
    online = _net(CARTPOLE)
    target = make_target(online)
    x = np.random.default_rng(1).normal(size=(32, 4))
    np.testing.assert_array_equal(target.forward(x)[0], online.forward(x)[0])


def test_make_target_does_not_change_online():
    online = _net()
    ids = {k: id(a) for k, a in online.params.items()}
    before = _snapshot(online)
    make_target(online)
    _assert_params_equal(online, before, "online after make_target")
    assert {k: id(a) for k, a in online.params.items()} == ids


def test_make_target_takes_only_the_online_network():
    # No rng parameter: building the target must not draw from network_init.
    assert list(inspect.signature(make_target).parameters) == ["online"]


def test_target_from_make_target_is_frozen_while_online_trains():
    # The shallow-copy trap: dict(online.params) passes every equality test
    # above, then follows the online network through every Adam step.
    online = _net(CARTPOLE)
    target = make_target(online)
    at_copy = _snapshot(online)
    _adam_steps(online, Adam(online.params, **HYPER), np.random.default_rng(2), n=5)
    assert any(not np.array_equal(online.params[k], at_copy[k]) for k in at_copy), \
        "online did not move, so this test proves nothing"
    _assert_params_equal(target, at_copy, "target after online trained")


def test_writing_target_does_not_change_online():
    online = _net()
    target = make_target(online)
    before = _snapshot(online)
    for a in target.params.values():
        a += 1.0
    _assert_params_equal(online, before, "online after target was written")


# ----------------------------------------------------------------------
# hard_sync
# ----------------------------------------------------------------------

def _diverged_pair(sizes=CARTPOLE, dtype=np.float64) -> tuple[MLP, MLP]:
    """target and online with the same architecture but different values."""
    return _net(sizes, seed=10, dtype=dtype), _net(sizes, seed=20, dtype=dtype)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_hard_sync_makes_target_equal_to_online(dtype):
    target, online = _diverged_pair(dtype=dtype)
    assert not np.array_equal(target.params["W0"], online.params["W0"])
    result = hard_sync(target, online)
    assert result is None
    _assert_params_equal(target, _snapshot(online), "target after sync")
    x = np.random.default_rng(1).normal(size=(32, 4))
    np.testing.assert_array_equal(target.forward(x)[0], online.forward(x)[0])


def test_hard_sync_copies_online_into_target_not_the_reverse():
    target, online = _diverged_pair()
    online_before = _snapshot(online)
    hard_sync(target, online)
    _assert_params_equal(online, online_before, "online after sync")
    _assert_params_equal(target, online_before, "target after sync")


def test_hard_sync_is_in_place():
    # Same dict, same array objects.  Rebinding target.params[k] to
    # online.params[k] would also pass the equality test, and alias.
    target, online = _diverged_pair()
    dict_id = id(target.params)
    ids = {k: id(a) for k, a in target.params.items()}
    hard_sync(target, online)
    assert id(target.params) == dict_id, "target.params was replaced"
    assert {k: id(a) for k, a in target.params.items()} == ids, "target arrays were rebound"


def test_hard_sync_leaves_online_arrays_alone():
    target, online = _diverged_pair()
    dict_id = id(online.params)
    ids = {k: id(a) for k, a in online.params.items()}
    hard_sync(target, online)
    assert id(online.params) == dict_id
    assert {k: id(a) for k, a in online.params.items()} == ids


def test_hard_sync_shares_no_memory_with_online():
    target, online = _diverged_pair()
    hard_sync(target, online)
    _assert_no_shared_memory(target, online)


def test_target_unchanged_between_syncs_and_follows_at_each_sync():
    # The full cycle as the training loop will drive it: train, sync, train,
    # sync.  Between syncs the target holds exactly the last synced values.
    online = _net(CARTPOLE)
    target = make_target(online)
    opt = Adam(online.params, **HYPER)
    rng = np.random.default_rng(3)
    for cycle in range(3):
        synced = _snapshot(online)
        _adam_steps(online, opt, rng, n=4)
        _assert_params_equal(target, synced, f"cycle {cycle}, between syncs")
        hard_sync(target, online)
        _assert_params_equal(target, _snapshot(online), f"cycle {cycle}, after sync")
    assert opt.t == 12


def test_hard_sync_twice_is_the_same_as_once():
    target, online = _diverged_pair()
    hard_sync(target, online)
    once = _snapshot(target)
    hard_sync(target, online)
    _assert_params_equal(target, once, "second sync")


@pytest.mark.parametrize("target_sizes, online_sizes", [
    ((3, 4, 2), (3, 5, 2)),          # hidden width differs
    ((3, 4, 2), (3, 4, 3)),          # n_actions differs
    ((3, 4, 4, 2), (3, 4, 2)),       # depth differs; W0 and b0 still match
    ((3, 4, 2), (3, 4, 4, 2)),
], ids=["width", "n_actions", "deeper_target", "deeper_online"])
def test_hard_sync_rejects_architecture_mismatch(target_sizes, online_sizes):
    # The depth cases have matching W0 and b0: a loop that copies as it
    # checks would write those before failing.  Validate first.
    target, online = _net(target_sizes, seed=10), _net(online_sizes, seed=20)
    before = _snapshot(target)
    with pytest.raises(ValueError):
        hard_sync(target, online)
    _assert_params_equal(target, before, "target after rejected sync")


def test_hard_sync_rejects_dtype_mismatch():
    # np.copyto and arr[...] = would silently cast float64 into float32.
    target = _net(seed=10, dtype=np.float32)
    online = _net(seed=20, dtype=np.float64)
    before = _snapshot(target)
    with pytest.raises(ValueError):
        hard_sync(target, online)
    _assert_params_equal(target, before, "target after rejected sync")


def test_hard_sync_rejects_key_mismatch():
    target, online = _net(seed=10), _net(seed=20)
    online.params["extra"] = np.zeros(2)
    before = _snapshot(target)
    with pytest.raises(ValueError):
        hard_sync(target, online)
    _assert_params_equal(target, before, "target after rejected sync")


def test_hard_sync_rejects_aliased_arrays():
    # A target built by shallow copy holds online's arrays.  Syncing it is a
    # no-op, so the only place the aliasing can surface is here.
    online = _net(seed=20)
    target = _net(seed=10)
    target.params = dict(online.params)
    with pytest.raises(ValueError):
        hard_sync(target, online)


def test_hard_sync_rejects_target_is_online():
    # `target = online` in the loop.  Copying a network onto itself is a
    # no-op, so without this check the loop runs with no target network.
    net = _net()
    with pytest.raises(ValueError):
        hard_sync(net, net)


# ----------------------------------------------------------------------
# sync_due
# ----------------------------------------------------------------------

def _due(step: int, learning_starts: int = CFG.learning_starts,
         freq: int = CFG.target_network_frequency) -> bool:
    return sync_due(step, learning_starts=learning_starts, target_network_frequency=freq)


@pytest.mark.parametrize("step, expected", [
    (0, False),          # 0 % 500 == 0, but learning has not started
    (500, False),
    (9_500, False),
    (10_000, False),     # == learning_starts: CleanRL's test is strictly >
    (10_001, False),
    (10_010, False),     # the first update step, not a sync step
    (10_499, False),
    (10_500, True),      # first sync
    (10_501, False),
    (11_000, True),
    (499_500, True),     # last sync of a 500k run
    (499_999, False),    # last step of a 500k run
], ids=lambda v: str(v))
def test_sync_due_cleanrl_defaults(step, expected):
    assert CFG.learning_starts == 10_000 and CFG.target_network_frequency == 500
    result = _due(step)
    assert isinstance(result, bool)
    assert result is expected


def test_sync_due_count_over_a_full_run():
    # Multiples of 500 in (10_000, 499_999]:  999 - 20 = 979.  Counting
    # updates instead of env steps gives ~98; >= instead of > gives 980.
    total = CFG.total_timesteps
    n = sum(_due(step) for step in range(total))
    f, ls = CFG.target_network_frequency, CFG.learning_starts
    assert n == (total - 1) // f - ls // f
    assert n == 979


def test_sync_due_counts_from_zero_not_from_learning_starts():
    # learning_starts not a multiple of the frequency.  Counting the period
    # from learning_starts would sync at 10_750; CleanRL syncs at 10_500.
    assert _due(10_500, learning_starts=10_250) is True
    assert _due(10_750, learning_starts=10_250) is False
    assert _due(10_000, learning_starts=10_250) is False


def test_sync_due_with_no_warmup():
    assert _due(0, learning_starts=0) is False      # 0 > 0 is false
    assert _due(1, learning_starts=0, freq=1) is True
    assert _due(500, learning_starts=0) is True


def test_sync_due_frequency_one_syncs_every_step_after_warmup():
    assert [_due(s, learning_starts=3, freq=1) for s in range(6)] == \
        [False, False, False, False, True, True]


def test_sync_due_schedule_is_keyword_only():
    # Two adjacent ints are easy to swap: (10_000, 500) vs (500, 10_000).
    with pytest.raises(TypeError):
        sync_due(10_500, 10_000, 500)


@pytest.mark.parametrize("kwargs", [
    dict(global_step=-1, learning_starts=0, target_network_frequency=500),
    dict(global_step=0, learning_starts=-1, target_network_frequency=500),
    dict(global_step=0, learning_starts=0, target_network_frequency=0),   # % 0
    dict(global_step=0, learning_starts=0, target_network_frequency=-500),
], ids=["neg_step", "neg_learning_starts", "zero_freq", "neg_freq"])
def test_sync_due_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        sync_due(**kwargs)
