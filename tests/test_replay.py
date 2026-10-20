"""Build step 5: replay buffer.  Written from the agreed interface spec before
the implementation exists (dqn/README.md).

Interface under test (dqn/replay.py):

    class Batch(NamedTuple):
        obs: np.ndarray          # (B, obs_dim), buffer dtype
        actions: np.ndarray      # (B,) int64
        rewards: np.ndarray      # (B,) buffer dtype
        next_obs: np.ndarray     # (B, obs_dim), buffer dtype
        terminated: np.ndarray   # (B,) bool

    class ReplayBuffer:
        capacity: int
        pos: int                 # next slot to write

        def __init__(self, capacity, obs_dim, *, rng, dtype=np.float64)
        def add(self, obs, action, reward, next_obs, terminated) -> None
        def sample(self, batch_size) -> Batch
        def __len__(self) -> int  # number of filled slots

    Circular: after n adds, len == min(n, capacity) and pos == n % capacity;
    the oldest transition is overwritten first.  add copies its inputs;
    sample returns copies.  sample draws uniformly WITH replacement from the
    filled slots [0, len) only, using the rng given at construction (the
    "replay" stream, seeding.py) and nothing else.  add draws no random
    numbers.  Only `terminated` is stored: add takes no `truncated` or
    `done` (dqn/README.md, "Terminated vs truncated").

    Invalid arguments raise ValueError; a rejected add leaves the buffer
    exactly as it was.

The reference is CleanRL's buffer (derived from Stable-Baselines3), which
samples uniformly with replacement.  Its sampling stream differs from ours,
so batches are not bit-identical to the reference.  They are not meant to be.

The tests are black-box: they read the buffer only through len, pos and
sample, so the storage layout inside is yours to name.  Each transition i is
built so that every field encodes i (see _tr).  A sampled row can then be
checked for coherence, i.e. that all five fields came from one transition.

Skipped until dqn/replay.py defines ReplayBuffer.  An import error inside the
file fails loudly instead of skipping.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

if not (Path(__file__).resolve().parents[1] / "dqn" / "replay.py").exists():
    pytest.skip("dqn/replay.py not written yet (build step 5)", allow_module_level=True)

import dqn.replay as replay  # noqa: E402

if not hasattr(replay, "ReplayBuffer"):
    pytest.skip("dqn/replay.py has no ReplayBuffer yet (build step 5)", allow_module_level=True)

from dqn.replay import Batch, ReplayBuffer  # noqa: E402

OBS_DIM = 3


def _buf(capacity: int = 5, obs_dim: int = OBS_DIM, seed: int = 0,
         dtype=np.float64) -> ReplayBuffer:
    return ReplayBuffer(capacity, obs_dim, rng=np.random.default_rng(seed), dtype=dtype)


def _tr(i: int, obs_dim: int = OBS_DIM) -> dict:
    """Transition i.  v = i + 1, so no transition is all zeros and an
    unwritten (zero) slot can never pass for a real one."""
    v = float(i + 1)
    return dict(obs=np.full(obs_dim, v), action=i % 4, reward=10.0 * v,
                next_obs=np.full(obs_dim, v + 0.5), terminated=(i % 3 == 0))


def _fill(buf: ReplayBuffer, n: int, start: int = 0) -> None:
    for i in range(start, start + n):
        buf.add(**_tr(i))


def _ids(batch: Batch) -> np.ndarray:
    """Transition index of every sampled row."""
    return (batch.obs[:, 0] - 1).astype(np.int64)


def _assert_rows_coherent(batch: Batch) -> None:
    # Every field of a row must come from the same transition.  Fails if a
    # field is written to the wrong slot (pos advanced between fields), or
    # gathered with different indices.
    ids = _ids(batch)
    v = ids + 1.0
    np.testing.assert_array_equal(batch.obs, np.repeat(v[:, None], batch.obs.shape[1], axis=1))
    np.testing.assert_array_equal(batch.next_obs, batch.obs + 0.5)
    np.testing.assert_array_equal(batch.rewards, 10.0 * v)
    np.testing.assert_array_equal(batch.actions, ids % 4)
    np.testing.assert_array_equal(batch.terminated, ids % 3 == 0)


def _stored_ids(buf: ReplayBuffer, draws: int = 4000) -> set[int]:
    """Which transitions the buffer holds, read through sample().  With
    replacement from <= 10 slots, 4000 draws miss a slot with probability
    < 10 * 0.9**4000, i.e. never; the seed is fixed anyway."""
    batch = buf.sample(draws)
    _assert_rows_coherent(batch)
    return set(_ids(batch).tolist())


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------

def test_new_buffer_is_empty():
    buf = _buf(capacity=7)
    assert len(buf) == 0
    assert buf.pos == 0
    assert buf.capacity == 7


def test_rng_is_keyword_only():
    with pytest.raises(TypeError):
        ReplayBuffer(5, OBS_DIM, np.random.default_rng(0))


def test_batch_fields():
    assert Batch._fields == ("obs", "actions", "rewards", "next_obs", "terminated")


def test_add_takes_terminated_and_nothing_like_done_or_truncated():
    params = list(inspect.signature(ReplayBuffer.add).parameters)
    assert params == ["self", "obs", "action", "reward", "next_obs", "terminated"]


@pytest.mark.parametrize("kwargs", [
    dict(capacity=0, obs_dim=3), dict(capacity=-1, obs_dim=3),
    dict(capacity=5, obs_dim=0), dict(capacity=5, obs_dim=-2),
], ids=["cap0", "cap_neg", "dim0", "dim_neg"])
def test_rejects_bad_construction(kwargs):
    with pytest.raises(ValueError):
        ReplayBuffer(**kwargs, rng=np.random.default_rng(0))


# ----------------------------------------------------------------------
# Circular write: len, pos, which transitions survive
# ----------------------------------------------------------------------

def test_len_and_pos_after_every_add():
    buf = _buf(capacity=5)
    for n in range(1, 13):
        buf.add(**_tr(n - 1))
        assert len(buf) == min(n, 5), f"after {n} adds"
        assert buf.pos == n % 5, f"after {n} adds"


def test_partially_filled_holds_exactly_what_was_added():
    buf = _buf(capacity=10)
    _fill(buf, 4)
    assert _stored_ids(buf) == {0, 1, 2, 3}


def test_exactly_full_holds_everything():
    buf = _buf(capacity=5)
    _fill(buf, 5)
    assert len(buf) == 5 and buf.pos == 0
    assert _stored_ids(buf) == {0, 1, 2, 3, 4}


def test_overwrites_oldest_first():
    # 7 adds into 5 slots: 0 and 1 are gone, 2..6 remain.
    buf = _buf(capacity=5)
    _fill(buf, 7)
    assert _stored_ids(buf) == {2, 3, 4, 5, 6}


def test_many_wraps():
    buf = _buf(capacity=5)
    _fill(buf, 23)
    assert len(buf) == 5 and buf.pos == 3
    assert _stored_ids(buf) == {18, 19, 20, 21, 22}


def test_capacity_one_holds_only_the_latest():
    buf = _buf(capacity=1)
    for i in range(4):
        buf.add(**_tr(i))
        assert len(buf) == 1 and buf.pos == 0
        assert _stored_ids(buf, draws=20) == {i}


# ----------------------------------------------------------------------
# Sampling: bounds, shape, dtype, distribution
# ----------------------------------------------------------------------

def test_never_samples_unwritten_slots():
    # Bug #1: indices from [0, capacity) instead of [0, len).  Unwritten
    # slots are zeros; every real transition has obs >= 1.
    buf = _buf(capacity=100)
    _fill(buf, 3)
    batch = buf.sample(1000)
    assert np.all(batch.obs >= 1.0), "sampled an unwritten (zero) slot"
    assert set(_ids(batch).tolist()) == {0, 1, 2}


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_batch_shapes_and_dtypes(dtype):
    buf = _buf(capacity=8, obs_dim=4, dtype=dtype)
    for i in range(6):
        buf.add(**_tr(i, obs_dim=4))
    batch = buf.sample(16)
    assert isinstance(batch, Batch)
    assert batch.obs.shape == (16, 4) and batch.obs.dtype == dtype
    assert batch.next_obs.shape == (16, 4) and batch.next_obs.dtype == dtype
    assert batch.rewards.shape == (16,) and batch.rewards.dtype == dtype
    assert batch.actions.shape == (16,) and batch.actions.dtype == np.int64
    assert batch.terminated.shape == (16,) and batch.terminated.dtype == np.bool_
    _assert_rows_coherent(batch)


def test_samples_with_replacement():
    # Batch larger than the buffer: only possible with replacement.
    buf = _buf(capacity=5)
    _fill(buf, 3)
    batch = buf.sample(10)
    assert len(batch.obs) == 10
    _assert_rows_coherent(batch)


def test_sampling_is_uniform_over_filled_slots():
    # 40,000 draws over 4 slots: expected 10,000 each, sd ~87.  400 is ~4.6 sd.
    buf = _buf(capacity=10, seed=5)
    _fill(buf, 4)
    counts = np.bincount(_ids(buf.sample(40_000)), minlength=10)
    assert np.all(np.abs(counts[:4] - 10_000) < 400), counts
    assert np.all(counts[4:] == 0), counts


def test_sampling_is_uniform_after_wraparound():
    # Slots no longer in time order; every survivor must still be equally likely.
    buf = _buf(capacity=4, seed=6)
    _fill(buf, 6)                                   # holds 2, 3, 4, 5
    counts = np.bincount(_ids(buf.sample(40_000)), minlength=6)
    assert np.all(counts[:2] == 0), counts
    assert np.all(np.abs(counts[2:] - 10_000) < 400), counts


# ----------------------------------------------------------------------
# Copy, not reference
# ----------------------------------------------------------------------

def test_add_copies_obs_and_next_obs():
    # Warm-up #1 inside the buffer.  An env that reuses its observation array
    # would otherwise rewrite every stored transition.
    buf = _buf(capacity=5)
    t = _tr(0)
    buf.add(**t)
    t["obs"][:] = -7.0
    t["next_obs"][:] = -7.0
    batch = buf.sample(5)
    assert np.all(batch.obs == 1.0)
    assert np.all(batch.next_obs == 1.5)


def test_batch_is_a_copy_of_storage():
    # Warm-up #4: a sliced batch is a view of storage.  Writing into the batch
    # must not reach the buffer.
    buf = _buf(capacity=5)
    _fill(buf, 5)
    batch = buf.sample(50)
    for field in batch:
        field[...] = 0
    assert _stored_ids(buf) == {0, 1, 2, 3, 4}


def test_batch_survives_later_overwrites():
    # The other direction: storage overwritten after sampling must not change
    # a batch the training loop is still holding.
    buf = _buf(capacity=1)
    buf.add(**_tr(0))
    batch = buf.sample(3)
    buf.add(**_tr(1))
    np.testing.assert_array_equal(_ids(batch), [0, 0, 0])
    _assert_rows_coherent(batch)


def test_batch_fields_do_not_share_memory():
    buf = _buf(capacity=5)
    _fill(buf, 5)
    a, b = buf.sample(8), buf.sample(8)
    for x in (*a, *b):
        for y in (*a, *b):
            if x is not y:
                assert not np.shares_memory(x, y)


# ----------------------------------------------------------------------
# Randomness: only the constructor's rng, and only in sample()
# ----------------------------------------------------------------------

def test_same_seed_same_batches():
    a, b = _buf(capacity=6, seed=9), _buf(capacity=6, seed=9)
    _fill(a, 8)
    _fill(b, 8)
    for _ in range(5):
        x, y = a.sample(32), b.sample(32)
        for fx, fy in zip(x, y):
            np.testing.assert_array_equal(fx, fy)


def test_different_seed_different_batches():
    a, b = _buf(capacity=6, seed=1), _buf(capacity=6, seed=2)
    _fill(a, 6)
    _fill(b, 6)
    assert not np.array_equal(a.sample(64).obs, b.sample(64).obs)


def test_sample_uses_the_constructor_rng():
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(5, OBS_DIM, rng=rng)
    _fill(buf, 5)
    before = rng.bit_generator.state
    buf.sample(4)
    assert rng.bit_generator.state != before, "sample() did not draw from its rng"


def test_sample_does_not_touch_numpy_global_rng():
    # np.random.randint would make batches depend on unrelated code.
    buf = _buf(capacity=5)
    _fill(buf, 5)
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    buf.sample(4)
    np.testing.assert_array_equal(np.random.get_state()[1], before)


def test_add_draws_no_random_numbers():
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(5, OBS_DIM, rng=rng)
    before = rng.bit_generator.state
    _fill(buf, 7)
    assert rng.bit_generator.state == before


# ----------------------------------------------------------------------
# Accepted input types
# ----------------------------------------------------------------------

def test_accepts_numpy_scalars_and_lists():
    # What the training loop will actually pass: np.int64 from argmax or
    # rng.integers, np.float64 rewards, np.bool_ flags, list observations.
    buf = _buf(capacity=3)
    buf.add(obs=[1.0, 1.0, 1.0], action=np.int64(0), reward=np.float64(10.0),
            next_obs=np.array([1.5, 1.5, 1.5]), terminated=np.bool_(True))
    buf.add(obs=np.full(3, 2.0), action=1, reward=20, next_obs=np.full(3, 2.5),
            terminated=False)
    _assert_rows_coherent(buf.sample(20))


# ----------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------

@pytest.mark.parametrize("override", [
    {"obs": np.ones(OBS_DIM + 1)},
    {"obs": np.ones((1, OBS_DIM))},
    {"obs": np.float64(1.0)},
    {"next_obs": np.ones(OBS_DIM - 1)},
    {"obs": np.array([1.0, np.nan, 1.0])},
    {"next_obs": np.array([np.inf, 1.0, 1.0])},
    {"action": 1.5},
    {"action": -1},
    {"reward": float("nan")},
    {"reward": float("inf")},
    {"terminated": 2},
    {"terminated": 0.5},
    {"terminated": None},
], ids=["obs_long", "obs_2d", "obs_scalar", "next_obs_short", "obs_nan", "next_obs_inf",
        "action_float", "action_neg", "reward_nan", "reward_inf",
        "term_2", "term_float", "term_none"])
def test_add_rejects_bad_transition(override):
    buf = _buf(capacity=5)
    _fill(buf, 2)
    with pytest.raises(ValueError):
        buf.add(**{**_tr(9), **override})


def test_rejected_add_changes_nothing():
    buf = _buf(capacity=3)
    _fill(buf, 2)
    with pytest.raises(ValueError):
        buf.add(**{**_tr(9), "next_obs": np.array([np.nan, 1.0, 1.0])})
    assert len(buf) == 2 and buf.pos == 2
    assert _stored_ids(buf) == {0, 1}


def test_sample_from_empty_buffer_raises():
    with pytest.raises(ValueError):
        _buf().sample(1)


@pytest.mark.parametrize("batch_size", [0, -3])
def test_sample_rejects_bad_batch_size(batch_size):
    buf = _buf()
    _fill(buf, 3)
    with pytest.raises(ValueError):
        buf.sample(batch_size)
