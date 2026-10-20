"""Build step 6: epsilon schedule and epsilon-greedy.  Written from the agreed
interface spec before the implementation exists (dqn/README.md).

Interface under test (dqn/exploration.py):

    def linear_epsilon(t, *, start_e, end_e, duration) -> float
        CleanRL linear_schedule (dqn.py, commit fe8d8a0):
            slope = (end_e - start_e) / duration
            return max(slope * t + start_e, end_e)
        t is the 0-based global ENVIRONMENT step, counted from step 0 (not
        from learning_starts, not in updates).  Pure: no state, no RNG.

    def greedy_action(q_values) -> int
        Index of the max of a 1-D q_values.  Ties go to the lowest index,
        as np.argmax and torch.argmax do.  Takes no rng and draws nothing.

    def epsilon_greedy(q_values, epsilon, rng) -> int
        u = rng.random(); if u < epsilon: rng.integers(n_actions), else
        greedy_action(q_values).  n_actions = len(q_values).  Exactly one
        uniform draw per call, plus one integer draw iff the random branch
        is taken.  Strict `<`, so epsilon 0 is always greedy and epsilon 1
        always random.  The random branch samples from ALL actions,
        including the greedy one.

Both action functions return a Python int in [0, n_actions).  Invalid inputs
raise ValueError before any RNG draw.

Not checked here (they belong to the training loop and eval runner, steps 7a
and 7b): that the caller passes global_step and not the update count, and that
evaluation never calls epsilon_greedy with the training "exploration" rng.

Skipped as a whole until dqn/exploration.py exists.  Once the file exists, an
import error inside it fails loudly instead of skipping.
"""

from __future__ import annotations

import inspect
import random
from pathlib import Path

import numpy as np
import pytest

if not (Path(__file__).resolve().parents[1] / "dqn" / "exploration.py").exists():
    pytest.skip("dqn/exploration.py not written yet (build step 6)", allow_module_level=True)

from dqn.config import DQNConfig  # noqa: E402
from dqn.exploration import epsilon_greedy, greedy_action, linear_epsilon  # noqa: E402

CFG = DQNConfig()
SCHED = dict(start_e=CFG.start_e, end_e=CFG.end_e, duration=CFG.exploration_steps)
# Unique argmax at index 2, so "greedy" is unambiguous.
Q4 = np.array([0.1, -0.3, 0.7, 0.2])


def _reference(q: np.ndarray, epsilons, seed: int):
    """What epsilon_greedy must do, step by step, on a fresh rng(seed).
    Returns (actions, branch_was_random, final rng state)."""
    ref = np.random.default_rng(seed)
    actions, random_branch = [], []
    for eps in epsilons:
        u = ref.random()
        if u < eps:
            actions.append(int(ref.integers(len(q))))
            random_branch.append(True)
        else:
            actions.append(int(np.argmax(q)))
            random_branch.append(False)
    return actions, random_branch, ref.bit_generator.state


# ----------------------------------------------------------------------
# Signatures
# ----------------------------------------------------------------------

def test_linear_epsilon_schedule_params_are_keyword_only():
    # start_e and end_e are both floats in [0, 1]; positional order is an
    # easy silent swap (CleanRL's own order is start, end, duration, t).
    params = inspect.signature(linear_epsilon).parameters
    assert list(params) == ["t", "start_e", "end_e", "duration"]
    for name in ("start_e", "end_e", "duration"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, name


def test_greedy_action_takes_no_rng():
    assert list(inspect.signature(greedy_action).parameters) == ["q_values"]


def test_epsilon_greedy_signature():
    assert list(inspect.signature(epsilon_greedy).parameters) == ["q_values", "epsilon", "rng"]


# ----------------------------------------------------------------------
# linear_epsilon: endpoints, clamp, shape
# ----------------------------------------------------------------------

def test_epsilon_at_step_zero_is_exactly_start_e():
    assert linear_epsilon(0, **SCHED) == CFG.start_e


@pytest.mark.parametrize("t, expected", [
    (10_000, 0.962),     # learning_starts: the anneal began at step 0, not here
    (125_000, 0.525),    # halfway through the anneal
    (249_999, 0.0500038),
])
def test_cartpole_default_values(t, expected):
    assert linear_epsilon(t, **SCHED) == pytest.approx(expected, abs=1e-12)


def test_epsilon_at_duration_is_end_e_up_to_rounding():
    # slope * duration + start_e is 0.050000000000000044 with the defaults,
    # so exact equality here would be the wrong test.
    assert linear_epsilon(int(CFG.exploration_steps), **SCHED) == pytest.approx(CFG.end_e, abs=1e-12)


@pytest.mark.parametrize("t", [250_001, 300_000, 499_999, 2_500_000, 10**12])
def test_after_duration_epsilon_is_exactly_end_e(t):
    # Without the clamp epsilon goes negative, and `u < negative` is never
    # true: exploration silently switches off instead of crashing.
    eps = linear_epsilon(t, **SCHED)
    assert eps == CFG.end_e
    assert eps >= 0.0


def test_clamp_is_a_floor_not_a_ceiling():
    # `min` instead of `max` gives end_e from step 0.
    assert linear_epsilon(1, **SCHED) > CFG.end_e
    assert linear_epsilon(100_000, **SCHED) > CFG.end_e


def test_schedule_is_linear_then_flat():
    kw = dict(start_e=1.0, end_e=0.2, duration=100)
    values = np.array([linear_epsilon(t, **kw) for t in range(300)])
    np.testing.assert_allclose(values[:101], 1.0 - 0.008 * np.arange(101), atol=1e-12)
    assert np.all(values[101:] == 0.2)
    assert np.all(np.diff(values) <= 0.0), "epsilon increased somewhere"
    assert values.min() >= 0.2 and values.max() <= 1.0


def test_flat_schedule_when_start_equals_end():
    kw = dict(start_e=0.3, end_e=0.3, duration=50)
    assert all(linear_epsilon(t, **kw) == 0.3 for t in (0, 1, 25, 50, 51, 10_000))


def test_fractional_duration():
    # cfg.exploration_steps is a float (exploration_fraction * total_timesteps),
    # and need not be a whole number.
    kw = dict(start_e=1.0, end_e=0.0, duration=2.5)
    assert linear_epsilon(1, **kw) == pytest.approx(0.6, abs=1e-12)
    assert linear_epsilon(2, **kw) == pytest.approx(0.2, abs=1e-12)
    assert linear_epsilon(3, **kw) == 0.0


def test_end_e_zero_never_goes_negative():
    kw = dict(start_e=1.0, end_e=0.0, duration=10)
    assert all(linear_epsilon(t, **kw) >= 0.0 for t in range(100))


def test_numpy_integer_step_accepted():
    assert linear_epsilon(np.int64(10_000), **SCHED) == pytest.approx(0.962, abs=1e-12)


def test_returns_python_float():
    assert type(linear_epsilon(0, **SCHED)) is float
    assert type(linear_epsilon(10**9, **SCHED)) is float


def test_schedule_is_pure():
    # A hidden step counter would make the value depend on call history.
    late = linear_epsilon(200_000, **SCHED)
    assert linear_epsilon(0, **SCHED) == CFG.start_e
    assert linear_epsilon(200_000, **SCHED) == late
    assert linear_epsilon(10_000, **SCHED) == linear_epsilon(10_000, **SCHED)


# ----------------------------------------------------------------------
# greedy_action
# ----------------------------------------------------------------------

@pytest.mark.parametrize("q, expected", [
    ([0.1, -0.3, 0.7, 0.2], 2),
    ([5.0], 0),
    ([-5.0, -1.0, -3.0], 1),          # all negative: max is not "largest magnitude"
    ([1e300, -1e300], 0),
])
def test_greedy_action_picks_argmax(q, expected):
    assert greedy_action(np.array(q)) == expected


@pytest.mark.parametrize("q, expected", [
    ([1.0, 3.0, 3.0, 0.0], 1),
    ([2.0, 2.0], 0),
    ([0.0, 0.0, 0.0, 0.0], 0),
    ([-1.0, 4.0, 2.0, 4.0, 4.0], 1),
])
def test_greedy_ties_go_to_lowest_index(q, expected):
    # Matches torch.argmax in CleanRL.  Deterministic and draws no rng.
    assert greedy_action(np.array(q)) == expected


def test_greedy_action_returns_python_int():
    a = greedy_action(Q4)
    assert type(a) is int


def test_greedy_action_accepts_float32_and_list_like():
    assert greedy_action(Q4.astype(np.float32)) == 2
    assert greedy_action([0.0, 1.0]) == 1


def test_greedy_action_does_not_modify_q():
    q = Q4.copy()
    greedy_action(q)
    np.testing.assert_array_equal(q, Q4)


# ----------------------------------------------------------------------
# epsilon_greedy: endpoints and draw accounting
# ----------------------------------------------------------------------

def test_epsilon_zero_is_always_greedy():
    rng = np.random.default_rng(0)
    assert all(epsilon_greedy(Q4, 0.0, rng) == 2 for _ in range(5_000))


def test_epsilon_one_is_always_random_branch():
    # Every call draws an integer: compare against the reference draw sequence.
    rng = np.random.default_rng(0)
    actions = [epsilon_greedy(Q4, 1.0, rng) for _ in range(2_000)]
    expected, branch, state = _reference(Q4, [1.0] * 2_000, seed=0)
    assert all(branch)
    assert actions == expected
    assert rng.bit_generator.state == state


def test_greedy_branch_draws_exactly_one_uniform():
    rng = np.random.default_rng(1)
    for _ in range(100):
        epsilon_greedy(Q4, 0.0, rng)
    ref = np.random.default_rng(1)
    ref.random(100)
    assert rng.bit_generator.state == ref.bit_generator.state


@pytest.mark.parametrize("eps", [0.05, 0.3, 0.5, 0.9])
def test_draw_sequence_matches_reference(eps):
    # One rng.random() per call, then rng.integers(n_actions) iff u < eps.
    # Any extra or missing draw desynchronises everything after it.
    rng = np.random.default_rng(7)
    actions = [epsilon_greedy(Q4, eps, rng) for _ in range(3_000)]
    expected, branch, state = _reference(Q4, [eps] * 3_000, seed=7)
    assert 0 < sum(branch) < len(branch), "test needs both branches exercised"
    assert actions == expected
    assert rng.bit_generator.state == state


def test_draw_sequence_matches_reference_along_the_schedule():
    # As the training loop will use it: epsilon recomputed from t every step.
    kw = dict(start_e=1.0, end_e=0.05, duration=1_000)
    epsilons = [linear_epsilon(t, **kw) for t in range(2_000)]
    rng = np.random.default_rng(3)
    actions = [epsilon_greedy(Q4, e, rng) for e in epsilons]
    expected, _, state = _reference(Q4, epsilons, seed=3)
    assert actions == expected
    assert rng.bit_generator.state == state


class _ZeroUniform(np.random.Generator):
    """A real Generator whose random() always returns 0.0, the one value at
    which `u < eps` and `u <= eps` disagree for eps = 0.  A real rng hits it
    with probability 2**-53, so only a stub can test the comparison."""

    def random(self, *args, **kwargs):
        return 0.0


def test_epsilon_zero_is_greedy_even_when_u_is_zero():
    # Strict `<`: u in [0, 1), so `u < 0.0` is never true.  With `<=`,
    # eps = 0 would still take the random branch on u == 0.0.
    rng = _ZeroUniform(np.random.PCG64(0))
    q = np.array([0.0, 0.0, 0.0, 1.0])   # greedy is the last index
    assert all(epsilon_greedy(q, 0.0, rng) == 3 for _ in range(20))


def test_greedy_branch_agrees_with_greedy_action_on_ties():
    q = np.array([1.0, 3.0, 3.0, 0.0])
    rng = np.random.default_rng(0)
    assert all(epsilon_greedy(q, 0.0, rng) == greedy_action(q) == 1 for _ in range(100))


def test_returns_python_int_on_both_branches():
    rng = np.random.default_rng(0)
    for eps in (0.0, 1.0):
        a = epsilon_greedy(Q4, eps, rng)
        assert type(a) is int, f"eps={eps}: got {type(a).__name__}"
        assert 0 <= a < len(Q4)


def test_single_action_always_zero():
    rng = np.random.default_rng(0)
    assert all(epsilon_greedy(np.array([3.0]), e, rng) == 0 for e in (0.0, 0.5, 1.0))


def test_epsilon_greedy_does_not_modify_q():
    q = Q4.copy()
    rng = np.random.default_rng(0)
    for eps in (0.0, 1.0, 0.5):
        epsilon_greedy(q, eps, rng)
    np.testing.assert_array_equal(q, Q4)


# ----------------------------------------------------------------------
# epsilon_greedy: empirical rates
#
# The random branch samples from ALL n actions, so with probability
# eps / n it returns the greedy action anyway.  Expected action frequencies:
#     greedy:      1 - eps + eps / n
#     each other:  eps / n
# so the observable "action != argmax" rate is eps * (n - 1) / n, not eps.
# ----------------------------------------------------------------------

@pytest.mark.parametrize("eps", [0.1, 0.5, 1.0])
def test_empirical_action_frequencies(eps):
    n_calls, n = 40_000, len(Q4)
    rng = np.random.default_rng(11)
    counts = np.bincount([epsilon_greedy(Q4, eps, rng) for _ in range(n_calls)], minlength=n)
    p = np.full(n, eps / n)
    p[2] += 1.0 - eps
    for a in range(n):
        sd = np.sqrt(n_calls * p[a] * (1.0 - p[a]))
        assert abs(counts[a] - n_calls * p[a]) <= 5 * sd + 1, (
            f"eps={eps} action {a}: {counts[a]} vs expected {n_calls * p[a]:.0f}")


def test_non_greedy_rate_is_eps_times_n_minus_1_over_n():
    n_calls, eps, n = 40_000, 0.5, 2
    q = np.array([0.0, 1.0])
    rng = np.random.default_rng(5)
    non_greedy = sum(epsilon_greedy(q, eps, rng) != 1 for _ in range(n_calls))
    p = eps * (n - 1) / n   # 0.25, not 0.5
    sd = np.sqrt(n_calls * p * (1 - p))
    assert abs(non_greedy - n_calls * p) <= 5 * sd


# ----------------------------------------------------------------------
# RNG isolation
# ----------------------------------------------------------------------

def test_no_global_rng_used():
    # Python's `random` and np.random's global state are what CleanRL uses.
    # Here everything must come from the rng argument.
    np.random.seed(123)
    random.seed(123)
    np_before = np.random.get_state()[1].copy()
    py_before = random.getstate()
    rng = np.random.default_rng(0)
    for eps in (0.0, 0.5, 1.0):
        for _ in range(200):
            epsilon_greedy(Q4, eps, rng)
    greedy_action(Q4)
    linear_epsilon(123, **SCHED)
    np.testing.assert_array_equal(np.random.get_state()[1], np_before)
    assert random.getstate() == py_before


def test_same_seed_same_actions():
    a = [epsilon_greedy(Q4, 0.4, r) for r in [np.random.default_rng(9)] for _ in range(500)]
    b = [epsilon_greedy(Q4, 0.4, r) for r in [np.random.default_rng(9)] for _ in range(500)]
    assert a == b


# ----------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------

@pytest.mark.parametrize("t", [-1, -10**9, 1.5, 10.0, True, "3", None])
def test_linear_epsilon_rejects_bad_step(t):
    # A float t usually means something other than global_step was passed.
    with pytest.raises(ValueError):
        linear_epsilon(t, **SCHED)


@pytest.mark.parametrize("override", [
    dict(duration=0), dict(duration=-5.0), dict(duration=float("nan")),
    dict(start_e=1.5), dict(start_e=-0.1),
    dict(end_e=-0.01), dict(end_e=1.1),
    dict(start_e=0.1, end_e=0.5),             # anneal upward: max() would floor at 0.5
    dict(start_e=float("nan")), dict(end_e=float("nan")),
])
def test_linear_epsilon_rejects_bad_schedule(override):
    with pytest.raises(ValueError):
        linear_epsilon(0, **{**SCHED, **override})


@pytest.mark.parametrize("q", [
    np.zeros((1, 4)),                          # passed q, not q[0]
    np.zeros((4, 1)),
    np.float64(1.0),
    np.array([]),
    np.array([0.0, np.nan, 1.0]),              # argmax would return the nan's index
    np.array([0.0, np.inf]),
    np.array([-np.inf, 0.0]),
])
def test_greedy_action_rejects_bad_q(q):
    with pytest.raises(ValueError):
        greedy_action(q)


@pytest.mark.parametrize("q", [np.zeros((1, 4)), np.array([]), np.array([0.0, np.nan])])
def test_epsilon_greedy_rejects_bad_q(q):
    with pytest.raises(ValueError):
        epsilon_greedy(q, 0.5, np.random.default_rng(0))


@pytest.mark.parametrize("eps", [-0.01, 1.01, float("nan"), float("inf")])
def test_epsilon_greedy_rejects_bad_epsilon(eps):
    with pytest.raises(ValueError):
        epsilon_greedy(Q4, eps, np.random.default_rng(0))


@pytest.mark.parametrize("rng", [None, 0, np.random.RandomState(0), random.Random(0)])
def test_epsilon_greedy_rejects_non_generator_rng(rng):
    # RandomState is the legacy global-style API; an int seed would make a
    # fresh generator every call and repeat the same "random" action.
    with pytest.raises(ValueError):
        epsilon_greedy(Q4, 0.5, rng)


@pytest.mark.parametrize("q, eps", [
    (np.zeros((1, 4)), 0.5),
    (np.array([0.0, np.nan]), 0.5),
    (Q4, 1.5),
    (Q4, float("nan")),
])
def test_rejected_call_draws_nothing(q, eps):
    rng = np.random.default_rng(0)
    before = rng.bit_generator.state
    with pytest.raises(ValueError):
        epsilon_greedy(q, eps, rng)
    assert rng.bit_generator.state == before
