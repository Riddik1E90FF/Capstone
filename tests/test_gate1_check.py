"""G1-c permutation null (scripts/gate1_check.py: spread_permutation_null).

The specification is fixed by D034.  These tests check it from three sides:

* exact behaviour: the draw pattern is pinned, so seed 0 always gives the
  same null array, the one the Gate 1 decision is recorded against;
* structure: 25 groups of the input's own sizes, never a 26th or an empty one;
* statistics: on data whose configuration labels carry no information the
  test rejects about 5% of the time, and on strongly ordered data it always
  rejects.  A null built by permuting means, or one that never actually
  shuffles, fails the first of these.

Each mutant this file is meant to catch is named in its test.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gate1_check", REPO / "scripts" / "gate1_check.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)

N_CONFIGS = len(g.CONFIG_IDS)


def _noise(rng, sizes, loc=30.0, scale=10.0):
    """{config: values}, every configuration drawn from the same distribution."""
    return {c: list(rng.normal(loc, scale, n)) for c, n in enumerate(sizes)}


def _observed(outcomes):
    """The observed spread, computed the way main() computes it."""
    means = np.array([np.mean(v) for v in outcomes.values()])
    return float(means.max() - means.min())


def _reference(outcomes, n_permutations, seed):
    """The D034 procedure written out directly, as an oracle for the draw pattern."""
    rng = np.random.default_rng(seed)
    groups = [np.asarray(v, dtype=float) for v in outcomes.values()]
    pooled = np.concatenate(groups)
    out = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(pooled)
        start, means = 0, []
        for grp in groups:
            means.append(shuffled[start:start + len(grp)].mean())
            start += len(grp)
        out.append(max(means) - min(means))
    return np.array(out)


# -- output contract ------------------------------------------------------

def test_returns_float_array_of_requested_length():
    data = _noise(np.random.default_rng(1), [20] * N_CONFIGS)
    null = g.spread_permutation_null(data, n_permutations=37, seed=0)
    assert isinstance(null, np.ndarray)
    assert null.shape == (37,)
    assert null.dtype == np.float64
    assert np.all(np.isfinite(null))


def test_defaults_are_d034_values():
    assert g.N_PERMUTATIONS == 10_000
    assert g.PERMUTATION_SEED == 0
    data = _noise(np.random.default_rng(1), [4] * N_CONFIGS)
    default = g.spread_permutation_null(data)
    assert default.shape == (10_000,)
    np.testing.assert_array_equal(
        default, g.spread_permutation_null(data, n_permutations=10_000, seed=0))


def test_input_is_not_modified():
    data = _noise(np.random.default_rng(2), [10] * N_CONFIGS)
    before = {c: list(v) for c, v in data.items()}
    g.spread_permutation_null(data, n_permutations=20, seed=0)
    assert data == before


# -- reproducibility ------------------------------------------------------

def test_same_seed_same_null():
    data = _noise(np.random.default_rng(3), [15] * N_CONFIGS)
    a = g.spread_permutation_null(data, n_permutations=50, seed=0)
    b = g.spread_permutation_null(data, n_permutations=50, seed=0)
    np.testing.assert_array_equal(a, b)


def test_different_seed_different_null():
    data = _noise(np.random.default_rng(3), [15] * N_CONFIGS)
    a = g.spread_permutation_null(data, n_permutations=50, seed=0)
    b = g.spread_permutation_null(data, n_permutations=50, seed=1)
    assert not np.array_equal(a, b)


def test_matches_reference_draw_pattern():
    """Pins the exact stream: one rng.permutation of the pool per permutation,
    groups in dict order.  Unequal sizes, so a misplaced cut point shows."""
    sizes = [3, 7, 1, 12, 5] + [8] * (N_CONFIGS - 5)
    data = _noise(np.random.default_rng(4), sizes)
    np.testing.assert_allclose(
        g.spread_permutation_null(data, n_permutations=200, seed=0),
        _reference(data, 200, 0), rtol=0, atol=1e-12)


def test_prefix_stable():
    """The first n permutations do not depend on how many are requested."""
    data = _noise(np.random.default_rng(5), [10] * N_CONFIGS)
    short = g.spread_permutation_null(data, n_permutations=30, seed=0)
    long = g.spread_permutation_null(data, n_permutations=300, seed=0)
    np.testing.assert_array_equal(short, long[:30])


# -- structure ------------------------------------------------------------

@pytest.mark.filterwarnings("error")
def test_single_configuration_spread_is_zero():
    """One group: max mean == min mean on every permutation.
    Catches cumsum - 1 (an extra one-element group -> non-zero spread) and
    cutting at every cumsum entry.  The latter adds an empty group whose mean
    is nan with a RuntimeWarning; builtin max/min skip a trailing nan, so the
    output is unchanged and only the warning, made an error here, shows it."""
    data = {0: list(np.random.default_rng(6).normal(30, 10, 50))}
    null = g.spread_permutation_null(data, n_permutations=20, seed=0)
    np.testing.assert_array_equal(null, np.zeros(20))


def test_constant_values_give_zero_spread():
    data = {c: [17.5] * 9 for c in range(N_CONFIGS)}
    null = g.spread_permutation_null(data, n_permutations=20, seed=0)
    np.testing.assert_array_equal(null, np.zeros(20))


def test_two_groups_exhaustive_values():
    """Sizes (1, 3) over values {0, 0, 0, 4}: the one-element group holds the 4
    with probability 1/4 (spread 4), otherwise spread |0 - 4/3| = 4/3."""
    data = {0: [4.0], 1: [0.0, 0.0, 0.0]}
    null = g.spread_permutation_null(data, n_permutations=4_000, seed=0)
    assert set(np.round(null, 9)) <= {round(4.0, 9), round(4 / 3, 9)}
    assert np.mean(np.isclose(null, 4.0)) == pytest.approx(0.25, abs=0.03)


def test_spreads_bounded_by_pool_range():
    data = _noise(np.random.default_rng(7), [6] * N_CONFIGS)
    pool = np.concatenate(list(data.values()))
    null = g.spread_permutation_null(data, n_permutations=200, seed=0)
    assert np.all(null >= 0)
    assert np.all(null <= pool.max() - pool.min())


def test_null_varies_between_permutations():
    """Catches splitting the unshuffled pool: every 'permutation' would then
    equal the observed spread exactly."""
    data = _noise(np.random.default_rng(8), [20] * N_CONFIGS)
    null = g.spread_permutation_null(data, n_permutations=100, seed=0)
    assert np.std(null) > 0
    assert not np.all(null == _observed(data))


# -- statistical behaviour ------------------------------------------------

def test_false_positive_rate_near_five_percent():
    """No real ordering: G1-c (observed > 95th percentile) should pass in
    about 5% of datasets.  Permuting means instead of episodes gives a null
    that is a single value equal to the observed spread, so the rate drops
    to 0; not shuffling at all does the same."""
    rng = np.random.default_rng(9)
    n_datasets, rejections = 200, 0
    for _ in range(n_datasets):
        data = _noise(rng, [20] * N_CONFIGS)
        null = g.spread_permutation_null(data, n_permutations=200, seed=0)
        rejections += _observed(data) > np.percentile(null, 95)
    rate = rejections / n_datasets
    assert 0.01 <= rate <= 0.11, f"false-positive rate {rate:.3f}"


def test_strong_ordering_is_detected():
    """Means rising 2 per configuration against SD 10, 200 episodes each."""
    rng = np.random.default_rng(10)
    data = {c: list(rng.normal(20 + 2 * c, 10, 200)) for c in range(N_CONFIGS)}
    null = g.spread_permutation_null(data, n_permutations=500, seed=0)
    assert _observed(data) > null.max()


def test_realistic_shape_runs():
    """The real call: 25 configurations x (200 episodes x 10 seeds)."""
    data = _noise(np.random.default_rng(11), [2_000] * N_CONFIGS)
    null = g.spread_permutation_null(data, n_permutations=20, seed=0)
    assert null.shape == (20,) and np.all(np.isfinite(null))
