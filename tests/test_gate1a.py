"""Gate 1a calibration and check (scripts/gate1a.py).  Written from the
interface and D029's confirmed metric definitions before the implementation
exists.

Runs are built in memory as lists of episode records.  Global steps are set
directly, so steps-to-threshold is exact by construction:

    _reaching(S, final)   100 episodes of return 500 ending at global_step S
                          (the moving average first hits 500 there), then 100
                          episodes of return `final`.
    _never(level)         200 episodes of return `level` (< 475: never reaches).

Skipped until scripts/gate1a.py has function bodies.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gate1a", REPO / "scripts" / "gate1a.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)

if g.moving_average([1.0, 2.0], window=2) is None:
    pytest.skip("scripts/gate1a.py not implemented yet", allow_module_level=True)


def _run_from(returns, steps):
    return [{"episode": i, "global_step": int(s), "return": float(r)}
            for i, (r, s) in enumerate(zip(returns, steps))]


def _reaching(step, final=500.0):
    first = [step - 99 * 10 + 10 * i for i in range(100)]          # last one is `step`
    later = [step + 10 * (i + 1) for i in range(100)]
    return _run_from([500.0] * 100 + [final] * 100, first + later)


def _never(level=100.0):
    return _run_from([level] * 200, [100 * (i + 1) for i in range(200)])


# -- moving_average -------------------------------------------------------

def test_moving_average_complete_windows_only():
    np.testing.assert_allclose(g.moving_average([1, 2, 3, 4, 5], window=3), [2.0, 3.0, 4.0])


def test_moving_average_short_input_is_empty():
    assert len(g.moving_average([1.0] * 99)) == 0


def test_moving_average_exactly_one_window():
    np.testing.assert_allclose(g.moving_average([2.0] * 100), [2.0])


# -- steps_to_threshold ---------------------------------------------------

def test_steps_first_episode_where_average_reaches_t():
    assert g.steps_to_threshold(_reaching(123_450), 475.0) == (123_450, True)


def test_steps_brute_force_on_a_ramp():
    returns = list(np.linspace(0, 500, 400))
    steps = list(np.cumsum(np.maximum(returns, 1)).astype(int))
    run = _run_from(returns, steps)
    expected = next(steps[i] for i in range(99, 400)
                    if np.mean(returns[i - 99:i + 1]) >= 300.0)
    assert g.steps_to_threshold(run, 300.0) == (expected, True)


def test_steps_average_equal_to_t_counts_as_reached():
    run = _run_from([475.0] * 100, [10 * (i + 1) for i in range(100)])
    assert g.steps_to_threshold(run, 475.0) == (1000, True)


def test_never_reaching_is_censored_at_total():
    assert g.steps_to_threshold(_never(), 475.0) == (500_000, False)
    assert g.steps_to_threshold(_never(), 475.0, total_timesteps=1234) == (1234, False)


def test_no_partial_windows():
    # 99 perfect episodes: no complete window yet, so not reached.
    run = _run_from([500.0] * 99, [500 * (i + 1) for i in range(99)])
    assert g.steps_to_threshold(run, 475.0) == (500_000, False)


# -- final_return ---------------------------------------------------------

def test_final_return_is_mean_of_last_100():
    returns = [0.0] * 50 + list(range(100))
    run = _run_from(returns, range(1, 151))
    assert g.final_return(run) == pytest.approx(49.5)


# -- iqm ------------------------------------------------------------------

def test_iqm_ten_values_drops_two_each_end():
    assert g.iqm(list(range(1, 11))) == pytest.approx(5.5)                    # mean of 3..8
    assert g.iqm([0, 0, 1, 2, 3, 4, 5, 6, 1000, 1000]) == pytest.approx(3.5)  # outliers ignored


def test_iqm_is_order_independent():
    assert g.iqm([5, 1, 9, 3, 7, 2, 8, 4, 10, 6]) == pytest.approx(5.5)


def test_iqm_matches_scipy_trim_mean():
    stats = pytest.importorskip("scipy.stats")
    x = np.random.default_rng(1).normal(size=10) * 50 + 300
    assert g.iqm(x) == pytest.approx(stats.trim_mean(x, 0.25))


def test_iqm_returns_python_float():
    assert type(g.iqm(list(range(10)))) is float


# -- choose_threshold -----------------------------------------------------

@pytest.mark.parametrize("max_mas, expected", [
    ([500] * 10, 475.0),                                  # never above the default
    ([480] * 8 + [100] * 2, 475.0),                       # exactly 8 reach 475
    ([500] * 7 + [460] * 3, 450.0),                       # 7 reach 475, all 10 reach 450
    ([480] * 7 + [440] * 3, 425.0),                       # 7 reach 450, all 10 reach 425
    ([450] * 8 + [0] * 2, 450.0),                         # exact multiple counts as reached
    ([474.9] * 10, 450.0),
])
def test_choose_threshold(max_mas, expected):
    assert g.choose_threshold(max_mas) == expected


# -- bootstrap_pairs ------------------------------------------------------

def test_bootstrap_pairs_spec():
    a, b = g.bootstrap_pairs(10)
    rng = np.random.default_rng(0)
    np.testing.assert_array_equal(a, rng.integers(0, 10, size=(10_000, 10)))
    np.testing.assert_array_equal(b, rng.integers(0, 10, size=(10_000, 10)))


def test_bootstrap_pairs_deterministic_and_seeded():
    a1, b1 = g.bootstrap_pairs(10, n_resamples=50, seed=3)
    a2, b2 = g.bootstrap_pairs(10, n_resamples=50, seed=3)
    a3, _ = g.bootstrap_pairs(10, n_resamples=50, seed=4)
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(b1, b2)
    assert not np.array_equal(a1, a3)


# -- calibrate ------------------------------------------------------------

def test_calibrate_identical_reference_seeds():
    cal = g.calibrate([_reaching(100_000)] * 10)
    assert cal["T"] == 475.0
    assert cal["k"] == 8
    assert cal["q"] == pytest.approx(1.0)
    assert cal["F"] == 2.0                  # floor applies
    assert cal["delta"] == pytest.approx(0.0)


REF_STEPS = [60_000, 70_000, 80_000, 90_000, 100_000, 110_000, 130_000, 150_000, 200_000, 230_000]
REF_FINALS = [480.0, 490.0, 495.0, 500.0, 470.0, 485.0, 499.0, 460.0, 500.0, 450.0]


def _reference():
    return [_reaching(s, f) for s, f in zip(REF_STEPS, REF_FINALS)]


def test_calibrate_matches_d029_rules_exactly():
    # Oracle: D029's rules written out directly on the known per-seed values.
    cal = g.calibrate(_reference())
    rng = np.random.default_rng(0)
    a = rng.integers(0, 10, size=(10_000, 10))
    b = rng.integers(0, 10, size=(10_000, 10))
    steps, finals = np.array(REF_STEPS, float), np.array(REF_FINALS)

    def iqm_rows(x):
        s = np.sort(x, axis=1)
        return s[:, 2:8].mean(axis=1)

    q = np.percentile(np.median(steps[a], axis=1) / np.median(steps[b], axis=1), 95)
    delta = np.percentile(iqm_rows(finals[a]) - iqm_rows(finals[b]), 95)
    assert cal["T"] == 475.0 and cal["k"] == 8
    assert cal["q"] == pytest.approx(q, rel=1e-12)
    assert cal["F"] == pytest.approx(max(2.0, q), rel=1e-12)
    assert cal["delta"] == pytest.approx(delta, rel=1e-12)
    assert cal["delta"] > 0


def test_calibrate_f_above_floor_when_reference_varies_a_lot():
    steps = [20_000, 25_000, 30_000, 40_000, 60_000, 200_000, 300_000, 350_000, 400_000, 450_000]
    cal = g.calibrate([_reaching(s) for s in steps])
    assert cal["q"] > 2.0
    assert cal["F"] == pytest.approx(cal["q"])


def test_calibrate_censored_seeds_count_as_total_timesteps():
    # 8 reach (so T stays 475), 2 censored.  The censored ones must enter the
    # ratios as 500,000, not be dropped.
    runs = [_reaching(50_000)] * 8 + [_never()] * 2
    cal = g.calibrate(runs)
    rng = np.random.default_rng(0)
    a = rng.integers(0, 10, size=(10_000, 10))
    b = rng.integers(0, 10, size=(10_000, 10))
    steps = np.array([50_000.0] * 8 + [500_000.0] * 2)
    q = np.percentile(np.median(steps[a], axis=1) / np.median(steps[b], axis=1), 95)
    assert cal["k"] == 6
    assert cal["q"] == pytest.approx(q, rel=1e-12)


def test_calibrate_lowers_t_when_reference_falls_short():
    runs = [_reaching(80_000)] * 7 + [_never(level=460.0)] * 3
    cal = g.calibrate(runs)
    assert cal["T"] == 450.0
    assert cal["k"] == 8                    # all 10 reach 450


def test_calibrate_is_deterministic():
    assert g.calibrate(_reference()) == g.calibrate(_reference())


# -- check_gate -----------------------------------------------------------

CAL = {"T": 475.0, "k": 8, "F": 2.0, "delta": 10.0}


def test_identical_to_reference_passes():
    res = g.check_gate(_reference(), _reference(), CAL)
    assert res["L3"] and res["L4_steps"] and res["L4_final"] and res["passed"]


def test_never_learning_fails_everything_it_should():
    res = g.check_gate([_never()] * 10, _reference(), CAL)
    assert not res["L3"]
    assert not res["L4_steps"]              # all censored: median undefined
    assert not res["passed"]


def test_l3_needs_k_seeds():
    ok = [_reaching(100_000)] * 8 + [_never()] * 2
    short = [_reaching(100_000)] * 7 + [_never()] * 3
    assert g.check_gate(ok, _reference(), CAL)["L3"]
    assert not g.check_gate(short, _reference(), CAL)["L3"]


def test_l4_steps_factor():
    ref_median = np.median(REF_STEPS)                       # 105,000
    within = [_reaching(int(s * 1.9)) for s in REF_STEPS]   # median 1.9x
    beyond = [_reaching(int(s * 2.1)) for s in REF_STEPS]   # median 2.1x
    assert np.median([s * 1.9 for s in REF_STEPS]) <= 2.0 * ref_median
    assert g.check_gate(within, _reference(), CAL)["L4_steps"]
    assert not g.check_gate(beyond, _reference(), CAL)["L4_steps"]


def test_l4_steps_exactly_half_censored_is_not_an_automatic_fail():
    # "More than half" censored fails outright; exactly half does not.
    runs = [_reaching(60_000)] * 5 + [_never()] * 5
    res = g.check_gate(runs, _reference(), {**CAL, "F": 10.0})
    assert res["L4_steps"]                  # median 280,000 <= 10 x 105,000
    runs6 = [_reaching(60_000)] * 4 + [_never()] * 6
    assert not g.check_gate(runs6, _reference(), {**CAL, "F": 100.0})["L4_steps"]


def test_l4_final_non_inferiority_margin():
    ref_iqm = g.iqm(REF_FINALS)
    just_ok = [_reaching(s, f - 9.0) for s, f in zip(REF_STEPS, REF_FINALS)]
    too_low = [_reaching(s, f - 11.0) for s, f in zip(REF_STEPS, REF_FINALS)]
    assert g.iqm([f - 9.0 for f in REF_FINALS]) >= ref_iqm - 10.0
    assert g.check_gate(just_ok, _reference(), CAL)["L4_final"]
    assert not g.check_gate(too_low, _reference(), CAL)["L4_final"]


def test_beating_the_reference_is_not_a_defect():
    better = [_reaching(s // 2, 500.0) for s in REF_STEPS]
    assert g.check_gate(better, _reference(), CAL)["passed"]


def test_passed_requires_all_three():
    too_low = [_reaching(s, f - 50.0) for s, f in zip(REF_STEPS, REF_FINALS)]
    res = g.check_gate(too_low, _reference(), CAL)
    assert res["L3"] and res["L4_steps"] and not res["L4_final"]
    assert not res["passed"]
