"""Stage-2 power simulation (scripts/stage2_power.py, D041).

The `power_at` tests are written against the contract before the function
exists.  Until it is written they xfail; after, they run for real.  None of
them uses the registered sizes, so the suite stays fast.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("stage2_power", REPO / "scripts" / "stage2_power.py")
sp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sp)

CLEAN = {"commit": "abc123", "dirty": False, "dirty_paths": []}

# Continuous values, so the permutation distribution has no ties.
NORMAL = np.random.default_rng(42).normal(0.0, 1.0, size=500)


def power(n, values=NORMAL, delta=0.0, alpha=0.05, n_sims=400, n_perm=199, seed=0):
    try:
        return sp.power_at(n, values, delta, alpha, n_sims, n_perm,
                           np.random.default_rng(seed))
    except NotImplementedError:
        pytest.xfail("power_at not written yet")


# -- power_at: contract ----------------------------------------------------

def test_returns_a_fraction_of_n_sims():
    p = power(10, delta=0.5, n_sims=250)
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0
    assert p * 250 == pytest.approx(round(p * 250))


def test_same_seed_same_result():
    assert power(15, delta=0.4, seed=3) == power(15, delta=0.4, seed=3)


def test_different_seeds_differ():
    assert power(15, delta=0.4, seed=3) != power(15, delta=0.4, seed=4)


def test_values_not_modified():
    v = NORMAL.copy()
    power(10, values=v, delta=0.3)
    assert np.array_equal(v, NORMAL)


# -- power_at: statistics --------------------------------------------------

def test_null_rejection_rate_is_alpha():
    """delta = 0: the test is exact for continuous values, so the rate is
    alpha up to Monte Carlo error (SE about 0.007 at 1,000 sims)."""
    rate = power(20, delta=0.0, alpha=0.05, n_sims=1000, n_perm=199)
    assert 0.025 <= rate <= 0.075, rate


def test_overwhelming_effect_is_always_detected():
    assert power(10, delta=20.0) == 1.0


def test_one_sided_negative_effect_is_never_detected():
    """A two-sided test (or B - A reversed) would reject here."""
    assert power(20, delta=-1.5) == 0.0


def test_reversed_sign_detects_negative_not_positive():
    assert power(20, delta=1.5) > 0.95


def test_p_value_has_the_plus_one():
    """With 19 permutations the smallest attainable p is 1/20 = 0.05, so no
    experiment can reach alpha = 0.04.  Without the +1, p can be 0."""
    assert power(20, delta=20.0, alpha=0.04, n_perm=19) == 0.0
    assert power(20, delta=20.0, alpha=0.05, n_perm=19) == 1.0


def test_power_rises_with_n():
    lo = power(8, delta=0.5, n_sims=400)
    hi = power(60, delta=0.5, n_sims=400)
    assert hi - lo > 0.3, (lo, hi)


def test_statistic_is_the_iqm_not_the_mean():
    """Values with 10% huge outliers.  The IQM ignores them and the shift is
    easy to see; a mean is drowned by them."""
    rng = np.random.default_rng(7)
    core = rng.normal(0.0, 0.1, size=900)
    outliers = rng.choice([-1000.0, 1000.0], size=100)
    v = np.concatenate([core, outliers])
    assert power(40, values=v, delta=0.15, n_sims=200, n_perm=199) > 0.9


def test_trim_is_25_percent_not_less():
    """12% of values are huge outliers on one side.  A 25% trim on each side
    almost always removes them, even in permuted groups, so the shift is seen
    (calibrated: ~0.99).  A 10% trim keeps some in most samples, and the
    shift is lost in their resampling noise (~0.13)."""
    rng = np.random.default_rng(11)
    v = np.concatenate([rng.normal(0.0, 0.1, size=880), np.full(120, 500.0)])
    assert power(60, values=v, delta=0.15, n_sims=200, n_perm=199) > 0.8


def test_runs_on_the_real_baseline_values():
    p = power(40, values=sp.load_values(), delta=sp.DELTA, alpha=sp.ALPHA,
              n_sims=100, n_perm=239)
    assert 0.0 <= p <= 1.0


# -- harness ---------------------------------------------------------------

def test_registered_constants():
    assert sp.DELTA == 0.07
    assert sp.ALPHA == pytest.approx(0.05 / 6)
    assert (sp.TARGET_POWER, sp.N_SIMS, sp.N_PERM, sp.SIM_SEED) == (0.80, 2000, 2000, 0)
    assert (sp.N_FLOOR, sp.N_STEP, sp.SEED_BASE) == (40, 10, 1000)


def test_load_values_reads_the_ten_baseline_dvs_in_seed_order():
    v = sp.load_values()
    rec = json.loads((REPO / "docs" / "stage2_baseline.json").read_text())
    assert len(v) == 10
    assert v.tolist() == [rec["per_seed"][str(s)]["primary_dv"] for s in range(101, 111)]
    assert v.mean() == pytest.approx(rec["summary"]["mean"])


def test_find_n_returns_first_candidate_reaching_target():
    n, table = sp.find_n(lambda n, rng: n / 100)
    assert n == 80
    assert list(table) == [40, 50, 60, 70, 80]


def test_find_n_respects_the_floor():
    n, table = sp.find_n(lambda n, rng: 1.0)
    assert n == 40 and list(table) == [40]


def test_find_n_target_is_inclusive():
    n, _ = sp.find_n(lambda n, rng: 0.80 if n == 50 else 0.5)
    assert n == 50


def test_find_n_gives_every_candidate_a_fresh_rng():
    firsts = []

    def fn(n, rng):
        firsts.append(rng.random())
        return 0.0 if n < 60 else 1.0
    sp.find_n(fn)
    assert len(firsts) == 3 and len(set(firsts)) == 1


def test_find_n_raises_when_unreachable():
    with pytest.raises(RuntimeError):
        sp.find_n(lambda n, rng: 0.1, n_max=70)


def test_find_n_rejects_a_floor_off_the_grid():
    with pytest.raises(ValueError):
        sp.find_n(lambda n, rng: 1.0, floor=45)


def test_main_refuses_dirty_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "collect_git",
                        lambda: {"commit": "abc", "dirty": True, "dirty_paths": ["x.py"]})
    out = tmp_path / "power.json"
    with pytest.raises(SystemExit):
        sp.main(["--out", str(out)])
    assert not out.exists()


def test_main_refuses_overwrite(tmp_path):
    out = tmp_path / "power.json"
    out.write_text("{}")
    with pytest.raises(SystemExit):
        sp.main(["--out", str(out)])
    assert out.read_text() == "{}"


def test_main_writes_the_record(tmp_path, monkeypatch):
    calls = []

    def fake(n, values, delta, alpha, n_sims, n_perm, rng):
        calls.append((n, delta, alpha, n_sims, n_perm, len(values)))
        return 0.5 if n < 60 else 0.85
    monkeypatch.setattr(sp, "collect_git", lambda: CLEAN)
    monkeypatch.setattr(sp, "power_at", fake)
    out = tmp_path / "power.json"
    assert sp.main(["--out", str(out)]) == 0
    rec = json.loads(out.read_text())
    assert rec["n_per_condition"] == 60
    assert rec["seeds"] == [1001, 1060]
    assert rec["power"] == {"40": 0.5, "50": 0.5, "60": 0.85}
    assert rec["git_commit"] == "abc123"
    assert calls[0] == (40, 0.07, 0.05 / 6, 2000, 2000, 10)
