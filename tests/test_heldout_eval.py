"""Primary-DV evaluator (evaluation/heldout.py) and the anchor script (D039)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from combat_env import N_ACTIONS, make_env
from evaluation import heldout as h
from evaluation.gate1 import evaluate_policy, random_policy, run_episode

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from axis_sweep import policy as scripted_policy  # noqa: E402

CFGS = (0, 12, 24)


def _anchors(random_r, scripted_r):
    return {"random": {"returns": {str(c): v for c, v in random_r.items()}},
            "scripted": {"returns": {str(c): v for c, v in scripted_r.items()}}}


# -- episodes -------------------------------------------------------------

@pytest.mark.parametrize("cfg", CFGS)
def test_return_is_the_sum_of_step_rewards(cfg):
    seed = 1_000_003
    env = make_env(cfg, seed=seed)
    obs, _ = env.reset(seed=seed)
    total, done = 0.0, False
    while not done:
        obs, r, term, trunc, _ = env.step(scripted_policy(obs))
        total += float(r)
        done = term or trunc
    ret, _ = h.run_episode_return(cfg, scripted_policy, seed)
    assert ret == total


@pytest.mark.parametrize("cfg", CFGS)
def test_same_episode_as_gate1(cfg):
    for seed in (1_000_000, 1_000_001):
        _, stats = h.run_episode_return(cfg, scripted_policy, seed)
        assert stats == run_episode(cfg, scripted_policy, seed)


def test_random_stream_matches_gate1_evaluation():
    """The random floor shares one Generator across configurations, so the
    two evaluators see the same episodes only if they consume it identically."""
    a = h.evaluate_returns(random_policy(np.random.default_rng(0), N_ACTIONS),
                           config_ids=CFGS, n_eval=5)
    b = evaluate_policy(random_policy(np.random.default_rng(0), N_ACTIONS),
                        config_ids=CFGS, n_eval=5)
    assert a["damage_taken"] == b["damage_taken"]


def test_evaluate_returns_shape_and_means():
    res = h.evaluate_returns(scripted_policy, config_ids=CFGS, n_eval=4)
    assert set(res["returns"]) == set(CFGS)
    for c in CFGS:
        assert len(res["episode_returns"][c]) == 4
        assert res["returns"][c] == pytest.approx(np.mean(res["episode_returns"][c]))


# -- normalization --------------------------------------------------------

def test_anchor_values_map_to_zero_and_one():
    random_r = {c: float(c) for c in range(25)}
    scripted_r = {c: float(c) + 10.0 for c in range(25)}
    anchors = _anchors(random_r, scripted_r)
    assert h.normalized_returns(random_r, anchors, [3, 7]) == {3: 0.0, 7: 0.0}
    assert h.normalized_returns(scripted_r, anchors, [3, 7]) == {3: 1.0, 7: 1.0}


def test_hand_computed_value_and_mean():
    random_r = {c: 0.0 for c in range(25)}
    scripted_r = {c: 20.0 for c in range(25)}
    scripted_r[5] = 40.0
    anchors = _anchors(random_r, scripted_r)
    returns = {5: 10.0, 6: 30.0}
    assert h.normalized_returns(returns, anchors, [5, 6]) == {5: 0.25, 6: 1.5}
    assert h.primary_dv(returns, anchors, [5, 6]) == pytest.approx(0.875)


def test_offset_cancels_in_contrasts():
    rng = np.random.default_rng(0)
    random_r = {c: float(rng.normal(0, 30)) for c in range(25)}
    gap = {c: float(rng.uniform(20, 60)) for c in range(25)}
    held = [4, 8, 12, 15, 21]
    a = {c: float(rng.normal(50, 10)) for c in held}
    b = {c: float(rng.normal(50, 10)) for c in held}
    base = _anchors(random_r, {c: random_r[c] + gap[c] for c in range(25)})
    shifted_random = {c: random_r[c] + 100.0 * (c + 1) for c in range(25)}
    shifted = _anchors(shifted_random, {c: shifted_random[c] + gap[c] for c in range(25)})
    diff_base = h.primary_dv(a, base, held) - h.primary_dv(b, base, held)
    diff_shift = h.primary_dv(a, shifted, held) - h.primary_dv(b, shifted, held)
    assert diff_base == pytest.approx(diff_shift)


def test_fallback_replaces_small_denominators_with_median():
    random_r = {c: 0.0 for c in range(25)}
    scripted_r = {c: 10.0 for c in range(25)}
    scripted_r[3] = 0.5          # 5% of the median gap: replaced
    scripted_r[4] = 1.0          # exactly 10%: kept (strictly below triggers)
    scripted_r[6] = -2.0         # scripted worse than random: replaced
    denom, replaced = h.denominators(random_r, scripted_r)
    assert replaced == {3, 6}
    assert denom[3] == denom[6] == 10.0
    assert denom[4] == 1.0


def test_nonpositive_median_gap_raises():
    with pytest.raises(ValueError):
        h.denominators({c: 5.0 for c in range(25)}, {c: 5.0 for c in range(25)})


def test_missing_held_out_return_raises():
    anchors = _anchors({c: 0.0 for c in range(25)}, {c: 1.0 for c in range(25)})
    with pytest.raises(ValueError):
        h.normalized_returns({4: 1.0}, anchors, [4, 8])


def test_mismatched_anchor_coverage_raises():
    with pytest.raises(ValueError):
        h.denominators({0: 0.0, 1: 0.0}, {0: 1.0})


def test_load_held_out_reads_the_committed_split():
    held = h.load_held_out(REPO / "docs" / "heldout_split.json")
    assert held == [4, 8, 12, 15, 21]


# -- anchor script --------------------------------------------------------

spec = importlib.util.spec_from_file_location(
    "measure_return_anchors", REPO / "scripts" / "measure_return_anchors.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_script_refuses_dirty_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "collect_git",
                        lambda: {"commit": "abc", "dirty": True, "dirty_paths": ["x"]})
    out = tmp_path / "anchors.json"
    with pytest.raises(SystemExit):
        m.main(["--out", str(out)])
    assert not out.exists()


def test_script_refuses_overwrite(tmp_path):
    out = tmp_path / "anchors.json"
    out.write_text("{}")
    with pytest.raises(SystemExit):
        m.main(["--out", str(out)])
    assert out.read_text() == "{}"


def test_floor_check_reports_mismatches():
    floors = {"floors": {"random": {"damage_taken": {str(c): 1.0 for c in range(25)}}}}
    damage = {c: 1.0 for c in range(25)}
    assert m.check_against_floors("random", damage, floors) == []
    damage[7] = 1.5
    assert len(m.check_against_floors("random", damage, floors)) == 1
