"""Stage-2 baseline launcher (scripts/stage2_baseline.py, D038 firewall)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
spec = importlib.util.spec_from_file_location("stage2_baseline", REPO / "scripts" / "stage2_baseline.py")
s2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s2)

DIRTY = {"commit": "abc", "dirty": True, "dirty_paths": ["x.py"]}


def test_reserved_seeds_are_101_to_110():
    assert s2.RESERVED_SEEDS == tuple(range(101, 111))


def test_reserved_seeds_do_not_overlap_gate1():
    assert not set(s2.RESERVED_SEEDS) & set(range(1, 11))


def test_trains_on_the_training_mixture_with_gate1_defaults():
    cmd = s2.train_command(103, Path("r"))
    assert cmd[cmd.index("--seed") + 1] == "103"
    sets = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--set"]
    assert sets == ["env_id=combat:train"], "anything else would differ from Gate 1's config"


def test_refuses_dirty_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(s2, "collect_git", lambda: DIRTY)
    out = tmp_path / "b.json"
    with pytest.raises(SystemExit):
        s2.main(["--out", str(out), "--run-root", str(tmp_path / "runs")])
    assert not out.exists() and not (tmp_path / "runs").exists()


def test_refuses_overwrite(tmp_path):
    out = tmp_path / "b.json"
    out.write_text("{}")
    with pytest.raises(SystemExit):
        s2.main(["--out", str(out), "--run-root", str(tmp_path / "runs")])
    assert out.read_text() == "{}"


def test_summarize_uses_sample_sd():
    stats = s2.summarize({101: 0.2, 102: 0.4, 103: 0.9})
    assert stats["mean"] == pytest.approx(0.5)
    assert stats["sd"] == pytest.approx(np.std([0.2, 0.4, 0.9], ddof=1))
    assert stats["n_seeds"] == 3


def test_summarize_needs_two_seeds():
    with pytest.raises(ValueError):
        s2.summarize({101: 0.3})


def _fake_run(d: Path, commit="abc", dirty=False, status="completed", env_id="combat:train"):
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"status": status}))
    (d / "meta.json").write_text(json.dumps({"git_commit": commit, "git_dirty": dirty}))
    (d / "config.json").write_text(json.dumps({"env_id": env_id}))
    return d


@pytest.mark.parametrize("kw, n", [
    ({}, 0),
    ({"commit": "zzz"}, 1),
    ({"dirty": True}, 1),
    ({"env_id": "combat:mix"}, 1),
    ({"status": "crashed"}, 1),
])
def test_provenance_check(tmp_path, kw, n):
    assert len(s2.check_provenance(_fake_run(tmp_path / "seed101", **kw), "abc")) == n


def test_unfinished_run_fails_provenance(tmp_path):
    (tmp_path / "seed101").mkdir()
    assert s2.check_provenance(tmp_path / "seed101", "abc")


def test_tiny_run_trains_and_evaluates(tmp_path):
    """End to end on a few thousand steps: the run lands on combat:train and
    the evaluation produces a finite DV over exactly the held-out set."""
    from evaluation.heldout import load_anchors, load_held_out
    run_dir = tmp_path / "seed101"
    s2.train_one(101, run_dir, ("total_timesteps=1500", "learning_starts=500",
                                "buffer_size=1000"))
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["training_configs"] == [c for c in range(25) if c not in (4, 8, 12, 15, 21)]
    held = load_held_out(REPO / "docs" / "heldout_split.json")
    res = s2.evaluate_run(run_dir, held, load_anchors(REPO / "docs" / "return_anchors.json"),
                          n_eval=2)
    assert np.isfinite(res["primary_dv"])
    assert sorted(int(c) for c in res["normalized"]) == held
    assert res["primary_dv"] == pytest.approx(np.mean(list(res["normalized"].values())))
