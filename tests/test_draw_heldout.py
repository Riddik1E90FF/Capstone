"""Held-out split draw (scripts/draw_heldout.py, D038).

No test here calls draw() with HELDOUT_SEED.  Running the suite must not
reveal the held-out set before the seed is committed; the real draw happens
once, through main(), from a clean tree.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("draw_heldout", REPO / "scripts" / "draw_heldout.py")
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)

CLEAN = {"commit": "abc123", "dirty": False, "dirty_paths": []}


def _is_transversal(cfgs):
    telegraph = sorted(c // 5 for c in cfgs)
    gap = sorted(c % 5 for c in cfgs)
    return telegraph == list(range(5)) and gap == list(range(5))


def test_there_are_120_transversals():
    ts = d.transversals()
    assert len(ts) == 120
    assert len(set(ts)) == 120
    assert all(_is_transversal(t) for t in ts)


def test_enumeration_order_is_lexicographic():
    ts = d.transversals()
    assert ts[0] == (0, 6, 12, 18, 24)          # identity permutation
    assert ts[-1] == (4, 8, 12, 16, 20)         # reversed: the anti-diagonal
    gap_perms = [tuple(c % 5 for c in t) for t in ts]
    assert gap_perms == sorted(gap_perms)


def test_78_eligible_without_corners():
    el = d.eligible()
    assert len(el) == 78                        # 120 - 24 - 24 + 6
    assert all(0 not in t and 24 not in t for t in el)
    assert all(_is_transversal(t) for t in el)


def test_draw_is_deterministic_and_eligible():
    for seed in (1, 2, 3):
        a, b = d.draw(seed), d.draw(seed)
        assert a == b
        index, held_out = a
        assert 0 <= index < 78
        assert held_out == d.eligible()[index]


def test_draw_uses_the_documented_stream():
    import numpy as np
    for seed in (5, 6):
        expected = int(np.random.default_rng(seed).integers(78))
        assert d.draw(seed)[0] == expected


def test_main_refuses_dirty_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "collect_git",
                        lambda: {"commit": "abc", "dirty": True, "dirty_paths": ["x.py"]})
    out = tmp_path / "split.json"
    with pytest.raises(SystemExit):
        d.main(["--out", str(out)])
    assert not out.exists()


def test_main_refuses_missing_git(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "collect_git",
                        lambda: {"commit": None, "dirty": None, "dirty_paths": None})
    with pytest.raises(SystemExit):
        d.main(["--out", str(tmp_path / "split.json")])


def test_main_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "collect_git", lambda: CLEAN)
    out = tmp_path / "split.json"
    out.write_text('{"already": "drawn"}')
    with pytest.raises(SystemExit):
        d.main(["--out", str(out)])
    assert out.read_text() == '{"already": "drawn"}'


def test_main_writes_a_consistent_record(tmp_path, monkeypatch):
    """Uses a stand-in seed so the real held-out set is not revealed."""
    monkeypatch.setattr(d, "collect_git", lambda: CLEAN)
    monkeypatch.setattr(d, "HELDOUT_SEED", 7)
    out = tmp_path / "split.json"
    assert d.main(["--out", str(out)]) == 0
    rec = json.loads(out.read_text())
    index, held_out = d.draw(7)
    assert rec["seed"] == 7 and rec["index"] == index
    assert rec["held_out"] == sorted(held_out)
    assert sorted(rec["held_out"] + rec["training"]) == list(range(25))
    assert len(rec["training"]) == 20
    assert rec["n_eligible"] == 78
    assert rec["git_commit"] == "abc123"
