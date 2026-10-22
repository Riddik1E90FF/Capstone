"""Gate 1a plumbing (scripts/gate1a.py): find_runs, load_episodes, main.

Run folders are created with the real dqn.runlog.RunLogger, so they have
exactly the layout the reference converter and our training loop write.

Skipped until find_runs, load_episodes and main have bodies.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import pytest

from dqn.config import DQNConfig
from dqn.runlog import RunLogger

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gate1a_cli", REPO / "scripts" / "gate1a.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)

with tempfile.TemporaryDirectory() as _empty:
    try:
        _stub = g.find_runs(_empty) is None
    except ValueError:
        _stub = False
if _stub:
    pytest.skip("gate1a.py plumbing not implemented yet", allow_module_level=True)


def _make_run(root: Path, seed: int, *, reach_step=100_000, final=500.0,
              total=500_000, status="completed", name=None) -> Path:
    """A run folder whose moving average first reaches 500 at reach_step."""
    run_dir = root / (name or f"CartPole-v1__test__seed{seed}")
    cfg = DQNConfig().with_overrides({"seed": seed, "total_timesteps": total})
    logger = RunLogger(run_dir, cfg)
    steps = [reach_step - 990 + 10 * i for i in range(100)] + \
            [reach_step + 10 * (i + 1) for i in range(100)]
    returns = [500.0] * 100 + [final] * 100
    for i, (s, r) in enumerate(zip(steps, returns)):
        logger.log_episode(**{"return": r}, global_step=s, episode=i, length=int(r),
                           terminated=r < 500, truncated=r >= 500, epsilon=0.05)
    logger.close(status)
    return run_dir


def _ten(root: Path, **kw) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for seed in range(1, 11):
        _make_run(root, seed, reach_step=50_000 + 10_000 * seed, **kw)
    return root


# -- find_runs ------------------------------------------------------------

def test_find_runs_maps_each_seed(tmp_path):
    root = _ten(tmp_path / "ref")
    runs = g.find_runs(root)
    assert sorted(runs) == list(range(1, 11))
    for seed, path in runs.items():
        assert json.loads((path / "config.json").read_text())["seed"] == seed


def test_find_runs_ignores_smoke_runs_and_other_files(tmp_path):
    root = _ten(tmp_path / "ref")
    _make_run(root, 3, total=20_000, name="CartPole-v1__smoke__seed3")
    (root / "notes.txt").write_text("not a run")
    (root / "empty_dir").mkdir()
    runs = g.find_runs(root)
    assert runs[3].name == "CartPole-v1__test__seed3"


def test_find_runs_missing_seed_raises(tmp_path):
    root = tmp_path / "ref"
    root.mkdir()
    for seed in range(1, 10):                       # no seed 10
        _make_run(root, seed)
    with pytest.raises(ValueError):
        g.find_runs(root)


def test_find_runs_duplicate_seed_raises(tmp_path):
    root = _ten(tmp_path / "ref")
    _make_run(root, 4, name="CartPole-v1__test__seed4__again")
    with pytest.raises(ValueError):
        g.find_runs(root)


# -- load_episodes --------------------------------------------------------

def test_load_episodes_in_order(tmp_path):
    run = _make_run(tmp_path, 1)
    eps = g.load_episodes(run)
    assert len(eps) == 200
    assert [e["episode"] for e in eps] == list(range(200))
    assert {"global_step", "return"} <= eps[0].keys()


@pytest.mark.parametrize("status", ["crashed", "running"])
def test_load_episodes_rejects_unfinished_runs(tmp_path, status):
    run = _make_run(tmp_path, 1, status=status)
    with pytest.raises(ValueError):
        g.load_episodes(run)


# -- main: calibrate ------------------------------------------------------

def test_calibrate_writes_numbers_and_sources(tmp_path, capsys):
    root = _ten(tmp_path / "ref")
    out = tmp_path / "calibration.json"
    assert g.main(["calibrate", "--reference", str(root), "--out", str(out)]) == 0
    cal = json.loads(out.read_text())
    for key in ("T", "k", "q", "F", "delta"):
        assert key in cal, key
    expected = g.calibrate([g.load_episodes(p) for _, p in sorted(g.find_runs(root).items())])
    for key in ("T", "k", "q", "F", "delta"):
        assert cal[key] == pytest.approx(expected[key]), key
    text = json.dumps(cal)
    assert all(p.name in text for p in g.find_runs(root).values()), "run folders not recorded"
    assert "475" in capsys.readouterr().out


def test_calibrate_refuses_to_overwrite(tmp_path):
    root = _ten(tmp_path / "ref")
    out = tmp_path / "calibration.json"
    out.write_text('{"already": "set"}')
    try:
        code = g.main(["calibrate", "--reference", str(root), "--out", str(out)])
    except (SystemExit, FileExistsError):
        code = 1
    assert code != 0
    assert out.read_text() == '{"already": "set"}'


# -- main: check ----------------------------------------------------------

def test_check_prints_decisions(tmp_path, capsys):
    ref = _ten(tmp_path / "ref")
    ours = _ten(tmp_path / "numpy")
    cal = tmp_path / "calibration.json"
    g.main(["calibrate", "--reference", str(ref), "--out", str(cal)])
    capsys.readouterr()
    assert g.main(["check", "--reference", str(ref), "--numpy", str(ours),
                   "--calibration", str(cal)]) == 0
    out = capsys.readouterr().out
    for word in ("L3", "L4_steps", "L4_final", "passed"):
        assert word in out, word


def test_check_failed_gate_still_exits_zero(tmp_path, capsys):
    ref = _ten(tmp_path / "ref")
    ours = _ten(tmp_path / "numpy", final=100.0)       # final returns far too low
    cal = tmp_path / "calibration.json"
    g.main(["calibrate", "--reference", str(ref), "--out", str(cal)])
    assert g.main(["check", "--reference", str(ref), "--numpy", str(ours),
                   "--calibration", str(cal)]) == 0
