"""Tests for experiment/manifest.py: the per-run JSON manifest.

Git behaviour is tested against throwaway repositories in tmp_path, never
against this repository's own state, so the tests pass on a dirty checkout.
"""

from __future__ import annotations

import ast
import json
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from experiment import manifest as mf
from experiment.manifest import (CONDITIONS, MANIFEST_NAME, SCHEMA_VERSION, RunManifest,
                                 build_manifest, collect_environment, collect_git,
                                 normalise_config, read_manifest)

REPO = Path(__file__).resolve().parents[1]


def _git(cwd: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A clean git repository with one commit."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "test")
    _git(root, "config", "commit.gpgsign", "false")
    (root / "tracked.py").write_text("x = 1\n")
    (root / ".gitignore").write_text("ignored/\n")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "init")
    return root


def _strict_load(path: Path) -> dict:
    """json.loads that rejects NaN/Infinity tokens."""
    def reject(token: str):
        raise ValueError(f"non-strict JSON token {token}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def _start(tmp_path: Path, repo: Path, **kw) -> RunManifest:
    args = dict(condition="RANDOM", seed=3, config={"lr": 1e-3}, repo_root=repo)
    args.update(kw)
    return RunManifest(tmp_path / "run", **args)


# ----------------------------------------------------------------------
# Conditions
# ----------------------------------------------------------------------

def test_conditions_match_preregistration_table():
    """CONDITIONS must be exactly the §4 table of the locked pre-registration."""
    text = (REPO / "docs" / "PREREGISTRATION.md").read_text(encoding="utf-8")
    section = text.split("## 4. Conditions", 1)[1].split("\n## ", 1)[0]
    table_ids = re.findall(r"^\| `([A-Z]+)` \|", section, flags=re.MULTILINE)
    assert tuple(table_ids) == CONDITIONS


@pytest.mark.parametrize("condition", CONDITIONS)
def test_every_condition_accepted(condition):
    assert build_manifest(condition, 0, {})["condition"] == condition


@pytest.mark.parametrize("bad", ["random", "EEG", "", "NEURO ", None])
def test_unknown_condition_rejected(bad):
    with pytest.raises(ValueError, match="unknown condition"):
        build_manifest(bad, 0, {})


# ----------------------------------------------------------------------
# Seed
# ----------------------------------------------------------------------

def test_seed_zero_accepted():
    assert build_manifest("STATIC", 0, {})["seed"] == 0


@pytest.mark.parametrize("bad", [True, 1.0, "1", np.int64(1)])
def test_non_int_seed_rejected(bad):
    with pytest.raises(TypeError, match="seed must be an int"):
        build_manifest("STATIC", bad, {})


def test_negative_seed_rejected():
    with pytest.raises(ValueError, match="seed must be >= 0"):
        build_manifest("STATIC", -1, {})


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class _Inner:
    sizes: tuple[int, ...] = (4, 8)


@dataclass(frozen=True)
class _Outer:
    lr: float = 0.5
    inner: _Inner = field(default_factory=_Inner)


def test_nested_dataclass_config_becomes_plain_json():
    assert normalise_config(_Outer()) == {"lr": 0.5, "inner": {"sizes": [4, 8]}}


def test_mapping_config_with_numpy_and_path_values():
    cfg = {"gamma": np.float64(0.99), "arr": np.arange(3), "out": Path("runs/x")}
    assert normalise_config(cfg) == {"gamma": 0.99, "arr": [0, 1, 2], "out": "runs/x"}


def test_dqn_config_matches_its_own_serialisation():
    from dqn.config import DQNConfig
    cfg = DQNConfig()
    assert normalise_config(cfg) == cfg.to_dict()


def test_env_config_serialises():
    from combat_env.config import DifficultyConfig, EnvConfig
    out = normalise_config(EnvConfig(DifficultyConfig.from_id(7)))
    assert out["difficulty"] == {"telegraph_level": 1, "gap_level": 2}
    assert {"params", "rewards"} <= set(out)
    json.dumps(out, allow_nan=False)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), np.float64("nan")])
def test_non_finite_config_value_rejected(bad):
    with pytest.raises(ValueError, match="non-finite"):
        normalise_config({"lr": bad})


def test_unserialisable_config_value_rejected():
    with pytest.raises(TypeError, match="not JSON-serialisable"):
        normalise_config({"fn": object()})


@pytest.mark.parametrize("bad", [[("lr", 1)], "lr=1", 3, _Outer])
def test_non_mapping_config_rejected(bad):
    with pytest.raises(TypeError, match="mapping or a dataclass instance"):
        normalise_config(bad)


def test_config_is_a_copy_not_a_reference():
    cfg = {"sizes": [1, 2]}
    out = normalise_config(cfg)
    cfg["sizes"].append(3)
    assert out == {"sizes": [1, 2]}


# ----------------------------------------------------------------------
# Git
# ----------------------------------------------------------------------

def test_clean_repo(repo):
    g = collect_git(repo)
    assert g == {"commit": _git(repo, "rev-parse", "HEAD"), "dirty": False, "dirty_paths": []}
    assert re.fullmatch(r"[0-9a-f]{40}", g["commit"])


def test_modified_tracked_file_is_dirty(repo):
    (repo / "tracked.py").write_text("x = 2\n")
    g = collect_git(repo)
    assert g["dirty"] is True and g["dirty_paths"] == ["tracked.py"]


def test_untracked_file_is_dirty(repo):
    (repo / "pkg").mkdir()
    (repo / "pkg" / "new.py").write_text("")
    g = collect_git(repo)
    assert g["dirty"] is True and g["dirty_paths"] == ["pkg/new.py"]


def test_ignored_file_is_not_dirty(repo):
    (repo / "ignored").mkdir()
    (repo / "ignored" / "out.json").write_text("{}")
    assert collect_git(repo)["dirty"] is False


def test_not_a_repository_records_nulls(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    assert collect_git(plain) == {"commit": None, "dirty": None, "dirty_paths": None}


def test_missing_git_binary_records_nulls(repo, monkeypatch):
    def no_git(*a, **k):
        raise FileNotFoundError("git")
    monkeypatch.setattr(mf.subprocess, "check_output", no_git)
    assert collect_git(repo) == {"commit": None, "dirty": None, "dirty_paths": None}


# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------

def test_environment_versions():
    env = collect_environment(("numpy", "definitely-not-installed-pkg"))
    assert env["python"] == platform.python_version()
    assert env["python_implementation"] == platform.python_implementation()
    assert env["packages"] == {"numpy": np.__version__, "definitely-not-installed-pkg": None}


def test_default_packages_recorded():
    pkgs = collect_environment()["packages"]
    assert set(pkgs) == set(mf.DEFAULT_PACKAGES)
    assert pkgs["numpy"] == np.__version__


# ----------------------------------------------------------------------
# build_manifest
# ----------------------------------------------------------------------

def test_build_manifest_fields(repo):
    m = build_manifest("NEURO", 7, {"a": 1}, repo_root=repo, extra={"note": "x"})
    assert set(m) == {"schema_version", "status", "condition", "seed", "config", "git",
                      "environment", "argv", "start_utc", "end_utc", "wall_seconds",
                      "error", "extra"}
    assert m["schema_version"] == SCHEMA_VERSION
    assert m["status"] == "running"
    assert (m["condition"], m["seed"], m["config"]) == ("NEURO", 7, {"a": 1})
    assert m["git"]["dirty"] is False
    assert m["argv"] == sys.argv
    assert datetime.fromisoformat(m["start_utc"]).utcoffset().total_seconds() == 0
    assert (m["end_utc"], m["wall_seconds"], m["error"]) == (None, None, None)
    assert m["extra"] == {"note": "x"}


def test_extra_with_nan_rejected():
    with pytest.raises(ValueError, match="extra contains a non-finite"):
        build_manifest("STATIC", 0, {}, extra={"x": float("nan")})


# ----------------------------------------------------------------------
# RunManifest lifecycle
# ----------------------------------------------------------------------

def test_written_at_start_as_running(tmp_path, repo):
    m = _start(tmp_path, repo)
    on_disk = _strict_load(m.path)
    assert m.path == tmp_path / "run" / MANIFEST_NAME
    assert on_disk["status"] == "running" and on_disk["end_utc"] is None
    assert on_disk == m.data
    assert not m.finished


def test_completed_on_clean_exit(tmp_path, repo):
    with _start(tmp_path, repo) as m:
        pass
    on_disk = _strict_load(m.path)
    assert on_disk["status"] == "completed" and on_disk["error"] is None
    start = datetime.fromisoformat(on_disk["start_utc"])
    end = datetime.fromisoformat(on_disk["end_utc"])
    assert end >= start
    assert on_disk["wall_seconds"] >= 0


def test_crashed_on_exception_and_exception_propagates(tmp_path, repo):
    with pytest.raises(RuntimeError, match="diverged"):
        with _start(tmp_path, repo) as m:
            raise RuntimeError("diverged")
    on_disk = _strict_load(m.path)
    assert on_disk["status"] == "crashed"
    assert on_disk["error"] == "RuntimeError: diverged"
    assert on_disk["end_utc"] is not None


def test_explicit_finish_not_overwritten_on_exit(tmp_path, repo):
    with _start(tmp_path, repo) as m:
        m.finish("crashed", error="watchdog")
    assert _strict_load(m.path)["status"] == "crashed"
    assert _strict_load(m.path)["error"] == "watchdog"


def test_finish_is_idempotent(tmp_path, repo):
    m = _start(tmp_path, repo)
    m.finish()
    first = m.path.read_text()
    m.finish("crashed", error="late")
    assert m.path.read_text() == first


@pytest.mark.parametrize("bad", ["running", "done", ""])
def test_finish_rejects_bad_status(tmp_path, repo, bad):
    m = _start(tmp_path, repo)
    with pytest.raises(ValueError, match="finish status"):
        m.finish(bad)
    assert not m.finished


def test_refuses_to_overwrite_existing_manifest(tmp_path, repo):
    first = _start(tmp_path, repo, seed=1)
    first.finish()
    before = first.path.read_text()
    with pytest.raises(FileExistsError):
        _start(tmp_path, repo, seed=2)
    assert first.path.read_text() == before


def test_coexists_with_other_files_in_run_dir(tmp_path, repo):
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text("{}")
    m = _start(tmp_path, repo)
    assert m.path.exists() and (run / "config.json").read_text() == "{}"


def test_invalid_arguments_leave_nothing_behind(tmp_path, repo):
    with pytest.raises(ValueError):
        _start(tmp_path, repo, condition="BOGUS")
    with pytest.raises(ValueError):
        _start(tmp_path, repo, config={"lr": float("nan")})
    assert not (tmp_path / "run").exists()


def test_no_temp_file_left_behind(tmp_path, repo):
    with _start(tmp_path, repo) as m:
        pass
    assert sorted(p.name for p in m.path.parent.iterdir()) == [MANIFEST_NAME]


def test_data_is_a_copy(tmp_path, repo):
    m = _start(tmp_path, repo)
    m.data["condition"] = "NEURO"
    assert m.data["condition"] == "RANDOM"


# ----------------------------------------------------------------------
# read_manifest
# ----------------------------------------------------------------------

def test_read_manifest_from_file_or_directory(tmp_path, repo):
    with _start(tmp_path, repo) as m:
        pass
    assert read_manifest(m.path) == read_manifest(m.path.parent) == _strict_load(m.path)


def test_read_manifest_rejects_unknown_schema(tmp_path):
    p = tmp_path / MANIFEST_NAME
    p.write_text(json.dumps({"schema_version": SCHEMA_VERSION + 1}))
    with pytest.raises(ValueError, match="unsupported manifest schema_version"):
        read_manifest(p)


# ----------------------------------------------------------------------
# Import boundary
# ----------------------------------------------------------------------

def test_experiment_package_imports_stdlib_only():
    """Experiment infrastructure must not pull in RL or EEG libraries.  Relative
    imports (within the package) are allowed."""
    offenders = []
    for path in (REPO / "experiment").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if top not in sys.stdlib_module_names and top != "__future__":
                    offenders.append(f"{path.name}: {name}")
    assert offenders == []
