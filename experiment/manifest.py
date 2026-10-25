"""Run manifest: one JSON file recording exactly what produced an experiment run.

Layout::

    runs/<run_name>/
        manifest.json

Contents (``schema_version`` 1)::

    schema_version   int, bumped on any incompatible change to this layout
    status           "running" | "completed" | "crashed"
    condition        one of CONDITIONS
    seed             master seed, int >= 0
    config           every parameter of the run, as plain JSON
    git              {"commit", "dirty", "dirty_paths"}
    environment      {"python", "python_implementation", "platform", "packages"}
    argv             the command line
    start_utc        ISO-8601 UTC
    end_utc          ISO-8601 UTC, null while running
    wall_seconds     float, null while running
    error            "ExcType: message" if crashed, else null
    extra            free-form caller metadata (e.g. a SHUFFLE permutation)

Why a manifest separate from ``dqn/runlog.py``'s ``meta.json``
---------------------------------------------------------------
``meta.json`` is algorithm-level provenance for DQN runs, and its dirty flag
covers only ``dqn/``, ``scripts/`` and ``configs/``.  An experiment run also
depends on ``combat_env/`` and the curriculum code, so the dirty flag here
covers the whole working tree.  The manifest also records the condition, the
one thing that differs between runs by design (pre-registration §4).

Lifecycle
---------
The manifest is written when the run starts, with ``status: "running"``, and
rewritten when it ends.  Every write is atomic (temp file, then
``os.replace``).  A run killed hard therefore leaves a manifest that still
says ``"running"`` with no end time.  It never leaves a half-written file.
That keeps the §9 rule enforceable: a seed is dropped only for a logged
technical failure, and the manifest is where the failure is logged.

An existing manifest is never overwritten, because runs are experimental
evidence.  Only ``manifest.json`` is checked, not the whole directory, so a
``RunLogger`` can own the same run directory.  Create the ``RunLogger`` first,
since it refuses a non-empty directory.

Strict JSON
-----------
Written with ``allow_nan=False``.  A NaN or infinity in a config is a bug, and
it fails when the run starts rather than when the file is read.  (Diverging
losses belong in the JSONL streams, which do allow NaN.)
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"

# Pre-registration §4, in table order.  REVERSE is D027 (confirmed).
CONDITIONS = ("STATIC", "RANDOM", "NEURO", "PERF", "BEHAV", "REVERSE", "SHUFFLE")

STATUSES = ("running", "completed", "crashed")

# Every third-party package named in requirements*.txt that can change what a
# run computes.  Recorded as null when not installed.  Test-only tools
# (pytest) and CleanRL's logging dependencies (tyro, tensorboard) are left out.
DEFAULT_PACKAGES = ("numpy", "gymnasium", "torch", "pygame")

_REPO_ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------

def validate_condition(condition: str) -> str:
    if condition not in CONDITIONS:
        raise ValueError(
            f"unknown condition {condition!r}; expected one of {', '.join(CONDITIONS)}")
    return condition


def validate_seed(seed: int) -> int:
    # bool is a subclass of int; seed=True is a bug, not seed 1.
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    return seed


def _json_default(value: Any) -> Any:
    """json.dumps `default=` hook for the types run configs actually contain."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    # NumPy scalars and arrays, without importing NumPy.
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"not JSON-serialisable: {type(value).__name__}")


def _to_plain_json(value: Any, what: str) -> Any:
    try:
        return json.loads(json.dumps(value, default=_json_default, allow_nan=False))
    except ValueError as exc:  # NaN / inf
        raise ValueError(f"{what} contains a non-finite float: {exc}") from None
    except TypeError as exc:
        raise TypeError(f"{what} is not JSON-serialisable: {exc}") from None


def normalise_config(config: Mapping[str, Any] | Any) -> dict[str, Any]:
    """A mapping or a dataclass instance, as plain JSON types (tuples become
    lists, nested dataclasses become dicts)."""
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        config = dataclasses.asdict(config)
    if not isinstance(config, Mapping):
        raise TypeError(
            f"config must be a mapping or a dataclass instance, got {type(config).__name__}")
    return _to_plain_json(dict(config), "config")


# ----------------------------------------------------------------------
# Collection
# ----------------------------------------------------------------------

def _git(*args: str, cwd: Path) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True,
                                       stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_git(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Commit and dirty state of the whole working tree.

    Untracked files count as dirty unless ignored: an uncommitted new module
    can change a run as much as an edited one.  All three fields are null
    when git or the repository is unavailable, which is recorded rather than
    raised so that a run never fails for want of provenance.  A null commit
    still marks the run as not reproducible from the manifest alone.
    """
    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    commit = _git("rev-parse", "HEAD", cwd=root)
    status = _git("status", "--porcelain", "--untracked-files=all", cwd=root)
    if commit is None or status is None:
        return {"commit": None, "dirty": None, "dirty_paths": None}
    # Porcelain v1 lines are "XY <path>"; the path starts at column 3.
    paths = sorted(line[3:] for line in status.splitlines() if line.strip())
    return {"commit": commit.strip(), "dirty": bool(paths), "dirty_paths": paths}


def collect_environment(packages: tuple[str, ...] = DEFAULT_PACKAGES) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": versions,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_manifest(condition: str, seed: int, config: Mapping[str, Any] | Any, *,
                   packages: tuple[str, ...] = DEFAULT_PACKAGES,
                   repo_root: str | Path | None = None,
                   extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The start-of-run manifest as a dict.  Validates every input."""
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "condition": validate_condition(condition),
        "seed": validate_seed(seed),
        "config": normalise_config(config),
        "git": collect_git(repo_root),
        "environment": collect_environment(packages),
        "argv": list(sys.argv),
        "start_utc": _utc_now(),
        "end_utc": None,
        "wall_seconds": None,
        "error": None,
        "extra": _to_plain_json(dict(extra or {}), "extra"),
    }


# ----------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------

def _write_atomic(path: Path, manifest: dict[str, Any]) -> None:
    text = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def read_manifest(path: str | Path) -> dict[str, Any]:
    """Load a manifest file (or a run directory containing one).

    Fails loudly on an unknown schema version rather than guessing.
    """
    path = Path(path)
    if path.is_dir():
        path = path / MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    version = manifest.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported manifest schema_version {version!r} "
                         f"(this code reads {SCHEMA_VERSION})")
    return manifest


class RunManifest:
    """Owns ``<run_dir>/manifest.json`` for one run.

    Use as a context manager, so the end state is written even if the run
    raises::

        with RunManifest(run_dir, condition="RANDOM", seed=3, config=cfg):
            ...  # train
    """

    def __init__(self, run_dir: str | Path, *, condition: str, seed: int,
                 config: Mapping[str, Any] | Any,
                 packages: tuple[str, ...] = DEFAULT_PACKAGES,
                 repo_root: str | Path | None = None,
                 extra: Mapping[str, Any] | None = None) -> None:
        # Build (and validate) before touching the filesystem, so a bad
        # argument leaves nothing behind.
        manifest = build_manifest(condition, seed, config, packages=packages,
                                  repo_root=repo_root, extra=extra)
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / MANIFEST_NAME
        if self.path.exists():
            raise FileExistsError(
                f"{self.path} already exists; refusing to overwrite a previous run's manifest")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._manifest = manifest
        self._t0 = time.monotonic()
        _write_atomic(self.path, self._manifest)

    @property
    def data(self) -> dict[str, Any]:
        """A copy of the manifest as currently written."""
        return json.loads(json.dumps(self._manifest))

    @property
    def finished(self) -> bool:
        return self._manifest["status"] != "running"

    def finish(self, status: str = "completed", error: str | None = None) -> None:
        """Record the end of the run.  Later calls are ignored, so an explicit
        ``finish`` inside a ``with`` block is not overwritten on exit."""
        if status not in STATUSES or status == "running":
            raise ValueError(f"finish status must be 'completed' or 'crashed', got {status!r}")
        if self.finished:
            return
        self._manifest.update(
            status=status,
            end_utc=_utc_now(),
            wall_seconds=round(time.monotonic() - self._t0, 3),
            error=error,
        )
        _write_atomic(self.path, self._manifest)

    def __enter__(self) -> "RunManifest":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.finish("completed")
        else:
            self.finish("crashed", error=f"{exc_type.__name__}: {exc}")
