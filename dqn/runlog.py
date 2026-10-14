"""Structured run logging: one directory per run, JSONL streams inside it.

Layout of a run directory::

    runs/<run_name>/
        config.json      fully resolved DQNConfig (every field, not just overrides)
        meta.json        provenance: git commit, dirty flag, versions, argv, start time
        episodes.jsonl   one record per completed training episode
        updates.jsonl    training-update diagnostics (loss, mean Q)
        evals.jsonl      greedy evaluation results (if eval_frequency > 0)
        checkpoints/     see dqn/README.md, "Checkpoint structure"
        summary.json     written by close(): end time, counts, exit status

The schema is shared with the CleanRL reference runner (Gate 1a, D029, build
step 8).  Both implementations produce identical files, so one analysis script
reads both.  That is why each stream has **required keys**, checked on every
write: a missing field fails at write time, not three days later in the
analysis.  Extra keys are allowed.

Lines are flushed as they are written, so a crashed run still leaves every
completed episode on disk.  An existing non-empty run directory is never
overwritten: runs are experimental evidence.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from .config import DQNConfig

# Required keys per stream.  Field meanings:
#   global_step  env steps taken so far, counted *after* the step that
#                finished the episode (or triggered the update/eval)
#   episode      0-based training-episode index
#   return       undiscounted sum of rewards for the episode
#   length       steps in the episode
#   terminated   episode ended in a true terminal state
#   truncated    episode was cut off by a time limit
#   epsilon      exploration rate in force on the episode's final step
#   loss         TD loss of this update's minibatch
#   q_mean       mean over the minibatch of Q(s, a) for the actions taken
#   mean_return  mean undiscounted return over the evaluation episodes
#   returns      per-episode evaluation returns, as a list
REQUIRED_KEYS: dict[str, frozenset[str]] = {
    "episodes": frozenset({"global_step", "episode", "return", "length",
                           "terminated", "truncated", "epsilon"}),
    "updates": frozenset({"global_step", "loss", "q_mean"}),
    "evals": frozenset({"global_step", "mean_return", "returns"}),
}


def _jsonable(value: Any) -> Any:
    """json.dumps `default=` hook: numpy scalars/arrays to Python types."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON-serialisable: {type(value).__name__}")


def _git(*args: str, cwd: Path) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance(repo_root: Path | None = None) -> dict[str, Any]:
    """Everything needed to say exactly which code produced a run."""
    root = repo_root or Path(__file__).resolve().parents[1]
    status = _git("status", "--porcelain", "--", "dqn", "scripts", "configs", cwd=root)
    return {
        "git_commit": _git("rev-parse", "HEAD", cwd=root),
        # Dirty = uncommitted changes in code that affects training.  A dirty
        # run is not reproducible from the commit alone.  Allowed while
        # developing, but no Gate 1a or experiment run should be dirty.
        "git_dirty": bool(status) if status is not None else None,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "argv": sys.argv,
        "start_utc": datetime.now(timezone.utc).isoformat(),
    }


class RunLogger:
    """Owns one run directory.  Use as a context manager so `summary.json`
    is written even if training raises."""

    def __init__(self, run_dir: str | Path, config: DQNConfig,
                 extra_meta: dict[str, Any] | None = None) -> None:
        self.run_dir = Path(run_dir)
        if self.run_dir.exists() and any(self.run_dir.iterdir()):
            raise FileExistsError(
                f"run directory {self.run_dir} already exists and is not empty; "
                "refusing to overwrite a previous run")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)

        config.to_json(self.run_dir / "config.json")
        meta = provenance()
        if extra_meta:
            meta.update(extra_meta)
        (self.run_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, default=_jsonable) + "\n", encoding="utf-8")

        self._files: dict[str, TextIO] = {}
        self._counts: dict[str, int] = {name: 0 for name in REQUIRED_KEYS}
        self._t0 = time.monotonic()
        self._closed = False

    @property
    def checkpoint_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    # -- writing -------------------------------------------------------

    def _write(self, stream: str, record: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("RunLogger is closed")
        missing = REQUIRED_KEYS[stream] - record.keys()
        if missing:
            raise KeyError(f"{stream} record missing required key(s): {sorted(missing)}")
        f = self._files.get(stream)
        if f is None:
            f = self._files[stream] = open(self.run_dir / f"{stream}.jsonl", "a", encoding="utf-8")
        # NaN/inf are written as the bare tokens NaN/Infinity.  That is not
        # strict JSON, but Python's json.loads reads it back, and a diverging
        # loss is information that must not be dropped or crash the run.
        f.write(json.dumps(record, default=_jsonable) + "\n")
        f.flush()
        self._counts[stream] += 1

    def log_episode(self, **record: Any) -> None:
        self._write("episodes", record)

    def log_update(self, **record: Any) -> None:
        self._write("updates", record)

    def log_eval(self, **record: Any) -> None:
        self._write("evals", record)

    # -- lifecycle -----------------------------------------------------

    def close(self, status: str = "completed", **extra: Any) -> None:
        if self._closed:
            return
        for f in self._files.values():
            f.close()
        summary = {
            "status": status,
            "end_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": round(time.monotonic() - self._t0, 3),
            "records": dict(self._counts),
            **extra,
        }
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=_jsonable) + "\n", encoding="utf-8")
        self._closed = True

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close("completed")
        else:
            self.close("crashed", error=f"{exc_type.__name__}: {exc}")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read one stream back.  Accepts the NaN/Infinity tokens written above."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows
