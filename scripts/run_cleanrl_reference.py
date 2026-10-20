"""Run the unmodified CleanRL DQN reference on CartPole-v1 and convert its logs
(build step 8, Gate 1a, D029).

Each seed runs `third_party/cleanrl/dqn.py --seed N` in a subprocess, with
CleanRL's own defaults and nothing else, inside its own run directory.  When
it finishes, its TensorBoard event file is converted into the same run-folder
format our DQN writes (dqn/runlog.py), so one analysis script reads both.

Layout per seed:

    runs/gate1a/cleanrl/CartPole-v1__cleanrl__seed<N>__<UTC>/
        config.json  meta.json  summary.json      written by RunLogger
        episodes.jsonl  updates.jsonl              converted from TensorBoard
        raw/stdout.log                             CleanRL's console output
        raw/runs/<cleanrl run name>/events.*       CleanRL's own TensorBoard log

Conversion rules (CleanRL logs only some of our fields):
  * global_step: CleanRL's 0-based loop index + 1 (runlog.py's convention).
  * episode: 0-based order of the episode records.
  * return, length: charts/episodic_return, charts/episodic_length.
  * truncated: length == the env's time limit (500 for CartPole-v1).
    terminated: not truncated.  CleanRL does not log either flag; an episode
    whose pole falls on exactly the last step counts as truncated here.  No
    Gate 1a metric uses these two fields.
  * epsilon: recomputed with CleanRL's linear_schedule at the 0-based step.
  * updates: losses/td_loss and losses/q_values, logged by CleanRL every 100
    steps on update steps.  They are float32 in TensorBoard.

Usage:
    python scripts/run_cleanrl_reference.py                   # seeds 1-10, protocol run
    python scripts/run_cleanrl_reference.py --seeds 3 --smoke-steps 20000
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dqn.config import DQNConfig  # noqa: E402
from dqn.runlog import RunLogger  # noqa: E402

REFERENCE_DIR = REPO_ROOT / "third_party" / "cleanrl"
REFERENCE_SCRIPT = REFERENCE_DIR / "dqn.py"
REFERENCE_COMMIT = "fe8d8a03c41a7ef5b523e2e354bd01c363e786bb"
ENV_ID = "CartPole-v1"

# CleanRL Args defaults at REFERENCE_COMMIT (the values the script runs with).
REFERENCE_DEFAULTS = dict(start_e=1.0, end_e=0.05, exploration_fraction=0.5,
                          total_timesteps=500_000)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def linear_schedule(start_e: float, end_e: float, duration: float, t: int) -> float:
    """CleanRL's own function, copied verbatim, to recompute logged epsilon."""
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def read_scalars(event_dir: Path) -> dict[str, list[tuple[int, float]]]:
    """All scalar series in a TensorBoard log directory, as (step, value)."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    acc.Reload()
    return {tag: [(int(e.step), float(e.value)) for e in acc.Scalars(tag)]
            for tag in acc.Tags()["scalars"]}


def convert(scalars: dict[str, list[tuple[int, float]]], logger: RunLogger, *,
            total_timesteps: int, time_limit: int) -> dict[str, int]:
    """Write CleanRL's scalars as our episode and update records."""
    returns = sorted(scalars.get("charts/episodic_return", []))
    lengths = sorted(scalars.get("charts/episodic_length", []))
    if [s for s, _ in returns] != [s for s, _ in lengths]:
        raise ValueError("episodic_return and episodic_length are logged at different steps")
    duration = REFERENCE_DEFAULTS["exploration_fraction"] * total_timesteps

    for i, ((step, ret), (_, length)) in enumerate(zip(returns, lengths)):
        length = int(length)
        truncated = length >= time_limit
        logger.log_episode(**{"return": ret}, global_step=step + 1, episode=i,
                           length=length, terminated=not truncated, truncated=truncated,
                           epsilon=linear_schedule(REFERENCE_DEFAULTS["start_e"],
                                                   REFERENCE_DEFAULTS["end_e"], duration, step))

    losses = dict(scalars.get("losses/td_loss", []))
    q_values = dict(scalars.get("losses/q_values", []))
    for step in sorted(losses):
        logger.log_update(global_step=step + 1, loss=losses[step], q_mean=q_values[step])
    return {"episodes": len(returns), "updates": len(losses)}


def run_seed(seed: int, runs_root: Path, smoke_steps: int | None) -> dict[str, Any]:
    import gymnasium as gym
    import torch

    total = smoke_steps or REFERENCE_DEFAULTS["total_timesteps"]
    tag = "cleanrl-smoke" if smoke_steps else "cleanrl"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = runs_root / f"{ENV_ID}__{tag}__seed{seed}__{stamp}"
    raw = run_dir / "raw"

    # Protocol runs pass only --seed.  Smoke runs also shorten the run.
    cmd = [sys.executable, str(REFERENCE_SCRIPT), "--seed", str(seed)]
    if smoke_steps:
        cmd += ["--total-timesteps", str(smoke_steps)]
    # One thread per process: the runs go in parallel, and thread count
    # changes only speed, not the algorithm.
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    meta = {
        "implementation": "cleanrl",
        "protocol_run": smoke_steps is None,
        "reference_commit": REFERENCE_COMMIT,
        "reference_sha256": {
            "dqn.py": sha256(REFERENCE_SCRIPT),
            "cleanrl_utils/buffers.py": sha256(REFERENCE_DIR / "cleanrl_utils" / "buffers.py"),
        },
        "command": cmd,
        "thread_env": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "gymnasium": gym.__version__,
        "python": platform.python_version(),
    }
    cfg = DQNConfig().with_overrides({"seed": seed, "total_timesteps": total})
    time_limit = gym.spec(ENV_ID).max_episode_steps

    with RunLogger(run_dir, cfg, extra_meta=meta) as logger:
        raw.mkdir()
        t0 = time.monotonic()
        with open(raw / "stdout.log", "w", encoding="utf-8") as out:
            proc = subprocess.run(cmd, cwd=raw, env=env, stdout=out, stderr=subprocess.STDOUT)
        wall = round(time.monotonic() - t0, 1)
        if proc.returncode != 0:
            raise RuntimeError(f"seed {seed}: CleanRL exited with {proc.returncode}; "
                               f"see {raw / 'stdout.log'}")
        event_dirs = sorted({p.parent for p in (raw / "runs").rglob("events.out.tfevents.*")})
        if len(event_dirs) != 1:
            raise RuntimeError(f"seed {seed}: expected one TensorBoard log, found {event_dirs}")
        counts = convert(read_scalars(event_dirs[0]), logger,
                         total_timesteps=total, time_limit=time_limit)
        logger.close("completed", reference_wall_seconds=wall, converted=counts)
    return {"seed": seed, "run_dir": str(run_dir), "wall_seconds": wall, **counts}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)),
                   help="reference seeds (D029: 1-10)")
    p.add_argument("--runs-root", type=Path, default=REPO_ROOT / "runs" / "gate1a" / "cleanrl")
    p.add_argument("--workers", type=int, default=5, help="seeds run in parallel")
    p.add_argument("--smoke-steps", type=int,
                   help="shorten each run to this many steps (not a protocol run)")
    args = p.parse_args(argv)

    print(f"CleanRL {REFERENCE_COMMIT[:7]}, seeds {args.seeds}, "
          f"{'SMOKE ' + str(args.smoke_steps) + ' steps' if args.smoke_steps else 'protocol run'}")
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_seed, s, args.runs_root, args.smoke_steps): s
                   for s in args.seeds}
        for fut, seed in futures.items():
            try:
                r = fut.result()
                print(f"seed {seed}: {r['episodes']} episodes, {r['updates']} update records, "
                      f"{r['wall_seconds']} s -> {r['run_dir']}", flush=True)
            except Exception as exc:  # report every seed, then fail overall
                failures += 1
                print(f"seed {seed}: FAILED: {exc}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
