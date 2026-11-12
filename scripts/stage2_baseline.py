"""Stage-2 baseline runs: the between-seed variance of the primary DV (D038, D040).

Trains the DQN with Gate 1's configuration on ``combat:train`` (uniform over
the 20 training configurations) for seeds 101-110, then evaluates each seed's
final network on the 5 held-out configurations with D039's normalized return
(`evaluation/heldout.py`, the Gate 1 evaluation protocol).  The per-seed
primary DV and its mean and SD go to `docs/stage2_baseline.json`, the input
the Stage-2 amendment's seed count needs.

The firewall (D038): these runs are operationally the RANDOM condition, so
they are used only for the variance estimate.  Their seeds are reserved
(`RESERVED_SEEDS`) and must never be used for condition runs; their results
are never pooled into any condition.

Provenance: refuses a working tree with uncommitted changes, and refuses to
evaluate any run whose manifest records a different commit or a dirty tree.
Completed runs are reused, so an interrupted launch can be resumed from the
same commit.  `docs/stage2_baseline.json` is never overwritten.

Usage (from the repo root):
    python scripts/stage2_baseline.py               # train, then evaluate
    python scripts/stage2_baseline.py --workers 5
"""

from __future__ import annotations

# Single-threaded BLAS in this process and every child; set before NumPy is
# imported.  Parallel runs with multi-threaded BLAS on small matrices are
# slower, and thread count is a suspected source of the run-to-run differences
# noted in docs/HP_SWEEP.md.
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.gate1 import EVAL_SEED_BASE, N_EVAL
from experiment.manifest import collect_git

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_ID = "combat:train"
RESERVED_SEEDS: tuple[int, ...] = tuple(range(101, 111))   # D038 firewall
RUN_ROOT = REPO_ROOT / "runs" / "stage2" / "baseline"
OUT_PATH = REPO_ROOT / "docs" / "stage2_baseline.json"


def run_dir_for(seed: int, root: Path = RUN_ROOT) -> Path:
    return root / f"seed{seed}"


def train_command(seed: int, run_dir: Path, extra_sets: tuple[str, ...] = ()) -> list[str]:
    """The ordinary training CLI with Gate 1's configuration; only env and seed differ."""
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "train_dqn.py"),
           "--set", f"env_id={ENV_ID}", "--seed", str(seed), "--run-dir", str(run_dir)]
    for item in extra_sets:
        cmd += ["--set", item]
    return cmd


def run_status(run_dir: Path) -> str | None:
    """'completed', another recorded status, or None if the run never finished."""
    summary = run_dir / "summary.json"
    if not summary.exists():
        return None
    return json.loads(summary.read_text(encoding="utf-8")).get("status")


def check_provenance(run_dir: Path, commit: str) -> list[str]:
    """Reasons this run cannot be used at `commit`; empty if it can."""
    problems = []
    if run_status(run_dir) != "completed":
        problems.append(f"{run_dir.name}: not completed")
        return problems
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    if meta.get("git_commit") != commit:
        problems.append(f"{run_dir.name}: trained at {meta.get('git_commit')}, HEAD is {commit}")
    if meta.get("git_dirty") is not False:
        problems.append(f"{run_dir.name}: trained from a dirty or unknown tree")
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if cfg.get("env_id") != ENV_ID:
        problems.append(f"{run_dir.name}: env_id {cfg.get('env_id')!r}, expected {ENV_ID!r}")
    return problems


def train_one(seed: int, run_dir: Path, extra_sets: tuple[str, ...] = ()) -> float:
    t0 = time.time()
    proc = subprocess.run(train_command(seed, run_dir, extra_sets), cwd=REPO_ROOT,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"seed {seed} failed (exit {proc.returncode}):\n"
                           f"{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}")
    return time.time() - t0


def evaluate_run(run_dir: Path, held_out: list[int], anchors: dict[str, Any],
                 n_eval: int = N_EVAL) -> dict[str, Any]:
    """Primary DV of one run's final network, with the per-configuration detail."""
    from dqn.checkpoint import latest_checkpoint, load_checkpoint
    from evaluation.heldout import evaluate_returns, normalized_returns, primary_dv
    from gate1_check import greedy_policy

    ckpt = latest_checkpoint(run_dir / "checkpoints")
    if ckpt is None:
        raise RuntimeError(f"no checkpoint under {run_dir / 'checkpoints'}")
    network, meta = load_checkpoint(ckpt)
    res = evaluate_returns(greedy_policy(network), config_ids=held_out, n_eval=n_eval)
    return {
        "checkpoint": ckpt.name,
        "primary_dv": primary_dv(res["returns"], anchors, held_out),
        "normalized": {str(c): v for c, v in
                       normalized_returns(res["returns"], anchors, held_out).items()},
        "returns": {str(c): v for c, v in res["returns"].items()},
        "damage_taken": {str(c): v for c, v in res["damage_taken"].items()},
    }


def summarize(dvs: dict[int, float]) -> dict[str, float]:
    """Mean and between-seed SD (ddof = 1) of the per-seed primary DV."""
    values = np.array([dvs[s] for s in sorted(dvs)], dtype=float)
    if len(values) < 2:
        raise ValueError("the between-seed SD needs at least two seeds")
    return {"n_seeds": len(values), "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)), "min": float(values.min()),
            "max": float(values.max())}


def _evaluate_task(args: tuple[str, list[int], dict[str, Any], int]) -> dict[str, Any]:
    run_dir, held_out, anchors, n_eval = args
    return evaluate_run(Path(run_dir), held_out, anchors, n_eval)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workers", type=int, default=10, help="parallel processes")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--run-root", type=Path, default=RUN_ROOT)
    args = ap.parse_args(argv)

    if args.out.exists():
        raise SystemExit(f"{args.out} already exists; the baseline is recorded once.")
    git = collect_git()
    if git["commit"] is None:
        raise SystemExit("git unavailable; the runs must record the commit that trained them.")
    if git["dirty"]:
        raise SystemExit("uncommitted changes in the working tree; commit first:\n  "
                         + "\n  ".join(git["dirty_paths"]))

    from evaluation.heldout import ANCHORS_PATH, SPLIT_PATH, load_anchors, load_held_out
    anchors = load_anchors(REPO_ROOT / ANCHORS_PATH)
    held_out = load_held_out(REPO_ROOT / SPLIT_PATH)

    # -- train ---------------------------------------------------------------
    todo = []
    for seed in RESERVED_SEEDS:
        rd = run_dir_for(seed, args.run_root)
        if rd.exists():
            problems = check_provenance(rd, git["commit"])
            if problems:
                raise SystemExit("an existing run cannot be reused; move it aside:\n  "
                                 + "\n  ".join(problems))
            print(f"seed {seed}: completed at this commit, reused")
        else:
            todo.append(seed)

    t0 = time.time()
    if todo:
        print(f"training seeds {todo} with {args.workers} workers")
        args.run_root.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {s: pool.submit(train_one, s, run_dir_for(s, args.run_root))
                       for s in todo}
            for s, f in futures.items():
                print(f"  seed {s}: {f.result():.0f} s", flush=True)
    train_wall = time.time() - t0

    problems = [p for s in RESERVED_SEEDS
                for p in check_provenance(run_dir_for(s, args.run_root), git["commit"])]
    if problems:
        raise SystemExit("runs failed the provenance check; nothing written:\n  "
                         + "\n  ".join(problems))

    # -- evaluate ------------------------------------------------------------
    print(f"evaluating on held-out {held_out}, {N_EVAL} episodes each")
    t1 = time.time()
    tasks = [(str(run_dir_for(s, args.run_root)), held_out, anchors, N_EVAL)
             for s in RESERVED_SEEDS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        per_seed = dict(zip(RESERVED_SEEDS, pool.map(_evaluate_task, tasks)))
    eval_wall = time.time() - t1

    stats = summarize({s: r["primary_dv"] for s, r in per_seed.items()})
    print(f"\n{'seed':>5} {'DV':>8}  " + " ".join(f"{c:>7}" for c in held_out))
    for s, r in per_seed.items():
        print(f"{s:>5} {r['primary_dv']:>8.4f}  "
              + " ".join(f"{r['normalized'][str(c)]:>7.3f}" for c in held_out))
    print(f"\nmean {stats['mean']:.4f}  SD {stats['sd']:.4f}  (n = {stats['n_seeds']})")

    payload = {
        "rule": "D038 Stage-2 baseline",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git["commit"],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "env_id": ENV_ID,
        "seeds": list(RESERVED_SEEDS),
        "held_out": held_out,
        "n_eval": N_EVAL,
        "eval_seed_base": EVAL_SEED_BASE,
        "train_wall_seconds": round(train_wall, 1),
        "eval_wall_seconds": round(eval_wall, 1),
        "summary": stats,
        "per_seed": {str(s): r for s, r in per_seed.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "x", encoding="utf-8") as f:
        f.write(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
