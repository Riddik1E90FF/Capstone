"""Measure the Gate 1 floors (D034), before any combat training run.

Two floors, both on all 25 configurations with the protocol fixed in
``evaluation/gate1.py``:

  random    uniform random action selection.  This is the one G1-a compares
            against: a seed passes only if its mean damage taken is at most
            0.75x this floor's mean (D034, metric per D035).
  scripted  the hand-written policy from ``scripts/axis_sweep.py``, the same
            one used for the difficulty-axis calibration in D026.  Not a pass
            criterion; it is context for reading the agent's number, and a
            check that the configurations are winnable at all.

Why floors come first
---------------------
D034 fixes the thresholds before outcome data exists, exactly as D029 did for
Gate 1a.  If the floors were measured after training, the margin could be
chosen to make the agent pass.  Run this once, append the numbers to D034 with
their date, and do not re-run it to "check" after training.

A floor is informative in both directions.  If random action selection already
does well on a configuration, that configuration cannot discriminate anything
and the axis calibration needs revisiting before Gate 2 rather than after.  It
was exactly this measurement that rejected success rate as the ranking
quantity (D035).

Usage (from the repo root):
    python scripts/gate1_floors.py
    python scripts/gate1_floors.py --n-eval 20        # quick smoke check
"""

from __future__ import annotations

# Make this script runnable from any directory: put the repo root (this
# file's parent's parent) on sys.path so `combat_env` and `experiment` import
# without PYTHONPATH.  Resolved from __file__, so a git worktree imports its
# own copy rather than another checkout's.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from combat_env import N_ACTIONS
from combat_env.config import DifficultyConfig
from evaluation.gate1 import (
    CONFIG_IDS, EVAL_SEED_BASE, N_EVAL, evaluate_policy, random_policy,
)

# The scripted policy lives beside this file in scripts/.  Python puts a
# script's own directory on sys.path, so a sibling import works; importing the
# module is safe because axis_sweep's global mutation of TELEGRAPH_LEVELS
# happens inside evaluate()/__main__, never at import time.
from axis_sweep import policy as scripted_policy

RANDOM_POLICY_SEED = 0


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def report(name: str, result: dict) -> None:
    print(f"\n{'=' * 66}")
    print(f"FLOOR: {name}")
    print("=" * 66)
    print(f"\n{'cfg':>4} {'telegraph':>10} {'gap':>7} {'damage':>8} {'win%':>7} "
          f"{'sat%':>6}   (n={result['n_eval']})")
    print("-" * 56)
    for cid in CONFIG_IDS:
        d = DifficultyConfig.from_id(cid)
        print(f"{cid:>4} {d.telegraph_duration:>10.2f} {d.attack_gap:>7.2f} "
              f"{result['damage_taken'][cid]:>8.2f} "
              f"{result['success_rate'][cid]:>7.3f} "
              f"{result['saturated'][cid]:>6.2f}")
    print(f"\n  mean damage taken across 25 configs : {result['mean']:.4f}")
    print(f"  spread (max - min)                   : {result['spread']:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=N_EVAL,
                    help=f"episodes per configuration (D034 fixes this at {N_EVAL})")
    ap.add_argument("--out", type=Path, default=Path("docs/gate1_floors.json"))
    args = ap.parse_args()

    if args.n_eval != N_EVAL:
        print(f"WARNING: n_eval={args.n_eval} differs from the D034 value of "
              f"{N_EVAL}. This is a smoke check, not the recorded floor.\n")

    if args.out.exists():
        raise SystemExit(
            f"{args.out} already exists. The floors are recorded once, before "
            f"training (D034). Delete it deliberately if you mean to redo them.")

    t0 = time.time()
    results = {}

    def tick(done: int, total: int) -> None:
        print(f"\r  config {done}/{total}", end="", flush=True)

    print(f"Measuring Gate 1 floors: {len(CONFIG_IDS)} configs x {args.n_eval} "
          f"episodes x 2 policies")
    print(f"Evaluation seeds: {EVAL_SEED_BASE} .. {EVAL_SEED_BASE + args.n_eval - 1}"
          f"  (disjoint from training seeds 1-10)\n")

    print("random policy")
    rng = np.random.default_rng(RANDOM_POLICY_SEED)
    results["random"] = evaluate_policy(
        random_policy(rng, N_ACTIONS), n_eval=args.n_eval, progress=tick)

    print("\nscripted policy")
    results["scripted"] = evaluate_policy(
        scripted_policy, n_eval=args.n_eval, progress=tick)

    for name in ("random", "scripted"):
        report(name, results[name])

    wall = time.time() - t0
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rule": "D034",
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "n_eval": args.n_eval,
        "eval_seed_base": EVAL_SEED_BASE,
        "random_policy_seed": RANDOM_POLICY_SEED,
        "wall_seconds": round(wall, 1),
        "floors": {
            name: {
                "damage_taken": {str(k): v for k, v in r["damage_taken"].items()},
                "success_rate": {str(k): v for k, v in r["success_rate"].items()},
                "saturated": {str(k): v for k, v in r["saturated"].items()},
                "mean": r["mean"],
                "spread": r["spread"],
            }
            for name, r in results.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

    f_random = results["random"]["mean"]
    print(f"\n{'=' * 66}")
    print(f"Wrote {args.out}  ({wall / 60:.1f} min)")
    print("=" * 66)
    print("\nG1-a threshold, now fixed:")
    print(f"  a seed passes if its mean damage taken <= "
          f"0.75 x {f_random:.4f} = {0.75 * f_random:.4f}")
    print("\nAppend the floors to D034 with today's date before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
