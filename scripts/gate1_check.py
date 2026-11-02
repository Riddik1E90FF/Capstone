"""Decide Gate 1 (D034) for the trained baseline agents.

Reads the ten `combat:mix` runs, evaluates each seed's final network on all 25
configurations with the protocol in `evaluation/gate1.py`, and applies the
three criteria fixed in D034 before any of these runs existed:

  G1-a  non-trivial   mean damage taken <= 0.75 x the random floor's mean
  G1-b  reproducible  at least k = 8 of 10 seeds satisfy G1-a
  G1-c  ordered       per-configuration spread exceeds the 95th percentile of
                      a configuration-label permutation null

The floors come from `docs/gate1_floors.json` and are never recomputed here.
If that file is missing, stop: the thresholds must predate the runs.

Usage (from the repo root):
    python scripts/gate1_check.py --runs runs/gate1/mix
"""

from __future__ import annotations

# Make this script runnable from any directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import re
from pathlib import Path

import numpy as np

from combat_env.config import DifficultyConfig
from dqn.checkpoint import latest_checkpoint, load_checkpoint
from dqn.exploration import greedy_action
from evaluation.gate1 import CONFIG_IDS, N_EVAL, evaluate_policy

K_SEEDS = 8          # D034 G1-b
G1A_RATIO = 0.75     # D034 G1-a
N_PERMUTATIONS = 10_000
PERMUTATION_SEED = 0
SATURATION_FLAG = 0.25   # D035


def greedy_policy(network):
    """Wrap a trained network as a Policy: the same shape a floor policy has.

    `evaluate_policy` cannot tell this from `random_policy`, which is what
    makes the agent and the floors comparable at all.
    """
    def act(obs: np.ndarray) -> int:
        q, _ = network.forward(np.asarray(obs)[None])
        return int(greedy_action(q[0]))
    return act


def find_runs(root: Path) -> dict[int, Path]:
    """{seed: run_dir} for every completed run under `root`."""
    runs: dict[int, Path] = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        m = re.search(r"__seed(\d+)__", d.name)
        if not m:
            continue
        summary = d / "summary.json"
        if summary.exists():
            status = json.loads(summary.read_text()).get("status")
            if status != "completed":
                print(f"  skipping {d.name}: status={status!r}")
                continue
        runs[int(m.group(1))] = d
    return runs


def spread_permutation_null(outcomes, n_permutations=N_PERMUTATIONS,
                            seed=PERMUTATION_SEED):
    """G1-c null distribution.  **Not yet implemented — this is yours.**

    Specification, fixed by D034 and not to be adjusted after seeing the
    observed value:

      `outcomes` is {config_id: [damage_taken, ...]}, one list of N_EVAL
      episode values per configuration, pooled across the seeds that passed
      G1-a.

      One permutation:
        1. Pool every episode value across all configurations into one array.
        2. Shuffle it with the run's Generator.
        3. Re-split it into 25 groups of the original sizes.
        4. Record max(group means) - min(group means).

      Return the array of `n_permutations` such spreads, drawn from
      `np.random.default_rng(seed)`.

    The null this builds is "configuration labels carry no information": any
    spread it produces is what label noise alone can manufacture.  The
    observed spread passes if it exceeds the 95th percentile.

    Shuffle the pooled *episode* values, not the per-configuration means.
    Permuting means would hold each configuration's internal variance fixed
    and give a null that is far too narrow, so almost any spread would clear
    it and G1-c would pass regardless of whether the ordering is real.
    """
    raise NotImplementedError(
        "G1-c permutation null is not implemented. See the docstring for the "
        "specification fixed in D034.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs/gate1/mix"))
    ap.add_argument("--floors", type=Path, default=Path("docs/gate1_floors.json"))
    ap.add_argument("--out", type=Path, default=Path("docs/gate1_result.json"))
    ap.add_argument("--n-eval", type=int, default=N_EVAL)
    ap.add_argument("--skip-g1c", action="store_true",
                    help="report G1-a and G1-b only, while G1-c is unimplemented")
    args = ap.parse_args()

    if not args.floors.exists():
        raise SystemExit(
            f"{args.floors} not found. The G1-a threshold must be fixed before "
            f"these runs existed (D034). Do not compute it from them.")
    floors = json.loads(args.floors.read_text())
    f_random = floors["floors"]["random"]["mean"]
    threshold = G1A_RATIO * f_random
    print(f"Floors from {args.floors} ({floors['created_utc'][:10]}, "
          f"commit {floors['git_commit'][:7]})")
    print(f"G1-a threshold: mean damage taken <= {G1A_RATIO} x {f_random:.4f} "
          f"= {threshold:.4f}\n")

    runs = find_runs(args.runs)
    if not runs:
        raise SystemExit(f"no completed runs under {args.runs}")

    per_seed = {}
    print(f"{'seed':>5} {'mean damage':>12} {'G1-a':>6} {'spread':>8} "
          f"{'win%':>7} {'flagged cfgs':>13}")
    print("-" * 58)
    for seed in sorted(runs):
        ckpt = latest_checkpoint(runs[seed] / "checkpoints")
        if ckpt is None:
            print(f"{seed:>5}  no checkpoint, skipped")
            continue
        network, _ = load_checkpoint(ckpt)
        result = evaluate_policy(greedy_policy(network), n_eval=args.n_eval)
        flagged = [c for c in CONFIG_IDS
                   if result["saturated"][c] > SATURATION_FLAG]
        passed = result["mean"] <= threshold
        per_seed[seed] = {"result": result, "passed": bool(passed),
                          "flagged": flagged}
        win = np.mean(list(result["success_rate"].values()))
        print(f"{seed:>5} {result['mean']:>12.3f} {'PASS' if passed else 'FAIL':>6} "
              f"{result['spread']:>8.3f} {win:>7.3f} "
              f"{','.join(map(str, flagged)) or '-':>13}")

    n_pass = sum(v["passed"] for v in per_seed.values())
    g1b = n_pass >= K_SEEDS
    print(f"\nG1-a   {n_pass}/{len(per_seed)} seeds under the threshold")
    print(f"G1-b   {'PASS' if g1b else 'FAIL'}   need k = {K_SEEDS}")

    if args.skip_g1c:
        print("G1-c   SKIPPED (--skip-g1c); the gate is not decided")
        verdict = "incomplete"
    else:
        passing = [s for s, v in per_seed.items() if v["passed"]]
        pooled = {c: [d for s in passing
                      for d in [st.damage_taken
                                for st in per_seed[s]["result"]["outcomes"][c]]]
                  for c in CONFIG_IDS}
        means = np.array([np.mean(pooled[c]) for c in CONFIG_IDS])
        observed = float(means.max() - means.min())
        null = spread_permutation_null(pooled)
        crit = float(np.percentile(null, 95))
        g1c = observed > crit
        print(f"G1-c   {'PASS' if g1c else 'FAIL'}   observed spread "
              f"{observed:.3f} vs 95th percentile {crit:.3f}")
        verdict = "PASS" if (g1b and g1c) else "FAIL"

    print(f"\nGate 1: {verdict}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "threshold": threshold,
        "floors_file": str(args.floors),
        "n_eval": args.n_eval,
        "per_seed": {str(s): {"mean_damage": v["result"]["mean"],
                              "spread": v["result"]["spread"],
                              "passed": v["passed"],
                              "flagged_configs": v["flagged"],
                              "damage_taken": {str(c): v["result"]["damage_taken"][c]
                                               for c in CONFIG_IDS}}
                     for s, v in per_seed.items()},
        "g1a_passing_seeds": n_pass,
        "g1b": bool(g1b),
        "verdict": verdict,
    }, indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
