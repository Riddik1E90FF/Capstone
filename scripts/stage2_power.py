"""Stage-2 seed count by power simulation (pre-registration §9, D041).

N is the smallest multiple of 10, at least 40, at which the registered test
has >= 80% power to detect a difference of 0.07 normalized-return units.
Everything that decides N is fixed here, as §9 fixes it, before the script is
run:

    test        one-sided permutation test on the difference of IQMs,
                p = (1 + #{permuted >= observed}) / (1 + n_perm)
    alpha       .05 / 6, the first Holm step
    data        each group resampled with replacement from the ten Stage-2
                baseline per-seed DVs; DELTA added to one group
    size        2,000 simulated experiments x 2,000 permutations
    rng         a fresh default_rng(0) for every candidate N, so each
                candidate's power does not depend on which candidates were
                computed before it
    candidates  40, 50, 60, ... in order; the first with power >= 0.80 is N

`power_at` follows Ryan's ten-step outline; the body was written by Claude
at Ryan's request (2026-11-16).  The search, loading, provenance guards and
output are Claude's.  Tested in tests/test_stage2_power.py.

The script refuses a working tree with uncommitted changes and never
overwrites `docs/stage2_power.json`.

Usage (from the repo root):
    python scripts/stage2_power.py
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.stats import trim_mean

from experiment.manifest import collect_git

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "docs" / "stage2_baseline.json"
OUT_PATH = REPO_ROOT / "docs" / "stage2_power.json"

DELTA: float = 0.07             # smallest effect of interest, §9
ALPHA: float = 0.05 / 6         # first Holm step, one-sided
TARGET_POWER: float = 0.80
N_SIMS: int = 2_000
N_PERM: int = 2_000
SIM_SEED: int = 0
N_FLOOR: int = 40
N_STEP: int = 10
N_MAX: int = 200                # search gives up here rather than run forever
SEED_BASE: int = 1000           # condition seeds are SEED_BASE + 1 .. SEED_BASE + N


def power_at(n: int, values: np.ndarray, delta: float, alpha: float,
             n_sims: int, n_perm: int, rng: np.random.Generator) -> float:
    """Estimated power of the registered test at `n` seeds per condition.

    Each of `n_sims` simulated experiments:
      * group A: `n` values drawn with replacement from `values`;
      * group B: `n` values drawn with replacement from `values`, plus `delta`;
      * statistic: IQM(B) - IQM(A), where IQM is the 25%-trimmed mean,
        exactly `scipy.stats.trim_mean(x, 0.25)`;
      * one-sided permutation test of that statistic over `n_perm` random
        relabellings of the pooled 2n values into two groups of n;
        p = (1 + #{permuted statistic >= observed}) / (1 + n_perm);
      * the experiment rejects if p <= alpha.

    Args:
        n: seeds per condition, >= 2.
        values: 1-D array of per-seed primary DVs to resample from.  Not
            modified.
        delta: true difference added to group B.  0 gives the type I error
            rate; a negative value must give power near 0 (one-sided).
        alpha: rejection threshold for p.
        n_sims, n_perm: positive ints.
        rng: the only source of randomness.  The same seed must give the same
            result.

    Returns:
        The fraction of the `n_sims` experiments that rejected: a float in
        [0, 1], an integer multiple of 1 / n_sims.

    Note: if alpha < 1 / (1 + n_perm) no p-value can reach it, so the power
    is exactly 0.  Budget: one call at the registered sizes (n = 40,
    2,000 x 2,000) should finish in minutes, not hours.
    """
    
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"values must be a non-empty 1-D array, got shape {values.shape}")
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    if n_sims < 1 or n_perm < 1:
        raise ValueError(f"n_sims and n_perm must be positive, got {n_sims}, {n_perm}")

    # 1. Check whether rejection is even possible
    if alpha < 1.0 / (1 + n_perm):
        return 0.0

    rejections = 0
    base = np.tile(np.arange(2 * n), (n_perm, 1))     # one row per permutation

    for sim in range(n_sims):

        # 2. Generate independent groups
        A = rng.choice(values, size=n, replace=True)
        B = rng.choice(values, size=n, replace=True) + delta

        # 3. Calculate observed IQM difference
        observed = trim_mean(B, 0.25) - trim_mean(A, 0.25)

        # 4. Pool the groups
        pooled = np.concatenate([A, B])

        # 5. Generate permutation indices (vectorized)
        order = rng.permuted(base, axis=1)

        # 6. Apply permutations
        permuted = pooled[order]                       # (n_perm, 2n)

        # 7. Calculate permutation statistics
        permuted_stats = (trim_mean(permuted[:, n:], 0.25, axis=1)
                          - trim_mean(permuted[:, :n], 0.25, axis=1))

        # 8. Calculate p-value
        p = (1 + np.count_nonzero(permuted_stats >= observed)) / (1 + n_perm)

        # 9. Record rejection
        if p <= alpha:
            rejections += 1

    # 10. Calculate estimated power
    return rejections / n_sims



# -- harness (Claude) ---------------------------------------------------------

def load_values(path: Path = BASELINE_PATH) -> np.ndarray:
    """The ten Stage-2 per-seed DVs, in seed order."""
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    per_seed = rec["per_seed"]
    return np.array([per_seed[s]["primary_dv"] for s in sorted(per_seed, key=int)],
                    dtype=float)


PowerFn = Callable[[int, np.random.Generator], float]


def find_n(power_fn: PowerFn, *, floor: int = N_FLOOR, step: int = N_STEP,
           n_max: int = N_MAX, target: float = TARGET_POWER, seed: int = SIM_SEED,
           progress: Callable[[int, float], None] | None = None
           ) -> tuple[int, dict[int, float]]:
    """Smallest candidate n (floor, floor + step, ...) with power >= target.

    `power_fn(n, rng)` gets a fresh `default_rng(seed)` for every candidate.
    Returns (n, {candidate: power}) for every candidate computed.  Raises if
    no candidate up to `n_max` reaches the target.
    """
    if floor % step:
        raise ValueError("the floor must itself be a multiple of the step")
    table: dict[int, float] = {}
    for n in range(floor, n_max + 1, step):
        table[n] = float(power_fn(n, np.random.default_rng(seed)))
        if progress is not None:
            progress(n, table[n])
        if table[n] >= target:
            return n, table
    raise RuntimeError(f"no n up to {n_max} reaches power {target}: {table}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args(argv)

    if args.out.exists():
        raise SystemExit(f"{args.out} already exists; N is determined once (D041).")
    git = collect_git()
    if git["commit"] is None:
        raise SystemExit("git unavailable; the result must record the commit that produced it.")
    if git["dirty"]:
        raise SystemExit("uncommitted changes in the working tree; commit first (D041):\n  "
                         + "\n  ".join(git["dirty_paths"]))

    values = load_values()
    print(f"values (n = {len(values)}): {np.round(values, 4).tolist()}")
    print(f"delta {DELTA}, alpha {ALPHA:.5f}, {N_SIMS} sims x {N_PERM} permutations, "
          f"target power {TARGET_POWER}")

    def report(n: int, power: float) -> None:
        print(f"  n = {n:>3}: power {power:.4f}", flush=True)

    t0 = time.time()
    n, table = find_n(lambda k, rng: power_at(k, values, DELTA, ALPHA, N_SIMS, N_PERM, rng),
                      progress=report)
    wall = time.time() - t0
    print(f"\nN = {n} seeds per condition: seeds {SEED_BASE + 1}-{SEED_BASE + n}, "
          f"{7 * n} runs")

    payload = {
        "rule": "pre-registration §9, D041",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git["commit"],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "values_source": str(BASELINE_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "values": values.tolist(),
        "delta": DELTA, "alpha": ALPHA, "target_power": TARGET_POWER,
        "n_sims": N_SIMS, "n_perm": N_PERM, "sim_seed": SIM_SEED,
        "rng": "fresh default_rng(sim_seed) per candidate",
        "floor": N_FLOOR, "step": N_STEP,
        "power": {str(k): v for k, v in table.items()},
        "n_per_condition": n,
        "seeds": [SEED_BASE + 1, SEED_BASE + n],
        "wall_seconds": round(wall, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "x", encoding="utf-8") as f:
        f.write(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
