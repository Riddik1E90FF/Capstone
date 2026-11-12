"""Measure the D039 return anchors, once, before any Stage-2 or condition run.

The random-action and scripted policies are evaluated on all 25
configurations with the Gate 1 protocol (`evaluation/gate1.py`: N_EVAL = 200,
the same evaluation seeds), and their mean episode returns are written to
`docs/return_anchors.json`.  The normalized return
(`evaluation/heldout.py`) is anchored on these: 0 = random, 1 = scripted.

The policies are the Gate 1 floor policies, built the same way and visited in
the same configuration order.  So the episodes are the floor episodes, and
mean damage taken must reproduce `docs/gate1_floors.json` exactly.  The
script checks that and refuses to write if it does not.

It refuses a working tree with uncommitted changes, so the recorded commit is
the code that measured, and it refuses to overwrite an existing file.

Usage (from the repo root):
    python scripts/measure_return_anchors.py
"""

from __future__ import annotations

# Make this script runnable from any directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from combat_env import N_ACTIONS
from evaluation.gate1 import CONFIG_IDS, EVAL_SEED_BASE, N_EVAL, random_policy
from evaluation.heldout import (
    ANCHORS_PATH, FALLBACK_FRACTION, SPLIT_PATH, denominators, evaluate_returns,
    load_held_out,
)
from experiment.manifest import collect_git

# Same import and seed as scripts/gate1_floors.py.
from axis_sweep import policy as scripted_policy
from gate1_floors import RANDOM_POLICY_SEED

FLOORS_PATH = Path("docs/gate1_floors.json")


def check_against_floors(name: str, damage: dict[int, float], floors: dict) -> list[str]:
    """Configurations whose mean damage taken differs from the recorded floor."""
    recorded = floors["floors"][name]["damage_taken"]
    return [f"{name} cfg {c}: {damage[c]!r} vs recorded {recorded[str(c)]!r}"
            for c in CONFIG_IDS if damage[c] != recorded[str(c)]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ANCHORS_PATH)
    args = ap.parse_args(argv)

    if args.out.exists():
        raise SystemExit(f"{args.out} already exists; the anchors are measured once (D039).")
    git = collect_git()
    if git["commit"] is None:
        raise SystemExit("git unavailable; the anchors must record the commit that measured them.")
    if git["dirty"]:
        raise SystemExit("uncommitted changes in the working tree; commit first (D039):\n  "
                         + "\n  ".join(git["dirty_paths"]))
    floors = json.loads(FLOORS_PATH.read_text(encoding="utf-8"))
    if floors["n_eval"] != N_EVAL or floors["eval_seed_base"] != EVAL_SEED_BASE \
            or floors["random_policy_seed"] != RANDOM_POLICY_SEED:
        raise SystemExit(f"{FLOORS_PATH} was measured under a different protocol")
    held_out = load_held_out(SPLIT_PATH)

    def tick(done: int, total: int) -> None:
        print(f"\r  config {done}/{total}", end="", flush=True)

    t0 = time.time()
    print("random policy")
    rng = np.random.default_rng(RANDOM_POLICY_SEED)
    results = {"random": evaluate_returns(random_policy(rng, N_ACTIONS), progress=tick)}
    print("\nscripted policy")
    results["scripted"] = evaluate_returns(scripted_policy, progress=tick)
    print()

    mismatches = (check_against_floors("random", results["random"]["damage_taken"], floors)
                  + check_against_floors("scripted", results["scripted"]["damage_taken"], floors))
    if mismatches:
        raise SystemExit("episodes do not reproduce the Gate 1 floors; nothing written:\n  "
                         + "\n  ".join(mismatches))

    random_r = results["random"]["returns"]
    scripted_r = results["scripted"]["returns"]
    denom, replaced = denominators(random_r, scripted_r)
    median = float(np.median([scripted_r[c] - random_r[c] for c in CONFIG_IDS]))

    print(f"\n{'cfg':>4} {'random R':>10} {'scripted R':>11} {'gap':>9}")
    print("-" * 38)
    for c in CONFIG_IDS:
        mark = "  held out" if c in held_out else ""
        mark += "  FALLBACK" if c in replaced else ""
        print(f"{c:>4} {random_r[c]:>10.3f} {scripted_r[c]:>11.3f} "
              f"{scripted_r[c] - random_r[c]:>9.3f}{mark}")
    print(f"\nmedian gap {median:.4f}; fallback threshold "
          f"{FALLBACK_FRACTION} x median = {FALLBACK_FRACTION * median:.4f}")
    print(f"held out {held_out}; fallback applied to held-out configs: "
          f"{sorted(set(held_out) & replaced) or 'none'}")
    print("Gate 1 floor damage reproduced exactly for both policies.")

    payload = {
        "rule": "D039",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git["commit"],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "n_eval": N_EVAL,
        "eval_seed_base": EVAL_SEED_BASE,
        "random_policy_seed": RANDOM_POLICY_SEED,
        "wall_seconds": round(time.time() - t0, 1),
        "floors_damage_reproduced": True,
        "held_out": held_out,
        "median_gap": median,
        "fallback_fraction": FALLBACK_FRACTION,
        "fallback_configs": sorted(replaced),
        "random": {"returns": {str(c): random_r[c] for c in CONFIG_IDS}},
        "scripted": {"returns": {str(c): scripted_r[c] for c in CONFIG_IDS}},
        "denominators": {str(c): denom[c] for c in CONFIG_IDS},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "x", encoding="utf-8") as f:
        f.write(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
