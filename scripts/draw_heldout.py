"""Draw the held-out configuration set (pre-registration §5.2, D038).

Run this once, after the rule and HELDOUT_SEED are committed.  It refuses a
working tree with uncommitted changes (so the recorded commit is the code
that drew) and refuses to overwrite an existing split.

Rule:
  1. A transversal of the 5 x 5 grid: one configuration per telegraph level
     and one per gap level.  config_id = telegraph_level * 5 + gap_level.
  2. Transversals containing configuration 0 or 24 are excluded (78 of 120
     remain).
  3. The 78 are enumerated in lexicographic order of the gap-level
     permutation, and one index is drawn with
     default_rng(HELDOUT_SEED).integers(78).

Usage (from the repo root):
    python scripts/draw_heldout.py
"""

from __future__ import annotations

# Make this script runnable from any directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiment.manifest import collect_git

HELDOUT_SEED = 20261112          # D038; committed before the draw
N_LEVELS = 5
EXCLUDED = (0, 24)               # ceiling and saturation corners (D038)
DEFAULT_OUT = Path("docs/heldout_split.json")


def transversals() -> list[tuple[int, ...]]:
    """Every transversal as config ids ordered by telegraph level, in
    lexicographic order of the gap-level permutation."""
    return [tuple(t * N_LEVELS + gap for t, gap in enumerate(perm))
            for perm in itertools.permutations(range(N_LEVELS))]


def eligible() -> list[tuple[int, ...]]:
    return [t for t in transversals() if not set(t) & set(EXCLUDED)]


def draw(seed: int) -> tuple[int, tuple[int, ...]]:
    """(index into eligible(), the held-out config ids)."""
    candidates = eligible()
    index = int(np.random.default_rng(seed).integers(len(candidates)))
    return index, candidates[index]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    if args.out.exists():
        raise SystemExit(f"{args.out} already exists; the split is drawn once (D038).")
    git = collect_git()
    if git["commit"] is None:
        raise SystemExit("git unavailable; the draw must record the commit it ran from.")
    if git["dirty"]:
        raise SystemExit(
            "uncommitted changes in the working tree; commit the rule and seed "
            "before drawing (D038):\n  " + "\n  ".join(git["dirty_paths"]))

    index, held_out = draw(HELDOUT_SEED)
    training = [c for c in range(N_LEVELS * N_LEVELS) if c not in held_out]
    record = {
        "rule": "D038: random transversal, configurations 0 and 24 excluded",
        "seed": HELDOUT_SEED,
        "n_eligible": len(eligible()),
        "index": index,
        "held_out": sorted(held_out),
        "training": training,
        "git_commit": git["commit"],
        "drawn_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "x", encoding="utf-8") as f:
        f.write(json.dumps(record, indent=2) + "\n")
    print(f"Held out: {record['held_out']}  (index {index} of {record['n_eligible']})")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
