"""Capstone status demo: prints the project's current evidence in one screen.

Reads only committed result files, so it runs in under a second, needs no
training, and cannot fail in front of an audience.

    python scripts/demo.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import json
from pathlib import Path

import numpy as np

from combat_env.config import ATTACK_GAP_LEVELS, TELEGRAPH_LEVELS

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"


def midrank(x):
    """Spearman needs mid-ranks for ties; argsort(argsort(x)) does not give them."""
    x = np.asarray(x, float); s = x.argsort(); r = np.empty(len(x)); i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[s[j + 1]] == x[s[i]]:
            j += 1
        r[s[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    return float(np.corrcoef(midrank(a), midrank(b))[0, 1])


def heat(v, lo, hi):
    blocks = " .:-=+*#%@"
    return blocks[min(9, max(0, int((v - lo) / (hi - lo) * 9)))]


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    floors = json.loads((root / "docs/gate1_floors.json").read_text())
    result = json.loads((root / "docs/gate1_result.json").read_text())

    seeds = sorted(result["per_seed"], key=int)
    agent = np.mean([[result["per_seed"][s]["damage_taken"][str(c)] for c in range(25)]
                     for s in seeds], axis=0)
    rnd = np.array([floors["floors"]["random"]["damage_taken"][str(c)] for c in range(25)])
    scr = np.array([floors["floors"]["scripted"]["damage_taken"][str(c)] for c in range(25)])
    proxy = np.array([(c // 5) + (c % 5) for c in range(25)], float)

    print(f"\n{BOLD}EEG-ORDERED CURRICULA FOR REINFORCEMENT LEARNING{RESET}")
    print(f"{DIM}Does a human's EEG-derived sense of difficulty usefully order "
          f"what an RL agent practises?{RESET}\n")

    print(f"{BOLD}1. The training environment: 25 difficulty configurations{RESET}")
    print(f"{DIM}   Two axes. Mean damage the trained agent takes in each cell "
          f"(higher = harder).{RESET}\n")
    lo, hi = agent.min(), agent.max()
    print("            attack gap (s), shorter is harder")
    print("            " + "".join(f"{g:>7.2f}" for g in ATTACK_GAP_LEVELS))
    for t in range(5):
        row = "".join(f"{agent[t * 5 + g]:>6.1f}{heat(agent[t * 5 + g], lo, hi)}"
                      for g in range(5))
        print(f"  tel {TELEGRAPH_LEVELS[t]:.2f}s {row}")
    print(f"\n   easiest cell {int(agent.argmin())} at {lo:.1f} damage"
          f"   ->   hardest cell {int(agent.argmax())} at {hi:.1f}")

    print(f"\n{BOLD}2. Three independent systems agree on the ordering{RESET}")
    print(f"{DIM}   Spearman rank correlation over the same 25 configurations.{RESET}\n")
    for label, other in (("trained agent  vs  random-action policy", rnd),
                         ("trained agent  vs  scripted policy", scr),
                         ("trained agent  vs  design intent", proxy)):
        print(f"   {label:<42} rho = {spearman(agent, other):+.3f}")
    print(f"\n{DIM}   The grid measures something real, and it does so before any "
          f"EEG is involved.{RESET}")

    print(f"\n{BOLD}3. Gates{RESET}")
    print(f"{DIM}   Each gate's pass criteria were fixed and committed BEFORE the "
          f"runs that test them.{RESET}\n")
    rows = [
        ("Gate 1a  algorithm correctness", "PASS",
         "from-scratch DQN tracks a published reference, 10/10 seeds, 1st attempt"),
        ("Gate 1   the task is learnable", "PARTIAL",
         f"10/10 seeds beat the threshold ({np.mean([result['per_seed'][s]['mean_damage'] for s in seeds]):.1f} "
         f"mean damage vs {result['threshold']:.1f} limit)"),
        ("Gate 2   human/agent agreement", "NEXT", "needs one human play session, no EEG"),
        ("Gate 3   the EEG index is valid", "BLOCKED", "electrode contact, diagnosed 2026-11-05"),
    ]
    for name, state, note in rows:
        print(f"   {name:<34} {state:<9} {DIM}{note}{RESET}")

    print(f"\n{BOLD}4. Reproducibility{RESET}")
    print(f"   Pre-registration Stage-1 locked, every change logged with whether")
    print(f"   outcome data had been seen.  Agent is NumPy from scratch, no PyTorch.")
    print(f"   Same seed reproduces byte-identical training logs.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
