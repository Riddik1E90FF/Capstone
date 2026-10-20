"""Compare candidate difficulty-axis settings (OPEN-02).

The problem: with the current settings the two axes are unbalanced. Attack gap
moves agent damage taken by ~43.5; telegraph duration moves it by ~17.3 and is
flat at the sparsest gap. Gate 2 needs both axes to carry difficulty for the
agent, or the human/agent ranking correlation is dragged down by a dead axis.

This sweeps candidate settings and reports, for each:
  - the main effect of each axis on damage taken
  - the main effect of each axis on HIT RATE (times_hit / attacks_faced)

Hit rate matters because damage taken conflates two different things:
  damage ~= (number of attacks faced) x (fraction that connect)
Attack gap controls the first term mechanically. Telegraph only affects the
second. So damage taken structurally favours the gap axis, and hit rate is the
cleaner read on whether telegraph is doing anything.
"""

from __future__ import annotations

# Make this script runnable from any directory: put the repo root (this
# file's parent's parent) on sys.path so `combat_env` and `dqn` import
# without PYTHONPATH.  Resolved from __file__, so a git worktree imports
# its own copy rather than another checkout's.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import numpy as np

import combat_env.config as C
from combat_env import Action, EnvParams, make_env
from combat_env.env import EnemyPhase


def policy(obs: np.ndarray) -> int:
    dist, stamina = obs[2], obs[4]
    telegraph, active, progress = obs[10], obs[11], obs[16]
    if (telegraph > 0.5 and progress > 0.65) or active > 0.5:
        return int(Action.DODGE) if stamina > 0.3 else int(Action.BLOCK)
    if dist < 0.30:
        return int(Action.ATTACK) if stamina > 0.25 else int(Action.MOVE_S)
    angle = np.arctan2(obs[0], obs[1])
    return int(Action.MOVE_N) + int(np.round(angle / (np.pi / 4))) % 8


def cell(cid: int, episodes: int, params: EnvParams) -> tuple[float, float]:
    dmg, hits, faced = [], 0, 0
    for e in range(episodes):
        env = make_env(cid, seed=e, params=params)
        obs, _ = env.reset(seed=e)
        while True:
            obs, _, term, trunc, info = env.step(policy(obs))
            if term or trunc:
                break
        s = info["stats"]
        dmg.append(s.damage_taken)
        hits += s.times_hit
        faced += s.attacks_faced
    return float(np.mean(dmg)), (hits / faced if faced else 0.0)


def evaluate(name: str, telegraph_levels, iframes: float, episodes: int = 30):
    C.TELEGRAPH_LEVELS = tuple(telegraph_levels)
    params = EnvParams(dodge_iframes=iframes)
    dmg = np.zeros((5, 5))
    hit = np.zeros((5, 5))
    for t in range(5):
        for g in range(5):
            d, h = cell(t * 5 + g, episodes, params)
            dmg[t, g], hit[t, g] = d, h

    out = {
        "name": name,
        "dmg_tele": dmg[4].mean() - dmg[0].mean(),
        "dmg_gap": dmg[:, 4].mean() - dmg[:, 0].mean(),
        "hit_tele": hit[4].mean() - hit[0].mean(),
        "hit_gap": hit[:, 4].mean() - hit[:, 0].mean(),
        "dmg": dmg,
    }
    out["balance"] = out["dmg_tele"] / out["dmg_gap"] if out["dmg_gap"] else 0.0
    return out


CANDIDATES = [
    ("current          ", (0.70, 0.55, 0.42, 0.32, 0.24), 0.20),
    ("wider telegraph  ", (0.90, 0.62, 0.42, 0.26, 0.14), 0.20),
    ("wider + tight if ", (0.90, 0.62, 0.42, 0.26, 0.14), 0.12),
    ("widest + tight if", (1.00, 0.66, 0.42, 0.24, 0.12), 0.10),
]

if __name__ == "__main__":
    original = C.TELEGRAPH_LEVELS
    print(f"{'candidate':<19} {'dmg:tele':>9} {'dmg:gap':>9} {'ratio':>7} "
          f"{'hit:tele':>9} {'hit:gap':>9}")
    print("-" * 68)
    results = []
    for name, levels, iframes in CANDIDATES:
        r = evaluate(name, levels, iframes)
        results.append(r)
        print(f"{name:<19} {r['dmg_tele']:>9.1f} {r['dmg_gap']:>9.1f} "
              f"{r['balance']:>7.2f} {r['hit_tele']:>9.3f} {r['hit_gap']:>9.3f}")
    C.TELEGRAPH_LEVELS = original

    print("\nratio = telegraph effect / gap effect. 1.00 would be perfectly balanced.")
    best = max(results, key=lambda r: min(r["balance"], 1 / r["balance"]) if r["balance"] else 0)
    print(f"most balanced: {best['name'].strip()}")
    print("\ndamage-taken grid for that candidate (rows = telegraph, cols = gap):")
    print("        gap0   gap1   gap2   gap3   gap4")
    for i, row in enumerate(best["dmg"]):
        print(f"tele{i}  " + "  ".join(f"{v:5.1f}" for v in row))
