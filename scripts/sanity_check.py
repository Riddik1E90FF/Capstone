"""Sanity check across the whole difficulty grid.

Runs a fixed scripted policy over every configuration and reports outcome rates
and damage taken.  This does not test learning; it tests that the difficulty
axes actually move something.  If damage taken is flat across the grid, the
axes are not doing their job and the environment needs revision *before* any
EEG work happens.

Usage:
    python scripts/sanity_check.py --episodes 60
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from combat_env import N_ACTIONS, N_CONFIGS, Action, DifficultyConfig, make_env


def reactive_policy(obs: np.ndarray, rng: np.random.Generator) -> int:
    """A crude hand-written baseline.

    Not a trained agent and not meant to be good.  It dodges when it believes an
    attack is landing, closes distance otherwise, and attacks in range.  Its
    only job is to be *sensitive to telegraph length* so the grid produces a
    gradient rather than uniform noise.
    """
    dist_norm = obs[2]
    stamina = obs[4]
    telegraph = obs[10]  # phase one-hot index 1 -> obs[9 + 1]
    active = obs[11]
    progress = obs[16]

    if (telegraph > 0.5 and progress > 0.65) or active > 0.5:
        if stamina > 0.3:
            return int(Action.DODGE)
        return int(Action.BLOCK)

    if dist_norm < 0.30:
        if stamina > 0.25:
            return int(Action.ATTACK)
        return int(Action.MOVE_S)

    # Close distance toward the enemy using the egocentric unit vector.
    ux, uy = obs[0], obs[1]
    angle = np.arctan2(ux, uy)  # 0 = north, clockwise
    idx = int(np.round(angle / (np.pi / 4))) % 8
    return int(Action.MOVE_N) + idx


def evaluate(config_id: int, episodes: int, seed0: int) -> dict:
    wins = losses = timeouts = 0
    dmg_taken = []
    dmg_dealt = []
    for e in range(episodes):
        env = make_env(config_id, seed=seed0 + e)
        obs, _ = env.reset(seed=seed0 + e)
        rng = np.random.default_rng(seed0 + e)
        while True:
            obs, _, term, trunc, info = env.step(reactive_policy(obs, rng))
            if term or trunc:
                break
        s = info["stats"]
        dmg_taken.append(s.damage_taken)
        dmg_dealt.append(s.damage_dealt)
        wins += s.outcome == "win"
        losses += s.outcome == "loss"
        timeouts += s.outcome == "timeout"
    return {
        "config_id": config_id,
        "win_rate": wins / episodes,
        "loss_rate": losses / episodes,
        "timeout_rate": timeouts / episodes,
        "damage_taken": float(np.mean(dmg_taken)),
        "damage_dealt": float(np.mean(dmg_dealt)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.perf_counter()
    rows = [evaluate(c, args.episodes, args.seed) for c in range(N_CONFIGS)]
    elapsed = time.perf_counter() - t0

    print(f"\n{'cfg':>4} {'tele':>6} {'gap':>6} {'win':>7} {'loss':>7} {'dmg_taken':>10}")
    print("-" * 46)
    for r in rows:
        d = DifficultyConfig.from_id(r["config_id"])
        print(f"{r['config_id']:>4} {d.telegraph_duration:>6.2f} {d.attack_gap:>6.2f} "
              f"{r['win_rate']:>7.2f} {r['loss_rate']:>7.2f} {r['damage_taken']:>10.1f}")

    easy = rows[0]["damage_taken"]
    hard = rows[-1]["damage_taken"]
    win_easy, win_hard = rows[0]["win_rate"], rows[-1]["win_rate"]

    print("\n--- gradient check ---")
    print(f"easiest cfg00: win {win_easy:.2f}, damage taken {easy:.1f}")
    print(f"hardest cfg24: win {win_hard:.2f}, damage taken {hard:.1f}")

    taken = np.array([r["damage_taken"] for r in rows])
    ranks = np.argsort(np.argsort(taken))
    ideal = np.arange(N_CONFIGS)
    rho = float(np.corrcoef(ranks, ideal)[0, 1])
    print(f"rank correlation between config_id and damage taken: {rho:+.3f}")
    print(f"\n{N_CONFIGS * args.episodes} episodes in {elapsed:.1f}s "
          f"({N_CONFIGS * args.episodes / elapsed:.0f} episodes/s)")


if __name__ == "__main__":
    main()
