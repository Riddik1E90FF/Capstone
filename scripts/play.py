"""Human play interface.

This is the *same simulation* the agent trains on, rendered and stepped at real
time.  That identity is not a convenience; if the human and the agent play
different games, the per-configuration difficulty rankings compared at Gate 2
are not comparable and the whole EEG half loses its footing.

Requires pygame:  pip install pygame

Controls
--------
    WASD / arrows   move
    J or left-click attack
    K or space      dodge (in the direction you last moved)
    L or shift      block (hold)
    Esc             quit

Usage
-----
    python scripts/play.py --config 12
    python scripts/play.py --session --out data/session_01.jsonl

`--session` runs the full recording protocol: every configuration, interleaved
in a seeded random order, with repeated probe configurations at the start,
middle and end so within-session drift can be measured (see the familiarity
bias discussion in the Technical Appendix).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from combat_env import N_CONFIGS, Action, DifficultyConfig, make_env
from combat_env.config import TICK_HZ
from combat_env.env import EnemyPhase

SCALE = 55  # pixels per world unit
PROBE_CONFIGS = (6, 12, 18)  # replayed at start, middle and end of a session


def _require_pygame():
    try:
        import pygame  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "pygame is required for human play.\n"
            "    pip install pygame"
        ) from exc
    import pygame
    return pygame


def read_action(pygame, keys, mouse) -> int:
    if keys[pygame.K_k] or keys[pygame.K_SPACE]:
        return int(Action.DODGE)
    if keys[pygame.K_j] or mouse[0]:
        return int(Action.ATTACK)
    if keys[pygame.K_l] or keys[pygame.K_LSHIFT]:
        return int(Action.BLOCK)

    up = keys[pygame.K_w] or keys[pygame.K_UP]
    down = keys[pygame.K_s] or keys[pygame.K_DOWN]
    left = keys[pygame.K_a] or keys[pygame.K_LEFT]
    right = keys[pygame.K_d] or keys[pygame.K_RIGHT]

    table = {
        (True, False, False, False): Action.MOVE_N,
        (True, False, False, True): Action.MOVE_NE,
        (False, False, False, True): Action.MOVE_E,
        (False, True, False, True): Action.MOVE_SE,
        (False, True, False, False): Action.MOVE_S,
        (False, True, True, False): Action.MOVE_SW,
        (False, False, True, False): Action.MOVE_W,
        (True, False, True, False): Action.MOVE_NW,
    }
    return int(table.get((up, down, left, right), Action.IDLE))


def draw(pygame, screen, env, font, label: str) -> None:
    w, h = screen.get_size()
    cx, cy = w // 2, h // 2
    screen.fill((18, 18, 22))

    def to_px(v):
        return int(cx + v[0] * SCALE), int(cy - v[1] * SCALE)

    pygame.draw.circle(screen, (44, 44, 52), (cx, cy),
                       int(env.p.arena_radius * SCALE), 2)

    phase_color = {
        EnemyPhase.IDLE: (150, 150, 160),
        EnemyPhase.TELEGRAPH: (235, 190, 60),
        EnemyPhase.ACTIVE: (235, 70, 70),
        EnemyPhase.RECOVERY: (110, 110, 130),
    }[env.enemy_phase]

    if env.enemy_phase in (EnemyPhase.TELEGRAPH, EnemyPhase.ACTIVE):
        from combat_env.config import ATTACKS
        reach = ATTACKS[env.enemy_attack_idx].reach
        pygame.draw.circle(screen, phase_color, to_px(env.enemy_pos),
                           int(reach * SCALE), 1)

    pygame.draw.circle(screen, phase_color, to_px(env.enemy_pos), 17)
    agent_color = (90, 200, 255) if env._invulnerable else (
        (120, 255, 160) if env.is_blocking else (235, 235, 245))
    pygame.draw.circle(screen, agent_color, to_px(env.agent_pos), 14)

    def bar(x, y, frac, color, width=190):
        pygame.draw.rect(screen, (50, 50, 58), (x, y, width, 12))
        pygame.draw.rect(screen, color, (x, y, int(width * max(0.0, frac)), 12))

    bar(20, 20, env.agent_health / env.p.agent_max_health, (220, 70, 70))
    bar(20, 38, env.agent_stamina / env.p.agent_max_stamina, (230, 200, 70))
    bar(w - 210, 20, env.enemy_health / env.p.enemy_max_health, (200, 90, 200))

    screen.blit(font.render(label, True, (170, 170, 180)), (20, 60))
    pygame.display.flip()


def play_episode(pygame, screen, font, config_id: int, seed: int,
                 label: str) -> dict:
    env = make_env(config_id, seed=seed)
    env.reset(seed=seed)
    clock = pygame.time.Clock()
    t_start = time.time()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                raise SystemExit

        action = read_action(pygame, pygame.key.get_pressed(),
                             pygame.mouse.get_pressed())
        _, _, terminated, truncated, info = env.step(action)
        draw(pygame, screen, env, font, label)
        clock.tick(TICK_HZ // env.p.action_repeat)

        if terminated or truncated:
            rec = info["stats"].as_dict()
            rec.update({
                "config_id": config_id,
                "seed": seed,
                "wall_start": t_start,
                "wall_end": time.time(),
            })
            return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=int, default=12,
                    help="difficulty configuration id (0-24)")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--session", action="store_true",
                    help="run the full interleaved recording protocol")
    ap.add_argument("--episodes-per-config", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None,
                    help="append episode records as JSON lines")
    args = ap.parse_args()

    pygame = _require_pygame()
    pygame.init()
    screen = pygame.display.set_mode((900, 700))
    pygame.display.set_caption("combat-env")
    font = pygame.font.SysFont("monospace", 16)

    rng = np.random.default_rng(args.seed)
    if args.session:
        order: list[int] = list(PROBE_CONFIGS)
        body = np.repeat(np.arange(N_CONFIGS), args.episodes_per_config)
        rng.shuffle(body)
        mid = len(body) // 2
        order += list(body[:mid]) + list(PROBE_CONFIGS) + list(body[mid:])
        order += list(PROBE_CONFIGS)
    else:
        order = [args.config] * args.episodes

    records = []
    try:
        for i, cid in enumerate(order):
            d = DifficultyConfig.from_id(int(cid))
            label = f"[{i + 1}/{len(order)}]  {d}"
            rec = play_episode(pygame, screen, font, int(cid),
                               seed=int(rng.integers(2**31)), label=label)
            rec["order_index"] = i
            rec["is_probe"] = int(cid) in PROBE_CONFIGS
            records.append(rec)
            print(f"{label} -> {rec['outcome']}  "
                  f"taken={rec['damage_taken']:.0f}  "
                  f"dodge_rate={rec['dodge_rate']:.2f}")
    except SystemExit:
        pass
    finally:
        pygame.quit()
        if args.out and records:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("a") as fh:
                for r in records:
                    fh.write(json.dumps(r) + "\n")
            print(f"\nwrote {len(records)} episodes to {args.out}")


if __name__ == "__main__":
    main()
