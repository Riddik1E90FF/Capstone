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

# Make this script runnable from any directory: put the repo root (this
# file's parent's parent) on sys.path so `combat_env` and `dqn` import
# without PYTHONPATH.  Resolved from __file__, so a git worktree imports
# its own copy rather than another checkout's.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from combat_env import N_CONFIGS, Action, DifficultyConfig, make_env
from combat_env.config import ATTACKS, TICK_HZ
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


# --------------------------------------------------------------------------
# Rendering
#
# Everything below READS environment state and draws it.  Nothing writes to the
# env, wraps it, or changes pacing.  The purpose is information parity with
# the agent's observation (ENVIRONMENT_SPEC section 4): the agent receives
# distance, its own cooldowns, enemy phase, attack type and phase progress as
# numbers, so the player has to be able to see the same things.  Otherwise
# human per-config difficulty partly measures UI legibility (Gate 2 confound).
#
# Effect timers run on env.tick (simulated time), not wall-clock time.
# --------------------------------------------------------------------------

BG = (18, 18, 22)
C_DIM = (80, 80, 95)
C_TEXT = (170, 170, 180)
C_READY = (120, 230, 140)
C_COOLING = (130, 130, 145)
C_NO_STAMINA = (240, 150, 60)
C_SWING = (245, 245, 210)
C_HIT = (255, 235, 110)
C_HURT = (240, 55, 55)
C_BLOCK = (120, 255, 160)
C_DODGE = (90, 200, 255)

PHASE_COLOR = {
    EnemyPhase.IDLE: (150, 150, 160),
    EnemyPhase.TELEGRAPH: (235, 190, 60),
    EnemyPhase.ACTIVE: (235, 70, 70),
    EnemyPhase.RECOVERY: (110, 110, 130),
}

AGENT_RADIUS_PX = 14
ENEMY_RADIUS_PX = 17


def _ticks(seconds: float) -> int:
    return max(1, int(round(seconds * TICK_HZ)))


FX_HIT_TICKS = _ticks(0.20)     # enemy body flash after your hit lands
FX_SWING_TICKS = _ticks(0.15)   # reach-disc flash when your swing resolves
FX_HURT_TICKS = _ticks(0.35)    # screen-edge flash when you take damage
FX_TEXT_TICKS = _ticks(0.70)    # floating numbers / labels


class Feedback:
    """Turns state *changes* between frames into short-lived visual events.

    Read-only with respect to the env: it copies plain numbers out of
    `env.stats` and `env.attack_resolve_tick` and compares them frame to frame.
    Counters are used rather than per-tick flags so nothing is missed when one
    `env.step()` covers several ticks (`action_repeat`).
    """

    NEVER = -10**9

    def __init__(self, env) -> None:
        self.hit_tick = self.NEVER
        self.swing_tick = self.NEVER
        self.swing_landed = False
        self.hurt_tick = self.NEVER
        self.hurt_scale = 0.0
        self.texts: list[tuple[int, str, tuple[int, int, int], str]] = []
        self._snapshot(env)

    def _snapshot(self, env) -> None:
        s = env.stats
        self._hits = s.hits_landed
        self._thrown = s.attacks_thrown
        self._dealt = s.damage_dealt
        self._times_hit = s.times_hit
        self._taken = s.damage_taken
        self._blocks = s.blocks_absorbed
        self._dodged = s.dodges_successful
        self._in_flight = env.attack_resolve_tick >= 0

    def update(self, env) -> None:
        s, t = env.stats, env.tick
        in_flight = env.attack_resolve_tick >= 0

        # Your swing resolved this step: it was in flight (or was thrown this
        # step) and is no longer.  Cooldown > wind-up, so at most one resolves.
        if (self._in_flight or s.attacks_thrown > self._thrown) and not in_flight:
            landed = s.hits_landed > self._hits
            self.swing_tick, self.swing_landed = t, landed
            if landed:
                self.hit_tick = t
                self.texts.append((t, f"-{_num(s.damage_dealt - self._dealt)}",
                                   C_HIT, "enemy"))
            else:
                self.texts.append((t, "miss", C_COOLING, "agent"))

        if s.times_hit > self._times_hit:
            blocked = s.blocks_absorbed > self._blocks
            dmg = _num(s.damage_taken - self._taken)
            self.hurt_tick = t
            self.hurt_scale = 0.45 if blocked else 1.0
            self.texts.append((t, f"blocked -{dmg}" if blocked else f"-{dmg}",
                               C_BLOCK if blocked else C_HURT, "agent"))

        # Credited by the env when the enemy's active window closes.
        if s.dodges_successful > self._dodged:
            self.texts.append((t, "dodged", C_DODGE, "agent"))

        self.texts = [e for e in self.texts if t - e[0] < FX_TEXT_TICKS]
        self._snapshot(env)

    @staticmethod
    def fade(start: int, now: int, duration: int) -> float:
        """1.0 at the event tick, falling linearly to 0.0 after `duration`."""
        age = now - start
        return 0.0 if age < 0 or age >= duration else 1.0 - age / duration


def _num(x: float) -> str:
    return f"{x:.1f}".rstrip("0").rstrip(".")


def _alpha_circle(pygame, screen, color, alpha, center, radius) -> None:
    radius = int(radius)
    if radius <= 0 or alpha <= 0:
        return
    surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(surf, (*color, int(alpha)), (radius, radius), radius)
    screen.blit(surf, (center[0] - radius, center[1] - radius))


def _arc(pygame, screen, color, center, radius, frac, width=3) -> None:
    """Clockwise arc from 12 o'clock covering `frac` of a full turn."""
    if frac <= 0.01:
        return
    if frac >= 0.999:
        pygame.draw.circle(screen, color, center, radius, width)
        return
    rect = (center[0] - radius, center[1] - radius, radius * 2, radius * 2)
    top = math.pi / 2
    pygame.draw.arc(screen, color, rect, top - 2 * math.pi * frac, top, width)


def draw(pygame, screen, env, fx: Feedback, fonts: dict, label: str) -> None:
    w, h = screen.get_size()
    cx, cy = w // 2, h // 2
    t = env.tick
    p = env.p
    font, small, big = fonts["normal"], fonts["small"], fonts["big"]
    screen.fill(BG)

    def to_px(v):
        return int(cx + v[0] * SCALE), int(cy - v[1] * SCALE)

    def px(r):
        return int(r * SCALE)

    pygame.draw.circle(screen, (44, 44, 52), (cx, cy), px(p.arena_radius), 2)

    agent_xy = to_px(env.agent_pos)
    enemy_xy = to_px(env.enemy_pos)
    dist = float(np.linalg.norm(env.enemy_pos - env.agent_pos))

    # -- enemy reach and phase -------------------------------------------
    # Same quantities as observation dims 9:17 (phase, attack type, progress).
    phase = env.enemy_phase
    pcol = PHASE_COLOR[phase]
    spec = ATTACKS[env.enemy_attack_idx]
    span = max(1, env.enemy_phase_end_tick - env.enemy_phase_start_tick)
    progress = min(1.0, max(0.0, (t - env.enemy_phase_start_tick) / span))
    reach_px = px(spec.reach)

    if phase == EnemyPhase.TELEGRAPH:
        # Danger zone grows from the enemy outward and reaches the outline
        # exactly when the attack becomes damaging.
        _alpha_circle(pygame, screen, pcol, 45, enemy_xy, reach_px * progress)
        pygame.draw.circle(screen, pcol, enemy_xy, reach_px, 2)
    elif phase == EnemyPhase.ACTIVE:
        _alpha_circle(pygame, screen, pcol, 95, enemy_xy, reach_px)
        pygame.draw.circle(screen, pcol, enemy_xy, reach_px, 3)

    # -- your attack reach -------------------------------------------------
    # Bright when the enemy centre is within agent_attack_reach - the exact
    # test the env applies when your swing resolves.
    my_reach_px = px(p.agent_attack_reach)
    in_reach = dist <= p.agent_attack_reach
    if env.attack_resolve_tick >= 0:          # wind-up in flight
        _alpha_circle(pygame, screen, C_SWING, 35, agent_xy, my_reach_px)
        pygame.draw.circle(screen, C_SWING, agent_xy, my_reach_px, 3)
    else:
        pygame.draw.circle(screen, C_READY if in_reach else C_DIM, agent_xy,
                           my_reach_px, 2 if in_reach else 1)
    swing = Feedback.fade(fx.swing_tick, t, FX_SWING_TICKS)
    if swing > 0:
        _alpha_circle(pygame, screen, C_HIT if fx.swing_landed else C_COOLING,
                      110 * swing, agent_xy, my_reach_px)

    # -- bodies --------------------------------------------------------------
    hit_flash = Feedback.fade(fx.hit_tick, t, FX_HIT_TICKS)
    pygame.draw.circle(screen, (255, 255, 255) if hit_flash > 0 else pcol,
                       enemy_xy, ENEMY_RADIUS_PX)
    if hit_flash > 0:
        pygame.draw.circle(screen, C_HIT, enemy_xy,
                           ENEMY_RADIUS_PX + 3 + int(6 * (1 - hit_flash)), 2)

    if phase == EnemyPhase.RECOVERY:
        # Punish window: the arc shows how much of it is left.
        _arc(pygame, screen, C_READY, enemy_xy, ENEMY_RADIUS_PX + 6,
             1.0 - progress, 3)

    phase_text = {
        EnemyPhase.IDLE: "",
        EnemyPhase.TELEGRAPH: f"{spec.name.upper()} wind-up",
        EnemyPhase.ACTIVE: f"{spec.name.upper()}!",
        EnemyPhase.RECOVERY: "open",
    }[phase]
    if phase_text:
        col = C_READY if phase == EnemyPhase.RECOVERY else pcol
        surf = small.render(phase_text, True, col)
        screen.blit(surf, (enemy_xy[0] - surf.get_width() // 2,
                           enemy_xy[1] + ENEMY_RADIUS_PX + 10))

    agent_color = C_DODGE if env.invulnerable else (
        C_BLOCK if env.is_blocking else (235, 235, 245))
    pygame.draw.circle(screen, agent_color, agent_xy, AGENT_RADIUS_PX)

    # -- cooldown readiness ----------------------------------------------------
    # Same normalisation as observation dims 6 and 7.  Near the body: an arc
    # fills while cooling and disappears when ready; an orange ring means the
    # cooldown is done but stamina is below the action's cost.
    atk_cd = env.attack_cooldown_fraction
    dodge_cd = env.dodge_cooldown_fraction
    atk_stamina_ok = env.agent_stamina >= p.agent_attack_stamina
    dodge_stamina_ok = env.agent_stamina >= p.dodge_stamina
    atk_ready = atk_cd == 0 and env.attack_resolve_tick < 0 and atk_stamina_ok
    dodge_ready = dodge_cd == 0 and dodge_stamina_ok

    for cd, stamina_ok, ready, r in (
        (atk_cd, atk_stamina_ok, atk_ready, AGENT_RADIUS_PX + 5),
        (dodge_cd, dodge_stamina_ok, dodge_ready, AGENT_RADIUS_PX + 10),
    ):
        if cd > 0:
            _arc(pygame, screen, C_COOLING, agent_xy, r, 1.0 - cd, 3)
        elif not stamina_ok:
            pygame.draw.circle(screen, C_NO_STAMINA, agent_xy, r, 1)

    # -- HUD -------------------------------------------------------------------
    def bar(x, y, frac, color, text=None, width=190):
        pygame.draw.rect(screen, (50, 50, 58), (x, y, width, 12))
        pygame.draw.rect(screen, color, (x, y, int(width * max(0.0, min(1.0, frac))), 12))
        if text:
            screen.blit(small.render(text, True, color), (x + width + 8, y - 1))

    hurt = Feedback.fade(fx.hurt_tick, t, FX_HURT_TICKS)
    bar(20, 20, env.agent_health / p.agent_max_health,
        (255, 255, 255) if hurt > 0.5 else (220, 70, 70),
        f"HP {env.agent_health:.0f}")
    bar(20, 38, env.agent_stamina / p.agent_max_stamina, (230, 200, 70),
        f"STA {env.agent_stamina:.0f}")
    # Cost thresholds on the stamina bar: attack and dodge.
    for cost, col in ((p.agent_attack_stamina, C_SWING), (p.dodge_stamina, C_DODGE)):
        x = 20 + int(190 * cost / p.agent_max_stamina)
        pygame.draw.line(screen, col, (x, 35), (x, 52), 2)

    def cd_bar(y, name, cd, stamina_ok, ready):
        if ready:
            col, state = C_READY, "READY"
        elif cd > 0:
            col, state = C_COOLING, "..."
        elif not stamina_ok:
            col, state = C_NO_STAMINA, "LOW STA"
        else:
            col, state = C_COOLING, "..."     # attack wind-up still in flight
        bar(20, y, 1.0 - cd, col, f"{name} {state}")

    cd_bar(56, "ATK", atk_cd, atk_stamina_ok, atk_ready)
    cd_bar(74, "DODGE", dodge_cd, dodge_stamina_ok, dodge_ready)

    bar(w - 210, 20, env.enemy_health / p.enemy_max_health,
        (255, 255, 255) if hit_flash > 0 else (200, 90, 200))
    screen.blit(small.render(f"ENEMY {env.enemy_health:.0f}", True, C_TEXT),
                (w - 210, 36))

    screen.blit(font.render(label, True, C_TEXT), (20, 98))

    # -- floating numbers / labels ---------------------------------------------
    for start, text, col, anchor in fx.texts:
        a = Feedback.fade(start, t, FX_TEXT_TICKS)
        if a <= 0:
            continue
        ax, ay = enemy_xy if anchor == "enemy" else agent_xy
        surf = big.render(text, True, col)
        surf.set_alpha(int(255 * a))
        rise = int(28 * (1.0 - a))
        screen.blit(surf, (ax - surf.get_width() // 2, ay - 48 - rise))

    # -- damage-taken screen-edge flash ------------------------------------------
    if hurt > 0:
        edge = pygame.Surface((w, h), pygame.SRCALPHA)
        bands, step = 8, 6
        for i in range(bands):
            alpha = int(170 * hurt * fx.hurt_scale * (1 - i / bands))
            o = i * step
            pygame.draw.rect(edge, (*C_HURT, alpha), (o, o, w - 2 * o, h - 2 * o), step)
        screen.blit(edge, (0, 0))

    pygame.display.flip()


def play_episode(pygame, screen, fonts, config_id: int, seed: int,
                 label: str) -> dict:
    env = make_env(config_id, seed=seed)
    env.reset(seed=seed)
    fx = Feedback(env)
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
        fx.update(env)
        draw(pygame, screen, env, fx, fonts, label)
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
    fonts = {
        "normal": pygame.font.SysFont("monospace", 16),
        "small": pygame.font.SysFont("monospace", 13),
        "big": pygame.font.SysFont("monospace", 22, bold=True),
    }

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
            rec = play_episode(pygame, screen, fonts, int(cid),
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
