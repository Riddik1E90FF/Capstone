"""Minimal ARPG-style combat environment.

Design constraints, in priority order:

1. **Deterministic.**  Same seed and same action sequence gives byte-identical
   trajectories.  Every experiment in this project depends on it.
2. **Headless and fast.**  Roughly sixty training runs have to be affordable.
3. **Playable by a human at real time**, using the *same* dynamics the agent
   trains on.  If the human and the agent play different games, the difficulty
   rankings compared at Gate 2 are not comparable.

All scheduling is done in **integer ticks**, never accumulated floats.  An
earlier version used float timers compared for equality; floating-point residue
of ~1e-17 meant the agent's attack never resolved and every episode was an
automatic loss.  Integer ticks make that class of bug impossible.

The API follows the Gymnasium convention (`reset` -> obs, info; `step` -> obs,
reward, terminated, truncated, info) without taking a hard dependency on it, so
the package runs with numpy alone.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Any

import numpy as np

from .config import (
    ATTACKS,
    DT,
    N_ATTACKS,
    TICK_HZ,
    AttackSpec,
    DifficultyConfig,
    EnvConfig,
)

# Geometry constants that are not in EnvParams.  They are part of the frozen
# dynamics all the same.
_SPAWN_DISTANCE = 3.2   # enemy's distance from the agent at reset
_ENGAGE_MARGIN = 0.8    # enemy starts an attack within reach + this margin
_EPS = 1e-8             # below this distance, a direction is undefined

# --------------------------------------------------------------------------
# Action space (discrete, 12)
# --------------------------------------------------------------------------

_DIAG = 0.7071067811865476  # 1/sqrt(2), written out so the value is fixed
# Unit (x, y) vectors, +y is north, in the same order as MOVE_N..MOVE_NW.
MOVE_DIRECTIONS: tuple[tuple[float, float], ...] = (
    (0.0, 1.0), (_DIAG, _DIAG), (1.0, 0.0), (_DIAG, -_DIAG),
    (0.0, -1.0), (-_DIAG, -_DIAG), (-1.0, 0.0), (-_DIAG, _DIAG),
)


class Action(IntEnum):
    """One action per decision, held for `action_repeat` ticks.

    Actions are mutually exclusive: there is no move-while-blocking.
    """

    IDLE = 0
    MOVE_N = 1
    MOVE_NE = 2
    MOVE_E = 3
    MOVE_SE = 4
    MOVE_S = 5
    MOVE_SW = 6
    MOVE_W = 7
    MOVE_NW = 8
    ATTACK = 9
    DODGE = 10
    BLOCK = 11


N_ACTIONS = len(Action)


class EnemyPhase(IntEnum):
    """Enemy attack cycle: IDLE -> TELEGRAPH -> ACTIVE -> RECOVERY -> IDLE.

    The integer values index the phase one-hot in the observation.
    """

    IDLE = 0
    TELEGRAPH = 1
    ACTIVE = 2
    RECOVERY = 3


OBS_DIM = 18


def secs_to_ticks(seconds: float) -> int:
    """Convert a duration to whole ticks, always at least one.

    Uses Python's `round`, so exact half-ticks round to even.
    """
    return max(1, int(round(seconds * TICK_HZ)))


@dataclass
class EpisodeStats:
    """Per-episode metrics.  These feed Gate 1, Gate 2 and the final analysis."""

    ticks: int = 0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    hits_landed: int = 0
    attacks_thrown: int = 0
    times_hit: int = 0
    dodges_attempted: int = 0
    dodges_successful: int = 0
    blocks_absorbed: int = 0
    attacks_faced: int = 0
    stamina_spent: float = 0.0
    outcome: str = "timeout"  # "win" | "loss" | "timeout"

    @property
    def success(self) -> bool:
        """The binary used for per-configuration success rate."""
        return self.outcome == "win"

    @property
    def dodge_rate(self) -> float:
        """Successful dodges / dodges attempted; 0.0 if none were attempted."""
        return self.dodges_successful / self.dodges_attempted if self.dodges_attempted else 0.0

    def as_dict(self) -> dict[str, Any]:
        """Flat, JSON-serialisable record including the derived fields."""
        return {
            "ticks": self.ticks,
            "seconds": self.ticks * DT,
            "damage_dealt": self.damage_dealt,
            "damage_taken": self.damage_taken,
            "hits_landed": self.hits_landed,
            "attacks_thrown": self.attacks_thrown,
            "times_hit": self.times_hit,
            "dodges_attempted": self.dodges_attempted,
            "dodges_successful": self.dodges_successful,
            "dodge_rate": self.dodge_rate,
            "blocks_absorbed": self.blocks_absorbed,
            "attacks_faced": self.attacks_faced,
            "stamina_spent": self.stamina_spent,
            "outcome": self.outcome,
            "success": self.success,
        }


class CombatEnv:
    """One agent, one enemy, one difficulty configuration.

    RNG use is part of the determinism contract: exactly one draw per reset
    (spawn angle) and one per enemy attack (which attack).  The constructor
    also performs a reset, so `reset()` without a seed continues the stream
    from after that first draw.
    """

    def __init__(self, config: EnvConfig, seed: int | None = None) -> None:
        # Short names `p` and `w` are read by scripts/play.py; keep them.
        self.cfg = config
        self.p = config.params
        self.w = config.rewards
        self._rng = np.random.default_rng(seed)

        # Pre-convert every duration to ticks once.
        self._attack_windup_ticks = secs_to_ticks(self.p.agent_attack_windup)
        self._attack_cooldown_ticks = secs_to_ticks(self.p.agent_attack_cooldown)
        self._dodge_duration_ticks = secs_to_ticks(self.p.dodge_duration)
        self._dodge_iframe_ticks = secs_to_ticks(self.p.dodge_iframes)
        self._dodge_cooldown_ticks = secs_to_ticks(self.p.dodge_cooldown)
        self._attack_gap_ticks = secs_to_ticks(config.difficulty.attack_gap)

        self._obs_buffer: deque[np.ndarray] = deque()
        self._reset_state()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode; reseed the RNG only if `seed` is given.

        The observation delay buffer is pre-filled with the initial frame, so
        the first `observation_delay_ticks` observations repeat it.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_state()
        raw = self._raw_observation()
        self._obs_buffer = deque(
            [raw.copy() for _ in range(self.p.observation_delay_ticks + 1)],
            maxlen=self.p.observation_delay_ticks + 1,
        )
        return self._obs_buffer[0].copy(), self._info()

    def _reset_state(self) -> None:
        self.tick = 0
        self.stats = EpisodeStats()

        angle = float(self._rng.uniform(0.0, 2.0 * math.pi))
        self.agent_pos = np.array([0.0, 0.0], dtype=np.float64)
        self.enemy_pos = np.array(
            [_SPAWN_DISTANCE * math.cos(angle), _SPAWN_DISTANCE * math.sin(angle)],
            dtype=np.float64,
        )
        self.facing = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)

        self.agent_health = self.p.agent_max_health
        self.agent_stamina = self.p.agent_max_stamina
        self.enemy_health = self.p.enemy_max_health

        # Integer tick schedules.  "ready_tick" = first tick the action is legal.
        self.attack_ready_tick = 0
        self.attack_resolve_tick = -1      # -1 = no attack in flight
        self.dodge_ready_tick = 0
        self.dodge_until_tick = -1
        self.iframe_until_tick = -1
        self.dodge_dir = np.zeros(2, dtype=np.float64)
        self.is_blocking = False

        self.enemy_phase = EnemyPhase.IDLE
        self.enemy_phase_start_tick = 0
        self.enemy_phase_end_tick = self._attack_gap_ticks
        self.enemy_attack_idx = 0  # index into ATTACKS; stale outside TELEGRAPH..RECOVERY
        self._active_hit_applied = False
        self._active_dodged = False

        self.terminated = False
        self.truncated = False

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance by `action_repeat` ticks holding `action` constant.

        Stops early if the episode ends mid-repeat.  `time_penalty` is charged
        once per call regardless.  The returned observation is the one from
        `observation_delay_ticks` buffer entries ago, not the current state.
        """
        if self.terminated or self.truncated:
            raise RuntimeError("step() called on a finished episode; call reset()")

        act = Action(int(action))
        reward = 0.0
        for _ in range(self.p.action_repeat):
            reward += self._advance_one_tick(act)
            if self.terminated or self.truncated:
                break

        reward += self.w.time_penalty
        self._obs_buffer.append(self._raw_observation())
        return (
            self._obs_buffer[0].copy(),
            float(reward),
            self.terminated,
            self.truncated,
            self._info(),
        )

    def _advance_one_tick(self, action: Action) -> float:
        """One simulation tick.  The order of the sub-steps is part of the dynamics."""
        self.tick += 1
        self.stats.ticks = self.tick

        self.is_blocking = action == Action.BLOCK
        self._apply_agent_action(action)
        reward = self._resolve_agent_attack()
        reward += self._update_enemy()
        self._regen_stamina()
        self._resolve_termination()

        if self.terminated:
            reward += (
                self.w.win_bonus if self.stats.outcome == "win" else self.w.loss_penalty
            )
        return reward

    # ------------------------------------------------------------------
    # Agent
    # ------------------------------------------------------------------

    @property
    def _dodging(self) -> bool:
        return self.tick <= self.dodge_until_tick

    @property
    def _invulnerable(self) -> bool:
        return self.tick <= self.iframe_until_tick
    
    @property
    def invulnerable(self) -> bool:
        """True while dodge i-frames are active. Public, for display."""
        return self._invulnerable

    @property
    def attack_cooldown_fraction(self) -> float:
        """0.0 when ready, 1.0 just after attacking. For display only."""
        return min(1.0, max(0, self.attack_ready_tick - self.tick) / self._attack_cooldown_ticks)
    
    @property
    def dodge_cooldown_fraction(self) -> float:
        """1.0 just after dodging, decaying to 0.0 when ready. Display only."""
        remaining = max(0, self.dodge_ready_tick - self.tick)
        return min(1.0, remaining / self._dodge_cooldown_ticks)

    def _apply_agent_action(self, action: Action) -> None:
        """Start moves, dodges and attacks.  Never produces reward directly."""
        # A dodge in progress overrides everything until it finishes.
        if self._dodging:
            self._move(self.agent_pos, self.dodge_dir, self.p.dodge_speed)
            return

        if Action.MOVE_N <= action <= Action.MOVE_NW:
            direction = np.array(MOVE_DIRECTIONS[action - Action.MOVE_N])
            speed = self.p.agent_speed
            # No blocking slow-down here: a move action is never a block action,
            # so the agent cannot block while moving (D030).
            # Practically unreachable but kept (D030): stamina never goes below
            # 0, and reaches exactly 0.0 only if a blocked hit lands with stamina
            # at exactly block_stamina_per_hit, which needs a float coincidence.
            if self.agent_stamina <= 0.0:
                speed *= self.p.exhausted_penalty_scale
            self._move(self.agent_pos, direction, speed)
            self.facing = direction

        elif action == Action.DODGE:
            if self.tick >= self.dodge_ready_tick and self.agent_stamina >= self.p.dodge_stamina:
                self.agent_stamina -= self.p.dodge_stamina
                self.stats.stamina_spent += self.p.dodge_stamina
                self.dodge_ready_tick = self.tick + self._dodge_cooldown_ticks
                self.dodge_until_tick = self.tick + self._dodge_duration_ticks
                self.iframe_until_tick = self.tick + self._dodge_iframe_ticks
                self.dodge_dir = self.facing.copy()
                self.stats.dodges_attempted += 1

        elif action == Action.ATTACK:
            legal = (
                self.tick >= self.attack_ready_tick
                and self.attack_resolve_tick < 0
                and self.agent_stamina >= self.p.agent_attack_stamina
            )
            if legal:
                self.agent_stamina -= self.p.agent_attack_stamina
                self.stats.stamina_spent += self.p.agent_attack_stamina
                self.stats.attacks_thrown += 1
                self.attack_ready_tick = self.tick + self._attack_cooldown_ticks
                self.attack_resolve_tick = self.tick + self._attack_windup_ticks

    def _resolve_agent_attack(self) -> float:
        """Land the agent's attack on exactly the scheduled tick."""
        if self.attack_resolve_tick != self.tick:
            return 0.0
        self.attack_resolve_tick = -1
        if self._distance() > self.p.agent_attack_reach:
            return 0.0
        self.enemy_health -= self.p.agent_attack_damage
        self.stats.damage_dealt += self.p.agent_attack_damage
        self.stats.hits_landed += 1
        return self.w.damage_dealt * self.p.agent_attack_damage

    def _move(self, pos: np.ndarray, direction: np.ndarray, speed: float) -> None:
        pos += direction * speed * DT
        radius = float(np.linalg.norm(pos))
        if radius > self.p.arena_radius:
            pos *= self.p.arena_radius / radius

    def _regen_stamina(self) -> None:
        if not self.is_blocking:
            self.agent_stamina = min(
                self.p.agent_max_stamina, self.agent_stamina + self.p.stamina_regen * DT
            )

    # ------------------------------------------------------------------
    # Enemy state machine
    # ------------------------------------------------------------------

    def _update_enemy(self) -> float:
        """Advance the enemy's attack cycle one tick; return the reward it causes."""
        reward = 0.0
        spec = ATTACKS[self.enemy_attack_idx]
        phase_over = self.tick >= self.enemy_phase_end_tick

        if self.enemy_phase == EnemyPhase.IDLE:
            self._approach()
            # The range test uses the *previous* attack's reach; the next attack
            # is only drawn once it passes.
            if phase_over and self._distance() <= spec.reach + _ENGAGE_MARGIN:
                self.enemy_attack_idx = int(self._rng.integers(N_ATTACKS))
                spec = ATTACKS[self.enemy_attack_idx]
                self._enter_phase(
                    EnemyPhase.TELEGRAPH,
                    secs_to_ticks(
                        self.cfg.difficulty.telegraph_duration * spec.telegraph_scale
                    ),
                )
                self.stats.attacks_faced += 1
                self._active_hit_applied = False
                self._active_dodged = False

        elif self.enemy_phase == EnemyPhase.TELEGRAPH:
            if phase_over:
                self._enter_phase(EnemyPhase.ACTIVE, secs_to_ticks(spec.active_duration))

        elif self.enemy_phase == EnemyPhase.ACTIVE:
            if spec.lunge_speed > 0.0:
                self._approach(speed=spec.lunge_speed)
            reward += self._resolve_enemy_hit(spec)
            if phase_over:
                if self._active_dodged and not self._active_hit_applied:
                    self.stats.dodges_successful += 1
                    reward += self.w.dodge_success_bonus
                self._enter_phase(
                    EnemyPhase.RECOVERY, secs_to_ticks(spec.recovery_duration)
                )

        elif self.enemy_phase == EnemyPhase.RECOVERY:
            if phase_over:
                self._enter_phase(EnemyPhase.IDLE, self._attack_gap_ticks)
        return reward

    def _enter_phase(self, phase: EnemyPhase, duration_ticks: int) -> None:
        self.enemy_phase = phase
        self.enemy_phase_start_tick = self.tick
        self.enemy_phase_end_tick = self.tick + duration_ticks

    def _approach(self, speed: float | None = None) -> None:
        """Move the enemy toward the agent.

        With no `speed` (normal walking) the enemy backs off when inside its
        preferred range.  With an explicit `speed` (a lunge) it always closes.
        """
        delta = self.agent_pos - self.enemy_pos
        dist = float(np.linalg.norm(delta))
        if dist < _EPS:
            return
        direction = delta / dist
        if speed is None:
            speed = self.p.enemy_speed
            if dist < self.p.enemy_preferred_range:
                direction = -direction
        self._move(self.enemy_pos, direction, speed)

    def _resolve_enemy_hit(self, spec: AttackSpec) -> float:
        """Apply at most one hit per active window; block reduces it if stamina allows."""
        if self._active_hit_applied:
            return 0.0
        if self._distance() > spec.reach:
            return 0.0
        if self._invulnerable:
            self._active_dodged = True  # credited when the active window closes
            return 0.0

        damage = spec.damage
        if self.is_blocking and self.agent_stamina >= self.p.block_stamina_per_hit:
            damage *= self.p.block_damage_scale
            self.agent_stamina -= self.p.block_stamina_per_hit
            self.stats.stamina_spent += self.p.block_stamina_per_hit
            self.stats.blocks_absorbed += 1

        self.agent_health -= damage
        self.stats.damage_taken += damage
        self.stats.times_hit += 1
        self._active_hit_applied = True
        return self.w.damage_taken * damage

    # ------------------------------------------------------------------
    # Termination, observation, info
    # ------------------------------------------------------------------

    def _resolve_termination(self) -> None:
        # Agent death is checked first, so a tick where both die is a loss
        # (D031).  Trading your own life for the kill must not earn the win
        # bonus or count as a success.
        if self.agent_health <= 0.0:
            self.terminated = True
            self.stats.outcome = "loss"
        elif self.enemy_health <= 0.0:
            self.terminated = True
            self.stats.outcome = "win"
        elif self.tick >= self.p.max_episode_ticks:
            self.truncated = True
            self.stats.outcome = "timeout"

    def _distance(self) -> float:
        return float(np.linalg.norm(self.agent_pos - self.enemy_pos))

    def _raw_observation(self) -> np.ndarray:
        """Agent-egocentric view.

        Contains nothing a human player cannot see on screen.  The enemy phase
        and its progress are included because a human reads the same information
        from the wind-up animation; the human-like limitation is imposed by
        `observation_delay_ticks`, not by hiding state.
        """
        delta = self.enemy_pos - self.agent_pos
        dist = float(np.linalg.norm(delta))
        unit = delta / dist if dist > _EPS else np.zeros(2)

        phase_onehot = np.zeros(4, dtype=np.float64)
        phase_onehot[int(self.enemy_phase)] = 1.0
        attack_onehot = np.zeros(N_ATTACKS, dtype=np.float64)
        attack_onehot[self.enemy_attack_idx] = 1.0

        span = max(1, self.enemy_phase_end_tick - self.enemy_phase_start_tick)
        progress = float(np.clip((self.tick - self.enemy_phase_start_tick) / span, 0.0, 1.0))

        attack_cd_norm = max(0, self.attack_ready_tick - self.tick) / self._attack_cooldown_ticks
        dodge_cd_norm = max(0, self.dodge_ready_tick - self.tick) / self._dodge_cooldown_ticks

        obs = np.concatenate([
            unit,                                                    # 0:2
            [min(dist / self.p.arena_radius, 1.0)],                  # 2
            [self.agent_health / self.p.agent_max_health],           # 3
            [self.agent_stamina / self.p.agent_max_stamina],         # 4
            [self.enemy_health / self.p.enemy_max_health],           # 5
            [min(attack_cd_norm, 1.0)],                              # 6
            [min(dodge_cd_norm, 1.0)],                               # 7
            [1.0 if self._invulnerable else 0.0],                    # 8
            phase_onehot,                                            # 9:13
            attack_onehot,                                           # 13:16
            [progress],                                              # 16
            [float(np.linalg.norm(self.agent_pos)) / self.p.arena_radius],  # 17
        ])
        assert obs.shape == (OBS_DIM,), f"expected {OBS_DIM} dims, got {obs.shape}"
        return obs.astype(np.float64)

    def _info(self) -> dict[str, Any]:
        """`stats` is the live object, not a snapshot: it keeps changing on
        later steps until the next reset replaces it."""
        return {
            "config_id": self.cfg.difficulty.config_id,
            "tick": self.tick,
            "agent_health": self.agent_health,
            "enemy_health": self.enemy_health,
            "stats": self.stats,
        }


def make_env(config_id: int, seed: int | None = None, **overrides: Any) -> CombatEnv:
    """Convenience constructor: `make_env(7, seed=0)`.

    `overrides` replace `EnvConfig` fields, e.g. `params=EnvParams(...)`.
    """
    cfg = EnvConfig(difficulty=DifficultyConfig.from_id(config_id))
    if overrides:
        cfg = replace(cfg, **overrides)
    return CombatEnv(cfg, seed=seed)
