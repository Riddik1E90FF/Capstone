"""Configuration space and physical constants for the combat environment.

The difficulty configuration space is the object the whole capstone is built
around.  It is deliberately small, discrete and fully enumerable: the human EEG
session must cover every cell in one sitting, which caps the space at roughly
15-30 cells.

Two axes, five levels each, gives 25 configurations.  Both axes were chosen
because they plausibly affect a human and an agent through the *same*
mechanism, which is the assumption Gate 2 tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

# --------------------------------------------------------------------------
# Simulation timing
# --------------------------------------------------------------------------

TICK_HZ: int = 30
DT: float = 1.0 / TICK_HZ

# --------------------------------------------------------------------------
# Difficulty axes
#
# Axis 1 - telegraph duration (seconds).  How long the enemy's wind-up is
#          visible before the attack becomes damaging.  Shorter compresses the
#          window for recognising and responding.
#
# Axis 2 - attack gap (seconds).  Idle time between the end of one attack and
#          the start of the next.  Shorter removes recovery windows and forces
#          stamina management under pressure.
# --------------------------------------------------------------------------

TELEGRAPH_LEVELS: tuple[float, ...] = (1.00, 0.66, 0.42, 0.24, 0.12)
ATTACK_GAP_LEVELS: tuple[float, ...] = (1.60, 1.25, 0.95, 0.72, 0.52)

N_TELEGRAPH_LEVELS = len(TELEGRAPH_LEVELS)
N_GAP_LEVELS = len(ATTACK_GAP_LEVELS)
N_CONFIGS = N_TELEGRAPH_LEVELS * N_GAP_LEVELS  # 25


@dataclass(frozen=True)
class DifficultyConfig:
    """One cell of the difficulty grid.

    `config_id` is the canonical index used everywhere downstream: in the EEG
    session log, in the curriculum sampler, and in per-configuration metrics.
    """

    telegraph_level: int
    gap_level: int

    @property
    def config_id(self) -> int:
        return self.telegraph_level * N_GAP_LEVELS + self.gap_level

    @property
    def telegraph_duration(self) -> float:
        return TELEGRAPH_LEVELS[self.telegraph_level]

    @property
    def attack_gap(self) -> float:
        return ATTACK_GAP_LEVELS[self.gap_level]

    @staticmethod
    def from_id(config_id: int) -> "DifficultyConfig":
        if not 0 <= config_id < N_CONFIGS:
            raise ValueError(f"config_id must be in [0, {N_CONFIGS}), got {config_id}")
        return DifficultyConfig(
            telegraph_level=config_id // N_GAP_LEVELS,
            gap_level=config_id % N_GAP_LEVELS,
        )

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"cfg{self.config_id:02d}(telegraph={self.telegraph_duration:.2f}s, "
            f"gap={self.attack_gap:.2f}s)"
        )


def all_configs() -> list[DifficultyConfig]:
    """Every configuration, ordered by config_id."""
    return [DifficultyConfig.from_id(i) for i in range(N_CONFIGS)]


# --------------------------------------------------------------------------
# Enemy attack repertoire
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AttackSpec:
    """One enemy attack.

    `telegraph_scale` multiplies the configuration's telegraph duration, so the
    three attacks stay distinguishable at every difficulty level while the axis
    still controls overall difficulty.
    """

    name: str
    telegraph_scale: float
    active_duration: float
    recovery_duration: float
    reach: float
    damage: float
    lunge_speed: float = 0.0


ATTACKS: tuple[AttackSpec, ...] = (
    AttackSpec("jab", telegraph_scale=0.80, active_duration=0.10,
               recovery_duration=0.30, reach=1.6, damage=10.0),
    AttackSpec("heavy", telegraph_scale=1.30, active_duration=0.14,
               recovery_duration=0.50, reach=2.1, damage=22.0),
    AttackSpec("lunge", telegraph_scale=1.00, active_duration=0.18,
               recovery_duration=0.45, reach=1.8, damage=16.0, lunge_speed=7.0),
)

N_ATTACKS = len(ATTACKS)


# --------------------------------------------------------------------------
# Environment parameters held constant across all configurations
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvParams:
    arena_radius: float = 6.0

    agent_max_health: float = 100.0
    agent_max_stamina: float = 100.0
    agent_speed: float = 3.4
    agent_attack_reach: float = 1.9
    agent_attack_damage: float = 11.0
    agent_attack_windup: float = 0.10
    agent_attack_cooldown: float = 0.45
    agent_attack_stamina: float = 14.0

    dodge_speed: float = 11.0
    dodge_duration: float = 0.18
    dodge_iframes: float = 0.20
    dodge_cooldown: float = 0.55
    dodge_stamina: float = 24.0

    block_damage_scale: float = 0.35
    block_stamina_per_hit: float = 18.0
    block_move_scale: float = 0.35

    stamina_regen: float = 26.0
    exhausted_penalty_scale: float = 0.55

    enemy_max_health: float = 100.0
    enemy_speed: float = 2.5
    enemy_preferred_range: float = 1.5

    max_episode_seconds: float = 40.0

    # -- human-like agent constraints (ADR-001) ----------------------------
    # Without these the agent reacts instantly and reads exact state, so its
    # difficulty ordering has no reason to resemble a human's.  These are the
    # main lever controlling whether Gate 2 can pass.
    observation_delay_ticks: int = 6   # 200 ms at 30 Hz
    action_repeat: int = 2             # agent re-decides at 15 Hz

    @property
    def max_episode_ticks(self) -> int:
        return int(self.max_episode_seconds * TICK_HZ)


# --------------------------------------------------------------------------
# Reward weights - PROVISIONAL
#
# OPEN-06 is not closed.  These defaults exist so the environment runs and Gate
# 1 can be attempted; they are not a decision.  The failure mode to watch for
# is a policy that survives by never attacking, which `time_penalty` and
# `win_bonus` are meant to discourage.  Any change here invalidates
# cross-experiment comparison and must be recorded before the pre-registration
# is locked.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RewardWeights:
    damage_dealt: float = 1.0      # per point of damage dealt to the enemy
    damage_taken: float = -1.2     # per point of damage taken
    win_bonus: float = 60.0
    loss_penalty: float = -30.0
    time_penalty: float = -0.05    # per agent decision step
    dodge_success_bonus: float = 2.0   # dodged through an attack's active window


@dataclass(frozen=True)
class EnvConfig:
    """Everything needed to construct a reproducible episode."""

    difficulty: DifficultyConfig
    params: EnvParams = field(default_factory=EnvParams)
    rewards: RewardWeights = field(default_factory=RewardWeights)

    def with_difficulty(self, config_id: int) -> "EnvConfig":
        return replace(self, difficulty=DifficultyConfig.from_id(config_id))
