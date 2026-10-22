"""Property-based tests for combat_env (Hypothesis).

`tests/test_env.py` checks a handful of hand-picked (config, seed, policy)
triples.  These tests state the same contracts as properties and let
Hypothesis search for a counterexample over every configuration, seed and
action sequence, shrinking any failure to a minimal one.

Properties:

1. Determinism: same config, seed and actions give byte-identical
   trajectories, including through env reuse and unseeded resets.
2. All 25 grid configurations reset and run to a well-formed episode end.
3. Observation shape and dtype never change.
4. Per-step rewards stay within the bounds implied by the documented reward
   weights (ENVIRONMENT_SPEC.md section 7), and an episode's return
   decomposes exactly into those documented terms.

Plus one ADR-001 check (observation delay), currently xfail; see its reason.

Example counts come from the Hypothesis profile in tests/conftest.py.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from combat_env import (
    N_ACTIONS,
    N_CONFIGS,
    OBS_DIM,
    Action,
    EnvParams,
    RewardWeights,
    make_env,
)
from combat_env.config import ATTACKS
from combat_env.env import secs_to_ticks

# --------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------

_PARAMS = EnvParams()
_WEIGHTS = RewardWeights()
# An episode lasts at most this many step() calls (40 s at 15 decisions/s).
MAX_STEPS = math.ceil(_PARAMS.max_episode_ticks / _PARAMS.action_repeat)

config_ids = st.integers(min_value=0, max_value=N_CONFIGS - 1)
seeds = st.integers(min_value=0, max_value=2**32 - 1)
# Up to a full episode of decisions.  Hypothesis shrinks a failing sequence
# toward short, low-index (IDLE-heavy) sequences, which makes failures readable.
action_seqs = st.lists(
    st.integers(min_value=0, max_value=N_ACTIONS - 1), min_size=1, max_size=MAX_STEPS
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def run_episode(env, actions, *, seed=None, to_end=False):
    """Reset `env` and replay `actions`, recording everything observable.

    With `to_end`, `actions` is cycled until the episode terminates or
    truncates, so termination paths are always exercised.  Otherwise stops at
    the end of `actions` or at episode end, whichever comes first.

    Returns (reset_obs, reset_info, steps), where each step is a dict holding
    the step's full return value plus a snapshot of the live stats.
    """
    obs, info = env.reset(seed=seed)
    reset_obs, reset_info = obs, dict(info)
    steps = []
    i = 0
    while True:
        if not to_end and i >= len(actions):
            break
        obs, reward, terminated, truncated, info = env.step(actions[i % len(actions)])
        steps.append({
            "obs": obs,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "tick": info["tick"],
            "agent_health": info["agent_health"],
            "enemy_health": info["enemy_health"],
            "config_id": info["config_id"],
            # info["stats"] is the live object; snapshot it.
            "stats": info["stats"].as_dict(),
        })
        i += 1
        if terminated or truncated:
            break
        assert i <= MAX_STEPS, "episode exceeded its 40 s limit without truncating"
    return reset_obs, reset_info, steps


def assert_trajectories_identical(a, b):
    """Byte-level equality: obs compared as raw bytes, floats with ==."""
    (obs_a, _, steps_a), (obs_b, _, steps_b) = a, b
    assert obs_a.tobytes() == obs_b.tobytes()
    assert len(steps_a) == len(steps_b)
    for k, (sa, sb) in enumerate(zip(steps_a, steps_b)):
        assert sa["obs"].tobytes() == sb["obs"].tobytes(), f"obs differ at step {k}"
        for key in ("reward", "terminated", "truncated", "tick",
                    "agent_health", "enemy_health", "stats"):
            assert sa[key] == sb[key], f"{key} differs at step {k}"


def reward_bounds(params: EnvParams, weights: RewardWeights) -> tuple[float, float]:
    """Tightest per-step() reward bounds implied by the documented terms.

    One step() is `action_repeat` ticks.  Within that window, each event can
    happen at most once:

    - the agent's hit: its attack cooldown is >= action_repeat ticks;
    - the enemy's hit, and the dodge bonus: each is tied to one ACTIVE window,
      and consecutive ACTIVE windows are separated by RECOVERY + IDLE +
      TELEGRAPH, each at least one tick (secs_to_ticks floors at 1), so at
      least 3 ticks apart;
    - the win/loss bonus: the episode ends on that tick.

    The time penalty is charged exactly once per step.  The preconditions are
    asserted so that a parameter change that breaks this argument fails here
    loudly, rather than silently loosening the test.
    """
    assert secs_to_ticks(params.agent_attack_cooldown) >= params.action_repeat
    assert params.action_repeat <= 3

    worst_hit = max(a.damage for a in ATTACKS)
    upper = (
        weights.time_penalty
        + max(weights.win_bonus, weights.loss_penalty, 0.0)
        + max(weights.damage_dealt * params.agent_attack_damage, 0.0)
        + max(weights.dodge_success_bonus, 0.0)
    )
    lower = (
        weights.time_penalty
        + min(weights.win_bonus, weights.loss_penalty, 0.0)
        + min(weights.damage_taken * worst_hit, 0.0)
    )
    return lower, upper


# --------------------------------------------------------------------------
# 1. Determinism
# --------------------------------------------------------------------------


@given(config_id=config_ids, seed=seeds, actions=action_seqs)
def test_same_seed_and_actions_give_identical_trajectories(config_id, seed, actions):
    a = run_episode(make_env(config_id, seed=seed), actions, seed=seed)
    b = run_episode(make_env(config_id, seed=seed), actions, seed=seed)
    assert_trajectories_identical(a, b)


@given(config_id=config_ids, seed=seeds, warmup=action_seqs, actions=action_seqs)
def test_reset_with_seed_erases_previous_episode(config_id, seed, warmup, actions):
    """A reused env reset with a seed matches a fresh one: no state leaks
    across reset (positions, cooldowns, enemy phase, the delay buffer, RNG)."""
    used = make_env(config_id, seed=seed + 1 if seed < 2**32 - 1 else 0)
    run_episode(used, warmup)
    reused = run_episode(used, actions, seed=seed)
    fresh = run_episode(make_env(config_id, seed=seed), actions, seed=seed)
    assert_trajectories_identical(reused, fresh)


@given(config_id=config_ids, seed=seeds, first=action_seqs, second=action_seqs)
def test_unseeded_reset_continues_stream_deterministically(config_id, seed, first, second):
    """CombatEnv docstring: the constructor resets, and reset() without a seed
    continues the RNG stream.  Two envs built with the same seed must therefore
    agree over a whole sequence of unseeded episodes."""
    env_a, env_b = make_env(config_id, seed=seed), make_env(config_id, seed=seed)
    for actions in (first, second):
        assert_trajectories_identical(
            run_episode(env_a, actions), run_episode(env_b, actions)
        )


# --------------------------------------------------------------------------
# 2. Every grid configuration resets and steps
# --------------------------------------------------------------------------


@pytest.mark.parametrize("config_id", range(N_CONFIGS), ids=lambda c: f"cfg{c:02d}")
@given(seed=seeds, actions=action_seqs)
def test_every_config_runs_to_a_well_formed_end(config_id, seed, actions):
    env = make_env(config_id, seed=seed)
    _, reset_info, steps = run_episode(env, actions, seed=seed, to_end=True)

    assert reset_info["config_id"] == config_id
    assert reset_info["tick"] == 0
    assert steps, "an episode must last at least one step"

    for k, s in enumerate(steps):
        assert s["config_id"] == config_id
        assert isinstance(s["reward"], float) and math.isfinite(s["reward"])
        assert isinstance(s["terminated"], bool) and isinstance(s["truncated"], bool)
        # Each step advances action_repeat ticks, or fewer only on the last.
        if k < len(steps) - 1:
            assert s["tick"] == (k + 1) * env.p.action_repeat
            assert not (s["terminated"] or s["truncated"])

    last = steps[-1]
    assert last["terminated"] != last["truncated"], "exactly one end flag"
    assert last["tick"] <= env.p.max_episode_ticks
    outcome = last["stats"]["outcome"]
    if last["truncated"]:
        assert outcome == "timeout"
        assert last["tick"] == env.p.max_episode_ticks
    else:
        assert outcome in {"win", "loss"}
        # D031: agent death is checked first, so a loss means the agent died.
        if outcome == "loss":
            assert last["agent_health"] <= 0.0
        else:
            assert last["enemy_health"] <= 0.0 < last["agent_health"]

    with pytest.raises(RuntimeError):
        env.step(Action.IDLE)


# --------------------------------------------------------------------------
# 3. Observation shape and dtype
# --------------------------------------------------------------------------


@given(config_id=config_ids, seed=seeds, actions=action_seqs)
def test_observation_shape_and_dtype_are_constant(config_id, seed, actions):
    env = make_env(config_id, seed=seed)
    reset_obs, _, steps = run_episode(env, actions, seed=seed, to_end=True)
    for k, obs in enumerate([reset_obs] + [s["obs"] for s in steps]):
        assert isinstance(obs, np.ndarray), f"step {k}"
        assert obs.shape == (OBS_DIM,), f"step {k}: shape {obs.shape}"
        assert obs.dtype == np.float64, f"step {k}: dtype {obs.dtype}"
        assert np.all(np.isfinite(obs)), f"step {k}: non-finite obs {obs}"


@given(config_id=config_ids, seed=seeds, actions=action_seqs)
def test_returned_observations_are_not_aliased(config_id, seed, actions):
    """A replay buffer stores the returned arrays.  If they aliased env state,
    later steps would silently rewrite stored transitions (env -> caller), or
    a caller writing into one would change the env's future output (caller ->
    env).  Either breaks replay determinism without failing loudly."""
    env = make_env(config_id, seed=seed)
    obs, _ = env.reset(seed=seed)
    stored = [obs]
    for a in actions:
        obs, _, term, trunc, _ = env.step(a)
        stored.append(obs)
        if term or trunc:
            break
    snapshot = [o.tobytes() for o in stored]

    # env -> caller: run a full further episode; nothing handed out may change.
    run_episode(env, actions, seed=seed, to_end=True)
    assert [o.tobytes() for o in stored] == snapshot

    # caller -> env: two envs in lockstep; scribbling on every array one of
    # them hands out must not change anything either of them returns later.
    clean, scribbled = make_env(config_id, seed=seed), make_env(config_id, seed=seed)
    obs_c, _ = clean.reset(seed=seed)
    obs_s, _ = scribbled.reset(seed=seed)
    for k, a in enumerate(actions):
        assert obs_c.tobytes() == obs_s.tobytes(), f"diverged before step {k}"
        obs_s[:] = np.nan
        obs_c, r_c, term, trunc, _ = clean.step(a)
        obs_s, r_s, *_ = scribbled.step(a)
        assert r_c == r_s
        if term or trunc:
            break
    assert obs_c.tobytes() == obs_s.tobytes()


# --------------------------------------------------------------------------
# 4. Reward bounds
# --------------------------------------------------------------------------


@given(config_id=config_ids, seed=seeds, actions=action_seqs)
def test_per_step_reward_within_documented_bounds(config_id, seed, actions):
    env = make_env(config_id, seed=seed)
    lower, upper = reward_bounds(env.p, env.w)
    _, _, steps = run_episode(env, actions, seed=seed, to_end=True)
    for k, s in enumerate(steps):
        assert lower <= s["reward"] <= upper, (
            f"step {k}: reward {s['reward']} outside [{lower}, {upper}]"
        )


def test_reward_bounds_match_documented_weights():
    """Pin the derived bounds to the ENVIRONMENT_SPEC section 7 table, so a
    silent change to a provisional weight (OPEN-06) shows up as a failing test
    that points at the deviations log."""
    lower, upper = reward_bounds(_PARAMS, _WEIGHTS)
    # -0.05 time, -30 loss, -1.2 x 22 (heavy)  |  -0.05 time, +60 win, +1.0 x 11, +2 dodge
    assert lower == pytest.approx(-0.05 - 30.0 - 1.2 * 22.0)
    assert upper == pytest.approx(-0.05 + 60.0 + 1.0 * 11.0 + 2.0)


@given(config_id=config_ids, seed=seeds, actions=action_seqs)
def test_episode_return_decomposes_into_documented_terms(config_id, seed, actions):
    """The return contains exactly the documented reward terms, weighted as
    documented, and nothing else.  Checks the reward against the episode's
    own stats, so an undocumented term or a double-counted event fails."""
    env = make_env(config_id, seed=seed)
    _, _, steps = run_episode(env, actions, seed=seed, to_end=True)
    stats = steps[-1]["stats"]
    w = env.w
    terminal = {"win": w.win_bonus, "loss": w.loss_penalty, "timeout": 0.0}
    expected = (
        w.damage_dealt * stats["damage_dealt"]
        + w.damage_taken * stats["damage_taken"]
        + w.dodge_success_bonus * stats["dodges_successful"]
        + w.time_penalty * len(steps)
        + terminal[stats["outcome"]]
    )
    total = sum(s["reward"] for s in steps)
    assert total == pytest.approx(expected, rel=1e-9, abs=1e-9)


# --------------------------------------------------------------------------
# ADR-001: human-like observation delay
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BUG (environment code, not fixed here): observation_delay_ticks is "
        "applied per step(), not per tick.  CombatEnv appends one frame to the "
        "delay buffer per step() call, and each step() is action_repeat=2 "
        "ticks, so the agent sees state 12 ticks (400 ms) old.  config.py "
        "documents the parameter as '6  # 200 ms at 30 Hz' (ADR-001) and "
        "ENVIRONMENT_SPEC.md section 4 as a '6-tick (200 ms) observation "
        "delay'.  This is one of the two ADR-001 human-likeness levers for "
        "Gate 2.  strict=True: when it is fixed, this test XPASSes and fails, "
        "as a reminder to remove the marker."
    ),
)
@pytest.mark.parametrize("config_id", [0, 12, 24])
def test_observation_delay_is_observation_delay_ticks(config_id):
    env = make_env(config_id, seed=0)
    obs, _ = env.reset(seed=0)
    delay = env.p.observation_delay_ticks
    raw_at_tick = {0: env._raw_observation()}
    returned_at_tick = {0: obs}
    # Alternate E/W so the agent stays inside the arena and every tick's state
    # differs from its neighbours'.
    for k in range(24):
        obs, _, term, trunc, _ = env.step(Action.MOVE_E if k % 4 < 2 else Action.MOVE_W)
        raw_at_tick[env.tick] = env._raw_observation()
        returned_at_tick[env.tick] = obs
        assert not (term or trunc)

    # Precondition: frames are distinguishable, so equality identifies a tick.
    frames = [r.tobytes() for r in raw_at_tick.values()]
    assert len(set(frames)) == len(frames)

    for tick, seen in returned_at_tick.items():
        expected = raw_at_tick[max(0, tick - delay)]
        assert np.array_equal(seen, expected), f"at tick {tick}"
