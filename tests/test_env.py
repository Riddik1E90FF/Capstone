"""Correctness tests. Determinism is the load-bearing one."""

import numpy as np
import pytest

from combat_env import N_ACTIONS, N_CONFIGS, OBS_DIM, DifficultyConfig, all_configs, make_env


def rollout(config_id, seed, policy_seed, steps=400):
    env = make_env(config_id, seed=seed)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(policy_seed)
    trace = [obs.copy()]
    total = 0.0
    info = {}
    for _ in range(steps):
        obs, r, term, trunc, info = env.step(int(rng.integers(N_ACTIONS)))
        trace.append(obs.copy())
        total += r
        if term or trunc:
            break
    return np.array(trace), total, info["stats"].outcome


def test_determinism_identical_seeds():
    a = rollout(7, seed=42, policy_seed=1)
    b = rollout(7, seed=42, policy_seed=1)
    assert np.array_equal(a[0], b[0])
    assert a[1] == b[1]
    assert a[2] == b[2]


def test_different_seeds_diverge():
    a = rollout(7, seed=42, policy_seed=1)
    b = rollout(7, seed=43, policy_seed=1)
    assert not np.array_equal(a[0], b[0])


def test_config_id_roundtrip():
    for i in range(N_CONFIGS):
        assert DifficultyConfig.from_id(i).config_id == i
    assert len(all_configs()) == N_CONFIGS


def test_difficulty_is_monotonic_in_axes():
    easiest = DifficultyConfig.from_id(0)
    hardest = DifficultyConfig.from_id(N_CONFIGS - 1)
    assert easiest.telegraph_duration > hardest.telegraph_duration
    assert easiest.attack_gap > hardest.attack_gap


def test_observation_shape_and_bounds():
    env = make_env(12, seed=0)
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    assert obs.shape == (OBS_DIM,)
    for _ in range(200):
        obs, _, term, trunc, _ = env.step(int(rng.integers(N_ACTIONS)))
        assert obs.shape == (OBS_DIM,)
        assert np.all(np.isfinite(obs))
        if term or trunc:
            break


def test_episode_terminates():
    for cid in (0, 12, 24):
        _, _, outcome = rollout(cid, seed=cid, policy_seed=cid, steps=5000)
        assert outcome in {"win", "loss", "timeout"}


def test_step_after_done_raises():
    env = make_env(24, seed=0)
    env.reset(seed=0)
    for _ in range(5000):
        _, _, term, trunc, _ = env.step(0)
        if term or trunc:
            break
    with pytest.raises(RuntimeError):
        env.step(0)
