"""The combat:mix env factory (Gate 1 baseline agent, D034).

The bugs these guard against are all quiet ones: a mixture that silently
collapses to a single configuration still trains, still logs, and still
produces a difficulty sweep.  It just produces a flat one, and nothing in the
run says why.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from combat_env import N_CONFIGS, OBS_DIM  # noqa: E402
from train_dqn import make_env_factory      # noqa: E402


def config_sequence(factory, seed: int, n: int) -> list[int]:
    env = factory(seed)
    _, info = env.reset(seed=seed)
    out = [info["config_id"]]
    for _ in range(n - 1):
        _, info = env.reset()
        out.append(info["config_id"])
    return out


@pytest.fixture(scope="module")
def factory():
    f, obs_dim, n_actions = make_env_factory("combat:mix")
    assert obs_dim == OBS_DIM
    return f


def test_every_configuration_appears(factory):
    """All 25 cells must be reachable, at roughly uniform frequency.

    Catches the selector being rebuilt inside reset(), which restarts its
    stream and yields the same configuration every episode.
    """
    counts = Counter(config_sequence(factory, 1234, 1000))
    assert len(counts) == N_CONFIGS, f"only {len(counts)}/{N_CONFIGS} configs appeared"
    # Uniform expectation is 40 per cell; a wide band still catches a selector
    # that is stuck, skewed, or drawing from the wrong range.
    assert min(counts.values()) >= 15, f"under-sampled cell: {counts}"
    assert max(counts.values()) <= 80, f"over-sampled cell: {counts}"


def test_stream_advances_between_episodes(factory):
    """Consecutive unseeded resets must not return the same configuration."""
    seq = config_sequence(factory, 7, 20)
    assert len(set(seq)) > 1, f"stuck on one configuration: {seq}"


def test_same_seed_reproduces_the_sequence(factory):
    """Byte-identical runs require a deterministic configuration order."""
    assert config_sequence(factory, 42, 50) == config_sequence(factory, 42, 50)


def test_different_seeds_differ(factory):
    assert config_sequence(factory, 42, 50) != config_sequence(factory, 43, 50)


def test_observation_is_unchanged_by_the_mixture(factory):
    """The architectural invariant: configuration never reaches the agent.

    If difficulty leaked into the observation the agent could condition on it
    directly, and Gate 2 would no longer measure what it claims to.
    """
    env = factory(5)
    obs, info = env.reset(seed=5)
    assert np.asarray(obs).shape == (OBS_DIM,)
    assert "config_id" in info, "config_id belongs in info, for logging"

    single, obs_dim, _ = make_env_factory("combat:12")
    s_env = single(5)
    s_obs, _ = s_env.reset(seed=5)
    assert obs_dim == OBS_DIM
    assert np.asarray(s_obs).shape == np.asarray(obs).shape


def test_step_matches_the_single_config_interface(factory):
    env = factory(5)
    env.reset(seed=5)
    out = env.step(0)
    assert len(out) == 5
    _, reward, terminated, truncated, _ = out
    assert isinstance(float(reward), float)
    assert isinstance(bool(terminated), bool)
    assert isinstance(bool(truncated), bool)


def test_step_before_reset_is_an_error(factory):
    with pytest.raises(RuntimeError):
        factory(5).step(0)


@pytest.mark.parametrize("bad", ["combat:99", "combat:-1", "combat:banana", "combat:"])
def test_bad_combat_ids_still_rejected(bad):
    """The combat:mix branch must not have widened what combat: accepts."""
    with pytest.raises(SystemExit):
        make_env_factory(bad)
