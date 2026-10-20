"""Build step 7b: greedy evaluation."""

from typing import Any, Callable, Sequence

import numpy as np

from dqn.exploration import greedy_action
from dqn.network import MLP

def evaluate(online: MLP, env_fn: Callable[[int | None], Any],
             episode_seeds: Sequence[int]) -> list[float]:
    """Play one greedy episode per seed and return each episode's return.

    Measures the policy the network has learned, with no exploration: every
    action is the network's greedy choice for the current observation.

    Separation from training (evaluation must leave a training run
    byte-identical to one without evaluation):
      * Environments come only from `env_fn`.  Never the training env, which
        may be mid-episode.
      * No random numbers are drawn, from any stream or from any global RNG.
      * The network is only read: its parameters are unchanged afterwards.

    Episodes:
      * Episode i starts from a reset seeded with episode_seeds[i], in order.
      * An episode ends when the env reports terminated or truncated.  A
        finished episode is never stepped again.
      * Its return is the undiscounted sum of its rewards.

    Returns one plain float per seed, in the same order as episode_seeds.
    Deterministic: the same network, env and seeds give the same returns.
    """
    # Check every seed before playing anything.  A float or None seed would
    # make env.reset() unseeded or differently seeded without any error.
    if len(episode_seeds) == 0:
        raise ValueError("episode_seeds is empty: nothing to evaluate")
    for s in episode_seeds:
        if isinstance(s, (bool, np.bool_)) or not isinstance(s, (int, np.integer)):
            raise ValueError(f"episode seeds must be ints, got {s!r}")

    results = []
    for seed in episode_seeds:
        env = env_fn(seed)
        obs, info = env.reset(seed=seed)
        done = False
        reward_total = 0.0
        while not done:
            q, _ = online.forward(obs[None])
            action = greedy_action(q[0])
            next_obs, reward, terminated, truncated, info = env.step(action)
            reward_total += reward
            obs = next_obs
            done = terminated or truncated
        results.append(float(reward_total))
    return results
