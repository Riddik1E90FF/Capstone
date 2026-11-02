"""Gate 1 evaluation harness (D034).

The floors and the trained agent must be measured *identically* or their
success rates are not comparable, and G1-a compares them directly.  So the
protocol lives here, in one place, and both callers import it rather than
each writing their own loop.

What is fixed here is fixed by D034 and must not change after the floors are
recorded:

  * ``N_EVAL``          episodes per configuration
  * ``EVAL_SEED_BASE``  the evaluation seed stream
  * the difficulty metric  mean ``EpisodeStats.damage_taken`` (D035).  Higher
                           means harder.  Success rate is reported as a
                           secondary descriptive measure and never ranks.
  * greedy evaluation   no exploration, no learning, nothing carried between
                        episodes

Evaluation seeds are the *same set for every configuration*.  That is
deliberate: it pairs the comparison, so a difference between two
configurations is not partly a difference between the random draws they
happened to get.  It also makes the agent and the floors face the same
episodes.

The seed base is far from the training seeds (1-10), so no evaluation episode
is one the agent trained on.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from combat_env import N_CONFIGS, make_env

# -- D034 constants.  Changing these after the floors are recorded invalidates
# -- the comparison; record a new attempt instead.
N_EVAL: int = 200
EVAL_SEED_BASE: int = 1_000_000
CONFIG_IDS: tuple[int, ...] = tuple(range(N_CONFIGS))

# A policy maps an observation to an action index.  The floors pass a plain
# function; the trained agent passes a greedy wrapper around its network.
# Neither the harness nor the environment can tell which it is.
Policy = Callable[[np.ndarray], int]


def eval_seeds(n_eval: int = N_EVAL) -> list[int]:
    """The evaluation seed set, identical for every configuration."""
    return [EVAL_SEED_BASE + i for i in range(n_eval)]


def run_episode(config_id: int, policy: Policy, seed: int) -> Any:
    """Run one greedy episode and return its EpisodeStats."""
    env = make_env(config_id, seed=seed)
    obs, _ = env.reset(seed=seed)
    while True:
        obs, _, terminated, truncated, info = env.step(policy(obs))
        if terminated or truncated:
            return info["stats"]


def evaluate_policy(
    policy: Policy,
    config_ids: Sequence[int] = CONFIG_IDS,
    n_eval: int = N_EVAL,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Success rate per configuration, plus the per-episode detail.

    Returns a dict with:
      ``damage_taken``  {config_id: mean damage taken}        <- primary (D035)
      ``success_rate``  {config_id: fraction of episodes won}  <- secondary
      ``saturated``     {config_id: fraction of episodes at the health bound}
      ``outcomes``      {config_id: [EpisodeStats, ...]} for later analysis
      ``mean``          mean damage taken across configurations
      ``spread``        max - min damage taken, the quantity G1-c tests

    ``saturated`` exists because damage taken is bounded above by agent maximum
    health: an agent that dies takes exactly that much and no more.  Where a
    configuration kills the agent often, the mean compresses toward the bound
    and its rank becomes unreliable.  D035 flags a configuration above 25%.

    ``spread`` is reported but NOT judged here.  Whether it exceeds chance is
    a permutation test against a config-label null (D034 G1-c), which belongs
    with the statistics, not in the measurement harness.
    """
    from combat_env.config import EnvParams
    health_bound = EnvParams().agent_max_health

    seeds = eval_seeds(n_eval)
    damage_taken: dict[int, float] = {}
    success_rate: dict[int, float] = {}
    saturated: dict[int, float] = {}
    outcomes: dict[int, list[Any]] = {}

    for i, cid in enumerate(config_ids):
        stats = [run_episode(cid, policy, s) for s in seeds]
        outcomes[cid] = stats
        damage_taken[cid] = float(np.mean([st.damage_taken for st in stats]))
        success_rate[cid] = float(np.mean([st.success for st in stats]))
        saturated[cid] = float(np.mean(
            [st.damage_taken >= health_bound for st in stats]))
        if progress is not None:
            progress(i + 1, len(config_ids))

    rates = np.array(list(damage_taken.values()), dtype=float)
    return {
        "damage_taken": damage_taken,
        "success_rate": success_rate,
        "saturated": saturated,
        "outcomes": outcomes,
        "mean": float(rates.mean()),
        "spread": float(rates.max() - rates.min()),
        "n_eval": n_eval,
        "eval_seed_base": EVAL_SEED_BASE,
    }


def random_policy(rng: np.random.Generator, n_actions: int) -> Policy:
    """Uniform random action selection: the G1-a floor.

    The generator is passed in rather than created here so the caller owns the
    stream and can record its seed.  A policy that silently seeds itself is
    not reproducible.
    """
    return lambda obs: int(rng.integers(n_actions))
