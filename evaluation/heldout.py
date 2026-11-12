"""Primary dependent variable: normalized return on the held-out set (D039).

Episodes are run with exactly the Gate 1 protocol (`evaluation/gate1.py`):
the same environment factory, the same evaluation seeds, and greedy policies
with nothing carried between episodes.  The difference is that the
undiscounted episode return is summed, which Gate 1's `run_episode` discards.

    normalized_c = (R_c - R_random,c) / (R_scripted,c - R_random,c)

R_random,c and R_scripted,c are the anchors, measured once by
`scripts/measure_return_anchors.py` and stored in `docs/return_anchors.json`.
A run's primary DV is the mean of normalized_c over the five held-out
configurations in `docs/heldout_split.json`.

Small-denominator fallback (D039): a held-out configuration whose denominator
is below FALLBACK_FRACTION of the median denominator over all 25
configurations uses the median denominator instead.

The offset R_random,c cancels in every between-condition contrast.  The
denominators only set how much each held-out configuration weighs in the
mean.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from combat_env import make_env
from evaluation.gate1 import CONFIG_IDS, N_EVAL, Policy, eval_seeds

FALLBACK_FRACTION: float = 0.10     # D039
ANCHORS_PATH = Path("docs/return_anchors.json")
SPLIT_PATH = Path("docs/heldout_split.json")


def run_episode_return(config_id: int, policy: Policy, seed: int) -> tuple[float, Any]:
    """One greedy episode: (undiscounted return, EpisodeStats).

    The same loop as `evaluation.gate1.run_episode`, plus the reward sum.
    """
    env = make_env(config_id, seed=seed)
    obs, _ = env.reset(seed=seed)
    total = 0.0
    while True:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        total += float(reward)
        if terminated or truncated:
            return total, info["stats"]


def evaluate_returns(
    policy: Policy,
    config_ids: Sequence[int] = CONFIG_IDS,
    n_eval: int = N_EVAL,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Mean return per configuration on the Gate 1 evaluation seeds.

    Returns a dict with:
      ``returns``        {config_id: mean episode return}
      ``damage_taken``   {config_id: mean damage taken}, for cross-checking
                         against Gate 1 measurements of the same episodes
      ``episode_returns`` {config_id: [return, ...]} in seed order

    Configurations are visited in the order given.  A stateful policy (the
    random floor shares one Generator) therefore sees the same stream as in
    `evaluate_policy` when called with the same order.
    """
    seeds = eval_seeds(n_eval)
    returns: dict[int, float] = {}
    damage: dict[int, float] = {}
    per_episode: dict[int, list[float]] = {}
    for i, cid in enumerate(config_ids):
        results = [run_episode_return(cid, policy, s) for s in seeds]
        per_episode[cid] = [r for r, _ in results]
        returns[cid] = float(np.mean(per_episode[cid]))
        damage[cid] = float(np.mean([st.damage_taken for _, st in results]))
        if progress is not None:
            progress(i + 1, len(config_ids))
    return {"returns": returns, "damage_taken": damage,
            "episode_returns": per_episode, "n_eval": n_eval}


def denominators(random_returns: Mapping[int, float],
                 scripted_returns: Mapping[int, float]) -> tuple[dict[int, float], set[int]]:
    """Per-configuration denominators after the D039 fallback, and the set of
    configurations the fallback replaced.

    The fallback threshold uses the median over every configuration given,
    which is all 25 in the recorded anchors.
    """
    if set(random_returns) != set(scripted_returns):
        raise ValueError("random and scripted anchors cover different configurations")
    raw = {c: scripted_returns[c] - random_returns[c] for c in random_returns}
    median = float(np.median(list(raw.values())))
    if median <= 0:
        raise ValueError(
            f"median scripted-minus-random gap is {median:.4f} <= 0; the "
            f"scripted policy does not beat random, so these anchors cannot "
            f"normalize anything")
    replaced = {c for c, v in raw.items() if v < FALLBACK_FRACTION * median}
    return {c: (median if c in replaced else v) for c, v in raw.items()}, replaced


def normalized_returns(returns: Mapping[int, float], anchors: Mapping[str, Any],
                       held_out: Sequence[int]) -> dict[int, float]:
    """normalized_c for each held-out configuration."""
    random_r = {int(c): float(v) for c, v in anchors["random"]["returns"].items()}
    scripted_r = {int(c): float(v) for c, v in anchors["scripted"]["returns"].items()}
    denom, _ = denominators(random_r, scripted_r)
    missing = [c for c in held_out if c not in returns]
    if missing:
        raise ValueError(f"no return measured for held-out configurations {missing}")
    return {c: (returns[c] - random_r[c]) / denom[c] for c in held_out}


def primary_dv(returns: Mapping[int, float], anchors: Mapping[str, Any],
               held_out: Sequence[int]) -> float:
    """Mean normalized return over the held-out set: one number per run."""
    if len(held_out) == 0:
        raise ValueError("empty held-out set")
    return float(np.mean(list(normalized_returns(returns, anchors, held_out).values())))


def load_anchors(path: str | Path = ANCHORS_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_held_out(path: str | Path = SPLIT_PATH) -> list[int]:
    return list(json.loads(Path(path).read_text(encoding="utf-8"))["held_out"])
