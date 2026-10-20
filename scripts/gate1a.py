"""Gate 1a (D029): calibrate L3/L4 numbers from the reference, then check our DQN.

Two jobs, in this order:
  1. calibrate: from the 10 CleanRL reference runs only, set T, k, F and delta
     by D029's rules.  The result is appended to D029 with its date before
     any full NumPy-DQN CartPole run.
  2. check: given the calibrated numbers, decide L3 and both parts of L4 for
     the 10 NumPy-DQN runs.

All metrics come from per-episode TRAINING returns in each run's
episodes.jsonl (exploration on, as CleanRL logs them).  An "episode record"
below is one parsed line of that file: a dict with at least "global_step"
and "return".

Metric definitions are fixed in D029 ("Metric definitions", confirmed
2026-10-20).  The functions below implement exactly those.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

# D029 constants.
WINDOW = 100                 # episodes in the moving average and the final-return mean
DEFAULT_T = 475.0
T_STEP = 25.0
MIN_SEEDS_REACHING_T = 8
N_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
F_FLOOR = 2.0
TOTAL_TIMESTEPS = 500_000


def moving_average(returns: Sequence[float], window: int = WINDOW) -> np.ndarray:
    """Moving average over complete windows only.

    Entry i is the mean of returns[i : i + window], so the array has
    len(returns) - window + 1 entries, and is empty when there are fewer than
    `window` returns.  Entry i belongs to episode i + window - 1, the newest
    episode in its window.
    """


def steps_to_threshold(episodes: list[dict[str, Any]], threshold: float,
                       total_timesteps: int = TOTAL_TIMESTEPS) -> tuple[int, bool]:
    """(steps, reached) for one run.

    reached: whether the 100-episode moving average is >= threshold at some
    episode.  steps: the global_step of the first episode at which it is;
    if it never is, the run is censored and steps = total_timesteps.
    """


def final_return(episodes: list[dict[str, Any]], window: int = WINDOW) -> float:
    """Mean return of the last `window` episodes of one run."""


def iqm(values: Sequence[float]) -> float:
    """Interquartile mean: the mean of the middle 50%.

    Sort, drop the lowest and highest floor(n / 4) values, and average the
    rest (scipy.stats.trim_mean(x, 0.25)).  For 10 values: drop 2 each end,
    average 6.
    """


def choose_threshold(max_moving_averages: Sequence[float]) -> float:
    """T from the reference seeds' best 100-episode moving averages.

    DEFAULT_T if at least MIN_SEEDS_REACHING_T seeds reach it.  Otherwise
    the highest multiple of T_STEP that at least MIN_SEEDS_REACHING_T seeds
    reach.  T is never above DEFAULT_T.
    """


def bootstrap_pairs(n_seeds: int, n_resamples: int = N_RESAMPLES,
                    seed: int = BOOTSTRAP_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Index arrays for resampled seed groups A and B.

    Both have shape (n_resamples, n_seeds), integers in [0, n_seeds): row r
    of A and row r of B are the r-th pair of groups, each drawn with
    replacement.  From np.random.default_rng(seed): all of A is drawn first,
    as one array, then all of B.  Fixed so the calibration is reproducible
    to the last digit.
    """


def calibrate(reference_runs: list[list[dict[str, Any]]],
              total_timesteps: int = TOTAL_TIMESTEPS) -> dict[str, Any]:
    """D029's numbers from the reference runs (one episode list per seed).

    Returns a dict with at least:
      T      choose_threshold of each seed's maximum moving average
      k      number of reference seeds reaching T, minus 2
      q      95th percentile over the bootstrap pairs of
             median(steps of A) / median(steps of B), steps from
             steps_to_threshold at T (censored seeds count as total_timesteps)
      F      max(F_FLOOR, q)
      delta  95th percentile over the same pairs of
             iqm(final returns of A) - iqm(final returns of B)

    Percentiles use NumPy's default method.  Uses no randomness other than
    bootstrap_pairs.
    """


def check_gate(numpy_runs: list[list[dict[str, Any]]],
               reference_runs: list[list[dict[str, Any]]],
               calibration: dict[str, Any],
               total_timesteps: int = TOTAL_TIMESTEPS) -> dict[str, Any]:
    """L3 and L4 for the NumPy DQN against the reference, using `calibration`.

    Returns a dict with at least these booleans:
      L3        numpy seeds reaching T >= k
      L4_steps  median NumPy steps-to-T <= F x median reference steps-to-T,
                and False if more than half the NumPy seeds are censored
      L4_final  iqm(NumPy final returns) >= iqm(reference final returns) - delta
      passed    all three
    plus the numbers each decision was made from, so the report can show them.
    """
