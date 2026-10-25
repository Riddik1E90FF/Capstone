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
    if len(returns) < window:
        return np.array([], dtype=float)
    
    averages = []
    for i in range(len(returns) - window + 1):
        averages.append(np.mean(returns[i : i + window]))
        
    return np.array(averages)


def steps_to_threshold(episodes: list[dict[str, Any]], threshold: float,
                       total_timesteps: int = TOTAL_TIMESTEPS) -> tuple[int, bool]:
    """(steps, reached) for one run.

    reached: whether the 100-episode moving average is >= threshold at some
    episode.  steps: the global_step of the first episode at which it is;
    if it never is, the run is censored and steps = total_timesteps.
    """
    window = 100
    # 1. Extract the return from each episode.
    returns = [episode["return"] for episode in episodes]
    
    # 2. Calculate moving averages over complete 100-episode windows
    averages = moving_average(returns, window=window)
    
    # 3. Find the FIRST moving average that reaches the threshold.
    for i, average in enumerate(averages):
        if average >= threshold:
            # 4. Map this moving average to its newest episode.
            episode_index = i + window - 1
            
            # 5. Return that episode's global training step.
            steps = episodes[episode_index]["global_step"]
            return int(steps), True
    # 6. No window reached the threshold: the run is censored.
    return total_timesteps, False


def final_return(episodes: list[dict[str, Any]], window: int = WINDOW) -> float:
    """Mean return of the last `window` episodes of one run."""
    returns = [episode["return"] for episode in episodes] 
    return float(np.mean(returns[-window:]))


def iqm(values: Sequence[float]) -> float:
    """Interquartile mean: the mean of the middle 50%.

    Sort, drop the lowest and highest floor(n / 4) values, and average the
    rest (scipy.stats.trim_mean(x, 0.25)).  For 10 values: drop 2 each end,
    average 6.
    """
    sorted_values = sorted(values)
    
    n = len(sorted_values)
    drop = n // 4
    
    # Remove 'drop' values from each end
    middle = sorted_values[drop : n - drop]
    # Average remaining values
    return float(np.mean(middle))


def choose_threshold(max_moving_averages: Sequence[float]) -> float:
    """T from the reference seeds' best 100-episode moving averages.

    DEFAULT_T if at least MIN_SEEDS_REACHING_T seeds reach it.  Otherwise
    the highest multiple of T_STEP that at least MIN_SEEDS_REACHING_T seeds
    reach.  T is never above DEFAULT_T.
    """
    threshold = DEFAULT_T
    
    while threshold >= 0:
        # Count seeds whose maximum moving average reaches the threshold.
        count = sum(
            value >= threshold
            for value in max_moving_averages
        )
        
        # If enough seeds reach it, return this threshold.
        if count >= MIN_SEEDS_REACHING_T:
            return float(threshold)
        
        # Otherwise, lower the threshold.
        threshold -= T_STEP
    
    raise ValueError("Not enough reference seeds reach any valid threshold")


def bootstrap_pairs(n_seeds: int, n_resamples: int = N_RESAMPLES,
                    seed: int = BOOTSTRAP_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Index arrays for resampled seed groups A and B.

    Both have shape (n_resamples, n_seeds), integers in [0, n_seeds): row r
    of A and row r of B are the r-th pair of groups, each drawn with
    replacement.  From np.random.default_rng(seed): all of A is drawn first,
    as one array, then all of B.  Fixed so the calibration is reproducible
    to the last digit.
    """
    rng = np.random.default_rng(seed)
    
    A = rng.integers(
        0, n_seeds,
        size=(n_resamples, n_seeds),
    )
    
    B = rng.integers(
        0, n_seeds, 
        size=(n_resamples, n_seeds),
    )
    
    return A, B


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
    if not reference_runs:
        raise ValueError("reference_runs must contain at least one run")
    
    # A. Measure each reference run.
    max_moving_averages = []
    final_returns = []
    
    for episodes in reference_runs:
        returns = [episode["return"] for episode in episodes]
        averages = moving_average(returns, window=100)
        
        if len(averages) == 0:
            raise ValueError("Each run needs at least 100 episodes")
        
        max_moving_averages.append(float(np.max(averages)))
        final_returns.append(final_return(episodes))
    
    # B. Select the shared threshold.
    T = choose_threshold(max_moving_averages)
    
    # C. Count seeds that reach T, then subtract 2
    k = sum(value >= T for value in max_moving_averages) - 2
    
    # D. Record steps to threshold for each seed.
    steps = [
        steps_to_threshold(episodes, T, total_timesteps)[0]
        for episodes in reference_runs
    ]
    
    # Generate all bootstrap pairs once.
    A, B = bootstrap_pairs(len(reference_runs))
    
    ratios = []
    deltas = []
    
    # F. Calculate one ratio and one difference per resample.
    for r in range(len(A)):
        # Compare the median steps to threshold.
        ratio = np.median(np.array(steps)[A[r]]) / np.median(np.array(steps)[B[r]])
        ratios.append(ratio)
        
        # Compare final returns using the interquartile mean.
        difference = (
            iqm(np.array(final_returns)[A[r]])
            - iqm(np.array(final_returns)[B[r]])
        )
        deltas.append(difference)
    
    # G. Summarize the bootstrap distributions.
    q = float(np.percentile(ratios, 95))
    F = max(F_FLOOR, q)
    delta = float(np.percentile(deltas, 95))
    
    return {
        "T": T,
        "k": k,
        "q": q,
        "F": F,
        "delta": delta,
    }


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
    if not numpy_runs or not reference_runs:
        raise ValueError("Both groups must contain at least one run")
    
    T = calibration["T"]
    k = calibration["k"]
    F = calibration["F"]
    delta = calibration["delta"]
    
    # 1. Measure steps to threshold and whether each run reached it.
    numpy_results = [
        steps_to_threshold(run, T, total_timesteps)
        for run in numpy_runs
    ]
    reference_results = [
        steps_to_threshold(run, T, total_timesteps)
        for run in reference_runs
    ]
    
    numpy_steps = np.array([result[0] for result in numpy_results])
    numpy_reached = np.array([result[1] for result in numpy_results])
    
    reference_steps = np.array(
        [result[0] for result in reference_results]
    )
    
    # 2. L3: enough NumPy seeds reached the threshold. 
    numpy_reaching = int(np.sum(numpy_reached))
    L3 = numpy_reaching >= k
    
    # 3. L4_steps: compare median steps, accounting for censoring.
    median_numpy_steps = float(np.median(numpy_steps))
    median_reference_steps = float(np.median(reference_steps))
    
    numpy_censored = int(np.sum(~numpy_reached))
    too_many_censored = numpy_censored > len(numpy_runs) / 2
    
    allowed_steps = F * median_reference_steps
    
    L4_steps = (
        not too_many_censored
        and median_numpy_steps <= allowed_steps
    )
    
    # 4. L4_final: compare final-window IQMs.
    numpy_final_returns = np.array([
        final_return(run) for run in numpy_runs 
    ])
    reference_final_returns = np.array([
        final_return(run) for run in reference_runs
    ])
    
    numpy_final_iqm = iqm(numpy_final_returns)
    reference_final_iqm = iqm(reference_final_returns)
    minimum_final_iqm = reference_final_iqm - delta
    
    L4_final = numpy_final_iqm >= minimum_final_iqm
    
    # 5. All three checks must pass.
    passed = L3 and L4_steps and L4_final 
    
    return {
        "L3": bool(L3),
        "L4_steps": bool(L4_steps),
        "L4_final": bool(L4_final),
        "passed": bool(passed),
        
        # L3 measurements
        "T": T,
        "k": k,
        "numpy_seeds_reaching_T": numpy_reaching,
        
        # L4_steps measurements
        "numpy_steps": numpy_steps,
        "reference_steps": reference_steps,
        "median_numpy_steps": median_numpy_steps,
        "median_reference_steps": median_reference_steps,
        "numpy_censored": numpy_censored,
        "too_many_censored": bool(too_many_censored),
        "F": F,
        "allowed_steps": allowed_steps,
        
        # L4_final measurements
        "numpy_final_returns": numpy_final_returns,
        "reference_final_returns": reference_final_returns,
        "numpy_final_iqm": float(numpy_final_iqm),
        "reference_final_iqm": float(reference_final_iqm),
        "delta": delta,
        "minimum_final_iqm": float(minimum_final_iqm),
    }
    
# ======================================================================
# Plumbing: finding and loading runs, and the command line
# ======================================================================
#
# Run folders are the ones written by dqn/runlog.py (our DQN) and by
# scripts/run_cleanrl_reference.py (the reference).  Both contain
# config.json, meta.json, summary.json and episodes.jsonl.

from pathlib import Path  # noqa: E402

SEEDS = tuple(range(1, 11))          # D029: seeds 1-10 for both implementations


def find_runs(root: str | Path, seeds: Sequence[int] = SEEDS,
              total_timesteps: int = TOTAL_TIMESTEPS) -> dict[int, Path]:
    """Map each seed to its one run folder under `root`.

    A folder counts if it is a direct child of `root` with a config.json
    whose "seed" is one of `seeds` and whose "total_timesteps" equals
    `total_timesteps` (so shortened smoke runs never count).  Raises
    ValueError if any seed has no folder, or more than one: Gate 1a must
    never silently use 9 seeds or pick between two runs of the same seed.
    """


def load_episodes(run_dir: str | Path) -> list[dict[str, Any]]:
    """The episode records of one finished run, in episode order.

    Raises ValueError unless the run's summary.json has status "completed":
    a crashed or still-running run is not evidence.  Reading episodes.jsonl
    can use dqn.runlog.read_jsonl.
    """


def main(argv: list[str] | None = None) -> int:
    """Command line.  Two subcommands:

      calibrate --reference DIR --out FILE
          Find and load the 10 reference runs, calibrate, print a per-seed
          table (seed, reached, steps-to-T, max moving average, final
          return) and the numbers T, k, q, F, delta, and write them as JSON
          to FILE together with the run folders they came from.  Refuses to
          overwrite an existing FILE: the numbers are set once (D029).

      check --reference DIR --numpy DIR --calibration FILE
          Load the calibration JSON and both sets of 10 runs, run
          check_gate, and print a per-seed table for each implementation
          and the L3 / L4_steps / L4_final / passed decisions with the
          numbers behind them.

    Returns the process exit code: 0 on success.  For `check`, a failed
    gate is still a successful run of the script (exit 0); the decision is
    in the output.
    """


if __name__ == "__main__":
    import sys

    sys.exit(main())
