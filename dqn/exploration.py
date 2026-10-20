"""Build step 6: epsilon schedule and epsilon-greedy action selection."""

import numpy as np


def _check_q(q_values) -> np.ndarray:
    """One row of Q-values: 1-D, non-empty, finite."""
    q = np.asarray(q_values)
    # Exactly 1-D: passing net.forward's (1, n_actions) output instead of q[0]
    # would make argmax index the flattened array, and len() return 1.
    if q.ndim != 1 or q.size == 0:
        raise ValueError(f"q_values must be 1-D and non-empty, got shape {q.shape}")
    # argmax returns the index of a nan, so a diverged network would look greedy.
    if not np.all(np.isfinite(q)):
        raise ValueError("q_values contains nan or inf")
    return q

def linear_epsilon(t: int, *, start_e: float, end_e: float, duration: float) -> float:
    """CleanRL linear_schedule, same formula and float order:
    slope = (end_e - start_e) / duration;  return max(slope * t + start_e, end_e).
    t = 0-based global env step.  Pure: no state, no RNG."""
    # A float t usually means something other than global_step was passed in.
    if isinstance(t, (bool, np.bool_)) or not isinstance(t, (int, np.integer)):
        raise ValueError(f"t must be an int (the global env step), got {type(t).__name__}")
    if t < 0:
        raise ValueError(f"t must be >= 0, got {t}")
    # Written as `not (x > 0)` so nan is rejected too.
    if not (duration > 0) or not np.isfinite(duration):
        raise ValueError(f"duration must be finite and > 0, got {duration}")
    # end_e <= start_e: with an upward anneal, max() would floor at end_e from step 0.
    if not (0.0 <= end_e <= start_e <= 1.0):
        raise ValueError(f"need 0 <= end_e <= start_e <= 1, got start_e={start_e}, end_e={end_e}")

    slope = (end_e - start_e) / duration  
    return float(max(slope * t + start_e, end_e))

def greedy_action(q_values: np.ndarray) -> int:
    """Index of the max of a 1-D q_values; ties -> lowest index.  No RNG."""
    q_values = _check_q(q_values)
    return int(np.argmax(q_values))

def epsilon_greedy(q_values: np.ndarray, epsilon: float,
                   rng: np.random.Generator) -> int:
    """u = rng.random(); if u < epsilon: rng.integers(n_actions) else greedy_action.
    Exactly one uniform draw per call, plus one integer draw iff random branch."""
    # Validate everything before the first draw, so a rejected call leaves
    # the exploration stream where it was.
    q_values = _check_q(q_values)
    # Written as `not (...)` so nan is rejected too; `u < nan` is always False
    # and would silently mean "never explore".
    if not (0.0 <= epsilon <= 1.0):
        raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
    # RandomState is the legacy global-style API.  An int seed would build a
    # fresh generator each call and repeat the same "random" action forever.
    if not isinstance(rng, np.random.Generator):
        raise ValueError(f"rng must be a np.random.Generator, got {type(rng).__name__}")

    u = rng.random()
    n_actions = len(q_values)
    if u < epsilon:
        return int(rng.integers(n_actions)) 
    else: 
        return greedy_action(q_values)