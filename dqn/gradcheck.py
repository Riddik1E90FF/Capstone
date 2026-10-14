"""Finite-difference gradient checking.  The oracle for build step 1 (backward pass).

Idea: for a scalar function f and one input element x_i,

    df/dx_i  ~=  (f(x + h*e_i) - f(x - h*e_i)) / (2h)

Central differences have O(h^2) truncation error, against O(h) for one-sided
differences, which is what makes a 1e-6 tolerance reachable at all.

Rules for using it correctly.  Each one prevents a false alarm or a false pass:

* **float64 only.**  In float32, the h^2 truncation error and the rounding
  error of the subtraction cannot both be made small, so a correct backward
  pass shows relative error around 1e-2 and a wrong one can hide in the noise.
  `numerical_gradient` refuses other dtypes.
* **Stay away from kinks.**  ReLU is not differentiable at 0.  If a
  pre-activation lies within h of 0, the two probes straddle the kink and the
  numerical gradient is meaningless there.  Tests should use inputs whose
  pre-activations are not within ~1e-3 of zero, or check that condition and
  re-draw.
* **f must be deterministic and depend on x only through the array passed
  in.**  `numerical_gradient` perturbs x in place and restores it, so f must
  read x at call time, not capture a copy.
* **Compare with relative error**, not absolute: gradients span orders of
  magnitude across layers.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def numerical_gradient(f: Callable[[], float], x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of the zero-argument scalar function `f`
    with respect to the array `x`, which `f` reads.

    `x` is modified in place during the call and restored exactly afterwards.
    """
    if x.dtype != np.float64:
        raise TypeError(f"gradient checks must run in float64, got {x.dtype}")
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=[["readwrite"]])
    for _ in it:
        idx = it.multi_index
        original = x[idx]
        x[idx] = original + h
        f_plus = float(f())
        x[idx] = original - h
        f_minus = float(f())
        x[idx] = original
        grad[idx] = (f_plus - f_minus) / (2.0 * h)
    return grad


def relative_error(a: np.ndarray, b: np.ndarray, floor: float = 1e-12) -> float:
    """max_i |a_i - b_i| / max(|a_i| + |b_i|, floor).

    The floor stops elements where both gradients are ~0 (e.g. dead ReLUs)
    from dividing by zero.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    denom = np.maximum(np.abs(a) + np.abs(b), floor)
    return float(np.max(np.abs(a - b) / denom)) if a.size else 0.0


def assert_gradient_close(f: Callable[[], float], x: np.ndarray, analytic: np.ndarray,
                          tol: float = 1e-6, h: float = 1e-6, name: str = "x") -> None:
    """Fail with a useful message if `analytic` disagrees with finite differences.

    On failure it reports the worst element, because "rel error 0.5" alone
    doesn't tell you whether one element is off or the whole array is scaled.
    """
    numeric = numerical_gradient(f, x, h=h)
    err = relative_error(analytic, numeric)
    if err > tol:
        denom = np.maximum(np.abs(analytic) + np.abs(numeric), 1e-12)
        worst = np.unravel_index(np.argmax(np.abs(analytic - numeric) / denom), x.shape)
        ratio = analytic[worst] / numeric[worst] if numeric[worst] != 0 else float("inf")
        raise AssertionError(
            f"gradient check failed for {name}: max relative error {err:.3e} > {tol:.1e}\n"
            f"  worst element {worst}: analytic {analytic[worst]:.6e}, "
            f"numeric {numeric[worst]:.6e}, ratio {ratio:.4f}\n"
            "  (a constant ratio across elements suggests a missing scale factor, "
            "e.g. 1/batch or 2; a ratio of -1 is a sign error)")
