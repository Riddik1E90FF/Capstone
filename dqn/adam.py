"""Build step 3: Adam."""

import numpy as np


class Adam:
    params: dict[str, np.ndarray]   # the caller's dict itself (net.params), not a copy
    lr: float
    beta1: float
    beta2: float
    eps: float
    t: int                          # completed steps; 0 before the first
    m: dict[str, np.ndarray]        # RAW moments, zeros_like each param, same keys/order
    v: dict[str, np.ndarray]     

    def __init__(self, params, *, lr, beta1, beta2, eps):
        if not params:
            raise ValueError("params is empty: nothing to optimise")
        if not lr > 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        # beta = 1 makes the bias correction 1 - 1^t = 0, a division by zero.
        for name, beta in (("beta1", beta1), ("beta2", beta2)):
            if not 0.0 <= beta < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {beta}")
        if not eps > 0:
            raise ValueError(f"eps must be > 0, got {eps}")

        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        # Initialize first and second moment estimates
        self.m = {
            k: np.zeros_like(p)
            for k, p in params.items()
        }
        self.v = {
            k: np.zeros_like(p)
            for k, p in params.items()
        }
    
        # Number of optimizer updates.
        self.t = 0
    
    
    def step(self, grads: dict[str, np.ndarray]) -> None:
        # Validate every gradient before touching anything, so a rejected step
        # leaves params, m, v and t exactly as they were.
        if set(grads) != set(self.params):
            missing = sorted(set(self.params) - set(grads))
            extra = sorted(set(grads) - set(self.params))
            raise ValueError(f"grads keys must match params: missing {missing}, unexpected {extra}")
        for k, p in self.params.items():
            g = np.asarray(grads[k])
            # Exact match: a scalar or (1, n) gradient would broadcast silently.
            if g.shape != p.shape:
                raise ValueError(f"grads[{k!r}] has shape {g.shape}, param has {p.shape}")
            # A nan reaching v never decays out of it.
            if not np.all(np.isfinite(g)):
                raise ValueError(f"grads[{k!r}] contains nan or inf (step {self.t + 1})")

        # Increment before calculating bias corrections
        self.t += 1
        bias_correction1 = 1 - self.beta1 ** self.t
        bias_correction2 = 1 - self.beta2 ** self.t
        
        for k, p in self.params.items():
            g = grads[k]
            
            # Update the first moment (mean of gradients)
            self.m[k] *= self.beta1
            self.m[k] += (1 - self.beta1) * g
            
            # Update the second moment (mean of squared gradients)
            self.v[k] *= self.beta2
            self.v[k] += (1 - self.beta2) * (g * g)
            
            # Bias-corrected estimates (temporary arrays)
            m_hat = self.m[k] / bias_correction1
            v_hat = self.v[k] / bias_correction2
            
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            