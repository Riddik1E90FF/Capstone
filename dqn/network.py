"""Build Step 1: multilayer perceptron with a hand-written backward pass."""

from __future__ import annotations
from typing import Sequence
import numpy as np

class MLP:
    def __init__(self, layer_sizes: Sequence[int], rng: np.random.Generator,
                 dtype=np.float64) -> None:
        sizes = tuple(int(s) for s in layer_sizes)
        if len(sizes) < 2 or any(s < 1 for s in sizes):
            raise ValueError(f"layer_sizes needs >= 2 entries, all >= 1; got {sizes}")
        self.n_layers = len(sizes) - 1
        self.layer_sizes = sizes
        self.dtype = np.dtype(dtype)
        self.params = {}
        
        for i, (fan_in, fan_out) in enumerate(zip(sizes, sizes[1:])):
            bound = 1.0 / np.sqrt(fan_in)
            self.params[f"W{i}"] = rng.uniform(-bound, bound, size=(fan_in, fan_out)).astype(self.dtype)
            self.params[f"b{i}"] = rng.uniform(-bound, bound, size=(fan_out,)).astype(self.dtype)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, tuple]:
        x = np.asarray(x, dtype=self.dtype)
        if x.ndim != 2 or x.shape[1] != self.layer_sizes[0]: 
            raise ValueError(f"x must have shape (batch, {self.layer_sizes[0]}), got {x.shape}")
        inputs, preacts = [], []
        h = x
        for i in range(self.n_layers):
            inputs.append(h)
            z = h @ self.params[f"W{i}"] + self.params[f"b{i}"]
            preacts.append(z)
            if i < self.n_layers - 1: 
                h = np.maximum(z, 0.0)
        q = z
        return q, (inputs, preacts)

    def backward(self, cache: tuple, grad_q: np.ndarray) -> dict[str, np.ndarray]:
        inputs, preacts = cache                                  # what forward recorded
        grad_q = np.asarray(grad_q, dtype=self.dtype)
        if grad_q.shape != preacts[-1].shape:                    # must match q's shape exactly
            raise ValueError(f"grad_q shape {grad_q.shape} != output shape {preacts[-1].shape}")

        grads = {}
        d = grad_q                                               # output layer is linear, so dL/dz = dL/dq
        for i in reversed(range(self.n_layers)):                 # L-1, ..., 1, 0
            # Job 1: this layer's own parameter gradients (summed over the batch).
            grads[f"W{i}"] = inputs[i].T @ d                     # (n_i, B) @ (B, n_i+1) -> shape of W_i
            grads[f"b{i}"] = d.sum(axis=0)                       # (n_i+1,)  -> shape of b_i
            # Job 2: pass d down, unless this is the first layer.
            if i > 0:
                dh = d @ self.params[f"W{i}"].T                  # dL/d(this layer's input)
                d = dh * (preacts[i - 1] > 0)                    # back through the ReLU that made that input
        return {k: grads[k] for k in self.params}                # same key order as params
