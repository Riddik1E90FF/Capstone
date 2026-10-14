import numpy as np

class MLP:
    layer_sizes: tuple[int, ...]     # (obs_dim, *hidden_sizes, n_actions)
    dtype: np.dtype
    params: dict[str, np.ndarray]    # "W0","b0","W1","b1",...  W_l: (n_l, n_{l+1}), b_l: (n_{l+1},)

    def __init__(self, layer_sizes, rng: np.random.Generator, dtype=np.float64): ...
    def forward(self, x) -> tuple[np.ndarray, Cache]: ...
    def backward(self, cache, grad_q) -> dict[str, np.ndarray]: ...