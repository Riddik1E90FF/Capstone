"""Step 5: Build replay buffer."""

from typing import NamedTuple

import numpy as np

class Batch(NamedTuple):
    obs: np.ndarray          # (B, obs_dim), buffer dtype
    actions: np.ndarray      # (B,) int64
    rewards: np.ndarray      # (B,) buffer dtype
    next_obs: np.ndarray     # (B, obs_dim), buffer dtype
    terminated: np.ndarray   # (B,) bool

class ReplayBuffer:
    capacity: int
    pos: int                 # next slot to write

    def __init__(self, capacity: int, obs_dim: int, *,
                 rng: np.random.Generator, dtype=np.float64) -> None:
        for name, value in (("capacity", capacity), ("obs_dim", obs_dim)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an int, got {type(value).__name__}")
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")

        self.capacity = capacity
        self.obs_dim = obs_dim
        self.rng = rng
        self.dtype = dtype
        self.size = 0
        self.pos = 0
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.obs = np.zeros((capacity, obs_dim), dtype=dtype)
        self.rewards = np.zeros((capacity,), dtype=dtype)
        self.terminated = np.zeros((capacity,), dtype=bool)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=dtype)
        
    def add(self, obs, action: int, reward: float, next_obs,
            terminated: bool) -> None:
        # Validate the whole transition before writing any field, so a
        # rejected add leaves every slot, pos and size as they were.
        obs, next_obs = np.asarray(obs), np.asarray(next_obs)
        for name, o in (("obs", obs), ("next_obs", next_obs)):
            # Exact match: a scalar or (1, obs_dim) would broadcast into the row.
            if o.shape != (self.obs_dim,):
                raise ValueError(f"{name} must have shape ({self.obs_dim},), got {o.shape}")
            if not np.all(np.isfinite(o)):
                raise ValueError(f"{name} contains nan or inf")
        # np.int64 from argmax or rng.integers is fine; 1.5 would be truncated
        # to 1 on write, and -1 would index the last action in td_loss_and_grad.
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise ValueError(f"action must be an int, got {type(action).__name__}")
        if action < 0:
            raise ValueError(f"action must be >= 0, got {action}")
        if np.ndim(reward) != 0 or not np.isfinite(reward):
            raise ValueError(f"reward must be a finite scalar, got {reward!r}")
        # Strictly bool: 2 or 0.5 would be stored as True.  A non-bool here
        # usually means a `done` or truncation flag computed upstream.
        if not isinstance(terminated, (bool, np.bool_)):
            raise ValueError(f"terminated must be a bool, got {type(terminated).__name__}")

        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_obs[self.pos] = next_obs
        self.terminated[self.pos] = terminated
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size: int) -> Batch:
        if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, (int, np.integer)):
            raise ValueError(f"batch_size must be an int, got {type(batch_size).__name__}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        # rng.integers(0, 0) raises its own, less helpful, error.
        if self.size == 0:
            raise ValueError("cannot sample from an empty buffer")

        idx = self.rng.integers(0, self.size, size=batch_size)
        return Batch(obs=self.obs[idx],
                     actions=self.actions[idx],
                     rewards=self.rewards[idx], 
                     next_obs=self.next_obs[idx],
                     terminated=self.terminated[idx])
        
    def __len__(self) -> int:
        return self.size