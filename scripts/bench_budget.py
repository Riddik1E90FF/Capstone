"""Measure what a training run actually costs.

Two things are timed separately, because they differ by an order of magnitude:

  1. The environment alone (what the 270 episodes/sec figure measured).
  2. A NumPy DQN's per-step cost: action selection plus one gradient step on a
     replay minibatch. This is what actually dominates a training run.
"""

# Make this script runnable from any directory: put the repo root (this
# file's parent's parent) on sys.path so `combat_env` and `dqn` import
# without PYTHONPATH.  Resolved from __file__, so a git worktree imports
# its own copy rather than another checkout's.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import time
import numpy as np

import sys
from combat_env import N_ACTIONS, make_env

# ---------------------------------------------------------------- env only
def bench_env(n_episodes=300):
    rng = np.random.default_rng(0)
    steps = 0
    t0 = time.perf_counter()
    for e in range(n_episodes):
        env = make_env(e % 25, seed=e)
        env.reset(seed=e)
        while True:
            _, _, term, trunc, _ = env.step(int(rng.integers(N_ACTIONS)))
            steps += 1
            if term or trunc:
                break
    dt = time.perf_counter() - t0
    return steps, n_episodes, dt


# ------------------------------------------------- NumPy MLP forward/backward
class MLP:
    """18 -> 128 -> 128 -> 12, the shape a DQN for this env would plausibly use."""

    def __init__(self, rng, din=18, h=128, dout=12):
        self.W1 = rng.normal(0, 0.1, (din, h)); self.b1 = np.zeros(h)
        self.W2 = rng.normal(0, 0.1, (h, h));   self.b2 = np.zeros(h)
        self.W3 = rng.normal(0, 0.1, (h, dout)); self.b3 = np.zeros(dout)

    def forward(self, x):
        z1 = np.maximum(0, x @ self.W1 + self.b1)
        z2 = np.maximum(0, z1 @ self.W2 + self.b2)
        return z2 @ self.W3 + self.b3, (x, z1, z2)

    def backward(self, cache, dout, lr=1e-3):
        x, z1, z2 = cache
        dW3 = z2.T @ dout; db3 = dout.sum(0)
        d2 = (dout @ self.W3.T) * (z2 > 0)
        dW2 = z1.T @ d2;   db2 = d2.sum(0)
        d1 = (d2 @ self.W2.T) * (z1 > 0)
        dW1 = x.T @ d1;    db1 = d1.sum(0)
        for p, g in ((self.W1, dW1), (self.b1, db1), (self.W2, dW2),
                     (self.b2, db2), (self.W3, dW3), (self.b3, db3)):
            p -= lr * g


def bench_dqn_step(n=4000, batch=64):
    rng = np.random.default_rng(0)
    online, target = MLP(rng), MLP(rng)
    obs = rng.normal(size=(1, 18))
    batch_obs = rng.normal(size=(batch, 18))
    batch_next = rng.normal(size=(batch, 18))
    rew = rng.normal(size=batch)
    acts = rng.integers(12, size=batch)

    t0 = time.perf_counter()
    for _ in range(n):
        online.forward(obs)                      # action selection
        qn, _ = target.forward(batch_next)       # target network
        q, cache = online.forward(batch_obs)     # online on minibatch
        tgt = rew + 0.99 * qn.max(1)
        grad = np.zeros_like(q)
        grad[np.arange(batch), acts] = (q[np.arange(batch), acts] - tgt) / batch
        online.backward(cache, grad)
    return n, time.perf_counter() - t0


if __name__ == "__main__":
    steps, eps, dt = bench_env(300)
    env_sps = steps / dt
    print(f"ENV ONLY")
    print(f"  {eps} episodes, {steps} agent steps in {dt:.2f}s")
    print(f"  {eps/dt:.0f} episodes/sec   {env_sps:.0f} agent-steps/sec")
    print(f"  mean {steps/eps:.0f} agent steps per episode")

    n, dtn = bench_dqn_step()
    net_sps = n / dtn
    print(f"\nDQN UPDATE (NumPy, 18-128-128-12, batch 64)")
    print(f"  {n} steps in {dtn:.2f}s -> {net_sps:.0f} steps/sec")

    combined = 1.0 / (1.0 / env_sps + 1.0 / net_sps)
    print(f"\nCOMBINED  {combined:.0f} agent-steps/sec  "
          f"({env_sps/combined:.0f}x slower than env alone)")

    print(f"\n{'env steps':>12} {'hours/run':>11} {'70 runs (h)':>12} {'8-way (h)':>11}")
    for budget in (200_000, 500_000, 1_000_000, 2_000_000):
        h = budget / combined / 3600
        print(f"{budget:>12,} {h:>11.2f} {h*70:>12.1f} {h*70/8:>11.1f}")
