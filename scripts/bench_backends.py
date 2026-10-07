"""Measure DQN update cost on YOUR machine, across backends and batch sizes.

Nothing in this file has been run on your hardware by anyone but you. The
numbers quoted in the project documents came from a single-core Linux container
with no GPU and should be treated as a floor, not a budget.

Run:
    python scripts/bench_backends.py
    python scripts/bench_backends.py --threads 8    # compare thread settings

What it answers:
    1. How fast is the NumPy DQN update on your CPU?
    2. Is PyTorch on CPU faster than NumPy at this size?
    3. Is PyTorch on CUDA faster, or is it launch-bound at small batch?
    4. At what batch size does the GPU start to win, if ever?

Expected result, stated in advance so the measurement can falsify it: at batch
64 with a 20k-parameter network, CUDA should be roughly a wash or slower than
CPU, because per-step kernel launch overhead dominates the arithmetic. The GPU
should only pull ahead at batch sizes far larger than DQN training uses.
"""

# --- thread limits MUST be set before numpy/torch are imported -------------
import argparse
import os
import sys

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--threads", type=int, default=1)
_known, _ = _pre.parse_known_args()
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = str(_known.threads)

import time  # noqa: E402

import numpy as np  # noqa: E402

DIN, HID, DOUT = 18, 128, 12
GAMMA = 0.99


def report_numpy_threads() -> None:
    try:
        from threadpoolctl import threadpool_info
        for pool in threadpool_info():
            print(f"    BLAS: {pool.get('internal_api')} "
                  f"threads={pool.get('num_threads')}")
    except ImportError:
        print("    (pip install threadpoolctl to confirm actual thread counts)")


# ------------------------------------------------------------------ NumPy
class NumpyMLP:
    def __init__(self, rng):
        self.W1 = rng.normal(0, 0.1, (DIN, HID)); self.b1 = np.zeros(HID)
        self.W2 = rng.normal(0, 0.1, (HID, HID)); self.b2 = np.zeros(HID)
        self.W3 = rng.normal(0, 0.1, (HID, DOUT)); self.b3 = np.zeros(DOUT)

    def forward(self, x):
        z1 = np.maximum(0, x @ self.W1 + self.b1)
        z2 = np.maximum(0, z1 @ self.W2 + self.b2)
        return z2 @ self.W3 + self.b3, (x, z1, z2)

    def backward(self, cache, dout, lr=1e-3):
        x, z1, z2 = cache
        dW3 = z2.T @ dout; db3 = dout.sum(0)
        d2 = (dout @ self.W3.T) * (z2 > 0)
        dW2 = z1.T @ d2; db2 = d2.sum(0)
        d1 = (d2 @ self.W2.T) * (z1 > 0)
        dW1 = x.T @ d1; db1 = d1.sum(0)
        for prm, grad in ((self.W1, dW1), (self.b1, db1), (self.W2, dW2),
                          (self.b2, db2), (self.W3, dW3), (self.b3, db3)):
            prm -= lr * grad


def bench_numpy(steps: int, batch: int) -> float:
    rng = np.random.default_rng(0)
    online, target = NumpyMLP(rng), NumpyMLP(rng)
    obs = rng.normal(size=(1, DIN))
    b_obs = rng.normal(size=(batch, DIN))
    b_next = rng.normal(size=(batch, DIN))
    rew = rng.normal(size=batch)
    acts = rng.integers(DOUT, size=batch)
    idx = np.arange(batch)

    t0 = time.perf_counter()
    for _ in range(steps):
        online.forward(obs)
        qn, _ = target.forward(b_next)
        q, cache = online.forward(b_obs)
        tgt = rew + GAMMA * qn.max(1)
        grad = np.zeros_like(q)
        grad[idx, acts] = (q[idx, acts] - tgt) / batch
        online.backward(cache, grad)
    return steps / (time.perf_counter() - t0)


# ---------------------------------------------------------------- PyTorch
def bench_torch(steps: int, batch: int, device: str) -> float | None:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None
    if device == "cuda" and not torch.cuda.is_available():
        return None

    torch.manual_seed(0)
    torch.set_num_threads(_known.threads)

    def mlp():
        return nn.Sequential(nn.Linear(DIN, HID), nn.ReLU(),
                             nn.Linear(HID, HID), nn.ReLU(),
                             nn.Linear(HID, DOUT)).to(device)

    online, target = mlp(), mlp()
    opt = torch.optim.Adam(online.parameters(), lr=1e-3)
    obs = torch.randn(1, DIN, device=device)
    b_obs = torch.randn(batch, DIN, device=device)
    b_next = torch.randn(batch, DIN, device=device)
    rew = torch.randn(batch, device=device)
    acts = torch.randint(DOUT, (batch,), device=device)

    for _ in range(50):  # warm-up: CUDA context, autotuning, allocator
        with torch.no_grad():
            online(obs)
        q = online(b_obs).gather(1, acts[:, None]).squeeze(1)
        with torch.no_grad():
            tgt = rew + GAMMA * target(b_next).max(1).values
        loss = ((q - tgt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    if device == "cuda":
        import torch as _t; _t.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        with torch.no_grad():
            online(obs)                      # action selection: forces a sync
        q = online(b_obs).gather(1, acts[:, None]).squeeze(1)
        with torch.no_grad():
            tgt = rew + GAMMA * target(b_next).max(1).values
        loss = ((q - tgt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    if device == "cuda":
        import torch as _t; _t.cuda.synchronize()
    return steps / (time.perf_counter() - t0)


def main() -> None:
    ap = argparse.ArgumentParser(parents=[_pre])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batches", type=int, nargs="+", default=[64, 256, 1024, 4096])
    args = ap.parse_args()

    print(f"thread limit per process: {args.threads}")
    print(f"network: {DIN}-{HID}-{HID}-{DOUT} "
          f"(~{DIN*HID + HID*HID + HID*DOUT:,} weights)")
    report_numpy_threads()
    try:
        import torch
        print(f"    torch {torch.__version__}, "
              f"cuda available: {torch.cuda.is_available()}"
              + (f", {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
    except ImportError:
        print("    torch not installed (pip install torch) - NumPy only")

    print(f"\n{'batch':>7} {'numpy':>12} {'torch cpu':>12} {'torch cuda':>12}   (updates/sec)")
    print("-" * 60)
    for batch in args.batches:
        n = bench_numpy(args.steps, batch)
        tc = bench_torch(args.steps, batch, "cpu")
        tg = bench_torch(args.steps, batch, "cuda")
        fmt = lambda v: f"{v:12,.0f}" if v else f"{'n/a':>12}"
        print(f"{batch:>7} {fmt(n)} {fmt(tc)} {fmt(tg)}")

    print("\nInterpretation:")
    print("  Batch 64 is what DQN actually uses. Compare that row.")
    print("  If cuda is not clearly ahead at batch 64, the GPU is launch-bound")
    print("  and moving the agent to it will not speed up training.")
    print("  Watch where cuda overtakes cpu as batch grows - if that point is")
    print("  far above 64, the GPU is solving a problem you do not have.")
    print("\n  Multiply updates/sec by your planned env steps to get run time.")
    print("  Combined throughput also pays for the environment step itself;")
    print("  run scripts/bench_budget.py for that half.")


if __name__ == "__main__":
    main()
