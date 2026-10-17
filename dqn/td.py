import numpy as np


def td_target(rewards: np.ndarray,          # (B,) float
            next_q_target: np.ndarray,    # (B, n_actions): target net's output on s'
            terminated: np.ndarray,       # (B,) bool or 0/1
            gamma: float) -> np.ndarray:  # (B,) y, dtype of next_q_target
    rewards = np.asarray(rewards)
    next_q_target = np.asarray(next_q_target)
    terminated = np.asarray(terminated)
    
    if next_q_target.ndim != 2 or rewards.ndim != 1 or terminated.ndim != 1:
        raise ValueError(f"expected rewards (B,), next_q_target (B, n), terminated (B,); got "
                    f"{rewards.shape}, {next_q_target.shape}, {terminated.shape}")
    if not len(rewards) == len(terminated) == next_q_target.shape[0]:
        raise ValueError(f"batch sizes differ: rewards {len(rewards)}, "
                        f"next_q_target {next_q_target.shape[0]}, terminated {len(terminated)}")

    not_done = 1.0 - np.asarray(terminated, dtype=next_q_target.dtype)
    y = rewards + gamma * not_done * next_q_target.max(axis=1)
    return y.astype(next_q_target.dtype, copy=False)

def td_loss_and_grad(q, actions, targets):
    q = np.asarray(q)
    actions = np.asarray(actions)
    targets = np.asarray(targets)

    if q.ndim != 2 or actions.ndim != 1 or targets.ndim != 1:
        raise ValueError(f"expected q (B, n), actions (B,), targets (B,); got "
                        f"{q.shape}, {actions.shape}, {targets.shape}")
    B, n_actions = q.shape
    if len(actions) != B or len(targets) != B:
        raise ValueError(f"batch sizes differ: q {B}, actions {len(actions)}, targets {len(targets)}")
    if not np.issubdtype(actions.dtype, np.integer):
        raise ValueError(f"actions must be integers, got {actions.dtype}")
    if actions.min() < 0 or actions.max() >= n_actions:
        raise ValueError(f"actions must be in [0, {n_actions}), got min {actions.min()}, max {actions.max()}")

    rows = np.arange(B)
    err = q[rows, actions] - targets          # (B,)
    loss = float(np.mean(err ** 2))
    grad_q = np.zeros_like(q)
    grad_q[rows, actions] = 2.0 * err / B     # dL/dq, nonzero only at the taken action
    return loss, grad_q