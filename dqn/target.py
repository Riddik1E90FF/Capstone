"""Build step 4: Target Network."""

import copy
import numpy as np
from dqn.network import MLP

def make_target(online: MLP) -> MLP:
    """Independent copy of `online`: same layer_sizes and dtype, bit-identical
    params, no array shared with `online`, no random numbers drawn."""
    return copy.deepcopy(online)

def hard_sync(target: MLP, online: MLP) -> None:
    """Copy online's values into target's EXISTING arrays, in place.
    After: bit-equal; target.params[k] is the same object it was before;
    no array shared with online; online unchanged.
    Mismatched keys / shapes / dtype -> ValueError, target untouched."""
    # Validate everything before copying anything.  A depth mismatch still
    # has matching W0 and b0, so a check inside the copy loop would write
    # those and then fail, leaving target half-synced.
    if target is online:
        raise ValueError("target is online: there is no separate target network")
    if list(target.params) != list(online.params):
        raise ValueError(f"param keys differ: target {list(target.params)}, "
                         f"online {list(online.params)}")
    for k, o in online.params.items():
        t = target.params[k]
        if t.shape != o.shape:
            raise ValueError(f"{k}: target shape {t.shape} != online shape {o.shape}")
        # np.copyto would silently cast float64 into float32.
        if t.dtype != o.dtype:
            raise ValueError(f"{k}: target dtype {t.dtype} != online dtype {o.dtype}")
        # Shared arrays make the copy a no-op and the target moves with Adam.
        if np.may_share_memory(t, o):
            raise ValueError(f"{k}: target and online share memory (aliased)")

    for k in online.params:
        np.copyto(target.params[k], online.params[k])
    

def sync_due(global_step: int, *, learning_starts: int,
             target_network_frequency: int) -> bool:
    """True if the target should be synced at this (0-based) global step,
    matching CleanRL lines 194 and 215."""
    if global_step < 0:
        raise ValueError(f"global_step must be >= 0, got {global_step}")
    if learning_starts < 0:
        raise ValueError(f"learning_starts must be >= 0, got {learning_starts}")
    # Also guards the modulo: x % 0 raises ZeroDivisionError, not ValueError.
    if target_network_frequency < 1:
        raise ValueError(f"target_network_frequency must be >= 1, got {target_network_frequency}")

    return global_step > learning_starts and global_step % target_network_frequency == 0