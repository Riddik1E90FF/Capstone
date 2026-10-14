"""From-scratch NumPy DQN for the neuroadaptive curriculum capstone.

This package must never import gymnasium, torch, or any RL library (D028,
D029).  `tests/test_dqn_scaffold.py` enforces that.  Environments are handed
in by the caller through the reset/step protocol that `combat_env.CombatEnv`
implements.

Component map and ownership: see dqn/README.md.
"""

from .config import DQNConfig

__all__ = ["DQNConfig"]
