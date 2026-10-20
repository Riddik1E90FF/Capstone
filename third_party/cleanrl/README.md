# CleanRL DQN: Gate 1a reference (D029)

These are **unmodified** copies of CleanRL files at commit
`fe8d8a03c41a7ef5b523e2e354bd01c363e786bb`, downloaded 2026-10-20 from
`https://raw.githubusercontent.com/vwxyzjn/cleanrl/<commit>/<path>`.

| File | Upstream path | sha256 |
|---|---|---|
| `dqn.py` | `cleanrl/dqn.py` | `a0effc7d087264da1644c3c79eb99cc3ab02b214a758ebf6e9c1443b21ad57d3` |
| `cleanrl_utils/buffers.py` | `cleanrl_utils/buffers.py` | `b43d54020783ac53f7d23fe7c0ed72aff20e1f377a633d134d66eb9bdbece4b8` |
| `cleanrl_utils/__init__.py` | `cleanrl_utils/__init__.py` | (empty file) |
| `LICENSE` | `LICENSE` | `c6c269472eecb009754856b5fe1029ee53b99b986f57bf22ad7c747b04799c69` |

MIT licensed (see `LICENSE`). `buffers.py` is CleanRL's adaptation of the
stable-baselines3 buffer and carries its own MIT notice.

**Do not edit these files.** D029 requires the reference to run unmodified.
Its output is converted to our log format afterwards, by
`scripts/run_cleanrl_reference.py`. `tests/test_cleanrl_reference.py` checks
the hashes above.

## Environment differences from CleanRL's own pins

CleanRL at this commit pins Python `>=3.8,<3.11` and `torch==2.4.1`. This
machine runs Python 3.12 and torch 2.5.0. The script runs unchanged on
both. The versions actually used are recorded in each reference run's
`meta.json`. `gymnasium==0.29.1` matches CleanRL's pin exactly, which is
the pin that matters for CartPole-v1's behaviour.

## Behaviour notes relevant to the comparison

- Replay sampling: uniform with replacement over the filled slots (the
  default `optimize_memory_usage=False` path), the same as `dqn/replay.py`.
- Only `terminated` reaches the buffer as `dones`
  (`handle_timeout_termination=False`). On truncation the real final
  observation is stored as `s'`. Both match `dqn/train.py`.
- Exploration uses Python's global `random` and the action space's own RNG.
  Our DQN uses named NumPy streams instead, so the two implementations never
  produce the same trajectory for the same seed. Gate 1a compares
  distributions over 10 seeds, not individual runs.
- Device: CleanRL uses CUDA when available (`--cuda` defaults to true). The
  device used is recorded per run.
- CleanRL logs only episodic return and length, at its 0-based
  `global_step`. The converter adds 1 (our convention) and derives the
  fields CleanRL does not log; see the converter's docstring.
