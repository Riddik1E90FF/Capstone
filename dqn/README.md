# `dqn` — from-scratch NumPy DQN

Must pass Gate 1a (`docs/DECISIONS.md` D029) before it touches the combat
environment. Imports NumPy and the standard library only. The import-boundary
tests in `tests/test_dqn_scaffold.py` enforce that.

## Build order and ownership

For every component the user writes, Claude first writes the tests from the
agreed interface spec, before the implementation exists.

| # | Component | File | Writer | Oracle |
|---|---|---|---|---|
| 0 | Config, seeding, run logging, gradient checker, import boundary | `config.py`, `seeding.py`, `runlog.py`, `gradcheck.py` | Claude | — |
| 0 | CLI and env factory | `scripts/train_dqn.py` | Claude | — |
| 1 | MLP forward, analytic backward, init | `network.py` (`tests/test_network.py`) | **user** | finite differences (`gradcheck.py`) |
| 2 | TD target and loss gradient w.r.t. Q | `td.py` (`tests/test_td.py`) | **user** | hand-computed cases incl. terminated vs truncated |
| 3 | Adam | `adam.py` (`tests/test_adam.py`) | **user** | `torch.optim.Adam` on identical gradients (dev-only) |
| 4 | Target network and hard sync | — | **user** | equal after sync, not aliased, unchanged between syncs |
| 5 | Replay buffer | — | **user** | wraparound, capacity, sampling bounds, copy-not-reference |
| 6 | ε schedule and ε-greedy | — | **user** | endpoints, clamping, empirical random-action rate |
| 7a | Training loop | `train.py` | **user** | L2: chain MDP with hand-derived Q\* |
| 7b | Checkpointing (structure below) and greedy eval runner | — | **user** | round-trip test; eval seeds distinct from training seeds |
| 8 | CleanRL reference runner, log conversion, comparison plots | `scripts/` | Claude | — |
| 9 | Gate 1a runs | — | together | D029 |

File names for 1–6 are settled in each component's interface spec.

## Terminated vs truncated: where each is handled

Only `terminated` masks the bootstrap (D029). A truncated transition gets the
ordinary bootstrap, so `td.td_target` takes no `truncated` argument: the bug
"mask on terminated or truncated" cannot be written there. Truncation is
handled upstream instead:

- **Step 5, replay buffer:** store only the `terminated` flag, in a field
  named `terminated`, not `done`. A line like `done=terminated or truncated`
  then reads as wrong in review. Timeouts are still recorded per episode in
  `episodes.jsonl`, so no diagnostics are lost.
- **Step 7a, training loop:** (1) end the episode, reset the env and close
  out episode logging on `terminated or truncated`. This is the only place
  the two flags are combined. (2) On a truncated step, store the real final
  observation returned by `step()` as `s'`, captured before `reset()`, never
  the post-reset observation.

L1 tests cannot see either of these. They are checked by L2: its MDP contains
both a terminal transition and a truncation.

## Run directory

Written by `runlog.RunLogger`. Field definitions are in `runlog.py`.

```
runs/<env>__<tag>__seed<N>__<UTC>/
    config.json  meta.json  summary.json
    episodes.jsonl  updates.jsonl  evals.jsonl
    checkpoints/
```

## Checkpoint structure

This is an outline for the user to implement in step 7b. It is not code.

### Tier 1: model checkpoint (implement)

Purpose: the greedy eval runner and any post-hoc analysis load a trained
network without the training loop. Pre-registration §8 evaluates at end of
training on the held-out set, so at minimum the **final** checkpoint must
exist. `cfg.checkpoint_frequency > 0` adds periodic ones.

```
checkpoints/
    step_000500000/
        params.npz        online network only
        checkpoint.json   metadata (below)
```

- **Directory name:** `step_{global_step:09d}`. Zero-padding makes lexical
  order equal numeric order, so "latest" is just the last name in sorted order.
- **`params.npz`:** one array per parameter. Keys are `W0, b0, W1, b1, ...` in
  layer order, saved with `np.savez`. Load with `allow_pickle=False`: an `.npz`
  that needs pickle is either a bug or not your file.
- **`checkpoint.json`:**
  ```json
  {
    "format_version": 1,
    "global_step": 500000,
    "episode": 2311,
    "updates": 49000,
    "obs_dim": 4,
    "n_actions": 2,
    "param_shapes": {"W0": [4, 120], "b0": [120], "...": "..."},
    "params_sha256": "...",
    "config": { "...the full DQNConfig dict..." }
  }
  ```
  The config is copied in so a checkpoint is self-contained if moved out of its
  run folder. `params_sha256` is the hash of the `.npz` file's bytes.
- **Atomic write:** write everything into `step_XXXXXXXXX.tmp/`, then
  `os.replace` it to the final name. A crash mid-save leaves only a `.tmp`
  directory, which loaders ignore. Without this, a crash produces a
  half-written checkpoint that loads fine but holds garbage.
- **Load contract:** read `checkpoint.json`, then check `format_version` is
  known, the hash matches, and every array's shape matches `param_shapes`.
  Also check those shapes agree with `obs_dim`, `config.hidden_sizes` and
  `n_actions`. Fail loudly on any mismatch.
- **What not to store:** target network, optimiser state, replay buffer. They
  are not needed to act, and leaving them out keeps the file small and the
  format simple.
- **Tests to expect** (written before you implement): save → load gives
  bit-identical arrays and identical greedy actions on a fixed batch of
  observations; a corrupted file fails the hash check; a `.tmp` directory is
  ignored; a shape mismatch raises.

### Tier 2: full training-state checkpoint (recommended: do not build yet)

This is what exact resume-after-crash would need: target params, Adam `m`, `v`
and `t`, replay-buffer arrays with write position and size, every RNG's
`bit_generator.state`, all counters, **and the environment's internal state**.

Reasons to skip it for now:

1. A run is minutes long (~8 min per million combat steps, pre-registration
   §9). Re-running from the same seed reproduces a crashed run exactly, which
   resume cannot promise.
2. Gymnasium environments do not expose serialisable internal state. Resuming
   mid-episode therefore cannot be bit-exact, and the resumed run would differ
   from an uninterrupted one. That is an unrecorded deviation in a
   reproducibility-first project.
3. It is a second code path that needs its own test (straight run vs. resumed
   run, bit-identical), and that test cannot pass because of point 2.

Revisit only if run length grows past roughly an hour or the machine turns out
to be unreliable. If it is built, resume only at episode boundaries, so the env
is freshly reset from a logged seed.
