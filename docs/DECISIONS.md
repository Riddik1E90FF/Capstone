# Decision Log

This document records project decisions and prevents recommendations from silently becoming requirements.

Status labels:

- **CONFIRMED** — explicitly established
- **TENTATIVE** — strongly favored but not final
- **PROPOSED** — new recommendation awaiting user/advisor approval
- **OPEN** — unresolved
- **REJECTED** — intentionally not used

---

## D001 — Genuine neuroscience component

**Status:** CONFIRMED

The project must include a genuine neuroscience/neurotechnology component.

A conventional artificial neural network does not qualify merely because it is called a neural network.

---

## D002 — EEG as candidate neural interface

**Status:** CONFIRMED

EEG is the intended neural-data modality.

Specific hardware remains open.

---

## D003 — EEG influences the environment/curriculum

**Status:** CONFIRMED

The central architecture is:

```text
EEG → state estimation → environment/curriculum modulation → RL
```

EEG should not directly control the agent's actions.

---

## D004 — No privileged human-state information to the agent

**Status:** CONFIRMED

The RL agent should not receive EEG, inferred human state, or equivalent privileged information as part of its observation/action/reward interface.

The agent learns from the resulting environment.

---

## D005 — ARPG-style environment

**Status:** CONFIRMED

The environment is intended to represent ARPG-style combat.

Exact implementation is open.

---

## D006 — Smallest viable experiment first

**Status:** CONFIRMED

The project should not begin by building a large Skyrim-like environment.

A minimal combat environment is preferred.

---

## D007 — Public EEG datasets for early development

**Status:** CONFIRMED

Public EEG datasets are an acceptable and preferred way to de-risk EEG processing before hardware procurement.

---

## D008 — Artifact handling is first-class

**Status:** CONFIRMED

Ocular, muscular, movement, and other physiological artifacts must be treated as a core engineering issue.

---

## D009 — Natural movement should be tolerated

**Status:** CONFIRMED

The system should not require continuous suppression of natural movement/ocular behavior.

The preferred design principle is:

> Natural behavior → artifact detection/handling → robust processing.

---

## D010 — Participant-specific movement is not automatically cognitive load

**Status:** CONFIRMED

Participant-specific motor/ocular behavior must not automatically be interpreted as cognitive load.

If such signals are studied, their interpretation requires independent validation.

---

## D011 — Human role during EEG recording

**Status:** OPEN / STRONGLY RECOMMENDED

Three architectures were considered:

A. Human plays the environment.
B. Human watches the agent.
C. Human and agent train in a live coupled loop.

Architecture A is strongly favored because it supports replay, reproducibility, many RL seeds, and simpler controls.

It is not yet a final decision.

---

## D012 — EEG replay architecture

**Status:** PROPOSED

Create an EEG source abstraction supporting:

- recorded replay
- synthetic data
- live stream

This should be designed early even if live EEG is not initially available.

---

## D013 — Custom simulator

**Status:** PROPOSED

A small custom Python simulator is currently favored over direct Skyrim integration or a large game engine because it can provide fast simulation, headless training, seed control, and precise environment parameters.

This must be evaluated against capstone requirements before adoption.

---

## D014 — RL algorithm

**Status:** SUPERSEDED by D028 (2026-10-07). Original text kept as history.

~~OPEN~~

DQN, PPO, and actor-critic approaches have been discussed.

No algorithm has been selected.

Selection should follow action-space/environment design and empirical benchmarking.

---

## D015 — Neural-state target

**Status:** OPEN

Candidates include:

- task difficulty
- cognitive effort
- engagement
- another experimentally validated neural/physiological correlate

No final target has been selected.

---

## D016 — Static/random/neuroadaptive experiment

**Status:** TENTATIVE / STRONGLY FAVORED

The experimental backbone is expected to compare:

- static curriculum
- random curriculum
- neuroadaptive curriculum

---

## D017 — Shuffled EEG control

**Status:** PROPOSED

A shuffled-EEG condition could test whether temporal structure or human-specific information contributes to any observed effect.

Not yet formally adopted.

---

## D018 — Agent-performance-adaptive baseline

**Status:** PROPOSED

An agent-performance-based curriculum is a useful additional baseline because it tests whether EEG provides value beyond a difficulty signal derived from the agent itself.

Not yet formally adopted.

---

## D019 — Primary dependent variable

**Status:** OPEN

Candidates include:

- generalization
- sample efficiency
- asymptotic performance
- learning speed
- robustness

Generalization is currently a strong recommendation, not a decision.

---

## D020 — Seed count

**Status:** OPEN

RL variance requires multiple seeds, but the final number should be determined from compute budget, pilot variance, effect-size expectations, and the capstone timeline.

---

## D021 — Game engine

**Status:** OPEN

Unity, Unreal, Godot, custom simulation, and Skyrim integration have been discussed.

None is selected.

---

## D022 — EEG hardware

**Status:** OPEN

OpenBCI and Emotiv were discussed as candidates.

No device is selected.

---

## D023 — EEG features

**Status:** OPEN

Frequency bands, bandpower, time/frequency features, and other features were discussed.

No final feature set is selected.

---

## D024 — ICA

**Status:** PROPOSED / OPEN

ICA was discussed as one possible artifact-handling technique.

It is not mandatory.

---

## D025 — Scientific language

**Status:** CONFIRMED

Use conservative language.

Preferred:

- EEG-derived feature
- model-estimated state
- neural/physiological correlate

Avoid:

- reads thoughts
- detects exact emotions
- objectively measures subjective mental state

---

## D026 — Difficulty axis calibration

**Status:** CONFIRMED (2026-10-05)

Telegraph duration levels widened from (0.70, 0.55, 0.42, 0.32, 0.24) s to
(1.00, 0.66, 0.42, 0.24, 0.12) s.

The original range moved agent hit rate by 0.016 across its whole span, which is
nothing. The widened range moves it by 0.205. Measured by `scripts/axis_sweep.py`.

Dodge invulnerability was swept over the same sweep and showed no measurable
effect; it is left at 0.20 s rather than changed without evidence.

Closes `OPEN-02`. Residual imbalance and the saturation limitation are documented
in `docs/ENVIRONMENT_SPEC.md` section 6.

---

## D027 — Reverse-ordered curriculum condition

**Status:** CONFIRMED (2026-09-16)

A seventh condition, `REVERSE`, inverts the EEG-derived difficulty ranking.

Motivated by Moravec's paradox: perception and reaction-speed demands that
dominate human difficulty may be trivial for an agent, so the most useful
human-derived curriculum could be the inverted one. The measured axis asymmetry
in this environment is consistent with that.

It is a deterministic transform of existing data and requires no additional
collection. Its value is converting a possible null into an informative result:
if forward ordering ties the shuffled control while reverse beats it, human
difficulty transfers in inverted form.

Pre-registered as H5, inside the Holm-corrected contrast family.

---

## D028 — From-scratch RL implementation, NumPy on CPU

**Status:** CONFIRMED (2026-09-16, backend measured 2026-09-23)

The DQN is implemented from scratch in NumPy. No Stable-Baselines3, no PyTorch
dependency for the agent.

Backend selection was measured rather than assumed. DQN update throughput for an
18-128-128-12 network at the batch size training actually uses (64):

| Backend | Updates/sec |
|---|---|
| NumPy (CPU) | 2,118 |
| PyTorch (CPU) | 645 |
| PyTorch (CUDA) | 292 |

CUDA is 7.3x slower than NumPy at batch 64, and its throughput is nearly flat
from batch 64 to 4096 — the signature of a launch-bound rather than compute-bound
workload. The GPU does not overtake PyTorch-CPU until roughly batch 1024, sixteen
times larger than DQN uses.

Caveat: the NumPy implementation used plain SGD while PyTorch used Adam, so part
of the CPU gap is optimiser cost rather than framework overhead.

Correctness validation (Gate 1a) is a prerequisite for Gate 1: component unit
tests, a benchmark task solved to its known threshold, and a reference
implementation's learning curve tracked on that task — all before the agent
touches the combat environment.

Measured by `scripts/bench_backends.py`.

---

## D029 — Gate 1a: benchmark, reference, and pass criteria

**Status:** CONFIRMED (2026-10-07) for the four-level structure, benchmark,
reference and dependency boundary. The criterion details that departed from the
user's first draft (10 seeds, non-inferiority instead of CI overlap, the F and δ
calibration rules) were CONFIRMED 2026-10-13. Calibration **numbers** are
**PENDING**: set from reference runs, before any full NumPy-DQN CartPole run.

Operationalises pre-registration §10, Gate 1a, which requires "tracks a reference
implementation's learning curve" but never defines *tracks*. Without a definition
written in advance, the check gets judged by eye after the curves exist. This is
an algorithm-correctness check on a benchmark task. It involves no combat-
environment data and no experimental-condition data, so it does not touch the
Stage-1 lock. It is recorded here and in §12 for provenance.

### Benchmark and reference

- **Benchmark:** Gymnasium `CartPole-v1`, with `gymnasium==0.29.1` (the version
  CleanRL pins). A self-written benchmark could share the bug it exists to
  detect, and the published threshold (475) only means something on the
  canonical implementation. CartPole-v1 truncates at 500 steps, which exercises
  the terminated/truncated distinction the combat env's 40 s timeout also has.
- **Reference:** CleanRL `cleanrl/dqn.py` at commit
  `fe8d8a03c41a7ef5b523e2e354bd01c363e786bb`, run **unmodified** with its own
  default hyperparameters. Its output is converted to our log format after the
  run, not by editing the script.
- **Matched setup.** The NumPy DQN matches the reference wherever the reference
  makes a choice: 120-84 ReLU MLP, PyTorch-default initialisation, Adam
  (lr 2.5e-4, β = 0.9/0.999, ε = 1e-8), MSE TD loss, hard target sync every 500
  steps, buffer 10,000, batch 128, γ = 0.99, learning starts at step 10,000,
  train every 10 steps, ε-greedy annealed linearly from 1.0 to 0.05 over the first
  250,000 of 500,000 steps, and only `terminated` (never `truncated`) masking the
  bootstrap.
- **Dependency boundary (does not conflict with D028).** Gymnasium, PyTorch and
  CleanRL's dependencies are dev/validation-only (`requirements-dev.txt`). The
  `dqn` package imports neither `gymnasium` nor `torch`, and a test enforces
  this. Environments reach the agent through an injected factory that follows the
  `reset`/`step` protocol `CombatEnv` already implements.

### The four levels

All four levels must pass, in order, before the agent touches the combat
environment.

| Level | Tests | Oracle | Catches |
|---|---|---|---|
| **L1** Component unit tests | Each component against its agreed interface spec, in isolation | The spec. Tests are written before the implementation, by a different author. Finite-difference gradients; `torch.optim.Adam` (dev-only) for the optimiser | A wrong component |
| **L2** Analytic-oracle integration | The full training loop, with real buffer, target net and schedule, on a small MDP with hand-derived Q\*. The MDP contains both a terminal transition and a truncation | Closed-form Q\*, derived on paper. Tolerance fixed in the test before it is run | Wrong composition: step ordering, episode-boundary handling, terminated/truncated routing through loop and buffer, γ, sync cadence |
| **L3** Benchmark threshold | NumPy DQN on CartPole-v1 | The published threshold, with the calibration rule below | "Doesn't learn" |
| **L4** Reference tracking | NumPy DQN against CleanRL, same protocol | Reference runs | "Learns, but wrongly": slower, noisier, or collapse-prone |

**L1 and L2 stay separate levels.** L2 is run by pytest, but its oracle is
independent mathematics, not a component spec. A loop built only from
L1-correct components can still fail L2. Examples: storing the post-reset
observation as `s'` on a truncated step, or syncing the target network before
the first update instead of after.

**L3 is nearly nested in L4.** Steps-to-threshold is only defined if the
threshold is reached. L3 is kept because it is the pre-registered absolute
criterion and is cheap to check first.

### Protocol

- **Per-seed metrics**, computed from per-episode training returns (exploration
  on, as CleanRL logs them):
  - *Steps-to-threshold:* the first `global_step` at which the 100-episode moving
    average reaches T. A seed that never reaches it is censored at
    `total_timesteps`.
  - *Final return:* the mean return of the last 100 training episodes.
- **Seeds.** **10 reference seeds and 10 NumPy-DQN seeds** (integers
  1–10), not the drafted 5. With 5 seeds an IQM is roughly the mean of the middle
  two or three values, and a bootstrap over 5 values has only 126 distinct
  resamples, too coarse for a CI. CartPole compute is negligible.
- **L3 pass:** the NumPy DQN reaches T in at least *k* of 10 seeds.
- **L4 pass, part 1:** median steps-to-threshold for the NumPy DQN ≤ F × the
  reference median. If more than half of the NumPy seeds are censored, the median
  is undefined and this part fails.
- **L4 pass, part 2.** **One-sided non-inferiority on the point
  estimate**, replacing the drafted "final-return IQM CIs overlap". Pass if
  IQM(NumPy) ≥ IQM(ref) − δ. Reason: CI overlap gets *easier to pass as variance
  grows*, and excess seed variance is the signature of the defects Gate 1a exists
  to catch. Beating the reference is not treated as a defect.

### Numbers: rule fixed now, values after reference runs

| Number | Fixed now | Set from reference runs by this rule |
|---|---|---|
| F (steps factor) | **2** | F = max(2, q), where q is the 95th percentile of the median-steps ratio between two bootstrap-resampled 10-seed groups of the reference itself. This stops the reference failing its own criterion. |
| T (threshold) | 475 by default | If fewer than 8 of 10 reference seeds reach 475, T = the highest multiple of 25 that ≥ 8 of 10 reference seeds reach. |
| *k* (L3 seed count) | — | The number of reference seeds reaching T, minus 2. |
| δ (final-return margin) | — | The 95th percentile of IQM(A) − IQM(B), where A and B are bootstrap-resampled 10-seed groups of the reference. A correct implementation then passes about 95% of the time. |

**Metric definitions** (CONFIRMED 2026-10-20 by the user, before any
reference result was looked at; the 10 reference runs were in progress):

1. **100-episode moving average:** the mean return of the latest 100 training
   episodes, first defined at the 100th episode, with no partial windows. A
   seed *reaches* T if this average is ≥ T at some episode. Steps-to-threshold
   is the `global_step` of the first such episode.
2. **IQM:** the mean of the middle 50% of the 10 per-seed values. Drop the 2
   lowest and the 2 highest, then average the remaining 6. This is
   `scipy.stats.trim_mean(x, 0.25)`, the rliable convention.
3. **Bootstrap:** 10,000 resamples, with replacement, NumPy
   `default_rng(0)`, and NumPy's default (linear) percentile.
4. **Censoring:** a seed that never reaches T counts as `total_timesteps`
   (500,000) inside the bootstrap median-steps ratios.

**Ownership:** the user writes the calibration and the L3/L4 check, including
their plumbing. Claude writes their tests.

The calibrated values are appended to this entry with their date **before** any
full NumPy-DQN CartPole run. Short smoke runs during development do not count as
outcome data for this gate, but their results must not be used to set any number
above.

### Attempts

Every Gate 1a attempt is logged with its date, commit, levels run and result,
including failures. Fixing a defect and re-running is the intended use of the
gate. Re-running unchanged code with new seeds until it passes is not.

| Date | Code commit | Level | Result |
|---|---|---|---|
| 2026-10-20 | `531f33d` | L1 | PASS. All component tests (steps 1–6, and step 7a's `gradient_update` against the same calls made by hand). Loop composition against a stub env: `tests/test_train.py`, 34 tests. |
| 2026-10-20 | `531f33d` | L2 | PASS, first run. `tests/test_l2_chain.py`: 3-state quit chain with a terminal action and a 4-step truncation, γ = 0.9. Q\* derived by hand by the user (S: 8.1 / 1, A: 9 / 0, B: 10 / 0), cross-checked by value iteration. Tolerance ±0.5 and config fixed before the first run. Seeds 1, 2, 3: max \|Q − Q\*\| = 0.001, 0.003, 0.001. Before this run, planted bugs in a scratch copy each failed by ≥ 4.5: masking on truncation, never storing `terminated`, storing the reset observation as s′, never syncing, wrong γ. |

Development smoke runs (not attempts, and not used for any calibration
number): 2026-10-20, 50k-step CartPole-v1 and 20k-step `combat:12` runs to
check that the CLI path executes end to end.

---

## D030 — Remove `block_move_scale` from `EnvParams`

**Status:** CONFIRMED (2026-10-14)

`EnvParams.block_move_scale` (0.35) and its one call site in
`CombatEnv._apply_agent_action` were removed.

**Why it was dead code, not an oversight.** The action space is discrete, with
one action per decision (D-pad style): idle, 8 moves, attack, dodge, block.
`is_blocking` is set to `action == BLOCK`, and the movement branch runs only for
move actions. A single action cannot be both a move and a block, so the agent
can never block and move on the same tick, and the blocking-movement penalty
could never apply. Do not re-add it unless the action space changes to allow
simultaneous block-and-move, e.g. a factored or multi-discrete action space.
That would be an environment redesign requiring a §12 deviation.

**Verified, not assumed.** Behaviour is byte-identical:

- `scripts/axis_sweep.py` output diffed before and after: empty.
- A behavioural fingerprint over all 25 configurations × 5 seeds with a random
  policy matched exactly. It hashes every observation, reward, termination flag,
  info dict, final episode stats and end-of-episode RNG state.

No pre-registration deviation is needed, since no behaviour changed.

**`exhausted_penalty_scale` is deliberately kept.** It looks similar but is not
structurally unreachable. Every stamina cost is gated on `stamina >= cost`, so
stamina never goes negative. It can still reach exactly 0.0 if a blocked hit
(cost 18) lands with stamina at exactly 18.0, since blocking also suppresses
regen. The next move would then be slowed. Landing on exactly 18.0 requires a
floating-point coincidence among regen steps of 26/30. It was observed zero
times in about 258k movement ticks, but it is not impossible, so it stays.

---

## D031 — Simultaneous death resolves as a loss

**Status:** CONFIRMED (2026-10-14)

`CombatEnv._resolve_termination` now checks agent death before enemy death. A
tick on which both reach 0 health ends with outcome `loss`: −30 loss penalty, no
win bonus, not a success.

**Before:** enemy death was checked first, so a mutual KO counted as a `win`.
That paid +60 on top of the damage penalty for the hit taken, and counted
toward per-configuration success rates. An instrumented probe found 4 such
episodes in 1,000: 25 configurations × 20 seeds × 2 policies.

**Why loss:**

- A value learner should not learn that trading its own life for the kill is
  rewarded as a win.
- "You died" is also what a human player sees, which matters for the
  human/agent difficulty comparison at Gate 2.
- The alternative, a separate `draw` outcome, was rejected: it adds a fourth
  outcome value that the success-rate definitions, tests and scripts would all
  need to handle.

**Impact:** a change to environment behaviour, and through the terminal bonus
to the provisional reward (OPEN-06), so it is recorded in pre-registration §12.
`scripts/axis_sweep.py` output and the 125-episode fingerprint were unchanged,
because none of those episodes contain a mutual KO. No agent-training,
human-session or experimental-condition data existed when it was made.
Regression test: `tests/test_env.py::test_simultaneous_death_is_a_loss`.
