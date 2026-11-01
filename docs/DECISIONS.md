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
   `default_rng(0)`, and NumPy's default (linear) percentile. **Draw order**
   (confirmed 2026-10-26, before calibration was run on the reference
   data): from one `default_rng(0)`, all group-A indices are drawn first, as
   one `(10000, 10)` array of integers in `[0, 10)`, then all group-B
   indices the same way. Row r of A and row r of B form the r-th pair. F
   and δ are computed from the **same** pairs. This makes the calibration
   reproducible to the last digit (`scripts/gate1a.py`, `bootstrap_pairs`).
4. **Censoring:** a seed that never reaches T counts as `total_timesteps`
   (500,000) inside the bootstrap median-steps ratios.

**Ownership:** the user writes the calibration and the L3/L4 check, including
their plumbing. Claude writes their tests.

The calibrated values are appended to this entry with their date **before** any
full NumPy-DQN CartPole run. Short smoke runs during development do not count as
outcome data for this gate, but their results must not be used to set any number
above.

### Calibrated values (2026-10-30)

Computed by `python scripts/gate1a.py calibrate --reference runs/gate1a/cleanrl
--out docs/gate1a_calibration.json` (code at `eb0dbd8`), from the 10 CleanRL
reference runs, seeds 1-10, completed 2026-10-20 at `4775283`. Run folders are
listed in `docs/gate1a_calibration.json`. No full NumPy-DQN CartPole run
existed when these were computed.

| Number | Value | How it came out |
|---|---|---|
| T | **475** | The default. At least 8 of 10 reference seeds reached it, so the lowering rule did not apply. |
| *k* | **8** | All 10 reference seeds reached T, minus 2. |
| q | 1.0614 | The 95th percentile of the bootstrap median-steps ratio. |
| F | **2** | max(2, q): the floor applies. |
| δ | **14.4008** | The 95th percentile of the bootstrap IQM difference in final return. |

**Gate 1a pass criteria for the NumPy DQN**, fixed from here on:

- **L3:** at least 8 of 10 seeds reach a 100-episode moving average of 475.
- **L4 steps:** median steps-to-475 ≤ 2 × the reference median. Fails
  automatically if more than 5 seeds are censored.
- **L4 final:** IQM of final return ≥ the reference IQM − 14.4008.

### Attempts

Every Gate 1a attempt is logged with its date, commit, levels run and result,
including failures. Fixing a defect and re-running is the intended use of the
gate. Re-running unchanged code with new seeds until it passes is not.

| Date | Code commit | Level | Result |
|---|---|---|---|
| 2026-10-20 | `531f33d` | L1 | PASS. All component tests (steps 1–6, and step 7a's `gradient_update` against the same calls made by hand). Loop composition against a stub env: `tests/test_train.py`, 34 tests. |
| 2026-10-20 | `531f33d` | L2 | PASS, first run. `tests/test_l2_chain.py`: 3-state quit chain with a terminal action and a 4-step truncation, γ = 0.9. Q\* derived by hand by the user (S: 8.1 / 1, A: 9 / 0, B: 10 / 0), cross-checked by value iteration. Tolerance ±0.5 and config fixed before the first run. Seeds 1, 2, 3: max \|Q − Q\*\| = 0.001, 0.003, 0.001. Before this run, planted bugs in a scratch copy each failed by ≥ 4.5: masking on truncation, never storing `terminated`, storing the reset observation as s′, never syncing, wrong γ. |
| 2026-10-30 | `1c0c0ed` | L3, L4 | **PASS, first attempt.** 10 NumPy-DQN runs (seeds 1-10, 500k steps, `runs/gate1a/numpy/`, all `completed`, clean tree at the calibration commit) against the 10 reference runs, with the calibration fixed beforehand. **L3:** 10 of 10 seeds reach T = 475 (need k = 8). **L4 steps:** median steps-to-T 268,244 vs reference 273,698 (allowed: 2 × = 547,395); 0 censored. **L4 final:** IQM final return 495.76 vs reference 497.85 (allowed: ≥ 483.44). `scripts/gate1a.py check`. **Gate 1a passed: L1-L4.** |
| 2026-11-01 | `c3cbd0e` | L3, L4 | **PASS, attempt 2 — NumPy 2.5.4 (D033).** Ten seeds re-run under the upgraded interpreter into `runs/gate1a/numpy_np2/`, checked against the **unchanged** calibration. Every metric reproduced exactly: L3 10/10 reach T; L4 steps median 268,244 vs 273,698; L4 final IQM 495.76 vs 497.85. Stronger than a re-pass: all ten `episodes.jsonl` logs are **bit-identical** (SHA-256) to the 1.26.4 runs, so the agent's trajectory is byte-for-byte reproducible across a major NumPy version bump. |

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

---

## D032 — EEGMAT verification: usable, but its rest/task contrast is void

**Status:** CONFIRMED (2026-11-01)

`scripts/verify_eegmat.py`, 3 subjects, frontal channels (F3, F4, Fz),
Welch PSD over 4-second windows, no preprocessing.

**Question 1 — is there real power below 30 Hz? PASS.** Power below 30 Hz
exceeds power above it by 41.3×. The PhysioNet page's "high-pass filter with a
30 Hz cut-off frequency" is a typo for *low-pass*. The dataset retains theta
and alpha and is usable for this project's index. This question is closed.

**Question 2 — originally "does frontal theta rise from rest to task?" VOID BY
DESIGN.** EEGMAT's rest recording is **eyes-closed** (Zyma et al. 2019: 180 s
eyes-closed rest, 60 s eyes-open serial subtraction). Rest and task therefore
differ in eye state, visual input and arousal as well as in workload, so a
theta difference across that contrast cannot be attributed to cognitive load.

The observed direction (theta rose in 1 of 3 subjects) is **not** evidence
against the theta term of the index, and running the remaining 33 subjects
would not make it so: the defect is bias, not variance. More observations on a
confounded design narrow the confidence interval around the wrong quantity.
Recorded explicitly because the opposite reading — "theta failed to replicate"
— would have been a reason to weaken or abandon the primary term of the index.

**Question 2 rewritten as a positive control.** Eyes-closed → eyes-open
suppresses alpha (the Berger effect). Mean frontal alpha fell from 3.308e-11
to 2.785e-11, −16%, the expected direction; the magnitude is modest because
alpha is weakest frontally. This validates the loading, channel-selection and
PSD code against real physiology without claiming anything about workload.

**Consequence for the experiment.** Workload validation for the index must come
from a contrast holding eye state constant. The 25 combat configurations do:
the participant is eyes-open, visually engaged and performing the same task
throughout, with only telegraph duration and attack gap varying. This is a
cleaner workload manipulation than EEGMAT offers, and should be stated as such
in the report.

**Retained use.** EEGMAT remains the development fixture for the EEG pipeline:
36 subjects of 21-channel, 500 Hz EDF for building and unit-testing loading,
filtering, epoching, artifact detection and band extraction before any of our
own recordings exist.

**Reference:** Zyma, I., Tukaev, S., Seleznov, I., et al. (2019).
Electroencephalograms during Mental Arithmetic Task Performance. *Data*,
4(1), 14.

---

## D033 — Pin the numerical environment; re-prove Gate 1a on NumPy 2.x

**Status:** CONFIRMED (2026-11-01)

**What happened.** Installing `mne` for D032 upgraded NumPy in the Anaconda
base environment from 1.26.4 to 2.5.4. Gate 1a passed under 1.26.4
(`runs/gate1a/numpy/*/meta.json`), so the environment that produced the passing
gate no longer exists. The drift was detected only because
`experiment/manifest.py` writes Python, NumPy and platform into every run.

**Why it matters.** The agent is implemented in NumPy, so NumPy is a dependency
of the numerical results, not merely of the tooling. Reduction and BLAS paths
changed between 1.26 and 2.5, and NumPy 2.0 removed a number of aliases, so
results are not guaranteed bit-identical and code may change behaviour silently.
`np.random.default_rng` (PCG64) is stream-stable across versions by documented
policy, so seeded action sequences are unaffected.

**Decision.** Adopt NumPy 2.x as the project version rather than reverting.
Reverting would restore a version that nothing else in the toolchain now wants,
and the gate is cheap to re-prove: the ten 500k-step CartPole seeds complete in
parallel in 3.6 minutes. Re-running converts an accidental drift into a
deliberate, reportable result — the gate holds across a major version bump.

**Procedure.**

1. Commit the working tree so `meta.json` records a real commit.
2. Re-run the ten NumPy-DQN seeds under 2.5.4 into `runs/gate1a/numpy_np2/`.
3. Re-run `scripts/gate1a.py check` against the **unchanged** calibration in
   `docs/gate1a_calibration.json`. The calibration is not recomputed: it was
   fixed before any outcome data and stays fixed. This is a second attempt
   against the same thresholds, not a new gate.
4. Append the attempt to D029's table with its date and commit.
5. `python -m pip freeze > requirements-lock.txt` from the environment that
   produced the passing runs, and commit it.

**If attempt 2 fails,** that is a real finding and is recorded as such. It would
mean the DQN depends on NumPy 1.x numerical behaviour, which is itself worth
knowing before the combat experiments. It is not a licence to revert and
pretend the first pass still stands.

**Verification result:** CONFIRMED (2026-11-01). Attempt 2 passed at commit
`c3cbd0e` under NumPy 2.5.4, recorded in D029's attempts table. All ten
`episodes.jsonl` logs hash identically (SHA-256) to the NumPy 1.26.4 runs, so
the version change altered nothing: not the PCG64 action stream, and not the
float64 arithmetic downstream of it. The concern that motivated this entry —
that reduction or BLAS changes between 1.26 and 2.5 might perturb results —
did not materialise at this network size and dtype. It was still worth
checking, and worth reporting: the claim is now measured rather than assumed.

`requirements-lock.txt` pins the environment that produced the passing runs.
Any future NumPy change repeats this procedure rather than assuming the result
carries over.

---

## D034 — Gate 1: operational criteria

**Status:** DECIDED (2026-11-01); metric amended by D035 same day; floors and attempts PENDING

Pre-registration §10 defines Gate 1 in prose: "a non-EEG baseline agent learns
the task to non-trivial performance, reproducibly across seeds, and its
per-configuration success rates are ordered rather than uniform." That is not
yet testable. This entry fixes the measurement and the thresholds **before any
combat training run**, for the same reason D029 did for Gate 1a: a threshold
chosen after seeing the outcome is not a threshold.

### What the baseline agent is

One DQN trained on a **uniform random mixture of all 25 configurations** — a
configuration is drawn uniformly at reset — then evaluated on each
configuration separately. One agent, not 25.

Chosen because §10 says "**its** per-configuration success rates", singular,
and because it matches the human protocol: a participant meets all 25
configurations in one session carrying general combat skill, and never relearns
from scratch. Twenty-five separately trained agents would measure learnability
from scratch, which is a different construct and a weaker match for the Gate 2
correlation. It is retained as an optional secondary analysis, not the primary.

**Stated openly:** a uniform mixture is operationally close to the RANDOM
experimental condition, so the Gate 2 reference agent is approximately the
random-curriculum agent. This is accepted deliberately — RANDOM is the natural
neutral reference, and no EEG information enters Gate 1 or Gate 2 — but it must
not be presented later as a condition-independent measurement.

### Measurement

- **Mean damage taken** for a configuration (primary, D035): the mean of
  `EpisodeStats.damage_taken` over `N_eval = 200` evaluation episodes, under
  the greedy policy (ε = 0), on fixed evaluation seeds disjoint from the
  training seeds. Higher means harder.
- **Success rate** (secondary, descriptive only): the fraction of those
  episodes ending with outcome `win`. Reported, never used to rank.
- **Saturation**: the fraction of episodes in which damage taken reaches agent
  maximum health, reported per configuration.
- **Seeds:** 10 training seeds (1–10), matching D029.
- **Budget:** 500k environment steps per seed, revisited only if G1-a fails
  for budget reasons rather than learning reasons.

### Floors, measured before training

Both floors are evaluated on the same 25 configurations with the same
`N_eval` and the same evaluation seeds:

- `F_random` — uniform random action selection.
- `F_scripted` — the existing `scripts/axis_sweep.py` policy.

Floor values are appended to this entry with their date **before** the first
training run, and are not recomputed afterwards.

### Pass criteria

- **G1-a (non-trivial).** For a seed to pass, its mean damage taken across the
  25 configurations must be **≤ 0.75 × `F_random`'s mean** — a reduction of at
  least 25% relative to random action selection. Expressed as a ratio rather
  than an absolute, because damage taken is on a 0–100 scale whose floor value
  is a property of the environment, not of the agent. Fixed in advance; a 25%
  reduction is unambiguously learning rather than seed noise.
- **G1-b (reproducible).** **≥ 8 of 10** seeds satisfy G1-a. k = 8 by analogy
  with D029, deliberately reused so the two gates demand the same consistency.
- **G1-c (ordered, not uniform).** The observed spread in per-configuration
  mean damage taken (max − min, pooled across passing seeds) must exceed the
  **95th percentile** of a null distribution built by permuting configuration
  labels across evaluation episodes, 10,000 permutations, seed 0. A permutation test
  is used instead of a fixed spread threshold because it needs no invented
  constant and reuses the machinery of the §9 analysis plan.

All three must hold. A pass releases the Stage-2 amendment: the between-seed
variance in the primary dependent variable, measured here, sets the seed count
and smallest detectable effect size.

### On failure

Gate 1a has already cleared the algorithm, so a Gate 1 failure is isolated to
the environment, the observation space, or the provisional reward (OPEN-06).
Diagnosis proceeds in that order. Any reward change is a pre-registration §12
deviation and invalidates cross-experiment comparison with runs made before it.
Re-running unchanged code with new seeds until it passes is not a diagnosis.

**Floors:** RECORDED 2026-11-01, commit `c3cbd0e`, NumPy
2.2.6, n_eval = 200, evaluation seeds
1000000–1000199, random-policy
seed 0. Full per-configuration values in
`docs/gate1_floors.json`.

| floor | mean damage taken | spread | mean win rate |
|---|---|---|---|
| random | **80.5196** | 61.9365 | 0.6450 |
| scripted | 46.1333 | 85.7800 | 0.9592 |

**G1-a threshold, now fixed: mean damage taken ≤ 0.75 × 80.5196 =
60.3897.** The scripted policy reaches 46.1333, so the bar is
attainable by a competent policy without being trivial — but the scripted
policy's value is context, not a criterion, and the threshold stands whatever
the agent turns out to score.

The scripted floor is monotone in both axes across effectively the whole grid
(11.16 at the easiest configuration to
96.94 at the hardest), which is the ordering
property Gate 2 depends on.

**Saturation flags.** Under the D035 rule (>25% of episodes reaching the health
bound), the scripted floor flags configurations 19, 24
— the two hardest cells. Every other configuration is at or near 0%. A trained
agent should saturate less, but if these configurations stay flagged their
ranks are treated as unreliable in the Gate 2 correlation rather than silently
included.
**Attempts:** PENDING — one row per attempt, as in D029.


---

## D035 — Gate 1/Gate 2 ranking metric: mean damage taken

**Status:** CONFIRMED (2026-11-01)

Pre-registration §6 originally ranked configurations by **success rate** for
both Gate 1 and Gate 2. The Gate 1 floors (D034), measured before any agent was
trained, showed that this cannot work.

### The defect

| policy | mean win rate | configs at exactly 1.000 | distinct values |
|---|---|---|---|
| random | 0.645 | 0 / 25 | 21 / 25 |
| scripted | 0.959 | **19 / 25** | **7 / 25** |

Random action selection produces a clean monotone gradient across both axes
(0.995 at the easiest configuration to 0.105 at the hardest). A merely
competent policy does not: it wins 19 of 25 configurations outright, collapsing
25 configurations into 7 distinguishable levels.

The trained agent should exceed the scripted policy, and the human participant
certainly will. Gate 2's Spearman correlation would then have been computed
between two vectors that are approximately three-quarters ties — on the human
side, potentially entirely tied. The project's central correspondence claim
would have rested on a near-degenerate statistic.

### Candidates evaluated

Four quantities available from `EpisodeStats`, both floor policies, 120
episodes per configuration. "Ratio" is the telegraph-axis effect divided by the
gap-axis effect; 1.00 means the two difficulty axes contribute equally. ρ is
Spearman against a monotone difficulty proxy.

Competent (scripted) policy:

| candidate | distinct | telegraph effect | gap effect | ratio | ρ |
|---|---|---|---|---|---|
| success rate | 7/25 | −0.130 | −0.192 | 0.68 | −0.195 |
| **mean damage taken** | **22/25** | **44.16** | **43.86** | **1.01** | **+0.938** |
| hit rate | 24/25 | 0.419 | −0.007 | −58.0 | +0.296 |
| episode length | 25/25 | 4.81 | 65.82 | 0.07 | +0.651 |

### Decision

**Mean damage taken.** It is the only candidate that discriminates for a
competent policy *and* responds to both difficulty axes. For the scripted
policy the axes contribute 44.16 and 43.86 — a ratio of 1.01, the axis balance
OPEN-02 was concerned with, achieved. It is also the quantity D026 calibrated
the axes on, so this is consistency with an existing decision rather than a new
invention, and it is recorded in `EpisodeStats` for human and agent alike,
which is what §6's comparability argument requires.

### Why hit rate was rejected

Recorded because it was proposed first, and the reasoning that produced it was
wrong in an instructive way.

`damage ≈ attacks_faced × fraction_connecting`. Attack gap controls the first
term mechanically; telegraph duration controls the second. Hit rate *is* the
second term alone, which is exactly why D026 used it: to test whether telegraph
was doing anything independent of gap.

That property makes it useless here. A metric built to isolate one axis cannot
represent two. The measurement confirms it: hit rate's gap-axis effect is
**−0.007**, indistinguishable from zero. Adopting it would have made attack gap
a dead axis — the OPEN-02 failure mode reintroduced through the metric rather
than the environment, and invisible until Gate 2 produced a weak correlation
for reasons nobody could locate.

The general lesson, worth carrying into the report: a diagnostic designed to
isolate a factor is the worst available choice for a measure that must
represent all factors. It was caught only because the candidates were measured
rather than argued about.

### Consequences

- Pre-registration §6 and §10 amended; §12 deviation logged, outcome data seen:
  No.
- Direction inverts: higher damage taken means harder, where higher success
  rate meant easier. Expected correlation signs are stated per hypothesis.
- Damage taken is effectively bounded by agent maximum health: the episode ends
  at death, though the fatal blow may overshoot the bound (the random floor
  records 106.65 at the hardest configuration). The fraction of episodes
  reaching the bound is reported per configuration; above 25% the rank is
  flagged unreliable.
- Damage taken's status in §8 as an **exploratory** dependent variable is
  unchanged. Ordering configurations and measuring whether the curriculum
  helped are different roles; the primary dependent variable remains mean
  normalized return on the held-out set.

### Open, not decided here

Conditions `PERF` and `BEHAV` are still defined in §4 by success rate — the
agent's running per-configuration success rate, and human success rate with
median reaction time. They face the same ceiling, and consistency argues they
should follow this change. They are part of the **independent variable**, so
changing them is a larger step than changing a gate's metric and is not made
here. Resolve before Gate 2.
