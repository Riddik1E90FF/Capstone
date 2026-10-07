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

**Status:** OPEN

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
