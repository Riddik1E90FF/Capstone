# Capstone Roadmap

**Status:** Living development plan  
**Status as of 2026-11-01 (week 4 of 10):** Phases 0 and 1 complete. The DQN is
implementation-verified (Gate 1a, L1–L4, D029) but has never been trained on the
combat environment. The EEG half is the critical path: hardware is assembled but
has not been powered on, and no preprocessing pipeline exists.

---

# Guiding principle

Build the **minimum viable scientific experiment** before building the impressive game.

The project succeeds if the research question can be tested rigorously. A large game is optional; a reproducible experiment is not.

---

# Phase 0 — Research and architecture  ✅ COMPLETE

## Goal

Remove the decisions that would invalidate or force major rewrites later.

### Tasks

- [x] Confirm the human EEG-recording architecture.
- [x] Define exactly what the human does during recording.
- [x] Define the candidate neural/physiological state.
- [x] Determine how that state can obtain defensible labels/ground truth.
- [x] Define what "improvement" means.
- [x] Select primary and secondary dependent variables.
- [x] Define the independent variable.
- [x] Define candidate controls/baselines.
- [x] Define the smallest viable combat task.
- [x] Determine whether a custom simulator is appropriate.
- [x] Determine institutional ethics/IRB requirements.
- [x] Survey public EEG datasets.
- [x] Create an initial preregistration/research plan.

### Gate

Do not move into large implementation until the research question, human role, minimum environment, and evaluation strategy are sufficiently defined.

---

# Phase 1 — Minimal combat environment  ✅ COMPLETE

## Goal

Create a tiny, fast, parameterized environment.

### Minimum candidate

- one player
- one enemy
- a few attack patterns
- movement
- dodge
- attack
- health
- stamina
- episode termination

### Requirements

- fast simulation
- reproducible seeds
- explicit difficulty parameters
- headless execution
- structured logging
- deterministic scenario generation where practical

### Gate

The environment must produce meaningful trajectories and expose measurable combat outcomes.

---

# Phase 2 — Non-EEG RL baseline  🟡 IN PROGRESS — Gate 1a passed; Gate 1 not attempted

## Goal

Prove that the environment can support learning before EEG is introduced.

### Tasks

- [x] Define observation space.
- [x] Define action space.
- [ ] Define reward.
- [x] Benchmark appropriate RL algorithms.
- [ ] Train baseline agents.
- [ ] Establish evaluation protocol.
- [ ] Measure learning curves.
- [ ] Measure combat metrics.
- [x] Establish random-seed handling.
- [x] Save checkpoints and experiment configurations.

### Gate

The agent must demonstrate learning on the minimal task.

If the baseline cannot learn, do not add EEG complexity yet.

---

# Phase 3 — EEG pipeline  🟡 STARTED — dataset verified (D032); pipeline not built

## Goal

Develop EEG processing independently from the RL environment.

### Tasks

- [x] Select development dataset(s).
- [x] Establish data provenance.
- [x] Load and inspect raw EEG.
- [ ] Establish signal-quality checks.
- [ ] Evaluate filtering.
- [ ] Evaluate artifact handling.
- [ ] Evaluate segmentation/windowing.
- [ ] Establish candidate features.
- [ ] Define normalization strategy.
- [ ] Create reproducible preprocessing pipeline.

### Artifact robustness

Explicitly test the impact of:

- ocular artifacts
- muscle activity
- movement
- other physiological noise

The system should not depend on continuous suppression of natural behavior.

### Gate

The preprocessing pipeline must be reproducible and its limitations documented.

---

# Phase 4 — Neural-state estimation  ⬜ NOT STARTED

## Goal

Determine whether a defensible EEG-derived state can be estimated.

### Tasks

- [ ] Select the state target.
- [ ] Define labels/ground truth.
- [ ] Establish train/validation/test separation.
- [ ] Build a simple baseline estimator.
- [ ] Evaluate more complex models only if justified.
- [ ] Perform artifact/confound checks.
- [ ] Evaluate generalization across sessions/subjects where data permits.
- [ ] Document uncertainty.

### Important

Do not assume that classifier accuracy automatically means the model learned the intended neural phenomenon.

### Gate

There must be a defensible argument for what the estimator measures and what it does not measure.

---

# Phase 5 — Curriculum interface  ⬜ NOT STARTED

## Goal

Create a clean interface between state estimation and environment configuration.

### Candidate structure

```text
state estimate
      ↓
curriculum policy
      ↓
environment parameters/scenario distribution
```

### Tasks

- [ ] Define curriculum API.
- [ ] Define parameter bounds.
- [ ] Define mapping function.
- [ ] Ensure mapping is reproducible.
- [ ] Implement static curriculum.
- [ ] Implement random curriculum.
- [ ] Implement neuroadaptive curriculum.
- [ ] Implement shuffled-EEG control if adopted.
- [ ] Implement agent-performance baseline if adopted.

### Gate

All curriculum conditions should use comparable environment interfaces.

---

# Phase 6 — Integrated neuroadaptive RL  ⬜ NOT STARTED

## Goal

Train agents under the competing curriculum conditions.

### Tasks

- [ ] Freeze the experimental configuration.
- [ ] Freeze evaluation scenarios.
- [ ] Generate/replay EEG source data.
- [ ] Train multiple seeds.
- [ ] Save complete experiment metadata.
- [ ] Monitor training failures.
- [ ] Preserve raw results.
- [ ] Evaluate held-out conditions.

### Gate

Experiments must be reproducible from recorded configuration and code version.

---

# Phase 7 — Statistical analysis  ⬜ NOT STARTED

## Goal

Determine whether observed differences support the research hypothesis.

### Tasks

- [ ] Aggregate per-seed results.
- [ ] Plot learning curves.
- [ ] Compare primary outcome.
- [ ] Compare secondary outcomes.
- [ ] Calculate uncertainty.
- [ ] Perform planned statistical tests.
- [ ] Analyze controls/ablations.
- [ ] Investigate alternative mechanisms.
- [ ] Report null results honestly.

### Gate

The conclusion must distinguish:

- evidence for improvement
- evidence against improvement
- inconclusive results
- evidence that an alternative mechanism explains the result

---

# Phase 8 — Robustness and extension  ⬜ STRETCH ONLY

Only after the core experiment works:

- [ ] increase combat complexity
- [ ] add more enemy patterns
- [ ] add more skills
- [ ] test more environment variation
- [ ] investigate live EEG
- [ ] investigate a presentation/demo mode
- [ ] consider more advanced RL algorithms
- [ ] consider cross-session/cross-subject experiments

These are optional extensions.

---

# Phase 9 — Capstone documentation and presentation  ⬜ NOT STARTED

### Deliverables

- [ ] technical architecture
- [ ] research methodology
- [ ] literature review
- [ ] experiment configuration
- [ ] results
- [ ] limitations
- [ ] reproducibility instructions
- [ ] demo
- [ ] final presentation
- [ ] final report

---

# Critical-path rule

If time becomes constrained, prioritize:

1. Research question
2. Minimal environment
3. RL baseline
4. EEG estimator validation
5. Curriculum interface
6. Controlled experiment
7. Analysis
8. Documentation
9. Visual polish

Do not sacrifice the experiment for game polish.

---

# Current next actions

Updated 2026-11-01. Ordered by how much each reduces the risk of the project
failing, not by phase number.

1. **Hardware validation gate.** Power on the Cyton + Ultracortex for the first
   time. Impedance per channel, eyes-closed alpha at Pz, blink and jaw-clench
   artifact check, 60 Hz notch, both ear clips. Untouched for three weeks and
   the only item with an irreducible lead time if something is faulty.
2. **Gate 1** — train the DQN on `combat:12`. The algorithm is cleared, so a
   failure now points at the environment, observation space or the provisional
   reward (OPEN-06) rather than at the learner.
3. **Agent per-configuration sweep** across all 25 configurations, to produce
   the agent difficulty ranking.
4. **Human play session, no EEG** (`scripts/play.py --session`), to produce the
   human difficulty ranking.
5. **Gate 2** — Spearman rank correlation between 3 and 4. Needs no hardware and
   is the experiment most able to falsify the project's central premise, so it
   should be run early rather than late.
6. Curriculum sampler (one implementation, ordering source injected).
7. EEG preprocessing pipeline, developed against EEGMAT.

Deferred until the above are done: Phase 8 entirely, and any further work on
combat feel, rendering or enemy variety.
