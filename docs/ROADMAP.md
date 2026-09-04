# Capstone Roadmap

**Status:** Living development plan  
**Starting state:** Phase 0 — no confirmed prototype, dataset, hardware, or final implementation stack

---

# Guiding principle

Build the **minimum viable scientific experiment** before building the impressive game.

The project succeeds if the research question can be tested rigorously. A large game is optional; a reproducible experiment is not.

---

# Phase 0 — Research and architecture

## Goal

Remove the decisions that would invalidate or force major rewrites later.

### Tasks

- [ ] Confirm the human EEG-recording architecture.
- [ ] Define exactly what the human does during recording.
- [ ] Define the candidate neural/physiological state.
- [ ] Determine how that state can obtain defensible labels/ground truth.
- [ ] Define what "improvement" means.
- [ ] Select primary and secondary dependent variables.
- [ ] Define the independent variable.
- [ ] Define candidate controls/baselines.
- [ ] Define the smallest viable combat task.
- [ ] Determine whether a custom simulator is appropriate.
- [ ] Determine institutional ethics/IRB requirements.
- [ ] Survey public EEG datasets.
- [ ] Create an initial preregistration/research plan.

### Gate

Do not move into large implementation until the research question, human role, minimum environment, and evaluation strategy are sufficiently defined.

---

# Phase 1 — Minimal combat environment

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

# Phase 2 — Non-EEG RL baseline

## Goal

Prove that the environment can support learning before EEG is introduced.

### Tasks

- [ ] Define observation space.
- [ ] Define action space.
- [ ] Define reward.
- [ ] Benchmark appropriate RL algorithms.
- [ ] Train baseline agents.
- [ ] Establish evaluation protocol.
- [ ] Measure learning curves.
- [ ] Measure combat metrics.
- [ ] Establish random-seed handling.
- [ ] Save checkpoints and experiment configurations.

### Gate

The agent must demonstrate learning on the minimal task.

If the baseline cannot learn, do not add EEG complexity yet.

---

# Phase 3 — EEG pipeline

## Goal

Develop EEG processing independently from the RL environment.

### Tasks

- [ ] Select development dataset(s).
- [ ] Establish data provenance.
- [ ] Load and inspect raw EEG.
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

# Phase 4 — Neural-state estimation

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

# Phase 5 — Curriculum interface

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

# Phase 6 — Integrated neuroadaptive RL

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

# Phase 7 — Statistical analysis

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

# Phase 8 — Robustness and extension

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

# Phase 9 — Capstone documentation and presentation

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

1. Resolve the human-recording architecture.
2. Define the minimum viable experiment.
3. Draft the preregistration.
4. Investigate ethics/IRB requirements.
5. Survey public EEG datasets.
6. Design the minimal parameterized combat environment.
7. Only then commit to the implementation stack.
