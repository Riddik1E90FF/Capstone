# Research Plan and Scientific Framing

**Status:** Living research document  
**Current phase:** Phase 0

---

# 1. Research motivation

The project began as an AI/game-playing problem:

> Can an AI learn useful ARPG combat behavior through reinforcement learning?

The project became a neurotechnology research problem by asking:

> Can human neural information influence the experiences from which an artificial learner learns, without directly controlling or informing the learner?

This distinction is central to the capstone.

---

# 2. Candidate research question

Broad version:

> Can a reinforcement-learning agent improve learning, robustness, or generalization when its training environment/curriculum is dynamically shaped using human EEG-derived state information?

More focused candidate:

> Does an EEG-derived estimate of human task difficulty provide a useful prior for curriculum generation in reinforcement learning, compared with static, random, shuffled-EEG, and agent-performance-adaptive curricula?

The final wording remains open.

---

# 3. Why this is scientifically interesting

A human's task difficulty and an artificial agent's learning difficulty are not assumed to be identical.

The experiment asks whether information about human interaction with a task contains useful information for designing the distribution of experiences used to train an RL agent.

Possible outcomes include:

- EEG-derived human difficulty improves agent generalization.
- EEG-derived human difficulty improves learning efficiency.
- EEG-derived information changes learning dynamics but does not improve final performance.
- EEG-derived curricula perform no better than conventional curricula.
- The apparent benefit comes from structured temporal variation rather than human-specific information.
- The EEG estimator does not provide a sufficiently valid signal.

All of these can be informative if the experiment is controlled.

---

# 4. Competing mechanisms

## H1 — Human difficulty transfer

If a human finds a combat configuration difficult, that configuration may contain information about a useful learning frontier for an artificial learner.

This is a hypothesis, not an established fact.

## H2 — Structured non-stationarity

An EEG-derived curriculum may introduce structured temporal variation.

The RL benefit, if any, could arise from the structure of the curriculum rather than the semantic meaning of the human neural state.

## H3 — No useful advantage

The EEG-derived signal may add complexity without improving the training process.

---

# 5. Candidate hypotheses

These should eventually be rewritten into preregistered statistical hypotheses.

### Primary candidate

An RL agent trained under a validated neuroadaptive curriculum will demonstrate better performance on held-out combat configurations than an agent trained under a conventional baseline curriculum.

### Secondary candidates

The neuroadaptive curriculum may:

- improve sample efficiency
- improve learning stability
- improve robustness to environmental variation
- alter the learning curve
- improve specific combat skills

The primary outcome is not yet selected.

---

# 6. Experimental conditions

## Condition A — Static

The environment follows a fixed training schedule.

## Condition B — Random

The environment varies according to a predefined randomized curriculum.

## Condition C — Neuroadaptive

The environment varies according to EEG-derived state.

## Candidate Condition D — Shuffled EEG

EEG-derived state is temporally shuffled or otherwise randomized while preserving appropriate distributional properties.

Purpose: test whether temporal structure and/or human-specific information matters.

## Candidate Condition E — Agent-performance adaptive

The environment adapts based on the agent's own performance.

Purpose: test whether EEG provides value beyond a curriculum signal that can be obtained directly from the learner.

---

# 7. Human recording architecture

The strongest current recommendation is:

1. Human plays the minimal combat environment.
2. EEG is recorded.
3. Environment configuration and event timestamps are recorded alongside EEG.
4. A neural-state estimator is trained/validated.
5. The resulting state estimate is replayable.
6. RL agents train offline using the same recorded state source.
7. Multiple random seeds can use the same human recording.

This avoids requiring the human to be present for every RL run.

It also makes controlled ablations substantially easier.

---

# 8. Neural-state target

The final target must be operationally defined.

Possible targets:

- task difficulty
- cognitive effort
- engagement
- another measurable neural/physiological correlate

The project should not assume that "cognitive load" is directly observable from EEG without a validation strategy.

---

# 9. Ground truth / labels

A critical research problem is obtaining a defensible target for state estimation.

Potential approaches include:

### Experimental difficulty manipulation

The human performs tasks at known difficulty levels.

Advantages:

- reproducible
- directly tied to task parameters

Limitation:

- the model may learn task difficulty rather than a more general cognitive state

### Subjective workload measures

For example, a validated workload questionnaire collected after blocks.

Advantages:

- directly captures subjective experience

Limitations:

- coarse
- low-frequency
- self-report dependent

### Behavioral measures

Potential signals include:

- reaction time
- error rate
- damage taken
- task completion
- successful dodges

These may provide objective task-performance proxies.

### Artifact controls

Deliberate artifact-control conditions can test whether a model is learning muscle/ocular/movement activity instead of the intended neural correlate.

This is especially important when EEG is collected during active gameplay.

---

# 10. EEG preprocessing research

Potential pipeline:

```text
Raw EEG
 ↓
Quality control
 ↓
Filtering
 ↓
Artifact identification
 ↓
Artifact handling
 ↓
Epoching
 ↓
Normalization
 ↓
Feature extraction
 ↓
State estimation
```

Candidate features:

- bandpower
- spectral power
- time-frequency features
- temporal changes
- other features supported by the literature

Candidate frequency bands:

- delta
- theta
- alpha
- beta
- gamma

No final feature set is selected.

---

# 11. Artifact research

Major artifact classes:

- eye blinks and eye movement
- facial muscle activity
- other EMG
- head movement
- electrode movement/contact changes
- other physiological activity

Candidate handling methods:

- filtering
- ICA
- artifact rejection
- artifact-heavy window detection
- regression-based correction
- robust feature/model design

The final method should be chosen empirically.

---

# 12. RL research

The RL agent should learn combat through the environment.

Potential skills:

- enemy attack recognition
- attack prediction
- dodge timing
- positioning
- stamina management
- attack timing
- risk/reward decisions

The first environment should focus on only a small subset.

The final algorithm should be selected after defining the action space and environment dynamics.

---

# 13. Environment research

The environment must expose explicit parameters that the curriculum can modify.

Candidate parameters:

- enemy aggression
- attack frequency
- attack timing
- timing variability
- encounter density
- scenario selection
- repetition
- overall difficulty

The mapping from state estimate to parameters is a central experimental design decision.

A hand-tuned mapping should be simple, interpretable, and documented.

---

# 14. Primary evaluation possibilities

### Generalization

Test on held-out:

- attack patterns
- timing combinations
- enemy configurations
- environment parameters

### Learning efficiency

Measure:

- reward versus environment steps
- time to performance threshold
- samples needed to achieve a target level

### Final performance

Measure:

- combat success
- survival
- damage
- dodge success
- stamina efficiency

The final primary dependent variable is open.

---

# 15. Statistical considerations

RL can exhibit large variance across random seeds.

Therefore, final experiments should:

- use multiple seeds
- preserve per-seed results
- report uncertainty
- avoid selecting only successful runs
- separate training and evaluation environments
- avoid tuning on the final test set

The exact number of seeds is not yet fixed.

---

# 16. Threats to validity

## EEG validity

A classifier may learn:

- muscle activity
- eye movement
- movement
- session-specific artifacts
- participant identity

instead of the intended neural phenomenon.

## Label validity

A task difficulty label does not necessarily equal subjective cognitive load.

## Curriculum validity

A null result may mean the chosen mapping was ineffective rather than that EEG-derived information is fundamentally useless.

## RL validity

A single seed or small number of runs can produce misleading comparisons.

## Generalization validity

The test distribution must actually represent conditions not used to tune the agent.

## External validity

A single participant and a small simulator cannot establish broad claims about all humans or all RL environments.

---

# 17. Ethical and privacy considerations

EEG is sensitive neural/physiological data.

The project should document:

- where raw data is stored
- who can access it
- whether it can be shared
- licensing of public datasets
- consent requirements
- institutional ethics/IRB requirements if applicable

The project is not a medical diagnostic system.

---

# 18. Scientific language

Avoid:

> The AI reads the user's thoughts.

Prefer:

> The system estimates a selected neural/physiological feature from EEG and uses that estimate as an input to curriculum generation.

Avoid:

> EEG detects stress with certainty.

Prefer:

> The model estimates a candidate state that may correlate with task difficulty or cognitive effort under the defined experimental conditions.

---

# 19. Research deliverables

Eventually produce:

- literature review
- formal research question
- preregistration/research plan
- EEG processing methodology
- state-estimator validation
- environment specification
- RL methodology
- experimental conditions
- statistical analysis
- results
- limitations
- reproducibility documentation

---

# 20. Open research questions

1. What exactly does the human do during EEG recording?
2. What state should EEG estimate?
3. What labels/ground truth are defensible?
4. Which public EEG dataset best supports early development?
5. Which preprocessing method is appropriate?
6. Which artifacts can be detected reliably?
7. What environment parameters should be modulated?
8. What mapping should connect state to parameters?
9. What is the primary dependent variable?
10. How many RL seeds are required?
11. What statistical test is appropriate?
12. Does a custom simulator satisfy the capstone's expected demonstration requirements?
13. When, if ever, should live EEG hardware be introduced?
