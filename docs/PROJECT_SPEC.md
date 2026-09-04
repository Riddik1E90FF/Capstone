# Project Specification
## EEG → AI → ARPG Neuroadaptive Reinforcement Learning

**Status:** Phase 0 — architecture/research definition  
**Specification type:** Living technical specification

---

# 1. Purpose

Build an experimental system combining:

- EEG/neural signals
- neuroscience/neurotechnology processing
- machine learning
- reinforcement learning
- an interactive ARPG-style combat environment
- adaptive curriculum generation

The purpose is to investigate whether information derived from a human's EEG can provide a useful signal for shaping the training experiences of an RL agent.

---

# 2. Central research concept

The core architecture is:

```text
Human
  ↓
EEG
  ↓
Preprocessing
  ↓
Artifact handling
  ↓
Feature extraction
  ↓
Neural-state estimation
  ↓
Curriculum / environment modulation
  ↓
ARPG environment
  ↓
RL agent
  ↓
Actions / rewards
  ↓
Learning
```

The defining principle is:

> **EEG-derived information changes the environment/curriculum rather than becoming privileged information available to the agent.**

The agent should learn from the resulting consequences.

---

# 3. Research question

Current broad formulation:

> Can a reinforcement-learning agent improve learning, robustness, or generalization when its training environment/curriculum is dynamically shaped using human EEG-derived state information?

A more specific candidate formulation proposed during later review is:

> Does an EEG-derived estimate of human task difficulty provide a useful prior for curriculum generation in reinforcement learning, compared with static, random, shuffled-EEG, and agent-performance-adaptive curricula?

The second formulation is a **candidate refinement**, not a final locked wording.

---

# 4. Project evolution

The project originated from an interest in AI learning ARPG combat, including a Skyrim/Helgen-style game-AI concept.

Desired combat abilities included:

- enemy moveset recognition
- attack prediction
- dodge timing
- positioning
- stamina/resource management
- attack timing
- adaptation
- risk/reward decisions

The neuroscience question emerged from asking how EEG could contribute without turning the project into a simple BCI controller.

That led to the current environment/curriculum-shaping concept.

---

# 5. What the project is not

The project is not intended to become:

- a medical device
- a diagnostic system
- a claim of thought reading
- a generic BCI controller
- a pure neuroscience study
- a generic game AI project without the EEG research component

A neural network alone does not make a project neuroscientific.

---

# 6. Human role

## Important unresolved decision

The original project documents did not fully specify what the human does during EEG recording.

Three architectures were identified:

### A — Human plays the environment

The human plays the same combat environment while EEG is recorded.

The EEG-derived state is associated with known environment configurations and task difficulty.

The recording can then be replayed during RL training.

### B — Human watches the agent

The human observes agent behavior while EEG is recorded.

This creates a human-coach interpretation but introduces major real-time and synchronization constraints.

### C — Live coupled loop

The human and agent interact with a live training loop.

This is visually compelling but substantially increases engineering complexity and reduces experimental convenience.

### Current recommendation

Architecture A is strongly favored because it:

- separates human recording from RL training
- permits EEG replay
- improves reproducibility
- enables multiple RL seeds from one recording
- simplifies shuffled-EEG controls
- reduces real-time synchronization risk
- creates an intermediate EEG validation task

**Status: Recommended, not yet confirmed.**

---

# 7. EEG role

EEG is intended to provide measurable neural/physiological information that can contribute to an estimated state.

Candidate targets include:

- cognitive effort
- task difficulty
- engagement
- another experimentally validated neural/physiological correlate

The final target remains open.

The project should avoid attempting instantaneous thought decoding.

---

# 8. EEG processing

Candidate processing stages:

```text
Raw EEG
  ↓
Signal quality checks
  ↓
Filtering
  ↓
Artifact handling
  ↓
Segmentation / epoching
  ↓
Feature extraction
  ↓
State estimation
```

Potential techniques discussed include:

- band-pass filtering
- notch filtering
- artifact rejection
- artifact-heavy window detection
- ICA
- normalization
- spectral analysis
- bandpower
- time/frequency features

None of these should be treated as mandatory until validated.

---

# 9. EEG frequency bands

Previously discussed approximate bands:

| Band | Approximate range |
|---|---:|
| Delta | ~0.5–4 Hz |
| Theta | ~4–8 Hz |
| Alpha | ~8–13 Hz |
| Beta | ~13–30 Hz |
| Gamma | 30+ Hz |

These are candidate analytical features, not a finalized feature set.

---

# 10. EEG hardware/data source

No hardware is selected.

Previously discussed candidates:

- OpenBCI
- Emotiv
- public EEG datasets

Public EEG data is an important early-development option because it permits progress before hardware procurement.

Potential development strategy:

1. Develop EEG processing with public data.
2. Validate the state-estimation pipeline offline.
3. Build the game/RL environment independently.
4. Simulate streaming/replay.
5. Integrate live EEG only if justified and feasible.

---

# 11. Artifact and movement robustness

EEG is susceptible to:

- ocular artifacts
- muscle/EMG artifacts
- movement artifacts
- electrode/contact changes
- other physiological contamination

The system should be designed around:

> **Natural behavior → artifact detection/handling → robust processing**

rather than requiring the participant to continuously suppress natural movement.

Participant-specific motor/ocular behavior should not automatically be interpreted as cognitive load.

If a participant-specific artifact is investigated scientifically, the interpretation must be independently validated.

---

# 12. RL system

Reinforcement learning is the primary AI paradigm.

Candidate algorithms discussed:

- DQN
- PPO
- actor-critic methods

No algorithm is selected.

Selection should follow environment/action-space design and empirical suitability.

---

# 13. RL formulation

## Observation

Candidate information:

- player position
- enemy position
- enemy attack state
- health
- stamina
- cooldowns
- incoming attack
- relevant timing information

Final observation space is open.

## Actions

Candidate actions:

- movement
- attack
- dodge
- block
- wait
- ability/use action

Final action space is open.

## Reward

Candidate reward components:

- survival
- damage avoidance
- effective attacks
- combat success
- stamina efficiency
- task completion

The final reward must avoid degenerate behavior such as surviving by refusing to engage.

## Episode

Likely a combat encounter/training scenario.

Exact episode boundaries are open.

---

# 14. Environment

The environment should be ARPG-style and intentionally small at first.

A candidate minimal environment:

- one player
- one enemy
- a small number of enemy attack patterns
- movement
- dodge
- attack
- health
- stamina
- clear episode termination

The environment must expose explicit, parameterized difficulty controls so the curriculum system has something meaningful to modulate.

---

# 15. Environment representation

The final choice among:

- custom Python simulator
- 2D game
- 3D engine
- Skyrim integration
- another existing game

is not finalized.

A custom fast simulator is currently a strong recommendation because it can provide:

- high simulation throughput
- deterministic seeds
- headless training
- precise difficulty controls
- low integration overhead

This is a recommendation, not a historical decision.

---

# 16. Curriculum modulation

Potential environment parameters include:

- enemy aggression
- attack frequency
- attack timing
- timing variability
- encounter density
- repetition frequency
- scenario selection
- overall difficulty

The mapping from estimated human state to environment parameters is currently unresolved.

The eventual mapping should be:

- simple
- interpretable
- monotonic where appropriate
- documented
- reproducible
- specified before final evaluation

---

# 17. EEG replay

The architecture should support an EEG source abstraction with at least conceptual support for:

```text
Recorded EEG
Synthetic EEG
Live EEG
```

Recorded replay is especially important because it allows one human recording to support multiple RL runs and seeds.

---

# 18. Experimental conditions

Core proposed conditions:

### A — Static

Fixed curriculum/environment schedule.

### B — Random

Randomly varying curriculum/environment.

### C — Neuroadaptive

Curriculum driven by EEG-derived state.

Potential additions:

### D — Shuffled EEG

Temporally shuffled EEG-derived state or equivalent randomized replay.

### E — Agent-performance adaptive

Curriculum driven by the agent's own performance/learning signal.

D and E are proposed controls/baselines and require formal design before implementation.

---

# 19. Candidate mechanisms

The project should distinguish at least three possible explanations:

### Mechanism 1 — Human difficulty transfer

Human task difficulty contains useful information about which training conditions are valuable to an artificial learner.

### Mechanism 2 — Structured non-stationarity

The temporal structure of the EEG-derived signal improves training even if its human-specific interpretation is not the causal mechanism.

### Mechanism 3 — No useful advantage

The EEG-derived curriculum does not provide meaningful benefit beyond conventional curriculum strategies.

---

# 20. Evaluation

Potential RL metrics:

- episode reward
- survival time
- task completion
- damage taken
- damage avoided
- combat success
- dodge success
- stamina efficiency

Potential learning metrics:

- reward versus training steps
- time/steps to performance threshold
- sample efficiency
- learning stability

Potential generalization metrics:

- unseen attack patterns
- altered timing
- unseen enemy configurations
- held-out environment parameters
- robustness to environmental variation

Potential EEG metrics:

- signal quality
- artifact percentage
- state-estimation performance
- latency
- robustness across sessions

The primary dependent variable is not finalized.

---

# 21. Statistical design

The project must account for RL variance across random seeds.

Before final experiments, specify:

- number of seeds
- training budget
- evaluation budget
- statistical test
- effect-size target where appropriate
- confidence intervals
- multiple-comparison handling where relevant
- primary versus secondary outcomes

Do not hard-code a seed count before the experimental budget and expected variance are understood.

---

# 22. Data synchronization

Where EEG and gameplay are linked, the system should be capable of associating:

- EEG timestamps
- environment events
- environment state changes
- actions
- rewards
- scenario/configuration identifiers

A replay architecture should minimize unnecessary real-time synchronization complexity.

---

# 23. Reproducibility requirements

Each experiment should record:

- Git commit
- random seed
- environment configuration
- curriculum configuration
- EEG source
- preprocessing configuration
- model configuration
- hyperparameters
- training budget
- evaluation configuration
- metrics
- checkpoint
- timestamps

---

# 24. Privacy and ethics

Raw EEG is sensitive personal data.

The project should consider:

- local/private storage
- anonymization
- access control
- secure backups
- licensing/data-use restrictions
- consent/ethics requirements

If additional human participants are introduced, institutional human-subject/IRB requirements must be investigated before data collection.

---

# 25. Current confirmed vs open status

## Confirmed

- EEG/neural signals are part of the project.
- RL is the AI paradigm.
- ARPG-style combat is the environment concept.
- EEG should influence environment/curriculum rather than privileged agent information.
- Artifact handling is important.
- Public EEG datasets are acceptable for development.
- Scientific claims should be conservative.
- The smallest viable experiment is preferred.

## Tentative

- Human plays the same environment during EEG recording.
- EEG estimates human task difficulty/cognitive effort or a related validated state.
- Custom simulator.
- Adaptive difficulty.
- Dodge/stamina as initial skills.
- Static/random/neuroadaptive comparisons.

## Not selected

- EEG hardware
- game engine
- RL algorithm
- final neural-state target
- final feature set
- final preprocessing pipeline
- final curriculum mapping
- final primary dependent variable
- final seed count
