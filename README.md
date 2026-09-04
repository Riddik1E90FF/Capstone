# EEG → AI → ARPG Neuroadaptive RL Capstone

## Overview

This repository contains the implementation, experiments, research documentation, and reproducibility infrastructure for a university AI Engineering capstone.

The project investigates whether **human EEG-derived information can be used as an adaptive curriculum signal for a reinforcement-learning agent learning ARPG-style combat**.

### Core idea

```text
Human
  ↓
EEG
  ↓
Preprocessing + artifact handling
  ↓
Neural-state estimation
  ↓
Environment / curriculum modulation
  ↓
ARPG combat environment
  ↓
RL agent
  ↓
Observations → Actions → Rewards
  ↓
Learning
```

The defining constraint is:

> **EEG shapes the environment the AI learns in; EEG does not directly control the AI agent.**

The agent should not receive privileged access to the human's EEG or inferred state.

---

## Current status

**Phase 0 — Research and architecture definition**

At repository initialization:

- No final EEG hardware has been selected.
- No final game engine has been selected.
- No final RL algorithm has been selected.
- No final neural-state target has been selected.
- No complete prototype is assumed to exist.
- The exact human EEG-recording protocol remains an important design decision.

A favored architecture is for the human to play the same minimal combat environment offline while EEG is recorded, followed by replay of the recorded EEG/state during RL training. This is a recommendation, not yet a locked decision.

---

## Research direction

The project evolved from an earlier interest in game-playing AI and ARPG/Skyrim-style combat into a neuroadaptive reinforcement-learning study.

The important research distinction is between:

**Ordinary game AI**

> Build an AI that learns to play an ARPG.

and:

**Current capstone**

> Build an RL agent that learns ARPG combat inside a training environment whose curriculum is adaptively shaped by human EEG-derived state.

The second formulation creates the neuroscience/neurotechnology research component.

---

## Documentation

| File | Purpose |
|---|---|
| `CLAUDE.md` | Instructions for Claude Code and project development |
| `docs/PROJECT_SPEC.md` | Current project specification and architecture |
| `docs/ROADMAP.md` | Development phases, gates, and priorities |
| `docs/DECISIONS.md` | Decision log with confirmed/tentative/proposed choices |
| `docs/RESEARCH.md` | Research question, hypotheses, methodology, evidence, and open questions |
| `docs/ORIGINAL_PROJECT_EXPORT.md` | Historical project conversation/export, when included |

---

## Planned high-level components

```text
src/
├── eeg/
├── state_estimation/
├── environment/
├── curriculum/
├── agents/
├── training/
├── evaluation/
└── common/
```

This structure is a starting convention, not an assumption about existing code.

---

## Experimental backbone

A strong candidate comparison is:

1. Static curriculum
2. Random curriculum
3. EEG/neuroadaptive curriculum

Potential additional controls/baselines include:

4. Shuffled-EEG curriculum
5. Agent-performance-adaptive curriculum

These additions are research proposals until formally adopted.

---

## Reproducibility

Experiments should record:

- code version / Git commit
- random seed
- environment configuration
- curriculum configuration
- EEG source and preprocessing configuration
- model configuration
- training budget
- evaluation metrics
- checkpoint path
- experiment timestamp

---

## Data

Do not commit raw EEG or other sensitive/large datasets to this public repository unless their licensing, consent, and privacy requirements explicitly permit it.

See `data/README.md`.

---

## Development philosophy

The project prioritizes:

- scientific falsifiability
- reproducibility
- modularity
- measurable outcomes
- controlled comparisons
- explicit uncertainty
- small testable increments

The goal is not to build the largest game possible.

> **The smallest viable experiment is more valuable than a giant unfinished game.**
