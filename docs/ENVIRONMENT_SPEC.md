# Environment Specification

Status: **implemented, Gate 1 not yet attempted.** Version 0.1, September 2026.

This document fixes the scope of the combat environment. Anything not listed
here is out of scope for the capstone regardless of how easy it would be to add.

## 1. Purpose

The environment is instrumentation, not a deliverable. It exists to provide a
small, enumerable set of difficulty configurations that a human can play and an
agent can train on **under identical dynamics**, so that per-configuration
difficulty rankings from the two are comparable.

## 2. Requirements it must satisfy

| # | Requirement | Why |
|---|---|---|
| R1 | Deterministic under a fixed seed | Every experiment depends on reproducibility |
| R2 | Headless, >100 episodes/sec | ~60 training runs must be affordable |
| R3 | Human-playable at real time, same dynamics | Gate 2 comparability |
| R4 | Discrete, enumerable difficulty grid | The EEG session must cover every cell once |
| R5 | No agent observation unavailable to a human | Architectural invariant |

Measured throughput: **~270 episodes/sec** single-threaded (R2 satisfied).

## 3. Entities and mechanics

- Two-dimensional circular arena, radius 6.0, no terrain or obstacles.
- One agent, one enemy. No allies, items, or progression.
- Fixed simulation tick at 30 Hz. Agent decides at 15 Hz (`action_repeat = 2`).

**Agent state:** position, facing, health (100), stamina (100), attack cooldown,
dodge cooldown, invulnerability window.

**Enemy state:** position, current attack, phase, phase timer. The enemy is
scripted and does not learn.

**Enemy phases:** `IDLE -> TELEGRAPH -> ACTIVE -> RECOVERY -> IDLE`.
Damage is applied only during `ACTIVE`, once per attack.

**Enemy attacks (3):** jab (fast, short, 10 dmg), heavy (slow, long, 22 dmg),
lunge (dashes forward during the active window, 16 dmg). Selection is uniform
random from the seeded generator.

**Agent actions (12, discrete):** idle, 8 movement directions, attack, dodge,
block. Dodge travels in the last movement direction and grants 0.20 s of
invulnerability. Block scales incoming damage to 0.35 at a stamina cost.

**Episode:** one encounter. Terminates on enemy death (`win`), agent death
(`loss`), or 40 s (`timeout`). Success is defined as `win`.

## 4. Observation space

18 dimensions, agent-egocentric: unit vector to enemy, normalized distance,
agent health and stamina, enemy health, attack and dodge cooldowns,
invulnerability flag, enemy phase one-hot (4), attack type one-hot (3), phase
progress, distance from arena centre.

Enemy phase and progress are included because a human reads the same information
from the wind-up animation. The human-like limitation is imposed by a **6-tick
(200 ms) observation delay**, not by hiding state.

## 5. Difficulty configuration space

Two axes, five levels each, giving **25 configurations** indexed `config_id`
0-24 (`config_id = telegraph_level * 5 + gap_level`).

| Axis | Levels | Mechanism |
|---|---|---|
| Telegraph duration | 0.70, 0.55, 0.42, 0.32, 0.24 s | Compresses the window to recognise and respond to an attack |
| Attack gap | 1.60, 1.25, 0.95, 0.72, 0.52 s | Removes recovery windows, forces stamina management |

All other parameters are constant across configurations.

Five configurations are reserved as the **held-out set** for generalization
testing and are never sampled during training in any condition. The split is
fixed before any experimental runs.

## 6. Known issue: axis imbalance

A scripted reactive policy over 40 episodes per cell produces a clear overall
gradient (damage taken 23.1 at the easiest cell, 90.2 at the hardest; win rate
1.00 to 0.65). The axes are **not balanced**, however:

- Shortening the gap raises damage taken by ~43.5 on average.
- Shortening the telegraph raises it by ~17.3, and has **no effect at all** at
  the sparsest gap level (32.8 across four of five telegraph levels).

The likely cause is that the 200 ms observation delay swamps the difference
between a 0.70 s and a 0.24 s telegraph for a purely reactive policy.

This is unresolved (see `OPEN-02`). Candidate responses, to be decided before
the human session is designed:

1. Widen the telegraph range, e.g. 0.90 s down to 0.15 s.
2. Reduce the observation delay, accepting a less human-like agent.
3. Shorten dodge invulnerability so timing precision matters more.
4. Accept the imbalance, on the grounds that a *trained* agent may exploit
   telegraph length differently than this scripted policy does.

Option 4 must not be assumed. The scripted policy is a smoke test, not a
substitute for Gate 1.

## 7. Reward function — PROVISIONAL

`OPEN-06` is not closed. Current defaults exist so the environment runs:

| Term | Weight |
|---|---|
| Damage dealt | +1.0 per point |
| Damage taken | -1.2 per point |
| Win | +60 |
| Loss | -30 |
| Time | -0.05 per decision step |
| Successful dodge | +2.0 |

The failure mode to watch for is a policy that survives by never attacking; the
time penalty and win bonus exist to discourage it. Any change to these weights
invalidates cross-experiment comparison and must be recorded before the
pre-registration is locked.

## 8. Out of scope

Multiple enemies, ranged attacks, terrain, items, character progression,
3D, networked play, sound, animation beyond the phase colour cue.
