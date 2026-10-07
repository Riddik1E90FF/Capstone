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
| Telegraph duration | 1.00, 0.66, 0.42, 0.24, 0.12 s | Compresses the window to recognise and respond to an attack |
| Attack gap | 1.60, 1.25, 0.95, 0.72, 0.52 s | Removes recovery windows, forces stamina management |

All other parameters are constant across configurations.

Five configurations are reserved as the **held-out set** for generalization
testing and are never sampled during training in any condition. The split is
fixed before any experimental runs.

## 6. Difficulty axis calibration

**Status: calibrated 2026-10-05.** The telegraph range was widened after the
original setting proved nearly inert for the agent. See `OPEN-02` (now closed)
and D026.

### The problem with the original setting

Under the original telegraph levels (0.70 - 0.24 s), a scripted reactive policy
over 40 episodes per cell produced a clear overall gradient but badly unbalanced
axes:

- Shortening the attack gap raised damage taken by ~43.5 on average.
- Shortening the telegraph raised it by ~17.3, with **no effect at all** at the
  sparsest gap level.

### Why damage taken is a misleading metric here

Damage taken conflates two quantities:

```
damage ~= (attacks faced) x (fraction that connect)
```

The attack gap axis controls the first term **mechanically** - halving the gap
roughly doubles the attacks faced regardless of skill. Telegraph duration only
affects the second term. Damage taken therefore structurally favours the gap
axis, and **hit rate** (`times_hit / attacks_faced`) is the cleaner read on
whether telegraph is doing anything.

### Sweep results

Measured by `scripts/axis_sweep.py`, 30 episodes per cell:

| Setting | Telegraph -> damage | Gap -> damage | Telegraph -> hit rate |
|---|---|---|---|
| Original (0.70-0.24 s) | 16.6 | 43.5 | 0.016 |
| **Selected (1.00-0.12 s)** | **30.6** | **45.0** | **0.205** |

The hit-rate column is the decisive one. Under the original setting, going from
the longest telegraph to the shortest changed hit rate by 0.016 - indistinguishable
from nothing. Under the selected setting it changes by 0.205.

Dodge invulnerability was also swept (0.20 / 0.12 / 0.10 s) and showed **no
measurable effect** under the scripted policy. It is therefore left unchanged at
0.20 s rather than altered without evidence.

### Residual limitation

At the sparsest attack gap, damage taken still saturates across the three
shortest telegraph levels, because encounters end before telegraph length can
express itself. The axes remain imperfectly balanced (telegraph effect is about
0.68 of the gap effect, improved from 0.38).

A scripted policy is a smoke test, not a trained agent. **Gate 2 is the real
test** of whether these axes carry difficulty for both learners.

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
