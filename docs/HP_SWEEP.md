# Hyperparameter sensitivity sweep — baseline agent on `combat:mix`

**Status:** COMPLETE (2026-11-02). **Result: nothing cleared D034's bar.
D034 rule 5 applies.** The CartPole `DQNConfig` defaults stand, and the
reportable finding is that the baseline agent is **not hyperparameter-limited
on this environment** over the grid below.

This document is a **proposal and a record**, not a decision. D034 and
`docs/PREREGISTRATION.md` are Ryan's to amend; nothing here edits either.

Driver: `scripts/hp_sweep.py` (`run` to measure, `report` to aggregate and
apply the rule). Raw per-run output is under `runs/hpsweep/`, which is
git-ignored, so every number worth keeping is reproduced here.

---

## 1. The question, and what this sweep is not

The Gate 1 agents ran on `DQNConfig` defaults tuned for CleanRL's CartPole.
`buffer_size = 10000` at roughly 142 steps per combat episode holds about 70
episodes, which is tight for a non-stationary uniform mixture over 25
configurations. G1-a and G1-b passed on all ten seeds, so this is not a Gate 1
problem. It is an **experiment** problem: an agent near the floor of its
capacity could mask real differences between the seven curriculum conditions
and cost power on H1 directly.

This sweep is baseline data, not experimental-condition data. It introduces no
EEG, changes no condition, and touches neither the environment, the reward, nor
the evaluation protocol.

**It is explicitly not a search for a better-looking Gate 1 result.** D034's
section "Hyperparameter sensitivity: decision rule, fixed 2026-11-01 before the
sweep" was written before any of these runs existed precisely to prevent that,
and §6 below applies it mechanically.

---

## 2. The decision rule, as written in D034

Reproduced so this document can be read on its own. D034 is authoritative.

1. The outcome of interest is **mean damage taken**, the same quantity Gate 1
   uses, averaged over the 3 seeds for each setting.
2. A setting replaces the current defaults only if it improves the mean by
   **more than 8.344** — one between-seed standard deviation from the Gate 1
   attempt. Smaller differences are inside seed noise and are not acted on.
3. If more than one setting clears that bar, the one with the **smallest**
   change from the current defaults is adopted, not the best-scoring one.
4. Whatever is adopted, **Gate 1 is re-run in full as a new attempt** under the
   new settings and must pass again on its own. The existing attempt is not
   retroactively reinterpreted.
5. If nothing clears the bar, the CartPole defaults stand and that is recorded
   as a result: the agent is not hyperparameter-limited on this environment.

The bar, 8.344, is the between-seed sample SD of mean damage taken over the ten
Gate 1 seeds (`docs/gate1_result.json`; recomputed here as 8.343869, quoted at
D034's precision). It is taken from the decision record, not recomputed to suit
this sweep.

### Two readings fixed before the results were looked at

Rule 2 says "improves the mean", where "the mean" is the 3-seed mean of a
setting. The comparison is therefore **between 3-seed means**, so the reference
is the **default setting's own 3-seed mean measured inside this sweep**, not
Gate 1's 10-seed mean. Same seeds, same code, same commit, same machine, same
protocol — the only thing that differs between a cell and the reference is the
hyperparameter. Gate 1's 10-seed mean (36.250) and its seeds-1–3 mean (30.587)
are reported alongside as context only. (§8 gives a concrete reason this
distinction matters.)

Rule 3 says "smallest change", which needs an ordering declared in advance or
it becomes a free choice after the fact. Fixed in `scripts/hp_sweep.py` as
`change_rank` before any cell finished:

1. fewer fields changed beats more fields changed;
2. then smaller multiplicative distance from the default,
   `max(v / default, default / v)` — all three fields are scale parameters, so
   ratio rather than difference is the natural metric;
3. then, on a ratio tie, the smaller absolute change in the field's value.

That gives: `lr5e-4` (2.0×) < `tf5` (2.0×, |Δ| 5) < `tf20` (2.0×, |Δ| 10) <
`lr1e-4` (2.5×) < `buf30k` (3.0×) < `buf3k` (3.33×) < `buf100k` (10×) <
`buf300k` (30×). The ordering was never exercised, because nothing qualified.

---

## 3. The grid, and why these ranges

Nine settings, 3 seeds each, 27 runs. A **coordinate (one-factor-at-a-time)**
design rather than a factorial one, for two reasons: rule 3 adopts the smallest
change from the defaults, which is only well defined if candidates are ordered
by distance from the defaults; and the compute budget (§4) did not stretch to a
factorial grid at the required 500k steps × 3 seeds.

| setting | `buffer_size` | `learning_rate` | `train_frequency` | distance from default |
|---|---|---|---|---|
| `default` | 10000 | 2.5e-4 | 10 | — |
| `buf3k` | **3000** | 2.5e-4 | 10 | 0.3× |
| `buf30k` | **30000** | 2.5e-4 | 10 | 3× |
| `buf100k` | **100000** | 2.5e-4 | 10 | 10× |
| `buf300k` | **300000** | 2.5e-4 | 10 | 30× |
| `lr1e-4` | 10000 | **1e-4** | 10 | 0.4× |
| `lr5e-4` | 10000 | **5e-4** | 10 | 2× |
| `tf5` | 10000 | 2.5e-4 | **5** | 0.5× (2× more updates) |
| `tf20` | 10000 | 2.5e-4 | **20** | 2× (2× fewer updates) |

**`buffer_size` gets the widest bracket — 0.3× to 30×, a 100-fold span — because
it is the field D034 names as the suspect.** The bracket deliberately extends
*below* the default as well as above: "does this metric move at all when replay
capacity changes by two orders of magnitude" is as informative as "does more
help", and a flat response across 3k–300k is much stronger evidence than a flat
response across 10k–100k would be.

**`learning_rate`: a 2–2.5× bracket either side.** 2.5e-4 is CleanRL's
CartPole value. The bracket stops short of 1e-3 on purpose: with a 500-step
hard target sync and no gradient clipping, vanilla DQN above roughly 1e-3 tends
to *diverge* rather than merely do worse, and a diverged cell would say nothing
about whether the agent is learning-rate-limited.

**`train_frequency`: a 2× bracket either side.** This field sets the replay
ratio (default 10 = 0.1 gradient updates per environment step, ~49k updates per
run). Replaying each stored transition more often is the other way to spend a
small buffer harder, so it is the natural companion to `buffer_size`: a bigger
buffer only matters if its contents are actually sampled.

### What was left out, and the consequence

A larger grid was planned and cut **before any cell completed**, when measured
throughput showed it would not finish (§4). Omitted: `train_frequency` 2 and 1
(≈3.5× and ≈6.5× the compute of a default run), `learning_rate` 1e-3, and a
two-factor `buffer_size = 100000, train_frequency = 2` interaction cell.

The honest consequences:

- Omitting the **more extreme single-factor** settings cannot change a
  recommendation, because rule 3 adopts the smallest qualifying change: if
  `tf1` had qualified, `tf5` or `tf20` would have had to qualify first to
  matter, and neither did. The observed direction is in any case *against*
  these settings (§5).
- Omitting the **interaction cell** is a genuine residual gap in §6's rule-5
  conclusion. This sweep rules out single-factor hyperparameter limitation over
  these ranges. It does not rule out a joint setting — most plausibly a large
  buffer combined with a higher replay ratio — that no coordinate move reaches.
  That gap is stated rather than papered over.

---

## 4. Protocol

Unchanged from Gate 1 wherever Gate 1 fixes something.

- **Environment:** `combat:mix` — `_MixtureEnv` in `scripts/train_dqn.py`,
  a uniform draw over all 25 configurations at each reset.
- **Training:** `scripts/train_dqn.py`, the ordinary CLI, one subprocess per
  run. 500k environment steps, seeds 1, 2, 3 (the first three Gate 1 training
  seeds, chosen so the default cell doubles as a reproducibility check on the
  Gate 1 attempt).
- **Evaluation:** `evaluation/gate1.evaluate_policy`, imported, not
  reimplemented. `N_EVAL = 200` episodes per configuration, all 25
  configurations, greedy (ε = 0), evaluation seeds 1000000–1000199 — disjoint
  from the training seeds. The greedy wrapper is the same construction
  `scripts/gate1_check.py::greedy_policy` uses, so the harness cannot tell a
  swept agent from a Gate 1 agent or a floor policy.
- **Outcome:** mean `EpisodeStats.damage_taken` across the 25 configurations,
  per D035. Lower is better. Success rate, saturation and spread are recorded
  as secondary descriptives and never used to rank.
- **Checkpoint:** each run's final checkpoint at step 500000.

Exact command per cell:

```
OMP_NUM_THREADS=1 python scripts/train_dqn.py \
    --set env_id=combat:mix --set total_timesteps=500000 \
    --set <key>=<value> --seed <n> --run-dir runs/hpsweep/<setting>/...
```

**Provenance.** All 27 runs at commit `b06dfa2`, Python 3.13.16, NumPy 2.5.3,
Linux `6.18.44-fc-v185`, OpenBLAS pinned to one thread per worker
(`OMP_NUM_THREADS=1` and friends, set in `scripts/hp_sweep.py` before NumPy
import). The working tree was **dirty** at run time: three agents shared the
clone and untracked files from the others were present. No tracked file this
sweep depends on (`combat_env/`, `dqn/`, `evaluation/`, `scripts/train_dqn.py`)
was modified — the default cell reproducing Gate 1 bit-for-bit on two of three
seeds (§8) is direct evidence of that.

Two environment-version notes, neither acted on: this container has NumPy
**2.5.3**, while D033 adopted 2.5.4 and D034's floors were recorded under
2.2.6. And `docs/gate1_floors.json` was **not** recomputed — D034 says floor
values "are not recomputed afterwards", so the G1-a threshold of 60.3897 is
used as recorded even though no cell's result depends on it.

**Compute.** A shared 2-core box, with another agent's CleanRL reference runs
consuming roughly 1.4 of the 2 cores throughout. A 500k-step combat run that
takes about 1.5 minutes unloaded took 5.5–9 minutes here, and evaluation
(5,000 greedy episodes per cell) added 2–4 minutes. That contention, not the
algorithm, is what forced the grid cut in §3.

---

## 5. Results

3 seeds per setting. **Mean damage taken, lower is better.** "vs default" is
`setting − default`, so a **negative** number is an improvement.

| setting | seed 1 | seed 2 | seed 3 | 3-seed mean | SD | vs default | improvement |
|---|---|---|---|---|---|---|---|
| **`default`** | 25.913 | 29.564 | 30.215 | **28.564** | 2.319 | — | — |
| `buf3k` | 30.131 | 28.976 | 27.995 | 29.034 | 1.069 | +0.470 | −0.470 |
| `buf30k` | 31.116 | 25.615 | 28.841 | 28.524 | 2.764 | −0.040 | +0.040 |
| `buf100k` | 30.128 | 23.658 | 39.378 | 31.055 | 7.901 | +2.491 | −2.491 |
| `buf300k` | 23.098 | 25.864 | 25.778 | **24.913** | 1.573 | −3.651 | **+3.651** |
| `lr1e-4` | 33.610 | 45.574 | 47.487 | 42.224 | 7.521 | +13.659 | −13.659 |
| `lr5e-4` | 28.829 | 62.134 | 39.562 | 43.509 | 16.999 | +14.944 | −14.944 |
| `tf5` | 37.725 | 37.746 | 35.590 | 37.021 | 1.239 | +8.456 | −8.456 |
| `tf20` | 82.326 | 41.472 | 30.888 | 51.562 | 27.163 | +22.998 | −22.998 |

Reference values: default's own 3-seed mean **28.564** (the rule-2 reference);
Gate 1 attempt, same seeds 1–3, **30.587**; Gate 1 attempt, all 10 seeds,
**36.250**.

### Secondary descriptives (never used to rank)

"flagged" is the mean number of configurations per seed whose episodes reach the
agent health bound more than 25% of the time (D035's saturation flag). "final
return" is the mean training return over each run's last 100 episodes — a
diagnostic for §7, not a criterion.

| setting | mean win rate | flagged configs | final return | seeds over the G1-a threshold (60.3897) |
|---|---|---|---|---|
| `default` | 0.988 | 0.3 | 127.7 | 0 |
| `buf3k` | 1.000 | 0.0 | 122.8 | 0 |
| `buf30k` | 1.000 | 0.0 | 126.5 | 0 |
| `buf100k` | 1.000 | 0.0 | 130.8 | 0 |
| `buf300k` | 1.000 | 0.0 | 127.5 | 0 |
| `lr1e-4` | 0.997 | 0.0 | 107.3 | 0 |
| `lr5e-4` | 0.932 | 1.7 | 121.6 | **1** (seed 2, 62.134) |
| `tf5` | 0.995 | 0.0 | 122.1 | 0 |
| `tf20` | 0.814 | 4.3 | 97.6 | **1** (seed 1, 82.326) |

---

## 6. D034's rule, applied

**Rule 1.** Outcome = mean damage taken, 3-seed mean per setting. As measured
in §5.

**Rule 2.** Bar: improvement over the default's 3-seed mean (28.564) of **more
than 8.344**.

| setting | improvement | > 8.344? |
|---|---|---|
| `buf300k` | **+3.651** | **no** (44% of the bar) |
| `buf30k` | +0.040 | no |
| `buf3k` | −0.470 | no |
| `buf100k` | −2.491 | no |
| `lr1e-4` | −13.659 | no (worse) |
| `lr5e-4` | −14.944 | no (worse) |
| `tf5` | −8.456 | no (worse) |
| `tf20` | −22.998 | no (worse) |

**Zero of eight settings cleared the bar.** Only two improved on the default at
all, and the larger of the two, `buf300k`, reached less than half the bar. Four
of the eight were *worse* by more than the bar in the adverse direction.

**Rule 3.** Not reached — no setting qualified, so there is no "smallest
qualifying change" to select. The ordering was fixed in advance regardless
(§2).

**Rule 4.** Not triggered. Nothing is adopted, so no new Gate 1 attempt is
required. Had something qualified, running that attempt would have been Ryan's,
not this sweep's.

**Rule 5 — this is the operative branch.**

> If nothing clears the bar, the CartPole defaults stand and that is recorded
> as a result: the agent is not hyperparameter-limited on this environment.

### Recommendation (proposal for Ryan to accept or reject)

**Keep the current `DQNConfig` defaults** — `buffer_size = 10000`,
`learning_rate = 2.5e-4`, `train_frequency = 10` — unchanged for the combat
experiment, and record D034's rule-5 outcome in the D034 attempts log with
this document as its evidence. Suggested wording:

> Hyperparameter sensitivity sweep run 2026-11-02 at commit `b06dfa2`,
> `combat:mix`, 9 settings × 3 seeds × 500k steps, evaluated under the Gate 1
> protocol (`docs/HP_SWEEP.md`). No setting improved mean damage taken by more
> than 8.344; the best, `buffer_size = 300000`, improved it by 3.651. **D034
> rule 5 applies: the CartPole defaults stand and the agent is not
> hyperparameter-limited on this environment over the grid tested.** Rules 3
> and 4 were not reached. The specific concern that motivated the sweep — a
> 10,000-transition buffer holding only ~70 episodes — is **not supported**:
> varying replay capacity 100-fold (3,000 to 300,000) moved the 3-seed mean by
> 6.1 damage, non-monotonically, against a noise bar of 8.344.

Three things this recommendation does **not** say, and should not be read as
saying:

- It does not say the defaults are optimal. It says no tested change is
  distinguishable from seed noise by D034's bar.
- It does not retroactively reinterpret the Gate 1 attempt. Rule 4's principle
  cuts both ways: the existing attempt stands on its own numbers.
- It does not close the question of whether the baseline agent is strong enough
  to resolve between-condition differences. That is a **power** question about
  the primary dependent variable, and §7 argues it is not a hyperparameter
  question at all.

### How much confidence rule 5 deserves here

Stated plainly, because "nothing cleared the bar" is only as strong as the
design's ability to detect something.

| | pooled within-setting SD | SE of a 3-vs-3 mean difference | bar 8.344, in SE |
|---|---|---|---|
| `buffer_size` family (default + 4 cells) | 3.976 | 3.247 | **2.57 SE** |
| all nine settings | 11.372 | 9.285 | 0.90 SE |

So the two halves of the conclusion are **not equally well supported**:

- **For `buffer_size`, the conclusion is solid.** Those five cells are tightly
  behaved (per-setting SD 1.07–7.90), the bar sits at about 2.6 standard errors,
  and the observed best effect is about 1.1 SE. A 100-fold change in replay
  capacity produced a non-monotonic 6.1-damage wobble
  (29.034 → 28.524 → 31.055 → 24.913 as capacity rises). Non-monotonicity at
  this magnitude is the signature of noise, not of a trend the bar happened to
  miss. **`buffer_size` is not what is limiting this agent.**
- **For `learning_rate` and `train_frequency`, the conclusion is weaker in
  principle but not in practice.** Pooled across all nine settings the bar is
  only 0.9 SE, which would normally mean very low power — but that pooled SD is
  inflated by the two settings that *destabilised* the agent (`lr5e-4` SD 17.0,
  `tf20` SD 27.2). Every `lr` and `tf` move was worse than the default, two of
  them by more than the bar in the adverse direction, and both produced a seed
  that would have **failed G1-a** (`tf20` seed 1 at 82.326 and `lr5e-4` seed 2
  at 62.134, against the threshold of 60.3897). The useful finding is not "no
  effect" but "an effect in the wrong direction": the CartPole defaults are not
  a stale leftover here, they sit inside the stable region for this environment
  too, and the stability margin on both sides is thin.

The design can therefore rule out a **large** single-factor hyperparameter
effect, and for `buffer_size` specifically a moderate one. It cannot rule out a
small one. Raising the seed count is the only fix for that, and it is not worth
spending: a hyperparameter effect too small for this design to see is also too
small to plausibly mask between-condition differences, which is the only reason
the question was asked.

---

## 7. What this suggests the limiting factor actually is

Four candidates, in descending order of how much I think they matter. **I own
none of these files and changed none of them.** Each is named for Ryan's
judgement, exactly as the constraint requires.

### 7.1 The reward function is the strongest candidate (`OPEN-06`)

The agent maximises return. `docs/ENVIRONMENT_SPEC.md` §7 gives it as
damage dealt +1.0/pt, damage taken **−1.2/pt**, win **+60**, loss −30, time
−0.05 per decision step, successful dodge +2.0.

D035 ranks on **mean damage taken** — one term of six. The win bonus and the
time penalty actively *pay* the agent to trade damage for a faster kill, so an
optimal policy under this reward has no reason to drive damage taken toward
zero. The metric the experiment reads and the objective the agent optimises are
related but not aligned, and the sweep data shows the gap directly:

- `buf100k` has the **highest** final training return of any setting (130.8)
  and a **worse** mean damage taken than the default (31.055 vs 28.564).
- `buf300k` has the **best** mean damage taken (24.913) on a final return
  indistinguishable from the default's (127.5 vs 127.7).

So the two quantities do not co-move across settings. If the experiment's
dependent variable is damage taken, then tuning the agent to be better at what
it is actually optimising will not reliably move that variable — which is a
much better explanation of the flat sweep than "the hyperparameters happen to
be well chosen".

This is the one I would look at first. **It is also the one I am least able to
touch:** changing the reward is a pre-registration §12 deviation that
invalidates cross-experiment comparison with every run made before it,
including Gate 1a and Gate 1. Naming it, not acting on it.

### 7.2 There is probably a policy-independent floor on damage taken

`docs/ENVIRONMENT_SPEC.md` §6 notes that `damage ≈ (attacks faced) × (fraction
connecting)`, and that the attack-gap axis sets the first term **mechanically**
— halving the gap roughly doubles attacks faced regardless of skill. At the
short-gap configurations some damage is unavoidable for any policy.

The sweep is consistent with a floor around 23–25: the best single seed anywhere
in 27 runs is 23.098, the best setting mean is 24.913, and four different
settings produced a seed in the 23–26 band. Against the random floor of 80.520
and the scripted floor of 46.133, an agent sitting at 24.9–28.6 has already
taken most of the available headroom. There may be very little left for *any*
hyperparameter to win.

A cheap way to find out, which I did not run because it changes the evaluation
I was told not to change: measure damage taken for a hand-written
*damage-avoidant* policy (one tuned to survive rather than to win) under the
same `evaluation/gate1.py` protocol. That would give a third floor —
"best achievable avoidance" — alongside `F_random` and `F_scripted`, and would
say whether 24.9 is near the floor or far from it. **Recommended as a separate
piece of work for Ryan**, since adding a floor is a D034 matter.

### 7.3 Win rate is at the ceiling, which caps what the metric can show

Five of nine settings win **100.0%** of all 5,000 evaluation episodes, and the
default wins 98.8%. This re-confirms D035's reason for abandoning success rate,
but it also means the only live signal is how much damage the agent eats while
winning something it was going to win anyway. Saturation is correspondingly
near zero: every `buffer_size` cell flags **no** configurations at D035's 25%
threshold, better than the Gate 1 attempt (which flagged configuration 24 for
seed 2) and better than the scripted floor's two flagged cells. The agent is
not dying; it is being chipped.

### 7.4 Deliberate handicaps, working as designed

The 6-tick (200 ms) observation delay (`ENVIRONMENT_SPEC.md` §4) is a
deliberate human-like limitation and caps reactive performance at the shortest
telegraph levels regardless of agent capacity. Not a defect — but it is part of
why the short-telegraph configurations do not improve with more replay data.

### 7.5 What was not swept

Network capacity (`hidden_sizes = (120, 84)`), `batch_size`,
`target_network_frequency`, `learning_starts`, `gamma`, and the exploration
schedule were all left at their defaults, as D034 scoped the sweep to three
fields. The sweep gives weak evidence against a *data*-starvation story — 100×
more replay capacity changed nothing — which argues indirectly against capacity
being the thing a bigger buffer would have fed, but does not test capacity
directly. If Ryan wants one more sweep, `hidden_sizes` is the field I would
pick, and it would need its own decision rule fixed in advance.

---

## 8. Reproducibility finding: the Gate 1 attempt's seed 3 does not reproduce

Not part of the sweep, but found by it and worth recording.

The default cell used seeds 1–3 deliberately, so it re-runs three of the ten
Gate 1 agents at the current commit. Comparing per-configuration mean damage
taken against `docs/gate1_result.json`, all 25 values each:

| seed | this sweep | Gate 1 attempt | configurations differing (of 25) |
|---|---|---|---|
| 1 | 25.91308 | 25.91308 | **0** |
| 2 | 29.56396 | 29.56396 | **0** |
| 3 | **30.21538** | **36.28376** | **25** |

Seeds 1 and 2 are **bit-identical across all 25 configurations** — which is
strong evidence that the environment, the training loop, the network and the
evaluation harness at commit `b06dfa2` are numerically identical to what
produced the Gate 1 attempt, and that this sweep's measurements are comparable
to it. Seed 3 differs on **every** configuration, and its spread differs too
(49.068 here against 41.796 recorded), so it is a genuinely different trained
policy rather than an evaluation difference.

**The cause cannot be determined from the repository.** `runs/` is git-ignored
and absent from this clone, and the Gate 1 commit `1287319` is not in the
shallow clone's history, so neither the seed-3 run's `meta.json` nor its
`config.json` can be inspected. A code change is ruled out by seeds 1 and 2
being bit-identical. Two observations that may be relevant:

- `scripts/train_dqn.py` does **not** pin BLAS thread counts, and the
  `meta.json` it writes does not record them — unlike
  `scripts/run_cleanrl_reference.py`, which pins `OMP_NUM_THREADS` and
  `MKL_NUM_THREADS` and records them as `thread_env`. A one-ULP difference in a
  matmul reduction can flip a tie in `greedy_action` (documented as
  lowest-index ties) and send a single run down a different trajectory while
  leaving most runs bit-identical. That matches the observed pattern — but it is
  a hypothesis, not a demonstration; this sweep pinned threads throughout and so
  cannot test it.
- The container's NumPy is 2.5.3; D033 adopted 2.5.4 and D034's floors record
  2.2.6. D033 verified that 1.26.4 → 2.5.4 left every CartPole
  `episodes.jsonl` hash-identical, so a NumPy difference is the less likely of
  the two.

**Suggested for Ryan, not done here:** compare `meta.json` and `config.json`
across `runs/gate1/mix/*seed3*` and its siblings — argv, commit, dirty flag,
NumPy version, start time. If nothing distinguishes seed 3, the honest
conclusion is that combat training runs are **not reliably bit-reproducible**,
which is worth a line in D034 and an argument for pinning and recording thread
counts in `train_dqn.py` the way the CleanRL runner already does.

It does **not** disturb the Gate 1 verdict: 36.284 and 30.215 are both far under
the G1-a threshold of 60.3897, so G1-a and G1-b pass either way. And it does not
affect §6, because rule 2's reference is this sweep's own default cell,
measured in this environment at this commit.

---

## 9. Reproducing this

```bash
# 27 runs; resumable — a cell whose eval_seed<N>.json exists is skipped.
python scripts/hp_sweep.py run --workers 2

# aggregate and apply D034's rule
python scripts/hp_sweep.py report

# print the grid, including each setting's change_rank
python scripts/hp_sweep.py grid
```

Per-run records land at
`runs/hpsweep/<setting>/eval_seed<N>.json` — resolved config, git commit and
dirty flag, checkpoint path and step, all 25 per-configuration means, success
rates and saturation fractions, plus wall times. The aggregate, including the
rule-5 verdict, lands at `runs/hpsweep/summary.json`. `runs/` is git-ignored, so
every number this document's conclusions rest on is also in the tables above.
