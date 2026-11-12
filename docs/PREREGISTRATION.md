# Pre-Registration

**Neuroadaptive Curriculum Shaping for Reinforcement Learning**

- **Version:** 1.0
- **Status:** STAGE-1 LOCKED (see §0)
- **Date:** 2026-10-05
- **Author:** Ryan Johnson, B.S. Artificial Intelligence Engineering, Neumont College of Computer Science
- **Repository:** this repository, tagged `prereg-v1.0`
- **Supersedes:** draft v0.1 (2026-09-04), which predates the accepted proposal

---

## 0. Lock status and procedure

This document uses a **two-stage lock**, because two parameters cannot honestly
be fixed before the baseline runs exist: the number of random seeds per
condition and the smallest effect size of interest both depend on the
seed-to-seed variance of this environment, which is measured at Gate 1.

**Stage 1 — locked now (2026-10-05).** Everything that must not be contaminated
by seeing results: research question, conditions, hypotheses, dependent
variables, analysis method, falsification criteria, gates, and the
confirmatory/exploratory split. Tagged in the repository.

**Stage 2 — one pre-specified amendment, after Gate 1.** Seed count and smallest
effect size are filled in from the variance observed in the non-EEG baseline
runs. Baseline runs are not experimental-condition data, so using them this way
does not compromise the pre-registration, but the amendment is declared here in
advance rather than explained afterward, and is recorded in §12 with its date and
an explicit statement that no condition data had been collected.

Any other change after Stage 1 is a deviation and is recorded in §12.

---

## 1. Research question

> Does an EEG-derived estimate of human task difficulty provide a useful ordering
> of training configurations for curriculum generation in reinforcement learning,
> compared with static, random, shuffled-EEG, agent-performance, human-behavioural,
> and reverse-ordered curricula?

### Architectural invariant

EEG-derived information influences **only** the distribution over environment
configurations presented during training. It never enters the agent's
observation space, action space, or reward function. The agent's decision problem
is identical in every condition; only the sampling distribution differs.

This invariant is what makes the study a curriculum experiment rather than a
control interface, and it is what makes the null hypothesis meaningful. It is not
revisable without a superseding ADR.

---

## 2. Competing mechanisms

| ID | Claim | Distinguished by |
|---|---|---|
| M1 | Human difficulty transfers to the agent | NEURO beats SHUFFLE |
| M1-inv | Human difficulty transfers **inverted** | REVERSE beats SHUFFLE while NEURO does not |
| M2 | Any structured non-uniform ordering helps | SHUFFLE beats RANDOM but ties NEURO |
| M3 | No effect | NEURO, REVERSE, SHUFFLE and RANDOM all tie |

M1-inv is motivated by Moravec's paradox (Moravec, 1988): perception and
reaction-speed demands that dominate human difficulty may be trivial for an
agent. Preliminary evidence from this environment is consistent with that
asymmetry (§5.3).

---

## 3. Hypotheses

**H1 (primary, directional).** NEURO achieves higher held-out performance than
SHUFFLE, RANDOM, and STATIC.

**H2 (primary, directional).** NEURO achieves higher held-out performance than
PERF and BEHAV.

**H5 (primary, directional).** REVERSE achieves higher held-out performance than
SHUFFLE.

H1 and H5 form a two-member directional family. Support for either is evidence
that human-derived ordering carries usable information; direction is the
secondary question. Both sit inside the Holm-corrected family of §9 with
directions fixed in advance.

**H2 is the justification test.** If H1 holds but H2 fails, the EEG signal carries
real information yet offers no advantage over signals that are already free. That
is recorded now as a legitimate finding and will be reported as such.

**H3 (secondary).** NEURO reaches a fixed performance threshold on the training
set in fewer environment steps.

**H4 (secondary).** NEURO attains higher final performance on the training set
under a fixed step budget.

**Prior expectation, recorded to prevent post-hoc reframing:** generalization
(H1/H2/H5) is the most defensible claim. Sample efficiency (H3) is least likely
to show a clean effect, since a curriculum driven by human difficulty perception
is not obviously aligned with what is efficient for a value-based learner. H3 and
H4 are reported regardless of outcome.

---

## 4. Conditions

Independent variable: curriculum condition, seven levels, between-runs, crossed
with random seed. All conditions share identical environment code, agent
architecture, hyperparameters, and training budget. **Only the source of the
ordering over configurations differs.**

| ID | Ordering source |
|---|---|
| `STATIC` | None; fixed configuration throughout (configuration 7, `DECISIONS.md` D040) |
| `RANDOM` | None; uniform i.i.d. sampling over the training set |
| `NEURO` | EEG-derived human difficulty index **(experimental condition)** |
| `PERF` | The agent's own running per-configuration mean damage taken |
| `BEHAV` | Human per-configuration mean damage taken and median reaction time, ranked separately (mid-ranks), ordered by the mean of the two ranks |
| `REVERSE` | EEG-derived index, rank-inverted |
| `SHUFFLE` | EEG-derived index values permuted across configurations |

`SHUFFLE` permutes the assignment of difficulty **values** to **configurations**,
preserving the marginal distribution of values while destroying the
configuration-to-difficulty mapping. It does not shuffle EEG in time. The
permutation is drawn once per seed from that seed and logged.

`REVERSE` and `SHUFFLE` are deterministic transforms of the `NEURO` ordering and
require no additional data collection.

`PERF` and `BEHAV` were first defined by success rate. That rate sits at its
ceiling in this environment, as D035 found, which would leave the `PERF`
ordering almost all ties. On 2026-11-12, before any condition data existed,
both were moved to mean damage taken, the quantity that ranks
configurations at every gate. See `DECISIONS.md` D037 and §12.

**Sampler requirement.** `NEURO`, `PERF`, `BEHAV`, `REVERSE` and `SHUFFLE` run
through **one shared frontier-sampler implementation** with the ordering source
injected as a dependency. A unit test asserts this. If the samplers differed
between conditions the comparison would be confounded.

---

## 5. Environment

### 5.1 Configuration space

Two axes, five levels each, 25 configurations, `config_id = telegraph_level * 5 + gap_level`.

| Axis | Levels | Mechanism |
|---|---|---|
| Telegraph duration | 1.00, 0.66, 0.42, 0.24, 0.12 s | Compresses the window to recognise and respond to an attack |
| Attack gap | 1.60, 1.25, 0.95, 0.72, 0.52 s | Removes recovery windows; forces stamina management |

All other parameters constant across configurations.

### 5.2 Held-out set

Five of the 25 configurations are reserved for generalization evaluation and are
never sampled during training in any condition. The split is fixed before any
experimental runs and recorded in the repository.

**Selection rule (fixed 2026-11-12, before the draw; `DECISIONS.md` D038).**
1. The held-out set is a *transversal* of the 5 × 5 grid: exactly one
   configuration at each telegraph level and exactly one at each gap level.
   Training then keeps four levels of each axis, and the test covers each
   axis's full range.
2. Transversals containing configuration 0 (easiest) or 24 (hardest) are
   excluded. A held-out cell at the ceiling or at saturation spends a fifth
   of the primary DV on a cell where conditions cannot differ. This leaves 78
   of the 120 transversals.
3. One of the 78 is drawn uniformly by `scripts/draw_heldout.py`, with seed
   `HELDOUT_SEED = 20261112`. The seed is committed before the draw. The draw
   refuses to run from a tree with uncommitted changes, and it writes
   `docs/heldout_split.json` once.

Gate 1's per-configuration results were seen before this rule was written.
The draw is random under a committed seed, so that knowledge cannot steer
which configurations are held out.

### 5.3 Axis calibration (completed 2026-10-05)

The telegraph range was widened from an earlier (0.70 … 0.24 s) setting after a
scripted-policy sweep showed the axis was nearly inert for the agent. Measured
main effects, 30 episodes per cell:

| Setting | Telegraph effect on damage taken | Gap effect | Telegraph effect on hit rate |
|---|---|---|---|
| Original (0.70–0.24 s) | 16.6 | 43.5 | 0.016 |
| **Selected (1.00–0.12 s)** | **30.6** | **45.0** | **0.205** |

Hit rate (`times_hit / attacks_faced`) is reported alongside damage taken because
damage conflates two quantities: damage ≈ attacks faced × fraction connecting.
The gap axis controls the first term mechanically while telegraph affects only
the second, so damage taken structurally favours the gap axis. Under the original
setting the telegraph axis moved hit rate by 0.016, which is nothing; under the
selected setting it moves it by 0.205.

Dodge invulnerability duration was also swept (0.20 / 0.12 / 0.10 s) and showed
**no measurable effect** under the scripted policy; it is therefore left
unchanged at 0.20 s rather than altered without evidence.

**Residual limitation.** At the sparsest attack gap, damage taken still saturates
across the three shortest telegraph levels, because encounters end before
telegraph length can express itself. The axes remain imperfectly balanced
(telegraph effect ≈ 0.68 × gap effect). A scripted policy is not a trained agent,
and Gate 2 is the real test.

---

## 6. Difficulty standardization across measurement systems

Three systems measure difficulty in incompatible units. They are made comparable
by reduction to **rank over the same fixed 25 configurations**.

| Gate | Quantity | Reduction |
|---|---|---|
| Gate 1 (agent) | Per-configuration mean damage taken, trained baseline agent | Rank |
| Gate 2 (human) | Per-configuration mean damage taken and median reaction time | Rank |
| Gate 3 (EEG) | Median workload index over clean windows per configuration | Rank |

**Direction.** Mean damage taken increases with difficulty, so it ranks in the
*opposite* direction to a performance measure. Every correlation in §6 and §10
is computed on the difficulty ordering, and the expected sign is stated with
each hypothesis rather than inferred from the metric.

**Saturation.** Damage taken is effectively bounded by agent maximum health —
the episode ends at death, though the fatal blow may overshoot it — so the
measure compresses wherever an agent dies often. The fraction of episodes reaching that
bound is reported per configuration alongside the mean, so compression is
visible rather than silent. If it exceeds 25% of episodes in a configuration,
that configuration's rank is flagged as unreliable in the analysis.

**Relationship to §8.** Damage taken is used here as the quantity that *orders
configurations*. Its status in §8 as an exploratory dependent variable is
unchanged, and it is not promoted to a hypothesis-tested outcome of the
experiment. The primary dependent variable remains mean normalized return on
the held-out configuration set. These are distinct roles: one defines the
curriculum's ordering, the other measures whether the curriculum helped.

**Pooling the agent ordering (fixed 2026-11-12, before the human session; D036).**
The agent side of Gate 2 is the per-configuration mean damage taken, pooled
over every evaluation episode of every Gate 1 seed (10 × 200 per
configuration), then ranked. Single-seed orderings are too noisy to stand in
for "the agent": pairwise Spearman between seeds averages +0.725 and falls as
low as +0.251. The ten per-seed correlations with the human ordering are
reported descriptively as an uncertainty band. They are not the gate.

**Flagged configurations stay in.** A configuration flagged unreliable for
saturation, on either side, stays in the primary correlation, so the gate
always runs over all 25. The correlation with the flagged configurations
removed is reported as a sensitivity analysis.

Agreement between systems is measured by Spearman rank correlation, with
mid-ranks for ties (D034). The EEG index
is never converted into a damage-taken equivalent or a difficulty percentage; it
sorts 25 configurations and nothing more. The frontier sampler consumes an
ordering, never calibrated magnitudes.

### Interpreting the Gate 3 correlation

Gate 3 and H2 pull in opposite directions, so the threshold is fixed in advance.
Gate 3 wants the EEG ranking to agree with behaviour, as evidence of validity. H2
requires the two to differ, because `BEHAV` is itself a condition and identical
rankings mean the EEG contributes nothing new.

The informative outcome is **partial agreement**. A Spearman correlation at or
above 0.90 between the EEG and behavioural rankings is pre-registered as evidence
of **redundancy**, not success, and will be reported as such.

---

## 7. EEG measurement

**Subject:** one (the author). Single-subject design. No cross-subject or
population claim is made or supported.

**Index:** a fixed, literature-based spectral formula. **No model is trained**, and
no parameters are transferred from any other dataset or task.

- Primary term: standardized frontal midline theta power (4–8 Hz, Fz with F3/F4).
- Secondary term: down-weighted standardized parietal alpha power (8–13 Hz, Pz with P3/P4).
- Weighting follows the meta-analytic effect sizes in Chikhi et al. (2022): theta g = 0.68, alpha g = −0.25.
- Standardization is within-session. Each configuration receives the median index over its clean windows; the 25 medians are ranked.

**Degraded mode, pre-registered.** If the hardware validation gate fails to
demonstrate an eyes-closed alpha increase at parietal sites, the alpha term is
dropped and the index runs on frontal theta alone. This is a planned contingency,
recorded in §12 when exercised, and must be declared before any experimental data
is collected.

**Public data use.** EEGMAT (PhysioNet, *EEG During Mental Arithmetic Tasks*,
v1.0.0, DOI 10.13026/C2JQ1P, Open Data Commons Attribution License v1.0; 36
subjects, 23 channels, EDF) is used **only** to verify that the index
implementation responds in the expected direction to a known workload contrast.
Nothing learned from it is applied to the author's recordings. There is no model
transfer and therefore no domain-transfer assumption beyond the direction of the
theta and alpha workload relationships themselves, which is listed as a
limitation in §11.

**EEGMAT verification required before use.** The dataset page states "a high-pass
filter with a 30 Hz cut-off frequency" was applied. Taken literally that would
remove everything below 30 Hz, deleting all theta and alpha and making the data
useless for this index. It is almost certainly a typo for *low-pass*, but this
must be verified empirically before the dataset is trusted: load one recording,
compute its power spectrum, and confirm real power exists in the 4–13 Hz band.
This is the first task of the EEG pipeline work.

**Artifact control.** The session includes a block in which movement is
deliberately elicited **without** the difficulty manipulation. If the index
responds to that block it is measuring movement, and Gate 3 fails. Per-configuration
window-rejection rates are logged and correlated with the difficulty ranking; a
strong correlation is reported as a confound rather than ignored.

**Familiarity control.** Familiarization is played to performance plateau before
any recording. Presentation order is randomized and interleaved, never blocked.
Probe configurations are repeated at the start, middle and end of the session to
measure within-session drift directly.

**Subjective instrument.** A 1–9 workload rating per configuration block, on the
scale used by the STEW dataset.

---

## 8. Dependent variables

**Primary.** Mean normalized return on the **held-out configuration set**,
evaluated at end of training over K evaluation episodes per held-out
configuration, exploration disabled, evaluation seeds distinct from training seeds.

*Normalized return (defined 2026-11-12, D039).* For each held-out
configuration c:

    normalized_c = (R_c − R_random,c) / (R_scripted,c − R_random,c)

- R_c is the run's mean episode return on c over K = 200 greedy evaluation
  episodes, on the Gate 1 evaluation seeds (`evaluation/gate1.py`).
- R_random,c and R_scripted,c are the random-action and scripted policies'
  mean returns on the same seeds. They are measured once, before any Stage-2
  or condition run, and committed.
- 0 means random-policy level; 1 means scripted-policy level. Values above 1
  are expected.
- The primary DV for a run is the mean of normalized_c over the five held-out
  configurations.

*Small-denominator fallback.* If a held-out configuration's denominator,
R_scripted,c − R_random,c, is below 10% of the median denominator over all 25
configurations, the median denominator is used for that configuration. This
is decided mechanically when the floors are measured and reported.

*Note on interpretation.* The offset R_random,c cancels in every
between-condition contrast. The denominators therefore only set how much
each held-out configuration weighs in the mean.

*Dependency on the reward.* The anchors are returns under the provisional
reward (OPEN-06). If the reward changes, they are re-measured and the change
is logged in §12.

**Secondary.** Environment steps to a fixed performance threshold on the training
set (sample efficiency); final performance on the training set (asymptotic).

**Exploratory — fixed as exploratory now, not eligible for promotion after seeing
results.** Dodge success rate, damage taken, hit rate, stamina efficiency,
survival time. Reported descriptively with confidence intervals and no hypothesis
tests.

**Instrumentation, reported separately.** Signal quality, window rejection rate,
index performance against labels, index response to the artifact-control block.

---

## 9. Analysis plan

**Unit of analysis.** One completed training run (one seed × one condition). Runs
are independent.

**Seeds.** **40 per condition, 280 runs** (Stage-2 amendment, 2026-11-12; §12).
The same 40 seeds are used in every condition. Seeds 101–110 belong to the
Stage-2 baseline runs and are never used for condition runs (`DECISIONS.md`
D038). The count was set from the between-seed SD of the primary DV in those
baseline runs: 0.093 (mean 1.127, n = 10). Compute is not binding: measured
2026-11-12, a Gate 1-configuration run takes about 2.8 minutes with ten running
in parallel, so 280 runs is roughly 80 minutes at ten workers.
*Stage-1 text: "Target ≥10 per condition, ~70 runs. Final count set in the
Stage-2 amendment from Gate 1 variance."*

**Smallest effect size of interest.** **0.07 normalized-return units**
(Stage-2 amendment, 2026-11-12; §12). The smallest effect of interest is 0.07
normalized-return units, equal to this design's Holm-corrected detection limit
at 40 seeds per condition. This is 55% of the measured gap between the
scripted anchor (1.0) and the non-EEG baseline agent (1.127). The design
cannot resolve smaller effects, and the curriculum literature suggests
plausible true effects may be smaller; this is recorded as a limitation, not
a claim that such effects are uninteresting. Basis and literature:
`DECISIONS.md` D041 and `docs/RELATED_WORK.md` (Wu, Dyer and Neyshabur, 2021).
*Stage-1 text: "Stage-2 amendment, in units of the primary DV."*

**Primary contrast family**, Holm-corrected at family-wise α = .05, directions
fixed in advance:

1. NEURO vs SHUFFLE (isolates M1 from M2)
2. NEURO vs PERF (H2)
3. NEURO vs BEHAV (H2)
4. NEURO vs RANDOM
5. NEURO vs STATIC
6. REVERSE vs SHUFFLE (H5)

**Inference.** Permutation tests for p-values; stratified bootstrap confidence
intervals on the difference of **interquartile means**, not raw means, following
Agarwal et al. (2021). Effect size and probability of improvement reported with
every contrast.

**Reporting.** All conditions and all seeds. A seed is dropped only for a
pre-specified logged technical failure (crashed run), recorded in §12.

**Secondary analyses** use the same family structure, corrected within their own
family and labelled secondary.

---

## 10. Gates

**Gate 1 — the learning problem works.** A non-EEG baseline agent learns the task
to non-trivial performance, reproducibly across seeds, and its per-configuration
mean damage taken is ordered rather than uniform. Produces the variance estimate for
the Stage-2 amendment.
*Fail →* environment, reward, or algorithm is wrong. No EEG work proceeds.

**Gate 1a — algorithm correctness (prerequisite).** The from-scratch DQN passes
component unit tests, solves a standard benchmark task to its known threshold,
and tracks a reference implementation's learning curve on that task. Completed
before the agent touches the combat environment, so that a Gate 1 failure is
isolated to the environment rather than ambiguous.
*Operational criteria* (benchmark, reference implementation, the four levels and
their pass criteria, and the rule for calibrating thresholds on reference runs)
are defined in `docs/DECISIONS.md` **D029**. Added 2026-10-07 as a clarification;
see §12.

**Gate 2 — human/agent difficulty correspondence.** Spearman correlation between
human and agent per-configuration rankings is positive and non-trivial. **Uses no
EEG.**
*Operational criterion (fixed 2026-11-12, before the human session; D036).*
- **The test:** Spearman ρ (mid-ranks) between two orderings over all 25
  configurations:
  - the human per-configuration mean damage taken;
  - the pooled agent ordering of §6.
- **Pass condition:** a one-sided permutation test gives p < .05, using
  10,000 permutations of the human ordering with seed 0. With 25
  configurations this is roughly ρ > 0.34.
- **Expect it to pass.** The random-action floor already correlates +0.96
  with the trained agent, so most agreement is the grid's built-in
  difficulty. The configurations where human and agent ranks disagree most
  are therefore pre-registered as a descriptive result. That disagreement is
  the evidence that bears on M1-inv.
- **Reaction time:** median reaction-time rank is reported against the agent
  ordering as a secondary correlation. It is not part of the gate.
*Fail →* the A3 premise does not hold for this environment. Adjust axes or
human-like agent constraints and re-run before any EEG work.

**Gate 3 — the index is real.** The EEG ranking correlates with behavioural and
subjective rankings (but see the redundancy threshold in §6), and does **not**
track the artifact-control block.
*Fail →* NEURO cannot run as specified; exercise §11 contingency.

**Hardware validation gate.** Per-channel impedance acceptable; eyes-closed alpha
increase demonstrated at parietal and occipital sites; blink and jaw-clench
responses confirmed; 60 Hz notch set and both ear-clip electrodes verified.

---

## 11. Falsification criteria and contingency

| Observed pattern | Interpretation |
|---|---|
| NEURO > SHUFFLE and NEURO > PERF, BEHAV | M1 supported with practical advantage. Strongest outcome. |
| NEURO > SHUFFLE, NEURO ≈ PERF or BEHAV | M1 supported, H2 refuted. EEG carries information but no advantage over free signals. |
| REVERSE > SHUFFLE, NEURO ≈ SHUFFLE | M1-inv. Human difficulty transfers inverted — the Moravec outcome. |
| NEURO ≈ SHUFFLE, both > RANDOM/STATIC | M2. Benefit is structure, not human information. |
| NEURO ≈ REVERSE ≈ SHUFFLE ≈ RANDOM | M3. Null for the primary hypothesis. |
| NEURO < RANDOM | Human ordering actively harmful. Reported and discussed. |

**H1 is refuted** if the NEURO vs SHUFFLE contrast fails Holm-corrected
significance with a confidence interval excluding the Stage-2 smallest effect size.

**Null results.** A null obtained from this design is an acceptable and complete
outcome. The prior probability that NEURO cleanly beats all baselines is judged
below 50%. The project is a falsification test, not a demonstration.

**EEG withdrawal contingency.** Decision point at end of week 6. If the recording
has not happened or Gate 2 fails, the EEG component is withdrawn and a reduced
study is executed: environment, baseline, human and agent difficulty
measurements, and the non-EEG curriculum comparisons (STATIC, RANDOM, PERF,
BEHAV). Same statistical treatment. This branch is pre-registered now so that
taking it is a plan, not a redesign under pressure.

---

## 12. Deviations log

Every change after the Stage-1 lock is recorded here with date, description,
reason, and whether outcome data had been seen.

| Date | Section | Change | Reason | Outcome data seen? |
|---|---|---|---|---|
| 2026-10-07 | §10 Gate 1a | **Clarification, no design change.** Gate 1a's criteria operationalised in `DECISIONS.md` D029: CartPole-v1 benchmark, CleanRL `dqn.py` reference, four levels, and pass-criterion form. Thresholds are calibrated on reference runs before any NumPy-DQN benchmark run. Pointer added to §10. | §10 said "tracks a reference implementation's learning curve" without defining *tracks*. The criterion has to be fixed before the curves exist rather than judged afterward. | No. No DQN, reference, or experimental-condition data existed. |
| 2026-10-14 | §5 Environment | **Environment behaviour change.** A tick on which agent and enemy both reach 0 health now resolves as `loss` (previously `win`, which paid the +60 win bonus and counted as a success). See `DECISIONS.md` D031. | Checking enemy death first rewarded trading the agent's life for the kill and counted mutual KOs as successes in the per-configuration rates used at Gates 1 and 2. | No. No agent-training, human-session or experimental-condition data existed. |
| 2026-11-01 | §6, §10 Gate 1 | **Ranking quantity changed.** Gates 1 and 2 rank configurations by per-configuration **mean damage taken** instead of success rate. Success rate is retained as a secondary descriptive measure. Direction, saturation reporting and the boundary with §8 added to §6. Candidate metrics were evaluated empirically before the choice; see `DECISIONS.md` **D035**. | Floor measurement (D034) showed a competent policy wins 19 of 25 configurations outright, giving only 7 distinct success-rate values across 25 configurations. A rank built on that is ~76% ties, and the Gate 2 correlation would have been computed over a near-degenerate vector. Mean damage taken gives 22 of 25 distinct values for the same policy, responds to both difficulty axes near-equally (44.2 vs 43.9, ratio 1.01), and is what D026 calibrated the axes on. | No. No agent has been trained on the combat environment, no human session has been run, and no experimental-condition data exists. The defect was found in the pre-training floors, which exist to be measured before the gate they calibrate. |

| 2026-11-12 | §6, §10 Gate 2 | **Clarification, no design change.** Gate 2's agent ordering is defined as pooled across the Gate 1 seeds. Saturation-flagged configurations stay in the correlation, with a sensitivity analysis. "Non-trivial" is made operational as a one-sided permutation p < .05 (10,000 permutations, seed 0). See `DECISIONS.md` D036. | §10 required "positive and non-trivial" without defining either term, and §6 did not say how ten agent seeds become one ordering. Both must be fixed before the human session. Otherwise they become choices made after seeing the correlation. | Gate 1 baseline agent data had been seen (per-configuration damage, seed-to-seed rank stability). No human-session, EEG or condition data existed. |
| 2026-11-12 | §4 | **Independent-variable definition changed.** `PERF` and `BEHAV` order configurations by mean damage taken; `BEHAV` uses the mean of its damage and reaction-time mid-ranks. They were previously defined by success rate. See `DECISIONS.md` D037. | D035 showed success rate at ceiling in this environment (agent win rates 0.966–1.000). A `PERF` ordering built on it would be almost all ties, and the frontier sampler would act close to random. That would weaken the H2 contrasts in a direction that favours `NEURO`. | Gate 1 baseline agent data had been seen. No human-session, EEG or condition data existed. No curriculum sampler existed. |
| 2026-11-12 | §5.2 | **Clarification, no design change.** The held-out selection rule is fixed: a random transversal excluding configurations 0 and 24, drawn with `HELDOUT_SEED = 20261112`, committed before the draw. See `DECISIONS.md` D038. | §5.2 said "five of 25" without saying which five or how they are chosen. | Gate 1 per-configuration results had been seen. The set is drawn at random under a seed committed before the draw, so that knowledge cannot steer it. No condition data existed. |
| 2026-11-12 | §8 | **Clarification, no design change.** "Normalized return" is defined as (R − R_random) / (R_scripted − R_random) per held-out configuration, averaged over the five, with K = 200 and a small-denominator fallback. See `DECISIONS.md` D039. | §8 named the primary DV but never defined the normalization, and the Stage-2 smallest effect size has to be stated in its units. | The Gate 1 baseline agents' training-episode returns exist in their run logs. No greedy evaluation return on any configuration had been computed, and floor returns had not been measured. No condition data existed. |
| 2026-11-12 | §4, §6, §10 Gate 2 | **Clarification, no design change.** Reaction time is defined: onset of TELEGRAPH to the first new dodge or movement input, in simulation ticks, window to the end of RECOVERY, anticipations under 100 ms excluded from the median, non-responses and censored attacks counted separately. `scripts/play.py` logs the raw per-step input. See `DECISIONS.md` D037. | §4 and §10 used median reaction time without defining it, and the recorder did not log it. It must be fixed before the human session. | No human session had been run. Gate 1 agent data had been seen; it contains no reaction times. No condition data existed. |
| 2026-11-12 | §4 | **Clarification, no design change.** `STATIC`'s fixed configuration is 7, the upper median of the training configurations by the pooled Gate 1 agent ordering. See `DECISIONS.md` D040. | §4 said "fixed configuration" without naming one. | Gate 1 per-configuration agent results had been seen and the rule uses them. No condition, human or EEG data existed. |
| 2026-11-12 | §9 | **Stage-2 amendment, pre-specified in §0.** Seeds per condition set to **40 (280 runs)**, from the Stage-1 target of "≥10 per condition, ~70 runs": a fourfold increase. Smallest effect size of interest set to **0.07 normalized-return units**, the Holm-corrected detection limit at 40 seeds and 55% of the gap between the scripted anchor (1.0) and the baseline agent (1.127). Effects smaller than this cannot be resolved; that is recorded as a limitation, not a claim that they are uninteresting. See `DECISIONS.md` D041. | §0 deferred both until seed-to-seed variance had been measured. The Stage-2 baseline runs (seeds 101–110, uniform sampling over the 20 training configurations, D038) give a primary-DV SD of 0.093. At 10 seeds per condition, only differences of roughly 0.12 DV units or more would be reliably detectable; at 40, roughly 0.05–0.07 (D041). | **No condition data had been collected.** No run of any of the seven conditions existed, and the curriculum sampler had not been written. The agent data seen were the Gate 1 baseline runs (D034) and the Stage-2 baseline runs (D038). The latter use `RANDOM`'s training distribution, but on reserved seeds that are never reused or pooled into `RANDOM`, so they are not condition data. |

**Stage-2 lock complete (2026-11-12).** Any further change is a deviation.

---

## 13. References

Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A., & Bellemare, M. G. (2021). Deep reinforcement learning at the edge of the statistical precipice. *Advances in Neural Information Processing Systems*, 34, 29304–29320.

Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum learning. *Proceedings of the 26th International Conference on Machine Learning*, 41–48.

Chikhi, S., Matton, N., & Blanchet, S. (2022). EEG power spectral measures of cognitive workload: A meta-analysis. *Psychophysiology*, 59(6), e14009.

Jiang, M., Grefenstette, E., & Rocktäschel, T. (2021). Prioritized level replay. *Proceedings of the 38th International Conference on Machine Learning*, PMLR 139, 4940–4950.

Moravec, H. (1988). *Mind Children: The Future of Robot and Human Intelligence*. Harvard University Press.

Zyma, I., Tukaev, S., Seleznov, I., et al. (2019). Electroencephalograms during mental arithmetic task performance. *Data*, 4(1), 14. Dataset: PhysioNet EEGMAT v1.0.0.
