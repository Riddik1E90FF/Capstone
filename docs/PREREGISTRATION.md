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
| `STATIC` | None; fixed configuration throughout |
| `RANDOM` | None; uniform i.i.d. sampling over the training set |
| `NEURO` | EEG-derived human difficulty index **(experimental condition)** |
| `PERF` | The agent's own running per-configuration success rate |
| `BEHAV` | Human success rate and median reaction time |
| `REVERSE` | EEG-derived index, rank-inverted |
| `SHUFFLE` | EEG-derived index values permuted across configurations |

`SHUFFLE` permutes the assignment of difficulty **values** to **configurations**,
preserving the marginal distribution of values while destroying the
configuration-to-difficulty mapping. It does not shuffle EEG in time. The
permutation is drawn once per seed from that seed and logged.

`REVERSE` and `SHUFFLE` are deterministic transforms of the `NEURO` ordering and
require no additional data collection.

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
| Gate 1 (agent) | Per-configuration success rate, trained baseline agent | Rank |
| Gate 2 (human) | Per-configuration success rate and median reaction time | Rank |
| Gate 3 (EEG) | Median workload index over clean windows per configuration | Rank |

Agreement between systems is measured by Spearman rank correlation. The EEG index
is never converted into a success-rate equivalent or a difficulty percentage; it
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

**Seeds.** Target ≥10 per condition, ~70 runs. Final count set in the Stage-2
amendment from Gate 1 variance. Backend benchmarking indicates compute is not
binding: measured 2,118 DQN updates/sec at batch 64 on the target machine, giving
roughly 8 minutes per million-step run and under two hours for all 70 runs across
eight single-threaded worker processes.

**Smallest effect size of interest.** Stage-2 amendment, in units of the primary DV.

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
success rates are ordered rather than uniform. Produces the variance estimate for
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

**Pending and expected:** the Stage-2 amendment to §9 (seed count, smallest
effect size) after Gate 1.

---

## 13. References

Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A., & Bellemare, M. G. (2021). Deep reinforcement learning at the edge of the statistical precipice. *Advances in Neural Information Processing Systems*, 34, 29304–29320.

Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum learning. *Proceedings of the 26th International Conference on Machine Learning*, 41–48.

Chikhi, S., Matton, N., & Blanchet, S. (2022). EEG power spectral measures of cognitive workload: A meta-analysis. *Psychophysiology*, 59(6), e14009.

Jiang, M., Grefenstette, E., & Rocktäschel, T. (2021). Prioritized level replay. *Proceedings of the 38th International Conference on Machine Learning*, PMLR 139, 4940–4950.

Moravec, H. (1988). *Mind Children: The Future of Robot and Human Intelligence*. Harvard University Press.

Zyma, I., Tukaev, S., Seleznov, I., et al. (2019). Electroencephalograms during mental arithmetic task performance. *Data*, 4(1), 14. Dataset: PhysioNet EEGMAT v1.0.0.
