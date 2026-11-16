# Completion plan

**Written 2026-11-16.** Build ends 2026-12-06; week 10 is presentation.
Ordered by dependency and by how much risk each item retires. Owner column:
**R** = Ryan (learning-critical: algorithms, statistics, the index, the
sampler), **C** = Claude (harnesses, logging, plotting, scaffolding).

Status at writing: Gates 1a and 1 passed. D035-D041 recorded. No EEG
recording of the author exists; no human session data exists.

---

## Today — the session (the single highest-value hour in the project)

The montage is rebuilt **first**, because §7's degraded-mode clause triggers on
failure to show eyes-closed alpha **at parietal sites**. Validating on the old
occipital montage would not test that clause.

| # | Task | Owner |
|---|---|---|
| T1 | Rebuild to the index montage: Fp1, Fp2, F3, Fz, F4, P3, Pz, P4 | R |
| T2 | Saline, both ear clips, impedance to green, then **turn impedance checking off** | R |
| T3 | 3-minute validation recording: 60 s eyes open, 60 s eyes closed, 10 blinks, jaw clench, head turns | R |
| T4 | Analyse: does alpha rise eyes-closed at Pz/P3/P4? Decides the hardware gate **and** whether degraded mode triggers | C |
| T5 | Familiarisation to performance plateau (§7 requires it **before** recording) | R |
| T6 | Rest baseline: 2 min eyes open, 2 min eyes closed | R |
| T7 | Movement block: movement **without** difficulty manipulation (§7 artifact control, a Gate 3 input) | R |
| T8 | Main session: `play.py --session --episodes-per-config 6`, EEG recording throughout | R |
| T9 | Record the wall-clock offset between EEG start and session start; verify alignment afterwards by checking the movement block appears as artifact at the expected time | R+C |

**One session, two gates.** §6 takes the human ranking from per-configuration
mean damage taken and median reaction time — both produced by the behavioural
log. So Gate 2 is unblocked by today *whether or not the EEG is usable*.

---

## Week 7 (to 2026-11-22)

| # | Task | Owner |
|---|---|---|
| T10 | `scripts/gate2_check.py`: load session, per-config damage + median RT, rank, Spearman vs pooled agent ordering, permutation null, apply D036 criteria | C harness / R statistics |
| T11 | **Run Gate 2.** Record the attempt in D036 whatever the outcome | R |
| T12 | `eeg/index.py`: the workload index. Theta (Fz, F3, F4) minus down-weighted alpha (Pz, P3, P4), within-session standardised, median per configuration | R |
| T13 | Gate 3: EEG ranking vs behavioural ranking; **and** the movement-block check — if the index responds to the movement block it is measuring movement and Gate 3 fails | R |
| T14 | `scripts/power_sim.py`: power of the registered test (IQM difference, permutation, Holm over 6) at Δ = 0.07; sweep n; find smallest n reaching 80% | R logic / C harness |
| T15 | Record the resulting seed count in D041 | R |

---

## Week 8 (to 2026-11-29, Thanksgiving week)

| # | Task | Owner |
|---|---|---|
| T16 | `curriculum/`: **one** sampler implementation, ordering source injected. The prereg requires a test asserting all seven conditions use the same code path | R |
| T17 | Wire the sampler into `_MixtureEnv._draw_config` — the seam already exists | R |
| T18 | Pilot: 2 seeds x 7 conditions, confirm conditions differ only in sampling order and nothing else | C |
| T19 | `scripts/run_experiment.py`: launch 7 conditions x N seeds (1001-10xx), manifest per run, parallel workers | C |
| T20 | **Launch the full experiment.** ~2 h compute, ~1 MB per run | R |
| T21 | Verify every run completed; no partial runs in the analysis set | C |

---

## Week 9 (to 2026-12-06) — analysis and writing

| # | Task | Owner |
|---|---|---|
| T22 | Primary analysis: IQM differences on the held-out set, stratified bootstrap CIs, permutation tests, Holm over the primary family | R |
| T23 | Apply the three-branch decision rule: effect supported / equivalence (evidence for M3) / inconclusive | R |
| T24 | Secondary and exploratory measures, reported descriptively with CIs | R |
| T25 | Figures: extend `scripts/make_figures.py` with the condition comparison | C |
| T26 | `docs/REPORT.md`: question, design, methods, gates, results, limitations | R draft / C structure and review |
| T27 | Limitations section: single subject, SESOI at the detection limit, Ewing et al. precedent, EEGMAT transfer assumption | R |
| T28 | Slides | C draft / R delivers |

---

## Week 10 — presentation

---

## Fallback, pre-declared

If today's session yields no usable EEG, the capstone still stands and the
report changes shape rather than collapsing:

- Gates 1a and 1 passed, with criteria fixed before the runs that tested them
- Gate 2 decided from behavioural data alone
- EEG pipeline built and validated against public data
- Hardware gate failed, with the impedance and spectral measurements showing why

That is a pre-registered study reporting an honest negative on the
instrumentation, not a failed capstone. Do not improvise a substitute
experiment under time pressure; the deviations log exists so that a documented
stop is a result.

## Cut order, if time runs short

1. Phase 8 entirely (already deferred)
2. Seed count down to the simulation's floor
3. The `REVERSE` condition (H5 is secondary; `SHUFFLE` already separates M1 from M2)

Never cut `SHUFFLE`, `PERF` or `BEHAV`. Those controls are the contribution.
