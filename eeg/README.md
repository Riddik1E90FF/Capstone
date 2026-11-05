# `eeg/` — EEG preprocessing

Loading, filtering, windowing, artifact detection and spectral features.
Developed and tested against EEGMAT (`docs/DECISIONS.md` **D032**), built for
the author's own recordings once the hardware question (**D022**) is settled.

```text
EDF or array
     ↓  eeg.loading      → Recording (float64 µV) + Provenance
     ↓  eeg.filtering    → bandpass, mains notch, edge trim
     ↓  eeg.windowing    → Windows (n_windows, n_channels, n_samples)
     ↓  eeg.artifacts    → ArtifactReport: per-window clean/reject mask + reasons
     ↓  eeg.spectral     → WindowFeatures: band power (n_windows, n_channels, n_bands)
     ╌╌ boundary ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
     ↓  the workload index  ← NOT IN THIS PACKAGE. The author writes it.
     ↓  the curriculum sampler
```

`eeg.pipeline.EEGPipeline` composes the five stages and records every
parameter it used.

## Quick start

```python
from eeg import EEGPipeline, PipelineConfig

pipe = EEGPipeline(PipelineConfig.for_eegmat())
result = pipe.run_edf("data/raw/eegmat/Subject00_1.edf")

result.artifacts.rejection_rate            # 0.111
result.features.power.shape                # (45, 19, 4)
result.features.roi_series(("Fz", "F3", "F4"), "theta", clean_only=True)
result.quality_summary()                   # per-channel amplitudes and rejection
result.save("runs/eeg/subject00_rest")     # JSON metadata + NPZ arrays
```

```bash
python scripts/eeg_demo.py                 # the full report over local subjects
python scripts/eeg_demo.py --download      # fetch EEGMAT subjects first
PYTHONPATH=. python -m pytest tests/test_eeg_*.py -q
```

Dependencies: `numpy`, `scipy` always; `mne` only for `load_edf` and for
montage positions, imported lazily so the array path and most of the test
suite need neither.

---

## Where this package stops

**At `WindowFeatures`.** The workload index of pre-registration §7 —
standardised frontal midline theta minus down-weighted standardised parietal
alpha — is not implemented here. `CLAUDE.md` assigns it to the author ("the
user writes … the workload index computation"), as learning-critical code.
So are the medians, the ranks, and every statistical test in §6 and §9.

### The interface the index plugs into

```python
import numpy as np
from eeg import WindowFeatures, apply_index

def workload_index(features: WindowFeatures) -> np.ndarray:
    theta = features.roi_series(("Fz", "F3", "F4"), "theta")   # (n_windows,) µV²
    alpha = features.roi_series(("Pz", "P3", "P4"), "alpha")   # (n_windows,) µV²
    # within-session standardisation and the Chikhi et al. (2022) weighting
    # go here — §7's formula, the author's to write
    return values        # (n_windows,) float, INCLUDING rejected windows

values = apply_index(result, workload_index)
```

**Contract.**

| | |
|---|---|
| Input | one `WindowFeatures` |
| Output | `np.ndarray`, shape `(n_windows,)`, floating dtype |
| Rejected windows | **must still have an entry.** Any value; `np.nan` is clearest |
| Side effects | none — must not modify the `WindowFeatures` |
| Dependencies | nothing outside its input. In particular **never** the agent's observations, actions or rewards (pre-registration §1) |

`apply_index` checks length, dtype, and finiteness at *clean* windows, and
raises `PipelineError` otherwise. It computes no median, rank or correlation.

**Why the length rule is strict.** An index that returned only its clean
windows would still produce a number for every configuration, and every
downstream median, rank and correlation would still compute — against values
misaligned with the blocks they came from. Nothing would error. That is the
single most dangerous mistake available at this boundary, so it is checked
here rather than discovered in a result.

### What `WindowFeatures` offers

| | |
|---|---|
| `power` | `(n_windows, n_channels, n_bands)`, absolute band power in µV² |
| `relative` | same shape, each band as a fraction of the window's total |
| `clean` | `(n_windows,)` bool — the artifact mask, carried not applied |
| `start_seconds`, `time_offset` | window times; `+ time_offset` gives original-recording time |
| `band_series(channel, band, …)` | one channel, one band, over windows |
| `roi_series(channels, band, …)` | mean over a channel cluster, within each window |
| `channel_index`, `band_index`, `clean_index`, `window_times` | lookups; raise on a missing name rather than returning a sentinel |

Features are computed for **every** window, rejected ones included. A consumer
handed only clean windows could not tell a 5% rejection rate from a 60% one,
and pre-registration §7 requires that number per configuration.

### For §7's rejection-rate confound check

```python
from eeg import rejection_rate_by_group
rates = rejection_rate_by_group(result.artifacts, configuration_id_per_window)
```

§7 correlates per-configuration rejection rate against the difficulty ranking,
because a rate that tracks difficulty is a confound — the index would then be
computed on a differently-filtered sample of each configuration. This function
counts; the correlation is the author's.

---

## Design commitments

**Artifacts are marked, never removed.** The participant has frequent
involuntary motor and ocular movements, and the session design is "natural
behaviour → artifact detection → robust processing" (CLAUDE.md, §7). Nothing
in `eeg.artifacts` modifies, interpolates or reconstructs a signal, and
nothing here justifies asking the participant to hold still. ICA (**D024**,
PROPOSED/OPEN) is deliberately *not* implemented: whether an
artifact-corrected window is better evidence than a rejected one is an open
empirical question.

**The rejection rate is reported, not just used.** Per reason and per channel,
because both failure directions are silent. Too permissive and blink and
muscle power inflate frontal theta — the index's primary term. Too strict and
the rate climbs where movement is most likely, manufacturing the confound §7
watches for. One channel failing in 90% of windows while the rest fail in 5%
is an impedance problem, and discarding the session for it would throw away a
usable recording over one badly seated electrode.

**No MNE object leaves `eeg.loading`.** Everything downstream works on plain
arrays, so nothing depends on an MNE version or a `Raw`'s internal state, and
a synthetic signal with a known spectrum is as valid an input as an EDF.

**No randomness.** Equal config plus equal input bytes gives bit-identical
band powers, asserted in `tests/test_eeg_pipeline.py` rather than assumed.
There is therefore no seed in `PipelineConfig` — a seed would imply a
stochastic step that does not exist. Callers that *do* randomise own their
seed; `scripts/eeg_demo.py` takes `--seed` for subject sampling.

**Provenance travels with the data.** Source path and SHA-256, channel names
before and after canonicalisation, what was dropped and why, the reference
applied, the unit scale factor, the montage resolved, every filter stage, and
a `warnings` list for honest degradation. `data/` is gitignored, so the hash is
the only link from a result back to the bytes that produced it.

---

## What EEGMAT can and cannot support

**It is a development fixture.** 36 subjects of 21-channel 500 Hz EDF, used
for building and unit-testing the machinery before any of the author's own
recordings exist. It contains all six channels §7's index needs (Fz, F3, F4,
Pz, P3, P4) plus Fp1/Fp2 for the ocular detector.

**Its rest/task contrast is void as a workload measure** (D032). Rest is 180 s
eyes-*closed*; task is 60 s eyes-open serial subtraction (Zyma et al. 2019).
The contrast therefore confounds workload with eye state, arousal and visual
input, and no amount of extra subjects fixes it — the defect is bias, not
variance. Nothing in this package or in `scripts/eeg_demo.py` uses it to
validate a workload effect, and theta is printed with that caveat attached.

**The one legitimate check it supports is the Berger effect.** Eyes-closed →
eyes-open suppresses alpha. It validates loading, channel selection,
filtering, windowing and PSD against real physiology while claiming nothing
about workload. Measured over the three locally available subjects, median
alpha over clean windows:

| Site group | Alpha fell in | Mean change |
|---|---|---|
| Parietal (Pz, P3, P4) | 3/3 subjects | −21.7% |
| Occipital (O1, O2) | 2/3 subjects | −17.2% |
| Frontal midline (Fz, F3, F4) | 2/3 subjects | −12.7% |

The posterior gradient is the expected one: alpha is strongest parietally and
occipitally and weakest frontally, which is why D032's frontal-only figure
(−16%) was modest. `tests/test_eeg_pipeline.py` asserts this as a regression
test, at parietal and occipital sites.

**Workload validation has to come from somewhere else** — a contrast that
holds eye state constant. The 25 combat configurations do: the participant is
eyes-open, visually engaged, and performing the same task throughout, with
only telegraph duration and attack gap varying. That is a cleaner workload
manipulation than EEGMAT offers, and D032 says to state it as such in the
report.

---

## OPEN ITEMS

**Every parameter below is a decision for Ryan.** Each has a justified default
so the pipeline runs; none of them is settled, and `D022` (hardware), `D023`
(features) and `D024` (ICA) are all OPEN in the decision log, so nothing here
*can* be settled yet. The recommendation column is a recommendation.

### Filtering

| # | Parameter | Default | Recommendation and tradeoff |
|---|---|---|---|
| E-01 | `bandpass_low` | 1.0 Hz | **Keep 1.0.** Removes drift, sweat potentials and slow movement baseline shifts that otherwise dominate peak-to-peak and make every amplitude threshold meaningless. 0.1 Hz keeps more delta but readmits drift; 2 Hz truncates delta. Note the cost either way: a 1 Hz high-pass attenuates a blink's slow component, which is why the ocular detector works on a 1–6 Hz band rather than on raw amplitude. Also note delta starts *at* 1 Hz, so its lowest bins sit on the filter's −3 dB point. |
| E-02 | `bandpass_high` | 45.0 Hz | **Keep 45.0** unless gamma enters scope. Above beta with roll-off margin, below both 50 and 60 Hz mains, and a near no-op on EEGMAT whose hardware already low-passed at 45 — deliberately, so fixture and hardware run identical code. Widening admits EMG, which overlaps 30–80 Hz and will raise rejection. |
| E-03 | `notch_freqs` | `()` (off) | **Keep off while `bandpass_high` ≤ 45.** There is no mains left to remove, and a filter stage logged in provenance but doing nothing misleads a reader of that record. The §10 "60 Hz notch set" is an acquisition-software setting, upstream of this code. If the passband is ever widened: **50 Hz for EEGMAT, 60 Hz for the author's hardware** — `EEGMAT_MAINS_HZ` and `LOCAL_MAINS_HZ` exist so the two are not confused. Measured: EEGMAT's 50 Hz line-to-beta ratio is 1.2e-4, i.e. absent. |
| E-04 | `bandpass_order`, `notch_quality` | 4, 30.0 | **Keep.** Order 4 as SOS gives ~24 dB/octave without the numerical fragility of a high-order IIR at a 0.004 cutoff-to-Nyquist ratio. Q=30 is a 2 Hz notch at 60 Hz: wide enough for mains' ±0.5 Hz wander, narrow enough to stay clear of beta. |
| E-05 | `zero_phase` | `True` | **Keep True for all offline analysis.** A causal filter delays each frequency differently, so a rejected window would be misaligned against the game event that caused it. It is non-causal and therefore impossible online — if the loop is ever closed in real time, that path needs `False` and its own validation. |
| E-06 | `edge_trim_seconds` | 1.0 | **Keep 1.0.** Removes the filter's edge transient, which the amplitude detector would otherwise rightly reject, giving every recording a dirty first window for reasons unrelated to the participant. Costs 2 s per recording. |
| E-07 | `PAD_TYPE` | `"even"` | **Keep `"even"`** (mirror), not SciPy's `"odd"` default. Measured, and the effect is large: odd extension builds an artificial `2·x[0]` ramp across the pad, which a low-cutoff filter responds to strongly. In the 1–6 Hz ocular band the first window's amplitude came out at 24.9 µV against a 4.5 µV median under `"odd"`, versus 4.75 µV under `"even"` with a long pad — and a longer pad makes `"odd"` *worse*, because the ramp grows with it. Under the default the detector flagged window 0 of every recording as ocular: a 3% rejection rate on data containing no blink. Mirror extension is also what MNE uses. This is an implementation correctness fix rather than a scientific choice, but it does change the data slightly, so it is listed. |

### Windowing

| # | Parameter | Default | Recommendation and tradeoff |
|---|---|---|---|
| E-08 | `window_seconds` | 4.0 | **Keep 4.0.** 0.25 Hz resolution, ≥16 theta and ≥32 alpha cycles per window — enough for a stable power estimate — and the same window `scripts/verify_eegmat.py` used, so D032's numbers stay comparable. It cannot resolve a change faster than 4 s, but §6 reduces each configuration to one median anyway, so finer resolution buys nothing the design uses. |
| E-09 | `hop_seconds` | `None` (non-overlapping) | **Decide deliberately.** Non-overlapping keeps windows independent, which matters because §7 takes a median over clean windows and a median over *correlated* windows has a narrower apparent spread than the data supports — understating uncertainty in exactly the quantity Gate 3's correlation is computed from. 50% overlap roughly doubles the window count and makes each configuration's median less noisy. The honest middle is non-overlapping for the pre-registered analysis, overlap for exploratory plots, with the choice recorded. |

### Artifact detection

| # | Parameter | Default | Recommendation and tradeoff |
|---|---|---|---|
| E-10 | `peak_to_peak_uv` | 150.0 | **Re-tune on the hardware validation recording, before any experimental data.** 150 µV is mid-range for published wet-electrode thresholds, and dry comb electrodes have a higher impedance and noise floor, so a threshold calibrated on EEGMAT would reject a large fraction of usable dry-electrode windows. Measured on EEGMAT: median channel p-p 53–65 µV, i.e. 2–3× headroom; worst channel (O2) 101 µV, 1.5×. Tuning it after seeing outcomes is not tuning, so this has to happen at the §10 gate. |
| E-11 | `flat_uv` | 0.5 | **Keep.** A disconnected electrode is the quietest channel in a recording and passes the ceiling; without this floor it would be the cleanest channel in every window. Real EEG at any usable impedance exceeds 0.5 µV p-p over 4 s. Fired on 0% of EEGMAT windows. |
| E-12 | `max_gradient_uv_per_sample` | 25.0 | **Keep, but note it is rate-dependent.** 25 µV/sample at 500 Hz is 12.5 mV/s, far faster than anything in a 1–45 Hz band; it catches electrode pops that leave the rest of the window quiet enough to pass the ceiling. A change of sampling rate means this number must change too. Fired on 0–20% of EEGMAT windows (0% for subject 0 rest, 20% for subject 1 task, which also had 33% amplitude rejection — a noisy recording, not a threshold problem). |
| E-13 | `ocular_channels` | `("Fp1", "Fp2")` | **Depends on E-18.** Frontopolar sites register the corneo-retinal dipole at the largest amplitude of any scalp site. It is a proxy and nothing more: with no dedicated EOG electrode it cannot separate a blink from a vertical saccade, and it will flag genuine large slow frontal activity. Claim "ocular-range frontopolar excursion", never "blink detection". |
| E-14 | `ocular_band` | (1.0, 6.0) Hz | **Keep.** A blink's energy concentrates below ~6 Hz; band-limiting lifts it above the surrounding alpha and beta so a robust amplitude statistic can find it. |
| E-15 | `ocular_z_threshold` | 5.0 | **Decide at the §10 gate, where blink rate can be observed directly.** Within-recording median/MAD normalisation is the right *shape*, because the noise floor differs between wet and dry electrodes and between sessions, so a fixed microvolt threshold would arrive miscalibrated. Its weakness is real and worth stating: if blinks are near-continuous the median is itself elevated and the detector under-flags — the statistic adapts to the contamination it is meant to find. A fixed absolute threshold has the opposite failure. On EEGMAT this reason fired on 0–8.9% of windows. |
| E-16 | `reject_if_any_channel` | `True` | **Keep True for the pre-registered analysis.** Per-channel masking would give different channels different window sets, so a frontal-minus-parietal index would combine terms computed over non-identical samples. Whole-window rejection throws away partially good data; the per-channel detail is returned either way, so the alternative stays available. |
| E-17 | ICA (**D024**) | not implemented | **Leave out unless an artifact rate forces it.** Rejection is a defensible, transparent policy with a reportable rate. ICA removes variance the author then has to justify, needs enough channels for a meaningful decomposition — questionable at 8 — and interacts badly with the "natural behaviour" design, since the components it would remove are the participant's ordinary movements. If it is ever added, it must be an ablation reported alongside rejection, not a silent replacement. |

### Hardware and channels

| # | Item | Status | Recommendation |
|---|---|---|---|
| E-18 | **Electrode placement** | **Blocking, and not yet a decision** | `docs/DECISIONS.md` **D022 is OPEN** — no device is selected — while `docs/ROADMAP.md` already names "Cyton + Ultracortex" in its next actions. Those two documents disagree and the conflict should be resolved rather than quietly inherited. More consequentially: the Ultracortex Mark IV's **default 8-channel layout does not include Fz, F3, F4, Pz, P3 or P4** — it is Fp1, Fp2, C3, C4, P7, P8, O1, O2 — and §7's index needs exactly those six. The arithmetic does work out: 6 index channels plus Fp1/Fp2 for the ocular detector is precisely 8, leaving nothing spare. Recommendation: **Fz, F3, F4, Pz, P3, P4, Fp1, Fp2**, with the relocation confirmed at the §10 gate. If Fp1/Fp2 are dropped to free channels, the ocular detector has no proxy and `require_ocular_channels` must be waived explicitly (E-13). |
| E-19 | `reference` | `"none"` | **Keep `"none"`.** Average referencing is standard with dense caps but unbiased only under roughly uniform coverage. With 8 electrodes concentrated on frontal and parietal midline, the average *contains* the frontal-midline activity the index's primary term measures, shrinking the effect toward zero — `eeg.loading` warns when the average spans fewer than 16 channels. Keeping the recorded reference also means fixture and experimental recordings go through identical arithmetic. The alternative worth considering is linked-ears, which the Cyton's ear clips make natural; it is not implemented. |
| E-20 | `channels`, `require_channels` | `()`, `()` | **Keep empty at load time.** Selection is deferred to the point of use because the artifact detector needs frontopolar channels the index does not, and dropping them at load would remove the only ocular proxy. `require_channels` is empty because §7 contains a *pre-registered degraded mode* — drop the parietal alpha term if the §10 gate fails at parietal sites — and a hard requirement here would make that contingency unrunnable. |
| E-21 | `montage` | `"standard_1020"` | **Keep.** Band power needs no geometry; positions are cheap insurance for ICA or any topographic figure, and attaching at load time means the montage used is recorded. MNE renamed this to `colin27_1020` in 1.13 and deprecates the old name for removal in 1.14; both are accepted and the resolved name is recorded. |

### Spectral features

| # | Parameter | Default | Recommendation and tradeoff |
|---|---|---|---|
| E-22 | `bands` (**D023**) | delta 1–4, theta 4–8, alpha 8–13, beta 13–30 | **Keep the conventional edges, and note they are conventional.** Individual alpha frequency varies by several Hz between people, so a fixed 8–13 Hz window may not be centred on this participant's alpha. An individually-determined alpha peak (from the §10 eyes-closed recording, via `dominant_frequency`) would fit better but adds a participant-specific parameter to a single-subject design that must be fixed before data collection. Recommendation: fix the conventional bands pre-registration; report the measured individual alpha peak as a limitation. |
| E-23 | `welch_segment_seconds`, `welch_overlap` | 2.0 s, 0.5 | **Keep.** Three averaged segments in a 4 s window at 0.5 Hz resolution — still 8 bins in theta and 10 in alpha, ample for a band *integral*. The whole window as one segment keeps 0.25 Hz resolution but averages nothing, so each band power carries the full chi-squared variance of a single periodogram. Measured: three segments give a visibly lower coefficient of variation (asserted in `tests/test_eeg_spectral.py`). |
| E-24 | absolute vs `relative_power` | both returned | **Decide which the index uses.** Absolute power scales with impedance, skin preparation and amplifier gain, none of which hold constant across sessions. Relative power is dimensionless and far more comparable — but it is *compositional*: the bands sum to 1, so alpha falling forces the others to rise even when their absolute power did not change, and a theta-minus-alpha index built on relative power cannot move its two terms independently. For a within-session standardised index (§7) absolute power is defensible, since the scale factors are constant within a session. Recommendation: **absolute**, with relative reported as a robustness check. |

### Engineering

| # | Item | Status | Recommendation |
|---|---|---|---|
| E-25 | `requirements-dev.txt` | needs `mne` and `scipy` | Neither is currently declared anywhere, though `scripts/verify_eegmat.py` already imports both. `scipy` is a hard dependency of `eeg/`; `mne` is needed only for `load_edf` and montage positions. That file is outside this branch's ownership, so it is flagged rather than edited. Note **D028** constrains the *`dqn`* package to NumPy — `eeg/` is a separate package and that constraint does not reach it — and **D033** pins the numerical environment, so adding to `requirements-lock.txt` should follow its procedure. |
| E-26 | Real-time path | not built | Offline only, by design. `zero_phase=True` reads future samples and the ocular detector's robust z needs the whole recording's window distribution, so neither can run online as written. Whether the experiment ever needs a live path is a design question; the pre-registration does not require one. |
| E-27 | Window → configuration mapping | not built | `start_seconds + time_offset` gives original-recording time, which is the join key, but nothing yet maps a time to the configuration block that was on screen. That belongs with the session recorder rather than here. §7's per-configuration rejection rates and §6's per-configuration medians both depend on it. |
