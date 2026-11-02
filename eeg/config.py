"""Every parameter of the EEG preprocessing pipeline, in one place.

Five frozen dataclasses — `LoadConfig`, `FilterConfig`, `EpochConfig`,
`ArtifactConfig`, `SpectralConfig` — composed by `PipelineConfig`.  The fully
resolved config is written into every pipeline result (`eeg/pipeline.py`), so a
run can be reproduced from its own output.

**None of these defaults is a decision.**  Each one is a justified starting
value with its reasoning in the field comment, and every one of them is listed
as an open item in `eeg/README.md` with its alternatives and the tradeoff.
`docs/DECISIONS.md` D023 (features) and D024 (ICA) are OPEN, and D022
(hardware) is OPEN, so nothing here *can* be settled yet.  A default exists so
the pipeline runs, not so the question is closed.  Where a value interacts with
the pre-registered design, the field comment says which section.

Two recording systems, one code path
------------------------------------
The pipeline is developed against EEGMAT (PhysioNet, 21 channels, 500 Hz,
recorded in Ukraine) and will run on the author's own hardware (D022 OPEN).
Several parameters differ between them in ways that are easy to get backwards
and silent when you do — mains frequency most of all.  `FilterConfig.for_eegmat`
and `PipelineConfig.for_eegmat` exist so the difference is an explicit named
preset rather than a number someone forgot to change.

Loading rules
-------------
* `from_dict` accepts a (possibly partial) nested dict.  Anything missing takes
  the default.  Unknown keys raise, so ``"bandpas_high"`` fails loudly instead
  of being silently ignored.
* `validate()` runs after every construction path, including the plain
  constructor via `__post_init__`.  An invalid config never reaches the data.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

# ----------------------------------------------------------------------
# Enumerated choices.  Each set is small on purpose: the field exists so the
# choice is *recorded* in every result, and so a later change is a visible
# config diff rather than an invisible code change.
# ----------------------------------------------------------------------

#: How channels are re-referenced after loading.  See `LoadConfig.reference`.
REFERENCE_SCHEMES = frozenset({"none", "average"})

#: Welch taper.  Hann is the conventional choice for EEG band power; the field
#: exists so a future change to, say, a multitaper estimate is explicit.
SPECTRAL_WINDOWS = frozenset({"hann", "hamming", "boxcar"})

#: Per-segment detrending inside Welch.  "constant" removes the segment mean,
#: which is what stops a DC offset from leaking into the lowest frequency bins.
DETREND_MODES = frozenset({"constant", "linear", "none"})

#: Sensor-position montage.  "none" skips montage attachment entirely, which is
#: legitimate for band power (it needs no geometry) and fatal for anything
#: topographic or for ICA's spatial interpretation.
#:
#: MNE renamed ``"standard_1020"`` to ``"colin27_1020"`` and deprecated the old
#: name in 1.13, for removal in 1.14.  Both are accepted, and
#: `eeg.loading._montage_positions` resolves whichever name the installed MNE
#: actually provides, recording the one it used.  The positions are identical;
#: only the name changed.  Pinning to one spelling would break this package on
#: either side of that version boundary.
MONTAGES = frozenset({"standard_1020", "colin27_1020", "none"})

#: Montage names that are the same montage under two spellings, newest first.
MONTAGE_ALIASES: dict[str, tuple[str, ...]] = {
    "standard_1020": ("standard_1020", "colin27_1020"),
    "colin27_1020": ("colin27_1020", "standard_1020"),
}

#: Mains frequency of the EEGMAT recordings (Ukraine).  Neither preset applies
#: a notch — the default 45 Hz low-pass already excludes both mains
#: frequencies — but these are named so that a configuration which *does*
#: widen the passband has the right number to hand instead of reaching for the
#: local one out of habit.  See `FilterConfig.notch_freqs`.
EEGMAT_MAINS_HZ = 50.0

#: Mains frequency where the author's own recordings are made (North America).
LOCAL_MAINS_HZ = 60.0

#: Canonical band edges, (name, low_hz, high_hz).  Half-open: a bin at exactly
#: `high` belongs to the next band, so bands partition the spectrum without
#: double-counting the shared edge.  Delta starts at 1 Hz rather than 0 because
#: below ~1 Hz is electrode drift and sweat potential, not cortical rhythm.
#: D023 (EEG features) is OPEN — these are the conventional edges, not a
#: decision.
DEFAULT_BANDS: tuple[tuple[str, float, float], ...] = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
)


class ConfigError(ValueError):
    """An invalid configuration.  Raised before any data is touched."""


def _need(errors: list[str], cond: bool, msg: str) -> None:
    if not cond:
        errors.append(msg)


# ======================================================================
# Loading
# ======================================================================


@dataclass(frozen=True)
class LoadConfig:
    """Channel selection, referencing, montage and units.

    What fails if you get this wrong: units and referencing are both silent.
    A pipeline that believes volts are microvolts rejects nothing, because
    every amplitude threshold is then 1e6 times too permissive; one that
    believes the reverse rejects everything.  An average reference computed
    over a channel set that includes an ECG trace subtracts a cardiac signal
    from every scalp channel.
    """

    # Scalp sites to keep, as canonical names or ROI names from
    # `eeg.channels.ROIS`.  Empty tuple means "every scalp site the file has".
    #
    # Default: everything.  Selection is deferred to the point of use
    # (spectral extraction, artifact detection) because the artifact detector
    # needs frontopolar channels that the index does not, and discarding them
    # at load time would remove the only blink proxy available.  A caller who
    # wants a narrow load for speed can set this.
    channels: tuple[str, ...] = ()

    # Drop channels that are not cortical measurement sites: EEGMAT's
    # "EEG A2-A1" bipolar derivation and "ECG ECG" cardiac trace, and anything
    # else `eeg.channels.classify` does not recognise as a scalp site.
    #
    # Default True.  The dropped names are recorded in provenance, so a
    # genuine scalp channel lost to an unrecognised spelling is visible in the
    # output rather than discovered months later.
    drop_non_scalp: bool = True

    # Re-referencing.
    #   "none"    - keep the file's own reference.
    #   "average" - common average over the retained scalp channels.
    #
    # Default "none".  Average referencing is standard practice with dense
    # caps, but it is only unbiased when the electrodes sample the head
    # roughly uniformly.  Two things argue against it here: EEGMAT is already
    # referenced (it ships an A2-A1 derivation alongside 19 referenced scalp
    # channels), and the author's own hardware is expected to carry 8 channels
    # concentrated over frontal and parietal midline (D022 OPEN).  An average
    # over 8 non-uniform electrodes subtracts a term that contains the very
    # frontal-midline activity the index's primary term measures, shrinking
    # the effect toward zero.  Keeping the recorded reference means the
    # development fixture and the experimental recordings go through identical
    # arithmetic.  Listed as an open item: this is a scientific choice, not a
    # technical one.
    reference: str = "none"

    # MNE returns EDF data in volts.  Rescale to microvolts, the unit every
    # amplitude threshold in `ArtifactConfig` is written in and the unit EEG
    # literature reports.  Recorded in provenance so the factor is auditable.
    scale_to_microvolts: bool = True

    # Attach sensor positions.  Band power needs none, so this is cheap
    # insurance for later: ICA (D024, PROPOSED/OPEN) and any topographic
    # figure in the report need positions, and attaching them at load time
    # means the montage used is recorded with the run.
    montage: str = "standard_1020"

    # Sites that must be present, or loading raises.  Empty = require nothing.
    #
    # Default empty, and this is the cautious choice rather than the lax one:
    # the pipeline is not the place to enforce the index's channel
    # requirements, because pre-registration §7 includes a *degraded mode*
    # that drops the parietal alpha term if the hardware gate fails at
    # parietal sites.  A hard requirement here would make the pre-registered
    # contingency unrunnable.  The caller states what it needs.
    require_channels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "LoadConfig":
        errors: list[str] = []
        _need(errors, self.reference in REFERENCE_SCHEMES,
              f"reference must be one of {sorted(REFERENCE_SCHEMES)}, got {self.reference!r}")
        _need(errors, self.montage in MONTAGES,
              f"montage must be one of {sorted(MONTAGES)}, got {self.montage!r}")
        if self.reference == "average" and not self.drop_non_scalp:
            errors.append(
                "reference='average' with drop_non_scalp=False would average an ECG or "
                "bipolar-derivation channel into every scalp channel")
        if errors:
            raise ConfigError("invalid LoadConfig:\n  " + "\n  ".join(errors))
        return self


# ======================================================================
# Filtering
# ======================================================================


@dataclass(frozen=True)
class FilterConfig:
    """Bandpass and mains notch.

    What fails if you get this wrong: a notch at the wrong mains frequency
    leaves the line artifact in place while reporting that it was removed, and
    a high-pass that is too aggressive eats the delta band it is supposed to
    protect.  Neither raises.
    """

    # Bandpass edges in Hz.  `None` on either end means that end is not
    # filtered (high-pass only, or low-pass only).
    #
    # 1.0 Hz high-pass: removes electrode drift, sweat potentials and the slow
    # baseline shifts that body movement produces, which otherwise dominate
    # peak-to-peak amplitude and make the artifact thresholds meaningless.
    # 1 Hz rather than 0.1 Hz because the band set's lowest band starts at
    # 1 Hz, so nothing reported is attenuated; rather than 2 Hz because that
    # would truncate delta.  Note the cost: a 1 Hz high-pass also attenuates
    # the slow component of a blink, which makes blinks *less* visible to the
    # amplitude detector — which is exactly why the ocular detector in
    # `ArtifactConfig` runs on a 1-6 Hz band rather than relying on raw p-p.
    bandpass_low: float | None = 1.0

    # 45.0 Hz low-pass: above the beta band (ends at 30 Hz) with margin for
    # the filter's roll-off, and below both 50 and 60 Hz mains.  This is a
    # no-op on EEGMAT, whose EDF header already reports a 45 Hz low-pass, and
    # that is deliberate: the development fixture and the hardware path then
    # run identical code.  Raising it above ~48 Hz means the notch, not the
    # bandpass, is all that stands between the data and mains hum.
    bandpass_high: float | None = 45.0

    # Butterworth order, applied as second-order sections.  4 gives ~24 dB per
    # octave, enough to separate the bands without the numerical fragility of
    # a very high-order IIR at these cutoff-to-Nyquist ratios.
    bandpass_order: int = 4

    # Mains frequencies to notch, in Hz.  Empty tuple disables the notch.
    #
    # Default: **empty**, and that needs explaining, because a 60 Hz notch is
    # the reflex in a North American lab and pre-registration §10's hardware
    # validation gate says "60 Hz notch set".
    #
    # With `bandpass_high = 45.0` there is nothing for a notch to do.  A 4th
    # order Butterworth low-pass at 45 Hz attenuates 60 Hz by roughly 20 dB
    # and 120 Hz by about 45 dB before the notch is ever reached, so a notch
    # stage here would be recorded in provenance as having filtered mains
    # while removing almost nothing that the low-pass had not already removed.
    # A filter stage that is logged but inert is worse than no stage: it makes
    # a reader believe mains was handled by this code when it was handled by
    # the band edge.  The §10 notch is also a different thing — it is set in
    # the acquisition software, upstream of this package, and that is where
    # the gate checks it.
    #
    # The notch is implemented, tested against a synthetic mains sine, and
    # configurable, for the case that actually needs it: a configuration whose
    # passband reaches mains, e.g. one widened to look at gamma or to inspect
    # raw line contamination. Whether this project analyses above 45 Hz at all
    # is an open item.
    #
    # If you do enable it: **EEGMAT is 50 Hz** (recorded in Ukraine), the
    # author's hardware is 60 Hz. Getting that backwards is the classic
    # cross-dataset error — a 60 Hz notch on 50 Hz-contaminated data removes
    # nothing and reports success. `FilterConfig.for_eegmat()` carries the
    # right value, and `PipelineConfig.validate` refuses a notch that sits
    # outside the passband rather than letting it be silently inert.
    notch_freqs: tuple[float, ...] = ()

    # Notch quality factor.  The -3 dB bandwidth is f0/Q, so Q=30 at 60 Hz is
    # a 2 Hz-wide notch: wide enough to track the ±0.5 Hz wander of real
    # mains, narrow enough not to reach down into beta (ends 30 Hz).  Lowering
    # Q to 10 would make a 6 Hz-wide notch at 60 Hz — still clear of beta, but
    # at 50 Hz a 5 Hz notch starts to matter if the bandpass is widened.
    notch_quality: float = 30.0

    # Zero-phase filtering (`scipy.signal.sosfiltfilt`, forward and backward).
    #
    # True for offline analysis: a causal IIR filter delays each frequency by
    # a different amount, which smears an artifact's onset in time and so
    # misaligns a rejected window against the game event that caused it.
    # Zero-phase filtering has no such delay, at the cost of being
    # non-causal — it reads future samples, so it is impossible in a real-time
    # path.  If this project ever closes the loop online, that path needs
    # `zero_phase=False` and its own validation; the offline analysis does not.
    zero_phase: bool = True

    # Seconds of signal at each end to discard after filtering, to remove the
    # filter's edge transient.  0.0 keeps everything.
    #
    # Default 1.0 s.  An IIR filter's response at the very start of a record
    # is dominated by its initial conditions, not by the data; `sosfiltfilt`
    # pads to reduce this but does not eliminate it, and a transient at the
    # record edge is a large excursion that the amplitude detector would
    # rightly reject — producing a run whose first windows are always dirty
    # for reasons that have nothing to do with the participant.  1 s at 500 Hz
    # is 500 samples, comfortably longer than a 4th-order Butterworth's
    # settling time at a 1 Hz cutoff.
    edge_trim_seconds: float = 1.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "FilterConfig":
        errors: list[str] = []
        if self.bandpass_low is not None:
            _need(errors, self.bandpass_low > 0, "bandpass_low must be > 0 Hz")
        if self.bandpass_high is not None:
            _need(errors, self.bandpass_high > 0, "bandpass_high must be > 0 Hz")
        if self.bandpass_low is not None and self.bandpass_high is not None:
            _need(errors, self.bandpass_low < self.bandpass_high,
                  "bandpass_low must be < bandpass_high")
        _need(errors, self.bandpass_order >= 1, "bandpass_order must be >= 1")
        _need(errors, all(f > 0 for f in self.notch_freqs),
              "every notch frequency must be > 0 Hz")
        _need(errors, self.notch_quality > 0, "notch_quality must be > 0")
        _need(errors, self.edge_trim_seconds >= 0, "edge_trim_seconds must be >= 0")
        if errors:
            raise ConfigError("invalid FilterConfig:\n  " + "\n  ".join(errors))
        return self

    @classmethod
    def for_eegmat(cls, **overrides: Any) -> "FilterConfig":
        """Preset for the EEGMAT development fixture.

        Identical to the default, deliberately: EEGMAT's EDF header reports a
        0.5-45 Hz hardware band, so this pipeline's 1-45 Hz bandpass does
        almost nothing to it, and no notch is needed because 50 Hz is already
        outside the passband.  That is the point of the fixture — it exercises
        the same code the real recordings will run, without the filters doing
        work the acquisition hardware has already done.

        The preset exists as a named thing anyway, so that if the EEGMAT path
        ever *does* need to differ (a widened passband, say, which would then
        need a 50 Hz notch rather than a 60 Hz one) the difference has one
        place to live instead of being a literal in a script.
        """
        return replace(cls(), **overrides)

    @classmethod
    def with_mains_notch(cls, mains_hz: float, *, bandpass_high: float = 80.0,
                         **overrides: Any) -> "FilterConfig":
        """A config whose passband reaches mains, with the notch enabled.

        For inspecting line contamination or anything above 45 Hz.  The
        passband is widened to 80 Hz by default because a notch inside the
        passband is the only kind that does anything, and
        `PipelineConfig.validate` enforces that.  Note the consequence: a
        passband this wide also admits EMG, which overlaps the whole
        30-80 Hz range, so amplitude thresholds tuned at 45 Hz will reject
        more.
        """
        return replace(cls(), notch_freqs=(float(mains_hz),),
                       bandpass_high=bandpass_high, **overrides)


# ======================================================================
# Epoching
# ======================================================================


@dataclass(frozen=True)
class EpochConfig:
    """Segmentation of a continuous recording into fixed-length windows."""

    # Window length in seconds.
    #
    # 4.0 s: gives 0.25 Hz frequency resolution over the whole window, and
    # contains at least 16 cycles of the slowest theta component (4 Hz) and 32
    # of the slowest alpha (8 Hz) — enough cycles for a stable power estimate.
    # It also matches the window `scripts/verify_eegmat.py` used, so D032's
    # recorded numbers stay comparable to anything this pipeline produces.
    #
    # The cost is temporal: a 4 s window cannot resolve a workload change
    # faster than 4 s, and in the combat environment a configuration block is
    # the unit anyway (pre-registration §6 takes one median per
    # configuration), so finer resolution would buy nothing the design uses.
    window_seconds: float = 4.0

    # Step between consecutive window starts, in seconds.  `None` means "equal
    # to window_seconds", i.e. non-overlapping, back-to-back windows.
    #
    # Default non-overlapping.  Pre-registration §7 takes the *median index
    # over clean windows* per configuration; overlapping windows share
    # samples, so they are not independent observations, and a median over
    # correlated windows has a narrower apparent spread than the data
    # supports.  That would understate uncertainty in exactly the quantity the
    # Gate 3 correlation is computed from.  The tradeoff is real: 50% overlap
    # would roughly double the window count and make each configuration's
    # median less noisy.  Listed as an open item.
    hop_seconds: float | None = None

    # Discard a trailing partial window rather than zero-padding it.
    #
    # True.  A zero-padded window has artificially low power in every band and
    # an artificially large step at the pad boundary — it would pass the
    # amplitude check and contribute a spuriously low band power to the
    # median.  Dropping it loses at most `window_seconds` of data per block.
    drop_partial: bool = True

    def __post_init__(self) -> None:
        self.validate()

    @property
    def effective_hop_seconds(self) -> float:
        """`hop_seconds` if set, else `window_seconds` (non-overlapping)."""
        return self.window_seconds if self.hop_seconds is None else self.hop_seconds

    def validate(self) -> "EpochConfig":
        errors: list[str] = []
        _need(errors, self.window_seconds > 0, "window_seconds must be > 0")
        if self.hop_seconds is not None:
            _need(errors, self.hop_seconds > 0, "hop_seconds must be > 0")
            _need(errors, self.hop_seconds <= self.window_seconds,
                  "hop_seconds must be <= window_seconds (a larger hop would silently "
                  "discard samples between windows)")
        if errors:
            raise ConfigError("invalid EpochConfig:\n  " + "\n  ".join(errors))
        return self


# ======================================================================
# Artifact detection
# ======================================================================


@dataclass(frozen=True)
class ArtifactConfig:
    """Thresholds for the per-window clean/reject mask.

    The design principle is **robustness, not suppression** (CLAUDE.md,
    pre-registration §7): the participant has frequent involuntary motor and
    ocular movements, the session is "natural behaviour -> artifact detection
    -> robust processing", and the correct response to a contaminated window is
    to mark it and move on.  Nothing here modifies, interpolates or
    reconstructs a signal, and nothing here asks the participant to hold still.

    What fails if you get this wrong, in both directions:

    * **Too permissive** and artifact power enters the band powers.  Blinks and
      jaw clenches are broadband and frontally dominant, so they inflate the
      frontal theta term — the index's *primary* term — and a difficulty
      ordering could then be an ordering of how much the participant moved.
    * **Too strict** and the rejection rate climbs toward 100% in the hardest
      configurations, where movement is most likely.  That is not a safe
      failure: §7 correlates per-configuration rejection rate against the
      difficulty ranking precisely because a rate that tracks difficulty is a
      confound, and an over-strict threshold manufactures one.

    Both failure modes are detectable only if the rejection rate and its
    reasons are reported, which is why `eeg.artifacts` returns per-reason and
    per-channel detail rather than a bare boolean.
    """

    # Channels the detector examines, as canonical site names or ROI names.
    # Empty = every scalp channel in the recording.
    #
    # Default empty.  A window contaminated at an occipital electrode is still
    # a contaminated recording, and limiting the check to the index's own
    # channels would let a whole-head movement artifact through whenever it
    # happened to spare Fz.
    channels: tuple[str, ...] = ()

    # --- Amplitude ---

    # Peak-to-peak ceiling in microvolts, per channel, per window.
    #
    # 150 uV.  Clean awake scalp EEG rarely exceeds ~100 uV p-p over 4 s;
    # published rejection thresholds cluster in the 100-200 uV range.  150 is
    # the middle of that range and is **expected to need re-tuning on the
    # author's own recordings**: dry comb electrodes have a higher impedance
    # and noise floor than the wet electrodes EEGMAT used, so a threshold
    # calibrated on EEGMAT and applied unchanged to dry-electrode data would
    # reject a large fraction of perfectly usable windows.  Re-tuning must
    # happen on the hardware validation recording, before any experimental
    # data is collected, so that the threshold is not chosen after seeing
    # outcomes.
    peak_to_peak_uv: float = 150.0

    # Flat-signal floor in microvolts, per channel, per window.  A window whose
    # p-p is *below* this has no physiological signal in it: the electrode is
    # disconnected, the amplifier has railed, or the channel is a dead input.
    # Such a window passes the ceiling check, so without this floor the
    # quietest channel in the recording would also be the cleanest.
    #
    # 0.5 uV over 4 s.  Real EEG at any usable impedance exceeds this.
    flat_uv: float = 0.5

    # Maximum absolute sample-to-sample change, in microvolts per sample.
    #
    # 25 uV/sample.  At 500 Hz that is 12.5 mV/s, orders of magnitude faster
    # than any cortical rhythm inside a 1-45 Hz band, so it fires only on
    # electrode pops, cable movement and amplifier saturation steps.  It
    # catches a brief step that leaves the rest of the window quiet enough to
    # pass the p-p ceiling.  Scaling note: this threshold is per *sample*, so
    # it depends on the sampling rate; the sampling rate is recorded with
    # every result, and a change of rate means this number must change too.
    max_gradient_uv_per_sample: float = 25.0

    # --- Ocular / blink proxy ---

    # Channels used as the blink/ocular proxy.  Frontopolar by default: Fp1
    # and Fp2 sit closest to the eyes and register the corneo-retinal dipole's
    # deflection at the largest amplitude of any scalp site.
    #
    # This is a **proxy, and a weak one**.  Neither recording system has a
    # dedicated EOG electrode, so the detector cannot separate a blink from a
    # vertical saccade or from genuine large, slow frontal cortical activity.
    # It detects *ocular-range frontopolar excursions*, and that is all that
    # should be claimed for it.
    ocular_channels: tuple[str, ...] = ("Fp1", "Fp2")

    # Band the ocular detector works in, in Hz.  A blink is a slow, large
    # monophasic deflection whose energy concentrates below ~6 Hz; band-
    # limiting to 1-6 Hz raises it above the surrounding alpha and beta so a
    # robust amplitude statistic can find it.
    ocular_band: tuple[float, float] = (1.0, 6.0)

    # Robust z-score threshold on the per-window maximum absolute amplitude of
    # the band-limited ocular channels.  The z uses the median and the MAD
    # (scaled by 1.4826 to match a Gaussian sigma) computed across all windows
    # **within the recording**, not a fixed microvolt level.
    #
    # 5.0.  Within-recording normalisation is the right shape here because the
    # noise floor differs between wet and dry electrodes and between sessions,
    # so a fixed microvolt threshold would be miscalibrated on arrival.  The
    # cost is a genuine weakness worth stating: if blinks are near-continuous,
    # the median itself is elevated and the detector under-flags — the
    # statistic adapts to the contamination it is meant to find.  A fixed
    # absolute threshold has the opposite failure.  Listed as an open item;
    # which is correct is an empirical question for the hardware validation
    # recording, where blink rate can be observed directly.
    ocular_z_threshold: float = 5.0

    # If the recording has none of `ocular_channels`, raise instead of silently
    # running without an ocular detector.
    #
    # True.  The author's expected 8-channel layout may not include Fp1/Fp2 at
    # all (D022 OPEN), and a run that quietly produced a mask with no ocular
    # component would look identical in the output to one that had it.  Set
    # False, deliberately and with the degradation recorded, to run anyway.
    require_ocular_channels: bool = True

    # --- Combination ---

    # Reject a window if *any* examined channel fails any check.
    #
    # True.  The alternative — per-channel masking, where a window survives for
    # the channels that were clean — gives different channels different window
    # sets, so a frontal-minus-parietal index would combine terms computed over
    # non-identical samples.  Whole-window rejection keeps the terms
    # commensurable at the cost of throwing away partially good data.  The
    # per-channel detail is returned either way, so the alternative stays
    # available.  Listed as an open item.
    reject_if_any_channel: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "ArtifactConfig":
        errors: list[str] = []
        _need(errors, self.peak_to_peak_uv > 0, "peak_to_peak_uv must be > 0")
        _need(errors, self.flat_uv >= 0, "flat_uv must be >= 0")
        _need(errors, self.flat_uv < self.peak_to_peak_uv,
              "flat_uv must be < peak_to_peak_uv (otherwise every window fails both)")
        _need(errors, self.max_gradient_uv_per_sample > 0,
              "max_gradient_uv_per_sample must be > 0")
        lo, hi = self.ocular_band
        _need(errors, 0 < lo < hi, "ocular_band must satisfy 0 < low < high")
        _need(errors, self.ocular_z_threshold > 0, "ocular_z_threshold must be > 0")
        if errors:
            raise ConfigError("invalid ArtifactConfig:\n  " + "\n  ".join(errors))
        return self


# ======================================================================
# Spectral estimation
# ======================================================================


@dataclass(frozen=True)
class SpectralConfig:
    """Welch PSD and band-power integration."""

    # Band edges, (name, low_hz, high_hz), half-open on the high edge.
    bands: tuple[tuple[str, float, float], ...] = DEFAULT_BANDS

    # Welch segment length in seconds, within each analysis window.  `None`
    # uses the whole window as one segment.
    #
    # 2.0 s inside a 4.0 s window at 50% overlap gives 3 averaged segments.
    # The tradeoff is explicit: one segment per window (None) preserves the
    # window's full 0.25 Hz resolution but averages nothing, so each band
    # power carries the full chi-squared variance of a single periodogram;
    # three segments cut that variance by roughly a third at the cost of
    # halving resolution to 0.5 Hz.  0.5 Hz still puts 8 bins in theta and 10
    # in alpha, which is ample for integrating a band, and the index consumes
    # band integrals rather than spectral peaks.  Listed as an open item.
    welch_segment_seconds: float | None = 2.0

    # Fractional overlap between Welch segments, in [0, 1).
    # 0.5 is the standard choice for a Hann taper: it recovers most of the
    # information the taper attenuates at the segment edges without making
    # neighbouring segments so correlated that averaging them stops helping.
    welch_overlap: float = 0.5

    # Taper and per-segment detrending.  "constant" removes each segment's
    # mean, which is what keeps a residual DC offset out of the lowest bins
    # and so out of the delta band.
    window: str = "hann"
    detrend: str = "constant"

    # Also report each band as a fraction of total power across all `bands`.
    #
    # True.  Absolute band power scales with electrode impedance, skin
    # preparation and amplifier gain, none of which are constant between
    # sessions or between the two recording systems; relative power is
    # dimensionless and far more comparable.  Both are returned because they
    # answer different questions and the index's own standardisation
    # (pre-registration §7, and not implemented here) may want either.
    relative_power: bool = True

    def __post_init__(self) -> None:
        self.validate()

    @property
    def band_names(self) -> tuple[str, ...]:
        return tuple(name for name, _, _ in self.bands)

    def validate(self) -> "SpectralConfig":
        errors: list[str] = []
        _need(errors, len(self.bands) >= 1, "bands must contain at least one band")
        seen: set[str] = set()
        for name, lo, hi in self.bands:
            _need(errors, bool(name), "every band must have a non-empty name")
            _need(errors, name not in seen, f"duplicate band name {name!r}")
            seen.add(name)
            _need(errors, 0 <= lo < hi, f"band {name!r} must satisfy 0 <= low < high")
        if self.welch_segment_seconds is not None:
            _need(errors, self.welch_segment_seconds > 0,
                  "welch_segment_seconds must be > 0 or None")
        _need(errors, 0.0 <= self.welch_overlap < 1.0, "welch_overlap must be in [0, 1)")
        _need(errors, self.window in SPECTRAL_WINDOWS,
              f"window must be one of {sorted(SPECTRAL_WINDOWS)}, got {self.window!r}")
        _need(errors, self.detrend in DETREND_MODES,
              f"detrend must be one of {sorted(DETREND_MODES)}, got {self.detrend!r}")
        if errors:
            raise ConfigError("invalid SpectralConfig:\n  " + "\n  ".join(errors))
        return self


# ======================================================================
# Composition
# ======================================================================

#: Bumped on any change to `PipelineConfig`'s structure that makes an older
#: serialised config un-loadable.  Stored in every result, so a saved run can
#: be told apart from one produced by incompatible code.
CONFIG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PipelineConfig:
    """Every parameter of one pipeline run.

    Serialised into every `PipelineResult`.  Two runs with equal configs and
    equal input bytes produce equal output: the pipeline contains no
    randomness, which is a stronger reproducibility guarantee than a seed
    would be, and `tests/test_eeg_pipeline.py` asserts it rather than assuming
    it.  `PipelineConfig.seed` therefore does not exist; callers that *do*
    randomise — `scripts/eeg_demo.py` sampling which subjects to run — own
    their own seed and record it alongside.
    """

    load: LoadConfig = LoadConfig()
    filter: FilterConfig = FilterConfig()
    epoch: EpochConfig = EpochConfig()
    artifact: ArtifactConfig = ArtifactConfig()
    spectral: SpectralConfig = SpectralConfig()

    #: Free-text note recorded with the result, e.g. which dataset or session
    #: this config was written for.  Never read by the code.
    note: str = ""

    schema_version: int = CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "PipelineConfig":
        """Cross-section checks.  Each section has already validated itself."""
        errors: list[str] = []

        # A notch inside the passband is pointless but harmless; a notch
        # *outside* it is a silent no-op that looks like mains suppression.
        if self.filter.bandpass_high is not None:
            for f in self.filter.notch_freqs:
                if f > self.filter.bandpass_high:
                    errors.append(
                        f"notch at {f} Hz is above bandpass_high={self.filter.bandpass_high} Hz, "
                        "so it removes nothing the low-pass had not already removed; either "
                        "widen the bandpass or drop the notch, but do not record a notch that "
                        "did nothing")

        # Welch segments must fit inside an analysis window.
        seg = self.spectral.welch_segment_seconds
        if seg is not None and seg > self.epoch.window_seconds:
            errors.append(
                f"welch_segment_seconds={seg} exceeds window_seconds="
                f"{self.epoch.window_seconds}")

        # Every reported band must lie inside the passband, or its power is an
        # artifact of the filter's roll-off rather than a property of the EEG.
        lo_edge = self.filter.bandpass_low
        hi_edge = self.filter.bandpass_high
        for name, lo, hi in self.spectral.bands:
            if lo_edge is not None and lo < lo_edge:
                errors.append(
                    f"band {name!r} starts at {lo} Hz, below bandpass_low={lo_edge} Hz: "
                    "its power would be shaped by the high-pass roll-off")
            if hi_edge is not None and hi > hi_edge:
                errors.append(
                    f"band {name!r} ends at {hi} Hz, above bandpass_high={hi_edge} Hz: "
                    "its power would be shaped by the low-pass roll-off")

        # The ocular detector band must also survive the bandpass.
        olo, ohi = self.artifact.ocular_band
        if lo_edge is not None and olo < lo_edge:
            errors.append(
                f"ocular_band starts at {olo} Hz, below bandpass_low={lo_edge} Hz")
        if hi_edge is not None and ohi > hi_edge:
            errors.append(
                f"ocular_band ends at {ohi} Hz, above bandpass_high={hi_edge} Hz")

        _need(errors, self.schema_version == CONFIG_SCHEMA_VERSION,
              f"schema_version must be {CONFIG_SCHEMA_VERSION}, got {self.schema_version}")

        if errors:
            raise ConfigError("invalid PipelineConfig:\n  " + "\n  ".join(errors))
        return self

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    @classmethod
    def for_eegmat(cls, **overrides: Any) -> "PipelineConfig":
        """Preset for the EEGMAT development fixture (D032).

        Identical to the default except for `note`, which records which
        dataset the result came from.  That is deliberate and is the fixture's
        whole value: it exercises the code the real recordings will run, not a
        variant of it.  See `FilterConfig.for_eegmat` for why no 50 Hz notch
        is needed here.
        """
        base = dict(
            load=LoadConfig(),
            filter=FilterConfig.for_eegmat(),
            epoch=EpochConfig(),
            artifact=ArtifactConfig(),
            spectral=SpectralConfig(),
            note="EEGMAT (PhysioNet eegmat v1.0.0) development fixture; 50 Hz mains",
        )
        base.update(overrides)
        return cls(**base)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON form.  Tuples become lists; nothing else changes."""
        return _jsonify(asdict(self))

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        """Build from a possibly partial nested dict.  Unknown keys raise."""
        sections = {
            "load": LoadConfig,
            "filter": FilterConfig,
            "epoch": EpochConfig,
            "artifact": ArtifactConfig,
            "spectral": SpectralConfig,
        }
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigError(f"unknown PipelineConfig key(s): {unknown}")

        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key in sections:
                kwargs[key] = _build_section(sections[key], value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def with_overrides(self, overrides: dict[str, Any]) -> "PipelineConfig":
        """Return a copy with dotted-key fields replaced.

        ``{"filter.bandpass_high": 40.0, "epoch.window_seconds": 2.0}``.
        Values may already be typed or may be strings from a command line;
        strings are coerced to the field's declared type.  Unknown keys raise.
        """
        sections = {"load", "filter", "epoch", "artifact", "spectral"}
        per_section: dict[str, dict[str, Any]] = {s: {} for s in sections}
        top: dict[str, Any] = {}

        for key, value in overrides.items():
            if "." in key:
                section, field_name = key.split(".", 1)
                if section not in sections:
                    raise ConfigError(
                        f"unknown config section {section!r} in override {key!r}; "
                        f"expected one of {sorted(sections)}")
                per_section[section][field_name] = value
            else:
                top[key] = value

        kwargs: dict[str, Any] = {}
        for section, updates in per_section.items():
            current = getattr(self, section)
            if updates:
                known = {f.name: f.type for f in fields(current)}
                bad = sorted(set(updates) - set(known))
                if bad:
                    raise ConfigError(f"unknown {section} field(s): {bad}")
                coerced = {k: _coerce(v, known[k], f"{section}.{k}")
                           for k, v in updates.items()}
                kwargs[section] = replace(current, **coerced)

        known_top = {f.name for f in fields(self)} - sections
        bad_top = sorted(set(top) - known_top)
        if bad_top:
            raise ConfigError(f"unknown PipelineConfig field(s): {bad_top}")
        kwargs.update(top)
        return replace(self, **kwargs)


def _build_section(cls: type, value: Any) -> Any:
    """Construct one config section from a dict, rejecting unknown keys."""
    if isinstance(value, cls):
        return value
    if not isinstance(value, dict):
        raise ConfigError(f"{cls.__name__} must be given as an object, got {type(value).__name__}")
    known = {f.name: f.type for f in fields(cls)}
    unknown = sorted(set(value) - set(known))
    if unknown:
        raise ConfigError(f"unknown {cls.__name__} key(s): {unknown}")
    coerced = {k: _coerce(v, known[k], f"{cls.__name__}.{k}") for k, v in value.items()}
    return cls(**coerced)


def _jsonify(obj: Any) -> Any:
    """Recursively turn tuples into lists so `json` can write them."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


# Field types arrive as strings because every module here uses
# `from __future__ import annotations`.  The pipeline's configs use only a
# handful of shapes, so matching on the annotation text is enough and avoids
# a `typing.get_type_hints` call per field.
def _coerce(value: Any, annotation: Any, where: str) -> Any:
    """Coerce a JSON or command-line value to a config field's declared type."""
    text = annotation if isinstance(annotation, str) else getattr(
        annotation, "__name__", str(annotation))
    try:
        if text == "bool":
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in ("1", "true", "yes", "on"):
                    return True
                if lowered in ("0", "false", "no", "off"):
                    return False
                raise ValueError("not a boolean")
            return bool(value)
        if text == "int":
            if isinstance(value, bool):
                raise ValueError("bool is not an int here")
            as_float = float(str(value).replace("_", "")) if isinstance(value, str) else float(value)
            if not as_float.is_integer():
                raise ValueError(f"{value!r} is not a whole number")
            return int(as_float)
        if text == "float":
            if isinstance(value, bool):
                raise ValueError("bool is not a float here")
            return float(value)
        if text in ("float | None", "Optional[float]"):
            if value is None or (isinstance(value, str) and value.strip().lower() == "none"):
                return None
            return float(value)
        if text == "str":
            if not isinstance(value, str):
                raise ValueError(f"expected a string, got {type(value).__name__}")
            return value
        if text == "tuple[str, ...]":
            if isinstance(value, str):
                return tuple(v for v in (p.strip() for p in value.split(",")) if v)
            return tuple(str(v) for v in value)
        if text == "tuple[float, ...]":
            if isinstance(value, str):
                return tuple(float(v) for v in (p.strip() for p in value.split(",")) if v)
            return tuple(float(v) for v in value)
        if text == "tuple[float, float]":
            if isinstance(value, str):
                parts = [p.strip() for p in value.split(",") if p.strip()]
            else:
                parts = list(value)
            if len(parts) != 2:
                raise ValueError(f"expected 2 values, got {len(parts)}")
            return (float(parts[0]), float(parts[1]))
        if text == "tuple[tuple[str, float, float], ...]":
            out = []
            for entry in value:
                if isinstance(entry, str):
                    entry = [p.strip() for p in entry.split(",")]
                name, lo, hi = entry
                out.append((str(name), float(lo), float(hi)))
            return tuple(out)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"bad value for {where}: {value!r} ({exc})") from None
    raise ConfigError(f"no coercion rule for {where} of type {text!r}")


def parse_override(text: str) -> tuple[str, str]:
    """Split a ``section.key=value`` command-line override."""
    if "=" not in text:
        raise ConfigError(f"override must look like section.key=value, got {text!r}")
    key, value = text.split("=", 1)
    return key.strip(), value.strip()
