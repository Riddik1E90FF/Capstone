"""Per-window artifact detection, returning a clean/reject mask.

The design principle, from CLAUDE.md and pre-registration §7
------------------------------------------------------------
**Robustness, not suppression.**  The participant has frequent involuntary
motor and ocular movements; the session design is "natural behaviour ->
artifact detection -> robust processing".  The correct response to a
contaminated window is to mark it and move on.

So, explicitly, this module:

* **marks** windows and never modifies, interpolates or reconstructs a signal;
* returns the **reason** each window was marked, per channel, so a rejection
  rate can be audited rather than merely trusted;
* does not ask, and never justifies asking, the participant to hold still.

Nothing here removes an artifact.  Removal is a different question — ICA is
`D024`, status PROPOSED/OPEN, and is deliberately not implemented in this
package.  Whether an artifact-corrected window is better evidence than a
rejected one is an empirical question nobody here has answered yet.

Why the rejection *rate* is itself an experimental variable
-----------------------------------------------------------
Pre-registration §7 requires per-configuration window-rejection rates to be
logged and correlated against the difficulty ranking, because a rate that
tracks difficulty is a confound: if the hardest configurations are also the
ones where the participant moves most, then an index computed only on clean
windows is computed on a *different sample* of each configuration, and the
difficulty ordering could be an ordering of how much got thrown away.

That makes both failure directions dangerous, and neither raises an error:

* **Thresholds too permissive** — blink and muscle power enters the band
  powers.  Both are broadband and frontally dominant, so they inflate frontal
  theta, which is the index's *primary* term.
* **Thresholds too strict** — rejection climbs where movement is most likely,
  which is plausibly where difficulty is highest, manufacturing exactly the
  confound §7 is watching for.

The only defence is that the rate and its reasons are reported per channel and
per reason, which is what `ArtifactReport` carries.

What the ocular detector actually detects
-----------------------------------------
Neither recording system has a dedicated EOG electrode.  The detector uses
frontopolar channels (Fp1, Fp2 by default) band-limited to the ocular range as
a **proxy**: those sites sit closest to the eyes and register the corneo-retinal
dipole's deflection at the largest amplitude of any scalp site.

It cannot distinguish a blink from a vertical saccade, and it will flag genuine
large, slow frontal cortical activity.  The claim it supports is "this window
contains an ocular-range frontopolar excursion", and no more.  Writing it up as
"blink detection" would overstate it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from eeg.channels import missing, resolve
from eeg.config import ArtifactConfig, EpochConfig
from eeg.filtering import bandpass
from eeg.loading import Recording
from eeg.windowing import Windows, make_windows

#: Zero-phase pad, in seconds, for the narrow ocular bandpass.
#:
#: A 1-6 Hz Butterworth has a group delay of hundreds of milliseconds, so
#: SciPy's default 27-sample (54 ms at 500 Hz) odd-extension pad leaves a
#: start-up transient inside the first analysis window.  Measured: with the
#: default pad, window 0's ocular-band amplitude came out 3-6x every other
#: window's, and the detector flagged window 0 of every recording — a 3%
#: rejection rate on synthetic data containing no blink at all, and 1 of the 4
#: ocular rejections on EEGMAT Subject00's rest recording.  Two seconds is
#: several times the filter's settling time at its 1 Hz edge.
OCULAR_PAD_SECONDS = 2.0

#: Scale factor making the median absolute deviation a consistent estimator of
#: the standard deviation for Gaussian data (1 / Phi^-1(0.75)).  Used so the
#: ocular threshold can be read as "this many sigma" even though it is
#: computed from medians, which a few large blinks cannot drag around the way
#: they would drag a mean and an SD.
MAD_TO_SIGMA = 1.4826

#: Reasons a window can be rejected.  Fixed set: a report always carries all
#: of them, so a missing key never has to be distinguished from a zero rate.
REASONS: tuple[str, ...] = ("nonfinite", "peak_to_peak", "flat", "gradient", "ocular")


class ArtifactError(ValueError):
    """Artifact detection cannot run as configured."""


@dataclass
class ArtifactReport:
    """Per-window verdicts, with the reasoning kept.

    `clean` is the mask everything downstream uses: ``True`` means the window
    passed every check.  It is the *only* thing most callers need; the rest of
    this object exists so a rejection rate can be explained.
    """

    #: ``(n_windows,)`` bool.  True = usable.
    clean: np.ndarray
    #: ``{reason: (n_windows,) bool}`` — did this reason fire for this window.
    by_reason: dict[str, np.ndarray]
    #: ``{reason: (n_windows, n_examined_channels) bool}``.  The ocular reason
    #: is channel-resolved over `ocular_channels`, not over
    #: `examined_channels`, so its array is keyed separately in
    #: `ocular_by_channel` and its entry here is the per-window summary
    #: broadcast across examined channels for uniformity of shape.
    by_reason_channel: dict[str, np.ndarray]
    #: Channels the amplitude and gradient checks examined, in array order.
    examined_channels: tuple[str, ...]
    #: Channels the ocular detector used.  Empty if it did not run.
    ocular_channels: tuple[str, ...]
    #: ``(n_windows, n_ocular_channels)`` robust z of ocular-band amplitude.
    #: Kept because it is a continuous quantity: a borderline threshold choice
    #: can be re-examined from a saved report without re-running the pipeline.
    ocular_z: np.ndarray
    config: ArtifactConfig
    #: Non-fatal degradations: a missing ocular channel, a degenerate MAD, too
    #: few windows for a stable robust statistic.
    warnings: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def n_windows(self) -> int:
        return int(self.clean.shape[0])

    @property
    def n_clean(self) -> int:
        return int(np.count_nonzero(self.clean))

    @property
    def rejection_rate(self) -> float:
        """Fraction of windows rejected.  NaN for an empty report."""
        if self.n_windows == 0:
            return float("nan")
        return 1.0 - self.n_clean / self.n_windows

    def rate_by_reason(self) -> dict[str, float]:
        """Fraction of windows each reason fired on.

        Reasons overlap — one window can be both high-amplitude and ocular —
        so these do not sum to `rejection_rate`.  That is informative, not a
        bug: a reason that only ever fires alongside another adds nothing.
        """
        if self.n_windows == 0:
            return {r: float("nan") for r in REASONS}
        return {r: float(np.count_nonzero(self.by_reason[r]) / self.n_windows)
                for r in REASONS}

    def rate_by_channel(self) -> dict[str, float]:
        """Fraction of windows in which each examined channel failed any check.

        This is the diagnostic that distinguishes a bad electrode from a
        restless participant.  One channel failing in 90% of windows while the
        rest fail in 5% is an impedance problem, and discarding the whole
        session for it would be throwing away a recording over one comb that
        was not seated.
        """
        if self.n_windows == 0 or not self.examined_channels:
            return {}
        stacked = np.zeros((self.n_windows, len(self.examined_channels)), dtype=bool)
        for reason in ("nonfinite", "peak_to_peak", "flat", "gradient"):
            stacked |= self.by_reason_channel[reason]
        return {name: float(np.count_nonzero(stacked[:, i]) / self.n_windows)
                for i, name in enumerate(self.examined_channels)}

    def summary(self) -> dict[str, Any]:
        """JSON-safe summary for a result file or a demo table."""
        return {
            "n_windows": self.n_windows,
            "n_clean": self.n_clean,
            "rejection_rate": self.rejection_rate,
            "rate_by_reason": self.rate_by_reason(),
            "rate_by_channel": self.rate_by_channel(),
            "examined_channels": list(self.examined_channels),
            "ocular_channels": list(self.ocular_channels),
            "warnings": list(self.warnings),
        }


def ocular_reference_windows(
    recording: Recording,
    epoch: EpochConfig,
    artifact: ArtifactConfig,
) -> Windows | None:
    """Band-limit the continuous recording to the ocular band, then window it.

    This is the accurate way to produce the ocular detector's input, and
    `eeg.pipeline` always uses it.  Filtering the continuous record once and
    then cutting it means every window's ocular amplitude is computed from a
    filter that had the neighbouring seconds available.  Band-limiting each
    4-second window separately instead leaves a filter transient at both of its
    edges, and a blink that happens to sit near a window boundary is then
    measured partly from the transient rather than from the blink.

    Returns ``None`` if the recording has none of `artifact.ocular_channels`,
    leaving the caller to decide whether that is fatal
    (`ArtifactConfig.require_ocular_channels`).
    """
    present = resolve(artifact.ocular_channels, list(recording.channel_names))
    if not present:
        return None
    lo, hi = artifact.ocular_band
    limited = bandpass(recording.pick(present), recording.sfreq, lo, hi,
                       order=4, zero_phase=True, pad_seconds=OCULAR_PAD_SECONDS)
    slim = Recording(
        data=limited,
        channel_names=tuple(present),
        sfreq=recording.sfreq,
        provenance=recording.provenance,
        positions={k: v for k, v in recording.positions.items() if k in present},
    )
    return make_windows(slim, epoch, time_offset=0.0)


def robust_z(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median/MAD z-score down axis 0.  Returns ``(z, median, sigma)``.

    `sigma` is ``MAD_TO_SIGMA * MAD``.  Where the MAD is zero — a constant
    column, which happens with a dead channel or a synthetic test signal — the
    z is returned as all zeros rather than as infinities, so nothing is
    flagged on the strength of a degenerate statistic.  The caller is expected
    to notice the zero sigma and say so; `detect_artifacts` does.
    """
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    sigma = MAD_TO_SIGMA * mad
    z = np.zeros_like(values)
    usable = sigma > 0
    if np.any(usable):
        z[:, usable] = (values[:, usable] - median[usable]) / sigma[usable]
    return z, median, sigma


def detect_artifacts(
    windows: Windows,
    config: ArtifactConfig | None = None,
    *,
    ocular_windows: Windows | None = None,
) -> ArtifactReport:
    """Mark each window clean or rejected, and record why.

    Checks, all per window and per channel except where noted:

    * **nonfinite** — any NaN or infinity.  Not configurable and not
      thresholded: a non-finite sample makes every statistic computed from the
      window meaningless, and `numpy` would propagate it into a band power as
      a NaN that looks like a missing measurement rather than a broken one.
    * **peak_to_peak** — range over the window above
      `ArtifactConfig.peak_to_peak_uv`.
    * **flat** — range *below* `ArtifactConfig.flat_uv`.  A disconnected
      electrode is the quietest channel in the recording and would otherwise be
      the cleanest.
    * **gradient** — any sample-to-sample step above
      `max_gradient_uv_per_sample`.  Catches an electrode pop that leaves the
      rest of the window quiet enough to pass the range check.
    * **ocular** — robust z of ocular-band amplitude at the frontopolar
      channels above `ocular_z_threshold`.  Per window, computed over the
      ocular channels only (see `ocular_reference_windows`).

    `ocular_windows`, when given, must have the same window count as `windows`
    — it is the ocular-band version of the same segmentation.  When omitted,
    the ocular band is extracted from each window independently, which is less
    accurate near window edges, and that degradation is recorded in
    `ArtifactReport.warnings` rather than passed over.

    Raises `ArtifactError` if the examined channel set is empty, or if
    `require_ocular_channels` is set and no ocular channel is present.
    """
    config = config or ArtifactConfig()
    config.validate()
    warnings: list[str] = []

    available = list(windows.channel_names)
    if config.channels:
        examined = resolve(config.channels, available)
        absent = missing(config.channels, available)
        if absent:
            warnings.append(
                f"ArtifactConfig.channels asked for {absent}, which these windows do not "
                "have; those channels went unchecked")
    else:
        examined = available
    if not examined:
        raise ArtifactError(
            f"no channels to examine: config asked for {list(config.channels)}, windows "
            f"have {available}")

    data = windows.pick(examined)                    # (n_win, n_ch, n_samp)
    n_win, n_ch, n_samp = data.shape
    if n_samp < 2:
        raise ArtifactError(
            f"windows are {n_samp} sample(s) long; the gradient check needs at least 2")

    per_channel: dict[str, np.ndarray] = {}

    # --- nonfinite -----------------------------------------------------
    finite = np.isfinite(data)
    per_channel["nonfinite"] = ~finite.all(axis=2)

    # Statistics below are computed on a copy with non-finite samples replaced,
    # so a single NaN does not turn every other check's verdict into NaN and
    # hide which channel was actually broken.  The nonfinite flag already
    # rejects those windows.
    safe = np.where(finite, data, 0.0)

    # --- amplitude -----------------------------------------------------
    ptp = safe.max(axis=2) - safe.min(axis=2)        # (n_win, n_ch)
    per_channel["peak_to_peak"] = ptp > config.peak_to_peak_uv
    per_channel["flat"] = ptp < config.flat_uv

    # --- gradient ------------------------------------------------------
    max_step = np.abs(np.diff(safe, axis=2)).max(axis=2)
    per_channel["gradient"] = max_step > config.max_gradient_uv_per_sample

    # --- ocular --------------------------------------------------------
    ocular_present = resolve(config.ocular_channels, available)
    if not ocular_present:
        message = (
            f"none of the ocular channels {list(config.ocular_channels)} is present; "
            f"windows have {available}. The mask has no ocular component, so a blink "
            "that stays under the amplitude ceiling will pass as clean.")
        if config.require_ocular_channels:
            raise ArtifactError(
                message + " Set require_ocular_channels=False to proceed deliberately, "
                "and record the degradation.")
        warnings.append(message)
        ocular_flag = np.zeros(n_win, dtype=bool)
        ocular_z = np.zeros((n_win, 0), dtype=np.float64)
        ocular_used: tuple[str, ...] = ()
    else:
        ocular_used = tuple(ocular_present)
        if ocular_windows is not None:
            if ocular_windows.n_windows != n_win:
                raise ArtifactError(
                    f"ocular_windows has {ocular_windows.n_windows} windows but the "
                    f"signal has {n_win}; they must be the same segmentation")
            ocular_data = ocular_windows.pick(ocular_used)
        else:
            lo, hi = config.ocular_band
            raw = windows.pick(ocular_used)
            ocular_data = bandpass(
                np.where(np.isfinite(raw), raw, 0.0), windows.sfreq, lo, hi,
                order=4, zero_phase=True)
            # No OCULAR_PAD_SECONDS here: a 4 s window cannot spare a 2 s pad
            # at each end, which is itself part of why this path is the
            # inaccurate one.
            warnings.append(
                "ocular band extracted per window rather than from the continuous "
                "recording; amplitudes near window edges include a filter transient. "
                "Pass ocular_reference_windows(...) for the accurate path.")

        amplitude = np.abs(ocular_data).max(axis=2)          # (n_win, n_ocular)
        ocular_z, _, sigma = robust_z(amplitude)
        if np.any(sigma <= 0):
            dead = [ocular_used[i] for i in np.flatnonzero(sigma <= 0)]
            warnings.append(
                f"ocular amplitude has zero spread across windows at {dead}; the robust "
                "z is degenerate there and flags nothing. Expected for a constant or "
                "dead channel, and for a synthetic signal with no blinks.")
        if n_win < 20:
            warnings.append(
                f"only {n_win} windows; the median and MAD behind the ocular z are "
                "estimated from too few observations to be stable, so the ocular "
                "rejection rate from this recording should not be trusted.")
        ocular_flag = (ocular_z > config.ocular_z_threshold).any(axis=1)

    per_channel["ocular"] = np.repeat(ocular_flag[:, None], n_ch, axis=1)

    # --- combine -------------------------------------------------------
    by_reason: dict[str, np.ndarray] = {}
    for reason in REASONS:
        if reason == "ocular":
            by_reason[reason] = ocular_flag
        else:
            by_reason[reason] = per_channel[reason].any(axis=1)

    if config.reject_if_any_channel:
        rejected = np.zeros(n_win, dtype=bool)
        for reason in REASONS:
            rejected |= by_reason[reason]
    else:
        # Every channel bad, or an ocular flag, rejects the window; a window
        # where some channels are usable stays clean and the caller is expected
        # to consult `by_reason_channel`.  Kept as an option because
        # whole-window rejection is a real cost, but note the consequence
        # documented on `ArtifactConfig.reject_if_any_channel`: different
        # channels then contribute different window sets.
        channel_bad = np.zeros((n_win, n_ch), dtype=bool)
        for reason in ("nonfinite", "peak_to_peak", "flat", "gradient"):
            channel_bad |= per_channel[reason]
        rejected = channel_bad.all(axis=1) | ocular_flag

    return ArtifactReport(
        clean=~rejected,
        by_reason=by_reason,
        by_reason_channel=per_channel,
        examined_channels=tuple(examined),
        ocular_channels=ocular_used,
        ocular_z=ocular_z,
        config=config,
        warnings=tuple(warnings),
    )


def rejection_rate_by_group(
    report: ArtifactReport, group_of_window: np.ndarray
) -> dict[Any, float]:
    """Rejection rate per group label, for pre-registration §7's confound check.

    §7 requires per-configuration window-rejection rates, correlated against
    the difficulty ranking.  `group_of_window` is one label per window — a
    configuration id during the combat session, or anything else that
    partitions the windows.  This function does the counting; the correlation
    is a statistical test and so is the author's to write (CLAUDE.md, "the user
    writes the statistical tests").
    """
    labels = np.asarray(group_of_window)
    if labels.shape[0] != report.n_windows:
        raise ArtifactError(
            f"group_of_window has {labels.shape[0]} entries but the report has "
            f"{report.n_windows} windows")
    out: dict[Any, float] = {}
    for label in dict.fromkeys(labels.tolist()):
        mask = labels == label
        total = int(np.count_nonzero(mask))
        clean = int(np.count_nonzero(report.clean[mask]))
        out[label] = 1.0 - clean / total if total else float("nan")
    return out
