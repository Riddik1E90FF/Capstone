"""Welch power spectral density and band power, per window per channel.

This is the last stage of the pipeline.  Its output, `WindowFeatures`, is the
interface the workload index consumes — see `eeg/README.md`, "Where this
package stops".

Units
-----
Input is microvolts.  The PSD is a density, uV^2/Hz.  Band power is the PSD
**integrated** over the band, so uV^2 — a power, not a density.  Integrating
rather than averaging matters because the bands have unequal widths (theta is
4 Hz wide, beta is 17 Hz), so a mean PSD height is not comparable across bands
while an integral is the actual power in each.  `scripts/verify_eegmat.py`
integrates with `np.trapezoid` for the same reason, and this module matches it
so D032's recorded numbers stay comparable.

Band edges are half-open, ``[low, high)``
-----------------------------------------
A frequency bin at exactly a band's upper edge belongs to the next band.  With
closed intervals, the 8 Hz bin would be counted in both theta and alpha, and
the two terms of the workload index — theta positive, alpha negative — would
share a bin and partly cancel.  That is a small error in a quantity whose sign
is the whole point.

Welch's tradeoff, stated once
-----------------------------
Welch splits a window into overlapping segments, periodograms each, and
averages.  More segments means lower variance per estimate and coarser
frequency resolution, because each segment is shorter.  For band *integrals*
over bands 4-5 Hz wide, resolution is cheap and variance is not, which is why
`SpectralConfig.welch_segment_seconds` defaults to 2 s inside a 4 s window (3
segments at 50% overlap, 0.5 Hz resolution, ~8 bins in theta) rather than
using the window whole (1 segment, no averaging, full chi-squared variance).
Both remain configurable; the choice is listed as an open item.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.signal import welch

from eeg.channels import canonicalise
from eeg.config import SpectralConfig
from eeg.windowing import Windows


class SpectralError(ValueError):
    """A spectrum cannot be estimated as configured."""


@dataclass
class PSD:
    """Welch power spectral density per window per channel.

    `power` is ``(n_windows, n_channels, n_freqs)`` in uV^2/Hz; `freqs` is
    ``(n_freqs,)`` in Hz.
    """

    freqs: np.ndarray
    power: np.ndarray
    channel_names: tuple[str, ...]
    sfreq: float
    #: Welch parameters actually used, in samples, so the resolution of a
    #: saved PSD is recoverable without the config.
    nperseg: int
    noverlap: int
    window: str
    detrend: str

    @property
    def freq_resolution(self) -> float:
        """Hz per bin."""
        return self.sfreq / self.nperseg

    def index_of(self, channel: str) -> int:
        canon = canonicalise(channel)
        try:
            return self.channel_names.index(canon)
        except ValueError:
            raise KeyError(f"channel {channel!r} not in this PSD; "
                           f"have {list(self.channel_names)}") from None


@dataclass
class WindowFeatures:
    """Band power per window per channel — the pipeline's output features.

    `power` is ``(n_windows, n_channels, n_bands)`` in uV^2, absolute band
    power.  `relative` is the same shape, each value divided by the total over
    all bands in that window and channel, or ``None`` if not computed.

    `clean` is the artifact mask from `eeg.artifacts`, carried along rather
    than applied.  **The features are computed for every window, including
    rejected ones.**  That is deliberate: a consumer that silently received
    only clean windows could not tell a recording with a 5% rejection rate from
    one with 60%, and §7 needs that number.  Consumers select with
    `clean_index()` or `band_series(..., clean_only=True)`.

    Why both absolute and relative power are returned: absolute power scales
    with electrode impedance, skin preparation and amplifier gain, none of
    which hold constant between sessions or between the two recording systems.
    Relative power is dimensionless and comparable, but it is also a
    *compositional* quantity — the bands sum to 1, so alpha falling forces the
    others to rise even if their absolute power did not change.  A workload
    index built on relative power therefore cannot move its theta and alpha
    terms independently.  Which to use is a scientific choice, left open.
    """

    power: np.ndarray
    channel_names: tuple[str, ...]
    band_names: tuple[str, ...]
    #: ``(n_windows,)`` window start times, in the windowed array's time base.
    start_seconds: np.ndarray
    #: Seconds to add to `start_seconds` for original-recording time.
    time_offset: float
    sfreq: float
    #: ``(n_windows,)`` bool clean mask, or all-True if detection did not run.
    clean: np.ndarray
    relative: np.ndarray | None = None
    config: SpectralConfig | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n_win, n_ch, n_band = self.power.shape
        if n_ch != len(self.channel_names):
            raise ValueError(
                f"power has {n_ch} channels but {len(self.channel_names)} names")
        if n_band != len(self.band_names):
            raise ValueError(f"power has {n_band} bands but {len(self.band_names)} names")
        if self.clean.shape != (n_win,):
            raise ValueError(f"clean has shape {self.clean.shape}, expected ({n_win},)")
        if self.start_seconds.shape != (n_win,):
            raise ValueError(
                f"start_seconds has shape {self.start_seconds.shape}, expected ({n_win},)")

    @property
    def n_windows(self) -> int:
        return self.power.shape[0]

    @property
    def n_clean(self) -> int:
        return int(np.count_nonzero(self.clean))

    def channel_index(self, channel: str) -> int:
        """Channel axis index, by canonical name.  Raises `KeyError` if absent.

        Raising rather than returning a sentinel is the point: a wrong channel
        index produces a plausible band power for the wrong electrode, which no
        downstream check would catch.
        """
        canon = canonicalise(channel)
        try:
            return self.channel_names.index(canon)
        except ValueError:
            raise KeyError(
                f"channel {channel!r} (canonical {canon!r}) not in these features; "
                f"have {list(self.channel_names)}") from None

    def band_index(self, band: str) -> int:
        """Band axis index, by name.  Raises `KeyError` if absent."""
        try:
            return self.band_names.index(band)
        except ValueError:
            raise KeyError(
                f"band {band!r} not in these features; have {list(self.band_names)}") from None

    def clean_index(self) -> np.ndarray:
        """Integer indices of clean windows, for slicing any per-window array."""
        return np.flatnonzero(self.clean)

    def band_series(
        self,
        channel: str,
        band: str,
        *,
        relative: bool = False,
        clean_only: bool = False,
    ) -> np.ndarray:
        """One channel's power in one band, as a 1-D series over windows.

        `clean_only=True` returns only the clean windows, in time order.  Use
        it together with `window_times(clean_only=True)` when the times matter;
        the two stay aligned.
        """
        source = self.relative if relative else self.power
        if source is None:
            raise SpectralError(
                "relative power was not computed; set SpectralConfig.relative_power=True")
        series = source[:, self.channel_index(channel), self.band_index(band)]
        return series[self.clean] if clean_only else series

    def roi_series(
        self,
        channels: tuple[str, ...] | list[str],
        band: str,
        *,
        relative: bool = False,
        clean_only: bool = False,
        require_all: bool = True,
    ) -> np.ndarray:
        """Mean over `channels` of one band's power, as a series over windows.

        The mean is over channels *within* each window, which is the right
        order for a cluster like Fz/F3/F4: averaging power across neighbouring
        electrodes reduces per-electrode noise without mixing windows together.

        `require_all=True` raises if any requested channel is absent, because a
        "frontal midline" mean silently computed over two of three electrodes
        is not the same quantity and nothing downstream would notice.  Set it
        False only for a deliberate, recorded degradation (pre-registration §7
        has one).
        """
        wanted = [canonicalise(c) for c in channels]
        present = [c for c in wanted if c in self.channel_names]
        absent = [c for c in wanted if c not in self.channel_names]
        if absent and require_all:
            raise KeyError(
                f"roi_series asked for {wanted} but {absent} are absent; have "
                f"{list(self.channel_names)}. Pass require_all=False to average over the "
                "channels that are present, and record the degradation.")
        if not present:
            raise KeyError(f"none of {wanted} is present; have {list(self.channel_names)}")
        source = self.relative if relative else self.power
        if source is None:
            raise SpectralError(
                "relative power was not computed; set SpectralConfig.relative_power=True")
        idx = [self.channel_index(c) for c in present]
        series = source[:, idx, self.band_index(band)].mean(axis=1)
        return series[self.clean] if clean_only else series

    def window_times(self, *, clean_only: bool = False, absolute: bool = False) -> np.ndarray:
        """Window start times, optionally only the clean ones."""
        times = self.start_seconds + self.time_offset if absolute else self.start_seconds
        return times[self.clean] if clean_only else times

    def to_summary(self) -> dict[str, Any]:
        """JSON-safe descriptive summary.  Medians over clean windows only.

        Medians rather than means because band power is strongly right-skewed:
        a single surviving artifact raises a mean far more than a median, and
        the median over clean windows is also the statistic pre-registration §6
        reduces each configuration to.
        """
        out: dict[str, Any] = {
            "n_windows": self.n_windows,
            "n_clean": self.n_clean,
            "channels": list(self.channel_names),
            "bands": list(self.band_names),
            "median_power_uv2": {},
        }
        if self.n_clean == 0:
            return out
        clean = self.power[self.clean]
        medians = np.median(clean, axis=0)      # (n_channels, n_bands)
        for i, channel in enumerate(self.channel_names):
            out["median_power_uv2"][channel] = {
                band: float(medians[i, j]) for j, band in enumerate(self.band_names)}
        if self.relative is not None:
            rel = np.median(self.relative[self.clean], axis=0)
            out["median_relative_power"] = {
                channel: {band: float(rel[i, j]) for j, band in enumerate(self.band_names)}
                for i, channel in enumerate(self.channel_names)}
        return out


# ----------------------------------------------------------------------
# Estimation
# ----------------------------------------------------------------------


def welch_psd(
    windows: Windows,
    config: SpectralConfig | None = None,
    *,
    channels: tuple[str, ...] | list[str] | None = None,
) -> PSD:
    """Welch PSD for every window and channel.

    Raises `SpectralError` rather than letting SciPy silently shorten a
    segment: SciPy clamps `nperseg` to the signal length with a warning, which
    would mean a run configured for 2 s segments quietly used whatever fitted,
    and the recorded config would no longer describe what happened.
    """
    config = config or SpectralConfig()
    config.validate()

    names = tuple(windows.channel_names) if channels is None else tuple(
        canonicalise(c) for c in channels)
    data = windows.data if channels is None else windows.pick(list(names))
    n_samp = data.shape[2]

    if config.welch_segment_seconds is None:
        nperseg = n_samp
    else:
        nperseg = int(round(config.welch_segment_seconds * windows.sfreq))
    if nperseg < 2:
        raise SpectralError(
            f"welch_segment_seconds={config.welch_segment_seconds} at {windows.sfreq} Hz "
            f"is {nperseg} samples; need at least 2")
    if nperseg > n_samp:
        raise SpectralError(
            f"Welch segment is {nperseg} samples but a window is only {n_samp}. "
            "SciPy would clamp the segment and warn, leaving the recorded config "
            "describing something that did not happen. Shorten "
            "welch_segment_seconds or lengthen window_seconds.")

    noverlap = int(round(config.welch_overlap * nperseg))
    if noverlap >= nperseg:
        noverlap = nperseg - 1

    detrend: str | bool = False if config.detrend == "none" else config.detrend
    freqs, power = welch(
        data, fs=windows.sfreq, window=config.window, nperseg=nperseg,
        noverlap=noverlap, detrend=detrend, scaling="density", axis=-1)
    return PSD(
        freqs=np.asarray(freqs, dtype=np.float64),
        power=np.asarray(power, dtype=np.float64),
        channel_names=names,
        sfreq=windows.sfreq,
        nperseg=nperseg,
        noverlap=noverlap,
        window=config.window,
        detrend=config.detrend,
    )


def band_power(
    psd: PSD, bands: tuple[tuple[str, float, float], ...]
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Integrate a PSD over each band.  Returns ``(power, band_names)``.

    `power` is ``(..., n_bands)`` with the PSD's leading axes preserved, in
    uV^2.  Integration is trapezoidal over the bins whose centre frequency
    falls in ``[low, high)``.

    Raises `SpectralError` if a band contains fewer than two bins, which is
    the silent-failure case worth catching: `np.trapezoid` over a single bin
    returns 0.0, so a misconfigured band would report zero power rather than
    an error, and a workload index whose theta term is structurally zero would
    still produce a difficulty ordering — of nothing.
    """
    power = np.empty(psd.power.shape[:-1] + (len(bands),), dtype=np.float64)
    names: list[str] = []
    for j, (name, lo, hi) in enumerate(bands):
        mask = (psd.freqs >= lo) & (psd.freqs < hi)
        n_bins = int(np.count_nonzero(mask))
        if n_bins < 2:
            raise SpectralError(
                f"band {name!r} ({lo}-{hi} Hz) contains {n_bins} frequency bin(s) at "
                f"{psd.freq_resolution:.4g} Hz resolution. Integrating over fewer than two "
                "bins returns 0.0 without erroring, which would make this band "
                "structurally zero in every window. Lengthen the Welch segment or widen "
                "the band.")
        power[..., j] = np.trapezoid(psd.power[..., mask], psd.freqs[mask], axis=-1)
        names.append(name)
    return power, tuple(names)


def extract_features(
    windows: Windows,
    config: SpectralConfig | None = None,
    *,
    clean: np.ndarray | None = None,
    channels: tuple[str, ...] | list[str] | None = None,
) -> WindowFeatures:
    """Welch PSD then band integration, packaged as `WindowFeatures`.

    `clean` is the artifact mask; all-True when omitted.  Features are
    computed for every window regardless — see `WindowFeatures.clean`.
    """
    config = config or SpectralConfig()
    psd = welch_psd(windows, config, channels=channels)
    power, band_names = band_power(psd, config.bands)

    relative = None
    if config.relative_power:
        total = power.sum(axis=-1, keepdims=True)
        # A window with zero total power is all-zero data (a dead channel, or a
        # synthetic silence).  0/0 would be NaN; reporting 0.0 keeps the array
        # finite, and the window is already rejected by the artifact detector's
        # flat-signal check, so no real analysis sees it.
        relative = np.divide(power, total, out=np.zeros_like(power), where=total > 0)

    if clean is None:
        clean_mask = np.ones(windows.n_windows, dtype=bool)
    else:
        clean_mask = np.asarray(clean, dtype=bool)
        if clean_mask.shape != (windows.n_windows,):
            raise SpectralError(
                f"clean has shape {clean_mask.shape}, expected ({windows.n_windows},)")

    return WindowFeatures(
        power=power,
        channel_names=psd.channel_names,
        band_names=band_names,
        start_seconds=windows.start_seconds.copy(),
        time_offset=windows.time_offset,
        sfreq=windows.sfreq,
        clean=clean_mask,
        relative=relative,
        config=config,
    )


def line_noise_ratio(
    data: np.ndarray,
    sfreq: float,
    mains_hz: float,
    *,
    half_width_hz: float = 1.0,
    reference_band: tuple[float, float] = (13.0, 30.0),
    segment_seconds: float = 4.0,
) -> np.ndarray:
    """Mains-band power divided by a reference band's, per channel.

    A **signal-quality diagnostic for the unfiltered signal**, not part of the
    pipeline's output.  It answers the question the notch config cannot:
    whether there is any mains contamination to remove in the first place.

    Measure it before filtering.  After a 45 Hz low-pass the ratio is a
    property of the filter's roll-off, not of the recording, and would say
    nothing about electrode contact.  That is also why it is useful at the
    hardware validation gate (pre-registration §10): a high ratio at one
    channel is the signature of a poorly seated or high-impedance electrode,
    which is actionable before a session rather than after it.

    Returns one ratio per channel, with `data` shaped ``(n_channels,
    n_samples)``.  A ratio near 1 is a clean electrode; tens or hundreds is
    mains dominating the EEG.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise SpectralError(f"data must be (n_channels, n_samples), got {data.shape}")
    nperseg = int(round(segment_seconds * sfreq))
    if nperseg > data.shape[1]:
        nperseg = data.shape[1]
    if nperseg < 4:
        raise SpectralError(
            f"signal is {data.shape[1]} samples; too short to estimate a spectrum")
    nyquist = sfreq / 2.0
    if mains_hz + half_width_hz >= nyquist:
        raise SpectralError(
            f"mains at {mains_hz} Hz (+/-{half_width_hz}) is at or above Nyquist "
            f"({nyquist} Hz) for a {sfreq} Hz recording; it is not in this signal at all")
    freqs, psd = welch(data, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2,
                       detrend="constant", scaling="density", axis=-1)
    line = (freqs >= mains_hz - half_width_hz) & (freqs <= mains_hz + half_width_hz)
    ref = (freqs >= reference_band[0]) & (freqs < reference_band[1])
    if not np.any(line) or not np.any(ref):
        raise SpectralError("no frequency bins in the mains or reference band")
    # Mean PSD height, not an integral: the two bands have very different
    # widths, and this is a ratio of densities by design.
    line_power = psd[..., line].mean(axis=-1)
    ref_power = psd[..., ref].mean(axis=-1)
    return np.divide(line_power, ref_power,
                     out=np.full_like(line_power, np.nan), where=ref_power > 0)


def dominant_frequency(psd: PSD, low: float, high: float) -> np.ndarray:
    """Frequency of the largest PSD bin within ``[low, high)``, per window/channel.

    Not used by the pipeline's output.  It exists for the hardware validation
    gate (pre-registration §10), which asks for an eyes-closed alpha increase
    at parietal and occipital sites: a genuine alpha rhythm shows a *peak* in
    the 8-13 Hz band, whereas broadband noise raises the band's integral
    without producing one.  Checking for the peak distinguishes the two.
    """
    mask = (psd.freqs >= low) & (psd.freqs < high)
    if not np.any(mask):
        raise SpectralError(f"no frequency bins in [{low}, {high}) Hz")
    band = psd.power[..., mask]
    return psd.freqs[mask][np.argmax(band, axis=-1)]
