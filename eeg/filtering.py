"""Bandpass and mains-notch filtering.

Every cutoff is an argument.  Nothing in this module knows a frequency that
did not arrive in a `FilterConfig`, because the one number most likely to be
wrong — the mains frequency — differs between the development fixture (50 Hz,
EEGMAT, recorded in Ukraine) and the author's own hardware (60 Hz, North
America), and a hardcoded notch would be wrong for exactly one of them while
reporting success for both.

Design choices and what fails if they are wrong
-----------------------------------------------
**Butterworth, as second-order sections.**  A high-order IIR expressed as
transfer-function coefficients loses precision badly when the cutoff is a small
fraction of Nyquist — 1 Hz against a 250 Hz Nyquist is a ratio of 0.004, well
into the range where a ``(b, a)`` representation of an order-8 filter is
numerically unstable and can produce output that is NaN or wildly scaled.  SOS
form is stable there.  `scipy.signal.butter(..., output="sos")`.

**Zero-phase by default** (`sosfiltfilt`, forward then backward).  A causal
filter delays each frequency by a different amount, so a blink's onset appears
later in the filtered signal than it really was, and by a different amount in
theta than in alpha.  A rejected window would then be misaligned against the
game event that caused it.  Zero-phase filtering has no delay; the price is
that it reads future samples, so it cannot run online.  A real-time path would
need `zero_phase=False` and its own validation.

**Edge trimming.**  Filtering a finite record leaves a transient at each end.
`sosfiltfilt` pads to suppress it but does not remove it, and the residue is a
large excursion — which the amplitude detector would correctly reject, giving
every recording a dirty first window for reasons that have nothing to do with
the participant.  Trimming is honest: it removes the samples rather than
keeping them and hoping.

**The notch is narrow on purpose.**  `notch_quality` (Q) sets the -3 dB width
to ``f0 / Q``, so Q=30 at 60 Hz is a 2 Hz notch.  A wide notch at 50 Hz would
reach down toward the beta band; a notch narrower than the mains' own ±0.5 Hz
wander would miss the line.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfiltfilt, tf2sos

from eeg.config import FilterConfig
from eeg.loading import Recording


class FilterError(ValueError):
    """A filter cannot be applied to this signal as configured."""


def _check_against_nyquist(freq: float, sfreq: float, what: str) -> None:
    nyquist = sfreq / 2.0
    if freq >= nyquist:
        raise FilterError(
            f"{what} at {freq} Hz is at or above the Nyquist frequency ({nyquist} Hz) for "
            f"a {sfreq} Hz recording; it cannot be represented, let alone filtered. "
            "Either the sampling rate or the cutoff is wrong.")


def design_bandpass(
    low: float | None, high: float | None, sfreq: float, order: int = 4
) -> np.ndarray:
    """Butterworth SOS coefficients for a band-, high- or low-pass.

    `low=None` gives a pure low-pass, `high=None` a pure high-pass; both
    `None` is an error, because a caller asking for no filter should not be
    calling this.
    """
    if low is None and high is None:
        raise FilterError("design_bandpass needs at least one of low, high")
    if sfreq <= 0:
        raise FilterError(f"sfreq must be > 0, got {sfreq}")
    if order < 1:
        raise FilterError(f"order must be >= 1, got {order}")

    nyquist = sfreq / 2.0
    if low is not None:
        _check_against_nyquist(low, sfreq, "high-pass cutoff")
    if high is not None:
        _check_against_nyquist(high, sfreq, "low-pass cutoff")

    if low is not None and high is not None:
        if low >= high:
            raise FilterError(f"high-pass cutoff {low} Hz must be below low-pass {high} Hz")
        return butter(order, (low / nyquist, high / nyquist), btype="bandpass", output="sos")
    if low is not None:
        return butter(order, low / nyquist, btype="highpass", output="sos")
    return butter(order, high / nyquist, btype="lowpass", output="sos")  # type: ignore[operator]


def design_notch(freq: float, quality: float, sfreq: float) -> np.ndarray:
    """SOS coefficients for one IIR notch at `freq` with quality factor `quality`.

    `scipy.signal.iirnotch` returns transfer-function coefficients; they are
    converted to SOS so a chain of notches composes with the bandpass in a
    single numerically stable representation.
    """
    if quality <= 0:
        raise FilterError(f"notch quality must be > 0, got {quality}")
    _check_against_nyquist(freq, sfreq, "notch frequency")
    b, a = iirnotch(freq, quality, fs=sfreq)
    return tf2sos(b, a)


#: Signal extension used at both ends for zero-phase filtering.
#:
#: ``"even"`` — mirror extension — not SciPy's ``"odd"`` default.  This was
#: measured, and the difference is large.
#:
#: Odd extension reflects through the endpoint: it continues the signal as
#: ``2 * x[0] - x[k]``.  For a signal whose first sample happens to be far
#: from zero — an EEG trace with any DC offset or caught mid-oscillation — that
#: builds an artificial ramp of magnitude ``2 * x[0]`` across the pad, which is
#: almost pure low-frequency content.  A filter with a low cutoff responds to
#: it strongly, and the response lands inside the first analysis window.
#:
#: Measured on a synthetic 120 s recording containing no artifact at all,
#: band-limited to the 1-6 Hz ocular range, comparing the first window's
#: amplitude against the median window's (4.53 uV):
#:
#: ===============  ==============  ==============
#: padtype          pad 27 samples  pad 1000
#: ===============  ==============  ==============
#: ``"odd"``        24.86 uV        21.75 uV
#: ``"constant"``   10.18 uV        10.18 uV
#: ``"even"``        8.61 uV         4.75 uV
#: ===============  ==============  ==============
#:
#: Only even extension plus a long pad brings the first window back to the
#: median.  A longer pad does **not** rescue odd extension, because the ramp it
#: creates grows with the pad.  With the SciPy default the artifact detector
#: flagged window 0 of every recording as ocular — a 3% rejection rate on data
#: containing no blink — which is a textbook instance of the over-strict
#: failure mode pre-registration §7 warns about, manufactured entirely by a
#: padding choice.  Mirror extension is also what MNE's own filtering uses.
PAD_TYPE = "even"


def _apply(
    sos: np.ndarray, data: np.ndarray, zero_phase: bool, pad_samples: int | None = None
) -> np.ndarray:
    """Run one SOS chain along the last axis.

    `pad_samples` overrides `sosfiltfilt`'s default pad length.  SciPy's
    default is ``3 * (2 * n_sections + 1)`` — 27 samples for an order-4
    bandpass, 54 ms at 500 Hz.  That is ample for a wide passband and short
    for a narrow low-frequency one, whose settling time runs to hundreds of
    milliseconds.  See `PAD_TYPE` for the measurement.
    """
    if zero_phase:
        default_pad = 3 * (2 * sos.shape[0] + 1)
        padlen = default_pad if pad_samples is None else int(pad_samples)
        if data.shape[-1] <= padlen:
            raise FilterError(
                f"signal is {data.shape[-1]} samples, too short for zero-phase filtering "
                f"with a pad of {padlen} (needs more). Either reduce the pad, lower the "
                "filter order, use zero_phase=False, or pass a longer segment.")
        return sosfiltfilt(sos, data, axis=-1, padtype=PAD_TYPE, padlen=padlen)
    return sosfilt(sos, data, axis=-1)


def bandpass(
    data: np.ndarray,
    sfreq: float,
    low: float | None,
    high: float | None,
    order: int = 4,
    zero_phase: bool = True,
    pad_seconds: float | None = None,
) -> np.ndarray:
    """Bandpass an array along its last axis.  Returns a new array.

    `pad_seconds` sets the zero-phase odd-extension pad.  Leave it `None` for
    SciPy's default, which suits a wide passband; set it to a couple of seconds
    for a narrow low-frequency band, where the default is far too short (see
    `_apply`).
    """
    sos = design_bandpass(low, high, sfreq, order)
    pad = None if pad_seconds is None else int(round(pad_seconds * sfreq))
    return _apply(sos, np.asarray(data, dtype=np.float64), zero_phase, pad)


def notch(
    data: np.ndarray,
    sfreq: float,
    freqs: tuple[float, ...] | list[float],
    quality: float = 30.0,
    zero_phase: bool = True,
    pad_seconds: float | None = None,
) -> np.ndarray:
    """Apply one notch per entry in `freqs`, in order, along the last axis."""
    out = np.asarray(data, dtype=np.float64)
    pad = None if pad_seconds is None else int(round(pad_seconds * sfreq))
    for freq in freqs:
        out = _apply(design_notch(freq, quality, sfreq), out, zero_phase, pad)
    return out


def apply_filters(recording: Recording, config: FilterConfig | None = None) -> Recording:
    """Filter a whole `Recording` and return a new one.

    Order: notch first, then bandpass, then edge trim.

    Why the notch comes first — the notch is the one filter whose target is a
    single known contaminant, and running it on the unaltered signal means its
    effect is interpretable on its own.  With the bandpass first, a 60 Hz notch
    on data already low-passed at 45 Hz operates on the bandpass's roll-off
    rather than on mains, and would appear to have done something when it did
    nothing.  The config's cross-section validation refuses that combination
    outright, but the ordering keeps the arithmetic honest if the validation is
    ever relaxed.

    The input is not modified.  `provenance.filters_applied` on the returned
    recording lists each stage, so a result carries its own filter history even
    if the config is separated from it.
    """
    config = config or FilterConfig()
    config.validate()
    data = recording.data
    sfreq = recording.sfreq
    stages: list[str] = []

    if config.notch_freqs:
        data = notch(data, sfreq, config.notch_freqs, config.notch_quality, config.zero_phase)
        stages.append(
            f"notch at {', '.join(f'{f:g}' for f in config.notch_freqs)} Hz, "
            f"Q={config.notch_quality:g}")

    if config.bandpass_low is not None or config.bandpass_high is not None:
        data = bandpass(data, sfreq, config.bandpass_low, config.bandpass_high,
                        config.bandpass_order, config.zero_phase)
        lo = "DC" if config.bandpass_low is None else f"{config.bandpass_low:g}"
        hi = "Nyquist" if config.bandpass_high is None else f"{config.bandpass_high:g}"
        stages.append(f"Butterworth order {config.bandpass_order} bandpass {lo}-{hi} Hz")

    stages.append("zero-phase (sosfiltfilt)" if config.zero_phase else "causal (sosfilt)")

    trimmed = 0
    if config.edge_trim_seconds > 0:
        trimmed = int(round(config.edge_trim_seconds * sfreq))
        if data.shape[1] <= 2 * trimmed:
            raise FilterError(
                f"edge_trim_seconds={config.edge_trim_seconds} would trim "
                f"{2 * trimmed} of {data.shape[1]} samples, leaving nothing. "
                "The recording is too short for this trim.")
        data = data[:, trimmed:data.shape[1] - trimmed]
        stages.append(f"trimmed {config.edge_trim_seconds:g} s ({trimmed} samples) from each end")

    provenance = replace(
        recording.provenance,
        n_samples=int(data.shape[1]),
        duration_seconds=float(data.shape[1] / sfreq),
        filters_applied=tuple(stages),
    )
    return Recording(
        data=np.ascontiguousarray(data),
        channel_names=recording.channel_names,
        sfreq=sfreq,
        provenance=provenance,
        positions=dict(recording.positions),
    )
