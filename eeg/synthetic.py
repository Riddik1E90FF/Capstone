"""Synthetic EEG with known spectral content, and artifacts with known shapes.

Two jobs.

**Testing.**  A test that asserts "a 10 Hz sine shows an alpha peak" is a real
test of the spectral code; a test that asserts the pipeline runs without
raising is not.  Everything here has a closed-form answer, so the tests can
check numbers rather than the absence of exceptions.

**The EEG source abstraction.**  CLAUDE.md requires the design to support
recorded replay, synthetic signals and live hardware behind one interface.
Synthetic is the cheapest of the three and the only one available with no data
and no hardware, so the pipeline being exercisable on it is what lets the
curriculum interface be built and tested before either exists.

Honesty about what this is
--------------------------
These signals are **not a model of EEG**.  A sum of sinusoids plus Gaussian
noise has none of the 1/f background, none of the spatial covariance between
neighbouring electrodes, and none of the non-stationarity of a real recording.
Nothing about physiology may be concluded from them, and no threshold should be
calibrated on them — a rejection threshold tuned against synthetic noise would
be tuned against the wrong noise distribution entirely.  They test *code*.

Every generator takes a seed and uses `numpy.random.default_rng`, whose PCG64
stream is stable across NumPy versions by documented policy (see D033, where
that property is what let Gate 1a be re-proved across a major version bump).
Same seed, same signal, on any machine.
"""

from __future__ import annotations

import numpy as np

from eeg.config import LoadConfig
from eeg.loading import Recording, prepare

#: Nominal per-channel background noise, microvolts RMS.  Chosen so that a
#: default-threshold artifact detector rejects essentially nothing in a clean
#: synthetic recording: 5 uV RMS gives roughly 30 uV peak-to-peak over a 4 s
#: window, well under the 150 uV default ceiling and well over the 0.5 uV flat
#: floor.  A test that needs rejection injects an artifact rather than raising
#: the noise.
DEFAULT_NOISE_UV = 5.0

#: Band the background noise is limited to, in Hz.
#:
#: This default is not cosmetic.  *White* noise at 500 Hz has enormous
#: sample-to-sample differences — at 5 uV RMS the adjacent-sample difference
#: has a standard deviation of 5 * sqrt(2) uV, so over a 2000-sample window
#: its maximum routinely exceeds 25 uV, which is `ArtifactConfig`'s default
#: gradient ceiling.  A "clean" white-noise fixture would therefore be
#: rejected wholesale, and the natural but wrong conclusion would be that the
#: gradient threshold is too strict.  It is not: real EEG arrives at the
#: detector already band-limited by `eeg.filtering` (the pipeline filters
#: before it windows), and band-limited noise has sample-to-sample steps
#: smaller by roughly the ratio of the bandwidth to Nyquist.  Measured on
#: EEGMAT, the gradient check fires on 0% of windows.
#:
#: So the fixture is band-limited to match what the detector actually sees.
#: Pass ``noise_band=None`` for genuinely white noise, which is the right
#: choice when testing the filters themselves.
DEFAULT_NOISE_BAND = (1.0, 45.0)

#: Seconds of extra noise generated at each end before band-limiting, then
#: cropped away, so the fixture carries none of the band-limiting filter's own
#: edge transient.  See the comment in `synthetic_recording`.
_NOISE_MARGIN_SECONDS = 4.0


def synthetic_recording(
    channels: tuple[str, ...] | list[str] = ("Fz", "F3", "F4", "Pz", "P3", "P4", "Fp1", "Fp2"),
    *,
    sfreq: float = 500.0,
    duration_seconds: float = 60.0,
    components: tuple[tuple[float, float], ...] = ((10.0, 20.0),),
    noise_uv: float = DEFAULT_NOISE_UV,
    noise_band: tuple[float, float] | None = DEFAULT_NOISE_BAND,
    seed: int = 0,
    per_channel_components: dict[str, tuple[tuple[float, float], ...]] | None = None,
    load_config: LoadConfig | None = None,
    source: str = "synthetic",
) -> Recording:
    """A `Recording` of sinusoids plus white noise, in microvolts.

    `components` is ``((frequency_hz, amplitude_uv), ...)`` applied to every
    channel; `per_channel_components` overrides it for named channels, which is
    how a test builds a recording with alpha only at Pz.

    A sinusoid of amplitude ``A`` has power ``A**2 / 2``, so the band
    containing it integrates to approximately that, plus the noise floor's
    contribution over the band's width.  That is the quantity
    `tests/test_eeg_spectral.py` checks against.

    `noise_band` band-limits the background noise (see `DEFAULT_NOISE_BAND`
    for why that matters), and is rescaled afterwards so `noise_uv` remains the
    realised RMS rather than the RMS before filtering.  Pass ``None`` for white
    noise.

    The montage defaults to ``"none"`` unless `load_config` says otherwise:
    synthetic data needs no sensor geometry, and requiring it would make these
    helpers depend on MNE.
    """
    rng = np.random.default_rng(seed)
    names = tuple(channels)
    n_samples = int(round(duration_seconds * sfreq))
    if n_samples < 1:
        raise ValueError(f"duration_seconds={duration_seconds} at {sfreq} Hz is no samples")
    t = np.arange(n_samples, dtype=np.float64) / sfreq

    data = np.empty((len(names), n_samples), dtype=np.float64)
    for i, name in enumerate(names):
        spec = (per_channel_components or {}).get(name, components)
        row = np.zeros(n_samples, dtype=np.float64)
        for freq, amplitude in spec:
            # A fixed phase per (channel, frequency), drawn from the seeded
            # stream rather than left at zero: with every channel in phase, an
            # average reference would cancel the signal exactly, which is a
            # behaviour no real recording has and would make the
            # average-reference test pass for the wrong reason.
            phase = rng.uniform(0.0, 2.0 * np.pi)
            row += amplitude * np.sin(2.0 * np.pi * freq * t + phase)
        if noise_uv > 0:
            if noise_band is None:
                noise = rng.normal(0.0, noise_uv, size=n_samples)
            else:
                from eeg.filtering import bandpass
                # Generate a margin at each end, filter, then crop it away.
                # Filtering exactly n_samples would leave the band-limiting
                # filter's own start-up transient at both ends of the fixture,
                # and a 1 Hz filter edge settles over hundreds of
                # milliseconds.  That transient is a large low-frequency
                # excursion, so a later narrow re-filter — the artifact
                # detector's 1-6 Hz ocular reference — would find it and flag
                # the first and last window of every synthetic recording as
                # ocular.  Measured before this crop was added: exactly that,
                # on data containing no blink at all.
                margin = int(round(_NOISE_MARGIN_SECONDS * sfreq))
                wide = rng.normal(0.0, noise_uv, size=n_samples + 2 * margin)
                wide = bandpass(wide, sfreq, noise_band[0], noise_band[1],
                                order=4, zero_phase=True)
                noise = wide[margin:margin + n_samples]
                # Band-limiting removes power, so the realised RMS would
                # otherwise be well under `noise_uv` and a caller reasoning
                # about amplitudes would be reasoning about the wrong number.
                realised = float(np.sqrt(np.mean(np.square(noise))))
                if realised > 0:
                    noise = noise * (noise_uv / realised)
            row += noise
        data[i] = row

    config = load_config or LoadConfig(montage="none")
    return prepare(data, list(names), sfreq, config,
                   source=source, already_microvolts=True)


def inject_excursion(
    recording: Recording,
    channel: str,
    start_seconds: float,
    duration_seconds: float,
    amplitude_uv: float,
) -> Recording:
    """Add a rectangular offset to one channel, in place, returning it.

    Tests the peak-to-peak check.  A rectangular block is the clearest case:
    its contribution to the window's range is exactly `amplitude_uv`, so a
    threshold test has an unambiguous expected verdict.
    """
    i = recording.index_of(channel)
    lo = int(round(start_seconds * recording.sfreq))
    hi = lo + int(round(duration_seconds * recording.sfreq))
    if lo < 0 or hi > recording.n_samples:
        raise ValueError(
            f"excursion {start_seconds}-{start_seconds + duration_seconds} s falls outside "
            f"a {recording.duration_seconds:g} s recording")
    recording.data[i, lo:hi] += amplitude_uv
    return recording


def inject_step(
    recording: Recording, channel: str, at_seconds: float, amplitude_uv: float
) -> Recording:
    """Add a single-sample jump, in place, returning the recording.

    Tests the gradient check specifically.  A one-sample step of ``A`` uV
    produces a sample-to-sample difference of ``A``, and because it is only one
    sample wide it barely moves the window's peak-to-peak range — which is the
    whole reason the gradient check exists as a separate test from the
    amplitude check.
    """
    i = recording.index_of(channel)
    k = int(round(at_seconds * recording.sfreq))
    if not 1 <= k < recording.n_samples:
        raise ValueError(f"at_seconds={at_seconds} is outside the recording's interior")
    recording.data[i, k] += amplitude_uv
    return recording


def inject_blink(
    recording: Recording,
    channels: tuple[str, ...] | list[str] = ("Fp1", "Fp2"),
    at_seconds: float = 10.0,
    amplitude_uv: float = 200.0,
    width_seconds: float = 0.3,
) -> Recording:
    """Add a blink-shaped deflection to the frontopolar channels, in place.

    The shape is a Gaussian bump: a blink's scalp signature is a large
    monophasic deflection of roughly 200-400 uV at Fp1/Fp2 lasting 200-400 ms,
    and a Gaussian of that width has its energy concentrated below ~6 Hz, which
    is the band `ArtifactConfig.ocular_band` looks in.

    This is a **caricature, not a model.**  A real blink is asymmetric, is
    correlated across frontal channels in a way this is not, and co-occurs with
    the eyelid muscle activity that puts broadband power into the signal.  It
    exists to prove the detector fires on an ocular-range frontopolar
    excursion, which is exactly the claim the detector makes.
    """
    centre = at_seconds
    sigma = width_seconds / 2.0
    t = np.arange(recording.n_samples, dtype=np.float64) / recording.sfreq
    bump = amplitude_uv * np.exp(-0.5 * ((t - centre) / sigma) ** 2)
    for channel in channels:
        recording.data[recording.index_of(channel)] += bump
    return recording


def inject_drift(
    recording: Recording, channel: str, amplitude_uv: float = 500.0, cycles: float = 0.5
) -> Recording:
    """Add a very slow baseline drift, in place.

    Tests that the high-pass does its job: a 500 uV drift over the whole record
    is far above the amplitude ceiling, and a pipeline whose high-pass works
    rejects nothing because of it while one whose high-pass is absent or
    mis-specified rejects nearly every window.
    """
    i = recording.index_of(channel)
    t = np.linspace(0.0, cycles * 2.0 * np.pi, recording.n_samples)
    recording.data[i] += amplitude_uv * np.sin(t)
    return recording


def inject_line_noise(
    recording: Recording, mains_hz: float = 60.0, amplitude_uv: float = 30.0
) -> Recording:
    """Add a mains sinusoid to every channel, in place.

    Tests the notch, and `eeg.spectral.line_noise_ratio` as a diagnostic.
    """
    t = np.arange(recording.n_samples, dtype=np.float64) / recording.sfreq
    recording.data += amplitude_uv * np.sin(2.0 * np.pi * mains_hz * t)
    return recording
