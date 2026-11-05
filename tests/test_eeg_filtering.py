"""Tests for eeg/filtering.py, against synthetic signals with known spectra.

Each test states the physical claim it checks, because "the filter ran" is not
a test of a filter.  A signal at a known frequency must survive the passband,
one outside it must not, a notch must attenuate its target and leave its
neighbours, and a drift the high-pass is supposed to remove must actually be
gone.
"""

from __future__ import annotations

import numpy as np
import pytest

from eeg.config import FilterConfig, LoadConfig
from eeg.filtering import (FilterError, apply_filters, bandpass, design_bandpass,
                           design_notch, notch)
from eeg.loading import prepare
from eeg.synthetic import inject_drift, inject_line_noise, synthetic_recording

SFREQ = 500.0


def _sine(freq: float, seconds: float = 20.0, amplitude: float = 10.0,
          sfreq: float = SFREQ) -> np.ndarray:
    t = np.arange(int(seconds * sfreq)) / sfreq
    return amplitude * np.sin(2.0 * np.pi * freq * t)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def _power_at(signal: np.ndarray, freq: float, sfreq: float = SFREQ,
              half_width: float = 1.0) -> float:
    """Mean PSD height in a narrow band around `freq`."""
    from scipy.signal import welch
    f, p = welch(signal, fs=sfreq, nperseg=int(4 * sfreq), detrend="constant")
    mask = (f >= freq - half_width) & (f <= freq + half_width)
    return float(p[mask].mean())


class TestDesign:
    def test_bandpass_sos_shape(self) -> None:
        sos = design_bandpass(1.0, 45.0, SFREQ, order=4)
        assert sos.shape[1] == 6
        assert np.all(np.isfinite(sos))

    def test_highpass_only(self) -> None:
        assert design_bandpass(1.0, None, SFREQ).shape[1] == 6

    def test_lowpass_only(self) -> None:
        assert design_bandpass(None, 45.0, SFREQ).shape[1] == 6

    def test_no_cutoff_at_all_is_an_error(self) -> None:
        with pytest.raises(FilterError, match="at least one of low, high"):
            design_bandpass(None, None, SFREQ)

    def test_cutoff_at_or_above_nyquist_is_an_error(self) -> None:
        """At 500 Hz the Nyquist is 250. A 300 Hz cutoff is not representable,
        and SciPy's own message does not say so in those terms."""
        with pytest.raises(FilterError, match="Nyquist"):
            design_bandpass(1.0, 300.0, SFREQ)
        with pytest.raises(FilterError, match="Nyquist"):
            design_bandpass(1.0, 250.0, SFREQ)

    def test_notch_above_nyquist_is_an_error(self) -> None:
        """The realistic version of this: a 60 Hz notch on a 100 Hz recording.
        Mains is not in that signal at all, it is aliased."""
        with pytest.raises(FilterError, match="Nyquist"):
            design_notch(60.0, 30.0, 100.0)

    def test_notch_sos_is_finite(self) -> None:
        assert np.all(np.isfinite(design_notch(60.0, 30.0, SFREQ)))

    def test_inverted_band_is_an_error(self) -> None:
        with pytest.raises(FilterError, match="must be below"):
            design_bandpass(45.0, 1.0, SFREQ)


class TestBandpass:
    def test_in_band_sine_survives(self) -> None:
        """A 10 Hz sine inside a 1-45 Hz passband keeps nearly all its
        amplitude."""
        x = _sine(10.0)
        y = bandpass(x, SFREQ, 1.0, 45.0)
        # Edges carry the filter transient; compare the interior.
        assert _rms(y[1000:-1000]) == pytest.approx(_rms(x[1000:-1000]), rel=0.02)

    def test_out_of_band_sine_is_attenuated(self) -> None:
        """A 100 Hz sine against a 45 Hz low-pass must lose most of its power.

        Order 4 applied zero-phase is effectively order 8, so at just over
        twice the cutoff the attenuation is large; 20x in amplitude is a
        conservative floor that does not encode the exact filter design.
        """
        x = _sine(100.0)
        y = bandpass(x, SFREQ, 1.0, 45.0)
        assert _rms(y[1000:-1000]) < _rms(x) / 20.0

    def test_dc_offset_is_removed_by_the_highpass(self) -> None:
        x = _sine(10.0) + 500.0
        y = bandpass(x, SFREQ, 1.0, 45.0)
        assert abs(float(np.mean(y[1000:-1000]))) < 1.0

    def test_slow_drift_is_removed(self) -> None:
        """The reason bandpass_low exists: a 500 uV baseline drift would
        dominate peak-to-peak amplitude and make every artifact threshold
        meaningless."""
        rec = synthetic_recording(("Fz",), duration_seconds=60.0, noise_uv=1.0, seed=1)
        inject_drift(rec, "Fz", amplitude_uv=500.0)
        before = float(np.ptp(rec.data[0]))
        after = float(np.ptp(bandpass(rec.data, SFREQ, 1.0, 45.0)[0, 1000:-1000]))
        assert before > 400.0
        assert after < 100.0

    def test_zero_phase_does_not_shift_a_pulse_in_time(self) -> None:
        """The reason zero_phase defaults True: a causal filter delays each
        frequency differently, so a rejected window would be misaligned
        against the game event that caused it."""
        x = np.zeros(int(20 * SFREQ))
        x[5000] = 100.0
        zero = bandpass(x, SFREQ, 1.0, 45.0, zero_phase=True)
        causal = bandpass(x, SFREQ, 1.0, 45.0, zero_phase=False)
        assert abs(int(np.argmax(np.abs(zero))) - 5000) <= 2
        assert int(np.argmax(np.abs(causal))) > 5002

    def test_signal_too_short_for_zero_phase_gives_a_useful_error(self) -> None:
        with pytest.raises(FilterError, match="too short for zero-phase"):
            bandpass(np.zeros(10), SFREQ, 1.0, 45.0, zero_phase=True)

    def test_input_is_not_modified(self) -> None:
        x = _sine(10.0)
        original = x.copy()
        bandpass(x, SFREQ, 1.0, 45.0)
        assert np.array_equal(x, original)


class TestNotch:
    def test_notch_attenuates_its_target(self) -> None:
        """30 uV of 60 Hz mains on top of a 10 Hz signal: after a 60 Hz notch
        the mains power must drop by at least 100x."""
        rec = synthetic_recording(("Fz",), duration_seconds=30.0, noise_uv=0.0,
                                  components=((10.0, 10.0),), seed=2)
        inject_line_noise(rec, mains_hz=60.0, amplitude_uv=30.0)
        before = _power_at(rec.data[0], 60.0)
        after = _power_at(notch(rec.data, SFREQ, (60.0,), 30.0)[0, 1000:-1000], 60.0)
        assert after < before / 100.0

    def test_notch_leaves_the_bands_of_interest_alone(self) -> None:
        """A Q=30 notch at 60 Hz is 2 Hz wide, so theta and alpha must be
        untouched. A wide notch would quietly attenuate beta."""
        rec = synthetic_recording(
            ("Fz",), duration_seconds=30.0, noise_uv=0.0,
            components=((6.0, 10.0), (10.0, 10.0), (25.0, 10.0)), seed=3)
        inject_line_noise(rec, 60.0, 30.0)
        after = notch(rec.data, SFREQ, (60.0,), 30.0)[0, 1000:-1000]
        for freq in (6.0, 10.0, 25.0):
            assert _power_at(after, freq) == pytest.approx(
                _power_at(rec.data[0, 1000:-1000], freq), rel=0.05)

    def test_wrong_mains_frequency_removes_nothing(self) -> None:
        """The classic cross-dataset error, asserted so the cost is explicit: a
        60 Hz notch on 50 Hz-contaminated data leaves the line in place."""
        rec = synthetic_recording(("Fz",), duration_seconds=30.0, noise_uv=0.0,
                                  components=((10.0, 10.0),), seed=4)
        inject_line_noise(rec, mains_hz=50.0, amplitude_uv=30.0)
        before = _power_at(rec.data[0], 50.0)
        wrong = _power_at(notch(rec.data, SFREQ, (60.0,), 30.0)[0, 1000:-1000], 50.0)
        right = _power_at(notch(rec.data, SFREQ, (50.0,), 30.0)[0, 1000:-1000], 50.0)
        assert wrong > before / 2.0           # essentially untouched
        assert right < before / 100.0

    def test_multiple_notches_compose(self) -> None:
        rec = synthetic_recording(("Fz",), duration_seconds=30.0, noise_uv=0.0,
                                  components=((10.0, 10.0),), seed=5)
        inject_line_noise(rec, 60.0, 30.0)
        inject_line_noise(rec, 120.0, 30.0)
        out = notch(rec.data, SFREQ, (60.0, 120.0), 30.0)[0, 1000:-1000]
        assert _power_at(out, 60.0) < _power_at(rec.data[0], 60.0) / 100.0
        assert _power_at(out, 120.0) < _power_at(rec.data[0], 120.0) / 100.0


class TestApplyFilters:
    def test_edge_trim_shortens_the_record_by_twice_the_trim(self) -> None:
        rec = synthetic_recording(("Fz", "Pz"), duration_seconds=30.0, seed=6)
        out = apply_filters(rec, FilterConfig(edge_trim_seconds=1.0))
        assert out.n_samples == rec.n_samples - 2 * int(SFREQ)
        assert out.duration_seconds == pytest.approx(28.0)

    def test_no_trim_keeps_every_sample(self) -> None:
        rec = synthetic_recording(("Fz",), duration_seconds=30.0, seed=7)
        out = apply_filters(rec, FilterConfig(edge_trim_seconds=0.0))
        assert out.n_samples == rec.n_samples

    def test_trim_longer_than_the_record_is_an_error(self) -> None:
        rec = synthetic_recording(("Fz",), duration_seconds=5.0, seed=8)
        with pytest.raises(FilterError, match="leaving nothing"):
            apply_filters(rec, FilterConfig(edge_trim_seconds=10.0))

    def test_provenance_records_every_stage(self) -> None:
        """A result separated from its config must still say what filtered it."""
        rec = synthetic_recording(("Fz",), duration_seconds=30.0, seed=9)
        out = apply_filters(rec, FilterConfig(
            bandpass_low=1.0, bandpass_high=80.0, notch_freqs=(60.0,)))
        stages = " | ".join(out.provenance.filters_applied)
        assert "notch at 60 Hz" in stages
        assert "bandpass 1-80 Hz" in stages
        assert "zero-phase" in stages
        assert "trimmed 1 s" in stages

    def test_provenance_duration_matches_the_array(self) -> None:
        rec = synthetic_recording(("Fz",), duration_seconds=30.0, seed=10)
        out = apply_filters(rec, FilterConfig())
        assert out.provenance.n_samples == out.n_samples
        assert out.provenance.duration_seconds == pytest.approx(out.duration_seconds)

    def test_input_recording_is_not_modified(self) -> None:
        rec = synthetic_recording(("Fz",), duration_seconds=30.0, seed=11)
        before = rec.data.copy()
        apply_filters(rec, FilterConfig())
        assert np.array_equal(rec.data, before)

    def test_channel_names_and_positions_survive(self) -> None:
        data = np.zeros((2, int(30 * SFREQ)))
        rec = prepare(data + 1.0, ["Fz", "Pz"], SFREQ,
                      LoadConfig(montage="standard_1020"), already_microvolts=True)
        out = apply_filters(rec, FilterConfig())
        assert out.channel_names == ("Fz", "Pz")
        assert set(out.positions) == {"Fz", "Pz"}

    def test_filtering_is_deterministic(self) -> None:
        rec = synthetic_recording(("Fz", "Pz"), duration_seconds=30.0, seed=12)
        a = apply_filters(rec, FilterConfig()).data
        b = apply_filters(rec, FilterConfig()).data
        assert np.array_equal(a, b)
