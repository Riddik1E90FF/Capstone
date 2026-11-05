"""Tests for eeg/spectral.py, against signals whose spectra are known exactly.

The anchor test is the one the pipeline's whole purpose rests on: a 10 Hz sine
must put its power in the alpha band and essentially nowhere else.  If that
fails, every band power the project ever reports is wrong, and no amount of
downstream analysis would reveal it.

Analytic expectation used throughout: a sinusoid of amplitude A has mean square
power A^2 / 2, so integrating a PSD over a band that contains it should recover
approximately A^2 / 2 in uV^2.  White noise of sigma uV RMS spreads sigma^2
uniformly over the full 0-Nyquist range, contributing
``sigma^2 * band_width / nyquist`` to a band.
"""

from __future__ import annotations

import numpy as np
import pytest

from eeg.config import EpochConfig, SpectralConfig
from eeg.spectral import (PSD, SpectralError, band_power, dominant_frequency, extract_features,
                          line_noise_ratio, welch_psd)
from eeg.synthetic import inject_line_noise, synthetic_recording
from eeg.windowing import make_windows

SFREQ = 500.0
BANDS = SpectralConfig().bands


def _windows(components, channels=("Fz",), duration=40.0, noise=0.0, seed=0,
             per_channel=None, window_seconds=4.0, noise_band=(1.0, 45.0)):
    rec = synthetic_recording(
        channels, duration_seconds=duration, components=components,
        noise_uv=noise, noise_band=noise_band, seed=seed,
        per_channel_components=per_channel)
    return make_windows(rec, EpochConfig(window_seconds=window_seconds))


class TestKnownSpectra:
    def test_ten_hertz_sine_shows_an_alpha_peak(self) -> None:
        """The anchor test. A 10 Hz sine must land in alpha (8-13 Hz) and
        dominate every other band."""
        w = _windows(((10.0, 20.0),))
        features = extract_features(w, SpectralConfig())
        power = features.power[:, 0, :].mean(axis=0)
        bands = dict(zip(features.band_names, power))
        assert bands["alpha"] == max(bands.values())
        for other in ("delta", "theta", "beta"):
            assert bands["alpha"] > 50.0 * bands[other], (
                f"alpha {bands['alpha']:.4g} not dominant over {other} {bands[other]:.4g}")

    @pytest.mark.parametrize("freq,expected_band", [
        (2.0, "delta"), (3.5, "delta"),
        (5.0, "theta"), (6.0, "theta"), (7.5, "theta"),
        (9.0, "alpha"), (10.0, "alpha"), (12.0, "alpha"),
        (15.0, "beta"), (20.0, "beta"), (28.0, "beta"),
    ])
    def test_each_band_claims_its_own_frequencies(self, freq: float,
                                                  expected_band: str) -> None:
        w = _windows(((freq, 20.0),), seed=int(freq * 10))
        features = extract_features(w, SpectralConfig())
        power = features.power[:, 0, :].mean(axis=0)
        winner = features.band_names[int(np.argmax(power))]
        assert winner == expected_band, f"{freq} Hz landed in {winner}, expected {expected_band}"

    def test_band_power_recovers_the_analytic_value(self) -> None:
        """A 20 uV sine has power 20^2/2 = 200 uV^2. Integrating the PSD over
        the band containing it must recover that, not a scaled version of it:
        a wrong scaling factor would be invisible in any ratio-based analysis
        but would make absolute microvolt thresholds meaningless."""
        w = _windows(((10.0, 20.0),), duration=60.0)
        features = extract_features(w, SpectralConfig())
        alpha = features.band_series("Fz", "alpha").mean()
        assert alpha == pytest.approx(200.0, rel=0.05)

    def test_two_components_add(self) -> None:
        """6 Hz at 10 uV (50 uV^2 in theta) and 10 Hz at 20 uV (200 uV^2 in
        alpha), independently recovered."""
        w = _windows(((6.0, 10.0), (10.0, 20.0)), duration=60.0, seed=21)
        features = extract_features(w, SpectralConfig())
        assert features.band_series("Fz", "theta").mean() == pytest.approx(50.0, rel=0.08)
        assert features.band_series("Fz", "alpha").mean() == pytest.approx(200.0, rel=0.08)

    def test_white_noise_spreads_across_bands_as_predicted(self) -> None:
        """sigma=10 uV spreads 100 uV^2 over 0-250 Hz, so theta's 4 Hz should
        hold about 100 * 4 / 250 = 1.6 uV^2.

        `noise_band=None` for genuinely white noise: the fixture band-limits by
        default, which changes the denominator from Nyquist to the bandwidth
        (see the next test).
        """
        w = _windows((), duration=120.0, noise=10.0, seed=22, noise_band=None)
        features = extract_features(w, SpectralConfig())
        theta = features.band_series("Fz", "theta").mean()
        assert theta == pytest.approx(100.0 * 4.0 / 250.0, rel=0.25)

    def test_band_limited_noise_concentrates_in_the_passband(self) -> None:
        """The same 100 uV^2 confined to 1-45 Hz puts about 100 * 4 / 44 = 9.1
        uV^2 in theta — roughly 6x the white-noise case. Worth asserting
        because it is the default the other tests run under, and because it is
        why a threshold calibrated against white noise would be miscalibrated.
        """
        w = _windows((), duration=120.0, noise=10.0, seed=22, noise_band=(1.0, 45.0))
        features = extract_features(w, SpectralConfig())
        theta = features.band_series("Fz", "theta").mean()
        assert theta == pytest.approx(100.0 * 4.0 / 44.0, rel=0.25)

    def test_per_channel_content_is_independent(self) -> None:
        """Alpha at Pz only. A channel-indexing bug would show up here as
        alpha appearing at Fz, which is exactly the failure mode that would
        make a frontal-minus-parietal index measure the wrong thing."""
        w = _windows(
            ((6.0, 10.0),), channels=("Fz", "Pz"), duration=60.0, seed=23,
            per_channel={"Pz": ((10.0, 20.0),)})
        features = extract_features(w, SpectralConfig())
        assert features.band_series("Pz", "alpha").mean() > 100.0
        assert features.band_series("Fz", "alpha").mean() < 5.0
        assert features.band_series("Fz", "theta").mean() > 40.0


class TestBandEdges:
    def test_the_edge_bin_belongs_to_the_upper_band_only(self) -> None:
        """A bin at exactly 8 Hz belongs to alpha, not theta.

        Tested analytically on a flat PSD, where the answer is exact.  With
        0.5 Hz bins, half-open [4, 8) selects 4.0 ... 7.5 and the trapezoid
        over a unit-height PSD is 3.5; closed [4, 8] would also take the 8.0
        bin and give 4.0.  Checking the number rather than a sine's leakage
        makes the distinction unambiguous.

        It matters because the workload index's theta term is positive and its
        alpha term negative: a shared bin would make the two partly cancel, in
        a quantity whose sign is the entire point.
        """
        freqs = np.arange(0.0, 50.5, 0.5)
        flat = np.ones((1, 1, freqs.size), dtype=np.float64)
        psd = PSD(freqs=freqs, power=flat, channel_names=("Fz",), sfreq=SFREQ,
                  nperseg=1000, noverlap=500, window="hann", detrend="constant")
        power, names = band_power(psd, (("theta", 4.0, 8.0), ("alpha", 8.0, 13.0)))
        values = dict(zip(names, power[0, 0, :]))
        assert values["theta"] == pytest.approx(3.5)
        assert values["alpha"] == pytest.approx(4.5)

    def test_an_eight_hertz_sine_lands_mostly_in_alpha(self) -> None:
        """The same property on real spectra, as a direction check only.

        A sine sitting exactly on a band boundary leaks across it — a Hann
        taper spreads energy over roughly +/-1.5 bins — so the ratio here is
        single digits, not the 50x of a mid-band sine. Asserting only the
        direction keeps this test about band membership rather than about the
        taper's sidelobes.
        """
        psd = welch_psd(_windows(((8.0, 20.0),), duration=60.0, seed=24),
                        SpectralConfig())
        power, names = band_power(psd, BANDS)
        bands = dict(zip(names, power[:, 0, :].mean(axis=0)))
        assert bands["alpha"] > 3.0 * bands["theta"]

    def test_a_band_with_fewer_than_two_bins_raises(self) -> None:
        """np.trapezoid over one bin returns 0.0 silently, so a misconfigured
        band would be structurally zero in every window and an index built on
        it would still produce an ordering - of nothing."""
        psd = welch_psd(_windows(((10.0, 20.0),)), SpectralConfig())
        with pytest.raises(SpectralError, match="frequency bin"):
            band_power(psd, (("sliver", 10.0, 10.2),))

    def test_custom_bands_are_honoured(self) -> None:
        """D023 (EEG features) is OPEN, so the band set must be a parameter."""
        w = _windows(((10.0, 20.0),), duration=60.0, seed=25)
        custom = SpectralConfig(bands=(("low_alpha", 8.0, 10.0), ("high_alpha", 10.0, 13.0)))
        features = extract_features(w, custom)
        assert features.band_names == ("low_alpha", "high_alpha")


class TestWelchParameters:
    def test_segment_length_sets_the_resolution(self) -> None:
        w = _windows(((10.0, 20.0),))
        assert welch_psd(w, SpectralConfig(welch_segment_seconds=2.0)).freq_resolution == 0.5
        assert welch_psd(w, SpectralConfig(welch_segment_seconds=None)).freq_resolution == 0.25

    def test_whole_window_segment_uses_the_window_length(self) -> None:
        w = _windows(((10.0, 20.0),))
        psd = welch_psd(w, SpectralConfig(welch_segment_seconds=None))
        assert psd.nperseg == w.n_samples

    def test_segment_longer_than_the_window_raises_instead_of_being_clamped(self) -> None:
        """SciPy clamps nperseg with a warning, which would leave the recorded
        config describing a run that did not happen."""
        w = _windows(((10.0, 20.0),), window_seconds=2.0)
        with pytest.raises(SpectralError, match="SciPy would clamp"):
            welch_psd(w, SpectralConfig(welch_segment_seconds=1.9))
            welch_psd(w, SpectralConfig(welch_segment_seconds=4.0))

    def test_averaging_segments_reduces_variance(self) -> None:
        """The documented tradeoff behind welch_segment_seconds=2.0, measured.
        Three averaged segments must give a less variable per-window alpha
        estimate than one unaveraged periodogram."""
        w = _windows((), duration=200.0, noise=10.0, seed=26)
        one = extract_features(w, SpectralConfig(welch_segment_seconds=None))
        three = extract_features(w, SpectralConfig(welch_segment_seconds=2.0))
        cv_one = one.band_series("Fz", "alpha").std() / one.band_series("Fz", "alpha").mean()
        cv_three = three.band_series("Fz", "alpha").std() / three.band_series("Fz", "alpha").mean()
        assert cv_three < cv_one

    def test_psd_records_what_it_used(self) -> None:
        psd = welch_psd(_windows(((10.0, 20.0),)), SpectralConfig(welch_segment_seconds=2.0))
        assert psd.nperseg == 1000
        assert psd.noverlap == 500
        assert psd.window == "hann"
        assert psd.detrend == "constant"


class TestWindowFeatures:
    def test_relative_power_sums_to_one(self) -> None:
        w = _windows(((6.0, 10.0), (10.0, 20.0)), duration=60.0, noise=5.0, seed=27)
        features = extract_features(w, SpectralConfig(relative_power=True))
        assert features.relative is not None
        assert np.allclose(features.relative.sum(axis=-1), 1.0)

    def test_relative_power_absent_when_disabled(self) -> None:
        w = _windows(((10.0, 20.0),))
        features = extract_features(w, SpectralConfig(relative_power=False))
        assert features.relative is None
        with pytest.raises(SpectralError, match="relative power was not computed"):
            features.band_series("Fz", "alpha", relative=True)

    def test_features_are_computed_for_rejected_windows_too(self) -> None:
        """A consumer handed only clean windows could not see the rejection
        rate, and pre-registration §7 requires it per configuration."""
        w = _windows(((10.0, 20.0),), duration=40.0, seed=28)
        clean = np.ones(w.n_windows, dtype=bool)
        clean[0] = False
        features = extract_features(w, SpectralConfig(), clean=clean)
        assert features.n_windows == w.n_windows
        assert features.n_clean == w.n_windows - 1
        assert np.all(np.isfinite(features.power))

    def test_clean_only_selects_and_stays_aligned_with_times(self) -> None:
        w = _windows(((10.0, 20.0),), duration=40.0, seed=29)
        clean = np.ones(w.n_windows, dtype=bool)
        clean[[0, 3]] = False
        features = extract_features(w, SpectralConfig(), clean=clean)
        series = features.band_series("Fz", "alpha", clean_only=True)
        times = features.window_times(clean_only=True)
        assert series.shape == times.shape == (w.n_windows - 2,)
        assert 0.0 not in times.tolist()

    def test_mismatched_clean_length_raises(self) -> None:
        w = _windows(((10.0, 20.0),), duration=40.0, seed=30)
        with pytest.raises(SpectralError, match="clean has shape"):
            extract_features(w, SpectralConfig(), clean=np.ones(3, dtype=bool))

    def test_roi_series_averages_over_channels_within_a_window(self) -> None:
        w = _windows(((10.0, 20.0),), channels=("Fz", "F3", "F4"), duration=60.0, seed=31)
        features = extract_features(w, SpectralConfig())
        roi = features.roi_series(("Fz", "F3", "F4"), "alpha")
        each = np.stack([features.band_series(c, "alpha") for c in ("Fz", "F3", "F4")])
        assert np.allclose(roi, each.mean(axis=0))

    def test_roi_series_refuses_a_partial_cluster_by_default(self) -> None:
        """A 'frontal midline' mean over two of three electrodes is a different
        quantity, and nothing downstream would notice the substitution."""
        w = _windows(((10.0, 20.0),), channels=("Fz", "F3"), duration=40.0, seed=32)
        features = extract_features(w, SpectralConfig())
        with pytest.raises(KeyError, match="require_all=False"):
            features.roi_series(("Fz", "F3", "F4"), "alpha")
        partial = features.roi_series(("Fz", "F3", "F4"), "alpha", require_all=False)
        assert partial.shape == (features.n_windows,)

    def test_unknown_channel_or_band_raises(self) -> None:
        w = _windows(((10.0, 20.0),), duration=40.0, seed=33)
        features = extract_features(w, SpectralConfig())
        with pytest.raises(KeyError, match="not in these features"):
            features.channel_index("Oz")
        with pytest.raises(KeyError, match="band 'gamma' not in"):
            features.band_index("gamma")

    def test_summary_is_json_safe(self) -> None:
        import json
        w = _windows(((10.0, 20.0),), channels=("Fz", "Pz"), duration=40.0, seed=34)
        features = extract_features(w, SpectralConfig())
        json.dumps(features.to_summary(), allow_nan=False)


class TestDiagnostics:
    def test_dominant_frequency_finds_the_peak(self) -> None:
        """The hardware validation gate (§10) wants an alpha *peak*, not just a
        raised band integral: broadband noise raises the integral without a
        peak, and the two mean different things."""
        w = _windows(((10.0, 20.0),), duration=60.0, seed=35)
        psd = welch_psd(w, SpectralConfig(welch_segment_seconds=None))
        peaks = dominant_frequency(psd, 8.0, 13.0)
        assert np.allclose(peaks, 10.0, atol=0.3)

    def test_dominant_frequency_with_no_bins_raises(self) -> None:
        psd = welch_psd(_windows(((10.0, 20.0),)), SpectralConfig())
        with pytest.raises(SpectralError, match="no frequency bins"):
            dominant_frequency(psd, 300.0, 310.0)

    def test_line_noise_ratio_detects_contamination(self) -> None:
        clean = synthetic_recording(("Fz",), duration_seconds=30.0, noise_uv=10.0, seed=36)
        dirty = synthetic_recording(("Fz",), duration_seconds=30.0, noise_uv=10.0, seed=36)
        inject_line_noise(dirty, 60.0, 30.0)
        clean_ratio = float(line_noise_ratio(clean.data, SFREQ, 60.0)[0])
        dirty_ratio = float(line_noise_ratio(dirty.data, SFREQ, 60.0)[0])
        assert clean_ratio < 5.0
        assert dirty_ratio > 50.0 * clean_ratio

    def test_line_noise_ratio_above_nyquist_raises(self) -> None:
        rec = synthetic_recording(("Fz",), sfreq=100.0, duration_seconds=30.0, seed=37)
        with pytest.raises(SpectralError, match="Nyquist"):
            line_noise_ratio(rec.data, 100.0, 60.0)
