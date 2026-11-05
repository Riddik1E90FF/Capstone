"""Tests for eeg/artifacts.py: the per-window clean/reject mask.

Both failure directions are tested, because both are silent and both are
dangerous in the specific way pre-registration §7 warns about:

* **too permissive** — artifact power enters the band powers, and because
  blinks and muscle activity are broadband and frontally dominant they inflate
  frontal theta, the index's primary term;
* **too strict** — the rejection rate climbs where movement is most likely,
  which is plausibly where difficulty is highest, manufacturing exactly the
  confound §7 correlates against the difficulty ranking.

The design principle is also tested directly: the detector must *mark* windows
and never modify a signal.
"""

from __future__ import annotations

import numpy as np
import pytest

from eeg.artifacts import (MAD_TO_SIGMA, REASONS, ArtifactError, detect_artifacts,
                           ocular_reference_windows, rejection_rate_by_group, robust_z)
from eeg.config import ArtifactConfig, EpochConfig, FilterConfig
from eeg.filtering import apply_filters
from eeg.synthetic import (inject_blink, inject_excursion, inject_step,
                           synthetic_recording)
from eeg.windowing import make_windows

SFREQ = 500.0
CHANNELS = ("Fz", "F3", "F4", "Pz", "P3", "P4", "Fp1", "Fp2")


def _clean_recording(duration: float = 120.0, seed: int = 0, channels=CHANNELS):
    """A quiet synthetic recording that the default thresholds should accept."""
    return synthetic_recording(channels, duration_seconds=duration,
                               components=((10.0, 15.0),), noise_uv=5.0, seed=seed)


def _windows(recording, window_seconds: float = 4.0):
    return make_windows(recording, EpochConfig(window_seconds=window_seconds))


def _detect(recording, config: ArtifactConfig | None = None, window_seconds: float = 4.0):
    """Detect with the accurate continuous-ocular path, as the pipeline does."""
    config = config or ArtifactConfig()
    epoch = EpochConfig(window_seconds=window_seconds)
    windows = make_windows(recording, epoch)
    ocular = ocular_reference_windows(recording, epoch, config)
    return detect_artifacts(windows, config, ocular_windows=ocular)


class TestCleanSignalPasses:
    def test_a_quiet_recording_is_almost_entirely_clean(self) -> None:
        """If the default thresholds reject a quiet signal, they would reject
        most of a real session too, and the rejection rate would stop carrying
        information about the participant."""
        report = _detect(_clean_recording())
        assert report.rejection_rate < 0.05, report.summary()

    def test_a_clean_report_fires_no_reason(self) -> None:
        report = _detect(_clean_recording(seed=1))
        rates = report.rate_by_reason()
        for reason in ("nonfinite", "peak_to_peak", "flat", "gradient"):
            assert rates[reason] == 0.0, f"{reason} fired on a clean signal"

    def test_every_reason_key_is_always_present(self) -> None:
        """A fixed reason set means a zero rate never has to be told apart
        from a missing key."""
        report = _detect(_clean_recording(seed=2))
        assert set(report.by_reason) == set(REASONS)
        assert set(report.rate_by_reason()) == set(REASONS)

    def test_mask_shape_and_dtype(self) -> None:
        rec = _clean_recording(duration=40.0, seed=3)
        report = _detect(rec)
        assert report.clean.dtype == bool
        assert report.clean.shape == (10,)
        assert report.n_windows == 10


class TestAmplitude:
    def test_a_large_excursion_is_rejected(self) -> None:
        rec = _clean_recording(seed=4)
        inject_excursion(rec, "Pz", start_seconds=20.0, duration_seconds=1.0,
                         amplitude_uv=400.0)
        report = _detect(rec)
        assert report.by_reason["peak_to_peak"][5]       # window 5 covers 20-24 s
        assert not report.clean[5]

    def test_only_the_affected_window_is_rejected(self) -> None:
        """Rejection must be local. A detector that threw away a whole
        recording for one excursion would discard most real sessions."""
        rec = _clean_recording(seed=5)
        inject_excursion(rec, "Pz", 20.0, 1.0, 400.0)
        report = _detect(rec)
        assert report.n_clean >= report.n_windows - 2

    def test_the_threshold_is_actually_the_threshold(self) -> None:
        """An excursion just under the configured ceiling passes; just over it
        does not. This is what makes peak_to_peak_uv a real parameter rather
        than a decorative one."""
        for amplitude, expect_reject in ((40.0, False), (300.0, True)):
            rec = synthetic_recording(CHANNELS, duration_seconds=40.0,
                                      components=((10.0, 5.0),), noise_uv=1.0, seed=6)
            inject_excursion(rec, "Pz", 20.0, 1.0, amplitude)
            report = _detect(rec, ArtifactConfig(peak_to_peak_uv=150.0))
            assert bool(report.by_reason["peak_to_peak"][5]) is expect_reject

    def test_raising_the_threshold_clears_the_amplitude_reason(self) -> None:
        """Asserted on the reason rather than on `clean`, because a rectangular
        excursion also has two single-sample 400 uV edges and so trips the
        gradient check independently. That overlap is the behaviour
        `rate_by_reason` exists to make visible."""
        rec = _clean_recording(seed=7)
        inject_excursion(rec, "Pz", 20.0, 1.0, 400.0)
        strict = _detect(rec, ArtifactConfig(peak_to_peak_uv=150.0))
        lax = _detect(rec, ArtifactConfig(peak_to_peak_uv=1000.0))
        assert strict.by_reason["peak_to_peak"][5]
        assert not lax.by_reason["peak_to_peak"][5]
        assert lax.by_reason["gradient"][5]

    def test_a_smooth_excursion_trips_amplitude_alone(self) -> None:
        """A blink-shaped bump has no step edge, so only the amplitude check
        fires; raising the ceiling then clears the window entirely."""
        rec = _clean_recording(seed=37)
        inject_blink(rec, ("Pz",), at_seconds=22.0, amplitude_uv=400.0,
                     width_seconds=0.4)
        strict = _detect(rec, ArtifactConfig(peak_to_peak_uv=150.0))
        lax = _detect(rec, ArtifactConfig(peak_to_peak_uv=1000.0))
        assert not strict.clean[5]
        assert lax.clean[5]


class TestFlatSignal:
    def test_a_dead_channel_is_rejected_not_treated_as_the_cleanest(self) -> None:
        """A disconnected electrode is the quietest channel in a recording, so
        it passes the amplitude ceiling. Without the flat floor it would be the
        cleanest channel in every window."""
        rec = _clean_recording(duration=40.0, seed=8)
        rec.data[rec.index_of("P3")] = 0.0
        report = _detect(rec)
        assert report.by_reason["flat"].all()
        assert report.n_clean == 0

    def test_the_dead_channel_is_named(self) -> None:
        """What distinguishes a bad electrode from a restless participant:
        without per-channel rates, the only visible fact is that everything
        was rejected."""
        rec = _clean_recording(duration=40.0, seed=9)
        rec.data[rec.index_of("P3")] = 0.0
        report = _detect(rec)
        rates = report.rate_by_channel()
        assert rates["P3"] == 1.0
        assert all(v == 0.0 for k, v in rates.items() if k != "P3")


class TestGradient:
    def test_a_single_sample_step_is_caught(self) -> None:
        """A one-sample 60 uV jump barely moves the window's peak-to-peak
        range, which is precisely why the gradient check exists separately."""
        rec = synthetic_recording(CHANNELS, duration_seconds=40.0,
                                  components=((10.0, 5.0),), noise_uv=1.0, seed=10)
        inject_step(rec, "Fz", at_seconds=22.0, amplitude_uv=60.0)
        report = _detect(rec, ArtifactConfig(max_gradient_uv_per_sample=25.0))
        assert report.by_reason["gradient"][5]
        assert not report.by_reason["peak_to_peak"][5]

    def test_ordinary_eeg_does_not_trip_the_gradient_check(self) -> None:
        """At 500 Hz, 25 uV between adjacent samples is 12.5 mV/s - far faster
        than anything inside a 1-45 Hz passband."""
        report = _detect(_clean_recording(seed=11))
        assert report.rate_by_reason()["gradient"] == 0.0


class TestNonFinite:
    def test_a_nan_rejects_its_window(self) -> None:
        rec = _clean_recording(duration=40.0, seed=12)
        rec.data[0, 10_500] = np.nan
        report = _detect(rec)
        assert report.by_reason["nonfinite"][5]
        assert not report.clean[5]

    def test_an_infinity_rejects_its_window(self) -> None:
        rec = _clean_recording(duration=40.0, seed=13)
        rec.data[0, 10_500] = np.inf
        report = _detect(rec)
        assert report.by_reason["nonfinite"][5]

    def test_a_nan_does_not_poison_the_other_checks(self) -> None:
        """One NaN must not turn every other channel's verdict into NaN, which
        would hide which channel was actually broken."""
        rec = _clean_recording(duration=40.0, seed=14)
        rec.data[0, 10_500] = np.nan
        report = _detect(rec)
        assert report.clean[0]
        assert report.rate_by_channel()["Fz"] > 0.0
        assert report.rate_by_channel()["Pz"] == 0.0


class TestOcular:
    def test_a_blink_is_flagged(self) -> None:
        rec = _clean_recording(seed=15)
        inject_blink(rec, ("Fp1", "Fp2"), at_seconds=22.0, amplitude_uv=250.0)
        report = _detect(rec)
        assert report.by_reason["ocular"][5], report.summary()
        assert not report.clean[5]

    def test_a_blink_too_small_for_the_amplitude_ceiling_is_still_caught(self) -> None:
        """The reason the ocular detector exists at all: the 1 Hz high-pass
        attenuates a blink's slow component, so a moderate blink can pass the
        peak-to-peak ceiling while still contaminating the low bands."""
        rec = synthetic_recording(CHANNELS, duration_seconds=120.0,
                                  components=((10.0, 15.0),), noise_uv=5.0, seed=16)
        inject_blink(rec, ("Fp1", "Fp2"), at_seconds=22.0, amplitude_uv=90.0)
        filtered = apply_filters(rec, FilterConfig(edge_trim_seconds=0.0))
        report = _detect(filtered, ArtifactConfig(peak_to_peak_uv=150.0))
        assert not report.by_reason["peak_to_peak"][5]
        assert report.by_reason["ocular"][5]

    def test_no_blink_means_no_ocular_rejection(self) -> None:
        report = _detect(_clean_recording(seed=17))
        assert report.rate_by_reason()["ocular"] == 0.0

    def test_raising_the_z_threshold_lets_the_blink_through(self) -> None:
        rec = _clean_recording(seed=18)
        inject_blink(rec, ("Fp1", "Fp2"), 22.0, 250.0)
        assert _detect(rec, ArtifactConfig(ocular_z_threshold=5.0)).by_reason["ocular"][5]
        assert not _detect(rec, ArtifactConfig(ocular_z_threshold=500.0)).by_reason["ocular"][5]

    def test_ocular_z_is_kept_so_a_threshold_can_be_revisited(self) -> None:
        rec = _clean_recording(duration=40.0, seed=19)
        inject_blink(rec, ("Fp1", "Fp2"), 22.0, 250.0)
        report = _detect(rec)
        assert report.ocular_z.shape == (10, 2)
        assert report.ocular_z[5].max() > 5.0

    def test_missing_ocular_channels_raise_by_default(self) -> None:
        """A mask with no ocular component looks identical in the output to one
        that has it. The author's 8-channel layout may not include Fp1/Fp2 at
        all (D022 OPEN), so this must be loud."""
        rec = _clean_recording(duration=40.0, seed=20, channels=("Fz", "F3", "F4", "Pz"))
        with pytest.raises(ArtifactError, match="none of the ocular channels"):
            _detect(rec)

    def test_missing_ocular_channels_can_be_waived_with_the_degradation_recorded(self) -> None:
        rec = _clean_recording(duration=40.0, seed=21, channels=("Fz", "F3", "F4", "Pz"))
        report = _detect(rec, ArtifactConfig(require_ocular_channels=False))
        assert report.ocular_channels == ()
        assert any("no ocular component" in w for w in report.warnings)

    def test_per_window_ocular_filtering_warns_about_edge_transients(self) -> None:
        """The inaccurate path must say so rather than be silently equivalent."""
        rec = _clean_recording(duration=40.0, seed=22)
        windows = _windows(rec)
        report = detect_artifacts(windows, ArtifactConfig())      # no ocular_windows
        assert any("per window rather than from the continuous" in w for w in report.warnings)

    def test_mismatched_ocular_window_count_is_an_error(self) -> None:
        rec = _clean_recording(duration=40.0, seed=23)
        windows = _windows(rec, window_seconds=4.0)
        other = _windows(rec, window_seconds=2.0)
        with pytest.raises(ArtifactError, match="same segmentation"):
            detect_artifacts(windows, ArtifactConfig(), ocular_windows=other)

    def test_few_windows_warns_that_the_robust_statistic_is_unstable(self) -> None:
        """EEGMAT's 60 s task recordings give 15 windows, under the 20 this
        warning triggers at. The warning is how that reaches a reader."""
        rec = _clean_recording(duration=40.0, seed=24)
        report = _detect(rec)
        assert report.n_windows == 10
        assert any("too few observations" in w for w in report.warnings)


class TestRobustZ:
    def test_matches_the_definition(self) -> None:
        values = np.array([[1.0], [2.0], [3.0], [4.0], [100.0]])
        z, median, sigma = robust_z(values)
        assert median[0] == pytest.approx(3.0)
        assert sigma[0] == pytest.approx(MAD_TO_SIGMA * 1.0)
        assert z[4, 0] == pytest.approx((100.0 - 3.0) / (MAD_TO_SIGMA * 1.0))

    def test_an_outlier_does_not_drag_the_centre(self) -> None:
        """The reason for median/MAD rather than mean/SD: a few large blinks
        would raise a mean and an SD enough to hide themselves."""
        values = np.array([[1.0], [1.0], [1.0], [1.0], [1000.0]])
        _, median, _ = robust_z(values)
        assert median[0] == pytest.approx(1.0)

    def test_zero_spread_gives_zeros_not_infinities(self) -> None:
        z, _, sigma = robust_z(np.ones((5, 2)))
        assert sigma.tolist() == [0.0, 0.0]
        assert np.all(z == 0.0)


class TestCombination:
    def test_any_channel_failing_rejects_the_window_by_default(self) -> None:
        rec = _clean_recording(duration=40.0, seed=25)
        inject_excursion(rec, "P3", 20.0, 1.0, 400.0)
        assert not _detect(rec, ArtifactConfig(reject_if_any_channel=True)).clean[5]

    def test_per_channel_mode_keeps_a_window_with_some_good_channels(self) -> None:
        """Kept as an option, with the documented consequence: different
        channels then contribute different window sets, so a
        frontal-minus-parietal index would combine terms computed over
        non-identical samples."""
        rec = _clean_recording(duration=40.0, seed=26)
        inject_excursion(rec, "P3", 20.0, 1.0, 400.0)
        report = _detect(rec, ArtifactConfig(reject_if_any_channel=False))
        assert report.clean[5]
        assert report.by_reason_channel["peak_to_peak"][5, report.examined_channels.index("P3")]

    def test_examined_channels_can_be_restricted(self) -> None:
        rec = _clean_recording(duration=40.0, seed=27)
        inject_excursion(rec, "P3", 20.0, 1.0, 400.0)
        restricted = _detect(rec, ArtifactConfig(channels=("Fz", "F3", "F4")))
        assert restricted.examined_channels == ("Fz", "F3", "F4")
        assert restricted.clean[5]

    def test_requesting_absent_channels_warns(self) -> None:
        rec = _clean_recording(duration=40.0, seed=28)
        report = _detect(rec, ArtifactConfig(channels=("Fz", "Oz")))
        assert any("went unchecked" in w for w in report.warnings)

    def test_no_examinable_channel_is_an_error(self) -> None:
        rec = _clean_recording(duration=40.0, seed=29)
        windows = _windows(rec)
        with pytest.raises(ArtifactError, match="no channels to examine"):
            detect_artifacts(windows, ArtifactConfig(channels=("Oz", "O1")))


class TestMarksRatherThanModifies:
    def test_the_signal_is_never_altered(self) -> None:
        """The design principle, asserted: robustness, not suppression.
        Nothing in this module may modify, interpolate or reconstruct."""
        rec = _clean_recording(duration=40.0, seed=30)
        inject_blink(rec, ("Fp1", "Fp2"), 22.0, 250.0)
        inject_excursion(rec, "P3", 10.0, 1.0, 400.0)
        windows = _windows(rec)
        before = windows.data.copy()
        ocular = ocular_reference_windows(rec, EpochConfig(), ArtifactConfig())
        detect_artifacts(windows, ArtifactConfig(), ocular_windows=ocular)
        assert np.array_equal(windows.data, before)

    def test_detection_is_deterministic(self) -> None:
        rec = _clean_recording(duration=40.0, seed=31)
        inject_blink(rec, ("Fp1", "Fp2"), 22.0, 250.0)
        a = _detect(rec)
        b = _detect(rec)
        assert np.array_equal(a.clean, b.clean)
        assert np.array_equal(a.ocular_z, b.ocular_z)


class TestRejectionRateByGroup:
    def test_counts_per_label(self) -> None:
        """Pre-registration §7 correlates per-configuration rejection rates
        against the difficulty ranking, as a confound check. This does the
        counting; the correlation is the author's to write."""
        rec = _clean_recording(duration=40.0, seed=32)
        inject_excursion(rec, "P3", 4.0, 1.0, 400.0)      # window 1 -> group 0
        report = _detect(rec)
        groups = np.array([0] * 5 + [1] * 5)
        rates = rejection_rate_by_group(report, groups)
        assert rates[0] == pytest.approx(0.2)
        assert rates[1] == pytest.approx(0.0)

    def test_mismatched_length_is_an_error(self) -> None:
        report = _detect(_clean_recording(duration=40.0, seed=33))
        with pytest.raises(ArtifactError, match="entries but the report has"):
            rejection_rate_by_group(report, np.zeros(3))

    def test_string_labels_work(self) -> None:
        report = _detect(_clean_recording(duration=40.0, seed=34))
        rates = rejection_rate_by_group(report, np.array(["a"] * 5 + ["b"] * 5))
        assert set(rates) == {"a", "b"}


class TestSummary:
    def test_summary_is_json_safe(self) -> None:
        import json
        rec = _clean_recording(duration=40.0, seed=35)
        inject_blink(rec, ("Fp1", "Fp2"), 22.0, 250.0)
        json.dumps(_detect(rec).summary(), allow_nan=False)

    def test_reason_rates_may_overlap_and_need_not_sum(self) -> None:
        """One window can be both high-amplitude and ocular. A reason that
        only ever fires alongside another adds nothing, and that is only
        visible if the rates are kept separate."""
        rec = _clean_recording(duration=40.0, seed=36)
        inject_blink(rec, ("Fp1", "Fp2"), 22.0, 600.0)
        report = _detect(rec)
        rates = report.rate_by_reason()
        assert rates["ocular"] > 0.0
        assert rates["peak_to_peak"] > 0.0
        assert sum(rates.values()) > report.rejection_rate
