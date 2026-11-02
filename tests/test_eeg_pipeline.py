"""Tests for eeg/pipeline.py: composition, reproducibility, and the index boundary.

Three groups matter most.

**Reproducibility.**  The pipeline has no randomness, so equal config plus
equal input must give bit-identical band powers.  That is asserted rather than
assumed — it is the property that makes a saved `PipelineResult` a record of
what happened rather than a report of one run among many.

**The index boundary.**  The workload index is not implemented in this package
(pre-registration §7; CLAUDE.md assigns it to the author).  `apply_index` is
the boundary, and its job is to catch a contract violation *at* the boundary
rather than let a misaligned array produce a plausible correlation three
stages later.  Each violation is tested.

**EEGMAT integration.**  Including the Berger-effect positive control from
D032: eyes-closed to eyes-open must suppress alpha.  These skip when the data
is absent, because `data/` is gitignored and a fresh checkout has none.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eeg.channels import ROI_OCCIPITAL, ROI_PARIETAL
from eeg.config import (ArtifactConfig, EpochConfig, FilterConfig, PipelineConfig, SpectralConfig)
from eeg.pipeline import (RESULT_SCHEMA_VERSION, EEGPipeline, PipelineError, WorkloadIndex,
                          apply_index, default_pipeline)
from eeg.spectral import WindowFeatures
from eeg.synthetic import inject_blink, inject_excursion, synthetic_recording

SFREQ = 500.0
CHANNELS = ("Fz", "F3", "F4", "Pz", "P3", "P4", "Fp1", "Fp2")

REPO = Path(__file__).resolve().parents[1]
EEGMAT_DIR = REPO / "data" / "raw" / "eegmat"

needs_eegmat = pytest.mark.skipif(
    not (EEGMAT_DIR / "Subject00_1.edf").exists(),
    reason="EEGMAT not downloaded; run scripts/eeg_demo.py --download")


def _recording(duration: float = 120.0, seed: int = 0, channels=CHANNELS,
               components=((10.0, 15.0),)):
    return synthetic_recording(channels, duration_seconds=duration,
                               components=components, noise_uv=5.0, seed=seed)


def _pipeline(**overrides) -> EEGPipeline:
    return EEGPipeline(PipelineConfig(**overrides))


class TestComposition:
    def test_a_run_produces_features_and_a_mask_of_matching_length(self) -> None:
        result = _pipeline().run(_recording())
        assert result.features.n_windows == result.artifacts.n_windows
        assert result.features.clean.shape == result.artifacts.clean.shape
        assert np.array_equal(result.features.clean, result.artifacts.clean)

    def test_window_count_follows_from_duration_trim_and_window_length(self) -> None:
        """120 s, 1 s trimmed at each end, 4 s windows: 118 / 4 = 29 whole
        windows. Pinned because the count is what every per-configuration
        median rests on."""
        result = _pipeline().run(_recording(duration=120.0))
        assert result.features.n_windows == 29

    def test_feature_shape_is_windows_channels_bands(self) -> None:
        result = _pipeline().run(_recording())
        assert result.features.power.shape == (29, len(CHANNELS), 4)
        assert result.features.band_names == ("delta", "theta", "alpha", "beta")

    def test_every_stage_is_recorded_in_order(self) -> None:
        result = _pipeline().run(_recording())
        assert [s.name for s in result.stages] == [
            "load", "filter", "window", "artifact", "spectral"]

    def test_each_stage_records_its_parameters(self) -> None:
        """A result must be criticisable from its own contents."""
        result = _pipeline().run(_recording())
        by_name = {s.name: s for s in result.stages}
        assert by_name["filter"].params["bandpass_high"] == 45.0
        assert by_name["epoch" if "epoch" in by_name else "window"].params[
            "window_seconds"] == 4.0
        assert by_name["artifact"].params["peak_to_peak_uv"] == 150.0
        assert by_name["spectral"].params["welch_overlap"] == 0.5

    def test_the_ocular_reference_comes_from_the_continuous_signal(self) -> None:
        """The accurate path. The per-window alternative leaves a filter
        transient at each window edge."""
        result = _pipeline().run(_recording())
        artifact_stage = next(s for s in result.stages if s.name == "artifact")
        assert artifact_stage.output["ocular_source"] == "continuous"
        assert not any("per window rather than" in w for w in result.warnings)

    def test_window_times_are_convertible_to_original_recording_time(self) -> None:
        """Edge trimming shifts every window. Without the offset a window
        could not be attributed to the configuration block it was recorded
        in, which pre-registration §7's per-configuration rates require."""
        result = _pipeline().run(_recording())
        assert result.features.start_seconds[0] == 0.0
        assert result.features.time_offset == 1.0
        assert result.features.window_times(absolute=True)[0] == 1.0

    def test_a_stage_failure_names_the_stage(self) -> None:
        """Underlying exceptions are accurate but do not say where in a
        five-stage run they came from."""
        pipe = _pipeline(epoch=EpochConfig(window_seconds=4.0))
        with pytest.raises(PipelineError, match="window stage failed"):
            pipe.run(_recording(duration=3.0))

    def test_run_array_defaults_to_microvolts(self) -> None:
        """Opposite of load_edf's volts: a synthetic signal is built in the
        unit its author chose, and provenance records which was assumed."""
        data = np.zeros((2, int(60 * SFREQ)))
        rng = np.random.default_rng(0)
        data += rng.normal(0.0, 5.0, data.shape)
        result = _pipeline(artifact=ArtifactConfig(require_ocular_channels=False)).run_array(
            data, ["Fz", "Pz"], SFREQ)
        assert result.provenance.scale_factor == 1.0


class TestReproducibility:
    def test_two_runs_of_one_pipeline_are_bit_identical(self) -> None:
        pipe = _pipeline()
        rec = _recording(seed=1)
        a, b = pipe.run(rec), pipe.run(rec)
        assert np.array_equal(a.features.power, b.features.power)
        assert np.array_equal(a.artifacts.clean, b.artifacts.clean)
        assert np.array_equal(a.artifacts.ocular_z, b.artifacts.ocular_z)

    def test_two_pipelines_with_equal_configs_agree_bit_for_bit(self) -> None:
        rec = _recording(seed=2)
        a = EEGPipeline(PipelineConfig()).run(rec)
        b = EEGPipeline(PipelineConfig()).run(rec)
        assert np.array_equal(a.features.power, b.features.power)

    def test_the_same_seed_gives_the_same_synthetic_input(self) -> None:
        """PCG64's stream is stable across NumPy versions by documented
        policy, which is what let Gate 1a be re-proved across a major version
        bump (D033). The same property is what makes these tests portable."""
        assert np.array_equal(_recording(seed=3).data, _recording(seed=3).data)
        assert not np.array_equal(_recording(seed=3).data, _recording(seed=4).data)

    def test_timing_is_the_only_nondeterministic_field(self) -> None:
        pipe = _pipeline()
        rec = _recording(seed=5)
        a, b = pipe.run(rec).to_dict(), pipe.run(rec).to_dict()
        a.pop("timing_seconds")
        b.pop("timing_seconds")
        a["provenance"].pop("loaded_utc")
        b["provenance"].pop("loaded_utc")
        assert a == b

    def test_changing_one_parameter_changes_the_output(self) -> None:
        """Otherwise a parameter is decorative and a config diff would be
        meaningless."""
        rec = _recording(seed=6)
        base = _pipeline().run(rec)
        narrower = EEGPipeline(PipelineConfig(
            filter=FilterConfig(bandpass_low=4.0),
            spectral=SpectralConfig(bands=(("theta", 4.0, 8.0), ("alpha", 8.0, 13.0))),
            artifact=ArtifactConfig(ocular_band=(4.0, 6.0)))).run(rec)
        assert base.features.power.shape != narrower.features.power.shape

    def test_one_pipeline_instance_processes_many_recordings_identically(self) -> None:
        """The point of holding the config on the object: a per-subject loop
        that rebuilt its parameters could drift between subjects with nothing
        in the output showing it."""
        pipe = _pipeline()
        results = [pipe.run(_recording(seed=s)) for s in range(3)]
        assert all(r.config == pipe.config for r in results)
        assert len({r.config for r in results}) == 1


class TestArtifactIntegration:
    def test_an_injected_blink_is_rejected_end_to_end(self) -> None:
        rec = _recording(seed=7)
        inject_blink(rec, ("Fp1", "Fp2"), at_seconds=22.0, amplitude_uv=250.0)
        result = _pipeline().run(rec)
        assert result.artifacts.rejection_rate > 0.0
        assert result.features.n_clean < result.features.n_windows

    def test_a_clean_recording_is_almost_entirely_clean(self) -> None:
        result = _pipeline().run(_recording(seed=8))
        assert result.artifacts.rejection_rate < 0.05, result.quality_summary()

    def test_features_exist_for_rejected_windows_too(self) -> None:
        """A consumer handed only clean windows could not see the rejection
        rate, which pre-registration §7 requires per configuration."""
        rec = _recording(seed=9)
        inject_excursion(rec, "Pz", 20.0, 1.0, 600.0)
        result = _pipeline().run(rec)
        assert result.features.n_clean < result.features.n_windows
        assert np.all(np.isfinite(result.features.power))

    def test_a_missing_ocular_channel_is_fatal_by_default(self) -> None:
        """The author's expected 8-channel layout may not include Fp1/Fp2 at
        all (D022 OPEN), and a mask silently missing its ocular component
        looks identical in the output to one that has it."""
        rec = _recording(seed=10, channels=("Fz", "F3", "F4", "Pz"))
        with pytest.raises(PipelineError, match="ocular"):
            _pipeline().run(rec)

    def test_the_waiver_records_the_degradation(self) -> None:
        rec = _recording(seed=11, channels=("Fz", "F3", "F4", "Pz"))
        result = _pipeline(artifact=ArtifactConfig(require_ocular_channels=False)).run(rec)
        assert any("no ocular component" in w for w in result.warnings)


class TestQualitySummary:
    def test_reports_median_peak_to_peak_against_the_threshold(self) -> None:
        """Seeing the median next to the ceiling shows how much headroom a
        recording had: a channel sitting just under the ceiling is one bad
        minute from wholesale rejection."""
        summary = _pipeline().run(_recording(seed=12)).quality_summary()
        assert summary["peak_to_peak_threshold_uv"] == 150.0
        for channel, stats in summary["per_channel"].items():
            assert 0.0 < stats["median_ptp_uv"] < 150.0, channel

    def test_names_dropped_channels_and_the_reference_applied(self) -> None:
        summary = _pipeline().run(_recording(seed=13)).quality_summary()
        assert summary["reference_applied"].startswith("none")
        assert "channels" in summary and len(summary["channels"]) == len(CHANNELS)

    def test_is_json_safe(self) -> None:
        summary = _pipeline().run(_recording(seed=14)).quality_summary()
        json.dumps(summary, allow_nan=False)


class TestSerialisation:
    def test_to_dict_carries_config_provenance_stages_and_quality(self) -> None:
        out = _pipeline().run(_recording(seed=15)).to_dict()
        assert out["schema_version"] == RESULT_SCHEMA_VERSION
        for key in ("config", "provenance", "stages", "artifacts", "features", "quality"):
            assert key in out

    def test_the_recorded_config_round_trips(self) -> None:
        """A saved result must be able to reconstruct the pipeline that made
        it, or it is not a reproducible record."""
        result = _pipeline().run(_recording(seed=16))
        assert PipelineConfig.from_dict(result.to_dict()["config"]) == result.config

    def test_save_writes_json_and_npz(self, tmp_path: Path) -> None:
        result = _pipeline().run(_recording(seed=17))
        paths = result.save(tmp_path, name="run")
        assert paths["json"].exists() and paths["npz"].exists()
        loaded = np.load(paths["npz"], allow_pickle=False)
        assert np.array_equal(loaded["band_power_uv2"], result.features.power)
        assert np.array_equal(loaded["clean"], result.features.clean)
        assert list(loaded["channel_names"]) == list(result.features.channel_names)

    def test_saved_arrays_allow_a_threshold_to_be_revisited_without_rerunning(
            self, tmp_path: Path) -> None:
        """Band powers do not change when the index or the threshold does."""
        result = _pipeline().run(_recording(seed=18))
        paths = result.save(tmp_path)
        loaded = np.load(paths["npz"], allow_pickle=False)
        assert loaded["ocular_z"].shape[0] == result.features.n_windows

    def test_arrays_can_be_inlined_when_asked(self) -> None:
        out = _pipeline().run(_recording(seed=19)).to_dict(include_arrays=True)
        assert len(out["arrays"]["band_power_uv2"]) == 29


class TestIndexBoundary:
    """The workload index is NOT implemented here (§7; CLAUDE.md). These test
    the boundary's guards, using deliberately trivial stand-ins that are not
    the index and make no scientific claim."""

    @staticmethod
    def _stub(features: WindowFeatures) -> np.ndarray:
        return np.zeros(features.n_windows, dtype=np.float64)

    def test_a_conforming_index_passes_through(self) -> None:
        result = _pipeline().run(_recording(seed=20))
        values = apply_index(result, self._stub)
        assert values.shape == (result.features.n_windows,)

    def test_the_protocol_is_satisfied_by_a_plain_callable(self) -> None:
        assert isinstance(self._stub, WorkloadIndex)

    def test_returning_only_clean_windows_is_rejected(self) -> None:
        """The most dangerous violation: a short array misaligns every value
        against the configuration block it came from, and the medians would
        still compute."""
        rec = _recording(seed=21)
        inject_blink(rec, ("Fp1", "Fp2"), 22.0, 250.0)
        result = _pipeline().run(rec)
        assert result.features.n_clean < result.features.n_windows

        def clean_only(features: WindowFeatures) -> np.ndarray:
            return np.zeros(features.n_clean, dtype=np.float64)

        with pytest.raises(PipelineError, match="one value per window"):
            apply_index(result, clean_only)

    def test_an_integer_index_is_rejected(self) -> None:
        result = _pipeline().run(_recording(seed=22))
        with pytest.raises(PipelineError, match="integer dtype would floor"):
            apply_index(result, lambda f: np.zeros(f.n_windows, dtype=np.int64))

    def test_a_nan_at_a_clean_window_is_rejected(self) -> None:
        result = _pipeline().run(_recording(seed=23))

        def broken(features: WindowFeatures) -> np.ndarray:
            out = np.zeros(features.n_windows, dtype=np.float64)
            out[np.flatnonzero(features.clean)[0]] = np.nan
            return out

        with pytest.raises(PipelineError, match="non-finite values at clean window"):
            apply_index(result, broken)

    def test_a_nan_at_a_rejected_window_is_allowed(self) -> None:
        """The mask says those windows are not to be used, so NaN is the
        clearest thing an index can put there."""
        rec = _recording(seed=24)
        inject_blink(rec, ("Fp1", "Fp2"), 22.0, 250.0)
        result = _pipeline().run(rec)

        def nan_on_rejected(features: WindowFeatures) -> np.ndarray:
            out = np.zeros(features.n_windows, dtype=np.float64)
            out[~features.clean] = np.nan
            return out

        values = apply_index(result, nan_on_rejected)
        assert np.all(np.isfinite(values[result.features.clean]))
        assert np.all(np.isnan(values[~result.features.clean]))

    def test_the_features_the_index_needs_are_reachable(self) -> None:
        """A contract test, not a test of the index: §7's terms are frontal
        midline theta and parietal alpha, and both must be addressable through
        the documented accessors."""
        result = _pipeline().run(_recording(seed=25))
        features = result.features
        theta = features.roi_series(("Fz", "F3", "F4"), "theta")
        alpha = features.roi_series(("Pz", "P3", "P4"), "alpha")
        assert theta.shape == alpha.shape == (features.n_windows,)
        assert np.all(theta > 0) and np.all(alpha > 0)

    def test_apply_index_computes_no_statistic_of_its_own(self) -> None:
        """Medians, ranks and correlations are the analysis (§6, §9) and are
        the author's to write. apply_index returns the raw array."""
        result = _pipeline().run(_recording(seed=26))
        values = apply_index(result, lambda f: np.arange(f.n_windows, dtype=np.float64))
        assert np.array_equal(values, np.arange(result.features.n_windows))


class TestPresets:
    def test_eegmat_preset_is_named_and_noted(self) -> None:
        pipe = default_pipeline("eegmat")
        assert "EEGMAT" in pipe.config.note

    def test_default_preset_has_no_note(self) -> None:
        assert default_pipeline("default").config.note == ""

    def test_an_unknown_preset_raises_rather_than_falling_back(self) -> None:
        """No fallback on purpose: the presets differ in mains frequency, and
        guessing wrong leaves the line artifact in place while reporting a
        notch that ran."""
        with pytest.raises(PipelineError, match="no fallback"):
            default_pipeline("openbci")


@needs_eegmat
class TestEegmatIntegration:
    """Real-data integration. D032 retains EEGMAT as the development fixture
    and **voids its rest/task contrast as a workload measure** — rest is
    eyes-closed, task is eyes-open, so the contrast confounds workload with
    eye state, arousal and visual input. Nothing here claims a workload
    effect."""

    @staticmethod
    def _run(subject: int, which: int):
        pipe = default_pipeline("eegmat")
        return pipe.run_edf(EEGMAT_DIR / f"Subject{subject:02d}_{which}.edf")

    def test_rest_recording_yields_the_expected_window_count(self) -> None:
        """182 s, 1 s trimmed each end, 4 s windows: 180 / 4 = 45."""
        result = self._run(0, 1)
        assert result.features.n_windows == 45

    def test_task_recording_yields_fifteen_windows(self) -> None:
        """62 s, trimmed to 60, 4 s windows. Under the 20 at which the ocular
        detector's robust statistics become unstable, so it must warn."""
        result = self._run(0, 2)
        assert result.features.n_windows == 15
        assert any("too few observations" in w for w in result.artifacts.warnings)

    def test_rejection_rate_is_neither_zero_nor_total(self) -> None:
        """Both extremes would mean the detector carries no information: zero
        says the thresholds are inert, total says they are manufacturing the
        confound §7 watches for."""
        result = self._run(0, 1)
        assert 0.0 < result.artifacts.rejection_rate < 0.5, result.quality_summary()

    def test_amplitudes_are_plausible_scalp_microvolts(self) -> None:
        summary = self._run(0, 1).quality_summary()
        for channel, stats in summary["per_channel"].items():
            assert 5.0 < stats["median_ptp_uv"] < 150.0, (channel, stats)

    def test_sub_thirty_hertz_power_dominates(self) -> None:
        """D032 question 1, re-checked through this pipeline.

        PhysioNet's EEGMAT page claims "a high-pass filter with a 30 Hz
        cut-off frequency", which taken literally would delete all theta and
        alpha and make the dataset useless for a theta/alpha index.  D032
        established it is a typo for *low-pass*, by 41.3x.

        The comparison is below 30 Hz against 30-45 Hz, matching what D032
        measured.  (Comparing delta+theta+alpha against beta alone would be a
        different and much weaker claim — beta is a 17 Hz-wide band of real
        EEG, so that ratio is around 4x and says nothing about the filter.)
        """
        pipe = EEGPipeline(PipelineConfig.for_eegmat(spectral=SpectralConfig(
            bands=(("below_30", 1.0, 30.0), ("above_30", 30.0, 45.0)))))
        features = pipe.run_edf(EEGMAT_DIR / "Subject00_1.edf").features
        below = float(np.median(features.band_series("Fz", "below_30", clean_only=True)))
        above = float(np.median(features.band_series("Fz", "above_30", clean_only=True)))
        assert below > 5.0 * above, (below, above)

    def test_berger_effect_alpha_falls_from_eyes_closed_to_eyes_open(self) -> None:
        """The one legitimate check EEGMAT supports (D032).

        Eyes-closed to eyes-open suppresses alpha (Berger, 1929). It validates
        the loading, filtering, windowing and PSD code against real physiology
        **without claiming anything about workload** — the contrast is
        confounded for workload and is void for that purpose.

        Checked at parietal and occipital sites, where alpha is strongest, and
        across all three locally available subjects. Measured: alpha fell in
        3/3 subjects at both site groups, mean -22.5% parietal and -21.3%
        occipital.
        """
        subjects = sorted({int(p.name[7:9]) for p in EEGMAT_DIR.glob("Subject*_1.edf")
                           if (EEGMAT_DIR / f"{p.name[:9]}_2.edf").exists()})
        assert subjects, "no complete rest/task subject pairs found"

        for group in (ROI_PARIETAL, ROI_OCCIPITAL):
            fell = 0
            for subject in subjects:
                rest = self._run(subject, 1).features.roi_series(
                    group, "alpha", clean_only=True, require_all=False)
                task = self._run(subject, 2).features.roi_series(
                    group, "alpha", clean_only=True, require_all=False)
                if float(np.median(task)) < float(np.median(rest)):
                    fell += 1
            assert fell > len(subjects) / 2, (
                f"alpha fell in only {fell}/{len(subjects)} subjects at {group}. "
                "Alpha must fall from eyes-closed to eyes-open. Suspect the loading, "
                "channel selection, filtering or PSD code before suspecting the data.")

    def test_the_index_channels_survive_the_whole_pipeline(self) -> None:
        features = self._run(0, 1).features
        for site in ("Fz", "F3", "F4", "Pz", "P3", "P4"):
            assert site in features.channel_names

    def test_a_result_saves_and_reloads(self, tmp_path: Path) -> None:
        result = self._run(0, 2)
        paths = result.save(tmp_path, name="subject00_task")
        reloaded = json.loads(paths["json"].read_text())
        assert reloaded["provenance"]["source_sha256"] == result.provenance.source_sha256
        assert PipelineConfig.from_dict(reloaded["config"]) == result.config

    def test_real_data_is_also_bit_reproducible(self) -> None:
        a, b = self._run(1, 2), self._run(1, 2)
        assert np.array_equal(a.features.power, b.features.power)
        assert np.array_equal(a.artifacts.clean, b.artifacts.clean)
