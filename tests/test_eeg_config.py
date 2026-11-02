"""Tests for eeg/config.py: validation, serialisation and the cross-checks.

The interesting tests here are the cross-section ones.  Each section can be
individually valid while the combination is silently useless — a notch outside
the passband that removes nothing, a band outside the passband whose power is
really the filter's roll-off, a Welch segment longer than the window SciPy
would quietly shorten.  None of those raises anywhere else, and each produces
numbers that look like measurements.
"""

from __future__ import annotations

import json

import pytest

from eeg.config import (ArtifactConfig, ConfigError, DEFAULT_BANDS, EEGMAT_MAINS_HZ,
                        EpochConfig, FilterConfig, LOCAL_MAINS_HZ, LoadConfig,
                        PipelineConfig, SpectralConfig, parse_override)


class TestDefaultsAreConstructible:
    def test_every_section_default_validates(self) -> None:
        """A default that does not validate would make the package unusable
        out of the box, and the cross-section checks make that a real risk."""
        for section in (LoadConfig(), FilterConfig(), EpochConfig(),
                        ArtifactConfig(), SpectralConfig()):
            assert section.validate() is section

    def test_pipeline_default_validates(self) -> None:
        PipelineConfig().validate()

    def test_eegmat_preset_validates(self) -> None:
        PipelineConfig.for_eegmat().validate()

    def test_default_bands_fit_inside_the_default_passband(self) -> None:
        """The cross-check that keeps a reported band from being the filter's
        roll-off rather than the EEG's power."""
        f = FilterConfig()
        for _, lo, hi in DEFAULT_BANDS:
            assert lo >= f.bandpass_low
            assert hi <= f.bandpass_high


class TestLoadConfig:
    def test_rejects_unknown_reference(self) -> None:
        with pytest.raises(ConfigError, match="reference must be one of"):
            LoadConfig(reference="linked_ears")

    def test_rejects_unknown_montage(self) -> None:
        with pytest.raises(ConfigError, match="montage must be one of"):
            LoadConfig(montage="biosemi64")

    def test_average_reference_with_non_scalp_retained_is_refused(self) -> None:
        """Averaging an ECG trace into every scalp channel is the exact failure
        this combination produces, and it is silent."""
        with pytest.raises(ConfigError, match="would average an ECG"):
            LoadConfig(reference="average", drop_non_scalp=False)

    def test_montage_alias_accepted(self) -> None:
        """MNE renamed standard_1020 to colin27_1020 in 1.13; both must load."""
        LoadConfig(montage="colin27_1020").validate()


class TestFilterConfig:
    def test_rejects_inverted_band(self) -> None:
        with pytest.raises(ConfigError, match="bandpass_low must be < bandpass_high"):
            FilterConfig(bandpass_low=45.0, bandpass_high=1.0)

    def test_rejects_nonpositive_cutoff(self) -> None:
        with pytest.raises(ConfigError, match="bandpass_low must be > 0"):
            FilterConfig(bandpass_low=0.0)

    def test_one_sided_filters_are_legal(self) -> None:
        FilterConfig(bandpass_low=None).validate()
        FilterConfig(bandpass_high=None).validate()

    def test_notch_is_off_by_default_because_the_lowpass_precedes_it(self) -> None:
        """Documented choice, asserted so a future change is deliberate: with a
        45 Hz low-pass there is no mains left for a notch to remove, and a
        logged-but-inert filter stage misleads a reader of the provenance."""
        assert FilterConfig().notch_freqs == ()

    def test_mains_constants_are_not_interchangeable(self) -> None:
        """EEGMAT is 50 Hz (Ukraine), the author's hardware 60 Hz. A notch at
        the wrong one removes nothing and reports success."""
        assert EEGMAT_MAINS_HZ == 50.0
        assert LOCAL_MAINS_HZ == 60.0

    def test_with_mains_notch_widens_the_passband_to_make_the_notch_real(self) -> None:
        cfg = FilterConfig.with_mains_notch(LOCAL_MAINS_HZ)
        assert cfg.notch_freqs == (60.0,)
        assert cfg.bandpass_high > 60.0
        PipelineConfig(filter=cfg).validate()

    def test_negative_edge_trim_rejected(self) -> None:
        with pytest.raises(ConfigError, match="edge_trim_seconds must be >= 0"):
            FilterConfig(edge_trim_seconds=-1.0)


class TestEpochConfig:
    def test_hop_defaults_to_window_length(self) -> None:
        assert EpochConfig(window_seconds=4.0).effective_hop_seconds == 4.0

    def test_explicit_hop_is_used(self) -> None:
        assert EpochConfig(window_seconds=4.0, hop_seconds=2.0).effective_hop_seconds == 2.0

    def test_hop_larger_than_window_rejected(self) -> None:
        """A hop beyond the window length silently discards the samples in
        between, which would be invisible in the output."""
        with pytest.raises(ConfigError, match="hop_seconds must be <= window_seconds"):
            EpochConfig(window_seconds=2.0, hop_seconds=4.0)

    def test_nonpositive_window_rejected(self) -> None:
        with pytest.raises(ConfigError, match="window_seconds must be > 0"):
            EpochConfig(window_seconds=0.0)


class TestArtifactConfig:
    def test_flat_floor_above_ptp_ceiling_rejected(self) -> None:
        """Inverted thresholds reject every window for two contradictory
        reasons at once."""
        with pytest.raises(ConfigError, match="flat_uv must be < peak_to_peak_uv"):
            ArtifactConfig(flat_uv=200.0, peak_to_peak_uv=150.0)

    def test_inverted_ocular_band_rejected(self) -> None:
        with pytest.raises(ConfigError, match="ocular_band must satisfy"):
            ArtifactConfig(ocular_band=(6.0, 1.0))

    def test_defaults_are_the_documented_ones(self) -> None:
        """Pinned so a change to any of these shows up as a failing test and
        therefore as a decision, not as a drifting default. Every one of them
        is an OPEN item in eeg/README.md."""
        cfg = ArtifactConfig()
        assert cfg.peak_to_peak_uv == 150.0
        assert cfg.flat_uv == 0.5
        assert cfg.max_gradient_uv_per_sample == 25.0
        assert cfg.ocular_channels == ("Fp1", "Fp2")
        assert cfg.ocular_band == (1.0, 6.0)
        assert cfg.ocular_z_threshold == 5.0
        assert cfg.require_ocular_channels is True
        assert cfg.reject_if_any_channel is True


class TestSpectralConfig:
    def test_duplicate_band_name_rejected(self) -> None:
        with pytest.raises(ConfigError, match="duplicate band name"):
            SpectralConfig(bands=(("theta", 4.0, 8.0), ("theta", 8.0, 13.0)))

    def test_inverted_band_rejected(self) -> None:
        with pytest.raises(ConfigError, match="must satisfy 0 <= low < high"):
            SpectralConfig(bands=(("theta", 8.0, 4.0),))

    def test_overlap_of_one_rejected(self) -> None:
        with pytest.raises(ConfigError, match="welch_overlap must be in"):
            SpectralConfig(welch_overlap=1.0)

    def test_band_names_property(self) -> None:
        assert SpectralConfig().band_names == ("delta", "theta", "alpha", "beta")

    def test_bands_are_half_open_and_partition_without_overlap(self) -> None:
        """Closed intervals would put the 8 Hz bin in both theta and alpha, so
        the index's positive theta term and negative alpha term would share a
        bin and partly cancel."""
        edges = SpectralConfig().bands
        for (_, _, hi), (_, lo_next, _) in zip(edges, edges[1:]):
            assert hi == lo_next


class TestPipelineCrossChecks:
    def test_notch_above_passband_rejected(self) -> None:
        with pytest.raises(ConfigError, match="removes nothing"):
            PipelineConfig(filter=FilterConfig(notch_freqs=(60.0,), bandpass_high=45.0))

    def test_band_below_passband_rejected(self) -> None:
        with pytest.raises(ConfigError, match="shaped by the high-pass"):
            PipelineConfig(
                filter=FilterConfig(bandpass_low=4.0),
                spectral=SpectralConfig(bands=(("delta", 1.0, 4.0), ("theta", 4.0, 8.0))))

    def test_band_above_passband_rejected(self) -> None:
        with pytest.raises(ConfigError, match="shaped by the low-pass"):
            PipelineConfig(
                filter=FilterConfig(bandpass_high=20.0),
                spectral=SpectralConfig(bands=(("beta", 13.0, 30.0),)))

    def test_welch_segment_longer_than_window_rejected(self) -> None:
        with pytest.raises(ConfigError, match="exceeds window_seconds"):
            PipelineConfig(
                epoch=EpochConfig(window_seconds=2.0),
                spectral=SpectralConfig(welch_segment_seconds=4.0))

    def test_ocular_band_outside_passband_rejected(self) -> None:
        with pytest.raises(ConfigError, match="ocular_band starts at"):
            PipelineConfig(
                filter=FilterConfig(bandpass_low=4.0),
                spectral=SpectralConfig(bands=(("theta", 4.0, 8.0),)),
                artifact=ArtifactConfig(ocular_band=(1.0, 6.0)))


class TestSerialisation:
    def test_round_trip_through_json_is_identical(self) -> None:
        cfg = PipelineConfig.for_eegmat()
        again = PipelineConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
        assert again == cfg

    def test_to_json_and_back(self, tmp_path) -> None:
        cfg = PipelineConfig(epoch=EpochConfig(window_seconds=2.0, hop_seconds=1.0))
        path = tmp_path / "cfg.json"
        cfg.to_json(path)
        assert PipelineConfig.from_json(path) == cfg

    def test_partial_dict_takes_defaults(self) -> None:
        cfg = PipelineConfig.from_dict({"epoch": {"window_seconds": 2.0}})
        assert cfg.epoch.window_seconds == 2.0
        assert cfg.filter == FilterConfig()

    def test_unknown_top_level_key_raises(self) -> None:
        with pytest.raises(ConfigError, match="unknown PipelineConfig key"):
            PipelineConfig.from_dict({"filtre": {}})

    def test_unknown_section_key_raises_rather_than_being_ignored(self) -> None:
        """A typo silently ignored is a run that did not use the parameter the
        author believed it used."""
        with pytest.raises(ConfigError, match="unknown FilterConfig key"):
            PipelineConfig.from_dict({"filter": {"bandpas_high": 40.0}})

    def test_to_dict_is_json_serialisable_with_no_nan(self) -> None:
        json.dumps(PipelineConfig().to_dict(), allow_nan=False)


class TestOverrides:
    def test_dotted_override(self) -> None:
        cfg = PipelineConfig().with_overrides({"epoch.window_seconds": 2.0})
        assert cfg.epoch.window_seconds == 2.0
        assert cfg.spectral == SpectralConfig(welch_segment_seconds=2.0)

    def test_string_values_are_coerced(self) -> None:
        cfg = PipelineConfig().with_overrides({
            "artifact.peak_to_peak_uv": "120",
            "artifact.require_ocular_channels": "false",
            "load.channels": "Fz,F3,F4",
        })
        assert cfg.artifact.peak_to_peak_uv == 120.0
        assert cfg.artifact.require_ocular_channels is False
        assert cfg.load.channels == ("Fz", "F3", "F4")

    def test_optional_float_accepts_none_as_a_string(self) -> None:
        cfg = PipelineConfig().with_overrides({"spectral.welch_segment_seconds": "none"})
        assert cfg.spectral.welch_segment_seconds is None

    def test_unknown_section_raises(self) -> None:
        with pytest.raises(ConfigError, match="unknown config section"):
            PipelineConfig().with_overrides({"filterr.bandpass_low": 1.0})

    def test_unknown_field_raises(self) -> None:
        with pytest.raises(ConfigError, match="unknown filter field"):
            PipelineConfig().with_overrides({"filter.cutoff": 1.0})

    def test_override_still_validates_cross_section(self) -> None:
        with pytest.raises(ConfigError, match="removes nothing"):
            PipelineConfig().with_overrides({"filter.notch_freqs": "60"})

    def test_parse_override(self) -> None:
        assert parse_override("epoch.window_seconds=2.0") == ("epoch.window_seconds", "2.0")
        with pytest.raises(ConfigError, match="must look like"):
            parse_override("epoch.window_seconds")


class TestImmutability:
    def test_sections_are_frozen(self) -> None:
        """A config mutated after a result recorded it would make the record a
        lie about what ran."""
        cfg = FilterConfig()
        with pytest.raises(Exception):
            cfg.bandpass_low = 2.0            # type: ignore[misc]

    def test_configs_compare_by_value(self) -> None:
        assert PipelineConfig() == PipelineConfig()
        assert PipelineConfig() != PipelineConfig(note="x")
