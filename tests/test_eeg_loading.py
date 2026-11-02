"""Tests for eeg/loading.py: channel exclusion, units, referencing, provenance.

Most of these run on arrays, because `prepare` is where all the decisions are
made and `load_edf` is a thin wrapper over `mne.io.read_raw_edf` plus
`prepare`.  The EDF tests at the bottom run against the real EEGMAT files and
skip when `data/raw/eegmat/` is absent — `data/` is gitignored (D032 retains
EEGMAT as the development fixture, but the files themselves are never
committed), so a fresh checkout has no data and must still pass its suite.

Every assertion here guards a silent failure.  Units, referencing and channel
exclusion all produce plausible-looking numbers when wrong: a pipeline that
believes volts are microvolts rejects nothing, because every threshold is then
a million times too permissive.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eeg.config import LoadConfig
from eeg.loading import (VOLTS_TO_MICROVOLTS, LoadError, Recording, load_edf, prepare,
                         sha256_file)

SFREQ = 500.0
REPO = Path(__file__).resolve().parents[1]
EEGMAT_DIR = REPO / "data" / "raw" / "eegmat"
EEGMAT_REST = EEGMAT_DIR / "Subject00_1.edf"
EEGMAT_TASK = EEGMAT_DIR / "Subject00_2.edf"

needs_eegmat = pytest.mark.skipif(
    not EEGMAT_REST.exists(),
    reason="EEGMAT not downloaded; run scripts/eeg_demo.py --download or "
           "scripts/verify_eegmat.py")

#: The real EEGMAT channel list, in file order, including the two channels that
#: must be excluded.
EEGMAT_NAMES = ["EEG Fp1", "EEG Fp2", "EEG F3", "EEG F4", "EEG F7", "EEG F8",
                "EEG T3", "EEG T4", "EEG C3", "EEG C4", "EEG T5", "EEG T6",
                "EEG P3", "EEG P4", "EEG O1", "EEG O2", "EEG Fz", "EEG Cz",
                "EEG Pz", "EEG A2-A1", "ECG ECG"]


def _volts(n_channels: int, n_samples: int = 10_000, amplitude_uv: float = 20.0):
    """`n_channels` sinusoids at `amplitude_uv`, expressed in volts."""
    t = np.arange(n_samples) / SFREQ
    rows = [amplitude_uv * 1e-6 * np.sin(2 * np.pi * 10.0 * t + i) for i in range(n_channels)]
    return np.stack(rows)


class TestChannelExclusion:
    def test_eegmat_layout_keeps_nineteen_scalp_channels(self) -> None:
        """21 channels in, 19 out: the A2-A1 bipolar derivation and the ECG
        trace are excluded. Both are typed `eeg` by MNE because the EDF header
        says nothing finer, so type cannot be relied on and the name decides."""
        rec = prepare(_volts(21), EEGMAT_NAMES, SFREQ, LoadConfig(montage="none"))
        assert rec.n_channels == 19
        assert "A2-A1" not in rec.channel_names
        assert "ECG" not in rec.channel_names
        assert rec.channel_names[0] == "Fp1"

    def test_exclusions_are_named_with_reasons(self) -> None:
        rec = prepare(_volts(21), EEGMAT_NAMES, SFREQ, LoadConfig(montage="none"))
        dropped = rec.provenance.dropped_channels
        assert "bipolar derivation" in dropped["A2-A1"]
        assert "non-EEG modality" in dropped["ECG"]

    def test_ecg_amplitude_cannot_reach_a_channel_average(self) -> None:
        """The concrete harm. An ECG trace at ~100x scalp amplitude inside an
        average reference subtracts a cardiac signal from every channel."""
        data = _volts(21)
        data[20] *= 100.0                                   # the ECG row
        rec = prepare(data, EEGMAT_NAMES, SFREQ, LoadConfig(reference="average"))
        assert float(np.abs(rec.data).max()) < 1e4          # uV, not 1e6

    def test_keeping_non_scalp_channels_is_possible_but_explicit(self) -> None:
        rec = prepare(_volts(21), EEGMAT_NAMES, SFREQ,
                      LoadConfig(drop_non_scalp=False, montage="none"))
        assert rec.n_channels == 21
        assert "ECG" in rec.channel_names

    def test_explicit_selection_keeps_requested_order(self) -> None:
        rec = prepare(_volts(21), EEGMAT_NAMES, SFREQ,
                      LoadConfig(channels=("Fz", "F3", "F4"), montage="none"))
        assert rec.channel_names == ("Fz", "F3", "F4")

    def test_requesting_an_absent_channel_warns(self) -> None:
        rec = prepare(_volts(21), EEGMAT_NAMES, SFREQ,
                      LoadConfig(channels=("Fz", "Oz"), montage="none"))
        assert rec.channel_names == ("Fz",)
        assert any("does not have" in w for w in rec.provenance.warnings)

    def test_selecting_nothing_available_is_an_error(self) -> None:
        with pytest.raises(LoadError, match="no channels left"):
            prepare(_volts(21), EEGMAT_NAMES, SFREQ,
                    LoadConfig(channels=("Oz",), montage="none"))

    def test_duplicate_sites_after_canonicalisation_are_refused(self) -> None:
        """Two rows claiming the same site cannot both be it, and `index_of`
        would silently return only the first."""
        with pytest.raises(LoadError, match="collide after canonicalisation"):
            prepare(_volts(2), ["EEG Fz", "Fz"], SFREQ, LoadConfig(montage="none"))


class TestRequiredChannels:
    def test_missing_required_channel_raises(self) -> None:
        with pytest.raises(LoadError, match=r"required channel\(s\) \['Oz'\]"):
            prepare(_volts(21), EEGMAT_NAMES, SFREQ,
                    LoadConfig(require_channels=("Fz", "Oz"), montage="none"))

    def test_present_required_channels_pass(self) -> None:
        prepare(_volts(21), EEGMAT_NAMES, SFREQ,
                LoadConfig(require_channels=("Fz", "Pz"), montage="none"))

    def test_nothing_is_required_by_default(self) -> None:
        """Pre-registration §7 has a pre-registered degraded mode that drops
        the parietal alpha term; a hard requirement here would make that
        contingency unrunnable."""
        assert LoadConfig().require_channels == ()


class TestUnits:
    def test_volts_are_rescaled_to_microvolts(self) -> None:
        """A 20 uV sine arriving as 2e-5 V must come out as 20, not 2e-5.
        Getting this wrong makes every amplitude threshold meaningless in one
        direction or the other, and nothing raises either way."""
        rec = prepare(_volts(1), ["EEG Fz"], SFREQ, LoadConfig(montage="none"))
        # rel=1e-3, not exact: a 10 Hz sine sampled at 500 Hz has 50 samples
        # per cycle and none of them lands exactly on the peak, so the realised
        # maximum is a shade under the amplitude. The tolerance is for the
        # sampling grid, not for the scaling, which must be exact.
        assert float(np.abs(rec.data).max()) == pytest.approx(20.0, rel=5e-3)
        assert rec.provenance.scale_factor == VOLTS_TO_MICROVOLTS

    def test_already_microvolts_is_not_rescaled_and_says_so(self) -> None:
        data = _volts(1) * 1e6
        rec = prepare(data, ["Fz"], SFREQ, LoadConfig(montage="none"),
                      already_microvolts=True)
        assert float(np.abs(rec.data).max()) == pytest.approx(20.0, rel=5e-3)
        assert rec.provenance.scale_factor == 1.0

    def test_scaling_can_be_disabled(self) -> None:
        rec = prepare(_volts(1), ["Fz"], SFREQ,
                      LoadConfig(scale_to_microvolts=False, montage="none"))
        assert rec.provenance.scale_factor == 1.0
        assert float(np.abs(rec.data).max()) < 1e-3


class TestReference:
    def test_none_leaves_the_data_alone(self) -> None:
        data = _volts(4) + 1e-5
        rec = prepare(data, ["Fz", "F3", "F4", "Pz"], SFREQ, LoadConfig(montage="none"))
        assert np.allclose(rec.data, data * VOLTS_TO_MICROVOLTS)
        assert rec.provenance.reference_applied.startswith("none")

    def test_average_reference_zeroes_the_across_channel_mean(self) -> None:
        rec = prepare(_volts(19), EEGMAT_NAMES[:19], SFREQ,
                      LoadConfig(reference="average", montage="none"))
        assert np.allclose(rec.data.mean(axis=0), 0.0, atol=1e-9)

    def test_average_reference_names_the_channels_it_averaged(self) -> None:
        rec = prepare(_volts(21), EEGMAT_NAMES, SFREQ,
                      LoadConfig(reference="average", montage="none"))
        assert "average over 19 channels" in rec.provenance.reference_applied
        assert "ECG" not in rec.provenance.reference_applied

    def test_sparse_average_reference_warns_about_its_own_bias(self) -> None:
        """An average over 8 non-uniform electrodes contains the midline
        activity it is then subtracted from, shrinking midline effects toward
        zero — and the index's primary term is a midline one."""
        rec = prepare(_volts(6), ["Fz", "F3", "F4", "Pz", "P3", "P4"], SFREQ,
                      LoadConfig(reference="average", montage="none"))
        assert any("shrinks midline effects" in w for w in rec.provenance.warnings)

    def test_dense_average_reference_does_not_warn(self) -> None:
        rec = prepare(_volts(19), EEGMAT_NAMES[:19], SFREQ,
                      LoadConfig(reference="average", montage="none"))
        assert not any("shrinks midline" in w for w in rec.provenance.warnings)


class TestMontage:
    def test_positions_are_attached_for_known_sites(self) -> None:
        rec = prepare(_volts(3), ["Fz", "Pz", "Fp1"], SFREQ,
                      LoadConfig(montage="standard_1020"))
        assert set(rec.positions) == {"Fz", "Pz", "Fp1"}
        assert all(len(p) == 3 for p in rec.positions.values())

    def test_montage_none_attaches_nothing_and_needs_no_mne(self) -> None:
        rec = prepare(_volts(2), ["Fz", "Pz"], SFREQ, LoadConfig(montage="none"))
        assert rec.positions == {}
        assert rec.provenance.montage == "none"

    def test_the_resolved_montage_name_is_recorded(self) -> None:
        """MNE renamed standard_1020 to colin27_1020 in 1.13 and deprecates the
        old name. Whichever name the installed MNE provides is the one
        recorded, so provenance says what was actually used."""
        rec = prepare(_volts(1), ["Fz"], SFREQ, LoadConfig(montage="standard_1020"))
        assert rec.provenance.montage in {"standard_1020", "colin27_1020"}

    def test_eegmat_scalp_channels_all_get_positions(self) -> None:
        rec = prepare(_volts(21), EEGMAT_NAMES, SFREQ,
                      LoadConfig(montage="standard_1020"))
        assert rec.provenance.montage_missing_positions == ()
        assert len(rec.positions) == 19


class TestRecording:
    def test_shape_mismatch_is_refused(self) -> None:
        with pytest.raises(ValueError, match="channel names"):
            prepare(_volts(3), ["Fz", "Pz"], SFREQ, LoadConfig(montage="none"))

    def test_one_dimensional_data_is_refused(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            prepare(np.zeros(100), ["Fz"], SFREQ, LoadConfig(montage="none"))

    def test_nonpositive_sfreq_is_refused(self) -> None:
        with pytest.raises(ValueError, match="sfreq must be > 0"):
            Recording(data=np.zeros((1, 10)), channel_names=("Fz",), sfreq=0.0,
                      provenance=prepare(_volts(1), ["Fz"], SFREQ,
                                         LoadConfig(montage="none")).provenance)

    def test_index_of_raises_rather_than_returning_a_sentinel(self) -> None:
        """A silently wrong row index produces a correct-looking band power for
        the wrong electrode."""
        rec = prepare(_volts(2), ["Fz", "Pz"], SFREQ, LoadConfig(montage="none"))
        with pytest.raises(KeyError, match="not in this recording"):
            rec.index_of("Oz")

    def test_lookup_canonicalises(self) -> None:
        rec = prepare(_volts(1), ["EEG Fz"], SFREQ, LoadConfig(montage="none"))
        assert rec.index_of("Fz") == rec.index_of("EEG Fz") == rec.index_of("fz") == 0

    def test_pick_returns_requested_order(self) -> None:
        rec = prepare(_volts(3), ["Fz", "F3", "F4"], SFREQ, LoadConfig(montage="none"))
        picked = rec.pick(["F4", "Fz"])
        assert np.array_equal(picked[0], rec.data[2])
        assert np.array_equal(picked[1], rec.data[0])

    def test_dtype_is_float64(self) -> None:
        rec = prepare(_volts(1).astype(np.float32), ["Fz"], SFREQ,
                      LoadConfig(montage="none"))
        assert rec.data.dtype == np.float64

    def test_input_array_is_not_modified(self) -> None:
        data = _volts(2)
        before = data.copy()
        prepare(data, ["Fz", "Pz"], SFREQ, LoadConfig(reference="average", montage="none"))
        assert np.array_equal(data, before)


class TestProvenance:
    def test_source_names_are_kept_verbatim(self) -> None:
        """So an unrecognised spelling that cost a channel is visible in the
        output rather than discovered months later."""
        rec = prepare(_volts(21), EEGMAT_NAMES, SFREQ, LoadConfig(montage="none"))
        assert list(rec.provenance.source_channel_names) == EEGMAT_NAMES

    def test_duration_and_sample_count_match_the_array(self) -> None:
        rec = prepare(_volts(2, n_samples=5000), ["Fz", "Pz"], SFREQ,
                      LoadConfig(montage="none"))
        assert rec.provenance.n_samples == 5000
        assert rec.provenance.duration_seconds == pytest.approx(10.0)

    def test_versions_are_recorded(self) -> None:
        rec = prepare(_volts(1), ["Fz"], SFREQ, LoadConfig(montage="none"))
        assert rec.provenance.numpy_version == np.__version__
        assert rec.provenance.loaded_utc

    def test_to_dict_is_json_safe(self) -> None:
        import json
        rec = prepare(_volts(21), EEGMAT_NAMES, SFREQ, LoadConfig(montage="none"))
        json.dumps(rec.provenance.to_dict(), allow_nan=False)

    def test_filters_applied_is_empty_before_filtering(self) -> None:
        rec = prepare(_volts(1), ["Fz"], SFREQ, LoadConfig(montage="none"))
        assert rec.provenance.filters_applied == ()


class TestSha256:
    def test_hash_matches_hashlib(self, tmp_path: Path) -> None:
        import hashlib
        path = tmp_path / "x.bin"
        payload = b"abc" * 1000
        path.write_bytes(payload)
        assert sha256_file(path) == hashlib.sha256(payload).hexdigest()

    def test_hash_changes_with_content(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a", tmp_path / "b"
        a.write_bytes(b"x")
        b.write_bytes(b"y")
        assert sha256_file(a) != sha256_file(b)


class TestLoadEdf:
    def test_missing_file_raises_before_mne_is_touched(self) -> None:
        with pytest.raises(LoadError, match="no such recording"):
            load_edf("data/raw/eegmat/Subject99_9.edf")

    @needs_eegmat
    def test_eegmat_rest_loads_with_the_expected_shape(self) -> None:
        """21 channels at 500 Hz, 182 s, per D032. Pinned so a dataset or
        reader change is noticed rather than absorbed."""
        rec = load_edf(EEGMAT_REST, LoadConfig())
        assert rec.sfreq == 500.0
        assert len(rec.provenance.source_channel_names) == 21
        assert rec.n_channels == 19
        assert rec.duration_seconds == pytest.approx(182.0, abs=0.01)

    @needs_eegmat
    def test_eegmat_task_is_the_shorter_eyes_open_recording(self) -> None:
        """62 s, against rest's 182 s (Zyma et al. 2019: 180 s eyes-closed
        rest, 60 s eyes-open arithmetic). The asymmetry is why D032 voids the
        rest/task workload contrast."""
        rec = load_edf(EEGMAT_TASK, LoadConfig())
        assert rec.duration_seconds == pytest.approx(62.0, abs=0.01)

    @needs_eegmat
    def test_eegmat_amplitudes_are_plausible_microvolts(self) -> None:
        """Scalp EEG is tens of microvolts. If this were volts the numbers
        would be ~1e-5, and every artifact threshold would be inert."""
        rec = load_edf(EEGMAT_REST, LoadConfig())
        rms = float(np.sqrt(np.mean(np.square(rec.data))))
        assert 1.0 < rms < 200.0

    @needs_eegmat
    def test_eegmat_header_filters_are_recorded(self) -> None:
        """EEGMAT's header reports 0.5-45 Hz, which is why this pipeline's own
        1-45 Hz bandpass is nearly a no-op there and why no mains notch is
        needed (D032, and FilterConfig.notch_freqs)."""
        rec = load_edf(EEGMAT_REST, LoadConfig())
        assert rec.provenance.source_lowpass_hz == pytest.approx(45.0)
        assert rec.provenance.source_highpass_hz == pytest.approx(0.5)

    @needs_eegmat
    def test_eegmat_index_channels_are_all_present(self) -> None:
        """Pre-registration §7 needs Fz/F3/F4 and Pz/P3/P4; the artifact
        detector needs Fp1/Fp2. EEGMAT has all eight."""
        rec = load_edf(EEGMAT_REST, LoadConfig())
        for site in ("Fz", "F3", "F4", "Pz", "P3", "P4", "Fp1", "Fp2"):
            assert site in rec.channel_names

    @needs_eegmat
    def test_eegmat_provenance_carries_a_hash_and_an_absolute_path(self) -> None:
        """`data/` is gitignored, so the files behind a result are not in the
        repository and the hash is the only link back to them."""
        rec = load_edf(EEGMAT_REST, LoadConfig())
        assert rec.provenance.source_sha256 is not None
        assert len(rec.provenance.source_sha256) == 64
        assert Path(rec.provenance.source).is_absolute()
        assert rec.provenance.mne_version is not None
