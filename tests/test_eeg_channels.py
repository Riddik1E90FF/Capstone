"""Tests for eeg/channels.py: name canonicalisation and site recognition.

The load-bearing assertions here are the *exclusions*.  EEGMAT's EDF contains
``"EEG A2-A1"`` (a bipolar derivation between two ear electrodes) and
``"ECG ECG"`` (a cardiac trace at roughly 100x scalp amplitude), and MNE types
both as ``eeg`` because the EDF header says nothing finer.  Either one reaching
an average reference or a channel mean corrupts every scalp channel, and
nothing errors when it does.
"""

from __future__ import annotations

import pytest

from eeg.channels import (CANONICAL_SITES, ROI_FRONTAL_MIDLINE, ROI_OCULAR_PROXY,
                          ROI_PARIETAL, ROIS, canonicalise, classify, is_modality_channel,
                          is_scalp_site, missing, resolve)


class TestCanonicalise:
    @pytest.mark.parametrize("raw,expected", [
        ("EEG Fz", "Fz"),
        ("EEG Fp1", "Fp1"),
        ("EEG F3", "F3"),
        ("EEG Pz", "Pz"),
        ("Fz", "Fz"),
        ("  Fz  ", "Fz"),
        ("fz", "Fz"),
        ("FZ", "Fz"),
        ("fp1", "Fp1"),
        ("FP2", "Fp2"),
        ("EEG-Cz", "Cz"),
        ("EEG_O1", "O1"),
        ("EEG:P4", "P4"),
    ])
    def test_recognised_sites(self, raw: str, expected: str) -> None:
        assert canonicalise(raw) == expected

    def test_eegmat_channel_names_all_canonicalise(self) -> None:
        """Every EEGMAT scalp channel must resolve to a known site.

        This is the exact channel list of Subject00_1.edf.  If a future MNE or
        EDF reader changes the spelling, this test is what notices — otherwise
        a renamed channel would simply be dropped as unrecognised and the
        pipeline would quietly run on 18 channels instead of 19.
        """
        eegmat = ["EEG Fp1", "EEG Fp2", "EEG F3", "EEG F4", "EEG F7", "EEG F8",
                  "EEG T3", "EEG T4", "EEG C3", "EEG C4", "EEG T5", "EEG T6",
                  "EEG P3", "EEG P4", "EEG O1", "EEG O2", "EEG Fz", "EEG Cz", "EEG Pz"]
        for name in eegmat:
            assert is_scalp_site(name), f"{name} not recognised as a scalp site"

    def test_unrecognised_name_returned_stripped_not_guessed(self) -> None:
        assert canonicalise("EEG Qq9") == "Qq9"
        assert not is_scalp_site("EEG Qq9")

    def test_bipolar_derivation_is_not_a_site(self) -> None:
        """A2-A1 is a difference between two electrodes, not a location."""
        assert canonicalise("EEG A2-A1") == "A2-A1"
        assert not is_scalp_site("EEG A2-A1")
        assert classify("EEG A2-A1") == "other"

    def test_ecg_is_not_a_site(self) -> None:
        assert canonicalise("ECG ECG") == "ECG"
        assert not is_scalp_site("ECG ECG")
        assert classify("ECG ECG") == "other"
        assert is_modality_channel("ECG ECG")

    def test_earlobes_are_recognised_but_not_scalp(self) -> None:
        """A1/A2 have montage positions but are reference locations."""
        for name in ("A1", "A2", "M1", "M2"):
            assert classify(name) == "reference"
            assert not is_scalp_site(name)

    def test_canonical_sites_are_self_canonical(self) -> None:
        """The table's own spellings must be fixed points of canonicalise."""
        for site in CANONICAL_SITES:
            assert canonicalise(site) == site

    def test_canonical_sites_have_no_duplicates(self) -> None:
        assert len(set(CANONICAL_SITES)) == len(CANONICAL_SITES)

    def test_old_and_new_temporal_names_are_distinct(self) -> None:
        """T3 and T7 are the same location under two conventions, but asking
        for one must not silently match the other: a selection that matched a
        site the file does not contain would report a band power for an
        electrode that was never there."""
        assert canonicalise("T3") == "T3"
        assert canonicalise("T7") == "T7"
        assert resolve(("T7",), ["T3", "Fz"]) == []


class TestResolve:
    def test_returns_requested_order_not_available_order(self) -> None:
        """Band-power arrays are indexed by position, so order is load-bearing.

        A caller asking for (Fz, F3, F4) that got montage order instead would
        compute correct numbers and label them wrongly.
        """
        available = ["F4", "Pz", "F3", "Fz"]
        assert resolve(("Fz", "F3", "F4"), available) == ["Fz", "F3", "F4"]

    def test_missing_sites_are_omitted_not_an_error(self) -> None:
        """Pre-registration §7 has a pre-registered degraded mode; resolve must
        permit a short list so that contingency is runnable."""
        assert resolve(("Fz", "F3", "Pz"), ["Fz", "F3"]) == ["Fz", "F3"]

    def test_missing_reports_what_resolve_dropped(self) -> None:
        assert missing(("Fz", "F3", "Pz"), ["Fz", "F3"]) == ["Pz"]
        assert missing(("Fz",), ["Fz"]) == []

    def test_roi_names_expand(self) -> None:
        available = ["Fz", "F3", "F4", "Pz", "P3", "P4"]
        assert resolve(("frontal_midline",), available) == list(ROI_FRONTAL_MIDLINE)
        assert resolve(("parietal",), available) == list(ROI_PARIETAL)

    def test_duplicates_collapse(self) -> None:
        assert resolve(("Fz", "Fz", "frontal_midline"), ["Fz", "F3", "F4"]) == [
            "Fz", "F3", "F4"]

    def test_raw_names_are_canonicalised_on_the_way_in(self) -> None:
        assert resolve(("EEG Fz", "fz "), ["Fz"]) == ["Fz"]

    def test_empty_request_gives_empty(self) -> None:
        assert resolve((), ["Fz"]) == []


class TestRois:
    def test_index_terms_match_the_preregistration(self) -> None:
        """Pre-registration §7 names Fz with F3/F4, and Pz with P3/P4.

        This test does not assert the index uses them — that is the index's
        business — only that the named groups say what §7 says.
        """
        assert ROI_FRONTAL_MIDLINE == ("Fz", "F3", "F4")
        assert ROI_PARIETAL == ("Pz", "P3", "P4")

    def test_every_roi_member_is_a_recognised_scalp_site(self) -> None:
        for name, group in ROIS.items():
            for site in group:
                assert is_scalp_site(site), f"ROI {name} contains non-site {site}"

    def test_ocular_proxy_is_frontopolar(self) -> None:
        assert ROI_OCULAR_PROXY == ("Fp1", "Fp2")
