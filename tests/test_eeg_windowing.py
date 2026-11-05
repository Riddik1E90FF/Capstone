"""Tests for eeg/windowing.py: window counts, times, and buffer independence.

The window count is load-bearing.  Pre-registration §6 reduces each
configuration to a median over its clean windows, so an off-by-one in the count
silently changes how many observations each median rests on, and the Gate 3
correlation is computed from those medians.  Nothing would error.
"""

from __future__ import annotations

import numpy as np
import pytest

from eeg.config import EpochConfig, FilterConfig
from eeg.filtering import apply_filters
from eeg.synthetic import synthetic_recording
from eeg.windowing import WindowError, make_windows, n_windows_for

SFREQ = 500.0


class TestWindowCount:
    @pytest.mark.parametrize("n_samples,window,hop,expected", [
        (1000, 1000, 1000, 1),      # exactly one window
        (999, 1000, 1000, 0),       # one sample short of one
        (2000, 1000, 1000, 2),      # exactly two, back to back
        (2999, 1000, 1000, 2),      # a 999-sample tail, dropped
        (3000, 1000, 500, 5),       # 50% overlap: starts at 0,500,1000,1500,2000
        (2000, 1000, 500, 3),       # starts at 0, 500, 1000
        (1000, 1, 1, 1000),         # degenerate but arithmetically defined
    ])
    def test_arithmetic(self, n_samples: int, window: int, hop: int, expected: int) -> None:
        assert n_windows_for(n_samples, window, hop) == expected

    def test_nonpositive_window_or_hop_is_an_error(self) -> None:
        with pytest.raises(WindowError):
            n_windows_for(1000, 0, 100)
        with pytest.raises(WindowError):
            n_windows_for(1000, 100, 0)

    def test_sixty_seconds_at_four_second_windows_gives_fifteen(self) -> None:
        rec = synthetic_recording(("Fz",), duration_seconds=60.0, seed=0)
        w = make_windows(rec, EpochConfig(window_seconds=4.0))
        assert w.n_windows == 15
        assert w.dropped_tail_samples == 0

    def test_overlap_roughly_doubles_the_count(self) -> None:
        rec = synthetic_recording(("Fz",), duration_seconds=60.0, seed=1)
        plain = make_windows(rec, EpochConfig(window_seconds=4.0))
        overlapped = make_windows(rec, EpochConfig(window_seconds=4.0, hop_seconds=2.0))
        assert overlapped.n_windows == 2 * plain.n_windows - 1


class TestShapesAndTimes:
    def test_shape_is_windows_channels_samples(self) -> None:
        rec = synthetic_recording(("Fz", "Pz", "Fp1"), duration_seconds=40.0, seed=2)
        w = make_windows(rec, EpochConfig(window_seconds=4.0))
        assert w.data.shape == (10, 3, 2000)
        assert w.n_windows == 10
        assert w.n_channels == 3
        assert w.n_samples == 2000

    def test_start_times_step_by_the_hop(self) -> None:
        rec = synthetic_recording(("Fz",), duration_seconds=40.0, seed=3)
        w = make_windows(rec, EpochConfig(window_seconds=4.0, hop_seconds=2.0))
        assert np.allclose(np.diff(w.start_seconds), 2.0)
        assert w.start_seconds[0] == 0.0

    def test_time_offset_restores_original_recording_time(self) -> None:
        """Edge trimming shifts every window time. Without the offset, a window
        could not be attributed to the configuration block that was on screen
        when it was recorded, which §7's per-configuration rejection rates need.
        """
        rec = synthetic_recording(("Fz",), duration_seconds=40.0, seed=4)
        filtered = apply_filters(rec, FilterConfig(edge_trim_seconds=1.0))
        w = make_windows(filtered, EpochConfig(window_seconds=4.0), time_offset=1.0)
        assert w.start_seconds[0] == 0.0
        assert w.absolute_start_seconds()[0] == 1.0

    def test_offset_is_recovered_from_provenance_when_not_given(self) -> None:
        rec = synthetic_recording(("Fz",), duration_seconds=40.0, seed=5)
        filtered = apply_filters(rec, FilterConfig(edge_trim_seconds=2.0))
        w = make_windows(filtered, EpochConfig(window_seconds=4.0))
        assert w.time_offset == pytest.approx(2.0)

    def test_offset_is_zero_for_an_unfiltered_recording(self) -> None:
        rec = synthetic_recording(("Fz",), duration_seconds=40.0, seed=6)
        assert make_windows(rec, EpochConfig()).time_offset == 0.0

    def test_dropped_tail_is_reported(self) -> None:
        """A dropped tail is data loss; reporting it keeps the loss visible."""
        rec = synthetic_recording(("Fz",), duration_seconds=10.0, seed=7)
        w = make_windows(rec, EpochConfig(window_seconds=4.0))
        assert w.n_windows == 2
        assert w.dropped_tail_samples == 1000      # 2 s at 500 Hz

    def test_channel_names_carry_through(self) -> None:
        rec = synthetic_recording(("Fz", "Pz"), duration_seconds=20.0, seed=8)
        assert make_windows(rec, EpochConfig()).channel_names == ("Fz", "Pz")


class TestContents:
    def test_window_contents_match_the_source_slice(self) -> None:
        rec = synthetic_recording(("Fz", "Pz"), duration_seconds=20.0, seed=9)
        w = make_windows(rec, EpochConfig(window_seconds=4.0))
        for i in range(w.n_windows):
            lo = i * 2000
            assert np.array_equal(w.data[i], rec.data[:, lo:lo + 2000])

    def test_windows_do_not_alias_each_other_or_the_source(self) -> None:
        """Overlapping windows built as a strided view would share a buffer, so
        writing into one would silently alter its neighbour. make_windows
        gathers by index for exactly this reason."""
        rec = synthetic_recording(("Fz",), duration_seconds=20.0, seed=10)
        w = make_windows(rec, EpochConfig(window_seconds=4.0, hop_seconds=2.0))
        neighbour_before = w.data[1].copy()
        source_before = rec.data.copy()
        w.data[0] += 1000.0
        assert np.array_equal(w.data[1], neighbour_before)
        assert np.array_equal(rec.data, source_before)

    def test_pick_returns_requested_channel_order(self) -> None:
        rec = synthetic_recording(("Fz", "Pz", "Fp1"), duration_seconds=20.0, seed=11)
        w = make_windows(rec, EpochConfig())
        picked = w.pick(["Fp1", "Fz"])
        assert picked.shape == (w.n_windows, 2, w.n_samples)
        assert np.array_equal(picked[:, 0], w.data[:, w.index_of("Fp1")])

    def test_missing_channel_raises_rather_than_returning_a_wrong_row(self) -> None:
        rec = synthetic_recording(("Fz",), duration_seconds=20.0, seed=12)
        w = make_windows(rec, EpochConfig())
        with pytest.raises(KeyError, match="not in these windows"):
            w.index_of("Pz")

    def test_channel_names_are_canonicalised_on_lookup(self) -> None:
        rec = synthetic_recording(("Fz",), duration_seconds=20.0, seed=13)
        w = make_windows(rec, EpochConfig())
        assert w.index_of("EEG Fz") == w.index_of("fz") == 0


class TestRefusals:
    def test_recording_shorter_than_one_window_is_an_error(self) -> None:
        """An empty Windows would propagate as a median over zero windows — a
        NaN that looks like an artifact problem rather than a short recording."""
        rec = synthetic_recording(("Fz",), duration_seconds=2.0, seed=14)
        with pytest.raises(WindowError, match="shorter than one"):
            make_windows(rec, EpochConfig(window_seconds=4.0))

    def test_zero_padding_is_refused_with_a_reason(self) -> None:
        rec = synthetic_recording(("Fz",), duration_seconds=10.0, seed=15)
        with pytest.raises(WindowError, match="zero-padded"):
            make_windows(rec, EpochConfig(window_seconds=4.0, drop_partial=False))

    def test_window_under_one_sample_is_an_error(self) -> None:
        rec = synthetic_recording(("Fz",), duration_seconds=10.0, seed=16)
        with pytest.raises(WindowError, match="under one sample"):
            make_windows(rec, EpochConfig(window_seconds=0.001))
