"""Segmentation of a continuous recording into fixed-length analysis windows.

The window is the unit of everything downstream: artifact detection returns one
clean/reject verdict per window, spectral estimation returns one band power per
window per channel, and the workload index (pre-registration §7, not in this
package) takes a median over the *clean windows* of a configuration block.  So
the window boundaries determine what "a measurement" means here.

Two properties matter and are both tested:

**Window count is exact and derivable.**  ``n_windows`` follows from the record
length, the window length and the hop, with no off-by-one slack.  A silent
off-by-one would not error; it would quietly change how many observations each
configuration's median is taken over, and the Gate 3 correlation is computed
from those medians.

**Partial windows are dropped, not padded.**  A zero-padded tail window has
artificially low power in every band and a step discontinuity at the pad
boundary.  It would pass the amplitude check — zeros are quiet — and contribute
a spuriously low band power to the median.  `EpochConfig.drop_partial` is True
by default for that reason, and the number dropped is recorded rather than
silently absorbed.

Window times are kept because they are the join key to anything else recorded
during a session.  Pre-registration §7 requires per-configuration rejection
rates, which means a window has to be attributable to the configuration block
that was on screen while it was recorded; `start_seconds` is how that
attribution will be made.  Note that the times are relative to the **filtered**
recording, so `FilterConfig.edge_trim_seconds` shifts them — `time_offset`
carries that shift so absolute session time stays recoverable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from eeg.config import EpochConfig
from eeg.loading import Provenance, Recording


class WindowError(ValueError):
    """A recording cannot be windowed as configured."""


@dataclass
class Windows:
    """Fixed-length analysis windows cut from one recording.

    `data` is ``(n_windows, n_channels, n_samples_per_window)``, float64,
    microvolts, with channel order matching `channel_names`.
    """

    data: np.ndarray
    channel_names: tuple[str, ...]
    sfreq: float
    #: Start time of each window, in seconds from the start of the array that
    #: was windowed (i.e. after any edge trim).  Shape ``(n_windows,)``.
    start_seconds: np.ndarray
    window_seconds: float
    hop_seconds: float
    #: Seconds removed from the front of the original recording before
    #: windowing — `FilterConfig.edge_trim_seconds`, when filtering ran.
    #: ``start_seconds + time_offset`` is the time in the original record.
    time_offset: float
    provenance: Provenance
    #: Samples at the tail that did not fill a whole window and were dropped.
    dropped_tail_samples: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise ValueError(
                "data must be 3-D (n_windows, n_channels, n_samples), got shape "
                f"{self.data.shape}")
        if self.data.shape[1] != len(self.channel_names):
            raise ValueError(
                f"data has {self.data.shape[1]} channels but {len(self.channel_names)} names")
        if self.start_seconds.shape != (self.data.shape[0],):
            raise ValueError(
                f"start_seconds has shape {self.start_seconds.shape}, expected "
                f"({self.data.shape[0]},)")

    @property
    def n_windows(self) -> int:
        return self.data.shape[0]

    @property
    def n_channels(self) -> int:
        return self.data.shape[1]

    @property
    def n_samples(self) -> int:
        return self.data.shape[2]

    def index_of(self, channel: str) -> int:
        """Channel axis index, by canonical name.  Raises `KeyError` if absent."""
        from eeg.channels import canonicalise
        canon = canonicalise(channel)
        try:
            return self.channel_names.index(canon)
        except ValueError:
            raise KeyError(
                f"channel {channel!r} (canonical {canon!r}) not in these windows; "
                f"have {list(self.channel_names)}") from None

    def pick(self, channels: tuple[str, ...] | list[str]) -> np.ndarray:
        """``(n_windows, len(channels), n_samples)``, channels in the order given."""
        return self.data[:, [self.index_of(c) for c in channels], :]

    def absolute_start_seconds(self) -> np.ndarray:
        """Window start times in the original recording's time base."""
        return self.start_seconds + self.time_offset


def n_windows_for(n_samples: int, window_samples: int, hop_samples: int) -> int:
    """How many whole windows fit.  The one place the arithmetic lives.

    A window starting at sample ``i * hop`` needs samples up to
    ``i * hop + window - 1``, so the last valid start is
    ``n_samples - window``, giving ``floor((n_samples - window) / hop) + 1``
    windows, or zero if the record is shorter than one window.
    """
    if window_samples <= 0 or hop_samples <= 0:
        raise WindowError("window and hop must both be positive sample counts")
    if n_samples < window_samples:
        return 0
    return (n_samples - window_samples) // hop_samples + 1


def make_windows(
    recording: Recording,
    config: EpochConfig | None = None,
    *,
    time_offset: float | None = None,
) -> Windows:
    """Cut `recording` into windows according to `config`.

    `time_offset` defaults to the edge trim recorded in the recording's
    provenance when one is detectable, and 0.0 otherwise.  Pass it explicitly
    when the caller knows the offset for certain.

    Raises `WindowError` if the recording is shorter than one window, rather
    than returning an empty `Windows`.  An empty result would propagate as a
    median over zero clean windows, which is a NaN that looks like an artifact
    problem rather than what it is — a recording too short to measure.
    """
    config = config or EpochConfig()
    config.validate()

    sfreq = recording.sfreq
    window_samples = int(round(config.window_seconds * sfreq))
    hop_samples = int(round(config.effective_hop_seconds * sfreq))
    if window_samples < 1:
        raise WindowError(
            f"window_seconds={config.window_seconds} at {sfreq} Hz is under one sample")
    if hop_samples < 1:
        raise WindowError(
            f"hop_seconds={config.effective_hop_seconds} at {sfreq} Hz is under one sample")

    n_samples = recording.n_samples
    count = n_windows_for(n_samples, window_samples, hop_samples)
    if count == 0:
        raise WindowError(
            f"recording is {recording.duration_seconds:.3f} s ({n_samples} samples), "
            f"shorter than one {config.window_seconds:g} s window. Nothing to analyse.")
    if not config.drop_partial:
        raise WindowError(
            "drop_partial=False is not implemented: a zero-padded window has "
            "artificially low power in every band and a step at the pad boundary, so it "
            "would pass the amplitude check and bias the median downward. If a partial "
            "tail must be analysed, shorten window_seconds instead.")

    # Gather by index rather than by stride tricks: an as_strided view over the
    # recording would alias the source buffer, so a later in-place operation on
    # one window would silently alter its neighbours wherever windows overlap.
    starts = np.arange(count, dtype=np.int64) * hop_samples
    offsets = np.arange(window_samples, dtype=np.int64)
    gather = starts[:, None] + offsets[None, :]            # (n_windows, n_samples)
    data = recording.data[:, gather]                        # (n_ch, n_win, n_samp)
    data = np.ascontiguousarray(np.transpose(data, (1, 0, 2)))

    last_end = int(starts[-1]) + window_samples
    dropped = n_samples - last_end

    if time_offset is None:
        time_offset = _trim_offset_from(recording.provenance)

    return Windows(
        data=data,
        channel_names=recording.channel_names,
        sfreq=sfreq,
        start_seconds=starts / sfreq,
        window_seconds=window_samples / sfreq,
        hop_seconds=hop_samples / sfreq,
        time_offset=float(time_offset),
        provenance=recording.provenance,
        dropped_tail_samples=int(dropped),
    )


def _trim_offset_from(provenance: Provenance) -> float:
    """Recover the edge trim, in seconds, from a provenance filter log.

    Best-effort and deliberately so: the authoritative value is the
    `FilterConfig` the caller used, and `eeg.pipeline` passes it explicitly.
    This fallback exists for the case of a `Recording` handed in from
    elsewhere, and returns 0.0 rather than guessing when it cannot tell.
    """
    for stage in provenance.filters_applied:
        if stage.startswith("trimmed "):
            try:
                return float(stage.split()[1])
            except (IndexError, ValueError):
                return 0.0
    return 0.0
