"""Loading EEG recordings, with provenance recorded.

What comes out
--------------
A `Recording`: a plain float64 array in microvolts, canonical channel names, a
sampling rate, optional sensor positions, and a `Provenance` record of
everything that was done to get there.  No MNE object escapes this module.
That boundary is deliberate — the rest of the package, and its tests, work on
arrays, so nothing downstream depends on an MNE version or on a `Raw`'s
internal state, and a synthetic signal with a known spectrum is as valid an
input as an EDF file.

Why provenance is part of the data structure
--------------------------------------------
Every step here is lossy or transformative, and each one is silent when wrong:
channels are dropped, units are rescaled, a reference may be subtracted, a
montage may fail to place an electrode.  A result that records only the final
array cannot answer "was the ECG channel in the average reference?" — and that
question decides whether a band power means anything.  So the answer travels
with the array, into `PipelineResult`, into the demo's output, and into
whatever the index eventually writes.

`Provenance.warnings` is the honest-degradation channel.  Pre-registration §7
has a *pre-registered degraded mode* (drop the parietal alpha term if the
hardware gate fails at parietal sites), so the pipeline must be able to run
short of its ideal input.  It may do that only by saying so in writing.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from eeg.channels import canonicalise, classify, is_modality_channel, missing
from eeg.config import LoadConfig

#: Volts to microvolts.  MNE returns EDF data in volts; every amplitude
#: threshold in `ArtifactConfig` and every number in the EEG literature is in
#: microvolts.  Named rather than inlined so the factor appears in provenance.
VOLTS_TO_MICROVOLTS = 1e6


class LoadError(RuntimeError):
    """A recording could not be loaded as configured."""


@dataclass(frozen=True)
class Provenance:
    """What was loaded, and what was done to it.

    Every field is either read from the source or decided by `LoadConfig`.
    Nothing here is inferred.
    """

    #: Absolute path, or a description such as ``"synthetic"`` for generated data.
    source: str
    #: SHA-256 of the source file, or ``None`` for non-file sources.  Makes a
    #: result traceable to exact input bytes, which matters because `data/` is
    #: gitignored: the files that produced a result are not in the repository,
    #: so the hash is the only link back to them.
    source_sha256: str | None
    sfreq_hz: float
    n_samples: int
    duration_seconds: float

    #: Channel names exactly as the source spelled them, before canonicalisation.
    source_channel_names: tuple[str, ...]
    #: Canonical names of the channels retained, in array row order.
    channel_names: tuple[str, ...]
    #: ``{canonical_or_raw_name: reason}`` for every channel dropped.
    dropped_channels: dict[str, str]

    #: What `LoadConfig.reference` actually did: ``"none"``, or
    #: ``"average over N channels: ..."``.
    reference_applied: str
    #: Multiplier applied to the source data (1e6, or 1.0 if already microvolts).
    scale_factor: float
    #: Montage name attached, or ``"none"``.
    montage: str
    #: Retained channels the montage had no position for.  Non-empty means any
    #: later topographic or ICA work is working with fewer electrodes than it
    #: appears to.
    montage_missing_positions: tuple[str, ...]

    #: Filter settings the source file's own header reports, when it has them.
    #: EEGMAT's header says 0.5-45 Hz, which is why this pipeline's own
    #: bandpass is close to a no-op there (see `FilterConfig.for_eegmat`).
    source_highpass_hz: float | None = None
    source_lowpass_hz: float | None = None
    source_meas_date: str | None = None

    loaded_utc: str = ""
    #: Library versions that produced the array.  `None` where a library was
    #: not used (synthetic input needs no MNE).
    mne_version: str | None = None
    numpy_version: str = ""

    #: Filter stages applied to the data, in order, as human-readable
    #: descriptions.  Written by `eeg.filtering.apply_filters`, so a result
    #: carries its own filter history even when separated from its config.
    #: Empty means unfiltered.
    filters_applied: tuple[str, ...] = ()

    #: Non-fatal degradations, in the order they happened.  A run that
    #: proceeded short of its ideal input must say so here.
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        out = asdict(self)
        out["source_channel_names"] = list(self.source_channel_names)
        out["channel_names"] = list(self.channel_names)
        out["montage_missing_positions"] = list(self.montage_missing_positions)
        out["filters_applied"] = list(self.filters_applied)
        out["warnings"] = list(self.warnings)
        return out


@dataclass
class Recording:
    """A continuous multi-channel recording in microvolts.

    `data` is ``(n_channels, n_samples)``, float64, row order matching
    `channel_names`.  Mutable on purpose: `eeg.filtering` returns a new
    `Recording` rather than editing one, but the demo and tests slice these
    freely and a frozen container would force copies for no benefit.
    """

    data: np.ndarray
    channel_names: tuple[str, ...]
    sfreq: float
    provenance: Provenance
    #: ``{channel: (x, y, z)}`` in metres, from the montage.  Empty if
    #: ``montage="none"`` or no position was found.
    positions: dict[str, tuple[float, float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError(f"data must be 2-D (n_channels, n_samples), got shape {self.data.shape}")
        if self.data.shape[0] != len(self.channel_names):
            raise ValueError(
                f"data has {self.data.shape[0]} rows but {len(self.channel_names)} channel "
                "names; row order and names must correspond")
        if self.sfreq <= 0:
            raise ValueError(f"sfreq must be > 0, got {self.sfreq}")

    @property
    def n_channels(self) -> int:
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]

    @property
    def duration_seconds(self) -> float:
        return self.n_samples / self.sfreq

    def index_of(self, channel: str) -> int:
        """Row index of a channel, by canonical name.

        Raises `KeyError` rather than returning -1: a silently wrong row index
        would compute a correct-looking band power for the wrong electrode.
        """
        canon = canonicalise(channel)
        try:
            return self.channel_names.index(canon)
        except ValueError:
            raise KeyError(
                f"channel {channel!r} (canonical {canon!r}) not in this recording; "
                f"have {list(self.channel_names)}") from None

    def pick(self, channels: tuple[str, ...] | list[str]) -> np.ndarray:
        """Rows for `channels`, in the order given.  Raises on a missing one."""
        return self.data[[self.index_of(c) for c in channels], :]


# ----------------------------------------------------------------------
# Core preparation, array-in / Recording-out
# ----------------------------------------------------------------------


def prepare(
    data: np.ndarray,
    channel_names: list[str] | tuple[str, ...],
    sfreq: float,
    config: LoadConfig | None = None,
    *,
    source: str = "array",
    source_sha256: str | None = None,
    already_microvolts: bool = False,
    source_highpass_hz: float | None = None,
    source_lowpass_hz: float | None = None,
    source_meas_date: str | None = None,
    mne_version: str | None = None,
) -> Recording:
    """Apply a `LoadConfig` to raw channel data and record what was done.

    This is where channel canonicalisation, non-scalp exclusion, explicit
    channel selection, unit scaling, re-referencing and montage attachment
    happen — in that order, and the order matters:

    1. **Canonicalise names**, so everything after this works on scalp sites.
    2. **Drop non-scalp channels**, so an ECG trace cannot reach the average
       reference or a channel average.
    3. **Apply explicit selection**, after the drop, so asking for a channel
       the file does not have is reported once rather than twice.
    4. **Scale to microvolts**, before referencing, so a reference subtraction
       is visible in the unit the thresholds use.
    5. **Re-reference**, over the retained scalp channels only.
    6. **Attach positions**, which changes no sample.

    `already_microvolts=True` is for synthetic test signals constructed
    directly in microvolts.  It sets `scale_factor` to 1.0 in provenance rather
    than lying about having scaled.

    Raises `LoadError` if `config.require_channels` names a site the recording
    does not have, or if a configured montage cannot be built.
    """
    config = config or LoadConfig()
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D (n_channels, n_samples), got shape {data.shape}")
    if data.shape[0] != len(channel_names):
        raise ValueError(
            f"data has {data.shape[0]} rows but {len(channel_names)} channel names")

    source_names = tuple(str(n) for n in channel_names)
    warnings: list[str] = []
    dropped: dict[str, str] = {}

    # 1. Canonicalise.
    canon = [canonicalise(n) for n in source_names]
    duplicates = {n for n in canon if canon.count(n) > 1}
    if duplicates:
        # Two rows claiming the same site cannot both be that site, and a
        # later `index_of` would silently return only the first.
        raise LoadError(
            f"channel names collide after canonicalisation: {sorted(duplicates)}; "
            "two rows cannot be the same scalp site")

    keep = list(range(len(canon)))

    # 2. Non-scalp exclusion.
    if config.drop_non_scalp:
        kept: list[int] = []
        for i, name in enumerate(canon):
            kind = classify(source_names[i])
            if kind == "scalp":
                kept.append(i)
            elif kind == "reference":
                dropped[name] = "reference electrode, not a cortical site"
            elif is_modality_channel(source_names[i]):
                dropped[name] = "non-EEG modality channel"
            else:
                dropped[name] = "not a recognised scalp site (bipolar derivation or unknown)"
        keep = kept

    # 3. Explicit selection, applied to what survived step 2.
    available = [canon[i] for i in keep]
    if config.channels:
        from eeg.channels import resolve
        wanted = resolve(config.channels, available)
        gone = missing(config.channels, available)
        if gone:
            warnings.append(
                f"LoadConfig.channels asked for {gone}, which this recording does not have")
        index_by_name = {canon[i]: i for i in keep}
        for name in available:
            if name not in wanted:
                dropped.setdefault(name, "not in LoadConfig.channels")
        keep = [index_by_name[n] for n in wanted]

    if not keep:
        raise LoadError(
            "no channels left after selection; "
            f"source had {list(source_names)}, dropped {dropped}")

    names = tuple(canon[i] for i in keep)
    out = data[keep, :].copy()

    # `require_channels` is checked here, after selection, so the error names
    # what is actually absent rather than what was filtered out on the way.
    if config.require_channels:
        absent = missing(config.require_channels, list(names))
        if absent:
            raise LoadError(
                f"required channel(s) {absent} absent; recording has {list(names)}")

    # 4. Units.
    scale = 1.0 if already_microvolts or not config.scale_to_microvolts else VOLTS_TO_MICROVOLTS
    if scale != 1.0:
        out *= scale

    # 5. Reference.
    if config.reference == "average":
        # Over the retained scalp channels only — step 2 has already removed
        # anything that is not one, and `LoadConfig.validate` refuses the
        # combination that would skip that removal.
        out = out - out.mean(axis=0, keepdims=True)
        reference_applied = f"average over {len(names)} channels: {', '.join(names)}"
        if len(names) < 16:
            # Not an error: an 8-channel average reference is a defensible
            # choice, but it is a *biased* one, and the bias lands on exactly
            # the midline sites the index uses.
            warnings.append(
                f"average reference computed over only {len(names)} channels; with sparse, "
                "non-uniform coverage the average contains the midline activity it is "
                "subtracted from, which shrinks midline effects toward zero")
    else:
        reference_applied = "none (source reference retained)"

    # 6. Positions.
    positions: dict[str, tuple[float, float, float]] = {}
    montage_missing: tuple[str, ...] = ()
    montage_used = config.montage
    if config.montage != "none":
        positions, montage_missing, montage_used = _montage_positions(config.montage, names)
        if montage_missing:
            warnings.append(
                f"montage {config.montage!r} has no position for {list(montage_missing)}; "
                "band power is unaffected, but ICA and any topographic figure would be "
                "working with fewer electrodes than they appear to")

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    provenance = Provenance(
        source=source,
        source_sha256=source_sha256,
        sfreq_hz=float(sfreq),
        n_samples=int(out.shape[1]),
        duration_seconds=float(out.shape[1] / sfreq),
        source_channel_names=source_names,
        channel_names=names,
        dropped_channels=dropped,
        reference_applied=reference_applied,
        scale_factor=scale,
        montage=montage_used,
        montage_missing_positions=montage_missing,
        source_highpass_hz=source_highpass_hz,
        source_lowpass_hz=source_lowpass_hz,
        source_meas_date=source_meas_date,
        loaded_utc=now,
        mne_version=mne_version,
        numpy_version=np.__version__,
        warnings=tuple(warnings),
    )
    return Recording(data=out, channel_names=names, sfreq=float(sfreq),
                     provenance=provenance, positions=positions)


def _montage_positions(
    montage: str, names: tuple[str, ...]
) -> tuple[dict[str, tuple[float, float, float]], tuple[str, ...], str]:
    """Sensor positions for `names` from a standard montage.

    Returns ``(positions, missing, resolved_name)``.  MNE is imported here
    rather than at module scope so that the array-only path — synthetic
    signals in tests, and any future non-EDF source — needs neither MNE nor a
    montage.

    MNE 1.13 deprecated ``"standard_1020"`` in favour of ``"colin27_1020"``
    for removal in 1.14; the positions are the same, only the name changed.
    Each accepted name therefore has an alias list and the first one the
    installed MNE provides wins, with the name actually used returned so it
    lands in provenance.  Trying the alias is not cosmetic: pinning to either
    spelling would make this package fail outright on one side of that
    version boundary, and failing on a *name* would look exactly like failing
    on the data.
    """
    try:
        import mne
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise LoadError(
            f"montage={montage!r} needs mne installed; use montage='none' for "
            "array input that does not require sensor geometry") from exc

    from eeg.config import MONTAGE_ALIASES
    candidates = MONTAGE_ALIASES.get(montage, (montage,))
    mon = None
    resolved = montage
    failures: list[str] = []
    for candidate in candidates:
        try:
            with warnings.catch_warnings():
                # The deprecation is handled by this very alias list, so
                # re-reporting it on every load would be noise the caller can
                # do nothing about.
                warnings.simplefilter("ignore", FutureWarning)
                mon = mne.channels.make_standard_montage(candidate)
            resolved = candidate
            break
        except Exception as exc:
            failures.append(f"{candidate}: {exc}")
    if mon is None:
        raise LoadError(
            f"could not build montage {montage!r} under any known name; tried "
            + "; ".join(failures))
    ch_pos = mon.get_positions()["ch_pos"]
    positions: dict[str, tuple[float, float, float]] = {}
    absent: list[str] = []
    for name in names:
        pos = ch_pos.get(name)
        if pos is None:
            absent.append(name)
        else:
            positions[name] = (float(pos[0]), float(pos[1]), float(pos[2]))
    return positions, tuple(absent), resolved


# ----------------------------------------------------------------------
# EDF
# ----------------------------------------------------------------------


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, streamed.  Used for provenance, not for security."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def load_edf(path: str | Path, config: LoadConfig | None = None) -> Recording:
    """Load one EDF recording and apply `config`.

    MNE is imported lazily: the rest of this package works on arrays, so only
    this function needs it, and a test suite that never touches an EDF never
    needs MNE installed.

    Note on channel types: MNE marks every EEGMAT channel ``eeg``, including
    ``"ECG ECG"`` and the bipolar ``"EEG A2-A1"``, because the EDF header says
    nothing finer.  Channel *type* is therefore not trustworthy here and is not
    consulted — `eeg.channels.classify` decides from the name instead.
    """
    path = Path(path)
    if not path.exists():
        raise LoadError(f"no such recording: {path}")
    try:
        import mne
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise LoadError("reading EDF needs mne installed: pip install mne") from exc

    previous = mne.set_log_level("ERROR", return_old_level=True)
    try:
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    except Exception as exc:
        raise LoadError(f"could not read {path}: {exc}") from exc
    finally:
        mne.set_log_level(previous)

    meas_date = raw.info.get("meas_date")
    return prepare(
        raw.get_data(),                       # volts
        list(raw.ch_names),
        float(raw.info["sfreq"]),
        config,
        source=str(path.resolve()),
        source_sha256=sha256_file(path),
        already_microvolts=False,
        source_highpass_hz=_opt_float(raw.info.get("highpass")),
        source_lowpass_hz=_opt_float(raw.info.get("lowpass")),
        source_meas_date=meas_date.isoformat() if meas_date is not None else None,
        mne_version=mne.__version__,
    )


def _opt_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
