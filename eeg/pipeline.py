"""The composable pipeline: load -> filter -> window -> detect -> features.

One object, `EEGPipeline`, holds one `PipelineConfig` and runs the five stages
in order.  One object, `PipelineResult`, carries everything that came out —
including the full config, the loading provenance, each stage's parameters, and
the artifact report — so that a result can be reproduced, and criticised, from
its own contents.

Reproducibility
---------------
The pipeline contains **no randomness**.  Equal config plus equal input bytes
gives bit-identical band powers, and `tests/test_eeg_pipeline.py` asserts that
rather than taking it on trust.  There is therefore no seed to record: a seed
would imply a stochastic step that does not exist.  The one non-deterministic
field in a result is `timing_seconds`, which is wall clock and is excluded from
any equality check.

Callers that *do* randomise own their own seed — `scripts/eeg_demo.py` takes
`--seed` for choosing which subjects to run and records it in its output.

Stage order, and why
--------------------
1. **Filter** the continuous recording, not the windows.  A filter applied per
   window leaves a transient at both of its edges, and the first and last
   fraction of a second of every window would then be filter response rather
   than signal.
2. **Window** the filtered, edge-trimmed recording.
3. **Band-limit for the ocular detector** from the same continuous filtered
   recording, then cut it with the identical segmentation, so the ocular
   amplitudes have no per-window transient either.
4. **Detect artifacts** and produce the mask.
5. **Extract features** for *every* window, clean or not, and attach the mask.
   A consumer handed only clean windows cannot see the rejection rate, and
   pre-registration §7 requires it.

Where this package stops
------------------------
At `WindowFeatures`.  The workload index — standardised frontal midline theta
minus down-weighted parietal alpha, pre-registration §7 — is **not implemented
here**.  It is the author's to write (CLAUDE.md, "the user writes the
learning-critical code"), and `WorkloadIndex` below is the interface it plugs
into.  `eeg/README.md` spells out the contract.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from eeg.artifacts import ArtifactReport, detect_artifacts, ocular_reference_windows
from eeg.config import PipelineConfig
from eeg.filtering import apply_filters
from eeg.loading import LoadConfig, Provenance, Recording, load_edf, prepare
from eeg.spectral import WindowFeatures, extract_features
from eeg.windowing import Windows, make_windows

#: Bumped when `PipelineResult.to_dict()`'s layout changes incompatibly.
RESULT_SCHEMA_VERSION = 1


class PipelineError(RuntimeError):
    """The pipeline could not complete.  The stage is named in the message."""


@dataclass(frozen=True)
class StageRecord:
    """One stage's parameters and what it produced.

    Kept per stage rather than only as a final config, because the useful
    question when a result looks wrong is "how many windows did windowing
    produce, and how many survived detection" — and that is answered by the
    shapes, not by the parameters.
    """

    name: str
    params: dict[str, Any]
    output: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "params": self.params, "output": self.output}


@dataclass
class PipelineResult:
    """Everything one pipeline run produced."""

    features: WindowFeatures
    artifacts: ArtifactReport
    windows: Windows
    recording: Recording
    config: PipelineConfig
    stages: tuple[StageRecord, ...]
    #: Wall-clock seconds per stage.  The only non-deterministic content of a
    #: result; excluded from any equality comparison.
    timing_seconds: dict[str, float] = field(default_factory=dict)
    schema_version: int = RESULT_SCHEMA_VERSION

    @property
    def provenance(self) -> Provenance:
        return self.recording.provenance

    @property
    def warnings(self) -> tuple[str, ...]:
        """Every non-fatal degradation from every stage, in stage order."""
        return tuple(self.recording.provenance.warnings) + tuple(self.artifacts.warnings)

    def quality_summary(self) -> dict[str, Any]:
        """Signal quality and rejection, as a demo or a report would print it.

        Per-channel median peak-to-peak over **clean** windows is the headline
        number: it is the quantity the rejection threshold is compared against,
        so seeing it next to the threshold shows how much headroom a recording
        had.  A channel whose median p-p sits just under the threshold is one
        bad minute away from being rejected wholesale, and that is worth
        knowing before a session rather than after.
        """
        data = self.windows.pick(list(self.artifacts.examined_channels))
        ptp = data.max(axis=2) - data.min(axis=2)              # (n_win, n_ch)
        clean = self.artifacts.clean
        per_channel: dict[str, dict[str, float]] = {}
        for i, name in enumerate(self.artifacts.examined_channels):
            column = ptp[:, i]
            usable = column[clean] if np.any(clean) else column[:0]
            per_channel[name] = {
                "median_ptp_uv": float(np.median(usable)) if usable.size else float("nan"),
                "p95_ptp_uv": float(np.percentile(usable, 95)) if usable.size else float("nan"),
                "max_ptp_uv": float(column.max()) if column.size else float("nan"),
                "reject_rate": self.artifacts.rate_by_channel().get(name, float("nan")),
            }
        return {
            "source": self.provenance.source,
            "sfreq_hz": self.provenance.sfreq_hz,
            "duration_seconds": self.provenance.duration_seconds,
            "n_channels": len(self.recording.channel_names),
            "channels": list(self.recording.channel_names),
            "dropped_channels": dict(self.provenance.dropped_channels),
            "reference_applied": self.provenance.reference_applied,
            "filters_applied": list(self.provenance.filters_applied),
            "n_windows": self.artifacts.n_windows,
            "n_clean_windows": self.artifacts.n_clean,
            "rejection_rate": self.artifacts.rejection_rate,
            "rate_by_reason": self.artifacts.rate_by_reason(),
            "peak_to_peak_threshold_uv": self.config.artifact.peak_to_peak_uv,
            "per_channel": per_channel,
            "warnings": list(self.warnings),
        }

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        """JSON-safe form.

        `include_arrays=False` by default: band powers for a 3-minute recording
        are a few thousand floats, which belong in a `.npz` next to the JSON
        rather than inside it.  `save` writes both.
        """
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "config": self.config.to_dict(),
            "provenance": self.provenance.to_dict(),
            "stages": [s.to_dict() for s in self.stages],
            "artifacts": self.artifacts.summary(),
            "features": self.features.to_summary(),
            "quality": self.quality_summary(),
            "timing_seconds": dict(self.timing_seconds),
        }
        if include_arrays:
            out["arrays"] = {
                "band_power_uv2": self.features.power.tolist(),
                "clean": self.features.clean.tolist(),
                "start_seconds": self.features.start_seconds.tolist(),
            }
        return out

    def save(self, directory: str | Path, *, name: str = "eeg_result") -> dict[str, Path]:
        """Write ``<name>.json`` (metadata) and ``<name>.npz`` (arrays).

        Returns the paths written.  Arrays go to `.npz` because they are the
        bulk of the output and because a saved `.npz` can be reloaded for a
        different threshold or a different index without re-running the
        pipeline — the band powers do not change when the index does.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / f"{name}.json"
        npz_path = directory / f"{name}.npz"
        json_path.write_text(
            json.dumps(self.to_dict(), indent=2, allow_nan=True) + "\n", encoding="utf-8")
        arrays: dict[str, np.ndarray] = {
            "band_power_uv2": self.features.power,
            "clean": self.features.clean,
            "start_seconds": self.features.start_seconds,
            "ocular_z": self.artifacts.ocular_z,
            "channel_names": np.asarray(self.features.channel_names),
            "band_names": np.asarray(self.features.band_names),
        }
        if self.features.relative is not None:
            arrays["relative_power"] = self.features.relative
        np.savez_compressed(npz_path, **arrays)
        return {"json": json_path, "npz": npz_path}


# ----------------------------------------------------------------------
# The index boundary
# ----------------------------------------------------------------------


@runtime_checkable
class WorkloadIndex(Protocol):
    """The interface the workload index implements.  Not implemented here.

    Pre-registration §7 specifies it: standardised frontal midline theta
    (4-8 Hz, Fz with F3/F4) minus down-weighted standardised parietal alpha
    (8-13 Hz, Pz with P3/P4), weighted by the meta-analytic effect sizes in
    Chikhi et al. (2022), standardised within session.  Writing it is the
    author's (CLAUDE.md: "the user writes ... the workload index
    computation"), and this package stops short of it on purpose.

    Contract
    --------
    * **Input** is a `WindowFeatures`: band power ``(n_windows, n_channels,
      n_bands)`` in uV^2, with `clean` the artifact mask, and the accessors
      `roi_series`, `band_series`, `channel_index`, `band_index`,
      `clean_index` and `window_times`.
    * **Output** is a float array of shape ``(n_windows,)``, one index value
      per window, in window order — *including* rejected windows, whose values
      may be anything (NaN is the clearest choice) because the mask says not
      to use them.  Returning only clean windows would break the alignment
      between an index value and the configuration block it came from.
    * The index must **not** modify the `WindowFeatures` it is given.
    * The index must not consult anything outside its input.  In particular it
      must not see the agent's observations, actions or rewards — the
      architectural invariant in pre-registration §1 runs in the other
      direction, and a dependency here would be the one place it could be
      violated without the agent ever receiving anything.

    `apply_index` below checks the output's shape and dtype so a contract
    violation is an error at the boundary rather than a strange correlation
    three stages later.
    """

    def __call__(self, features: WindowFeatures) -> np.ndarray:
        ...


def apply_index(result: PipelineResult, index: WorkloadIndex) -> np.ndarray:
    """Run a `WorkloadIndex` over a result's features and validate its output.

    Returns the index array.  Raises `PipelineError` on a contract violation:

    * wrong length — an index that returned only its clean windows would
      silently misalign every value against its configuration block;
    * non-float dtype — an integer index would floor the standardised values;
    * a non-finite value at a **clean** window — the mask says that window is
      usable, so a NaN there is a defect in the index, not a rejected window.
      Non-finite values at rejected windows are expected and allowed.

    This function deliberately does **not** compute a median, a rank, or a
    correlation.  Those are the analysis (pre-registration §6 and §9) and are
    the author's to write.
    """
    values = index(result.features)
    values = np.asarray(values)
    n = result.features.n_windows
    if values.shape != (n,):
        raise PipelineError(
            f"workload index returned shape {values.shape}, expected ({n},) — one value "
            "per window including rejected ones. Returning only clean windows breaks the "
            "alignment between an index value and the block it was recorded in.")
    if not np.issubdtype(values.dtype, np.floating):
        raise PipelineError(
            f"workload index returned dtype {values.dtype}; a standardised index is "
            "real-valued and an integer dtype would floor it")
    clean = result.features.clean
    if np.any(clean) and not np.all(np.isfinite(values[clean])):
        bad = np.flatnonzero(clean & ~np.isfinite(values))
        raise PipelineError(
            f"workload index returned non-finite values at clean window(s) "
            f"{bad[:10].tolist()}{' ...' if bad.size > 10 else ''}. The mask says those "
            "windows are usable, so this is a defect in the index rather than a rejection.")
    return values


# ----------------------------------------------------------------------
# The pipeline
# ----------------------------------------------------------------------


class EEGPipeline:
    """Load, filter, window, detect artifacts, extract band power.

    >>> pipe = EEGPipeline(PipelineConfig.for_eegmat())        # doctest: +SKIP
    >>> result = pipe.run_edf("data/raw/eegmat/Subject00_1.edf")  # doctest: +SKIP
    >>> result.artifacts.rejection_rate                        # doctest: +SKIP

    The config is held, not copied per call, and is frozen, so one pipeline
    instance processes many recordings under provably identical parameters.
    That is the point of the object: a per-subject loop that rebuilt its
    parameters each iteration could drift between subjects without anything
    in the output showing it.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = (config or PipelineConfig()).validate()

    # -- entry points ---------------------------------------------------

    def run_edf(self, path: str | Path) -> PipelineResult:
        """Load an EDF with `config.load`, then `run`."""
        recording = load_edf(path, self.config.load)
        return self.run(recording)

    def run_array(
        self,
        data: np.ndarray,
        channel_names: list[str] | tuple[str, ...],
        sfreq: float,
        *,
        source: str = "array",
        already_microvolts: bool = True,
    ) -> PipelineResult:
        """Run on an in-memory array, for synthetic signals and tests.

        `already_microvolts` defaults True here, the opposite of `load_edf`'s
        volts: a synthetic signal is constructed in whatever unit the author
        chose, and for every test in this package that unit is microvolts.
        The resulting `scale_factor` in provenance says which was assumed.
        """
        recording = prepare(
            data, channel_names, sfreq, self.config.load,
            source=source, already_microvolts=already_microvolts)
        return self.run(recording)

    # -- the stages -----------------------------------------------------

    def run(self, recording: Recording) -> PipelineResult:
        """Run every stage on an already-loaded `Recording`.

        Each stage's failure is re-raised as a `PipelineError` naming the
        stage, because the underlying exceptions (`FilterError`,
        `WindowError`, `ArtifactError`, `SpectralError`) are accurate but do
        not say where in a five-stage run they came from.
        """
        config = self.config
        stages: list[StageRecord] = []
        timing: dict[str, float] = {}

        stages.append(StageRecord(
            name="load",
            params=_section_dict(config.load),
            output={
                "n_channels": recording.n_channels,
                "channels": list(recording.channel_names),
                "sfreq_hz": recording.sfreq,
                "duration_seconds": recording.duration_seconds,
                "dropped_channels": dict(recording.provenance.dropped_channels),
                "reference_applied": recording.provenance.reference_applied,
            }))

        # 1. Filter the continuous recording.
        t0 = time.perf_counter()
        filtered = _stage("filter", lambda: apply_filters(recording, config.filter))
        timing["filter"] = time.perf_counter() - t0
        stages.append(StageRecord(
            name="filter",
            params=_section_dict(config.filter),
            output={
                "stages_applied": list(filtered.provenance.filters_applied),
                "duration_seconds": filtered.duration_seconds,
                "samples_removed_by_trim": recording.n_samples - filtered.n_samples,
            }))

        # 2. Window it.  The time offset is the edge trim, known here exactly,
        #    so window times stay convertible to original-recording time.
        t0 = time.perf_counter()
        windows = _stage("window", lambda: make_windows(
            filtered, config.epoch, time_offset=config.filter.edge_trim_seconds))
        timing["window"] = time.perf_counter() - t0
        stages.append(StageRecord(
            name="window",
            params=_section_dict(config.epoch),
            output={
                "n_windows": windows.n_windows,
                "window_samples": windows.n_samples,
                "window_seconds": windows.window_seconds,
                "hop_seconds": windows.hop_seconds,
                "time_offset_seconds": windows.time_offset,
                "dropped_tail_samples": windows.dropped_tail_samples,
            }))

        # 3 and 4. Ocular reference from the same continuous signal, then detect.
        t0 = time.perf_counter()
        ocular = _stage("artifact", lambda: ocular_reference_windows(
            filtered, config.epoch, config.artifact))
        report = _stage("artifact", lambda: detect_artifacts(
            windows, config.artifact, ocular_windows=ocular))
        timing["artifact"] = time.perf_counter() - t0
        stages.append(StageRecord(
            name="artifact",
            params=_section_dict(config.artifact),
            output={
                "ocular_source": "continuous" if ocular is not None else "none",
                **report.summary(),
            }))

        # 5. Features for every window, mask attached.
        t0 = time.perf_counter()
        features = _stage("spectral", lambda: extract_features(
            windows, config.spectral, clean=report.clean))
        timing["spectral"] = time.perf_counter() - t0
        stages.append(StageRecord(
            name="spectral",
            params=_section_dict(config.spectral),
            output={
                "n_windows": features.n_windows,
                "n_clean": features.n_clean,
                "bands": list(features.band_names),
                "channels": list(features.channel_names),
                "power_shape": list(features.power.shape),
            }))

        return PipelineResult(
            features=features,
            artifacts=report,
            windows=windows,
            recording=filtered,
            config=config,
            stages=tuple(stages),
            timing_seconds=timing,
        )


def _stage(name: str, fn: Any) -> Any:
    try:
        return fn()
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError(f"{name} stage failed: {type(exc).__name__}: {exc}") from exc


def _section_dict(section: Any) -> dict[str, Any]:
    """One config section as plain JSON, tuples flattened to lists."""
    from dataclasses import asdict

    from eeg.config import _jsonify
    return _jsonify(asdict(section))


def default_pipeline(dataset: str = "eegmat") -> EEGPipeline:
    """A pipeline preset by dataset name.

    ``"eegmat"`` uses `PipelineConfig.for_eegmat()` (50 Hz mains);
    ``"default"`` uses `PipelineConfig()` (60 Hz mains, for the author's own
    hardware once D022 is settled).  A name that is neither is an error rather
    than a fallback — falling back to 60 Hz on EEGMAT would leave its mains
    line un-notched while reporting that a notch ran.
    """
    if dataset == "eegmat":
        return EEGPipeline(PipelineConfig.for_eegmat())
    if dataset == "default":
        return EEGPipeline(PipelineConfig())
    raise PipelineError(
        f"unknown dataset preset {dataset!r}; expected 'eegmat' or 'default'. "
        "There is no fallback on purpose: the presets differ in mains frequency, and "
        "guessing wrong leaves the line artifact in place.")


__all__ = [
    "EEGPipeline",
    "PipelineError",
    "PipelineResult",
    "RESULT_SCHEMA_VERSION",
    "StageRecord",
    "WorkloadIndex",
    "apply_index",
    "default_pipeline",
    "LoadConfig",
]
