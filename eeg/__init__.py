"""EEG preprocessing for the neuroadaptive-curriculum capstone.

Stages, each in its own module and each usable alone:

* `eeg.loading` — EDF or array in, a float64 microvolt `Recording` plus a
  `Provenance` record out.  No MNE object crosses this boundary.
* `eeg.filtering` — bandpass and mains notch, every cutoff an argument.
* `eeg.windowing` — fixed-length analysis windows, the unit of measurement
  for everything after this point.
* `eeg.artifacts` — a per-window clean/reject mask with its reasons.  Marks,
  never modifies: the design is "natural behaviour -> artifact detection ->
  robust processing", not movement suppression.
* `eeg.spectral` — Welch PSD and band power, delivered as `WindowFeatures`.
* `eeg.pipeline` — `EEGPipeline`, which composes the five and records every
  parameter it used.
* `eeg.config` — every parameter, in frozen dataclasses.  **None of the
  defaults is a settled decision**; see `eeg/README.md`, "Open items".

Where this package stops
------------------------
At `WindowFeatures`.  The workload index of pre-registration §7 — standardised
frontal midline theta minus down-weighted parietal alpha — is not implemented
here; `eeg.pipeline.WorkloadIndex` is the interface it plugs into and
`eeg/README.md` states the contract.

Language discipline (`docs/DECISIONS.md` D025): what this package produces is
an **EEG-derived spectral feature**.  It is not a measurement of workload, of
cognitive load, or of any mental state.  Whether any function of these features
orders task difficulty usefully is the experiment's question, and Gate 3 is
where it gets answered.
"""

from __future__ import annotations

from eeg.artifacts import (ArtifactError, ArtifactReport, detect_artifacts,
                           ocular_reference_windows, rejection_rate_by_group)
from eeg.channels import ROIS, canonicalise, classify, is_scalp_site, resolve
from eeg.config import (ArtifactConfig, ConfigError, EpochConfig, FilterConfig, LoadConfig,
                        PipelineConfig, SpectralConfig)
from eeg.filtering import FilterError, apply_filters, bandpass, notch
from eeg.loading import LoadError, Provenance, Recording, load_edf, prepare
from eeg.pipeline import (EEGPipeline, PipelineError, PipelineResult, WorkloadIndex,
                          apply_index, default_pipeline)
from eeg.spectral import (PSD, SpectralError, WindowFeatures, band_power, extract_features,
                          welch_psd)
from eeg.windowing import WindowError, Windows, make_windows

__all__ = [
    # config
    "ArtifactConfig", "ConfigError", "EpochConfig", "FilterConfig", "LoadConfig",
    "PipelineConfig", "SpectralConfig",
    # channels
    "ROIS", "canonicalise", "classify", "is_scalp_site", "resolve",
    # loading
    "LoadError", "Provenance", "Recording", "load_edf", "prepare",
    # filtering
    "FilterError", "apply_filters", "bandpass", "notch",
    # windowing
    "WindowError", "Windows", "make_windows",
    # artifacts
    "ArtifactError", "ArtifactReport", "detect_artifacts", "ocular_reference_windows",
    "rejection_rate_by_group",
    # spectral
    "PSD", "SpectralError", "WindowFeatures", "band_power", "extract_features", "welch_psd",
    # pipeline
    "EEGPipeline", "PipelineError", "PipelineResult", "WorkloadIndex", "apply_index",
    "default_pipeline",
]
