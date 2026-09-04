# Data Directory

## Purpose

This directory contains documentation and, when appropriate, small configuration/metadata files describing project datasets.

---

# Raw EEG data

Raw EEG should **not** be committed to this repository by default.

Reasons include:

- file size
- privacy
- licensing
- consent/data-use restrictions
- reproducibility requirements that may be better served by dataset provenance instructions

The `.gitignore` intentionally excludes common EEG file formats and `data/raw/`.

---

# Dataset provenance

For every external EEG dataset used, document:

- dataset name
- source
- URL or persistent identifier
- version/date accessed
- license
- participant/task description
- sampling rate
- channel information
- relevant labels
- preprocessing already applied by the dataset provider
- any restrictions on redistribution

Do not claim that a dataset is suitable until its actual contents and licensing have been checked.

---

# Local data layout

A suggested local structure is:

```text
data/
├── raw/
├── interim/
├── processed/
├── metadata/
└── README.md
```

Only `metadata/` and small, non-sensitive project configuration files should normally be version-controlled.

---

# EEG recording data

If live EEG is eventually collected, document:

- device
- firmware/software version
- channel count
- sampling rate
- electrode/channel configuration
- timestamps
- synchronization method
- task protocol
- environment configuration
- participant/session identifier
- artifact annotations
- preprocessing version

Raw personal EEG should remain in an appropriately secured location.

---

# Reproducibility

Every experiment using EEG should record the exact data source and preprocessing configuration.

A training run should never depend on an undocumented local dataset.

Where redistribution is prohibited, provide instructions for obtaining the dataset instead of committing the data itself.
