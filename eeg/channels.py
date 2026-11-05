"""Channel naming, scalp-site recognition, and regions of interest.

Why this module exists
----------------------
Two recording systems feed the same code, and they disagree about names.

* **EEGMAT** (the development fixture, D032) labels its EDF channels
  ``"EEG Fz"``, ``"EEG Fp1"``, ``"EEG A2-A1"``, ``"ECG ECG"`` — a instrument
  prefix, a bipolar reference derivation, and an ECG trace, all with
  ``eeg`` as their MNE channel type because the EDF header says nothing finer.
* **The author's own hardware** (OpenBCI, still `D022` OPEN) labels channels by
  board input number unless the acquisition software is told otherwise.

Everything downstream — the artifact detector's ocular channels, the spectral
band extraction, and eventually the workload index's Fz/F3/F4 and Pz/P3/P4
terms — selects channels *by scalp site*. So the site has to be recovered from
whatever the file happens to call it, once, at load time, and recorded.

What fails if you get this wrong
--------------------------------
Silently, and in the direction that looks like a result.

* ``"EEG A2-A1"`` is a *difference between two ear electrodes*, not a scalp
  site. Left in an average reference it contaminates every channel. Left in a
  "mean over frontal channels" it adds a trace with completely different
  statistics.
* ``"ECG ECG"`` is a cardiac trace at roughly 100x scalp amplitude. One ECG
  channel inside an amplitude-threshold artifact detector rejects every window
  in the recording; inside a band-power average it dominates the result.
* Case matters to MNE's montage matching: ``"FZ"`` does not match the
  ``standard_1020`` montage's ``"Fz"``, so a channel named ``"FZ"`` silently
  gets no sensor position. Nothing errors — a later topographic plot or ICA
  just has one fewer electrode than you think.

So recognition here is explicit and allow-list based: a channel is a scalp site
only if its canonical spelling is a name this module knows. Anything else is
kept in the recording, labelled non-scalp, and excluded from EEG selections,
rather than guessed at.
"""

from __future__ import annotations

import re

# ----------------------------------------------------------------------
# Canonical scalp sites
# ----------------------------------------------------------------------

# Every 10-20 and extended 10-20 site this project might see, spelled the way
# MNE's "standard_1020" montage spells it.  The spelling is what makes montage
# attachment work, so this tuple is the authority on capitalisation.
#
# T3/T4/T5/T6 are the pre-1991 names for T7/T8/P7/P8.  EEGMAT uses the old
# names; modern caps use the new ones.  Both are listed because
# "standard_1020" carries positions for both, so neither needs renaming — but
# note that T3 and T7 are then *different entries* here and a selection asking
# for "T7" will not match an EEGMAT file.  That is deliberate: a silent alias
# would mean a selection could match a site the file does not actually contain.
CANONICAL_SITES: tuple[str, ...] = (
    # Frontopolar / frontal
    "Fp1", "Fpz", "Fp2",
    "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    # Fronto-central / central
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T3", "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "T4",
    # Centro-parietal / parietal
    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
    "T5", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "T6",
    # Parieto-occipital / occipital
    "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2",
    # Mastoids / earlobes, used as references rather than as signal sites.
    "A1", "A2", "M1", "M2",
)

# Lookup from an upper-cased spelling to the canonical one.
_BY_UPPER: dict[str, str] = {site.upper(): site for site in CANONICAL_SITES}

# Earlobe/mastoid electrodes are real sites with real positions, but they are
# reference locations, not cortical measurement sites.  They are recognised
# (so a montage can place them) and excluded from signal selections.
REFERENCE_SITES: frozenset[str] = frozenset({"A1", "A2", "M1", "M2"})

# Instrument prefixes EDF writers put in front of the site name.  Stripped
# before matching.  Anchored at the start and followed by a separator, so a
# channel genuinely named "EEGZ" (if such a thing existed) is not mangled.
_PREFIX = re.compile(r"^(EEG|EOG|ECG|EKG|EMG|CHAN|CH)\s*[-_: ]\s*", re.IGNORECASE)

# Channel names that are a non-EEG modality repeated as its own label, e.g.
# EEGMAT's "ECG ECG" -> "ECG" after prefix stripping.
_MODALITY_ONLY = frozenset({"ECG", "EKG", "EOG", "EMG", "RESP", "GSR", "TEMP",
                            "TRIGGER", "STATUS", "MARKER", "STI", "EVENT"})


def canonicalise(name: str) -> str:
    """Return the canonical scalp-site spelling for a raw channel name.

    Strips an instrument prefix and normalises capitalisation.  If the result
    is not a recognised scalp site, the *stripped and whitespace-trimmed*
    name is returned unchanged, so the caller can see what it actually was.

    >>> canonicalise("EEG Fz")
    'Fz'
    >>> canonicalise("fp1")
    'Fp1'
    >>> canonicalise("EEG A2-A1")
    'A2-A1'
    >>> canonicalise("ECG ECG")
    'ECG'
    """
    stripped = _PREFIX.sub("", name).strip()
    return _BY_UPPER.get(stripped.upper(), stripped)


def is_scalp_site(name: str) -> bool:
    """True if `name` names a cortical measurement site this module knows.

    Reference electrodes (A1, A2, M1, M2) are recognised sites but are **not**
    scalp measurement sites, so this returns False for them.  Bipolar
    derivations such as ``"A2-A1"`` are not sites either: the canonical form
    still contains the hyphen and so matches nothing.
    """
    canon = canonicalise(name)
    return canon in _BY_UPPER.values() and canon not in REFERENCE_SITES


def classify(name: str) -> str:
    """Label a raw channel name as ``"scalp"``, ``"reference"``, or ``"other"``.

    ``"other"`` covers every channel that must stay out of EEG selections:
    non-EEG modalities (ECG, EOG, trigger lines) and bipolar derivations.  It
    is a catch-all on purpose — an unrecognised channel is excluded rather
    than assumed to be cortical.
    """
    canon = canonicalise(name)
    if canon in REFERENCE_SITES:
        return "reference"
    if canon in _BY_UPPER.values():
        return "scalp"
    return "other"


def is_modality_channel(name: str) -> bool:
    """True for a channel whose name says it is a non-EEG modality.

    Used for reporting (`"ECG ECG" was dropped as ECG`) rather than for
    exclusion, which `classify` already handles.  Distinguishing the two
    matters when auditing a file: an ECG channel being excluded is expected; an
    unrecognised channel named ``"Fz2"`` being excluded probably is not.
    """
    return canonicalise(name).upper() in _MODALITY_ONLY


# ----------------------------------------------------------------------
# Regions of interest
# ----------------------------------------------------------------------
#
# These are *named groups*, not decisions about what the index uses.  The group
# membership below mirrors pre-registration §7's description of the index terms
# so that the sites it needs have a name, but which sites the index actually
# averages is the index's own business (see eeg/README.md, "Where this package
# stops").

#: Frontal midline cluster.  Pre-registration §7 names Fz with F3/F4 for the
#: primary (theta) term.
ROI_FRONTAL_MIDLINE: tuple[str, ...] = ("Fz", "F3", "F4")

#: Parietal cluster.  Pre-registration §7 names Pz with P3/P4 for the
#: secondary (alpha) term.
ROI_PARIETAL: tuple[str, ...] = ("Pz", "P3", "P4")

#: Frontopolar sites, used by the artifact detector as a blink/ocular proxy
#: because they sit closest to the eyes and pick up corneo-retinal dipole
#: deflections at the largest amplitude of any scalp site.  They are a *proxy*:
#: no dedicated EOG electrode exists in either recording system, so this
#: detector cannot separate a blink from a vertical saccade, and it will also
#: flag genuine frontal cortical activity that happens to be large and slow.
#: See eeg/artifacts.py.
ROI_OCULAR_PROXY: tuple[str, ...] = ("Fp1", "Fp2")

#: Occipital sites.  The hardware validation gate (pre-registration §10) wants
#: an eyes-closed alpha increase demonstrated "at parietal and occipital
#: sites", so the group is named here for that check.
ROI_OCCIPITAL: tuple[str, ...] = ("O1", "O2", "Oz")

ROIS: dict[str, tuple[str, ...]] = {
    "frontal_midline": ROI_FRONTAL_MIDLINE,
    "parietal": ROI_PARIETAL,
    "ocular_proxy": ROI_OCULAR_PROXY,
    "occipital": ROI_OCCIPITAL,
}


def resolve(requested: tuple[str, ...] | list[str], available: list[str]) -> list[str]:
    """Match requested scalp sites against the channel names a recording has.

    `requested` is canonical site names (or ROI names from `ROIS`).
    `available` is the recording's channel names, already canonicalised by
    `eeg.loading`.  Returns the subset of `available`, **in the order
    `requested` gives**, that is present.

    The order matters: band-power arrays are indexed by position, and a caller
    that asks for ``("Fz", "F3", "F4")`` should get those three in that order
    regardless of how the EDF happened to lay out its channels.  A caller that
    silently got montage order instead would compute the right numbers and
    label them wrongly.

    Missing sites are **not** an error here — `eeg.loading` reports them, and
    the pipeline decides whether a missing site is fatal.  Returning a short
    list rather than raising lets a degraded-mode run (pre-registration §7)
    proceed with the sites it does have, provided the degradation is recorded.
    """
    expanded: list[str] = []
    for item in requested:
        if item in ROIS:
            expanded.extend(ROIS[item])
        else:
            expanded.append(canonicalise(item))
    have = set(available)
    out: list[str] = []
    for site in expanded:
        if site in have and site not in out:
            out.append(site)
    return out


def missing(requested: tuple[str, ...] | list[str], available: list[str]) -> list[str]:
    """Requested sites that the recording does not contain, in requested order."""
    expanded: list[str] = []
    for item in requested:
        if item in ROIS:
            expanded.extend(ROIS[item])
        else:
            expanded.append(canonicalise(item))
    have = set(available)
    out: list[str] = []
    for site in expanded:
        if site not in have and site not in out:
            out.append(site)
    return out
