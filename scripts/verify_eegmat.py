"""Verify the EEGMAT dataset is usable for this project's workload index.

Two questions, in order of importance:

1. **Is there real power below 30 Hz?**
   The PhysioNet page says "a high-pass filter with a 30 Hz cut-off frequency"
   was applied.  Taken literally that removes everything below 30 Hz, which
   would delete all the theta (4-8 Hz) and alpha (8-13 Hz) this project's index
   depends on.  It is almost certainly a typo for *low-pass*, but "almost
   certainly" is not a basis for building a pipeline.

2. **Does frontal theta rise from rest to task?**
   Bonus, and more interesting.  EEGMAT recorded each subject at rest (_1) and
   during mental arithmetic (_2).  If frontal theta is higher during the task,
   that is the workload effect this project's whole index is built on, visible
   in independent data before any of our own hardware is involved.

Usage (from the repo root):
    python scripts/verify_eegmat.py
    python scripts/verify_eegmat.py --subjects 5     # more subjects

Downloads ~15 MB into data/raw/eegmat/ (gitignored) on first run.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np

BASE_URL = "https://physionet.org/files/eegmat/1.0.0"
DATA_DIR = Path("data/raw/eegmat")

# Canonical EEG bands.  The index in this project uses theta and alpha.
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),     # primary term of the workload index
    "alpha": (8.0, 13.0),    # secondary term
    "beta": (13.0, 30.0),
    "above_30": (30.0, 45.0),  # what would survive a literal 30 Hz high-pass
}

# Frontal midline is where the workload theta effect is strongest.
FRONTAL = ["Fz", "EEG Fz", "F3", "EEG F3", "F4", "EEG F4"]


def download(subject: int) -> tuple[Path, Path] | None:
    """Fetch one subject's rest (_1) and task (_2) recordings."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("_1", "_2"):
        name = f"Subject{subject:02d}{suffix}.edf"
        dest = DATA_DIR / name
        if not dest.exists():
            url = f"{BASE_URL}/{name}"
            print(f"  downloading {name} ...", end="", flush=True)
            try:
                urllib.request.urlretrieve(url, dest)
                print(f" {dest.stat().st_size / 1e6:.1f} MB")
            except Exception as exc:
                print(f" FAILED: {exc}")
                return None
        paths.append(dest)
    return paths[0], paths[1]


def band_powers(raw, picks) -> dict[str, float]:
    """Mean absolute power in each band, via Welch's method."""
    from scipy.signal import welch

    data = raw.get_data(picks=picks)          # (n_channels, n_samples)
    fs = raw.info["sfreq"]
    # 4-second windows give ~0.25 Hz resolution, ample for 4-13 Hz bands.
    freqs, psd = welch(data, fs=fs, nperseg=int(4 * fs), axis=-1)
    psd = psd.mean(axis=0)                     # average across channels

    out = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        # Integrate the PSD over the band (uV^2), not just the mean height.
        out[name] = float(np.trapezoid(psd[mask], freqs[mask]))
    return out


def pick_frontal(raw) -> list[int]:
    """Indices of frontal channels, falling back to all channels."""
    names = raw.ch_names
    picks = [i for i, n in enumerate(names)
             if any(n.strip().upper() == f.upper() for f in FRONTAL)]
    return picks if picks else list(range(len(names)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=3,
                    help="how many subjects to check (default 3)")
    args = ap.parse_args()

    try:
        import mne
        mne.set_log_level("ERROR")
    except ImportError:
        print("mne is required:  pip install mne")
        return 1

    print(f"Checking {args.subjects} EEGMAT subjects\n")

    rest_rows, task_rows = [], []
    channel_names = None

    for subj in range(args.subjects):
        print(f"Subject {subj:02d}")
        got = download(subj)
        if got is None:
            continue
        rest_path, task_path = got

        rest = mne.io.read_raw_edf(rest_path, preload=True, verbose=False)
        task = mne.io.read_raw_edf(task_path, preload=True, verbose=False)

        if channel_names is None:
            channel_names = rest.ch_names
            print(f"  {len(channel_names)} channels @ {rest.info['sfreq']:.0f} Hz")
            print(f"  {', '.join(channel_names[:8])} ...")

        picks = pick_frontal(rest)
        rest_rows.append(band_powers(rest, picks))
        task_rows.append(band_powers(task, picks))
        print(f"  rest {rest.n_times / rest.info['sfreq']:.0f}s, "
              f"task {task.n_times / task.info['sfreq']:.0f}s")

    if not rest_rows:
        print("\nNo data downloaded. Check your network connection.")
        return 1

    # ---------------- Question 1: is there power below 30 Hz? ----------------
    print("\n" + "=" * 62)
    print("QUESTION 1  Is there real power below 30 Hz?")
    print("=" * 62)
    print(f"\n{'band':>10} {'rest power':>14} {'task power':>14}")
    print("-" * 42)
    below_30 = 0.0
    for band in BANDS:
        r = np.mean([row[band] for row in rest_rows])
        t = np.mean([row[band] for row in task_rows])
        print(f"{band:>10} {r:>14.4g} {t:>14.4g}")
        if band != "above_30":
            below_30 += r

    above_30 = np.mean([row["above_30"] for row in rest_rows])
    ratio = below_30 / above_30 if above_30 > 0 else float("inf")
    print(f"\n  power below 30 Hz / power above 30 Hz = {ratio:.1f}x")

    if ratio > 5:
        print("  PASS - the data is dominated by sub-30 Hz activity.")
        print("  The page's '30 Hz high-pass' is a typo for low-pass, as suspected.")
    elif ratio > 1:
        print("  MARGINAL - sub-30 Hz power exists but is not dominant.")
        print("  Inspect the spectrum before trusting this dataset.")
    else:
        print("  FAIL - almost no power below 30 Hz.")
        print("  The filter description may be literal. This dataset would be")
        print("  unusable for a theta/alpha index. Find another one.")

    # ------------- Question 2: does frontal theta rise with load? ------------
    print("\n" + "=" * 62)
    print("QUESTION 2  Does frontal theta rise from rest to task?")
    print("=" * 62)
    print("\n  (This is the effect the project's workload index is built on.)\n")

    print(f"{'subject':>8} {'theta rest':>12} {'theta task':>12} {'change':>10}")
    print("-" * 46)
    increases = 0
    for i, (r, t) in enumerate(zip(rest_rows, task_rows)):
        pct = (t["theta"] - r["theta"]) / r["theta"] * 100 if r["theta"] else 0.0
        arrow = "up" if pct > 0 else "down"
        increases += pct > 0
        print(f"{i:>8} {r['theta']:>12.4g} {t['theta']:>12.4g} {pct:>8.1f}% {arrow}")

    n = len(rest_rows)
    print(f"\n  theta increased in {increases}/{n} subjects")
    if increases > n / 2:
        print("  Consistent with the workload literature (Chikhi et al. 2022, theta g=0.68).")
    else:
        print("  Not the expected direction here. Small n; not alarming on its own,")
        print("  but worth noting before relying on the theta term.")

    print("\nDone. Record the outcome in docs/DECISIONS.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
