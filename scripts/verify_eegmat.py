"""Verify the EEGMAT dataset is usable for this project's workload index.

Two questions, in order of importance:

1. **Is there real power below 30 Hz?**
   The PhysioNet page says "a high-pass filter with a 30 Hz cut-off frequency"
   was applied.  Taken literally that removes everything below 30 Hz, which
   would delete all the theta (4-8 Hz) and alpha (8-13 Hz) this project's index
   depends on.  It is almost certainly a typo for *low-pass*, but "almost
   certainly" is not a basis for building a pipeline.

2. **Does the pipeline reproduce a known physiological effect?**
   NOTE (2026-11-01): this question originally asked whether frontal theta
   rises from rest to task, as a check on the workload literature.  That was a
   design error.  EEGMAT's rest recording is **eyes-closed** (Zyma et al. 2019;
   180 s eyes-closed rest, 60 s eyes-open arithmetic), so rest and task differ
   in eye state, arousal and visual input as well as in workload.  A theta
   difference across that contrast is uninterpretable, and no amount of extra
   subjects fixes it -- more data on a confounded design only narrows the
   confidence interval around the wrong quantity.

   What the contrast IS good for is a positive control on our own code.
   Eyes-closed to eyes-open reliably *suppresses* alpha (the Berger effect,
   1929).  If our loading, filtering and PSD code is correct, alpha must fall
   from _1 to _2.  That validates the pipeline without claiming anything about
   workload.

Usage (from the repo root):
    python scripts/verify_eegmat.py
    python scripts/verify_eegmat.py --subjects 5     # more subjects

Downloads ~15 MB into data/raw/eegmat/ (gitignored) on first run.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np

# PhysioNet's /files/ endpoint 302-redirects to this S3 bucket and returns
# HTTP 500 to urllib's default client.  Fetching the bucket directly is the
# same published open-access data with one hop fewer.
BASE_URL = "https://physionet-open.s3.amazonaws.com/eegmat/1.0.0"
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
                # An explicit User-Agent: some mirrors reject urllib's default.
                req = urllib.request.Request(
                    url, headers={"User-Agent": "capstone-eegmat-verify/1.0"})
                with urllib.request.urlopen(req, timeout=120) as resp, \
                        dest.open("wb") as fh:
                    shutil.copyfileobj(resp, fh)
                print(f" {dest.stat().st_size / 1e6:.1f} MB")
            except Exception as exc:
                # Remove a truncated file, or the next run would skip the
                # download and hand mne a corrupt EDF.
                dest.unlink(missing_ok=True)
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

    # ---- Question 2: positive control -- does alpha drop, as it must? -------
    print("\n" + "=" * 62)
    print("QUESTION 2  Does the pipeline reproduce a known effect?")
    print("=" * 62)
    print("\n  EEGMAT rest is EYES-CLOSED, task is eyes-open (Zyma et al. 2019).")
    print("  Eyes-closed -> eyes-open suppresses alpha (Berger, 1929). If our")
    print("  loading/PSD/band code is right, alpha must FALL from _1 to _2.\n")

    print(f"{'subject':>8} {'alpha rest':>12} {'alpha task':>12} {'change':>10}")
    print("-" * 46)
    drops = 0
    for i, (r, t) in enumerate(zip(rest_rows, task_rows)):
        pct = (t["alpha"] - r["alpha"]) / r["alpha"] * 100 if r["alpha"] else 0.0
        drops += pct < 0
        print(f"{i:>8} {r['alpha']:>12.4g} {t['alpha']:>12.4g} {pct:>8.1f}%"
              f" {'down' if pct < 0 else 'up'}")

    n = len(rest_rows)
    print(f"\n  alpha fell in {drops}/{n} subjects")
    if drops > n / 2:
        print("  PASS - the expected direction. The pipeline reads real physiology.")
    else:
        print("  UNEXPECTED - alpha should fall. Suspect the loading, channel")
        print("  selection or PSD code before suspecting the data.")

    # ---- Theta, reported but NOT interpreted --------------------------------
    print("\n" + "-" * 62)
    print("  Theta, for the record only (NOT a workload test):")
    print("-" * 62)
    print(f"{'subject':>8} {'theta rest':>12} {'theta task':>12} {'change':>10}")
    print("-" * 46)
    for i, (r, t) in enumerate(zip(rest_rows, task_rows)):
        pct = (t["theta"] - r["theta"]) / r["theta"] * 100 if r["theta"] else 0.0
        print(f"{i:>8} {r['theta']:>12.4g} {t['theta']:>12.4g} {pct:>8.1f}%"
              f" {'up' if pct > 0 else 'down'}")
    print("\n  Rest is eyes-closed, so this contrast confounds workload with eye")
    print("  state, arousal and visual input. Whichever way it points, it is")
    print("  NOT evidence for or against the theta term of our index. The")
    print("  workload validation has to come from a contrast where eye state is")
    print("  held constant -- which our own combat configurations are.")

    print("\nDone. Record the outcome in docs/DECISIONS.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
