"""Run the `eeg/` pipeline over EEGMAT subjects and report what came out.

What this is for
----------------
Three things, in order of importance:

1. **Signal quality and rejection rates on real data.**  Synthetic signals
   prove the code computes what it claims; only real data shows whether the
   thresholds in `eeg.config` are set anywhere near sensibly.  Every number in
   the per-subject table is a check on a default, and the defaults are all
   open items (`eeg/README.md`).
2. **The Berger-effect positive control** (`docs/DECISIONS.md` D032).
   Eyes-closed to eyes-open suppresses alpha.  It validates loading, channel
   selection, filtering, windowing and PSD against real physiology.
3. **D032 question 1, re-checked** — that EEGMAT really does retain power
   below 30 Hz, so the PhysioNet page's "30 Hz high-pass" is the typo for
   *low-pass* that D032 concluded it was.

What this is NOT for
--------------------
**EEGMAT cannot validate a workload effect, and this script does not try.**
Its rest recording is eyes-closed and its task is eyes-open (Zyma et al. 2019:
180 s eyes-closed rest, 60 s eyes-open serial subtraction), so the rest/task
contrast confounds workload with eye state, arousal and visual input.  D032
records that contrast as **void by design** for workload purposes, and notes
that running the remaining subjects would not fix it: the defect is bias, not
variance, and more observations on a confounded design only narrow the
confidence interval around the wrong quantity.

Theta is printed for the record, with that caveat attached, and must not be
read as evidence for or against the theta term of the index.  The workload
validation has to come from a contrast that holds eye state constant, which
the 25 combat configurations do.

Usage, from the repo root::

    python scripts/eeg_demo.py                     # every locally present subject
    python scripts/eeg_demo.py --subjects 5        # the first 5 subject ids
    python scripts/eeg_demo.py --sample 5 --seed 7 # 5 at random, reproducibly
    python scripts/eeg_demo.py --download          # fetch missing subjects first
    python scripts/eeg_demo.py --set artifact.peak_to_peak_uv=120
    python scripts/eeg_demo.py --save-dir runs/eeg_demo
    python scripts/eeg_demo.py --json              # machine-readable summary

Data lives in `data/raw/eegmat/`, which is gitignored.  Nothing is committed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Make `python scripts/eeg_demo.py` work from the repo root without PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eeg.channels import (ROI_FRONTAL_MIDLINE, ROI_OCCIPITAL,  # noqa: E402
                          ROI_PARIETAL)
from eeg.config import (EEGMAT_MAINS_HZ, PipelineConfig, SpectralConfig,  # noqa: E402
                        parse_override)
from eeg.pipeline import EEGPipeline, PipelineError, PipelineResult  # noqa: E402
from eeg.spectral import line_noise_ratio  # noqa: E402

# PhysioNet's /files/ endpoint 302-redirects to this S3 bucket and returns
# HTTP 500 to urllib's default client.  Fetching the bucket directly is the
# same published open-access data with one hop fewer.  Same approach as
# `scripts/verify_eegmat.py`.
BASE_URL = "https://physionet-open.s3.amazonaws.com/eegmat/1.0.0"
DATA_DIR = Path("data/raw/eegmat")
N_EEGMAT_SUBJECTS = 36

#: Band set used only for the D032 sub-30 Hz check, which needs a band above
#: 30 Hz that the project's own band set does not contain.
D032_BANDS = SpectralConfig(bands=(("below_30", 1.0, 30.0), ("above_30", 30.0, 45.0)))


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------


def download(subject: int, data_dir: Path) -> bool:
    """Fetch one subject's rest (_1) and task (_2) recordings.  True on success."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("_1", "_2"):
        name = f"Subject{subject:02d}{suffix}.edf"
        dest = data_dir / name
        if dest.exists():
            continue
        print(f"  downloading {name} ...", end="", flush=True)
        try:
            request = urllib.request.Request(
                f"{BASE_URL}/{name}",
                headers={"User-Agent": "capstone-eeg-demo/1.0"})
            with urllib.request.urlopen(request, timeout=180) as response, \
                    dest.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            print(f" {dest.stat().st_size / 1e6:.1f} MB")
        except Exception as exc:
            # Remove a truncated file, or the next run skips the download and
            # hands MNE a corrupt EDF.
            dest.unlink(missing_ok=True)
            print(f" FAILED: {exc}")
            return False
    return True


def available_subjects(data_dir: Path) -> list[int]:
    """Subject ids with *both* recordings present locally.

    Both, not either: the Berger check is a within-subject comparison, and a
    subject with only one recording would silently drop out of it.
    """
    ids: list[int] = []
    for path in sorted(data_dir.glob("Subject*_1.edf")):
        stem = path.name[len("Subject"):len("Subject") + 2]
        if not stem.isdigit():
            continue
        subject = int(stem)
        if (data_dir / f"Subject{subject:02d}_2.edf").exists():
            ids.append(subject)
    return ids


# ----------------------------------------------------------------------
# Per-subject run
# ----------------------------------------------------------------------


@dataclass
class SubjectRun:
    subject: int
    rest: PipelineResult
    task: PipelineResult

    def alpha_change(self, group: tuple[str, ...]) -> tuple[float, float, float]:
        """``(rest_median, task_median, percent_change)`` of alpha over `group`."""
        rest = float(np.median(self.rest.features.roi_series(
            group, "alpha", clean_only=True, require_all=False)))
        task = float(np.median(self.task.features.roi_series(
            group, "alpha", clean_only=True, require_all=False)))
        pct = (task - rest) / rest * 100.0 if rest else float("nan")
        return rest, task, pct

    def theta_change(self, group: tuple[str, ...]) -> tuple[float, float, float]:
        rest = float(np.median(self.rest.features.roi_series(
            group, "theta", clean_only=True, require_all=False)))
        task = float(np.median(self.task.features.roi_series(
            group, "theta", clean_only=True, require_all=False)))
        pct = (task - rest) / rest * 100.0 if rest else float("nan")
        return rest, task, pct


def run_subject(pipe: EEGPipeline, subject: int, data_dir: Path) -> SubjectRun:
    return SubjectRun(
        subject=subject,
        rest=pipe.run_edf(data_dir / f"Subject{subject:02d}_1.edf"),
        task=pipe.run_edf(data_dir / f"Subject{subject:02d}_2.edf"),
    )


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------


def print_config(pipe: EEGPipeline) -> None:
    config = pipe.config
    print("=" * 78)
    print("PIPELINE CONFIGURATION")
    print("=" * 78)
    if config.note:
        print(f"  {config.note}")
    bandpass = (f"{config.filter.bandpass_low or 'DC'}-"
                f"{config.filter.bandpass_high or 'Nyquist'} Hz")
    notch = (", ".join(f"{f:g} Hz" for f in config.filter.notch_freqs)
             or "none (the low-pass already excludes mains)")
    print(f"  bandpass            {bandpass}, Butterworth order "
          f"{config.filter.bandpass_order}, "
          f"{'zero-phase' if config.filter.zero_phase else 'causal'}")
    print(f"  notch               {notch}")
    print(f"  edge trim           {config.filter.edge_trim_seconds:g} s each end")
    print(f"  window              {config.epoch.window_seconds:g} s, hop "
          f"{config.epoch.effective_hop_seconds:g} s "
          f"({'overlapping' if config.epoch.hop_seconds else 'non-overlapping'})")
    print(f"  Welch segment       {config.spectral.welch_segment_seconds or 'whole window'} s"
          f", overlap {config.spectral.welch_overlap:g}, {config.spectral.window}")
    print(f"  bands               "
          f"{', '.join(f'{n} {lo:g}-{hi:g}' for n, lo, hi in config.spectral.bands)} Hz")
    print(f"  reject p-p >        {config.artifact.peak_to_peak_uv:g} uV")
    print(f"  reject flat <       {config.artifact.flat_uv:g} uV")
    print(f"  reject gradient >   {config.artifact.max_gradient_uv_per_sample:g} uV/sample")
    print(f"  ocular              {', '.join(config.artifact.ocular_channels)}, "
          f"{config.artifact.ocular_band[0]:g}-{config.artifact.ocular_band[1]:g} Hz, "
          f"robust z > {config.artifact.ocular_z_threshold:g}")
    print(f"  reference           {config.load.reference}")
    print()
    print("  Every one of these is an OPEN item. See eeg/README.md for the")
    print("  options and tradeoffs. A default is a starting value, not a decision.")
    print()


def print_rejection_table(runs: list[SubjectRun]) -> None:
    print("=" * 78)
    print("SIGNAL QUALITY AND WINDOW REJECTION")
    print("=" * 78)
    print("\n  Rejection rate is itself an experimental variable: pre-registration §7")
    print("  correlates per-configuration rejection against the difficulty ranking,")
    print("  because a rate that tracks difficulty is a confound. A rate near 0 means")
    print("  the thresholds are inert; a rate near 1 manufactures that confound.\n")
    header = (f"{'subj':>4} {'rec':>5} {'win':>4} {'clean':>6} {'rej':>6} "
              f"{'p-p':>6} {'amp':>6} {'flat':>6} {'grad':>6} {'ocular':>7}")
    print(header)
    print("-" * len(header))
    for run in runs:
        for label, result in (("rest", run.rest), ("task", run.task)):
            artifacts = result.artifacts
            rates = artifacts.rate_by_reason()
            ptp = np.median([s["median_ptp_uv"]
                             for s in result.quality_summary()["per_channel"].values()])
            print(f"{run.subject:>4} {label:>5} {artifacts.n_windows:>4} "
                  f"{artifacts.n_clean:>6} {artifacts.rejection_rate:>5.1%} "
                  f"{ptp:>5.1f}u {rates['peak_to_peak']:>5.1%} {rates['flat']:>5.1%} "
                  f"{rates['gradient']:>5.1%} {rates['ocular']:>6.1%}")
    print()
    print("  p-p = median across channels of each channel's median peak-to-peak over")
    print("  clean windows, in uV. The remaining columns are the fraction of windows")
    print("  each rejection reason fired on; reasons overlap, so they do not sum to rej.")
    print()


def print_channel_table(run: SubjectRun) -> None:
    print("=" * 78)
    print(f"PER-CHANNEL QUALITY, SUBJECT {run.subject:02d} REST")
    print("=" * 78)
    print("\n  A single channel failing in most windows while the rest are clean is an")
    print("  impedance problem, not a restless participant. Discarding a session for")
    print("  one badly seated electrode would be throwing away a usable recording.\n")
    summary = run.rest.quality_summary()
    threshold = summary["peak_to_peak_threshold_uv"]
    print(f"{'channel':>8} {'med p-p':>9} {'p95 p-p':>9} {'max p-p':>9} "
          f"{'headroom':>9} {'reject':>8}")
    print("-" * 58)
    for channel, stats in summary["per_channel"].items():
        headroom = threshold / stats["median_ptp_uv"] if stats["median_ptp_uv"] else float("nan")
        print(f"{channel:>8} {stats['median_ptp_uv']:>8.1f}u {stats['p95_ptp_uv']:>8.1f}u "
              f"{stats['max_ptp_uv']:>8.1f}u {headroom:>8.1f}x {stats['reject_rate']:>7.1%}")
    print(f"\n  headroom = {threshold:g} uV threshold / median p-p. A channel near 1x is one")
    print("  bad minute away from wholesale rejection.")
    print()


def print_berger(runs: list[SubjectRun]) -> dict[str, Any]:
    print("=" * 78)
    print("POSITIVE CONTROL: THE BERGER EFFECT (D032)")
    print("=" * 78)
    print("\n  EEGMAT rest is EYES-CLOSED, task is eyes-open (Zyma et al. 2019).")
    print("  Eyes-closed -> eyes-open suppresses alpha (Berger, 1929). If the loading,")
    print("  channel-selection, filtering, windowing and PSD code is right, alpha must")
    print("  FALL from _1 to _2 -- strongest posteriorly, where alpha lives.")
    print("\n  This validates the CODE against real physiology. It says nothing about")
    print("  workload: the contrast confounds workload with eye state, arousal and")
    print("  visual input, and D032 records it as void for that purpose.\n")

    out: dict[str, Any] = {}
    for group, name in ((ROI_PARIETAL, "parietal"), (ROI_OCCIPITAL, "occipital"),
                        (ROI_FRONTAL_MIDLINE, "frontal midline")):
        print(f"  {name}  ({', '.join(group)})")
        print(f"    {'subj':>6} {'alpha rest':>12} {'alpha task':>12} {'change':>9}")
        print("    " + "-" * 42)
        changes: list[float] = []
        for run in runs:
            rest, task, pct = run.alpha_change(group)
            changes.append(pct)
            print(f"    {run.subject:>6} {rest:>11.3f} {task:>11.3f} {pct:>8.1f}%"
                  f"  {'down' if pct < 0 else 'UP'}")
        fell = sum(1 for c in changes if c < 0)
        mean = float(np.mean(changes)) if changes else float("nan")
        verdict = "PASS" if fell > len(changes) / 2 else "UNEXPECTED"
        print(f"\n    alpha fell in {fell}/{len(changes)} subjects, mean {mean:+.1f}%  "
              f"-> {verdict}")
        if verdict == "UNEXPECTED":
            print("    Alpha must fall. Suspect the loading, channel selection,")
            print("    filtering or PSD code before suspecting the data.")
        print()
        out[name] = {"n_subjects": len(changes), "n_fell": fell,
                     "mean_percent_change": mean, "verdict": verdict,
                     "per_subject_percent": changes}
    return out


def print_d032_check(pipe: EEGPipeline, subjects: list[int], data_dir: Path) -> dict[str, Any]:
    print("=" * 78)
    print("D032 QUESTION 1 RE-CHECKED: IS THERE POWER BELOW 30 Hz?")
    print("=" * 78)
    print("\n  PhysioNet's EEGMAT page says 'a high-pass filter with a 30 Hz cut-off")
    print("  frequency' was applied. Taken literally that deletes all theta and alpha")
    print("  and makes the dataset useless for this project's index. D032 found it is")
    print("  a typo for LOW-pass, by 41.3x on unfiltered data. Re-checked here through")
    print("  the full pipeline, where our own 45 Hz low-pass raises the ratio further.\n")
    probe = EEGPipeline(PipelineConfig.for_eegmat(spectral=D032_BANDS))
    print(f"  {'subj':>6} {'<30 Hz':>12} {'30-45 Hz':>12} {'ratio':>9}")
    print("  " + "-" * 42)
    ratios: list[float] = []
    for subject in subjects:
        features = probe.run_edf(data_dir / f"Subject{subject:02d}_1.edf").features
        below = float(np.median(features.roi_series(
            ROI_FRONTAL_MIDLINE, "below_30", clean_only=True, require_all=False)))
        above = float(np.median(features.roi_series(
            ROI_FRONTAL_MIDLINE, "above_30", clean_only=True, require_all=False)))
        ratio = below / above if above else float("inf")
        ratios.append(ratio)
        print(f"  {subject:>6} {below:>11.2f} {above:>11.3f} {ratio:>8.1f}x")
    mean = float(np.mean(ratios)) if ratios else float("nan")
    verdict = "PASS" if mean > 5.0 else "FAIL"
    print(f"\n  mean ratio {mean:.1f}x -> {verdict}")
    if verdict == "PASS":
        print("  Sub-30 Hz activity dominates. The page's '30 Hz high-pass' is a typo")
        print("  for low-pass, as D032 concluded. The dataset retains theta and alpha.")
    else:
        print("  Almost no power below 30 Hz. The filter description may be literal,")
        print("  which would make this dataset unusable for a theta/alpha index.")
    print()
    return {"mean_ratio": mean, "per_subject_ratio": ratios, "verdict": verdict}


def print_line_noise(runs: list[SubjectRun], data_dir: Path) -> None:
    print("=" * 78)
    print("LINE-NOISE DIAGNOSTIC")
    print("=" * 78)
    print("\n  Mains-band power over beta-band power, measured on the UNFILTERED")
    print(f"  signal at {EEGMAT_MAINS_HZ:g} Hz (Ukraine). After a 45 Hz low-pass this ratio")
    print("  would describe the filter's roll-off rather than the recording, so it")
    print("  only means anything before filtering. At the hardware validation gate")
    print("  (§10) a high ratio on one channel is a poorly seated electrode, which is")
    print("  actionable before a session rather than after it.\n")
    from eeg.loading import load_edf
    print(f"  {'subj':>6} {'median':>12} {'worst':>8} {'worst ratio':>14}")
    print("  " + "-" * 44)
    medians: list[float] = []
    for run in runs:
        raw = load_edf(data_dir / f"Subject{run.subject:02d}_1.edf",
                       run.rest.config.load)
        ratios = line_noise_ratio(raw.data, raw.sfreq, EEGMAT_MAINS_HZ)
        median = float(np.nanmedian(ratios))
        medians.append(median)
        worst = int(np.nanargmax(ratios))
        print(f"  {run.subject:>6} {median:>12.2e} {raw.channel_names[worst]:>8} "
              f"{float(ratios[worst]):>14.2e}")
    overall = float(np.mean(medians)) if medians else float("nan")
    print(f"\n  mean median ratio {overall:.2e}")
    if overall < 0.1:
        print("  Essentially no mains line in these recordings -- expected, and a check")
        print("  on the diagnostic rather than a surprise: EEGMAT's own acquisition")
        print("  hardware low-passed at 45 Hz (its EDF header says so), which is below")
        print("  50 Hz mains. This is also why FilterConfig applies no notch by default:")
        print("  with a 45 Hz passband there is no line left for one to remove, and a")
        print("  notch logged in provenance but doing nothing would mislead a reader.")
        print("  On the author's own 60 Hz hardware this number will not be near zero,")
        print("  and that is when it earns its place at the §10 validation gate.")
    print()


def print_theta_for_the_record(runs: list[SubjectRun]) -> None:
    print("=" * 78)
    print("FRONTAL MIDLINE THETA -- FOR THE RECORD ONLY, NOT A WORKLOAD TEST")
    print("=" * 78)
    print(f"\n  {'subj':>6} {'theta rest':>12} {'theta task':>12} {'change':>9}")
    print("  " + "-" * 42)
    for run in runs:
        rest, task, pct = run.theta_change(ROI_FRONTAL_MIDLINE)
        print(f"  {run.subject:>6} {rest:>11.3f} {task:>11.3f} {pct:>8.1f}%")
    print("\n  EEGMAT's rest is eyes-closed, so this contrast confounds workload with")
    print("  eye state, arousal and visual input. Whichever way it points, it is NOT")
    print("  evidence for or against the theta term of the index (D032: void by")
    print("  design; the defect is bias, not variance, so more subjects would not")
    print("  help). The workload validation has to come from a contrast that holds")
    print("  eye state constant -- which the 25 combat configurations do.")
    print()


def print_index_boundary() -> None:
    print("=" * 78)
    print("WHERE THIS PIPELINE STOPS")
    print("=" * 78)
    print("""
  The pipeline's output is band power per window per channel, plus the clean
  mask. The workload index -- standardised frontal midline theta minus
  down-weighted standardised parietal alpha (pre-registration §7) -- is NOT
  implemented in `eeg/`. It is the author's to write (CLAUDE.md: "the user
  writes ... the workload index computation").

  The interface it plugs into is `eeg.pipeline.WorkloadIndex`:

      def my_index(features: WindowFeatures) -> np.ndarray:  # (n_windows,)
          theta = features.roi_series(("Fz", "F3", "F4"), "theta")
          alpha = features.roi_series(("Pz", "P3", "P4"), "alpha")
          ...   # within-session standardisation and weighting go here
          return values           # one per window, INCLUDING rejected ones

      values = apply_index(result, my_index)

  `apply_index` validates the contract at the boundary -- length, dtype, and
  finiteness at clean windows -- so a misaligned array fails there rather than
  producing a plausible correlation three stages later. It computes no median,
  rank or correlation: those are the analysis (§6, §9) and the author's too.
""")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help=f"where the EDFs live (default {DATA_DIR})")
    parser.add_argument("--subjects", type=int, default=None, metavar="N",
                        help="run the first N locally available subjects")
    parser.add_argument("--sample", type=int, default=None, metavar="N",
                        help="run N randomly chosen subjects (see --seed)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for --sample; the pipeline itself has no "
                             "randomness, so this is the only seed in play (default 0)")
    parser.add_argument("--download", action="store_true",
                        help="fetch missing subjects from PhysioNet's S3 bucket first")
    parser.add_argument("--download-subjects", type=int, default=3, metavar="N",
                        help="how many subjects --download should fetch (default 3)")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="section.key=value",
                        help="override a config field, e.g. "
                             "--set artifact.peak_to_peak_uv=120")
    parser.add_argument("--save-dir", type=Path, default=None,
                        help="write each run's JSON metadata and NPZ arrays here")
    parser.add_argument("--json", action="store_true",
                        help="print a machine-readable summary instead of the tables")
    parser.add_argument("--skip-line-noise", action="store_true",
                        help="skip the line-noise diagnostic (it re-reads each EDF)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir: Path = args.data_dir

    if args.download:
        print(f"Downloading up to {args.download_subjects} EEGMAT subjects "
              f"into {data_dir}\n")
        for subject in range(args.download_subjects):
            if not download(subject, data_dir):
                print("\n  Download failed. PhysioNet's /files/ endpoint 302-redirects and")
                print("  returns HTTP 500 to urllib, which is why this uses the S3 bucket")
                print("  directly; a failure here usually means no network route to it.")
                break
        print()

    subjects = available_subjects(data_dir)
    if not subjects:
        print(f"No complete EEGMAT subject pairs in {data_dir}.")
        print("Run with --download, or fetch them with scripts/verify_eegmat.py.")
        print(f"Source: {BASE_URL}/Subject00_1.edf  (36 subjects, _1 rest, _2 task)")
        return 1

    if args.sample is not None:
        rng = np.random.default_rng(args.seed)
        n = min(args.sample, len(subjects))
        subjects = sorted(rng.choice(subjects, size=n, replace=False).tolist())
    elif args.subjects is not None:
        subjects = subjects[:args.subjects]

    config = PipelineConfig.for_eegmat()
    if args.overrides:
        config = config.with_overrides(dict(parse_override(o) for o in args.overrides))
    pipe = EEGPipeline(config)

    if not args.json:
        print()
        print_config(pipe)
        print(f"Running {len(subjects)} subject(s) of {N_EEGMAT_SUBJECTS} in the dataset: "
              f"{subjects}\n")

    runs: list[SubjectRun] = []
    failures: dict[int, str] = {}
    for subject in subjects:
        try:
            runs.append(run_subject(pipe, subject, data_dir))
        except (PipelineError, OSError) as exc:
            # One unreadable EDF must not take down the whole report; which
            # subject failed, and why, is itself a finding.
            failures[subject] = f"{type(exc).__name__}: {exc}"

    if not runs:
        print("Every subject failed:")
        for subject, reason in failures.items():
            print(f"  Subject{subject:02d}: {reason}")
        return 1

    if args.save_dir:
        for run in runs:
            for label, result in (("rest", run.rest), ("task", run.task)):
                result.save(args.save_dir, name=f"subject{run.subject:02d}_{label}")
        if not args.json:
            print(f"Saved {2 * len(runs)} result(s) to {args.save_dir}\n")

    if args.json:
        payload: dict[str, Any] = {
            "config": pipe.config.to_dict(),
            "subjects": subjects,
            "seed": args.seed,
            "failures": failures,
            "per_recording": [
                {"subject": run.subject, "recording": label,
                 "quality": result.quality_summary()}
                for run in runs for label, result in (("rest", run.rest), ("task", run.task))],
            "berger": {
                name: {"n_subjects": len(runs),
                       "n_fell": sum(1 for r in runs if r.alpha_change(group)[2] < 0),
                       "mean_percent_change": float(np.mean(
                           [r.alpha_change(group)[2] for r in runs]))}
                for group, name in ((ROI_PARIETAL, "parietal"),
                                    (ROI_OCCIPITAL, "occipital"),
                                    (ROI_FRONTAL_MIDLINE, "frontal_midline"))},
            "note": ("EEGMAT's rest/task contrast is void as a workload measure "
                     "(D032): rest is eyes-closed, task eyes-open. The Berger "
                     "alpha-suppression check validates the code, not a workload "
                     "effect."),
        }
        print(json.dumps(payload, indent=2, allow_nan=True))
        return 0

    print_rejection_table(runs)
    print_channel_table(runs[0])
    berger = print_berger(runs)
    print_d032_check(pipe, subjects, data_dir)
    if not args.skip_line_noise:
        print_line_noise(runs, data_dir)
    print_theta_for_the_record(runs)
    print_index_boundary()

    if failures:
        print("Subjects that failed to run:")
        for subject, reason in failures.items():
            print(f"  Subject{subject:02d}: {reason}")
        print()

    posterior_pass = all(berger[name]["verdict"] == "PASS"
                         for name in ("parietal", "occipital"))
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  {len(runs)} subject(s), {2 * len(runs)} recordings, "
          f"{sum(r.rest.artifacts.n_windows + r.task.artifacts.n_windows for r in runs)} "
          "windows analysed")
    mean_rejection = float(np.mean(
        [r.rest.artifacts.rejection_rate for r in runs]
        + [r.task.artifacts.rejection_rate for r in runs]))
    print(f"  mean window rejection rate: {mean_rejection:.1%}")
    print(f"  Berger positive control at parietal and occipital sites: "
          f"{'PASS' if posterior_pass else 'FAILED'}")
    print("  Workload: NOT TESTED and not testable on EEGMAT (D032).")
    print()
    return 0 if posterior_pass else 2


if __name__ == "__main__":
    sys.exit(main())
