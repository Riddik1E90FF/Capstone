"""Hyperparameter sensitivity sweep for the combat baseline agent (D034).

D034's section "Hyperparameter sensitivity: decision rule, fixed 2026-11-01
before the sweep" fixes the question and the decision rule; this script runs
the measurement and applies the rule mechanically.  It does not decide
anything the rule does not already decide, and it changes nothing about the
environment, the reward, or the evaluation protocol.

What it does, per (setting, seed):

  1. Trains with `scripts/train_dqn.py --set env_id=combat:mix --set <k>=<v>
     --seed <n>`, i.e. the same CLI any other run uses.  No training code is
     touched.
  2. Loads the run's final checkpoint and evaluates it with
     `evaluation/gate1.evaluate_policy` at `N_EVAL = 200` on all 25
     configurations, which is the Gate 1 protocol unchanged.  The greedy
     wrapper is the same construction `scripts/gate1_check.py` uses.
  3. Writes `eval.json` next to the run directory: the fully resolved config,
     the git commit, and every per-configuration number.

`report` then aggregates to 3-seed means and applies D034's rule:

  * rule 1 — outcome is mean damage taken, averaged over the 3 seeds.
  * rule 2 — a setting qualifies only if it improves that mean by MORE than
    `SD_GATE1 = 8.344`, one between-seed SD from the Gate 1 attempt.
  * rule 3 — among qualifying settings, the SMALLEST change from the current
    defaults is recommended, not the best-scoring one.  "Smallest" is ordered
    by `CHANGE_RANK` below, which is declared here rather than inferred from
    the results.
  * rule 5 — if nothing qualifies, that is the reported result.

Rule 4 (any adopted setting triggers a full new Gate 1 attempt) is Ryan's to
run; this script only ever recommends.

Usage (from the repo root):
    python scripts/hp_sweep.py run --workers 2
    python scripts/hp_sweep.py report

`run` is resumable: a (setting, seed) whose `eval.json` already exists is
skipped, so an interrupted sweep continues rather than restarting.
"""

from __future__ import annotations

# Keep BLAS single-threaded in this process and every child.  Set before NumPy
# is imported anywhere, because the thread count is read at import time.  With
# several runs in parallel, multi-threaded BLAS on small matrices is slower
# than one thread each and makes wall-clock times incomparable between runs.
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

# Make this script runnable from any directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ----------------------------------------------------------------------
# Fixed quantities.  None of these are chosen from the sweep's own output.
# ----------------------------------------------------------------------

# D034 rule 2.  The between-seed SD of mean damage taken in the Gate 1
# attempt (docs/gate1_result.json, 10 seeds, sample SD): 8.343869 -> 8.344 as
# D034 writes it.  Quoted to D034's precision deliberately: the threshold is
# the number in the decision record, not a number recomputed here.
SD_GATE1 = 8.344

# docs/gate1_result.json, for reference in the report only.
GATE1_10SEED_MEAN = 36.24995
GATE1_SEEDS_1_3_MEAN = 30.58693   # seeds 1, 2, 3 — the seeds this sweep uses

ENV_ID = "combat:mix"
SEEDS = (1, 2, 3)                 # D034: 3 seeds per setting
TOTAL_TIMESTEPS = 500_000         # D034: 500k steps, as the Gate 1 attempt used

# Current DQNConfig defaults for the three swept fields.  Read from the
# dataclass rather than written out, so a change to the defaults cannot leave
# this script quietly comparing against a stale baseline.
def current_defaults() -> dict[str, Any]:
    from dqn.config import DQNConfig
    cfg = DQNConfig()
    return {"buffer_size": cfg.buffer_size,
            "learning_rate": cfg.learning_rate,
            "train_frequency": cfg.train_frequency}


# ----------------------------------------------------------------------
# The grid.  One factor at a time around the defaults.  Rationale is in
# docs/HP_SWEEP.md; the short version is that D034 rule 3 adopts the
# *smallest* change from the defaults, which only has a meaning if the
# candidates are ordered by distance from the defaults, and a coordinate
# sweep gives that ordering directly.
#
# `change_rank` implements rule 3's "smallest change", declared here before
# any run so it cannot be chosen to suit an outcome.  Lower = smaller.  The
# ordering is:
#
#   1. fewer fields changed beats more fields changed;
#   2. then smaller multiplicative distance from the default
#      (max(v / default, default / v)) — all three fields are scale
#      parameters, so ratio, not difference, is the natural metric;
#   3. then, if the ratio ties, the smaller absolute change in the field's
#      value.  This is what separates train_frequency 5 (|5 - 10| = 5) from
#      train_frequency 20 (|20 - 10| = 10): both are exactly 2x.
#
# Resulting ranks: lr5e-4 (2.0x), tf5 (2.0x, abs 5), tf20 (2.0x, abs 10),
# lr1e-4 (2.5x), buf30k (3.0x), buf3k (3.33x), buf100k (10x), buf300k (30x).
# ----------------------------------------------------------------------

GRID: tuple[dict[str, Any], ...] = (
    # name, overrides relative to DQNConfig defaults, change_rank
    {"name": "default",      "overrides": {},                                            "change_rank": 0},

    # learning_rate: 2.5e-4 is CleanRL's CartPole value.  A 2-2.5x bracket
    # either side.  Beyond ~1e-3, vanilla DQN with a 500-step hard target sync
    # is routinely divergent rather than merely worse, so the bracket stops
    # before the regime where a bad result would say nothing useful.
    {"name": "lr5e-4",       "overrides": {"learning_rate": 5e-4},                        "change_rank": 1},
    {"name": "lr1e-4",       "overrides": {"learning_rate": 1e-4},                        "change_rank": 4},

    # train_frequency: sets the replay ratio.  Default 10 = 0.1 updates per
    # env step.  Fewer steps per update means each stored transition is
    # replayed more often, which is the other way to spend a small buffer
    # harder.  2x bracket either side.
    {"name": "tf5",          "overrides": {"train_frequency": 5},                         "change_rank": 2},
    {"name": "tf20",         "overrides": {"train_frequency": 20},                        "change_rank": 3},

    # buffer_size: the field D034 names as the suspect, so it gets the widest
    # bracket (0.3x to 30x).  10k at ~142 steps per episode is ~70 episodes of
    # history for a uniform mixture over 25 configurations.  Bracketed below
    # (3k) as well as above, because "does buffer size move this metric at all
    # in this range" is as informative as "does more help".
    {"name": "buf30k",       "overrides": {"buffer_size": 30_000},                        "change_rank": 5},
    {"name": "buf3k",        "overrides": {"buffer_size": 3_000},                         "change_rank": 6},
    {"name": "buf100k",      "overrides": {"buffer_size": 100_000},                       "change_rank": 7},
    {"name": "buf300k",      "overrides": {"buffer_size": 300_000},                       "change_rank": 8},
)

GRID_BY_NAME = {g["name"]: g for g in GRID}

SWEEP_ROOT = REPO_ROOT / "runs" / "hpsweep"


# ----------------------------------------------------------------------
# Running one (setting, seed)
# ----------------------------------------------------------------------

def git_commit() -> tuple[str, bool]:
    """(commit, dirty) for the whole tree, as experiment/manifest.py records it."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                                capture_output=True, text=True, check=True).stdout
        return commit, bool(status.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", False


def run_dir_for(name: str, seed: int) -> Path:
    # `__seed<N>__` so scripts/gate1_check.py's find_runs() could read these
    # directories unchanged if anyone ever wants to.
    return SWEEP_ROOT / name / f"combat-mix__{name}__seed{seed}__run"


def eval_path(name: str, seed: int) -> Path:
    return SWEEP_ROOT / name / f"eval_seed{seed}.json"


def train_one(name: str, seed: int, run_dir: Path) -> float:
    """Train via the ordinary CLI.  Returns wall seconds."""
    overrides = GRID_BY_NAME[name]["overrides"]
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "train_dqn.py"),
           "--set", f"env_id={ENV_ID}",
           "--set", f"total_timesteps={TOTAL_TIMESTEPS}",
           "--seed", str(seed),
           "--run-dir", str(run_dir)]
    for key, value in overrides.items():
        cmd += ["--set", f"{key}={value}"]

    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    wall = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"training failed for {name} seed {seed} (exit {proc.returncode})\n"
            f"cmd: {' '.join(cmd)}\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}")
    return wall


def evaluate_one(name: str, seed: int, run_dir: Path, n_eval: int) -> dict[str, Any]:
    """Gate 1 protocol, unchanged: evaluation/gate1.evaluate_policy at n_eval."""
    from dqn.checkpoint import latest_checkpoint, load_checkpoint
    from dqn.exploration import greedy_action
    from evaluation.gate1 import CONFIG_IDS, EVAL_SEED_BASE, evaluate_policy

    ckpt = latest_checkpoint(run_dir / "checkpoints")
    if ckpt is None:
        raise RuntimeError(f"no checkpoint under {run_dir / 'checkpoints'}")
    network, meta = load_checkpoint(ckpt)

    # Identical to scripts/gate1_check.py::greedy_policy.  The harness cannot
    # tell this from a floor policy, which is what makes them comparable.
    def policy(obs: np.ndarray) -> int:
        q, _ = network.forward(np.asarray(obs)[None])
        return int(greedy_action(q[0]))

    t0 = time.time()
    result = evaluate_policy(policy, n_eval=n_eval)
    eval_wall = time.time() - t0

    cfg = json.loads((run_dir / "config.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    commit, dirty = git_commit()
    return {
        "setting": name,
        "overrides": GRID_BY_NAME[name]["overrides"],
        "seed": seed,
        "mean_damage_taken": result["mean"],
        "spread": result["spread"],
        "mean_win_rate": float(np.mean(list(result["success_rate"].values()))),
        "damage_taken": {str(c): result["damage_taken"][c] for c in CONFIG_IDS},
        "success_rate": {str(c): result["success_rate"][c] for c in CONFIG_IDS},
        "saturated": {str(c): result["saturated"][c] for c in CONFIG_IDS},
        "n_eval": n_eval,
        "eval_seed_base": EVAL_SEED_BASE,
        "checkpoint": str(ckpt.relative_to(REPO_ROOT)),
        "checkpoint_step": meta["global_step"],
        "config": cfg,
        "train_summary": summary,
        "eval_wall_seconds": round(eval_wall, 1),
        "git_commit": commit,
        "git_dirty": dirty,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
    }


def do_one(name: str, seed: int, n_eval: int, force: bool) -> str:
    out = eval_path(name, seed)
    if out.exists() and not force:
        return f"skip   {name:>13} seed {seed}  (eval.json present)"
    run_dir = run_dir_for(name, seed)
    train_wall = 0.0
    if not (run_dir / "summary.json").exists() or force:
        if run_dir.exists():
            import shutil
            shutil.rmtree(run_dir)
        train_wall = train_one(name, seed, run_dir)
    record = evaluate_one(name, seed, run_dir, n_eval)
    record["train_wall_seconds"] = round(train_wall, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    return (f"done   {name:>13} seed {seed}  mean damage "
            f"{record['mean_damage_taken']:8.3f}  "
            f"({train_wall:.0f}s train, {record['eval_wall_seconds']:.0f}s eval)")


# ----------------------------------------------------------------------
# Reporting and D034's rule
# ----------------------------------------------------------------------

def final_training_stats(name: str, seed: int, last: int = 100) -> dict[str, float]:
    """Mean return and length over the last `last` training episodes.

    A diagnostic, not a criterion.  The reward the agent maximises
    (`docs/ENVIRONMENT_SPEC.md` §7) is not the quantity D035 ranks on: damage
    taken is one term of six, and the win bonus plus the time penalty pay for
    trading damage for a faster kill.  So a setting can improve what the agent
    is actually optimising without improving mean damage taken.  Reading the
    two side by side is what separates "the agent did not get better" from
    "the agent got better at something else".
    """
    path = run_dir_for(name, seed) / "episodes.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    tail = rows[-last:]
    if not tail:
        return {}
    return {
        "episodes": float(len(rows)),
        "final_return": float(np.mean([r["return"] for r in tail])),
        "final_length": float(np.mean([r["length"] for r in tail])),
        "timeout_fraction": float(np.mean([bool(r["truncated"]) for r in tail])),
    }


def load_results(n_eval: int) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for g in GRID:
        rows = []
        for seed in SEEDS:
            p = eval_path(g["name"], seed)
            if p.exists():
                row = json.loads(p.read_text())
                row["final_training"] = final_training_stats(g["name"], seed)
                rows.append(row)
        if rows:
            out[g["name"]] = rows
    return out


def report(n_eval: int) -> int:
    results = load_results(n_eval)
    if "default" not in results:
        raise SystemExit("the default setting has no results; it is the reference")

    defaults = current_defaults()
    print(f"Current DQNConfig defaults for the swept fields: {defaults}")
    print(f"D034 rule 2 bar: improvement in mean damage taken > {SD_GATE1}\n")

    means: dict[str, float] = {}
    extras: dict[str, dict[str, float]] = {}
    print(f"{'setting':>14} {'seeds':>6} {'per-seed mean damage':>34} "
          f"{'3-seed mean':>12} {'vs default':>11} {'win%':>6} "
          f"{'final ret':>10} {'sat cfgs':>9}")
    print("-" * 113)
    for g in GRID:
        name = g["name"]
        if name not in results:
            print(f"{name:>14} {'-':>6}  (no results)")
            continue
        rows = sorted(results[name], key=lambda r: r["seed"])
        vals = [r["mean_damage_taken"] for r in rows]
        means[name] = float(np.mean(vals))
        per_seed = "  ".join(f"{v:9.3f}" for v in vals)
        win = float(np.mean([r["mean_win_rate"] for r in rows]))
        ret = [r["final_training"].get("final_return") for r in rows
               if r.get("final_training")]
        ret_mean = float(np.mean(ret)) if ret and None not in ret else float("nan")
        # D035 flags a configuration whose episodes hit the health bound >25%.
        n_sat = float(np.mean([sum(1 for v in r["saturated"].values() if v > 0.25)
                               for r in rows]))
        extras[name] = {"mean_win_rate": win, "final_return": ret_mean,
                        "mean_flagged_configs": n_sat,
                        "mean_spread": float(np.mean([r["spread"] for r in rows]))}
        delta = means[name] - means.get("default", float("nan"))
        print(f"{name:>14} {len(vals):>6} {per_seed:>34} {means[name]:>12.3f} "
              f"{delta:>+11.3f} {win:>6.3f} {ret_mean:>10.1f} {n_sat:>9.1f}")

    ref = means["default"]
    print(f"\nReference (D034 rule 2): the default setting's own 3-seed mean, "
          f"{ref:.3f}.")
    print(f"  Gate 1 attempt, same seeds 1-3:  {GATE1_SEEDS_1_3_MEAN:.3f}")
    print(f"  Gate 1 attempt, all 10 seeds:    {GATE1_10SEED_MEAN:.3f}")

    # Rule 2.  Mean damage taken is a cost: an improvement is a decrease.
    print("\nD034 rule 2 — qualifying settings (improvement > "
          f"{SD_GATE1}):")
    qualifying = []
    for g in GRID:
        name = g["name"]
        if name == "default" or name not in means:
            continue
        improvement = ref - means[name]
        ok = improvement > SD_GATE1
        if ok:
            qualifying.append((g["change_rank"], name, improvement))
        print(f"  {name:>14}  improvement {improvement:>+8.3f}  "
              f"{'QUALIFIES' if ok else 'does not qualify'}")

    if not qualifying:
        print("\nD034 rule 5 — nothing cleared the bar.  The CartPole defaults "
              "stand, and the recorded result is that the agent is not "
              "hyperparameter-limited on this environment over this grid.")
        verdict: dict[str, Any] = {"qualifying": [], "recommendation": None,
                                   "rule_applied": 5}
    else:
        qualifying.sort()
        rank, name, improvement = qualifying[0]
        print(f"\nD034 rule 3 — {len(qualifying)} setting(s) qualified; the "
              f"smallest change from the defaults is adopted, not the "
              f"best-scoring one.")
        best_scoring = min(((means[n], n) for _, n, _ in qualifying))
        print(f"  smallest change: {name}  (change_rank {rank}, "
              f"improvement {improvement:+.3f})")
        print(f"  best scoring:    {best_scoring[1]}  "
              f"(3-seed mean {best_scoring[0]:.3f})")
        print(f"\nRecommendation: {name} -> "
              f"{GRID_BY_NAME[name]['overrides']}")
        print("D034 rule 4 — adopting it requires a full new Gate 1 attempt, "
              "which is Ryan's to run.  This script does not touch "
              "docs/DECISIONS.md or docs/PREREGISTRATION.md.")
        verdict = {"qualifying": [{"setting": n, "improvement": i}
                                  for _, n, i in qualifying],
                   "recommendation": name,
                   "recommended_overrides": GRID_BY_NAME[name]["overrides"],
                   "rule_applied": 3}

    commit, dirty = git_commit()
    summary = {
        "sd_gate1": SD_GATE1,
        "env_id": ENV_ID,
        "seeds": list(SEEDS),
        "total_timesteps": TOTAL_TIMESTEPS,
        "n_eval": n_eval,
        "current_defaults": defaults,
        "reference_mean": ref,
        "gate1_seeds_1_3_mean": GATE1_SEEDS_1_3_MEAN,
        "gate1_10seed_mean": GATE1_10SEED_MEAN,
        "setting_means": means,
        "setting_diagnostics": extras,
        "per_seed": {n: {str(r["seed"]): r["mean_damage_taken"]
                         for r in sorted(rows, key=lambda r: r["seed"])}
                     for n, rows in results.items()},
        "verdict": verdict,
        "git_commit": commit,
        "git_dirty": dirty,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
    }
    out = SWEEP_ROOT / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote {out}")
    return 0


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="train and evaluate the grid (resumable)")
    r.add_argument("--workers", type=int, default=2,
                   help="parallel runs; one BLAS thread each (default 2)")
    r.add_argument("--only", action="append", default=[], metavar="SETTING",
                   help="restrict to these grid settings; repeatable")
    r.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    r.add_argument("--n-eval", type=int, default=None,
                   help="evaluation episodes per configuration (default: the "
                        "Gate 1 protocol's N_EVAL, which is what D034 requires)")
    r.add_argument("--force", action="store_true", help="re-run completed cells")

    p = sub.add_parser("report", help="aggregate and apply D034's rule")
    p.add_argument("--n-eval", type=int, default=None)

    g = sub.add_parser("grid", help="print the grid and exit")

    args = ap.parse_args(argv)

    from evaluation.gate1 import N_EVAL
    n_eval = getattr(args, "n_eval", None) or N_EVAL

    if args.cmd == "grid":
        print(json.dumps([{k: v for k, v in g.items()} for g in GRID], indent=2))
        return 0

    if args.cmd == "report":
        return report(n_eval)

    names = args.only or [g["name"] for g in GRID]
    unknown = [n for n in names if n not in GRID_BY_NAME]
    if unknown:
        raise SystemExit(f"unknown setting(s): {unknown}")

    tasks = [(n, s) for n in names for s in args.seeds]
    print(f"{len(tasks)} cells, {args.workers} workers, n_eval={n_eval}")
    t0 = time.time()
    failures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(do_one, n, s, n_eval, args.force): (n, s)
                for n, s in tasks}
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            n, s = futs[fut]
            try:
                print(f"[{i}/{len(tasks)} {time.time() - t0:6.0f}s] {fut.result()}",
                      flush=True)
            except Exception as exc:          # noqa: BLE001 - report and continue
                failures.append((n, s, str(exc)))
                print(f"[{i}/{len(tasks)}] FAIL {n} seed {s}: {exc}", flush=True)

    print(f"\n{len(tasks) - len(failures)}/{len(tasks)} cells ok in "
          f"{time.time() - t0:.0f}s")
    if failures:
        print("failures:")
        for n, s, msg in failures:
            print(f"  {n} seed {s}: {msg.splitlines()[0]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
