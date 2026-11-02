"""Regenerate every report figure from committed data and from Gate 1a runs.

Figures are written to `figures/` as both PNG (300 dpi) and PDF.  Nothing is
drawn from a number typed into this file: every value comes from one of the
committed JSON result files or from a Gate 1a run directory on disk.

Data sources
------------
`docs/gate1_floors.json`       per-configuration mean damage taken, win rate and
                               saturation for the random and scripted floor
                               policies, 200 evaluation episodes per
                               configuration (D034).
`docs/gate1_result.json`       per-seed, per-configuration mean damage taken for
                               the 10 trained agents, plus the G1-a threshold
                               (D034 attempt of 2026-11-01).
`docs/gate1a_calibration.json` the Gate 1a thresholds T, k, F, delta (D029).
`runs/gate1a/cleanrl/`         10 CleanRL reference runs, seeds 1-10.
`runs/gate1a/numpy/`           10 NumPy-DQN runs, seeds 1-10.

Figures 1-4 need only the committed JSON.  Figure 5 needs the run directories,
which are gitignored and must be regenerated:

    python scripts/run_cleanrl_reference.py --seeds 1 2 3 4 5 6 7 8 9 10
    for s in 1 2 3 4 5 6 7 8 9 10; do \
        python scripts/train_dqn.py --config configs/dqn/cartpole_cleanrl.json \
            --seed $s --runs-root runs/gate1a/numpy --tag numpy; done

Set `OMP_NUM_THREADS=1` per worker.  Pass `--skip-runs` to draw only the
figures that need no regeneration.

Usage
-----
    python scripts/make_figures.py                      # all five figures
    python scripts/make_figures.py --skip-runs          # figures 1-4 only
    python scripts/make_figures.py --only 2 --only 4
    python scripts/make_figures.py --formats png svg

Metric definitions are not re-implemented here.  The Gate 1a quantities
(100-episode moving average, steps-to-threshold, final return, IQM, bootstrap
draw order) are imported from `scripts/gate1a.py`, which is the authority for
them under D029.
"""

from __future__ import annotations

# Make this script runnable from any directory: put the repo root (this
# file's parent's parent) on sys.path so repo packages import without
# PYTHONPATH.  Resolved from __file__, so a git worktree imports its own copy.
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

import argparse
import itertools
import json
import textwrap
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

import gate1a  # scripts/gate1a.py: the D029 metric definitions

REPO_ROOT = Path(__file__).resolve().parents[1]

# ----------------------------------------------------------------------
# Environment constants.  Pre-registration section 5.1: two axes, five
# levels each, config_id = telegraph_level * 5 + gap_level.
# ----------------------------------------------------------------------
N_CONFIGS = 25
TELEGRAPH_S = (1.00, 0.66, 0.42, 0.24, 0.12)   # seconds, easiest first
ATTACK_GAP_S = (1.60, 1.25, 0.95, 0.72, 0.52)  # seconds, easiest first
SEEDS = tuple(range(1, 11))                    # D029/D034: seeds 1-10

TELEGRAPH_LEVEL = np.array([c // 5 for c in range(N_CONFIGS)])
GAP_LEVEL = np.array([c % 5 for c in range(N_CONFIGS)])

# The "monotone difficulty proxy".  D034's diagnostics quote a Spearman
# correlation of +0.915 against "a monotone difficulty proxy", but no
# committed file defines it, so that exact number cannot be reproduced from
# the repository.  Rather than reprint it, this script states its own
# definition -- the sum of the two axis level indices, which is monotone
# non-decreasing in both axes by construction -- and plots the correlation it
# computes.  The definition should be recorded in DECISIONS.md and this
# comment removed once it is.
PROXY = TELEGRAPH_LEVEL + GAP_LEVEL
PROXY_LABEL = "monotone axis-level proxy\n(telegraph level + gap level)"

# ----------------------------------------------------------------------
# Style.  Okabe-Ito categorical hues (assigned in fixed order, never cycled)
# and cividis for magnitude: both are chosen for colour-vision deficiency,
# and cividis is perceptually uniform and survives greyscale printing.
# Every categorical series also carries a distinct marker or line style, so
# identity is never colour-alone.
# ----------------------------------------------------------------------
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
INK = "#1a1a1a"
MUTED = "#5c5c5c"
FAINT = "#9a9a9a"
SEQ = "cividis"

PLOT_STYLE = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "axes.labelcolor": INK,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.7,
    "axes.titlepad": 6.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "text.color": INK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
}


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def _grid(mapping: dict[str, float]) -> np.ndarray:
    """A 25-vector indexed by config_id from a JSON object keyed by its string."""
    return np.array([float(mapping[str(c)]) for c in range(N_CONFIGS)])


def load_floors(path: Path) -> dict[str, Any]:
    """The two floor policies' per-configuration measurements (docs/gate1_floors.json)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, Any] = {"meta": {k: v for k, v in raw.items() if k != "floors"}}
    for name, block in raw["floors"].items():
        out[name] = {
            "damage_taken": _grid(block["damage_taken"]),
            "success_rate": _grid(block["success_rate"]),
            "saturated": _grid(block["saturated"]),
            "mean": float(block["mean"]),
            "spread": float(block["spread"]),
        }
    return out


def load_agent(path: Path) -> dict[str, Any]:
    """Per-seed trained-agent measurements (docs/gate1_result.json).

    Returns the 10 x 25 matrix of per-configuration mean damage taken, the
    per-seed grand means as recorded, the G1-a threshold, and the saturation
    flags.  The per-seed means are checked against the row means of the matrix
    so a corrupted file is caught rather than plotted.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    per = raw["per_seed"]
    seeds = tuple(sorted((int(s) for s in per), key=int))
    if seeds != SEEDS:
        raise ValueError(f"expected seeds {SEEDS}, found {seeds}")

    damage = np.vstack([_grid(per[str(s)]["damage_taken"]) for s in seeds])
    seed_means = np.array([float(per[str(s)]["mean_damage"]) for s in seeds])
    if not np.allclose(damage.mean(axis=1), seed_means):
        raise ValueError("per-seed mean_damage disagrees with its own damage_taken grid")

    return {
        "seeds": seeds,
        "damage": damage,                    # (10, 25)
        "seed_means": seed_means,            # (10,)
        "pooled": damage.mean(axis=0),       # (25,) pooled over seeds
        "threshold": float(raw["threshold"]),
        "n_eval": int(raw["n_eval"]),
        "passed": np.array([bool(per[str(s)]["passed"]) for s in seeds]),
        "flagged": {s: list(per[str(s)]["flagged_configs"]) for s in seeds},
    }


def load_gate1a_runs(root: Path, subdir: str) -> dict[int, list[dict[str, Any]]]:
    """Episode records for seeds 1-10 under `root / subdir`.

    Uses `gate1a.find_runs` and `gate1a.load_episodes`, so a smoke run, a
    duplicate seed or an unfinished run raises instead of being plotted.
    """
    runs = gate1a.find_runs(root / subdir, seeds=SEEDS)
    return {seed: gate1a.load_episodes(runs[seed]) for seed in SEEDS}


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------
def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation, average ranks for ties.

    Implemented here rather than imported so the figures need no SciPy; it is
    Pearson's r on average ranks, which is the definition.
    """
    rx, ry = _average_ranks(np.asarray(x, float)), _average_ranks(np.asarray(y, float))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom == 0:
        raise ValueError("Spearman undefined: a vector has zero rank variance")
    return float((rx * ry).sum() / denom)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks from 1, ties sharing their average rank."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * ((i + 1) + (j + 1))
        i = j + 1
    return ranks


def iqm_bootstrap_ci(
    per_seed: np.ndarray,
    n_resamples: int = gate1a.N_RESAMPLES,
    seed: int = gate1a.BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interquartile mean and a stratified bootstrap CI, per column.

    `per_seed` is (n_seeds, n_points).  Following Agarwal et al. (2021,
    pre-registration section 13), the point estimate is the interquartile mean
    across seeds and the interval comes from resampling runs with replacement
    -- here a single stratum, because Gate 1a has one task, so the stratified
    bootstrap reduces to resampling the 10 seeds.  One index matrix is drawn
    for all columns, so the band describes a resampled *set of runs* rather
    than independently resampled points, which is the point of the method.

    Returns (iqm, lower, upper).  Columns where any seed is missing (NaN) are
    returned as NaN rather than filled in.
    """
    n_seeds = per_seed.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_seeds, size=(n_resamples, n_seeds))

    point = _iqm_columns(per_seed)
    draws = np.full((n_resamples, per_seed.shape[1]), np.nan)
    for r in range(n_resamples):
        draws[r] = _iqm_columns(per_seed[idx[r]])

    usable = ~np.isnan(point)
    lower = np.full_like(point, np.nan)
    upper = np.full_like(point, np.nan)
    if usable.any():
        lower[usable] = np.percentile(draws[:, usable], 100 * alpha / 2, axis=0)
        upper[usable] = np.percentile(draws[:, usable], 100 * (1 - alpha / 2), axis=0)
    return point, lower, upper


def _iqm_columns(block: np.ndarray) -> np.ndarray:
    """Column-wise interquartile mean; NaN for any column with a missing seed."""
    out = np.full(block.shape[1], np.nan)
    complete = ~np.isnan(block).any(axis=0)
    if complete.any():
        sub = np.sort(block[:, complete], axis=0)
        drop = block.shape[0] // 4
        out[complete] = sub[drop : block.shape[0] - drop].mean(axis=0)
    return out


def binned_return_curves(
    episode_lists: dict[int, list[dict[str, Any]]], bin_width: int, total_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-seed mean episodic return in fixed environment-step bins.

    An episode is assigned to the bin containing its `global_step`, the step at
    which it ended, and each bin holds the mean of the returns of the episodes
    in it.  No interpolation: a bin with no episodes for a seed stays NaN and
    is excluded from the IQM for that bin.

    Returns (bin_centres, curves) with curves shaped (n_seeds, n_bins).
    """
    edges = np.arange(0, total_steps + bin_width, bin_width)
    centres = 0.5 * (edges[:-1] + edges[1:])
    curves = np.full((len(episode_lists), len(centres)), np.nan)

    for row, seed in enumerate(sorted(episode_lists)):
        episodes = episode_lists[seed]
        steps = np.array([e["global_step"] for e in episodes], float)
        returns = np.array([e["return"] for e in episodes], float)
        which = np.clip(np.digitize(steps, edges) - 1, 0, len(centres) - 1)
        for b in range(len(centres)):
            in_bin = returns[which == b]
            if in_bin.size:
                curves[row, b] = in_bin.mean()
    return centres, curves


# ----------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------
def provenance(fig: plt.Figure, text: str, width: int = 118) -> None:
    """A small provenance note under the figure naming the data behind it.

    Wrapped by hand rather than by Matplotlib, whose auto-wrapping ignores the
    tight bounding box and so runs past the page edge.
    """
    wrapped = "\n".join(textwrap.wrap(" ".join(text.split()), width=width))
    fig.text(0.0, -0.025, wrapped, fontsize=6.5, color=FAINT, ha="left", va="top",
             linespacing=1.5)


def save(fig: plt.Figure, out_dir: Path, stem: str, formats: Sequence[str]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path)
        written.append(path)
    plt.close(fig)
    return written


def thousands(value: float, _pos: int | None = None) -> str:
    """Tick label in thousands of steps: 0, 100k, 200k ...

    Preferred over Matplotlib's scientific-notation offset, which it parks in
    the corner where it collides with the axis label.
    """
    if value == 0:
        return "0"
    return f"{value / 1000:,.0f}k"


def axis_ticks(ax: plt.Axes) -> None:
    """Grid-cell axes: telegraph duration down, attack gap across, in seconds."""
    ax.set_xticks(range(5), [f"{g:.2f}" for g in ATTACK_GAP_S])
    ax.set_yticks(range(5), [f"{t:.2f}" for t in TELEGRAPH_S])
    ax.set_xlabel("attack gap (s)")
    ax.set_ylabel("telegraph duration (s)")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


# ----------------------------------------------------------------------
# Figure 1 -- difficulty-grid heatmaps
# ----------------------------------------------------------------------
def figure_difficulty_grids(floors: dict[str, Any], agent: dict[str, Any],
                            out_dir: Path, formats: Sequence[str]) -> list[Path]:
    panels = [
        ("Random floor policy", floors["random"]["damage_taken"], "uniform random actions"),
        ("Scripted floor policy", floors["scripted"]["damage_taken"], "hand-written policy"),
        ("Trained agent", agent["pooled"], f"pooled over {len(agent['seeds'])} seeds"),
    ]
    vmax = max(float(np.max(v)) for _, v, _ in panels)
    vmin = min(float(np.min(v)) for _, v, _ in panels)

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8))
    for ax, (title, values, subtitle) in zip(axes, panels):
        grid = values.reshape(5, 5)
        im = ax.imshow(grid, cmap=SEQ, vmin=vmin, vmax=vmax, origin="upper")
        axis_ticks(ax)
        ax.set_title(f"{title}\n{subtitle}", fontsize=8.5, color=INK)
        mid = 0.5 * (vmin + vmax)
        for (r, c), value in np.ndenumerate(grid):
            ax.text(c, r, f"{value:.0f}", ha="center", va="center", fontsize=6.4,
                    color="white" if value < mid else "#20201a")
        if ax is not axes[0]:
            ax.set_ylabel("")

    cbar = fig.colorbar(im, ax=axes, fraction=0.022, pad=0.02)
    cbar.set_label("mean damage taken (health points per episode)", fontsize=7.5)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=2, labelsize=7)

    fig.suptitle("Mean damage taken per difficulty configuration, three measurement systems",
                 fontsize=10, y=1.04)
    provenance(fig, "Floors: docs/gate1_floors.json (200 evaluation episodes per configuration). "
                    "Agent: docs/gate1_result.json (200 episodes per configuration per seed, "
                    "10 seeds). Shared colour scale; cell labels rounded to whole health points.")
    return save(fig, out_dir, "fig1_difficulty_grids", formats)


# ----------------------------------------------------------------------
# Figure 2 -- three-way ordering agreement
# ----------------------------------------------------------------------
def figure_ordering_agreement(floors: dict[str, Any], agent: dict[str, Any],
                              out_dir: Path, formats: Sequence[str]) -> list[Path]:
    agent_rank = _average_ranks(agent["pooled"])
    series = [
        ("Random floor policy", _average_ranks(floors["random"]["damage_taken"]), BLUE, "o"),
        ("Scripted floor policy", _average_ranks(floors["scripted"]["damage_taken"]), VERMILLION, "s"),
        (PROXY_LABEL, _average_ranks(PROXY.astype(float)), GREEN, "^"),
    ]

    fig = plt.figure(figsize=(7.0, 3.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.12], wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot([0.5, 25.5], [0.5, 25.5], color=FAINT, lw=0.8, ls=(0, (4, 3)), zorder=1,
            label="identical ordering")

    for label, ranks, colour, marker in series:
        rho = spearman(agent["pooled"], ranks)
        ax.scatter(agent_rank, ranks, s=24, marker=marker, facecolor="none",
                   edgecolor=colour, linewidth=1.1, zorder=3,
                   label=f"{label.splitlines()[0]}   $\\rho$ = {rho:+.3f}")
    ax.set_xlim(0.5, 26.5)
    ax.set_ylim(-5.5, 25.5)
    ax.set_xticks([1, 5, 10, 15, 20, 25])
    ax.set_yticks([1, 5, 10, 15, 20, 25])
    ax.set_xlabel("difficulty rank, trained agent (1 = easiest of 25)")
    ax.set_ylabel("difficulty rank, comparison system")
    ax.set_title("A  Rank against the trained agent's ranking", loc="left", fontsize=9)
    ax.grid(True, lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_bounds(1, 25)
    ax.legend(loc="lower right", bbox_to_anchor=(1.02, -0.02), labelspacing=0.65,
              handletextpad=0.4, borderpad=0.2)

    # Panel B: a slope chart over the four orderings, one line per
    # configuration, coloured by the agent's rank so crossings stay traceable.
    # The ramp is truncated short of its pale end, which would disappear
    # against the white surface.
    axb = fig.add_subplot(gs[0, 1])
    columns = [("Trained\nagent", agent_rank),
               ("Random\nfloor", series[0][1]),
               ("Scripted\nfloor", series[1][1]),
               ("Axis-level\nproxy", series[2][1])]
    xs = np.arange(len(columns))
    cmap = matplotlib.colormaps[SEQ]
    for c in range(N_CONFIGS):
        ys = [col[1][c] for col in columns]
        colour = cmap(0.02 + 0.76 * (agent_rank[c] - 1) / (N_CONFIGS - 1))
        axb.plot(xs, ys, color=colour, lw=1.0, zorder=2)
        axb.plot(xs, ys, marker="o", ms=3.0, ls="none", color=colour, zorder=3)
    axb.set_xticks(xs, [c[0] for c in columns])
    axb.set_xlim(-0.3, len(columns) - 0.7)
    axb.set_ylim(25.9, 0.1)
    axb.set_yticks([1, 5, 10, 15, 20, 25])
    axb.set_ylabel("difficulty rank (1 = easiest)")
    axb.set_title("B  The same 25 configurations across the four orderings",
                  loc="left", fontsize=9)
    axb.set_xlabel("line colour: the configuration's rank under the agent, dark = easiest",
                   fontsize=7, color=FAINT, labelpad=8)
    axb.grid(True, axis="y", lw=0.5, alpha=0.6)
    axb.set_axisbelow(True)
    for spine in ("top", "right", "bottom"):
        axb.spines[spine].set_visible(False)
    axb.tick_params(axis="x", length=0)

    fig.suptitle("Rank agreement between four difficulty orderings of the same 25 configurations",
                 fontsize=10, y=1.02)
    provenance(fig, "Trained-agent ranking: mean damage taken pooled over 10 seeds, "
                    "docs/gate1_result.json. Floor rankings: docs/gate1_floors.json. "
                    "Axis-level proxy: telegraph level + gap level, defined in this script "
                    "(see its module docstring). Spearman rank correlation, average ranks for "
                    "ties; the proxy has ties by construction.")
    return save(fig, out_dir, "fig2_ordering_agreement", formats)


# ----------------------------------------------------------------------
# Figure 3 -- per-seed variability against the G1-a threshold
# ----------------------------------------------------------------------
def figure_seed_variability(floors: dict[str, Any], agent: dict[str, Any],
                            out_dir: Path, formats: Sequence[str]) -> list[Path]:
    seeds = np.array(agent["seeds"])
    means = agent["seed_means"]
    order = np.argsort(means)
    y = np.arange(len(seeds))

    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    random_mean = floors["random"]["mean"]
    threshold = agent["threshold"]
    scripted_mean = floors["scripted"]["mean"]

    top = len(seeds) - 0.3
    ax.axvspan(0, threshold, color=GREEN, alpha=0.07, lw=0, zorder=0)
    # Direct labels above the three reference lines rather than a legend box:
    # the plot area left of the threshold is occupied by the seeds themselves.
    for value, colour, style, label in (
        (scripted_mean, PURPLE, (0, (1, 2)), f"scripted floor\n{scripted_mean:.4f}"),
        (threshold, VERMILLION, "-", f"G1-a threshold\n{threshold:.4f}"),
        (random_mean, BLUE, (0, (5, 2)), f"random floor\n{random_mean:.4f}"),
    ):
        ax.axvline(value, color=colour, lw=1.1, ls=style, zorder=2)
        ax.annotate(label, (value, top), xytext=(0, 5), textcoords="offset points",
                    fontsize=6.8, color=colour, ha="center", va="bottom",
                    linespacing=1.4, annotation_clip=False)

    ax.hlines(y, means[order], threshold, color=FAINT, lw=0.7, zorder=2)
    ax.scatter(means[order], y, s=34, color=INK, zorder=4)
    for row, idx in enumerate(order):
        ax.annotate(f"{means[idx]:.3f}", (means[idx], row), textcoords="offset points",
                    xytext=(-7, 0), ha="right", va="center", fontsize=6.8, color=MUTED)

    ax.set_yticks(y, [f"seed {seeds[i]}" for i in order])
    ax.set_ylim(-0.7, len(seeds) - 0.3)
    ax.set_xlim(0, max(random_mean, float(means.max())) * 1.12)
    ax.set_xlabel("mean damage taken across the 25 configurations (health points per episode)")

    n_pass = int(agent["passed"].sum())
    ax.set_title(
        f"Each of the {len(seeds)} trained seeds against the pre-registered G1-a threshold\n"
        f"{n_pass} of {len(seeds)} seeds below the threshold; "
        f"seed means {means.min():.3f} to {means.max():.3f}",
        loc="left", fontsize=9.5, pad=26)
    ax.grid(True, axis="x", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    provenance(fig, "Each dot is one seed's mean damage taken over the 25 configurations, 200 "
                    "evaluation episodes per configuration: docs/gate1_result.json. Floor means "
                    "from docs/gate1_floors.json. The G1-a criterion is the threshold only; the "
                    "scripted floor is shown as context and is not a criterion (D034). Values as "
                    "recorded in those files; none rescaled.")
    return save(fig, out_dir, "fig3_seed_variability", formats)


# ----------------------------------------------------------------------
# Figure 4 -- rank stability across seeds
# ----------------------------------------------------------------------
def figure_rank_stability(floors: dict[str, Any], agent: dict[str, Any],
                          out_dir: Path, formats: Sequence[str]) -> list[Path]:
    seeds = agent["seeds"]
    damage = agent["damage"]
    n = len(seeds)

    matrix = np.full((n, n), np.nan)
    pairs = []
    for i, j in itertools.combinations(range(n), 2):
        rho = spearman(damage[i], damage[j])
        matrix[i, j] = matrix[j, i] = rho
        pairs.append(rho)
    pairs = np.array(pairs)

    pooled_vs = [
        ("vs random floor", spearman(agent["pooled"], floors["random"]["damage_taken"]), BLUE, "o"),
        ("vs scripted floor", spearman(agent["pooled"], floors["scripted"]["damage_taken"]), VERMILLION, "s"),
        ("vs axis-level proxy", spearman(agent["pooled"], PROXY.astype(float)), GREEN, "^"),
    ]

    fig = plt.figure(figsize=(7.0, 3.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.1], wspace=0.22)

    # Panel A: the lower triangle only -- the matrix is symmetric and the
    # diagonal is 1 by construction.  Every cell carries its value, so the
    # colour is a secondary cue and no colourbar is needed.
    ax = fig.add_subplot(gs[0, 0])
    shown = np.array(matrix)
    shown[np.triu_indices(n)] = np.nan
    vmin = float(pairs.min())
    ax.imshow(shown, cmap=SEQ, vmin=vmin, vmax=1.0)
    mid = 0.5 * (vmin + 1.0)
    for (r, c), value in np.ndenumerate(shown):
        if not np.isnan(value):
            ax.text(c, r, f"{value:.2f}".lstrip("0"), ha="center", va="center",
                    fontsize=5.4, color="white" if value < mid else "#20201a")
    ax.set_xticks(range(n), [str(s) for s in seeds])
    ax.set_yticks(range(n), [str(s) for s in seeds])
    ax.set_xlabel("training seed")
    ax.set_ylabel("training seed")
    ax.set_title("A  Spearman $\\rho$ between every pair of seeds\n"
                 "darker = weaker agreement", loc="left", fontsize=8.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    axb = fig.add_subplot(gs[0, 1])
    rng = np.random.default_rng(gate1a.BOOTSTRAP_SEED)
    jitter = rng.uniform(-0.055, 0.055, size=pairs.size)
    axb.scatter(pairs, jitter, s=18, facecolor="none", edgecolor=MUTED, linewidth=0.8,
                label=f"{pairs.size} pairs of single-seed orderings")
    axb.plot([pairs.mean()], [0.0], marker="|", ms=16, mew=2.0, color=INK, ls="none",
             label=f"mean of the pairs = {pairs.mean():+.3f}\n"
                   f"(range {pairs.min():+.3f} to {pairs.max():+.3f})")

    for offset, (label, rho, colour, marker) in enumerate(pooled_vs, start=1):
        axb.plot([rho], [offset * 0.3], marker=marker, ms=6.5, color=colour, ls="none",
                 label=f"pooled ordering {label}:  $\\rho$ = {rho:+.3f}")

    axb.set_ylim(-0.22, 0.3 * (len(pooled_vs) + 1.9))
    axb.set_yticks([])
    axb.set_xlim(0.0, 1.04)
    axb.set_xlabel("Spearman $\\rho$ between difficulty orderings of the 25 configurations")
    axb.set_title("B  Single-seed agreement against pooled-ordering agreement",
                  loc="left", fontsize=8.5)
    axb.grid(True, axis="x", lw=0.5, alpha=0.6)
    axb.set_axisbelow(True)
    axb.spines["left"].set_visible(False)
    axb.legend(loc="upper left", bbox_to_anchor=(-0.01, 1.0), labelspacing=0.75,
               handletextpad=0.5, borderpad=0.2, fontsize=6.9)

    fig.suptitle("Stability of the per-configuration difficulty ordering across training seeds",
                 fontsize=10, y=1.03)
    provenance(fig, "docs/gate1_result.json: per-seed, per-configuration mean damage taken, "
                    "200 evaluation episodes per cell. Panel B's pooled values use the "
                    "seed-pooled ordering against docs/gate1_floors.json and the axis-level "
                    "proxy defined in this script. The two quantities are both Spearman "
                    "correlations over the same 25 configurations.")
    return save(fig, out_dir, "fig4_rank_stability", formats)


# ----------------------------------------------------------------------
# Figure 5 -- Gate 1a learning curves (needs regenerated runs)
# ----------------------------------------------------------------------
def figure_gate1a_curves(runs_root: Path, calibration: dict[str, Any],
                         out_dir: Path, formats: Sequence[str],
                         bin_width: int = 5_000) -> list[Path]:
    numpy_eps = load_gate1a_runs(runs_root, "numpy")
    ref_eps = load_gate1a_runs(runs_root, "cleanrl")

    threshold = float(calibration["T"])
    factor = float(calibration["F"])
    delta = float(calibration["delta"])
    total = gate1a.TOTAL_TIMESTEPS

    groups = [
        ("NumPy DQN (from scratch)", numpy_eps, BLUE, "-", "o"),
        ("CleanRL reference", ref_eps, VERMILLION, (0, (5, 2)), "s"),
    ]

    fig = plt.figure(figsize=(7.0, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 1.0], hspace=0.52, wspace=0.26)
    ax = fig.add_subplot(gs[0, :])

    summary = {}
    for label, eps, colour, dash, marker in groups:
        centres, curves = binned_return_curves(eps, bin_width, total)
        point, low, high = iqm_bootstrap_ci(curves)
        ok = ~np.isnan(point)
        ax.fill_between(centres[ok], low[ok], high[ok], color=colour, alpha=0.18, lw=0)
        ax.plot(centres[ok], point[ok], color=colour, lw=1.6, ls=dash, label=label)

        steps, reached = zip(*(gate1a.steps_to_threshold(eps[s], threshold) for s in SEEDS))
        finals = [gate1a.final_return(eps[s]) for s in SEEDS]
        summary[label] = {
            "steps": np.array(steps, float),
            "reached": np.array(reached, bool),
            "finals": np.array(finals, float),
            "colour": colour,
            "marker": marker,
        }

    ax.axhline(threshold, color=INK, lw=0.9, ls=(0, (1, 2)),
               label=f"Gate 1a threshold T = {threshold:.0f}")
    ax.set_xlim(0, total)
    ax.set_xlabel("environment steps")
    ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    ax.set_ylabel("episodic return (CartPole-v1)")
    ax.set_title("A  Training return, interquartile mean over 10 seeds with 95% stratified "
                 "bootstrap interval", loc="left", fontsize=9)
    ax.grid(True, lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")

    # Panel B: steps-to-threshold per seed, with the medians and the F x rule.
    axb = fig.add_subplot(gs[1, 0])
    ref_median = float(np.median(summary["CleanRL reference"]["steps"]))
    for row, (label, block) in enumerate(summary.items()):
        jitter = np.linspace(-0.16, 0.16, len(SEEDS))
        censored = ~block["reached"]
        axb.scatter(block["steps"][~censored], np.full((~censored).sum(), row) + jitter[~censored],
                    s=24, facecolor="none", edgecolor=block["colour"], linewidth=1.0,
                    marker=block["marker"])
        if censored.any():
            axb.scatter(block["steps"][censored], np.full(censored.sum(), row) + jitter[censored],
                        s=30, color=block["colour"], marker="x", linewidth=1.1)
        median = float(np.median(block["steps"]))
        axb.plot([median, median], [row - 0.28, row + 0.28], color=INK, lw=1.6)
        axb.annotate(f"median {median:,.0f}", (median, row + 0.33), fontsize=6.8,
                     color=MUTED, ha="center", va="bottom")
    axb.axvline(factor * ref_median, color=VERMILLION, lw=1.0, ls=(0, (4, 2)))
    axb.annotate(f"{factor:g} x reference median\n= {factor * ref_median:,.0f}",
                 (factor * ref_median, -0.46), fontsize=6.8, color=VERMILLION,
                 ha="right", va="bottom", xytext=(-4, 0), textcoords="offset points")
    axb.set_yticks([0, 1], ["NumPy\nDQN", "CleanRL\nreference"])
    for tick, colour in zip(axb.get_yticklabels(), (BLUE, VERMILLION)):
        tick.set_color(colour)
    axb.set_ylim(-0.6, 1.6)
    axb.set_xlabel(f"environment steps to a 100-episode mean return of {threshold:.0f}")
    axb.set_title("B  Steps to threshold, per seed", loc="left", fontsize=9)
    axb.grid(True, axis="x", lw=0.5, alpha=0.6)
    axb.set_axisbelow(True)
    axb.tick_params(axis="y", length=0)
    axb.xaxis.set_major_formatter(FuncFormatter(thousands))

    # Panel C: final return per seed, the IQMs, and the non-inferiority margin.
    axc = fig.add_subplot(gs[1, 1])
    ref_iqm = gate1a.iqm(summary["CleanRL reference"]["finals"])
    for row, (label, block) in enumerate(summary.items()):
        jitter = np.linspace(-0.16, 0.16, len(SEEDS))
        axc.scatter(block["finals"], np.full(len(SEEDS), row) + jitter, s=24,
                    facecolor="none", edgecolor=block["colour"], linewidth=1.0,
                    marker=block["marker"])
        value = gate1a.iqm(block["finals"])
        axc.plot([value, value], [row - 0.28, row + 0.28], color=INK, lw=1.6)
        axc.annotate(f"IQM {value:.1f}", (value, row + 0.33), fontsize=6.8,
                     color=MUTED, ha="center", va="bottom")
    axc.axvline(ref_iqm - delta, color=VERMILLION, lw=1.0, ls=(0, (4, 2)))
    axc.annotate(f"reference IQM $-\\ \\delta$\n= {ref_iqm - delta:.2f}",
                 (ref_iqm - delta, -0.46), fontsize=6.8, color=VERMILLION,
                 ha="right", va="bottom", xytext=(-4, 0), textcoords="offset points")
    axc.set_yticks([0, 1], ["NumPy\nDQN", "CleanRL\nreference"])
    for tick, colour in zip(axc.get_yticklabels(), (BLUE, VERMILLION)):
        tick.set_color(colour)
    axc.set_ylim(-0.6, 1.6)
    axc.set_xlabel("mean return of the last 100 training episodes")
    axc.set_title("C  Final return, per seed", loc="left", fontsize=9)
    axc.grid(True, axis="x", lw=0.5, alpha=0.6)
    axc.set_axisbelow(True)
    axc.tick_params(axis="y", length=0)

    # Panels B and C need no legend box: the row labels are coloured to match
    # the series and every summary mark is labelled in place.  One marker has
    # no row label, so it is explained only when it actually occurs.
    censored_total = int(sum((~b["reached"]).sum() for b in summary.values()))
    if censored_total:
        axb.annotate("x = censored seed, never reached T", (0.0, -0.44),
                     xycoords="axes fraction", fontsize=6.8, color=MUTED,
                     ha="left", va="top")

    fig.suptitle("Gate 1a: the from-scratch NumPy DQN against the CleanRL reference on CartPole-v1",
                 fontsize=10, y=0.975)
    provenance(fig, "Regenerated runs: runs/gate1a/numpy and runs/gate1a/cleanrl, seeds 1-10, "
                    f"{total:,} environment steps each; episodic returns are logged with "
                    "exploration on, per D029. Thresholds T, F and delta from "
                    "docs/gate1a_calibration.json. Panel A bins episodes by the step at which "
                    f"they ended ({bin_width:,}-step bins, no interpolation); the interval is "
                    f"{gate1a.N_RESAMPLES:,} bootstrap resamples of the 10 runs "
                    "(Agarwal et al. 2021).")
    return save(fig, out_dir, "fig5_gate1a_learning_curves", formats)


# ----------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "figures",
                   help="where the figures are written (default: figures/)")
    p.add_argument("--docs-dir", type=Path, default=REPO_ROOT / "docs",
                   help="directory holding the committed Gate 1 / Gate 1a JSON")
    p.add_argument("--runs-root", type=Path, default=REPO_ROOT / "runs" / "gate1a",
                   help="directory holding the cleanrl/ and numpy/ run folders")
    p.add_argument("--skip-runs", action="store_true",
                   help="draw only the figures that need no regenerated runs (1-4)")
    p.add_argument("--only", type=int, action="append", choices=[1, 2, 3, 4, 5], default=None,
                   metavar="N", help="draw only figure N; repeatable")
    p.add_argument("--formats", nargs="+", default=["png", "pdf"],
                   help="output formats (default: png pdf)")
    p.add_argument("--bin-width", type=int, default=5_000,
                   help="environment-step bin width for the figure 5 curves")
    args = p.parse_args(argv)

    wanted = set(args.only) if args.only else {1, 2, 3, 4, 5}
    if args.skip_runs:
        wanted -= {5}

    floors = load_floors(args.docs_dir / "gate1_floors.json")
    agent = load_agent(args.docs_dir / "gate1_result.json")
    calibration = json.loads((args.docs_dir / "gate1a_calibration.json").read_text(encoding="utf-8"))

    written: list[Path] = []
    with plt.rc_context(PLOT_STYLE):
        if 1 in wanted:
            written += figure_difficulty_grids(floors, agent, args.out_dir, args.formats)
        if 2 in wanted:
            written += figure_ordering_agreement(floors, agent, args.out_dir, args.formats)
        if 3 in wanted:
            written += figure_seed_variability(floors, agent, args.out_dir, args.formats)
        if 4 in wanted:
            written += figure_rank_stability(floors, agent, args.out_dir, args.formats)
        if 5 in wanted:
            written += figure_gate1a_curves(args.runs_root, calibration,
                                            args.out_dir, args.formats, args.bin_width)

    for path in written:
        print(path.relative_to(REPO_ROOT) if REPO_ROOT in path.parents else path)
    if 5 not in wanted:
        print("figure 5 skipped: needs runs/gate1a/{numpy,cleanrl} (see --help)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
