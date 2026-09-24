"""Compare observation modalities: vector (state) vs. pixel observations.

Pixel runs are separated by the camera they render from

Able to plot both curves and bars, sharing the same loader and selection flags:

    # learning curves (default)
    python -m results.plotting_results.obs_comparison --envs safe_goal_point --algos ppo_lag

    # final-performance bars
    python -m results.plotting_results.obs_comparison --bars --envs safe_goal_point --algos ppo_lag

    # all modes side by side, algos grouped within each
    python -m results.plotting_results.obs_comparison --grouped \
        --envs safe_goal_point safe_push_point safe_circle_point

One figure per algorithm, since obs mode is already the line/bar dimension and
overlaying algorithms on top of it is unreadable. The exception is `--grouped`,
a single summary bar figure with the modes on the x axis and the algorithms
grouped within each.

Data layout (see `download.main_results.obs_mode_segment`):
    vector  -> data/<env>/level_<l>/<algo>/seed_<n>.parquet
    pixels  -> data/<env>/level_<l>/<algo>/vision_<camera>/seed_<n>.parquet

The external camera differs per task (`track` where `fixedfar` is too coarse to
resolve the objects, `fixedfar` elsewhere), so both are plotted as one
"allocentric" mode, opposite the egocentric one. See `common.ALLOCENTRIC_CAMERAS`.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from results import cli
from results.common import (
    BASELINES_COLORS,
    DEFAULT_METRIC_COLS as METRIC_COLS,
    DEFAULT_OBS_MODES,
    OBS_MODE_COLORS,
    TRANSLATIONS,
    align_and_stack,
    get_series,
    legend_ncol,
    moving_average,
    obs_mode_dir,
    results_path,
    set_mpl_style,
)

# (env, algo, obs_mode, metric) -> one DataFrame['_step', 'value'] per seed
RunStore = Dict[Tuple[str, str, str, str], List[pd.DataFrame]]

# Axis-tick labels for --grouped
SHORT_MODE_LABELS: Dict[str, str] = {
    "vector": "Vector",
    "vision_vision": "Ego",
    "vision_allocentric": "Allo",
    "vision_fixedfar": "Fixed",
    "vision_track": "Track",
    "vision_vision_back": "Rear",
}


def load_runs(base: Path, env: str, level: int, algo: str, obs_mode: str,
              seeds: List[int], metrics: List[str]) -> RunStore:
    """Load every seed of one (env, level, algo, obs_mode) cell."""
    out: RunStore = {}
    folder = base / env / f"level_{level}" / algo / obs_mode_dir(obs_mode, env)
    for metric in metrics:
        key = (env, algo, obs_mode, metric)
        out[key] = []
        for seed in seeds:
            fp = folder / f"seed_{seed}.parquet"
            if not fp.exists():
                continue
            df = pd.read_parquet(fp, engine="pyarrow")
            if "_step" not in df:
                continue
            series = get_series(df, algo=algo, metric=metric,
                                metric_cols=METRIC_COLS, env_name=env)
            if series is None:
                continue
            out[key].append(pd.DataFrame({
                "_step": df["_step"].astype(np.int64),
                "value": series.astype(np.float32),
            }).dropna())
    return out


def load_all(args: argparse.Namespace) -> RunStore:
    base = results_path(args.input)
    store: RunStore = {}
    for env in args.envs:
        for algo in args.algos:
            for obs_mode in args.obs_modes:
                store.update(load_runs(base, env, args.level, algo, obs_mode,
                                       args.seeds, args.metrics))
    return store


def _present_obs_modes(store: RunStore, args: argparse.Namespace, algo: str) -> List[str]:
    """Obs modes that actually have data for this algo, in the requested order."""
    return [
        om for om in args.obs_modes
        if any(store.get((env, algo, om, metric))
               for env in args.envs for metric in args.metrics)
    ]


def _present_algos(store: RunStore, args: argparse.Namespace, obs_mode: str) -> List[str]:
    """Algos that actually have data for this obs mode, in the requested order.

    Only `plot_grouped_bars` needs this. The per-figure plots split by algo.
    """
    return [
        algo for algo in args.algos
        if any(store.get((env, algo, obs_mode, metric))
               for env in args.envs for metric in args.metrics)
    ]


def _metric_grid(args: argparse.Namespace, bottom: Optional[float] = None):
    """One panel per (metric, env): metrics down the rows, envs across the columns.

    Reward on top and cost below.
    """
    n_rows, n_cols = len(args.metrics), len(args.envs)
    fig, axs = plt.subplots(n_rows, n_cols,
                            figsize=(args.panel_w * n_cols, args.panel_h * n_rows),
                            squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92,
                        bottom=0.12 if bottom is None else bottom,
                        wspace=0.25, hspace=0.30)
    return fig, axs


def _decorate(ax, args: argparse.Namespace, env: str, metric: str,
              row: int, col: int, handles: Dict[str, plt.Line2D]) -> None:
    """Shared per-panel labelling: env titles on top, metric names on the left."""
    if row == 0:
        ax.set_title(TRANSLATIONS.get(env, env), pad=8)
    if col == 0:
        ax.set_ylabel(TRANSLATIONS.get(metric, metric.capitalize()))

    if metric == "cost" and not args.no_threshold:
        thr = ax.axhline(args.threshold, linestyle="--", color="red", linewidth=1.8)
        handles.setdefault("Threshold", thr)


def _finalize(fig, handles: Dict[str, plt.Line2D], n_entries: int,
              args: argparse.Namespace, suffix: str, kind: str) -> None:
    """Attach the shared bottom legend and write the figure out."""
    if handles:
        labels, hs = zip(*handles.items())
        labels = [TRANSLATIONS.get(lbl, lbl) for lbl in labels]
        fig.legend(hs, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
                   ncol=legend_ncol(n_entries, len(labels)),
                   fancybox=True, shadow=True)

    out_dir = results_path(args.output_fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.out_name}_{kind}_level_{args.level}_{suffix}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def plot_curves(store: RunStore, args: argparse.Namespace, algo: str) -> None:
    """Reward/cost training curves, one line per observation modality."""
    set_mpl_style()
    envs, metrics = args.envs, args.metrics
    fig, axs = _metric_grid(args)

    handles: Dict[str, plt.Line2D] = {}
    obs_modes = _present_obs_modes(store, args, algo)

    for env_i, env in enumerate(envs):
        for metric_i, metric in enumerate(metrics):
            ax = axs[metric_i][env_i]

            non_negative = True

            for obs_mode in obs_modes:
                runs = store.get((env, algo, obs_mode, metric), [])
                if not runs:
                    continue
                steps, vals = align_and_stack(runs)
                if vals.size == 0:
                    continue
                mode_non_negative = float(vals.min()) >= 0.0
                non_negative &= mode_non_negative

                x = steps.astype(float)
                mean = vals.mean(axis=0)
                ci = 1.96 * vals.std(axis=0) / np.sqrt(max(vals.shape[0], 1))
                if args.smoothing_window:
                    mean = moving_average(mean, args.smoothing_window)
                    ci = moving_average(ci, args.smoothing_window)

                lower = mean - ci
                if mode_non_negative:
                    lower = np.maximum(lower, 0.0)

                line, = ax.plot(x, mean, label=obs_mode,
                                color=OBS_MODE_COLORS.get(obs_mode))
                ax.fill_between(x, lower, mean + ci, alpha=0.25,
                                color=line.get_color())
                handles.setdefault(obs_mode, line)

            if non_negative:
                ax.set_ylim(bottom=0.0)

            # Only the bottom row carries the x-axis label. The columns share it.
            if metric_i == len(metrics) - 1:
                ax.set_xlabel("Steps")
            if ax.get_ylim()[1] >= 1000:
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.yaxis.get_major_formatter().set_useOffset(False)
            ax.set_xlim(0.0, args.x_max)

            _decorate(ax, args, env, metric, metric_i, env_i, handles)
            if args.grid:
                ax.grid(True, linestyle="--", linewidth=0.9, alpha=0.45)

    _finalize(fig, handles, len(obs_modes), args, algo, "curves")


def _final_value(runs: List[pd.DataFrame], last_k: int) -> Optional[Tuple[float, float, int]]:
    """(mean, 95% CI half-width, n_seeds) of the last `last_k` points, over seeds."""
    if not runs:
        return None
    _, vals = align_and_stack(runs)
    if vals.size == 0:
        return None
    k = min(last_k, vals.shape[1])
    per_seed = vals[:, -k:].mean(axis=1)
    ci = 1.96 * per_seed.std() / np.sqrt(max(per_seed.size, 1))
    return float(per_seed.mean()), float(ci), int(per_seed.size)


def _clipped_yerr(heights: List[float], errors: List[float]) -> np.ndarray:
    """Asymmetric yerr whose lower whisker stops at zero.

    Reward and cost cannot be negative, so the lower half of a symmetric CI on a
    small mean would point at values no run ever reached.
    """
    lower = [min(err, height) if height >= 0 else err
             for height, err in zip(heights, errors)]
    return np.array([lower, errors])


def plot_bars(store: RunStore, args: argparse.Namespace, algo: str) -> None:
    """Final-performance bars, one bar per observation modality."""
    set_mpl_style()
    envs, metrics = args.envs, args.metrics
    fig, axs = _metric_grid(args)

    handles: Dict[str, plt.Line2D] = {}
    obs_modes = _present_obs_modes(store, args, algo)

    for env_i, env in enumerate(envs):
        for metric_i, metric in enumerate(metrics):
            ax = axs[metric_i][env_i]

            positions, heights, errors, colors, drawn = [], [], [], [], []
            for i, obs_mode in enumerate(obs_modes):
                stat = _final_value(store.get((env, algo, obs_mode, metric), []), args.last_k)
                if stat is None:
                    continue
                mean, ci, n_seeds = stat
                positions.append(i)
                heights.append(mean)
                errors.append(ci)
                colors.append(OBS_MODE_COLORS.get(obs_mode))
                drawn.append(obs_mode)

            if positions:
                bars = ax.bar(positions, heights, yerr=_clipped_yerr(heights, errors),
                              capsize=4, color=colors, edgecolor="black", linewidth=0.6)
                for obs_mode, bar in zip(drawn, bars):
                    handles.setdefault(obs_mode, bar)
                if min(heights) >= 0.0:
                    ax.set_ylim(bottom=0.0)

            ax.set_xticks(range(len(obs_modes)))
            # Bars are labelled by the legend; the tick marks only anchor them.
            ax.set_xticklabels([""] * len(obs_modes))
            ax.axhline(0.0, color="black", linewidth=0.8)

            _decorate(ax, args, env, metric, metric_i, env_i, handles)
            if args.grid:
                ax.grid(True, axis="y", linestyle="--", linewidth=0.9, alpha=0.45)

    _finalize(fig, handles, len(obs_modes), args, algo, "bars")


def plot_grouped_bars(store: RunStore, args: argparse.Namespace) -> None:
    """Obs modes on the x axis, algos as grouped bars within each mode.

    Every other mode here splits obs mode across figures. This one keeps all of
    them side by side so the modality gap and its consistency across algorithms
    are readable in a single panel.
    """
    set_mpl_style()
    envs, metrics = args.envs, args.metrics
    # A single row of panels leaves no gap under the axes for the figure-level
    # legend, which would then land on top of the bars. Reserve the band here.
    fig, axs = _metric_grid(args, bottom=0.30 if len(metrics) == 1 else 0.15)

    handles: Dict[str, plt.Line2D] = {}
    obs_modes = [om for om in args.obs_modes if _present_algos(store, args, om)]
    algos = [
        algo for algo in args.algos
        if any(store.get((env, algo, om, metric))
               for env in envs for om in obs_modes for metric in metrics)
    ]
    group_w = 0.8
    bar_w = group_w / max(len(algos), 1)

    for env_i, env in enumerate(envs):
        for metric_i, metric in enumerate(metrics):
            ax = axs[metric_i][env_i]

            panel_min = 0.0
            for algo_i, algo in enumerate(algos):
                positions, heights, errors = [], [], []
                for group_i, obs_mode in enumerate(obs_modes):
                    stat = _final_value(store.get((env, algo, obs_mode, metric), []),
                                        args.last_k)
                    if stat is None:
                        # Missing cell: leave its slot empty rather than
                        # shifting the rest, so bars stay aligned by algo.
                        continue
                    mean, ci, n_seeds = stat
                    positions.append(group_i - group_w / 2 + bar_w * (algo_i + 0.5))
                    heights.append(mean)
                    errors.append(ci)

                if not positions:
                    continue
                panel_min = min(panel_min, min(heights))
                bars = ax.bar(positions, heights, width=bar_w,
                              yerr=_clipped_yerr(heights, errors), capsize=3,
                              color=BASELINES_COLORS.get(algo),
                              edgecolor="black", linewidth=0.6)
                handles.setdefault(algo, bars[0])

            if panel_min >= 0.0:
                ax.set_ylim(bottom=0.0)

            ax.set_xticks(range(len(obs_modes)))
            ax.set_xticklabels([SHORT_MODE_LABELS.get(om, TRANSLATIONS.get(om, om))
                                for om in obs_modes])
            ax.axhline(0.0, color="black", linewidth=0.8)

            _decorate(ax, args, env, metric, metric_i, env_i, handles)
            if args.grid:
                ax.grid(True, axis="y", linestyle="--", linewidth=0.9, alpha=0.45)

    _finalize(fig, handles, len(algos), args, "grouped", "bars")


def main(args: argparse.Namespace) -> None:
    store = load_all(args)
    if not any(store.values()):
        raise SystemExit(
            "No data loaded. Check --input/--envs/--algos/--level/--seeds, and that "
            "the pixel runs were downloaded (main_results.py --obs vision)."
        )

    if args.grouped:
        plot_grouped_bars(store, args)
        return

    for algo in args.algos:
        if not any(store.get((env, algo, om, metric))
                   for env in args.envs for om in args.obs_modes for metric in args.metrics):
            print(f"No data for algo '{algo}', skipping.")
            continue
        if args.bars:
            plot_bars(store, args, algo)
        else:
            plot_curves(store, args, algo)


def build_args() -> argparse.ArgumentParser:
    p = cli.plot_parser(
        "Compare vector vs. pixel observations (per camera) for CRAX runs.",
        level_arg="single",
        omit=("ci_method", "last_frac", "max_cols"),
        stats=True,
        out_name="obs_comparison",
        envs=["safe_goal_point", "safe_push_point", "safe_circle_point"],
        algos=["ppo", "ppo_lag", "p3o", "focops"],
        panel_w=4.0,
        panel_h=2.5,
    )
    p.add_argument("--obs_modes", type=str, nargs="+", default=list(DEFAULT_OBS_MODES),
                   help="Observation modes to compare. 'vector' is the state-observation "
                        "baseline, 'vision_vision' the agent's own (egocentric) camera, and "
                        "'vision_allocentric' the external view, resolved per env to "
                        "whichever camera that env used.")
    p.add_argument("--bars", action="store_true",
                   help="Draw final-performance bars instead of training curves.")
    p.add_argument("--grouped", action="store_true",
                   help="A single bar chart with the observation modes on the x axis and "
                        "the algorithms grouped within each mode.")
    p.add_argument("--x_max", type=float, default=5e8, help="Curves only: upper x limit (env steps)")
    p.add_argument("--last_k", type=int, default=10,
                   help="--bars only: logged points averaged to get each run's final value.")
    p.add_argument("--no_threshold", action="store_true", help="Hide safety threshold lines.")
    return p


if __name__ == "__main__":
    main(build_args().parse_args())
