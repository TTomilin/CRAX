"""Counterfactual hazard relabelling across environments and algorithms.

One column per (env, level, algo) setting and one row per metric, with a line
per relabelling mode (mean ± 95% CI across seeds). Settings wrap after
--max_cols columns. Reads the output of `results.download.counterfactual`:

    <input>/counterfactual/<env>/level_<k>/<algo>/<bound>/<mode>/seed_<n>.parquet
"""

import argparse
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from results import cli
from results.common import TRANSLATIONS, align_and_stack, moving_average, results_path, set_mpl_style
from results.plotting_results.counterfactual import MODE_COLORS, MODE_LABELS, load_runs

DEFAULT_SETTINGS = [
    "safe_pathway_walker2d:3:ppo_lag",
    "safe_pathway_walker2d:3:crpo",
    "safe_pathway_walker2d:3:p3o",
    "safe_pathway_walker2d:4:ppo_lag",
    "safe_pathway_walker2d:4:crpo",
    "safe_reacher:3:ppo_lag",
    "safe_reacher:3:crpo",
    "safe_button_point:3:ppo_lag",
    "safe_button_point:3:crpo",
]


def parse_setting(setting: str) -> Tuple[str, int, str]:
    env, level, algo = setting.split(":")
    return env, int(level), algo


def plot_overview(args: argparse.Namespace) -> None:
    set_mpl_style()
    settings = [parse_setting(s) for s in args.settings]
    metrics = args.metrics
    n_cols = min(len(settings), args.max_cols)
    n_blocks = -(-len(settings) // n_cols)
    # A thin empty row between blocks keeps a block's x labels clear of the next block's titles
    spacer = 0.35
    height_ratios = ([1] * len(metrics) + [spacer]) * n_blocks
    height_ratios = height_ratios[:-1]
    fig = plt.figure(figsize=(args.panel_w * n_cols, args.panel_h * sum(height_ratios)))
    grid = fig.add_gridspec(len(height_ratios), n_cols, height_ratios=height_ratios, hspace=0.25, wspace=0.35)
    axs = np.empty((n_blocks * len(metrics), n_cols), dtype=object)
    for block in range(n_blocks):
        for metric_i in range(len(metrics)):
            for col in range(n_cols):
                row = block * (len(metrics) + 1) + metric_i
                sharex = axs[block * len(metrics), col] if metric_i else None
                ax = fig.add_subplot(grid[row, col], sharex=sharex)
                ax.set_visible(False)
                axs[block * len(metrics) + metric_i, col] = ax

    legend_handles: Dict[str, plt.Line2D] = {}
    seed_counts: Dict[str, List[int]] = {mode: [] for mode in args.modes}
    for i, (env, level, algo) in enumerate(settings):
        block, col = divmod(i, n_cols)
        setting_args = argparse.Namespace(**{**vars(args), "env": env, "level": level, "algo": algo})
        data = load_runs(results_path(args.input), setting_args)
        for metric_i, metric in enumerate(metrics):
            ax = axs[block * len(metrics) + metric_i, col]
            ax.set_visible(True)
            x_max = 0.0
            for mode in args.modes:
                runs = data.get((args.bounds[0], mode, metric), [])
                if not runs:
                    continue
                steps, vals = align_and_stack(runs)
                if vals.size == 0:
                    continue
                if metric_i == 0:
                    seed_counts[mode].append(vals.shape[0])
                mean = vals.mean(axis=0)
                ci = 1.96 * vals.std(axis=0) / np.sqrt(vals.shape[0])
                if args.smoothing_window:
                    mean = moving_average(mean, args.smoothing_window)
                    ci = moving_average(ci, args.smoothing_window)
                color = MODE_COLORS.get(mode)
                x = steps.astype(float)
                x_max = max(x_max, x[-1])
                ax.plot(x, mean, color=color)
                ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.2)
                legend_handles.setdefault(mode, plt.Line2D([0], [0], color=color, lw=2.5))

            if metric == "cost" and not args.no_threshold:
                thr = ax.axhline(args.threshold, linestyle="--", color="red", linewidth=1.4)
                legend_handles.setdefault("threshold", thr)
            if metric_i == 0:
                env_title = TRANSLATIONS.get(env, env)
                ax.set_title(f"{env_title} L{level}, {TRANSLATIONS.get(algo, algo)}", fontsize=10)
            if metric_i == len(metrics) - 1:
                ax.set_xlabel("Steps")
            else:
                ax.tick_params(labelbottom=False)
                ax.xaxis.offsetText.set_visible(False)
            if col == 0:
                ax.set_ylabel(TRANSLATIONS.get(metric, metric.capitalize()))
            if x_max:
                ax.set_xlim(0, x_max)
            if ax.get_ylim()[1] >= 1000:
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            if args.grid:
                ax.grid(True, linestyle="--", linewidth=0.9, alpha=0.45)

    labels = []
    for key in legend_handles:
        if key == "threshold":
            labels.append("Threshold")
            continue
        counts = sorted(set(seed_counts[key]))
        n = f"n={counts[0]}" if len(counts) == 1 else f"n={counts[0]}-{counts[-1]}"
        labels.append(f"{MODE_LABELS.get(key, key)} ({n})")
    if legend_handles:
        fig.legend(list(legend_handles.values()), labels, loc="upper center", bbox_to_anchor=(0.5, 0.0),
                   ncol=len(legend_handles), fancybox=True, shadow=True)

    out_dir = results_path(args.output_fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.out_name}.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"Saved figure: {out_path}")


def build_args() -> argparse.ArgumentParser:
    p = cli.plot_parser(
        "Plot counterfactual hazard relabelling across environments and algorithms.",
        level_arg=None,
        omit=("envs", "algos"),
        out_name="counterfactual_overview",
        seeds=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        max_cols=3,
        panel_w=3.2,
        panel_h=2.4,
        smoothing_window=10,
    )
    p.add_argument("--settings", type=str, nargs="+", default=DEFAULT_SETTINGS,
                   help="Settings to draw, as env:level:algo")
    p.add_argument("--modes", type=str, nargs="+", default=["none", "optimistic", "adaptive_k10"],
                   help="Relabelling modes to draw")
    p.add_argument("--bounds", type=str, nargs=1, default=["observed_length"],
                   choices=["observed_length", "fixed_length"], help="Per-step bound conversion to draw")
    p.add_argument("--threshold", type=float, default=cli.DEFAULT_SAFETY_THRESHOLD,
                   help="Safety threshold drawn on the cost panels")
    p.add_argument("--no_threshold", action="store_true", help="Hide the safety threshold line")
    return p


if __name__ == "__main__":
    plot_overview(build_args().parse_args())
