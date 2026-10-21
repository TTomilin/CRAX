"""Training curves of counterfactual hazard relabelling.

One line per relabelling mode (none / random / optimistic), mean ± 95% CI across
seeds, for each per-step bound conversion requested with --bounds. Reads the
output of `results.download.counterfactual`:

    <input>/counterfactual/<env>/level_<k>/<algo>/<bound>/<mode>/seed_<n>.parquet
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from results import cli
from results.common import (
    DEFAULT_METRIC_COLS as METRIC_COLS,
    TRANSLATIONS,
    align_and_stack,
    get_series,
    moving_average,
    results_path,
    set_mpl_style,
)

MODE_LABELS = {
    "none": "No relabelling",
    "random": "Random relabelling",
    "optimistic": "Optimistic relabelling",
}
MODE_COLORS = {
    "none": "#2F2F2F",
    "random": plt.get_cmap("tab10")(0),
    "optimistic": plt.get_cmap("tab10")(1),
}
BOUND_LINESTYLES = {
    "observed_length": "-",
    "fixed_length": ":",
}
BOUND_LABELS = {
    "observed_length": "Observed episode length bound",
    "fixed_length": "Fixed episode length bound",
}

# (bound, mode, metric) -> list of per-seed DataFrames with ['_step', 'value']
RunStore = Dict[Tuple[str, str, str], List[pd.DataFrame]]


def load_runs(base: Path, args: argparse.Namespace) -> RunStore:
    out: RunStore = {}
    root = base / "counterfactual" / args.env / f"level_{args.level}" / args.algo
    for bound in args.bounds:
        for mode in args.modes:
            for metric in args.metrics:
                runs = out.setdefault((bound, mode, metric), [])
                for seed in args.seeds:
                    fp = root / bound / mode / f"seed_{seed}.parquet"
                    if not fp.exists():
                        continue
                    df = pd.read_parquet(fp, engine="pyarrow")
                    series = get_series(df, algo=args.algo, metric=metric, metric_cols=METRIC_COLS, env_name=args.env)
                    if series is None or "_step" not in df:
                        print(f"Warning: missing {metric} or _step in {fp}. Skipping.")
                        continue
                    d = pd.DataFrame({"_step": df["_step"].astype(np.int64), "value": series}).dropna()
                    runs.append(d)
            n = len(out[(bound, mode, args.metrics[0])])
            print(f"{bound}/{mode}: {n} seed(s)")
    return out


def plot_curves(data: RunStore, args: argparse.Namespace) -> None:
    set_mpl_style()

    metrics = args.metrics
    fig, axs = plt.subplots(1, len(metrics), figsize=(args.panel_w * len(metrics), args.panel_h), squeeze=False)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.85, bottom=0.2, wspace=0.35)

    legend_handles: Dict[str, plt.Line2D] = {}
    for metric_i, metric in enumerate(metrics):
        ax = axs[0, metric_i]
        for bound in args.bounds:
            for mode in args.modes:
                runs = data.get((bound, mode, metric), [])
                if not runs:
                    continue
                steps, vals = align_and_stack(runs)
                if vals.size == 0:
                    continue

                mean = vals.mean(axis=0)
                ci = 1.96 * vals.std(axis=0) / np.sqrt(vals.shape[0])
                if args.smoothing_window:
                    mean = moving_average(mean, args.smoothing_window)
                    ci = moving_average(ci, args.smoothing_window)

                color = MODE_COLORS.get(mode)
                x = steps.astype(float)
                ax.plot(x, mean, color=color, linestyle=BOUND_LINESTYLES[bound])
                ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.2)

                label = MODE_LABELS.get(mode, mode)
                if len(args.bounds) > 1:
                    label = f"{label}, {BOUND_LABELS[bound].lower()}"
                label = f"{label} (n={vals.shape[0]})"
                legend_handles.setdefault(label, plt.Line2D(
                    [0], [0], color=color, linestyle=BOUND_LINESTYLES[bound], lw=2.5))

        if metric == "cost" and not args.no_threshold:
            thr = ax.axhline(args.threshold, linestyle="--", color="red", linewidth=1.6)
            legend_handles.setdefault("Threshold", thr)

        ax.set_xlabel("Steps")
        ax.set_ylabel(TRANSLATIONS.get(metric, metric.capitalize()))
        ax.set_xlim(0, args.x_max)
        if ax.get_ylim()[1] >= 1000:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        else:
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.yaxis.get_major_formatter().set_useOffset(False)
        if args.grid:
            ax.grid(True, linestyle="--", linewidth=0.9, alpha=0.45)

    env_title = TRANSLATIONS.get(args.env, args.env)
    algo_title = TRANSLATIONS.get(args.algo, args.algo)
    fig.suptitle(f"{env_title} (Level {args.level}), {algo_title}", fontsize=14, y=0.98)

    if legend_handles:
        fig.legend(
            list(legend_handles.values()), list(legend_handles.keys()),
            loc="upper center", bbox_to_anchor=(0.5, 0.02),
            ncol=min(len(legend_handles), 2), fancybox=True, shadow=True,
        )

    out_dir = results_path(args.output_fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.out_name}_{args.env}_level_{args.level}.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    print(f"Saved figure: {out_path}")


def main(args: argparse.Namespace) -> None:
    plot_curves(load_runs(results_path(args.input), args), args)


def build_args() -> argparse.ArgumentParser:
    p = cli.plot_parser(
        "Plot counterfactual hazard relabelling training curves.",
        omit=("envs", "algos", "max_cols"),
        out_name="counterfactual",
        seeds=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        level=3,
        panel_w=3.6,
        panel_h=2.6,
    )
    p.add_argument("--env", type=str, default="safe_pathway_walker2d")
    p.add_argument("--algo", type=str, default="ppo_lag")
    p.add_argument("--modes", type=str, nargs="+", default=list(MODE_LABELS),
                   help="Relabelling modes to draw")
    p.add_argument("--bounds", type=str, nargs="+", default=["observed_length"],
                   choices=list(BOUND_LINESTYLES), help="Per-step bound conversions to draw")
    p.add_argument("--threshold", type=float, default=cli.DEFAULT_SAFETY_THRESHOLD,
                   help="Safety threshold drawn on the cost panel")
    p.add_argument("--no_threshold", action="store_true", help="Hide the safety threshold line")
    p.add_argument("--x_max", type=float, default=5e8, help="Max x-axis limit (environment steps)")
    return p


if __name__ == "__main__":
    main(build_args().parse_args())
