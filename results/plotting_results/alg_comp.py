import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

from results.common import (
    DEFAULT_METRIC_COLS as METRIC_COLS,
    get_series,
    align_and_stack,
    set_mpl_style,
    nice_grid,
    moving_average,
    BASELINES_COLORS,
)

# Pretty labels (edit if you care)
TRANSLATIONS = {
    "reward": "Reward",
    "cost": "Cost",
    "ppo": "PPO",
    "ppo_cost": "PPOCost",
    "ppo_lag": "PPOLag",
    "ppo_pid": "PPOPID",
    "ppo_saute": "PPOSaute",
    "p3o": "P3O",
    "focops": "FOCOPS",
}

# Optional safety thresholds per env for the cost plot (None disables line)
SAFETY_THRESHOLDS: Dict[str, float] = {
    "safe_point_goal": 25.0,
    "safe_reacher": 25.0,
    "safe_walker": 25.0,
}

# Per-env x-axis limits (env steps)
ENV_X_MAX: Dict[str, int] = {
    "safe_point_goal": 5_000_000_00,
    "safe_reacher": 5_000_000_00,
    "safe_walker": 5_000_000_00,
}


def load_runs(base: Path, env: str, level: int, algo: str, seeds: List[int], metrics: List[str]) -> Dict[
    Tuple[str, str, str], List[pd.DataFrame]]:
    """Return dict[(env, algo, metric)] -> list of per-seed DataFrames with ['_step', 'value'].

    Uses get_series to reconstruct true reward for ppo_cost by adding cost back.
    """
    out: Dict[Tuple[str, str, str], List[pd.DataFrame]] = {}
    for metric in metrics:
        key = (env, algo, metric)
        out[key] = []
        for seed in seeds:
            fp = base / env / f"level_{level}" / algo / f"seed_{seed}.parquet"
            if not fp.exists():
                # skip missing seed
                print(f"File not found: {fp}")
                continue
            df = pd.read_parquet(fp, engine="pyarrow")
            if "_step" not in df:
                continue
            series = get_series(df, algo=algo, metric=metric, metric_cols=METRIC_COLS, env_name=env)
            if series is None:
                continue
            d = pd.DataFrame({
                "_step": df["_step"].astype(np.int64),
                "value": series.astype(np.float32),
            }).dropna()
            out[key].append(d)
    return out


def plot_metrics(data: Dict[Tuple[str, str, str], List[pd.DataFrame]], args: argparse.Namespace) -> None:
    set_mpl_style()

    envs = args.envs
    metrics = args.metrics

    n_env = len(envs)
    nrows_env, ncols_env = nice_grid(n_env, max_cols=args.max_cols)

    m = len(metrics)
    total_rows = nrows_env
    total_cols = ncols_env * m

    fig_w = args.panel_w * total_cols
    fig_h = args.panel_h * total_rows
    fig, axs = plt.subplots(total_rows, total_cols, figsize=(fig_w, fig_h), squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.14, wspace=0.35, hspace=0.55)

    # map (env_i, metric_i) -> axis
    def get_ax(env_i: int, metric_i: int):
        gr = env_i // ncols_env
        gc = env_i % ncols_env
        r = gr
        c = gc * m + metric_i
        return axs[r, c]

    # collect handles for a single legend at the bottom
    legend_handles: Dict[str, plt.Line2D] = {}

    for env_i, env in enumerate(envs):
        env_title = env.replace("_", " ").title()

        for metric_i, metric in enumerate(metrics):
            ax = get_ax(env_i, metric_i)
            label_y = TRANSLATIONS.get(metric, metric.capitalize())

            # track max x for this axis so we can set xlim(0, xmax)
            x_max_for_axis = None

            for algo in args.algos:
                key = (env, algo, metric)
                runs = data.get(key, [])
                if not runs:
                    continue

                steps, vals = align_and_stack(runs)
                if vals.size == 0:
                    continue

                # Optionally rescale x to "environment steps" if user provided total_iterations
                if args.total_iterations is not None and len(steps) > 0:
                    x = np.linspace(0.0, args.total_iterations, num=len(steps), endpoint=True)
                else:
                    x = steps.astype(float)

                if len(x) == 0:
                    continue

                x_max_for_axis = x[-1] if x_max_for_axis is None else max(x_max_for_axis, x[-1])

                mean = vals.mean(axis=0)
                if args.smoothing_window:
                    mean = moving_average(mean, args.smoothing_window)
                ci = 1.96 * vals.std(axis=0) / np.sqrt(max(vals.shape[0], 1))
                if args.smoothing_window:
                    ci = moving_average(ci, args.smoothing_window)

                line, = ax.plot(x, mean, label=algo, color=BASELINES_COLORS.get(algo))
                # make CI visible
                ax.fill_between(x, mean - ci, mean + ci, alpha=0.25)

                # only store one handle per algo for global legend
                if algo not in legend_handles:
                    legend_handles[algo] = line

            ax.set_xlabel("Steps")
            ax.set_ylabel(label_y)

            y_max = ax.get_ylim()[1]
            if y_max >= 1000:
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            else:
                ax.ticklabel_format(axis="y", style="plain", useOffset=False) # Use plain style for non-scientific, no offset
            ax.yaxis.get_major_formatter().set_useOffset(False) # Ensure no offset is used for formatting

            # safety threshold line (red dashed) with legend entry, no text on plot
            if metric == "cost" and not args.no_threshold:
                thr = SAFETY_THRESHOLDS.get(env, None)
                if thr is not None:
                    thr_line = ax.axhline(thr, linestyle="--", color="red", linewidth=1.8)
                    if "Threshold" not in legend_handles:
                        legend_handles["Threshold"] = thr_line

            # Per-env x max takes priority, then CLI x_max, then data-driven max
            x_max = ENV_X_MAX.get(env, None)
            if x_max is None:
                x_max = args.x_max if args.x_max is not None else x_max_for_axis
            ax.set_xlim(0.0, x_max)

            if env == "safe_reacher":
                ax.set_ylim(-15, None)

            if args.grid:
                ax.grid(True, linestyle="--", linewidth=0.9, alpha=0.45)

        # Center env title above the metric group for this env
        left_bbox = get_ax(env_i, 0).get_position()
        right_bbox = get_ax(env_i, m - 1).get_position()
        row_x = 0.5 * (left_bbox.x0 + right_bbox.x1)
        row_y = max(left_bbox.y1, right_bbox.y1) + 0.01
        fig.text(row_x, row_y, env_title, ha="center", va="bottom", fontsize=14)

    # hide any unused env cells
    for env_i in range(n_env, nrows_env * ncols_env):
        for metric_i in range(m):
            ax = get_ax(env_i, metric_i)
            ax.axis("off")

    # single legend at the bottom
    if legend_handles:
        labels, handles = zip(*legend_handles.items())
        labels = [TRANSLATIONS.get(lbl, lbl) for lbl in labels]
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=min(len(labels), 10),
                   fancybox=True,
                   shadow=True)

    out_dir = Path(args.output_fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.out_name}_level_{args.level}.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    print(f"Saved figure: {out_path}")


def main(args: argparse.Namespace) -> None:
    base = Path(__file__).parent.parent.resolve() / args.input
    # load all data upfront
    store: Dict[Tuple[str, str, str], List[pd.DataFrame]] = {}
    for env in args.envs:
        for algo in args.algos:
            loaded = load_runs(base, env, args.level, algo, args.seeds, args.metrics)
            store.update(loaded)
    plot_metrics(store, args)


def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot Safe-Brax Parquet results.")
    p.add_argument("--input", type=str, default="data",
                   help="Base directory with <env>/<level>/<algo>/seed_*.parquet")
    p.add_argument("--envs", type=str, nargs="+", default=["safe_point_goal", "safe_reacher", "safe_walker", "safe_velocity"])
    p.add_argument("--algos", type=str, nargs="+",
                   default=["ppo", "ppo_cost", "ppo_lag", "ppo_pid", "ppo_saute", "p3o", "focops"])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--metrics", type=str, nargs="+", default=["reward", "cost"], choices=list(METRIC_COLS.keys()))
    p.add_argument("--x_max", type=int, default=5e8)
    p.add_argument("--total_iterations", type=float, default=None,
                   help="If set, x-axis is rescaled to this many env steps.")
    p.add_argument("--no_threshold", action="store_true", help="Hide safety threshold lines.")
    p.add_argument("--grid", action="store_true")
    p.add_argument("--smoothing_window", type=int, default=1, help="Moving average window size for smoothing.")
    p.add_argument("--max_cols", type=int, default=2, help="Max env columns in grid.")
    p.add_argument("--panel_w", type=float, default=3.1, help="Width per metric subplot.")
    p.add_argument("--panel_h", type=float, default=2.3, help="Height per env row.")
    p.add_argument("--output_fig_dir", type=str, default="figures")
    p.add_argument("--out_name", type=str, default="baselines")
    return p


if __name__ == "__main__":
    args = build_args().parse_args()
    main(args)
