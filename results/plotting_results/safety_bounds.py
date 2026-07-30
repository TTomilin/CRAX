import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from results.common import align_and_stack, set_mpl_style, nice_grid, moving_average, REWARD_METRIC_MAP, \
    DEFAULT_REWARD_METRIC, TRANSLATIONS

# Map short metric names -> Parquet column names
METRIC_COLS = {
    "reward": "episodic/sum_reward",
    "cost": "episodic/cost",
}

# Per-env x-axis limits (env steps)
ENV_X_MAX: Dict[str, int] = {
    "safe_goal_point": 1_000_000_00,
    "safe_reacher": 5_000_000_00,
    "safe_block_push": 5_000_000_00,
    "safe_point_circle": 5_000_000_00,
}


def _bound_value(bound) -> float:
    """Extract numeric value from a bound identifier.

    Accepts:
    - int/float: returns float(bound)
    - str like 'bound_10' or 'safety_bound_10': parses trailing number
    """
    try:
        # numeric already
        if isinstance(bound, (int, float, np.integer, np.floating)):
            return float(bound)
        # string pattern
        if isinstance(bound, str):
            tail = bound.split("_")[-1]
            return float(tail)
    except Exception:
        pass
    return np.nan


def load_runs(
        base: Path,
        env: str,
        level: int,
        algo: str,
        bound: str,
        seeds: List[int],
        metrics: List[str],
) -> Dict[Tuple[str, str, str], List[pd.DataFrame]]:
    """
    Return dict[(env, bound, metric)] -> list of per-seed DataFrames with ['_step', 'value'].

    Folder structure:
        base / <env> / <algo> / <bound> / seed_<seed>.parquet
    """
    out: Dict[Tuple[str, str, str], List[pd.DataFrame]] = {}
    for metric in metrics:
        key = (env, bound, metric)
        out[key] = []

        # Determine the column name based on the environment and metric
        if metric == "reward":
            col_name = REWARD_METRIC_MAP.get(env, DEFAULT_REWARD_METRIC)
        elif metric == "cost":
            col_name = METRIC_COLS["cost"]
        else:
            col_name = METRIC_COLS.get(metric)

        if col_name is None:
            print(f"Warning: Unknown metric {metric} for env {env}. Skipping.")
            continue

        for seed in seeds:
            fp = base / env / f"level_{level}" / algo / f"safety_bound_{bound}" / f"seed_{seed}.parquet"
            if not fp.exists():
                print(f"File not found: {fp}")
                continue
            df = pd.read_parquet(fp, engine="pyarrow")
            if "_step" not in df or col_name not in df:
                print(f"Warning: Missing column {col_name} or _step in {fp}. Skipping.")
                continue
            d = df[["_step", col_name]].rename(columns={col_name: "value"}).dropna()
            d = d.astype({"_step": np.int64, "value": np.float32})
            out[key].append(d)
    return out


def align_and_stack(dfs: List[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align multiple runs by truncating to the min length after sorting by _step.
    Returns (steps, values) with shape [runs, T].
    """
    if not dfs:
        return np.array([]), np.array([[]])
    lens = [len(d) for d in dfs]
    T = min(lens)
    trimmed = [d.sort_values("_step", kind="mergesort").iloc[:T] for d in dfs]
    steps = trimmed[0]["_step"].to_numpy(copy=True)
    vals = np.stack([d["value"].to_numpy(copy=True) for d in trimmed], axis=0)  # [R, T]
    return steps, vals


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
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.0, wspace=0.45, hspace=0.75)

    def get_ax(env_i: int, metric_i: int):
        gr = env_i // ncols_env
        gc = env_i % ncols_env
        r = gr
        c = gc * m + metric_i
        return axs[r, c]

    # sort bounds by numeric value: smaller = stricter
    all_bounds = set(key[1] for key in data.keys())
    all_bounds = sorted(all_bounds)

    # map each discovered bound to its numeric value for drawing threshold lines
    bound_vals = {b: _bound_value(b) for b in all_bounds}

    # one color family
    colors = plt.get_cmap("tab20c")
    bound_colors = {bound: colors(4 + i) for i, bound in enumerate(all_bounds)}

    legend_handles: Dict[str, plt.Line2D] = {}

    for env_i, env in enumerate(envs):
        env_title = env.replace("_", " ").title()

        for metric_i, metric in enumerate(metrics):
            ax = get_ax(env_i, metric_i)
            label_y = TRANSLATIONS.get(metric, metric.capitalize())

            x_max_for_axis = None

            for bound in all_bounds:
                key = (env, bound, metric)
                runs = data.get(key, [])
                if not runs:
                    continue

                steps, vals = align_and_stack(runs)
                if vals.size == 0:
                    continue

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

                bound_val = bound_vals[bound]
                label_key = f"{bound_val:g}" if not np.isnan(bound_val) else str(bound)
                color = bound_colors[bound]

                (line,) = ax.plot(x, mean, label=f"Bound {label_key}", color=color)
                ax.fill_between(x, mean - ci, mean + ci, alpha=0.25, color=color)

                if label_key not in legend_handles:
                    legend_handles[label_key] = line

                # For cost: add dashed horizontal line at this bound value
                if metric == "cost" and not args.no_bound_lines and not np.isnan(bound_val):
                    ax.axhline(bound_val, linestyle=":", linewidth=1.8, color="darkgreen")

            ax.set_xlabel("Steps")
            ax.set_ylabel(label_y)

            y_max = ax.get_ylim()[1]
            if y_max >= 1000:
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            else:
                ax.ticklabel_format(axis="y", style="plain",
                                    useOffset=False)  # Use plain style for non-scientific, no offset
            ax.yaxis.get_major_formatter().set_useOffset(False)  # Ensure no offset is used for formatting

            # Per-env x max takes priority, then CLI x_max, then data-driven max
            x_max = ENV_X_MAX.get(env, None)
            if x_max is None:
                x_max = args.x_max if args.x_max is not None else x_max_for_axis
            ax.set_xlim(0.0, x_max)

            if args.grid:
                ax.grid(True, linestyle="--", linewidth=0.9, alpha=0.45)

        # center env title across metric group
        left_bbox = get_ax(env_i, 0).get_position()
        right_bbox = get_ax(env_i, m - 1).get_position()
        row_x = 0.5 * (left_bbox.x0 + right_bbox.x1)
        row_y = max(left_bbox.y1, right_bbox.y1) + 0.01
        fig.text(row_x, row_y, env_title, ha="center", va="bottom", fontsize=14)

    # hide unused env slots
    for env_i in range(n_env, nrows_env * ncols_env):
        for metric_i in range(m):
            ax = get_ax(env_i, metric_i)
            ax.axis("off")

    # single legend at bottom
    if legend_handles:
        labels, handles = zip(*legend_handles.items())
        labels = [f"Bound {label}" for label in labels]
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.3),
            ncol=min(len(labels), 10),
            fancybox=True,
            shadow=True,
        )

    out_dir = Path(args.output_fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        args.out_name if args.out_name.endswith(".pdf") else f"{args.out_name}_level_{args.level}.pdf"
    )
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    print(f"Saved figure: {out_path}")


def main(args: argparse.Namespace) -> None:
    base = Path(__file__).parent.parent.resolve() / args.input

    store: Dict[Tuple[str, str, str], List[pd.DataFrame]] = {}
    for env in args.envs:
        for bound in args.bounds:
            loaded = load_runs(base, env, args.level, args.algo, bound, args.seeds, args.metrics)
            store.update(loaded)

    plot_metrics(store, args)


def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot CRAX results for different safety bounds."
    )
    p.add_argument(
        "--input",
        type=str,
        default="data",
        help="Base directory with <env>/<algo>/<bound>/seed_*.parquet",
    )
    p.add_argument(
        "--envs",
        type=str,
        nargs="+",
        default=["safe_goal_point", "safe_reacher", "safe_block_push", "safe_point_circle"],
    )
    p.add_argument(
        "--algo",
        type=str,
        default="ppo_lag",
        help="Algorithm subfolder to use",
    )
    p.add_argument(
        "--bounds",
        type=int,
        nargs="+",
        default=[15, 25, 35],
        help="Safety bound subfolders under the algo",
    )
    p.add_argument(
        "--level",
        type=int,
        default=1,
        help="Difficulty level",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[*range(1, 11)],
    )
    p.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["reward", "cost"],
        choices=list(METRIC_COLS.keys()),
    )
    p.add_argument(
        "--x_max",
        type=float,
        default=5e8,
        help="Max x-axis limit (environment steps).",
    )
    p.add_argument(
        "--total_iterations",
        type=float,
        default=None,
        help="If set, x-axis rescaled to this many env steps.",
    )
    p.add_argument(
        "--no_bound_lines",
        action="store_true",
        help="Disable horizontal dashed lines at each bound in cost plots.",
    )
    p.add_argument(
        "--grid",
        action="store_true",
        help="Turn on grid.",
    )

    p.add_argument("--smoothing_window", type=int, default=1, help="Moving average window size for smoothing.")
    p.add_argument("--max_cols", type=int, default=2, help="Max env columns in grid.")
    p.add_argument("--panel_w", type=float, default=2.6, help="Width per metric subplot.")
    p.add_argument("--panel_h", type=float, default=1.8, help="Height per env row.")

    p.add_argument(
        "--output_fig_dir",
        type=str,
        default="figures",
    )
    p.add_argument(
        "--out_name",
        type=str,
        default="safety_bounds",
    )
    return p


if __name__ == "__main__":
    args = build_args().parse_args()
    main(args)
