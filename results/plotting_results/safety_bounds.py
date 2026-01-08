import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Map short metric names -> Parquet column names
METRIC_COLS = {
    "reward": "episodic/reward",
    "cost": "episodic/cost",
}

# Pretty labels
TRANSLATIONS = {
    "reward": "Reward",
    "cost": "Cost",
    "ppo_pid": "PPO_PID",
}


def _bound_value(bound: str) -> float:
    """Extract numeric value from a bound name like 'bound_10' -> 10.0."""
    try:
        return float(bound.split("_")[-1])
    except Exception:
        return np.nan


def load_runs(
        base: Path,
        env: str,
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
        col = METRIC_COLS[metric]
        for seed in seeds:
            fp = base / env / algo / bound / f"seed_{seed}.parquet"
            if not fp.exists():
                print(f"File not found: {fp}")
                continue
            df = pd.read_parquet(fp, engine="pyarrow")
            if "_step" not in df or col not in df:
                continue
            d = df[["_step", col]].rename(columns={col: "value"}).dropna()
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


def plot_metrics(
        data: Dict[Tuple[str, str, str], List[pd.DataFrame]],
        args: argparse.Namespace,
) -> None:
    plt.rcParams.update({"figure.dpi": 300})
    plt.style.use("seaborn-v0_8-paper")

    nrows = len(args.envs)
    ncols = len(args.metrics)

    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(2.4 * ncols, 2.1 * nrows),
        squeeze=True,
    )
    axs = np.atleast_2d(axs)

    fig.subplots_adjust(
        left=0.0,
        right=1.0,
        top=0.90,
        bottom=0.18,
        wspace=0.28,
        hspace=0.68,
    )

    # sort bounds by numeric value: smaller = stricter
    all_bounds = set(key[1] for key in data.keys())
    all_bounds = sorted(all_bounds, key=lambda x: float(x.split('_')[-1]))

    bound_vals = {b: _bound_value(b) for b in args.bounds}

    # one color family, darker for stricter (smaller) bounds
    colors = plt.get_cmap("tab20c")

    # Assign colors to bounds
    bound_colors = {bound: colors(4 + i) for i, bound in enumerate(all_bounds)}

    legend_handles: Dict[str, plt.Line2D] = {}

    for r, env in enumerate(args.envs):
        env_title = TRANSLATIONS.get(env, env)

        for c, metric in enumerate(args.metrics):
            ax = axs[r, c]
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
                    x = np.linspace(
                        0.0,
                        args.total_iterations,
                        num=len(steps),
                        endpoint=True,
                    )
                else:
                    x = steps.astype(float)

                if len(x) == 0:
                    continue

                x_max_for_axis = x[-1] if x_max_for_axis is None else max(
                    x_max_for_axis, x[-1]
                )

                mean = vals.mean(axis=0)
                ci = 1.96 * vals.std(axis=0) / np.sqrt(max(vals.shape[0], 1))

                bound_val = bound_vals[bound]
                label = f"bound_{bound_val:g}" if not np.isnan(bound_val) else bound
                color = bound_colors[bound]

                (line,) = ax.plot(x, mean, label=label, color=color)
                ax.fill_between(x, mean - ci, mean + ci, alpha=0.25, color=color)

                if label not in legend_handles:
                    legend_handles[label] = line

                # For cost: add dashed horizontal line at this bound value
                if metric == "cost" and not args.no_bound_lines and not np.isnan(bound_val):
                    ax.axhline(
                        bound_val,
                        linestyle=":",
                        # color=color,
                        linewidth=1.0,
                    )

            ax.set_xlabel("Steps")
            ax.set_ylabel(label_y)

            x_max = args.x_max if args.x_max is not None else x_max_for_axis
            if x_max is not None:
                ax.set_xlim(0.0, x_max)

            if args.grid:
                ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

        # center env name above row
        left_bbox = axs[r, 0].get_position()
        right_bbox = axs[r, -1].get_position()
        row_x = 0.5 * (left_bbox.x0 + right_bbox.x1)
        row_y = right_bbox.y1 + 0.01
        fig.text(
            row_x,
            row_y,
            env_title,
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # single legend at bottom
    if legend_handles:
        labels, handles = zip(*legend_handles.items())
        # labels = [TRANSLATIONS.get(lbl, lbl) for lbl in labels]
        labels = [label.replace('_', ' ').title() for label in labels]
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0),
            ncol=len(labels),
            fancybox=True,
            shadow=True,
        )

    out_dir = Path(args.output_fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        args.out_name if args.out_name.endswith(".pdf") else f"{args.out_name}.pdf"
    )
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    print(f"Saved figure: {out_path}")


def main(args: argparse.Namespace) -> None:
    base = Path(__file__).parent.parent.resolve() / args.input

    store: Dict[Tuple[str, str, str], List[pd.DataFrame]] = {}
    for env in args.envs:
        for bound in args.bounds:
            loaded = load_runs(
                base,
                env,
                args.algo,
                bound,
                args.seeds,
                args.metrics,
            )
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
        default=["safe_point_goal"],
    )
    p.add_argument(
        "--algo",
        type=str,
        default="ppo_pid",
        help="Algorithm subfolder to use (e.g., ppo_pid).",
    )
    p.add_argument(
        "--bounds",
        type=str,
        nargs="+",
        default=["bound_10", "bound_15", "bound_20"],
        help="Bound subfolders under the algo, e.g. bound_10 bound_15.",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3],
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
        default=3e8,
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
    p.add_argument(
        "--output_fig_dir",
        type=str,
        default="figures",
    )
    p.add_argument(
        "--out_name",
        type=str,
        default="point_goal_bounds",
    )
    return p


if __name__ == "__main__":
    args = build_args().parse_args()
    main(args)
