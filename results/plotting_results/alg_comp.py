import argparse
import os
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

# Pretty labels (edit if you care)
TRANSLATIONS = {
    "safe_point_goal_12_cylinders": "SafePointGoal (12 Cylinders)",
    "safe_point_goal_mixed_hazards": "SafePointGoal (Mixed Hazards)",
    "reward": "Reward",
    "cost": "Cost",
    "ppo": "PPO",
    "ppo_lag": "PPO_Lag",
    "ppo_pid": "PPO_PID",
}

# Optional safety thresholds per env for the cost plot (None disables line)
SAFETY_THRESHOLDS: Dict[str, float] = {
    "safe_point_goal_12_cylinders": 15.0,
    "safe_point_goal_mixed_hazards": 15.0,
}


def load_runs(base: Path, env: str, algo: str, seeds: List[int], metrics: List[str]) -> Dict[Tuple[str, str, str], List[pd.DataFrame]]:
    """Return dict[(env, algo, metric)] -> list of per-seed DataFrames with ['_step', 'value']."""
    out: Dict[Tuple[str, str, str], List[pd.DataFrame]] = {}
    for metric in metrics:
        key = (env, algo, metric)
        out[key] = []
        col = METRIC_COLS[metric]
        for seed in seeds:
            fp = base / env / algo / f"seed_{seed}.parquet"
            if not fp.exists():
                # skip missing seed
                print(f"File not found: {fp}")
                continue
            df = pd.read_parquet(fp, engine="pyarrow")
            if "_step" not in df or col not in df:
                continue
            # Keep only the two columns we need and rename to a common schema
            d = df[["_step", col]].rename(columns={col: "value"}).dropna()
            # enforce numeric
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
    # sort each by step, then truncate to min length
    lens = [len(d) for d in dfs]
    T = min(lens)
    trimmed = [d.sort_values("_step", kind="mergesort").iloc[:T] for d in dfs]
    steps = trimmed[0]["_step"].to_numpy(copy=True)
    vals = np.stack([d["value"].to_numpy(copy=True) for d in trimmed], axis=0)  # [R, T]
    return steps, vals


def plot_metrics(data: Dict[Tuple[str, str, str], List[pd.DataFrame]], args: argparse.Namespace) -> None:
    plt.rcParams.update({"figure.dpi": 300})
    plt.style.use("seaborn-v0_8-paper")

    nrows = len(args.envs)
    ncols = len(args.metrics)
    # smaller figure size
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(2.25 * ncols, 2 * nrows),
        squeeze=True
    )
    # leave more space at bottom for global legend
    fig.subplots_adjust(left=0.0, right=1.0, top=0.90, bottom=0.18, wspace=0.28, hspace=0.68)

    # collect handles for a single legend at the bottom
    legend_handles: Dict[str, plt.Line2D] = {}

    for r, env in enumerate(args.envs):
        env_title = TRANSLATIONS.get(env, env)

        for c, metric in enumerate(args.metrics):
            ax = axs[r, c]
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
                ci = 1.96 * vals.std(axis=0) / np.sqrt(max(vals.shape[0], 1))

                line, = ax.plot(x, mean, label=algo)
                # make CI visible
                ax.fill_between(x, mean - ci, mean + ci, alpha=0.25)

                # only store one handle per algo for global legend
                if algo not in legend_handles:
                    legend_handles[algo] = line

            ax.set_xlabel("Steps")
            ax.set_ylabel(label_y)

            # safety threshold line (red dashed) with legend entry, no text on plot
            if metric == "cost" and not args.no_threshold:
                thr = SAFETY_THRESHOLDS.get(env, None)
                if thr is not None:
                    thr_line = ax.axhline(thr, linestyle="--", color="red")
                    if "Threshold" not in legend_handles:
                        legend_handles["Threshold"] = thr_line

            x_max = args.x_max if args.x_max is not None else x_max_for_axis
            ax.set_xlim(0.0, x_max)

            if args.grid:
                ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

        # center env name above both columns for this row
        left_bbox = axs[r, 0].get_position()
        right_bbox = axs[r, -1].get_position()
        row_x = 0.5 * (left_bbox.x0 + right_bbox.x1)
        row_y = right_bbox.y1 + 0.01
        fig.text(row_x, row_y, env_title, ha="center", va="bottom", fontsize=10)

    # single legend at the bottom
    if legend_handles:
        labels, handles = zip(*legend_handles.items())
        labels = [TRANSLATIONS.get(lbl, lbl) for lbl in labels]
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0), ncol=len(labels), fancybox=True, shadow=True)

    out_dir = Path(args.output_fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (args.out_name if args.out_name.endswith(".pdf") else f"{args.out_name}.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    print(f"Saved figure: {out_path}")


def main(args: argparse.Namespace) -> None:
    base = Path(__file__).parent.parent.resolve() / args.input
    # load all data upfront
    store: Dict[Tuple[str, str, str], List[pd.DataFrame]] = {}
    for env in args.envs:
        for algo in args.algos:
            loaded = load_runs(base, env, algo, args.seeds, args.metrics)
            store.update(loaded)
    plot_metrics(store, args)


def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot Safe-Brax Parquet results.")
    p.add_argument("--input", type=str, default="data",
                   help="Base directory with <env>/<algo>/seed_*.parquet")
    p.add_argument("--envs", type=str, nargs="+",
                   default=["safe_point_goal_12_cylinders", "safe_point_goal_mixed_hazards"])
    p.add_argument("--algos", type=str, nargs="+", default=["ppo", "ppo_lag", "ppo_pid"])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--metrics", type=str, nargs="+", default=["reward", "cost"],
                   choices=list(METRIC_COLS.keys()))
    p.add_argument("--x_max", type=int, default=3e8)
    p.add_argument("--total_iterations", type=float, default=None,
                   help="If set, x-axis is rescaled to this many env steps.")
    p.add_argument("--no_threshold", action="store_true", help="Hide safety threshold lines.")
    p.add_argument("--grid", action="store_true")
    p.add_argument("--output_fig_dir", type=str, default="figures")
    p.add_argument("--out_name", type=str, default="point_goal_baselines")
    return p


if __name__ == "__main__":
    args = build_args().parse_args()
    main(args)
