"""Compare normal training, curriculum, and transfer training methods.

This script generates:
1. Training curves grid comparing all methods
2. Final results on level 3 (bar plot comparison)

For curriculum and transfer, data from different stages/phases are connected.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    BASELINES_COLORS, TRANSLATIONS,
)

METHOD_LINESTYLES = {
    "normal": "-",
    "curriculum": "--",
    "transfer": ":",
}
METHOD_HATCHES = {
    "normal": "",
    "curriculum": "///",
    "transfer": "xx",
}

SAFETY_THRESHOLDS: Dict[str, float] = {
    "safe_goal_point": 25.0,
    "safe_reacher": 25.0,
    "safe_walker": 25.0,
}


def load_normal_runs(
        base: Path,
        env: str,
        level: int,
        algo: str,
        seeds: List[int],
        metrics: List[str],
) -> Dict[str, List[pd.DataFrame]]:
    """Load normal training runs (trained directly on target level)."""
    out: Dict[str, List[pd.DataFrame]] = {m: [] for m in metrics}

    for seed in seeds:
        fp = base / env / f"level_{level}" / algo / f"seed_{seed}.parquet"
        if not fp.exists():
            print(f"Normal run not found: {fp}")
            continue

        df = pd.read_parquet(fp, engine="pyarrow")
        if "_step" not in df.columns:
            continue

        for metric in metrics:
            series = get_series(df, algo=algo, metric=metric, metric_cols=METRIC_COLS, env_name=env)
            if series is None:
                continue
            d = pd.DataFrame({
                "_step": df["_step"].astype(np.int64),
                "value": series.astype(np.float32),
            }).dropna()
            out[metric].append(d)

    return out


def load_curriculum_runs(
        base: Path,
        env: str,
        algo: str,
        seeds: List[int],
        metrics: List[str],
        total_steps: int,
) -> Dict[str, List[pd.DataFrame]]:
    """Load curriculum training runs (trained across levels 1->2->3).

    Curriculum data uses global_step to track cumulative steps.
    """
    out: Dict[str, List[pd.DataFrame]] = {m: [] for m in metrics}

    curriculum_base = base / "curriculum" / env / algo

    for seed in seeds:
        fp = curriculum_base / f"seed_{seed}.parquet"
        if not fp.exists():
            print(f"Curriculum run not found: {fp}")
            continue

        df = pd.read_parquet(fp, engine="pyarrow")

        # Curriculum data should have global_step
        step_col = "global_step" if "global_step" in df.columns else "_step"
        if step_col not in df.columns:
            continue

        for metric in metrics:
            series = get_series(df, algo=algo, metric=metric, metric_cols=METRIC_COLS, env_name=env)
            if series is None:
                continue
            d = pd.DataFrame({
                "_step": df[step_col].astype(np.int64),
                "value": series.astype(np.float32),
            }).dropna()
            # Truncate to total_steps budget
            d = d[d["_step"] <= total_steps]
            out[metric].append(d)

    return out


def load_transfer_runs(
        base: Path,
        env: str,
        level: int,
        algo: str,
        seeds: List[int],
        metrics: List[str],
        unsafe_steps: int,
        total_steps: int,
) -> Dict[str, List[pd.DataFrame]]:
    """Load transfer training runs (PPO pretrain + safe fine-tuning).

    Transfer has two phases:
    - Unsafe PPO pretraining (stored separately)
    - Safe algorithm fine-tuning

    We concatenate them and offset the safe phase steps.
    """
    out: Dict[str, List[pd.DataFrame]] = {m: [] for m in metrics}

    transfer_base = base / "transfer" / env / f"level_{level}"
    pretrain_base = transfer_base / "ppo_pretrain"
    safe_base = transfer_base / algo

    for seed in seeds:
        # Try to load pretrain data
        pretrain_fp = pretrain_base / f"seed_{seed}.parquet"
        safe_fp = safe_base / f"seed_{seed}.parquet"

        if not safe_fp.exists():
            print(f"Transfer safe phase not found: {safe_fp}")
            continue

        safe_df = pd.read_parquet(safe_fp, engine="pyarrow")
        if "_step" not in safe_df.columns:
            continue

        # Load pretrain if available
        pretrain_df = None
        if pretrain_fp.exists():
            pretrain_df = pd.read_parquet(pretrain_fp, engine="pyarrow")
            if "_step" not in pretrain_df.columns:
                print(f"  Warning: Pretrain file missing _step column: {pretrain_fp}")
                pretrain_df = None
            else:
                print(f"  Loaded PPO pretrain for seed {seed}: {len(pretrain_df)} rows")
        else:
            print(f"  PPO pretrain not found for seed {seed}: {pretrain_fp}")

        for metric in metrics:
            frames = []

            # Pretrain phase (if available)
            if pretrain_df is not None:
                pretrain_series = get_series(
                    pretrain_df, algo="ppo", metric=metric, metric_cols=METRIC_COLS, env_name=env
                )
                if pretrain_series is not None:
                    p_df = pd.DataFrame({
                        "_step": pretrain_df["_step"].astype(np.int64),
                        "value": pretrain_series.astype(np.float32),
                    }).dropna()
                    # Truncate pretrain to unsafe_steps
                    p_df = p_df[p_df["_step"] <= unsafe_steps]
                    frames.append(p_df)

            # Safe phase - offset by unsafe_steps
            safe_series = get_series(safe_df, algo=algo, metric=metric, metric_cols=METRIC_COLS, env_name=env)
            if safe_series is not None:
                s_df = pd.DataFrame({
                    "_step": safe_df["_step"].astype(np.int64) + unsafe_steps,
                    "value": safe_series.astype(np.float32),
                }).dropna()
                # Truncate to total budget
                s_df = s_df[s_df["_step"] <= total_steps]
                frames.append(s_df)

            if frames:
                combined = pd.concat(frames, ignore_index=True).sort_values("_step")
                out[metric].append(combined)

    return out


def plot_training_curves(
        data_by_env: Dict[str, Dict[str, Dict[str, Dict[str, List[pd.DataFrame]]]]],
        args: argparse.Namespace,
) -> None:
    """
    One figure per env.
    Columns = metrics.
    Lines: algo = color, training type = linestyle.
    """
    set_mpl_style()

    metrics = args.metrics
    methods = ["normal", "curriculum", "transfer"]

    out_dir = Path(args.output_fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for env in args.envs:
        env_pack = data_by_env.get(env, {})
        m = len(metrics)

        fig_w = args.panel_w * m
        fig_h = args.panel_h
        fig, axs = plt.subplots(1, m, figsize=(fig_w, fig_h), squeeze=False)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.22, wspace=0.35)

        # Build legends: one for algos (colors), one for methods (linestyles)
        algo_handles = {}
        method_handles = {}

        for metric_i, metric in enumerate(metrics):
            ax = axs[0, metric_i]
            ax.set_xlabel("Steps")
            ax.set_ylabel(TRANSLATIONS.get(metric, metric.capitalize()))
            y_max = ax.get_ylim()[1]
            if y_max >= 1000:
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            else:
                ax.ticklabel_format(axis="y", style="plain", useOffset=False) # Use plain style for non-scientific, no offset
            ax.yaxis.get_major_formatter().set_useOffset(False) # Ensure no offset is used for formatting
            ax.set_xlim(0, args.total_steps)

            for algo in args.algos:
                algo_color = BASELINES_COLORS.get(algo, None)

                for method in methods:
                    runs = (
                        env_pack
                        .get(algo, {})
                        .get(method, {})
                        .get(metric, [])
                    )
                    if not runs:
                        continue

                    steps, vals = align_and_stack(runs)
                    if vals.size == 0 or len(steps) == 0:
                        continue

                    x = steps.astype(float)
                    mean = vals.mean(axis=0)
                    ci = 1.96 * vals.std(axis=0) / np.sqrt(max(vals.shape[0], 1))

                    ls = METHOD_LINESTYLES[method]
                    line, = ax.plot(
                        x, mean,
                        color=algo_color,
                        linestyle=ls,
                        linewidth=1.8,
                    )
                    ax.fill_between(x, mean - ci, mean + ci, color=algo_color, alpha=0.15)

                    # Capture handles for legends
                    if algo not in algo_handles:
                        algo_handles[algo] = plt.Line2D([0], [0], color=algo_color, lw=2.5)
                    if method not in method_handles:
                        method_handles[method] = plt.Line2D([0], [0], color="black", lw=2.5, linestyle=ls)

            # Threshold
            if metric == "cost" and not args.no_threshold:
                thr = SAFETY_THRESHOLDS.get(env, None)
                if thr is not None:
                    ax.axhline(thr, linestyle="--", color="red", linewidth=1.6)

            if args.grid:
                ax.grid(True, linestyle="--", linewidth=0.9, alpha=0.45)

        env_title = env.replace("_", " ").title()
        fig.suptitle(f"{env_title} - Training Curves", fontsize=14, y=0.98)

        # Two legends: algos (colors) and methods (linestyles)
        if algo_handles:
            algo_labels = [TRANSLATIONS.get(a, a.upper()) for a in algo_handles.keys()]
            fig.legend(
                list(algo_handles.values()),
                algo_labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.06),
                ncol=min(len(algo_labels), 6),
                fancybox=True,
                shadow=True,
                title="Algo (color)",
            )
        if method_handles:
            method_labels = [TRANSLATIONS.get(m, m.capitalize()) for m in method_handles.keys()]
            fig.legend(
                list(method_handles.values()),
                method_labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=len(method_labels),
                fancybox=True,
                shadow=True,
                title="Training type (line)",
            )

        out_path = out_dir / f"{args.out_name}_curves_{env}.pdf"
        plt.savefig(out_path, bbox_inches="tight")
        plt.show()
        print(f"Saved training curves: {out_path}")


def compute_final_values(
        data: Dict[str, List[pd.DataFrame]],
        metrics: List[str],
        last_k: int = 10,
) -> Dict[str, Tuple[float, float, int]]:
    """Compute final values (mean, CI, n) for each metric."""
    results = {}
    for metric in metrics:
        runs = data.get(metric, [])
        if not runs:
            results[metric] = (np.nan, 0.0, 0)
            continue

        final_vals = []
        for df in runs:
            if len(df) == 0:
                continue
            vals = df["value"].dropna().astype(np.float32)
            if len(vals) == 0:
                continue
            k = max(1, min(last_k, len(vals)))
            final_vals.append(float(vals.iloc[-k:].mean()))

        if not final_vals:
            results[metric] = (np.nan, 0.0, 0)
        else:
            arr = np.array(final_vals)
            mean = arr.mean()
            ci = 1.96 * arr.std() / np.sqrt(len(arr))
            results[metric] = (mean, ci, len(arr))

    return results


def plot_final_comparison(
        data_by_env: Dict[str, Dict[str, Dict[str, Dict[str, List[pd.DataFrame]]]]],
        args: argparse.Namespace,
) -> None:
    """
    Plot final results as grouped bars, similar to alg_comp_bars.py.

    Layout: rows of envs, each env has len(metrics) columns (reward, cost side-by-side).
    X-axis: training methods (normal, curriculum, transfer) as groups.
    Bars within each group: algorithms (colored).
    Single legend at bottom for algorithms.
    Env title centered above each env's subplots (not repeated per metric).
    """
    set_mpl_style()

    metrics = args.metrics
    envs = args.envs
    algos = args.algos
    methods = ["normal", "curriculum", "transfer"]

    n_env = len(envs)
    nrows_env, ncols_env = nice_grid(n_env, max_cols=args.max_cols)

    # Each env occupies len(metrics) columns (reward+cost side-by-side)
    m = len(metrics)
    total_rows = nrows_env
    total_cols = ncols_env * m

    fig_w = args.panel_w * total_cols
    fig_h = args.panel_h * total_rows
    fig, axs = plt.subplots(total_rows, total_cols, figsize=(fig_w, fig_h), squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.14, wspace=0.35, hspace=0.55)

    legend_handles: Dict[str, Any] = {}

    # x positions: methods as groups, algos as bars within each group
    M = len(methods)
    A = len(algos)
    x = np.arange(M)

    group_width = 0.82
    bar_w = group_width / max(A, 1)
    offsets = (np.arange(A) - (A - 1) / 2.0) * bar_w

    def get_ax(env_i: int, metric_i: int):
        """Return axis for a given env and metric index (metric side-by-side)."""
        gr = env_i // ncols_env
        gc = env_i % ncols_env
        r = gr
        c = gc * m + metric_i
        return axs[r, c]

    # Plot per env
    for env_i, env in enumerate(envs):
        env_pack = data_by_env.get(env, {})
        env_title = TRANSLATIONS.get(env, env.replace("_", " ").title())

        # Plot each metric into its side-by-side axis
        for metric_i, metric in enumerate(metrics):
            ax = get_ax(env_i, metric_i)
            ylab = TRANSLATIONS.get(metric, metric.capitalize())

            # Compute finals: finals[method][algo] -> (mean, ci, n)
            finals: Dict[str, Dict[str, Tuple[float, float, int]]] = {}
            for method in methods:
                finals[method] = {}
                for algo in algos:
                    runs = env_pack.get(algo, {}).get(method, {}).get(metric, [])
                    tmp = {metric: runs}
                    res = compute_final_values(tmp, [metric], args.last_k)[metric]
                    finals[method][algo] = res

            # Plot bars: algos within each method group
            for a_i, algo in enumerate(algos):
                ys = []
                es = []
                for method in methods:
                    mean, ci, _n = finals[method][algo]
                    ys.append(mean)
                    es.append(ci)

                bars = ax.bar(
                    x + offsets[a_i], ys,
                    width=bar_w,
                    yerr=es,
                    capsize=2,
                    color=BASELINES_COLORS.get(algo, None),
                    label=algo,
                )
                if algo not in legend_handles:
                    legend_handles[algo] = bars[0]

            ax.set_xticks(x)
            ax.set_xticklabels([TRANSLATIONS.get(m, m.capitalize()) for m in methods], rotation=30, ha="right")
            ax.set_ylabel(ylab)
            ax.set_ylim(bottom=0)
            
            y_max = ax.get_ylim()[1]
            if y_max >= 1000:
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            else:
                ax.ticklabel_format(axis="y", style="plain", useOffset=False) # Use plain style for non-scientific, no offset
            ax.yaxis.get_major_formatter().set_useOffset(False) # Ensure no offset is used for formatting

            # Threshold only on cost axis
            if metric == "cost" and not args.no_threshold:
                thr = SAFETY_THRESHOLDS.get(env, None)
                if thr is not None:
                    thr_line = ax.axhline(thr, linestyle="--", color="red")
                    if "Threshold" not in legend_handles:
                        legend_handles["Threshold"] = thr_line

            if args.grid:
                ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

        # Center env title above the reward+cost pair (spans m subplots)
        left_bbox = get_ax(env_i, 0).get_position()
        right_bbox = get_ax(env_i, m - 1).get_position()
        mid_x = 0.5 * (left_bbox.x0 + right_bbox.x1)
        top_y = max(left_bbox.y1, right_bbox.y1) + 0.01
        fig.text(mid_x, top_y, env_title, ha="center", va="bottom", fontsize=13)

    # Hide unused env slots (all metric columns for that env cell)
    for env_i in range(n_env, nrows_env * ncols_env):
        for metric_i in range(m):
            ax = get_ax(env_i, metric_i)
            ax.axis("off")

    # Global legend at bottom
    if legend_handles:
        labels = list(legend_handles.keys())
        handles = [legend_handles[k] for k in labels]
        labels = [TRANSLATIONS.get(lbl, lbl) for lbl in labels]
        fig.legend(
            handles, labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=min(len(labels), 10),
            fancybox=True,
            shadow=True,
        )

    out_dir = Path(args.output_fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.out_name}.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    print(f"Saved final comparison: {out_path}")


def main(args: argparse.Namespace) -> None:
    base = Path(__file__).parent.parent.resolve() / args.input
    unsafe_steps = int(args.total_steps * args.transfer_unsafe_fraction)

    print(f"Total budget: {args.total_steps:,} steps")
    print(f"Transfer unsafe phase: {unsafe_steps:,} steps")
    print(f"Envs: {', '.join(args.envs)}")
    print(f"Algos: {', '.join(args.algos)}")

    # data_by_env[env][algo][method][metric] -> list[df]
    data_by_env: Dict[str, Dict[str, Dict[str, Dict[str, List[pd.DataFrame]]]]] = {}

    for env in args.envs:
        data_by_env[env] = {}
        for algo in args.algos:
            print(f"\nLoading {env} / {algo}")

            normal_data = load_normal_runs(
                base, env, args.target_level, algo, args.seeds, args.metrics
            )
            curriculum_data = load_curriculum_runs(
                base, env, algo, args.seeds, args.metrics, args.total_steps
            )
            transfer_data = load_transfer_runs(
                base, env, args.target_level, algo, args.seeds, args.metrics,
                unsafe_steps, args.total_steps
            )

            data_by_env[env][algo] = {
                "normal": normal_data,
                "curriculum": curriculum_data,
                "transfer": transfer_data,
            }

            n_norm = sum(len(v) for v in normal_data.values())
            n_curr = sum(len(v) for v in curriculum_data.values())
            n_trans = sum(len(v) for v in transfer_data.values())
            print(f"  series: normal={n_norm}, curriculum={n_curr}, transfer={n_trans}")

    if args.plot_curves:
        plot_training_curves(data_by_env, args)

    if args.plot_final:
        plot_final_comparison(data_by_env, args)


def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare normal, curriculum, and transfer training")

    # Data paths
    p.add_argument("--input", type=str, default="data",
                   help="Base data directory")
    p.add_argument(
        "--envs", type=str, nargs="+",
        default=["safe_goal_point", "safe_reacher", "safe_walker", "safe_height"],
        help="Environments to compare"
    )
    p.add_argument(
        "--algos", type=str, nargs="+",
        default=["ppo_lag", "ppo_pid", "p3o", "focops"],
        help="Algorithms to compare (rows in the plot grid)"
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--target_level", type=int, default=3,
                   help="Target difficulty level (for normal and transfer)")
    p.add_argument("--metrics", type=str, nargs="+", default=["reward", "cost"],
                   choices=list(METRIC_COLS.keys()))

    # Training budget
    p.add_argument("--total_steps", type=int, default=int(3e8),
                   help="Total training budget (default: 300M)")
    p.add_argument("--transfer_unsafe_fraction", type=float, default=0.5,
                   help="Fraction of budget for transfer unsafe phase")

    # Plot settings
    p.add_argument("--plot_curves", action="store_true", default=True,
                   help="Generate training curve plots")
    p.add_argument("--plot_final", action="store_true", default=True,
                   help="Generate final comparison bar plots")
    p.add_argument("--no_threshold", action="store_true",
                   help="Hide safety threshold lines")
    p.add_argument("--grid", action="store_true")
    p.add_argument("--last_k", type=int, default=10,
                   help="Number of final values to average for final comparison")

    # Figure sizing
    p.add_argument("--panel_w", type=float, default=3.1)
    p.add_argument("--panel_h", type=float, default=2.5)
    p.add_argument("--max_cols", type=int, default=2, help="Max env columns in grid.")

    # Output
    p.add_argument("--output_fig_dir", type=str, default="figures")
    p.add_argument("--out_name", type=str, default="curriculum_transfer")

    return p


if __name__ == "__main__":
    args = build_args().parse_args()
    main(args)
