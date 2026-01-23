import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from results.common import (
    DEFAULT_METRIC_COLS as METRIC_COLS,
    get_series,
    set_mpl_style,
    nice_grid as _nice_grid,
    BASELINES_COLORS,
)

# Pretty labels
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
    "safe_point_goal": "Safe Point Goal",
    "safe_reacher": "Safe Reacher",
    "safe_walker": "Safe Walker",
}

# Optional safety thresholds per env for cost bars (None disables)
SAFETY_THRESHOLDS: Dict[str, float] = {
    "safe_point_goal": 25.0,
    "safe_reacher": 25.0,
    "safe_walker": 25.0,
}


def load_final_values(
        base: Path,
        envs: List[str],
        levels: List[int],
        algos: List[str],
        seeds: List[int],
        metrics: List[str],
        final_mode: str = "mean_last_k",  # "last" or "mean_last_k"
        last_k: int = 10,
) -> pd.DataFrame:
    """
    Returns long DataFrame with columns:
      env, level, algo, metric, seed, value
    """
    rows = []
    for env in envs:
        for level in levels:
            for algo in algos:
                for seed in seeds:
                    fp = base / env / f"level_{level}" / algo / f"seed_{seed}.parquet"
                    if not fp.exists():
                        print(f"File not found: {fp}")
                        continue

                    df = pd.read_parquet(fp, engine="pyarrow")

                    if "_step" not in df.columns:
                        continue

                    df = df.sort_values("_step", kind="mergesort")

                    for metric in metrics:
                        series = get_series(df, algo=algo, metric=metric, metric_cols=METRIC_COLS, env_name=env)
                        if series is None:
                            continue
                        series = series.dropna().astype(np.float32)
                        if len(series) == 0:
                            continue

                        if final_mode == "last":
                            v = float(series.iloc[-1])
                        elif final_mode == "mean_last_k":
                            k = max(1, min(last_k, len(series)))
                            v = float(series.iloc[-k:].mean())
                        else:
                            raise ValueError(f"Unknown final_mode: {final_mode}")

                        rows.append(
                            dict(env=env, level=level, algo=algo, metric=metric, seed=seed, value=v)
                        )

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate seeds -> mean + 95% CI.
    Output columns: env, level, algo, metric, mean, ci, n
    """
    if df.empty:
        return df

    g = df.groupby(["env", "level", "algo", "metric"], as_index=False)
    out = g["value"].agg(["mean", "std", "count"]).reset_index()
    out.rename(columns={"count": "n"}, inplace=True)
    out["ci"] = 1.96 * out["std"].fillna(0.0) / np.sqrt(out["n"].clip(lower=1))
    out.drop(columns=["std"], inplace=True)
    return out


def plot_final_bars(stats: pd.DataFrame, args: argparse.Namespace) -> None:
    set_mpl_style()

    envs = args.envs
    metrics = args.metrics  # expects ["reward","cost"] by default
    levels = args.levels
    algos = args.algos

    n_env = len(envs)
    nrows_env, ncols_env = _nice_grid(n_env, max_cols=args.max_cols)

    # Each env occupies len(metrics) columns (reward+cost side-by-side)
    m = len(metrics)
    total_rows = nrows_env
    total_cols = ncols_env * m

    # panel_w is now "per metric subplot"; panel_h is "per env row"
    fig_w = args.panel_w * total_cols
    fig_h = args.panel_h * total_rows
    fig, axs = plt.subplots(total_rows, total_cols, figsize=(fig_w, fig_h), squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.14, wspace=0.35, hspace=0.55)

    legend_handles: Dict[str, plt.Line2D] = {}

    # x positions: levels as groups, algos as bars within each group
    L = len(levels)
    A = len(algos)
    x = np.arange(L)

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

    # plot per env
    for env_i, env in enumerate(envs):
        env_title = TRANSLATIONS.get(env, env)

        # plot each metric into its side-by-side axis
        for metric_i, metric in enumerate(metrics):
            ax = get_ax(env_i, metric_i)
            ylab = TRANSLATIONS.get(metric, metric.capitalize())

            for a_i, algo in enumerate(algos):
                ys = []
                es = []
                for level in levels:
                    sub = stats[
                        (stats["env"] == env)
                        & (stats["metric"] == metric)
                        & (stats["algo"] == algo)
                        & (stats["level"] == level)
                        ]
                    if len(sub) == 0:
                        ys.append(np.nan)
                        es.append(0.0)
                    else:
                        ys.append(float(sub["mean"].iloc[0]))
                        es.append(float(sub["ci"].iloc[0]))

                bars = ax.bar(x + offsets[a_i], ys, width=bar_w, yerr=es, capsize=2, label=algo, color=BASELINES_COLORS.get(algo))

                if algo not in legend_handles:
                    legend_handles[algo] = bars[0]
                ax.set_xticks(x)
                ax.set_xticklabels([str(lv) for lv in levels])
                ax.set_xlabel("Level")
                ax.set_ylabel(ylab)

                y_max = ax.get_ylim()[1]
                if y_max >= 1000:
                    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
                else:
                    ax.ticklabel_format(axis="y", style="plain", useOffset=False) # Use plain style for non-scientific, no offset
                ax.yaxis.get_major_formatter().set_useOffset(False) # Ensure no offset is used for formatting
            # threshold only on cost axis
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

    # hide unused env slots (all metric columns for that env cell)
    for env_i in range(n_env, nrows_env * ncols_env):
        for metric_i in range(m):
            ax = get_ax(env_i, metric_i)
            ax.axis("off")

    # global legend at bottom
    if legend_handles:
        labels = list(legend_handles.keys())
        handles = [legend_handles[k] for k in labels]
        labels = [TRANSLATIONS.get(lbl, lbl) for lbl in labels]
        fig.legend(
            handles, labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=min(len(labels), 10),
            fancybox=True,
            shadow=True,
        )

    out_dir = Path(args.output_fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.out_name}_final.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    print(f"Saved figure: {out_path}")


def generate_latex_table(
        stats: pd.DataFrame,
        envs: List[str],
        levels: List[int],
        algos: List[str],
        safety_threshold: float = 25.0,
) -> str:
    """
    Generate a LaTeX table with results.

    - Rows: algorithms
    - Main columns: environments with levels as sub-columns
    - Each level has reward (↑) and cost (↓)
    - Safe costs (< threshold) are colored green
    - Best result per (env, level) is bolded: highest reward among safe algorithms
    """
    # Build lookup: (env, level, algo, metric) -> (mean, ci)
    lookup = {}
    for _, row in stats.iterrows():
        key = (row["env"], row["level"], row["algo"], row["metric"])
        lookup[key] = (row["mean"], row["ci"])

    # Determine best algo per (env, level): highest reward among those with cost < threshold
    best_algo = {}
    for env in envs:
        for level in levels:
            best_reward = -float("inf")
            best_a = None
            for algo in algos:
                cost_key = (env, level, algo, "cost")
                reward_key = (env, level, algo, "reward")
                if cost_key not in lookup or reward_key not in lookup:
                    continue
                cost_mean, _ = lookup[cost_key]
                reward_mean, _ = lookup[reward_key]
                if cost_mean < safety_threshold and reward_mean > best_reward:
                    best_reward = reward_mean
                    best_a = algo
            best_algo[(env, level)] = best_a

    # Build LaTeX
    lines = []

    # Preamble comment
    lines.append("% Auto-generated LaTeX table")
    lines.append("% Requires: \\usepackage{booktabs}, \\usepackage{xcolor}, \\usepackage{multirow}")
    lines.append("% Define: \\definecolor{safegreen}{RGB}{34, 139, 34}")
    lines.append("")

    # Calculate column structure
    # For each env: levels * 2 columns (reward + cost per level)
    n_data_cols = len(envs) * len(levels) * 2

    # Column spec: l for algo name, then c for each data column (no vertical lines)
    col_spec = "l" + "c" * n_data_cols

    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header row 1: Environment names spanning their columns
    header1_parts = [""]
    for env in envs:
        env_label = TRANSLATIONS.get(env, env)
        n_cols = len(levels) * 2
        header1_parts.append(f"\\multicolumn{{{n_cols}}}{{c}}{{{env_label}}}")
    lines.append(" & ".join(header1_parts) + " \\\\")

    # Add cmidrules under each environment name
    cmidrule_parts = []
    col_idx = 2  # Start at column 2 (after algo name)
    for env in envs:
        n_cols = len(levels) * 2
        cmidrule_parts.append(f"\\cmidrule(lr){{{col_idx}-{col_idx + n_cols - 1}}}")
        col_idx += n_cols
    lines.append(" ".join(cmidrule_parts))

    # Header row 2: Level numbers spanning reward+cost pairs
    header2_parts = [""]
    for env in envs:
        for level in levels:
            header2_parts.append(f"\\multicolumn{{2}}{{c}}{{Level {level}}}")
    lines.append(" & ".join(header2_parts) + " \\\\")

    # Add cmidrules under each level
    cmidrule_parts = []
    col_idx = 2
    for env in envs:
        for level in levels:
            cmidrule_parts.append(f"\\cmidrule(lr){{{col_idx}-{col_idx + 1}}}")
            col_idx += 2
    lines.append(" ".join(cmidrule_parts))

    # Header row 3: Reward↑ / Cost↓ labels
    header3_parts = ["Algorithm"]
    for env in envs:
        for level in levels:
            header3_parts.append("R $\\uparrow$")
            header3_parts.append("C $\\downarrow$")
    lines.append(" & ".join(header3_parts) + " \\\\")
    lines.append("\\midrule")

    # Data rows: one per algorithm
    for algo in algos:
        algo_label = TRANSLATIONS.get(algo, algo)
        row_parts = [algo_label]

        for env in envs:
            for level in levels:
                # Reward
                reward_key = (env, level, algo, "reward")
                if reward_key in lookup:
                    r_mean, r_ci = lookup[reward_key]
                    r_str = f"{r_mean:.1f}"
                    # Bold if this is the best algo for this (env, level)
                    if best_algo.get((env, level)) == algo:
                        r_str = f"\\textbf{{{r_str}}}"
                else:
                    r_str = "--"
                row_parts.append(r_str)

                # Cost
                cost_key = (env, level, algo, "cost")
                if cost_key in lookup:
                    c_mean, c_ci = lookup[cost_key]
                    c_str = f"{c_mean:.1f}"
                    # Color green if safe
                    if c_mean < safety_threshold:
                        c_str = f"\\textcolor{{safegreen}}{{{c_str}}}"
                    # Bold if this is the best algo for this (env, level)
                    if best_algo.get((env, level)) == algo:
                        c_str = f"\\textbf{{{c_str}}}"
                else:
                    c_str = "--"
                row_parts.append(c_str)

        lines.append(" & ".join(row_parts) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Algorithm comparison across environments and difficulty levels. "
                 "R: Reward ($\\uparrow$ higher is better), C: Cost ($\\downarrow$ lower is better). "
                 f"\\textcolor{{safegreen}}{{Green}} indicates safe (cost $< {safety_threshold:.0f}$). "
                 "\\textbf{Bold} indicates best safe result.}")
    lines.append("\\label{tab:alg_comparison}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def print_latex_table(stats: pd.DataFrame, args: argparse.Namespace) -> None:
    """Generate and print the LaTeX table."""
    latex = generate_latex_table(
        stats=stats,
        envs=args.envs,
        levels=args.levels,
        algos=args.algos,
        safety_threshold=args.safety_threshold,
    )
    print("\n" + "=" * 80)
    print("LATEX TABLE (copy below into your .tex file):")
    print("=" * 80 + "\n")
    print(latex)
    print("\n" + "=" * 80)

    # Optionally save to file
    if args.output_latex:
        out_path = Path(args.output_latex)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(latex)
        print(f"LaTeX table saved to: {out_path}")


def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot final results as grouped bars across levels.")
    p.add_argument("--input", type=str, default="data",
                   help="Base directory with <env>/level_<k>/<algo>/seed_*.parquet")
    p.add_argument("--envs", type=str, nargs="+", default=["safe_point_goal", "safe_reacher", "safe_walker"])
    p.add_argument("--algos", type=str, nargs="+", default=["ppo", "ppo_cost", "ppo_lag", "ppo_pid", "ppo_saute", "p3o", "focops"])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--metrics", type=str, nargs="+", default=["reward", "cost"], choices=list(METRIC_COLS.keys()))

    p.add_argument("--final_mode", type=str, default="mean_last_k", choices=["last", "mean_last_k"],
                   help="How to define 'final' value per seed.")
    p.add_argument("--last_k", type=int, default=10, help="Used when final_mode=mean_last_k.")

    p.add_argument("--no_threshold", action="store_true", help="Hide safety threshold lines (cost only).")
    p.add_argument("--grid", action="store_true")

    p.add_argument("--max_cols", type=int, default=2, help="Max env columns in grid.")
    p.add_argument("--panel_w", type=float, default=3.1, help="Width per env column.")
    p.add_argument("--panel_h", type=float, default=2.3, help="Height per metric row per env row.")

    p.add_argument("--output_fig_dir", type=str, default="figures")
    p.add_argument("--out_name", type=str, default="baselines")

    # LaTeX table options
    p.add_argument("--latex", action="store_true", help="Generate and print LaTeX table.")
    p.add_argument("--output_latex", type=str, default=None,
                   help="Optional path to save LaTeX table (e.g., tables/results.tex).")
    p.add_argument("--safety_threshold", type=float, default=25.0,
                   help="Cost threshold for marking results as safe (green) in LaTeX table.")
    p.add_argument("--no_plot", action="store_true", help="Skip plotting, only generate LaTeX table.")
    return p


def main(args: argparse.Namespace) -> None:
    base = Path(__file__).parent.parent.resolve() / args.input
    df = load_final_values(
        base=base,
        envs=args.envs,
        levels=args.levels,
        algos=args.algos,
        seeds=args.seeds,
        metrics=args.metrics,
        final_mode=args.final_mode,
        last_k=args.last_k,
    )
    if df.empty:
        raise SystemExit("No data loaded. Check paths/envs/levels/algos/seeds.")
    stats = summarize(df)

    # Generate LaTeX table if requested
    if args.latex or args.output_latex:
        print_latex_table(stats, args)

    # Plot unless --no_plot is set
    if not args.no_plot:
        plot_final_bars(stats, args)


if __name__ == "__main__":
    args = build_args().parse_args()
    main(args)
