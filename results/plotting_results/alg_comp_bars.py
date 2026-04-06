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
    BASELINES_COLORS, TRANSLATIONS,
)


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


def _draw_break_pattern(ax, bar, y_clip, bar_width):
    """Draw a zigzag break pattern at the top of a truncated bar."""
    x_center = bar.get_x() + bar.get_width() / 2
    x_left = x_center - bar_width / 2
    x_right = x_center + bar_width / 2

    # Draw white rectangle to "cut" the bar
    cut_height = y_clip * 0.06
    rect = plt.Rectangle(
        (x_left, y_clip - cut_height / 2),
        bar_width,
        cut_height,
        facecolor='white',
        edgecolor='none',
        zorder=10
    )
    ax.add_patch(rect)

    # Draw zigzag lines
    n_zigs = 3
    x_points = np.linspace(x_left, x_right, n_zigs * 2 + 1)
    y_bottom = y_clip - cut_height / 2
    y_top = y_clip + cut_height / 2
    y_points = [y_bottom if i % 2 == 0 else y_top for i in range(len(x_points))]

    ax.plot(x_points, y_points, color='black', linewidth=0.8, zorder=11)


def plot_final_bars(stats: pd.DataFrame, args: argparse.Namespace) -> None:
    set_mpl_style()

    envs = args.envs
    metrics = args.metrics  # expects ["reward","cost"] by default
    levels = args.levels
    algos = args.algos
    threshold = 25.0

    n_env = len(envs)
    nrows_env, ncols_env = _nice_grid(n_env, max_cols=args.max_cols)

    # Each env occupies len(metrics) columns (reward+cost side-by-side)
    m = len(metrics)
    total_rows = nrows_env
    total_cols = ncols_env * m

    # panel_w is now "per metric subplot"; panel_h is "per env row"
    fig_w = args.panel_w * total_cols
    fig_h = args.panel_h * total_rows
    single_level_mode = len(levels) == 1

    fig, axs = plt.subplots(total_rows, total_cols, figsize=(fig_w, fig_h), squeeze=False)
    hspace = 0.45 if single_level_mode else 0.55
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.12, wspace=0.35, hspace=hspace)

    legend_handles: Dict[str, plt.Line2D] = {}

    if single_level_mode:
        # x positions: algos as individual bars
        A = len(algos)
        x = np.arange(A)
        group_width = 0.82
        bar_w = group_width
        offsets = np.zeros(A)  # unused in single-level mode
    else:
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

    # First pass: collect all values to determine y-limits with truncation
    env_metric_values: Dict[tuple, List[float]] = {}
    for env_i, env in enumerate(envs):
        for metric_i, metric in enumerate(metrics):
            values = []
            for algo in algos:
                for level in levels:
                    sub = stats[
                        (stats["env"] == env)
                        & (stats["metric"] == metric)
                        & (stats["algo"] == algo)
                        & (stats["level"] == level)
                    ]
                    if len(sub) > 0:
                        values.append(float(sub["mean"].iloc[0]))
            env_metric_values[(env, metric)] = values

    # plot per env
    for env_i, env in enumerate(envs):
        env_title = TRANSLATIONS.get(env, env)

        # plot each metric into its side-by-side axis
        for metric_i, metric in enumerate(metrics):
            ax = get_ax(env_i, metric_i)
            ylab = TRANSLATIONS.get(metric, metric.capitalize())

            # Determine y-clip for this subplot
            values = env_metric_values.get((env, metric), [])

            # Compute smart y-clip based on data distribution
            y_clip = None
            truncated_bars = []  # Track which bars get truncated

            if args.truncate_outliers and metric == "cost" and threshold is not None and len(values) > 0:
                # Sort values to find the "gap" between safe and unsafe
                sorted_vals = sorted([v for v in values if not np.isnan(v)])

                if len(sorted_vals) > 1:
                    # Find values that are much larger than the threshold
                    safe_vals = [v for v in sorted_vals if v <= threshold * args.truncate_factor]
                    outlier_vals = [v for v in sorted_vals if v > threshold * args.truncate_factor]

                    if outlier_vals and safe_vals:
                        # Set y_clip to show detail in the safe region
                        y_clip = threshold * args.truncate_factor
                    elif not safe_vals and outlier_vals:
                        # All values are outliers - use a reasonable max
                        y_clip = threshold * args.truncate_factor

            if single_level_mode:
                # One bar per algo on x-axis
                level = levels[0]
                for a_i, algo in enumerate(algos):
                    sub = stats[
                        (stats["env"] == env)
                        & (stats["metric"] == metric)
                        & (stats["algo"] == algo)
                        & (stats["level"] == level)
                    ]
                    if len(sub) == 0:
                        y, e = np.nan, 0.0
                    else:
                        y = float(sub["mean"].iloc[0])
                        e = float(sub["ci"].iloc[0])

                    trunc = y_clip is not None and not np.isnan(y) and y > y_clip
                    y_display = y_clip if trunc else y
                    e_display = 0.0 if trunc else e

                    bars = ax.bar(
                        [a_i], [y_display], width=bar_w,
                        yerr=[e_display], capsize=2, label=algo,
                        color=BASELINES_COLORS.get(algo)
                    )

                    if algo not in legend_handles:
                        legend_handles[algo] = bars[0]

                    if trunc and not np.isnan(y):
                        _draw_break_pattern(ax, bars[0], y_clip, bar_w)
                        label_text = f"{y / 1000:.1f}k" if y >= 1000 else f"{y:.0f}"
                        ax.annotate(
                            label_text,
                            xy=(a_i, y_clip),
                            xytext=(0, 4),
                            textcoords='offset points',
                            ha='center', va='bottom',
                            fontsize=8, fontweight='bold',
                            color=BASELINES_COLORS.get(algo, 'black'),
                            rotation=90 if len(algos) > 5 else 0,
                        )

                ax.set_xticks([])
                ax.set_xlabel("")
            else:
                # Collect data for all algorithms
                algo_data = []
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
                    algo_data.append((algo, ys, es))

                # Plot bars
                for a_i, (algo, ys, es) in enumerate(algo_data):
                    # Clip values and errors for display
                    ys_display = []
                    es_display = []
                    is_truncated = []

                    for y, e in zip(ys, es):
                        if y_clip is not None and not np.isnan(y) and y > y_clip:
                            ys_display.append(y_clip)
                            es_display.append(0)  # Don't show error bar for truncated
                            is_truncated.append(True)
                        else:
                            ys_display.append(y)
                            es_display.append(e)
                            is_truncated.append(False)

                    bars = ax.bar(
                        x + offsets[a_i], ys_display, width=bar_w,
                        yerr=es_display, capsize=2, label=algo,
                        color=BASELINES_COLORS.get(algo)
                    )

                    if algo not in legend_handles:
                        legend_handles[algo] = bars[0]

                    # Handle truncated bars
                    for bar_idx, (bar, trunc, orig_y) in enumerate(zip(bars, is_truncated, ys)):
                        if trunc and not np.isnan(orig_y):
                            # Draw break pattern
                            _draw_break_pattern(ax, bar, y_clip, bar_w)

                            # Add value label above the bar
                            x_pos = bar.get_x() + bar.get_width() / 2
                            if orig_y >= 1000:
                                label_text = f"{orig_y / 1000:.1f}k"
                            else:
                                label_text = f"{orig_y:.0f}"

                            ax.annotate(
                                label_text,
                                xy=(x_pos, y_clip),
                                xytext=(0, 4),
                                textcoords='offset points',
                                ha='center', va='bottom',
                                fontsize=8,
                                fontweight='bold',
                                color=BASELINES_COLORS.get(algo, 'black'),
                                rotation=90 if len(algos) > 5 else 0,
                            )

                ax.set_xticks(x)
                ax.set_xticklabels([str(lv) for lv in levels])
                ax.set_xlabel("Level")
            ax.set_ylabel(ylab)

            ax.set_ylim(bottom=0)

            # Set y-limit if truncating
            if y_clip is not None:
                ax.set_ylim(top=y_clip * 1.15)  # Add space for labels

            y_max = ax.get_ylim()[1]
            if y_max >= 1000:
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            else:
                ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.yaxis.get_major_formatter().set_useOffset(False)

            # threshold only on cost axis
            if metric == "cost" and not args.no_threshold:
                if threshold is not None:
                    thr_line = ax.axhline(threshold, linestyle="--", color="red", linewidth=1.5, zorder=5)
                    if "Threshold" not in legend_handles:
                        legend_handles["Threshold"] = thr_line

            if args.grid:
                ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

        # Center env title above the reward+cost pair (spans m subplots)
        left_bbox = get_ax(env_i, 0).get_position()
        right_bbox = get_ax(env_i, m - 1).get_position()
        mid_x = 0.5 * (left_bbox.x0 + right_bbox.x1)
        top_y = max(left_bbox.y1, right_bbox.y1) + 0.01
        fig.text(mid_x, top_y, env_title, ha="center", va="bottom", fontsize=16)

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
            bbox_to_anchor=(0.5, 0.04 if single_level_mode else 0.0),
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


def _build_lookup_and_best_algo(
        stats: pd.DataFrame,
        envs: List[str],
        levels: List[int],
        algos: List[str],
        safety_threshold: float,
) -> tuple:
    """Build lookup dict and best_algo dict for table generation."""
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

    return lookup, best_algo


def generate_latex_table_appendix(
        stats: pd.DataFrame,
        envs: List[str],
        levels: List[int],
        algos: List[str],
        safety_threshold: float = 25.0,
        envs_per_row: int = 2,
) -> str:
    """
    Generate a LaTeX table split across multiple rows for the appendix.

    - Each major row contains at most `envs_per_row` environments
    - Column headers are repeated for each major row
    - All rows use the same fixed column widths for vertical alignment
    - Rows: algorithms
    - Main columns: environments with levels as sub-columns
    - Each level has reward (↑) and cost (↓)
    - Safe costs (< threshold) are colored green
    - Best result per (env, level) is bolded: highest reward among safe algorithms
    """
    lookup, best_algo = _build_lookup_and_best_algo(stats, envs, levels, algos, safety_threshold)

    # Split envs into chunks
    env_chunks = [envs[i:i + envs_per_row] for i in range(0, len(envs), envs_per_row)]

    # Fixed column widths for consistent alignment
    algo_col_width = "1.8cm"
    data_col_width = "0.85cm"

    lines = []

    # Preamble comment
    lines.append("% Auto-generated LaTeX table (Appendix - split across rows)")
    lines.append("% Requires: \\usepackage{booktabs}, \\usepackage{xcolor}, \\usepackage{array}")
    lines.append("% Define: \\definecolor{safegreen}{RGB}{34, 139, 34}")
    lines.append("")

    lines.append("\\begin{table*}[htbp]")
    lines.append("\\centering")
    lines.append("\\footnotesize")

    # Full width column spec (always same number of columns for alignment)
    n_full_data_cols = envs_per_row * len(levels) * 2
    col_spec = f"p{{{algo_col_width}}}" + f"p{{{data_col_width}}}" * n_full_data_cols

    for chunk_idx, env_chunk in enumerate(env_chunks):
        # Use left-aligned minipage for partial rows
        if len(env_chunk) < envs_per_row:
            lines.append("\\noindent")

        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")

        # Header row 1: Environment names spanning their columns
        header1_parts = [""]
        for env in env_chunk:
            env_label = TRANSLATIONS.get(env, env)
            n_cols = len(levels) * 2
            header1_parts.append(f"\\multicolumn{{{n_cols}}}{{c}}{{{env_label}}}")
        # Fill remaining columns with empty multicolumn for alignment
        remaining_envs = envs_per_row - len(env_chunk)
        for _ in range(remaining_envs):
            n_cols = len(levels) * 2
            header1_parts.append(f"\\multicolumn{{{n_cols}}}{{c}}{{}}")
        lines.append(" & ".join(header1_parts) + " \\\\")

        # Add cmidrules under each environment name (only for actual envs)
        cmidrule_parts = []
        col_idx = 2
        for env in env_chunk:
            n_cols = len(levels) * 2
            cmidrule_parts.append(f"\\cmidrule(lr){{{col_idx}-{col_idx + n_cols - 1}}}")
            col_idx += n_cols
        lines.append(" ".join(cmidrule_parts))

        # Header row 2: Level numbers spanning reward+cost pairs
        header2_parts = [""]
        for env in env_chunk:
            for level in levels:
                header2_parts.append(f"\\multicolumn{{2}}{{c}}{{Level {level}}}")
        # Fill remaining columns
        for _ in range(remaining_envs):
            for level in levels:
                header2_parts.append(f"\\multicolumn{{2}}{{c}}{{}}")
        lines.append(" & ".join(header2_parts) + " \\\\")

        # Add cmidrules under each level (only for actual envs)
        cmidrule_parts = []
        col_idx = 2
        for env in env_chunk:
            for level in levels:
                cmidrule_parts.append(f"\\cmidrule(lr){{{col_idx}-{col_idx + 1}}}")
                col_idx += 2
        lines.append(" ".join(cmidrule_parts))

        # Header row 3: Reward↑ / Cost↓ labels
        header3_parts = ["Algorithm"]
        for env in env_chunk:
            for level in levels:
                header3_parts.append("R $\\uparrow$")
                header3_parts.append("C $\\downarrow$")
        # Fill remaining columns
        for _ in range(remaining_envs):
            for level in levels:
                header3_parts.append("")
                header3_parts.append("")
        lines.append(" & ".join(header3_parts) + " \\\\")
        lines.append("\\midrule")

        # Data rows: one per algorithm
        for algo in algos:
            algo_label = TRANSLATIONS.get(algo, algo)
            row_parts = [algo_label]

            for env in env_chunk:
                for level in levels:
                    # Reward
                    reward_key = (env, level, algo, "reward")
                    if reward_key in lookup:
                        r_mean, r_ci = lookup[reward_key]
                        r_str = f"{r_mean:.1f}"
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
                        if c_mean < safety_threshold:
                            c_str = f"\\textcolor{{safegreen}}{{{c_str}}}"
                        if best_algo.get((env, level)) == algo:
                            c_str = f"\\textbf{{{c_str}}}"
                    else:
                        c_str = "--"
                    row_parts.append(c_str)

            # Fill remaining columns with empty cells
            for _ in range(remaining_envs):
                for level in levels:
                    row_parts.append("")
                    row_parts.append("")

            lines.append(" & ".join(row_parts) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")

        # Add spacing between chunks (but not after the last one)
        if chunk_idx < len(env_chunks) - 1:
            lines.append("\\\\[1em]")  # Vertical space between sub-tables

    lines.append("\\caption{Detailed algorithm comparison across environments and difficulty levels. "
                 "R: Reward ($\\uparrow$ higher is better), C: Cost ($\\downarrow$ lower is better). "
                 f"\\textcolor{{safegreen}}{{Green}} indicates safe (cost $< {safety_threshold:.0f}$). "
                 "\\textbf{Bold} indicates best safe result.}")
    lines.append("\\label{tab:alg_comparison_detailed}")
    lines.append("\\end{table*}")

    return "\n".join(lines)


def generate_latex_summary_table(
        stats: pd.DataFrame,
        envs: List[str],
        levels: List[int],
        algos: List[str],
        safety_threshold: float = 25.0,
) -> str:
    """
    Generate a compact summary LaTeX table.

    - Rows: algorithms
    - Columns: levels (1, 2, 3) + Total
    - Each cell shows: #Wins / %Safe
      - #Wins: number of times this algo was the best (bold) across all envs for this level
      - %Safe: percentage of envs where cost < threshold for this level
    - Total column: sum of wins, average of safety percentage
    """
    lookup, best_algo = _build_lookup_and_best_algo(stats, envs, levels, algos, safety_threshold)

    # Compute wins and safety percentage per (algo, level)
    summary_data = {}
    for algo in algos:
        for level in levels:
            wins = 0
            safe_count = 0
            total_envs = 0

            for env in envs:
                cost_key = (env, level, algo, "cost")
                if cost_key not in lookup:
                    continue

                total_envs += 1
                cost_mean, _ = lookup[cost_key]

                # Check if safe
                if cost_mean < safety_threshold:
                    safe_count += 1

                # Check if best
                if best_algo.get((env, level)) == algo:
                    wins += 1

            safe_pct = (safe_count / total_envs * 100) if total_envs > 0 else 0
            summary_data[(algo, level)] = (wins, safe_pct, total_envs)

    # Compute totals per algorithm
    algo_totals = {}
    for algo in algos:
        total_wins = 0
        safe_pcts = []
        for level in levels:
            wins, safe_pct, _ = summary_data.get((algo, level), (0, 0, 0))
            total_wins += wins
            safe_pcts.append(safe_pct)
        avg_safe_pct = sum(safe_pcts) / len(safe_pcts) if safe_pcts else 0
        algo_totals[algo] = (total_wins, avg_safe_pct)

    lines = []

    # Preamble
    lines.append("% Auto-generated LaTeX summary table")
    lines.append("% Requires: \\usepackage{booktabs}, \\usepackage{xcolor}")
    lines.append("% Define: \\definecolor{safegreen}{RGB}{34, 139, 34}")
    lines.append("")

    # Column spec: l for algo, then 2 columns per level (Wins, Safe%), plus 2 for Total
    n_cols = (len(levels) + 1) * 2  # +1 for Total column
    col_spec = "l" + "c" * n_cols

    lines.append("\\begin{table*}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header row 1: Level spanning + Total
    header1_parts = [""]
    for level in levels:
        header1_parts.append(f"\\multicolumn{{2}}{{c}}{{Level {level}}}")
    header1_parts.append("\\multicolumn{2}{c}{Total}")
    lines.append(" & ".join(header1_parts) + " \\\\")

    # Cmidrules under levels and Total
    cmidrule_parts = []
    col_idx = 2
    for level in levels:
        cmidrule_parts.append(f"\\cmidrule(lr){{{col_idx}-{col_idx + 1}}}")
        col_idx += 2
    # Total column cmidrule
    cmidrule_parts.append(f"\\cmidrule(lr){{{col_idx}-{col_idx + 1}}}")
    lines.append(" ".join(cmidrule_parts))

    # Header row 2: Wins↑ / Safe%↑
    header2_parts = ["Algorithm"]
    for level in levels:
        header2_parts.append("Wins $\\uparrow$")
        header2_parts.append("Safe\\% $\\uparrow$")
    # Total columns
    header2_parts.append("Wins $\\uparrow$")
    header2_parts.append("Safe\\% $\\uparrow$")
    lines.append(" & ".join(header2_parts) + " \\\\")
    lines.append("\\midrule")

    # Find max wins per level and max total wins for highlighting
    max_wins_per_level = {}
    for level in levels:
        max_wins = 0
        for algo in algos:
            wins, _, _ = summary_data.get((algo, level), (0, 0, 0))
            if wins > max_wins:
                max_wins = wins
        max_wins_per_level[level] = max_wins

    max_total_wins = max(algo_totals[algo][0] for algo in algos) if algos else 0

    # Data rows
    for algo in algos:
        algo_label = TRANSLATIONS.get(algo, algo)
        row_parts = [algo_label]

        for level in levels:
            wins, safe_pct, total_envs = summary_data.get((algo, level), (0, 0, 0))

            # Wins column - bold if max
            wins_str = str(wins)
            if wins > 0 and wins == max_wins_per_level[level]:
                wins_str = f"\\textbf{{{wins_str}}}"
            row_parts.append(wins_str)

            # Safe% column - green if 100%
            safe_str = f"{safe_pct:.0f}"
            if safe_pct == 100:
                safe_str = f"\\textcolor{{safegreen}}{{{safe_str}}}"
            row_parts.append(safe_str)

        # Total columns
        total_wins, avg_safe_pct = algo_totals[algo]

        # Total wins - bold if max
        total_wins_str = str(total_wins)
        if total_wins > 0 and total_wins == max_total_wins:
            total_wins_str = f"\\textbf{{{total_wins_str}}}"
        row_parts.append(total_wins_str)

        # Average safe% - green if 100%
        avg_safe_str = f"{avg_safe_pct:.0f}"
        if avg_safe_pct == 100:
            avg_safe_str = f"\\textcolor{{safegreen}}{{{avg_safe_str}}}"
        row_parts.append(avg_safe_str)

        lines.append(" & ".join(row_parts) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{Algorithm summary across {len(envs)} environments. "
                 f"\\textbf{{Wins}}: number of environments where the algorithm achieved the highest reward "
                 f"while being safe (cost $< {safety_threshold:.0f}$). "
                 f"\\textbf{{Safe\\%}}: percentage of environments where the algorithm was safe. "
                 f"\\textbf{{Total}}: sum of wins and average safety percentage across levels. "
                 f"\\textcolor{{safegreen}}{{Green}} indicates 100\\% safe.}}")
    lines.append("\\label{tab:alg_summary}")
    lines.append("\\end{table*}")

    return "\n".join(lines)


def generate_latex_table(
        stats: pd.DataFrame,
        envs: List[str],
        levels: List[int],
        algos: List[str],
        safety_threshold: float = 25.0,
) -> str:
    """
    Generate a LaTeX table with results (original single-table format).

    - Rows: algorithms
    - Main columns: environments with levels as sub-columns
    - Each level has reward (↑) and cost (↓)
    - Safe costs (< threshold) are colored green
    - Best result per (env, level) is bolded: highest reward among safe algorithms
    """
    lookup, best_algo = _build_lookup_and_best_algo(stats, envs, levels, algos, safety_threshold)

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

    lines.append("\\begin{table*}[htbp]")
    lines.append("\\centering")
    lines.append("\\footnotesize")
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
    lines.append("\\end{table*}")

    return "\n".join(lines)


def print_latex_table(stats: pd.DataFrame, args: argparse.Namespace) -> None:
    """Generate and print LaTeX tables (summary + detailed appendix)."""

    # Generate summary table (compact, for main paper)
    summary_latex = generate_latex_summary_table(
        stats=stats,
        envs=args.envs,
        levels=args.levels,
        algos=args.algos,
        safety_threshold=args.safety_threshold,
    )

    # Generate detailed appendix table (split across rows)
    appendix_latex = generate_latex_table_appendix(
        stats=stats,
        envs=args.envs,
        levels=args.levels,
        algos=args.algos,
        safety_threshold=args.safety_threshold,
        envs_per_row=args.envs_per_row,
    )

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE (for main paper):")
    print("=" * 80 + "\n")
    print(summary_latex)

    # Print detailed appendix table
    print("\n" + "=" * 80)
    print("DETAILED TABLE (for appendix):")
    print("=" * 80 + "\n")
    print(appendix_latex)
    print("\n" + "=" * 80)

    # Optionally save to files
    if args.output_latex:
        out_path = Path(args.output_latex)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Save summary table
        summary_path = out_path.with_stem(out_path.stem + "_summary")
        summary_path.write_text(summary_latex)
        print(f"Summary table saved to: {summary_path}")

        # Save appendix table
        appendix_path = out_path.with_stem(out_path.stem + "_appendix")
        appendix_path.write_text(appendix_latex)
        print(f"Appendix table saved to: {appendix_path}")


def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot final results as grouped bars across levels.")
    p.add_argument("--input", type=str, default="data",
                   help="Base directory with <env>/level_<k>/<algo>/seed_*.parquet")
    p.add_argument("--envs", type=str, nargs="+", default=["safe_reacher", "safe_goal_point", "safe_push_point", "safe_lift_spider", "safe_circle_point", "safe_height_humanoid", "safe_pathway_walker2d", "safe_velocity_humanoid"])
    p.add_argument("--algos", type=str, nargs="+", default=["ppo", "ppo_cost", "ppo_lag", "ppo_pid", "ppo_saute", "p3o", "focops"])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--single_level", type=int, default=None,
                   help="Plot results for a single level only (algos on x-axis). "
                        "Overrides --levels. Default: None (disabled). "
                        "Use e.g. --single_level 1 for level 1.")
    p.add_argument("--metrics", type=str, nargs="+", default=["reward", "cost"], choices=list(METRIC_COLS.keys()))

    p.add_argument("--final_mode", type=str, default="mean_last_k", choices=["last", "mean_last_k"],
                   help="How to define 'final' value per seed.")
    p.add_argument("--last_k", type=int, default=10, help="Used when final_mode=mean_last_k.")

    p.add_argument("--no_threshold", action="store_true", help="Hide safety threshold lines (cost only).")
    p.add_argument("--grid", action="store_true")

    # Truncation options for handling outliers
    p.add_argument("--truncate_outliers", action="store_true", default=True,
                   help="Truncate bars that exceed the threshold by a large factor (default: True).")
    p.add_argument("--no_truncate", dest="truncate_outliers", action="store_false",
                   help="Disable truncation of outlier bars.")
    p.add_argument("--truncate_factor", type=float, default=2.5,
                   help="Truncate cost bars exceeding threshold * factor (default: 2.5).")

    p.add_argument("--max_cols", type=int, default=2, help="Max env columns in grid.")
    p.add_argument("--panel_w", type=float, default=3.1, help="Width per env column.")
    p.add_argument("--panel_h", type=float, default=2.3, help="Height per metric row per env row.")

    p.add_argument("--output_fig_dir", type=str, default="figures")
    p.add_argument("--out_name", type=str, default="baselines")

    # LaTeX table options
    p.add_argument("--latex", action="store_true", help="Generate and print LaTeX tables (summary + appendix).")
    p.add_argument("--output_latex", type=str, default=None,
                   help="Optional base path to save LaTeX tables (e.g., tables/results.tex). "
                        "Creates _summary.tex and _appendix.tex files.")
    p.add_argument("--safety_threshold", type=float, default=25.0,
                   help="Cost threshold for marking results as safe (green) in LaTeX table.")
    p.add_argument("--envs_per_row", type=int, default=2,
                   help="Max environments per major row in appendix table (default: 2).")
    p.add_argument("--no_plot", action="store_true", help="Skip plotting, only generate LaTeX table.")
    return p


def main(args: argparse.Namespace) -> None:
    if args.single_level is not None:
        args.levels = [args.single_level]

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
