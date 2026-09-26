"""Figures and tables for the hyperparameter sweeps."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm

from results import cli
from results.common import (BASELINES_COLORS, RL_CORE_HPARAMS, RL_SWEEP_SPEC,
                            SAFETY_SWEEP_SPEC, TRANSLATIONS, legend_ncol, legend_rows,
                            results_path, set_mpl_style)
from results.plotting_results.safety_param_sweep import discover_values
from results.plotting_results.seed_variance import ci95, load_final_value

# The two hyperparameter groups, keyed the way the rest of the sweep scripts key them.
PARAM_GROUPS: Dict[str, Dict[str, List[str]]] = {"rl": RL_SWEEP_SPEC, "safety": SAFETY_SWEEP_SPEC}
GROUP_LABELS: Dict[str, str] = {
    "rl": "Shared RL hyperparameters",
    "safety": "Safety-specific hyperparameters",
}
GROUP_FIG_PREFIX: Dict[str, str] = {"rl": "stage1_sweep", "safety": "hparam_sweep"}

# Algorithm order requested for the cross-algorithm figure and the heatmap rows.
ALGO_ORDER: List[str] = [
    "ppo", "ppo_cost", "ppo_lag", "ppo_pid", "ppo_saute", "p3o", "focops", "crpo",
]

# Colours already used by the sweep panels: reward left axis blue, cost right axis red.
REWARD_COLOR = "tab:blue"
COST_COLOR = "tab:red"
THRESHOLD_COLOR = "darkred"
MISSING_CELL_COLOR = "0.85"
COST_COLOR_PERCENTILE = 85
PANEL_W, PANEL_H = 4.5, 4.0
COMPARISON_PANEL_H = 3.6


@dataclass
class SweepData:
    """Final-performance statistics for one (algo, hyperparameter) sweep."""

    algo: str
    hparam: str
    group: str
    values: List[str]
    # (env, value, metric) -> (mean, ci95 half-width, n_seeds)
    stats: Dict[Tuple[str, str, str], Tuple[float, float, int]] = field(default_factory=dict)

    def get(self, env: str, value: str, metric: str) -> Optional[Tuple[float, float, int]]:
        return self.stats.get((env, value, metric))

    def series(self, env: str, metric: str) -> Tuple[np.ndarray, np.ndarray]:
        """(means, ci half-widths) over `self.values`, NaN where nothing was loaded."""
        means, cis = [], []
        for value in self.values:
            entry = self.get(env, value, metric)
            means.append(np.nan if entry is None else entry[0])
            cis.append(0.0 if entry is None else entry[1])
        return np.array(means, dtype=np.float64), np.array(cis, dtype=np.float64)


class SkipLog:
    """Collects everything that could not be loaded, for the end-of-run summary."""

    def __init__(self) -> None:
        self.entries: List[Tuple[str, str, str, str, str]] = []

    def add(self, reason: str, algo: str, hparam: str, env: str = "", value: str = "") -> None:
        self.entries.append((reason, algo, hparam, env, value))

    def summary(self, max_lines: int = 20) -> str:
        if not self.entries:
            return "Skipped nothing: every (algorithm, hyperparameter, environment, value) had data."
        by_reason: Dict[str, List[Tuple[str, str, str, str]]] = defaultdict(list)
        for reason, algo, hparam, env, value in self.entries:
            by_reason[reason].append((algo, hparam, env, value))

        lines = [f"Skipped {len(self.entries)} combination(s):"]
        for reason, items in sorted(by_reason.items()):
            lines.append(f"  {reason} ({len(items)}):")
            for algo, hparam, env, value in items[:max_lines]:
                parts = [p for p in (algo, hparam, env, value) if p]
                lines.append(f"    - {' / '.join(parts)}")
            if len(items) > max_lines:
                lines.append(f"    ... and {len(items) - max_lines} more")
        return "\n".join(lines)


def sort_values(values: Sequence[str]) -> List[str]:
    """Numeric sort of folder-name value suffixes, falling back to string order."""
    try:
        return sorted(values, key=lambda v: float(v))
    except ValueError:
        return sorted(values)


def discover_seeds(value_dir: Path) -> List[int]:
    """Seed numbers with a parquet in `value_dir`, numerically sorted."""
    seeds = []
    for fp in value_dir.glob("seed_*.parquet"):
        try:
            seeds.append(int(fp.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(seeds)


def load_sweep(
        base: Path, envs: List[str], level: int, algo: str, hparam: str, group: str,
        seeds: Optional[List[int]], last_frac: float, ci_method: str, skips: SkipLog,
) -> Optional[SweepData]:
    """Load one sweep, or None when no environment has any data for it.

    `seeds=None` uses every seed found on disk, which differs per value directory
    when a sweep point was only partially rerun.
    """
    values = sort_values(set().union(*[set(discover_values(base, env, level, algo, hparam))
                                       for env in envs]) or set())
    if not values:
        skips.add("no sweep values on disk", algo, hparam)
        return None

    data = SweepData(algo=algo, hparam=hparam, group=group, values=values)
    for env in envs:
        for value in values:
            value_dir = base / env / f"level_{level}" / algo / f"{hparam}_{value}"
            if not value_dir.is_dir():
                skips.add("missing value directory", algo, hparam, env, value)
                continue

            available = discover_seeds(value_dir)
            wanted = available if seeds is None else [s for s in seeds if s in available]
            if not wanted:
                skips.add("no matching seeds", algo, hparam, env, value)
                continue

            for metric in ("reward", "cost"):
                vals = []
                for seed in wanted:
                    v = load_final_value(value_dir / f"seed_{seed}.parquet", algo, env,
                                         metric, last_frac)
                    if v is not None:
                        vals.append(v)
                if not vals:
                    skips.add(f"unreadable {metric} runs", algo, hparam, env, value)
                    continue
                arr = np.array(vals, dtype=np.float64)
                if len(arr) >= 2:
                    mean, _, ci = ci95(arr, method=ci_method)
                else:
                    mean, ci = float(arr[0]), 0.0
                data.stats[(env, value, metric)] = (mean, ci, len(arr))

    if not data.stats:
        skips.add("no readable runs", algo, hparam)
        return None
    return data


def add_threshold_line(ax: plt.Axes, threshold: float) -> Optional[plt.Line2D]:
    """Draw the cost threshold, unless it is so far off-scale that it hides the data.

    Returns the line for the legend, or None when it was dropped.
    """
    low, high = ax.get_ylim()
    span = high - low
    if not (low - 0.5 * span <= threshold <= high + 0.5 * span):
        return None
    line = ax.axhline(threshold, linestyle=(0, (2, 2)), color=THRESHOLD_COLOR,
                      linewidth=1.3, alpha=0.9, zorder=1)
    ax.set_ylim(min(low, threshold - 0.05 * span), max(high, threshold + 0.05 * span))
    return line


def plot_single_sweep(data: SweepData, envs: List[str], out_path: Path,
                      threshold: float) -> None:
    """One row of panels: reward (left axis) and cost (right axis) against the value."""
    fig, axes = plt.subplots(1, len(envs), figsize=(PANEL_W * len(envs), PANEL_H), squeeze=False)
    axes = axes[0]

    x = np.arange(len(data.values))
    handles: Dict[str, plt.Line2D] = {}
    for ax, env in zip(axes, envs):
        reward_means, reward_cis = data.series(env, "reward")
        cost_means, cost_cis = data.series(env, "cost")

        ax_cost = ax.twinx()
        r_line = ax.errorbar(x, reward_means, yerr=reward_cis, marker="o", color=REWARD_COLOR,
                             label="Reward", capsize=3)
        c_line = ax_cost.errorbar(x, cost_means, yerr=cost_cis, marker="s", color=COST_COLOR,
                                  linestyle="--", label="Cost", capsize=3)
        t_line = add_threshold_line(ax_cost, threshold)

        handles.setdefault("Reward", r_line)
        handles.setdefault("Cost", c_line)
        if t_line is not None:
            handles.setdefault(f"Threshold ({threshold:.0f})", t_line)

        ax.set_xticks(x)
        ax.set_xticklabels(data.values)
        ax.set_xlabel(TRANSLATIONS.get(data.hparam, data.hparam))
        ax.set_ylabel("Reward", color=REWARD_COLOR)
        ax_cost.set_ylabel("Cost", color=COST_COLOR)
        ax.tick_params(axis="y", labelcolor=REWARD_COLOR)
        ax_cost.tick_params(axis="y", labelcolor=COST_COLOR)
        ax.set_title(TRANSLATIONS.get(env, env))
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(f"{TRANSLATIONS.get(data.algo, data.algo)}: sweep over "
                 f"{TRANSLATIONS.get(data.hparam, data.hparam)}")
    labels = list(handles)
    fig.tight_layout()
    fig.legend(list(handles.values()), labels, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=len(labels), fancybox=True, shadow=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(hparam: str, sweeps: Dict[Tuple[str, str], SweepData], envs: List[str],
                    out_path: Path, threshold: float) -> bool:
    """Reward (top row) and cost (bottom row) against one hyperparameter, all algorithms."""
    algos = [a for a in ALGO_ORDER if (a, hparam) in sweeps]
    if not algos:
        return False

    # Algorithms may have swept different grids of the same hyperparameter.
    values = sort_values(set().union(*[set(sweeps[(a, hparam)].values) for a in algos]))
    pos = {v: i for i, v in enumerate(values)}

    fig, axes = plt.subplots(2, len(envs), figsize=(PANEL_W * len(envs), COMPARISON_PANEL_H * 2),
                             squeeze=False, sharex=True)
    handles: Dict[str, plt.Line2D] = {}
    for row, metric in enumerate(("reward", "cost")):
        for col, env in enumerate(envs):
            ax = axes[row][col]
            for algo in algos:
                data = sweeps[(algo, hparam)]
                xs, ys, es = [], [], []
                for value in data.values:
                    entry = data.get(env, value, metric)
                    if entry is None:
                        continue
                    xs.append(pos[value])
                    ys.append(entry[0])
                    es.append(entry[1])
                if not xs:
                    continue
                line = ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3,
                                   color=BASELINES_COLORS.get(algo), label=algo)
                handles.setdefault(algo, line)

            if metric == "cost":
                t_line = add_threshold_line(ax, threshold)
                if t_line is not None:
                    handles.setdefault("threshold", t_line)
                ax.set_xlabel(TRANSLATIONS.get(hparam, hparam))
                ax.set_xticks(np.arange(len(values)))
                ax.set_xticklabels(values)
            if col == 0:
                ax.set_ylabel(TRANSLATIONS.get(metric, metric))
            if row == 0:
                ax.set_title(TRANSLATIONS.get(env, env))
            ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(f"Sweep over {TRANSLATIONS.get(hparam, hparam)}")
    labels = [TRANSLATIONS.get(k, "Threshold" if k == "threshold" else k) for k in handles]
    rows = legend_rows(len(algos))
    fig.tight_layout()
    fig.legend(list(handles.values()), labels, loc="upper center",
               bbox_to_anchor=(0.5, -0.01 * rows),
               ncol=legend_ncol(len(algos), len(labels)), fancybox=True, shadow=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def reward_sensitivity(data: SweepData, envs: List[str]
                       ) -> Tuple[float, List[Tuple[str, float, float]]]:
    ratios: List[float] = []
    undefined: List[Tuple[str, float, float]] = []
    for env in envs:
        means = []
        for value in data.values:
            entry = data.get(env, value, "reward")
            if entry is not None:
                means.append(entry[0])
        if len(means) < 2:
            continue
        best, worst = max(means), min(means)
        if worst <= 0:
            undefined.append((env, best, worst))
            continue
        ratios.append(best / worst)
    return (float(np.mean(ratios)) if ratios else float("nan")), undefined


def cost_sensitivity(data: SweepData, envs: List[str], env_agg: str) -> float:
    per_env = []
    for env in envs:
        means = []
        for value in data.values:
            entry = data.get(env, value, "cost")
            if entry is not None:
                means.append(entry[0])
        if means:
            per_env.append(max(means))
    if not per_env:
        return float("nan")
    return float(np.max(per_env) if env_agg == "max" else np.mean(per_env))


def ordered_hparams(sweeps: Dict[Tuple[str, str], SweepData]) -> List[Tuple[str, str]]:
    """(group, hparam) columns for the heatmaps: shared RL ones first, then safety ones."""
    present = {(d.group, d.hparam) for d in sweeps.values()}
    rl = [("rl", h) for h in RL_CORE_HPARAMS if ("rl", h) in present]
    safety = [("safety", h) for algo in ALGO_ORDER for h in SAFETY_SWEEP_SPEC.get(algo, [])
              if ("safety", h) in present]
    # dict.fromkeys keeps the first occurrence of a hyperparameter shared by two algos.
    return list(dict.fromkeys(rl + safety))


def plot_sensitivity(sweeps: Dict[Tuple[str, str], SweepData], envs: List[str], metric: str,
                     out_path: Path, threshold: float, env_agg: str) -> bool:
    """Algorithms x hyperparameters heatmap, the two hyperparameter groups kept apart."""
    columns = ordered_hparams(sweeps)
    algos = [a for a in ALGO_ORDER if any(key[0] == a for key in sweeps)]
    if not columns or not algos:
        return False

    matrix = np.full((len(algos), len(columns)), np.nan)
    flagged: List[Tuple[int, int]] = []
    notes: List[str] = []
    for i, algo in enumerate(algos):
        for j, (_, hparam) in enumerate(columns):
            data = sweeps.get((algo, hparam))
            if data is None:
                continue
            if metric == "reward":
                value, undefined = reward_sensitivity(data, envs)
                if undefined:
                    flagged.append((i, j))
                    for env, best, worst in undefined:
                        notes.append(f"  {TRANSLATIONS.get(algo, algo)} / "
                                     f"{TRANSLATIONS.get(hparam, hparam)} / "
                                     f"{TRANSLATIONS.get(env, env)}: best {best:.1f}, "
                                     f"worst {worst:.1f} (<= 0, ratio undefined)")
            else:
                value = cost_sensitivity(data, envs, env_agg)
            matrix[i, j] = value

    finite = matrix[np.isfinite(matrix)]
    vmin = float(finite.min()) if finite.size else 0.0
    vmax = float(finite.max()) if finite.size else 1.0

    fig, ax = plt.subplots(figsize=(1.0 * len(columns) + 3.5, 0.55 * len(algos) + 3.0))
    if metric == "reward":
        cmap = plt.get_cmap("viridis").copy()
        norm = Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-9))
    else:
        cmap = plt.get_cmap("coolwarm").copy()
        over = finite[finite > threshold]
        upper = float(np.percentile(over, COST_COLOR_PERCENTILE)) if over.size else vmax
        upper = min(max(upper, threshold * 1.05), max(vmax, threshold * 1.05))
        if vmin < threshold < upper:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=threshold, vmax=upper)
        else:
            norm = Normalize(vmin=min(vmin, threshold), vmax=max(upper, threshold + 1e-9))
        extend = "max" if vmax > upper + 1e-9 else "neither"
    cmap.set_bad(MISSING_CELL_COLOR)

    im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, norm=norm, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02,
                        extend="neither" if metric == "reward" else extend)
    cbar.set_label("Best / worst reward" if metric == "reward" else "Max final cost")
    if metric == "cost":
        cbar.ax.axhline(threshold, color="black", linewidth=1.2)

    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels([TRANSLATIONS.get(h, h) for _, h in columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(algos)))
    ax.set_yticklabels([TRANSLATIONS.get(a, a) for a in algos])
    ax.set_xticks(np.arange(len(columns) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(algos) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    for i in range(len(algos)):
        for j in range(len(columns)):
            value = matrix[i, j]
            if not np.isfinite(value):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=10, color="0.35")
                continue
            # White only on the dark ends of either colormap, black on the light middle.
            shade = float(np.clip(norm(value), 0.0, 1.0))
            dark = shade > 0.8 if metric == "reward" else (shade < 0.15 or shade > 0.85)
            label = f"{value:.1f}" + ("*" if (i, j) in flagged else "")
            ax.text(j, i, label, ha="center", va="center", fontsize=10,
                    color="white" if dark else "black")

    # Separate the two hyperparameter groups and label them above the columns.
    boundaries = [j for j in range(1, len(columns)) if columns[j][0] != columns[j - 1][0]]
    for j in boundaries:
        ax.axvline(j - 0.5, color="black", linewidth=2.5)
    start = 0
    for end in boundaries + [len(columns)]:
        group = columns[start][0]
        ax.text((start + end - 1) / 2, -0.85, GROUP_LABELS[group], ha="center", va="bottom",
                fontsize=12.5, fontweight="bold", clip_on=False)
        start = end

    title = ("Reward sensitivity: best / worst final reward across a hyperparameter's values"
             if metric == "reward" else
             f"Cost exposure: highest final cost across a hyperparameter's values "
             f"({'worst' if env_agg == 'max' else 'mean'} environment)")
    ax.set_title(title, pad=34)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    if notes:
        print("Reward ratios excluded (worst value <= 0), marked '*' in the heatmap:")
        print("\n".join(notes))
    return True


def latex_table(group: str, sweeps: Dict[Tuple[str, str], SweepData], envs: List[str],
                threshold: float) -> Optional[str]:
    """Raw final numbers for one hyperparameter group, as a booktabs table."""
    keys = [(a, h) for a in ALGO_ORDER for h in PARAM_GROUPS[group].get(a, [])
            if (a, h) in sweeps]
    if not keys:
        return None

    lines = [
        f"% Auto-generated LaTeX table ({GROUP_LABELS[group]} sweep, appendix)",
        "% Requires: \\usepackage{booktabs}, \\usepackage{xcolor}",
        "% Define: \\definecolor{safegreen}{RGB}{34, 139, 34}",
        "",
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\footnotesize",
        f"\\begin{{tabular}}{{lll{'cc' * len(envs)}}}",
        "\\toprule",
    ]

    header = ["", "", ""] + [f"\\multicolumn{{2}}{{c}}{{{TRANSLATIONS.get(env, env)}}}"
                             for env in envs]
    lines.append(" & ".join(header) + " \\\\")
    lines.append(" ".join(f"\\cmidrule(lr){{{4 + 2 * i}-{5 + 2 * i}}}" for i in range(len(envs))))
    lines.append(" & ".join(["Algorithm", "Hyperparameter", "Value"]
                            + ["R $\\uparrow$", "C $\\downarrow$"] * len(envs)) + " \\\\")
    lines.append("\\midrule")

    for block, (algo, hparam) in enumerate(keys):
        if block:
            lines.append("\\midrule")
        data = sweeps[(algo, hparam)]
        for row, value in enumerate(data.values):
            cells = [TRANSLATIONS.get(algo, algo) if row == 0 else "",
                     TRANSLATIONS.get(hparam, hparam) if row == 0 else "",
                     value]
            for env in envs:
                reward = data.get(env, value, "reward")
                cost = data.get(env, value, "cost")
                cells.append("--" if reward is None else f"${reward[0]:.1f} \\pm {reward[1]:.1f}$")
                if cost is None:
                    cells.append("--")
                else:
                    cost_str = f"${cost[0]:.1f} \\pm {cost[1]:.1f}$"
                    if cost[0] < threshold:
                        cost_str = f"\\textcolor{{safegreen}}{{{cost_str}}}"
                    cells.append(cost_str)
            lines.append(" & ".join(cells) + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{GROUP_LABELS[group]}: final reward (R) and cost (C), mean $\\pm$ 95\\% CI "
        "over seeds, averaged over the last fraction of training. "
        f"\\textcolor{{safegreen}}{{Green}} indicates safe (cost $< {threshold:.0f}$). "
        "Missing entries (--) had no completed runs.}",
        f"\\label{{tab:sweep_{group}}}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


def main(args: argparse.Namespace) -> None:
    set_mpl_style()
    base = results_path(args.input)
    fig_dir = results_path(args.output_fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    skips = SkipLog()

    sweeps: Dict[Tuple[str, str], SweepData] = {}
    for group, spec in PARAM_GROUPS.items():
        for algo, hparams in spec.items():
            for hparam in hparams:
                data = load_sweep(base, args.envs, args.level, algo, hparam, group,
                                  args.seeds, args.last_frac, args.ci_method, skips)
                if data is not None:
                    sweeps[(algo, hparam)] = data
    print(f"Loaded {len(sweeps)} (algorithm, hyperparameter) sweep(s) from {base}")

    if "sweeps" in args.outputs:
        for (algo, hparam), data in sweeps.items():
            out_path = fig_dir / f"{GROUP_FIG_PREFIX[data.group]}_{algo}_{hparam}.pdf"
            plot_single_sweep(data, args.envs, out_path, args.threshold)
            print(f"Saved figure: {out_path}")

    if "comparison" in args.outputs:
        out_path = fig_dir / f"sweep_comparison_{args.compare_param}.pdf"
        if plot_comparison(args.compare_param, sweeps, args.envs, out_path, args.threshold):
            print(f"Saved figure: {out_path}")
        else:
            print(f"No algorithm swept '{args.compare_param}', skipping the comparison figure.")

    if "sensitivity" in args.outputs:
        for metric in ("reward", "cost"):
            out_path = fig_dir / f"sweep_sensitivity_{metric}.pdf"
            if plot_sensitivity(sweeps, args.envs, metric, out_path, args.threshold,
                                args.cost_env_agg):
                print(f"Saved figure: {out_path}")
            else:
                print(f"Not enough data for the {metric} sensitivity heatmap.")

    if "tables" in args.outputs:
        for group in PARAM_GROUPS:
            table = latex_table(group, sweeps, args.envs, args.threshold)
            if table is None:
                print(f"No data for the '{group}' table, skipping.")
                continue
            out_path = fig_dir / f"sweep_table_{group}.tex"
            out_path.write_text(table + "\n")
            print(f"\n{'=' * 80}\n{GROUP_LABELS[group]} table ({out_path}):\n{'=' * 80}\n")
            print(table)

    print(f"\n{skips.summary()}")


def build_args() -> argparse.ArgumentParser:
    p = cli.plot_parser(
        "Appendix figures and tables for the hyperparameter sweeps.",
        omit=("algos", "metrics", "layout", "out_name", "seeds"),
        stats=True,
        envs=cli.DEFAULT_PLOT_ENVS,
        ci_method="normal",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Seeds to include (default: every seed present on disk)")
    p.add_argument("--compare_param", type=str, default="entropy_cost",
                   help="Hyperparameter for the cross-algorithm comparison figure")
    p.add_argument("--cost_env_agg", type=str, default="max", choices=["max", "mean"],
                   help="How the cost heatmap aggregates each sweep over environments")
    p.add_argument("--outputs", type=str, nargs="+",
                   default=["sweeps", "comparison", "sensitivity", "tables"],
                   choices=["sweeps", "comparison", "sensitivity", "tables"],
                   help="Which outputs to produce")
    return p


if __name__ == "__main__":
    main(build_args().parse_args())
