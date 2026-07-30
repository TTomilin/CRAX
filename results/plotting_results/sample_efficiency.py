"""Sample-efficiency metrics: reward AUC and cumulative constraint violation.

Two metrics per (env, algo, seed), computed directly from each run's own
training curve (no cross-seed alignment needed since these are per-seed
scalars):

1. Reward efficiency: area under the reward-vs-steps curve, normalized by the
   number of steps spanned, then further normalized by the max final reward
   achieved on that task (across all methods/seeds). This makes it "fraction
   of best achievable reward, integrated over training" — values land in
   ~[0, 1], higher meaning the method reached good performance sooner. Final
   reward per run is the mean of the last `last_frac` fraction of the curve
   (same convention as seed_variance.py).
2. Safety efficiency (cumulative constraint violation): area by which cost
   exceeds the safety threshold, i.e. integral of max(cost - threshold, 0)
   over steps, also normalized by steps spanned. Lower is better.

Outputs a table (printed + saved as CSV) and a bar-chart grid (env rows x
[reward efficiency, violation] columns), plus a saved PDF figure.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from results.common import (
    DEFAULT_METRIC_COLS as METRIC_COLS,
    get_series,
    set_mpl_style,
    nice_grid,
    BASELINES_COLORS, TRANSLATIONS,
)
from results.plotting_results.seed_variance import ci95

EfficiencyBySeed = Dict[
    Tuple[str, str], Dict[int, Tuple[float, float]]]  # (env, algo) -> {seed: (reward_eff, violation_eff)}
RawEfficiencyBySeed = Dict[Tuple[str, str], Dict[
    int, Tuple[float, float, float]]]  # ... -> {seed: (reward_auc/span, violation_eff, final_reward)}


def per_seed_efficiency(
        fp: Path, algo: str, env: str, threshold: float, last_frac: float,
) -> Tuple[float, float, float] | None:
    """Return (raw_reward_auc_per_step, violation_efficiency, final_reward) for a single run, or None if unreadable.

    `final_reward` is the mean of the last `last_frac` fraction of the reward curve (same
    convention as seed_variance.load_final_value), used later to normalize reward efficiency
    by the best final performance achieved on that task.
    """
    if not fp.exists():
        return None
    df = pd.read_parquet(fp, engine="pyarrow")
    if "_step" not in df.columns:
        return None
    df = df.sort_values("_step", kind="mergesort")

    reward = get_series(df, algo=algo, metric="reward", metric_cols=METRIC_COLS, env_name=env)
    cost = get_series(df, algo=algo, metric="cost", metric_cols=METRIC_COLS, env_name=env)
    if reward is None or cost is None:
        return None

    steps = df["_step"].to_numpy(dtype=np.float64)
    if len(steps) < 2:
        return None
    span = steps[-1] - steps[0]
    if span <= 0:
        return None

    reward_vals = reward.to_numpy(dtype=np.float64)
    reward_auc = np.trapz(reward_vals, x=steps)
    violation = np.clip(cost.to_numpy(dtype=np.float64) - threshold, 0.0, None)
    violation_auc = np.trapz(violation, x=steps)

    n_tail = max(1, int(np.ceil(len(reward_vals) * last_frac)))
    final_reward = float(reward_vals[-n_tail:].mean())

    return reward_auc / span, violation_auc / span, final_reward


def load_efficiency(
        base: Path, envs: List[str], algos: List[str], level: int, seeds: List[int], threshold: float,
        last_frac: float,
) -> EfficiencyBySeed:
    raw: RawEfficiencyBySeed = {}
    for env in envs:
        for algo in algos:
            per_seed: Dict[int, Tuple[float, float, float]] = {}
            for seed in seeds:
                fp = base / env / f"level_{level}" / algo / f"seed_{seed}.parquet"
                vals = per_seed_efficiency(fp, algo, env, threshold, last_frac)
                if vals is not None:
                    per_seed[seed] = vals
                else:
                    print(f"Skipping (missing/unreadable): {fp}")
            raw[(env, algo)] = per_seed

    # Normalize reward efficiency by the max final reward achieved on each task (across
    # all methods/seeds), so it reads as "fraction of best achievable, integrated over training".
    out: EfficiencyBySeed = {}
    for env in envs:
        finals = [v[2] for algo in algos for v in raw.get((env, algo), {}).values()]
        max_final = max(finals) if finals else float("nan")
        for algo in algos:
            per_seed = raw.get((env, algo), {})
            if max_final and max_final > 0:
                out[(env, algo)] = {s: (r / max_final, v) for s, (r, v, _) in per_seed.items()}
            else:
                out[(env, algo)] = {s: (float("nan"), v) for s, (r, v, _) in per_seed.items()}
    return out


def build_table(
        data: EfficiencyBySeed, envs: List[str], algos: List[str], ci_method: str,
) -> List[List]:
    """One row per (env, algo), plus an 'Overall' row per algo averaged across envs."""
    rows: List[List] = []
    overall: Dict[str, List[Tuple[float, float]]] = {algo: [] for algo in algos}

    for env in envs:
        for algo in algos:
            per_seed = data.get((env, algo), {})
            if not per_seed:
                continue
            reward_vals = np.array([v[0] for v in per_seed.values()])
            viol_vals = np.array([v[1] for v in per_seed.values()])
            r_mean, _, r_ci = ci95(reward_vals, method=ci_method)
            v_mean, _, v_ci = ci95(viol_vals, method=ci_method)
            rows.append([
                TRANSLATIONS.get(env, env), TRANSLATIONS.get(algo, algo), len(per_seed),
                r_mean, r_ci, v_mean, v_ci,
            ])
            overall[algo].append((r_mean, v_mean))

    for algo in algos:
        pairs = overall[algo]
        if not pairs:
            continue
        r_mean, _, r_ci = ci95(np.array([p[0] for p in pairs]), method=ci_method)
        v_mean, _, v_ci = ci95(np.array([p[1] for p in pairs]), method=ci_method)
        rows.append(["Overall", TRANSLATIONS.get(algo, algo), len(pairs), r_mean, r_ci, v_mean, v_ci])

    return rows


def format_table(rows: List[List], headers: List[str]) -> str:
    def fmt(c) -> str:
        return f"{c:.4g}" if isinstance(c, float) else str(c)

    str_rows = [[fmt(c) for c in row] for row in rows]
    widths = [max(len(h), *(len(r[i]) for r in str_rows)) if str_rows else len(h)
              for i, h in enumerate(headers)]

    def fmt_row(cells: List[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths))

    lines = [fmt_row(headers), fmt_row(["-" * w for w in widths])]
    lines += [fmt_row(r) for r in str_rows]
    return "\n".join(lines)


def plot_efficiency(
        data: EfficiencyBySeed, envs: List[str], algos: List[str], ci_method: str,
        max_cols: int, panel_w: float, panel_h: float, out_path: Path,
) -> None:
    set_mpl_style()

    n_env = len(envs)
    nrows, ncols_env = nice_grid(n_env, max_cols=max_cols)
    total_cols = ncols_env * 2

    fig, axs = plt.subplots(nrows, total_cols, figsize=(panel_w * total_cols, panel_h * nrows), squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.16, wspace=0.45, hspace=0.6)

    def get_ax(env_i: int, col: int):
        r = env_i // ncols_env
        c = (env_i % ncols_env) * 2 + col
        return axs[r, c]

    legend_handles: Dict[str, Patch] = {}
    x = np.arange(len(algos))
    colors = [BASELINES_COLORS.get(a, "gray") for a in algos]

    for env_i, env in enumerate(envs):
        reward_means, reward_cis, viol_means, viol_cis = [], [], [], []
        for algo in algos:
            per_seed = data.get((env, algo), {})
            if per_seed:
                r_vals = np.array([v[0] for v in per_seed.values()])
                v_vals = np.array([v[1] for v in per_seed.values()])
                r_mean, _, r_ci = ci95(r_vals, method=ci_method)
                v_mean, _, v_ci = ci95(v_vals, method=ci_method)
            else:
                r_mean = r_ci = v_mean = v_ci = float("nan")
            reward_means.append(r_mean);
            reward_cis.append(r_ci)
            viol_means.append(v_mean);
            viol_cis.append(v_ci)

        ax_r = get_ax(env_i, 0)
        ax_r.bar(x, reward_means, yerr=reward_cis, color=colors, capsize=3)
        ax_r.set_xticks(x)
        ax_r.set_xticklabels([])
        ax_r.set_ylabel("Reward eff. (frac. of best)")
        ax_r.grid(True, axis="y", linestyle="--", alpha=0.4)

        ax_v = get_ax(env_i, 1)
        ax_v.bar(x, viol_means, yerr=viol_cis, color=colors, capsize=3)
        ax_v.set_xticks(x)
        ax_v.set_xticklabels([])
        ax_v.set_ylabel("Violation eff. (AUC/step)")
        ax_v.grid(True, axis="y", linestyle="--", alpha=0.4)

        for algo, color in zip(algos, colors):
            if algo not in legend_handles:
                legend_handles[algo] = Patch(color=color, label=algo)

        left_bbox = ax_r.get_position()
        right_bbox = ax_v.get_position()
        row_x = 0.5 * (left_bbox.x0 + right_bbox.x1)
        row_y = max(left_bbox.y1, right_bbox.y1) + 0.015
        fig.text(row_x, row_y, TRANSLATIONS.get(env, env), ha="center", va="bottom", fontsize=13)

    for env_i in range(n_env, nrows * ncols_env):
        for col in range(2):
            get_ax(env_i, col).axis("off")

    if legend_handles:
        handles = list(legend_handles.values())
        labels = [TRANSLATIONS.get(a, a) for a in legend_handles.keys()]
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
                   ncol=min(len(labels), 10), fancybox=True, shadow=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.show()


def main(args: argparse.Namespace) -> None:
    base = Path(__file__).parent.parent.resolve() / args.input

    data = load_efficiency(base, args.envs, args.algos, args.level, args.seeds, args.threshold, args.last_frac)

    rows = build_table(data, args.envs, args.algos, args.ci_method)
    headers = ["Env", "Algo", "N", "RewardEff", "RewardEff_CI95", "ViolationEff", "ViolationEff_CI95"]
    print(f"\nThreshold: {args.threshold}  |  final-perf window: last {args.last_frac:.0%} of steps  |  "
          f"CI method: {args.ci_method}\n")
    print(format_table(rows, headers))

    out_dir = Path(args.output_fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.out_name}_table.csv"
    pd.DataFrame(rows, columns=headers).to_csv(csv_path, index=False)
    print(f"\nSaved table: {csv_path}")

    fig_path = out_dir / f"{args.out_name}_level_{args.level}.pdf"
    plot_efficiency(data, args.envs, args.algos, args.ci_method, args.max_cols, args.panel_w, args.panel_h, fig_path)
    print(f"Saved figure: {fig_path}")


def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reward-AUC and cumulative-constraint-violation efficiency metrics.")
    p.add_argument("--input", type=str, default="data")
    p.add_argument("--envs", type=str, nargs="+",
                   default=["safe_reacher", "safe_goal_point", "safe_push_point", "safe_lift_spider",
                            "safe_circle_point", "safe_height_humanoid", "safe_pathway_walker2d",
                            "safe_velocity_humanoid"])
    p.add_argument("--algos", type=str, nargs="+",
                   default=["ppo_lag", "ppo_pid", "p3o", "focops"])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--threshold", type=float, default=25.0, help="Safety cost threshold.")
    p.add_argument("--last_frac", type=float, default=0.1,
                   help="Fraction of the tail of each run averaged for final reward (reward-eff normalizer).")
    p.add_argument("--ci_method", type=str, default="t", choices=["normal", "t"])
    p.add_argument("--max_cols", type=int, default=2, help="Max env columns in grid.")
    p.add_argument("--panel_w", type=float, default=3.1)
    p.add_argument("--panel_h", type=float, default=3.0)
    p.add_argument("--output_fig_dir", type=str, default="figures")
    p.add_argument("--out_name", type=str, default="sample_efficiency")
    return p


if __name__ == "__main__":
    args = build_args().parse_args()
    main(args)
