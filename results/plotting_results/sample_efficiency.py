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
2. Cumulative violation (CumViol): area by which cost exceeds the safety
   threshold, i.e. integral of max(cost - threshold, 0) over steps, also
   normalized by steps spanned. Lower is better. The raw value is in
   task-specific cost units, so it is not comparable across tasks. To make it
   comparable, it is also normalized by the same statistic computed for an
   unconstrained reference algorithm (`--ref_algo`, default "ppo") on the same
   task: CumViol_norm = CumViol / CumViol_ref. Controlled by `--viol_norm`
   ("reference" by default, or "none" to keep only the raw value).

Outputs a table (printed + saved as CSV) and a heatmap figure (algorithms x
environments, one panel for reward efficiency and one for cumulative
violation), saved as a PDF.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from results.common import (
    DEFAULT_METRIC_COLS as METRIC_COLS,
    get_series,
    set_mpl_style,
    TRANSLATIONS,
)
from results.plotting_results.seed_variance import ci95

# (env, algo) -> {seed: (reward_eff, viol_raw, viol_norm)}
EfficiencyBySeed = Dict[Tuple[str, str], Dict[int, Tuple[float, float, float]]]
# (env, algo) -> {seed: (reward_auc/span, viol_auc/span, final_reward)}
RawEfficiencyBySeed = Dict[Tuple[str, str], Dict[int, Tuple[float, float, float]]]


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
    reward_auc = np.trapezoid(reward_vals, x=steps)
    violation = np.clip(cost.to_numpy(dtype=np.float64) - threshold, 0.0, None)
    violation_auc = np.trapezoid(violation, x=steps)

    n_tail = max(1, int(np.ceil(len(reward_vals) * last_frac)))
    final_reward = float(reward_vals[-n_tail:].mean())

    return reward_auc / span, violation_auc / span, final_reward


def load_efficiency(
        base: Path, envs: List[str], algos: List[str], level: int, seeds: List[int], threshold: float,
        last_frac: float, ref_algo: str, viol_norm: str,
) -> EfficiencyBySeed:
    # Always load the reference algo too (even if not requested in --algos), so its
    # violation stat is available to normalize every other algo's CumViol per env.
    algos_to_load = list(algos)
    if ref_algo not in algos_to_load:
        algos_to_load.append(ref_algo)

    raw: RawEfficiencyBySeed = {}
    for env in envs:
        for algo in algos_to_load:
            per_seed: Dict[int, Tuple[float, float, float]] = {}
            for seed in seeds:
                fp = base / env / f"level_{level}" / algo / f"seed_{seed}.parquet"
                vals = per_seed_efficiency(fp, algo, env, threshold, last_frac)
                if vals is not None:
                    per_seed[seed] = vals
                else:
                    print(f"Skipping (missing/unreadable): {fp}")
            raw[(env, algo)] = per_seed

    # Reference violation per env: mean raw CumViol across the reference algo's seeds.
    ref_viol: Dict[str, float] = {}
    for env in envs:
        ref_vals = [v[1] for v in raw.get((env, ref_algo), {}).values()]
        ref_viol[env] = float(np.mean(ref_vals)) if ref_vals else float("nan")

    # Normalize reward efficiency by the max final reward achieved on each task (across
    # all requested methods/seeds), so it reads as "fraction of best achievable, integrated
    # over training".
    out: EfficiencyBySeed = {}
    for env in envs:
        finals = [v[2] for algo in algos for v in raw.get((env, algo), {}).values()]
        max_final = max(finals) if finals else float("nan")

        rv = ref_viol.get(env, float("nan"))
        use_ref_norm = viol_norm == "reference" and np.isfinite(rv) and rv > 0
        if viol_norm == "reference" and not use_ref_norm:
            print(f"Warning: no usable reference violation for env={env!r} "
                  f"(ref_algo={ref_algo!r}); leaving CumViol unnormalized for this env.")

        for algo in algos:
            per_seed = raw.get((env, algo), {})
            entries: Dict[int, Tuple[float, float, float]] = {}
            for s, (r, v, _) in per_seed.items():
                r_out = r / max_final if max_final and max_final > 0 else float("nan")
                v_norm = v / rv if use_ref_norm else v
                entries[s] = (r_out, v, v_norm)
            out[(env, algo)] = entries
    return out


def build_table(
        data: EfficiencyBySeed, envs: List[str], algos: List[str], ci_method: str,
) -> List[List]:
    """One row per (env, algo), plus an 'Overall' row per algo averaged across envs."""
    rows: List[List] = []
    overall: Dict[str, List[Tuple[float, float, float]]] = {algo: [] for algo in algos}

    for env in envs:
        for algo in algos:
            per_seed = data.get((env, algo), {})
            if not per_seed:
                continue
            reward_vals = np.array([v[0] for v in per_seed.values()])
            viol_vals = np.array([v[1] for v in per_seed.values()])
            viol_norm_vals = np.array([v[2] for v in per_seed.values()])
            r_mean, _, r_ci = ci95(reward_vals, method=ci_method)
            v_mean, _, v_ci = ci95(viol_vals, method=ci_method)
            vn_mean, _, vn_ci = ci95(viol_norm_vals, method=ci_method)
            rows.append([
                TRANSLATIONS.get(env, env), TRANSLATIONS.get(algo, algo), len(per_seed),
                r_mean, r_ci, v_mean, v_ci, vn_mean, vn_ci,
            ])
            overall[algo].append((r_mean, v_mean, vn_mean))

    for algo in algos:
        triples = overall[algo]
        if not triples:
            continue
        r_mean, _, r_ci = ci95(np.array([t[0] for t in triples]), method=ci_method)
        v_mean, _, v_ci = ci95(np.array([t[1] for t in triples]), method=ci_method)
        vn_mean, _, vn_ci = ci95(np.array([t[2] for t in triples]), method=ci_method)
        rows.append(["Overall", TRANSLATIONS.get(algo, algo), len(triples),
                     r_mean, r_ci, v_mean, v_ci, vn_mean, vn_ci])

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


def plot_efficiency_heatmap(
        data: EfficiencyBySeed, envs: List[str], algos: List[str], ci_method: str,
        viol_norm: str, annotate_ci: bool, out_path: Path,
) -> None:
    """Two heatmaps side by side: reward efficiency and cumulative violation, algos x envs."""
    set_mpl_style()

    n_env = len(envs)
    n_algo = len(algos)

    reward_mean = np.full((n_algo, n_env), np.nan)
    reward_ci = np.full((n_algo, n_env), np.nan)
    viol_mean = np.full((n_algo, n_env), np.nan)
    viol_ci = np.full((n_algo, n_env), np.nan)

    for i, algo in enumerate(algos):
        for j, env in enumerate(envs):
            per_seed = data.get((env, algo), {})
            if not per_seed:
                continue
            r_vals = np.array([v[0] for v in per_seed.values()])
            v_vals = np.array([(v[2] if viol_norm == "reference" else v[1]) for v in per_seed.values()])
            r_mean, _, r_ci = ci95(r_vals, method=ci_method)
            v_mean, _, v_ci = ci95(v_vals, method=ci_method)
            reward_mean[i, j] = r_mean
            reward_ci[i, j] = r_ci
            viol_mean[i, j] = v_mean
            viol_ci[i, j] = v_ci

    reward_masked = np.ma.masked_invalid(reward_mean)
    viol_masked = np.ma.masked_invalid(viol_mean)

    if viol_norm == "reference":
        viol_vmax = 1.0
    else:
        finite_viol = viol_mean[np.isfinite(viol_mean)]
        viol_vmax = float(np.percentile(finite_viol, 95)) if finite_viol.size else 1.0

    panel_w = 0.75 * n_env + 1.6
    panel_h = 0.45 * n_algo + 1.4
    fig, (ax_r, ax_v) = plt.subplots(1, 2, figsize=(panel_w * 2, panel_h))

    env_labels = [TRANSLATIONS.get(e, e) for e in envs]
    algo_labels = [TRANSLATIONS.get(a, a) for a in algos]

    cmap_r = plt.get_cmap("Greens").copy()
    cmap_r.set_bad(color="lightgrey")
    cmap_v = plt.get_cmap("Reds").copy()
    cmap_v.set_bad(color="lightgrey")

    im_r = ax_r.imshow(reward_masked, cmap=cmap_r, vmin=0, vmax=1, aspect="auto")
    im_v = ax_v.imshow(viol_masked, cmap=cmap_v, vmin=0, vmax=viol_vmax, aspect="auto")

    for ax, title in ((ax_r, r"Reward efficiency $\uparrow$"), (ax_v, r"Cumulative violation $\downarrow$")):
        ax.set_xticks(np.arange(n_env))
        ax.set_xticklabels(env_labels, rotation=30, ha="right")
        ax.set_yticks(np.arange(n_algo))
        ax.set_yticklabels(algo_labels)
        ax.set_title(title)
        ax.grid(False)

    def annotate(ax, mean: np.ndarray, ci: np.ndarray, scale_max: float) -> None:
        base_fs = 10
        for i in range(n_algo):
            for j in range(n_env):
                m = mean[i, j]
                if not np.isfinite(m):
                    ax.text(j, i, "–", ha="center", va="center", fontsize=base_fs, color="dimgray")
                    continue
                frac = m / scale_max if scale_max else 0.0
                color = "white" if frac > 0.6 else "black"
                y_off = -0.15 if (annotate_ci and np.isfinite(ci[i, j])) else 0.0
                ax.text(j, i + y_off, f"{m:.2f}", ha="center", va="center", fontsize=base_fs, color=color)
                if annotate_ci and np.isfinite(ci[i, j]):
                    ax.text(j, i + 0.28, f"±{ci[i, j]:.2f}", ha="center", va="center",
                            fontsize=base_fs * 0.7, color=color)

    annotate(ax_r, reward_mean, reward_ci, scale_max=1.0)
    annotate(ax_v, viol_mean, viol_ci, scale_max=viol_vmax)

    fig.colorbar(im_r, ax=ax_r, label="RewEff", fraction=0.046, pad=0.04)
    fig.colorbar(im_v, ax=ax_v, label="CumViol", fraction=0.046, pad=0.04)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.show()


def main(args: argparse.Namespace) -> None:
    base = Path(__file__).parent.parent.resolve() / args.input

    data = load_efficiency(base, args.envs, args.algos, args.level, args.seeds, args.threshold, args.last_frac,
                            args.ref_algo, args.viol_norm)

    rows = build_table(data, args.envs, args.algos, args.ci_method)
    headers = ["Env", "Algo", "N", "RewardEff", "RewardEff_CI95",
               "ViolationEff", "ViolationEff_CI95", "ViolationEff_Norm", "ViolationEff_Norm_CI95"]
    print(f"\nThreshold: {args.threshold}  |  final-perf window: last {args.last_frac:.0%} of steps  |  "
          f"CI method: {args.ci_method}  |  viol_norm: {args.viol_norm} (ref_algo={args.ref_algo})\n")
    print(format_table(rows, headers))

    out_dir = Path(args.output_fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.out_name}_table.csv"
    pd.DataFrame(rows, columns=headers).to_csv(csv_path, index=False)
    print(f"\nSaved table: {csv_path}")

    fig_path = out_dir / f"{args.out_name}_level_{args.level}.pdf"
    plot_efficiency_heatmap(data, args.envs, args.algos, args.ci_method, args.viol_norm, args.annotate_ci, fig_path)
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
    p.add_argument("--ref_algo", type=str, default="ppo",
                   help="Unconstrained reference algo used to normalize CumViol across tasks.")
    p.add_argument("--viol_norm", type=str, default="reference", choices=["none", "reference"],
                   help="'reference' divides CumViol by the ref_algo's mean CumViol on the same env; "
                        "'none' keeps only the raw (task-scale) value.")
    p.add_argument("--ci_method", type=str, default="t", choices=["normal", "t"])
    p.add_argument("--annotate_ci", action="store_true", default=False,
                   help="Add a second ±CI line under each heatmap cell's mean.")
    p.add_argument("--output_fig_dir", type=str, default="figures")
    p.add_argument("--out_name", type=str, default="sample_efficiency")
    return p


if __name__ == "__main__":
    args = build_args().parse_args()
    main(args)
