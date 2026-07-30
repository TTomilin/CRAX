"""Compare variance / confidence-interval width across different seed-set sizes.

Answers: "how much does the CI on final performance shrink if we run more seeds?"
Compares a small seed set (default {1,2,3}) against a large one (default {1..10}),
averaged across algorithms, per environment. Also plots the full CI-vs-#seeds
trend using cumulative seed prefixes (1, 1-2, 1-2-3, ..., 1-N).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from results.common import (
    DEFAULT_METRIC_COLS as METRIC_COLS,
    get_series,
    set_mpl_style,
    TRANSLATIONS,
)

# Two-tailed 95% Student-t critical values by degrees of freedom (n - 1).
# Small-sample CIs use these instead of the z=1.96 normal approximation,
# since n is as low as 3 here and the t-distribution has heavier tails.
T_CRIT_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    25: 2.060, 30: 2.042,
}


def t_crit(df: int) -> float:
    """95% two-tailed t critical value for the given degrees of freedom."""
    if df <= 0:
        return float("nan")
    if df in T_CRIT_95:
        return T_CRIT_95[df]
    keys = sorted(T_CRIT_95)
    if df > keys[-1]:
        return 1.960
    # linear-interpolate between the nearest tabulated dof
    lo = max(k for k in keys if k < df)
    hi = min(k for k in keys if k > df)
    frac = (df - lo) / (hi - lo)
    return T_CRIT_95[lo] + frac * (T_CRIT_95[hi] - T_CRIT_95[lo])


def load_final_value(fp: Path, algo: str, env: str, metric: str, last_frac: float) -> Optional[float]:
    """Return the mean of the last `last_frac` fraction of the run's metric curve."""
    if not fp.exists():
        return None
    df = pd.read_parquet(fp, engine="pyarrow")
    if "_step" not in df.columns:
        return None
    df = df.sort_values("_step", kind="mergesort")
    series = get_series(df, algo=algo, metric=metric, metric_cols=METRIC_COLS, env_name=env)
    if series is None or series.empty:
        return None
    n_tail = max(1, int(np.ceil(len(series) * last_frac)))
    return float(series.iloc[-n_tail:].mean())


def load_all_final_values(
        base: Path, envs: List[str], algos: List[str], level: int,
        seeds: List[int], metric: str, last_frac: float,
) -> Dict[Tuple[str, str], Dict[int, float]]:
    """dict[(env, algo)] -> {seed: final_value}, skipping missing/unreadable runs."""
    out: Dict[Tuple[str, str], Dict[int, float]] = {}
    for env in envs:
        for algo in algos:
            per_seed: Dict[int, float] = {}
            for seed in seeds:
                fp = base / env / f"level_{level}" / algo / f"seed_{seed}.parquet"
                val = load_final_value(fp, algo, env, metric, last_frac)
                if val is not None:
                    per_seed[seed] = val
            out[(env, algo)] = per_seed
    return out


def ci95(values: np.ndarray, method: str = "normal") -> Tuple[float, float, float]:
    """(mean, std, CI95 half-width) for a 1-D array of per-seed scalars.

    method="normal" (default): matches the rest of the plotting scripts (e.g.
    alg_comp.py), i.e. ci = 1.96 * population_std(ddof=0) / sqrt(n). Used for the
    paper's original results.
    method="t": Student's t critical value with n-1 dof and sample std (ddof=1).
    More accurate at small n (heavier tails), but not what the paper originally used.
    """
    n = len(values)
    if n < 2:
        return float(values.mean()) if n else float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    if method == "t":
        std = float(values.std(ddof=1))
        crit = t_crit(n - 1)
    else:
        std = float(values.std(ddof=0))
        crit = 1.960
    return mean, std, crit * std / np.sqrt(n)


def per_algo_ci(
        final_values: Dict[Tuple[str, str], Dict[int, float]],
        env: str, algos: List[str], seeds: List[int], method: str = "normal",
) -> Tuple[List[float], List[float], List[float], List[int]]:
    """Per-algo (std, CI95 abs, CI95 relative-to-|mean| in %, actual seed count).

    `counts` has one entry per algo in `algos` (actual seeds found, even if <2 and thus
    excluded from the other three lists) so callers can detect crashed/missing runs.
    """
    stds, cis, rel_cis, counts = [], [], [], []
    for algo in algos:
        per_seed = final_values.get((env, algo), {})
        vals = np.array([per_seed[s] for s in seeds if s in per_seed], dtype=np.float64)
        counts.append(len(vals))
        if len(vals) < 2:
            continue
        mean, std, ci = ci95(vals, method=method)
        stds.append(std)
        cis.append(ci)
        if abs(mean) > 1e-8:
            rel_cis.append(ci / abs(mean) * 100.0)
    return stds, cis, rel_cis, counts


def build_table(
        final_values_by_metric: Dict[str, Dict[Tuple[str, str], Dict[int, float]]],
        envs: List[str], algos: List[str], metrics: List[str],
        seeds_small: List[int], seeds_large: List[int], method: str = "normal",
) -> List[List]:
    """One row per env: CI95% is a single number, pooled (averaged) across both algos and metrics."""
    rows = []
    for env in envs:
        rel_s_all, rel_l_all, counts_s_all, counts_l_all = [], [], [], []
        for metric in metrics:
            final_values = final_values_by_metric[metric]
            _, _, rel_s, counts_s = per_algo_ci(final_values, env, algos, seeds_small, method=method)
            _, _, rel_l, counts_l = per_algo_ci(final_values, env, algos, seeds_large, method=method)
            rel_s_all.extend(rel_s)
            rel_l_all.extend(rel_l)
            counts_s_all.extend(counts_s)
            counts_l_all.extend(counts_l)
        if not rel_s_all or not rel_l_all:
            continue
        rel_ci_s, rel_ci_l = float(np.mean(rel_s_all)), float(np.mean(rel_l_all))
        reduction = (1.0 - rel_ci_l / rel_ci_s) * 100.0 if rel_ci_s > 0 else float("nan")
        avail_s = f"{min(counts_s_all)}-{max(counts_s_all)}/{len(seeds_small)}"
        avail_l = f"{min(counts_l_all)}-{max(counts_l_all)}/{len(seeds_large)}"
        rows.append([
            TRANSLATIONS.get(env, env),
            avail_s, rel_ci_s,
            avail_l, rel_ci_l,
            reduction,
        ])
    return rows


def missing_seed_report(
        final_values: Dict[Tuple[str, str], Dict[int, float]],
        envs: List[str], algos: List[str], seeds: List[int],
) -> List[str]:
    """One line per (env, algo) that is missing at least one of `seeds` (crashed/absent runs)."""
    lines = []
    for env in envs:
        for algo in algos:
            per_seed = final_values.get((env, algo), {})
            missing = [s for s in seeds if s not in per_seed]
            if missing:
                lines.append(
                    f"  {TRANSLATIONS.get(env, env)} / {algo}: missing seeds {missing} "
                    f"({len(per_seed)}/{len(seeds)} available)"
                )
    return lines


def format_table(rows: List[List], headers: List[str]) -> str:
    str_rows = [[f"{c:.3f}" if isinstance(c, float) else str(c) for c in row] for row in rows]
    widths = [max(len(h), *(len(r[i]) for r in str_rows)) if str_rows else len(h)
              for i, h in enumerate(headers)]

    def fmt_row(cells: List[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths))

    lines = [fmt_row(headers), fmt_row(["-" * w for w in widths])]
    lines += [fmt_row(r) for r in str_rows]
    return "\n".join(lines)


def build_trend(
        final_values_by_metric: Dict[str, Dict[Tuple[str, str], Dict[int, float]]],
        envs: List[str], algos: List[str], metrics: List[str], seeds_large: List[int],
        method: str = "normal",
) -> Dict[str, List[float]]:
    """trend[env] = relative CI95 (%) pooled across algos and metrics, for cumulative prefixes seeds_large[:k]."""
    trend: Dict[str, List[float]] = {}
    for env in envs:
        vals = []
        for k in range(2, len(seeds_large) + 1):
            rel_cis_all = []
            for metric in metrics:
                _, _, rel_cis, _ = per_algo_ci(final_values_by_metric[metric], env, algos, seeds_large[:k],
                                               method=method)
                rel_cis_all.extend(rel_cis)
            vals.append(float(np.mean(rel_cis_all)) if rel_cis_all else float("nan"))
        trend[env] = vals
    return trend


def plot_results(
        rows: List[List], trend: Dict[str, List[float]], envs: List[str],
        seeds_small: List[int], seeds_large: List[int], out_path: Path,
) -> None:
    set_mpl_style()
    fig, (ax_bar, ax_line) = plt.subplots(1, 2, figsize=(13, 5))

    env_labels = [r[0] for r in rows]
    ci_small = [r[2] for r in rows]
    ci_large = [r[4] for r in rows]

    x = np.arange(len(env_labels))
    width = 0.35
    ax_bar.bar(x - width / 2, ci_small, width, label=f"n={len(seeds_small)} seeds", color="tab:orange")
    ax_bar.bar(x + width / 2, ci_large, width, label=f"n={len(seeds_large)} seeds", color="tab:blue")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(env_labels, rotation=30, ha="right")
    ax_bar.set_ylabel("95% CI half-width (% of mean)")
    ax_bar.set_title("CI width: small vs. large seed set")
    ax_bar.legend()
    ax_bar.grid(True, axis="y", linestyle="--", alpha=0.4)

    ks = list(range(2, len(seeds_large) + 1))
    for env in envs:
        vals = trend.get(env, [])
        if not vals:
            continue
        ax_line.plot(ks, vals, marker="o", markersize=3, label=TRANSLATIONS.get(env, env), alpha=0.85)

    overall = np.nanmean(np.array([trend[e] for e in envs if trend.get(e)]), axis=0)
    ax_line.plot(ks, overall, color="black", linewidth=3, label="Average", zorder=10)

    ax_line.axvline(len(seeds_small), linestyle="--", color="gray", linewidth=1)
    ax_line.axvline(len(seeds_large), linestyle="--", color="gray", linewidth=1)
    ax_line.set_xlabel("Number of seeds (cumulative)")
    ax_line.set_ylabel("95% CI half-width (% of mean)")
    ax_line.set_title("CI width vs. number of seeds")
    ax_line.legend(fontsize=9, ncol=1, loc="upper right")
    ax_line.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.show()


def main(args: argparse.Namespace) -> None:
    base = Path(__file__).parent.parent.resolve() / args.input
    seeds_large = sorted(args.seeds_large)
    seeds_small = sorted(args.seeds_small)
    all_seeds = sorted(set(seeds_small) | set(seeds_large))

    final_values_by_metric = {
        metric: load_all_final_values(base, args.envs, args.algos, args.level, all_seeds, metric, args.last_frac)
        for metric in args.metrics
    }

    for metric, final_values in final_values_by_metric.items():
        missing_lines = missing_seed_report(final_values, args.envs, args.algos, all_seeds)
        if missing_lines:
            print(f"\nMissing runs (crashed or absent), {len(missing_lines)} env/algo combos affected ({metric}):")
            print("\n".join(missing_lines))

    rows = build_table(final_values_by_metric, args.envs, args.algos, args.metrics,
                       seeds_small, seeds_large, method=args.ci_method)
    headers = [
        "Env",
        "avail_small(min-max/req)", "CI95%(n_small)",
        "avail_large(min-max/req)", "CI95%(n_large)",
        "CI reduction %",
    ]
    ci_desc = "1.96 * population std / sqrt(n) [matches other plotting scripts]" if args.ci_method == "normal" \
        else "Student's t(n-1) * sample std / sqrt(n)"
    print(f"\nMetrics: {args.metrics} (pooled into a single CI per env)  |  final-perf window: last "
          f"{args.last_frac:.0%} of steps  |  averaged across algos: {args.algos}  |  CI method: {ci_desc}\n")
    print(format_table(rows, headers))

    if rows:
        avg_reduction = float(np.mean([r[-1] for r in rows]))
        print(f"\nAverage CI reduction across envs going from {len(seeds_small)} to "
              f"{len(seeds_large)} seeds: {avg_reduction:.1f}%")

    trend = build_trend(final_values_by_metric, args.envs, args.algos, args.metrics, seeds_large,
                        method=args.ci_method)
    out_path = Path(args.output_fig_dir) / f"{args.out_name}.pdf"
    plot_results(rows, trend, args.envs, seeds_small, seeds_large, out_path)
    print(f"Saved figure: {out_path}")


def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare CI/variance of final performance across seed-set sizes.")
    p.add_argument("--input", type=str, default="data")
    p.add_argument("--envs", type=str, nargs="+",
                   default=["safe_reacher", "safe_goal_point", "safe_push_point", "safe_lift_spider",
                            "safe_circle_point", "safe_height_humanoid", "safe_pathway_walker2d",
                            "safe_velocity_humanoid"])
    p.add_argument("--algos", type=str, nargs="+",
                   default=["ppo", "ppo_cost", "ppo_lag", "ppo_pid", "ppo_saute", "p3o", "focops"])
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--metrics", type=str, nargs="+", default=["reward", "cost"], choices=list(METRIC_COLS.keys()))
    p.add_argument("--seeds_small", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--seeds_large", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    p.add_argument("--last_frac", type=float, default=0.1,
                   help="Fraction of the tail of each run averaged to obtain the per-seed final performance.")
    p.add_argument("--ci_method", type=str, default="t", choices=["normal", "t"],
                   help="'normal' (default): 1.96 * population std / sqrt(n). 't': Student's t(n-1) * sample "
                        "std / sqrt(n), more accurate at small n.")
    p.add_argument("--output_fig_dir", type=str, default="figures")
    p.add_argument("--out_name", type=str, default="seed_variance")
    return p


if __name__ == "__main__":
    args = build_args().parse_args()
    main(args)
