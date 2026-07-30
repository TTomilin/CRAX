"""Seed-count CI comparison for curriculum and transfer training.

Each (env, mode) pair is treated as its own row/line (label "<Env> (Curriculum)"
/ "<Env> (Transfer)"), so both regimes show up side by side in the same table
and trend plot.

Data layout:
- Curriculum: <input>/curriculum/<env>/<algo>/seed_<n>.parquet, steps in
  `global_step` (cumulative across difficulty stages).
- Transfer: <input>/transfer/<env>/level_<level>/<algo>/seed_<n>.parquet
  (the safe fine-tuning phase; the unsafe PPO pretrain phase is ignored here
  since it doesn't affect the final-performance tail average).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from results.common import DEFAULT_METRIC_COLS as METRIC_COLS, get_series, TRANSLATIONS
from results.plotting_results.seed_variance import (
    build_table, missing_seed_report, format_table, build_trend, plot_results,
)

MODES = ("curriculum", "transfer")
MODE_LABELS = {"curriculum": "Curriculum", "transfer": "Transfer"}


def group_label(env: str, mode: str) -> str:
    env_label = TRANSLATIONS.get(env, env.replace("_", " ").title())
    return f"{env_label} ({MODE_LABELS[mode]})"


def build_fp(base: Path, mode: str, env: str, level: int, algo: str, seed: int) -> Path:
    if mode == "curriculum":
        return base / "curriculum" / env / algo / f"seed_{seed}.parquet"
    if mode == "transfer":
        return base / "transfer" / env / f"level_{level}" / algo / f"seed_{seed}.parquet"
    raise ValueError(f"Unknown mode: {mode}")


def load_final_value(fp: Path, algo: str, env: str, metric: str, last_frac: float) -> Optional[float]:
    """Same tail-average convention as seed_variance.load_final_value, but sorts by
    `global_step` when present (curriculum's cumulative-across-stages step counter)
    instead of assuming `_step`."""
    if not fp.exists():
        return None
    df = pd.read_parquet(fp, engine="pyarrow")
    step_col = "global_step" if "global_step" in df.columns else "_step"
    if step_col not in df.columns:
        return None
    df = df.sort_values(step_col, kind="mergesort")
    series = get_series(df, algo=algo, metric=metric, metric_cols=METRIC_COLS, env_name=env)
    if series is None or series.empty:
        return None
    n_tail = max(1, int(np.ceil(len(series) * last_frac)))
    return float(series.iloc[-n_tail:].mean())


def load_all_final_values(
        base: Path, envs: List[str], modes: List[str], algos: List[str], level: int,
        seeds: List[int], metric: str, last_frac: float,
) -> Dict[Tuple[str, str], Dict[int, float]]:
    """dict[(group_label, algo)] -> {seed: final_value}, one group per (env, mode)."""
    out: Dict[Tuple[str, str], Dict[int, float]] = {}
    for env in envs:
        for mode in modes:
            label = group_label(env, mode)
            for algo in algos:
                per_seed: Dict[int, float] = {}
                for seed in seeds:
                    fp = build_fp(base, mode, env, level, algo, seed)
                    val = load_final_value(fp, algo, env, metric, last_frac)
                    if val is not None:
                        per_seed[seed] = val
                out[(label, algo)] = per_seed
    return out


def main(args: argparse.Namespace) -> None:
    base = Path(__file__).parent.parent.resolve() / args.input
    seeds_large = sorted(args.seeds_large)
    seeds_small = sorted(args.seeds_small)
    all_seeds = sorted(set(seeds_small) | set(seeds_large))
    groups = [group_label(env, mode) for env in args.envs for mode in args.modes]

    final_values_by_metric = {
        metric: load_all_final_values(base, args.envs, args.modes, args.algos, args.level, all_seeds, metric,
                                      args.last_frac)
        for metric in args.metrics
    }

    for metric, final_values in final_values_by_metric.items():
        missing_lines = missing_seed_report(final_values, groups, args.algos, all_seeds)
        if missing_lines:
            print(f"\nMissing runs (crashed or absent), {len(missing_lines)} group/algo combos affected ({metric}):")
            print("\n".join(missing_lines))

    rows = build_table(final_values_by_metric, groups, args.algos, args.metrics,
                       seeds_small, seeds_large, method=args.ci_method)
    headers = [
        "Env (Mode)",
        "avail_small(min-max/req)", "CI95%(n_small)",
        "avail_large(min-max/req)", "CI95%(n_large)",
        "CI reduction %",
    ]
    ci_desc = "1.96 * population std / sqrt(n) [matches other plotting scripts]" if args.ci_method == "normal" \
        else "Student's t(n-1) * sample std / sqrt(n)"
    print(f"\nMetrics: {args.metrics} (pooled into a single CI per group)  |  final-perf window: last "
          f"{args.last_frac:.0%} of steps  |  averaged across algos: {args.algos}  |  CI method: {ci_desc}\n")
    print(format_table(rows, headers))

    if rows:
        for mode in args.modes:
            mode_label = MODE_LABELS[mode]
            mode_reductions = [r[-1] for r in rows if r[0].endswith(f"({mode_label})")]
            if mode_reductions:
                print(f"Average CI reduction ({mode_label}) going from {len(seeds_small)} to "
                      f"{len(seeds_large)} seeds: {np.mean(mode_reductions):.1f}%")

    trend = build_trend(final_values_by_metric, groups, args.algos, args.metrics, seeds_large,
                        method=args.ci_method)
    out_path = Path(args.output_fig_dir) / f"{args.out_name}.pdf"
    plot_results(rows, trend, groups, seeds_small, seeds_large, out_path)
    print(f"Saved figure: {out_path}")


def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare CI/variance of final performance across seed-set sizes, "
                                            "for curriculum and transfer training.")
    p.add_argument("--input", type=str, default="data")
    p.add_argument("--envs", type=str, nargs="+",
                   default=["safe_point_goal", "safe_reacher", "safe_walker", "safe_height"],
                   help="Env identifiers as they appear under results/data/{curriculum,transfer}/.")
    p.add_argument("--modes", type=str, nargs="+", default=list(MODES), choices=list(MODES))
    p.add_argument("--algos", type=str, nargs="+",
                   default=["ppo_lag", "ppo_pid", "p3o", "focops"])
    p.add_argument("--level", type=int, default=3, help="Target difficulty level (transfer data is stored per-level).")
    p.add_argument("--metrics", type=str, nargs="+", default=["reward", "cost"], choices=list(METRIC_COLS.keys()))
    p.add_argument("--seeds_small", type=int, nargs="+", default=[*range(1, 6)])
    p.add_argument("--seeds_large", type=int, nargs="+", default=[*range(1, 21)])
    p.add_argument("--last_frac", type=float, default=0.1,
                   help="Fraction of the tail of each run averaged to obtain the per-seed final performance.")
    p.add_argument("--ci_method", type=str, default="t", choices=["normal", "t"],
                   help="'normal' (default): 1.96 * population std / sqrt(n). 't': Student's t(n-1) * sample "
                        "std / sqrt(n), more accurate at small n.")
    p.add_argument("--output_fig_dir", type=str, default="figures")
    p.add_argument("--out_name", type=str, default="seed_variance_curriculum_transfer")
    return p


if __name__ == "__main__":
    args = build_args().parse_args()
    main(args)
