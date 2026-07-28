"""Analyze the Safe-RL-method-specific hyperparameter sweep.

For each (algo, hyperparameter) pair in HPARAM_SWEEP_SPEC, compares final
performance (reward and cost) across the swept values, per environment kept
separate (reward/cost magnitudes differ too much across envs to average).
Prints one table and saves one figure per (algo, hyperparameter).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from results.common import HPARAM_SWEEP_SPEC, TRANSLATIONS, set_mpl_style
from results.plotting_results.seed_variance import ci95, format_table, load_final_value


def discover_values(base: Path, env: str, level: int, algo: str, hparam: str) -> List[str]:
    """Return sorted (numerically) folder-name value suffixes for a swept hyperparameter."""
    algo_dir = base / env / f"level_{level}" / algo
    if not algo_dir.is_dir():
        return []
    prefix = f"{hparam}_"
    values = [p.name[len(prefix):] for p in algo_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    return sorted(values, key=lambda v: float(v))


def load_sweep_data(
        base: Path, envs: List[str], level: int, algo: str, hparam: str,
        values: List[str], seeds: List[int], metric: str, last_frac: float,
) -> Dict[Tuple[str, str], List[float]]:
    """dict[(env, value)] -> list of per-seed final-performance scalars."""
    out: Dict[Tuple[str, str], List[float]] = {}
    for env in envs:
        for value in values:
            vals = []
            for seed in seeds:
                fp = base / env / f"level_{level}" / algo / f"{hparam}_{value}" / f"seed_{seed}.parquet"
                v = load_final_value(fp, algo, env, metric, last_frac)
                if v is not None:
                    vals.append(v)
            out[(env, value)] = vals
    return out


def build_table(
        reward_data: Dict[Tuple[str, str], List[float]],
        cost_data: Dict[Tuple[str, str], List[float]],
        envs: List[str], values: List[str], method: str = "normal",
) -> Tuple[List[List], List[str]]:
    headers = ["Value"]
    for env in envs:
        env_label = TRANSLATIONS.get(env, env)
        headers += [f"{env_label} Reward", f"{env_label} Cost"]

    rows = []
    for value in values:
        row = [value]
        for env in envs:
            for data in (reward_data, cost_data):
                vals = np.array(data.get((env, value), []), dtype=np.float64)
                if len(vals) == 0:
                    row.append("n/a")
                elif len(vals) == 1:
                    row.append(f"{vals[0]:.2f} (n=1)")
                else:
                    mean, _, ci = ci95(vals, method=method)
                    row.append(f"{mean:.2f} ± {ci:.2f}")
        rows.append(row)
    return rows, headers


def plot_sweep(
        reward_data: Dict[Tuple[str, str], List[float]],
        cost_data: Dict[Tuple[str, str], List[float]],
        envs: List[str], values: List[str], algo: str, hparam: str, out_path: Path,
        method: str = "normal",
) -> None:
    set_mpl_style()
    fig, axes = plt.subplots(1, len(envs), figsize=(4.5 * len(envs), 4), squeeze=False)
    axes = axes[0]

    x = np.arange(len(values))
    for ax, env in zip(axes, envs):
        reward_means, reward_cis = [], []
        cost_means, cost_cis = [], []
        for value in values:
            r_vals = np.array(reward_data.get((env, value), []), dtype=np.float64)
            c_vals = np.array(cost_data.get((env, value), []), dtype=np.float64)
            if len(r_vals) >= 2:
                m, _, ci = ci95(r_vals, method=method)
            else:
                m, ci = (r_vals[0] if len(r_vals) else np.nan), 0.0
            reward_means.append(m)
            reward_cis.append(ci)
            if len(c_vals) >= 2:
                m, _, ci = ci95(c_vals, method=method)
            else:
                m, ci = (c_vals[0] if len(c_vals) else np.nan), 0.0
            cost_means.append(m)
            cost_cis.append(ci)

        ax_cost = ax.twinx()
        ax.errorbar(x, reward_means, yerr=reward_cis, marker="o", color="tab:blue",
                    label="Reward", capsize=3)
        ax_cost.errorbar(x, cost_means, yerr=cost_cis, marker="s", color="tab:red",
                          linestyle="--", label="Cost", capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(values)
        ax.set_xlabel(TRANSLATIONS.get(hparam, hparam))
        ax.set_ylabel("Reward", color="tab:blue")
        ax_cost.set_ylabel("Cost", color="tab:red")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax_cost.tick_params(axis="y", labelcolor="tab:red")
        ax.set_title(TRANSLATIONS.get(env, env))
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(f"{TRANSLATIONS.get(algo, algo)}: sweep over {TRANSLATIONS.get(hparam, hparam)}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")


def main(args: argparse.Namespace) -> None:
    base = Path(__file__).parent.parent.resolve() / args.input

    algos = args.algos or list(HPARAM_SWEEP_SPEC.keys())
    for algo in algos:
        hparams = HPARAM_SWEEP_SPEC.get(algo)
        if hparams is None:
            print(f"Skipping '{algo}': no hyperparameter sweep spec defined for it.")
            continue

        for hparam in hparams:
            values = sorted(
                set().union(*[discover_values(base, env, args.level, algo, hparam) for env in args.envs]),
                key=lambda v: float(v),
            )
            if not values:
                print(f"\nNo data found for {algo}/{hparam}, skipping.")
                continue

            reward_data = load_sweep_data(base, args.envs, args.level, algo, hparam, values,
                                           args.seeds, "reward", args.last_frac)
            cost_data = load_sweep_data(base, args.envs, args.level, algo, hparam, values,
                                         args.seeds, "cost", args.last_frac)

            rows, headers = build_table(reward_data, cost_data, args.envs, values, method=args.ci_method)
            print(f"\n=== {algo} / sweep over {hparam} (values: {values}) ===")
            print(format_table(rows, headers))

            out_path = Path(args.output_fig_dir) / f"hparam_sweep_{algo}_{hparam}.pdf"
            plot_sweep(reward_data, cost_data, args.envs, values, algo, hparam, out_path,
                       method=args.ci_method)
            print(f"Saved figure: {out_path}")
            plt.show()


def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze the Safe-RL-method hyperparameter sweep.")
    p.add_argument("--input", type=str, default="data")
    p.add_argument("--envs", type=str, nargs="+",
                   default=["safe_goal_point", "safe_reacher", "safe_push_point"])
    p.add_argument("--algos", type=str, nargs="+", default=None,
                   choices=list(HPARAM_SWEEP_SPEC.keys()),
                   help="Which methods to analyze (default: all methods in HPARAM_SWEEP_SPEC)")
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--last_frac", type=float, default=0.1,
                   help="Fraction of the tail of each run averaged for final performance.")
    p.add_argument("--ci_method", type=str, default="normal", choices=["normal", "t"])
    p.add_argument("--output_fig_dir", type=str, default="figures")
    return p


if __name__ == "__main__":
    args = build_args().parse_args()
    main(args)
