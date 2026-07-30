"""Analyze the core PPO/MJX hyperparameter sweep.

For each (algo, hyperparameter) pair, compares final
performance (reward and cost) across the swept values, per environment kept
separate (reward/cost magnitudes differ too much across envs to average).
Prints one table and saves one figure per (algo, hyperparameter).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from results.common import RL_SWEEP_SPEC
from results.plotting_results.safety_param_sweep import build_table, discover_values, load_sweep_data, plot_sweep
from results.plotting_results.seed_variance import format_table


def main(args: argparse.Namespace) -> None:
    base = Path(__file__).parent.parent.resolve() / args.input

    algos = args.algos or list(RL_SWEEP_SPEC.keys())
    for algo in algos:
        hparams = RL_SWEEP_SPEC.get(algo)
        if hparams is None:
            print(f"Skipping '{algo}': no Stage 1 sweep spec defined for it.")
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

            out_path = Path(args.output_fig_dir) / f"stage1_sweep_{algo}_{hparam}.pdf"
            plot_sweep(reward_data, cost_data, args.envs, values, algo, hparam, out_path,
                       method=args.ci_method)
            print(f"Saved figure: {out_path}")
            plt.show()


def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze the Stage 1 core PPO/MJX hyperparameter sweep.")
    p.add_argument("--input", type=str, default="data")
    p.add_argument("--envs", type=str, nargs="+",
                   default=["safe_goal_point", "safe_reacher", "safe_push_point"])
    p.add_argument("--algos", type=str, nargs="+", default=None,
                   choices=list(RL_SWEEP_SPEC.keys()),
                   help="Which methods to analyze (default: all methods in STAGE1_SWEEP_SPEC)")
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--seeds", type=int, nargs="+", default=[*range(1, 6)])
    p.add_argument("--last_frac", type=float, default=0.1,
                   help="Fraction of the tail of each run averaged for final performance.")
    p.add_argument("--ci_method", type=str, default="normal", choices=["normal", "t"])
    p.add_argument("--output_fig_dir", type=str, default="figures")
    return p


if __name__ == "__main__":
    args = build_args().parse_args()
    main(args)
