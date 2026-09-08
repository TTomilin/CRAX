"""Download WandB runs for the core PPO/MJX hyperparameter sweep.

Each algo varies one core PPO optimization hyperparameter (learning_rate,
entropy_cost, discounting, gae_lambda, clipping_epsilon) at a time, others
left at train_env.py's default. Key the output folder by the swept hyperparameter
name so it slots into the existing data/<env>/level_<level>/<algo>/... tree:

    data/<env>/level_<level>/<algo>/<hparam>_<value>/seed_<n>.parquet
"""
import argparse
import copy

import wandb

from results import cli
from results.common import RL_SWEEP_SPEC
from results.download.main_results import build_filters, store_data


def main(args: argparse.Namespace) -> None:
    api = wandb.Api()

    algos = args.algos or list(RL_SWEEP_SPEC.keys())
    for algo in algos:
        hparams = RL_SWEEP_SPEC.get(algo)
        if hparams is None:
            print(f"Skipping '{algo}': no Stage 1 sweep spec defined for it.")
            continue

        for hparam in hparams:
            print(f"\n=== Downloading {algo} / sweep over {hparam} ===")
            run_args = copy.deepcopy(args)
            run_args.algos = [algo]
            run_args.extra_attribute = hparam

            filters = build_filters(run_args)
            runs = api.runs(args.project, filters=filters, order="-created_at", per_page=200)

            n = 0
            for run in runs:
                store_data(run, run_args)
                n += 1
            print(f"Found {n} runs for {algo}/{hparam}")


def build_args() -> argparse.ArgumentParser:
    parser = cli.download_parser(
        "Download Stage 1 hyperparameter sweep results from WandB.",
        omit=("algos",),
        envs=cli.DEFAULT_PLOT_ENVS,
        levels=[1],
        max_age_days=1,
    )
    parser.add_argument("--algos", type=str, nargs='+', default=None,
                        choices=list(RL_SWEEP_SPEC.keys()),
                        help="Which methods to download (default: all methods in RL_SWEEP_SPEC)")
    return parser


if __name__ == "__main__":
    parser = build_args()
    main(parser.parse_args())
