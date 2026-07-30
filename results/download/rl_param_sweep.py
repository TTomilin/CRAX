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
    parser = argparse.ArgumentParser(description="Download Stage 1 hyperparameter sweep results from WandB.")
    parser.add_argument("--seeds", type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        help="Seed(s) of the run(s) to download")
    parser.add_argument("--algos", type=str, nargs='+', default=None,
                        choices=list(RL_SWEEP_SPEC.keys()),
                        help="Which methods to download (default: all methods in STAGE1_SWEEP_SPEC)")
    parser.add_argument("--envs", type=str, nargs='+',
                        default=["safe_goal_point", "safe_reacher", "safe_push_point"],
                        help="Environments to download")
    parser.add_argument("--levels", type=int, nargs='+', default=[1], help="Levels to download")
    parser.add_argument("--output", type=str, default='data', help="Base output directory to store the data")
    parser.add_argument("--metrics", type=str, nargs='+', default=['episodic/sum_reward', 'episodic/cost'],
                        help="Name of the metrics to download")
    parser.add_argument("--project", type=str, required=True, help="Name of the WandB project")
    parser.add_argument("--max_age_days", type=float, default=1,
                        help="Only download runs created at most this many days ago.")
    parser.add_argument("--wandb_tags", type=str, nargs='+', default=[], help="WandB tags to filter runs")
    parser.add_argument("--overwrite", default=False, action='store_true', help="Overwrite existing files")
    parser.add_argument("--include_runs", type=str, nargs="+", default=[],
                        help="List of runs that shouldn't be filtered out")
    return parser


if __name__ == "__main__":
    parser = build_args()
    main(parser.parse_args())
