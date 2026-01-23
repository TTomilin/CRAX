"""Download transfer and curriculum experiment results from wandb.

This script downloads results tagged with TRANSFER or CURRICULUM from wandb.

For TRANSFER runs:
  - Phase 1: Unsafe PPO pre-training
  - Phase 2: Safe algorithm fine-tuning
  Data is stored as: <output>/transfer/<env>/level_<k>/<safe_algo>/seed_<n>.parquet

For CURRICULUM runs:
  - Sequential training across difficulty levels (1->2->3)
  - global_step tracks cumulative steps across all stages
  Data is stored as: <output>/curriculum/<env>/<algo>/seed_<n>.parquet
"""

import argparse
import os
from pathlib import Path

import wandb
from wandb.apis.public import Run

# Defines the primary reward metric for specific environments.
# If an environment is not listed here, it defaults to 'episodic/sum_reward'.
REWARD_METRIC_MAP = {
    'safe_walker': 'episodic/reward_forward',
    'safe_velocity': 'episodic/forward_reward',
}
DEFAULT_REWARD_METRIC = 'episodic/sum_reward'


def main(args: argparse.Namespace) -> None:
    api = wandb.Api()
    filters = build_filters(args)
    runs = api.runs(args.project, filters=filters, order="-created_at", per_page=200)

    for run in runs:
        store_data(run, args)


def build_filters(args: argparse.Namespace) -> dict:
    """Server-side filters for wandb.Api().runs()."""
    f = {"state": "finished"}

    # Algorithm filtering - need to include 'ppo' for transfer unsafe phase
    algos_to_filter = list(args.algos) if args.algos else []
    if args.include_unsafe_phase and 'ppo' not in algos_to_filter:
        algos_to_filter.append('ppo')

    if algos_to_filter:
        # For curriculum, the algo is stored in config.alg
        # For transfer, we filter by config.algorithm
        f["config.alg"] = {"$in": algos_to_filter}
    if args.envs:
        f["config.env_name"] = {"$in": args.envs}
    if args.seeds:
        f["config.seed"] = {"$in": args.seeds}

    # Filter by tags (TRANSFER or CURRICULUM)
    if args.wandb_tags:
        f["tags"] = {"$in": args.wandb_tags}

    if args.include_runs:
        ors = [{"display_name": {"$in": args.include_runs}}]
        f = {"$or": [f, *ors]}

    return f


def store_data(run: Run, args: argparse.Namespace) -> None:
    """Store run data to parquet file."""
    metrics = list(args.metrics)
    config = run.config
    run_id = run.id
    tags = run.tags

    seed = config.get('seed')
    env = config.get('env_name')

    # Automatically select the correct reward metric based on the environment
    if DEFAULT_REWARD_METRIC in metrics:
        reward_metric = REWARD_METRIC_MAP.get(env, DEFAULT_REWARD_METRIC)
        metrics[metrics.index(DEFAULT_REWARD_METRIC)] = reward_metric

    # Determine if this is a transfer or curriculum run
    is_transfer = 'TRANSFER' in tags
    is_curriculum = 'CURRICULUM' in tags

    if not is_transfer and not is_curriculum:
        print(f"Skipping run {run_id}: not tagged as TRANSFER or CURRICULUM")
        return

    root_dir = Path(__file__).parent.resolve()

    if is_curriculum:
        # Curriculum run: algo stored in config.alg
        algo = config.get('alg', config.get('algorithm', 'unknown'))
        folder_path = root_dir / args.output / 'curriculum' / env / algo
        file_path = folder_path / f"seed_{seed}.parquet"
        key_metrics = metrics + ['global_step', 'curriculum_stage']
    else:
        # Transfer run: has phase info
        algo = config.get('algorithm', config.get('alg', 'unknown'))
        phase = config.get('phase', 'unknown')
        difficulty = config.get('difficulty', 3)  # Transfer typically targets level 3

        if phase == 'unsafe':
            # Skip unsafe phase runs - we only care about the safe fine-tuning results
            # But we can optionally store them
            if not args.include_unsafe_phase:
                print(f"Skipping unsafe phase run {run_id}")
                return
            folder_path = root_dir / args.output / 'transfer' / env / f"level_{difficulty}" / 'ppo_pretrain'
        else:
            folder_path = root_dir / args.output / 'transfer' / env / f"level_{difficulty}" / algo

        file_path = folder_path / f"seed_{seed}.parquet"
        key_metrics = metrics

    os.makedirs(folder_path, exist_ok=True)

    if file_path.exists() and not args.overwrite:
        print(f"Skipping existing file: {file_path}")
        return

    try:
        df = run.history(keys=key_metrics)
        if df is None or df.empty:
            print(f"No history for run {run_id}")
            return

        # Ensure _step column exists
        if '_step' not in df.columns:
            df['_step'] = range(len(df))

        df.to_parquet(file_path)
        print(f"Saved data for run {run_id} ({algo}, seed {seed}) to {file_path}")
    except Exception as e:
        print(f"Error downloading data for run {run_id}: {e}")


def build_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download transfer/curriculum results from wandb")
    parser.add_argument("--seeds", type=int, nargs='+', default=[1, 2, 3, 4, 5],
                        help="Seed(s) of the run(s) to download")
    parser.add_argument("--algos", type=str, nargs='+',
                        default=["ppo", "ppo_lag", "ppo_pid", "focops", "p3o"],
                        help="Algorithms to download")
    parser.add_argument("--envs", type=str, nargs='+', help="Environments to download")
    parser.add_argument("--output", type=str, default='data', help="Base output directory")
    parser.add_argument("--metrics", type=str, nargs='+',
                        default=['episodic/sum_reward', 'episodic/cost'],
                        help="Metrics to download")
    parser.add_argument("--project", type=str, required=True, help="WandB project name")
    parser.add_argument("--wandb_tags", type=str, nargs='+', default=['TRANSFER', 'CURRICULUM'],
                        help="WandB tags to filter runs (TRANSFER and/or CURRICULUM)")
    parser.add_argument("--overwrite", default=False, action='store_true',
                        help="Overwrite existing files")
    parser.add_argument("--include_runs", type=str, nargs="+", default=[],
                        help="Include specific runs by display name")
    parser.add_argument("--include_unsafe_phase", default=True, action='store_true',
                        help="Download unsafe PPO phase for transfer runs (default: True)")
    parser.add_argument("--no_unsafe_phase", dest='include_unsafe_phase', action='store_false',
                        help="Skip downloading unsafe PPO phase for transfer runs")
    return parser


if __name__ == "__main__":
    parser = build_args()
    main(parser.parse_args())
