import argparse
import os
from pathlib import Path

import wandb
from wandb.apis.public import Run

from results import cli
from results.common import apply_max_age_filter, get_metrics_for_env


def main(args: argparse.Namespace) -> None:
    api = wandb.Api()
    filters = build_filters(args)
    runs = api.runs(args.project, filters=filters, order="-created_at", per_page=200)

    for run in runs:
        store_data(run, args)


def build_filters(args: argparse.Namespace) -> dict:
    """Server-side filters for wandb.Api().runs()."""
    f = {"state": {"$in": list(args.states)}}

    # config.* filters
    if args.algos:
        f["config.alg"] = {"$in": args.algos}
    if args.envs:
        f["config.env_name"] = {"$in": args.envs}
    if args.levels:
        f["config.difficulty"] = {"$in": args.levels}
    if args.seeds:
        f["config.seed"] = {"$in": args.seeds}

    # only runs created within the last `max_age_days` days
    apply_max_age_filter(f, args)

    # tags live on the run, not in config
    if args.wandb_tags:
        f["tags"] = {"$in": args.wandb_tags}

    # include specific runs by name (display_name) as an OR clause
    # if include_runs is set, we *add* them even if they don't match other filters
    if args.include_runs:
        ors = [{"display_name": {"$in": args.include_runs}}]
        # $or coexists with the ANDed top-level filters:
        f = {"$or": [f, *ors]}

    return f


def store_data(run: Run, args: argparse.Namespace) -> None:
    config = run.config
    run_id = run.id
    seed = config['seed']
    env = config['env_name']
    level = config['difficulty']
    algo = config['alg']
    extra_attribute = ''

    metrics = get_metrics_for_env(env, args.metrics)

    attribute_key = args.extra_attribute
    if attribute_key:
        if attribute_key not in config:
            raise ValueError(f"Extra attribute_key '{attribute_key}' not found in run config.")
        attribute_val = config[attribute_key]
        extra_attribute = f"{attribute_key}_{attribute_val}"

    # Construct folder path for each configuration
    root_dir = Path(__file__).parent.parent.resolve()
    folder_path = root_dir / args.output / env / f"level_{level}" / algo / extra_attribute
    os.makedirs(folder_path, exist_ok=True)  # Ensure the directory exists

    file_path = folder_path / f"seed_{seed}.parquet"

    # Skip if file already exists and overwrite flag not set
    if file_path.exists() and not args.overwrite:
        print(f"Skipping existing file: {file_path}")
        return

    try:
        df = run.history(keys=metrics)
        if df is None or df.empty:
            print(f"No history for run {run_id}")
            return
        df.to_parquet(file_path)
        print(f"Saved data for run {run_id} to {file_path}")
    except Exception as e:
        print(f"Error downloading data for run: {run_id}: {e}")
        return


def common_dl_args() -> argparse.ArgumentParser:
    """Parser shared by the download scripts that reuse `build_filters`/`store_data`."""
    parser = cli.download_parser("Download benchmark results from WandB.")
    parser.add_argument("--extra_attribute", type=str, default=None,
                        help="Config attribute to store data by")
    return parser


if __name__ == "__main__":
    parser = common_dl_args()
    main(parser.parse_args())
