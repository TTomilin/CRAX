"""Download counterfactual hazard relabelling runs from wandb.

Selects runs tagged COUNTERFACTUAL (submitted by `crax_counterfactual.sh` on
branch counterfactual-hazards) and stores them by relabelling mode
(`config.counterfactual_mode`: none / random / optimistic / adaptive / gated, the
last two as `<mode>_k<config.counterfactual_gate_sharpness>`) and by how the
episodic cost bound is converted to a per-step one
(`config.observed_episode_length_bound`):

    <output>/counterfactual/<env>/level_<k>/<algo>/<bound>/<mode>/seed_<n>.parquet

with <bound> = `observed_length` (`safety_bound / observed episode length`) or
`fixed_length` (`safety_bound / episode_length`, the default of every safe algo).
"""

import argparse
import os

import wandb
from wandb.apis.public import Run

from results import cli
from results.common import WANDB_ENV_NAME_ALIASES, apply_max_age_filter, env_name_variants, get_metrics_for_env, \
    results_path

BOUND_FOLDERS = {True: "observed_length", False: "fixed_length"}


def main(args: argparse.Namespace) -> None:
    api = wandb.Api()
    runs = api.runs(args.project, filters=build_filters(args), order="-created_at", per_page=200)
    for run in runs:
        store_data(run, args)


def build_filters(args: argparse.Namespace) -> dict:
    """Server-side filters for wandb.Api().runs()."""
    f = {"state": {"$in": list(args.states)}, "tags": {"$in": args.wandb_tags}}
    if args.algos:
        f["config.alg"] = {"$in": args.algos}
    if args.envs:
        f["config.env_name"] = {"$in": sorted({v for env in args.envs for v in env_name_variants(env)})}
    if args.levels:
        f["config.difficulty"] = {"$in": args.levels}
    if args.modes:
        f["config.counterfactual_mode"] = {"$in": args.modes}
    if args.bound != "all":
        f["config.observed_episode_length_bound"] = args.bound == "observed_length"
    apply_max_age_filter(f, args)
    return f


def _seed(config: dict):
    """train_env logs the seed list (`--seeds`) rather than a single `seed`."""
    seed = config.get("seed")
    if seed is None:
        seeds = config.get("seeds") or [None]
        seed = seeds[0]
    return seed


def store_data(run: Run, args: argparse.Namespace) -> None:
    """Store run data to parquet file."""
    config = run.config
    seed = _seed(config)
    if args.seeds and seed not in args.seeds:
        return

    env = config.get("env_name")
    canonical_env = next((name for name, aliases in WANDB_ENV_NAME_ALIASES.items() if env in aliases), env)
    algo = config.get("alg", "unknown")
    mode = config.get("counterfactual_mode", "none")
    if mode in ("adaptive", "gated"):
        # k=0 is the fixed 50/50 optimistic/random mix, so the gate slope is part of the mode
        mode = f"{mode}_k{config.get('counterfactual_gate_sharpness', 10.0):g}"
    if config.get("resample_layouts", False):
        # Layout-resampling control: fresh layout per episode, with or without relabelling
        mode = "resample" if mode == "none" else f"{mode}_resample"
    bound = BOUND_FOLDERS[bool(config.get("observed_episode_length_bound", False))]
    level = config.get("difficulty")

    folder_path = results_path(args.output) / "counterfactual" / canonical_env / f"level_{level}" / algo / bound / mode
    file_path = folder_path / f"seed_{seed}.parquet"
    os.makedirs(folder_path, exist_ok=True)

    if file_path.exists() and not args.overwrite:
        print(f"Skipping existing file: {file_path}")
        return

    try:
        df = run.history(keys=get_metrics_for_env(canonical_env, args.metrics), samples=args.samples)
        if df is None or df.empty:
            print(f"No history for run {run.id}")
            return
        if "_step" not in df.columns:
            df["_step"] = range(len(df))
        df.to_parquet(file_path)
        print(f"Saved data for run {run.id} ({algo}, {bound}, {mode}, seed {seed}) to {file_path}")
    except Exception as e:
        print(f"Error downloading data for run {run.id}: {e}")


def build_args() -> argparse.ArgumentParser:
    parser = cli.download_parser(
        "Download counterfactual relabelling results from wandb",
        algos=["ppo_lag", "ppo_pid"],
        envs=["safe_pathway_walker2d"],
        levels=[3],
        wandb_tags=["COUNTERFACTUAL"],
    )
    parser.add_argument("--modes", type=str, nargs="+", default=["none", "random", "optimistic", "adaptive", "gated"],
                        help="Counterfactual relabelling modes to download")
    parser.add_argument("--bound", type=str, default="all", choices=["all", *BOUND_FOLDERS.values()],
                        help="Per-step bound conversion of the runs to download")
    parser.add_argument("--samples", type=int, default=2000,
                        help="History points per run (wandb subsamples evenly)")
    return parser


if __name__ == "__main__":
    main(build_args().parse_args())
