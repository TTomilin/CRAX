"""Download OmniSafe runs from WandB.

Results are uploaded to WandB via omnisafe's own logger
(logger_cfgs.use_wandb=True), using the same keys as its local
progress.csv: `TotalEnvSteps`, `Metrics/EpRet`, `Metrics/EpCost`.

Output matches what `results/plotting_results/omnisafe_vs_crax_ant_velocity.py`
expects: `<output>/seed_{seed}.csv` with columns (step, reward, cost, seed).

Usage:
    python -m results.download_omnisafe_ant_velocity \\
        --project omnisafe --seeds 1 2 3 4 5 \\
        --output results/data/omnisafe_ant_velocity
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import wandb

from results import cli
from results.common import apply_max_age_filter

STEP_KEY = "TotalEnvSteps"
REWARD_KEY = "Metrics/EpRet"
COST_KEY = "Metrics/EpCost"


def build_filters(args: argparse.Namespace) -> dict:
    f = {"state": "finished"}
    if args.seeds:
        f["config.seed"] = {"$in": args.seeds}
    if args.wandb_tags:
        f["tags"] = {"$in": args.wandb_tags}
    # only runs created within the last `max_age_days` days
    apply_max_age_filter(f, args)
    return f


def main(args: argparse.Namespace) -> None:
    api = wandb.Api()
    filters = build_filters(args)
    runs = api.runs(args.project, filters=filters, order="-created_at", per_page=200)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    seen_seeds = set()
    n = 0
    for run in runs:
        if args.run_name_contains and args.run_name_contains not in run.name:
            print(f"Skipping run '{run.name}' (doesn't contain '{args.run_name_contains}')")
            continue

        seed = run.config.get("seed")
        if seed is None:
            print(f"Skipping run '{run.name}' (id={run.id}): no 'seed' in config")
            continue
        if seed in seen_seeds:
            print(f"Skipping run '{run.name}' (id={run.id}): seed {seed} already downloaded "
                  f"(pass --overwrite/rerun with only the runs you want if this is wrong)")
            continue

        file_path = output_dir / f"seed_{seed}.csv"
        if file_path.exists() and not args.overwrite:
            print(f"Skipping existing file: {file_path}")
            seen_seeds.add(seed)
            continue

        df = run.history(keys=[STEP_KEY, REWARD_KEY, COST_KEY])
        if df is None or df.empty or STEP_KEY not in df.columns:
            print(f"Skipping run '{run.name}' (id={run.id}): missing {STEP_KEY}/{REWARD_KEY}/{COST_KEY} "
                  f"(columns found: {list(df.columns) if df is not None else None})")
            continue

        out = pd.DataFrame({
            "step": df[STEP_KEY],
            "reward": df[REWARD_KEY],
            "cost": df[COST_KEY],
            "seed": seed,
        }).dropna(subset=["step", "reward", "cost"]).sort_values("step")

        out.to_csv(file_path, index=False)
        print(f"Saved run '{run.name}' (id={run.id}, seed={seed}) to {file_path}")
        seen_seeds.add(seed)
        n += 1

    print(f"\nDownloaded {n} run(s) to {output_dir}/")
    missing = [s for s in args.seeds if s not in seen_seeds]
    if missing:
        print(f"WARNING: no run found for seeds {missing}")


def build_args() -> argparse.ArgumentParser:
    p = cli.download_parser(
        "Download OmniSafe ant-velocity (PPOLag) runs from WandB.",
        level_arg=None,
        project_default="omnisafe",
        omit=("envs", "algos", "metrics", "include_runs"),
        output="results/data/omnisafe_ant_velocity",
    )
    p.add_argument("--run_name_contains", type=str, default="AntVelocity",
                   help="Only keep runs whose name contains this substring (client-side filter; "
                        "set to '' to disable). Guards against pulling unrelated runs from the project.")
    return p


if __name__ == "__main__":
    main(build_args().parse_args())
