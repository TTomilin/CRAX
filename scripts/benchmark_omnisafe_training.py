#!/usr/bin/env python3
"""Training SPS benchmark for OmniSafe + Safety-Gymnasium (CPU).

Runs PPOLag training with varying numbers of parallel environments and records
the Steps-Per-Second reported by the training loop (Time/FPS in the omnisafe
logger), i.e. throughput including gradient updates.

Usage:
    # Default: all envs that have a CRAX counterpart, num_envs=[1,2,4,8,16,32]
    conda run -n omnisafe python scripts/benchmark_omnisafe_training.py

    # Single env, custom num_envs sweep
    conda run -n omnisafe python scripts/benchmark_omnisafe_training.py \\
        --envs SafetyPointGoal1-v0 \\
        --num-envs 1 4 16 32
"""

from __future__ import annotations

import os
import sys
import types

import tensorflow as _tf_ns  # loads the (empty) namespace package

_tf_io = types.ModuleType("tensorflow.io")
_tf_gfile = types.SimpleNamespace(join=os.path.join, get_filesystem=None)
_tf_io.gfile = _tf_gfile
_tf_ns.io = _tf_io
sys.modules["tensorflow.io"] = _tf_io
# ---------------------------------------------------------------------------

import argparse
import csv
import shutil
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

# ---------------------------------------------------------------------------
# Default environment list: Safety Gymnasium envs that have a CRAX counterpart
#
# CRAX env               → Safety Gymnasium env
# safe_goal_point        → SafetyPointGoal1-v0
# safe_circle_point      → SafetyPointCircle1-v0
# safe_button_point      → SafetyPointButton1-v0
# safe_push_point        → SafetyPointPush1-v0
# safe_velocity_ant      → SafetyAntVelocity-v1
# safe_velocity_halfcheetah → SafetyHalfCheetahVelocity-v1
# safe_velocity_hopper   → SafetyHopperVelocity-v1
# safe_velocity_humanoid → SafetyHumanoidVelocity-v1
# safe_velocity_swimmer  → SafetySwimmerVelocity-v1
# safe_velocity_walker2d → SafetyWalker2dVelocity-v1
#
# No SG match: safe_pathway_walker2d, safe_height_humanoid, safe_lift_*,
#              safe_lift_spider, safe_reacher
# ---------------------------------------------------------------------------
DEFAULT_ENVS: List[str] = [
    "SafetyPointGoal1-v0",
    "SafetyPointCircle1-v0",
    "SafetyPointPush1-v0",
    "SafetyHumanoidVelocity-v1",
]

# steps_per_epoch is computed per-run as STEPS_PER_ENV * num_envs, ensuring
# each env gets at least one full episode per epoch (Safety Gymnasium default
# episode length = 1000 steps). A fixed global constant would make
# steps_per_env < episode_length at high num_envs, causing trajectory cut-offs
# and NaN costs in the Lagrangian update. To keep wall time comparable across
# num_envs, total_steps is fixed. num_epochs is computed as max(TOTAL_STEPS // steps_per_epoch,
# MIN_EPOCHS). Wall time will decrease as num_envs grows up to the break-even
# point (TOTAL_STEPS / (MIN_EPOCHS * STEPS_PER_ENV)), then stays flat because
# the MIN_EPOCHS floor keeps a minimum amount of training data.
STEPS_PER_ENV = 1_000  # steps per environment per epoch (≥ episode length)
TOTAL_STEPS = 60_000  # fixed budget; wall time decreases with more envs up to break-even
MIN_EPOCHS = 5  # need at least 1 warmup + 4 stable SPS samples

DEFAULT_NUM_ENVS: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256]


# ---------------------------------------------------------------------------

def benchmark_run(env_name: str, num_envs: int, scratch_dir: Path) -> Dict:
    """Run one PPOLag training run and return per-epoch SPS statistics."""
    import omnisafe  # deferred so the TF patch above is already applied

    steps_per_epoch = STEPS_PER_ENV * num_envs
    num_epochs = max(TOTAL_STEPS // steps_per_epoch, MIN_EPOCHS)
    total_steps = steps_per_epoch * num_epochs

    log_dir = scratch_dir / f"{env_name}_envs{num_envs}"
    log_dir.mkdir(parents=True, exist_ok=True)

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024

    t0 = time.time()

    agent = omnisafe.Agent(
        "PPOLag",
        env_name,
        custom_cfgs={
            "train_cfgs": {
                "vector_env_nums": num_envs,
                "total_steps": total_steps,
                "device": "cpu",
            },
            "algo_cfgs": {
                "steps_per_epoch": steps_per_epoch,
            },
            "logger_cfgs": {
                "use_tensorboard": False,
                "use_wandb": False,
                "log_dir": str(log_dir),
            },
        },
    )
    agent.learn()

    wall_time = time.time() - t0
    mem_after = process.memory_info().rss / 1024 / 1024
    mem_used = max(0.0, mem_after - mem_before)

    progress_files = list(log_dir.rglob("progress.csv"))
    if not progress_files:
        print("  WARNING: progress.csv not found - no SPS data")
        return {}

    df = pd.read_csv(progress_files[0])
    if "Time/FPS" not in df.columns:
        print(f"  WARNING: Time/FPS column missing. Columns: {df.columns.tolist()}")
        return {}

    sps_samples: List[float] = df["Time/FPS"].tolist()
    if not sps_samples:
        return {}

    # Discard the first epoch (initialisation / cache warmup overhead)
    stable = sps_samples[1:] if len(sps_samples) > 1 else sps_samples

    result = {
        "env": env_name,
        "num_envs": num_envs,
        "sps_mean": float(np.mean(stable)),
        "sps_median": float(np.median(stable)),
        "sps_max": float(np.max(stable)),
        "sps_min": float(np.min(stable)),
        "sps_first": sps_samples[0],
        "num_samples": len(sps_samples),
        "wall_time_s": wall_time,
        "cpu_memory_mb": mem_used,
    }

    print(f"  SPS (stable mean):  {result['sps_mean']:,.0f}")
    print(f"  SPS (stable max):   {result['sps_max']:,.0f}")
    print(f"  Wall time:          {wall_time:.1f}s")

    shutil.rmtree(log_dir, ignore_errors=True)
    return result


def plot_results(results: List[Dict], env_names: List[str], num_envs_list: List[int], output_dir: Path) -> None:
    if not results:
        return

    plt.style.use("seaborn-v0_8-paper")

    # One row per env: throughput | scaling efficiency
    n_envs = len(env_names)
    fig, axes = plt.subplots(n_envs, 2, figsize=(12, 4 * n_envs), squeeze=False)

    for row, env_name in enumerate(env_names):
        env_results = [r for r in results if r["env"] == env_name]
        if not env_results:
            continue

        x = [r["num_envs"] for r in env_results]
        y_mean = [r["sps_mean"] for r in env_results]
        y_max = [r["sps_max"] for r in env_results]
        y_first = [r["sps_first"] for r in env_results]

        ax = axes[row, 0]
        ax.plot(x, y_mean, "o-", label="Stable mean", linewidth=2, markersize=7)
        ax.plot(x, y_max, "s--", label="Stable max", linewidth=1.5, markersize=5, alpha=0.7)
        ax.plot(x, y_first, "^:", label="First epoch (warmup)", linewidth=1.5, markersize=5, alpha=0.6)
        ax.set_xlabel("Number of Parallel Environments")
        ax.set_ylabel("Steps Per Second")
        ax.set_title(env_name)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        ax = axes[row, 1]
        if len(env_results) > 1:
            baseline = env_results[0]["sps_mean"]
            speedup = [r["sps_mean"] / baseline for r in env_results]
            ideal = [r["num_envs"] / env_results[0]["num_envs"] for r in env_results]
            ax.plot(x, speedup, "o-", label="Actual speedup", linewidth=2, markersize=7)
            ax.plot(x, ideal, "k--", alpha=0.4, label="Ideal (linear)")
        ax.set_xlabel("Number of Parallel Environments")
        ax.set_ylabel("Speedup over 1-env baseline")
        ax.set_title(f"{env_name} — Scaling")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle(
        f"OmniSafe PPOLag Training Throughput  |  steps_per_env={STEPS_PER_ENV:,}  total_steps={TOTAL_STEPS:,}",
        fontsize=13,
    )
    plt.tight_layout()

    for suffix in ("png", "pdf"):
        path = output_dir / f"omnisafe_training_benchmark.{suffix}"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.close()


def generate_latex_table(results: List[Dict], env_names: List[str], output_dir: Path) -> None:
    if not results:
        return

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{OmniSafe PPOLag training throughput (SPS during training)}",
        r"\label{tab:omnisafe_training_benchmark}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Environment & Envs & Mean SPS & Max SPS & First-epoch SPS & Wall time (s) \\",
        r"\midrule",
    ]

    for env_name in env_names:
        env_results = [r for r in results if r["env"] == env_name]
        for i, r in enumerate(env_results):
            env_col = env_name if i == 0 else ""
            lines.append(
                f"{env_col} & {r['num_envs']} & "
                f"{r['sps_mean']:,.0f} & "
                f"{r['sps_max']:,.0f} & "
                f"{r['sps_first']:,.0f} & "
                f"{r['wall_time_s']:.0f} \\\\"
            )
        if env_name != env_names[-1]:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path = output_dir / "omnisafe_training_benchmark_table.tex"
    path.write_text("\n".join(lines))
    print(f"Saved: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OmniSafe PPOLag training SPS benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=DEFAULT_ENVS,
        metavar="ENV",
        help="Safety Gymnasium environment IDs to benchmark (default: all envs with a CRAX counterpart)",
    )
    parser.add_argument(
        "--num-envs",
        nargs="+",
        type=int,
        default=DEFAULT_NUM_ENVS,
        metavar="N",
        help="Number of parallel environments to sweep over (default: 1 2 4 8 16 32)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_names: List[str] = args.envs
    num_envs_list: List[int] = sorted(args.num_envs)

    print("=" * 60)
    print("OmniSafe PPOLag Training SPS Benchmark")
    print("=" * 60)
    print(f"environments     : {env_names}")
    print(f"num_envs sweep   : {num_envs_list}")
    print(f"steps_per_env    : {STEPS_PER_ENV:,}  (steps_per_epoch = steps_per_env × num_envs)")
    print(f"total_steps      : {TOTAL_STEPS:,}  (fixed; num_epochs computed per run)")
    print(f"min_epochs       : {MIN_EPOCHS}  (break-even at num_envs ≤ {TOTAL_STEPS // (MIN_EPOCHS * STEPS_PER_ENV)})")

    output_dir = Path(f"omnisafe_training_benchmark_{time.strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = output_dir / "_runs"
    scratch_dir.mkdir(exist_ok=True)

    csv_path = output_dir / "results.csv"
    results: List[Dict] = []

    for env_name in env_names:
        print(f"\n{'#' * 60}")
        print(f"# {env_name}")
        print(f"{'#' * 60}")
        for num_envs in num_envs_list:
            print(f"\n  --- num_envs={num_envs} ---")
            try:
                r = benchmark_run(env_name, num_envs, scratch_dir)
                if r:
                    results.append(r)
                    file_exists = csv_path.exists()
                    with open(csv_path, "a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=r.keys())
                        if not file_exists:
                            writer.writeheader()
                        writer.writerow(r)
            except Exception as e:
                print(f"  FAILED for {env_name} num_envs={num_envs}: {e}")
                import traceback
                traceback.print_exc()

    shutil.rmtree(scratch_dir, ignore_errors=True)

    completed_envs = [e for e in env_names if any(r["env"] == e for r in results)]
    if results:
        plot_results(results, completed_envs, num_envs_list, output_dir)
        generate_latex_table(results, completed_envs, output_dir)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for env_name in completed_envs:
        env_results = [r for r in results if r["env"] == env_name]
        print(f"\n  {env_name}")
        for r in env_results:
            print(f"    num_envs={r['num_envs']:>3}  SPS={r['sps_mean']:>10,.0f}  wall={r['wall_time_s']:.0f}s")
        best = max(env_results, key=lambda r: r["sps_mean"])
        print(f"    best: num_envs={best['num_envs']}  ({best['sps_mean']:,.0f} SPS)")

    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()
