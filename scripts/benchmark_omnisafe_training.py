#!/usr/bin/env python3
"""Training SPS benchmark for OmniSafe + Safety-Gymnasium (CPU).

Runs PPOLag training on SafetyPointGoal1-v0 with varying numbers of parallel
environments and records the Steps-Per-Second reported by the training loop
(Time/FPS in the omnisafe logger), i.e. the throughput including gradient updates.

This mirrors benchmark_crax_training.py so results can be compared directly.

Usage:
    conda run -n omnisafe python scripts/benchmark_omnisafe_training.py
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Fix broken TF namespace package that causes tensorboard import to fail.
# The user's ~/.local site-packages has an empty tensorflow namespace package
# that satisfies `import tensorflow` but has no attributes. We patch it before
# any torch/omnisafe imports trigger the lazy tensorboard-compat loader.
# ---------------------------------------------------------------------------
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
# Benchmark parameters
# ---------------------------------------------------------------------------
ENV_NAME = "SafetyPointGoal1-v0"

# steps_per_epoch must be divisible by every value in NUM_ENVS_LIST.
# LCM(1,2,4,8,16,32) = 32  →  4000 / 32 = 125  ✓
STEPS_PER_EPOCH = 4_000
NUM_EPOCHS = 15           # epochs per run; first is discarded (warmup)
TOTAL_STEPS = STEPS_PER_EPOCH * NUM_EPOCHS  # 60_000

NUM_ENVS_LIST = [1, 2, 4, 8, 16, 32]


# ---------------------------------------------------------------------------

def benchmark_num_envs(num_envs: int, scratch_dir: Path) -> Dict:
    """Run a short PPOLag training run and return per-epoch SPS statistics."""
    import omnisafe  # deferred so the TF patch above is already applied

    print(f"\n{'=' * 50}")
    print(f"num_envs = {num_envs}")
    print(f"{'=' * 50}")

    log_dir = scratch_dir / f"omnisafe_run_envs{num_envs}"
    log_dir.mkdir(parents=True, exist_ok=True)

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024

    t0 = time.time()

    agent = omnisafe.Agent(
        "PPOLag",
        ENV_NAME,
        custom_cfgs={
            "train_cfgs": {
                "vector_env_nums": num_envs,
                "total_steps": TOTAL_STEPS,
                "device": "cpu",
            },
            "algo_cfgs": {
                "steps_per_epoch": STEPS_PER_EPOCH,
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

    # Read the per-epoch metrics written by the omnisafe logger
    progress_files = list(log_dir.rglob("progress.csv"))
    if not progress_files:
        print("  WARNING: progress.csv not found - no SPS data")
        return {}

    df = pd.read_csv(progress_files[0])
    if "Time/FPS" not in df.columns:
        print(f"  WARNING: Time/FPS column missing. Columns: {df.columns.tolist()}")
        return {}

    sps_samples: List[float] = df["Time/FPS"].tolist()
    print(f"  Collected {len(sps_samples)} SPS samples across {NUM_EPOCHS} epochs")

    if not sps_samples:
        return {}

    # Discard first epoch (JIT / cache warmup)
    stable = sps_samples[1:] if len(sps_samples) > 1 else sps_samples

    result = {
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

    # Clean up the run directory to save disk space
    shutil.rmtree(log_dir, ignore_errors=True)

    return result


def plot_results(results: List[Dict], output_dir: Path) -> None:
    if not results:
        return

    plt.style.use("seaborn-v0_8-paper")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = [r["num_envs"] for r in results]
    y_mean = [r["sps_mean"] for r in results]
    y_max = [r["sps_max"] for r in results]
    y_first = [r["sps_first"] for r in results]

    ax = axes[0]
    ax.plot(x, y_mean, "o-", label="Stable mean SPS", linewidth=2, markersize=8)
    ax.plot(x, y_max, "s--", label="Stable max SPS", linewidth=1.5, markersize=6, alpha=0.7)
    ax.plot(x, y_first, "^:", label="First epoch SPS (incl. warmup)", linewidth=1.5, markersize=6, alpha=0.6)
    ax.set_xlabel("Number of Parallel Environments")
    ax.set_ylabel("Steps Per Second (training)")
    ax.set_title("OmniSafe PPOLag: Training SPS vs num_envs")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    if len(results) > 1:
        baseline = results[0]["sps_mean"]
        speedup = [r["sps_mean"] / baseline for r in results]
        ideal = [r["num_envs"] / results[0]["num_envs"] for r in results]
        ax.plot(x, speedup, "o-", label="Actual speedup", linewidth=2, markersize=8)
        ax.plot(x, ideal, "k--", alpha=0.4, label="Ideal (linear)")
    ax.set_xlabel("Number of Parallel Environments")
    ax.set_ylabel("Speedup over 1-env baseline")
    ax.set_title("OmniSafe PPOLag: Scaling Efficiency")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.suptitle(
        f"OmniSafe PPOLag Training Throughput  |  env={ENV_NAME}  |  "
        f"steps_per_epoch={STEPS_PER_EPOCH}",
        fontsize=12,
    )
    plt.tight_layout()

    for suffix in ("png", "pdf"):
        path = output_dir / f"omnisafe_training_benchmark.{suffix}"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")


def generate_latex_table(results: List[Dict], output_dir: Path) -> None:
    if not results:
        return
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{OmniSafe PPOLag training throughput (SPS during training)}",
        r"\label{tab:omnisafe_training_benchmark}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Envs & Mean SPS & Max SPS & First-epoch SPS & Wall time (s) \\",
        r"\midrule",
    ]
    for r in results:
        lines.append(
            f"{r['num_envs']} & "
            f"{r['sps_mean']:,.0f} & "
            f"{r['sps_max']:,.0f} & "
            f"{r['sps_first']:,.0f} & "
            f"{r['wall_time_s']:.0f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path = output_dir / "omnisafe_training_benchmark_table.tex"
    path.write_text("\n".join(lines))
    print(f"Saved: {path}")


def main() -> None:
    print("=" * 60)
    print("OmniSafe PPOLag Training SPS Benchmark")
    print("=" * 60)
    print(f"env              : {ENV_NAME}")
    print(f"steps_per_epoch  : {STEPS_PER_EPOCH:,}")
    print(f"num_epochs       : {NUM_EPOCHS}  (total_steps={TOTAL_STEPS:,})")
    print(f"num_envs sweep   : {NUM_ENVS_LIST}")

    output_dir = Path(f"omnisafe_training_benchmark_{time.strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = output_dir / "_runs"
    scratch_dir.mkdir(exist_ok=True)

    csv_path = output_dir / "results.csv"
    results: List[Dict] = []

    for num_envs in NUM_ENVS_LIST:
        try:
            r = benchmark_num_envs(num_envs, scratch_dir)
            if r:
                results.append(r)
                file_exists = csv_path.exists()
                with open(csv_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=r.keys())
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(r)
        except Exception as e:
            print(f"  FAILED for num_envs={num_envs}: {e}")
            import traceback
            traceback.print_exc()

    # Clean up the scratch directory
    shutil.rmtree(scratch_dir, ignore_errors=True)

    if results:
        plot_results(results, output_dir)
        generate_latex_table(results, output_dir)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for r in results:
        print(f"  num_envs={r['num_envs']:>3}  SPS={r['sps_mean']:>10,.0f}  wall={r['wall_time_s']:.0f}s")

    if results:
        best = max(results, key=lambda r: r["sps_mean"])
        print(f"\nBest config : num_envs={best['num_envs']}  ({best['sps_mean']:,.0f} SPS)")

    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()
