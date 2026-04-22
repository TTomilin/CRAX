#!/usr/bin/env python3
"""
Training-integrated SPS benchmark for CRAX (SafeBrax + JAX/GPU).

Unlike benchmark_crax.py which measures raw env-stepping throughput, this
script runs actual PPO-Lag training and captures the SPS reported by the
training loop itself (i.e., including gradient updates, normalisation, etc.).
"""

import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import psutil

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def setup_gpu_environment():
    os.environ['MUJOCO_GL'] = 'egl'
    xla_flags = os.environ.get('XLA_FLAGS', '')
    if '--xla_gpu_triton_gemm_any=True' not in xla_flags:
        xla_flags += ' --xla_gpu_triton_gemm_any=True'
        os.environ['XLA_FLAGS'] = xla_flags


setup_gpu_environment()

import jax
from brax import envs
from brax.training.agents.ppo_lag import train as ppo_lag

print(f"JAX backend : {jax.default_backend()}")
print(f"JAX devices : {jax.devices()}")

# ---------------------------------------------------------------------------
# Benchmark parameters
# ---------------------------------------------------------------------------
ENV_NAME = 'safe_goal_point'
EPISODE_LENGTH = 1000

# We run for enough timesteps to get several SPS measurements (evals).
# Keep it short so the whole sweep finishes in a reasonable time.
NUM_TIMESTEPS = 5_000_000
NUM_EVALS = 5          # progress_fn is called this many times → 5 SPS samples

# Fixed algorithm hyper-parameters (match typical CRAX defaults, independent
# of num_envs so that only parallelism varies).
BASE_KWARGS = dict(
    num_timesteps=NUM_TIMESTEPS,
    episode_length=EPISODE_LENGTH,
    num_evals=NUM_EVALS,
    learning_rate=3e-4,
    entropy_cost=1e-2,
    discounting=0.97,
    unroll_length=20,
    batch_size=1024,
    num_minibatches=32,
    num_updates_per_batch=4,
    normalize_observations=True,
    reward_scaling=0.1,
    clipping_epsilon=0.2,
    gae_lambda=0.95,
    # disable eval envs and internal step-level logger (we get SPS from progress_fn)
    num_eval_envs=0,
    log_training_metrics=False,
    seed=0,
)

# NUM_ENVS_LIST = [64, 128, 256, 512, 1024, 2048]
NUM_ENVS_LIST = [2048]


# ---------------------------------------------------------------------------

def benchmark_num_envs(num_envs: int) -> Dict:
    """Run a short PPO-Lag training run and return per-epoch SPS statistics."""
    print(f"\n{'=' * 50}")
    print(f"num_envs = {num_envs}")
    print(f"{'=' * 50}")

    sps_samples: List[float] = []

    def progress_fn(step: int, metrics: dict) -> None:
        sps = float(np.asarray(metrics.get('training/sps', 0)).mean())
        sps_samples.append(sps)
        print(f"  step={step:>10,}  SPS={sps:>12,.0f}")

    env = envs.get_environment(ENV_NAME)
    eval_env = envs.get_environment(ENV_NAME)

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024

    t0 = time.time()
    _make_inference, _params, _metrics, _eval_env = ppo_lag(
        environment=env,
        eval_env=eval_env,
        num_envs=num_envs,
        progress_fn=progress_fn,
        **BASE_KWARGS,
    )
    wall_time = time.time() - t0

    mem_after = process.memory_info().rss / 1024 / 1024
    mem_used = max(0.0, mem_after - mem_before)

    if not sps_samples:
        print("  WARNING: no SPS samples collected")
        return {}

    # Discard the first sample (JIT compile epoch) if we have more than one
    stable_samples = sps_samples[1:] if len(sps_samples) > 1 else sps_samples

    result = {
        'num_envs': num_envs,
        'sps_mean': float(np.mean(stable_samples)),
        'sps_median': float(np.median(stable_samples)),
        'sps_max': float(np.max(stable_samples)),
        'sps_min': float(np.min(stable_samples)),
        'sps_first': sps_samples[0],   # includes JIT overhead
        'num_samples': len(sps_samples),
        'wall_time_s': wall_time,
        'cpu_memory_mb': mem_used,
    }

    print(f"  → SPS (stable mean): {result['sps_mean']:,.0f}")
    print(f"  → SPS (stable max):  {result['sps_max']:,.0f}")
    print(f"  → Wall time:         {wall_time:.1f}s")
    return result


def plot_results(results: List[Dict], output_dir: Path):
    if not results:
        return

    plt.style.use('seaborn-v0_8-paper')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = [r['num_envs'] for r in results]
    y_mean = [r['sps_mean'] for r in results]
    y_max = [r['sps_max'] for r in results]
    y_first = [r['sps_first'] for r in results]

    ax = axes[0]
    ax.plot(x, y_mean, 'o-', label='Stable mean SPS', linewidth=2, markersize=8)
    ax.plot(x, y_max, 's--', label='Stable max SPS', linewidth=1.5, markersize=6, alpha=0.7)
    ax.plot(x, y_first, '^:', label='First epoch SPS (incl. JIT)', linewidth=1.5, markersize=6, alpha=0.6)
    ax.set_xlabel('Number of Parallel Environments')
    ax.set_ylabel('Steps Per Second (training)')
    ax.set_title('CRAX: Training SPS vs num_envs')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    if len(results) > 1:
        baseline = results[0]['sps_mean']
        speedup = [r['sps_mean'] / baseline for r in results]
        ideal = [r['num_envs'] / results[0]['num_envs'] for r in results]
        ax.plot(x, speedup, 'o-', label='Actual speedup', linewidth=2, markersize=8)
        ax.plot(x, ideal, 'k--', alpha=0.4, label='Ideal (linear)')
    ax.set_xlabel('Number of Parallel Environments')
    ax.set_ylabel('Speedup over smallest config')
    ax.set_title('CRAX: Scaling Efficiency')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.suptitle(f'CRAX PPO-Lag Training Throughput  |  env={ENV_NAME}', fontsize=13)
    plt.tight_layout()

    for suffix in ('png', 'pdf'):
        path = output_dir / f'crax_training_benchmark.{suffix}'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")


def generate_latex_table(results: List[Dict], output_dir: Path):
    if not results:
        return
    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\caption{CRAX PPO-Lag training throughput (SPS during training)}',
        r'\label{tab:crax_training_benchmark}',
        r'\begin{tabular}{lrrrr}',
        r'\toprule',
        r'Envs & Mean SPS & Max SPS & First-epoch SPS & Wall time (s) \\',
        r'\midrule',
    ]
    for r in results:
        lines.append(
            f"{r['num_envs']} & "
            f"{r['sps_mean']:,.0f} & "
            f"{r['sps_max']:,.0f} & "
            f"{r['sps_first']:,.0f} & "
            f"{r['wall_time_s']:.0f} \\\\"
        )
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    path = output_dir / 'crax_training_benchmark_table.tex'
    path.write_text('\n'.join(lines))
    print(f"Saved: {path}")


def main():
    print('=' * 60)
    print('CRAX PPO-Lag Training SPS Benchmark')
    print('=' * 60)
    print(f'env            : {ENV_NAME}')
    print(f'num_timesteps  : {NUM_TIMESTEPS:,}')
    print(f'num_evals      : {NUM_EVALS}')
    print(f'num_envs sweep : {NUM_ENVS_LIST}')

    output_dir = Path(f"crax_training_benchmark_{time.strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'results.csv'

    results = []
    for num_envs in NUM_ENVS_LIST:
        try:
            r = benchmark_num_envs(num_envs)
            if r:
                results.append(r)
                file_exists = csv_path.exists()
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=r.keys())
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(r)
        except Exception as e:
            print(f"  FAILED for num_envs={num_envs}: {e}")
            import traceback; traceback.print_exc()

    if results:
        plot_results(results, output_dir)
        generate_latex_table(results, output_dir)

    print('\n' + '=' * 60)
    print('Summary')
    print('=' * 60)
    for r in results:
        print(f"  num_envs={r['num_envs']:>5}  SPS={r['sps_mean']:>12,.0f}  wall={r['wall_time_s']:.0f}s")

    if results:
        best = max(results, key=lambda r: r['sps_mean'])
        print(f"\nBest config : num_envs={best['num_envs']}  ({best['sps_mean']:,.0f} SPS)")

    print(f"\nResults saved to: {output_dir}/")


if __name__ == '__main__':
    main()
