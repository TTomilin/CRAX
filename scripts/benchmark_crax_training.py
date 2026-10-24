#!/usr/bin/env python3
"""
Training-integrated SPS benchmark for CRAX (SafeBrax + JAX/GPU).

This script runs actual PPO-Lag training and captures the SPS reported by the
training loop itself (i.e., including gradient updates, normalization, etc.).
"""

import argparse
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
from crax import envs
from crax.training.agents.ppo_lag import train as ppo_lag

print(f"JAX backend : {jax.default_backend()}")
print(f"JAX devices : {jax.devices()}")


# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='CRAX PPO-Lag training SPS benchmark',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--envs', nargs='+',
        default=["safe_reacher", "safe_goal_point", "safe_push_point", "safe_lift_spider", "safe_circle_point",
                 "safe_height_humanoid", "safe_pathway_walker2d", "safe_velocity_humanoid"],
        metavar='ENV',
        help='One or more SafeBrax environment names to benchmark',
    )
    parser.add_argument(
        '--num_envs', nargs='+', type=int,
        default=[64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
        metavar='N',
        help='Parallel environment counts to sweep over',
    )
    parser.add_argument(
        '--num_timesteps', type=int, default=5_000_000,
        help='Total training timesteps per run',
    )
    parser.add_argument(
        '--num_evals', type=int, default=5,
        help='Number of progress_fn callbacks (= SPS samples per run)',
    )
    parser.add_argument(
        '--episode_length', type=int, default=1000,
        help='Episode length',
    )
    return parser.parse_args()


def benchmark_num_envs(num_envs: int, env_name: str, base_kwargs: dict) -> Dict:
    """Run a short PPO-Lag training run and return per-epoch SPS statistics."""
    print(f"\n{'=' * 50}")
    print(f"num_envs = {num_envs}")
    print(f"{'=' * 50}")

    sps_samples: List[float] = []

    def progress_fn(step: int, metrics: dict) -> None:
        sps = float(np.asarray(metrics.get('training/sps', 0)).mean())
        sps_samples.append(sps)
        print(f"  step={step:>10,}  SPS={sps:>12,.0f}")

    env = envs.get_environment(env_name)
    eval_env = envs.get_environment(env_name)

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024

    t0 = time.time()
    _make_inference, _params, _metrics, _eval_env = ppo_lag(
        environment=env,
        eval_env=eval_env,
        num_envs=num_envs,
        progress_fn=progress_fn,
        **base_kwargs,
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
        'env_name': env_name,
        'num_envs': num_envs,
        'sps_mean': float(np.mean(stable_samples)),
        'sps_median': float(np.median(stable_samples)),
        'sps_max': float(np.max(stable_samples)),
        'sps_min': float(np.min(stable_samples)),
        'sps_first': sps_samples[0],  # includes JIT overhead
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

    env_names = list(dict.fromkeys(r['env_name'] for r in results))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, env_name in enumerate(env_names):
        env_results = [r for r in results if r['env_name'] == env_name]
        color = colors[i % len(colors)]
        x = [r['num_envs'] for r in env_results]
        y_mean = [r['sps_mean'] for r in env_results]
        y_max = [r['sps_max'] for r in env_results]
        y_first = [r['sps_first'] for r in env_results]

        axes[0].plot(x, y_mean, 'o-', label=f'{env_name} mean', color=color, linewidth=2, markersize=8)
        axes[0].plot(x, y_max, 's--', label=f'{env_name} max', color=color, linewidth=1.5, markersize=6, alpha=0.7)
        axes[0].plot(x, y_first, '^:', label=f'{env_name} first (JIT)', color=color, linewidth=1.5, markersize=6,
                     alpha=0.5)

        if len(env_results) > 1:
            baseline = env_results[0]['sps_mean']
            speedup = [r['sps_mean'] / baseline for r in env_results]
            ideal = [r['num_envs'] / env_results[0]['num_envs'] for r in env_results]
            axes[1].plot(x, speedup, 'o-', label=env_name, color=color, linewidth=2, markersize=8)
            if i == 0:
                axes[1].plot(x, ideal, 'k--', alpha=0.4, label='Ideal (linear)')

    axes[0].set_xlabel('Number of Parallel Environments')
    axes[0].set_ylabel('Steps Per Second (training)')
    axes[0].set_title('CRAX: Training SPS vs num_envs')
    axes[0].set_xscale('log', base=2)
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel('Number of Parallel Environments')
    axes[1].set_ylabel('Speedup over smallest config')
    axes[1].set_title('CRAX: Scaling Efficiency')
    axes[1].set_xscale('log', base=2)
    axes[1].set_yscale('log', base=2)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    plt.suptitle('CRAX PPO-Lag Training Throughput', fontsize=13)
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
        r'\begin{tabular}{llrrrr}',
        r'\toprule',
        r'Environment & Envs & Mean SPS & Max SPS & First-epoch SPS & Wall time (s) \\',
        r'\midrule',
    ]
    for r in results:
        lines.append(
            f"{r['env_name']} & "
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
    args = parse_args()

    print('=' * 60)
    print('CRAX PPO-Lag Training SPS Benchmark')
    print('=' * 60)
    print(f'environments   : {args.envs}')
    print(f'num_envs sweep : {args.num_envs}')
    print(f'num_timesteps  : {args.num_timesteps:,}')
    print(f'num_evals      : {args.num_evals}')
    print(f'episode_length : {args.episode_length}')

    base_kwargs = dict(
        num_timesteps=args.num_timesteps,
        episode_length=args.episode_length,
        num_evals=args.num_evals,
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
        num_eval_envs=0,
        log_training_metrics=False,
        seed=0,
    )

    output_dir = Path(f"crax_training_benchmark_{time.strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'results.csv'

    results = []
    for env_name in args.envs:
        for num_envs in args.num_envs:
            try:
                r = benchmark_num_envs(num_envs, env_name, base_kwargs)
                if r:
                    results.append(r)
                    file_exists = csv_path.exists()
                    with open(csv_path, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=r.keys())
                        if not file_exists:
                            writer.writeheader()
                        writer.writerow(r)
            except Exception as e:
                print(f"  FAILED for env={env_name} num_envs={num_envs}: {e}")
                import traceback;
                traceback.print_exc()

    if results:
        plot_results(results, output_dir)
        generate_latex_table(results, output_dir)

    print('\n' + '=' * 60)
    print('Summary')
    print('=' * 60)
    for r in results:
        print(
            f"  {r['env_name']:<25}  num_envs={r['num_envs']:>5}  SPS={r['sps_mean']:>12,.0f}  wall={r['wall_time_s']:.0f}s")

    if results:
        best = max(results, key=lambda r: r['sps_mean'])
        print(f"\nBest : env={best['env_name']}  num_envs={best['num_envs']}  ({best['sps_mean']:,.0f} SPS)")

    print(f"\nResults saved to: {output_dir}/")


if __name__ == '__main__':
    main()
