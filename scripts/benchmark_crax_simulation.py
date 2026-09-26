#!/usr/bin/env python3
"""
Benchmark script for CRAX with parallel batched environments.
Benchmarks CRAX using JAX vectorization for parallel execution.
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np

# Force unbuffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


# GPU and performance optimizations
def setup_gpu_environment():
    """Setup GPU environment for optimal performance."""
    # Check for GPU availability
    try:
        if subprocess.run(['nvidia-smi'], capture_output=True).returncode != 0:
            print("⚠️ Warning: Cannot communicate with GPU. Running on CPU.")
    except FileNotFoundError:
        print("⚠️ Warning: nvidia-smi not found. Running on CPU.")

    # Configure MuJoCo to use the EGL rendering backend (requires GPU)
    os.environ['MUJOCO_GL'] = 'egl'

    # Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
    xla_flags = os.environ.get('XLA_FLAGS', '')
    if '--xla_gpu_triton_gemm_any=True' not in xla_flags:
        xla_flags += ' --xla_gpu_triton_gemm_any=True'
        os.environ['XLA_FLAGS'] = xla_flags
        print("✓ XLA Triton GEMM optimization enabled (~30% speedup)")

    print(f"✓ XLA flags configured: {os.environ.get('XLA_FLAGS', '')}")


# Apply optimizations before importing JAX
setup_gpu_environment()

# System monitoring
import psutil

try:
    import GPUtil

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("GPUtil not available - GPU metrics will be skipped")

# JAX/CRAX (import after setting environment)
import jax
import jax.numpy as jnp
from crax import envs

# Verify JAX backend
print(f"✓ JAX backend: {jax.default_backend()}", flush=True)
print(f"✓ JAX devices: {jax.devices()}", flush=True)
sys.stdout.flush()


def measure_crax_throughput(num_envs: int, num_steps: int = 1_000_000, env_name: str = 'safe_goal_point') -> Dict:
    """Measure CRAX throughput using random actions (no training).

    Args:
        num_envs: Number of parallel batched environments.
        num_steps: TOTAL environment steps across all envs (matches PPO semantics).
        env_name: CRAX environment to benchmark.
    
    Returns:
        Dictionary with benchmark metrics.
    """
    print(f"\n📊 CRAX benchmark (num_envs={num_envs})...")

    # Get baseline memory before creating environment
    process = psutil.Process()
    mem_baseline = process.memory_info().rss / 1024 / 1024  # MB

    # Create a batched environment
    try:
        env = envs.create(
            env_name=env_name,
            batch_size=num_envs,
            episode_length=1000,
            action_repeat=1,
            auto_reset=True,
        )
    except Exception:
        # Fallback in case create is unavailable
        env = envs.get_environment(env_name)

    # Monitor resources after environment creation
    mem_after_creation = process.memory_info().rss / 1024 / 1024

    if GPU_AVAILABLE:
        try:
            gpus = GPUtil.getGPUs()
            gpu_mem_before = gpus[0].memoryUsed if gpus else 0
        except:
            gpu_mem_before = 0
    else:
        gpu_mem_before = 0

    # RNG and initial state
    rng = jax.random.PRNGKey(42)
    state = env.reset(rng)

    # Monitor CPU utilization
    # Use Slurm-allocated CPUs if available, otherwise fall back to system count
    cpu_count = int(os.environ.get('SLURM_CPUS_PER_TASK', psutil.cpu_count(logical=True)))
    cpu_count_physical = psutil.cpu_count(logical=False)

    # Build a chunked JIT-compiled random-action rollout to avoid huge compilations
    def make_rollout(length: int):
        def rollout(rng_key, init_state):
            def body(carry, _):
                rng_inner, s = carry
                rng_inner, key = jax.random.split(rng_inner)
                actions = jax.random.uniform(
                    key,
                    shape=(num_envs, env.action_size),
                    minval=-1.0,
                    maxval=1.0,
                )
                s = env.step(s, actions)
                return (rng_inner, s), None

            (rng_out, s_out), _ = jax.lax.scan(body, (rng_key, init_state), xs=None, length=length)
            return rng_out, s_out

        return jax.jit(rollout)

    # Translate total env-steps to per-env steps (ceil) to match PPO semantics
    steps_per_env = (num_steps + max(1, num_envs) - 1) // max(1, num_envs)

    # Chunking setup based on per-env steps to avoid huge compilations
    chunk_len = 4096 if steps_per_env >= 4096 else steps_per_env
    rollout_chunk = make_rollout(chunk_len)
    num_full = steps_per_env // chunk_len
    remainder = steps_per_env - num_full * chunk_len

    # Precisely measure JIT compile time vs execution time
    jit_time = 0.0
    exec_time = 0.0

    # Sample CPU utilization during benchmark
    cpu_samples = []
    sample_interval = max(1, num_full // 20) if num_full > 0 else 1

    # Compile chunk kernel once
    print(f"  Compiling JIT kernel (num_envs={num_envs}, chunk_len={chunk_len})...", flush=True)
    sys.stdout.flush()
    t0 = time.time()
    compiled_chunk = rollout_chunk.lower(rng, state).compile()
    jit_time += time.time() - t0
    print(f"  ✓ Compilation complete in {jit_time:.1f}s", flush=True)

    # Execute chunk kernel num_full times
    print("  Running benchmark...")
    t1 = time.time()
    for i in range(num_full):
        rng, state = compiled_chunk(rng, state)

        # Sample CPU utilization periodically
        if i % sample_interval == 0:
            cpu_samples.append(psutil.cpu_percent(interval=None))

    exec_time += time.time() - t1

    # Compile and execute remainder kernel if needed
    if remainder:
        print(f"  Compiling remainder kernel (remainder={remainder})...", flush=True)
        sys.stdout.flush()
        rollout_rem = make_rollout(remainder)
        t2 = time.time()
        compiled_rem = rollout_rem.lower(rng, state).compile()
        jit_time += time.time() - t2
        print(f"  ✓ Remainder compilation complete in {time.time() - t2:.1f}s", flush=True)
        sys.stdout.flush()
        t3 = time.time()
        rng, state = compiled_rem(rng, state)
        exec_time += time.time() - t3

    # Ensure all computations finish before final timing
    _ = jnp.sum(state.obs).block_until_ready()

    # Get final CPU utilization
    cpu_percent_after = psutil.cpu_percent(interval=0.1)
    avg_cpu_percent = np.mean(cpu_samples) if cpu_samples else cpu_percent_after
    max_cpu_percent = max(cpu_samples) if cpu_samples else cpu_percent_after

    # Monitor resources after benchmark
    mem_after_benchmark = process.memory_info().rss / 1024 / 1024

    # Use peak memory (max of after creation and after benchmark) minus baseline
    mem_peak = max(mem_after_creation, mem_after_benchmark)
    mem_used = max(0, mem_peak - mem_baseline)

    if GPU_AVAILABLE:
        try:
            gpus = GPUtil.getGPUs()
            gpu_mem_after = gpus[0].memoryUsed if gpus else 0
            gpu_mem_used = max(0, gpu_mem_after - gpu_mem_before)
        except:
            gpu_mem_used = 0
    else:
        gpu_mem_used = 0

    # Steps per second counts all env steps; use execution time only (exclude JIT)
    total_env_steps = steps_per_env * max(1, num_envs)
    sps = total_env_steps / exec_time if exec_time > 0 else float('inf')

    # Calculate efficiency metrics
    cpu_efficiency = (avg_cpu_percent / 100.0) * (num_envs / cpu_count) if cpu_count > 0 else 0
    mem_per_env = mem_used / num_envs if num_envs > 0 else mem_used

    print(f"  ✓ SPS: {sps:,.0f}", flush=True)
    print(f"  ✓ Memory: CPU={mem_used:.0f}MB, GPU={gpu_mem_used:.0f}MB", flush=True)
    print(f"  ✓ CPU: {avg_cpu_percent:.1f}% avg ({max_cpu_percent:.1f}% peak) of {cpu_count} cores", flush=True)
    print(f"  ✓ Time: {exec_time:.1f}s (JIT {jit_time:.1f}s)", flush=True)
    print(f"  ✓ Steps: {total_env_steps:,} total ({steps_per_env:,} per env)", flush=True)
    sys.stdout.flush()

    return {
        'framework': 'CRAX',
        'env': env_name,
        'num_envs': num_envs,
        'steps_per_second': sps,
        'cpu_memory_mb': mem_used,
        'gpu_memory_mb': gpu_mem_used,
        'total_time': exec_time,
        'jit_time': jit_time,
        'num_steps': total_env_steps,
        'cpu_percent_avg': avg_cpu_percent,
        'cpu_percent_max': max_cpu_percent,
        'cpu_count': cpu_count,
        'cpu_count_physical': cpu_count_physical,
        'cpu_efficiency': cpu_efficiency,
        'mem_per_env_mb': mem_per_env,
    }



# Same folder the plotting scripts read from (results/data/performance), regardless of cwd.
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / 'results' / 'data' / 'performance'


def main():
    """Measure CRAX throughput for a single (env, num_envs) configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', type=str, default='safe_goal_point', help='CRAX environment name.')
    parser.add_argument('--num_envs', type=int, required=True, help='Number of parallel envs')
    parser.add_argument('--num_steps', type=int, default=500_000, help='Total env steps to run')
    parser.add_argument('--output_root', type=str, default=str(DEFAULT_OUTPUT_ROOT),
                        help='Parent dir for the results folder.')
    args = parser.parse_args()

    output_dir = Path(args.output_root) / (
        f"crax_benchmark_results_{args.env}_n{args.num_envs}_{time.strftime('%Y%m%d_%H%M%S')}")

    result = measure_crax_throughput(args.num_envs, num_steps=args.num_steps, env_name=args.env)
    if not result:
        sys.exit(f"❌ Benchmark failed for env={args.env} num_envs={args.num_envs}")

    # Only create the folder once there is a result, so failed runs leave nothing behind.
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'benchmark_results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)
    print(f"✓ {args.env} @ {args.num_envs} envs: {result['steps_per_second']:,.0f} SPS -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
