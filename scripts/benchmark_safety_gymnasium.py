#!/usr/bin/env python3
"""
Benchmark script for Safety Gymnasium with parallel environments.
Benchmarks Safety Gymnasium using SafetyAsyncVectorEnv for parallel execution.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
# System monitoring
import psutil

try:
    import GPUtil

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("GPUtil not available - GPU metrics will be skipped")

# Safety Gymnasium imports
try:
    import safety_gymnasium
    from safety_gymnasium.vector import SafetyAsyncVectorEnv
except ImportError as e:
    print(f"❌ Error importing Safety Gymnasium: {e}")
    print("Please install safety-gymnasium: pip install safety-gymnasium")
    exit(1)


def measure_safety_gymnasium_throughput(num_envs: int, num_steps: int = 100_000,
                                        env_id: str = 'SafetyPointGoal1-v0') -> Dict:
    """Measure Safety-Gymnasium throughput with parallel environments.
    
    Args:
        num_envs: Number of parallel environments.
        num_steps: Total environment steps across all environments.
        env_id: Safety-Gymnasium environment id.
    
    Returns:
        Dictionary with benchmark metrics.
    """
    print(f"\n📊 Safety-Gymnasium benchmark (num_envs={num_envs})...")

    # Get baseline memory before creating environment
    process = psutil.Process()
    mem_baseline = process.memory_info().rss / 1024 / 1024  # MB

    # Create environment(s) first
    if num_envs == 1:
        # Single environment (no overhead)
        env = safety_gymnasium.make(env_id)
        use_vector = False
    else:
        # Parallel environments using SafetyAsyncVectorEnv
        env_fns = [lambda: safety_gymnasium.make(env_id) for _ in range(num_envs)]
        env = SafetyAsyncVectorEnv(env_fns, shared_memory=True)
        use_vector = True

    # Reset environment(s)
    if use_vector:
        obs, info = env.reset(seed=42)
    else:
        obs, _ = env.reset(seed=42)

    # Monitor resources after environment creation
    mem_after_creation = process.memory_info().rss / 1024 / 1024

    # For async vector envs, try to measure child processes
    if use_vector:
        try:
            children = process.children(recursive=True)
            for child in children:
                try:
                    mem_after_creation += child.memory_info().rss / 1024 / 1024
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except (AttributeError, psutil.AccessDenied):
            # Fallback: just measure main process
            pass

    if GPU_AVAILABLE:
        try:
            gpus = GPUtil.getGPUs()
            gpu_mem_before = gpus[0].memoryUsed if gpus else 0
        except:
            gpu_mem_before = 0
    else:
        gpu_mem_before = 0

    # Warmup phase
    # Note: For async vector envs, we need more warmup to allow processes to stabilize
    print("  Warming up...")
    warmup_steps = 200 if use_vector else 100  # More warmup for parallel envs
    for _ in range(warmup_steps):
        if use_vector:
            action = env.action_space.sample()
            obs, reward, cost, terminated, truncated, info = env.step(action)
            # Reset terminated/truncated environments
            if np.any(terminated | truncated):
                obs, info = env.reset()
        else:
            action = env.action_space.sample()
            obs, reward, cost, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()

    # Main benchmark
    print("  Running benchmark...")

    # Monitor CPU utilization during benchmark
    cpu_percent_before = psutil.cpu_percent(interval=0.1)
    # Use Slurm-allocated CPUs if available, otherwise fall back to system count
    cpu_count = int(os.environ.get('SLURM_CPUS_PER_TASK', psutil.cpu_count(logical=True)))
    cpu_count_physical = psutil.cpu_count(logical=False)

    start_time = time.time()
    steps_done = 0

    # Sample CPU utilization during benchmark (non-blocking)
    cpu_samples = []
    sample_interval = max(1, num_steps // 20)  # Sample ~20 times during benchmark

    # Calculate steps per environment
    steps_per_env = (num_steps + num_envs - 1) // num_envs  # Ceiling division

    if use_vector:
        # Vectorized environment
        for step_idx in range(steps_per_env):
            action = env.action_space.sample()
            obs, reward, cost, terminated, truncated, info = env.step(action)
            steps_done += num_envs  # Count all environment steps

            # Sample CPU utilization periodically
            if step_idx % sample_interval == 0:
                cpu_samples.append(psutil.cpu_percent(interval=None))

            # Reset terminated/truncated environments
            if np.any(terminated | truncated):
                obs, info = env.reset()
    else:
        # Single environment
        step_idx = 0
        while steps_done < num_steps:
            action = env.action_space.sample()
            obs, reward, cost, terminated, truncated, info = env.step(action)
            steps_done += 1
            step_idx += 1

            # Sample CPU utilization periodically
            if step_idx % sample_interval == 0:
                cpu_samples.append(psutil.cpu_percent(interval=None))

            if terminated or truncated:
                obs, _ = env.reset()

    # Get final CPU utilization
    cpu_percent_after = psutil.cpu_percent(interval=0.1)
    avg_cpu_percent = np.mean(cpu_samples) if cpu_samples else cpu_percent_after
    max_cpu_percent = max(cpu_samples) if cpu_samples else cpu_percent_after

    total_time = time.time() - start_time
    sps = steps_done / total_time if total_time > 0 else 0.0

    # Monitor resources after benchmark
    mem_after_benchmark = process.memory_info().rss / 1024 / 1024

    # For async vector envs, try to measure child processes
    if use_vector:
        try:
            children = process.children(recursive=True)
            for child in children:
                try:
                    mem_after_benchmark += child.memory_info().rss / 1024 / 1024
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except (AttributeError, psutil.AccessDenied):
            # Fallback: just measure main process
            pass

    # Use peak memory (max of after creation and after benchmark) minus baseline
    mem_peak = max(mem_after_creation, mem_after_benchmark)
    mem_used = max(0, mem_peak - mem_baseline)  # Ensure non-negative

    if GPU_AVAILABLE:
        try:
            gpus = GPUtil.getGPUs()
            gpu_mem_after = gpus[0].memoryUsed if gpus else 0
            gpu_mem_used = max(0, gpu_mem_after - gpu_mem_before)
        except:
            gpu_mem_used = 0
    else:
        gpu_mem_used = 0

    # Cleanup
    env.close()

    # Give processes time to clean up
    import gc
    gc.collect()
    time.sleep(0.1)  # Brief pause for cleanup

    print(f"  ✓ SPS: {sps:,.0f}")
    print(f"  ✓ Memory: CPU={mem_used:.0f}MB, GPU={gpu_mem_used:.0f}MB")
    print(f"  ✓ CPU: {avg_cpu_percent:.1f}% avg ({max_cpu_percent:.1f}% peak) of {cpu_count} cores")
    print(f"  ✓ Time: {total_time:.1f}s")
    print(f"  ✓ Steps: {steps_done:,} total ({steps_done // num_envs:,} per env)")

    # Calculate efficiency metrics
    cpu_efficiency = (avg_cpu_percent / 100.0) * (num_envs / cpu_count) if cpu_count > 0 else 0
    mem_per_env = mem_used / num_envs if num_envs > 0 else mem_used

    return {
        'framework': 'Safety-Gymnasium',
        'env': env_id,
        'num_envs': num_envs,
        'steps_per_second': sps,
        'cpu_memory_mb': mem_used,
        'gpu_memory_mb': gpu_mem_used,
        'total_time': total_time,
        'jit_time': 0,  # Not applicable for Safety Gymnasium
        'num_steps': steps_done,
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
    """Measure Safety-Gymnasium throughput for a single (env, num_envs) configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', type=str, default='SafetyPointGoal1-v0', help='Safety-Gymnasium env id.')
    parser.add_argument('--num_envs', type=int, required=True, help='Number of parallel envs')
    parser.add_argument('--num_steps', type=int, default=500_000, help='Total env steps to run')
    parser.add_argument('--output_root', type=str, default=str(DEFAULT_OUTPUT_ROOT),
                        help='Parent dir for the results folder.')
    args = parser.parse_args()

    output_dir = Path(args.output_root) / (
        f"safety_gym_benchmark_results_{args.env}_n{args.num_envs}_{time.strftime('%Y%m%d_%H%M%S')}")

    result = measure_safety_gymnasium_throughput(args.num_envs, num_steps=args.num_steps, env_id=args.env)
    if not result:
        sys.exit(f"❌ Benchmark failed for env={args.env} num_envs={args.num_envs}")

    # Only create the folder once there is a result, so failed runs leave nothing behind.
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'benchmark_results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)
    print(f"✓ {args.env} @ {args.num_envs} envs: {result['steps_per_second']:,.0f} SPS -> {csv_path}")


if __name__ == "__main__":
    main()
