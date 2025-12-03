#!/usr/bin/env python3
"""
Benchmark script for SafeBrax with parallel batched environments.
Benchmarks SafeBrax using JAX vectorization for parallel execution.
"""

import sys
import time
import os
import csv
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import subprocess

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

# JAX/SafeBrax (import after setting environment)
import jax
import jax.numpy as jnp
from brax import envs

# Verify JAX backend
print(f"✓ JAX backend: {jax.default_backend()}", flush=True)
print(f"✓ JAX devices: {jax.devices()}", flush=True)
sys.stdout.flush()


def measure_safebrax_throughput(num_envs: int, num_steps: int = 1_000_000) -> Dict:
    """Measure SafeBrax throughput using random actions (no training).

    Args:
        num_envs: Number of parallel batched environments.
        num_steps: TOTAL environment steps across all envs (matches PPO semantics).
    
    Returns:
        Dictionary with benchmark metrics.
    """
    print(f"\n📊 SafeBrax benchmark (num_envs={num_envs})...")
    
    # Get baseline memory before creating environment
    process = psutil.Process()
    mem_baseline = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create a batched environment
    try:
        env = envs.create(
            env_name='safe_point_goal',
            batch_size=num_envs,
            episode_length=1000,
            action_repeat=1,
            auto_reset=True,
        )
    except Exception:
        # Fallback in case create is unavailable
        env = envs.get_environment('safe_point_goal')
    
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
        'framework': 'SafeBrax',
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


def plot_results(results: List[Dict], output_dir: Path):
    """Generate comparison plots."""
    if not results:
        print("No results to plot")
        return
    
    # Set style
    plt.style.use('seaborn-v0_8-paper')
    
    # Prepare data
    safebrax_results = [r for r in results if r['framework'] == 'SafeBrax']
    
    if not safebrax_results:
        print("No SafeBrax results to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Throughput scaling
    ax = axes[0, 0]
    x = [r['num_envs'] for r in safebrax_results]
    y = [r['steps_per_second'] for r in safebrax_results]
    ax.plot(x, y, 'o-', label='SafeBrax', linewidth=2, markersize=8, color='C0')
    
    ax.set_xlabel('Number of Parallel Environments')
    ax.set_ylabel('Steps Per Second')
    ax.set_title('Training Throughput')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Scaling efficiency
    ax = axes[0, 1]
    if len(safebrax_results) > 0:
        baseline = safebrax_results[0]['steps_per_second']
        x = [r['num_envs'] for r in safebrax_results]
        y = [r['steps_per_second'] / baseline for r in safebrax_results]
        ax.plot(x, y, 'o-', label='SafeBrax', linewidth=2, markersize=8, color='C0')
        
        # Ideal scaling line
        ax.plot(x, x, 'k--', alpha=0.5, label='Ideal scaling')
    
    ax.set_xlabel('Number of Parallel Environments')
    ax.set_ylabel('Speedup Factor')
    ax.set_title('Scaling Efficiency')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 3. CPU Utilization
    ax = axes[1, 0]
    if 'cpu_percent_avg' in safebrax_results[0]:
        x = [r['num_envs'] for r in safebrax_results]
        y_avg = [r.get('cpu_percent_avg', 0) for r in safebrax_results]
        y_max = [r.get('cpu_percent_max', 0) for r in safebrax_results]
        cpu_count = safebrax_results[0].get('cpu_count', 1)
        
        ax.plot(x, y_avg, 'o-', label='Average CPU %', linewidth=2, markersize=8, color='C2')
        ax.plot(x, y_max, 's--', label='Peak CPU %', linewidth=2, markersize=6, color='C3', alpha=0.7)
        
        # Mark CPU saturation (100% line)
        ax.axhline(y=100, color='r', linestyle=':', alpha=0.5, label='CPU Saturation')
        
        # Mark available cores
        ax.axhline(y=100 * cpu_count, color='g', linestyle=':', alpha=0.3, 
                  label=f'Max ({cpu_count} cores × 100%)')
        
        ax.set_xlabel('Number of Parallel Environments')
        ax.set_ylabel('CPU Utilization (%)')
        ax.set_title(f'CPU Usage (System has {cpu_count} cores)')
        ax.set_xscale('log', base=2)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # 4. Memory per Environment
    ax = axes[1, 1]
    if 'mem_per_env_mb' in safebrax_results[0]:
        x = [r['num_envs'] for r in safebrax_results]
        y = [r.get('mem_per_env_mb', 0) for r in safebrax_results]
        ax.plot(x, y, '^-', label='Memory per Env', linewidth=2, markersize=8, color='C4')
        
        ax.set_xlabel('Number of Parallel Environments')
        ax.set_ylabel('Memory per Environment (MB)')
        ax.set_title('Memory Efficiency')
        ax.set_xscale('log', base=2)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.suptitle('SafeBrax: Parallel Environment Benchmark', fontsize=14)
    plt.tight_layout()
    
    output_path = output_dir / 'safebrax_benchmark.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📈 Plot saved to: {output_path}")
    
    # Also save as PDF
    output_path_pdf = output_dir / 'safebrax_benchmark.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"📄 PDF saved to: {output_path_pdf}")


def generate_latex_table(results: List[Dict], output_dir: Path):
    """Generate LaTeX table for results."""
    if not results:
        return
    
    safebrax_results = [r for r in results if r['framework'] == 'SafeBrax']
    
    if not safebrax_results:
        return
    
    latex_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{SafeBrax parallel environment benchmark results}",
        r"\label{tab:safebrax_benchmark}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Envs & SPS & CPU Mem (MB) & GPU Mem (MB) & Time (s) & JIT (s) \\",
        r"\midrule"
    ]
    
    # Add results
    for r in safebrax_results:
        latex_lines.append(
            f"{r['num_envs']} & "
            f"{r['steps_per_second']:,.0f} & "
            f"{r['cpu_memory_mb']:.0f} & "
            f"{r['gpu_memory_mb']:.0f} & "
            f"{r['total_time']:.1f} & "
            f"{r['jit_time']:.1f} \\\\"
        )
    
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])
    
    # Save table
    table_path = output_dir / 'safebrax_benchmark_table.tex'
    with open(table_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"📄 LaTeX table saved to: {table_path}")


def main():
    """Run SafeBrax parallel environment benchmark."""
    print("=" * 60, flush=True)
    print("🏁 SafeBrax Parallel Environment Benchmark", flush=True)
    print("=" * 60, flush=True)
    sys.stdout.flush()
    
    # Configuration - powers of 2 up to 2048
    num_envs_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    output_dir = Path(f"safebrax_benchmark_results_{time.strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir}", flush=True)
    print(f"🔢 Testing with num_envs: {num_envs_list}", flush=True)
    sys.stdout.flush()
    
    results = []
    csv_path = output_dir / 'benchmark_results.csv'
    
    # Run benchmarks
    print("\n" + "=" * 40, flush=True)
    print("Running SafeBrax benchmarks...", flush=True)
    print("=" * 40, flush=True)
    sys.stdout.flush()
    
    for num_envs in num_envs_list:
        try:
            print(f"\n[{time.strftime('%H:%M:%S')}] Starting benchmark for {num_envs} envs...", flush=True)
            sys.stdout.flush()
            # Use consistent number of steps across all benchmarks
            num_steps = 500_000  # Total steps across all environments
            result = measure_safebrax_throughput(num_envs, num_steps=num_steps)
            if result:
                results.append(result)
                
                # Save to CSV incrementally
                file_exists = csv_path.exists()
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=result.keys())
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(result)
            
            # Brief pause between benchmarks to allow cleanup
            if num_envs < num_envs_list[-1]:  # Don't pause after last one
                time.sleep(0.5)
        except Exception as e:
            print(f"  ❌ Failed for num_envs={num_envs}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate plots and tables
    print("\n" + "=" * 40)
    print("Generating plots and tables...")
    print("=" * 40)
    
    if results:
        plot_results(results, output_dir)
        generate_latex_table(results, output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Benchmark Summary")
    print("=" * 60)
    
    if results:
        safebrax_results = [r for r in results if r['framework'] == 'SafeBrax']
        
        if safebrax_results:
            max_sps = max(r['steps_per_second'] for r in safebrax_results)
            best = max(safebrax_results, key=lambda r: r['steps_per_second'])
            cpu_count = safebrax_results[0].get('cpu_count', 1)
            
            print("\nSafeBrax:")
            print(f"  Peak throughput: {max_sps:,.0f} SPS")
            print(f"  Best config: {best['num_envs']} envs")
            
            # CPU analysis
            if 'cpu_percent_avg' in best:
                cpu_avg = best.get('cpu_percent_avg', 0)
                cpu_max = best.get('cpu_percent_max', 0)
                print(f"\n  CPU Utilization (best config):")
                print(f"    Average: {cpu_avg:.1f}%")
                print(f"    Peak: {cpu_max:.1f}%")
                print(f"    Available cores: {cpu_count}")
                
                # Check if CPU is saturated
                if cpu_max >= 95:
                    print(f"    ⚠️  CPU is saturated! Consider using fewer envs or more CPU cores")
                elif cpu_max < 50:
                    print(f"    ✓ CPU has headroom - could potentially use more envs")
                else:
                    print(f"    ⚠️  CPU is moderately utilized")
            
            # Memory analysis
            if 'mem_per_env_mb' in best:
                mem_per_env = best.get('mem_per_env_mb', 0)
                total_mem = best.get('cpu_memory_mb', 0)
                print(f"\n  Memory Usage (best config):")
                print(f"    Per environment: {mem_per_env:.0f} MB")
                print(f"    Total: {total_mem:.0f} MB")
            
            # GPU analysis
            if 'gpu_memory_mb' in best:
                gpu_mem = best.get('gpu_memory_mb', 0)
                print(f"\n  GPU Memory Usage (best config):")
                print(f"    GPU memory: {gpu_mem:.0f} MB")
            
            # Scaling analysis
            if len(safebrax_results) > 1:
                single_sps = safebrax_results[0]['steps_per_second']
                peak_speedup = max_sps / single_sps
                print(f"\n  Scaling Performance:")
                print(f"    Peak speedup: {peak_speedup:.2f}x")
                
                # Efficiency (how close to linear scaling)
                ideal_sps = single_sps * best['num_envs']
                efficiency = max_sps / ideal_sps * 100
                print(f"    Scaling efficiency: {efficiency:.1f}%")
                
                # Find where scaling starts to degrade
                if len(safebrax_results) >= 3:
                    speedups = [r['steps_per_second'] / single_sps for r in safebrax_results]
                    # Find first point where speedup improvement < 10%
                    for i in range(1, len(speedups)):
                        improvement = (speedups[i] - speedups[i-1]) / speedups[i-1] * 100
                        if improvement < 10:
                            print(f"\n  💡 Scaling Recommendation:")
                            print(f"    Diminishing returns start at {safebrax_results[i-1]['num_envs']} envs")
                            print(f"    Optimal likely between {safebrax_results[i-1]['num_envs']}-{safebrax_results[i]['num_envs']} envs")
                            break
    
    print("\n✅ Benchmark complete!")
    print(f"📁 Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()

