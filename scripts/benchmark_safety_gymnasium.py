#!/usr/bin/env python3
"""
Benchmark script for Safety Gymnasium with parallel environments.
Benchmarks Safety Gymnasium using SafetyAsyncVectorEnv for parallel execution.
"""

import time
import os
import csv
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt

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


def measure_safety_gymnasium_throughput(num_envs: int, num_steps: int = 100_000) -> Dict:
    """Measure Safety-Gymnasium throughput with parallel environments.
    
    Args:
        num_envs: Number of parallel environments.
        num_steps: Total environment steps across all environments.
    
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
        env = safety_gymnasium.make('SafetyPointGoal1-v0')
        use_vector = False
    else:
        # Parallel environments using SafetyAsyncVectorEnv
        env_fns = [lambda: safety_gymnasium.make('SafetyPointGoal1-v0') for _ in range(num_envs)]
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


def plot_results(results: List[Dict], output_dir: Path):
    """Generate comparison plots."""
    if not results:
        print("No results to plot")
        return
    
    # Set style
    plt.style.use('seaborn-v0_8-paper')
    
    # Prepare data
    safety_results = [r for r in results if r['framework'] == 'Safety-Gymnasium']
    
    if not safety_results:
        print("No Safety-Gymnasium results to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Throughput scaling
    ax = axes[0, 0]
    x = [r['num_envs'] for r in safety_results]
    y = [r['steps_per_second'] for r in safety_results]
    ax.plot(x, y, 's-', label='Safety-Gymnasium', linewidth=2, markersize=8, color='C1')
    
    ax.set_xlabel('Number of Parallel Environments')
    ax.set_ylabel('Steps Per Second')
    ax.set_title('Training Throughput')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Scaling efficiency
    ax = axes[0, 1]
    if len(safety_results) > 0:
        baseline = safety_results[0]['steps_per_second']
        x = [r['num_envs'] for r in safety_results]
        y = [r['steps_per_second'] / baseline for r in safety_results]
        ax.plot(x, y, 's-', label='Safety-Gymnasium', linewidth=2, markersize=8, color='C1')
        
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
    if 'cpu_percent_avg' in safety_results[0]:
        x = [r['num_envs'] for r in safety_results]
        y_avg = [r.get('cpu_percent_avg', 0) for r in safety_results]
        y_max = [r.get('cpu_percent_max', 0) for r in safety_results]
        cpu_count = safety_results[0].get('cpu_count', 1)
        
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
    if 'mem_per_env_mb' in safety_results[0]:
        x = [r['num_envs'] for r in safety_results]
        y = [r.get('mem_per_env_mb', 0) for r in safety_results]
        ax.plot(x, y, '^-', label='Memory per Env', linewidth=2, markersize=8, color='C4')
        
        ax.set_xlabel('Number of Parallel Environments')
        ax.set_ylabel('Memory per Environment (MB)')
        ax.set_title('Memory Efficiency')
        ax.set_xscale('log', base=2)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.suptitle('Safety-Gymnasium: Parallel Environment Benchmark', fontsize=14)
    plt.tight_layout()
    
    output_path = output_dir / 'safety_gym_benchmark.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📈 Plot saved to: {output_path}")
    
    # Also save as PDF
    output_path_pdf = output_dir / 'safety_gym_benchmark.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"📄 PDF saved to: {output_path_pdf}")


def generate_latex_table(results: List[Dict], output_dir: Path):
    """Generate LaTeX table for results."""
    if not results:
        return
    
    safety_results = [r for r in results if r['framework'] == 'Safety-Gymnasium']
    
    if not safety_results:
        return
    
    latex_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Safety-Gymnasium parallel environment benchmark results}",
        r"\label{tab:safety_gym_benchmark}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Envs & SPS & CPU Mem (MB) & GPU Mem (MB) & Time (s) \\",
        r"\midrule"
    ]
    
    # Add results
    for r in safety_results:
        latex_lines.append(
            f"{r['num_envs']} & "
            f"{r['steps_per_second']:,.0f} & "
            f"{r['cpu_memory_mb']:.0f} & "
            f"{r['gpu_memory_mb']:.0f} & "
            f"{r['total_time']:.1f} \\\\"
        )
    
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])
    
    # Save table
    table_path = output_dir / 'safety_gym_benchmark_table.tex'
    with open(table_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"📄 LaTeX table saved to: {table_path}")


def main():
    """Run Safety Gymnasium parallel environment benchmark."""
    print("=" * 60)
    print("🏁 Safety Gymnasium Parallel Environment Benchmark")
    print("=" * 60)
    
    # Configuration
    num_envs_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]  # Powers of 2 up to 16
    output_dir = Path(f"safety_gym_benchmark_results_{time.strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir}")
    print(f"🔢 Testing with num_envs: {num_envs_list}")
    
    results = []
    csv_path = output_dir / 'benchmark_results.csv'
    
    # Run benchmarks
    print("\n" + "=" * 40)
    print("Running Safety-Gymnasium benchmarks...")
    print("=" * 40)
    
    for num_envs in num_envs_list:
        try:
            # Use consistent number of steps across all benchmarks (same as SafeBrax for fair comparison)
            num_steps = 500_000  # Total steps across all environments
            result = measure_safety_gymnasium_throughput(num_envs, num_steps=num_steps)
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
        safety_results = [r for r in results if r['framework'] == 'Safety-Gymnasium']
        
        if safety_results:
            max_sps = max(r['steps_per_second'] for r in safety_results)
            best = max(safety_results, key=lambda r: r['steps_per_second'])
            cpu_count = safety_results[0].get('cpu_count', 1)
            
            print("\nSafety-Gymnasium:")
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
            
            # Scaling analysis
            if len(safety_results) > 1:
                single_sps = safety_results[0]['steps_per_second']
                peak_speedup = max_sps / single_sps
                print(f"\n  Scaling Performance:")
                print(f"    Peak speedup: {peak_speedup:.2f}x")
                
                # Efficiency (how close to linear scaling)
                ideal_sps = single_sps * best['num_envs']
                efficiency = max_sps / ideal_sps * 100
                print(f"    Scaling efficiency: {efficiency:.1f}%")
                
                # Find where scaling starts to degrade
                if len(safety_results) >= 3:
                    speedups = [r['steps_per_second'] / single_sps for r in safety_results]
                    # Find first point where speedup improvement < 10%
                    for i in range(1, len(speedups)):
                        improvement = (speedups[i] - speedups[i-1]) / speedups[i-1] * 100
                        if improvement < 10:
                            print(f"\n  💡 Scaling Recommendation:")
                            print(f"    Diminishing returns start at {safety_results[i-1]['num_envs']} envs")
                            print(f"    Optimal likely between {safety_results[i-1]['num_envs']}-{safety_results[i]['num_envs']} envs")
                            break
    
    print("\n✅ Benchmark complete!")
    print(f"📁 Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()

