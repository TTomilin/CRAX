"""Plot performance comparison between CRAX and Safety-Gymnasium."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configure matplotlib
plt.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Data paths
RESULTS_DIR = Path(__file__).parent.parent / "data" / "performance"

# Load CRAX data from all sources and combine
crax_main = pd.read_csv(RESULTS_DIR / "crax_benchmark_results_20260129_044133" / "benchmark_results.csv")
crax_extra = pd.read_csv(RESULTS_DIR / "crax_benchmark_results_20260129_032509" / "benchmark_results.csv")
crax_large = pd.read_csv(RESULTS_DIR / "crax_benchmark_results_20260129_082641" / "benchmark_results.csv")
crax_df = pd.concat([crax_extra, crax_main, crax_large], ignore_index=True).drop_duplicates(
    subset=['num_envs']).sort_values('num_envs')

# Load Safety-Gymnasium data
safety_gym_df = pd.read_csv(RESULTS_DIR / "safety_gym_benchmark_results_20260129_054930" / "benchmark_results.csv")

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)


def plot_comparison():
    """Plot throughput comparison and scaling efficiency."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    # Left plot: Throughput comparison (log-log)
    ax1.plot(crax_df['num_envs'], crax_df['steps_per_second'],
             'o-', color='#2E86AB', linewidth=2.5, markersize=8, label='CRAX (Ours)')
    ax1.plot(safety_gym_df['num_envs'], safety_gym_df['steps_per_second'],
             's-', color='#E94F37', linewidth=2.5, markersize=8, label='Safety-Gymnasium')

    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of Parallel Environments')
    ax1.set_ylabel('Steps per Second (SPS)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, which='both')

    # Set x-axis ticks for left plot
    all_envs = sorted(set(crax_df['num_envs']) | set(safety_gym_df['num_envs']))
    ax1.set_xticks(all_envs)
    ax1.set_xticklabels([str(int(x)) for x in all_envs], rotation=45)

    # Right plot: Scaling Efficiency
    # Calculate speedup relative to single env performance
    crax_base_sps = crax_df[crax_df['num_envs'] == 1]['steps_per_second'].values[0]
    safety_base_sps = safety_gym_df[safety_gym_df['num_envs'] == 1]['steps_per_second'].values[0]

    crax_speedup = crax_df['steps_per_second'] / crax_base_sps
    safety_speedup = safety_gym_df['steps_per_second'] / safety_base_sps

    ax2.plot(crax_df['num_envs'], crax_speedup,
             'o-', color='#2E86AB', linewidth=2.5, markersize=8, label='CRAX (Ours)')
    ax2.plot(safety_gym_df['num_envs'], safety_speedup,
             's-', color='#E94F37', linewidth=2.5, markersize=8, label='Safety-Gymnasium')

    # Ideal scaling line (linear)
    max_envs = max(crax_df['num_envs'].max(), safety_gym_df['num_envs'].max())
    ideal_x = np.array([1, max_envs])
    ideal_y = ideal_x
    ax2.plot(ideal_x, ideal_y, '--', color='gray', linewidth=2, alpha=0.7, label='Ideal Scaling')

    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log', base=2)
    ax2.set_xlabel('Number of Parallel Environments')
    ax2.set_ylabel('Speedup Factor')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, which='both')

    # Set x-axis ticks for right plot
    ax2.set_xticks(all_envs)
    ax2.set_xticklabels([str(int(x)) for x in all_envs], rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'throughput_comparison.pdf')
    print(f"Saved to {OUTPUT_DIR / 'throughput_comparison.pdf'}")
    plt.show()
    plt.close()


if __name__ == "__main__":
    plot_comparison()
