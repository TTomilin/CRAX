"""Throughput (SPS vs. parallel envs) of CRAX and Safety-Gymnasium across several tasks."""
import argparse
import re
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from results.common import results_path

plt.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# (panel title, CRAX env, Safety-Gymnasium env id)
TASKS = [
    ("Point Goal", "safe_goal_point", "SafetyPointGoal1-v0"),
    ("Ant Velocity", "safe_velocity_ant", "SafetyAntVelocity-v1"),
    ("Humanoid Velocity", "safe_velocity_humanoid", "SafetyHumanoidVelocity-v1"),
]
LEGACY_ENV = {"crax": "safe_goal_point", "safety_gym": "SafetyPointGoal1-v0"}

STYLE = {
    "crax": dict(marker='o', color='#2E86AB', label='CRAX (Ours)'),
    "safety_gym": dict(marker='s', color='#E94F37', label='Safety-Gymnasium'),
}


def load_runs(data_dir: Path, prefix: str, env: str) -> Optional[pd.DataFrame]:
    """Merge all `<prefix>_benchmark_results_*` folders that belong to `env`."""
    pattern = re.compile(rf"^{prefix}_benchmark_results_(?:(?P<env>.+?)_)?(?:n\d+_)?\d{{8}}_\d{{6}}$")
    frames = []
    for d in sorted(data_dir.iterdir()):  # sorted by name -> timestamp order within an env
        m = pattern.match(d.name)
        csv = d / "benchmark_results.csv"
        if not m or not csv.exists():
            continue
        if (m.group("env") or LEGACY_ENV[prefix]) == env:
            frames.append(pd.read_csv(csv))
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=["num_envs"], keep="last").sort_values("num_envs")


def plot(data_dir: Path, out_path: Path, tasks: List[tuple]) -> None:
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.8), sharey=True)
    axes = [axes] if len(tasks) == 1 else list(axes)

    for ax, (title, crax_env, sg_env) in zip(axes, tasks):
        xs = set()
        for prefix, env in (("crax", crax_env), ("safety_gym", sg_env)):
            df = load_runs(data_dir, prefix, env)
            if df is None:
                print(f"No {prefix} results for {env!r} in {data_dir}")
                continue
            ax.plot(df['num_envs'], df['steps_per_second'], '-', linewidth=2.2, markersize=6, **STYLE[prefix])
            xs |= set(df['num_envs'])

        ax.set_title(title)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel('Parallel Environments')
        ax.grid(True, alpha=0.3, which='both')
        if xs:
            ticks = [x for x in sorted(xs) if x in (1, 4, 16, 64, 256, 1024, 4096, 16384)]
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{x // 1024}K" if x >= 1024 else str(x) for x in ticks])

    axes[0].set_ylabel('Steps per Second (SPS)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 1.0), frameon=False)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"Saved to {out_path}")
    plt.show()
    plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", type=str, default=str(results_path("data", "performance")))
    p.add_argument("--out", type=str, default=str(results_path("figures", "throughput_multi_env.pdf")))
    args = p.parse_args()
    plot(Path(args.data_dir), Path(args.out), TASKS)
