#!/usr/bin/env python3
"""
Generate 2 plots used in the observation routing ablation.
"""

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

CONDITIONS = ['none', 'cost', 'all_critics', 'reward', 'state_oracle']
CONDITION_LABELS = {
    'none': 'Ego baseline',
    'cost': 'Cost privilege',
    'all_critics': 'All-critic privilege',
    'reward': 'Reward privilege',
    'state_oracle': 'State cost privilege',
}
CONDITION_COLORS = {
    'none': '#2F2F2F',
    'cost': '#D55E00',
    'all_critics': '#0072B2',
    'reward': '#009E73',
    'state_oracle': '#CC79A7',
}

DEFAULT_COST_BUDGET = 25.0
DEFAULT_LAMBDA_RUNAWAY = 200.0

_BANNER = re.compile(r'condition=(\w+)\s+seed=(\d+)')
_STEP = re.compile(r'^Step (\d+):\s*$')
_NUMBER = r'[-+]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][-+]?\d+)?'
_KV = re.compile(rf'^\s+([\w/]+):\s+({_NUMBER}|nan|[-+]?inf)\s*$')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-seed privileged-vision analysis")
    p.add_argument("log_dirs", type=str, nargs='+',
                   help="Directories containing vision_privilege_*.out files")
    p.add_argument("--out", type=str, default=None,
                   help="Output directory for the JSON/markdown/figure")
    p.add_argument("--cost_budget", type=float, default=DEFAULT_COST_BUDGET)
    p.add_argument("--lambda_runaway", type=float, default=DEFAULT_LAMBDA_RUNAWAY)
    p.add_argument("--summary_start_step", type=int, default=75_000_000,
                   help="First environment step included in training summaries")
    p.add_argument("--summary_end_step", type=int, default=100_007_936,
                   help="Last environment step included in training summaries")
    p.add_argument("--max_step", type=int, default=None,
                   help="Discard records beyond this env step")
    p.add_argument("--common_prefix", action='store_true',
                   help="Truncate every run to the shortest run's last step. Use "
                        "when some tasks hit the wall clock: comparing a "
                        "condition that trained for 100M steps against one "
                        "stopped at 88M would confound the condition with the "
                        "training budget.")
    p.add_argument("--smooth", type=int, default=8,
                   help="Window size for moving average smoothing across steps")
    p.add_argument("--paper_dir", type=str, default="paper/figures",
                   help="Directory in paper to copy generated figures to (or '' to disable)")
    p.add_argument("--no_plot", action='store_true')
    return p.parse_args()


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """Smooth data with a simple moving average that handles boundaries correctly."""
    if window_size <= 1:
        return data
    data_sum = np.convolve(data, np.ones(window_size), 'same')
    counts = np.convolve(np.ones_like(data), np.ones(window_size), 'same')
    return data_sum / counts


def parse_log(path: Path) -> Optional[Dict]:
    """-> {'condition','seed','path','recs':[{'step':int, metric:float}]}"""
    condition = seed = None
    recs: List[Dict[str, float]] = []
    current: Optional[Dict[str, float]] = None
    with path.open(errors='replace') as fh:
        for line in fh:
            m = _BANNER.search(line)
            if m:
                if condition is not None:
                    recs = []
                    current = None
                condition, seed = m.group(1), int(m.group(2))
                continue
            m = _STEP.match(line)
            if m:
                current = {'step': int(m.group(1))}
                recs.append(current)
                continue
            if current is not None:
                m = _KV.match(line)
                if m:
                    try:
                        current[m.group(1)] = float(m.group(2))
                    except ValueError:
                        pass
                elif line.strip() and not line.startswith(' '):
                    current = None
    if condition is None or not recs:
        return None
    for record in recs:
        if any(key.startswith('eval/') for key in record):
            record['kind'] = 'evaluation'
        elif any(key.startswith('episodic/') for key in record):
            record['kind'] = 'interval'
        else:
            record['kind'] = 'epoch_summary'
    return {'condition': condition, 'seed': seed, 'path': str(path), 'recs': recs}


def series(recs: Sequence[Dict[str, float]], key: str,
           kind: Optional[str] = None,
           start_step: Optional[int] = None,
           end_step: Optional[int] = None):
    steps, values = [], []
    for r in recs:
        if kind is not None and r.get('kind') != kind:
            continue
        if start_step is not None and r['step'] < start_step:
            continue
        if end_step is not None and r['step'] > end_step:
            continue
        if key in r and np.isfinite(r[key]):
            steps.append(r['step'])
            values.append(r[key])
    return np.asarray(steps, dtype=np.int64), np.asarray(values, dtype=np.float64)


def _first_crossing(steps: np.ndarray, values: np.ndarray,
                    threshold: float) -> Optional[int]:
    idx = np.nonzero(values > threshold)[0]
    return int(steps[idx[0]]) if idx.size else None


def summarise_run(run: Dict, args: argparse.Namespace) -> Dict:
    recs = run['recs']
    out = {'condition': run['condition'], 'seed': run['seed'], 'path': run['path']}
    for name in ('reward', 'cost'):
        _, training_values = series(
            recs, f'episodic/{name}', kind='interval',
            start_step=args.summary_start_step,
            end_step=args.summary_end_step,
        )
        out[f'training_{name}_mean'] = (
            float(training_values.mean()) if training_values.size else float('nan')
        )
        _, eval_values = series(recs, f'eval/episode_{name}', kind='evaluation')
        out[f'eval_{name}'] = (
            float(eval_values[-1]) if eval_values.size else float('nan')
        )

    steps, lam = series(recs, 'training/lambda_lagr', kind='interval')
    out['lambda_max'] = float(lam.max()) if lam.size else float('nan')
    out['lambda_runaway_step'] = _first_crossing(steps, lam, args.lambda_runaway)

    for key, name in (('training/cost_value_ev', 'cost_value_ev'),
                      ('training/value_ev', 'value_ev'),
                      ('training/cost_v_loss', 'cost_v_loss')):
        _, values = series(
            recs, key, kind='interval',
            start_step=args.summary_start_step,
            end_step=args.summary_end_step,
        )
        out[name] = float(values.mean()) if values.size else None

    out['last_step'] = int(recs[-1]['step'])
    return out


def aggregate(runs: List[Dict], args: argparse.Namespace) -> Dict[str, Dict]:
    by_condition: Dict[str, List[Dict]] = defaultdict(list)
    for run in runs:
        by_condition[run['condition']].append(run)

    agg = {}
    for condition, group in by_condition.items():
        cost = np.array([g['eval_cost'] for g in group], dtype=float)
        reward = np.array([g['eval_reward'] for g in group], dtype=float)
        runaway = [g['lambda_runaway_step'] for g in group]
        evs = [g['cost_value_ev'] for g in group if g['cost_value_ev'] is not None]
        r_evs = [g['value_ev'] for g in group if g['value_ev'] is not None]
        agg[condition] = {
            'n_seeds': len(group),
            'seeds': sorted(g['seed'] for g in group),
            'cost_mean': float(np.nanmean(cost)),
            'cost_std': float(np.nanstd(cost, ddof=1)) if len(cost) > 1 else 0.0,
            'cost_worst': float(np.nanmax(cost)),
            'reward_mean': float(np.nanmean(reward)),
            'reward_std': float(np.nanstd(reward, ddof=1)) if len(reward) > 1 else 0.0,
            'training_cost_mean': float(np.nanmean(
                [g['training_cost_mean'] for g in group])),
            'training_reward_mean': float(np.nanmean(
                [g['training_reward_mean'] for g in group])),
            # Safe% over seeds
            'safe_pct': 100.0 * float(np.mean(cost <= args.cost_budget)),
            'n_runaway': int(sum(s is not None for s in runaway)),
            'runaway_steps': runaway,
            'lambda_max': float(np.nanmax([g['lambda_max'] for g in group])),
            'cost_value_ev': float(np.mean(evs)) if evs else None,
            'value_ev': float(np.mean(r_evs)) if r_evs else None,
            'truncated': [g['seed'] for g in group
                          if g['last_step'] < 0.98 * max(h['last_step'] for h in group)],
        }
    return agg


def markdown_table(agg: Dict[str, Dict], args: argparse.Namespace) -> str:
    lines = [
        f"| Condition | seeds | final eval return | final eval cost | worst cost | safe seeds (budget {args.cost_budget:g}) "
        f"| runaway seeds | max lambda | cost EV | reward EV |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        a = agg.get(condition)
        if a is None:
            continue
        ev = '--' if a['cost_value_ev'] is None else f"{a['cost_value_ev']:.3f}"
        rev = '--' if a['value_ev'] is None else f"{a['value_ev']:.3f}"
        lines.append(
            f"| {CONDITION_LABELS[condition]} | {a['n_seeds']} "
            f"| {a['reward_mean']:.2f} $\pm$ {a['reward_std']:.2f} "
            f"| {a['cost_mean']:.1f} $\pm$ {a['cost_std']:.1f} | {a['cost_worst']:.1f} "
            f"| {a['safe_pct']:.0f}% | {a['n_runaway']}/{a['n_seeds']} "
            f"| {a['lambda_max']:.0f} | {ev} | {rev} |"
        )
    return "\n".join(lines)


def seed_markdown_table(summaries: Sequence[Dict], args: argparse.Namespace) -> str:
    lines = [
        "| Condition | Seed | Final eval return | Final eval cost | Safe | "
        "Training return | Training cost | Runaway |",
        "|---|---:|---:|---:|:---:|---:|---:|:---:|",
    ]
    order = {condition: index for index, condition in enumerate(CONDITIONS)}
    for run in sorted(summaries, key=lambda r: (order.get(r['condition'], 99), r['seed'])):
        runaway = '--' if run['lambda_runaway_step'] is None else f"{run['lambda_runaway_step']:,}"
        lines.append(
            f"| {CONDITION_LABELS.get(run['condition'], run['condition'])} "
            f"| {run['seed']} | {run['eval_reward']:.2f} | {run['eval_cost']:.2f} "
            f"| {'yes' if run['eval_cost'] <= args.cost_budget else 'no'} "
            f"| {run['training_reward_mean']:.2f} | {run['training_cost_mean']:.2f} "
            f"| {runaway} |"
        )
    return "\n".join(lines)


def plot(runs: List[Dict], out_dir: Path,
         args: argparse.Namespace) -> List[Path]:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"plotting skipped ({exc})")
        return []

    panel_groups = [
      ('vision_privilege_learning_curves', [
        ('episodic/reward', 'Episodic return', None, None, False),
        ('episodic/cost', 'Episodic cost', args.cost_budget, None, False),
      ]),
      ('vision_privilege_diagnostics', [
        ('training/lambda_lagr', 'Lagrange multiplier', None, None, True),
        ('training/cost_value_ev', 'Cost-critic explained variance', 0.0, (-0.3, 1.05), False),
        ('training/value_ev', 'Reward-critic explained variance', 0.0, (-0.3, 1.05), False),
      ]),
    ]
    by_condition: Dict[str, List[Dict]] = defaultdict(list)
    for run in runs:
        by_condition[run['condition']].append(run)
    outputs = []
    smooth = getattr(args, 'smooth', 8)
    for stem, panels in panel_groups:
        fig_width = 5.4 * len(panels)
        fig, axes = plt.subplots(1, len(panels), figsize=(fig_width, 4.2))
        axes = np.atleast_1d(axes)
        for ax, (key, ylabel, hline, ylim, is_lambda) in zip(axes, panels):
            for condition in CONDITIONS:
                c_runs = by_condition.get(condition, [])
                seed_data = []
                for r in c_runs:
                    s, v = series(r['recs'], key, kind='interval')
                    if len(s) > 1:
                        seed_data.append((s, v))
                if not seed_data:
                    continue

                # Common step grid (longest series)
                grid_steps = max((s for s, _ in seed_data), key=len)

                aligned = []
                for s, v in seed_data:
                    if not is_lambda and 'ev' in key:
                        v = np.clip(v, -1.0, 1.0)
                    interp_v = np.interp(grid_steps, s, v)
                    if smooth > 1:
                        interp_v = moving_average(interp_v, smooth)
                    aligned.append(interp_v)
                stacked = np.stack(aligned, axis=0)

                color = CONDITION_COLORS[condition]
                label = CONDITION_LABELS[condition]

                if is_lambda:
                    log_v = np.log10(np.maximum(1e-3, stacked))
                    mean_log = np.mean(log_v, axis=0)
                    std_log = np.std(log_v, axis=0)
                    mean = 10 ** mean_log
                    lower = 10 ** (mean_log - std_log)
                    upper = 10 ** (mean_log + std_log)
                else:
                    mean = np.mean(stacked, axis=0)
                    std = np.std(stacked, axis=0)
                    lower = mean - std
                    upper = mean + std

                ax.plot(grid_steps, mean, label=label, color=color, linewidth=2.0)
                ax.fill_between(grid_steps, lower, upper, color=color, alpha=0.18)

            if hline is not None:
                ax.axhline(hline, color='r', linestyle='--', linewidth=1.2)
            ax.axvspan(args.summary_start_step, args.summary_end_step,
                       color='#BBBBBB', alpha=0.15)
            ax.set_xlabel('Environment steps', fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.tick_params(labelsize=10)
            if is_lambda:
                ax.set_yscale('log')
            if ylim is not None:
                ax.set_ylim(ylim)

        handles = [plt.Line2D([], [], color=CONDITION_COLORS[c],
                              label=CONDITION_LABELS[c], linewidth=2.0)
                   for c in CONDITIONS if c in by_condition]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.06),
                   ncol=len(handles), frameon=True, fontsize=10)
        fig.tight_layout()
        for suffix in ('png', 'pdf'):
            output = out_dir / f'{stem}.{suffix}'
            fig.savefig(output, bbox_inches='tight', dpi=200)
            outputs.append(output)
        plt.close(fig)
    return outputs


def main() -> None:
    args = parse_args()
    runs = []
    for d in args.log_dirs:
        directory = Path(d)
        if not directory.exists():
            print(f"warning: {directory} does not exist", file=sys.stderr)
            continue
        for path in sorted(directory.glob('vision_privilege_*.out')):
            parsed = parse_log(path)
            if parsed is None:
                print(f"warning: no usable records in {path.name}", file=sys.stderr)
                continue
            runs.append(parsed)
    if not runs:
        raise SystemExit('No parsable vision_privilege_*.out logs found.')

    # Log directories are ordered oldest to newest.
    unique_runs = {}
    for run in runs:
        key = (run['condition'], run['seed'])
        if key in unique_runs:
            print(
                f"warning: replacing duplicate {key[0]} seed {key[1]}: "
                f"{unique_runs[key]['path']} -> {run['path']}",
                file=sys.stderr,
            )
        unique_runs[key] = run
    runs = list(unique_runs.values())

    limit = args.max_step
    if args.common_prefix:
        shortest = min(r['recs'][-1]['step'] for r in runs)
        limit = shortest if limit is None else min(limit, shortest)
        print(f"common prefix: truncating every run at step {shortest:,}")
    if limit is not None:
        for r in runs:
            r['recs'] = [rec for rec in r['recs'] if rec['step'] <= limit]
        runs = [r for r in runs if r['recs']]

    summaries = [summarise_run(r, args) for r in runs]
    agg = aggregate(summaries, args)

    print(f"\nParsed {len(runs)} runs:")
    for s in sorted(summaries, key=lambda s: (s['condition'], s['seed'])):
        print(f"  {s['condition']:14s} seed {s['seed']}  "
              f"last step {s['last_step']:>10,}  eval cost {s['eval_cost']:7.1f}  "
              f"eval return {s['eval_reward']:6.2f}  lambda_max {s['lambda_max']:8.0f}")
    print()
    table = markdown_table(agg, args)
    print(table)
    seed_table = seed_markdown_table(summaries, args)

    out_dir = Path(args.out) if args.out else Path('results/vision_privilege/pooled')
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'table.json').write_text(json.dumps(
        {'conditions': agg, 'runs': summaries,
         'cost_budget': args.cost_budget,
         'lambda_runaway_threshold': args.lambda_runaway,
         'summary_step_window': [args.summary_start_step,
                                 args.summary_end_step]}, indent=2))
    (out_dir / 'table.md').write_text(table + "\n")
    (out_dir / 'final_table.md').write_text(seed_table + "\n")
    if not args.no_plot:
        for fig_path in plot(runs, out_dir, args):
            print(f"\nWrote {fig_path}")
        paper_dir = Path(args.paper_dir) if args.paper_dir else None
        if paper_dir and paper_dir.exists():
            for fig_name in ('vision_privilege_learning_curves.pdf', 'vision_privilege_diagnostics.pdf',
                             'vision_privilege_learning_curves.png', 'vision_privilege_diagnostics.png'):
                src = out_dir / fig_name
                if src.exists():
                    shutil.copy2(src, paper_dir / fig_name)
                    print(f"Copied {src} -> {paper_dir / fig_name}")
    print(f"{out_dir}")


if __name__ == "__main__":
    main()
