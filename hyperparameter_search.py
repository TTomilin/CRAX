"""
Hyperparameter search script for Safe-Brax PPO-Lagrange.

Usage:
    python hyperparameter_search.py --base_config CONFIG.json --output_dir results/hparam_search
"""

import argparse
import itertools
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


def create_search_space() -> Dict[str, List[Any]]:
    """Define the hyperparameter search space.
    
    Modify these values to adjust your search.
    """
    return {
        # Batch configuration
        'batch_size': [512, 1024, 2048],
        'num_minibatches': [16, 32, 64],
        'num_updates_per_batch': [4, 6, 8],
        
        # Learning rates
        'learning_rate': [1e-4, 3e-4, 5e-4, 1e-3],
        
        # PPO-Lagrange specific
        'lagrangian_coef_rate': [0.001, 0.01, 0.05, 0.1],
    }


def create_focused_search_space() -> Dict[str, List[Any]]:
    """Smaller search space for faster overnight runs.
    
    This is a curated subset focusing on likely good values.
    """
    return {
        'batch_size': [1024, 2048],
        'num_minibatches': [32, 64],
        'num_updates_per_batch': [6, 8],
        'learning_rate': [3e-4, 5e-4],
        'lagrangian_coef_rate': [0.001, 0.01],
    }


def create_random_search_configs(base_config: Dict, search_space: Dict, n_samples: int, seed: int = 0) -> List[Dict]:
    """Create random search configurations.
    
    Args:
        base_config: Base configuration to modify
        search_space: Dictionary of parameter: [values] to search
        n_samples: Number of random configurations to generate
        seed: Random seed for reproducibility
    
    Returns:
        List of configuration dictionaries
    """
    import random
    random.seed(seed)
    
    configs = []
    for i in range(n_samples):
        config = base_config.copy()
        config['hparam_search_id'] = i
        
        # Randomly sample one value from each parameter
        for param, values in search_space.items():
            config[param] = random.choice(values)
        
        configs.append(config)
    
    return configs


def create_grid_search_configs(base_config: Dict, search_space: Dict) -> List[Dict]:
    """Create grid search configurations (all combinations).
    
    Args:
        base_config: Base configuration to modify
        search_space: Dictionary of parameter: [values] to search
    
    Returns:
        List of configuration dictionaries
    """
    # Get all parameter names and their values
    param_names = list(search_space.keys())
    param_values = [search_space[name] for name in param_names]
    
    configs = []
    for i, combination in enumerate(itertools.product(*param_values)):
        config = base_config.copy()
        config['hparam_search_id'] = i
        
        # Set each parameter value
        for param_name, param_value in zip(param_names, combination):
            config[param_name] = param_value
        
        configs.append(config)
    
    return configs


def save_config(config: Dict, output_path: str) -> None:
    """Save configuration to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)


def run_training(config_path: str, seeds: List[int], log_dir: str) -> Dict[str, Any]:
    """Run training with the given configuration.
    
    Args:
        config_path: Path to configuration file
        seeds: List of random seeds to run
        log_dir: Directory to save logs
    
    Returns:
        Dictionary with run information
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{Path(config_path).stem}_{timestamp}.log")
    
    # Build command
    cmd = [
        "python", "train_from_config.py",
        "--config", config_path,
        "--seeds"] + [str(s) for s in seeds] + [
        "--quiet"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print(f"Log file: {log_file}")
    
    start_time = time.time()
    
    try:
        # Run training and capture output
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=os.getcwd()
            )
        
        elapsed_time = time.time() - start_time
        
        return {
            'config_path': config_path,
            'log_file': log_file,
            'return_code': result.returncode,
            'elapsed_time': elapsed_time,
            'success': result.returncode == 0,
            'seeds': seeds
        }
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"Error running training: {e}")
        
        return {
            'config_path': config_path,
            'log_file': log_file,
            'return_code': -1,
            'elapsed_time': elapsed_time,
            'success': False,
            'error': str(e),
            'seeds': seeds
        }


def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter search')
    
    # Core arguments
    parser.add_argument('--base_config', type=str, required=True,
                        help='Path to base configuration file')
    parser.add_argument('--output_dir', type=str, default='results/hparam_search',
                        help='Directory to save configurations and results')
    
    # Search strategy
    parser.add_argument('--search_type', type=str, default='focused',
                        choices=['grid', 'random', 'focused'],
                        help='Type of search: grid (all combinations), random (sample), or focused (curated subset)')
    parser.add_argument('--n_random_samples', type=int, default=20,
                        help='Number of random samples (only for random search)')
    parser.add_argument('--random_seed', type=int, default=0,
                        help='Random seed for random search')
    
    # Training arguments
    parser.add_argument('--seeds', type=int, nargs='+', default=[0],
                        help='Random seeds to run for each configuration')
    parser.add_argument('--dry_run', action='store_true',
                        help='Generate configs but do not run training')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    configs_dir = output_dir / 'configs'
    configs_dir.mkdir(exist_ok=True)
    
    logs_dir = output_dir / 'logs'
    logs_dir.mkdir(exist_ok=True)
    
    # Load base configuration
    with open(args.base_config, 'r') as f:
        base_config = json.load(f)
    
    # Create search space
    if args.search_type == 'focused':
        search_space = create_focused_search_space()
    else:
        search_space = create_search_space()
    
    # Generate configurations
    print(f"\n{'='*60}")
    print(f"Hyperparameter Search Configuration")
    print(f"{'='*60}")
    print(f"Base config: {args.base_config}")
    print(f"Search type: {args.search_type}")
    print(f"Output dir: {output_dir}")
    print(f"\nSearch space:")
    for param, values in search_space.items():
        print(f"  {param}: {values}")
    
    if args.search_type == 'grid':
        configs = create_grid_search_configs(base_config, search_space)
    elif args.search_type == 'random':
        configs = create_random_search_configs(base_config, search_space, 
                                                args.n_random_samples, args.random_seed)
    else:  # focused
        configs = create_grid_search_configs(base_config, search_space)
    
    total_combinations = len(configs)
    total_runs = total_combinations * len(args.seeds)
    
    print(f"\nTotal configurations: {total_combinations}")
    print(f"Seeds per config: {len(args.seeds)}")
    print(f"Total training runs: {total_runs}")
    
    # Save all configurations
    config_paths = []
    for i, config in enumerate(configs):
        # Update wandb group for this search
        if 'wandb_group' in config:
            config['wandb_group'] = f"hparam_search_{args.search_type}"
        
        # Add hparam identifier to tags
        if 'wandb_tags' in config:
            config['wandb_tags'] = config['wandb_tags'] + [f"hparam_id_{i}"]
        
        config_name = f"config_{i:04d}.json"
        config_path = configs_dir / config_name
        save_config(config, str(config_path))
        config_paths.append(str(config_path))
        
        if i < 3:  # Print first few configs
            print(f"\nConfig {i}:")
            for param in search_space.keys():
                print(f"  {param}: {config[param]}")
    
    # Save search summary
    summary = {
        'search_type': args.search_type,
        'base_config': args.base_config,
        'search_space': search_space,
        'total_configs': total_combinations,
        'seeds': args.seeds,
        'total_runs': total_runs,
        'timestamp': datetime.now().isoformat(),
        'config_paths': config_paths
    }
    
    summary_path = output_dir / 'search_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Configurations saved to: {configs_dir}")
    print(f"Search summary saved to: {summary_path}")
    print(f"{'='*60}\n")
    
    if args.dry_run:
        print("Dry run - not executing training.")
        return
    
    # Run training for each configuration
    print("Starting hyperparameter search...\n")
    
    results = []
    start_time = time.time()
    
    for i, config_path in enumerate(config_paths):
        print(f"\n{'='*60}")
        print(f"Configuration {i+1}/{total_combinations}")
        print(f"{'='*60}")
        
        result = run_training(config_path, args.seeds, str(logs_dir))
        results.append(result)
        
        # Save intermediate results
        results_path = output_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        if result['success']:
            print(f"✓ Completed successfully in {result['elapsed_time']:.1f}s")
        else:
            print(f"✗ Failed with return code {result['return_code']}")
        
        # Estimate remaining time
        elapsed = time.time() - start_time
        avg_time_per_config = elapsed / (i + 1)
        remaining_configs = total_combinations - (i + 1)
        estimated_remaining = avg_time_per_config * remaining_configs
        
        print(f"\nProgress: {i+1}/{total_combinations} configs")
        print(f"Elapsed time: {elapsed/3600:.1f}h")
        print(f"Estimated remaining: {estimated_remaining/3600:.1f}h")
    
    # Final summary
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r['success'])
    
    print(f"\n{'='*60}")
    print(f"Hyperparameter Search Complete!")
    print(f"{'='*60}")
    print(f"Total time: {total_time/3600:.2f}h")
    print(f"Successful runs: {successful}/{total_combinations}")
    print(f"Failed runs: {total_combinations - successful}/{total_combinations}")
    print(f"\nResults saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

