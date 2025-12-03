"""
Training script for Safe-Brax experiments with configs.
Based on mourad_lag.ipynb training approach.
"""

import argparse
import csv
import functools
import inspect
import json
import os
import time
from datetime import datetime
from typing import Dict, Any, List
from typing import Optional

import imageio.v3 as iio
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt

from brax import envs
from brax.envs import Env
from brax.envs.base import State as BraxState, Wrapper

# Import training modules conditionally to handle missing modules gracefully
try:
    from brax.training.agents.ppo.train import train as ppo_train
except ImportError:
    try:
        from brax.training.agents.ppo import train as ppo_train
    except ImportError:
        try:
            from brax.training.agents import ppo

            ppo_train = ppo.train
        except:
            ppo_train = None

try:
    from brax.training.agents.ppo.ppo_cost import train_ppo_cost, RewardMinusCostWrapper
except ImportError:
    try:
        from brax.training.agents.ppo import train_ppo_cost
        from brax.training.agents.ppo import RewardMinusCostWrapper  # type: ignore
    except ImportError:
        train_ppo_cost = None
        RewardMinusCostWrapper = None  # type: ignore

try:
    from brax.training.agents.ppo_saute import train as ppo_saute
except ImportError:
    ppo_saute = None

from brax.training.agents.ppo_lag import train as ppo_lag
from brax.training.agents.ppo_pid import train as ppo_pid
from brax.io import model as brax_model
from brax.io import json as brax_json
import wandb

try:
    from brax.envs.wrappers.saute import SauteWrapper as _SauteWrapper
except Exception:
    _SauteWrapper = None  # type: ignore


# Configure environment for GPU usage
def setup_gpu_environment():
    """Setup GPU environment for MuJoCo and XLA."""
    # Configure MuJoCo to use the EGL rendering backend (requires GPU)
    os.environ['MUJOCO_GL'] = 'egl'

    # Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
    xla_flags = os.environ.get('XLA_FLAGS', '')
    xla_flags += ' --xla_gpu_triton_gemm_any=True'
    os.environ['XLA_FLAGS'] = xla_flags

    # Check installation
    try:
        print('Checking that the installation succeeded:')
        mujoco.MjModel.from_xml_string('<mujoco/>')
        print('Installation successful.')
    except Exception as e:
        raise RuntimeError(
            'Something went wrong during installation. Check the error message above '
            'for more information.'
        ) from e


class CostExtraWrapper(Wrapper):
    """Wrapper that moves cost from info to extras for PPO Lagrange."""

    def step(self, state: BraxState, action: jax.Array) -> BraxState:
        next_state = self.env.step(state, action)

        # PPO Lagrange expects cost in state.info during collection
        if 'cost' not in next_state.info:
            if 'cost' in next_state.metrics:
                next_state.info['cost'] = next_state.metrics['cost']
            else:
                next_state.info['cost'] = jnp.zeros_like(next_state.reward)

        return next_state

    def reset(self, rng: jax.Array) -> BraxState:
        state = self.env.reset(rng)
        # Ensure cost is initialized in info
        if 'cost' not in state.info:
            state.info['cost'] = jnp.zeros_like(state.reward)
        return state


def wrap_env_with_cost(env: envs.Env) -> envs.Env:
    """Wrap environment with cost handling for PPO Lagrange."""
    return CostExtraWrapper(env)


# Global metrics buffer instance
metrics_buffer = []


def custom_progress_fn(num_steps: int, metrics: Dict[str, Any], use_wandb: bool = False, verbose: bool = True) -> None:
    """
    Progress function to print metrics and log to Weights & Biases.

    Args:
        num_steps: Current training step
        metrics: Metrics dictionary
        use_wandb: Whether to use wandb logging
        verbose: Whether to print metrics to console
    """
    global metrics_buffer

    if verbose:
        print(f"Step {num_steps}:")

    log_data = {}
    for key, value in metrics.items():
        if verbose and any(tok in key for tok in ("lambda", "cost", "constraint", "reward")):
            print(f"  {key}: {value}")
        log_data[key] = value

    metrics_buffer.append({"step": num_steps, **log_data})

    if use_wandb and wandb.run is not None and log_data:
        for row in metrics_buffer:
            wandb.log(log_data, step=row["step"])

        # clear the logged history from the buffer
        metrics_buffer.clear()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge override config into base config."""
    config = base_config.copy()
    config.update(override_config)
    return config


def get_algorithm_train_fn(alg_name: str):
    """Get the appropriate training function based on algorithm name."""
    alg_map = {
        'ppo': ppo_train,
        'ppo_cost': train_ppo_cost,
        'ppo_lag': ppo_lag,
        'ppo_pid': ppo_pid,
        'ppo_saute': ppo_saute,
    }

    train_fn = alg_map.get(alg_name)
    if train_fn is None:
        available = [k for k, v in alg_map.items() if v is not None]
        raise ValueError(f"Algorithm '{alg_name}' not available or not installed. Available: {available}")

    return train_fn


def filter_kwargs_for_fn(fn, cfg):
    sig = inspect.signature(fn)
    valid_keys = set(sig.parameters.keys())
    return {k: v for k, v in cfg.items() if k in valid_keys}


def train_from_config(config: argparse.Namespace, seed: int, use_wandb: bool = True,
                      verbose: bool = True) -> tuple[Any, Any, Any, Env]:
    """
    Train an agent using the provided configuration.

    Returns:
        Tuple of (make_inference_fn, params, final_eval_metrics)
    """
    env_name = config.env_name
    alg_name = config.alg

    # Create environments
    train_environment = envs.get_environment(env_name, **config.env_kwargs)
    eval_env = envs.get_environment(env_name, **config.env_kwargs)

    print(f"Training environment '{env_name}' instantiated.")
    print(f"Evaluation environment '{env_name}' instantiated.")

    # Setup wandb if requested
    cfg = vars(config)
    cfg['seed'] = seed
    if use_wandb:
        # Prepare wandb config
        wandb_config = cfg.copy()
        wandb_config['seed'] = seed

        # Initialize wandb
        run_name = f"{env_name}_{alg_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{seed}"
        wandb_project = config.wandb_project
        wandb_group = config.wandb_group if config.wandb_group else env_name
        wandb_tags = config.wandb_tags

        wandb.init(
            project=wandb_project,
            name=run_name,
            id=run_name,
            config=wandb_config,
            group=wandb_group,
            job_type=alg_name,
            tags=wandb_tags,
        )

    # Setup metrics collection
    progress_fn = functools.partial(custom_progress_fn, use_wandb=use_wandb, verbose=verbose,)

    # Get the appropriate training function
    train_fn_base = get_algorithm_train_fn(alg_name)
    train_kwargs = filter_kwargs_for_fn(train_fn_base, cfg)

    # Create the training function
    train_fn = functools.partial(train_fn_base, **train_kwargs)

    # Train the agent
    print(f"Starting {alg_name} training for {env_name}...")
    make_inference_fn, params, final_eval_metrics = train_fn(
        environment=train_environment,
        eval_env=eval_env,
        progress_fn=progress_fn
    )
    print("Training finished.")

    # Log final metrics to wandb
    if use_wandb and wandb.run is not None and final_eval_metrics:
        final_log_data = {}
        for key, value in final_eval_metrics.items():
            if value is not None:
                final_log_data[key] = value
        if final_log_data:
            wandb.log(final_log_data, step=int(float(np.asarray(config.num_timesteps).reshape(()))))

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = config.model_dir
    os.makedirs(model_dir, exist_ok=True)
    model_path = f'{model_dir}/{env_name.lower()}_{alg_name}_seed{seed}_{timestamp}'
    brax_model.save_params(model_path, params)
    print(f"Trained model parameters saved to: {model_path}")

    return make_inference_fn, params, final_eval_metrics, eval_env


def collect_rollout_metrics(env_name: str, make_inference_fn, params,
                            num_steps: int = 5000, seed: int = None,
                            save_trajectory: bool = True,
                            save_plots: bool = True,
                            env_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, List]:
    """
    Collect detailed metrics during a rollout.

    Returns:
        Dictionary containing all collected metrics
    """
    # Create evaluation environment (respect env_kwargs)
    eval_environment = envs.get_environment(env_name, **(env_kwargs or {}))

    # JIT compile reset and step
    jit_eval_reset = jax.jit(eval_environment.reset)
    jit_eval_step = jax.jit(eval_environment.step)

    # Create inference function
    inference_fn = make_inference_fn(params)
    jit_inference_fn = jax.jit(inference_fn)

    print(f"Inference function for rollout created for {env_name}.")

    # Initialize data collection
    rollout_frames = []
    rollout_metrics_data = {
        'distance_to_goal': [],
        'last_dist_goal': [],
        'reward': [],
        'dist_reward': [],
        'goal_reward': [],
        'orientation_reward': [],
        'ctrl_cost': [],
        'x_position': [],
        'y_position': [],
        'agent_pos_x': [],
        'agent_pos_y': [],
        'goal_pos_x': [],
        'goal_pos_y': [],
        'x_velocity': [],
        'y_velocity': [],
        'goals_reached_count': [],
        'cost': []
    }
    actions = []

    # Initialize rollout
    if seed is None:
        seed = int(time.time())
    rng_rollout = jax.random.PRNGKey(seed)
    eval_state = jit_eval_reset(rng_rollout)

    print(f"Starting rollout for {num_steps} steps...")
    for i in range(num_steps):
        act_rng, rng_rollout = jax.random.split(rng_rollout)
        action, _ = jit_inference_fn(eval_state.obs, act_rng)
        actions.append(action)

        eval_state = jit_eval_step(eval_state, action)
        rollout_frames.append(eval_state.pipeline_state)

        # Collect metrics from eval_state.metrics
        rollout_metrics_data['distance_to_goal'].append(eval_state.metrics.get('distance_to_goal', np.nan))
        rollout_metrics_data['reward'].append(eval_state.metrics.get('reward', np.nan))
        rollout_metrics_data['cost'].append(eval_state.metrics.get('cost', np.nan))
        rollout_metrics_data['dist_reward'].append(eval_state.metrics.get('dist_reward', np.nan))
        rollout_metrics_data['goal_reward'].append(eval_state.metrics.get('goal_reward', np.nan))
        rollout_metrics_data['orientation_reward'].append(eval_state.metrics.get('orientation_reward', np.nan))
        rollout_metrics_data['ctrl_cost'].append(eval_state.metrics.get('ctrl_cost', np.nan))
        rollout_metrics_data['x_position'].append(eval_state.metrics.get('x_position', np.nan))
        rollout_metrics_data['y_position'].append(eval_state.metrics.get('y_position', np.nan))
        rollout_metrics_data['x_velocity'].append(eval_state.metrics.get('x_velocity', np.nan))
        rollout_metrics_data['y_velocity'].append(eval_state.metrics.get('y_velocity', np.nan))
        rollout_metrics_data['goals_reached_count'].append(eval_state.metrics.get('goals_reached_count', np.nan))

        # Collect metrics from eval_state.info
        rollout_metrics_data['last_dist_goal'].append(eval_state.info.get('last_dist_goal', np.nan))
        current_agent_pos = eval_state.info.get('agent_pos', np.array([np.nan, np.nan, np.nan]))
        current_goal_pos = eval_state.info.get('goal_pos', np.array([np.nan, np.nan, np.nan]))
        rollout_metrics_data['agent_pos_x'].append(current_agent_pos[0])
        rollout_metrics_data['agent_pos_y'].append(current_agent_pos[1])
        rollout_metrics_data['goal_pos_x'].append(current_goal_pos[0])
        rollout_metrics_data['goal_pos_y'].append(current_goal_pos[1])

        if i % 100 == 0 or i == num_steps - 1:
            print(
                f"Rollout step {i + 1}/{num_steps} completed. Goals reached: {eval_state.metrics.get('goals_reached_count', 0)}")

        if eval_state.done:
            print(f"Rollout terminated early at step {i + 1} due to done signal.")
            remaining_steps = num_steps - (i + 1)
            for key_metric in rollout_metrics_data.keys():
                rollout_metrics_data[key_metric].extend([np.nan] * remaining_steps)
            break

    print("Rollout finished.")

    # Save trajectory if requested
    if save_trajectory:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs('trajectories', exist_ok=True)
        rollout_trajectory_path = f'trajectories/{env_name}_rollout_{timestamp}.json'
        brax_json.save(rollout_trajectory_path, eval_environment.sys, rollout_frames)
        print(f"Rollout trajectory saved to {rollout_trajectory_path}")

    # Create plots if requested
    if save_plots:
        create_rollout_plots(rollout_metrics_data, env_name)

    return rollout_metrics_data


def verify_ppoc_shaping(env_name: str, make_inference_fn, params,
                        num_steps: int, seed: int,
                        cost_weight: float = 1.0,
                        env_kwargs: Optional[Dict[str, Any]] = None,
                        out_dir: str = 'runs/smoke') -> str:
    """Runs a short rollout on a RewardMinusCost-wrapped env and logs per-step
    raw_reward, shaped_reward, and cost for verifying shaped ≈ raw - cost.

    Returns the CSV path written.
    """
    # Create evaluation environment and wrap it if available
    eval_environment = envs.get_environment(env_name, **(env_kwargs or {}))
    if RewardMinusCostWrapper is not None:
        eval_environment = RewardMinusCostWrapper(eval_environment, cost_weight=cost_weight)

    # JIT compile reset and step
    jit_eval_reset = jax.jit(eval_environment.reset)
    jit_eval_step = jax.jit(eval_environment.step)

    # Create inference function
    inference_fn = make_inference_fn(params)
    jit_inference_fn = jax.jit(inference_fn)

    # Initialize rollout
    if seed is None:
        seed = int(time.time())
    rng_rollout = jax.random.PRNGKey(seed)
    eval_state = jit_eval_reset(rng_rollout)

    # Prepare CSV logging
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"{out_dir}/ppoc_verify_{env_name}_seed{seed}_{timestamp}.csv"
    fieldnames = [
        'step', 'raw_reward', 'shaped_reward', 'cost',
        'reward_delta_vs_raw_minus_cost'
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(num_steps):
            act_rng, rng_rollout = jax.random.split(rng_rollout)
            action, _ = jit_inference_fn(eval_state.obs, act_rng)
            eval_state = jit_eval_step(eval_state, action)

            # Extract values
            info = eval_state.info
            raw_reward = info.get('raw_reward', eval_state.reward)
            shaped_reward = info.get('shaped_reward', eval_state.reward)
            cost = info.get('cost', eval_state.metrics.get('cost', 0.0))

            # Convert to Python floats
            def to_float(x):
                try:
                    return float(x)
                except Exception:
                    return float(getattr(x, 'item', lambda: 0.0)())

            rr = to_float(raw_reward)
            sr = to_float(shaped_reward)
            cc = to_float(cost)
            delta = sr - (rr - cc * cost_weight)

            writer.writerow({
                'step': i,
                'raw_reward': rr,
                'shaped_reward': sr,
                'cost': cc,
                'reward_delta_vs_raw_minus_cost': delta,
            })

            if eval_state.done:
                # continue but metrics may be meaningless; break could be used
                pass

    print(f"PPO-C verify log written to: {csv_path}")
    return csv_path


def create_rollout_plots(rollout_metrics_data: Dict[str, List], env_name: str) -> None:
    """Create and save plots from rollout metrics."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)
    plot_path_base = f'{plot_dir}/{env_name}_rollout_{timestamp}'

    num_steps = len(rollout_metrics_data['distance_to_goal'])
    time_steps = np.arange(num_steps)

    plt.style.use('seaborn-v0_8-darkgrid')

    # Plot 1: Distance and Last Distance to Goal
    plt.figure(figsize=(12, 7))
    plt.plot(time_steps, rollout_metrics_data['distance_to_goal'], label='Current Distance to Goal', linestyle='-')
    plt.plot(time_steps, rollout_metrics_data['last_dist_goal'], label='Last Distance to Goal', linestyle='--')
    plt.xlabel("Time Step")
    plt.ylabel("Distance")
    plt.title(f"{env_name} - Rollout: Goal Tracking")
    plt.legend()
    plt.tight_layout()
    goal_tracking_plot_path = f'{plot_path_base}_goal_distances.png'
    plt.savefig(goal_tracking_plot_path)
    plt.close()
    print(f"Goal tracking plot saved to: {goal_tracking_plot_path}")

    # Plot 2: Cost Plot
    plt.figure(figsize=(12, 7))
    plt.plot(time_steps, rollout_metrics_data['cost'], label='Cost', linestyle='-')
    plt.xlabel("Time Step")
    plt.ylabel("Cost")
    plt.title(f"{env_name} - Rollout: Cost")
    plt.legend()
    plt.tight_layout()
    cost_plot_path = f'{plot_path_base}_cost.png'
    plt.savefig(cost_plot_path)
    plt.close()
    print(f"Cost plot saved to: {cost_plot_path}")

    # Plot 3: Cumulative Cost
    cumulative_cost = np.cumsum(rollout_metrics_data['cost'])
    plt.figure(figsize=(12, 7))
    plt.plot(time_steps, cumulative_cost, label='Cumulative Cost', color='red')
    plt.xlabel("Time Step")
    plt.ylabel("Cumulative Cost")
    plt.title(f"{env_name} - Rollout: Cumulative Cost Over Time")
    plt.legend()
    plt.tight_layout()
    cumulative_cost_plot_path = f'{plot_path_base}_cumulative_cost.png'
    plt.savefig(cumulative_cost_plot_path)
    plt.close()
    print(f"Cumulative cost plot saved to: {cumulative_cost_plot_path}")

    # Plot 4: Reward Component Breakdown
    plt.figure(figsize=(12, 7))
    plt.plot(time_steps, rollout_metrics_data['dist_reward'], label='Distance Reward', alpha=0.7)
    plt.plot(time_steps, rollout_metrics_data['goal_reward'], label='Goal Reward', alpha=0.7)
    plt.plot(time_steps, rollout_metrics_data['orientation_reward'], label='Orientation Reward', alpha=0.7)
    plt.plot(time_steps, -np.array(rollout_metrics_data['ctrl_cost']), label='Negative Control Cost', alpha=0.7)
    plt.plot(time_steps, rollout_metrics_data['reward'], label='Total Reward', linestyle='--', color='black',
             linewidth=2)
    plt.xlabel("Time Step")
    plt.ylabel("Reward Value")
    plt.title(f"{env_name} - Rollout: Reward Component Breakdown")
    plt.legend()
    plt.tight_layout()
    reward_breakdown_plot_path = f'{plot_path_base}_reward_breakdown.png'
    plt.savefig(reward_breakdown_plot_path)
    plt.close()
    print(f"Reward breakdown plot saved to: {reward_breakdown_plot_path}")

    # Plot 5: X-Y Trajectory
    plt.figure(figsize=(10, 8))
    valid_x = np.array(rollout_metrics_data['x_position'])
    valid_y = np.array(rollout_metrics_data['y_position'])
    goal_x_series = np.array(rollout_metrics_data['goal_pos_x'])
    goal_y_series = np.array(rollout_metrics_data['goal_pos_y'])

    # Filter out NaNs
    valid_indices_agent = ~(np.isnan(valid_x) | np.isnan(valid_y))
    valid_x_agent = valid_x[valid_indices_agent]
    valid_y_agent = valid_y[valid_indices_agent]

    valid_indices_goal = ~(np.isnan(goal_x_series) | np.isnan(goal_y_series))
    valid_x_goal = goal_x_series[valid_indices_goal]
    valid_y_goal = goal_y_series[valid_indices_goal]

    if len(valid_x_agent) > 0 and len(valid_y_agent) > 0:
        plt.plot(valid_x_agent, valid_y_agent, 'k-', alpha=0.7, label='Agent Path')
        plt.scatter(valid_x_agent[0], valid_y_agent[0], c='green', s=100, label='Agent Start', zorder=5, marker='o')
        plt.scatter(valid_x_agent[-1], valid_y_agent[-1], c='red', s=100, label='Agent End', zorder=5, marker='x')

        if len(valid_x_goal) > 0 and len(valid_y_goal) > 0:
            plt.scatter(valid_x_goal[0], valid_y_goal[0], c='blue', s=150, label='Initial Goal', zorder=4, marker='*')
            if any(g_x != valid_x_goal[0] for g_x in valid_x_goal) or any(
                    g_y != valid_y_goal[0] for g_y in valid_y_goal):
                plt.plot(valid_x_goal, valid_y_goal, 'b--', alpha=0.5, label='Goal Path')
                plt.scatter(valid_x_goal[-1], valid_y_goal[-1], c='purple', s=150, label='Final Goal', zorder=4,
                            marker='*')

        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title(f"{env_name} - Rollout: X-Y Trajectory")
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, "No valid position data for trajectory plot", ha='center', va='center')

    plt.tight_layout()
    trajectory_plot_path = f'{plot_path_base}_xy_trajectory.png'
    plt.savefig(trajectory_plot_path)
    plt.close()
    print(f"X-Y trajectory plot saved to: {trajectory_plot_path}")


def record_episode_video(
        env,
        make_inference_fn,
        params,
        steps: int = 2500,
        cameras: List[str] | List[int] = (0,),  # camera names or ids
        width: int = 320,
        height: int = 240,
        fps: int = 100,
        frame_stride=1,
        out_name: str = "rollout",
        log_to_wandb: bool = True,
        seed: int = 0,
        show_metrics: bool = True,  # Print the cost on the screen
        font: str = "DejaVuSans-Bold"  # Font for overlay text, if available
):
    """
    Renders a fresh eval rollout controlled by your trained policy.
    We step the *env* for observations/actions, and in parallel step a MuJoCo
    simulator for pretty pixels.
    """
    # 1) Ensure headless GPU rendering (you might need to do this before importing mujoco)
    os.environ.setdefault("MUJOCO_GL", "egl")

    start_time = os.times()

    # 2) JIT policy
    infer = jax.jit(make_inference_fn(params))
    reset_fn = env.reset
    step_fn = env.step

    @jax.jit
    def rollout(key):
        state = reset_fn(key)

        def step_body(carry, _):
            state, key = carry
            key, sk = jax.random.split(key)
            action, _ = infer(state.obs, sk)
            next_state = step_fn(state, action)

            frame = next_state.pipeline_state  # for render
            reward = next_state.reward  # scalar
            cost = next_state.info["cost"]  # scalar

            return (next_state, key), (frame, reward, cost)

        (final_state, _), (frames, rewards, costs) = jax.lax.scan(
            step_body, (state, key), xs=None, length=steps
        )
        return frames, rewards, costs, final_state

    # 3) Run rollout to collect frames
    key = jax.random.PRNGKey(seed)
    frames_batched, rewards_batched, costs_batched, final_state = rollout(key)  # PyTree with leading T

    print("Rollout took %.2f seconds." % (os.times()[4] - start_time[4]))
    start_time = os.times()

    frames_batched = jax.device_get(frames_batched)

    T = int(frames_batched.qpos.shape[0])
    frames = [jax.tree.map(lambda x: x[i], frames_batched) for i in range(T)]

    # Build indices of frames we keep
    keep_idx = np.arange(0, T, frame_stride, dtype=int)

    # Downsample frames
    frames = [frames[i] for i in keep_idx]

    # Keep full-resolution rewards/costs for exact cumulative values
    rewards_full = np.asarray(jax.device_get(rewards_batched))
    costs_full = np.asarray(jax.device_get(costs_batched))

    # Exact cumulative totals at the exact original step indices that we render
    cum_rewards_full = np.cumsum(rewards_full)
    cum_costs_full = np.cumsum(costs_full)

    cum_rewards_at_frames = cum_rewards_full[keep_idx]
    cum_costs_at_frames = cum_costs_full[keep_idx]

    # If you still want per-frame (downsampled) instantaneous values for something else:
    rewards = rewards_full[keep_idx]
    costs = costs_full[keep_idx]

    # 4) Render the episode
    for camera in cameras:
        rendering = env.render(frames, width=width, height=height, camera=camera)
        print("Rendering took %.2f seconds." % (os.times()[4] - start_time[4]))

        # 5) Add reward/cost overlay
        if show_metrics:
            start_time = os.times()
            rendering_with_metrics = []
            try:
                # Try to load a font, fallback to default if not available
                font = ImageFont.truetype(f"{font}.ttf", 20)
            except (OSError, IOError):
                font = ImageFont.load_default()

            for i, (frame, reward, cost) in enumerate(zip(rendering, rewards, costs)):
                # Convert frame to PIL Image
                img = Image.fromarray(frame.astype(np.uint8))
                draw = ImageDraw.Draw(img)

                # Extract metrics from the state info
                total_reward = float(cum_rewards_at_frames[i])
                total_cost = float(cum_costs_at_frames[i])

                # Add cost text overlay
                reward_text = f"Reward: {total_reward:.2f}"
                cost_text = f"Cost: {total_cost:.2f}"

                # Color
                text_color_reward = (50, 220, 50)  # Green text
                text_color_cost = (230, 60, 60)  # Red text
                outline_color = (0, 0, 0)  # Black outline

                # Position text in top-left corner
                x_rew, y_rew = 10, 10
                x_cost, y_cost = 10, 40

                # Draw text with outline for better visibility
                draw.text((x_rew, y_rew), reward_text, font=font, fill=text_color_reward, stroke_width=2,
                          stroke_fill=outline_color)
                draw.text((x_cost, y_cost), cost_text, font=font, fill=text_color_cost, stroke_width=2,
                          stroke_fill=outline_color)

                # Convert back to numpy array
                rendering_with_metrics.append(np.array(img))

            rendering = rendering_with_metrics
            print("Overlay text took %.2f seconds." % (os.times()[4] - start_time[4]))

        # 6) Save mp4 (and log)
        file_name = f"{out_name}_{camera}.mp4"
        os.makedirs("videos", exist_ok=True)
        mp4_path = os.path.join("videos", file_name)
        iio.imwrite(mp4_path, np.stack(rendering), fps=fps)

        if log_to_wandb and wandb.run is not None:
            wandb.log({f"video/{camera}": wandb.Video(mp4_path, fps=fps, format="mp4")})

        print("Saved video:", mp4_path)


def _json_type(s):
    if s is None or isinstance(s, dict):
        return s
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON string for env_kwargs: {e}")


def main():
    """Main function to run training from command line."""
    parser = argparse.ArgumentParser(description='Train Safe-Brax agents from config files')

    # --- Core ---
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="Random seeds")

    # --- Experiment Control ---
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity")
    parser.add_argument("--skip-rollout", action="store_true", help="Skip rollout evaluation after training")
    parser.add_argument("--skip-video", action="store_true", help="Skip video recording after training")
    parser.add_argument("--out_dir", type=str, default="runs/experimental_results", help="Directory for metrics/outputs")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save model parameters")

    # --- Environment ---
    parser.add_argument("--env_name", type=str, default="safe_point_goal", help="Env name")
    parser.add_argument(
        "--env_kwargs",
        type=_json_type,
        default={
            "physics": {
                "timestep": 0.02,
                "n_frames": 4
            },
            "cost": {
                "scaler": 1.0,
                "ctrl_cost_weight": 0.001
            },
            "reward": {
                "reward_goal": 1.0,
                "dense_scale": 0.0
            }
        },
        help="JSON for env kwargs"
    )

    # --- Algorithm ---
    parser.add_argument("--alg", type=str, default="ppo_lag", help="Algorithm name (e.g., ppo, ppo_lag)")
    parser.add_argument("--max_devices_per_host", type=int, default=None, help="Limit devices per host")

    # --- Training Scale / Rollout ---
    parser.add_argument("--num_timesteps", type=float, default=100_000_000, help="Total training timesteps")
    parser.add_argument("--episode_length", type=int, default=2000, help="Episode length")
    parser.add_argument("--num_envs", type=int, default=2048, help="Number of parallel envs")
    parser.add_argument("--unroll_length", type=int, default=8, help="Unroll length")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--num_minibatches", type=int, default=32, help="Number of minibatches")
    parser.add_argument("--num_updates_per_batch", type=int, default=6, help="SGD updates per batch")
    parser.add_argument("--rollout-steps", dest="rollout_steps", type=int, default=5000,
                        help="Steps for post-training rollout evaluation")

    # --- Optimization / PPO Core ---
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--entropy_cost", type=float, default=5e-3, help="Entropy coefficient")
    parser.add_argument("--discounting", type=float, default=0.99, help="Discount factor (gamma)")
    parser.add_argument("--reward_scaling", type=float, default=0.1, help="Reward scaling")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clipping_epsilon", type=float, default=0.3, help="PPO clipping epsilon")
    parser.add_argument("--normalize_observations", type=bool, default=True,
                        help="Normalize observations (true/false)")

    # --- Evaluation / Logging cadence ---
    parser.add_argument("--num_evals", type=int, default=5, help="Number of eval passes during training")
    parser.add_argument("--num_eval_envs", type=int, default=128, help="Parallel envs during eval")
    parser.add_argument("--deterministic_eval", type=bool, default=False, help="Deterministic eval policy")
    parser.add_argument("--training_metrics_steps", type=float, default=1e6,
                        help="Env steps between training metrics logs")

    # --- PPO-Lagrange ---
    parser.add_argument("--safety_bound", type=float, default=25.0, help="Episodic safety constraint bound")
    parser.add_argument("--lagrangian_coef_rate", type=float, default=10.0, help="Lagrange multiplier LR")
    parser.add_argument("--initial_lambda_lagr", type=float, default=0.0, help="Initial lambda value")

    # --- PPO-PID Lagrange ---
    parser.add_argument("--pid_kp", type=float, default=0.1, help="PID: proportional gain")
    parser.add_argument("--pid_ki", type=float, default=0.01, help="PID: integral gain")
    parser.add_argument("--pid_kd", type=float, default=0.01, help="PID: derivative gain")
    parser.add_argument("--pid_integral_clip", type=float, default=10.0, help="PID: anti-windup cap for integral term")
    parser.add_argument("--pid_lambda_clip", type=float, default=1e6, help="PID: clamp for lambda")
    parser.add_argument("--pid_deriv_ema_beta", type=float, default=0.95, help="PID: derivative EMA smoothing")

    # --- PPO-Saute ---
    parser.add_argument("--saute-initial-budget", dest="initial_budget", type=float, default=15.0, help="Initial discounted safety budget b0")
    parser.add_argument("--saute-gamma-budget", dest="gamma_budget", type=float, default=None, help="Budget discount factor; defaults to --discounting if None")
    parser.add_argument("--saute-termination-on-violation", dest="termination_on_violation", type=int, default=1, help="Terminate episode on budget violation (1/0)")
    parser.add_argument("--saute-violation-penalty", dest="violation_penalty", type=float, default=0.0, help="Optional terminal penalty added only on violation step")
    parser.add_argument("--saute-normalize-budget-obs", dest="normalize_budget_obs", type=int, default=1, help="Normalize budget observation by initial budget")

    # --- PPO-Cost verification ---
    parser.add_argument("--ppoc-verify-log-steps", type=int, default=0,
                        help="If >0 and alg is ppo_cost, run verify rollout and log shaping")
    parser.add_argument("--ppoc-cost-weight", type=float, default=1.0,
                        help="Cost weight used for PPO-C verify logging")

    # --- WandB ---
    parser.add_argument("--use_wandb", type=bool, default=True, help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="safe-brax-experimental-results", help="W&B project")
    parser.add_argument("--wandb_group", type=str, default=None, help="W&B group")
    parser.add_argument("--wandb_tags", type=str, nargs='+', help="JSON list or path of tags")

    # --- Video Recording ---
    parser.add_argument("--cameras", type=str, nargs="+", default=["fixedfar", "vision"], help="Camera names or ids")
    parser.add_argument("--video_width", type=int, default=320, help="Output video width")
    parser.add_argument("--video_height", type=int, default=240, help="Output video height")
    parser.add_argument("--video_length", type=int, default=None, help="Number of frames in the video")
    parser.add_argument("--video_fps", type=int, default=100, help="Output video FPS")
    parser.add_argument("--video_frame_stride", type=int, default=1, help="Output video frame stride")

    config = parser.parse_args()

    # Setup GPU environment
    setup_gpu_environment()

    # Run training for each seed
    for seed in config.seeds:
        print(f"\n{'=' * 50}")
        print(f"Running experiment with seed {seed}")
        print(f"{'=' * 50}\n")

        # Train the agent
        make_inference_fn, params, final_metrics, eval_env = train_from_config(
            config=config,
            seed=seed,
            use_wandb=config.use_wandb,
            verbose=not config.quiet
        )

        # Perform rollout evaluation if not skipped
        if not config.skip_rollout:
            print(f"\nPerforming rollout evaluation...")
            rollout_env_name = config.env_name
            rollout_metrics = collect_rollout_metrics(
                env_name=rollout_env_name,
                make_inference_fn=make_inference_fn,
                params=params,
                num_steps=config.rollout_steps,
                seed=seed,
                save_trajectory=True,
                save_plots=True,
                env_kwargs=config.env_kwargs
            )

        if not config.skip_video:
            video_length = config.video_length if config.video_length else config.episode_length
            record_episode_video(
                env=eval_env,
                make_inference_fn=make_inference_fn,
                params=params,
                steps=video_length,
                cameras=config.cameras,
                width=config.video_width,
                height=config.video_height,
                fps=config.video_fps,
                frame_stride=config.video_frame_stride,
                out_name=f"{config.env_name}_{config.alg}_seed{seed}",
                log_to_wandb=config.use_wandb,
                seed=seed,
            )

        # PPO-C verify shaping log if requested
        if config.alg in ('ppo_cost', 'ppoc') and config.ppoc_verify_log_steps > 0:
            print("\nRunning PPO-C shaping verification rollout...")
            rollout_env_name = config.env_name
            _ = verify_ppoc_shaping(
                env_name=rollout_env_name,
                make_inference_fn=make_inference_fn,
                params=params,
                num_steps=config.ppoc_verify_log_steps,
                seed=seed,
                cost_weight=config.cost_weight,
                out_dir=config.out_dir
            )

        # Finish wandb run if active
        if config.use_wandb and wandb.run is not None:
            wandb.finish()

    print("\nAll experiments completed!")


if __name__ == "__main__":
    main()
