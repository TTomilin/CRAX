import inspect
import os
import time
from datetime import datetime
from typing import Optional, Dict, Any, List

import jax
import mujoco
import numpy as np
from PIL import ImageFont, Image, ImageDraw
from imageio import v3 as iio
from jax import numpy as jnp
from matplotlib import pyplot as plt

import wandb
from crax import envs
from crax.io import json as brax_json
from crax.training.agents.crpo import train as crpo
from crax.training.agents.focops import train as focops
from crax.training.agents.p3o import train as p3o
from crax.training.agents.ppo import train_ppo_cost
from crax.training.agents.ppo.train import train as ppo_train
from crax.training.agents.ppo_lag import train as ppo_lag
from crax.training.agents.ppo_pid import train as ppo_pid
from crax.training.agents.ppo_saute import train as ppo_saute
from crax.training.agents.sac.train import train as sac_train
from crax.training.agents.sac_lag import train as sac_lag
from crax.training.agents.sac_pid import train as sac_pid

# Global metrics buffer instance
metrics_buffer = []

# Pixel-observation training camera
VISION_CAMERA_OVERRIDES = {
    'humanoid': 'track',
    'ant': 'track',
    'cheetah': 'track',
    'walker2d': 'track',
    'spider': 'track',
    'reacher': 'fixedfar',
}


def morphology_override(env_name, overrides):
    """First substring match of `overrides`' keys found in `env_name`, else None."""
    for key, value in overrides.items():
        if key in env_name:
            return value
    return None


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

    def _mean_value(val):
        # Convert JAX/NumPy arrays or lists/tuples to a scalar by averaging
        arr = np.asarray(val)
        # If it's already scalar, return item; otherwise mean
        if arr.ndim == 0 or arr.size == 1:
            return arr.reshape(-1)[0].item()
        return arr.reshape(-1).mean().item()

    if verbose:
        print(f"Step {num_steps}:")

    log_data = {}
    for key, value in metrics.items():
        value = _mean_value(value)
        # Print only key categories to keep console light
        if verbose and any(tok in key for tok in ("lambda", "cost", "constraint", "reward")):
            print(f"  {key}: {value}")
        log_data[key] = value

    # If nothing to log, exit early
    if not log_data:
        return

    metrics_buffer.append({"step": num_steps, **log_data})

    if use_wandb and wandb.run is not None:
        for row in metrics_buffer:
            wandb.log({k: row[k] for k in log_data.keys()}, step=row["step"])  # log summarized scalars only
        # clear the logged history from the buffer
        metrics_buffer.clear()


def setup_gpu_environment(vision: bool = False):
    """Setup GPU environment for MuJoCo and XLA.

    Args:
      vision: if True, caps JAX's GPU memory preallocation so MJWarp (the
        pixel-obs renderer) has room to instantiate its own CUDA graphs/
        buffers alongside JAX in the same process. JAX preallocates ~75-90%
        of GPU memory by default. On small GPUs this starves MJWarp. Only
        applied if the user hasn't already set XLA_PYTHON_CLIENT_MEM_FRACTION.
    """
    # Configure MuJoCo to use the EGL rendering backend (requires GPU)
    os.environ['MUJOCO_GL'] = 'egl'

    # Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
    xla_flags = os.environ.get('XLA_FLAGS', '')
    xla_flags += ' --xla_gpu_triton_gemm_any=True'
    os.environ['XLA_FLAGS'] = xla_flags

    if vision and 'XLA_PYTHON_CLIENT_MEM_FRACTION' not in os.environ:
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'
        print(
            "Vision mode: capped XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 to leave "
            "GPU headroom for MJWarp's renderer (override by setting the env "
            "var yourself before running)."
        )

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


def get_algorithm_train_fn(alg_name: str):
    """Get the appropriate training function based on algorithm name."""
    alg_map = {
        'ppo': ppo_train,
        'ppo_cost': train_ppo_cost,
        'ppo_lag': ppo_lag,
        'ppo_pid': ppo_pid,
        'ppo_saute': ppo_saute,
        'p3o': p3o,
        'focops': focops,
        'crpo': crpo,
        'sac': sac_train,
        'sac_lag': sac_lag,
        'sac_pid': sac_pid,
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


def make_vision_network_factory(alg_name: str, **vision_net_kwargs):
    """Create a vision-aware network factory for the given algorithm.

    On-policy safe RL algorithms (ppo_lag, ppo_pid, focops, p3o, crpo; plus
    ppo_saute and plain ppo, which pass through to the same PPO trainer) need
    a cost_value_network in addition to policy and value networks. Off-policy
    safe RL algorithms (sac_lag, sac_pid) need CNN reward- and cost-Q-critics
    instead, and plain sac needs a CNN reward-Q-critic only (no cost side).
    This factory routes to the correct network builder based on the
    algorithm.

    Args:
        alg_name: Algorithm name (e.g., 'ppo', 'ppo_lag', 'sac_lag').
        **vision_net_kwargs: Extra kwargs passed to the underlying
            make_*_networks_vision factory (e.g., normalise_channels,
            policy_obs_key, value_obs_key).

    Returns:
        A network_factory callable compatible with the algorithm's training
        loop.
    """
    def _bind(builder, **default_kwargs):
        """Wrap a make_*_networks_vision builder into a network_factory closure.

        Precedence for a given kwarg: explicit call-site kwargs > factory-level
        vision_net_kwargs > alg-specific default_kwargs.
        """

        def network_factory(observation_size, action_size, **kwargs):
            merged = {**default_kwargs, **vision_net_kwargs, **kwargs}
            return builder(observation_size=observation_size, action_size=action_size, **merged)

        return network_factory

    if alg_name == 'sac':
        from crax.training.agents.sac.networks import make_sac_networks_vision
        return _bind(make_sac_networks_vision)

    if alg_name in {'sac_lag', 'sac_pid'}:
        from crax.training.agents.sac_lag.networks import make_sac_lag_networks_vision
        return _bind(make_sac_lag_networks_vision)

    from crax.training.agents.ppo.networks_vision import make_ppo_networks_vision

    safe_algs = {'ppo_lag', 'ppo_pid', 'focops', 'p3o', 'crpo'}
    defaults = {'cost_value_hidden_layer_sizes': (256,) * 5} if alg_name in safe_algs else {}
    return _bind(make_ppo_networks_vision, **defaults)


def collect_rollout_metrics(env_name: str, make_inference_fn, params,
                            num_steps: int = 5000, seed: int = None,
                            save_trajectory: bool = True,
                            save_plots: bool = True,
                            level: Optional[int] = None,
                            env_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, List]:
    """
    Collect detailed metrics during a rollout.

    Returns:
        Dictionary containing all collected metrics
    """
    # Create evaluation environment (with optional difficulty level)
    eval_environment = envs.get_environment(env_name, level=level, **(env_kwargs or {}))

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
            def _scalar_bool(x):
                x = jax.device_get(x)
                x = np.asarray(x)
                if x.size == 0:
                    return False
                return bool(x.reshape(-1)[0])

            done_goal = _scalar_bool(eval_state.info.get('done_goal', False))
            done_nan = _scalar_bool(eval_state.info.get('done_nan', False))
            done_unhealthy = _scalar_bool(eval_state.info.get('done_unhealthy', False))
            print(
                f"Rollout terminated early at step {i + 1} due to done signal. "
                f"done_goal={done_goal}, done_nan={done_nan}, done_unhealthy={done_unhealthy}"
            )
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
        font: str = "DejaVuSans-Bold",  # Font for overlay text, if available
        num_episodes: int = 1,
):
    """
    Render one or more fresh eval episodes controlled by your trained policy.
    Steps the env for observations/actions, and in parallel steps a MuJoCo
    simulator for pretty pixels. All recorded episodes are concatenated in
    sequence into a single video per camera.
    """
    # 1) Ensure headless GPU rendering (you might need to do this before importing mujoco)
    os.environ.setdefault("MUJOCO_GL", "egl")

    start_time = os.times()

    # 2) JIT policy
    infer = jax.jit(make_inference_fn(params))
    reset_fn = env.reset
    step_fn = env.step

    @jax.jit
    def rollout_one(key):
        state = reset_fn(key)

        def step_body(carry, _):
            state, key = carry
            key, sk = jax.random.split(key)
            action, _ = infer(state.obs, sk)
            next_state = step_fn(state, action)

            frame = next_state.pipeline_state  # for render
            reward = next_state.reward  # scalar
            # Be robust to envs without a 'cost' signal
            cost = next_state.info.get("cost", jnp.zeros_like(next_state.reward))  # scalar
            # Also mark termination if env signals done OR NaNs appear in obs/reward
            obs_for_nan = next_state.obs
            if isinstance(obs_for_nan, dict):
                # Vision mode: only check state vector, not pixel arrays
                obs_for_nan = obs_for_nan.get('state', jnp.zeros(()))
            nan_done = jnp.isnan(reward) | jnp.any(jnp.isnan(obs_for_nan))
            done_base = jnp.asarray(next_state.done, dtype=bool)
            done_flag = jnp.logical_or(done_base, nan_done)
            done_goal = next_state.info.get("done_goal", jnp.zeros_like(done_base, dtype=bool))
            done_nan = next_state.info.get("done_nan", jnp.zeros_like(done_base, dtype=bool))
            done_unhealthy = next_state.info.get(
                "done_unhealthy", jnp.zeros_like(done_base, dtype=bool)
            )

            return (next_state, key), (frame, reward, cost, done_flag, done_goal, done_nan, done_unhealthy)

        (final_state, _), (frames, rewards, costs, dones, done_goal, done_nan, done_unhealthy) = jax.lax.scan(
            step_body, (state, key), xs=None, length=steps
        )
        return frames, rewards, costs, dones, done_goal, done_nan, done_unhealthy

    # 3) Run N episodes to collect frames
    key = jax.random.PRNGKey(seed)

    all_frames = []  # list of pipeline_state lists (concatenated later)
    all_rewards = []  # list of np arrays after downsampling per episode
    all_costs = []  # list of np arrays after downsampling per episode

    for ep in range(max(1, int(num_episodes))):
        key, ep_key = jax.random.split(key)
        frames_batched, rewards_batched, costs_batched, dones_batched, done_goal_batched, done_nan_batched, done_unhealthy_batched = rollout_one(
            ep_key)

        frames_batched = jax.device_get(frames_batched)
        rewards_batched_np = np.asarray(jax.device_get(rewards_batched))
        costs_batched_np = np.asarray(jax.device_get(costs_batched))
        dones_batched_np = np.asarray(jax.device_get(dones_batched)).astype(bool)
        done_goal_np = np.asarray(jax.device_get(done_goal_batched)).astype(bool)
        done_nan_np = np.asarray(jax.device_get(done_nan_batched)).astype(bool)
        done_unhealthy_np = np.asarray(jax.device_get(done_unhealthy_batched)).astype(bool)

        # Trim to first termination if any
        if dones_batched_np.ndim == 0:
            done_index = int(dones_batched_np)
        else:
            done_hits = np.where(dones_batched_np)[0]
            done_index = int(done_hits[0] + 1) if done_hits.size > 0 else int(rewards_batched_np.shape[0])

        T = int(done_index)
        if done_index > 0:
            idx = done_index - 1

            # Handle possible batch dimension by taking the first element
            def _first(x):
                return x[idx][0] if x.ndim > 1 else x[idx]

            print(
                f"Done flags at step {idx}: "
                f"goal={_first(done_goal_np)}, "
                f"nan={_first(done_nan_np)}, "
                f"unhealthy={_first(done_unhealthy_np)}"
            )
        frames = [jax.tree.map(lambda x, i=i: x[i], frames_batched) for i in range(T)]

        # Downsample per episode
        keep_idx = np.arange(0, T, frame_stride, dtype=int)
        frames = [frames[i] for i in keep_idx]

        # Compute per-episode cumulative (reset each episode)
        cum_rewards = np.cumsum(rewards_batched_np[:T])
        cum_costs = np.cumsum(costs_batched_np[:T])

        # Keep values aligned with kept frames
        rewards_kept = rewards_batched_np[:T][keep_idx]
        costs_kept = costs_batched_np[:T][keep_idx]
        cum_rewards_kept = cum_rewards[keep_idx]
        cum_costs_kept = cum_costs[keep_idx]

        # Stash
        all_frames.extend(frames)
        # Store tuples so we can overlay later without recomputing
        all_rewards.extend(list(zip(rewards_kept, cum_rewards_kept)))
        all_costs.extend(list(zip(costs_kept, cum_costs_kept)))

    print("Rollouts took %.2f seconds." % (os.times()[4] - start_time[4]))
    start_time = os.times()

    # 4) Render the concatenated episodes
    for camera in cameras:
        rendering = env.render(all_frames, width=width, height=height, camera=camera)
        print("Rendering took %.2f seconds." % (os.times()[4] - start_time[4]))

        # 5) Add reward/cost overlay
        if show_metrics:
            start_time = os.times()
            rendering_with_metrics = []
            try:
                # Try to load a font, fallback to default if not available
                font_obj = ImageFont.truetype(f"{font}.ttf", 20)
            except (OSError, IOError):
                font_obj = ImageFont.load_default()

            # Flatten stored rewards/costs tuples for overlay
            # all_rewards[i] = (inst_reward, cum_reward); all_costs[i] likewise
            for i, frame in enumerate(rendering):
                r_inst, r_cum = all_rewards[i]
                c_inst, c_cum = all_costs[i]

                img = Image.fromarray(frame.astype(np.uint8))
                draw = ImageDraw.Draw(img)

                reward_text = f"Reward: {float(r_cum):.2f}"
                cost_text = f"Cost: {float(c_cum):.2f}"

                text_color_reward = (50, 220, 50)  # Green text
                text_color_cost = (230, 60, 60)  # Red text
                outline_color = (0, 0, 0)  # Black outline

                x_rew, y_rew = 10, 10
                x_cost, y_cost = 10, 40

                draw.text((x_rew, y_rew), reward_text, font=font_obj, fill=text_color_reward, stroke_width=2,
                          stroke_fill=outline_color)
                draw.text((x_cost, y_cost), cost_text, font=font_obj, fill=text_color_cost, stroke_width=2,
                          stroke_fill=outline_color)

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


def make_periodic_vision_video_fn(
        env,
        every_steps: int,
        steps: int = 300,
        pixel_camera: str = 'vision',
        extra_cameras: Optional[List[str]] = None,
        high_res: bool = True,
        frame_stack: int = 1,
        width: int = 320,
        height: int = 240,
        fps: int = 30,
        out_dir: str = "videos",
        run_name: str = "run",
        deterministic: bool = False,
        log_to_wandb: bool = True,
        seed: int = 0,
):
    """Builds a `policy_params_fn`-compatible callback that periodically
    renders short clips and logs them to wandb during training.

    The rollout always drives its actions off `env`'s real GPU-rendered
    (MJWarp) pixel observations (the same ones the policy is actually
    trained on). So `env` must stay wrapped with `GpuPixelObservationWrapper`
    at the policy's native training resolution (e.g. via
    `envs.get_environment(..., vision=True, vision_kwargs=...)`) with
    `num_envs=1` and the same camera/height/width/obs_mode/frame_stack it was
    trained with, or actions won't match training behavior.

    That's independent from what resolution the video itself is saved at,
    controlled by `high_res`:
      - `high_res=True` (default): `pixel_camera`'s video is rendered at
        `width`x`height` (e.g. 320x240, not the policy's tiny training
        resolution, e.g. 64x64) via a CPU MuJoCo render pass over the
        rollout's own `pipeline_state`s.
      - `high_res=False`: `pixel_camera`'s video instead reads frames straight
        off the low-res GPU pixel observations already computed to drive the
        policy. We get zero-cost, but it's capped at the training resolution.

    `extra_cameras`, if given, adds further videos from the *same* rollout for
    parity with vector-obs training's multi-camera video (e.g. `--cameras
    fixedfar vision`): these never feed the policy (only `pixel_camera`'s
    GPU observations do) and are always rendered the `high_res=True` way,
    regardless of the `high_res` setting for `pixel_camera` itself.

    `ppo.train()` calls `policy_params_fn(step, make_policy, params)` once
    per eval iteration (see crax/training/agents/ppo/train.py), so the
    actual cadence is bounded below by the eval interval (`num_evals`) --
    this can't fire more often than evals run, only skip evals until
    `every_steps` has elapsed.

    Args:
        env: single-env (num_envs=1), vision-wrapped rollout environment.
        every_steps: minimum env-step gap between clips. <=0 disables.
        steps: env steps to render per clip (kept short; this runs many
            times over a training run).
        pixel_camera: the camera `env` is wrapped with. Drives the policy's
            actual observations always. Also one of the logged clips (at
            `width`x`height` if `high_res`, else the training resolution).
        extra_cameras: additional camera names to log alongside
            `pixel_camera`, cosmetic only (see above). None/[] skips the CPU
            render pass entirely when `high_res=False` (same cost as before
            this arg existed). With `high_res=True` the pass is already
            happening for `pixel_camera` so extra cameras ride along nearly
            free.
        high_res: see above.
        frame_stack: must match the env's frame_stack; only used when
            `high_res=False`, to keep the most recent RGB frame per step (a
            raw frame_stack>1 observation encodes stale history in its extra
            channels, not a paintable image).
        width, height: render size for `high_res=True` clips (`pixel_camera`
            when enabled, and always for `extra_cameras`).
        fps: output video fps.
        out_dir: local directory for the mp4s (also uploaded to wandb).
        run_name: filename prefix.
        deterministic: whether to use the deterministic policy for the clip.
        log_to_wandb: whether to wandb.log the clip (skipped if wandb.run
            is None even when True).
        seed: seed for the rollout RNG stream (advances across calls).
    """
    extra_cameras = list(extra_cameras or [])
    cpu_cameras = list(extra_cameras) + ([pixel_camera] if high_res else [])
    obs_key = f'pixels/{pixel_camera}'
    os.makedirs(out_dir, exist_ok=True)

    mj_model = None
    cpu_renderer = None
    if cpu_cameras:
        mj_model = env.sys.mj_model
        cpu_renderer = mujoco.Renderer(mj_model, height=height, width=width)

    state = {'key': jax.random.PRNGKey(seed), 'last_bucket': -1, 'jit_rollout': None}

    def _build_rollout(make_policy):
        def rollout(params, key):
            policy = make_policy(params, deterministic=deterministic)
            reset_key, roll_key = jax.random.split(key)
            init_state = env.reset(reset_key)

            def step_body(carry, _):
                carry_state, k = carry
                k, sk = jax.random.split(k)
                action, _ = policy(carry_state.obs, sk)
                next_state = env.step(carry_state, action)
                # pipeline_state is retained even when extra_cameras is empty
                # (it's already computed as part of next_state -- free) so a
                # later call can add extra_cameras without rebuilding the JIT.
                return (next_state, k), (next_state.obs[obs_key], next_state.pipeline_state)

            (_, _), (frames, pipeline_states) = jax.lax.scan(
                step_body, (init_state, roll_key), xs=None, length=steps
            )
            return frames, pipeline_states  # (steps, num_envs=1, ...) each

        return jax.jit(rollout)

    def _cpu_render_frame(mj_model, single_pipeline_state, camera):
        d = mujoco.MjData(mj_model)
        d.qpos[:] = np.asarray(single_pipeline_state.qpos)
        d.qvel[:] = np.asarray(single_pipeline_state.qvel)
        if mj_model.nmocap > 0:
            d.mocap_pos[:] = np.asarray(single_pipeline_state.mocap_pos)
            d.mocap_quat[:] = np.asarray(single_pipeline_state.mocap_quat)
        mujoco.mj_forward(mj_model, d)
        cpu_renderer.update_scene(d, camera=camera)
        return cpu_renderer.render().copy()

    def video_fn(step: int, make_policy, params, force: bool = False):
        if every_steps <= 0 and not force:
            return
        bucket = step // every_steps if every_steps > 0 else 0
        if not force and bucket <= state['last_bucket']:
            return
        state['last_bucket'] = bucket

        if state['jit_rollout'] is None:
            # Built lazily (once) on first call and reused for every later
            # call -- `make_policy` is the same object every time ppo.train()
            # invokes this, so this is a one-time JIT compile, not per-clip.
            state['jit_rollout'] = _build_rollout(make_policy)

        state['key'], call_key = jax.random.split(state['key'])
        frames, pipeline_states = jax.device_get(state['jit_rollout'](params, call_key))

        clips = {}
        if not high_res:
            # Zero-cost path: read straight off the GPU pixel obs that already
            # drove the policy, at the training resolution.
            pixel_frames = np.asarray(frames)[:, 0]  # drop num_envs=1 axis -> (T, H, W, C)
            if frame_stack > 1:
                pixel_frames = pixel_frames[..., -3:]  # most recent RGB frame only
            clips[pixel_camera] = pixel_frames

        for camera in cpu_cameras:
            clips[camera] = np.stack([
                _cpu_render_frame(
                    mj_model,
                    jax.tree_util.tree_map(lambda x, t=t: x[t, 0], pipeline_states),
                    camera,
                )
                for t in range(steps)
            ])

        for camera, camera_frames in clips.items():
            mp4_path = os.path.join(out_dir, f"{run_name}_step{step}_{camera}.mp4")
            iio.imwrite(mp4_path, camera_frames, fps=fps)
            print(f"[periodic vision video] step {step}: saved {mp4_path}")

            if log_to_wandb and wandb.run is not None:
                wandb.log({f"video/{camera}": wandb.Video(mp4_path, fps=fps, format="mp4")}, step=step)

    return video_fn


def record_episode_video_simple(
        env,
        steps: int = 500,
        policy=None,
        action_mode: str = "random",
        cameras: List[str] | List[int] = (0,),
        width: int = 320,
        height: int = 240,
        fps: int = 50,
        frame_stride: int = 1,
        out_name: str = "rollout",
        seed: int = 0,
        show_metrics: bool = True,
        font: str = "DejaVuSans-Bold",
        num_episodes: int = 1,
        extra_metrics: Optional[List[str]] = None,
):
    """
    Record episode video with optional policy or simple action modes.

    This is a simpler version of record_episode_video that doesn't require
    make_inference_fn + params. Useful for tests and quick visualizations.

    Args:
        env: The environment to record.
        steps: Number of steps per episode.
        policy: Optional callable (obs, rng) -> (action, extra). If None, uses action_mode.
        action_mode: How to generate actions if policy is None.
            - "random": Uniform random actions in [-1, 1].
            - "zero": Zero actions.
            - "periodic": Sinusoidal periodic actions.
        cameras: List of camera names or IDs to render.
        width: Video width.
        height: Video height.
        fps: Video frames per second.
        frame_stride: Only keep every N frames.
        out_name: Base name for output video file.
        seed: Random seed.
        show_metrics: Whether to overlay reward/cost text on frames.
        font: Font name for overlay text.
        num_episodes: Number of episodes to record.
        extra_metrics: Optional list of additional metric names to display.

    Returns:
        Path to the saved video file.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    if policy is not None:
        jit_policy = jax.jit(policy)
    else:
        jit_policy = None

    action_size = env.action_size

    def get_action(obs, rng, t):
        if jit_policy is not None:
            action, _ = jit_policy(obs, rng)
            return action
        elif action_mode == "random":
            return jax.random.uniform(rng, (action_size,), minval=-1.0, maxval=1.0)
        elif action_mode == "zero":
            return jnp.zeros(action_size)
        elif action_mode == "periodic":
            phases = jnp.linspace(0.0, 2 * jnp.pi, num=action_size, endpoint=False)
            omega = 0.2
            scale = 0.5
            return scale * jnp.sin(omega * t + phases)
        else:
            raise ValueError(f"Unknown action_mode: {action_mode}")

    all_frames = []
    all_rewards = []
    all_costs = []
    all_extra = {k: [] for k in (extra_metrics or [])}

    key = jax.random.PRNGKey(seed)

    for ep in range(max(1, int(num_episodes))):
        key, ep_key = jax.random.split(key)
        state = reset_fn(ep_key)

        ep_frames = []
        ep_rewards = []
        ep_costs = []
        ep_extra = {k: [] for k in (extra_metrics or [])}

        cum_reward = 0.0
        cum_cost = 0.0

        for t in range(steps):
            key, step_key = jax.random.split(key)
            action = get_action(state.obs, step_key, t)
            state = step_fn(state, action)

            ep_frames.append(state.pipeline_state)

            r = float(np.asarray(jax.device_get(state.reward)).reshape(()))
            c = state.info.get("cost", state.metrics.get("cost", jnp.array(0.0)))
            c = float(np.asarray(jax.device_get(c)).reshape(()))

            cum_reward += r
            cum_cost += c
            ep_rewards.append((r, cum_reward))
            ep_costs.append((c, cum_cost))

            for mk in (extra_metrics or []):
                val = state.metrics.get(mk, state.info.get(mk, jnp.array(np.nan)))
                ep_extra[mk].append(float(np.asarray(jax.device_get(val)).reshape(())))

            done = bool(np.asarray(jax.device_get(state.done)).reshape(()))
            if done or np.isnan(r):
                break

        keep_idx = list(range(0, len(ep_frames), frame_stride))
        all_frames.extend([ep_frames[i] for i in keep_idx])
        all_rewards.extend([ep_rewards[i] for i in keep_idx])
        all_costs.extend([ep_costs[i] for i in keep_idx])
        for mk in (extra_metrics or []):
            all_extra[mk].extend([ep_extra[mk][i] for i in keep_idx])

        print(f"Episode {ep + 1}/{num_episodes}: {len(ep_frames)} steps, "
              f"reward={cum_reward:.2f}, cost={cum_cost:.2f}")

    video_path = None
    for camera in cameras:
        rendering = env.render(all_frames, width=width, height=height, camera=camera)

        if show_metrics:
            try:
                font_obj = ImageFont.truetype(f"{font}.ttf", 14)
            except (OSError, IOError):
                font_obj = ImageFont.load_default()

            out_frames = []
            for i, frame in enumerate(rendering):
                _, r_cum = all_rewards[i]
                _, c_cum = all_costs[i]

                img = Image.fromarray(frame.astype(np.uint8))
                draw = ImageDraw.Draw(img)

                texts = [
                    (f"Reward: {r_cum:.2f}", (50, 220, 50)),
                    (f"Cost: {c_cum:.2f}", (230, 60, 60)),
                ]
                for mk in (extra_metrics or []):
                    val = all_extra[mk][i]
                    texts.append((f"{mk}: {val:.3f}", (200, 200, 255)))

                for idx, (txt, color) in enumerate(texts):
                    draw.text((10, 10 + 25 * idx), txt, font=font_obj, fill=color,
                              stroke_width=2, stroke_fill=(0, 0, 0))

                out_frames.append(np.array(img))
            rendering = out_frames

        os.makedirs("videos", exist_ok=True)
        file_name = f"{out_name}_{camera}.mp4" if len(cameras) > 1 else f"{out_name}.mp4"
        video_path = os.path.join("videos", file_name)
        iio.imwrite(video_path, np.stack(rendering), fps=fps)
        print(f"Saved video: {video_path}")

    return video_path
