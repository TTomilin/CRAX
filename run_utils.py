import inspect
import os
import time
from datetime import datetime
from typing import Optional, Dict, Any, List

import jax
import mujoco
import numpy as np
import wandb
from PIL import ImageFont, Image, ImageDraw
from imageio import v3 as iio
from jax import numpy as jnp
from matplotlib import pyplot as plt

from brax import envs
from brax.io import json as brax_json
from brax.training.agents.focops import train as focops
from brax.training.agents.p3o import train as p3o
from brax.training.agents.ppo import train_ppo_cost
from brax.training.agents.ppo.train import train as ppo_train
from brax.training.agents.ppo_lag import train as ppo_lag
from brax.training.agents.ppo_pid import train as ppo_pid
from brax.training.agents.ppo_saute import train as ppo_saute

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
            nan_done = jnp.isnan(reward) | jnp.any(jnp.isnan(next_state.obs))
            done_base = jnp.asarray(next_state.done, dtype=bool)
            done_flag = jnp.logical_or(done_base, nan_done)

            return (next_state, key), (frame, reward, cost, done_flag)

        (final_state, _), (frames, rewards, costs, dones) = jax.lax.scan(
            step_body, (state, key), xs=None, length=steps
        )
        return frames, rewards, costs, dones

    # 3) Run N episodes to collect frames
    key = jax.random.PRNGKey(seed)

    all_frames = []  # list of pipeline_state lists (concatenated later)
    all_rewards = []  # list of np arrays after downsampling per episode
    all_costs = []  # list of np arrays after downsampling per episode

    for ep in range(max(1, int(num_episodes))):
        key, ep_key = jax.random.split(key)
        frames_batched, rewards_batched, costs_batched, dones_batched = rollout_one(ep_key)

        frames_batched = jax.device_get(frames_batched)
        rewards_batched_np = np.asarray(jax.device_get(rewards_batched))
        costs_batched_np = np.asarray(jax.device_get(costs_batched))
        dones_batched_np = np.asarray(jax.device_get(dones_batched)).astype(bool)

        # Trim to first termination if any
        if dones_batched_np.ndim == 0:
            done_index = int(dones_batched_np)
        else:
            done_hits = np.where(dones_batched_np)[0]
            done_index = int(done_hits[0] + 1) if done_hits.size > 0 else int(rewards_batched_np.shape[0])

        T = int(done_index)
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
