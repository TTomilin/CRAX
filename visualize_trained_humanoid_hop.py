#!/usr/bin/env python3
"""Visualize a trained HumanoidHop policy to see constraint behavior."""

import jax
import jax.numpy as jnp
from brax import envs
import pickle
import os
import sys

# For rendering
from brax.io import html
import imageio.v3 as iio
import numpy as np
from jax import lax


def load_trained_policy(checkpoint_path):
    """Load a trained policy from checkpoint."""
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    # Extract make_inference_fn and params
    make_inference_fn = checkpoint_data.get('make_inference_fn')
    params = checkpoint_data.get('params')
    
    if make_inference_fn is None or params is None:
        raise ValueError("Checkpoint must contain 'make_inference_fn' and 'params'")
    
    return make_inference_fn, params


def visualize_trained_policy(
    checkpoint_path,
    hopping_leg='left',
    n_steps=500,
    save_html=True,
    save_video=True,
    video_name=None
):
    """Visualize a trained policy on HumanoidHop environment.
    
    Args:
        checkpoint_path: Path to saved checkpoint (.pkl file)
        hopping_leg: Which leg should be hopping
        n_steps: Number of steps to simulate
        save_html: Save interactive HTML visualization
        save_video: Save MP4 video
        video_name: Custom video name (default: auto-generated)
    """
    print(f"\n{'='*60}")
    print(f"Visualizing TRAINED Policy on HumanoidHop")
    print(f"Hopping leg: {hopping_leg}")
    print(f"{'='*60}\n")
    
    # Load policy
    print(f"Loading policy from: {checkpoint_path}")
    make_inference_fn, params = load_trained_policy(checkpoint_path)
    inference_fn = make_inference_fn(params)
    jit_inference_fn = jax.jit(inference_fn)
    
    # Create environment
    env = envs.create(
        'humanoid_hop',
        episode_length=2000,
        hopping_leg=hopping_leg,
        contact_threshold=0.5,
        backend='mjx'
    )
    
    print(f"Environment created:")
    print(f"  - Observation dim: {env.observation_size}")
    print(f"  - Action dim: {env.action_size}")
    print(f"  - Hopping leg: {hopping_leg}\n")
    
    # Run rollout
    print(f"Running {n_steps} steps with TRAINED policy...")
    
    rng = jax.random.PRNGKey(42)
    state = jax.jit(env.reset)(rng)
    
    states = [state]
    total_reward = 0.0
    total_cost = 0.0
    violations = 0
    
    for step in range(n_steps):
        rng, action_rng = jax.random.split(rng)
        
        # Use trained policy to select action
        action, _ = jit_inference_fn(state.obs, action_rng)
        state = jax.jit(env.step)(state, action)
        states.append(state)
        
        # Track metrics
        reward = float(state.reward)
        cost = float(state.info.get('cost', 0.0))
        violation = float(state.info.get('contact_violation', 0.0))
        
        total_reward += reward
        total_cost += cost
        violations += int(violation > 0.5)
        
        if step % 50 == 0:
            left_force = float(state.info.get('left_foot_contact_force', 0.0))
            right_force = float(state.info.get('right_foot_contact_force', 0.0))
            print(f"  Step {step:3d}: Reward={reward:.2f}, Cost={cost:.2f}, "
                  f"L_force={left_force:.1f}N, R_force={right_force:.1f}N")
    
    print(f"\n{'='*60}")
    print(f"Rollout Statistics (TRAINED POLICY):")
    print(f"{'='*60}")
    print(f"  Total reward:        {total_reward:.1f}")
    print(f"  Average reward:      {total_reward/n_steps:.3f}")
    print(f"  Total cost:          {total_cost:.1f}")
    print(f"  Average cost:        {total_cost/n_steps:.3f}")
    print(f"  Violation rate:      {100*violations/n_steps:.1f}% ({violations}/{n_steps} steps)")
    print(f"{'='*60}\n")
    
    # Create output directory
    os.makedirs("videos", exist_ok=True)
    
    # Generate base name for outputs
    if video_name is None:
        video_name = f"humanoid_hop_trained_{hopping_leg}_leg"
    
    # Save HTML visualization
    if save_html:
        print("Generating HTML visualization...")
        html_path = f"videos/{video_name}.html"
        html_string = html.render(
            env.sys.tree_replace({'opt.timestep': env.dt}),
            states[:min(len(states), 1000)]
        )
        with open(html_path, 'w') as f:
            f.write(html_string)
        print(f"  ✓ HTML saved to: {html_path}")
    
    # Save MP4 video
    if save_video:
        print("\nGenerating MP4 video...")
        video_path = f"videos/{video_name}.mp4"
        
        # Use record_episode_video approach
        @jax.jit
        def rollout(key):
            state = env.reset(key)
            
            def step_fn(carry, _):
                state, key = carry
                key, sk = jax.random.split(key)
                action, _ = jit_inference_fn(state.obs, sk)
                next_state = env.step(state, action)
                return (next_state, key), next_state.pipeline_state
            
            (final_state, _), frames = lax.scan(
                step_fn, (state, key), xs=None, length=min(n_steps, 500)
            )
            return frames, final_state
        
        # Run rollout
        key = jax.random.PRNGKey(42)
        frames_batched, _ = rollout(key)
        frames_batched = jax.device_get(frames_batched)
        
        # Unstack to list
        leaves = jax.tree_util.tree_leaves(frames_batched)
        def index_t(t):
            return jax.tree.map(lambda x: x[t], frames_batched)
        
        frames_list = [index_t(t) for t in range(int(leaves[0].shape[0]))]
        
        # Render frames
        rendering = env.render(frames_list, width=640, height=480, camera=0)
        
        # Save video
        iio.imwrite(video_path, np.stack(rendering), fps=30)
        print(f"  ✓ Video saved to: {video_path}")
    
    print(f"\n{'='*60}")
    print("Visualization complete!")
    print(f"{'='*60}\n")
    
    return states


def visualize_random_for_comparison(hopping_leg='left', n_steps=200):
    """Run random policy for comparison to show what violations look like."""
    print(f"\n{'='*60}")
    print(f"Visualizing RANDOM Policy for Comparison")
    print(f"(This should show many constraint violations)")
    print(f"{'='*60}\n")
    
    env = envs.create(
        'humanoid_hop',
        episode_length=2000,
        hopping_leg=hopping_leg,
        contact_threshold=0.5,
        backend='mjx'
    )
    
    rng = jax.random.PRNGKey(42)
    state = jax.jit(env.reset)(rng)
    
    print(f"Running {n_steps} steps with RANDOM policy...")
    
    total_cost = 0.0
    violations = 0
    
    for step in range(n_steps):
        rng, action_rng = jax.random.split(rng)
        action = jax.random.uniform(action_rng, (env.action_size,), minval=-0.3, maxval=0.3)
        state = jax.jit(env.step)(state, action)
        
        cost = float(state.info.get('cost', 0.0))
        violation = float(state.info.get('contact_violation', 0.0))
        total_cost += cost
        violations += int(violation > 0.5)
    
    print(f"\n{'='*60}")
    print(f"Random Policy Statistics:")
    print(f"{'='*60}")
    print(f"  Total cost:          {total_cost:.1f}")
    print(f"  Average cost:        {total_cost/n_steps:.3f}")
    print(f"  Violation rate:      {100*violations/n_steps:.1f}% ({violations}/{n_steps} steps)")
    print(f"{'='*60}\n")
    
    print("→ Compare this to the trained policy to see the effect of the constraint!\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python visualize_trained_humanoid_hop.py <checkpoint_path> [left|right]")
        print("")
        print("Example:")
        print("  python visualize_trained_humanoid_hop.py runs/humanoid_hop_experiments/checkpoint_final.pkl left")
        print("")
        print("Optional: Add 'compare' to also run random policy for comparison")
        print("  python visualize_trained_humanoid_hop.py <checkpoint_path> left compare")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    hopping_leg = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] in ['left', 'right'] else 'left'
    show_comparison = 'compare' in sys.argv
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    # Visualize trained policy
    visualize_trained_policy(
        checkpoint_path=checkpoint_path,
        hopping_leg=hopping_leg,
        n_steps=500,
        save_html=True,
        save_video=True
    )
    
    # Optionally show random policy for comparison
    if show_comparison:
        visualize_random_for_comparison(hopping_leg=hopping_leg, n_steps=200)


