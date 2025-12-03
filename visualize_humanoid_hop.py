#!/usr/bin/env python3
"""Visualization script for HumanoidHop environment."""

import jax
import jax.numpy as jnp
from brax import envs
from brax.io import html
from brax.io import image
from datetime import datetime
import os

# Optional: mediapy for video generation
try:
    import mediapy as media
    MEDIAPY_AVAILABLE = True
except ImportError:
    MEDIAPY_AVAILABLE = False
    print("Note: mediapy not installed. Video generation will be skipped.")

def visualize_random_policy(hopping_leg='left', n_steps=200, save_video=True):
    """Visualize humanoid hopping with random policy.
    
    Args:
        hopping_leg: Which leg to hop on ('left' or 'right')
        n_steps: Number of simulation steps
        save_video: If True, saves video to file
    """
    print(f"\n{'='*60}")
    print(f"Visualizing HumanoidHop: Hopping on {hopping_leg} leg")
    print(f"{'='*60}\n")
    
    # Create environment
    env = envs.create('humanoid_hop', 
                      episode_length=2000,
                      hopping_leg=hopping_leg,
                      contact_threshold=0.5,
                      backend='mjx')
    
    print(f"Environment created:")
    print(f"  - Observation dim: {env.observation_size}")
    print(f"  - Action dim: {env.action_size}")
    print(f"  - Hopping leg: {hopping_leg}")
    
    # Initialize
    rng = jax.random.PRNGKey(42)
    state = jax.jit(env.reset)(rng)
    
    # Collect trajectory
    print(f"\nRunning {n_steps} steps with random policy...")
    states = [state]
    total_cost = 0.0
    violations = 0
    
    for step in range(n_steps):
        rng, action_rng = jax.random.split(rng)
        # Use smaller action magnitudes for more stable behavior
        action = jax.random.uniform(action_rng, (env.action_size,), 
                                    minval=-0.3, maxval=0.3)
        state = jax.jit(env.step)(state, action)
        states.append(state)
        
        cost = float(state.info['cost'])
        violation = float(state.info['contact_violation'])
        total_cost += cost
        violations += int(violation > 0.5)
        
        if step % 50 == 0:
            left_force = float(state.info['left_foot_contact_force'])
            right_force = float(state.info['right_foot_contact_force'])
            print(f"  Step {step:3d}: Cost={cost:.2f}, "
                  f"L_force={left_force:.1f}N, R_force={right_force:.1f}N, "
                  f"Violations={violations}")
    
    print(f"\nTrajectory Statistics:")
    print(f"  - Total cost: {total_cost:.1f}")
    print(f"  - Violation rate: {100*violations/n_steps:.1f}% ({violations}/{n_steps} steps)")
    print(f"  - Average cost per step: {total_cost/n_steps:.3f}")
    
    # Create visualization
    print(f"\nGenerating visualization...")
    
    # Create HTML visualization
    html_path = f"videos/humanoid_hop_{hopping_leg}_leg.html"
    os.makedirs("videos", exist_ok=True)
    
    html_string = html.render(env.sys.tree_replace({'opt.timestep': env.dt}), 
                             states[:min(len(states), 500)])  # Limit to 500 frames
    
    with open(html_path, 'w') as f:
        f.write(html_string)
    
    print(f"  ✓ HTML visualization saved to: {html_path}")
    
    # Try to create video if mediapy is available
    if save_video and MEDIAPY_AVAILABLE:
        try:
            print(f"\nGenerating video frames...")
            video_path = f"videos/humanoid_hop_{hopping_leg}_leg.mp4"
            
            # Render frames
            frames = []
            for i, s in enumerate(states[::2]):  # Every 2nd frame to reduce size
                if i >= 250:  # Limit to 250 frames (500 steps / 2)
                    break
                if i % 25 == 0:
                    print(f"  Rendering frame {i}/250...")
                
                frame = image.render_array(
                    env.sys.tree_replace({'opt.timestep': env.dt}),
                    s.pipeline_state,
                    width=640,
                    height=480
                )
                frames.append(frame)
            
            # Save video
            media.write_video(video_path, frames, fps=50)
            print(f"  ✓ Video saved to: {video_path}")
            
        except Exception as e:
            print(f"  ⚠ Could not create video: {e}")
            print(f"  (Install mediapy: pip install mediapy)")
    elif save_video and not MEDIAPY_AVAILABLE:
        print(f"  ℹ Video generation skipped (mediapy not installed)")
        print(f"  Install with: pip install mediapy")
    
    return states, html_path

def compare_left_vs_right():
    """Create side-by-side comparison of left vs right leg hopping."""
    print("\n" + "="*60)
    print("Creating Comparison: Left vs Right Leg Hopping")
    print("="*60)
    
    results = {}
    for leg in ['left', 'right']:
        print(f"\n--- {leg.upper()} LEG ---")
        states, html_path = visualize_random_policy(
            hopping_leg=leg, 
            n_steps=150,
            save_video=True
        )
        results[leg] = {
            'states': states,
            'html_path': html_path
        }
    
    print("\n" + "="*60)
    print("Comparison Complete!")
    print("="*60)
    print("\nView the HTML files in your browser:")
    for leg, data in results.items():
        print(f"  - {leg} leg: {data['html_path']}")
    print("\nNote: The HTML files contain interactive 3D visualizations!")
    print("="*60 + "\n")

def quick_test():
    """Quick test with just a few steps to verify setup."""
    print("\n" + "="*60)
    print("Quick Visualization Test")
    print("="*60)
    
    env = envs.create('humanoid_hop', episode_length=1000, backend='mjx')
    rng = jax.random.PRNGKey(0)
    state = jax.jit(env.reset)(rng)
    
    print("\nTaking 20 steps...")
    states = [state]
    for _ in range(20):
        action = jnp.zeros(env.action_size)
        state = jax.jit(env.step)(state, action)
        states.append(state)
    
    print("Creating HTML visualization...")
    html_path = "videos/humanoid_hop_test.html"
    os.makedirs("videos", exist_ok=True)
    
    html_string = html.render(env.sys.tree_replace({'opt.timestep': env.dt}), states)
    with open(html_path, 'w') as f:
        f.write(html_string)
    
    print(f"✓ Test visualization saved to: {html_path}")
    print("Open this file in your browser to view the animation!\n")

if __name__ == '__main__':
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            quick_test()
        elif sys.argv[1] == 'compare':
            compare_left_vs_right()
        elif sys.argv[1] in ['left', 'right']:
            visualize_random_policy(hopping_leg=sys.argv[1], n_steps=200, save_video=True)
        else:
            print("Usage:")
            print("  python visualize_humanoid_hop.py test      # Quick test")
            print("  python visualize_humanoid_hop.py left      # Visualize left leg hopping")
            print("  python visualize_humanoid_hop.py right     # Visualize right leg hopping")
            print("  python visualize_humanoid_hop.py compare   # Compare both")
    else:
        # Default: create comparison
        compare_left_vs_right()

