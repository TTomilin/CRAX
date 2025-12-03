import os

from brax.envs.SafePointGoal import SafePointGoal_MixedHazards
from train_from_config import record_episode_video
from utils import make_multi_directional_policy


def test_record_video_mixed_hazards():
    """Test recording a video of circular motion with cylinder hazards."""
    print("\nRecording an episode with mixed hazards...")
    start_time = os.times()

    # Create an environment with cylinder hazards
    env = SafePointGoal_MixedHazards()
    cube_hazards = env._hazard_manager.get_hazards_by_type("cube")
    cylinder_hazards = env._hazard_manager.get_hazards_by_type("cylinder")
    print(f"✓ Environment created in {os.times()[4] - start_time[4]:.2f} seconds with {env._num_hazards} hazards "
          f"({len(cube_hazards)} cubes, {len(cylinder_hazards)} cylinders)")
    start_time = os.times()
    steps = 1000

    # Make a deterministic policy
    make_infer = make_multi_directional_policy(
        action_dim=env.action_size,
        thrust=1.0,  # forward push along agent x
        thrust_idx=0,  # actuator 0 = site motor along x
        yaw_idx=1,  # actuator 1 = velocity actuator on hinge 'z'
        steps=steps,  # total steps in episode
        num_yaw_values=10,  # number of different yaw values to sample
        rng_seed=8,
    )
    print(f"✓ Policy created {os.times()[4] - start_time[4]:.2f} seconds")
    start_time = os.times()

    # Record the episode
    record_episode_video(
        env=env,
        make_inference_fn=make_infer,
        params=None,  # policy ignores params
        steps=steps,
        cameras=["fixedfar", "vision"],
        width=640,
        height=480,
        fps=100,
        frame_stride=1,
        out_name="mixed_hazards",
        log_to_wandb=False,
        seed=6,
        show_metrics=True,
    )
    print(f"✓ Test completed in {os.times()[4] - start_time[4]:.2f} seconds")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    test_record_video_mixed_hazards()
    print("\n" + "=" * 60)
    print("✓ Video successfully recorded with mixed hazards.")
