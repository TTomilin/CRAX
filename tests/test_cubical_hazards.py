import os

from brax.envs.SafePointGoal import default_config, SafePointGoal
from train_from_config import record_episode_video
from utils import make_circular_policy


def test_record_video_cubical():
    """Test recording a video of circular motion with cubical hazards."""
    num_hazards = 8
    out_name = f"cubical_hazards_{num_hazards}.mp4"

    print(f"\nRecording an episode with {num_hazards} cubical hazards...")
    start_time = os.times()

    cfg = default_config()
    cfg.goals.type = 'cube'
    cfg.goals.count = 1
    cfg.hazards.specs = [
        {"type": "cube", "count": num_hazards},
    ]

    # Create an environment with cylinder hazards
    env = SafePointGoal(cfg)
    print(f"Environment creation took {os.times()[4] - start_time[4]:.2f} seconds with {env._num_hazards} hazards")
    start_time = os.times()

    # Make the constant-action circular policy
    make_infer = make_circular_policy(
        action_dim=env.action_size,
        thrust=1.0,  # forward push along agent x
        yaw_rate=0.3,  # steady left turn -> circle
        thrust_idx=0,  # actuator 0 = site motor along x
        yaw_idx=1,  # actuator 1 = velocity actuator on hinge 'z'
    )
    print(f"Policy creation {os.times()[4] - start_time[4]:.2f} seconds")
    start_time = os.times()

    record_episode_video(
        env=env,
        make_inference_fn=make_infer,
        params=None,  # policy ignores params
        steps=2500,
        cameras=["fixedfar", "vision"],
        width=640,
        height=480,
        fps=100,
        frame_stride=1,
        out_name=out_name,
        log_to_wandb=False,
        seed=0,
        show_metrics=True,  # Enable cost display on video
    )
    print(f"✓ Test completed in {os.times()[4] - start_time[4]:.2f} seconds")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    test_record_video_cubical()
    print("\n" + "=" * 60)
    print("✓ Video successfully recorded with cubical hazards.")
