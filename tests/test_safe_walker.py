import argparse
import os
import sys
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from brax.envs.safe_walker import SafeWalker
from brax.training.agents.ppo import checkpoint as ppo_checkpoint

DEF_CKPT_ENV_VARS = [
    "SAFE_WALKER_CKPT",
    "SAFE_WALKER_PPO_CKPT",
    "CKPT_PATH",
]


def _maybe_get_checkpoint_path():
    for env in DEF_CKPT_ENV_VARS:
        p = os.environ.get(env, "").strip()
        if p:
            return p
    # Try a couple of common places used in this repo layout
    candidates = [
        "wandb/latest-run/files/ppo/",  # if organized under algo name
        "wandb/latest-run/files/",  # generic
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.isdir(c):
            return c
    return None


def _load_policy_or_skip():
    ckpt = _maybe_get_checkpoint_path()
    if not ckpt or not os.path.exists(ckpt):
        print(
            "Skipping test_safe_walker: set SAFE_WALKER_CKPT to a PPO checkpoint directory.\n"
            "Looked for: {} and default candidates.".format(DEF_CKPT_ENV_VARS)
        )
        # Graceful exit for non-pytest usage; if using pytest, mark xfail
        if "pytest" in sys.modules:
            import pytest
            pytest.skip("No checkpoint found for SafeWalker policy rollouts")
        else:
            raise SystemExit(0)
    try:
        inference = ppo_checkpoint.load_policy(ckpt, deterministic=True)
    except Exception as e:
        print(f"Failed to load policy from '{ckpt}': {e}")
        if "pytest" in sys.modules:
            import pytest
            pytest.skip("Could not load policy from checkpoint")
        else:
            raise SystemExit(0)
    return inference


def _load_policy_from_model_arg(model_arg: str):
    """
    Loads a policy by resolving a checkpoint path from a CLI-specified model name/path.
    Resolution:
      1) Use the provided path if it exists
      2) Fallback to models/<model_arg>
    Exits the script gracefully on failure.
    """
    candidates = []
    # As-is (absolute or relative path)
    if model_arg:
        candidates.append(model_arg)
    # Under repo-root models directory
    candidates.append(os.path.join("models", model_arg))

    ckpt = None
    for c in candidates:
        if os.path.exists(c):
            ckpt = c
            break

    if ckpt is None:
        print(
            f"Model path not found. Tried: {candidates}.\n"
            f"Provide a valid checkpoint directory name under models/ or a full path."
        )
        raise SystemExit(2)

    try:
        return ppo_checkpoint.load_policy(ckpt, deterministic=True)
    except Exception as e:
        print(f"Failed to load policy from '{ckpt}': {e}")
        raise SystemExit(2)


def run_policy_episodes(env, policy, steps_per_ep: int = 200, num_episodes: int = 1, seed: int = 0):
    print(f"Starting {num_episodes} episodes using loaded policy")
    jit_step = jax.jit(env.step)
    jit_reset = jax.jit(env.reset)
    jit_policy = jax.jit(policy)

    all_states = []
    all_rewards = []
    all_costs = []

    key = jax.random.PRNGKey(seed)

    t0 = time.perf_counter()

    for ep in range(num_episodes):
        key, ep_key = jax.random.split(key)
        state = jit_reset(ep_key)
        if ep == 0:
            all_states.append(state)

        cum_r = 0.0
        cum_c = 0.0

        step_key = ep_key
        for _ in range(steps_per_ep):
            step_key, k = jax.random.split(step_key)
            action, _ = jit_policy(state.obs, k)
            state = jit_step(state, action)
            all_states.append(state)

            r = float(np.asarray(jax.device_get(state.reward)).reshape(()))
            c = state.info.get("cost", state.metrics.get("cost", jnp.array(0.0)))
            c = float(np.asarray(jax.device_get(c)).reshape(()))

            cum_r += r
            cum_c += c
            all_rewards.append((r, cum_r))
            all_costs.append((c, cum_c))

        print(f"Finished episode {ep + 1}/{num_episodes}")

    rollout_s = time.perf_counter() - t0
    print(f"[timing] rollout: {rollout_s:.3f}s  ({num_episodes*steps_per_ep} steps)")

    print("Finished all episodes")
    return all_states, all_rewards, all_costs


def render_states(env, states, rewards, costs, base_agent_name="safe_walker_policy",
                  camera="fixedfar", width=320, height=240, fps=100,
                  show_metrics=True, font="DejaVuSans-Bold"):
    t0 = time.perf_counter()
    pipeline_states = [s.pipeline_state for s in states]

    t_render0 = time.perf_counter()
    frames = env.render(
        pipeline_states,
        width=width,
        height=height,
        camera=camera,
    )
    render_s = time.perf_counter() - t_render0

    if show_metrics:
        try:
            font_obj = ImageFont.truetype(f"{font}.ttf", 14)
        except (OSError, IOError):
            font_obj = ImageFont.load_default()

        out = []
        for i, frame in enumerate(frames):
            if i == 0:
                r_cum = 0.0
                c_cum = 0.0
            else:
                _, r_cum = rewards[i - 1]
                _, c_cum = costs[i - 1]

            img = Image.fromarray(frame.astype(np.uint8))
            draw = ImageDraw.Draw(img)

            reward_text = f"Reward: {r_cum:.2f}"
            cost_text = f"Cost: {c_cum:.2f}"

            draw.text((10, 10), reward_text, font=font_obj, fill=(50, 220, 50),
                      stroke_width=2, stroke_fill=(0, 0, 0))
            draw.text((10, 40), cost_text, font=font_obj, fill=(230, 60, 60),
                      stroke_width=2, stroke_fill=(0, 0, 0))

            out.append(np.array(img))
        frames = out

    os.makedirs("videos", exist_ok=True)
    path = f"videos/{base_agent_name}.mp4"
    t_write0 = time.perf_counter()
    iio.imwrite(path, np.stack(frames), fps=fps)
    write_s = time.perf_counter() - t_write0
    print("Saved video", path)
    total_s = time.perf_counter() - t0
    print(f"[timing] render: {render_s:.3f}s | write: {write_s:.3f}s | total video: {total_s:.3f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run SafeWalker rollouts with a trained PPO policy")
    parser.add_argument("--model", type=str, default=None, help="Checkpoint path or file name located under models/")
    parser.add_argument("--steps", type=int, default=1000, help="Steps per episode")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for episodes")
    parser.add_argument("--camera", type=str, default="fixedfar", help="Camera name for rendering")
    parser.add_argument("--out", type=str, default="safe_walker_policy", help="Video base name (without extension)")
    parser.add_argument("--cpu", action="store_true", help="Force JAX to run on CPU (set JAX_PLATFORM_NAME=cpu)")
    args = parser.parse_args()

    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"

    import jax
    import jax.numpy as jnp


    kwargs = {
        "cost": {
            "scaler": 0.1
        },
        "reward": {
            "scaler": 0.01
        }
    }
    env = SafeWalker(**kwargs)
    if args.model:
        root_dir = Path(__file__).parent.parent.resolve()
        model_path = root_dir / "models" / args.model
        policy = ppo_checkpoint.load_policy(model_path)
    else:
        def policy(obs, rng):
            act = jax.random.uniform(rng, (env.action_size,), minval=-1.0, maxval=1.0)
            return act, None

    states, rewards, costs = run_policy_episodes(env, policy, steps_per_ep=args.steps, num_episodes=args.episodes, seed=args.seed)
    render_states(env, states, rewards, costs, base_agent_name=args.out, camera=args.camera)
