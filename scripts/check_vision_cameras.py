"""Sanity-check that every registered env's egocentric 'vision' camera (the
one GpuPixelObservationWrapper / `--vision` training actually renders from)
exists and is placed somewhere sensible.

For each registered env: verifies the camera exists in the model, then runs
a short random-action rollout and renders it from that camera to an mp4 so
placement can be eyeballed. Envs missing the camera are reported, not
silently skipped.

Usage:
    python scripts/check_vision_cameras.py
    python scripts/check_vision_cameras.py --envs ant safe_goal_point --steps 200
    python scripts/check_vision_cameras.py --camera fixedfar  # check a different camera
    python scripts/check_vision_cameras.py --downscale        # render at agent's 64x64 obs res
    python scripts/check_vision_cameras.py --downscale 84     # ... or a custom resolution
"""
import argparse
import os

import jax.numpy as jnp
import mujoco

from crax import envs
from run_utils import VISION_CAMERA_OVERRIDES, morphology_override, record_episode_video_simple


def _turn_policy(action_size, turn_rate):
    """Spins the agent in place: zero everywhere except the last actuator,
    held at a constant rate."""
    action = jnp.zeros(action_size).at[-1].set(turn_rate)

    def policy(obs, rng):
        del obs, rng
        return action, {}

    return policy


def _check_one_env(env_name, args, results):
    camera = args.camera or morphology_override(env_name, VISION_CAMERA_OVERRIDES) or 'vision'
    used_backend = 'mjx'
    try:
        env = envs.get_environment(env_name, level=args.level, backend='mjx')
    except Exception as e:
        print(f"  backend='mjx' failed ({e}); falling back to default backend")
        used_backend = 'default'
        try:
            env = envs.get_environment(env_name, level=args.level)
        except Exception as e2:
            print(f"  FAIL: could not construct env at all: {e2}")
            results.append((env_name, 'CTOR_FAIL', str(e2)))
            return

    try:
        mj_model = env.sys.mj_model
    except AttributeError:
        print("  N/A: not an MJCF/MuJoCo-backed env (no camera concept applies)")
        results.append((env_name, 'N/A', 'not MJCF-backed'))
        return

    cam_names = [mj_model.camera(i).name for i in range(mj_model.ncam)]
    cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cam_id == -1:
        print(f"  MISSING '{camera}' camera. Available cameras: {cam_names}")
        results.append((env_name, 'MISSING_CAMERA', f'available: {cam_names}'))
        return

    policy = None
    action_mode = args.action_mode
    if action_mode == 'turn':
        policy = _turn_policy(env.action_size, args.turn_rate)
        action_mode = None  # record_episode_video_simple ignores action_mode when policy is given

    # --downscale renders at the agent's actual --vision observation resolution
    # (square) instead of --width/--height, so what you eyeball is what the
    # policy actually sees, not an upscaled/prettier stand-in.
    render_width = args.downscale or args.width
    render_height = args.downscale or args.height
    out_suffix = f'_{render_width}x{render_height}' if args.downscale else ''

    try:
        video_path = record_episode_video_simple(
            env,
            steps=args.steps,
            policy=policy,
            action_mode=action_mode,
            cameras=[camera],
            width=render_width,
            height=render_height,
            fps=args.fps,
            out_name=f'vision_check_{env_name}{out_suffix}',
            seed=args.seed,
            show_metrics=False,
        )
        print(f"  OK ({used_backend} backend, camera='{camera}', {render_width}x{render_height}) -> {video_path}")
        results.append((env_name, 'OK', video_path))
    except Exception as e:
        print(f"  FAIL rendering: {e}")
        results.append((env_name, 'RENDER_FAIL', str(e)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--steps', type=int, default=150, help='random-action steps per env')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--downscale', type=int, nargs='?', const=64, default=None,
                         help="render at the agent's actual --vision observation resolution "
                              "instead of --width/--height (square, e.g. 64x64 to match "
                              "configs/training_config.py's --vision_height/--vision_width "
                              "defaults). Bare '--downscale' uses 64; '--downscale 84' uses 84.")
    parser.add_argument('--camera', type=str, default=None, help="camera name to check for every env")
    parser.add_argument('--level', type=int, default=1, help="difficulty level")
    parser.add_argument('--envs', type=str, nargs='*', default=None,
                         help='subset of env names to check (default: all registered envs)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--action-mode', type=str, default='periodic',
                         choices=['periodic', 'random', 'zero', 'turn'],
                         help="periodic (default) is gentler on floppy chains "
                              "(hopper/walker2d/humanoid/swimmer) than fully "
                              "independent per-step random torques, which can "
                              "blow the physics solver up within ~100 steps. "
                              "'turn' spins the agent in place.")
    parser.add_argument('--turn-rate', type=float, default=0.6,
                         help="constant action value for --action-mode turn (range depends "
                              "on the actuator; Point's turn actuator is ctrlrange [-1, 1])")
    args = parser.parse_args()

    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.makedirs('videos', exist_ok=True)

    env_names = args.envs or sorted(envs._envs.keys())
    results = []  # (env_name, status, detail)

    for env_name in env_names:
        print(f"\n=== {env_name} ===")
        try:
            _check_one_env(env_name, args, results)
        except Exception as e:
            # Belt-and-suspenders: any unforeseen crash reports and moves on
            # instead of taking down the whole sweep.
            print(f"  UNEXPECTED_FAIL: {e}")
            results.append((env_name, 'UNEXPECTED_FAIL', str(e)))

    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    for env_name, status, detail in results:
        suffix = f'  {detail}' if status != 'OK' else ''
        print(f'  {status:16s} {env_name:28s}{suffix}')

    n_ok = sum(1 for _, s, _ in results if s == 'OK')
    n_missing = sum(1 for _, s, _ in results if s == 'MISSING_CAMERA')
    camera_desc = f"'{args.camera}' (forced)" if args.camera else "each env's --vision training default"
    print(f'\n{n_ok}/{len(results)} envs rendered OK from {camera_desc}.')
    if n_missing:
        print(f'{n_missing} env(s) are MISSING the camera entirely — --vision '
              f'training will hard-fail on those until the XML is fixed.')
    print('Videos saved under videos/vision_check_<env_name>.mp4 — eyeball them '
          'for sane egocentric placement (not clipping through the body, '
          'reasonable height/angle, etc).')


if __name__ == '__main__':
    main()
