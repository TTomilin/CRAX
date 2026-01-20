#!/usr/bin/env python3
"""
Render initial frames for a given environment across difficulty levels and save images.

Usage examples:
  python scripts/render_env_levels.py --env safe_point_goal --levels 1 2 3 --outdir assets/envs
  python scripts/render_env_levels.py --env safe_walker --levels 1 2 3
  python scripts/render_env_levels.py  # uses defaults (safe_point_goal, levels 1-3)

Notes:
- Uses the centralized difficulty system in brax.envs
- Automatically applies difficulty level overrides when creating environments
"""
import argparse
import os
from pathlib import Path

from jax import random as jrandom

from brax import envs
from brax.envs.difficulty import get_supported_levels, supports_difficulty
from brax.io import image as brax_image


def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render initial frames for env difficulty levels")
    p.add_argument("--env", type=str, default="safe_point_goal", help="Environment name")
    p.add_argument("--levels", type=int, nargs='+', default=None,
                   help="Levels to render (default: all supported levels for the env)")
    p.add_argument("--outdir", type=str, default="assets/envs", help="Output directory for images")
    p.add_argument("--width", type=int, default=640, help="Image width")
    p.add_argument("--height", type=int, default=480, help="Image height")
    p.add_argument("--seed", type=int, default=0, help="Random seed for environment reset")
    p.add_argument("--camera", type=str, default="fixedfar", help="Camera name/index for rendering")
    return p


def main(args: argparse.Namespace) -> None:
    os.environ['MUJOCO_GL'] = 'egl'

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_name = args.env

    # Determine which levels to render
    if args.levels is not None:
        levels = args.levels
    elif supports_difficulty(env_name):
        levels = get_supported_levels(env_name)
        print(f"Environment '{env_name}' supports levels: {levels}")
    else:
        levels = [None]  # No difficulty support - render base env
        print(f"Environment '{env_name}' does not support difficulty levels, rendering base env")

    key = jrandom.PRNGKey(args.seed)
    saved = []
    failed = []

    for level in levels:
        try:
            # Create environment with difficulty level
            env = envs.create(env_name=env_name, level=level, auto_reset=False)

            key, sub = jrandom.split(key)
            state = env.reset(sub)

            # Build render kwargs
            render_kwargs = {
                'height': args.height,
                'width': args.width,
                'fmt': 'png',
            }
            if args.camera:
                render_kwargs['camera'] = args.camera

            img = brax_image.render(
                env.sys,
                [state.pipeline_state],
                **render_kwargs,
            )

            # Generate output filename
            if level is not None:
                file_stem = f"{env_name}_level_{level}"
            else:
                file_stem = env_name

            out_path = out_dir / f"{file_stem}.png"
            with open(out_path, "wb") as f:
                f.write(img)

            level_str = f"level {level}" if level is not None else "base"
            print(f"Saved: {out_path} ({level_str})")
            saved.append(str(out_path))

        except Exception as e:
            level_str = f"level {level}" if level is not None else "base"
            print(f"Failed to render {env_name} {level_str}: {e}")
            failed.append((env_name, level, str(e)))

    print("\nSummary:")
    print(f"  Saved {len(saved)} image(s)")
    if failed:
        print(f"  Failed ({len(failed)}):")
        for env, lvl, err in failed:
            print(f"    - {env} level {lvl}: {err}")


if __name__ == "__main__":
    main(build_args().parse_args())
