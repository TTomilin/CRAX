#!/usr/bin/env python3
"""
Render initial frames for a given environment across three difficulty levels and save images.

By default, targets the SafePointGoal environment (levels 1, 2, 3) and writes PNGs
under assets/envs.

Usage examples:
  python scripts/render_env_levels.py --env safe_point_goal --outdir assets/envs --width 640 --height 480
  python scripts/render_env_levels.py  # uses defaults

Notes:
- Attempts to auto-select a working MUJOCO_GL backend for headless rendering.
- For safe_point_goal, this script uses the dedicated constructors for Level 1/2/3.
- For other env names, it will try names with suffixes "_level_1", "_level_2", "_level_3"
  from the env registry if present. If unavailable, it will fall back to rendering just the base env.
"""
import argparse
import os
from pathlib import Path
from typing import List, Tuple

from jax import random as jrandom

from brax import envs
from brax.io import image as brax_image


def _try_backend(gl_backend: str) -> bool:
    os.environ["MUJOCO_GL"] = gl_backend
    try:
        test_env = envs.create(env_name="inverted_pendulum", auto_reset=False)
        key = jrandom.PRNGKey(0)
        state = test_env.reset(key)
        _ = brax_image.render(test_env.sys, [state.pipeline_state], height=32, width=32, fmt="png")
        return True
    except Exception:
        return False


def resolve_levels(env_name: str) -> List[Tuple[str, str]]:
    """Returns a list of (label, registry_name) to render for the given env.

    - For safe_point_goal: use dedicated constructors, labeled Level 1..3.
    - Otherwise: try env_name_level_1..3 from registry; include those that exist.
    """
    name_norm = env_name.strip().lower().replace("safety_point_goal", "safe_point_goal")

    if name_norm == "safe_point_goal":
        return [("level_1", name_norm + "_level_1"), ("level_2", name_norm + "_level_2"),
                ("level_3", name_norm + "_level_3")]

    # Generic: probe registry keys
    labels = []
    for k in (1, 2, 3):
        key = f"{name_norm}_level_{k}"
        if key in getattr(envs, "_envs", {}):
            labels.append((f"level_{k}", key))
    if labels:
        return labels

    # Fallback: no levels registered; just return the base env once
    return [("base", name_norm)]


def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render initial frames for env levels and save PNGs")
    p.add_argument("--env", type=str, default="safe_point_goal",
                   help="Base environment name (default: safe_point_goal)")
    p.add_argument("--outdir", type=str, default="../assets/envs", help="Output directory to store images")
    p.add_argument("--width", type=int, default=640, help="Image width")
    p.add_argument("--height", type=int, default=480, help="Image height")
    p.add_argument("--seed", type=int, default=0, help="Random seed for resets")
    p.add_argument("--camera", type=str, default="fixedfar", help="Optional camera name/index for rendering")
    return p


def main(args: argparse.Namespace) -> None:
    os.environ['MUJOCO_GL'] = 'egl'

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_name = args.env
    levels = resolve_levels(env_name)

    # PRNG
    key = jrandom.PRNGKey(args.seed)

    saved = []
    failed = []

    for label, registry_name in levels:
        try:
            env = envs.create(env_name=registry_name, auto_reset=False)

            key, sub = jrandom.split(key)
            state = env.reset(sub)

            img = brax_image.render(
                env.sys,
                [state.pipeline_state],
                height=args.height,
                width=args.width,
                camera=args.camera,
                fmt="png",
            )

            file_stem = f"{env_name.lower()}_{label}"
            out_path = out_dir / f"{file_stem}.png"
            with open(out_path, "wb") as f:
                f.write(img)
            print(f"Saved: {out_path}")
            saved.append(str(out_path))
        except Exception as e:
            print(f"Failed to render {registry_name}: {e}")
            failed.append((registry_name, str(e)))

    print("\nSummary:")
    print(f"  Saved {len(saved)} image(s).")
    if failed:
        print(f"  Failed ({len(failed)}): {[name for name, _ in failed]}")


if __name__ == "__main__":
    main(build_args().parse_args())
