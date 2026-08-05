#!/usr/bin/env python3
"""Visualize MuJoCo XML files.

This script provides multiple ways to visualize MuJoCo XML models:
1. Static image rendering (PNG)
2. Interactive HTML viewer
3. Simulated trajectory (GIF/MP4)

Usage:
    # Render a static image
    python scripts/visualize_xml.py crax/test_data/double_pendulum.xml

    # Render multiple XMLs
    python scripts/visualize_xml.py crax/test_data/*.xml --output-dir renders/

    # Create an interactive HTML viewer
    python scripts/visualize_xml.py crax/test_data/capsule.xml --mode html

    # Simulate and create a GIF
    python scripts/visualize_xml.py crax/test_data/triple_pendulum.xml --mode gif --steps 200

    # List all available XMLs in test_data
    python scripts/visualize_xml.py --list
"""

import argparse
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def find_xml_files(search_dirs=None):
    """Find all XML files in common locations."""
    if search_dirs is None:
        base = Path(__file__).parent.parent
        search_dirs = [
            base / "crax" / "test_data",
            base / "crax" / "envs" / "assets",
            base / "crax" / "envs" / "assets" / "safe",
        ]

    xml_files = []
    for d in search_dirs:
        if d.exists():
            xml_files.extend(sorted(d.glob("*.xml")))
    return xml_files


def render_static_image(xml_path: str, width: int = 640, height: int = 480,
                        camera: str = None):
    """Render a static image of the MuJoCo model."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Step once to initialize
    mujoco.mj_step(model, data)

    # Create renderer
    renderer = mujoco.Renderer(model, height=height, width=width)

    # Update scene and render
    renderer.update_scene(data, camera=camera)
    image = renderer.render()

    return image


def render_trajectory(xml_path: str, steps: int = 100, width: int = 640,
                      height: int = 480, camera: str = None,
                      random_actions: bool = True) -> list:
    """Simulate and render a trajectory."""
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Initialize
    mujoco.mj_resetData(model, data)

    # Create renderer
    renderer = mujoco.Renderer(model, height=height, width=width)

    frames = []
    rng = np.random.default_rng(42)

    for step in range(steps):
        # Apply random actions if the model has actuators
        if random_actions and model.nu > 0:
            ctrl_range = model.actuator_ctrlrange
            if ctrl_range.size > 0:
                low = ctrl_range[:, 0]
                high = ctrl_range[:, 1]
                data.ctrl[:] = rng.uniform(low, high)
            else:
                data.ctrl[:] = rng.uniform(-1, 1, size=model.nu)

        # Step simulation
        mujoco.mj_step(model, data)

        # Render frame
        renderer.update_scene(data, camera=camera)
        frame = renderer.render()
        frames.append(frame)

    return frames


def save_gif(frames: list, output_path: str, fps: int = 30):
    """Save frames as a GIF."""
    from PIL import Image
    images = [Image.fromarray(f) for f in frames]
    duration = int(1000 / fps)  # ms per frame
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )


def save_mp4(frames: list, output_path: str, fps: int = 30):
    """Save frames as an MP4 video."""
    try:
        import imageio.v3 as iio
        iio.imwrite(output_path, frames, fps=fps, codec='libx264')
    except ImportError:
        print("Warning: imageio not available. Install with: pip install imageio[ffmpeg]")
        # Fallback to GIF
        gif_path = output_path.replace('.mp4', '.gif')
        save_gif(frames, gif_path, fps)
        print(f"Saved as GIF instead: {gif_path}")


def create_html_viewer(xml_path: str, output_path: str = None, steps: int = 100):
    """Create an interactive HTML viewer using Brax's visualization."""
    import jax
    import jax.numpy as jp
    import mujoco
    from crax.io import mjcf, html

    model = mujoco.MjModel.from_xml_path(xml_path)
    sys = mjcf.load_model(model)

    # Create a simple trajectory by stepping the physics
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Import brax pipeline
    from crax.mjx import pipeline

    # Initialize pipeline state
    state = pipeline.init(sys, sys.init_q, jp.zeros(sys.qd_size()))

    states = [state]
    rng = jax.random.PRNGKey(42)

    for _ in range(steps):
        # Random action if there are actuators
        if sys.act_size() > 0:
            rng, subkey = jax.random.split(rng)
            action = jax.random.uniform(subkey, (sys.act_size(),), minval=-1, maxval=1)
        else:
            action = jp.zeros((max(1, sys.act_size()),))

        state = pipeline.step(sys, state, action)
        states.append(state)

    # Generate HTML
    html_content = html.render(sys, states)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(html_content)
        print(f"Saved HTML viewer: {output_path}")

    return html_content


def main():
    parser = argparse.ArgumentParser(
        description="Visualize MuJoCo XML files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "xml_files", nargs="*",
        help="XML file(s) to visualize. Supports glob patterns."
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available XML files in test_data and assets"
    )
    parser.add_argument(
        "--mode", choices=["png", "gif", "mp4", "html"], default="png",
        help="Output mode: png (static), gif (animated), mp4 (video), html (interactive)"
    )
    parser.add_argument(
        "--output-dir", "-o", default=".",
        help="Output directory for rendered files"
    )
    parser.add_argument(
        "--width", type=int, default=640,
        help="Image width"
    )
    parser.add_argument(
        "--height", type=int, default=480,
        help="Image height"
    )
    parser.add_argument(
        "--steps", type=int, default=100,
        help="Number of simulation steps for animated output"
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Frames per second for animated output"
    )
    parser.add_argument(
        "--camera", default=None,
        help="Camera name to use for rendering"
    )
    parser.add_argument(
        "--no-random", action="store_true",
        help="Don't apply random actions during simulation"
    )

    args = parser.parse_args()

    # List mode
    if args.list:
        print("Available XML files:\n")
        xml_files = find_xml_files()
        for f in xml_files:
            print(f"  {f}")
        print(f"\nTotal: {len(xml_files)} files")
        return

    # Validate inputs
    if not args.xml_files:
        parser.error("Please provide XML file(s) or use --list to see available files")

    # Expand glob patterns and validate paths
    xml_paths = []
    for pattern in args.xml_files:
        path = Path(pattern)
        if path.exists():
            xml_paths.append(path)
        else:
            # Try as glob pattern
            matches = list(Path(".").glob(pattern))
            if matches:
                xml_paths.extend(matches)
            else:
                print(f"Warning: No files found matching '{pattern}'")

    if not xml_paths:
        print("Error: No valid XML files found")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each file
    for xml_path in xml_paths:
        print(f"Processing: {xml_path}")

        try:
            base_name = xml_path.stem

            if args.mode == "png":
                from PIL import Image
                image = render_static_image(
                    str(xml_path),
                    width=args.width,
                    height=args.height,
                    camera=args.camera
                )
                output_path = output_dir / f"{base_name}.png"
                Image.fromarray(image).save(output_path)
                print(f"  Saved: {output_path}")

            elif args.mode == "gif":
                frames = render_trajectory(
                    str(xml_path),
                    steps=args.steps,
                    width=args.width,
                    height=args.height,
                    camera=args.camera,
                    random_actions=not args.no_random
                )
                output_path = output_dir / f"{base_name}.gif"
                save_gif(frames, str(output_path), fps=args.fps)
                print(f"  Saved: {output_path}")

            elif args.mode == "mp4":
                frames = render_trajectory(
                    str(xml_path),
                    steps=args.steps,
                    width=args.width,
                    height=args.height,
                    camera=args.camera,
                    random_actions=not args.no_random
                )
                output_path = output_dir / f"{base_name}.mp4"
                save_mp4(frames, str(output_path), fps=args.fps)
                print(f"  Saved: {output_path}")

            elif args.mode == "html":
                output_path = output_dir / f"{base_name}.html"
                create_html_viewer(str(xml_path), str(output_path), steps=args.steps)

        except Exception as e:
            print(f"  Error: {e}")
            continue

    print("\nDone!")


if __name__ == "__main__":
    sys.exit(main() or 0)
