"""GPU-based egocentric pixel observation wrapper using pixelbrax renderer.

Renders pixel observations fully on GPU using JAX rasterization — no CPU
callbacks.

Geom world poses are read directly from mjx.Data.geom_xpos / geom_xmat, which
MJX's forward pass (smooth.kinematics) computes for every geom — including geoms
whose parent bodies are mocap-controlled.  Camera pose comes from
mjx.Data.cam_xpos / cam_xmat, computed by smooth.camlight, also called by the
forward pass.  Both quantities are therefore always up-to-date without any manual
body-transform arithmetic on our side.

Usage:

    env = GpuPixelObservationWrapper(env, camera="vision")
    # → state.obs = {"pixels/vision": (H, W, 3) uint8}

If the environment XML has no camera named <camera_name> the wrapper raises
at construction time so you can add the camera element to the XML.

Observation key: 'pixels/<camera_name>'  (e.g. 'pixels/vision')
"""

import os
import sys
from collections import namedtuple
from typing import Dict, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from crax.envs.base import Env, State, Wrapper

# MuJoCo geom type constants
_MJ_GEOM_PLANE = 0
_MJ_GEOM_SPHERE = 2
_MJ_GEOM_CAPSULE = 3
_MJ_GEOM_CYLINDER = 5
_MJ_GEOM_BOX = 6
_MJ_GEOM_MESH = 7

# Lightweight struct: geom index in the MJX data arrays + pre-built mesh.
_GeomInfo = namedtuple('_GeomInfo', ['geom_idx', 'base_instance'])


def _build_renderer_objects(mj_model, geom_group_filter):
    """Build pixelbrax ModelObject list from MuJoCo model geometry.

    Each entry records the geom's index in mjx.Data.geom_xpos / geom_xmat so
    that world transforms can be read directly at render time — no manual
    body-transform arithmetic.

    Args:
        mj_model: mujoco.MjModel
        geom_group_filter: if not None, only render geoms in these groups.

    Returns:
        list of _GeomInfo named tuples.
    """
    _repo_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    from crax.renderer import create_capsule, create_cube, UpAxis
    from crax.renderer.model import Model as RendererMesh, ModelObject as Instance

    geom_infos = []
    for i in range(mj_model.ngeom):
        geom_type = int(mj_model.geom_type[i])
        geom_group = int(mj_model.geom_group[i])

        if geom_group_filter is not None and geom_group not in geom_group_filter:
            continue

        rgba = mj_model.geom_rgba[i].astype(np.float32)
        tex = jnp.array(rgba[:3].reshape(1, 1, 3))
        spec_map = jnp.full((1, 1), 2.0)
        size = mj_model.geom_size[i]

        try:
            if geom_type == _MJ_GEOM_CAPSULE:
                model = create_capsule(
                    radius=float(size[0]),
                    half_height=float(size[1]),
                    up_axis=UpAxis.Z,
                    diffuse_map=tex,
                    specular_map=spec_map,
                )
            elif geom_type == _MJ_GEOM_BOX:
                model = create_cube(
                    half_extents=jnp.array(size[:3], dtype=jnp.float32),
                    diffuse_map=tex,
                    texture_scaling=jnp.array(16.0),
                    specular_map=spec_map,
                )
            elif geom_type == _MJ_GEOM_SPHERE:
                model = create_capsule(
                    radius=float(size[0]),
                    half_height=0.0,
                    up_axis=UpAxis.Z,
                    diffuse_map=tex,
                    specular_map=spec_map,
                )
            elif geom_type == _MJ_GEOM_PLANE:
                # Render the ground plane as a large thin box using the geom's
                # actual colour rather than a hardcoded grey.
                model = create_cube(
                    half_extents=jnp.array([50.0, 50.0, 0.001], dtype=jnp.float32),
                    diffuse_map=tex,
                    texture_scaling=jnp.array(128.0),
                    specular_map=spec_map,
                )
            elif geom_type == _MJ_GEOM_CYLINDER:
                model = create_capsule(
                    radius=float(size[0]),
                    half_height=float(size[1]),
                    up_axis=UpAxis.Z,
                    diffuse_map=tex,
                    specular_map=spec_map,
                )
            elif geom_type == _MJ_GEOM_MESH:
                try:
                    import trimesh
                except ImportError:
                    continue
                mesh_id = int(mj_model.geom_dataid[i])
                if mesh_id < 0:
                    continue
                v_start = int(mj_model.mesh_vertadr[mesh_id])
                v_count = int(mj_model.mesh_vertnum[mesh_id])
                f_start = int(mj_model.mesh_faceadr[mesh_id])
                f_count = int(mj_model.mesh_facenum[mesh_id])
                verts = mj_model.mesh_vert[v_start: v_start + v_count]
                faces = mj_model.mesh_face[f_start: f_start + f_count]
                tm = trimesh.Trimesh(vertices=verts, faces=faces)
                model = RendererMesh.create(
                    verts=jnp.array(tm.vertices, dtype=jnp.float32),
                    norms=jnp.array(tm.vertex_normals, dtype=jnp.float32),
                    uvs=jnp.zeros((len(tm.vertices), 2), dtype=jnp.int32),
                    faces=jnp.array(tm.faces, dtype=jnp.int32),
                    diffuse_map=tex,
                )
            else:
                continue
        except Exception:
            continue

        geom_infos.append(_GeomInfo(
            geom_idx=i,
            base_instance=Instance(model=model),
        ))

    return geom_infos


class GpuPixelObservationWrapper(Wrapper):
    """Wraps a CRAX/MJX environment to add GPU-rendered pixel observations.

    Rendering is done entirely in JAX on the GPU using pixelbrax's rasterizer.
    All world transforms are read from mjx.Data fields that the MJX forward pass
    already computes:
      - geom_xpos / geom_xmat  (smooth.kinematics, always computed)
      - cam_xpos  / cam_xmat   (smooth.camlight, always computed)

    Observation key: 'pixels/<camera_name>'  (e.g. 'pixels/vision').

    Args:
        env: The environment to wrap.
        camera: Name of the MuJoCo camera to render from (must exist in XML).
        height: Render height in pixels.
        width: Render width in pixels.
        obs_mode: 'pixels', 'pixels+state', or 'state'.
        frame_stack: Number of frames to stack channel-wise.
        geom_group_filter: If given, only render geoms in these MuJoCo groups.
    """

    def __init__(
        self,
        env: Env,
        camera: str = 'vision',
        height: int = 64,
        width: int = 64,
        obs_mode: str = 'pixels',
        frame_stack: int = 1,
        geom_group_filter: Optional[Sequence[int]] = None,
    ):
        super().__init__(env)

        if obs_mode not in ('pixels', 'pixels+state', 'state'):
            raise ValueError(
                f"obs_mode must be 'pixels', 'pixels+state', or 'state', got '{obs_mode}'"
            )

        self._obs_key = f'pixels/{camera}'
        self._height = height
        self._width = width
        self._obs_mode = obs_mode
        self._frame_stack = max(1, frame_stack)
        self._channels = 3

        mj_model = self.sys.mj_model

        # Resolve camera index
        cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if cam_id == -1:
            available = [mj_model.camera(i).name for i in range(mj_model.ncam)]
            raise ValueError(
                f"Camera '{camera}' not found in model. Available: {available}"
            )

        # Vertical FOV from XML; derive horizontal FOV for non-square renders.
        vfov = float(mj_model.cam_fovy[cam_id])
        hfov = float(
            2.0 * np.degrees(
                np.arctan(np.tan(np.radians(vfov) / 2.0) * width / height)
            )
        )

        geom_infos = _build_renderer_objects(mj_model, geom_group_filter)
        if not geom_infos:
            raise RuntimeError(
                "GPU renderer: no renderable geometry found. "
                "Check geom_group_filter or ensure the model has visual geoms."
            )

        self._jit_render = _make_render_fn(
            geom_infos=geom_infos,
            cam_id=cam_id,
            height=height,
            width=width,
            hfov=hfov,
            vfov=vfov,
        )

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _render_pixels(self, pipeline_state) -> jnp.ndarray:
        """Render (H, W, 3) uint8 image from MJX pipeline state."""
        return self._jit_render(
            pipeline_state.geom_xpos,
            pipeline_state.geom_xmat,
            pipeline_state.cam_xpos,
            pipeline_state.cam_xmat,
        )

    def _build_obs(self, state_obs, pixels):
        if self._obs_mode == 'state':
            return state_obs
        if self._obs_mode == 'pixels':
            return {self._obs_key: pixels}
        # pixels+state
        obs: Dict[str, jnp.ndarray] = {self._obs_key: pixels}
        if isinstance(state_obs, Mapping):
            obs['state'] = state_obs.get('state', state_obs)
        else:
            obs['state'] = state_obs
        return obs

    def _init_frame_buffer(self, pixels):
        if self._frame_stack <= 1:
            return pixels
        return jnp.concatenate([pixels] * self._frame_stack, axis=-1)

    def _update_frame_buffer(self, new_pixels, prev_stacked):
        if self._frame_stack <= 1:
            return new_pixels
        return jnp.concatenate(
            [prev_stacked[..., self._channels:], new_pixels], axis=-1
        )

    # ------------------------------------------------------------------
    # Wrapper interface
    # ------------------------------------------------------------------

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        if self._obs_mode == 'state':
            return state

        pixels = self._render_pixels(state.pipeline_state)

        if self._frame_stack > 1:
            stacked = self._init_frame_buffer(pixels)
            state.info['_gpu_pixel_buffer'] = stacked
            pixels_out = stacked
        else:
            pixels_out = pixels

        return state.replace(obs=self._build_obs(state.obs, pixels_out))

    def step(self, state: State, action: jax.Array) -> State:
        state = self.env.step(state, action)
        if self._obs_mode == 'state':
            return state

        pixels = self._render_pixels(state.pipeline_state)

        if self._frame_stack > 1:
            prev = state.info.get('_gpu_pixel_buffer', self._init_frame_buffer(pixels))
            stacked = self._update_frame_buffer(pixels, prev)
            state.info['_gpu_pixel_buffer'] = stacked
            pixels_out = stacked
        else:
            pixels_out = pixels

        return state.replace(obs=self._build_obs(state.obs, pixels_out))

    # ------------------------------------------------------------------
    # Observation size
    # ------------------------------------------------------------------

    @property
    def observation_size(self):
        if self._obs_mode == 'state':
            return self.env.observation_size

        stacked_channels = self._channels * self._frame_stack
        obs_size: Dict[str, Tuple[int, ...]] = {
            self._obs_key: (self._height, self._width, stacked_channels),
        }
        if self._obs_mode == 'pixels+state':
            inner_size = self.env.observation_size
            if isinstance(inner_size, int):
                obs_size['state'] = (inner_size,)
            elif isinstance(inner_size, Mapping):
                obs_size['state'] = inner_size.get('state', inner_size)
            else:
                obs_size['state'] = (inner_size,)
        return obs_size


def _make_render_fn(
    geom_infos,
    cam_id: int,
    height: int,
    width: int,
    hfov: float,
    vfov: float,
):
    """Build a jax.jit-compiled single-frame render function.

    Signature:
        render_fn(geom_xpos, geom_xmat, cam_xpos, cam_xmat) -> (H, W, 3) uint8

    All inputs come directly from mjx.Data, which MJX's forward pass populates:
      - geom_xpos: (ngeom, 3)   world position of every geom centroid
      - geom_xmat: (ngeom, 3,3) world rotation matrix of every geom
                   columns = geom axes expressed in world frame
      - cam_xpos:  (ncam, 3)    world position of every camera
      - cam_xmat:  (ncam, 3,3)  world rotation matrix of every camera
                   col 1 = camera Y (up in image)
                   col 2 = camera Z (backward; camera looks in -Z)

    The pixelbrax renderer returns a canvas of shape (width, height, 3).
    We transpose to (height, width, 3) and flip the row axis to convert
    from OpenGL bottom-up convention to standard top-down image convention.

    Args:
        geom_infos: list of _GeomInfo built at wrapper init.
        cam_id: integer index of the camera in the model.
        height, width: image dimensions (static).
        hfov: horizontal field of view in degrees.
        vfov: vertical field of view in degrees.

    Returns:
        jax.jit-compiled callable.
    """
    from crax.renderer import CameraParameters, LightParameters, Renderer

    default_light = LightParameters(
        direction=jnp.array([0.57735, -0.57735, 0.57735], dtype=jnp.float32),
        colour=jnp.ones(3, dtype=jnp.float32),
        ambient=jnp.array([0.8, 0.8, 0.8], dtype=jnp.float32),
        diffuse=jnp.array([0.8, 0.8, 0.8], dtype=jnp.float32),
        specular=jnp.array([0.6, 0.6, 0.6], dtype=jnp.float32),
    )

    def render_fn(geom_xpos, geom_xmat, cam_xpos, cam_xmat):
        # --- Update geom world transforms from MJX pre-computed data ---
        instances = []
        for gi in geom_infos:
            idx = gi.geom_idx
            inst = gi.base_instance
            inst = inst.replace_with_position(geom_xpos[idx])
            inst = inst.replace_with_orientation(rotation_matrix=geom_xmat[idx])
            instances.append(inst)

        # --- Camera pose from mjx.Data.cam_xpos / cam_xmat ---
        # cam_xmat columns: col 1 = camera Y (up), col 2 = camera Z (backward)
        cam_pos = cam_xpos[cam_id]
        mat = cam_xmat[cam_id]            # (3, 3)
        cam_forward = -mat[:, 2]          # camera looks in -Z
        cam_up = mat[:, 1]                # camera Y is up in image

        camera = CameraParameters(
            viewWidth=width,
            viewHeight=height,
            position=cam_pos,
            target=cam_pos + cam_forward,
            up=cam_up,
            hfov=hfov,
            vfov=vfov,
        )

        # --- Render ---
        # pixelbrax canvas: (width, height, 3), OpenGL bottom-up convention.
        # Transpose → (height, width, 3), then flip rows for top-down images.
        img = Renderer.get_camera_image(
            objects=instances,
            light=default_light,
            camera=camera,
            width=width,
            height=height,
        )
        img = jnp.clip(jnp.flip(img.transpose(1, 0, 2), axis=0), 0.0, 1.0)
        return (img * 255).astype(jnp.uint8)

    return jax.jit(render_fn)
