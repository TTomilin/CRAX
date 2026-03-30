"""GPU-based egocentric pixel observation wrapper using pixelbrax renderer.

Renders pixel observations fully on GPU using JAX rasterization, with no CPU
callbacks. Uses pixelbrax's renderer to produce egocentric camera images
directly from MJX physics state (xpos/xquat body transforms).

The rendering pipeline:
  1. At init: extract geometry from mj_model, build pixelbrax Model objects.
  2. At step/reset: in a JIT-compiled function, read xpos/xquat from mjx.Data,
     compute geom world transforms, update ModelObject transforms, call renderer.
  3. Return (H, W, 3) uint8 pixel arrays under key 'pixels/ego'.
"""

import sys
import os
from collections import namedtuple
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from brax import math as brax_math
from brax.envs.base import Env, State, Wrapper

# MuJoCo geom type constants
_MJ_GEOM_PLANE = 0
_MJ_GEOM_SPHERE = 2
_MJ_GEOM_CAPSULE = 3
_MJ_GEOM_CYLINDER = 5
_MJ_GEOM_BOX = 6
_MJ_GEOM_MESH = 7

# Named tuple holding static per-geom data used to rebuild instances each step
_GeomInfo = namedtuple('_GeomInfo', ['body_id', 'local_pos', 'local_quat', 'base_instance'])


def _build_renderer_objects(mj_model, geom_group_filter):
    """Build a list of _GeomInfo from a MuJoCo model.

    Converts MuJoCo geometry (capsules, boxes, spheres, meshes, planes) into
    pixelbrax ModelObject instances with identity transforms.  The body_id and
    local pos/quat are stored separately so they can be updated each step from
    mjx.Data.xpos / mjx.Data.xquat without Python overhead.

    Args:
        mj_model: mujoco.MjModel instance.
        geom_group_filter: if given, only include geoms whose group is in this
            set.  Pass None to include all groups.

    Returns:
        List of _GeomInfo named tuples.
    """
    # pixelbrax imports are deferred to avoid hard dependency at module load
    _repo_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    from brax.renderer import (
        create_capsule,
        create_cube,
        UpAxis,
    )
    from brax.renderer import Model as RendererMesh, ModelObject as Instance

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
                # Render as large thin box at the plane's position
                floor_tex = jnp.array(
                    np.array([0.78, 0.78, 0.78], dtype=np.float32).reshape(1, 1, 3)
                )
                model = create_cube(
                    half_extents=jnp.array([50.0, 50.0, 0.001], dtype=jnp.float32),
                    diffuse_map=floor_tex,
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

        body_id = int(mj_model.geom_bodyid[i])
        local_pos = jnp.array(mj_model.geom_pos[i], dtype=jnp.float32)
        local_quat = jnp.array(mj_model.geom_quat[i], dtype=jnp.float32)  # (w,x,y,z)

        geom_infos.append(_GeomInfo(
            body_id=body_id,
            local_pos=local_pos,
            local_quat=local_quat,
            base_instance=Instance(model=model),
        ))

    return geom_infos


class GpuPixelObservationWrapper(Wrapper):
    """Wraps a CRAX/MJX environment to add GPU-rendered egocentric pixel obs.

    Rendering is done entirely in JAX on the GPU using pixelbrax's rasterizer.
    No CPU callbacks are used; the renderer is called inside JIT-compiled code.

    Camera modes:
      egocentric_rotate=False (default): camera offset is in world frame, camera
          always stays in the same orientation relative to the world.
      egocentric_rotate=True: camera offset is rotated by the agent's body
          quaternion, so the camera rotates with the agent.  This gives a true
          egocentric / first-person-like view.

    Observation key: 'pixels/ego' (H, W, 3) uint8.

    Args:
        env: The environment to wrap.
        height: Render height in pixels.
        width: Render width in pixels.
        obs_mode: 'pixels', 'pixels+state', or 'state'.
        frame_stack: Number of frames to stack channel-wise.
        camera_body_index: Index of the MuJoCo body the camera is attached to.
            Body 0 is always the world body; the agent body is typically 1.
        camera_offset: Camera position offset from the body.  In world frame
            when egocentric_rotate=False; in body frame when True.
        camera_target_offset: Where the camera looks, as an offset from the
            camera body's position.  Same frame convention as camera_offset.
        camera_up: World-space up vector for the camera.
        hfov: Horizontal field of view in degrees.
        egocentric_rotate: If True, rotate camera offset by agent quaternion.
        geom_group_filter: If given, only render geoms in these MuJoCo groups.
    """

    _CAMERA_KEY = 'pixels/ego'

    def __init__(
        self,
        env: Env,
        height: int = 64,
        width: int = 64,
        obs_mode: str = 'pixels+state',
        frame_stack: int = 1,
        camera_body_index: int = 1,
        camera_offset: Tuple[float, float, float] = (0.0, -2.5, 1.0),
        camera_target_offset: Tuple[float, float, float] = (0.0, 0.0, 0.5),
        camera_up: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        hfov: float = 60.0,
        egocentric_rotate: bool = True,
        geom_group_filter: Optional[Sequence[int]] = None,
    ):
        super().__init__(env)

        if obs_mode not in ('pixels', 'pixels+state', 'state'):
            raise ValueError(
                f"obs_mode must be 'pixels', 'pixels+state', or 'state', got '{obs_mode}'"
            )

        self._height = height
        self._width = width
        self._obs_mode = obs_mode
        self._frame_stack = max(1, frame_stack)
        self._channels = 3  # always RGB

        mj_model = self.sys.mj_model
        geom_infos = _build_renderer_objects(mj_model, geom_group_filter)
        if not geom_infos:
            raise RuntimeError(
                "GPU renderer: no renderable geometry found in the model. "
                "Check geom_group_filter or ensure the model has visual geoms."
            )

        # Build the JIT-compiled single-frame render function.
        self._jit_render = _make_render_fn(
            geom_infos=geom_infos,
            height=height,
            width=width,
            camera_body_index=camera_body_index,
            camera_offset=jnp.array(camera_offset, dtype=jnp.float32),
            camera_target_offset=jnp.array(camera_target_offset, dtype=jnp.float32),
            camera_up=jnp.array(camera_up, dtype=jnp.float32),
            hfov=float(hfov),
            egocentric_rotate=egocentric_rotate,
        )

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _render_pixels(self, pipeline_state) -> jnp.ndarray:
        """Render (H, W, 3) uint8 image from the MJX pipeline state."""
        return self._jit_render(pipeline_state.xpos, pipeline_state.xquat)

    def _build_obs(
        self,
        state_obs: Union[jnp.ndarray, Mapping],
        pixels: jnp.ndarray,
    ) -> Union[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        """Combine state obs and pixel image into the observation dict."""
        if self._obs_mode == 'state':
            return state_obs

        if self._obs_mode == 'pixels':
            return {self._CAMERA_KEY: pixels}

        # pixels+state
        obs: Dict[str, jnp.ndarray] = {self._CAMERA_KEY: pixels}
        if isinstance(state_obs, Mapping):
            obs['state'] = state_obs.get('state', state_obs)
        else:
            obs['state'] = state_obs
        return obs

    def _init_frame_buffer(
        self, pixels: jnp.ndarray
    ) -> jnp.ndarray:
        """Initialize frame buffer by tiling the first frame."""
        if self._frame_stack <= 1:
            return pixels
        return jnp.concatenate([pixels] * self._frame_stack, axis=-1)

    def _update_frame_buffer(
        self,
        new_pixels: jnp.ndarray,
        prev_stacked: jnp.ndarray,
    ) -> jnp.ndarray:
        """Shift frame buffer left and append the new frame."""
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

        obs = self._build_obs(state.obs, pixels_out)
        state.info['_render_step'] = jnp.array(0, dtype=jnp.int32)
        return state.replace(obs=obs)

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

        obs = self._build_obs(state.obs, pixels_out)
        state.info['_render_step'] = state.info.get('_render_step', jnp.array(0)) + 1
        return state.replace(obs=obs)

    # ------------------------------------------------------------------
    # Observation size property
    # ------------------------------------------------------------------

    @property
    def observation_size(self) -> Union[int, Mapping[str, Tuple[int, ...]]]:
        if self._obs_mode == 'state':
            return self.env.observation_size

        stacked_channels = self._channels * self._frame_stack
        obs_size: Dict[str, Tuple[int, ...]] = {
            self._CAMERA_KEY: (self._height, self._width, stacked_channels),
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
    height: int,
    width: int,
    camera_body_index: int,
    camera_offset: jnp.ndarray,
    camera_target_offset: jnp.ndarray,
    camera_up: jnp.ndarray,
    hfov: float,
    egocentric_rotate: bool,
) -> callable:
    """Build a JIT-compiled function that renders one frame from body transforms.

    The returned function has signature:
        render_fn(xpos: (nbody, 3), xquat: (nbody, 4)) -> (H, W, 3) uint8

    It is safe to call from inside jax.vmap (e.g. via VmapWrapper).

    Args:
        geom_infos: list of _GeomInfo built at wrapper init.
        height, width: image dimensions (static).
        camera_body_index: body to attach camera to.
        camera_offset: offset from body to camera position.
        camera_target_offset: offset from body to camera look-at point.
        camera_up: world-space up vector.
        hfov: horizontal field of view in degrees.
        egocentric_rotate: whether to rotate the offset by agent quaternion.

    Returns:
        A jax.jit-compiled callable.
    """
    from brax.renderer import (
        CameraParameters,
        LightParameters,
        Renderer,
    )

    vfov = hfov * height / width

    default_light = LightParameters(
        direction=jnp.array([0.57735, -0.57735, 0.57735], dtype=jnp.float32),
        colour=jnp.ones(3, dtype=jnp.float32),
        ambient=jnp.array([0.8, 0.8, 0.8], dtype=jnp.float32),
        diffuse=jnp.array([0.8, 0.8, 0.8], dtype=jnp.float32),
        specular=jnp.array([0.6, 0.6, 0.6], dtype=jnp.float32),
    )

    def render_fn(xpos, xquat):
        # --- Update ModelObject transforms from physics state ---
        instances = []
        for gi in geom_infos:
            bid = gi.body_id
            b_pos = xpos[bid]     # (3,)
            b_quat = xquat[bid]   # (4,) w,x,y,z

            w_pos = b_pos + brax_math.rotate(gi.local_pos, b_quat)
            w_quat = brax_math.quat_mul(b_quat, gi.local_quat)

            inst = gi.base_instance
            inst = inst.replace_with_position(w_pos)
            inst = inst.replace_with_orientation(w_quat)
            instances.append(inst)

        # --- Egocentric camera ---
        agent_pos = xpos[camera_body_index]   # (3,)
        agent_quat = xquat[camera_body_index]  # (4,)

        if egocentric_rotate:
            cam_pos = agent_pos + brax_math.rotate(camera_offset, agent_quat)
            cam_target = agent_pos + brax_math.rotate(camera_target_offset, agent_quat)
        else:
            cam_pos = agent_pos + camera_offset
            cam_target = agent_pos + camera_target_offset

        camera = CameraParameters(
            viewWidth=width,
            viewHeight=height,
            position=cam_pos,
            target=cam_target,
            up=camera_up,
            hfov=hfov,
            vfov=vfov,
        )

        # --- Render ---
        img = Renderer.get_camera_image(
            objects=instances,
            light=default_light,
            camera=camera,
            width=width,
            height=height,
        )
        img = jnp.clip(img, 0.0, 1.0)
        return (img * 255).astype(jnp.uint8)

    return jax.jit(render_fn)
