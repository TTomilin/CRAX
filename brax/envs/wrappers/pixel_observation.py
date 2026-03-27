"""Pixel observation wrapper for CRAX environments.

Renders pixel observations from MuJoCo cameras during environment stepping.
Uses jax.pure_callback to bridge GPU-based MJX physics with CPU-based
MuJoCo rendering.
"""

from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from brax.envs.base import Env, State, Wrapper


class PixelObservationWrapper(Wrapper):
    """Wraps a CRAX environment to add pixel observations from MuJoCo cameras.

    This wrapper intercepts reset() and step() to render pixel observations
    using MuJoCo's CPU renderer via jax.pure_callback. It converts flat
    state-vector observations into dict observations containing pixel arrays
    and optionally the original state vector.

    The wrapper should be applied AFTER UnifiedEnvAdapter and BEFORE the
    training wrappers (VmapWrapper, EpisodeWrapper, AutoResetWrapper).

    Args:
        env: The environment to wrap.
        cameras: Camera names from the MuJoCo XML to render from.
        height: Render height in pixels.
        width: Render width in pixels.
        obs_mode: Observation mode — 'pixels', 'pixels+state', or 'state'.
        frame_stack: Number of consecutive frames to stack (channel-wise).
        render_every_n: Render every N steps, reusing the last frame otherwise.
        grayscale: If True, convert rendered images to grayscale.
        num_render_workers: Number of threads for parallel CPU rendering.
    """

    def __init__(
        self,
        env: Env,
        cameras: Sequence[str] = ('vision',),
        height: int = 84,
        width: int = 84,
        obs_mode: str = 'pixels+state',
        frame_stack: int = 1,
        render_every_n: int = 1,
        grayscale: bool = False,
        num_render_workers: int = 4,
    ):
        super().__init__(env)
        if obs_mode not in ('pixels', 'pixels+state', 'state'):
            raise ValueError(
                f"obs_mode must be 'pixels', 'pixels+state', or 'state', "
                f"got '{obs_mode}'"
            )

        self._cameras = tuple(cameras)
        self._height = height
        self._width = width
        self._obs_mode = obs_mode
        self._frame_stack = max(1, frame_stack)
        self._render_every_n = max(1, render_every_n)
        self._grayscale = grayscale
        self._num_render_workers = num_render_workers
        self._channels = 1 if grayscale else 3

        # Get the MuJoCo model from the inner environment
        self._mj_model = self.sys.mj_model

        # Resolve camera IDs
        self._camera_ids = {}
        for cam_name in self._cameras:
            cam_id = mujoco.mj_name2id(
                self._mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name
            )
            if cam_id == -1:
                available = [
                    self._mj_model.camera(i).name
                    for i in range(self._mj_model.ncam)
                ]
                raise ValueError(
                    f"Camera '{cam_name}' not found in model. "
                    f"Available cameras: {available}"
                )
            self._camera_ids[cam_name] = cam_id

        # Single persistent Renderer + MjData, matching Brax's image.py
        # pattern. EGL contexts are thread-bound and cannot be shared across
        # threads, so we render sequentially in the callback thread.
        self._renderer = mujoco.Renderer(
            self._mj_model, height=self._height, width=self._width
        )
        self._mj_data = mujoco.MjData(self._mj_model)

        # Determine state observation size from inner env
        inner_obs_size = self.env.observation_size
        if isinstance(inner_obs_size, int):
            self._state_obs_size = inner_obs_size
        elif isinstance(inner_obs_size, Mapping):
            self._state_obs_size = inner_obs_size.get('state', inner_obs_size)
        else:
            self._state_obs_size = inner_obs_size

    def _render_single(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        mocap_pos: Optional[np.ndarray] = None,
        mocap_quat: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Render a single environment state on CPU."""
        d = self._mj_data
        d.qpos[:] = q
        d.qvel[:] = qd
        if mocap_pos is not None and self._mj_model.nmocap > 0:
            d.mocap_pos[:] = mocap_pos
            d.mocap_quat[:] = mocap_quat
        mujoco.mj_forward(self._mj_model, d)

        pixels = {}
        for cam_name, cam_id in self._camera_ids.items():
            self._renderer.update_scene(d, camera=cam_id)
            img = self._renderer.render().copy()  # (H, W, 3) uint8
            if self._grayscale:
                # ITU-R BT.601 luma
                img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
                img = img[..., np.newaxis].astype(np.uint8)
            pixels[cam_name] = img
        return pixels

    def _render_batch(
        self,
        q_batch: np.ndarray,
        qd_batch: np.ndarray,
        mocap_pos_batch: Optional[np.ndarray] = None,
        mocap_quat_batch: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Render a batch of environments sequentially."""
        batch_size = q_batch.shape[0]
        has_mocap = mocap_pos_batch is not None

        results = []
        for i in range(batch_size):
            mp = mocap_pos_batch[i] if has_mocap else None
            mq = mocap_quat_batch[i] if has_mocap else None
            results.append(self._render_single(q_batch[i], qd_batch[i], mp, mq))

        # Stack into batched arrays: {cam_name: (batch, H, W, C)}
        batched = {}
        for cam_name in self._cameras:
            batched[cam_name] = np.stack(
                [r[cam_name] for r in results], axis=0
            )
        return batched

    def _render_callback(self, *flat_args):
        """Callback for jax.pure_callback — renders on CPU.

        Receives flattened numpy arrays, returns pixel dict values as a flat
        tuple matching the result_shape_dtypes structure.

        When called under vmap with vmap_method='broadcast_all', args may
        arrive with multiple leading batch dimensions (e.g. device, env).
        We flatten them, render as a single batch, then reshape back.
        """
        import sys as _sys
        if not hasattr(self, '_render_call_count'):
            self._render_call_count = 0
        self._render_call_count += 1
        if self._render_call_count <= 3 or self._render_call_count % 100 == 0:
            print(f"[DEBUG pixel_obs] _render_callback call #{self._render_call_count}, args shapes: {[a.shape for a in flat_args]}")
            _sys.stdout.flush()

        q, qd = flat_args[0], flat_args[1]
        mocap_pos = flat_args[2] if len(flat_args) > 2 else None
        mocap_quat = flat_args[3] if len(flat_args) > 3 else None

        q_np = np.asarray(q)
        qd_np = np.asarray(qd)
        mp_np = np.asarray(mocap_pos) if mocap_pos is not None else None
        mq_np = np.asarray(mocap_quat) if mocap_quat is not None else None

        # qpos has shape (*batch_dims, nq). Flatten all leading batch dims.
        nq = self._mj_model.nq
        nv = self._mj_model.nv
        batch_shape = q_np.shape[:-1]  # e.g. (1, 2048)
        is_batched = len(batch_shape) > 0 and np.prod(batch_shape) > 1

        if is_batched:
            flat_size = int(np.prod(batch_shape))
            q_flat = q_np.reshape(flat_size, nq)
            qd_flat = qd_np.reshape(flat_size, nv)
            mp_flat = mp_np.reshape(flat_size, *mp_np.shape[len(batch_shape):]) if mp_np is not None else None
            mq_flat = mq_np.reshape(flat_size, *mq_np.shape[len(batch_shape):]) if mq_np is not None else None

            pixels = self._render_batch(q_flat, qd_flat, mp_flat, mq_flat)

            # Reshape back: (flat_size, H, W, C) -> (*batch_shape, H, W, C)
            pixels = {
                k: v.reshape(*batch_shape, *v.shape[1:])
                for k, v in pixels.items()
            }
        else:
            if len(batch_shape) == 0:
                pixels = self._render_single(q_np, qd_np, mp_np, mq_np)
            else:
                # Single element batch — squeeze, render, unsqueeze
                pixels = self._render_single(
                    q_np.reshape(nq), qd_np.reshape(nv),
                    mp_np.reshape(*mp_np.shape[len(batch_shape):]) if mp_np is not None else None,
                    mq_np.reshape(*mq_np.shape[len(batch_shape):]) if mq_np is not None else None,
                )
                pixels = {k: v.reshape(*batch_shape, *v.shape) for k, v in pixels.items()}

        # Return as tuple in camera order (must match result_shape_dtypes)
        return tuple(pixels[cam] for cam in self._cameras)

    def _get_pixel_result_shapes(self, is_batched: bool, batch_size: int = 0):
        """Get the result_shape_dtypes for jax.pure_callback."""
        pixel_shape = (self._height, self._width, self._channels)
        if is_batched:
            pixel_shape = (batch_size,) + pixel_shape
        return tuple(
            jax.ShapeDtypeStruct(pixel_shape, jnp.uint8)
            for _ in self._cameras
        )

    def _render_pixels(self, pipeline_state) -> Dict[str, jnp.ndarray]:
        """Render pixels from the current pipeline state using pure_callback."""
        q = pipeline_state.q
        qd = pipeline_state.qd

        # Determine if batched
        is_batched = q.ndim > 1
        batch_size = q.shape[0] if is_batched else 0

        import sys as _sys
        if not hasattr(self, '_render_pixels_count'):
            self._render_pixels_count = 0
        self._render_pixels_count += 1
        if self._render_pixels_count <= 3:
            print(f"[DEBUG pixel_obs] _render_pixels call #{self._render_pixels_count}: q.shape={q.shape}, is_batched={is_batched}, batch_size={batch_size}")
            _sys.stdout.flush()

        # Build callback args
        callback_args = [q, qd]
        if hasattr(pipeline_state, 'mocap_pos') and self._mj_model.nmocap > 0:
            callback_args.append(pipeline_state.mocap_pos)
            callback_args.append(pipeline_state.mocap_quat)

        result_shapes = self._get_pixel_result_shapes(is_batched, batch_size)

        if self._render_pixels_count <= 3:
            print(f"[DEBUG pixel_obs] result_shapes: {result_shapes}")
            _sys.stdout.flush()

        pixel_arrays = jax.pure_callback(
            self._render_callback,
            result_shapes,
            *callback_args,
            vmap_method='broadcast_all',
        )

        # Build pixel observation dict — keep as uint8; networks normalize on-the-fly
        pixels = {}
        for i, cam_name in enumerate(self._cameras):
            pixels[f'pixels/{cam_name}'] = pixel_arrays[i]

        return pixels

    def _build_obs(
        self, state_obs: Union[jnp.ndarray, Mapping], pixels: Dict[str, jnp.ndarray]
    ) -> Union[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        """Build the observation dict from state obs and rendered pixels."""
        if self._obs_mode == 'state':
            return state_obs

        if self._obs_mode == 'pixels':
            return pixels

        # pixels+state mode
        obs = dict(pixels)
        if isinstance(state_obs, Mapping):
            obs['state'] = state_obs.get('state', state_obs)
        else:
            obs['state'] = state_obs
        return obs

    def _init_frame_buffer(
        self, pixels: Dict[str, jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        """Initialize frame buffer by repeating the first frame."""
        if self._frame_stack <= 1:
            return pixels

        stacked = {}
        for key, img in pixels.items():
            if key.startswith('pixels/'):
                # Stack the same frame frame_stack times along channel dim
                stacked[key] = jnp.concatenate(
                    [img] * self._frame_stack, axis=-1
                )
            else:
                stacked[key] = img
        return stacked

    def _update_frame_buffer(
        self,
        new_pixels: Dict[str, jnp.ndarray],
        prev_buffer: Dict[str, jnp.ndarray],
    ) -> Dict[str, jnp.ndarray]:
        """Shift the frame buffer and append new frames."""
        if self._frame_stack <= 1:
            return new_pixels

        updated = {}
        for key in prev_buffer:
            if key.startswith('pixels/'):
                # Shift left by channels, append new frame
                old = prev_buffer[key]
                new = new_pixels[key]
                updated[key] = jnp.concatenate(
                    [old[..., self._channels:], new], axis=-1
                )
            else:
                updated[key] = new_pixels.get(key, prev_buffer[key])
        return updated

    def reset(self, rng: jax.Array) -> State:
        import sys as _sys
        if not hasattr(self, '_reset_count'):
            self._reset_count = 0
        self._reset_count += 1
        if self._reset_count <= 3:
            print(f"[DEBUG pixel_obs] reset() call #{self._reset_count}")
            _sys.stdout.flush()

        state = self.env.reset(rng)

        if self._obs_mode == 'state':
            return state

        if self._reset_count <= 3:
            print(f"[DEBUG pixel_obs] inner reset done, rendering pixels...")
            _sys.stdout.flush()
        pixels = self._render_pixels(state.pipeline_state)
        obs = self._build_obs(state.obs, pixels)

        if self._frame_stack > 1:
            obs = self._init_frame_buffer(obs)
            state.info['_pixel_buffer'] = {
                k: v for k, v in obs.items() if k.startswith('pixels/')
            }

        state.info['_render_step'] = jnp.array(0)

        return state.replace(obs=obs)

    def step(self, state: State, action: jax.Array) -> State:
        state = self.env.step(state, action)

        if self._obs_mode == 'state':
            return state

        render_step = state.info.get('_render_step', jnp.array(0))
        should_render = (render_step % self._render_every_n) == 0

        new_pixels = self._render_pixels(state.pipeline_state)

        if self._render_every_n > 1:
            # Use previous pixels if not rendering this step
            prev_buffer = state.info.get('_pixel_buffer', new_pixels)
            new_pixels = jax.tree.map(
                lambda new, old: jnp.where(should_render, new, old),
                new_pixels,
                {k: v for k, v in prev_buffer.items() if k.startswith('pixels/')},
            )

        if self._frame_stack > 1:
            prev_buffer = state.info.get('_pixel_buffer', new_pixels)
            stacked_pixels = self._update_frame_buffer(new_pixels, prev_buffer)
            obs = self._build_obs(state.obs, stacked_pixels)
            state.info['_pixel_buffer'] = {
                k: v for k, v in obs.items() if k.startswith('pixels/')
            }
        else:
            obs = self._build_obs(state.obs, new_pixels)

        state.info['_render_step'] = render_step + 1

        return state.replace(obs=obs)

    @property
    def observation_size(self) -> Union[int, Mapping[str, Tuple[int, ...]]]:
        if self._obs_mode == 'state':
            return self.env.observation_size

        obs_size = {}
        stacked_channels = self._channels * self._frame_stack
        for cam_name in self._cameras:
            obs_size[f'pixels/{cam_name}'] = (
                self._height,
                self._width,
                stacked_channels,
            )

        if self._obs_mode == 'pixels+state':
            inner_size = self.env.observation_size
            if isinstance(inner_size, int):
                obs_size['state'] = (inner_size,)
            elif isinstance(inner_size, Mapping):
                if 'state' in inner_size:
                    obs_size['state'] = inner_size['state']
                else:
                    # Flat obs — wrap as tuple
                    obs_size['state'] = inner_size
            else:
                obs_size['state'] = (inner_size,)

        return obs_size
