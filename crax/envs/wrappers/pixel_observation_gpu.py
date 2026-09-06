"""GPU-based egocentric pixel observation wrapper using MJWarp's batch
ray-traced renderer.

Renders pixel observations fully on GPU. Geom/camera world poses are read
directly from mjx.Data.geom_xpos / geom_xmat / cam_xpos / cam_xmat, which
MJX's forward pass (smooth.kinematics, smooth.camlight) already computes.
No manual body-transform arithmetic. Those pose arrays are bridged into
MJWarp's warp-native render buffers with zero host copies, so rendering
stays on the GPU inside the training loop.

IMPORTANT — placement in the wrapper stack:
MJWarp's renderer needs a *statically* sized render context (`num_envs`
fixed at construction) and expects to be called directly with the full
batch of poses. It cannot itself be `jax.vmap`'d. Wrapping a single,
unbatched env with this class and then `jax.vmap`-ing the whole stack (the
way `crax.envs.training.wrap`'s `VmapWrapper` works) fails: the underlying
FFI call shape-checks against the unbatched per-example shape before vmap's
batching rule ever applies. So this wrapper must be applied to an ALREADY
VECTORIZED env, i.e. *after* `VmapWrapper` / `crax.envs.training.wrap`,
not before. `crax.training.agents.ppo.train._maybe_wrap_env` does this:
`wrap_for_training(env, ...)` runs first, then this wrapper is applied to
the result, using the same `num_envs` (or `num_eval_envs`) already known
at that point to size the render context.

Observation key: 'pixels/<camera_name>'  (e.g. 'pixels/vision')
"""

from typing import Dict, Mapping, Tuple

import jax
import jax.numpy as jnp
import mujoco
import mujoco_warp as mjw
import warp as wp
from warp.jax_experimental import ffi

from crax.envs.base import Env, State, Wrapper


@wp.kernel
def _write_cam_pose(
        cam_xpos_in: wp.array1d(dtype=wp.vec3f),
        cam_xmat_in: wp.array1d(dtype=wp.mat33f),
        cam_id: int,
        cam_xpos_out: wp.array2d(dtype=wp.vec3f),
        cam_xmat_out: wp.array2d(dtype=wp.mat33f),
):
    """Scatters the (num_envs,) pose of the one camera we render into the
    (num_envs, ncam) slot MJWarp's Data expects it in. Only the active
    camera's column is ever read at render time, so other columns are left
    stale/uninitialized without consequence.
    """
    w = wp.tid()
    cam_xpos_out[w, cam_id] = cam_xpos_in[w]
    cam_xmat_out[w, cam_id] = cam_xmat_in[w]


class GpuPixelObservationWrapper(Wrapper):
    """Wraps an already-vectorized CRAX/MJX env to add MJWarp-rendered pixel
    observations.

    Args:
        env: The (already-vectorized) environment to wrap.
        num_envs: Number of parallel worlds `env` produces per step/reset.
        camera: Name of the MuJoCo camera to render from (must exist in XML).
        height: Render height in pixels.
        width: Render width in pixels.
        obs_mode: 'pixels', 'pixels+state', or 'state'.
        frame_stack: Number of frames to stack channel-wise.
        use_shadows: Whether MJWarp should ray-trace shadows (slower).
    """

    def __init__(
            self,
            env: Env,
            num_envs: int,
            camera: str = 'vision',
            height: int = 64,
            width: int = 64,
            obs_mode: str = 'pixels',
            frame_stack: int = 1,
            use_shadows: bool = False,
    ):
        super().__init__(env)

        if obs_mode not in ('pixels', 'pixels+state', 'state'):
            raise ValueError(
                f"obs_mode must be 'pixels', 'pixels+state', or 'state', got '{obs_mode}'"
            )

        self._obs_key = f'pixels/{camera}'
        self._num_envs = num_envs
        self._height = height
        self._width = width
        self._obs_mode = obs_mode
        self._frame_stack = max(1, frame_stack)
        self._channels = 3

        mj_model = self.sys.mj_model
        if getattr(self, 'backend', None) != 'mjx':
            raise ValueError(
                "GpuPixelObservationWrapper (MJWarp) requires backend='mjx' "
                f"(geom_xpos/cam_xpos are only populated by the MJX pipeline), "
                f"got backend='{getattr(self, 'backend', None)}'. Pass "
                "backend='mjx' to get_environment()/create()."
            )

        cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if cam_id == -1:
            available = [mj_model.camera(i).name for i in range(mj_model.ncam)]
            raise ValueError(
                f"Camera '{camera}' not found in model. Available: {available}"
            )
        self._cam_id = cam_id

        # Scratch MjData purely to seed MJWarp's model/render-context
        # construction (mesh/texture/light setup) — its dynamic fields are
        # never read; poses come from mjx.Data every step via the bridge below.
        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_forward(mj_model, mj_data)

        self._m = mjw.put_model(mj_model)
        self._d = mjw.put_data(mj_model, mj_data, nworld=num_envs)
        self._rc = mjw.create_render_context(
            mj_model,
            nworld=num_envs,
            cam_res=(width, height),
            render_rgb=True,
            use_shadows=use_shadows,
            cam_active=[i == cam_id for i in range(mj_model.ncam)],
        )
        # cam_active has exactly one True entry (cam_id) -> its render-context
        # -local index (rc.cam_id_map) is always 0.
        self._local_cam_index = 0

        # Warm up eagerly (outside any CUDA graph capture) so the render
        # megakernel is already JIT-compiled/loaded on-device before the
        # jax_callable below captures its CUDA graph on first real call.
        # Loading a new CUDA module during graph capture is illegal and
        # would otherwise crash the very first render.
        mjw.refit_bvh(self._m, self._d, self._rc)
        mjw.render(self._m, self._d, self._rc)
        wp.synchronize()

        self._render_pixels_fn = self._build_render_fn()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _build_render_fn(self):
        m, d, rc = self._m, self._d, self._rc
        cam_id = self._cam_id
        local_cam_index = self._local_cam_index
        num_envs, height, width = self._num_envs, self._height, self._width

        def warp_render(
                geom_xpos_in: wp.array2d(dtype=wp.vec3f),
                geom_xmat_in: wp.array2d(dtype=wp.mat33f),
                cam_xpos_in: wp.array1d(dtype=wp.vec3f),
                cam_xmat_in: wp.array1d(dtype=wp.mat33f),
                rgb_out: wp.array3d(dtype=wp.vec3f),
        ):
            wp.copy(d.geom_xpos, geom_xpos_in)
            wp.copy(d.geom_xmat, geom_xmat_in)
            wp.launch(
                _write_cam_pose, dim=num_envs,
                inputs=[cam_xpos_in, cam_xmat_in, cam_id],
                outputs=[d.cam_xpos, d.cam_xmat],
            )
            mjw.refit_bvh(m, d, rc)
            mjw.render(m, d, rc)
            rgb_local = wp.zeros((num_envs, height, width), dtype=wp.vec3f)
            mjw.get_rgb(rc, camera_index=local_cam_index, rgb_out=rgb_local)
            wp.copy(rgb_out, rgb_local)

        render_callable = ffi.jax_callable(
            warp_render,
            num_outputs=1,
            output_dims={'rgb_out': (num_envs, height, width)},
            # GraphMode.JAX (the default) lets XLA try to capture our warp
            # kernel launches as a child node inside its own CUDA graph.
            # Unsupported on at least some driver/arch combos.
            # GraphMode.WARP has warp capture+replay its own self-contained
            # graph instead of participating in XLA's, avoiding the nesting
            # entirely. This is also what mujoco_warp's own jax_callable
            # usage does.
            graph_mode=ffi.GraphMode.WARP,
        )

        def render_pixels(pipeline_state) -> jnp.ndarray:
            """(num_envs, H, W, 3) uint8 from a BATCHED mjx pipeline state."""
            cam_xpos = pipeline_state.cam_xpos[:, cam_id]
            cam_xmat = pipeline_state.cam_xmat[:, cam_id]
            (rgb,) = render_callable(
                pipeline_state.geom_xpos, pipeline_state.geom_xmat,
                cam_xpos, cam_xmat,
            )
            return (jnp.clip(rgb, 0.0, 1.0) * 255).astype(jnp.uint8)

        return render_pixels

    def _render_pixels(self, pipeline_state) -> jnp.ndarray:
        return self._render_pixels_fn(pipeline_state)

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

        orig_obs = state.obs
        pixels = self._render_pixels(state.pipeline_state)

        if self._frame_stack > 1:
            stacked = self._init_frame_buffer(pixels)
            state.info['_gpu_pixel_buffer'] = stacked
            pixels_out = stacked
        else:
            pixels_out = pixels

        # Stash the inner env's native obs so step() can hand it back down
        # unchanged on the next call. The inner Episode/AutoReset/Vmap chain
        # (below) is never aware we replace `obs` with a pixel dict. Feeding
        # it our dict back in would break its internal action_repeat scan
        # (carry pytree must stay the inner env's own plain-obs shape across
        # iterations).
        state.info['_orig_state_obs'] = orig_obs
        return state.replace(obs=self._build_obs(orig_obs, pixels_out))

    def step(self, state: State, action: jax.Array) -> State:
        inner_state = state.replace(obs=state.info['_orig_state_obs'])
        state = self.env.step(inner_state, action)
        if self._obs_mode == 'state':
            return state

        orig_obs = state.obs
        pixels = self._render_pixels(state.pipeline_state)

        if self._frame_stack > 1:
            prev = state.info.get('_gpu_pixel_buffer', self._init_frame_buffer(pixels))
            stacked = self._update_frame_buffer(pixels, prev)
            state.info['_gpu_pixel_buffer'] = stacked
            pixels_out = stacked
        else:
            pixels_out = pixels

        state.info['_orig_state_obs'] = orig_obs

        return state.replace(obs=self._build_obs(orig_obs, pixels_out))

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
