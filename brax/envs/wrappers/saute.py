# Copyright 2025 Safe-Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sauté RL environment wrapper.

This wrapper augments the observation with a running discounted safety budget
and early-terminates an episode if the (discounted) budget is exceeded.
It is JAX-friendly and works with vectorized envs.

References:
- Sootla et al. (2022) Sauté RL: Almost Surely Safe Reinforcement Learning Using State Augmentation
"""

import jax.numpy as jnp

from brax.envs.base import Env, Wrapper, State


class SauteWrapper(Wrapper):
    def __init__(
            self,
            env: Env,
            initial_budget: float = 1.0,
            gamma_budget: float = 0.99,
            termination_on_violation: bool = True,
            violation_penalty: float = 0.0,
            normalize_budget_obs: bool = True,
    ):
        super().__init__(env)
        self._b0 = float(initial_budget)
        self._gamma = float(gamma_budget)
        self._terminate = bool(termination_on_violation)
        self._viol_pen = float(violation_penalty)
        self._normalize = bool(normalize_budget_obs)

    def _augment_obs(self, obs: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        b_obs = b / self._b0 if self._normalize and self._b0 != 0.0 else b
        return jnp.concatenate([obs, jnp.expand_dims(b_obs, -1)], axis=-1)

    def reset(self, rng: jnp.ndarray) -> State:
        state = self.env.reset(rng)
        info = state.info.copy()

        # Ensure that the cost is present
        if 'cost' not in info:
            info['cost'] = jnp.zeros_like(state.reward)

        # Initialize budget and violation flag
        b = jnp.ones_like(state.reward) * self._b0
        info['saute_budget'] = b
        info['saute_violated'] = jnp.zeros_like(b)

        # Augment observation
        obs = self._augment_obs(state.obs, b)

        # Initialize metrics
        metrics = state.metrics.copy()
        metrics['saute_budget'] = b
        metrics['saute_violated'] = jnp.zeros_like(b)

        return state.replace(obs=obs, info=info, metrics=metrics)

    def step(self, state: State, action: jnp.ndarray) -> State:
        # Step underlying env
        next_state = self.env.step(state, action)
        info = next_state.info.copy()

        # Cost signal (default 0 if missing)
        cost = info.get('cost', jnp.zeros_like(next_state.reward))

        # Did the *previous* state mark an episode end? (EpisodeWrapper's done)
        done_prev = state.done.astype(jnp.bool_)

        # Previous budget (fallback to full budget if missing)
        b_prev_raw = state.info.get(
            'saute_budget',
            jnp.ones_like(next_state.reward) * self._b0,
            )

        # If previous step was done, start a fresh budget at b0 for the new episode
        b_prev = jnp.where(
            done_prev,
            jnp.ones_like(b_prev_raw) * self._b0,
            b_prev_raw,
            )

        # Sauté update: b̃_{t+1} = (b_t - c_t) / gamma
        b_candidate = (b_prev - cost) / self._gamma

        # Budget violation check on the candidate value
        violated = b_candidate < 0.0

        # Clamp budget to [0, +∞) to avoid huge negative values after violation
        b_next = jnp.where(
            violated,
            jnp.zeros_like(b_candidate),
            b_candidate,
        )

        # Termination on violation, but keep dtype consistent with env
        base_done = next_state.done
        base_dtype = base_done.dtype
        if self._terminate:
            done_bool = jnp.logical_or(
                base_done.astype(jnp.bool_),
                violated,
            )
            done = done_bool.astype(base_dtype)
        else:
            done = base_done

        # Optional violation penalty
        reward = next_state.reward
        if self._viol_pen != 0.0:
            reward = reward + jnp.where(violated, self._viol_pen, 0.0)

        # Update info
        info['saute_budget'] = b_next
        info['saute_violated'] = violated.astype(jnp.float32)

        # Augment observation with (normalized) budget
        obs = self._augment_obs(next_state.obs, b_next)

        # Update metrics
        metrics = next_state.metrics.copy()
        metrics['saute_budget'] = b_next
        metrics['saute_violated'] = violated.astype(jnp.float32)

        return next_state.replace(
            obs=obs,
            reward=reward,
            done=done,
            info=info,
            metrics=metrics,
        )
