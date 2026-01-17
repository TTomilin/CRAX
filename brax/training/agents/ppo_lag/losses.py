# Copyright 2024 The Brax Authors.
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

"""PPO-Lagrange loss functions.

Extends PPO with Lagrangian-based constrained optimization for safe RL.
See: https://arxiv.org/pdf/1707.06347.pdf
"""

from typing import Any, Tuple

import flax
import jax
import jax.numpy as jnp

from brax.training import types
from brax.training.agents.ppo.losses import compute_gae
from brax.training.agents.ppo_lag import networks as ppo_lag_networks
from brax.training.types import Params

# Re-export compute_gae from ppo.losses
__all__ = ['PPONetworkParams', 'compute_gae', 'compute_ppo_lagrange_loss']


@flax.struct.dataclass
class PPONetworkParams:
    """Network parameters for PPO-Lagrange.

    Extends PPO params with cost_value network parameters.
    """
    policy: Params
    value: Params
    cost_value: Params


def compute_ppo_lagrange_loss(
    params: PPONetworkParams,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    ppo_network: ppo_lag_networks.PPONetworks,
    lambda_lagr: jnp.ndarray,
    safety_bound: float = 0.0,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,
) -> Tuple[jnp.ndarray, types.Metrics]:
    """Computes PPO-Lagrange loss.

    Modifies the PPO objective by subtracting lambda * cost_advantage from
    the reward advantage, encouraging the policy to avoid costly states.

    Args:
        params: Network parameters (policy, value, cost_value).
        normalizer_params: Parameters of the observation normalizer.
        data: Transition batch with leading dimension [B, T]. Extra fields
            required: ['state_extras']['truncation'], ['state_extras']['cost'],
            ['policy_extras']['raw_action'], ['policy_extras']['log_prob'].
        rng: Random key for entropy computation.
        ppo_network: PPO-Lagrange networks.
        lambda_lagr: Current Lagrange multiplier.
        safety_bound: Safety constraint bound (unused in loss, for reference).
        entropy_cost: Entropy regularization coefficient.
        discounting: Discount factor for GAE.
        reward_scaling: Reward scaling factor.
        gae_lambda: GAE lambda parameter.
        clipping_epsilon: PPO clipping epsilon.
        normalize_advantage: Whether to normalize advantages.

    Returns:
        A tuple (loss, metrics) where metrics contains component losses.
    """
    parametric_action_distribution = ppo_network.parametric_action_distribution
    policy_apply = ppo_network.policy_network.apply
    value_apply = ppo_network.value_network.apply
    cost_value_apply = ppo_network.cost_value_network.apply

    # Put the time dimension first.
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
    policy_logits = policy_apply(
        normalizer_params, params.policy, data.observation
    )

    baseline = value_apply(normalizer_params, params.value, data.observation)
    cost_baseline = cost_value_apply(normalizer_params, params.cost_value, data.observation)

    terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
    bootstrap_value = value_apply(normalizer_params, params.value, terminal_obs)
    bootstrap_cost_value = cost_value_apply(normalizer_params, params.cost_value, terminal_obs)

    rewards = data.reward * reward_scaling
    # Get costs from state_extras (which comes from state.info during rollout)
    state_extras = data.extras.get('state_extras', {})
    costs = state_extras.get('cost', jnp.zeros_like(rewards))

    truncation = data.extras['state_extras']['truncation']
    termination = (1 - data.discount) * (1 - truncation)

    target_action_log_probs = parametric_action_distribution.log_prob(
        policy_logits, data.extras['policy_extras']['raw_action']
    )
    behaviour_action_log_probs = data.extras['policy_extras']['log_prob']

    # Compute regular advantages and returns
    vs, advantages = compute_gae(
        truncation=truncation,
        termination=termination,
        rewards=rewards,
        values=baseline,
        bootstrap_value=bootstrap_value,
        lambda_=gae_lambda,
        discount=discounting,
    )

    # Compute cost advantages and returns
    cost_vs, cost_advantages = compute_gae(
        truncation=truncation,
        termination=termination,
        rewards=costs,
        values=cost_baseline,
        bootstrap_value=bootstrap_cost_value,
        lambda_=gae_lambda,
        discount=discounting,
    )

    # Modify advantages with Lagrangian term
    if normalize_advantage:
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        cadv = (cost_advantages - cost_advantages.mean()) / (cost_advantages.std() + 1e-8)
    else:
        adv = advantages
        cadv = cost_advantages

    modified_advantages = adv - lambda_lagr * cadv

    rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)

    surrogate_loss1 = rho_s * modified_advantages
    surrogate_loss2 = (
        jnp.clip(rho_s, 1 - clipping_epsilon, 1 + clipping_epsilon) * modified_advantages
    )

    policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

    # Value function loss
    v_error = vs - baseline
    v_loss = jnp.mean(v_error * v_error) * 0.5 * 0.5

    # Cost value function loss
    cost_v_error = cost_vs - cost_baseline
    cost_v_loss = jnp.mean(cost_v_error * cost_v_error) * 0.5 * 0.5

    # Entropy reward
    entropy = jnp.mean(parametric_action_distribution.entropy(policy_logits, rng))
    entropy_loss = entropy_cost * -entropy

    # Compute mean cost for metrics (needed for Lagrange multiplier update)
    mean_cost = jnp.mean(costs)

    total_loss = policy_loss + v_loss + cost_v_loss + entropy_loss
    return total_loss, {
        'total_loss': total_loss,
        'policy_loss': policy_loss,
        'v_loss': v_loss,
        'cost_v_loss': cost_v_loss,
        'entropy_loss': entropy_loss,
        'mean_cost': mean_cost,
    }
