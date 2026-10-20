"""PPO-Lagrange loss functions for constrained RL."""

from typing import Any, Tuple

import jax
import jax.numpy as jnp

from training import types
from training.agents.ppo import networks as ppo_networks
from training.agents.ppo.losses import compute_gae

# Re-export for backward compatibility
from training.agents.ppo.losses import PPONetworkParams, compute_ppo_loss, with_shared_latent

__all__ = ["PPONetworkParams", "compute_gae", "compute_ppo_loss", "compute_ppo_lagrange_loss"]


def compute_ppo_lagrange_loss(
    params: PPONetworkParams,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    ppo_network: ppo_networks.PPONetworks,
    lambda_lagr: jnp.ndarray,
    safety_bound: float = 0.0,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,
) -> Tuple[jnp.ndarray, types.Metrics]:
    """Computes PPO-Lagrange loss for constrained RL.

    Args:
      params: Network parameters (must include cost_value).
      normalizer_params: Parameters of the normalizer.
      data: Transition that with leading dimension [B, T]. extra fields required
        are ['state_extras']['truncation'] ['policy_extras']['raw_action']
        ['policy_extras']['log_prob'] and ['state_extras']['cost']
      rng: Random key
      ppo_network: PPO networks (must include cost_value_network).
      lambda_lagr: Current Lagrange multiplier
      safety_bound: Safety constraint bound (per-step)
      entropy_cost: entropy cost.
      discounting: discounting,
      reward_scaling: reward multiplier.
      gae_lambda: General advantage estimation lambda.
      clipping_epsilon: Policy loss clipping epsilon
      normalize_advantage: whether to normalize advantage estimate

    Returns:
      A tuple (loss, metrics)
    """
    parametric_action_distribution = ppo_network.parametric_action_distribution
    policy_apply = ppo_network.policy_network.apply
    value_apply = ppo_network.value_network.apply
    cost_value_apply = ppo_network.cost_value_network.apply

    # Put the time dimension first.
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
    obs = with_shared_latent(ppo_network, params, normalizer_params, data.observation)
    policy_logits = policy_apply(normalizer_params, params.policy, obs)

    baseline = value_apply(normalizer_params, params.value, obs)
    cost_baseline = cost_value_apply(normalizer_params, params.cost_value, obs)

    terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
    terminal_obs = with_shared_latent(ppo_network, params, normalizer_params, terminal_obs)
    bootstrap_value = value_apply(normalizer_params, params.value, terminal_obs)
    bootstrap_cost_value = cost_value_apply(normalizer_params, params.cost_value, terminal_obs)

    rewards = data.reward * reward_scaling
    state_extras = data.extras.get('state_extras', {})
    costs = state_extras.get('cost', jnp.zeros_like(rewards))

    truncation = data.extras['state_extras']['truncation']
    termination = (1 - data.discount) * (1 - truncation)
    # Counterfactual (relabelled) transitions carry truncated IS weights; real ones are on-policy
    is_weight = state_extras.get('is_weight')
    is_real = state_extras.get('is_real', jnp.ones_like(rewards))

    target_action_log_probs = parametric_action_distribution.log_prob(
        policy_logits, data.extras['policy_extras']['raw_action']
    )
    behaviour_action_log_probs = data.extras['policy_extras']['log_prob']

    # Compute regular advantages and returns
    vs, advantages = compute_gae(
        truncation=truncation, termination=termination, rewards=rewards,
        values=baseline, bootstrap_value=bootstrap_value,
        lambda_=gae_lambda, discount=discounting, importance_weights=is_weight,
    )

    # Compute cost advantages and returns
    cost_vs, cost_advantages = compute_gae(
        truncation=truncation, termination=termination, rewards=costs,
        values=cost_baseline, bootstrap_value=bootstrap_cost_value,
        lambda_=gae_lambda, discount=discounting, importance_weights=is_weight,
    )

    # Modify advantages with Lagrangian term
    if normalize_advantage:
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        cadv = (cost_advantages - cost_advantages.mean()) / (cost_advantages.std() + 1e-8)
    else:
        adv = advantages
        cadv = cost_advantages

    modified_advantages = adv - lambda_lagr * cadv
    log_rho = target_action_log_probs - behaviour_action_log_probs
    if is_weight is not None:
        # Relabelled actions can sit far in the tail of the policy at the counterfactual observation, where the
        # ratio overflows after a few SGD steps; the unselected branch of the PPO minimum then backpropagates
        # 0 * inf = NaN. The bound is far outside the clipping range, so on-policy samples are unaffected.
        log_rho = jnp.clip(log_rho, -20.0, 20.0)
    rho_s = jnp.exp(log_rho)

    surrogate_loss1 = rho_s * modified_advantages
    surrogate_loss2 = jnp.clip(rho_s, 1 - clipping_epsilon, 1 + clipping_epsilon) * modified_advantages
    surrogate = jnp.minimum(surrogate_loss1, surrogate_loss2)
    if is_weight is not None:
        surrogate = is_weight * surrogate
    policy_loss = -jnp.mean(surrogate)

    # Value function loss
    v_error = vs - baseline
    v_loss = jnp.mean(v_error * v_error) * 0.5 * 0.5

    # Cost value function loss
    cost_v_error = cost_vs - cost_baseline
    cost_v_loss = jnp.mean(cost_v_error * cost_v_error) * 0.5 * 0.5

    # Entropy reward
    entropy = jnp.mean(parametric_action_distribution.entropy(policy_logits, rng))
    entropy_loss = entropy_cost * -entropy

    # Only real transitions estimate the policy's cost (drives the multiplier update)
    mean_cost = jnp.sum(costs * is_real) / jnp.maximum(jnp.sum(is_real), 1.0)

    # This is to collect metrics for the observation routing ablation in the paper
    # Explained variance of the GAE return targets ssays whether a critic is actually better than predicting the mean
    cost_value_ev = 1.0 - jnp.var(cost_v_error) / (jnp.var(cost_vs) + 1e-8)
    value_ev = 1.0 - jnp.var(v_error) / (jnp.var(vs) + 1e-8)

    total_loss = policy_loss + v_loss + cost_v_loss + entropy_loss
    return total_loss, {
        'total_loss': total_loss,
        'policy_loss': policy_loss,
        'v_loss': v_loss,
        'cost_v_loss': cost_v_loss,
        'entropy_loss': entropy_loss,
        'mean_cost': mean_cost,
        'cost_value_ev': cost_value_ev,
        'value_ev': value_ev,
        'cost_value_mse': jnp.mean(cost_v_error * cost_v_error),
        'value_mse': jnp.mean(v_error * v_error),
        'cost_return_var': jnp.var(cost_vs),
    }
