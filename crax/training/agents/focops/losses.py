"""FOCOPS (First Order Constrained Optimization in Policy Space) loss functions.

Reference: Zhang et al., "First Order Constrained Optimization in Policy Space",
NeurIPS 2020.
https://arxiv.org/abs/2002.06506

The optimal policy takes the form:
    π*(a|s) ∝ π_k(a|s) × exp[(1/λ)(A^R(s,a) - ν×A^C(s,a))]

Where:
    - λ (focops_lam): Fixed temperature parameter controlling exploration
    - ν (nu): Lagrange multiplier for cost constraint, adapted during training

The loss minimizes KL divergence from this optimal policy, leading to:
    L(θ) = D_KL(π_θ || π*) ≈ -E[(1/λ)(A^R - ν×A^C) × ratio] + KL regularization
"""

from typing import Any, Tuple

import jax
import jax.numpy as jnp

from crax.training import types
from crax.training.agents.ppo import networks as ppo_networks
from crax.training.agents.ppo.losses import compute_gae, PPONetworkParams


def compute_focops_loss(
        params: PPONetworkParams,
        normalizer_params: Any,
        data: types.Transition,
        rng: jnp.ndarray,
        ppo_network: ppo_networks.PPONetworks,
        nu: jnp.ndarray,
        focops_lam: float = 1.5,
        focops_eta: float = 0.02,
        entropy_cost: float = 1e-4,
        discounting: float = 0.99,
        reward_scaling: float = 1.0,
        gae_lambda: float = 0.95,
        normalize_advantage: bool = True,
) -> Tuple[jnp.ndarray, types.Metrics]:
    """Computes FOCOPS loss.

    Args:
      params: Network parameters.
      normalizer_params: Parameters of the normalizer.
      data: Transition with leading dimension [B, T].
      rng: Random key.
      ppo_network: PPO networks.
      nu: Lagrange multiplier for cost constraint.
      focops_lam: Temperature parameter λ (fixed, controls exploration).
      focops_eta: KL threshold for gradient masking.
      entropy_cost: Entropy bonus coefficient.
      discounting: Discount factor gamma.
      reward_scaling: Reward multiplier.
      gae_lambda: GAE lambda parameter.
      normalize_advantage: Whether to normalize advantage estimates.

    Returns:
      A tuple (loss, metrics).
    """
    parametric_action_distribution = ppo_network.parametric_action_distribution
    policy_apply = ppo_network.policy_network.apply
    value_apply = ppo_network.value_network.apply
    cost_value_apply = ppo_network.cost_value_network.apply

    # Put the time dimension first.
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    # Current policy distribution
    policy_logits = policy_apply(
        normalizer_params, params.policy, data.observation
    )

    baseline = value_apply(normalizer_params, params.value, data.observation)
    cost_baseline = cost_value_apply(normalizer_params, params.cost_value, data.observation)

    terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
    bootstrap_value = value_apply(normalizer_params, params.value, terminal_obs)
    bootstrap_cost_value = cost_value_apply(normalizer_params, params.cost_value, terminal_obs)

    rewards = data.reward * reward_scaling
    state_extras = data.extras.get('state_extras', {})
    costs = state_extras.get('cost', jnp.zeros_like(rewards))

    truncation = data.extras['state_extras']['truncation']
    termination = (1 - data.discount) * (1 - truncation)

    # Log probs
    target_action_log_probs = parametric_action_distribution.log_prob(
        policy_logits, data.extras['policy_extras']['raw_action']
    )
    behaviour_action_log_probs = data.extras['policy_extras']['log_prob']

    # Compute reward advantages and returns
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

    # Normalize reward advantages only (not cost advantages!)
    # Cost advantages need their raw scale for proper constraint enforcement
    if normalize_advantage:
        adv_r = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    else:
        adv_r = advantages
    # Keep cost advantages unnormalized to preserve scale
    adv_c = cost_advantages

    # FOCOPS combined advantage: (A^R - ν×A^C)
    # Scaled by 1/(1+ν) for numerical stability when ν is large
    nu_scalar = nu[0] if nu.ndim > 0 else nu
    combined_advantage = (adv_r - nu_scalar * adv_c) / (1.0 + nu_scalar)

    # Importance sampling ratio
    log_ratio = target_action_log_probs - behaviour_action_log_probs
    ratio = jnp.exp(log_ratio)

    # KL divergence (approximate): D_KL ≈ (ratio - 1) - log(ratio)
    # Or simply use the difference in log probs for a simpler approximation
    kl = jnp.mean((ratio - 1) - log_ratio)

    # FOCOPS loss: maximize combined advantage weighted by ratio
    # With KL masking: only apply gradient when KL is within threshold
    # L = -E[ratio × combined_advantage] when KL <= eta, else just KL penalty

    # Policy gradient term (maximize advantage)
    pg_loss = -jnp.mean(ratio * combined_advantage)

    # Apply FOCOPS eta masking: reduce policy gradient when KL is large
    # Mask = 1 when kl <= eta, smoothly reduces otherwise
    kl_mask = jnp.where(kl <= focops_eta, 1.0, focops_eta / (kl + 1e-8))

    # Scaled policy loss with temperature λ
    policy_loss = pg_loss / focops_lam * kl_mask

    # Value function loss (reward)
    v_error = vs - baseline
    v_loss = jnp.mean(v_error * v_error) * 0.5 * 0.5

    # Cost value function loss
    cost_v_error = cost_vs - cost_baseline
    cost_v_loss = jnp.mean(cost_v_error * cost_v_error) * 0.5 * 0.5

    # Entropy bonus
    entropy = jnp.mean(parametric_action_distribution.entropy(policy_logits, rng))
    entropy_loss = entropy_cost * -entropy

    # Mean cost for metrics
    mean_cost = jnp.mean(costs)

    total_loss = policy_loss + v_loss + cost_v_loss + entropy_loss

    return total_loss, {
        'total_loss': total_loss,
        'policy_loss': policy_loss,
        'pg_loss': pg_loss,
        'v_loss': v_loss,
        'cost_v_loss': cost_v_loss,
        'entropy_loss': entropy_loss,
        'entropy': entropy,
        'kl': kl,
        'kl_mask': kl_mask,
        'mean_cost': mean_cost,
        'nu': nu_scalar,
        'combined_advantage_mean': jnp.mean(combined_advantage),
        'ratio_mean': jnp.mean(ratio),
    }
