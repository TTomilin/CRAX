"""P3O (Penalized Proximal Policy Optimization) loss functions.

Reference: Zhang et al., "Penalized Proximal Policy Optimization for Safe
Reinforcement Learning", IJCAI 2022.
https://arxiv.org/abs/2205.11814

The P3O objective is:
    L^P3O(θ) = L_R^CLIP(θ) + κ · max{0, L_C^CLIP(θ)}

Where:
    L_R^CLIP(θ) = E[-min{r(θ)A_R, clip(r(θ), 1-ε, 1+ε)A_R}]  (standard PPO)
    L_C^CLIP(θ) = E[max{r(θ)A_C, clip(r(θ), 1-ε, 1+ε)A_C}] + (1-γ)(J_C - d)

The cost loss uses max instead of min to be pessimistic about cost reduction.
The (1-γ)(J_C - d) term represents the gap between current cost and constraint.
"""

from typing import Any, Tuple

import jax
import jax.numpy as jnp

from brax.training import types
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo.losses import compute_gae, PPONetworkParams


def compute_p3o_loss(
        params: PPONetworkParams,
        normalizer_params: Any,
        data: types.Transition,
        rng: jnp.ndarray,
        ppo_network: ppo_networks.PPONetworks,
        kappa: jnp.ndarray,
        cost_violation: jnp.ndarray,
        safety_bound: float = 0.0,
        episode_length: int = 1000,
        entropy_cost: float = 1e-4,
        discounting: float = 0.99,
        reward_scaling: float = 1.0,
        gae_lambda: float = 0.95,
        clipping_epsilon: float = 0.2,
        normalize_advantage: bool = True,
) -> Tuple[jnp.ndarray, types.Metrics]:
    """Computes P3O loss with clipped cost penalty.

    Args:
      params: Network parameters.
      normalizer_params: Parameters of the normalizer.
      data: Transition that with leading dimension [B, T]. extra fields required
        are ['state_extras']['truncation'] ['policy_extras']['raw_action']
        ['policy_extras']['log_prob'] and costs.
      rng: Random key.
      ppo_network: PPO networks.
      kappa: Penalty coefficient for cost constraint.
      cost_violation: Current constraint violation (J_C - d), computed from
        episodic costs in the training loop.
      safety_bound: Safety constraint bound (episodic).
      episode_length: Episode length for per-step bound conversion.
      entropy_cost: Entropy cost coefficient.
      discounting: Discount factor gamma.
      reward_scaling: Reward multiplier.
      gae_lambda: GAE lambda parameter.
      clipping_epsilon: PPO clipping epsilon.
      normalize_advantage: Whether to normalize reward advantage estimates.
        Note: Cost advantages are NOT normalized to preserve scale information.

    Returns:
      A tuple (loss, metrics).
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

    # Compute regular advantages and returns for reward
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
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    else:
        adv = advantages
    # Keep cost advantages unnormalized to preserve scale
    cadv = cost_advantages

    # Importance sampling ratio
    rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)
    rho_clipped = jnp.clip(rho_s, 1 - clipping_epsilon, 1 + clipping_epsilon)

    # Standard PPO clipped reward loss: L_R^CLIP = -min{r*A, clip(r)*A}
    surrogate_loss_r1 = rho_s * adv
    surrogate_loss_r2 = rho_clipped * adv
    reward_loss = -jnp.mean(jnp.minimum(surrogate_loss_r1, surrogate_loss_r2))

    # P3O clipped cost loss: L_C^CLIP = max{r*A_C, clip(r)*A_C} + (1-γ)(J_C - d)
    # Uses max instead of min to be pessimistic about cost reduction
    surrogate_loss_c1 = rho_s * cadv
    surrogate_loss_c2 = rho_clipped * cadv
    clipped_cost_surrogate = jnp.mean(jnp.maximum(surrogate_loss_c1, surrogate_loss_c2))

    # Use the cost_violation passed in from training loop (computed from episodic costs)
    # This is more accurate than using the value network estimate early in training
    # The constraint gap term is (1-γ) * (J_C - d) where cost_violation = J_C - d
    cost_violation_scalar = cost_violation[0] if cost_violation.ndim > 0 else cost_violation
    constraint_gap = (1 - discounting) * cost_violation_scalar

    # Total cost loss before penalty
    cost_loss_unpenalized = clipped_cost_surrogate + constraint_gap

    # Apply ReLU: only penalize when constraint is violated
    # Note: we also add a small positive term based purely on violation to ensure gradient signal
    cost_loss_relu = jax.nn.relu(cost_loss_unpenalized)

    # Also add direct violation penalty (ensures gradient even when surrogate is negative)
    direct_violation_penalty = jax.nn.relu(cost_violation_scalar)

    # Apply penalty coefficient kappa
    kappa_scalar = kappa[0] if kappa.ndim > 0 else kappa
    cost_penalty = kappa_scalar * (cost_loss_relu + 0.1 * direct_violation_penalty)

    # Policy loss combines reward maximization and cost penalty
    policy_loss = reward_loss + cost_penalty

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
    mean_cost_value = jnp.mean(cost_baseline)

    total_loss = policy_loss + v_loss + cost_v_loss + entropy_loss

    return total_loss, {
        'total_loss': total_loss,
        'policy_loss': policy_loss,
        'reward_loss': reward_loss,
        'cost_penalty': cost_penalty,
        'cost_loss_relu': cost_loss_relu,
        'direct_violation_penalty': direct_violation_penalty,
        'cost_loss_unpenalized': cost_loss_unpenalized,
        'constraint_gap': constraint_gap,
        'clipped_cost_surrogate': clipped_cost_surrogate,
        'v_loss': v_loss,
        'cost_v_loss': cost_v_loss,
        'entropy_loss': entropy_loss,
        'entropy': entropy,
        'mean_cost': mean_cost,
        'mean_cost_value': mean_cost_value,
        'cost_violation': cost_violation_scalar,
        'kappa': kappa_scalar,
    }
