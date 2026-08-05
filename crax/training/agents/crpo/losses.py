"""CRPO (Constrained Rectified Policy Optimization) loss functions.

Reference: Xu, Liang, Lan, "CRPO: A New Approach for Safe Reinforcement
Learning with Convergence Guarantee", ICML 2021.
https://arxiv.org/abs/2011.05869

CRPO does not use a Lagrange multiplier or a fixed penalty weight. Instead,
at each iteration it "rectifies" the objective: if the constraint is
currently satisfied (within tolerance eta), it takes a pure policy-gradient
step on the reward objective; if the constraint is violated, it takes a pure
policy-gradient step that minimizes cost instead, ignoring reward entirely
for that update. We use the standard PPO clipped surrogate for whichever
single objective is selected, consistent with the other PPO-flavored
baselines in this repo (P3O, FOCOPS):

    optimize_reward:  L(θ) = -min{r(θ)A_R, clip(r(θ), 1-ε, 1+ε)A_R}
    optimize_cost:    L(θ) = -min{r(θ)(-A_C), clip(r(θ), 1-ε, 1+ε)(-A_C)}

`regime` (1.0 = constraint satisfied/optimize reward, 0.0 = constraint
violated/optimize cost) is decided once per training_step from the previous
iteration's episodic cost and threaded in via aux_state (see train.py). The
value and cost-value critics are always fitted regardless of regime.
"""

from typing import Any, Tuple

import jax
import jax.numpy as jnp

from crax.training import types
from crax.training.agents.ppo import networks as ppo_networks
from crax.training.agents.ppo.losses import compute_gae, PPONetworkParams

__all__ = ["PPONetworkParams", "compute_gae", "compute_crpo_loss"]


def compute_crpo_loss(
        params: PPONetworkParams,
        normalizer_params: Any,
        data: types.Transition,
        rng: jnp.ndarray,
        ppo_network: ppo_networks.PPONetworks,
        regime: jnp.ndarray,
        entropy_cost: float = 1e-4,
        discounting: float = 0.99,
        reward_scaling: float = 1.0,
        gae_lambda: float = 0.95,
        clipping_epsilon: float = 0.2,
        normalize_advantage: bool = True,
) -> Tuple[jnp.ndarray, types.Metrics]:
    """Computes the CRPO loss for a single rectified policy-gradient step.

    Args:
      params: Network parameters.
      normalizer_params: Parameters of the normalizer.
      data: Transition with leading dimension [B, T]. Extra fields required
        are ['state_extras']['truncation'], ['policy_extras']['raw_action'],
        ['policy_extras']['log_prob'], and costs.
      rng: Random key.
      ppo_network: PPO networks.
      regime: 1.0 to optimize the reward objective, 0.0 to optimize the cost
        objective, decided in the training loop from the previous
        iteration's constraint violation.
      entropy_cost: Entropy cost coefficient.
      discounting: Discount factor gamma.
      reward_scaling: Reward multiplier.
      gae_lambda: GAE lambda parameter.
      clipping_epsilon: PPO clipping epsilon.
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
    policy_logits = policy_apply(normalizer_params, params.policy, data.observation)

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

    if normalize_advantage:
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        cadv = (cost_advantages - cost_advantages.mean()) / (cost_advantages.std() + 1e-8)
    else:
        adv = advantages
        cadv = cost_advantages

    rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)
    rho_clipped = jnp.clip(rho_s, 1 - clipping_epsilon, 1 + clipping_epsilon)

    # Reward-objective clipped surrogate (standard PPO)
    reward_surrogate1 = rho_s * adv
    reward_surrogate2 = rho_clipped * adv
    reward_policy_loss = -jnp.mean(jnp.minimum(reward_surrogate1, reward_surrogate2))

    # Cost-objective clipped surrogate: maximize -cost, i.e. minimize cost
    neg_cadv = -cadv
    cost_surrogate1 = rho_s * neg_cadv
    cost_surrogate2 = rho_clipped * neg_cadv
    cost_policy_loss = -jnp.mean(jnp.minimum(cost_surrogate1, cost_surrogate2))

    # Rectified switch: pure reward step if feasible, pure cost step otherwise
    regime_scalar = regime[0] if regime.ndim > 0 else regime
    policy_loss = jnp.where(regime_scalar > 0.5, reward_policy_loss, cost_policy_loss)

    # Value function loss (reward) - fitted regardless of regime
    v_error = vs - baseline
    v_loss = jnp.mean(v_error * v_error) * 0.5 * 0.5

    # Cost value function loss - fitted regardless of regime
    cost_v_error = cost_vs - cost_baseline
    cost_v_loss = jnp.mean(cost_v_error * cost_v_error) * 0.5 * 0.5

    # Entropy bonus
    entropy = jnp.mean(parametric_action_distribution.entropy(policy_logits, rng))
    entropy_loss = entropy_cost * -entropy

    mean_cost = jnp.mean(costs)

    total_loss = policy_loss + v_loss + cost_v_loss + entropy_loss

    return total_loss, {
        'total_loss': total_loss,
        'policy_loss': policy_loss,
        'reward_policy_loss': reward_policy_loss,
        'cost_policy_loss': cost_policy_loss,
        'v_loss': v_loss,
        'cost_v_loss': cost_v_loss,
        'entropy_loss': entropy_loss,
        'entropy': entropy,
        'mean_cost': mean_cost,
        'regime': regime_scalar,
    }
