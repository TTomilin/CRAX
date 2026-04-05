"""SAC-Lagrangian losses.

Extends standard SAC (arxiv 1812.05905) with:
  - A cost Q-network Qc trained on cumulative discounted cost.
  - A Lagrange-penalised actor objective: alpha*log_prob - Q + lambda*Qc.

The cost critic uses the same double-Q architecture as the reward critic but
does NOT subtract the entropy term from the target (CMDP convention: cost is
purely a constraint signal, not an objective to be entropy-regularised).
"""

from typing import Any

from brax.training import types
from brax.training.agents.sac_lag import networks as sac_lag_networks
from brax.training.types import Params, PRNGKey
import jax
import jax.numpy as jnp

Transition = types.Transition


def make_losses(
    sac_network: sac_lag_networks.SACLagNetworks,
    reward_scaling: float,
    discounting: float,
    action_size: int,
):
    """Creates the SAC-Lag losses (alpha, reward critic, cost critic, actor)."""

    target_entropy = -0.5 * action_size
    policy_network = sac_network.policy_network
    q_network = sac_network.q_network
    qc_network = sac_network.qc_network
    parametric_action_distribution = sac_network.parametric_action_distribution

    def alpha_loss(
        log_alpha: jnp.ndarray,
        policy_params: Params,
        normalizer_params: Any,
        transitions: Transition,
        key: PRNGKey,
    ) -> jnp.ndarray:
        """Entropy temperature loss (identical to SAC)."""
        dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.observation
        )
        action = parametric_action_distribution.sample_no_postprocessing(
            dist_params, key
        )
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        alpha = jnp.exp(log_alpha)
        alpha_loss = alpha * jax.lax.stop_gradient(-log_prob - target_entropy)
        return jnp.mean(alpha_loss)

    def critic_loss(
        q_params: Params,
        policy_params: Params,
        normalizer_params: Any,
        target_q_params: Params,
        alpha: jnp.ndarray,
        transitions: Transition,
        key: PRNGKey,
    ) -> jnp.ndarray:
        """Reward critic loss (identical to SAC)."""
        q_old_action = q_network.apply(
            normalizer_params, q_params, transitions.observation, transitions.action
        )
        next_dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.next_observation
        )
        next_action = parametric_action_distribution.sample_no_postprocessing(
            next_dist_params, key
        )
        next_log_prob = parametric_action_distribution.log_prob(
            next_dist_params, next_action
        )
        next_action = parametric_action_distribution.postprocess(next_action)
        next_q = q_network.apply(
            normalizer_params, target_q_params, transitions.next_observation, next_action
        )
        next_v = jnp.min(next_q, axis=-1) - alpha * next_log_prob
        target_q = jax.lax.stop_gradient(
            transitions.reward * reward_scaling
            + transitions.discount * discounting * next_v
        )
        q_error = q_old_action - jnp.expand_dims(target_q, -1)
        truncation = transitions.extras['state_extras']['truncation']
        q_error *= jnp.expand_dims(1 - truncation, -1)
        return 0.5 * jnp.mean(jnp.square(q_error))

    def cost_critic_loss(
        qc_params: Params,
        policy_params: Params,
        normalizer_params: Any,
        target_qc_params: Params,
        transitions: Transition,
        key: PRNGKey,
    ) -> jnp.ndarray:
        """Cost critic loss.

        Same structure as the reward critic, but:
          - Uses per-step cost instead of reward.
          - Does NOT subtract alpha*log_prob from the target value (CMDP convention).
        """
        qc_old_action = qc_network.apply(
            normalizer_params, qc_params, transitions.observation, transitions.action
        )
        next_dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.next_observation
        )
        next_action = parametric_action_distribution.sample_no_postprocessing(
            next_dist_params, key
        )
        next_action = parametric_action_distribution.postprocess(next_action)
        next_qc = qc_network.apply(
            normalizer_params, target_qc_params, transitions.next_observation, next_action
        )
        # No entropy subtraction — cost is a constraint, not entropy-regularised
        next_vc = jnp.min(next_qc, axis=-1)
        cost = transitions.extras['state_extras']['cost']
        target_qc = jax.lax.stop_gradient(
            cost + transitions.discount * discounting * next_vc
        )
        qc_error = qc_old_action - jnp.expand_dims(target_qc, -1)
        truncation = transitions.extras['state_extras']['truncation']
        qc_error *= jnp.expand_dims(1 - truncation, -1)
        return 0.5 * jnp.mean(jnp.square(qc_error))

    def actor_loss(
        policy_params: Params,
        normalizer_params: Any,
        q_params: Params,
        qc_params: Params,
        alpha: jnp.ndarray,
        lambda_lagr: jnp.ndarray,
        transitions: Transition,
        key: PRNGKey,
    ) -> jnp.ndarray:
        """Lagrange-penalised actor loss.

        L_pi = E[alpha * log_prob - Q(s,a) + lambda * Qc(s,a)]

        The lambda term penalises the policy for visiting high-cost states.
        """
        dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.observation
        )
        action = parametric_action_distribution.sample_no_postprocessing(
            dist_params, key
        )
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        action = parametric_action_distribution.postprocess(action)
        q_action = q_network.apply(
            normalizer_params, q_params, transitions.observation, action
        )
        min_q = jnp.min(q_action, axis=-1)
        qc_action = qc_network.apply(
            normalizer_params, qc_params, transitions.observation, action
        )
        min_qc = jnp.min(qc_action, axis=-1)
        actor_loss = alpha * log_prob - min_q + lambda_lagr * min_qc
        return jnp.mean(actor_loss)

    return alpha_loss, critic_loss, cost_critic_loss, actor_loss
