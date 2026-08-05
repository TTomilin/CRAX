"""PPO networks."""

from typing import Optional, Sequence, Tuple

from crax.training import distribution
from crax.training import networks
from crax.training import types
from crax.training.types import PRNGKey
import flax
from flax import linen


@flax.struct.dataclass
class PPONetworks:
  policy_network: networks.FeedForwardNetwork
  value_network: networks.FeedForwardNetwork
  parametric_action_distribution: distribution.ParametricDistribution
  cost_value_network: Optional[networks.FeedForwardNetwork] = None


def make_inference_fn(ppo_networks: PPONetworks):
  """Creates params and inference function for the PPO agent."""

  def make_policy(
      params: types.Params, deterministic: bool = False
  ) -> types.Policy:
    policy_network = ppo_networks.policy_network
    parametric_action_distribution = ppo_networks.parametric_action_distribution

    def policy(
        observations: types.Observation, key_sample: PRNGKey
    ) -> Tuple[types.Action, types.Extra]:
      param_subset = (params[0], params[1])  # normalizer and policy params
      logits = policy_network.apply(*param_subset, observations)
      if deterministic:
        return ppo_networks.parametric_action_distribution.mode(logits), {}
      raw_actions = parametric_action_distribution.sample_no_postprocessing(
          logits, key_sample
      )
      log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
      postprocessed_actions = parametric_action_distribution.postprocess(
          raw_actions
      )
      return postprocessed_actions, {
          'log_prob': log_prob,
          'raw_action': raw_actions,
      }

    return policy

  return make_policy


def make_ppo_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    cost_value_hidden_layer_sizes: Optional[Sequence[int]] = None,
    activation: networks.ActivationFn = linen.swish,
    policy_obs_key: str = 'state',
    value_obs_key: str = 'state',
) -> PPONetworks:
  """Make PPO networks with preprocessor.

  Args:
    observation_size: Size of observations.
    action_size: Size of actions.
    preprocess_observations_fn: Function to preprocess observations.
    policy_hidden_layer_sizes: Hidden layer sizes for policy network.
    value_hidden_layer_sizes: Hidden layer sizes for value network.
    cost_value_hidden_layer_sizes: Hidden layer sizes for cost value network.
      If None, no cost value network is created (standard PPO).
      If provided, creates a cost value network for constrained RL variants.
    activation: Activation function for networks.
    policy_obs_key: Key for policy observations.
    value_obs_key: Key for value observations.

  Returns:
    PPONetworks with policy, value, and optionally cost_value networks.
  """
  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=action_size
  )
  policy_network = networks.make_policy_network(
      parametric_action_distribution.param_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=policy_hidden_layer_sizes,
      activation=activation,
      obs_key=policy_obs_key,
  )
  value_network = networks.make_value_network(
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=value_hidden_layer_sizes,
      activation=activation,
      obs_key=value_obs_key,
  )

  cost_value_network = None
  if cost_value_hidden_layer_sizes is not None:
    cost_value_network = networks.make_value_network(
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=cost_value_hidden_layer_sizes,
        activation=activation,
        obs_key=value_obs_key,
    )

  return PPONetworks(
      policy_network=policy_network,
      value_network=value_network,
      parametric_action_distribution=parametric_action_distribution,
      cost_value_network=cost_value_network,
  )
