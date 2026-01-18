"""PPO vision networks."""

from typing import Any, Callable, Mapping, Sequence, Tuple

from brax.training import distribution
from brax.training import networks
from brax.training import types
import flax
from flax import linen
import jax.numpy as jp


ModuleDef = Any
ActivationFn = Callable[[jp.ndarray], jp.ndarray]
Initializer = Callable[..., Any]


@flax.struct.dataclass
class PPONetworks:
  policy_network: networks.FeedForwardNetwork
  value_network: networks.FeedForwardNetwork
  parametric_action_distribution: distribution.ParametricDistribution


def make_ppo_networks_vision(
    observation_size: Mapping[str, Tuple[int, ...]],
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (256, 256),
    value_hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.swish,
    normalise_channels: bool = False,
    policy_obs_key: str = "",
    value_obs_key: str = "",
) -> PPONetworks:
  """Make Vision PPO networks with preprocessor."""

  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=action_size
  )

  policy_network = networks.make_policy_network_vision(
      observation_size=observation_size,
      output_size=parametric_action_distribution.param_size,
      preprocess_observations_fn=preprocess_observations_fn,
      activation=activation,
      hidden_layer_sizes=policy_hidden_layer_sizes,
      state_obs_key=policy_obs_key,
      normalise_channels=normalise_channels,
  )

  value_network = networks.make_value_network_vision(
      observation_size=observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      activation=activation,
      hidden_layer_sizes=value_hidden_layer_sizes,
      state_obs_key=value_obs_key,
      normalise_channels=normalise_channels,
  )

  return PPONetworks(
      policy_network=policy_network,
      value_network=value_network,
      parametric_action_distribution=parametric_action_distribution,
  )
