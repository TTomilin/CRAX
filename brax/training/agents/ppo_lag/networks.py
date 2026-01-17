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

"""PPO-Lagrange networks.

Extends PPO networks with an additional cost value network for constrained optimization.
"""

from typing import Sequence

from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.agents.ppo.networks import make_inference_fn
import flax
from flax import linen

# Re-export make_inference_fn from ppo (identical implementation)
__all__ = ['PPONetworks', 'make_inference_fn', 'make_ppo_networks']


@flax.struct.dataclass
class PPONetworks:
    """PPO-Lagrange networks with cost value network."""
    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork
    cost_value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


def make_ppo_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    cost_value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    activation: networks.ActivationFn = linen.swish,
    policy_obs_key: str = 'state',
    value_obs_key: str = 'state',
) -> PPONetworks:
    """Make PPO-Lagrange networks with preprocessor.

    Extends standard PPO networks with an additional cost value network
    for estimating expected cumulative costs.

    Args:
        observation_size: Size of the observation space.
        action_size: Size of the action space.
        preprocess_observations_fn: Function to preprocess observations.
        policy_hidden_layer_sizes: Hidden layer sizes for policy network.
        value_hidden_layer_sizes: Hidden layer sizes for value network.
        cost_value_hidden_layer_sizes: Hidden layer sizes for cost value network.
        activation: Activation function for hidden layers.
        policy_obs_key: Key for policy observations in dict observations.
        value_obs_key: Key for value observations in dict observations.

    Returns:
        PPONetworks dataclass with policy, value, cost_value networks and
        action distribution.
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
        cost_value_network=cost_value_network,
        parametric_action_distribution=parametric_action_distribution,
    )
