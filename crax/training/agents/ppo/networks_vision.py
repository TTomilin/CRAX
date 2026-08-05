"""PPO vision networks."""

from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

from crax.training import distribution
from crax.training import networks
from crax.training import types
from crax.training.agents.ppo.networks import PPONetworks
from flax import linen
import jax.numpy as jp


ModuleDef = Any
ActivationFn = Callable[[jp.ndarray], jp.ndarray]
Initializer = Callable[..., Any]


def make_ppo_networks_vision(
    observation_size: Mapping[str, Tuple[int, ...]],
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (256, 256),
    value_hidden_layer_sizes: Sequence[int] = (256, 256),
    cost_value_hidden_layer_sizes: Optional[Sequence[int]] = None,
    activation: ActivationFn = linen.swish,
    normalise_channels: bool = False,
    policy_obs_key: str = "",
    value_obs_key: str = "",
) -> PPONetworks:
    """Make Vision PPO networks with preprocessor.

    Args:
        observation_size: Dict mapping observation keys to their shapes.
        action_size: Size of the action space.
        preprocess_observations_fn: Function to preprocess observations.
        policy_hidden_layer_sizes: Hidden layer sizes for the policy MLP head.
        value_hidden_layer_sizes: Hidden layer sizes for the value MLP head.
        cost_value_hidden_layer_sizes: Hidden layer sizes for the cost value
            network. If None, no cost value network is created (standard PPO).
            If provided, creates a cost value network for constrained RL.
        activation: Activation function.
        normalise_channels: Whether to apply per-channel layer normalization
            to pixel inputs.
        policy_obs_key: Key for state observations in the policy network.
        value_obs_key: Key for state observations in the value network.

    Returns:
        PPONetworks with policy, value, and optionally cost_value networks.
    """
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

    cost_value_network = None
    if cost_value_hidden_layer_sizes is not None:
        cost_value_network = networks.make_value_network_vision(
            observation_size=observation_size,
            preprocess_observations_fn=preprocess_observations_fn,
            activation=activation,
            hidden_layer_sizes=cost_value_hidden_layer_sizes,
            state_obs_key=value_obs_key,
            normalise_channels=normalise_channels,
        )

    return PPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
        cost_value_network=cost_value_network,
    )
