"""SAC-Lagrangian networks — SAC extended with a cost Q-network."""

from typing import Sequence, Tuple

from crax.training import distribution
from crax.training import networks
from crax.training import types
from crax.training.types import PRNGKey
import flax
from flax import linen


@flax.struct.dataclass
class SACLagNetworks:
    policy_network: networks.FeedForwardNetwork
    q_network: networks.FeedForwardNetwork
    qc_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


def make_inference_fn(sac_lag_networks: SACLagNetworks):
    """Creates inference function for the SAC-Lag agent (same as SAC)."""

    def make_policy(
        params: types.PolicyParams, deterministic: bool = False
    ) -> types.Policy:

        def policy(
            observations: types.Observation, key_sample: PRNGKey
        ) -> Tuple[types.Action, types.Extra]:
            logits = sac_lag_networks.policy_network.apply(*params, observations)
            if deterministic:
                return sac_lag_networks.parametric_action_distribution.mode(logits), {}
            return (
                sac_lag_networks.parametric_action_distribution.sample(
                    logits, key_sample
                ),
                {},
            )

        return policy

    return make_policy


def make_sac_lag_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    policy_network_layer_norm: bool = False,
    q_network_layer_norm: bool = False,
) -> SACLagNetworks:
    """Make SAC-Lag networks (adds cost Q-network to standard SAC)."""
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    policy_network = networks.make_policy_network(
        parametric_action_distribution.param_size,
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        layer_norm=policy_network_layer_norm,
    )
    q_network = networks.make_q_network(
        observation_size,
        action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        layer_norm=q_network_layer_norm,
    )
    qc_network = networks.make_q_network(
        observation_size,
        action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        layer_norm=q_network_layer_norm,
    )
    return SACLagNetworks(
        policy_network=policy_network,
        q_network=q_network,
        qc_network=qc_network,
        parametric_action_distribution=parametric_action_distribution,
    )
