"""Checkpointing for SAC."""

from typing import Any, Union

from crax.training import checkpoint
from crax.training import types
from crax.training.agents.sac import networks as sac_networks
from etils import epath
from ml_collections import config_dict

_CONFIG_FNAME = 'sac_network_config.json'


def save(
    path: Union[str, epath.Path],
    step: int,
    params: Any,
    config: config_dict.ConfigDict,
):
  """Saves a checkpoint."""
  return checkpoint.save(path, step, params, config, _CONFIG_FNAME)


def load(
    path: Union[str, epath.Path],
):
  """Loads SAC checkpoint."""
  return checkpoint.load(path)


def network_config(
    observation_size: types.ObservationSize,
    action_size: int,
    normalize_observations: bool,
    network_factory: types.NetworkFactory[sac_networks.SACNetworks],
) -> config_dict.ConfigDict:
  """Returns a config dict for re-creating a network from a checkpoint."""
  return checkpoint.network_config(
      observation_size, action_size, normalize_observations, network_factory
  )


def _get_network(
    config: config_dict.ConfigDict,
    network_factory: types.NetworkFactory[sac_networks.SACNetworks],
) -> sac_networks.SACNetworks:
  """Generates a SAC network given config."""
  return checkpoint.get_network(config, network_factory)  # pytype: disable=bad-return-type


def load_config(
    path: Union[str, epath.Path],
) -> config_dict.ConfigDict:
  """Loads SAC config from checkpoint."""
  path = epath.Path(path)
  config_path = path / _CONFIG_FNAME
  return checkpoint.load_config(config_path)


def load_policy(
    path: Union[str, epath.Path],
    network_factory: types.NetworkFactory[
        sac_networks.SACNetworks
    ] = sac_networks.make_sac_networks,
    deterministic: bool = True,
):
  """Loads policy inference function from SAC checkpoint."""
  path = epath.Path(path)
  config = load_config(path)
  params = load(path)
  sac_network = _get_network(config, network_factory)
  make_inference_fn = sac_networks.make_inference_fn(sac_network)

  return make_inference_fn(params, deterministic=deterministic)
