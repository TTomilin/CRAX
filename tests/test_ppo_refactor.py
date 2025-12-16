"""
Validation tests for PPO consolidation refactor.
Verifies that ppo, ppo_lag, and ppo_pid trainers work correctly after refactoring.
"""

import jax
import jax.numpy as jnp
import pytest

from brax import envs
from brax.training.agents.ppo import train as ppo_train
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo_lag import train as ppo_lag_train
from brax.training.agents.ppo_lag import losses as ppo_lag_losses
from brax.training.agents.ppo_lag import networks as ppo_lag_networks
from brax.training.agents.ppo_pid import train as ppo_pid_train
from brax.training.agents.ppo_pid import losses as ppo_pid_losses


# ============= Import Tests =============

def test_ppo_imports():
    """Verify all PPO module imports work correctly."""
    assert hasattr(ppo_train, 'train')
    assert hasattr(ppo_train, 'TrainingState')
    assert hasattr(ppo_losses, 'compute_ppo_loss')
    assert hasattr(ppo_losses, 'compute_ppo_lagrange_loss')
    assert hasattr(ppo_losses, 'compute_gae')
    assert hasattr(ppo_losses, 'PPONetworkParams')
    assert hasattr(ppo_networks, 'make_ppo_networks')
    assert hasattr(ppo_networks, 'PPONetworks')
    print("✓ PPO imports OK")


def test_ppo_lag_imports():
    """Verify all PPO-Lagrange module imports work correctly."""
    assert hasattr(ppo_lag_train, 'train')
    assert hasattr(ppo_lag_train, 'TrainingState')
    assert hasattr(ppo_lag_losses, 'compute_ppo_loss')
    assert hasattr(ppo_lag_losses, 'compute_ppo_lagrange_loss')
    assert hasattr(ppo_lag_networks, 'make_ppo_networks')
    assert ppo_lag_train.TrainingState is ppo_train.TrainingState
    print("✓ PPO-Lagrange imports OK")


def test_ppo_pid_imports():
    """Verify all PPO-PID module imports work correctly."""
    assert hasattr(ppo_pid_train, 'train')
    assert hasattr(ppo_pid_train, 'TrainingState')
    assert hasattr(ppo_pid_losses, 'compute_ppo_loss')
    assert hasattr(ppo_pid_losses, 'compute_ppo_lagrange_loss')
    assert ppo_pid_train.TrainingState is ppo_train.TrainingState
    print("✓ PPO-PID imports OK")


# ============= Network Tests =============

def test_ppo_networks_without_cost():
    """Test PPO networks creation without cost value network."""
    networks = ppo_networks.make_ppo_networks(
        observation_size=10,
        action_size=4,
        policy_hidden_layer_sizes=(32, 32),
        value_hidden_layer_sizes=(32, 32),
    )
    assert networks.policy_network is not None
    assert networks.value_network is not None
    assert networks.cost_value_network is None
    print("✓ PPO networks (no cost) OK")


def test_ppo_networks_with_cost():
    """Test PPO networks creation with cost value network."""
    networks = ppo_networks.make_ppo_networks(
        observation_size=10,
        action_size=4,
        policy_hidden_layer_sizes=(32, 32),
        value_hidden_layer_sizes=(32, 32),
        cost_value_hidden_layer_sizes=(32, 32),
    )
    assert networks.policy_network is not None
    assert networks.value_network is not None
    assert networks.cost_value_network is not None
    print("✓ PPO networks (with cost) OK")


# ============= Loss Function Tests =============

def test_compute_gae():
    """Test GAE computation."""
    truncation = jnp.zeros((10,))
    termination = jnp.zeros((10,))
    rewards = jnp.ones((10,))
    values = jnp.ones((11,))
    bootstrap_value = jnp.array(1.0)
    
    vs, advantages = ppo_losses.compute_gae(
        truncation=truncation,
        termination=termination,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        lambda_=0.95,
        discount=0.99,
    )
    assert vs.shape == (10,)
    assert advantages.shape == (10,)
    print("✓ GAE computation OK")


# ============= TrainingState Tests =============

def test_training_state_aux_state():
    """Test that TrainingState supports aux_state field."""
    from brax.training.agents.ppo.train import TrainingState
    import dataclasses
    field_names = [f.name for f in dataclasses.fields(TrainingState)]
    assert 'aux_state' in field_names
    print("✓ TrainingState aux_state field OK")


# ============= Training Smoke Tests =============

@pytest.mark.slow
def test_ppo_train_smoke():
    """Smoke test for base PPO training."""
    env = envs.get_environment('ant')
    make_policy, params, metrics, _ = ppo_train.train(
        environment=env,
        num_timesteps=1000,
        episode_length=100,
        num_envs=4,
        batch_size=32,
        num_minibatches=2,
        num_updates_per_batch=1,
        unroll_length=10,
        num_evals=0,
        seed=0,
    )
    assert make_policy is not None
    assert params is not None
    print("✓ PPO training smoke test OK")


@pytest.mark.slow
def test_ppo_lag_train_smoke():
    """Smoke test for PPO-Lagrange training."""
    env = envs.get_environment('ant')
    make_policy, params, metrics, _ = ppo_lag_train.train(
        environment=env,
        num_timesteps=1000,
        episode_length=100,
        num_envs=4,
        batch_size=32,
        num_minibatches=2,
        num_updates_per_batch=1,
        unroll_length=10,
        num_evals=0,
        safety_bound=10.0,
        lagrangian_coef_rate=0.01,
        initial_lambda_lagr=0.0,
        seed=0,
    )
    assert make_policy is not None
    assert params is not None
    print("✓ PPO-Lagrange training smoke test OK")


@pytest.mark.slow
def test_ppo_pid_train_smoke():
    """Smoke test for PPO-PID training."""
    env = envs.get_environment('ant')
    make_policy, params, metrics, _ = ppo_pid_train.train(
        environment=env,
        num_timesteps=1000,
        episode_length=100,
        num_envs=4,
        batch_size=32,
        num_minibatches=2,
        num_updates_per_batch=1,
        unroll_length=10,
        num_evals=0,
        safety_bound=10.0,
        pid_kp=1.0,
        pid_ki=0.01,
        pid_kd=0.01,
        seed=0,
    )
    assert make_policy is not None
    assert params is not None
    print("✓ PPO-PID training smoke test OK")


if __name__ == '__main__':
    print("\n=== PPO Refactor Validation Tests ===\n")
    
    print("1. Import Tests:")
    test_ppo_imports()
    test_ppo_lag_imports()
    test_ppo_pid_imports()
    
    print("\n2. Network Tests:")
    test_ppo_networks_without_cost()
    test_ppo_networks_with_cost()
    
    print("\n3. Loss Function Tests:")
    test_compute_gae()
    
    print("\n4. TrainingState Tests:")
    test_training_state_aux_state()
    
    print("\n5. Training Smoke Tests (this may take a minute):")
    test_ppo_train_smoke()
    test_ppo_lag_train_smoke()
    test_ppo_pid_train_smoke()
    
    print("\n=== All Tests Passed! ===")
