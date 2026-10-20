"""Tests for counterfactual hazard relabelling."""

from unittest import mock

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from crax import envs
from training import acting
from training import counterfactual


def _random_policy(action_size):
  def policy(obs, key):
    action = jax.random.uniform(key, obs.shape[:-1] + (action_size,), minval=-1.0, maxval=1.0)
    return action, {}
  return policy


class CounterfactualTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.env = envs.get_environment('safe_pathway_walker2d', level=3)
    # Short episodes so the unroll crosses auto-resets
    self.wrapped = envs.training.wrap(self.env, episode_length=20, action_repeat=1)

  def _unroll(self, num_steps):
    state = jax.jit(self.wrapped.reset)(jax.random.split(jax.random.PRNGKey(0), 1))
    unroll = jax.jit(
        lambda s, k: acting.generate_unroll(
            self.wrapped, s, _random_policy(self.env.action_size), k, num_steps,
            extra_fields=('truncation', 'cost') + counterfactual.CF_EXTRA_FIELDS))
    _, data = unroll(state, jax.random.PRNGKey(1))
    # [T, B, ...] -> [B, T, ...]
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
    return state, data

  def testRelabelWithRealLayoutIsIdentity(self):
    state, data = self._unroll(50)
    self.assertGreater(float(jnp.sum(1 - data.discount)), 0, 'unroll should cross a reset')
    layout = state.info['hazard_positions'][0]
    relabel = counterfactual.make_hazard_relabel_fn(self.env, mode='random')
    with mock.patch.object(counterfactual, '_sample_layout_x', lambda env, key: (layout[:, 0], layout[:, 1])):
      cf, _ = jax.jit(relabel)(data, jax.random.PRNGKey(2))

    np.testing.assert_allclose(cf.observation, data.observation, atol=1e-5)
    se, cf_se = data.extras['state_extras'], cf.extras['state_extras']
    np.testing.assert_allclose(cf_se['cost'], se['cost'], atol=1e-5)
    # next_observation is the post-reset observation at episode ends
    not_done = data.discount[..., None] > 0
    np.testing.assert_allclose(
        jnp.where(not_done, cf.next_observation, 0), jnp.where(not_done, data.next_observation, 0), atol=1e-5)

  def testWindowedCostMatchesFullLayout(self):
    layout = self.env.sample_hazard_positions(jax.random.PRNGKey(3))
    # Feet on the ground at hazards 10 and 11, slightly off-centre
    feet = layout[10:12].at[:, :2].add(0.05).at[:, 2].set(0.0)
    full = self.env.hazard_cost_from_feet(feet, layout)
    ids = jnp.arange(4, 20)
    windowed = self.env.hazard_cost_from_feet(feet, layout[ids], ids)
    self.assertGreater(float(full), 0)
    np.testing.assert_allclose(windowed, full, rtol=1e-6)

  def testOptimisticNeverCostlierThanRandom(self):
    _, data = self._unroll(16)
    # Plant both feet on the ground at x in [2, 6] so random layouts incur hazard cost
    feet = data.extras['state_extras']['cf_feet']
    xs = jnp.linspace(2.0, 6.0, feet.shape[1])[None, :, None]
    feet = feet.at[..., 0].set(xs).at[..., 1].set(0.0).at[..., 2].set(0.0)
    data.extras['state_extras']['cf_feet'] = feet
    data = jax.tree_util.tree_map(lambda x: jnp.repeat(x, 64, axis=0), data)
    costs = {}
    for mode in ('random', 'optimistic'):
      relabel = counterfactual.make_hazard_relabel_fn(self.env, mode=mode, num_candidates=8)
      _, metrics = jax.jit(relabel)(data, jax.random.PRNGKey(4))
      costs[mode] = float(metrics['cf/hazard_cost'])
    self.assertGreater(costs['random'], 0)
    self.assertLess(costs['optimistic'], costs['random'])


if __name__ == '__main__':
  absltest.main()
