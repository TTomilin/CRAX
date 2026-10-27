"""Tests for counterfactual hazard relabelling."""

from unittest import mock

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from crax import envs
from training import acting
from training import counterfactual
from training import types
from training.agents.crpo import losses as crpo_losses
from training.agents.p3o import losses as p3o_losses
from training.agents.ppo import losses as ppo_losses
from training.agents.ppo import networks as ppo_networks
from training.agents.ppo_lag import losses as ppo_lag_losses


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

    # 'adaptive' interpolates between the two through the optimistic share
    relabel = jax.jit(counterfactual.make_hazard_relabel_fn(self.env, mode='adaptive', num_candidates=8))
    shares, adaptive_costs = [], []
    for p in (0.0, 1.0):
      _, metrics = relabel(data, jax.random.PRNGKey(4), p)
      shares.append(float(metrics['cf/optimistic_share']))
      adaptive_costs.append(float(metrics['cf/hazard_cost']))
    self.assertEqual(shares, [0.0, 1.0])
    self.assertGreater(adaptive_costs[0], adaptive_costs[1])
    np.testing.assert_allclose(adaptive_costs[1], costs['optimistic'], rtol=1e-5)

  def testGatedKeepsRealDataWhenNotOptimistic(self):
    _, data = self._unroll(16)
    data = jax.tree_util.tree_map(lambda x: jnp.repeat(x, 8, axis=0), data)
    relabel = jax.jit(counterfactual.make_hazard_relabel_fn(self.env, mode='gated', num_candidates=4))
    cf, metrics = relabel(data, jax.random.PRNGKey(4), 0.0)
    self.assertEqual(float(metrics['cf/optimistic_share']), 0.0)
    np.testing.assert_array_equal(cf.observation, data.observation)
    np.testing.assert_array_equal(cf.next_observation, data.next_observation)
    np.testing.assert_array_equal(cf.extras['state_extras']['cost'], data.extras['state_extras']['cost'])
    # Optimistic segments match 'adaptive' mode
    cf, _ = relabel(data, jax.random.PRNGKey(4), 1.0)
    adaptive = jax.jit(counterfactual.make_hazard_relabel_fn(self.env, mode='adaptive', num_candidates=4))
    reference, _ = adaptive(data, jax.random.PRNGKey(4), 1.0)
    np.testing.assert_array_equal(cf.observation, reference.observation)

  def testSlackGate(self):
    gate = counterfactual.make_slack_gate(lambda aux_state: 0.01, sharpness=10.0)
    self.assertAlmostEqual(float(gate(None, 0.01)), 0.5, places=5)
    self.assertGreater(float(gate(None, 0.0)), 0.99)
    self.assertLess(float(gate(None, 0.02)), 0.01)


class ConstrainedLossesTest(absltest.TestCase):
  """Constrained PPO losses on a batch of real and relabelled segments."""

  def _batch(self):
    num_real, num_cf, t, obs_size, action_size = 4, 4, 5, 6, 2
    b = num_real + num_cf
    k_obs, k_act = jax.random.split(jax.random.PRNGKey(0))
    is_real = jnp.concatenate([jnp.ones((num_real, t)), jnp.zeros((num_cf, t))])
    # Relabelled actions far in the tail: their log-ratio would overflow without the bound
    log_prob = jnp.where(is_real > 0, -1.0, -1e3)
    episode_done = jnp.zeros((b, t)).at[:, -1].set(1.0)
    data = types.Transition(
        observation=jax.random.normal(k_obs, (b, t, obs_size)),
        action=jnp.zeros((b, t, action_size)),
        reward=jnp.ones((b, t)),
        discount=jnp.ones((b, t)),
        next_observation=jax.random.normal(k_obs, (b, t, obs_size)),
        extras={
            'policy_extras': {
                'raw_action': jax.random.normal(k_act, (b, t, action_size)),
                'log_prob': log_prob,
            },
            'state_extras': {
                'truncation': jnp.zeros((b, t)),
                # Real cost 1, relabelled cost 100: the cost estimate must ignore the latter
                'cost': jnp.where(is_real > 0, 1.0, 100.0),
                'episode_done': episode_done,
                'episode_metrics': {'length': jnp.full((b, t), 7.0)},
                'is_real': is_real,
                'is_weight': jnp.where(is_real > 0, 1.0, 0.5),
            },
        })
    network = ppo_networks.make_ppo_networks(
        obs_size, action_size, policy_hidden_layer_sizes=(8,), value_hidden_layer_sizes=(8,),
        cost_value_hidden_layer_sizes=(8,))
    keys = jax.random.split(jax.random.PRNGKey(1), 3)
    params = ppo_losses.PPONetworkParams(
        policy=network.policy_network.init(keys[0]),
        value=network.value_network.init(keys[1]),
        cost_value=network.cost_value_network.init(keys[2]))
    return data, network, params, num_real

  def testRealOnlyCostAndFiniteGradients(self):
    data, network, params, num_real = self._batch()
    common = dict(normalizer_params=None, data=data, rng=jax.random.PRNGKey(2), ppo_network=network)
    losses = {
        'ppo_lag': lambda p: ppo_lag_losses.compute_ppo_lagrange_loss(
            params=p, lambda_lagr=jnp.array([1.0]), **common),
        'crpo_reward': lambda p: crpo_losses.compute_crpo_loss(params=p, regime=jnp.array([1.0]), **common),
        'crpo_cost': lambda p: crpo_losses.compute_crpo_loss(params=p, regime=jnp.array([0.0]), **common),
        'p3o': lambda p: p3o_losses.compute_p3o_loss(
            params=p, kappa=jnp.array([1.0]), cost_violation=jnp.array([0.5]), **common),
    }
    for name, loss_fn in losses.items():
      grads, metrics = jax.grad(loss_fn, has_aux=True)(params)
      self.assertAlmostEqual(float(metrics['mean_cost']), 1.0, places=5, msg=name)
      self.assertEqual(float(metrics['episodes_done']), num_real, msg=name)
      self.assertAlmostEqual(float(metrics['episode_length_sum']), 7.0 * num_real, places=4, msg=name)
      for leaf in jax.tree_util.tree_leaves(grads):
        self.assertTrue(bool(jnp.all(jnp.isfinite(leaf))), msg=name)



class ReacherCounterfactualTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.env = envs.get_environment('safe_reacher', level=3)
    # Short episodes so the unroll crosses auto-resets
    self.wrapped = envs.training.wrap(self.env, episode_length=15, action_repeat=1)

  def _unroll(self, num_steps, num_envs=2):
    state = jax.jit(self.wrapped.reset)(jax.random.split(jax.random.PRNGKey(0), num_envs))
    unroll = jax.jit(
        lambda s, k: acting.generate_unroll(
            self.wrapped, s, _random_policy(self.env.action_size), k, num_steps,
            extra_fields=('truncation', 'cost') + counterfactual.REACHER_CF_EXTRA_FIELDS))
    _, data = unroll(state, jax.random.PRNGKey(1))
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
    return state, data

  def testRelabelWithRealLayoutIsIdentity(self):
    state, data = self._unroll(40)
    self.assertGreater(float(jnp.sum(data.extras['state_extras']['truncation'])), 0, 'unroll should cross a reset')
    self.assertGreater(float(jnp.sum(data.extras['state_extras']['cost'])), 0, 'random arm should hit hazards')
    layouts = state.info['hazard_positions']
    for i in range(layouts.shape[0]):
      segment = jax.tree_util.tree_map(lambda x: x[i:i + 1], data)
      # A fresh relabel_fn per layout: a reused one would replay the first trace
      relabel = counterfactual.make_reacher_relabel_fn(self.env, mode='random')
      with mock.patch.object(type(self.env.unwrapped), 'sample_hazard_positions_batched',
                             lambda env, key, goal, layout=layouts[i]: layout):
        cf, _ = jax.jit(relabel)(segment, jax.random.PRNGKey(2))
      np.testing.assert_allclose(cf.observation, segment.observation, atol=1e-5)
      np.testing.assert_allclose(cf.next_observation, segment.next_observation, atol=1e-5)
      np.testing.assert_allclose(cf.extras['state_extras']['cost'], segment.extras['state_extras']['cost'], atol=1e-5)

  def testBatchedLayoutsAreValid(self):
    env = self.env.unwrapped
    goal = jnp.array([0.1, -0.08])
    layouts = jax.jit(jax.vmap(lambda k: env.sample_hazard_positions_batched(k, goal)))(
        jax.random.split(jax.random.PRNGKey(3), 256))
    rect_r = jnp.sqrt(jnp.sum(env._hazard_half_extents ** 2, axis=1))
    haz_r = jnp.where(env._hazard_is_rect, rect_r, env._hazard_radii)
    xy = layouts[..., :2]
    d = jnp.linalg.norm(xy[:, :, None] - xy[:, None, :], axis=-1)
    min_d = haz_r[:, None] + haz_r[None, :]
    off_diag = ~jnp.eye(xy.shape[1], dtype=bool)
    self.assertTrue(bool(jnp.all(jnp.where(off_diag, d > min_d, True))), 'hazards overlap')
    self.assertTrue(bool(jnp.all(jnp.linalg.norm(xy - goal, axis=-1) > haz_r + env._goal_radius)))
    self.assertTrue(bool(jnp.all(jnp.linalg.norm(xy, axis=-1) <= env._hazard_placement_radius)))
    # Same placement distribution as the env's rejection sampler
    reference = jax.jit(jax.vmap(lambda k: env.sample_hazard_positions(k, goal)))(
        jax.random.split(jax.random.PRNGKey(4), 256))
    np.testing.assert_allclose(jnp.mean(jnp.linalg.norm(xy, axis=-1)),
                               jnp.mean(jnp.linalg.norm(reference[..., :2], axis=-1)), rtol=0.05)

  def testOptimisticNeverCostlierThanRandom(self):
    _, data = self._unroll(16, num_envs=64)
    costs = {}
    for mode in ('random', 'optimistic'):
      relabel = counterfactual.make_reacher_relabel_fn(self.env, mode=mode, num_candidates=8)
      _, metrics = jax.jit(relabel)(data, jax.random.PRNGKey(5))
      costs[mode] = float(metrics['cf/hazard_cost'])
    self.assertGreater(costs['random'], 0)
    self.assertLess(costs['optimistic'], costs['random'])


  def testGatedKeepsRealDataWhenNotOptimistic(self):
    _, data = self._unroll(16, num_envs=8)
    relabel = jax.jit(counterfactual.make_reacher_relabel_fn(self.env, mode='gated', num_candidates=4))
    cf, metrics = relabel(data, jax.random.PRNGKey(5), 0.0)
    self.assertEqual(float(metrics['cf/optimistic_share']), 0.0)
    np.testing.assert_array_equal(cf.observation, data.observation)
    np.testing.assert_array_equal(cf.next_observation, data.next_observation)
    np.testing.assert_array_equal(cf.extras['state_extras']['cost'], data.extras['state_extras']['cost'])
    cf, _ = relabel(data, jax.random.PRNGKey(5), 1.0)
    adaptive = jax.jit(counterfactual.make_reacher_relabel_fn(self.env, mode='adaptive', num_candidates=4))
    reference, _ = adaptive(data, jax.random.PRNGKey(5), 1.0)
    np.testing.assert_array_equal(cf.observation, reference.observation)


class ButtonCounterfactualTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.env = envs.get_environment('safe_button_point', level=3)
    # Short episodes so the unroll crosses auto-resets
    self.wrapped = envs.training.wrap(self.env, episode_length=15, action_repeat=1)

  def _unroll(self, num_steps, num_envs=2):
    state = jax.jit(self.wrapped.reset)(jax.random.split(jax.random.PRNGKey(0), num_envs))
    unroll = jax.jit(
        lambda s, k: acting.generate_unroll(
            self.wrapped, s, _random_policy(self.env.action_size), k, num_steps,
            extra_fields=('truncation', 'cost') + counterfactual.BUTTON_CF_EXTRA_FIELDS))
    _, data = unroll(state, jax.random.PRNGKey(1))
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
    return state, data

  def testRelabelWithRealLayoutIsIdentity(self):
    state, data = self._unroll(40)
    self.assertGreater(float(jnp.sum(data.extras['state_extras']['truncation'])), 0, 'unroll should cross a reset')
    env = self.env.unwrapped
    floors = state.info['hazard_positions'][:, env._cf_hazard_indices, :2]
    for i in range(floors.shape[0]):
      segment = jax.tree_util.tree_map(lambda x: x[i:i + 1], data)
      # A fresh relabel_fn per layout: a reused one would replay the first trace
      relabel = counterfactual.make_button_relabel_fn(self.env, mode='random')
      with mock.patch.object(type(env), 'sample_cf_hazard_positions',
                             lambda env, key, anchors, layout=floors[i]: layout):
        cf, _ = jax.jit(relabel)(segment, jax.random.PRNGKey(2))
      np.testing.assert_allclose(cf.observation, segment.observation, atol=1e-5)
      np.testing.assert_allclose(cf.next_observation, segment.next_observation, atol=1e-5)
      np.testing.assert_allclose(cf.extras['state_extras']['cost'], segment.extras['state_extras']['cost'], atol=1e-5)

  def testOnlyFloorHazardsMove(self):
    state, data = self._unroll(16, num_envs=4)
    env = self.env.unwrapped
    cf, _ = jax.jit(counterfactual.make_button_relabel_fn(self.env, mode='random'))(data, jax.random.PRNGKey(3))
    tail = env.hazard_obs_size
    np.testing.assert_array_equal(cf.observation[..., :-tail], data.observation[..., :-tail])
    self.assertGreater(float(jnp.mean(jnp.abs(cf.observation[..., -tail:] - data.observation[..., -tail:]))), 0)
    # Gremlin collisions and wrong buttons are not floor-hazard cost and stay
    se, cf_se = data.extras['state_extras'], cf.extras['state_extras']
    np.testing.assert_allclose(cf_se['cost'] - cf_se['hazard_cost'], se['cost'] - se['hazard_cost'], atol=1e-5)

  def testSampledLayoutsKeepClear(self):
    env = self.env.unwrapped
    state = jax.jit(env.reset)(jax.random.PRNGKey(4))
    anchors = state.info['cf_anchors']
    layouts = jax.jit(jax.vmap(lambda k: env.sample_cf_hazard_positions(k, anchors)))(
        jax.random.split(jax.random.PRNGKey(5), 256))
    radii = env._hazard_keepouts[env._cf_hazard_indices]
    to_anchor = jnp.linalg.norm(layouts[:, :, None] - anchors[None, None], axis=-1)
    clear = to_anchor >= radii[None, :, None] + env._cf_anchor_keepouts[None, None]
    self.assertGreater(float(jnp.mean(jnp.all(clear, axis=-1))), 0.99)
    min_x, min_y, max_x, max_y = env._placement_extents
    self.assertTrue(bool(jnp.all((layouts[..., 0] >= min_x) & (layouts[..., 0] <= max_x)
                                 & (layouts[..., 1] >= min_y) & (layouts[..., 1] <= max_y))))

  def testOptimisticNeverCostlierThanRandom(self):
    _, data = self._unroll(16, num_envs=64)
    costs = {}
    for mode in ('random', 'optimistic'):
      relabel = counterfactual.make_button_relabel_fn(self.env, mode=mode, num_candidates=8)
      _, metrics = jax.jit(relabel)(data, jax.random.PRNGKey(5))
      costs[mode] = float(metrics['cf/hazard_cost'])
    self.assertGreater(costs['random'], 0)
    self.assertLess(costs['optimistic'], costs['random'])


class GridCounterfactualTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.env = envs.get_environment('safe_grid', level=3)
    # Short episodes so the unroll crosses auto-resets
    self.wrapped = envs.training.wrap(self.env, episode_length=15, action_repeat=1)

  def _unroll(self, num_steps, num_envs=2):
    state = jax.jit(self.wrapped.reset)(jax.random.split(jax.random.PRNGKey(0), num_envs))
    unroll = jax.jit(
        lambda s, k: acting.generate_unroll(
            self.wrapped, s, _random_policy(self.env.action_size), k, num_steps,
            extra_fields=('truncation', 'cost') + counterfactual.GRID_CF_EXTRA_FIELDS))
    _, data = unroll(state, jax.random.PRNGKey(1))
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
    return state, data

  def testRelabelWithRealLayoutIsIdentity(self):
    state, data = self._unroll(40)
    self.assertGreater(float(jnp.sum(data.extras['state_extras']['truncation'])), 0, 'unroll should cross a reset')
    self.assertGreater(float(jnp.sum(data.extras['state_extras']['cost'])), 0, 'random walk should touch lava')
    lavas = state.pipeline_state.lava
    for i in range(lavas.shape[0]):
      segment = jax.tree_util.tree_map(lambda x: x[i:i + 1], data)
      # A fresh relabel_fn per layout: a reused one would replay the first trace
      relabel = counterfactual.make_grid_relabel_fn(self.env, mode='random')
      with mock.patch.object(type(self.env.unwrapped), 'sample_lava',
                             lambda env, key, keep_clear=None, lava=lavas[i]: lava):
        cf, _ = jax.jit(relabel)(segment, jax.random.PRNGKey(2))
      np.testing.assert_array_equal(cf.observation, segment.observation)
      np.testing.assert_array_equal(cf.next_observation, segment.next_observation)
      np.testing.assert_array_equal(cf.extras['state_extras']['cost'], segment.extras['state_extras']['cost'])

  def testOnlyLavaChanges(self):
    _, data = self._unroll(16, num_envs=4)
    env = self.env.unwrapped
    cf, _ = jax.jit(counterfactual.make_grid_relabel_fn(self.env, mode='random'))(data, jax.random.PRNGKey(3))
    window = (2 * env._view_radius + 1) ** 2
    lava = slice(4 + window, 4 + 2 * window)
    keep = np.ones(cf.observation.shape[-1], dtype=bool)
    keep[lava] = False
    np.testing.assert_array_equal(cf.observation[..., keep], data.observation[..., keep])
    self.assertGreater(float(jnp.mean(jnp.abs(cf.observation[..., lava] - data.observation[..., lava]))), 0)
    # Goals are never on lava
    pos, goal = env.decode_cells(cf.observation)
    on_goal = jnp.all(pos == goal, axis=-1)
    center = 4 + window + window // 2
    self.assertTrue(bool(jnp.all(jnp.where(on_goal, cf.observation[..., center] == 0, True))))

  def testOptimisticNeverCostlierThanRandom(self):
    _, data = self._unroll(16, num_envs=64)
    costs = {}
    for mode in ('random', 'optimistic'):
      relabel = counterfactual.make_grid_relabel_fn(self.env, mode=mode, num_candidates=8)
      _, metrics = jax.jit(relabel)(data, jax.random.PRNGKey(5))
      costs[mode] = float(metrics['cf/hazard_cost'])
    self.assertGreater(costs['random'], 0)
    self.assertLess(costs['optimistic'], costs['random'])


class LayoutResampleTest(absltest.TestCase):

  def _unroll(self, resample_layouts, num_envs=3, num_steps=40):
    env = envs.get_environment('safe_reacher', level=3)
    wrapped = envs.training.wrap(env, episode_length=15, action_repeat=1, resample_layouts=resample_layouts)
    state = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(0), num_envs))
    unroll = jax.jit(
        lambda s, k: acting.generate_unroll(
            wrapped, s, _random_policy(env.action_size), k, num_steps,
            extra_fields=('truncation', 'cost', 'hazard_positions') + counterfactual.REACHER_CF_EXTRA_FIELDS))
    final_state, data = unroll(state, jax.random.PRNGKey(1))
    return env.unwrapped, state, final_state, data

  def testFixedLayoutsByDefault(self):
    _, state, _, data = self._unroll(False)
    layouts = data.extras['state_extras']['hazard_positions']  # [T, B, H, 3]
    np.testing.assert_array_equal(layouts, jnp.broadcast_to(state.info['hazard_positions'], layouts.shape))

  def testFreshLayoutPerEpisode(self):
    env, state, final_state, data = self._unroll(True)
    se = data.extras['state_extras']
    layouts, done = se['hazard_positions'], se['truncation'] > 0  # [T, B, H, 3], [T, B]
    # Constant within an episode and new at every reset (steps 14 and 29 end episodes); the
    # reset step's info already holds the next episode's layout
    np.testing.assert_array_equal(done[:, 0], (jnp.arange(40) % 15) == 14)
    for t in range(1, 40):
      changed = jnp.any(layouts[t] != layouts[t - 1], axis=(-2, -1))
      np.testing.assert_array_equal(changed, done[t])
    # The goal stays, and the layout keeps clear of it
    goal = final_state.pipeline_state.mocap_pos[:, env._target_mocap_id, :2]
    np.testing.assert_array_equal(goal, state.pipeline_state.mocap_pos[:, env._target_mocap_id, :2])
    self.assertTrue(bool(jnp.all(jnp.linalg.norm(final_state.info['hazard_positions'][..., :2] - goal[:, None], axis=-1)
                                 > env._goal_radius)))
    # Observation and cost come from the current layout
    np.testing.assert_allclose(
        final_state.obs, jax.vmap(env._get_obs)(final_state.pipeline_state, final_state.info['hazard_positions']),
        atol=1e-6)
    # The cost of a step uses the layout it was taken in, which a reset step's info already replaced
    cost = jax.vmap(jax.vmap(env.hazard_cost_from_angles))(se['cf_theta'], layouts)
    np.testing.assert_allclose(jnp.where(done, 0, se['cost']), jnp.where(done, 0, cost), atol=1e-6)


if __name__ == '__main__':
  absltest.main()
