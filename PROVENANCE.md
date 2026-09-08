# Provenance

CRAX is a fork of [Brax](https://github.com/google/brax) 0.12.3, used under the Apache License 2.0. This file records which parts of the source tree come from Brax and which are CRAX's own, so that the distinction is visible without reading diffs.

**This file is generated.** Run `python scripts/check_provenance.py` to regenerate it. Do not edit it by hand.

## Summary

| Origin | Files | Lines of code | Share |
|---|---:|---:|---:|
| Verbatim Brax | 55 | 4,535 | 19% |
| Brax, small changes | 18 | 2,577 | 11% |
| Brax, substantially modified | 15 | 4,072 | 17% |
| CRAX original | 56 | 12,110 | 52% |
| **Total** | **144** | **23,294** | |

Lines of code exclude comments and blank lines. Files are compared against the upstream Brax source with the CRAX renamings undone, so a file that differs only because `brax` was renamed to `crax`, or because `brax.training` became the top-level `training` package, counts as verbatim.

## Why CRAX vendors Brax rather than depending on it

Brax is vendored rather than imported for two reasons:

1. **Dependency weight.** Brax's own install pulls in flask, flask_cors, jinja2, grpcio, gym, pytinyrenderer, scipy, optax and orbax-checkpoint. Depending on it would put all of those in CRAX's base install, when the point of the `crax` package is that it can be installed without an RL stack at all.
2. **Changes inside Brax internals.** Several modifications cannot be expressed as subclasses: `crax/io/mjcf.py` loads collision functions from `crax.mjx.collisions`, `crax/mjx/pipeline.py` changes the signature of `_reformat_contact`, and `crax/envs/base.py` alters `PipelineEnv` construction and `Wrapper.render`.

Every file derived from Brax keeps the original copyright header and carries a notice describing how it was changed, as required by section 4(b) of the Apache License. See also [NOTICE](NOTICE).

## Brax, substantially modified (15 files)

Files taken from Brax and substantially modified. Each carries a header notice describing the changes.

| File | LOC | Changed lines |
|---|---:|---:|
| `training/networks.py` | 642 | 328 |
| `training/agents/ppo/train.py` | 758 | 294 |
| `crax/envs/__init__.py` | 214 | 156 |
| `training/agents/sac/train.py` | 610 | 136 |
| `training/replay_buffers.py` | 442 | 93 |
| `training/agents/ppo/networks_vision.py` | 128 | 88 |
| `training/logger.py` | 91 | 80 |
| `training/agents/ppo/losses.py` | 155 | 58 |
| `training/checkpoint.py` | 134 | 47 |
| `training/agents/sac/networks.py` | 107 | 42 |
| `training/agents/ppo/networks.py` | 107 | 37 |
| `crax/mjx/pipeline.py` | 80 | 19 |
| `crax/io/mjcf.py` | 433 | 18 |
| `crax/envs/fast.py` | 113 | 14 |
| `training/agents/ppo/checkpoint.py` | 58 | 13 |

## Brax, small changes (18 files)

Files taken from Brax with small changes (at most 12 lines). Each carries a header notice.

| File | LOC | Changed lines |
|---|---:|---:|
| `training/agents/sac/checkpoint.py` | 58 | 12 |
| `crax/envs/base.py` | 159 | 11 |
| `crax/envs/swimmer.py` | 138 | 9 |
| `crax/envs/walker2d.py` | 187 | 9 |
| `crax/envs/wrappers/gym.py` | 107 | 9 |
| `training/__init__.py` | 7 | 7 |
| `training/agents/ppo/__init__.py` | 7 | 7 |
| `training/agents/ppo/train_test.py` | 219 | 4 |
| `training/types.py` | 121 | 4 |
| `crax/envs/ant.py` | 209 | 3 |
| `crax/envs/half_cheetah.py` | 146 | 3 |
| `crax/envs/hopper.py` | 201 | 3 |
| `crax/envs/humanoid.py` | 278 | 3 |
| `crax/envs/reacher.py` | 164 | 3 |
| `crax/__init__.py` | 7 | 2 |
| `crax/envs/wrappers/training.py` | 193 | 2 |
| `crax/envs/humanoidstandup.py` | 216 | 1 |
| `crax/envs/pusher.py` | 160 | 1 |

## Verbatim Brax (55 files)

Files used unchanged from Brax, apart from the package rename. These keep their original Brax copyright header.

| File | LOC |
|---|---:|
| `crax/actuator.py` | 30 |
| `crax/base.py` | 461 |
| `crax/com.py` | 31 |
| `crax/contact.py` | 39 |
| `crax/envs/inverted_double_pendulum.py` | 131 |
| `crax/envs/inverted_pendulum.py` | 93 |
| `crax/envs/wrappers/__init__.py` | 0 |
| `crax/envs/wrappers/dm_env.py` | 90 |
| `crax/envs/wrappers/torch.py` | 23 |
| `crax/fluid.py` | 49 |
| `crax/generalized/__init__.py` | 0 |
| `crax/generalized/base.py` | 69 |
| `crax/generalized/constraint.py` | 157 |
| `crax/generalized/dynamics.py` | 150 |
| `crax/generalized/integrator.py` | 53 |
| `crax/generalized/mass.py` | 60 |
| `crax/generalized/pipeline.py` | 64 |
| `crax/io/__init__.py` | 0 |
| `crax/io/image.py` | 55 |
| `crax/io/json.py` | 112 |
| `crax/io/torch.py` | 67 |
| `crax/kinematics.py` | 274 |
| `crax/math.py` | 301 |
| `crax/mjx/__init__.py` | 0 |
| `crax/mjx/base.py` | 5 |
| `crax/positional/__init__.py` | 0 |
| `crax/positional/base.py` | 25 |
| `crax/positional/collisions.py` | 173 |
| `crax/positional/integrator.py` | 68 |
| `crax/positional/joints.py` | 209 |
| `crax/positional/pipeline.py` | 84 |
| `crax/scan.py` | 141 |
| `crax/spring/__init__.py` | 0 |
| `crax/spring/base.py` | 27 |
| `crax/spring/collisions.py` | 62 |
| `crax/spring/integrator.py` | 36 |
| `crax/spring/joints.py` | 246 |
| `crax/spring/perf_test.py` | 19 |
| `crax/spring/pipeline.py` | 96 |
| `crax/test_utils.py` | 100 |
| `training/acme/__init__.py` | 0 |
| `training/acme/running_statistics.py` | 169 |
| `training/acme/specs.py` | 14 |
| `training/acme/types.py` | 15 |
| `training/acting.py` | 123 |
| `training/agents/__init__.py` | 0 |
| `training/agents/ppo/checkpoint_test.py` | 75 |
| `training/agents/sac/__init__.py` | 0 |
| `training/agents/sac/checkpoint_test.py` | 71 |
| `training/agents/sac/losses.py` | 101 |
| `training/agents/sac/train_test.py` | 78 |
| `training/distribution.py` | 108 |
| `training/gradients.py` | 41 |
| `training/pmap.py` | 40 |
| `training/spectral_norm.py` | 100 |

## CRAX original (56 files)

Files written for CRAX, with no Brax counterpart.

| File | LOC |
|---|---:|
| `crax/envs/builder.py` | 377 |
| `crax/envs/difficulty.py` | 229 |
| `crax/envs/env_utils.py` | 551 |
| `crax/envs/goals.py` | 215 |
| `crax/envs/hazards.py` | 387 |
| `crax/envs/humanoid_hop.py` | 204 |
| `crax/envs/humanoid_hop_airtime.py` | 112 |
| `crax/envs/safe_ant.py` | 42 |
| `crax/envs/safe_button.py` | 756 |
| `crax/envs/safe_circle.py` | 648 |
| `crax/envs/safe_goal.py` | 722 |
| `crax/envs/safe_height.py` | 330 |
| `crax/envs/safe_lift.py` | 430 |
| `crax/envs/safe_pathway.py` | 443 |
| `crax/envs/safe_push.py` | 869 |
| `crax/envs/safe_reacher.py` | 345 |
| `crax/envs/safe_spider.py` | 122 |
| `crax/envs/safe_velocity.py` | 328 |
| `crax/envs/velocity_constraints.py` | 73 |
| `crax/envs/wrappers/pixel_observation.py` | 310 |
| `crax/envs/wrappers/pixel_observation_gpu.py` | 225 |
| `crax/envs/wrappers/saute.py` | 71 |
| `crax/mjx/collisions.py` | 160 |
| `training/agents/crpo/__init__.py` | 5 |
| `training/agents/crpo/losses.py` | 138 |
| `training/agents/crpo/train.py` | 197 |
| `training/agents/focops/__init__.py` | 8 |
| `training/agents/focops/checkpoint.py` | 9 |
| `training/agents/focops/losses.py` | 129 |
| `training/agents/focops/networks.py` | 3 |
| `training/agents/focops/train.py` | 113 |
| `training/agents/p3o/__init__.py` | 8 |
| `training/agents/p3o/checkpoint.py` | 9 |
| `training/agents/p3o/losses.py` | 150 |
| `training/agents/p3o/networks.py` | 3 |
| `training/agents/p3o/train.py` | 118 |
| `training/agents/ppo/ppo_cost.py` | 112 |
| `training/agents/ppo_lag/__init__.py` | 13 |
| `training/agents/ppo_lag/losses.py` | 103 |
| `training/agents/ppo_lag/networks.py` | 7 |
| `training/agents/ppo_lag/train.py` | 194 |
| `training/agents/ppo_pid/__init__.py` | 3 |
| `training/agents/ppo_pid/losses.py` | 8 |
| `training/agents/ppo_pid/train.py` | 115 |
| `training/agents/ppo_saute/__init__.py` | 1 |
| `training/agents/ppo_saute/train.py` | 96 |
| `training/agents/sac_lag/__init__.py` | 2 |
| `training/agents/sac_lag/losses.py` | 148 |
| `training/agents/sac_lag/networks.py` | 133 |
| `training/agents/sac_lag/train.py` | 694 |
| `training/agents/sac_pid/__init__.py` | 1 |
| `training/agents/sac_pid/train.py` | 120 |
| `training/config.py` | 184 |
| `training/curriculum.py` | 157 |
| `training/run_utils.py` | 772 |
| `training/transfer.py` | 408 |

