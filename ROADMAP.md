# CRAX Roadmap

This document collects the directions we intend to take CRAX in, along with some 
shortcomings. It is a statement of intent rather than a schedule.

Status legend: 🟢 in progress · 🟡 planned · 🔵 exploratory

---

## 1. Environment coverage

The task–agent matrix in the README describes the intended scope of the
benchmark, but the registry in `crax/envs/__init__.py` currently exposes only a
subset of those combinations. Every suite already has an abstract base class
(`SafeGoal`, `SafeCircle`, `SafeButton`, `SafePush`, `SafeHeight`,
`SafePathway`) with the task logic agent-agnostic, so the remaining work is
mostly per-agent configuration: spawn poses, observation layout, contact
geoms and difficulty tuning.

| Suite | Registered today | Missing | Status |
|---|---|---|---|
| Goal | Point | Ant, Spider, Humanoid, Swimmer | 🟢 |
| Circle | Point | Ant, Spider, Humanoid, Swimmer | 🟡 |
| Button | Point | Ant, Spider, Humanoid, Swimmer | 🟡 |
| Push | Point | Ant, Spider, Humanoid, Swimmer | 🟡 |
| Height | Humanoid | HalfCheetah, Hopper, Walker2D | 🟡 |
| Pathway | Walker2D | HalfCheetah, Hopper | 🟡 |
| Lift | Ant, Humanoid, Spider | — | ✅ |
| Velocity | 6 agents | — | ✅ |
| Reacher | Reacher | — | ✅ |

Related work:

- **Difficulty calibration.** Each new `(task, agent)` pair needs level 1/2/3
  entries in `crax/envs/difficulty.py`, tuned such that an unconstrained PPO agent
  meaningfully violates the constraint at every level. For the Velocity suite
  the thresholds are derived from unconstrained baselines
  (`DEFAULT_THRESHOLDS` in `crax/envs/safe_velocity.py`). We want the same
  data-driven procedure applied across all suites, and documented.
- **Multi-constraint tasks.** 🔵 Today each environment exposes a single scalar
  `cost`. Many realistic problems have several heterogeneous constraints with
  separate budgets. We would like a vector-valued cost channel plus algorithms
  that accept per-constraint bounds.
- **New task families.** 🔵 Candidates: object manipulation with fragile
  objects, constrained multi-agent navigation, and tasks where the constraint
  is on the *state distribution* (e.g. staying inside a region) rather than a
  per-step indicator.

## 2. Algorithms

The current suite (see the README table) covers first-order Lagrangian,
penalty, and rectification methods on PPO plus a small SAC family.

- 🟡 **Second-order / trust-region methods**: CPO, PCPO, TRPO-Lagrangian.
  These are the standard points of comparison in the Safe RL literature and are
  the most-requested gap. The trainer hooks (`loss_fn`, `post_step_fn`,
  `init_aux_state_fn`) do not yet accommodate a constrained line search, so this
  also requires a small extension of `training/agents/ppo/train.py`.
- 🟡 **More off-policy baselines**: DDPG/TD3 with Lagrangian and PID variants,
  to complement `sac_lag` and `sac_pid`.
- 🔵 **Safety-critic and shielding methods**: methods that act on the policy at
  execution time rather than through the objective.

## 3. Offlined dataset generation script

- 🔵 **Offline safe RL**: a data-collection script plus offline baselines that generates hdf5 files (reusing the existing checkpointing and replay-buffer code). Reference: https://github.com/liuzuxin/DSRL

## 4. Safe multi-agent (SafeMARL)

- 🔵 Non-stationary multi-agent safety (like car driving)
- 🔵 Decentralized SafeMARL
- 🔵 Compeititve SafeMARL

Reference: https://arxiv.org/pdf/2505.17342v2

<!-- ## 5. Robust SafeRL -->

## 5. Observations and vision

GPU pixel observations already work through `GpuPixelObservationWrapper`
(`crax/envs/wrappers/pixel_observation_gpu.py`), documented in
[docs/VISION_TRAINING.md](docs/VISION_TRAINING.md).

- 🟢 Extend vision support to the remaining suites and verify camera placement
  per agent (`scripts/check_vision_cameras.py`).
- 🟡 **Lidar / pseudo-lidar observations** matching the Safety-Gym convention, so
  results transfer between benchmarks without changing the policy architecture.
- 🟡 **Frame stacking and recurrent policies** for partially observable variants.
- 🔵 **Depth and segmentation channels** as alternatives to RGB.

## 6. Benchmarking and reproducibility

- 🟢 **Reference results.** The download/aggregation pipeline lives in
  `results/` (see [results/README.md](results/README.md)). We want a published
  set of baseline curves for every `(env, level, algorithm)` cell, with seed
  counts and confidence intervals, so new methods can be compared without
  re-running the baselines.

## 6. Performance

- 🟢 Continued MuJoCo Warp integration for both physics and rendering.
- 🟡 **Hasard overhead.** Hasard-heavy environments become very slow. Investigate 
  ways to reduce the overhead of Hasard's safety checks, e.g. by caching or vectorizing.
- 🔵 TPU support verification. The code is TPU-compatible in principle but has
  not been tested.

## 7. Testing and CI

This is the largest engineering gap and the easiest place to contribute.

- 🟡 **Environment smoke tests**: for every registered environment and level,
  assert that `reset`/`step` jit-compile, that shapes are stable, that `cost`
  appears in both `state.info` and `state.metrics`, and that rollouts contain no
  NaNs.
- 🟡 **Determinism tests**: identical seeds produce identical trajectories.
- 🟡 **Short training tests**: a few thousand steps per algorithm, asserting the
  loss is finite and the Lagrange multiplier updates in the expected direction.
- 🟡 **GitHub Actions workflow** running the CPU-only subset on every pull
  request; the repository has no `.github/` directory yet.

## 8. Documentation and examples

- 🟡 **Per-suite documentation** describing each task's reward, cost, observation
  space and difficulty levels — the level of detail currently only available for
  the Velocity suite by reading the source.
- 🟡 **API reference** built from the docstrings.
- 🟡 **Contribution guide** (`CONTRIBUTING.md`) with the environment-authoring
  walkthrough: subclass the suite base, add difficulty entries, register in
  `crax/envs/__init__.py`, add tests.

## 9. Ecosystem and packaging

- 🟡 **Gymnasium / Safety-Gymnasium API parity.** Wrappers exist
  (`crax/envs/wrappers/gym.py`, `dm_env.py`, `torch.py`); we want them tested
  against the current Gymnasium API and documented, so CRAX environments drop
  into existing training code.
- 🟡 **OmniSafe interoperability.** The comparison tooling already exists in
  `results/download/omnisafe.py`; the environment side should be a supported,
  tested path rather than benchmark-only glue.
- 🟡 **PyPI release.** The package is at version `0.0.1` and installs from
  source only. A tagged release with pinned, tested dependency ranges is a
  prerequisite for citing a specific version.

---

## Contributing

Contributions are welcome, and this roadmap is the best place to find work that
we know is needed. Good entry points, roughly in order of how self-contained
they are:

1. **Tests** (§6) — no GPU required, and the acceptance criteria are clear.
2. **New `(task, agent)` environments** (§1) — the suite base classes do the
   task logic; you add agent configuration, difficulty entries and a registry
   line.
3. **Example scripts and documentation** (§7).
4. **New algorithms** (§2) — implement against the existing trainer hooks;
   see the architecture section of the [README](README.md#architecture).

Before starting something substantial, please open an issue describing the
approach so we can avoid duplicated work — especially for items marked 🔵,
where the design is not settled. Bug reports and reproductions are equally
useful: include the environment name, difficulty level, algorithm, and the
command you ran.
