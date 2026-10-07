"""
Population-based hyperparameter scheduling for PPO: MF-PBT, PBT or random search.

Implements Multiple-Frequencies PBT (Doulazmi et al., 2025,
https://arxiv.org/abs/2506.03225; reference code https://github.com/WaelDLZ/MF-PBT)
on top of CRAX's PPO. A population of agents is trained in rounds of
`--pbt_steps_per_round` env steps; after each round every agent is scored and
the evolution step (training/pbt/genetics.py) decides which agents copy the
weights and (perturbed) hyperparameters of better ones. The result is a
per-agent hyperparameter *schedule*, not a single fixed setting.

Multi-GPU: start one process per GPU with the same arguments plus
`--pbt_worker_id i --pbt_num_workers N` (see scripts/snellius/). Workers own
agents `i, i+N, i+2N, ...` and synchronise through `--pbt_dir`. Re-running
the same command resumes from the last completed round.

Example (single GPU, small smoke test):
    python -m training.train_pbt --env_name safe_velocity_ant --alg ppo --vision \
        --vision_limb_colors --pbt_num_agents 4 --pbt_frequencies 1 \
        --num_envs 64 --batch_size 64 --num_minibatches 4 --num_eval_envs 8 \
        --num_timesteps 2e5 --pbt_steps_per_round 2e4 --use_wandb false
"""

import functools
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import jax
import numpy as np

import wandb
from crax import envs
from crax.envs.limb_colors import colorize_env_limbs
from training.config import build_base_parser, bool_type, _json_type
from training.pbt import genetics
from training.pbt.ppo_population import DYNAMIC_HPARAMS, PPOPopulationTrainer
from training.pbt.store import PopulationStore
from training.run_utils import (
    setup_gpu_environment, make_vision_network_factory, morphology_override, VISION_CAMERA_OVERRIDES,
    make_periodic_vision_video_fn,
)

PBT_ALGORITHMS = ('mfpbt', 'pbt', 'random_search')


def add_pbt_args(parser):
    g = parser.add_argument_group('Population-based training')
    g.add_argument('--pbt_algorithm', choices=PBT_ALGORITHMS, default='mfpbt',
                   help="'mfpbt': Multiple-Frequencies PBT; 'pbt': standard PBT (one population evolving "
                        "every round, 25%% truncation); 'random_search': no evolution (independent agents "
                        "with randomly drawn, fixed hyperparameters).")
    g.add_argument('--pbt_num_agents', type=int, default=16,
                   help='Population size. MF-PBT needs num_agents / len(frequencies) to be a multiple of 4.')
    g.add_argument('--pbt_frequencies', type=int, nargs='+', default=[1, 5, 10, 20],
                   help='MF-PBT: evolution period (in rounds) of each sub-population.')
    g.add_argument('--pbt_steps_per_round', type=float, default=5e6,
                   help="Env steps each agent trains between evolution steps (t_ready in the paper). "
                        "Rounded up to a whole number of PPO training steps.")
    g.add_argument('--pbt_num_rounds', type=int, default=None,
                   help='Number of rounds; default: ceil(num_timesteps / steps_per_round), i.e. '
                        '--num_timesteps is the per-agent budget.')
    g.add_argument('--pbt_hparams', type=str, nargs='+',
                   default=['learning_rate', 'entropy_cost', 'clipping_epsilon'],
                   help=f'Hyperparameters to search/schedule (subset of {list(DYNAMIC_HPARAMS)}). '
                        f'The others stay at their --<name> value.')
    g.add_argument('--pbt_search_space', type=_json_type, default=None,
                   help='JSON overrides of the default search space (training/pbt/genetics.py), e.g. '
                        '\'{"learning_rate": {"init_low": 3e-5, "init_high": 3e-4}}\'.')
    g.add_argument('--pbt_perturb_factors', type=float, nargs='+', default=[0.8, 1.25],
                   help='Explore step: multiply each copied hyperparameter by a factor drawn from these.')
    g.add_argument('--pbt_explore', type=bool_type, nargs='?', const=True, default=True,
                   help="False = 'variance exploitation' control: evolution only selects/copies agents, "
                        "hyperparameters are never perturbed.")
    g.add_argument('--pbt_fitness_key', type=str, default='eval/episode_reward',
                   help="Metric to rank agents by. 'eval/...' keys run an evaluation (--num_eval_envs "
                        "episodes) after every round; 'episodic/...' keys use the training episodes "
                        "finished during the round (free, but noisier and lagging).")
    g.add_argument('--pbt_dir', type=str, default='runs/pbt',
                   help='Shared directory for agent checkpoints, worker barriers and resume state.')
    g.add_argument('--pbt_exp_name', type=str, default=None,
                   help='Experiment name (subdirectory of --pbt_dir, wandb run id). Reusing a name resumes it.')
    g.add_argument('--pbt_worker_id', type=int, default=0, help='Index of this worker process.')
    g.add_argument('--pbt_num_workers', type=int, default=1, help='Number of worker processes (GPUs).')
    g.add_argument('--pbt_launch_id', type=str, default=default_launch_id(),
                   help='Shared by all workers of one launch (default: $SLURM_JOB_ID_$SLURM_RESTART_COUNT); keeps '
                        'round barriers from counting markers left by a crashed earlier attempt.')
    g.add_argument('--pbt_barrier_timeout', type=float, default=4 * 3600,
                   help='Seconds to wait for the other workers to finish a round before aborting.')
    g.add_argument('--pbt_video_every_rounds', type=int, default=10,
                   help='--vision only: log a clip of the current best agent every this many rounds (0 = off).')
    g.add_argument('--pbt_table_every_rounds', type=int, default=10,
                   help='Log the full per-agent table (hyperparameters, fitness, lineage) every this many rounds.')
    return parser


def default_launch_id() -> Optional[str]:
    job = os.environ.get('SLURM_JOB_ID')
    # A requeued SLURM job keeps its id, so include the restart count.
    return f"{job}_{os.environ.get('SLURM_RESTART_COUNT', '0')}" if job else None


def default_exp_name(config) -> str:
    name = f"{config.pbt_algorithm}_{config.env_name}_L{config.difficulty}_{config.alg}"
    if config.vision:
        name += f"_vision_{config.vision_camera}"
    return f"{name}_n{config.pbt_num_agents}_seed{config.seeds[0]}"


def frequencies_for(config) -> List[int]:
    if config.pbt_algorithm == 'mfpbt':
        return list(config.pbt_frequencies)
    return [1]


def build_envs(config):
    """Training/eval envs and vision kwargs, set up the same way as training/train_env.py."""
    env_name = config.env_name
    env_kwargs = dict(config.env_kwargs or {})
    if env_name == 'safe_velocity':
        env_kwargs['agent'] = config.agent
    vision_kwargs = None
    if config.vision:
        env_kwargs.setdefault('backend', 'mjx')
        vision_kwargs = dict(
            camera=config.vision_camera,
            height=config.vision_height,
            width=config.vision_width,
            obs_mode=config.vision_obs_mode,
            frame_stack=config.vision_frame_stack,
        )
    env = envs.get_environment(env_name, level=config.difficulty, **env_kwargs)
    eval_env = envs.get_environment(env_name, level=config.difficulty, **env_kwargs)
    if config.vision and config.vision_limb_colors:
        colorize_env_limbs(env, env_name)
        colorize_env_limbs(eval_env, env_name)
    episode_length = config.episode_length or env_kwargs.get('episode_length') or getattr(env, 'episode_length', None)
    return env, eval_env, env_kwargs, vision_kwargs, episode_length


def build_video_fn(config, env_kwargs, vision_kwargs, episode_length, run_name):
    if not config.vision or config.skip_video or config.pbt_video_every_rounds <= 0:
        return None
    video_env_kwargs = {k: v for k, v in env_kwargs.items() if k != 'episode_length'}
    use_limb_colors = config.vision_limb_colors
    video_env = envs.create(
        config.env_name, level=config.difficulty,
        episode_length=episode_length,
        auto_reset=True,
        batch_size=1,
        vision=True,
        vision_kwargs=dict(**vision_kwargs, num_envs=1),
        pre_vision_fn=(lambda e: colorize_env_limbs(e, config.env_name)) if use_limb_colors else None,
        **video_env_kwargs,
    )
    return make_periodic_vision_video_fn(
        video_env,
        every_steps=1,  # cadence is controlled by the round loop (force=True)
        steps=config.periodic_video_steps,
        num_episodes=config.num_video_episodes,
        pixel_camera=config.vision_camera,
        extra_cameras=[c for c in config.cameras if c != config.vision_camera],
        frame_stack=config.vision_frame_stack,
        width=config.video_width,
        height=config.video_height,
        fps=config.video_fps,
        run_name=run_name,
        deterministic=config.deterministic_eval,
        log_to_wandb=config.use_wandb,
        seed=config.seeds[0],
    )


def compute_fitness(metrics: Dict[str, float], key: str) -> float:
    value = metrics.get(key)
    if value is None or not np.isfinite(value):
        return float('-inf')
    return float(value)


def initial_hparams(config, search_space, num_agents: int, seed: int) -> List[Dict[str, float]]:
    """Fixed hyperparameters from the CLI; searched ones drawn per agent (identically on every worker)."""
    base = {h: float(getattr(config, h)) for h in DYNAMIC_HPARAMS}
    rng = np.random.default_rng(seed)
    draws = {name: spec.sample(rng, num_agents) for name, spec in search_space.items()}
    return [{**base, **{name: float(draws[name][a]) for name in search_space}} for a in range(num_agents)]


def summarise_population(
        summaries: List[Dict[str, Any]],
        frequencies: List[int],
        search_space: Dict[str, genetics.HyperparameterSpec],
        inheritance: Optional[genetics.Inheritance],
) -> Dict[str, float]:
    """Flat dict of population-level metrics for wandb (computed on worker 0)."""
    n = len(summaries)
    fitness = np.array([s['fitness'] if s['fitness'] is not None else -np.inf for s in summaries], dtype=np.float64)
    finite = np.isfinite(fitness)
    best = int(np.argmax(fitness))
    log: Dict[str, float] = {}

    f = fitness[finite] if finite.any() else np.array([np.nan])
    log.update({
        'fitness/best': float(np.max(f)), 'fitness/mean': float(np.mean(f)),
        'fitness/median': float(np.median(f)), 'fitness/worst': float(np.min(f)),
        'fitness/std': float(np.std(f)), 'fitness/num_diverged': int(n - finite.sum()),
        'pbt/best_agent': best,
    })
    per_pop = n // len(frequencies)
    for p, freq in enumerate(frequencies):
        pf = fitness[p * per_pop:(p + 1) * per_pop]
        pf = pf[np.isfinite(pf)]
        if pf.size:
            log[f'fitness_by_freq/f{freq}_best'] = float(pf.max())
            log[f'fitness_by_freq/f{freq}_mean'] = float(pf.mean())

    for a, s in enumerate(summaries):
        log[f'agents/fitness/agent_{a:02d}'] = s['fitness'] if s['fitness'] is not None else float('nan')
        for name in search_space:
            log[f'agents/{name}/agent_{a:02d}'] = s['hparams'][name]

    for name, spec in search_space.items():
        values = np.array([s['hparams'][name] for s in summaries])
        coords = np.array([spec.to_coord(v) for v in values])
        log[f'hparams/best/{name}'] = float(values[best])
        log[f'hparams/geomean/{name}'] = float(spec.from_coord(np.exp(np.mean(np.log(coords)))))
        log[f'hparams/min/{name}'] = float(values.min())
        log[f'hparams/max/{name}'] = float(values.max())
        # Diversity left in the population (collapse to ~0 means the search has converged).
        log[f'hparams/log_spread/{name}'] = float(np.std(np.log(coords)))

    # Best agent's own metrics, both prefixed and under the plain names used by single-agent
    # runs (eval/episode_forward_reward, episodic/..., training/...) so they plot side by side.
    for key, value in summaries[best]['metrics'].items():
        log[f'best/{key}'] = value
        log[key] = value
    # Population means of the headline metrics.
    for key in summaries[best]['metrics']:
        if key.startswith(('eval/episode_', 'eval/avg_episode_length', 'episodic/')) and not key.endswith('_std'):
            vals = [s['metrics'].get(key) for s in summaries]
            vals = [v for v in vals if v is not None and np.isfinite(v)]
            if vals:
                log[f'population/{key}'] = float(np.mean(vals))

    if inheritance is not None:
        log.update({f'evolution/{k}': v for k, v in inheritance.summary().items()})
        for p, freq in enumerate(frequencies):
            log[f'evolution/evolved_f{freq}'] = int(inheritance.evolving_populations[p])
    return log


def agent_table(round_index, summaries, search_space, inheritance):
    columns = ['round', 'agent', 'population', 'fitness'] + list(search_space) + [
        'parent_network', 'parent_hps', 'explored']
    table = wandb.Table(columns=columns)
    for a, s in enumerate(summaries):
        table.add_data(
            round_index, a, s['population'], s['fitness'],
            *[s['hparams'][h] for h in search_space],
            int(inheritance.parent_network[a]) if inheritance else a,
            int(inheritance.parent_hps[a]) if inheritance else a,
            bool(inheritance.need_explore[a]) if inheritance else False,
        )
    return table


def main():
    parser = add_pbt_args(build_base_parser(description='Population-based training (MF-PBT) for CRAX PPO'))
    config = parser.parse_args()

    if config.alg != 'ppo':
        raise ValueError(f"train_pbt currently supports --alg ppo only (got '{config.alg}').")
    if config.vision_privilege_mode != 'none':
        raise ValueError("train_pbt does not support --vision_privilege_mode.")
    if config.vision_camera is None:
        config.vision_camera = morphology_override(config.env_name, VISION_CAMERA_OVERRIDES) or 'vision'

    seed = config.seeds[0]
    if len(config.seeds) > 1:
        print(f"train_pbt runs one population per invocation; using seed {seed} only.")
    worker_id, num_workers = config.pbt_worker_id, config.pbt_num_workers
    num_agents = config.pbt_num_agents
    frequencies = frequencies_for(config)
    genetics.check_population_layout(num_agents, frequencies)
    if not 0 <= worker_id < num_workers:
        raise ValueError(f"--pbt_worker_id must be in [0, {num_workers}), got {worker_id}")
    search_space = genetics.build_search_space(config.pbt_hparams, config.pbt_search_space)
    if config.pbt_fitness_key.startswith('eval/') and config.num_eval_envs <= 0:
        raise ValueError(f"--pbt_fitness_key {config.pbt_fitness_key} needs --num_eval_envs > 0")

    exp_name = config.pbt_exp_name or default_exp_name(config)
    is_main = worker_id == 0
    owned = list(range(worker_id, num_agents, num_workers))
    tag = f"[pbt worker {worker_id}/{num_workers}]"

    setup_gpu_environment(vision=config.vision)
    print(f"{tag} devices={jax.devices()} owns agents {owned}")

    env, eval_env, env_kwargs, vision_kwargs, episode_length = build_envs(config)
    network_factory = None
    if config.vision:
        state_obs_key = 'state' if config.vision_obs_mode == 'pixels+state' else ''
        network_factory = make_vision_network_factory(
            config.alg, policy_obs_key=state_obs_key, value_obs_key=state_obs_key,
        )
    trainer_kwargs = dict(network_factory=network_factory) if network_factory else {}
    trainer = PPOPopulationTrainer(
        env,
        steps_per_round=int(config.pbt_steps_per_round),
        episode_length=episode_length,
        num_envs=config.num_envs,
        unroll_length=config.unroll_length,
        batch_size=config.batch_size,
        num_minibatches=config.num_minibatches,
        num_updates_per_batch=config.num_updates_per_batch,
        eval_env=eval_env,
        num_eval_envs=config.num_eval_envs if config.pbt_fitness_key.startswith('eval/') else 0,
        deterministic_eval=config.deterministic_eval,
        normalize_observations=config.normalize_observations,
        max_grad_norm=config.max_grad_norm,
        augment_pixels=config.vision and config.vision_augment,
        vision_kwargs=vision_kwargs,
        seed=seed + 1000 * worker_id,
        **trainer_kwargs,
    )
    steps_per_round = trainer.steps_per_round
    num_rounds = config.pbt_num_rounds or int(np.ceil(config.num_timesteps / steps_per_round))
    print(f"{tag} {num_rounds} rounds x {steps_per_round} env steps per agent "
          f"({trainer.num_training_steps_per_round} PPO training steps/round), "
          f"population budget {num_rounds * steps_per_round * num_agents:.3e} env steps")

    if num_workers > 1 and not config.pbt_launch_id:
        print(f"{tag} WARNING: no --pbt_launch_id / $SLURM_JOB_ID; resuming after a crash may mix rounds.")
    store = PopulationStore(os.path.join(config.pbt_dir, exp_name), num_agents, num_workers, worker_id,
                            launch_id=config.pbt_launch_id)
    store.check_or_write_meta(dict(
        num_agents=num_agents, frequencies=frequencies,
        pbt_algorithm=config.pbt_algorithm, hparams=list(search_space), seed=seed,
        env_name=config.env_name, difficulty=config.difficulty, vision=config.vision,
        steps_per_round=steps_per_round, num_envs=config.num_envs,
    ))

    def population_of(agent: int) -> int:
        return agent // (num_agents // len(frequencies))

    def train_key(agent: int, round_index: int):
        return jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(seed), agent), round_index)

    def eval_key(round_index: int):
        # Shared by all agents of a round: everyone is scored from the same initial states.
        return jax.random.fold_in(jax.random.PRNGKey(seed + 7919), round_index)

    # --- Population state (owned agents only) ---------------------------------
    hparams = {a: hp for a, hp in enumerate(initial_hparams(config, search_space, num_agents, seed)) if a in owned}
    states = {}
    env_states = {}
    env_steps = {a: 0 for a in owned}

    def evolve(round_index: int, summaries: List[Dict[str, Any]]) -> Optional[genetics.Inheritance]:
        """Applies the evolution step after `round_index` to the owned agents."""
        if config.pbt_algorithm == 'random_search' or round_index >= num_rounds - 1:
            return None
        fitness = [s['fitness'] if s['fitness'] is not None else float('-inf') for s in summaries]
        inheritance = genetics.genetics(fitness, round_index, frequencies)
        for a in owned:
            parent_net = int(inheritance.parent_network[a])
            parent_hp = int(inheritance.parent_hps[a])
            if parent_net != a:
                states[a] = trainer.import_agent(store.load_state(round_index, parent_net), env_steps[a])
            if parent_hp != a:
                new_hp = {h: float(v) for h, v in summaries[parent_hp]['hparams'].items()}
                if inheritance.need_explore[a] and config.pbt_explore:
                    new_hp = genetics.explore(
                        new_hp, search_space, genetics.explore_rng(seed, round_index, a),
                        tuple(config.pbt_perturb_factors),
                    )
                hparams[a] = new_hp
            if parent_net != a or parent_hp != a:
                print(f"{tag} round {round_index}: agent {a} <- network {parent_net}, hparams {parent_hp}"
                      f"{' (explored)' if inheritance.need_explore[a] and config.pbt_explore else ''}")
        return inheritance

    last_round = store.latest_complete_round()
    if last_round is None:
        start_round = 0
        for a in owned:
            states[a] = trainer.init_agent(jax.random.fold_in(jax.random.PRNGKey(seed + 1), a))
    else:
        print(f"{tag} resuming {exp_name} after completed round {last_round}")
        summaries = store.load_summaries(last_round)
        for a in owned:
            env_steps[a] = int(summaries[a]['env_steps'])
            hparams[a] = {h: float(v) for h, v in summaries[a]['hparams'].items()}
            states[a] = trainer.import_agent(store.load_state(last_round, a), env_steps[a])
        evolve(last_round, summaries)
        start_round = last_round + 1
    for a in owned:
        env_states[a] = trainer.reset(jax.random.fold_in(jax.random.PRNGKey(seed + 2), a * 100003 + start_round))

    # --- Logging (worker 0) ----------------------------------------------------
    run_name = f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    video_fn = None
    ckpt_root = None
    if is_main:
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                name=exp_name,
                id=exp_name.replace('/', '_'),
                resume='allow',
                group=config.wandb_group or f"pbt_{config.env_name}",
                job_type=f"{config.pbt_algorithm}_{config.alg}",
                tags=(config.wandb_tags or []) + ['PBT', config.pbt_algorithm.upper()],
                config={
                    **vars(config), 'seed': seed, 'episode_length': episode_length,
                    'pbt_effective_frequencies': frequencies, 'pbt_num_rounds_effective': num_rounds,
                    'pbt_steps_per_round_effective': steps_per_round,
                    'pbt_search_space_effective': {k: v.__dict__ for k, v in search_space.items()},
                },
            )
            wandb.define_metric('pbt/round')
        video_fn = build_video_fn(config, env_kwargs, vision_kwargs, episode_length, run_name)
        if config.store_model:
            ckpt_root = Path(__file__).parent.parent.resolve() / config.model_dir / exp_name
            os.makedirs(ckpt_root, exist_ok=True)

    run_start = time.time()
    best_fitness_so_far = float('-inf')

    for round_index in range(start_round, num_rounds):
        round_start = time.time()
        for a in owned:
            states[a], env_states[a], metrics = trainer.train_round(
                states[a], env_states[a], hparams[a], train_key(a, round_index),
            )
            env_steps[a] += steps_per_round
            if config.pbt_fitness_key.startswith('eval/'):
                metrics.update(trainer.evaluate(states[a], eval_key(round_index)))
            fitness = compute_fitness(metrics, config.pbt_fitness_key)
            if not np.isfinite(fitness):
                print(f"{tag} WARNING: agent {a} has non-finite fitness in round {round_index} "
                      f"('{config.pbt_fitness_key}' = {metrics.get(config.pbt_fitness_key)}); ranking it last.")
            store.save_agent(
                round_index, a,
                summary=dict(
                    agent=a, population=population_of(a), round=round_index, env_steps=env_steps[a],
                    fitness=fitness, hparams=hparams[a], metrics=metrics,
                ),
                state=trainer.export_agent(states[a]),
            )
            print(f"{tag} round {round_index}/{num_rounds - 1} agent {a}: fitness={fitness:.4f} "
                  f"sps={metrics['training/sps']:.0f} "
                  + " ".join(f"{h}={hparams[a][h]:.3g}" for h in search_space))
        train_time = time.time() - round_start
        store.mark_done(round_index)
        wait_time = store.wait_for_round(round_index, timeout_s=config.pbt_barrier_timeout)

        summaries = store.load_summaries(round_index)
        inheritance = evolve(round_index, summaries)

        if is_main:
            step = (round_index + 1) * steps_per_round
            log = summarise_population(summaries, frequencies, search_space, inheritance)
            log.update({
                'pbt/round': round_index,
                'pbt/env_steps_per_agent': step,
                'pbt/population_env_steps': step * num_agents,
                'pbt/round_time': time.time() - round_start,
                'pbt/worker0_train_time': train_time,
                'pbt/worker0_barrier_wait': wait_time,
                'pbt/walltime': time.time() - run_start,
            })
            print(f"{tag} round {round_index}: best fitness {log['fitness/best']:.4f} (agent {log['pbt/best_agent']}), "
                  f"mean {log['fitness/mean']:.4f}, " + " ".join(
                      f"best_{h}={log[f'hparams/best/{h}']:.3g}" for h in search_space))
            store.append_history(dict(round=round_index, summaries=[
                {k: s[k] for k in ('agent', 'fitness', 'hparams')} for s in summaries
            ], inheritance=inheritance.summary() if inheritance else None))
            if config.use_wandb:
                last = round_index == num_rounds - 1
                if config.pbt_table_every_rounds > 0 and ((round_index + 1) % config.pbt_table_every_rounds == 0 or last):
                    log['pbt/agents_table'] = agent_table(round_index, summaries, search_space, inheritance)
                wandb.log(log, step=step)

            best = int(log['pbt/best_agent'])
            is_video_round = (round_index + 1) % max(config.pbt_video_every_rounds, 1) == 0
            last = round_index == num_rounds - 1
            improved = log['fitness/best'] > best_fitness_so_far
            if (video_fn is not None and (is_video_round or last)) or (ckpt_root is not None and (improved or last)):
                best_state = trainer.import_agent(store.load_state(round_index, best), step)
                best_params = trainer.policy_params(best_state)
                if ckpt_root is not None and (improved or last):
                    from training.agents.ppo import checkpoint
                    checkpoint.save(ckpt_root, step, best_params, trainer.network_config())
                if video_fn is not None and (is_video_round or last):
                    video_fn(step, trainer.make_policy, best_params, force=True)
            best_fitness_so_far = max(best_fitness_so_far, log['fitness/best'])
            # Everyone has loaded what they need from earlier rounds (they are past this round's
            # barrier), so only this round is needed to resume.
            store.cleanup_before(round_index)

    if is_main and config.use_wandb and wandb.run is not None:
        wandb.finish()
    print(f"{tag} done.")


if __name__ == '__main__':
    main()
