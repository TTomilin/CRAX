"""Evolution step of Multiple-Frequencies Population-Based Training (MF-PBT).

Port of the reference implementation (https://github.com/WaelDLZ/MF-PBT,
`hpo/utils/mf_pbt_genetics.py`; Doulazmi et al., 2025,
https://arxiv.org/abs/2506.03225). The population is split into
sub-populations that evolve at different frequencies (every `f` rounds):

  * internal exploit: in every sub-population that evolves this round, the
    bottom quarter copies the network and (perturbed) hyperparameters of the
    top quarter;
  * asymmetric migration: the third quarter of an evolving sub-population is
    replaced by better agents from the other sub-populations. The migrant's
    network is always taken; its hyperparameters only when it comes from a
    sub-population that evolves *less* often (i.e. the hyperparameters have
    proven themselves over a longer horizon).

With a single frequency `[1]` this reduces to standard PBT with 25%
truncation selection, so the same code path serves as the PBT baseline.

Everything here is pure numpy and deterministic given the inputs (plus the
explicit RNG passed to `explore`), so every worker process of a multi-GPU run
reaches the same decisions independently from the shared fitness table.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

POPULATION_QUARTERS = 4
MIGRATION_BRACKET_START = 2  # Which quarter of the local ranking may be replaced by migrants (0-indexed)
MIGRATION_BRACKET_END = 3


@dataclass
class Inheritance:
    """Per-agent outcome of one evolution step (indices are global agent ids)."""

    parent_network: np.ndarray  # Agent whose params/optimizer/normalizer state to copy (self = keep)
    parent_hps: np.ndarray  # Agent whose hyperparameters to copy (self = keep)
    need_explore: np.ndarray  # Whether the copied hyperparameters are perturbed
    evolving_populations: List[bool]

    def summary(self) -> Dict[str, int]:
        agents = np.arange(len(self.parent_network))
        return {
            'num_network_copies': int(np.sum(self.parent_network != agents)),
            'num_hps_copies': int(np.sum(self.parent_hps != agents)),
            'num_explored': int(np.sum(self.need_explore)),
        }


def evolving_populations(round_index: int, frequencies: Sequence[int]) -> List[bool]:
    """Sub-population `p` evolves after every `frequencies[p]`-th round."""
    return [(round_index + 1) % freq == 0 for freq in frequencies]


def genetics(fitness: Sequence[float], round_index: int, frequencies: Sequence[int]) -> Inheritance:
    """Computes who inherits from whom after round `round_index`.

    Args:
      fitness: one score per agent, higher is better. Agents are laid out by
        sub-population: agent `g` belongs to sub-population
        `g // (num_agents // len(frequencies))`.
      round_index: index of the round that just finished.
      frequencies: evolution period (in rounds) of each sub-population.
    """
    fitness = np.asarray(fitness, dtype=np.float64)
    # NaN fitness (diverged agent) ranks last
    fitness = np.where(np.isfinite(fitness), fitness, -np.inf)
    num_agents = len(fitness)
    num_populations = len(frequencies)
    per_pop = num_agents // num_populations
    check_population_layout(num_agents, frequencies)

    global_ranking = np.argsort(-fitness, kind='stable')
    local_rankings = [
        np.argsort(-fitness[p * per_pop:(p + 1) * per_pop], kind='stable')
        for p in range(num_populations)
    ]
    # Higher is better; used to compare an agent against a potential migrant.
    inverse_ranking = np.empty(num_agents, dtype=np.int64)
    inverse_ranking[global_ranking] = num_agents - np.arange(num_agents)

    parent_network = np.arange(num_agents)
    parent_hps = np.arange(num_agents)
    need_explore = np.zeros(num_agents, dtype=bool)
    evolving = evolving_populations(round_index, frequencies)

    def glob(local: int, population: int) -> int:
        return int(local + per_pop * population)

    share = per_pop // POPULATION_QUARTERS

    # Internal exploit: the bottom quarter copies the top quarter.
    for p in range(num_populations):
        if not evolving[p]:
            continue
        ranking = local_rankings[p]
        for loser, winner in zip(ranking[-share:], ranking[:share]):
            parent_network[glob(loser, p)] = glob(winner, p)
            parent_hps[glob(loser, p)] = glob(winner, p)
            need_explore[glob(loser, p)] = True

    # Asymmetric migration into the third quarter.
    for p in range(num_populations):
        if not evolving[p]:
            continue
        ranking = local_rankings[p]
        bracket = [glob(a, p) for a in ranking[MIGRATION_BRACKET_START * share:MIGRATION_BRACKET_END * share]]
        external = [int(a) for a in global_ranking if a // per_pop != p]
        for agent in bracket:
            if not external:
                break
            migrant = external[0]
            if inverse_ranking[agent] >= inverse_ranking[migrant]:
                continue
            parent_network[agent] = migrant
            source = migrant // per_pop
            if frequencies[p] < frequencies[source]:
                parent_hps[agent] = migrant
            elif frequencies[p] > frequencies[source]:
                parent_hps[agent] = glob(ranking[0], p)
            external.pop(0)

    return Inheritance(parent_network, parent_hps, need_explore, evolving)


def check_population_layout(num_agents: int, frequencies: Sequence[int]) -> None:
    if not frequencies or any(int(f) < 1 for f in frequencies):
        raise ValueError(f"PBT frequencies must be positive integers, got {list(frequencies)}")
    if num_agents % len(frequencies) != 0:
        raise ValueError(
            f"num_agents ({num_agents}) must be divisible by the number of frequencies "
            f"({len(frequencies)}) so every sub-population has the same size."
        )
    if (num_agents // len(frequencies)) % POPULATION_QUARTERS != 0:
        raise ValueError(
            f"Each sub-population needs a multiple of {POPULATION_QUARTERS} agents (it is split into "
            f"quarters), but num_agents / len(frequencies) = {num_agents // len(frequencies)}."
        )


# ---------------------------------------------------------------------------
# Hyperparameter search space
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HyperparameterSpec:
    """How one continuous hyperparameter is initialised and perturbed.

    `space` is the coordinate the hyperparameter lives in:
      'log'       -- multiply by a perturbation factor (lr, entropy, clip, ...)
      'one_minus' -- multiply (1 - x) by the factor, for values close to 1
                     (discounting, gae_lambda), so 0.99 -> 0.992 / 0.9875
    Initial values are drawn log-uniformly in that coordinate from
    [init_low, init_high]; perturbed values are clipped to [low, high].
    """

    init_low: float
    init_high: float
    low: float
    high: float
    space: str = 'log'

    def to_coord(self, value: float) -> float:
        return 1.0 - value if self.space == 'one_minus' else value

    def from_coord(self, coord: float) -> float:
        return 1.0 - coord if self.space == 'one_minus' else coord

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        a, b = sorted((self.to_coord(self.init_low), self.to_coord(self.init_high)))
        coords = np.exp(rng.uniform(math.log(a), math.log(b), size))
        return np.clip(self.from_coord(coords), self.low, self.high)

    def perturb(self, value: float, factor: float) -> float:
        return float(np.clip(self.from_coord(self.to_coord(value) * factor), self.low, self.high))


# Defaults for every PPO hyperparameter the population trainer can change between rounds without
# recompiling. Ranges follow the MF-PBT Brax configs (lr 1e-5..1e-3, entropy 1e-3..1e-1), widened
# downwards for entropy because the vision Ant collapses coincided with high policy entropy.
DEFAULT_SEARCH_SPACE: Dict[str, HyperparameterSpec] = {
    'learning_rate': HyperparameterSpec(1e-5, 1e-3, 1e-6, 1e-2),
    'entropy_cost': HyperparameterSpec(1e-4, 5e-2, 1e-6, 0.5),
    'clipping_epsilon': HyperparameterSpec(0.1, 0.3, 0.02, 0.5),
    'discounting': HyperparameterSpec(0.95, 0.997, 0.9, 0.999, space='one_minus'),
    'gae_lambda': HyperparameterSpec(0.9, 0.99, 0.8, 0.999, space='one_minus'),
    'reward_scaling': HyperparameterSpec(0.01, 1.0, 1e-3, 10.0),
    'lagrangian_coef_rate': HyperparameterSpec(0.1, 30.0, 1e-3, 300.0),
}


def penalised_fitness(
        score: Optional[float],
        cost: Optional[float] = None,
        cost_limit: Optional[float] = None,
        cost_penalty: float = 0.0,
) -> float:
    """Ranking score of one agent: `score - cost_penalty * max(0, cost - cost_limit)`.

    Without a cost limit (or penalty) this is the score itself. A missing or
    non-finite score ranks last (-inf); so does a missing cost when a limit is
    set, so a broken cost metric can't make an agent look safe.
    """
    if score is None or not np.isfinite(score):
        return float('-inf')
    if cost_limit is None or cost_penalty <= 0:
        return float(score)
    if cost is None or not np.isfinite(cost):
        return float('-inf')
    return float(score - cost_penalty * max(0.0, cost - cost_limit))


def build_search_space(names: Sequence[str], overrides: Optional[Mapping[str, Mapping]] = None) -> Dict[str, HyperparameterSpec]:
    """Picks `names` from DEFAULT_SEARCH_SPACE, with per-field overrides, e.g.
    `{"learning_rate": {"init_low": 3e-5, "init_high": 3e-4}}`."""
    overrides = dict(overrides or {})
    unknown = [n for n in list(names) + list(overrides) if n not in DEFAULT_SEARCH_SPACE]
    if unknown:
        raise ValueError(
            f"Unsupported PBT hyperparameter(s) {unknown}; choose from {sorted(DEFAULT_SEARCH_SPACE)}."
        )
    space = {}
    for name in names:
        fields = {**DEFAULT_SEARCH_SPACE[name].__dict__, **overrides.get(name, {})}
        space[name] = HyperparameterSpec(**fields)
    return space


def explore(
        hparams: Mapping[str, float],
        search_space: Mapping[str, HyperparameterSpec],
        rng: np.random.Generator,
        factors: Tuple[float, ...] = (0.8, 1.25),
) -> Dict[str, float]:
    """PBT 'perturb' explore: multiply every searched hyperparameter by a factor drawn from `factors`."""
    out = dict(hparams)
    for name, spec in search_space.items():
        out[name] = spec.perturb(out[name], float(rng.choice(factors)))
    return out


def explore_rng(seed: int, round_index: int, agent: int) -> np.random.Generator:
    """RNG keyed on (seed, round, agent) so any worker draws the same perturbation for an agent."""
    return np.random.default_rng([seed, round_index, agent])
