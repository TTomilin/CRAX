"""Shared-filesystem state of a population-based training run.

Multi-GPU runs use one worker process per GPU, each owning a fixed slice of
agent slots. After every round each worker writes, per owned agent:

  round_XXXX/agent_YYY.json  -- fitness, hyperparameters, round metrics (small, read by everyone)
  round_XXXX/agent_YYY.pkl   -- params, normalizer and optimizer state (read only when copied)

and then a `worker_Z.done` marker. Workers block on the markers of the whole
population (a file barrier), so there is no coordinator process: everyone
reads the same fitness table and runs the same deterministic evolution step.
The directory doubles as the resume point when a SLURM job is requeued.
"""

import json
import os
import pickle
import shutil
import time
from typing import Any, Dict, List, Optional

import numpy as np


def _atomic_write(path: str, write_fn, mode: str) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, mode) as f:
        write_fn(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.generic, np.ndarray)):
        return value.tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return None  # strict JSON has no NaN/inf
    return value


class PopulationStore:

    def __init__(self, root: str, num_agents: int, num_workers: int, worker_id: int,
                 launch_id: Optional[str] = None):
        """`launch_id` identifies one launch of all workers (e.g. the SLURM job id). The barrier only
        counts markers written by the same launch, so markers left behind by a crashed attempt can't
        release it before the restarted workers have redone the round."""
        self.root = root
        self.num_agents = num_agents
        self.num_workers = num_workers
        self.worker_id = worker_id
        self.launch_id = str(launch_id) if launch_id is not None else ''
        os.makedirs(root, exist_ok=True)

    # --- Layout -----------------------------------------------------------

    def round_dir(self, round_index: int) -> str:
        return os.path.join(self.root, f"round_{round_index:04d}")

    def _agent_path(self, round_index: int, agent: int, ext: str) -> str:
        return os.path.join(self.round_dir(round_index), f"agent_{agent:03d}.{ext}")

    def _marker_path(self, round_index: int, worker: int) -> str:
        return os.path.join(self.round_dir(round_index), f"worker_{worker}.done")

    # --- Experiment metadata ---------------------------------------------

    def check_or_write_meta(self, meta: Dict[str, Any]) -> None:
        """Pins the population layout so a resumed run can't silently change it."""
        path = os.path.join(self.root, 'meta.json')
        meta = _jsonable(meta)
        if os.path.exists(path):
            with open(path) as f:
                old = json.load(f)
            mismatched = {k: (old.get(k), v) for k, v in meta.items() if old.get(k) != v}
            if mismatched:
                raise ValueError(
                    f"PBT directory {self.root} was created with a different configuration "
                    f"(key: (stored, requested)): {mismatched}. Use a new --pbt_exp_name to start fresh."
                )
        elif self.worker_id == 0:
            _atomic_write(path, lambda f: json.dump(meta, f, indent=2), 'w')

    # --- Agents -----------------------------------------------------------

    def save_agent(self, round_index: int, agent: int, summary: Dict[str, Any], state: Any) -> None:
        os.makedirs(self.round_dir(round_index), exist_ok=True)
        _atomic_write(self._agent_path(round_index, agent, 'pkl'), lambda f: pickle.dump(state, f), 'wb')
        _atomic_write(self._agent_path(round_index, agent, 'json'),
                      lambda f: json.dump(_jsonable(summary), f), 'w')

    def load_summary(self, round_index: int, agent: int) -> Dict[str, Any]:
        with open(self._agent_path(round_index, agent, 'json')) as f:
            return json.load(f)

    def load_summaries(self, round_index: int) -> List[Dict[str, Any]]:
        return [self.load_summary(round_index, a) for a in range(self.num_agents)]

    def load_state(self, round_index: int, agent: int) -> Any:
        with open(self._agent_path(round_index, agent, 'pkl'), 'rb') as f:
            return pickle.load(f)

    # --- Barrier ----------------------------------------------------------

    def mark_done(self, round_index: int) -> None:
        _atomic_write(self._marker_path(round_index, self.worker_id), lambda f: f.write(self.launch_id), 'w')

    def _marker_ok(self, round_index: int, worker: int, current_launch: bool) -> bool:
        path = self._marker_path(round_index, worker)
        if not os.path.exists(path):
            return False
        if not current_launch or not self.launch_id:
            return True
        with open(path) as f:
            return f.read() == self.launch_id

    def is_complete(self, round_index: int, current_launch: bool = False) -> bool:
        return all(self._marker_ok(round_index, w, current_launch) for w in range(self.num_workers))

    def wait_for_round(self, round_index: int, timeout_s: float = 6 * 3600, poll_s: float = 2.0) -> float:
        """Blocks until every worker finished `round_index`; returns the time spent waiting."""
        start = time.time()
        while not self.is_complete(round_index, current_launch=True):
            if time.time() - start > timeout_s:
                missing = [w for w in range(self.num_workers)
                           if not self._marker_ok(round_index, w, current_launch=True)]
                raise TimeoutError(
                    f"Timed out after {timeout_s:.0f}s waiting for workers {missing} to finish round "
                    f"{round_index} (did a worker crash? check its log)."
                )
            time.sleep(poll_s)
        return time.time() - start

    def latest_complete_round(self) -> Optional[int]:
        rounds = []
        for name in os.listdir(self.root):
            if name.startswith('round_') and os.path.isdir(os.path.join(self.root, name)):
                try:
                    rounds.append(int(name.split('_')[1]))
                except ValueError:
                    continue
        for r in sorted(rounds, reverse=True):
            if self.is_complete(r):
                return r
        return None

    def cleanup_before(self, round_index: int) -> None:
        """Drops rounds older than `round_index` (call only after the barrier of `round_index`)."""
        for name in os.listdir(self.root):
            if not name.startswith('round_'):
                continue
            try:
                r = int(name.split('_')[1])
            except ValueError:
                continue
            if r < round_index:
                shutil.rmtree(os.path.join(self.root, name), ignore_errors=True)

    # --- History (round-by-round population log, worker 0 only) ----------

    def append_history(self, record: Dict[str, Any]) -> None:
        with open(os.path.join(self.root, 'history.jsonl'), 'a') as f:
            f.write(json.dumps(_jsonable(record)) + '\n')
