"""Pair search steps by committed task order, within one search seed.

A step is one propose/evaluate/observe round (eight charged candidates for a full
REINFORCE group). Every round is checkpointed, including singleton rounds and the
last partial group. cum_evals, not step, is the candidate-compute axis.

Replay starts with a fresh optimizer and Executor against the existing cache. We
rewrite the deterministic trajectory from the start; the runtime cache handles
interrupted work, so there is no second checkpoint/repeat allocator to reconcile.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Sequence

from data.loaders.math import Task
from optimizers.base import Optimizer, canonical
from runtime.executor import EvalResult, Executor


@dataclass(frozen=True, slots=True)
class TaskBatches:
    """Deterministic minibatch sequence, shared across optimizers at the same seed.

    Two axes, deliberately separated:

    - `seed` is the committed data-split seed. It fixes the task order so the sequence
      is reproducible from the repository alone.
    - `search_seed` is the optimizer's seed. It must enter the sequence, because a
      rerun of the whole study would draw different minibatches; holding the sequence
      fixed across seeds removes that variation, makes the 20 incumbents more alike
      than a real rerun, and so makes `frozen` verdicts more likely and the
      seed-dimension interval too narrow.

    Two optimizers at the *same* search seed still see identical batches, which is the
    pairing the plateau test and the discriminability diagnostic rely on.
    """

    tasks: tuple[Task, ...]
    tasks_per_eval: int
    seed: int
    search_seed: int = 0

    @classmethod
    def from_split(cls, tasks: Sequence[Task], tasks_per_eval: int, *,
                   split_path: str | Path | None = None,
                   search_seed: int = 0) -> TaskBatches:
        if any(task.split_role != "train" for task in tasks):
            raise ValueError("optimizer batches accept training tasks only")
        path = (Path(split_path) if split_path is not None else
                Path(__file__).resolve().parents[1] / "data" / "splits" / "math_v1.json")
        split = json.loads(path.read_text(encoding="utf-8"))
        order = split["splits"]["train"]
        by_id = {task.task_id: task for task in tasks}
        if (len(by_id) != len(tasks) or len(set(order)) != len(order)
                or set(by_id) != set(order)):
            raise ValueError("tasks must match the committed training split exactly")
        return cls(tuple(by_id[task_id] for task_id in order), tasks_per_eval,
                   split["seed"], search_seed)

    def __post_init__(self) -> None:
        if type(self.seed) is not int or type(self.search_seed) is not int:
            raise ValueError("batch seeds must be integers")
        if (type(self.tasks_per_eval) is not int
                or not 1 <= self.tasks_per_eval <= len(self.tasks)):
            raise ValueError("tasks_per_eval must fit the training split without duplicates")
        if any(task.split_role != "train" for task in self.tasks):
            raise ValueError("optimizer batches accept training tasks only")
        if len({task.task_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("duplicate training task ids")

    def batch(self, step: int) -> tuple[Task, ...]:
        if type(step) is not int or step < 0:
            raise ValueError("step must be a nonnegative integer")
        # Hash ranking avoids dependence even on a local RNG's consumption history.
        def priority(index: int) -> bytes:
            identity = (self.seed, self.search_seed, step, index,
                        self.tasks[index].task_id)
            return hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).digest()

        indices = sorted(range(len(self.tasks)), key=priority)[:self.tasks_per_eval]
        return tuple(self.tasks[index] for index in indices)


def discriminability(evals: Sequence[EvalResult]) -> dict:
    """Paired top-two difference; aliases are repeats of a single computation.

    Average repeated computations within each task before choosing the top two.
    A singleton computation (or fewer than two tasks) has no estimable paired SE:
    the numeric zero placeholders with n_tasks=0 mark an unavailable diagnostic,
    not evidence of equal performance. Nothing in search consumes this diagnostic.
    """
    unavailable = {"top2_paired_diff": 0.0, "se": 0.0, "n_tasks": 0}
    if not evals:
        return unavailable
    ids = evals[0].task_ids
    if any(result.task_ids != ids or len(result.scores) != len(ids) for result in evals):
        raise ValueError("discriminability requires the same ordered task batch")
    grouped: dict[str, list[tuple[float, ...]]] = {}
    for result in evals:
        grouped.setdefault(result.exec_hash, []).append(result.scores)
    if len(grouped) < 2 or len(ids) < 2:
        return unavailable
    scores = {key: tuple(map(fmean, zip(*rows))) for key, rows in grouped.items()}
    first, second = sorted(scores, key=lambda key: (-fmean(scores[key]), key))[:2]
    differences = [a - b for a, b in zip(scores[first], scores[second])]
    return {"top2_paired_diff": fmean(differences),
            "se": stdev(differences) / math.sqrt(len(ids)), "n_tasks": len(ids)}


def run_optimizer(optimizer: Optimizer, executor: Executor, tasks: Sequence[Task], *,
                  runs_dir: str | Path = "runs", split_path: str | Path | None = None,
                  repeat_hint: int = 0) -> Path:
    """Run the whole candidate budget using only Executor.evaluate_batch.

    repeat_hint is a fixed lower bound, forwarded on every call. Passing the step
    number would skip hits for infrequently requested graph/task pairs and could
    reuse a tensor sample before the first R-hit cycle is exhausted. A fresh
    Executor is required for each optimizer/seed run to isolate its hit counters.
    """
    batches = TaskBatches.from_split(tasks, optimizer.budget.tasks_per_eval,
                                     split_path=split_path,
                                     search_seed=optimizer.seed)
    if type(repeat_hint) is not int or repeat_hint < 0:
        raise ValueError("repeat_hint must be a nonnegative integer")
    if optimizer.budget.spent_evals or optimizer.budget.spent_tokens:
        raise ValueError("replay with a fresh optimizer and budget")
    for component in (executor.exp, optimizer.name):
        if not component or Path(component).name != component or component in (".", ".."):
            raise ValueError("experiment and optimizer names must be directory names")
    path = Path(runs_dir) / executor.exp / optimizer.name / f"seed{optimizer.seed}" / "trajectory.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        step = 0
        while optimizer.budget.left():
            candidates = [canonical(graph) for graph in optimizer.propose()]
            if not candidates or len(candidates) > optimizer.budget.left():
                raise ValueError("optimizer must propose within its remaining budget")
            batch = batches.batch(step)
            evals = executor.evaluate_batch(candidates, batch, seed=optimizer.seed,
                                            repeat_hint=repeat_hint)
            optimizer.observe(evals)
            record = {
                "step": step, "candidates": [graph.graph_hash for graph in candidates],
                "exec_hashes": [graph.exec_hash for graph in candidates],
                "task_ids": [task.task_id for task in batch],
                "scores": [list(result.scores) for result in evals],
                "cum_evals": optimizer.budget.spent_evals,
                "cum_tokens": optimizer.budget.spent_tokens,
                "incumbent": canonical(optimizer.incumbent()).graph_hash,
                "state": optimizer.state(), "discriminability": discriminability(evals),
            }
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False) + "\n")
            stream.flush()
            step += 1
    return path
