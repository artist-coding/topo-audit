"""Keep search state separate from the authority to execute or read scores.

Only a matched, complete EvalResult can spend a candidate evaluation. In particular,
canonical aliases and cache hits still cost evaluations and their reported tokens;
counting endpoint calls instead would give replayed searches free compute.
"""

from __future__ import annotations

import itertools
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Protocol, Sequence

from runtime.executor import EvalResult
from space.topology import Topology


@dataclass(slots=True)
class Budget:
    max_evals: int
    tasks_per_eval: int
    spent_evals: int = 0
    spent_tokens: int = 0

    def __post_init__(self) -> None:
        values = (self.max_evals, self.tasks_per_eval, self.spent_evals, self.spent_tokens)
        if any(type(value) is not int for value in values):
            raise ValueError("budget values must be integers")
        if (self.max_evals < 0 or self.tasks_per_eval < 1 or self.spent_tokens < 0
                or not 0 <= self.spent_evals <= self.max_evals):
            raise ValueError("invalid candidate budget")

    def left(self) -> int:
        return self.max_evals - self.spent_evals


class Optimizer(Protocol):
    name: str
    budget: Budget
    seed: int

    def __init__(self, space: Sequence[Topology], budget: Budget, seed: int): ...
    def propose(self) -> list[Topology]: ...
    def observe(self, evals: Sequence[EvalResult]) -> None: ...
    def incumbent(self) -> Topology: ...
    def state(self) -> dict: ...


def canonical(topology: Topology) -> Topology:
    return Topology.canonical(topology.nodes, topology.edges, topology.space_id,
                              topology.aggregation)


def distinct_computations(space: Sequence[Topology]) -> tuple[Topology, ...]:
    """Keep the first labelled representative, without overweighting mask aliases."""
    by_execution: dict[str, Topology] = {}
    for graph in space:
        graph = canonical(graph)
        by_execution.setdefault(graph.exec_hash, graph)
    return tuple(by_execution.values())


@dataclass(frozen=True, slots=True)
class EdgeSettings:
    group_size: int
    lr: float
    theta_clip: float
    entropy_reg: float

    @classmethod
    def from_config(cls, path: str | Path | None = None) -> EdgeSettings:
        """Read the frozen numeric stanza without adding a YAML dependency.

        This deliberately accepts only the four scalar settings needed here. It is
        not a general YAML loader; unsupported syntax fails instead of silently
        falling back to a second set of experimental defaults.
        """
        path = (Path(path) if path is not None
                else Path(__file__).resolve().parents[1] / "configs" / "e0.yaml")
        values: dict[str, int | float] = {}
        inside = False
        for line in path.read_text(encoding="utf-8").splitlines():
            source = line.split("#", 1)[0].rstrip()
            if not source.strip():
                continue
            if not source.startswith(" "):
                if inside:
                    break
                inside = source == "reinforce_edge:"
            elif inside:
                key, value = source.strip().split(":", 1)
                values[key] = json.loads(value)
        return cls(**values)

    def __post_init__(self) -> None:
        if type(self.group_size) is not int or self.group_size < 1:
            raise ValueError("group_size must be positive")
        if (any(not math.isfinite(v) for v in (self.lr, self.theta_clip, self.entropy_reg))
                or self.lr <= 0 or self.theta_clip <= 0 or self.entropy_reg < 0):
            raise ValueError("invalid edge learning settings")


class EdgeSpace:
    """Recover declared nodes before editing edges, including currently pruned nodes.

    Bernoulli policies require a full optional-edge space. Restricted spaces such as
    S0-strict need a different policy, rather than rejection sampling that silently
    changes the independent Bernoulli likelihood.
    """

    def __init__(self, space: Sequence[Topology]):
        self.anchor = max(space, key=lambda graph: len(graph.nodes))
        declared = {node.id: node for node in self.anchor.nodes}
        for graph in space:
            if (graph.terminal != self.anchor.terminal
                    or graph.aggregation != self.anchor.aggregation
                    or any(declared.get(node.id) != node for node in graph.nodes)
                    or tuple(n for n in self.anchor.nodes if n.id in graph.node_ids)
                    != graph.nodes):
                raise ValueError("edge search requires one declared node configuration")
        self.edges = tuple(itertools.combinations(self.anchor.node_ids, 2))
        self._allowed = {graph.exec_hash for graph in space}
        # Aliased labelled DAGs may be omitted, but every computation must be present.
        for mask in itertools.product((0, 1), repeat=len(self.edges)):
            if self.topology(mask).exec_hash not in self._allowed:
                raise ValueError("edge search requires the full optional-edge space")

    def topology(self, mask: Sequence[int]) -> Topology:
        return Topology.canonical(
            self.anchor.nodes, (edge for edge, bit in zip(self.edges, mask) if bit),
            self.anchor.space_id, self.anchor.aggregation)

    def mask(self, graph: Topology) -> tuple[int, ...]:
        return tuple(int(edge in graph.edges) for edge in self.edges)


@dataclass(slots=True)
class _Estimate:
    total: float = 0.0
    count: int = 0

    @property
    def mean(self) -> float:
        return self.total / self.count


class SearchOptimizer:
    """Shared proposal/result matching; subclasses never receive an executor."""

    name: str

    def __init__(self, space: Sequence[Topology], budget: Budget, seed: int):
        graphs = {graph.graph_hash: graph for graph in map(canonical, space)}
        if not graphs:
            raise ValueError("optimizer space must not be empty")
        if type(seed) is not int:
            raise ValueError("optimizer seed must be an integer")
        if budget.spent_evals or budget.spent_tokens:
            raise ValueError("replay with a fresh optimizer and budget")
        self.space = tuple(graphs.values())
        self.budget, self.seed = budget, seed
        self._rng = random.Random(seed)
        self._graphs = graphs
        self._allowed = {graph.exec_hash for graph in self.space}
        self._estimates: dict[str, _Estimate] = {}
        self._pending: tuple[Topology, ...] = ()

    def propose(self) -> list[Topology]:
        # Repeating a proposal before observation must not draw a different policy.
        if self._pending:
            return list(self._pending)
        if not self.budget.left():
            return []
        candidates = tuple(map(canonical, self._propose()))
        if not candidates or len(candidates) > self.budget.left():
            raise ValueError("proposal must fit the remaining candidate budget")
        if any(graph.exec_hash not in self._allowed for graph in candidates):
            raise ValueError("proposal is outside the search space")
        self._pending = candidates
        return list(candidates)

    def observe(self, evals: Sequence[EvalResult]) -> None:
        if not self._pending or len(evals) != len(self._pending):
            raise ValueError("observe requires the complete pending proposal")
        task_ids: tuple[str, ...] | None = None
        for graph, result in zip(self._pending, evals):
            if not isinstance(result, EvalResult):
                raise TypeError("scores must be EvalResults from evaluate_batch")
            if result.exec_hash != graph.exec_hash:
                raise ValueError("evaluation does not match the proposed computation")
            if (len(result.task_ids) != self.budget.tasks_per_eval
                    or len(result.scores) != self.budget.tasks_per_eval
                    or len(set(result.task_ids)) != len(result.task_ids)):
                raise ValueError("one evaluation must contain tasks_per_eval distinct tasks")
            if task_ids is not None and result.task_ids != task_ids:
                raise ValueError("candidates must share the same ordered task batch")
            task_ids = result.task_ids
            if any(not math.isfinite(score) or not 0 <= score <= 1 for score in result.scores):
                raise ValueError("scores must be finite probabilities")
            if any(type(tokens) is not int or tokens < 0
                   for tokens in (result.prompt_tokens, result.gen_tokens)):
                raise ValueError("token counts must be nonnegative integers")
        for graph, result in zip(self._pending, evals):
            identity = graph.graph_hash
            self._graphs[identity] = graph
            estimate = self._estimates.setdefault(identity, _Estimate())
            estimate.total += fmean(result.scores)
            estimate.count += 1
        self.budget.spent_evals += len(evals)
        self.budget.spent_tokens += sum(r.prompt_tokens + r.gen_tokens for r in evals)
        self._observe(evals)
        self._pending = ()

    def _propose(self) -> list[Topology]:
        raise NotImplementedError

    def _observe(self, evals: Sequence[EvalResult]) -> None:
        pass

    def _best(self, min_count: int = 1) -> Topology:
        eligible = [key for key, value in self._estimates.items() if value.count >= min_count]
        if not eligible:
            # An unconfirmed incumbent is a fixed prior, never a lucky one-shot winner.
            return self.space[0]
        key = min(eligible, key=lambda key: (-self._estimates[key].mean, key))
        return self._graphs[key]

    def incumbent(self) -> Topology:
        return self._best()

    def state(self) -> dict:
        return {"estimates": {
            key: {"mean": value.mean, "count": value.count}
            for key, value in sorted(self._estimates.items())}}
