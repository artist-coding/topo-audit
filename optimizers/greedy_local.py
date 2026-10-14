"""Local improvements need a freshly scored reference on the same tasks.

Comparing new neighbours with an old current-point score would confound graph
quality with task-batch difficulty. The reference evaluation is charged just like
each neighbour. Restarts change the search point, while the best running empirical
estimate remains available as the incumbent.
"""

from __future__ import annotations

from statistics import fmean
from typing import Sequence

from optimizers.base import Budget, EdgeSpace, SearchOptimizer, distinct_computations
from runtime.executor import EvalResult
from space.topology import Topology


class GreedyLocal(SearchOptimizer):
    name = "greedy_local"

    def __init__(self, space: Sequence[Topology], budget: Budget, seed: int):
        super().__init__(space, budget, seed)
        self._edge_space = EdgeSpace(self.space)
        self._choices = distinct_computations(self.space)
        self._current = self._rng.choice(self._choices)
        self._neighbourhood_scores: dict[str, float] = {}
        self._comparison_point = self._current.graph_hash
        self._restart_count = 0
        self._neighbourhood_complete = False

    def _propose(self) -> list[Topology]:
        mask = self._edge_space.mask(self._current)
        neighbours: dict[str, Topology] = {}
        for j in range(len(mask)):
            changed = list(mask)
            changed[j] = 1 - changed[j]
            graph = self._edge_space.topology(changed)
            if graph.exec_hash != self._current.exec_hash:
                neighbours.setdefault(graph.exec_hash, graph)
        candidates = list(neighbours.values())
        self._rng.shuffle(candidates)
        self._neighbourhood_complete = len(candidates) + 1 <= self.budget.left()
        return [self._current, *candidates[:self.budget.left() - 1]]

    def _observe(self, evals: Sequence[EvalResult]) -> None:
        self._comparison_point = self._current.graph_hash
        self._neighbourhood_scores = {graph.graph_hash: fmean(result.scores)
                                      for graph, result in zip(self._pending, evals)}
        best = min(self._pending,
                   key=lambda graph: (-self._neighbourhood_scores[graph.graph_hash], graph.graph_hash))
        if self._neighbourhood_scores[best.graph_hash] > self._neighbourhood_scores[self._current.graph_hash]:
            self._current = best
        elif self._neighbourhood_complete and self.budget.left():
            alternatives = [graph for graph in self._choices
                            if graph.exec_hash != self._current.exec_hash]
            if alternatives:
                self._current = self._rng.choice(alternatives)
                self._restart_count += 1

    def state(self) -> dict:
        return {**super().state(), "current": self._current.graph_hash,
                "comparison_point": self._comparison_point,
                "neighbourhood_scores": dict(self._neighbourhood_scores),
                "neighbourhood_complete": self._neighbourhood_complete,
                "restart_count": self._restart_count}
