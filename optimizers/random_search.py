"""Uniform computational coverage is a baseline, even when S0 is exhausted.

Equivalent labelled DAGs must not get extra lottery tickets. Sampling is with
replacement over distinct exec_hashes, and a graph needs two observations before
its empirical mean can displace the fixed initial incumbent. Every draw is charged,
including repeated computations after coverage; early checkpoints retain the regime
in which the empirical optimization gap has not yet vanished.
"""

from __future__ import annotations

from typing import Sequence

from optimizers.base import Budget, SearchOptimizer, distinct_computations
from space.topology import Topology


class RandomSearch(SearchOptimizer):
    name = "random_search"

    def __init__(self, space: Sequence[Topology], budget: Budget, seed: int):
        super().__init__(distinct_computations(space), budget, seed)

    def _propose(self) -> list[Topology]:
        return [self._rng.choice(self.space)]

    def incumbent(self) -> Topology:
        return self._best(min_count=2)

    def state(self) -> dict:
        return {**super().state(), "confirmed": any(
            value.count >= 2 for value in self._estimates.values())}
