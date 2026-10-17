"""Search policies consume training evaluations, never runtime capabilities."""

from __future__ import annotations

from typing import Sequence

from optimizers.base import Budget, EdgeSettings, Optimizer
from optimizers.greedy_local import GreedyLocal
from optimizers.prune_from_dense import PruneFromDense
from optimizers.random_search import RandomSearch
from optimizers.reinforce_edge import ReinforceEdge
from space.topology import Topology

OPTIMIZER_NAMES = ("random_search", "reinforce_edge", "prune_from_dense", "greedy_local")


def build_optimizer(name: str, space: Sequence[Topology], budget: Budget, seed: int,
                    settings: EdgeSettings) -> Optimizer:
    """The one place a config's optimizer name becomes an optimizer.

    The edge settings are passed explicitly so a run reads the config it was given,
    not whatever configs/e0.yaml happens to contain.
    """
    if name == "random_search":
        return RandomSearch(space, budget, seed)
    if name == "reinforce_edge":
        return ReinforceEdge(space, budget, seed, settings=settings)
    if name == "prune_from_dense":
        return PruneFromDense(space, budget, seed, lr=settings.lr,
                              magnitude_reg=settings.entropy_reg)
    if name == "greedy_local":
        return GreedyLocal(space, budget, seed)
    raise ValueError(f"unknown optimizer {name!r}; expected one of {OPTIMIZER_NAMES}")


__all__ = ["OPTIMIZER_NAMES", "Budget", "EdgeSettings", "GreedyLocal", "Optimizer",
           "PruneFromDense", "RandomSearch", "ReinforceEdge", "build_optimizer"]
