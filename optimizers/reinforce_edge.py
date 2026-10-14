"""Group-relative policy gradients on latent masks, scores on canonical graphs.

Several masks can request the same computation. Their raw Bernoulli draws remain
the likelihood's sufficient statistics, while each canonical proposal gets its own
charged evaluation and fresh repeat. Substituting the pruned mask in the gradient
would bias learning for edges that happened not to reach the terminal.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from statistics import fmean, pstdev
from typing import Sequence

from optimizers.base import Budget, EdgeSettings, EdgeSpace, SearchOptimizer
from runtime.executor import EvalResult
from space.topology import Topology


class ReinforceEdge(SearchOptimizer):
    name = "reinforce_edge"

    def __init__(self, space: Sequence[Topology], budget: Budget, seed: int, *,
                 settings: EdgeSettings | None = None):
        super().__init__(space, budget, seed)
        self.settings = settings if settings is not None else EdgeSettings.from_config()
        self._edge_space = EdgeSpace(self.space)
        self.logits = [0.0] * len(self._edge_space.edges)
        self._sampled_masks: list[tuple[int, ...]] = []
        self._group_scores: list[float] = []
        self._advantages: list[float] = []

    def _probabilities(self) -> list[float]:
        return [1 / (1 + math.exp(-theta)) for theta in self.logits]

    def _propose(self) -> list[Topology]:
        probabilities = self._probabilities()
        self._sampled_masks = [
            tuple(int(self._rng.random() < probability) for probability in probabilities)
            for _ in range(min(self.settings.group_size, self.budget.left()))]
        return [self._edge_space.topology(mask) for mask in self._sampled_masks]

    def _observe(self, evals: Sequence[EvalResult]) -> None:
        self._group_scores = [fmean(result.scores) for result in evals]
        mean, std = fmean(self._group_scores), pstdev(self._group_scores)
        self._advantages = [(score - mean) / (std + 1e-6) for score in self._group_scores]
        for j, probability in enumerate(self._probabilities()):
            gradient = fmean(advantage * (mask[j] - probability)
                             for advantage, mask in zip(self._advantages, self._sampled_masks))
            # dH(Bernoulli(sigmoid(theta)))/dtheta = -theta * p * (1-p).
            gradient -= self.settings.entropy_reg * self.logits[j] * probability * (1 - probability)
            updated = self.logits[j] + self.settings.lr * gradient
            self.logits[j] = max(-self.settings.theta_clip, min(self.settings.theta_clip, updated))

    def incumbent(self) -> Topology:
        return self._edge_space.topology([int(p >= 0.5) for p in self._probabilities()])

    def state(self) -> dict:
        return {**super().state(), "edges": [list(edge) for edge in self._edge_space.edges],
                "settings": asdict(self.settings),
                "logits": list(self.logits), "sampled_masks": [list(m) for m in self._sampled_masks],
                "group_scores": list(self._group_scores), "advantages": list(self._advantages)}
