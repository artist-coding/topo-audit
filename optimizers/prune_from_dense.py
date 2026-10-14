"""Prune-from-dense: a deliberately narrow, monotone edge-removal search path.

Named for what it does rather than for AgentPrune, which it is not. AgentPrune
learns a continuous mask by policy gradient with a nuclear-norm penalty and then
prunes by magnitude; here the executor returns black-box rewards, so there is no
gradient to take through a soft edge. The paper cites AgentPrune as the source of
the "start dense, remove edges" idea and reports this as prune-from-dense.

Only about one graph per prune round is visited. Its empirical optimization gap can
therefore be much larger than random search's; adding exploration would erase the
property this baseline is meant to measure. The budget is divided over the path,
with any remaining evaluations spent at the terminal point, never on new branches.

The executor supplies black-box rewards, not derivatives through soft edges. We fit
an online linear reward surrogate to the visited binary masks with L1 shrinkage of
its continuous edge weights, then remove the smallest absolute weight. Co-occurring
edges are not individually identifiable along this path; seeded initial weights
break that symmetry. This is a limited surrogate, not a claim of recovering edge-level causal
effects or of differentiating through endpoint calls.

The brief supplies no separate mask hyperparameters. Defaults reuse the committed
edge learner's lr and regularisation scale; both can be explicitly overridden.
"""

from __future__ import annotations

import math
from statistics import fmean
from typing import Sequence

from optimizers.base import Budget, EdgeSettings, EdgeSpace, SearchOptimizer
from runtime.executor import EvalResult
from space.topology import Topology


class PruneFromDense(SearchOptimizer):
    name = "prune_from_dense"

    def __init__(self, space: Sequence[Topology], budget: Budget, seed: int, *,
                 lr: float | None = None, magnitude_reg: float | None = None):
        super().__init__(space, budget, seed)
        defaults = EdgeSettings.from_config()
        self.lr = defaults.lr if lr is None else lr
        self.magnitude_reg = defaults.entropy_reg if magnitude_reg is None else magnitude_reg
        if (not math.isfinite(self.lr) or self.lr <= 0
                or not math.isfinite(self.magnitude_reg) or self.magnitude_reg < 0):
            raise ValueError("invalid mask learning settings")
        self._edge_space = EdgeSpace(self.space)
        self._current = self._edge_space.topology([1] * len(self._edge_space.edges))
        self._mask_values = [self._rng.random() for _ in self._edge_space.edges]
        self._bias = 0.0
        self._prune_order: list[tuple[str, str]] = []
        self._round_evals = max(1, budget.max_evals // (len(self._edge_space.edges) + 1))
        self._round_count = 0

    def _propose(self) -> list[Topology]:
        return [self._current]

    def _observe(self, evals: Sequence[EvalResult]) -> None:
        active = self._edge_space.mask(self._current)
        scale = max(1, len(active))
        prediction = self._bias + sum(w * bit for w, bit in zip(self._mask_values, active)) / scale
        error = fmean(evals[0].scores) - prediction
        self._bias += self.lr * error
        for j, bit in enumerate(active):
            if bit:
                weight = self._mask_values[j] + self.lr * error / scale
                # Proximal L1 step avoids oscillating across zero under shrinkage.
                self._mask_values[j] = math.copysign(
                    max(0.0, abs(weight) - self.lr * self.magnitude_reg), weight)
        self._round_count += 1
        if self._round_count >= self._round_evals and any(active) and self.budget.left():
            j = min((i for i, bit in enumerate(active) if bit),
                    key=lambda i: (abs(self._mask_values[i]), i))
            self._prune_order.append(self._edge_space.edges[j])
            remaining = list(active)
            remaining[j] = 0
            self._current = self._edge_space.topology(remaining)
            # Canonicalisation can remove additional edges with unreachable outputs.
            for i, bit in enumerate(self._edge_space.mask(self._current)):
                if not bit:
                    self._mask_values[i] = 0.0
            self._round_count = 0

    def state(self) -> dict:
        return {**super().state(), "current": self._current.graph_hash,
                "edges": [list(edge) for edge in self._edge_space.edges],
                "mask_values": list(self._mask_values), "surrogate_bias": self._bias,
                "prune_order": [list(edge) for edge in self._prune_order],
                "round_evals": self._round_evals, "round_count": self._round_count,
                "lr": self.lr, "magnitude_reg": self.magnitude_reg}
