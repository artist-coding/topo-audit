"""Confidence intervals for quantities that are all paired differences.

Every number the paper reports is a difference between two graphs scored on the *same
tasks*, so the resampling has to preserve that pairing: resample task indices, then
recompute both sides on the resampled tasks. Resampling the two sides independently
would inflate the variance by the between-task variance, which is the dominant term —
task difficulty is far more variable than the gap between two topologies.

Two levels of resampling, because the quantities have two sources of noise:

- **tasks**, for anything that compares fixed graphs (C, E, and the levels);
- **tasks and seeds**, for anything involving the learned graph, since each seed stops
  at a different incumbent -- but only while they *do* stop in different places. Once
  the searches converge, every seed holds the same pair and resampling seeds adds
  nothing, so the task term is the floor. :func:`required_tasks` takes no seed count
  for exactly this reason.

A caveat that must travel with these intervals: seeds share tensor samples (the repeat
allocator recycles cells once the R draws per cell are used up), so the seed-level
component understates the variability of rerunning the whole pipeline including fresh
LLM sampling. The per-cell repeat variance bounds what was recycled and is reported
alongside.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

DEFAULT_ITERATIONS = 2000


@dataclass(frozen=True, slots=True)
class Interval:
    point: float
    low: float
    high: float
    level: float = 0.95

    @property
    def width(self) -> float:
        return self.high - self.low

    def excludes(self, value: float) -> bool:
        return value < self.low or value > self.high

    def disjoint_from(self, other: "Interval") -> bool:
        """Whether two terms can be told apart -- the E0 acceptance criterion."""
        return self.high < other.low or other.high < self.low

    def as_dict(self) -> dict:
        return {"point": self.point, "low": self.low, "high": self.high,
                "level": self.level, "width": self.width}


def _percentile_interval(samples: np.ndarray, point: float, level: float) -> Interval:
    tail = (1.0 - level) / 2.0
    low, high = np.nanpercentile(samples, [100 * tail, 100 * (1 - tail)])
    return Interval(point=point, low=float(low), high=float(high), level=level)


def paired_task_bootstrap(
    statistic: Callable[[Sequence[int]], float],
    n_tasks: int,
    *,
    iterations: int = DEFAULT_ITERATIONS,
    level: float = 0.95,
    seed: int = 20261014,
) -> Interval:
    """Resample task indices; `statistic` recomputes everything on those tasks.

    Passing indices rather than scores is what keeps the pairing intact: both sides of
    whatever difference is being measured see the identical resampled task set.
    """
    rng = np.random.default_rng(seed)
    point = statistic(range(n_tasks))
    draws = np.empty(iterations)
    for i in range(iterations):
        draws[i] = statistic(rng.integers(0, n_tasks, n_tasks))
    return _percentile_interval(draws, point, level)


def paired_task_seed_bootstrap(
    statistic: Callable[[Sequence[int], Sequence[int]], float],
    n_tasks: int,
    n_seeds: int,
    *,
    iterations: int = DEFAULT_ITERATIONS,
    level: float = 0.95,
    seed: int = 20261014,
) -> Interval:
    """Resample tasks and seeds together, for quantities involving the learned graph."""
    rng = np.random.default_rng(seed)
    point = statistic(range(n_tasks), range(n_seeds))
    draws = np.empty(iterations)
    for i in range(iterations):
        draws[i] = statistic(rng.integers(0, n_tasks, n_tasks),
                             rng.integers(0, n_seeds, n_seeds))
    return _percentile_interval(draws, point, level)


def paired_difference_interval(
    left: np.ndarray, right: np.ndarray, *, iterations: int = DEFAULT_ITERATIONS,
    level: float = 0.95, seed: int = 20261014,
) -> Interval:
    """Interval for mean(left - right) over shared tasks.

    `left` and `right` are per-task score vectors for two graphs on the same tasks.
    """
    if left.shape != right.shape:
        raise ValueError("paired vectors must have the same shape")
    diff = left - right
    finite = ~np.isnan(diff)
    values = diff[finite]
    if values.size == 0:
        raise ValueError("no overlapping tasks")
    return paired_task_bootstrap(
        lambda idx: float(np.mean(values[np.asarray(list(idx), dtype=int)])),
        values.size, iterations=iterations, level=level, seed=seed)


def paired_se(left: np.ndarray, right: np.ndarray) -> float:
    """Analytic standard error of a paired mean difference.

    Used for the discriminability diagnostic and for power calculations, where a
    closed form is wanted rather than a resample. For binary scores this reduces to
    sqrt(q / n) with q the fraction of tasks on which the two graphs disagree, which
    is why disagreement rate -- not accuracy -- sets the resolution of the study.
    """
    diff = left - right
    values = diff[~np.isnan(diff)]
    if values.size < 2:
        return float("nan")
    return float(np.std(values, ddof=1) / math.sqrt(values.size))


def disagreement_rate(left: np.ndarray, right: np.ndarray) -> float:
    """Fraction of shared tasks where two graphs give different scores."""
    diff = left - right
    values = diff[~np.isnan(diff)]
    return float(np.mean(values != 0)) if values.size else float("nan")


def required_tasks(epsilon: float, disagreement: float, *,
                   level: float = 0.95) -> int:
    """Validation tasks needed before `epsilon` can be excluded.

    Inverts the one-sided bound z * sqrt(q(1-q)/n) < epsilon at a true improvement of
    zero. Deliberately takes no seed count: in the converged regime a plateau test
    runs in, every seed holds the same pair of incumbents and is scored on the same
    split, so averaging over seeds does not reduce this noise. An earlier version
    divided by the seed count and understated the requirement by 3-7x.
    """
    if epsilon <= 0 or not 0 < disagreement <= 1:
        raise ValueError("epsilon must be positive and disagreement in (0, 1]")
    z = {0.90: 1.2816, 0.95: 1.6449, 0.99: 2.3263}.get(level, 1.6449)
    variance = disagreement * (1.0 - disagreement)
    return int(math.ceil((z * z * variance) / (epsilon * epsilon)))


def repeat_variance(values: np.ndarray) -> float:
    """Mean within-cell variance across repeats.

    Bounds how much sampling noise the tensor recycles when seeds share cells, which
    is what the seed-level interval fails to capture. `values[i, j, r]` is the score of
    graph i on task j at repeat r.
    """
    if values.ndim != 3:
        raise ValueError("expected a (graph, task, repeat) array")
    with np.errstate(invalid="ignore"):
        per_cell = np.nanvar(values, axis=2, ddof=1)
    return float(np.nanmean(per_cell))
