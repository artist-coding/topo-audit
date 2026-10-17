"""The finite-budget plateau test, and the two ways a curve can stop moving.

The claim is deliberately one-sided: a plateau is declared only when the data can
**exclude** further improvement above a pre-registered epsilon. "No significant change
in the last few steps" is not a plateau — it is a failure to detect one, and the
report says so in those words.

    improvement(t, dt) = P_val(G_learned at t+dt) - P_val(G_learned at t)
    plateau at t  <=>  upper end of the 95% one-sided interval on improvement < eps

Two cases that look identical in a learning curve and mean different things:

- **frozen**: every seed holds the same incumbent at t and t+dt. The paired difference
  is then identically zero with zero variance, and the test passes trivially. This says
  the optimizer stopped moving, not that nothing better was reachable.
- **moved but flat**: incumbents changed and the improvement is still bounded below
  epsilon. This is the substantive version of the claim.

They are reported separately. Collapsing them is how a search that merely got stuck
gets written up as a ceiling.

**Power, and the two errors that were in it.** The resolution of the test is set by
the variance of the per-task paired difference d, not by accuracy. For binary scores
with disagreement rate q, Var(d) = q - delta^2, which is q when the true improvement is
zero -- the case a power calculation is about.

The first error divided by n_tasks * n_seeds, which assumes each seed contributes
independent task noise. It does not. Every seed is scored on the *same* validation
split, and in the regime that matters -- a plateau, where the searches have converged
-- most seeds hold the *same* pair of incumbents, so averaging over seeds averages one
number repeatedly.

The second error used q(1-q), the variance of the disagreement indicator, instead of
q. The floor is therefore 1.645*sqrt(q/n): 3.0 points at q=0.1 and 4.25 at q=0.2 on
300 tasks, regardless of seed count (a 4,000-draw simulation gives 2.99 and 4.31).
Excluding epsilon = 2.0 at q=0.2 needs about 1,350 validation tasks; on 300 tasks the
smallest excludable epsilon is about 4.3. :func:`power_report` judges sufficiency
against the floor and accepts a *measured* variance of paired differences, which is
what should go into the pre-registration -- per-task scores averaged over repeats are
not binary, so no q-based formula is exact for them.

Note that :func:`assess_plateau` itself was never wrong: it resamples the observed
diffs matrix, so identical seed rows widen the interval on their own. Only the
pre-registration arithmetic was optimistic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from analysis.bootstrap import (
    DEFAULT_ITERATIONS,
    Interval,
    resolve_variance,
    disagreement_rate,
    paired_se,
    paired_variance,
    required_tasks,
    z_value,
)

FROZEN = "frozen"
MOVED_FLAT = "moved_but_flat"
NOT_DECLARED = "not_declared"


@dataclass(frozen=True, slots=True)
class PlateauVerdict:
    optimizer: str
    step: int
    delta_t: int
    epsilon: float
    improvement: float
    upper_bound: float
    status: str
    n_seeds: int
    n_tasks: int
    frozen_seeds: int
    disagreement: float

    @property
    def declared(self) -> bool:
        return self.status in (FROZEN, MOVED_FLAT)

    def describe(self) -> str:
        if self.status == FROZEN:
            return (f"{self.optimizer} step {self.step}: incumbent frozen in "
                    f"{self.frozen_seeds}/{self.n_seeds} seeds; plateau by definition, "
                    "not evidence of a ceiling")
        if self.status == MOVED_FLAT:
            return (f"{self.optimizer} step {self.step}: improvement bounded by "
                    f"{self.upper_bound:+.2f} < eps={self.epsilon}; plateau")
        return (f"{self.optimizer} step {self.step}: upper bound "
                f"{self.upper_bound:+.2f} >= eps={self.epsilon}; "
                "plateau not declared (not the same as no improvement)")

    def as_dict(self) -> dict:
        return {
            "optimizer": self.optimizer, "step": self.step, "delta_t": self.delta_t,
            "epsilon": self.epsilon, "improvement": self.improvement,
            "upper_bound": self.upper_bound, "status": self.status,
            "declared": self.declared, "n_seeds": self.n_seeds,
            "n_tasks": self.n_tasks, "frozen_seeds": self.frozen_seeds,
            "disagreement": self.disagreement, "note": self.describe(),
        }


def _per_seed_differences(
    incumbents: Mapping[int, Mapping[int, str]],
    val_rows: Mapping[str, np.ndarray],
    step: int,
    delta_t: int,
) -> tuple[np.ndarray, int]:
    """Per-seed, per-task paired differences between the t+dt and t incumbents.

    Every seed must cover both budgets. Dropping the seeds that do not would compute
    the verdict on whichever seeds happen to align with the grid -- for optimizers with
    multi-evaluation rounds that is a biased subset, or nothing at all. Put the
    trajectories on a common budget axis first (see :mod:`analysis.trajectory`).
    """
    missing = [seed for seed in sorted(incumbents)
               if step not in incumbents[seed] or step + delta_t not in incumbents[seed]]
    if missing:
        raise ValueError(
            f"seeds {missing} do not cover budgets {step} and {step + delta_t}; "
            "align trajectories to a common budget grid before testing")
    rows, frozen = [], 0
    for seed in sorted(incumbents):
        before, after = incumbents[seed][step], incumbents[seed][step + delta_t]
        if before == after:
            frozen += 1
        rows.append(val_rows[after] - val_rows[before])
    if not rows:
        raise ValueError("no seeds to test")
    return np.vstack(rows), frozen


def assess_plateau(
    optimizer: str,
    incumbents: Mapping[int, Mapping[int, str]],
    val_rows: Mapping[str, np.ndarray],
    *,
    step: int,
    delta_t: int,
    epsilon: float,
    iterations: int = DEFAULT_ITERATIONS,
    level: float = 0.95,
    seed: int = 20261014,
) -> PlateauVerdict:
    """One verdict for one optimizer at one checkpoint.

    `val_rows[exec_hash]` is that graph's per-task validation score vector. Resampling
    is over tasks and seeds jointly: a seed is a draw of "where this learner stopped",
    and that variation is part of the claim.
    """
    diffs, frozen = _per_seed_differences(incumbents, val_rows, step, delta_t)
    n_seeds, n_tasks = diffs.shape
    point = float(np.nanmean(diffs))

    if frozen == n_seeds:
        # Degenerate by construction: identical graphs, so the difference is exactly
        # zero and resampling adds nothing. Declared, but labelled as frozen.
        before = next(iter(incumbents.values()))[step]
        after = next(iter(incumbents.values()))[step + delta_t]
        return PlateauVerdict(
            optimizer=optimizer, step=step, delta_t=delta_t, epsilon=epsilon,
            improvement=0.0, upper_bound=0.0, status=FROZEN, n_seeds=n_seeds,
            n_tasks=n_tasks, frozen_seeds=frozen,
            disagreement=disagreement_rate(val_rows[after], val_rows[before]))

    rng = np.random.default_rng(seed)
    draws = np.empty(iterations)
    for i in range(iterations):
        task_idx = rng.integers(0, n_tasks, n_tasks)
        seed_idx = rng.integers(0, n_seeds, n_seeds)
        draws[i] = float(np.nanmean(diffs[np.ix_(seed_idx, task_idx)]))
    upper = float(np.nanpercentile(draws, 100 * level))

    status = MOVED_FLAT if upper < epsilon else NOT_DECLARED
    mean_disagreement = float(np.nanmean(np.abs(diffs) > 0))
    return PlateauVerdict(
        optimizer=optimizer, step=step, delta_t=delta_t, epsilon=epsilon,
        improvement=point, upper_bound=upper, status=status, n_seeds=n_seeds,
        n_tasks=n_tasks, frozen_seeds=frozen, disagreement=mean_disagreement)


def scan_plateau(
    optimizer: str,
    incumbents: Mapping[int, Mapping[int, str]],
    val_rows: Mapping[str, np.ndarray],
    *,
    steps: Sequence[int],
    delta_t: int,
    epsilon: float,
    **kwargs,
) -> list[PlateauVerdict]:
    """Verdicts across checkpoints; the first declared step is the reported plateau.

    A step that some seed does not cover raises instead of being skipped: a silently
    empty or partial scan reads exactly like "no plateau" in the report.
    """
    return [assess_plateau(optimizer, incumbents, val_rows, step=step,
                           delta_t=delta_t, epsilon=epsilon, **kwargs)
            for step in steps]


def testable_steps(budgets: Sequence[int], delta_t: int) -> list[int]:
    """Budgets t on a common grid for which t + delta_t is on the grid too."""
    available = set(budgets)
    return [t for t in sorted(available) if t + delta_t in available]


def first_plateau(verdicts: Sequence[PlateauVerdict]) -> PlateauVerdict | None:
    return next((v for v in verdicts if v.declared), None)


# ------------------------------------------------------------------------ power

@dataclass(frozen=True, slots=True)
class PowerReport:
    """Two bounds, because which one applies depends on seed diversity.

    `floor` assumes every seed converged to the same pair of incumbents, so seeds buy
    nothing. `optimistic` assumes each seed holds an independent pair. A plateau is by
    definition the converged case, so `sufficient` is judged against the floor.
    """

    epsilon: float
    disagreement: float | None
    variance: float
    n_tasks: int
    n_seeds: int
    floor_upper_bound: float
    optimistic_upper_bound: float
    sufficient: bool
    tasks_needed: int

    @property
    def expected_upper_bound(self) -> float:
        """The bound to design against: seeds cannot be assumed to help."""
        return self.floor_upper_bound

    @property
    def smallest_excludable_epsilon(self) -> float:
        """The floor itself: no smaller epsilon can ever be excluded on this split."""
        return self.floor_upper_bound

    def as_dict(self) -> dict:
        return {
            "epsilon": self.epsilon, "disagreement": self.disagreement,
            "variance": self.variance,
            "n_tasks": self.n_tasks, "n_seeds": self.n_seeds,
            "floor_upper_bound": self.floor_upper_bound,
            "optimistic_upper_bound": self.optimistic_upper_bound,
            "smallest_excludable_epsilon": self.smallest_excludable_epsilon,
            "sufficient": self.sufficient,
            "tasks_needed": self.tasks_needed,
            "note": ("floor assumes all seeds share one incumbent pair, which is the "
                     "converged regime a plateau test runs in; optimistic assumes "
                     "independent pairs per seed. variance is Var(d) of per-task "
                     "paired differences (q for binary scores under the null)"),
        }


def power_report(epsilon: float, disagreement: float | None = None,
                 n_tasks: int = 0, n_seeds: int = 1, *, level: float = 0.95,
                 variance: float | None = None) -> PowerReport:
    """Can this design exclude `epsilon` at all?

    Run before committing epsilon to the pre-registration: if the bound under a true
    improvement of zero already exceeds epsilon, the test can never declare a plateau
    and the design has to change, not the interpretation.

    Pass the measured `variance` of paired per-task differences when you have it (see
    :func:`observed_paired_variance`); otherwise a disagreement rate q is converted
    with Var(d) = q, its value for binary scores under the null.

    Seeds enter the optimistic bound only. They cannot enter the floor, because a
    plateau is the regime where searches have converged onto the same incumbents and
    averaging across seeds then averages one number repeatedly.
    """
    var = resolve_variance(disagreement, variance)
    z = z_value(level)
    floor = z * float(np.sqrt(var / max(n_tasks, 1)))
    optimistic = z * float(np.sqrt(var / (max(n_tasks, 1) * max(n_seeds, 1))))
    return PowerReport(
        epsilon=epsilon, disagreement=disagreement, variance=var, n_tasks=n_tasks,
        n_seeds=n_seeds, floor_upper_bound=floor, optimistic_upper_bound=optimistic,
        sufficient=floor < epsilon,
        tasks_needed=required_tasks(epsilon, level=level, variance=var))


def observed_disagreement(val_rows: Mapping[str, np.ndarray],
                          pairs: Sequence[tuple[str, str]]) -> float:
    """Mean disagreement over graph pairs, reported beside the variance."""
    rates = [disagreement_rate(val_rows[a], val_rows[b]) for a, b in pairs]
    finite = [r for r in rates if not np.isnan(r)]
    return float(np.mean(finite)) if finite else float("nan")


def observed_paired_variance(val_rows: Mapping[str, np.ndarray],
                             pairs: Sequence[tuple[str, str]]) -> float:
    """Mean Var(d) over graph pairs: the input :func:`power_report` should get.

    Measured on the per-task scores the plateau test actually uses (means over the
    validation repeats), so it needs no binary-score model.
    """
    values = [paired_variance(val_rows[a], val_rows[b]) for a, b in pairs]
    finite = [v for v in values if not np.isnan(v)]
    return float(np.mean(finite)) if finite else float("nan")


def discriminability(left: np.ndarray, right: np.ndarray) -> dict:
    """Paired gap between the top two candidates and its standard error.

    Recorded alongside each optimizer's stopping point to explain *why* it stalled.
    Never a stopping rule: repeated evaluation can shrink the standard error, so any
    fixed signal-to-noise threshold would be an artefact of the evaluation budget.
    """
    gap = float(np.nanmean(left - right))
    se = paired_se(left, right)
    return {"paired_diff": gap, "se": se,
            "ratio": gap / se if se and not np.isnan(se) and se > 0 else float("nan"),
            "disagreement": disagreement_rate(left, right)}
