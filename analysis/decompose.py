"""The four-term decomposition, and the ladder it is really made of.

The identity is a telescoping sum, so it needs no assumptions:

    P(G_learned) - P(pi_single) = C - E - O - Delta_gen

    C         = P(G_star)      - P(pi_single)     capacity of the space
    E         = P(G_star)      - P(G_hat_N)       cost of selecting on N tasks
    O         = Phat_N(G_hat_N) - Phat_N(G_learned)   training-objective shortfall
    Delta_gen = [P(G_hat_N) - Phat_N(G_hat_N)] - [P(G_learned) - Phat_N(G_learned)]

Two readings of the same numbers, and the second is the one that drives decisions.

**As a ladder.** Add O and Delta_gen and the training-set terms cancel:

    O + Delta_gen = P(G_hat_N) - P(G_learned)

So the four terms are three adjacent gaps on one chain of population quantities,
P(pi_single) <= P(G_learned) <= P(G_hat_N) <= P(G_star), and each gap maps to exactly
one action:

    P(G_star)   - P(G_hat_N)   = E              -> add tasks
    P(G_hat_N)  - P(G_learned) = O + Delta_gen  -> keep searching
    P(G_star)   - P(pi_single) = C              -> enlarge the space, or stop

Delta_gen is therefore not a fourth thing needing its own action; it is the exchange
rate that converts O, a training-objective quantity, into a population one. O stays
reported separately because it is what the optimizer itself can see, which is what the
discriminability hypothesis is about. The ladder is also why the main figure plots four
levels rather than a stack of signed terms: no sign confusion, and the identity closes
by construction.

**Selection bias is unavoidable, so it is reported rather than assumed away.**
Even an exhaustively enumerable space yields only the best graph *on a finite audit
sample*; max over 25 candidates on 300 items is optimistic about its own score. Two
estimators are produced and they answer different questions:

- `plugin`: pick argmax on audit, score it on test. Unbiased for *that graph's*
  population score, hence a slightly conservative estimate of max_G P(G). This is the
  headline, because an independent test split makes it clean.
- `cross_fitted`: split audit in half, pick on one half, score on the other, average
  over many random halvings. This estimates the performance of *the selection
  procedure*, not the maximum, and it is a downward-biased estimate of max_G P(G).
  Its value here is the gap to the naive single-max, which quantifies the winner's
  curse for Fig S1.

Calling the cross-fitted number an unbiased estimate of P(G_star) would be wrong, and
mixing the two inside one decomposition would break the identity assertion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

#: The identity must close to this tolerance on every reported row.
IDENTITY_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class ScoreMatrix:
    """Per-graph, per-task scores for one split.

    `values[i, j]` is the score of graph `exec_hashes[i]` on task `task_ids[j]`,
    averaged over repeats. NaN marks a cell that was never run.
    """

    exec_hashes: tuple[str, ...]
    task_ids: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.values.shape != (len(self.exec_hashes), len(self.task_ids)):
            raise ValueError("values shape does not match labels")

    def row(self, exec_hash: str) -> np.ndarray:
        return self.values[self.exec_hashes.index(exec_hash)]

    def means(self) -> np.ndarray:
        return np.nanmean(self.values, axis=1)

    def argmax_hash(self) -> str:
        return self.exec_hashes[int(np.nanargmax(self.means()))]

    def subset(self, task_indices: Sequence[int]) -> "ScoreMatrix":
        idx = list(task_indices)
        return ScoreMatrix(self.exec_hashes,
                           tuple(self.task_ids[i] for i in idx),
                           self.values[:, idx])


@dataclass(frozen=True, slots=True)
class Decomposition:
    """One row: one optimizer, one seed, one checkpoint."""

    optimizer: str
    seed: int
    step: int
    g_star: str
    g_hat_n: str
    g_learned: str
    # population levels, from the test split
    p_single: float
    p_learned: float
    p_hat_n: float
    p_star: float
    # training-objective levels, from the train split
    train_hat_n: float
    train_learned: float

    @property
    def capacity(self) -> float:
        return self.p_star - self.p_single

    @property
    def estimation(self) -> float:
        return self.p_star - self.p_hat_n

    @property
    def optimization(self) -> float:
        return self.train_hat_n - self.train_learned

    @property
    def generalization(self) -> float:
        return ((self.p_hat_n - self.train_hat_n)
                - (self.p_learned - self.train_learned))

    @property
    def achieved_gain(self) -> float:
        return self.p_learned - self.p_single

    @property
    def search_value(self) -> float:
        """O + Delta_gen = P(G_hat_N) - P(G_learned): what perfect search would add."""
        return self.p_hat_n - self.p_learned

    def residual(self) -> float:
        """Identity residual; asserted to be ~0 on every row."""
        return self.achieved_gain - (self.capacity - self.estimation
                                     - self.optimization - self.generalization)

    def check(self, tolerance: float = IDENTITY_TOLERANCE) -> None:
        """Assert the identity, as the hard rule requires.

        Worth being precise about what this can catch. Given one stored value per
        level, the identity is a telescoping sum of those six fields, so it holds for
        *any* field values -- the assertion is a regression guard on the four term
        formulas above, not a check on the data.

        The failure it is really standing in for is mixing estimators: computing C
        from one estimate of P(G_star) and E from another. That cannot happen through
        this dataclass, because `p_star` is stored once and both terms read it. See
        :func:`mixed_star_residual` for what it would cost if it could.
        """
        if not math.isfinite(self.residual()) or abs(self.residual()) > tolerance:
            raise AssertionError(
                f"identity violated by {self.residual():.3e} for "
                f"{self.optimizer} seed={self.seed} step={self.step}")

    def as_dict(self) -> dict:
        return {
            "optimizer": self.optimizer, "seed": self.seed, "step": self.step,
            "g_star": self.g_star, "g_hat_n": self.g_hat_n,
            "g_learned": self.g_learned,
            "p_single": self.p_single, "p_learned": self.p_learned,
            "p_hat_n": self.p_hat_n, "p_star": self.p_star,
            "train_hat_n": self.train_hat_n, "train_learned": self.train_learned,
            "C": self.capacity, "E": self.estimation, "O": self.optimization,
            "delta_gen": self.generalization,
            "achieved_gain": self.achieved_gain, "search_value": self.search_value,
            "residual": self.residual(),
        }


def mixed_star_residual(row: Decomposition, p_star_for_capacity: float) -> float:
    """Residual if capacity used a different estimate of P(G_star) than estimation did.

    The module deliberately offers two estimators of P(G_star) -- the plug-in argmax
    scored on test, and the cross-fitted selection estimate -- because they answer
    different questions. Feeding one into C while E keeps the other breaks the
    identity by exactly their difference, which is the winner's curse in score points.

    This exists so that the hazard is measurable rather than merely warned about: a
    report that wants the cross-fitted number quotes it beside C, never inside it.
    """
    capacity = p_star_for_capacity - row.p_single
    return row.achieved_gain - (capacity - row.estimation
                                - row.optimization - row.generalization)


# --------------------------------------------------------------- selecting graphs

def select_g_hat_n(train: ScoreMatrix) -> str:
    """Empirical maximiser on the training split: the optimizer's unreachable target."""
    return train.argmax_hash()


def select_g_star_plugin(audit: ScoreMatrix) -> str:
    """Argmax on audit. Scored on test later, which is what keeps it honest."""
    return audit.argmax_hash()


@dataclass(frozen=True, slots=True)
class CrossFitResult:
    cross_fitted: float
    naive_max: float
    n_splits: int
    per_split: tuple[float, ...] = field(default=())

    @property
    def selection_bias(self) -> float:
        """Naive single-max minus cross-fitted: the winner's curse, in score points."""
        return self.naive_max - self.cross_fitted


def cross_fit_capacity(audit: ScoreMatrix, n_splits: int = 20,
                       seed: int = 20261014) -> CrossFitResult:
    """Quantify the winner's curse by selecting and scoring on disjoint audit halves.

    Returns the cross-fitted mean *and* the naive single-max so the difference can be
    reported. The cross-fitted value estimates the selection procedure's expected
    score, which is below max_G P(G); it is a diagnostic, not the headline estimate
    of capacity.
    """
    rng = np.random.default_rng(seed)
    n_tasks = len(audit.task_ids)
    if n_tasks < 4:
        raise ValueError("need at least 4 audit tasks to split")
    held: list[float] = []
    for _ in range(n_splits):
        order = rng.permutation(n_tasks)
        half = n_tasks // 2
        for pick, score in ((order[:half], order[half:]), (order[half:], order[:half])):
            chosen = audit.subset(pick).argmax_hash()
            held.append(float(np.nanmean(audit.subset(score).row(chosen))))
    return CrossFitResult(
        cross_fitted=float(np.mean(held)),
        naive_max=float(np.nanmax(audit.means())),
        n_splits=n_splits, per_split=tuple(held))


def capacity_upper_bound(audit: ScoreMatrix, alpha: float = 0.05) -> float:
    """Finite-class upper confidence bound on max_G P(G).

    max over |S| candidates on n tasks, via a union bound:
    naive max + sqrt(ln(2|S|/alpha) / (2n)). Loose by construction -- with 25 graphs
    and 300 tasks the band is around 11 score points, far wider than the
    pre-registered epsilon. Reported as the upper end of an interval for P(G_star),
    never as the headline number.
    """
    n_graphs, n_tasks = len(audit.exec_hashes), len(audit.task_ids)
    slack = math.sqrt(math.log(2 * n_graphs / alpha) / (2 * n_tasks))
    return float(np.nanmax(audit.means()) + slack)


# ------------------------------------------------------------------ decomposing

def decompose_row(
    optimizer: str,
    seed: int,
    step: int,
    g_learned: str,
    *,
    train: ScoreMatrix,
    test: ScoreMatrix,
    g_star: str,
    g_hat_n: str,
    p_single: float,
) -> Decomposition:
    """Build one row and assert the identity.

    Population levels all come from `test` and training levels all from `train`. Using
    one estimator of P(G_star) throughout is what makes the residual vanish; swapping
    in the cross-fitted value for `p_star` alone would break it.
    """
    for label, matrix, graph in (("test", test, g_learned), ("test", test, g_hat_n),
                                 ("test", test, g_star), ("train", train, g_hat_n),
                                 ("train", train, g_learned)):
        if graph not in matrix.exec_hashes:
            raise KeyError(f"{graph[:12]} missing from the {label} matrix")
        if np.all(np.isnan(matrix.row(graph))):
            raise ValueError(f"{graph[:12]} has no scored {label} tasks")

    row = Decomposition(
        optimizer=optimizer, seed=seed, step=step,
        g_star=g_star, g_hat_n=g_hat_n, g_learned=g_learned,
        p_single=p_single,
        p_learned=float(np.nanmean(test.row(g_learned))),
        p_hat_n=float(np.nanmean(test.row(g_hat_n))),
        p_star=float(np.nanmean(test.row(g_star))),
        train_hat_n=float(np.nanmean(train.row(g_hat_n))),
        train_learned=float(np.nanmean(train.row(g_learned))),
    )
    row.check()
    return row


def decompose_trajectory(
    optimizer: str,
    incumbents: Mapping[int, Mapping[int, str]],
    *,
    train: ScoreMatrix,
    audit: ScoreMatrix,
    test: ScoreMatrix,
    p_single: float,
) -> list[Decomposition]:
    """Every (seed, step) checkpoint of one optimizer.

    `incumbents[seed][step]` is the exec_hash the optimizer held at that step.
    """
    g_star = select_g_star_plugin(audit)
    g_hat_n = select_g_hat_n(train)
    rows = []
    for seed in sorted(incumbents):
        for step in sorted(incumbents[seed]):
            rows.append(decompose_row(
                optimizer, seed, step, incumbents[seed][step],
                train=train, test=test, g_star=g_star, g_hat_n=g_hat_n,
                p_single=p_single))
    return rows


def ladder_levels(rows: Sequence[Decomposition]) -> dict:
    """The four population levels for the main figure, averaged across seeds per step.

    P(G_star), P(G_hat_N) and P(pi_single) do not depend on the step; only the learned
    level moves. A plateau is that one line flattening while the others stay put.
    """
    steps = sorted({row.step for row in rows})
    learned = [float(np.mean([r.p_learned for r in rows if r.step == s])) for s in steps]
    reference = rows[0]
    return {
        "steps": steps,
        "p_learned": learned,
        "p_hat_n": reference.p_hat_n,
        "p_star": reference.p_star,
        "p_single": reference.p_single,
    }
