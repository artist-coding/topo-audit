"""Acceptance tests for build step 6 (analysis/).

Three things are load-bearing and each has a test that would fail loudly if broken:

1. the identity closes to 1e-9 on every row, including when Delta_gen is negative;
2. O + Delta_gen equals P(G_hat_N) - P(G_learned), the gap the "keep searching"
   decision reads;
3. the plateau test distinguishes a frozen incumbent from a genuinely flat curve, and
   refuses to declare a plateau when the data cannot exclude epsilon.

The numeric fixture is the worked example from the proposal (60 / 65 / 66 / 68 with
training scores 70 and 67), whose hand-computed terms are C=8, E=2, O=3,
Delta_gen=-2.
"""

from __future__ import annotations

import numpy as np
import pytest

from analysis.bootstrap import (
    Interval,
    disagreement_rate,
    paired_difference_interval,
    paired_se,
    paired_task_bootstrap,
    repeat_variance,
    required_tasks,
)
from analysis.decompose import (
    IDENTITY_TOLERANCE,
    Decomposition,
    ScoreMatrix,
    capacity_upper_bound,
    cross_fit_capacity,
    decompose_row,
    decompose_trajectory,
    ladder_levels,
    mixed_star_residual,
    prescribe,
    select_g_hat_n,
    select_g_star_plugin,
)
from analysis.plateau import (
    FROZEN,
    MOVED_FLAT,
    NOT_DECLARED,
    discriminability,
    first_plateau,
    power_report,
    assess_plateau,
    scan_plateau,
)

STAR, HAT, LEARNED = "h_star", "h_hat", "h_learned"


def _matrix(rows: dict[str, list[float]], n_tasks: int) -> ScoreMatrix:
    hashes = tuple(rows)
    values = np.array([rows[h] for h in hashes], dtype=float)
    assert values.shape[1] == n_tasks
    return ScoreMatrix(hashes, tuple(f"t{i}" for i in range(n_tasks)), values)


def _worked_example() -> Decomposition:
    """Proposal's example: p_single 60, learned 65, hat 66, star 68; train 70 / 67."""
    train = _matrix({STAR: [0.69], HAT: [0.70], LEARNED: [0.67]}, 1)
    test = _matrix({STAR: [0.68], HAT: [0.66], LEARNED: [0.65]}, 1)
    return decompose_row("opt", seed=0, step=10, g_learned=LEARNED, train=train,
                         test=test, g_star=STAR, g_hat_n=HAT, p_single=0.60)


# -------------------------------------------------------------------- identity

def test_worked_example_reproduces_the_hand_computed_terms():
    row = _worked_example()
    assert row.capacity == pytest.approx(0.08)
    assert row.estimation == pytest.approx(0.02)
    assert row.optimization == pytest.approx(0.03)
    assert row.generalization == pytest.approx(-0.02)
    assert row.achieved_gain == pytest.approx(0.05)


def test_identity_closes_to_tolerance():
    assert abs(_worked_example().residual()) < IDENTITY_TOLERANCE


def test_identity_closes_with_a_positive_delta_gen():
    """The learned graph overfitting its own batches flips Delta_gen's sign."""
    train = _matrix({STAR: [0.69], HAT: [0.70], LEARNED: [0.72]}, 1)
    test = _matrix({STAR: [0.68], HAT: [0.66], LEARNED: [0.60]}, 1)
    row = decompose_row("opt", 0, 10, LEARNED, train=train, test=test,
                        g_star=STAR, g_hat_n=HAT, p_single=0.60)
    assert row.generalization > 0
    assert abs(row.residual()) < IDENTITY_TOLERANCE


def test_identity_is_structural_given_one_estimator_per_level():
    """Documents the limit of the hard rule's assertion.

    With one stored value per level the identity is a telescoping sum of those
    fields, so it closes for arbitrary values. The assertion guards the term
    formulas; it cannot detect bad data. Stating that here keeps anyone from reading
    a passing check as evidence the numbers are right.
    """
    rng = np.random.default_rng(11)
    for _ in range(50):
        p_single, p_learned, p_hat_n, p_star, t_hat, t_learned = rng.random(6)
        row = Decomposition(
            optimizer="o", seed=0, step=0, g_star=STAR, g_hat_n=HAT,
            g_learned=LEARNED, p_single=p_single, p_learned=p_learned,
            p_hat_n=p_hat_n, p_star=p_star, train_hat_n=t_hat,
            train_learned=t_learned)
        assert abs(row.residual()) < IDENTITY_TOLERANCE
        row.check()


def test_mixing_estimators_of_p_star_breaks_the_identity():
    """The failure the hard rule is really guarding against.

    Using the cross-fitted estimate for C while E keeps the plug-in one shifts the
    residual by exactly their difference, so the two may be reported side by side
    but never inside one decomposition.
    """
    row = _worked_example()
    cross_fitted = row.p_star - 0.03
    residual = mixed_star_residual(row, cross_fitted)
    assert abs(residual) == pytest.approx(0.03)
    assert abs(mixed_star_residual(row, row.p_star)) < IDENTITY_TOLERANCE


def test_decompose_row_refuses_a_graph_with_no_scored_tasks():
    train = _matrix({STAR: [0.69], HAT: [0.70], LEARNED: [float("nan")]}, 1)
    test = _matrix({STAR: [0.68], HAT: [0.66], LEARNED: [0.65]}, 1)
    with pytest.raises(ValueError, match="no scored train tasks"):
        decompose_row("opt", 0, 10, LEARNED, train=train, test=test,
                      g_star=STAR, g_hat_n=HAT, p_single=0.60)


def test_decompose_row_refuses_a_graph_missing_from_a_split():
    train = _matrix({STAR: [0.69], HAT: [0.70]}, 1)
    test = _matrix({STAR: [0.68], HAT: [0.66], LEARNED: [0.65]}, 1)
    with pytest.raises(KeyError, match="missing from the train matrix"):
        decompose_row("opt", 0, 10, LEARNED, train=train, test=test,
                      g_star=STAR, g_hat_n=HAT, p_single=0.60)


def test_search_value_equals_o_plus_delta_gen():
    """The quantity the 'keep searching' decision reads must not need train scores."""
    row = _worked_example()
    assert row.search_value == pytest.approx(row.optimization + row.generalization)
    assert row.search_value == pytest.approx(row.p_hat_n - row.p_learned)


def test_learner_at_the_space_optimum_shows_achieved_gain_equal_to_capacity():
    """E large with Delta_gen = -E means the learner already sits at P(G_star)."""
    train = _matrix({STAR: [0.69], HAT: [0.70], LEARNED: [0.70]}, 1)
    test = _matrix({STAR: [0.68], HAT: [0.66], LEARNED: [0.68]}, 1)
    row = decompose_row("opt", 0, 10, LEARNED, train=train, test=test,
                        g_star=STAR, g_hat_n=HAT, p_single=0.60)
    assert row.p_learned == pytest.approx(row.p_star)
    assert row.achieved_gain == pytest.approx(row.capacity)
    assert row.search_value == pytest.approx(-row.estimation)


# ------------------------------------------------------------------- selection

def test_g_hat_n_is_the_training_argmax_and_g_star_the_audit_argmax():
    train = _matrix({STAR: [0.1, 0.2], HAT: [0.9, 0.8], LEARNED: [0.5, 0.5]}, 2)
    audit = _matrix({STAR: [0.9, 0.9], HAT: [0.2, 0.2], LEARNED: [0.5, 0.5]}, 2)
    assert select_g_hat_n(train) == HAT
    assert select_g_star_plugin(audit) == STAR


def test_cross_fitting_sits_below_the_naive_max():
    """Winner's curse: selecting and scoring on the same data is optimistic."""
    rng = np.random.default_rng(0)
    n_graphs, n_tasks = 25, 300
    truth = np.linspace(0.40, 0.60, n_graphs)
    values = (rng.random((n_graphs, n_tasks)) < truth[:, None]).astype(float)
    audit = ScoreMatrix(tuple(f"g{i}" for i in range(n_graphs)),
                        tuple(f"t{j}" for j in range(n_tasks)), values)
    result = cross_fit_capacity(audit, n_splits=20)
    assert result.cross_fitted < result.naive_max
    assert result.selection_bias > 0


def test_capacity_upper_bound_is_above_the_naive_max():
    rng = np.random.default_rng(1)
    values = (rng.random((25, 300)) < 0.5).astype(float)
    audit = ScoreMatrix(tuple(f"g{i}" for i in range(25)),
                        tuple(f"t{j}" for j in range(300)), values)
    assert capacity_upper_bound(audit) > float(np.nanmax(audit.means()))


# ------------------------------------------------------------------- bootstrap

def test_paired_bootstrap_is_much_tighter_than_unpaired_on_correlated_tasks():
    """Pairing is not cosmetic: task difficulty dominates the variance."""
    rng = np.random.default_rng(3)
    difficulty = rng.random(400)
    left = (rng.random(400) < difficulty * 0.9 + 0.05).astype(float)
    right = left.copy()
    flip = rng.random(400) < 0.15
    right[flip] = 1 - right[flip]

    paired = paired_difference_interval(left, right, iterations=400)
    unpaired = paired_task_bootstrap(
        lambda idx: float(np.mean(left[np.asarray(list(idx), int)])
                          - np.mean(right[np.asarray(list(idx), int)[::-1]])),
        400, iterations=400)
    assert paired.width < unpaired.width


def test_paired_se_matches_sqrt_disagreement_over_n_for_binary_scores():
    left = np.array([1.0] * 80 + [0.0] * 20)
    right = np.array([1.0] * 80 + [1.0] * 20)
    q = disagreement_rate(left, right)
    assert q == pytest.approx(0.20)
    assert paired_se(left, right) == pytest.approx(np.sqrt(q * (1 - q) / 99), rel=0.05)


def test_required_tasks_scales_with_the_inverse_square_of_epsilon():
    few = required_tasks(epsilon=0.04, disagreement=0.2)
    many = required_tasks(epsilon=0.02, disagreement=0.2)
    assert many == pytest.approx(4 * few, rel=0.02)


def test_required_tasks_ignores_seed_count():
    """Seeds cannot buy task-level precision in the converged regime.

    The earlier version of this test asserted the opposite and encoded the bug: it
    divided the requirement by the seed count, which understated it by 3-7x.
    """
    import inspect

    assert "n_seeds" not in inspect.signature(required_tasks).parameters
    # Excluding 2 points at q=0.2 needs ~1,100 tasks, not ~22 as 50 seeds implied.
    assert 1_000 <= required_tasks(0.02, 0.2) <= 1_200


def test_interval_disjointness_is_the_e0_criterion():
    a = Interval(point=0.03, low=0.02, high=0.04)
    b = Interval(point=0.005, low=0.00, high=0.01)
    assert a.disjoint_from(b)
    assert not a.disjoint_from(Interval(point=0.02, low=0.015, high=0.03))


def test_repeat_variance_bounds_recycled_sampling_noise():
    values = np.zeros((2, 3, 4))
    values[0, 0] = [1, 0, 1, 0]
    assert repeat_variance(values) > 0
    assert repeat_variance(np.zeros((2, 3, 4))) == pytest.approx(0.0)


# --------------------------------------------------------------------- plateau

def _val_rows(n: int = 300) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    base = (rng.random(n) < 0.5).astype(float)
    better = base.copy()
    better[:40] = 1.0                      # a clearly better graph
    marginal = base.copy()
    marginal[:2] = 1.0                     # better by well under epsilon
    return {"a": base, "b": better, "c": marginal}


def test_frozen_incumbent_is_declared_but_labelled_frozen():
    rows = _val_rows()
    incumbents = {s: {100: "a", 150: "a"} for s in range(5)}
    verdict = assess_plateau("opt", incumbents, rows, step=100, delta_t=50,
                           epsilon=0.02, iterations=200)
    assert verdict.status == FROZEN and verdict.declared
    assert verdict.frozen_seeds == 5
    assert "not evidence of a ceiling" in verdict.describe()


def test_a_real_improvement_prevents_declaring_a_plateau():
    rows = _val_rows()
    incumbents = {s: {100: "a", 150: "b"} for s in range(5)}
    verdict = assess_plateau("opt", incumbents, rows, step=100, delta_t=50,
                           epsilon=0.02, iterations=400)
    assert verdict.status == NOT_DECLARED and not verdict.declared
    assert "not the same as no improvement" in verdict.describe()


def test_a_sub_epsilon_improvement_is_a_plateau_that_moved():
    rows = _val_rows()
    incumbents = {s: {100: "a", 150: "c"} for s in range(20)}
    verdict = assess_plateau("opt", incumbents, rows, step=100, delta_t=50,
                           epsilon=0.05, iterations=400)
    assert verdict.status == MOVED_FLAT and verdict.declared
    assert verdict.frozen_seeds == 0


def test_scan_reports_the_first_declared_step():
    rows = _val_rows()
    incumbents = {s: {50: "a", 100: "b", 150: "b", 200: "b"} for s in range(5)}
    verdicts = scan_plateau("opt", incumbents, rows, steps=[50, 100, 150],
                            delta_t=50, epsilon=0.02, iterations=200)
    first = first_plateau(verdicts)
    assert first is not None and first.step == 100 and first.status == FROZEN


# ----------------------------------------------------------------------- power

def test_epsilon_of_one_point_is_unreachable_at_150_tasks():
    report = power_report(epsilon=0.01, disagreement=0.2, n_tasks=150, n_seeds=5)
    assert not report.sufficient
    assert report.floor_upper_bound > 0.01


def test_seeds_do_not_move_the_floor():
    """A plateau is the converged regime, where every seed holds the same pair.

    Simulation at 300 tasks, true improvement zero: the bootstrap's 95% upper bound is
    3.88 points at 20 seeds and 4.14 at 50 when seeds share a pair, against a formula
    that claimed 0.95 and 0.60. The floor is what the design must clear.
    """
    few = power_report(0.02, 0.2, n_tasks=300, n_seeds=5)
    many = power_report(0.02, 0.2, n_tasks=300, n_seeds=50)
    assert few.floor_upper_bound == pytest.approx(many.floor_upper_bound)
    assert many.optimistic_upper_bound < many.floor_upper_bound
    assert not many.sufficient      # 3.8 points of floor against a 2-point epsilon


def test_two_points_needs_about_eleven_hundred_tasks():
    assert not power_report(0.02, 0.2, n_tasks=300, n_seeds=20).sufficient
    assert power_report(0.02, 0.2, n_tasks=1200, n_seeds=20).sufficient


def test_floor_matches_the_closed_form():
    report = power_report(0.02, 0.2, n_tasks=300, n_seeds=20)
    expected = 1.6449 * np.sqrt(0.2 * 0.8 / 300)
    assert report.floor_upper_bound == pytest.approx(expected, rel=1e-3)
    assert report.floor_upper_bound == pytest.approx(0.038, abs=0.002)


def test_widening_the_split_is_the_only_lever_on_the_floor():
    narrow = power_report(0.02, 0.2, n_tasks=300, n_seeds=50)
    wide = power_report(0.02, 0.2, n_tasks=1200, n_seeds=5)
    assert wide.floor_upper_bound < narrow.floor_upper_bound


def test_discriminability_reports_gap_and_noise_without_a_threshold():
    rows = _val_rows()
    diagnostic = discriminability(rows["b"], rows["a"])
    assert diagnostic["paired_diff"] > 0
    assert diagnostic["se"] > 0
    assert set(diagnostic) == {"paired_diff", "se", "ratio", "disagreement"}


# ------------------------------------------------------------------ trajectory

def test_decompose_trajectory_builds_one_row_per_seed_and_step():
    train = _matrix({STAR: [0.69], HAT: [0.70], LEARNED: [0.67]}, 1)
    audit = _matrix({STAR: [0.80], HAT: [0.60], LEARNED: [0.55]}, 1)
    test = _matrix({STAR: [0.68], HAT: [0.66], LEARNED: [0.65]}, 1)
    incumbents = {0: {10: LEARNED, 20: HAT}, 1: {10: LEARNED, 20: LEARNED}}
    rows = decompose_trajectory("opt", incumbents, train=train, audit=audit,
                                test=test, p_single=0.60)
    assert len(rows) == 4
    assert all(abs(r.residual()) < IDENTITY_TOLERANCE for r in rows)
    assert {r.g_star for r in rows} == {STAR}


def test_ladder_levels_move_only_the_learned_line():
    train = _matrix({STAR: [0.69], HAT: [0.70], LEARNED: [0.67]}, 1)
    audit = _matrix({STAR: [0.80], HAT: [0.60], LEARNED: [0.55]}, 1)
    test = _matrix({STAR: [0.68], HAT: [0.66], LEARNED: [0.65]}, 1)
    incumbents = {0: {10: LEARNED, 20: HAT}}
    rows = decompose_trajectory("opt", incumbents, train=train, audit=audit,
                                test=test, p_single=0.60)
    ladder = ladder_levels(rows)
    assert ladder["steps"] == [10, 20]
    assert ladder["p_learned"] == [pytest.approx(0.65), pytest.approx(0.66)]
    assert ladder["p_star"] == pytest.approx(0.68)
    assert ladder["p_single"] == pytest.approx(0.60)


# ------------------------------------------------------- the inversion branch (#3)

def _iv(point: float, low: float, high: float) -> Interval:
    return Interval(point=point, low=low, high=high)


def test_prescribe_keeps_searching_when_search_value_is_positive():
    out = prescribe(capacity=_iv(0.08, 0.05, 0.11), estimation=_iv(0.02, 0.00, 0.04),
                    search_value=_iv(0.03, 0.01, 0.05),
                    gap_to_optimum=_iv(0.05, 0.03, 0.07), epsilon=0.02)
    assert out["action"] == "keep_searching"
    assert out["searching_helps"]


def test_prescribe_does_not_prescribe_adding_tasks_when_the_gap_is_closed():
    """The failure the naive 'largest term wins' reading produces.

    search_value = -E means the learner declined an argmax that had overfitted the
    training split, and E + search_value = 0, so it already sits at P(G_star). E is
    large and completely unactionable.
    """
    out = prescribe(capacity=_iv(0.08, 0.05, 0.11), estimation=_iv(0.05, 0.03, 0.07),
                    search_value=_iv(-0.05, -0.07, -0.03),
                    gap_to_optimum=_iv(0.00, -0.01, 0.01), epsilon=0.02)
    assert out["action"] == "stop_space_delivered"
    assert out["learner_at_or_past_empirical_max"]
    assert not out["estimation_actionable"]


def test_prescribe_adds_tasks_when_search_is_exhausted_but_a_gap_remains():
    out = prescribe(capacity=_iv(0.10, 0.07, 0.13), estimation=_iv(0.06, 0.04, 0.08),
                    search_value=_iv(-0.01, -0.03, 0.01),
                    gap_to_optimum=_iv(0.05, 0.03, 0.07), epsilon=0.02)
    assert out["action"] == "add_tasks"
    assert out["estimation_actionable"]
    assert not out["searching_helps"]


def test_prescribe_stops_when_capacity_cannot_clear_epsilon():
    out = prescribe(capacity=_iv(0.01, -0.01, 0.015), estimation=_iv(0.05, 0.03, 0.07),
                    search_value=_iv(0.04, 0.02, 0.06),
                    gap_to_optimum=_iv(0.09, 0.07, 0.11), epsilon=0.02)
    assert out["action"] == "stop_or_enlarge_space"
    assert "epsilon" in out["why"]


def test_prescribe_handles_a_negative_capacity():
    """C < 0 is a finding -- the space loses to pi_single at matched compute."""
    out = prescribe(capacity=_iv(-0.03, -0.05, -0.01), estimation=_iv(0.01, 0.0, 0.02),
                    search_value=_iv(0.01, 0.0, 0.02),
                    gap_to_optimum=_iv(0.02, 0.0, 0.04), epsilon=0.02)
    assert out["action"] == "stop_or_enlarge_space"


def test_prescribe_is_inconclusive_rather_than_guessing():
    out = prescribe(capacity=_iv(0.10, 0.07, 0.13), estimation=_iv(0.02, -0.01, 0.05),
                    search_value=_iv(0.01, -0.02, 0.04),
                    gap_to_optimum=_iv(0.05, 0.02, 0.08), epsilon=0.02)
    assert out["action"] == "inconclusive"
