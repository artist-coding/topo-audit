"""Acceptance tests for build step 2 (data/).

Two acceptance criteria from the framework guide:
  - the four roles never overlap and re-running the split script is byte-identical
  - 20 hand-checked answers all score correctly

The 20 cases below were read off real MATH Level 4-5 solutions (integers, fractions,
tuples, radicals, percentages, degrees, expressions) plus the failure shapes we
measured on the gateway: a generation truncated mid-reasoning returns empty content.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from data.loaders.math import (
    answers_match,
    extract_boxed,
    extract_prediction,
    load_pool,
    normalize_answer,
    score_output,
    task_id_for,
)

SPLITS = pathlib.Path("data/splits/math_v1.json")
RAW = pathlib.Path("data/raw/math_hendrycks.parquet")

# ------------------------------------------------- 20 hand-checked scoring cases
# (model output, gold answer, expected score, note)
HAND_CHECKED: list[tuple[str, str, float, str]] = [
    (r"so the total is \boxed{5040}. FINAL: 5040", "5040", 1.0, "plain integer"),
    (r"FINAL: \boxed{192}", "192", 1.0, "boxed inside FINAL line"),
    (r"\boxed{\dfrac{49}{8}}", r"\frac{49}{8}", 1.0, "dfrac vs frac"),
    (r"\boxed{\left( \frac{5}{4}, \frac{7}{4} \right)}", r"(\frac{5}{4},\frac{7}{4})",
     1.0, "left/right and spaces stripped"),
    (r"\boxed{4\%}", "4", 1.0, "percent sign"),
    (r"\boxed{\frac12}", r"\frac{1}{2}", 1.0, "short frac form"),
    (r"\boxed{\sqrt2}", r"\sqrt{2}", 1.0, "short sqrt form"),
    (r"\boxed{007}", "7", 1.0, "leading zeros"),
    (r"\boxed{1,000}", "1000", 1.0, "thousands separator"),
    (r"\boxed{\text{3}}", "3", 1.0, "text wrapper"),
    (r"\boxed{$17$}", "17", 1.0, "dollar wrappers"),
    (r"\boxed{+5}", "5", 1.0, "explicit plus"),
    (r"\boxed{90^\circ}", "90", 1.0, "degree marker"),
    (r"\boxed{0.5}", r"\frac{1}{2}", 1.0, "decimal vs fraction (sympy)"),
    (r"\boxed{2/4}", r"\frac{1}{2}", 1.0, "unreduced fraction (sympy)"),
    (r"\boxed{\frac{\sqrt{13}+1}{2}}", r"\frac{1 + \sqrt{13}}{2}", 1.0,
     "reordered sum (sympy)"),
    (r"\boxed{5041}", "5040", 0.0, "off by one"),
    (r"\boxed{(1,2)}", "(2,1)", 0.0, "tuple order matters"),
    (r"\boxed{8n^2+4n+1}", "8n^2 + 4n + 1", 1.0, "expression, spacing only"),
    (r"\boxed{725}", "841", 0.0, "wrong integer"),
]


@pytest.mark.parametrize("output,gold,expected,note", HAND_CHECKED,
                         ids=[c[3] for c in HAND_CHECKED])
def test_hand_checked_scoring(output: str, gold: str, expected: float, note: str):
    result = score_output(output, gold)
    assert not result.parse_fail, f"{note}: should have parsed"
    assert result.score == expected, f"{note}: pred={result.prediction!r}"


# ------------------------------------------------------------- parse failures

def test_truncated_generation_is_parse_fail_not_a_wrong_answer():
    """The measured failure mode: thinking eats the cap, content comes back empty."""
    result = score_output("", "5040")
    assert result.parse_fail and result.score == 0.0 and result.prediction is None


def test_reasoning_without_an_answer_is_parse_fail():
    text = "Let me compute. First, 17 times 23. We can write 17*23 = 17*20 + 17*3"
    result = score_output(text, "391")
    assert result.parse_fail and result.score == 0.0


def test_whitespace_only_is_parse_fail():
    assert score_output("   \n  ", "1").parse_fail


# ---------------------------------------------------------------- extraction

def test_last_boxed_wins():
    assert extract_boxed(r"first \boxed{1} then \boxed{2}") == "2"


def test_nested_braces_are_balanced():
    assert extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"


def test_final_line_without_boxed():
    assert extract_prediction("reasoning here\nFINAL: 42") == "42"


def test_boxed_preferred_over_final_line():
    assert extract_prediction(r"\boxed{42}\nFINAL: something else") == "42"


def test_normalize_keeps_distinct_answers_distinct():
    assert normalize_answer("10") != normalize_answer("1")
    assert normalize_answer("0.5") != normalize_answer("5")


def test_normalize_is_idempotent():
    for case in (r"\dfrac{1}{2}", r"\left(1,2\right)", "007", r"90^\circ"):
        once = normalize_answer(case)
        assert normalize_answer(once) == once


def test_answers_match_handles_none():
    assert not answers_match(None, "1")


# -------------------------------------------------------------------- splits

@pytest.fixture(scope="module")
def splits() -> dict:
    if not SPLITS.exists():
        pytest.skip("run data/make_splits.py first")
    return json.loads(SPLITS.read_text())


def test_four_roles_have_the_declared_sizes(splits):
    assert splits["sizes"] == {"train": 256, "audit": 300, "val": 300, "test": 1000}


def test_roles_do_not_overlap(splits):
    sets = {role: set(ids) for role, ids in splits["splits"].items()}
    for role, ids in sets.items():
        assert len(ids) == len(splits["splits"][role]), f"{role} has duplicates"
    roles = list(sets)
    for i, a in enumerate(roles):
        for b in roles[i + 1:]:
            assert not sets[a] & sets[b], f"{a} and {b} overlap"


def test_test_split_contains_the_whole_math500_anchor(splits):
    anchor = set(splits["test_anchor"]["task_ids"])
    assert len(anchor) == 262
    assert anchor <= set(splits["splits"]["test"])


def test_train_audit_val_come_from_the_training_pool_only(splits):
    """Nothing in train/audit/val may come from the MATH test split."""
    if not RAW.exists():
        pytest.skip("raw parquet not downloaded")
    test_pool = set(load_pool(RAW, "test", (4, 5))["task_id"])
    for role in ("train", "audit", "val"):
        assert not set(splits["splits"][role]) & test_pool, f"{role} leaks test-pool items"


def test_every_task_has_an_answer_and_a_level(splits):
    all_ids = [tid for ids in splits["splits"].values() for tid in ids]
    assert set(all_ids) == set(splits["answers"])
    assert set(all_ids) == set(splits["levels_by_task"])
    assert set(splits["levels_by_task"].values()) == {4, 5}


def test_task_id_is_stable_under_whitespace():
    assert task_id_for("  2+2=? ") == task_id_for("2+2=?")


# --------------------------------------------- the scorer must be time-bounded (#8)

def test_scorer_is_bounded_on_adversarial_latex():
    """score_output runs once per tensor cell, so it may never run unbounded.

    Measured failure: a generation containing a LaTeX relation sent sympy's
    parse_latex into Relational.__new__ -> is_eq -> evalf_integral -> mpmath
    quadrature *inside the parser*, and a probe burned 43 minutes of CPU with no
    progress. The grammar is now restricted until evaluation is bounded by
    construction -- no subprocess, no timeout, nothing that can loop.
    """
    import time

    from data.loaders.math import _latex_to_arithmetic

    adversarial = [
        r"\int_0^1 \frac{\sin(x^x)}{\log(1+x^{1/3})}\,dx = \frac{\pi}{4}",
        r"\sum_{n=1}^\infty \frac{1}{n^s} < \infty",
        r"\lim_{x\to 0}\frac{\tan(\sin x)-\sin(\tan x)}{x^7}",
        r"x = \oint_C \frac{dz}{z^2+1}",
        r"\begin{cases} 1 & x>0 \\ 0 & x\le 0 \end{cases}",
    ]
    started = time.monotonic()
    for text in adversarial:
        assert _latex_to_arithmetic(text) is None, f"should reject {text[:34]!r}"
        assert score_output(f"FINAL: {text}", "1").score == 0.0
    assert time.monotonic() - started < 0.5


def test_restricted_grammar_still_decides_what_it_exists_for():
    """The three equivalences that motivated a fallback at all."""
    assert answers_match("0.5", r"\frac{1}{2}")
    assert answers_match("2/4", r"\frac{1}{2}")
    assert answers_match(r"\frac{\sqrt{13}+1}{2}", r"\frac{1 + \sqrt{13}}{2}")
    assert not answers_match("0.6", r"\frac{1}{2}")


def test_symbolic_answers_are_handled_by_normalisation_not_evaluation():
    from data.loaders.math import _latex_to_arithmetic

    assert _latex_to_arithmetic("8n^2+4n+1") is None   # has a free symbol
    assert answers_match("8n^2+4n+1", "8n^2 + 4n + 1")  # but normalises equal


def test_evaluator_admits_only_arithmetic():
    from data.loaders.math import _evaluate

    assert _evaluate("((1)/(2))") == pytest.approx(0.5)
    assert _evaluate("pi") == pytest.approx(3.14159, abs=1e-4)
    assert _evaluate("__import__('os').system('true')") is None
    assert _evaluate("open('x')") is None
    assert _evaluate("1/0") is None


def test_long_answers_are_rejected_before_parsing():
    from data.loaders.math import _NUMERIC_MAX_CHARS, _latex_to_arithmetic

    assert _latex_to_arithmetic("1+" * _NUMERIC_MAX_CHARS) is None


def test_repeated_comparisons_are_cached():
    from data.loaders.math import _numerically_equal

    _numerically_equal.cache_clear()
    _numerically_equal("2/4", r"\frac{1}{2}")
    before = _numerically_equal.cache_info().hits
    _numerically_equal("2/4", r"\frac{1}{2}")
    assert _numerically_equal.cache_info().hits == before + 1
