"""MATH loader and scorer.

Answers come out of ``\\boxed{...}`` (100% of Level 4-5 training solutions have one).
Scoring is exact match after normalisation, with a bounded numeric fallback for forms
that are written differently but equal (``\\frac{1}{2}`` vs ``0.5``, ``2/4``, reordered
sums of closed-form numbers); see the note above ``_LATEX_TO_PY`` for why it is no
longer sympy.

A model answer that cannot be parsed scores 0 and is flagged ``parse_fail`` in the
RunRecord, so truncated generations are counted as a defect rather than silently
becoming wrong answers. Truncation rate is an acceptance metric, not a score.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import math
import re
from dataclasses import dataclass
from typing import Iterator, Literal

import pandas as pd

SplitRole = Literal["train", "audit", "val", "test"]

FINAL_RE = re.compile(r"FINAL\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE | re.DOTALL)


@dataclass(frozen=True, slots=True)
class Task:
    task_id: str
    family: str
    split_role: SplitRole
    prompt: str
    answer: str
    meta: dict

    @property
    def level(self) -> int | None:
        return self.meta.get("level")


# ------------------------------------------------------------------ extraction

def extract_boxed(text: str) -> str | None:
    """Content of the last ``\\boxed{...}``, brace-balanced."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    start = text.find("{", idx)
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i]
    return None


def extract_prediction(output: str) -> str | None:
    """Pull the answer out of a node's generation.

    Prefers ``\\boxed{...}``, then a ``FINAL:`` line. Returns None when neither is
    present, which is the common shape of a generation truncated mid-reasoning.
    """
    if not output or not output.strip():
        return None
    boxed = extract_boxed(output)
    if boxed is not None:
        return boxed
    match = FINAL_RE.search(output)
    if match:
        tail = match.group(1).strip()
        inner = extract_boxed(tail)
        return inner if inner is not None else (tail.splitlines()[0].strip() or None)
    return None


# ----------------------------------------------------------------- normalising

_STRIP_WRAPPERS = ("\\left", "\\right", "\\!", "\\,", "\\;", "\\ ", "\\$", "$", " ")
_UNITS_RE = re.compile(r"\\text\s*\{([^}]*)\}")
_MBOX_RE = re.compile(r"\\mbox\s*\{([^}]*)\}")
_SQRT_RE = re.compile(r"\\sqrt(?!\[)\s*([0-9a-zA-Z])")
_FRAC_SHORT_RE = re.compile(r"\\frac\s*([0-9a-zA-Z])\s*([0-9a-zA-Z])")


def normalize_answer(ans: str | None) -> str | None:
    """Canonical string form. Conservative: never changes mathematical content."""
    if ans is None:
        return None
    s = ans.strip()
    if not s:
        return None
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = _UNITS_RE.sub(r"\1", s)
    s = _MBOX_RE.sub(r"\1", s)
    s = _SQRT_RE.sub(r"\\sqrt{\1}", s)          # \sqrt2 -> \sqrt{2}
    s = _FRAC_SHORT_RE.sub(r"\\frac{\1}{\2}", s)  # \frac12 -> \frac{1}{2}
    for token in _STRIP_WRAPPERS:
        s = s.replace(token, "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    s = s.rstrip(".")
    s = s.replace("^\\circ", "").replace("^{\\circ}", "")
    s = re.sub(r"\\%|%", "", s)
    if s.startswith("+"):
        s = s[1:]
    # Order matters: drop thousands separators before touching zeros, or "1,000"
    # loses its zeros to the leading-zero rule and becomes "10".
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    s = re.sub(r"^(-?)0+(?=\d)", r"\1", s)  # 007 -> 7; keeps 0.5, 10, 100
    return s or None


#: Why there is no symbolic fallback here any more.
#:
#: Measured, not hypothesised: a generation containing a LaTeX relation sent sympy's
#: `parse_latex` into `convert_relation` -> `Relational.__new__` -> `is_eq` ->
#: `as_real_imag` -> `evalf_integral` -> mpmath quadrature, *inside the parser*, and a
#: probe burned 43 minutes of CPU with no progress. `score_output` runs once per tensor
#: cell -- over 100,000 times in a sweep -- so nothing unbounded may live here.
#:
#: A subprocess with a timeout was the first fix, but the Executor scores inside a
#: thread pool and forking from a multi-threaded process risks deadlocking on a lock
#: another thread holds. So instead the grammar is restricted until the computation is
#: bounded by construction: the three equivalences the fallback existed for --
#: 0.5 vs \frac{1}{2}, 2/4 vs \frac{1}{2}, and \frac{\sqrt{13}+1}{2} vs
#: \frac{1+\sqrt{13}}{2} -- are all closed-form numbers, so evaluating both sides
#: numerically decides them without any symbolic machinery. Symbolic answers such as
#: 8n^2+4n+1 are already handled by normalisation and never reach this point.
_LATEX_TO_PY = (
    (r"\cdot", "*"), (r"\times", "*"), (r"\div", "/"), (r"\pi", "pi"),
)
#: Characters admissible after translation. Anything else means "cannot decide".
_NUMERIC_CHARS = set("0123456789.+-*/()^ pie")
_NUMERIC_MAX_CHARS = 80
_NUMERIC_TOLERANCE = 1e-9


def _latex_to_arithmetic(text: str) -> str | None:
    r"""Rewrite a restricted LaTeX subset as a Python arithmetic expression.

    Handles \frac{a}{b} and \sqrt{a} with balanced braces, plus the few operator
    aliases above. Returns None the moment anything outside the subset appears, which
    is what keeps evaluation bounded.
    """
    if len(text) > _NUMERIC_MAX_CHARS:
        return None
    out, i = [], 0
    while i < len(text):
        if text.startswith(r"\frac", i):
            first, rest = _take_group(text, i + len(r"\frac"))
            if first is None:
                return None
            second, rest = _take_group(text, rest)
            if second is None:
                return None
            inner_a, inner_b = _latex_to_arithmetic(first), _latex_to_arithmetic(second)
            if inner_a is None or inner_b is None:
                return None
            out.append(f"(({inner_a})/({inner_b}))")
            i = rest
            continue
        if text.startswith(r"\sqrt", i):
            group, rest = _take_group(text, i + len(r"\sqrt"))
            if group is None:
                return None
            inner = _latex_to_arithmetic(group)
            if inner is None:
                return None
            out.append(f"(({inner})**0.5)")
            i = rest
            continue
        for token, replacement in _LATEX_TO_PY:
            if text.startswith(token, i):
                out.append(replacement)
                i += len(token)
                break
        else:
            if text[i] == "\\" or text[i] in "{}":
                return None
            out.append(text[i])
            i += 1
    expression = "".join(out).replace("^", "**")
    if set(expression) - _NUMERIC_CHARS - set("*"):
        return None
    return expression


def _take_group(text: str, i: int) -> tuple[str | None, int]:
    """Contents of the brace group starting at `i`, and the index just past it."""
    if i >= len(text) or text[i] != "{":
        return None, i
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j], j + 1
    return None, i


_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
)
_ALLOWED_NAMES = {"pi": math.pi, "e": math.e}


def _evaluate(expression: str) -> float | None:
    """Evaluate the translated expression, or None if it is not plain arithmetic.

    The AST walk is the bound: only arithmetic nodes and two constants survive, so
    there is no call, no name lookup and nothing that can loop.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return None
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_NAMES:
            return None
    try:
        value = eval(compile(tree, "<answer>", "eval"), {"__builtins__": {}},
                     dict(_ALLOWED_NAMES))
    except (ArithmeticError, ValueError, TypeError):
        return None
    return float(value) if isinstance(value, (int, float)) else None


@functools.lru_cache(maxsize=65536)
def _numerically_equal(a: str, b: str) -> bool:
    """Whether two answers denote the same number. Bounded by construction."""
    left, right = _latex_to_arithmetic(a), _latex_to_arithmetic(b)
    if left is None or right is None:
        return False
    x, y = _evaluate(left), _evaluate(right)
    if x is None or y is None:
        return False
    return abs(x - y) <= _NUMERIC_TOLERANCE * max(1.0, abs(x), abs(y))


def answers_match(pred: str | None, gold: str) -> bool:
    np_, ng = normalize_answer(pred), normalize_answer(gold)
    if np_ is None or ng is None:
        return False
    if np_ == ng:
        return True
    if np_.lower() == ng.lower():
        return True
    return _numerically_equal(np_, ng)


@dataclass(frozen=True, slots=True)
class Score:
    score: float
    parse_fail: bool
    prediction: str | None


def score_output(output: str, gold: str) -> Score:
    pred = extract_prediction(output)
    if pred is None:
        return Score(0.0, True, None)
    return Score(float(answers_match(pred, gold)), False, pred)


# -------------------------------------------------------------------- loading

PROMPT_TEMPLATE = (
    "Solve the following competition mathematics problem.\n\n"
    "{problem}\n\n"
    "Put your final answer in \\boxed{{}} and end your reply with a line of the form\n"
    "FINAL: <answer>"
)


def task_id_for(problem: str) -> str:
    """Stable id from the problem text, so dedup and splits survive re-download."""
    return hashlib.sha256(problem.strip().encode()).hexdigest()[:16]


def _level_int(value) -> int | None:
    if isinstance(value, (int, float)) and value == value:
        return int(value)
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else None


def load_pool(
    path: str,
    source_split: str | None = None,
    levels: tuple[int, ...] = (4, 5),
) -> pd.DataFrame:
    """Load a MATH parquet into a frame with ``task_id``, ``prompt``, ``answer``.

    Rows whose level cannot be parsed (``Level ?``) or whose solution has no
    ``\\boxed`` are dropped and counted by the caller.
    """
    df = pd.read_parquet(path)
    if source_split is not None and "source_split" in df.columns:
        df = df[df["source_split"] == source_split]
    df = df.copy()
    df["level_int"] = df["level"].map(_level_int)
    df = df[df["level_int"].isin(levels)]
    if "answer" in df.columns:
        df["answer_str"] = df["answer"].astype(str)
    else:
        df["answer_str"] = df["solution"].map(extract_boxed)
    df = df[df["answer_str"].notna() & (df["answer_str"].astype(str) != "")]
    df["task_id"] = df["problem"].map(task_id_for)
    df = df.drop_duplicates(subset="task_id", keep="first")
    df["prompt"] = df["problem"].map(lambda p: PROMPT_TEMPLATE.format(problem=p.strip()))
    return df.reset_index(drop=True)


def tasks_from_frame(df: pd.DataFrame, split_role: SplitRole) -> Iterator[Task]:
    for row in df.itertuples():
        yield Task(
            task_id=row.task_id,
            family="math",
            split_role=split_role,
            prompt=row.prompt,
            answer=str(row.answer_str),
            meta={
                "level": int(row.level_int),
                "subject": getattr(row, "subject", None),
                "type": getattr(row, "type", None),
            },
        )
