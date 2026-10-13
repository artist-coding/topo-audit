"""MATH loader and scorer.

Answers come out of ``\\boxed{...}`` (100% of Level 4-5 training solutions have one).
Scoring is exact match after normalisation, with a sympy fallback for forms that are
written differently but equal (``\\frac{1}{2}`` vs ``0.5``, ``2/4``, reordered sums).

A model answer that cannot be parsed scores 0 and is flagged ``parse_fail`` in the
RunRecord, so truncated generations are counted as a defect rather than silently
becoming wrong answers. Truncation rate is an acceptance metric, not a score.
"""

from __future__ import annotations

import hashlib
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


def _sympy_equal(a: str, b: str) -> bool:
    try:
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
    except Exception:
        return False
    try:
        diff = simplify(parse_latex(a) - parse_latex(b))
        return bool(diff == 0)
    except Exception:
        return False


def answers_match(pred: str | None, gold: str) -> bool:
    np_, ng = normalize_answer(pred), normalize_answer(gold)
    if np_ is None or ng is None:
        return False
    if np_ == ng:
        return True
    if np_.lower() == ng.lower():
        return True
    return _sympy_equal(np_, ng)


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
