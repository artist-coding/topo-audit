"""DeepMath-103K: one difficulty band, the same prompt and scorer as MATH.

The pool behind the pending main-cell decision (see CLAUDE.md): MATH Level 4-5 is
saturated for the default-thinking model, and DeepMath's continuous ``difficulty``
field lets the band be chosen so single-agent accuracy lands inside the gate.

Bands are half-open, [low, low + width), except the top band, which keeps its upper
edge. Closed intervals put every problem rated exactly on a boundary into two bands.
"""

from __future__ import annotations

import math
from typing import Iterator

import pandas as pd

from data.loaders.math import PROMPT_TEMPLATE, Task, extract_boxed, normalize_answer, task_id_for

DEFAULT_RAW = "data/raw/deepmath.parquet"
BAND_WIDTH = 0.5
ANSWER_FILTER = "normalized_nonbinary_v2"
CATEGORICAL_ANSWERS = frozenset({
    "yes", "no", "true", "false", "converges", "diverges", "convergent", "divergent",
})


def answer_key(answer: object) -> str:
    """Normalize labels for filtering/profiling without changing the scoring gold."""
    if pd.isna(answer):
        return ""
    text = str(answer)
    boxed = extract_boxed(text)
    normalized = normalize_answer(text if boxed is None else boxed)
    return (normalized or "").casefold().strip().rstrip(".!?,;:")


def _band_frame(path: str, low: float, width: float) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame = frame[in_band(frame["difficulty"], low, width)].copy()
    frame["task_id"] = frame["question"].map(task_id_for)
    return frame.drop_duplicates(subset="task_id", keep="first")


def answer_profile(path: str, low: float, width: float = BAND_WIDTH, top: int = 20) -> dict:
    """Inspect frequent normalized answers before choosing a pool or making splits.

    The explicit label list cannot detect every two-choice question. Frequencies
    expose other labels for review; frequent numeric answers are never auto-excluded.
    """
    frame = _band_frame(path, low, width)
    keys = frame["final_answer"].map(answer_key)
    keep = keys.ne("") & ~keys.isin(CATEGORICAL_ANSWERS)
    return {"band": [low, low + width], "answer_filter": ANSWER_FILTER,
            "total": len(frame), "kept": int(keep.sum()),
            "excluded": int((~keep).sum()),
            "top_answers_before": keys.value_counts().head(top).to_dict(),
            "top_answers_after": keys[keep].value_counts().head(top).to_dict()}


def in_band(difficulty: pd.Series, low: float, width: float = BAND_WIDTH) -> pd.Series:
    if not math.isfinite(low) or not math.isfinite(width) or width <= 0:
        raise ValueError("band lower bound must be finite and width positive and finite")
    high = low + width
    upper = difficulty <= high if high >= difficulty.max() else difficulty < high
    return (difficulty >= low) & upper


def load_band(path: str, low: float, width: float = BAND_WIDTH) -> pd.DataFrame:
    """Problems of one band with ``task_id``, ``prompt``, ``answer_str``, deduplicated."""
    frame = _band_frame(path, low, width)
    answers = frame["final_answer"].map(answer_key)
    # Use the same non-binary pool for P1, P2 and the four experiment splits.
    # Binary-answer prevalence varies with difficulty and confounds the band gate.
    frame = frame[answers.ne("") & ~answers.isin(CATEGORICAL_ANSWERS)].copy()
    frame["answer_str"] = frame["final_answer"].astype(str)
    frame["prompt"] = frame["question"].map(
        lambda q: PROMPT_TEMPLATE.format(problem=q.strip()))
    return frame.reset_index(drop=True)


def tasks_from_band_frame(frame: pd.DataFrame, split_role: str) -> Iterator[Task]:
    for row in frame.itertuples():
        yield Task(task_id=row.task_id, family="deepmath", split_role=split_role,
                   prompt=row.prompt, answer=str(row.answer_str),
                   meta={"difficulty": float(row.difficulty),
                         "topic": getattr(row, "topic", None)})


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Inspect DeepMath answer frequencies without generating splits")
    parser.add_argument("--raw", default=DEFAULT_RAW)
    parser.add_argument("--band", type=float, required=True)
    parser.add_argument("--band-width", type=float, default=BAND_WIDTH)
    args = parser.parse_args()
    print(json.dumps(answer_profile(args.raw, args.band, args.band_width), indent=2,
                     ensure_ascii=False))
