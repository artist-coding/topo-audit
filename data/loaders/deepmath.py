"""DeepMath-103K: one difficulty band, the same prompt and scorer as MATH.

The pool behind the pending main-cell decision (see CLAUDE.md): MATH Level 4-5 is
saturated for the default-thinking model, and DeepMath's continuous ``difficulty``
field lets the band be chosen so single-agent accuracy lands inside the gate.

Bands are half-open, [low, low + width), except the top band, which keeps its upper
edge. Closed intervals put every problem rated exactly on a boundary into two bands.
"""

from __future__ import annotations

from typing import Iterator

import pandas as pd

from data.loaders.math import PROMPT_TEMPLATE, Task, task_id_for

DEFAULT_RAW = "data/raw/deepmath.parquet"
BAND_WIDTH = 0.5


def in_band(difficulty: pd.Series, low: float, width: float = BAND_WIDTH) -> pd.Series:
    high = low + width
    upper = difficulty <= high if high >= difficulty.max() else difficulty < high
    return (difficulty >= low) & upper


def load_band(path: str, low: float, width: float = BAND_WIDTH) -> pd.DataFrame:
    """Problems of one band with ``task_id``, ``prompt``, ``answer_str``, deduplicated."""
    frame = pd.read_parquet(path)
    frame = frame[in_band(frame["difficulty"], low, width)].copy()
    answers = frame["final_answer"].astype(str).str.strip().str.lower()
    # Use the same non-binary pool for P1, P2 and the four experiment splits.
    # Binary-answer prevalence varies with difficulty and confounds the band gate.
    frame = frame[frame["final_answer"].notna()
                  & ~answers.isin({"", "yes", "no", "true", "false"})].copy()
    frame["task_id"] = frame["question"].map(task_id_for)
    frame = frame.drop_duplicates(subset="task_id", keep="first")
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
