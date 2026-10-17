"""DeepMath pool: half-open bands, a disjoint split from one band, and task loading.

pyarrow is not needed: read_parquet is replaced by a synthetic frame.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from data.loaders.deepmath import in_band, load_band
from data.make_splits import build_deepmath
from data.tasks import load_split_tasks


def _frame(n: int = 400) -> pd.DataFrame:
    return pd.DataFrame([{"question": f"Q{i}: compute {i} * 2.", "final_answer": str(2 * i),
                          "difficulty": (8.0, 8.5, 9.0, 9.5)[i % 4], "topic": "algebra"}
                         for i in range(n)])


@pytest.fixture
def fake_parquet(monkeypatch):
    frame = _frame()
    monkeypatch.setattr(pd, "read_parquet", lambda path, *a, **k: frame.copy())
    return frame


def test_bands_are_half_open_except_the_top():
    difficulty = pd.Series([8.0, 8.5, 9.0, 9.5, 10.0])
    assert list(in_band(difficulty, 8.5)) == [False, True, False, False, False]
    assert list(in_band(difficulty, 9.5)) == [False, False, False, True, True]


def test_split_is_disjoint_deterministic_and_keeps_p2_problems_out(fake_parquet):
    band_ids = set(load_band("x", 8.5)["task_id"])
    assert len(band_ids) == 100                   # only the 8.5 problems, not 9.0
    excluded = frozenset(sorted(band_ids)[:10])
    first = build_deepmath(1, 20, 20, 20, 20, 8.5, raw="x", exclude=excluded)
    assert first == build_deepmath(1, 20, 20, 20, 20, 8.5, raw="x", exclude=excluded)
    ids = [t for role_ids in first["splits"].values() for t in role_ids]
    assert len(ids) == len(set(ids)) == 80
    assert set(ids) <= band_ids and not set(ids) & excluded
    assert first["pool_sizes"] == {"band_pool": 100, "excluded_p2": 10, "usable": 90}
    assert first["family"] == "deepmath" and first["band"] == [8.5, 9.0]


def test_split_resolves_to_tasks_with_the_recorded_answers(fake_parquet, tmp_path):
    payload = build_deepmath(1, 10, 10, 10, 10, 8.5, raw="x")
    path = tmp_path / "split.json"
    path.write_text(json.dumps(payload))
    tasks = load_split_tasks({"task": {"splits": str(path), "raw": "x"}}, ["train", "test"])
    assert len(tasks["train"]) == 10 and len(tasks["test"]) == 10
    task = tasks["test"][0]
    assert task.family == "deepmath" and task.split_role == "test"
    assert task.answer == payload["answers"][task.task_id]
    assert "Solve the following competition mathematics problem." in task.prompt


def test_a_band_too_small_for_the_roles_fails_loudly(fake_parquet):
    with pytest.raises(SystemExit, match="usable problems"):
        build_deepmath(1, 50, 50, 50, 50, 8.5, raw="x")
