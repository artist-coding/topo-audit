"""DeepMath filtering, profiling, disjoint splits and probe compatibility."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from data.loaders.deepmath import ANSWER_FILTER, answer_profile, in_band, load_band
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
    assert first["answer_filter"] == ANSWER_FILTER == "normalized_nonbinary_v2"


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


def test_nonbinary_pool_is_shared_by_probe_and_splits(tmp_path):
    from scripts.p2_spread import probe_tasks

    frame = _frame(80)
    excluded = [" Yes ", "NO", "true", " False\n", "", None]
    frame = pd.concat([frame, pd.DataFrame([
        {"question": f"binary {i}", "final_answer": answer, "difficulty": 8.5}
        for i, answer in enumerate(excluded)])], ignore_index=True)
    path = tmp_path / "deepmath.parquet"
    frame.to_parquet(path)
    pool = load_band(path, 8.5)
    assert len(pool) == 20
    assert not pool.question.str.startswith("binary").any()
    probe = probe_tasks({"task": {"deepmath_raw": str(path)}}, "deepmath", 4, 7, 8.5)
    excluded_ids = frozenset(t.task_id for t in probe)
    split = build_deepmath(7, 4, 4, 4, 4, 8.5, raw=str(path), exclude=excluded_ids)
    split_ids = {t for ids in split["splits"].values() for t in ids}
    assert split_ids | excluded_ids == set(pool.task_id)
    assert not split_ids & excluded_ids


def test_normalized_labels_are_excluded_but_math_gold_is_preserved(tmp_path):
    excluded = [r"\text{Yes}", "Yes.", r"\text{No}", r"\boxed{No}",
                r"$\boxed{\text{No}}$", "TRUE!", r"\mbox{False}",
                "converges", "Diverges.", r"\boxed{Convergent}", "divergent", None, ""]
    kept = ["0", "1", r"\boxed{\frac{1}{2}}", "No solution", "converges for x > 1"]
    frame = pd.DataFrame([{"question": f"Q{i}", "final_answer": a, "difficulty": 8.5}
                          for i, a in enumerate(excluded + kept)])
    path = tmp_path / "answers.parquet"
    frame.to_parquet(path)
    pool = load_band(path, 8.5)
    assert pool.answer_str.tolist() == kept
    assert pool.final_answer.tolist() == kept
    profile = answer_profile(path, 8.5)
    assert profile["total"] == len(frame)
    assert profile["excluded"] == len(excluded)
    assert profile["kept"] == len(kept)
    assert profile["top_answers_before"]["no"] == 3
    assert profile["top_answers_before"]["yes"] == 2
    assert "no" not in profile["top_answers_after"]
    assert profile["top_answers_after"]["0"] == 1


def test_p1_thinking_pools_keep_the_expected_columns(tmp_path, monkeypatch):
    import os

    # The legacy P1 module removes proxy variables on import; isolate that side effect.
    monkeypatch.setattr(os, "environ", os.environ.copy())
    from scripts import p1_thinking_axis as p1

    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "data/raw/deepmath.parquet"
    raw.parent.mkdir(parents=True)
    frame = pd.DataFrame([
        {"question": f"Q{band}-{i}", "final_answer": ans, "difficulty": band}
        for band in (7.0, 8.5) for i, ans in enumerate([r"\boxed{No}", "2", "3"])])
    frame.to_parquet(raw)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"splits": {"train": []}}))
    monkeypatch.setattr(p1, "SPLITS", split)
    monkeypatch.setattr(p1, "load_pool", lambda *a: pd.DataFrame([
        {"task_id": "math1", "problem": "1+1", "answer_str": "2"}]))
    pools = p1.pools(2)
    for label in ("DeepMath 7.0", "DeepMath 8.5"):
        assert list(pools[label].columns) == ["task_id", "problem", "answer"]
        assert set(pools[label].answer) == {"2", "3"}


def test_wide_band_probe_and_split_loader_use_the_same_pool(fake_parquet, tmp_path):
    from scripts.p2_spread import probe_tasks

    config = {"task": {"deepmath_raw": "x"}}
    small = probe_tasks(config, "deepmath", 5, 19, 8.0, width=1.0)
    probe = probe_tasks(config, "deepmath", 20, 19, 8.0, width=1.0)
    assert [t.task_id for t in small] == [t.task_id for t in probe[:5]]
    excluded = frozenset(t.task_id for t in probe)
    split = build_deepmath(3, 30, 30, 30, 30, 8.0, raw="x", width=1.0, exclude=excluded)
    assert split["band"] == [8.0, 9.0]
    assert split["pool_sizes"]["band_pool"] == 200
    path = tmp_path / "split.json"
    path.write_text(json.dumps(split))
    loaded = load_split_tasks({"task": {"raw": "x", "splits": str(path)}},
                              ["train", "audit", "val", "test"])
    tasks = [t for role in loaded.values() for t in role]
    assert len(tasks) == 120
    assert {t.meta["difficulty"] for t in tasks} == {8.0, 8.5}
    assert not {t.task_id for t in tasks} & excluded
    narrow = build_deepmath(3, 10, 10, 10, 10, 8.0, raw="x")
    assert narrow["band"] == [8.0, 8.5]
    assert narrow["fingerprint"] != split["fingerprint"]


@pytest.mark.parametrize("width", [0, -1, float("nan"), float("inf")])
def test_invalid_band_width_is_rejected(width):
    with pytest.raises(ValueError, match="width"):
        in_band(pd.Series([8.0, 8.5, 9.0]), 8.0, width)
