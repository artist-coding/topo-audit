"""Resolve a committed split file into Task objects, for every driver.

Shared by audit/run_audit.py, runtime/single_agent.py and optimizers/sweep.py so they
cannot disagree about which problems a role contains or what their gold answers are.
Answers and levels are checked against the split file, so a re-downloaded raw pool
that drifted fails loudly instead of silently changing the benchmark.
"""

from __future__ import annotations

import json
import pathlib
from typing import Sequence

import pandas as pd

from data.loaders.math import Task, load_pool, tasks_from_frame

DEFAULT_RAW = "data/raw/math_hendrycks.parquet"


def read_split(config: dict) -> dict:
    return json.loads(pathlib.Path(config["task"]["splits"]).read_text())


def load_split_tasks(config: dict, roles: Sequence[str]) -> dict[str, list[Task]]:
    """{role: [Task]} for the requested roles of the configured split file."""
    split = read_split(config)
    family = split.get("family", "math")
    if family != "math":
        raise NotImplementedError(f"no loader for task family {family!r} yet")
    levels = tuple(split.get("levels", (4, 5)))
    raw = config["task"].get("raw", DEFAULT_RAW)
    answers, levels_by_task = split["answers"], split["levels_by_task"]
    wanted = {tid for role in roles for tid in split["splits"][role]}

    frames = []
    for source in ("train", "test"):
        frame = load_pool(raw, source, levels)
        frames.append(frame[frame["task_id"].isin(wanted)])
    pool = pd.concat(frames, ignore_index=True).drop_duplicates(subset="task_id")

    out: dict[str, list[Task]] = {}
    for role in roles:
        ids = set(split["splits"][role])
        frame = pool[pool["task_id"].isin(ids)]
        if len(frame) != len(ids):
            raise SystemExit(f"{role}: resolved {len(frame)} of {len(ids)} task ids; "
                             "re-run data/fetch_math.py")
        tasks = list(tasks_from_frame(frame, role))
        for task in tasks:
            if task.answer != str(answers[task.task_id]):
                raise SystemExit(f"answer drift for {task.task_id}")
            if task.meta["level"] != levels_by_task[task.task_id]:
                raise SystemExit(f"level drift for {task.task_id}")
        out[role] = tasks
    return out
