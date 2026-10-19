"""Driver tests: the tensor build accounting and the replay-only optimizer sweep.

Both drivers used to be untested glue. The properties that matter:

- the sweep reads a finished tensor with zero endpoint calls and fails loudly, naming
  the cell, when the tensor has a hole;
- the tensor manifest separates tokens spent now from cells cached by earlier runs,
  reconciles them against the raw endpoint usage, and gates the per-row truncation
  spread, which is what leaks into comparisons between graphs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from audit.run_audit import build_tensor, distinct_rows
from data.loaders.math import Task
from optimizers.base import EdgeSettings
from optimizers.sweep import run_sweep
from runtime.cache import RunCache
from runtime.endpoints import ReplayOnlyEndpoint
from runtime.executor import CacheMiss, Executor
from tests.mock_endpoint import MockEndpoint

MODEL = "mock"
SETTINGS = EdgeSettings(group_size=4, lr=0.1, theta_clip=4.0, entropy_reg=0.01)


def _config() -> dict:
    return {
        "model": {"name": MODEL, "max_tokens_per_node": 128, "temperature": 0.7},
        "tensor": {"repeats": {"train": 2}},
        "budget": {"max_evals": 12, "tasks_per_eval": 3},
        "acceptance": {"max_truncation_rate": 0.15, "max_row_truncation_spread": 0.03,
                       "token_reconciliation_error": 0.01},
    }


def _tasks(n: int = 6) -> list[Task]:
    return [Task(f"tr{i}", "math", "train", f"Compute {i} + {i}.", str(2 * i), {"level": 4})
            for i in range(1, n + 1)]


def _executor(root: Path, endpoint, **kwargs) -> Executor:
    return Executor(endpoint, RunCache(root / "cache.sqlite", exp="drv"),
                    max_tokens_per_node=128, temperature=0.7, train_repeats=2,
                    runs_dir=root / "runs", **kwargs)


def _cells(tasks, repeats: int = 2):
    return [(topology, task, 0, repeat) for topology in distinct_rows(MODEL)
            for task in tasks for repeat in range(repeats)]


def _split_file(root: Path, tasks) -> Path:
    path = root / "split.json"
    path.write_text(json.dumps({"seed": 7, "splits": {"train": [t.task_id for t in tasks]}}))
    return path


# ------------------------------------------------------------------ the sweep

def test_sweep_replays_a_complete_tensor_without_any_endpoint(tmp_path):
    tasks = _tasks()
    build_tensor(_config(), _executor(tmp_path, MockEndpoint(tasks)), _cells(tasks),
                 concurrency=4, roles=["train"], n_rows=25)
    manifest = run_sweep(_config(), ReplayOnlyEndpoint(MODEL),
                         RunCache(tmp_path / "cache.sqlite", exp="drv"), tasks,
                         optimizers=["random_search", "reinforce_edge", "greedy_local",
                                     "prune_from_dense"],
                         seeds=[0, 1], settings=SETTINGS, runs_dir=tmp_path / "runs",
                         split_path=_split_file(tmp_path, tasks), replay_only=True)
    assert manifest["replay_only"]
    for name, by_seed in manifest["runs"].items():
        assert sorted(by_seed) == ["0", "1"], name
        assert all(run["cum_evals"] == 12 for run in by_seed.values()), name
        assert all(Path(run["trajectory"]).exists() for run in by_seed.values())


def test_sweep_names_the_missing_cell_instead_of_generating_it(tmp_path):
    tasks = _tasks()
    build_tensor(_config(), _executor(tmp_path, MockEndpoint(tasks)), _cells(tasks[:-1]),
                 concurrency=4, roles=["train"], n_rows=25)
    with pytest.raises(CacheMiss, match="replay-only"):
        run_sweep(_config(), ReplayOnlyEndpoint(MODEL),
                  RunCache(tmp_path / "cache.sqlite", exp="drv"), tasks,
                  optimizers=["random_search"], seeds=[0], settings=SETTINGS,
                  runs_dir=tmp_path / "runs", split_path=_split_file(tmp_path, tasks),
                  replay_only=True)


# ------------------------------------------------------------ tensor manifest

def test_manifest_separates_fresh_from_cached_tokens_and_reconciles(tmp_path):
    tasks = _tasks(2)
    endpoint = MockEndpoint(tasks)
    first = build_tensor(_config(), _executor(tmp_path, endpoint), _cells(tasks),
                         concurrency=4, roles=["train"], n_rows=25)
    assert first["tokens"]["fresh"] == first["tokens"]["logical"] > 0
    assert first["acceptance"]["token_reconciliation_ok"]
    assert first["acceptance"]["token_reconciliation_error"] == 0.0
    calls = endpoint.call_count
    again = build_tensor(_config(), _executor(tmp_path, endpoint), _cells(tasks),
                         concurrency=4, roles=["train"], n_rows=25)
    assert endpoint.call_count == calls
    assert again["tokens"]["fresh"] == 0
    assert again["tokens"]["logical"] == first["tokens"]["logical"]


def test_per_row_truncation_spread_is_gated(tmp_path):
    tasks = _tasks(2)
    # Node index 1 is truncated everywhere: single-node rows never lose a run, every
    # deeper row loses all of them -- the depth-dependent penalty in its extreme form.
    endpoint = MockEndpoint(tasks, truncated_nodes=(1,))
    manifest = build_tensor(_config(), _executor(tmp_path, endpoint), _cells(tasks),
                            concurrency=4, roles=["train"], n_rows=25)
    block = manifest["truncation_by_row"]["train"]
    assert block["min_row"] == 0.0 and block["max_row"] == 1.0
    assert not manifest["acceptance"]["row_truncation_spread_ok"]
