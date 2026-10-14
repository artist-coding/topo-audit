"""Tests for the cache-to-matrix bridge.

The load-bearing property: a cell the tensor never reached stays NaN. Turning it into
0.0 would assert "this graph answered wrong", which is a different claim from "we
never ran it", and the difference flows straight into C, E and O.
"""

from __future__ import annotations

import numpy as np
import pytest

from analysis.matrices import build_matrix, coverage, repeat_cube
from runtime.executor import RunRecord


def _record(exec_hash: str, task_id: str, score: float, *, repeat: int = 0,
            split_role: str = "test") -> RunRecord:
    return RunRecord(
        run_id=f"{exec_hash}-{task_id}-{repeat}", exp="t", model="m",
        graph_hash=exec_hash + "-labelled", exec_hash=exec_hash, task_id=task_id,
        split_role=split_role, seed=0, repeat=repeat, score=score, parse_fail=False,
        prediction="x", prompt_tokens=1, gen_tokens=1, reasoning_tokens=None,
        truncated_nodes=(), latency_s=0.1, attempts=1, node_outputs_path="/dev/null",
        endpoint_usage={}, effective_roles={})


def test_repeats_average_into_one_cell():
    records = [_record("g1", "t1", 1.0, repeat=0), _record("g1", "t1", 0.0, repeat=1)]
    matrix = build_matrix(records, "test")
    assert matrix.row("g1")[0] == pytest.approx(0.5)


def test_unrun_cells_stay_nan_not_zero():
    """A zero would be read as a wrong answer; NaN is read as absent."""
    records = [_record("g1", "t1", 1.0)]
    matrix = build_matrix(records, "test", exec_hashes=("g1", "g2"),
                          task_ids=("t1", "t2"))
    assert matrix.values[0, 0] == 1.0
    assert np.isnan(matrix.values[0, 1])
    assert np.all(np.isnan(matrix.values[1]))


def test_other_splits_are_excluded():
    records = [_record("g1", "t1", 1.0, split_role="test"),
               _record("g1", "t2", 0.0, split_role="train")]
    matrix = build_matrix(records, "test")
    assert matrix.task_ids == ("t1",)


def test_requested_row_order_is_preserved():
    records = [_record("gb", "t1", 1.0), _record("ga", "t1", 0.0)]
    matrix = build_matrix(records, "test", exec_hashes=("gb", "ga"))
    assert matrix.exec_hashes == ("gb", "ga")
    assert matrix.row("gb")[0] == 1.0


def test_means_ignore_missing_tasks():
    records = [_record("g1", "t1", 1.0)]
    matrix = build_matrix(records, "test", exec_hashes=("g1",), task_ids=("t1", "t2"))
    assert matrix.means()[0] == pytest.approx(1.0)


def test_coverage_reports_holes_and_empty_rows():
    records = [_record("g1", "t1", 1.0)]
    matrix = build_matrix(records, "test", exec_hashes=("g1", "g2"),
                          task_ids=("t1", "t2"))
    report = coverage(matrix)
    assert report == {"rows": 2, "tasks": 2, "filled": pytest.approx(0.25),
                      "empty_rows": ["g2"], "min_row_coverage": pytest.approx(0.0)}


def test_repeat_cube_keeps_draws_separate_for_variance():
    records = [_record("g1", "t1", 1.0, repeat=0, split_role="train"),
               _record("g1", "t1", 0.0, repeat=1, split_role="train")]
    rows, cols, cube = repeat_cube(records, "train", repeats=4)
    assert rows == ("g1",) and cols == ("t1",)
    assert cube.shape == (1, 1, 4)
    assert list(cube[0, 0, :2]) == [1.0, 0.0]
    assert np.all(np.isnan(cube[0, 0, 2:]))


def test_repeat_cube_drops_repeats_beyond_the_declared_count():
    records = [_record("g1", "t1", 1.0, repeat=9, split_role="train")]
    _, _, cube = repeat_cube(records, "train", repeats=2)
    assert np.all(np.isnan(cube))


# ---------------------------------------------- one tensor must serve every seed

def test_search_seeds_share_the_tensor(tmp_path):
    """The premise the whole budget rests on.

    A tensor cell is one LLM draw for (exec_hash, task, repeat). With the search seed
    in the cache key, 49 of 50 seeds miss every cell and the optimizer sweep costs
    8.7x the tensor; with it out, the sweep is free.
    """
    from data.loaders.math import Task
    from runtime.cache import RunCache
    from runtime.executor import Executor
    from space.s0 import S0_ROLES, declared_nodes
    from space.topology import Topology
    from tests.mock_endpoint import MockEndpoint

    tasks = [Task(task_id="t0", family="math", split_role="train",
                  prompt="P", answer="1", meta={})]
    endpoint = MockEndpoint(tasks, success_rate=1.0, max_concurrency=1)
    cache = RunCache(tmp_path / "c.sqlite", exp="shared")
    topology = Topology.canonical(declared_nodes(S0_ROLES, endpoint.name),
                                  [("n1", "n4")], space_id="s4")

    builder = Executor(endpoint, cache, max_tokens_per_node=128, temperature=0.7,
                       train_repeats=8, runs_dir=tmp_path / "runs")
    for repeat in range(8):
        builder.run(topology, tasks[0], builder.sample_seed, repeat)
    built = len(endpoint.calls)
    assert built > 0

    for seed in (0, 7, 23, 49):
        searcher = Executor(endpoint, cache, max_tokens_per_node=128, temperature=0.7,
                            train_repeats=8, runs_dir=tmp_path / "runs")
        searcher.evaluate_batch([topology], tasks, seed=seed, repeat_hint=0)
        assert len(endpoint.calls) == built, f"seed {seed} missed the tensor"


def test_search_seeds_draw_different_samples_from_the_tensor(tmp_path):
    """Sharing cells must not make two seeds see identical reward sequences."""
    from data.loaders.math import Task
    from runtime.cache import RunCache
    from runtime.executor import Executor
    from space.s0 import S0_ROLES, declared_nodes
    from space.topology import Topology
    from tests.mock_endpoint import MockEndpoint

    tasks = [Task(task_id=f"t{i}", family="math", split_role="train",
                  prompt=f"P{i}", answer="1", meta={}) for i in range(6)]
    endpoint = MockEndpoint(tasks, success_rate=0.5, max_concurrency=1)
    cache = RunCache(tmp_path / "c.sqlite", exp="draws")
    topology = Topology.canonical(declared_nodes(S0_ROLES, endpoint.name),
                                  [("n1", "n4")], space_id="s4")

    def repeats_drawn(seed: int) -> list[int]:
        executor = Executor(endpoint, cache, max_tokens_per_node=128, temperature=0.7,
                            train_repeats=8, runs_dir=tmp_path / "runs")
        return [executor._repeat_index(seed, topology.exec_hash, task.task_id, 0)
                for task in tasks]

    assert repeats_drawn(0) != repeats_drawn(7)
