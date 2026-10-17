"""P2 probe: the summary must catch a flat top that the old gate would wave through."""

from __future__ import annotations

import numpy as np
import pytest

from data.loaders.math import Task
from runtime.cache import RunCache
from runtime.executor import Executor, RunRecord
from runtime.single_agent import SingleAgentRunner, StrategyRun
from scripts.p2_spread import probe_graphs, run_probe, summarize_probe
from tests.mock_endpoint import MockEndpoint
from tests.test_single_agent import _templates

ACCEPTANCE = {"max_truncation_rate": 0.15, "max_row_truncation_spread": 0.03,
              "token_reconciliation_error": 0.01}


def _record(exec_hash: str, task_id: str, score: float) -> RunRecord:
    return RunRecord(run_id=f"{exec_hash}{task_id}", exp="p2", model="m",
                     graph_hash=exec_hash, exec_hash=exec_hash, task_id=task_id,
                     split_role="audit", seed=0, repeat=0, score=score, parse_fail=False,
                     prediction="1", prompt_tokens=50, gen_tokens=50, reasoning_tokens=None,
                     truncated_nodes=(), latency_s=0.0, attempts=1, node_outputs_path="",
                     endpoint_usage={}, effective_roles={})


def test_a_flat_top_fails_even_when_best_minus_worst_is_large():
    graphs = probe_graphs("m", "hand6")
    labels = list(graphs)
    rng = np.random.default_rng(0)
    n = 200
    base = (rng.random(n) < 0.7).astype(float)
    records = []
    for i, label in enumerate(labels):
        scores = base.copy()
        if i == len(labels) - 1:
            scores = (rng.random(n) < 0.3).astype(float)     # one terrible graph
        else:
            scores[i] = 1 - scores[i]                         # five near-identical ones:
                                                              # one task apart (0.5 pt)
        records += [_record(graphs[label].exec_hash, f"t{j}", s) for j, s in enumerate(scores)]
    summary = summarize_probe(records, [], graphs, [f"t{j}" for j in range(n)])
    assert summary["gates"]["old_best_minus_worst"]["pass"]
    assert not summary["gates"]["top_end_not_flat"]["pass"]
    assert summary["spread"]["within_2_points_of_best"] == 5


def test_capacity_is_cross_fitted_against_pi_single_at_the_budget():
    graphs = probe_graphs("m", "hand6")
    n = 2000                       # enough tasks that sampling noise sits well below 0.05
    task_ids = [f"t{j}" for j in range(n)]
    rng = np.random.default_rng(1)
    records = [_record(t.exec_hash, tid, float(rng.random() < 0.6))
               for t in graphs.values() for tid in task_ids]
    # pi_single is as good as the graphs: C should be about zero, not the naive max.
    single = [StrategyRun("self_consistency", 3, tid, 0, float(rng.random() < 0.6), False,
                          "1", 60, 40, None, 3) for tid in task_ids]
    summary = summarize_probe(records, single, graphs, task_ids)
    capacity = summary["capacity"]
    assert capacity["cross_fitted"] < capacity["naive"]["diff"]
    assert abs(capacity["cross_fitted"]) < 0.05


def test_run_probe_end_to_end_with_the_mock(tmp_path):
    tasks = [Task(f"p{i}", "math", "audit", f"Compute {i} + {i}.", str(2 * i), {"level": 4})
             for i in range(1, 13)]
    endpoint = MockEndpoint(tasks, max_concurrency=4)
    cache = RunCache(tmp_path / "cache.sqlite", exp="p2_test")
    executor = Executor(endpoint, cache, max_tokens_per_node=128, temperature=0.7,
                        train_repeats=2, runs_dir=tmp_path / "runs")
    runner = SingleAgentRunner(endpoint, cache, _templates(), exp="p2_test",
                               max_tokens=128, temperature=0.7, samples=4,
                               output_dir=tmp_path / "runs")
    config = {"acceptance": ACCEPTANCE, "single_agent": {"budget_sweep": [1, 2, 4]}}
    summary = run_probe(config, executor, runner, tasks, probe_graphs("mock", "hand6"),
                        repeats=1, concurrency=4)
    assert len(summary["ranking"]) == 6 and summary["tasks"] == 12
    assert summary["power"]["variance"] > 0
    assert set(summary["power"]["val_tasks_needed"]) == {"0.02", "0.03", "0.04"}
    assert "cross_fitted" in summary["capacity"]
    assert summary["single_agent_frontier"]
    assert summary["tokens"]["fresh"] > 0
