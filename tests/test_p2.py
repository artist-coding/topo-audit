"""P2 probe: the summary must catch a flat top that the old gate would wave through."""

from __future__ import annotations

from dataclasses import replace
import json
import numpy as np
import pytest

from data.loaders.math import Task
from runtime.cache import RunCache
from runtime.executor import Executor, RunRecord
from runtime.single_agent import SingleAgentRunner, StrategyRun
from scripts.p2_spread import gate_label, probe_graphs, probe_tasks, run_probe, summarize_probe
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
    endpoint = MockEndpoint(tasks, max_concurrency=4, transient_error_rate=0.25)
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
    paid = sum(c["usage"]["prompt_tokens"] + c["usage"]["completion_tokens"]
               for c in endpoint.calls)
    assert summary["tokens"]["logical"] == summary["tokens"]["fresh"] == paid
    assert summary["tokens"]["pi_single"]["calls"] == len(tasks) * (4 + 4 + 6)
    assert summary["tokens"]["pi_single"]["fresh"] > 0
    count = endpoint.call_count
    again = run_probe(config, executor, runner, tasks, probe_graphs("mock", "hand6"),
                      repeats=1, concurrency=4)
    assert endpoint.call_count == count
    assert again["tokens"]["fresh"] == 0
    assert again["tokens"]["logical"] == paid


@pytest.mark.parametrize("damage", ["missing_task", "missing_repeat", "duplicate", "nan"])
def test_incomplete_tensor_cannot_produce_a_ranking(damage):
    graphs = probe_graphs("m", "hand6")
    ids = [f"t{i}" for i in range(10)]
    records = [replace(_record(g.exec_hash, tid, float(i < 5)), repeat=r)
               for g in graphs.values() for i, tid in enumerate(ids) for r in range(2)]
    if damage == "missing_task":
        first = next(iter(graphs.values())).exec_hash
        records = [r for r in records if r.exec_hash != first or r.score == 1]
    elif damage == "missing_repeat":
        records.pop()
    elif damage == "duplicate":
        records.append(records[0])
    else:
        records[0] = replace(records[0], score=float("nan"))
    with pytest.raises(ValueError, match="Incomplete P2 tensor"):
        summarize_probe(records, [], graphs, ids, repeats=2)


def test_failed_probe_resumes_only_missing_cells_and_reports_no_partial_gates(tmp_path):
    from runtime.endpoints import EndpointHTTPError, current_call_context

    tasks = [Task(f"p{i}", "math", "audit", f"Compute {i} + {i}.", str(2*i), {})
             for i in range(6)]
    graphs = probe_graphs("mock", "hand6")
    first_hash = next(iter(graphs.values())).exec_hash
    endpoint = MockEndpoint(tasks, max_attempts=1)
    responder = endpoint._responder

    def fail_one(payload):
        ctx = current_call_context()
        if ctx.exec_hash == first_hash and ctx.task_id == "p5" and ctx.repeat == 1:
            raise EndpointHTTPError(502)
        return responder(payload)

    endpoint._responder = fail_one
    cache = RunCache(tmp_path / "cache.sqlite", exp="p2_resume")
    executor = Executor(endpoint, cache, max_tokens_per_node=128, temperature=0.7,
                        train_repeats=2, runs_dir=tmp_path / "runs")
    runner = SingleAgentRunner(endpoint, cache, _templates(), exp="p2_resume",
                               max_tokens=128, temperature=0.7, samples=2,
                               output_dir=tmp_path / "runs")
    config = {"acceptance": ACCEPTANCE, "single_agent": {"budget_sweep": [1, 2]}}
    with pytest.raises(ValueError, match="1 missing"):
        run_probe(config, executor, runner, tasks, graphs, repeats=2, concurrency=2)
    assert not any(r.split_role == "pi_single" for r in cache.export(executor.exp))
    endpoint._responder = responder
    before = endpoint.call_count
    report = run_probe(config, executor, None, tasks, graphs, repeats=2, concurrency=2)
    assert endpoint.call_count == before + 1
    assert report["tokens"]["pi_single"]["fresh"] == 0
    assert report["tokens"]["fresh"] < report["tokens"]["logical"]
    assert len(report["ranking"]) == 6
    assert report["gates"]["capacity_positive"]["pass"] is None


def test_unevaluated_gates_are_null_and_display_na():
    graphs = probe_graphs("m", "hand6")
    ids = [f"t{i}" for i in range(10)]
    records = [_record(g.exec_hash, t, float(i % 2))
               for g in graphs.values() for i, t in enumerate(ids)]
    report = summarize_probe(records, [], graphs, ids, max_row_truncation_spread=None)
    for name in ("capacity_positive", "row_truncation_spread"):
        passed = report["gates"][name]["pass"]
        assert passed is None
        assert gate_label(passed) == "n/a"
    assert gate_label(True) == "PASS" and gate_label(False) == "FAIL"
    single = [StrategyRun("cot", 1, tid, 0, 1., False, "1", 60, 40, None, 1)
              for tid in ids]
    report = summarize_probe(records, single, graphs, ids, budget_tokens=1)
    assert report["gates"]["capacity_positive"]["pass"] is None
    report = summarize_probe(records, single, graphs, ids, budget_tokens=100)
    assert report["gates"]["capacity_positive"]["pass"] is False
    assert report["gates"]["row_truncation_spread"]["pass"] is True


def test_math_probe_checks_size_after_excluding_assigned_tasks(tmp_path, monkeypatch):
    import pandas as pd
    import scripts.p2_spread as p2

    split = tmp_path / "split.json"
    split.write_text(json.dumps({"splits": {"train": ["taken"]}}))
    frame = pd.DataFrame([
        {"task_id": tid, "problem": "1+1", "prompt": "1+1", "answer_str": "2",
         "level_int": 4, "type": "algebra"} for tid in ("taken", "free1", "free2")])
    monkeypatch.setattr(p2, "load_pool", lambda *a: frame)
    config = {"task": {"splits": str(split)}}
    with pytest.raises(SystemExit, match="2 held-out problems < requested 3"):
        probe_tasks(config, "math", 3, 0)
    assert {t.task_id for t in probe_tasks(config, "math", 2, 0)} == {"free1", "free2"}
