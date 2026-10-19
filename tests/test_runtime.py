"""Runtime acceptance and failure accounting without any network requests.

Transport tests inject HTTP-shaped replies into the real client; integration tests
use the deterministic endpoint over the same retry/semaphore code. No test needs
gateway credentials, data downloads, or an external service.
"""

from __future__ import annotations

import io
import json
import multiprocessing
import os
import shutil
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from data.loaders.math import Task, load_pool, tasks_from_frame
from runtime.cache import RunCache, key_for
from runtime.endpoints import (
    EndpointError,
    EndpointHTTPError,
    MockEndpoint as TransportMock,
    OpenAICompatEndpoint,
    current_call_context,
)
from runtime.executor import EvalResult, ExecutionError, Executor
from space.s0 import S0_ROLES, declared_nodes, enumerate_space
from space.topology import Topology
from tests.mock_endpoint import MockEndpoint


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("runtime tests must not open network connections")

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", forbidden)


@pytest.fixture
def tasks():
    return [Task(f"task-{i}", "math", "train", f"Compute {i} + {i}.", str(2 * i),
                 {"level": 4}) for i in range(1, 6)]


def topology(edges=(), model="mock"):
    return Topology.canonical(declared_nodes(S0_ROLES, model), edges, space_id="s4")


def executor_at(root, endpoint, *, exp="runtime-test", **kwargs):
    settings = {"max_tokens_per_node": 128, "temperature": 0.7, "train_repeats": 8}
    settings.update(kwargs)
    cache = RunCache(Path(root) / "cache.sqlite", exp=exp)
    return Executor(endpoint, cache, runs_dir=Path(root) / "runs", **settings)


def artifact(record):
    return json.loads(Path(record.node_outputs_path).read_text())


def reply(text="FINAL: 2", *, finish_reason="stop", usage=None):
    return {"choices": [{"message": {"content": text}, "finish_reason": finish_reason}],
            "usage": usage if usage is not None else {
                "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


def transport(responder, *, max_attempts=3, max_concurrency=4, **kwargs):
    return TransportMock("mock", responder=responder, max_concurrency=max_concurrency,
                         max_attempts=max_attempts, backoff_s=(1, 4, 16),
                         sleep=kwargs.pop("sleep", lambda delay: None), **kwargs)


def test_all_25_computations_times_five_tasks_reconcile_and_replay(tmp_path, tasks):
    graphs = list({graph.exec_hash: graph for graph in enumerate_space(model="mock")}.values())
    assert len(graphs) == 25
    endpoint = MockEndpoint(tasks, transient_error_rate=0.25)
    executor = executor_at(tmp_path, endpoint)
    records = [executor.run(graph, task, seed=9, repeat=0) for graph in graphs for task in tasks]
    assert len(records) == 125
    assert any(record.attempts > len(graph.nodes) for graph in graphs
               for record in records if record.exec_hash == graph.exec_hash)
    for record in records:
        nodes = artifact(record)["nodes"]
        assert record.prompt_tokens == sum(node["prompt_tokens"] for node in nodes.values())
        assert record.gen_tokens == sum(node["gen_tokens"] for node in nodes.values())
        assert record.attempts == sum(len(node["attempts"]) for node in nodes.values())
        assert record.endpoint_usage == {
            nid: [attempt["raw_usage"] for attempt in node["attempts"]]
            for nid, node in nodes.items()}
    for field, record_field in (("prompt_tokens", "prompt_tokens"),
                                ("completion_tokens", "gen_tokens")):
        assert sum(getattr(record, record_field) for record in records) == sum(
            call["usage"][field] for call in endpoint.calls)
    assert sum(record.prompt_tokens + record.gen_tokens for record in records) == sum(
        call["usage"]["total_tokens"] for call in endpoint.calls)
    assert sum(record.reasoning_tokens for record in records) == sum(
        call["usage"]["completion_tokens_details"]["reasoning_tokens"] for call in endpoint.calls)
    assert len(list(executor.cache.export(executor.exp))) == 125
    assert list(executor.cache.export_attempts(executor.exp)) == []
    calls_before = endpoint.call_count
    replay = [executor.run(graph, task, seed=9, repeat=0) for graph in graphs for task in tasks]
    assert replay == records
    assert endpoint.call_count == calls_before
    assert executor.cache.stats() == {"hits": 125, "misses": 125, "writes": 125}
    fresh_endpoint = MockEndpoint(tasks)
    fresh_executor = executor_at(tmp_path, fresh_endpoint)
    assert [fresh_executor.run(graph, task, 9, 0) for graph in graphs for task in tasks] == records
    assert fresh_endpoint.call_count == 0


def test_equivalent_labelled_graphs_share_one_sample_and_keep_its_provenance(tmp_path, tasks):
    endpoint = MockEndpoint(tasks)
    executor = executor_at(tmp_path, endpoint)
    solver, critic = topology((("n1", "n4"),)), topology((("n2", "n4"),))
    assert solver.graph_hash != critic.graph_hash and solver.exec_hash == critic.exec_hash
    original = executor.run(solver, tasks[0], 0, 0)
    assert executor.run(critic, tasks[0], 0, 0) == original
    assert original.graph_hash == solver.graph_hash
    assert endpoint.call_count == 2
    # Independent execution of the alias has identical prompts and request seeds.
    other_endpoint = MockEndpoint(tasks)
    executor_at(tmp_path / "other", other_endpoint).run(critic, tasks[0], 0, 0)
    assert [call["payload"] for call in endpoint.calls] == [
        call["payload"] for call in other_endpoint.calls]


def test_parent_outputs_follow_execution_order_and_effective_roles(tmp_path, tasks):
    graph = topology((("n3", "n4"), ("n1", "n4"), ("n2", "n4"), ("n1", "n2")))
    endpoint = transport(lambda payload: reply(f"FINAL: {current_call_context().node_index}"))
    record = executor_at(tmp_path, endpoint).run(graph, tasks[0], 0, 0)
    nodes = artifact(record)["nodes"]
    assert list(nodes) == list(graph.node_ids)
    assert record.effective_roles == {"n1": "solver", "n2": "critic", "n3": "solver",
                                      "n4": "finalizer"}
    assert nodes["n3"]["prompt_id"] == "solver.v1"
    assert nodes["n3"]["prompt"] == nodes["n1"]["prompt"]
    final_prompt = nodes["n4"]["prompt"]
    assert final_prompt.index("[solver]\nFINAL: 0") < final_prompt.index("[critic]\nFINAL: 1")
    assert final_prompt.index("[critic]\nFINAL: 1") < final_prompt.index("[solver]\nFINAL: 2")
    assert "n1" not in final_prompt and "n3" not in final_prompt
    assert record.prediction == "3" and record.score == 0.0 and not record.parse_fail


@pytest.mark.parametrize("role", ("solver", "critic", "reviser", "finalizer"))
def test_frozen_templates_have_both_inputs_and_required_answer_format(role):
    source = Path(f"prompts/{role}.v1.txt").read_text()
    rendered = source.format(task="TASK {inputs}", inputs="PARENT")
    assert "TASK {inputs}" in rendered and "PARENT" in rendered
    assert r"\boxed{}" in rendered and "FINAL: <answer>" in rendered


@pytest.mark.parametrize("seed", range(6))
def test_equivalent_mask_hits_draw_fresh_repeats_and_replay_deterministically(tmp_path, tasks, seed):
    endpoint = MockEndpoint(tasks)
    executor = executor_at(tmp_path, endpoint)
    solver, critic = topology((("n1", "n4"),)), topology((("n2", "n4"),))
    results = executor.evaluate_batch([solver, critic], tasks[:1], seed, repeat_hint=0)
    results += executor.evaluate_batch([critic, solver] * 3, tasks[:1], seed, repeat_hint=0)
    records = list(executor.cache.export(executor.exp))
    assert {record.repeat for record in records} == set(range(8))
    assert len(records) == 8 and endpoint.call_count == 16
    assert all(isinstance(result, EvalResult) and result.exec_hash == solver.exec_hash
               for result in results)
    assert len({call["payload"]["seed"] for call in endpoint.calls}) == 16
    calls_before = endpoint.call_count
    executor.evaluate_batch([solver, critic], tasks[:1], seed, repeat_hint=0)
    assert endpoint.call_count == calls_before  # R exhausted: documented tensor reuse.
    fresh_endpoint = MockEndpoint(tasks)
    fresh_executor = executor_at(tmp_path, fresh_endpoint)
    replay = fresh_executor.evaluate_batch([solver, critic], tasks[:1], seed, repeat_hint=0)
    replay += fresh_executor.evaluate_batch([critic, solver] * 3, tasks[:1], seed, repeat_hint=0)
    assert replay == results and fresh_endpoint.call_count == 0


def test_repeat_hint_and_counts_are_scoped_by_seed_computation_and_task(tmp_path, tasks):
    endpoint = MockEndpoint(tasks)
    executor = executor_at(tmp_path, endpoint)
    graph = topology()
    evaluation, = executor.evaluate_batch([graph], tasks[:2], seed=5, repeat_hint=3)
    records = list(executor.cache.export(executor.exp))
    other_endpoint = MockEndpoint(tasks)
    other = executor_at(tmp_path / "sequential", other_endpoint)
    other.evaluate_batch([graph] * 4, tasks[:1], seed=5, repeat_hint=0)
    assert endpoint.calls[0]["payload"] == other_endpoint.calls[-1]["payload"]
    assert evaluation.task_ids == tuple(task.task_id for task in tasks[:2])
    assert evaluation.prompt_tokens == sum(record.prompt_tokens for record in records)
    assert evaluation.gen_tokens == sum(record.gen_tokens for record in records)
    executor.evaluate_batch([graph], tasks[:1], seed=6, repeat_hint=0)
    other.evaluate_batch([graph], tasks[:1], seed=6, repeat_hint=0)
    # A new search seed starts its own counter at hit 0 in both executors, whatever
    # seed 5 consumed, so both request the same tensor cell. (Comparing "the last
    # endpoint call" instead depended on whether that cell happened to be cached
    # already, i.e. on the exec_hash.)
    h, t0 = graph.exec_hash, tasks[0].task_id
    assert executor._hits[(5, h, t0)] == 4 and other._hits[(5, h, t0)] == 4
    assert executor._hits[(6, h, t0)] == 1 and other._hits[(6, h, t0)] == 1
    cell = executor._repeat_index(6, h, t0, 0)
    assert cell == other._repeat_index(6, h, t0, 0)
    for ex in (executor, other):
        assert (h, t0, cell) in {(r.exec_hash, r.task_id, r.repeat)
                                 for r in ex.cache.export(ex.exp)}


@pytest.mark.parametrize("node_index", (0, 1))
def test_any_truncated_node_is_an_explicit_parse_failure(tmp_path, tasks, node_index):
    graph = topology((("n1", "n4"),))
    endpoint = MockEndpoint(tasks, truncated_nodes=(node_index,), success_rate=1.0)
    record = executor_at(tmp_path, endpoint).run(graph, tasks[0], 0, 0)
    assert record.parse_fail and record.score == 0.0 and record.prediction is None
    assert record.truncated_nodes == (graph.nodes[node_index].id,)
    assert artifact(record)["nodes"][graph.nodes[node_index].id]["text"] == ""
    assert record.gen_tokens >= 128


def test_length_with_parseable_content_still_counts_as_truncation(tmp_path, tasks):
    endpoint = transport(lambda payload: reply("FINAL: 2", finish_reason="length"))
    record = executor_at(tmp_path, endpoint).run(topology(), tasks[0], 0, 0)
    assert record.parse_fail and record.score == 0.0 and record.truncated_nodes == ("n4",)


def test_missing_reasoning_itemization_is_none_even_when_truncated(tmp_path, tasks):
    endpoint = MockEndpoint(tasks, truncation_rate=1.0, itemize_reasoning=False)
    record = executor_at(tmp_path, endpoint).run(topology(), tasks[0], 0, 0)
    assert record.reasoning_tokens is None and record.gen_tokens == 128
    assert record.endpoint_usage["n4"][0] == endpoint.calls[0]["usage"]


@pytest.mark.parametrize("text", (None, "", "Still reasoning, no answer yet."))
def test_empty_or_unparseable_untruncated_content_is_distinguished(tmp_path, tasks, text):
    endpoint = transport(lambda payload: reply(text))
    record = executor_at(tmp_path, endpoint).run(topology(), tasks[0], 0, 0)
    assert record.parse_fail and record.score == 0.0 and record.truncated_nodes == ()


def test_retry_accounts_for_every_billable_attempt_and_logs_it(tmp_path, tasks, caplog):
    endpoint = MockEndpoint(tasks, transient_error_rate=1.0, success_rate=1.0)
    with caplog.at_level("INFO", logger="runtime.endpoints"):
        record = executor_at(tmp_path, endpoint).run(topology(), tasks[0], 0, 0)
    assert record.attempts == 2 and record.score == 1.0
    assert record.prompt_tokens + record.gen_tokens == sum(c["usage"]["total_tokens"]
                                                             for c in endpoint.calls)
    attempts = artifact(record)["nodes"]["n4"]["attempts"]
    assert [attempt["status_code"] for attempt in attempts] == [502, 200]
    assert len(caplog.records) == 2 and "status=502" in caplog.records[0].message


def test_retry_exhaustion_is_audited_raised_and_never_cached_as_a_score(tmp_path, tasks):
    endpoint = MockEndpoint(tasks, transient_error_rate=1.0, fail_first_attempts=3)
    executor = executor_at(tmp_path, endpoint)
    with pytest.raises(ExecutionError) as caught:
        executor.evaluate_batch([topology()], tasks[:1], 0, 0)
    record = caught.value.record
    assert isinstance(caught.value.__cause__, EndpointError)
    assert record.attempts == endpoint.call_count == 3
    assert record.prompt_tokens + record.gen_tokens == sum(c["usage"]["total_tokens"]
                                                             for c in endpoint.calls)
    assert artifact(record)["status"] == "failed"
    assert executor.cache.get(key_for(record)) is None
    assert list(executor.cache.export(executor.exp)) == []
    assert list(executor.cache.export_attempts(executor.exp)) == [record]
    assert executor._hits == {}
    # A retry of the experiment can complete this key; failure history is retained.
    successful, = executor.evaluate_batch([topology()], tasks[:1], 0, 0)
    assert successful.task_ids == (tasks[0].task_id,)
    assert len(list(executor.cache.export(executor.exp))) == 1
    assert list(executor.cache.export_attempts(executor.exp)) == [record]


def test_nontransient_error_is_not_retried(tmp_path, tasks):
    def bad_request(payload):
        raise EndpointHTTPError(400)

    with pytest.raises(ExecutionError) as caught:
        executor_at(tmp_path, transport(bad_request)).run(topology(), tasks[0], 0, 0)
    assert caught.value.record.attempts == 1
    assert caught.value.record.prompt_tokens == caught.value.record.gen_tokens == 0


@pytest.mark.parametrize("role", ("train", "audit", "val", "test"))
def test_load_and_execute_every_split_role_including_test(tmp_path, role):
    raw = tmp_path / "math.parquet"
    import pandas as pd

    pd.DataFrame([{"problem": "Compute 1 + 1.", "solution": r"\boxed{2}",
                   "level": "Level 4", "source_split": "test"}]).to_parquet(raw)
    task, = tasks_from_frame(load_pool(raw, "test"), role)
    endpoint = MockEndpoint([task], success_rate=1.0)
    record = executor_at(tmp_path, endpoint).run(topology(), task, 0, 0)
    assert record.split_role == role and record.score == 1.0 and endpoint.call_count == 1


@pytest.mark.parametrize("role", ("audit", "val", "test"))
def test_optimizer_boundary_rejects_nontraining_tasks_before_any_calls(tmp_path, tasks, role):
    task = replace(tasks[0], split_role=role)
    endpoint = MockEndpoint([task])
    executor = executor_at(tmp_path, endpoint)
    with pytest.raises(ValueError, match="training"):
        executor.evaluate_batch([topology()], [tasks[1], task], 0, 0)
    assert endpoint.call_count == 0


def test_concurrent_runs_share_one_cache_entry(tmp_path, tasks):
    endpoint = MockEndpoint(tasks)
    executors = [executor_at(tmp_path, endpoint) for _ in range(4)]
    graphs = [topology(((nid, "n4"),)) for nid in ("n1", "n2", "n3")]
    with ThreadPoolExecutor(max_workers=12) as workers:
        records = list(workers.map(
            lambda i: executors[i % 4].run(graphs[i % 3], tasks[0], 0, 0), range(12)))
    assert all(record == records[0] for record in records)
    assert endpoint.call_count == 2
    assert sum(executor.cache.stats()["writes"] for executor in executors) == 1


def _process_run(root: str, task: Task):
    endpoint = MockEndpoint([task])
    record = executor_at(root, endpoint).run(topology((("n1", "n4"),)), task, 0, 0)
    return record, endpoint.call_count


def test_cache_single_sample_survives_concurrent_processes(tmp_path, tasks):
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as workers:
        futures = [workers.submit(_process_run, str(tmp_path), tasks[0]) for _ in range(8)]
        results = [future.result() for future in futures]
    assert all(record == results[0][0] for record, _ in results)
    assert sum(calls for _, calls in results) == 2


def test_semaphore_bounds_requests_while_distinct_keys_write_concurrently(tmp_path, tasks):
    lock = threading.Lock()
    active = peak = 0

    def respond(payload):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(active, peak)
        time.sleep(0.02)
        with lock:
            active -= 1
        return reply()

    endpoint = transport(respond, max_concurrency=2)
    executor = executor_at(tmp_path, endpoint)
    with ThreadPoolExecutor(max_workers=8) as workers:
        records = list(workers.map(lambda repeat: executor.run(topology(), tasks[0], 0, repeat),
                                   range(16)))
    assert peak == 2 and len(list(executor.cache.export(executor.exp))) == 16
    assert sum(record.attempts for record in records) == 16


@pytest.mark.parametrize("changed", ({"temperature": 0.9}, {"max_tokens_per_node": 256},
                                    {"train_repeats": 4}))
def test_experiment_rejects_changed_settings_and_new_exp_can_run(tmp_path, tasks, changed):
    endpoint = MockEndpoint(tasks)
    original = executor_at(tmp_path, endpoint).run(topology(), tasks[0], 0, 0)
    with pytest.raises(ValueError, match="new exp"):
        executor_at(tmp_path, endpoint, **changed)
    another = executor_at(tmp_path, endpoint, exp="changed", **changed)
    record = another.run(topology(), tasks[0], 0, 0)
    assert record.exp == "changed" and record.run_id != original.run_id
    assert list(another.cache.export("runtime-test")) == [original]
    assert list(another.cache.export("changed")) == [record]


def test_experiment_rejects_changed_templates(tmp_path, tasks):
    endpoint = MockEndpoint(tasks)
    executor_at(tmp_path, endpoint)
    prompt_root = tmp_path / "prompts"
    shutil.copytree("prompts", prompt_root)
    solver = prompt_root / "solver.v1.txt"
    solver.write_text(solver.read_text() + "New instruction.\n")
    with pytest.raises(ValueError, match="new exp"):
        executor_at(tmp_path, endpoint, prompts_dir=prompt_root)


def test_generation_parameters_cannot_change_in_a_live_executor(tmp_path, tasks):
    executor = executor_at(tmp_path, MockEndpoint(tasks))
    for attribute, value in (("temperature", 0.2), ("max_tokens_per_node", 4), ("train_repeats", 2)):
        with pytest.raises(AttributeError):
            setattr(executor, attribute, value)


def test_missing_usage_is_audited_as_an_error_instead_of_free_tokens(tmp_path, tasks):
    endpoint = transport(lambda payload: reply(usage={}))
    executor = executor_at(tmp_path, endpoint)
    with pytest.raises(ExecutionError) as caught:
        executor.run(topology(), tasks[0], 0, 0)
    assert caught.value.record.attempts == 1
    assert artifact(caught.value.record)["nodes"]["n4"]["attempts"][0]["raw_usage"] == {}
    assert list(executor.cache.export(executor.exp)) == []


def test_partial_reasoning_itemization_does_not_claim_a_complete_total(tmp_path, tasks):
    def respond(payload):
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        if current_call_context().node_index == 0:
            usage["completion_tokens_details"] = {"reasoning_tokens": 3}
        return reply(usage=usage)

    record = executor_at(tmp_path, transport(respond)).run(
        topology((("n1", "n4"),)), tasks[0], 0, 0)
    assert record.reasoning_tokens is None and record.gen_tokens == 10
    assert record.endpoint_usage["n1"][0]["completion_tokens_details"]["reasoning_tokens"] == 3


def test_unknown_template_or_mismatched_model_is_rejected_before_calling(tmp_path, tasks):
    endpoint = MockEndpoint(tasks)
    executor = executor_at(tmp_path, endpoint)
    with pytest.raises(ValueError, match="model"):
        executor.run(topology(model="other"), tasks[0], 0, 0)
    # The terminal keeps its declared template only when it has inputs (a lone
    # Finalizer plays Solver), so plant the unknown template on a fed terminal.
    graph = topology((("n1", "n4"),))
    graph = replace(graph, nodes=(graph.nodes[0],
                                  replace(graph.nodes[1], prompt_id="missing.v1")))
    with pytest.raises(ValueError, match="template"):
        executor.run(graph, tasks[0], 0, 0)
    assert endpoint.call_count == 0


def test_cache_refuses_overwriting_a_sample(tmp_path, tasks):
    executor = executor_at(tmp_path, MockEndpoint(tasks))
    record = executor.run(topology(), tasks[0], 0, 0)
    executor.cache.put(record)
    with pytest.raises(ValueError, match="overwrite"):
        executor.cache.put(replace(record, score=1 - record.score))
    assert executor.cache.get(key_for(record)) == record
    assert executor.cache.stats()["writes"] == 1


def http_endpoint(tmp_path, **kwargs):
    settings = {"base_url": "http://127.0.0.1:9876/v1", "max_concurrency": 2,
                "max_attempts": 3, "backoff_s": (1, 4, 16), "timeout_s": 10,
                "env_file": tmp_path / ".env", "sleep": lambda delay: None}
    settings.update(kwargs)
    return OpenAICompatEndpoint("mock", **settings)


def test_http_transport_bypasses_all_proxy_variables_without_changing_environment(
        tmp_path, tasks, monkeypatch):
    proxy_names = ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
    for name in proxy_names:
        monkeypatch.setenv(name, "http://unreachable.invalid:9999")
    monkeypatch.delenv("LLM_GATEWAY_API_KEY", raising=False)
    (tmp_path / ".env").write_text('export LLM_GATEWAY_API_KEY="fixture-secret" # local test\n')
    endpoint = http_endpoint(tmp_path)
    # build_opener omits the empty ProxyHandler entirely, so no proxy_open hook exists.
    assert not any(isinstance(handler, urllib.request.ProxyHandler)
                   for handler in endpoint._opener.handlers)
    requests = []
    raw_usage = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18,
                 "gateway_extension": {"verbatim": True}}

    def open_mock(request, timeout):
        requests.append((request, timeout))
        return io.BytesIO(json.dumps(reply(usage=raw_usage)).encode())

    monkeypatch.setattr(endpoint._opener, "open", open_mock)
    record = executor_at(tmp_path, endpoint).run(topology(), tasks[0], 7, 0)
    request, timeout = requests[0]
    payload = json.loads(request.data)
    assert request.full_url == "http://127.0.0.1:9876/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer fixture-secret" and timeout == 10
    assert payload["n"] == 1 and payload["model"] == "mock"
    assert payload["max_tokens"] == 128 and payload["temperature"] == 0.7
    assert "seed" in payload and "thinking" not in payload and "reasoning_effort" not in payload
    assert record.endpoint_usage == {"n4": [raw_usage]} and record.reasoning_tokens is None
    assert all(os.environ[name] == "http://unreachable.invalid:9999" for name in proxy_names)
    assert "fixture-secret" not in Path(record.node_outputs_path).read_text()


def test_http_502_and_timeout_use_backoff_and_preserve_raw_usage(tmp_path, tasks, monkeypatch):
    sleeps = []
    endpoint = http_endpoint(tmp_path, sleep=sleeps.append)
    calls = 0
    failed_usage = {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}

    def open_mock(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.HTTPError(request.full_url, 502, "Bad Gateway", {},
                                         io.BytesIO(json.dumps({"usage": failed_usage}).encode()))
        if calls == 2:
            raise urllib.error.URLError("timed out")
        return io.BytesIO(json.dumps(reply()).encode())

    monkeypatch.setattr(endpoint._opener, "open", open_mock)
    record = executor_at(tmp_path, endpoint).run(topology(), tasks[0], 0, 0)
    assert record.attempts == calls == 3 and sleeps == [1, 4]
    assert record.prompt_tokens == 14 and record.gen_tokens == 7
    assert record.endpoint_usage["n4"] == [failed_usage, {}, reply()["usage"]]


@pytest.mark.parametrize("url", ("https://api.openai.com/v1", "http://example.com/v1",
                              "http://127.0.0.1.evil.example/v1", "file:///tmp/foo",
                              "http://user:secret@localhost/v1", "http://localhost/v1?api_key=x"))
def test_endpoint_allowlist_rejects_external_or_credential_urls(tmp_path, url):
    with pytest.raises(ValueError, match="loopback"):
        http_endpoint(tmp_path, base_url=url)


@pytest.mark.parametrize("url", ("http://localhost:8000/v1", "http://[::1]:8000/v1",
                              "http://aimaster-llm-gateway-service.aimaster-web:8080/llm/v1"))
def test_allowlisted_client_construction_does_not_make_requests(tmp_path, monkeypatch, url):
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "fixture-only")
    assert http_endpoint(tmp_path, base_url=url).base_url == url


def test_gateway_requires_key_without_making_requests(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_GATEWAY_API_KEY", raising=False)
    with pytest.raises(ValueError, match="LLM_GATEWAY_API_KEY"):
        http_endpoint(tmp_path, base_url="http://aimaster-llm-gateway-service.aimaster-web:8080/llm/v1")


def test_redirects_cannot_escape_the_allowlist(tmp_path):
    endpoint = http_endpoint(tmp_path)
    redirect, = [handler for handler in endpoint._opener.handlers
                 if isinstance(handler, urllib.request.HTTPRedirectHandler)]
    request = urllib.request.Request("http://localhost/v1")
    assert redirect.redirect_request(request, None, 302, "redirect", {},
                                     "https://external.invalid/") is None


def test_mock_accuracy_is_stable_and_increases_with_structure(tasks):
    by_count = {}
    for graph in enumerate_space(model="mock"):
        value = MockEndpoint.latent_accuracy(graph.exec_hash, len(graph.nodes))
        assert value == MockEndpoint.latent_accuracy(graph.exec_hash, len(graph.nodes))
        by_count.setdefault(len(graph.nodes), []).append(value)
    assert all(max(by_count[count]) < min(by_count[count + 1]) for count in (1, 2, 3))


def test_mock_outputs_are_deterministic_but_repeats_use_different_draws(tmp_path, tasks):
    left, right = MockEndpoint(tasks), MockEndpoint(tasks)
    first, second = executor_at(tmp_path / "a", left), executor_at(tmp_path / "b", right)
    for repeat in range(8):
        a = first.run(topology(), tasks[0], 4, repeat)
        b = second.run(topology(), tasks[0], 4, repeat)
        assert (a.prediction, a.score, a.endpoint_usage) == (b.prediction, b.score, b.endpoint_usage)
    assert [call["text"] for call in left.calls] == [call["text"] for call in right.calls]
    assert len({call["payload"]["seed"] for call in left.calls}) == 8
    assert len({call["text"] for call in left.calls}) == 2


# ------------------------------------------------- request options and replay-only

def _capturing_transport(name="mock", **kwargs):
    payloads = []

    def respond(payload):
        payloads.append(payload)
        return reply("\\boxed{2}\nFINAL: 2", usage={"prompt_tokens": 3,
                                                     "completion_tokens": 2})
    return TransportMock(name, responder=respond, max_concurrency=1, max_attempts=1,
                         backoff_s=(), **kwargs), payloads


def test_request_options_reach_the_payload_and_cannot_override_core_fields(tmp_path, tasks):
    endpoint, payloads = _capturing_transport(request_options={"reasoning_effort": "none"})
    executor_at(tmp_path, endpoint).run(topology(), tasks[0], 0, 0)
    assert payloads[0]["reasoning_effort"] == "none" and payloads[0]["n"] == 1
    with pytest.raises(ValueError, match="may not set"):
        _capturing_transport(request_options={"temperature": 0.0})


def test_reasoning_effort_is_part_of_the_experiment_identity(tmp_path, tasks):
    default, _ = _capturing_transport()
    executor_at(tmp_path, default)
    off, _ = _capturing_transport(request_options={"reasoning_effort": "none"})
    with pytest.raises(ValueError, match="new exp"):
        executor_at(tmp_path, off)
    # An empty option set leaves the settings blob exactly as before the field existed.
    import sqlite3
    settings, = sqlite3.connect(tmp_path / "cache.sqlite").execute(
        "SELECT settings FROM experiments WHERE exp = 'runtime-test'").fetchone()
    assert "request_options" not in settings


def test_replay_only_endpoint_opens_the_same_experiment_and_never_generates(tmp_path, tasks):
    from runtime.endpoints import ReplayOnlyEndpoint, endpoint_from_config

    live = MockEndpoint(tasks)
    executor = executor_at(tmp_path, live)
    graph = topology((("n1", "n4"),))
    executor.run(graph, tasks[0], 0, 0)
    replay = executor_at(tmp_path, ReplayOnlyEndpoint("mock"), replay_only=True)
    assert replay.run(graph, tasks[0], 0, 0).task_id == tasks[0].task_id
    with pytest.raises(RuntimeError, match="replay-only"):
        ReplayOnlyEndpoint("mock").complete("p", seed=0, max_tokens=1, temperature=0.0)
    # No API key is needed to build a replay endpoint for the gateway.
    model = {"name": "DeepSeek", "max_concurrency": 1, "retry": {},
             "endpoint": "http://aimaster-llm-gateway-service.aimaster-web:8080/llm/v1"}
    assert isinstance(endpoint_from_config(model, replay_only=True), ReplayOnlyEndpoint)


def test_scoped_bindings_do_not_disturb_the_main_experiment(tmp_path):
    cache = RunCache(tmp_path / "c.sqlite", exp="e")
    cache.bind_experiment({"a": 1})
    cache.bind_experiment({"selfcheck": "v1"}, scope="pi_single")
    cache.bind_experiment({"a": 1})
    with pytest.raises(ValueError, match="pi_single"):
        cache.bind_experiment({"selfcheck": "v2"}, scope="pi_single")
