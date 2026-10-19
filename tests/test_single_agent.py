"""Acceptance tests for build step 4 (pi_single).

The invariant that matters most is the last one: pi_single's CoT call and an
input-less solver node inside a topology must issue byte-identical prompts. C is the
difference between those two worlds, so any prompt asymmetry would be absorbed into
the paper's headline quantity.
"""

from __future__ import annotations

import pathlib

import pytest

from data.loaders.math import Task
from runtime.cache import RunCache
from runtime.executor import Executor, render_inputs
from runtime.single_agent import (
    BEST_OF_N_SELFCHECK,
    COT,
    SELF_CONSISTENCY,
    SELF_REFINE,
    FrontierPoint,
    SingleAgentRunner,
    majority_vote,
    pick_headline,
    strategy_hash,
)
from space.s0 import S0_ROLES, declared_nodes
from space.topology import Node, Topology
from tests.mock_endpoint import MockEndpoint

PROMPTS = pathlib.Path("prompts")


def _tasks(n: int = 3) -> list[Task]:
    return [
        Task(task_id=f"t{i}", family="math", split_role="val",
             prompt=f"Problem {i}: compute something.", answer=str(i + 1), meta={})
        for i in range(n)
    ]


def _templates() -> dict[str, str]:
    return {
        f"{name}.v1": (PROMPTS / f"{name}.v1.txt").read_text()
        for name in ("solver", "critic", "reviser", "finalizer", "selfcheck")
    }


@pytest.fixture
def runner(tmp_path):
    tasks = _tasks()
    endpoint = MockEndpoint(tasks, success_rate=1.0, max_concurrency=2)
    cache = RunCache(tmp_path / "cache.sqlite", exp="t_single")
    return SingleAgentRunner(
        endpoint, cache, _templates(), exp="t_single", max_tokens=256,
        temperature=0.7, samples=4, output_dir=tmp_path / "runs"), endpoint, tasks


# ------------------------------------------------------------------ aggregation

def test_majority_vote_counts_normalised_answers_together():
    assert majority_vote([r"\frac{1}{2}", "0.5", "7"]) in (r"\frac{1}{2}", "0.5")
    assert majority_vote(["7", "7", r"\frac{1}{2}"]) == "7"


def test_majority_vote_ignores_unparsed_samples():
    assert majority_vote([None, "4", None]) == "4"
    assert majority_vote([None, None]) is None


def test_majority_vote_breaks_ties_toward_the_earliest_sample():
    assert majority_vote(["3", "4"]) == "3"


# -------------------------------------------------------------------- frontier

def _point(strategy: str, param: int, acc: float, tokens: float) -> FrontierPoint:
    return FrontierPoint(strategy=strategy, param=param, n_tasks=100, accuracy=acc,
                         accuracy_se=0.01, mean_tokens=tokens, parse_fail_rate=0.0)


def test_pick_headline_takes_the_strongest_affordable_strategy():
    frontier = [_point(COT, 1, 0.50, 1_000),
                _point(SELF_CONSISTENCY, 4, 0.62, 4_000),
                _point(SELF_CONSISTENCY, 8, 0.66, 8_000)]
    assert pick_headline(frontier, 4_500).param == 4
    assert pick_headline(frontier, 100_000).param == 8
    assert pick_headline(frontier, 500) is None


def test_frontier_point_ci_is_clipped_to_the_unit_interval():
    assert _point(COT, 1, 0.99, 1).ci95[1] <= 1.0
    assert _point(COT, 1, 0.01, 1).ci95[0] >= 0.0


def test_strategy_hash_separates_strategies_params_and_prompts():
    base = strategy_hash(COT, 1, ["solver.v1"])
    assert base != strategy_hash(SELF_CONSISTENCY, 1, ["solver.v1"])
    assert base != strategy_hash(COT, 2, ["solver.v1"])
    assert base != strategy_hash(COT, 1, ["critic.v1"])


# --------------------------------------------------------------------- running

def test_one_sample_draw_serves_the_whole_k_sweep(runner):
    runner_, endpoint, tasks = runner
    runs = runner_.run_task(tasks[0], seed=0)
    solver_calls = [c for c in endpoint.calls
                    if "Solve the" in c["payload"]["messages"][0]["content"]]
    # 4 samples cover self-consistency@1..4 and best-of-N@1..4; no per-k redraw.
    assert len(solver_calls) == 4
    ks = sorted(r.param for r in runs if r.strategy == SELF_CONSISTENCY)
    assert ks == [1, 2, 3, 4]


def test_every_declared_strategy_appears_on_the_frontier(runner):
    runner_, _, tasks = runner
    frontier, _ = runner_.frontier(tasks, seed=0)
    assert {p.strategy for p in frontier} == {
        COT, SELF_CONSISTENCY, BEST_OF_N_SELFCHECK, SELF_REFINE}


def test_cost_increases_with_budget_parameter(runner):
    runner_, _, tasks = runner
    frontier, _ = runner_.frontier(tasks, seed=0)
    sc = sorted((p for p in frontier if p.strategy == SELF_CONSISTENCY),
                key=lambda p: p.param)
    assert [p.param for p in sc] == [1, 2, 3, 4]
    assert all(a.mean_tokens < b.mean_tokens for a, b in zip(sc, sc[1:]))


def test_self_refine_costs_one_plus_two_r_calls(runner):
    runner_, _, tasks = runner
    runs = runner_.run_task(tasks[0], seed=0)
    by_r = {r.param: r.calls for r in runs if r.strategy == SELF_REFINE}
    assert by_r == {1: 3, 2: 5, 3: 7}


def test_replay_issues_zero_endpoint_calls(runner):
    runner_, endpoint, tasks = runner
    runner_.run_task(tasks[0], seed=0)
    before = len(endpoint.calls)
    runner_.run_task(tasks[0], seed=0)
    assert len(endpoint.calls) == before


def test_token_accounting_matches_the_endpoint(runner):
    runner_, endpoint, tasks = runner
    runs = runner_.run_task(tasks[0], seed=0)
    cot = next(r for r in runs if r.strategy == COT)
    first = endpoint.calls[0]["usage"]
    assert cot.prompt_tokens == first["prompt_tokens"]
    assert cot.gen_tokens == first["completion_tokens"]


# -------------------------------------------------- the prompt-identity invariant

def test_pi_single_cot_prompt_matches_an_input_less_solver_node(tmp_path):
    """C must not absorb a prompt difference between the two worlds it compares."""
    tasks = _tasks(1)
    task = tasks[0]

    endpoint = MockEndpoint(tasks, success_rate=1.0, max_concurrency=1)
    cache = RunCache(tmp_path / "c.sqlite", exp="t_invariant")
    executor = Executor(endpoint, cache, max_tokens_per_node=256, temperature=0.7,
                        train_repeats=1, runs_dir=tmp_path / "runs")
    # A one-node Solver topology: the in-space analogue of a bare CoT call.
    solo = Topology.canonical([Node(id="n1", role="solver", prompt_id="solver.v1",
                                    model=endpoint.name)], [], space_id="probe")
    executor.run(solo, task, seed=0, repeat=0)
    node_prompt = endpoint.calls[-1]["payload"]["messages"][0]["content"]

    templates = _templates()
    cot_prompt = templates["solver.v1"].format(task=task.prompt, inputs=render_inputs([]))
    assert node_prompt == cot_prompt


def test_input_less_node_prompt_has_no_dangling_received_work_header():
    prompt = _templates()["solver.v1"].format(task="T", inputs=render_inputs([]))
    assert "Received work" not in prompt


def test_node_with_parents_gets_the_received_work_section():
    nodes = declared_nodes(S0_ROLES, "m")
    chain = Topology.canonical(nodes, [("n1", "n4")], space_id="s4")
    assert chain.parents("n4") == ("n1",)
    rendered = render_inputs([("solver", "partial answer")])
    assert "Received work" in rendered and "[solver]" in rendered


def test_zero_edge_graph_issues_the_cot_prompt_byte_for_byte(tmp_path):
    """G_empty is the in-space single agent: its lone Finalizer must play Solver.

    Before, it received the Finalizer template ("reconcile the received work") with
    nothing received, so P(G_empty) and P(CoT) compared two different prompts.
    """
    tasks = _tasks(1)
    endpoint = MockEndpoint(tasks, success_rate=1.0, max_concurrency=1)
    executor = Executor(endpoint, RunCache(tmp_path / "c.sqlite", exp="t_empty"),
                        max_tokens_per_node=256, temperature=0.7, train_repeats=1,
                        runs_dir=tmp_path / "runs")
    g_empty = Topology.canonical(declared_nodes(S0_ROLES, endpoint.name), [],
                                 space_id="s4")
    assert g_empty.node_ids == ("n4",)
    record = executor.run(g_empty, tasks[0], seed=0, repeat=0)
    assert record.effective_roles == {"n4": "solver"}
    node_prompt = endpoint.calls[-1]["payload"]["messages"][0]["content"]
    cot_prompt = _templates()["solver.v1"].format(task=tasks[0].prompt,
                                                  inputs=render_inputs([]))
    assert node_prompt == cot_prompt


# ---------------------------------------------- scoring rule, lean evaluation, selection

def _truncating_runner(tmp_path, truncate_first_sample: bool):
    """Every reply contains a correct boxed answer; the first solver sample is cut off."""
    from runtime.endpoints import MockEndpoint as TransportMock, current_call_context

    task = Task(task_id="t0", family="math", split_role="val", prompt="Problem: 1+1.",
                answer="2", meta={})

    def respond(payload):
        context = current_call_context()
        is_first_sample = (truncate_first_sample and context.repeat == 0
                           and "Solve the" in payload["messages"][0]["content"])
        return {"choices": [{"message": {"content": "work... \\boxed{2}\nFINAL: 2"},
                             "finish_reason": "length" if is_first_sample else "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3}}

    endpoint = TransportMock("mock", responder=respond, max_concurrency=1,
                             max_attempts=1, backoff_s=())
    runner = SingleAgentRunner(endpoint, RunCache(tmp_path / "c.sqlite", exp="t_trunc"),
                               _templates(), exp="t_trunc", max_tokens=64,
                               temperature=0.7, samples=3, output_dir=tmp_path / "runs")
    return runner, task


def test_a_truncated_call_zeroes_the_strategy_run_like_a_truncated_node(tmp_path):
    runner, task = _truncating_runner(tmp_path, truncate_first_sample=True)
    runs = {(r.strategy, r.param): r for r in runner.run_task(task, seed=0)}
    cot = runs[(COT, 1)]
    assert cot.truncated and cot.score == 0.0 and cot.parse_fail and cot.prediction is None
    assert runs[(SELF_CONSISTENCY, 3)].score == 0.0      # it used the cut-off sample
    assert runs[(SELF_REFINE, 1)].score == 0.0           # it started from it
    clean, task = _truncating_runner(tmp_path / "clean", truncate_first_sample=False)
    assert all(r.score == 1.0 and not r.truncated for r in clean.run_task(task, seed=0))


@pytest.mark.parametrize("strategy,param,expected_calls", [
    (COT, 1, 1), (SELF_CONSISTENCY, 3, 3), (BEST_OF_N_SELFCHECK, 2, 4), (SELF_REFINE, 2, 5)])
def test_run_strategy_makes_only_the_calls_it_needs_and_matches_the_frontier(
        tmp_path, strategy, param, expected_calls):
    tasks = _tasks(1)
    lean_endpoint = MockEndpoint(tasks, max_concurrency=1)
    lean = SingleAgentRunner(lean_endpoint, RunCache(tmp_path / "a.sqlite", exp="lean"),
                             _templates(), exp="lean", max_tokens=256, temperature=0.7,
                             samples=4, output_dir=tmp_path / "runs")
    run = lean.run_strategy(tasks[0], 0, strategy, param)
    assert lean_endpoint.call_count == expected_calls and run.calls == expected_calls
    full = SingleAgentRunner(MockEndpoint(tasks, max_concurrency=1),
                             RunCache(tmp_path / "b.sqlite", exp="full"), _templates(),
                             exp="full", max_tokens=256, temperature=0.7, samples=4,
                             output_dir=tmp_path / "runs")
    point = next(r for r in full.run_task(tasks[0], 0)
                 if (r.strategy, r.param) == (strategy, param))
    assert (run.score, run.prediction, run.prompt_tokens, run.gen_tokens) == (
        point.score, point.prediction, point.prompt_tokens, point.gen_tokens)


def test_pi_single_is_selected_on_val_and_only_the_choice_is_scored_on_test(tmp_path):
    from runtime.single_agent import select_on_val_report_on_test

    val = _tasks(4)
    test = [Task(task_id=f"x{i}", family="math", split_role="test",
                 prompt=f"Test problem {i}.", answer=str(i + 1), meta={}) for i in range(5)]
    endpoint = MockEndpoint(val + test, max_concurrency=2)
    runner = SingleAgentRunner(endpoint, RunCache(tmp_path / "c.sqlite", exp="sel"),
                               _templates(), exp="sel", max_tokens=256, temperature=0.7,
                               samples=4, output_dir=tmp_path / "runs")
    result = select_on_val_report_on_test(runner, val, test, budget_tokens=10_000,
                                          concurrency=2)
    headline = result["headline"]
    val_points = [p for p in result["val_frontier"] if p["mean_tokens"] <= 10_000]
    assert headline["val_accuracy"] == max(p["accuracy"] for p in val_points)
    assert headline["selected_on"] == "val" and headline["reported_on"] == "test"
    assert set(headline["task_scores"]) == {t.task_id for t in test}
    assert headline["accuracy"] == pytest.approx(
        sum(headline["task_scores"].values()) / len(test))
    test_contexts = [c["context"] for c in endpoint.calls if c["context"].task_id.startswith("x")]
    per_task = len(test_contexts) / len(test)
    expected = {COT: 1, SELF_CONSISTENCY: headline["param"],
                BEST_OF_N_SELFCHECK: 2 * headline["param"],
                SELF_REFINE: 1 + 2 * headline["param"]}[headline["strategy"]]
    assert per_task == expected        # nothing else was run on test


def test_headline_budget_is_the_median_over_distinct_computations_on_audit():
    from runtime.executor import RunRecord
    from runtime.single_agent import s0_token_median

    def record(exec_hash, tokens, split="audit"):
        return RunRecord(run_id=f"{exec_hash}{tokens}{split}", exp="e", model="m",
                         graph_hash=exec_hash, exec_hash=exec_hash, task_id="t",
                         split_role=split, seed=0, repeat=0, score=1.0, parse_fail=False,
                         prediction="1", prompt_tokens=tokens, gen_tokens=0,
                         reasoning_tokens=None, truncated_nodes=(), latency_s=0.0,
                         attempts=1, node_outputs_path="", endpoint_usage={},
                         effective_roles={})

    records = [record("a", 100), record("a", 300), record("b", 400), record("c", 1000),
               record("c", 9999, split="test")]
    assert s0_token_median(records) == pytest.approx(400.0)   # means 200, 400, 1000
    assert s0_token_median([record("a", 1, split="val")]) is None
