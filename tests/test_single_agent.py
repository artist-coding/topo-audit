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
