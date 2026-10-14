"""Search acceptance through local endpoints and the real executor boundary.

The original noisy mock gives its third and fourth computations accuracies less
than 0.001 apart. A finite-budget top-three guarantee, or a universally worse prune
path, is not a valid correctness assertion on that landscape. Convergence tests use
an explicit, separated parallel-aggregation landscape here; integration/replay tests
also use the unchanged noisy mock. Neither oracle is available to an optimizer.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import itertools
import json
import math
import random
import urllib.request
from pathlib import Path
from statistics import fmean, pstdev

import pytest

from data.loaders.math import Task
from optimizers import AgentPrune, Budget, GreedyLocal, RandomSearch, ReinforceEdge
from optimizers.base import EdgeSettings, SearchOptimizer, canonical, distinct_computations
from optimizers.runner import TaskBatches, discriminability, run_optimizer
from runtime.cache import RunCache
from runtime.endpoints import MockEndpoint as TransportMock, current_call_context
from runtime.executor import EvalResult, Executor
from space.s0 import S0_ROLES, declared_nodes, enumerate_space, forward_edges
from space.topology import Topology
from tests.mock_endpoint import MockEndpoint

POLICIES = (RandomSearch, ReinforceEdge, AgentPrune, GreedyLocal)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("optimizer tests must not open network connections")

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", forbidden)


@pytest.fixture(scope="module")
def space():
    return enumerate_space(model="mock")


@pytest.fixture
def training(tmp_path):
    tasks = [Task(f"train-{i}", "math", "train", f"Compute {i} + 1.", str(i + 1),
                  {"index": i}) for i in range(24)]
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"seed": 20261013,
                                     "splits": {"train": [t.task_id for t in tasks]}}))
    return tasks, split_path


class RecordingExecutor(Executor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batches = []

    def evaluate_batch(self, topologies, tasks, seed, repeat_hint):
        topologies, tasks = tuple(topologies), tuple(tasks)
        results = super().evaluate_batch(topologies, tasks, seed, repeat_hint)
        self.batches.append((topologies, tasks, repeat_hint, results))
        return results


def executor_at(root, endpoint, *, exp="optimizers"):
    return RecordingExecutor(endpoint, RunCache(root / "cache.sqlite", exp=exp),
                             max_tokens_per_node=128, temperature=0.7, train_repeats=8,
                             runs_dir=root / "runs")


def results_for(graphs, rewards, n_tasks=2):
    """Unit-level observations; integration tests obtain these from evaluate_batch."""
    ids = tuple(f"task-{i}" for i in range(n_tasks))
    return [EvalResult(graph.exec_hash, ids, (reward,) * n_tasks, 11, 7)
            for graph, reward in zip(graphs, rewards)]


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("values", [(-1, 24), (1, 0), (1, 2, 2), (1, 2, 0, -1),
                                   (1.0, 24), (True, 24)])
def test_budget_rejects_invalid_counts(values):
    with pytest.raises(ValueError):
        Budget(*values)


@pytest.mark.parametrize("policy", POLICIES)
def test_empty_space_and_partial_state_cannot_start_a_search(policy, space):
    with pytest.raises(ValueError, match="empty"):
        policy([], Budget(10, 2), 0)
    with pytest.raises(ValueError, match="fresh"):
        policy(space, Budget(10, 2, spent_evals=1), 0)
    optimizer = policy(space, Budget(0, 2), 0)
    assert optimizer.propose() == []
    assert optimizer.budget.left() == 0
    json.dumps(optimizer.state(), allow_nan=False)


def test_edge_defaults_are_the_committed_experimental_settings():
    assert EdgeSettings.from_config() == EdgeSettings(8, 0.1, 4.0, 0.01)


def test_random_search_deduplicates_computations_and_requires_confirmation(space):
    optimizer = RandomSearch(space, Budget(200, 2), 0)
    assert len(optimizer.space) == 25
    two = RandomSearch([space[0], space[-1]], Budget(3, 2), 0)
    first = two.propose()
    assert first[0].graph_hash == space[-1].graph_hash
    two.observe(results_for(first, [0.2]))
    assert two.incumbent().graph_hash == space[0].graph_hash  # fixed, unconfirmed prior
    second = two.propose()
    assert second[0].graph_hash == first[0].graph_hash
    two.observe(results_for(second, [0.4]))
    assert two.incumbent().graph_hash == first[0].graph_hash
    third = two.propose()
    assert third[0].graph_hash != first[0].graph_hash
    two.observe(results_for(third, [1.0]))  # one lucky draw must not win
    assert two.incumbent().graph_hash == first[0].graph_hash
    assert two.state()["estimates"][first[0].graph_hash] == {"mean": pytest.approx(0.3), "count": 2}


def test_duplicate_masks_and_global_random_state_do_not_bias_uniform_search(space):
    reference = RandomSearch(space, Budget(200, 2), 4)
    aliases = RandomSearch([*space, *space, *([space[0]] * 100)], Budget(200, 2), 4)
    global_state = random.getstate()
    seen = set()
    try:
        for step in range(200):
            a = reference.propose()
            random.seed(step)
            random.random()
            b = aliases.propose()
            assert [g.graph_hash for g in a] == [g.graph_hash for g in b]
            seen.add(a[0].exec_hash)
            reference.observe(results_for(a, [0.5]))
            aliases.observe(results_for(b, [0.5]))
    finally:
        random.setstate(global_state)
    assert len(seen) == 25
    assert sum(v["count"] for v in reference.state()["estimates"].values()) == 200


@pytest.mark.parametrize("defect", ("missing", "wrong_hash", "wrong_tasks", "short_scores",
                                    "duplicate_tasks", "nonfinite", "negative_tokens", "not_result"))
def test_bad_observations_do_not_spend_budget_or_change_estimates(space, defect):
    optimizer = ReinforceEdge(space, Budget(2, 2), 0)
    proposal = optimizer.propose()
    assert optimizer.propose() == proposal
    valid = results_for(proposal, [0.2, 0.8])
    bad = list(valid)
    if defect == "missing":
        bad.pop()
    elif defect == "wrong_hash":
        bad[0] = bad[0]._replace(exec_hash="wrong")
    elif defect == "wrong_tasks":
        bad[1] = bad[1]._replace(task_ids=("different", "tasks"))
    elif defect == "short_scores":
        bad[0] = bad[0]._replace(scores=(0.0,))
    elif defect == "duplicate_tasks":
        bad[0] = bad[0]._replace(task_ids=("same", "same"))
    elif defect == "nonfinite":
        bad[0] = bad[0]._replace(scores=(float("nan"), 0.0))
    elif defect == "negative_tokens":
        bad[0] = bad[0]._replace(prompt_tokens=-1)
    else:
        bad[0] = tuple(bad[0])
    previous = optimizer.state()
    with pytest.raises((ValueError, TypeError)):
        optimizer.observe(bad)
    assert optimizer.state() == previous
    assert (optimizer.budget.spent_evals, optimizer.budget.spent_tokens) == (0, 0)
    optimizer.observe(valid)
    assert (optimizer.budget.spent_evals, optimizer.budget.spent_tokens) == (2, 36)
    with pytest.raises(ValueError, match="pending"):
        optimizer.observe(valid)


def test_reinforce_gradient_uses_raw_masks_and_group_relative_advantages(space):
    optimizer = ReinforceEdge(space, Budget(8, 2), 0)
    proposals = optimizer.propose()
    masks = optimizer.state()["sampled_masks"]
    canonical_masks = [optimizer._edge_space.mask(graph) for graph in proposals]
    assert any(tuple(raw) != pruned for raw, pruned in zip(masks, canonical_masks))
    rewards = [len(graph.edges) / 6 for graph in proposals]
    advantages = [(reward - fmean(rewards)) / (pstdev(rewards) + 1e-6) for reward in rewards]
    expected = [0.1 * fmean(a * (mask[j] - 0.5) for a, mask in zip(advantages, masks))
                for j in range(6)]
    optimizer.observe(results_for(proposals, rewards))
    assert optimizer.logits == pytest.approx(expected)
    assert optimizer.state()["advantages"] == pytest.approx(advantages)
    assert optimizer.state()["group_scores"] == rewards
    rounded = optimizer._edge_space.topology([int(theta >= 0) for theta in expected])
    assert optimizer.incumbent().graph_hash == rounded.graph_hash


def test_entropy_gradient_clipping_and_constant_group_rewards(space):
    optimizer = ReinforceEdge(space, Budget(8, 2), 2)
    optimizer.logits[:] = [3.0, -3.0] * 3
    before = list(optimizer.logits)
    proposals = optimizer.propose()
    optimizer.observe(results_for(proposals, [0.5] * 8))
    assert optimizer.state()["advantages"] == [0.0] * 8
    for old, new in zip(before, optimizer.logits):
        p = 1 / (1 + math.exp(-old))
        assert new == pytest.approx(old - 0.1 * 0.01 * old * p * (1 - p))
        assert abs(new) < abs(old)
    clipped = ReinforceEdge(space, Budget(8, 2), 0,
                            settings=EdgeSettings(8, 100.0, 4.0, 0.01))
    proposals = clipped.propose()
    clipped.observe(results_for(proposals, [len(g.nodes) / 4 for g in proposals]))
    assert max(map(abs, clipped.logits)) == 4.0


def test_pruning_stays_on_one_removal_path_and_uses_the_full_budget(space):
    optimizer = AgentPrune(space, Budget(200, 2), 1)
    visited = []
    while optimizer.budget.left():
        graph, = optimizer.propose()
        assert graph.graph_hash == canonical(graph).graph_hash
        if not visited or graph.graph_hash != visited[-1].graph_hash:
            if visited:
                assert set(graph.edges) < set(visited[-1].edges)
            visited.append(graph)
        optimizer.observe(results_for([graph], [len(graph.nodes) / 4]))
    assert len(visited[0].edges) == 6
    assert len(visited) <= 7
    assert visited[-1].edges == ()
    assert optimizer.budget.spent_evals == 200
    assert len(optimizer.state()["prune_order"]) == len(visited) - 1
    assert optimizer.incumbent().graph_hash in {g.graph_hash for g in visited}


def test_mask_learning_consumes_rewards_and_magnitude_regularisation(space):
    high = AgentPrune(space, Budget(200, 2), 0, magnitude_reg=0.0)
    low = AgentPrune(space, Budget(200, 2), 0, magnitude_reg=0.0)
    regularized = AgentPrune(space, Budget(200, 2), 0, magnitude_reg=0.5)
    for optimizer, reward in ((high, 1.0), (low, 0.0), (regularized, 1.0)):
        optimizer.observe(results_for(optimizer.propose(), [reward]))
    assert all(a > b for a, b in zip(high.state()["mask_values"], low.state()["mask_values"]))
    assert sum(map(abs, regularized.state()["mask_values"])) < sum(map(abs, high.state()["mask_values"]))


def test_greedy_compares_a_paired_reference_then_moves_or_restarts(space):
    optimizer = GreedyLocal(space, Budget(30, 2), 2)
    proposals = optimizer.propose()
    current = proposals[0]
    mask = optimizer._edge_space.mask(current)
    reachable = set()
    for j in range(len(mask)):
        changed = list(mask)
        changed[j] = 1 - changed[j]
        reachable.add(optimizer._edge_space.topology(changed).exec_hash)
    assert all(graph.exec_hash in reachable for graph in proposals[1:])
    assert len({g.exec_hash for g in proposals}) == len(proposals)
    optimizer.observe(results_for(proposals, [0.2, 0.9, *([0.1] * (len(proposals) - 2))]))
    assert optimizer.state()["current"] == proposals[1].graph_hash
    assert optimizer.incumbent().graph_hash == proposals[1].graph_hash
    assert optimizer.state()["restart_count"] == 0
    proposals = optimizer.propose()
    optimizer.observe(results_for(proposals, [0.1] * len(proposals)))
    assert optimizer.state()["restart_count"] == 1
    assert optimizer.state()["current"] != proposals[0].graph_hash
    assert optimizer.incumbent().graph_hash == proposals[0].graph_hash


def test_batches_follow_manifest_order_and_ignore_global_random_state(training):
    tasks, split = training
    sequence = TaskBatches.from_split(tasks, 7, split_path=split)
    shuffled = TaskBatches.from_split(list(reversed(tasks)), 7, split_path=split)
    assert sequence.tasks == tuple(tasks)
    assert [sequence.batch(i) for i in range(8)] == [shuffled.batch(i) for i in range(8)]
    assert sequence.batch(0) != sequence.batch(1)
    state = random.getstate()
    try:
        random.seed(777)
        assert shuffled.batch(5) == sequence.batch(5)
        assert sequence.batch(0) == shuffled.batch(0)  # random access, no consumption state
    finally:
        random.setstate(state)
    with pytest.raises(ValueError, match="exactly"):
        TaskBatches.from_split(tasks[:-1], 7, split_path=split)
    with pytest.raises(ValueError, match="without duplicates"):
        TaskBatches.from_split(tasks, 25, split_path=split)


def test_default_batches_use_the_actual_committed_train_order():
    manifest = json.loads(Path("data/splits/math_v1.json").read_text())
    order = manifest["splits"]["train"]
    tasks = [Task(task_id, "math", "train", "fixture", "1", {}) for task_id in reversed(order)]
    batches = TaskBatches.from_split(tasks, 24)
    assert [task.task_id for task in batches.tasks] == order
    assert batches.seed == manifest["seed"]
    assert len(batches.batch(0)) == 24


@pytest.mark.parametrize("policy", POLICIES)
def test_real_executor_accounting_every_checkpoint_and_byte_identical_replay(
        tmp_path, space, training, policy):
    tasks, split = training
    endpoint = MockEndpoint(tasks)
    executor = executor_at(tmp_path, endpoint)
    optimizer = policy(space, Budget(11, 4), 3)
    path = run_optimizer(optimizer, executor, tasks, runs_dir=tmp_path / "runs", split_path=split)
    original = path.read_bytes()
    trajectory = rows(path)
    assert path == tmp_path / "runs" / "optimizers" / optimizer.name / "seed3" / "trajectory.jsonl"
    assert [row["step"] for row in trajectory] == list(range(len(executor.batches)))
    count, tokens = 0, 0
    for row, (graphs, batch, hint, evaluations) in zip(trajectory, executor.batches):
        count += len(evaluations)
        tokens += sum(r.prompt_tokens + r.gen_tokens for r in evaluations)
        assert row["cum_evals"] == count
        assert row["cum_tokens"] == tokens
        assert row["candidates"] == [g.graph_hash for g in graphs]
        assert row["exec_hashes"] == [g.exec_hash for g in graphs]
        assert row["task_ids"] == [t.task_id for t in batch]
        assert row["scores"] == [list(r.scores) for r in evaluations]
        assert row["discriminability"] == discriminability(evaluations)
        assert all(g.graph_hash == canonical(g).graph_hash for g in graphs)
        assert hint == 0
    assert count == optimizer.budget.spent_evals == 11
    assert tokens == optimizer.budget.spent_tokens
    if policy is RandomSearch or policy is AgentPrune:
        assert len(trajectory) == 11
    if policy is ReinforceEdge:
        assert [len(row["candidates"]) for row in trajectory] == [8, 3]
    fresh_endpoint = MockEndpoint(tasks)
    fresh_executor = executor_at(tmp_path, fresh_endpoint)
    fresh = policy(space, Budget(11, 4), 3)
    replay = run_optimizer(fresh, fresh_executor, tasks, runs_dir=tmp_path / "runs", split_path=split)
    assert replay.read_bytes() == original
    assert fresh_endpoint.call_count == 0
    assert fresh.budget.spent_tokens == tokens  # replay does not turn compute into zero


def test_optimizers_and_seeds_receive_identical_tasks_at_each_step(tmp_path, space, training):
    tasks, split = training
    expected = None
    for seed, policy in itertools.product((0, 4), POLICIES):
        endpoint = MockEndpoint(tasks, success_rate=1.0)
        executor = executor_at(tmp_path, endpoint)
        optimizer = policy(space, Budget(9, 3), seed)
        path = run_optimizer(optimizer, executor, tasks, runs_dir=tmp_path / "runs", split_path=split)
        first_two = [row["task_ids"] for row in rows(path)[:2]]
        assert len(first_two) == 2
        if expected is None:
            expected = first_two
        assert first_two == expected


@pytest.mark.parametrize("role", ("audit", "val", "test"))
@pytest.mark.parametrize("policy", POLICIES)
def test_held_out_tasks_are_rejected_before_any_optimizer_scores(
        tmp_path, space, training, role, policy):
    tasks, split = training
    forbidden = [dataclasses.replace(task, split_role=role) for task in tasks]
    endpoint = MockEndpoint(forbidden)
    executor = executor_at(tmp_path, endpoint)
    optimizer = policy(space, Budget(8, 2), 0)
    with pytest.raises(ValueError, match="training tasks only"):
        run_optimizer(optimizer, executor, forbidden, runs_dir=tmp_path / "runs", split_path=split)
    assert endpoint.call_count == 0
    assert optimizer.budget.spent_evals == optimizer.budget.spent_tokens == 0
    assert optimizer.state()["estimates"] == {}


@pytest.mark.parametrize("role", ("audit", "val", "test"))
def test_executor_still_rejects_held_out_tasks_if_batch_selection_is_bypassed(
        tmp_path, space, training, monkeypatch, role):
    tasks, split = training
    forbidden = tuple(dataclasses.replace(task, split_role=role) for task in tasks[:2])
    monkeypatch.setattr(TaskBatches, "batch", lambda self, step: forbidden)
    endpoint = MockEndpoint(tasks)
    executor = executor_at(tmp_path, endpoint)
    optimizer = ReinforceEdge(space, Budget(8, 2), 0)
    with pytest.raises(ValueError, match="optimizer evaluate_batch accepts training tasks only"):
        run_optimizer(optimizer, executor, tasks, runs_dir=tmp_path / "runs", split_path=split)
    assert endpoint.call_count == 0
    assert optimizer.budget.spent_evals == optimizer.budget.spent_tokens == 0


@pytest.mark.parametrize("policy", POLICIES)
def test_optimizers_have_no_runtime_capabilities_or_task_references(space, policy):
    optimizer = policy(space, Budget(8, 2), 0)
    optimizer.observe(results_for(optimizer.propose(), [0.5] * 8))
    seen = set()

    def walk(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        assert not isinstance(value, (Executor, RunCache, TransportMock, Task))
        assert not (callable(value) and isinstance(getattr(value, "__self__", None), Executor))
        if isinstance(value, dict):
            for key, child in value.items():
                walk(key)
                walk(child)
        elif isinstance(value, (tuple, list, set, frozenset)):
            for child in value:
                walk(child)
        elif dataclasses.is_dataclass(value):
            for field in dataclasses.fields(value):
                walk(getattr(value, field.name))
        elif hasattr(value, "__dict__"):
            walk(vars(value))

    walk(optimizer)
    for module in (inspect.getmodule(policy), inspect.getmodule(SearchOptimizer)):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("runtime"):
                assert node.module == "runtime.executor"
                assert {alias.name for alias in node.names} == {"EvalResult"}
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {"evaluate_batch", "run", "complete", "export"}
    json.dumps(optimizer.state(), allow_nan=False)


def test_repeat_hints_preserve_fresh_samples_for_canonical_aliases(tmp_path, training):
    tasks, split = training
    nodes = declared_nodes(S0_ROLES, "mock")
    aliases = [Topology.canonical(nodes, [(source, "n4")], "s4") for source in ("n1", "n2")]
    assert aliases[0].exec_hash == aliases[1].exec_hash

    class AliasSearch(SearchOptimizer):
        name = "aliases"

        def _propose(self):
            return list(self.space[:min(2, self.budget.left())])

    endpoint = MockEndpoint(tasks, success_rate=1.0)
    executor = executor_at(tmp_path, endpoint)
    optimizer = AliasSearch(aliases, Budget(10, 24), 0)
    path = run_optimizer(optimizer, executor, tasks, runs_dir=tmp_path / "runs", split_path=split)
    records = list(executor.cache.export(executor.exp))
    for task in tasks:
        repeats = [r.repeat for r in records if r.task_id == task.task_id]
        assert sorted(repeats) == list(range(8))
    assert endpoint.call_count == 2 * 24 * 8
    assert optimizer.budget.spent_evals == 10
    assert optimizer.budget.spent_tokens == sum(
        r.prompt_tokens + r.gen_tokens for *_, results in executor.batches for r in results)
    assert all(row["discriminability"]["n_tasks"] == 0 for row in rows(path))


def test_nonzero_repeat_hint_is_forwarded_unchanged(tmp_path, space, training):
    tasks, split = training
    executor = executor_at(tmp_path, MockEndpoint(tasks, success_rate=1.0))
    optimizer = RandomSearch([space[0]], Budget(3, 2), 0)
    run_optimizer(optimizer, executor, tasks, runs_dir=tmp_path / "runs",
                  split_path=split, repeat_hint=3)
    assert [hint for _, _, hint, _ in executor.batches] == [3, 3, 3]


def test_interruption_restarts_from_cache_without_custom_checkpoint_state(
        tmp_path, space, training, monkeypatch):
    tasks, split = training
    endpoint = MockEndpoint(tasks, success_rate=1.0)
    executor = executor_at(tmp_path, endpoint)
    real_run = executor.run
    calls = 0

    def interrupt(topology, task, seed, repeat):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("simulated interruption inside a candidate")
        return real_run(topology, task, seed, repeat)

    monkeypatch.setattr(executor, "run", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_optimizer(RandomSearch(space, Budget(4, 2), 3), executor, tasks,
                      runs_dir=tmp_path / "runs", split_path=split)
    path = tmp_path / "runs" / "optimizers" / "random_search" / "seed3" / "trajectory.jsonl"
    prefix = path.read_bytes()
    assert len(rows(path)) == 1
    fresh = executor_at(tmp_path, MockEndpoint(tasks, success_rate=1.0))
    run_optimizer(RandomSearch(space, Budget(4, 2), 3), fresh, tasks,
                  runs_dir=tmp_path / "runs", split_path=split)
    assert path.read_bytes().startswith(prefix)
    assert fresh.cache.stats()["hits"] >= 3
    completed = path.read_bytes()
    replay_endpoint = MockEndpoint(tasks, success_rate=1.0)
    run_optimizer(RandomSearch(space, Budget(4, 2), 3), executor_at(tmp_path, replay_endpoint), tasks,
                  runs_dir=tmp_path / "runs", split_path=split)
    assert path.read_bytes() == completed
    assert replay_endpoint.call_count == 0


def test_raw_masks_are_canonical_before_they_enter_the_trajectory(tmp_path, training):
    tasks, split = training
    nodes = declared_nodes(S0_ROLES, "mock")
    edges = forward_edges([node.id for node in nodes])
    raw = [Topology(nodes, tuple(edge for edge, bit in zip(edges, mask) if bit), "raw")
           for mask in itertools.product((0, 1), repeat=6)]
    optimizer = RandomSearch(raw, Budget(40, 2), 2)
    executor = executor_at(tmp_path, MockEndpoint(tasks, success_rate=1.0))
    path = run_optimizer(optimizer, executor, tasks, runs_dir=tmp_path / "runs", split_path=split)
    legal = {canonical(graph).graph_hash for graph in raw}
    assert len(optimizer.space) == 25
    for row in rows(path):
        assert set(row["candidates"]) <= legal
        assert row["incumbent"] in legal


def test_discriminability_uses_paired_variance_and_averages_alias_repeats():
    ids = ("a", "b", "c", "d")
    a = EvalResult("first", ids, (1.0, 1.0, 1.0, 0.0), 1, 1)
    b = EvalResult("second", ids, (1.0, 0.0, 0.0, 0.0), 1, 1)
    c = EvalResult("third", ids, (0.0, 0.0, 0.0, 0.0), 1, 1)
    diagnostic = discriminability([a, b, c])
    assert diagnostic == {"top2_paired_diff": 0.5, "se": pytest.approx(1 / math.sqrt(12)),
                          "n_tasks": 4}
    duplicate = a._replace(scores=(0.0, 0.0, 0.0, 0.0))
    assert discriminability([a, duplicate, b])["top2_paired_diff"] == 0.125
    assert discriminability([a, duplicate]) == {"top2_paired_diff": 0.0, "se": 0.0, "n_tasks": 0}
    with pytest.raises(ValueError, match="ordered task"):
        discriminability([a, b._replace(task_ids=tuple(reversed(ids)))])


def test_discriminability_is_never_a_stopping_rule(tmp_path, space, training, monkeypatch):
    tasks, split = training
    monkeypatch.setattr("optimizers.runner.discriminability",
                        lambda results: {"top2_paired_diff": 0.0, "se": 1.0, "n_tasks": 2})
    optimizer = RandomSearch([space[0]], Budget(200, 2), 0)
    executor = executor_at(tmp_path, MockEndpoint(tasks, success_rate=1.0))
    path = run_optimizer(optimizer, executor, tasks, runs_dir=tmp_path / "runs", split_path=split)
    assert len(rows(path)) == optimizer.budget.spent_evals == 200


class ParallelLandscape(TransportMock):
    """Exact 24-task accuracies reward parallel inputs and penalize internal edges.

    The strongest computations are separated by several tasks, unlike the original
    mock's tiny jitter. All rewards are still parsed from endpoint text by Executor;
    the oracle below is used only by the test after search has completed.
    """

    def __init__(self, tasks, space):
        self.tasks = {task.task_id: task for task in tasks}
        self.correct_tasks = {graph.exec_hash: self.quality(graph) for graph in space}
        self.call_count = 0
        super().__init__("mock", responder=self._respond, max_attempts=1,
                         max_concurrency=1, backoff_s=())

    @staticmethod
    def quality(graph):
        internal = sum(destination != graph.terminal for _, destination in graph.edges)
        return max(0, 8 * len(graph.parents(graph.terminal)) - 12 * internal)

    def _respond(self, payload):
        context = current_call_context()
        task = self.tasks[context.task_id]
        correct = task.meta["index"] < self.correct_tasks[context.exec_hash]
        self.call_count += 1
        return {"choices": [{"message": {"content": f"FINAL: {task.answer}" if correct else "No result."},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}


@pytest.mark.parametrize("seed", range(5))
def test_three_searches_reach_true_top_three_and_pruning_has_a_larger_gap(
        tmp_path, space, training, seed):
    tasks, split = training
    computations = distinct_computations(space)
    ranked = sorted(computations, key=lambda g: (-ParallelLandscape.quality(g), g.exec_hash))
    top_three = {graph.exec_hash for graph in ranked[:3]}
    optimum = ParallelLandscape.quality(ranked[0]) / 24
    gaps = {}
    for policy in POLICIES:
        endpoint = ParallelLandscape(tasks, space)
        executor = executor_at(tmp_path, endpoint)
        optimizer = policy(space, Budget(200, 24), seed)
        path = run_optimizer(optimizer, executor, tasks, runs_dir=tmp_path / "runs", split_path=split)
        learned = optimizer.incumbent()
        gaps[policy] = optimum - ParallelLandscape.quality(learned) / 24
        if policy is not AgentPrune:
            assert learned.exec_hash in top_three, (policy.name, seed, str(learned))
        else:
            assert len({g for row in rows(path) for g in row["candidates"]}) <= 7
        assert optimizer.budget.spent_evals == 200
        assert optimizer.budget.spent_tokens == sum(
            result.prompt_tokens + result.gen_tokens
            for _, _, _, results in executor.batches for result in results)
    assert gaps[AgentPrune] > gaps[RandomSearch]
