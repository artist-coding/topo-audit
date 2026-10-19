"""End-to-end tests for analysis/report.py on a synthetic tensor.

The regression these guard against: optimizers spend evaluations in rounds of
different sizes, and the report used to look incumbents up at exact budgets. That
silently removed reinforce_edge (rounds of 8) from the plateau test and computed
greedy_local's early intervals on whichever seeds happened to land on the budget.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from analysis.plateau import scan_plateau
from analysis.report import build_report, load_pi_single
from analysis.trajectory import align, on_budget_grid
from optimizers import GreedyLocal, RandomSearch, ReinforceEdge
from optimizers.base import Budget, EdgeSettings
from runtime.executor import EvalResult, RunRecord
from space.s0 import S0_ROLES, enumerate_space

MODEL = "mock"
SPLIT_SIZES = {"train": (12, 2), "audit": (10, 2), "val": (10, 2), "test": (20, 1)}


def _config(**overrides) -> dict:
    config = {
        "exp": "t_report",
        "model": {"name": MODEL},
        "plateau": {"epsilon": 2.0, "delta_t": 10},
        "budget": {"max_evals": 60, "tasks_per_eval": 4, "report_at": [5, 60]},
        "seeds": [0, 1, 2],
        "optimizers": ["random_search", "reinforce_edge", "greedy_local"],
        "tensor": {"repeats": {"train": 2}},
        "bootstrap": {"iterations": 40},
        "acceptance": {"max_row_truncation_spread": 0.05},
    }
    config.update(overrides)
    return config


def _space():
    return enumerate_space(S0_ROLES, model=MODEL)


def _latent(exec_hash: str) -> float:
    return 0.4 + 0.4 * random.Random(exec_hash).random()


def _records() -> list[RunRecord]:
    rows = sorted({t.exec_hash for t in _space()})
    rng = random.Random(3)
    out = []
    for split, (n_tasks, repeats) in SPLIT_SIZES.items():
        for exec_hash in rows:
            for i in range(n_tasks):
                for repeat in range(repeats):
                    out.append(RunRecord(
                        run_id=f"{split}-{exec_hash[:6]}-{i}-{repeat}", exp="t_report",
                        model=MODEL, graph_hash=exec_hash, exec_hash=exec_hash,
                        task_id=f"{split}{i}", split_role=split, seed=0, repeat=repeat,
                        score=float(rng.random() < _latent(exec_hash)), parse_fail=False,
                        prediction="1", prompt_tokens=10, gen_tokens=20,
                        reasoning_tokens=None, truncated_nodes=(), latency_s=0.0,
                        attempts=1, node_outputs_path="/dev/null", endpoint_usage={},
                        effective_roles={}))
    return out


def _write_trajectory(runs: Path, optimizer, seed: int) -> None:
    """Drive a real optimizer on synthetic scores and keep what the report reads."""
    rng = random.Random(seed)
    task_ids = tuple(f"train{i}" for i in range(optimizer.budget.tasks_per_eval))
    path = runs / "t_report" / optimizer.name / f"seed{seed}" / "trajectory.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        while optimizer.budget.left():
            candidates = optimizer.propose()
            optimizer.observe([
                EvalResult(c.exec_hash, task_ids,
                           tuple(float(rng.random() < _latent(c.exec_hash))
                                 for _ in task_ids), 10, 20)
                for c in candidates])
            stream.write(json.dumps({"cum_evals": optimizer.budget.spent_evals,
                                     "incumbent": optimizer.incumbent().graph_hash}) + "\n")


def _frontier(n_test: int, paired: bool = True) -> dict:
    scores = {f"test{i}": float(i % 2) for i in range(n_test)}
    headline = {"strategy": "self_consistency", "param": 3, "accuracy": 0.5,
                "val_accuracy": 0.55, "budget_tokens": 100.0, "selected_on": "val"}
    if paired:
        headline["task_scores"] = scores
    return {"headline": headline}


@pytest.fixture
def runs(tmp_path):
    settings = EdgeSettings(group_size=8, lr=0.1, theta_clip=4.0, entropy_reg=0.01)
    space = _space()
    for seed in (0, 1, 2):
        budget = lambda: Budget(60, 4)  # noqa: E731
        _write_trajectory(tmp_path, RandomSearch(space, budget(), seed), seed)
        _write_trajectory(tmp_path, ReinforceEdge(space, budget(), seed, settings=settings), seed)
        _write_trajectory(tmp_path, GreedyLocal(space, budget(), seed), seed)
    return tmp_path


def test_every_optimizer_reaches_the_plateau_test(runs):
    report, rows = build_report(_config(), _records(), runs,
                                frontier=_frontier(SPLIT_SIZES["test"][0]))
    for optimizer in ("random_search", "reinforce_edge", "greedy_local"):
        block = report["optimizers"][optimizer]
        assert block["coverage"]["seeds"] == [0, 1, 2]
        assert report["plateau"][optimizer]["tested_budgets"] > 0, optimizer
    # reinforce_edge's first round ends at 8, so budget 5 is explicitly uncovered ...
    assert report["optimizers"]["reinforce_edge"]["intervals"]["5"]["status"] == "not_covered"
    # ... while the final budget is covered for every optimizer.
    for optimizer in ("random_search", "reinforce_edge", "greedy_local"):
        assert "C" in report["optimizers"][optimizer]["intervals"]["60"]
        assert report["optimizers"][optimizer]["decision"]["60"]["action"]
    assert report["e0_gate"]["identity_closes"]
    assert len(rows) == sum(report["optimizers"][o]["rows"] for o in report["optimizers"])


def test_pi_single_is_paired_when_every_test_task_is_scored(runs):
    report, _ = build_report(_config(), _records(), runs,
                             frontier=_frontier(SPLIT_SIZES["test"][0]))
    assert report["pi_single"]["paired"] and report["e0_gate"]["pi_single_paired"]
    assert report["pi_single"]["accuracy"] == pytest.approx(0.5)


def test_partial_pi_single_scores_fall_back_to_unpaired_and_say_why():
    frontier = _frontier(5)            # only 5 of 20 test tasks
    single = load_pi_single(frontier, [f"test{i}" for i in range(20)])
    assert not single.paired
    assert "5 of 20" in single.selection["unpaired_reason"]


def test_power_is_measured_on_near_optimal_pairs(runs):
    report, _ = build_report(_config(), _records(), runs,
                             frontier=_frontier(SPLIT_SIZES["test"][0]))
    power = report["power"]
    assert power["pairs"].startswith("top 5")
    assert power["variance"] == pytest.approx(power["measured_variance_top_pairs"])
    assert power["n_tasks"] == SPLIT_SIZES["val"][0]


def test_truncation_spread_is_reported_per_split(runs):
    records = _records()
    target = records[0].exec_hash
    rebuilt = []
    for r in records:
        if r.exec_hash == target and r.split_role == "test":
            fields = {name: getattr(r, name) for name in RunRecord.__dataclass_fields__}
            fields["truncated_nodes"] = ("n4",)
            r = RunRecord(**fields)
        rebuilt.append(r)
    report, _ = build_report(_config(), rebuilt, runs,
                             frontier=_frontier(SPLIT_SIZES["test"][0]))
    block = report["truncation"]["test"]
    assert block["max_row"] == pytest.approx(1.0) and block["min_row"] == pytest.approx(0.0)
    assert block["spread_ok"] is False


# ------------------------------------------------------------ budget alignment

def test_on_budget_grid_forward_fills_between_rounds_and_never_extrapolates():
    rounds = {8: "a", 16: "b", 20: "c"}
    grid = on_budget_grid(rounds, range(1, 25))
    assert 7 not in grid                       # nothing chosen before the first round
    assert grid[8] == grid[15] == "a" and grid[16] == "b" and grid[20] == "c"
    assert 21 not in grid                      # the run ended at 20


def test_align_excludes_incomplete_seeds_and_lists_them():
    per_round = {0: {8: "a", 16: "b"}, 1: {8: "a", 12: "b", 16: "c"}, 2: {8: "a"}}
    aligned, axis = align(per_round, max_evals=16)
    assert sorted(aligned) == [0, 1]
    assert axis["excluded_incomplete_seeds"] == [2]
    assert axis["first_common_budget"] == 8 and axis["last_common_budget"] == 16
    assert all(set(aligned[s]) == set(axis["budgets"]) for s in aligned)


def test_plateau_scan_refuses_a_step_some_seed_does_not_cover():
    val = {"a": np.zeros(10), "b": np.ones(10)}
    incumbents = {0: {8: "a", 16: "b"}, 1: {8: "a", 12: "b"}}
    with pytest.raises(ValueError, match="do not cover"):
        scan_plateau("opt", incumbents, val, steps=[8], delta_t=8, epsilon=0.02)
