"""Optimizer trajectories on one common budget axis.

Optimizers spend candidate evaluations in rounds of different sizes: random search and
prune_from_dense spend one per round, reinforce_edge a group of eight, greedy_local a
whole neighbourhood (five to seven, varying). Their ``cum_evals`` grids therefore never
line up. Looking up the incumbent at an exact budget t silently drops every optimizer
whose rounds do not land on t: reinforce_edge never has both t and t+50, because 50 is
not a multiple of 8, so the plateau test returned nothing for it; and greedy_local hit
t=25 in about one seed in twenty, so an interval "at 25" was computed on whichever
seeds happened to align.

The convention here is the anytime one: the incumbent at budget t is the graph the
learner held after its last completed round with ``cum_evals <= t`` -- what it would
hand over if it were stopped at t. A seed covers t only between its first and its last
completed round, so an interrupted run is never extrapolated, and a seed that did not
finish its budget is excluded and listed rather than quietly shortening the axis.
"""

from __future__ import annotations

import json
import pathlib
from typing import Iterable, Mapping


def read_rounds(path: pathlib.Path, graph_to_exec: Mapping[str, str]) -> dict[int, str]:
    """{cum_evals: exec_hash} for one trajectory file, one entry per completed round."""
    rounds: dict[int, str] = {}
    previous = 0
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        cum_evals = int(row["cum_evals"])
        if cum_evals <= previous:
            raise ValueError(f"{path}:{number}: cum_evals must increase strictly")
        graph_hash = row["incumbent"]
        if graph_hash not in graph_to_exec:
            raise ValueError(f"{path}:{number}: incumbent {graph_hash[:12]} is not in the space")
        rounds[cum_evals] = graph_to_exec[graph_hash]
        previous = cum_evals
    return rounds


def read_optimizer(runs_dir: pathlib.Path, exp: str, optimizer: str,
                   graph_to_exec: Mapping[str, str]) -> dict[int, dict[int, str]]:
    """{seed: {cum_evals: exec_hash}} from every ``seed*/trajectory.jsonl``."""
    out: dict[int, dict[int, str]] = {}
    for path in sorted((runs_dir / exp / optimizer).glob("seed*/trajectory.jsonl")):
        seed = int(path.parent.name.removeprefix("seed"))
        rounds = read_rounds(path, graph_to_exec)
        if rounds:
            out[seed] = rounds
    return out


def on_budget_grid(rounds: Mapping[int, str], budgets: Iterable[int]) -> dict[int, str]:
    """Forward-fill one seed's per-round incumbents onto integer budgets.

    Budgets before the first completed round or after the last one are absent, never
    filled: before the first round no evaluation has been spent on a choice, and after
    the last one the run did not happen.
    """
    marks = sorted(rounds)
    if not marks:
        return {}
    out: dict[int, str] = {}
    index = -1
    for budget in sorted(set(budgets)):
        while index + 1 < len(marks) and marks[index + 1] <= budget:
            index += 1
        if index >= 0 and budget <= marks[-1]:
            out[budget] = rounds[marks[index]]
    return out


def align(per_round: Mapping[int, Mapping[int, str]], max_evals: int
          ) -> tuple[dict[int, dict[int, str]], dict]:
    """Common-grid incumbents for every complete seed, plus what was left out and why.

    Returns ``({seed: {budget: exec_hash}}, coverage)``. Every returned seed has an
    entry at every budget in ``coverage["budgets"]``, so downstream code may index any
    (seed, budget) pair without a membership test.
    """
    complete = {seed: rounds for seed, rounds in per_round.items()
                if rounds and max(rounds) == max_evals}
    incomplete = sorted(set(per_round) - set(complete))
    grid = range(1, max_evals + 1)
    filled = {seed: on_budget_grid(rounds, grid) for seed, rounds in complete.items()}
    budgets = sorted(set.intersection(*(set(v) for v in filled.values()))) if filled else []
    aligned = {seed: {b: filled[seed][b] for b in budgets} for seed in sorted(filled)}
    coverage = {
        "seeds": sorted(aligned),
        "excluded_incomplete_seeds": incomplete,
        "first_common_budget": budgets[0] if budgets else None,
        "last_common_budget": budgets[-1] if budgets else None,
        "budgets": budgets,
        "round_sizes": sorted({b - a for rounds in complete.values()
                               for a, b in zip([0, *sorted(rounds)], sorted(rounds))}),
    }
    return aligned, coverage
