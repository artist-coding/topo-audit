"""Turn cached RunRecords into the score matrices the decomposition reads.

`analysis/` never calls an endpoint. It reads the cache, averages repeats into one
value per (computation, task) cell, and hands the result to decompose.py. Keeping the
aggregation here means the decomposition never sees a repeat index and cannot
accidentally treat two draws of one cell as two observations.

Cells the tensor never reached stay NaN rather than becoming zeros. A zero would be
read as "this graph got the answer wrong", which is a different claim from "we never
ran it", and the difference lands directly in the terms.
"""

from __future__ import annotations

import collections
from typing import Iterable, Sequence

import numpy as np

from analysis.decompose import ScoreMatrix
from runtime.executor import RunRecord


def build_matrix(records: Iterable[RunRecord], split_role: str,
                 *, exec_hashes: Sequence[str] | None = None,
                 task_ids: Sequence[str] | None = None) -> ScoreMatrix:
    """Mean score per (exec_hash, task) over repeats, for one split."""
    by_cell: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    seen_rows: set[str] = set()
    seen_tasks: set[str] = set()
    for record in records:
        if record.split_role != split_role:
            continue
        by_cell[(record.exec_hash, record.task_id)].append(record.score)
        seen_rows.add(record.exec_hash)
        seen_tasks.add(record.task_id)

    rows = tuple(exec_hashes) if exec_hashes is not None else tuple(sorted(seen_rows))
    cols = tuple(task_ids) if task_ids is not None else tuple(sorted(seen_tasks))
    values = np.full((len(rows), len(cols)), np.nan)
    for i, exec_hash in enumerate(rows):
        for j, task_id in enumerate(cols):
            scores = by_cell.get((exec_hash, task_id))
            if scores:
                values[i, j] = float(np.mean(scores))
    return ScoreMatrix(rows, cols, values)


def repeat_cube(records: Iterable[RunRecord], split_role: str,
                repeats: int) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray]:
    """(graph, task, repeat) array, for the within-cell variance that bounds recycling.

    Seeds share tensor cells once the R draws are used up, so the seed-level
    confidence interval understates a full rerun. This cube is how much sampling noise
    was recycled.
    """
    cells: dict[tuple[str, str, int], float] = {}
    rows, cols = set(), set()
    for record in records:
        if record.split_role != split_role:
            continue
        cells[(record.exec_hash, record.task_id, record.repeat)] = record.score
        rows.add(record.exec_hash)
        cols.add(record.task_id)
    row_ids, col_ids = tuple(sorted(rows)), tuple(sorted(cols))
    cube = np.full((len(row_ids), len(col_ids), repeats), np.nan)
    for (exec_hash, task_id, repeat), score in cells.items():
        if repeat < repeats:
            cube[row_ids.index(exec_hash), col_ids.index(task_id), repeat] = score
    return row_ids, col_ids, cube


def coverage(matrix: ScoreMatrix) -> dict:
    """What fraction of the tensor actually exists, per row and overall."""
    present = ~np.isnan(matrix.values)
    return {
        "rows": len(matrix.exec_hashes),
        "tasks": len(matrix.task_ids),
        "filled": float(present.mean()),
        "empty_rows": [matrix.exec_hashes[i]
                       for i in range(len(matrix.exec_hashes))
                       if not present[i].any()],
        "min_row_coverage": float(present.mean(axis=1).min()) if present.size else 0.0,
    }


def truncation_summary(records: Iterable[RunRecord], rows_order: Sequence[str] | None,
                       max_spread: float | None) -> dict:
    """Per-row truncation rate on every split.

    A truncated node zeroes its whole run, so a deep graph has more chances to be
    zeroed than a single node: with a 3% per-node rate a four-node chain loses about
    11.5% of its runs to truncation alone. The *spread* across rows, not the overall
    rate, is what leaks into the comparison between graphs.
    """
    counts: dict[tuple[str, str], list[int]] = {}
    for record in records:
        cell = counts.setdefault((record.split_role, record.exec_hash), [0, 0])
        cell[0] += bool(record.truncated_nodes)
        cell[1] += 1
    out: dict = {}
    for split in sorted({split for split, _ in counts}):
        rows = rows_order if rows_order is not None else sorted(
            {h for s, h in counts if s == split})
        rates = {h: counts[(split, h)][0] / counts[(split, h)][1]
                 for h in rows if (split, h) in counts}
        if not rates:
            continue
        values = np.array(list(rates.values()))
        spread = float(values.max() - values.min())
        worst = sorted(rates, key=lambda h: -rates[h])[:3]
        out[split] = {
            "overall": float(sum(counts[(split, h)][0] for h in rates)
                             / sum(counts[(split, h)][1] for h in rates)),
            "max_row": float(values.max()), "min_row": float(values.min()),
            "spread": spread,
            "worst_rows": {h[:12]: rates[h] for h in worst},
            "spread_ok": None if max_spread is None else spread <= max_spread,
        }
    return out
