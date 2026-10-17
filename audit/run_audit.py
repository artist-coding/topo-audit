"""Build the score tensor: every distinct computation in S0, on every split.

This is the one expensive step, and everything downstream is a lookup into it. The
tensor is addressed by `exec_hash`, so the 34 labelled DAGs of S0 cost only 25 rows --
role degradation makes nine of them duplicate computations, and paying twice for a
byte-identical prompt would buy nothing.

What the tensor covers, and why each part exists:

    train  256 x R=8   the optimizers' whole world; R draws per cell so that a
                       learner re-evaluating a graph sees fresh sampling noise
                       rather than a memoised constant
    audit  300 x 2     selection of G_star, and the cross-fitting that measures the
                       winner's curse
    val    300 x 2     learning curves and the plateau test
    test  1000 x 1     the population levels in the decomposition; the 1,000
                       include all 262 MATH-500 Level 4-5 problems as an anchor

Once it exists, every optimizer, seed, budget and N value replays offline for free,
provided the search seed stays out of the cache key (see Executor.sample_seed).
Seeds are 20, not 50: replay makes them cheap but not independent -- they share tensor
cells once the R draws per cell are used -- and in the converged regime a plateau test
runs in, seeds do not reduce the task-level noise at all (see analysis/plateau.py).

Resumability is the cache's job, not this script's: re-running issues zero endpoint
calls for cells already present, so an interrupted sweep is restarted by rerunning
the same command.

Usage:
    python3 audit/run_audit.py --config configs/e0.yaml [--splits train,audit]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import yaml

from analysis.matrices import truncation_summary
from data.loaders.math import Task
from data.tasks import load_split_tasks
from runtime.cache import RunCache, key_for
from runtime.endpoints import endpoint_from_config, usage_totals
from runtime.executor import ExecutionError, Executor
from space.s0 import S0_ROLES, enumerate_space


def distinct_rows(model: str) -> list:
    """One representative topology per exec_hash: the rows we actually pay for."""
    by_exec: dict[str, object] = {}
    for topology in enumerate_space(S0_ROLES, model=model):
        by_exec.setdefault(topology.exec_hash, topology)
    return list(by_exec.values())


def load_tasks(config: dict, roles: list[str]) -> dict[str, list[Task]]:
    """Kept for callers of the old name; the implementation lives in data/tasks.py."""
    return load_split_tasks(config, roles)


def build_tensor(config: dict, executor: Executor, cells: list, *, concurrency: int,
                 roles: list[str], n_rows: int, config_path: str = "") -> dict:
    """Run every cell through the Executor and return the audit manifest.

    Separated from argument parsing so the accounting (fresh vs cached tokens, the
    per-row truncation gate, token reconciliation) can be tested with a mock endpoint.
    """
    cache, exp = executor.cache, executor.exp
    node_calls = sum(len(topology.nodes) for topology, *_ in cells)
    # Cells already in the cache cost nothing now; keep them out of this run's bill.
    cached_before = cache.keys(exp)

    done = {"n": 0, "failed": 0, "truncated": 0, "parse_fail": 0,
            "prompt_tokens": 0, "gen_tokens": 0,
            "fresh_runs": 0, "fresh_prompt_tokens": 0, "fresh_gen_tokens": 0,
            "fresh_raw_usage_tokens": 0}
    records: list = []
    lock = threading.Lock()
    started = time.monotonic()

    def execute(cell) -> None:
        topology, task, seed, repeat = cell
        try:
            record = executor.run(topology, task, seed, repeat)
        except ExecutionError as exc:
            with lock:
                done["n"] += 1
                done["failed"] += 1
            print(f"  FAILED {exc.record.run_id}: {exc.record.node_outputs_path}")
            return
        fresh = key_for(record) not in cached_before
        raw = usage_totals(usage for usages in record.endpoint_usage.values()
                           for usage in usages)
        with lock:
            records.append(record)
            done["n"] += 1
            done["truncated"] += bool(record.truncated_nodes)
            done["parse_fail"] += bool(record.parse_fail)
            done["prompt_tokens"] += record.prompt_tokens
            done["gen_tokens"] += record.gen_tokens
            if fresh:
                done["fresh_runs"] += 1
                done["fresh_prompt_tokens"] += record.prompt_tokens
                done["fresh_gen_tokens"] += record.gen_tokens
                done["fresh_raw_usage_tokens"] += raw[0] + raw[1]
            if done["n"] % 500 == 0:
                elapsed = time.monotonic() - started
                rate = done["n"] / elapsed
                left = (len(cells) - done["n"]) / rate / 3600 if rate else float("nan")
                print(f"  {done['n']:,}/{len(cells):,}  {rate:.1f} runs/s  "
                      f"{left:.1f}h left  trunc {done['truncated'] / done['n']:.1%}  "
                      f"fail {done['failed']}", flush=True)

    with cf.ThreadPoolExecutor(concurrency) as ex:
        list(ex.map(execute, cells))

    stats = cache.stats()
    acceptance = config["acceptance"]
    per_row = truncation_summary(records, None, acceptance.get("max_row_truncation_spread"))
    unscored = list(cache.export_attempts(exp))
    fresh_total = done["fresh_prompt_tokens"] + done["fresh_gen_tokens"]
    reconciliation_error = (abs(fresh_total - done["fresh_raw_usage_tokens"])
                            / max(done["fresh_raw_usage_tokens"], 1))
    manifest = {
        "exp": exp, "config": config_path, "splits": roles,
        "distinct_computations": n_rows,
        "cells": len(cells), "node_calls": node_calls,
        "wall_s": time.monotonic() - started,
        "cache": stats, **done,
        "truncation_rate": done["truncated"] / max(done["n"], 1),
        "parse_fail_rate": done["parse_fail"] / max(done["n"], 1),
        "truncation_by_row": per_row,
        "tokens": {
            "note": "logical = every requested cell, including ones cached by earlier "
                    "runs; fresh = generated by this invocation; unscored = spent by "
                    "failed or interrupted runs that hold no cell. The gateway bill "
                    "should match fresh + unscored.",
            "logical": done["prompt_tokens"] + done["gen_tokens"],
            "fresh": fresh_total,
            "unscored_attempts": sum(r.prompt_tokens + r.gen_tokens for r in unscored),
            "unscored_attempt_runs": len(unscored),
        },
        "acceptance": {
            "max_truncation_rate": acceptance["max_truncation_rate"],
            "truncation_ok": (done["truncated"] / max(done["n"], 1)
                              <= acceptance["max_truncation_rate"]),
            "max_row_truncation_spread": acceptance.get("max_row_truncation_spread"),
            "row_truncation_spread_ok": all(
                block["spread_ok"] is not False for block in per_row.values()),
            "token_reconciliation_error": reconciliation_error,
            "token_reconciliation_ok": (reconciliation_error
                                        <= acceptance["token_reconciliation_error"]),
            "no_failures": done["failed"] == 0,
        },
    }
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/e0.yaml")
    ap.add_argument("--splits", default="train,audit,val,test")
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--limit-tasks", type=int, default=0,
                    help="smoke test: first N tasks per split, all rows and repeats")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--exp", default=None,
                    help="override config exp; use a separate one for smoke tests")
    args = ap.parse_args()

    config = yaml.safe_load(pathlib.Path(args.config).read_text())
    roles = args.splits.split(",")
    repeats = config["tensor"]["repeats"]
    exp = args.exp or config["exp"]

    rows = distinct_rows(config["model"]["name"])
    tasks = load_tasks(config, roles)

    if args.limit_tasks:
        tasks = {role: items[:args.limit_tasks] for role, items in tasks.items()}
    cells = [
        (topology, task, 0, repeat)
        for role in roles
        for topology in rows
        for task in tasks[role]
        for repeat in range(repeats[role])
    ]

    node_calls = sum(len(topology.nodes) for topology, *_ in cells)
    print(f"tensor: {len(rows)} distinct computations x "
          + " + ".join(f"{role} {len(tasks[role])}x{repeats[role]}" for role in roles)
          + f" = {len(cells):,} runs, {node_calls:,} node calls")
    if args.dry_run:
        return

    endpoint = endpoint_from_config(config["model"], concurrency=args.concurrency)
    cache = RunCache(f"runs/{exp}/cache.sqlite", exp=exp)
    executor = Executor(
        endpoint, cache, max_tokens_per_node=config["model"]["max_tokens_per_node"],
        temperature=config["model"]["temperature"],
        train_repeats=repeats["train"], runs_dir="runs")
    manifest = build_tensor(
        config, executor, cells, roles=roles, n_rows=len(rows), config_path=args.config,
        concurrency=args.concurrency or config["model"]["max_concurrency"])
    done, per_row, stats = manifest, manifest["truncation_by_row"], manifest["cache"]
    fresh_total = manifest["tokens"]["fresh"]
    reconciliation_error = manifest["acceptance"]["token_reconciliation_error"]
    path = pathlib.Path(f"runs/{exp}/manifest_audit.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1))
    print(f"\n{done['n']:,} runs, {done['failed']} failed, "
          f"truncation {manifest['truncation_rate']:.1%} "
          f"(gate {config['acceptance']['max_truncation_rate']:.0%}), "
          f"{done['prompt_tokens'] + done['gen_tokens']:,} tokens")
    for split, block in per_row.items():
        flag = "" if block["spread_ok"] is not False else "  <-- exceeds the row-spread gate"
        print(f"  truncation {split}: overall {block['overall']:.1%}, rows "
              f"{block['min_row']:.1%}..{block['max_row']:.1%}{flag}")
    print(f"tokens: fresh {fresh_total:,}, unscored attempts "
          f"{manifest['tokens']['unscored_attempts']:,}, reconciliation error "
          f"{reconciliation_error:.2e}")
    print(f"cache: {stats}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
