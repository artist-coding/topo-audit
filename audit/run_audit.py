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

from data.loaders.math import Task, load_pool, tasks_from_frame
from runtime.cache import RunCache
from runtime.endpoints import endpoint_from_config
from runtime.executor import ExecutionError, Executor
from space.s0 import S0_ROLES, enumerate_space


def distinct_rows(model: str) -> list:
    """One representative topology per exec_hash: the rows we actually pay for."""
    by_exec: dict[str, object] = {}
    for topology in enumerate_space(S0_ROLES, model=model):
        by_exec.setdefault(topology.exec_hash, topology)
    return list(by_exec.values())


def load_tasks(config: dict, roles: list[str]) -> dict[str, list[Task]]:
    split_file = pathlib.Path(config["task"]["splits"])
    split = json.loads(split_file.read_text())
    answers, levels = split["answers"], split["levels_by_task"]
    wanted = {tid for role in roles for tid in split["splits"][role]}

    frames = []
    for source in ("train", "test"):
        frame = load_pool("data/raw/math_hendrycks.parquet", source, (4, 5))
        frames.append(frame[frame["task_id"].isin(wanted)])
    import pandas as pd
    pool = pd.concat(frames, ignore_index=True).drop_duplicates(subset="task_id")

    out: dict[str, list[Task]] = {}
    for role in roles:
        ids = set(split["splits"][role])
        frame = pool[pool["task_id"].isin(ids)]
        if len(frame) != len(ids):
            raise SystemExit(f"{role}: resolved {len(frame)} of {len(ids)} task ids; "
                             "re-run data/fetch_math.py")
        tasks = list(tasks_from_frame(frame, role))
        for task in tasks:
            if task.answer != str(answers[task.task_id]):
                raise SystemExit(f"answer drift for {task.task_id}")
            if task.meta["level"] != levels[task.task_id]:
                raise SystemExit(f"level drift for {task.task_id}")
        out[role] = tasks
    return out


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

    done = {"n": 0, "failed": 0, "truncated": 0, "parse_fail": 0,
            "prompt_tokens": 0, "gen_tokens": 0}
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
        with lock:
            done["n"] += 1
            done["truncated"] += bool(record.truncated_nodes)
            done["parse_fail"] += bool(record.parse_fail)
            done["prompt_tokens"] += record.prompt_tokens
            done["gen_tokens"] += record.gen_tokens
            if done["n"] % 500 == 0:
                elapsed = time.monotonic() - started
                rate = done["n"] / elapsed
                left = (len(cells) - done["n"]) / rate / 3600 if rate else float("nan")
                print(f"  {done['n']:,}/{len(cells):,}  {rate:.1f} runs/s  "
                      f"{left:.1f}h left  trunc {done['truncated'] / done['n']:.1%}  "
                      f"fail {done['failed']}", flush=True)

    with cf.ThreadPoolExecutor(args.concurrency or config["model"]["max_concurrency"]) as ex:
        list(ex.map(execute, cells))

    stats = cache.stats()
    manifest = {
        "exp": exp, "config": args.config, "splits": roles,
        "distinct_computations": len(rows),
        "cells": len(cells), "node_calls": node_calls,
        "wall_s": time.monotonic() - started,
        "cache": stats, **done,
        "truncation_rate": done["truncated"] / max(done["n"], 1),
        "parse_fail_rate": done["parse_fail"] / max(done["n"], 1),
        "acceptance": {
            "max_truncation_rate": config["acceptance"]["max_truncation_rate"],
            "truncation_ok": (done["truncated"] / max(done["n"], 1)
                              <= config["acceptance"]["max_truncation_rate"]),
            "no_failures": done["failed"] == 0,
        },
    }
    path = pathlib.Path(f"runs/{exp}/manifest_audit.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1))
    print(f"\n{done['n']:,} runs, {done['failed']} failed, "
          f"truncation {manifest['truncation_rate']:.1%} "
          f"(gate {config['acceptance']['max_truncation_rate']:.0%}), "
          f"{done['prompt_tokens'] + done['gen_tokens']:,} tokens")
    print(f"cache: {stats}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
