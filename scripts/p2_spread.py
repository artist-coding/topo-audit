"""P2: is there anything for the decomposition to find, and can val resolve it?

Run before paying for the tensor, on problems held out of every split:

    python3 scripts/p2_spread.py --config configs/e0.yaml --pool math --n 150
    python3 scripts/p2_spread.py --config configs/e0.yaml --pool deepmath --band 8.5

The original gate was "best graph minus worst graph >= 5 points". That is necessary
but not what the terms depend on:

- **E and O live at the top of the space.** If the best five computations are within
  a point or two, any reasonable learner lands near the optimum and E and O are small
  by construction, however bad the worst graph is. So the probe reports best-minus-
  second and best-minus-fifth, with paired standard errors.
- **C is best graph minus pi_single at matched tokens**, not best minus G_empty. The
  probe runs the declared single-agent strategies on the same problems and reports C
  cross-fitted (pick both sides on one half of the problems, score on the other), so
  the winner's curse of picking the best of 25 does not inflate it.
- **The plateau test's resolution is set by Var(d) of near-optimal pairs.** The probe
  measures it on the top-5 pairs and converts it into the validation size needed for
  each candidate epsilon.

The probe uses its own exp (default ``p2_<pool>``), so nothing it runs is ever read by
the E0 analysis. Graphs default to all 25 computations at one repeat; ``--graphs
hand6`` runs the six hand-picked graphs of the pre-experiment table instead (cheaper,
but it can miss the top of the space).
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
from collections import Counter
from typing import Sequence

if __package__ in (None, ""):  # run as a script: make the repo root importable
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from analysis.bootstrap import (  # noqa: E402
    paired_se,
    paired_variance,
    required_tasks,
    z_value,
)
from data.loaders.deepmath import BAND_WIDTH  # noqa: E402
from data.loaders.math import Task, load_pool, tasks_from_frame  # noqa: E402
from runtime.cache import key_for  # noqa: E402
from runtime.single_agent import SingleAgentRunner, StrategyRun, _accumulate  # noqa: E402
from space.s0 import S0_ROLES, declared_nodes  # noqa: E402
from space.topology import Topology  # noqa: E402

HAND6 = {
    "G_empty": [],
    "solver->finalizer": [("n1", "n4")],
    "chain S->C->R->F": [("n1", "n2"), ("n2", "n3"), ("n3", "n4")],
    "vote (three->F)": [("n1", "n4"), ("n2", "n4"), ("n3", "n4")],
    "S->C->F": [("n1", "n2"), ("n2", "n4")],
    "full (6 edges)": [("n1", "n2"), ("n1", "n3"), ("n1", "n4"),
                       ("n2", "n3"), ("n2", "n4"), ("n3", "n4")],
}


# ----------------------------------------------------------------- problems

def probe_tasks(config: dict, pool: str, n: int, seed: int,
                band: float | None = None, width: float = BAND_WIDTH) -> list[Task]:
    """n problems held out of every committed split (MATH) or from one DeepMath band."""
    rng = np.random.default_rng(seed)
    if pool == "math":
        split = json.loads(pathlib.Path(config["task"]["splits"]).read_text())
        assigned = {tid for ids in split["splits"].values() for tid in ids}
        frame = load_pool(config["task"].get("raw", "data/raw/math_hendrycks.parquet"),
                          "train", tuple(split.get("levels", (4, 5))))
        held_out = frame[~frame["task_id"].isin(assigned)].reset_index(drop=True)
        if len(held_out) < n:
            raise SystemExit(f"MATH has {len(held_out)} held-out problems < requested {n}; "
                             "reduce --n or provide a larger held-out pool")
        picked = held_out.iloc[rng.permutation(len(held_out))[:n]]
        return list(tasks_from_frame(picked, "audit"))
    if pool == "deepmath":
        from data.loaders.deepmath import DEFAULT_RAW, load_band, tasks_from_band_frame

        if band is None:
            raise SystemExit("--band is required for the deepmath pool")
        frame = load_band(config["task"].get("deepmath_raw", DEFAULT_RAW), band, width)
        if len(frame) < n:
            raise SystemExit(f"DeepMath band {band} has {len(frame)} non-binary problems "
                             f"< requested {n}; download more shards or reduce --n")
        picked = frame.iloc[rng.permutation(len(frame))[:n]]
        return list(tasks_from_band_frame(picked, "audit"))
    raise SystemExit(f"unknown pool {pool!r}")


def probe_graphs(model: str, which: str) -> dict[str, Topology]:
    """{label: topology}, one per distinct computation."""
    nodes = declared_nodes(S0_ROLES, model)
    if which == "hand6":
        graphs = {label: Topology.canonical(nodes, edges, space_id="s4")
                  for label, edges in HAND6.items()}
    elif which == "all":
        from audit.run_audit import distinct_rows

        graphs = {str(t): t for t in distinct_rows(model)}
    else:
        raise SystemExit(f"unknown graph set {which!r}")
    seen, out = set(), {}
    for label, topology in graphs.items():
        if topology.exec_hash not in seen:
            seen.add(topology.exec_hash)
            out[label] = topology
    return out


# ----------------------------------------------------------------- analysis

def require_complete_probe(records, graphs, task_ids, repeats: int) -> None:
    """No ranking or gate may use a tensor with missing/duplicate/nonfinite cells."""
    if repeats < 1 or len(task_ids) < 2 or len(set(task_ids)) != len(task_ids):
        raise ValueError("P2 needs positive repeats and at least two distinct tasks")
    expected = {(g.exec_hash, tid, repeat) for g in graphs.values()
                for tid in task_ids for repeat in range(repeats)}
    counts = Counter((r.exec_hash, r.task_id, r.repeat) for r in records)
    missing = sorted(expected - counts.keys())
    invalid = {key for key, count in counts.items() if count != 1 or key not in expected}
    invalid.update((r.exec_hash, r.task_id, r.repeat) for r in records
                   if not np.isfinite(r.score))
    if missing or invalid:
        raise ValueError(
            f"Incomplete P2 tensor: {len(missing)} missing, {len(invalid)} invalid cells; "
            f"examples: {(missing + sorted(invalid))[:3]}. "
            "Rerun the same probe to fill missing cells from cache before ranking.")


def _per_task(records, exec_hash: str, task_ids: Sequence[str]) -> np.ndarray:
    by_task: dict[str, list[float]] = {}
    for record in records:
        if record.exec_hash == exec_hash:
            by_task.setdefault(record.task_id, []).append(record.score)
    return np.array([np.mean(by_task[t]) if t in by_task else np.nan for t in task_ids])


def summarize_probe(records, single_runs: Sequence[StrategyRun],
                    graphs: dict[str, Topology], task_ids: Sequence[str], *,
                    budget_tokens: float | None = None, val_tasks: int = 300,
                    epsilons: Sequence[float] = (0.02, 0.03, 0.04),
                    top_gap: float = 0.02, old_gate: float = 0.05,
                    max_row_truncation_spread: float | None = 0.03,
                    splits: int = 50, seed: int = 20261018, repeats: int = 1) -> dict:
    """Everything P2 decides on, from the probe's records. Pure: no endpoint."""
    task_ids = list(task_ids)
    require_complete_probe(records, graphs, task_ids, repeats)
    rows = {label: _per_task(records, t.exec_hash, task_ids) for label, t in graphs.items()}
    stats = {}
    for label, topology in graphs.items():
        mine = [r for r in records if r.exec_hash == topology.exec_hash]
        stats[label] = {
            "accuracy": float(np.nanmean(rows[label])),
            "se": float(np.nanstd(rows[label], ddof=1) / np.sqrt(np.sum(~np.isnan(rows[label])))),
            "nodes": len(topology.nodes),
            "mean_tokens": float(np.mean([r.prompt_tokens + r.gen_tokens for r in mine])),
            "truncation_rate": float(np.mean([bool(r.truncated_nodes) for r in mine])),
            "runs": len(mine),
        }
    ranked = sorted(stats, key=lambda label: -stats[label]["accuracy"])
    best = ranked[0]

    def gap(label: str) -> dict:
        return {"graph": label,
                "diff": stats[best]["accuracy"] - stats[label]["accuracy"],
                "paired_se": paired_se(rows[best], rows[label])}

    spread = {"best": best, "second": gap(ranked[1]) if len(ranked) > 1 else None,
              "fifth": gap(ranked[4]) if len(ranked) > 4 else None,
              "worst": gap(ranked[-1]),
              "within_2_points_of_best": sum(
                  stats[best]["accuracy"] - stats[label]["accuracy"] <= 0.02
                  for label in ranked)}

    top = ranked[:5]
    variance = float(np.nanmean([paired_variance(rows[a], rows[b])
                                 for a, b in itertools.combinations(top, 2)]))
    z = z_value(0.95)
    power = {
        "pairs": f"top {len(top)} graphs on the probe",
        "variance": variance,
        "floor_at_val_tasks": {"val_tasks": val_tasks,
                               "floor": z * float(np.sqrt(variance / val_tasks))},
        "val_tasks_needed": {f"{eps:.2f}": required_tasks(eps, variance=variance)
                             for eps in epsilons} if variance > 0 else {},
    }

    truncation = [stats[label]["truncation_rate"] for label in ranked]
    trunc = {"min_row": min(truncation), "max_row": max(truncation),
             "spread": max(truncation) - min(truncation),
             "spread_ok": (None if max_row_truncation_spread is None else
                           max(truncation) - min(truncation) <= max_row_truncation_spread)}

    capacity = _capacity(rows, single_runs, task_ids, budget_tokens
                         if budget_tokens is not None
                         else float(np.median([s["mean_tokens"] for s in stats.values()])),
                         splits=splits, seed=seed)

    second_or_fifth = spread["fifth"] or spread["second"]
    return {
        "tasks": len(task_ids),
        # make_splits.py --exclude-p2 keeps these out of every role of the real split.
        "task_ids": task_ids,
        "graphs": stats,
        "ranking": ranked,
        "spread": spread,
        "capacity": capacity,
        "power": power,
        "truncation": trunc,
        "gates": {
            "old_best_minus_worst": {"threshold": old_gate,
                                     "pass": spread["worst"]["diff"] >= old_gate},
            "top_end_not_flat": {"threshold": top_gap,
                                 "measure": "best - fifth (or second)",
                                 "pass": bool(second_or_fifth
                                              and second_or_fifth["diff"] >= top_gap)},
            "capacity_positive": {
                "pass": (capacity["cross_fitted"] > 0
                         if "cross_fitted" in capacity else None)},
            "row_truncation_spread": {"pass": trunc["spread_ok"]},
        },
    }


def _capacity(rows: dict[str, np.ndarray], single_runs: Sequence[StrategyRun],
              task_ids: Sequence[str], budget_tokens: float, *, splits: int,
              seed: int) -> dict:
    """C on the probe, cross-fitted over random halves of the problems.

    On each half the best graph and the best affordable pi_single point are chosen,
    and their difference is scored on the other half. Choosing both on the same
    problems they are scored on would hand C the winner's curse of a max over 25.
    """
    if not single_runs:
        return {"note": "pi_single not run"}
    index = {t: i for i, t in enumerate(task_ids)}
    points = sorted({(r.strategy, r.param) for r in single_runs})
    single_rows = {}
    for point in points:
        vector = np.full(len(task_ids), np.nan)
        for run in single_runs:
            if (run.strategy, run.param) == point and run.task_id in index:
                vector[index[run.task_id]] = run.score
        single_rows[point] = vector
    tokens = {point: float(np.mean([r.total_tokens for r in single_runs
                                    if (r.strategy, r.param) == point]))
              for point in points}
    affordable = [p for p in points if tokens[p] <= budget_tokens]
    if not affordable:
        return {"budget_tokens": budget_tokens, "note": "no pi_single point fits"}
    rng = np.random.default_rng(seed)
    n = len(task_ids)
    held = []
    for _ in range(splits):
        order = rng.permutation(n)
        for pick, score in ((order[: n // 2], order[n // 2:]), (order[n // 2:], order[: n // 2])):
            graph = max(rows, key=lambda g: (np.nanmean(rows[g][pick]), g))
            single = max(affordable, key=lambda p: (np.nanmean(single_rows[p][pick]), p))
            held.append(float(np.nanmean(rows[graph][score])
                              - np.nanmean(single_rows[single][score])))
    full_graph = max(rows, key=lambda g: (np.nanmean(rows[g]), g))
    full_single = max(affordable, key=lambda p: (np.nanmean(single_rows[p]), p))
    return {
        "budget_tokens": budget_tokens,
        "cross_fitted": float(np.mean(held)),
        "cross_fitted_range": [float(np.percentile(held, 5)), float(np.percentile(held, 95))],
        "naive": {"graph": full_graph, "pi_single": list(full_single),
                  "diff": float(np.nanmean(rows[full_graph])
                                - np.nanmean(single_rows[full_single])),
                  "paired_se": paired_se(rows[full_graph], single_rows[full_single]),
                  "note": "both sides chosen on the same problems; optimistic"},
    }


def run_probe(config: dict, executor, runner: SingleAgentRunner | None,
              tasks: Sequence[Task], graphs: dict[str, Topology], *, repeats: int,
              concurrency: int, budget_tokens: float | None = None) -> dict:
    """Execute the probe through the Executor (and pi_single), then summarize."""
    from concurrent.futures import ThreadPoolExecutor

    from audit.run_audit import build_tensor

    task_ids = [t.task_id for t in tasks]
    if repeats < 1 or len(task_ids) < 2 or len(set(task_ids)) != len(task_ids):
        raise ValueError("P2 needs positive repeats and at least two distinct tasks")
    cells = [(topology, task, 0, repeat) for topology in graphs.values()
             for task in tasks for repeat in range(repeats)]
    manifest = build_tensor(config, executor, cells, concurrency=concurrency,
                            roles=["probe"], n_rows=len(graphs))
    wanted = {t.task_id for t in tasks}
    hashes = {t.exec_hash for t in graphs.values()}
    records = [r for r in executor.cache.export(executor.exp)
               if r.task_id in wanted and r.exec_hash in hashes and r.repeat < repeats
               and r.seed == executor.sample_seed]
    # Do not spend more on pi_single or emit selection gates after failed graph calls.
    require_complete_probe(records, graphs, task_ids, repeats)
    single_runs: list[StrategyRun] = []
    single_tokens = {"logical": 0, "fresh": 0, "calls": 0}
    if runner is not None:
        sweep = sorted({int(k) for k in config["single_agent"]["budget_sweep"]})
        runner.cache.bind_experiment(
            {"selfcheck": runner.templates["selfcheck.v1"], "samples": runner.samples,
             "budget_sweep": sweep, "refine_rounds": [1, 2, 3], "seed": 0},
            scope="pi_single")
        cached_before = runner.cache.keys(runner.exp)
        params = {"self_consistency": sweep, "best_of_n_selfcheck": sweep}
        with ThreadPoolExecutor(max(1, concurrency)) as pool:
            for runs in pool.map(lambda task: runner.run_task(task, 0, params), tasks):
                single_runs += runs
        # Frontier points share draws and checks. Count cached physical calls once,
        # rather than summing the overlapping logical costs of every strategy.
        calls = [r for r in runner.cache.export(runner.exp)
                 if r.split_role == "pi_single" and r.task_id in wanted and r.seed == 0]
        single_tokens = {
            "logical": sum(r.prompt_tokens + r.gen_tokens for r in calls),
            "fresh": sum(r.prompt_tokens + r.gen_tokens for r in calls
                         if key_for(r) not in cached_before),
            "calls": len(calls),
        }
    summary = summarize_probe(
        records, single_runs, graphs, [t.task_id for t in tasks],
        budget_tokens=budget_tokens, repeats=repeats,
        val_tasks=config.get("splits", {}).get("val", 300),
        max_row_truncation_spread=config["acceptance"].get("max_row_truncation_spread"))
    summary["single_agent_frontier"] = [
        {"strategy": p.strategy, "param": p.param, "accuracy": p.accuracy,
         "mean_tokens": p.mean_tokens} for p in _accumulate(single_runs)]
    topology_tokens = manifest["tokens"]
    summary["tokens"] = {
        "logical": topology_tokens["logical"] + single_tokens["logical"],
        "fresh": topology_tokens["fresh"] + single_tokens["fresh"],
        "topology": topology_tokens,
        "pi_single": single_tokens,
        "note": "Totals include topology and unique pi_single calls (shared frontier "
                "draws counted once); fresh excludes cache hits. Historical unscored "
                "topology attempts are reported separately under topology.",
    }
    return summary


def gate_label(passed: bool | None) -> str:
    return "n/a" if passed is None else ("PASS" if passed else "FAIL")


def main() -> None:
    import yaml

    from runtime.cache import RunCache
    from runtime.endpoints import endpoint_from_config
    from runtime.executor import Executor
    from runtime.single_agent import load_templates

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/e0.yaml")
    ap.add_argument("--pool", choices=("math", "deepmath"), default="math")
    ap.add_argument("--band", type=float, default=None, help="DeepMath band lower edge")
    ap.add_argument("--band-width", type=float, default=BAND_WIDTH)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--graphs", choices=("all", "hand6"), default="all")
    ap.add_argument("--no-single", action="store_true", help="skip pi_single")
    ap.add_argument("--budget", type=float, default=None,
                    help="pi_single token budget; default is the graphs' median cost")
    ap.add_argument("--exp", default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20261018)
    args = ap.parse_args()

    config = yaml.safe_load(pathlib.Path(args.config).read_text())
    model = config["model"]
    exp = args.exp or f"p2_{args.pool}" + (f"_{args.band}" if args.band is not None else "")
    if args.exp is None and args.pool == "deepmath" and args.band_width != BAND_WIDTH:
        exp += f"_w{args.band_width}"
    endpoint = endpoint_from_config(model, concurrency=args.concurrency)
    cache = RunCache(pathlib.Path("runs") / exp / "cache.sqlite", exp=exp)
    executor = Executor(endpoint, cache, max_tokens_per_node=model["max_tokens_per_node"],
                        temperature=model["temperature"],
                        train_repeats=config["tensor"]["repeats"]["train"], runs_dir="runs")
    runner = None if args.no_single else SingleAgentRunner(
        endpoint, cache, load_templates(), exp=exp, max_tokens=model["max_tokens_per_node"],
        temperature=model["temperature"],
        samples=max(int(k) for k in config["single_agent"]["budget_sweep"]),
        output_dir="runs")
    tasks = probe_tasks(config, args.pool, args.n, args.seed, args.band, args.band_width)
    graphs = probe_graphs(model["name"], args.graphs)
    print(f"P2: {len(graphs)} computations x {len(tasks)} problems x {args.repeats} "
          f"repeat(s){'' if runner is None else ' + pi_single frontier'} -> exp {exp}")
    summary = run_probe(config, executor, runner, tasks, graphs, repeats=args.repeats,
                        concurrency=args.concurrency or model["max_concurrency"],
                        budget_tokens=args.budget)
    out = pathlib.Path("runs") / exp / "p2_report.json"
    out.write_text(json.dumps(summary, indent=1, default=float))

    s = summary["spread"]
    print(f"\nbest {s['best']} {summary['graphs'][s['best']]['accuracy']:.3f}")
    for key in ("second", "fifth", "worst"):
        if s[key]:
            print(f"  best - {key:6s} {s[key]['diff']:+.3f} (paired SE {s[key]['paired_se']:.3f})"
                  f"  [{s[key]['graph']}]")
    print(f"  graphs within 2 points of the best: {s['within_2_points_of_best']}")
    c = summary["capacity"]
    if "cross_fitted" in c:
        print(f"C on the probe (cross-fitted): {c['cross_fitted']:+.3f} "
              f"[{c['cross_fitted_range'][0]:+.3f}, {c['cross_fitted_range'][1]:+.3f}] "
              f"at {c['budget_tokens']:.0f} tokens")
    p = summary["power"]
    print(f"Var(d) of top pairs {p['variance']:.4f}: floor at {p['floor_at_val_tasks']['val_tasks']} "
          f"val tasks = {p['floor_at_val_tasks']['floor']:.3f}; val tasks needed "
          + ", ".join(f"eps={k}: {v:,}" for k, v in p["val_tasks_needed"].items()))
    t = summary["truncation"]
    print(f"truncation by row {t['min_row']:.1%}..{t['max_row']:.1%}")
    for name, gate in summary["gates"].items():
        print(f"  gate {name}: {gate_label(gate['pass'])}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
