"""The only script that reads the test split. Produces every reported number.

Discipline, restated because this is where it would be easiest to break:

- Selection happens on `audit`. The test split is *evaluated* for all 25 computations
  -- it has to be, since the tensor is exhaustive -- but no argmax, ranking or
  threshold is ever applied to it here. The hazard is not reading the split; it is
  quoting `max_g P_test(g)` as the ceiling, so that quantity is computed once,
  labelled `diagnostic_naive_test_max`, and excluded from every term.
- Trajectories record the incumbent as a *labelled* graph_hash, because the optimizer
  chose a specific edge mask. Scores are keyed by `exec_hash`. The mapping between
  them is rebuilt from the enumerated space rather than inferred, since several masks
  share one computation and getting this backwards would silently mix rows.
- Terms use one estimator of P(G_star) throughout. The cross-fitted value is reported
  beside C as the winner's-curse diagnostic, never inside it.

Usage:
    python3 analysis/report.py --config configs/e0.yaml
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from analysis.bootstrap import (
    Interval,
    paired_task_bootstrap,
    paired_task_seed_bootstrap,
    repeat_variance,
)
from analysis.decompose import (
    capacity_upper_bound,
    cross_fit_capacity,
    decompose_trajectory,
    ladder_levels,
    select_g_hat_n,
    select_g_star_plugin,
)
from analysis.matrices import build_matrix, coverage, repeat_cube
from analysis.plateau import observed_disagreement, power_report, scan_plateau
from runtime.cache import RunCache
from space.s0 import S0_ROLES, enumerate_space

SPLITS = ("train", "audit", "val", "test")


def hash_maps(model: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    """graph_hash -> exec_hash, and the reverse grouping."""
    forward, reverse = {}, {}
    for topology in enumerate_space(S0_ROLES, model=model):
        forward[topology.graph_hash] = topology.exec_hash
        reverse.setdefault(topology.exec_hash, []).append(topology.graph_hash)
    return forward, reverse


def read_incumbents(runs_dir: pathlib.Path, exp: str, optimizer: str,
                    graph_to_exec: dict[str, str]) -> dict[int, dict[int, str]]:
    """{seed: {cum_evals: exec_hash}} from the trajectory files."""
    out: dict[int, dict[int, str]] = {}
    for path in sorted((runs_dir / exp / optimizer).glob("seed*/trajectory.jsonl")):
        seed = int(path.parent.name.removeprefix("seed"))
        per_step: dict[int, str] = {}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            graph_hash = row["incumbent"]
            if graph_hash not in graph_to_exec:
                raise SystemExit(f"{path}: incumbent {graph_hash[:12]} is not in S0")
            per_step[int(row["cum_evals"])] = graph_to_exec[graph_hash]
        if per_step:
            out[seed] = per_step
    return out


def term_intervals(rows, matrices, p_single: float, step: int,
                   iterations: int) -> dict[str, Interval]:
    """Paired intervals for the four terms at one checkpoint.

    Tasks are resampled jointly for every quantity in a row, and seeds too wherever
    the learned graph enters, because a seed is a draw of where the learner stopped.
    """
    at_step = [row for row in rows if row.step == step]
    if not at_step:
        return {}
    test, train = matrices["test"], matrices["train"]
    g_star, g_hat = at_step[0].g_star, at_step[0].g_hat_n
    learned = [row.g_learned for row in at_step]
    n_test, n_train = len(test.task_ids), len(train.task_ids)

    def mean(matrix, graph, idx):
        return float(np.nanmean(matrix.row(graph)[np.asarray(list(idx), int)]))

    out: dict[str, Interval] = {}
    out["C"] = paired_task_bootstrap(
        lambda idx: mean(test, g_star, idx) - p_single, n_test, iterations=iterations)
    out["E"] = paired_task_bootstrap(
        lambda idx: mean(test, g_star, idx) - mean(test, g_hat, idx), n_test,
        iterations=iterations)
    out["O"] = paired_task_seed_bootstrap(
        lambda t, s: float(np.mean([
            mean(train, g_hat, t) - mean(train, learned[i], t) for i in s])),
        n_train, len(learned), iterations=iterations)
    # O + Delta_gen is the population gap the "keep searching" decision reads.
    out["search_value"] = paired_task_seed_bootstrap(
        lambda t, s: float(np.mean([
            mean(test, g_hat, t) - mean(test, learned[i], t) for i in s])),
        n_test, len(learned), iterations=iterations)
    out["achieved_gain"] = paired_task_seed_bootstrap(
        lambda t, s: float(np.mean([mean(test, learned[i], t) for i in s])) - p_single,
        n_test, len(learned), iterations=iterations)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/e0.yaml")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--p-single", type=float, default=None,
                    help="override; normally read from the pi_single frontier")
    ap.add_argument("--iterations", type=int, default=None)
    args = ap.parse_args()

    config = yaml.safe_load(pathlib.Path(args.config).read_text())
    exp = config["exp"]
    runs_dir = pathlib.Path(args.runs_dir)
    iterations = args.iterations or config["bootstrap"]["iterations"]

    cache = RunCache(f"{args.runs_dir}/{exp}/cache.sqlite", exp=exp)
    records = list(cache.export(exp))
    if not records:
        raise SystemExit(f"cache for exp={exp} is empty; run audit/run_audit.py first")

    graph_to_exec, exec_to_graphs = hash_maps(config["model"]["name"])
    rows_order = tuple(sorted(exec_to_graphs))
    matrices = {role: build_matrix(records, role, exec_hashes=rows_order)
                for role in SPLITS}
    cover = {role: coverage(matrices[role]) for role in SPLITS}

    g_star = select_g_star_plugin(matrices["audit"])
    g_hat_n = select_g_hat_n(matrices["train"])
    cross = cross_fit_capacity(matrices["audit"], n_splits=20)

    p_single = args.p_single
    if p_single is None:
        frontier_path = runs_dir / exp / "pi_single" / "frontier.json"
        if not frontier_path.exists():
            raise SystemExit("no pi_single frontier; run runtime/single_agent.py "
                             "or pass --p-single")
        p_single = json.loads(frontier_path.read_text())["headline"]["accuracy"]

    report: dict = {
        "exp": exp,
        "space": {"distinct_computations": len(rows_order),
                  "labelled_graphs": len(graph_to_exec),
                  "collapsed": {k: v for k, v in exec_to_graphs.items() if len(v) > 1}},
        "coverage": cover,
        "selection": {
            "g_star_plugin": g_star,
            "g_hat_n": g_hat_n,
            "p_single": p_single,
            "cross_fitted_capacity_estimate": cross.cross_fitted,
            "naive_audit_max": cross.naive_max,
            "selection_bias": cross.selection_bias,
            "capacity_upper_bound": capacity_upper_bound(matrices["audit"]),
        },
        # Computed once, named as a diagnostic, and used by nothing. Quoting this as
        # the ceiling is the failure mode the hard rule guards against.
        "diagnostic_naive_test_max": float(np.nanmax(matrices["test"].means())),
        "optimizers": {},
        "plateau": {},
        "power": {},
    }

    _, _, cube = repeat_cube(records, "train", config["tensor"]["repeats"]["train"])
    report["recycled_sampling_noise"] = {
        "train_within_cell_variance": repeat_variance(cube),
        "note": "seeds share tensor cells once R draws are used, so seed-level "
                "intervals understate a rerun with fresh sampling",
    }

    val_rows = {h: matrices["val"].row(h) for h in rows_order}
    pairs = [(rows_order[i], rows_order[j])
             for i in range(len(rows_order)) for j in range(i + 1, len(rows_order))]
    disagreement = observed_disagreement(val_rows, pairs[:200])
    report["power"] = power_report(
        epsilon=config["plateau"]["epsilon"] / 100.0,
        disagreement=disagreement,
        n_tasks=len(matrices["val"].task_ids),
        n_seeds=len(config["seeds"])).as_dict()

    for optimizer in config["optimizers"]:
        incumbents = read_incumbents(runs_dir, exp, optimizer, graph_to_exec)
        if not incumbents:
            continue
        rows = decompose_trajectory(optimizer, incumbents, train=matrices["train"],
                                    audit=matrices["audit"], test=matrices["test"],
                                    p_single=p_single)
        steps = sorted({row.step for row in rows})
        report["optimizers"][optimizer] = {
            "seeds": sorted(incumbents),
            "rows": [row.as_dict() for row in rows],
            "ladder": ladder_levels(rows),
            "max_abs_residual": max(abs(row.residual()) for row in rows),
            "intervals": {
                str(step): {name: iv.as_dict()
                            for name, iv in term_intervals(
                                rows, matrices, p_single, step, iterations).items()}
                for step in config["budget"]["report_at"] if step in steps
            },
        }
        report["plateau"][optimizer] = [
            verdict.as_dict() for verdict in scan_plateau(
                optimizer, incumbents, val_rows, steps=steps,
                delta_t=config["plateau"]["delta_t"],
                epsilon=config["plateau"]["epsilon"] / 100.0,
                iterations=min(iterations, 500))
        ]

    # E0 gate: can the terms tell at least two causes apart?
    gate = {}
    for optimizer, block in report["optimizers"].items():
        for step, ivs in block["intervals"].items():
            named = {k: Interval(**{kk: vv for kk, vv in v.items() if kk != "width"})
                     for k, v in ivs.items()}
            keys = ["C", "E", "O", "search_value"]
            disjoint = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]
                        if a in named and b in named and named[a].disjoint_from(named[b])]
            gate[f"{optimizer}@{step}"] = {
                "disjoint_pairs": [list(p) for p in disjoint],
                "two_causes_separable": len(disjoint) >= 1,
            }
    report["e0_gate"] = {
        "identity_closes": all(b["max_abs_residual"] < 1e-9
                               for b in report["optimizers"].values()),
        "selection_bias_quantified": cross.selection_bias is not None,
        "term_separability": gate,
        "any_cell_separable": any(v["two_causes_separable"] for v in gate.values()),
    }

    out = runs_dir / exp / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=float))
    print(f"wrote {out}")
    print(f"  distinct computations {len(rows_order)}, "
          f"coverage test {cover['test']['filled']:.1%}")
    print(f"  P(pi_single) {p_single:.4f}  selection bias "
          f"{cross.selection_bias:+.4f}")
    print(f"  identity closes: {report['e0_gate']['identity_closes']}")
    print(f"  two causes separable somewhere: "
          f"{report['e0_gate']['any_cell_separable']}")
    print(f"  plateau power sufficient: {report['power']['sufficient']}")


if __name__ == "__main__":
    main()
