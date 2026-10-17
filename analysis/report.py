"""The only script that reads the test split. Produces every reported number.

Discipline, restated because this is where it would be easiest to break:

- Selection happens on `audit`. The test split is *evaluated* for all 25 computations
  -- it has to be, since the tensor is exhaustive -- but no argmax, ranking or
  threshold is ever applied to it here. The hazard is not reading the split; it is
  quoting `max_g P_test(g)` as the ceiling, so that quantity is computed once,
  labelled `diagnostic_naive_test_max`, and excluded from every term.
- pi_single is selected on `val` by runtime/single_agent.py and only the selected
  strategy is scored on test. Its per-task test scores are read back so that C and
  the achieved gain are *paired* differences, like every other term.
- Trajectories record the incumbent as a *labelled* graph_hash, because the optimizer
  chose a specific edge mask. Scores are keyed by `exec_hash`. The mapping between
  them is rebuilt from the enumerated space rather than inferred, since several masks
  share one computation and getting this backwards would silently mix rows.
- Every optimizer is put on one budget axis before anything is compared (see
  analysis/trajectory.py). Exact lookups at budget t dropped reinforce_edge from the
  plateau test entirely and computed greedy_local's early intervals on the few seeds
  whose rounds happened to land on t.
- Terms use one estimator of P(G_star) throughout. The cross-fitted value is reported
  beside C as the winner's-curse diagnostic, never inside it.

Usage:
    python3 analysis/report.py --config configs/e0.yaml
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from analysis.bootstrap import (  # noqa: E402
    Interval,
    paired_task_bootstrap,
    paired_task_seed_bootstrap,
    repeat_variance,
)
from analysis.decompose import (  # noqa: E402
    ScoreMatrix,
    capacity_upper_bound,
    cross_fit_capacity,
    decompose_trajectory,
    ladder_levels,
    prescribe,
    select_g_hat_n,
    select_g_star_plugin,
)
from analysis.matrices import (  # noqa: E402
    build_matrix,
    coverage,
    repeat_cube,
    truncation_summary,
)
from analysis.plateau import (  # noqa: E402
    first_plateau,
    observed_disagreement,
    observed_paired_variance,
    power_report,
    scan_plateau,
    testable_steps,
)
from analysis.trajectory import align, read_optimizer  # noqa: E402
from runtime.cache import RunCache  # noqa: E402
from runtime.executor import RunRecord  # noqa: E402
from space.s0 import S0_ROLES, enumerate_space  # noqa: E402

SPLITS = ("train", "audit", "val", "test")
#: Pairs for the power calculation come from the best few graphs *on audit*: the
#: plateau test compares near-optimal incumbents, and audit is where selection is
#: allowed.
POWER_TOP_K = 5


def hash_maps(model: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    """graph_hash -> exec_hash, and the reverse grouping."""
    forward, reverse = {}, {}
    for topology in enumerate_space(S0_ROLES, model=model):
        forward[topology.graph_hash] = topology.exec_hash
        reverse.setdefault(topology.exec_hash, []).append(topology.graph_hash)
    return forward, reverse


# ------------------------------------------------------------------- pi_single

@dataclass(frozen=True)
class PiSingle:
    """P(pi_single) as a scalar and, when available, as per-task test scores."""

    accuracy: float
    vector: np.ndarray | None
    source: str
    selection: dict = field(default_factory=dict)

    @property
    def paired(self) -> bool:
        return self.vector is not None

    def mean(self, idx: Sequence[int] | np.ndarray) -> float:
        if self.vector is None:
            return self.accuracy
        return float(np.mean(self.vector[np.asarray(idx, dtype=int)]))

    def as_dict(self) -> dict:
        return {"accuracy": self.accuracy, "paired": self.paired, "source": self.source,
                **({"selection": self.selection} if self.selection else {})}


def load_pi_single(frontier: dict | None, test_task_ids: Sequence[str],
                   override: float | None = None) -> PiSingle:
    """Read runtime/single_agent.py's frontier.json, aligned to the test matrix.

    Pairing needs a score for every test task in the matrix; a partial evaluation
    (a smoke run with --limit-tasks) falls back to the unpaired scalar and says so.
    """
    if override is not None:
        return PiSingle(float(override), None, "override")
    if frontier is None:
        raise SystemExit("no pi_single frontier; run runtime/single_agent.py "
                         "or pass --p-single")
    headline = frontier["headline"]
    selection = {k: headline.get(k) for k in
                 ("strategy", "param", "budget_tokens", "val_accuracy", "selected_on")}
    scores = headline.get("task_scores")
    if scores:
        vector = np.array([scores.get(task_id, np.nan) for task_id in test_task_ids],
                          dtype=float)
        if not np.isnan(vector).any():
            return PiSingle(float(vector.mean()), vector, "frontier", selection)
        selection["unpaired_reason"] = (
            f"pi_single scored {int((~np.isnan(vector)).sum())} of "
            f"{len(test_task_ids)} test tasks")
    return PiSingle(float(headline["accuracy"]), None, "frontier", selection)


# ---------------------------------------------------------------- diagnostics

def power_section(matrices: dict[str, ScoreMatrix], rows_order: Sequence[str],
                  epsilon: float, n_seeds: int, top_k: int = POWER_TOP_K) -> dict:
    """Can the validation split exclude epsilon, measured on near-optimal pairs?

    Pairs are drawn among the top-k graphs by *audit* mean, because the plateau test
    compares incumbents near the top of the space; the variance of their paired
    differences is measured on the *val* rows the test actually uses.
    """
    audit_means = matrices["audit"].means()
    ranked = [rows_order[i] for i in np.argsort(-np.nan_to_num(audit_means, nan=-1.0))]
    top = ranked[:top_k]
    val_rows = {h: matrices["val"].row(h) for h in rows_order}
    top_pairs = list(itertools.combinations(top, 2))
    all_pairs = list(itertools.combinations(rows_order, 2))
    variance = observed_paired_variance(val_rows, top_pairs)
    disagreement = observed_disagreement(val_rows, top_pairs)
    n_tasks = len(matrices["val"].task_ids)
    out = {
        "pairs": f"top {len(top)} graphs by audit mean",
        "measured_variance_top_pairs": variance,
        "measured_variance_all_pairs": observed_paired_variance(val_rows, all_pairs),
        "disagreement_top_pairs": disagreement,
    }
    if np.isfinite(variance) and variance > 0:
        out.update(power_report(epsilon=epsilon, n_tasks=n_tasks, n_seeds=n_seeds,
                                variance=variance).as_dict())
        out["disagreement"] = disagreement
    else:
        out.update({"sufficient": None, "note": "no finite paired variance on val yet"})
    return out


# -------------------------------------------------------------------- intervals

def term_intervals(rows_at_step, matrices: dict[str, ScoreMatrix], single: PiSingle,
                   iterations: int) -> dict[str, Interval]:
    """Paired intervals for the terms at one budget.

    Tasks are resampled jointly for every quantity in a row, and seeds too wherever
    the learned graph enters, because a seed is a draw of where the learner stopped.
    pi_single enters with its per-task test scores when they exist, so C and the
    achieved gain are paired like the rest.
    """
    if not rows_at_step:
        return {}
    test, train = matrices["test"], matrices["train"]
    g_star, g_hat = rows_at_step[0].g_star, rows_at_step[0].g_hat_n
    learned = [row.g_learned for row in rows_at_step]
    n_test, n_train = len(test.task_ids), len(train.task_ids)

    rows_of = {(id(m), g): m.row(g) for m in (test, train)
               for g in {g_star, g_hat, *learned}}

    def mean(matrix: ScoreMatrix, graph: str, idx) -> float:
        return float(np.nanmean(rows_of[(id(matrix), graph)][np.asarray(idx, dtype=int)]))

    def learned_mean(matrix: ScoreMatrix, t, s) -> float:
        # Seeds often converge onto the same few graphs: score each graph once.
        t = np.asarray(t, dtype=int)
        per_graph: dict[str, float] = {}
        for i in s:
            graph = learned[i]
            if graph not in per_graph:
                per_graph[graph] = mean(matrix, graph, t)
        return float(np.mean([per_graph[learned[i]] for i in s]))

    out: dict[str, Interval] = {}
    out["C"] = paired_task_bootstrap(
        lambda idx: mean(test, g_star, idx) - single.mean(idx), n_test,
        iterations=iterations)
    out["E"] = paired_task_bootstrap(
        lambda idx: mean(test, g_star, idx) - mean(test, g_hat, idx), n_test,
        iterations=iterations)
    out["O"] = paired_task_seed_bootstrap(
        lambda t, s: mean(train, g_hat, t) - learned_mean(train, t, s),
        n_train, len(learned), iterations=iterations)
    # O + Delta_gen is the population gap the "keep searching" decision reads.
    out["search_value"] = paired_task_seed_bootstrap(
        lambda t, s: mean(test, g_hat, t) - learned_mean(test, t, s),
        n_test, len(learned), iterations=iterations)
    # E + search_value = P(G_star) - P(G_learned), resampled jointly because E and
    # search_value share the G_hat_N column and are correlated.
    out["gap_to_optimum"] = paired_task_seed_bootstrap(
        lambda t, s: mean(test, g_star, t) - learned_mean(test, t, s),
        n_test, len(learned), iterations=iterations)
    out["achieved_gain"] = paired_task_seed_bootstrap(
        lambda t, s: learned_mean(test, t, s) - single.mean(t),
        n_test, len(learned), iterations=iterations)
    return out


# ---------------------------------------------------------------------- report

def build_report(config: dict, records: list[RunRecord], runs_dir: pathlib.Path, *,
                 frontier: dict | None, p_single_override: float | None = None,
                 iterations: int | None = None) -> tuple[dict, list[dict]]:
    """Everything except file IO, so it can be tested on a synthetic tensor."""
    exp = config["exp"]
    iterations = iterations or config["bootstrap"]["iterations"]
    epsilon = config["plateau"]["epsilon"] / 100.0
    delta_t = config["plateau"]["delta_t"]
    max_evals = config["budget"]["max_evals"]

    graph_to_exec, exec_to_graphs = hash_maps(config["model"]["name"])
    rows_order = tuple(sorted(exec_to_graphs))
    matrices = {role: build_matrix(records, role, exec_hashes=rows_order)
                for role in SPLITS}
    cover = {role: coverage(matrices[role]) for role in SPLITS}

    g_star = select_g_star_plugin(matrices["audit"])
    g_hat_n = select_g_hat_n(matrices["train"])
    cross = cross_fit_capacity(matrices["audit"], n_splits=20)
    single = load_pi_single(frontier, matrices["test"].task_ids, p_single_override)

    report: dict = {
        "exp": exp,
        "space": {"distinct_computations": len(rows_order),
                  "labelled_graphs": len(graph_to_exec),
                  "collapsed": {k: v for k, v in exec_to_graphs.items() if len(v) > 1}},
        "coverage": cover,
        "pi_single": single.as_dict(),
        "selection": {
            "g_star_plugin": g_star,
            "g_hat_n": g_hat_n,
            "p_single": single.accuracy,
            "cross_fitted_capacity_estimate": cross.cross_fitted,
            "naive_audit_max": cross.naive_max,
            "selection_bias": cross.selection_bias,
            "capacity_upper_bound": capacity_upper_bound(matrices["audit"]),
        },
        # Computed once, named as a diagnostic, and used by nothing. Quoting this as
        # the ceiling is the failure mode the hard rule guards against.
        "diagnostic_naive_test_max": float(np.nanmax(matrices["test"].means())),
        "truncation": truncation_summary(
            records, rows_order, config.get("acceptance", {}).get("max_row_truncation_spread")),
        "optimizers": {},
        "plateau": {},
    }

    _, _, cube = repeat_cube(records, "train", config["tensor"]["repeats"]["train"])
    report["recycled_sampling_noise"] = {
        "train_within_cell_variance": repeat_variance(cube),
        "note": "seeds share tensor cells once R draws are used, so seed-level "
                "intervals understate a rerun with fresh sampling",
    }
    report["power"] = power_section(matrices, rows_order, epsilon, len(config["seeds"]))

    val_rows = {h: matrices["val"].row(h) for h in rows_order}
    all_rows: list[dict] = []
    for optimizer in config["optimizers"]:
        per_round = read_optimizer(runs_dir, exp, optimizer, graph_to_exec)
        if not per_round:
            report["optimizers"][optimizer] = {"status": "no trajectories"}
            continue
        incumbents, axis = align(per_round, max_evals)
        if not incumbents:
            report["optimizers"][optimizer] = {"status": "no complete seeds",
                                               "coverage": axis}
            continue
        rows = decompose_trajectory(optimizer, incumbents, train=matrices["train"],
                                    audit=matrices["audit"], test=matrices["test"],
                                    p_single=single.accuracy)
        all_rows += [row.as_dict() for row in rows]
        budgets = set(axis["budgets"])
        intervals: dict[str, dict] = {}
        decisions: dict[str, dict] = {}
        for budget in config["budget"]["report_at"]:
            if budget not in budgets:
                intervals[str(budget)] = {
                    "status": "not_covered",
                    "first_common_budget": axis["first_common_budget"]}
                continue
            ivs = term_intervals([row for row in rows if row.step == budget], matrices,
                                 single, iterations)
            intervals[str(budget)] = {name: iv.as_dict() for name, iv in ivs.items()}
            decisions[str(budget)] = prescribe(ivs["C"], ivs["E"], ivs["search_value"],
                                               ivs["gap_to_optimum"], epsilon)
        report["optimizers"][optimizer] = {
            "coverage": {k: v for k, v in axis.items() if k != "budgets"},
            "rows": len(rows),
            "ladder": ladder_levels(rows),
            "max_abs_residual": max(abs(row.residual()) for row in rows),
            "intervals": intervals,
            "decision": decisions,
        }
        verdicts = scan_plateau(optimizer, incumbents, val_rows,
                                steps=testable_steps(axis["budgets"], delta_t),
                                delta_t=delta_t, epsilon=epsilon,
                                iterations=min(iterations, 500))
        first = first_plateau(verdicts)
        report["plateau"][optimizer] = {
            "tested_budgets": len(verdicts),
            "first_declared": first.as_dict() if first else None,
            "verdicts": [verdict.as_dict() for verdict in verdicts],
        }

    gate = {}
    for optimizer, block in report["optimizers"].items():
        for budget, ivs in block.get("intervals", {}).items():
            if ivs.get("status") == "not_covered":
                gate[f"{optimizer}@{budget}"] = {"status": "not_covered"}
                continue
            named = {k: Interval(**{kk: vv for kk, vv in v.items() if kk != "width"})
                     for k, v in ivs.items()}
            keys = ["C", "E", "O", "search_value"]
            disjoint = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]
                        if a in named and b in named and named[a].disjoint_from(named[b])]
            gate[f"{optimizer}@{budget}"] = {
                "disjoint_pairs": [list(p) for p in disjoint],
                "two_causes_separable": len(disjoint) >= 1,
            }
    report["e0_gate"] = {
        "identity_closes": all(b["max_abs_residual"] < 1e-9
                               for b in report["optimizers"].values()
                               if "max_abs_residual" in b),
        "selection_bias_quantified": cross.selection_bias is not None,
        "pi_single_paired": single.paired,
        "term_separability": gate,
        "any_cell_separable": any(v.get("two_causes_separable") for v in gate.values()),
    }
    return report, all_rows


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

    cache = RunCache(runs_dir / exp / "cache.sqlite", exp=exp)
    records = list(cache.export(exp))
    if not records:
        raise SystemExit(f"cache for exp={exp} is empty; run audit/run_audit.py first")
    frontier_path = runs_dir / exp / "pi_single" / "frontier.json"
    frontier = (json.loads(frontier_path.read_text())
                if frontier_path.exists() else None)

    report, rows = build_report(config, records, runs_dir, frontier=frontier,
                                p_single_override=args.p_single,
                                iterations=args.iterations)
    out = runs_dir / exp / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=float))
    rows_path = runs_dir / exp / "decomposition_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, default=float) + "\n")

    cover = report["coverage"]
    print(f"wrote {out} and {rows_path} ({len(rows)} rows)")
    print(f"  distinct computations {report['space']['distinct_computations']}, "
          f"coverage test {cover['test']['filled']:.1%}")
    print(f"  P(pi_single) {report['pi_single']['accuracy']:.4f} "
          f"({'paired' if report['pi_single']['paired'] else 'UNPAIRED'})  "
          f"selection bias {report['selection']['selection_bias']:+.4f}")
    for optimizer, block in report["optimizers"].items():
        axis = block.get("coverage", {})
        print(f"  {optimizer}: {block.get('status', 'ok')}, seeds "
              f"{len(axis.get('seeds', []))}, budgets from {axis.get('first_common_budget')}"
              + (f", excluded {axis['excluded_incomplete_seeds']}"
                 if axis.get("excluded_incomplete_seeds") else ""))
    for split, block in report["truncation"].items():
        print(f"  truncation {split}: overall {block['overall']:.1%}, "
              f"row spread {block['spread']:.1%}")
    print(f"  identity closes: {report['e0_gate']['identity_closes']}")
    print(f"  two causes separable somewhere: {report['e0_gate']['any_cell_separable']}")
    print(f"  plateau power sufficient: {report['power'].get('sufficient')}")


if __name__ == "__main__":
    main()
