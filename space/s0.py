"""S0 and S1: the enumerable audit spaces.

S0 is the ruler the whole paper is calibrated against: 4 nodes in fixed execution
order (Solver, Critic, Reviser, Finalizer) and 6 optional forward edges. The 2^6 = 64
edge subsets collapse to **34** distinct executable DAGs under the pruning rule in
:func:`space.topology.Topology.canonical`.

Two nested spaces are enumerated and both are reported:

- ``S0`` (34 graphs) allows input-less Critic/Reviser nodes, which degrade to Solver.
  (The lone Finalizer of the zero-edge graph G_empty plays Solver as well, so G_empty
  is the in-space single agent and matches pi_single's CoT prompt exactly.)
  The graphs with input-less nodes are exactly the parallel-vote topologies — the three-way vote graph is
  ``n1->n4 n2->n4 n3->n4`` — and they are the most likely members to be optimal on
  math, so excluding them would purge the space of its best candidates.
- ``S0_strict`` (16 graphs) requires every executing Critic/Reviser to have an
  in-edge, leaving only chain-like communication structures.

``S0_strict`` is a subset of ``S0``, so one score tensor covers both and
``C(S0) - C(S0_strict)`` measures the share of capacity that comes from parallel
sampling rather than from communication structure.

Closed form for k nodes in fixed order with all forward edges optional::

    f(k) = sum_{m=0}^{k-1} C(k-1, m) * h(m),   h(m) = prod_{j=1}^{m} (2^j - 1)

because after pruning every surviving node reaches the terminal, so a node with i
admissible out-edges only needs at least one of them. This gives 34 / 422 / 11586 for
k = 4 / 5 / 6 and is used as the unit-test oracle.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import pathlib
from typing import Sequence

from space.topology import (
    CRITIC,
    FINALIZER,
    INPUT_DEPENDENT_ROLES,
    REVISER,
    SOLVER,
    Node,
    Topology,
)

# --------------------------------------------------------------------------- spaces

S0_ROLES: tuple[tuple[str, str], ...] = (
    ("n1", SOLVER),
    ("n2", CRITIC),
    ("n3", REVISER),
    ("n4", FINALIZER),
)

#: 5-node extension: a second Solver is added rather than another reviewer, so the
#: space gains parallel breadth as well as depth. 10 optional forward edges, 422 DAGs.
S1_ROLES: tuple[tuple[str, str], ...] = (
    ("n1", SOLVER),
    ("n2", SOLVER),
    ("n3", CRITIC),
    ("n4", REVISER),
    ("n5", FINALIZER),
)


def declared_nodes(roles: Sequence[tuple[str, str]], model: str) -> tuple[Node, ...]:
    return tuple(
        Node(id=nid, role=role, prompt_id=f"{role}.v1", model=model)
        for nid, role in roles
    )


def forward_edges(node_ids: Sequence[str]) -> tuple[tuple[str, str], ...]:
    return tuple(itertools.combinations(node_ids, 2))


# ------------------------------------------------------------------- enumeration

def enumerate_space(
    roles: Sequence[tuple[str, str]] = S0_ROLES,
    model: str = "placeholder",
    strict_inputs: bool = False,
) -> list[Topology]:
    """All distinct executable DAGs, de-duplicated by ``graph_hash``.

    ``strict_inputs`` drops graphs in which an executing Critic or Reviser has no
    in-edge (the S0_strict / 16-graph subspace).
    """
    nodes = declared_nodes(roles, model)
    edges = forward_edges([n.id for n in nodes])
    seen: dict[str, Topology] = {}
    for mask in range(1 << len(edges)):
        subset = [e for i, e in enumerate(edges) if mask >> i & 1]
        topo = Topology.canonical(nodes, subset, space_id="")
        if strict_inputs and not _inputs_satisfied(topo):
            continue
        seen.setdefault(topo.graph_hash, topo)
    out = sorted(seen.values(), key=lambda t: (len(t.nodes), len(t.edges), str(t)))
    space_id = _space_id(roles, strict_inputs)
    return [
        Topology(
            nodes=t.nodes, edges=t.edges, space_id=space_id,
            aggregation=t.aggregation, pruned=t.pruned,
        )
        for t in out
    ]


def _inputs_satisfied(topo: Topology) -> bool:
    return all(
        topo.parents(n.id) or n.role not in INPUT_DEPENDENT_ROLES for n in topo.nodes
    )


def _space_id(roles: Sequence[tuple[str, str]], strict_inputs: bool) -> str:
    base = f"s{len(roles)}"
    return f"{base}_strict" if strict_inputs else base


# ------------------------------------------------------------------- closed form

def h(m: int) -> int:
    """Executable DAGs on m ordered nodes plus a terminal: prod (2^j - 1)."""
    out = 1
    for j in range(1, m + 1):
        out *= (1 << j) - 1
    return out


def count_closed_form(k: int) -> int:
    """f(k) = sum_m C(k-1, m) h(m). Oracle for :func:`enumerate_space`."""
    return sum(math.comb(k - 1, m) * h(m) for m in range(k))


# ------------------------------------------------------------------- reporting

def summarise(space: Sequence[Topology]) -> dict:
    by_nodes: dict[int, int] = {}
    for t in space:
        by_nodes[len(t.nodes)] = by_nodes.get(len(t.nodes), 0) + 1
    exec_groups: dict[str, list[str]] = {}
    for t in space:
        exec_groups.setdefault(t.exec_hash, []).append(str(t))
    return {
        "space_id": space[0].space_id if space else None,
        "n_graphs": len(space),
        "n_distinct_computations": len(exec_groups),
        "by_node_count": {str(k): by_nodes[k] for k in sorted(by_nodes)},
        "total_node_executions": sum(len(t.nodes) for t in space),
        # Labelled graphs that are the same computation, so share tensor rows.
        "collapsed": {k: v for k, v in exec_groups.items() if len(v) > 1},
        "graphs": [
            {
                "graph_hash": t.graph_hash,
                "exec_hash": t.exec_hash,
                "short": t.short_hash,
                "edges": [list(e) for e in t.edges],
                "nodes": list(t.node_ids),
                "effective_roles": t.effective_roles(),
                "pruned": list(t.pruned),
                "label": str(t),
            }
            for t in space
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--enumerate", action="store_true")
    ap.add_argument("--nodes", type=int, default=4, choices=(4, 5, 6))
    ap.add_argument("--out", default="runs/e0/space_s0.json")
    args = ap.parse_args()

    roles = {4: S0_ROLES, 5: S1_ROLES}.get(args.nodes)
    if roles is None:
        print(f"k={args.nodes}: closed form only -> {count_closed_form(args.nodes)} graphs")
        return

    full = enumerate_space(roles)
    strict = enumerate_space(roles, strict_inputs=True)
    expect = count_closed_form(args.nodes)
    sf, ss = summarise(full), summarise(strict)
    print(f"k={args.nodes}  edge_masks={1 << len(forward_edges([n for n, _ in roles]))}"
          f"  labelled={len(full)}  closed_form={expect}  "
          f"{'OK' if len(full) == expect else 'MISMATCH'}")
    print(f"  distinct computations: {sf['n_distinct_computations']}  "
          f"({len(full) - sf['n_distinct_computations']} labelled graphs collapse)")
    print(f"  by node count: {sf['by_node_count']}")
    print(f"  strict (Critic/Reviser need inputs): labelled={len(strict)}  "
          f"distinct={ss['n_distinct_computations']}  by node count: {ss['by_node_count']}")
    hashes = {t.graph_hash for t in full}
    assert {t.graph_hash for t in strict} <= hashes, "strict space must nest inside full"
    print("  nesting: S_strict subset of S_full OK")

    if args.enumerate:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"full": summarise(full), "strict": summarise(strict),
                   "closed_form": expect}
        out.write_text(json.dumps(payload, indent=1))
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
