"""Topology: the identity of a MAS graph.

A topology is a set of role-labelled nodes executed in a fixed order, plus forward
communication edges. Two things matter for the audit and for cache replay:

1. **Canonicalisation.** Nodes whose output cannot reach the terminal node never run,
   so they are pruned before the graph gets an identity. 64 edge subsets over 4 nodes
   collapse to 34 distinct executable DAGs.
2. **Hashing.** `graph_hash` is a SHA-256 over the canonical form and *includes* the
   aggregation rule and each node's effective role, model and tools. Anything that
   changes what the endpoint is asked must change the hash, or cached scores from a
   different configuration would be silently reused.

Graph identity is the hash. Never compare topologies by edge list.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Iterable, Iterator, Sequence


@lru_cache(maxsize=8)
def _permutations(n: int) -> tuple[tuple[int, ...], ...]:
    return tuple(itertools.permutations(range(n)))

# --------------------------------------------------------------------------- roles

SOLVER = "solver"
CRITIC = "critic"
REVISER = "reviser"
FINALIZER = "finalizer"

#: Roles that read peer output to do their job. With no in-edges they have nothing to
#: work on, so they degrade to SOLVER (decision 2026-10-13: allow input-less nodes and
#: record the degradation, rather than forbidding the graph).
INPUT_DEPENDENT_ROLES = frozenset({CRITIC, REVISER})

#: How a node combines multiple in-edges. Frozen for E0; part of the hash because it
#: changes the score of every multi-in-edge graph (notably the parallel-vote graphs).
AGGREGATION_CONCAT_RERASON = "concat_rereason"
AGGREGATIONS = frozenset({AGGREGATION_CONCAT_RERASON})


@dataclass(frozen=True, slots=True)
class Node:
    """One agent: a role prompt plus a single LLM call. No tools, no memory in E0."""

    id: str
    role: str
    prompt_id: str
    model: str
    tools: tuple[str, ...] = ()

    def effective_role(self, has_inputs: bool) -> str:
        """Role actually played once the graph is known.

        A Critic or Reviser with no in-edges receives nothing, so it independently
        solves the task. The distinction is logged and hashed: a 'Critic' that never
        read a solution is a second Solver, and the parallel-vote graphs in S0 are
        built entirely out of such nodes.
        """
        if not has_inputs and self.role in INPUT_DEPENDENT_ROLES:
            return SOLVER
        return self.role

    def effective_prompt_id(self, has_inputs: bool) -> str:
        """Prompt the executor will actually send.

        Follows the effective role: a degraded Critic gets the Solver template, not a
        Critic template pointed at an empty input.
        """
        role = self.effective_role(has_inputs)
        return self.prompt_id if role == self.role else f"{role}.v1"


@dataclass(frozen=True, slots=True)
class Topology:
    """An executable MAS graph, canonical by construction (see :meth:`canonical`)."""

    nodes: tuple[Node, ...]
    edges: tuple[tuple[str, str], ...]
    space_id: str
    aggregation: str = AGGREGATION_CONCAT_RERASON
    #: Nodes of the declared space that this graph drops, in declared order. Kept for
    #: reporting only; it does not affect the hash (the pruned graph is the identity).
    pruned: tuple[str, ...] = ()

    # -- structure ---------------------------------------------------------------

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(n.id for n in self.nodes)

    @property
    def terminal(self) -> str:
        """Last node in execution order: the one whose output is the answer."""
        return self.nodes[-1].id

    def parents(self, node_id: str) -> tuple[str, ...]:
        """In-neighbours, in execution order (fixes prompt concatenation order)."""
        order = {nid: i for i, nid in enumerate(self.node_ids)}
        src = [s for s, d in self.edges if d == node_id]
        return tuple(sorted(src, key=order.__getitem__))

    def children(self, node_id: str) -> tuple[str, ...]:
        order = {nid: i for i, nid in enumerate(self.node_ids)}
        dst = [d for s, d in self.edges if s == node_id]
        return tuple(sorted(dst, key=order.__getitem__))

    def effective_roles(self) -> dict[str, str]:
        return {n.id: n.effective_role(bool(self.parents(n.id))) for n in self.nodes}

    # -- identity ----------------------------------------------------------------

    def canonical_form(self) -> dict:
        """Labelled identity: node ids retained.

        ``space_id`` is deliberately absent — the space a graph was drawn from is
        provenance, not identity. Keeping it out is what lets a single score tensor
        serve S0 and the nested S0_strict, and S0 graphs reachable from any later space.
        """
        return {
            "aggregation": self.aggregation,
            "nodes": [
                {
                    "id": n.id,
                    "role": n.effective_role(bool(self.parents(n.id))),
                    "prompt_id": n.effective_prompt_id(bool(self.parents(n.id))),
                    "model": n.model,
                    "tools": sorted(n.tools),
                }
                for n in self.nodes
            ],
            "edges": sorted(self.edges),
        }

    def exec_canonical_form(self) -> dict:
        """Computational identity: nodes relabelled to their execution-order index.

        Role degradation makes some labelled graphs identical computations. With
        ``n2`` (Critic) input-less it becomes a Solver, so ``n1->n4`` and ``n2->n4``
        issue byte-identical prompts. In S0 the 34 labelled DAGs are only 25 distinct
        computations; in S1 the 422 are 298.

        The cache and the score tensor key on this, so equivalent masks share samples
        instead of paying twice. Repeat indices must also be counted per ``exec_hash``,
        or two equivalent masks would both draw repeat 0 and an optimizer would read
        one sample as two independent confirmations.
        """
        n = len(self.nodes)
        pos = {nid: i for i, nid in enumerate(self.node_ids)}
        roles = [
            self.nodes[i].effective_role(bool(self.parents(self.node_ids[i])))
            for i in range(n)
        ]
        prompts = [
            self.nodes[i].effective_prompt_id(bool(self.parents(self.node_ids[i])))
            for i in range(n)
        ]
        models = [node.model for node in self.nodes]
        tools = [tuple(sorted(node.tools)) for node in self.nodes]
        slots = [(roles[i], prompts[i], models[i], tools[i]) for i in range(n)]
        edges = [(pos[s], pos[d]) for s, d in self.edges]

        best: dict | None = None
        best_key: str | None = None
        for perm in _permutations(n):
            # perm[i] is the new index of old index i; only swap interchangeable slots
            if any(slots[i] != slots[perm.index(i)] for i in range(n)):
                continue
            moved = sorted((perm[s], perm[d]) for s, d in edges)
            if any(s >= d for s, d in moved):  # must stay forward in execution order
                continue
            form = {
                "aggregation": self.aggregation,
                "slots": [list(slots[perm.index(i)]) for i in range(n)],
                "edges": [list(e) for e in moved],
            }
            key = json.dumps(form, sort_keys=True, separators=(",", ":"))
            if best_key is None or key < best_key:
                best, best_key = form, key
        assert best is not None  # identity permutation always qualifies
        return best

    @property
    def graph_hash(self) -> str:
        """Identity of the labelled graph: what the optimizer picks."""
        blob = json.dumps(self.canonical_form(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    @property
    def exec_hash(self) -> str:
        """Identity of the computation: what the cache and the tensor key on."""
        blob = json.dumps(
            self.exec_canonical_form(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    @property
    def short_hash(self) -> str:
        return self.graph_hash[:12]

    def __str__(self) -> str:
        if not self.edges:
            return f"[{self.terminal}]"
        return " ".join(f"{s}->{d}" for s, d in sorted(self.edges))

    # -- construction ------------------------------------------------------------

    @classmethod
    def canonical(
        cls,
        nodes: Sequence[Node],
        edges: Iterable[tuple[str, str]],
        space_id: str,
        aggregation: str = AGGREGATION_CONCAT_RERASON,
    ) -> "Topology":
        """Build the canonical (pruned) topology for a declared node set and edge set.

        Pruning rule: the terminal node always runs; every other node runs only if its
        output reaches the terminal. Edges touching a pruned node are dropped. The
        result is the graph that will actually be executed, and the thing we hash.
        """
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"unknown aggregation rule: {aggregation!r}")
        by_id = {n.id: n for n in nodes}
        if len(by_id) != len(nodes):
            raise ValueError("duplicate node ids")
        order = {n.id: i for i, n in enumerate(nodes)}

        edge_set = set()
        for src, dst in edges:
            if src not in by_id or dst not in by_id:
                raise ValueError(f"edge {(src, dst)} references an unknown node")
            if order[src] >= order[dst]:
                raise ValueError(f"edge {(src, dst)} is not forward in execution order")
            edge_set.add((src, dst))

        terminal = nodes[-1].id
        reaches = {terminal}
        # Walk backwards in execution order: a node reaches the terminal iff it has an
        # out-edge to some node that does. Forward-only edges make one pass enough.
        for node in reversed(nodes[:-1]):
            if any(dst in reaches for src, dst in edge_set if src == node.id):
                reaches.add(node.id)

        kept = tuple(n for n in nodes if n.id in reaches)
        pruned = tuple(n.id for n in nodes if n.id not in reaches)
        kept_edges = tuple(
            sorted((s, d) for s, d in edge_set if s in reaches and d in reaches)
        )
        return cls(
            nodes=kept, edges=kept_edges, space_id=space_id,
            aggregation=aggregation, pruned=pruned,
        )

    def with_models(self, assignment: dict[str, str]) -> "Topology":
        """Reassign per-node models (S_hetero). Changes the hash, as it must."""
        nodes = tuple(
            replace(n, model=assignment.get(n.id, n.model)) for n in self.nodes
        )
        return replace(self, nodes=nodes)

    # -- execution ---------------------------------------------------------------

    def execution_order(self) -> Iterator[Node]:
        """Nodes in the fixed declared order, which is already a topological order."""
        return iter(self.nodes)

    def token_cost_nodes(self) -> int:
        """Number of endpoint calls one run of this graph costs."""
        return len(self.nodes)
