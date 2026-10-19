"""Acceptance tests for build step 1 (space/).

The enumeration counts are checked against the closed form derived in space/s0.py,
so a regression in the pruning rule cannot quietly change the size of the ruler.
"""

from __future__ import annotations

import itertools

import pytest

from space.s0 import (
    S0_ROLES,
    S1_ROLES,
    count_closed_form,
    declared_nodes,
    enumerate_space,
    h,
    summarise,
)
from space.topology import (
    AGGREGATION_CONCAT_RERASON,
    CRITIC,
    FINALIZER,
    REVISER,
    SOLVER,
    Node,
    Topology,
)

# --------------------------------------------------------------------- counts

def test_closed_form_matches_known_values():
    assert [h(m) for m in range(6)] == [1, 1, 3, 21, 315, 9765]
    assert count_closed_form(4) == 34
    assert count_closed_form(5) == 422
    assert count_closed_form(6) == 11586


def test_s0_enumerates_34_graphs():
    space = enumerate_space(S0_ROLES)
    assert len(space) == 34
    assert summarise(space)["by_node_count"] == {"1": 1, "2": 3, "3": 9, "4": 21}


def test_s0_strict_enumerates_16_graphs():
    strict = enumerate_space(S0_ROLES, strict_inputs=True)
    assert len(strict) == 16
    assert summarise(strict)["by_node_count"] == {"1": 1, "2": 1, "3": 4, "4": 10}


def test_strict_space_nests_inside_full():
    full = {t.graph_hash for t in enumerate_space(S0_ROLES)}
    strict = {t.graph_hash for t in enumerate_space(S0_ROLES, strict_inputs=True)}
    assert strict < full


def test_s1_enumerates_422_graphs():
    space = enumerate_space(S1_ROLES)
    assert len(space) == 422 == count_closed_form(5)


def test_all_64_edge_subsets_map_into_s0():
    """Every raw edge subset canonicalises to a member of the enumerated space."""
    nodes = declared_nodes(S0_ROLES, "placeholder")
    ids = [n.id for n in nodes]
    space = {t.graph_hash for t in enumerate_space(S0_ROLES)}
    subsets = list(itertools.combinations(ids, 2))
    assert len(subsets) == 6
    for mask in range(1 << 6):
        chosen = [e for i, e in enumerate(subsets) if mask >> i & 1]
        topo = Topology.canonical(nodes, chosen, space_id="s4")
        assert topo.graph_hash in space


# --------------------------------------------------------------------- pruning

def _nodes():
    return declared_nodes(S0_ROLES, "placeholder")


def test_terminal_always_executes_and_empty_graph_is_the_single_agent():
    empty = Topology.canonical(_nodes(), [], space_id="s4")
    assert empty.node_ids == ("n4",)
    assert empty.edges == ()
    assert empty.pruned == ("n1", "n2", "n3")


def test_nodes_that_cannot_reach_terminal_are_pruned():
    # n1->n2 alone: neither reaches n4, so only n4 runs. Same graph as the empty one.
    topo = Topology.canonical(_nodes(), [("n1", "n2")], space_id="s4")
    assert topo.node_ids == ("n4",)
    empty = Topology.canonical(_nodes(), [], space_id="s4")
    assert topo.graph_hash == empty.graph_hash


def test_indirect_path_to_terminal_keeps_nodes():
    topo = Topology.canonical(
        _nodes(), [("n1", "n2"), ("n2", "n4")], space_id="s4")
    assert topo.node_ids == ("n1", "n2", "n4")
    assert topo.edges == (("n1", "n2"), ("n2", "n4"))


def test_only_three_graphs_have_exactly_two_nodes():
    space = enumerate_space(S0_ROLES)
    two = sorted(str(t) for t in space if len(t.nodes) == 2)
    assert two == ["n1->n4", "n2->n4", "n3->n4"]


# --------------------------------------------------------------------- hashing

def test_hash_is_invariant_to_edge_order():
    edges = [("n1", "n2"), ("n2", "n4"), ("n1", "n4")]
    hashes = {
        Topology.canonical(_nodes(), list(p), space_id="s4").graph_hash
        for p in itertools.permutations(edges)
    }
    assert len(hashes) == 1


def test_hash_tracks_model_assignment():
    base = Topology.canonical(_nodes(), [("n1", "n4")], space_id="s4")
    swapped = base.with_models({"n1": "other-model"})
    assert base.graph_hash != swapped.graph_hash


def test_hash_tracks_effective_role_not_declared_role():
    """A Critic with no in-edges is a Solver, and gets the Solver prompt.

    n1->n4 and n2->n4 stay distinct *labelled* graphs (different node ids) but are the
    same *computation*, so they share an exec_hash and therefore a tensor row.
    """
    nodes = _nodes()
    via_solver = Topology.canonical(nodes, [("n1", "n4")], space_id="s4")
    via_critic = Topology.canonical(nodes, [("n2", "n4")], space_id="s4")
    assert via_solver.effective_roles()["n1"] == SOLVER
    assert via_critic.effective_roles()["n2"] == SOLVER
    assert via_critic.nodes[0].effective_prompt_id(has_inputs=False) == "solver.v1"
    assert via_solver.graph_hash != via_critic.graph_hash
    assert via_solver.exec_hash == via_critic.exec_hash


def test_space_id_is_provenance_not_identity():
    """Same graph in S0 and S0_strict must hash the same, or one tensor cannot serve both."""
    nodes = _nodes()
    edges = [("n1", "n2"), ("n2", "n3"), ("n3", "n4")]
    a = Topology.canonical(nodes, edges, space_id="s4")
    b = Topology.canonical(nodes, edges, space_id="s4_strict")
    assert a.graph_hash == b.graph_hash
    assert a.exec_hash == b.exec_hash


def test_distinct_computation_counts():
    """Labelled size is what the optimizer searches; exec size is what we pay for."""
    full = enumerate_space(S0_ROLES)
    assert len({t.exec_hash for t in full}) == 25
    strict = enumerate_space(S0_ROLES, strict_inputs=True)
    assert len({t.exec_hash for t in strict}) == 16  # no degradation, so none collapse
    s1 = enumerate_space(S1_ROLES)
    assert len({t.exec_hash for t in s1}) == 298


def test_vote_family_collapses_to_one_computation_per_width():
    """'k independent solvers -> finalizer' is one computation however it is labelled."""
    nodes = _nodes()
    one = {
        Topology.canonical(nodes, [(s, "n4")], space_id="s4").exec_hash
        for s in ("n1", "n2", "n3")
    }
    two = {
        Topology.canonical(nodes, [(a, "n4"), (b, "n4")], space_id="s4").exec_hash
        for a, b in itertools.combinations(("n1", "n2", "n3"), 2)
    }
    assert len(one) == 1 and len(two) == 1
    assert one != two  # one solver is not two solvers


def test_hash_tracks_aggregation_rule():
    nodes = _nodes()
    edges = [("n1", "n4"), ("n2", "n4")]
    a = Topology.canonical(nodes, edges, space_id="s4")
    b = Topology.canonical(nodes, edges, space_id="s4",
                           aggregation=AGGREGATION_CONCAT_RERASON)
    assert a.graph_hash == b.graph_hash
    with pytest.raises(ValueError):
        Topology.canonical(nodes, edges, space_id="s4", aggregation="majority_vote")


def test_backward_edges_are_rejected():
    with pytest.raises(ValueError, match="not forward"):
        Topology.canonical(_nodes(), [("n4", "n1")], space_id="s4")


# ------------------------------------------------------- the vote / chain families

def test_three_way_vote_graph_exists_and_is_all_solvers():
    """n1->n4, n2->n4, n3->n4: three independent solvers feeding the finalizer.

    This is self-consistency@3 expressed as a topology, the member most likely to be
    optimal on math, and precisely what S0_strict removes.
    """
    topo = Topology.canonical(
        _nodes(), [("n1", "n4"), ("n2", "n4"), ("n3", "n4")], space_id="s4")
    assert len(topo.nodes) == 4
    roles = topo.effective_roles()
    assert [roles[n] for n in ("n1", "n2", "n3")] == [SOLVER, SOLVER, SOLVER]
    assert roles["n4"] == FINALIZER

    strict = {t.graph_hash for t in enumerate_space(S0_ROLES, strict_inputs=True)}
    assert topo.graph_hash not in strict


def test_full_chain_survives_strict_space():
    topo = Topology.canonical(
        _nodes(), [("n1", "n2"), ("n2", "n3"), ("n3", "n4")], space_id="s4")
    roles = topo.effective_roles()
    assert roles["n2"] == CRITIC and roles["n3"] == REVISER
    strict = {t.graph_hash for t in enumerate_space(S0_ROLES, strict_inputs=True)}
    assert topo.graph_hash in strict


def test_parents_are_in_execution_order():
    topo = Topology.canonical(
        _nodes(), [("n3", "n4"), ("n1", "n4"), ("n2", "n4")], space_id="s4")
    assert topo.parents("n4") == ("n1", "n2", "n3")


def test_node_execution_count_drives_token_cost():
    space = enumerate_space(S0_ROLES)
    assert sum(t.token_cost_nodes() for t in space) == 118  # 1 + 3*2 + 9*3 + 21*4


def test_zero_edge_graph_is_a_lone_solver_and_stays_in_the_strict_space():
    g_empty = Topology.canonical(_nodes(), [], space_id="s4")
    assert g_empty.effective_roles() == {"n4": SOLVER}
    assert g_empty.canonical_form()["nodes"][0]["prompt_id"] == "solver.v1"
    strict = {t.graph_hash for t in enumerate_space(S0_ROLES, strict_inputs=True)}
    assert g_empty.graph_hash in strict


def test_a_finalizer_with_inputs_keeps_its_role():
    topo = Topology.canonical(_nodes(), [("n1", "n4")], space_id="s4")
    assert topo.effective_roles()["n4"] == FINALIZER
