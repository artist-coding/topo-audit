# Brief: build step 5 — `optimizers/`

Read `CLAUDE.md` first; its hard rules override `topo-audit-docs/`, which is the older
design doc. Steps 1–4 are committed. **Do not modify `space/`, `data/`, `runtime/`, or
any existing test file.** You add `optimizers/` and `tests/test_optimizers.py` only.

Mock endpoint only. No real gateway calls. Do not `git commit`.

## What this step delivers

`optimizers/base.py`, `optimizers/random_search.py`, `optimizers/reinforce_edge.py`,
`optimizers/agentprune.py`, `optimizers/greedy_local.py`, `optimizers/runner.py`,
`tests/test_optimizers.py`.

## The contract

```python
# optimizers/base.py
@dataclass
class Budget:
    max_evals: int
    tasks_per_eval: int
    spent_evals: int = 0
    spent_tokens: int = 0
    def left(self) -> int: ...

class Optimizer(Protocol):
    def __init__(self, space: Sequence[Topology], budget: Budget, seed: int): ...
    def propose(self) -> list[Topology]: ...      # next batch of candidates
    def observe(self, evals: Sequence[EvalResult]) -> None: ...
    def incumbent(self) -> Topology: ...          # best by its OWN empirical estimate
    def state(self) -> dict: ...                  # JSON-serialisable, goes in trajectory
```

`EvalResult` comes from `runtime.executor` and is the **only** way to get a score.
An optimizer must not touch the endpoint, the cache, or any split other than `train`.
`Executor.evaluate_batch` already rejects non-training tasks — add a test that proves
an optimizer cannot route around it.

One **candidate evaluation** = one graph on `budget.tasks_per_eval` training tasks.
`budget.max_evals` counts those, not endpoint calls. `spent_tokens` accumulates the
`prompt_tokens + gen_tokens` the EvalResults report, which is what the equal-compute
comparison later divides by.

## Shared task batching (this is what makes comparisons paired)

`optimizers/runner.py` owns a deterministic batch sequence: batch *i* is the same 24
training tasks for **every** optimizer and **every** seed. Two optimizers evaluated at
step *i* therefore saw identical tasks, which is the pairing the plateau test and the
discriminability diagnostic both rely on. Derive the sequence from the committed split
order plus a fixed seed — never from `random` module state.

Pass `repeat_hint` through to `evaluate_batch` so the repeat allocator in step 3 keeps
handing out fresh samples; read `runtime/executor.py` for how hit counters work before
you wire this up.

## The four optimizers

| name | candidate selection | extra state to log |
| --- | --- | --- |
| `random_search` | uniform over the de-duplicated space; incumbent = highest running mean among graphs with ≥ 2 evaluations (one lucky draw must not win) | per-graph mean and count |
| `reinforce_edge` | one logit per optional edge, independent Bernoulli sampling; each step draws a group of 8 graphs, evaluates them on the **same** task batch, computes group-relative advantage `(r_i - mean) / (std + 1e-6)`, updates by REINFORCE with lr 0.1, clips logits to [-4, 4], entropy regularisation 0.01; incumbent = the graph obtained by rounding edge probabilities | logits, group scores, advantages |
| `agentprune` | start from the fully-connected graph, learn a continuous per-edge mask with magnitude regularisation, then prune the smallest-magnitude edge each round | mask values, prune order |
| `greedy_local` | hill climbing over the add-one-edge / remove-one-edge neighbourhood; evaluate neighbours, move to the best improvement, restart from a fresh random graph when no neighbour improves | current point, neighbourhood scores, restart count |

Canonicalise every proposal through `Topology.canonical` before evaluating — several
edge masks collapse to the same executable graph, and the optimizer should pay for the
evaluation it actually requested.

### Two things not to "fix"

- **`agentprune` visits very few graphs.** Its monotone edge-removal path touches
  roughly one graph per prune round, so its empirical optimization gap will be much
  larger than random search's. That is a property of the search path, not a defect —
  it is exactly why it is in the study. Document it in the module docstring; do not add
  exploration to make it competitive.
- **`random_search` will exhaust S0.** With 25 distinct computations and 200
  evaluations it covers the space several times over, so the O term converges to zero.
  Expected. The runner must therefore record a checkpoint at **every step**, so the
  early-`t` regime where O is non-zero is visible. Do not shorten the budget.

## Trajectory

`runs/<exp>/<optimizer>/seed<k>/trajectory.jsonl`, one line per step:

```
{"step", "candidates": [graph_hash...], "exec_hashes": [...], "task_ids": [...],
 "scores": [[per-task...]...], "cum_evals", "cum_tokens",
 "incumbent": graph_hash, "state": {...},
 "discriminability": {"top2_paired_diff": float, "se": float, "n_tasks": int}}
```

`discriminability` is the paired score difference between the two best current
candidates on the same task batch, plus its standard error. It is **recorded for
analysis only** — never a stopping rule, never a threshold. The analysis step uses it
to explain when an optimizer stalls.

## Acceptance

1. `python3 -m pytest tests/ -q` green, including all 131 existing tests.
2. On the mock endpoint, `random_search`, `reinforce_edge` and `greedy_local` each
   push `incumbent()` into the true top 3 of the space within budget, on 5 seeds.
   (`agentprune` is exempt by design — assert instead that its gap is *larger*.)
3. `budget.spent_evals` equals the number of candidate evaluations, and
   `spent_tokens` equals the sum over EvalResults.
4. Trajectories replay byte-identically: same seed + same cache → same JSONL.
5. Two optimizers at step *i* received the same task ids.
6. An optimizer handed audit/val/test tasks raises rather than scoring them.
7. No optimizer holds a reference to an endpoint or cache — assert structurally.
8. Every proposal written to the trajectory is a canonical topology.

## House style

Match `space/topology.py` and `runtime/executor.py`: `from __future__ import
annotations`, frozen slotted dataclasses where the value is immutable, type hints,
module docstrings that explain *why* a constraint exists rather than restating the
code. No new third-party dependencies.

When done, print: files added, test counts, and anything in this brief you judged
wrong.
