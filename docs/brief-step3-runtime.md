# Brief: build step 3 — `runtime/`

You are building one PR-sized step of the `topo-audit` framework. Read `CLAUDE.md`
first; its hard rules override anything in `topo-audit-docs/`, which is the older
design doc. Steps 1 (`space/`) and 2 (`data/`) are done and committed — **do not
modify `space/`, `data/`, `tests/test_space.py` or `tests/test_data.py`.**

## What this step delivers

`runtime/endpoints.py`, `runtime/cache.py`, `runtime/executor.py`,
`prompts/*.txt`, `tests/mock_endpoint.py`, `tests/test_runtime.py`.

Everything must work end to end against the **mock endpoint only**. Do not call the
real gateway — the smoke test against it is a later, separately-approved step.

## Measured facts about the real endpoint (do not re-derive, do not "fix")

These were probed on 2026-10-13 and are recorded in `runs/probe/thinking_budget.json`:

- OpenAI-compatible, base URL `http://aimaster-llm-gateway-service.aimaster-web:8080/llm/v1`,
  key from the `LLM_GATEWAY_API_KEY` env var (loaded from `.env`, never hardcoded).
- **The client must explicitly clear `http_proxy`/`https_proxy`/`all_proxy` (and the
  uppercase forms) for its requests.** With the ambient clash proxy set, this host
  returns 502. (HuggingFace needs the opposite — but that is `data/`'s problem.)
- **`n` > 1 is rejected** (`"Only n=1 is supported."`). k samples cost k calls.
- **Thinking cannot be disabled.** `thinking: {"type": "disabled"}` and
  `reasoning_effort: "low"` are accepted and ignored. `usage` may carry
  `completion_tokens_details.reasoning_tokens`; for some models it is absent even
  though the model is clearly reasoning. Treat it as optional.
- **Truncation is common and matters.** With `max_tokens` too low the model spends the
  whole budget on reasoning and returns `finish_reason="length"` with empty `content`.
  This must surface as a first-class signal, never as a wrong answer.
- The gateway returns sporadic **502**s. Retry with backoff; log every attempt.
- `seed` is accepted. Do not assume bit-level reproducibility.

## Contracts (later steps depend on these exact shapes)

```python
# runtime/endpoints.py
@dataclass(frozen=True)
class CallResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int | None     # None when the endpoint does not itemise it
    finish_reason: str
    truncated: bool                  # finish_reason == "length"
    raw_usage: dict                  # verbatim, for reconciliation against the bill
    attempts: int                    # 1 when it worked first time
    latency_s: float

class Endpoint(Protocol):
    name: str
    def complete(self, prompt: str, *, seed: int, max_tokens: int,
                 temperature: float) -> CallResult: ...

class OpenAICompatEndpoint(Endpoint):  # semaphore-bounded concurrency, retry, proxy-clearing
class MockEndpoint(Endpoint):          # see "mock" below
```

```python
# runtime/cache.py
# key = (model_name, exec_hash, task_id, seed, repeat)   <-- exec_hash, NOT graph_hash
class RunCache:
    def get(self, key: CacheKey) -> RunRecord | None: ...
    def put(self, record: RunRecord) -> None: ...
    def stats(self) -> dict          # hits, misses, writes
    def export(self, exp: str) -> Iterator[RunRecord]
```

SQLite or a sharded JSON tree, your call. Requirements: safe under concurrent writers,
re-running an experiment issues zero endpoint calls, and records are exportable by `exp`.

```python
# runtime/executor.py
@dataclass(frozen=True)
class RunRecord:
    run_id: str; exp: str; model: str
    graph_hash: str; exec_hash: str          # both stored; cache keys on exec_hash
    task_id: str; split_role: str; seed: int; repeat: int
    score: float; parse_fail: bool; prediction: str | None
    prompt_tokens: int; gen_tokens: int; reasoning_tokens: int | None
    truncated_nodes: tuple[str, ...]         # nodes that hit finish_reason=length
    latency_s: float; attempts: int
    node_outputs_path: str                   # runs/<exp>/outputs/<run_id>.json
    endpoint_usage: dict                     # per-node raw usage, for reconciliation
    effective_roles: dict[str, str]

class Executor:
    def run(self, topology, task, seed, repeat) -> RunRecord: ...
    def evaluate_batch(self, topologies, tasks, seed, repeat_hint) -> list[EvalResult]: ...
```

`EvalResult = (exec_hash, task_ids, scores, prompt_tokens, gen_tokens)` — this is the
**only** way an optimizer is ever allowed to obtain a score.

## Execution semantics

- Iterate `topology.execution_order()` (already a topological order).
- A node's prompt = its role template, filled with the task text and, when it has
  in-edges, the parent outputs **in execution order** each tagged with the parent's
  *effective* role. The aggregation rule is `concat_rereason`: concatenate and
  re-reason. Do not implement voting.
- Use `node.effective_prompt_id(has_inputs)` to pick the template — an input-less
  Critic or Reviser gets the Solver template. `space/topology.py` already computes this.
- The answer is the terminal node's output, scored with
  `data.loaders.math.score_output`.
- `prompt_tokens`/`gen_tokens` are summed over nodes. Retried attempts count too:
  tokens spent are tokens spent.
- Write every node's output to `node_outputs_path`; the RunRecord stores only the path.

## Prompts (`prompts/`)

Four templates, `solver.v1`, `critic.v1`, `reviser.v1`, `finalizer.v1`, plain text with
`{task}` and `{inputs}` placeholders. Frozen for E0 — no optimisation. Requirements:
every template must instruct the model to end with a `\boxed{}` answer and a final
line `FINAL: <answer>`; Critic must critique the received solution without being told
the answer; Finalizer must reconcile everything it received and commit to one answer.
Keep them short and neutral — they are an experimental control, not a prompt-
engineering exercise.

## Mock endpoint

`tests/mock_endpoint.py` must let the runtime tests be deterministic and exercise the
real failure shapes:

- Each `exec_hash` gets a fixed latent accuracy, monotonically increasing in node count
  plus a hash-derived jitter, so "more structure is slightly better" holds on average.
- Outputs are synthesised to be scorable: on a success draw emit the gold answer in a
  `FINAL:` line, otherwise a plausible wrong one.
- Configurable truncation rate (empty content, `finish_reason="length"`) and
  transient-error rate, so the retry path and `parse_fail` path are both covered.
- Report token counts so the accounting tests have something to reconcile.

## Acceptance (your work is done when all of these pass)

1. `python3 -m pytest tests/ -q` is green, including the 58 pre-existing tests.
2. Mock run of all 25 S0 exec rows x 5 tasks completes; summed per-node
   `prompt_tokens + gen_tokens` equals the endpoint-reported usage exactly.
3. Re-running the same sweep issues **zero** endpoint calls (assert on a call counter).
4. Two different labelled graphs with the same `exec_hash` share a cache entry.
5. The repeat index is allocated per `(seed, exec_hash, task_id, hit_index)` so a
   second equivalent mask draws a *fresh* repeat rather than re-reading repeat 0.
6. A truncated generation yields `parse_fail=True`, `score=0.0`, and the node listed in
   `truncated_nodes` — not a silently wrong answer.
7. A transient 502 is retried, succeeds, and `attempts > 1` is recorded.
8. Loading a task and running it works for every `split_role` **including `test`** —
   the rule is "no argmax/ranking/thresholding on test", not "test is unreadable".
   Add a test asserting the executor does not special-case `test`.

## House style

Match the existing code: `from __future__ import annotations`, frozen slotted
dataclasses, type hints, module docstrings that explain *why* a constraint exists.
Comment density like `space/topology.py` — explain the non-obvious, skip the obvious.
No new third-party dependencies beyond `requirements.txt`.

Do not `git commit`; leave the working tree dirty for review. When you finish, print a
short summary: files added, test counts, and anything in this brief you judged wrong.
