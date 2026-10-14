"""The only runtime boundary that calls endpoints or exposes optimizer scores.

Execution is available for every split, including test; selection is not. Optimizer
batches accept training tasks only. Replay preserves the original labelled graph's
provenance even when another graph requests the same exec_hash and sample.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from string import Formatter
from typing import Iterable, NamedTuple, Sequence

from data.loaders.math import Score, Task, score_output
from runtime.cache import RunCache
from runtime.endpoints import CallAttempt, CallContext, Endpoint, observe_call, usage_totals
from space.topology import AGGREGATION_CONCAT_RERASON, Topology


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: str
    exp: str
    model: str
    graph_hash: str
    exec_hash: str
    task_id: str
    split_role: str
    seed: int
    repeat: int
    score: float
    parse_fail: bool
    prediction: str | None
    prompt_tokens: int
    gen_tokens: int
    reasoning_tokens: int | None
    truncated_nodes: tuple[str, ...]
    latency_s: float
    attempts: int
    node_outputs_path: str
    endpoint_usage: dict
    effective_roles: dict[str, str]


class EvalResult(NamedTuple):
    exec_hash: str
    task_ids: tuple[str, ...]
    scores: tuple[float, ...]
    prompt_tokens: int
    gen_tokens: int


class CacheMiss(LookupError):
    """A replay-only Executor was asked for a cell the tensor does not hold."""

    def __init__(self, key: tuple, split_role: str):
        model, exec_hash, task_id, seed, repeat = key
        super().__init__(
            f"replay-only: no cached run for exec_hash={exec_hash[:12]} "
            f"task={task_id} split={split_role} seed={seed} repeat={repeat}")
        self.key = key


class ExecutionError(RuntimeError):
    """An audited failure, never returned as a fabricated optimizer evaluation."""

    def __init__(self, record: RunRecord):
        super().__init__(f"run {record.run_id} failed; see {record.node_outputs_path}")
        self.record = record


def _digest(*values: object) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def render_inputs(received: Sequence[tuple[str, str]]) -> str:
    """The `{inputs}` substitution: a whole section, or nothing at all.

    Shared with `runtime.single_agent` so that a solver sample drawn for pi_single and
    an input-less solver inside a topology receive byte-identical prompts. If they
    differed, C would absorb the difference.
    """
    if not received:
        return ""
    body = "\n\n".join(f"[{role}]\n{text}\n[/{role}]" for role, text in received)
    return f"\nReceived work:\n\n{body}\n"


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Executor:
    def __init__(self, endpoint: Endpoint, cache: RunCache, *, max_tokens_per_node: int,
                 temperature: float, train_repeats: int, runs_dir: str | Path = "runs",
                 prompts_dir: str | Path | None = None, sample_seed: int = 0,
                 replay_only: bool = False):
        if max_tokens_per_node < 1 or train_repeats < 1:
            raise ValueError("token cap and train_repeats must be positive")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if not cache.exp or Path(cache.exp).name != cache.exp or cache.exp in (".", ".."):
            raise ValueError("exp must be a single directory name")
        self.endpoint, self.cache, self.exp = endpoint, cache, cache.exp
        self._max_tokens_per_node, self._temperature = max_tokens_per_node, temperature
        self._train_repeats = train_repeats
        if type(sample_seed) is not int or sample_seed < 0:
            raise ValueError("sample_seed must be a nonnegative integer")
        self._sample_seed = sample_seed
        self._replay_only = bool(replay_only)
        self._output_dir = Path(runs_dir) / self.exp / "outputs"
        prompt_root = (Path(prompts_dir) if prompts_dir is not None
                       else Path(__file__).resolve().parents[1] / "prompts")
        self._templates = {
            f"{role}.v1": (prompt_root / f"{role}.v1.txt").read_text(encoding="utf-8")
            for role in ("solver", "critic", "reviser", "finalizer")
        }
        for template in self._templates.values():
            fields = {field for _, field, _, _ in Formatter().parse(template)
                      if field is not None}
            if fields != {"task", "inputs"}:
                raise ValueError("each role template must have only {task} and {inputs} fields")
            template.format(task="", inputs="")
        cache.bind_experiment({
            "model": endpoint.name, "endpoint": getattr(endpoint, "base_url", None),
            "max_tokens_per_node": max_tokens_per_node, "temperature": temperature,
            "train_repeats": train_repeats, "templates": self._templates,
            "aggregation": AGGREGATION_CONCAT_RERASON,
        })
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._hits: dict[tuple[int, str, str], int] = {}
        self._batch_lock = threading.Lock()

    @property
    def max_tokens_per_node(self) -> int:
        return self._max_tokens_per_node

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def train_repeats(self) -> int:
        return self._train_repeats

    @property
    def replay_only(self) -> bool:
        """Refuse to call the endpoint; a cache miss is an error instead.

        Turned on for optimizer sweeps after the tensor is built. "Every seed replays
        for free" is otherwise only a verbal claim: a bug in the repeat allocator, a
        changed prompt, or a graph outside the tensor would quietly commission new
        generations and inflate the bill by multiples. With this on, that becomes a
        loud failure naming the missing cell.
        """
        return self._replay_only

    @property
    def sample_seed(self) -> int:
        """Identity of the tensor's sample set, and part of every cache key.

        A tensor cell is one LLM draw for (exec_hash, task, repeat); which searcher
        asked for it is not part of its identity. Keeping the optimizer's seed out of
        the key is what lets one tensor serve every seed -- with it in, 49 of 50 seeds
        miss every cell and the sweep costs 8.7x the tensor instead of nothing.
        """
        return self._sample_seed

    def run(self, topology: Topology, task: Task, seed: int, repeat: int) -> RunRecord:
        if repeat < 0:
            raise ValueError("repeat must be nonnegative")
        if topology.aggregation != AGGREGATION_CONCAT_RERASON:
            raise ValueError("only concat_rereason is supported")
        if any(node.model != self.endpoint.name or node.tools for node in topology.nodes):
            raise ValueError("every node must use this endpoint's model and no tools")
        for node in topology.execution_order():
            prompt_id = node.effective_prompt_id(bool(topology.parents(node.id)))
            if prompt_id not in self._templates:
                raise ValueError(f"unknown prompt template: {prompt_id}")
        key = (self.endpoint.name, topology.exec_hash, task.task_id, seed, repeat)
        with self.cache.claim(key):
            cached = self.cache.get(key)
            if cached is not None:
                return cached
            if self._replay_only:
                raise CacheMiss(key, task.split_role)
            return self._execute(topology, task, seed, repeat)

    def _execute(self, topology: Topology, task: Task, seed: int, repeat: int) -> RunRecord:
        started = time.monotonic()
        run_id = uuid.uuid4().hex
        path = self._output_dir / f"{run_id}.json"
        roles = topology.effective_roles()
        exec_hash, graph_hash = topology.exec_hash, topology.graph_hash
        outputs: dict[str, dict] = {}
        artifact = {
            "run_id": run_id, "exp": self.exp, "model": self.endpoint.name,
            "graph_hash": graph_hash, "exec_hash": exec_hash, "task_id": task.task_id,
            "split_role": task.split_role, "seed": seed, "repeat": repeat,
            "status": "running", "nodes": outputs,
        }

        def record(score: Score = Score(0.0, True, None)) -> RunRecord:
            endpoint_usage = {
                node_id: [attempt["raw_usage"] for attempt in output["attempts"]]
                for node_id, output in outputs.items()
            }
            prompt, completion, reasoning = usage_totals(
                usage for usages in endpoint_usage.values() for usage in usages)
            return RunRecord(
                run_id=run_id, exp=self.exp, model=self.endpoint.name,
                graph_hash=graph_hash, exec_hash=exec_hash, task_id=task.task_id,
                split_role=task.split_role, seed=seed, repeat=repeat,
                score=score.score, parse_fail=score.parse_fail, prediction=score.prediction,
                prompt_tokens=prompt, gen_tokens=completion, reasoning_tokens=reasoning,
                truncated_nodes=tuple(nid for nid, output in outputs.items()
                                      if output.get("truncated", False)),
                latency_s=time.monotonic() - started,
                attempts=sum(len(output["attempts"]) for output in outputs.values()),
                node_outputs_path=str(path), endpoint_usage=endpoint_usage,
                effective_roles=roles,
            )

        try:
            for node_index, node in enumerate(topology.execution_order()):
                parents = topology.parents(node.id)
                prompt_id = node.effective_prompt_id(bool(parents))
                # Labels carry the *effective* role, so equivalent graphs issue
                # identical prompts. The whole block lives inside `inputs` so that a
                # node with no parents gets no dangling "Received work:" header — the
                # bare solvers in the vote graphs and pi_single's CoT must not be
                # handed a degraded prompt relative to nodes that do receive work.
                inputs = render_inputs(
                    [(roles[parent], outputs[parent]["text"]) for parent in parents])
                prompt = self._templates[prompt_id].format(task=task.prompt, inputs=inputs)
                node_seed = int(_digest(seed, exec_hash, task.task_id, repeat, node_index)[:8], 16)
                output = {
                    "effective_role": roles[node.id], "prompt_id": prompt_id,
                    "prompt": prompt, "seed": node_seed, "text": "", "attempts": [],
                }
                outputs[node.id] = output

                def observe(attempt: CallAttempt) -> None:
                    output["attempts"].append(asdict(attempt))
                    if attempt.error is None:
                        output["text"] = attempt.text
                        output["finish_reason"] = attempt.finish_reason
                        output["truncated"] = attempt.finish_reason == "length"
                    _write_json(path, artifact)
                    self.cache.record_attempt(record())

                context = CallContext(exec_hash, task.task_id, len(topology.nodes),
                                      node_index, seed, repeat)
                with observe_call(context, observe):
                    result = self.endpoint.complete(
                        prompt, seed=node_seed, max_tokens=self.max_tokens_per_node,
                        temperature=self.temperature)
                if not output["attempts"]:
                    # A minimal third-party Endpoint can report one attempt through
                    # CallResult alone. Retrying adapters must publish every attempt.
                    if result.attempts != 1:
                        raise ValueError("retrying endpoints must emit per-attempt usage")
                    observe(CallAttempt(1, 200, None, result.text, result.finish_reason,
                                        result.raw_usage, result.latency_s))
                totals = usage_totals(a["raw_usage"] for a in output["attempts"])
                if totals != (result.prompt_tokens, result.completion_tokens,
                              result.reasoning_tokens) or len(output["attempts"]) != result.attempts:
                    raise ValueError("endpoint result does not reconcile with its attempts")
                if result.truncated != (result.finish_reason == "length"):
                    raise ValueError("inconsistent endpoint truncation signal")
                output.update(text=result.text, finish_reason=result.finish_reason,
                              truncated=result.truncated, prompt_tokens=result.prompt_tokens,
                              gen_tokens=result.completion_tokens,
                              reasoning_tokens=result.reasoning_tokens, latency_s=result.latency_s)

            # Even a parseable terminal answer is unusable if an upstream node was
            # truncated: the requested computation did not finish as specified.
            score = (Score(0.0, True, None) if any(o["truncated"] for o in outputs.values())
                     else score_output(outputs[topology.terminal]["text"], task.answer))
            completed = record(score)
            artifact["status"] = "complete"
            _write_json(path, artifact)
            self.cache.put(completed)
            return completed
        except Exception as exc:
            artifact.update(status="failed", error=type(exc).__name__)
            _write_json(path, artifact)
            failed = record()
            self.cache.record_attempt(failed)
            raise ExecutionError(failed) from exc

    def _repeat_index(self, seed: int, exec_hash: str, task_id: str, hit_index: int) -> int:
        """Hash each hit, resolving collisions within an R-hit cycle.

        Plain hash(...) % R can repeat the very first sample on the next mask.
        Linear probing gives R fresh samples before deliberate tensor reuse.
        """
        cycle_start = hit_index - hit_index % self.train_repeats
        used: set[int] = set()
        for hit in range(cycle_start, hit_index + 1):
            repeat = int(_digest(seed, exec_hash, task_id, hit), 16) % self.train_repeats
            while repeat in used:
                repeat = (repeat + 1) % self.train_repeats
            used.add(repeat)
        return repeat

    def evaluate_batch(self, topologies: Iterable[Topology], tasks: Iterable[Task],
                       seed: int, repeat_hint: int) -> list[EvalResult]:
        """Training scores only; repeat_hint is a lower bound on each hit counter.

        `seed` is the *search* seed. It decides which candidates were proposed and
        which repeat index each hit draws, but it is deliberately absent from the
        cache key: see :attr:`sample_seed`. Two searches at different seeds therefore
        read different samples out of the same tensor rather than commissioning their
        own, and their search randomness stays fully independent while the LLM
        sampling noise is shared. That shared part is what `repeat_variance` in
        analysis/bootstrap.py bounds.

        Counters span calls and equivalent masks within this Executor. Replaying an
        optimizer uses a fresh Executor and the same proposal/hint sequence;
        run(..., repeat=...) replays an explicit tensor cell statelessly.
        """
        tasks = tuple(tasks)
        if any(task.split_role != "train" for task in tasks):
            raise ValueError("optimizer evaluate_batch accepts training tasks only")
        if repeat_hint < 0:
            raise ValueError("repeat_hint must be nonnegative")
        evaluations: list[EvalResult] = []
        with self._batch_lock:
            for topology in topologies:
                records: list[RunRecord] = []
                for task in tasks:
                    # The search seed picks *which* repeat is drawn; it is not part
                    # of the cell's identity. So it enters _repeat_index but never the
                    # cache key, and every seed reads the one shared tensor.
                    counter_key = (seed, topology.exec_hash, task.task_id)
                    hit = max(repeat_hint, self._hits.get(counter_key, 0))
                    repeat = self._repeat_index(*counter_key, hit)
                    records.append(
                        self.run(topology, task, self._sample_seed, repeat))
                    self._hits[counter_key] = hit + 1
                evaluations.append(EvalResult(
                    topology.exec_hash, tuple(record.task_id for record in records),
                    tuple(record.score for record in records),
                    sum(record.prompt_tokens for record in records),
                    sum(record.gen_tokens for record in records),
                ))
        return evaluations
