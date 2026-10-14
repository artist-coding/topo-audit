"""pi_single: the declared single-agent strategy set, measured as a frontier.

C = P(G_star) - P(pi_single) is the paper's headline quantity, so pi_single decides
C's sign. Two things follow and are implemented here.

**The frontier is primary, the headline point is derived.** The proposal asks for a
frontier rather than a single number; the framework guide hardcoded "the median S0
token cost". Those conflict, and the conflict matters because self-consistency@k gets
stronger with budget. So this module measures accuracy *and* realised token cost at
every budget level, and :func:`pick_headline` selects the strategy at a given budget
afterwards. The S0 token median is not known until the audit tensor exists, which is
why it cannot be baked in here.

**One sample draw serves the whole frontier.** The gateway rejects n > 1, so k samples
cost k calls — but k samples also *contain* every smaller k. Drawing 8 independent
solver samples once gives self-consistency@1..8 by prefix and, with one self-check
call per sample, best-of-N@1..8 as well. Only self-refine is irreducibly sequential
(1 + 2r calls), so it is run per r. Curves at different k therefore share samples and
are positively correlated: that is deliberate, since every comparison the paper makes
between budget levels is a paired one, but it means the k-curve's pointwise CIs are
not independent of each other.

**Self-refine is a topology in disguise.** Running critique-then-revise with the same
critic.v1 / reviser.v1 templates the audit space uses makes self-refine@1 nearly the
same computation as the chain n1->n2->n3->n4, minus the finalizer. Keeping it in the
set is the honest reading of H1 ("a sufficiently capable single controller can simulate
any MAS workflow"), and it means C measures what topology adds *beyond sequential
self-criticism by the same model* — the strongest form of the claim, and the hardest.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from data.loaders.math import answers_match, extract_prediction, normalize_answer
from runtime.cache import RunCache
from runtime.endpoints import CallContext, Endpoint, observe_call, usage_totals
from runtime.executor import render_inputs

COT = "cot"
SELF_CONSISTENCY = "self_consistency"
BEST_OF_N_SELFCHECK = "best_of_n_selfcheck"
SELF_REFINE = "self_refine"
STRATEGIES = (COT, SELF_CONSISTENCY, BEST_OF_N_SELFCHECK, SELF_REFINE)

#: Independent solver samples drawn per task. Sets the maximum k and N on the frontier.
DEFAULT_SAMPLES = 8


@dataclass(frozen=True, slots=True)
class StrategyRun:
    """One strategy at one budget level on one task."""

    strategy: str
    param: int                      # k, N, or r; 1 for plain CoT
    task_id: str
    seed: int
    score: float
    parse_fail: bool
    prediction: str | None
    prompt_tokens: int
    gen_tokens: int
    reasoning_tokens: int | None
    calls: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.gen_tokens


@dataclass(frozen=True, slots=True)
class FrontierPoint:
    strategy: str
    param: int
    n_tasks: int
    accuracy: float
    accuracy_se: float
    mean_tokens: float
    parse_fail_rate: float

    @property
    def ci95(self) -> tuple[float, float]:
        half = 1.96 * self.accuracy_se
        return max(0.0, self.accuracy - half), min(1.0, self.accuracy + half)


def strategy_hash(strategy: str, param: int, prompt_ids: Sequence[str]) -> str:
    """Cache identity for a single-agent strategy.

    Mirrors ``Topology.exec_hash``: anything that changes what the endpoint is asked
    changes the key. Lives in the same cache as the topology runs so that token
    accounting and replay have exactly one implementation.
    """
    blob = json.dumps({"pi_single": strategy, "param": param,
                       "prompts": list(prompt_ids)},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


# ------------------------------------------------------------------ aggregation

def majority_vote(predictions: Sequence[str | None]) -> str | None:
    """Most common normalised answer; ties break toward the earliest sample.

    Normalisation before counting is what makes the vote meaningful: ``\\frac{1}{2}``
    and ``0.5`` are the same vote. The returned value is the raw prediction of the
    winning group so downstream scoring sees an unmodified model answer.
    """
    groups: dict[str, list[str]] = collections.OrderedDict()
    for pred in predictions:
        key = normalize_answer(pred)
        if key is None:
            continue
        groups.setdefault(key, []).append(pred)  # type: ignore[arg-type]
    if not groups:
        return None
    best = max(groups.items(), key=lambda kv: (len(kv[1]), -list(groups).index(kv[0])))
    return best[1][0]


def _accumulate(results: Iterable[StrategyRun]) -> list[FrontierPoint]:
    buckets: dict[tuple[str, int], list[StrategyRun]] = {}
    for run in results:
        buckets.setdefault((run.strategy, run.param), []).append(run)
    points = []
    for (strategy, param), runs in sorted(buckets.items()):
        n = len(runs)
        acc = sum(r.score for r in runs) / n
        se = math.sqrt(max(acc * (1 - acc), 1e-12) / n)
        points.append(FrontierPoint(
            strategy=strategy, param=param, n_tasks=n, accuracy=acc, accuracy_se=se,
            mean_tokens=sum(r.total_tokens for r in runs) / n,
            parse_fail_rate=sum(r.parse_fail for r in runs) / n,
        ))
    return points


def pick_headline(
    frontier: Sequence[FrontierPoint], budget_tokens: float
) -> FrontierPoint | None:
    """Strongest declared strategy whose mean cost fits the budget.

    This is P(pi_single) in the decomposition. Called with the S0 token median for the
    headline number, and swept across budgets to report C as a curve.
    """
    affordable = [p for p in frontier if p.mean_tokens <= budget_tokens]
    return max(affordable, key=lambda p: p.accuracy) if affordable else None


# -------------------------------------------------------------------- execution

class SingleAgentRunner:
    """Runs the declared strategy set, reusing the topology cache and accounting."""

    def __init__(
        self,
        endpoint: Endpoint,
        cache: RunCache,
        templates: Mapping[str, str],
        *,
        exp: str,
        max_tokens: int,
        temperature: float,
        samples: int = DEFAULT_SAMPLES,
        output_dir: str | Path = "runs",
    ) -> None:
        missing = {"solver.v1", "critic.v1", "reviser.v1", "selfcheck.v1"} - set(templates)
        if missing:
            raise ValueError(f"missing prompt templates: {sorted(missing)}")
        self.endpoint = endpoint
        self.cache = cache
        self.templates = dict(templates)
        self.exp = exp
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.samples = samples
        self.output_dir = Path(output_dir)

    # -- one cached endpoint call ------------------------------------------------

    def _call(self, prompt: str, *, strategy: str, param: int, prompt_ids: Sequence[str],
              task_id: str, seed: int, repeat: int) -> tuple[str, dict]:
        """Single endpoint call, memoised in the shared cache.

        Returns the text plus a usage dict. Retried attempts are already folded into
        the endpoint's reported usage, so tokens spent on a failed attempt are counted.
        """
        from runtime.executor import RunRecord  # local import: avoids a cycle

        identity = strategy_hash(strategy, param, prompt_ids)
        key = (self.endpoint.name, identity, task_id, seed, repeat)
        with self.cache.claim(key):
            cached = self.cache.get(key)
            if cached is not None:
                payload = json.loads(Path(cached.node_outputs_path).read_text())
                return payload["text"], payload["usage"]

            # Collect every attempt, not just the one that succeeded: a retried 502
            # still spent tokens, and the equal-compute comparison may not under-count
            # pi_single's bill relative to a topology's.
            attempts: list[dict] = []
            context = CallContext(exec_hash=identity, task_id=task_id, node_count=1,
                                  node_index=0, seed=seed, repeat=repeat)
            with observe_call(context, lambda a: attempts.append(asdict(a))):
                result = self.endpoint.complete(
                    prompt, seed=seed + repeat * 100_003, max_tokens=self.max_tokens,
                    temperature=self.temperature)
            raw_usages = [a["raw_usage"] for a in attempts] or [result.raw_usage]
            prompt_tok, gen_tok, reasoning = usage_totals(raw_usages)

            run_id = hashlib.sha256(
                json.dumps(key, sort_keys=True).encode()).hexdigest()[:24]
            path = self.output_dir / self.exp / "pi_single" / f"{run_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            usage = {
                "prompt_tokens": prompt_tok, "gen_tokens": gen_tok,
                "reasoning_tokens": reasoning, "truncated": result.truncated,
                "attempts": len(raw_usages), "raw": raw_usages,
            }
            path.write_text(json.dumps(
                {"text": result.text, "prompt": prompt, "usage": usage,
                 "attempts": attempts}, indent=1))
            self.cache.put(RunRecord(
                run_id=run_id, exp=self.exp, model=self.endpoint.name,
                graph_hash=identity, exec_hash=identity, task_id=task_id,
                split_role="pi_single", seed=seed, repeat=repeat,
                score=0.0, parse_fail=not result.text.strip(), prediction=None,
                prompt_tokens=prompt_tok, gen_tokens=gen_tok,
                reasoning_tokens=reasoning,
                truncated_nodes=("solver",) if result.truncated else (),
                latency_s=result.latency_s, attempts=result.attempts,
                node_outputs_path=str(path),
                endpoint_usage={"solver": raw_usages},
                effective_roles={"solver": "solver"},
            ))
            return result.text, usage

    # -- the four strategies -----------------------------------------------------

    def run_task(self, task, seed: int, params: Mapping[str, Sequence[int]] | None = None
                 ) -> list[StrategyRun]:
        """Every frontier point for one task, sharing the solver samples.

        CoT, self-consistency and best-of-N all read the same ``samples`` draws, so the
        whole k/N sweep costs ``samples`` solver calls plus ``samples`` self-check
        calls. Self-refine is sequential and pays per r.
        """
        wanted = dict(params or {})
        ks = wanted.get(SELF_CONSISTENCY, range(1, self.samples + 1))
        ns = wanted.get(BEST_OF_N_SELFCHECK, range(1, self.samples + 1))
        rs = wanted.get(SELF_REFINE, (1, 2, 3))

        solver_prompt = self.templates["solver.v1"].format(
            task=task.prompt, inputs=render_inputs([]))
        draws = [
            self._call(solver_prompt, strategy="solver_sample", param=0,
                       prompt_ids=["solver.v1"], task_id=task.task_id,
                       seed=seed, repeat=i)
            for i in range(self.samples)
        ]
        texts = [text for text, _ in draws]
        usages = [usage for _, usage in draws]
        preds = [extract_prediction(text) for text in texts]

        out: list[StrategyRun] = []

        def emit(strategy: str, param: int, prediction: str | None,
                 used: Sequence[dict], calls: int) -> None:
            prompt_tok = sum(u["prompt_tokens"] for u in used)
            gen_tok = sum(u["gen_tokens"] for u in used)
            known = [u["reasoning_tokens"] for u in used
                     if u["reasoning_tokens"] is not None]
            reasoning = sum(known) if known else None
            out.append(StrategyRun(
                strategy=strategy, param=param, task_id=task.task_id, seed=seed,
                score=float(prediction is not None
                            and answers_match(prediction, task.answer)),
                parse_fail=prediction is None, prediction=prediction,
                prompt_tokens=prompt_tok, gen_tokens=gen_tok,
                reasoning_tokens=reasoning, calls=calls))

        emit(COT, 1, preds[0], usages[:1], 1)
        for k in ks:
            emit(SELF_CONSISTENCY, int(k), majority_vote(preds[:k]), usages[:k], int(k))

        # best-of-N needs one self-check score per sample; N <= samples is then free.
        checks = []
        for i, text in enumerate(texts):
            prompt = self.templates["selfcheck.v1"].format(
                task=task.prompt,
                inputs=render_inputs([("candidate", text or "(no answer produced)")]))
            reply, usage = self._call(prompt, strategy="selfcheck", param=0,
                                      prompt_ids=["selfcheck.v1"],
                                      task_id=task.task_id, seed=seed, repeat=i)
            checks.append((_confidence(reply), usage))
        for n in ns:
            n = int(n)
            ranked = sorted(range(n), key=lambda i: -checks[i][0])
            best = next((preds[i] for i in ranked if preds[i] is not None), None)
            used = usages[:n] + [u for _, u in checks[:n]]
            emit(BEST_OF_N_SELFCHECK, n, best, used, 2 * n)

        # self-refine: solve once, then r rounds of critique + revise.
        refine_usages = [usages[0]]
        current = texts[0]
        for r in range(1, max(rs, default=0) + 1):
            critique, usage = self._call(
                self.templates["critic.v1"].format(
                    task=task.prompt, inputs=render_inputs([("solver", current)])),
                strategy=SELF_REFINE, param=r, prompt_ids=["critic.v1"],
                task_id=task.task_id, seed=seed, repeat=0)
            refine_usages.append(usage)
            revised, usage = self._call(
                self.templates["reviser.v1"].format(
                    task=task.prompt,
                    inputs=render_inputs([("solver", current), ("critic", critique)])),
                strategy=SELF_REFINE, param=r, prompt_ids=["reviser.v1"],
                task_id=task.task_id, seed=seed, repeat=0)
            refine_usages.append(usage)
            current = revised
            if r in rs:
                emit(SELF_REFINE, r, extract_prediction(current),
                     list(refine_usages), 1 + 2 * r)
        return out

    def frontier(self, tasks, seed: int = 0,
                 params: Mapping[str, Sequence[int]] | None = None
                 ) -> tuple[list[FrontierPoint], list[StrategyRun]]:
        runs = [r for task in tasks for r in self.run_task(task, seed, params)]
        return _accumulate(runs), runs


def _confidence(reply: str) -> float:
    """Self-check score in [0, 1]; unparseable replies sort last."""
    import re

    match = re.search(r"(?:SCORE|CONFIDENCE)\s*[:=]\s*([0-9]*\.?[0-9]+)", reply or "",
                      re.IGNORECASE)
    if not match:
        return -1.0
    value = float(match.group(1))
    return value / 10.0 if value > 1.0 else value
