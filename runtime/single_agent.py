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

**Scored under the executor's rule.** A topology run in which any node was
truncated scores 0, because the requested computation did not finish. A strategy run
is scored the same way: if any call it used was truncated, it scores 0, even when the
partial text happens to contain a boxed answer. Otherwise the same computation would
be scored differently on the two sides of C -- G_empty and CoT now issue identical
prompts, so they must also be judged by identical rules.

**Selected on val, reported on test.** :func:`select_on_val_report_on_test` picks the
strategy and budget parameter on the validation split at the headline token budget,
then evaluates only that strategy on test, keeping per-task scores so that C is a
paired difference. Picking the best strategy on test would be the argmax-on-test the
hard rules forbid.

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

if __package__ in (None, ""):  # run as a script: make the repo root importable
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    truncated: bool = False

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
    truncation_rate: float = 0.0

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
            truncation_rate=sum(r.truncated for r in runs) / n,
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

    # -- building blocks shared by run_task and run_strategy ----------------------

    def _draws(self, task, seed: int, n: int) -> tuple[list[str], list[dict]]:
        """The first n independent solver samples; identical prompts to a bare node."""
        solver_prompt = self.templates["solver.v1"].format(
            task=task.prompt, inputs=render_inputs([]))
        draws = [
            self._call(solver_prompt, strategy="solver_sample", param=0,
                       prompt_ids=["solver.v1"], task_id=task.task_id,
                       seed=seed, repeat=i)
            for i in range(n)
        ]
        return [text for text, _ in draws], [usage for _, usage in draws]

    def _checks(self, task, seed: int, texts: Sequence[str]) -> list[tuple[float, dict]]:
        """One self-check score per sample, in sample order."""
        checks = []
        for i, text in enumerate(texts):
            prompt = self.templates["selfcheck.v1"].format(
                task=task.prompt,
                inputs=render_inputs([("candidate", text or "(no answer produced)")]))
            reply, usage = self._call(prompt, strategy="selfcheck", param=0,
                                      prompt_ids=["selfcheck.v1"],
                                      task_id=task.task_id, seed=seed, repeat=i)
            checks.append((_confidence(reply), usage))
        return checks

    def _refine(self, task, seed: int, first_text: str, first_usage: dict,
                rounds: int) -> list[tuple[int, str | None, list[dict]]]:
        """(r, prediction after r rounds, every usage so far) for r = 1..rounds."""
        usages = [first_usage]
        current = first_text
        out = []
        for r in range(1, rounds + 1):
            critique, usage = self._call(
                self.templates["critic.v1"].format(
                    task=task.prompt, inputs=render_inputs([("solver", current)])),
                strategy=SELF_REFINE, param=r, prompt_ids=["critic.v1"],
                task_id=task.task_id, seed=seed, repeat=0)
            usages.append(usage)
            revised, usage = self._call(
                self.templates["reviser.v1"].format(
                    task=task.prompt,
                    inputs=render_inputs([("solver", current), ("critic", critique)])),
                strategy=SELF_REFINE, param=r, prompt_ids=["reviser.v1"],
                task_id=task.task_id, seed=seed, repeat=0)
            usages.append(usage)
            current = revised
            out.append((r, extract_prediction(current), list(usages)))
        return out

    @staticmethod
    def _best_of(preds: Sequence[str | None], checks: Sequence[tuple[float, dict]],
                 n: int) -> str | None:
        ranked = sorted(range(n), key=lambda i: -checks[i][0])
        return next((preds[i] for i in ranked if preds[i] is not None), None)

    @staticmethod
    def _scored(task, seed: int, strategy: str, param: int, prediction: str | None,
                used: Sequence[dict], calls: int) -> StrategyRun:
        """One strategy run, scored under the executor's truncation rule."""
        truncated = any(usage.get("truncated", False) for usage in used)
        if truncated:
            prediction = None
        known = [u["reasoning_tokens"] for u in used if u["reasoning_tokens"] is not None]
        return StrategyRun(
            strategy=strategy, param=param, task_id=task.task_id, seed=seed,
            score=float(prediction is not None and answers_match(prediction, task.answer)),
            parse_fail=prediction is None, prediction=prediction,
            prompt_tokens=sum(u["prompt_tokens"] for u in used),
            gen_tokens=sum(u["gen_tokens"] for u in used),
            reasoning_tokens=sum(known) if known else None, calls=calls,
            truncated=truncated)

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

        texts, usages = self._draws(task, seed, self.samples)
        preds = [extract_prediction(text) for text in texts]
        out = [self._scored(task, seed, COT, 1, preds[0], usages[:1], 1)]
        for k in ks:
            k = int(k)
            out.append(self._scored(task, seed, SELF_CONSISTENCY, k,
                                    majority_vote(preds[:k]), usages[:k], k))
        # best-of-N needs one self-check score per sample; N <= samples is then free.
        checks = self._checks(task, seed, texts)
        for n in ns:
            n = int(n)
            out.append(self._scored(task, seed, BEST_OF_N_SELFCHECK, n,
                                    self._best_of(preds, checks, n),
                                    usages[:n] + [u for _, u in checks[:n]], 2 * n))
        # self-refine: solve once, then r rounds of critique + revise.
        for r, prediction, used in self._refine(task, seed, texts[0], usages[0],
                                                max(rs, default=0)):
            if r in rs:
                out.append(self._scored(task, seed, SELF_REFINE, r, prediction, used,
                                        1 + 2 * r))
        return out

    def run_strategy(self, task, seed: int, strategy: str, param: int) -> StrategyRun:
        """One strategy at one budget parameter, issuing only the calls it needs.

        Uses the cache identities of :meth:`run_task`, so its result equals the
        corresponding frontier point; evaluating the val-selected strategy on test
        costs k (or N + N, or 1 + 2r) calls per task instead of the whole frontier.
        """
        if type(param) is not int or param < 1:
            raise ValueError("param must be a positive integer")
        if strategy == COT:
            texts, usages = self._draws(task, seed, 1)
            return self._scored(task, seed, COT, 1, extract_prediction(texts[0]), usages, 1)
        if strategy == SELF_CONSISTENCY:
            texts, usages = self._draws(task, seed, param)
            preds = [extract_prediction(text) for text in texts]
            return self._scored(task, seed, SELF_CONSISTENCY, param,
                                majority_vote(preds), usages, param)
        if strategy == BEST_OF_N_SELFCHECK:
            texts, usages = self._draws(task, seed, param)
            preds = [extract_prediction(text) for text in texts]
            checks = self._checks(task, seed, texts)
            return self._scored(task, seed, BEST_OF_N_SELFCHECK, param,
                                self._best_of(preds, checks, param),
                                usages + [u for _, u in checks], 2 * param)
        if strategy == SELF_REFINE:
            texts, usages = self._draws(task, seed, 1)
            r, prediction, used = self._refine(task, seed, texts[0], usages[0], param)[-1]
            return self._scored(task, seed, SELF_REFINE, r, prediction, used, 1 + 2 * r)
        raise ValueError(f"unknown strategy {strategy!r}")

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


# ------------------------------------------------------------------ selection

def s0_token_median(records: Iterable, split: str = "audit") -> float | None:
    """Median over S0's distinct computations of their mean tokens per run.

    The headline budget for pi_single. Taken over the tensor's rows (one per
    exec_hash), so equivalent edge masks do not count twice, and on audit, where the
    tensor is complete. Tokens are prompt + generation, the same bill pi_single pays.
    """
    per_row: dict[str, list[int]] = {}
    for record in records:
        if record.split_role == split:
            per_row.setdefault(record.exec_hash, []).append(
                record.prompt_tokens + record.gen_tokens)
    if not per_row:
        return None
    means = sorted(sum(v) / len(v) for v in per_row.values())
    mid = len(means) // 2
    return float(means[mid] if len(means) % 2 else (means[mid - 1] + means[mid]) / 2)


def select_on_val_report_on_test(runner: "SingleAgentRunner", val_tasks, test_tasks,
                                 budget_tokens: float, *, seed: int = 0,
                                 params: Mapping[str, Sequence[int]] | None = None,
                                 concurrency: int = 1) -> dict:
    """Choose pi_single on val at the budget, then score only that choice on test."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max(1, concurrency)) as pool:
        val_runs = [run for runs in pool.map(
            lambda task: runner.run_task(task, seed, params), val_tasks) for run in runs]
    val_frontier = _accumulate(val_runs)
    chosen = pick_headline(val_frontier, budget_tokens)
    if chosen is None:
        cheapest = min(val_frontier, key=lambda p: p.mean_tokens)
        raise SystemExit(f"no declared strategy fits {budget_tokens:.0f} tokens "
                         f"(cheapest is {cheapest.strategy}@{cheapest.param} at "
                         f"{cheapest.mean_tokens:.0f})")
    with ThreadPoolExecutor(max(1, concurrency)) as pool:
        test_runs = list(pool.map(
            lambda task: runner.run_strategy(task, seed, chosen.strategy, chosen.param),
            test_tasks))
    n = len(test_runs)
    return {
        "seed": seed,
        "samples": runner.samples,
        "val_frontier": [asdict(point) for point in val_frontier],
        "headline": {
            "strategy": chosen.strategy, "param": chosen.param,
            "budget_tokens": budget_tokens,
            "selected_on": "val", "reported_on": "test",
            "val_accuracy": chosen.accuracy, "val_mean_tokens": chosen.mean_tokens,
            "n_val_tasks": chosen.n_tasks,
            "accuracy": sum(r.score for r in test_runs) / n,
            "mean_tokens": sum(r.total_tokens for r in test_runs) / n,
            "n_tasks": n,
            "parse_fail_rate": sum(r.parse_fail for r in test_runs) / n,
            "truncation_rate": sum(r.truncated for r in test_runs) / n,
            "task_scores": {r.task_id: r.score for r in test_runs},
        },
    }


def load_templates(prompts_dir: str | Path | None = None) -> dict[str, str]:
    root = (Path(prompts_dir) if prompts_dir is not None
            else Path(__file__).resolve().parents[1] / "prompts")
    return {f"{name}.v1": (root / f"{name}.v1.txt").read_text(encoding="utf-8")
            for name in ("solver", "critic", "reviser", "selfcheck")}


def main() -> None:
    """python3 runtime/single_agent.py --config configs/e0.yaml

    Writes runs/<exp>/pi_single/frontier.json, which analysis/report.py reads.
    """
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import yaml

    from data.tasks import load_split_tasks
    from runtime.endpoints import endpoint_from_config
    from runtime.executor import Executor

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--config", default="configs/e0.yaml")
    ap.add_argument("--exp", default=None, help="override config exp (smoke tests)")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--budget", type=float, default=None,
                    help="token budget; default is the S0 token median on audit")
    ap.add_argument("--limit-tasks", type=int, default=0,
                    help="smoke test: first N tasks of val and of test")
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    exp = args.exp or config["exp"]
    model = config["model"]
    runs_dir = Path(args.runs_dir)
    sweep = sorted({int(k) for k in config["single_agent"]["budget_sweep"]})
    params = {SELF_CONSISTENCY: sweep, BEST_OF_N_SELFCHECK: sweep}
    samples = max(sweep)

    endpoint = endpoint_from_config(model, concurrency=args.concurrency)
    cache = RunCache(runs_dir / exp / "cache.sqlite", exp=exp)
    # Constructing the Executor binds the experiment, so pi_single cannot run
    # against a cache built with another temperature, token cap or template set.
    Executor(endpoint, cache, max_tokens_per_node=model["max_tokens_per_node"],
             temperature=model["temperature"],
             train_repeats=config["tensor"]["repeats"]["train"], runs_dir=runs_dir)
    templates = load_templates()
    cache.bind_experiment({"selfcheck": templates["selfcheck.v1"], "samples": samples,
                           "budget_sweep": sweep, "refine_rounds": [1, 2, 3],
                           "seed": args.seed}, scope="pi_single")

    headline = config["single_agent"].get("headline_budget", "s0_token_median")
    if args.budget is not None:
        budget, source = float(args.budget), "--budget"
    elif isinstance(headline, (int, float)):
        budget, source = float(headline), "config single_agent.headline_budget"
    else:
        budget = s0_token_median(cache.export(exp))
        source = "S0 token median on audit"
        if budget is None:
            raise SystemExit("no audit cells in the cache to take the S0 token median "
                             "from; build the tensor first or pass --budget")

    tasks = load_split_tasks(config, ["val", "test"])
    if args.limit_tasks:
        tasks = {role: items[:args.limit_tasks] for role, items in tasks.items()}
    runner = SingleAgentRunner(endpoint, cache, templates, exp=exp,
                               max_tokens=model["max_tokens_per_node"],
                               temperature=model["temperature"], samples=samples,
                               output_dir=runs_dir)
    result = select_on_val_report_on_test(
        runner, tasks["val"], tasks["test"], budget, seed=args.seed, params=params,
        concurrency=args.concurrency or model["max_concurrency"])
    result.update({"exp": exp, "model": model["name"],
                   "budget": {"tokens": budget, "source": source}})

    out = runs_dir / exp / "pi_single" / "frontier.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    h = result["headline"]
    print(f"wrote {out}")
    print(f"  budget {budget:.0f} tokens ({source})")
    print(f"  selected on val: {h['strategy']}@{h['param']} "
          f"(val {h['val_accuracy']:.3f}, {h['val_mean_tokens']:.0f} tokens)")
    print(f"  test: {h['accuracy']:.3f} on {h['n_tasks']} tasks, "
          f"truncation {h['truncation_rate']:.1%}, parse fail {h['parse_fail_rate']:.1%}")


if __name__ == "__main__":
    main()
