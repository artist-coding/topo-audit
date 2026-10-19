"""P1c: accuracy and cost as a function of reasoning effort.

Thinking turned out to be controllable after all: `reasoning_effort="none"` and
`chat_template_kwargs={"enable_thinking": False}` both switch it off, and the
gateway's own error message reveals effort also takes low / high / xhigh / max or an
integer in [1, 10]. So effort is a dial, not a nuisance, which changes three things:

- MATH Level 4-5 was ruled out at p = 0.917, but that was *with* thinking. Without
  it the same split may land inside the [0.35, 0.85] gate, in which case the main
  cell never needed to move to DeepMath.
- Generation length, and therefore the tensor's wall clock, drops with effort.
- "Does topology add anything a reasoning model has not already internalised?" stops
  being a confounded cross-model comparison and becomes one dial on one model.

Usage (gateway, proxy OFF):
    python3 scripts/p1_thinking_axis.py --n 50
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from data.loaders.deepmath import load_band
from data.loaders.math import PROMPT_TEMPLATE, load_pool, score_output

BASE = "http://aimaster-llm-gateway-service.aimaster-web:8080/llm/v1"
OUT = pathlib.Path("runs/probe/p1_thinking_axis.json")
SPLITS = pathlib.Path("data/splits/math_v1.json")

for var in ("http_proxy", "https_proxy", "all_proxy",
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(var, None)

#: label -> request fields. "none" is accepted even though the gateway's validation
#: message omits it; `chat_template_kwargs` is the vLLM-level equivalent.
EFFORTS: dict[str, dict] = {
    "off": {"reasoning_effort": "none"},
    "low": {"reasoning_effort": "low"},
    "high": {"reasoning_effort": "high"},
    "default": {},
}


def call(model: str, prompt: str, cap: int, extra: dict, attempts: int = 3) -> dict:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": cap, "temperature": 0.7, "seed": 0,
                       **extra}).encode()
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(
            BASE + "/chat/completions", data=body,
            headers={"Authorization": "Bearer " + os.environ["LLM_GATEWAY_API_KEY"],
                     "Content-Type": "application/json"})
        started = time.monotonic()
        try:
            payload = json.load(urllib.request.urlopen(request, timeout=900))
            choice, usage = payload["choices"][0], payload["usage"]
            return {"text": choice["message"].get("content") or "",
                    "finish": choice.get("finish_reason"),
                    "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"],
                    "reasoning_tokens": (usage.get("completion_tokens_details") or {}
                                         ).get("reasoning_tokens"),
                    "latency_s": round(time.monotonic() - started, 2)}
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(2 ** attempt)
    return {"error": last[:160]}


def pools(n: int) -> dict[str, pd.DataFrame]:
    """MATH L4-5 held out of every split, plus two DeepMath bands if present."""
    assigned = {tid for ids in json.loads(SPLITS.read_text())["splits"].values()
                for tid in ids}
    rng = np.random.default_rng(31337)
    out: dict[str, pd.DataFrame] = {}

    math = load_pool("data/raw/math_hendrycks.parquet", "train", (4, 5))
    held = math[~math["task_id"].isin(assigned)].reset_index(drop=True)
    picked = held.iloc[rng.permutation(len(held))[:n]]
    out["MATH L4-5"] = picked.rename(columns={"answer_str": "answer"})[
        ["task_id", "problem", "answer"]]

    deepmath = pathlib.Path("data/raw/deepmath.parquet")
    if deepmath.exists():
        for low in (7.0, 8.5):
            band = load_band(deepmath, low)
            sample = band.iloc[rng.permutation(len(band))[:n]]
            out[f"DeepMath {low}"] = sample.rename(
                columns={"question": "problem", "final_answer": "answer"})[
                ["task_id", "problem", "answer"]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--cap", type=int, default=16384)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--model", default="DeepSeek")
    ap.add_argument("--efforts", nargs="*", default=list(EFFORTS))
    args = ap.parse_args()

    datasets = pools(args.n)
    jobs = [(name, effort, row)
            for name, frame in datasets.items()
            for effort in args.efforts
            for row in frame.itertuples()]
    print(f"{len(jobs)} calls: {len(datasets)} pools x {len(args.efforts)} efforts "
          f"x {args.n} problems", flush=True)

    def work(job):
        name, effort, row = job
        res = call(args.model, PROMPT_TEMPLATE.format(problem=row.problem.strip()),
                   args.cap, EFFORTS[effort])
        base = {"pool": name, "effort": effort, "task_id": row.task_id}
        if "error" in res:
            return {**base, "error": res["error"]}
        scored = score_output(res["text"], str(row.answer))
        return {**base, "score": scored.score, "parse_fail": scored.parse_fail,
                **{k: v for k, v in res.items() if k != "text"}}

    # Written incrementally: an earlier run stalled at 450/600 and lost everything,
    # because results were only serialised at the end.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with cf.ThreadPoolExecutor(args.concurrency) as ex:
        for i, row in enumerate(ex.map(work, jobs), 1):
            rows.append(row)
            if i % 25 == 0 or i == len(jobs):
                OUT.write_text(json.dumps(
                    {"model": args.model, "cap": args.cap, "complete": i == len(jobs),
                     "rows": rows}, indent=1))
                print(f"  {i}/{len(jobs)}", flush=True)

    print(f"\n{'pool':14s} {'effort':8s} {'n':>4s} {'err':>4s} {'p':>6s} {'95% CI':>13s} "
          f"{'gen_med':>8s} {'reason_med':>10s} {'lat_med':>8s}  gate")
    for name in datasets:
        for effort in args.efforts:
            rs = [r for r in rows if r["pool"] == name and r["effort"] == effort]
            ok = [r for r in rs if "error" not in r]
            if not ok:
                continue
            scores = np.array([r["score"] for r in ok])
            gen = np.array([r["completion_tokens"] for r in ok])
            reason = np.array([r["reasoning_tokens"] or 0 for r in ok])
            lat = np.array([r["latency_s"] for r in ok])
            p = float(scores.mean())
            half = 1.96 * float(np.sqrt(max(p * (1 - p), 1e-9) / len(ok)))
            gate = "PASS" if 0.35 <= p <= 0.85 else "fail"
            print(f"{name:14s} {effort:8s} {len(ok):4d} {len(rs)-len(ok):4d} {p:6.3f} "
                  + f"[{max(0,p-half):.2f},{min(1,p+half):.2f}]".rjust(13)
                  + f" {np.median(gen):8.0f} {np.median(reason):10.0f} "
                    f"{np.median(lat):8.1f}  {gate}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
