"""P1: single-agent headroom and the generation-length distribution.

Answers two questions with one sweep, on problems held out of all four roles:

1. **Headroom.** Single-agent accuracy p per model. The pre-experiment gate keeps
   only model x task cells with p in [0.35, 0.85] — below that the task is too hard to
   show topology differences, above it there is nothing left to win.
2. **The cap.** `max_tokens_per_node` is currently a guess. Running at a deliberately
   generous cap and recording the realised completion length lets us read off the
   truncation rate *any* smaller cap would have produced, instead of re-probing per cap.

The probe slice is drawn from the ~3,100 Level 4-5 training-pool problems that
`make_splits.py` did not assign to train/audit/val/test, so nothing here peeks at a
split that later decisions depend on.

Usage (needs the gateway, so the proxy must be OFF):
    python3 scripts/p1_headroom.py --n 60 --cap 16384
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from data.loaders.math import PROMPT_TEMPLATE, load_pool, score_output  # noqa: E402

BASE = "http://aimaster-llm-gateway-service.aimaster-web:8080/llm/v1"
MODELS = ["DeepSeek", "GLM-5.3-Flash", "gpt-oss-120b", "Laguna-S-2.1"]
RAW = pathlib.Path("data/raw/math_hendrycks.parquet")
SPLITS = pathlib.Path("data/splits/math_v1.json")
OUT = pathlib.Path("runs/probe/p1_headroom.json")

for var in ("http_proxy", "https_proxy", "all_proxy",
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(var, None)


def call(model: str, prompt: str, cap: int, seed: int, attempts: int = 3) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": cap,
        "temperature": 0.7,
        "seed": seed,
    }).encode()
    last = ""
    for attempt in range(1, attempts + 1):
        req = urllib.request.Request(
            BASE + "/chat/completions", data=body,
            headers={"Authorization": "Bearer " + os.environ["LLM_GATEWAY_API_KEY"],
                     "Content-Type": "application/json"})
        start = time.monotonic()
        try:
            payload = json.load(urllib.request.urlopen(req, timeout=1200))
            choice, usage = payload["choices"][0], payload["usage"]
            return {
                "text": choice["message"].get("content") or "",
                "finish": choice.get("finish_reason"),
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "reasoning_tokens": (usage.get("completion_tokens_details") or {}
                                     ).get("reasoning_tokens"),
                "attempts": attempt,
                "latency_s": round(time.monotonic() - start, 2),
            }
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(2 ** attempt)
    return {"error": last[:200]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--cap", type=int, default=16384)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--models", nargs="*", default=MODELS)
    args = ap.parse_args()

    assigned = {tid for ids in json.loads(SPLITS.read_text())["splits"].values()
                for tid in ids}
    pool = load_pool(RAW, "train", (4, 5))
    held_out = pool[~pool["task_id"].isin(assigned)].reset_index(drop=True)
    rng = np.random.default_rng(777)
    probe = held_out.iloc[rng.permutation(len(held_out))[:args.n]]
    print(f"probe slice: {len(probe)} problems held out of all splits "
          f"(pool {len(pool)}, assigned {len(assigned)})")

    jobs = [(m, row) for m in args.models for row in probe.itertuples()]

    def work(job):
        model, row = job
        res = call(model, PROMPT_TEMPLATE.format(problem=row.problem.strip()),
                   args.cap, seed=0)
        if "error" in res:
            return {"model": model, "task_id": row.task_id, "level": int(row.level_int),
                    "error": res["error"]}
        scored = score_output(res["text"], str(row.answer_str))
        return {"model": model, "task_id": row.task_id, "level": int(row.level_int),
                "score": scored.score, "parse_fail": scored.parse_fail,
                "prediction": scored.prediction, **{k: v for k, v in res.items()
                                                    if k != "text"}}

    rows: list[dict] = []
    with cf.ThreadPoolExecutor(args.concurrency) as pool_ex:
        for i, row in enumerate(pool_ex.map(work, jobs), 1):
            rows.append(row)
            if i % 20 == 0:
                print(f"  {i}/{len(jobs)} done", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"cap": args.cap, "n": args.n, "rows": rows}, indent=1))

    print(f"\n{'model':16s} {'n':>4s} {'err':>4s} {'p':>6s} {'parse_fail':>10s} "
          f"{'gen_med':>8s} {'gen_p90':>8s} {'gen_max':>8s}  truncation rate if cap were")
    caps = [1024, 2048, 4096, 8192, 16384]
    print(" " * 72 + "  " + "  ".join(f"{c:>6d}" for c in caps))
    for model in args.models:
        rs = [r for r in rows if r["model"] == model]
        ok = [r for r in rs if "error" not in r]
        if not ok:
            print(f"{model:16s} {len(rs):4d} {len(rs):4d}  (all failed)")
            continue
        gen = np.array([r["completion_tokens"] for r in ok])
        p = float(np.mean([r["score"] for r in ok]))
        pf = float(np.mean([r["parse_fail"] for r in ok]))
        # A run that realised g tokens would have been cut off by any cap <= g.
        rates = ["{:6.0%}".format(float(np.mean(gen >= c))) for c in caps]
        gate = "PASS" if 0.35 <= p <= 0.85 else "FAIL"
        print(f"{model:16s} {len(ok):4d} {len(rs)-len(ok):4d} {p:6.3f} {pf:10.3f} "
              f"{np.median(gen):8.0f} {np.percentile(gen,90):8.0f} {gen.max():8.0f}  "
              + "  ".join(rates) + f"   gate={gate}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
