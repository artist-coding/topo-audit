"""P1b: find the DeepMath difficulty band where single-agent accuracy lands in the gate.

MATH Level 4-5 is saturated for 2026 reasoning models (p = 0.91-0.93, see
runs/probe/p1_headroom.json), so the audit space has no room to show a capacity term.
DeepMath-103K carries a continuous `difficulty` field, which turns difficulty into a
dial: sweep bands, keep the one where p falls inside [0.35, 0.85].

Also re-measures the generation-length distribution, since harder problems think
longer and `max_tokens_per_node` was fixed at 8192 on the easier MATH slice.

Usage (gateway, so the proxy must be OFF):
    python3 scripts/p1_deepmath_bands.py --per-band 40 --models DeepSeek
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from data.loaders.math import PROMPT_TEMPLATE, score_output, task_id_for  # noqa: E402
from scripts.p1_headroom import call  # noqa: E402

RAW = pathlib.Path("data/raw/deepmath.parquet")
OUT = pathlib.Path("runs/probe/p1_deepmath_bands.json")
BANDS = [(5.0, 5.5), (6.0, 6.5), (7.0, 7.5), (8.0, 8.5), (9.0, 10.0)]

for var in ("http_proxy", "https_proxy", "all_proxy",
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(var, None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-band", type=int, default=40)
    ap.add_argument("--cap", type=int, default=16384)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--models", nargs="*", default=["DeepSeek"])
    ap.add_argument("--bands", type=float, nargs="*", default=None,
                    help="band lower bounds; each spans +0.5 (e.g. 8.5 -> [8.5, 9.0])")
    args = ap.parse_args()

    pool = pd.read_parquet(RAW)
    pool["task_id"] = pool["question"].map(task_id_for)
    pool = pool.drop_duplicates(subset="task_id")
    rng = np.random.default_rng(20261014)

    bands = ([(low, low + 0.5) for low in args.bands] if args.bands else BANDS)
    jobs = []
    for low, high in bands:
        band = pool[(pool["difficulty"] >= low) & (pool["difficulty"] <= high)]
        if len(band) < args.per_band:
            print(f"  band [{low}, {high}] has only {len(band)} problems, skipping")
            continue
        picked = band.iloc[rng.permutation(len(band))[:args.per_band]]
        for model in args.models:
            jobs += [(model, low, row) for row in picked.itertuples()]
    print(f"{len(jobs)} calls: {len(bands)} bands x {args.per_band} problems "
          f"x {len(args.models)} model(s)", flush=True)

    def work(job):
        model, low, row = job
        res = call(model, PROMPT_TEMPLATE.format(problem=row.question.strip()),
                   args.cap, seed=0)
        base = {"model": model, "band": low, "task_id": row.task_id,
                "difficulty": float(row.difficulty), "topic": row.topic}
        if "error" in res:
            return {**base, "error": res["error"]}
        scored = score_output(res["text"], str(row.final_answer))
        return {**base, "score": scored.score, "parse_fail": scored.parse_fail,
                "prediction": scored.prediction,
                **{k: v for k, v in res.items() if k != "text"}}

    rows: list[dict] = []
    with cf.ThreadPoolExecutor(args.concurrency) as ex:
        for i, row in enumerate(ex.map(work, jobs), 1):
            rows.append(row)
            if i % 20 == 0:
                print(f"  {i}/{len(jobs)} done", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"cap": args.cap, "rows": rows}, indent=1))

    caps = [4096, 8192, 16384]
    print(f"\n{'model':14s} {'band':>6s} {'n':>4s} {'err':>4s} {'p':>6s} {'95% CI':>14s} "
          f"{'pfail':>6s} {'gen_med':>8s} {'gen_p90':>8s} "
          + " ".join(f"tr@{c}" for c in caps) + "  gate")
    for model in args.models:
        for low, _ in bands:
            rs = [r for r in rows if r["model"] == model and r["band"] == low]
            ok = [r for r in rs if "error" not in r]
            if not ok:
                continue
            scores = np.array([r["score"] for r in ok])
            gen = np.array([r["completion_tokens"] for r in ok])
            p = float(scores.mean())
            half = 1.96 * float(np.sqrt(max(p * (1 - p), 1e-9) / len(ok)))
            tr = " ".join(f"{float(np.mean(gen >= c)):6.0%}" for c in caps)
            gate = "PASS" if 0.35 <= p <= 0.85 else "fail"
            print(f"{model:14s} {low:6.1f} {len(ok):4d} {len(rs)-len(ok):4d} {p:6.3f} "
                  f"[{max(0,p-half):.2f},{min(1,p+half):.2f}]".rjust(14)
                  + f" {np.mean([r['parse_fail'] for r in ok]):6.2f} "
                  f"{np.median(gen):8.0f} {np.percentile(gen,90):8.0f} {tr}  {gate}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
