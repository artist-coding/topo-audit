"""P0: the gateway's usable concurrency, which sets the tensor's wall clock.

The score tensor is ~368k node calls. At the measured ~150 tok/s per stream the
total wall clock is (calls x mean_latency / concurrency), so concurrency is the only
lever that moves a 29-day run to a 2-day one. This ramps it until throughput stops
rising or errors start, and reports the knee.

Deliberately short prompts: this measures how many streams the gateway will serve at
once, not how fast one stream generates. Thinking cannot be disabled, so even a
trivial prompt costs ~30 reasoning tokens, which is enough to hold a slot open.

Usage (gateway, so the proxy must be OFF):
    python3 scripts/p0_concurrency.py --model DeepSeek
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import time

import numpy as np

BASE = "http://aimaster-llm-gateway-service.aimaster-web:8080/llm/v1"
OUT = pathlib.Path("runs/probe/p0_concurrency.json")

for var in ("http_proxy", "https_proxy", "all_proxy",
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(var, None)

PROMPT = "What is 12 + 7? Reply with only the number."


def one(model: str, cap: int) -> dict:
    import urllib.error
    import urllib.request

    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": PROMPT}],
                       "max_tokens": cap, "temperature": 0.7}).encode()
    request = urllib.request.Request(
        BASE + "/chat/completions", data=body,
        headers={"Authorization": "Bearer " + os.environ["LLM_GATEWAY_API_KEY"],
                 "Content-Type": "application/json"})
    started = time.monotonic()
    try:
        payload = json.load(urllib.request.urlopen(request, timeout=300))
        return {"ok": True, "latency_s": time.monotonic() - started,
                "completion_tokens": payload["usage"]["completion_tokens"]}
    except urllib.error.HTTPError as exc:
        return {"ok": False, "latency_s": time.monotonic() - started,
                "status": exc.code}
    except Exception as exc:
        return {"ok": False, "latency_s": time.monotonic() - started,
                "status": type(exc).__name__}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="DeepSeek")
    ap.add_argument("--levels", type=int, nargs="*",
                    default=[8, 16, 32, 64, 128, 256])
    ap.add_argument("--calls-per-level", type=int, default=0,
                    help="default: 3x the concurrency level")
    ap.add_argument("--cap", type=int, default=256)
    args = ap.parse_args()

    results: dict[str, dict] = {}
    print(f"{'conc':>5s} {'calls':>6s} {'ok':>5s} {'err':>5s} {'wall_s':>8s} "
          f"{'calls/s':>8s} {'lat_med':>8s} {'lat_p90':>8s}  statuses")
    for level in args.levels:
        n_calls = args.calls_per_level or level * 3
        started = time.monotonic()
        with cf.ThreadPoolExecutor(level) as ex:
            rows = list(ex.map(lambda _: one(args.model, args.cap), range(n_calls)))
        wall = time.monotonic() - started
        ok = [r for r in rows if r["ok"]]
        bad = [r for r in rows if not r["ok"]]
        lat = np.array([r["latency_s"] for r in ok]) if ok else np.array([np.nan])
        statuses = {}
        for row in bad:
            statuses[str(row["status"])] = statuses.get(str(row["status"]), 0) + 1
        results[str(level)] = {
            "n_calls": n_calls, "ok": len(ok), "errors": len(bad),
            "wall_s": wall, "throughput_calls_s": len(ok) / wall if wall else 0.0,
            "latency_median_s": float(np.nanmedian(lat)),
            "latency_p90_s": float(np.nanpercentile(lat, 90)),
            "error_statuses": statuses,
        }
        row = results[str(level)]
        print(f"{level:5d} {n_calls:6d} {len(ok):5d} {len(bad):5d} {wall:8.1f} "
              f"{row['throughput_calls_s']:8.2f} {row['latency_median_s']:8.1f} "
              f"{row['latency_p90_s']:8.1f}  {statuses or '-'}", flush=True)
        if len(bad) > 0.1 * n_calls:
            print(f"  >10% errors at concurrency {level}; stopping the ramp")
            break

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"model": args.model, "cap": args.cap,
                               "levels": results}, indent=1))

    # A rate-limited level rejects most calls in milliseconds, so ok/wall looks
    # excellent while almost nothing got served. Only clean levels can be the knee.
    clean = {level: row for level, row in results.items()
             if row["errors"] <= 0.02 * row["n_calls"]}
    best = max(clean.items(), key=lambda kv: kv[1]["throughput_calls_s"])
    rejected = {level: row["error_statuses"] for level, row in results.items()
                if row["errors"] > 0.02 * row["n_calls"]}
    if rejected:
        print(f"rate-limited levels (excluded from the knee): {rejected}")
    print(f"\nknee: concurrency {best[0]} at "
          f"{best[1]['throughput_calls_s']:.2f} calls/s")
    print("\nNOTE: these are trivial prompts, dominated by fixed per-call\n"
          "overhead, so this measures admission capacity, not streaming throughput.\n"
          "Project the tensor from a realistic-prompt benchmark instead:\n"
          "  python3 audit/run_audit.py --exp e0_smoke --limit N --concurrency 32\n")
    print("for reference only, assuming per-call latency carries over "
          "(382,320 node calls):")
    bands = json.loads(pathlib.Path("runs/probe/p1_deepmath_bands.json").read_text())
    rows = [r for r in bands["rows"] if "error" not in r]
    for band in sorted({r["band"] for r in rows}):
        mean_lat = float(np.mean([r["latency_s"] for r in rows if r["band"] == band]))
        days = 368_000 * mean_lat / int(best[0]) / 86_400
        print(f"  band {band:4.1f}: {days:6.1f} days")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
