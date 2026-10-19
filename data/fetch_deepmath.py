"""Download DeepMath-103K shards and keep only the columns we need.

MATH Level 4-5 turned out to be saturated for 2026 reasoning models (single-agent
accuracy 0.91-0.93, see runs/probe/p1_headroom.json), so the audit needs a task pool
with a difficulty dial. DeepMath-103K carries a continuous `difficulty` field, which
lets us pick the slice that puts single-agent accuracy inside the [0.35, 0.85] gate
instead of hoping a fixed benchmark lands there.

Needs the clash proxy ON (HuggingFace), unlike the LLM gateway.
"""
from __future__ import annotations

import io
import pathlib
import sys
import urllib.request

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from data.loaders.deepmath import answer_profile

URL = ("https://huggingface.co/datasets/zwhe99/DeepMath-103K/resolve/main/"
       "data/train-{i:05d}-of-00010.parquet")
KEEP = ["question", "final_answer", "difficulty", "topic"]
OUT = pathlib.Path("data/raw/deepmath.parquet")


def main(n_shards: int = 3, band: float | None = None) -> None:
    frames = []
    for i in range(n_shards):
        url = URL.format(i=i)
        for attempt in range(1, 4):
            try:
                blob = urllib.request.urlopen(url, timeout=900).read()
                break
            except Exception as exc:
                print(f"  shard {i} attempt {attempt} failed: "
                      f"{type(exc).__name__}", flush=True)
                if attempt == 3:
                    raise
        df = pd.read_parquet(io.BytesIO(blob), columns=KEEP)
        frames.append(df)
        print(f"  shard {i}: {len(df)} rows", flush=True)

    allm = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    allm.to_parquet(OUT)
    print(f"\n{len(allm)} problems -> {OUT}")
    bands = allm["difficulty"].value_counts().sort_index()
    print("\ndifficulty distribution:")
    print(bands.to_string())
    print("\ncumulative at or above:")
    for t in sorted(allm["difficulty"].unique()):
        print(f"  >= {t:>4}: {int((allm['difficulty'] >= t).sum()):>6,}")
    if band is not None:
        import json

        print(json.dumps(answer_profile(OUT, band), indent=2, ensure_ascii=False))
    else:
        print("Before making splits, inspect the chosen band with: "
              "python3 -m data.loaders.deepmath --band <lower-bound>")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n_shards", type=int, nargs="?", default=3)
    parser.add_argument("--band", type=float, help="inspect this band's answer frequencies after download")
    args = parser.parse_args()
    main(args.n_shards, args.band)
