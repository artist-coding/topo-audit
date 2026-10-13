"""Download MATH (train/test, all 7 subjects) and MATH-500 into data/raw/.

Needs the clash proxy (the LLM gateway needs it *off*; HuggingFace needs it *on*).
Raw files are gitignored; data/splits/*.json is the committed artifact.
"""
from __future__ import annotations

import io
import json
import pathlib
import urllib.request

import pandas as pd

SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus"]
PARQUET = ("https://huggingface.co/api/datasets/EleutherAI/hendrycks_math/parquet"
           "/{subject}/{split}/0.parquet")
MATH500 = ("https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl")
RAW = pathlib.Path("data/raw")


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=180) as fh:
        return fh.read()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    frames = []
    for split in ("train", "test"):
        for subject in SUBJECTS:
            blob = get(PARQUET.format(subject=subject, split=split))
            df = pd.read_parquet(io.BytesIO(blob))
            df["subject"], df["source_split"] = subject, split
            frames.append(df)
            print(f"  {subject:28s} {split:5s} {len(df):5d} rows  cols={list(df.columns)[:5]}")
    allm = pd.concat(frames, ignore_index=True)
    allm.to_parquet(RAW / "math_hendrycks.parquet")
    print(f"MATH total {len(allm)} rows -> {RAW / 'math_hendrycks.parquet'}")
    print(allm.groupby(["source_split", "level"]).size().unstack(fill_value=0))

    rows = [json.loads(line) for line in get(MATH500).decode().splitlines() if line.strip()]
    m500 = pd.DataFrame(rows)
    m500.to_parquet(RAW / "math500.parquet")
    print(f"\nMATH-500 {len(m500)} rows  cols={list(m500.columns)}")
    if "level" in m500:
        print(m500["level"].value_counts().sort_index())


if __name__ == "__main__":
    main()
