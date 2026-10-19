"""Build the four-role task split and commit it.

Four roles, zero overlap, fixed seed, byte-identical on re-run:

    train  optimizer candidate evaluation; the only split it may ever see
    audit  full-space evaluation and cross-fitted selection of G_star
    val    learning curves and the plateau test
    test   final numbers only; no argmax, ranking or thresholding ever runs on it

train/audit/val are drawn from the MATH *training* split (Level 4-5, 3,994 problems).
test is drawn from the MATH *test* split (Level 4-5, 2,538 problems) and always
contains the whole MATH-500 Level 4-5 subset (262 problems) so results stay
comparable with the community number; the rest is topped up at random to widen the
confidence intervals on C and E, which the 262-problem subset alone cannot carry.

DeepMath-103K (the pending main cell) has a single source split, so all four roles are
drawn from one difficulty band, disjoint, with the P2 probe problems optionally held
out so that the problems used to choose the cell never become test problems.

Usage:
    python3 data/make_splits.py                       # MATH Level 4-5 (math_v1.json)
    python3 data/make_splits.py --family deepmath --band 8.5 \
        --exclude-p2 runs/p2_deepmath_8.5/p2_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from data.loaders.deepmath import (  # noqa: E402
    ANSWER_FILTER, BAND_WIDTH, DEFAULT_RAW as DEEPMATH_RAW, answer_profile, load_band,
)
from data.loaders.math import load_pool, task_id_for  # noqa: E402

RAW = pathlib.Path("data/raw")
DEFAULT_OUT = pathlib.Path("data/splits/math_v1.json")


def build(
    seed: int,
    n_train: int,
    n_audit: int,
    n_val: int,
    n_test: int,
    levels: tuple[int, ...] = (4, 5),
) -> dict:
    train_pool = load_pool(RAW / "math_hendrycks.parquet", "train", levels)
    test_pool = load_pool(RAW / "math_hendrycks.parquet", "test", levels)
    m500 = load_pool(RAW / "math500.parquet", None, levels)

    # MATH-500 is a subset of the MATH test split, so match it back by task_id.
    m500_ids = set(m500["task_id"])
    anchored = test_pool[test_pool["task_id"].isin(m500_ids)]
    rest = test_pool[~test_pool["task_id"].isin(m500_ids)]

    need = n_train + n_audit + n_val
    if len(train_pool) < need:
        raise SystemExit(f"train pool has {len(train_pool)} < {need} problems")
    if n_test < len(anchored):
        raise SystemExit(
            f"n_test={n_test} is smaller than the MATH-500 anchor ({len(anchored)}); "
            "the anchor must be contained in test")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_pool))
    picked = train_pool.iloc[order[:need]]
    train = picked.iloc[:n_train]
    audit = picked.iloc[n_train:n_train + n_audit]
    val = picked.iloc[n_train + n_audit:]

    top_up = n_test - len(anchored)
    rest_order = rng.permutation(len(rest))
    test = pd.concat([anchored, rest.iloc[rest_order[:top_up]]], ignore_index=True)

    roles = {"train": train, "audit": audit, "val": val, "test": test}
    ids = {role: list(df["task_id"]) for role, df in roles.items()}

    # Zero overlap across roles, and no duplicate problem text anywhere.
    seen: dict[str, str] = {}
    for role, id_list in ids.items():
        for tid in id_list:
            if tid in seen:
                raise SystemExit(f"task {tid} appears in both {seen[tid]} and {role}")
            seen[tid] = role

    payload = {
        "family": "math",
        "seed": seed,
        "levels": list(levels),
        "source": {
            "train_audit_val": "EleutherAI/hendrycks_math train split",
            "test": "EleutherAI/hendrycks_math test split, anchored on HuggingFaceH4/MATH-500",
        },
        "sizes": {role: len(v) for role, v in ids.items()},
        "test_anchor": {
            "name": "MATH-500 Level 4-5",
            "n": len(anchored),
            "task_ids": sorted(anchored["task_id"]),
        },
        "pool_sizes": {
            "train_pool": len(train_pool),
            "test_pool": len(test_pool),
            "math500_at_levels": len(m500),
        },
        "splits": {role: sorted(v) for role, v in ids.items()},
        "answers": {
            tid: str(ans)
            for df in roles.values()
            for tid, ans in zip(df["task_id"], df["answer_str"])
        },
        "levels_by_task": {
            tid: int(lvl)
            for df in roles.values()
            for tid, lvl in zip(df["task_id"], df["level_int"])
        },
    }
    blob = json.dumps(
        {k: payload[k] for k in ("family", "seed", "levels", "sizes", "splits")},
        sort_keys=True)
    payload["fingerprint"] = hashlib.sha256(blob.encode()).hexdigest()[:16]
    return payload


def build_deepmath(seed: int, n_train: int, n_audit: int, n_val: int, n_test: int,
                   band: float, raw: str = DEEPMATH_RAW,
                   exclude: frozenset[str] = frozenset(), width: float = BAND_WIDTH) -> dict:
    """Four disjoint roles from one DeepMath band."""
    full = load_band(raw, band, width)
    pool = full[~full["task_id"].isin(exclude)].reset_index(drop=True)
    need = n_train + n_audit + n_val + n_test
    if len(pool) < need:
        raise SystemExit(f"band {band} has {len(pool)} usable problems < {need}; download "
                         "more shards (data/fetch_deepmath.py N) or shrink the roles")
    order = np.random.default_rng(seed).permutation(len(pool))
    picked = pool.iloc[order[:need]]
    bounds = np.cumsum([0, n_train, n_audit, n_val, n_test])
    roles = {role: picked.iloc[bounds[i]:bounds[i + 1]]
             for i, role in enumerate(("train", "audit", "val", "test"))}
    ids = {role: list(df["task_id"]) for role, df in roles.items()}
    payload = {
        "family": "deepmath",
        "answer_filter": ANSWER_FILTER,
        "seed": seed,
        "band": [band, band + width],
        "source": {"all_roles": "zwhe99/DeepMath-103K (single source split)"},
        "sizes": {role: len(v) for role, v in ids.items()},
        "pool_sizes": {"band_pool": len(full), "excluded_p2": len(full) - len(pool),
                       "usable": len(pool)},
        "splits": {role: sorted(v) for role, v in ids.items()},
        "answers": {tid: str(ans) for df in roles.values()
                    for tid, ans in zip(df["task_id"], df["answer_str"])},
        "difficulty_by_task": {tid: float(d) for df in roles.values()
                               for tid, d in zip(df["task_id"], df["difficulty"])},
    }
    blob = json.dumps({k: payload[k] for k in
                      ("family", "answer_filter", "seed", "band", "sizes", "splits")},
                      sort_keys=True)
    payload["fingerprint"] = hashlib.sha256(blob.encode()).hexdigest()[:16]
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", choices=("math", "deepmath"), default="math")
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--train", type=int, default=256)
    ap.add_argument("--audit", type=int, default=300)
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--band", type=float, default=None, help="DeepMath band lower edge")
    ap.add_argument("--band-width", type=float, default=BAND_WIDTH)
    ap.add_argument("--raw", default=DEEPMATH_RAW, help="DeepMath raw parquet snapshot")
    ap.add_argument("--exclude-p2", type=pathlib.Path, default=None,
                    help="p2_report.json whose probe problems must stay out of every role")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    if args.family == "deepmath":
        if args.band is None:
            raise SystemExit("--band is required for --family deepmath")
        print(json.dumps(answer_profile(args.raw, args.band, args.band_width), indent=2,
                         ensure_ascii=False), flush=True)
        exclude = frozenset(json.loads(args.exclude_p2.read_text())["task_ids"]
                            if args.exclude_p2 else ())
        payload = build_deepmath(args.seed, args.train, args.audit, args.val, args.test,
                                 args.band, raw=args.raw, exclude=exclude, width=args.band_width)
        suffix = "" if args.band_width == BAND_WIDTH else f"_w{args.band_width}"
        out = args.out or pathlib.Path(f"data/splits/deepmath_b{args.band}{suffix}_v1.json")
    else:
        payload = build(args.seed, args.train, args.audit, args.val, args.test)
        out = args.out or DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True))

    print(f"wrote {out}  fingerprint={payload['fingerprint']}")
    print(f"  sizes      {payload['sizes']}")
    print(f"  pools      {payload['pool_sizes']}")
    if args.family == "math":
        print(f"  test anchor {payload['test_anchor']['n']} MATH-500 L4-5 problems included")
        lv = pd.Series(payload["levels_by_task"]).value_counts().sort_index()
        print(f"  levels     {lv.to_dict()}")


if __name__ == "__main__":
    main()
