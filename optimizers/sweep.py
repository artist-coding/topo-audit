"""Replay every optimizer x seed against the finished tensor.

    python3 optimizers/sweep.py --config configs/e0.yaml

This is the step the hard rules say must be replay-only: once the tensor exists, the
sweep may read cells but never commission a generation. With ``budget.replay_only``
set (the default in configs/e0.yaml) the endpoint is a ReplayOnlyEndpoint, which needs
no API key and cannot reach the network, and the Executor raises CacheMiss on the
first missing cell -- naming it -- instead of quietly paying for it.

Each (optimizer, seed) gets a fresh Executor, because the repeat allocator's hit
counters must start at zero for every search; trajectories are rewritten from the
start on re-run, which is free because every evaluation is a cache hit.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Sequence

if __package__ in (None, ""):  # run as a script: make the repo root importable
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from data.loaders.math import Task  # noqa: E402
from optimizers import build_optimizer  # noqa: E402
from optimizers.base import Budget, EdgeSettings  # noqa: E402
from optimizers.runner import run_optimizer  # noqa: E402
from runtime.cache import RunCache  # noqa: E402
from runtime.executor import CacheMiss, Executor  # noqa: E402
from space.s0 import S0_ROLES, enumerate_space  # noqa: E402


def run_sweep(config: dict, endpoint, cache: RunCache, train_tasks: Sequence[Task], *,
              optimizers: Sequence[str], seeds: Sequence[int], settings: EdgeSettings,
              runs_dir: pathlib.Path, split_path: str | pathlib.Path,
              replay_only: bool) -> dict:
    """Run the sweep and return its manifest. Raises CacheMiss on the first hole."""
    model = config["model"]
    budget_cfg = config["budget"]
    space = enumerate_space(S0_ROLES, model=model["name"])
    manifest: dict = {"exp": cache.exp, "replay_only": replay_only, "runs": {}}
    started = time.monotonic()
    for name in optimizers:
        for seed in seeds:
            executor = Executor(
                endpoint, cache, max_tokens_per_node=model["max_tokens_per_node"],
                temperature=model["temperature"],
                train_repeats=config["tensor"]["repeats"]["train"],
                runs_dir=runs_dir, replay_only=replay_only)
            optimizer = build_optimizer(
                name, space, Budget(budget_cfg["max_evals"], budget_cfg["tasks_per_eval"]),
                seed, settings)
            path = run_optimizer(optimizer, executor, train_tasks, runs_dir=runs_dir,
                                 split_path=split_path)
            manifest["runs"].setdefault(name, {})[str(seed)] = {
                "trajectory": str(path),
                "rounds": sum(1 for line in path.read_text().splitlines() if line.strip()),
                "cum_evals": optimizer.budget.spent_evals,
                "cum_tokens": optimizer.budget.spent_tokens,
            }
    manifest["cache"] = cache.stats()
    manifest["wall_s"] = time.monotonic() - started
    return manifest


def main() -> None:
    import yaml

    from data.tasks import load_split_tasks
    from runtime.endpoints import endpoint_from_config

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/e0.yaml")
    ap.add_argument("--exp", default=None)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--optimizers", default=None, help="comma list; default from config")
    ap.add_argument("--seeds", default=None, help="comma list; default from config")
    ap.add_argument("--allow-generation", action="store_true",
                    help="override budget.replay_only (development against a mock only)")
    args = ap.parse_args()

    config = yaml.safe_load(pathlib.Path(args.config).read_text())
    exp = args.exp or config["exp"]
    runs_dir = pathlib.Path(args.runs_dir)
    replay_only = bool(config["budget"].get("replay_only", True)) and not args.allow_generation
    optimizers = args.optimizers.split(",") if args.optimizers else config["optimizers"]
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds else config["seeds"])

    endpoint = endpoint_from_config(config["model"], replay_only=replay_only)
    cache = RunCache(runs_dir / exp / "cache.sqlite", exp=exp)
    train = load_split_tasks(config, ["train"])["train"]
    try:
        manifest = run_sweep(config, endpoint, cache, train, optimizers=optimizers,
                             seeds=seeds, settings=EdgeSettings.from_config(args.config),
                             runs_dir=runs_dir, split_path=config["task"]["splits"],
                             replay_only=replay_only)
    except CacheMiss as miss:
        raise SystemExit(f"{miss}\nThe tensor is incomplete: finish audit/run_audit.py "
                         "for the train split before sweeping.") from miss
    out = runs_dir / exp / "sweep_manifest.json"
    out.write_text(json.dumps(manifest, indent=1))
    total = sum(len(v) for v in manifest["runs"].values())
    print(f"wrote {out}: {total} trajectories, cache {manifest['cache']}, "
          f"replay_only={replay_only}")


if __name__ == "__main__":
    main()
