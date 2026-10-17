"""Experiment-scoped replay keyed by computation, never by labelled graph.

SQLite transactions protect records; per-key file locks also protect the work
between a miss and its write, including across processes. Experiments have separate
namespaces because temperature, token caps and frozen templates are not in exec_hash.
Unfinished/failed runs retain an accounting journal without becoming cached scores.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from runtime.executor import RunRecord

CacheKey = tuple[str, str, str, int, int]


def key_for(record: RunRecord) -> CacheKey:
    return record.model, record.exec_hash, record.task_id, record.seed, record.repeat


def _decode(blob: str) -> RunRecord:
    from runtime.executor import RunRecord

    values = json.loads(blob)
    values["truncated_nodes"] = tuple(values["truncated_nodes"])
    return RunRecord(**values)


class RunCache:
    def __init__(self, path: str | Path, *, exp: str):
        self.path = Path(path).resolve()
        self.exp = exp
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_dir = self.path.with_name(self.path.name + ".locks")
        self._lock_dir.mkdir(exist_ok=True)
        self._stats = {"hits": 0, "misses": 0, "writes": 0}
        self._stats_lock = threading.Lock()
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS experiments (
                    exp TEXT PRIMARY KEY, settings TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS records (
                    exp TEXT NOT NULL, model TEXT NOT NULL, exec_hash TEXT NOT NULL,
                    task_id TEXT NOT NULL, seed INTEGER NOT NULL, repeat INTEGER NOT NULL,
                    record TEXT NOT NULL,
                    PRIMARY KEY (exp, model, exec_hash, task_id, seed, repeat)
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    run_id TEXT PRIMARY KEY, exp TEXT NOT NULL, record TEXT NOT NULL
                );
            """)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def bind_experiment(self, settings: dict, *, scope: str = "") -> None:
        """Reject silent reuse when settings change; choose a new exp instead.

        `scope` binds an additional settings group under the same experiment (e.g. the
        pi_single self-check template and sample count) without altering the main
        blob, so existing caches stay valid.
        """
        key = f"{self.exp}::{scope}" if scope else self.exp
        blob = json.dumps(settings, sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO experiments VALUES (?, ?)", (key, blob))
            saved, = connection.execute(
                "SELECT settings FROM experiments WHERE exp = ?", (key,)).fetchone()
            if saved != blob:
                raise ValueError(f"experiment settings changed ({key}); use a new exp")

    @contextmanager
    def claim(self, key: CacheKey) -> Iterator[None]:
        """A completed run wins the key before another worker can call the endpoint."""
        digest = hashlib.sha256(json.dumps((self.exp, key)).encode()).hexdigest()
        with (self._lock_dir / digest).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def get(self, key: CacheKey) -> RunRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT record FROM records WHERE exp = ? AND model = ? AND exec_hash = ?"
                " AND task_id = ? AND seed = ? AND repeat = ?", (self.exp, *key),
            ).fetchone()
        with self._stats_lock:
            self._stats["hits" if row else "misses"] += 1
        return _decode(row[0]) if row else None

    def put(self, record: RunRecord) -> None:
        if record.exp != self.exp:
            raise ValueError("record belongs to a different experiment")
        blob = json.dumps(asdict(record), sort_keys=True, allow_nan=False)
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO records VALUES (?, ?, ?, ?, ?, ?, ?)",
                (self.exp, *key_for(record), blob))
            inserted = bool(cursor.rowcount)
            if not inserted:
                saved, = connection.execute(
                    "SELECT record FROM records WHERE exp = ? AND model = ? AND exec_hash = ?"
                    " AND task_id = ? AND seed = ? AND repeat = ?",
                    (self.exp, *key_for(record))).fetchone()
                if saved != blob:
                    raise ValueError("refusing to overwrite a cached sample")
            connection.execute("DELETE FROM attempts WHERE run_id = ?", (record.run_id,))
        if inserted:
            with self._stats_lock:
                self._stats["writes"] += 1

    def record_attempt(self, record: RunRecord) -> None:
        """Durable unscored journal; only put() promotes a run to replayable results."""
        if record.exp != self.exp:
            raise ValueError("record belongs to a different experiment")
        with self._connect() as connection:
            connection.execute("INSERT OR REPLACE INTO attempts VALUES (?, ?, ?)",
                               (record.run_id, record.exp,
                                json.dumps(asdict(record), sort_keys=True, allow_nan=False)))

    def stats(self) -> dict:
        with self._stats_lock:
            return dict(self._stats)

    def export(self, exp: str) -> Iterator[RunRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT record FROM records WHERE exp = ?"
                " ORDER BY model, exec_hash, task_id, seed, repeat", (exp,)).fetchall()
        yield from (_decode(row[0]) for row in rows)

    def export_attempts(self, exp: str) -> Iterator[RunRecord]:
        """Failed or interrupted accounting, kept separate from optimizer scores."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT record FROM attempts WHERE exp = ? ORDER BY run_id", (exp,)
            ).fetchall()
        yield from (_decode(row[0]) for row in rows)
