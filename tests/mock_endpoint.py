"""Deterministic task-aware responses over the runtime's actual retry machinery.

Gold answers live in this fixture's registry, never in rendered role prompts.
Accuracy increases with node count plus small exec_hash jitter, so optimizers can
later be tested against a weak structural signal without accessing the gateway.
"""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Iterable

from data.loaders.math import Task
from runtime.endpoints import (
    EndpointHTTPError,
    MockEndpoint as TransportMock,
    current_call_context,
)


def _draw(*values: object) -> float:
    digest = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
    return int(digest[:13], 16) / 16 ** 13


class MockEndpoint(TransportMock):
    def __init__(self, tasks: Iterable[Task], *, name: str = "mock",
                 truncation_rate: float = 0.0, transient_error_rate: float = 0.0,
                 success_rate: float | None = None, itemize_reasoning: bool = True,
                 truncated_nodes: tuple[int, ...] = (), fail_first_attempts: int = 1,
                 max_concurrency: int = 4, max_attempts: int = 3):
        for rate in (truncation_rate, transient_error_rate, success_rate):
            if rate is not None and not 0 <= rate <= 1:
                raise ValueError("mock rates must be probabilities")
        self.tasks = {task.task_id: task for task in tasks}
        self.truncation_rate, self.transient_error_rate = truncation_rate, transient_error_rate
        self.success_rate, self.itemize_reasoning = success_rate, itemize_reasoning
        self.truncated_nodes, self.fail_first_attempts = truncated_nodes, fail_first_attempts
        self.calls: list[dict] = []
        self._call_lock = threading.Lock()
        self._attempts_by_request: dict[tuple, int] = {}
        super().__init__(name, responder=self._respond, max_concurrency=max_concurrency,
                         max_attempts=max_attempts, backoff_s=(0.0,) * (max_attempts - 1))

    @staticmethod
    def latent_accuracy(exec_hash: str, node_count: int) -> float:
        jitter = 0.04 * (_draw("accuracy", exec_hash) - 0.5)
        return min(0.95, 0.35 + 0.10 * node_count + jitter)

    @property
    def call_count(self) -> int:
        with self._call_lock:
            return len(self.calls)

    def _usage(self, prompt_tokens: int, completion_tokens: int, reasoning_tokens: int) -> dict:
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                 "total_tokens": prompt_tokens + completion_tokens}
        if self.itemize_reasoning:
            usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
        return usage

    def _respond(self, payload: dict) -> dict:
        context = current_call_context()
        task = self.tasks[context.task_id]
        identity = (context.exec_hash, context.task_id, context.seed, context.repeat,
                    context.node_index, payload["seed"])
        with self._call_lock:
            attempt = self._attempts_by_request.get(identity, 0) + 1
            self._attempts_by_request[identity] = attempt
        transient = (attempt <= self.fail_first_attempts
                     and _draw("transient", *identity) < self.transient_error_rate)
        if transient:
            # Some failed responses report spent tokens; those must survive retry.
            usage = self._usage(5, 3, 2)
            with self._call_lock:
                self.calls.append({"payload": payload, "context": context, "usage": usage,
                                   "status": 502, "text": None})
            raise EndpointHTTPError(502, usage)

        truncated = (context.node_index in self.truncated_nodes
                     or _draw("truncation", *identity) < self.truncation_rate)
        accuracy = (self.success_rate if self.success_rate is not None
                    else self.latent_accuracy(context.exec_hash, context.node_count))
        correct = _draw("success", *identity) < accuracy
        if correct:
            answer = task.answer
        else:
            try:
                answer = str(int(task.answer) + 1)
            except ValueError:
                answer = f"({task.answer}) + 1"
        text = "" if truncated else f"The computed answer is \\boxed{{{answer}}}.\nFINAL: {answer}"
        prompt_tokens = max(1, (len(payload["messages"][0]["content"].encode()) + 3) // 4)
        gen_tokens = (payload["max_tokens"] if truncated
                      else min(payload["max_tokens"], 7 + (len(text.encode()) + 3) // 4))
        reasoning_tokens = gen_tokens if truncated else min(7, gen_tokens)
        usage = self._usage(prompt_tokens, gen_tokens, reasoning_tokens)
        with self._call_lock:
            self.calls.append({"payload": payload, "context": context, "usage": usage,
                               "status": 200, "text": text})
        return {"choices": [{"message": {"role": "assistant", "content": text},
                             "finish_reason": "length" if truncated else "stop"}],
                "usage": usage}
