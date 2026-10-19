"""One completion per request, with explicit transport and retry accounting.

The gateway returns one sample per request (``n > 1`` is rejected). Thinking is on by
default; P1c found ``reasoning_effort`` controls it (none / low / high / xhigh / max).
Any such extra request field goes through ``request_options``, which the Executor
binds into the experiment identity, so a different effort can never share a cache
with the default. Proxy bypass is local to this client's opener: changing os.environ
would break concurrent data downloads. Raw usage is retained per attempt because a
failed request can still have consumed tokens, and missing reasoning details do not
mean zero reasoning.
"""

from __future__ import annotations

import ipaddress
import http.client
import json
import logging
import math
import os
import shlex
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Protocol


@dataclass(frozen=True, slots=True)
class CallResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int | None
    finish_reason: str
    truncated: bool
    raw_usage: dict
    attempts: int
    latency_s: float


class Endpoint(Protocol):
    name: str

    def complete(self, prompt: str, *, seed: int, max_tokens: int,
                 temperature: float) -> CallResult: ...


#: Fields every request sets itself; request_options may not override them.
RESERVED_FIELDS = frozenset({"model", "messages", "n", "seed", "max_tokens",
                             "temperature", "stream", "stream_options"})


def validate_request_options(options: Mapping[str, object] | None) -> dict:
    """Extra request fields (e.g. reasoning_effort), JSON-safe and non-overlapping."""
    options = dict(options or {})
    clash = RESERVED_FIELDS & set(options)
    if clash:
        raise ValueError(f"request_options may not set {sorted(clash)}")
    json.dumps(options, sort_keys=True, allow_nan=False)
    return options


@dataclass(frozen=True, slots=True)
class CallAttempt:
    number: int
    status_code: int | None
    error: str | None
    text: str | None
    finish_reason: str | None
    raw_usage: dict
    latency_s: float


@dataclass(frozen=True, slots=True)
class CallContext:
    """Local provenance for logging and mocks; never sent in an HTTP request."""

    exec_hash: str
    task_id: str
    node_count: int
    node_index: int
    seed: int
    repeat: int


_context: ContextVar[CallContext | None] = ContextVar("call_context", default=None)
_observer: ContextVar[Callable[[CallAttempt], None] | None] = ContextVar(
    "attempt_observer", default=None)
_log = logging.getLogger(__name__)


def current_call_context() -> CallContext:
    context = _context.get()
    if context is None:
        raise RuntimeError("this endpoint requires an Executor call context")
    return context


@contextmanager
def observe_call(context: CallContext, observer: Callable[[CallAttempt], None]
                 ) -> Iterator[None]:
    context_token = _context.set(context)
    observer_token = _observer.set(observer)
    try:
        yield
    finally:
        _observer.reset(observer_token)
        _context.reset(context_token)


def usage_totals(usages: Iterable[dict]) -> tuple[int, int, int | None]:
    """Unknown usage remains in the log as {}, never as fabricated token counts."""
    present = [usage for usage in usages if usage]
    prompt = sum(usage.get("prompt_tokens", 0) for usage in present)
    completion = sum(usage.get("completion_tokens", 0) for usage in present)
    reasoning = [
        (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        for usage in present
    ]
    known = bool(reasoning) and all(value is not None for value in reasoning)
    return prompt, completion, sum(reasoning) if known else None


class EndpointHTTPError(RuntimeError):
    """A status failure can carry billable usage even though it has no answer."""

    def __init__(self, status_code: int, raw_usage: dict | None = None):
        super().__init__(f"endpoint HTTP {status_code}")
        self.status_code = status_code
        self.raw_usage = raw_usage if raw_usage is not None else {}


class EndpointError(RuntimeError):
    def __init__(self, attempts: tuple[CallAttempt, ...]):
        super().__init__(f"endpoint failed after {len(attempts)} attempt(s)")
        self.attempts = attempts


class EndpointStreamError(OSError):
    """An incomplete HTTP-200 stream; preserve any usage received before failure."""

    def __init__(self, raw_usage: dict):
        super().__init__("incomplete or invalid completion stream")
        self.raw_usage = raw_usage


class _RetryingEndpoint:
    def __init__(self, name: str, *, max_concurrency: int, max_attempts: int,
                 backoff_s: Iterable[float], sleep: Callable[[float], None] = time.sleep,
                 request_options: Mapping[str, object] | None = None):
        self.name = name
        self.request_options = validate_request_options(request_options)
        self.max_attempts = max_attempts
        self.backoff_s = tuple(backoff_s)
        if max_concurrency < 1 or max_attempts < 1:
            raise ValueError("concurrency and attempts must be positive")
        if len(self.backoff_s) < max_attempts - 1 or any(t < 0 for t in self.backoff_s):
            raise ValueError("provide a nonnegative backoff for each retry")
        self._semaphore = threading.BoundedSemaphore(max_concurrency)
        self._sleep = sleep

    def _request(self, payload: dict) -> dict:
        raise NotImplementedError

    def complete(self, prompt: str, *, seed: int, max_tokens: int,
                 temperature: float) -> CallResult:
        if max_tokens < 1 or not math.isfinite(temperature) or temperature < 0:
            raise ValueError("max_tokens must be positive and temperature finite/nonnegative")
        payload = {
            "model": self.name, "messages": [{"role": "user", "content": prompt}],
            "n": 1, "seed": seed, "max_tokens": max_tokens,
            "temperature": temperature, **self.request_options,
        }
        started = time.monotonic()
        attempts: list[CallAttempt] = []
        for number in range(1, self.max_attempts + 1):
            attempt_started = time.monotonic()
            raw_usage: dict = {}
            text, finish_reason = None, None
            status_code = None
            failure: Exception | None = None
            retryable = False
            try:
                with self._semaphore:
                    response = self._request(payload)
                status_code = 200
                raw_usage = response.get("usage") or {}
                if not isinstance(raw_usage, dict):
                    raw_usage = {}
                    raise ValueError("usage must be an object")
                for field in ("prompt_tokens", "completion_tokens"):
                    value = raw_usage.get(field)
                    if type(value) is not int or value < 0:
                        raise ValueError("successful responses must report nonnegative token counts")
                choice, = response["choices"]
                content = choice["message"].get("content")
                text = "" if content is None else content
                finish_reason = choice["finish_reason"]
                if not isinstance(text, str) or not isinstance(finish_reason, str):
                    raise ValueError("invalid completion content or finish reason")
            except EndpointHTTPError as exc:
                failure = exc
                status_code, raw_usage = exc.status_code, exc.raw_usage
                retryable = status_code in (408, 429) or 500 <= status_code < 600
            except EndpointStreamError as exc:
                failure, retryable = exc, True
                status_code, raw_usage = 200, exc.raw_usage
            except (urllib.error.URLError, OSError) as exc:
                failure, retryable = exc, True
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                failure = exc

            attempt = CallAttempt(
                number, status_code, type(failure).__name__ if failure else None,
                text, finish_reason, raw_usage, time.monotonic() - attempt_started,
            )
            attempts.append(attempt)
            # Log only metadata: exception bodies can echo prompts or credentials.
            _log.log(logging.WARNING if failure else logging.INFO,
                     "endpoint=%s attempt=%s status=%s error=%s usage=%s",
                     self.name, number, status_code, attempt.error, raw_usage)
            observer = _observer.get()
            if observer is not None:
                observer(attempt)
            if failure is None:
                prompt_tokens, completion_tokens, reasoning_tokens = usage_totals(
                    item.raw_usage for item in attempts)
                return CallResult(
                    text=text, prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens, reasoning_tokens=reasoning_tokens,
                    finish_reason=finish_reason, truncated=finish_reason == "length",
                    raw_usage=raw_usage, attempts=number,
                    latency_s=time.monotonic() - started,
                )
            if not retryable or number == self.max_attempts:
                raise EndpointError(tuple(attempts)) from failure
            # Release the concurrency slot during backoff so other calls can progress.
            self._sleep(self.backoff_s[number - 1])
        raise AssertionError("unreachable")


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: urllib.request.Request, fp: object, code: int,
                         msg: str, headers: object, newurl: str) -> None:
        # A redirect must not move a gateway credential outside the allowlist.
        return None


def _validate_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    host = parsed.hostname or ""
    try:
        local = ipaddress.ip_address(host).is_loopback
    except ValueError:
        local = host == "localhost"
    if (parsed.scheme not in ("http", "https")
            or not (local or host == "aimaster-llm-gateway-service.aimaster-web")
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment):
        raise ValueError("endpoint must be the internal gateway or a loopback address")
    return base_url.rstrip("/")


def _read_api_key(env_name: str, env_file: Path) -> str | None:
    if os.environ.get(env_name):
        return os.environ[env_name]
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:].lstrip()
            key, sep, value = line.partition("=")
            if sep and key.strip() == env_name:
                parts = shlex.split(value, comments=True)
                if len(parts) != 1:
                    raise ValueError(f"invalid {env_name} entry in env file")
                return parts[0]
    return None


class OpenAICompatEndpoint(_RetryingEndpoint):
    """Standard-library OpenAI chat transport; no SDK retries or implicit proxies."""

    def __init__(self, name: str, *, base_url: str, max_concurrency: int,
                 max_attempts: int, backoff_s: Iterable[float], timeout_s: float,
                 api_key_env: str = "LLM_GATEWAY_API_KEY", env_file: str | Path = ".env",
                 sleep: Callable[[float], None] = time.sleep,
                 request_options: Mapping[str, object] | None = None,
                 stream: bool = False):
        super().__init__(name, max_concurrency=max_concurrency,
                         max_attempts=max_attempts, backoff_s=backoff_s, sleep=sleep,
                         request_options=request_options)
        self.base_url = _validate_url(base_url)
        if type(stream) is not bool:
            raise ValueError("stream must be boolean")
        self.stream = stream
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._timeout_s = timeout_s
        self._api_key = _read_api_key(api_key_env, Path(env_file))
        if "aimaster-llm-gateway-service.aimaster-web" == urllib.parse.urlsplit(base_url).hostname:
            if not self._api_key:
                raise ValueError(f"missing {api_key_env}; set it in the environment or .env")
        # An explicit empty proxy map ignores http/https/all_proxy in BOTH cases.
        # Do not clear process-wide environment variables used by HuggingFace.
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirects())

    def _request(self, payload: dict) -> dict:
        if self.stream:
            payload = {**payload, "stream": True,
                       "stream_options": {"include_usage": True}}
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        try:
            with self._opener.open(request, timeout=self._timeout_s) as response:
                return self._read_stream(response) if self.stream else json.load(response)
        except urllib.error.HTTPError as exc:
            try:
                body = json.loads(exc.read())
                usage = body.get("usage") or (body.get("error") or {}).get("usage") or {}
                if not isinstance(usage, dict):
                    usage = {}
            except (ValueError, AttributeError, TypeError):
                usage = {}
            finally:
                exc.close()
            raise EndpointHTTPError(exc.code, usage) from exc

    @staticmethod
    def _read_stream(response) -> dict:
        content, data = [], []
        usage, finish = {}, None
        try:
            from itertools import chain
            for raw_line in chain(response, (b"\n",)):
                line = raw_line.decode("utf-8").rstrip("\r\n")
                if line.startswith("data:"):
                    data.append(line[5:].lstrip(" "))
                elif not line and data:
                    event = "\n".join(data)
                    data.clear()
                    if event == "[DONE]":
                        if finish is None or not usage:
                            raise EndpointStreamError(usage)
                        return {"choices": [{"message": {"content": "".join(content)},
                                             "finish_reason": finish}], "usage": usage}
                    chunk = json.loads(event)
                    if isinstance(chunk.get("usage"), dict) and chunk["usage"]:
                        usage = chunk["usage"]
                    if "error" in chunk:
                        raise EndpointStreamError(usage)
                    for choice in chunk.get("choices") or []:
                        if choice.get("index", 0) != 0:
                            raise EndpointStreamError(usage)
                        text = (choice.get("delta") or {}).get("content")
                        if text is not None:
                            if not isinstance(text, str) or (finish is not None and text):
                                raise EndpointStreamError(usage)
                            content.append(text)
                        if choice.get("finish_reason") is not None:
                            finish = choice["finish_reason"]
            raise EndpointStreamError(usage)
        except (OSError, http.client.HTTPException, ValueError, KeyError,
                TypeError, AttributeError) as exc:
            raise EndpointStreamError(usage) from exc


class MockEndpoint(_RetryingEndpoint):
    """In-process transport using the very same semaphore and retry implementation.

    The supplied responder returns an OpenAI-shaped dictionary or raises
    EndpointHTTPError. The accuracy/task fixture belongs in tests/mock_endpoint.py;
    production prompts never need mock metadata or gold answers injected into them.
    """

    def __init__(self, name: str, *, responder: Callable[[dict], dict],
                 max_concurrency: int, max_attempts: int, backoff_s: Iterable[float],
                 sleep: Callable[[float], None] = time.sleep,
                 request_options: Mapping[str, object] | None = None):
        super().__init__(name, max_concurrency=max_concurrency,
                         max_attempts=max_attempts, backoff_s=backoff_s, sleep=sleep,
                         request_options=request_options)
        self._responder = responder

    def _request(self, payload: dict) -> dict:
        return self._responder(payload)


class ReplayOnlyEndpoint:
    """Stands in for the gateway when only cached cells may be read.

    Carries the identity fields the Executor binds (name, base_url, request_options)
    so it opens the same experiment, needs no API key, and refuses to generate. The
    Executor's replay_only flag already raises CacheMiss first; this is the second
    lock, so a replay can never reach the network even if that flag is forgotten.
    """

    def __init__(self, name: str, *, base_url: str | None = None,
                 request_options: Mapping[str, object] | None = None):
        self.name = name
        self.base_url = _validate_url(base_url) if base_url else None
        self.request_options = validate_request_options(request_options)

    def complete(self, prompt: str, *, seed: int, max_tokens: int,
                 temperature: float) -> CallResult:
        raise RuntimeError("replay-only endpoint: this cell is not in the tensor")


def endpoint_from_config(model: Mapping, *, concurrency: int | None = None,
                         replay_only: bool = False):
    """The one place a config's `model:` block becomes an endpoint."""
    options = model.get("request_options") or {}
    if replay_only:
        endpoint = ReplayOnlyEndpoint(model["name"], base_url=model["endpoint"],
                                      request_options=options)
        endpoint.stream = model.get("stream", False)
        return endpoint
    retry = model["retry"]
    return OpenAICompatEndpoint(
        name=model["name"], base_url=model["endpoint"],
        api_key_env=model.get("api_key_env", "LLM_GATEWAY_API_KEY"),
        max_concurrency=concurrency or model["max_concurrency"],
        max_attempts=retry["max_attempts"], backoff_s=retry["backoff_s"],
        timeout_s=model.get("timeout_s", 900), request_options=options,
        stream=model.get("stream", False))
