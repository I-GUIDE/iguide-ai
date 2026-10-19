"""The agent's own OpenAI-compatible chat client (architecture stage 88).

Phase 1 of the agent's own framework. LangChain's ``ChatOpenAI`` sat underneath every model
call, and four defects kept being routed around rather than fixed:

* no call could be stopped or timed out from our side. The openai SDK waits up to 600 s per
  attempt and retries twice, so a Lumen "Upstream error" 500 that arrives after ~6 minutes cost
  18 minutes of a turn (prod sweep, 2026-10-09), and Stop only took effect at the NEXT step;
* the reasoning text was dropped, so it took a ``ChatOpenAI`` subclass to keep it;
* a callback's exception is swallowed, which is why ``TurnCancelled`` had to become a
  ``BaseException`` (stage 78);
* per-turn cost rides a ContextVar configure hook that has to be carried into threads by hand.

This module is the transport only: HTTP to ``/chat/completions``, a deadline per attempt, Stop,
bounded retries and streaming. It imports nothing from LangChain. ``model_client_langchain``
wraps it as a LangChain chat model so ``build_default_llm`` / ``build_llm`` callers do not
change, and ``AGENT_MODEL_CLIENT=own`` selects it (the default stays ``langchain``).

How a call is bounded. The request runs on a helper thread; the calling thread waits in short
slices and, between slices, checks two things:

* ``turn_control.cancel_requested()`` — the person pressed Stop. The call is aborted and
  :class:`~agent_runtime.turn_control.TurnCancelled` is raised at once, mid-call, not at the
  next step;
* the attempt's deadline (``AGENT_LLM_TIMEOUT_SECONDS``). The call is aborted and, if a retry
  is left, tried again; otherwise :class:`ModelCallTimeout` is raised.

"Aborted" means the socket is shut down, not only abandoned: every attempt holds its own
connection pool, whose sockets this module records as it opens them (``_TrackingBackend``), so
an abort wakes the helper thread out of its blocking read and the connection is gone. A pool is
reused by the next call after a clean finish, so ordinary calls keep their keep-alive
connection. A streamed call is bounded by the same deadline between chunks (an idle timeout),
since a long answer that keeps arriving is progress and a silent socket is not.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import random
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import httpcore
import httpx

logger = logging.getLogger(__name__)

CLIENT_ENV = "AGENT_MODEL_CLIENT"           # own | langchain (default)
TIMEOUT_ENV = "AGENT_LLM_TIMEOUT_SECONDS"   # per attempt; shared with the ChatOpenAI path
RETRIES_ENV = "AGENT_LLM_MAX_RETRIES"       # retries after the first attempt

# Per attempt. Long enough for the slowest legitimate call measured on the harness (a code-peer
# step writing a long script on deepseek-v4-flash), short of the ~6 min after which Lumen answers
# a long-context call with an "Upstream error" 500, and below AnvilGPT's 300 s gateway limit.
DEFAULT_TIMEOUT_S = 180.0
# One retry: a transient 5xx or a dropped connection gets a second chance, a stalled provider
# costs two deadlines (6 min) rather than the SDK's three 10-minute attempts.
DEFAULT_MAX_RETRIES = 1
CONNECT_TIMEOUT_S = 15.0
# How often the waiting thread looks at Stop and the deadline.
_POLL_S = 0.2
# The statuses the openai SDK also retries, and nothing else: a 400/401/404/422 is the same
# answer the second time.
_RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_BACKOFF_BASE_S = 0.5
_BACKOFF_MAX_S = 8.0
_RETRY_AFTER_MAX_S = 30.0
_ERROR_EXCERPT = 600
_POOL_IDLE_MAX = 4


def client_choice() -> str:
    """``own`` or ``langchain``: which client builds the agent's chat models."""
    value = (os.getenv(CLIENT_ENV) or "").strip().lower()
    return "own" if value == "own" else "langchain"


def _float_env(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", name, raw, default)
        return default
    return value if value > 0 else default


def _int_env(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return max(0, int(float(raw)))
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", name, raw, default)
        return default


def default_timeout() -> float:
    return _float_env(TIMEOUT_ENV, DEFAULT_TIMEOUT_S)


def default_max_retries() -> int:
    return _int_env(RETRIES_ENV, DEFAULT_MAX_RETRIES)


class ModelCallError(RuntimeError):
    """A model call failed after its retries. The text keeps the openai SDK's
    ``Error code: <status> - <body>`` shape, which log readers and the server's error
    classification already match on."""

    def __init__(self, message: str, *, status: Optional[int] = None, body: str = "",
                 attempts: int = 1, retryable: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.body = body
        self.attempts = attempts
        self.retryable = retryable


class ModelCallTimeout(ModelCallError):
    """No answer within ``AGENT_LLM_TIMEOUT_SECONDS`` on every attempt."""


def _cancel_requested() -> bool:
    try:
        from agent_runtime import turn_control

        return turn_control.cancel_requested()
    except Exception:  # noqa: BLE001 - a missing turn table never stops a call
        return False


def _raise_cancelled() -> None:
    from agent_runtime import turn_control

    turn_control.raise_if_cancelled()
    raise turn_control.TurnCancelled("stopped by the user")


class _TrackingBackend(httpcore.SyncBackend):
    """httpcore's own backend, remembering every socket it opens so an abort can shut it.

    ``socket.close()`` from another thread does not wake a ``recv`` blocked on Linux;
    ``shutdown(SHUT_RDWR)`` does. Every socket of a pool belongs to one attempt at a time
    (``_Pool`` lends it exclusively), so shutting all of them aborts exactly that attempt.
    """

    def __init__(self) -> None:
        super().__init__()
        self._socks: List[socket.socket] = []
        self._lock = threading.Lock()

    def connect_tcp(self, *args: Any, **kwargs: Any) -> Any:
        stream = super().connect_tcp(*args, **kwargs)
        sock = stream.get_extra_info("socket")
        if sock is not None:
            with self._lock:
                self._socks = [s for s in self._socks if s.fileno() != -1] + [sock]
        return stream

    def abort(self) -> None:
        with self._lock:
            socks, self._socks = self._socks, []
        for sock in socks:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


class _Lease:
    """One httpx client, lent to one attempt at a time."""

    def __init__(self) -> None:
        self.backend = _TrackingBackend()
        transport = httpx.HTTPTransport(retries=0)
        # httpx 0.28 builds its httpcore pool in __init__ with the default backend and offers no
        # argument for another. Replaced here, and pinned by a test so an httpx that renames it
        # fails loudly instead of silently losing the hard abort (the deadline still holds).
        pool = getattr(transport, "_pool", None)
        self.abortable = pool is not None and hasattr(pool, "_network_backend")
        if self.abortable:
            pool._network_backend = self.backend
        self.client = httpx.Client(transport=transport, follow_redirects=False)

    def abort(self) -> None:
        self.backend.abort()
        try:
            self.client.close()
        except Exception:  # noqa: BLE001 - closing a client we are discarding
            pass


class _Pool:
    """Idle clients for reuse, so ordinary calls keep their keep-alive connection."""

    def __init__(self) -> None:
        self._idle: List[_Lease] = []
        self._lock = threading.Lock()

    def take(self) -> _Lease:
        with self._lock:
            if self._idle:
                return self._idle.pop()
        return _Lease()

    def give_back(self, lease: _Lease) -> None:
        with self._lock:
            if len(self._idle) < _POOL_IDLE_MAX:
                self._idle.append(lease)
                return
        lease.abort()


@dataclass
class _Outcome:
    status: int
    headers: Dict[str, str]
    body: bytes


class _Retry(Exception):
    """One attempt failed in a way worth another attempt."""

    def __init__(self, error: ModelCallError, retry_after: Optional[float] = None) -> None:
        super().__init__(str(error))
        self.error = error
        self.retry_after = retry_after


def _excerpt(body: bytes) -> str:
    text = body.decode("utf-8", "replace").strip()
    return text[:_ERROR_EXCERPT] + ("…" if len(text) > _ERROR_EXCERPT else "")


def _retry_after(headers: Dict[str, str]) -> Optional[float]:
    raw = headers.get("retry-after-ms")
    try:
        if raw:
            return float(raw) / 1000.0
        raw = headers.get("retry-after")
        return float(raw) if raw else None
    except ValueError:
        return None


class ChatClient:
    """``POST {base_url}/chat/completions`` with a deadline, Stop and bounded retries.

    Thread-safe: concurrent calls (parallel peers) each lease their own connection pool.
    """

    def __init__(self, base_url: str, api_key: Optional[str], *,
                 timeout: Optional[float] = None, max_retries: Optional[int] = None,
                 headers: Optional[Dict[str, str]] = None) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self._api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._headers = dict(headers or {})
        self._pool = _Pool()

    # -- settings, read per call so an operator's env change applies to the next call ----------
    def _timeout(self, override: Optional[float]) -> float:
        for value in (override, self.timeout):
            if value is not None and float(value) > 0:
                return float(value)
        return default_timeout()

    def _retries(self) -> int:
        return self.max_retries if self.max_retries is not None else default_max_retries()

    def _request_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   **self._headers}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    @property
    def url(self) -> str:
        return f"{self.base_url}/chat/completions"

    # -- public ---------------------------------------------------------------------------------
    def complete(self, body: Dict[str, Any], *, timeout: Optional[float] = None) -> Dict[str, Any]:
        """One non-streamed completion. Returns the provider's JSON."""
        payload = {**body, "stream": False}
        deadline_s = self._timeout(timeout)
        retries = self._retries()
        for n in range(retries + 1):
            t0 = time.monotonic()
            try:
                return self._complete_once(payload, deadline_s)
            except _Retry as exc:
                self._after_failure(exc, n, retries, payload, t0)
        raise AssertionError("unreachable")  # pragma: no cover

    def stream(self, body: Dict[str, Any], *,
               timeout: Optional[float] = None) -> Iterator[Dict[str, Any]]:
        """A streamed completion: yields each ``data:`` chunk's JSON.

        Retried only until the first chunk arrives. After that a retry would replay text the
        caller has already been given, so a failure mid-stream is raised.
        """
        payload = {**body, "stream": True}
        payload.setdefault("stream_options", {"include_usage": True})
        idle_s = self._timeout(timeout)
        retries = self._retries()
        for n in range(retries + 1):
            t0 = time.monotonic()
            started = False
            try:
                for chunk in self._stream_once(payload, idle_s):
                    started = True
                    yield chunk
                return
            except _Retry as exc:
                if started:
                    exc.error.attempts = n + 1
                    raise exc.error from exc
                self._after_failure(exc, n, retries, payload, t0)

    # -- retries --------------------------------------------------------------------------------
    def _after_failure(self, exc: "_Retry", n: int, retries: int, payload: Dict[str, Any],
                       t0: float) -> None:
        """Raise the error when no retry is left; otherwise wait out the backoff."""
        exc.error.attempts = n + 1
        if n >= retries:
            raise exc.error from exc
        wait = self._backoff(n, exc.retry_after)
        logger.warning("model call to %s failed (%s) after %.1f s; retry %d/%d in %.1f s",
                       payload.get("model"), exc.error, time.monotonic() - t0, n + 1, retries,
                       wait)
        self._sleep(wait)

    @staticmethod
    def _backoff(n: int, retry_after: Optional[float]) -> float:
        if retry_after is not None and 0 <= retry_after <= _RETRY_AFTER_MAX_S:
            return retry_after
        wait = min(_BACKOFF_BASE_S * (2 ** n), _BACKOFF_MAX_S)
        return wait * (1 - 0.25 * random.random())

    @staticmethod
    def _sleep(seconds: float) -> None:
        end = time.monotonic() + seconds
        while True:
            if _cancel_requested():
                _raise_cancelled()
            left = end - time.monotonic()
            if left <= 0:
                return
            time.sleep(min(_POLL_S, left))

    # -- one attempt ----------------------------------------------------------------------------
    def _complete_once(self, payload: Dict[str, Any], deadline_s: float) -> Dict[str, Any]:
        outcome = self._exchange(payload, deadline_s)
        try:
            data = json.loads(outcome.body)
        except ValueError as exc:
            raise _Retry(ModelCallError(
                f"provider sent a non-JSON reply ({outcome.status}): {_excerpt(outcome.body)}",
                status=outcome.status, body=_excerpt(outcome.body), retryable=True)) from exc
        if not isinstance(data, dict):
            raise ModelCallError(f"provider sent {type(data).__name__}, not an object",
                                 status=outcome.status)
        return data

    def _exchange(self, payload: Dict[str, Any], deadline_s: float) -> _Outcome:
        lease = self._pool.take()
        box: Dict[str, Any] = {}
        done = threading.Event()
        timeout = httpx.Timeout(deadline_s + 5.0, connect=min(CONNECT_TIMEOUT_S, deadline_s))

        def work() -> None:
            try:
                r = lease.client.post(self.url, json=payload, headers=self._request_headers(),
                                      timeout=timeout)
                box["outcome"] = _Outcome(r.status_code, dict(r.headers), r.content)
            except BaseException as exc:  # noqa: BLE001 - handed to the waiting thread
                box["error"] = exc
            finally:
                done.set()

        if _cancel_requested():
            self._pool.give_back(lease)
            _raise_cancelled()
        threading.Thread(target=work, name="model-call", daemon=True).start()
        deadline = time.monotonic() + deadline_s
        while not done.wait(_POLL_S):
            if _cancel_requested():
                lease.abort()
                _raise_cancelled()
            if time.monotonic() >= deadline:
                lease.abort()
                raise _Retry(ModelCallTimeout(
                    f"model call to {payload.get('model')}: no answer within {deadline_s:g} s",
                    retryable=True))
        error = box.get("error")
        if error is not None:
            lease.abort()
            if isinstance(error, httpx.TimeoutException):
                raise _Retry(ModelCallTimeout(
                    f"model call to {payload.get('model')}: {type(error).__name__} "
                    f"after {deadline_s:g} s", retryable=True)) from error
            if isinstance(error, httpx.TransportError):
                raise _Retry(ModelCallError(
                    f"connection to {self.base_url} failed: {type(error).__name__}: {error}",
                    retryable=True)) from error
            raise error
        outcome: _Outcome = box["outcome"]
        self._pool.give_back(lease)
        if outcome.status >= 400:
            self._raise_status(outcome.status, outcome.headers, outcome.body)
        return outcome

    def _raise_status(self, status: int, headers: Dict[str, str], body: bytes) -> None:
        text = _excerpt(body)
        error = ModelCallError(f"Error code: {status} - {text}", status=status, body=text,
                               retryable=status in _RETRY_STATUSES)
        if status in _RETRY_STATUSES:
            raise _Retry(error, _retry_after({k.lower(): v for k, v in headers.items()}))
        raise error

    def _stream_once(self, payload: Dict[str, Any], idle_s: float) -> Iterator[Dict[str, Any]]:
        lease = self._pool.take()
        items: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        timeout = httpx.Timeout(idle_s + 5.0, connect=min(CONNECT_TIMEOUT_S, idle_s))

        def work() -> None:
            try:
                with lease.client.stream("POST", self.url, json=payload,
                                         headers=self._request_headers(),
                                         timeout=timeout) as r:
                    if r.status_code >= 400:
                        items.put(("status", (r.status_code, dict(r.headers), r.read())))
                        return
                    for line in r.iter_lines():
                        items.put(("line", line))
                items.put(("end", None))
            except BaseException as exc:  # noqa: BLE001 - handed to the waiting thread
                items.put(("error", exc))

        if _cancel_requested():
            self._pool.give_back(lease)
            _raise_cancelled()
        threading.Thread(target=work, name="model-stream", daemon=True).start()
        clean = False
        try:
            deadline = time.monotonic() + idle_s
            while True:
                try:
                    kind, value = items.get(timeout=_POLL_S)
                except queue.Empty:
                    if _cancel_requested():
                        _raise_cancelled()          # the finally below aborts the lease
                    if time.monotonic() >= deadline:
                        raise _Retry(ModelCallTimeout(
                            f"stream from {payload.get('model')}: nothing for {idle_s:g} s",
                            retryable=True))
                    continue
                deadline = time.monotonic() + idle_s
                if kind == "end":
                    clean = True
                    return
                if kind == "status":
                    clean = True
                    status, headers, body = value
                    self._raise_status(status, headers, body)
                if kind == "error":
                    if isinstance(value, httpx.TimeoutException):
                        raise _Retry(ModelCallTimeout(
                            f"stream from {payload.get('model')}: {type(value).__name__}",
                            retryable=True)) from value
                    if isinstance(value, httpx.TransportError):
                        raise _Retry(ModelCallError(
                            f"connection to {self.base_url} failed: {type(value).__name__}: "
                            f"{value}", retryable=True)) from value
                    raise value
                line = (value or "").strip()
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    continue
                try:
                    chunk = json.loads(data)
                except ValueError:
                    logger.debug("unparseable stream line from %s: %.200s",
                                 payload.get("model"), data)
                    continue
                if isinstance(chunk, dict) and chunk.get("error"):
                    raise ModelCallError(
                        "stream error: " + _excerpt(json.dumps(chunk["error"]).encode()))
                if isinstance(chunk, dict):
                    yield chunk
        finally:
            if clean:
                self._pool.give_back(lease)
            else:
                lease.abort()


__all__ = ["CLIENT_ENV", "ChatClient", "DEFAULT_MAX_RETRIES", "DEFAULT_TIMEOUT_S",
           "ModelCallError", "ModelCallTimeout", "RETRIES_ENV", "TIMEOUT_ENV", "client_choice",
           "default_max_retries", "default_timeout"]
