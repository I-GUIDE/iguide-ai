"""Stopping a turn on the server, not just in the browser.

The map UI's Stop used to abort the browser's fetch and nothing else: the worker thread kept
going, the sandbox container kept computing for up to its whole time limit, and the model calls
after it were still made and charged. This module gives each streamed turn a handle the client
can cancel by id, and the places that do slow work a way to notice.

Two ways a cancellation takes effect:

* **between steps** — the streaming callback handler calls :func:`raise_if_cancelled` at every
  model call and tool start, which raises :class:`TurnCancelled`;
* **inside a code run** — the sandbox's supervision loop polls :func:`cancel_requested` and
  kills the container (code_execution._supervise).

:class:`TurnCancelled` derives from ``BaseException`` on purpose: the tool node, the callback
manager and the supervisor's containment wrappers all catch ``Exception`` and turn it into a
recoverable observation, which for a cancellation would mean the agent carries on and tries
something else. The two supervisor wrappers that catch ``BaseException`` re-raise it by name.
"""

from __future__ import annotations

import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional


class TurnCancelled(BaseException):
    """The person who started this turn stopped it."""


@dataclass
class TurnHandle:
    id: str
    owner: Optional[str]
    started: float = field(default_factory=time.monotonic)
    event: threading.Event = field(default_factory=threading.Event)
    reason: Optional[str] = None

    @property
    def cancelled(self) -> bool:
        return self.event.is_set()


_TURNS: Dict[str, TurnHandle] = {}
_LOCK = threading.Lock()
_CURRENT: ContextVar[Optional[TurnHandle]] = ContextVar("agent_turn_handle", default=None)
# A turn left in the table by a crashed worker must not hold memory forever.
_MAX_AGE_S = 6 * 3600


def begin(owner: Optional[str]) -> TurnHandle:
    """Register a new turn. ``owner`` is who may cancel it (None: any caller of this process)."""
    handle = TurnHandle(id=uuid.uuid4().hex, owner=owner)
    now = time.monotonic()
    with _LOCK:
        for key in [k for k, h in _TURNS.items() if now - h.started > _MAX_AGE_S]:
            _TURNS.pop(key, None)
        _TURNS[handle.id] = handle
    return handle


def end(handle: Optional[TurnHandle]) -> None:
    if handle is None:
        return
    with _LOCK:
        _TURNS.pop(handle.id, None)


@contextmanager
def bound(handle: Optional[TurnHandle]) -> Iterator[None]:
    """Make *handle* the current turn for this context (the worker thread)."""
    token = _CURRENT.set(handle)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def current() -> Optional[TurnHandle]:
    return _CURRENT.get()


def cancel_requested() -> bool:
    handle = _CURRENT.get()
    return bool(handle is not None and handle.cancelled)


def raise_if_cancelled() -> None:
    handle = _CURRENT.get()
    if handle is not None and handle.cancelled:
        raise TurnCancelled(handle.reason or "stopped by the user")


def request_cancel(turn_id: str, owner: Optional[str], *,
                   reason: str = "stopped by the user") -> str:
    """Cancel a running turn. Returns 'cancelled', 'not_found' or 'not_yours'.

    A turn someone else started reads as not found to anyone but its owner would leak that it
    exists; 'not_yours' is returned so the route can answer with the same 404 either way while
    the log still says which it was.
    """
    with _LOCK:
        handle = _TURNS.get(str(turn_id or ""))
    if handle is None:
        return "not_found"
    if handle.owner is not None and handle.owner != owner:
        return "not_yours"
    handle.reason = reason
    handle.event.set()
    return "cancelled"


def running() -> int:
    with _LOCK:
        return len(_TURNS)


__all__ = ["TurnCancelled", "TurnHandle", "begin", "bound", "cancel_requested", "current", "end",
           "raise_if_cancelled", "request_cancel", "running"]
