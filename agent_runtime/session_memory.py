"""Process-local, per-``thread_id`` conversation memory.

This is the session-local analogue of the durable OpenSearch store in
``rag_pipeline/memory_module.py``.  The two layers are intentionally distinct:

* **Persistent memory** (OpenSearch, gated by ``use_persistent_memory``) is the
  durable record that survives process restarts.
* **Session memory** (this module) preserves conversation turns *within a single
  process lifetime* keyed by ``thread_id`` so that, even when persistent memory is
  switched OFF, follow-up turns ("show me the code", "explain that") still see the
  prior conversation.

Both layers store turns in the same shape — ``{"userQuery": ..., "answer": ...}`` —
so ``agent_chat_service._build_chat_history`` can consume either one transparently.

The store is bounded (per-thread turn cap + global LRU thread cap) to avoid
unbounded growth in a long-lived process, and is thread-safe (the streaming and
non-streaming entry points may run concurrently).

Multi-worker contract
----------------------
This store (and the LangGraph ``BoundedInMemorySaver`` checkpointer in
``executor_factory``) are **process-local**. Across multiple server workers,
multi-turn continuity is only guaranteed when EITHER:

* requests for a given conversation are routed to the same worker (sticky
  sessions, keyed on ``thread_id`` / ``memory_id``), OR
* persistent memory is enabled (``use_persistent_memory`` → OpenSearch), which is
  shared across workers and survives restarts.

Without one of those, a follow-up turn handled by a different worker will not see
the prior conversation. Single-worker / sticky deployments need no extra config.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional

_LOCK = threading.RLock()
# Insertion-ordered so we can evict the least-recently-used thread first.
_STORE: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
# Per-thread set of file_ids attached anywhere in the session, so an upload from one
# turn stays accessible to the agent on later turns ("visualize it" / "execute it").
_FILES: "OrderedDict[str, List[str]]" = OrderedDict()

# What the agent DID, as opposed to what it said. The turn store above keeps
# {userQuery, answer} — prose — so everything a tool produced is discarded at the end of the
# turn. Observed: a clay embedding was computed with pixel_ground_m in its result, the user
# asked "original resolution or downsampled?" on the next turn, and the supervisor (which sees
# no prior-turn state) routed to search because as far as it could tell nothing had happened.
# Forty-nine keyword searches later the payload exceeded the model's context window and the
# turn died. The answer had been in hand the whole time; nothing remembered it.
_ACTIONS: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()

_DEFAULT_MAX_TURNS = 50
_DEFAULT_MAX_THREADS = 500
_DEFAULT_MAX_FILES = 50
# Rows, not turns: one substantial turn can be a dozen tool calls. Small on purpose — this is
# injected into a routing decision, and a ledger that crowds the window recreates the very
# failure it exists to prevent.
_DEFAULT_MAX_ACTIONS = 80


def _max_files() -> int:
    try:
        return max(1, int(os.getenv("AGENT_SESSION_MEMORY_MAX_FILES", str(_DEFAULT_MAX_FILES))))
    except (TypeError, ValueError):
        return _DEFAULT_MAX_FILES


def _max_actions() -> int:
    try:
        return max(1, int(os.getenv("AGENT_SESSION_MEMORY_MAX_ACTIONS", str(_DEFAULT_MAX_ACTIONS))))
    except (TypeError, ValueError):
        return _DEFAULT_MAX_ACTIONS


def _max_turns() -> int:
    try:
        return max(1, int(os.getenv("AGENT_SESSION_MEMORY_MAX_TURNS", str(_DEFAULT_MAX_TURNS))))
    except (TypeError, ValueError):
        return _DEFAULT_MAX_TURNS


def _max_threads() -> int:
    try:
        return max(1, int(os.getenv("AGENT_SESSION_MEMORY_MAX_THREADS", str(_DEFAULT_MAX_THREADS))))
    except (TypeError, ValueError):
        return _DEFAULT_MAX_THREADS


def get_session_history(thread_id: Optional[str]) -> List[Dict[str, Any]]:
    """Return a copy of the recorded turns for *thread_id* (oldest first).

    Returns an empty list when *thread_id* is falsy or unknown.
    """
    if not thread_id:
        return []
    with _LOCK:
        turns = _STORE.get(thread_id)
        if not turns:
            return []
        _STORE.move_to_end(thread_id)  # mark as recently used
        return [dict(turn) for turn in turns]


def build_session_memory_doc(thread_id: Optional[str]) -> Dict[str, Any]:
    """Return a ``memory_doc``-shaped dict ``{"chat_history": [...]}``.

    This mirrors the shape returned by ``rag_pipeline.memory_module.get_memory`` so
    it can be fed directly to ``agent_chat_service._build_chat_history``.
    """
    return {"chat_history": get_session_history(thread_id)}


def append_session_turn(thread_id: Optional[str], user_query: str, answer: str) -> None:
    """Record one conversation turn for *thread_id*.

    No-op when *thread_id* is falsy (there is no stable key to attach the turn to).
    Enforces the per-thread turn cap and the global LRU thread cap.
    """
    if not thread_id:
        return
    entry = {"userQuery": str(user_query or ""), "answer": str(answer or "")}
    with _LOCK:
        turns = _STORE.get(thread_id)
        if turns is None:
            turns = []
            _STORE[thread_id] = turns
        turns.append(entry)

        max_turns = _max_turns()
        if len(turns) > max_turns:
            del turns[: len(turns) - max_turns]

        _STORE.move_to_end(thread_id)  # most-recently used

        max_threads = _max_threads()
        while len(_STORE) > max_threads:
            _STORE.popitem(last=False)  # evict least-recently used


def get_session_files(thread_id: Optional[str]) -> List[str]:
    """Return file_ids attached anywhere in this session (oldest first)."""
    if not thread_id:
        return []
    with _LOCK:
        ids = _FILES.get(thread_id)
        if not ids:
            return []
        _FILES.move_to_end(thread_id)
        return list(ids)


def append_session_files(thread_id: Optional[str], file_ids: Optional[List[str]]) -> None:
    """Record file_ids attached this turn so later turns can still reach them.

    Deduped (preserving order), bounded per-thread and globally (LRU). No-op when
    *thread_id* is falsy or no file_ids are given.
    """
    ids = [str(f).strip() for f in (file_ids or []) if str(f).strip()]
    if not thread_id or not ids:
        return
    with _LOCK:
        cur = _FILES.get(thread_id)
        if cur is None:
            cur = []
            _FILES[thread_id] = cur
        for f in ids:
            if f not in cur:
                cur.append(f)
        max_files = _max_files()
        if len(cur) > max_files:
            del cur[: len(cur) - max_files]
        _FILES.move_to_end(thread_id)
        max_threads = _max_threads()
        while len(_FILES) > max_threads:
            _FILES.popitem(last=False)


def get_session_actions(thread_id: Optional[str]) -> List[Dict[str, Any]]:
    """What this conversation has already DONE, oldest first."""
    if not thread_id:
        return []
    with _LOCK:
        rows = _ACTIONS.get(thread_id)
        if not rows:
            return []
        _ACTIONS.move_to_end(thread_id)
        return [dict(r) for r in rows]


def append_session_actions(thread_id: Optional[str],
                           actions: Optional[List[Dict[str, Any]]]) -> None:
    """Record what this turn's tools did, so a later turn need not redo it.

    Deliberately NOT the durable record: this is process-local working context for routing,
    the same lifetime and bounds as the turn store beside it.
    """
    rows = [dict(a) for a in (actions or []) if isinstance(a, dict)]
    if not thread_id or not rows:
        return
    with _LOCK:
        cur = _ACTIONS.get(thread_id)
        if cur is None:
            cur = []
            _ACTIONS[thread_id] = cur
        cur.extend(rows)
        max_actions = _max_actions()
        if len(cur) > max_actions:
            # Drop the OLDEST: a follow-up is nearly always about recent work.
            del cur[: len(cur) - max_actions]
        _ACTIONS.move_to_end(thread_id)
        max_threads = _max_threads()
        while len(_ACTIONS) > max_threads:
            _ACTIONS.popitem(last=False)


def clear_session(thread_id: Optional[str]) -> None:
    """Drop all recorded turns, files and actions for *thread_id* (no-op if unknown)."""
    if not thread_id:
        return
    with _LOCK:
        _STORE.pop(thread_id, None)
        _FILES.pop(thread_id, None)
        _ACTIONS.pop(thread_id, None)


def reset_all() -> None:
    """Clear the entire session store (primarily for tests)."""
    with _LOCK:
        _STORE.clear()
        _FILES.clear()
        _ACTIONS.clear()


__all__ = [
    "append_session_turn",
    "append_session_files",
    "append_session_actions",
    "get_session_actions",
    "build_session_memory_doc",
    "clear_session",
    "get_session_files",
    "get_session_history",
    "reset_all",
]
