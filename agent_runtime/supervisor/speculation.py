"""Composing the answer while the supervisor decides whether the turn is done (Stage 92).

After a peer finishes, two model calls ran one after the other: the decider (about 1.7–1.9 s on
Lumen deepseek-v4-flash, measured live on 2026-11-13) and then the answer writer
(`do_synthesize`, 2.7–4.9 s). In 331 of the 363 harness turns kept in gis_harness/baselines the
decider's answer after the first peer was `done`, so the writer's call was waiting on a decision
that almost always let it run.

Now the supervisor starts the writer's call in the background as soon as it can build the
writer's inputs, and synthesize takes the draft only when the inputs it builds itself are
identical (`take`, compared as canonical JSON). Otherwise it calls the writer as before, so the
answer the user sees is the one the old path would have produced from the same inputs. A draft
nobody takes (the decider chose another peer, or the inputs changed) is dropped; its model call
still runs to completion and is still billed.

`AGENT_SPECULATIVE_SYNTHESIS=0` turns it off.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

ENV = "AGENT_SPECULATIVE_SYNTHESIS"
# How long synthesize waits for a matching draft before writing its own. A draft that is still
# running when synthesize starts is normally the faster of the two, since it started earlier.
WAIT_S = float(os.getenv("AGENT_SPECULATIVE_WAIT_S", "300") or 300)

_lock = threading.Lock()
_drafts: Dict[str, Tuple[str, Future]] = {}
_pool: Optional[ThreadPoolExecutor] = None


def enabled() -> bool:
    return (os.getenv(ENV, "1") or "1").strip().lower() not in {"0", "false", "no", "off"}


def fingerprint(args: Any) -> Optional[str]:
    """Canonical JSON of the writer's inputs, hashed; None when they cannot be serialised
    faithfully (then nothing is shared, which is always safe)."""
    try:
        text = json.dumps(args, sort_keys=True, ensure_ascii=True, default=_strict)
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _strict(obj: Any) -> Any:
    # A default=str fallback would turn objects into "<X at 0x7f..>", which differs between
    # two equal states and, worse, could make two different ones look alike. Refuse instead.
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=repr)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"not JSON: {type(obj).__name__}")


def _executor() -> ThreadPoolExecutor:
    # Built on first use, so a forked worker never inherits a pool from its parent.
    global _pool
    with _lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="speculative-synth")
        return _pool


def start(key: Optional[str], args: Tuple[Any, ...], compose: Callable[..., Any]) -> bool:
    """Run compose(*args) in the background for this turn. The caller's context (the trace
    stream, the request's settings, turn control) is copied into the worker thread."""
    if not key or not enabled():
        return False
    fp = fingerprint(args)
    if fp is None:
        # Logged, because a turn that never drafts saves nothing and says nothing otherwise.
        logger.info("speculative synthesis not started for turn %s: inputs not comparable", key)
        return False
    ctx = contextvars.copy_context()
    future = _executor().submit(ctx.run, compose, *args)
    with _lock:
        _drafts[key] = (fp, future)
    logger.info("speculative synthesis started for turn %s", key)
    return True


def take(key: Optional[str], args: Tuple[Any, ...]) -> Optional[Any]:
    """The draft for this turn if it was composed from exactly these inputs, else None."""
    if not key:
        return None
    with _lock:
        entry = _drafts.pop(key, None)
    if entry is None:
        return None
    fp, future = entry
    if fingerprint(args) != fp:
        logger.info("speculative synthesis discarded for turn %s: inputs changed", key)
        return None
    try:
        answer = future.result(timeout=WAIT_S)
    except Exception as exc:  # noqa: BLE001 - the normal path writes the answer instead
        logger.info("speculative synthesis failed for turn %s: %s", key, exc)
        return None
    logger.info("speculative synthesis used for turn %s", key)
    return answer


def drop(key: Optional[str]) -> None:
    """Forget this turn's draft (the decider chose another step)."""
    if not key:
        return
    with _lock:
        _drafts.pop(key, None)


__all__ = ["ENV", "drop", "enabled", "fingerprint", "start", "take"]
