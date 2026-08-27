"""Runtime trace hooks for streamed agent observability.

The Flask SSE endpoint consumes events from a queue while the LangChain agent
runs in a worker thread.  This module provides the context-local bridge between
LangChain callbacks, MCP tool wrappers, and that queue.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Dict, Iterator, Optional

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)


TraceSink = Callable[[Dict[str, Any]], None]


# ---------------------------------------------------------------------------
# AGENT_DEV verbosity gate
# ---------------------------------------------------------------------------
# When AGENT_DEV is off (default) the stream only carries coarse execution-state
# *status* events suitable as user-facing references.  When on, detailed
# input/output events (tool args, tool results, LLM interactions, routing
# decisions) are also surfaced for debugging.

_STATUS_TIER_EVENTS = frozenset(
    {
        "status",
        "subagent_started",
        "subagent_completed",
        "node_started",
        "node_completed",
        "search_complete",
        "grounding_audit",
        "final_answer",
        "completed",
        "error",
        # Plottable geometry from geo tools (e.g. overpass_search). Status-tier so a
        # map client receives it even when detail-tier tracing (agent_dev) is off.
        "map_layer",
    }
)


def is_agent_dev() -> bool:
    """Return True when detailed agent I/O should be surfaced via SSE.

    Controlled by the ``AGENT_DEV`` environment variable (truthy: 1/true/yes/on).
    """
    return (os.getenv("AGENT_DEV") or "").strip().lower() in {"1", "true", "yes", "on"}


def is_status_tier_event(event: str) -> bool:
    """Whether *event* is a coarse status event that is always emitted."""
    return event in _STATUS_TIER_EVENTS


@dataclass
class _TraceState:
    sink: TraceSink
    handler: Any
    agent_role: str
    sequence: int = 0
    # Per-request override for detail-tier verbosity. None -> fall back to the
    # AGENT_DEV env var; True/False -> force on/off for this stream.
    agent_dev: Optional[bool] = None


_TRACE_STATE: ContextVar[Optional[_TraceState]] = ContextVar("agent_stream_trace_state", default=None)
_TRACE_AGENT: ContextVar[str] = ContextVar("agent_stream_trace_agent", default="agent")


# Detail-tier text/JSON truncation. Defaults preserve production behavior; a full-trace
# capture run can raise them via AGENT_TRACE_TEXT_LIMIT / AGENT_TRACE_JSON_LIMIT (read at
# import, so set the env before importing this module).
_TEXT_LIMIT = int(os.environ.get("AGENT_TRACE_TEXT_LIMIT") or 1200)
_JSON_LIMIT = int(os.environ.get("AGENT_TRACE_JSON_LIMIT") or 3000)


def _short_text(value: Any, *, limit: Optional[int] = None) -> str:
    if value is None:
        return ""
    limit = _TEXT_LIMIT if limit is None else limit
    text = value if isinstance(value, str) else str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else f"{text[:limit]}..."


def _json_safe(value: Any, *, limit: Optional[int] = None) -> Any:
    if limit is None:
        limit = _JSON_LIMIT
    try:
        text = json.dumps(value, ensure_ascii=True, default=str)
    except Exception:
        return _short_text(value, limit=limit)
    if len(text) <= limit:
        try:
            return json.loads(text)
        except Exception:
            return text
    return f"{text[:limit]}..."


def _message_content(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
            else:
                parts.append(str(getattr(part, "text", part)))
        content = "".join(parts)
    return str(content or "")


def _normalize_tool_args(raw: Any) -> Any:
    if isinstance(raw, dict):
        return _json_safe(raw)
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return {}
        try:
            return _json_safe(json.loads(stripped))
        except Exception:
            return stripped
    return _json_safe(raw)


def _normalize_tool_call(call: Any) -> Dict[str, Any]:
    if isinstance(call, dict):
        name = call.get("name") or call.get("tool") or call.get("function", {}).get("name")
        args = call.get("args")
        if args is None and isinstance(call.get("function"), dict):
            args = call["function"].get("arguments")
        return {
            "name": str(name or "unknown_tool"),
            "args": _normalize_tool_args(args),
            "id": call.get("id") or call.get("tool_call_id"),
        }
    return {
        "name": str(getattr(call, "name", None) or "unknown_tool"),
        "args": _normalize_tool_args(getattr(call, "args", None)),
        "id": getattr(call, "id", None),
    }


def current_agent_role() -> str:
    return _TRACE_AGENT.get() or "agent"


def _emit_with_state(
    state: Optional[_TraceState],
    event: str,
    data: Optional[Dict[str, Any]] = None,
    *,
    agent_role: Optional[str] = None,
    node: Optional[str] = None,
) -> None:
    if state is None:
        return
    # Detail-tier events are suppressed unless dev mode is enabled. The
    # per-request flag on the trace state wins; otherwise fall back to AGENT_DEV.
    dev_enabled = state.agent_dev if state.agent_dev is not None else is_agent_dev()
    if not is_status_tier_event(event) and not dev_enabled:
        return

    payload: Dict[str, Any] = dict(data or {})
    context_role = current_agent_role()
    role = agent_role or payload.get("agent") or (context_role if context_role != "agent" else state.agent_role)
    if role and "agent" not in payload:
        payload["agent"] = role
    if "sequence" not in payload:
        state.sequence += 1
        payload["sequence"] = state.sequence

    item: Dict[str, Any] = {"event": event, "data": payload}
    if role:
        item["agent_role"] = role
    if node:
        item["node"] = node

    try:
        state.sink(item)
    except Exception:
        logger.debug("Failed to emit streamed trace event %s", event, exc_info=True)


def emit_trace_event(
    event: str,
    data: Optional[Dict[str, Any]] = None,
    *,
    agent_role: Optional[str] = None,
    node: Optional[str] = None,
) -> None:
    """Emit one trace event to the active stream, if there is one."""
    _emit_with_state(_TRACE_STATE.get(), event, data, agent_role=agent_role, node=node)


class StreamingTraceCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that forwards live LLM/tool events to SSE."""

    run_inline = True
    raise_error = False
    ignore_llm = False
    ignore_chat_model = False
    ignore_chain = False
    ignore_agent = False
    ignore_retriever = False
    ignore_retry = False
    ignore_custom_event = False

    def __init__(self) -> None:
        super().__init__()
        self._state: Optional[_TraceState] = None
        self._tool_runs: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()

    def _tool_run_key(self, run_id: Any) -> str:
        return str(run_id or "")

    def _emit(self, event: str, data: Optional[Dict[str, Any]] = None) -> None:
        _emit_with_state(self._state or _TRACE_STATE.get(), event, data)

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: Any, **kwargs: Any) -> None:
        name = (serialized or {}).get("name") or (serialized or {}).get("id") or "chat_model"
        message_count = sum(len(group or []) for group in messages or []) if isinstance(messages, list) else None
        self._emit(
            "llm_start",
            {
                "kind": "llm_start",
                "label": "LLM request",
                "message": f"{name} started" + (f" with {message_count} message(s)" if message_count else ""),
                "model": name,
            },
        )

    def on_llm_start(self, serialized: Dict[str, Any], prompts: Any, **kwargs: Any) -> None:
        name = (serialized or {}).get("name") or (serialized or {}).get("id") or "llm"
        self._emit(
            "llm_start",
            {
                "kind": "llm_start",
                "label": "LLM request",
                "message": f"{name} started",
                "model": name,
                "prompt_count": len(prompts or []) if isinstance(prompts, list) else None,
            },
        )

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        generations = getattr(response, "generations", None) or []
        for group in generations:
            for generation in group or []:
                message = getattr(generation, "message", None)
                content = _message_content(message) if message is not None else str(getattr(generation, "text", "") or "")
                raw_tool_calls = getattr(message, "tool_calls", None) if message is not None else None
                if isinstance(raw_tool_calls, list) and raw_tool_calls:
                    # Tool decisions are surfaced once, as `tool_call` events from
                    # ``on_tool_start``. Emitting them here too made every decision
                    # appear twice in the stream — skip the redundant copy.
                    continue
                if content.strip():
                    self._emit(
                        "llm_interaction",
                        {
                            "kind": "llm_message",
                            "label": "LLM message",
                            "content": _short_text(content),
                            "message": _short_text(content),
                        },
                    )

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self._emit(
            "llm_error",
            {
                "kind": "llm_error",
                "label": "LLM error",
                "message": f"{type(error).__name__}: {error}",
            },
        )

    def on_tool_start(self, serialized: Dict[str, Any], input_str: Any, **kwargs: Any) -> None:
        tool_name = str((serialized or {}).get("name") or kwargs.get("name") or "unknown_tool")
        args = _normalize_tool_args(input_str)
        run_key = self._tool_run_key(kwargs.get("run_id"))
        with self._lock:
            self._tool_runs[run_key] = {"name": tool_name, "args": args}
        self._emit(
            "tool_call",
            {
                "kind": "llm_tool_decision",
                "label": "Tool started",
                "name": tool_name,
                "args": args,
                "tool_calls": [{"name": tool_name, "args": args}],
                "message": f"{tool_name}({json.dumps(args or {}, ensure_ascii=True, default=str)})",
            },
        )

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        run_key = self._tool_run_key(kwargs.get("run_id"))
        with self._lock:
            meta = self._tool_runs.pop(run_key, {})
        tool_name = str(meta.get("name") or kwargs.get("name") or "unknown_tool")
        self._emit(
            "tool_result",
            {
                "kind": "tool_result",
                "label": f"Tool result {tool_name}",
                "tool_name": tool_name,
                "name": tool_name,
                "content": _short_text(output),
                "message": _short_text(output),
            },
        )
        # Geometry-bearing results (e.g. overpass_search) also stream as an untruncated
        # `map_layer` event so a map client can plot them live; the `content` above is
        # truncated and not reliably parseable.
        try:
            from agent_runtime.map_layers import build_map_layers

            for layer in build_map_layers(tool_name, output):
                self._emit("map_layer", layer)
        except Exception:  # pragma: no cover - never let map extraction break the stream
            logger.debug("map_layer extraction failed for %s", tool_name, exc_info=True)

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        run_key = self._tool_run_key(kwargs.get("run_id"))
        with self._lock:
            meta = self._tool_runs.pop(run_key, {})
        tool_name = str(meta.get("name") or kwargs.get("name") or "unknown_tool")
        self._emit(
            "tool_error",
            {
                "kind": "tool_error",
                "label": f"Tool error {tool_name}",
                "tool_name": tool_name,
                "name": tool_name,
                "message": f"{type(error).__name__}: {error}",
            },
        )


def active_callback_handler() -> Optional[StreamingTraceCallbackHandler]:
    state = _TRACE_STATE.get()
    return state.handler if state is not None else None


def attach_streaming_callbacks(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Attach the active trace callback handler to a LangChain config."""
    merged: Dict[str, Any] = {**(config or {})}
    handler = active_callback_handler()
    if handler is None:
        return merged
    callbacks = list(merged.get("callbacks") or [])
    if handler not in callbacks:
        callbacks.append(handler)
    merged["callbacks"] = callbacks
    return merged


@contextmanager
def trace_context(
    sink: TraceSink,
    *,
    agent_role: str = "orchestrator_agent",
    agent_dev: Optional[bool] = None,
) -> Iterator[None]:
    """Enable streamed trace emission for the current thread/context.

    ``agent_dev`` overrides detail-tier verbosity for this stream (None falls
    back to the ``AGENT_DEV`` env var).
    """
    handler = StreamingTraceCallbackHandler()
    state = _TraceState(sink=sink, handler=handler, agent_role=agent_role, agent_dev=agent_dev)
    handler._state = state
    state_token = _TRACE_STATE.set(state)
    agent_token = _TRACE_AGENT.set(agent_role)
    try:
        yield
    finally:
        _TRACE_AGENT.reset(agent_token)
        _TRACE_STATE.reset(state_token)


@contextmanager
def trace_agent(agent_role: str) -> Iterator[None]:
    """Temporarily label emitted trace events with an agent role."""
    token = _TRACE_AGENT.set(agent_role)
    try:
        yield
    finally:
        _TRACE_AGENT.reset(token)


__all__ = [
    "attach_streaming_callbacks",
    "current_agent_role",
    "emit_trace_event",
    "is_agent_dev",
    "is_status_tier_event",
    "trace_agent",
    "trace_context",
]
