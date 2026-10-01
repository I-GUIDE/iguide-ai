"""Runtime trace hooks for streamed agent observability.

The Flask SSE endpoint consumes events from a queue while the LangChain agent
runs in a worker thread.  This module provides the context-local bridge between
LangChain callbacks, MCP tool wrappers, and that queue.
"""

from __future__ import annotations

import json
import time
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
    # A second sink that receives EVERY event, before the detail-tier filter below. The client's
    # stream and the durable record answer different questions: a viewer asked for a readable
    # trace, while a record exists to reproduce the turn later, and a record that only holds what
    # someone happened to switch on is not a record. Everything the client sees, the recorder
    # also sees; the reverse is not true.
    recorder: Optional[TraceSink] = None


_TRACE_STATE: ContextVar[Optional[_TraceState]] = ContextVar("agent_stream_trace_state", default=None)
_TRACE_AGENT: ContextVar[str] = ContextVar("agent_stream_trace_agent", default="agent")


# Detail-tier text/JSON truncation. Defaults preserve production behavior; a full-trace
# capture run can raise them via AGENT_TRACE_TEXT_LIMIT / AGENT_TRACE_JSON_LIMIT (read at
# import, so set the env before importing this module).
_TEXT_LIMIT = int(os.environ.get("AGENT_TRACE_TEXT_LIMIT") or 1200)
_JSON_LIMIT = int(os.environ.get("AGENT_TRACE_JSON_LIMIT") or 3000)


def _outcome(output: Any) -> Optional[str]:
    """What a tool RETURNED, in a few words, or None when it cannot be said briefly.

    The trace showed that a tool was called and never what came back, so a search finding eight
    documents, a search finding none, and a search that failed all rendered as the same single
    line. Two wrong diagnoses in one afternoon came out of that: a tool failing in a second and
    being retried looked exactly like the same tool running twice.

    Deliberately a HEADLINE, not the payload — the truncated result content is already available
    to anyone who wants it, and a trace that prints result bodies is the noise this line has to
    stay clear of. Reads the fields the tools already set; returns None rather than inventing a
    summary for a shape it does not recognise, in which case the caller still has the duration.
    """
    # Unwrap first. LangChain hands on_tool_end a ToolMessage in some versions and the raw
    # string in others; _short_text stringifies either, which is why `content` was right while
    # the outcome came back empty and the trace line showed a duration and nothing else.
    body = getattr(output, "content", output)
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8", "replace")
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except ValueError:
            return None
    if not isinstance(body, dict):
        return None

    # A failure first, and never silently: this is the case the line exists for. Four shapes,
    # because the tools do not agree on one — execute_code reports an exit code and a timeout
    # flag, the service tools an `ok` flag, and some return a bare `error`.
    failed = (body.get("ok") is False
              or body.get("timed_out") is True
              or (body.get("exit_code") not in (None, 0))
              or (body.get("error") and body.get("ok") is not True))
    if failed:
        reason = body.get("error") or body.get("detail") or body.get("stderr") or body.get("hint")
        if not reason and body.get("timed_out"):
            reason = "timed out"
        if not reason and body.get("exit_code") not in (None, 0):
            reason = f"exit code {body['exit_code']}"
        # 600, not 140: the row CLAMPS at 140 in the transcript and expands on click, so the
        # cap here decides what there is to expand INTO. At 140 a python traceback lost the
        # last frame — the one naming the error — which is the only part worth reading.
        return f"failed — {_short_text(reason or 'failed', limit=600)}"

    # `count` is what every search tool's _build_payload already reports.
    for key in ("count", "feature_count", "zones_with_pixels", "row_count"):
        value = body.get(key)
        if isinstance(value, int):
            noun = {"count": "result", "feature_count": "feature",
                    "zones_with_pixels": "zone with pixels", "row_count": "row"}[key]
            return f"{value:,} {noun}{'' if value == 1 else 's'}"
    for key in ("documents", "results", "matched"):
        value = body.get(key)
        if isinstance(value, list):
            return f"{len(value):,} {key.rstrip('s')}{'' if len(value) == 1 else 's'}"

    layers = body.get("map_layers")
    if isinstance(layers, list) and layers:
        return f"{len(layers)} layers on the map"
    if isinstance(body.get("map_layer"), dict):
        return "1 layer on the map"
    if body.get("filename"):
        return _short_text(body["filename"], limit=60)
    if body.get("ok") is True:
        return "ok"
    return None


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

    # The recorder runs FIRST and unfiltered. Sequence numbers are assigned above, so both sinks
    # agree on ordering even though the client sees a subset.
    if state.recorder is not None:
        try:
            state.recorder(item)
        except Exception:
            logger.debug("Trace recorder rejected event %s", event, exc_info=True)

    dev_enabled = state.agent_dev if state.agent_dev is not None else is_agent_dev()
    if not is_status_tier_event(event) and not dev_enabled:
        return

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
        # tool name -> {"error": str, "attempts": int}. Per HANDLER, which is per turn, so a
        # failure never colours a later conversation. Cleared when the tool succeeds.
        self._tool_failures: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()

    def _tool_run_key(self, run_id: Any) -> str:
        return str(run_id or "")

    def _emit(self, event: str, data: Optional[Dict[str, Any]] = None) -> None:
        _emit_with_state(self._state or _TRACE_STATE.get(), event, data)

    @staticmethod
    def _model_label(serialized: Optional[Dict[str, Any]], kwargs: Dict[str, Any]) -> str:
        """Which MODEL is answering, not which LangChain class wraps it.

        serialized["name"] is the class, and AnvilGPT, vLLM and any other
        OpenAI-compatible endpoint all arrive as ChatOpenAI — so the trace read
        "ChatOpenAI started" while qwen3.6:27b or gpt-oss:120b did the work. That is the
        same confusion active_llm_description() exists to prevent: the transport does not
        tell you who answered. The invocation params carry the id the user actually picked.
        """
        params = kwargs.get("invocation_params") or {}
        meta = kwargs.get("metadata") or {}
        for value in (params.get("model"), params.get("model_name"),
                      meta.get("ls_model_name")):
            if value:
                return str(value)
        return (serialized or {}).get("name") or (serialized or {}).get("id") or "chat_model"

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: Any, **kwargs: Any) -> None:
        name = self._model_label(serialized, kwargs)
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
        name = self._model_label(serialized, kwargs)
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
            self._tool_runs[run_key] = {"name": tool_name, "args": args,
                                        "started": time.monotonic()}
            prior = self._tool_failures.get(tool_name)
        # THE REPAIR, said out loud. A tool that fails and is immediately retried is the single
        # most misleading thing this trace could show, because two calls of one tool render
        # identically whether the first worked or not — that is exactly how a 1.6-second
        # rejection read as a duplicate tile sweep for two rounds of diagnosis. One retry is
        # also below the dead-end detector's threshold of two, so nothing else reports it.
        if prior:
            self._emit(
                "tool_retry",
                {"kind": "tool_retry", "label": "Retrying", "name": tool_name,
                 "attempt": prior["attempts"] + 1,
                 "message": f"retrying {tool_name} after: "
                            f"{_short_text(prior['error'], limit=600)}"},
            )
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
        # The HEADLINE goes out beside the content: `outcome` is what the trace line says, and
        # `duration_s` is what makes a fast failure distinguishable from a real run — the two
        # facts that were missing when a 1.6-second rejection read as a duplicate tile sweep.
        outcome = _outcome(output)
        started = meta.get("started")
        duration = round(time.monotonic() - started, 2) if isinstance(started, float) else None
        payload: Dict[str, Any] = {
            "kind": "tool_result",
            "label": f"Tool result {tool_name}",
            "tool_name": tool_name,
            "name": tool_name,
            "content": _short_text(output),
            "message": _short_text(output),
        }
        if outcome:
            payload["outcome"] = outcome
        if duration is not None:
            payload["duration_s"] = duration
        self._emit("tool_result", payload)

        # Track the failure/repair pair so the NEXT call can name what it is retrying, and so a
        # success after a failure is reported as a recovery rather than passing silently. A turn
        # that quietly needed two attempts is a turn whose tool contract is wrong, and that is
        # worth seeing: one retry sits below the dead-end detector's threshold of two.
        failed = bool(outcome and outcome.startswith("failed"))
        with self._lock:
            prior = self._tool_failures.get(tool_name)
            if failed:
                self._tool_failures[tool_name] = {
                    "error": (outcome or "")[len("failed — "):] or "failed",
                    "attempts": (prior or {}).get("attempts", 0) + 1}
            elif prior:
                self._tool_failures.pop(tool_name, None)
        if not failed and prior:
            attempts = prior["attempts"] + 1
            self._emit(
                "tool_recovered",
                {"kind": "tool_recovered", "label": "Recovered", "name": tool_name,
                 "attempts": attempts,
                 "message": f"{tool_name} succeeded on attempt {attempts} — "
                            f"the first failed with: {_short_text(prior['error'], limit=400)}"},
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
    recorder: Optional[TraceSink] = None,
) -> Iterator[None]:
    """Enable streamed trace emission for the current thread/context.

    ``agent_dev`` overrides detail-tier verbosity for this stream (None falls
    back to the ``AGENT_DEV`` env var). ``recorder``, when given, receives every
    event regardless of that setting -- see ``_TraceState.recorder``.
    """
    handler = StreamingTraceCallbackHandler()
    state = _TraceState(sink=sink, handler=handler, agent_role=agent_role, agent_dev=agent_dev,
                        recorder=recorder)
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
