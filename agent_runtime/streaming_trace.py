"""Runtime trace hooks for streamed agent observability.

The Flask SSE endpoint consumes events from a queue while the LangChain agent
runs in a worker thread.  This module provides the context-local bridge between
LangChain callbacks, MCP tool wrappers, and that queue.
"""

from __future__ import annotations

import json
import re
import time
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Dict, Iterator, List, Optional

from langchain_core.callbacks import BaseCallbackHandler

from agent_runtime import turn_control

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
        # A code run's heartbeat (elapsed time, the concurrent review's state) and that review's
        # verdict. Status tier: someone waiting on a three-minute run needs to see it is alive
        # whether or not they asked for the detailed trace.
        "code_progress",
        "code_review",
        "turn",
        "cancelled",
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


_EXCEPTION_LINE = re.compile(
    r"^(?:[A-Za-z_][\w.]*\.)?[A-Z]\w*(?:Error|Exception|Exit|Interrupt|Fault)\b\s*(?::.*)?$")
_RAISED_LINE = re.compile(r"^[A-Za-z_][\w.]*(?::\s.*)?$")
_WARNING_LINE = re.compile(r"\b[A-Z]\w*Warning\b")


def _exception_line(text: str) -> Optional[str]:
    """The line of a failure's text that names the error, or None.

    stderr is read top-down, but a Python failure names its error on the LAST line, after the
    traceback — and whatever printed first is often a library warning. On 2026-10-15 every
    failed execute_code in the Chicago SKATER step showed "failed — FutureWarning: …" (a
    libpysal deprecation printed before the code even ran), so the agent's retries were read
    as the same warning again and the error that mattered was never visible.
    """
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    for ln in reversed(lines):
        if _EXCEPTION_LINE.match(ln):
            return ln
    # After a traceback, the last `Name: message` line is the error whatever it is called
    # (pyarrow's ArrowInvalid, a library's own class).
    if any(ln.startswith("Traceback (most recent call last)") for ln in lines):
        for ln in reversed(lines):
            if _RAISED_LINE.match(ln) and not _WARNING_LINE.search(ln):
                return ln
    if _WARNING_LINE.search(lines[0]):
        rest = [ln for ln in lines if not _WARNING_LINE.search(ln) and not ln.startswith(
            ("warnings.warn", "File \"", "^"))]
        if rest:
            return rest[-1]
    return None


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
        text = reason if isinstance(reason, str) else _short_text(reason or "failed", limit=600)
        exc = _exception_line(text)
        if exc:
            # The error first, where the clamped row shows it; the rest after it, for the expand.
            return f"failed — {_short_text(exc + ' · ' + text, limit=600)}"
        return f"failed — {_short_text(text or 'failed', limit=600)}"

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
            pass
        # str() of a dict is a Python literal, not JSON; read it as one before giving up.
        try:
            import ast
            value = ast.literal_eval(stripped)
            if isinstance(value, (dict, list)):
                return _json_safe(value)
        except Exception:
            pass
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
        "ChatOpenAI started" while qwen3.8:27b or gpt-oss:120b did the work. That is the
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
        turn_control.raise_if_cancelled()   # Stop takes effect at the next step
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
        turn_control.raise_if_cancelled()   # Stop takes effect at the next step
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
                    # The calls themselves are surfaced once, as `tool_call` events from
                    # ``on_tool_start``. The TEXT beside them is the model saying what it is
                    # about to do and why ("I'll inspect both files to understand their
                    # structure…"): deepseek wrote one in 6 of 6 probes, 2026-10-17. It was
                    # dropped here with the duplicate calls, so the trace showed bare function
                    # calls with no reason. It is the step's purpose, emitted before its calls.
                    plan = content.strip()
                    if plan:
                        names = [str((c or {}).get("name") or "") for c in raw_tool_calls
                                 if isinstance(c, dict)]
                        self._emit("tool_plan", {"kind": "tool_plan", "label": "Plan",
                                                 "message": _short_text(plan, limit=400),
                                                 "tools": [n for n in names if n]})
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
        turn_control.raise_if_cancelled()   # Stop takes effect at the next step
        tool_name = str((serialized or {}).get("name") or kwargs.get("name") or "unknown_tool")
        # `inputs` is the call's own argument dict. `input_str` is str() of it, a Python repr
        # ("{'file_id': …}") that JSON cannot read, so the client received the arguments as
        # one opaque string and could not say which file a call was about.
        inputs = kwargs.get("inputs")
        args = _normalize_tool_args(inputs if isinstance(inputs, dict) else input_str)
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
            from agent_runtime.client_profile import chat_only
            from agent_runtime.map_layers import build_map_layers

            # A chat-only client draws nothing, so no layer is sent to it (client_profile).
            for layer in ([] if chat_only() else build_map_layers(tool_name, output)):
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


class UsageCallbackHandler(BaseCallbackHandler):
    """One `llm_usage` trace event per model call: the provider's own token counts.

    `turn_instrumentation` records tokens only for the peers' agent loops, because it is
    middleware on `create_agent`. The supervisor's decider, the synthesis and the grounding
    audit call the model directly and never pass through it. On a 2026-10-08 probe turn the
    stream showed 2 model calls while the turn made at least 4. A cost that counts only the
    peers is an undercount whose size depends on the route.

    So this handler is NOT attached per call site. `trace_context` puts it in a context
    variable that LangChain's configure hook reads, and every runnable configured in that
    context, at any depth and on any thread that copied the context, gets it. A model call
    that nobody remembered to wire still reports.
    """

    raise_error = False
    run_inline = True

    def __init__(self) -> None:
        super().__init__()
        self._models: Dict[str, str] = {}
        # Each run's endpoint, so a call can be recognized as Lumen's by WHERE it went: a model id
        # like gpt-4o can exist on Lumen and on OpenAI, billing different accounts in different
        # currencies. Only the base URL is kept; serialized kwargs also carry the API key slot.
        self._bases: Dict[str, str] = {}
        self._lock = Lock()
        # Every call of the turn, priced, so the turn can report what it cost
        # (model_pricing.summarize). One handler per turn, so this never mixes turns.
        self.calls: List[Dict[str, Any]] = []

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: Any, **kwargs: Any) -> None:
        params = kwargs.get("invocation_params") or {}
        base = (((serialized or {}).get("kwargs") or {}).get("openai_api_base")
                or params.get("base_url") or params.get("openai_api_base"))
        with self._lock:
            run = str(kwargs.get("run_id"))
            self._models[run] = StreamingTraceCallbackHandler._model_label(serialized, kwargs)
            if isinstance(base, str) and base:
                self._bases[run] = base

    def on_llm_start(self, serialized: Dict[str, Any], prompts: Any, **kwargs: Any) -> None:
        self.on_chat_model_start(serialized, prompts, **kwargs)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        with self._lock:
            model = self._models.pop(str(kwargs.get("run_id")), None)
            base = self._bases.pop(str(kwargs.get("run_id")), None)
        usage: Dict[str, Any] = {}
        for group in getattr(response, "generations", None) or []:
            for generation in group or []:
                um = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if isinstance(um, dict) and um:
                    usage = um
                    break
            if usage:
                break
        if not usage:
            # Providers that report only on the legacy field (llm_output.token_usage).
            legacy = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
            if legacy:
                usage = {"input_tokens": legacy.get("prompt_tokens"),
                         "output_tokens": legacy.get("completion_tokens")}
        payload: Dict[str, Any] = {"kind": "llm_usage", "label": "LLM usage", "model": model}
        if usage:
            payload["input_tokens"] = usage.get("input_tokens")
            payload["output_tokens"] = usage.get("output_tokens")
            details_in = usage.get("input_token_details") or {}
            details_out = usage.get("output_token_details") or {}
            if details_in.get("cache_read"):
                payload["cached_input_tokens"] = details_in.get("cache_read")
            if details_out.get("reasoning"):
                payload["reasoning_tokens"] = details_out.get("reasoning")
        else:
            # "The provider sent no usage" is a different fact from "the call used nothing".
            payload["usage"] = "absent"
        if usage:
            # What this call cost, priced as it ends, so each step of the turn carries its own
            # cost and not only the total. A Lumen call is billed in coins at Lumen's published
            # rate (lumen_coins) and never at an OpenAI dollar rate for the same model id.
            lumen = False
            try:
                from agent_runtime import lumen_coins

                lumen = lumen_coins.is_lumen(base)
                if lumen:
                    payload["provider"] = "lumen"
                    payload["cost_coins"] = lumen_coins.call_coins(
                        model, payload.get("input_tokens"), payload.get("output_tokens"))
            except Exception:  # noqa: BLE001 - pricing must never break a model call
                payload["cost_coins"] = None
            try:
                from agent_runtime.model_pricing import call_cost

                payload["cost_usd"] = None if lumen else call_cost(
                    model, payload.get("input_tokens"), payload.get("output_tokens"),
                    payload.get("cached_input_tokens"))
            except Exception:  # noqa: BLE001 - pricing must never break a model call
                payload["cost_usd"] = None
        agent = _TRACE_AGENT.get()
        with self._lock:
            self.calls.append({**payload, **({"agent": agent} if agent else {})})
        emit_trace_event("llm_usage", payload)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        with self._lock:
            self._models.pop(str(kwargs.get("run_id")), None)
            self._bases.pop(str(kwargs.get("run_id")), None)


_USAGE_HANDLER: ContextVar[Optional[UsageCallbackHandler]] = ContextVar(
    "agent_stream_usage_handler", default=None)

try:  # pragma: no cover - exercised by the import itself
    from langchain_core.tracers.context import register_configure_hook

    register_configure_hook(_USAGE_HANDLER, inheritable=True)
except Exception:  # noqa: BLE001 - usage accounting must never break the agent
    logger.warning("llm_usage events disabled: configure hook unavailable", exc_info=True)


@contextmanager
def usage_collection() -> Iterator[UsageCallbackHandler]:
    """Collect this context's model calls, for a turn run outside ``trace_context``.

    The streaming path already installs a handler in ``trace_context``; the blocking
    ``/agent/chat`` path does not, and would otherwise report no cost. Nested use reuses the
    handler already active, so a turn is never split across two."""
    existing = _USAGE_HANDLER.get()
    if existing is not None:
        yield existing
        return
    handler = UsageCallbackHandler()
    token = _USAGE_HANDLER.set(handler)
    try:
        yield handler
    finally:
        _USAGE_HANDLER.reset(token)


def turn_usage() -> Optional[Dict[str, Any]]:
    """This turn's cost so far (model_pricing.summarize), or None outside a turn."""
    handler = _USAGE_HANDLER.get()
    if handler is None:
        return None
    try:
        from agent_runtime.model_pricing import summarize

        with handler._lock:
            calls = list(handler.calls)
        balance = None
        if any(c.get("provider") == "lumen" for c in calls):
            # Once per turn, at its end: what the Lumen account has left after this turn.
            from agent_runtime import lumen_coins

            balance = lumen_coins.coins_available()
            try:
                from agent_runtime import lumen_failover

                lumen_failover.observe(balance)
            except Exception:  # noqa: BLE001 - the failover check must never break a turn
                logger.warning("Lumen failover observe failed", exc_info=True)
        return summarize(calls, coins_available=balance)
    except Exception:  # noqa: BLE001 - a cost summary must never break a turn
        logger.warning("turn usage summary failed", exc_info=True)
        return None


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
    usage_token = _USAGE_HANDLER.set(UsageCallbackHandler())
    try:
        yield
    finally:
        _USAGE_HANDLER.reset(usage_token)
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
