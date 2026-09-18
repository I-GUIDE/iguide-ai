from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Mapping, Optional, Sequence
from uuid import uuid4

from .file_store import get_file_record
from .graph_runtime import run_agent_query, stream_agent_query_events
from .runtime_utils import sanitize_answer_links
from .session_memory import (
    append_session_files,
    append_session_turn,
    build_session_memory_doc,
    get_session_files,
)
from rag_pipeline.memory_module import (create_memory, get_or_create_memory, save_turn_trace,
                                        update_memory)

logger = logging.getLogger(__name__)


def _coerce_recent_history(chat_history: Sequence[Mapping[str, Any]], recent_k: Optional[int]) -> Sequence[Mapping[str, Any]]:
    if recent_k is None:
        return chat_history
    if recent_k <= 0:
        return []
    return chat_history[-recent_k:]


def _build_chat_history(memory_doc: Optional[Mapping[str, Any]], recent_k: Optional[int] = None) -> List[Dict[str, str]]:
    history = (memory_doc or {}).get("chat_history") or []
    selected = _coerce_recent_history(history, recent_k)
    messages: List[Dict[str, str]] = []
    for entry in selected:
        user_query = str(entry.get("userQuery") or "").strip()
        answer = str(entry.get("answer") or "").strip()
        if user_query:
            messages.append({"role": "user", "content": user_query})
        if answer:
            messages.append({"role": "assistant", "content": answer})
    return messages


def _extract_agent_answer(result: Mapping[str, Any]) -> str:
    final_answer = result.get("final_answer")
    if isinstance(final_answer, str) and final_answer.strip():
        return final_answer.strip()

    analysis_result = result.get("analysis_result")
    if isinstance(analysis_result, Mapping):
        messages = analysis_result.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                if isinstance(message, Mapping):
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        return content.strip()
                content = getattr(message, "content", None)
                if isinstance(content, str) and content.strip():
                    return content.strip()
    return ""


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    content = getattr(value, "content", None)
    if content is not None:
        payload: Dict[str, Any] = {
            "type": value.__class__.__name__,
            "content": _json_safe(content),
        }
        name = getattr(value, "name", None)
        if name:
            payload["name"] = str(name)
        tool_call_id = getattr(value, "tool_call_id", None)
        if tool_call_id:
            payload["tool_call_id"] = str(tool_call_id)
        tool_calls = getattr(value, "tool_calls", None)
        if tool_calls:
            payload["tool_calls"] = _json_safe(tool_calls)
        return payload

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _json_safe(model_dump())
        except Exception:
            pass

    return str(value)


def _extract_opengeodata_results(agent_result: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Project the OpenGeoData hits out of the run's evidence so the client receives them as
    structured JSON objects (title, url, source, provider, bbox, links, ...) alongside the
    markdown answer. Returns [] when the run used no OpenGeoData (or a non-supervisor path).

    Never raises: this runs while building the terminal streaming response, so any failure here
    must degrade to [] rather than abort the stream (which would deprive the client of its
    result event).
    """
    try:
        if not isinstance(agent_result, Mapping):
            return []
        orch = agent_result.get("orchestration_result")
        evidence = orch.get("evidence") if isinstance(orch, Mapping) else None
        results: List[Dict[str, Any]] = []
        for doc in (evidence or []):
            if not isinstance(doc, Mapping):
                continue
            src = doc.get("document") if isinstance(doc.get("document"), Mapping) else doc
            etype = str(src.get("element_type") or src.get("resource-type") or "").strip().lower()
            srcname = str(src.get("source") or src.get("source_system") or "").strip().lower()
            if etype == "opengeodata" or srcname == "opengeodata":
                results.append(_json_safe(dict(src)))
        return results
    except Exception as exc:  # pragma: no cover - defensive; must not break the response
        logger.warning("Failed to extract opengeodata_results: %s", exc)
        return []


def _normalize_file_paths(file_paths: Optional[Sequence[Any]]) -> List[str]:
    if isinstance(file_paths, (str, bytes)):
        file_paths = [file_paths]
    normalized: List[str] = []
    for value in file_paths or []:
        text = str(value or "").strip()
        if not text:
            continue
        try:
            normalized.append(str(Path(text).expanduser()))
        except Exception:
            normalized.append(text)
    return normalized


def _augment_user_input_with_files(user_input: str, file_paths: Sequence[str]) -> str:
    if not file_paths:
        return user_input
    attachment_lines = "\n".join(f"- {path}" for path in file_paths)
    return (
        f"{user_input}\n\n"
        "Attached files are available to the agent via local file tools. "
        "Use the provided paths if file inspection is needed:\n"
        f"{attachment_lines}"
    )


def _normalize_file_ids(file_ids: Optional[Sequence[Any]]) -> List[str]:
    if isinstance(file_ids, (str, bytes)):
        file_ids = [file_ids]
    normalized: List[str] = []
    for value in file_ids or []:
        text = str(value or "").strip()
        if text:
            normalized.append(text)
    return normalized


def _augment_user_input_with_file_ids(user_input: str, file_ids: Sequence[str]) -> str:
    if not file_ids:
        return user_input

    lines: List[str] = []
    for file_id in file_ids:
        record = get_file_record(file_id)
        if record:
            lines.append(f"- {file_id} ({record.get('filename', 'unknown')})")
        else:
            lines.append(f"- {file_id}")
    attachment_lines = "\n".join(lines)
    return (
        f"{user_input}\n\n"
        "Uploaded files are available to the agent via local file tools. "
        "Use the exact uploaded file ids, not the display filenames, when inspecting files. "
        "To read an uploaded file inside `execute_code`, pass its file_id(s) in the "
        "`input_files` argument; the file is then available in the working directory under "
        "both its file_id and its original filename. "
        "Use `write_output_file` for downloadable outputs:\n"
        f"{attachment_lines}"
    )


def _normalize_enabled_search_methods(enabled_search_methods: Optional[Sequence[Any]]) -> Optional[List[str]]:
    """Canonical retrieval allowlist, or None for 'all methods'.

    Shares agent_runtime.search_methods with the API layer so a name is validated the same way on
    every entry point; an unknown name raises ValueError rather than silently disabling retrieval.
    """
    from agent_runtime.search_methods import normalize_search_methods

    return normalize_search_methods(enabled_search_methods)




def _normalize_skill_roots(skill_roots: Optional[Sequence[Any]]) -> Optional[List[str]]:
    if skill_roots is None:
        return None
    if isinstance(skill_roots, (str, bytes)):
        skill_roots = [item.strip() for item in str(skill_roots).split(",")]
    normalized: List[str] = []
    for value in skill_roots or []:
        text = str(value or "").strip()
        if not text:
            continue
        try:
            normalized.append(str(Path(text).expanduser()))
        except Exception:
            normalized.append(text)
    return normalized



def _llm_for_request(provider: Optional[str], model: Optional[str],
                     reasoning_effort: Optional[str] = None) -> Optional[Any]:
    """A per-request model, or None to let the graph use the process default.

    Absent both, the default is whatever build_default_llm resolves — OpenAI gpt-4o in this
    deployment. A bad provider/model is raised here, at the edge, rather than surfacing as an
    opaque 404 from the provider mid-turn.
    """
    if not provider and not model and not reasoning_effort:
        return None
    from agent_runtime.executor_factory import build_llm, supports_reasoning_effort

    llm = build_llm(provider=provider, model=model, reasoning_effort=reasoning_effort)
    # Log what was actually resolved. Which model answered is otherwise invisible: the
    # transport log shows only the host, and a dropped reasoning_effort looks identical to an
    # applied one.
    logger.info("per-request LLM: provider=%s model=%s reasoning_effort=%s%s",
                provider or "(inferred)", getattr(llm, "model_name", model), reasoning_effort or "-",
                "" if (not reasoning_effort or supports_reasoning_effort(
                    getattr(llm, "model_name", model))) else " (dropped: model does not accept it)")
    return llm


def run_agent_chat(
    *,
    user_input: str,
    thread_id: Optional[str] = None,
    memory_id: Optional[str] = None,
    conversation_name: Optional[str] = None,
    recent_k: Optional[int] = None,
    tool_strategy: str = "granular",
    include_mcp_tools: bool = False,
    mcp_modules: Optional[List[str]] = None,
    enabled_search_methods: Optional[Sequence[Any]] = None,
    use_persistent_memory: bool = True,
    smart_tool_routing: bool = True,
    forced_intent: Optional[str] = None,
    file_paths: Optional[Sequence[Any]] = None,
    file_ids: Optional[Sequence[Any]] = None,
    skill_roots: Optional[Sequence[Any]] = None,
    verbose: bool = False,
    code_exec: Optional[bool] = None,
    code_peer: Optional[str] = None,
    code_peer_model: Optional[str] = None,
    unified_peer: Optional[bool] = None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    effective_thread_id = thread_id or memory_id
    effective_memory_id = memory_id or thread_id
    memory_doc: Optional[Mapping[str, Any]] = None
    memory_warning: Optional[str] = None
    normalized_file_paths = _normalize_file_paths(file_paths)
    normalized_file_ids = _normalize_file_ids(file_ids)
    normalized_enabled_search_methods = _normalize_enabled_search_methods(enabled_search_methods)
    normalized_skill_roots = _normalize_skill_roots(skill_roots)

    if use_persistent_memory:
        try:
            if effective_memory_id:
                memory_doc = get_or_create_memory(effective_memory_id)
            else:
                effective_memory_id = create_memory(conversation_name or "agent-chat")
                memory_doc = get_or_create_memory(effective_memory_id)
        except Exception as exc:
            logger.warning("Persistent agent chat memory unavailable: %s", exc)
            memory_warning = f"persistent_memory_unavailable: {exc}"
            effective_memory_id = memory_id
            # Fall back to session-local memory so the turn is not memoryless.
            memory_doc = build_session_memory_doc(effective_thread_id)
    else:
        effective_memory_id = None
        # Persistent memory is off, but conversation context must still be
        # preserved locally within the session (keyed by thread_id).
        memory_doc = build_session_memory_doc(effective_thread_id)

    # Files attached on ANY earlier turn of this session stay accessible: union the
    # session's tracked file_ids with this turn's, so "visualize it" / "execute it"
    # still reach an upload from a previous turn.
    effective_file_ids = list(dict.fromkeys([*get_session_files(effective_thread_id), *normalized_file_ids]))

    chat_history = _build_chat_history(memory_doc, recent_k=recent_k)
    effective_input = _augment_user_input_with_files(user_input, normalized_file_paths)
    effective_input = _augment_user_input_with_file_ids(effective_input, effective_file_ids)

    result = run_agent_query(
        effective_input,
        llm=_llm_for_request(llm_provider, llm_model, reasoning_effort),
        chat_history=chat_history,
        verbose=verbose,
        return_intermediate_steps=True,
        tool_strategy=tool_strategy,
        include_mcp_tools=include_mcp_tools,
        mcp_modules=mcp_modules,
        enabled_search_methods=normalized_enabled_search_methods,
        smart_tool_routing=smart_tool_routing,
        forced_intent=forced_intent,
        thread_id=effective_thread_id,
        skill_roots=normalized_skill_roots,
        code_exec=code_exec,
        code_peer=code_peer,
        code_peer_model=code_peer_model,
        unified_peer=unified_peer,
        input_file_ids=effective_file_ids,
    )

    answer = sanitize_answer_links(_extract_agent_answer(result))
    message_id = str(uuid4())
    effective_thread_id = result.get("thread_id") or effective_thread_id
    # Track this turn's uploads in the session so later turns can still use them.
    append_session_files(effective_thread_id, normalized_file_ids)
    persisted_to_opensearch = False
    if use_persistent_memory and effective_memory_id:
        try:
            update_memory(
                effective_memory_id,
                user_query=user_input,
                message_id=message_id,
                answer=answer,
                elements=[],
            )
            persisted_to_opensearch = True
        except Exception as exc:
            logger.warning("Failed to persist agent chat turn for %s: %s", effective_memory_id, exc)
            if memory_warning is None:
                memory_warning = f"persistent_memory_update_failed: {exc}"

    # Always record the turn in session-local memory unless it was already
    # durably persisted to OpenSearch, so follow-up turns keep their context
    # within the process even when persistent memory is off (or unavailable).
    if not persisted_to_opensearch:
        append_session_turn(effective_thread_id, user_input, answer)

    response: Dict[str, Any] = {
        "answer": answer,
        "message_id": message_id,
        "memory_id": effective_memory_id,
        "thread_id": result.get("thread_id") or effective_thread_id,
        "file_paths": normalized_file_paths,
        "file_ids": normalized_file_ids,
        "skill_roots": normalized_skill_roots,
        "available_skills": result.get("available_skills") or [],
        "enabled_search_methods": normalized_enabled_search_methods,
        "use_persistent_memory": use_persistent_memory,
        "route_trace": result.get("route_trace") or {},
        "opengeodata_results": _extract_opengeodata_results(result),
        "agent_result": _json_safe(result),
    }
    if memory_warning:
        response["warning"] = memory_warning
    return response


def _trace_event_cap() -> int:
    """How many raw events one turn may record, from ``AGENT_TRACE_MAX_EVENTS``.

    A separate limit from the store's byte cap and earlier in the pipeline: this one stops an
    unbounded list growing in memory during a runaway turn, which is a problem well before the
    bytes would be.
    """
    raw = str(os.getenv("AGENT_TRACE_MAX_EVENTS") or "").strip()
    try:
        return max(1, int(raw)) if raw else 4000
    except ValueError:
        return 4000


def stream_agent_chat_events(
    *,
    user_input: str,
    thread_id: Optional[str] = None,
    memory_id: Optional[str] = None,
    conversation_name: Optional[str] = None,
    recent_k: Optional[int] = None,
    tool_strategy: str = "granular",
    include_mcp_tools: bool = False,
    mcp_modules: Optional[List[str]] = None,
    enabled_search_methods: Optional[Sequence[Any]] = None,
    use_persistent_memory: bool = True,
    smart_tool_routing: bool = True,
    forced_intent: Optional[str] = None,
    file_paths: Optional[Sequence[Any]] = None,
    file_ids: Optional[Sequence[Any]] = None,
    skill_roots: Optional[Sequence[Any]] = None,
    verbose: bool = False,
    agent_dev: Optional[bool] = None,
    code_exec: Optional[bool] = None,
    code_peer: Optional[str] = None,
    code_peer_model: Optional[str] = None,
    unified_peer: Optional[bool] = None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> Generator[Dict[str, Any], None, None]:
    effective_thread_id = thread_id or memory_id
    effective_memory_id = memory_id or thread_id
    memory_doc: Optional[Mapping[str, Any]] = None
    memory_warning: Optional[str] = None
    normalized_file_paths = _normalize_file_paths(file_paths)
    normalized_file_ids = _normalize_file_ids(file_ids)
    normalized_enabled_search_methods = _normalize_enabled_search_methods(enabled_search_methods)
    normalized_skill_roots = _normalize_skill_roots(skill_roots)

    yield {
        "event": "status",
        "data": {
            "stage": "agent_chat_started",
            "memory_id": effective_memory_id,
            "thread_id": effective_thread_id,
            "file_paths": normalized_file_paths,
            "file_ids": normalized_file_ids,
            "skill_roots": normalized_skill_roots,
            "enabled_search_methods": normalized_enabled_search_methods,
            "use_persistent_memory": use_persistent_memory,
        },
    }

    if use_persistent_memory:
        try:
            if effective_memory_id:
                memory_doc = get_or_create_memory(effective_memory_id)
            else:
                effective_memory_id = create_memory(conversation_name or "agent-chat")
                memory_doc = get_or_create_memory(effective_memory_id)
            yield {
                "event": "memory_loaded",
                "data": {
                    "memory_id": effective_memory_id,
                    "recent_k": recent_k,
                    "history_length": len((memory_doc or {}).get("chat_history") or []),
                },
            }
        except Exception as exc:
            logger.warning("Persistent agent chat memory unavailable: %s", exc)
            memory_warning = f"persistent_memory_unavailable: {exc}"
            effective_memory_id = memory_id
            # Fall back to session-local memory so the turn is not memoryless.
            memory_doc = build_session_memory_doc(effective_thread_id)
            yield {
                "event": "warning",
                "data": {
                    "stage": "memory_load",
                    "message": memory_warning,
                },
            }
    else:
        effective_memory_id = None
        # Persistent memory is off, but conversation context must still be
        # preserved locally within the session (keyed by thread_id).
        memory_doc = build_session_memory_doc(effective_thread_id)
        yield {
            "event": "status",
            "data": {
                "stage": "persistent_memory_disabled",
                "history_length": len((memory_doc or {}).get("chat_history") or []),
            },
        }

    # Carry files attached on earlier turns of this session (see run_agent_chat).
    effective_file_ids = list(dict.fromkeys([*get_session_files(effective_thread_id), *normalized_file_ids]))

    chat_history = _build_chat_history(memory_doc, recent_k=recent_k)
    effective_input = _augment_user_input_with_files(user_input, normalized_file_paths)
    effective_input = _augment_user_input_with_file_ids(effective_input, effective_file_ids)
    completed_response: Optional[Dict[str, Any]] = None
    # The durable record of this turn. Collected unfiltered — `agent_dev` decides what the
    # VIEWER sees, and a record that only holds what someone happened to switch on is not a
    # record. Bounded here as well as at the store, because an unbounded list on a runaway turn
    # is a memory problem long before it is a storage one.
    trace_events: List[Dict[str, Any]] = []
    trace_cap = _trace_event_cap()

    def _record(event: Dict[str, Any]) -> None:
        if len(trace_events) < trace_cap:
            trace_events.append(event)

    for event in stream_agent_query_events(
        effective_input,
        llm=_llm_for_request(llm_provider, llm_model, reasoning_effort),
        chat_history=chat_history,
        verbose=verbose,
        return_intermediate_steps=True,
        tool_strategy=tool_strategy,
        include_mcp_tools=include_mcp_tools,
        mcp_modules=mcp_modules,
        enabled_search_methods=normalized_enabled_search_methods,
        smart_tool_routing=smart_tool_routing,
        forced_intent=forced_intent,
        thread_id=effective_thread_id,
        skill_roots=normalized_skill_roots,
        agent_dev=agent_dev,
        code_exec=code_exec,
        code_peer=code_peer,
        code_peer_model=code_peer_model,
        unified_peer=unified_peer,
        input_file_ids=effective_file_ids,
        trace_recorder=_record,
    ):
        if event.get("event") == "completed" and isinstance(event.get("data"), Mapping):
            completed_response = dict(event["data"])
        yield event

    answer = sanitize_answer_links(_extract_agent_answer(completed_response or {}))
    message_id = str(uuid4())
    effective_thread_id = (completed_response or {}).get("thread_id") or effective_thread_id
    # Track this turn's uploads in the session so later turns can still use them.
    append_session_files(effective_thread_id, normalized_file_ids)

    # Beside the conversation, not inside it: the snapshot has a 5 MB cap and is fetched to
    # render a sidebar. Deliberately unable to fail the turn — save_turn_trace swallows its own
    # errors, because diagnostics are the least important thing happening here.
    if use_persistent_memory and effective_memory_id and trace_events:
        stored = save_turn_trace(
            effective_memory_id, thread_id=effective_thread_id, query=user_input,
            events=trace_events, answer=answer, model=llm_model, provider=llm_provider)
        if stored.get("stored"):
            yield {"event": "trace_saved", "data": stored}

    persisted_to_opensearch = False
    if use_persistent_memory and effective_memory_id:
        try:
            update_memory(
                effective_memory_id,
                user_query=user_input,
                message_id=message_id,
                answer=answer,
                elements=[],
            )
            persisted_to_opensearch = True
            yield {
                "event": "memory_saved",
                "data": {
                    "memory_id": effective_memory_id,
                    "message_id": message_id,
                },
            }
        except Exception as exc:
            logger.warning("Failed to persist agent chat turn for %s: %s", effective_memory_id, exc)
            if memory_warning is None:
                memory_warning = f"persistent_memory_update_failed: {exc}"
            yield {
                "event": "warning",
                "data": {
                    "stage": "memory_save",
                    "message": memory_warning,
                },
            }

    # Record the turn in session-local memory unless it was already durably
    # persisted to OpenSearch, so follow-up turns keep their context within the
    # process even when persistent memory is off (or unavailable).
    if not persisted_to_opensearch:
        append_session_turn(effective_thread_id, user_input, answer)
        yield {
            "event": "memory_saved",
            "data": {
                "scope": "session",
                "thread_id": effective_thread_id,
                "message_id": message_id,
            },
        }

    final_response: Dict[str, Any] = {
        "answer": answer,
        "message_id": message_id,
        "memory_id": effective_memory_id,
        "thread_id": (completed_response or {}).get("thread_id") or effective_thread_id,
        "file_paths": normalized_file_paths,
        "file_ids": normalized_file_ids,
        "skill_roots": normalized_skill_roots,
        "available_skills": (completed_response or {}).get("available_skills") or [],
        "enabled_search_methods": normalized_enabled_search_methods,
        "use_persistent_memory": use_persistent_memory,
        "route_trace": (completed_response or {}).get("route_trace") or {},
        "opengeodata_results": _extract_opengeodata_results(completed_response or {}),
        "agent_result": _json_safe(completed_response or {}),
    }
    if memory_warning:
        final_response["warning"] = memory_warning
    yield {"event": "response", "data": final_response}


__all__ = ["run_agent_chat", "stream_agent_chat_events"]
