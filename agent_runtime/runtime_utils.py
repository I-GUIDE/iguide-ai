"""Response parsing, artifact extraction, and trace building utilities.

Pure functions that inspect agent results and build structured traces
for observability.  No side effects, no LLM calls, no external I/O.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Answer link sanitation
# ---------------------------------------------------------------------------
# LLMs sometimes prefix URLs with the "sandbox:" pseudo-scheme they saw in training
# (e.g. [map](sandbox:/agent/files/<id>/download)) and sometimes cite a tool's INTERNAL
# filesystem path (e.g. /app/agent_chat_files/qgis_jobs/.../rendered_map.png) instead of the
# registered download_url. Both render as broken links in any client. The correct link is
# already appended deterministically (_append_image_embeds), so here we strip the pseudo-scheme
# and defuse links that point at unreachable local paths.
_SANDBOX_URI_RE = re.compile(r"\bsandbox:(?=(?:https?:)?/)", re.IGNORECASE)
_MD_LINK_RE = re.compile(r"(!?)\[([^\]]*)\]\(\s*([^)\s]+)\s*\)")
_SERVABLE_PREFIXES = ("http://", "https://", "data:", "mailto:", "#", "/agent/files/")
# A URL that claims to be an agent file/artifact. Any such target must match a file the run
# actually produced — models otherwise invent plausible hosts/paths for a real internal path
# (observed live: https://agent-chat-files.s3.amazonaws.com/qgis_jobs/.../rendered_map.png).
_AGENT_FILE_HINT_RE = re.compile(r"/agent/files/|agent[_-]chat[_-]files|qgis_jobs/|/outputs?/", re.I)
_FILE_ID_RE = re.compile(r"(file_[0-9a-f]{6,})", re.I)
# A link that OFFERS A FILE: an artifact-ish extension or a download-y label. These must resolve
# to something real (a produced artifact or an evidence URL) — otherwise the model can invent a
# plausible placeholder host that dodges the agent-file heuristic above (observed live:
# https://example.com/path-to-buffer.geojson).
_FILE_EXT_RE = re.compile(
    r"\.(?:png|jpe?g|gif|webp|svg|tif{1,2}|geojson|json|csv|tsv|zip|gpkg|shp|pdf|txt|py|ipynb)"
    r"(?:[?#].*)?$", re.I,
)
_DOWNLOAD_LABEL_RE = re.compile(r"\bdownload\b", re.I)


def strip_sandbox_uris(text: Optional[str]) -> str:
    """Remove the LLM 'sandbox:' pseudo-scheme from URLs in *text*."""
    if not text:
        return text or ""
    return _SANDBOX_URI_RE.sub("", text)


def sanitize_answer_links(
    text: Optional[str],
    *,
    allowed_file_ids: Optional[Sequence[str]] = None,
    allowed_urls: Optional[Sequence[str]] = None,
) -> str:
    """Make every markdown link/image in an answer client-renderable and REAL.

    * strips the ``sandbox:`` pseudo-scheme (``sandbox:/agent/files/x`` -> ``/agent/files/x``);
    * defuses targets the client cannot fetch — an internal filesystem path, or a URL that claims
      to be an agent file but does not match an artifact this run produced (models invent hosts,
      e.g. an ``…s3.amazonaws.com/qgis_jobs/…`` URL built from a real internal path);
    * IMAGES must resolve to a produced artifact (or be a ``data:`` URI) — an unverifiable image
      is dropped rather than rendered as a broken thumbnail; a text link degrades to its label.

    ``allowed_file_ids`` / ``allowed_urls`` come from the artifacts the run actually registered.
    When neither is supplied, only the path/scheme checks apply (no artifact verification).
    Prose mentioning "sandbox" and ordinary citation links (http(s), mailto:, #anchor) are kept.
    """
    if not text:
        return text or ""
    out = strip_sandbox_uris(text)
    ids = {str(i).lower() for i in (allowed_file_ids or []) if i}
    urls = {str(u) for u in (allowed_urls or []) if u}
    verifying = bool(ids or urls)

    def _is_known_artifact(url: str) -> bool:
        if url in urls:
            return True
        match = _FILE_ID_RE.search(url)
        return bool(match and match.group(1).lower() in ids)

    def _fix(match: "re.Match[str]") -> str:
        bang, label, url = match.group(1), match.group(2), match.group(3)
        low = url.lower()
        drop = "" if bang else label

        if low.startswith(("data:", "mailto:", "#")):
            return match.group(0)
        # A file OFFER (artifact-ish extension or a "download" label) must resolve to something
        # real, whatever host it claims.
        if verifying and not _is_known_artifact(url) and (
            _FILE_EXT_RE.search(url) or _DOWNLOAD_LABEL_RE.search(label)
        ):
            return drop
        # Anything presenting itself as an agent file must be a real produced artifact.
        if _AGENT_FILE_HINT_RE.search(url):
            if verifying and not _is_known_artifact(url):
                return drop
            if low.startswith(("http://", "https://", "/agent/files/")):
                return match.group(0)
            return drop                              # internal path form
        if low.startswith("file:") or url.startswith("/"):
            return drop                              # unreachable local path
        if bang and low.startswith(("http://", "https://")):
            # An image we cannot tie to a produced artifact would render as a broken thumbnail.
            return match.group(0) if (not verifying or _is_known_artifact(url)) else ""
        if low.startswith(("http://", "https://")):
            return match.group(0)                    # ordinary citation link
        return match.group(0)

    return _MD_LINK_RE.sub(_fix, out).strip()


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_final_answer(result: Any) -> Optional[str]:
    """Extract the final answer string from an agent result dict."""
    if not isinstance(result, dict):
        return None
    answer = result.get("final_answer")
    if isinstance(answer, str) and answer.strip():
        return answer.strip()

    # Direct fallback path
    if result.get("fallback") == "direct_rag_tool":
        direct = result.get("result")
        if isinstance(direct, dict):
            answer = direct.get("answer")
            if isinstance(answer, str) and answer.strip():
                return answer.strip()

    messages = result.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            content = getattr(msg, "content", None)
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(msg, dict):
                c = msg.get("content")
                if isinstance(c, str) and c.strip():
                    return c.strip()
    return None


# ---------------------------------------------------------------------------
# Search artifact extraction
# ---------------------------------------------------------------------------

def extract_search_artifacts(result: Any) -> Dict[str, Any]:
    """Parse agent messages into tool_calls, tool_results, and raw_messages."""
    artifacts: Dict[str, Any] = {"tool_calls": [], "tool_results": [], "raw_messages": []}
    if not isinstance(result, dict):
        return artifacts
    messages = result.get("messages")
    if not isinstance(messages, list):
        return artifacts

    for msg in messages:
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content.strip():
            artifacts["raw_messages"].append(content.strip())
        elif isinstance(msg, dict):
            text = msg.get("content")
            if isinstance(text, str) and text.strip():
                artifacts["raw_messages"].append(text.strip())

        tool_calls = getattr(msg, "tool_calls", None)
        if isinstance(tool_calls, list):
            for call in tool_calls:
                artifacts["tool_calls"].append(
                    {
                        "name": call.get("name", "unknown_tool"),
                        "args": call.get("args", {}),
                        # The id is the ONLY thing saying which result belongs to which call.
                        # Without it the ledger paired by position within a tool name, so a
                        # fail-then-succeed pair put the failed call's arguments on the
                        # successful call's result — and since a failed result carried no
                        # curated facts and was dropped entirely, the good run's file_id and
                        # map_layer landed on the failed row and read as a delivered layer.
                        "id": call.get("id"),
                    }
                )

        name = getattr(msg, "name", None)
        tool_call_id = getattr(msg, "tool_call_id", None)
        if name and tool_call_id:
            text = content if isinstance(content, str) else str(content)
            artifacts["tool_results"].append(
                {
                    "name": name,
                    "tool_call_id": tool_call_id,
                    "content": text,
                }
            )
    return artifacts


# ---------------------------------------------------------------------------
# History repair (tool_call / tool-message pairing)
# ---------------------------------------------------------------------------

def _message_tool_call_ids(msg: Any) -> List[str]:
    """Return the tool_call ids requested by an assistant message (or [])."""
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls is None and isinstance(msg, dict):
        tool_calls = msg.get("tool_calls")
    ids: List[str] = []
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if isinstance(call, dict):
                cid = call.get("id")
            else:
                cid = getattr(call, "id", None)
            if cid:
                ids.append(cid)
    return ids


def _message_tool_response_id(msg: Any) -> Optional[str]:
    """Return the tool_call_id a tool message responds to (or None)."""
    tcid = getattr(msg, "tool_call_id", None)
    if tcid is None and isinstance(msg, dict):
        tcid = msg.get("tool_call_id")
    return tcid


def repair_tool_call_sequence(messages: Any) -> tuple[Any, bool]:
    """Drop messages that make a tool_call/tool-message sequence invalid.

    Providers like OpenAI reject a request where an assistant message with
    ``tool_calls`` is not followed by a tool message for every ``tool_call_id``
    (a "dangling" tool call left in checkpointer state poisons every later turn).

    This returns ``(repaired_messages, changed)``:
      * an assistant message is dropped if *any* of its tool_calls has no
        matching tool response anywhere in the list;
      * a tool message is dropped if its ``tool_call_id`` belongs to a dropped
        (or absent) assistant message (orphan response);
      * all other messages are preserved in order.

    When nothing needs repair, the original list object is returned unchanged.
    """
    if not isinstance(messages, list) or not messages:
        return messages, False

    responded = {
        rid
        for rid in (_message_tool_response_id(msg) for msg in messages)
        if rid
    }

    kept: List[Any] = []
    kept_call_ids: set = set()
    changed = False
    for msg in messages:
        call_ids = _message_tool_call_ids(msg)
        if call_ids:
            if all(cid in responded for cid in call_ids):
                kept.append(msg)
                kept_call_ids.update(call_ids)
            else:
                changed = True  # assistant tool_call(s) with no response -> drop
            continue
        response_id = _message_tool_response_id(msg)
        if response_id is not None:
            if response_id in kept_call_ids:
                kept.append(msg)
            else:
                changed = True  # orphan tool response -> drop
            continue
        kept.append(msg)

    return (kept, True) if changed else (messages, False)


def extract_tool_result_json(artifacts: Dict[str, Any], tool_name: str) -> Optional[Dict[str, Any]]:
    """Find and parse the last JSON tool result for *tool_name*."""
    for item in reversed(artifacts.get("tool_results") or []):
        if str(item.get("name") or "") != tool_name:
            continue
        content = item.get("content")
        if not isinstance(content, str):
            continue
        try:
            parsed = json.loads(content)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


# ---------------------------------------------------------------------------
# Orchestration trace
# ---------------------------------------------------------------------------

def _nested_called_tools_from_analysis_payload(artifacts: Dict[str, Any]) -> List[str]:
    payload = extract_tool_result_json(artifacts, "analysis_agent_answer") or {}
    analysis_result = payload.get("analysis_result")
    nested_artifacts = extract_search_artifacts(analysis_result if isinstance(analysis_result, dict) else {})
    return [str(item.get("name") or "") for item in nested_artifacts.get("tool_calls") or []]


def _selected_skill_names(artifacts: Dict[str, Any]) -> List[str]:
    selected: List[str] = []

    def _collect_from_tool_results(tool_results: Sequence[Dict[str, Any]]) -> None:
        for item in tool_results:
            if str(item.get("name") or "") != "load_skill":
                continue
            payload = parse_tool_result_payload(item)
            if not isinstance(payload, dict) or payload.get("status") != "ok":
                continue
            skill = payload.get("skill")
            if isinstance(skill, dict):
                name = str(skill.get("name") or "").strip()
                if name and name not in selected:
                    selected.append(name)

    _collect_from_tool_results(artifacts.get("tool_results") or [])
    analysis_payload = extract_tool_result_json(artifacts, "analysis_agent_answer") or {}
    analysis_result = analysis_payload.get("analysis_result")
    nested_artifacts = extract_search_artifacts(analysis_result if isinstance(analysis_result, dict) else {})
    _collect_from_tool_results(nested_artifacts.get("tool_results") or [])
    return selected


# Maps supervisor peer names to the legacy tool names the trace/SSE consumers expect
# (e.g. the streaming layer keys ``search_complete`` off ``search_agent_evidence``).
_PEER_TO_TOOL = {
    "search": "search_agent_evidence",
    "analyze": "analysis_agent_answer",
    "code": "code_agent_answer",
}


def _is_supervisor_state(result: Any) -> bool:
    """Whether *result* is a supervisor ``sup_state`` (vs an agents-as-tools result).

    The supervisor returns a dict carrying ``actions``/``next_action`` and no
    top-level ``messages``; the legacy path returns ``{"messages": [...]}``.
    """
    return isinstance(result, dict) and "messages" not in result and (
        "actions" in result or "next_action" in result
    )


def build_supervisor_trace(
    *,
    query: str,
    chat_history: Optional[List[Any]],
    available_agent_names: Sequence[str],
    sup_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Derive a route trace from the supervisor's ``actions`` (not message shape).

    The supervisor records the peers it dispatched in ``sup_state['actions']``;
    this maps them to the legacy tool vocabulary so existing trace/SSE consumers
    keep working on the default supervisor path.
    """
    actions = [a for a in (sup_state.get("actions") or []) if a in _PEER_TO_TOOL]
    distinct: List[str] = []
    for a in actions:
        if a not in distinct:
            distinct.append(a)
    called_tools = [_PEER_TO_TOOL[a] for a in actions]
    route = ("supervisor:" + "→".join(distinct)) if distinct else "orchestrator_only"
    audit = sup_state.get("audit") or {}
    return {
        "query": query,
        "route": route,
        "available_agents": list(available_agent_names),
        "called_tools": called_tools,
        "analysis_called_tools": [],
        "selected_skills": [],
        "chat_history_available": bool(chat_history),
        # Supervisor-specific signal (the richer view that used to be discarded):
        "supervisor_actions": list(sup_state.get("actions") or []),
        "document_count": len(sup_state.get("evidence") or []),
        "has_analysis": sup_state.get("analysis_results") is not None,
        "has_code": sup_state.get("code_result") is not None,
        "audit_severity": audit.get("severity"),
    }


def build_orchestration_trace(
    *,
    query: str,
    chat_history: Optional[List[Any]],
    available_agent_names: Sequence[str],
    orchestration_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Analyse *orchestration_result* to determine the actual route taken."""
    # Default supervisor path: derive the route from sup_state['actions'] rather
    # than the agents-as-tools message shape (which a sup_state does not have).
    if _is_supervisor_state(orchestration_result):
        return build_supervisor_trace(
            query=query,
            chat_history=chat_history,
            available_agent_names=available_agent_names,
            sup_state=orchestration_result,
        )
    artifacts = extract_search_artifacts(orchestration_result)
    tool_calls = artifacts.get("tool_calls") or []
    called_tools = [str(item.get("name") or "") for item in tool_calls]
    called_set = set(called_tools)
    nested_analysis_called_tools = _nested_called_tools_from_analysis_payload(artifacts)
    nested_analysis_called_set = set(nested_analysis_called_tools)
    selected_skills = _selected_skill_names(artifacts)
    if "answer_from_memory" in called_set:
        memory_payload = extract_tool_result_json(artifacts, "answer_from_memory") or {}
        if memory_payload.get("can_answer") and memory_payload.get("answer"):
            route = "direct_answer"
        elif "search_agent_evidence" in called_set and "analysis_agent_answer" in called_set and "code_agent_answer" in nested_analysis_called_set:
            route = "search_then_analysis_with_code"
        elif "analysis_agent_answer" in called_set and "code_agent_answer" in nested_analysis_called_set:
            route = "analysis_with_code"
        elif "search_agent_evidence" in called_set and "analysis_agent_answer" in called_set:
            route = "search_then_analysis"
        elif "search_agent_evidence" in called_set:
            route = "search"
        elif "analysis_agent_answer" in called_set:
            route = "analysis"
        else:
            route = "direct_answer_attempted"
    elif "search_agent_evidence" in called_set and "analysis_agent_answer" in called_set and "code_agent_answer" in nested_analysis_called_set:
        route = "search_then_analysis_with_code"
    elif "analysis_agent_answer" in called_set and "code_agent_answer" in nested_analysis_called_set:
        route = "analysis_with_code"
    elif "search_agent_evidence" in called_set and "analysis_agent_answer" in called_set:
        route = "search_then_analysis"
    elif "search_agent_evidence" in called_set:
        route = "search"
    elif "analysis_agent_answer" in called_set:
        route = "analysis"
    else:
        route = "orchestrator_only"
    return {
        "query": query,
        "route": route,
        "available_agents": list(available_agent_names),
        "called_tools": called_tools,
        "analysis_called_tools": nested_analysis_called_tools,
        "selected_skills": selected_skills,
        "chat_history_available": bool(chat_history),
    }


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------

def message_role(msg: Any) -> str:
    """Extract a normalised role string from a LangChain message."""
    if isinstance(msg, dict):
        role = msg.get("role") or msg.get("type")
        if isinstance(role, str) and role.strip():
            return role.strip()
    role = getattr(msg, "type", None)
    if isinstance(role, str) and role.strip():
        return role.strip()
    role = getattr(msg, "role", None)
    if isinstance(role, str) and role.strip():
        return role.strip()
    return msg.__class__.__name__.replace("Message", "").lower() or "message"


def message_text(msg: Any) -> str:
    """Extract text content from a LangChain message."""
    content = getattr(msg, "content", None)
    if isinstance(msg, dict) and content is None:
        content = msg.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    return ""


def parse_tool_result_payload(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Try to JSON-parse the content of a tool result."""
    content = item.get("content")
    if not isinstance(content, str):
        return None
    try:
        parsed = json.loads(content)
    except Exception:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


# ---------------------------------------------------------------------------
# LLM interaction trace building
# ---------------------------------------------------------------------------

def build_search_payload_interactions(
    payload: Dict[str, Any],
    *,
    agent_name: str,
    parent_tool_name: str,
    sequence_start: int,
) -> List[Dict[str, Any]]:
    """Build interaction entries from a search_agent_evidence payload."""
    interactions: List[Dict[str, Any]] = []
    sequence = sequence_start

    route_trace = payload.get("route_trace")
    if isinstance(route_trace, dict):
        interactions.append(
            {
                "sequence": sequence,
                "kind": "agent_route_decision",
                "agent": agent_name,
                "parent_tool": parent_tool_name,
                "route_trace": route_trace,
            }
        )
        sequence += 1

    for call in payload.get("search_agent_tool_calls") or []:
        interactions.append(
            {
                "sequence": sequence,
                "kind": "llm_tool_decision",
                "agent": agent_name,
                "parent_tool": parent_tool_name,
                "tool_name": call.get("name", "unknown_tool"),
                "tool_args": call.get("args", {}),
            }
        )
        sequence += 1

    for result in payload.get("search_agent_tool_results") or []:
        interactions.append(
            {
                "sequence": sequence,
                "kind": "tool_result",
                "agent": agent_name,
                "parent_tool": parent_tool_name,
                "tool_name": result.get("name", "unknown_tool"),
                "tool_call_id": result.get("tool_call_id"),
                "content": result.get("content", ""),
            }
        )
        sequence += 1

    summary = payload.get("search_agent_summary")
    if isinstance(summary, str) and summary.strip():
        interactions.append(
            {
                "sequence": sequence,
                "kind": "llm_message",
                "agent": agent_name,
                "parent_tool": parent_tool_name,
                "role": "assistant",
                "content": summary.strip(),
            }
        )

    return interactions


def build_llm_interaction_trace(
    result: Any,
    *,
    agent_name: str,
    sequence_start: int = 1,
) -> List[Dict[str, Any]]:
    """Recursively build an interaction trace from agent messages."""
    if not isinstance(result, dict):
        return []

    messages = result.get("messages")
    if not isinstance(messages, list):
        return []

    interactions: List[Dict[str, Any]] = []
    sequence = sequence_start

    for index, msg in enumerate(messages):
        role = message_role(msg)
        content = message_text(msg)
        tool_calls = getattr(msg, "tool_calls", None)
        if isinstance(msg, dict) and tool_calls is None:
            tool_calls = msg.get("tool_calls")

        if isinstance(tool_calls, list) and tool_calls:
            interactions.append(
                {
                    "sequence": sequence,
                    "kind": "llm_tool_decision",
                    "agent": agent_name,
                    "message_index": index,
                    "role": role,
                    "content": content,
                    "tool_calls": [
                        {
                            "name": call.get("name", "unknown_tool"),
                            "args": call.get("args", {}),
                        }
                        for call in tool_calls
                    ],
                }
            )
            sequence += 1
            continue

        name = getattr(msg, "name", None)
        tool_call_id = getattr(msg, "tool_call_id", None)
        if isinstance(msg, dict):
            if name is None:
                name = msg.get("name")
            if tool_call_id is None:
                tool_call_id = msg.get("tool_call_id")

        if name and tool_call_id:
            payload = parse_tool_result_payload(
                {
                    "name": str(name),
                    "tool_call_id": str(tool_call_id),
                    "content": content,
                }
            )
            tool_name = str(name)
            if isinstance(payload, dict):
                if tool_name == "search_agent_evidence":
                    nested = build_search_payload_interactions(
                        payload,
                        agent_name="search_agent",
                        parent_tool_name=tool_name,
                        sequence_start=sequence,
                    )
                    interactions.extend(nested)
                    if nested:
                        sequence = nested[-1]["sequence"] + 1
                elif tool_name == "analysis_agent_answer":
                    analysis_result = payload.get("analysis_result")
                    nested = build_llm_interaction_trace(
                        analysis_result,
                        agent_name="analysis_agent",
                        sequence_start=sequence,
                    )
                    interactions.extend(nested)
                    if nested:
                        sequence = nested[-1]["sequence"] + 1
                elif tool_name == "code_agent_answer":
                    code_result = payload.get("code_result")
                    nested = build_llm_interaction_trace(
                        code_result,
                        agent_name="code_agent",
                        sequence_start=sequence,
                    )
                    interactions.extend(nested)
                    if nested:
                        sequence = nested[-1]["sequence"] + 1
                elif tool_name == "answer_from_memory":
                    interactions.append(
                        {
                            "sequence": sequence,
                            "kind": "memory_decision",
                            "agent": agent_name,
                            "tool_name": tool_name,
                            "can_answer": payload.get("can_answer"),
                            "reason": payload.get("reason"),
                            "answer": payload.get("answer", ""),
                        }
                    )
                    sequence += 1

            interactions.append(
                {
                    "sequence": sequence,
                    "kind": "tool_result",
                    "agent": agent_name,
                    "message_index": index,
                    "tool_name": tool_name,
                    "tool_call_id": str(tool_call_id),
                    "content": content,
                }
            )
            sequence += 1
            continue

        if content:
            interactions.append(
                {
                    "sequence": sequence,
                    "kind": "llm_message",
                    "agent": agent_name,
                    "message_index": index,
                    "role": role,
                    "content": content,
                }
            )
            sequence += 1

    return interactions


# ---------------------------------------------------------------------------
# Search evidence payload builder
# ---------------------------------------------------------------------------

def build_search_evidence_payload(
    query: str,
    search_response: Any,
    route_trace: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Bundle search results into a payload for analysis/code agents."""
    search_summary = extract_final_answer(search_response) or ""
    search_artifacts = extract_search_artifacts(search_response)
    return {
        "user_query": query,
        "route_trace": route_trace,
        "search_agent_summary": search_summary,
        "search_agent_tool_calls": search_artifacts["tool_calls"],
        "search_agent_tool_results": search_artifacts["tool_results"],
    }
