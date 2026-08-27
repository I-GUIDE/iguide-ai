"""Intent classification and route selection for the agent runtime.

Determines *what kind* of task a query represents (discovery, analysis,
code, hybrid) and *which route* (search, analysis, direct_answer) the
orchestrator should take.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Keyword hint sets used by _classify_intent
# ---------------------------------------------------------------------------
ANALYSIS_HINTS: set[str] = {
    "analyze",
    "analysis",
    "count",
    "join",
    "spatial join",
    "buffer",
    "intersect",
    "within",
    "map",
    "plot",
    "statistics",
    "statistical",
    "hotspot",
    # Remote-sensing embeddings are analysis, not retrieval: naming a foundation
    # model used to send the turn to search, which then hunted the KB for a
    # dataset called "gse model embedding" and blew the context window.
    "embed",
    "embedding",
    "embeddings",
    "satellite",
    "remote sensing",
    "land cover",
    "segment",
    "segmentation",
}

CODE_HINTS: set[str] = {
    "code",
    "python",
    "script",
    "function",
    "class",
    "implement",
    "implementation",
    "api",
    "endpoint",
    "refactor",
    "debug",
    "fix",
    "unit test",
    "sql",
}

DISCOVERY_HINTS: set[str] = {
    "what is",
    "overview",
    "background",
    "resources",
    "dataset",
    "datasets",
    "publication",
    "publications",
    "notebook",
    "notebooks",
    "find",
    "discover",
    "file",
    "csv",
    "json",
    "column",
    "columns",
    "attached",
    "attachment",
}

FILE_ANALYSIS_HINTS: set[str] = {
    "inspect",
    "attached file",
    "attached files",
    "attachment",
    "attachments",
    "main columns",
    "column names",
    "save a short summary",
    "uploaded file",
    "file id",
    "downloadable",
}


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

def classify_intent(query: str) -> Dict[str, Any]:
    """Classify *query* into an intent using keyword hint matching."""
    text = (query or "").strip().lower()
    analysis_hits = sorted([kw for kw in ANALYSIS_HINTS if kw in text])
    code_hits = sorted([kw for kw in CODE_HINTS if kw in text])
    discovery_hits = sorted([kw for kw in DISCOVERY_HINTS if kw in text])
    file_analysis_hits = sorted([kw for kw in FILE_ANALYSIS_HINTS if kw in text])
    has_attached_files = "attached files are available to the agent via local file tools" in text

    if has_attached_files and file_analysis_hits:
        intent = "hybrid"
        reason = "attached_file_analysis_request"
    elif code_hits:
        intent = "code_task"
        reason = "matched_code_hints"
    elif analysis_hits and discovery_hits:
        intent = "hybrid"
        reason = "matched_analysis_and_discovery_hints"
    elif analysis_hits:
        intent = "analysis_task"
        reason = "matched_analysis_hints"
    else:
        intent = "general_discovery"
        reason = "default_to_discovery"
    return {
        "intent": intent,
        "reason": reason,
        "analysis_hits": analysis_hits,
        "code_hits": code_hits,
        "discovery_hits": discovery_hits,
        "file_analysis_hits": file_analysis_hits,
        "has_attached_files": has_attached_files,
    }


# ---------------------------------------------------------------------------
# Route helpers
# ---------------------------------------------------------------------------

def build_available_routes(
    *,
    available_tool_names: Sequence[str],
    chat_history: Optional[List[Any]],
) -> List[Dict[str, Any]]:
    """Return the set of routes the orchestrator may choose from."""
    routes: List[Dict[str, Any]] = []
    history_available = bool(chat_history)
    if history_available:
        routes.append(
            {
                "route": "direct_answer",
                "agent": "direct_answer_agent",
                "description": "Answer only from the current conversation history already loaded for this turn.",
                "requirements": ["chat_history"],
            }
        )
    if available_tool_names:
        routes.append(
            {
                "route": "search",
                "agent": "search_agent",
                "description": "Use the available tools to retrieve evidence and answer factual or discovery questions.",
                "tool_names": list(available_tool_names),
            }
        )
        routes.append(
            {
                "route": "analysis",
                "agent": "analysis_agent",
                "description": "Perform synthesis or analysis, calling SearchAgent for evidence when needed.",
                "tool_names": list(available_tool_names),
            }
        )
    return routes


def query_has_file_context(query: str) -> bool:
    """Return True when the query text indicates uploaded/attached files."""
    text = (query or "").lower()
    return (
        "attached files are available to the agent via local file tools" in text
        or "uploaded files are available to the agent via local file tools" in text
    )


def chat_history_preview(
    chat_history: Optional[List[Any]], max_items: int = 6
) -> List[Dict[str, str]]:
    """Return a normalised, truncated preview of *chat_history*."""
    preview: List[Dict[str, str]] = []
    for item in (chat_history or [])[-max_items:]:
        role = "user"
        content = ""
        if isinstance(item, dict):
            role = str(item.get("role") or "user")
            content = str(item.get("content") or "")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            role = str(item[0])
            content = str(item[1])
        else:
            content = str(item)
        content = " ".join(content.split())
        if content:
            preview.append({"role": role, "content": content[:400]})
    return preview


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of the first JSON object from *text*."""
    if not text:
        return None
    candidates = [text.strip()]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


# ---------------------------------------------------------------------------
# Route decision strategies
# ---------------------------------------------------------------------------

def _forced_route_from_value(
    value: Optional[str], available_routes: Sequence[Dict[str, Any]]
) -> Optional[str]:
    if not value:
        return None
    normalized = str(value).strip().lower()
    available = {str(item.get("route") or "").strip().lower() for item in available_routes}
    mapping = {
        "analysis": "analysis",
        "analysis_task": "analysis",
        "hybrid": "analysis",
        "search": "search",
        "general_discovery": "search",
        "code_task": "search",
        "direct_answer": "direct_answer",
        "memory_direct": "direct_answer",
    }
    forced = mapping.get(normalized)
    if forced in available:
        return forced
    return None


def heuristic_route_decision(
    query: str,
    available_routes: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Pick a route using keyword heuristics (no LLM call)."""
    route_names = {str(item.get("route") or "") for item in available_routes}
    lowered = (query or "").strip().lower()
    classification = classify_intent(query)
    if "direct_answer" in route_names:
        memory_phrases = (
            "what is my ",
            "what's my ",
            "what was my ",
            "what did i say ",
            "did i mention ",
            "what did we discuss ",
        )
        if lowered.startswith(memory_phrases):
            return {
                "route": "direct_answer",
                "reason": "heuristic_memory_reference",
                "intent": "memory_lookup",
                "router_type": "heuristic",
            }
    if classification["intent"] in {"analysis_task", "hybrid"} and "analysis" in route_names:
        return {
            "route": "analysis",
            "reason": classification["reason"],
            "intent": classification["intent"],
            "router_type": "heuristic",
        }
    if "search" in route_names:
        return {
            "route": "search",
            "reason": classification["reason"],
            "intent": classification["intent"],
            "router_type": "heuristic",
        }
    fallback = next(iter(route_names), "search")
    return {
        "route": fallback,
        "reason": "fallback_first_available_route",
        "intent": classification["intent"],
        "router_type": "heuristic",
    }


def llm_route_decision(
    *,
    query: str,
    chat_history: Optional[List[Any]],
    available_routes: Sequence[Dict[str, Any]],
    llm: Optional[Any],
    build_default_llm: Any,
) -> Optional[Dict[str, Any]]:
    """Ask the LLM to pick a route. Returns *None* on failure."""
    if not available_routes:
        return None
    active_llm = llm or build_default_llm()
    route_names = [str(item.get("route") or "") for item in available_routes if item.get("route")]
    prompt = (
        "You are a routing model for a multi-agent assistant.\n"
        "Choose exactly one route from the available routes.\n"
        "Return JSON only with keys: route, reason, confidence.\n"
        "Use direct_answer only when the question can be answered from chat history alone.\n"
        "Use analysis for synthesis, comparison, transformation, coding, statistics, or multi-step reasoning.\n"
        "Use search for retrieval or factual lookup that needs tools."
    )
    router_input = {
        "query": query,
        "available_routes": list(available_routes),
        "chat_history_preview": chat_history_preview(chat_history),
    }
    response = active_llm.invoke(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(router_input, ensure_ascii=True)},
        ]
    )
    content = getattr(response, "content", "")
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else getattr(part, "text", str(part))
            for part in content
        )
    parsed = extract_json_object(str(content))
    if not parsed:
        return None
    route = str(parsed.get("route") or "").strip()
    if route not in route_names:
        return None
    return {
        "route": route,
        "reason": str(parsed.get("reason") or "llm_router"),
        "confidence": parsed.get("confidence"),
        "router_type": "llm",
    }


def build_route_trace(
    query: str,
    available_tool_names: Sequence[str],
    available_routes: Sequence[Dict[str, Any]],
    chat_history: Optional[List[Any]] = None,
    llm: Optional[Any] = None,
    forced_intent: Optional[str] = None,
    *,
    build_default_llm: Any = None,
    select_allowed_tools: Any = None,
) -> Dict[str, Any]:
    """Build a complete routing trace for *query*.

    ``build_default_llm`` and ``select_allowed_tools`` are injected by
    the caller to avoid circular imports.
    """
    classification = classify_intent(query)
    forced_route = _forced_route_from_value(forced_intent, available_routes)
    allowed = select_allowed_tools(classification["intent"], available_tool_names) if select_allowed_tools else list(available_tool_names)
    if forced_route:
        return {
            "query": query,
            "route": forced_route,
            "intent": classification["intent"],
            "forced_intent": forced_intent,
            "reason": "forced_route",
            "analysis_hits": classification["analysis_hits"],
            "code_hits": classification["code_hits"],
            "discovery_hits": classification["discovery_hits"],
            "available_tools": list(available_tool_names),
            "available_routes": list(available_routes),
            "allowed_tools": allowed,
            "router_type": "forced",
        }
    llm_choice = llm_route_decision(
        query=query,
        chat_history=chat_history,
        available_routes=available_routes,
        llm=llm,
        build_default_llm=build_default_llm,
    )
    route = str((llm_choice or {}).get("route") or "")
    if not route:
        llm_choice = heuristic_route_decision(query, available_routes)
        route = str(llm_choice.get("route") or "search")
    return {
        "query": query,
        "route": route,
        "intent": classification["intent"],
        "forced_intent": forced_intent,
        "reason": llm_choice.get("reason") or classification["reason"],
        "confidence": llm_choice.get("confidence"),
        "analysis_hits": classification["analysis_hits"],
        "code_hits": classification["code_hits"],
        "discovery_hits": classification["discovery_hits"],
        "available_tools": list(available_tool_names),
        "available_routes": list(available_routes),
        "allowed_tools": allowed,
        "router_type": llm_choice.get("router_type"),
    }
