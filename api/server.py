import os
import logging
import json
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, send_file, stream_with_context
from flask_cors import CORS
from flasgger import Swagger

from rag_pipeline.agent_file_store import (may_read as file_store_may_read,
                                           require_file_record, reset_session as reset_file_store_session,
                                           resolve_file_id, save_uploaded_file,
                                           set_session as set_file_store_session)
from rag_pipeline.agent_chat_service import run_agent_chat, stream_agent_chat_events
from agent_runtime import deployment_mode, identity, platform_endpoints
from rag_pipeline.memory_module import (MemoryAccessDenied, SnapshotTooLarge,
                                        assert_owner as assert_memory_owner,
                                        get_session_snapshot, get_turn_trace, list_memories,
                                        list_turn_traces, save_session_snapshot)
from rag_pipeline.pipeline import run_pipeline

app = Flask(__name__)

# Load environment variables
load_dotenv()

CORS(app)

swagger = Swagger(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# API key auth
# ---------------------------------------------------------------------------

def _coalesce(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _demo_mode() -> bool:
    """True when this deployment is in demo mode — see ``agent_runtime.deployment_mode``.

    Demo turns OFF the API-key check on every agent endpoint and tells the UI to hide its
    connection settings, so a page can be handed to an audience without also handing them a
    credential to paste. That is the whole point of it, and it is also exactly why it is not the
    default: with it on, anyone who finds the URL can run turns that spend this deployment's
    Earth Engine quota and LLM budget. Set it only on a deployment you are willing to have used.

    Kept as a named helper rather than inlined: it is read from a dozen places here, and the
    legacy ``DEMO_MODE=true`` env still selects it when ``AGENT_MODE`` is unset.
    """
    return deployment_mode.is_demo()


# Which model answers in demo mode. Configurable, but with a real default rather than falling
# through to OPENAI_CHAT_MODEL: a demo is handed to an audience, and "whatever this deployment
# happens to be set to" is not a demo decision.
DEMO_MODEL_DEFAULT = "gpt-5.6-luna"
DEMO_PROVIDER_DEFAULT = "openai"


def _demo_model() -> tuple:
    """(provider, model) for demo mode, from DEMO_MODEL / DEMO_MODEL_PROVIDER."""
    return (str(os.getenv("DEMO_MODEL_PROVIDER") or DEMO_PROVIDER_DEFAULT).strip(),
            str(os.getenv("DEMO_MODEL") or DEMO_MODEL_DEFAULT).strip())


def _client_may_choose_the_model() -> bool:
    """Whether a model named by the request is honoured, or dropped for the deployment's own.

    The only control that can change the model is the settings panel, and BOTH demo and token
    mode hide it — demo because there is nothing to configure, token because the credential is a
    cookie rather than something you paste. A value left in a returning visitor's localStorage
    then pins them to a model they can neither see nor change.

    That is not hypothetical. Token mode was switched on while browsers still held
    `gpt-oss:120b`; the deployment default moved to gpt-5.6-luna and every request from the map
    UI kept asking for the old one, with no picker to correct it. Demo mode already had this
    guard for exactly this reason; token mode hid the same panel without it.
    """
    return not (_demo_mode() or deployment_mode.is_token())


def _get_agent_chat_api_key() -> str:
    return str(os.getenv("AGENT_CHAT_API_KEY") or "").strip()


# At import, so it appears once in the container log rather than per request. An open
# deployment should never be a thing someone discovers from its behaviour, and the mode an
# operator THINKS is set is the one thing worth stating out loud on every boot.
logger.info("Agent deployment mode: %s (api key %s, platform tier %s)",
            deployment_mode.current_mode(),
            "configured" if _get_agent_chat_api_key() else "NOT configured",
            platform_endpoints.current_tier() or "unset")
if platform_endpoints.consistency_warning():
    logger.warning("%s", platform_endpoints.consistency_warning())
# Said at boot for the same reason as the tier/cookie mismatch above: an agent quietly talking
# to a cluster the rest of the tier has moved off is not a state anyone goes looking for.
if platform_endpoints.opensearch_drift_warning():
    logger.warning("%s", platform_endpoints.opensearch_drift_warning())
# Names only, never values — the same rule /agent/whoami follows for cookies.
if platform_endpoints.opensearch_credential_warning():
    logger.warning("%s", platform_endpoints.opensearch_credential_warning())
# Supported and deliberate, but not a thing to discover from surprising search results.
if platform_endpoints.search_tier_note():
    logger.info("%s", platform_endpoints.search_tier_note())
if deployment_mode.boot_warning():
    logger.warning("%s", deployment_mode.boot_warning())


def _extract_presented_api_key() -> str:
    header_key = str(request.headers.get("X-API-KEY") or "").strip()
    if header_key:
        return header_key
    auth = str(request.headers.get("Authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


def _require_agent_chat_api_key(user: Optional[Any] = None) -> None:
    # A verified user IS a credential, and the stronger one: the API key says only "someone who
    # has the key", the JWT says WHO. Requiring both would mean a signed-in visitor is refused
    # for lacking a key that token mode gives them no way to enter — the settings panel is
    # hidden precisely because there is nothing to paste. Observed live: sign in, then
    # "You are not signed in".
    #
    # The key remains the way a caller with no browser gets in (the eval harness, scripts), so
    # the two are ALTERNATIVES, never a pair.
    if user is not None:
        return
    # Checked BEFORE the key is read, so a deployment can keep AGENT_CHAT_API_KEY configured and
    # simply stop enforcing it for the duration of a demo — rather than having to unset the
    # secret and remember to put it back.
    if _demo_mode():
        return
    expected = _get_agent_chat_api_key()
    if not expected:
        return  # auth disabled when key is not configured
    presented = _extract_presented_api_key()
    if not presented or presented != expected:
        raise PermissionError("Forbidden: invalid API key.")


def _service_key_presented() -> bool:
    """True when this request carried the VALID service key (not merely some key)."""
    expected = _get_agent_chat_api_key()
    return bool(expected) and _extract_presented_api_key() == expected


def _token_strict() -> bool:
    """See ``identity.token_strict`` — the memory store needs the same answer."""
    return identity.token_strict()


def _extract_user_token() -> str:
    """The platform access token for this request.

    The cookie is the real path: it is httpOnly, scoped to `.i-guide.io`, and arrives here on
    its own because the UI is served from this same origin. A Bearer value is accepted too, but
    ONLY when it is structurally a JWT — `Authorization: Bearer` is also how a service caller
    presents the API key, and treating that key as a token would turn a valid service request
    into a confusing 403 about signatures.
    """
    cookie = str(request.cookies.get(identity.cookie_name()) or "").strip()
    if cookie:
        return cookie
    auth = str(request.headers.get("Authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        value = auth[7:].strip()
        if value.count(".") == 2:
            return value
    return ""


def _require_user():
    """Identify the caller. Returns a ``User``, or None when this deployment has no identity.

    Only token mode identifies anyone. In dev and demo the answer is None and the file store
    keeps scoping by conversation — which is the important part: "no identity" must resolve to
    SESSION scoping, never to one shared owner that every anonymous caller lands in and can
    read each other's files through.
    """
    if not deployment_mode.is_token():
        return None
    token = _extract_user_token()
    # A service caller — the eval harness, a script, anything without a browser — has no JWT and
    # presents the service key instead. `_require_agent_chat_api_key` already validated it; such
    # a request simply gets no user, and falls back to session scoping like dev mode.
    if not token and _service_key_presented():
        return None
    try:
        # `identify`, not `decode_token`: against production this deployment verifies by ASKING
        # the platform rather than by holding its signing secret. See identity.verify_mode.
        user = identity.identify(token)
        identity.authorize(user)
    except identity.IdentityError:
        # Non-strict relaxes OWNERSHIP of records written before ownership existed. It does not
        # relax "who are you", and it never did anyone a favour by trying: a signed-out visitor
        # whose identity error was swallowed here fell through to the API-key gate and was told
        # "Forbidden: invalid API key" — a message about a credential token mode gives them no
        # way to enter, for a problem that is actually "please sign in". Observed live.
        #
        # A caller holding the service key is the one exception, because it is a real credential
        # and belongs to something without a browser.
        if not _token_strict() and _service_key_presented():
            return None
        raise
    return user


def _identity_reason(exc: Exception) -> str:
    """One vocabulary for "why not", shared by the refusals and by /agent/whoami.

    /agent/whoami answers 200 by design — it exists to EXPLAIN a refusal, so it cannot make
    one — which means a client reading it has no status code to act on. Without this it would
    have to match on the prose in `reason`, and the whole point of labelling refusals was that
    nobody should ever do that. Derived from the same exception here so the two cannot drift:
    a 401 on /agent/chat and a `token_expired` from whoami are the same fact.
    """
    if isinstance(exc, identity.TokenExpired):
        return "token_expired"
    if isinstance(exc, identity.InsufficientRole):
        return "insufficient_role"
    if isinstance(exc, identity.TokenMissing):
        return "not_signed_in"
    if isinstance(exc, identity.IdentityNotConfigured):
        return "identity_not_configured"
    return "token_invalid"


def _identity_error_response(exc: Exception):
    """Map an identity failure to a status a client can ACT on without reading prose.

    401 and 403 mean different things here and the split is the whole contract: 401 says the
    token was fine and has simply aged out, so refresh once and retry; 403 says stop. Collapsing
    them leaves the UI unable to tell those apart, so it either never refreshes or refreshes
    forever against a token that will never validate. `reason` carries the same distinction in
    machine-readable form, because a client should never have to match on an error string.
    """
    if isinstance(exc, identity.TokenExpired):
        return jsonify({"error": "Your session has expired.",
                        "reason": "token_expired"}), 401
    if isinstance(exc, identity.InsufficientRole):
        # The NAMES travel with the numbers. The threshold is the platform's scale, which the
        # client has no way to learn; without these it can only say "role 8 needs 4 or lower",
        # or hardcode a copy of the scale that stops matching the day a tier is added.
        return jsonify({"error": "This account is not permitted to use the agent.",
                        "reason": "insufficient_role",
                        "role": exc.role, "requiredRole": exc.required,
                        "roleName": identity.role_name(exc.role),
                        "requiredRoleName": identity.role_name(exc.required)}), 403
    if isinstance(exc, identity.TokenMissing):
        return jsonify({"error": "Please sign in to use the agent.",
                        "reason": "not_signed_in"}), 403
    if isinstance(exc, identity.IdentityNotConfigured):
        # Fails CLOSED: a deployment that cannot verify identity must not serve as if it had.
        logger.error("Identity misconfigured: %s", exc)
        return jsonify({"error": "Server misconfiguration: identity is not configured.",
                        "reason": "identity_not_configured"}), 500
    return jsonify({"error": "Please sign in to use the agent.",
                    "reason": "token_invalid"}), 403


# ---------------------------------------------------------------------------
# Request / response normalization
# ---------------------------------------------------------------------------

def _normalize_agent_chat_request(data: dict) -> dict:
    """Normalize camelCase frontend fields to internal snake_case, accepting both."""
    user_query = _coalesce(data.get("userQuery"), data.get("user_input"))
    memory_id = _coalesce(data.get("memoryId"), data.get("memory_id"))
    thread_id = _coalesce(data.get("threadId"), data.get("thread_id"))
    conversation_name = _coalesce(data.get("conversationName"), data.get("conversation_name"))
    recent_k = _coalesce(data.get("recentK"), data.get("recent_k"))
    tool_strategy = _coalesce(data.get("toolStrategy"), data.get("tool_strategy"), "granular")
    # include_mcp_tools is tri-state: absent (None) falls back to the AGENT_INCLUDE_MCP_TOOLS
    # env default (ON), so the analyze peer's MCP tools (spatial/data) are available by default.
    from agent_runtime.langchain_mcp_tools import mcp_tools_enabled
    _mcp_raw = _coalesce(data.get("includeMcpTools"), data.get("include_mcp_tools"))
    include_mcp_tools = mcp_tools_enabled() if _mcp_raw is None else bool(_mcp_raw)
    mcp_modules = _coalesce(data.get("mcpModules"), data.get("mcp_modules"))
    enabled_search_methods = _coalesce(data.get("enabledSearchMethods"), data.get("enabled_search_methods"))
    use_persistent_memory = bool(_coalesce(data.get("usePersistentMemory"), data.get("use_persistent_memory"), True))
    smart_tool_routing = bool(_coalesce(data.get("smartToolRouting"), data.get("smart_tool_routing"), True))
    forced_intent = _coalesce(data.get("forcedIntent"), data.get("forced_intent"))
    file_paths = _coalesce(data.get("filePaths"), data.get("file_paths"))
    file_ids = _coalesce(data.get("fileIds"), data.get("file_ids"))
    skill_roots = _coalesce(data.get("skillPaths"), data.get("skill_paths"), data.get("skillRoots"), data.get("skill_roots"))
    verbose = bool(_coalesce(data.get("verbose"), False))
    # agent_dev is tri-state: present (True/False) overrides per request, absent
    # (None) falls back to the AGENT_DEV env var in the streaming layer.
    agent_dev_raw = _coalesce(data.get("agentDev"), data.get("agent_dev"))
    agent_dev = None if agent_dev_raw is None else bool(agent_dev_raw)
    # code_exec is tri-state: absent (None) falls back to the AGENT_CODE_EXEC env
    # default (ON; set AGENT_CODE_EXEC=0/false to disable). Controls the sandboxed
    # execute_code tool for this request.
    code_exec_raw = _coalesce(data.get("codeExec"), data.get("code_exec"))
    # Per-request model selection. Absent BOTH, the agent uses its configured default —
    # OpenAI gpt-4o in this deployment — so an old client behaves exactly as before.
    llm_provider = _coalesce(data.get("llmProvider"), data.get("provider"), data.get("llm_provider"))
    llm_model = _coalesce(data.get("llmModel"), data.get("model"), data.get("llm_model"))
    # Only meaningful for the gpt-5.x / o-series reasoning models; build_llm drops it for
    # models that would reject the argument.
    reasoning_effort = _coalesce(data.get("reasoningEffort"), data.get("reasoning_effort"))
    code_exec = None if code_exec_raw is None else bool(code_exec_raw)
    # Which code-peer backend runs THIS request: "langchain" (the built-in peer),
    # "opencode" or "claude". Absent, the AGENT_CODE_PEER env default applies, so a
    # client that never sends it behaves exactly as before.
    code_peer = _coalesce(data.get("codePeer"), data.get("code_peer"))
    # Model for a CLI code peer — an alias like "sonnet"/"opus", or a full id. Only
    # meaningful when that peer is the one running; the built-in peer uses the same
    # model as the answer, which the `model` field above already selects.
    code_peer_model = _coalesce(data.get("codePeerModel"), data.get("code_peer_model"))
    # Per-request A/B of the orchestration shape. Without this the two architectures could
    # only be compared by restarting the deployment between arms.
    unified_peer = _coalesce(data.get("unifiedPeer"), data.get("unified_peer"))

    return {
        "user_query": str(user_query).strip() if user_query is not None else "",
        "memory_id": str(memory_id).strip() if memory_id is not None else None,
        "thread_id": str(thread_id).strip() if thread_id is not None else None,
        "conversation_name": str(conversation_name).strip() if conversation_name is not None else None,
        "recent_k": recent_k,
        "tool_strategy": str(tool_strategy).strip(),
        "include_mcp_tools": include_mcp_tools,
        "mcp_modules": mcp_modules,
        "enabled_search_methods": enabled_search_methods,
        "use_persistent_memory": use_persistent_memory,
        "smart_tool_routing": smart_tool_routing,
        "forced_intent": forced_intent,
        "file_paths": file_paths,
        "file_ids": file_ids,
        "skill_roots": skill_roots,
        "verbose": verbose,
        "agent_dev": agent_dev,
        "code_exec": code_exec,
        "code_peer": (str(code_peer).strip() or None) if code_peer else None,
        "code_peer_model": (str(code_peer_model).strip() or None) if code_peer_model else None,
        "unified_peer": (str(unified_peer).strip().lower() in {"1", "true", "yes", "on"}
                         if unified_peer is not None else None),
        # FORCED in demo mode and DROPPED in token mode, neither of them merely defaulted. The
        # control that would let anyone change the model is the settings panel, and both modes
        # hide it — so a value left in a returning visitor's localStorage would pin them to a
        # model they can neither see nor change. That is the same trap the spatial-tools toggle
        # had. In demo it also bounds what an unauthenticated endpoint can be made to spend.
        #
        # The two differ in what replaces the client's choice. Demo names its own model, because
        # "whatever this deployment happens to be set to" is not a demo decision. Token mode
        # passes None, which means the agent uses its configured default — so changing the
        # deployment's model actually changes what signed-in users get.
        "llm_provider": _demo_model()[0] if _demo_mode() else (
            (str(llm_provider).strip() or None)
            if (llm_provider and _client_may_choose_the_model()) else None),
        "llm_model": _demo_model()[1] if _demo_mode() else (
            (str(llm_model).strip() or None)
            if (llm_model and _client_may_choose_the_model()) else None),
        "reasoning_effort": (str(reasoning_effort).strip() or None) if reasoning_effort else None,
    }


def _format_agent_chat_result(payload: dict) -> dict:
    """Align output shape to the legacy Node /llm/search result object."""
    answer = payload.get("answer") or ""
    elements = payload.get("elements") or []
    retrieval_steps = payload.get("retrievalSteps") or payload.get("retrieval_steps") or []
    react_history = payload.get("reactHistory") or payload.get("react_history") or []

    formatted = {
        "answer": answer,
        "message_id": payload.get("message_id"),
        "elements": elements,
        "count": int(payload.get("count") or (len(elements) if isinstance(elements, list) else 0)),
        "retrievalSteps": retrieval_steps,
        "reactHistory": react_history,
        "memoryId": payload.get("memoryId") or payload.get("memory_id"),
        "threadId": payload.get("threadId") or payload.get("thread_id"),
        "routeTrace": payload.get("routeTrace") or payload.get("route_trace") or {},
        "artifacts": payload.get("artifacts") or {},
        "agent_result": payload.get("agent_result") or {},
        "fileIds": payload.get("fileIds") or payload.get("file_ids") or [],
        "filePaths": payload.get("filePaths") or payload.get("file_paths") or [],
        "skillPaths": payload.get("skillPaths") or payload.get("skill_roots") or [],
        "availableSkills": payload.get("availableSkills") or payload.get("available_skills") or [],
        "opengeodata_results": payload.get("opengeodata_results") or [],
    }
    warning = payload.get("warning")
    if warning:
        formatted["warning"] = warning
    return formatted


def _humanize_stage(value: str) -> str:
    text = str(value or "").strip().replace("_", " ")
    return text[:1].upper() + text[1:] if text else "Status"


def _category_for_agent_role(role: object) -> str:
    role_text = str(role or "").strip().lower()
    if role_text in {"search", "search_agent"} or "search" in role_text:
        return "search"
    if role_text in {"analysis", "analysis_agent", "code", "code_agent", "verification"}:
        return "analysis"
    return "analysis"


# 900 chars cut a python traceback off before its last frame — the one naming the error — so the
# expandable trace row had nothing useful to expand into. 4000 is enough for a stack trace and a
# tool's argument dict while still being a bound: the trace is a stream, not a transport for
# results, and a 15 MB GeoJSON must never travel down it.
_TRACE_TEXT_LIMIT = int(os.getenv("AGENT_TRACE_SSE_TEXT_LIMIT") or 4000)


def _short_text(value: object, limit: int = _TRACE_TEXT_LIMIT) -> str:
    text = value if isinstance(value, str) else str(value or "")
    text = " ".join(text.split())
    return text if len(text) <= limit else f"{text[:limit]}..."


def _agent_trace_event(payload: dict) -> dict:
    """Convert low-level LLM/tool payloads into browser-friendly trace events."""
    kind = str(payload.get("kind") or payload.get("type") or "").strip()
    if kind == "llm_tool_decision":
        calls = payload.get("tool_calls") or []
        if not calls and payload.get("name"):
            calls = [{"name": payload.get("name"), "args": payload.get("args") or {}}]
        return {
            "type": "tool_call",
            "agent": payload.get("agent") or "",
            "label": "LLM tool decision",
            "message": "; ".join(
                f"{call.get('name', 'unknown_tool')}({json.dumps(call.get('args') or {}, ensure_ascii=True, default=str)})"
                for call in calls
                if isinstance(call, dict)
            ),
            "tool_calls": calls,
            "detail": payload,
        }
    if kind == "tool_result":
        tool_name = payload.get("tool_name") or payload.get("name") or ""
        content = payload.get("content")
        parsed = payload.get("parsed")
        return {
            "type": "tool_result",
            "agent": payload.get("agent") or "",
            "label": f"Tool result {tool_name}".strip(),
            "message": _short_text(parsed if parsed is not None else content),
            "tool_name": tool_name,
            "detail": payload,
        }
    if kind == "llm_message":
        return {
            "type": "llm_message",
            "agent": payload.get("agent") or "",
            "label": "LLM message",
            "message": _short_text(payload.get("content") or ""),
            "detail": payload,
        }
    if kind == "llm_start":
        return {
            "type": "llm_start",
            "agent": payload.get("agent") or "",
            "label": payload.get("label") or "LLM request",
            "message": _short_text(payload.get("message") or "LLM request started"),
            "detail": payload,
        }
    if kind == "llm_error":
        return {
            "type": "llm_error",
            "agent": payload.get("agent") or "",
            "label": payload.get("label") or "LLM error",
            "message": _short_text(payload.get("message") or payload),
            "detail": payload,
        }
    if kind in {"mcp_call_start", "mcp_call_end", "mcp_call_error"}:
        tool_name = payload.get("tool_name") or payload.get("mcp_tool_name") or "mcp tool"
        labels = {
            "mcp_call_start": "MCP call started",
            "mcp_call_end": "MCP call completed",
            "mcp_call_error": "MCP call failed",
        }
        return {
            "type": kind,
            "agent": payload.get("agent") or "",
            "label": payload.get("label") or labels[kind],
            "message": _short_text(payload.get("message") or tool_name),
            "tool_name": tool_name,
            "detail": payload,
        }
    if kind == "tool_error":
        tool_name = payload.get("tool_name") or payload.get("name") or ""
        return {
            "type": "tool_error",
            "agent": payload.get("agent") or "",
            "label": f"Tool error {tool_name}".strip(),
            "message": _short_text(payload.get("message") or payload),
            "tool_name": tool_name,
            "detail": payload,
        }
    if kind == "agent_route_decision":
        route_trace = payload.get("route_trace") or {}
        route = route_trace.get("route") or payload.get("route") or "unknown"
        intent = route_trace.get("intent") or payload.get("intent") or "unknown"
        allowed = payload.get("allowed_tools") or route_trace.get("allowed_tools") or []
        extra = f"; tools={', '.join(map(str, allowed[:6]))}" if isinstance(allowed, list) and allowed else ""
        return {
            "type": "route_decision",
            "agent": payload.get("agent") or "",
            "label": "Route decision",
            "message": f"route={route}; intent={intent}{extra}",
            "detail": payload,
        }
    return {
        "type": kind or "message",
        "agent": payload.get("agent") or "",
        "label": kind or "Trace",
        "message": _short_text(payload.get("content") or payload.get("message") or payload),
        "detail": payload,
    }


def _diagnostic_agent_trace_events(diagnostics: object) -> list[dict]:
    """Expand recursion diagnostics into readable SSE trace events."""
    if not isinstance(diagnostics, dict):
        return []
    events: list[dict] = []
    call_counts = diagnostics.get("tool_call_counts")
    if isinstance(call_counts, dict) and call_counts:
        events.append(
            {
                "type": "diagnostic_summary",
                "agent": "agent",
                "label": "Recursion diagnostic",
                "message": "Tool calls observed: "
                + ", ".join(f"{name} x{count}" for name, count in sorted(call_counts.items())),
                "detail": {
                    "thread_id": diagnostics.get("thread_id"),
                    "recursion_limit": diagnostics.get("recursion_limit"),
                    "tool_call_counts": call_counts,
                },
            }
        )

    for idx, item in enumerate(diagnostics.get("recent_messages") or [], start=1):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").lower()
        content = _short_text(item.get("content") or "")
        tool_calls = item.get("tool_calls")
        name = str(item.get("name") or "").strip()
        if role in {"human", "humanmessage", "user"}:
            events.append(
                {
                    "type": "user_message",
                    "agent": "",
                    "label": "User",
                    "message": content,
                    "sequence": idx,
                    "detail": item,
                }
            )
        elif isinstance(tool_calls, list) and tool_calls:
            events.append(
                {
                    "type": "tool_call",
                    "agent": "llm",
                    "label": "LLM tool decision",
                    "message": "; ".join(
                        f"{call.get('name', 'unknown_tool')}({json.dumps(call.get('args') or {}, ensure_ascii=True, default=str)})"
                        for call in tool_calls
                        if isinstance(call, dict)
                    ),
                    "sequence": idx,
                    "tool_calls": tool_calls,
                    "detail": item,
                }
            )
        elif role in {"tool", "toolmessage"} or name:
            events.append(
                {
                    "type": "tool_result",
                    "agent": "tool",
                    "label": f"Tool result {name}".strip(),
                    "message": content,
                    "sequence": idx,
                    "tool_name": name,
                    "detail": item,
                }
            )
        elif role in {"ai", "aimessage", "assistant"}:
            events.append(
                {
                    "type": "llm_message",
                    "agent": "llm",
                    "label": "LLM message",
                    "message": content or "(empty message)",
                    "sequence": idx,
                    "detail": item,
                }
            )
    return events


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _format_elements(retrieved_documents):
    """Convert internal evidence entries into the simplified element payload expected by the UI."""
    elements = []
    opengeodata_count = 0
    for entry in retrieved_documents:
        document = (entry or {}).get("document") or {}
        metadata = (entry or {}).get("metadata") or {}
        source = entry.get("source", "unknown")
        resource_type = document.get("resource-type") or document.get("element_type")

        if source == "opengeodata" or resource_type == "opengeodata":
            opengeodata_count += 1

        elements.append(
            {
                "_id": document.get("doc_id") or metadata.get("hit_id"),
                "_score": entry.get("score"),
                "contributor": document.get("contributor"),
                "contents": document.get("contents"),
                "resource-type": resource_type,
                "title": document.get("title"),
                "authors": document.get("authors") or [],
                "tags": document.get("tags") or [],
                "thumbnail-image": document.get("thumbnail-image") or document.get("thumbnail_image"),
                "source": source,
            }
        )
    if opengeodata_count > 0:
        logger.info(f"Formatted {opengeodata_count} opengeodata results in {len(elements)} total elements")
    return elements


def _pipeline_params(data):
    return data.get('params', {
        "top_k": 8,
        "max_context_tokens": 6000,
        "enable_llm_reranker": True
    })


def _pipeline_response(result):
    evidence = result.get("evidence", {}) or {}
    retrieved_documents = evidence.get("retrieved_documents", []) or []
    elements = _format_elements(retrieved_documents)
    trace = result.get("trace_observability", {}) or {}
    planner = result.get("planner_reasoning", {}) or {}

    return {
        "answer": (
            (result.get("answer") or {}).get("final_composed_answer")
            or "No answer"
        ),
        "message_id": str(uuid4()),
        "elements": elements,
        "count": len(elements),
        "retrievalSteps": trace.get("retrieval_routing_decisions") or [],
        "reactHistory": planner.get("react_history") or trace.get("react_history") or [],
    }


def _parse_mcp_modules(value):
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    raise ValueError("mcp_modules must be a list of strings or a comma-separated string")


def _parse_enabled_search_methods(value):
    """Validate + normalize the retrieval allowlist (unknown name -> ValueError -> HTTP 400).

    An unrecognized method used to be dropped silently, leaving the agent with no retrieval tools
    and an unexplained "no evidence" answer; see agent_runtime.search_methods.
    """
    from agent_runtime.search_methods import normalize_search_methods

    return normalize_search_methods(value)


def _sse_event(name, data):
    payload = json.dumps(data or {}, ensure_ascii=True, default=str)
    return f"event: {name}\ndata: {payload}\n\n"


def _normalize_uploaded_files():
    files = []
    files.extend(request.files.getlist("files"))
    single = request.files.get("file")
    if single is not None:
        files.append(single)
    return [item for item in files if getattr(item, "filename", None)]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint.

    ---
    tags:
      - Health
    produces:
      - application/json
    responses:
      200:
        description: Service is healthy.
        schema:
          type: object
          properties:
            status:
              type: string
              example: healthy
            service:
              type: string
              example: rag-pipeline
    """
    return jsonify({"status": "healthy", "service": "rag-pipeline"}), 200


@app.route('/agent/dashboard', methods=['GET'])
def agent_dashboard():
    """Serve the local streaming agent dashboard."""
    dashboard_path = Path(__file__).resolve().parent.parent / "examples" / "agent_chat_stream_demo.html"
    return send_file(dashboard_path)


def _list_code_peers():
    """The code-peer backends a request may select, and which one is the default.

    A CLI peer needs BOTH a credential and its sandbox image. Reporting only the
    credential offered opencode as available on a host where its image had never
    been built, which fails at run time with a docker error — the picker has to
    tell "not configured" apart from "not built".
    """
    import os as _os
    import subprocess as _sp

    from agent_runtime.claude_peer import (DEFAULT_CLAUDE_IMAGE, SELECTABLE_MODELS,
                                           resolve_claude_settings, selects_claude)
    from agent_runtime.opencode_peer import (CODE_PEER_ENV, DEFAULT_OPENCODE_IMAGE,
                                             resolve_llm_settings, selects_opencode)

    def _image_present(env_var, default_name):
        name = _os.getenv(env_var, default_name)
        try:
            return _sp.run(["docker", "image", "inspect", name],
                           capture_output=True, timeout=5).returncode == 0
        except Exception:  # noqa: BLE001 - no docker, no answer, never a page error
            return False

    env_value = _os.getenv(CODE_PEER_ENV) or ""
    default = ("opencode" if selects_opencode(env_value)
               else "claude" if selects_claude(env_value) else "langchain")
    claude = resolve_claude_settings()
    try:
        opencode_ready = bool(resolve_llm_settings().get("api_key"))
    except Exception:  # noqa: BLE001
        opencode_ready = False
    claude_image = _image_present("AGENT_CLAUDE_IMAGE", DEFAULT_CLAUDE_IMAGE)
    opencode_image = _image_present("AGENT_OPENCODE_IMAGE", DEFAULT_OPENCODE_IMAGE)
    return {
        "default": default,
        "peers": [
            {"id": "langchain", "label": "Built-in peer (generates code, runs execute_code)",
             "available": True},
            {"id": "claude", "label": "Claude Code CLI (sandboxed, iterates on its own)",
             "available": bool(claude["credential"]) and claude_image,
             "reason": None if (claude["credential"] and claude_image)
                       else ("no credential" if not claude["credential"] else "image not built"),
             "model": claude["model"], "auth": claude["auth"],
             # Aliases, so the list cannot pin itself to a retired id. The CLI refuses
             # an unknown one locally, before any API call, so a bad pick costs nothing.
             "models": list(SELECTABLE_MODELS)},
            {"id": "opencode", "label": "opencode CLI (sandboxed, iterates on its own)",
             "available": opencode_ready and opencode_image,
             "reason": None if (opencode_ready and opencode_image)
                       else ("no credential" if not opencode_ready else "image not built")},
        ],
    }


@app.route('/agent/ui-config', methods=['GET'])
def agent_ui_config():
    """What the browser needs to know before it can render its own chrome.

    Deliberately UNAUTHENTICATED and deliberately tiny: a client that cannot yet authenticate is
    exactly the client that needs to ask whether it has to. It answers only whether this
    deployment is in demo mode and whether a key is required — never the key itself, and nothing
    else about the configuration.
    """
    demo = _demo_mode()
    body = {
        # `mode` is the field to read. `demo_mode` stays for clients built before modes existed:
        # dropping it would blank the settings panel on every page still holding an old bundle.
        "mode": deployment_mode.current_mode(),
        "demo_mode": demo,
        "api_key_required": bool(_get_agent_chat_api_key()) and not demo,
    }
    if deployment_mode.is_token():
        # Where the CLIENT refreshes an expired token. The agent never handles refresh tokens:
        # the browser calls the platform directly, which re-mints the .i-guide.io cookie. Sent
        # from here rather than compiled into the bundle so the same build runs against any
        # tier — the dev and production backends are different hosts.
        body["refresh_url"] = platform_endpoints.refresh_url()
        body["signin_url"] = platform_endpoints.signin_url()
    return jsonify(body)


@app.route('/agent/whoami', methods=['GET'])
def agent_whoami():
    """
    Who the server thinks you are, and — when it thinks nobody — why.
    ---
    tags:
      - agent
    produces:
      - application/json
    responses:
      200:
        description: >-
          `{ mode, verify, signedIn, user: {id, role, roleName?}|null, permitted, reason,
          cookiesSeen, expectedCookie, platformTier, checkTokensUrl, signinUrl, requiredRole,
          requiredRoleName }`. Always 200, even when the caller is anonymous or refused: this
          endpoint exists to EXPLAIN a refusal, so answering with one would defeat it.
          Everything the profile renders comes from here, including where to sign in.
    """
    body = {
        "mode": deployment_mode.current_mode(),
        "signedIn": False,
        "user": None,
        "permitted": False,
        "reason": None,
        # Names only, never values. A cookie value is a live credential; the NAME is what is
        # actually in question when a tier turns out to use a different one than configured,
        # and no amount of guessing beats the server saying what arrived.
        "cookiesSeen": sorted(request.cookies.keys()),
        "expectedCookie": identity.cookie_name(),
        # Machine-readable twin of `reason`. Clients act on this; `reason` is for a human
        # reading a log or a support request.
        "reasonCode": None,
    }
    # Every field below reads configuration that REFUSES TO GUESS: an unrecognised
    # PLATFORM_TIER raises rather than picking a platform, and an unparseable AGENT_MIN_ROLE
    # raises rather than widening access. Both are right — and both would turn this endpoint
    # into a 500 in precisely the situation it exists to explain. So each is reported when it
    # can be read, left null when it cannot, and the misconfiguration itself becomes the
    # `reason`, which is the one place a human is going to look.
    #
    # `signinUrl` is here, and not only on /agent/ui-config, because the profile renders from
    # this response alone: the state that most needs a sign-in link is the signed-out one,
    # where whoami already knows everything else.
    for key, read in (
        ("platformTier", platform_endpoints.current_tier),
        ("checkTokensUrl", platform_endpoints.check_tokens_url),
        ("signinUrl", platform_endpoints.signin_url),
        ("requiredRole", identity.min_role),
    ):
        try:
            body[key] = read()
        except (identity.IdentityError, ValueError) as exc:
            body[key] = None
            body["reason"] = f"{type(exc).__name__}: {exc}"
    body["requiredRoleName"] = identity.role_name(body.get("requiredRole"))

    try:
        body["verify"] = identity.verify_mode()
    except identity.IdentityError as exc:
        body["verify"] = None
        body["reason"], body["reasonCode"] = str(exc), _identity_reason(exc)
        return jsonify(body)

    if not deployment_mode.is_token():
        body["reason"] = "this deployment does not identify callers"
        body["reasonCode"] = "no_identity_in_this_mode"
        return jsonify(body)

    token = _extract_user_token()
    if not token:
        body["reason"] = "no access token was presented"
        body["reasonCode"] = "not_signed_in"
        return jsonify(body)
    try:
        user = identity.identify(token)
    except identity.IdentityError as exc:
        # `token_expired` here is the one a client must ACT on rather than display: the access
        # cookie aged out, the refresh cookie almost certainly has not, and telling someone to
        # sign in again when one silent refresh would do it is the worst of the options.
        body["reason"], body["reasonCode"] = f"{type(exc).__name__}: {exc}", _identity_reason(exc)
        return jsonify(body)

    body["signedIn"] = True
    body["user"] = user.to_dict()
    try:
        identity.authorize(user)
        body["permitted"] = True
    except identity.InsufficientRole as exc:
        body["reason"], body["reasonCode"] = str(exc), _identity_reason(exc)
    return jsonify(body)


@app.route('/agent/conversations', methods=['GET'])
def agent_conversations():
    """
    The signed-in user's own conversations, newest first.
    ---
    tags:
      - agent
    produces:
      - application/json
    responses:
      200:
        description: >-
          `{ "conversations": [ { memoryId, conversationName, createdAt, updatedAt } ] }`.
          Summaries only, never transcripts. Outside token mode this is always empty: a
          deployment that identifies nobody has no "your" conversations to list.
      401:
        description: The access token expired. Refresh it and retry.
      403:
        description: Not signed in, or this account is not permitted to use the agent.
    """
    try:
        try:
            user = _require_user()
        except identity.IdentityError as exc:
            return _identity_error_response(exc)
        if not user:
            # Deliberately 200-with-nothing rather than an error: dev and demo have no users,
            # and a client that shows a history pane should render it empty, not break.
            return jsonify({"conversations": []})
        token = identity.set_user(user)
        try:
            return jsonify({"conversations": list_memories(limit=50)})
        finally:
            identity.reset_user(token)
    except Exception as exc:  # noqa: BLE001
        logger.error("Error listing conversations: %s", exc, exc_info=True)
        return jsonify({"error": f"Internal server error: {exc}"}), 500


@app.route('/agent/conversations/<memory_id>', methods=['GET', 'PUT'])
def agent_conversation(memory_id):
    """
    Read or write the client's view of one conversation.
    ---
    tags:
      - agent
    produces:
      - application/json
    parameters:
      - in: path
        name: memory_id
        type: string
        required: true
      - in: body
        name: body
        required: false
        description: >-
          On PUT, the client's own conversation record — the `StoredSession` shape from
          `map-ui-prototype/src/sessionStore.ts`: messages, layer DESCRIPTORS (a sourceUrl to
          re-fetch, or small inline geometry), every fileId used across the session, region,
          model and provider. `owner_id`, `chat_history` and the timestamps are server-owned and
          ignored if sent.
        schema:
          type: object
    responses:
      200:
        description: The stored record (GET), or what was written (PUT).
      401:
        description: The access token expired. Refresh it and retry.
      403:
        description: Not signed in, or not permitted.
      404:
        description: No such conversation, or it belongs to someone else.
      413:
        description: The record is larger than this store will hold.
    """
    try:
        try:
            user = _require_user()
        except identity.IdentityError as exc:
            return _identity_error_response(exc)
        token = identity.set_user(user)
        try:
            try:
                assert_memory_owner(memory_id)
            except MemoryAccessDenied:
                # Same 404-not-403 rule as everywhere else: a 403 confirms the id exists.
                logger.info("Conversation refused: %s is not this caller's", memory_id)
                return jsonify({"error": "No conversation found for that id.",
                                "reason": "not_your_conversation"}), 404

            if request.method == 'GET':
                snapshot = get_session_snapshot(memory_id)
                if snapshot is None:
                    return jsonify({"error": "No conversation found for that id."}), 404
                return jsonify(snapshot)

            body = request.get_json(silent=True)
            if not isinstance(body, dict):
                return jsonify({"error": "Body must be a conversation object."}), 400
            try:
                return jsonify(save_session_snapshot(memory_id, body))
            except SnapshotTooLarge as exc:
                # A real ceiling rather than a silent truncation: a conversation that came back
                # missing half its layers would look like data loss with no explanation.
                return jsonify({"error": str(exc), "reason": "conversation_too_large"}), 413
        finally:
            identity.reset_user(token)
    except Exception as exc:  # noqa: BLE001
        logger.error("Error on conversation %s: %s", memory_id, exc, exc_info=True)
        return jsonify({"error": f"Internal server error: {exc}"}), 500


@app.route('/agent/conversations/<memory_id>/traces', methods=['GET'])
def agent_conversation_traces(memory_id):
    """
    The raw trace of every recorded turn in this conversation.
    ---
    tags:
      - agent
    parameters:
      - in: path
        name: memory_id
        required: true
        type: string
      - in: query
        name: traceId
        type: string
        description: >-
          Return this one turn with its full event list. Without it, a summary per turn and no
          events -- a conversation's traces are large, and a caller choosing which turn to look
          at should not have to download all of them to choose.
    produces:
      - application/json
    responses:
      200:
        description: >-
          `{ traces: [ { traceId, query, answer, model, provider, eventCount, createdAt } ] }`,
          or a single `{ traceId, events: [...] }` when `traceId` is given.
      404:
        description: No such conversation or trace, or it belongs to someone else.
    """
    try:
        try:
            user = _require_user()
        except identity.IdentityError as exc:
            return _identity_error_response(exc)
        token = identity.set_user(user)
        try:
            try:
                # Ownership is asserted on the CONVERSATION, not on the trace: the trace is a
                # property of the conversation and inherits its access, and checking the parent
                # means a trace can never be reachable by an id its conversation would refuse.
                assert_memory_owner(memory_id)
            except MemoryAccessDenied:
                logger.info("Traces refused: %s is not this caller's", memory_id)
                return jsonify({"error": "No conversation found for that id.",
                                "reason": "not_your_conversation"}), 404

            trace_id = (request.args.get("traceId") or "").strip()
            if trace_id:
                trace = get_turn_trace(trace_id)
                # The id embeds its conversation, but that is a convenience and not a check:
                # confirm the trace really belongs to the conversation just authorised.
                if not trace or trace.get("memory_id") != memory_id:
                    return jsonify({"error": "No trace found for that id."}), 404
                return jsonify(trace)

            limit = request.args.get("limit", type=int) or 20
            return jsonify({"traces": list_turn_traces(memory_id, limit=limit)})
        finally:
            identity.reset_user(token)
    except Exception as exc:  # noqa: BLE001
        logger.error("Error on traces for %s: %s", memory_id, exc, exc_info=True)
        return jsonify({"error": f"Internal server error: {exc}"}), 500


@app.route('/agent/models', methods=['GET'])
def agent_models():
    """
    List the chat models a request may select, per provider.
    ---
    tags:
      - agent
    produces:
      - application/json
    responses:
      200:
        description: >-
          Selectable models. `default` is what a request with no `model`/`provider` uses —
          OpenAI gpt-4o in this deployment. AnvilGPT's list is fetched live from its own
          /api/models, so an id it no longer serves is never offered; `stale: true` on a
          provider means that fetch failed and known ids are being shown instead.
        schema:
          type: object
          properties:
            default:
              type: object
            providers:
              type: array
              items:
                type: object
    """
    from agent_runtime.executor_factory import list_available_models

    try:
        catalogue = list_available_models()
    except Exception as exc:  # never let a catalogue lookup break the page
        logger.warning("model catalogue unavailable: %s", exc)
        catalogue = {"default": {"provider": "openai", "model": "gpt-4o-2024-11-20"},
                     "providers": [], "error": str(exc)}
    # The code PEER is a second, independent axis: which agent writes the code, not
    # which model writes the answer. It rides along on this endpoint so a client
    # needs one call, but it is reported under its own key so nothing conflates them.
    try:
        catalogue["code_peers"] = _list_code_peers()
    except Exception as exc:  # noqa: BLE001 - a peer list must never break the page
        logger.warning("code peer catalogue unavailable: %s", exc)
    return jsonify(catalogue)


@app.route('/agent/files/upload', methods=['POST'])
def upload_agent_files():
    """
    Upload one or more files for the agent to inspect.

    The returned `file_id` values can be passed to `/agent/chat` or
    `/agent/chat/stream` via the JSON field `fileIds`.

    ---
    tags:
      - Agent Files
    consumes:
      - multipart/form-data
    produces:
      - application/json
    parameters:
      - in: formData
        name: file
        type: file
        required: false
        description: Single file upload.
      - in: formData
        name: files
        type: file
        required: false
        description: Multi-file upload (repeat this field per file).
    responses:
      200:
        description: Upload succeeded.
        schema:
          type: object
          properties:
            files:
              type: array
              items:
                type: object
                properties:
                  file_id:
                    type: string
                    example: file_0123456789ab
                  filename:
                    type: string
                    example: data.csv
                  kind:
                    type: string
                    example: upload
                  size_bytes:
                    type: integer
                    example: 1234
                  download_url:
                    type: string
                    example: /agent/files/file_0123456789ab/download
            count:
              type: integer
              example: 2
      400:
        description: No files provided or invalid request.
      500:
        description: Internal server error.
    """
    try:
        # This endpoint had NO guard of any kind — not identity, not even the API key — so the
        # deployed server accepted uploads from anyone who could reach it, and the files landed
        # in a store the download endpoint then served. Measured live before this fix: a keyless
        # POST returned 200.
        #
        # Identity first, then the key, exactly as the chat endpoints do: a signed-in caller is
        # already a credential, and running the key gate first refuses them for lacking one they
        # have no way to supply.
        try:
            _upload_user = _require_user()
        except identity.IdentityError as exc:
            return _identity_error_response(exc)
        try:
            _require_agent_chat_api_key(_upload_user)
        except PermissionError as exc:
            return jsonify({"error": str(exc)}), 403
        except RuntimeError as exc:
            logger.error("Agent file upload API key misconfigured: %s", exc)
            return jsonify({"error": "Server misconfiguration: API key not set"}), 500

        files = _normalize_uploaded_files()
        if not files:
            return jsonify({"error": "No files uploaded. Use form field `file` or `files`."}), 400

        # Bound so save_uploaded_file stamps the owner. Without this every upload was written
        # owner_id: None, which made AGENT_TOKEN_STRICT=1 unreachable in practice: a user could
        # not download their own attachment.
        _upload_token = identity.set_user(_upload_user)
        try:
            uploaded = [save_uploaded_file(file_storage) for file_storage in files]
        finally:
            identity.reset_user(_upload_token)
        return jsonify({"files": uploaded, "count": len(uploaded)}), 200
    except ValueError as e:
        logger.error(f"Agent file upload validation error: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error uploading agent files: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route('/agent/files/<file_id>/download', methods=['GET'])
def download_agent_file(file_id):
    """
    Download an uploaded (or generated) file by file_id.

    ---
    tags:
      - Agent Files
    produces:
      - application/octet-stream
    parameters:
      - in: path
        name: file_id
        type: string
        required: true
        description: File identifier returned by `/agent/files/upload` or `write_output_file`.
    responses:
      200:
        description: File bytes.
      404:
        description: Unknown file_id.
      500:
        description: Internal server error.
    """
    try:
        # Identity matters here as much as on /agent/chat, and for a while it was missing: this
        # endpoint served ANY file to ANYONE who could guess or be handed an id, which also made
        # every download_url in an answer a permanent public link.
        try:
            _download_user = _require_user()
        except identity.IdentityError as exc:
            return _identity_error_response(exc)
        _download_token = identity.set_user(_download_user)
        try:
            record = require_file_record(file_id)
            # Unowned records are readable only while the backfill is still running
            # (AGENT_TOKEN_STRICT=0). Once strict, a file nobody owns is a file nobody reads.
            if not file_store_may_read(record, allow_unowned=not _token_strict()):
                # 404 rather than 403: a 403 would confirm that this id exists, turning the
                # endpoint into an oracle for enumerating other people's files.
                logger.info("Download refused: %s does not belong to this caller", file_id)
                return jsonify({"error": f"No file found for id {file_id}"}), 404
            path = resolve_file_id(file_id)
        finally:
            identity.reset_user(_download_token)
        mimetype = None
        suffix = Path(record.get("filename", "")).suffix.lower()
        image_types = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".gif": "image/gif", ".webp": "image/webp", ".svg": "image/svg+xml",
            ".bmp": "image/bmp", ".avif": "image/avif",
        }
        if suffix == ".csv":
            mimetype = "text/csv"
        elif suffix == ".json":
            mimetype = "application/json"
        elif suffix in image_types:
            mimetype = image_types[suffix]
        elif suffix in {".txt", ".md", ".py"}:
            mimetype = "text/plain"
        # Raster images are previewable inline (so an <img>/new-tab view renders instead of
        # forcing a download); everything else, and any explicit ?download=1, is an attachment.
        # SVG is EXCLUDED from inline: it can carry script that executes on top-level navigation
        # (stored XSS). An <img src> still renders an SVG fine; only a direct tab-open would have
        # run it — and that now downloads instead.
        inline_ok = suffix in image_types and suffix != ".svg"
        force_download = request.args.get("download") in ("1", "true", "yes")
        resp = send_file(path, as_attachment=(force_download or not inline_ok),
                         download_name=record.get("filename") or path.name, mimetype=mimetype)
        # Never let a browser content-sniff a stored file into an executable type.
        resp.headers["X-Content-Type-Options"] = "nosniff"
        return resp
    except ValueError as e:
        logger.error(f"Agent file download validation error: {str(e)}")
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Error downloading agent file {file_id}: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route('/query', methods=['POST'])
def query():
    """
Main RAG pipeline endpoint with LLM routing and optional reranking.

This endpoint accepts a natural language query and executes the full
retrieval-augmented generation (RAG) pipeline, including:
- retriever selection / routing
- optional LLM-based reranking
- context assembly
- answer synthesis
- trace and provenance collection

---
tags:
  - RAG
  - Query
consumes:
  - application/json
produces:
  - application/json

parameters:
  - in: body
    name: body
    required: true
    schema:
      type: object
      required:
        - user_input
      properties:
        user_input:
          type: string
          description: Natural language query from the user.
          example: "What is telecoupling in sustainability science?"
        memory_id:
          type: string
          nullable: true
          description: Optional session or memory identifier for conversational context.
          example: "session-abc123"
        params:
          type: object
          description: Parameters controlling retrieval and generation behavior.
          properties:
            top_k:
              type: integer
              default: 8
              description: Number of documents to retrieve.
            max_context_tokens:
              type: integer
              default: 6000
              description: Maximum token budget for assembled context.
            enable_llm_reranker:
              type: boolean
              default: true
              description: Whether to apply an LLM-based reranker to retrieved documents.
        session_context:
          type: object
          description: Arbitrary session-level context passed to the pipeline.
          example: {}
        recent_k:
          type: integer
          description: Number of recent conversational turns to include.
          example: 5
        extra_state:
          type: object
          description: Optional additional state injected into the pipeline.
          example: {}

responses:
  200:
    description: Query processed successfully.
    schema:
      type: object
      properties:
        answer:
          type: string
          description: Final composed answer generated by the RAG pipeline.
        message_id:
          type: string
          description: Unique identifier for this query-response pair.
        elements:
          type: array
          description: Retrieved and formatted evidence elements.
          items:
            type: object
        count:
          type: integer
          description: Number of retrieved evidence elements.
        retrievalSteps:
          type: array
          description: Retrieval routing and decision trace.
          items:
            type: object
        reactHistory:
          type: array
          description: ReAct-style reasoning or planner trace, if available.
          items:
            type: object

  400:
    description: Invalid request (e.g., missing user_input).
    schema:
      type: object
      properties:
        error:
          type: string

  500:
    description: Internal server error during pipeline execution.
    schema:
      type: object
      properties:
        error:
          type: string
"""
    try:
        data = request.get_json() or {}
        user_input = data.get('user_input')

        if not user_input:
            return jsonify({"error": "user_input is required"}), 400

        logger.info(f"Processing query: {user_input[:100]}...")

        result = run_pipeline(
            user_input=user_input,
            memory_id=data.get('memory_id'),
            params=_pipeline_params(data),
            session_context=data.get('session_context', {}),
            recent_k=data.get('recent_k'),
            extra_state=data.get('extra_state', {})
        )
        response = _pipeline_response(result)

        logger.info(f"Query processed successfully. Retrieved {response['count']} documents.")
        return jsonify(response), 200

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route('/agent/chat', methods=['POST'])
def agent_chat():
    """
    Non-streaming agent chat — the single-response variant of `POST /agent/chat/stream`, which
    is **the primary agent chat endpoint**. It accepts the SAME request body, fields, and
    defaults (see `/agent/chat/stream` for the full field list, the minimum request, and the
    file-upload flow); camelCase or snake_case are both accepted (camelCase wins) and only
    `userQuery` is required. Instead of an SSE stream, this returns the final result as one JSON
    response. The streaming-only `agentDev` field does not apply here.

    ---
    tags:
      - Agent Chat
    consumes:
      - application/json
    produces:
      - application/json
    parameters:
      - in: header
        name: X-API-KEY
        type: string
        required: false
        description: API key for agent chat endpoints when env var AGENT_CHAT_API_KEY is configured. The Authorization header with a Bearer token is also accepted.
      - in: body
        name: body
        required: true
        description: Agent chat request. CamelCase frontend fields and snake_case backend aliases are both accepted; when both are present, the camelCase value is used.
        schema:
          type: object
          required:
            - userQuery
          properties:
            userQuery:
              type: string
              description: Natural-language user prompt. Required unless the snake_case alias `user_input` is supplied.
              example: Inspect the uploaded Chicago crime CSV and summarize trends by primary type.
            user_input:
              type: string
              description: Snake_case alias for `userQuery`.
              example: Inspect the uploaded Chicago crime CSV and summarize trends by primary type.
            memoryId:
              type: string
              nullable: true
              description: Persistent memory id. If omitted and `usePersistentMemory` is true, a new memory record is created. Also used as the thread id when `threadId` is omitted.
              example: agent-memory-chicago-001
            memory_id:
              type: string
              nullable: true
              description: Snake_case alias for `memoryId`.
              example: agent-memory-chicago-001
            threadId:
              type: string
              nullable: true
              description: LangGraph/checkpointer thread id for agent execution. If omitted, `memoryId` is used.
              example: agent-thread-chicago-001
            thread_id:
              type: string
              nullable: true
              description: Snake_case alias for `threadId`.
              example: agent-thread-chicago-001
            conversationName:
              type: string
              nullable: true
              description: Friendly name used when the endpoint creates a new persistent memory.
              example: Chicago crime analysis
            conversation_name:
              type: string
              nullable: true
              description: Snake_case alias for `conversationName`.
              example: Chicago crime analysis
            recentK:
              type: integer
              nullable: true
              description: Number of recent memory turns to include in chat history. Use 0 to ignore history for this turn.
              default: null
              example: 8
            recent_k:
              type: integer
              nullable: true
              description: Snake_case alias for `recentK`.
              example: 8
            toolStrategy:
              type: string
              enum:
                - granular
                - full_pipeline
              default: granular
              description: Agent tool mode. `granular` exposes individual keyword, semantic, Neo4j, spatial, OpenGeoData, QGIS, file, and skill tools. `full_pipeline` exposes the compatibility RAG pipeline tool.
              example: granular
            tool_strategy:
              type: string
              enum:
                - granular
                - full_pipeline
              description: Snake_case alias for `toolStrategy`.
              example: granular
            includeMcpTools:
              type: boolean
              nullable: true
              description: Include MCP-backed tools (spatial/data analysis) in the agent toolset. Omit to use the server default from AGENT_INCLUDE_MCP_TOOLS (ON); send false to disable for this request.
              example: true
            include_mcp_tools:
              type: boolean
              description: Snake_case alias for `includeMcpTools`.
              example: false
            mcpModules:
              type: array
              items:
                type: string
              nullable: true
              description: MCP module allowlist. May be an array of module names or a comma-separated string.
              example: ["search_tools", "data_tools"]
            mcp_modules:
              type: array
              items:
                type: string
              nullable: true
              description: Snake_case alias for `mcpModules`.
              example: ["search_tools", "data_tools"]
            enabledSearchMethods:
              type: array
              items:
                type: string
                enum:
                  - keyword_search
                  - semantic_search
                  - neo4j_search
                  - neo4j_get_element_by_id
                  - neo4j_explore_related_nodes
                  - spatial_search
                  - opengeodata_search
                  - web_search
                  - web_fetch
                  - agent_kb_search
                  - get_kb_block
              nullable: true
              description: >-
                Optional retrieval tool allowlist used with the granular strategy. May be an array or a comma-separated string. Omit it (or send an empty list) to use ALL methods. Names are case-insensitive and common short forms are accepted (`keyword`, `semantic`, `neo4j`, `spatial`, `opengeodata`, `web`, `webfetch`); an UNRECOGNIZED name is rejected with HTTP 400 rather than silently disabling retrieval. When `neo4j_search` is enabled, the companion Neo4j id and related-node tools are also available.
              example: ["keyword_search", "semantic_search", "opengeodata_search"]
            enabled_search_methods:
              type: array
              items:
                type: string
              nullable: true
              description: Snake_case alias for `enabledSearchMethods`.
              example: ["keyword_search", "semantic_search"]
            usePersistentMemory:
              type: boolean
              default: true
              description: Whether to load and update persistent conversation memory.
              example: true
            use_persistent_memory:
              type: boolean
              description: Snake_case alias for `usePersistentMemory`.
              example: true
            smartToolRouting:
              type: boolean
              default: true
              description: Whether the orchestration layer should classify intent and restrict allowed tools before invoking the agent.
              example: true
            smart_tool_routing:
              type: boolean
              description: Snake_case alias for `smartToolRouting`.
              example: true
            forcedIntent:
              type: string
              nullable: true
              description: Optional override for the routing intent classifier. Use only for debugging or deterministic tests.
              example: search
            forced_intent:
              type: string
              nullable: true
              description: Snake_case alias for `forcedIntent`.
              example: search
            filePaths:
              type: array
              items:
                type: string
              nullable: true
              description: Local filesystem paths to expose to the agent through file tools. Prefer `fileIds` for files uploaded through `/agent/files/upload`.
              example: ["./data/Crimes_-_2026_20260406.csv"]
            file_paths:
              type: array
              items:
                type: string
              nullable: true
              description: Snake_case alias for `filePaths`.
              example: ["./data/Crimes_-_2026_20260406.csv"]
            fileIds:
              type: array
              items:
                type: string
              nullable: true
              description: Managed file ids returned by `/agent/files/upload` or by agent-generated output tools such as `write_output_file`.
              example: ["file_0123456789ab"]
            file_ids:
              type: array
              items:
                type: string
              nullable: true
              description: Snake_case alias for `fileIds`.
              example: ["file_0123456789ab"]
            skillPaths:
              type: array
              items:
                type: string
              nullable: true
              description: Directories searched for agent skills. Also accepted as `skillRoots` or `skill_roots`.
              example: ["./skills"]
            skill_paths:
              type: array
              items:
                type: string
              nullable: true
              description: Snake_case alias for `skillPaths`.
              example: ["./skills"]
            skillRoots:
              type: array
              items:
                type: string
              nullable: true
              description: Alternate camelCase name for skill search roots.
              example: ["./skills"]
            verbose:
              type: boolean
              default: false
              description: Enables verbose LangChain/agent execution logging.
              example: false
            codeExec:
              type: boolean
              nullable: true
              description: Enable the sandboxed `execute_code` tool. Omit to use the server default from AGENT_CODE_EXEC (ON); send false to disable for this request.
              example: true
            code_exec:
              type: boolean
              nullable: true
              description: Snake_case alias for `codeExec`.
          example:
            userQuery: Inspect the uploaded Chicago crime CSV and summarize trends by primary type.
            memoryId: agent-memory-chicago-001
            threadId: agent-thread-chicago-001
            conversationName: Chicago crime analysis
            recentK: 6
            toolStrategy: granular
            enabledSearchMethods: ["keyword_search", "semantic_search", "opengeodata_search"]
            usePersistentMemory: true
            smartToolRouting: true
            forcedIntent: search
            fileIds: ["file_0123456789ab"]
            skillPaths: ["./skills"]
            verbose: false
    responses:
      200:
        description: Agent chat response payload.
        schema:
          type: object
          properties:
            answer:
              type: string
              description: Final answer extracted from the agent result.
              example: The uploaded CSV contains reported Chicago crime incidents. Theft, battery, and criminal damage are the most frequent primary types in this sample.
            message_id:
              type: string
              description: UUID for the persisted chat turn.
              example: 9dc05b2c-4d1b-46f8-b640-ef1f490f0b62
            elements:
              type: array
              description: Legacy UI evidence elements. Agent chat currently returns an empty list unless upstream payloads provide elements.
              items:
                type: object
              example: []
            count:
              type: integer
              description: Count of returned `elements`.
              example: 0
            retrievalSteps:
              type: array
              description: Legacy retrieval-step trace. Agent chat currently returns an empty list unless upstream payloads provide retrieval steps.
              items:
                type: object
              example: []
            reactHistory:
              type: array
              description: Legacy ReAct history. Agent chat currently returns an empty list unless upstream payloads provide ReAct history.
              items:
                type: object
              example: []
            memoryId:
              type: string
              nullable: true
              description: Effective persistent memory id used for the turn.
              example: agent-memory-chicago-001
            threadId:
              type: string
              nullable: true
              description: Effective agent execution thread id.
              example: agent-thread-chicago-001
            routeTrace:
              type: object
              description: Summary of the route and tool calls selected by the orchestration agent.
              properties:
                query:
                  type: string
                  example: Inspect the uploaded Chicago crime CSV and summarize trends by primary type.
                route:
                  type: string
                  example: search_then_analysis
                available_agents:
                  type: array
                  items:
                    type: string
                  example: ["answer_from_memory", "search_agent_evidence", "analysis_agent_answer"]
                called_tools:
                  type: array
                  items:
                    type: string
                  example: ["search_agent_evidence", "analysis_agent_answer"]
                analysis_called_tools:
                  type: array
                  items:
                    type: string
                  example: ["read_uploaded_file"]
                selected_skills:
                  type: array
                  items:
                    type: string
                  example: ["chicago-crime-analysis"]
                chat_history_available:
                  type: boolean
                  example: true
            artifacts:
              type: object
              description: Reserved object for generated artifacts when supplied by upstream agent payloads.
              example: {}
            opengeodata_results:
              type: array
              description: >-
                Structured OpenGeoData hits (from `opengeodata_search`) surfaced alongside the
                markdown answer, projected from the run's evidence. Empty when the run used no
                external OpenGeoData. Each item carries title, url (landing page), source/provider,
                and — when available — bbox, datetime, license, links.
              items:
                type: object
                properties:
                  doc_id: { type: string, example: cmr-1 }
                  title: { type: string, example: US Army Corps National Inventory of Dams }
                  url: { type: string, example: "https://doi.org/10.5066/F7833R62" }
                  source: { type: string, example: opengeodata }
                  provider: { type: string, example: Data.gov }
                  bbox:
                    type: array
                    items: { type: number }
                    example: [-91.5, 37.0, -87.5, 42.5]
              example: []
            agent_result:
              type: object
              description: JSON-safe raw agent execution result, including orchestration result, route trace, final answer, thread id, and available skills.
              properties:
                final_answer:
                  type: string
                  example: The uploaded CSV contains reported Chicago crime incidents.
                thread_id:
                  type: string
                  example: agent-thread-chicago-001
                route_trace:
                  type: object
                available_skills:
                  type: array
                  items:
                    type: object
                orchestration_result:
                  type: object
            fileIds:
              type: array
              description: Normalized managed file ids made available to the agent.
              items:
                type: string
              example: ["file_0123456789ab"]
            filePaths:
              type: array
              description: Normalized local file paths made available to the agent.
              items:
                type: string
              example: []
            skillPaths:
              type: array
              description: Normalized skill root directories used for skill discovery.
              items:
                type: string
              example: ["skills"]
            availableSkills:
              type: array
              description: Skills discovered from `skillPaths`.
              items:
                type: object
                properties:
                  name:
                    type: string
                    example: chicago-crime-analysis
                  description:
                    type: string
                    example: Analyze Chicago crime datasets.
                  path:
                    type: string
                    example: skills/chicago-crime-analysis
                  allowed_tools:
                    type: array
                    items:
                      type: string
                    example: ["keyword_search", "semantic_search"]
                  tags:
                    type: array
                    items:
                      type: string
                    example: ["crime", "chicago"]
            warning:
              type: string
              description: Optional warning, commonly persistent-memory load or update failures.
              example: "persistent_memory_unavailable: connection refused"
        examples:
          application/json:
            answer: The uploaded CSV contains reported Chicago crime incidents. Theft, battery, and criminal damage are the most frequent primary types in this sample.
            message_id: 9dc05b2c-4d1b-46f8-b640-ef1f490f0b62
            elements: []
            count: 0
            retrievalSteps: []
            reactHistory: []
            memoryId: agent-memory-chicago-001
            threadId: agent-thread-chicago-001
            routeTrace:
              query: Inspect the uploaded Chicago crime CSV and summarize trends by primary type.
              route: search_then_analysis
              available_agents: ["answer_from_memory", "search_agent_evidence", "analysis_agent_answer"]
              called_tools: ["search_agent_evidence", "analysis_agent_answer"]
              analysis_called_tools: ["read_uploaded_file"]
              selected_skills: ["chicago-crime-analysis"]
              chat_history_available: true
            artifacts: {}
            agent_result:
              final_answer: The uploaded CSV contains reported Chicago crime incidents.
              thread_id: agent-thread-chicago-001
              route_trace:
                route: search_then_analysis
              available_skills:
                - name: chicago-crime-analysis
                  description: Analyze Chicago crime datasets.
                  path: skills/chicago-crime-analysis
                  allowed_tools: ["keyword_search", "semantic_search"]
                  tags: ["crime", "chicago"]
              orchestration_result:
                messages: []
            fileIds: ["file_0123456789ab"]
            filePaths: []
            skillPaths: ["skills"]
            availableSkills:
              - name: chicago-crime-analysis
                description: Analyze Chicago crime datasets.
                path: skills/chicago-crime-analysis
                allowed_tools: ["keyword_search", "semantic_search"]
                tags: ["crime", "chicago"]
      400:
        description: Validation error (e.g., missing userQuery).
        schema:
          type: object
          properties:
            error:
              type: string
              example: Missing userQuery in request body.
      403:
        description: Forbidden - invalid or missing API key when AGENT_CHAT_API_KEY is configured.
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Forbidden: invalid API key."
      500:
        description: Internal server error.
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Internal server error: agent execution failed"
            diagnostics:
              type: object
              description: Optional structured agent diagnostics.
            diagnosticText:
              type: string
              description: Optional readable diagnostic trace.
    """
    try:
        # Identity first: it can satisfy the credential check on its own, and running the key
        # gate ahead of it refuses a signed-in visitor before anyone asks who they are.
        try:
            _request_user = _require_user()
        except identity.IdentityError as exc:
            return _identity_error_response(exc)

        try:
            _require_agent_chat_api_key(_request_user)
        except PermissionError as exc:
            return jsonify({"error": str(exc)}), 403
        except RuntimeError as exc:
            logger.error("Agent chat API key misconfigured: %s", exc)
            return jsonify({"error": "Server misconfiguration: API key not set"}), 500

        data = request.get_json() or {}
        normalized = _normalize_agent_chat_request(data)
        user_query = normalized["user_query"]
        if not user_query:
            return jsonify({"error": "Missing userQuery in request body."}), 400

        logger.info(f"Processing agent chat: {user_query[:100]}...")
        # Bind the conversation for the whole turn, so every file a tool writes records who
        # wrote it and every lookup sees this conversation's files first. Set at the EDGE
        # because nothing deeper knows what a session is. JWT will change where this id comes
        # from, not what it does with it.
        _session_token = set_file_store_session(normalized.get("thread_id"))
        _user_token = identity.set_user(_request_user)
        try:
            # Checked INSIDE the identity binding: assert_owner reads the caller from the same
            # ContextVar everything else does, so before this line it would see nobody and wave
            # every conversation through.
            if normalized.get("memory_id"):
                try:
                    assert_memory_owner(normalized["memory_id"])
                except MemoryAccessDenied:
                    # 404 for the same reason the download endpoint uses it: a 403 confirms the
                    # conversation exists, and a memory_id is the only thing guarding it.
                    logger.info("Chat refused: memory %s is not this caller's",
                                normalized["memory_id"])
                    return jsonify({"error": "No conversation found for that id.",
                                    "reason": "not_your_conversation"}), 404
            raw = run_agent_chat(
                user_input=user_query,
                thread_id=normalized.get("thread_id"),
                memory_id=normalized.get("memory_id"),
                conversation_name=normalized.get("conversation_name"),
                recent_k=normalized.get("recent_k"),
                tool_strategy=normalized.get("tool_strategy", "granular"),
                include_mcp_tools=bool(normalized.get("include_mcp_tools", False)),
                mcp_modules=_parse_mcp_modules(normalized.get("mcp_modules")),
                enabled_search_methods=_parse_enabled_search_methods(normalized.get("enabled_search_methods")),
                use_persistent_memory=bool(normalized.get("use_persistent_memory", True)),
                smart_tool_routing=bool(normalized.get("smart_tool_routing", True)),
                forced_intent=normalized.get("forced_intent"),
                file_paths=normalized.get("file_paths"),
                file_ids=normalized.get("file_ids"),
                skill_roots=normalized.get("skill_roots"),
                verbose=bool(normalized.get("verbose", False)),
                code_exec=normalized.get("code_exec"),
                code_peer=normalized.get("code_peer"),
                unified_peer=normalized.get("unified_peer"),
                code_peer_model=normalized.get("code_peer_model"),
                llm_provider=normalized.get("llm_provider"),
                llm_model=normalized.get("llm_model"),
                reasoning_effort=normalized.get("reasoning_effort"),
            )
            return jsonify(_format_agent_chat_result(raw)), 200
        finally:
            reset_file_store_session(_session_token)
            identity.reset_user(_user_token)
    except ValueError as e:
        logger.error(f"Agent chat validation error: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error processing agent chat: {str(e)}", exc_info=True)
        payload = {"error": f"Internal server error: {str(e)}"}
        diagnostics = getattr(e, "diagnostics", None)
        if diagnostics:
            payload["diagnostics"] = diagnostics
            diagnostic_text = diagnostics.get("readable_trace") if isinstance(diagnostics, dict) else None
            if diagnostic_text:
                payload["diagnosticText"] = diagnostic_text
        return jsonify(payload), 500


@app.route('/agent/chat/stream', methods=['POST'])
def agent_chat_stream():
    """
    Stream agent chat events using Server-Sent Events (SSE). **This is the primary agent chat
    endpoint**; `POST /agent/chat` is the non-streaming variant that returns the same final
    payload as a single JSON response.

    The response is `text/event-stream` with blocks of the form:
    `event: <name>\\ndata: <json>\\n\\n`

    Node-compatible events: `status`, `result`, `error`.
    Additional categorized events: `routing`, `search`, `analysis`, `agent_trace`, `answer`, `file`.
    `agent_trace` includes live LLM messages, LLM tool decisions, and MCP call lifecycle events.

    Accepts both camelCase (frontend) and snake_case field names; when both are present the
    camelCase value wins. **Only `userQuery` is required** — every other field has a default.

    Absolute minimum: `{ "userQuery": "Explain the National Inventory of Dams dataset" }`

    Recommended — supply a client-chosen `memoryId` so you control the conversation id from the
    first turn: a fresh id starts a new conversation, and reusing the same id on later turns
    continues it so follow-ups retain context.

    - New conversation (client picks the id):
      `{ "userQuery": "Explain the National Inventory of Dams dataset", "memoryId": "agent-mem-1" }`
    - Follow-up (same id continues it):
      `{ "userQuery": "What are its related elements?", "memoryId": "agent-mem-1" }`

    Omitting `memoryId` instead mints a NEW memory server-side and returns its id (in the terminal
    `result` event), which you then reuse on later turns.

    **Defaults when a field is omitted**

    - `userQuery`: required (400 if missing).
    - `memoryId` / `threadId`: none. With persistent memory on and no `memoryId`, a NEW memory
      record is created and its id is returned (in the terminal `result` event), and a `threadId`
      is auto-generated. Reuse the returned `memoryId`/`threadId` on later turns for continuity —
      otherwise each call starts a new, history-less conversation.
    - `conversationName`: `"agent-chat"` (name given to an auto-created memory).
    - `recentK`: null = include ALL recorded turns; `0` = ignore history this turn.
    - `usePersistentMemory`: `true` (read/write OpenSearch-backed memory).
    - `toolStrategy`: `"granular"`.
    - `enabledSearchMethods`: null = ALL granular retrieval tools — `keyword_search`,
      `semantic_search`, `neo4j_search` (+ its `neo4j_get_element_by_id` /
      `neo4j_explore_related_nodes` companions), `spatial_search`, `opengeodata_search`,
      `web_search`, `agent_kb_search`, `get_kb_block`. Supplying a list restricts to those names
      (the neo4j companions are kept whenever `neo4j_search` is listed;
      `agent_kb_search`/`get_kb_block` are dropped unless explicitly named).
    - `web_search` queries the LIVE open web and returns metadata only (title, url, snippet) — no
      page content. It is capped per turn (`AGENT_WEB_MAX_SEARCHES_PER_TURN`, default 3), never
      runs as part of the deterministic multi-method sweep, and can be turned off for the whole
      deployment with `AGENT_WEB_ENABLED=false`. Only URLs a search actually returned may appear as
      download links in the answer; others are stripped.
    - `web_fetch` is the second step: it READS one page found by `web_search` and returns only the
      passages bearing on the question (boilerplate stripped, `AGENT_WEB_FETCH_MAX_CHARS`, default
      6000). Capped separately (`AGENT_WEB_MAX_FETCHES_PER_TURN`, default 3) and enabled
      automatically whenever `web_search` is — asking for `web_search` alone would otherwise let
      the agent find pages it cannot read. Only http/https on standard web ports
      (`AGENT_WEB_ALLOWED_PORTS`, default `80,443`) can be fetched; loopback, private addresses,
      this deployment's own service hosts, and every redirect hop are checked, so it cannot be
      steered into the internal network. Page text is returned to the model as untrusted evidence.
    - `includeMcpTools`: server default `AGENT_INCLUDE_MCP_TOOLS` (ON); send `false` to disable.
    - `mcpModules`: null = all MCP modules (when MCP tools are on).
    - `smartToolRouting`: `true`.
    - `codeExec`: server default `AGENT_CODE_EXEC` (ON = sandboxed `execute_code` tool); send
      `false` to disable.
    - `agentDev`: server default `AGENT_DEV` (off = status-only SSE). When true, also emit the SSE
      detail tier — tool args/results and LLM interactions. (Streaming-only; ignored by
      `/agent/chat`.)
    - `forcedIntent`: none (automatic intent classification).
    - `fileIds` / `filePaths` / `skillPaths`: none.
    - `verbose`: `false`.

    **Uploading files**

    1. POST the file(s) to `/agent/files/upload` (multipart/form-data, form field `file` or
       `files`); the response returns a `file_id` for each file.
    2. Send those ids here as `fileIds`, with your `userQuery` and (for continuity) a stable
       `threadId`/`memoryId`. The agent stages each file into its code/geo tools — inside
       `execute_code` the file is reachable via the `input_files` argument and appears in the
       working directory under both its `file_id` and its original filename. An extracted
       shapefile (.shp/.shx/.dbf uploaded separately) can be referenced by ANY one component's
       `file_id`; the siblings are auto-discovered. A file stays attached to the session
       (keyed by `threadId`) on later turns, so a follow-up like "now plot it" can omit `fileIds`.

       Example: `{ "userQuery": "Inspect this CSV and summarize by primary type",
       "fileIds": ["file_0123456789ab"], "threadId": "agent-thread-1", "memoryId": "agent-mem-1" }`

    **Downloadable files (how clients render download links)**

    Every stored file — uploads and agent-generated artifacts (plots, exports, executed source) —
    is described by a file record `{ "file_id", "filename", "download_url", "kind" }` whose
    `download_url` is a HOST-RELATIVE path: `/agent/files/<file_id>/download`. Clients must
    resolve it against the API origin they call (e.g. `new URL(download_url, apiOrigin)`); the
    download endpoint is a plain unauthenticated GET. Alternatively, set the server env
    `AGENT_PUBLIC_BASE_URL` (e.g. `http://149.165.147.219:3500`) and every emitted
    `download_url` — including the image URLs embedded in the answer markdown — is already an
    absolute URL, so clients need no resolution step. File records can appear at several places
    in the stream — `file` events, tool results inside `search`/`analysis` detail payloads, and
    the terminal `result` — so a robust client collects them from any event (deduping by
    `file_id`) rather than watching a single event type. Image artifacts (PNG/JPG maps, plots)
    may ALSO be embedded inline in the answer markdown as `![caption](download_url)`; when
    rendering a separate attachments list, skip records whose resolved URL (or `/<file_id>/`
    path segment) already appears in an inline image to avoid showing them twice. The reference
    implementation is `examples/iguide_chat_prototype.html` (`collectDownloads` / `absoluteUrl` /
    `renderFiles`).

    ---
    tags:
      - Agent Chat
    consumes:
      - application/json
    produces:
      - text/event-stream
    parameters:
      - in: header
        name: X-API-KEY
        type: string
        required: false
        description: API key for agent chat endpoints when env var AGENT_CHAT_API_KEY is configured. The Authorization header with a Bearer token is also accepted.
      - in: body
        name: body
        required: true
        description: Agent chat stream request. CamelCase frontend fields and snake_case backend aliases are both accepted; when both are present, the camelCase value is used.
        schema:
          type: object
          required:
            - userQuery
          properties:
            userQuery:
              type: string
              description: Natural-language user prompt. Required unless the snake_case alias `user_input` is supplied.
              example: Inspect the attached CSV and summarize the main columns.
            user_input:
              type: string
              description: Snake_case alias for `userQuery`.
              example: Inspect the attached CSV and summarize the main columns.
            memoryId:
              type: string
              nullable: true
              description: Persistent memory id. If omitted and `usePersistentMemory` is true, a new memory record is created. Also used as the thread id when `threadId` is omitted.
              example: demo-session-1
            memory_id:
              type: string
              nullable: true
              description: Snake_case alias for `memoryId`.
              example: demo-session-1
            threadId:
              type: string
              nullable: true
              description: LangGraph/checkpointer thread id for streamed agent execution. If omitted, `memoryId` is used.
              example: demo-thread-1
            thread_id:
              type: string
              nullable: true
              description: Snake_case alias for `threadId`.
              example: demo-thread-1
            conversationName:
              type: string
              nullable: true
              description: Friendly name used when the endpoint creates a new persistent memory.
              example: Streaming CSV analysis
            conversation_name:
              type: string
              nullable: true
              description: Snake_case alias for `conversationName`.
              example: Streaming CSV analysis
            recentK:
              type: integer
              nullable: true
              description: Number of recent memory turns to include in chat history. Use 0 to ignore history for this turn.
              default: null
              example: 8
            recent_k:
              type: integer
              nullable: true
              description: Snake_case alias for `recentK`.
              example: 8
            toolStrategy:
              type: string
              enum:
                - granular
                - full_pipeline
              default: granular
              description: Agent tool mode. `granular` exposes individual keyword, semantic, Neo4j, spatial, OpenGeoData, QGIS, file, and skill tools. `full_pipeline` exposes the compatibility RAG pipeline tool.
              example: granular
            tool_strategy:
              type: string
              enum:
                - granular
                - full_pipeline
              description: Snake_case alias for `toolStrategy`.
              example: granular
            includeMcpTools:
              type: boolean
              nullable: true
              description: Include MCP-backed tools (spatial/data analysis) in the agent toolset. Omit to use the server default from AGENT_INCLUDE_MCP_TOOLS (ON); send false to disable for this request.
              example: true
            include_mcp_tools:
              type: boolean
              description: Snake_case alias for `includeMcpTools`.
              example: false
            mcpModules:
              type: array
              items:
                type: string
              nullable: true
              description: MCP module allowlist. May be an array of module names or a comma-separated string.
              example: ["search_tools", "data_tools"]
            mcp_modules:
              type: array
              items:
                type: string
              nullable: true
              description: Snake_case alias for `mcpModules`.
              example: ["search_tools", "data_tools"]
            enabledSearchMethods:
              type: array
              items:
                type: string
                enum:
                  - keyword_search
                  - semantic_search
                  - neo4j_search
                  - neo4j_get_element_by_id
                  - neo4j_explore_related_nodes
                  - spatial_search
                  - opengeodata_search
                  - web_search
                  - web_fetch
                  - agent_kb_search
                  - get_kb_block
              nullable: true
              description: >-
                Optional retrieval tool allowlist used with the granular strategy. May be an array or a comma-separated string. Omit it (or send an empty list) to use ALL methods. Names are case-insensitive and common short forms are accepted (`keyword`, `semantic`, `neo4j`, `spatial`, `opengeodata`, `web`, `webfetch`); an UNRECOGNIZED name is rejected with HTTP 400 rather than silently disabling retrieval. When `neo4j_search` is enabled, the companion Neo4j id and related-node tools are also available.
              example: ["keyword_search", "semantic_search", "opengeodata_search"]
            enabled_search_methods:
              type: array
              items:
                type: string
              nullable: true
              description: Snake_case alias for `enabledSearchMethods`.
              example: ["keyword_search", "semantic_search"]
            usePersistentMemory:
              type: boolean
              default: true
              description: Whether to load and update persistent conversation memory.
              example: true
            use_persistent_memory:
              type: boolean
              description: Snake_case alias for `usePersistentMemory`.
              example: true
            smartToolRouting:
              type: boolean
              default: true
              description: Whether the orchestration layer should classify intent and restrict allowed tools before invoking the agent.
              example: true
            smart_tool_routing:
              type: boolean
              description: Snake_case alias for `smartToolRouting`.
              example: true
            forcedIntent:
              type: string
              nullable: true
              description: Optional override for the routing intent classifier. Use only for debugging or deterministic tests.
              example: search
            forced_intent:
              type: string
              nullable: true
              description: Snake_case alias for `forcedIntent`.
              example: search
            filePaths:
              type: array
              items:
                type: string
              nullable: true
              description: Local filesystem paths to expose to the agent through file tools. Prefer `fileIds` for files uploaded through `/agent/files/upload`.
              example: ["./data/Crimes_-_2026_20260406.csv"]
            file_paths:
              type: array
              items:
                type: string
              nullable: true
              description: Snake_case alias for `filePaths`.
              example: ["./data/Crimes_-_2026_20260406.csv"]
            fileIds:
              type: array
              items:
                type: string
              nullable: true
              description: Managed file ids returned by `/agent/files/upload` or by agent-generated output tools such as `write_output_file`.
              example: ["file_0123456789ab"]
            file_ids:
              type: array
              items:
                type: string
              nullable: true
              description: Snake_case alias for `fileIds`.
              example: ["file_0123456789ab"]
            skillPaths:
              type: array
              items:
                type: string
              nullable: true
              description: Directories searched for agent skills. Also accepted as `skillRoots` or `skill_roots`.
              example: ["./skills"]
            skill_paths:
              type: array
              items:
                type: string
              nullable: true
              description: Snake_case alias for `skillPaths`.
              example: ["./skills"]
            skillRoots:
              type: array
              items:
                type: string
              nullable: true
              description: Alternate camelCase name for skill search roots.
              example: ["./skills"]
            verbose:
              type: boolean
              default: false
              description: Enables verbose LangChain/agent execution logging.
              example: false
            codeExec:
              type: boolean
              nullable: true
              description: Enable the sandboxed `execute_code` tool. Omit to use the server default from AGENT_CODE_EXEC (ON); send false to disable for this request.
              example: true
            code_exec:
              type: boolean
              nullable: true
              description: Snake_case alias for `codeExec`.
            agentDev:
              type: boolean
              nullable: true
              description: Streaming-only. When true, emit the SSE detail tier (tool args/results, LLM interactions); omit to use the server default from AGENT_DEV (off = status-only events).
              example: false
            agent_dev:
              type: boolean
              nullable: true
              description: Snake_case alias for `agentDev`.
          example:
            userQuery: Inspect the attached CSV and summarize the main columns.
            memoryId: demo-session-1
            threadId: demo-thread-1
            conversationName: Streaming CSV analysis
            recentK: 8
            toolStrategy: granular
            enabledSearchMethods: ["keyword_search", "semantic_search", "opengeodata_search"]
            usePersistentMemory: true
            smartToolRouting: true
            forcedIntent: search
            fileIds: ["file_0123456789ab"]
            skillPaths: ["./skills"]
            verbose: false
            agentDev: false
    responses:
      200:
        description: |
          Server-Sent Events stream. Each event block is formatted as `event: <name>\\ndata: <json>\\n\\n`.

          Node-compatible events:
          - `status`: progress message with a `status` string.
          - `result`: final payload with the same JSON shape as `/agent/chat`.
          - `error`: streamed error payload. Missing `userQuery` is reported this way after the stream is opened.

          Additional UI events:
          - `node`: graph node lifecycle (`node_started` / `node_completed`) with a
            human-readable `message`. This carries MOST of the progress text a user reads —
            triage, the supervisor's next step, each peer starting and finishing. A client
            that has no branch for it drops that text into whatever its default branch does,
            which is how one client rendered nine rows of routing bookkeeping per turn.
          - `routing`: route initialization, intent/policy state, and final route trace.
          - `search`: search-agent starts, completions, tool calls, tool results, and tool errors.
          - `analysis`: analysis/code-agent starts, completions, tool calls, tool results, and tool errors.
          - `agent_trace`: normalized live LLM messages, LLM tool decisions, MCP call lifecycle events, and diagnostic trace events.
          - `answer`: intermediate completed/final-answer payloads.
          - `file`: generated artifact notifications.
        schema:
          type: string
          example: |
            event: status
            data: {"status":"Agent chat started"}

            event: routing
            data: {"type":"initialized","detail":{"stage":"initialized","thread_id":"demo-thread-1","tool_strategy":"granular","available_agents":["answer_from_memory","search_agent_evidence","analysis_agent_answer"],"available_skills":[]}}

            event: node
            data: {"type":"node_started","stage":"triage","node":"triage","agent":"","message":"Routing the request","detail":{"stage":"triage","message":"Routing the request"}}

            event: node
            data: {"type":"node_completed","stage":"supervisor","node":"supervisor","agent":"","message":"Next: analyze","detail":{"stage":"supervisor","route":"analyze","message":"Next: analyze"}}

            event: agent_trace
            data: {"type":"route_decision","agent":"orchestrator_agent","label":"Route decision","message":"route=search_then_analysis; intent=search","detail":{"kind":"agent_route_decision","route":"search_then_analysis"}}

            event: answer
            data: {"type":"result","answer":"The CSV includes ID, date, primary type, description, location, latitude, and longitude columns.","detail":{"answer":"The CSV includes ID, date, primary type, description, location, latitude, and longitude columns."}}

            event: result
            data: {"answer":"The CSV includes ID, date, primary type, description, location, latitude, and longitude columns.","message_id":"9dc05b2c-4d1b-46f8-b640-ef1f490f0b62","elements":[],"count":0,"retrievalSteps":[],"reactHistory":[],"memoryId":"demo-session-1","threadId":"demo-thread-1","routeTrace":{"route":"search_then_analysis","called_tools":["search_agent_evidence","analysis_agent_answer"]},"artifacts":{},"agent_result":{"final_answer":"The CSV includes ID, date, primary type, description, location, latitude, and longitude columns."},"fileIds":["file_0123456789ab"],"filePaths":[],"skillPaths":["skills"],"availableSkills":[]}
        examples:
          text/event-stream: |
            event: status
            data: {"status":"Agent chat started"}

            event: routing
            data: {"type":"route_trace","detail":{"query":"Inspect the attached CSV and summarize the main columns.","route":"search_then_analysis","available_agents":["answer_from_memory","search_agent_evidence","analysis_agent_answer"],"called_tools":["search_agent_evidence","analysis_agent_answer"],"analysis_called_tools":["read_uploaded_file"],"selected_skills":[],"chat_history_available":true}}

            event: search
            data: {"type":"tool_call","agent":"search_agent","node":null,"detail":{"name":"read_uploaded_file","args":{"file_id":"file_0123456789ab"}}}

            event: agent_trace
            data: {"type":"tool_call","agent":"search_agent","label":"LLM tool decision","message":"read_uploaded_file({\"file_id\": \"file_0123456789ab\"})","tool_calls":[{"name":"read_uploaded_file","args":{"file_id":"file_0123456789ab"}}],"detail":{"kind":"llm_tool_decision","agent":"search_agent","name":"read_uploaded_file","args":{"file_id":"file_0123456789ab"}}}

            event: result
            data: {"answer":"The CSV includes ID, date, primary type, description, location, latitude, and longitude columns.","message_id":"9dc05b2c-4d1b-46f8-b640-ef1f490f0b62","elements":[],"count":0,"retrievalSteps":[],"reactHistory":[],"memoryId":"demo-session-1","threadId":"demo-thread-1","routeTrace":{"query":"Inspect the attached CSV and summarize the main columns.","route":"search_then_analysis","available_agents":["answer_from_memory","search_agent_evidence","analysis_agent_answer"],"called_tools":["search_agent_evidence","analysis_agent_answer"],"analysis_called_tools":["read_uploaded_file"],"selected_skills":[],"chat_history_available":true},"artifacts":{},"agent_result":{"final_answer":"The CSV includes ID, date, primary type, description, location, latitude, and longitude columns.","thread_id":"demo-thread-1","route_trace":{"route":"search_then_analysis"},"available_skills":[],"orchestration_result":{"messages":[]}},"fileIds":["file_0123456789ab"],"filePaths":[],"skillPaths":["skills"],"availableSkills":[]}
      400:
        description: Pre-stream validation error. Most body validation failures are returned as an SSE `error` event with HTTP 200 after the stream is opened.
        schema:
          type: object
          properties:
            error:
              type: string
              example: enabled_search_methods must be a list of strings or a comma-separated string
      403:
        description: Forbidden - invalid or missing API key when AGENT_CHAT_API_KEY is configured.
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Forbidden: invalid API key."
      500:
        description: Internal server error.
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Internal server error: agent execution failed"
    """
    try:
        # Identity first: it can satisfy the credential check on its own, and running the key
        # gate ahead of it refuses a signed-in visitor before anyone asks who they are.
        try:
            _request_user = _require_user()
        except identity.IdentityError as exc:
            return _identity_error_response(exc)

        try:
            _require_agent_chat_api_key(_request_user)
        except PermissionError as exc:
            return jsonify({"error": str(exc)}), 403
        except RuntimeError as exc:
            logger.error("Agent chat stream API key misconfigured: %s", exc)
            return jsonify({"error": "Server misconfiguration: API key not set"}), 500

        data = request.get_json() or {}
        normalized = _normalize_agent_chat_request(data)
        user_query = normalized["user_query"]
        if not user_query:
            def _single_error():
                yield _sse_event("error", {"error": "Missing userQuery in request body."})
            return Response(
                stream_with_context(_single_error()),
                mimetype="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
            )

        logger.info(f"Streaming agent chat: {user_query[:100]}...")

        @stream_with_context
        def generate():
            error_emitted = False
            stream_session_token = set_file_store_session(normalized.get("thread_id"))
            stream_user_token = identity.set_user(_request_user)
            try:
                if normalized.get("memory_id"):
                    try:
                        assert_memory_owner(normalized["memory_id"])
                    except MemoryAccessDenied:
                        logger.info("Stream refused: memory %s is not this caller's",
                                    normalized["memory_id"])
                        # Headers are long gone by now, so the refusal travels as an SSE frame
                        # rather than a status code. Same wording as the JSON path.
                        yield _sse_event("error", {"error": "No conversation found for that id.",
                                                   "reason": "not_your_conversation"})
                        return
                for item in stream_agent_chat_events(
                    user_input=user_query,
                    thread_id=normalized.get("thread_id"),
                    memory_id=normalized.get("memory_id"),
                    conversation_name=normalized.get("conversation_name"),
                    recent_k=normalized.get("recent_k"),
                    tool_strategy=normalized.get("tool_strategy", "granular"),
                    include_mcp_tools=bool(normalized.get("include_mcp_tools", False)),
                    mcp_modules=_parse_mcp_modules(normalized.get("mcp_modules")),
                    enabled_search_methods=_parse_enabled_search_methods(normalized.get("enabled_search_methods")),
                    use_persistent_memory=bool(normalized.get("use_persistent_memory", True)),
                    smart_tool_routing=bool(normalized.get("smart_tool_routing", True)),
                    forced_intent=normalized.get("forced_intent"),
                    file_paths=normalized.get("file_paths"),
                    file_ids=normalized.get("file_ids"),
                    skill_roots=normalized.get("skill_roots"),
                    verbose=bool(normalized.get("verbose", False)),
                    agent_dev=normalized.get("agent_dev"),
                    code_exec=normalized.get("code_exec"),
                    code_peer=normalized.get("code_peer"),
                    unified_peer=normalized.get("unified_peer"),
                    code_peer_model=normalized.get("code_peer_model"),
                    llm_provider=normalized.get("llm_provider"),
                    llm_model=normalized.get("llm_model"),
                    reasoning_effort=normalized.get("reasoning_effort"),
                ):
                    event_name = str(item.get("event") or "message")
                    payload = item.get("data") or {}
                    agent_role = item.get("agent_role") or payload.get("role")
                    node_name = item.get("node")

                    # Heartbeat during quiet stretches (long LLM/sandbox runs): an SSE COMMENT
                    # line keeps bytes flowing so clients/proxies (e.g. Node fetch's 300s body
                    # timeout) don't kill the stream; SSE parsers ignore comment lines.
                    if event_name == "keepalive":
                        yield ": keepalive\n\n"
                        continue

                    # Graph node lifecycle (triage / fast_answer / orchestrate):
                    # surface distinctly so the UI can show pipeline progress.
                    if event_name in {"node_started", "node_completed"}:
                        yield _sse_event(
                            "node",
                            {
                                "type": event_name,
                                "stage": payload.get("stage") or node_name,
                                "node": node_name or payload.get("stage"),
                                "agent": str(agent_role or payload.get("agent") or ""),
                                "message": payload.get("message"),
                                "detail": payload,
                            },
                        )
                        continue

                    # Categorized events for richer frontend consumption
                    if event_name == "route_trace":
                        yield _sse_event("routing", {"type": "route_trace", "detail": payload})

                    if event_name == "status":
                        stage = payload.get("stage")
                        if stage in {"initialized", "intent_classified", "policy_resolved"}:
                            yield _sse_event("routing", {"type": stage, "detail": payload})

                    if event_name in {"subagent_started", "subagent_completed"}:
                        role = payload.get("role") or agent_role
                        category = _category_for_agent_role(role)
                        yield _sse_event(
                            category,
                            {"type": event_name, "agent": str(role or ""), "node": node_name, "detail": payload},
                        )

                    if event_name in {"tool_call", "tool_result", "tool_error"}:
                        category = _category_for_agent_role(agent_role)
                        display_agent_role = str(agent_role or payload.get("agent") or "")
                        yield _sse_event(
                            category,
                            {"type": event_name, "agent": display_agent_role, "node": node_name, "detail": payload},
                        )
                        trace_agent_role = display_agent_role
                        trace_kind = {
                            "tool_call": "llm_tool_decision",
                            "tool_result": "tool_result",
                            "tool_error": "tool_error",
                        }[event_name]
                        yield _sse_event(
                            "agent_trace",
                            _agent_trace_event(
                                {
                                    **payload,
                                    "kind": trace_kind,
                                    "agent": trace_agent_role,
                                }
                            ),
                        )

                    if event_name in {
                        "llm_interaction",
                        "llm_start",
                        "llm_error",
                        "decision",
                        "mcp_call_start",
                        "mcp_call_end",
                        "mcp_call_error",
                        # The error-and-repair story. Without these three the trace shows a tool
                        # being called twice and never says the first attempt failed, which is
                        # how a 1.6s rejection read as a duplicate tile sweep for two rounds of
                        # diagnosis. tool_dead_end is not new — the supervisor has emitted it
                        # whenever a tool failed twice and it re-ran the peer with an
                        # observation, and it was dropped HERE, so the one repair mechanism the
                        # agent already had has never been visible to anyone watching.
                        "tool_retry",
                        "tool_recovered",
                        "tool_dead_end",
                    }:
                        yield _sse_event("agent_trace", _agent_trace_event(payload))

                    if event_name == "artifact":
                        yield _sse_event("file", {"type": "artifact", "agent": str(agent_role or ""), "detail": payload})

                    if event_name == "map_layer":
                        # Untruncated plottable geometry from a geo tool (overpass_search, ...).
                        # Forwarded verbatim so a map client can add it as a live layer.
                        yield _sse_event("map_layer", payload)

                    if event_name == "completed":
                        yield _sse_event(
                            "answer",
                            {"type": "completed", "final_answer": payload.get("final_answer"), "detail": payload},
                        )

                    if event_name == "response":
                        yield _sse_event("answer", {"type": "result", "answer": payload.get("answer"), "detail": payload})

                    # Node-compatible events: status / result / error
                    if event_name == "status":
                        stage = payload.get("status") or payload.get("stage")
                        if stage:
                            yield _sse_event("status", {"status": _humanize_stage(stage)})
                        continue

                    if event_name == "memory_loaded":
                        yield _sse_event("status", {"status": "Augmenting question"})
                        continue

                    if event_name == "memory_saved":
                        yield _sse_event("status", {"status": "Updating memory"})
                        continue

                    if event_name == "warning":
                        yield _sse_event("status", {"status": str(payload.get("message") or "Warning")})
                        continue

                    if event_name in {"subagent_started", "subagent_completed"}:
                        role = payload.get("role") or "agent"
                        msg = f"{'Running' if event_name == 'subagent_started' else 'Completed'} {role}"
                        yield _sse_event("status", {"status": msg})
                        continue

                    if event_name == "verification_result":
                        yield _sse_event("status", {"status": "Verification"})
                        continue

                    if event_name == "response":
                        yield _sse_event("result", _format_agent_chat_result(payload))
                        continue

                    if event_name == "error":
                        message = payload.get("error") or payload.get("message") or "Unknown error"
                        diagnostic_text = payload.get("diagnosticText")
                        error_text = str(message)
                        if diagnostic_text:
                            error_text = f"{error_text}\n\n{diagnostic_text}"
                        error_payload = {"error": error_text}
                        if diagnostic_text:
                            error_payload["diagnosticText"] = diagnostic_text
                        if payload.get("diagnostics"):
                            error_payload["diagnostics"] = payload.get("diagnostics")
                            for trace_event in _diagnostic_agent_trace_events(payload.get("diagnostics")):
                                yield _sse_event("agent_trace", trace_event)
                        error_emitted = True
                        yield _sse_event("error", error_payload)
                        continue

            except ValueError as e:
                logger.error(f"Agent chat stream validation error: {str(e)}")
                yield _sse_event("error", {"error": str(e)})
            except Exception as e:
                logger.error(f"Error streaming agent chat: {str(e)}", exc_info=True)
                if error_emitted:
                    return
                error_payload = {"error": str(e)}
                diagnostics = getattr(e, "diagnostics", None)
                if diagnostics:
                    error_payload["diagnostics"] = diagnostics
                    diagnostic_text = diagnostics.get("readable_trace") if isinstance(diagnostics, dict) else None
                    if diagnostic_text:
                        error_payload["diagnosticText"] = diagnostic_text
                        error_payload["error"] = f"{error_payload['error']}\n\n{diagnostic_text}"
                    for trace_event in _diagnostic_agent_trace_events(diagnostics):
                        yield _sse_event("agent_trace", trace_event)
                yield _sse_event("error", error_payload)

            finally:
                # A stream that is cancelled mid-flight must not leave this thread bound to
                # the conversation; the next request on it would inherit the scope.
                reset_file_store_session(stream_session_token)
                identity.reset_user(stream_user_token)
        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )
    except ValueError as e:
        logger.error(f"Agent chat stream validation error: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error processing agent chat stream: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route('/query/batch', methods=['POST'])
def batch_query():
    """
    Batch query endpoint for processing multiple queries in one call.

    ---
    tags:
      - RAG
      - Query
    consumes:
      - application/json
    produces:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - queries
          properties:
            queries:
              type: array
              items:
                type: object
                properties:
                  user_input:
                    type: string
                    example: What datasets are available for Chicago crime?
                  memory_id:
                    type: string
                    nullable: true
                  params:
                    type: object
                    nullable: true
                  session_context:
                    type: object
                    nullable: true
    responses:
      200:
        description: Batch query results.
        schema:
          type: object
          properties:
            results:
              type: array
              items:
                type: object
                properties:
                  success:
                    type: boolean
                  answer:
                    type: object
                  evidence:
                    type: object
                  error:
                    type: string
      400:
        description: Validation error.
      500:
        description: Internal server error.
    """
    try:
        data = request.get_json() or {}
        queries = data.get('queries', [])

        if not queries or not isinstance(queries, list):
            return jsonify({"error": "queries must be a non-empty list"}), 400

        results = []
        for query_data in queries:
            try:
                result = run_pipeline(
                    user_input=query_data.get('user_input'),
                    memory_id=query_data.get('memory_id'),
                    params=_pipeline_params(query_data),
                    session_context=query_data.get('session_context', {})
                )
                results.append({
                    "success": True,
                    "answer": result.get("answer", {}),
                    "evidence": result.get("evidence", {})
                })
            except Exception as e:
                results.append({
                    "success": False,
                    "error": str(e)
                })

        return jsonify({"results": results}), 200

    except Exception as e:
        logger.error(f"Error processing batch query: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5002))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
