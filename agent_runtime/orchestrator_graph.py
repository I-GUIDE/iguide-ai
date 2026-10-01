"""Hybrid orchestrator graph.

An explicit LangGraph ``StateGraph`` that triages each query:

* **fast path** — clearly-trivial inputs (greetings / chit-chat) are answered by
  a single direct LLM call, skipping the heavyweight orchestrator (which would
  otherwise spin up search/analysis sub-agents and make several LLM calls);
* **orchestrate path** — everything else is delegated to the existing LLM
  orchestrator (agents-as-tools), so substantive queries keep their full,
  flexible behavior unchanged.

The graph is intentionally thin: the ``orchestrate`` node reuses
``collect_orchestration_tools`` / ``build_orchestrator_agent_executor`` /
``invoke_agent_with_payload_fallback`` verbatim.  Typed state gives a single home
for the eventual shared-evidence/ledger work.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import START, END, StateGraph

from agent_runtime.executor_factory import (
    DEFAULT_CHECKPOINTER,
    build_default_llm,
)
from agent_runtime.streaming_trace import emit_trace_event


# Informational list of the agents the orchestrate path can delegate to.
ORCHESTRATOR_AGENT_NAMES = [
    "answer_from_memory",
    "search_agent_evidence",
    "analysis_agent_answer",
]

FAST_ANSWER_PROMPT = (
    "You are the I-GUIDE assistant. I-GUIDE helps users discover geospatial "
    "datasets, publications, and notebooks, run spatial analyses, and generate "
    "code. Respond briefly and warmly to the user's greeting or simple message. "
    "If they ask what you can do, give a one- or two-sentence overview. Do not "
    "fabricate data or claim to have run any search."
)

# A query is fast-pathed only when the WHOLE message is a greeting / trivial
# phrase. Anything with substantive content falls through to the orchestrator.
_TRIVIAL_RE = re.compile(
    # Optional greeting prefix so "Hi who are you" / "hello, what can you do" also fast-path
    # (previously only a BARE trivial phrase matched, so a greeting + question fell through to
    # retrieval and came back as a no-evidence refusal).
    r"^\s*(?:(?:hi|hello|hey|hiya|yo|greetings|good\s+(?:morning|afternoon|evening|day))"
    r"[\s,!.]+)?("
    r"hi|hello|hey|hiya|yo|greetings|"
    r"good\s+(morning|afternoon|evening|day)|"
    r"thanks|thank\s+you|thx|ty|"
    r"ok|okay|cool|great|nice|"
    r"bye|goodbye|see\s+you|"
    r"who\s+are\s+you|what\s+are\s+you|who\s+made\s+you|who\s+built\s+you|"
    r"are\s+you\s+(?:an?\s+)?(?:ai|bot|human|llm)|what\s+(?:model|llm)\s+are\s+you|"
    r"what\s+is\s+(?:i-guide|iguide)|what'?s\s+i-guide|"
    r"what\s+can\s+you\s+do|what\s+do\s+you\s+do|"
    r"help"
    r")\s*[!.?]*\s*$",
    re.IGNORECASE,
)


def is_trivial_query(query: str) -> bool:
    """True when *query* is a greeting/trivial phrase safe to fast-path."""
    return bool(_TRIVIAL_RE.match(query or ""))


class OrchestratorState(TypedDict, total=False):
    """Shared state threaded through the hybrid orchestrator graph."""

    query: str
    chat_history: List[Any]
    thread_id: Optional[str]
    route: str  # "fast" | "orchestrate"
    final_answer: str
    orchestration_result: Any
    available_agent_names: List[str]
    audit: Dict[str, Any]  # grounding-audit verdict, lifted from the supervisor
    # Reserved for shared, deduplicated evidence across sub-agents (future work).
    evidence: List[Dict[str, Any]]


def _content_to_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
            else:
                parts.append(str(getattr(part, "text", part)))
        return "".join(parts).strip()
    return str(content or "").strip()


def build_orchestrator_graph(
    *,
    llm: Optional[Any] = None,
    verbose: bool = False,
    return_intermediate_steps: bool = True,
    tool_strategy: str = "granular",
    include_mcp_tools: bool = False,
    mcp_modules: Optional[List[str]] = None,
    enabled_search_methods: Optional[List[str]] = None,
    smart_tool_routing: bool = True,
    forced_intent: Optional[str] = None,
    thread_id: Optional[str] = None,
    checkpointer: Optional[Any] = DEFAULT_CHECKPOINTER,
    skill_roots: Optional[List[str]] = None,
    code_exec: Optional[bool] = None,
    code_peer: Optional[str] = None,
    code_peer_model: Optional[str] = None,
    unified_peer: Optional[bool] = None,
    input_file_ids: Optional[List[str]] = None,
) -> Any:
    """Compile the hybrid orchestrator graph for one request's configuration.

    """

    def triage_node(state: OrchestratorState) -> Dict[str, Any]:
        from agent_runtime.capabilities import is_capability_query

        query = state.get("query", "")
        if is_capability_query(query):
            # "what tools do you have" is about the ASSISTANT, not the knowledge base — no
            # peer in the supervisor pipeline sees the tool registry, so retrieval would
            # return irrelevant KB docs (or nothing). Answer from the live registries.
            route = "capabilities"
        else:
            route = "fast" if is_trivial_query(query) else "orchestrate"
        emit_trace_event(
            "node_started",
            {"stage": "triage", "message": "Routing the request"},
            node="triage",
        )
        # `fast` / `capabilities` / `orchestrate` are node names in THIS graph. "Routed to
        # orchestrate" told the reader which node was next, which is not a fact about their
        # request — and for the other two it also duplicated the destination's own first line.
        _ROUTE_NAMES = {
            "orchestrate": "Routed to the full agent",
            "fast": "Routed to a direct answer",
            "capabilities": "Routed to the capability summary",
        }
        emit_trace_event(
            "node_completed",
            {"stage": "triage", "route": route,
             "message": _ROUTE_NAMES.get(route, f"Routed to {route}")},
            node="triage",
        )
        return {"route": route}

    def capabilities_node(state: OrchestratorState) -> Dict[str, Any]:
        from agent_runtime.capabilities import describe_capabilities

        emit_trace_event(
            "node_started",
            {"stage": "capabilities", "message": "Describing available tools"},
            node="capabilities",
        )
        answer = describe_capabilities(
            llm=llm or build_default_llm(),
            # Without this the model never saw the question and answered the generic one.
            query=state.get("query") or "",
            enabled_search_methods=enabled_search_methods,
            include_mcp_tools=include_mcp_tools,
            mcp_modules=mcp_modules,
            code_exec=code_exec,
            skill_roots=skill_roots,
        )
        emit_trace_event(
            "node_completed",
            {"stage": "capabilities", "message": "Capability summary ready"},
            node="capabilities",
        )
        return {"final_answer": answer, "available_agent_names": []}

    def fast_answer_node(state: OrchestratorState) -> Dict[str, Any]:
        emit_trace_event(
            "node_started",
            {"stage": "fast_answer", "message": "Answering directly"},
            agent_role="direct_answer",
            node="fast_answer",
        )
        active_llm = llm or build_default_llm()
        response = active_llm.invoke(
            [
                {"role": "system", "content": FAST_ANSWER_PROMPT},
                {"role": "user", "content": state.get("query", "")},
            ]
        )
        answer = _content_to_text(response)
        emit_trace_event(
            "node_completed",
            {"stage": "fast_answer", "message": "Direct answer ready"},
            agent_role="direct_answer",
            node="fast_answer",
        )
        return {"final_answer": answer, "available_agent_names": []}

    def orchestrate_node(state: OrchestratorState) -> Dict[str, Any]:
        # Single fork between the two INDEPENDENT paths, resolved via the strategy registry.
        # Each path owns its entrypoint (agent_runtime.{legacy,supervisor}.orchestration) and
        # returns the same OrchestratorState key set, so the public contract is path-agnostic.
        from agent_runtime.strategy import OrchestrationConfig, get_orchestration_strategy

        cfg = OrchestrationConfig(
            llm=llm, verbose=verbose, return_intermediate_steps=return_intermediate_steps,
            tool_strategy=tool_strategy, include_mcp_tools=include_mcp_tools, mcp_modules=mcp_modules,
            enabled_search_methods=enabled_search_methods, smart_tool_routing=smart_tool_routing,
            forced_intent=forced_intent, thread_id=thread_id, checkpointer=checkpointer,
            skill_roots=skill_roots, code_exec=code_exec, input_file_ids=input_file_ids,
            code_peer=code_peer,
            code_peer_model=code_peer_model,
            unified_peer=unified_peer,
        )
        strategy = get_orchestration_strategy()
        return strategy(state.get("query", ""), state.get("chat_history") or None, cfg)

    builder = StateGraph(OrchestratorState)
    builder.add_node("triage", triage_node)
    builder.add_node("fast_answer", fast_answer_node)
    builder.add_node("capabilities", capabilities_node)
    builder.add_node("orchestrate", orchestrate_node)
    builder.add_edge(START, "triage")
    builder.add_conditional_edges(
        "triage",
        lambda state: state.get("route", "orchestrate"),
        {"fast": "fast_answer", "capabilities": "capabilities", "orchestrate": "orchestrate"},
    )
    builder.add_edge("fast_answer", END)
    builder.add_edge("capabilities", END)
    builder.add_edge("orchestrate", END)
    # No checkpointer on the outer graph: it is stateless per request, while the
    # inner orchestrator agent keeps its own checkpointed short-term memory.
    return builder.compile()
