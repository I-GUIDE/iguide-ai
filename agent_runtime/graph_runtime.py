"""Public API for running agent queries.

This is the main entry point for the agent runtime.  It exposes
``run_agent_query``, ``stream_agent_query_events``, and
which are called by the chat service and
the Flask API layer.
"""

from __future__ import annotations

import argparse
import contextvars
import json
import os
import queue
import threading
import time
from typing import Any, Dict, Generator, List, Optional

from agent_runtime.executor_factory import (
    AgentInvocationError,
    DEFAULT_CHECKPOINTER,
    agent_config,
    build_code_agent_executor,
    child_thread_id,
    invoke_agent_with_payload_fallback,
    resolve_thread_id,
)
from agent_runtime.runtime_utils import (
    build_orchestration_trace,
    extract_final_answer,
    extract_search_artifacts,
)
from agent_runtime.orchestrator_graph import (
    ORCHESTRATOR_AGENT_NAMES,
    build_orchestrator_graph,
)
from agent_runtime.skills import SkillRegistry
from agent_runtime.streaming_trace import is_agent_dev, trace_context
from rag_pipeline.search import web_utils


# ---------------------------------------------------------------------------
# Synchronous query execution
# ---------------------------------------------------------------------------

def _with_citation_urls(orchestration_result: Any) -> Any:
    """Give every evidence document the link a client should cite it at.

    The answer body already renders citations as links, because `default_synthesize_fn` runs the
    documents through ``_element_url``. The structured evidence the client shows under "Sources
    used" did not: internal knowledge elements reach it with ``url: ""`` — `_normalize_hits`
    fills that field only for external hits — so the same element was a link in the prose and
    plain text in the source list.

    Computed HERE rather than in the client on purpose. The scheme
    (``{FRONTEND_DOMAIN}/{type-plural}/{doc_id}``, url-first for anything that carries one) is
    already implemented and tested once; duplicating it in TypeScript is how it drifts, and a
    drifted copy is exactly what produced ``/osm_features/osm:node:…`` 404s in the prose.

    Non-destructive: only fills a MISSING url, never overwrites one, and leaves the doc alone
    when no link can be formed.
    """
    if not isinstance(orchestration_result, dict):
        return orchestration_result
    docs = orchestration_result.get("evidence")
    if not isinstance(docs, list):
        return orchestration_result
    try:
        from agent_runtime.supervisor.evidence_subgraph import _element_url
    except Exception:      # noqa: BLE001 - a citation link is never worth failing a turn over
        return orchestration_result
    for doc in docs:
        src = doc.get("document") if isinstance(doc, dict) and isinstance(doc.get("document"), dict) else doc
        if not isinstance(src, dict) or str(src.get("url") or "").strip():
            continue
        try:
            url = _element_url(src)
        except Exception:  # noqa: BLE001
            continue
        if url:
            src["url"] = url
    return orchestration_result


def run_agent_query(
    query: str,
    *,
    chat_history: Optional[List[Any]] = None,
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
) -> dict:
    """Run one query through the hybrid orchestrator graph."""
    effective_thread_id = resolve_thread_id(thread_id, checkpointer)
    graph = build_orchestrator_graph(
        llm=llm,
        verbose=verbose,
        return_intermediate_steps=return_intermediate_steps,
        tool_strategy=tool_strategy,
        include_mcp_tools=include_mcp_tools,
        mcp_modules=mcp_modules,
        enabled_search_methods=enabled_search_methods,
        smart_tool_routing=smart_tool_routing,
        forced_intent=forced_intent,
        thread_id=effective_thread_id,
        checkpointer=checkpointer,
        skill_roots=skill_roots,
        code_exec=code_exec,
        code_peer=code_peer,
        code_peer_model=code_peer_model,
        unified_peer=unified_peer,
        input_file_ids=input_file_ids,
    )
    web_utils.begin_turn()  # reset the per-turn open-web budget (see note in the streaming path)
    final_state = graph.invoke(
        {
            "query": query,
            "chat_history": chat_history or [],
            "thread_id": effective_thread_id,
        }
    )
    orchestration_result = _with_citation_urls(final_state.get("orchestration_result"))
    available_agent_names = final_state.get("available_agent_names") or []
    skill_registry = SkillRegistry.discover(skill_roots)
    response: Dict[str, Any] = {
        "orchestration_result": orchestration_result,
        "route_trace": build_orchestration_trace(
            query=query,
            chat_history=chat_history,
            available_agent_names=available_agent_names,
            orchestration_result=orchestration_result if isinstance(orchestration_result, dict) else {},
        ),
        "available_skills": skill_registry.catalog(),
    }
    final_answer = final_state.get("final_answer") or extract_final_answer(orchestration_result)
    if final_answer:
        response["final_answer"] = final_answer
    audit = final_state.get("audit")
    if audit:
        response["grounding_audit"] = audit
    if effective_thread_id:
        response["thread_id"] = effective_thread_id
    return response


# ---------------------------------------------------------------------------
# Streaming query execution
# ---------------------------------------------------------------------------

def stream_agent_query_events(
    query: str,
    *,
    chat_history: Optional[List[Any]] = None,
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
    agent_dev: Optional[bool] = None,
    code_exec: Optional[bool] = None,
    code_peer: Optional[str] = None,
    code_peer_model: Optional[str] = None,
    unified_peer: Optional[bool] = None,
    input_file_ids: Optional[List[str]] = None,
) -> Generator[Dict[str, Any], None, None]:
    """Yield structured SSE events while running a query.

    ``agent_dev`` controls whether detail-tier events (tool I/O, LLM
    interactions, routing detail) are streamed in addition to the always-on
    execution-state status events.  None falls back to the ``AGENT_DEV`` env var.
    """
    effective_thread_id = resolve_thread_id(thread_id, checkpointer)
    dev_enabled = agent_dev if agent_dev is not None else is_agent_dev()
    yield {
        "event": "status",
        "data": {
            "stage": "started",
            "thread_id": effective_thread_id or thread_id,
            "tool_strategy": tool_strategy,
        },
    }
    # Advertise the agents that will ACTUALLY run. There is one orchestration path, and it
    # executes search/analyze/code peers — the agents-as-tools names it used to be able to
    # advertise instead went with that arm.
    available_agent_names = ["search", "analyze", "code"]
    skill_registry = SkillRegistry.discover(skill_roots)
    yield {
        "event": "status",
        "data": {
            "stage": "initialized",
            "thread_id": effective_thread_id or thread_id,
            "tool_strategy": tool_strategy,
            "available_agents": available_agent_names,
            "available_skills": skill_registry.catalog(),
        },
    }
    yield {"event": "status", "data": {"stage": "orchestration_agent_started"}}

    outbox: "queue.Queue[Dict[str, Any]]" = queue.Queue()
    worker_error: List[BaseException] = []

    def _enqueue(item: Dict[str, Any]) -> None:
        outbox.put(item)

    def _worker() -> None:
        try:
            # The open-web budget lives in a ContextVar, and a new thread starts with a FRESH
            # context -- so resetting it in the caller would be invisible here. It has to happen on
            # this thread, and it has to happen at all: a gunicorn sync worker reuses its thread
            # across requests, so without a reset the previous turn's spend would starve this one.
            web_utils.begin_turn()
            with trace_context(_enqueue, agent_role="orchestrator_agent", agent_dev=agent_dev):
                graph = build_orchestrator_graph(
                    llm=llm,
                    verbose=verbose,
                    return_intermediate_steps=return_intermediate_steps,
                    tool_strategy=tool_strategy,
                    include_mcp_tools=include_mcp_tools,
                    mcp_modules=mcp_modules,
                    enabled_search_methods=enabled_search_methods,
                    smart_tool_routing=smart_tool_routing,
                    forced_intent=forced_intent,
                    thread_id=effective_thread_id,
                    checkpointer=checkpointer,
                    skill_roots=skill_roots,
                    code_exec=code_exec,
                    code_peer=code_peer,
                    code_peer_model=code_peer_model,
                    unified_peer=unified_peer,
                    input_file_ids=input_file_ids,
                )
                final_state = graph.invoke(
                    {
                        "query": query,
                        "chat_history": chat_history or [],
                        "thread_id": effective_thread_id,
                    }
                )
                # Same citation links as the non-streaming path. Applied HERE as well because
                # this path builds its own orchestration_result and never calls
                # run_agent_query — the map UI streams, so decorating only that function left
                # the "Sources used" list unlinked in the only client that uses it.
                orchestration_result = _with_citation_urls(final_state.get("orchestration_result"))
                available_agent_names_local = final_state.get("available_agent_names") or available_agent_names
                artifacts = extract_search_artifacts(orchestration_result if isinstance(orchestration_result, dict) else {})
                route_trace = build_orchestration_trace(
                    query=query,
                    chat_history=chat_history,
                    available_agent_names=available_agent_names_local,
                    orchestration_result=orchestration_result if isinstance(orchestration_result, dict) else {},
                )
                # Detail-tier post-run summary: emit ONLY the route trace + routing
                # decision (which the live callbacks don't produce). The per-step
                # llm_interaction / tool_call / tool_result events were already
                # streamed live during the run by the callback handler — replaying
                # them here duplicated every step (and is dead on the supervisor
                # path, whose result carries no message-shaped artifacts).
                if dev_enabled:
                    _enqueue({"event": "route_trace", "data": route_trace})
                    _enqueue(
                        {
                            "event": "decision",
                            "data": {
                                "kind": "agent_route_decision",
                                "agent": "orchestrator_agent",
                                "query": query,
                                "route": route_trace.get("route"),
                                "called_tools": route_trace.get("called_tools") or [],
                                "analysis_called_tools": route_trace.get("analysis_called_tools") or [],
                                "selected_skills": route_trace.get("selected_skills") or [],
                                "chat_history_available": route_trace.get("chat_history_available"),
                                "route_trace": route_trace,
                            },
                        }
                    )
                if "search_agent_evidence" in route_trace.get("called_tools", []):
                    _enqueue(
                        {
                            "event": "search_complete",
                            "data": {
                                "summary": extract_final_answer(orchestration_result) or "",
                                "tool_call_count": len(artifacts.get("tool_calls") or []),
                                "tool_result_count": len(artifacts.get("tool_results") or []),
                            },
                        }
                    )
                final_answer = final_state.get("final_answer") or extract_final_answer(orchestration_result)
                if final_answer:
                    _enqueue({"event": "final_answer", "data": {"answer": final_answer}})
                _enqueue({"event": "status", "data": {"stage": "orchestration_agent_completed"}})
                response: Dict[str, Any] = {
                    "orchestration_result": orchestration_result,
                    "route_trace": route_trace,
                    "available_skills": skill_registry.catalog(),
                }
                if final_answer:
                    response["final_answer"] = final_answer
                audit = final_state.get("audit")
                if audit:
                    response["grounding_audit"] = audit
                if effective_thread_id:
                    response["thread_id"] = effective_thread_id
                _enqueue({"event": "completed", "data": response})
        except AgentInvocationError as exc:
            readable_trace = exc.diagnostics.get("readable_trace") if exc.diagnostics else None
            _enqueue(
                {
                    "event": "error",
                    "data": {
                        "message": str(exc),
                        "diagnosticText": readable_trace,
                        "diagnostics": exc.diagnostics,
                    },
                }
            )
            worker_error.append(exc)
        except Exception as exc:
            diagnostics = getattr(exc, "diagnostics", None)
            payload = {"message": str(exc)}
            if diagnostics:
                payload["diagnostics"] = diagnostics
                if isinstance(diagnostics, dict) and diagnostics.get("readable_trace"):
                    payload["diagnosticText"] = diagnostics.get("readable_trace")
            _enqueue({"event": "error", "data": payload})
            worker_error.append(exc)
        finally:
            _enqueue({"event": "__worker_done__", "data": {}})

    # The worker inherits the REQUEST's context. ContextVars do not cross threads, so anything
    # bound at the edge for the duration of a turn — the file store's session, and any future
    # per-request scope — was simply absent in the code that does the work: a file written by a
    # tool came out with session=None while the request had one bound. copy_context() takes a
    # snapshot here, on the caller's side, and ctx.run executes the worker inside it.
    _ctx = contextvars.copy_context()
    worker = threading.Thread(target=lambda: _ctx.run(_worker),
                              name="agent-stream-worker", daemon=True)
    worker.start()

    # Emit a keepalive whenever the agent goes quiet (long LLM turn / sandbox run) so no
    # client or intermediary times out the stream mid-run. Node's fetch (undici) kills a
    # response body that is silent for 300s ("Body Timeout Error"); proxies have similar
    # idle limits. The API layer renders this event as an SSE comment line.
    try:
        heartbeat_s = max(1.0, float(os.getenv("AGENT_STREAM_HEARTBEAT_SECONDS", "20")))
    except (TypeError, ValueError):
        heartbeat_s = 20.0
    last_emit = time.monotonic()

    while True:
        try:
            item = outbox.get(timeout=0.25)
        except queue.Empty:
            if worker.is_alive():
                if time.monotonic() - last_emit >= heartbeat_s:
                    last_emit = time.monotonic()
                    yield {"event": "keepalive", "data": {}}
                continue
            break
        if item.get("event") == "__worker_done__":
            break
        last_emit = time.monotonic()
        yield item

    worker.join(timeout=0)
    while not outbox.empty():
        item = outbox.get()
        if item.get("event") != "__worker_done__":
            yield item
    if worker_error:
        raise worker_error[0]


# ---------------------------------------------------------------------------
# Code agent query
# ---------------------------------------------------------------------------


def _print_tool_trace(result: Any) -> None:
    if not isinstance(result, dict):
        return

    def _print_messages(agent_name: str, payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return False

        printed_local = False
        print(f"- AGENT {agent_name} INVOKE")
        for msg in messages:
            tool_calls = getattr(msg, "tool_calls", None)
            if isinstance(tool_calls, list) and tool_calls:
                for call in tool_calls:
                    name = call.get("name", "unknown_tool")
                    args = call.get("args", {})
                    print(f"  - CALL {name} args={args}")
                    printed_local = True
                continue

            name = getattr(msg, "name", None)
            tool_call_id = getattr(msg, "tool_call_id", None)
            content = getattr(msg, "content", None)
            if name and tool_call_id:
                text = content if isinstance(content, str) else str(content)
                snippet = text if len(text) <= 240 else f"{text[:240]}..."
                print(f"  - RESULT {name} ({tool_call_id}): {snippet}")
                printed_local = True

        if not printed_local:
            print("  - (no tool calls)")
        return True

    print("\nTool trace:")
    has_two_agent_result = isinstance(result.get("search_result"), dict) or isinstance(result.get("analysis_result"), dict)
    if has_two_agent_result:
        printed_any = False
        printed_any = _print_messages("SearchAgent", result.get("search_result")) or printed_any
        printed_any = _print_messages("AnalysisAgent", result.get("analysis_result")) or printed_any
        if not printed_any:
            print("- (no agent trace)")
        return

    if isinstance(result.get("code_result"), dict):
        printed_any = False
        printed_any = _print_messages("CodeAgent", result.get("code_result")) or printed_any
        for idx, item in enumerate(result.get("code_agent_search_invocations") or [], start=1):
            route = item.get("route_trace") if isinstance(item, dict) else None
            intent = route.get("intent") if isinstance(route, dict) else None
            search_payload = item.get("search_result") if isinstance(item, dict) else None
            if _print_messages(f"SearchAgent (via CodeAgent #{idx})", search_payload):
                if intent:
                    print(f"  - ROUTE intent={intent}")
                printed_any = True
            else:
                print(f"- AGENT SearchAgent (via CodeAgent #{idx}) INVOKE")
                if intent:
                    print(f"  - ROUTE intent={intent}")
                print("  - (no tool calls)")
                printed_any = True
        if not printed_any:
            print("- (no agent trace)")
        return

    if not _print_messages("Agent", result):
        print("- (no tool calls)")


def _print_route_trace(result: Any) -> None:
    if not isinstance(result, dict):
        return
    route = result.get("route_trace")
    if not isinstance(route, dict):
        return
    print("\nRoute trace:")
    print(f"- route: {route.get('route')}")
    print(f"- intent: {route.get('intent')}")
    print(f"- reason: {route.get('reason')}")
    print(f"- analysis_hits: {route.get('analysis_hits')}")
    print(f"- code_hits: {route.get('code_hits')}")
    print(f"- discovery_hits: {route.get('discovery_hits')}")
    print(f"- allowed_tools: {route.get('allowed_tools')}")
    print(f"- selected_skills: {route.get('selected_skills')}")


def main() -> None:
    """CLI entry point for running a single agent query."""
    parser = argparse.ArgumentParser(description="Run the LangChain RAG agent once.")
    parser.add_argument("query", help="User query for the agent.")
    parser.add_argument("--verbose", action="store_true", help="Enable AgentExecutor verbose logs.")
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Conversation thread id used by the LangGraph checkpointer for short-term memory.",
    )
    parser.add_argument(
        "--agent-mode",
        default="analysis",
        choices=["analysis", "code"],
        help="Agent pipeline mode: analysis (SearchAgent -> AnalysisAgent) or code (CodeAgent with SearchAgent tool).",
    )
    parser.add_argument(
        "--tool-strategy",
        default="granular",
        choices=["granular"],
        help="Tool mode: granular (the only mode; full_pipeline was removed).",
    )
    parser.add_argument(
        "--include-mcp-tools",
        action="store_true",
        help="Also load MCP tools from MCP_server/tools (search/data/spatial/image adapters).",
    )
    parser.add_argument(
        "--mcp-modules",
        default="search_tools,data_tools,spatial_analysis_tools,image_tools",
        help="Comma-separated MCP tool module names to load when --include-mcp-tools is set.",
    )
    parser.add_argument(
        "--no-smart-routing",
        action="store_true",
        help="Disable intent-based tool filtering and allow the full selected tool set.",
    )
    parser.add_argument(
        "--force-intent",
        default=None,
        choices=["general_discovery", "analysis_task", "code_task", "hybrid"],
        help="Force routing intent instead of automatic classification.",
    )
    parser.add_argument(
        "--skill-paths",
        default=None,
        help="Comma-separated directories containing SKILL.md skill bundles. Defaults to skills/ and .agents/skills/.",
    )
    args = parser.parse_args()
    selected_mcp_modules = [item.strip() for item in args.mcp_modules.split(",") if item.strip()]
    selected_skill_paths = None
    if args.skill_paths:
        selected_skill_paths = [item.strip() for item in args.skill_paths.split(",") if item.strip()]

    runner = run_agent_query
    try:
        result = runner(
            args.query,
            verbose=args.verbose,
            thread_id=args.thread_id,
            tool_strategy=args.tool_strategy,
            include_mcp_tools=args.include_mcp_tools,
            mcp_modules=selected_mcp_modules,
            smart_tool_routing=not args.no_smart_routing,
            forced_intent=args.force_intent,
            skill_roots=selected_skill_paths,
        )
    except AgentInvocationError as exc:
        print("\nAgent invocation failed.")
        print(str(exc))
        if exc.diagnostics:
            readable_trace = exc.diagnostics.get("readable_trace")
            if readable_trace:
                print("\nInteraction trace:")
                print(readable_trace)
            else:
                print("\nDiagnostics:")
                print(json.dumps(exc.diagnostics, ensure_ascii=True, indent=2, default=str))
        raise
    print(result)
    _print_route_trace(result)
    _print_tool_trace(result)
    final_answer = extract_final_answer(result)
    if final_answer:
        print("\nFinal answer:")
        print(final_answer)


if __name__ == "__main__":
    main()
