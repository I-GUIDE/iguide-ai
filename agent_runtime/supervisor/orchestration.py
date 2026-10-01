"""Supervisor-over-peers orchestration entrypoint (the supervisor arm of the fork).

Exposed via the strategy registry (``agent_runtime.strategy``) and called by
``orchestrator_graph.orchestrate_node`` when the supervisor strategy is selected. Returns
the same ``OrchestratorState`` key set as the legacy arm so the public response contract is
path-agnostic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent_runtime.streaming_trace import emit_trace_event


def run_supervisor_orchestration(query: str, chat_history: Optional[List[Any]], cfg) -> Dict[str, Any]:
    # Imported at call time so monkeypatched test doubles on agent_runtime.supervisor.graph
    # (run_supervisor / default_*_fn) are honored, and to keep import-time deps light.
    from agent_runtime.supervisor.graph import (
        default_analyze_fn,
        default_code_fn,
        default_search_fn,
        run_supervisor,
    )

    # This said "Orchestrator agent started" for years. It was copied from the agents-as-tools
    # arm, where an orchestrator LLM really was wrapped over the sub-agents as tools and the
    # string was true. Here it names nothing: this function builds three peer callables and
    # hands them to run_supervisor — no LLM, no decision. The pair was asymmetric too, opening
    # as the orchestrator and closing as the supervisor graph. Renamed rather than deleted,
    # because every node_started has a node_completed and other clients read that lifecycle.
    # The arm that made it true was removed in 2026-09; see docs/agent-architecture-changes.md.
    emit_trace_event(
        "node_started",
        {"stage": "orchestrate", "message": "Supervisor started"},
        agent_role="supervisor",
        node="orchestrate",
    )
    sup_state = run_supervisor(
        query,
        chat_history=chat_history,
        llm=cfg.llm,
        thread_id=cfg.thread_id,
        search_fn=default_search_fn(
            llm=cfg.llm,
            tool_strategy=cfg.tool_strategy,
            include_mcp_tools=cfg.include_mcp_tools,
            mcp_modules=cfg.mcp_modules,
            enabled_search_methods=cfg.enabled_search_methods,
            skill_roots=cfg.skill_roots,
        ),
        analyze_fn=default_analyze_fn(
            llm=cfg.llm,
            include_mcp_tools=cfg.include_mcp_tools,
            mcp_modules=cfg.mcp_modules,
            skill_roots=cfg.skill_roots,
            code_exec=cfg.code_exec,
            input_file_ids=cfg.input_file_ids,
            # Mirrors the search_fn above: in unified mode the analyse peer owns retrieval, so
            # a request's enabledSearchMethods has to reach it or the allowlist silently
            # applies to only one of the two arms.
            enabled_search_methods=cfg.enabled_search_methods,
        ),
        code_fn=default_code_fn(
            llm=cfg.llm, skill_roots=cfg.skill_roots, code_exec=cfg.code_exec,
            input_file_ids=cfg.input_file_ids, code_peer=cfg.code_peer,
            code_peer_model=cfg.code_peer_model,
        ),
        # The search NODE needs the allowlist too, so its no-platform-evidence web fallback
        # respects a request that excluded web_search.
        enabled_search_methods=cfg.enabled_search_methods,
        unified_peer=getattr(cfg, "unified_peer", None),
    )
    emit_trace_event(
        "node_completed",
        {"stage": "orchestrate", "message": "Supervisor finished"},
        agent_role="supervisor",
        node="orchestrate",
    )
    return {
        "orchestration_result": sup_state,
        "final_answer": sup_state.get("final_answer", ""),
        "available_agent_names": ["search", "analyze", "code"],
        "audit": sup_state.get("audit") or {},
    }
