from __future__ import annotations

import argparse
from typing import Any

from .executor_factory import (
    DEFAULT_CHECKPOINTER,
    build_agent_executor,
    build_code_agent_executor,
)
from .graph_runtime import (
    run_agent_query,
    stream_agent_query_events,
)
from .runtime_utils import extract_final_answer


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
    printed_any = False
    for label, key in (
        ("SearchAgent", "search_result"),
        ("AnalysisAgent", "analysis_result"),
        ("CodeAgent", "code_result"),
    ):
        printed_any = _print_messages(label, result.get(key)) or printed_any
    if not printed_any:
        print("- (no agent trace)")


def _print_route_trace(result: Any) -> None:
    if not isinstance(result, dict):
        return
    route = result.get("route_trace")
    if not isinstance(route, dict):
        return
    print("\nRoute trace:")
    print(f"- intent: {route.get('intent')}")
    print(f"- role: {route.get('role')}")
    print(f"- route_type: {route.get('route_type')}")
    print(f"- reason: {route.get('reason')}")
    print(f"- allowed_tools: {route.get('allowed_tools')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LangGraph agent once.")
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
        help="Agent pipeline mode: analysis or code.",
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
        help="Also load MCP tools from MCP_server/tools.",
    )
    parser.add_argument(
        "--mcp-modules",
        default="search_tools,data_tools,spatial_analysis_tools,biomass_tools,image_tools",
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
    args = parser.parse_args()
    selected_mcp_modules = [item.strip() for item in args.mcp_modules.split(",") if item.strip()]

    runner = run_agent_query
    result = runner(
        args.query,
        verbose=args.verbose,
        thread_id=args.thread_id,
        tool_strategy=args.tool_strategy,
        include_mcp_tools=args.include_mcp_tools,
        mcp_modules=selected_mcp_modules,
        smart_tool_routing=not args.no_smart_routing,
        forced_intent=args.force_intent,
    )
    print(result)
    _print_route_trace(result)
    _print_tool_trace(result)
    final_answer = extract_final_answer(result)
    if final_answer:
        print("\nFinal answer:")
        print(final_answer)


__all__ = [
    "DEFAULT_CHECKPOINTER",
    "build_agent_executor",
    "build_code_agent_executor",
    "run_agent_query",
    "stream_agent_query_events",
]


if __name__ == "__main__":
    main()
