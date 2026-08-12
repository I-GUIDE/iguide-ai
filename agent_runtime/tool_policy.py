"""Tool filtering and collection for the agent runtime.

Decides which tools are *available* (``collect_tools``) and which subset
an agent is *allowed* to use for a given intent (``select_allowed_tools``).
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from agent_runtime.graph_state import (
    ANALYSIS_TOOL_NAMES,
    DISCOVERY_TOOL_NAMES,
    EXECUTION_TOOL_NAMES,
    FILE_TOOL_NAMES,
    QUALITY_TOOL_NAMES,
    RAG_COMPONENT_TOOL_NAMES,
    SKILL_TOOL_NAMES,
)


# ---------------------------------------------------------------------------
# Tool selection based on intent
# ---------------------------------------------------------------------------

def select_allowed_tools(intent: str, available_tool_names: Sequence[str]) -> List[str]:
    """Return the subset of *available_tool_names* allowed for *intent*."""
    available = set(available_tool_names)
    selected: List[str] = []

    if intent == "analysis_task":
        # Retrieval belongs here too. This branch used to select ANALYSIS_TOOL_NAMES ALONE —
        # 5 crime-MCP + 5 QGIS tools — which on the legacy path is applied to the SearchAgent
        # itself (legacy/graph_nodes.py passes the orchestrator's intent straight through), so
        # "Analyze flood risk in Illinois and map the results" built the evidence-gathering
        # agent with ZERO retrieval tools and the question was answered ungrounded.
        #
        # The `if not selected` safety net below never fired, because file/skill/quality tools
        # had already been appended — so the stripping was silent. An analysis still has to be
        # grounded in evidence; this makes analysis_task a superset of hybrid rather than a
        # disjoint set.
        preferred = (DISCOVERY_TOOL_NAMES | RAG_COMPONENT_TOOL_NAMES
                     | ANALYSIS_TOOL_NAMES | FILE_TOOL_NAMES)
        selected = [name for name in available_tool_names if name in preferred]
    elif intent == "code_task":
        preferred = DISCOVERY_TOOL_NAMES | RAG_COMPONENT_TOOL_NAMES | FILE_TOOL_NAMES
        selected = [name for name in available_tool_names if name in preferred]
    elif intent == "general_discovery":
        preferred = DISCOVERY_TOOL_NAMES | RAG_COMPONENT_TOOL_NAMES | FILE_TOOL_NAMES
        selected = [name for name in available_tool_names if name in preferred]
    else:  # hybrid
        preferred = DISCOVERY_TOOL_NAMES | RAG_COMPONENT_TOOL_NAMES | ANALYSIS_TOOL_NAMES | FILE_TOOL_NAMES
        selected = [name for name in available_tool_names if name in preferred]

    for file_tool_name in FILE_TOOL_NAMES:
        if file_tool_name in available and file_tool_name not in selected:
            selected.append(file_tool_name)
    for skill_tool_name in SKILL_TOOL_NAMES:
        if skill_tool_name in available and skill_tool_name not in selected:
            selected.append(skill_tool_name)
    for quality_tool_name in QUALITY_TOOL_NAMES:
        if quality_tool_name in available and quality_tool_name not in selected:
            selected.append(quality_tool_name)
    for exec_tool_name in EXECUTION_TOOL_NAMES:
        if exec_tool_name in available and exec_tool_name not in selected:
            selected.append(exec_tool_name)

    if not selected:
        selected = list(available_tool_names)
    return selected


# ---------------------------------------------------------------------------
# Tool collection (instantiation)
# ---------------------------------------------------------------------------

def collect_tools(
    *,
    tool_strategy: str,
    include_mcp_tools: bool,
    mcp_modules: Optional[List[str]],
    enabled_search_methods: Optional[List[str]] = None,
    include_file_tools: bool = True,
    session_id: Optional[str] = None,
    skill_roots: Optional[List[str]] = None,
) -> List[Any]:
    """Build the concrete list of LangChain ``StructuredTool`` objects.

    Imports are deferred so that this module stays lightweight when only
    ``select_allowed_tools`` is needed.
    """
    from agent_runtime.langchain_file_tools import make_langchain_file_tools
    from agent_runtime.langchain_granular_tools import make_langchain_granular_tools
    from agent_runtime.langchain_mcp_tools import make_langchain_mcp_tools
    from agent_runtime.langchain_quality_tools import make_quality_tools
    from agent_runtime.langchain_tool import make_langchain_rag_tool
    from agent_runtime.skills import make_skill_tools

    # Default to the granular tool set: full_pipeline (rag_tool) is the deprecated
    # path and must never be the silent fallback for a missing/empty strategy.
    strategy = (tool_strategy or "granular").strip().lower()
    if strategy == "granular":
        tools = make_langchain_granular_tools(
            enabled_search_methods=enabled_search_methods,
            include_file_tools=include_file_tools,
            session_id=session_id,
        )
    elif strategy == "full_pipeline":
        tools = [make_langchain_rag_tool(), *make_langchain_file_tools()]
    else:
        raise ValueError("tool_strategy must be either 'full_pipeline' or 'granular'.")
    if include_mcp_tools:
        tools.extend(make_langchain_mcp_tools(include_modules=mcp_modules))
    tools.extend(make_quality_tools())
    tools.extend(make_skill_tools(skill_roots=skill_roots))
    return tools
