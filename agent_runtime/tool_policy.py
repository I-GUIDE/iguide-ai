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
        selected = [name for name in ANALYSIS_TOOL_NAMES if name in available]
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
    from agent_runtime.skills import make_skill_tools

    # One tool set. `full_pipeline` — a single rag_tool wrapping the whole stage-1 pipeline —
    # was removed with the agents-as-tools arm: the granular tools are its superset, and a
    # strategy that bypasses them also bypasses everything built on them.
    strategy = (tool_strategy or "granular").strip().lower()
    if strategy != "granular":
        raise ValueError(
            f"tool_strategy must be 'granular'; {tool_strategy!r} is no longer supported "
            "(full_pipeline was removed — the granular tools cover it).")
    tools = make_langchain_granular_tools(
        enabled_search_methods=enabled_search_methods,
        include_file_tools=include_file_tools,
        session_id=session_id,
    )
    if include_mcp_tools:
        tools.extend(make_langchain_mcp_tools(include_modules=mcp_modules))
    tools.extend(make_quality_tools())
    tools.extend(make_skill_tools(skill_roots=skill_roots))
    return tools
