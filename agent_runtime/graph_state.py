"""Shared type definitions and constants for the agent runtime.

Every other module in ``agent_runtime`` may import from here.
Nothing in this file should import from ``rag_pipeline`` so the
dependency direction stays clean.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence

# ---------------------------------------------------------------------------
# Intent / role / route literals
# ---------------------------------------------------------------------------
AgentIntent = Literal["general_discovery", "analysis_task", "code_task", "hybrid"]
AgentRole = Literal["search", "analysis", "code", "direct_answer", "orchestrator"]
RouteType = Literal["search", "analysis", "code", "direct_answer"]

# ---------------------------------------------------------------------------
# Tool-name sets
#
# These are the *canonical* names that tool_policy and intent_classifier
# use to decide which tools an agent is allowed to call.  If a new MCP
# tool is added, register it in the appropriate set here.
# ---------------------------------------------------------------------------
ANALYSIS_TOOL_NAMES: set[str] = {
    "mcp_load_chicago_community_areas",
    "mcp_load_chicago_crime_data",
    "mcp_get_crime_statistics",
    "mcp_count_crimes_per_community",
    "mcp_generate_crime_map",
    "qgis_processing_help",
    "qgis_processing_run",
    "qgis_metric_buffer",
    "pyqgis_layer_summary",
    "pyqgis_render_map",
}

DISCOVERY_TOOL_NAMES: set[str] = {
    "rag_tool",
    "mcp_search_geospatial_resources",
    "mcp_search_publications",
}

RAG_COMPONENT_TOOL_NAMES: set[str] = {
    "keyword_search",
    "semantic_search",
    "neo4j_search",
    "neo4j_get_element_by_id",
    "neo4j_explore_related_nodes",
    "spatial_search",
    "opengeodata_search",
    # Open-web tools. Omitting them here does not merely skip a nicety: tool_policy filters the
    # agent's toolset against these sets, so a name absent from here is stripped for EVERY intent
    # on the smart-routing path — the tool would be registered, documented and unreachable.
    "web_search",
    "web_fetch",
    "agent_kb_search",
    "get_kb_block",
    # The method library. Same filter caveat as the web tools above: absent from this set, both
    # are registered, described and unreachable for every intent.
    "kb_method_search",
    "get_method_contract",
    # Staging is what turns a dataset element from readable into runnable: the generated loaders
    # take a staged path and, before these, nothing could produce one.
    "stage_element",
    "stage_url",
    "list_staged_inputs",
    # Reading one element's source file, by id. The live MCP server exposes this as
    # `mcp_fetch_element_source`; SEARCH_AGENT_PROMPT rule 8 has always told the model to call
    # it, under the wrong (unprefixed) name AND with the real name absent from every set here,
    # so it was stripped for every intent except analysis_task — where it survived only because
    # the empty-selection fallback at tool_policy.py fires and returns everything.
    "mcp_fetch_element_source",
}

FILE_TOOL_NAMES: set[str] = {
    "read_text_file",
    "inspect_file_for_analysis",
    "write_text_file",
    "write_output_file",
}

SKILL_TOOL_NAMES: set[str] = {
    "list_available_skills",
    "load_skill",
}

# Evidence-quality tools (LLM rerank + grounding/hallucination audit). Always
# allowed regardless of intent — they are post-processing helpers, not retrieval.
QUALITY_TOOL_NAMES: set[str] = {
    "rerank_evidence",
    "audit_answer_grounding",
}

# Sandboxed code execution (container-per-run). Always allowed when present; only
# attached to the code/analysis agents, and only when AGENT_CODE_EXEC is enabled.
EXECUTION_TOOL_NAMES: set[str] = {
    "execute_code",
}

# Alias kept for backward compatibility — identical to RAG_COMPONENT_TOOL_NAMES.
IGUIDE_SEARCH_TOOL_NAMES: set[str] = RAG_COMPONENT_TOOL_NAMES.copy()
