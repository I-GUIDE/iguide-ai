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
    "qgis_map_image",
    # Elevation. Needs no credential and no attached file — the region can be a bbox, a point,
    # or a boundary the same turn just fetched — so it belongs to the analysis intent rather
    # than to the upload-gated set. On the supervisor path the peers get their tools directly
    # and never consult this set; this entry is what keeps it from being stripped on the
    # smart-routing path, where a name absent from every set is registered and unreachable.
    "dem_for_region",
    # The raster -> zone bridge. Its inputs are file_ids, but they are ones the same turn
    # PRODUCED (a DEM, a boundary) rather than ones a user uploaded, so it is not upload-gated
    # either; a turn that asks for elevation per tract reaches both of these or neither.
    "zonal_stats_for_raster",
    # Arithmetic on a DEM this deployment already fetched — no service, no credential, and the
    # same reachability as dem_for_region, which is the tool a turn arrives at them through.
    "terrain_derivative",
    "inundation_at_level",
}

DISCOVERY_TOOL_NAMES: set[str] = {
    "rag_tool",
    # NOTE: mcp_web_search_geo_links (formerly mcp_search_geospatial_resources) is deliberately
    # absent. It is a plain DuckDuckGo web search — links only, no geometry — whose old name
    # out-competed the real geo tools for anything "geospatial". web_search/web_fetch + opengeodata_search cover discovery;
    # overpass_search covers real-world features. Leaving it out of every policy set keeps the
    # smart-routing path from selecting it.
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
    "overpass_search",
    # Open-web tools. Omitting them here does not merely skip a nicety: tool_policy filters the
    # agent's toolset against these sets, so a name absent from here is stripped for EVERY intent
    # on the smart-routing path — the tool would be registered, documented and unreachable.
    "web_search",
    "web_fetch",
    "agent_kb_search",
    "get_kb_block",
}

FILE_TOOL_NAMES: set[str] = {
    "read_text_file",
    "inspect_file_for_analysis",
    "write_text_file",
    "write_output_file",
    "list_conversation_files",
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
