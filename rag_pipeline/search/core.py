"""
search_core.py
--------------
Orchestrates all retrieval strategies for the I-GUIDE RAG pipeline.

Search order:
  1. Keyword search      (always)
  2. Semantic search     (always)
  3. Neo4j graph search  (LLM-routed — uses 3-tier agent in search_agents.py)
  4. Spatial search      (LLM-routed)
  5. OpenGeoData         (always)
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, MutableMapping

from .opengeodata import retrieve_opengeodata
from .keyword import retrieve_keyword
from .neo4j import retrieve_neo4j          # basic keyword fallback, kept for reference
from .semantic import retrieve_semantic
from .spatial import retrieve_spatial
from ..state import (
    AgentState,
    RoutingDecision,
    ensure_state_shapes,
    get_query_text,
    merge_retrieval,
    summarize_evidence,
)
from .utils import get_logger
from .utils import default_top_k  # shared retrieval window

logger = get_logger("search_core")

# Neo4j agent (3-tier: pattern tools → Text2Cypher → keyword fallback)
try:
    from .agents import get_neo4j_agent_results
    NEO4J_AGENT_AVAILABLE = True
except ImportError as exc:
    logger.warning("Neo4j agent search unavailable: %s", exc)
    NEO4J_AGENT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record_decision(decisions: List[RoutingDecision], source: str, reason: str) -> None:
    decisions.append(RoutingDecision(source=source, reason=reason))


def _limit_for(state: MutableMapping[str, Any]) -> int:
    params = state.get("params") or {}
    try:
        return max(1, int(params.get("top_k", default_top_k())))
    except (TypeError, ValueError):
        return 8


def _log_hits(label: str, hits: List[Any], max_shown: int = 10) -> None:
    titles = [h.get("_source", {}).get("title", "No title") for h in hits]
    logger.info("%s returned %d documents:", label, len(titles))
    for i, title in enumerate(titles[:max_shown], 1):
        logger.info("  %d. %s", i, title)
    if len(titles) > max_shown:
        logger.info("  ... and %d more", len(titles) - max_shown)


# ---------------------------------------------------------------------------
# LLM routing
# ---------------------------------------------------------------------------

def _llm_route(query: str, decisions: List[RoutingDecision]) -> tuple[bool, bool, dict]:
    """
    Ask the LLM router whether to run graph and/or spatial search.
    Returns (use_graph, use_spatial, rationale_dict).
    Falls back to conservative heuristics on any error.
    """
    use_llm_routing = os.getenv("USE_LLM_ROUTING", "true").lower() in ("true", "1", "yes")

    if use_llm_routing:
        try:
            from ..router_llm import LLMRouter, ModuleRegistry
            registry = ModuleRegistry()
            router = LLMRouter(registry=registry)
            # Read decision directly — plan.chosen_modules filters by registry
            # which is empty by default, so graph/spatial get dropped from chosen_modules
            decision = router.llm_engine.decide(query)
            rationale = decision.rationale or {}

            # Check if LLM failed internally and returned conservative fallback
            llm_failed = any(
                "failed" in str(v).lower() or "keyword-only" in str(v).lower()
                for v in rationale.values()
            )
            if not llm_failed:
                use_graph   = decision.use_graph
                use_spatial = decision.use_spatial
                logger.info(
                    "LLM Router → graph=%s spatial=%s | rationale: %s",
                    use_graph, use_spatial, rationale,
                )
                _record_decision(
                    decisions, "llm_router",
                    f"graph={use_graph} spatial={use_spatial}",
                )
                return use_graph, use_spatial, rationale

            logger.warning("LLM router returned failure rationale; using pattern fallback.")
            _record_decision(decisions, "llm_router_fallback", "LLM call failed internally")

        except Exception as exc:
            logger.warning("LLM routing failed (%s); using pattern fallback.", exc)
            _record_decision(decisions, "llm_router_fallback", f"error: {str(exc)[:100]}")

    # Simple heuristic fallback — uses pattern detection so author/org/tag
    # queries still trigger graph search even when the LLM router is unavailable
    from .neo4j_graph_tools import detect_pattern  # noqa: F811
    use_graph = detect_pattern(query) is not None
    use_spatial = False
    _record_decision(decisions, "heuristic_routing", "LLM routing disabled or errored")
    return use_graph, use_spatial, {}


# ---------------------------------------------------------------------------
# Main retrieval orchestrator
# ---------------------------------------------------------------------------

def run_retrieval(state: MutableMapping[str, Any]) -> AgentState:
    state = ensure_state_shapes(state)
    limit = _limit_for(state)
    query = get_query_text(state)
    decisions: List[RoutingDecision] = []

    logger.info("=== Retrieval start | query='%s' top_k=%d ===", query, limit)

    # ── 1. Keyword search (always) ────────────────────────────────────────
    keyword_hits = retrieve_keyword(state)
    _log_hits("📚 KEYWORD", keyword_hits)
    appended = merge_retrieval(state, source="keyword", hits=keyword_hits, limit=None)
    _record_decision(decisions, "keyword", f"hits:{len(keyword_hits)} appended:{len(appended)}")

    # ── 2. Semantic search (always) ───────────────────────────────────────
    semantic_hits = retrieve_semantic(state)
    _log_hits("🔍 SEMANTIC", semantic_hits)
    appended = merge_retrieval(state, source="semantic", hits=semantic_hits, limit=None)
    _record_decision(decisions, "semantic", f"hits:{len(semantic_hits)} appended:{len(appended)}")

    # ── 3. LLM routing for graph + spatial ───────────────────────────────
    use_graph, use_spatial, rationale = _llm_route(query, decisions)

    # ── 4. Neo4j graph search (LLM-routed) ───────────────────────────────
    if use_graph:
        use_agent = os.getenv("USE_NEO4J_AGENT_SEARCH", "true").lower() in ("true", "1", "yes")

        if use_agent and NEO4J_AGENT_AVAILABLE:
            try:
                neo_hits = get_neo4j_agent_results(query, limit=limit)
                search_method = "neo4j_agent"
            except Exception as exc:
                logger.warning("Neo4j agent search failed (%s); falling back to basic.", exc)
                neo_hits = retrieve_neo4j(state)
                search_method = "neo4j_basic_fallback"
        else:
            logger.info("Basic Neo4j search (set USE_NEO4J_AGENT_SEARCH=true for agent).")
            neo_hits = retrieve_neo4j(state)
            search_method = "neo4j_basic"

        _log_hits("🕸️  NEO4J", neo_hits)
        appended = merge_retrieval(state, source="neo4j", hits=neo_hits, limit=None)
        reason = rationale.get("graph", "enabled")
        _record_decision(
            decisions, "neo4j",
            f"method:{search_method} hits:{len(neo_hits)} appended:{len(appended)} reason:{reason}",
        )
    else:
        _record_decision(decisions, "neo4j", f"skipped reason:{rationale.get('graph', 'disabled')}")

    # ── 5. Spatial search (LLM-routed) ───────────────────────────────────
    if use_spatial:
        spatial_hits = retrieve_spatial(state)
        _log_hits("🗺️  SPATIAL", spatial_hits)
        appended = merge_retrieval(state, source="spatial", hits=spatial_hits, limit=None)
        reason = rationale.get("spatial", "enabled")
        _record_decision(
            decisions, "spatial",
            f"hits:{len(spatial_hits)} appended:{len(appended)} reason:{reason}",
        )
    else:
        _record_decision(decisions, "spatial", f"skipped reason:{rationale.get('spatial', 'disabled')}")

    # ── 6. OpenGeoData (always, no limit cap) ────────────────────────────
    opengeo_hits = retrieve_opengeodata(state)
    _log_hits("🌍 OPENGEODATA", opengeo_hits)
    appended = merge_retrieval(
        state,
        source="opengeodata",
        hits=opengeo_hits,
        limit=None,   # always include — capped later during reranking
    )
    logger.info("OpenGeoData: %d documents appended to evidence", len(appended))
    _record_decision(
        decisions, "opengeodata",
        f"hits:{len(opengeo_hits)} appended:{len(appended)}",
    )

    # ── Summary ───────────────────────────────────────────────────────────
    total = len(state.get("evidence", {}).get("retrieved_documents", []))
    logger.info("=== Retrieval complete: %d total documents in evidence ===", total)

    trace = state.setdefault("trace_observability", {})
    trace["retrieval_summary"] = summarize_evidence(state["evidence"])
    trace["retrieval_routing_decisions"] = [
        {"source": d.source, "reason": d.reason} for d in decisions
    ]
    return state  # type: ignore[return-value]


__all__ = ["run_retrieval"]