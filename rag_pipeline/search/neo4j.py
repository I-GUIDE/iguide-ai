from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Dict, List, MutableMapping, Optional

from .utils import get_logger, getenv, normalize_source_fields, safe_score
from ..state import EvidenceEntry, ensure_state_shapes, get_query_text, merge_retrieval
from .utils import default_top_k  # shared retrieval window

log = get_logger("search_neo4j")


@lru_cache(maxsize=1)
def _neo4j_components() -> Optional[Dict[str, Any]]:
    try:
        from neo4j import GraphDatabase
        from neo4j.graph import Node as Neo4jNode
    except Exception as exc:  # pragma: no cover - optional dependency
        log.debug("Neo4j driver unavailable: %s", exc)
        return None
    return {"GraphDatabase": GraphDatabase, "Node": Neo4jNode}


@lru_cache(maxsize=1)
def _neo4j_driver() -> Optional[Any]:
    components = _neo4j_components()
    if not components:
        return None

    uri = (
        getenv("NEO4J_CONNECTION_STRING", required=False)
        or getenv("NEO4J_URI", required=False)
        or getenv("NEO4J_CONNECTION_STRING")
    )
    user = (
        getenv("NEO4J_USER", required=False)
        or getenv("NEO4J_USERNAME", required=False)
        or getenv("NEO4J_USER")
    )
    password = getenv("NEO4J_PASSWORD")
    if not neo4j_enabled():
        log.info("NEO4J_ENABLED=0; skipping Neo4j driver creation")
        return None
    GraphDatabase = components["GraphDatabase"]
    try:
        return GraphDatabase.driver(uri, auth=(user, password), max_connection_lifetime=300)
    except Exception as exc:
        log.error("Failed to create Neo4j driver: %s", exc)
        return None


def neo4j_enabled() -> bool:
    """Whether the Neo4j-backed tools should be attempted at all.

    ``NEO4J_ENABLED=0`` short-circuits BEFORE the driver connects. Without it every turn on a
    machine that cannot reach the graph pays a full connect/auth round trip before falling
    back — observed both as "Unable to retrieve routing information" (host unreachable) and
    as an AuthError (host reachable, stale credentials). Both degrade correctly but cost
    latency on every single turn, which is exactly what a dev machine should be able to opt
    out of.

    Default ON: a deployment must not silently lose the graph because a variable is unset.
    """
    import os as _os
    return (_os.getenv("NEO4J_ENABLED", "1") or "").strip().lower() not in {"0", "false", "no", "off"}


def _neo4j_db() -> Optional[str]:
    value = getenv("NEO4J_DB", required=False, default="").strip()
    return value or None


def _neo4j_run(cypher: str, params: Dict[str, Any], driver: Optional[Any] = None) -> List[Dict[str, Any]]:
    db_driver = driver or _neo4j_driver()
    if db_driver is None:
        return []

    database = _neo4j_db()
    session_factory = db_driver.session
    with (session_factory(database=database) if database else session_factory()) as session:
        return list(session.run(cypher, **params))


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from",
    "has", "have", "how", "in", "is", "it", "its", "of", "on", "or", "that", "the", "their",
    "there", "these", "this", "to", "what", "when", "where", "which", "who", "with",
    # domain-generic words that match nearly everything in a geospatial catalog
    "data", "dataset", "datasets", "notebook", "notebooks", "platform", "me", "show", "find",
}


def neo4j_query_terms(query: str, *, max_terms: int = 8) -> List[str]:
    """Content terms for graph matching.

    The Cypher used to test ``toLower(r.title) CONTAINS toLower($q)`` with $q bound to the
    WHOLE query, i.e. a phrase-substring match. Measured against the live graph:

        "spatial accessibility hospitals" -> 0 hits
        "spatial accessibility"           -> 9 hits
        "accessibility"                   -> 9 hits

    So the graph arm returned nothing for essentially any natural-language question, and only
    worked when the entire query happened to be a literal substring. The Neo4j outage was
    masking this: the arm looked broken because it was unreachable, and stayed broken after
    it became reachable.
    """
    tokens = re.findall(r"[A-Za-z0-9_]+", (query or "").lower())
    terms = [t for t in tokens if len(t) > 2 and t not in _STOPWORDS]
    seen, out = set(), []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:max_terms] or tokens[:max_terms]


def _build_neo4j_keyword_cypher() -> str:
    """Match on ANY content term, and rank by HOW MANY matched.

    Any-term recall with match-count ranking behaves like the keyword arm the rest of the
    system uses: a question mentioning four concepts should surface the element covering
    three of them, not nothing at all.

    Restricted to knowledge-element labels via ``$labels``. Bare ``MATCH (r)`` also swept the
    1195 :Alias and 1191 :Contributor nodes — 2386 of the graph's 3205 nodes are not knowledge
    elements, and none of them carry ``visibility``, which ``is_public_visibility`` treats as
    public. So a person's name matching a query term could be returned as a search result.
    """
    return """
    MATCH (r)
    WHERE
      any(l IN labels(r) WHERE l IN $labels) AND
      any(t IN $terms WHERE
        (r.title IS NOT NULL    AND toLower(r.title)    CONTAINS t) OR
        (r.contents IS NOT NULL AND toLower(r.contents) CONTAINS t) OR
        (r.tags IS NOT NULL     AND any(tag IN r.tags WHERE toLower(tag) CONTAINS t)))
    WITH r,
         size([t IN $terms WHERE r.title IS NOT NULL AND toLower(r.title) CONTAINS t]) AS title_hits,
         size([t IN $terms WHERE r.contents IS NOT NULL AND toLower(r.contents) CONTAINS t]) AS body_hits
    WITH r, title_hits, body_hits,
         (2.0 * title_hits) + (1.0 * body_hits) AS relevance,
         coalesce(log10(toFloat(coalesce(r.click_count, 0)) + 1), 0) AS popularity,
         coalesce(toFloat(count { (r)--() }), 0) AS connectivity
    WITH r,
         relevance + (popularity * 0.2) + (connectivity * 0.05) AS score
    RETURN r AS node, score
    ORDER BY score DESC
    LIMIT $limit
    """


def _extract_node_from_record(record: Dict[str, Any], node_cls: Optional[type]) -> Optional[Any]:
    node = record.get("node")
    if node_cls and isinstance(node, node_cls):
        return node
    for value in record.values():
        if node_cls and isinstance(value, node_cls):
            return value
    return None


def _records_to_hits(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    components = _neo4j_components()
    node_cls = components["Node"] if components else None

    hits: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        node = _extract_node_from_record(record, node_cls)
        score = safe_score(record.get("score", 1.0))

        if node is not None:
            properties = dict(node)
            # The platform UUID, NOT Neo4j's internal element id. Measured on the live graph:
            # the `_id` property exists on 0 of 3205 nodes, so this fallback chain always
            # reached `element_id` and emitted "4:f84f361b-...:532" as the doc_id. Every other
            # retrieval arm keys doc_id on the platform UUID, so graph hits could never dedupe
            # against them, could never match an expected id, and cited a link that resolves
            # to nothing. `id` is present on all 819 knowledge-element nodes.
            doc_id = (properties.get("id")
                      or properties.get("_id")
                      or getattr(node, "element_id", f"node:{idx}"))
        else:
            properties = {k: v for k, v in record.items() if isinstance(v, (str, int, float, list, dict))}
            doc_id = properties.get("doc_id", f"row:{idx}")

        from rag_pipeline.search.neo4j_graph_tools import is_public_visibility

        if not is_public_visibility(properties.get("visibility")):
            continue  # unlisted element -> never surfaced by search
        source = normalize_source_fields(properties, str(doc_id))
        hits.append(
            {
                "_id": str(doc_id),
                "_score": score,
                "_source": source,
            }
        )
    return hits


def get_neo4j_search_results(
    user_query: str,
    limit: int = 12,
    *,
    driver: Optional[Any] = None,
    cypher: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Perform a keyword-style search against Neo4j without any LLM dependency.

    Environment variables used when `driver` is not supplied:
    - NEO4J_CONNECTION_STRING / NEO4J_URI
    - NEO4J_USER (or NEO4J_USERNAME)
    - NEO4J_PASSWORD
    - NEO4J_DB (optional)

    For testing, pass a mocked `driver` or override the `cypher`.
    Returns hits in the same shape as `get_keyword_search_results`.
    """
    query = (user_query or "").strip()
    if not query:
        return []

    # $terms, not $q: the old binding made this a whole-phrase substring match, so any
    # multi-word question returned nothing. Keep "q" too for a caller-supplied cypher.
    terms = neo4j_query_terms(query)
    if not terms:
        return []
    from rag_pipeline.search.neo4j_graph_tools import _get_resource_labels

    params = {"q": query, "terms": terms, "labels": sorted(_get_resource_labels()),
              "limit": max(1, min(int(limit or default_top_k()), 100))}
    cypher_stmt = cypher or _build_neo4j_keyword_cypher()

    try:
        records = _neo4j_run(cypher_stmt, params, driver=driver)
    except Exception as exc:
        log.error("Neo4j keyword query failed: %s", exc)
        return []

    if not records:
        return []

    return _records_to_hits(records)


def retrieve_neo4j(state: MutableMapping[str, Any]) -> List[Dict[str, Any]]:
    """
    Execute the Neo4j retriever and return raw hits.
    """
    ensure_state_shapes(state)
    query = get_query_text(state).strip()
    if not query:
        log.debug("Neo4j retriever skipped: empty query.")
        return []

    params = state.get("params") or {}
    try:
        limit = int(params.get("top_k", default_top_k()))
    except (TypeError, ValueError):
        limit = 8

    return get_neo4j_search_results(query, limit=limit)


def run_neo4j_search(
    state: MutableMapping[str, Any],
    *,
    query: Optional[str] = None,
    limit: int = 12,
    max_total: Optional[int] = None,
    dedupe: bool = True,
    source: str = "neo4j",
) -> List[EvidenceEntry]:
    ensure_state_shapes(state)
    actual_query = (query or get_query_text(state)).strip()
    if not actual_query:
        log.debug("Neo4j search skipped: empty query.")
        return []

    hits = get_neo4j_search_results(actual_query, limit=limit)
    if not hits:
        return []

    return merge_retrieval(
        state,
        source=source,
        hits=hits,
        limit=max_total,
        dedupe=dedupe,
    )


__all__ = ["get_neo4j_search_results", "run_neo4j_search", "retrieve_neo4j"]
