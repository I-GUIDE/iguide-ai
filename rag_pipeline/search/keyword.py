from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, MutableMapping, Optional

from opensearchpy import OpenSearch

from .utils import default_top_k, get_logger, getenv, normalize_source_fields, safe_score
from ..state import EvidenceEntry, ensure_state_shapes, get_query_text, merge_retrieval

log = get_logger("search_keyword")


@lru_cache(maxsize=1)
def _os_client() -> OpenSearch:
    node = getenv("OPENSEARCH_NODE")
    user = getenv("OPENSEARCH_USERNAME", required=False, default="")
    pwd = getenv("OPENSEARCH_PASSWORD", required=False, default="")
    return OpenSearch(
        hosts=[node],
        http_auth=(user, pwd) if (user or pwd) else None,
        use_ssl=node.lower().startswith("https"),
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        timeout=30,
        max_retries=2,
        retry_on_timeout=True,
    )


def _os_index(default: Optional[str] = None) -> str:
    index = default or getenv("OPENSEARCH_INDEX")
    if not index:
        raise RuntimeError("OPENSEARCH_INDEX must not be empty")
    return index


def get_keyword_search_results(
    user_query: str,
    size: Optional[int] = None,
    *,
    client: Optional[OpenSearch] = None,
    index: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Perform a keyword search on the OpenSearch index.

    Environment variables used when `client`/`index` are not supplied:
    - OPENSEARCH_NODE
    - OPENSEARCH_USERNAME / OPENSEARCH_PASSWORD (optional)
    - OPENSEARCH_INDEX

    The optional `client` parameter allows dependency injection for testing.
    """
    query = (user_query or "").strip()
    if not query:
        return []

    # None (or 0) -> the shared retrieval window, resolved at call time. This signature
    # default used to be a hardcoded 12, a third independent window alongside the
    # wrapper's 8 and the tools' 8.
    size = max(1, min(int(size or default_top_k()), 100))
    body = {"size": size, "query": {"match": {"contents": query}}}

    os_client = client or _os_client()
    os_index = _os_index(index)

    try:
        response = os_client.search(index=os_index, body=body)
        payload = response if isinstance(response, dict) else response.body
        hits = payload.get("hits", {}).get("hits", []) or []
    except Exception as exc:
        log.error("OpenSearch keyword query failed: %s", exc)
        return []

    results: List[Dict[str, Any]] = []
    for hit in hits:
        hit_id = str(hit.get("_id", ""))
        source = normalize_source_fields(hit.get("_source", {}) or {}, hit_id)
        results.append(
            {
                "_id": hit_id,
                "_score": safe_score(hit.get("_score", 1.0)),
                "_source": source,
            }
        )
    return results


def retrieve_keyword(state: MutableMapping[str, Any]) -> List[Dict[str, Any]]:
    """
    Execute the keyword retriever and return raw hits for downstream normalization.
    """
    ensure_state_shapes(state)
    query = get_query_text(state).strip()
    if not query:
        log.debug("Keyword retriever skipped: empty query.")
        return []

    params = state.get("params") or {}
    try:
        size = int(params.get("top_k", default_top_k()))
    except (TypeError, ValueError):
        size = default_top_k()

    return get_keyword_search_results(query, size=size)


def run_keyword_search(
    state: MutableMapping[str, Any],
    *,
    query: Optional[str] = None,
    limit: int = 12,
    max_total: Optional[int] = None,
    dedupe: bool = True,
    source: str = "keyword",
) -> List[EvidenceEntry]:
    ensure_state_shapes(state)
    actual_query = (query or get_query_text(state)).strip()
    if not actual_query:
        log.debug("Keyword search skipped: empty query.")
        return []

    hits = get_keyword_search_results(actual_query, size=limit)
    if not hits:
        return []

    return merge_retrieval(
        state,
        source=source,
        hits=hits,
        limit=max_total,
        dedupe=dedupe,
    )


__all__ = ["get_keyword_search_results", "run_keyword_search", "retrieve_keyword"]
