import os
import time
from functools import lru_cache
from typing import Any, Dict, List, MutableMapping, Optional

import requests
from flask import Flask, jsonify, request
from opensearchpy import OpenSearch

from dotenv import load_dotenv

from .utils import default_top_k, get_logger, getenv

load_dotenv()

logger = get_logger(__name__)
app = Flask(__name__)


@lru_cache(maxsize=1)
def _os_client() -> OpenSearch:
    node = getenv("OPENSEARCH_NODE")
    user = getenv("OPENSEARCH_USERNAME", required=False, default="")
    pwd = getenv("OPENSEARCH_PASSWORD", required=False, default="")
    use_ssl = node.lower().startswith("https")
    return OpenSearch(
        hosts=[node],
        http_auth=(user, pwd) if (user or pwd) else None,
        use_ssl=use_ssl,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        timeout=30,
        max_retries=2,
        retry_on_timeout=True,
    )


def _os_index() -> str:
    return getenv("OPENSEARCH_INDEX")


_LOGGED_EMBED_URL: Optional[str] = None


def _embedding_url() -> str:
    """The resolved embedding endpoint, logged ONCE at first use.

    Only the *unset* case was ever reported. A URL that is set but wrong — the state this
    repo shipped in, pointing at a decommissioned host — produced a connection error per
    query, buried among other logs, and semantic search silently returned nothing. "No
    semantic results" and "the embedder is unreachable" then look identical from the outside,
    and every recall number measured in that state understates the system.
    """
    global _LOGGED_EMBED_URL
    url = (os.getenv("FLASK_EMBEDDING_URL") or "").rstrip("/")
    if url != _LOGGED_EMBED_URL:
        logger.info("Embedding endpoint resolved to %s", url or "(unset)")
        _LOGGED_EMBED_URL = url
    return url


def _fetch_embedding_from_service(user_query: str) -> Optional[List[float]]:
    flask_url = _embedding_url()
    if not flask_url:
        logger.error("FLASK_EMBEDDING_URL environment variable not set.")
        return None

    try:
        response = requests.post(
            f"{flask_url}/get_embedding",
            json={"text": user_query},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        embedding = payload.get("embedding")
        if isinstance(embedding, list):
            return embedding
        logger.error("Embedding payload missing 'embedding': %s", payload)
    except Exception as exc:
        logger.error("Error getting embedding from Flask server: %s", exc)
    return None


def get_embedding(user_query: str) -> Optional[List[float]]:
    query = (user_query or "").strip()
    if not query:
        return None
    return _fetch_embedding_from_service(query)


@app.route("/get_embedding", methods=["POST"])
def get_embedding_endpoint():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

    embedding = get_embedding(text)
    if embedding is None:
        return jsonify({"error": "embedding service unavailable"}), 503
    return jsonify({"embedding": embedding})

def semantic_search(query: str, size: Optional[int] = None) -> List[Dict[str, Any]]:
    # None -> the shared retrieval window, resolved at call time (a literal default here
    # was a third independent window alongside the wrapper's 8 and the tools' 8).
    size = max(1, min(int(size or default_top_k()), 100))
    start_time = time.time()
    logger.info(f"Starting semantic search for query: {query}")

    embedding = get_embedding(query)
    if not embedding:
        logger.error("Failed to get embedding for the query")
        return []

    try:
        response = _os_client().search(
            index=_os_index(),
            body={
                "size": size,
                "query": {
                    "knn": {
                        "contents-embedding": {
                            "vector": embedding,
                            "k": size
                        }
                    }
                }
            }
        )
    except Exception as e:
        logger.error(f"OpenSearch query failed: {e}")
        return []

    results = []
    hits = response.get('hits', {}).get('hits', [])
    logger.info(f"Retrieved {len(hits)} hits from OpenSearch")

    for hit in hits:
        source = hit.get('_source', {})
        score = hit.get('_score', 0)
        doc_id = hit.get('_id', '')

        if 'inner_hits' in hit and 'pdf_chunk_hits' in hit['inner_hits']:
            chunk_hits = hit['inner_hits']['pdf_chunk_hits']['hits']['hits']
            if chunk_hits:
                chunk_source = chunk_hits[0].get('_source', {})
                source['pdf_chunk'] = {
                    'chunk_id': chunk_source.get('chunk_id'),
                    'text': chunk_source.get('text')
                }

        results.append({
            '_id': doc_id,
            '_score': score,
            '_source': source
        })

    end_time = time.time()
    logger.info(f"Semantic search completed in {end_time - start_time:.2f} seconds")
    return results


def retrieve_semantic(state: MutableMapping[str, Any]) -> List[Dict[str, Any]]:
    """
    Execute semantic retrieval using the query stored in the shared state.
    """
    ensure_state_shapes(state)
    query = get_query_text(state).strip()
    if not query:
        logger.debug("Semantic retriever skipped: empty query.")
        return []

    params = state.get("params") or {}
    try:
        size = int(params.get("top_k", default_top_k()))
    except (TypeError, ValueError):
        size = default_top_k()

    return semantic_search(query, size=size)


# --- State-aligned helper ---
from ..state import EvidenceEntry, ensure_state_shapes, get_query_text, merge_retrieval


def run_semantic_search(
    state: MutableMapping[str, Any],
    *,
    query: Optional[str] = None,
    limit: int = 12,
    max_total: Optional[int] = None,
    dedupe: bool = True,
    source: str = "semantic",
) -> List[EvidenceEntry]:
    ensure_state_shapes(state)
    actual_query = (query or get_query_text(state)).strip()
    if not actual_query:
        logger.debug("Semantic search skipped: empty query.")
        return []

    hits = semantic_search(actual_query, size=limit)
    if not hits:
        return []

    return merge_retrieval(
        state,
        source=source,
        hits=hits,
        limit=max_total,
        dedupe=dedupe,
    )


__all__ = ["semantic_search", "run_semantic_search", "retrieve_semantic"]
