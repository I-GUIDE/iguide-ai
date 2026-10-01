"""
test_semantic.py
----------------
Live test for the dense-vector (kNN) semantic search tool.

  TEST 0: Environment check — required env vars present
  TEST 1: Embedding service reachability — POST /get_embedding returns a vector
  TEST 2: Vector dimensionality — embedding matches index expectations
  TEST 3: Single-query semantic search — basic connectivity and hit shape
  TEST 4: Semantic vs keyword divergence — semantic can surface different docs
  TEST 5: LangChain wrapper — semantic_search_tool() returns a valid JSON payload

Run from repo root:
    python tests/live/test_semantic.py

Env vars read (from repo-root .env):
    FLASK_EMBEDDING_URL — required (e.g. http://localhost:5001)
    OPENSEARCH_NODE     — required
    OPENSEARCH_INDEX    — required
    OPENSEARCH_USERNAME — optional
    OPENSEARCH_PASSWORD — optional
"""
from __future__ import annotations

import json
import logging
import os
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")
logging.basicConfig(level=logging.WARNING)

import requests

from rag_pipeline.search.semantic import get_embedding, semantic_search
from rag_pipeline.langchain_granular_tools import semantic_search_tool


def _banner(title: str) -> None:
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


def test_env_check() -> bool:
    _banner("TEST 0: Environment check")
    required = ["FLASK_EMBEDDING_URL", "OPENSEARCH_NODE", "OPENSEARCH_INDEX"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        print(f"❌  Missing required env vars: {missing}")
        print("    Set them in .env and rerun.")
        return False
    print(f"✅  FLASK_EMBEDDING_URL = {os.getenv('FLASK_EMBEDDING_URL')}")
    print(f"✅  OPENSEARCH_NODE     = {os.getenv('OPENSEARCH_NODE')}")
    print(f"✅  OPENSEARCH_INDEX    = {os.getenv('OPENSEARCH_INDEX')}")
    return True


def test_embedding_service() -> bool:
    _banner("TEST 1: Embedding service reachability")
    embedding_url = (os.getenv("FLASK_EMBEDDING_URL") or "").rstrip("/")
    try:
        resp = requests.post(
            f"{embedding_url}/get_embedding",
            json={"text": "geospatial flood risk"},
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        print(f"❌  Cannot reach embedding service: {exc}")
        return False

    embedding = payload.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        print(f"❌  Unexpected response shape: {payload}")
        return False

    print(f"✅  Embedding service returned vector of dim={len(embedding)}")
    return True


def test_vector_dimensionality() -> None:
    _banner("TEST 2: Vector dimensionality sanity check")
    vec = get_embedding("climate change dataset")
    if vec is None:
        print("⚠️   get_embedding() returned None — service may be down")
        return
    dim = len(vec)
    if dim in (384, 768, 1024, 1536, 3072):
        print(f"✅  Embedding dim={dim} matches a known model dimension")
    else:
        print(f"⚠️   Embedding dim={dim} — verify this matches the OpenSearch index mapping")


def test_single_query() -> None:
    _banner("TEST 3: Single-query semantic search")
    query = "flood risk geospatial analysis"
    hits = semantic_search(query, size=5)
    if not hits:
        print(f"⚠️   0 hits for {query!r} — kNN index may be empty or embedding failed")
        return
    print(f"✅  {len(hits)} hits returned for {query!r}")
    first = hits[0]
    for key in ("_id", "_score", "_source"):
        assert key in first, f"Missing key {key!r} in hit"
    title = (first.get("_source") or {}).get("title") or "<untitled>"
    print(f"   Top hit: [{first['_score']:.4f}] {title[:70]}")
    for hit in hits[:3]:
        t = (hit.get("_source") or {}).get("title") or "<untitled>"
        print(f"   [{hit['_score']:.4f}] {t[:70]}")


# The only test here that needs the OPENSEARCH keyword path; the rest exercise the embedding
# service and pass without it, so this guard is per-test rather than module-level.
@pytest.mark.skipif(not (os.getenv("OPENSEARCH_NODE") or "").strip(),
                    reason="live backend not configured: OPENSEARCH_NODE unset")
def test_semantic_vs_keyword_divergence() -> None:
    _banner("TEST 4: Semantic vs keyword divergence")
    from rag_pipeline.search.keyword import get_keyword_search_results

    query = "water resource management irrigation"
    sem_hits = semantic_search(query, size=5)
    kw_hits = get_keyword_search_results(query, size=5)

    sem_ids = {h.get("_id") for h in sem_hits}
    kw_ids = {h.get("_id") for h in kw_hits}
    shared = sem_ids & kw_ids
    only_sem = sem_ids - kw_ids

    print(f"   Semantic hits: {len(sem_hits)}   Keyword hits: {len(kw_hits)}")
    print(f"   Shared docs: {len(shared)}   Semantic-only docs: {len(only_sem)}")
    if only_sem:
        print("✅  Semantic search surfaces docs that keyword search misses (expected)")
    else:
        print("⚠️   No semantic-only docs found — results may overlap fully or index is small")


def test_langchain_wrapper() -> None:
    _banner("TEST 5: LangChain wrapper — semantic_search_tool()")
    payload_str = semantic_search_tool("climate dataset Illinois", limit=5)
    try:
        payload = json.loads(payload_str)
    except json.JSONDecodeError as exc:
        print(f"❌  Wrapper returned non-JSON: {exc}")
        return
    expected_keys = {"source", "count", "documents", "citation_ids"}
    missing = expected_keys - set(payload.keys())
    if missing:
        print(f"❌  Missing keys in payload: {missing}")
        return
    print(f"✅  Payload shape OK — source={payload['source']} count={payload['count']}")
    if payload["documents"]:
        print(f"   First doc: {payload['documents'][0].get('title', '<untitled>')[:70]}")


def main() -> None:
    if not test_env_check():
        return
    embedding_ok = test_embedding_service()
    if not embedding_ok:
        print("\n⚠️   Skipping tests that require the embedding service.")
        return
    test_vector_dimensionality()
    test_single_query()
    test_semantic_vs_keyword_divergence()
    test_langchain_wrapper()
    _banner("SUMMARY")
    print("All tests executed. Check ✅/❌/⚠️ above for failures.")


if __name__ == "__main__":
    main()
