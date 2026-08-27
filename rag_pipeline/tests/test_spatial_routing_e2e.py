#!/usr/bin/env python3
"""
End-to-end test for spatial routing → retrieval → generation path.

This test validates the complete flow:
    routing → spatial_search.retrieve_spatial → merge_retrieval → generation

⚠️  LIVE TESTING: This test uses REAL APIs:
    - Google Maps API for geocoding (spatial search)
    - OpenSearch for spatial queries
    - AnvilGPT API for LLM generation

Run with:
    pytest -q rag_pipeline/tests/test_spatial_routing_e2e.py
    
or:
    PYTHONPATH=. python -m pytest rag_pipeline/tests/test_spatial_routing_e2e.py -v

Requirements:
    - GOOGLE_MAPS_API_KEY must be set in .env.local
    - OPENSEARCH_NODE must be configured
    - ANVILGPT_URL and ANVILGPT_KEY must be set for generation
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, MutableMapping

import pytest
from dotenv import load_dotenv

from rag_pipeline.routing import rag_pipeline
from rag_pipeline.state import ensure_state_shapes

# Load environment variables for OpenSearch/Google Maps configuration
_env_path = Path(__file__).parent.parent / ".env.local"
if _env_path.exists():
    load_dotenv(_env_path)


def make_state(query: str) -> MutableMapping[str, Any]:
    """Build a valid AgentState using the canonical shape from state.py"""
    state: Dict[str, Any] = {
        "query_information": {"raw_text": query},
        "session_context": {"use_spatial": True},  # Enable spatial routing
        "params": {"top_k": 5},
        "evidence": {"retrieved_documents": [], "sources": {}},
        "answer": {"final_composed_answer": None, "citations": [], "confidence_score": None},
        "planner_reasoning": {},
        "safety_checks": {},
        "trace_observability": {},
    }
    return ensure_state_shapes(state)




# Live-service test: needs a reachable OpenSearch cluster, the embedding server and
# Neo4j. Marked `integration` so a clean `pytest` on a fresh clone is GREEN — these
# three were the only red in the suite, and a suite that is red by default trains
# everyone to ignore it, which is how a real regression gets through.
# Run them with: pytest -m integration
@pytest.mark.integration
def test_spatial_routing_to_generation_e2e(monkeypatch):
    """
    Validate that spatial routing flows correctly through retrieval and generation.
    
    This test:
    1. Forces routing to select ONLY spatial search (suppresses keyword/semantic)
    2. Uses LIVE spatial retriever with Google Maps API and OpenSearch
    3. Validates evidence structure after merge_retrieval
    4. Uses LIVE AnvilGPT API for generation with real LLM calls
    
    ⚠️ Requires valid API keys in .env.local:
       - GOOGLE_MAPS_API_KEY (for spatial geocoding)
       - ANVILGPT_URL and ANVILGPT_KEY (for generation)
    """
    # Query with strong spatial intent (Hagerstown, MD location)
    query = ("Find flood risk map layers within 10 km of Hagerstown, Washington County, MD; "
             "include bounding boxes and data source.")
    
    # === FORCE SPATIAL-ONLY ROUTING ===
    # Monkeypatch keyword and semantic to return empty results
    # This isolates the spatial path without changing production code
    
    # Patch where the functions are USED, not where they are defined: core.py does
    # `from .keyword import retrieve_keyword`, which binds the function into core's own
    # namespace at import time. Patching the attribute on the defining module afterwards
    # leaves that binding pointing at the real implementation, so keyword hits kept flowing
    # through and this test asserted "all docs are spatial" against a mixed result set.
    # Every NON-spatial source has to be suppressed, not just the two that existed when this
    # test was written: core.py now also runs neo4j and opengeodata, and either one leaking a
    # hit breaks the "all docs are spatial" assertion.
    for symbol in ("retrieve_keyword", "retrieve_semantic", "retrieve_neo4j",
                   "retrieve_opengeodata"):
        monkeypatch.setattr(f"rag_pipeline.search.core.{symbol}",
                            lambda state, _s=symbol: (
                                print(f"   [Test] Suppressing {_s} to isolate spatial path") or []))
    # The graph tier that actually runs is get_neo4j_agent_results (core.py:160);
    # retrieve_neo4j above is only the keyword fallback beneath it, and is never reached when
    # the agent is available. It takes (query, limit=...), not (state).
    monkeypatch.setattr("rag_pipeline.search.core.get_neo4j_agent_results",
                        lambda *a, **k: (
                            print("   [Test] Suppressing neo4j agent to isolate spatial path")
                            or []))
    
    # === RUN PIPELINE WITH LIVE APIS ===
    # - Spatial search will use real Google Maps API for geocoding
    # - Generation will use real AnvilGPT API for LLM calls
    
    print(f"\n   [Test] Running LIVE test with real APIs:")
    print(f"   [Test] - Google Maps API (spatial geocoding)")
    print(f"   [Test] - OpenSearch (spatial queries)")
    print(f"   [Test] - AnvilGPT API (LLM generation)")
    
    state = make_state(query)
    out = rag_pipeline(state)
    
    # === ASSERTIONS ===
    
    # 1. Evidence was retrieved from spatial source
    evidence = out["evidence"]
    docs = evidence["retrieved_documents"]
    assert len(docs) >= 1, f"Expected at least 1 spatial document, got {len(docs)}"
    
    # 2. All docs are from spatial source (since we suppressed others)
    for doc in docs:
        assert doc["source"] == "spatial", f"Expected spatial source, got {doc['source']}"
    
    # 3. Evidence sources tracking
    assert "spatial" in evidence["sources"], "spatial source not tracked in evidence.sources"
    assert evidence["sources"]["spatial"]["cumulative"] >= 1, (
        f"Expected spatial cumulative >= 1, got {evidence['sources']['spatial']}"
    )
    
    # 4. Answer was generated by REAL LLM
    answer = out["answer"]
    final_answer = answer["final_composed_answer"]
    assert final_answer, "Expected non-empty answer from generation"
    assert isinstance(final_answer, str), f"Answer should be string, got {type(final_answer)}"
    assert len(final_answer) > 20, f"Answer seems too short ({len(final_answer)} chars): {final_answer[:100]}"
    
    # 5. Citations exist and reference spatial docs
    citations = answer["citations"]
    assert len(citations) >= 1, f"Expected at least 1 citation in answer, got {len(citations)}"
    
    cited_doc_ids = {c["doc_id"] for c in citations}
    retrieved_doc_ids = {d["document"]["doc_id"] for d in docs}
    
    # Citations should reference retrieved docs (some LLMs might not cite all)
    assert len(cited_doc_ids) > 0, "No citations found in answer"
    
    # 6. All citations are from spatial source
    for citation in citations:
        assert citation["source"] == "spatial", (
            f"Citation source should be spatial, got {citation['source']}"
        )
    
    # 7. Validate document structure conforms to EvidenceEntry schema
    for doc in docs:
        assert "document" in doc, "Missing document field in EvidenceEntry"
        assert "doc_id" in doc["document"], "Missing doc_id in DocumentPayload"
        assert "title" in doc["document"], "Missing title in DocumentPayload"
        assert "contents" in doc["document"], "Missing contents in DocumentPayload"
        assert "score" in doc, "Missing score in EvidenceEntry"
        assert "retrieval_rank" in doc, "Missing retrieval_rank in EvidenceEntry"
        assert "metadata" in doc, "Missing metadata in EvidenceEntry"
    
    # === SUCCESS OUTPUT ===
    
    print(f"\n✅ Spatial routing E2E test PASSED with LIVE APIs!")
    print(f"   Query: {query[:60]}...")
    print(f"   Retrieved: {len(docs)} spatial documents from OpenSearch")
    print(f"   Top doc: {docs[0]['document']['title'][:60]}")
    print(f"   Generated: {len(final_answer)} char answer from AnvilGPT")
    print(f"   Citations: {len(citations)} docs cited")
    print(f"\n   First 200 chars of answer:")
    print(f"   {final_answer[:200]}...")
    
    # Verify spatial metadata is present
    first_doc = docs[0]
    if "spatial_metadata" in first_doc["metadata"]:
        print(f"\n   ✓ Spatial metadata confirmed (Maps API used for geocoding)")
    if "spatial-bounding-box-geojson" in first_doc["document"]:
        print(f"   ✓ Bounding box data present")




if __name__ == "__main__":
    # Allow running directly for debugging
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

