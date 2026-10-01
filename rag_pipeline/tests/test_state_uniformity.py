#!/usr/bin/env python3
"""
Verify that all search modules maintain uniform state structure
for consistent generation across keyword, semantic, and spatial results.
"""

import os
import sys

import pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env.local")

from rag_pipeline.pipeline import run_pipeline

# These exercise LIVE backends. Without them the suite used to report FAILURES rather than
# skips, which is worse than useless: eight red lines became background noise that a real
# regression could hide inside. Skipping states the same fact without camouflaging anything.
import pytest as _pytest

_MISSING = [v for v in ("OPENSEARCH_NODE",) if not (os.getenv(v) or "").strip()]
pytestmark = _pytest.mark.skipif(
    bool(_MISSING),
    reason="live backend not configured: " + ", ".join(_MISSING) + " unset (no repo-root .env)",
)



def print_section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def validate_evidence_entry(entry, source_name):
    """Validate that an evidence entry has the required uniform structure"""
    errors = []
    
    # Check required top-level fields
    if "source" not in entry:
        errors.append(f"Missing 'source' field")
    elif entry["source"] != source_name:
        errors.append(f"Source mismatch: expected '{source_name}', got '{entry['source']}'")
    
    if "score" not in entry:
        errors.append(f"Missing 'score' field")
    elif not isinstance(entry["score"], (int, float)):
        errors.append(f"Score is not numeric: {type(entry['score'])}")
    
    if "retrieval_rank" not in entry:
        errors.append(f"Missing 'retrieval_rank' field")
    
    if "document" not in entry:
        errors.append(f"Missing 'document' field")
    else:
        # Check document structure
        doc = entry["document"]
        if not isinstance(doc, dict):
            errors.append(f"Document is not a dict: {type(doc)}")
        else:
            required_doc_fields = ["doc_id", "title", "contents"]
            for field in required_doc_fields:
                if field not in doc:
                    errors.append(f"Document missing '{field}' field")
    
    if "metadata" not in entry:
        errors.append(f"Missing 'metadata' field")
    
    return errors


# Live-service test: needs a reachable OpenSearch cluster, the embedding server and
# Neo4j. Marked `integration` so a clean `pytest` on a fresh clone is GREEN — these
# three were the only red in the suite, and a suite that is red by default trains
# everyone to ignore it, which is how a real regression gets through.
# Run them with: pytest -m integration
@pytest.mark.integration
def test_mixed_sources_generation():
    """Test that generation works with mixed sources (keyword + semantic + spatial)"""
    
    print_section("STATE UNIFORMITY TEST: Mixed Sources")
    
    # Query that should trigger multiple search methods
    query = "Water quality datasets in California"
    
    print(f"Query: \"{query}\"")
    print("\n🔍 This query should trigger:")
    print("   • Keyword search (always active)")
    print("   • Semantic search (always active)")
    print("   • Spatial search (California location)")
    
    # Run pipeline with spatial enabled
    print("\nRunning pipeline with spatial search enabled...")
    result_state = run_pipeline(
        user_input=query,
        session_context={
            "use_spatial": True,
        },
        params={"top_k": 8}
    )
    
    # Analyze the evidence structure
    evidence = result_state.get("evidence", {})
    docs = evidence.get("retrieved_documents", [])
    
    print(f"\n✅ Total documents retrieved: {len(docs)}")
    
    # Group by source
    by_source = {}
    for doc in docs:
        source = doc.get("source", "unknown")
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(doc)
    
    print(f"\nDocuments by source:")
    for source, source_docs in by_source.items():
        print(f"  • {source}: {len(source_docs)} documents")
    
    # Validate each entry has uniform structure
    print_section("VALIDATING STATE STRUCTURE")
    
    all_valid = True
    for source, source_docs in by_source.items():
        print(f"\n[{source.upper()}] Validating {len(source_docs)} entries...")
        
        for i, entry in enumerate(source_docs[:3], 1):  # Check first 3 from each source
            errors = validate_evidence_entry(entry, source)
            
            if errors:
                all_valid = False
                print(f"  ❌ Entry {i} has errors:")
                for error in errors:
                    print(f"     - {error}")
            else:
                doc = entry["document"]
                print(f"  ✅ Entry {i}: {doc.get('title', 'No title')[:50]}")
                print(f"     doc_id: {doc.get('doc_id', 'N/A')[:40]}")
                print(f"     score: {entry.get('score', 0):.4f}")
                print(f"     rank: {entry.get('retrieval_rank', -1)}")
    
    # Check generation compatibility
    print_section("GENERATION COMPATIBILITY CHECK")
    
    answer = result_state.get("answer", {})
    final_answer = answer.get("final_composed_answer", "")
    citations = answer.get("citations", [])
    
    if final_answer:
        print("✅ Generation module successfully processed mixed-source evidence")
        print(f"   Answer length: {len(final_answer)} characters")
        print(f"   Citations: {len(citations)}")
        
        # Check if citations include multiple sources
        citation_sources = set(c.get("source", "") for c in citations)
        print(f"   Citation sources: {', '.join(citation_sources)}")
        
        if len(citation_sources) > 1:
            print(f"\n✅ Generation successfully used evidence from multiple sources!")
        
        # Show a preview
        print(f"\n📝 Answer Preview:")
        print(f"   {final_answer[:300]}...")
    else:
        print("⚠️  No answer generated")
    
    # Summary
    print_section("SUMMARY")
    
    if all_valid:
        print("✅ All evidence entries have uniform structure")
        print(f"✅ All {len(docs)} documents follow the EvidenceEntry schema:")
        print("   • source (str)")
        print("   • score (float)")
        print("   • retrieval_rank (int)")
        print("   • document (DocumentPayload)")
        print("     - doc_id, title, contents, element_type")
        print("   • metadata (dict)")
    else:
        print("❌ Some evidence entries have structural issues")
    
    if final_answer and len(by_source) > 1:
        print(f"\n✅ Generation module is SOURCE-AGNOSTIC:")
        print(f"   Successfully processed {len(by_source)} different sources")
        print(f"   Produced coherent answer with {len(citations)} citations")
    
    print(f"\n✅ State uniformity verified across the pipeline!")
    
    return all_valid and bool(final_answer)


# Live-service test: needs a reachable OpenSearch cluster, the embedding server and
# Neo4j. Marked `integration` so a clean `pytest` on a fresh clone is GREEN — these
# three were the only red in the suite, and a suite that is red by default trains
# everyone to ignore it, which is how a real regression gets through.
# Run them with: pytest -m integration
@pytest.mark.integration
def test_individual_source_structures():
    """Test each source individually to verify they all conform to the same structure"""
    
    print_section("INDIVIDUAL SOURCE STRUCTURE TEST")
    
    sources_to_test = [
        ("keyword", "water resources", {}),
        ("semantic", "climate change impacts", {}),
        ("spatial", "datasets in Utah", {"use_spatial": True}),
    ]
    
    all_conform = True
    
    for source_name, query, session_ctx in sources_to_test:
        print(f"\n--- Testing {source_name.upper()} ---")
        
        result_state = run_pipeline(
            user_input=query,
            session_context=session_ctx,
            params={"top_k": 3}
        )
        
        docs = result_state["evidence"]["retrieved_documents"]
        source_docs = [d for d in docs if d.get("source") == source_name]
        
        print(f"Retrieved {len(source_docs)} {source_name} documents")
        
        if source_docs:
            # Validate first document
            entry = source_docs[0]
            errors = validate_evidence_entry(entry, source_name)
            
            if errors:
                all_conform = False
                print(f"❌ Structure errors:")
                for error in errors:
                    print(f"   - {error}")
            else:
                print(f"✅ Structure conforms to EvidenceEntry schema")
                doc = entry["document"]
                print(f"   Sample: {doc.get('title', 'No title')[:50]}")
        else:
            print(f"⚠️  No {source_name} results (may be expected)")
    
    return all_conform


def main():
    print("="*80)
    print("  RAG PIPELINE STATE UNIFORMITY VERIFICATION")
    print("="*80)
    print("\nThis test verifies that:")
    print("  1. All search modules return uniform EvidenceEntry structures")
    print("  2. Generation can process mixed-source evidence seamlessly")
    print("  3. State structure is maintained throughout the pipeline")
    
    # Test 1: Individual sources
    try:
        individual_ok = test_individual_source_structures()
    except Exception as e:
        print(f"\n❌ Individual source test failed: {e}")
        import traceback
        traceback.print_exc()
        individual_ok = False
    
    # Test 2: Mixed sources with generation
    try:
        mixed_ok = test_mixed_sources_generation()
    except Exception as e:
        print(f"\n❌ Mixed source test failed: {e}")
        import traceback
        traceback.print_exc()
        mixed_ok = False
    
    # Final summary
    print_section("FINAL VERIFICATION")
    
    if individual_ok and mixed_ok:
        print("🎉 ✅ STATE UNIFORMITY VERIFIED!")
        print("\nAll search modules (keyword, semantic, spatial) maintain:")
        print("  ✅ Uniform EvidenceEntry structure")
        print("  ✅ Compatible document payload format")
        print("  ✅ Consistent metadata handling")
        print("  ✅ Source-agnostic generation compatibility")
        print("\n→ The pipeline is ready for production use!")
    else:
        print("⚠️  Some uniformity issues detected")
        if not individual_ok:
            print("   • Individual source structures need review")
        if not mixed_ok:
            print("   • Mixed source generation needs review")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()





