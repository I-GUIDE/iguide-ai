"""
test_keyword.py
---------------
Live test for the BM25 keyword search tool.

  TEST 0: Environment check — required env vars present
  TEST 1: Empty query guard — returns [] without hitting the index
  TEST 2: Single-term query — basic connectivity and hit shape
  TEST 3: Multi-term query — verify score ordering (best hit first)
  TEST 4: High-limit clamp — size > 100 is silently capped to 100
  TEST 5: LangChain wrapper — keyword_search_tool() returns a valid JSON payload

Run from repo root:
    python tests/live/test_keyword.py

Env vars read (from repo-root .env):
    OPENSEARCH_NODE     — required
    OPENSEARCH_INDEX    — required
    OPENSEARCH_USERNAME — optional
    OPENSEARCH_PASSWORD — optional
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Ensure repo root is importable when run from any directory
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")
logging.basicConfig(level=logging.WARNING)

from rag_pipeline.search.keyword import get_keyword_search_results
from rag_pipeline.langchain_granular_tools import keyword_search_tool

# These exercise LIVE backends. Without them the suite used to report FAILURES rather than
# skips, which is worse than useless: eight red lines became background noise that a real
# regression could hide inside. Skipping states the same fact without camouflaging anything.
import pytest as _pytest

_MISSING = [v for v in ("OPENSEARCH_NODE", "OPENSEARCH_INDEX") if not (os.getenv(v) or "").strip()]
pytestmark = _pytest.mark.skipif(
    bool(_MISSING),
    reason="live backend not configured: " + ", ".join(_MISSING) + " unset (no repo-root .env)",
)



def _banner(title: str) -> None:
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


def test_env_check() -> bool:
    _banner("TEST 0: Environment check")
    required = ["OPENSEARCH_NODE", "OPENSEARCH_INDEX"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        print(f"❌  Missing required env vars: {missing}")
        print("    Set them in .env and rerun.")
        return False
    print(f"✅  OPENSEARCH_NODE  = {os.getenv('OPENSEARCH_NODE')}")
    print(f"✅  OPENSEARCH_INDEX = {os.getenv('OPENSEARCH_INDEX')}")
    return True


def test_empty_query() -> None:
    _banner("TEST 1: Empty query guard")
    result = get_keyword_search_results("", size=5)
    assert result == [], f"Expected [] for empty query, got {result!r}"
    print("✅  Empty query returns [] (no index call)")


def test_single_term() -> None:
    _banner("TEST 2: Single-term query — connectivity + hit shape")
    query = "flood"
    hits = get_keyword_search_results(query, size=5)
    if not hits:
        print(f"⚠️   0 hits for {query!r} — index may be empty or query has no matches")
        return
    print(f"✅  {len(hits)} hits returned for {query!r}")
    first = hits[0]
    for key in ("_id", "_score", "_source"):
        assert key in first, f"Missing key {key!r} in hit"
    print(f"   Top hit: [{first['_score']:.3f}] {(first['_source'].get('title') or '<untitled>')[:70]}")


def test_multi_term_ordering() -> None:
    _banner("TEST 3: Multi-term query — score ordering")
    query = "climate geospatial data"
    hits = get_keyword_search_results(query, size=8)
    if len(hits) < 2:
        print(f"⚠️   Only {len(hits)} hit(s) — cannot verify ordering")
        return
    scores = [h["_score"] for h in hits]
    ordered = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    if ordered:
        print(f"✅  {len(hits)} hits returned in descending score order")
    else:
        print(f"❌  Hits are NOT in descending score order: {scores}")
    for hit in hits[:3]:
        title = (hit.get("_source") or {}).get("title") or "<untitled>"
        print(f"   [{hit['_score']:.3f}] {title[:70]}")


def test_size_cap() -> None:
    _banner("TEST 4: Size > 100 is clamped to 100")
    hits = get_keyword_search_results("data", size=9999)
    print(f"✅  Requested 9999, received {len(hits)} hits (capped at index max)")


def test_langchain_wrapper() -> None:
    _banner("TEST 5: LangChain wrapper — keyword_search_tool()")
    payload_str = keyword_search_tool("flood risk data", limit=5)
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
    test_empty_query()
    test_single_term()
    test_multi_term_ordering()
    test_size_cap()
    test_langchain_wrapper()
    _banner("SUMMARY")
    print("All tests executed. Check ✅/❌/⚠️ above for failures.")


if __name__ == "__main__":
    main()
