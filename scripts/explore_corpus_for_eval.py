"""READ-ONLY exploration of the live I-GUIDE Smart Search index.

Purpose: characterize the public catalog and surface candidate knowledge
elements for the GeoPathfinder evaluation table (anchor notebook, composable
complements, distractors). Performs only .count()/.search() — no writes.
"""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# override=False: an explicitly exported variable must WIN over the file. With
# override=True a stale .env value (e.g. a decommissioned FLASK_EMBEDDING_URL)
# silently replaced whatever the caller set, making a per-run override impossible
# and the reason invisible.
from dotenv import load_dotenv
_repo = Path(__file__).resolve().parents[1]
# .env (root) has stale creds; rag_pipeline/.env + .env.bak authenticate.
for _cand in (_repo / "rag_pipeline" / ".env", _repo / ".env.bak", _repo / ".env"):
    if _cand.exists():
        load_dotenv(dotenv_path=_cand)
        break

import os
from opensearchpy import OpenSearch

NODE = os.getenv("OPENSEARCH_NODE", "")
INDEX = os.getenv("OPENSEARCH_INDEX", "")
USER = os.getenv("OPENSEARCH_USERNAME", "")
PWD = os.getenv("OPENSEARCH_PASSWORD", "")


def client() -> OpenSearch:
    return OpenSearch(
        hosts=[NODE],
        http_auth=(USER, PWD) if (USER or PWD) else None,
        use_ssl=NODE.lower().startswith("https"),
        verify_certs=False, ssl_assert_hostname=False, ssl_show_warn=False,
        timeout=30, max_retries=2, retry_on_timeout=True,
    )


def snippet(src: dict, n: int = 200) -> str:
    for k in ("abstract", "contents", "description"):
        v = src.get(k)
        if v:
            return " ".join(str(v).split())[:n]
    return ""


def rtype(src: dict) -> str:
    return str(src.get("resource-type") or src.get("element_type") or src.get("resourceType") or "?")


def main() -> None:
    host = urlparse(NODE).hostname or "<unset>"
    print(f"Connecting to host={host}  index={INDEX!r}")
    c = client()
    info_ok = False
    try:
        total = c.count(index=INDEX).get("count")
        print(f"OK — total docs in index: {total}")
        info_ok = True
    except Exception as exc:
        print(f"CONNECT/COUNT FAILED: {type(exc).__name__}: {exc}")
        return

    # --- schema peek: keys on a sample doc ---
    try:
        sample = c.search(index=INDEX, body={"size": 1, "query": {"match_all": {}}})
        hits = sample.get("hits", {}).get("hits", [])
        if hits:
            print("\nSample _source keys:", sorted((hits[0].get("_source") or {}).keys()))
    except Exception as exc:
        print(f"(sample fetch failed: {exc})")

    # --- resource-type distribution (try agg, fall back to sampling) ---
    print("\n== resource-type distribution ==")
    got_agg = False
    for field in ("resource-type.keyword", "resource-type", "element_type.keyword", "element_type"):
        try:
            agg = c.search(index=INDEX, body={
                "size": 0,
                "aggs": {"types": {"terms": {"field": field, "size": 30}}},
            })
            buckets = agg.get("aggregations", {}).get("types", {}).get("buckets", [])
            if buckets:
                for b in buckets:
                    print(f"  {b['key']:<28} {b['doc_count']}")
                got_agg = True
                break
        except Exception:
            continue
    if not got_agg:
        print("  (no keyword agg available; sampling 200 docs instead)")
        s = c.search(index=INDEX, body={"size": 200, "query": {"match_all": {}}})
        from collections import Counter
        cnt = Counter(rtype(h.get("_source") or {}) for h in s.get("hits", {}).get("hits", []))
        for k, v in cnt.most_common():
            print(f"  {k:<28} {v}  (of 200 sampled)")

    # --- targeted searches for candidate eval elements ---
    queries = {
        "ANCHOR: chicago crime": "chicago crime",
        "chicago (geography/boundaries)": "chicago community area boundary",
        "crime (other regions = distractors)": "crime incidents",
        "flood / flooding": "flood risk inundation",
        "social vulnerability / svi": "social vulnerability index",
        "census / demographics": "census tract demographics population",
        "urban heat": "urban heat island temperature",
        "land cover / land use": "land cover land use classification",
        "notebook + geopandas": "jupyter notebook geopandas spatial analysis",
        "redlining / equity": "redlining housing equity",
    }
    for label, q in queries.items():
        print(f"\n== {label}  (q={q!r}) ==")
        try:
            r = c.search(index=INDEX, body={"size": 5, "query": {"multi_match": {
                "query": q, "fields": ["title^3", "abstract^2", "contents", "tags"]}}})
            hh = r.get("hits", {}).get("hits", [])
            if not hh:
                print("  (no hits)")
                continue
            for h in hh:
                src = h.get("_source") or {}
                title = (src.get("title") or "<untitled>")
                print(f"  [{h.get('_score', 0):.2f}] ({rtype(src)}) {title[:90]}")
                print(f"        id={h.get('_id')}  {snippet(src, 160)}")
        except Exception as exc:
            print(f"  query failed: {exc}")


if __name__ == "__main__":
    main()
