"""READ-ONLY full export of the live I-GUIDE OpenSearch index.

Writes three artifacts to outputs/:
  - <index>.jsonl          : every doc as {"_id":..., "_source":...} (re-ingestable)
  - <index>.mapping.json   : index mappings + settings (to recreate the index)
  - <index>.manifest.csv   : id, resource-type, title, tags, has_spatial (browsable)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
# override=False: an explicitly exported variable must WIN over the file. With
# override=True a stale .env value (e.g. a decommissioned FLASK_EMBEDDING_URL)
# silently replaced whatever the caller set, making a per-run override impossible
# and the reason invisible.
from dotenv import load_dotenv
for _cand in (REPO / "rag_pipeline" / ".env", REPO / ".env.bak"):
    if _cand.exists():
        load_dotenv(dotenv_path=_cand)
        break

import os
from opensearchpy import OpenSearch, helpers

NODE = os.getenv("OPENSEARCH_NODE", "")
INDEX = os.getenv("OPENSEARCH_INDEX", "")
OUT = REPO / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def client() -> OpenSearch:
    return OpenSearch(hosts=[NODE],
                      http_auth=(os.getenv("OPENSEARCH_USERNAME", ""), os.getenv("OPENSEARCH_PASSWORD", "")),
                      use_ssl=NODE.lower().startswith("https"), verify_certs=False,
                      ssl_assert_hostname=False, ssl_show_warn=False, timeout=60, max_retries=3,
                      retry_on_timeout=True)


def iter_docs(c: OpenSearch):
    """All docs via scan (scroll); fall back to a single large search."""
    try:
        yield from helpers.scan(c, index=INDEX, query={"query": {"match_all": {}}},
                                preserve_order=False, size=500)
        return
    except Exception as exc:
        print(f"  scan failed ({type(exc).__name__}); falling back to size=10000 search")
        r = c.search(index=INDEX, body={"size": 10000, "query": {"match_all": {}}})
        yield from r.get("hits", {}).get("hits", [])


def main() -> int:
    c = client()
    total = c.count(index=INDEX).get("count")
    print(f"Index {INDEX!r}: {total} docs")

    jsonl_path = OUT / f"{INDEX}.jsonl"
    csv_path = OUT / f"{INDEX}.manifest.csv"
    map_path = OUT / f"{INDEX}.mapping.json"

    n = 0
    with jsonl_path.open("w", encoding="utf-8") as jf, csv_path.open("w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["id", "resource-type", "title", "tags", "has_spatial", "abstract"])
        for hit in iter_docs(c):
            src = hit.get("_source", {}) or {}
            jf.write(json.dumps({"_id": hit.get("_id"), "_source": src}, ensure_ascii=False) + "\n")
            tags = src.get("tags")
            tags = ";".join(tags) if isinstance(tags, list) else (tags or "")
            has_spatial = bool(src.get("spatial-bounding-box-geojson") or src.get("spatial-geometry-geojson")
                               or src.get("spatial-coverage"))
            abstract = " ".join(str(src.get("contents") or "").split())[:300]
            w.writerow([hit.get("_id"), src.get("resource-type") or src.get("element_type") or "",
                        src.get("title") or "", tags, has_spatial, abstract])
            n += 1

    # mapping + settings
    try:
        mapping = {"mappings": c.indices.get_mapping(index=INDEX),
                   "settings": c.indices.get_settings(index=INDEX)}
        map_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        map_path = None
        print(f"  mapping export failed: {type(exc).__name__}: {exc}")

    def mb(p): return f"{p.stat().st_size/1e6:.2f} MB"
    print(f"\nExported {n} docs (index reported {total}).")
    print(f"  docs     : {jsonl_path}  ({mb(jsonl_path)})")
    print(f"  manifest : {csv_path}  ({mb(csv_path)})")
    if map_path:
        print(f"  mapping  : {map_path}  ({mb(map_path)})")
    if n != total:
        print(f"  ! WARNING: exported {n} != reported {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
