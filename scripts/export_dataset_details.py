#!/usr/bin/env python3
"""Re-read every cached dataset file and export what extraction actually found.

    python scripts/export_dataset_details.py --out outputs/dataset_details.json

``outputs/extract_dataset.json`` records the OUTCOME per element (which stage it reached, how many
files, any error) but not the content — so "schema, CRS, bounds" never reaches a reader from there.
This walks the cached files, runs ``DataExtractor`` over each, and writes the extracted payload
plus the generated loader's signature.

Storage is redirected to a temporary root, so nothing here touches the real method library or the
local agent-KB store. That precaution is not theoretical: a probe of mine that set the wrong
environment variable wrote 12 duplicate units into the real registry.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Files in the cache that are the harness's own bookkeeping, not corpus datasets.
_SKIP_PREFIXES = ("_elements", "_meta")


def _is_dataset_file(path: Path) -> bool:
    if path.is_dir() or path.stat().st_size == 0:
        return False
    if path.name.startswith(_SKIP_PREFIXES):
        return False
    return path.suffix.lower() not in {".ipynb", ".pdf"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(REPO / ".corpus_cache"))
    ap.add_argument("--out", default=str(REPO / "outputs" / "dataset_details.json"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="dsexport_")
    os.environ["AGENT_FILE_STORAGE_ROOT"] = tmp
    os.environ["AGENT_KB_STORE_DIR"] = os.path.join(tmp, "kb")
    os.environ["AGENT_METHOD_LIBRARY_DIR"] = os.path.join(tmp, "lib")
    os.environ["AGENT_KB_BACKEND"] = "local"

    from dotenv import load_dotenv
    load_dotenv(REPO / ".env", override=False)

    from extractors.base import EMIT_LIBRARY, EMIT_OPENSEARCH, ExtractContext
    from extractors.data_extractor import DataExtractor

    # Titles, so a row reads as a dataset rather than as a filename. The listing cache is the only
    # source that works without Neo4j credentials.
    titles = {}
    listing = Path(args.cache) / "_elements_dataset.json"
    if listing.is_file():
        try:
            data = json.loads(listing.read_text())
            for el in (data if isinstance(data, list) else data.get("elements") or []):
                titles[str(el.get("id"))[:8]] = str(el.get("title") or "")
        except (OSError, ValueError):
            pass

    paths = sorted(p for p in Path(args.cache).iterdir() if _is_dataset_file(p))
    if args.limit:
        paths = paths[: args.limit]

    rows = []
    for path in paths:
        short = path.name.split("__", 1)[0]
        title = titles.get(short) or path.name
        ctx = ExtractContext(element_id=short, element_type="dataset",
                             fields={"title": title},
                             targets=[EMIT_OPENSEARCH, EMIT_LIBRARY])
        row = {"element": short, "file": path.name, "title": title,
               "bytes": path.stat().st_size}
        try:
            result = DataExtractor().extract(str(path), ctx=ctx)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"[:200]
            rows.append(row)
            print(f"  !! {path.name[:52]:<54}{row['error'][:60]}")
            continue
        for asset in result.assets:
            if asset.kind == "dataset":
                row["extracted"] = {k: v for k, v in (asset.extracted or {}).items()
                                    if v not in (None, "", [], {})}
                envelope = (asset.spatial or {}).get("spatial-bounding-box-geojson")
                if envelope:
                    row["envelope"] = envelope
            elif asset.kind == "method_unit" and asset.unit:
                row["loader"] = {"signature": asset.unit.get("signature"),
                                 "summary": asset.unit.get("doc_summary"),
                                 "returns": asset.unit.get("returns"),
                                 "invariants": [i.get("check") for i
                                                in (asset.unit.get("invariants") or [])],
                                 "requires": (asset.unit.get("requirements") or {}).get("pip")}
        ex = row.get("extracted") or {}
        print(f"  ok {path.name[:52]:<54}{str(ex.get('format') or '?'):<10}"
              f"{len(ex.get('schema') or [])} field(s)")
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cache": args.cache, "count": len(rows), "rows": rows},
                              indent=2, default=str) + "\n")
    described = sum(1 for r in rows if (r.get("extracted") or {}).get("schema")
                    or (r.get("extracted") or {}).get("crs")
                    or (r.get("extracted") or {}).get("variables"))
    print(f"\n{len(rows)} cached dataset file(s); {described} yielded a schema, CRS or variables")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
