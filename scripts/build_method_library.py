"""Fetch the notebook corpus and build the method library from it.

M3: the library was 16 units from 14 locally-cached notebooks. This drives the real path —
platform API → :mod:`extractors.sources` → notebook extractor → callability analyzer → slice →
``iguide_methods`` package — over every element the platform declares.

Two properties this script exists to preserve:

* **Resumable.** Fetched sources are cached by element id and re-used, so a rerun after a
  network blip does not re-download 174 notebooks.
* **Honest about failure.** Every element lands in exactly one bucket (fetched / unfetchable /
  unparseable / no units) and the counts are printed. "The library has N units" means nothing
  without "…out of M elements, and here is where the other M-N went".

Usage
-----
    python scripts/build_method_library.py --type notebook --limit 200
    python scripts/build_method_library.py --type notebook --dry-run   # no library written
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv  # noqa: E402

for candidate in (REPO / ".env", Path("/Users/yfkang/i-guide-platform-flask-servers/.env")):
    if candidate.exists():
        load_dotenv(candidate)
        break

import requests  # noqa: E402

from extractors.base import (EMIT_LIBRARY, EMIT_OPENSEARCH,  # noqa: E402
                             ExtractContext)
from extractors.emitters import library_emitter  # noqa: E402
from extractors.manifest import UnifiedManifest  # noqa: E402
from extractors.notebook_extractor import NotebookExtractor  # noqa: E402
from extractors.sources import SourceError, resolve_and_fetch  # noqa: E402

BACKEND = "https://backend.i-guide.io"


def list_elements(element_type: str, limit: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    page = 0
    while len(out) < limit:
        r = requests.get(f"{BACKEND}/api/elements",
                         params={"element-type": element_type, "from": page * 50, "size": 50},
                         timeout=60)
        r.raise_for_status()
        batch = (r.json() or {}).get("elements") or []
        if not batch:
            break
        out.extend(batch)
        page += 1
    return out[:limit]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type", default="notebook")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--cache", default=".corpus_cache")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--index", action="store_true",
                    help="also emit extracted docs to the agent OpenSearch indices "
                         "(requires AGENT_KB_BACKEND=opensearch and a reachable embedder)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    elements = list_elements(args.type, args.limit)
    print(f"{args.type}: {len(elements)} elements\n")

    manifest = UnifiedManifest(repo_id="iguide-corpus", source_url=BACKEND, cloned_at="now")
    stats: collections.Counter = collections.Counter()
    verdicts: collections.Counter = collections.Counter()
    per_element: List[Dict[str, Any]] = []

    for i, el in enumerate(elements, 1):
        eid = el.get("id") or ""
        short = eid[:8]
        row: Dict[str, Any] = {"id": eid, "title": (el.get("title") or "")[:70]}
        try:
            meta = requests.get(f"{BACKEND}/api/elements/{eid}", timeout=60).json()
        except Exception as exc:
            stats["metadata_error"] += 1
            row.update(stage="metadata", error=str(exc)[:100])
            per_element.append(row)
            continue

        cached = sorted(cache.glob(f"{short}__*"))
        try:
            if cached:
                local = cached[0]
                row["cached"] = True
            else:
                src = resolve_and_fetch(meta, cache, element_id=eid,
                                        filename=f"{short}__{Path(str(meta.get('notebook-file') or 'nb.ipynb')).name}")
                local = Path(src.local_path)
                row.update(sha256=src.sha256, bytes=src.bytes, ref=src.ref)
            stats["fetched"] += 1
        except SourceError as exc:
            stats[f"unfetchable:{exc.kind}"] += 1
            row.update(stage="fetch", error=str(exc)[:110])
            per_element.append(row)
            continue

        ctx = ExtractContext(element_id=short, element_type=args.type,
                             source_url=str(meta.get("notebook-url") or ""),
                             fields={"title": el.get("title") or short,
                                     "tags": meta.get("tags") or []},
                             targets=[EMIT_OPENSEARCH, EMIT_LIBRARY])
        try:
            result = NotebookExtractor().extract(str(local), ctx=ctx)
        except Exception as exc:
            stats["unparseable"] += 1
            row.update(stage="extract", error=f"{type(exc).__name__}: {exc}"[:110])
            per_element.append(row)
            continue

        units = [a for a in result.assets if getattr(a, "unit", None)]
        for a in units:
            verdicts[(a.unit.get("callability") or {}).get("verdict", "?")] += 1
        callable_units = [a for a in units
                          if (a.unit.get("callability") or {}).get("verdict") == "callable"]
        row.update(units=len(units), callable=len(callable_units))
        stats["extracted"] += 1
        stats["with_callable_unit"] += 1 if callable_units else 0
        manifest.add_result(f"notebook:{short}", result)
        per_element.append(row)

        if i % 20 == 0:
            print(f"  … {i}/{len(elements)}  fetched={stats['fetched']} "
                  f"callable_elements={stats['with_callable_unit']}")

    print(f"\n{'='*60}")
    print(f"{'elements':<26}{len(elements)}")
    for key in ("fetched", "extracted", "with_callable_unit", "unparseable", "metadata_error"):
        if stats[key]:
            print(f"{key:<26}{stats[key]}")
    for key in sorted(k for k in stats if k.startswith("unfetchable")):
        print(f"{key:<26}{stats[key]}")
    print(f"\nunit verdicts: {dict(verdicts)}")

    if args.dry_run:
        print("\n--dry-run: library not written")
    else:
        from agent_runtime.file_store import storage_root
        root = Path(storage_root()) / "method_library"
        out = library_emitter.emit(manifest, root=root)
        print(f"\nlibrary: {root}")
        print(f"  modules written {len(out['written'])}")
        print(f"  registry size   {out['registry_size']}")
        print(f"  skipped         {len(out['skipped'])}")
        for s in out["skipped"][:5]:
            print(f"    - {s.get('unit')}: {s.get('reason')}")

    if args.index and not args.dry_run:
        import os as _os
        from extractors.emitters import opensearch_emitter
        _os.environ.setdefault("AGENT_KB_BACKEND", "opensearch")
        print("\nindexing to the agent KB …")
        summary = opensearch_emitter.emit(manifest)
        print(f"  backend  {summary.get('backend')}")
        print(f"  docs     {summary.get('doc_count') or summary.get('indexed')}")
        for idx, n in sorted((summary.get('indices') or {}).items()):
            print(f"    {idx:<40}{n}")
        if summary.get("errors"):
            print(f"  errors   {len(summary['errors'])}: {summary['errors'][:3]}")

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"stats": dict(stats), "verdicts": dict(verdicts),
                                 "elements": per_element}, indent=2))
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
