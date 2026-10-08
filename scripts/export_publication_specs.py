#!/usr/bin/env python3
"""Run method extraction over every readable open-access document and export the specs.

    LLM_PROVIDER=claude-cli CLAUDE_CLI_MODEL=sonnet \
        python scripts/export_publication_specs.py --out outputs/publication_specs.json

``audit_publication_reach.py`` answers "is this readable" — a fact about the bytes, with no model
involved. This answers the different question the reader actually wants: *what did we get out of
it*. Ordered steps, the datasets and tools the paper names, its declared parameters.

Resumable by default: a document whose spec is already in ``--out`` is skipped, so a run that dies
at document 40 of 72 resumes rather than re-paying for the first 39. Pass ``--refresh`` to redo
them.

Storage is redirected to a temporary root so nothing here touches the real method library or the
local agent-KB store.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _titles(cache: Path) -> dict:
    out = {}
    listing = cache / "_elements_publication.json"
    if listing.is_file():
        try:
            data = json.loads(listing.read_text())
            for el in (data if isinstance(data, list) else data.get("elements") or []):
                out[str(el.get("id"))[:8]] = str(el.get("title") or "")
        except (OSError, ValueError):
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(REPO / ".corpus_cache"))
    ap.add_argument("--out", default=str(REPO / "outputs" / "publication_specs.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-chunks", type=int, default=3,
                    help="PUB_MAX_CHUNKS; caps how much of a long paper is read")
    ap.add_argument("--refresh", action="store_true", help="redo documents already exported")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="pubspec_")
    os.environ["AGENT_FILE_STORAGE_ROOT"] = tmp
    os.environ["AGENT_KB_STORE_DIR"] = os.path.join(tmp, "kb")
    os.environ["AGENT_METHOD_LIBRARY_DIR"] = os.path.join(tmp, "lib")
    os.environ["AGENT_KB_BACKEND"] = "local"
    os.environ["PUB_MAX_CHUNKS"] = str(args.max_chunks)

    from dotenv import load_dotenv
    load_dotenv(REPO / ".env", override=False)
    if not (os.getenv("LLM_PROVIDER") or "").strip():
        print("LLM_PROVIDER is unset — every document would degrade to `llm_unavailable`.\n"
              "Re-run with LLM_PROVIDER=claude-cli (development only).")
        return 2

    from extractors.base import EMIT_OPENSEARCH, ExtractContext
    from extractors.publication_extractor import PublicationExtractor

    out_path = Path(args.out)
    existing = {}
    if out_path.is_file() and not args.refresh:
        try:
            existing = {r["element"]: r
                        for r in (json.loads(out_path.read_text()).get("rows") or [])}
        except (OSError, ValueError, KeyError):
            existing = {}

    cache = Path(args.cache)
    titles = _titles(cache)
    paths = sorted(cache.glob("*oa.pdf")) + sorted(cache.glob("*oa.html"))
    if args.limit:
        paths = paths[: args.limit]

    rows, started = [], time.monotonic()
    for i, path in enumerate(paths, 1):
        short = path.name.split("__", 1)[0]
        if short in existing:
            rows.append(existing[short])
            continue
        title = titles.get(short) or short
        ctx = ExtractContext(element_id=short, element_type="publication",
                             fields={"title": title}, targets=[EMIT_OPENSEARCH])
        try:
            asset = PublicationExtractor().extract(str(path), ctx=ctx).assets[0]
        except Exception as exc:
            rows.append({"element": short, "title": title,
                         "error": f"{type(exc).__name__}: {exc}"[:200]})
            print(f"  {i:>3}/{len(paths)}  !! {short}  {type(exc).__name__}")
            continue
        ex = asset.extracted or {}
        rows.append({
            "element": short, "file": path.name, "title": title,
            "status": ex.get("status"), "degraded": ex.get("degraded"),
            "is_method_spec": ex.get("is_method_spec"),
            "summary": (asset.contents or "").split("\n", 1)[-1][:900],
            "steps": ex.get("steps") or [],
            "datasets_referenced": ex.get("datasets_referenced") or [],
            "tools_referenced": ex.get("tools_referenced") or [],
            "params": ex.get("params") or {},
            "chunks_parsed": ex.get("chunks_parsed"),
            "chunks_attempted": ex.get("chunks_attempted"),
            "chunks_total": ex.get("chunks_total"),
            "chars_seen": ex.get("chars_seen"), "chars_total": ex.get("chars_total"),
            "read_note": ex.get("read_note"), "source_kind": ex.get("source_kind"),
        })
        print(f"  {i:>3}/{len(paths)}  {short}  {str(ex.get('status')):<20}"
              f"{len(ex.get('steps') or []):>3} steps  "
              f"{int(time.monotonic() - started):>5}s elapsed")
        # Written after every document, so a run that dies keeps everything it paid for.
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"count": len(rows), "rows": rows},
                                       indent=2, default=str) + "\n")

    specs = sum(1 for r in rows if r.get("is_method_spec"))
    steps = sum(len(r.get("steps") or []) for r in rows)
    print(f"\n{len(rows)} document(s): {specs} carry a method spec, {steps} steps total")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
