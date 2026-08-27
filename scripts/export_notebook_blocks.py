#!/usr/bin/env python3
"""Re-read every cached notebook and export the artifacts extraction produced from it.

    python scripts/export_notebook_blocks.py --out outputs/notebook_blocks.json

Notebooks are the largest source in the corpus and the only one with nothing to browse. The
per-element outcome files record dataset, publication and code runs; nothing writes down what a
notebook yielded, so the blocks and units that ARE the extraction have never been inspectable
outside a search index.

Each row is one notebook: its blocks in cell order (title, code, the markdown that named it,
resolved tools, imports, file references), the method units promoted out of it, and the
whole-notebook workflow descriptor when every cell parsed. That is enough to answer "what did we
get out of this notebook, and why did the rest not make it" without a cluster.

Storage is redirected to a temporary root, so nothing here touches the real method library or the
local agent-KB store — a probe that set the wrong environment variable once wrote 12 duplicate
units into the real registry.
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

MAX_CODE_CHARS = 4000
MAX_MARKDOWN_CHARS = 1200


def _clip(text, limit):
    text = str(text or "")
    return text if len(text) <= limit else text[:limit] + f"\n… [{len(text) - limit} more chars]"


def _titles(cache: Path) -> dict:
    """Element titles from the listing cache — the only source that works without Neo4j."""
    out = {}
    listing = cache / "_elements_notebook.json"
    if not listing.is_file():
        return out
    try:
        data = json.loads(listing.read_text())
    except (OSError, ValueError):
        return out
    for el in (data if isinstance(data, list) else data.get("elements") or []):
        short = str(el.get("id"))[:8]
        out[short] = {"title": str(el.get("title") or ""),
                      "tags": el.get("tags") or [],
                      "authors": el.get("authors") or [],
                      "repo": el.get("notebook-repo") or el.get("repo") or ""}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(REPO / ".corpus_cache"))
    ap.add_argument("--out", default=str(REPO / "outputs" / "notebook_blocks.json"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="nbexport_")
    os.environ["AGENT_FILE_STORAGE_ROOT"] = tmp
    os.environ["AGENT_KB_STORE_DIR"] = os.path.join(tmp, "kb")
    os.environ["AGENT_METHOD_LIBRARY_DIR"] = os.path.join(tmp, "lib")
    os.environ["AGENT_KB_BACKEND"] = "local"

    from dotenv import load_dotenv
    load_dotenv(REPO / ".env", override=False)

    from extractors.base import EMIT_LIBRARY, EMIT_OPENSEARCH, ExtractContext
    from extractors.notebook_extractor import NotebookExtractor

    cache = Path(args.cache)
    meta = _titles(cache)
    paths = sorted(p for p in cache.iterdir() if p.suffix.lower() == ".ipynb")
    if args.limit:
        paths = paths[: args.limit]

    rows = []
    for path in paths:
        short = path.name.split("__", 1)[0]
        info = meta.get(short) or {}
        title = info.get("title") or path.stem
        row = {"element": short, "file": path.name, "title": title,
               "tags": info.get("tags") or [], "authors": info.get("authors") or [],
               "repo": info.get("repo") or "", "bytes": path.stat().st_size,
               "blocks": [], "units": [], "workflow": None}
        ctx = ExtractContext(element_id=short, element_type="notebook",
                             fields={"title": title, "tags": info.get("tags") or []},
                             targets=[EMIT_OPENSEARCH, EMIT_LIBRARY])
        try:
            result = NotebookExtractor().extract(str(path), ctx=ctx)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"[:200]
            rows.append(row)
            print(f"  !! {path.name[:52]:<54}{row['error'][:60]}")
            continue

        for asset in result.assets:
            block = asset.block or {}
            if asset.kind == "notebook_block":
                row["blocks"].append({
                    "doc_id": asset.doc_id,
                    # The block's own name, without the ` — <element>` suffix the index carries:
                    # inside a notebook's own page the element name is on every row already.
                    "title": str(asset.title or "").rsplit(" — ", 1)[0],
                    "order": block.get("order"),
                    "code": _clip(block.get("code"), MAX_CODE_CHARS),
                    "markdown": _clip(block.get("markdown_context"), MAX_MARKDOWN_CHARS),
                    "parse_ok": block.get("parse_ok"),
                    "tools": block.get("resolved_tools") or [],
                    "imports": block.get("imports") or [],
                    # `file_io` is `{"referenced": [...]}`; unwrap it, because a consumer that
                    # expects a list of filenames gets an object it cannot iterate.
                    "file_refs": (block.get("file_io") or {}).get("referenced") or [],
                    "constructs": sorted({c.get("category") for c
                                          in (block.get("constructs") or []) if c.get("category")}),
                })
            elif asset.kind == "method_unit" and asset.unit:
                unit = asset.unit
                call = unit.get("callability")
                call = call if isinstance(call, dict) else {}
                row["units"].append({
                    "doc_id": asset.doc_id,
                    "symbol": unit.get("library_symbol") or unit.get("qualified_name") or "",
                    "signature": unit.get("signature") or "",
                    "summary": (unit.get("doc_summary") or "").strip(),
                    "returns": unit.get("returns") or "",
                    "verdict": call.get("verdict") or "",
                    # Why a unit did NOT make it is the more useful half: `global_reads` names
                    # the hidden dependency, `free_names` the unresolved one.
                    "reason": call.get("reason") or "",
                    "global_reads": call.get("global_reads") or [],
                    "free_names": call.get("free_names") or [],
                    "requires_units": call.get("requires_units") or [],
                    "invariants": [i.get("check") for i in (unit.get("invariants") or [])
                                   if isinstance(i, dict)],
                    "requires": (unit.get("requirements") or {}).get("pip") or [],
                    "slice_sha": unit.get("slice_sha") or "",
                    # No cell order here on purpose: a unit's provenance records the element and
                    # the source file, never the cell it came from. Emitting the key anyway gave
                    # the page a column that was blank for all 382 units.
                })
            elif asset.runnable:
                row["workflow"] = {
                    "workflow_id": asset.runnable.get("workflow_id") or "",
                    "mode": asset.runnable.get("mode") or "",
                    "entrypoint": asset.runnable.get("entrypoint") or "",
                    "params": asset.runnable.get("params") or [],
                }

        row["blocks"].sort(key=lambda b: (b.get("order") is None, b.get("order")))
        # A block whose title still reads `cell N` is one no author named — the count is the
        # honest measure of how far the naming tiers reach on this notebook.
        row["unnamed_blocks"] = sum(1 for b in row["blocks"]
                                    if str(b["title"]).startswith("cell "))
        rows.append(row)
        print(f"  ok {path.name[:52]:<54}{len(row['blocks']):>3} block(s)"
              f"{len(row['units']):>4} unit(s)"
              f"{'  workflow' if row['workflow'] else ''}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cache": args.cache, "count": len(rows), "rows": rows},
                              indent=2, default=str) + "\n")
    blocks = sum(len(r["blocks"]) for r in rows)
    units = sum(len(r["units"]) for r in rows)
    unnamed = sum(r.get("unnamed_blocks") or 0 for r in rows)
    with_units = sum(1 for r in rows if r["units"])
    callable_units = sum(1 for r in rows for u in r["units"] if u.get("verdict") == "callable")
    print(f"\n{len(rows)} notebook(s): {blocks} block(s), {units} unit(s) "
          f"from {with_units} notebook(s); {callable_units} callable")
    print(f"{blocks - unnamed}/{blocks} blocks carry an authored name")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
