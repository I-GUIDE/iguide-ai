#!/usr/bin/env python3
"""Generate a self-contained HTML browser over everything extraction produced.

    python scripts/build_extraction_browser.py
    python scripts/build_extraction_browser.py --out outputs/extraction_browser.html

Reads the run records under ``outputs/`` and the on-disk method-library registry, normalizes them
into one record list, and writes a single HTML file with the data inlined. Self-contained on
purpose: the page has to open from disk and render inside an Artifact, where a strict CSP blocks
every external request, so there is nowhere to fetch data from at view time.

Re-run it after any extraction pass and the page is current. Nothing here writes to the corpus,
the library, or a cluster.

Sources, each optional — a missing one is reported, never silently skipped:
  outputs/_registry (agent_chat_files/method_library/…)   the unit contracts
  outputs/dataset_details.json    per-file schema/CRS/bounds (scripts/export_dataset_details.py)
  outputs/extract_dataset.json    per-element dataset outcome
  outputs/publication_reach.json  per-document readability
  outputs/extract_publication_*.json   per-element publication outcome
  outputs/extract_code_gated.json      per-element code outcome
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MAX_EVIDENCE = 180


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _units(registry: dict) -> list:
    out = []
    for key, entry in (registry or {}).items():
        if not isinstance(entry, dict) or not entry.get("signature"):
            continue
        if entry.get("alias_for") or entry.get("ambiguous"):
            continue
        prov = entry.get("provenance") or {}
        params = []
        for p in (entry.get("params") or []):
            if not isinstance(p, dict):
                continue
            params.append({
                "name": p.get("name"),
                "annotation": p.get("annotation") or "",
                "type": p.get("inferred_type") or "",
                "unit": p.get("declared_unit") or "",
                "crs": p.get("crs_expectation") or "",
                "required": bool(p.get("required")),
                "default": p.get("default") or "",
                "evidence": str(p.get("evidence") or "")[:MAX_EVIDENCE],
            })
        summary = (entry.get("doc_summary") or "").strip()
        annotated = any(p["annotation"] for p in params)
        flags = []
        if not params:
            flags.append("zero-arg")
        if not summary and not annotated and not entry.get("returns"):
            flags.append("naked")
        if entry.get("invariants"):
            flags.append("has-invariants")
        if summary:
            flags.append("documented")
        out.append({
            "kind": "unit",
            "id": key,
            "name": str(entry.get("library_symbol") or key).rsplit(".", 1)[-1],
            "title": str(entry.get("library_symbol") or key).rsplit(".", 1)[-1],
            "element": prov.get("element_id") or "",
            "extractor": prov.get("extractor") or "",
            "source": prov.get("source_rel_path") or "",
            "unit_kind": entry.get("unit_kind") or "",
            "signature": entry.get("signature") or "",
            "summary": summary,
            "returns": entry.get("returns") or "",
            "params": params,
            "invariants": [{"check": i.get("check"), "target": i.get("target"),
                            "args": i.get("args") or {},
                            "evidence": str(i.get("evidence") or "")[:MAX_EVIDENCE]}
                           for i in (entry.get("invariants") or []) if isinstance(i, dict)],
            "requires": (entry.get("requirements") or {}).get("pip") or [],
            "slice_sha": entry.get("slice_sha") or "",
            "import": (f"from {entry.get('module')} import {entry.get('library_symbol')}"
                       if entry.get("module") and entry.get("library_symbol") else ""),
            "flags": flags,
        })
    return out


def _datasets(details: dict, outcomes: dict) -> list:
    by_element = {}
    for row in ((outcomes or {}).get("elements") or []):
        by_element[str(row.get("id"))[:8]] = row
    out = []
    for row in ((details or {}).get("rows") or []):
        ex = row.get("extracted") or {}
        outcome = by_element.get(row.get("element")) or {}
        flags = []
        if ex.get("schema"):
            flags.append("has-schema")
        if ex.get("crs"):
            flags.append("has-crs")
        if ex.get("variables"):
            flags.append("has-variables")
        if row.get("loader"):
            flags.append("has-loader")
        if ex.get("primary_member"):
            flags.append("from-archive")
        out.append({
            "kind": "dataset",
            "id": row.get("element"),
            "name": row.get("file"),
            "title": row.get("title") or row.get("file"),
            "bytes": row.get("bytes"),
            "format": ex.get("format") or "",
            "family": ex.get("family") or "",
            "row_count": ex.get("row_count"),
            "schema": ex.get("schema") or [],
            "crs": ex.get("crs") or "",
            "bounds": ex.get("bounds") or [],
            "geometry_type": ex.get("geometry_type") or "",
            "variables": ex.get("variables") or [],
            "dims": ex.get("dims") or {},
            "resolution": ex.get("resolution") or [],
            "bands": ex.get("bands"),
            "dtypes": ex.get("dtypes") or [],
            "primary_member": ex.get("primary_member") or "",
            "member_families": ex.get("member_families") or {},
            "archive_note": ex.get("archive_note") or "",
            "geometry_from": ex.get("geometry_from") or {},
            "envelope": row.get("envelope") or None,
            "loader": row.get("loader") or None,
            "stage": outcome.get("stage") or "extracted-ok",
            "error": outcome.get("error") or "",
            "flags": flags,
        })
    # Elements that produced no local file at all — still worth browsing, that IS the finding.
    seen = {r["id"] for r in out}
    for short, row in by_element.items():
        if short in seen:
            continue
        out.append({
            "kind": "dataset", "id": short, "name": short,
            "title": row.get("title") or short,
            "stage": row.get("stage") or "", "error": row.get("error") or "",
            "link_kind": row.get("link_kind") or "", "note": row.get("note") or "",
            "members": row.get("members"), "size_mb": row.get("size_mb"),
            "files_listed": row.get("files_listed"), "primary": row.get("primary") or "",
            "listed_bytes": row.get("listed_bytes"), "crs": row.get("crs") or "",
            "schema": [], "flags": ["no-local-file"],
        })
    return out


def _publications(reach: dict, outcomes: dict, specs: dict = None) -> list:
    read_by = {r.get("element"): r for r in ((reach or {}).get("rows") or [])}
    spec_by = {r.get("element"): r for r in ((specs or {}).get("rows") or [])}
    out = []
    for row in ((outcomes or {}).get("elements") or []):
        short = str(row.get("id"))[:8]
        r = read_by.get(short) or {}
        outcome = r.get("outcome") or ("not fetched" if row.get("error") else "fetched")
        flags = []
        if row.get("doi"):
            flags.append("has-doi")
        if row.get("licence"):
            flags.append(str(row["licence"]))
        if r.get("outcome") == "readable document":
            flags.append("readable")
        if row.get("pdf_from"):
            flags.append(f"pdf-from-{row['pdf_from']}")
        if row.get("source_kind"):
            flags.append(str(row["source_kind"]))
        spec = spec_by.get(short) or {}
        if spec.get("steps"):
            flags.append("has-method-spec")
        if spec.get("status") == "no_method_described":
            flags.append("no-method")
        out.append({
            "kind": "publication",
            "id": short,
            "name": short,
            "title": row.get("title") or short,
            # What extraction actually got OUT of the PDF, not just whether it opened.
            "spec_status": spec.get("status") or "",
            "spec_summary": (spec.get("summary") or "").strip(),
            "steps": spec.get("steps") or [],
            "datasets_referenced": spec.get("datasets_referenced") or [],
            "tools_referenced": spec.get("tools_referenced") or [],
            # NOT `params`. A unit's `params` is a LIST of parameter objects and the page's search
            # index calls .map() on it; a publication's declared parameters are an OBJECT. Naming
            # both `params` put two shapes under one key, `{}.map is not a function` threw while
            # building the index, and the whole script died before rendering a single row -- a
            # blank page from one field name.
            "declared_params": spec.get("params") or {},
            "chunks_parsed": spec.get("chunks_parsed"),
            "chunks_total": spec.get("chunks_total"),
            "chars_seen": spec.get("chars_seen"),
            "chars_total": spec.get("chars_total"),
            "doi": row.get("doi") or "",
            "licence": row.get("licence") or "",
            "outcome": outcome,
            "reason": r.get("reason") or row.get("error") or "",
            "chars": r.get("chars"),
            "sniffed": r.get("sniffed") or row.get("source_kind") or "",
            "bytes": r.get("bytes") or row.get("bytes"),
            "sha256": row.get("sha256") or "",
            "stage": row.get("stage") or "",
            "flags": flags,
        })
    return out


def _code(outcomes: dict) -> list:
    out = []
    for row in ((outcomes or {}).get("elements") or []):
        flags = []
        if row.get("stage") == "relevance":
            flags.append("off-domain")
        if row.get("units"):
            flags.append("produced-units")
        out.append({
            "kind": "code",
            "id": str(row.get("id"))[:8],
            "name": str(row.get("id"))[:8],
            "title": row.get("title") or "",
            "stage": row.get("stage") or "extracted-ok",
            "relevance": row.get("relevance"),
            "off_domain": row.get("off_domain") or [],
            "units": row.get("units"),
            "callable": row.get("callable"),
            "files": row.get("files"),
            "error": row.get("error") or "",
            "flags": flags,
        })
    return out


def _link_units_to_elements(records: list, units: list) -> None:
    """Give every element record the units extraction produced FROM it.

    "units 6, callable 5" is a count with nothing behind it: the question a reader actually has is
    *which five*. The library holds only units that shipped, so the ones listed here ARE the
    reachable ones — and where an element's `callable` count exceeds what is listed, that gap is
    itself worth seeing rather than smoothing over.
    """
    by_element: Dict[str, list] = {}
    for unit in units:
        element = str(unit.get("element") or "")
        if element:
            by_element.setdefault(element, []).append(unit)

    for record in records:
        if record.get("kind") == "unit":
            continue
        found = by_element.get(str(record.get("id") or ""))
        if not found:
            continue
        record["units_in_library"] = [
            {"id": u["id"], "name": u["name"], "signature": u.get("signature") or "",
             "summary": u.get("summary") or "", "unit_kind": u.get("unit_kind") or "",
             "source": u.get("source") or "", "flags": u.get("flags") or []}
            for u in sorted(found, key=lambda x: x["name"].lower())]
        if "has-units" not in record.setdefault("flags", []):
            record["flags"].append("has-units")


def collect(repo: Path) -> dict:
    outputs = repo / "outputs"
    registry_path = (repo / "agent_chat_files" / "method_library" / "iguide_methods"
                     / "_registry.json")
    sources, missing = {}, []

    def take(label, path):
        data = _load(Path(path))
        if data is None:
            missing.append(str(Path(path).relative_to(repo)))
        sources[label] = data
        return data

    registry = take("registry", registry_path)
    details = take("dataset_details", outputs / "dataset_details.json")
    ds_out = take("dataset_outcomes", outputs / "extract_dataset.json")
    reach = take("publication_reach", outputs / "publication_reach.json")
    pub_out = None
    for name in ("extract_publication_v2.json", "extract_publication_rederive.json",
                 "extract_publication.json"):
        candidate = outputs / name
        if candidate.is_file():
            pub_out = _load(candidate)
            sources["publication_outcomes"] = name
            break
    if pub_out is None:
        missing.append("outputs/extract_publication_*.json")
    code_out = take("code_outcomes", outputs / "extract_code_gated.json")

    specs = _load(outputs / "publication_specs.json")
    if specs is None:
        missing.append("outputs/publication_specs.json")

    unit_records = _units(registry or {})
    records = (unit_records + _datasets(details or {}, ds_out or {})
               + _publications(reach or {}, pub_out or {}, specs or {})
               + _code(code_out or {}))
    _link_units_to_elements(records, unit_records)
    counts = {}
    for r in records:
        counts[r["kind"]] = counts.get(r["kind"], 0) + 1
    return {"records": records, "counts": counts, "missing": missing,
            "publication_source": sources.get("publication_outcomes", "")}


TEMPLATE = Path(__file__).with_name("_extraction_browser_template.html")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "outputs" / "extraction_browser.html"))
    ap.add_argument("--stamp", default="", help="text shown as the generation time")
    args = ap.parse_args()

    payload = collect(REPO)
    if not payload["records"]:
        print("no extraction records found — run an extraction pass first")
        return 2
    for name in payload["missing"]:
        print(f"  missing (rows from it will be absent): {name}")

    template = TEMPLATE.read_text(encoding="utf-8")
    blob = json.dumps({**payload, "generated": args.stamp}, separators=(",", ":"),
                      default=str)
    # Only `</script` can terminate the block; escaping it is enough and keeps the JSON readable.
    page = template.replace("__DATA__", blob.replace("</script", "<\\/script"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"\n{sum(payload['counts'].values())} records: "
          + ", ".join(f"{v} {k}" for k, v in sorted(payload["counts"].items())))
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
