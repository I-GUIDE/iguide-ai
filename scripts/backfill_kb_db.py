#!/usr/bin/env python3
"""Load everything extraction has already produced into the Postgres record.

    python scripts/backfill_kb_db.py
    python scripts/backfill_kb_db.py --dsn postgresql://… --dry-run

Reads the on-disk method library and the JSON exports under ``outputs/``. **Nothing is
re-extracted**: no corpus fetch, no LLM call, no cluster query. That is the point of backfilling
from what exists — the 203 publications cost an LLM call each and the datasets cost a download,
so the first thing the new store has to prove is that those results are recoverable without
paying for them twice.

Sources:
  agent_chat_files/method_library/…/_registry.json  + v_<sha>.py   the contracts AND their slices
  outputs/notebook_blocks.json    blocks, and every promoted unit including the refused ones
  outputs/dataset_details.json + extract_dataset.json
  outputs/publication_specs.json + publication_reach.json + extract_publication_*.json
  outputs/extract_code_gated.json

Idempotent: every write is an upsert keyed the way the schema is keyed, so running it twice is
the same as running it once. Run it again after any extraction pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from extractors import kb_db  # noqa: E402

LIBRARY = REPO / "agent_chat_files" / "method_library" / "iguide_methods"


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _slice_source(module: str, element_package: str) -> str:
    """The slice as it sits on disk, so the mounted library becomes a projection of this row."""
    if not module:
        return ""
    name = module.rsplit(".", 1)[-1]
    path = LIBRARY / element_package / f"{name}.py"
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def collect_elements(sources: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Every element id any source mentions, with the best title available for it.

    Built first and inserted first: `unit`, `block`, `dataset_file` and `publication` all carry a
    foreign key to `element`, which is the point — an orphan block becomes a constraint violation
    at write time instead of something a browser page discovers months later.
    """
    elements: Dict[str, Dict[str, Any]] = {}

    def note(element_id, *, title="", element_type="", **extra):
        element_id = str(element_id or "")[:8]
        if not element_id:
            return
        row = elements.setdefault(element_id, {"id": element_id, "title": "", "element_type": "",
                                               "tags": [], "authors": [], "source_url": "",
                                               "doi": "", "fields": {}})
        if title and not row["title"]:
            row["title"] = str(title)
        if element_type and not row["element_type"]:
            row["element_type"] = element_type
        for key, value in extra.items():
            if value and not row.get(key):
                row[key] = value

    for row in ((sources.get("notebooks") or {}).get("rows") or []):
        note(row.get("element"), title=row.get("title"), element_type="notebook",
             tags=row.get("tags") or [], authors=row.get("authors") or [],
             source_url=row.get("repo") or "")
    for row in ((sources.get("dataset_details") or {}).get("rows") or []):
        note(row.get("element"), title=row.get("title"), element_type="dataset")
    for row in ((sources.get("dataset_outcomes") or {}).get("elements") or []):
        note(row.get("id"), title=row.get("title"), element_type="dataset")
    for row in ((sources.get("publication_outcomes") or {}).get("elements") or []):
        note(row.get("id"), title=row.get("title"), element_type="publication",
             doi=row.get("doi") or "")
    for row in ((sources.get("code_outcomes") or {}).get("elements") or []):
        note(row.get("id"), title=row.get("title"), element_type="code")
    # A unit whose element no source named still needs a parent row, or the FK rejects it. That
    # is the constraint doing its job; give it the minimum truthful row rather than dropping the
    # unit, and let the empty title show which elements have no metadata anywhere.
    for entry in (sources.get("registry") or {}).values():
        if isinstance(entry, dict) and entry.get("signature"):
            prov = entry.get("provenance") or {}
            note(prov.get("element_id"), element_type=prov.get("extractor") or "")
    return elements


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be written and roll back")
    args = ap.parse_args()

    outputs = REPO / "outputs"
    sources = {
        "registry": _load(LIBRARY / "_registry.json") or {},
        "notebooks": _load(outputs / "notebook_blocks.json") or {},
        "dataset_details": _load(outputs / "dataset_details.json") or {},
        "dataset_outcomes": _load(outputs / "extract_dataset.json") or {},
        "publication_reach": _load(outputs / "publication_reach.json") or {},
        "publication_specs": _load(outputs / "publication_specs.json") or {},
        "code_outcomes": _load(outputs / "extract_code_gated.json") or {},
    }
    for name in ("extract_publication_v3.json", "extract_publication_v2.json",
                 "extract_publication_rederive.json", "extract_publication.json"):
        data = _load(outputs / name)
        if data:
            sources["publication_outcomes"] = data
            break
    sources.setdefault("publication_outcomes", {})

    empty = [k for k, v in sources.items() if not v]
    for name in empty:
        print(f"  no data for {name} — its rows will be absent")

    elements = collect_elements(sources)
    written = {"element": 0, "unit": 0, "block": 0, "dataset_file": 0, "publication": 0}
    skipped: list = []

    with kb_db.connect(args.dsn) as conn:
        kb_db.ensure_schema(conn)
        with conn.cursor() as cur:
            # ---- elements ------------------------------------------------------------
            for row in elements.values():
                cur.execute("""
                    INSERT INTO element (id, title, element_type, tags, authors, source_url, doi)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        title = CASE WHEN excluded.title <> '' THEN excluded.title
                                     ELSE element.title END,
                        element_type = CASE WHEN excluded.element_type <> ''
                                            THEN excluded.element_type
                                            ELSE element.element_type END,
                        tags = excluded.tags, authors = excluded.authors,
                        source_url = excluded.source_url, doi = excluded.doi,
                        updated_at = now()
                """, (row["id"], row["title"], row["element_type"],
                      json.dumps(row["tags"]), json.dumps(row["authors"]),
                      row["source_url"], row["doi"]))
                written["element"] += 1

            # ---- units, with their slices --------------------------------------------
            for key, entry in (sources["registry"] or {}).items():
                if not isinstance(entry, dict) or not entry.get("signature"):
                    continue
                # Aliases duplicate a qualified entry, and the ambiguity stubs exist ONLY because
                # a flat `{package}.{symbol}` key could not hold two units of the same name. The
                # composite key here can, so neither is a row.
                if entry.get("alias_for") or entry.get("ambiguous"):
                    continue
                prov = entry.get("provenance") or {}
                element_id = str(prov.get("element_id") or "")[:8]
                symbol = str(entry.get("library_symbol") or "").rsplit(".", 1)[-1]
                source_rel = str(prov.get("source_rel_path") or "")
                sha = str(entry.get("slice_sha") or "")
                if not (element_id and symbol and sha):
                    skipped.append({"key": key, "reason": "missing element, symbol or sha"})
                    continue
                module = str(entry.get("module") or entry.get("library_module") or "")
                package = module.split(".")[1] if module.count(".") >= 1 else ""
                call = entry.get("callability") if isinstance(entry.get("callability"), dict) else {}
                symbol_text = kb_db.expand_identifiers(f"{symbol} {key}")
                element_text = kb_db.expand_identifiers(
                    f"{(elements.get(element_id) or {}).get('title') or ''} {element_id}")
                cur.execute("""
                    INSERT INTO unit (element_id, source_rel_path, symbol, slice_sha,
                        qualified_name, library_module, signature, returns, return_kind,
                        unit_kind, doc_summary, docstring, verdict, extractor, fast_path,
                        callability, params, invariants, requirements, provenance,
                        slice_source, symbol_text, element_text)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                            %s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (element_id, source_rel_path, symbol, slice_sha) DO UPDATE SET
                        qualified_name = excluded.qualified_name,
                        library_module = excluded.library_module,
                        signature = excluded.signature, returns = excluded.returns,
                        return_kind = excluded.return_kind, unit_kind = excluded.unit_kind,
                        doc_summary = excluded.doc_summary, docstring = excluded.docstring,
                        verdict = excluded.verdict, extractor = excluded.extractor,
                        fast_path = excluded.fast_path, callability = excluded.callability,
                        params = excluded.params, invariants = excluded.invariants,
                        requirements = excluded.requirements, provenance = excluded.provenance,
                        slice_source = excluded.slice_source,
                        symbol_text = excluded.symbol_text, element_text = excluded.element_text,
                        updated_at = now()
                """, (
                    element_id, source_rel, symbol, sha,
                    str(entry.get("qualified_name") or key), module,
                    str(entry.get("signature") or ""), str(entry.get("returns") or ""),
                    str(entry.get("return_kind") or ""), str(entry.get("unit_kind") or ""),
                    str(entry.get("doc_summary") or ""), str(entry.get("docstring") or ""),
                    str(call.get("verdict") or "callable"), str(prov.get("extractor") or ""),
                    bool(entry.get("fast_path")),
                    json.dumps(call), json.dumps(entry.get("params") or []),
                    json.dumps(entry.get("invariants") or []),
                    json.dumps(entry.get("requirements") or {}), json.dumps(prov),
                    _slice_source(module, package), symbol_text, element_text))
                written["unit"] += 1

            # ---- units the analyzer REFUSED --------------------------------------------
            #
            # The registry holds only what shipped, so loading it alone leaves the store
            # answering "why was this not promoted" with silence. A refused unit names the
            # blocker — `needs_globals` on a module-level frame, `needs_instance` on a bound
            # method — and that is the list of extraction limits in priority order. It carries
            # no slice and no sha, which the empty `slice_sha` and the CHECK constraint say
            # exactly.
            for row in ((sources["notebooks"] or {}).get("rows") or []):
                element_id = str(row.get("element") or "")[:8]
                source_rel = str(row.get("file") or "")
                for unit in (row.get("units") or []):
                    if unit.get("verdict") == "callable":
                        continue          # already loaded from the registry, WITH its slice
                    symbol = str(unit.get("symbol") or "").rsplit(".", 1)[-1]
                    if not (element_id and symbol):
                        continue
                    call = {"verdict": unit.get("verdict"), "reason": unit.get("reason") or "",
                            "global_reads": unit.get("global_reads") or [],
                            "free_names": unit.get("free_names") or [],
                            "requires_units": unit.get("requires_units") or []}
                    cur.execute("""
                        INSERT INTO unit (element_id, source_rel_path, symbol, slice_sha,
                            qualified_name, signature, returns, doc_summary, verdict, extractor,
                            callability, symbol_text, element_text)
                        VALUES (%s,%s,%s,'',%s,%s,%s,%s,%s,'notebook',%s,%s,%s)
                        ON CONFLICT (element_id, source_rel_path, symbol, slice_sha)
                        DO UPDATE SET verdict = excluded.verdict,
                                      callability = excluded.callability,
                                      doc_summary = excluded.doc_summary,
                                      signature = excluded.signature, updated_at = now()
                    """, (element_id, source_rel, symbol,
                          str(unit.get("symbol") or symbol),
                          str(unit.get("signature") or ""), str(unit.get("returns") or ""),
                          str(unit.get("summary") or ""), str(unit.get("verdict") or ""),
                          json.dumps(call),
                          kb_db.expand_identifiers(symbol),
                          kb_db.expand_identifiers(
                              f"{(elements.get(element_id) or {}).get('title') or ''}")))
                    written["unit_refused"] = written.get("unit_refused", 0) + 1

            # ---- blocks ---------------------------------------------------------------
            for row in ((sources["notebooks"] or {}).get("rows") or []):
                element_id = str(row.get("element") or "")[:8]
                for block in (row.get("blocks") or []):
                    cur.execute("""
                        INSERT INTO block (doc_id, element_id, ord, title, code, markdown,
                            parse_ok, tools, imports, file_refs, constructs)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (doc_id) DO UPDATE SET
                            element_id = excluded.element_id, ord = excluded.ord,
                            title = excluded.title, code = excluded.code,
                            markdown = excluded.markdown, parse_ok = excluded.parse_ok,
                            tools = excluded.tools, imports = excluded.imports,
                            file_refs = excluded.file_refs, constructs = excluded.constructs,
                            updated_at = now()
                    """, (block.get("doc_id"), element_id, block.get("order"),
                          str(block.get("title") or ""), str(block.get("code") or ""),
                          str(block.get("markdown") or ""), block.get("parse_ok"),
                          json.dumps(block.get("tools") or []),
                          json.dumps(block.get("imports") or []),
                          json.dumps(block.get("file_refs") or []),
                          json.dumps(block.get("constructs") or [])))
                    written["block"] += 1

            # ---- datasets --------------------------------------------------------------
            outcome_by = {str(r.get("id"))[:8]: r
                          for r in ((sources["dataset_outcomes"] or {}).get("elements") or [])}
            for row in ((sources["dataset_details"] or {}).get("rows") or []):
                element_id = str(row.get("element") or "")[:8]
                ex = row.get("extracted") or {}
                outcome = outcome_by.get(element_id) or {}
                cur.execute("""
                    INSERT INTO dataset_file (element_id, file, bytes, format, family, row_count,
                        crs, geometry_type, bounds, columns, variables, extracted, envelope,
                        loader, stage, error)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (element_id, file) DO UPDATE SET
                        bytes = excluded.bytes, format = excluded.format,
                        family = excluded.family, row_count = excluded.row_count,
                        crs = excluded.crs, geometry_type = excluded.geometry_type,
                        bounds = excluded.bounds, columns = excluded.columns,
                        variables = excluded.variables, extracted = excluded.extracted,
                        envelope = excluded.envelope, loader = excluded.loader,
                        stage = excluded.stage, error = excluded.error, updated_at = now()
                """, (element_id, str(row.get("file") or ""), row.get("bytes"),
                      str(ex.get("format") or ""), str(ex.get("family") or ""),
                      ex.get("row_count"), str(ex.get("crs") or ""),
                      str(ex.get("geometry_type") or ""),
                      json.dumps(ex.get("bounds") or []),
                      json.dumps(ex.get("schema") or []),
                      json.dumps(ex.get("variables") or []),
                      json.dumps(ex), json.dumps(row.get("envelope")) if row.get("envelope") else None,
                      json.dumps(row.get("loader")) if row.get("loader") else None,
                      str(outcome.get("stage") or ""), str(outcome.get("error") or "")))
                written["dataset_file"] += 1

            # Every dataset element's outcome, including the 102 that never produced a local
            # file. "Why there is no file" is the extraction finding for those, and a record
            # that held only the 43 readable ones would misreport the corpus.
            for element_id, row in outcome_by.items():
                cur.execute("""
                    INSERT INTO dataset_outcome (element_id, stage, error, link_kind, note,
                        primary_file, files_listed, members, listed_bytes, detail)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (element_id) DO UPDATE SET
                        stage = excluded.stage, error = excluded.error,
                        link_kind = excluded.link_kind, note = excluded.note,
                        primary_file = excluded.primary_file,
                        files_listed = excluded.files_listed, members = excluded.members,
                        listed_bytes = excluded.listed_bytes, detail = excluded.detail,
                        updated_at = now()
                """, (element_id, str(row.get("stage") or ""), str(row.get("error") or ""),
                      str(row.get("link_kind") or ""), str(row.get("note") or ""),
                      str(row.get("primary") or ""), row.get("files_listed"),
                      row.get("members"), row.get("listed_bytes"), json.dumps(row)))
                written["dataset_outcome"] = written.get("dataset_outcome", 0) + 1

            # ---- publications ----------------------------------------------------------
            reach_by = {r.get("element"): r
                        for r in ((sources["publication_reach"] or {}).get("rows") or [])}
            spec_by = {r.get("element"): r
                       for r in ((sources["publication_specs"] or {}).get("rows") or [])}
            for row in ((sources["publication_outcomes"] or {}).get("elements") or []):
                element_id = str(row.get("id") or "")[:8]
                if not element_id:
                    continue
                reach = reach_by.get(element_id) or {}
                spec = spec_by.get(element_id) or {}
                cur.execute("""
                    INSERT INTO publication (element_id, doi, licence, outcome, reason, status,
                        summary, steps, datasets_referenced, tools_referenced, declared_params,
                        chars, sha256)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (element_id) DO UPDATE SET
                        doi = excluded.doi, licence = excluded.licence,
                        outcome = excluded.outcome, reason = excluded.reason,
                        status = excluded.status, summary = excluded.summary,
                        steps = excluded.steps,
                        datasets_referenced = excluded.datasets_referenced,
                        tools_referenced = excluded.tools_referenced,
                        declared_params = excluded.declared_params, chars = excluded.chars,
                        sha256 = excluded.sha256, updated_at = now()
                """, (element_id, str(row.get("doi") or ""), str(row.get("licence") or ""),
                      str(reach.get("outcome") or ""), str(reach.get("reason") or ""),
                      str(spec.get("status") or ""), str(spec.get("summary") or ""),
                      json.dumps(spec.get("steps") or []),
                      json.dumps(spec.get("datasets_referenced") or []),
                      json.dumps(spec.get("tools_referenced") or []),
                      json.dumps(spec.get("params") or {}),
                      reach.get("chars"), str(row.get("sha256") or "")))
                written["publication"] += 1

        if args.dry_run:
            conn.rollback()
            print("\n-- dry run, rolled back --")
        else:
            conn.commit()

        counts = kb_db.table_counts(conn)

    print("\nwritten this run: " + ", ".join(f"{v} {k}" for k, v in written.items() if v))
    if skipped:
        print(f"skipped {len(skipped)}: " + "; ".join(
            f"{s['key']} ({s['reason']})" for s in skipped[:5]))
    print("rows in the database: " + ", ".join(f"{v} {k}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
