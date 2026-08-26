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
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from extractors.sources import (SourceError, fetch_url,  # noqa: E402
                                resolve_and_fetch, source_link, source_link_or_doi)

BACKEND = "https://backend.i-guide.io"


def _get_with_backoff(url: str, params: Dict[str, Any], *, attempts: int = 6):
    """GET with exponential backoff on 429.

    The platform API rate-limits a listing sweep, and a bare `raise_for_status` turns that into
    an aborted corpus build — the failure looks like "the element type has no elements" in the
    summary, which is the wrong conclusion entirely. A 429 is the server asking for patience,
    so wait rather than give up or hammer.
    """
    import time as _t

    delay = 5.0
    for attempt in range(attempts):
        resp = requests.get(url, params=params, timeout=60)
        if resp.status_code != 429:
            return resp
        wait = float(resp.headers.get("Retry-After") or delay)
        if attempt + 1 < attempts:
            print(f"  … rate-limited, waiting {wait:.0f}s "
                  f"(attempt {attempt + 2} of {attempts})", flush=True)
            _t.sleep(wait)
            delay = min(delay * 2, 120.0)
    return resp


def _element_metadata(element_id: str, cache: Optional[Path] = None) -> Dict[str, Any]:
    """One element's full record, cached on disk.

    This was an uncached GET per element, so every corpus build issued one request per element —
    750 for a full sweep — and the platform rate-limited it. The failure then arrived as
    `metadata_error` on EVERY element, which reads as "the corpus is unreachable" rather than
    "we asked too fast". The record does not change between two runs an hour apart, so caching
    it removes the load entirely and makes a re-run nearly free.
    """
    path = (cache / "_meta" / f"{element_id}.json") if cache is not None else None
    if path is not None and path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    resp = _get_with_backoff(f"{BACKEND}/api/elements/{element_id}", {})
    resp.raise_for_status()
    meta = resp.json()
    if path is not None and isinstance(meta, dict) and meta:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(meta), encoding="utf-8")
        except OSError:
            pass
    return meta


def list_elements(element_type: str, limit: int, cache: Optional[Path] = None,
                  *, refresh: bool = False) -> List[Dict[str, Any]]:
    """The element listing, cached to disk.

    The sources were already cached by element id, but the LISTING was not — so a 429 from the
    platform API aborted the whole run even with 387 MB of notebooks sitting in the cache, which
    made "resumable" untrue in exactly the case resumability is for. The listing is also the
    cheapest thing to cache: one file, and the corpus does not change between two runs an hour
    apart.

    A partial listing is never cached: writing one would silently shrink the corpus on every
    later run, and a rate limit halfway through page 3 is the normal way that happens.
    """
    path = (cache / f"_elements_{element_type}.json") if cache is not None else None
    if path is not None and path.is_file() and not refresh:
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(rows, list) and rows:
                print(f"element listing: {len(rows)} from cache ({path})")
                return rows[:limit]
        except (OSError, ValueError):
            pass

    out: List[Dict[str, Any]] = []
    page = 0
    complete = False
    try:
        while len(out) < limit:
            r = _get_with_backoff(f"{BACKEND}/api/elements",
                                  {"element-type": element_type, "from": page * 50, "size": 50})
            r.raise_for_status()
            batch = (r.json() or {}).get("elements") or []
            if not batch:
                complete = True
                break
            out.extend(batch)
            page += 1
        else:
            complete = True
    except requests.RequestException as exc:
        if not out:
            raise
        print(f"  ! listing stopped after {len(out)} elements: {exc}")

    if path is not None and complete and out:
        try:
            path.write_text(json.dumps(out), encoding="utf-8")
        except OSError:
            pass
    return out[:limit]


# --------------------------------------------------------------------------- #
# Per-type extraction
# --------------------------------------------------------------------------- #

def _extractor_for(element_type: str):
    """The extractor class for an element type, or None when the type has no route.

    `ingest_submission` (the webhook path) has always dispatched on type; this driver called
    `NotebookExtractor()` unconditionally at every one of its runs, which is the entire reason
    the corpus library is notebook-only. `--type` was accepted, passed into the ExtractContext,
    and then ignored by the one line that mattered.
    """
    if element_type == "notebook":
        from extractors.notebook_extractor import NotebookExtractor
        return NotebookExtractor
    if element_type == "code":
        from extractors.code_extractor import CodeExtractor
        return CodeExtractor
    if element_type == "dataset":
        from extractors.data_extractor import DataExtractor
        return DataExtractor
    if element_type == "publication":
        from extractors.publication_extractor import PublicationExtractor
        return PublicationExtractor
    return None


def _files_for(element_type: str, local: Path) -> List[Path]:
    """Which files to run the extractor over.

    A notebook, dataset or publication element resolves to ONE file. A code element resolves to
    a repository, so its unit supply is every module in it — which is why the code type is the
    largest untapped source in the corpus and also why it needs a different shape here.
    """
    if local.is_dir():
        from extractors.fileclass import classify_github

        found = classify_github(str(local))
        from extractors.base import KIND_CODE_BLOCK, KIND_NOTEBOOK_BLOCK

        key = KIND_NOTEBOOK_BLOCK if element_type == "notebook" else KIND_CODE_BLOCK
        return [Path(f) for f in (found.get(key) or [])]
    return [local]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type", default="notebook")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--max-files", type=int, default=60,
                    help="per element; a repo can hold hundreds of modules")
    ap.add_argument("--cache", default=".corpus_cache")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--relevance-gate", action="store_true",
                    help="skip elements whose README/description is off-domain (see "
                         "extractors/relevance.py); recommended for --type code")
    ap.add_argument("--relevance-threshold", type=float, default=3.0)
    ap.add_argument("--no-graph", action="store_true",
                    help="use the rate-limited REST API instead of the platform graph")
    ap.add_argument("--refresh-listing", action="store_true",
                    help="re-fetch the element listing instead of using the cached copy")
    ap.add_argument("--index", action="store_true",
                    help="also emit extracted docs to the agent OpenSearch indices "
                         "(requires AGENT_KB_BACKEND=opensearch and a reachable embedder)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    # The GRAPH is preferred over the REST API: it is the store the API is a view of, it answers
    # a whole label in one read-only query, and it does not rate-limit a corpus sweep. The API
    # path issued one request per element — 750 for a full pass — and returned 429 partway,
    # which surfaced as `metadata_error` on every element and read as "the corpus is
    # unreachable". Private elements are excluded there, fail-closed.
    from_graph = False
    if not args.no_graph:
        try:
            from extractors import platform_graph

            if platform_graph.is_enabled():
                elements = platform_graph.elements(args.type, limit=args.limit)
                from_graph = True
                print(f"{args.type}: {len(elements)} public elements from the platform graph")
        except Exception as exc:
            print(f"  ! graph unavailable ({type(exc).__name__}: {exc}); using the REST API")
    if not from_graph:
        elements = list_elements(args.type, args.limit, cache, refresh=args.refresh_listing)
        print(f"{args.type}: {len(elements)} elements")
    print()

    manifest = UnifiedManifest(repo_id="iguide-corpus", source_url=BACKEND, cloned_at="now")
    stats: collections.Counter = collections.Counter()
    verdicts: collections.Counter = collections.Counter()
    per_element: List[Dict[str, Any]] = []

    for i, el in enumerate(elements, 1):
        eid = el.get("id") or ""
        short = eid[:8]
        row: Dict[str, Any] = {"id": eid, "title": (el.get("title") or "")[:70]}
        try:
            # A graph record IS the full metadata; the API's per-element GET is only needed
            # when falling back to it.
            meta = el if from_graph else _element_metadata(eid, cache)
        except Exception as exc:
            stats["metadata_error"] += 1
            row.update(stage="metadata", error=f"{type(exc).__name__}: {exc}"[:100])
            per_element.append(row)
            continue

        # A CODE element is a repository, not a file: its unit supply is every module in it.
        # `resolve_and_fetch` returns one file, so the code path clones instead.
        if args.type == "code":
            repo_url = str(meta.get("github-repo-link") or meta.get("github_url") or "").strip()
            if not repo_url:
                stats["unfetchable:unsupported"] += 1
                row.update(stage="fetch", error="no github-repo-link")
                per_element.append(row)
                continue
            clone_dir = cache / "_repos" / short
            try:
                if not clone_dir.exists():
                    subprocess.run(["git", "clone", "--depth", "1", "--quiet",
                                    repo_url, str(clone_dir)],
                                   check=True, capture_output=True, timeout=600)
                else:
                    row["cached"] = True
                local = clone_dir
                stats["fetched"] += 1
            except Exception as exc:
                stats["unfetchable:clone"] += 1
                row.update(stage="fetch", error=f"{type(exc).__name__}: {exc}"[:100])
                per_element.append(row)
                continue
        elif args.type == "publication":
            # `external_link` is a DOI or a publisher landing page for 198 of 200 publications,
            # not a PDF — which is why this type contributed zero method specs. Unpaywall maps
            # the DOI to the LEGALLY OPEN copy when one exists (94 of 176 resolvable DOIs, 73 of
            # them CC-BY), so the corpus only ingests text the publisher or author put in the
            # open. "Paywalled" is a fact about the world, recorded as its own outcome: a re-run
            # will not change it.
            from extractors.open_access import extract_doi, resolve
            from extractors.publication_extractor import read_document

            cached_pdf = sorted(cache.glob(f"{short}__*.pdf"))
            if cached_pdf:
                local, row["cached"] = cached_pdf[0], True
                stats["fetched"] += 1
            else:
                # Both vocabularies, via the shared lookup. Reading only `external_link` found
                # a DOI on 8 of 203 publications -- and those 8 were the ones with a cached PDF,
                # so the REST path had resolved exactly zero. The API names this field
                # `external-link-publication`.
                doi = extract_doi(source_link_or_doi(meta))
                if not doi:
                    stats["unfetchable:no_doi"] += 1
                    # Name the fields actually searched. Saying "external_link" was doubly
                    # misleading: it is not the field the API uses, and it was the wrong name that
                    # caused this branch to fire 195 times in the first place.
                    row.update(stage="fetch",
                               error="no DOI in any recorded source link "
                                     "(external-link-publication / doi / direct-download-link)")
                    per_element.append(row)
                    continue
                oa = resolve(doi)
                row["doi"], row["licence"] = doi, oa.licence
                if oa.pdf_from:
                    row["pdf_from"] = oa.pdf_from
                if not oa.pdf_url and not (oa.is_oa and oa.landing_url):
                    stats[f"unfetchable:{'closed' if not oa.is_oa else 'oa_no_pdf'}"] += 1
                    row.update(stage="fetch", error=oa.reason[:110])
                    per_element.append(row)
                    continue

                # Two shapes of the same open copy, tried in order. The reader added in M8.10
                # routes on the file's first bytes rather than its extension, so a full-text HTML
                # article is now as readable as a PDF -- which turns "open access, but no direct
                # PDF url" from a dead end into a second attempt. 39 of the corpus's 177 DOIs
                # landed in exactly that bucket.
                attempts = []
                if oa.pdf_url:
                    attempts.append((oa.pdf_url, f"{short}__oa.pdf", "pdf_get"))
                if oa.landing_url and oa.landing_url != oa.pdf_url:
                    attempts.append((oa.landing_url, f"{short}__oa.html", "landing_get"))

                last_error = ""
                for url, name, failure_kind in attempts:
                    try:
                        # HTML is the payload for a landing page, not a redirect to refuse.
                        src = fetch_url(url, cache / name, element_id=eid,
                                        allow_html=name.endswith(".html"))
                    except Exception as exc:
                        last_error = f"{type(exc).__name__}: {exc}"[:100]
                        continue
                    candidate = Path(src.local_path)
                    # A 200 is not a document. Ask the reader BEFORE counting this a success, so
                    # a bot check or a paywall interstitial falls through to the next attempt
                    # instead of being recorded as a fetched paper.
                    text, read_note = read_document(str(candidate))
                    if not text.strip():
                        last_error = read_note or "fetched, but no readable text"
                        try:
                            candidate.unlink()
                        except OSError:
                            pass
                        continue
                    local = candidate
                    row.update(sha256=src.sha256, bytes=src.bytes, chars=len(text),
                               source_kind=("html" if name.endswith(".html") else "pdf"))
                    if read_note:
                        row["read_note"] = read_note[:100]
                    stats["fetched"] += 1
                    break
                else:
                    stats[f"unfetchable:{attempts[-1][2] if attempts else 'pdf_get'}"] += 1
                    row.update(stage="fetch", error=last_error[:110] or "no attempt succeeded")
                    per_element.append(row)
                    continue
        else:
            cached = sorted(cache.glob(f"{short}__*"))
            try:
                if cached:
                    local = cached[0]
                    row["cached"] = True
                else:
                    # The filename must come from the ELEMENT's own source, not from the
                    # notebook field. Hardcoding `nb.ipynb` saved every dataset as
                    # `<id>__nb.ipynb`, so DataExtractor routed a CSV or a zip by the wrong
                    # extension and emitted no loader — 30 datasets fetched, 0 units, and
                    # nothing anywhere said why.
                    from urllib.parse import urlparse

                    if args.type == "notebook":
                        name = Path(str(meta.get("notebook-file") or "nb.ipynb")).name
                    else:
                        source = str(meta.get("direct-download-link")
                                     or meta.get("external-link") or "")
                        name = Path(urlparse(source).path).name or "data.bin"
                    src = resolve_and_fetch(meta, cache, element_id=eid,
                                            filename=f"{short}__{name}")
                    local = Path(src.local_path)
                    row.update(sha256=src.sha256, bytes=src.bytes, ref=src.ref)
                stats["fetched"] += 1
            except SourceError as exc:
                # Too large to store is not the same as impossible to describe. A ZIP's central
                # directory is at the end of the file, so its members and a shapefile's CRS come
                # from a couple of ranged reads — 8.5 GB of the platform's own LiDAR, imagery and
                # hydrogeology was being reported as simply unfetchable.
                # A landing page is not a dead end. 36 of the 64 are a DEPOSIT behind a human
                # page (figshare, Zenodo, HydroShare, Dataverse, Hugging Face, GitHub) whose
                # files come from a documented API; 20 are a data PORTAL's front door, where the
                # element is a pointer to a website and no file was ever there. Reporting both as
                # "unfetchable" loses the first and misdescribes the second.
                if exc.kind == "landing_page" and args.type == "dataset":
                    from extractors.dataset_repositories import resolve_files

                    source = str(meta.get("direct-download-link")
                                 or meta.get("external-link") or "")
                    found = resolve_files(source) if source else {"kind": "?", "files": []}
                    row["link_kind"] = found.get("kind")
                    if found.get("files"):
                        biggest = max(found["files"], key=lambda f: f.get("bytes") or 0)
                        row.update(stage="repository", files_listed=len(found["files"]),
                                   primary=str(biggest.get("name"))[:60],
                                   listed_bytes=sum(f.get("bytes") or 0
                                                    for f in found["files"]))
                        stats["resolved_from_repository"] += 1
                        per_element.append(row)
                        continue
                    if found.get("kind") == "portal":
                        stats["external_portal"] += 1
                        row.update(stage="portal", error=None,
                                   note=found.get("note", "")[:110])
                        per_element.append(row)
                        continue
                if exc.kind == "too_large" and args.type == "dataset":
                    source = str(meta.get("direct-download-link")
                                 or meta.get("external-link") or "")
                    remote = None
                    if source:
                        from extractors.data_extractor import (
                            extract_remote_dataset_metadata)

                        remote = extract_remote_dataset_metadata(source)
                    if remote and remote.get("member_count"):
                        stats["described_remotely"] += 1
                        row.update(stage="remote", described_remotely=True,
                                   members=remote.get("member_count"),
                                   crs=remote.get("crs"),
                                   size_mb=round((remote.get("size_bytes") or 0) / 1e6, 1))
                        per_element.append(row)
                        continue
                stats[f"unfetchable:{exc.kind}"] += 1
                row.update(stage="fetch", error=str(exc)[:110])
                per_element.append(row)
                continue

        ctx = ExtractContext(element_id=short, element_type=args.type,
                             source_url=str(meta.get("notebook-url") or ""),
                             fields={"title": el.get("title") or short,
                                     "tags": meta.get("tags") or []},
                             targets=[EMIT_OPENSEARCH, EMIT_LIBRARY])
        # Element-level relevance gate, BEFORE any extraction work. A repository is uniformly
        # relevant or uniformly not, so rejecting one is a single decision rather than 200
        # unit-level ones — and the README is the honest signal, written by a human before
        # anyone thought about extraction. Skipping the raw sweep here is what keeps 1,695 units
        # of ML plumbing out of the library.
        if args.relevance_gate:
            from extractors.relevance import score_element

            verdict = score_element(meta)
            if verdict["score"] < args.relevance_threshold and (
                    verdict["has_readme"] or verdict["chars"] >= 120):
                stats["skipped_off_domain"] += 1
                row.update(stage="relevance", relevance=verdict["score"],
                           off_domain=verdict["off_domain"],
                           error=f"below relevance threshold {args.relevance_threshold}")
                per_element.append(row)
                continue
            row["relevance"] = verdict["score"]

        extractor_cls = _extractor_for(args.type)
        if extractor_cls is None:
            stats[f"no_extractor:{args.type}"] += 1
            row.update(stage="extract", error=f"no extractor for element type {args.type!r}")
            per_element.append(row)
            continue

        targets = _files_for(args.type, local)
        if not targets:
            stats["no_extractable_files"] += 1
            row.update(stage="extract", error="source resolved but held no extractable file")
            per_element.append(row)
            continue

        # A repository yields many files; one bad module must cost that module, not the element.
        from extractors.base import ExtractionResult

        result = ExtractionResult(assets=[], edges=[], warnings=[])
        errors = 0
        for target in targets[: args.max_files]:
            try:
                one = extractor_cls().extract(str(target), ctx=ctx)
            except Exception as exc:
                errors += 1
                if errors == 1:
                    row["first_error"] = f"{type(exc).__name__}: {exc}"[:90]
                continue
            result.assets.extend(one.assets)
            result.edges.extend(one.edges)
            result.warnings.extend(one.warnings)
        row["files"] = len(targets)
        if errors:
            row["file_errors"] = errors
        if not result.assets:
            stats["unparseable"] += 1
            row.update(stage="extract",
                       error=row.get("first_error") or "no assets produced")
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
        print(f"  modules written {len(out.get('written') or [])}")
        # `.get`: the emitter returns early when a manifest carries no library units at all —
        # true for dataset and publication sweeps, whose value is index docs, not importable
        # code. A KeyError there reports a crash where the honest answer is "0 units, as
        # expected for this type".
        print(f"  registry size   {out.get('registry_size', 0)}")
        print(f"  skipped         {len(out.get('skipped') or [])}")
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
