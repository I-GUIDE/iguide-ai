"""OpenSearch emitter — index extracted assets into AGENT-ONLY indices.

For each asset with target ``opensearch``: build a doc reusing the platform
``_source`` fields (doc_id, title, contents, resource-type + inherited
``source_fields`` = tags/authors/contributor/abstract) + an additive ``extracted``
object (block/runnable/file_io/provenance); index into the agent-only index for its
resource-type via ``indices.index_for`` (separate from the general ``OPENSEARCH_INDEX``
so platform search can't see it). Docs are indexed FIRST, embeddings second, so a
down embedder never loses docs.

``build_docs`` is pure/testable (no I/O). ``emit`` does the I/O and accepts an
injected ``client`` (for tests) or a ``dry_run`` flag. The agent's search peer must
query ``indices.all_agent_indices()`` to retrieve these (search-side follow-on).

Embedding text is prose-first (markdown context + title + tools + imports), NOT raw
code, so the shared kNN field retrieves well against natural-language queries.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from ..indices import index_for
from ..manifest import UnifiedManifest

logger = logging.getLogger(__name__)

DocTuple = Tuple[str, str, Dict[str, Any]]  # (index, doc_id, _source)


# --------------------------------------------------------------------------- #
# Pure doc construction (testable without a cluster)
# --------------------------------------------------------------------------- #
def import_line_for(unit: Dict[str, Any]) -> str:
    """``from <module> import <symbol>`` for a unit, or "" when it was not emitted to a library.

    Duplicated deliberately from ``agent_runtime.method_library.import_line``: this module is the
    extraction side and must not import the agent runtime. Keeping it as pure string assembly is
    what lets an index hit be self-sufficient — see ``_unit_payload``.
    """
    module, symbol = unit.get("library_module"), unit.get("library_symbol")
    return f"from {module} import {symbol}" if module and symbol else ""


def _unit_payload(unit: Dict[str, Any]) -> Dict[str, Any]:
    """The contract, mirrored into the document, minus the one unbounded field.

    Measured before this existed: 349 indexed ``method_unit`` documents carried ``unit_name`` and
    ``callable`` and nothing else. Signature, parameter types, declared units, CRS expectations,
    invariants, pip requirements, module and slice_sha — the entire contract, which is the whole
    reason these documents exist — were dropped here, because ``_build_source`` mirrored
    ``block``/``runnable``/``spatial`` and never learned about ``unit``.

    The consequence was not a crash. ``kb_method_search`` reads the on-disk registry, so it kept
    working, and the defect only showed up on the OTHER path: a unit surfaced by
    ``agent_kb_search`` told the agent that a function exists while withholding every fact needed
    to call it. Two stores, one of them lossy, and the lossy one is what general retrieval reads.

    ``docstring`` is the only field left out: it is unbounded, and ``doc_summary`` plus
    ``contents`` already carry its first line. Per-parameter ``evidence`` IS kept — it is short,
    and it is how a reader audits an inferred unit or CRS claim instead of taking it on faith.
    """
    payload = {k: v for k, v in unit.items() if k != "docstring"}
    line = import_line_for(unit)
    if line:
        payload["import_line"] = line
    return payload


def _embed_text(asset: Dict[str, Any]) -> str:
    """Prose-first text to embed (avoid embedding raw code)."""
    block = asset.get("block") or {}
    unit = asset.get("unit") or {}
    parts: List[str] = [asset.get("title") or ""]
    if block:
        parts.append(block.get("markdown_context") or "")
        parts.append(" ".join(block.get("resolved_tools") or []))
        parts.append(" ".join(block.get("imports") or []))
    elif unit:
        # A unit asset used to fall through to `contents` here, so the signature was embedded
        # only by accident (it happens to be the first line of `contents`) and the parameter
        # types, declared units and dependencies were embedded nowhere. "how do I buffer a
        # GeoDataFrame in metres" should match a unit whose parameter is annotated
        # `gpd.GeoDataFrame` and whose declared unit is metres, and that only works if those
        # words are in the embedded text.
        parts.append(unit.get("doc_summary") or "")
        parts.append(unit.get("signature") or "")
        parts.append(" ".join(str(p.get("inferred_type") or "") for p in (unit.get("params") or [])))
        parts.append(" ".join(str(p.get("declared_unit") or "") for p in (unit.get("params") or [])))
        parts.append(" ".join(str(p.get("crs_expectation") or "") for p in (unit.get("params") or [])))
        parts.append(" ".join((unit.get("requirements") or {}).get("pip") or []))
    else:
        parts.append(asset.get("contents") or "")
    text = " ".join(p for p in parts if p).strip()
    return text or (asset.get("contents") or asset.get("title") or "")


# Keys the agent's readers and the reconciler depend on. A platform form field with one of these
# names must not be able to redefine it.
RESERVED_KEYS = ("doc_id", "title", "contents", "resource-type", "element_type", "extracted",
                 "contents-embedding")


def _parent_of(doc_id: str) -> str:
    """The element a derived doc_id hangs off — the doc_ids rule, restated in one place.

    Dataset and publication assets carry no ``extracted.parent_doc_id``: their document IS the
    element, so nothing set it. Reconciliation keys on that field, so those documents could never
    be found by the diff that decides what to delete, and their orphans would have survived
    forever while the run reported ``deleted_orphans: 0``.
    """
    return str(doc_id).split("::", 1)[0]


def _build_source(asset: Dict[str, Any], edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    doc_id = asset["doc_id"]
    # Platform form fields FIRST, canonical keys second. The other order let a submission field
    # named `contents` or `doc_id` overwrite the document's identity, and a doc_id that disagrees
    # with its own _id is unreachable by every reader here.
    src: Dict[str, Any] = {k: v for k, v in (asset.get("source_fields") or {}).items()
                           if k not in RESERVED_KEYS}
    dropped = sorted(set(asset.get("source_fields") or {}) & set(RESERVED_KEYS))
    src.update({
        "doc_id": doc_id,
        "title": asset.get("title") or "",
        "contents": asset.get("contents") or "",
        "resource-type": asset.get("resource_type"),
        "element_type": asset.get("resource_type"),
    })
    # spatial geo_shape, if present
    spatial = asset.get("spatial") or {}
    if spatial.get("spatial-bounding-box-geojson"):
        src["spatial-bounding-box-geojson"] = spatial["spatial-bounding-box-geojson"]
    # agent-specific structured payload (stored, not the general schema). Sub-payloads are
    # omitted when absent rather than written as explicit nulls: every one of the 349 indexed
    # unit docs carried `"block": null, "runnable": null, "spatial": null`, which costs bytes on
    # every read and tells a reader nothing.
    extracted: Dict[str, Any] = {
        **(asset.get("extracted") or {}),
        "kind": asset.get("kind"),
        "source_rel_path": asset.get("source_rel_path"),
        "embed_text": _embed_text(asset),
    }
    extracted.setdefault("parent_doc_id", _parent_of(doc_id))
    for name in ("block", "runnable"):
        if asset.get(name):
            extracted[name] = asset[name]
    if spatial:
        # The envelope is already at the top level as a geo_shape; repeating it inside `extracted`
        # gave the same coordinates a second, dynamically-mapped home and no second reader.
        extracted["spatial"] = {k: v for k, v in spatial.items()
                                if k != "spatial-bounding-box-geojson"}
    if asset.get("unit"):
        extracted["unit"] = _unit_payload(asset["unit"])
    if dropped:
        extracted["source_fields_dropped"] = dropped
    src["extracted"] = extracted
    related = [e for e in edges if e.get("src") == doc_id or e.get("dst") == doc_id]
    if related:
        src["extracted"]["provenance"] = related
    return src


def build_docs(manifest: UnifiedManifest) -> List[DocTuple]:
    """Pure: turn a manifest into (index, doc_id, _source) tuples for OpenSearch."""
    d = manifest.to_dict() if isinstance(manifest, UnifiedManifest) else dict(manifest)
    edges = d.get("provenance_edges") or []
    docs: List[DocTuple] = []
    for asset in d.get("assets") or []:
        if "opensearch" not in (asset.get("emit_targets") or []):
            continue
        index = index_for(asset.get("resource_type"))
        docs.append((index, asset["doc_id"], _build_source(asset, edges)))
    return docs


# --------------------------------------------------------------------------- #
# I/O (live cluster)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _os_client():
    from opensearchpy import OpenSearch

    # Where the agent KB READS (platform_endpoints.search_cluster), so an ingest lands on the
    # cluster the agent will search. Bare names here wrote one cluster while a tiered deployment
    # read another, and sent the untiered credential to a tier's node.
    from agent_runtime.platform_endpoints import search_cluster

    node, user, pwd = search_cluster()
    return OpenSearch(
        hosts=[node],
        http_auth=(user, pwd) if (user or pwd) else None,
        use_ssl=node.lower().startswith("https"),
        verify_certs=False, ssl_assert_hostname=False, ssl_show_warn=False,
        timeout=30, max_retries=2, retry_on_timeout=True,
    )


def _embedding_url() -> str:
    url = os.getenv("FLASK_EMBEDDING_URL", "http://127.0.0.1:5000")
    return url if url.rstrip("/").endswith("get_embedding") else url.rstrip("/") + "/get_embedding"


def _embed_dim() -> int:
    return int(os.getenv("AGENT_KB_EMBED_DIM", "384"))  # all-MiniLM-L6-v2


def index_mapping() -> Dict[str, Any]:
    """The mapping an agent index should have.

    Everything the code TERM-queries or filters on is declared here as ``keyword``. Left to
    dynamic mapping a string becomes ``text`` + ``.keyword``, and a term query against analyzed
    text matches only when the value happens to survive the standard analyzer as a single token.
    Measured on the live cluster: ``extracted.parent_doc_id`` is dynamically mapped as ``text``,
    and the reconciler's term query finds 73 of 73 parents purely because this corpus's element
    ids are 8-character lowercase hex. The same query against a full platform UUID
    (``cca9b545-3b1e-…``) would be tokenized on the hyphens and match nothing — and the failure
    mode is an empty orphan set reported as a clean index, not an error.

    ``extracted`` stays dynamic below these paths: the contract grows new fields as the analyzer
    improves, and a strict mapping would reject the document rather than store the new field.
    """
    return {
        "settings": {"index": {"knn": True}},
        "mappings": {"properties": {
            "doc_id": {"type": "keyword"},
            "resource-type": {"type": "keyword"},
            "element_type": {"type": "keyword"},
            "title": {"type": "text"},
            "contents": {"type": "text"},
            "tags": {"type": "keyword"},
            "spatial-bounding-box-geojson": {"type": "geo_shape"},
            "contents-embedding": {"type": "knn_vector", "dimension": _embed_dim()},
            "extracted": {"type": "object", "properties": {
                "parent_doc_id": {"type": "keyword"},   # the reconciler's term query
                "parent_type": {"type": "keyword"},
                "kind": {"type": "keyword"},
                "callable": {"type": "boolean"},        # kb_method_search's filter
                "unit_name": {"type": "keyword"},
                "status": {"type": "keyword"},          # publication degradation status
                "degraded": {"type": "boolean"},
                "embed_text": {"type": "text"},
                "unit": {"type": "object", "properties": {
                    "qualified_name": {"type": "keyword"},
                    "library_symbol": {"type": "keyword"},
                    "library_module": {"type": "keyword"},
                    "slice_sha": {"type": "keyword"},
                    "unit_kind": {"type": "keyword"},
                    "signature": {"type": "text"},
                    "doc_summary": {"type": "text"},
                    "import_line": {"type": "keyword"},
                }},
            }},
        }},
    }


def _flatten_mapping(props: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name, spec in (props or {}).items():
        path = f"{prefix}{name}"
        if isinstance(spec, dict) and spec.get("properties"):
            out[path] = spec.get("type") or "object"
            out.update(_flatten_mapping(spec["properties"], path + "."))
        elif isinstance(spec, dict):
            out[path] = str(spec.get("type") or "?")
    return out


def mapping_drift(client, index: str) -> Dict[str, Any]:
    """Fields whose LIVE mapping contradicts ``index_mapping()``.

    An existing index cannot have a ``text`` field changed to ``keyword`` in place, so this
    reports rather than repairs. It exists because ``ensure_index`` returns early when the index
    is present: without this, a schema change lands in the code, silently fails to apply to the
    four indices that already exist, and every subsequent measurement is taken against a mapping
    nobody is looking at.
    """
    try:
        if not client.indices.exists(index=index):
            return {"index": index, "exists": False, "drift": {}}
        live = _flatten_mapping(
            (client.indices.get_mapping(index=index)[index]["mappings"].get("properties") or {}))
    except Exception as exc:                                    # pragma: no cover - cluster only
        return {"index": index, "error": str(exc)[:200], "drift": {}}
    want = _flatten_mapping(index_mapping()["mappings"]["properties"])
    drift = {path: {"want": kind, "live": live[path]}
             for path, kind in want.items()
             if path in live and live[path] != kind}
    missing = sorted(p for p in want if p not in live)
    return {"index": index, "exists": True, "drift": drift, "not_yet_present": missing}


def ensure_index(client, index: str) -> None:
    """Create the agent index with a kNN mapping for contents-embedding if missing.

    When the index already exists its mapping is checked, not assumed. Drift is logged as a
    warning naming the fields and the reindex it needs — a term-queried field that is silently
    ``text`` returns zero hits, which reads exactly like "nothing to do".
    """
    if client.indices.exists(index=index):
        report = mapping_drift(client, index)
        if report.get("drift"):
            logger.warning(
                "index %s mapping drift on %d field(s): %s. These are term-queried; until the "
                "index is recreated (scripts/create_agent_indices.py --recreate) those queries "
                "can silently return nothing.",
                index, len(report["drift"]),
                ", ".join(f"{k}: live={v['live']} want={v['want']}"
                          for k, v in sorted(report["drift"].items())[:6]))
        return
    client.indices.create(index=index, body=index_mapping())


def _get_embedding(text: str) -> Optional[List[float]]:
    import requests
    try:
        r = requests.post(_embedding_url(), json={"text": text}, timeout=30)
        r.raise_for_status()
        return r.json().get("embedding")
    except Exception:
        return None



# --------------------------------------------------------------------------- #
# Reconciliation. Re-ingest is not "write the new docs" — it is "make the index
# match what the source produces NOW", which means deleting what it no longer does.
# --------------------------------------------------------------------------- #

INGEST_RUNS_INDEX = "iguide_agent_ingest_runs"
SCHEMA_VERSION = 1


def _assert_agent_indices(indices) -> None:
    """Refuse to touch anything that is not an agent index.

    This module deletes documents. A misconfigured ``AGENT_KB_INDEX_PREFIX`` that happened to
    resolve to ``OPENSEARCH_INDEX`` would otherwise let a re-ingest delete platform records,
    which is unrecoverable from here. Cheap assertion, catastrophic omission.
    """
    from ..indices import is_agent_index

    # The platform's index under EVERY name it goes by: the bare variable and the search tier's
    # (`search_index`). Checking only the bare one guarded the wrong index on a tiered deployment.
    general = {os.getenv("OPENSEARCH_INDEX") or ""}
    try:
        from agent_runtime.platform_endpoints import search_index

        general.add(search_index())
    except Exception:  # a bad SEARCH_TIER must not disarm the guard; the bare name still counts
        pass
    general.discard("")
    for name in indices:
        if not is_agent_index(name):
            raise RuntimeError(f"refusing to write/delete in non-agent index {name!r}")
        if name in general:
            raise RuntimeError(f"agent index {name!r} collides with OPENSEARCH_INDEX")


def existing_doc_ids(client, index: str, parent_doc_id: str, *, limit: int = 10000) -> set:
    """Every doc_id currently indexed under one parent element.

    Scoped by ``extracted.parent_doc_id`` and read BEFORE writing, so the diff is against what
    is really there rather than what we assume we put there last time.
    """
    try:
        if not client.indices.exists(index=index):
            return set()
        resp = client.search(index=index, body={
            "size": limit, "_source": ["doc_id"],
            "query": {"term": {"extracted.parent_doc_id": parent_doc_id}}})
    except Exception as exc:
        logger.warning("could not list existing docs for %s in %s: %s", parent_doc_id, index, exc)
        return set()
    out = set()
    for hit in (resp.get("hits", {}).get("hits") or []):
        doc_id = (hit.get("_source") or {}).get("doc_id") or hit.get("_id")
        if doc_id:
            out.add(str(doc_id))
    return out


def reconcile_plan(client, docs) -> Dict[str, Any]:
    """What to write and what to DELETE, per (index, parent).

    Without this, an element that loses a cell leaves its old ``::block::<n>`` documents in the
    index forever — there is no delete anywhere else in this repo — and they keep being
    retrieved as evidence for code that no longer exists.
    """
    produced: Dict[tuple, set] = {}
    for index, doc_id, source in docs:
        parent = ((source.get("extracted") or {}).get("parent_doc_id")
                  or source.get("doc_id") or doc_id)
        produced.setdefault((index, str(parent)), set()).add(str(doc_id))

    orphans: Dict[str, set] = {}
    for (index, parent), ids in produced.items():
        stale = existing_doc_ids(client, index, parent) - ids
        if stale:
            orphans.setdefault(index, set()).update(stale)
    return {"produced": produced, "orphans": orphans,
            "orphan_count": sum(len(v) for v in orphans.values())}


def _bulk_write(client, docs) -> int:
    """One bulk request per batch instead of one HTTP round trip per document.

    The corpus backfill wrote 4,179 docs as 4,179 index calls plus 4,179 updates for the
    embeddings — ~8,300 round trips where a handful of bulk requests will do.
    """
    from opensearchpy import helpers

    actions = [{"_op_type": "index", "_index": index, "_id": doc_id, "_source": source}
               for index, doc_id, source in docs]
    if not actions:
        return 0
    ok, errors = helpers.bulk(client, actions, raise_on_error=False, stats_only=False)
    for err in (errors or [])[:5]:
        logger.warning("bulk index error: %s", str(err)[:300])
    return int(ok)


def _bulk_embed(client, docs) -> int:
    from opensearchpy import helpers

    actions = []
    for index, doc_id, source in docs:
        text = (source.get("extracted") or {}).get("embed_text") or source.get("contents") or ""
        vec = _get_embedding(text)
        if vec is None:
            continue
        actions.append({"_op_type": "update", "_index": index, "_id": doc_id,
                        "doc": {"contents-embedding": vec}})
    if not actions:
        return 0
    ok, errors = helpers.bulk(client, actions, raise_on_error=False, stats_only=False)
    for err in (errors or [])[:5]:
        logger.warning("bulk embed error: %s", str(err)[:300])
    return int(ok)


def _delete_orphans(client, orphans: Dict[str, set]) -> int:
    from opensearchpy import helpers

    actions = [{"_op_type": "delete", "_index": index, "_id": doc_id}
               for index, ids in orphans.items() for doc_id in ids]
    if not actions:
        return 0
    # Deleting by explicit id rather than delete_by_query: the ids come from a diff we just
    # computed, so there is no query that could match more than intended.
    ok, errors = helpers.bulk(client, actions, raise_on_error=False, stats_only=False)
    for err in (errors or [])[:5]:
        logger.warning("bulk delete error: %s", str(err)[:300])
    return int(ok)


def run_fingerprint(manifest) -> str:
    """Content fingerprint of everything a manifest would write.

    Keyed on the DOCS, not on a commit sha: a re-ingest of the same commit through a changed
    extractor must not be skipped, and that is exactly when skipping would hide a regression.
    """
    import hashlib

    parts = []
    for index, doc_id, source in build_docs(manifest):
        body = json.dumps(source, sort_keys=True, default=str)
        parts.append(f"{index}|{doc_id}|{hashlib.sha1(body.encode()).hexdigest()}")
    blob = "\n".join(sorted(parts))
    return f"v{SCHEMA_VERSION}-" + hashlib.sha256(blob.encode()).hexdigest()[:16]


def previous_run(client, element_id: str) -> Optional[Dict[str, Any]]:
    try:
        if not client.indices.exists(index=INGEST_RUNS_INDEX):
            return None
        resp = client.get(index=INGEST_RUNS_INDEX, id=element_id)
        return resp.get("_source") or None
    except Exception:
        return None


def record_run(client, element_id: str, fingerprint: str, summary: Dict[str, Any]) -> None:
    import datetime as _dt

    try:
        if not client.indices.exists(index=INGEST_RUNS_INDEX):
            client.indices.create(index=INGEST_RUNS_INDEX, body={"mappings": {"properties": {
                "element_id": {"type": "keyword"}, "fingerprint": {"type": "keyword"},
                "schema_version": {"type": "integer"}, "at": {"type": "date"}}}})
        client.index(index=INGEST_RUNS_INDEX, id=element_id, body={
            "element_id": element_id, "fingerprint": fingerprint,
            "schema_version": SCHEMA_VERSION,
            "at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "doc_count": summary.get("doc_count"), "indices": summary.get("indices")})
    except Exception as exc:
        logger.warning("could not record ingest run for %s: %s", element_id, exc)


def emit(manifest: UnifiedManifest, *, client=None, embed: bool = True,
         dry_run: bool = False, reconcile: bool = True,
         skip_unchanged: bool = True) -> Dict[str, Any]:
    """Index extracted assets into the agent KB. Backend defaults to LOCAL
    (file-backed) — nothing reaches the real OpenSearch unless AGENT_KB_BACKEND=
    opensearch (or a client is injected). Returns a summary.

    ``skip_unchanged`` re-checks the element's last run and does nothing when this run would write
    byte-identical documents. ``run_fingerprint`` / ``previous_run`` / ``record_run`` were written
    and tested for exactly this and then never called from anywhere but the tests, so every
    re-index rewrote the whole corpus and ``iguide_agent_ingest_runs`` did not exist on the
    cluster at all. Pass ``skip_unchanged=False`` to force a rewrite.
    """
    from .. import kb_store

    docs = build_docs(manifest)
    by_index: Dict[str, int] = {}
    for index, _id, _src in docs:
        by_index[index] = by_index.get(index, 0) + 1

    if dry_run:
        return {"dry_run": True, "backend": "dry_run", "doc_count": len(docs),
                "indices": by_index, "doc_ids": [d[1] for d in docs]}

    # LOCAL backend (default): write to the file-backed store, never the real DB.
    if client is None and kb_store.kb_backend() != "opensearch":
        for index, doc_id, source in docs:
            kb_store.local_upsert(index, doc_id, source)
        return {"dry_run": False, "backend": "local", "doc_count": len(docs),
                "indices": by_index, "indexed": len(docs), "embedded": 0,
                "store_dir": str(kb_store.store_dir())}

    # OpenSearch backend (explicit opt-in / injected client).
    client = client or _os_client()
    _assert_agent_indices(by_index)

    element_id = str(getattr(manifest, "element_id", "") or "")
    fingerprint = run_fingerprint(manifest)
    if skip_unchanged and element_id and docs:
        prior = previous_run(client, element_id)
        if (prior and prior.get("fingerprint") == fingerprint
                and int(prior.get("schema_version") or -1) == SCHEMA_VERSION):
            # A matching fingerprint means "this run would write the same documents", NOT "those
            # documents are in the index". An index that was recreated or wiped would otherwise
            # stay empty forever, with every re-ingest reporting a clean skip. So confirm the
            # documents are actually there before trusting the record — one search per index.
            present = set()
            for index in by_index:
                for parent in {str((src.get("extracted") or {}).get("parent_doc_id")
                                   or src.get("doc_id") or doc_id)
                               for i, doc_id, src in docs if i == index}:
                    present |= existing_doc_ids(client, index, parent)
            expected = {doc_id for _i, doc_id, _s in docs}
            if expected <= present:
                return {"dry_run": False, "backend": "opensearch", "skipped": True,
                        "reason": "unchanged since the last run", "doc_count": len(docs),
                        "indices": by_index, "indexed": 0, "embedded": 0,
                        "fingerprint": fingerprint}
            logger.info("fingerprint matches for %s but %d of %d documents are missing from the "
                        "index — rewriting", element_id, len(expected - present), len(expected))

    # Diff BEFORE writing: the orphan set is (what is indexed now) - (what we are about to
    # write), so it has to be read while the old state is still there.
    plan = reconcile_plan(client, docs) if reconcile else {"orphans": {}, "orphan_count": 0}

    for index in by_index:
        ensure_index(client, index)
    # 1) index docs first (so a down embedder never loses docs)
    indexed = _bulk_write(client, docs)
    # 2) embed second
    embedded = _bulk_embed(client, docs) if embed else 0
    # 3) delete what this element no longer produces
    deleted = _delete_orphans(client, plan.get("orphans") or {}) if reconcile else 0

    if element_id:
        # Written AFTER the docs land, so a crashed run does not record a success that would make
        # the next run skip it.
        record_run(client, element_id, fingerprint,
                   {"doc_count": len(docs), "indices": by_index})

    return {"dry_run": False, "backend": "opensearch", "skipped": False,
            "doc_count": len(docs), "indices": by_index, "indexed": indexed,
            "embedded": embedded, "deleted_orphans": deleted,
            "orphans_found": plan.get("orphan_count", 0), "fingerprint": fingerprint}


__all__ = ["build_docs", "emit", "ensure_index", "index_mapping", "mapping_drift",
           "import_line_for", "RESERVED_KEYS"]
