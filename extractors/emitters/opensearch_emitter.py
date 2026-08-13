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
def _embed_text(asset: Dict[str, Any]) -> str:
    """Prose-first text to embed (avoid embedding raw code)."""
    block = asset.get("block") or {}
    parts: List[str] = [asset.get("title") or ""]
    if block:
        parts.append(block.get("markdown_context") or "")
        parts.append(" ".join(block.get("resolved_tools") or []))
        parts.append(" ".join(block.get("imports") or []))
    else:
        parts.append(asset.get("contents") or "")
    text = " ".join(p for p in parts if p).strip()
    return text or (asset.get("contents") or asset.get("title") or "")


def _build_source(asset: Dict[str, Any], edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    doc_id = asset["doc_id"]
    src: Dict[str, Any] = {
        "doc_id": doc_id,
        "title": asset.get("title") or "",
        "contents": asset.get("contents") or "",
        "resource-type": asset.get("resource_type"),
        "element_type": asset.get("resource_type"),
    }
    # inherited platform form fields (tags/authors/contributor/abstract/...) at top level
    src.update(asset.get("source_fields") or {})
    # spatial geo_shape, if present
    spatial = asset.get("spatial") or {}
    if spatial.get("spatial-bounding-box-geojson"):
        src["spatial-bounding-box-geojson"] = spatial["spatial-bounding-box-geojson"]
    # agent-specific structured payload (stored, not the general schema)
    src["extracted"] = {
        **(asset.get("extracted") or {}),
        "kind": asset.get("kind"),
        "source_rel_path": asset.get("source_rel_path"),
        "block": asset.get("block"),
        "runnable": asset.get("runnable"),
        "spatial": spatial or None,
        "embed_text": _embed_text(asset),
    }
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
    node = os.getenv("OPENSEARCH_NODE", "")
    user = os.getenv("OPENSEARCH_USERNAME", "")
    pwd = os.getenv("OPENSEARCH_PASSWORD", "")
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


def ensure_index(client, index: str) -> None:
    """Create the agent index with a kNN mapping for contents-embedding if missing."""
    if client.indices.exists(index=index):
        return
    body = {
        "settings": {"index": {"knn": True}},
        "mappings": {"properties": {
            "doc_id": {"type": "keyword"},
            "resource-type": {"type": "keyword"},
            "element_type": {"type": "keyword"},
            "title": {"type": "text"},
            "contents": {"type": "text"},
            "spatial-bounding-box-geojson": {"type": "geo_shape"},
            "extracted": {"type": "object", "enabled": True},
            "contents-embedding": {"type": "knn_vector", "dimension": _embed_dim()},
        }},
    }
    client.indices.create(index=index, body=body)


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

    general = os.getenv("OPENSEARCH_INDEX") or ""
    for name in indices:
        if not is_agent_index(name):
            raise RuntimeError(f"refusing to write/delete in non-agent index {name!r}")
        if general and name == general:
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
         dry_run: bool = False, reconcile: bool = True) -> Dict[str, Any]:
    """Index extracted assets into the agent KB. Backend defaults to LOCAL
    (file-backed) — nothing reaches the real OpenSearch unless AGENT_KB_BACKEND=
    opensearch (or a client is injected). Returns a summary."""
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

    return {"dry_run": False, "backend": "opensearch", "doc_count": len(docs),
            "indices": by_index, "indexed": indexed, "embedded": embedded,
            "deleted_orphans": deleted, "orphans_found": plan.get("orphan_count", 0)}


__all__ = ["build_docs", "emit", "ensure_index"]
