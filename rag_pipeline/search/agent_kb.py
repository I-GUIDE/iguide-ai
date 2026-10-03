"""Agent knowledge-base search — retrieval over the AGENT-ONLY indices.

The ingestion pipeline (``extractors/``) writes fine-grained, runnable-aware docs
(notebook blocks, code assets, dataset metadata, publication method-specs) into
separate ``iguide_agent_*`` indices, invisible to general platform search. This
module is the agent's read path into them: keyword (BM25) + semantic (kNN) over those
indices, with every hit linked back to its **original knowledge element** via the
``element_id`` anchor.

Design notes:
- kNN stays *within* the agent indices (all built at AGENT_KB_EMBED_DIM), so there is
  no cross-index dimension mismatch with the general index.
- Pure helpers (``build_keyword_query`` / ``build_knn_query`` / ``normalize_hits`` /
  ``group_by_parent``) are testable without a cluster; ``agent_kb_search`` does the I/O
  and accepts an injected ``client``.
- Failures degrade to an empty result with a note (agent tools must not raise).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# Pure query / normalization helpers (no I/O)
# --------------------------------------------------------------------------- #
def build_keyword_query(query: str, size: int) -> Dict[str, Any]:
    return {
        "size": size,
        "query": {"multi_match": {"query": query, "fields": ["title^2", "contents", "extracted.embed_text"]}},
    }


def build_knn_query(vector: List[float], size: int) -> Dict[str, Any]:
    return {"size": size, "query": {"knn": {"contents-embedding": {"vector": vector, "k": size}}}}


def _parent_of(doc_id: str, source: Dict[str, Any]) -> str:
    extracted = source.get("extracted") or {}
    if extracted.get("parent_doc_id"):
        return str(extracted["parent_doc_id"])
    return doc_id.split("::", 1)[0] if "::" in doc_id else doc_id


def normalize_hit(hit: Dict[str, Any], matched: str) -> Dict[str, Any]:
    source = hit.get("_source") or {}
    doc_id = str(source.get("doc_id") or hit.get("_id") or "")
    extracted = source.get("extracted") or {}
    runnable = (extracted.get("runnable") or {}) if isinstance(extracted, dict) else {}
    return {
        "doc_id": doc_id,
        "source_index": hit.get("_index"),
        "parent_doc_id": _parent_of(doc_id, source),
        "resource_type": source.get("resource-type") or source.get("element_type"),
        "title": source.get("title") or "Untitled",
        # keep enough to carry a full function/method body for verbatim reuse
        "contents": (source.get("contents") or "")[:4000],
        "resolved_tools": (extracted.get("block") or {}).get("resolved_tools") if isinstance(extracted, dict) else None,
        "runnable_tool": runnable.get("runnable_tool"),
        "score": hit.get("_score", 0.0),
        "matched": matched,
    }


def normalize_hits(keyword_hits: List[Dict[str, Any]],
                   semantic_hits: List[Dict[str, Any]], size: int) -> List[Dict[str, Any]]:
    """Merge keyword + semantic hits, dedup by doc_id (keyword first), cap at size."""
    by_id: Dict[str, Dict[str, Any]] = {}
    for hit in keyword_hits:
        n = normalize_hit(hit, "keyword")
        if n["doc_id"] and n["doc_id"] not in by_id:
            by_id[n["doc_id"]] = n
    for hit in semantic_hits:
        n = normalize_hit(hit, "semantic")
        if not n["doc_id"]:
            continue
        if n["doc_id"] in by_id:
            by_id[n["doc_id"]]["matched"] = "keyword+semantic"
        else:
            by_id[n["doc_id"]] = n
    return list(by_id.values())[:size]


def group_by_parent(docs: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for d in docs:
        out.setdefault(d["parent_doc_id"], []).append(d["doc_id"])
    return out


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _os_client():
    from opensearchpy import OpenSearch
    node = os.getenv("OPENSEARCH_NODE", "")
    user = os.getenv("OPENSEARCH_USERNAME", "")
    pwd = os.getenv("OPENSEARCH_PASSWORD", "")
    return OpenSearch(
        hosts=[node], http_auth=(user, pwd) if (user or pwd) else None,
        use_ssl=node.lower().startswith("https"), verify_certs=False,
        ssl_assert_hostname=False, ssl_show_warn=False, timeout=30,
        max_retries=2, retry_on_timeout=True,
    )


def _agent_index_target() -> str:
    from extractors.indices import all_agent_indices
    return ",".join(all_agent_indices())


def _embedding(text: str) -> Optional[List[float]]:
    import requests
    url = os.getenv("FLASK_EMBEDDING_URL", "http://127.0.0.1:5000")
    if not url.rstrip("/").endswith("get_embedding"):
        url = url.rstrip("/") + "/get_embedding"
    try:
        r = requests.post(url, json={"text": text}, timeout=30)
        r.raise_for_status()
        return r.json().get("embedding")
    except Exception:
        return None


def resolve_parent_elements(docs: List[Dict[str, Any]], client) -> Dict[str, Dict[str, Any]]:
    """Fetch the ORIGINAL knowledge elements (general index) for the parents of these
    docs, keyed by element_id, for citation/context."""
    parent_ids = sorted({d["parent_doc_id"] for d in docs if d.get("parent_doc_id")})
    if not parent_ids:
        return {}
    # Through the shared helper so it follows SEARCH_TIER like every other index read; a
    # direct os.getenv here would have been the one module still querying the other tier.
    from .utils import getenv as _tiered
    general = _tiered("OPENSEARCH_INDEX", required=False, default="")
    if not general:
        return {}
    try:
        resp = client.search(index=general, body={
            "size": len(parent_ids),
            "query": {"terms": {"doc_id": parent_ids}},
        })
    except Exception:
        return {}
    elements: Dict[str, Dict[str, Any]] = {}
    for hit in (resp.get("hits", {}).get("hits", []) or []):
        s = hit.get("_source") or {}
        eid = str(s.get("doc_id") or hit.get("_id") or "")
        elements[eid] = {
            "element_id": eid,
            "title": s.get("title") or s.get("name"),
            "authors": s.get("authors"),
            "contributor": s.get("contributor"),
            "resource-type": s.get("resource-type"),
        }
    return elements


def resolve_parent_elements_local(hits: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """LOCAL parent resolution: derive the original-element stub from each block's own
    stored fields (the original element is not separately stored in local mode)."""
    elements: Dict[str, Dict[str, Any]] = {}
    for h in hits:
        s = h.get("_source") or {}
        ex = s.get("extracted") or {}
        parent = ex.get("parent_doc_id") or (str(s.get("doc_id") or "").split("::", 1)[0])
        if parent and parent not in elements:
            elements[parent] = {
                "element_id": parent,
                "title": ex.get("parent_title") or None,
                "authors": s.get("authors"),
                "resource-type": ex.get("parent_type"),
            }
    return elements


def agent_kb_search(query: str, *, size: int = 8, client=None, embed: bool = True,
                    resolve_parents: bool = True) -> Dict[str, Any]:
    """Search the agent KB; return normalized, parent-linked evidence.

    Backend is LOCAL by default (file-backed store) so this runs offline and never
    touches the real OpenSearch; set AGENT_KB_BACKEND=opensearch (or inject a client)
    to use the cluster. Never raises (agent tools must return, not throw)."""
    base = {"source": "agent_kb", "count": 0, "documents": [], "citation_ids": [], "elements": {}}
    try:
        # Imported inside the try so a missing/unpackaged `extractors` degrades to a
        # benign note instead of crashing the agent turn.
        from extractors import kb_store
        from extractors.indices import all_agent_indices
        use_opensearch = client is not None or kb_store.kb_backend() == "opensearch"
        if not use_opensearch:
            hits = kb_store.local_search(query, all_agent_indices(), size)
            docs = normalize_hits(hits, [], size)
            elements = resolve_parent_elements_local(hits) if resolve_parents else {}
        else:
            if client is None and not os.getenv("OPENSEARCH_NODE"):
                return {**base, "note": "AGENT_KB_BACKEND=opensearch but OPENSEARCH_NODE not set"}
            client = client or _os_client()
            index = ",".join(all_agent_indices())
            kw = client.search(index=index, body=build_keyword_query(query, size))
            kw_hits = kw.get("hits", {}).get("hits", []) or []
            sem_hits: List[Dict[str, Any]] = []
            if embed:
                vec = _embedding(query)
                if vec:
                    sem = client.search(index=index, body=build_knn_query(vec, size))
                    sem_hits = sem.get("hits", {}).get("hits", []) or []
            docs = normalize_hits(kw_hits, sem_hits, size)
            elements = resolve_parent_elements(docs, client) if resolve_parents else {}
        if elements:
            for d in docs:
                d["element"] = elements.get(d["parent_doc_id"])
        return {
            "source": "agent_kb", "backend": ("opensearch" if use_opensearch else "local"),
            "count": len(docs), "documents": docs,
            "citation_ids": [d["parent_doc_id"] for d in docs],   # cite the ORIGINAL element
            "block_ids": [d["doc_id"] for d in docs],
            "elements": elements,
        }
    except Exception as exc:
        return {**base, "note": f"agent_kb_search error: {type(exc).__name__}: {exc}"}


def _element_block_bundle(element_id: str, blocks: List) -> Dict[str, Any]:
    """Synthesize a single 'whole-element' doc from its blocks (code concatenated in
    order), so a bare element_id resolves to the full notebook source for reuse."""
    parts: List[str] = []
    block_ids: List[str] = []
    title = element_id
    for did, src in blocks:
        block_ids.append(did)
        if src.get("title"):
            title = src["title"]
        code = ((src.get("extracted") or {}).get("block") or {}).get("code") or ""
        if code:
            parts.append(f"# --- {did} ---\n{code}")
    return {
        "doc_id": element_id, "found": True, "is_element": True, "block_ids": block_ids,
        "source": {"doc_id": element_id, "title": title,
                   "extracted": {"block": {"code": "\n\n".join(parts)}}},
    }


def get_kb_block(doc_id: str, *, client=None) -> Dict[str, Any]:
    """Fetch the FULL stored agent-KB doc by id (incl. extracted.block.code).

    Accepts either a block doc_id (``{element_id}::block::{n}``) OR a bare
    ``element_id`` — in the latter case the element's blocks are concatenated into one
    whole-notebook source (the consumer often only knows the cited element_id). The
    search/evidence view truncates contents; this returns the complete code/method body
    for verbatim reuse. Local by default; never raises."""
    try:
        from extractors import kb_store
        from extractors.indices import all_agent_indices
        indices = all_agent_indices()
        use_opensearch = client is not None or kb_store.kb_backend() == "opensearch"
        if not use_opensearch:
            idx, src = kb_store.local_get(doc_id, indices)
            if src is not None:
                return {"doc_id": doc_id, "found": True, "index": idx, "source": src}
            # bare element_id -> bundle all its blocks
            blocks = kb_store.local_blocks_for_parent(doc_id, indices)
            if blocks:
                return _element_block_bundle(doc_id, blocks)
            return {"doc_id": doc_id, "found": False}
        client = client or _os_client()
        for idx in indices:
            try:
                resp = client.get(index=idx, id=doc_id)
                if resp.get("found"):
                    return {"doc_id": doc_id, "found": True, "index": idx, "source": resp.get("_source")}
            except Exception:
                continue
        # bare element_id -> search blocks whose parent is this element, then bundle
        try:
            resp = client.search(index=",".join(indices), body={"size": 300, "query": {"bool": {"should": [
                {"prefix": {"doc_id": f"{doc_id}::block::"}},
                {"term": {"extracted.parent_doc_id": doc_id}},
            ]}}})
            hits = resp.get("hits", {}).get("hits", []) or []
            blocks = [(str((h.get("_source") or {}).get("doc_id") or h.get("_id")), h.get("_source") or {}) for h in hits]
            blocks.sort(key=lambda b: int(b[0].rsplit("::", 1)[-1]) if b[0].rsplit("::", 1)[-1].isdigit() else 9999)
            if blocks:
                return _element_block_bundle(doc_id, blocks)
        except Exception:
            pass
        return {"doc_id": doc_id, "found": False}
    except Exception as exc:
        return {"doc_id": doc_id, "found": False, "note": f"{type(exc).__name__}: {exc}"}


__all__ = [
    "agent_kb_search", "get_kb_block", "build_keyword_query", "build_knn_query", "normalize_hit",
    "normalize_hits", "group_by_parent", "resolve_parent_elements", "resolve_parent_elements_local",
]
