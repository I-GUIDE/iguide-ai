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
from .utils import default_top_k  # shared retrieval window
from agent_runtime.extraction_flag import extraction_enabled


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


def _method_payload(extracted: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The part of a unit's contract that changes what the agent DOES, or None.

    This is the last hop of a chain that was broken at every link. The extractor computed the
    contract; the emitter dropped it (fixed); the fan-out ran before the module path existed
    (fixed); and this function — which builds the payload the agent actually reads — carried
    ``title``, ``contents`` and ``runnable_tool`` and nothing about the unit. So a method surfaced
    by ``agent_kb_search`` still arrived as a name with no signature, no dependencies and no import
    line, and the agent's only options were to guess or to fall back to re-implementing it.

    Deliberately a projection, not the whole contract: this payload goes into a token-limited
    evidence view, so it carries what is needed to CALL the unit and leaves the per-parameter
    inference evidence to ``get_method_contract``.
    """
    unit = extracted.get("unit") if isinstance(extracted, dict) else None
    if not isinstance(unit, dict) or not unit:
        return None
    # Only the parameters that carry a PRECONDITION, plus a count of the rest.
    #
    # This function's own docstring says it "leaves the per-parameter inference evidence to
    # get_method_contract", and it did not: it shipped annotation, inferred_type, declared_unit,
    # crs_expectation and required for every parameter, at first contact, while the agent is
    # still deciding whether the method is relevant. For an 8-parameter unit that is a wall of
    # JSON restating what `signature` already says. Measured: ~3,800 tokens per search, most of
    # it this.
    #
    # What survives is what the SIGNATURE cannot express — a unit expectation or a CRS
    # expectation is a precondition the caller must satisfy or get a plausible wrong number.
    all_params = [p for p in (unit.get("params") or []) if isinstance(p, dict)]
    params = [{k: p.get(k) for k in ("name", "declared_unit", "crs_expectation")
               if p.get(k) not in (None, "")}
              for p in all_params if p.get("declared_unit") or p.get("crs_expectation")]
    payload = {
        "symbol": unit.get("library_symbol") or unit.get("qualified_name"),
        "signature": unit.get("signature"),
        "doc_summary": unit.get("doc_summary"),
        "params_with_preconditions": params,
        "param_count": len(all_params) or None,
        "returns": unit.get("returns"),
        "invariants": [i.get("check") for i in (unit.get("invariants") or [])
                       if isinstance(i, dict) and i.get("check")],
        "requirements": (unit.get("requirements") or {}).get("pip") or [],
        "import_line": unit.get("import_line"),
        "slice_sha": unit.get("slice_sha"),
    }
    payload = {k: v for k, v in payload.items() if v not in (None, "", [], {})}
    if not extraction_enabled():
        # The extraction bundle is off (agent_runtime/extraction_flag.py), so no library is
        # mounted and nothing here can be imported. What the unit IS stays — it is true of the
        # element in any deployment. What it would take to CALL it goes: a model handed an import
        # line writes the import, and the import fails.
        for key in ("import_line", "slice_sha"):
            payload.pop(key, None)
        return payload

    # Callability is three-valued, and flattening it to a bool was wrong in both directions. An
    # ABSENT verdict is not "not callable" — it means nothing analyzed it — and reporting False
    # there invites the agent to skip a usable unit; while a `needs_globals` verdict reported as a
    # bare False loses the one thing that makes it actionable, which is WHY. The same
    # fail/cannot-determine distinction the invariant gate makes.
    verdict = (unit.get("callability") or {}).get("verdict")
    if verdict:
        payload["callable"] = verdict == "callable"
        if verdict != "callable":
            payload["not_callable"] = verdict
            reason = (unit.get("callability") or {}).get("reason")
            if reason:
                # "reads the module-level global PARAMS" tells the agent to pass it as an
                # argument. Without it the unit just looks broken.
                payload["not_callable_reason"] = str(reason)[:200]
    return payload



# Set from the corpus, not guessed. 3,830 cells: median 328 characters, p75 894, p90 2,096.
# A first guess of 700 cut off inside the p75 cell, and the agent said so exactly — "the
# relevant explanatory cell is truncated before the weight values appear" — losing the numbers
# that were the point of the retrieval. 1,800 keeps three quarters of cells whole and still
# roughly halves the old 4,000.
_EXCERPT_CHARS = 1800


def _excerpt(text: str) -> str:
    """First contact carries enough to judge relevance; `get_kb_block` carries the body."""
    text = str(text or "")
    if len(text) <= _EXCERPT_CHARS:
        return text
    return text[:_EXCERPT_CHARS].rstrip() + f"\n… [{len(text) - _EXCERPT_CHARS} more chars — call get_kb_block for the full cell]"


def normalize_hit(hit: Dict[str, Any], matched: str) -> Dict[str, Any]:
    source = hit.get("_source") or {}
    doc_id = str(source.get("doc_id") or hit.get("_id") or "")
    extracted = source.get("extracted") or {}
    runnable = (extracted.get("runnable") or {}) if isinstance(extracted, dict) else {}
    method = _method_payload(extracted)
    return {
        **({"method": method} if method else {}),
        "doc_id": doc_id,
        "source_index": hit.get("_index"),
        "parent_doc_id": _parent_of(doc_id, source),
        "resource_type": source.get("resource-type") or source.get("element_type"),
        "title": source.get("title") or "Untitled",
        # An EXCERPT at first contact, not the body.
        #
        # This carried 4,000 characters per cell so a retrieved block could be reused verbatim.
        # But `get_kb_block` is the reader for exactly that, and eight cells at 4,000 characters
        # is most of a ~4,000-token search payload spent before the agent has decided which cell
        # it wants. Measured: trimming unit parameters saved almost nothing because cells were
        # the bulk. The full body is one `get_kb_block(doc_id)` away, and the doc_id is right here.
        # A METHOD SPEC is already the distilled form and its producer bounded it at 6,000;
        # re-trimming it here with the CELL budget cut it from 5,262 characters to 1,858 and
        # took the decay weights with it — a budget set in the producer, silently overridden by
        # a different budget in the consumer. Type decides which rule applies.
        "contents": (str(source.get("contents") or "")
                     if source.get("resource-type") == "PublicationMethodSpec"
                     else _excerpt(source.get("contents") or "")),
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
def _cluster() -> tuple:
    """``(node, username, password)`` of the cluster the agent KB reads.

    The SEARCH tier's (``platform_endpoints.search_cluster``): the cluster the keyword and semantic
    search beside it read, with the credential that follows that node. This read the bare names,
    so under a tier it could address a different cluster, with a different credential, from the
    search its results are joined to, and a refused query here reads as "no results".
    """
    from agent_runtime.platform_endpoints import search_cluster

    return search_cluster()


@lru_cache(maxsize=1)
def _os_client():
    from opensearchpy import OpenSearch
    node, user, pwd = _cluster()
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


_LOCAL_BACKEND_WARNED = False


def _warn_local_backend_once() -> None:
    """Say it once per process. Same reasoning as the resolved-embedding-URL log: a silently
    wrong backend produces plausible output and no signal at all."""
    global _LOCAL_BACKEND_WARNED
    if _LOCAL_BACKEND_WARNED:
        return
    _LOCAL_BACKEND_WARNED = True
    try:
        import logging
        logging.getLogger(__name__).warning(
            "agent KB is using the LOCAL file-backed store while OPENSEARCH_NODE is set; "
            "export AGENT_KB_BACKEND=opensearch to search the indexed corpus")
    except Exception:
        pass


def agent_kb_search(query: str, *, size: Optional[int] = None, client=None, embed: bool = True,
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

        # The Postgres record, when it is configured and no client was injected. It holds every
        # block and unit extraction produced, so it answers the same question the cluster does —
        # and it is checked FIRST because the alternative default is the file-backed local store,
        # which is EMPTY on a machine that has never run an ingest. An empty store returns
        # `count: 0`, which reads as "the corpus does not cover this" rather than "nothing was
        # ever written here" — the most expensive kind of silence in this system, and the reason
        # an A/B of the KB measured nothing until this existed.
        if client is None:
            try:
                from extractors import kb_db
            except Exception:
                kb_db = None
            if kb_db is not None and kb_db.enabled():
                with kb_db.connect() as conn:
                    hits = kb_db.search_kb(conn, query, size=size or 8)
                    docs = normalize_hits(hits, [], size or 8)
                    elements = (kb_db.parent_elements(
                        conn, [d["parent_doc_id"] for d in docs]) if resolve_parents else {})
                for d in docs:
                    d["element"] = elements.get(d["parent_doc_id"])
                return {"source": "agent_kb", "backend": "postgres", "count": len(docs),
                        "documents": docs,
                        "citation_ids": [d["parent_doc_id"] for d in docs],
                        "block_ids": [d["doc_id"] for d in docs], "elements": elements}

        use_opensearch = client is not None or kb_store.kb_backend() == "opensearch"
        base["backend"] = "opensearch" if use_opensearch else "local"
        if not use_opensearch:
            hits = kb_store.local_search(query, all_agent_indices(), size)
            docs = normalize_hits(hits, [], size)
            elements = resolve_parent_elements_local(hits) if resolve_parents else {}
            if _cluster()[0]:
                # The default is local so tests and offline runs never touch the cluster, and
                # that default is right. But it means a server that simply does not set the
                # variable reads a file-backed store scored by token overlap instead of the
                # indexed corpus — and the symptom is "fewer results", which reads as a
                # retrieval-quality problem rather than a configuration one.
                #
                # Observed: a live prototype turn showed `agent_kb_search -> no results` while
                # the same query against the cluster returned 8. Nothing anywhere said which
                # store had been consulted.
                base["note"] = ("searched the LOCAL file-backed store, NOT the cluster — a "
                                "cluster IS configured (OPENSEARCH_NODE is set); export "
                                "AGENT_KB_BACKEND=opensearch to search the indexed corpus")
                _warn_local_backend_once()
        else:
            if client is None and not _cluster()[0]:
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
        out = {
            "source": "agent_kb", "backend": ("opensearch" if use_opensearch else "local"),
            "count": len(docs), "documents": docs,
            "citation_ids": [d["parent_doc_id"] for d in docs],   # cite the ORIGINAL element
            "block_ids": [d["doc_id"] for d in docs],
            "elements": elements,
        }
        # Carried, not rebuilt away. The success path used to construct a fresh dict, so the
        # "you are reading the local store while a cluster is configured" note was computed and
        # then dropped — a diagnostic that exists only in a variable is not a diagnostic. `count`
        # and `documents` precede nothing that gets truncated, but `note` must survive too.
        if base.get("note"):
            out["note"] = base["note"]
        return out
    except Exception as exc:
        return {**base, "note": f"agent_kb_search error: {type(exc).__name__}: {exc}"}


# What it would take to IMPORT a unit, as opposed to what the unit is.
_IMPORT_FIELDS = ("import_line", "import_line_candidates", "library_module", "slice_sha",
                  "callability")


def _reference_only(source: Any) -> Any:
    """A stored doc as ``get_kb_block`` may show it: with the extraction bundle off, a unit stays a
    reference and loses what it would take to import it, by the same rule as ``_method_payload``.

    ``get_kb_block`` returns the RAW stored document, and a unit doc carries ``extracted.unit``
    with its import line and library module. Prod's index holds none today (0 of its unit docs),
    but the first ingest with this branch's emitter writes them, and a model handed an import line
    writes the import.
    """
    if extraction_enabled() or not isinstance(source, dict):
        return source
    extracted = source.get("extracted")
    unit = extracted.get("unit") if isinstance(extracted, dict) else None
    if not isinstance(unit, dict):
        return source
    kept = {k: v for k, v in unit.items() if k not in _IMPORT_FIELDS}
    return {**source, "extracted": {**extracted, "unit": kept}}


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
                return {"doc_id": doc_id, "found": True, "index": idx,
                        "source": _reference_only(src)}
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
                    return {"doc_id": doc_id, "found": True, "index": idx,
                            "source": _reference_only(resp.get("_source"))}
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



# --------------------------------------------------------------------------- #
# Joining the agent KB to search results BY ELEMENT ID
#
# The KB was only ever reachable by TEXT: `agent_kb_search` matches a query against
# `title`/`contents`/`extracted.embed_text`. That leaves a hole with a sharp edge — a hit found by
# SPATIAL search (a bounding box), by GRAPH search (a relation), or by a keyword that appears in
# the platform record but nowhere in the extracted sub-documents, can never surface its own
# extracted content. The element and its blocks, units, schema and method spec sit in the same
# corpus, keyed by the same id, and nothing joined them.
#
# So: look the KB up by id, attach it to the element it belongs to, and fold away the standalone
# rows that would otherwise compete with their own parent for an evidence slot.
# --------------------------------------------------------------------------- #

# How much extracted detail rides along with one element. A 200-cell notebook must not consume the
# whole evidence budget just because its element matched.
MAX_BLOCKS_PER_ELEMENT = 4
MAX_UNITS_PER_ELEMENT = 6
MAX_STEPS_PER_ELEMENT = 12


def _kb_indices() -> List[str]:
    from extractors.indices import all_agent_indices

    return list(all_agent_indices())


def _summarize_for_element(sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One element's extracted content, compacted into what a reader or the model can use."""
    blocks: List[Dict[str, Any]] = []
    units: List[Dict[str, Any]] = []
    dataset: Dict[str, Any] = {}
    spec: Dict[str, Any] = {}

    for source in sources:
        extracted = source.get("extracted") or {}
        if not isinstance(extracted, dict):
            continue
        kind = str(extracted.get("kind") or "")
        doc_id = str(source.get("doc_id") or "")
        if kind == "method_unit" or extracted.get("unit"):
            method = _method_payload(extracted)
            if method:
                units.append({**method, "doc_id": doc_id})
        elif kind == "notebook_block" or extracted.get("block"):
            block = extracted.get("block") or {}
            blocks.append({
                "doc_id": doc_id,
                "order": extracted.get("order"),
                "context": (block.get("markdown_context") or "")[:280],
                "tools": block.get("resolved_tools") or [],
                "imports": block.get("imports") or [],
            })
        elif kind == "dataset":
            dataset = {k: extracted.get(k) for k in
                       ("format", "family", "row_count", "crs", "bounds", "geometry_type",
                        "schema", "variables", "dims", "primary_member")
                       if extracted.get(k) not in (None, "", [], {})}
        elif kind == "publication":
            spec = {k: extracted.get(k) for k in
                    ("status", "steps", "datasets_referenced", "tools_referenced",
                     "is_method_spec")
                    if extracted.get(k) not in (None, "", [], {})}
            if spec.get("steps"):
                spec["steps"] = spec["steps"][:MAX_STEPS_PER_ELEMENT]

    blocks.sort(key=lambda b: b.get("order") if isinstance(b.get("order"), int) else 9999)
    # Contract-bearing units first: an importable one is worth more evidence budget than a bare
    # name, and the cap means the ordering decides what survives it.
    units.sort(key=lambda u: (0 if u.get("import_line") else 1, str(u.get("symbol") or "")))

    out: Dict[str, Any] = {}
    if units:
        out["units"] = units[:MAX_UNITS_PER_ELEMENT]
        out["unit_count"] = len(units)
    if blocks:
        out["blocks"] = blocks[:MAX_BLOCKS_PER_ELEMENT]
        out["block_count"] = len(blocks)
    if dataset:
        out["dataset"] = dataset
    if spec:
        out["publication"] = spec
    return out


def kb_for_elements(element_ids, *, client=None) -> Dict[str, Dict[str, Any]]:
    """``{element_id: extracted summary}`` for the elements that have any, keyed BY ID.

    Works on either backend, because the caller should not have to know which one is configured:
    the local file store is scanned, the cluster is queried with one terms lookup per index.
    An element with nothing extracted is simply absent from the result — never a stub, so a caller
    can test membership.
    """
    wanted = [str(e).strip() for e in (element_ids or []) if str(e or "").strip()]
    if not wanted:
        return {}
    wanted_set = set(wanted)
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    from extractors import kb_store

    if client is None and kb_store.kb_backend() != "opensearch":
        for element in wanted_set:
            for _doc_id, source in kb_store.local_blocks_for_parent(element, _kb_indices()):
                grouped.setdefault(element, []).append(source)
            # A dataset or publication element IS its own document, so it never appears in
            # `local_blocks_for_parent`, which excludes the element id itself.
            _index, direct = kb_store.local_get(element, _kb_indices())
            if direct:
                grouped.setdefault(element, []).append(direct)
    else:
        client = client or _os_client()
        for index in _kb_indices():
            try:
                if not client.indices.exists(index=index):
                    continue
                resp = client.search(index=index, body={
                    "size": 500,
                    "query": {"bool": {"should": [
                        {"terms": {"extracted.parent_doc_id": wanted}},
                        {"terms": {"doc_id": wanted}},
                    ], "minimum_should_match": 1}}})
            except Exception:                              # pragma: no cover
                # An index that is absent or unreachable costs its own contribution, never the
                # whole join: a search turn must not fail because enrichment could not run.
                continue
            for hit in (resp.get("hits", {}).get("hits") or []):
                source = hit.get("_source") or {}
                parent = _parent_of(str(source.get("doc_id") or hit.get("_id") or ""), source)
                if parent in wanted_set:
                    grouped.setdefault(parent, []).append(source)

    return {element: summary for element, sources in grouped.items()
            if (summary := _summarize_for_element(sources))}


def attach_kb_to_documents(documents, *, client=None) -> Dict[str, Any]:
    """Attach each element's extracted content to its own search result, and fold the duplicates.

    Two things happen, and the second is as important as the first:

    * every platform document gains ``extracted`` — the units, blocks, schema or method spec that
      belong to that element — so a spatial or graph hit carries its contract even though no text
      matched;
    * standalone KB rows whose parent is already in the result set are REMOVED, because they were
      competing with their own element for an evidence slot. Deduplicating by parent was an
      explicit exit criterion that the text-only union could not meet.

    Returns ``{documents, attached, folded, actionable}``. ``actionable`` is the flat list of
    import lines across everything attached — the things the agent can actually run, as opposed to
    read.
    """
    docs = [d for d in (documents or []) if isinstance(d, dict)]
    if not docs:
        return {"documents": [], "attached": 0, "folded": 0, "actionable": []}

    platform_ids, kb_rows = [], []
    for doc in docs:
        if str(doc.get("source") or "") in ("agent_kb", "method_library"):
            kb_rows.append(doc)
        elif doc.get("doc_id"):
            platform_ids.append(str(doc["doc_id"]))

    summaries = kb_for_elements(platform_ids, client=client) if platform_ids else {}

    out, folded = [], 0
    for doc in docs:
        source = str(doc.get("source") or "")
        if source in ("agent_kb", "method_library"):
            parent = str(doc.get("parent_doc_id") or "")
            if parent and parent in summaries:
                folded += 1        # its content now rides on the element itself
                continue
            out.append(doc)
            continue
        summary = summaries.get(str(doc.get("doc_id") or ""))
        out.append({**doc, "extracted": summary} if summary else doc)

    actionable = []
    for element, summary in summaries.items():
        for unit in summary.get("units") or []:
            if unit.get("import_line"):
                actionable.append({"element": element, "symbol": unit.get("symbol"),
                                   "signature": unit.get("signature"),
                                   "import_line": unit["import_line"],
                                   "requirements": unit.get("requirements") or []})
    return {"documents": out, "attached": len(summaries), "folded": folded,
            "actionable": actionable}


__all__ = [
    "agent_kb_search", "get_kb_block", "build_keyword_query", "build_knn_query", "normalize_hit",
    "normalize_hits", "group_by_parent", "resolve_parent_elements", "resolve_parent_elements_local",
    "kb_for_elements", "attach_kb_to_documents",
]
