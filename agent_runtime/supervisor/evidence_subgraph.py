"""Shared-state evidence/quality sub-graph: retrieve → rerank → analyze → audit.

This models the RAG core as an explicit LangGraph ``StateGraph`` over a single
typed ``EvidenceState`` instead of the nested ``search_agent_evidence`` /
``analysis_agent_answer`` tool calls. Benefits over agents-as-tools here:

* **Evidence is first-class state** — no per-tool dedup workaround, no
  serialization through tool args/returns.
* **Rerank + grounding audit are deterministic edges** — always applied (true
  parity with the deprecated RAG pipeline), not optional LLM tool calls.
* **Flattened depth** — one linear pipeline rather than orchestrator→analysis→search.

Context hygiene rule: the heavy retrieved documents live *in the state*; callers
should surface only the **distilled** payload upward (``distill_evidence_state``)
so a parent agent's context isn't bloated with full documents.

Everything is dependency-injected (``retrieve_fn`` / ``compose_fn`` / ``llm``) so
the graph is fully unit-testable without a live LLM or search backends.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from agent_runtime.evidence_quality import audit_answer_grounding, rerank_documents
from agent_runtime.streaming_trace import emit_trace_event

# (query, state) -> list of document dicts
RetrieveFn = Callable[[str, "EvidenceState"], List[Any]]
# (query, documents, chat_history) -> answer text
ComposeFn = Callable[[str, List[Any], Optional[List[Any]]], str]


class EvidenceState(TypedDict, total=False):
    """Shared state threaded through the evidence/quality sub-graph."""

    query: str
    chat_history: List[Any]
    thread_id: Optional[str]
    documents: List[Any]          # structured evidence (heavy; stays in state)
    answer: str
    audit: Dict[str, Any]
    distilled: Dict[str, Any]     # compact payload to surface upward
    # config
    top_k: int
    do_rerank: bool
    do_audit: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc_field(doc: Any, *keys: str, default: str = "") -> str:
    if isinstance(doc, dict):
        src = doc.get("document") if isinstance(doc.get("document"), dict) else doc
        for k in keys:
            v = src.get(k)
            if v:
                return str(v)
    return default


def _element_url(doc: Any) -> str:
    """The link target for a document, so the synthesizer can render a clickable citation.

    * Internal knowledge element -> ``{FRONTEND_DOMAIN}/{element_type-plural}/{doc_id}``
      (plural, except ``code`` stays ``code``) — same scheme as the smart-search frontend.
    * External (OpenGeoData catalog record, open-web page) -> its own landing ``url`` from the
      search payload. Pluralizing those element types would fabricate a platform page that does
      not exist (``/webs/web-1a2b…``).
    Returns "" when no link can be formed (the synthesizer then cites the title in bold).
    """
    src = doc.get("document") if isinstance(doc, dict) and isinstance(doc.get("document"), dict) else doc
    if not isinstance(src, dict):
        return ""
    etype = str(src.get("element_type") or src.get("resource-type") or "").strip().lower()
    if etype in {"opengeodata", "web"}:
        return str(src.get("url") or "")
    doc_id = str(src.get("doc_id") or src.get("id") or src.get("_id") or "")
    if not doc_id:
        return str(src.get("url") or "")
    if not etype or etype == "resource":
        # No usable element type -> only link if the payload already carries a url.
        return str(src.get("url") or "")
    plural = etype if etype == "code" else f"{etype}s"
    base = os.getenv("FRONTEND_DOMAIN", "https://platform.i-guide.io").rstrip("/")
    return f"{base}/{plural}/{doc_id}"


# How much extracted detail one element may add to the evidence view. Bounded because this rides
# on TOP of the document's own contents, and an element with 200 blocks would otherwise crowd out
# every other result.
EXTRACTED_MAX_CHARS = 900


def _render_extracted(extracted: Any, element_hint: str = "<element id>") -> str:
    """What extraction produced from this element, as REFERENCES and ACTIONABLE items.

    The distinction is the point. A block or a method spec is a reference — it tells the model what
    the element contains and lets it answer accurately. An import line is actionable: the method
    library is mounted read-only inside ``execute_code``, so that exact line runs there without an
    install, a download or a network hop. Rendering both as undifferentiated prose was how the
    agent kept saying "adapt this notebook" while an importable function sat one field away.
    """
    if not isinstance(extracted, dict) or not extracted:
        return ""
    parts: List[str] = []

    units = [u for u in (extracted.get("units") or []) if isinstance(u, dict)]
    runnable = [u for u in units if u.get("import_line")]
    if runnable:
        lines = ["RUNNABLE METHODS extracted from this element — the import line works verbatim "
                 "inside execute_code (the library is mounted read-only); prefer it over "
                 "re-implementing:"]
        for unit in runnable:
            lines.append(f"  - {unit.get('signature') or unit.get('symbol')}")
            if unit.get("doc_summary"):
                lines.append(f"      {unit['doc_summary']}")
            lines.append(f"      import: {unit['import_line']}")
            if unit.get("requirements"):
                lines.append(f"      requires: {', '.join(map(str, unit['requirements']))}")
            if unit.get("invariants"):
                lines.append(f"      enforced: {', '.join(map(str, unit['invariants']))}")
            # A loader advertises `load_x(staged_path)` and nothing said where a staged_path comes
            # from. Measured on a live turn: the agent read this signature, called the loader
            # without staging, got "block not found", and fell back to a different tool over a
            # different subset of the data (49,789 geocoded incidents instead of 128,886 records)
            # -- reporting THEFT as 9,993 where the file says 27,824. The parameter has to carry
            # its own instructions, at the point of use, because a generic prompt rule did not
            # survive the moment of choice.
            if "staged_path" in str(unit.get("signature") or ""):
                lines.append(f"      FIRST call stage_element(\"{element_hint}\") to obtain "
                             f"staged_path — this file is not in the sandbox until you do.")
        parts.append("\n".join(lines))

    named_only = [u for u in units if not u.get("import_line")]
    if named_only:
        parts.append("METHODS PRESENT but not importable (reference only): "
                     + ", ".join(str(u.get("symbol")) for u in named_only if u.get("symbol")))
    if extracted.get("unit_count") and len(units) < int(extracted["unit_count"]):
        parts.append(f"({extracted['unit_count']} methods in total; "
                     f"{len(units)} shown — call kb_method_search for the rest)")

    dataset = extracted.get("dataset")
    if isinstance(dataset, dict) and dataset:
        bits = []
        for key in ("format", "family", "row_count", "crs", "geometry_type"):
            if dataset.get(key):
                bits.append(f"{key}={dataset[key]}")
        if dataset.get("schema"):
            bits.append("fields=" + ", ".join(map(str, dataset["schema"][:14])))
        if dataset.get("variables"):
            bits.append("variables=" + ", ".join(map(str, dataset["variables"][:14])))
        if bits:
            parts.append("DATASET STRUCTURE (extracted from the file itself): " + "; ".join(bits))

    publication = extracted.get("publication")
    if isinstance(publication, dict) and publication.get("steps"):
        steps = "\n".join(f"  {i}. {s}" for i, s in enumerate(publication["steps"][:8], 1))
        parts.append("METHOD THIS PAPER DESCRIBES (extracted from its full text):\n" + steps)

    blocks = [b for b in (extracted.get("blocks") or []) if isinstance(b, dict)]
    if blocks:
        summary = "; ".join(str(b.get("context") or "").replace("\n", " ")[:70]
                            for b in blocks if b.get("context"))
        if summary:
            parts.append(f"NOTEBOOK SECTIONS ({extracted.get('block_count', len(blocks))} cells): "
                         + summary)

    return ("\n".join(parts))[:EXTRACTED_MAX_CHARS]


def _doc_block(doc: Any, *, max_chars: int = 2500) -> str:
    # One evidence item as title + url + contents. We deliberately do NOT lead with the raw
    # [doc_id]: showing it trained the synthesizer to cite "[<uuid>]" instead of the hyperlink
    # Rule 2 asks for. Title + url are the only citation handles the model sees.
    title = _doc_field(doc, "title", "name", "element_type", default="Untitled")
    contents = _doc_field(doc, "contents", "snippet", "text", "abstract", "description")
    url = _element_url(doc)
    head = f"title: {title}" + (f"\nurl: {url}" if url else "")

    # An extracted METHOD SPEC is exempt from the generic budget, because it is not a document
    # excerpt — it is the whole method in numbered steps, and its parameters live in the middle
    # of the list.
    #
    # This is the fourth budget between the record and the model. `kb_db` carries a spec whole at
    # 6,000 characters; `agent_kb.normalize_hit` re-trimmed it to 1,800 with the CELL budget;
    # this line cut what survived to 2,500; and `EXTRACTED_MAX_CHARS` caps the enrichment at 900.
    # Each was reasonable where it was written and none knew about the others, so the producer's
    # budget meant nothing and nobody could predict what the model actually saw. Measured: asked
    # for a paper's distance-decay weights, the agent answered "the retrieved evidence does not
    # state the actual numeric weight values" while the record held "Apply distance-decay weights
    # (1, 0.68, 0.22)" — cut, twice, between the two.
    is_spec = (isinstance(doc, dict)
               and doc.get("resource_type") == "PublicationMethodSpec")
    body = f"{head}\n{contents if is_spec else contents[:max_chars]}"
    # Everything extraction produced for this element, joined on by id in _direct_search_sweep.
    # Without this the join enriched a document the model never saw the enrichment of.
    extracted = doc.get("extracted") if isinstance(doc, dict) else None
    rendered = _render_extracted(
        extracted, str((doc.get("doc_id") if isinstance(doc, dict) else "") or "<element id>"))
    return f"{body}\n{rendered}" if rendered else body


def _format_related_two_buckets(documents: List[Any], *, max_chars: int = 2500) -> str:
    """Render a related-element result as two clearly-separated buckets so the synthesizer
    presents contributor-specified links apart from similarity hits (and the grounding auditor
    can tell them apart). Triggered whenever any doc carries a ``provenance`` tag."""
    seed = [d for d in documents if isinstance(d, dict) and d.get("provenance") == "seed"]
    curated = [d for d in documents if isinstance(d, dict) and d.get("provenance") == "curated"]
    content = [d for d in documents if isinstance(d, dict) and d.get("provenance") == "content"]
    parts: List[str] = []
    if seed:
        parts.append("[QUERIED ELEMENT — the resource whose related elements were requested. Use "
                     "THIS title/link when naming the resource in the answer; do not infer its "
                     "identity from the other items.]")
        parts.append("\n\n".join(_doc_block(d, max_chars=max_chars) for d in seed))
    parts.extend([
        "\n[CURATED related elements — specified by the contributor. Authoritative: present "
        "THESE as the element's related elements.]",
        "\n\n".join(_doc_block(d, max_chars=max_chars) for d in curated) if curated
        else "(none — the contributor has not specified any related elements for this element)",
        "\n[CONTENT-RELATED elements — found by similarity search. These are NOT contributor-"
        "specified relationships; present them in a SEPARATE section as topically similar, not "
        "as curated links.]",
        "\n\n".join(_doc_block(d, max_chars=max_chars) for d in content) if content
        else "(no content-similar elements found)",
    ])
    return "\n".join(parts)


def _format_documents(documents: List[Any], *, limit: int = 8, max_chars: int = 2500) -> str:
    # A related-element result carries provenance tags -> render two labeled buckets.
    if any(isinstance(d, dict) and d.get("provenance") in ("seed", "curated", "content") for d in documents):
        return _format_related_two_buckets(documents, max_chars=max_chars)
    blocks = [_doc_block(doc, max_chars=max_chars) for doc in documents[:limit]]
    return "\n\n".join(blocks) if blocks else "(no evidence)"


def extract_documents_from_search_evidence(payload: Any) -> List[Dict[str, Any]]:
    """Best-effort: pull structured documents out of a search-evidence payload.

    Accepts the dict produced by ``runtime_utils.build_search_evidence_payload``
    (``search_agent_tool_results`` = list of ``{name, content}`` where content is
    a JSON string from a granular search tool). Robust to varied backend shapes.
    """
    docs: List[Dict[str, Any]] = []
    if not isinstance(payload, dict):
        return docs
    results = payload.get("search_agent_tool_results") or []
    list_keys = ("documents", "retrieved_documents", "results", "hits", "items", "records")
    for item in results:
        content = item.get("content") if isinstance(item, dict) else None
        if not isinstance(content, str):
            continue
        try:
            parsed = json.loads(content)
        except Exception:
            continue
        candidates: List[Any] = []
        if isinstance(parsed, list):
            candidates = parsed
        elif isinstance(parsed, dict):
            for key in list_keys:
                if isinstance(parsed.get(key), list):
                    candidates = parsed[key]
                    break
        for c in candidates:
            if isinstance(c, dict):
                docs.append(c)
    return docs


def distill_evidence_state(state: EvidenceState) -> Dict[str, Any]:
    """Compact payload to surface upward (keeps heavy docs out of parent context)."""
    docs = state.get("documents") or []
    return {
        "answer": state.get("answer", ""),
        "audit": state.get("audit") or {},
        "doc_ids": [_doc_field(d, "doc_id", "id", "_id", default=f"doc-{i}") for i, d in enumerate(docs)],
        "document_count": len(docs),
    }


def _content_to_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "".join(
            str(p.get("text") or p.get("content") or "") if isinstance(p, dict) else str(getattr(p, "text", p))
            for p in content
        )
    return str(content or "").strip()


def default_compose_fn(llm: Optional[Any] = None) -> ComposeFn:
    """A grounded-answer composer using the agent's LLM (injectable for tests).

    Uses the supervisor's own tool-free ``SYNTHESIS_PROMPT`` (same grounding rules —
    use-only-evidence, cite doc_ids, state uncertainty, never invent — without the
    legacy AnalysisAgent's tool-calling rules).
    """

    def compose(query: str, documents: List[Any], chat_history: Optional[List[Any]] = None) -> str:
        from agent_runtime.supervisor.prompts import SYNTHESIS_PROMPT

        active = llm
        if active is None:
            from agent_runtime.executor_factory import build_default_llm

            active = build_default_llm()
        prompt = (
            f"{SYNTHESIS_PROMPT}\n\n"
            f"Question:\n{query}\n\nEvidence:\n{_format_documents(documents)}\n"
        )
        if hasattr(active, "invoke"):
            return _content_to_text(active.invoke(prompt))
        if callable(active):
            return str(active(prompt))
        raise TypeError("llm must expose .invoke() or be a str->str callable")

    return compose


# ---------------------------------------------------------------------------
# Sub-graph
# ---------------------------------------------------------------------------

def build_evidence_subgraph(
    retrieve_fn: RetrieveFn,
    *,
    compose_fn: Optional[ComposeFn] = None,
    llm: Optional[Any] = None,
    top_k: int = 5,
    do_rerank: bool = True,
    do_audit: bool = True,
) -> Any:
    """Compile the retrieve → rerank → analyze → audit sub-graph.

    ``retrieve_fn(query, state) -> [documents]`` and ``compose_fn(query, docs,
    chat_history) -> answer`` are injected; ``llm`` powers rerank + audit (and the
    default composer). The ``do_rerank`` / ``do_audit`` defaults are baked into the
    initial state but can be overridden per-invocation via the state.
    """
    composer = compose_fn or default_compose_fn(llm=llm)

    def retrieve_node(state: EvidenceState) -> Dict[str, Any]:
        emit_trace_event("node_started", {"stage": "retrieve", "message": "Retrieving evidence"}, node="retrieve")
        docs = retrieve_fn(state.get("query", ""), state) or []
        emit_trace_event(
            "node_completed",
            {"stage": "retrieve", "message": f"Retrieved {len(docs)} documents"},
            node="retrieve",
        )
        return {"documents": list(docs)}

    def rerank_node(state: EvidenceState) -> Dict[str, Any]:
        docs = state.get("documents") or []
        k = state.get("top_k", top_k)
        if not state.get("do_rerank", do_rerank):
            return {"documents": docs[:k] if k else docs}
        emit_trace_event("node_started", {"stage": "rerank", "message": "Re-ranking evidence"}, node="rerank")
        reranked = rerank_documents(state.get("query", ""), docs, top_k=k, llm=llm)
        emit_trace_event(
            "node_completed",
            {"stage": "rerank", "message": f"Ranked {len(reranked)} documents"},
            node="rerank",
        )
        return {"documents": reranked}

    def analyze_node(state: EvidenceState) -> Dict[str, Any]:
        emit_trace_event("node_started", {"stage": "analyze", "message": "Composing answer"}, node="analyze")
        answer = composer(state.get("query", ""), state.get("documents") or [], state.get("chat_history"))
        emit_trace_event("node_completed", {"stage": "analyze", "message": "Answer composed"}, node="analyze")
        return {"answer": answer}

    def audit_node(state: EvidenceState) -> Dict[str, Any]:
        if not state.get("do_audit", do_audit):
            distilled_state = {**state, "audit": {}}
            return {"audit": {}, "distilled": distill_evidence_state(distilled_state)}
        emit_trace_event("node_started", {"stage": "audit", "message": "Auditing grounding"}, node="audit")
        verdict = audit_answer_grounding(
            state.get("query", ""), state.get("answer", ""), state.get("documents") or [], llm=llm
        )
        emit_trace_event(
            "node_completed",
            {"stage": "audit", "message": verdict.get("summary") or "Grounding audited"},
            node="audit",
        )
        merged = {**state, "audit": verdict}
        return {"audit": verdict, "distilled": distill_evidence_state(merged)}

    builder = StateGraph(EvidenceState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("rerank", rerank_node)
    builder.add_node("analyze", analyze_node)
    builder.add_node("audit", audit_node)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "analyze")
    builder.add_edge("analyze", "audit")
    builder.add_edge("audit", END)
    return builder.compile()


def run_evidence_pipeline(
    query: str,
    retrieve_fn: RetrieveFn,
    *,
    chat_history: Optional[List[Any]] = None,
    compose_fn: Optional[ComposeFn] = None,
    llm: Optional[Any] = None,
    top_k: int = 5,
    do_rerank: bool = True,
    do_audit: bool = True,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience: build + invoke the sub-graph; return the full final state."""
    graph = build_evidence_subgraph(
        retrieve_fn, compose_fn=compose_fn, llm=llm, top_k=top_k, do_rerank=do_rerank, do_audit=do_audit
    )
    return graph.invoke(
        {
            "query": query,
            "chat_history": chat_history or [],
            "thread_id": thread_id,
            "top_k": top_k,
            "do_rerank": do_rerank,
            "do_audit": do_audit,
        }
    )


__all__ = [
    "EvidenceState",
    "build_evidence_subgraph",
    "run_evidence_pipeline",
    "default_compose_fn",
    "extract_documents_from_search_evidence",
    "distill_evidence_state",
]
