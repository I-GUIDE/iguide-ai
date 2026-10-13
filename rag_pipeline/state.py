from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, TypedDict

from .search.utils import is_linkable_copy, normalize_source_fields, safe_score


class DocumentPayload(TypedDict, total=False):
    doc_id: str
    element_type: str
    title: str
    contents: str
    contributor: Optional[str]
    authors: Optional[Sequence[str]]
    tags: Optional[Sequence[str]]
    thumbnail_image: Any


class EvidenceEntry(TypedDict, total=False):
    source: str
    score: float
    retrieval_rank: int
    document: DocumentPayload
    metadata: Dict[str, Any]


class AnswerState(TypedDict, total=False):
    final_composed_answer: Optional[str]
    citations: List[Dict[str, Any]]
    confidence_score: Optional[float]


class EvidenceState(TypedDict, total=False):
    retrieved_documents: List[EvidenceEntry]
    sources: Dict[str, Any]


class AgentState(TypedDict, total=False):
    query_information: Dict[str, Any]
    session_context: Dict[str, Any]
    evidence: EvidenceState
    planner_reasoning: Dict[str, Any]
    safety_checks: Dict[str, Any]
    trace_observability: Dict[str, Any]
    answer: AnswerState


def ensure_state_shapes(state: MutableMapping[str, Any]) -> AgentState:
    """
    Guarantee that downstream code can rely on required top-level keys.
    """
    if "query_information" not in state or not isinstance(state["query_information"], dict):
        state["query_information"] = {}

    if "session_context" not in state or not isinstance(state["session_context"], dict):
        state["session_context"] = {}

    evidence = state.get("evidence")
    if not isinstance(evidence, dict):
        evidence = {}
        state["evidence"] = evidence

    if "retrieved_documents" not in evidence or not isinstance(evidence["retrieved_documents"], list):
        evidence["retrieved_documents"] = []

    sources = evidence.get("sources")
    if not isinstance(sources, dict):
        evidence["sources"] = {}

    answer = state.get("answer")
    if not isinstance(answer, dict):
        answer = {}
        state["answer"] = answer

    answer.setdefault("final_composed_answer", None)
    answer.setdefault("citations", [])
    answer.setdefault("confidence_score", None)

    for key in ("planner_reasoning", "safety_checks", "trace_observability"):
        if key not in state or not isinstance(state[key], dict):
            state[key] = {}

    return state  # type: ignore[return-value]


def get_query_text(state: Mapping[str, Any], fallback: str = "") -> str:
    """
    Extract the primary query string from the shared state.
    """
    info = state.get("query_information") or {}
    if isinstance(info, dict):
        candidate = info.get("raw_text") or info.get("query")
        if candidate:
            return str(candidate).strip()
    return str(fallback).strip()


def _normalize_hit(hit: Dict[str, Any]) -> Tuple[DocumentPayload, Dict[str, Any], float]:
    raw_source = hit.get("_source") or hit.get("document") or {}
    hit_id = hit.get("_id") or raw_source.get("doc_id") or raw_source.get("id")
    normalized = normalize_source_fields(raw_source, str(hit_id or ""))
    raw_score = hit.get("_score", hit.get("score", 0.0))
    score = safe_score(raw_score, 0.0)
    metadata = {
        "raw_score": raw_score,
        "hit_id": hit_id,
        "original": {k: v for k, v in hit.items() if k not in {"_source", "document"}},
    }
    return normalized, metadata, score


def build_evidence_items(
    hits: Iterable[Dict[str, Any]],
    *,
    source: str,
    start_rank: int = 0,
) -> List[EvidenceEntry]:
    items: List[EvidenceEntry] = []
    for offset, hit in enumerate(hits):
        document, metadata, score = _normalize_hit(hit)
        items.append(
            EvidenceEntry(
                source=source,
                score=score,
                retrieval_rank=start_rank + offset,
                document=document,
                metadata=metadata,
            )
        )
    return items


def merge_retrieval(
    state: MutableMapping[str, Any],
    *,
    source: str,
    hits: Iterable[Dict[str, Any]],
    limit: Optional[int] = None,
    dedupe: bool = True,
) -> List[EvidenceEntry]:
    """
    Convert raw search hits into the shared evidence shape and merge them into the state.
    Returns the list of appended evidence entries.
    """
    ensure_state_shapes(state)
    evidence_state: EvidenceState = state["evidence"]  # type: ignore[assignment]
    current_docs = evidence_state["retrieved_documents"]

    start_rank = len(current_docs)
    items = build_evidence_items(hits, source=source, start_rank=start_rank)

    if dedupe:
        # A duplicate is dropped, unless it is linkable and the copy already held is not: then it
        # takes that copy's place (see is_linkable_copy).
        held = {entry["document"]["doc_id"]: ("current", i) for i, entry in enumerate(current_docs)}
        deduped: List[EvidenceEntry] = []
        for entry in items:
            doc_id = entry["document"].get("doc_id") or ""
            if doc_id in held:
                where, i = held[doc_id]
                kept = current_docs if where == "current" else deduped
                if is_linkable_copy(entry["document"]) and not is_linkable_copy(kept[i]["document"]):
                    kept[i] = entry
                continue
            held[doc_id] = ("new", len(deduped))
            deduped.append(entry)
        items = deduped

    if limit is not None and limit >= 0:
        space = max(limit - len(current_docs), 0)
        items = items[:space]

    current_docs.extend(items)
    evidence_state["sources"][source] = {
        "total_appended": len(items),
        "cumulative": len(current_docs),
    }
    return items


@dataclass(frozen=True)
class RoutingDecision:
    source: str
    reason: str


def summarize_evidence(evidence: EvidenceState) -> Dict[str, Any]:
    """
    Lightweight overview for debugging or analytics dashboards.
    """
    docs = evidence.get("retrieved_documents", [])
    if not docs:
        return {"count": 0, "sources": {}}

    source_breakdown: Dict[str, int] = {}
    for entry in docs:
        src = entry.get("source", "unknown")
        source_breakdown[src] = source_breakdown.get(src, 0) + 1

    top_score = max((entry.get("score", float("-inf")) for entry in docs), default=None)
    return {
        "count": len(docs),
        "sources": source_breakdown,
        "top_score": None if top_score in (None, float("-inf"), math.inf) else top_score,
    }
