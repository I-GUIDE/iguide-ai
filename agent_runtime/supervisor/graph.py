"""Shared-state supervisor-over-peers orchestration.

Search / analyze / code are **peer** capability nodes (same level) that share one
typed ``SupervisorState``; an LLM **supervisor** decides the next action and the
graph **loops** back to it. When the supervisor is ``done``, a dedicated
**synthesize** node composes the final, grounded answer.

Single-responsibility split:
* **search**   — retrieve evidence (rerank bundled in).
* **analyze**  — *execute a GIS/data analysis workflow* (run spatial/stat tools),
  writing ``analysis_results`` to shared state. It does NOT compose prose.
* **code**     — produce runnable code, writing ``code_result``.
* **synthesize** — compose the final answer (tool-free ``SYNTHESIS_PROMPT``)
  from evidence + analysis_results + code_result, then audit grounding.

The supervisor only ever sees a *distilled* view (counts/flags), never the heavy
documents. Everything is dependency-injected so the graph is unit-testable with no
live LLM/backends. Default adapters wire to existing agents (best-effort; need
live validation). Default ON; per-request override ``use_supervisor``; env opt-out
``AGENT_SUPERVISOR=0``.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from agent_runtime.evidence_quality import _extract_json_object, audit_answer_grounding, rerank_documents
from agent_runtime.supervisor.evidence_subgraph import (
    _content_to_text,
    _doc_field,
    _format_documents,
    extract_documents_from_search_evidence,
)
from agent_runtime.streaming_trace import emit_trace_event

# Words that carry no retrieval signal — stripped when judging topical coverage and when building
# a fallback query reformulation.
_QUERY_FILLER = {
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "were", "about", "into",
    "data", "dataset", "datasets", "database", "information", "info", "find", "search", "show",
    "list", "give", "get", "please", "want", "need", "any", "all", "some", "related", "available",
    "using", "use", "how", "what", "which", "where", "when", "why", "can", "could", "would",
    "there", "their", "have", "has", "does", "did", "should", "may", "might", "will",
}

ALLOWED_ACTIONS = ("search", "analyze", "code", "done")
DEFAULT_MAX_STEPS = 8


def _max_searches() -> int:
    """Hard cap on how many times the search peer may run per request."""
    try:
        return max(1, int(os.getenv("AGENT_SUPERVISOR_MAX_SEARCHES", "2")))
    except (TypeError, ValueError):
        return 2


def _max_peer_runs() -> int:
    """Cap on how many times any single peer may run (bounds needs-driven re-run loops)."""
    try:
        return max(1, int(os.getenv("AGENT_SUPERVISOR_MAX_PEER_RUNS", "3")))
    except (TypeError, ValueError):
        return 3


def _default_top_k() -> int:
    """Evidence kept after the search rerank. A single search action fans out across several
    retrieval methods, so truncating to the historical 5 made listing answers incomplete."""
    try:
        return max(1, int(os.getenv("AGENT_SUPERVISOR_TOP_K", "8")))
    except (TypeError, ValueError):
        return 8


def _search_exhausted(state: "SupervisorState") -> bool:
    """Whether further searching is pointless: it hit the attempt cap, or the most
    recent search returned NO new evidence. Stops the supervisor (and peer 'needs')
    from hammering the search agent when the knowledge base has nothing to return."""
    if state.get("search_attempts", 0) >= _max_searches():
        return True
    return state.get("search_empty_streak", 0) >= 1

# Injected callables
DecideFn = Callable[["SupervisorState", Dict[str, Any]], str]    # (state, distilled) -> action
SearchFn = Callable[[str, "SupervisorState"], List[Any]]          # (query, state) -> documents
AnalyzeFn = Callable[[str, List[Any], "SupervisorState"], Any]    # (query, evidence, state) -> analysis_results
CodeFn = Callable[[str, List[Any], "SupervisorState"], Any]       # (query, evidence, state) -> code_result
# (query, evidence, analysis_results, code_result, chat_history) -> answer
SynthesizeFn = Callable[[str, List[Any], Any, Any, Optional[List[Any]]], str]

from agent_runtime.supervisor.prompts import ANALYSIS_WORKFLOW_PROMPT, CODE_PEER_PROMPT, NO_GROUNDING_FALLBACK


class SupervisorState(TypedDict, total=False):
    query: str
    chat_history: List[Any]
    thread_id: Optional[str]
    evidence: List[Any]            # accumulated, dedup'd documents (shared, heavy)
    analysis_results: Any          # outputs of the analysis workflow
    code_result: Any
    answer: str
    audit: Dict[str, Any]
    needs: List[Dict[str, Any]]    # queue of capability requests from peers (FIFO)
    actions: List[str]             # supervisor decision history
    next_action: str
    step: int
    max_steps: int
    final_answer: str
    distilled: Dict[str, Any]
    search_attempts: int           # how many times the search peer has run
    search_empty_streak: int       # consecutive searches that added NO new evidence
    searched_queries: List[str]    # every query string actually searched (incl. refinements)


def is_supervisor_enabled() -> bool:
    """Whether the orchestrate path should use the supervisor-over-peers graph.

    Default **on**; set ``AGENT_SUPERVISOR`` to a falsy value (0/false/no/off) to
    fall back to the legacy agents-as-tools orchestrator.
    """
    return (os.getenv("AGENT_SUPERVISOR") or "").strip().lower() not in {"0", "false", "no", "off"}


# ---------------------------------------------------------------------------
# Shared-state helpers
# ---------------------------------------------------------------------------

def _doc_key(doc: Any) -> str:
    """Stable dedup key for a document.

    Uses an explicit id when present; otherwise falls back to a content hash
    (title + contents) so ID-less documents from the multi-pass search the design
    encourages still dedup against each other — the previous positional fallbacks
    (``_{i}`` vs ``new-{j}``) lived in disjoint namespaces and could never collide.
    """
    ident = _doc_field(doc, "doc_id", "id", "_id", default="")
    if ident:
        return f"id:{ident}"
    import hashlib

    title = _doc_field(doc, "title", default="")
    contents = _doc_field(doc, "contents", "content", "text", "summary", default="")
    digest = hashlib.sha1(f"{title}\n{contents}".strip().encode("utf-8", "replace")).hexdigest()
    return f"c:{digest}"


def _merge_dedup(existing: List[Any], new: List[Any]) -> List[Any]:
    merged = list(existing or [])
    seen = {_doc_key(d) for d in merged}
    for d in new or []:
        key = _doc_key(d)
        if key in seen:
            continue
        seen.add(key)
        merged.append(d)
    return merged


def _focus_terms(query: str) -> List[str]:
    """Subject terms of a query: lowercase, punctuation-stripped, filler removed.

    Used to judge whether retrieved evidence is actually ABOUT the request (and to build a
    fallback reformulation), so a query made of filler cannot look like a topical match.
    """
    tokens = re.findall(r"[a-z0-9][a-z0-9\-']*", str(query or "").lower())
    return [t for t in tokens if len(t) >= 3 and t not in _QUERY_FILLER]


def _term_coverage(docs: List[Any], query: str) -> Optional[float]:
    """Fraction of documents mentioning at least one subject term of *query*.

    None when there is nothing to judge (no docs, or a query of pure filler). This is the
    supervisor's cheap, deterministic signal for "did we retrieve the right thing?" — it needs no
    LLM call and cannot be fooled by a large but off-topic result set.
    """
    terms = _focus_terms(query)
    if not docs or not terms:
        return None
    hits = 0
    for d in docs:
        text = " ".join([
            _doc_field(d, "title", "name", default=""),
            _doc_field(d, "contents", "snippet", "text", "abstract", default="")[:600],
        ]).lower()
        if any(t in text for t in terms):
            hits += 1
    return round(hits / len(docs), 2)


# --- query refinement (opt-in: AGENT_SEARCH_REFINE) --------------------------------
# Without this the loop can only re-run the IDENTICAL query, so a second search returns the same
# documents and the exhaustion guard (correctly) stops it. Refinement makes a retry meaningful:
# the query actually changes before searching again.
def _refine_enabled() -> bool:
    """Whether an unproductive search may be retried with a REFORMULATED query (default off)."""
    return (os.getenv("AGENT_SEARCH_REFINE") or "").strip().lower() in {"1", "true", "yes", "on"}


def _max_refinements() -> int:
    try:
        return max(1, int(os.getenv("AGENT_SEARCH_REFINE_MAX", "1")))
    except (TypeError, ValueError):
        return 1


def _min_coverage() -> float:
    """Topical coverage below which a result set counts as off-topic (0..1)."""
    try:
        return min(1.0, max(0.0, float(os.getenv("AGENT_SEARCH_MIN_COVERAGE", "0.34"))))
    except (TypeError, ValueError):
        return 0.34


def _results_are_poor(docs: List[Any], query: str) -> bool:
    """True when a search returned nothing, or nothing that mentions the request's subject."""
    if not docs:
        return True
    coverage = _term_coverage(docs, query)
    return coverage is not None and coverage < _min_coverage()


def _fallback_refinement(query: str, tried: List[str]) -> Optional[str]:
    """Deterministic reformulation used when no LLM is available (or it declines).

    Step 1: the subject terms alone (filler removed). Step 2: the two most specific terms
    (longest), i.e. a broader query. Returns None once both have been tried.
    """
    terms = _focus_terms(query)
    if not terms:
        return None
    focused = " ".join(terms)
    if focused and focused not in tried and focused.lower() != query.strip().lower():
        return focused
    broader = " ".join(sorted(terms, key=len, reverse=True)[:2])
    if broader and broader not in tried:
        return broader
    return None


def _refine_query(llm: Optional[Any], query: str, docs: List[Any], tried: List[str]) -> Optional[str]:
    """A better query to try next, or None. LLM-written when available, else deterministic."""
    try:
        from agent_runtime.supervisor.prompts import QUERY_REFINEMENT_PROMPT

        if llm is not None:
            titles = "\n".join(
                f"- {_doc_field(d, 'title', 'name', default='Untitled')[:90]}" for d in docs[:6]
            ) or "(nothing was returned)"
            prompt = QUERY_REFINEMENT_PROMPT.format(
                query=query, tried="\n".join(f"- {t}" for t in tried), titles=titles)
            raw = _content_to_text(llm.invoke(prompt)) if hasattr(llm, "invoke") else str(llm(prompt))
            candidate = " ".join(str(raw or "").strip().splitlines()[:1]).strip().strip('"')
            if candidate and candidate.upper() != "NONE" and candidate not in tried:
                return candidate[:200]
    except Exception:
        pass
    return _fallback_refinement(query, tried)


def _distill(state: SupervisorState) -> Dict[str, Any]:
    """Compact progress view for the supervisor.

    Deliberately excludes the heavy documents, but DOES include enough about them — titles,
    per-method counts, topical coverage — for the decider to judge whether the evidence answers
    the request. Counts alone ("8 documents") cannot distinguish 8 on-topic hits from 8 unrelated
    ones, which is why an off-topic result set used to end the loop as if it had succeeded.
    """
    docs = state.get("evidence") or []
    audit = state.get("audit") or {}
    actions = list(state.get("actions") or [])
    query = state.get("query", "")

    sources: Dict[str, int] = {}
    for d in docs:
        key = str(_doc_field(d, "source", default="") or "unknown")
        sources[key] = sources.get(key, 0) + 1

    titles = [_doc_field(d, "title", "name", default="Untitled")[:90] for d in docs[:6]]
    scores = [d.get("score") for d in docs if isinstance(d, dict) and isinstance(d.get("score"), (int, float))]
    artifacts = _collect_image_artifacts(state.get("analysis_results"), state.get("code_result"))

    def _peer_summary(result: Any) -> Optional[str]:
        if isinstance(result, dict):
            text = str(result.get("summary") or result.get("answer") or "").strip()
            return text[:220] or None
        return None

    return {
        "has_evidence": bool(docs),
        "document_count": len(docs),
        # WHAT was retrieved, not just how much — the decider can now spot off-topic results.
        "evidence_titles": titles,
        "evidence_sources": sources,
        "topical_coverage": _term_coverage(docs, query),
        "top_score": round(max(scores), 3) if scores else None,
        "queries_searched": list(state.get("searched_queries") or []),
        "has_analysis": state.get("analysis_results") is not None,
        "analysis_summary": _peer_summary(state.get("analysis_results")),
        "has_code": state.get("code_result") is not None,
        "code_summary": _peer_summary(state.get("code_result")),
        "artifacts_produced": [a.get("filename") for a in artifacts],
        "has_answer": bool((state.get("answer") or "").strip()),
        "audit_severity": audit.get("severity"),
        "pending_needs": [n.get("capability") for n in (state.get("needs") or []) if isinstance(n, dict)],
        "actions_taken": actions,
        "action_counts": {c: actions.count(c) for c in ("search", "analyze", "code") if actions.count(c)},
        "search_attempts": state.get("search_attempts", 0),
        "search_exhausted": _search_exhausted(state),
    }


_CAPABILITIES = ("search", "analyze", "code")


def _extract_needs(result: Any):
    """Split a worker result into ``(clean_result, [request, ...])``.

    A worker signals what it needs by returning a dict containing a ``needs`` key —
    a list of capability names (``"search"``/``"analyze"``/``"code"``) or
    ``{"capability", "reason"}`` dicts it wants fulfilled before its work completes.
    """
    if isinstance(result, dict) and result.get("needs"):
        raw = result.get("needs") or []
        clean = {k: v for k, v in result.items() if k != "needs"}
        norm: List[Dict[str, Any]] = []
        for n in raw:
            if isinstance(n, str) and n in _CAPABILITIES:
                norm.append({"capability": n, "reason": ""})
            elif isinstance(n, dict) and n.get("capability") in _CAPABILITIES:
                norm.append({"capability": n["capability"], "reason": str(n.get("reason") or "")})
        return clean, norm
    return result, []


def _enqueue_needs(existing: Optional[List[Dict[str, Any]]], raw_needs: List[Dict[str, Any]], requester: str):
    """Append the requested capabilities + a re-run of the requester to the queue."""
    if not raw_needs:
        return None
    queue = [{**n, "by": requester} for n in raw_needs]
    queue.append({"capability": requester, "reason": "re-run after needs met", "by": requester})
    return [*(existing or []), *queue]


def _make_request_tool():
    """A `request_capability` tool an agent can call to signal what it needs.

    Returns ``(tool, requests)`` where ``requests`` accumulates the agent's calls.
    A tool call is structured LLM output, so this makes the "needs" signal
    model-driven — the agent decides, mid-reasoning, that it needs another peer.
    """
    from langchain_core.tools import StructuredTool

    requests: List[Dict[str, str]] = []

    def request_capability(capability: str, reason: str = "") -> str:
        cap = (capability or "").strip().lower()
        if cap in _CAPABILITIES:
            requests.append({"capability": cap, "reason": reason or ""})
            return (
                f"Recorded request for '{cap}'. The supervisor will fulfill it and re-run "
                "you afterward; stop now and do not guess the missing information."
            )
        return f"Ignored: '{capability}' is not a known capability (search/analyze/code)."

    tool = StructuredTool.from_function(
        func=request_capability,
        name="request_capability",
        description=(
            "Request another capability (search/analyze/code) when you cannot complete your "
            "task without it — e.g. you need evidence from the knowledge base, prior analysis "
            "results, or generated code. The supervisor fulfills the request and re-runs you."
        ),
    )
    return tool, requests


def _heuristic_decision(distilled: Dict[str, Any]) -> str:
    """Fallback decider: search once if there's no evidence, then finish.

    (The LLM decider drives analyze/code; this only prevents runaway loops.)
    """
    if not distilled.get("has_evidence") and "search" not in distilled.get("actions_taken", []):
        return "search"
    return "done"


def _is_unproductive_repeat(nxt: str, state: SupervisorState) -> bool:
    """True if *nxt* re-runs the peer that JUST ran and already produced a result,
    with no pending need driving it.

    Applies to ``analyze`` / ``code``: each overwrites a single result slot and
    iterates internally (the code peer runs+debugs its own code), so re-running it
    back-to-back with the same inputs just reproduces the same result — the
    signature of a decision loop. ``search`` is intentionally NOT guarded: it
    *accumulates* (dedup-merges) into evidence, so a follow-up search can add new
    documents. A genuine multi-hop refinement interleaves a *different* peer (or a
    request_capability need), so only consecutive same-peer repeats are blocked.
    """
    actions = state.get("actions") or []
    if not actions or actions[-1] != nxt:
        return False  # not a back-to-back repeat
    if nxt == "code":
        return state.get("code_result") is not None
    if nxt == "analyze":
        return state.get("analysis_results") is not None
    return False


# Severities at which the grounding audit appends a user-visible caveat. Only HIGH —
# confident factual fabrications/contradictions. Reasonable interpretive elaboration is rated
# none/low (and occasionally medium by a strict small judge); warning on those is a false
# positive that erodes trust, so it does not surface a caveat.
_AUDIT_FLAG_SEVERITIES = {"high"}


def _audit_flagged(audit: Optional[Dict[str, Any]]) -> bool:
    """Whether the grounding audit found a problem worth warning the user about.

    Gated on SEVERITY (not the raw ``hallucination_detected`` flag): the auditor sets that flag
    true even for soft medium-severity over-reach, so keying off it would re-introduce the
    false positives this gate exists to suppress.
    """
    if not audit:
        return False
    severity = str(audit.get("severity") or "").strip().lower()
    return severity in _AUDIT_FLAG_SEVERITIES


def _apply_grounding_caveat(answer: str, audit: Optional[Dict[str, Any]]) -> str:
    """Append a clearly-marked grounding caveat to *answer* when the audit flags it.

    This is what makes the grounding audit non-cosmetic: a flagged verdict changes
    the text the user actually sees, rather than being computed and discarded.
    """
    if not _audit_flagged(audit):
        return answer
    severity = str((audit or {}).get("severity") or "").strip().lower()
    summary = str((audit or {}).get("summary") or "").strip()
    note = (
        "⚠️ Grounding check: parts of this answer may not be fully supported by the "
        "retrieved evidence"
    )
    if severity:
        note += f" (severity: {severity})"
    note += f". {summary}" if summary else "."
    return f"{answer}\n\n---\n\n{note}" if (answer or "").strip() else note


_ARTIFACT_CLAIM_MARKERS = (
    "generat", "creat", "produc", "render", "successfully", "download",
    "available", "you can view", "here is the", "has been",
)
# When the auditor's OWN reason concedes the claim is fine, the issue is a self-contradicting
# false positive — drop it, unless the reason also carries a genuine contradiction marker.
_GROUNDED_REASON_MARKERS = (
    "is grounded", "are grounded", "claim is grounded", "is fully grounded", "fully grounded",
    "is supported", "are supported", "execution record supports", "record supports",
    "is correct", "is accurate", "is consistent", "rather than a full hallucination",
    "matches the execution", "matches the record", "this claim is grounded",
)
_CONTRADICTION_MARKERS = (
    "not supported", "not grounded", "unsupported", "no evidence", "no basis", "no support",
    "fabricat", "invent", "made up", "hallucinat", "unverified", "incorrect", "is wrong",
    "does not match", "doesn't match", "cannot be", "can't be",
)


def _claim_numbers(text: str) -> List[str]:
    """Significant (3+ digit) numbers in a claim, comma-normalized for record matching."""
    return [m.replace(",", "") for m in re.findall(r"\d[\d,]{2,}", text or "")]


def _reconcile_audit_with_artifacts(audit: Optional[Dict[str, Any]],
                                    artifacts: List[Dict[str, str]],
                                    execution_context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Deterministic override of LLM-auditor false positives. Drops an audit issue when it:
    (1) merely disputes artifact generation/availability and an artifact WAS produced,
    (2) disputes a numeric value that actually appears in the execution record, or
    (3) carries a reason that itself concedes the claim is grounded/correct (and no genuine
    contradiction marker). The verdict is cleared if no substantive issues remain. Genuine
    unsupported claims (a wrong statistic, an invented finding) are preserved."""
    if not _audit_flagged(audit):
        return audit
    issues = (audit or {}).get("issues") or []
    blob = ""
    if execution_context is not None:
        try:
            blob = json.dumps(execution_context, default=str)
        except Exception:
            blob = str(execution_context)
        blob = blob.replace(",", "")
    kept = []
    for it in issues:
        if isinstance(it, dict):
            claim = str(it.get("claim") or "").lower()
            reason = str(it.get("reason") or "").lower()
        else:
            # Tolerate a malformed issue (e.g. a bare string) from a strict small judge that
            # ignored the {claim, reason} schema — never crash synthesize over audit shape.
            claim, reason = str(it or "").lower(), ""
        if artifacts and any(m in claim for m in _ARTIFACT_CLAIM_MARKERS):
            continue  # (1) artifact dispute, but an artifact was produced
        if any(g in reason for g in _GROUNDED_REASON_MARKERS) and not any(c in reason for c in _CONTRADICTION_MARKERS):
            continue  # (3) the auditor's own reason concedes grounding
        nums = _claim_numbers(claim)
        if nums and blob and all(n in blob for n in nums):
            continue  # (2) every disputed number is present in the execution record
        kept.append(it)
    if not kept:
        return {"hallucination_detected": False, "severity": "none", "issues": [],
                "summary": "Grounded: flagged claims are supported by the produced artifact(s) and the execution record."}
    return {**(audit or {}), "issues": kept}


# A request that genuinely needs I-GUIDE evidence: asking for platform content (elements,
# datasets, notebooks, publications, code, OERs), a specific element/id, or a search/listing.
# Everything else — general geospatial/technical questions, definitions, how-tos, chit-chat — can
# be answered from the model's own knowledge, so an empty knowledge base must not produce a
# refusal for those.
_RETRIEVAL_REQUEST_RE = re.compile(
    r"\b(?:find|search|look\s+up|list|show\s+me|any|which|recommend|suggest)\b[^.?!]*"
    r"\b(?:datasets?|notebooks?|publications?|papers?|oers?|elements?|code|collections?|"
    r"resources?|maps?|contributors?|authors?)\b"
    r"|\bknowledge\s+elements?\b|\bon\s+(?:the\s+)?i-?guide\b|\bin\s+(?:the\s+)?(?:platform|kb|"
    r"knowledge\s+base)\b|\brelated\s+(?:elements?|resources?)\b|\bmost\s+popular\b"
    r"|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.I,
)


def _needs_kb_evidence(query: str) -> bool:
    """True when the request is for I-GUIDE platform content (so 'no evidence' is a real answer)."""
    return bool(_RETRIEVAL_REQUEST_RE.search(query or ""))


def _has_grounding(evidence: Any, analysis_results: Any, code_result: Any,
                   artifacts: List[Dict[str, str]]) -> bool:
    """True if the run has ANY real basis for an answer: retrieved evidence, a produced
    artifact, or a peer that returned actual content. Used to refuse fabricating an answer
    when nothing was retrieved or produced (e.g. the search backend is down / empty KB)."""
    if evidence or artifacts:
        return True
    for r in (analysis_results, code_result):
        if isinstance(r, dict):
            if str(r.get("answer") or r.get("summary") or "").strip():
                return True
            if r.get("tool_results") or r.get("file_id") or r.get("artifacts"):
                return True
        elif r:
            return True
    return False


_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", ".avif")


def _collect_image_artifacts(*sources: Any) -> List[Dict[str, str]]:
    """Walk peer results (analysis_results / code_result, incl. JSON-encoded tool outputs)
    for image artifacts and return ``[{filename, download_url, file_id}]`` (deduped, ordered).
    """
    found: Dict[str, Dict[str, str]] = {}

    def walk(v: Any) -> None:
        if isinstance(v, dict):
            url = v.get("download_url")
            name = str(v.get("filename") or "")
            if url and name.lower().endswith(_IMAGE_EXTS):
                key = str(v.get("file_id") or url)
                found.setdefault(key, {"filename": name, "download_url": str(url),
                                       "file_id": str(v.get("file_id") or "")})
            for child in v.values():
                walk(child)
        elif isinstance(v, (list, tuple)):
            for child in v:
                walk(child)
        elif isinstance(v, str):  # tool results are usually JSON-encoded strings
            s = v.strip()
            if s[:1] in ("{", "["):
                try:
                    walk(json.loads(s))
                except Exception:
                    pass

    for src in sources:
        walk(src)
    return list(found.values())


def _raw_history_text(chat_history: Optional[List[Any]]) -> str:
    """Concatenate raw chat-history contents (NOT image-stripped) for reference scanning."""
    parts: List[str] = []
    for item in chat_history or []:
        if isinstance(item, dict) and "content" in item:
            parts.append(str(item.get("content") or ""))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            parts.append(str(item[1]))
        else:
            parts.append(str(item))
    return "\n".join(parts)


def _drop_previously_shown(images: List[Dict[str, str]], chat_history: Optional[List[Any]]) -> List[Dict[str, str]]:
    """Drop artifacts already displayed in an EARLIER turn.

    An image/map/plot belongs to the turn that produced it. The code peer keeps a
    checkpointed thread, so on later turns its replayed tool results re-surface a
    prior turn's artifact in ``code_result``; without this filter the synthesizer
    would embed that stale artifact again (the reported "image carried across
    chats" bug). We treat an artifact as already-shown if its download_url or its
    file_id (as a ``/<id>/`` URL path segment) appears anywhere in prior history —
    the same match rule ``_append_image_embeds`` uses for the current answer.
    """
    if not images:
        return images
    history = _raw_history_text(chat_history)
    if not history:
        return images
    kept: List[Dict[str, str]] = []
    for img in images:
        url = img.get("download_url") or ""
        fid = img.get("file_id") or ""
        if (url and url in history) or (fid and (f"/{fid}/" in history or f"/{fid}?" in history)):
            continue
        kept.append(img)
    return kept


def _collect_download_refs(*sources: Any) -> Dict[str, List[str]]:
    """Every artifact this run registered: ``{"file_ids": [...], "urls": [...]}``.

    Unlike :func:`_collect_image_artifacts` this is not limited to images — a GeoJSON/CSV the
    answer offers for download must be verifiable too. Used to reject links that merely LOOK
    like agent files (fabricated hosts, internal paths).
    """
    ids: List[str] = []
    urls: List[str] = []

    def walk(v: Any) -> None:
        if isinstance(v, dict):
            url, fid = v.get("download_url"), v.get("file_id")
            if url:
                urls.append(str(url))
            if fid:
                ids.append(str(fid))
            for child in v.values():
                walk(child)
        elif isinstance(v, (list, tuple)):
            for child in v:
                walk(child)
        elif isinstance(v, str):
            t = v.strip()
            if t[:1] in ("{", "["):
                try:
                    walk(json.loads(t))
                except Exception:
                    pass

    for src in sources:
        walk(src)
    return {"file_ids": list(dict.fromkeys(ids)), "urls": list(dict.fromkeys(urls))}


def _append_image_embeds(answer: str, images: List[Dict[str, str]]) -> str:
    """Append markdown image embeds for produced images not already referenced in *answer*.

    This guarantees a generated map/plot renders inline in the (always-delivered) final
    answer, independent of whether the model embedded it and of detail-event gating.
    """
    if not images:
        return answer or ""
    body = answer or ""
    blocks: List[str] = []
    for img in images:
        url, name, fid = img["download_url"], img["filename"], img.get("file_id") or ""
        # Skip if already referenced: by exact url, or by the file_id as a URL path segment
        # (/<file_id>/) — a bare-substring match would false-trip on incidental occurrences.
        if (url and url in body) or (fid and (f"/{fid}/" in body or f"/{fid}?" in body)):
            continue
        blocks.append(f"![{name}]({url})")
    if not blocks:
        return body
    sep = "\n\n" if body.strip() else ""
    return f"{body}{sep}" + "\n\n".join(blocks)


# Markdown image embed: ![alt](url). Stripped from replayed history so the
# synthesizer can't re-embed a plot/map produced in an EARLIER turn into the
# current answer (it should only embed artifacts from the current turn's
# evidence/results). The alt text is kept as a plain marker so the model still
# knows an image was shown before, just without a URL it can copy.
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")


def _strip_image_markdown(text: Any) -> str:
    return _MD_IMAGE_RE.sub(
        lambda m: f"[image shown earlier: {m.group(1).strip() or 'figure'}]", str(text)
    )


def _format_chat_history(chat_history: Optional[List[Any]], *, max_items: int = 8, max_chars: int = 4000) -> str:
    """Render recent chat history as compact 'role: content' lines for prompts."""
    if not chat_history:
        return ""
    lines: List[str] = []
    for item in list(chat_history)[-max_items:]:
        if isinstance(item, dict) and "role" in item and "content" in item:
            role, content = item.get("role"), item.get("content")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            role, content = item[0], item[1]
        else:
            role, content = "user", item
        lines.append(f"{role}: {_strip_image_markdown(content)}")
    text = "\n".join(lines)
    return text if len(text) <= max_chars else "…" + text[-max_chars:]


def default_decide_fn(llm: Optional[Any] = None) -> DecideFn:
    """LLM-driven next-action chooser with a deterministic heuristic fallback."""

    def decide(state: SupervisorState, distilled: Dict[str, Any]) -> str:
        history = _format_chat_history(state.get("chat_history"))
        prompt = (
            "You are the orchestration supervisor for a geospatial research agent.\n"
            "Choose the SINGLE next action. Capabilities are peers you can use in any "
            "order and repeat as needed:\n"
            "- search: retrieve evidence (datasets, publications, notebooks)\n"
            "- analyze: run a GIS/data analysis workflow with EXISTING purpose-built tools "
            "(QGIS/PyQGIS, spatial ops, statistics, vector inspect/plot/reproject) over the "
            "evidence or uploaded files\n"
            "- code: produce and run NEW code for work no existing tool covers\n"
            "- done: stop; a grounded final answer is composed automatically from the "
            "conversation + evidence + analysis results + code\n\n"
            "ANALYZE BEFORE CODE: for any analysis/GIS/mapping task, pick 'analyze' FIRST — it "
            "owns the purpose-built tools and is more reliable than writing fresh code. Choose "
            "'code' only when analyze has already run and could not do it (has_analysis is true "
            "but the task is unmet, or analyze reported a missing capability), or when the user "
            "explicitly asks for code/a script. Do not start with 'code' for a task an existing "
            "tool plausibly covers.\n"
            "Use the conversation so far for context. If the request refers to something "
            "ALREADY produced earlier in the conversation (e.g. 'show me the code', 'explain "
            "that', 'what did you find'), do NOT search again — choose 'done' so the answer is "
            "composed from the conversation, unless genuinely new external information is needed.\n"
            "Each peer ITERATES INTERNALLY (the code peer runs AND debugs its own code; search "
            "issues multiple queries in one pass). So once a peer has produced its result "
            "(see has_code / has_analysis / has_evidence and action_counts in Progress), do NOT "
            "pick it again to 'retry' or 'improve' — that just repeats work. Choose 'done' once "
            "the request is covered; the final answer is composed automatically. Only pick a peer "
            "again if you genuinely need NEW work it has not done yet.\n"
            "If 'search_exhausted' is true in Progress, the knowledge base returned nothing new — "
            "do NOT choose 'search' again. Proceed with analyze/code (which can work on uploaded "
            "files and prior results) or choose 'done'.\n"
            "Peers may also REQUEST a capability they need (e.g. code needs evidence); such "
            "requests are fulfilled automatically before you are consulted again.\n\n"
            "Respond ONLY with JSON: {\"next\": \"search|analyze|code|done\", \"reason\": \"...\"}\n\n"
            + (f"Conversation so far:\n{history}\n\n" if history else "")
            + f"User request:\n{state.get('query', '')}\n\n"
            + f"Progress so far:\n{json.dumps(distilled, ensure_ascii=True)}\n"
        )
        try:
            active = llm
            if active is None:
                from agent_runtime.executor_factory import build_default_llm

                active = build_default_llm()
            raw = active.invoke(prompt) if hasattr(active, "invoke") else active(prompt)
            text = _content_to_text(raw)
            # Reuse the fenced-block-aware extractor (handles ```json fences and
            # prose around the object) instead of naive first-{/last-} slicing.
            parsed = _extract_json_object(text)
            nxt = str((parsed or {}).get("next") or "").strip().lower() if isinstance(parsed, dict) else ""
            if nxt in ALLOWED_ACTIONS:
                # Surface the model's stated rationale as a (detail-tier) trace event.
                # The decider contract returns only the action string, so without this
                # the 'reason' the LLM produced would be discarded.
                reason = str((parsed or {}).get("reason") or "").strip() if isinstance(parsed, dict) else ""
                emit_trace_event(
                    "supervisor_decision",
                    {"stage": "supervisor", "next": nxt, "reason": reason,
                     "message": nxt + (f" — {reason}" if reason else "")},
                    node="supervisor",
                )
                return nxt
        except Exception:
            pass
        # The decider output was unusable; fall back to the deterministic heuristic.
        # Emit a (detail-tier) marker so this degraded path is distinguishable from a
        # genuine LLM "done" in the trace.
        emit_trace_event(
            "decider_fallback",
            {"stage": "supervisor", "message": "decider output not parseable; used heuristic fallback"},
            node="supervisor",
        )
        return _heuristic_decision(distilled)

    return decide


# ---------------------------------------------------------------------------
# Default worker adapters (best-effort; wire to existing agents — need live validation)
# ---------------------------------------------------------------------------

def _as_retrieval_request(query: str) -> str:
    """Reframe the (possibly action-shaped) user query as a RETRIEVAL task for the
    search peer, so it gathers evidence instead of trying to perform the task itself
    (which makes capable models loop). The original query is still used for citation."""
    return (
        "Retrieve relevant evidence for the request below. Use the search tools to "
        "gather documents/code, then STOP and return the evidence — do NOT perform "
        "the task, write code, or produce the final answer yourself.\n\n"
        f"Request: {query}"
    )


# --- related-knowledge-element lookup (deterministic two-bucket) --------------
# A "related elements of <UUID>" request is NOT a generic retrieval — it has a precise,
# deterministic answer. We split it into two clearly-separated buckets so the agent never
# again presents a similarity search as if it were a curated relationship (the bug that
# triggered a HIGH grounding flag):
#   * CURATED  — contributor-specified :RELATED neighbors from the Neo4j graph (authoritative).
#   * CONTENT  — semantically similar elements (explicitly NOT curated links).
# Each doc is tagged ``provenance`` so the formatter renders two labeled sections and the
# grounding auditor distinguishes a curated link from a topical match.
_UUID_RE = re.compile(
    r"\b([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\b"
)
_RELATED_INTENT_RE = re.compile(r"\b(related|connected|linked|associated|relationship)s?\b", re.I)


def _detect_related_elements_request(query: str) -> Optional[str]:
    """Return the element UUID iff *query* asks for the related elements of a specific element.

    Deterministic gate: requires BOTH an element UUID and a related/connected intent word, so
    a generic "datasets related to floods" (no UUID) is left to normal search.
    """
    if not query or not _RELATED_INTENT_RE.search(query):
        return None
    m = _UUID_RE.search(query)
    return m.group(1) if m else None


def _related_elements_evidence(element_id: str, *, depth: int = 2,
                               curated_cap: int = 25, content_k: int = 6) -> List[Dict[str, Any]]:
    """Deterministic two-bucket evidence for a related-elements request.

    Bucket 1 (CURATED): contributor-specified related elements — the Neo4j :RELATED traversal
    first; when the graph yields nothing (missing edges, stale node, auth failure) fall back to
    the platform API's ``related-elements`` field (what the contributor actually set). Bucket 2
    (CONTENT): semantically similar elements, explicitly framed as similarity (never curated).
    The SEED element itself is included first (tagged ``provenance='seed'``) so the synthesizer
    can name the queried resource instead of guessing its identity. Every doc is tagged
    ``provenance`` ('seed' | 'curated' | 'content'). Never raises — degrades to whatever it
    could gather.
    """
    docs: List[Dict[str, Any]] = []
    seen_ids = {str(element_id)}
    seed_title = ""

    # bucket 1 — curated graph relationships (authoritative when present)
    try:
        from rag_pipeline.search.agents import explore_neo4j_related_nodes

        payload = explore_neo4j_related_nodes(element_id, depth=depth, limit=50) or {}
    except Exception:
        payload = {}
    seed_title = str((payload.get("seed") or {}).get("title") or "").strip()
    curated: List[Dict[str, Any]] = []
    for d in (payload.get("documents") or [])[:curated_cap]:
        if not isinstance(d, dict):
            continue
        did = str(d.get("doc_id") or d.get("id") or "")
        if did and did in seen_ids:
            continue
        if did:
            seen_ids.add(did)
        tagged = dict(d)
        tagged["provenance"] = "curated"
        tagged.setdefault("source", "graph")
        curated.append(tagged)

    # Platform metadata: the authoritative element identity + the contributor-specified
    # related-elements list (used as the curated fallback when the graph has no edges).
    meta: Dict[str, Any] = {}
    try:
        from agent_runtime.element_resolver import resolve_element

        meta = resolve_element(element_id) or {}
    except Exception:
        meta = {}
    api_title = str(meta.get("title") or "").strip()
    if api_title:
        # The platform is the source of truth for the element's identity; a stale graph node
        # can carry a different title (observed live), which would mislabel the whole answer.
        seed_title = api_title

    # curated FALLBACK: the platform API's contributor-specified related-elements.
    if not curated:
        for rel in (meta.get("related") or [])[:curated_cap]:
            if not isinstance(rel, dict):
                continue
            rid = str(rel.get("element_id") or rel.get("id") or "")
            if not rid or rid in seen_ids:
                continue
            seen_ids.add(rid)
            curated.append({
                "doc_id": rid,
                "title": str(rel.get("title") or "Untitled"),
                "element_type": str(rel.get("resource_type") or rel.get("resource-type") or "resource"),
                "contents": "",
                "provenance": "curated",
                "source": "platform_api",
            })

    # SEED element first, so the synthesizer names the queried resource correctly.
    if seed_title:
        docs.append({
            "doc_id": str(element_id),
            "title": seed_title,
            "element_type": str(meta.get("resource_type") or "resource"),
            "contents": str(meta.get("abstract") or "")[:800],
            "provenance": "seed",
            "source": "platform_api" if api_title else "graph",
        })
    docs.extend(curated)

    # bucket 2 — content-similar elements (similarity, explicitly NOT curated)
    if seed_title:
        try:
            from rag_pipeline.search.agents import _hit_to_document
            from rag_pipeline.search.semantic import semantic_search

            hits = semantic_search(seed_title, size=content_k + 4) or []
        except Exception:
            hits = []
        added = 0
        for hit in hits:
            try:
                doc = _hit_to_document(hit, source_name="semantic")
            except Exception:
                continue
            did = str(doc.get("doc_id") or "")
            if did and did in seen_ids:   # exclude the seed AND anything already curated
                continue
            if did:
                seen_ids.add(did)
            doc["provenance"] = "content"
            docs.append(doc)
            added += 1
            if added >= content_k:
                break
    return docs


# Strong popularity-intent phrases only. Deliberately TIGHTER than the graph tier's own
# by_popularity pattern (whose bare "popular" would hijack e.g. "explain popular culture in
# geography"); execution still goes through the same tier-1 dispatch, and detection stricter
# than execution is safe (missed phrasings just take the normal search path).
_POPULARITY_RE = re.compile(
    r"\b(?:most\s+(?:popular|clicked|viewed|visited|accessed)|"
    r"top\s+(?:clicked|viewed|rated)|highest\s+clicks?|trending)\b",
    re.I,
)


def _detect_popularity_request(query: str) -> bool:
    """True iff *query* explicitly asks for a popularity/usage ranking."""
    return bool(_POPULARITY_RE.search(query or ""))


def _popularity_evidence(query: str, *, limit: int = 10) -> List[Dict[str, Any]]:
    """Deterministic popularity lookup: run the graph's tier-1 dispatch (by_popularity Cypher,
    sorted by real click_count) and normalize to evidence docs. The click count is appended to
    each doc's contents so the synthesizer reports actual popularity, not topical similarity.
    Returns [] when the graph yields nothing (caller falls back to the normal search agent)."""
    try:
        from rag_pipeline.search.agents import _hit_to_document, get_neo4j_agent_results

        hits = get_neo4j_agent_results(query, limit=limit) or []
    except Exception:
        return []
    docs: List[Dict[str, Any]] = []
    for hit in hits:
        try:
            doc = _hit_to_document(hit, source_name="neo4j")
        except Exception:
            continue
        clicks = hit.get("_score")
        if isinstance(clicks, (int, float)) and clicks > 0:
            doc["click_count"] = int(clicks)
            doc["contents"] = (str(doc.get("contents") or "").strip() +
                               f"\n[popularity: {int(clicks)} clicks]").strip()
        docs.append(doc)
    return docs


_ELEMENT_LOOKUP_INTENT_RE = re.compile(
    r"\b(explain|describe|summari[sz]e|what\s+is|what'?s|tell\s+me\s+about|"
    r"details?\s+(?:of|about|on|for)|info(?:rmation)?\s+(?:on|about|for)|overview\s+of|about)\b",
    re.I,
)


def _detect_element_lookup_request(query: str) -> Optional[str]:
    """Return the element UUID iff *query* asks to explain/describe a specific element by id, or
    is essentially just a bare UUID. Returns None for a related-elements request (handled by
    _detect_related_elements_request) and for any query without a UUID.
    """
    if not query:
        return None
    m = _UUID_RE.search(query)
    if not m:
        return None
    uuid = m.group(1)
    if query.strip() == uuid:               # the query IS just the id -> look it up
        return uuid
    if _RELATED_INTENT_RE.search(query):    # "related elements of <id>" has its own handler
        return None
    return uuid if _ELEMENT_LOOKUP_INTENT_RE.search(query) else None


def _element_lookup_evidence(element_id: str) -> List[Dict[str, Any]]:
    """Deterministic by-id element fetch for an "explain/describe <UUID>" request. Tries the
    graph node first (rich contents); falls back to the platform backend API so it still works
    when the element isn't in — or the agent can't reach — Neo4j. Returns 0..1 evidence docs.
    Never raises.
    """
    try:
        from rag_pipeline.search.agents import _hit_to_document, get_neo4j_element_by_id_results

        hits = get_neo4j_element_by_id_results(element_id) or []
        if hits:
            doc = _hit_to_document(hits[0], source_name="neo4j")
            if str(doc.get("title") or "").strip() and doc.get("title") != "Untitled":
                return [doc]
    except Exception:
        pass
    try:
        from agent_runtime.element_resolver import resolve_element

        meta = resolve_element(element_id) or {}
        if str(meta.get("title") or "").strip():
            return [{
                "doc_id": element_id,
                "source": "backend_api",
                "title": str(meta.get("title")),
                "element_type": str(meta.get("resource_type") or "resource"),
                "contents": str(meta.get("abstract") or ""),
                "authors": meta.get("authors") or [],
                "tags": meta.get("tags") or [],
            }]
    except Exception:
        pass
    return []


# Follow-up phrasings that refer to the element already under discussion WITHOUT repeating its
# id. Anchored + subject-less so "datasets related to floods" / "explain dam failures" do NOT
# match (those carry their own subject -> normal search), while "what are the related elements"
# / "explain it" DO -> we then recall the element id from the conversation.
_RELATED_FOLLOWUP_RE = re.compile(
    r"^\s*(?:(?:please\s+)?(?:show|list|give|find|get|display|tell)\s+(?:me\s+)?)?"
    r"(?:what(?:'?s| are| is)(?:\s+(?:it|this|that))?\s+)?(?:the\s+|its\s+|their\s+)?"
    r"related(?:\s+knowledge)?(?:\s+(?:elements?|nodes?|resources?|ones?|items?))?"
    r"\s*(?:to|for|of)?\s*(?:it|this|that)?\s*\??\s*$",
    re.I,
)
_EXPLAIN_FOLLOWUP_RE = re.compile(
    r"^\s*(?:please\s+)?(?:explain|describe|summari[sz]e|tell\s+me(?:\s+more)?(?:\s+about)?|"
    r"what(?:'?s| is)|more\s+(?:details?|info(?:rmation)?)|info(?:rmation)?|details?)\s+"
    r"(?:about\s+|on\s+|of\s+|for\s+)?"
    r"(?:it|this|that|(?:the|this|that)\s+(?:element|dataset|resource|item|one|notebook|publication))"
    r"\s*\??\s*$",
    re.I,
)


def _chat_item_text(item: Any) -> str:
    """Flatten any chat-history item shape ({role,content} | {userQuery,answer} | (role,content)
    | raw) into one searchable string."""
    if isinstance(item, dict):
        return " ".join(str(item.get(k) or "") for k in ("content", "userQuery", "answer", "text", "query"))
    if isinstance(item, (list, tuple)):
        return " ".join(str(x) for x in item)
    return str(item or "")


def _chat_item_user_text(item: Any) -> str:
    """USER-authored text only, so id-recall keys off the subject the user actually stated — not a
    UUID the assistant merely cited in a prior answer (citation URLs embed a related element's
    UUID, which would otherwise hijack a follow-up to the wrong element). Returns "" for
    assistant/system/tool turns; for the {userQuery, answer} turn shape, only the query side.
    """
    if isinstance(item, dict):
        role = str(item.get("role") or item.get("type") or "").strip().lower()
        if role == "user":
            return str(item.get("content") or item.get("text") or item.get("query") or "")
        if role:                       # assistant / system / tool -> not user text
            return ""
        return str(item.get("userQuery") or item.get("query") or "")   # {userQuery, answer} turn
    if isinstance(item, (list, tuple)) and item:
        return " ".join(str(x) for x in item[1:]) if str(item[0]).strip().lower() == "user" else ""
    return str(item or "")


def _recall_recent_element_id(chat_history: Optional[List[Any]], *, max_items: int = 8) -> Optional[str]:
    """The element UUID a follow-up that omits the id ('what are the related elements', 'explain
    it') should resolve to. Prefers the user's OWN most-recent UUID (their stated subject) over a
    UUID the assistant merely cited in a prior answer; falls back to any mention only when the
    user never typed one (assistant-only reference / unknown history shapes). Newest-first.
    """
    if not chat_history:
        return None
    window = list(chat_history)[-max_items:]
    for item in reversed(window):              # pass 1: the user's stated subject
        m = _UUID_RE.search(_chat_item_user_text(item))
        if m:
            return m.group(1)
    for item in reversed(window):              # pass 2: fall back to any mention
        m = _UUID_RE.search(_chat_item_text(item))
        if m:
            return m.group(1)
    return None


# --- coverage floor: query features that IMPLY a retrieval method ------------------
# Tool choice is the LLM's, but a needed method must never be skipped. These cheap detectors let
# the deterministic sweep add the implied methods after the peer runs (observed live: "satellite
# imagery of wildfires in California" used neither spatial_search nor opengeodata_search).
_GEO_NOUN_RE = re.compile(
    r"\b(count(?:y|ies)|states?|provinces?|cit(?:y|ies)|towns?|villages?|rivers?|lakes?|basins?|"
    r"watersheds?|regions?|coasts?|islands?|mountains?|valleys?|deltas?|national\s+parks?|"
    r"municipalit(?:y|ies)|districts?|prefectures?|catchments?)\b", re.I)
# "in/near/across <Capitalized>" — a place, unless it follows an authorship cue ("by <Name>").
_PLACE_PHRASE_RE = re.compile(
    r"(?<!\bby)\b(?:in|near|around|within|across|throughout|along|over)\s+(?:the\s+)?"
    r"([A-Z][\w'.-]+(?:\s+[A-Z][\w'.-]+)?)")
_EXTERNAL_DATA_RE = re.compile(
    r"\b(satellite|imagery|remote[\s-]?sensing|earth\s+observation|landsat|sentinel|modis|viirs|"
    r"aster|dem|lidar|elevation|land\s?cover|land\s+use|climate|weather|precipitation|rainfall|"
    r"temperature|reanalysis|census|acs|noaa|nasa|usgs|epa|open\s+data|public\s+data|"
    r"external\s+data|third[\s-]party|global\s+dataset)\b", re.I)


_AUTHORSHIP_RE = re.compile(r"\bby\s+[A-Z]")


def _mentions_place(query: str) -> bool:
    """True when the request names a location (so place-aware search is implied).

    An explicit geographic noun always counts. Otherwise a capitalized "in/near/across X" phrase
    counts — except in an author-scoped request, where such a phrase is usually a venue or a
    surname ("papers by Wang in Nature"), not a place.
    """
    text = query or ""
    if _GEO_NOUN_RE.search(text):
        return True
    if _AUTHORSHIP_RE.search(text):
        return False
    return bool(_PLACE_PHRASE_RE.search(text))


def _wants_external_data(query: str) -> bool:
    """True when the request is for data types that live in EXTERNAL open-data catalogs."""
    return bool(_EXTERNAL_DATA_RE.search(query or ""))


# KB tools given to the peers that WRITE AND RUN code. Hoisted to module scope because both
# the code peer and the analyze peer need the same set, and the analyze peer also holds
# `execute_code` — a peer that can run analysis but cannot discover an existing callable method
# will re-implement it. Deliberately independent of the request's enabled_search_methods: this
# is a capability of those peers, not a per-request search preference.
_CODE_PEER_KB_TOOLS = {"agent_kb_search", "get_kb_block",
                       "kb_method_search", "get_method_contract"}


def _method_units_as_documents(query: str, k: int) -> List[Dict[str, Any]]:
    """Library methods rendered as evidence documents.

    The contents field carries the signature, summary AND the exact import line, because the
    import line is the part that makes the answer actionable — an answer that names a method
    without saying how to import it is barely better than naming a notebook.
    """
    from agent_runtime.method_library import search_methods

    hits = [h for h in search_methods(query, limit=max(1, min(int(k), 8)))
            if not h.get("ambiguous")]
    # A relevance floor RELATIVE to the best hit. Unlike the tool — where the model can judge a
    # ranked list for itself — this path spends evidence slots without being asked, so it takes
    # only the clear matches. Measured on "choropleth map of Chicago crime": plot_choropleth_map
    # 23.0 and load_chicago_crime_data 21.0 are the answer, while display_code_txt and
    # process_weather_file_to_24h score 6.5 purely on the generic words "code", "data" and "map".
    if hits:
        floor = 0.4 * float(hits[0].get("score") or 0.0)
        hits = [h for h in hits if float(h.get("score") or 0.0) >= floor][:4]

    docs: List[Dict[str, Any]] = []
    for hit in hits:
        symbol = str(hit.get("symbol") or "")
        contents = "\n".join(filter(None, [
            str(hit.get("signature") or ""),
            str(hit.get("doc_summary") or ""),
            f"import: {hit['import_line']}" if hit.get("import_line") else "",
            f"requires: {', '.join(hit.get('requirements') or [])}"
            if hit.get("requirements") else "",
        ]))
        docs.append({
            "doc_id": f"method::{symbol}",
            "title": f"{symbol.split('.')[-1]} — callable method",
            "contents": contents,
            "source": "method_library",
            "resource_type": "MethodUnit",
            # Cite the SOURCE ELEMENT, not the synthetic method id: a unit is evidence about
            # the element it came from, and that is the id a reader can open.
            "citation_ids": [hit["element_id"]] if hit.get("element_id") else [],
            "element_id": hit.get("element_id"),
            "import_line": hit.get("import_line"),
            "score": hit.get("score"),
        })
    return docs


def _direct_search_sweep(query: str, enabled_search_methods: Optional[List[str]],
                         *, k: Optional[int] = None) -> List[Dict[str, Any]]:
    """Deterministic multi-method retrieval sweep: run keyword AND semantic search directly
    (cheap OpenSearch calls, no LLM) so every search turn has baseline coverage from BOTH
    core methods regardless of which tools the LLM SearchAgent chose to call — it frequently
    stops after a single tool, leaving results incomplete. Respects the request's
    enabled_search_methods allowlist. Never raises; each method degrades independently."""
    # Resolved here, not as a default argument: a default binds at import and would freeze
    # the window regardless of AGENT_SEARCH_TOP_K.
    if k is None:
        from rag_pipeline.search.utils import default_top_k
        k = default_top_k()
    allow = ({str(m).strip() for m in enabled_search_methods}
             if enabled_search_methods is not None else None)

    def permitted(name: str) -> bool:
        return allow is None or name in allow

    docs: List[Dict[str, Any]] = []

    def _public(hit: Any) -> bool:
        from rag_pipeline.search.neo4j_graph_tools import is_public_visibility

        src = hit.get("_source") if isinstance(hit, dict) else None
        return is_public_visibility((src or {}).get("visibility"))

    if permitted("keyword_search"):
        try:
            from rag_pipeline.search.agents import _hit_to_document
            from rag_pipeline.search.keyword import get_keyword_search_results

            docs.extend(_hit_to_document(h, source_name="keyword")
                        for h in (get_keyword_search_results(query, size=k) or []) if _public(h))
        except Exception:
            pass
    if permitted("semantic_search"):
        try:
            from rag_pipeline.search.agents import _hit_to_document
            from rag_pipeline.search.semantic import semantic_search

            docs.extend(_hit_to_document(h, source_name="semantic")
                        for h in (semantic_search(query, size=k) or []) if _public(h))
        except Exception:
            pass
    # Conditional methods the QUERY implies — added regardless of what the LLM chose to call.
    if permitted("spatial_search") and _mentions_place(query):
        try:
            from rag_pipeline.search.agents import _hit_to_document
            from rag_pipeline.search.spatial import get_spatial_search_results

            docs.extend(_hit_to_document(h, source_name="spatial")
                        for h in (get_spatial_search_results(query, size=k) or []) if _public(h))
        except Exception:
            pass
    if permitted("opengeodata_search") and _wants_external_data(query):
        try:
            # Normalized like the tool payload (keeps url/abstract/provider) so external hits stay
            # citable as links rather than losing their landing page.
            from agent_runtime.langchain_granular_tools import _normalize_hits
            from rag_pipeline.search.opengeodata import get_opengeodata_results

            docs.extend(_normalize_hits(get_opengeodata_results(query, limit=k) or [],
                                        source="opengeodata"))
        except Exception:
            pass
    if permitted("agent_kb_search"):
        try:
            from rag_pipeline.search.agent_kb import agent_kb_search

            payload = agent_kb_search(query, size=k) or {}
            # `documents`, NOT `results` — and they are ALREADY normalized (doc_id, title,
            # contents, parent_doc_id), so they must not be run through _normalize_hits, which
            # expects raw OpenSearch hits. The first version of this block read `results` and
            # re-normalized: no exception, no log, just a permanently empty arm. The `except`
            # below could never have caught it, which is exactly why the wrong key survived.
            for d in payload.get("documents") or []:
                if not isinstance(d, dict):
                    continue
                parent = d.get("parent_doc_id")
                docs.append({**d, "source": "agent_kb",
                             # Cite the ELEMENT, not the block: a block id is not something a
                             # reader can open, and the other sweep arms cite elements.
                             "citation_ids": [parent] if parent else []})
        except Exception:
            pass
    if permitted("kb_method_search"):
        # Local registry read — no network, no cluster, sub-millisecond. Unioned rather than
        # left to tool choice for the reason stated at the top of default_search_fn: the model
        # does not reliably reach for a tool it was merely offered. Measured across three runs
        # of the same question ("is there code I can reuse for a Chicago crime choropleth?"),
        # with the tool registered, policy-allowed, request-enabled and named in the persona's
        # COVERAGE and REUSE rules, the peer called it ZERO times and answered "adapt this
        # notebook" while `plot_choropleth_map` sat in the library with a working import line.
        try:
            docs.extend(_method_units_as_documents(query, k))
        except Exception:
            pass
    # web_search is deliberately NOT part of this sweep. Every other method here is a cheap call to
    # infrastructure we own; the open web is a live third-party network hop, so unioning it in would
    # put every single turn on the internet. It stays LLM-elected (and budget-capped) — plus the
    # last-resort fallback in _web_fallback_evidence, which fires only when the platform found
    # NOTHING.
    return [d for d in docs if isinstance(d, dict)]


# Sources that are NOT the platform: external catalogs and the open web. Everything else counts as
# our own evidence — deliberately the wrong way round from "list the platform's sources", because
# the consequence of a misclassification is asymmetric. Treating an unrecognized document as
# external would send a turn to the web even though we DID find something (an earlier version of
# this check keyed on a positive list of source names and did exactly that, firing whenever a
# document lacked a `source` field). Failing closed only skips the fallback.
_EXTERNAL_SOURCES = {"web", "opengeodata", "datacite"}


def _platform_docs(docs: Any) -> List[Any]:
    """The subset of *docs* that did NOT come from the open web or an external catalog."""
    kept: List[Any] = []
    for doc in docs or []:
        if not isinstance(doc, dict):
            continue
        src = doc.get("document") if isinstance(doc.get("document"), dict) else doc
        if not isinstance(src, dict):
            continue
        name = str(src.get("source") or src.get("source_system") or "").strip().lower()
        etype = str(src.get("element_type") or "").strip().lower()
        if name in _EXTERNAL_SOURCES or etype in _EXTERNAL_SOURCES:
            continue
        kept.append(doc)
    return kept


def _has_platform_evidence(docs: Any) -> bool:
    """Whether the run holds any evidence that did NOT come from the open web or a catalog."""
    return bool(_platform_docs(docs))


def _platform_evidence_is_unhelpful(docs: Any, query: str) -> bool:
    """Whether the platform gave us nothing USEFUL for *query*.

    "Nothing at all" is the wrong bar. Keyword search is a nearest-match engine: it returns its
    eight closest documents for any query, so an unknown subject comes back with a full result set
    that mentions none of it — and the answer then reads "the provided evidence does not include
    specific resources explaining <subject>". Reusing the refinement loop's own judgement
    (empty OR below the topical-coverage floor) makes the fallback fire for exactly that case.
    """
    return _results_are_poor(_platform_docs(docs), query)


# Requests whose subject is I-GUIDE's OWN catalogue. Deliberately narrower than the catalog
# search's intent gate, which is answering a different question: `wants_external_data` treats "find
# datasets … on I-GUIDE" as external (the "find datasets" cue wins) and a standards-version question
# as internal, so neither of its answers is the one needed here.
_PLATFORM_HOLDINGS_RE = re.compile(
    r"(?:\bi-?guide\b"
    r"|\bknowledge element"
    r"|\brelated element"
    r"|\bthis platform\b|\bthe platform\b"
    r"|\bmost (?:popular|viewed|clicked|downloaded)\b|\btrending\b"
    r"|\b(?:uploaded|attached)\s+(?:file|dataset)\b|\bthis (?:file|csv|spreadsheet)\b"
    r"|\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b)",
    re.IGNORECASE,
)


def _asks_about_platform_holdings(query: str) -> bool:
    """Whether the question is about what I-GUIDE itself contains (so the web cannot answer it)."""
    return bool(_PLATFORM_HOLDINGS_RE.search(str(query or "")))


def _web_fallback_enabled() -> bool:
    """Whether to consult the open web when the platform yields nothing (default on)."""
    raw = str(os.getenv("AGENT_WEB_FALLBACK", "")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _emit_web_tool_call(name: str, args: Dict[str, Any]) -> None:
    """Trace a fallback web call in the SAME shape the LLM-elected path emits.

    This path is deterministic, so nothing in the LangChain callback chain reports it: before this,
    a fallback showed up as a single node line and the query it actually sent — and the results it
    got — were invisible to every client. Reusing the ``tool_call``/``tool_result`` event names means
    existing UIs render it with no changes; ``automatic`` marks that no model chose it.
    """
    emit_trace_event(
        "tool_call",
        {"kind": "llm_tool_decision", "label": f"Tool started (automatic) {name}",
         "name": name, "args": args, "automatic": True,
         "tool_calls": [{"name": name, "args": args}],
         "message": f"{name}({json.dumps(args, ensure_ascii=True, default=str)})"},
        node="search",
    )


def _emit_web_tool_result(name: str, payload: Dict[str, Any]) -> None:
    """Trace the RESULT of a fallback web call: what came back, titles and urls only."""
    clean = {k: v for k, v in payload.items() if v is not None}
    body = json.dumps(clean, ensure_ascii=True, default=str)
    emit_trace_event(
        "tool_result",
        {"kind": "tool_result", "label": f"Tool result {name}", "tool_name": name, "name": name,
         "automatic": True, "content": body, "message": body},
        node="search",
    )


def _web_fallback_evidence(query: str, enabled_search_methods: Optional[List[str]],
                           *, k: int = 6) -> List[Dict[str, Any]]:
    """Open-web evidence for a query the I-GUIDE platform could not answer at all.

    This is the one place the web is reached deterministically rather than by the LLM electing it.
    The justification is narrow: when the knowledge base returns nothing, the alternative is telling
    the user we found nothing while a public answer exists.

    It also FETCHES the top result rather than stopping at snippets. On this path the documents go
    straight into evidence and the synthesizer never gets a chance to call web_fetch itself, so
    without the fetch the fallback would supply ~300-character engine snippets as the sole grounding
    for the whole answer — the exact failure the two-step design exists to avoid.
    """
    allow = ({str(m).strip() for m in enabled_search_methods}
             if enabled_search_methods is not None else None)
    if allow is not None and "web_search" not in allow:
        return []

    # A question ABOUT THE PLATFORM's own holdings ("what datasets does I-GUIDE have on X", "the
    # related elements of <uuid>") cannot be answered by the open web — only I-GUIDE knows what
    # I-GUIDE contains. For those an empty result IS the answer, and substituting web pages would
    # dress up a miss as a hit.
    if _asks_about_platform_holdings(query):
        return []

    from rag_pipeline.search import web_utils as WU

    if not WU.web_enabled() or not _web_fallback_enabled():
        return []

    try:
        from agent_runtime.langchain_granular_tools import _normalize_hits
        from rag_pipeline.search.web import results_to_hits, run_web_search

        _emit_web_tool_call("web_search", {"query": query, "limit": k})
        result = run_web_search(query, limit=k)
        if result.get("error") or not result.get("count"):
            _emit_web_tool_result("web_search", {
                "source": "web", "count": result.get("count") or 0,
                "error": result.get("error"), "search_query": result.get("search_query"),
            })
            return []
        docs = _normalize_hits(results_to_hits(result), source="web")
        _emit_web_tool_result("web_search", {
            "source": "web",
            "count": result.get("count") or len(docs),
            "provider": result.get("provider"),
            "search_query": result.get("search_query") or query,
            "candidates_found": result.get("candidates_found"),
            "filtered_out": result.get("filtered_out"),
            # Titles and urls only — the trace shows WHAT was found, not the page bodies.
            "documents": [{"title": d.get("title"), "url": d.get("url")} for d in docs],
        })
    except Exception:
        return []

    # Read the single most promising page so the answer rests on real content, not a snippet.
    try:
        from rag_pipeline.search.web_fetch import fetch_and_extract

        top = next((d.get("url") for d in docs if d.get("url")), "")
        if top:
            _emit_web_tool_call("web_fetch", {"url": top, "focus": query})
            page = fetch_and_extract(top, focus=query)
            text = (page.get("text") or "").strip()
            _emit_web_tool_result("web_fetch", {
                "url": page.get("url") or top,
                "title": page.get("title"),
                "status": page.get("status"),
                "chars": page.get("chars"),
                "paragraphs_kept": page.get("paragraphs_kept"),
                "paragraphs_total": page.get("paragraphs_total"),
                "cached": page.get("cached"),
                "error": page.get("error"),
                "blocked": page.get("blocked"),
            })
            if text and not page.get("error"):
                for doc in docs:
                    if doc.get("url") == top:
                        # Replace the snippet with the extracted passages for this one document.
                        doc["contents"] = text
                        doc["abstract"] = text
                        break
    except Exception:
        pass
    return [d for d in docs if isinstance(d, dict)]


def default_search_fn(*, llm: Optional[Any] = None, tool_strategy: str = "granular",
                      include_mcp_tools: bool = False, mcp_modules: Optional[List[str]] = None,
                      enabled_search_methods: Optional[List[str]] = None,
                      skill_roots: Optional[List[str]] = None) -> SearchFn:
    def fn(query: str, state: SupervisorState) -> List[Any]:
        from agent_runtime.executor_factory import (
            agent_config,
            build_search_agent_executor,
            child_thread_id,
            invoke_agent_with_payload_fallback,
        )
        from agent_runtime.runtime_utils import build_search_evidence_payload

        # Deterministic short-circuits for id-bearing queries — do NOT rely on the LLM picking
        # the right tool (the original failures were that nothing steered it to the by-id /
        # related tools, so it ran a generic search and fabricated/whiffed).
        #   * "related elements of <UUID>"  -> graph traversal + similarity (two buckets)
        #   * "explain/describe <UUID>"     -> by-id element fetch (graph, then backend API)
        # When the id is OMITTED in a follow-up ("what are the related elements", "explain it"),
        # recall the element under discussion from the conversation so the same path still fires.
        chat_history = state.get("chat_history")
        related_id = _detect_related_elements_request(query)
        if not related_id and _RELATED_FOLLOWUP_RE.match(query or ""):
            related_id = _recall_recent_element_id(chat_history)
        if related_id:
            emit_trace_event(
                "node_started",
                {"stage": "search", "message": f"Related-element lookup for {related_id}"},
                node="search",
            )
            return _related_elements_evidence(related_id)
        lookup_id = _detect_element_lookup_request(query)
        if not lookup_id and _EXPLAIN_FOLLOWUP_RE.match(query or ""):
            lookup_id = _recall_recent_element_id(chat_history)
        if lookup_id:
            emit_trace_event(
                "node_started",
                {"stage": "search", "message": f"Element lookup for {lookup_id}"},
                node="search",
            )
            return _element_lookup_evidence(lookup_id)
        # "most popular / most viewed / trending ..." -> the graph's click_count ranking, not a
        # semantic search whose topical hits would be misrepresented as popularity.
        if _detect_popularity_request(query):
            emit_trace_event(
                "node_started",
                {"stage": "search", "message": "Popularity ranking from the knowledge graph"},
                node="search",
            )
            pop_docs = _popularity_evidence(query)
            if pop_docs:
                return pop_docs
            # graph empty/unreachable -> fall through to the normal search agent

        executor = build_search_agent_executor(
            llm=llm, tool_strategy=tool_strategy, include_mcp_tools=include_mcp_tools,
            mcp_modules=mcp_modules, enabled_search_methods=enabled_search_methods,
            skill_roots=skill_roots,
        )
        resp = invoke_agent_with_payload_fallback(
            executor, query=_as_retrieval_request(query), chat_history=None,
            config=agent_config(child_thread_id(state.get("thread_id"), "sup_search")),
        )
        harvested = extract_documents_from_search_evidence(build_search_evidence_payload(query, resp, None))
        # Completeness sweep: union in direct keyword+semantic hits so one search turn always
        # carries multi-method coverage, even when the LLM peer called a single tool.
        return _merge_dedup(harvested, _direct_search_sweep(query, enabled_search_methods))

    return fn


# --- deterministic QGIS map workflow -------------------------------------------
# When the user explicitly asks for QGIS (or for a map drawn "on a basemap"/"map layer"), the
# LLM peer used to write matplotlib/geopandas code instead: no basemap, and a buffer computed in
# DEGREES (~21.5 km instead of 25 km, varying with latitude). Detect that request and run the
# real QGIS chain deterministically — metric buffer in a projected CRS, then a PyQGIS render
# over an OSM basemap — so neither the projection nor the basemap depends on tool-choice whim.
_QGIS_MAP_RE = re.compile(
    r"\bqgis\b|\bpyqgis\b|\bbase\s?map\b|\bmap\s+layer\b|on\s+top\s+of\s+(?:a\s+)?map",
    re.I,
)
_DISTANCE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(kilometers?|kilometres?|km|meters?|metres?|miles?|mi|m)\b", re.I
)
_VECTOR_EXTS = (".geojson", ".json", ".shp", ".gpkg", ".zip", ".gml", ".kml")


def _detect_qgis_map_request(query: str) -> Optional[Dict[str, Any]]:
    """Return ``{"distance_meters": float|None}`` iff *query* asks for a QGIS/basemap map."""
    if not _QGIS_MAP_RE.search(query or ""):
        return None
    distance_m: Optional[float] = None
    m = _DISTANCE_RE.search(query or "")
    if m:
        value, unit = float(m.group(1)), m.group(2).lower()
        if unit.startswith(("km", "kilomet")):
            distance_m = value * 1000.0
        elif unit in ("mi", "mile", "miles"):
            distance_m = value * 1609.344
        else:
            distance_m = value
    return {"distance_meters": distance_m}


def _first_vector_path(input_file_ids: Optional[List[str]]) -> Optional[str]:
    """On-disk path of the first uploaded vector dataset, or None."""
    from agent_runtime.file_store import get_file_record, resolve_file_id

    for fid in (input_file_ids or []):
        try:
            record = get_file_record(str(fid)) or {}
            name = str(record.get("filename") or "").lower()
            if name.endswith(_VECTOR_EXTS):
                return str(resolve_file_id(str(fid)))
        except Exception:
            continue
    return None


def _run_qgis_map_workflow(query: str, *, input_file_ids: Optional[List[str]],
                           thread_id: Optional[str],
                           distance_meters: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Run buffer(optional) -> render-with-basemap via the QGIS tools. None if not applicable."""
    from rag_pipeline.qgis_headless_tools import (
        pyqgis_available,
        pyqgis_render_map_tool,
        qgis_metric_buffer_tool,
        qgis_process_available,
    )

    layer_path = _first_vector_path(input_file_ids)
    if not layer_path:
        return None
    want_buffer = bool(distance_meters and distance_meters > 0)
    if want_buffer and not qgis_process_available():
        return None          # buffering needs the qgis_process CLI
    if not pyqgis_available():
        return None          # rendering needs the PyQGIS bindings

    from agent_runtime.executor_factory import child_thread_id

    session = child_thread_id(thread_id, "analysis_qgis") or "qgis"
    steps: List[Dict[str, Any]] = []
    layers: List[Dict[str, Any]] = []
    buffer_km: Optional[float] = None

    if want_buffer:
        try:
            buf = json.loads(qgis_metric_buffer_tool(
                input_layer=layer_path, distance_meters=float(distance_meters),
                output_filename="buffer.geojson", session_id=session))
        except Exception as exc:
            return {"summary": f"QGIS buffer failed: {exc}", "qgis_workflow": True}
        steps.append({"step": "qgis_metric_buffer", "result": buf})
        if not buf.get("ok") or not buf.get("output_path"):
            return {"summary": "QGIS metric buffer did not produce an output layer.",
                    "steps": steps, "qgis_workflow": True}
        buffer_km = float(distance_meters) / 1000.0
        layers.append({"path": buf["output_path"], "name": f"{buffer_km:g} km buffer",
                       "style": {"fill_color": "#5DCAA555", "stroke_color": "#0F6E56",
                                 "stroke_width": 0.6}})
    layers.append({"path": layer_path, "name": "input layer",
                   "style": {"fill_color": "#D85A30", "size": 3.0}})

    try:
        render = json.loads(pyqgis_render_map_tool(
            layers_json=json.dumps(layers), output_filename="qgis_map.png",
            width=1100, height=1200, basemap="osm", session_id=session))
    except Exception as exc:
        return {"summary": f"QGIS render failed: {exc}", "steps": steps, "qgis_workflow": True}
    steps.append({"step": "pyqgis_render_map", "result": render})

    basemap = str(render.get("basemap") or "")
    parts = []
    if buffer_km:
        crs = (steps[0]["result"] or {}).get("projected_crs")
        parts.append(f"Computed a true {buffer_km:g} km buffer with QGIS "
                     f"(native:buffer in the projected CRS {crs}, reprojected back to EPSG:4326)")
    parts.append(f"rendered the layers with headless PyQGIS over the {basemap or 'no'} basemap"
                 f" (map CRS {render.get('crs')})" if render.get("ok")
                 else "the PyQGIS render did not complete")
    return {"summary": ". ".join(parts) + ".", "steps": steps, "qgis_workflow": True,
            "basemap": basemap or None}


def default_analyze_fn(*, llm: Optional[Any] = None, include_mcp_tools: bool = True,
                       mcp_modules: Optional[List[str]] = None,
                       skill_roots: Optional[List[str]] = None,
                       code_exec: Optional[bool] = None,
                       input_file_ids: Optional[List[str]] = None) -> AnalyzeFn:
    """Run the GIS/data analysis workflow (QGIS + spatial-analysis MCP tools)."""

    def fn(query: str, evidence: List[Any], state: SupervisorState) -> Any:
        from agent_runtime.executor_factory import (
            agent_config,
            build_agent_executor,
            child_thread_id,
            invoke_agent_with_payload_fallback,
        )
        from agent_runtime.langchain_granular_tools import make_langchain_qgis_tools

        # Deterministic QGIS chain when the user asked for QGIS / a basemap map: guarantees a
        # metric buffer and a real basemap instead of a matplotlib fallback.
        qgis_req = _detect_qgis_map_request(query)
        if qgis_req:
            emit_trace_event(
                "node_started",
                {"stage": "analyze", "message": "Running QGIS map workflow"},
                node="analyze",
            )
            qgis_result = _run_qgis_map_workflow(
                query, input_file_ids=input_file_ids, thread_id=state.get("thread_id"),
                distance_meters=qgis_req.get("distance_meters"),
            )
            if qgis_result:
                emit_trace_event(
                    "node_completed",
                    {"stage": "analyze", "message": "QGIS map ready"},
                    node="analyze",
                )
                return qgis_result


        from agent_runtime.runtime_utils import extract_final_answer, extract_search_artifacts

        thread_id = state.get("thread_id")
        request_tool, requests = _make_request_tool()
        tools = list(make_langchain_qgis_tools(session_id=child_thread_id(thread_id, "analysis_qgis")))
        # Same KB set the code peer gets. This peer also holds `execute_code`, so without these
        # it can RUN analysis code while being unable to discover that the platform already has
        # a callable method for the step it is about to re-implement — the exact gap fixed for
        # the code peer in M2.7, one peer over.
        try:
            from agent_runtime.langchain_granular_tools import make_langchain_granular_tools
            tools.extend(t for t in make_langchain_granular_tools(
                enabled_search_methods=sorted(_CODE_PEER_KB_TOOLS))
                if getattr(t, "name", "") in _CODE_PEER_KB_TOOLS)
        except Exception:
            pass
        if include_mcp_tools:
            from agent_runtime.langchain_mcp_tools import make_langchain_mcp_tools

            tools.extend(make_langchain_mcp_tools(include_modules=mcp_modules or ["spatial_analysis_tools"]))
        # Geospatial KB tools: run extracted spatial functions + chain GIS ops by file_id
        # (the GIS runs as executed tool steps; GeoDataFrames pass as files, not in memory).
        try:
            from extractors.geo_handles import make_geo_analysis_tools
            tools.extend(make_geo_analysis_tools())
        except Exception:
            pass
        # Agent-side geocoding (the code sandbox has NO network): named places/institutions
        # -> coordinates, so maps of named locations never require asking the user.
        try:
            from agent_runtime.langchain_granular_tools import make_langchain_geocode_tools
            tools.extend(make_langchain_geocode_tools())
        except Exception:
            pass
        tools.append(request_tool)
        # When files are attached to the conversation, let the analysis peer inspect
        # them directly (read_text_file / inspect_file_for_analysis) instead of only
        # being able to touch them via execute_code.
        if input_file_ids:
            from agent_runtime.langchain_file_tools import make_langchain_file_tools

            tools.extend(make_langchain_file_tools())
            # Vector / shapefile tools (read + visualize + analyze uploaded TIGER files,
            # zip or extracted). Guarded so a missing geopandas never breaks the agent.
            try:
                from agent_runtime.langchain_geo_tools import make_langchain_geo_tools
                tools.extend(make_langchain_geo_tools(default_input_file_ids=input_file_ids))
            except Exception:
                pass
        from agent_runtime.code_execution import is_code_exec_enabled

        if code_exec if code_exec is not None else is_code_exec_enabled():
            from agent_runtime.langchain_exec_tools import make_code_execution_tools

            # session_id makes the sandbox workspace persist across calls WITHIN this turn,
            # so a multi-step analysis can build state. Keyed on the conversation thread so
            # two conversations never share a workspace.
            tools.extend(make_code_execution_tools(
                default_input_file_ids=input_file_ids,
                session_id=child_thread_id(state.get("thread_id"), "code_exec"),
            ))
        executor = build_agent_executor(
            llm=llm, preloaded_tools=tools, system_prompt_override=ANALYSIS_WORKFLOW_PROMPT,
            agent_name="analysis_agent", skill_roots=skill_roots,
        )
        q = query
        if evidence:
            q = f"{query}\n\nContext evidence:\n{_format_documents(evidence)}"
        # Cross-turn continuity comes from this peer's own checkpointed child
        # thread (and the supervisor's chat_history drives routing/synthesis), so
        # we do NOT re-feed chat_history here — that would replay prior turns twice
        # on re-runs. Mirrors the search peer.
        resp = invoke_agent_with_payload_fallback(
            executor, query=q, chat_history=None,
            config=agent_config(child_thread_id(thread_id, "analysis")),
        )
        artifacts = extract_search_artifacts(resp)
        result: Dict[str, Any] = {
            "summary": extract_final_answer(resp) or "",
            "tool_calls": artifacts.get("tool_calls") or [],
            "tool_results": artifacts.get("tool_results") or [],
        }
        caps = list(dict.fromkeys(r["capability"] for r in requests))
        if caps:
            result["needs"] = caps  # model-driven request(s)
        return result

    return fn


def default_code_fn(*, llm: Optional[Any] = None, skill_roots: Optional[List[str]] = None,
                    code_exec: Optional[bool] = None,
                    input_file_ids: Optional[List[str]] = None) -> CodeFn:
    """Code peer: writes code, and can request_capability(search/analyze) when it
    lacks the context to do so (model-driven — no nested search tool)."""

    def fn(query: str, evidence: List[Any], state: "SupervisorState") -> Any:
        # AGENT_CODE_PEER=opencode swaps the whole peer for a sandboxed opencode
        # run (it iterates internally — no request_capability / no nested tools).
        from agent_runtime.opencode_peer import is_opencode_peer_enabled, run_opencode_code_peer

        if is_opencode_peer_enabled():
            return run_opencode_code_peer(
                query, evidence=evidence, state=state, input_file_ids=input_file_ids,
            )
        from agent_runtime.executor_factory import (
            agent_config,
            build_agent_executor,
            child_thread_id,
            invoke_agent_with_payload_fallback,
        )
        from agent_runtime.runtime_utils import extract_final_answer, extract_search_artifacts
        from agent_runtime.skills import make_skill_tools

        request_tool, requests = _make_request_tool()
        tools = [*make_skill_tools(skill_roots=skill_roots), request_tool]
        # KB read tools so the code peer can pull the FULL source of referenced blocks
        # (get_kb_block) and reuse it verbatim instead of stubbing loaders, PLUS the method
        # library, which matters most exactly here: this is the peer that writes and runs the
        # code, and the library is mounted in its sandbox.
        #
        # This allowlist is hardcoded — it does NOT follow the request's
        # enabled_search_methods — so it is a fifth independent gate on the same names.
        # Observed with the method tools absent: the peer was told by its own prompt to call
        # kb_method_search, did not have it, and guessed the package name from the directory
        # instead — `from method_library import ...`, which fails. The package is
        # `iguide_methods`.
        try:
            from agent_runtime.langchain_granular_tools import make_langchain_granular_tools
            tools.extend(t for t in make_langchain_granular_tools(
                enabled_search_methods=sorted(_CODE_PEER_KB_TOOLS))
                if getattr(t, "name", "") in _CODE_PEER_KB_TOOLS)
        except Exception:
            pass
        # Geocoding runs agent-side (the sandbox has NO network): lets the peer turn named
        # places/institutions into coordinates and pass them into execute_code as data,
        # instead of asking the user for coordinates.
        try:
            from agent_runtime.langchain_granular_tools import make_langchain_geocode_tools
            tools.extend(make_langchain_geocode_tools())
        except Exception:
            pass
        # QGIS tools run in the AGENT environment (where QGIS is installed) — the code sandbox
        # image has no `qgis` package, so without these the peer could only attempt an
        # `import qgis` that always fails. Registered only when a backend is actually present.
        try:
            from agent_runtime.langchain_granular_tools import make_langchain_qgis_tools
            tools.extend(make_langchain_qgis_tools(
                session_id=child_thread_id(state.get("thread_id"), "code_qgis")))
        except Exception:
            pass
        # When files are attached, give the code peer the vector/shapefile tools too, so it
        # can inspect an uploaded TIGER shapefile's schema/CRS before writing code (and
        # plot/convert/reproject without round-tripping through the sandbox).
        if input_file_ids:
            try:
                from agent_runtime.langchain_geo_tools import make_langchain_geo_tools
                tools.extend(make_langchain_geo_tools(default_input_file_ids=input_file_ids))
            except Exception:
                pass
        from agent_runtime.code_execution import is_code_exec_enabled

        if code_exec if code_exec is not None else is_code_exec_enabled():
            from agent_runtime.langchain_exec_tools import make_code_execution_tools

            # session_id makes the sandbox workspace persist across calls WITHIN this turn,
            # so a multi-step analysis can build state. Keyed on the conversation thread so
            # two conversations never share a workspace.
            tools.extend(make_code_execution_tools(
                default_input_file_ids=input_file_ids,
                session_id=child_thread_id(state.get("thread_id"), "code_exec"),
            ))
        executor = build_agent_executor(
            llm=llm, preloaded_tools=tools, system_prompt_override=CODE_PEER_PROMPT,
            agent_name="code_agent", skill_roots=skill_roots,
        )
        parts = [query]
        if evidence:
            parts.append(f"Evidence:\n{_format_documents(evidence)}")
        if state.get("analysis_results"):
            parts.append(
                f"Analysis results:\n{json.dumps(state['analysis_results'], ensure_ascii=True, default=str)[:1500]}"
            )
        # See analyze peer: continuity is owned by this peer's checkpointed thread,
        # so chat_history is not re-fed here (avoids double-replay on re-runs).
        resp = invoke_agent_with_payload_fallback(
            executor, query="\n\n".join(parts), chat_history=None,
            config=agent_config(child_thread_id(state.get("thread_id"), "code")),
        )
        # Flat result: the human-readable answer + a compact artifacts extract.
        # Do NOT nest the whole raw response object (it would crowd out / truncate
        # the real code+output when synthesis serializes code_result).
        artifacts = extract_search_artifacts(resp)
        result: Dict[str, Any] = {
            "answer": extract_final_answer(resp) or "",
            "tool_calls": artifacts.get("tool_calls") or [],
            "tool_results": artifacts.get("tool_results") or [],
        }
        caps = list(dict.fromkeys(r["capability"] for r in requests))
        if caps:
            result["needs"] = caps  # model-driven request(s)
        return result

    return fn


def _compose_general_answer(llm: Optional[Any], query: str) -> str:
    """Answer a GENERAL question (no platform evidence needed) from the model's own knowledge.

    Used when nothing was retrieved and the question is not a request for I-GUIDE content, so the
    assistant is helpful instead of refusing. The prompt forbids inventing citations, element
    links, or claims about what the platform holds. Returns "" on any failure (caller falls back
    to the honest no-evidence reply). Never raises.
    """
    if not (query or "").strip():
        return ""
    from agent_runtime.supervisor.prompts import GENERAL_ANSWER_PROMPT

    try:
        active = llm
        if active is None:
            from agent_runtime.executor_factory import build_default_llm

            active = build_default_llm()
        prompt = GENERAL_ANSWER_PROMPT.format(question=query)
        if hasattr(active, "invoke"):
            text = _content_to_text(active.invoke(prompt))
        elif callable(active):
            text = str(active(prompt))
        else:
            return ""
        return (text or "").strip()
    except Exception:
        return ""


def _compose_insufficiency_reply(llm: Optional[Any], query: str) -> str:
    """LLM-compose a contextual, grounding-SAFE "no supporting evidence" reply.

    Reached only in the genuinely-cold case (nothing retrieved AND no conversation to draw on).
    ``INSUFFICIENT_EVIDENCE_PROMPT`` forbids answering the question or inventing facts — the
    model only acknowledges the gap and helps the user re-ask. Returns "" on any failure or an
    empty result so the caller can fall back to the deterministic ``NO_GROUNDING_FALLBACK``
    constant. Never raises — it must not break synthesize.
    """
    if not (query or "").strip():
        return ""
    from agent_runtime.supervisor.prompts import INSUFFICIENT_EVIDENCE_PROMPT

    try:
        active = llm
        if active is None:
            from agent_runtime.executor_factory import build_default_llm

            active = build_default_llm()
        prompt = INSUFFICIENT_EVIDENCE_PROMPT.format(question=query)
        if hasattr(active, "invoke"):
            text = _content_to_text(active.invoke(prompt))
        elif callable(active):
            text = str(active(prompt))
        else:
            return ""
        return (text or "").strip()
    except Exception:
        return ""


def default_synthesize_fn(llm: Optional[Any] = None) -> SynthesizeFn:
    """Compose the final grounded answer in the original AnalysisAgent format."""

    def fn(query: str, evidence: List[Any], analysis_results: Any, code_result: Any,
           chat_history: Optional[List[Any]] = None) -> str:
        from agent_runtime.supervisor.prompts import SYNTHESIS_PROMPT

        active = llm
        if active is None:
            from agent_runtime.executor_factory import build_default_llm

            active = build_default_llm()
        parts = [SYNTHESIS_PROMPT]
        history = _format_chat_history(chat_history)
        if history:
            parts.append(f"Conversation so far:\n{history}")
        parts.append(f"Question:\n{query}")
        parts.append(f"Evidence:\n{_format_documents(evidence)}")
        if analysis_results:
            parts.append(f"Analysis results:\n{json.dumps(analysis_results, ensure_ascii=True, default=str)[:2000]}")
        if code_result:
            # Prefer the code peer's human-readable answer; only fall back to a
            # serialized dump if no answer text is present (keeps the real code /
            # output from being truncated away by a large nested object).
            if isinstance(code_result, dict) and str(code_result.get("answer") or "").strip():
                parts.append(f"Code result:\n{str(code_result['answer'])[:2000]}")
            else:
                parts.append(f"Code result:\n{json.dumps(code_result, ensure_ascii=True, default=str)[:2000]}")
        prompt = "\n\n".join(parts)
        if hasattr(active, "invoke"):
            return _content_to_text(active.invoke(prompt))
        if callable(active):
            return str(active(prompt))
        raise TypeError("llm must expose .invoke() or be a str->str callable")

    return fn


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_supervisor_graph(
    *,
    decide_fn: Optional[DecideFn] = None,
    search_fn: Optional[SearchFn] = None,
    analyze_fn: Optional[AnalyzeFn] = None,
    code_fn: Optional[CodeFn] = None,
    synthesize_fn: Optional[SynthesizeFn] = None,
    llm: Optional[Any] = None,
    top_k: Optional[int] = None,
    do_rerank: bool = True,
    do_audit: bool = True,
    # Needed by the node itself, not just by the search peer: the last-resort web fallback must
    # honour a request that deliberately excluded web_search.
    enabled_search_methods: Optional[List[str]] = None,
) -> Any:
    """Compile the supervisor-over-peers graph. Workers default to existing agents."""
    top_k = top_k if top_k is not None else _default_top_k()
    decide = decide_fn or default_decide_fn(llm=llm)
    do_search = search_fn or default_search_fn(llm=llm)
    do_analyze = analyze_fn or default_analyze_fn(llm=llm)
    do_code = code_fn or default_code_fn(llm=llm)
    do_synthesize = synthesize_fn or default_synthesize_fn(llm=llm)

    def supervisor_node(state: SupervisorState) -> Dict[str, Any]:
        step = state.get("step", 0)
        actions = state.get("actions") or []
        needs = list(state.get("needs") or [])
        # Drop peer requests that can no longer be productive, so a peer that keeps
        # asking for the same dead capability can't loop us:
        #  - 'search' once it is exhausted (cap hit / last search returned nothing)
        #  - ANY capability already run the max number of times
        def _dead(n):
            cap = n.get("capability")
            # Unknown capability → drop just this request; never let it terminate the
            # whole run (routing an unknown cap to "done" would discard the queue).
            if cap not in _CAPABILITIES:
                return True
            # Peer-requested re-search dies only at the hard attempt cap — a single empty
            # result must NOT permanently close search (allows a narrowed re-try).
            if cap == "search" and state.get("search_attempts", 0) >= _max_searches():
                return True
            return actions.count(cap) >= _max_peer_runs()
        needs = [n for n in needs if not _dead(n)]

        if step >= state.get("max_steps", DEFAULT_MAX_STEPS):
            nxt, remaining, why = "done", [], "max_steps"
        elif needs:
            # Fulfill the oldest peer request first (FIFO), then continue the loop.
            req = needs[0]
            cap = req.get("capability")
            nxt = cap if cap in _CAPABILITIES else "done"
            remaining, why = needs[1:], f"request by {req.get('by')}"
        else:
            nxt = decide(state, _distill(state))
            if nxt not in ALLOWED_ACTIONS:
                nxt = "done"
            remaining, why = needs, "decision"
            # Backstop: a peer that just ran and already produced its result should
            # not be re-run back-to-back (it self-iterates internally). Prevents the
            # decider from looping on the same action until max_steps.
            if nxt != "done" and _is_unproductive_repeat(nxt, state):
                nxt, why = "done", f"no-progress repeat ({nxt})"
            # Don't keep hitting the search agent once the KB has nothing left to give.
            elif nxt == "search" and _search_exhausted(state):
                nxt, why = "done", "search exhausted"
        emit_trace_event(
            "node_completed",
            {"stage": "supervisor", "route": nxt, "message": f"supervisor → {nxt} ({why})"},
            node="supervisor",
        )
        return {
            "next_action": nxt,
            "actions": [*(state.get("actions") or []), nxt],
            "step": step + 1,
            "needs": remaining,
        }

    def search_node(state: SupervisorState) -> Dict[str, Any]:
        q = state.get("query", "")
        emit_trace_event("node_started", {"stage": "search", "message": "Searching"}, node="search")
        raw = do_search(q, state) or []
        if isinstance(raw, dict):
            docs = raw.get("documents") or []
            _, needs = _extract_needs(raw)
        else:
            docs, needs = raw, []

        # Retry with a REFORMULATED query when the first attempt returned nothing (or nothing on
        # topic). Re-running the identical query can only return the identical documents, so
        # without this the loop cannot recover from a bad phrasing.
        tried: List[str] = list(state.get("searched_queries") or [])
        if q and q not in tried:
            tried.append(q)
        if _refine_enabled():
            for _ in range(_max_refinements()):
                if not _results_are_poor(docs, q):
                    break
                refined = _refine_query(llm, q, docs, tried)
                if not refined:
                    break
                tried.append(refined)
                emit_trace_event(
                    "node_started",
                    {"stage": "search", "message": f"Retrying with a refined query: {refined}"},
                    node="search",
                )
                more = do_search(refined, state) or []
                if isinstance(more, dict):
                    more_docs = more.get("documents") or []
                    _, more_needs = _extract_needs(more)
                    needs = [*(needs or []), *(more_needs or [])]
                else:
                    more_docs = more
                if more_docs:
                    # Judge the merged set against the REFINED query too: a retry that finally
                    # found on-topic material must be able to end the loop.
                    docs = _merge_dedup(docs, more_docs)
                    if not _results_are_poor(more_docs, refined):
                        break
        # LAST RESORT: the platform found nothing, even after the refined retry. Consult the open
        # web rather than reporting no results while a public answer exists. Placed AFTER the
        # refinement loop on purpose — a bad phrasing should be retried against our own index
        # before going to a third party — and it cannot fire when the KB returned anything at all.
        if _platform_evidence_is_unhelpful(_merge_dedup(state.get("evidence") or [], docs), q):
            web_docs = _web_fallback_evidence(q, enabled_search_methods)
            if web_docs:
                emit_trace_event(
                    "node_started",
                    {"stage": "search",
                     "message": f"No I-GUIDE evidence found; searched the open web ({len(web_docs)} results)"},
                    node="search",
                )
                docs = _merge_dedup(docs, web_docs)

        # Skip rerank/top_k for a two-bucket related-element result: reranking would interleave
        # and truncate the curated vs content buckets. Their order/grouping is handled downstream.
        has_provenance = any(isinstance(d, dict) and d.get("provenance") in ("seed", "curated", "content") for d in docs)
        if do_rerank and len(docs) > 1 and not has_provenance:
            docs = rerank_documents(q, docs, top_k=top_k, llm=llm)  # operator bundled into search
        before = len(state.get("evidence") or [])
        merged = _merge_dedup(state.get("evidence") or [], docs)
        added = len(merged) - before
        emit_trace_event(
            "node_completed", {"stage": "search", "message": f"{len(merged)} docs in evidence"}, node="search"
        )
        # Track productivity so the supervisor stops searching when it adds nothing.
        prev_streak = state.get("search_empty_streak", 0)
        update: Dict[str, Any] = {
            "evidence": merged,
            "search_attempts": state.get("search_attempts", 0) + 1,
            "search_empty_streak": 0 if added > 0 else prev_streak + 1,
            "searched_queries": tried,
        }
        enq = _enqueue_needs(state.get("needs"), needs, "search")
        if enq is not None:
            update["needs"] = enq
        return update

    def analysis_node(state: SupervisorState) -> Dict[str, Any]:
        q = state.get("query", "")
        emit_trace_event("node_started", {"stage": "analyze", "message": "Running analysis workflow"}, node="analyze")
        clean, needs = _extract_needs(do_analyze(q, state.get("evidence") or [], state))
        emit_trace_event("node_completed", {"stage": "analyze", "message": "Analysis workflow complete"}, node="analyze")
        update: Dict[str, Any] = {"analysis_results": clean}
        enq = _enqueue_needs(state.get("needs"), needs, "analyze")
        if enq is not None:
            update["needs"] = enq
        return update

    def code_node(state: SupervisorState) -> Dict[str, Any]:
        q = state.get("query", "")
        emit_trace_event("node_started", {"stage": "code", "message": "Generating code"}, node="code")
        clean, needs = _extract_needs(do_code(q, state.get("evidence") or [], state))
        emit_trace_event("node_completed", {"stage": "code", "message": "Code ready"}, node="code")
        update: Dict[str, Any] = {"code_result": clean}
        enq = _enqueue_needs(state.get("needs"), needs, "code")
        if enq is not None:
            update["needs"] = enq
        return update

    def synthesize_node(state: SupervisorState) -> Dict[str, Any]:
        q = state.get("query", "")
        evidence = state.get("evidence") or []
        ar, cr = state.get("analysis_results"), state.get("code_result")
        # Scope artifacts to THIS turn: a plot/map shown in an earlier turn must not be
        # re-embedded here (the code peer's checkpointed thread replays it into code_result).
        artifacts = _drop_previously_shown(_collect_image_artifacts(ar, cr), state.get("chat_history"))
        emit_trace_event("node_started", {"stage": "synthesize", "message": "Composing answer"}, node="synthesize")
        has_grounding = _has_grounding(evidence, ar, cr, artifacts)
        has_history = bool(state.get("chat_history") or [])
        # A general question (definition, how-to, concept, chit-chat) does not need platform
        # evidence — answer it from general knowledge instead of refusing. Only a genuine
        # content/retrieval request gets the "no supporting evidence" reply.
        if not has_grounding and not has_history and not _needs_kb_evidence(q):
            emit_trace_event(
                "node_started",
                {"stage": "synthesize", "message": "Answering from general knowledge"},
                node="synthesize",
            )
            general = _compose_general_answer(llm, q)
            if general:
                merged_g = {**state, "answer": general, "audit": {}}
                emit_trace_event(
                    "node_completed",
                    {"stage": "synthesize", "message": "General answer composed"},
                    node="synthesize",
                )
                return {"answer": general, "final_answer": general, "audit": {},
                        "distilled": {**_distill(merged_g), "answer": general}}
        if not has_grounding and not has_history:
            # Nothing was retrieved or produced AND there's no conversation to draw on (e.g. a
            # cold first-turn query whose search backend is down or the KB has no match). Compose
            # an honest, query-specific "no supporting evidence" reply with the LLM — the prompt
            # forbids answering the question or inventing facts, so this acknowledges the gap
            # without fabricating. Fall back to a deterministic (env-overridable) constant if the
            # model is unavailable or returns nothing, so we never ship an empty answer.
            final = (_compose_insufficiency_reply(llm, q)
                     or os.getenv("AGENT_NO_GROUNDING_MESSAGE")
                     or NO_GROUNDING_FALLBACK)
            audit = {}
        else:
            # We have retrieval/execution grounding OR a conversation to work from. The latter
            # covers conversational/meta requests — "summarize our discussion", a recap, a
            # follow-up that refers back to earlier turns — which are answerable from
            # chat_history alone; the synthesizer (SYNTHESIS_PROMPT) still states insufficiency
            # rather than guessing if it lacks the facts for a substantive question.
            answer = do_synthesize(q, evidence, ar, cr, state.get("chat_history"))
            # Audit only when there's actual retrieval/execution grounding to check against.
            # A purely conversational answer (composed from chat_history with no evidence or
            # artifacts) has nothing for the grounding auditor to compare to and would be
            # false-flagged against empty evidence — skip the audit for it.
            # Artifacts + tool outputs are first-class grounding: pass the execution record so
            # a genuinely-produced map/file/count is not flagged as hallucination.
            audit = audit_answer_grounding(
                q, answer, evidence, llm=llm,
                execution_context={"analysis_results": ar, "code_result": cr, "artifacts": artifacts},
            ) if (do_audit and (answer or "").strip() and has_grounding) else {}
            # Deterministic reconciliation: produced artifacts + the execution record are
            # ground truth, so the LLM auditor can't false-flag a genuinely-generated
            # map/file or a number/method it actually computed.
            audit = _reconcile_audit_with_artifacts(
                audit, artifacts,
                execution_context={"analysis_results": ar, "code_result": cr, "artifacts": artifacts})
            # Act on the verdict: a flagged audit appends a user-visible caveat to the answer.
            final = _apply_grounding_caveat(answer, audit)
            # Embed produced image artifacts (maps/plots) inline so they render in markdown.
            final = _append_image_embeds(final, artifacts)
            # Defuse sandbox: pseudo-URLs, internal filesystem paths, and any link that merely
            # LOOKS like an agent file (fabricated host) but is not an artifact this run produced.
            from agent_runtime.runtime_utils import sanitize_answer_links

            refs = _collect_download_refs(ar, cr)
            # Evidence URLs are legitimate targets too (platform element pages, external
            # OpenGeoData landing pages), so citing them is never mistaken for a fabricated file.
            from agent_runtime.supervisor.evidence_subgraph import _element_url

            evidence_urls = [u for u in (_element_url(d) for d in evidence) if u]
            # Open-web URLs this turn actually surfaced. A file OFFER pointing at the web ("[Download
            # the CSV](https://…/x.csv)") is only kept when a search really returned that URL — the
            # model cannot invent a plausible one.
            from rag_pipeline.search.web_utils import allowed_urls as web_allowed_urls

            final = sanitize_answer_links(
                final,
                allowed_file_ids=refs["file_ids"],
                allowed_urls=[*refs["urls"], *evidence_urls, *web_allowed_urls()],
            )
            if not (final or "").strip():
                # Never ship an empty answer with a success status.
                final = ("I wasn't able to produce an answer for this request. Please try "
                         "rephrasing it or adding more detail.")
        # Surface the verdict as its own event so clients can show grounded/flagged.
        if audit:
            emit_trace_event(
                "grounding_audit",
                {
                    "stage": "grounding_audit",
                    "flagged": _audit_flagged(audit),
                    "hallucination_detected": bool(audit.get("hallucination_detected")),
                    "severity": audit.get("severity"),
                    "issues": audit.get("issues") or [],
                    "message": audit.get("summary") or "Grounding audit complete",
                },
                node="synthesize",
            )
        emit_trace_event(
            "node_completed",
            {"stage": "synthesize", "message": audit.get("summary") or "Answer composed"},
            node="synthesize",
        )
        merged = {**state, "answer": final, "audit": audit}
        return {"answer": final, "final_answer": final, "audit": audit, "distilled": {**_distill(merged), "answer": final}}

    builder = StateGraph(SupervisorState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("search", search_node)
    builder.add_node("analyze", analysis_node)
    builder.add_node("code", code_node)
    builder.add_node("synthesize", synthesize_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        lambda s: s.get("next_action", "done"),
        {"search": "search", "analyze": "analyze", "code": "code", "done": "synthesize"},
    )
    # Peers loop back to the supervisor (restores dynamic ordering / multi-hop).
    builder.add_edge("search", "supervisor")
    builder.add_edge("analyze", "supervisor")
    builder.add_edge("code", "supervisor")
    builder.add_edge("synthesize", END)
    return builder.compile()


def run_supervisor(
    query: str,
    *,
    chat_history: Optional[List[Any]] = None,
    llm: Optional[Any] = None,
    thread_id: Optional[str] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    **graph_kwargs: Any,
) -> Dict[str, Any]:
    """Build + run the supervisor graph; return the full final state."""
    graph = build_supervisor_graph(llm=llm, **graph_kwargs)
    return graph.invoke(
        {
            "query": query,
            "chat_history": chat_history or [],
            "thread_id": thread_id,
            "evidence": [],
            "needs": [],
            "actions": [],
            "step": 0,
            "max_steps": max_steps,
        }
    )


__all__ = [
    "SupervisorState",
    "build_supervisor_graph",
    "run_supervisor",
    "is_supervisor_enabled",
    "default_decide_fn",
    "default_search_fn",
    "default_analyze_fn",
    "default_code_fn",
    "default_synthesize_fn",
]
