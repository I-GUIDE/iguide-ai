"""Evidence-quality utilities for the agent path: LLM rerank + grounding audit.

These port the capabilities that previously lived ONLY in the deprecated
``rag_pipeline`` (``reranker_llm`` / ``hallucination_check``) into
``agent_runtime`` so the agent is a true *superset* of the RAG pipeline — with
two differences that fit the agent architecture:

* no dependency on ``rag_pipeline`` (operates on normalized shapes here);
* uses the *agent's own* LLM (``build_default_llm``), injectable for testing.

Both functions accept an optional ``llm`` that is either a chat model exposing
``.invoke(...)`` (returns a message with ``.content``) or a plain ``str -> str``
callable (handy in tests).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

LLMLike = Union[Any, Callable[[str], str]]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def _content_to_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
            else:
                parts.append(str(getattr(part, "text", part)))
        return "".join(parts)
    return str(content or "")


def _invoke_llm(llm: Optional[LLMLike], prompt: str) -> str:
    if llm is None:
        from agent_runtime.executor_factory import build_default_llm

        llm = build_default_llm()
    if hasattr(llm, "invoke"):
        return _content_to_text(llm.invoke(prompt))
    if callable(llm):
        return str(llm(prompt))
    raise TypeError("llm must expose .invoke() or be a str->str callable")


def _normalize_document(entry: Any, index: int) -> Dict[str, Any]:
    """Coerce a document into ``{doc_id, title, contents}`` regardless of source."""
    if isinstance(entry, str):
        return {"doc_id": f"doc-{index}", "title": "Untitled", "contents": entry}
    if not isinstance(entry, dict):
        return {"doc_id": f"doc-{index}", "title": "Untitled", "contents": str(entry)}
    # Support both the agent's flat docs and the rag_pipeline {"document": {...}} shape.
    document = entry.get("document") if isinstance(entry.get("document"), dict) else entry
    meta = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    doc_id = str(
        document.get("doc_id")
        or entry.get("doc_id")
        or meta.get("hit_id")
        or f"doc-{index}"
    )
    title = str(document.get("title") or document.get("element_type") or "Untitled").strip()
    contents = str(document.get("contents") or document.get("snippet") or document.get("text") or "").strip()
    return {"doc_id": doc_id, "title": title, "contents": contents, "_entry": entry}


# Reciprocal Rank Fusion constant. 60 is the value from the original TREC work and the de facto
# default; it damps the gap between adjacent top ranks so one source cannot dominate on position 1
# alone. Not sensitive enough to be worth tuning here.
_RRF_K = 60


def _doc_source(entry: Any) -> str:
    """Which retrieval method produced this doc, for per-source ranking."""
    if not isinstance(entry, dict):
        return "unknown"
    document = entry.get("document") if isinstance(entry.get("document"), dict) else entry
    return str(document.get("source") or document.get("source_system") or "unknown").strip().lower()


def _doc_score(entry: Any) -> float:
    """The doc's score AS REPORTED BY ITS OWN SOURCE.

    Only ever compared against scores from the SAME source: BM25 relevance (~4-9) and the catalog
    scorer's own scale (~0-1) are not commensurable, which is the whole reason this module fuses
    ranks rather than values.
    """
    if not isinstance(entry, dict):
        return 0.0
    document = entry.get("document") if isinstance(entry.get("document"), dict) else entry
    try:
        return float(document.get("score", entry.get("score", 0.0)) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def rrf_order(documents: List[Any], *, k: int = _RRF_K) -> List[Any]:
    """Order documents across sources by Reciprocal Rank Fusion.

    Each source is ranked internally by its own score — valid, because a score is comparable within
    the source that produced it — and the fused score is ``sum(1 / (k + rank))`` over the sources a
    document appears in. Only RANKS cross the source boundary, never score magnitudes, so a BM25
    8.58 and a catalog 1.0 no longer decide the order by scale.

    This is a fallback for when LLM rerank is unavailable, and it fuses on retrieval position only:
    it cannot judge topical fitness, so a hit that ranks high within its own source stays high here
    even if it is off-topic. That judgement is the reranker's job.
    """
    if not isinstance(documents, list) or len(documents) <= 1:
        return documents

    by_source: Dict[str, List[Tuple[int, Any]]] = {}
    for index, entry in enumerate(documents):
        by_source.setdefault(_doc_source(entry), []).append((index, entry))

    # index -> fused score. Keyed by position so unhashable/duplicate entries are still distinct.
    fused: Dict[int, float] = {index: 0.0 for index in range(len(documents))}
    for group in by_source.values():
        # Sort by the source's own score, keeping the original order as the tie-break so a source
        # that reports no scores at all retains the order it returned.
        ranked = sorted(group, key=lambda pair: (-_doc_score(pair[1]), pair[0]))
        for rank, (index, _entry) in enumerate(ranked, start=1):
            fused[index] += 1.0 / (k + rank)

    order = sorted(range(len(documents)), key=lambda i: (-fused[i], i))
    return [documents[i] for i in order]


def _extract_json_object(text: str) -> Optional[Any]:
    if not text:
        return None
    if "```" in text:
        start = text.find("```")
        end = text.find("```", start + 3)
        if end != -1:
            candidate = text[start + 3 : end].strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            try:
                return json.loads(candidate)
            except Exception:
                pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Rerank
# ---------------------------------------------------------------------------

def _rerank_prompt(query: str, docs: List[Dict[str, Any]]) -> str:
    items = "\n\n".join(
        f"{i+1}. doc_id={d['doc_id']}\ntitle={d['title']}\nexcerpt={(d['contents'] or 'No excerpt available.')[:400]}"
        for i, d in enumerate(docs)
    )
    return (
        "You are an expert relevance judge for a retrieval-augmented QA system.\n"
        "Re-rank the candidate documents by true semantic relevance to the query.\n"
        "Assign each a score in [0,1] with meaningful variance; reward documents that\n"
        "directly answer the query, penalize off-topic/redundant ones.\n"
        "Return JSON ONLY: {\"ranking\": [{\"doc_id\": \"<id>\", \"score\": <float>, "
        "\"reason\": \"<brief>\"}, ...]} including every doc_id exactly once.\n\n"
        f"User query:\n{query}\n\nCandidate documents:\n{items}\n"
    )


def rerank_documents(
    query: str,
    documents: List[Any],
    *,
    top_k: Optional[int] = None,
    llm: Optional[LLMLike] = None,
) -> List[Any]:
    """Reorder *documents* by LLM-judged relevance to *query*.

    Returns the original document objects, reordered (and truncated to ``top_k``).
    A failed/unparseable LLM call leaves the order unchanged. <=1 doc is a no-op.
    """
    if not isinstance(documents, list) or len(documents) <= 1 or not (query or "").strip():
        return documents

    normalized = [_normalize_document(entry, i) for i, entry in enumerate(documents)]
    by_id = {d["doc_id"]: d for d in normalized}

    failure = ""
    try:
        raw = _invoke_llm(llm, _rerank_prompt(query, normalized))
        parsed = _extract_json_object(raw) or {}
        ranking = parsed.get("ranking") if isinstance(parsed, dict) else None
        if not isinstance(ranking, list) or not ranking:
            failure = "the model returned no usable ranking"
    except Exception as exc:
        ranking = None
        failure = f"{type(exc).__name__}: {exc}"

    if failure:
        # Degrading SILENTLY here was the real defect: the caller got the merged sweep order, which
        # is arbitrary across sources, AND kept every document because top_k was only applied on
        # the success path. Fall back to a deterministic rank fusion, still truncate, and say so.
        logger.warning(
            "Evidence rerank unavailable (%s); falling back to reciprocal rank fusion over %d docs",
            failure, len(documents),
        )
        fused = rrf_order(documents)
        return fused[:top_k] if top_k is not None else fused

    ordered: List[Any] = []
    seen = set()
    for item in ranking:
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("doc_id") or "").strip()
        d = by_id.get(doc_id)
        if not d or doc_id in seen:
            continue
        seen.add(doc_id)
        ordered.append(d.get("_entry", d))
    # Append any documents the model omitted, preserving their original order.
    for d in normalized:
        if d["doc_id"] not in seen:
            ordered.append(d.get("_entry", d))

    if top_k is not None:
        ordered = ordered[:top_k]
    return ordered


# ---------------------------------------------------------------------------
# Grounding / hallucination audit
# ---------------------------------------------------------------------------

# Audited CLAIM BY CLAIM, with the ledger emitted BEFORE the verdict.
#
# The previous prompt did not discriminate at all: measured on a real run it returned "multiple
# high-severity hallucinations" for a correct, fully grounded answer AND the same verdict for that
# answer with blatant falsehoods appended (a fabricated journal, date, institution, benchmark score,
# price and adopter) — naming only the LEGITIMATE claims in both cases and never noticing a single
# fabrication. A caveat that says "high" either way carries no information.
#
# Two structural causes, both measured rather than guessed:
#  * VERDICT BEFORE PROOF. hallucination_detected and severity were the FIRST keys of the JSON, so
#    the model committed to a verdict autoregressively and then backfilled rationalisations. The
#    ledger now comes first and the verdict is read off it.
#  * NO OBLIGATION TO LOOK. Nothing forced the model to locate supporting text, so it asserted
#    "not directly supported" about claims stated almost verbatim in the evidence. Each row now
#    demands a VERBATIM span, and a row without one cannot be "supported".
#
# Prose rules alone did NOT work: a "paraphrase is not hallucination" paragraph and an explicit
# instruction not to flag when its own reason began "the evidence mentions X but..." were both
# ignored. Four formulations were measured against a fixed bar (a correct answer must clear, an
# answer with injected falsehoods must still be caught BY NAME) and independently re-verified; two
# of the four passed the bar but broke under further probing. Keep changes to this prompt measured.
_AUDIT_PROMPT = (
    'You are auditing an answer produced by a TOOL-USING agent for HALLUCINATION: claims\n'
    "grounded in NEITHER the retrieved evidence NOR the agent's execution record (the tools it\n"
    'actually ran and the artifacts it produced). The execution record is FIRST-CLASS grounding:\n'
    'a result a tool genuinely produced — a generated map/plot/image, a computed count, a written\n'
    'file_id — grounds an answer that presents it, even if no document mentions it.\n'
    '\n'
    'You must work CLAIM BY CLAIM, and in this order. Do not write a verdict before the ledger\n'
    'exists; a verdict that is not read off the ledger is a failed audit.\n'
    '\n'
    'STEP 1 - BUILD THE CLAIM LEDGER.\n'
    'Walk the answer from its first sentence to its LAST sentence and list every substantive,\n'
    'checkable claim: concrete facts, numbers, statistics, dates, names of people / institutions /\n'
    'journals / organizations / models, benchmark results, prices, capabilities, citations. Cover\n'
    'the WHOLE answer - the closing paragraph matters as much as the opening one, and a claim\n'
    'buried in the last sentences must still appear in the ledger. Skip pure framing, transitions,\n'
    'and offers of further help.\n'
    'DECOMPOSE SPECIFICS. Every number and every proper name in the answer gets its OWN row. Never\n'
    'fold a figure, benchmark, dataset, model, institution, venue or price into a sentence-level row:\n'
    'a sentence reading "improves performance by 41.2 points on GeoBench-Pro" yields a row for "41.2\n'
    'points" AND a row for "GeoBench-Pro", each needing its own verbatim span. Measured failure this\n'
    'prevents: an invented figure and benchmark embedded in an otherwise-supported sentence were\n'
    'summarised into one "supported" row and passed clean 3 times out of 3.\n'
    'Emit one row per claim:\n'
    '  {{"claim": "<short quote of the claim>",\n'
    '    "evidence_quote": "<[doc_id] + a VERBATIM span of 5-25 words copied from the evidence or\n'
    '      execution record that carries this claim>",\n'
    '    "status": "supported" | "contradicted" | "absent"}}\n'
    'How to fill a row:\n'
    '* You must actually COPY a real span. If you cannot copy one from the text you were given,\n'
    '  the status is not "supported".\n'
    '* "supported" - a real span carries the claim IN SUBSTANCE. Paraphrase, condensation,\n'
    '  rewording, and combining two sentences into one all count as supported; matching wording is\n'
    '  NOT required, and ONE supporting document is enough even when every other document is\n'
    '  irrelevant. If the span carries the point but not the exact phrasing, the row is\n'
    '  "supported".\n'
    '* "contradicted" - a real span asserts something incompatible with the claim: a different\n'
    '  number, date, name, venue or outcome. Copy that conflicting span.\n'
    '* "absent" - you searched the evidence AND the execution record and found NO span on this\n'
    '  subject at all: the number, name, venue, price or event simply does not occur anywhere.\n'
    '  Set evidence_quote to "none".\n'
    'ELIDED RECORDS. A record may carry a marker like "... [4213 chars elided] ...". That text\n'
    'was cut to fit, NOT withheld: what it held is unknown to you, so its absence is not\n'
    'evidence that a claim is unsupported. When a claim is on the SUBJECT of an elided section -\n'
    'the same tool, the same list, the same file - and you can find no span for it, mark it\n'
    '"absent" but say in the reason that the record was TRUNCATED rather than that the subject\n'
    'never occurs, and treat it as a minor over-reach rather than an invented specific.\n'
    '\n'
    'STEP 2 - VERDICT, DERIVED ONLY FROM THE LEDGER.\n'
    '* issues = exactly the rows whose status is "contradicted" or "absent", one issue each.\n'
    '  Never raise an issue for a row you marked "supported".\n'
    '* hallucination_detected = true if and only if issues is non-empty.\n'
    '* severity: "high" if any issue is an invented specific (a number, date, journal, institution,\n'
    '  price, organization, adopter or result found nowhere) or a contradiction; "low" if the only\n'
    '  issues are minor over-reach; "none" if there are no issues.\n'
    '* If every row is "supported": hallucination_detected=false, severity "none", issues [].\n'
    '\n'
    'Respond ONLY with JSON:\n'
    '{{"claim_ledger": [{{"claim": "...", "evidence_quote": "...", "status": "..."}}],\n'
    ' "hallucination_detected": true|false,\n'
    ' "severity": "none"|"low"|"medium"|"high",\n'
    ' "issues": [{{"claim": "...", "reason": "..."}}],\n'
    ' "summary": "one sentence verdict"}}\n'
    "Each issue's reason must state what a reader could check and find FALSE, quoting the\n"
    'conflicting span or saying that the evidence never mentions the subject at all.\n'
    '\n'
    'Question:\n'
    '{question}\n'
    '\n'
    'Answer:\n'
    '{answer}\n'
    '\n'
    'Retrieved evidence:\n'
    '{evidence}\n'
    '\n'
    'Execution record:\n'
    '{execution}\n'
    '\n'
)


# The auditor's evidence window MUST match the synthesizer's, which formats 8 documents at 2500
# characters each (_format_documents in agent_runtime/supervisor/evidence_subgraph.py, which imports
# from this module — so the constants live here to keep that direction of dependency).
#
# They did not match: the audit showed 5 documents at 600 characters, so the auditor was handed
# strictly LESS evidence than the writer and then asked whether the writer invented things. Measured
# on a real run, the one document supporting the answer held 5923 characters of a fetched paper and
# the auditor saw the first 600 — every claim drawn from the rest of it looked unsupported.
AUDIT_DOC_LIMIT = 8
AUDIT_DOC_CHARS = 2500


def _audit_window() -> tuple:
    """(doc limit, chars per doc) for the audit, env-tunable for a deployment that needs to trim."""
    def _int(name: str, default: int) -> int:
        try:
            return max(1, int(str(os.getenv(name, "")).strip() or default))
        except (TypeError, ValueError):
            return default

    return _int("AGENT_AUDIT_DOC_LIMIT", AUDIT_DOC_LIMIT), _int("AGENT_AUDIT_DOC_CHARS", AUDIT_DOC_CHARS)


def _format_evidence(evidence: Any, *, limit: int = AUDIT_DOC_LIMIT,
                     max_chars: int = AUDIT_DOC_CHARS) -> str:
    if isinstance(evidence, str):
        return evidence[: max_chars * limit].strip() or "(no evidence supplied)"
    if not isinstance(evidence, list) or not evidence:
        return "(no evidence supplied)"
    lines: List[str] = []
    for i, entry in enumerate(evidence[:limit]):
        d = _normalize_document(entry, i)
        lines.append(f"[{d['doc_id']}] {d['title']}\n{d['contents'][:max_chars]}")
    return "\n\n".join(lines) if lines else "(no evidence supplied)"


# The ledger arrives ALREADY budgeted by its producer (_LEDGER_MAX_CHARS in
# supervisor/graph.py). Re-truncating it here at the generic 2200 is what made the answerer
# and the auditor disagree: the answerer was told to answer from a line the auditor could not
# see, and the auditor then flagged the result as high-severity hallucination.
_PRIOR_ACTIONS_MAX_CHARS = 8000


# This turn's execution record was capped at 2,200 chars while prior_actions gets 8,000 — so
# the auditor saw LESS of the run it was auditing than of earlier turns. Worse, the cap was a
# blind prefix cut over a JSON dump that is overwhelmingly coordinate data.
#
# Measured on a reconstruction of the observed failure (26 OSM county boundaries, the payload
# behind "Which counties border Champaign County?"): the record was 87,648 chars, the auditor
# saw 2,218, and exactly ONE of the 26 county names survived. Coordinate arrays were 94.2% of
# those chars. The auditor then reported, honestly, that "the county-bordering claims are not
# grounded in the supplied evidence or execution record" — it could not see them.
#
# So the fix is to spend the budget on what an ANSWER can quote. Nothing ever cites a coordinate
# pair or an embedding component; claims cite names, counts, statistics and stdout. Eliding bulk
# numeric arrays turns that 87,648 into a few thousand chars with every name intact.
_EXEC_RECORD_MAX_CHARS = 8000
# A coordinate ring or an embedding vector, versus a short numeric field an answer might quote
# (a bbox is 4 numbers, an rgb triple 3, a small distance list a handful).
_MAX_INLINE_NUMBERS = 32
_BULK_NUMERIC_KEYS = {"coordinates", "embedding", "embeddings", "vector", "vectors", "values"}


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _numeric_count(value: Any, _depth: int = 0) -> int:
    """How many numbers a structure holds — the test for whether it is BULK."""
    if _depth > 12:
        return 0
    if isinstance(value, (list, tuple)):
        return sum(_numeric_count(v, _depth + 1) for v in value)
    return 1 if _is_number(value) else 0


def _summarize_numeric(value: Any) -> str:
    """Replace a bulk numeric structure with its shape, which is all an audit needs from it."""
    outer = len(value) if isinstance(value, (list, tuple)) else 0
    depth, probe = 0, value
    while isinstance(probe, (list, tuple)):
        depth += 1
        probe = probe[0] if probe else None

    return (f"<{_numeric_count(value)} numbers, nesting depth {depth}, "
            f"outer length {outer} - elided>")


def _compact_for_audit(obj: Any, _depth: int = 0) -> Any:
    """Strip bulk numeric payloads from an execution record, keeping every quotable fact.

    Also parses JSON-string tool results, because tool `content` arrives as a STRING: without
    parsing, the walk cannot reach the coordinates inside it and the elision does nothing.
    """
    if _depth > 12:
        return obj
    if isinstance(obj, str):
        # Only worth parsing when it is big enough to matter and actually looks like JSON.
        stripped = obj.strip()
        if len(obj) > 512 and stripped[:1] in "{[":
            try:
                return _compact_for_audit(json.loads(stripped), _depth + 1)
            except Exception:      # noqa: BLE001 - not JSON after all; keep the text
                return obj
        return obj
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for key, val in obj.items():
            if (str(key).lower() in _BULK_NUMERIC_KEYS
                    and isinstance(val, (list, tuple))
                    and _numeric_count(val) > _MAX_INLINE_NUMBERS):
                # Size-gated on purpose: a Point geometry is 2 numbers and a bbox is 4, and
                # answers DO quote those ("the bounding box is -88.28, 40.06, -88.16, 40.16").
                # Only a ring or a vector is bulk.
                out[key] = _summarize_numeric(val)
            else:
                out[key] = _compact_for_audit(val, _depth + 1)
        return out
    if isinstance(obj, (list, tuple)):
        if len(obj) > _MAX_INLINE_NUMBERS and all(_is_number(v) for v in obj):
            return _summarize_numeric(obj)
        return [_compact_for_audit(v, _depth + 1) for v in obj]
    return obj


def _elided(text: str, max_chars: int) -> str:
    """*text* capped at *max_chars*, and SAYING SO when it was cut.

    A blind slice makes a cut span indistinguishable from a span that was never there — and the
    auditor's rule for "absent" is "you searched and found nothing", so a claim supported at
    char 9000 of an 8000-char cap comes back as an invented specific at high severity, over an
    answer that was right. `_render_execution_record` already marked its cut; prior_actions and
    the environment lines did not, which is the half that produced a false caveat on a correct
    cross-turn answer. The marker is worded the same in both, so the prompt explains it once.
    """
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n... [{len(text) - max_chars} chars elided] ..."


def _render_execution_record(key: str, value: Any, max_chars: int) -> str:
    """One execution-record section, compacted, and cut at BOTH ends rather than just the head.

    A head-only cut is what lost 25 of 26 names: the informative tail of a feature list never
    arrived. Keeping a tail costs nothing and preserves the end of whatever overflows.
    """
    try:
        rendered = json.dumps(_compact_for_audit(value), default=str)
    except Exception:      # noqa: BLE001 - never let rendering break the audit
        try:
            rendered = json.dumps(value, default=str)
        except Exception:  # noqa: BLE001
            rendered = str(value)
    if len(rendered) <= max_chars:
        return f"{key}: {rendered}"
    head = int(max_chars * 0.7)
    tail = max_chars - head
    if tail <= 0:
        # rendered[-0:] is the WHOLE string, not the empty one, so a zero tail would emit the
        # entire record and silently defeat the cap. Degrade to a head-only cut instead.
        return f"{key}: {rendered[:max_chars]}\n... [{len(rendered) - max_chars} chars elided]"
    return (f"{key}: {rendered[:head]}\n... [{len(rendered) - max_chars} chars elided] ...\n"
            f"{rendered[-tail:]}")


def _format_execution_context(execution_context: Any, *, max_chars: int = 2200) -> str:
    """Render the agent's execution outcomes (tool calls/results, produced artifacts) so
    the auditor can treat genuinely-produced outputs as grounding."""
    if not execution_context:
        return "(no tools were executed)"
    if isinstance(execution_context, str):
        return _elided(execution_context, max_chars)
    parts: List[str] = []
    if isinstance(execution_context, dict):
        # FIRST, because it is the turn under audit and everything after it may be elided.
        #
        # Earlier turns reach the auditor as purpose-built lines — `tool(args) -> facts`, and
        # `FAILED tool(args) -> DID NOT RUN: <error>`. The CURRENT turn reached it only as a
        # JSON dump of the peers' results, so the turn whose answer was on trial was the one
        # described worst: a tool's ARGUMENTS were quotable for every turn except this one, and
        # a failed call looked like any other entry. An answer saying "embedded for 2022" is
        # grounded by `embed_zones(year=2022)`, which is an argument, not a result.
        this_turn = execution_context.get("this_turn")
        if this_turn:
            rendered_turn = ("\n".join(str(p) for p in this_turn)
                             if isinstance(this_turn, (list, tuple)) else str(this_turn))
            parts.append("tool calls and results from THIS TURN, the one being audited (first-"
                         "class grounding, exactly like the evidence — an ARGUMENT grounds a "
                         "claim about what was asked for, and a line beginning FAILED means "
                         "that tool produced NOTHING, so a claim resting on it is "
                         "contradicted, not merely absent):\n"
                         + _elided(rendered_turn, max(max_chars, _EXEC_RECORD_MAX_CHARS)))
        # The execution record gets the SAME budget as prior_actions: the auditor should not
        # see less of the turn it is auditing than of earlier ones.
        record_cap = max(max_chars, _EXEC_RECORD_MAX_CHARS)
        for key in ("analysis_results", "code_result"):
            val = execution_context.get(key)
            if val:
                parts.append(_render_execution_record(key, val, record_cap))
        arts = execution_context.get("artifacts")
        if arts:
            parts.append(f"produced artifacts (files/images generated by tools): "
                         f"{json.dumps(arts, default=str)[:max_chars]}")
        # Facts about the environment the answer was produced into — currently what the user's
        # map client lets them do with a delivered layer. Kept OUT of prior_actions because that
        # section is headed "from EARLIER TURNS", and this is true of the run now.
        #
        # This section exists because of a structural gap, not a model failure: STEP 1 above
        # requires a verbatim span for any row marked "supported", and an affordance
        # (pan/zoom/toggle/click) is a property of the VIEWER that no tool result can ever
        # report. Asked whether "you can pan, zoom, and click the hospital markers" was
        # supported, the auditor could only answer "absent" — and did, at high severity, over an
        # otherwise fully-verified answer whose 49-feature layer had really been delivered.
        env = execution_context.get("environment")
        if env:
            rendered_env = ("\n".join(str(p) for p in env) if isinstance(env, (list, tuple))
                            else str(env))
            parts.append("facts about the environment this answer was produced in (first-class "
                         "grounding: quote these spans when the answer describes them):\n"
                         + _elided(rendered_env, max_chars))
        prior = execution_context.get("prior_actions")
        if prior:
            rendered = ("\n".join(str(p) for p in prior) if isinstance(prior, (list, tuple))
                        else str(prior))
            parts.append("tool calls and results from EARLIER TURNS of this same conversation "
                         "(the agent legitimately answers follow-up questions from these — treat "
                         "them as grounding exactly like this turn's tool output):\n"
                         + _elided(rendered, max(max_chars, _PRIOR_ACTIONS_MAX_CHARS)))
    else:
        parts.append(_elided(json.dumps(execution_context, default=str), max_chars))
    return "\n".join(parts) if parts else "(no tools were executed)"


def _default_audit_verdict(summary: str, severity: str = "none") -> Dict[str, Any]:
    return {
        "hallucination_detected": False,
        "severity": severity,
        "issues": [],
        "summary": summary,
    }


def audit_answer_grounding(
    question: str,
    answer: str,
    evidence: Any,
    *,
    llm: Optional[LLMLike] = None,
    execution_context: Any = None,
    evidence_limit: Optional[int] = None,
    snippet_chars: Optional[int] = None,
) -> Dict[str, Any]:
    """Audit whether *answer* is grounded in *evidence* and/or the agent's
    *execution_context* (tool outputs + produced artifacts); returns a verdict dict.

    *evidence* may be a string, a list of docs, or a list of text snippets.
    *execution_context* (optional) carries what the agent's tools actually produced, so
    answers that present genuinely-generated artifacts are not falsely flagged.
    Missing inputs return a benign "insufficient data" verdict (never raises).
    """
    if not (question or "").strip() or not (answer or "").strip() or (not evidence and not execution_context):
        return _default_audit_verdict("Insufficient data to evaluate hallucinations.")

    window_limit, window_chars = _audit_window()
    prompt = _AUDIT_PROMPT.format(
        question=question,
        answer=answer,
        evidence=_format_evidence(evidence,
                                  limit=evidence_limit if evidence_limit is not None else window_limit,
                                  max_chars=snippet_chars if snippet_chars is not None else window_chars),
        execution=_format_execution_context(execution_context),
    )
    try:
        parsed = _extract_json_object(_invoke_llm(llm, prompt))
    except Exception:
        return _default_audit_verdict("Audit LLM call failed.", severity="unknown")
    if not isinstance(parsed, dict):
        return _default_audit_verdict("LLM response could not be parsed.", severity="unknown")
    parsed.setdefault("hallucination_detected", False)
    parsed.setdefault("severity", "none")
    parsed.setdefault("issues", [])
    parsed.setdefault("summary", "")
    return parsed


__all__ = ["rerank_documents", "audit_answer_grounding"]
