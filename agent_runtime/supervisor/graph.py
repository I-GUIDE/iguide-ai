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
live validation). It is the only orchestration path.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, TypedDict, Tuple

from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)

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
# (query, evidence, analysis_results, code_result, chat_history, prior_actions_note) -> answer
# The 6th argument is POSITIONAL-with-default on purpose: custom synthesize_fn doubles are
# overwhelmingly `lambda *a: ...`, which accepts an extra positional but not a keyword.
SynthesizeFn = Callable[[str, List[Any], Any, Any, Optional[List[Any]], Optional[str]], str]

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
    evidence_summary: Optional[str]
    max_steps: int
    final_answer: str
    distilled: Dict[str, Any]
    search_attempts: int           # how many times the search peer has run
    search_empty_streak: int       # consecutive searches that added NO new evidence
    searched_queries: List[str]    # every query string actually searched (incl. refinements)
    action_rows: List[Dict[str, Any]]  # ledger rows the peers produced THIS turn (see
                                       # _record_actions); cleared by synthesize so the
                                       # client payload never carries them
    unified_peer: Optional[bool]   # per-request override of AGENT_UNIFIED_PEER
    grounding_gaps: List[str]      # claims the audit could not ground, for a re-grounding pass
    grounding_retries: int         # how many re-grounding passes this turn has spent (cap 1)
    reground: bool                 # synthesize -> supervisor instead of END, for that one pass




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


# ---------------------------------------------------------------------------
# What the evidence SAYS, not just how much of it there is
# ---------------------------------------------------------------------------
# The decider sees titles, per-source counts, a top score and `topical_coverage`. Those are
# cheap, deterministic and cannot be talked into anything — but they are all LEXICAL. They can
# say "8 documents, 0.75 of them mention your subject terms" and still leave the decider unable
# to tell a set of PySAL accessibility notebooks from a set of DEM sources, because both mention
# "elevation". Measured live on a self-hosted model: two full search rounds where the second
# added nothing the first had not, because "is this enough?" was being answered from counts.
#
# So the model that just read the documents writes a few lines about what is actually in them.
#
# Two rules keep this from making things worse:
#
#  * It DESCRIBES, and names gaps. It does not rule on sufficiency. That judgement belongs to
#    the decider, which also sees the deterministic signals and the action history; a summary
#    that announced "this is enough" would collapse two independent checks into one and give a
#    weak model's opinion the final word.
#  * It never REPLACES the lexical signals — it sits beside them. If the summary is wrong, the
#    decider still has coverage, scores and titles to disagree with.
#
# Failure is always None: a summary is an aid to a decision, never a precondition for one.

_EVIDENCE_SUMMARY_MAX_CHARS = 700
_EVIDENCE_SUMMARY_DOCS = 8
_EVIDENCE_SUMMARY_SNIPPET = 400


def _evidence_summary_enabled() -> bool:
    return str(os.getenv("AGENT_EVIDENCE_SUMMARY", "1")).strip().lower() not in {
        "0", "false", "no", "off"}


_EVIDENCE_SUMMARY_PROMPT = (
    "You are helping an agent decide what to do next. Below are search results retrieved for a "
    "user's request.\n\n"
    "Write at most four sentences covering:\n"
    "1. What these results actually contain — the kinds of things, not their titles restated.\n"
    "2. Which parts of the request they DO address.\n"
    "3. Which parts they do NOT address, or say 'they address the request directly' if nothing "
    "is missing.\n\n"
    "Describe only. Do NOT recommend an action, do not say whether to search again, and do not "
    "say whether the evidence is sufficient — another step decides that and needs your "
    "description, not your verdict. If the results are off-topic, say so plainly.\n\n"
    "REQUEST: {query}\n\nRESULTS:\n{docs}"
)


def _summarize_evidence(llm: Any, query: str, docs: List[Any]) -> Optional[str]:
    """A few lines on what the retrieved evidence contains. None if unavailable."""
    if not llm or not docs or not _evidence_summary_enabled():
        return None
    lines: List[str] = []
    for i, d in enumerate(docs[:_EVIDENCE_SUMMARY_DOCS], 1):
        title = _doc_field(d, "title", "name", default="Untitled")
        body = _doc_field(d, "contents", "content", "text", "abstract", "description", default="")
        lines.append(f"[{i}] {title}\n{str(body)[:_EVIDENCE_SUMMARY_SNIPPET]}")
    prompt = _EVIDENCE_SUMMARY_PROMPT.format(query=query, docs="\n\n".join(lines))
    try:
        resp = llm.invoke(prompt)
    except Exception as exc:  # noqa: BLE001 - an aid to a decision, never a precondition
        logger.warning("evidence summary failed, continuing without it: %s", exc)
        return None
    text = getattr(resp, "content", resp)
    text = str(text or "").strip()
    if not text:
        return None
    return text[:_EVIDENCE_SUMMARY_MAX_CHARS]


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


_LEDGER_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# What this conversation has already done
# ---------------------------------------------------------------------------
# The decision payload below is built from per-turn state, so on turn 2 it reports
# has_evidence=False / has_analysis=False / artifacts_produced=[] no matter what turn 1
# produced — and routing to `search` is then the only sensible read of it. That is how "do you
# use the original resolution of clay or do you downsample it" became forty-nine keyword
# searches and a blown context window, when `pixel_ground_m` was already sitting in the
# previous turn's tool result.
#
# Curated, not dumped. These are the argument and result fields that say WHAT was done and
# would answer a follow-up about it; everything else is noise in a routing decision.
_LEDGER_ARGS = (
    "model", "area", "state", "level", "subdivide", "place", "feature", "query",
    "lon", "lat", "bbox", "start", "end", "year", "file_id", "zone_id_field", "zone_ids",
    # embed_region takes a LIST of models; "model"/"year" above miss it entirely.
    "models", "years", "buffer_m", "max_tiles", "clusters",
    # WHICH variable, WHICH neighbours, WHICH estimator. Without these,
    # local_moran_lisa(column="income", weights="queen") recorded as
    # `local_moran_lisa (file_id=f2)` and a follow-up asking what was tested was unanswerable.
    "column", "columns", "y_column", "x_columns", "value_column", "by", "time_column",
    "statistic", "weights", "method", "n_regions", "freq", "render",
    # buffer_layer takes distance + units (there is no `distance_m`), so a 2 km buffer
    # recorded as `buffer_layer (file_id=f1)`.
    "distance", "units", "cell_km",
    # execute_code produced no row at all. `code` stays OUT on purpose: 80 chars of a program
    # is noise, and label/entrypoint already name it.
    "language", "label", "entrypoint", "dependencies",
    "limit", "element_id", "url", "doc_id",
    # DELIBERATELY EXCLUDED: permutations (999), tile_px (200), timeout_seconds (120/300).
    # _reconcile_audit_with_artifacts drops any audit issue whose 3+ digit numbers all appear
    # in the json-dumped execution record, so curating an internal knob converts it into
    # blanket amnesty for a fabricated 3-digit figure. Rule: curate a 3+ digit argument only
    # when a user would plausibly quote it back. distance/buffer_m/years pass; a permutation
    # count does not.
)
_LEDGER_FACTS = (
    "model", "dims", "dim", "scale_m", "pixel_ground_m", "scale_m_mercator", "year",
    "geoid", "feature_count", "count", "zone_id_field", "level", "tiles_fetched",
    # An on-the-fly model reports its geometry differently and has NO scale_m at all:
    # clay's 36-key meta carries image_size / input_size_hw / patch_size / grid_hw_tokens
    # instead. Without these, "original resolution or downsampled?" is unanswerable for
    # every model except the precomputed ones — which is exactly how it read.
    "image_size", "input_size_hw", "patch_size", "grid_hw_tokens", "source", "sensor",
    # Spatial statistics were lost entirely. `verdict` is the tool's own plain sentence and
    # carries the statistic with its expectation, which is what an answer needs.
    "verdict", "features_analyzed", "column", "crs", "filename",
    # Coverage. embed_zones is uncapped by default, but a caller-set max_tiles can still cut
    # a sweep short, and `truncated` is the tool SAYING so — losing it is how "the whole county
    # was embedded" gets asserted over a fraction of the tiles.
    "zones_total", "zones_with_pixels", "tiles_planned", "truncated", "row_count", "cells",
    # Whether a layer was a SUBSET. The note described a sampled layer exactly as it described a
    # complete one, and "show the rest" is a normal follow-up the peer had no way to know was
    # needed. NOT the payload's `sampled` bool: _pick keeps False, so it would stamp ": False"
    # onto every complete layer. The full count says it by comparison with feature_count, the
    # count actually mapped.
    "features_total",
    # search, after the rename at capture
    "search_method", "results_returned",
)
_LEDGER_SEARCH_TOOLS = frozenset({
    "keyword_search", "semantic_search", "neo4j_search", "neo4j_get_element_by_id",
    "neo4j_explore_related_nodes", "spatial_search", "opengeodata_search", "agent_kb_search",
    "get_kb_block", "overpass_search", "web_search", "web_fetch", "baseline_sweep",
    "web_fallback", "related_elements", "element_lookup", "popularity_ranking",
})
_LEDGER_SEARCH_RENAMES = {"source": "search_method", "count": "results_returned"}
_LEDGER_VALUE_CHARS = 80
_LEDGER_ROWS_SHOWN = 25
# A hard ceiling on the rendered ledger, independent of the row count. 25 rows of long args
# could reach several thousand tokens, and this thing exists BECAUSE a turn overflowed the
# context window — it must not be able to cause that itself. Oldest rows drop first.
_LEDGER_MAX_CHARS = int(os.getenv("AGENT_LEDGER_MAX_CHARS") or "6000")
# A single tool must not crowd out the others. Retrieval is the high-cardinality producer (12
# tools x several queries x several turns); without this cap it evicts the analysis rows that
# answer the follow-ups this feature exists for. Newest kept.
_LEDGER_ROWS_PER_TOOL = 3


def _json_size(row: Dict[str, Any]) -> int:
    """A row's cost to a consumer that receives the raw rows."""
    return len(json.dumps(row, default=str))


def _budgeted(rows: List[Dict[str, Any]],
              *, size: Optional[Callable[[Dict[str, Any]], int]] = None) -> List[Dict[str, Any]]:
    """The newest rows that fit the char budget, oldest dropped first.

    ``size`` measures a row in THE UNIT THE CONSUMER ACTUALLY PAYS, because the two consumers
    do not agree. The router receives the raw rows and pays for their JSON; the answering model
    and the auditor receive _ledger_lines, where every fact goes through the phrase book and a
    single one can expand 3.7x ("features_total": 801 costs 23 JSON chars and renders 80).

    Measuring JSON for both let the RENDERED ledger run far past _LEDGER_MAX_CHARS, whose own
    comment calls it "a hard ceiling on the rendered ledger": 25 rows carrying only ordinary
    facts passed at 5,665 JSON chars and rendered 14,764 — a 2.6x breach of the ceiling, by the
    one mechanism that exists BECAUSE a turn overflowed the context window.
    """
    measure = size or _json_size
    per_tool: Dict[str, int] = {}
    thinned: List[Dict[str, Any]] = []
    for row in reversed(rows or []):
        tool = str(row.get("tool"))
        if per_tool.get(tool, 0) >= _LEDGER_ROWS_PER_TOOL:
            continue
        per_tool[tool] = per_tool.get(tool, 0) + 1
        thinned.append(row)
    thinned.reverse()
    rows = thinned

    kept: List[Dict[str, Any]] = []
    total = 0
    for row in reversed(rows[-_LEDGER_ROWS_SHOWN:]):
        row_size = measure(row)
        if kept and total + row_size > _LEDGER_MAX_CHARS:
            break
        kept.append(row)
        total += row_size
    kept.reverse()
    return kept


# `verdict` and `error` ARE the answer to "what did it find" / "why did it fail"; the default
# 80 chars cuts both mid-sentence. Everything else stays at the default.
_LEDGER_VALUE_CHARS_BY_KEY = {"verdict": 200, "error": 160}
# How many layer labels one ledger row may carry. A tool that maps forty zones separately
# must not push everything else out of the ledger; the row says how many it dropped.
_LEDGER_LAYERS_PER_ROW = 6


def _layer_labels(value: Any) -> List[str]:
    """The layer labels on a ledger row, however the row spells them.

    Rows carry a LIST since a single tool call can map several layers at once, but a bare
    string is still accepted: rows are process-local and outlive no restart, yet a shape guard
    here is cheaper than a crash in the one path that tells the model what is on the screen.
    """
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v not in (None, "")]
    return [str(value)]


def _layer_phrase(value: Any) -> str:
    """One layer reads exactly as it always did; several read as a list."""
    return ", ".join(repr(label) for label in _layer_labels(value))


def _ledger_value(value: Any, key: str = "") -> Any:
    """A value small enough to sit in a routing payload."""
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    text = str(value)
    limit = _LEDGER_VALUE_CHARS_BY_KEY.get(key, _LEDGER_VALUE_CHARS)
    return text if len(text) <= limit else text[:limit] + "…"


def _pick(source: Any, keys) -> Dict[str, Any]:
    if not isinstance(source, dict):
        return {}
    out = {}
    for k in keys:
        if k in source and source[k] not in (None, "", [], {}):
            out[k] = _ledger_value(source[k], k)
    return out


def _row_has_result(row: Dict[str, Any]) -> bool:
    return any(k in row for k in ("facts", "file_id", "map_layer", "failed"))


def _search_row(tool: str, query: str, method: str, returned: int,
                **extra: Any) -> Dict[str, Any]:
    """A ledger row for a retrieval step that leaves no tool artifact to extract.

    The deterministic sweep, the open-web fallback and the three short-circuits are all real
    searches made with direct backend calls, so nothing reaches ``extract_search_artifacts``.
    A ledger that omits them says the conversation never looked — which is exactly the answer
    the feature exists to prevent ("the available evidence does not specify…" over work already
    done). Facts are deliberately only the METHOD and the COUNT: titles and doc_ids are the
    payload bloat this module exists to avoid, and the documents themselves are in `evidence`
    for this turn and in the answer text thereafter.
    """
    args = {"query": _ledger_value(query, "query")}
    args.update({k: _ledger_value(v, k) for k, v in extra.items() if v not in (None, "", [], {})})
    return {"tool": tool, "args": args,
            "facts": {"search_method": method, "results_returned": int(returned)}}


def _delivers_layer(tool_name: str, payload: Any) -> bool:
    """Same authority the supervisor uses, so ledger and predicate cannot disagree."""
    try:
        from agent_runtime.map_layers import delivers_map_layer

        return delivers_map_layer(str(tool_name or ""), payload)
    except Exception:
        return False


def _ledger_rows(*contexts: Any) -> List[Dict[str, Any]]:
    """One row per tool INVOCATION: what ran, on what, and what came back.

    A call and its result arrive as separate records; they are merged here so the ledger
    reads as "embed_region(model=clay, …) -> pixel_ground_m=10" rather than as two half-rows
    the decider has to correlate itself.
    """
    # (name, kind, call_id, part) in ENCOUNTER order. The previous version bucketed by tool
    # name and zipped calls to results positionally, so a fail-then-succeed pair of the same
    # tool put the failed call's arguments on the successful call's result.
    records: List[Tuple[str, str, Optional[str], Dict[str, Any]]] = []

    def note(obj: Any) -> None:
        if not isinstance(obj, dict):
            return
        name = str(obj.get("name") or obj.get("tool_name") or "").strip()
        if not name:
            return
        # `content` is the shape extract_search_artifacts actually produces
        # ({name, tool_call_id, content}), and it is a JSON *string*. Reading only
        # output/result silently yielded empty payloads against real artifacts while every
        # unit test passed, because the tests were written to the assumed shape.
        payload = obj.get("content")
        if payload is None:
            payload = obj.get("output")
        if payload is None:
            payload = obj.get("result")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except ValueError:
                payload = None
        args = _pick(obj.get("args") or obj.get("arguments"), _LEDGER_ARGS)

        part: Dict[str, Any] = {}
        if args:
            part["args"] = args
        if isinstance(payload, dict):
            facts = _pick(payload, _LEDGER_FACTS)
            # rs-embed nests the fields that say what a number MEANS under `provenance` — and
            # embed_region nests that AGAIN, one entry per model under `models`. Reading only
            # the top level found nothing: the tool result's own keys are compute /
            # embedding_package / map_layer / models, so every fact that answers "what
            # resolution was that" sits two levels down.
            for entry in (payload.get("models") or []):
                if isinstance(entry, dict):
                    facts = {**facts, **_pick(entry, _LEDGER_FACTS)}
                    if isinstance(entry.get("provenance"), dict):
                        facts = {**_pick(entry["provenance"], _LEDGER_FACTS), **facts}
            prov = payload.get("provenance")
            if isinstance(prov, dict):
                facts = {**_pick(prov, _LEDGER_FACTS), **facts}
            # The esda tools nest their numbers a level down under `results`
            # ({morans_i: {statistic, p_value, significance}}) — the same shape problem
            # `provenance` had. A flat fact key would capture a stringified dict cut at 80
            # chars instead of the number, so Moran's I was simply lost.
            res = payload.get("results")
            if isinstance(res, dict):
                for stat, body in list(res.items())[:4]:
                    if isinstance(body, dict) and body.get("statistic") is not None:
                        sig = body.get("significance") or f"p={body.get('p_value')}"
                        facts[stat] = _ledger_value(f"{body['statistic']} ({sig})", stat)
            # The SAME json key means different things depending on the tool. A retrieval
            # payload's top-level `source` is the search METHOD ("keyword") and its `count` is a
            # hit count, while rs-embed's `source` is the IMAGERY ("sentinel-2"). Renamed at
            # capture, keyed on the tool, because the phrase book cannot tell them apart: the
            # sentence it actually produced was "imagery source: keyword".
            if name in _LEDGER_SEARCH_TOOLS:
                facts = {_LEDGER_SEARCH_RENAMES.get(k, k): v for k, v in facts.items()}
            if facts:
                part["facts"] = facts
            # A failed call used to render IDENTICALLY to a success, under a note instructing
            # the model not to re-derive the work. Row-level, not a fact, so the fact-dict
            # merges above cannot reorder it away.
            if (payload.get("ok") is False
                    or (payload.get("exit_code") not in (None, 0))
                    or payload.get("timed_out") is True):
                part["failed"] = True
                err = payload.get("error") or payload.get("hint")
                if err:
                    part["error"] = _ledger_value(err, "error")
            arts = payload.get("artifacts")
            if isinstance(arts, list):
                names = [str(a.get("filename")) for a in arts[:3]
                         if isinstance(a, dict) and a.get("filename")]
                if names:
                    part["outputs"] = _ledger_value(", ".join(names))
            if payload.get("file_id"):
                part["file_id"] = _ledger_value(payload["file_id"])
            # The ledger is now the ONLY cross-turn map signal (_map_delivered_earlier reads
            # this field), so a real delivery that carries no label must still land a row —
            # otherwise a layer from turn 2 stops counting in turn 4 and the auditor staples a
            # hallucination caveat onto a layer that is on the user's screen.
            # A tool can put SEVERAL layers on the map in one call — embed_zones emits the
            # pixel raster AND the zone groups, embed_region one per model — and they live
            # under the PLURAL key, with `map_layer` holding only the first. Reading the
            # singular alone recorded one and lost the rest, so the mechanism that exists to
            # stop the answerer claiming "no map was produced" under-reported the map itself.
            labels: List[Any] = []
            for cand in (payload.get("map_layers") or []):
                if isinstance(cand, dict) and cand.get("label"):
                    one = _ledger_value(cand["label"])
                    if one not in labels:
                        labels.append(one)
            ml = payload.get("map_layer")
            if isinstance(ml, dict) and ml.get("label"):
                one = _ledger_value(ml["label"])
                if one not in labels:
                    labels.append(one)
            if labels:
                kept, dropped = labels[:_LEDGER_LAYERS_PER_ROW], len(labels) - _LEDGER_LAYERS_PER_ROW
                if dropped > 0:
                    kept = kept + [f"+{dropped} more"]
                part["map_layer"] = kept
            elif _delivers_layer(name, payload):
                part["map_layer"] = [_ledger_value(name or "map layer")]
        # `walk` visits EVERY nested dict, and some carry a `name` that is not a tool
        # (admin_boundary's `matched` entries are {"name": "Champaign County", ...}), so the
        # empty-part guard has to stay for those. But never drop something that is
        # STRUCTURALLY a call or a result: "embed_zones ran and failed" is information, and
        # discarding it is what shifted the positional pairing in the first place.
        is_call = "args" in obj or "arguments" in obj
        is_result = bool(obj.get("tool_call_id")) or payload is not None
        if not part and not (is_call or is_result):
            return
        if is_call and not is_result:
            kind, cid = "call", obj.get("id")
        elif is_result and not is_call:
            kind, cid = "result", obj.get("tool_call_id")
        else:
            # Ambiguous (or a peer's flat dict): fall back to what the part looks like.
            looks_like_result = any(k in part for k in ("facts", "file_id", "map_layer", "failed"))
            kind = "result" if looks_like_result else "call"
            cid = obj.get("tool_call_id") or obj.get("id")
        records.append((name, kind, cid, part))

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            note(obj)
            for v in obj.values():
                walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v)

    for ctx in contexts:
        walk(ctx)

    calls = [(i, n, cid, p) for i, (n, k, cid, p) in enumerate(records) if k == "call"]
    results = [(i, n, cid, p) for i, (n, k, cid, p) in enumerate(records) if k == "result"]
    by_id = {cid: (i, p) for i, _n, cid, p in results if cid}
    used: set = set()

    rows: List[Dict[str, Any]] = []
    for _i, name, cid, part in calls:
        row: Dict[str, Any] = {"tool": name, **part}
        if cid and cid in by_id:
            ri, rpart = by_id[cid]
            row.update(rpart)
            used.add(ri)
        rows.append(row)

    # The CLI peers build {"name", "args"} / {"name", "content"} with no ids at all
    # (claude_peer, opencode_peer), so an id-less pair still needs positional matching — but
    # only among the leftovers, and only within one tool name.
    leftover = [(i, n, p) for i, n, cid, p in results if i not in used and not cid]
    for idx, (ri, name, rpart) in enumerate(leftover):
        target = next((r for r in rows if r["tool"] == name and not _row_has_result(r)), None)
        if target is not None:
            target.update(rpart)
            used.add(ri)

    # A result with no recorded call is information, not noise.
    for i, name, _cid, part in results:
        if i not in used:
            rows.append({"tool": name, **part})

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        key = json.dumps(row, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    return deduped


# The key names are the service's, not the user's. Given `scale_m=10` the model answered
# "the available evidence does not specify the ground-resolution" — it did not connect the
# two. Spelling the field out in the words a user would use is what closes that gap.
_FACT_PHRASES = {
    "image_size": "model input chip: {v} px on a side — the imagery is RESAMPLED to this, so "
                  "it is not the sensor's native resolution",
    "input_size_hw": "model input size (h, w) in px: {v} — imagery is resampled to this",
    "patch_size": "ViT patch size: {v} px, so each output token covers {v}x{v} input pixels",
    "grid_hw_tokens": "output token grid (h, w): {v}",
    "features_total": "features in the full set: {v} — a lower shown count means the map has a SAMPLE",
    "source": "data source: {v}",
    "sensor": "sensor: {v}",
    "scale_m": "ground resolution / pixel size: {v} m per pixel (this IS the resolution it was computed at)",
    "pixel_ground_m": "ground resolution / pixel size: {v} m per pixel",
    "scale_m_mercator": "web-mercator pixel size: {v} m",
    "dims": "embedding dimensions: {v}",
    "dim": "embedding dimensions: {v}",
    "model": "model: {v}",
    "year": "imagery year: {v}",
    "geoid": "GEOID: {v}",
    "feature_count": "features: {v}",
    # The four that used to fall through to a bare key=value — the exact failure the phrase
    # table was built to prevent.
    "count": "items: {v}",
    "level": "administrative level: {v}",
    "tiles_fetched": "imagery tiles fetched: {v}",
    "zone_id_field": "zone id column: {v} (pass this as zone_id_field)",
    "verdict": "statistical verdict: {v}",
    "features_analyzed": "features analysed: {v}",
    "column": "variable analysed: {v}",
    "crs": "coordinate system: {v}",
    "filename": "output file: {v}",
    "zones_total": "zones in the layer: {v}",
    "zones_with_pixels": "zones that actually got pixels: {v}",
    "tiles_planned": "imagery tiles a full sweep would need: {v}",
    "truncated": "PARTIAL COVERAGE — {v}",
    "row_count": "rows: {v}",
    "cells": "grid cells: {v}",
    "search_method": "retrieval method: {v}",
    "results_returned": "documents returned: {v}",
    "morans_i": "Moran's I: {v}",
    "gearys_c": "Geary's C: {v}",
    "getis_ord_g": "Getis-Ord G: {v}",
}


def _fact_phrase(key: str, value: Any) -> str:
    template = _FACT_PHRASES.get(key)
    return template.format(v=value) if template else f"{key}={value}"


def _ledger_lines(rows: List[Dict[str, Any]]) -> List[str]:
    """One human-readable line per ledger row, budget-trimmed.

    Shared by the two consumers that must agree: the note handed to the ANSWERING model, and
    the execution record handed to the grounding AUDITOR. When only the answerer got these
    lines, a follow-up correctly answered from an earlier turn's tool result was audited
    against evidence that never mentioned it and flagged as high-severity hallucination —
    the feature's two halves contradicting each other in front of the user.
    """
    # Budgeted by RENDERED length — the unit _LEDGER_MAX_CHARS documents — plus one char per
    # line for the newline the caller joins them with.
    return [_ledger_line(r)
            for r in _budgeted(rows or [], size=lambda row: len(_ledger_line(row)) + 1)]


def _ledger_line(r: Dict[str, Any]) -> str:
    """One row exactly as the answering model and the auditor read it."""
    bits = [("FAILED " if r.get("failed") else "") + str(r.get("tool"))]
    if r.get("args"):
        bits.append("(" + ", ".join(f"{k}={v}" for k, v in r["args"].items()) + ")")
    if r.get("failed"):
        # A failed call has no facts. Showing its ARGUMENTS and its error is the half that
        # matters: the note tells the model not to re-derive completed work, and a failure
        # rendered as a success is exactly how "already done" gets said about work that
        # never happened.
        bits.append(f"-> DID NOT RUN: {r.get('error') or 'the tool returned ok=false'}")
    elif r.get("facts"):
        bits.append("-> " + ", ".join(_fact_phrase(k, v) for k, v in r["facts"].items()))
    # A failed call delivered nothing. Rendering its layer or its outputs would contradict
    # the visible-state section built from these same rows, and it is the same bug class as
    # a failed call wearing a successful one's result.
    if not r.get("failed"):
        if r.get("outputs"):
            # The id as well as the name: it was captured and never rendered, so a peer saw
            # "[produced champaign.geojson]" and had to hope a bare filename resolved.
            # Gated on `outputs`, NOT on file_id: read_text_file and
            # inspect_file_for_analysis both return the file_id of the file the USER
            # uploaded and create nothing, so keying on file_id alone claimed they had
            # produced it — a fabrication the grounding auditor then confirms, since it
            # reads these same lines as evidence.
            fid = r.get("file_id")
            bits.append(f"[produced {r['outputs']}, file_id {fid}]" if fid
                        else f"[produced {r['outputs']}]")
        if r.get("map_layer"):
            bits.append(f"[on the map as {_layer_phrase(r['map_layer'])}]")
    return "- " + " ".join(bits)


# The section header the synthesizer prompt names, so the two cannot drift apart.
_LEDGER_HEADING = "What this conversation already did (tool records from EARLIER turns)"


# Extensions that cannot carry geometry. A tool needing a vector FILE is unusable when the only
# upload is one of these — offering it is not just wasted schema, it invites a call that must
# fail. Anything not on this list, including an unrecognised extension, counts as possibly-vector
# and binds everything: misclassifying here would recreate the absent-tool bug that caused half
# the selection failures in this repo's history, and a per-call filter cannot widen afterwards.
_TABULAR_ONLY_SUFFIXES = frozenset({".csv", ".tsv", ".txt", ".xlsx", ".xls"})
# Needs a vector file to do anything. add_map_layer and render_map_image are deliberately absent:
# they work from geometry the peer already holds.
_VECTOR_FILE_TOOLS = frozenset({"inspect_vector", "reproject_vector", "vector_spatial_join",
                                "vector_to_geojson"})


def _uploads_are_tabular_only(input_file_ids: Optional[List[str]]) -> bool:
    """True only when EVERY upload is a recognised non-geometry format.

    Fails open in every uncertain case — no ids, an id that will not resolve, an extension not
    on the list. The saving is small (~857 schema tokens); the reason to do it is that a CSV
    cannot be reprojected, and a tool that cannot work is a worse thing to offer than a tool
    that is merely irrelevant.
    """
    ids = [str(i) for i in (input_file_ids or []) if i]
    if not ids:
        return False
    try:
        from pathlib import PurePosixPath

        from agent_runtime.file_store import get_file_record

        for fid in ids:
            record = get_file_record(fid) or {}
            name = str(record.get("filename") or "")
            if not name:
                return False                       # unknown -> assume it may be vector
            if PurePosixPath(name.lower()).suffix not in _TABULAR_ONLY_SUFFIXES:
                return False
        return True
    except Exception:      # noqa: BLE001 - never let this decide by crashing
        return False


def _visible_state_lines(rows: List[Dict[str, Any]]) -> List[str]:
    """What the user can still SEE and download from earlier turns.

    A projection of the ledger rows, not new state: it regroups the `map_layer` and `outputs`
    fields ``_ledger_rows`` already sets, so it costs almost nothing on top of a note that is
    being sent anyway.

    Worth stating separately because the per-row form was not usable as an answer. Two
    mechanisms had to reconstruct exactly this by hand: the map-delivery predicate walked rows
    hunting for a layer, and ``_refs_in_history`` regexed the CONVERSATION TEXT to recover
    download links, because nothing carried them forward. And the map is persistent — a layer
    added in turn 2 is still on screen in turn 4 — so "no map was produced" is a false statement
    the answerer had no way to check.

    Only earlier turns, matching the note's heading: this turn's own layers and files are in the
    answer path already.
    """
    layers, files = [], []
    for row in rows or []:
        if not isinstance(row, dict) or row.get("failed"):
            continue                       # a failed call delivered nothing to look at
        for label in _layer_labels(row.get("map_layer")):
            if label not in layers:
                layers.append(label)
        out = row.get("outputs")
        if out and str(out) not in files:
            files.append(str(out))
    lines = []
    if layers:
        lines.append("- still on the user's map from earlier turns: " + ", ".join(layers))
    if files:
        lines.append("- already produced and downloadable: " + ", ".join(files))
    return lines


# What the client the layer landed in actually does. Stated to the AUDITOR only, because the
# audit prompt demands a VERBATIM span be copied for any row it marks "supported" and no tool
# result can ever provide one for an affordance: pan/zoom/toggle/click are properties of the
# viewer, not of the data. Without a span to copy the auditor has no honest option but "absent",
# which is how a correct answer earned a high-severity hallucination caveat for the one sentence
# describing the map it had just filled.
#
# Every clause is true of the deployed client: MapLibre supplies drag-pan and scroll-zoom, the
# layers panel toggles and removes each layer (App.tsx toggleLayer/removeLayerById), and the
# deck.gl overlay is wired with getTooltip and onClick over pickable vector layers
# (components/AgentMap.tsx). Heatmap and raster layers are deliberately NOT pickable, hence
# "vector" — do not widen this line past what the client does, since its whole purpose is to be
# a span the auditor can trust.
#
# This is the SECOND half of the fix, not a replacement for the first: _is_map_claim still drops
# the issue deterministically. The audit prompt's own history in evidence_quality.py records
# four prose formulations that were measured and did not hold, so prose alone is not the
# guarantee — it just stops the auditor from being asked an unanswerable question.
_MAP_CLIENT_AFFORDANCES = (
    "- the map these layers are on is a live client the user drives directly: they pan and zoom "
    "it, show/hide or remove any layer from the layers panel, and click a vector feature to see "
    "its attributes. This is a fact about the environment, not a claim needing evidence."
)


def _map_environment_lines(delivered: bool) -> List[str]:
    """The client-affordance line for the auditor's record, when a layer is actually there."""
    return [_MAP_CLIENT_AFFORDANCES] if delivered else []


# --- re-grounding: make the audit a gate, not just an annotation -------------------------
#
# Observed on "Which counties border Champaign County, Illinois?": the peer called
# admin_boundary, then ran an EXPLORATORY execute_code that downloaded a Census gazetteer and
# printed its filename list — computing no adjacency, and from a gazetteer it could not (those
# carry centroids, not geometry) — and then answered with six county names out of the model's
# own memory. The audit caught it exactly right: "the listed bordering counties and directions
# are not supported by the supplied evidence or execution record."
#
# The detection worked; there was nowhere for it to go. `synthesize` had an unconditional edge
# to END, so a correct finding of ungroundedness could only be stapled to the answer as a
# caveat, and the unfinished work stayed unfinished. This routes that finding back into the
# loop ONCE, telling the peer what was not grounded and that finishing the computation or
# admitting it cannot are both acceptable — inventing is not.
#
# Bounded deliberately: ONE pass per turn (_MAX_GROUNDING_RETRIES). The audit has a documented
# false-positive history, so an unbounded gate would let a wrong verdict spend the whole step
# budget. It also runs only AFTER _reconcile_audit_with_artifacts, whose four deterministic
# drops remove the affordance/artifact/number classes — so what reaches here is substantive by
# construction.
_MAX_GROUNDING_RETRIES = 1

_REGROUND_DIRECTIVE = (
    "IMPORTANT — a previous attempt at this same question produced an answer whose key claims "
    "were NOT present in any tool result, so it was rejected. The claims that could not be "
    "grounded were:\n{gaps}\n\n"
    "Do NOT restate them from your own knowledge. Either (a) actually compute or retrieve them "
    "now with the tools you have, so the values appear in a tool result, or (b) say plainly "
    "which parts you could not establish. A partial answer that is fully grounded is better "
    "than a complete one that is not. Note that downloading or inspecting a file is not the "
    "same as computing the answer: finish the computation and print the result."
)


def _reground_note(state: SupervisorState) -> Optional[str]:
    """The re-grounding directive for the peer's TASK text, or None when not re-grounding.

    Appended to the task only — never merged into ``query`` — because ``query`` is regexed by
    _detect_qgis_map_request, _WANTS_MAP_RE and _models_named_in, and by `analysis_node` into
    `searched_queries`. Folding a paragraph of quoted claims into it would corrupt all four.
    """
    gaps = [str(g).strip() for g in (state.get("grounding_gaps") or []) if str(g).strip()]
    if not gaps:
        return None
    return _REGROUND_DIRECTIVE.format(gaps="\n".join(f"  - {g}" for g in gaps))


def _unsupported_claims(audit: Optional[Dict[str, Any]], limit: int = 6) -> List[str]:
    """The claims a flagged audit could not ground, as plain strings."""
    out: List[str] = []
    for item in ((audit or {}).get("issues") or []):
        if isinstance(item, dict):
            claim = str(item.get("claim") or "").strip()
        else:
            claim = str(item or "").strip()
        if claim:
            out.append(claim)
        if len(out) >= limit:
            break
    return out


def _reground_target(state: SupervisorState) -> Optional[str]:
    """Which peer should try again — or None when no pass is available or useful.

    The corrective depends on what kind of answer went ungrounded. A COMPUTED answer failed
    because the computation was not finished, so analyze (which owns execute_code and the
    spatial toolkit) is the peer that can finish it. A RETRIEVED answer failed because nothing
    was found to support it, and there the corrective is another search, not an analysis run —
    routing every case to analyze would push a retrieval-only turn into the analysis peer for
    no reason.

    Every bound here is load-bearing: the retry counter caps the feature at one pass, the step
    budget is shared with every other route, and the per-peer cap is the same one `_dead`
    applies to a queued need — without it we would route to a peer whose need the supervisor is
    about to discard, spending a step to arrive back at the same answer.
    """
    if state.get("grounding_retries", 0) >= _MAX_GROUNDING_RETRIES:
        return None
    # synthesize -> supervisor -> peer is two steps before an answer can be composed again.
    if state.get("step", 0) + 2 > state.get("max_steps", DEFAULT_MAX_STEPS):
        return None
    actions = state.get("actions") or []
    computed = (state.get("analysis_results") is not None
                or state.get("code_result") is not None)
    if computed and actions.count("analyze") < _max_peer_runs():
        return "analyze"
    if (not computed and actions.count("search") < _max_peer_runs()
            and not _search_exhausted(state)):
        return "search"
    return None


def _can_reground(state: SupervisorState) -> bool:
    """Kept as the boolean form of :func:`_reground_target` for readability at the call site."""
    return _reground_target(state) is not None


def _prior_actions_note(rows: List[Dict[str, Any]]) -> Optional[str]:
    """The ledger as a line-per-action note for the ANSWERING model.

    Injected into synthesis as well as into routing, because the two failures are separate.
    Routing waste is expensive; an answer that says "the available evidence does not specify
    the ground resolution" when scale_m=10 is sitting in the previous turn's tool result is
    simply wrong, and it stays wrong however the supervisor routed.
    """
    lines = _ledger_lines(rows)
    if not lines:
        return None
    preamble = (f"{_LEDGER_HEADING}:\n"
                "These are facts about work already done — if the user is asking about it, answer "
                "from here rather than saying the information is unavailable, and do not re-derive "
                "it. A line marked FAILED records a tool that did NOT work: that work was never "
                "done, its result does not exist, and re-running it may be the right move — never "
                "describe it as completed.\n")
    # _LEDGER_MAX_CHARS is documented as "a hard ceiling on the rendered ledger", and it exists
    # BECAUSE a turn overflowed the context window — so it must not be able to cause that
    # itself. _ledger_lines honours it, but the visible-state section was appended afterwards
    # from the FULL row list, outside the budget, so the note this function returns could run
    # past the ceiling however tightly the lines were trimmed. Both halves are inside it now,
    # and the visible-state section is trimmed rather than dropped: telling the answerer what is
    # on the user's screen is the thing it was added for.
    body = "\n".join(lines)
    visible = _visible_state_note(rows)
    room = _LEDGER_MAX_CHARS - len(body)
    if visible and len(visible) > room:
        marker = " …"
        visible = visible[:max(0, room - len(marker))].rstrip()
        if visible:
            visible += marker
    return preamble + body + visible


def _visible_state_note(rows: List[Dict[str, Any]]) -> str:
    """The visible-state section, appended to the note when there is anything to see."""
    visible = _visible_state_lines(rows)
    if not visible:
        return ""
    return ("\n\nWHAT THE USER IS LOOKING AT (already delivered — do not say it was not "
            "produced, and do not re-add a layer that is already there):\n"
            + "\n".join(visible))


def _prior_actions(state: SupervisorState) -> List[Dict[str, Any]]:
    """The ledger for this thread, EXCLUDING anything recorded for the current turn."""
    thread_id = state.get("thread_id")
    if not thread_id:
        return []
    try:
        from agent_runtime.session_memory import get_session_actions

        return get_session_actions(str(thread_id))
    except Exception:  # noqa: BLE001 - a missing ledger must never break routing
        return []


def _record_actions(state: SupervisorState, *contexts: Any,
                    extra_rows: Optional[List[Dict[str, Any]]] = None) -> None:
    """Append this turn's rows to the thread's ledger.

    *extra_rows* carries rows a peer built itself because its work left no tool artifact to
    extract — the search node's deterministic sweep, the open-web fallback and the
    short-circuits.
    """
    thread_id = state.get("thread_id")
    if not thread_id:
        return
    rows = [*_ledger_rows(*contexts), *(extra_rows or [])]
    if not rows:
        # A ledger that silently records nothing is indistinguishable from one that is
        # working, which is exactly how this shipped inert the first time.
        _LEDGER_LOG.info("turn ledger: nothing extracted from %s",
                    [sorted(c)[:10] if isinstance(c, dict) else type(c).__name__
                     for c in contexts])
        return
    _LEDGER_LOG.info("turn ledger: recorded %d row(s) for thread %s: %s",
                len(rows), thread_id, sorted({str(r.get("tool")) for r in rows}))
    try:
        from agent_runtime.session_memory import append_session_actions

        append_session_actions(str(thread_id), rows)
        emit_trace_event(
            "turn_ledger_recorded",
            {"stage": "synthesize", "rows": len(rows),
             "tools": sorted({str(r.get("tool")) for r in rows}),
             "message": f"recorded {len(rows)} action(s) for follow-up turns"},
            node="synthesize",
        )
    except Exception:  # noqa: BLE001
        pass


def _distill(state: SupervisorState, *, for_decision: bool = False) -> Dict[str, Any]:
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
        # What the evidence SAYS. Written by the model that read it; descriptive, never a
        # verdict on sufficiency. Sits beside the lexical signals so a wrong summary can be
        # disagreed with rather than obeyed.
        "evidence_summary": state.get("evidence_summary"),
        "top_score": round(max(scores), 3) if scores else None,
        "queries_searched": list(state.get("searched_queries") or []),
        "has_analysis": state.get("analysis_results") is not None,
        "analysis_summary": _peer_summary(state.get("analysis_results")),
        # Whether the thing the user asked to SEE is already in front of them. The decider had
        # no way to know this: `has_analysis` says a peer ran, `artifacts_produced` lists images,
        # and neither answers "is the deliverable delivered?" — so after analyze put a DEM on the
        # map it routed to code, which fetched the same DEM again. Measured: 266s and 16
        # execute_code iterations to redo work already done in one call.
        "map_layer_delivered": _map_delivered_this_turn(state.get("analysis_results"),
                                                        state.get("code_result")),
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
        # What the decider may actually choose this step (see _available_actions).
        "available_actions": _available_actions(state),
        # Decision-only: this is the one consumer that needs to know the conversation did
        # not start just now. Kept out of the client payload, which is a per-turn record.
        # THIS turn's ledger, in the same rendering the answering model and the grounding
        # auditor read. The rows were always kept — they are what the trace shows as
        # "dem_for_region(...) -> 1 layer on the map" — but _ledger_lines had exactly two
        # consumers and the decider was not one of them. It saw counts and flags about the
        # current turn and the ledger only of PREVIOUS turns, so it could not tell that the
        # tool it was about to route to had already run and produced the answer.
        **({"this_turn": _ledger_lines([*_ledger_rows(state.get("analysis_results"),
                                                      state.get("code_result")),
                                        *(state.get("action_rows") or [])]),
            "this_turn_note": (
                "What THIS turn has already done, oldest first — the same record the answering "
                "model and the auditor see. A line here is work that is DONE: routing to a peer "
                "to redo it produces a second copy, not a better answer. A line marked FAILED "
                "means the tool did not run and its result does not exist.")}
           if for_decision else {}),
        **({"prior_turns_in_this_conversation": _budgeted(_prior_actions(state)),
            "prior_turns_note": (
                "What THIS conversation already did, oldest first. If the user's question is "
                "about work already listed here, choose 'done' — the answer is in hand and "
                "re-running search or analyze would only rediscover it. Treat these as a "
                "record of past turns, NOT as inputs to reuse blindly: check the args match "
                "what the user is asking about now.")}
           if for_decision and _prior_actions(state) else {}),
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


def _available_actions(state: SupervisorState) -> List[str]:
    """The actions that are legal RIGHT NOW, in decider-menu order.

    The supervisor already vetoes an exhausted ``search`` and a back-to-back peer
    repeat *after* the decider answers. Computing the same set here lets the decider
    be shown what it may actually pick, instead of the full menu plus prose telling
    it which entries are forbidden — the veto stays as a backstop rather than being
    the mechanism.
    """
    actions: List[str] = []
    # With the peers merged there is no separate retrieval peer to route to: the one agent
    # retrieves and analyses in the same loop, so offering `search` would route to a node that
    # duplicates what `analyze` already does — and split the context again.
    if not unified_peer_enabled(state) and not _search_exhausted(state):
        actions.append("search")
    for cap in ("analyze", "code"):
        if not _is_unproductive_repeat(cap, state):
            actions.append(cap)
    # With the peers merged, `done` must not be legal before ANYTHING has run. Removing
    # `search` from the menu also removed the decider's cue that retrieval was needed:
    # measured, "Find flood risk datasets on I-GUIDE" went straight to done at step 0 and
    # answered "I couldn't find any supporting material" without ever retrieving. In the
    # peered shape `search` was the obvious opening move and carried that signal implicitly.
    if unified_peer_enabled(state) and not (state.get("actions") or []) \
            and not (state.get("evidence") or []) and state.get("analysis_results") is None:
        return actions or ["analyze"]
    actions.append("done")
    return actions


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


def _correct_artifact_claims(answer: str, *contexts: Any,
                             prior_rows: Optional[List[Dict[str, Any]]] = None) -> str:
    """Deterministic corrections for an answer that misdescribes what was delivered.

    Both cases were produced by one live query. Neither was caught by the LLM grounding
    audit — it read the answer as well-supported, because the web results it cited were real;
    they just were not where the delivered raster came from.
    """
    text = str(answer or "")
    if not text.strip():
        return answer
    notes = []

    used = set()
    for ctx in contexts:
        used |= _models_used(ctx)
    if used:
        claimed = _models_named_in(text) - used
        if claimed:
            ran = ", ".join(sorted(used))
            notes.append(
                f"This embedding was produced by the **{ran}** model, not "
                f"{' / '.join(sorted(claimed))}. Any description above of where the vectors "
                "come from (an external collection, grid or dimensionality) describes that "
                f"other model, not the layer you were given — which was computed here by {ran}."
            )

    if _DENIES_MAP_RE.search(text) and (any(_map_delivered_this_turn(c) for c in contexts)
                                         or _map_delivered_earlier(prior_rows)):
        notes.append(
            "The layer is already on your interactive map — it was added automatically. "
            "There is no need to add it from a URL; the download link is only if you want a "
            "copy of the file."
        )

    if not notes:
        return answer
    return text + "\n\n---\n\n" + "\n\n".join(f"⚠️ Correction: {n}" for n in notes)


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


# The auditor compares an answer against retrieved DOCUMENTS, and no document ever says
# "a layer is on the user's map" — so a correct map claim looks unsupported to it. Observed:
# a run that really did deliver a 31,977-point density layer plus a 708-cell grid choropleth
# was stamped "high-severity hallucination ... unsupported claims about the interactive map".
# The tool record settles it, so reconcile against that instead of warning the user off a
# true statement. (A claim made with NO layer delivered still gets flagged — that is the
# failure the analyze peer's own map check exists to catch.)
_MAP_CLAIM_MARKERS = (
    "interactive map", "on the map", "on your map", "map beside", "map layer",
    "heat layer", "density layer", "displayed on", "shown on the map", "panning", "zooming",
)


# An AFFORDANCE claim — what the user can DO with the map a layer just landed on — is the
# residue left after the markers above. Pan, zoom, layer toggle and click-for-attributes are
# properties of the CLIENT (deck.gl's getTooltip/onClick over MapLibre, plus the layer panel's
# show/hide), not of the data, so NO tool result can ever carry a span evidencing one. The
# auditor therefore marks the row "absent" every single time the synthesizer describes the map
# it was just told to describe — and SYNTHESIS_PROMPT rule 7 and VISUALIZATION_ROUTES_RULE both
# tell it to, because the alternative (answers claiming no map exists, or pointing the user at
# QGIS to view their own result) is the bug those rules were written to fix.
#
# Observed: "Show hospitals near Chicago on the map" delivered one layer with 49 features, the
# auditor accepted the count, the location and the OpenStreetMap source, and flagged only
# "You can pan, zoom, and click the hospital markers for details" — high severity, so the user
# got a hallucination caveat stapled to a wholly correct answer.
#
# Matched with WORD BOUNDARIES and in one of two shapes, never as a bare substring: "pan" as a
# substring hits "expand"/"Japan"/"company", and "click" hits the "[popularity: 42 clicks]" that
# real evidence carries — which would turn a fabricated click-count into an amnestied claim.
#   1. a capability frame aimed at the user ("you can …", "lets you …") + any affordance verb
#   2. a GESTURE verb applied to a map noun ("click the hospital markers for details"), with no
#      frame needed because the sentence often has none.
#
# The two verb sets differ on purpose. "select" and "inspect" describe analysis as readily as
# interaction, so tier 2 would read "the model selected 4096 features" as a map affordance and
# amnesty an invented figure; they are admitted only under tier 1's explicit frame. Tier 2 is
# restricted to verbs that mean nothing else here — no analysis step pans or zooms.
_MAP_AFFORDANCE_VERBS = (r"pan(?:s|ned|ning)?|zoom(?:s|ed|ing)?|toggl(?:e|es|ed|ing)|"
                         r"click(?:s|ed|ing)?|tap(?:s|ped|ping)?|hover(?:s|ed|ing)?|"
                         r"select(?:s|ed|ing)?|inspect(?:s|ed|ing)?|explor(?:e|es|ed|ing)|"
                         r"drag(?:s|ged|ging)?|show/hide|hide")
_MAP_GESTURE_VERBS = (r"pan(?:s|ned|ning)?|zoom(?:s|ed|ing)?|toggl(?:e|es|ed|ing)|"
                      r"click(?:s|ed|ing)?|tap(?:s|ped|ping)?|hover(?:s|ed|ing)?|"
                      r"explor(?:e|es|ed|ing)|show/hide")
_MAP_AFFORDANCE_NOUNS = (r"map|layers?|markers?|features?|points?|polygons?|shapes?|"
                         r"attributes?|details?|popup|pop-up|tooltip|legend")
_MAP_AFFORDANCE_RE = re.compile(
    rf"\b(?:you|users?|they)\s+(?:can|could|may|are\s+able\s+to|will\s+be\s+able\s+to)\b"
    rf"[^.;]{{0,120}}?\b(?:{_MAP_AFFORDANCE_VERBS})\b"
    rf"|\b(?:allows?|lets?|enables?)\s+(?:you|users?|them)\b[^.;]{{0,120}}?"
    rf"\b(?:{_MAP_AFFORDANCE_VERBS})\b"
    rf"|\b(?:{_MAP_GESTURE_VERBS})\b[^.;]{{0,40}}?\b(?:{_MAP_AFFORDANCE_NOUNS})\b",
    re.I,
)


def _is_map_claim(claim: str) -> bool:
    """A claim about the user's map: that a layer is on it, or what they can do with it there.

    Only consulted when a layer really was delivered (this turn or an earlier one) — with no
    delivery every one of these still gets flagged, which is what keeps a FAILED
    ``admin_boundary`` from claiming a layer it never produced.
    """
    text = str(claim or "")
    low = text.lower()
    return (any(m in low for m in _MAP_CLAIM_MARKERS)
            or bool(_MAP_AFFORDANCE_RE.search(text)))


# A tool result usually arrives as a JSON STRING, not a parsed dict, so a structural walk
# alone misses the delivery: the spatial toolkit (buffer_layer, aggregate_to_grid,
# cluster_points, …) reports on_map/map_layer inside that string and is not named
# add_map_layer, which is how a genuinely-delivered 2 km buffer still drew a
# "hallucinated claims about buffering and map display" caveat over nine real artifacts.
# Match the payload itself rather than enumerating tool names, so new layer-emitting tools
# are covered the day they are added.
def _map_delivered_this_turn(*contexts: Any) -> bool:
    """A layer THIS turn actually reached the user's map.

    Asks the delivery boundary itself (:func:`map_layers.delivers_map_layer`) per tool result,
    and requires the tool to have SUCCEEDED. The four signals this replaces each answered by
    pattern: a tool NAME in tool_calls, a bare ``"on_map": true`` anywhere in a nested payload,
    a regex over the JSON blob. Every one of them said "delivered" for a failed
    ``admin_boundary`` — which returns ``{"ok": false}`` with no descriptor on its ambiguity and
    error paths — so the supervisor suppressed its own corrective retry, wrote the conclusion
    into its result, and then RE-READ that conclusion as evidence a layer existed.

    Reads only ``tool_results`` entries and the peer-level descriptor, never a bare ``on_map``
    key. That is what severs the feedback loop: the supervisor's own conclusion (stored as
    ``result["on_map"]``) is no longer visible to this predicate, while ``on_map`` stays a
    legitimate protocol field for the tools that emit it.
    """
    from agent_runtime.map_layers import delivers_map_layer

    def _payload(content: Any) -> Any:
        if isinstance(content, str):
            try:
                return json.loads(content)
            except Exception:
                return None
        return content

    def walk(obj: Any) -> bool:
        # The execution context is a WRAPPER — {"analysis_results": ..., "code_result": ...} —
        # so tool_results sit a level down; the analyze peer passes its artifacts directly, so
        # they sit at the top. Both shapes reach here, hence the descent.
        if isinstance(obj, dict):
            # A real descriptor anywhere is a real delivery (the CLI peers put theirs at the
            # top level of their result). A bare `on_map` or a tool NAME is not: delivers_
            # map_layer requires a descriptor with a url, or inline features. That asymmetry is
            # what keeps the supervisor's own result["on_map"] from proving itself.
            if delivers_map_layer("", obj):
                return True
            for entry in obj.get("tool_results") or []:
                if not isinstance(entry, dict):
                    continue
                payload = _payload(entry.get("content"))
                if isinstance(payload, dict) and payload.get("ok") is False:
                    continue                # a tool that failed delivered nothing
                if delivers_map_layer(str(entry.get("name") or ""), entry.get("content")):
                    return True
            return any(walk(v) for k, v in obj.items() if k != "tool_results")
        if isinstance(obj, (list, tuple)):
            return any(walk(v) for v in obj)
        return False

    return any(walk(ctx) for ctx in contexts)


def _map_delivered_earlier(prior_rows: Optional[List[Dict[str, Any]]]) -> bool:
    """A layer from an EARLIER turn is still on the user's map.

    The map is persistent: a layer added in turn 2 is still on screen in turn 4, so "the tracts
    are shown on the map" is true then and must not be audited as an unsupported claim.

    Reads the ledger row's ``map_layer`` FIELD. The previous version matched the literal
    ``"[on the map as "`` that ``_ledger_lines`` writes — two hand-synced strings in different
    functions, where a formatting change in one would silently switch the other off and hand the
    user a hallucination caveat over a layer that really is on their screen.
    """
    return any(isinstance(r, dict) and r.get("map_layer") for r in (prior_rows or []))


def _map_layer_was_delivered(execution_context: Optional[Dict[str, Any]],
                             prior_rows: Optional[List[Dict[str, Any]]] = None) -> bool:
    """This turn, or any earlier one. The single predicate every caller uses."""
    return _map_delivered_this_turn(execution_context) or _map_delivered_earlier(prior_rows)


def _reconcile_audit_with_artifacts(audit: Optional[Dict[str, Any]],
                                    artifacts: List[Dict[str, str]],
                                    execution_context: Optional[Dict[str, Any]] = None,
                                    prior_rows: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """Deterministic override of LLM-auditor false positives. Drops an audit issue when it:
    (1) merely disputes artifact generation/availability and an artifact WAS produced,
    (2) disputes a numeric value that actually appears in the execution record, or
    (3) carries a reason that itself concedes the claim is grounded/correct (and no genuine
    contradiction marker), or (4) disputes a claim about the interactive map when a layer was
    actually delivered to it. The verdict is cleared if no substantive issues remain. Genuine
    unsupported claims (a wrong statistic, an invented finding, a map that never got a layer)
    are preserved."""
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
    map_delivered = _map_layer_was_delivered(execution_context, prior_rows)
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
        if map_delivered and _is_map_claim(claim):
            continue  # (4) a map claim, and a layer really did reach the map
        if any(g in reason for g in _GROUNDED_REASON_MARKERS) and not any(c in reason for c in _CONTRADICTION_MARKERS):
            continue  # (3) the auditor's own reason concedes grounding
        nums = _claim_numbers(claim)
        if nums and blob and all(n in blob for n in nums):
            continue  # (2) every disputed number is present in the execution record
        kept.append(it)
    if not kept:
        return {"hallucination_detected": False, "severity": "none", "issues": [],
                "summary": "Grounded: flagged claims are supported by the produced artifact(s), the "
                           "delivered map layer(s) and the execution record."}
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


# A file id as it appears in an agent download URL or cited bare in an answer.
_HISTORY_FILE_ID_RE = re.compile(r"\bfile_[0-9a-f]{6,}\b", re.I)
_HISTORY_FILE_URL_RE = re.compile(r"[^\s\(\)\[\]\"'<>]*/agent/files/[^/\s]+/download", re.I)


def _refs_in_history(chat_history: Optional[List[Any]]) -> Dict[str, List[str]]:
    """Artifact references the conversation ALREADY offered, as ``{"file_ids", "urls"}``.

    ``sanitize_answer_links`` verifies a download link only against the allowlist handed to it
    (``runtime_utils.sanitize_answer_links``) — there is no file-store lookup — and the
    allowlist is built from THIS turn's ``analysis_results``/``code_result``. Today an earlier
    turn's artifact survives by accident, because a peer's checkpointed thread replays its old
    tool results into this turn's payload. Scope those artifacts to the turn that produced them
    (the correct fix for four verifiers that are currently fooled by the same replay) and the
    accident stops: a turn-4 answer offering a turn-1 CSV would have its link silently degraded
    to plain text.

    So read the references out of the conversation itself. This is strictly better than relying
    on the replay even before that change lands: the ``claude`` and ``opencode`` peers return a
    plain dict and never had a checkpointed thread, so an artifact THEY produced in an earlier
    turn has never been re-offerable.

    A file id is only trusted here because it was already emitted to this user in this
    conversation — an id the model invents still fails the check.
    """
    text = _raw_history_text(chat_history)
    if not text:
        return {"file_ids": [], "urls": []}
    return {
        "file_ids": sorted({m.group(0) for m in _HISTORY_FILE_ID_RE.finditer(text)}),
        "urls": sorted({m.group(0) for m in _HISTORY_FILE_URL_RE.finditer(text)}),
    }


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


def _capability_inventory(capability: str) -> str:
    """What a peer can actually do, from ``agent_runtime.capability_registry``.

    GENERATED rather than written here on purpose. The hand-written version drifted behind the
    peers three times without anyone noticing — terrain, administrative boundaries and geocoding
    were all bound to a peer while the supervisor's description of it never mentioned them, and
    a DEM request became a knowledge-base search as a direct result. The reasoning guidance in
    this prompt stays hand-written, because that is judgement rather than inventory.
    """
    try:
        from agent_runtime.capability_registry import describe
        return describe(capability)
    except Exception:  # noqa: BLE001 - a prompt must still be produced
        logger.exception("capability inventory unavailable; falling back to a generic phrase")
        return "geospatial analysis over the evidence or uploaded files"


def default_decide_fn(llm: Optional[Any] = None) -> DecideFn:
    """LLM-driven next-action chooser with a deterministic heuristic fallback."""

    def decide(state: SupervisorState, distilled: Dict[str, Any]) -> str:
        history = _format_chat_history(state.get("chat_history"))
        available = distilled.get("available_actions") or list(ALLOWED_ACTIONS)
        prompt = (
            "You are the orchestration supervisor for a geospatial research agent.\n"
            "Choose the SINGLE next action. Capabilities are peers you can use in any "
            "order and repeat as needed:\n"
            "- search: retrieve evidence (datasets, publications, notebooks)\n"
            "- analyze: run a workflow with EXISTING purpose-built tools over the evidence or "
            "uploaded files. It can currently do: "
            + _capability_inventory("analyze") + ". "
            "Anything in that list is analyze work, not a retrieval question — a DEM, a "
            "boundary and a geocode all come from live services, not from the knowledge base, "
            "so searching for them finds writing ABOUT them and never the thing itself. "
            "It ALSO computes remote-sensing foundation-model embeddings for a map region: "
            "embedding a drawn area, segmenting it into look-alike zones, measuring how much "
            "it changed across years, comparing two areas, and running pretrained heads. "
            "Embedding also saves a package of the real vectors, so work those tools cannot "
            "express — an unusual k, a metric of your own, more than two periods, a per-pixel "
            "change surface — can be written against that package instead and delivered with "
            "add_map_layer or add_raster_layer. That route needs code execution; the one-shot "
            "tools do not. "
            "Model names (gse, tessera, prithvi, terrafm, satmae, ...) are ARGUMENTS, not datasets "
            "to retrieve — a request naming one is analyze work, not search.\n"
            "- code: produce and run NEW code for work no existing tool covers. It binds the "
            "same toolkit as analyze, plus packaged skills and saved workflows\n"
            "- done: stop; a grounded final answer is composed automatically from the "
            "conversation + evidence + analysis results + code\n\n"
            f"Actions available this step: {', '.join(available)}. "
            "Anything else has been ruled out already — a search whose sources are exhausted, or "
            "a peer that just ran and would only repeat itself.\n"
            "Each peer iterates internally: the code peer runs and debugs its own code, search "
            "issues several queries in one pass, and analyze chains its tools. A peer's result "
            "therefore already reflects the work it could do with the inputs it had.\n"
            "Use the conversation so far for context: when the request refers to something "
            "already produced earlier (e.g. 'show me the code', 'explain that', 'what did you "
            "find'), the answer is composed from that conversation, so 'done' is enough unless "
            "genuinely new external information is needed.\n"
            "Peers may also REQUEST a capability they need (e.g. code needs evidence); such "
            "requests are fulfilled automatically before you are consulted again.\n"
            "`map_layer_delivered` in Progress means a layer is ALREADY on the user's map. When "
            "the request was to see something and it is there, choose `done` — `code` exists for "
            "work no existing tool covers, not for redoing work a tool has already done, and a "
            "second pass fetches the same data again and draws a second copy of the same layer. "
            "Choose `code` after a successful analyze only when the request asks for something "
            "the delivered result does not contain.\n"
            "`evidence_summary` in Progress is a DESCRIPTION of what was retrieved, written by "
            "the model that read it. It deliberately does not say whether the evidence is "
            "sufficient — that is your call. Search again only when it names a specific gap a "
            "DIFFERENT query could fill; repeating a search because the count looks small "
            "returns the same documents and wastes the step. And when the request is work for a "
            "tool rather than a question about the literature — computing a DEM, buffering, "
            "embedding a region — retrieval cannot help at all, however thin the evidence "
            "looks.\n\n"
            "Respond ONLY with JSON: {\"next\": \"" + "|".join(available) + "\", \"reason\": \"...\"}\n\n"
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


def _direct_search_sweep(query: str, enabled_search_methods: Optional[List[str]],
                         *, k: int = 8) -> List[Dict[str, Any]]:
    """Deterministic multi-method retrieval sweep: run keyword AND semantic search directly
    (cheap OpenSearch calls, no LLM) so every search turn has baseline coverage from BOTH
    core methods regardless of which tools the LLM SearchAgent chose to call — it frequently
    stops after a single tool, leaving results incomplete. Respects the request's
    enabled_search_methods allowlist. Never raises; each method degrades independently."""
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
            open_peer_session,
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
            _docs = _related_elements_evidence(related_id)
            return {"documents": _docs,
                    "action_rows": [_search_row("related_elements", query, "knowledge graph",
                                                len(_docs or []), element_id=related_id)]}
        lookup_id = _detect_element_lookup_request(query)
        if not lookup_id and _EXPLAIN_FOLLOWUP_RE.match(query or ""):
            lookup_id = _recall_recent_element_id(chat_history)
        if lookup_id:
            emit_trace_event(
                "node_started",
                {"stage": "search", "message": f"Element lookup for {lookup_id}"},
                node="search",
            )
            _docs = _element_lookup_evidence(lookup_id)
            return {"documents": _docs,
                    "action_rows": [_search_row("element_lookup", query, "by id",
                                                len(_docs or []), element_id=lookup_id)]}
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
                return {"documents": pop_docs,
                        "action_rows": [_search_row("popularity_ranking", query,
                                                    "click_count ranking", len(pop_docs))]}
            # graph empty/unreachable -> fall through to the normal search agent

        executor = build_search_agent_executor(
            llm=llm, tool_strategy=tool_strategy, include_mcp_tools=include_mcp_tools,
            mcp_modules=mcp_modules, enabled_search_methods=enabled_search_methods,
            skill_roots=skill_roots,
        )
        # The search peer has its OWN checkpointed thread that accumulates across turns — it is
        # NOT fresh each turn, whatever this comment used to say. What it lacks is any view of
        # the OTHER peers' work, so the answer can sit in the analyze peer's thread while the
        # router sends the follow-up here. That is how "what resolution was that" became a sweep for
        # SoilGrids soil clay while scale_m sat in the previous turn's tool result.
        _retrieval_q = _as_retrieval_request(query)
        _search_note = _prior_actions_note(_prior_actions(state))
        if _search_note:
            _retrieval_q = f"{_retrieval_q}\n\n{_search_note}"
        # `_reground_target` routes a RETRIEVED answer back here, so this peer needs the
        # directive as much as analyze does — without it a retrieval-side re-grounding pass
        # re-runs blind and most likely reproduces the same unsupported answer. Appended to the
        # retrieval task, never to `query`, which the short-circuit detectors above regex.
        _search_reground = _reground_note(state)
        if _search_reground:
            _retrieval_q = f"{_retrieval_q}\n\n{_search_reground}"
        # One session per turn: the peer thread is checkpointed under a stable child id, so
        # without this the harvest returns an earlier turn's documents as this turn's evidence.
        _session = open_peer_session(
            executor, agent_config(child_thread_id(state.get("thread_id"), "sup_search")))
        _run = _session.run(_retrieval_q)
        harvested = extract_documents_from_search_evidence(
            {"search_agent_tool_results": _run.artifacts.get("tool_results") or []})
        # The peer's OWN tool calls become rows the same way the analyze/code peers' do, so a
        # follow-up can be answered with "we already searched X and got N" instead of searching
        # again. Curated by the same allowlists; the retrieval renames in _LEDGER_SEARCH_TOOLS
        # stop `source`/`count` reading as imagery provenance.
        rows = _ledger_rows(_run.artifacts)
        # Completeness sweep: union in direct keyword+semantic hits so one search turn always
        # carries multi-method coverage, even when the LLM peer called a single tool.
        sweep = _direct_search_sweep(query, enabled_search_methods)
        if sweep:
            rows.append(_search_row("baseline_sweep", query, "keyword+semantic", len(sweep)))
        return {"documents": _merge_dedup(harvested, sweep), "action_rows": rows}

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
    steps.append({"step": "qgis_map_image", "result": render})

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


# --- did the interactive map actually get a layer? -----------------------------
# Observed: asked for "a heat map of these incidents on the map", the peer ran execute_code
# to build a GeoJSON, then heatmap_image to draw a PNG, and answered "you can view the heat
# map directly on the interactive map ... visualized as a heat layer". add_map_layer was
# never called, so the map got nothing; the grounding audit even logged "minor hallucination
# about an interactive map" and the claim shipped anyway. A static PNG cannot be panned,
# zoomed or clicked, so this is not a wording quibble — the deliverable was missing. Verified
# structurally (like the unrun-code check) rather than demanded in the prompt.
# The tools EXPECTED to deliver a map layer. Documentation and a test invariant
# (test_rs_embed_zonal asserts the zonal tools are in here) — deliberately NOT a delivery
# signal any more: a tool NAME says nothing about whether the call succeeded, and matching on it
# is what let a failed admin_boundary report a layer. Ask _map_delivered_this_turn instead.
# The map-delivery tools handed to a peer that has NO uploads — the drawn-region case, which is
# exactly when an embedding gets clustered or differenced in code and the result is pixels rather
# than geometry. Filtering to add_map_layer alone left that work with nowhere to put its raster.
_MAP_DELIVERY_TOOLS = ("add_map_layer", "add_raster_layer")

_MAP_LAYER_TOOLS = ("add_map_layer", "add_raster_layer", "overpass_search", "spatial_search",
                    "embed_region", "embed_zones",
                    "fit_zone_model", "admin_boundary")
_WANTS_MAP_RE = re.compile(
    r"\b(?:on|in|onto|to)\s+(?:the\s+|a\s+|my\s+)?(?:interactive\s+)?map\b"
    r"|\binteractive\s+map\b|\bmap\s+view\b|\bheat\s?map\b|\bchoropleth\b"
    r"|\bmap\s+(?:of|showing)\b",
    re.I,
)
_CLAIMS_MAP_RE = re.compile(
    r"\binteractive\s+map\b|\bon\s+the\s+map\b|\bmap\s+beside\b|\bheat\s+layer\b|\bmap\s+layer\b",
    re.I,
)
_MAP_NOT_DELIVERED_OBSERVATION = (
    "This turn has no add_map_layer record, so the user's interactive map received nothing. "
    "A PNG from heatmap_image / choropleth_image / render_map_image is a static picture: it "
    "cannot be panned, zoomed or clicked, and it is not what 'on the map' means — describing "
    "an image as a layer on their map would be false. Put the data on the map with "
    "add_map_layer(file_id=<the geodata file you produced>, render='heatmap'|'choropleth'|"
    "'points'|'shapes', column=<numeric column, for choropleth>, name=<short purpose name>), "
    "then say what is on it. Keep the PNG too if it is worth having."
)


# --- the answer must describe the artifact that was actually produced -----------
# Observed, live: "show me the clay embedding of urbana at 2025/03/01-2025/05/01" ran
# embed_region with NO `model`, so the default (gse) was embedded — and the answer then said
# "Here's the Clay v1.5 embedding … extracted from the global LGND Clay Embeddings – Sentinel-2
# collection … 2.56 km MajorTOM grid cell", provenance lifted wholesale from a web-search hit
# for the word "clay". The map legend beside it read "gse embedding (PCA-RGB)". The LLM
# grounding audit did not flag any of it, which is why these two checks are deterministic.
_EMBED_MODELS_CACHE: Optional[frozenset] = None


def _known_embedding_models() -> frozenset:
    """Model ids the embedding service offers, fetched once per process.

    Probed rather than hardcoded, for the same reason as everything else here: a list in the
    source silently stops matching the deployment. An unreachable service returns nothing,
    which disables the checks below rather than making them wrong.
    """
    global _EMBED_MODELS_CACHE
    if _EMBED_MODELS_CACHE is not None:
        return _EMBED_MODELS_CACHE
    ids: set = set()
    try:
        import requests

        from agent_runtime.rs_embed_tools import RS_EMBED_URL

        resp = requests.get(f"{RS_EMBED_URL}/api/models", timeout=10)
        if resp.status_code < 400:
            payload = resp.json()
            rows = payload.get("models") if isinstance(payload, dict) else payload
            for row in rows or []:
                name = row.get("id") if isinstance(row, dict) else row
                if isinstance(name, str) and name.strip():
                    ids.add(name.strip().lower())
    except Exception:  # noqa: BLE001 - a failed probe must not break a turn
        return frozenset()
    if ids:
        _EMBED_MODELS_CACHE = frozenset(ids)
    return frozenset(ids)


def _models_named_in(text: str) -> set:
    """Known model ids mentioned as whole words in *text*."""
    known = _known_embedding_models()
    if not known:
        return set()
    words = set(re.findall(r"[a-z][a-z0-9]*", str(text or "").lower()))
    return {m for m in known if m in words}


def _models_used(execution_context: Any) -> set:
    """Known model ids that a tool actually RAN with, read out of the artifacts."""
    known = _known_embedding_models()
    if not known:
        return set()
    try:
        blob = json.dumps(execution_context, default=str)
    except Exception:  # noqa: BLE001
        blob = str(execution_context)
    blob = blob.replace('\\"', '"')
    found = {m.lower() for m in re.findall(r'"model"\s*:\s*"([A-Za-z0-9_.-]+)"', blob)}
    return found & known


_MODEL_MISMATCH_OBSERVATION = (
    "The user named a specific embedding model ({wanted}) and the run used {used} instead — "
    "the embedding tools default their model argument, so leaving it out silently embeds "
    "with something else. Call the embedding tool again naming {wanted} explicitly: "
    "embed_region takes models=['{wanted}'] (a LIST, and it accepts several at once), while "
    "embed_zones takes model='{wanted}'. Then describe the model that actually ran."
)

# An answer that tells the user to add a layer they can already see. The map is delivered as
# an SSE event and rendered before the answer is read, so "paste this URL into the map" is not
# a harmless extra instruction — it tells the user the delivery failed when it did not.
_DENIES_MAP_RE = re.compile(
    r"\badd\s+(?:the\s+)?(?:layer|it|this|the\s+file)\b[^.\n]{0,40}\b(?:to|into)\b[^.\n]{0,30}\bmap\b"
    r"|\badd\s+layer\s*(?:→|->|:)\s*from\s+url"
    r"|\bpaste\s+the\s+(?:download\s+)?link\b"
    r"|\b(?:load|import|upload|drag)\s+(?:it|this|the\s+\w+)\s+(?:in)?to\b[^.\n]{0,30}\bmap\b",
    re.I,
)


def _called_tool(artifacts: Dict[str, Any], names) -> bool:
    """Whether this peer run called any of *names*."""
    wanted = {names} if isinstance(names, str) else set(names)
    for key in ("tool_calls", "tool_results"):
        for item in artifacts.get(key) or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("name") or item.get("tool_name") or "") in wanted:
                return True
    return False


# --- a tool that keeps failing is a dead end, not a reason to stop --------------
# Observed: regionalize(method='maxp') crashed with "'DataFrame' object has no attribute
# 'to_list'" — a raw exception naming nothing relevant. The peer retried the identical call,
# got the identical error, and then STOPPED to ask: "I recommend switching to a manual
# implementation ... Let me know if you'd like me to proceed." A whole turn spent, no result,
# and on an earlier run the same situation ended in an answer claiming the tool had succeeded.
# The peer already has everything needed to route around a broken tool; what it lacked was
# permission to, and the instruction to say so.
_TOOL_FAIL_REPEATS = 2


def _repeatedly_failed_tools(artifacts: Dict[str, Any]) -> Dict[str, str]:
    """``{tool_name: error}`` for tools that returned ok=false at least _TOOL_FAIL_REPEATS times."""
    counts: Dict[str, int] = {}
    seen: Dict[str, List[str]] = {}
    for item in artifacts.get("tool_results") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        body = str(item.get("content") or "")
        if not name or '"ok"' not in body:
            continue
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = None
        if isinstance(parsed, dict) and parsed.get("ok") is False:
            counts[name] = counts.get(name, 0) + 1
            seen.setdefault(name, []).append(str(parsed.get("error") or "")[:300])

    # The LATEST error, and a note when the failures were not the same one.
    #
    # This kept the FIRST error while only counting occurrences, so a peer that fixed
    # ModuleNotFoundError and then hit a KeyError was handed back the ModuleNotFoundError as
    # the thing failing repeatedly — and sent off to re-fix a problem it had already solved.
    # Two DIFFERENT errors are also the run/read/fix loop working rather than a dead end, so
    # the observation has to say which it is instead of flattening both into one string.
    out: Dict[str, str] = {}
    for name, count in counts.items():
        if count < _TOOL_FAIL_REPEATS:
            continue
        history = seen.get(name) or [""]
        latest = history[-1]
        distinct = list(dict.fromkeys(e for e in history if e))
        if len(distinct) > 1:
            latest = (f"{latest} (this tool failed {count} times with "
                      f"{len(distinct)} different errors; this is the most recent)")
        out[name] = latest
    return out


def _tool_stuck_observation(failures: Dict[str, str]) -> str:
    listed = "; ".join(f"{n}: {e or 'repeated failure'}" for n, e in failures.items())
    return (
        f"These tools failed repeatedly this turn with the same error — {listed}. Calling them "
        "again will not help: the fault is inside the tool, not in your arguments. Do the work "
        "another way instead of stopping. execute_code has geopandas, shapely, libpysal, esda, "
        "spreg and pygeoda available, so the computation is reachable directly; write it, then "
        "deliver the result with add_map_layer as usual. In your answer, state plainly which "
        "tool failed, quote its error, and say you computed the result in code instead — do not "
        "describe a failed tool as having worked. Proceed now; do not ask whether to."
    )

# ---------------------------------------------------------------------------
# The unified peer (experimental)
# ---------------------------------------------------------------------------
# Collapses SEARCH and ANALYZE into one agent with one context and one tool list, leaving the
# supervisor as a verifier plus the router for the code peer. Behind a flag so both shapes can
# be A/B'd against the same deployment rather than swapped blind.
#
# Why: the peers never ran concurrently (decide() returns one action per step), and every hard
# failure came from state split across them — the answer to "what resolution was that" sat in
# the analyse peer's thread while the router sent the follow-up to the SEARCH peer, whose own
# thread holds its own history and never saw it. (An earlier version of this comment said the
# search peer "starts fresh each turn". It does not: the checkpointer is passed on a stable
# child id. That claim is intermittently true only under multiple workers, because the
# checkpointer is process-global — which is how it came to be written down.) See AGENTS.md,
# "Why this workload is a poor fit for multiple agents".
UNIFIED_PEER_ENV = "AGENT_UNIFIED_PEER"

# Tools whose output is EVIDENCE. This allowlist is the whole reason the merge is safe:
# extract_documents_from_search_evidence is name-agnostic and harvests any tool result
# carrying "results"/"items"/"hits", so in a merged agent geocode_places({"results": [...]})
# and overpass_search would silently become retrieved "documents" the answer then cites.
_RETRIEVAL_TOOLS = frozenset({
    "agent_kb_search", "get_kb_block", "keyword_search", "semantic_search", "spatial_search",
    "opengeodata_search", "neo4j_search", "neo4j_explore_related_nodes",
    "neo4j_get_element_by_id", "web_search",
})


def _decision_sentence(nxt: str, why: str) -> Optional[str]:
    """A sentence for a supervisor decision a reader would otherwise misread, or None.

    None for the ordinary case — the decider simply chose, and saying "(decision)" adds a row
    without adding a fact. Everything else here is the loop declining to do the obvious thing,
    which is exactly when a reader needs to be told why rather than left to guess.
    """
    if why == "decision":
        return None
    if why == "max_steps":
        return "Stopping: this turn reached its step limit"
    if why == "search exhausted":
        return "Stopping: the knowledge base has nothing further to give"
    if why == "nothing has run yet":
        return f"Starting with {nxt}: nothing has run yet this turn"
    if why.startswith("no-progress repeat"):
        # The repeating action is named INSIDE the parens, and by this point `nxt` has already
        # been overwritten with "done" — reading it here says "done would repeat", which is
        # both wrong and confusing about what the loop declined to do.
        repeated = why.partition("(")[2].rstrip(")").strip() or "that step"
        return f"Stopping: {repeated} would repeat with nothing new to work from"
    if why.startswith("request by "):
        return f"Running {nxt}, asked for by {why[len('request by '):]}"
    return f"{nxt}: {why}"


def unified_peer_enabled(state: Optional[Dict[str, Any]] = None) -> bool:
    """Whether search and analyze run as ONE agent. Off by default.

    Per-REQUEST when the caller sets it, falling back to the env default — the same shape as
    code_peer. Without that, comparing the two architectures would mean restarting the
    deployment between arms, so the two shapes could never be exercised side by side.
    """
    if isinstance(state, dict):
        override = state.get("unified_peer")
        if override is not None:
            return bool(override)
    return (os.getenv(UNIFIED_PEER_ENV, "") or "").strip().lower() in {"1", "true", "yes", "on"}


def _dedup_tools(tools: List[Any]) -> List[Any]:
    """First tool wins per name.

    Naive concatenation of the two lists yields 102 entries with 14 duplicate NAMES in the
    deployed config. LangChain's ToolNode silently keeps the last of each, but bind_tools
    ships all 102 to the provider — so dedup here, where the choice is visible.
    """
    seen = set()
    out: List[Any] = []
    for tool in tools:
        name = getattr(tool, "name", None)
        if name in seen:
            continue
        if name:
            seen.add(name)
        out.append(tool)
    return out


def _evidence_from_artifacts(artifacts: Dict[str, Any]) -> List[Any]:
    """Documents the unified peer actually retrieved, by tool NAME.

    Deliberately not the name-agnostic harvester: see _RETRIEVAL_TOOLS.
    """
    from agent_runtime.supervisor.evidence_subgraph import extract_documents_from_search_evidence

    rows = [
        {"name": str(r.get("name") or ""), "content": r.get("content")}
        for r in (artifacts.get("tool_results") or [])
        if isinstance(r, dict) and str(r.get("name") or "") in _RETRIEVAL_TOOLS
    ]
    if not rows:
        return []
    try:
        return extract_documents_from_search_evidence({"search_agent_tool_results": rows}) or []
    except Exception:  # noqa: BLE001 - evidence is a bonus here, never the run
        return []


def default_analyze_fn(*, llm: Optional[Any] = None, include_mcp_tools: bool = True,
                       mcp_modules: Optional[List[str]] = None,
                       skill_roots: Optional[List[str]] = None,
                       code_exec: Optional[bool] = None,
                       input_file_ids: Optional[List[str]] = None,
                       enabled_search_methods: Optional[List[str]] = None) -> AnalyzeFn:
    """Run the GIS/data analysis workflow (QGIS + spatial-analysis MCP tools)."""

    def fn(query: str, evidence: List[Any], state: SupervisorState) -> Any:
        from agent_runtime.executor_factory import (
            agent_config,
            build_agent_executor,
            child_thread_id,
            invoke_agent_with_payload_fallback,
            open_peer_session,
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
        # Named US areas -> a boundary file, with NO upload. This MUST sit outside the
        # `if input_file_ids:` gate below. It was accidentally INSIDE it, so the analyse peer
        # had no admin_boundary whenever nothing was attached, and "show me the boundary of
        # Urbana city limits" fell through to geocode_places + embed_region — a rectangle — on
        # every model tried. Not needing an upload is the entire point of the tool.
        try:
            from agent_runtime.admin_boundary_tools import make_admin_boundary_tools
            tools.extend(make_admin_boundary_tools())
        except Exception:
            pass
        # `add_map_layer` is the ONLY geo tool that needs no uploaded file: it registers a layer
        # from geometry the peer already has. Bound unconditionally so "show me X on the map"
        # can be delivered with nothing attached — previously the peer had no way to deliver,
        # and the corrective retry was gated on the same flag, so the gap was silent.
        #
        # Only that one tool. The other five (inspect_vector, reproject_vector,
        # vector_spatial_join, vector_to_geojson, render_map_image) all need a vector file that
        # exists only on an upload turn, and render_map_image is the static-PNG route the map
        # observation exists to discourage: hoisting the whole factory costs ~1,479 tokens every
        # turn against ~299 for this.
        #
        # The corrective map retry stays gated on input_file_ids ON PURPOSE. Three separate
        # changes now make it fire more readily (turn-scoping, success-required delivery, this),
        # and _WANTS_MAP_RE matches a bare "on the map" — so un-gating it would have an ordinary
        # follow-up told the map received nothing and redundantly re-add a layer already on
        # screen. The tool is available; we simply do not nag.
        try:
            from agent_runtime.langchain_geo_tools import make_langchain_geo_tools

            if not input_file_ids:
                tools.extend(t for t in make_langchain_geo_tools(default_input_file_ids=None)
                             if str(getattr(t, "name", "")) in _MAP_DELIVERY_TOOLS)
        except Exception:
            pass
        # The conversation's own file listing, on EVERY turn — not gated on an upload like the
        # rest of the file toolset. Files get created without one (a boundary, an embedding, a
        # plot), and that is precisely when "what have you saved?" is asked. Ungated, the peer
        # answered it out of `execute_code` and listed the sandbox working directory. When an
        # upload IS present the full toolset below carries this tool already, so add it only in
        # the other case and no dedup is needed.
        if not input_file_ids:
            try:
                from agent_runtime.langchain_file_tools import make_conversation_file_tools

                tools.extend(make_conversation_file_tools())
            except Exception:  # noqa: BLE001 - one optional tool must not break the peer
                pass
        if input_file_ids:
            from agent_runtime.langchain_file_tools import make_langchain_file_tools

            tools.extend(make_langchain_file_tools())
            # Vector / shapefile tools (read + visualize + analyze uploaded TIGER files,
            # zip or extracted). Guarded so a missing geopandas never breaks the agent.
            try:
                from agent_runtime.langchain_geo_tools import make_langchain_geo_tools

                _geo = make_langchain_geo_tools(default_input_file_ids=input_file_ids)
                if _uploads_are_tabular_only(input_file_ids):
                    _geo = [t for t in _geo
                            if str(getattr(t, "name", "")) not in _VECTOR_FILE_TOOLS]
                tools.extend(_geo)
            except Exception:
                pass
            # Overlay / aggregation / temporal analysis tools. Same guard and the same
            # "files are attached" condition as the vector tools above: each factory is
            # imported separately so one missing optional dependency costs only its own
            # module instead of the whole analysis toolset.
            try:
                from agent_runtime.analysis_overlay_tools import make_overlay_tools
                tools.extend(make_overlay_tools(default_input_file_ids=input_file_ids))
            except Exception:
                pass
            try:
                from agent_runtime.analysis_aggregate_tools import make_aggregate_tools
                tools.extend(make_aggregate_tools(default_input_file_ids=input_file_ids))
            except Exception:
                pass
            try:
                from agent_runtime.analysis_temporal_tools import make_temporal_tools
                tools.extend(make_temporal_tools(default_input_file_ids=input_file_ids))
            except Exception:
                pass
            # Spatial statistics (libpysal/esda/spreg/pygeoda): weights, Moran/Geary, LISA,
            # Getis-Ord Gi*, spatial regression, GeoDa regionalization. Guarded like the rest —
            # a deployment without the PySAL stack loses these tools and nothing else.
            try:
                from agent_runtime.analysis_spatial_stats_tools import make_spatial_stats_tools
                tools.extend(make_spatial_stats_tools(default_input_file_ids=input_file_ids))
            except Exception:
                pass
        # Remote-sensing foundation-model embeddings (rs-embed service). NOT gated on
        # attached files: the region can come from the map's Region tool or a place name,
        # with nothing uploaded at all.
        try:
            from agent_runtime.rs_embed_tools import make_rs_embed_tools
            tools.extend(make_rs_embed_tools(default_input_file_ids=input_file_ids))
        except Exception:  # noqa: BLE001 - a broken toolset must not take the whole turn down
            # LOGGED, not swallowed. This guard exists for a missing optional dependency, but it
            # catches everything: a NameError from a bad edit to rs_embed_tools silently removes
            # all nine remote-sensing tools, and the turn then answers "I have no way to embed a
            # region" — indistinguishable, from the outside, from the service being down. One
            # such NameError reached a merge in this repo and only 32 unit tests caught it.
            logger.exception("remote-sensing toolset failed to build; those tools are UNAVAILABLE "
                             "this turn")
        # Elevation, on the same footing as the remote-sensing tools and for the same reason:
        # the region can arrive as a bbox from the map, a point, or a boundary this turn just
        # fetched, so gating it on an upload would hide it from every request that names a
        # place. Unlike those tools it costs nothing to run — USGS 3DEP takes no credential.
        try:
            from agent_runtime.terrain_tools import make_terrain_tools
            tools.extend(make_terrain_tools(default_input_file_ids=input_file_ids))
        except Exception:  # noqa: BLE001 - one optional toolset must not break the peer
            pass
        # Per-zone embeddings + the model fitted on them. These used to be gated on attached
        # files, because a polygon layer could only arrive by upload. admin_boundary can now
        # produce one from a place name mid-turn, so gating them here would hide the tool that
        # consumes it: the model would fetch Champaign County and have nothing to embed it with.
        try:
            from agent_runtime.rs_embed_tools import make_rs_embed_zonal_tools
            tools.extend(make_rs_embed_zonal_tools(default_input_file_ids=input_file_ids))
        except Exception:
            pass
        from agent_runtime.code_execution import is_code_exec_enabled

        if code_exec if code_exec is not None else is_code_exec_enabled():
            from agent_runtime.langchain_exec_tools import make_code_execution_tools

            tools.extend(make_code_execution_tools(
                default_input_file_ids=input_file_ids,
                session_id=child_thread_id(state.get("thread_id"), "codeexec")))
        if unified_peer_enabled(state):
            # One agent, one tool list: fold in the retrieval set the search peer used to own.
            try:
                from agent_runtime.langchain_granular_tools import make_langchain_granular_tools

                # The allowlist a request set with enabledSearchMethods was ignored here, so
                # in unified mode the merged peer got the FULL retrieval set regardless — the
                # one place the search node honours it (orchestration passes it there) and this
                # one did not.
                tools = _dedup_tools([*tools, *make_langchain_granular_tools(
                    include_file_tools=False, session_id=state.get("thread_id"),
                    enabled_search_methods=enabled_search_methods)])
            except Exception:  # noqa: BLE001 - never let the merge break the analyse peer
                pass
        executor = build_agent_executor(
            llm=llm, preloaded_tools=tools, system_prompt_override=ANALYSIS_WORKFLOW_PROMPT,
            agent_name="analysis_agent", skill_roots=skill_roots,
        )
        q = query
        if evidence:
            q = f"{query}\n\nContext evidence:\n{_format_documents(evidence)}"
        # The ledger has to reach the peer that CALLS TOOLS, not only the router and the
        # synthesizer. Measured: with the boundary already fetched and its file_id sitting in
        # the ledger, "now embed those zones" still went geocode_places -> embed_region (a
        # rectangle) on both gpt-4o and gpt-5.6-luna — because this peer had no idea the
        # polygon existed. It cannot call embed_zones(file_id=...) for a file it never heard of.
        _ledger_note = _prior_actions_note(_prior_actions(state))
        if _ledger_note:
            q = f"{q}\n\n{_ledger_note}"
        # A re-grounding pass has to say WHAT was ungrounded, or the peer re-runs blind and
        # most likely repeats the same unsupported answer. Appended to the task text like the
        # ledger note above, which is the channel this peer already reads.
        _reground = _reground_note(state)
        if _reground:
            q = f"{q}\n\n{_reground}"
        # Cross-turn continuity comes from this peer's own checkpointed child
        # thread (and the supervisor's chat_history drives routing/synthesis), so
        # we do NOT re-feed chat_history here — that would replay prior turns twice
        # on re-runs. Mirrors the search peer.
        # One session for the whole turn. Every invoke below reports only ITS OWN calls, while
        # _session.turn_artifacts accumulates the turn — the retries used to concatenate slices
        # that each already contained the previous invocation.
        _session = open_peer_session(executor,
                                    agent_config(child_thread_id(thread_id, "analysis")))
        _run = _session.run(q)
        resp = _run.resp
        artifacts = _run.artifacts
        result: Dict[str, Any] = {
            "summary": extract_final_answer(resp) or "",
            "tool_calls": artifacts.get("tool_calls") or [],
            "tool_results": artifacts.get("tool_results") or [],
        }
        caps = list(dict.fromkeys(r["capability"] for r in requests))

        # A tool that failed twice the same way is a dead end. Give the peer that fact and
        # explicit licence to route around it — once — rather than letting the turn end in
        # "shall I implement it by hand?" or, worse, a claim that the tool worked.
        stuck = _repeatedly_failed_tools(artifacts)
        if stuck and not caps:
            emit_trace_event(
                "tool_dead_end",
                {"stage": "analyze", "tools": sorted(stuck),
                 "message": f"{', '.join(sorted(stuck))} failed repeatedly; "
                            "handing the peer the observation and an alternative route"},
                node="analyze",
            )
            _alt_run = _session.run(_tool_stuck_observation(stuck))
            resp_alt, alt = _alt_run.resp, _alt_run.artifacts
            result["summary"] = extract_final_answer(resp_alt) or result["summary"]
            # The session owns the turn's total; hand-appending slices is what double-counted
            # invocation 1 on every retry.
            result["tool_calls"] = list(_session.turn_artifacts["tool_calls"])
            result["tool_results"] = list(_session.turn_artifacts["tool_results"])
            artifacts = {"tool_calls": result["tool_calls"], "tool_results": result["tool_results"]}
        # Carried downstream so synthesis cannot describe a failed tool as a success.
        if stuck:
            result["tool_failures"] = stuck

        # The map is a deliverable, not a figure of speech: if the user asked to see this on
        # the map (or the answer says it is there) and no layer-emitting tool ran, hand the
        # peer that observation once. A peer that asked for another capability is stopping
        # legitimately, so it is left alone. Only meaningful when geo tools were loaded.
        # Any tool that actually DELIVERS a layer counts — not just add_map_layer, or the
        # spatial toolkit's own layers (buffer_layer, aggregate_to_grid, …) would look
        # undelivered. Turn-scoped and success-required: this drives the corrective retry below,
        # which is a question about THIS turn, and a tool that returned ok=false delivered
        # nothing however promising its name.
        on_map = _map_delivered_this_turn(artifacts)
        wants_map = bool(_WANTS_MAP_RE.search(query or "")
                         or _CLAIMS_MAP_RE.search(result["summary"] or ""))
        if input_file_ids and wants_map and not on_map and not caps:
            emit_trace_event(
                "map_layer_not_delivered",
                {"stage": "analyze",
                 "message": "map requested but no add_map_layer record; retrying once"},
                node="analyze",
            )
            _retry_run = _session.run(_MAP_NOT_DELIVERED_OBSERVATION)
            resp_retry, retry_artifacts = _retry_run.resp, _retry_run.artifacts
            on_map = _map_delivered_this_turn(retry_artifacts)
            result["summary"] = extract_final_answer(resp_retry) or result["summary"]
            result["tool_calls"] = list(_session.turn_artifacts["tool_calls"])
            result["tool_results"] = list(_session.turn_artifacts["tool_results"])
        # The user named a model and a different one ran. Fixing it here rather than only
        # correcting the wording later is the point: the user asked for that model's
        # embedding, and a caveat on the wrong raster is not what they asked for.
        wanted = _models_named_in(query) - _models_used(result)
        if len(wanted) == 1 and _models_used(result) and not caps:
            used = ", ".join(sorted(_models_used(result)))
            asked = next(iter(wanted))
            emit_trace_event(
                "embedding_model_mismatch",
                {"stage": "analyze", "requested": asked, "used": used,
                 "message": f"asked for {asked}, ran {used}; retrying once"},
                node="analyze",
            )
            _retry_run = _session.run(
                _MODEL_MISMATCH_OBSERVATION.format(wanted=asked, used=used))
            resp_retry, retry_artifacts = _retry_run.resp, _retry_run.artifacts
            result["summary"] = extract_final_answer(resp_retry) or result["summary"]
            result["tool_calls"] = list(_session.turn_artifacts["tool_calls"])
            result["tool_results"] = list(_session.turn_artifacts["tool_results"])
            on_map = on_map or _map_delivered_this_turn(retry_artifacts)
        # Carried downstream so synthesis can describe the map honestly either way.
        result["on_map"] = bool(on_map)
        # execute_code is bound to THIS peer as well, and the router sends most code-shaped
        # work here — so the same honesty the code peer enforces has to hold here, or a peer
        # can hand back a code block it never ran and nothing says so.
        if _apply_execution_honesty(
                _session, result, prose_key="summary",
                exec_available=(code_exec if code_exec is not None
                                else is_code_exec_enabled()),
                caps=caps, node="analyze", any_fence=False):
            caps = list(dict.fromkeys(r["capability"] for r in requests))
        if caps:
            result["needs"] = caps  # model-driven request(s)
        return result

    return fn


_CODE_FENCE_RE = re.compile(r"^```[\w+-]*\s*$", re.M)
# A fence that DECLARES a programming language. `_CODE_FENCE_RE` matches any fence, including
# the untagged ones prose uses for a table of counts or a stdout excerpt — fine on the code
# peer, whose deliverable IS code, but a false positive on a peer whose deliverable is prose.
_CODE_LANG_FENCE_RE = re.compile(
    r"^```(?:python|py|r|sql|js|javascript|ts|typescript|bash|sh|shell|zsh|ruby|rb|java|"
    r"scala|julia|matlab|c|cpp|c\+\+|go|rust|perl|php|lua|swift|kotlin|dockerfile)\s*$",
    re.M | re.I)


def _has_execution_record(artifacts: Dict[str, Any]) -> bool:
    """Whether this peer run CALLED ``execute_code`` at all.

    Gates the did-you-actually-run-it retry, which is about a peer that handed back a code
    block without trying it. A call that ran and FAILED is not that case — telling such a peer
    "you did not run it" would be false.
    """
    return _called_tool(artifacts, "execute_code")


def _execution_outcome(artifacts: Dict[str, Any]) -> Tuple[bool, str]:
    """``(a run succeeded, the last error)`` across this turn's execute_code calls.

    ``executed`` is read downstream as "the code ran", so deriving it from the CALL meant a
    non-zero exit was reported as a success and synthesis described a failed run as a working
    one. The sandbox reports failure as data — ``ok`` is computed from exit_code/timeout/error
    — so the outcome has to be read out of the payload, not inferred from the call.
    """
    ran, error = False, ""
    for item in artifacts.get("tool_results") or []:
        if not isinstance(item, dict) or str(item.get("name") or "") != "execute_code":
            continue
        try:
            parsed = json.loads(str(item.get("content") or ""))
        except Exception:  # noqa: BLE001 - an unparseable result is not a successful one
            continue
        if not isinstance(parsed, dict):
            continue
        if parsed.get("ok") is True:
            ran = True
        else:
            error = str(parsed.get("error") or parsed.get("stderr") or "")[:300] or error
    return ran, error


def _apply_execution_honesty(session: Any, result: Dict[str, Any], *, prose_key: str,
                             exec_available: bool, caps: List[str], node: str,
                             any_fence: bool = True) -> bool:
    """Make a peer's result tell the truth about whether its code RAN. Returns: did we re-run.

    This lives here, and takes the peer as a parameter, because the invariant is about the
    TOOL: any peer with execute_code bound can hand back a code block it never ran, and
    ``executed`` has to mean a run wherever that happens. It was implemented on the code peer
    only — and the router sends this work to the ANALYZE peer, which has execute_code bound
    too, so the guard was missing exactly where it was needed. Observed live: a turn returned
    a Socrata loader as "the code you actually ran" with no execute_code record anywhere in it.

    A capability request no longer suppresses the challenge, only changes it. Being blocked is
    a reason a peer cannot RUN the code; it is not a reason to present unrun code as executed,
    and the two were conflated — a live turn requested a capability and still returned a
    network loader as "the code you actually ran". The blocked variant asks it to run what it
    can and otherwise label the code UNRUN, keeping the request either way.
    """
    from agent_runtime.runtime_utils import extract_final_answer

    turn = {"tool_calls": result.get("tool_calls") or [],
            "tool_results": result.get("tool_results") or []}
    reran = False
    if (exec_available
            and not _has_execution_record(turn)
            and _ships_unrun_code(result.get(prose_key) or "", any_fence=any_fence)):
        blocked = bool(caps)
        emit_trace_event(
            "code_not_executed",
            {"stage": node, "blocked_on_capability": blocked,
             "message": "code returned without an execute_code record; retrying once"},
            node=node,
        )
        _retry = session.run(_CODE_NOT_RUN_BLOCKED_OBSERVATION if blocked
                             else _CODE_NOT_RUN_OBSERVATION)
        result[prose_key] = extract_final_answer(_retry.resp) or result.get(prose_key)
        result["tool_calls"] = list(session.turn_artifacts["tool_calls"])
        result["tool_results"] = list(session.turn_artifacts["tool_results"])
        turn = {"tool_calls": result["tool_calls"], "tool_results": result["tool_results"]}
        reran = True

    _record_execution_outcome(result, turn)
    return reran


def _record_execution_outcome(result: Dict[str, Any], turn: Dict[str, Any]) -> bool:
    """Write ``executed`` and ``execution_error`` from what the turn's tool results show.

    The pair must be written TOGETHER, and the CLEAR is the half that gets forgotten. This
    logic was written out twice — once here and once in the code peer's dead-end branch — and
    only one copy removed a stale error, so a second run that succeeded shipped beside the
    failure it had just fixed and the grounding auditor read a working run as a broken one.
    One function, so the two cannot drift apart again.

    Returns whether a run is recorded.
    """
    ran, error = _execution_outcome(turn)
    result["executed"] = bool(ran)
    # Only when a run actually failed: an absent run is already said by executed=False, and a
    # stale error beside a later success would read as a failure that did not happen.
    if error and not ran:
        result["execution_error"] = error
    elif "execution_error" in result:
        result.pop("execution_error")
    return bool(ran)


def _ships_unrun_code(answer: str, *, any_fence: bool = True) -> bool:
    """Whether an answer hands back a code block as its result.

    ``any_fence`` is right for the code peer: its deliverable IS code, so a fence is a fair
    proxy whether or not it names a language.

    It is wrong for a peer whose deliverable is prose plus map layers. An analyze summary
    routinely fences a table of counts or a stdout excerpt with a bare ```, and treating that
    as shipped code spent a whole extra model run challenging the peer about code it had never
    written — then, if the peer stood by its answer, told the reader it had been challenged.
    So there, require the fence to declare a language.
    """
    text = str(answer or "")
    if any_fence:
        return bool(_CODE_FENCE_RE.search(text))
    return bool(_CODE_LANG_FENCE_RE.search(text))


# The prompt used to carry this as a threat ("an answer that only pastes code … is a
# FAILURE") with nothing checking it, which is the shape most likely to backfire: a model
# told that not running code is a failure will claim it ran when the sandbox dies. Verify
# instead, and hand the peer the observation so it can act on it.
_CODE_NOT_RUN_OBSERVATION = (
    "Your previous reply returned code, but this turn has no execute_code record — so the "
    "code was never run, its output is unverified, and any files it would have written do "
    "not exist for the user. Run it with execute_code, read stdout/stderr, fix what the "
    "sandbox reports and re-run until it works; then report the result. If it genuinely "
    "cannot be run here, say so and why."
)

# The same challenge for a peer that has asked for another capability. It may genuinely be
# unable to run the code yet, so demanding a run would be the wrong instruction — but shipping
# code that READS as executed is not made acceptable by being blocked, which is the whole
# point of challenging it here.
_CODE_NOT_RUN_BLOCKED_OBSERVATION = (
    "Your previous reply returned code, but this turn has no execute_code record, and you have "
    "asked for another capability — so the code was NOT run, its output is unverified, and any "
    "files it would have written do not exist for the user. If you can run it now with what you "
    "already have, do that and report the real output. If you genuinely cannot until the "
    "capability arrives, keep the request and say plainly that the code is UNRUN and its "
    "results are not yet established — do not present it, or any numbers, as though it had "
    "executed."
)


def default_code_fn(*, llm: Optional[Any] = None, skill_roots: Optional[List[str]] = None,
                    code_exec: Optional[bool] = None,
                    input_file_ids: Optional[List[str]] = None,
                    code_peer: Optional[str] = None,
                    code_peer_model: Optional[str] = None) -> CodeFn:
    """Code peer: writes code, and can request_capability(search/analyze) when it
    lacks the context to do so (model-driven — no nested search tool)."""

    def fn(query: str, evidence: List[Any], state: "SupervisorState") -> Any:
        # AGENT_CODE_PEER swaps the whole peer for a sandboxed agentic CLI, which
        # iterates internally — no request_capability, no nested tools. Two are
        # wired: `opencode` (OpenAI-compatible endpoint) and `claude` (Anthropic).
        # A per-request `code_peer` overrides the env default; anything else
        # (including "langchain") means the built-in peer below.
        import os as _os

        from agent_runtime.opencode_peer import CODE_PEER_ENV, selects_opencode
        from agent_runtime.claude_peer import selects_claude

        choice = code_peer if code_peer else _os.getenv(CODE_PEER_ENV)
        if selects_opencode(choice):
            from agent_runtime.opencode_peer import run_opencode_code_peer

            return run_opencode_code_peer(
                query, evidence=evidence, state=state, input_file_ids=input_file_ids,
            )
        if selects_claude(choice):
            from agent_runtime.claude_peer import run_claude_code_peer

            return run_claude_code_peer(
                query, evidence=evidence, state=state, input_file_ids=input_file_ids,
                model=code_peer_model,
            )
        from agent_runtime.executor_factory import (
            agent_config,
            build_agent_executor,
            child_thread_id,
            invoke_agent_with_payload_fallback,
            open_peer_session,
        )
        from agent_runtime.runtime_utils import extract_final_answer, extract_search_artifacts
        from agent_runtime.skills import make_skill_tools

        request_tool, requests = _make_request_tool()
        tools = [*make_skill_tools(skill_roots=skill_roots), request_tool]
        # KB read tools so the code peer can pull the FULL source of referenced blocks
        # (get_kb_block) and reuse it verbatim instead of stubbing loaders.
        #
        # web_search/web_fetch join them because the KB covers I-GUIDE's OWN content and not
        # library documentation, and the failure this peer actually has is plausible code
        # against a misremembered API — a keyword argument that moved, a function that returns
        # a tuple now. It had no way to check: the sandbox has no network, so a lookup has to
        # happen agent-side, before the code runs. Asking the factory for web_search also
        # yields web_fetch by its own rule (finding a page and being unable to read it is not
        # a capability). Four names out of the family's twenty-two.
        try:
            from agent_runtime.langchain_granular_tools import make_langchain_granular_tools
            tools.extend(t for t in make_langchain_granular_tools(
                enabled_search_methods=["agent_kb_search", "get_kb_block", "web_search"])
                if getattr(t, "name", "") in {"agent_kb_search", "get_kb_block",
                                              "web_search", "web_fetch"})
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
        # admin_boundary sits OUTSIDE that gate: it exists to produce a boundary when the user
        # attached nothing, so gating it on an attachment would defeat it.
        try:
            from agent_runtime.admin_boundary_tools import make_admin_boundary_tools
            tools.extend(make_admin_boundary_tools())
        except Exception:
            pass
        # `add_map_layer` is the ONLY geo tool that needs no uploaded file: it registers a layer
        # from geometry the peer already has. Bound unconditionally so "show me X on the map"
        # can be delivered with nothing attached — previously the peer had no way to deliver,
        # and the corrective retry was gated on the same flag, so the gap was silent.
        #
        # Only that one tool. The other five (inspect_vector, reproject_vector,
        # vector_spatial_join, vector_to_geojson, render_map_image) all need a vector file that
        # exists only on an upload turn, and render_map_image is the static-PNG route the map
        # observation exists to discourage: hoisting the whole factory costs ~1,479 tokens every
        # turn against ~299 for this.
        #
        # The corrective map retry stays gated on input_file_ids ON PURPOSE. Three separate
        # changes now make it fire more readily (turn-scoping, success-required delivery, this),
        # and _WANTS_MAP_RE matches a bare "on the map" — so un-gating it would have an ordinary
        # follow-up told the map received nothing and redundantly re-add a layer already on
        # screen. The tool is available; we simply do not nag.
        try:
            from agent_runtime.langchain_geo_tools import make_langchain_geo_tools

            if not input_file_ids:
                tools.extend(t for t in make_langchain_geo_tools(default_input_file_ids=None)
                             if str(getattr(t, "name", "")) in _MAP_DELIVERY_TOOLS)
        except Exception:
            pass
        # The conversation's own file listing, on EVERY turn — not gated on an upload like the
        # rest of the file toolset. Files get created without one (a boundary, an embedding, a
        # plot), and that is precisely when "what have you saved?" is asked. Ungated, the peer
        # answered it out of `execute_code` and listed the sandbox working directory. When an
        # This peer never attaches that toolset at all, so there is nothing to gate against.
        try:
            from agent_runtime.langchain_file_tools import make_conversation_file_tools

            tools.extend(make_conversation_file_tools())
        except Exception:  # noqa: BLE001 - one optional tool must not break the peer
            pass
        if input_file_ids:
            try:
                from agent_runtime.langchain_geo_tools import make_langchain_geo_tools

                _geo = make_langchain_geo_tools(default_input_file_ids=input_file_ids)
                if _uploads_are_tabular_only(input_file_ids):
                    _geo = [t for t in _geo
                            if str(getattr(t, "name", "")) not in _VECTOR_FILE_TOOLS]
                tools.extend(_geo)
            except Exception:
                pass
            # Overlay / aggregation / temporal tools, same as the analysis peer: the code
            # peer should reach for a ready-made clip/buffer/time-series tool before
            # writing the same thing by hand in the sandbox. Independently guarded.
            try:
                from agent_runtime.analysis_overlay_tools import make_overlay_tools
                tools.extend(make_overlay_tools(default_input_file_ids=input_file_ids))
            except Exception:
                pass
            try:
                from agent_runtime.analysis_aggregate_tools import make_aggregate_tools
                tools.extend(make_aggregate_tools(default_input_file_ids=input_file_ids))
            except Exception:
                pass
            try:
                from agent_runtime.analysis_temporal_tools import make_temporal_tools
                tools.extend(make_temporal_tools(default_input_file_ids=input_file_ids))
            except Exception:
                pass
            # Spatial statistics (libpysal/esda/spreg/pygeoda): weights, Moran/Geary, LISA,
            # Getis-Ord Gi*, spatial regression, GeoDa regionalization. Guarded like the rest —
            # a deployment without the PySAL stack loses these tools and nothing else.
            try:
                from agent_runtime.analysis_spatial_stats_tools import make_spatial_stats_tools
                tools.extend(make_spatial_stats_tools(default_input_file_ids=input_file_ids))
            except Exception:
                pass
        # Remote-sensing foundation-model embeddings (rs-embed service). NOT gated on
        # attached files: the region can come from the map's Region tool or a place name,
        # with nothing uploaded at all.
        try:
            from agent_runtime.rs_embed_tools import make_rs_embed_tools
            tools.extend(make_rs_embed_tools(default_input_file_ids=input_file_ids))
        except Exception:  # noqa: BLE001 - a broken toolset must not take the whole turn down
            # LOGGED, not swallowed. This guard exists for a missing optional dependency, but it
            # catches everything: a NameError from a bad edit to rs_embed_tools silently removes
            # all nine remote-sensing tools, and the turn then answers "I have no way to embed a
            # region" — indistinguishable, from the outside, from the service being down. One
            # such NameError reached a merge in this repo and only 32 unit tests caught it.
            logger.exception("remote-sensing toolset failed to build; those tools are UNAVAILABLE "
                             "this turn")
        # Elevation, on the same footing as the remote-sensing tools and for the same reason:
        # the region can arrive as a bbox from the map, a point, or a boundary this turn just
        # fetched, so gating it on an upload would hide it from every request that names a
        # place. Unlike those tools it costs nothing to run — USGS 3DEP takes no credential.
        try:
            from agent_runtime.terrain_tools import make_terrain_tools
            tools.extend(make_terrain_tools(default_input_file_ids=input_file_ids))
        except Exception:  # noqa: BLE001 - one optional toolset must not break the peer
            pass
        # Per-zone embeddings + the model fitted on them. These used to be gated on attached
        # files, because a polygon layer could only arrive by upload. admin_boundary can now
        # produce one from a place name mid-turn, so gating them here would hide the tool that
        # consumes it: the model would fetch Champaign County and have nothing to embed it with.
        try:
            from agent_runtime.rs_embed_tools import make_rs_embed_zonal_tools
            tools.extend(make_rs_embed_zonal_tools(default_input_file_ids=input_file_ids))
        except Exception:
            pass
        from agent_runtime.code_execution import is_code_exec_enabled

        if code_exec if code_exec is not None else is_code_exec_enabled():
            from agent_runtime.langchain_exec_tools import make_code_execution_tools

            tools.extend(make_code_execution_tools(
                default_input_file_ids=input_file_ids,
                session_id=child_thread_id(state.get("thread_id"), "codeexec")))
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
        # What earlier runs in this conversation left on disk. Without this the peer
        # rebuilds work it already has — especially after the user steers ("now a heatmap
        # of that"), where "that" is a file the previous step wrote.
        try:
            from agent_runtime.code_execution import session_workspace_listing
            existing = session_workspace_listing(child_thread_id(state.get("thread_id"), "codeexec"))
            if existing:
                parts.append(
                    "Already in this conversation's working directory (open them directly in "
                    "execute_code; no need to rebuild). A .py here is a program you can re-run "
                    "with execute_code(entrypoint=...) and change with edit_workspace_file — "
                    "read it first, and do not re-send a whole script to alter part of it:\n"
                    + "\n".join(f"- {f['name']} ({f['size_bytes']} bytes)" for f in existing))
        except Exception:
            pass
        # The workspace listing above says what FILES exist; the ledger says what was DONE
        # and with what — the county already fetched, the model and dates already used. The
        # code peer re-derives both without it.
        _code_note = _prior_actions_note(_prior_actions(state))
        if _code_note:
            parts.append(_code_note)
        # Same reason as the analyze peer: a re-grounding pass that does not say WHAT was
        # ungrounded makes the peer re-run blind and most likely repeat the same unsupported
        # answer. This node was the only one of the three that never received it.
        _code_reground = _reground_note(state)
        if _code_reground:
            parts.append(_code_reground)
        # See analyze peer: continuity is owned by this peer's checkpointed thread,
        # so chat_history is not re-fed here (avoids double-replay on re-runs).
        _session = open_peer_session(
            executor, agent_config(child_thread_id(state.get("thread_id"), "code")))
        _run = _session.run("\n\n".join(parts))
        resp = _run.resp
        # Flat result: the human-readable answer + a compact artifacts extract.
        # Do NOT nest the whole raw response object (it would crowd out / truncate
        # the real code+output when synthesis serializes code_result).
        artifacts = _run.artifacts
        result: Dict[str, Any] = {
            "answer": _run.answer,
            "tool_calls": artifacts.get("tool_calls") or [],
            "tool_results": artifacts.get("tool_results") or [],
        }
        caps = list(dict.fromkeys(r["capability"] for r in requests))

        # Verify the peer actually ran what it wrote. A peer that asked for another
        # capability is stopping legitimately, so it is left alone; otherwise give it one
        # chance to run the code, with the observation that it did not.
        exec_available = code_exec if code_exec is not None else is_code_exec_enabled()
        extra_run = _apply_execution_honesty(
            _session, result, prose_key="answer", exec_available=exec_available,
            caps=caps, node="code")
        if extra_run:
            caps = list(dict.fromkeys(r["capability"] for r in requests))

        turn = {"tool_calls": result["tool_calls"], "tool_results": result["tool_results"]}
        ran = bool(result.get("executed"))

        # A tool that failed the same way twice is a dead end. The analyze peer has had this
        # for a while; the code peer never did, even though execute_code's payload carries the
        # very `ok` key the detector reads — so two identical sandbox failures produced no
        # intervention at all. At most ONE extra run per turn, so a peer that already got the
        # did-you-run-it observation is left alone rather than paying for both.
        if not extra_run and not caps and not ran:
            stuck = _repeatedly_failed_tools(turn)
            if stuck:
                emit_trace_event(
                    "tool_dead_end",
                    {"stage": "code", "tools": sorted(stuck),
                     "message": f"{', '.join(sorted(stuck))} failed repeatedly; "
                                "handing the peer the observation and an alternative route"},
                    node="code",
                )
                _alt_run = _session.run(_tool_stuck_observation(stuck))
                result["answer"] = extract_final_answer(_alt_run.resp) or result["answer"]
                result["tool_calls"] = list(_session.turn_artifacts["tool_calls"])
                result["tool_results"] = list(_session.turn_artifacts["tool_results"])
                turn = {"tool_calls": result["tool_calls"], "tool_results": result["tool_results"]}
                ran = _record_execution_outcome(result, turn)
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


def _execution_note(result: Any) -> str:
    """One line stating whether the code in a peer result actually RAN.

    The flag exists so synthesis cannot describe a failed run as a working one, and it was left
    to SURVIVE a serialization rather than being stated. It did not. In analysis_results it sits
    after tool_calls/tool_results — the two biggest fields — and `json.dumps(...)[:2000]` cut it
    off. For the code peer it was never serialized at all: that branch prefers the peer's
    `answer` text and never dumps the dict. So the field that justified the whole
    execution-honesty series reached the answering model through neither path.

    Stated, not smuggled.
    """
    if not isinstance(result, dict) or "executed" not in result:
        return ""
    if result.get("executed"):
        return "EXECUTION: the code in this result RAN and its output is real."
    error = str(result.get("execution_error") or "").strip()
    tail = f" The failure was: {error[:200]}" if error else ""
    return ("EXECUTION: the code in this result did NOT run." + tail
            + " Do not present it as executed, and do not describe its output as a result.")


def default_synthesize_fn(llm: Optional[Any] = None) -> SynthesizeFn:
    """Compose the final grounded answer in the original AnalysisAgent format."""

    def fn(query: str, evidence: List[Any], analysis_results: Any, code_result: Any,
           chat_history: Optional[List[Any]] = None,
           prior_actions_note: Optional[str] = None) -> str:
        from agent_runtime.supervisor.prompts import SYNTHESIS_PROMPT

        active = llm
        if active is None:
            from agent_runtime.executor_factory import build_default_llm

            active = build_default_llm()
        parts = [SYNTHESIS_PROMPT]
        history = _format_chat_history(chat_history)
        if history:
            parts.append(f"Conversation so far:\n{history}")
        # Its OWN section, not an item inside chat_history. Smuggled through the history it was
        # item 0 of a list rendered as `[-8:]` and then tail-truncated at 4000 chars — both trims
        # cut exactly where it sat, so it vanished at 8+ history items while the auditor still
        # received it, and the two consumers that must agree systematically disagreed.
        if prior_actions_note:
            parts.append(prior_actions_note)
        parts.append(f"Question:\n{query}")
        parts.append(f"Evidence:\n{_format_documents(evidence)}")
        if analysis_results:
            parts.append(f"Analysis results:\n{json.dumps(analysis_results, ensure_ascii=True, default=str)[:2000]}")
            _note = _execution_note(analysis_results)
            if _note:
                parts.append(_note)
        if code_result:
            # Prefer the code peer's human-readable answer; only fall back to a
            # serialized dump if no answer text is present (keeps the real code /
            # output from being truncated away by a large nested object).
            if isinstance(code_result, dict) and str(code_result.get("answer") or "").strip():
                parts.append(f"Code result:\n{str(code_result['answer'])[:2000]}")
            else:
                parts.append(f"Code result:\n{json.dumps(code_result, ensure_ascii=True, default=str)[:2000]}")
            _note = _execution_note(code_result)
            if _note:
                parts.append(_note)
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
            _decision_payload = _distill(state, for_decision=True)
            nxt = decide(state, _decision_payload)
            _LEDGER_LOG.info(
                "supervisor step=%s prior_actions=%d -> %s",
                state.get("step"),
                len(_decision_payload.get("prior_turns_in_this_conversation") or []),
                nxt)
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
        # A VETO, not a hint: _available_actions only shapes the menu the decider is shown,
        # and it answered `done` at step 0 anyway. Measured on the merged shape — "Find flood
        # risk datasets on I-GUIDE" finished without retrieving and replied "I couldn't find
        # any supporting material". Deleting the search peer also deleted the decider's cue
        # that retrieval was the opening move, so the floor has to be enforced here.
        if (nxt == "done" and unified_peer_enabled(state)
                and not (state.get("actions") or [])
                and not (state.get("evidence") or [])
                and state.get("analysis_results") is None
                and state.get("code_result") is None):
            nxt, why = "analyze", "nothing has run yet"
        # `supervisor -> analyze (decision)` — an arrow, two internal node names, and a
        # parenthetical whose commonest value means "no special reason". But the OTHER five
        # values of `why` are the most informative thing in the whole trace: they say why the
        # loop did something a reader would otherwise call a bug (stopped early, stopped
        # without searching, ran analysis on a request that named no analysis). Those get a
        # sentence AND their own event, so folding the routing ladder cannot hide them.
        emit_trace_event(
            "node_completed",
            {"stage": "supervisor", "route": nxt, "message": f"Next: {nxt}"},
            node="supervisor",
        )
        _reason = _decision_sentence(nxt, why)
        if _reason:
            emit_trace_event(
                "decision",
                {"kind": "supervisor_decision", "route": nxt, "why": why,
                 "message": _reason},
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
        try:
            raw = do_search(q, state) or []
        except Exception as exc:  # noqa: BLE001 - a dead peer must not be a dead turn
            # The third node with the same gap. Search is usually the FIRST peer to run, so a
            # raise here loses the turn before any other peer has contributed anything.
            logger.exception("search peer failed; continuing the turn without it")
            emit_trace_event(
                "node_failed",
                {"stage": "search", "message": f"search peer failed: {type(exc).__name__}"},
                node="search",
            )
            # Only DECLARED state keys: SupervisorState has no search_error channel, and an
            # unknown key would make this handler raise the very error it exists to absorb.
            # The attempt is counted and the streak advanced, so the exhaustion logic stops
            # routing to a peer that keeps dying instead of looping on it.
            return {
                "evidence": state.get("evidence") or [],
                "search_attempts": state.get("search_attempts", 0) + 1,
                "search_empty_streak": state.get("search_empty_streak", 0) + 1,
            }
        # A search_fn may return a plain list (every test double does, and so may a custom one)
        # or a dict carrying documents + the ledger rows for what it actually did. Both stay
        # supported; only the dict form contributes rows.
        new_rows: List[Dict[str, Any]] = []
        if isinstance(raw, dict):
            docs = raw.get("documents") or []
            _, needs = _extract_needs(raw)
            new_rows.extend(raw.get("action_rows") or [])
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
                    new_rows.extend(more.get("action_rows") or [])
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
                # Direct backend calls, so no artifact to extract — but leaving the open web out
                # of the ledger is how a follow-up gets told the information is unavailable
                # right after the agent went and found it.
                new_rows.append(_search_row("web_fallback", q, "open web", len(web_docs)))
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
            "evidence_summary": _summarize_evidence(llm, q, merged) or state.get("evidence_summary"),
            "search_attempts": state.get("search_attempts", 0) + 1,
            "search_empty_streak": 0 if added > 0 else prev_streak + 1,
            "searched_queries": tried,
            # Accumulate: SupervisorState has no reducers, so a plain overwrite would lose the
            # rows from an earlier search step in the same turn. Rows, not raw artifacts — the
            # search tool_results ARE the whole document payload, and this state is both
            # checkpointed and shipped to the client.
            "action_rows": [*(state.get("action_rows") or []), *new_rows],
        }
        enq = _enqueue_needs(state.get("needs"), needs, "search")
        if enq is not None:
            update["needs"] = enq
        return update

    def analysis_node(state: SupervisorState) -> Dict[str, Any]:
        q = state.get("query", "")
        emit_trace_event("node_started", {"stage": "analyze", "message": "Running analysis workflow"}, node="analyze")
        try:
            clean, needs = _extract_needs(do_analyze(q, state.get("evidence") or [], state))
        except Exception as exc:  # noqa: BLE001 - a dead peer must not be a dead turn
            # Same reason code_node has one, and more pressing: this is the peer the router
            # actually sends code-shaped work to (measured: 7 of 7 turns reported
            # peer=analysis), and it invokes the model up to five times per turn — the initial
            # run plus the stuck, map-not-delivered, model-mismatch and execution-honesty
            # retries. Any of them can hit the recursion limit, and without a handler that
            # raised straight out of the graph: SSE error, no synthesized answer, even when
            # search had already found something worth saying.
            logger.exception("analyze peer failed; continuing the turn without it")
            emit_trace_event(
                "node_failed",
                {"stage": "analyze", "message": f"analyze peer failed: {type(exc).__name__}"},
                node="analyze",
            )
            return {"analysis_results": {"summary": "", "tool_calls": [], "tool_results": [],
                                         "error": f"{type(exc).__name__}: {exc}"[:300]}}
        emit_trace_event("node_completed", {"stage": "analyze", "message": "Analysis workflow complete"}, node="analyze")
        update: Dict[str, Any] = {"analysis_results": clean}
        if unified_peer_enabled(state) and isinstance(clean, dict):
            # The state KEYS stay exactly as they were. evidence_quality.py and
            # runtime_utils.py read "analysis_results"/"evidence" by name, and a rename there
            # degrades silently rather than raising — so the merge changes who FILLS these,
            # never what they are called.
            docs = _evidence_from_artifacts(clean)
            if docs:
                merged = _merge_dedup(state.get("evidence") or [], docs)
                update["evidence"] = merged
                update["search_attempts"] = state.get("search_attempts", 0) + 1
                update["searched_queries"] = [*(state.get("searched_queries") or []), q]
                update["search_empty_streak"] = 0
        enq = _enqueue_needs(state.get("needs"), needs, "analyze")
        if enq is not None:
            update["needs"] = enq
        return update

    def code_node(state: SupervisorState) -> Dict[str, Any]:
        q = state.get("query", "")
        emit_trace_event("node_started", {"stage": "code", "message": "Generating code"}, node="code")
        try:
            clean, needs = _extract_needs(do_code(q, state.get("evidence") or [], state))
        except Exception as exc:  # noqa: BLE001 - a dead peer must not be a dead turn
            # This node had no handler, so a peer that hit the recursion limit raised straight
            # out of the supervisor graph and the whole turn died: the user got an SSE error
            # and no synthesized answer, even when search and analyze had already produced
            # something worth saying. Degrade to a code_result that states the failure and let
            # synthesis answer from what the other peers found.
            logger.exception("code peer failed; continuing the turn without it")
            emit_trace_event(
                "node_failed",
                {"stage": "code", "message": f"code peer failed: {type(exc).__name__}"},
                node="code",
            )
            return {"code_result": {"answer": "", "executed": False, "tool_calls": [],
                                    "tool_results": [],
                                    "error": f"{type(exc).__name__}: {exc}"[:300]}}
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
        # Hoisted above both branches: _correct_artifact_claims runs on the insufficiency path
        # too, and it needs the ledger to know a layer from an earlier turn is still on screen.
        _rows = _prior_actions(state)
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
            # The answering model needs the ledger too, not just the router: the router only
            # decides whether to search, while THIS is what decides whether the answer is right.
            _history = list(state.get("chat_history") or [])
            # THE single rendering, and the auditor gets the visible-state summary too: "the
            # layer is on your map" is precisely the claim it used to flag as unsupported,
            # because nothing in its evidence said a layer from an earlier turn still exists.
            _ledger_text = "\n".join([*_ledger_lines(_rows), *_visible_state_lines(_rows)])
            # What the CLIENT does, stated only when a layer really is on it — see
            # _map_environment_lines. It travels as its own exec_ctx key rather than inside
            # prior_actions, whose heading says "from EARLIER TURNS": an environment fact filed
            # under that heading would read as a stale artifact of some previous turn. The
            # answerer's note deliberately does NOT get this line — SYNTHESIS_PROMPT already
            # covers the map, and the gap being closed here is the auditor's alone.
            _map_env = _map_environment_lines(_map_layer_was_delivered(
                {"analysis_results": ar, "code_result": cr, "artifacts": artifacts}, _rows))
            _note = _prior_actions_note(_rows)
            answer = do_synthesize(q, evidence, ar, cr, _history, _note)
            # The auditor must be given the SAME earlier-turn tool records the answerer was
            # told to answer from. Without them a correct cross-turn answer ("the gse run used
            # 64 dims at 7.645 m/px", read off turn 2's ledger) is audited against this turn's
            # execution only and flagged as unsupported.
            # This turn's rows, rendered by the SAME function that renders earlier turns.
            # _record_actions extracts these again on the way out (line ~4405) rather than
            # taking them from here: extraction is a pure walk over dicts already in memory,
            # and threading a value through the recording path to save it would couple the
            # audit to the ledger write for no measurable gain.
            _turn_rows = [*_ledger_rows(ar, cr), *(state.get("action_rows") or [])]
            exec_ctx = {"analysis_results": ar, "code_result": cr, "artifacts": artifacts,
                        "this_turn": _ledger_lines(_turn_rows),
                        "prior_actions": _ledger_text, "environment": _map_env}
            # Audit only when there's actual retrieval/execution grounding to check against.
            # A purely conversational answer (composed from chat_history with no evidence or
            # artifacts) has nothing for the grounding auditor to compare to and would be
            # false-flagged against empty evidence — skip the audit for it.
            # Artifacts + tool outputs are first-class grounding: pass the execution record so
            # a genuinely-produced map/file/count is not flagged as hallucination.
            audit = audit_answer_grounding(
                q, answer, evidence, llm=llm, execution_context=exec_ctx,
            ) if (do_audit and (answer or "").strip() and has_grounding) else {}
            # Deterministic reconciliation: produced artifacts + the execution record are
            # ground truth, so the LLM auditor can't false-flag a genuinely-generated
            # map/file or a number/method it actually computed.
            audit = _reconcile_audit_with_artifacts(audit, artifacts, execution_context=exec_ctx,
                                                    prior_rows=_rows)
            # THE GATE. A surviving flag means the audit still cannot find these claims in the
            # record after all four deterministic drops — the shape of "answered from memory".
            # Send the work back once instead of shipping a caveat over unfinished work.
            _reground_to = _reground_target(state) if _audit_flagged(audit) else None
            if _reground_to:
                gaps = _unsupported_claims(audit)
                if gaps:
                    emit_trace_event(
                        "node_completed",
                        {"stage": "synthesize",
                         "message": f"answer not grounded — re-running {_reground_to} to "
                                    f"establish: {'; '.join(g[:80] for g in gaps[:2])}"},
                        node="synthesize",
                    )
                    _LEDGER_LOG.info("re-grounding pass: %d unsupported claim(s): %s",
                                     len(gaps), gaps[:3])
                    return {
                        # Routed through the needs FIFO on purpose: the supervisor fulfils a
                        # need BEFORE consulting the decider, and the decider is what already
                        # said "done" on this state. A plain edge back would just be told done
                        # again.
                        "needs": [*(state.get("needs") or []),
                                  {"capability": _reground_to, "by": "synthesize",
                                   "reason": "answer was not grounded in the execution record"}],
                        "grounding_gaps": gaps,
                        "grounding_retries": state.get("grounding_retries", 0) + 1,
                        "reground": True,
                    }
            # Act on the verdict: a flagged audit appends a user-visible caveat to the answer.
            final = _apply_grounding_caveat(answer, audit)
            # Embed produced image artifacts (maps/plots) inline so they render in markdown.
            final = _append_image_embeds(final, artifacts)
            # Defuse sandbox: pseudo-URLs, internal filesystem paths, and any link that merely
            # LOOKS like an agent file (fabricated host) but is not an artifact this run produced.
            from agent_runtime.runtime_utils import sanitize_answer_links

            refs = _collect_download_refs(ar, cr)
            # An artifact from an EARLIER turn is still downloadable, so a link to it is not a
            # fabrication. Read those ids out of the conversation rather than depending on a
            # peer thread replaying its old tool results into this turn — see _refs_in_history.
            hist_refs = _refs_in_history(state.get("chat_history"))
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
                allowed_file_ids=[*refs["file_ids"], *hist_refs["file_ids"]],
                allowed_urls=[*refs["urls"], *hist_refs["urls"], *evidence_urls,
                              *web_allowed_urls()],
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
        final = _correct_artifact_claims(final, ar, cr, prior_rows=_rows)
        # Record what this turn DID before the state is discarded — the next turn's routing
        # decision is the only thing standing between a follow-up and a redundant search.
        _record_actions(state, ar, cr, extra_rows=state.get("action_rows"))
        merged = {**state, "answer": final, "audit": audit}
        # Clear the turn's rows on the way out. They are already in the thread ledger, this
        # state ships to the client verbatim, and an empty list makes double-recording
        # impossible if synthesize is ever re-entered.
        return {"answer": final, "final_answer": final, "audit": audit, "action_rows": [],
                # Cleared so a second synthesize cannot re-route, and so the gaps do not leak
                # into the client payload or a later turn's state.
                "reground": False, "grounding_gaps": [],
                "distilled": {**_distill(merged), "answer": final}}

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
    # synthesize is no longer unconditionally terminal: the grounding gate can send one pass
    # back through the supervisor, which fulfils the queued `analyze` need before deciding.
    builder.add_conditional_edges(
        "synthesize",
        lambda s: "supervisor" if s.get("reground") else "done",
        {"supervisor": "supervisor", "done": END},
    )
    return builder.compile()


def run_supervisor(
    query: str,
    *,
    chat_history: Optional[List[Any]] = None,
    llm: Optional[Any] = None,
    thread_id: Optional[str] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    unified_peer: Optional[bool] = None,
    **graph_kwargs: Any,
) -> Dict[str, Any]:
    """Build + run the supervisor graph; return the full final state."""
    graph = build_supervisor_graph(llm=llm, **graph_kwargs)
    return graph.invoke(
        {
            "query": query,
            "chat_history": chat_history or [],
            "thread_id": thread_id,
            "unified_peer": unified_peer,
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
    "default_decide_fn",
    "default_search_fn",
    "default_analyze_fn",
    "default_code_fn",
    "default_synthesize_fn",
]
