"""Publication methodological + provenance extractor (#3) — webhook (upload) path.

Reads a document (.tex/.txt/.md/.rst directly; .pdf via pypdf if available; .docx via
python-docx if available), then LLM-extracts the described method/workflow into
{summary, steps, datasets_referenced, tools_referenced, params}. Emits ONE
PublicationMethodSpec AssetRecord (index-only) + provenance edges
(DESCRIBES_METHOD, USES). NEVER executable.

The LLM step (rag_pipeline.llm_utils.call_llm) is OPTIONAL: with no LLM endpoint it
degrades to a text-excerpt method-spec + a note, so ingestion still succeeds offline.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import (
    EMIT_OPENSEARCH,
    KIND_PUBLICATION,
    AssetRecord,
    ExtractContext,
    Extractor,
    ExtractionResult,
    ProvenanceEdge,
)
from .doc_ids import publication_methodspec_doc_id, resource_type_for

_PROMPT = (
    "You extract the computational METHOD/WORKFLOW described in a scientific document.\n"
    "Return JSON ONLY with keys: summary (1-2 sentences), steps (ordered list of short "
    "strings), datasets_referenced (list), tools_referenced (list), params (object).\n"
    "If the document does not describe a method, return empty lists.\n\nDOCUMENT:\n"
)


def _read_text(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in {".tex", ".txt", ".md", ".rst"}:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    if ext == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore
            return "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
        except Exception:
            return ""
    if ext in {".docx", ".doc"}:
        try:
            import docx  # type: ignore
            return "\n".join(p.text for p in docx.Document(path).paragraphs)
        except Exception:
            return ""
    return ""


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    for cand in ([text.strip()] + ([m.group(0)] if m else [])):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


# First-class extraction status. This used to be a free-text `note`, which meant the ONE thing
# a reader needs to know — "was a method actually extracted, or is this an empty shell?" —
# required string matching on a message that also carried exception class names.
STATUS_EXTRACTED = "llm_extracted"
STATUS_UNPARSEABLE = "llm_unparseable"
STATUS_UNAVAILABLE = "llm_unavailable"
STATUS_NO_TEXT = "no_text"

DEGRADED_STATUSES = {STATUS_UNPARSEABLE, STATUS_UNAVAILABLE, STATUS_NO_TEXT}


def _is_fatal(exc: BaseException) -> bool:
    """Whether this failure will repeat for every remaining call, so a batch must stop.

    An expired credential is the motivating case, and it is not hypothetical — the CLI's OAuth
    token expired mid-session during this work. Matched on the TYPE the provider raises, not on
    the wording of a message.
    """
    try:
        from rag_pipeline.llm_claude_cli import ClaudeCliUnavailable
    except Exception:
        return False
    return isinstance(exc, ClaudeCliUnavailable)


def _empty_spec(status: str, summary: str = "", **extra: Any) -> Dict[str, Any]:
    spec = {"summary": summary, "steps": [], "datasets_referenced": [], "tools_referenced": [],
            "params": {}, "status": status, "degraded": status in DEGRADED_STATUSES}
    spec.update(extra)
    return spec


STATUS_PARTIAL = "llm_partial"

# Not in DEGRADED_STATUSES: a partial extraction produced real steps and is a usable method spec.
# It is a *qualified* success, and conflating it with "the LLM was unreachable" would throw away
# the steps that were extracted.


def paragraph_chunks(text: str, *, max_chars: int = 12000, max_chunks: int = 0) -> List[str]:
    """Split on PARAGRAPH boundaries, never mid-sentence.

    ``text[:12000]`` cut the document at a fixed offset — mid-word, mid-sentence, and for any
    paper longer than ~12 KB it discarded the Methods section entirely whenever that section came
    late, which in a standard paper layout is most of the time. The model then reported the
    honest truth about the text it was shown, and the result was recorded as the method of the
    paper.

    A paragraph that is itself longer than ``max_chars`` is emitted whole rather than split: a
    hard cut inside a paragraph is the very thing this exists to avoid, and the model tolerates
    an over-long chunk better than a truncated sentence.
    """
    body = (text or "").strip()
    if not body:
        return []
    cap = max(1, int(max_chars))
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    if not paragraphs:
        paragraphs = [body]

    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        if not current:
            current = para
        elif len(current) + 2 + len(para) <= cap:
            current = f"{current}\n\n{para}"
        else:
            chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    limit = max_chunks if max_chunks > 0 else _max_chunks()
    return chunks[:limit] if limit > 0 else chunks


def _max_chunks() -> int:
    """Chunk budget per publication. Each chunk is one LLM call, so this is the cost dial."""
    raw = (os.getenv("PUB_MAX_CHUNKS") or "4").strip()
    try:
        return max(1, min(20, int(raw)))
    except ValueError:
        return 4


def _merge_specs(specs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Reduce per-chunk specs into one, preserving step ORDER and de-duplicating.

    Order matters because ``steps`` is documented as ordered: a method's third step read before
    its first is not a method. Chunks arrive in document order, so concatenating in arrival order
    preserves it; the dedupe is case-insensitive on whitespace-normalised text, because the same
    step described in two chunks is one step, not two.
    """
    steps: List[str] = []
    seen_steps: set = set()
    datasets: List[Any] = []
    tools: List[Any] = []
    params: Dict[str, Any] = {}
    summaries: List[str] = []

    for spec in specs:
        for step in (spec.get("steps") or []):
            key = " ".join(str(step).split()).lower()
            if key and key not in seen_steps:
                seen_steps.add(key)
                steps.append(step)
        for src, dst in ((spec.get("datasets_referenced") or [], datasets),
                         (spec.get("tools_referenced") or [], tools)):
            for item in src:
                if item not in dst:
                    dst.append(item)
        got = spec.get("params")
        if isinstance(got, dict):
            for k, v in got.items():
                params.setdefault(k, v)      # first chunk to name a parameter wins
        summary = str(spec.get("summary") or "").strip()
        if summary and summary not in summaries:
            summaries.append(summary)

    return {"summary": " ".join(summaries)[:1000], "steps": steps,
            "datasets_referenced": datasets, "tools_referenced": tools, "params": params}


def extract_method(text: str, *, max_chars: int = 12000,
                   max_chunks: int = 0) -> Dict[str, Any]:
    """LLM method extraction over paragraph chunks, with an explicit STATUS.

    The distinction that matters: an empty ``steps`` list from a paper that genuinely describes
    no reproducible method, and an empty ``steps`` list because the LLM was unreachable, are
    completely different facts. Downstream both used to render as "this publication describes no
    method" — a claim the extractor was never entitled to make.

    Chunked rather than truncated, and the coverage is RECORDED: ``chunks_used`` /
    ``chunks_total`` / ``chars_seen`` / ``chars_total``. A spec built from 4 of 30 chunks and one
    built from the whole paper must not be indistinguishable, which is exactly what a silent
    ``text[:12000]`` made them.
    """
    if not text.strip():
        return _empty_spec(STATUS_NO_TEXT)

    all_chunks = paragraph_chunks(text, max_chars=max_chars, max_chunks=10 ** 6)
    budget = max_chunks if max_chunks > 0 else _max_chunks()
    chunks = all_chunks[:budget]
    # Three counts, because two different things reduce coverage and they have DIFFERENT
    # remedies: the budget caps how many chunks are attempted (raise PUB_MAX_CHUNKS), and a
    # crashing model reduces how many of those produce anything (re-run). Reporting one number
    # for both told an operator to raise a budget that was never the constraint.
    coverage = {"chunks_total": len(all_chunks), "chunks_attempted": len(chunks),
                "chars_total": len(text)}

    try:
        from rag_pipeline.llm_utils import call_llm
    except Exception as exc:
        return _empty_spec(STATUS_UNAVAILABLE, text[:500].strip(),
                           error=f"{type(exc).__name__}: {exc}"[:200],
                           chunks_parsed=0, chars_seen=0, **coverage)

    parsed_specs: List[Dict[str, Any]] = []
    failures: List[str] = []
    chars_parsed = 0
    for index, chunk in enumerate(chunks):
        try:
            parsed = _extract_json(call_llm(_PROMPT + chunk))
        except Exception as exc:
            failures.append(f"chunk {index}: {type(exc).__name__}: {exc}"[:160])
            if _is_fatal(exc):
                # A credential does not fix itself. Continuing would call the model once per
                # remaining chunk of every remaining paper, turning one expired token into a
                # corpus of empty specs that each look like "this paper describes no method" —
                # and burning the batch's whole runtime to produce them. Stop and say why.
                raise
            continue
        if parsed:
            parsed_specs.append(parsed)
            chars_parsed += len(chunk)
        else:
            failures.append(f"chunk {index}: unparseable")

    if not parsed_specs:
        # Nothing usable came back. Which kind of nothing depends on whether the calls raised.
        status = (STATUS_UNAVAILABLE
                  if any("unparseable" not in f for f in failures) else STATUS_UNPARSEABLE)
        return _empty_spec(status, text[:500].strip(),
                           error="; ".join(failures)[:300] or None,
                           chunks_parsed=0, chars_seen=0, **coverage)

    merged = _merge_specs(parsed_specs)
    complete = len(parsed_specs) == len(all_chunks) and not failures
    merged["status"] = STATUS_EXTRACTED if complete else STATUS_PARTIAL
    # `chars_seen` is the text the spec is actually BASED ON, so a chunk that crashed does not
    # count towards it. It read `sum(len(c) for c in chunks)` — every chunk launched — which
    # reported full coverage for a run where half of them died.
    merged["chars_seen"] = chars_parsed
    # `degraded` stays False: real steps were extracted and the spec is usable. `status` and the
    # coverage counts carry the qualification, so a reader can tell a full pass from a partial one
    # without the two being collapsed into one flag.
    merged["degraded"] = False
    merged.update(coverage)
    merged["chunks_parsed"] = len(parsed_specs)
    merged["chars_seen"] = chars_parsed
    if failures:
        merged["chunk_failures"] = failures[:8]
    return merged


def implemented_by_edges(spec_doc_id: str, tools_referenced: Any) -> List[ProvenanceEdge]:
    """IMPLEMENTED_BY edges from a method spec to library units whose symbol it names.

    Deliberately conservative:

    * matches the FULL symbol only, so "model" does not link to every unit containing it;
    * skips names shorter than 4 characters and a small stoplist of generic verbs, which are
      the ones that would otherwise link a paper to half the library;
    * marks every edge ``confidence: low`` / ``by: symbol_match``. An edge asserting that a
      paper's method IS this function, on the strength of a shared name, would be a fabricated
      provenance claim — and provenance is the one thing here that has to be trustworthy.

    Returns [] when no library has been built, rather than guessing at symbols.
    """
    generic = {"load", "read", "plot", "map", "run", "main", "model", "train", "test",
               "data", "get", "set", "make", "build", "process", "analyze", "compute"}
    try:
        from agent_runtime.method_library import load_registry
        registry = load_registry()
    except Exception:
        return []
    if not registry:
        return []
    by_symbol: Dict[str, List[str]] = {}
    for key, entry in registry.items():
        if not isinstance(entry, dict) or entry.get("ambiguous") or entry.get("alias_for"):
            continue
        symbol = str(entry.get("library_symbol") or "").strip().lower()
        if symbol:
            by_symbol.setdefault(symbol, []).append(key)

    edges: List[ProvenanceEdge] = []
    seen: set = set()
    for raw in (tools_referenced or []):
        name = str(raw or "").strip().lower()
        # A paper writes "we used geopandas.sjoin"; the unit is named `sjoin`.
        name = name.rsplit(".", 1)[-1].rsplit("(", 1)[0].strip()
        if len(name) < 4 or name in generic:
            continue
        for qualified in by_symbol.get(name, []):
            if qualified in seen:
                continue
            seen.add(qualified)
            edges.append(ProvenanceEdge(
                src=spec_doc_id, rel="IMPLEMENTED_BY", dst=qualified,
                detail={"confidence": "low", "by": "symbol_match", "matched_name": name}))
    return edges


class PublicationExtractor:
    name = "publication"

    def extract(self, path: str, *, ctx: ExtractContext) -> ExtractionResult:
        fname = os.path.basename(path)
        anchor = ctx.anchor() or fname
        f = ctx.fields or {}
        title = str(f.get("title") or fname)

        text = _read_text(path)
        method = extract_method(text)

        doc_id = publication_methodspec_doc_id(anchor)
        steps = method.get("steps") or []
        body = method.get("summary") or ""
        if steps:
            body += "\n\nSteps:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
        status = method.get("status") or STATUS_EXTRACTED
        degraded = bool(method.get("degraded"))
        contents = f"{title}\n{body}".strip()
        total = method.get("chunks_total")
        attempted = method.get("chunks_attempted")
        parsed = method.get("chunks_parsed")
        if degraded:
            # Prefixed, not appended: the evidence view truncates, and a caveat that only
            # appears after 4000 characters is a caveat nobody reads. An empty `steps` list must
            # never be presentable as "this paper describes no method".
            contents = (f"[METHOD SPEC UNAVAILABLE: {status}] No method steps were extracted "
                        f"from this publication — this is an extraction failure, NOT evidence "
                        f"that the paper describes no method.\n\n" + contents)
        elif status == STATUS_PARTIAL:
            # A qualified success, and the qualification is prefixed for the same reason. Steps
            # WERE extracted, so this is not a failure — but a spec built from part of a paper
            # must not read as the paper's whole method. Without this the coverage counts existed
            # only in `extracted` and nothing the agent reads ever mentioned them.
            # The count that matters is chunks that PRODUCED a spec. Rendering chunks launched
            # said "4 of 4 sections" for a run where two of them crashed — complete coverage,
            # next to a status of `llm_partial` that said the opposite.
            seen = (f"{parsed} of {total} sections" if parsed and total
                    else "part of the document")
            why = ""
            if attempted and total and attempted < total:
                why = (f" Coverage was capped at {attempted} section(s) by PUB_MAX_CHUNKS.")
            if parsed is not None and attempted and parsed < attempted:
                why += (f" {attempted - parsed} section(s) failed to extract and were lost, "
                        f"which a re-run may recover.")
            contents = (f"[PARTIAL METHOD SPEC] Extracted from {seen} of this publication, so "
                        f"steps described elsewhere in the paper may be missing. What follows is "
                        f"real but may be incomplete.{why}\n\n" + contents)

        source_fields = {k: f[k] for k in ("authors", "contributor", "abstract", "tags", "license", "doi")
                         if f.get(k)}
        asset = AssetRecord(
            asset_id=doc_id, kind=KIND_PUBLICATION, resource_type=resource_type_for(KIND_PUBLICATION),
            doc_id=doc_id, emit_targets=[EMIT_OPENSEARCH], source_rel_path=fname, title=title,
            contents=contents, source_fields=source_fields,
            extracted={"steps": steps, "datasets_referenced": method.get("datasets_referenced") or [],
                       "tools_referenced": method.get("tools_referenced") or [],
                       "params": method.get("params") or {},
                       "status": status,
                       "degraded": degraded,
                       "is_method_spec": not degraded,
                       # Coverage is queryable, not just prose: "which specs were built from a
                       # truncated read" is a corpus-quality question someone will need to ask.
                       "chunks_total": total, "chunks_attempted": attempted,
                       "chunks_parsed": parsed,
                       "chars_seen": method.get("chars_seen"),
                       "chars_total": method.get("chars_total"),
                       "chunk_failures": method.get("chunk_failures") or [],
                       "error": method.get("error"),
                       "parent_type": "Publication", "parent_title": title},
        )
        edges: List[ProvenanceEdge] = [
            ProvenanceEdge(src=anchor, rel="DESCRIBES_METHOD", dst=doc_id)
        ]
        for ds in (method.get("datasets_referenced") or []):
            edges.append(ProvenanceEdge(src=doc_id, rel="USES", dst=str(ds),
                                        detail={"confidence": "low", "by": "name_match"}))
        # IMPLEMENTED_BY: link a described method to callable units that appear to implement it.
        # Declared in base.py from the start and never written by anything, so a paper's method
        # spec and the code that realises it had no connection at all — which is the whole point
        # of extracting both. Name matching only, and it SAYS so: `confidence: low` and
        # `by: symbol_match`, because a shared name is a hint, not proof.
        edges.extend(implemented_by_edges(doc_id, method.get("tools_referenced") or []))
        warnings = ([f"publication: {status}" + (f" ({method['error']})" if method.get("error") else "")]
                    if degraded else [])
        if not degraded and status == STATUS_PARTIAL:
            # Name the ACTUAL constraint. "raise PUB_MAX_CHUNKS" is useless advice when the
            # budget was never the limit, and it sends an operator to change the one thing that
            # would not have helped.
            if attempted and total and attempted < total:
                warnings.append(f"publication: {status} — budget capped the read at "
                                f"{attempted} of {total} sections; raise PUB_MAX_CHUNKS")
            if parsed is not None and attempted and parsed < attempted:
                warnings.append(f"publication: {status} — {attempted - parsed} of {attempted} "
                                f"section(s) failed to extract; a re-run may recover them")
        return ExtractionResult(assets=[asset], edges=edges, warnings=warnings)


_: Extractor = PublicationExtractor()  # type: ignore[assignment]

__all__ = ["PublicationExtractor", "extract_method", "paragraph_chunks", "STATUS_EXTRACTED",
           "STATUS_PARTIAL", "STATUS_UNPARSEABLE", "STATUS_UNAVAILABLE", "STATUS_NO_TEXT",
           "DEGRADED_STATUSES"]
