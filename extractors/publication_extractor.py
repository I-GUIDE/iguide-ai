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


def _empty_spec(status: str, summary: str = "", **extra: Any) -> Dict[str, Any]:
    spec = {"summary": summary, "steps": [], "datasets_referenced": [], "tools_referenced": [],
            "params": {}, "status": status, "degraded": status in DEGRADED_STATUSES}
    spec.update(extra)
    return spec


def extract_method(text: str, *, max_chars: int = 12000) -> Dict[str, Any]:
    """LLM method extraction, with an explicit STATUS rather than a free-text note.

    The distinction that matters: an empty ``steps`` list from a paper that genuinely describes
    no reproducible method, and an empty ``steps`` list because the LLM was unreachable, are
    completely different facts. Downstream both used to render as "this publication describes no
    method" — a claim the extractor was never entitled to make.
    """
    if not text.strip():
        return _empty_spec(STATUS_NO_TEXT)
    try:
        from rag_pipeline.llm_utils import call_llm
        raw = call_llm(_PROMPT + text[:max_chars])
        parsed = _extract_json(raw)
        if parsed:
            parsed.setdefault("steps", []); parsed.setdefault("datasets_referenced", [])
            parsed.setdefault("tools_referenced", []); parsed.setdefault("params", {})
            parsed.setdefault("summary", "")
            parsed["status"] = STATUS_EXTRACTED
            parsed["degraded"] = False
            return parsed
        return _empty_spec(STATUS_UNPARSEABLE, text[:500].strip())
    except Exception as exc:
        return _empty_spec(STATUS_UNAVAILABLE, text[:500].strip(),
                           error=f"{type(exc).__name__}: {exc}"[:200])


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
        if degraded:
            # Prefixed, not appended: the evidence view truncates, and a caveat that only
            # appears after 4000 characters is a caveat nobody reads. An empty `steps` list must
            # never be presentable as "this paper describes no method".
            contents = (f"[METHOD SPEC UNAVAILABLE: {status}] No method steps were extracted "
                        f"from this publication — this is an extraction failure, NOT evidence "
                        f"that the paper describes no method.\n\n" + contents)

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
        return ExtractionResult(assets=[asset], edges=edges, warnings=warnings)


_: Extractor = PublicationExtractor()  # type: ignore[assignment]

__all__ = ["PublicationExtractor", "extract_method", "STATUS_EXTRACTED",
           "STATUS_UNPARSEABLE", "STATUS_UNAVAILABLE", "STATUS_NO_TEXT", "DEGRADED_STATUSES"]
