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
from html.parser import HTMLParser
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


def sniff_kind(path: str) -> str:
    """What the file's first bytes say it is — ``pdf`` / ``html`` / ``zip`` / ``text``.

    The extension lies. ``03bc2865__oa.pdf`` in the corpus cache begins ``<head`` — an HTML page
    saved with a .pdf name, because the DOI's "open access" link served a landing page and the
    fetcher trusted the URL. pypdf answers ``invalid pdf header``, ``_read_text`` swallowed it, and
    the element was reported as ``no_text``, which points a reader at OCR for a file that never
    contained a page image.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(1024)
    except OSError:
        return "unreadable"
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"PK\x03\x04"):
        return "zip"
    lowered = head.lstrip()[:400].lower()
    if lowered.startswith((b"<!doctype html", b"<html", b"<head", b"<?xml")) or b"<body" in lowered:
        return "html"
    return "text"


class _TextFromHTML(HTMLParser):
    """Visible text from an HTML document, using only the standard library.

    Worth having rather than relabelling: a substantial share of the corpus's open-access links
    resolve to a full-text HTML article (PMC, MDPI, Copernicus), which carries the same methods
    section the PDF would. Treating those as unreadable discards the content over a file
    extension.
    """

    _SKIP = {"script", "style", "noscript", "svg", "head", "nav", "footer", "form"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._depth = 0
        self._parts: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._depth:
            self._depth -= 1
        elif tag in ("p", "div", "br", "li", "h1", "h2", "h3", "h4", "tr"):
            self._parts.append("\n")

    def handle_data(self, data):
        if not self._depth and data.strip():
            self._parts.append(data.strip())

    def text(self) -> str:
        joined = " ".join(self._parts)
        return re.sub(r"[ \t]*\n[ \t\n]*", "\n", re.sub(r"[ \t]{2,}", " ", joined)).strip()


def _html_to_text(raw: str) -> str:
    parser = _TextFromHTML()
    try:
        parser.feed(raw)
        parser.close()
    except Exception:
        return ""
    return parser.text()


# Pages a publisher serves INSTEAD of the document. Matched only on short documents, because a
# real paper about, say, web security could legitimately contain any of these phrases.
_INTERSTITIALS = (
    ("a bot-check page", ("confirm you are a human", "are you a robot", "verify you are human",
                          "enable javascript and cookies", "checking your browser",
                          "ticking the box below", "cf-browser-verification")),
    ("a paywall or login page", ("purchase access", "institutional login", "sign in to continue",
                                 "subscribe to view", "get access to this article",
                                 "you do not have access")),
    ("an error page", ("404 not found", "page not found", "403 forbidden",
                       "service unavailable", "we apologize for the inconvenience")),
    ("a cookie consent page", ("we use cookies", "accept all cookies", "cookie preferences")),
)

_INTERSTITIAL_MAX_CHARS = 4000


def _interstitial_kind(text: str) -> str:
    """Name the wall a publisher served instead of the paper, or "" if this looks like a document.

    A bot check is a refusal, and the correct response to a refusal is to record it — never to
    attempt to satisfy it. This function exists so the corpus does not silently acquire CAPTCHA
    notices filed as methods sections, and so "N reachable open-access PDFs" means N documents.
    """
    if len(text) > _INTERSTITIAL_MAX_CHARS:
        return ""
    lowered = text.lower()
    for label, needles in _INTERSTITIALS:
        if any(needle in lowered for needle in needles):
            return label
    return ""


def read_document(path: str) -> tuple:
    """``(text, reason)`` — the document's text, and why it is empty when it is.

    Returning a bare ``""`` for every failure was the defect: a file that is not a PDF, a missing
    ``pypdf``, and a scanned PDF with no text layer produced identical output and one status
    (``no_text``), while needing three different fixes. Routing on sniffed CONTENT rather than the
    extension also means an HTML article reached by a ``.pdf`` URL is read instead of discarded.
    """
    ext = Path(path).suffix.lower()
    kind = sniff_kind(path)
    if kind == "unreadable":
        return "", "file could not be opened"
    try:
        if os.path.getsize(path) == 0:
            # A zero-byte file is a download that failed and was saved anyway. Two of the corpus
            # cache's fetched documents are exactly this, and without a reason here they read as
            # "the document contains nothing" rather than "the fetch produced nothing".
            return "", "the file is empty (0 bytes) — the fetch produced no content"
    except OSError:
        pass
    if kind == "zip" and ext != ".docx":
        return "", f"content is a zip archive, not a document (extension {ext or 'none'})"

    if kind == "html":
        try:
            raw = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return "", f"could not read: {exc}"
        text = _html_to_text(raw)
        if not text:
            return "", f"HTML carried no extractable text (extension {ext or 'none'})"
        interstitial = _interstitial_kind(text)
        if interstitial:
            # NOT an article. `03bc2865__oa.pdf` is IOP Publishing's bot-check page — "please can
            # you confirm you are a human by ticking the box below" — 356 characters that the
            # fetcher counted as a downloaded PDF. Indexing it would put a CAPTCHA notice in the
            # corpus as a paper's method. Reported so the count of reachable open-access PDFs is
            # honest; deliberately never worked around.
            return "", (f"the server returned {interstitial}, not the document "
                        f"(extension {ext or 'none'}, {len(text)} chars of HTML)")
        note = "" if ext in {".html", ".htm", ".xhtml"} else (
            f"content is HTML although the file is named {ext or 'without an extension'}; "
            f"read as HTML")
        return text, note

    if kind == "pdf" or ext == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError:
            return "", "pypdf is not installed, so no PDF can be read"
        try:
            text = "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
        except Exception as exc:
            return "", f"pypdf could not parse the file: {type(exc).__name__}: {exc}"[:200]
        if not text.strip():
            return "", "PDF parsed but carried no text layer (likely scanned; needs OCR)"
        return text, ""

    if ext in {".docx", ".doc"}:
        try:
            import docx  # type: ignore
        except ImportError:
            return "", "python-docx is not installed"
        try:
            return "\n".join(p.text for p in docx.Document(path).paragraphs), ""
        except Exception as exc:
            return "", f"python-docx could not parse the file: {type(exc).__name__}"

    if ext in {".tex", ".txt", ".md", ".rst"} or kind == "text":
        try:
            return Path(path).read_text(encoding="utf-8", errors="replace"), ""
        except OSError as exc:
            return "", f"could not read: {exc}"
    return "", f"no reader for extension {ext or 'none'} (content sniffed as {kind})"


def _read_text(path: str) -> str:
    """Text only — kept for callers that do not want the reason."""
    return read_document(path)[0]


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

# The document was read and understood, and it describes NO computational method. Editorials,
# opinion pieces, conceptual papers and publisher cover pages are all legitimately this.
#
# Measured on the 7 readable open-access documents in the corpus cache: THREE are this --
# `d78ceebb` is a PDXScholar citation/cover page, `f94c3e60` an editorial on pharmaceutical waste,
# `fd728b4e` an argumentative paper on AI ethics. All three were being emitted with
# `status: llm_extracted`, `degraded: False` and `is_method_spec: True` on an EMPTY `steps` list,
# because `is_method_spec` was defined as `not degraded` -- which conflates "the extractor
# succeeded" with "there is a method here". Nearly half the type was indexed as a method spec
# describing no method.
#
# Not degraded: nothing failed. The distinction from `no_text` is the whole point -- one is a
# fetch or parse failure to retry, the other is a fact about the document.
STATUS_NO_METHOD = "no_method_described"

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
    if not (merged.get("steps") or []):
        # Read, parsed, and there is no method to record. Distinguishing this from a failure is
        # what keeps "publications with no extractable method" a countable quality metric instead
        # of a silent third of the type.
        merged["status"] = STATUS_NO_METHOD if complete else STATUS_PARTIAL
    else:
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

        text, read_note = read_document(path)
        method = extract_method(text)
        if read_note:
            # WHY the text is empty (or which reader was used) reaches the record. "no_text" alone
            # sent a reader looking for a scanned page in a file that was HTML.
            method = {**method, "read_note": read_note}

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
            why_empty = f" Reason: {read_note}." if read_note and not text.strip() else ""
            contents = (f"[METHOD SPEC UNAVAILABLE: {status}] No method steps were extracted "
                        f"from this publication — this is an extraction failure, NOT evidence "
                        f"that the paper describes no method.{why_empty}\n\n" + contents)
        elif status == STATUS_NO_METHOD:
            # Prefixed for the same reason the other qualifications are: the evidence view is
            # truncated, and this sentence is the difference between "the extractor is broken" and
            # "this paper is an editorial".
            contents = (f"[NO COMPUTATIONAL METHOD] This publication was read in full and "
                        f"describes no computational method or workflow — it is not an extraction "
                        f"failure. The summary below says what the document is.\n\n" + contents)
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
                       # It IS a method spec only if it holds steps. Defined as `not degraded`,
                       # this was True for every editorial and cover page in the corpus.
                       "is_method_spec": bool(steps) and not degraded,
                       # Coverage is queryable, not just prose: "which specs were built from a
                       # truncated read" is a corpus-quality question someone will need to ask.
                       "chunks_total": total, "chunks_attempted": attempted,
                       "chunks_parsed": parsed,
                       "chars_seen": method.get("chars_seen"),
                       "chars_total": method.get("chars_total"),
                       "chunk_failures": method.get("chunk_failures") or [],
                       "error": method.get("error"),
                       # Which reader ran, and why it produced nothing when it did. A queryable
                       # answer to "how many of these are HTML pretending to be PDFs".
                       "read_note": read_note or None,
                       "source_kind": sniff_kind(path),
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
           "STATUS_NO_METHOD",
           "DEGRADED_STATUSES"]
