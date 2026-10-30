from __future__ import annotations

import csv
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .file_store import (create_output_file, current_session, find_files, get_file_record,
                         managed_path_record, resolve_file_id, storage_root)
from agent_runtime.tool_args import accept_null_defaults

DEFAULT_MAX_CHARS = 12000
DEFAULT_MAX_ROWS = 20


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _allowed_roots() -> List[Path]:
    roots = {_repo_root().resolve(), storage_root().resolve()}
    upload_folder = os.getenv("UPLOAD_FOLDER")
    if upload_folder:
        roots.add(Path(upload_folder).expanduser().resolve())
    return sorted(roots)


def _find_managed_record_by_name(filename: str) -> Optional[Dict[str, Any]]:
    """The stored file a bare ``filename`` names for THIS caller, or None: the newest record
    with exactly that name among those ``find_files`` lets this conversation and this user see.

    This was a scan of every conversation's uploads/ and outputs/ for an on-disk name ending in
    ``__<filename>``, and it checked neither the conversation nor the owner, so bob read alice's
    ``summary.md`` by name while find_files hid it from him. find_files is where both scopes are
    applied, so a name goes through it.

    The unowned legacy pool stays readable, because find_files keeps it for reuse, and it competes
    on recency like everything else. Uploads carry no conversation either (the upload route binds
    the user, not the thread), so ranking this conversation's own files first would put an older
    output above the file the user just re-uploaded under the same name.
    """
    if not filename or any(sep in filename for sep in ("/", "\\")):
        return None
    # find_files matches a substring, so only the exact name counts: "summary.md" is neither
    # "old_summary.md" nor "final__summary.md", which the suffix scan matched. And no page, since
    # find_files sorts every match before cutting one, and newer near-misses would fill it.
    for record in find_files(name=filename, limit=sys.maxsize):
        if record.get("filename") == filename:
            return record
    return None


def _resolve_local_allowed_path(path: str, *, must_exist: bool = True, for_write: bool = False
                                ) -> Tuple[Path, Optional[Dict[str, Any]]]:
    ref = str(path or "").strip()
    raw_candidate = Path(ref).expanduser()
    if raw_candidate.is_absolute():
        candidates = [raw_candidate.resolve()]
    else:
        candidates = [
            (storage_root() / raw_candidate).resolve(),
            (_repo_root() / raw_candidate).resolve(),
        ]

    allowed = _allowed_roots()
    for candidate in candidates:
        # A path into the store's own directories is the record it names, with that record's
        # checks, whichever allowed root it came in under. A refused read is skipped exactly as a
        # missing file is; a refused write says why, and never falls through to another root.
        try:
            record = managed_path_record(candidate, write=for_write)
        except ValueError:
            if for_write:
                raise
            continue
        if record is not None:
            return resolve_file_id(str(record["file_id"])), record
        if not any(candidate == root or root in candidate.parents for root in allowed):
            continue
        if must_exist and not candidate.exists():
            continue
        return candidate, None

    allowed_list = ", ".join(str(root) for root in allowed)
    if must_exist:
        checked = ", ".join(str(candidate) for candidate in candidates)
        raise ValueError(f"file does not exist: {ref}; checked: {checked}")
    raise ValueError(f"path must be inside an allowed root: {allowed_list}")


def _resolve_allowed_path(path: str, *, must_exist: bool = True, for_write: bool = False
                          ) -> Tuple[Path, Optional[Dict[str, Any]]]:
    ref = str(path or "").strip()
    if not ref:
        raise ValueError("path is required")

    # A write names a path. write_text_file sends anything that could be an id or a stored
    # filename to create_output_file, so only a path reaches here to be written.
    if for_write:
        return _resolve_local_allowed_path(ref, must_exist=must_exist, for_write=True)

    record = get_file_record(ref)
    if record:
        return resolve_file_id(ref), record

    record = _find_managed_record_by_name(ref)
    if record:
        return resolve_file_id(str(record["file_id"])), record

    return _resolve_local_allowed_path(ref, must_exist=must_exist)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _display_filename(path: Path, record: Optional[Dict[str, Any]]) -> str:
    if record and record.get("filename"):
        return str(record["filename"])
    name = path.name
    if "__" in name:
        return name.split("__", 1)[1] or name
    return name


def _truncate(text: str, max_chars: int) -> str:
    max_chars = max(256, int(max_chars or DEFAULT_MAX_CHARS))
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n\n[truncated {len(text) - max_chars} characters]"


def _csv_preview(text: str, max_rows: int) -> Dict[str, Any]:
    # The preview used to be the first max_rows lines and nothing else: header plus 19 data rows
    # of a 40-row file, with no sign there was more. On 2026-10-23 (trap test T1, twice) the
    # model answered "19 stores" from it, having typed those rows into its code. The whole file
    # is counted now, the preview says it is one, and the lat/lon check reads every row.
    cap = max(1, int(max_rows or DEFAULT_MAX_ROWS))
    all_rows = [[str(item) for item in row] for row in csv.reader(io.StringIO(text))]
    rows = all_rows[:cap]
    header = rows[0] if rows else []
    data_total = max(0, len(all_rows) - 1)
    data_shown = max(0, len(rows) - 1)
    out: Dict[str, Any] = {"format": "csv", "header": header, "rows": rows,
                           "total_data_rows": data_total, "data_rows_shown": data_shown,
                           "truncated": data_shown < data_total}
    if out["truncated"]:
        out["note"] = (f"PREVIEW ONLY: {data_shown} of {data_total} data rows are shown. Read the "
                       "whole file by its path in code (e.g. pandas.read_csv) and never copy "
                       "rows from this preview into code; an answer from these rows alone "
                       "leaves out the rest of the file.")
    try:
        from agent_runtime.coord_sanity import latlon_swap_hint

        hint = latlon_swap_hint(header, all_rows[1:])
        if hint:
            out.update(hint)
    except Exception:  # noqa: BLE001 - a hint must never cost the preview
        pass
    return out


def _geojson_preview(text: str, max_chars: int) -> Dict[str, Any]:
    """The text (truncated as before), plus the feature count, the coordinate bounds and a CRS
    warning when those bounds cannot be degrees (see coord_sanity.crs_check)."""
    out: Dict[str, Any] = {"format": "geojson", "content": _truncate(text, max_chars)}
    parsed = json.loads(text)
    feats = parsed.get("features") if isinstance(parsed, dict) else None
    if isinstance(feats, list):
        out["feature_count"] = len(feats)
    xs: List[float] = []
    ys: List[float] = []

    def walk(c: Any) -> None:
        if isinstance(c, (list, tuple)):
            if len(c) >= 2 and all(isinstance(v, (int, float)) for v in c[:2]):
                xs.append(float(c[0]))
                ys.append(float(c[1]))
            else:
                for item in c:
                    walk(item)

    for f in feats or []:
        walk(((f or {}).get("geometry") or {}).get("coordinates"))
    if xs:
        bounds = [min(xs), min(ys), max(xs), max(ys)]
        out["bounds"] = bounds
        declared = None
        crs_member = parsed.get("crs") if isinstance(parsed, dict) else None
        if isinstance(crs_member, dict):
            declared = ((crs_member.get("properties") or {}).get("name"))
        from agent_runtime.coord_sanity import crs_check

        warn = crs_check(bounds, declared)
        if warn:
            out.update(warn)
    return out


def _json_preview(text: str) -> Dict[str, Any]:
    parsed = json.loads(text)
    if isinstance(parsed, dict):
        keys = list(parsed.keys())[:50]
        preview = {key: parsed[key] for key in keys[:10]}
        return {"format": "json", "top_level_type": "object", "keys": keys, "preview": preview}
    if isinstance(parsed, list):
        preview = parsed[:5]
        return {"format": "json", "top_level_type": "array", "length": len(parsed), "preview": preview}
    return {"format": "json", "top_level_type": type(parsed).__name__, "preview": parsed}


# The SEARCH peer retrieves outside evidence. It also holds these file tools (they are part of
# the shared toolset), and on 2026-10-15 (Chicago demo, SKATER step) it was routed in after the
# analysis had already delivered the map, and spent 3 min 16 s reading 45 of the conversation's
# own files one by one: the scripts, CSVs and layers the analysis had just written, some twice,
# its prompt growing from 8k to 56k tokens, until "2 consecutive steps added nothing" ended it.
# None of it could improve the answer: those results are already in the turn's record.
# In the search peer, a file the conversation PRODUCED is refused, a user's upload may be read,
# and at most SEARCH_READ_BUDGET distinct files per run, each once.
SEARCH_READ_BUDGET = int(os.getenv("AGENT_SEARCH_FILE_READS", "3") or 3)
_SEARCH_READS: Dict[str, List[str]] = {}


def _search_read_refusal(path: str, record: Optional[Dict[str, Any]]) -> Optional[str]:
    try:
        from agent_runtime import turn_log

        if turn_log.active_peer() != "search":
            return None
        log = turn_log.active()
        # A run id ("search#2") repeats in every turn; the log id makes it this turn's.
        run = f"{getattr(log, 'id', '')}:{turn_log.active_run() or 'search'}"
    except Exception:  # noqa: BLE001 - outside a turn there is nothing to limit
        return None
    if record and str(record.get("kind") or "") == "output":
        return ("not read: this file was produced by this conversation's own analysis, and its "
                "results are already in the turn's record. The search step retrieves OUTSIDE "
                "evidence (the knowledge base, catalogs, the web).")
    key = str((record or {}).get("file_id") or path)
    seen = _SEARCH_READS.setdefault(run, [])
    if key in seen:
        return "not read again: this file was already read in this search step."
    if len(seen) >= SEARCH_READ_BUDGET:
        return (f"not read: the search step reads at most {SEARCH_READ_BUDGET} files. "
                "Answer from what you have, or say what is missing.")
    seen.append(key)
    if len(_SEARCH_READS) > 256:
        _SEARCH_READS.pop(next(iter(_SEARCH_READS)))
    return None


def read_text_file_tool(path: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    resolved, record = _resolve_allowed_path(path, must_exist=True)
    refusal = _search_read_refusal(path, record)
    if refusal:
        return json.dumps({"ok": False, "file_id": (record or {}).get("file_id"),
                           "filename": _display_filename(resolved, record), "error": refusal})
    if not resolved.is_file():
        raise ValueError(f"path is not a file: {resolved}")

    text = _read_text(resolved)
    payload = {
        "file_id": (record or {}).get("file_id"),
        "filename": _display_filename(resolved, record),
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "content": _truncate(text, max_chars),
    }
    if record:
        payload["download_url"] = record.get("download_url")
    return json.dumps(payload, ensure_ascii=True, default=str)


def inspect_file_for_analysis_tool(
    path: str,
    question: Optional[str] = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> str:
    resolved, record = _resolve_allowed_path(path, must_exist=True)
    refusal = _search_read_refusal(path, record)
    if refusal:
        return json.dumps({"ok": False, "file_id": (record or {}).get("file_id"),
                           "filename": _display_filename(resolved, record), "error": refusal})
    if not resolved.is_file():
        raise ValueError(f"path is not a file: {resolved}")

    suffix = resolved.suffix.lower()
    text = _read_text(resolved)
    payload: Dict[str, Any] = {
        "file_id": (record or {}).get("file_id"),
        "path": str(resolved),
        "filename": _display_filename(resolved, record),
        "question": question or "",
        "size_bytes": resolved.stat().st_size,
    }
    if record:
        payload["download_url"] = record.get("download_url")

    try:
        if suffix == ".csv":
            payload["analysis_ready_content"] = _csv_preview(text, max_rows=max_rows)
        elif suffix == ".geojson":
            payload["analysis_ready_content"] = _geojson_preview(text, max_chars)
        elif suffix == ".json":
            payload["analysis_ready_content"] = _json_preview(text)
        else:
            payload["analysis_ready_content"] = {"format": "text", "content": _truncate(text, max_chars)}
    except Exception:
        payload["analysis_ready_content"] = {"format": "text", "content": _truncate(text, max_chars)}

    return json.dumps(payload, ensure_ascii=True, default=str)


def write_text_file_tool(path: str, content: str, overwrite: bool = False) -> str:
    if path and not any(sep in str(path) for sep in ("/", "\\")) and not str(path).startswith("."):
        record = create_output_file(str(path), content or "", overwrite=overwrite)
        payload = {
            "file_id": record["file_id"],
            "filename": record["filename"],
            "path": record["path"],
            "size_bytes": record["size_bytes"],
            "download_url": record["download_url"],
            "overwritten": bool(overwrite),
        }
        return json.dumps(payload, ensure_ascii=True, default=str)

    resolved, record = _resolve_allowed_path(path, must_exist=False, for_write=True)
    existed_before = resolved.exists()
    if existed_before and not overwrite:
        raise ValueError(f"file already exists: {resolved}")

    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content or "", encoding="utf-8")
    payload = {
        "file_id": (record or {}).get("file_id"),
        "filename": _display_filename(resolved, record),
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "overwritten": bool(overwrite and existed_before),
    }
    if record:
        payload["download_url"] = record.get("download_url")
    return json.dumps(payload, ensure_ascii=True, default=str)


def write_output_file_tool(filename: str, content: str, overwrite: bool = False) -> str:
    record = create_output_file(filename, content or "", overwrite=overwrite)
    payload = {
        "file_id": record["file_id"],
        "filename": record["filename"],
        "path": record["path"],
        "size_bytes": record["size_bytes"],
        "download_url": record["download_url"],
        "overwritten": bool(overwrite),
    }
    return json.dumps(payload, ensure_ascii=True, default=str)



# How many of this conversation's files to describe. Generous: the point of the tool is that the
# answer is COMPLETE, and a turn that produced twelve artifacts must not be told about ten.
_CONVERSATION_FILE_MAX = 200


def list_conversation_files_tool(name: Optional[str] = None, limit: int = 50) -> str:
    """Every file THIS conversation has made or been given, newest first."""
    session = current_session()
    capped = max(1, min(int(limit or 50), _CONVERSATION_FILE_MAX))
    # include_unowned=False: records written before sessions existed belong to no conversation,
    # and answering "what have you saved for me" with the deployment's whole history would be
    # worse than answering with nothing.
    records = find_files(name, limit=capped + 1, include_unowned=False)
    files = [{
        "file_id": r.get("file_id"),
        "filename": r.get("filename"),
        "kind": r.get("kind"),
        "size_bytes": r.get("size_bytes"),
        "download_url": r.get("download_url"),
    } for r in records[:capped]]

    payload: Dict[str, Any] = {"ok": True, "count": len(files), "files": files}
    if len(records) > capped:
        payload["truncated"] = f"showing the {capped} newest; ask for more with a higher limit"
    if not session:
        # No conversation bound (a CLI run, or a request that never went through the API edge).
        # Saying "you have no files" would be a lie of a different kind, so name the reason.
        payload["scope_unknown"] = ("No conversation is bound to this request, so files cannot "
                                    "be attributed to it. This list is not authoritative.")
    elif not files:
        payload["note"] = ("Nothing has been saved in this conversation yet. Files from other "
                           "conversations are deliberately not listed.")
    else:
        payload["note"] = ("This is the complete list for this conversation. Quote these "
                           "filenames and links rather than any remembered from the transcript, "
                           "and do not describe a file as saved unless it appears here.")
    return json.dumps(payload, ensure_ascii=True, default=str)


def make_langchain_file_tools() -> List[Any]:
    try:
        from langchain_core.tools import StructuredTool
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "LangChain is not installed. Add `langchain-core` (or langchain) to dependencies."
        ) from exc

    return [
        StructuredTool.from_function(func=accept_null_defaults(read_text_file_tool),
            name="read_text_file",
            description=(
                "Read a UTF-8 text-like file from an allowed local path and return its contents. "
                "Input may be either a local path or an uploaded file_id. "
                "Use for txt, md, py, csv, json, or other text files."
            ),
            metadata={"category": "io"},
        ),
        StructuredTool.from_function(func=accept_null_defaults(inspect_file_for_analysis_tool),
            name="inspect_file_for_analysis",
            description=(
                "Load a local file or uploaded file_id into an LLM-friendly JSON payload for interpretation. "
                "For CSV returns header and sample rows, for JSON returns structural preview, "
                "otherwise returns text content."
            ),
            metadata={"category": "io"},
        ),
        StructuredTool.from_function(func=accept_null_defaults(write_text_file_tool),
            name="write_text_file",
            description=(
                "Write text output to a local file under an allowed root. "
                "Use for saving summaries, reports, code, or extracted notes."
            ),
            metadata={"category": "io"},
        ),
        StructuredTool.from_function(func=accept_null_defaults(write_output_file_tool),
            name="write_output_file",
            description=(
                "Write downloadable output for the user into managed agent storage using only a filename. "
                "Returns file_id and download_url for the generated file."
            ),
            metadata={"category": "io"},
        ),
        StructuredTool.from_function(func=accept_null_defaults(list_conversation_files_tool),
            name="list_conversation_files",
            description=(
                "List the files THIS conversation has produced or been given, newest first, with "
                "file_id, filename and download_url. USE IT whenever the user asks what files "
                "exist, what was saved, or for a link to something made earlier — the transcript "
                "is not a reliable record of that and earlier links may have been dropped from "
                "context. Optional `name` filters by filename substring."
            ),
            metadata={"category": "io"},
        ),
    ]


def make_conversation_file_tools() -> List[Any]:
    """Just the listing, for turns with no upload.

    The full file toolset is attached only when the user uploaded something, which is the wrong
    condition for this one tool: a turn creates files without any upload — a boundary, an
    embedding, a plot — and it is exactly then that "what did you save?" gets asked. Without it
    the analyse peer reached for `execute_code` and listed the sandbox working directory instead,
    inventing its own scratch script as one of the conversation's artifacts.
    """
    tools = make_langchain_file_tools()
    return [t for t in tools if str(getattr(t, "name", "")) == "list_conversation_files"]


__all__ = [
    "inspect_file_for_analysis_tool",
    "make_conversation_file_tools",
    "list_conversation_files_tool",
    "make_langchain_file_tools",
    "read_text_file_tool",
    "write_output_file_tool",
    "write_text_file_tool",
]
