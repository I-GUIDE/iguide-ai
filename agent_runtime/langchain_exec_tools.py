"""The `execute_code` tool — run/debug code in the sandbox (see code_execution.py)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from agent_runtime.tool_args import accept_null_defaults

# Bounds on how much gets auto-staged into a sandbox run (conversation files +
# explicitly requested files). Keeps a large session from blowing up disk/time.
DEFAULT_MAX_INPUT_FILES = 20
DEFAULT_MAX_INPUT_MB = 200


def _max_input_files() -> int:
    try:
        return max(1, int(os.getenv("AGENT_CODE_EXEC_MAX_INPUT_FILES", str(DEFAULT_MAX_INPUT_FILES))))
    except (TypeError, ValueError):
        return DEFAULT_MAX_INPUT_FILES


def _max_input_bytes() -> int:
    try:
        mb = float(os.getenv("AGENT_CODE_EXEC_MAX_INPUT_MB", str(DEFAULT_MAX_INPUT_MB)))
    except (TypeError, ValueError):
        mb = DEFAULT_MAX_INPUT_MB
    return int(max(1.0, mb) * 1024 * 1024)


def _resolve_input_file(ref: str) -> Tuple[Path, Optional[Dict[str, Any]]]:
    """Resolve an uploaded ``file_id`` (or an allowed local path) to a host path.

    Returns ``(host_path, record_or_None)``.  Raises ``ValueError`` if the
    reference cannot be resolved.
    """
    from agent_runtime.file_store import get_file_record, resolve_file_id

    ref = str(ref or "").strip()
    if not ref:
        raise ValueError("empty file reference")
    record = get_file_record(ref)
    if record:
        return resolve_file_id(ref), record
    # Fall back to an allowed local path (same policy as the file tools).
    from agent_runtime.langchain_file_tools import _resolve_allowed_path

    path, rec = _resolve_allowed_path(ref, must_exist=True)
    return path, rec


def _free_dest(filename: str, taken: Any) -> str:
    """A name in the work dir that nothing has claimed, derived from ``filename``.

    Reached only when every name a file could use is taken by another input. An ugly name the
    model can open beats a file it cannot reach at all.

    ``taken`` must include the names later inputs will legitimately own, not just the ones
    already handed out: searching only the latter let a derived name land on a filename a
    subsequent input actually has, and that input then lost its own name and became reachable
    by file_id alone — or, with no file_id, not at all.
    """
    base = str(filename or "input")
    head, dot, tail = base.rpartition(".")
    stem, suffix = (head, "." + tail) if dot and head else (base, "")
    n = 2
    while f"{stem}_{n}{suffix}" in taken:
        n += 1
    return f"{stem}_{n}{suffix}"


def _build_staging(refs: List[str]) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]], List[Dict[str, str]], List[Dict[str, Any]]]:
    """Resolve file references into copy specs for the sandbox work dir.

    Each resolved file is staged under BOTH its file_id and its original filename.
    Dedupes by resolved host path and enforces file-count / total-size caps.

    Returns ``(staging_specs, staged_info, errors, skipped)``.
    """
    staging: List[Dict[str, str]] = []
    staged_info: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    skipped: List[Dict[str, Any]] = []

    max_files = _max_input_files()
    max_bytes = _max_input_bytes()
    seen_sources: set[str] = set()
    total_bytes = 0

    # PASS 1 — resolve, dedupe by source, and apply the caps. Names are not handed out yet:
    # who owns a contested filename depends on the whole list, so it cannot be decided while
    # walking it.
    resolved: List[Dict[str, Any]] = []
    for ref in refs:
        try:
            host_path, record = _resolve_input_file(ref)
        except Exception as exc:
            errors.append({"ref": str(ref), "error": str(exc)})
            continue
        src = str(host_path)
        if src in seen_sources:  # same file referenced by id and filename/path
            continue
        try:
            size = int((record or {}).get("size_bytes") or host_path.stat().st_size)
        except OSError:
            size = 0
        if len(resolved) >= max_files:
            skipped.append({"ref": str(ref), "reason": "max input files exceeded", "limit": max_files})
            continue
        if total_bytes + size > max_bytes:
            skipped.append({"ref": str(ref), "reason": "max total input size exceeded",
                            "limit_bytes": max_bytes, "size_bytes": size})
            continue
        seen_sources.add(src)
        total_bytes += size
        resolved.append({"ref": str(ref), "src": src,
                         "filename": (record or {}).get("filename") or host_path.name,
                         "file_id": (record or {}).get("file_id")})

    # PASS 2 — allocate names.
    #
    # A file_id is unique by construction, so every input that has one keeps it. The human
    # FILENAME is what gets contested, and it goes to the LAST input that claims it.
    #
    # That direction is the fix. `refs` arrives oldest-first — the session's earlier files,
    # then this turn's uploads (get_session_files is documented as returning ids oldest
    # first) — so handing the plain name to the FIRST claimant gave it to a file from an
    # earlier turn. Upload a corrected data.csv, ask about it, and the peer opened the name it
    # was given in the question and read the previous turn's data instead: a wrong answer with
    # nothing on the surface to show for it. The reverse mistake — asking for an older file by
    # bare name after re-uploading a different one under that name — is both rarer and
    # ambiguous to a human reader too.
    filename_owner: Dict[str, int] = {}
    for i, entry in enumerate(resolved):
        if entry["filename"]:
            filename_owner[str(entry["filename"])] = i

    # Every name that will legitimately be owned, so a derived fallback cannot take one.
    taken: set[str] = {str(e["file_id"]) for e in resolved if e["file_id"]} | set(filename_owner)

    for i, entry in enumerate(resolved):
        names: List[str] = []
        if entry["file_id"]:
            names.append(str(entry["file_id"]))
        filename = str(entry["filename"] or "")
        if filename and filename_owner.get(filename) == i and filename not in names:
            names.append(filename)
        if not names:
            dest = _free_dest(filename or "input", taken)
            taken.add(dest)
            names.append(dest)
        for dest in names:
            staging.append({"source": entry["src"], "dest": dest})
        # `available_as` is what the model is told to open, so it must be the names this file
        # ACTUALLY has — not the ones it would have had if nothing else were staged.
        staged_info.append({"ref": entry["ref"], "file_id": entry["file_id"],
                            "filename": entry["filename"], "available_as": names})

    return staging, staged_info, errors, skipped


def make_code_execution_tools(
    executor: Optional[Any] = None,
    default_input_file_ids: Optional[List[str]] = None,
    session_id: Optional[str] = None,
) -> List[Any]:
    """Build the `execute_code` StructuredTool (container-per-run sandbox).

    ``default_input_file_ids`` are the files attached to the current conversation;
    they are auto-staged into EVERY run so the model can read them without naming
    them, and are unioned (deduped) with any explicit ``input_files`` it passes.
    """
    from langchain_core.tools import StructuredTool

    from agent_runtime.code_execution import DEFAULT_TIMEOUT, get_code_executor

    default_ids = [str(x).strip() for x in (default_input_file_ids or []) if str(x).strip()]

    def execute_code(
        code: str = "",
        language: str = "python",
        timeout_seconds: int = DEFAULT_TIMEOUT,
        dependencies: Optional[List[str]] = None,
        input_files: Optional[List[str]] = None,
        label: Optional[str] = None,
        entrypoint: Optional[str] = None,
    ) -> str:
        ex = executor or get_code_executor()

        # Union: conversation-attached files (auto) first, then any explicitly
        # named files, order-preserving and deduped.
        explicit = [str(r).strip() for r in (input_files or []) if str(r).strip()]
        refs = list(dict.fromkeys([*default_ids, *explicit]))
        staging, staged_info, input_errors, skipped = _build_staging(refs)

        # `label` only names the saved source, so pass it optionally: an executor
        # implementing the older signature (or a test double) still works.
        extra = {"label": label} if label else {}
        if session_id:
            extra["session"] = session_id   # durable workspace across runs in this conversation
        if entrypoint:
            extra["entrypoint"] = entrypoint  # run a file already in that workspace
        result = ex.execute(
            code,
            language=language,
            timeout=timeout_seconds,
            dependencies=dependencies,
            input_files=staging,
            **extra,
        )
        payload = result.to_dict()
        if staged_info:
            payload["input_files"] = staged_info
        if input_errors:
            payload["input_file_errors"] = input_errors
        if skipped:
            payload["input_files_skipped"] = skipped
        return json.dumps(payload, ensure_ascii=True, default=str)

    # ---------------------------------------------------------------- workspace edits
    # Without these, every fix is a whole new program: the model re-emits a 200-line script to
    # change line 40, through a full round trip carrying the peer's entire context. These make
    # the same fix a patch. They act on the conversation's durable working directory directly
    # (no container), so they are cheap and take effect on the next execute_code.
    #
    # The agent's general file tools cannot serve this purpose — they are rooted at the repo,
    # the file store and UPLOAD_FOLDER, and turn a bare filename into a managed store output,
    # so they would appear to edit the workspace while never touching it.

    def _workspace_error(exc: Exception) -> str:
        return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=True)

    def write_workspace_file(path: str, content: str) -> str:
        from agent_runtime.code_execution import resolve_workspace_file

        try:
            target = resolve_workspace_file(session_id, path)
        except ValueError as exc:
            return _workspace_error(exc)
        existed = target.is_file()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content or "", encoding="utf-8")
        except OSError as exc:
            return _workspace_error(exc)
        text = content or ""
        return json.dumps({
            "ok": True, "path": path, "replaced": existed,
            "bytes": len(text.encode("utf-8")), "lines": text.count("\n") + (1 if text else 0),
        }, ensure_ascii=True)

    def read_workspace_file(path: str, offset: int = 1, limit: int = 400) -> str:
        from agent_runtime.code_execution import resolve_workspace_file

        try:
            target = resolve_workspace_file(session_id, path)
        except ValueError as exc:
            return _workspace_error(exc)
        if not target.is_file():
            return _workspace_error(ValueError(f"no file {path!r} in the working directory"))
        try:
            lines = target.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            return _workspace_error(ValueError(
                f"{path!r} is not UTF-8 text, so it cannot be shown or patched as text; "
                "read it in code with execute_code instead"))
        except OSError as exc:
            return _workspace_error(exc)
        start = max(1, int(offset or 1))
        end = start + max(1, int(limit or 400))
        window = lines[start - 1:end - 1]
        return json.dumps({
            "ok": True, "path": path, "total_lines": len(lines),
            "from_line": start, "to_line": start + len(window) - 1,
            # Numbered so an edit can quote an exact line rather than approximate it.
            "content": "\n".join(f"{start + i}\t{ln}" for i, ln in enumerate(window)),
        }, ensure_ascii=True)

    def edit_workspace_file(path: str, old_text: str, new_text: str) -> str:
        from agent_runtime.code_execution import resolve_workspace_file

        try:
            target = resolve_workspace_file(session_id, path)
        except ValueError as exc:
            return _workspace_error(exc)
        if not target.is_file():
            return _workspace_error(ValueError(f"no file {path!r} in the working directory"))
        try:
            # STRICT, and the whole file is written back below. With errors="replace" every
            # undecodable byte becomes U+FFFD before the replacement is applied, so editing
            # one line of a latin-1 CSV would silently corrupt every other line in it.
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return _workspace_error(ValueError(
                f"{path!r} is not UTF-8 text; editing it as text would corrupt the bytes that "
                "cannot be decoded. Rewrite it in code with execute_code instead"))
        except OSError as exc:
            return _workspace_error(exc)
        if not old_text:
            return _workspace_error(ValueError(
                "old_text is empty; to create or overwrite the file use write_workspace_file"))
        hits = text.count(old_text)
        # A silent partial match is the failure mode that matters here: replacing the wrong
        # one of three identical lines produces code that runs and is wrong. So say which.
        if hits == 0:
            return _workspace_error(ValueError(
                f"old_text does not appear in {path!r}; read_workspace_file first and copy the "
                "text exactly, including indentation"))
        if hits > 1:
            return _workspace_error(ValueError(
                f"old_text appears {hits} times in {path!r}; include enough surrounding lines "
                "to make it unique"))
        try:
            target.write_text(text.replace(old_text, new_text or ""), encoding="utf-8")
        except OSError as exc:
            return _workspace_error(exc)
        return json.dumps({"ok": True, "path": path, "replacements": 1}, ensure_ascii=True)

    tool = StructuredTool.from_function(func=accept_null_defaults(execute_code),
        name="execute_code",
        description=(
            "Execute code in an isolated, sandboxed container and return JSON with "
            "exit_code, stdout, stderr, timed_out, the executed `code`, `installed`, and "
            "`artifacts` (the source is saved as a downloadable .py named from `label`, plus "
            "any files the run wrote). Pass `dependencies` (a list of pip specs, e.g. "
            "[\"numpy\", \"pandas==2.2\"]) to install third-party packages before running — "
            "they are installed with network in a separate step, then the code runs with NO "
            "network. Files attached to this conversation are AUTOMATICALLY available in the "
            "working directory under both their file_id and their original filename (e.g. "
            "open('data.csv') or pd.read_csv('data.csv')). To read any OTHER file, add its "
            "file_id to `input_files` — an upload, or a file_id an earlier TOOL returned in "
            "this conversation (e.g. an embedding package's embedding_package.file_id, to "
            "cluster or difference its vectors). Use this to RUN and DEBUG code: run, read "
            "stdout/stderr, fix, re-run. Files you write persist in this conversation's working "
            "directory, so a later run can open what an earlier one produced and keep building "
            "on it (the container itself is fresh each time). `label` is a short slug for what this particular run "
            "does (e.g. \"csv_to_geojson\", \"rivers_buffer\") and becomes the saved source's "
            "filename — several runs in one conversation otherwise arrive as identically-named "
            "downloads; name the files your code writes for their contents too, for the same "
            "reason. To change one part of a program you already wrote, do NOT re-send the "
            "whole thing: write it to a named file with write_workspace_file, then run it with "
            "`entrypoint` (e.g. entrypoint=\"main.py\", no `code`), and fix it with "
            "edit_workspace_file between runs."
        ),
    )
    tools = [tool]
    if session_id:
        # Only useful with a durable working directory to act on; without one they could
        # never do anything but explain that there is no workspace.
        tools += [
            StructuredTool.from_function(func=accept_null_defaults(write_workspace_file),
                name="write_workspace_file",
                description=(
                    "Create or overwrite a file in this conversation's working directory — the "
                    "same directory execute_code runs in. Use it to keep a program in a named "
                    "file (e.g. 'main.py', 'clean.py') instead of re-sending it every run: then "
                    "execute_code(entrypoint='main.py') runs it and edit_workspace_file changes "
                    "it in place. Also fine for data or config the code reads."
                ),
            ),
            StructuredTool.from_function(func=accept_null_defaults(read_workspace_file),
                name="read_workspace_file",
                description=(
                    "Read a file from this conversation's working directory, with line numbers. "
                    "Read before you edit: edit_workspace_file matches text exactly, so guessing "
                    "at what a line says wastes the call. `offset`/`limit` window a long file."
                ),
            ),
            StructuredTool.from_function(func=accept_null_defaults(edit_workspace_file),
                name="edit_workspace_file",
                description=(
                    "Replace an exact snippet in a file in this conversation's working directory "
                    "— the way to fix a few lines of a program without re-sending it. `old_text` "
                    "must appear EXACTLY ONCE, whitespace and indentation included; include "
                    "surrounding lines to make it unique. Then re-run with "
                    "execute_code(entrypoint=...)."
                ),
            ),
        ]
    return tools


__all__ = ["make_code_execution_tools"]
