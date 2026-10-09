"""The `execute_code` tool — run/debug code in the sandbox (see code_execution.py)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from agent_runtime.tool_args import accept_null_defaults
from agent_runtime.extraction_flag import extraction_enabled

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


# The shape of every id the store mints (file_store: f"file_{uuid4().hex[:12]}"). Bounded on both
# sides so a longer identifier that merely contains one is not read as one.
_MINTED_FILE_ID = re.compile(r"(?<![A-Za-z0-9_])file_[0-9a-f]{12}(?![A-Za-z0-9_])")


def _file_ids_named_in(source: str) -> List[str]:
    """The file_ids the program names literally, first-seen order, each once.

    Every tool that makes a file (admin_boundary, overpass_search, add_map_layer, an upload)
    hands the model a file_id, and staging puts each input under that file_id as well as its
    filename. But it staged only what `input_files` listed, so a program that simply opened what
    it had been given, gpd.read_file("file_2272c8426ec9"), ran against an empty directory. In all
    three live turns of 2026-10-08 that was the first run: the model listed the directory, found
    nothing, and retried with `input_files`. Code that NAMES an id is asking for that file.

    Only minted ids are taken. Anything else the code mentions is a path or a filename, which a
    scan of program text cannot tell apart from a string it builds for some other purpose.
    """
    return list(dict.fromkeys(_MINTED_FILE_ID.findall(str(source or ""))))


def _readable_ids(ids: List[str]) -> Tuple[List[str], List[Dict[str, str]]]:
    """Split ids the code names into those THIS caller may read and those it may not.

    Through ``get_file_record`` only, the lookup an explicit ``input_files`` id gets, with its
    owner check (Stage 30). Another user's id is refused the same way as one never minted. The
    path fallback ``_resolve_input_file`` has is deliberately not used here: a token found in
    program text is never read as a host path.
    """
    from agent_runtime.file_store import get_file_record

    readable: List[str] = []
    refused: List[Dict[str, str]] = []
    for fid in ids:
        if get_file_record(fid):
            readable.append(fid)
        else:
            refused.append({"ref": fid, "error": "named in the code, but no file you can read "
                                                 "has this file_id"})
    return readable, refused


def _names_file(source: str, filename: str) -> bool:
    """Whether the program text names ``filename`` as a whole file name.

    Bounded so a longer name that merely contains it does not count: 'my_data.csv' and
    'data.csv.bak' do not name data.csv. A directory in front ('./data.csv', 'out/data.csv')
    still does, since the file may be what that path means, and staging it costs nothing if not.
    """
    if not filename:
        return False
    pattern = r"(?<![\w.\-])" + re.escape(filename) + r"(?![\w\-]|\.\w)"
    return re.search(pattern, source) is not None


def _conversation_files_named_in(source: str, taken: set, workspace: Optional[Path]
                                 ) -> List[Dict[str, Any]]:
    """Files of THIS conversation and THIS owner whose filename the program names, one record
    per name (the newest), skipping names the run will already have.

    The filename half of what ``_file_ids_named_in`` does for ids. A tool's answer names its
    file both ways, and code opens either: "Champaign_County.geojson" failed a first run just as
    "file_2272c8426ec9" did, in the 2026-10-08 replays and PR #85's. A filename is not unique the
    way an id is, so it is looked up only among this conversation's own files, through the
    store's scoped lookup (``find_files``, Stages 21/30) and then the owner compared exactly:
    never the legacy pool, never another conversation of the same user, never a raw path.

    Skipped, so the scan only fills a gap and never changes which file a name means:
    * a name already taken by a listed or attached input, whose own allocation stands;
    * a name the conversation's workspace already holds. A file there is one an earlier run
      WROTE (staged inputs are never copied back unchanged), and the program means that one.
    """
    from agent_runtime.file_store import current_owner, current_session, find_files, record_owner

    session = current_session()
    if not session or not source:
        return []
    owner = current_owner()
    chosen: Dict[str, Dict[str, Any]] = {}
    # Newest first, so the first record seen under a name is the one a lookup by that name finds.
    for record in find_files(session=session, include_unowned=False, limit=500):
        filename = str(record.get("filename") or "")
        if (not filename or filename in chosen or filename in taken
                or record.get("session") != session or record_owner(record) != owner):
            continue
        if workspace is not None and (workspace / filename).exists():
            continue
        if _names_file(source, filename):
            chosen[filename] = record
    return list(chosen.values())


# Appended to execute_code's description only while the extraction bundle is on
# (agent_runtime/extraction_flag.py): it describes the gate and the mounted library, and a
# model told about a library that is not mounted guesses at it.
_EXTRACTION_NOTE = (
    # Stated here because a peer that skips kb_method_search will otherwise GUESS the
    # package name: one run guessed `from method_library import ...` (the host
    # directory name) and failed with ModuleNotFoundError. The importable package is
    # `iguide_methods`, whatever the mount is called.
    # The gate can only check a UNIT if the run declares one; nothing in a frame
    # distinguishes 21500 metres from 21500 feet.
    "VERIFICATION: a deterministic invariant gate inspects your live frames after the "
    "run (projected-CRS-before-measuring, entirely-null columns, join cardinality) and "
    "returns findings in `verification`. If it reports a failure, FIX AND RE-RUN — a "
    "failed gate means the reported numbers are not verified and the answer will say "
    "so. For any number your answer will quote, ASSIGN a module-level dict "
    "IGUIDE_OUTPUTS = {\"name\": {\"value\": 25000, \"unit\": \"metres\"}} "
    "(optional \"min\"/\"max\" get range-checked); the gate reads the variable, so "
    "printing it checks nothing, and a null unit blocks verification. "
    "The I-GUIDE METHOD LIBRARY is importable in the sandbox as the package "
    "`iguide_methods` — extracted, independently callable functions from platform "
    "elements, already present with NO install and NO network. Get an exact, "
    "version-pinned import line from `kb_method_search` / `get_method_contract` "
    "rather than guessing a module path, and still declare the method's own "
    "`dependencies` (e.g. geopandas), which are NOT preinstalled."
)


def make_code_execution_tools(
    executor: Optional[Any] = None,
    default_input_file_ids: Optional[List[str]] = None,
    session_id: Optional[str] = None,
) -> List[Any]:
    """Build the `execute_code` StructuredTool (container-per-run sandbox).

    ``default_input_file_ids`` are the files attached to the current conversation;
    they are auto-staged into EVERY run so the model can read them without naming
    them, and are unioned (deduped) with any explicit ``input_files`` it passes.

    ``session_id`` makes the sandbox workspace PERSIST across calls within a turn, so a
    multi-step workflow can build state (step 2 reads what step 1 wrote). Without it every
    call gets a throwaway directory, which is what made multi-step analysis impossible.
    """
    from langchain_core.tools import StructuredTool

    from agent_runtime.code_execution import get_code_executor

    default_ids = [str(x).strip() for x in (default_input_file_ids or []) if str(x).strip()]

    def execute_code(
        code: str = "",
        language: str = "python",
        timeout_seconds: Optional[int] = None,
        dependencies: Optional[List[str]] = None,
        input_files: Optional[List[str]] = None,
        label: Optional[str] = None,
        entrypoint: Optional[str] = None,
    ) -> str:
        ex = executor or get_code_executor()

        # Union: conversation-attached files (auto) first, then any explicitly
        # named files, order-preserving and deduped.
        explicit = [str(r).strip() for r in (input_files or []) if str(r).strip()]
        # …then every file_id the program itself names. An entrypoint run has no inline
        # source, so its file is what gets read; a file the workspace refuses names nothing.
        source = code or ""
        if entrypoint and not source.strip() and session_id:
            try:
                from agent_runtime.code_execution import resolve_workspace_file
                source = resolve_workspace_file(session_id, entrypoint).read_text(
                    encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                source = ""
        listed = set(default_ids) | set(explicit)
        named, refused = _readable_ids(
            [fid for fid in _file_ids_named_in(source) if fid not in listed])
        refs = list(dict.fromkeys([*default_ids, *explicit, *named]))
        staging, staged_info, input_errors, skipped = _build_staging(refs)
        # …and every file of this conversation the program names by FILENAME, unless a name is
        # one the run already has. Looked up after the first allocation, so it can see which.
        taken = {str(n) for info in staged_info for n in info.get("available_as") or []}
        workspace = None
        if session_id:
            from agent_runtime.code_execution import session_workspace_dir
            workspace = session_workspace_dir(session_id)
        by_name = {str(r["file_id"]): r for r in _conversation_files_named_in(source, taken, workspace)}
        if by_name:
            refs = list(dict.fromkeys([*refs, *by_name]))
            staging, staged_info, input_errors, skipped = _build_staging(refs)
        for info in staged_info:
            if info.get("ref") in named or info.get("ref") in by_name:
                info["staged_because"] = "named in the code"
        input_errors = [*input_errors, *refused]

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
        if session_id:
            # Tell the model the workspace persists; otherwise it will not use it and will
            # keep re-deriving state it already computed.
            payload["workspace"] = {
                "persistent": True,
                "note": "Files you write persist for the rest of this conversation; a later "
                        "execute_code call can read them from the working directory.",
            }
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
            "cluster or difference its vectors). A file_id your code names literally (e.g. "
            "gpd.read_file(\"file_2272c8426ec9\")) is staged too, without listing it, and so "
            "is a file this conversation's tools made that your code opens by filename. "
            "Writing under an input's name keeps your output. "
            "Use this to RUN and DEBUG code: run, read "
            "stdout/stderr, fix, re-run. Files you write persist in this conversation's working "
            "directory, so a later run can open what an earlier one produced and keep building "
            "on it (the container itself is fresh each time). `label` is a short slug for what this particular run "
            "does (e.g. \"csv_to_geojson\", \"rivers_buffer\") and becomes the saved source's "
            "filename — several runs in one conversation otherwise arrive as identically-named "
            "downloads; name the files your code writes for their contents too, for the same "
            "reason. To change one part of a program you already wrote, do NOT re-send the "
            "whole thing: write it to a named file with write_workspace_file, then run it with "
            "`entrypoint` (e.g. entrypoint=\"main.py\", no `code`), and fix it with "
            "edit_workspace_file between runs. "
            + (_EXTRACTION_NOTE if extraction_enabled() else "")
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
