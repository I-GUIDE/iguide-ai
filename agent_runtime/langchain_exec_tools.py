"""The `execute_code` tool — run/debug code in the sandbox (see code_execution.py)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
        if len(staged_info) >= max_files:
            skipped.append({"ref": str(ref), "reason": "max input files exceeded", "limit": max_files})
            continue
        if total_bytes + size > max_bytes:
            skipped.append({"ref": str(ref), "reason": "max total input size exceeded",
                            "limit_bytes": max_bytes, "size_bytes": size})
            continue

        seen_sources.add(src)
        total_bytes += size
        filename = (record or {}).get("filename") or host_path.name
        file_id = (record or {}).get("file_id")
        names = list(dict.fromkeys(n for n in (file_id, filename) if n))
        for dest in names:
            staging.append({"source": src, "dest": dest})
        staged_info.append({"ref": str(ref), "file_id": file_id, "filename": filename, "available_as": names})

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

    ``session_id`` makes the sandbox workspace PERSIST across calls within a turn, so a
    multi-step workflow can build state (step 2 reads what step 1 wrote). Without it every
    call gets a throwaway directory, which is what made multi-step analysis impossible.
    """
    from langchain_core.tools import StructuredTool

    from agent_runtime.code_execution import get_code_executor

    default_ids = [str(x).strip() for x in (default_input_file_ids or []) if str(x).strip()]

    def execute_code(
        code: str,
        language: str = "python",
        timeout_seconds: Optional[int] = None,
        dependencies: Optional[List[str]] = None,
        input_files: Optional[List[str]] = None,
        tier: Optional[str] = None,
    ) -> str:
        ex = executor or get_code_executor()

        # Union: conversation-attached files (auto) first, then any explicitly
        # named files, order-preserving and deduped.
        explicit = [str(r).strip() for r in (input_files or []) if str(r).strip()]
        refs = list(dict.fromkeys([*default_ids, *explicit]))
        staging, staged_info, input_errors, skipped = _build_staging(refs)

        # timeout_seconds defaults to None, NOT to DEFAULT_TIMEOUT. It used to default to
        # 60, which is truthy, so it would have overridden the execution tier's timeout on
        # every single call and the tiers would have had no effect at all.
        result = ex.execute(
            code,
            language=language,
            timeout=timeout_seconds,
            dependencies=dependencies,
            input_files=staging,
            session_id=session_id,
            tier=tier,
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

    tool = StructuredTool.from_function(
        func=execute_code,
        name="execute_code",
        description=(
            "Execute code in an isolated, sandboxed container and return JSON with "
            "exit_code, stdout, stderr, timed_out, the executed `code`, `installed`, and "
            "`artifacts` (the source is saved as a downloadable `executed_code.py`, plus "
            "any files the run wrote). Pass `dependencies` (a list of pip specs, e.g. "
            "[\"numpy\", \"pandas==2.2\"]) to install third-party packages before running — "
            "they are installed with network in a separate step, then the code runs with NO "
            "network. Files attached to this conversation are AUTOMATICALLY available in the "
            "working directory under both their file_id and their original filename (e.g. "
            "open('data.csv') or pd.read_csv('data.csv')). To read any other uploaded file, "
            "add its file_id to `input_files`. Use this to RUN and DEBUG code: run, read "
            "stdout/stderr, fix, re-run. "
            "The working directory PERSISTS across calls in this conversation, so build a "
            "multi-step analysis incrementally: write an intermediate result to a file in "
            "one call and read it in the next instead of recomputing it. Installed "
            "`dependencies` also persist, so ask for them once. "
            "Set `tier` to size the run: 'quick' (60s/512MB) for a small check, 'standard' "
            "(300s/2GB, the default) for real analysis, 'heavy' (900s/6GB) for large "
            "geospatial joins where available. Only pass `timeout_seconds` to override the "
            "tier deliberately. "
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
            "so. For any number your answer will quote, declare it as "
            "IGUIDE_OUTPUTS = {\"name\": {\"value\": 25000, \"unit\": \"metres\"}} "
            "(optional \"min\"/\"max\" get range-checked); a null unit blocks verification. "
            "The I-GUIDE METHOD LIBRARY is importable in the sandbox as the package "
            "`iguide_methods` — extracted, independently callable functions from platform "
            "elements, already present with NO install and NO network. Get an exact, "
            "version-pinned import line from `kb_method_search` / `get_method_contract` "
            "rather than guessing a module path, and still declare the method's own "
            "`dependencies` (e.g. geopandas), which are NOT preinstalled."
        ),
    )
    return [tool]


__all__ = ["make_code_execution_tools"]
