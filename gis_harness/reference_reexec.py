"""Re-run a turn's recorded code under the gate's reference recorder, for the validators bench.

The reference recomputation of plan step 4b runs INSIDE the sandbox, on the run's live objects
(agent_runtime/reference_sandbox.py), so it cannot be replayed from what a turn recorded: no
archived run had the recorder. This re-executes the archived `execute_code` calls that could
have produced a reference record, in the code sandbox image with no network and with the
task's input files staged under the names the run used. It then puts the gate's `reference`
list into the recorded result, where the agent-side check reads it
(`reference_checks.from_sandbox`). Nothing else in the recorded turn changes. The run's
recorded stdout is kept, so a figure is still tied to what the agent saw.

Only calls whose code solves a linear system or calls esda's G_Local/Moran are re-run; no other
code can produce a record. Results are cached by the script's hash, so a second pass costs
nothing. No model is called.

`typed_coverage.py --reexec` (stage 48) re-runs every call of every CORRECT turn and keys the
result by archive path. The bench needs wrong turns most of all, from the state it replays, so
this is per call and state-driven. The staging rule is the same: a task file goes in under
every name in the recorded `input_files`.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_IMAGE = os.environ.get("GIS_HARNESS_REEXEC_IMAGE", "iguide-codeexec:reference-bench")
TRIGGER = re.compile(r"linalg\.(?:solve|lstsq)|\b(?:solve|lstsq)\(|G_Local|\bMoran\(")
_FILE_ID = re.compile(r"\bfile_[0-9a-f]{12}(?:__[\w.\-]+)?")


def _cache_dir() -> Path:
    d = Path(os.environ.get("GIS_HARNESS_REEXEC_CACHE")
             or Path.home() / ".cache" / "iguide" / "reference-reexec")
    d.mkdir(parents=True, exist_ok=True)
    return d


def task_inputs(task_id: str, data_root: Path) -> Dict[str, Path]:
    """The task's input files by filename, built by the harness's own dataset builders."""
    from gis_harness import datasets as D
    from gis_harness.tasks import TASKS

    task = next((t for t in TASKS if t.id == task_id.split(".")[0]), None)
    if task is None or task.dataset not in D.BUILDERS:
        return {}
    out = data_root / task.dataset
    if not out.exists() or not any(out.iterdir()):
        files, _ = D.build(task.dataset, out, data_root / "_cache")
        return {k: Path(v) for k, v in files.items()}
    return {p.name: p for p in out.iterdir() if p.is_file()}


def _body(content: Any) -> Optional[Dict[str, Any]]:
    try:
        body = json.loads(content) if isinstance(content, str) else content
    except (TypeError, ValueError):
        return None
    return body if isinstance(body, dict) else None


def _stage(work: Path, code: str, body: Dict[str, Any], inputs: Dict[str, Path]) -> None:
    """Every name the run could open a task file by: its filename, its file_id, the names the
    staging step reported (`available_as`), and any `file_<id>__<name>` the code spells."""
    names: Dict[str, Path] = {}
    for item in body.get("input_files") or []:
        if not isinstance(item, dict):
            continue
        src = inputs.get(str(item.get("filename") or ""))
        if src is None:
            continue
        for alias in [item.get("filename"), item.get("file_id"), *(item.get("available_as") or [])]:
            if alias:
                names[str(alias)] = src
    single = next(iter(inputs.values())) if len(inputs) == 1 else None
    for token in _FILE_ID.findall(code):
        if token in names:
            continue
        tail = token.split("__", 1)[1] if "__" in token else ""
        src = inputs.get(tail) or single
        if src is not None:
            names[token] = src
    for fname, src in inputs.items():
        names.setdefault(fname, src)
    for alias, src in names.items():
        dest = work / alias
        if "/" not in alias and not dest.exists():
            shutil.copyfile(src, dest)


def reexec(code: str, body: Dict[str, Any], inputs: Dict[str, Path], *,
           image: str = DEFAULT_IMAGE, timeout: int = 240) -> Dict[str, Any]:
    """`{reference: [...], exit_code, error?}` for one recorded run, re-executed."""
    from agent_runtime import sandbox_verify as SV

    script = SV.prologue_source() + "\n" + code + SV.epilogue_source()
    digest = hashlib.sha256((image + "\0" + script + "\0" + json.dumps(
        sorted((k, str(v)) for k, v in inputs.items()))).encode()).hexdigest()[:24]
    cached = _cache_dir() / f"{digest}.json"
    if cached.exists():
        return json.loads(cached.read_text())
    work = Path(tempfile.mkdtemp(prefix="reexec_"))
    try:
        _stage(work, code, body, inputs)
        (work / "script.py").write_text(script)
        try:
            proc = subprocess.run(
                ["docker", "run", "--rm", "--network", "none", "--memory", "4g",
                 "-v", f"{work}:/work", "-w", "/work", "-e", "HOME=/tmp", "-e",
                 "MPLCONFIGDIR=/tmp/mpl", image, "python", "script.py"],
                capture_output=True, text=True, timeout=timeout)
            exit_code, err = proc.returncode, proc.stderr[-400:]
        except subprocess.TimeoutExpired:
            exit_code, err = None, "timed out"
        report: Dict[str, Any] = {}
        checks = work / SV.CHECKS_FILENAME
        if checks.exists():
            try:
                report = json.loads(checks.read_text())
            except ValueError:
                report = {}
        out = {"reference": report.get("reference") or [], "exit_code": exit_code,
               **({"error": err} if exit_code else {})}
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if out.get("exit_code") is not None:
        cached.write_text(json.dumps(out))
    return out


def augment_state(state: Dict[str, Any], task_id: str, data_root: Path, *,
                  image: str = DEFAULT_IMAGE) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """The turn's state with each re-run call's `verification.reference` filled in, and counts
    of what was re-run."""
    from agent_runtime.code_execution import _compact_reference

    st = copy.deepcopy(state)
    info = {"rerun": 0, "recorded": 0, "failed": 0}
    inputs: Optional[Dict[str, Path]] = None
    for key in ("analysis_results", "code_result"):
        peer = st.get(key)
        if not isinstance(peer, dict):
            continue
        codes = {c.get("id"): (c.get("args") or {}).get("code")
                 for c in peer.get("tool_calls") or []
                 if isinstance(c, dict) and c.get("name") == "execute_code"
                 and isinstance(c.get("args"), dict)}
        for res in peer.get("tool_results") or []:
            if not isinstance(res, dict) or res.get("name") != "execute_code":
                continue
            code = codes.get(res.get("tool_call_id"))
            body = _body(res.get("content"))
            if not code or body is None or not body.get("ok") or not TRIGGER.search(code):
                continue
            if inputs is None:
                inputs = task_inputs(task_id, data_root)
            out = reexec(code, body, inputs, image=image)
            info["rerun"] += 1
            if out.get("exit_code"):
                info["failed"] += 1
            refs = [_compact_reference(r) for r in out.get("reference") or []
                    if isinstance(r, dict)]
            if refs:
                info["recorded"] += 1
                ver = dict(body.get("verification") or {})
                ver["reference"] = refs[:8]
                body["verification"] = ver
                res["content"] = json.dumps(body)
    return st, info


def resolver(task_id: str, data_root: Path, run_dir: Optional[Path] = None):
    """A file_id -> path function for replaying the tool-side references: the file the run's
    server stored, when the archive kept it (`server_files/`), else the task's input when the
    task has exactly one. An id that resolves to neither raises, and that reference is skipped."""
    inputs = task_inputs(task_id, data_root)
    single = next(iter(inputs.values())) if len(inputs) == 1 else None

    def resolve(file_id: str) -> Path:
        if run_dir is not None:
            for sub in ("outputs", "uploads"):
                hits = sorted((run_dir / "server_files" / sub).glob(f"{file_id}*"))
                if hits:
                    return hits[0]
        if file_id in inputs:
            return inputs[file_id]
        if single is not None:
            return single
        raise FileNotFoundError(file_id)

    return resolve
