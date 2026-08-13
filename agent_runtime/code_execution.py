"""Sandboxed code execution for the agents (container-per-run).

Lets the code / analysis agents **run and debug** code in an isolated sandbox.
The default backend launches a **fresh, hardened container per run**:

  docker run --rm --network none --read-only --cap-drop ALL
    --security-opt no-new-privileges --memory … --cpus … --pids-limit …
    --tmpfs /tmp --user <host-uid> -v <workdir>:/work:rw <image> python /work/script.py

i.e. no network, read-only root filesystem, dropped capabilities, no privilege
escalation, CPU/memory/pid limits, a wall-clock timeout, and the *only* writable
location is a throwaway work dir whose files are persisted as output artifacts.

A ``local`` subprocess backend exists for development only — it is **NOT a
sandbox** (it runs on the host) and must be opted into explicitly.

Gated by ``AGENT_CODE_EXEC`` (off by default). Treat all executed code as
untrusted.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_IMAGE = os.getenv("AGENT_CODE_EXEC_IMAGE", "python:3.11-slim")
DEFAULT_TIMEOUT = int(os.getenv("AGENT_CODE_EXEC_TIMEOUT", "60"))
DEFAULT_INSTALL_TIMEOUT = int(os.getenv("AGENT_CODE_EXEC_INSTALL_TIMEOUT", "300"))
DEFAULT_MEMORY = os.getenv("AGENT_CODE_EXEC_MEMORY", "512m")
# The deps-install phase needs more headroom than execution: building/installing
# the scientific stack (numpy/pandas/scipy) under the 512m exec limit was getting
# OOM-killed (exit 137). Give install its own, larger budget.
DEFAULT_INSTALL_MEMORY = os.getenv("AGENT_CODE_EXEC_INSTALL_MEMORY", "1g")
DEFAULT_CPUS = os.getenv("AGENT_CODE_EXEC_CPUS", "1.0")
DEFAULT_PIDS = os.getenv("AGENT_CODE_EXEC_PIDS", "256")
MAX_OUTPUT_CHARS = 20_000
MAX_ARTIFACTS = 20
MAX_DEPS = 50
# Deps install under this dir inside the work dir; added to PYTHONPATH for the run.
DEPS_DIRNAME = ".deps"
# pip's scratch space during install — kept on the host-backed work dir (real disk)
# rather than the small in-container tmpfs, so installing big wheels (numpy/matplotlib)
# doesn't run out of space.
PIPTMP_DIRNAME = ".piptmp"
# When set, per-run sandbox work dirs are created under this path instead of the
# system temp dir. This is REQUIRED for Docker-out-of-Docker (agent runs in a
# container, shelling out to the host daemon): the path must exist at the SAME
# absolute location on the host (a shared bind mount), so the `docker run -v <work>:/work`
# the agent issues resolves to a real host directory the daemon can mount.
WORK_ROOT_ENV = "AGENT_CODE_EXEC_WORK_ROOT"

# --- persistent per-session workspaces -------------------------------------------------
# Runs used to get a fresh mkdtemp() that was rmtree'd in a `finally`, so a workspace
# never outlived one call and step 2 could not read step 1's output. That was the hard
# ceiling on multi-step workflows: no matter how capable the model, "load the data, then
# join it, then map it" could not be expressed across calls.
SESSIONS_DIRNAME = "sessions"
# The extracted method library is mounted READ-ONLY so the agent can
# `from iguide_methods import ...` and compose validated units in Python instead of chaining
# tool calls. Read-only because the library is generated from ingested elements: a run must
# never be able to edit the thing later runs will trust.
METHOD_LIBRARY_DIRNAME = "method_library"
METHOD_LIBRARY_MOUNT = "/opt/iguide_methods"
# Records (mtime_ns, size) per file so a run persists only what it actually produced.
ARTIFACT_INDEX_FILENAME = ".iguide_artifact_index.json"
# Reclamation: workspaces are swept by age, and capped in size so one runaway session
# cannot fill the shared work-root bind mount.
WORKSPACE_TTL_HOURS = float(os.getenv("AGENT_CODE_EXEC_WORKSPACE_TTL_HOURS", "24"))
WORKSPACE_MAX_MB = float(os.getenv("AGENT_CODE_EXEC_WORKSPACE_MAX_MB", "2048"))

# Execution tiers. 512m/1cpu/60s is a quick-tool budget, not an analysis budget: a
# county-level spatial join on a national layer OOMs or times out under it, which teaches
# the model to avoid real computation. `heavy` is gated because it is a real resource
# commitment on a shared host.
EXEC_TIERS: Dict[str, Dict[str, str]] = {
    "quick":    {"timeout": "60",  "memory": "512m", "cpus": "1.0"},
    "standard": {"timeout": "300", "memory": "2g",   "cpus": "2.0"},
    "heavy":    {"timeout": "900", "memory": "6g",   "cpus": "4.0"},
}
DEFAULT_TIER = os.getenv("AGENT_CODE_EXEC_DEFAULT_TIER", "standard")


def _allow_heavy() -> bool:
    return (os.getenv("AGENT_CODE_EXEC_ALLOW_HEAVY", "0") or "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_tier(tier: Optional[str]) -> Tuple[str, Dict[str, str]]:
    """Return (tier_name, limits). Unknown or ungated tiers fall back, never raise."""
    name = (tier or DEFAULT_TIER or "standard").strip().lower()
    if name not in EXEC_TIERS:
        name = "standard" if "standard" in EXEC_TIERS else "quick"
    if name == "heavy" and not _allow_heavy():
        logger.info("execution tier 'heavy' requested but AGENT_CODE_EXEC_ALLOW_HEAVY is off; using 'standard'")
        name = "standard"
    return name, dict(EXEC_TIERS[name])


def _safe_session_id(session_id: Optional[str]) -> str:
    """Sanitise a session id for use as a single path segment.

    Modelled on ``rag_pipeline/qgis_headless_tools.py:153`` but NOT identical, because that
    version is unsafe for this use: its allowlist ``[^A-Za-z0-9_.-]`` permits dots, so the
    id ``".."`` survives sanitisation untouched. ``<sessions_root>/..`` is the work root
    itself — and since ``sweep_workspaces`` removes expired workspaces, an id of ``".."``
    could take the whole shared work root with it.

    So any purely-dot name is rejected outright. (``../../etc/passwd`` was already safe: the
    separators become underscores, leaving one harmless segment.)
    """
    value = (session_id or "default").strip() or "default"
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:120]
    if not cleaned or set(cleaned) <= {"."}:      # "", ".", "..", "..." -> path-special
        return "default"
    return cleaned


def _sessions_root() -> Optional[Path]:
    root = _work_root()
    base = Path(root) if root else Path(tempfile.gettempdir())
    return base / SESSIONS_DIRNAME


def session_work_dir(session_id: str) -> Path:
    """The persistent workspace for *session_id*, created if absent.

    Defence in depth: the resolved path is asserted to sit inside the sessions root, so a
    future change to the sanitiser cannot silently reintroduce an escape.
    """
    root = (_sessions_root() or Path(tempfile.gettempdir()))
    root.mkdir(parents=True, exist_ok=True)
    path = root / _safe_session_id(session_id)
    root_resolved = root.resolve()
    if not str(path.resolve()).startswith(str(root_resolved)):
        logger.error("session workspace %r escaped the sessions root; using 'default'", session_id)
        path = root / "default"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _deps_marker_path(work: Path) -> Path:
    return work / DEPS_DIRNAME / ".installed.json"


def _deps_satisfied(work: Path, dependencies: List[str]) -> bool:
    """True when every requested dep was already installed into this workspace's .deps.

    Only meaningful for a persistent workspace: a throwaway dir never has a prior install.
    Compared as an exact set of the sanitised specs, so a version change reinstalls.
    """
    try:
        recorded = set(json.loads(_deps_marker_path(work).read_text(encoding="utf-8")) or [])
    except (OSError, ValueError):
        return False
    return bool(recorded) and set(dependencies).issubset(recorded)


def _record_deps(work: Path, dependencies: List[str]) -> None:
    marker = _deps_marker_path(work)
    try:
        prior = set(json.loads(marker.read_text(encoding="utf-8")) or [])
    except (OSError, ValueError):
        prior = set()
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(sorted(prior | set(dependencies))), encoding="utf-8")
    except OSError:
        pass


def method_library_dir() -> Optional[Path]:
    """Host path of the generated method library, or None when nothing is ingested yet."""
    override = (os.getenv("AGENT_METHOD_LIBRARY_DIR") or "").strip()
    if override:
        p = Path(override).expanduser()
        return p if p.is_dir() else None
    try:
        from agent_runtime.file_store import storage_root
        p = Path(storage_root()) / METHOD_LIBRARY_DIRNAME
    except Exception:
        return None
    return p if p.is_dir() else None


def _dir_size_mb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total / (1024 * 1024)


def sweep_workspaces(*, ttl_hours: Optional[float] = None) -> Dict[str, int]:
    """Remove session workspaces older than the TTL. Returns a small summary.

    Cheap and best-effort: called opportunistically before creating a workspace rather than
    on a timer, because there is no scheduler in this process.
    """
    ttl = WORKSPACE_TTL_HOURS if ttl_hours is None else ttl_hours
    root = _sessions_root()
    summary = {"examined": 0, "removed": 0}
    if not root or ttl <= 0:
        return summary
    cutoff = time.time() - (ttl * 3600)
    try:
        entries = list(root.iterdir())
    except OSError:
        return summary
    for entry in entries:
        if not entry.is_dir():
            continue
        summary["examined"] += 1
        try:
            if entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
                summary["removed"] += 1
        except OSError:
            continue
    return summary

# A conservative pip requirement spec: name[extras]version-specifiers. No flags,
# no whitespace, no shell/path characters — deps are passed as argv (never a shell).
_DEP_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._-]+\])?"
    r"([<>=!~]=?[A-Za-z0-9._*+!-]+(,[<>=!~]=?[A-Za-z0-9._*+!-]+)*)?$"
)


# Heavy frameworks that take minutes to install and are rarely needed for the
# geospatial/data tasks this sandbox serves. Rejected unless an explicit
# AGENT_CODE_EXEC_PIP_ALLOW opts them in — this stops an ungrounded request (e.g.
# "forecast with an LSTM") from burning the whole install budget before the agent
# can recover. Tune via AGENT_CODE_EXEC_PIP_DENY (comma-separated, replaces this set).
_DEFAULT_PIP_DENY = {
    "tensorflow", "tensorflow-gpu", "tensorflow-cpu", "torch", "torchvision",
    "torchaudio", "jax", "jaxlib", "transformers", "keras", "mxnet", "paddlepaddle",
}


def _pip_denylist() -> set:
    raw = os.getenv("AGENT_CODE_EXEC_PIP_DENY")
    if raw is None:
        return _DEFAULT_PIP_DENY
    return {x.strip().lower() for x in raw.split(",") if x.strip()}


def _sanitize_deps(dependencies: Optional[List[Any]]) -> Tuple[List[str], List[str]]:
    """Return (allowed, rejected) pip specs. Rejects flags/odd specs; honors an explicit
    allowlist (AGENT_CODE_EXEC_PIP_ALLOW), else a denylist of heavy frameworks."""
    if not dependencies:
        return [], []
    allow = {x.strip().lower() for x in (os.getenv("AGENT_CODE_EXEC_PIP_ALLOW") or "").split(",") if x.strip()}
    deny = _pip_denylist()
    allowed: List[str] = []
    rejected: List[str] = []
    for raw in dependencies:
        spec = str(raw).strip()
        if not spec or spec.startswith("-") or any(c.isspace() for c in spec) or not _DEP_RE.match(spec):
            rejected.append(spec)
            continue
        name = re.split(r"[<>=!~\[]", spec, maxsplit=1)[0].strip().lower()
        if allow:
            if name not in allow:
                rejected.append(spec)
                continue
        elif name in deny:
            rejected.append(spec)
            continue
        allowed.append(spec)
    return allowed[:MAX_DEPS], rejected


def artifacts_enabled() -> bool:
    from agent_runtime.artifacts import artifacts_enabled as _enabled
    return _enabled()


def invariant_gate_enabled() -> bool:
    """Whether to append the in-sandbox invariant checks. **On by default.**

    Default-on is deliberate: the failure it catches — a distance computed in a geographic
    CRS — produces a plausible number and no error, so a gate that is off by default protects
    nobody. Set ``AGENT_INVARIANT_GATE`` to a falsy value to disable it. The epilogue cannot
    fail a run (every check is guarded and the writer swallows OSError), so the cost of
    leaving it on is one JSON file.
    """
    return (os.getenv("AGENT_INVARIANT_GATE", "1") or "").strip().lower() not in {
        "0", "false", "no", "off"}


def is_code_exec_enabled() -> bool:
    """Whether code execution is enabled. **On by default**; set ``AGENT_CODE_EXEC`` to a falsy
    value (0/false/no/off) to disable the sandboxed ``execute_code`` tool."""
    return (os.getenv("AGENT_CODE_EXEC", "1") or "").strip().lower() not in {"0", "false", "no", "off"}


def _clip(text: Any, limit: int = MAX_OUTPUT_CHARS) -> str:
    s = "" if text is None else str(text)
    return s if len(s) <= limit else (s[:limit] + f"\n…[truncated {len(s) - limit} chars]")


@dataclass
class ExecResult:
    exit_code: Optional[int]
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    error: Optional[str] = None
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    backend: str = ""
    code: str = ""   # the executed source (also saved as a downloadable artifact)
    installed: List[str] = field(default_factory=list)  # pip deps installed before the run
    # Findings from the in-sandbox invariant gate. Empty dict = the gate did not run.
    verification: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and not self.timed_out and self.exit_code == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
            "error": self.error,
            "code": self.code,
            "installed": self.installed,
            "artifacts": self.artifacts,
            "backend": self.backend,
            # Surfaced INSIDE the tool result, not appended to the answer afterwards, so the
            # model can react in-loop: reproject and re-run rather than caveat a wrong number.
            "verification": self.verification,
        }


def _read_checks(work: Path) -> Dict[str, Any]:
    """Load the invariant gate's report, if it wrote one.

    Absent is not a failure: the gate may be disabled, or the run may have died before the
    epilogue. Absent and "checked, all fine" must stay distinguishable, so this returns {} for
    absent rather than a synthetic pass.
    """
    from agent_runtime.sandbox_verify import CHECKS_FILENAME

    path = work / CHECKS_FILENAME
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    # Keep the payload small: the model needs the verdict and what failed, not every pass.
    findings = [f for f in (data.get("findings") or [])
                if isinstance(f, dict) and f.get("status") != "pass"]
    return {"verdict": data.get("verdict"), "counts": data.get("counts") or {},
            "inspected": data.get("inspected") or [], "findings": findings[:12],
            **({"error": data["error"]} if data.get("error") else {})}


def _artifact_index_path(work: Path) -> Path:
    return work / ARTIFACT_INDEX_FILENAME


def _read_artifact_index(work: Path) -> Dict[str, List[int]]:
    try:
        return json.loads(_artifact_index_path(work).read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return {}


def _write_artifact_index(work: Path, index: Dict[str, List[int]]) -> None:
    try:
        _artifact_index_path(work).write_text(json.dumps(index), encoding="utf-8")
    except OSError:
        pass


def _persist_artifacts(work: Path, exclude: set, *, incremental: bool = False) -> List[Dict[str, Any]]:
    """Persist files the run created in *work* to the agent file store.

    With ``incremental=True`` only files that are NEW or CHANGED since the previous run in
    this workspace are persisted, judged by (mtime_ns, size) against a small on-disk index.

    This is not an optimisation — it is required for a persistent workspace to be usable.
    This function walks the whole tree and stops at ``MAX_ARTIFACTS`` (20), so once a
    workspace survives between calls, step 1's leftovers occupy the budget in sorted-path
    order and step 5's actual output silently never gets persisted. Incremental persistence
    and the persistent workspace therefore have to land together.
    """
    from agent_runtime.file_store import create_output_file_from_path

    index = _read_artifact_index(work) if incremental else {}
    new_index: Dict[str, List[int]] = dict(index)
    artifacts: List[Dict[str, Any]] = []
    skipped_unchanged = 0

    for path in sorted(work.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(work)
        rel_str = str(rel)
        if rel_str in exclude or rel_str == ARTIFACT_INDEX_FILENAME:
            continue
        if rel.parts and rel.parts[0] in {"__pycache__", DEPS_DIRNAME, PIPTMP_DIRNAME}:
            continue
        try:
            stat = path.stat()
            stamp = [stat.st_mtime_ns, stat.st_size]
        except OSError:
            continue
        if incremental:
            if index.get(rel_str) == stamp:
                skipped_unchanged += 1
                continue
            new_index[rel_str] = stamp
        if len(artifacts) >= MAX_ARTIFACTS:
            break
        try:
            rec = create_output_file_from_path(path, filename=path.name)
            artifacts.append(
                {
                    "file_id": rec["file_id"],
                    "filename": rec["filename"],
                    "download_url": rec.get("download_url"),
                    "size_bytes": rec.get("size_bytes"),
                }
            )
        except Exception:
            continue

    if incremental:
        _write_artifact_index(work, new_index)
        if skipped_unchanged:
            logger.debug("skipped %d unchanged file(s) in %s", skipped_unchanged, work)
    return artifacts


def _persist_source(code: str, *, filename: str = "executed_code.py") -> List[Dict[str, Any]]:
    """Save the executed source as a downloadable output artifact."""
    from agent_runtime.file_store import create_output_file

    try:
        rec = create_output_file(filename, code or "")
        return [{
            "file_id": rec["file_id"],
            "filename": rec["filename"],
            "download_url": rec.get("download_url"),
            "size_bytes": rec.get("size_bytes"),
            "kind": "source",
        }]
    except Exception:
        return []


def _stage_inputs(work: Path, input_files: Optional[List[Dict[str, str]]]) -> Tuple[List[str], List[Dict[str, str]]]:
    """Copy requested input files into *work* so the sandboxed code can read them.

    Each spec is ``{"source": <host_path>, "dest": <relative_name>}``.  ``dest`` must
    be a plain filename (no path separators / traversal) that stays inside *work* —
    *work* is the only writable mount, so files placed here appear at ``/work/<dest>``
    inside the container (the run's working directory).

    Returns ``(staged_dest_names, errors)``.
    """
    staged: List[str] = []
    errors: List[Dict[str, str]] = []
    work_resolved = work.resolve()
    for spec in input_files or []:
        source = str((spec or {}).get("source") or "").strip()
        dest = str((spec or {}).get("dest") or "").strip()
        if not source or not dest:
            continue
        if "/" in dest or "\\" in dest or dest in {".", ".."}:
            errors.append({"dest": dest, "error": "invalid destination name"})
            continue
        target = (work / dest).resolve()
        if target.parent != work_resolved:
            errors.append({"dest": dest, "error": "destination escapes work dir"})
            continue
        src = Path(source).expanduser()
        if not src.is_file():
            errors.append({"source": source, "error": "source file not found"})
            continue
        try:
            shutil.copyfile(src, target)
            staged.append(dest)
        except Exception as exc:  # pragma: no cover - defensive
            errors.append({"source": source, "dest": dest, "error": f"{type(exc).__name__}: {exc}"})
    return staged, errors


def _work_root() -> Optional[str]:
    """Directory under which per-run work dirs are created (None = system temp).

    Set via ``AGENT_CODE_EXEC_WORK_ROOT`` for Docker-out-of-Docker so the work dir
    sits on a host-shared path the host daemon can bind-mount.
    """
    root = (os.getenv(WORK_ROOT_ENV) or "").strip()
    if not root:
        return None
    try:
        os.makedirs(root, exist_ok=True)
        return root
    except OSError:
        return None


def _host_user() -> Optional[str]:
    getuid = getattr(os, "getuid", None)
    getgid = getattr(os, "getgid", None)
    if callable(getuid) and callable(getgid):
        return f"{getuid()}:{getgid()}"
    return None


# ---------------------------------------------------------------------------
# Executors
# ---------------------------------------------------------------------------

class CodeExecutor:
    """Base: write code to a throwaway dir, run it, persist artifacts, clean up."""

    backend = "base"

    def execute(self, code: str, *, language: str = "python", timeout: Optional[int] = None,
                env: Optional[Dict[str, str]] = None,
                dependencies: Optional[List[Any]] = None,
                input_files: Optional[List[Dict[str, str]]] = None,
                session_id: Optional[str] = None,
                tier: Optional[str] = None) -> ExecResult:
        """Run *code* in the sandbox.

        With a ``session_id`` the workspace PERSISTS between calls, so a multi-step workflow
        can build state: step 2 reads what step 1 wrote. Without one the old behaviour is
        kept exactly — a throwaway dir removed afterwards — because sessionless runs sharing
        a directory would leak between unrelated turns.

        ``tier`` selects the resource budget (quick | standard | heavy); see ``EXEC_TIERS``.
        """
        if (language or "python").lower() != "python":
            return ExecResult(exit_code=None, error=f"unsupported language: {language}",
                              backend=self.backend, code=(code or ""))

        tier_name, limits = resolve_tier(tier)
        # An explicit timeout still wins: callers that know their workload override the tier.
        timeout = int(timeout or limits.get("timeout") or DEFAULT_TIMEOUT)
        deps, rejected = _sanitize_deps(dependencies)

        persistent = bool((session_id or "").strip())
        try:
            if persistent:
                sweep_workspaces()
                work = session_work_dir(session_id or "")
            else:
                work = Path(tempfile.mkdtemp(prefix="agentexec_", dir=_work_root()))
        except OSError as exc:
            # A missing/unwritable work root (e.g. the AGENT_CODE_EXEC_WORK_ROOT bind mount not
            # present in this deployment) must surface as a TOOL error the agent can report —
            # never crash the whole turn/stream.
            return ExecResult(
                exit_code=None,
                error=(f"code-execution work dir unavailable: {exc}. "
                       f"Check {WORK_ROOT_ENV} and its bind mount in the deployment."),
                backend=self.backend, code=(code or ""),
            )
        try:
            # The invariant gate runs as an epilogue on the SAME interpreter, so it inspects
            # the live frames the code produced rather than its source text — the only way to
            # know what CRS a frame was actually in when .buffer() was called. Appended to the
            # written script but NOT to `code`, so the source persisted as an artifact and
            # echoed back to the model stays exactly what the model wrote.
            script = (code or "")
            if invariant_gate_enabled():
                from agent_runtime.sandbox_verify import epilogue_source
                script = script + epilogue_source()
            (work / "script.py").write_text(script, encoding="utf-8")
            # Stage uploaded/input files into the work dir so the code can read them.
            staged, stage_errors = _stage_inputs(work, input_files)
            try:
                os.chmod(work, 0o777)  # let a non-root container user write outputs
            except OSError:
                pass
            exit_code, stdout, stderr, timed_out, error = self._run(work, timeout, deps, limits=limits)
            # Output files the run produced, plus the executed source itself (downloadable).
            # Staged input files are excluded so uploads aren't re-persisted as outputs.
            # A persistent workspace MUST persist incrementally: this walk stops at
            # MAX_ARTIFACTS, so step 1's leftovers would otherwise consume the budget in
            # sorted-path order and step 5's real output would never be persisted at all.
            artifacts = [*_persist_source(code or ""),
                         *_persist_artifacts(work, {"script.py", *staged}, incremental=persistent)]
            if rejected:
                stderr = (str(stderr or "") + f"\n[ignored unsafe dependencies: {rejected}]").strip()
            if stage_errors:
                stderr = (str(stderr or "") + f"\n[input file staging errors: {stage_errors}]").strip()
            if persistent:
                size_mb = _dir_size_mb(work)
                if size_mb > WORKSPACE_MAX_MB:
                    stderr = (str(stderr or "") + f"\n[workspace {size_mb:.0f}MB exceeds "
                              f"{WORKSPACE_MAX_MB:.0f}MB cap; older files may be reclaimed]").strip()
            verification = _read_checks(work)
            # The reproducible record: run.py + manifest.json (image DIGEST, in-sandbox
            # environment, input hashes, library slice_shas) + inputs.jsonl. Emitted before
            # returning so it lands beside the run, and guarded inside emit() so a failure to
            # write provenance can never fail a successful analysis.
            if artifacts_enabled():
                from agent_runtime import artifacts as _artifacts
                _artifacts.emit(code=(code or ""), work=work,
                                image=getattr(self, "image", ""), backend=self.backend,
                                dependencies=deps, tier=tier_name, staged=sorted(staged),
                                verification=verification)
            return ExecResult(exit_code, _clip(stdout), _clip(stderr), timed_out, error,
                              artifacts, self.backend, code=(code or ""), installed=deps,
                              verification=verification)
        finally:
            if not persistent:
                shutil.rmtree(work, ignore_errors=True)

    def _run(self, work: Path, timeout: int,
             dependencies: Optional[List[str]] = None,
             *, limits: Optional[Dict[str, str]] = None) -> Tuple[Optional[int], str, str, bool, Optional[str]]:
        raise NotImplementedError


class DockerCodeExecutor(CodeExecutor):
    """Container-per-run sandbox (the production backend)."""

    backend = "docker"

    def __init__(self, *, image: str = DEFAULT_IMAGE, memory: str = DEFAULT_MEMORY,
                 cpus: str = DEFAULT_CPUS, pids: str = DEFAULT_PIDS,
                 install_memory: str = DEFAULT_INSTALL_MEMORY) -> None:
        self.image = image
        self.memory = memory
        self.cpus = cpus
        self.pids = pids
        self.install_memory = install_memory

    def build_argv(self, work: Path, name: str,
                   *, limits: Optional[Dict[str, str]] = None) -> List[str]:
        """Execution phase: NO network, read-only rootfs, deps importable via PYTHONPATH.

        ``limits`` comes from the execution tier and overrides the constructor defaults, so a
        real analysis run is not held to the quick-tool budget of 512m/1cpu.
        """
        limits = limits or {}
        argv = [
            "docker", "run", "--rm", "--init", "--name", name,
            "--network", "none",            # no network during execution
            "--read-only",                  # read-only root fs
            "--cap-drop", "ALL",            # drop all capabilities
            "--security-opt", "no-new-privileges",
            "--memory", str(limits.get("memory") or self.memory),
            "--cpus", str(limits.get("cpus") or self.cpus),
            "--pids-limit", self.pids,
            "--workdir", "/work",
            "--tmpfs", "/tmp:rw,size=64m,exec",
            "--env", "HOME=/tmp",
            "--env", f"PYTHONPATH=/work/{DEPS_DIRNAME}:{METHOD_LIBRARY_MOUNT}",
            "-v", f"{work}:/work:rw",       # only writable mount
        ]
        lib = method_library_dir()
        if lib:
            argv += ["-v", f"{lib}:{METHOD_LIBRARY_MOUNT}:ro"]
        user = _host_user()
        if user:
            argv += ["--user", user]
        argv += [self.image, "python", "/work/script.py"]
        return argv

    def build_install_argv(self, work: Path, deps: List[str], name: str) -> List[str]:
        """Install phase: network ON, pip install into the shared work dir (/work/.deps)."""
        argv = [
            "docker", "run", "--rm", "--init", "--name", name,
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--memory", self.install_memory,   # larger budget than exec (avoids OOM)
            "--cpus", self.cpus,
            "--pids-limit", self.pids,
            "--workdir", "/work",
            "--tmpfs", "/tmp:rw,size=128m,exec",
            "--env", "HOME=/tmp",
            # pip scratch on the host-backed work dir (real disk), not the tmpfs.
            "--env", f"TMPDIR=/work/{PIPTMP_DIRNAME}",
            "-v", f"{work}:/work:rw",
        ]
        user = _host_user()
        if user:
            argv += ["--user", user]
        argv += [self.image, "pip", "install", "--no-cache-dir", "--target", f"/work/{DEPS_DIRNAME}", *deps]
        return argv

    def _run(self, work: Path, timeout: int, dependencies: Optional[List[str]] = None,
             *, limits: Optional[Dict[str, str]] = None):
        # Phase 1 (deps): a separate container WITH network installs into /work/.deps.
        # With a persistent workspace .deps survives, so a session installs geopandas once
        # instead of once per call -- the single largest latency win of the workspace change.
        if dependencies and _deps_satisfied(work, dependencies):
            logger.debug("deps already present in %s; skipping install", work / DEPS_DIRNAME)
            dependencies = []
        if dependencies:
            (work / PIPTMP_DIRNAME).mkdir(parents=True, exist_ok=True)  # pip TMPDIR (real disk)
            iname = f"agentexec_pip_{uuid.uuid4().hex[:12]}"
            try:
                inst = subprocess.run(self.build_install_argv(work, dependencies, iname),
                                      capture_output=True, text=True, timeout=DEFAULT_INSTALL_TIMEOUT + 5)
            except subprocess.TimeoutExpired as exc:
                subprocess.run(["docker", "kill", iname], capture_output=True)
                return None, (exc.stdout or ""), (exc.stderr or ""), True, "dependency install timed out"
            except FileNotFoundError:
                return None, "", "", False, "docker executable not found"
            if inst.returncode != 0:
                return None, inst.stdout, _clip(inst.stderr), False, "dependency install failed"
            _record_deps(work, dependencies)
        # Phase 2 (exec): NO network.
        name = f"agentexec_{uuid.uuid4().hex[:12]}"
        try:
            proc = subprocess.run(self.build_argv(work, name, limits=limits),
                                  capture_output=True, text=True, timeout=timeout + 5)
            return proc.returncode, proc.stdout, proc.stderr, False, None
        except subprocess.TimeoutExpired as exc:
            subprocess.run(["docker", "kill", name], capture_output=True)
            return None, (exc.stdout or ""), (exc.stderr or ""), True, "execution timed out"
        except FileNotFoundError:
            return None, "", "", False, "docker executable not found"
        except Exception as exc:  # pragma: no cover - defensive
            return None, "", "", False, f"{type(exc).__name__}: {exc}"


class LocalSubprocessExecutor(CodeExecutor):
    """DEV ONLY — runs on the host, NOT a sandbox. Opt-in via AGENT_CODE_EXEC_BACKEND=local."""

    backend = "local-unsafe"

    def _run(self, work: Path, timeout: int, dependencies: Optional[List[str]] = None,
             *, limits: Optional[Dict[str, str]] = None):
        deps_dir = work / DEPS_DIRNAME
        # Same as the docker backend: a persistent workspace keeps .deps, so skip a
        # reinstall the session has already paid for.
        requested = list(dependencies or [])
        if requested and _deps_satisfied(work, requested):
            logger.debug("deps already present in %s; skipping install", deps_dir)
            dependencies = []
        if dependencies:
            try:
                inst = subprocess.run(
                    [sys.executable or "python", "-m", "pip", "install", "--no-cache-dir",
                     "--target", str(deps_dir), *dependencies],
                    capture_output=True, text=True, timeout=DEFAULT_INSTALL_TIMEOUT,
                )
            except subprocess.TimeoutExpired as exc:
                return None, (exc.stdout or ""), (exc.stderr or ""), True, "dependency install timed out"
            if inst.returncode != 0:
                return None, inst.stdout, _clip(inst.stderr), False, "dependency install failed"
            _record_deps(work, dependencies)
        env = {"PATH": os.environ.get("PATH", ""), "HOME": str(work)}
        # PYTHONPATH must be set whenever deps exist on disk, not only when we just
        # installed them -- otherwise a skipped reinstall would make the imports fail.
        if requested or deps_dir.exists():
            env["PYTHONPATH"] = str(deps_dir)
        try:
            proc = subprocess.run(
                [sys.executable or "python", "script.py"],
                cwd=str(work), env=env, capture_output=True, text=True, timeout=timeout,
            )
            return proc.returncode, proc.stdout, proc.stderr, False, None
        except subprocess.TimeoutExpired as exc:
            return None, (exc.stdout or ""), (exc.stderr or ""), True, "execution timed out"
        except Exception as exc:  # pragma: no cover - defensive
            return None, "", "", False, f"{type(exc).__name__}: {exc}"


class DisabledExecutor(CodeExecutor):
    """Returned when the requested backend is unavailable; never runs code."""

    backend = "disabled"

    def __init__(self, reason: str) -> None:
        self._reason = reason

    def _run(self, work: Path, timeout: int, dependencies: Optional[List[str]] = None,
             *, limits: Optional[Dict[str, str]] = None):
        return None, "", "", False, self._reason


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def get_code_executor() -> CodeExecutor:
    """Pick an executor from env.

    ``AGENT_CODE_EXEC_BACKEND``: ``docker`` (default) | ``local`` (UNSAFE, dev).
    Docker is *not* silently downgraded to local — if docker is unavailable a
    DisabledExecutor is returned so untrusted code is never run unsandboxed.
    """
    backend = (os.getenv("AGENT_CODE_EXEC_BACKEND") or "docker").strip().lower()
    if backend == "local":
        return LocalSubprocessExecutor()
    if backend == "docker":
        if _docker_available():
            return DockerCodeExecutor()
        return DisabledExecutor(
            "docker backend unavailable (docker not found). Set AGENT_CODE_EXEC_BACKEND=local "
            "to run on the host for development (UNSAFE — not a sandbox)."
        )
    return DisabledExecutor(f"unknown AGENT_CODE_EXEC_BACKEND: {backend}")


__all__ = [
    "ExecResult",
    "CodeExecutor",
    "DockerCodeExecutor",
    "LocalSubprocessExecutor",
    "DisabledExecutor",
    "get_code_executor",
    "is_code_exec_enabled",
]
