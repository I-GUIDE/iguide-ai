"""Sandboxed `opencode` backend for the code peer (``AGENT_CODE_PEER=opencode``).

Runs the open-source opencode coding agent (https://opencode.ai) headlessly
(``opencode run``) inside a **fresh, hardened container per run** — the same
flag family as ``code_execution.DockerCodeExecutor`` with two deliberate
differences: the container keeps **network access** (opencode must reach the
OpenAI-compatible LLM endpoint, and its bash tool may pip-install), and it gets
a larger memory/pids budget (node runtime + agentic loop).

Unlike the LangChain code peer (generate → ``execute_code`` tool), opencode
iterates internally: it writes code into the throwaway work dir, runs it, reads
errors, and retries until done. Files it leaves in the work dir are persisted
to the agent file store as downloadable artifacts, exactly like ``execute_code``
outputs.

LLM wiring reuses ``build_default_llm``'s env contract (``VLLM_*`` →
``OPENAI_*`` precedence): a per-run ``opencode.json`` in the work dir declares
an ``@ai-sdk/openai-compatible`` provider pointing at the same endpoint. The
API key is passed to the container as an env var and referenced from the config
via ``{env:...}`` — it is never written into the work dir (which is persisted
as artifacts) — and session sharing/autoupdate are disabled so nothing leaves
the deployment.

The sandbox image must have opencode installed — see ``Dockerfile.opencode``
at the repo root; override the name via ``AGENT_OPENCODE_IMAGE``. Under
Docker-out-of-Docker the work dir must live on the host-shared bind mount
(``AGENT_CODE_EXEC_WORK_ROOT``), same as the execute_code sandbox. NOTE: the
LLM base URL is resolved from *inside* the container — a ``localhost`` vLLM
endpoint is unreachable there; set ``AGENT_OPENCODE_BASE_URL`` (e.g. to
``http://host.docker.internal:8000/v1``) for local development.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_runtime.code_execution import (
    MAX_ARTIFACTS,
    _clip,
    _host_user,
    _stage_inputs,
    _work_root,
)

# Env var read by is_opencode_peer_enabled(); any other value keeps the default
# LangChain code peer.
CODE_PEER_ENV = "AGENT_CODE_PEER"

DEFAULT_OPENCODE_IMAGE = "agent-opencode:latest"
# Agentic write→run→debug loops are much slower than a single execute_code run.
DEFAULT_OPENCODE_TIMEOUT = 600
DEFAULT_OPENCODE_MEMORY = "2g"
DEFAULT_OPENCODE_CPUS = "2.0"
DEFAULT_OPENCODE_PIDS = "1024"

# Provider id inside the generated opencode.json; the model ref is "<id>/<model>".
_PROVIDER_ID = "vllm"
# The key travels via this container env var, referenced from opencode.json as
# {env:...} so it never lands on disk in the (artifact-persisted) work dir.
_API_KEY_ENV = "AGENT_OPENCODE_API_KEY"

_CONFIG_FILENAME = "opencode.json"

# Bound how much retrieved evidence / upstream analysis is inlined into the
# opencode prompt (mirrors the caps used by the LangChain code peer).
MAX_EVIDENCE_CHARS = 6000
MAX_ANALYSIS_CHARS = 1500

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def selects_opencode(value: Optional[str]) -> bool:
    """Does this AGENT_CODE_PEER value name this backend? (pure, so the env default
    and a per-request override share one definition)."""
    return (value or "").strip().lower() == "opencode"


def is_opencode_peer_enabled() -> bool:
    """Whether the deployment default selects opencode for the code peer."""
    return selects_opencode(os.getenv(CODE_PEER_ENV))


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text or "")


def resolve_llm_settings() -> Dict[str, Optional[str]]:
    """Resolve model / base URL / API key for opencode.

    Same precedence as ``executor_factory.build_default_llm`` (VLLM_* →
    OPENAI_* → defaults), with ``AGENT_OPENCODE_*`` overrides on top so the
    code peer can run a stronger model than the rest of the agents.
    """
    from agent_runtime.executor_factory import normalize_openai_base_url

    model = (
        os.getenv("AGENT_OPENCODE_MODEL")
        or os.getenv("VLLM_MODEL")
        or os.getenv("OPENAI_CHAT_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "Qwen/Qwen3.5-9B"
    )
    base_url = normalize_openai_base_url(
        os.getenv("AGENT_OPENCODE_BASE_URL") or os.getenv("VLLM_PROXY") or os.getenv("OPENAI_BASE_URL")
    )
    api_key = (
        os.getenv(_API_KEY_ENV)
        or os.getenv("VLLM_API_KEY")
        or os.getenv("OPENAI_KEY")
    )
    return {"model": model, "base_url": base_url, "api_key": api_key}


def model_ref(model: str) -> str:
    """opencode model reference: ``<provider>/<model>`` (model may contain slashes)."""
    return f"{_PROVIDER_ID}/{model}"


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def build_opencode_config(model: str, base_url: Optional[str]) -> Dict[str, Any]:
    """The per-run opencode.json: one OpenAI-compatible provider, nothing external."""
    options: Dict[str, Any] = {"apiKey": f"{{env:{_API_KEY_ENV}}}"}
    if base_url:
        options["baseURL"] = base_url
    # Custom providers get no metadata from models.dev, so opencode falls back to
    # a 32000 max_tokens default that many endpoints reject (gpt-4o caps at
    # 16384). Declare explicit limits; `output` is what is sent as max_tokens.
    limit = {
        "context": _int_env("AGENT_OPENCODE_CONTEXT_LIMIT", 128_000),
        "output": _int_env("AGENT_OPENCODE_OUTPUT_LIMIT", 8_192),
    }
    return {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            _PROVIDER_ID: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Platform LLM",
                "options": options,
                "models": {model: {"name": model, "limit": limit}},
            }
        },
        "model": model_ref(model),
        # Headless run — never block on an approval prompt.
        "permission": {"edit": "allow", "bash": "allow"},
        # Nothing leaves the deployment: no session-share uploads, no self-update.
        "share": "disabled",
        "autoupdate": False,
    }


def build_docker_argv(work: Path, name: str, model: str, prompt: str) -> List[str]:
    """``docker run`` argv for one opencode run.

    Hardened like the execute_code sandbox (read-only rootfs, cap-drop, no
    privilege escalation, cpu/mem/pid limits, /work the only writable mount) but
    WITH network — opencode is useless without its LLM endpoint. HOME=/work so
    all opencode state lands in the throwaway work dir (dot-dirs are excluded
    from artifact persistence).
    """
    argv = [
        "docker", "run", "--rm", "--init", "--name", name,
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--read-only",
        "--memory", os.getenv("AGENT_OPENCODE_MEMORY", DEFAULT_OPENCODE_MEMORY),
        "--cpus", os.getenv("AGENT_OPENCODE_CPUS", DEFAULT_OPENCODE_CPUS),
        "--pids-limit", os.getenv("AGENT_OPENCODE_PIDS", DEFAULT_OPENCODE_PIDS),
        "--workdir", "/work",
        "--tmpfs", "/tmp:rw,size=256m,exec",
        "--env", "HOME=/work",
        "--env", f"OPENCODE_CONFIG=/work/{_CONFIG_FILENAME}",
        "--env", "OPENCODE_DISABLE_AUTOUPDATE=true",
        # Name-only form: docker copies the value from the client process env
        # (set by run_opencode), so the key never appears in the argv.
        "--env", _API_KEY_ENV,
        "-v", f"{work}:/work:rw",
    ]
    network = (os.getenv("AGENT_OPENCODE_NETWORK") or "").strip()
    if network:
        argv += ["--network", network]
    user = _host_user()
    if user:
        argv += ["--user", user]
    image = os.getenv("AGENT_OPENCODE_IMAGE", DEFAULT_OPENCODE_IMAGE)
    argv += [image, "opencode", "run", "--model", model_ref(model), prompt]
    return argv


def _timeout_seconds() -> int:
    try:
        return max(30, int(os.getenv("AGENT_OPENCODE_TIMEOUT", str(DEFAULT_OPENCODE_TIMEOUT))))
    except (TypeError, ValueError):
        return DEFAULT_OPENCODE_TIMEOUT


def _persist_artifacts(work: Path, exclude: set) -> List[Dict[str, Any]]:
    """Persist files opencode left in *work* to the agent file store.

    Skips dot-prefixed top-level entries (opencode/HOME state: ``.local``,
    ``.config``, ``.cache``, …) and anything in *exclude* (the generated config,
    staged input files).
    """
    try:
        from agent_runtime.file_store import create_output_file_from_path
    except Exception:
        return []

    artifacts: List[Dict[str, Any]] = []
    for path in sorted(work.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(work)
        if not rel.parts or rel.parts[0].startswith(".") or rel.parts[0] == "__pycache__":
            continue
        if str(rel) in exclude:
            continue
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
                    # The path RELATIVE to the work dir, which the basename cannot stand in
                    # for once directories nest. A caller that persists across turns needs
                    # to know exactly which files reached the store — the cap above and the
                    # except below both mean "walked, not delivered".
                    "path": str(rel),
                }
            )
        except Exception:
            continue
    return artifacts


def _stage_conversation_files(work: Path, input_file_ids: Optional[List[str]]) -> Dict[str, Any]:
    """Copy conversation-attached files into *work* (same policy/caps as execute_code)."""
    refs = [str(x).strip() for x in (input_file_ids or []) if str(x).strip()]
    if not refs:
        return {"staged": [], "staged_info": [], "errors": [], "skipped": []}
    from agent_runtime.langchain_exec_tools import _build_staging

    staging, staged_info, errors, skipped = _build_staging(refs)
    staged, stage_errors, _shadowed = _stage_inputs(work, staging)
    return {
        "staged": staged,
        "staged_info": staged_info,
        "errors": [*errors, *stage_errors],
        "skipped": skipped,
    }


def run_opencode(
    prompt: str,
    *,
    input_file_ids: Optional[List[str]] = None,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """One sandboxed ``opencode run``; returns a JSON-serializable result dict."""
    settings = resolve_llm_settings()
    if not settings["api_key"]:
        return {
            "ok": False, "exit_code": None, "answer": "", "stderr": "", "timed_out": False,
            "error": f"{_API_KEY_ENV} (or VLLM_API_KEY / OPENAI_KEY) is required for the opencode code peer.",
            "artifacts": [], "backend": "opencode-docker", "model": None,
        }
    model = settings["model"]
    timeout = int(timeout or _timeout_seconds())
    try:
        work = Path(tempfile.mkdtemp(prefix="agentoc_", dir=_work_root()))
    except OSError as exc:
        return {
            "ok": False, "exit_code": None, "answer": "", "stderr": "", "timed_out": False,
            "error": (f"opencode work dir unavailable: {exc}. "
                      "Check AGENT_CODE_EXEC_WORK_ROOT and its bind mount in the deployment."),
            "artifacts": [], "backend": "opencode-docker", "model": model_ref(model),
        }
    try:
        (work / _CONFIG_FILENAME).write_text(
            json.dumps(build_opencode_config(model, settings["base_url"]), indent=2),
            encoding="utf-8",
        )
        staging = _stage_conversation_files(work, input_file_ids)
        try:
            os.chmod(work, 0o777)  # non-root container user must write here
        except OSError:
            pass
        name = f"agentoc_{uuid.uuid4().hex[:12]}"
        argv = build_docker_argv(work, name, model, prompt)
        env = {**os.environ, _API_KEY_ENV: settings["api_key"]}
        exit_code: Optional[int] = None
        stdout, stderr, timed_out, error = "", "", False, None
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=timeout + 5)
            exit_code, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as exc:
            subprocess.run(["docker", "kill", name], capture_output=True)
            stdout, stderr = str(exc.stdout or ""), str(exc.stderr or "")
            timed_out, error = True, f"opencode run timed out after {timeout}s"
        except FileNotFoundError:
            error = "docker executable not found"
        except Exception as exc:  # pragma: no cover - defensive
            error = f"{type(exc).__name__}: {exc}"

        answer = _clip(_strip_ansi(stdout).strip())
        artifacts = _persist_artifacts(work, {_CONFIG_FILENAME, *staging["staged"]})
        # No tools means no add_map_layer means nothing checked what this wrote. See
        # layer_qa.inspect_artifacts.
        from agent_runtime.layer_qa import inspect_artifacts

        qa = inspect_artifacts(str(work), [a.get("filename") for a in artifacts])
        # A peer with no tools still wrote geodata. Turn it into layer descriptors here,
        # while the work dir still exists — the wrapper emits them from the request context.
        from agent_runtime.map_layers import layers_for_artifacts

        map_layers = layers_for_artifacts(work, artifacts)
        result: Dict[str, Any] = {
            "ok": error is None and not timed_out and exit_code == 0,
            "exit_code": exit_code,
            "answer": answer,
            "stderr": _clip(_strip_ansi(stderr).strip()),
            "timed_out": timed_out,
            "error": error,
            "artifacts": artifacts,
            "backend": "opencode-docker",
            "model": model_ref(model),
        }
        if qa:
            result["output_warnings"] = qa
        if map_layers:
            result["map_layers"] = map_layers
            # The supervisor's delivery check reads this off the execution record; without
            # it a turn that DID put something on the map still counts as undelivered.
            result["on_map"] = True
        if staging["staged_info"]:
            result["input_files"] = staging["staged_info"]
        if staging["errors"]:
            result["input_file_errors"] = staging["errors"]
        if staging["skipped"]:
            result["input_files_skipped"] = staging["skipped"]
        return result
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _build_peer_prompt(
    query: str,
    evidence: Optional[List[Any]],
    analysis_results: Any,
    staged_names: Optional[List[str]] = None,
) -> str:
    """The headless task brief, mirroring the context the LangChain code peer gets."""
    parts = [
        "You are the code peer of a multi-agent geospatial analysis platform, running "
        "headless in a sandboxed container. The working directory is yours: write code, "
        "RUN it, and debug until it works (install Python packages with "
        "`pip install --user <pkg>` if needed). Save any output files (plots, tables, "
        "data) to the working directory. Finish with a concise summary of what you did, "
        "the key results, and the files you wrote.",
        f"Task:\n{query}",
    ]
    if evidence:
        try:
            from agent_runtime.supervisor.evidence_subgraph import _format_documents

            parts.append(f"Evidence:\n{_format_documents(evidence)[:MAX_EVIDENCE_CHARS]}")
        except Exception:
            pass
    if analysis_results:
        parts.append(
            "Analysis results:\n"
            + json.dumps(analysis_results, ensure_ascii=True, default=str)[:MAX_ANALYSIS_CHARS]
        )
    if staged_names:
        parts.append("Input files already in the working directory: " + ", ".join(staged_names))
    return "\n\n".join(parts)


def run_opencode_code_peer(
    query: str,
    evidence: Optional[List[Any]] = None,
    state: Optional[Dict[str, Any]] = None,
    input_file_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Code-peer adapter: returns the same flat shape as ``default_code_fn``
    (``answer`` + compact ``tool_calls``/``tool_results``) so synthesis and the
    trace pipeline are agnostic to which backend produced the code result."""
    from agent_runtime.map_layers import build_map_layers
    from agent_runtime.streaming_trace import emit_trace_event

    # Resolve refs to the names the files will be staged under (file_id AND
    # original filename) so the prompt tells opencode what is actually on disk.
    staged_names: List[str] = []
    refs = [str(x).strip() for x in (input_file_ids or []) if str(x).strip()]
    if refs:
        try:
            from agent_runtime.langchain_exec_tools import _build_staging

            _, staged_info, _, _ = _build_staging(refs)
            for info in staged_info:
                staged_names.extend(info.get("available_as") or [])
        except Exception:
            staged_names = list(refs)
    prompt = _build_peer_prompt(
        query,
        evidence,
        (state or {}).get("analysis_results"),
        staged_names=staged_names or None,
    )
    call_args = {"model": resolve_llm_settings()["model"], "prompt_chars": len(prompt)}
    emit_trace_event("tool_call", {"name": "opencode_run", "args": call_args}, node="code")
    result = run_opencode(prompt, input_file_ids=input_file_ids)
    emit_trace_event(
        "tool_result",
        {
            "name": "opencode_run",
            "content": {k: result.get(k) for k in
                        ("ok", "exit_code", "timed_out", "error", "artifacts", "backend", "model")},
        },
        node="code",
    )
    # The peer has no tools, so nothing emitted a map_layer on its behalf. This wrapper
    # runs in the request's trace context — the same place a tool callback would — so the
    # descriptors go out here, through the same build_map_layers boundary every tool's
    # layer crosses, and get the same validation.
    for layer in build_map_layers("opencode_run", result):
        emit_trace_event("map_layer", layer, node="code")

    answer = result.get("answer") or ""
    warnings = result.get("output_warnings") or []
    if warnings:
        lines = [f"- {w['file']}: {'; '.join(w['problems'])}" for w in warnings]
        answer = "\n\n".join(x for x in (answer, "Checks on the files this run produced "
                                                   "found problems — say so rather than "
                                                   "presenting them as results:\n"
                                                   + "\n".join(lines)) if x)
    if not result.get("ok"):
        failure = result.get("error") or f"opencode exited with code {result.get('exit_code')}"
        detail = str(result.get("stderr") or "")[-2000:]
        answer = "\n\n".join(
            x for x in (f"opencode code peer failed: {failure}", detail, answer) if x
        )
    return {
        "answer": answer,
        "tool_calls": [{"name": "opencode_run", "args": call_args}],
        "tool_results": [{"name": "opencode_run", "content": result}],
    }


__all__ = [
    "CODE_PEER_ENV",
    "is_opencode_peer_enabled",
    "resolve_llm_settings",
    "build_opencode_config",
    "build_docker_argv",
    "model_ref",
    "run_opencode",
    "run_opencode_code_peer",
]
