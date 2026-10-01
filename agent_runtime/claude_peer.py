"""Sandboxed `claude` (Claude Code) backend for the code peer (``AGENT_CODE_PEER=claude``).

Sibling of :mod:`agent_runtime.opencode_peer`, and deliberately its twin: one
**fresh hardened container per run** (read-only rootfs, dropped capabilities,
no-new-privileges, cpu/mem/pid limits, ``/work`` the only writable mount), the
agentic CLI iterating internally — write code, run it, read the error, retry —
and everything it leaves in the work dir persisted to the agent file store as
downloadable artifacts, exactly like ``execute_code`` output.

Two differences from the opencode backend, both forced by what the CLI talks to:

* **Auth is Anthropic's, not the deployment's OpenAI-compatible endpoint.** There
  is no provider config file to generate. Either credential works and they bill
  differently: ``CLAUDE_CODE_OAUTH_TOKEN`` from ``claude setup-token``
  authenticates as a **Claude subscription** (requests count against that
  person's plan), while ``ANTHROPIC_API_KEY`` is metered API billing.
  ``ANTHROPIC_BASE_URL`` is honoured for a gateway. Whichever is used travels to
  the container by NAME only, so it never appears in the argv or in the work dir
  that gets persisted as artifacts.
* **The model is Anthropic's**, named by alias (``sonnet``/``opus``) or full id,
  independent of ``VLLM_*``/``OPENAI_*``. A deployment can therefore run its
  answers on one provider and its code peer on another.

The flags here are the ones this CLI actually has — checked against the version
the image installs rather than assumed. ``--max-turns`` does NOT exist in 2.1.x
and an unknown flag makes the CLI exit non-zero, so the agentic loop is bounded
by the run timeout and the container, not by a turn count. ``--bare`` skips
hooks, LSP, plugin sync, auto-memory, keychain reads and CLAUDE.md discovery,
all of which a throwaway container wants — but it also pins auth to
``ANTHROPIC_API_KEY`` and never reads OAuth, so it is used ONLY on the API-key
path. On the subscription path the CLAUDE.md half of that protection is
restored by :func:`neutralize_instruction_files`.

``--dangerously-skip-permissions`` is required: there is no TTY to approve tool
use, so without it every run stalls on the first permission prompt. Its own help
recommends it "only for sandboxes with no internet access", and this sandbox HAS
network — it must reach the Anthropic API. The container is the mitigation, not
the flag: read-only root, no capabilities, non-root user, throwaway work dir.
That is the same trade the opencode peer already makes.

**The tool surface is deliberately unrestricted, and that is a decision rather
than an oversight.** Measured from inside the sandbox: the CLI has Bash, Read,
Write, Edit, Agent, Skill, Workflow and (deferred) WebFetch/WebSearch, and
generated code reaches the internet — ``urlopen("https://api.github.com/zen")``
returned 200. Two consequences worth stating plainly, because both look like
bugs if you meet them cold:

* Code written by THIS peer can fetch things; the same code under the built-in
  peer cannot, because ``execute_code`` runs ``--network none``. Swapping the
  peer swaps the network posture of generated code.
* Its WebFetch/WebSearch do not pass through the agent's own open-web
  governance — the two-step search/fetch design, the per-turn caps, or
  ``AGENT_WEB_ALLOWED_PORTS``, which exists because this deployment's own
  services sit on public addresses.

A code peer that can install a package, read an error and try again is the point
of running one, so the surface stays wide and the CONTAINER carries the safety.
``AGENT_CLAUDE_ALLOWED_TOOLS`` narrows it for a deployment that wants that; it is
unset by default on purpose.

The image must have Claude Code installed — see ``Dockerfile.claude`` at the
repo root; override the name via ``AGENT_CLAUDE_IMAGE``. Under
Docker-out-of-Docker the work dir must live on the host-shared bind mount
(``AGENT_CODE_EXEC_WORK_ROOT``), same as the execute_code sandbox.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_runtime.code_execution import _clip, _host_user, _work_root
from agent_runtime.opencode_peer import (
    CODE_PEER_ENV,
    _build_peer_prompt,
    _persist_artifacts,
    _stage_conversation_files,
    _strip_ansi,
)

DEFAULT_CLAUDE_IMAGE = "agent-claude:latest"
# Agentic write->run->debug loops are much slower than a single execute_code run.
DEFAULT_CLAUDE_TIMEOUT = 600
DEFAULT_CLAUDE_MEMORY = "2g"
DEFAULT_CLAUDE_CPUS = "2.0"
DEFAULT_CLAUDE_PIDS = "1024"
# An alias, not a pinned id: the CLI resolves it to the current model, so the
# sandbox does not silently pin itself to a retired one.
DEFAULT_CLAUDE_MODEL = "sonnet"
# node:22-slim ships a `node` user here, and the work dir is chmod 0777, so this
# uid can write everything the run needs. See sandbox_user() for why not root.
DEFAULT_SANDBOX_USER = "1000:1000"

# Credentials travel via these container env vars, passed by NAME so the value
# never lands in the argv (visible in `docker ps` / `ps`) or on the persisted
# work dir. Two kinds, and the CLI treats them very differently — see _auth_of.
_API_KEY_ENV = "ANTHROPIC_API_KEY"
_OAUTH_TOKEN_ENV = "CLAUDE_CODE_OAUTH_TOKEN"

logger = logging.getLogger(__name__)

_ACCEPTED_FLAG_VALUES = {"claude", "claude-code", "claude_code"}

# What the picker offers. ALIASES, not pinned ids: the CLI resolves each to the
# current model of that family, so the list does not rot when an id is retired.
# `haiku` was verified against the installed CLI (an unknown alias is refused
# locally, before any API call), the others are named in its own --help.
SELECTABLE_MODELS = ("sonnet", "opus", "haiku", "fable")

# Files a Claude Code session picks up from the working directory as INSTRUCTIONS
# rather than as data. Staged conversation files are user uploads, so one named
# CLAUDE.md would be read as a brief by an agent running with tool permissions
# skipped and network access. --bare disables that discovery; subscription auth
# cannot use --bare, so the staging guard below closes the same door either way.
_INSTRUCTION_FILENAMES = {"claude.md", "claude.local.md", ".claude"}


# --- persistent per-thread project directory ------------------------------------------
#
# A run used to get mkdtemp + rmtree, so every turn started from nothing: files gone,
# installed packages gone, and no memory of having worked on any of it. Keeping the
# directory across turns of the same conversation gives the CLI a PROJECT — and, because
# CLAUDE_CONFIG_DIR points inside it, its own session history too.
#
# Measured on the deployment before building this: run 1 wrote notes.txt containing ZEBRA;
# run 2 with --continue answered "ZEBRA" without reading the file; a fresh run without
# --continue answered "UNKNOWN — I have no record of writing anything to notes.txt". So
# both halves survive, and the control rules out it having simply read the file.
#
# What does NOT persist is the container: every turn still gets a fresh hardened one, and
# only the bind mount carries over. That is the whole reason this is safe to default on.
DEFAULT_SESSION_TTL_HOURS = 72
_MANIFEST_NAME = ".agent_artifacts.json"
_LOCK_NAME = ".agent_lock"
# A lock older than this is a crashed run, not a running one: the wall-clock ceiling for a
# run is DEFAULT_CLAUDE_TIMEOUT, so anything past it plus slack cannot still be alive.
def _lock_stale_after() -> float:
    """When a claim can only belong to a run that died.

    Derived from the CONFIGURED timeout, not the default: a deployment that raises
    AGENT_CLAUDE_TIMEOUT for long agentic loops would otherwise have every run past twenty
    minutes declared crashed, and a second container would mount the same directory while
    the first was still writing to it.
    """
    return _timeout_seconds() * 2 + 60


def _work_root_path() -> Path:
    """The directory session directories live under.

    ``_work_root()`` returns None when AGENT_CODE_EXEC_WORK_ROOT is unset — which the
    documented local-dev command never sets — and ``Path(None)`` raises TypeError, not the
    OSError run_claude guards against. The old code passed the None straight to
    ``mkdtemp(dir=...)``, where it means "the system temp dir", so this keeps that meaning
    explicit instead of crashing the whole graph run on the local configuration.
    """
    root = _work_root()
    return Path(root) if root else Path(tempfile.gettempdir())


def persistence_enabled() -> bool:
    """Whether a conversation keeps its project directory between turns."""
    return (os.getenv("AGENT_CLAUDE_PERSIST", "1") or "").strip().lower() not in {
        "0", "false", "no", "off"}


def _session_ttl_hours() -> float:
    try:
        return max(0.0, float(os.getenv("AGENT_CLAUDE_SESSION_TTL_HOURS",
                                        str(DEFAULT_SESSION_TTL_HOURS))))
    except (TypeError, ValueError):
        return float(DEFAULT_SESSION_TTL_HOURS)


def _sweep_sessions(root: Path) -> None:
    """Drop session directories nobody has touched inside the TTL.

    Without this a busy deployment accumulates one directory per conversation forever,
    including every one-off question. Failure here is never fatal: a full disk is a
    problem, but so is refusing to answer because a stale directory would not delete.
    """
    ttl = _session_ttl_hours()
    if ttl <= 0:
        return
    import time

    cutoff = time.time() - ttl * 3600
    try:
        candidates = list(root.glob("claudesess_*"))
    except OSError:
        return
    for path in candidates:
        try:
            if path.is_dir() and path.stat().st_mtime < cutoff:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            continue


def _acquire(work: Path) -> bool:
    """Claim a session directory for this run, or report that another run holds it.

    Two turns of one conversation in one directory would interleave file writes and
    corrupt the CLI's own session history. A caller that cannot claim it falls back to a
    throwaway directory rather than failing the turn — losing continuity beats losing the
    answer.
    """
    import time

    lock = work / _LOCK_NAME
    try:
        if lock.exists() and (time.time() - lock.stat().st_mtime) > _lock_stale_after():
            lock.unlink(missing_ok=True)          # a crashed run, not a live one
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.close(fd)
        return True
    except FileExistsError:
        return False
    except OSError:
        return False


def _throwaway(root: Path) -> Dict[str, Any]:
    return {"path": Path(tempfile.mkdtemp(prefix="agentcc_", dir=str(root))),
            "persistent": False, "resumed": False}


def session_dir(thread_id: Optional[str]) -> Dict[str, Any]:
    """The directory this run works in: the conversation's own, or a throwaway.

    Returns ``{"path", "persistent", "resumed"}``. ``resumed`` means the CLI has session
    history here already, which is what ``--continue`` needs to mean anything.
    """
    root = _work_root_path()
    if not thread_id or not persistence_enabled():
        return _throwaway(root)
    _sweep_sessions(root)
    from agent_runtime.executor_factory import child_thread_id

    # Same scoping the execute_code sandbox and the QGIS jobs already use. The name carries
    # a DIGEST as well as a readable slug: sanitising alone is many-to-one — "sess:42" and
    # "sess_42" both sanitise to "sess_42" — and two conversations sharing one directory
    # means one reading the other's files and resuming the other's session.
    scoped = str(child_thread_id(thread_id, "claudepeer"))
    slug = re.sub(r"[^A-Za-z0-9_.-]", "_", scoped)[:40]
    digest = hashlib.sha256(scoped.encode("utf-8")).hexdigest()[:16]
    work = root / f"claudesess_{slug}_{digest}"
    try:
        work.mkdir(parents=True, exist_ok=True)
    except OSError:
        return _throwaway(root)
    if not _acquire(work):
        logger.info("claude peer session %s is busy; using a throwaway directory", slug)
        return _throwaway(root)
    return {"path": work, "persistent": True, "resumed": (work / ".claude").is_dir()}


def _manifest_path(work: Path) -> Path:
    return work / _MANIFEST_NAME


def _fingerprints(work: Path) -> Dict[str, List[float]]:
    """size and mtime for every file the peer could persist, keyed by relative path."""
    out: Dict[str, List[float]] = {}
    try:
        entries = sorted(work.rglob("*"))
    except OSError:
        return out
    for path in entries:
        try:
            if not path.is_file():
                continue
            rel = path.relative_to(work)
            if not rel.parts or rel.parts[0].startswith(".") or rel.parts[0] == "__pycache__":
                continue
            stat = path.stat()
            out[str(rel)] = [float(stat.st_size), float(stat.st_mtime)]
        except OSError:
            continue
    return out


def already_persisted(work: Path) -> set:
    """Files a previous turn already sent to the file store, unchanged since.

    Artifact persistence walks the whole work dir, which was right when the dir lasted one
    run. Across turns it would re-upload every earlier file on every turn — new file ids,
    duplicate download links, and the map layers re-emitted each time. Size and mtime are
    enough to tell "same file" from "the CLI edited it", and a file it edited SHOULD be
    sent again.
    """
    try:
        stored = json.loads(_manifest_path(work).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - no manifest, a corrupt one: persist everything
        return set()
    if not isinstance(stored, dict):
        return set()
    current = _fingerprints(work)
    return {rel for rel, meta in stored.items()
            if isinstance(meta, list) and current.get(rel) == meta}


def record_persisted(work: Path, delivered: Optional[List[str]] = None) -> None:
    """Remember what actually REACHED the file store, so the next turn sends the rest.

    Recording everything on disk was wrong in a way that only bites once the directory
    persists: ``_persist_artifacts`` stops at MAX_ARTIFACTS and swallows per-file upload
    errors, so files it merely WALKED were being marked delivered. On the next turn they
    matched the manifest, went into the exclude set, and were withheld — permanently, and
    a .geojson among them never became a map layer either. The rule is meant to be "a
    duplicate is noise, a withheld one is a lost result"; this had it backwards.

    Cumulative: files delivered on earlier turns stay recorded as long as they are
    unchanged on disk.
    """
    current = _fingerprints(work)
    keep = (already_persisted(work) | {str(rel) for rel in (delivered or [])}) & set(current)
    try:
        _manifest_path(work).write_text(
            json.dumps({rel: current[rel] for rel in sorted(keep)}), encoding="utf-8")
    except OSError:  # pragma: no cover - a manifest that cannot be written re-sends, no worse
        logger.debug("claude peer: could not write the artifact manifest")


def selects_claude(value: Optional[str]) -> bool:
    """Does this AGENT_CODE_PEER value name this backend?

    A pure predicate so the env default and a per-request override are decided by
    the SAME accepted-value set — two copies of it would drift, and the drift
    would show up as a request silently getting a different peer than it asked
    for."""
    return (value or "").strip().lower() in _ACCEPTED_FLAG_VALUES


def is_claude_peer_enabled() -> bool:
    """Whether the deployment default selects Claude Code for the code peer."""
    return selects_claude(os.getenv(CODE_PEER_ENV))


def resolve_claude_settings(model: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Model / credential / base URL for the Claude Code peer.

    Two credentials are accepted, and they are not interchangeable:

    * ``CLAUDE_CODE_OAUTH_TOKEN`` — a long-lived token from ``claude
      setup-token``, which authenticates as a **Claude subscription**. Requests
      count against that person's plan limits, not a metered API balance.
    * ``ANTHROPIC_API_KEY`` — a metered API key.

    The subscription token wins when both are set: you have to run
    ``setup-token`` on purpose, so its presence is the more deliberate signal,
    and silently billing an API key while a token sits unused is the kind of
    surprise that shows up on an invoice.

    Deliberately independent of the ``VLLM_*``/``OPENAI_*`` chain that answers
    the user: this peer talks to Anthropic, so borrowing the deployment's
    OpenAI key would send a key to a host that cannot use it and fail with an
    authentication error naming the wrong provider.
    """
    token = (os.getenv("AGENT_CLAUDE_OAUTH_TOKEN") or os.getenv(_OAUTH_TOKEN_ENV) or "").strip()
    key = (os.getenv("AGENT_CLAUDE_API_KEY") or os.getenv(_API_KEY_ENV) or "").strip()
    base = (os.getenv("AGENT_CLAUDE_BASE_URL") or os.getenv("ANTHROPIC_BASE_URL") or "").strip()
    # A per-request model beats the deployment default. An unknown one is rejected by
    # the CLI LOCALLY — duration_api_ms 0, cost 0 — so a bad pick costs nothing but an
    # error, which is why this is not validated against a hardcoded list here.
    model = (model or os.getenv("AGENT_CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL).strip()
    credential, auth = (token, "subscription") if token else ((key, "api_key") if key else (None, None))
    return {
        "model": model or DEFAULT_CLAUDE_MODEL,
        "auth": auth,
        "credential_env": _OAUTH_TOKEN_ENV if auth == "subscription" else _API_KEY_ENV,
        "credential": credential or None,
        "base_url": base or None,
    }


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _timeout_seconds() -> int:
    return max(30, _int_env("AGENT_CLAUDE_TIMEOUT", DEFAULT_CLAUDE_TIMEOUT))


def sandbox_user() -> str:
    """The uid:gid the sandbox runs as — never root.

    ``_host_user()`` reports the AGENT process's uid, and under Docker-out-of-Docker
    that process runs as root: the compose service needs root to reach the mounted
    Docker socket. Inheriting it makes the sandbox root, and Claude Code then
    refuses outright — *"--dangerously-skip-permissions cannot be used with
    root/sudo privileges for security reasons"* — which surfaces as exit 1 with an
    empty answer and nothing pointing at the cause.

    So root is replaced with an unprivileged uid. The work dir is chmod 0777
    before the run, so any uid can write there, and the agent (root) can read back
    whatever it wrote. Override with ``AGENT_CLAUDE_USER`` if a deployment needs a
    specific one.
    """
    override = (os.getenv("AGENT_CLAUDE_USER") or "").strip()
    if override:
        return override
    user = _host_user()
    if user and not user.startswith("0:"):
        return user
    return DEFAULT_SANDBOX_USER


def build_docker_argv(work: Path, name: str, model: str, prompt: str,
                      base_url: Optional[str] = None,
                      credential_env: str = _API_KEY_ENV,
                      resume: bool = False) -> List[str]:
    """``docker run`` argv for one Claude Code run.

    Hardened like the execute_code sandbox but WITH network — the CLI is useless
    without the Anthropic API. ``HOME=/work`` so all CLI state lands in the
    throwaway work dir (dot-dirs are excluded from artifact persistence).

    ``--bare`` is used ONLY with an API key. Its own help says that under it
    "Anthropic auth is strictly ANTHROPIC_API_KEY or apiKeyHelper via --settings
    (OAuth and keychain are never read)" — so passing it alongside a
    subscription token would ignore the credential and fail as if none had been
    supplied. That is a silent misconfiguration, not an error message, which is
    why the flag is conditional rather than always-on.
    """
    argv = [
        "docker", "run", "--rm", "--init", "--name", name,
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--read-only",
        "--memory", os.getenv("AGENT_CLAUDE_MEMORY", DEFAULT_CLAUDE_MEMORY),
        "--cpus", os.getenv("AGENT_CLAUDE_CPUS", DEFAULT_CLAUDE_CPUS),
        "--pids-limit", os.getenv("AGENT_CLAUDE_PIDS", DEFAULT_CLAUDE_PIDS),
        "--workdir", "/work",
        "--tmpfs", "/tmp:rw,size=256m,exec",
        "--env", "HOME=/work",
        "--env", "CLAUDE_CONFIG_DIR=/work/.claude",
        # Nothing about a throwaway sandbox should phone home or self-update
        # mid-run: an autoupdate would change the CLI under a running analysis.
        "--env", "DISABLE_AUTOUPDATER=1",
        "--env", "DISABLE_TELEMETRY=1",
        "--env", "DISABLE_ERROR_REPORTING=1",
        "--env", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1",
        # Name-only form: docker copies the value from the client process env
        # (set by run_claude), so the credential never appears in the argv.
        "--env", credential_env,
        "-v", f"{work}:/work:rw",
    ]
    if base_url:
        argv += ["--env", f"ANTHROPIC_BASE_URL={base_url}"]
    network = (os.getenv("AGENT_CLAUDE_NETWORK") or "").strip()
    if network:
        argv += ["--network", network]
    argv += ["--user", sandbox_user()]
    image = os.getenv("AGENT_CLAUDE_IMAGE", DEFAULT_CLAUDE_IMAGE)
    argv += [
        image, "claude",
        "--print",                          # non-interactive: answer and exit
        "--output-format", "json",          # an envelope with result + cost, not bare text
    ]
    if credential_env == _API_KEY_ENV:
        argv.append("--bare")               # no hooks/LSP/plugins/keychain/CLAUDE.md
    argv += ["--dangerously-skip-permissions"]   # no TTY to approve; see module docstring
    if resume:
        # "Continue the most recent conversation in the current directory" — and the
        # directory is this conversation's own, with CLAUDE_CONFIG_DIR inside it. Verified
        # end to end: a resumed run answered from what it had written on a previous turn
        # while a fresh one said it had no record of it. Only passed when history exists,
        # since --continue with nothing to continue is an error rather than a no-op.
        argv.append("--continue")
    # Unset by default, and that is the deliberate choice: a peer that can install a
    # package, read the traceback and try again is the reason to run one at all. A
    # deployment that wants a narrower surface names the tools it will allow, e.g.
    # "Bash Read Write Edit" — which also drops WebFetch/WebSearch, the two that reach the
    # open web outside this agent's own caps.
    allowed = (os.getenv("AGENT_CLAUDE_ALLOWED_TOOLS") or "").strip()
    if allowed:
        argv += ["--allowedTools", *allowed.replace(",", " ").split()]
    argv += ["--model", model, prompt]
    return argv


def neutralize_instruction_files(work: Path, staged: Optional[List[str]] = None) -> List[str]:
    """Rename anything a user UPLOADED that the CLI would read as INSTRUCTIONS.

    A file called ``CLAUDE.md`` is not data to a Claude Code session — it is a
    brief, loaded automatically, for an agent that runs here with tool
    permissions skipped and network access. ``--bare`` turns that discovery off,
    but subscription auth cannot use ``--bare``, so the door has to be closed
    here as well as there.

    Scoped to files staged THIS TURN, which matters now that the directory
    persists. Sweeping the whole directory was right when it was a throwaway and
    ``.claude`` could only have come from an upload; with persistence it is the
    CLI's own state from the previous turn, and renaming it took the session
    history out from under ``--continue`` — observed live: the peer answered "I
    had to search for the file, this conversation had no memory of it", and its
    config files were uploaded as artifacts out of ``uploaded_.claude/``.

    Renamed rather than deleted: the user uploaded it, so it stays available as
    data and as a downloadable artifact under a name that is not a directive.
    Returns the names that were moved.
    """
    moved: List[str] = []
    allowed = {str(n) for n in (staged or [])}
    try:
        entries = list(work.iterdir())
    except OSError:
        return moved
    for entry in entries:
        if entry.name.lower() not in _INSTRUCTION_FILENAMES:
            continue
        # `.claude` is the CLI's OWN state directory once the work dir persists, so it is
        # only touched when THIS turn's upload created it. A CLAUDE.md is never legitimate
        # peer state, so it is neutralised wherever it came from — including one the CLI
        # re-created by renaming last turn's uploaded_CLAUDE.md back, which the
        # staged-only rule would have waved through on a turn with no attachments.
        if (entry.name.lower() == ".claude" and staged is not None
                and entry.name not in allowed):
            continue
        target = entry.with_name(f"uploaded_{entry.name}")
        try:
            entry.rename(target)
            moved.append(entry.name)
        except OSError:
            continue
    return moved


def parse_cli_output(stdout: str) -> Dict[str, Any]:
    """Pull the answer out of ``--output-format json``, tolerating anything else.

    The envelope carries what a caller wants to report — the text, whether the
    run errored, what it cost, how many turns it took. If it is not JSON (an
    early crash, a usage message), the raw text is still the best answer
    available, so this never raises and never returns nothing.
    """
    text = _strip_ansi(stdout or "").strip()
    if not text:
        return {"answer": "", "envelope": None}
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return {"answer": text, "envelope": None}
    if isinstance(obj, list):                      # stream-json, if someone overrides it
        obj = next((x for x in reversed(obj) if isinstance(x, dict) and "result" in x), None)
        if obj is None:
            return {"answer": text, "envelope": None}
    if not isinstance(obj, dict):
        return {"answer": text, "envelope": None}
    answer = obj.get("result")
    if not isinstance(answer, str):
        answer = text
    return {
        "answer": answer,
        "envelope": {k: obj.get(k) for k in
                     ("is_error", "subtype", "num_turns", "total_cost_usd", "duration_ms",
                      "session_id") if k in obj},
    }


def run_claude(
    prompt: str,
    *,
    input_file_ids: Optional[List[str]] = None,
    timeout: Optional[int] = None,
    model: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """One sandboxed ``claude --print`` run; returns a JSON-serializable result dict.

    With a ``thread_id`` the conversation keeps its project directory between turns, so the
    CLI can carry on with the files it wrote and the packages it installed — and resume its
    own session history. The CONTAINER is still fresh every run.
    """
    settings = resolve_claude_settings(model)
    model = str(settings["model"])
    if not settings["credential"]:
        return {
            "ok": False, "exit_code": None, "answer": "", "stderr": "", "timed_out": False,
            "error": (f"The Claude Code peer needs a credential: either {_OAUTH_TOKEN_ENV} "
                      f"(from `claude setup-token`, authenticating as a Claude subscription) "
                      f"or {_API_KEY_ENV} (metered API billing). Neither is set, and neither "
                      "is shared with the OpenAI-compatible endpoint that answers the user."),
            "artifacts": [], "backend": "claude-docker", "model": model,
        }
    timeout = int(timeout or _timeout_seconds())
    try:
        session = session_dir(thread_id)
        work, persistent, resumed = session["path"], session["persistent"], session["resumed"]
    except OSError as exc:
        return {
            "ok": False, "exit_code": None, "answer": "", "stderr": "", "timed_out": False,
            "error": (f"claude work dir unavailable: {exc}. "
                      "Check AGENT_CODE_EXEC_WORK_ROOT and its bind mount in the deployment."),
            "artifacts": [], "backend": "claude-docker", "model": model,
        }
    try:
        staging = _stage_conversation_files(work, input_file_ids)
        renamed = neutralize_instruction_files(work, staging["staged"])
        try:
            os.chmod(work, 0o777)  # non-root container user must write here
        except OSError:
            pass
        name = f"agentcc_{uuid.uuid4().hex[:12]}"
        credential_env = str(settings["credential_env"])
        argv = build_docker_argv(work, name, model, prompt, settings["base_url"], credential_env,
                                 resume=resumed)
        env = {**os.environ, credential_env: str(settings["credential"])}
        exit_code: Optional[int] = None
        stdout, stderr, timed_out, error = "", "", False, None
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=timeout + 5)
            exit_code, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as exc:
            subprocess.run(["docker", "kill", name], capture_output=True)
            stdout, stderr = str(exc.stdout or ""), str(exc.stderr or "")
            timed_out, error = True, f"claude run timed out after {timeout}s"
        except FileNotFoundError:
            error = "docker executable not found"
        except Exception as exc:  # pragma: no cover - defensive
            error = f"{type(exc).__name__}: {exc}"

        parsed = parse_cli_output(stdout)
        envelope = parsed["envelope"] or {}
        excluded = (set(staging["staged"]) | {f"uploaded_{n}" for n in renamed}
                    | already_persisted(work))
        artifacts = _persist_artifacts(work, excluded)
        if persistent:
            record_persisted(work, [a.get("path") for a in artifacts if a.get("path")])
        # The tool path runs these checks inside add_map_layer. This peer has none of the
        # AGENT's tools, so
        # without this nothing between the CLI and the user ever looks at what it wrote —
        # and a blank figure is exactly what a peer's own summary will not mention. Runs
        # here because the work dir is deleted in the finally below.
        from agent_runtime.layer_qa import inspect_artifacts

        qa = inspect_artifacts(str(work), [a.get("filename") for a in artifacts])
        # No add_map_layer here, but it still wrote geodata. Turn it into descriptors here,
        # while the work dir still exists — the wrapper emits them from the request context.
        from agent_runtime.map_layers import layers_for_artifacts

        map_layers = layers_for_artifacts(work, artifacts)
        result: Dict[str, Any] = {
            # The CLI can exit 0 and still report is_error in the envelope, so both count.
            "ok": (error is None and not timed_out and exit_code == 0
                   and not envelope.get("is_error")),
            "exit_code": exit_code,
            "answer": _clip(parsed["answer"]),
            "stderr": _clip(_strip_ansi(stderr).strip()),
            "timed_out": timed_out,
            "error": error,
            "artifacts": artifacts,
            "backend": "claude-docker",
            "model": model,
            # Which account paid, in the record rather than inferred from a bill.
            "auth": settings["auth"],
            # Which project this ran in, and whether it picked up where it left off.
            "session": ("resumed" if resumed else "new") if persistent else "ephemeral",
        }
        if renamed:
            result["renamed_instruction_files"] = renamed
        if qa:
            result["output_warnings"] = qa
        if map_layers:
            result["map_layers"] = map_layers
            # The supervisor's delivery check reads this off the execution record; without
            # it a turn that DID put something on the map still counts as undelivered.
            result["on_map"] = True
        # What the run cost is part of the record, not a detail: this peer spends
        # on a different account from the one answering the user.
        for key in ("num_turns", "total_cost_usd", "duration_ms", "session_id"):
            if envelope.get(key) is not None:
                result[key] = envelope[key]
        if staging["staged_info"]:
            result["input_files"] = staging["staged_info"]
        if staging["errors"]:
            result["input_file_errors"] = staging["errors"]
        if staging["skipped"]:
            result["input_files_skipped"] = staging["skipped"]
        return result
    finally:
        if persistent:
            # The project stays; only this run's claim on it is released.
            (work / _LOCK_NAME).unlink(missing_ok=True)
        else:
            shutil.rmtree(work, ignore_errors=True)


def run_claude_code_peer(
    query: str,
    evidence: Optional[List[Any]] = None,
    state: Optional[Dict[str, Any]] = None,
    input_file_ids: Optional[List[str]] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Code-peer adapter: the same flat shape as ``default_code_fn`` and the
    opencode peer, so synthesis and the trace pipeline stay agnostic to which
    backend produced the result."""
    from agent_runtime.map_layers import build_map_layers
    from agent_runtime.streaming_trace import emit_trace_event

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
        query, evidence, (state or {}).get("analysis_results"),
        staged_names=staged_names or None,
    )
    call_args = {"model": resolve_claude_settings(model)["model"], "prompt_chars": len(prompt)}
    emit_trace_event("tool_call", {"name": "claude_run", "args": call_args}, node="code")
    result = run_claude(prompt, input_file_ids=input_file_ids, model=model,
                        thread_id=(state or {}).get("thread_id"))
    emit_trace_event(
        "tool_result",
        {
            "name": "claude_run",
            "content": {k: result.get(k) for k in
                        ("ok", "exit_code", "timed_out", "error", "artifacts", "backend",
                         "model", "auth", "session", "num_turns", "total_cost_usd")},
        },
        node="code",
    )
    # No add_map_layer ran, so nothing emitted a map_layer on its behalf. This wrapper
    # runs in the request's trace context — the same place a tool callback would — so the
    # descriptors go out here, through the same build_map_layers boundary every tool's
    # layer crosses, and get the same validation.
    for layer in build_map_layers("claude_run", result):
        emit_trace_event("map_layer", layer, node="code")

    answer = result.get("answer") or ""
    warnings = result.get("output_warnings") or []
    if warnings:
        # Appended to the ANSWER, not left in a field: the peer wrote the summary without
        # looking at its own output, and synthesis reads this text.
        lines = [f"- {w['file']}: {'; '.join(w['problems'])}" for w in warnings]
        answer = "\n\n".join(x for x in (answer, "Checks on the files this run produced "
                                                   "found problems — say so rather than "
                                                   "presenting them as results:\n"
                                                   + "\n".join(lines)) if x)
    if not result.get("ok"):
        failure = result.get("error") or f"claude exited with code {result.get('exit_code')}"
        detail = str(result.get("stderr") or "")[-2000:]
        answer = "\n\n".join(
            x for x in (f"Claude Code peer failed: {failure}", detail, answer) if x
        )
    return {
        "answer": answer,
        "tool_calls": [{"name": "claude_run", "args": call_args}],
        "tool_results": [{"name": "claude_run", "content": result}],
    }


__all__ = [
    "CODE_PEER_ENV",
    "is_claude_peer_enabled",
    "resolve_claude_settings",
    "build_docker_argv",
    "neutralize_instruction_files",
    "parse_cli_output",
    "run_claude",
    "run_claude_code_peer",
]
