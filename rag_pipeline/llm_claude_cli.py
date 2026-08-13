"""`claude` CLI backend for :func:`rag_pipeline.llm_utils.call_llm`.

DEVELOPMENT AND EXPERIMENTS ONLY.
---------------------------------
This exists so extraction batches, eval sweeps and local experiments do not bill an API.
The publication extractor alone runs over ~180 elements, and the retrieval/rerank/audit
paths call ``call_llm`` on every turn, so the recurring cost of the beta work sits here
rather than in the agent peers.

It must NOT back a deployed server. Anthropic's consumer terms restrict access "through
automated or non-human means" to API-key access, and separately forbid making the account
available to others — serving platform users through a personal subscription does both.
Deployed configuration uses self-hosted vLLM or a Console API key; ``check_not_deployed()``
is the assertion that keeps that honest in CI.

Model choice
------------
``CLAUDE_CLI_MODEL`` (default ``sonnet``) — the cost/performance balance for this
workload. Structured extraction into a fixed JSON shape may hold on ``haiku``; reserve
``opus`` for the tail that comes back unparseable. Every call records the model it used via
:func:`last_model` so a result is attributable — a recall figure produced under ``opus`` is
not comparable to one from a self-hosted 7B.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sonnet"
DEFAULT_TIMEOUT = 300

_last_model: Optional[str] = None


class ClaudeCliUnavailable(RuntimeError):
    """The `claude` executable is missing or not usable."""


def is_selected() -> bool:
    return str(os.getenv("LLM_PROVIDER") or "").strip().lower() in {"claude-cli", "claude_cli", "claude"}


def model() -> str:
    return str(os.getenv("CLAUDE_CLI_MODEL") or DEFAULT_MODEL).strip() or DEFAULT_MODEL


def last_model() -> Optional[str]:
    """Model used by the most recent call, for recording in extraction/eval records."""
    return _last_model


def _timeout() -> int:
    try:
        return max(10, int(os.getenv("CLAUDE_CLI_TIMEOUT", str(DEFAULT_TIMEOUT))))
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT


def check_not_deployed() -> None:
    """Raise when this backend is selected in something that looks like a deployment.

    Called from the provider dispatch, so a deployed server configured this way fails at
    the first LLM call with a clear reason instead of silently billing a personal
    subscription for user traffic.
    """
    if not is_selected():
        return
    marker = next((v for v in ("AGENT_DEPLOYED", "KUBERNETES_SERVICE_HOST", "ECS_CONTAINER_METADATA_URI")
                   if os.getenv(v)), None)
    if marker:
        raise RuntimeError(
            f"LLM_PROVIDER=claude-cli is a development-only backend but {marker} is set, "
            "which indicates a deployed environment. Anthropic's consumer terms restrict "
            "automated access to API-key access and forbid serving other users through a "
            "personal subscription. Use LLM_PROVIDER=vllm or openai in deployment."
        )


def _load_token_from_env_file() -> None:
    """Make CLAUDE_CODE_OAUTH_TOKEN available to subprocesses started from a script.

    `claude setup-token` prints a token the developer exports in their own shell. A batch
    script, a test runner, or an agent tool started from elsewhere does not inherit that
    session, so the CLI falls back to the expired keychain token and every call 401s with no
    obvious cause. The rest of this repo already resolves configuration from .env, so read it
    from there too — the environment still wins, and this only fills a gap.
    """
    if str(os.getenv("CLAUDE_CODE_OAUTH_TOKEN") or "").strip():
        return
    from pathlib import Path

    here = Path(__file__).resolve().parent.parent
    for candidate in (here / ".env", Path("/Users/yfkang/i-guide-platform-flask-servers/.env")):
        try:
            if not candidate.exists():
                continue
            for line in candidate.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                if key.strip() != "CLAUDE_CODE_OAUTH_TOKEN":
                    continue
                val = val.strip().strip('"').strip("'")
                if val:
                    os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = val
                    logger.debug("loaded CLAUDE_CODE_OAUTH_TOKEN from %s", candidate)
                    return
        except OSError:
            continue


def available() -> bool:
    return shutil.which("claude") is not None


def use_bare() -> bool:
    """Whether to pass ``--bare``.

    ``--bare`` is what makes a run reproducible: it skips hooks, plugins, auto-memory and
    CLAUDE.md auto-discovery, so this repo's own instructions are not silently prepended to
    every extraction prompt (which would skew results AND make them depend on the checkout).

    But its help text is explicit that under ``--bare`` "Anthropic auth is strictly
    ANTHROPIC_API_KEY or apiKeyHelper ... OAuth and keychain are never read". So --bare and
    subscription auth are mutually exclusive, and the default has to follow the credential
    that is actually present:

      ANTHROPIC_API_KEY set   -> --bare  (reproducible; also the terms-compliant path for
                                          automated use)
      otherwise               -> no --bare (falls back to the interactive login, which is
                                          appropriate for a developer running batches by
                                          hand, at the cost of inheriting project context)

    Override with ``CLAUDE_CLI_BARE=0|1`` when you need to be explicit.
    """
    override = str(os.getenv("CLAUDE_CLI_BARE") or "").strip().lower()
    if override in {"0", "false", "no", "off"}:
        return False
    if override in {"1", "true", "yes", "on"}:
        return True
    return bool(str(os.getenv("ANTHROPIC_API_KEY") or "").strip())


_AUTH_HINT = (
    "claude CLI is not authenticated (or its token has expired). Options, best first:\n"
    "  (a) SUBSCRIPTION, long-lived — run `claude setup-token` once, then export the token it\n"
    "      prints as CLAUDE_CODE_OAUTH_TOKEN. Survives across sessions, so batch runs do not\n"
    "      keep dying on an expired access token. Requires CLAUDE_CLI_BARE=0 (--bare never\n"
    "      reads OAuth). Check state with `claude auth status`.\n"
    "  (b) SUBSCRIPTION, interactive — run `claude` and `/login`. Same constraint, but the\n"
    "      access token expires and has to be refreshed by hand.\n"
    "  (c) API KEY — export ANTHROPIC_API_KEY=... Works with --bare, so runs are reproducible\n"
    "      (no CLAUDE.md or hooks injected), and it is the path Anthropic's terms require for\n"
    "      automated use. Costs API credit.\n"
    "Or set LLM_PROVIDER=vllm|openai to bypass this backend entirely."
)


def _is_auth_failure(text: str) -> bool:
    lowered = (text or "").lower()
    return any(s in lowered for s in (
        "not logged in", "please run /login", "oauth access token has expired",
        "failed to authenticate", "401",
    ))


_DEFAULT_SYSTEM = ("You are a language model answering a single request. Answer directly and "
                   "completely. You have no tools and no files to consult.")

# Every tool the CLI would otherwise expose. Denied by name because --bare (which would also
# do this) cannot read OAuth, so a subscription run has to be constrained explicitly.
#
# The second group matters more than it looks. When this list covered only the file/shell
# tools, the CLI still advertised its REMAINING tools to the model — and under a long prompt
# the model believed that list over the one described in the prompt. Observed verbatim in an
# I-GUIDE agent turn: "Only a limited set of tools (AskUserQuestion, ScheduleWakeup,
# ShareOnboardingGuide, Skill, and ToolSearch) are callable here" — precisely the tools left
# allowed at the time — followed by a refusal to run code because `execute_code` was
# "not available in this environment", while it was bound and working. Leaving the CLI with
# ZERO tools of its own removes the competing list.
_AGENT_TOOLS = (
    "Bash,Read,Edit,Write,Glob,Grep,WebFetch,WebSearch,Task,NotebookEdit,"
    "TodoWrite,SlashCommand,KillShell,BashOutput,"
    "AskUserQuestion,ScheduleWakeup,ShareOnboardingGuide,Skill,ToolSearch,Artifact,Monitor,"
    "ReportFindings,SendUserFile,Workflow,CronCreate,CronList,CronDelete,"
    "TaskCreate,TaskUpdate,TaskList,TaskGet,TaskOutput,TaskStop,SendMessage,"
    "EnterPlanMode,ExitPlanMode,LSP,PushNotification,RemoteTrigger,DesignSync,"
    "EnterWorktree,ExitWorktree"
)


@lru_cache(maxsize=1)
def _neutral_cwd() -> str:
    """An empty directory to run the CLI from.

    Without it the CLI inherits the current project: it discovers CLAUDE.md, reads the repo
    and behaves like the coding agent rather than a model. Measured on one tool-selection
    prompt from this repo's root: **78.8s across 4 turns**, and the "answer" was commentary on
    this repo's own source. From an empty directory with the flags below: **3.5s, 1 turn**,
    returning exactly the requested JSON.
    """
    import tempfile
    path = os.path.join(tempfile.gettempdir(), "iguide_claude_cli_cwd")
    os.makedirs(path, exist_ok=True)
    return path


def _isolation_argv() -> list:
    """Flags that stop `claude -p` from behaving like an agent.

    ``--bare`` does all of this and more, but it never reads OAuth, so it is unavailable on
    the subscription path this project uses for development. These flags are the
    subscription-compatible equivalent. Set ``CLAUDE_CLI_ISOLATE=0`` to run without them (for
    debugging what the agent would do), accepting the latency and the project contamination.
    """
    if str(os.getenv("CLAUDE_CLI_ISOLATE") or "").strip().lower() in {"0", "false", "no", "off"}:
        return []
    return ["--disallowed-tools", _AGENT_TOOLS,
            "--strict-mcp-config",          # ignore every MCP server not passed explicitly
            "--setting-sources", "",        # no user/project/local settings, no CLAUDE.md
            "--no-session-persistence"]


def _build_argv(exe: str, prompt: str, mdl: str, system: Optional[str] = None) -> list:
    argv = [exe, "-p", prompt, "--output-format", "json", "--model", mdl]
    if use_bare():
        argv.append("--bare")
    else:
        # Under --bare these are redundant; without it they are what keeps a call cheap.
        argv += _isolation_argv()
    argv += ["--system-prompt", system or _DEFAULT_SYSTEM]
    budget = str(os.getenv("CLAUDE_CLI_MAX_BUDGET_USD") or "").strip()
    if budget:
        argv += ["--max-budget-usd", budget]
    return argv


def call(prompt: str, *, system: Optional[str] = None) -> str:
    """Run one `claude -p` turn, retrying only a SIGNAL death.

    A negative ``returncode`` means the subprocess was killed by a signal — observed live as
    ``exited -11`` (SIGSEGV) with empty stderr, twice in one server log. That is a crash of the
    tool, not an answer about the request, and it is transient by nature. Everything else is
    NOT retried: an auth failure, a budget refusal, a timeout and an API error are all reproducible
    and re-running them wastes the user's quota to reach the same conclusion slower.

    Why this matters more than a dev-only backend suggests: one crash killed a turn that had
    already completed two search sweeps, resolved a bounding box, selected a library method and
    written an evidence summary. Retrying costs one subprocess; not retrying costs all of that.
    """
    attempts = max(1, _signal_retries() + 1)
    last: Optional[BaseException] = None
    for attempt in range(attempts):
        try:
            return _call_once(prompt, system=system)
        except _Transient as exc:
            last = exc
            if attempt + 1 < attempts:
                logger.warning("claude CLI %s; retrying (%d of %d)",
                               exc, attempt + 2, attempts)
                time.sleep(min(2.0 * (attempt + 1), 5.0))
                continue
            # Out of retries: surface it as the RuntimeError callers already handle, with the
            # attempt count in the message so a persistent crash is distinguishable from a blip.
            raise RuntimeError(f"{exc} after {attempts} attempt(s)") from exc
    raise RuntimeError(str(last) if last else "claude CLI failed")   # unreachable


class _Transient(RuntimeError):
    """A failure worth retrying. Internal: `call` converts it to RuntimeError when exhausted."""


class _SignalDeath(_Transient):
    """The CLI was killed by a signal."""


class _TransientApi(_Transient):
    """The upstream API returned a status that resolves on its own."""


# Statuses that mean "ask again", not "this request is wrong". 429 is a rate limit and 529 is
# Anthropic's overloaded signal; both are explicitly retryable, and 5xx is a server fault rather
# than a property of the prompt. 401/403/400 are excluded on purpose: retrying an expired token
# or a malformed request reaches the same answer slower while spending the user's quota.
_TRANSIENT_API_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})


def _signal_retries() -> int:
    """Retries for a signal death only. 0 disables, which keeps the old behaviour available."""
    raw = (os.getenv("CLAUDE_CLI_SIGNAL_RETRIES") or "2").strip()
    try:
        return max(0, min(5, int(raw)))
    except ValueError:
        return 2


def _call_once(prompt: str, *, system: Optional[str] = None) -> str:
    """Run one non-interactive `claude -p` turn and return its text.

    ``--output-format json`` is used rather than plain text because the wrapper object
    carries ``is_error`` / ``subtype`` / ``result`` even on a failed run — and the CLI exits
    non-zero *while still emitting that JSON*, so the reason must be parsed before the exit
    code is judged. Reading the exit code first turns "not logged in" into an unreadable
    dump of usage counters.
    """
    global _last_model

    exe = shutil.which("claude")
    if not exe:
        raise ClaudeCliUnavailable(
            "LLM_PROVIDER=claude-cli but the `claude` executable is not on PATH. "
            "Install Claude Code, or set LLM_PROVIDER=vllm|openai."
        )

    _load_token_from_env_file()
    mdl = model()
    try:
        proc = subprocess.run(_build_argv(exe, prompt, mdl, system), capture_output=True,
                              text=True, timeout=_timeout(), cwd=_neutral_cwd())
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"claude CLI timed out after {_timeout()}s (model={mdl})") from exc

    raw = (proc.stdout or "").strip()
    payload = None
    if raw:
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = None

    # Parse BEFORE judging returncode: a failed run still returns a structured reason.
    if isinstance(payload, dict):
        detail = str(payload.get("result") or "")

        # A COMPLETED answer is never thrown away over an exit status. The CLI writes its result
        # object when the turn finishes; if the process is then killed during teardown, the work
        # is done and paid for. The previous order raised
        #   RuntimeError: claude CLI error (exit=-11): REAL ANSWER
        # -- discarding the answer into the text of the exception complaining about it.
        if not payload.get("is_error") and isinstance(payload.get("result"), str):
            if proc.returncode != 0:
                logger.warning("claude CLI returned a complete result then exited %s; "
                               "using the result", proc.returncode)
            _last_model = mdl
            return payload["result"]

        if _is_auth_failure(detail):
            raise ClaudeCliUnavailable(f"{detail.strip()}\n\n{_AUTH_HINT}")

        # A retryable upstream status. The CLI reports it structurally as `api_error_status`, so
        # this is a field read rather than a guess at the wording of an error string.
        status = payload.get("api_error_status")
        if isinstance(status, int) and status in _TRANSIENT_API_STATUSES:
            raise _TransientApi(
                f"upstream API returned {status} (model={mdl}), which is transient"
                f"{': ' + detail[:160] if detail else ''}")

        # Killed by a signal WITH a payload: still a crash, still transient, still retried.
        # Checking the signal only in the no-payload branch below made the retry unreachable
        # whenever the dying process had already flushed partial JSON -- which is the common
        # case, since it writes progress as it goes.
        if proc.returncode < 0:
            raise _SignalDeath(
                f"was killed by signal {-proc.returncode} (model={mdl}) after emitting "
                f"{'an error payload' if payload.get('is_error') else 'partial output'}"
                f"{': ' + detail[:160] if detail else ''}")

        if payload.get("is_error") or proc.returncode != 0:
            raise RuntimeError(
                f"claude CLI error (model={mdl}, subtype={payload.get('subtype')}, "
                f"exit={proc.returncode}): {detail[:300]}"
            )

    if proc.returncode != 0:
        tail = (proc.stderr or raw or "").strip()[-400:]
        if _is_auth_failure(tail):
            raise ClaudeCliUnavailable(f"{tail}\n\n{_AUTH_HINT}")
        if proc.returncode < 0:
            # Killed by a signal (-11 = SIGSEGV observed). Distinguished here rather than in the
            # caller because only this frame knows it was a signal and not an exit status: the
            # returncode is normalised away by the time a RuntimeError message is read.
            raise _SignalDeath(
                f"was killed by signal {-proc.returncode} (model={mdl})"
                f"{': ' + tail if tail else ' with no diagnostic output'}")
        raise RuntimeError(f"claude CLI exited {proc.returncode} (model={mdl}): {tail}")

    if not raw:
        raise RuntimeError(f"claude CLI produced no output (model={mdl})")
    logger.warning("claude CLI output had no string `result`; using it verbatim.")
    _last_model = mdl
    return raw


def preflight() -> dict:
    """One-shot diagnosis, so an auth problem is a single command away from an answer.

        python -m rag_pipeline.llm_claude_cli
    """
    # Resolve the .env token BEFORE reporting, or the diagnostic says "no token" for a
    # working setup — the exact confusion this function exists to prevent.
    _load_token_from_env_file()
    info = {
        "executable": shutil.which("claude"),
        "model": model(),
        "bare": use_bare(),
        "anthropic_api_key_set": bool(str(os.getenv("ANTHROPIC_API_KEY") or "").strip()),
        "oauth_token_set": bool(str(os.getenv("CLAUDE_CODE_OAUTH_TOKEN") or "").strip()),
        "ok": False,
        "detail": "",
    }
    if not info["executable"]:
        info["detail"] = "claude not on PATH"
        return info
    try:
        info["detail"] = call("Reply with exactly: OK").strip()[:80]
        info["ok"] = True
    except Exception as exc:  # noqa: BLE001 — reporting the failure IS the purpose
        info["detail"] = f"{type(exc).__name__}: {exc}"
    return info


if __name__ == "__main__":  # pragma: no cover
    import pprint
    os.environ.setdefault("LLM_PROVIDER", "claude-cli")
    pprint.pprint(preflight())
