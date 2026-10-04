"""Agent executor builders and LLM configuration.

Constructs LangGraph agents (via ``langchain.agents.create_agent``) for each
agent role, wires system prompts, and manages LLM / checkpointer setup.
"""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, OrderedDict
from uuid import uuid4

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

from agent_runtime.tool_policy import collect_tools as _collect_tools
from agent_runtime.streaming_trace import attach_streaming_callbacks

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent system prompts
# ---------------------------------------------------------------------------

# Agent personas live in dedicated prompt modules. The shared SearchAgent/CodeAgent personas
# and the generic default come from agent_runtime.prompts; importing them here also keeps
# existing `from agent_runtime.executor_factory import SEARCH_AGENT_PROMPT/CODE_AGENT_PROMPT`
# callers working. Legacy personas (ANALYSIS_AGENT_PROMPT / ORCHESTRATOR_AGENT_PROMPT) live in
# agent_runtime.legacy.prompts; the supervisor's live in agent_runtime.supervisor.prompts.
from agent_runtime.prompts import (  # noqa: E402  (kept here for back-compat re-export)
    CODE_AGENT_PROMPT,
    DEFAULT_AGENT_PROMPT,
    SEARCH_AGENT_PROMPT,
)

class BoundedInMemorySaver(InMemorySaver):
    """``InMemorySaver`` with LRU eviction of whole threads.

    The stock saver keeps every thread's checkpoints for the process lifetime — a
    slow but unbounded memory leak in a long-lived server (each conversation, plus
    its ``<thread>::orchestrator``/``::code``… child threads, lives forever).

    This caps the number of distinct thread ids; the least-recently-written thread
    is evicted past the cap (``AGENT_CHECKPOINT_MAX_THREADS``, default 500).
    Eviction is best-effort and never breaks ``put`` — if LangGraph internals differ
    in a future version, tracking simply no-ops rather than raising.

    NOTE (multi-worker contract): this is **process-local**, like ``session_memory``.
    Across multiple workers, multi-turn continuity requires sticky sessions (route a
    conversation's ``thread_id`` to the same worker) or persistent memory
    (``use_persistent_memory`` / OpenSearch). See ``session_memory`` for the
    session-history analogue.
    """

    def __init__(self, *args: Any, max_threads: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if max_threads is None:
            try:
                max_threads = max(1, int(os.getenv("AGENT_CHECKPOINT_MAX_THREADS", "500")))
            except (TypeError, ValueError):
                max_threads = 500
        self._max_threads = max_threads
        self._thread_lru: "OrderedDict[str, bool]" = OrderedDict()

    def put(self, *args: Any, **kwargs: Any) -> Any:
        result = super().put(*args, **kwargs)
        config = args[0] if args else kwargs.get("config")
        try:
            thread_id = ((config or {}).get("configurable") or {}).get("thread_id")
        except Exception:
            thread_id = None
        if thread_id:
            self._touch_thread(str(thread_id))
        return result

    def _touch_thread(self, thread_id: str) -> None:
        lru = self._thread_lru
        lru[thread_id] = True
        lru.move_to_end(thread_id)
        while len(lru) > self._max_threads:
            old_thread, _ = lru.popitem(last=False)
            self._evict_thread(old_thread)

    def _evict_thread(self, thread_id: str) -> None:
        # Best-effort removal across the known InMemorySaver internals; never raise.
        try:
            storage = getattr(self, "storage", None)
            if isinstance(storage, dict):
                storage.pop(thread_id, None)
        except Exception:
            pass
        try:
            writes = getattr(self, "writes", None)
            if isinstance(writes, dict):
                for key in [k for k in list(writes.keys()) if isinstance(k, tuple) and k and k[0] == thread_id]:
                    writes.pop(key, None)
        except Exception:
            pass


DEFAULT_CHECKPOINTER = BoundedInMemorySaver()


class AgentInvocationError(RuntimeError):
    """Runtime error with structured diagnostics from an agent invocation."""

    def __init__(self, message: str, *, diagnostics: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {}

# ---------------------------------------------------------------------------
# Environment / LLM helpers
# ---------------------------------------------------------------------------


def load_env() -> None:
    """Load the single canonical .env from the repo root."""
    repo_root = Path(__file__).resolve().parent.parent
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)


load_env()


def normalize_openai_base_url(url: Optional[str]) -> Optional[str]:
    """Strip trailing path segments from an OpenAI-compatible base URL."""
    if not url:
        return None
    normalized = url.rstrip("/")
    suffixes = (
        "/chat/completions",
        "/completions",
        "/chat",
    )
    lowered = normalized.lower()
    for suffix in suffixes:
        if lowered.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


# Deliberately NO max_tokens. AnvilGPT's qwen3 line are reasoning models: they spend their
# first tokens on `reasoning_content` and only then write `content`, so a tight ceiling returns
# finish_reason="length" with content=None — an EMPTY answer that reads as a model failure
# rather than a truncation. Measured on qwen3.6:27b: max_tokens=20 produced no content at all
# (all 20 spent thinking); 800 answered in 41; unset completes normally at 57. The successor
# behaves the same way and more so — qwen3.8:27b spent 34 of 38 completion tokens on reasoning
# just to answer "OK" (2026-10-01) — so the ceiling stays off. Since the endpoint imposes no
# small default of its own, any ceiling invented here could only truncate a long answer.


def _anvilgpt_settings() -> Optional[Dict[str, Any]]:
    """AnvilGPT (Purdue RCAC, Open WebUI) config, or None when it is not configured.

    Selected by AGENT_LLM_PROVIDER=anvilgpt, so setting the variables alone never silently
    moves every request onto a different model.
    """
    if (os.getenv("AGENT_LLM_PROVIDER") or "").strip().lower() != "anvilgpt":
        return None
    key = os.getenv("ANVILGPT_KEY")
    if not key:
        raise RuntimeError(
            "AGENT_LLM_PROVIDER=anvilgpt but ANVILGPT_KEY is unset. Create a key at "
            "https://anvilgpt.rcac.purdue.edu (avatar -> Settings -> Account -> API Keys)."
        )
    # Its chat path is /api/chat/completions, so the OpenAI-compatible base is /api — which
    # normalize_openai_base_url already produces by stripping the /chat/completions suffix.
    base_url = normalize_openai_base_url(
        os.getenv("ANVILGPT_URL") or "https://anvilgpt.rcac.purdue.edu/api/chat/completions")
    return {
        "api_key": key,
        "base_url": base_url,
        # Open WebUI names models like "qwen3.8:27b" — NOT the HuggingFace "Qwen/Qwen3.8-27B"
        # form a vLLM server uses. Ask /api/models for the exact id; a wrong one 404s.
        "model": os.getenv("ANVILGPT_MODEL") or "qwen3.8:27b",
    }


# --- reasoning carry-over ------------------------------------------------------------------
# Reasoning models served over an OpenAI-compatible shim (AnvilGPT/Ollama's gpt-oss:120b is
# the one we hit) return their chain-of-thought in `message.reasoning_content` with
# `content: None`, alongside the tool call. langchain_openai 1.6 parses neither field onto the
# message — not into `additional_kwargs`, not into `response_metadata` — so the assistant turn
# that gets replayed on the next step is `{role: assistant, content: None, tool_calls: [...]}`:
# a bare tool call with no trace of why it was made.
#
# The model then re-derives its plan from the user request alone and reaches the same
# conclusion, so it re-issues the same call. Observed live: `admin_boundary` called five times
# in one turn — the first two with IDENTICAL arguments — then `execute_code` cycling through
# guesses at a filename it had already been given.
#
# This applies ONLY to OpenAI-*compatible shims* that put the chain of thought on the wire.
# OpenAI proper never does: measured against api.openai.com with tools bound, the assistant
# message keys are exactly ['annotations','audio','content','function_call','refusal','role',
# 'tool_calls'] for gpt-4o, gpt-5.6-luna, gpt-5.2 and o4-mini alike — `reasoning_content` is
# ABSENT, not null, even when the model demonstrably reasoned (gpt-5.2 at effort=high burned
# 128-214 reasoning tokens and returned none of the text; only
# usage.completion_tokens_details.reasoning_tokens records it). So on the OpenAI path this
# subclass is a permanent no-op — `_reasoning_text` finds nothing and returns early — and
# nothing at our layer could preserve what the API does not send. Only `/v1/responses`, which
# we do not use, carries reasoning items across steps.
#
# Do NOT explain gpt-4o's immunity as "its plan lives in content": measured, gpt-4o also
# returns content=None on a tool-calling step. It is immune because it has no chain of thought
# to lose, not because it round-trips one.
#
# So keep it: stash the reasoning on the message, and when the provider left `content` empty on
# a TOOL-CALLING step, promote the reasoning into `content` so it survives serialization the
# way an ordinary assistant turn does. Gated on `tool_calls` on purpose — an intermediate step
# is the only place this is safe; promoting it on a FINAL message would leak deliberation into
# the user's answer.
_REASONING_KEYS = ("reasoning_content", "reasoning", "thinking")
_REASONING_CARRYOVER_MAX = 4000


def _reasoning_text(choice: Any) -> str:
    """The chain-of-thought on a raw API choice, if the provider sent one."""
    msg = (choice or {}).get("message") if isinstance(choice, dict) else None
    if not isinstance(msg, dict):
        return ""
    for key in _REASONING_KEYS:
        val = msg.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _reasoning_preserving_chat_openai() -> Any:
    """``ChatOpenAI`` that keeps a reasoning model's own thinking attached to the message."""
    from langchain_openai import ChatOpenAI

    class ReasoningPreservingChatOpenAI(ChatOpenAI):  # type: ignore[misc]
        def _create_chat_result(self, response: Any, generation_info: Any = None) -> Any:
            result = super()._create_chat_result(response, generation_info)
            try:
                raw = response if isinstance(response, dict) else response.model_dump()
                choices = (raw or {}).get("choices") or []
                for gen, choice in zip(result.generations, choices):
                    msg = getattr(gen, "message", None)
                    reasoning = _reasoning_text(choice)
                    if msg is None or not reasoning:
                        continue
                    msg.additional_kwargs.setdefault("reasoning_content", reasoning)
                    content = msg.content if isinstance(msg.content, str) else ""
                    if not content.strip() and getattr(msg, "tool_calls", None):
                        # Keep the TAIL: the decision the model reached sits at the end.
                        msg.content = reasoning[-_REASONING_CARRYOVER_MAX:]
            except Exception:  # noqa: BLE001 - never fail a turn over a provider quirk
                return result
            return result

    return ReasoningPreservingChatOpenAI


def build_default_llm() -> Any:
    """Build a ``ChatOpenAI`` instance from environment variables.

    Priority: AGENT_LLM_PROVIDER=anvilgpt → VLLM_* → OPENAI_* → defaults.
    """
    try:
        ChatOpenAI = _reasoning_preserving_chat_openai()
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency `langchain-openai`. Install it to use the default LLM builder."
        ) from exc

    anvil = _anvilgpt_settings()
    if anvil:
        return ChatOpenAI(temperature=0.0, **anvil)

    api_key = os.getenv("VLLM_API_KEY") or os.getenv("OPENAI_KEY")
    if not api_key:
        raise RuntimeError("VLLM_API_KEY (or OPENAI_KEY) is required to build the default LangChain LLM.")

    model = (
        os.getenv("VLLM_MODEL")
        or os.getenv("OPENAI_CHAT_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "Qwen/Qwen3.5-9B"
    )
    base_url = normalize_openai_base_url(os.getenv("VLLM_PROXY") or os.getenv("OPENAI_BASE_URL"))

    # Same two guards the per-request builder applies. They live here as well because this
    # path is reachable by env alone: OPENAI_CHAT_MODEL=gpt-5.6-luna would otherwise build a
    # client with no reasoning_effort, and every tool-bound call would 400 ("Function tools
    # with reasoning_effort are not supported ... set reasoning_effort to 'none'"). Not
    # reachable as currently deployed, so this is a guardrail rather than a live outage.
    kwargs: Dict[str, Any] = {"model": model, "api_key": api_key}
    if supports_temperature(model):
        kwargs["temperature"] = 0.0
    forced = resolve_effort(model, None)
    if forced:
        kwargs["reasoning_effort"] = forced
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


# --- explicit per-request provider/model selection ------------------------------------
# The UI can offer a model picker, so a turn needs to be able to say which model it wants
# without changing process-wide env. Absent both, the DEFAULT IS OPENAI gpt-4o: it is what
# the deployment has been validated against, and a reasoning model's latency profile is
# quite different (see the qwen3.8:27b notes above).
DEFAULT_PROVIDER = "openai"
DEFAULT_OPENAI_MODEL = "gpt-4o-2024-11-20"

# Offered in the picker. AnvilGPT's list is fetched live because it changes and a stale
# hardcoded id 404s; these are the fallback if the fetch fails. The roster really does move:
# qwen3.6:27b sat here while it was listed, "Recommended", and serving nothing at all, and by
# 2026-10-01 Purdue had removed it entirely and put qwen3.8:27b in its place. Probe
# /api/models rather than trusting this tuple.
_ANVIL_FALLBACK_MODELS = ("qwen3.8:27b", "qwen3:32b", "qwen3-coder:30b", "qwen3-vl:32b")

# This agent ALWAYS binds function tools, and on /v1/chat/completions the legal
# reasoning_effort values depend on the model — a prefix rule got it wrong in both
# directions and produced hard 400s mid-turn. The table below is what the API actually
# answered when probed with tools attached (one row per model, three probes each, repeated):
#
#   gpt-4o*, gpt-4.1      any effort -> "Unrecognized request argument supplied:
#                         reasoning_effort".  Must not be sent at all.
#   gpt-5.6-*             NO effort -> "Function tools with reasoning_effort are not supported
#                         ... use /v1/responses or set reasoning_effort to 'none'"; 'none' ->
#                         ok; 'low' -> same rejection. So 'none' is REQUIRED, not optional.
#   gpt-5.5, gpt-5.4,     omitted or 'none' -> ok; any real level -> the same "Function tools
#   gpt-5.4-mini          with reasoning_effort are not supported" rejection.
#   gpt-5.2               omitted, 'none', or any level -> ok. The only model that can both
#                         call tools and actually think harder on request.
#   o4-mini               any level -> ok; 'none' -> "does not support 'none' with this model.
#                         Supported values are: 'low', 'medium', 'high', 'xhigh'".
#
# Two ids that were offered cannot serve a tool-using agent at all and are no longer listed:
# gpt-5.5-pro ("This is not a chat model") and gpt-5.3-chat-latest ("has been deprecated").
_EFFORT_NONE: tuple = ()
_EFFORT_ONLY_NONE = ("none",)
_EFFORT_LEVELS = ("low", "medium", "high", "xhigh")
_TOOL_EFFORT: Dict[str, Dict[str, Any]] = {
    "gpt-4o-2024-11-20": {"options": _EFFORT_NONE, "required": None},
    "gpt-4o-mini": {"options": _EFFORT_NONE, "required": None},
    "gpt-4.1-2025-04-14": {"options": _EFFORT_NONE, "required": None},
    "gpt-5.6-luna": {"options": _EFFORT_ONLY_NONE, "required": "none"},
    "gpt-5.6-sol": {"options": _EFFORT_ONLY_NONE, "required": "none"},
    "gpt-5.6-terra": {"options": _EFFORT_ONLY_NONE, "required": "none"},
    "gpt-5.5": {"options": _EFFORT_ONLY_NONE, "required": None},
    "gpt-5.4": {"options": _EFFORT_ONLY_NONE, "required": None},
    "gpt-5.4-mini": {"options": _EFFORT_ONLY_NONE, "required": None},
    "gpt-5.2": {"options": ("none", *_EFFORT_LEVELS), "required": None},
    "o4-mini-2025-04-16": {"options": _EFFORT_LEVELS, "required": None},
}
_OPENAI_MODELS = tuple(_TOOL_EFFORT)

# Every value the API named across those probes, for validating an incoming request.
REASONING_EFFORTS = ("none", *_EFFORT_LEVELS)


def effort_options(model: Optional[str]) -> List[str]:
    """reasoning_effort values this model accepts WITH function tools attached.

    Empty means the argument must not be sent. An unknown id also returns empty: omitting
    the parameter is what every tool-capable model here tolerates, so a model added upstream
    degrades to working-without-effort rather than to a 400.
    """
    return list((_TOOL_EFFORT.get((model or "").strip()) or {}).get("options") or ())


def required_effort(model: Optional[str]) -> Optional[str]:
    """The value this model REQUIRES when tools are attached, if any (gpt-5.6-*)."""
    return (_TOOL_EFFORT.get((model or "").strip()) or {}).get("required")


def supports_reasoning_effort(model: Optional[str]) -> bool:
    """Whether `model` takes a reasoning_effort argument at all (tools attached)."""
    return bool(effort_options(model))


# Some ids reject an explicit temperature outright, and since we send `temperature=0.0` on
# every OpenAI build that made them 100% unusable from the picker — every tool-bound call
# answered HTTP 400 "Unsupported value: 'temperature' does not support 0.0 with this model.
# Only the default (1) value is supported." Measured by sweeping every OpenAI id the picker
# offers, with tools bound and the effort this table forces: gpt-5.5 and o4-mini-2025-04-16
# fail, all nine others accept 0.0, and both failures succeed with the parameter simply
# omitted. So omit it for those two rather than dropping them from the picker — they work
# fine, we were just sending an argument they refuse. Determinism is worth keeping everywhere
# it IS accepted, which is why this is a deny-list and not a blanket removal.
_NO_TEMPERATURE = frozenset({"gpt-5.5", "o4-mini-2025-04-16"})


def supports_temperature(model: Optional[str]) -> bool:
    """Whether an explicit ``temperature`` may be sent for *model* alongside tools."""
    return (model or "").strip() not in _NO_TEMPERATURE


def resolve_effort(model: Optional[str], effort: Optional[str]) -> Optional[str]:
    """The value to actually send for (model, requested effort).

    Coerces rather than raising: the request is already in flight with tools bound, and a
    rejected argument fails the whole turn. The caller logs what it sent.
    """
    want = (effort or "").strip().lower() or None
    allowed = effort_options(model)
    forced = required_effort(model)
    if forced:
        return forced                      # gpt-5.6-*: tools are refused without it
    if not allowed or not want:
        return None
    return want if want in allowed else None


def build_llm(provider: Optional[str] = None, model: Optional[str] = None,
              reasoning_effort: Optional[str] = None) -> Any:
    """Build a chat model for an EXPLICIT provider/model, falling back to the default.

    ``provider=None and model=None`` reproduces :func:`build_default_llm` exactly, so an
    unspecified request behaves as it always has.
    """
    prov = (provider or "").strip().lower()
    effort = (reasoning_effort or "").strip().lower() or None
    if effort and effort not in REASONING_EFFORTS:
        raise ValueError(
            f"reasoning_effort={reasoning_effort!r} is not one of {', '.join(REASONING_EFFORTS)}")
    if not prov and not model and not effort:
        return build_default_llm()
    if not prov:
        # A bare model name: infer the provider from the shape rather than guessing wrong.
        # Open WebUI ids look like "qwen3.8:27b"; OpenAI's never contain a colon.
        prov = ("anthropic" if str(model).startswith("claude-")
                else "anvilgpt" if ":" in str(model) else DEFAULT_PROVIDER)

    # Same subclass as the default builder: the model PICKER is where gpt-oss:120b actually
    # gets selected, so the carry-over has to be here or the fix misses the real path.
    ChatOpenAI = _reasoning_preserving_chat_openai()

    if prov == "anvilgpt":
        key = os.getenv("ANVILGPT_KEY")
        if not key:
            raise ValueError(
                "provider='anvilgpt' needs ANVILGPT_KEY. Create one at "
                "https://anvilgpt.rcac.purdue.edu (avatar -> Settings -> Account -> API Keys).")
        base_url = normalize_openai_base_url(
            os.getenv("ANVILGPT_URL") or "https://anvilgpt.rcac.purdue.edu/api/chat/completions")
        # No max_tokens: qwen3.x reasons before it answers, so a ceiling truncates the thinking
        # and returns an EMPTY content rather than a short answer.
        return ChatOpenAI(model=model or os.getenv("ANVILGPT_MODEL") or "qwen3.8:27b",
                          api_key=key, base_url=base_url, temperature=0.0)
    if prov == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        token = os.getenv("CLAUDE_CODE_OAUTH_TOKEN")
        if not key and not token:
            raise ValueError(
                "provider='anthropic' needs ANTHROPIC_API_KEY, or CLAUDE_CODE_OAUTH_TOKEN "
                "from `claude setup-token` (a subscription credential scoped user:inference, "
                "which this path sends as a Bearer token).")
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise ValueError(
                "provider='anthropic' needs the langchain-anthropic package "
                "(pip install langchain-anthropic)") from exc
        # No max_tokens ceiling for the same reason AnvilGPT gets none — except Anthropic
        # REQUIRES the field, so it is set high rather than left off.
        if effort:
            logger.info("reasoning_effort %r is an OpenAI parameter; not sent to Anthropic",
                        effort)
        # No temperature. The Claude 5 family REFUSES it — "`temperature` is deprecated for
        # this model", HTTP 400, which fails the turn rather than being ignored — while
        # older ids still accept it. Sending nothing works on both, and this provider is
        # here to be selected freely across the list.
        llm = ChatAnthropic(model=model or os.getenv("ANTHROPIC_MODEL")
                            or _ANTHROPIC_DEFAULT_MODEL,
                            api_key=key or "unused-when-bearer",
                            max_tokens=8192,
                            default_headers=_OAUTH_BETA_HEADER if not key else None)
        if not key:
            _use_bearer_client(llm, token)
        return llm
    if prov in ("openai", "default"):
        key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("provider='openai' needs OPENAI_KEY.")
        resolved = (model or os.getenv("OPENAI_CHAT_MODEL") or os.getenv("OPENAI_MODEL")
                    or DEFAULT_OPENAI_MODEL)
        kwargs: Dict[str, Any] = {"model": resolved, "api_key": key}
        if supports_temperature(resolved):
            kwargs["temperature"] = 0.0
        base_url = normalize_openai_base_url(os.getenv("OPENAI_BASE_URL"))
        if base_url:
            kwargs["base_url"] = base_url
        # Send only what this model accepts alongside tools, and send what it REQUIRES even
        # when the caller asked for nothing: a UI that leaves the control set while switching
        # models, or that leaves it empty for gpt-5.6, must not turn into a 400.
        sending = resolve_effort(kwargs["model"], effort)
        if sending:
            kwargs["reasoning_effort"] = sending
        if effort and sending != effort:
            logger.info("reasoning_effort %r not usable with tools on %s; sent %r",
                        effort, kwargs["model"], sending)
        return ChatOpenAI(**kwargs)
    raise ValueError(
        f"unknown provider {provider!r}; expected 'openai', 'anthropic' or 'anvilgpt'")


# Fallback ids when Anthropic's own /v1/models cannot be reached (no key, or the call
# failed). Real ids, not the CLI's `sonnet`/`opus` aliases: the Messages API takes ids,
# and the aliases only exist inside Claude Code.
# The subscription credential is a Bearer token, not an api key, and the API only accepts it
# with this beta header — verified live: the same token sent as x-api-key returns 401
# "invalid x-api-key", and sent as Authorization + this header returns 200.
_OAUTH_BETA_HEADER = {"anthropic-beta": "oauth-2025-04-20"}


def _use_bearer_client(llm: Any, token: str) -> None:
    """Point a ChatAnthropic at SDK clients built with ``auth_token``.

    ChatAnthropic exposes ``anthropic_api_key`` and ``default_headers`` but no auth_token,
    and the SDK sends ``x-api-key`` whenever it has one — which the API prefers over the
    Authorization header and rejects. Handing it clients constructed with ``auth_token``
    omits x-api-key entirely, which is the only combination that authenticates.

    Swapping the private clients is deliberate and pinned by a test: if a future
    langchain-anthropic renames them, that test fails loudly instead of every Claude turn
    failing with an auth error that names the wrong credential.
    """
    import anthropic

    for attr, factory in (("_client", anthropic.Anthropic),
                          ("_async_client", anthropic.AsyncAnthropic)):
        if not hasattr(llm, attr):
            raise ValueError(
                f"this langchain-anthropic exposes no {attr}; a subscription token cannot be "
                "attached. Set ANTHROPIC_API_KEY instead, or pin langchain-anthropic.")
        setattr(llm, attr, factory(auth_token=token, default_headers=_OAUTH_BETA_HEADER))


_ANTHROPIC_FALLBACK_MODELS = (
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-haiku-4-5-20251001",
)
# The default when a request names the provider but no model.
_ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-5"


def list_available_models(*, timeout: float = 6.0) -> Dict[str, Any]:
    """Models offerable in a picker, per provider, with the default marked.

    AnvilGPT is queried live: its catalogue changes, and offering an id it no longer serves
    produces a 404 at request time instead of an honest "unavailable" in the UI.
    """
    # The default the PICKER shows must be the default the agent would actually use, and this
    # dict used to hardcode OpenAI. With AGENT_LLM_PROVIDER=anvilgpt — a supported setting —
    # the label read "Agent default (gpt-4o-2024-11-20)" while every unqualified request ran
    # gpt-oss:120b. active_llm_description() is the one function that resolves the real
    # default, so read it rather than restating its logic and drifting from it.
    active = active_llm_description()
    out: Dict[str, Any] = {
        "default": {"provider": active.get("provider") or DEFAULT_PROVIDER,
                    "model": active.get("model") or os.getenv("OPENAI_CHAT_MODEL")
                    or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL},
        "providers": [],
    }
    openai_models = list(dict.fromkeys(
        [os.getenv("OPENAI_CHAT_MODEL") or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL,
         *_OPENAI_MODELS]))
    out["reasoning_efforts"] = list(REASONING_EFFORTS)
    out["providers"].append({
        "provider": "openai", "label": "OpenAI",
        "configured": bool(os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")),
        "models": [m for m in openai_models if m],
        # Which of them accept reasoning_effort, so the picker can show the control
        # conditionally instead of offering a setting that 400s.
        "reasoning_models": [m for m in openai_models if m and supports_reasoning_effort(m)],
        # The legal values PER MODEL. A single global list is what let the UI offer 'high' on
        # a model that refuses any real level once tools are attached; the picker can now
        # offer exactly what the API accepts, and nothing else.
        "effort_options": {m: effort_options(m) for m in openai_models
                           if m and effort_options(m)},
        # Models that refuse tools unless this exact value is sent, so the UI can show the
        # control as fixed rather than as a choice the user appears to have.
        "effort_required": {m: required_effort(m) for m in openai_models
                            if m and required_effort(m)},
    })

    anvil: Dict[str, Any] = {"provider": "anvilgpt", "label": "AnvilGPT (Purdue RCAC)",
                             "configured": bool(os.getenv("ANVILGPT_KEY")), "models": []}
    if anvil["configured"]:
        base = normalize_openai_base_url(
            os.getenv("ANVILGPT_URL") or "https://anvilgpt.rcac.purdue.edu/api/chat/completions")
        try:
            import requests

            resp = requests.get(f"{base}/models",
                                headers={"Authorization": f"Bearer {os.getenv('ANVILGPT_KEY')}"},
                                timeout=timeout)
            resp.raise_for_status()
            ids = [m.get("id") for m in (resp.json().get("data") or []) if m.get("id")]
            anvil["models"] = sorted(ids)
        except Exception as exc:
            logger.info("AnvilGPT model list unavailable (%s); offering known ids", exc)
            anvil["models"] = list(_ANVIL_FALLBACK_MODELS)
            anvil["stale"] = True
    out["providers"].append(anvil)

    # Anthropic, queried live for the same reason AnvilGPT is: a hardcoded id that has been
    # retired 404s at request time instead of being absent from the picker.
    # `configured` is credential presence, deliberately, and NOT a liveness probe. A probe was
    # tried and removed: on a subscription credential the limits are PER MODEL — measured the
    # same minute, claude-haiku-4-5 answered with tool calls while claude-sonnet-5 and
    # claude-opus-5 both returned 429 — so any verdict cached for a page load is wrong for
    # some model within minutes, in both directions. Credential presence does not flap; the
    # per-request error carries the API's own sentence, which is the only current truth.
    api_key = os.getenv("ANTHROPIC_API_KEY")
    token = os.getenv("CLAUDE_CODE_OAUTH_TOKEN")
    anthropic: Dict[str, Any] = {
        "provider": "anthropic", "label": "Anthropic (Claude)",
        "configured": bool(api_key or token), "models": [],
    }
    if token and not api_key:
        # Set expectations rather than gate: this route works (verified, with tool calls) but
        # is rate-limited per model, and third-party use draws from extra usage rather than
        # the plan's included quota.
        anthropic["caveat"] = ("subscription token: rate-limited per model, and third-party "
                               "use draws from extra usage rather than plan limits")
    if anthropic["configured"]:
        try:
            import requests

            headers = {"anthropic-version": "2023-06-01"}
            if api_key:
                headers["x-api-key"] = api_key
            else:
                headers["Authorization"] = f"Bearer {token}"
                headers.update(_OAUTH_BETA_HEADER)
            resp = requests.get("https://api.anthropic.com/v1/models",
                                headers=headers, timeout=timeout)
            resp.raise_for_status()
            ids = [m.get("id") for m in (resp.json().get("data") or []) if m.get("id")]
            anthropic["models"] = ids or list(_ANTHROPIC_FALLBACK_MODELS)
        except Exception as exc:
            logger.info("Anthropic model list unavailable (%s); offering known ids", exc)
            anthropic["models"] = list(_ANTHROPIC_FALLBACK_MODELS)
            anthropic["stale"] = True
    else:
        # Listed but disabled, with the reason the API itself gave. Absent would read as
        # "this deployment cannot speak Claude"; "configured" would advertise an option that
        # fails on the first turn. Neither is what is true.
        anthropic["models"] = list(_ANTHROPIC_FALLBACK_MODELS)
        anthropic["needs"] = "ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN"
    out["providers"].append(anthropic)
    return out


def active_llm_description() -> Dict[str, Any]:
    """Which provider/model a run would actually use — for logs and smoke tests.

    A silent fallback to OpenAI looks exactly like success, so make the choice inspectable
    rather than inferring it from whether a call worked.
    """
    anvil = _anvilgpt_settings()
    if anvil:
        return {"provider": "anvilgpt", "model": anvil["model"],
                "base_url": anvil["base_url"], "max_tokens": "unset (server default)"}
    if os.getenv("VLLM_MODEL") or os.getenv("VLLM_PROXY"):
        return {"provider": "vllm",
                "model": os.getenv("VLLM_MODEL"),
                "base_url": normalize_openai_base_url(os.getenv("VLLM_PROXY"))}
    return {"provider": "openai",
            "model": os.getenv("OPENAI_CHAT_MODEL") or os.getenv("OPENAI_MODEL"),
            "base_url": normalize_openai_base_url(os.getenv("OPENAI_BASE_URL"))}


# ---------------------------------------------------------------------------
# Message / thread helpers
# ---------------------------------------------------------------------------

def messages_payload(query: str, chat_history: Optional[List[Any]]) -> dict:
    """Normalise *chat_history* + *query* into a ``messages`` payload."""
    messages: List[dict] = []
    for item in chat_history or []:
        if isinstance(item, dict) and "role" in item and "content" in item:
            messages.append({"role": str(item["role"]), "content": str(item["content"])})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            messages.append({"role": str(item[0]), "content": str(item[1])})
        else:
            messages.append({"role": "user", "content": str(item)})
    messages.append({"role": "user", "content": query})
    return {"messages": messages}


def agent_config(thread_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Build a LangGraph configurable dict for *thread_id*."""
    if not thread_id:
        return None
    return {"configurable": {"thread_id": thread_id}}


def resolve_thread_id(thread_id: Optional[str], checkpointer: Optional[Any]) -> Optional[str]:
    """Return *thread_id* or auto-generate one when a checkpointer is present."""
    if thread_id:
        return thread_id
    if checkpointer is None:
        return None
    return f"auto-thread-{uuid4()}"


def child_thread_id(thread_id: Optional[str], label: str) -> Optional[str]:
    """Create a hierarchical child thread id."""
    if not thread_id:
        return None
    return f"{thread_id}::{label}"


def _is_graph_recursion_error(exc: Exception) -> bool:
    return exc.__class__.__name__ == "GraphRecursionError" or "recursion limit" in str(exc).lower()


def _content_snippet(value: Any, *, limit: int = 700) -> str:
    text = value if isinstance(value, str) else str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else f"{text[:limit]}..."


def _message_diagnostics(messages: List[Any]) -> Dict[str, Any]:
    from agent_runtime.runtime_utils import extract_search_artifacts, message_role, message_text

    artifacts = extract_search_artifacts({"messages": messages})
    tool_calls = artifacts.get("tool_calls") or []
    tool_results = artifacts.get("tool_results") or []
    call_counts = Counter(str(item.get("name") or "unknown_tool") for item in tool_calls)

    recent_messages: List[Dict[str, Any]] = []
    for msg in messages[-12:]:
        item: Dict[str, Any] = {
            "role": message_role(msg),
            "content": _content_snippet(message_text(msg), limit=500),
        }
        tool_call_items = getattr(msg, "tool_calls", None)
        if isinstance(tool_call_items, list) and tool_call_items:
            item["tool_calls"] = [
                {
                    "name": call.get("name", "unknown_tool"),
                    "args": call.get("args", {}),
                    "id": call.get("id"),
                }
                for call in tool_call_items
            ]
        name = getattr(msg, "name", None)
        tool_call_id = getattr(msg, "tool_call_id", None)
        if name:
            item["name"] = str(name)
        if tool_call_id:
            item["tool_call_id"] = str(tool_call_id)
        recent_messages.append(item)

    return {
        "message_count": len(messages),
        "tool_calls": tool_calls,
        "tool_call_counts": dict(call_counts),
        "tool_results": [
            {
                "name": item.get("name"),
                "tool_call_id": item.get("tool_call_id"),
                "content": _content_snippet(item.get("content", ""), limit=700),
            }
            for item in tool_results[-12:]
        ],
        "recent_messages": recent_messages,
    }


def format_agent_diagnostics(diagnostics: Dict[str, Any]) -> str:
    """Render recursion diagnostics as a compact human-readable timeline."""
    lines: List[str] = []
    thread_id = diagnostics.get("thread_id")
    recursion_limit = diagnostics.get("recursion_limit")
    if thread_id or recursion_limit:
        parts = []
        if thread_id:
            parts.append(f"thread={thread_id}")
        if recursion_limit:
            parts.append(f"recursion_limit={recursion_limit}")
        lines.append(f"Stopped before final answer ({', '.join(parts)}).")

    call_counts = diagnostics.get("tool_call_counts")
    if isinstance(call_counts, dict) and call_counts:
        repeated = ", ".join(f"{name} x{count}" for name, count in sorted(call_counts.items()))
        lines.append(f"Tool calls observed: {repeated}.")

    messages = diagnostics.get("recent_messages")
    if isinstance(messages, list) and messages:
        lines.append("Recent interaction timeline:")
        for idx, item in enumerate(messages, start=1):
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").lower()
            content = str(item.get("content") or "").strip()
            name = str(item.get("name") or "").strip()
            tool_calls = item.get("tool_calls")

            if role in {"human", "humanmessage", "user"}:
                label = "User"
                text = content
            elif isinstance(tool_calls, list) and tool_calls:
                label = "LLM tool decision"
                calls = []
                for call in tool_calls:
                    if not isinstance(call, dict):
                        continue
                    args = call.get("args") or {}
                    calls.append(f"{call.get('name', 'unknown_tool')}({json.dumps(args, ensure_ascii=True, default=str)})")
                text = "; ".join(calls)
                if content:
                    text = f"{text} | message: {content}" if text else content
            elif role in {"ai", "aimessage", "assistant"}:
                label = "LLM message"
                text = content or "(empty message)"
            elif role in {"tool", "toolmessage"} or name:
                label = f"Tool result {name}" if name else "Tool result"
                text = content
            else:
                label = role or "Message"
                text = content

            if text:
                lines.append(f"{idx}. {label}: {text}")

    last_final = diagnostics.get("last_final_answer")
    if isinstance(last_final, str) and last_final.strip():
        lines.append(f"Last final answer candidate: {last_final.strip()}")
    return "\n".join(lines).strip()


def _agent_state_diagnostics(
    executor: Any,
    *,
    config: Dict[str, Any],
    exc: Exception,
) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {
        "error_type": exc.__class__.__name__,
        "message": str(exc),
        "recursion_limit": config.get("recursion_limit"),
        "thread_id": ((config.get("configurable") or {}).get("thread_id")),
    }
    get_state = getattr(executor, "get_state", None)
    if not callable(get_state):
        diagnostics["state_error"] = "executor does not expose get_state"
        return diagnostics
    try:
        snapshot = get_state(config)
        values = getattr(snapshot, "values", None)
        if not isinstance(values, dict):
            diagnostics["state_error"] = "state snapshot has no dict values"
            return diagnostics
        messages = values.get("messages")
        if isinstance(messages, list):
            diagnostics.update(_message_diagnostics(messages))
            diagnostics["readable_trace"] = format_agent_diagnostics(diagnostics)
        else:
            diagnostics["state_keys"] = sorted(str(key) for key in values.keys())
    except Exception as state_exc:
        diagnostics["state_error"] = f"{type(state_exc).__name__}: {state_exc}"
    return diagnostics


def invoke_agent_with_payload_fallback(
    executor: Any,
    *,
    query: str,
    chat_history: Optional[List[Any]],
    config: Optional[Dict[str, Any]] = None,
) -> Any:
    """Invoke *executor* trying the modern messages format first, then legacy."""
    msg_payload = messages_payload(query, chat_history)
    legacy_payload = {"input": query, "chat_history": chat_history or []}
    config = attach_streaming_callbacks(config)
    try:
        recursion_limit = max(25, int(os.getenv("AGENT_RECURSION_LIMIT", "60")))
    except (TypeError, ValueError):
        recursion_limit = 60
    config.setdefault("recursion_limit", recursion_limit)
    try:
        return executor.invoke(msg_payload, config=config)
    except Exception as exc:
        text = str(exc).lower()
        # A 400 about tool_call / tool-message ordering means the conversation
        # history is corrupted (an assistant tool_call without its response),
        # NOT a payload-shape mismatch — retrying the legacy payload would just
        # re-send the same broken state. Only fall back for true shape errors.
        message_sequence_error = (
            "tool_call" in text or "tool_calls" in text or "tool_call_id" in text or "tool message" in text
        )
        payload_shape_error = (not message_sequence_error) and (
            ("input" in text and "messages" in text)
            or ("invalid" in text and "messages" in text)
            or ("missing" in text and "messages" in text)
        )
        if payload_shape_error:
            return executor.invoke(legacy_payload, config=config)
        if _is_graph_recursion_error(exc):
            diagnostics = _agent_state_diagnostics(executor, config=config, exc=exc)
            logger.error(
                "Agent invocation hit recursion limit: %s",
                json.dumps(diagnostics, ensure_ascii=True, default=str),
            )
            raise AgentInvocationError(
                "Agent graph recursion limit reached before a final answer.",
                diagnostics=diagnostics,
            ) from exc
        raise


# --- per-turn scoping of a peer's own work -------------------------------------------------
#
# A peer thread OUTLIVES the turn. The supervisor graph is compiled without a checkpointer, so
# its state is per-turn, but each peer's create_agent graph shares the process-global
# DEFAULT_CHECKPOINTER under a stable child id (f"{thread_id}::{label}"), and the map UI sends a
# required thread_id. So `extract_search_artifacts(resp)` walks EVERY prior turn's messages, and
# four verifiers that ask "what happened THIS turn?" get the whole conversation:
#
#   * _has_execution_record -> last turn's execute_code makes this turn's bare code fence read as
#     executed=True, and the corrective retry is skipped;
#   * _repeatedly_failed_tools -> one failure per turn crosses the repeat threshold on turn 2,
#     costing a round trip and reporting a tool as a dead end;
#   * _models_used -> a model from turn 1 is subtracted from this turn's mismatch check;
#   * the search peer's evidence harvest -> turn 1's documents are returned as turn 2's evidence.
#
# Scoping is PER INVOCATION, not per turn: default_analyze_fn invokes the same thread up to three
# times in one turn (main, map retry, mismatch retry) and CONCATENATES the extracted slices, so a
# per-turn watermark would re-yield the first invocation each time — two failures of one tool
# would render as four and trip the threshold on their own.
#
# Cross-turn continuity is deliberate and stays: it is documented at both peer call sites, and
# the intra-turn retries send bare follow-up observations that are meaningless without it. What
# changes is only which slice the verifiers are shown.


def peer_thread_messages(executor: Any, config: Optional[Dict[str, Any]]) -> List[Any]:
    """The messages already checkpointed on this peer thread, or [] when unknowable."""
    try:
        get_state = getattr(executor, "get_state", None)
        if not callable(get_state) or not config:
            return []
        values = getattr(get_state(config), "values", None)
        messages = values.get("messages") if isinstance(values, dict) else None
        return list(messages) if isinstance(messages, list) else []
    except Exception:      # a missing or half-evicted thread simply has no watermark
        return []


def _prefix_is_intact(messages: List[Any], prev_ids: List[Any]) -> bool:
    """Whether *messages* still begins with the ids we saw before this invocation.

    The guard that makes slicing safe. Ids are present and stable across the checkpointer's
    serialize round trip (object identity is not), so a real thread slices. Anything that
    REWROTE the history — a future summarizing middleware — changes them, and we fall back to
    today's whole-thread behaviour rather than risk cutting away this turn's own records.

    It also keeps the existing peer test doubles working: they fake a non-cumulative thread with
    hand-built messages whose ``id`` is None, so the guard fails and nothing is sliced. That is
    worth stating plainly — those doubles therefore give this mechanism ZERO coverage, which is
    why it is tested against a real create_agent graph instead.
    """
    if not prev_ids:
        return False
    if len(messages) < len(prev_ids):
        return False
    for msg, prev in zip(messages[:len(prev_ids)], prev_ids):
        got = getattr(msg, "id", None)
        if got is None or got != prev:
            return False
    return True


@dataclass
class PeerRun:
    """One invocation's own response, artifacts and answer."""

    resp: Any
    artifacts: Dict[str, List[Dict[str, Any]]]
    answer: str


class PeerSession:
    """One peer thread, for the duration of one turn.

    Colocates the invoke with the extraction so the two cannot drift: every call returns only
    what THAT invocation produced, while ``turn_artifacts`` accumulates the turn.
    """

    def __init__(self, executor: Any, config: Optional[Dict[str, Any]]) -> None:
        self._executor = executor
        self._config = config
        existing = peer_thread_messages(executor, config)
        self._seen = len(existing)
        self._ids = [getattr(m, "id", None) for m in existing]
        self.turn_artifacts: Dict[str, List[Dict[str, Any]]] = {"tool_calls": [], "tool_results": []}

    def run(self, query: str, *, chat_history: Optional[List[Any]] = None) -> PeerRun:
        from agent_runtime.runtime_utils import extract_final_answer, extract_search_artifacts

        resp = invoke_agent_with_payload_fallback(
            self._executor, query=query, chat_history=chat_history, config=self._config)
        messages = (resp or {}).get("messages") if isinstance(resp, dict) else None
        scoped: Any = resp
        if isinstance(messages, list) and self._seen and _prefix_is_intact(messages, self._ids):
            scoped = {**resp, "messages": messages[self._seen:]}
        if isinstance(messages, list):
            self._seen = len(messages)
            self._ids = [getattr(m, "id", None) for m in messages]
        artifacts = extract_search_artifacts(scoped)
        for key in ("tool_calls", "tool_results"):
            self.turn_artifacts[key].extend(artifacts.get(key) or [])
        return PeerRun(resp=resp, artifacts=artifacts, answer=extract_final_answer(scoped) or "")


def open_peer_session(executor: Any, config: Optional[Dict[str, Any]]) -> PeerSession:
    """Take the watermark BEFORE the first invocation of this turn."""
    return PeerSession(executor, config)


def is_empty_model_response_error(exc: Exception) -> bool:
    """Return True when *exc* looks like an empty model response."""
    text = str(exc).lower()
    return (
        "nonetype" in text and "model_dump" in text
    ) or ("none" in text and "chat result" in text)


# ---------------------------------------------------------------------------
# Agent executor builders
# ---------------------------------------------------------------------------

def _make_history_repair_middleware() -> Any:
    """Build a wrap_model_call middleware that repairs invalid tool-call history.

    A dangling assistant ``tool_calls`` message (one whose tool response never
    got written, e.g. after an interrupted run) otherwise poisons every later
    turn on the thread.  This sanitizes the outgoing message list before each
    model call so such a thread self-heals (see runtime_utils.repair_tool_call_sequence).
    """
    from langchain.agents.middleware import wrap_model_call
    from agent_runtime.runtime_utils import repair_tool_call_sequence

    @wrap_model_call
    def repair_history(request: Any, handler: Any) -> Any:
        fixed, changed = repair_tool_call_sequence(request.messages)
        if changed:
            request = request.override(messages=fixed)
        return handler(request)

    return repair_history


# ---------------------------------------------------------------------------
# Context budget
# ---------------------------------------------------------------------------
# Nothing measured the outgoing payload before this. A peer thread accumulates every tool
# result for the life of a conversation — BoundedInMemorySaver caps the number of THREADS, not
# the messages inside one — and the deployed default (gpt-oss:120b, 65,536) has no fallback
# configured, so the overflow surfaces as a raw provider 400 in the user's chat:
#   "Input length (66,275) exceeds model's maximum context length (65,536)"
# Measured on one clay turn: 199,605. The trim is a floor under that, not a strategy.
#
# Budget in TOKENS, approximated. The model-based counter is not an option: ChatOpenAI's
# get_num_tokens_from_messages raises NotImplementedError for the AnvilGPT model ids this
# deployment defaults to, so asking for exact counts breaks the default provider outright.
CONTEXT_BUDGET_ENV = "AGENT_CONTEXT_BUDGET_TOKENS"
DEFAULT_CONTEXT_BUDGET = 48_000

# Context windows, by model-id prefix. CONFIGURED, not detected: no provider here reports its
# window, and the only evidence in this repo for the deployed default is a quoted provider error
# ("maximum context length (65,536)"). So the fallback is that observed value — the conservative
# choice, since guessing high produces the 400 this whole mechanism exists to prevent — and a
# deployment that knows better sets AGENT_MODEL_CONTEXT_WINDOW.
#
# A flat budget was wrong in BOTH directions: 48,000 throws away context that fits comfortably
# in gpt-4o's 128k, and on gpt-oss:120b's 65,536 it leaves headroom that the tool schemas and
# system prompt then eat without anything counting them.
# Every number here was MEASURED, not assumed: no provider in this stack reports a context
# window (OpenAI's /v1/models returns only id/created/owned_by/shutdown_date), so each was
# obtained by sending a deliberately oversized request and reading the provider's rejection,
# which names the limit. Measured 2026-09-02 across all 26 chat models the deployed key can
# see. Guessing high produces the 400 this machinery exists to prevent.
#
# ORDER IS LOAD-BEARING: matching is by prefix, first hit wins, so a specific id MUST precede
# the family it belongs to. The real trap here is gpt-5.4-mini (272,000), which is a prefix
# extension of gpt-5.4 (922,000) — reversed, every mini call would be handed 3.4x its real
# window. The family sizes do NOT follow version order, so no broad "gpt-5" rule is safe:
# gpt-5.4 is 922,000 while gpt-5.2 and gpt-5.4-mini are 272,000.
_MODEL_WINDOWS = (
    # --- measured: the 922,000 tier (includes the deployed default, gpt-5.6-luna) ---
    ("gpt-5.6-", 922_000),        # luna, sol, terra all measured at 922,000
    ("gpt-5.5", 922_000),
    # --- measured: the 272,000 tier. gpt-5.4-mini MUST stay ahead of gpt-5.4 ---
    ("gpt-5.4-mini", 272_000),
    ("gpt-5.4", 922_000),
    ("gpt-5.2", 272_000),
    ("gpt-5-mini", 272_000),
    ("gpt-5-nano", 272_000),
    # Last of the family on purpose: an unlisted future gpt-5.x lands on the family's
    # CONSERVATIVE size rather than its largest, so a new model wastes context instead of 400ing.
    ("gpt-5", 272_000),
    # --- measured: earlier families. gpt-4.1 was assumed 128,000 and is really 1,047,576 ---
    ("gpt-4.1", 1_047_576),       # mini and nano measured identical
    ("gpt-4o", 128_000),          # the one prior assumption that was correct; mini identical
    ("o4-mini", 200_000),         # was assumed 128,000
    ("o3", 200_000),              # covers o3-mini
    # --- quoted, not measured here ---
    ("gpt-oss", 65_536),          # from AnvilGPT's own error: "exceeds ... (65,536)"
    ("claude-", 200_000),         # not reachable with this key; unverified
)
# Deliberately the SMALLEST window we have seen, because being wrong low only wastes context
# while being wrong high is a user-visible 400. It is a floor, not an estimate — which is
# exactly why a model landing here is worth a warning.
_DEFAULT_WINDOW = 65_536
# Model ids already warned about, so an unlisted model costs one line per process, not one
# per call.
_UNKNOWN_WINDOW_WARNED: set = set()
# Output has to fit in the same window on most providers, and the AnvilGPT path sets no
# max_tokens at all, so reserve room rather than discovering it as a truncated answer.
_OUTPUT_RESERVE_ENV = "AGENT_CONTEXT_OUTPUT_RESERVE"
_DEFAULT_OUTPUT_RESERVE = 4_096
# count_tokens_approximately is chars/4. Measured against o200k_base, that UNDERCOUNTS the
# payloads this agent actually moves: 1.72x for GeoJSON boundaries, 2.25x for per-zone embedding
# vectors (it overcounts prose slightly). A ceiling computed from a real window using an
# undercounting estimate is still unsafe, so treat the estimate as optimistic by this factor.
_PAYLOAD_SAFETY_ENV = "AGENT_CONTEXT_PAYLOAD_SAFETY"
_DEFAULT_PAYLOAD_SAFETY = 1.8


def _int_env(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return max(0, int(float(raw)))
    except (TypeError, ValueError):
        return default


def _model_context_window(model: Any = None) -> int:
    """This model's window: explicit env, else the prefix table, else the observed default."""
    override = _int_env("AGENT_MODEL_CONTEXT_WINDOW", 0)
    if override:
        return override
    name = ""
    for attr in ("model_name", "model", "model_id"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            name = value.lower()
            break
    for prefix, window in _MODEL_WINDOWS:
        if name.startswith(prefix):
            return window
    # Falling back is safe but lossy, and it was SILENT: the deployed default (gpt-5.6-luna)
    # sat on a 65,536 guess against a real 922,000 limit until instrumentation exposed it.
    # The model is chosen per REQUEST by the client, so an unlisted id is normal operation,
    # not a misconfiguration — say so once instead of hiding it.
    if name and name not in _UNKNOWN_WINDOW_WARNED:
        _UNKNOWN_WINDOW_WARNED.add(name)
        logger.warning(
            "context window for %r is not configured; assuming the %d floor. If this model's "
            "window is larger, every call needlessly trims. Add it to _MODEL_WINDOWS or set "
            "AGENT_MODEL_CONTEXT_WINDOW.", name, _DEFAULT_WINDOW)
    return _DEFAULT_WINDOW


def _context_budget() -> int:
    """Explicit override for the message ceiling, or 0 when none is set.

    0 no longer means "disabled": it means "derive it from the model and the bound tools", which
    is what _derive_budget does per request. An explicit value still wins, and a NEGATIVE one
    disables trimming for a deployment that wants the raw behaviour back.
    """
    raw = (os.getenv(CONTEXT_BUDGET_ENV) or "").strip()
    if not raw:
        return 0
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return 0


# Rendering every bound tool schema to JSON is not free: the analyze peer binds ~30 tools and
# the unified peer ~86, and the budgeter needs the number on EVERY model call. Cache it by the
# tool-name fingerprint — a peer's bound set is stable across a turn, so this renders once per
# distinct set per process instead of once per call.
_SCHEMA_COST_CACHE: Dict[str, Tuple[int, Dict[str, int]]] = {}


def _tool_names(tools: List[Any]) -> List[str]:
    names: List[str] = []
    for t in tools:
        name = getattr(t, "name", None)
        if not isinstance(name, str) and isinstance(t, dict):
            name = (t.get("function") or {}).get("name") or t.get("name")
        names.append(str(name or "?"))
    return names


def _toolset_fingerprint(names: List[str]) -> str:
    import hashlib

    return hashlib.sha1("\n".join(sorted(names)).encode()).hexdigest()[:12]


def _schema_cost(tools: List[Any]) -> Tuple[int, Dict[str, int]]:
    """(total tokens for all tool schemas, per-tool tokens). Cached by fingerprint.

    The per-tool breakdown is what makes a decision about pruning the bound set possible: the
    total says the schemas are expensive, only the breakdown says which ones are worth cutting.
    """
    from langchain_core.messages.utils import count_tokens_approximately

    if not tools:
        return 0, {}
    names = _tool_names(tools)
    key = _toolset_fingerprint(names)
    hit = _SCHEMA_COST_CACHE.get(key)
    if hit is not None:
        return hit
    per: Dict[str, int] = {}
    total = 0
    try:
        from langchain_core.utils.function_calling import convert_to_openai_tool

        for name, tool in zip(names, tools):
            try:
                rendered = json.dumps(convert_to_openai_tool(tool), default=str)
            except Exception:  # noqa: BLE001 - a tool that will not render is not worth a crash
                continue
            cost = count_tokens_approximately([("user", rendered)])
            per[name] = cost
            total += cost
    except Exception:  # noqa: BLE001
        return 0, {}
    _SCHEMA_COST_CACHE[key] = (total, per)
    return total, per


def _overhead_tokens(request: Any) -> int:
    """The parts of the payload the old accounting could not see.

    ModelRequest.messages is annotated "excluding system message", and `tools` is a sibling
    field — both are added downstream in langchain's model node. So the previous count omitted
    the system prompt AND every tool schema, which for the analyze peer with an upload is the
    larger share of the request.
    """
    from langchain_core.messages.utils import count_tokens_approximately

    total = 0
    system = getattr(request, "system_message", None)
    if system is not None:
        try:
            total += count_tokens_approximately([system])
        except Exception:      # noqa: BLE001
            pass
    total += _schema_cost(list(getattr(request, "tools", None) or []))[0]
    return total


def _derive_budget(request: Any) -> int:
    """The message ceiling for THIS request: window - output - overhead, or the explicit env."""
    explicit = _context_budget()
    if explicit:
        return max(0, explicit)
    window = _model_context_window(getattr(request, "model", None))
    reserve = _int_env(_OUTPUT_RESERVE_ENV, _DEFAULT_OUTPUT_RESERVE)
    ceiling = window - reserve - _overhead_tokens(request)
    # Never trim to nothing: a floor keeps a coherent question+answer pair possible even when
    # the bound schemas are enormous, and makes the failure a visible warning rather than an
    # empty call.
    if ceiling < 4_000:
        logger.warning(
            "context budget floor hit: window %d leaves only %d for messages after output "
            "reserve and %d tokens of system prompt + tool schemas",
            window, ceiling, _overhead_tokens(request))
        return 4_000
    return ceiling


def _make_context_budget_middleware() -> Any:
    """Keep the outgoing message list under a token budget.

    Non-destructive, like the history repairer beside it: it rewrites the OUTGOING request via
    request.override(), never the checkpointed state, so a trimmed call cannot corrupt the
    thread it was trimmed from.

    What it keeps, in priority order: every system message, the last human message, and then
    as much of the recent tail as fits. Dropping from the OLDEST end is deliberate — a
    follow-up is nearly always about recent work, and the turn ledger
    (supervisor/graph.py) re-injects the older facts in a form that costs ~130 tokens
    instead of the thousands the raw tool results cost.
    """
    from langchain.agents.middleware import wrap_model_call
    from langchain_core.messages import HumanMessage
    from langchain_core.messages.utils import count_tokens_approximately

    safety = float(os.getenv(_PAYLOAD_SAFETY_ENV) or _DEFAULT_PAYLOAD_SAFETY) or 1.0

    def _cost(msgs: Any) -> int:
        """The estimate, marked up for how badly it undercounts real payloads."""
        return int(count_tokens_approximately(msgs) * safety)

    @wrap_model_call
    def budget_context(request: Any, handler: Any) -> Any:
        if _context_budget() < 0:
            return handler(request)                 # explicitly disabled
        # Per REQUEST, not per process: the ceiling depends on the model's window and on how
        # many tool schemas THIS peer bound, and both vary. A flat 48,000 threw away context
        # that fits in gpt-4o's 128k while leaving gpt-oss:120b's 65,536 to be overrun by the
        # schemas nothing was counting.
        budget = _derive_budget(request)
        if budget <= 0:
            return handler(request)
        messages = list(getattr(request, "messages", None) or [])
        if not messages:
            return handler(request)
        try:
            total = _cost(messages)
        except Exception:  # noqa: BLE001 - counting must never break a call
            return handler(request)
        if total <= budget:
            return handler(request)

        # request.messages EXCLUDES the system message (langchain prepends it downstream), so
        # this filter found nothing and the "pin every system message" docstring described a
        # no-op. Pin the last HUMAN message — the question, whose loss makes the call pointless —
        # and keep the original order when reassembling, since in a ReAct loop the last human
        # message is NOT last: tool-call pairs follow it, and reordering splits them.
        anchor = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        if anchor is None:
            anchor = messages[-1]
        pinned_ids = {id(anchor)}
        body = [m for m in messages if id(m) not in pinned_ids]

        try:
            used = _cost([anchor])
        except Exception:  # noqa: BLE001
            return handler(request)
        kept_ids = set(pinned_ids)
        for msg in reversed(body):
            try:
                size = _cost([msg])
            except Exception:  # noqa: BLE001
                break
            if used + size > budget:
                break
            kept_ids.add(id(msg))
            used += size

        trimmed = [m for m in messages if id(m) in kept_ids]
        # A tool result whose call was dropped is a 400 on most providers; the repair
        # middleware beside this one exists for exactly that, so hand it a coherent list.
        try:
            from agent_runtime.runtime_utils import repair_tool_call_sequence

            trimmed, _ = repair_tool_call_sequence(trimmed)
        except Exception:  # noqa: BLE001
            pass
        if not trimmed:
            # Repair is purely subtractive, so it can cascade the anchor away as an orphan and
            # leave NOTHING — previously shipped as a question-less call while the log claimed a
            # number well over budget.
            trimmed = [anchor]
            logger.warning("context budget: repair emptied the payload; sending the question only")
        try:
            after = _cost(trimmed)                  # the truth is the POST-repair count
        except Exception:  # noqa: BLE001
            after = used
        logger.info(
            "context budget: trimmed %d message(s), ~%d -> ~%d tokens (ceiling %d = window %d "
            "- output reserve - %d overhead; x%.1f payload safety)",
            len(messages) - len(trimmed), total, after, budget,
            _model_context_window(getattr(request, "model", None)),
            _overhead_tokens(request), safety,
        )
        return handler(request.override(messages=trimmed))

    return budget_context


# --- per-turn instrumentation --------------------------------------------------------------
#
# Nothing measured a normal turn before. `_message_diagnostics` counts tool calls per name but
# is only reached on a GraphRecursionError, and TraceStore is in-memory and capped at 100 — so
# every claim about whether tool BREADTH costs selection accuracy, and every claim about how
# badly the token estimator undercounts, rested on offline arithmetic instead of production
# data. This closes that gap with one structured line per model call.
#
# The measurement obtainable no other way is `usage_metadata.input_tokens` — the provider's OWN
# count of what it received. Against our estimate it yields the real undercount ratio, for real
# payloads, per model, replacing the 1.72-2.25x measured offline on synthetic GeoJSON.
#
# Deliberately NOT a failure label. Sorting "absent" from "present-not-chosen" is a judgement
# about intent that this layer cannot make without guessing. What it records instead are the
# facts needed to assign one later: which tools were BOUND (the catalog line) and which were
# CALLED (each call line).
INSTRUMENTATION_ENV = "AGENT_TURN_INSTRUMENTATION"

# Fingerprints whose per-tool schema costs have already been logged this process. The catalog
# is emitted once per distinct bound set rather than on every call: 86 tool names and costs is
# ~2 KB, which is worth logging once and pure noise logged per request.
_TOOLSET_LOGGED: set = set()

# Per (conversation, peer) call counter. The thread id identifies a CONVERSATION, not a turn,
# so without an ordinal the lines of a multi-turn conversation cannot even be sequenced.
_CALL_SEQ: Dict[Any, int] = {}


def _instrumentation_enabled() -> bool:
    """On by default — the cost is one log line per model call, the alternative is guessing."""
    return (os.getenv(INSTRUMENTATION_ENV) or "").strip().lower() not in {"0", "false", "no", "off"}


def _active_thread_id() -> Optional[str]:
    """The thread id for the call in flight, read from langchain's active-config contextvar.

    ModelRequest carries `state` and `runtime` but no config, and Runtime exposes only
    (context, store, stream_writer, previous) — so this contextvar is the only route to the
    thread id from inside a wrap_model_call handler. Verified to hold `{thread}::{label}` for
    peer threads, which is what makes per-turn grouping and peer attribution possible.
    """
    try:
        from langchain_core.runnables.config import var_child_runnable_config

        cfg = var_child_runnable_config.get()
        tid = ((cfg or {}).get("configurable") or {}).get("thread_id")
        return str(tid) if tid else None
    except Exception:  # noqa: BLE001
        return None


def _usage_and_calls(response: Any) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Pull provider usage and tool calls out of a ModelResponse.

    The handler returns a ModelResponse whose only field is `result` — a LIST of messages.
    Reading `response.usage_metadata` (or treating `result` as a message) yields None on every
    call, which would make this instrumentation silently report nothing forever.
    """
    usage: Optional[Dict[str, Any]] = None
    called: List[str] = []
    messages = getattr(response, "result", None)
    if not isinstance(messages, list):
        messages = [messages] if messages is not None else []
    for msg in messages:
        um = getattr(msg, "usage_metadata", None)
        if isinstance(um, dict) and usage is None:
            usage = um
        for call in getattr(msg, "tool_calls", None) or []:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name:
                called.append(str(name))
    return usage, called


def _make_instrumentation_middleware() -> Any:
    """Record what each model call actually cost and which tools it actually used.

    Placed LAST in the middleware list, which makes it INNERMOST: the first handler is the
    outermost wrapper, so only the last one sees the request as it finally goes to the provider
    — after repair and after the budget trim. Measuring any earlier would report a payload that
    was never sent.
    """
    from langchain.agents.middleware import wrap_model_call
    from langchain_core.messages.utils import count_tokens_approximately

    @wrap_model_call
    def instrument(request: Any, handler: Any) -> Any:
        if not _instrumentation_enabled():
            return handler(request)

        record: Dict[str, Any] = {}
        try:
            messages = list(getattr(request, "messages", None) or [])
            tools = list(getattr(request, "tools", None) or [])
            names = _tool_names(tools)
            fingerprint = _toolset_fingerprint(names) if names else "none"
            schema_tokens, per_tool = _schema_cost(tools)
            system = getattr(request, "system_message", None)
            system_tokens = count_tokens_approximately([system]) if system is not None else 0
            thread = _active_thread_id() or ""
            # Peer threads are checkpointed under "{thread_id}::{label}", so the suffix is the
            # peer name. The prefix is the CONVERSATION id (observed live: "sess-4fa362d4-..."),
            # NOT a turn id — every turn of a conversation shares it, so this field cannot
            # group a turn on its own. `seq` below is what orders the calls; a turn boundary
            # shows up as `messages` dropping back down.
            session, _, label = thread.partition("::")
            model = getattr(request, "model", None)
            key = (session, label)
            _CALL_SEQ[key] = _CALL_SEQ.get(key, 0) + 1
            record = {
                "session": session,
                "seq": _CALL_SEQ[key],
                "peer": label or "supervisor",
                "model": next((getattr(model, a) for a in ("model_name", "model", "model_id")
                               if isinstance(getattr(model, a, None), str)), "unknown"),
                "toolset": fingerprint,
                "tools_bound": len(tools),
                "messages": len(messages),
                "est_message_tokens": count_tokens_approximately(messages) if messages else 0,
                "schema_tokens": schema_tokens,
                # Counted separately because the provider's input_tokens includes it: a ratio
                # against messages+schemas alone silently omits ~1,000 tokens and reports an
                # overcount that is really a mis-comparison.
                "system_tokens": system_tokens,
                "ceiling": _derive_budget(request),
            }
            if fingerprint not in _TOOLSET_LOGGED and per_tool:
                _TOOLSET_LOGGED.add(fingerprint)
                logger.info("turn_instrumentation_toolset %s", json.dumps(
                    {"toolset": fingerprint, "peer": label or "supervisor",
                     "tools_bound": len(tools), "schema_tokens": schema_tokens,
                     "per_tool": per_tool}, default=str))
        except Exception:  # noqa: BLE001 - instrumentation must never break a turn
            record = {}

        try:
            response = handler(request)
        except Exception as exc:  # noqa: BLE001
            # The previous version logged only after a SUCCESSFUL call, so a provider rejection
            # — the 400 the whole context budget exists to prevent — produced no line at all.
            # That censored the sample precisely at the tail worth measuring.
            if record:
                record["error"] = type(exc).__name__
                record["error_detail"] = str(exc)[:300]
                try:
                    logger.info("turn_instrumentation %s", json.dumps(record, default=str))
                except Exception:  # noqa: BLE001
                    pass
            raise

        try:
            usage, called = _usage_and_calls(response)
            record["tools_called"] = called
            if usage:
                real_in = usage.get("input_tokens")
                record["real_input_tokens"] = real_in
                record["output_tokens"] = usage.get("output_tokens")
                # Must mirror exactly what the provider counted: messages + tool schemas +
                # the system prompt. Omitting any part makes the ratio meaningless.
                est = ((record.get("est_message_tokens") or 0)
                       + (record.get("schema_tokens") or 0)
                       + (record.get("system_tokens") or 0))
                if isinstance(real_in, int) and est > 0:
                    # >1 means the estimator UNDERCOUNTED, the direction that causes a 400.
                    record["undercount_ratio"] = round(real_in / est, 3)
            else:
                # Distinguishes "the provider reported nothing" from "we failed to read it" —
                # the third failure label the research asked to be able to assign.
                record["usage"] = "absent"
        except Exception:  # noqa: BLE001
            pass

        if record:
            try:
                logger.info("turn_instrumentation %s", json.dumps(record, default=str))
            except Exception:  # noqa: BLE001
                pass
        return response

    return instrument


def _default_middleware() -> List[Any]:
    """The middleware stack, in order. The FIRST handler is the OUTERMOST wrapper.

    Repair first, so the budgeter always sees a coherent tool-call list; budget next; and
    instrumentation LAST so it is innermost and measures the request exactly as it goes to the
    provider — after both have rewritten it. This order is load-bearing, hence a named function
    with a test on it rather than a literal inside build_agent_executor.
    """
    return [_make_history_repair_middleware(),
            _make_context_budget_middleware(),
            _make_instrumentation_middleware()]


def build_agent_executor(
    *,
    llm: Optional[Any] = None,
    verbose: bool = False,
    return_intermediate_steps: bool = True,
    tool_strategy: str = "granular",
    include_mcp_tools: bool = False,
    mcp_modules: Optional[List[str]] = None,
    enabled_search_methods: Optional[List[str]] = None,
    allowed_tool_names: Optional[List[str]] = None,
    preloaded_tools: Optional[List[Any]] = None,
    system_prompt_override: Optional[str] = None,
    agent_name: str = "rag_agent",
    checkpointer: Optional[Any] = DEFAULT_CHECKPOINTER,
    session_id: Optional[str] = None,
    skill_roots: Optional[List[str]] = None,
) -> Any:
    """Create a LangGraph agent (``create_agent``) wired with the given tools."""
    if preloaded_tools is not None:
        # `skill_roots` was accepted and then forwarded ONLY into _collect_tools, which this
        # branch skips — so a caller that passed both (the analyze peer does) silently got no
        # list_available_skills / load_skill at all. Fixed at the boundary rather than the call
        # sites: there are three callers and the next one would forget too. The dedup makes it a
        # no-op for the code peer, which already calls make_skill_tools itself.
        tools = list(preloaded_tools)
        try:
            from agent_runtime.skills import make_skill_tools

            have = {str(getattr(t, "name", "")) for t in tools}
            tools += [t for t in make_skill_tools(skill_roots=skill_roots)
                      if str(getattr(t, "name", "")) not in have]
        except Exception:      # skills are optional; never fail executor construction over them
            pass
    else:
        tools = _collect_tools(
            tool_strategy=tool_strategy,
            include_mcp_tools=include_mcp_tools,
            mcp_modules=mcp_modules,
            enabled_search_methods=enabled_search_methods,
            session_id=session_id,
            skill_roots=skill_roots,
        )
    if allowed_tool_names:
        allowed = set(allowed_tool_names)
        filtered = [tool for tool in tools if getattr(tool, "name", "") in allowed]
        if filtered:
            tools = filtered
    active_llm = llm or build_default_llm()

    system_prompt = system_prompt_override or DEFAULT_AGENT_PROMPT

    # LangGraph agent (langchain>=1.0). ``create_agent`` returns a compiled
    # StateGraph whose invocation accepts ``{"messages": [...]}`` and returns the
    # same shape — which ``runtime_utils`` already parses. Recursion is bounded by
    # LangGraph's ``recursion_limit`` (set per invocation via ``agent_config``);
    # a ``GraphRecursionError`` surfaces as ``AgentInvocationError`` with
    # diagnostics (see ``invoke_agent_with_payload_fallback``).
    try:
        from langchain.agents import create_agent
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Missing compatible LangChain dependencies. Install `langchain>=1.0`, "
            "`langchain-core`, `langgraph`, and `langchain-openai`."
        ) from exc
    return create_agent(
        model=active_llm,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        middleware=_default_middleware(),
        debug=verbose,
        name=agent_name,
    )


# The MCP tool modules the search peer binds when a request names none. This used to be None,
# which make_langchain_mcp_tools reads as every tool the MCP server registers, less the two it
# unbinds by default: 14 of 16 tools, 2,248 of the peer's 7,183 schema tokens (o200k) and about
# 1,584 real input tokens on every search model call, for crime-data loaders and analysis, image
# description, the notebook builder, its executors and a repository ingester. The deployed
# journal holds 166 search calls (09-22 to 10-02) and none called one, and the decider's search
# line ("retrieve evidence") never mentioned them. What SEARCH_AGENT_PROMPT does name is rule 8's
# `mcp_fetch_element_source`, a cited element's source file by id, which is the only tool in
# element_tools. The analyze peer is scoped the same way, to spatial_analysis_tools, and a
# request's own mcpModules still replaces this. See docs/agent-architecture-changes.md.
SEARCH_MCP_MODULES: Tuple[str, ...] = ("element_tools",)


def build_search_agent_executor(
    *,
    llm: Optional[Any] = None,
    verbose: bool = False,
    return_intermediate_steps: bool = True,
    tool_strategy: str = "granular",
    include_mcp_tools: bool = False,
    mcp_modules: Optional[List[str]] = None,
    enabled_search_methods: Optional[List[str]] = None,
    allowed_tool_names: Optional[List[str]] = None,
    preloaded_tools: Optional[List[Any]] = None,
    checkpointer: Optional[Any] = DEFAULT_CHECKPOINTER,
    session_id: Optional[str] = None,
    skill_roots: Optional[List[str]] = None,
) -> Any:
    return build_agent_executor(
        llm=llm,
        verbose=verbose,
        return_intermediate_steps=return_intermediate_steps,
        tool_strategy=tool_strategy,
        include_mcp_tools=include_mcp_tools,
        mcp_modules=mcp_modules or list(SEARCH_MCP_MODULES),
        enabled_search_methods=enabled_search_methods,
        allowed_tool_names=allowed_tool_names,
        preloaded_tools=preloaded_tools,
        system_prompt_override=SEARCH_AGENT_PROMPT,
        agent_name="search_agent",
        checkpointer=checkpointer,
        session_id=session_id,
        skill_roots=skill_roots,
    )


def build_code_agent_executor(
    *,
    llm: Optional[Any] = None,
    verbose: bool = False,
    return_intermediate_steps: bool = True,
    tools: Optional[List[Any]] = None,
    checkpointer: Optional[Any] = DEFAULT_CHECKPOINTER,
    skill_roots: Optional[List[str]] = None,
) -> Any:
    return build_agent_executor(
        llm=llm,
        verbose=verbose,
        return_intermediate_steps=return_intermediate_steps,
        tool_strategy="granular",
        include_mcp_tools=False,
        mcp_modules=None,
        preloaded_tools=tools or [],
        system_prompt_override=CODE_AGENT_PROMPT,
        agent_name="code_agent",
        checkpointer=checkpointer,
        skill_roots=skill_roots,
    )
