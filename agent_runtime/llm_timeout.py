"""A request timeout and a retry cap for every model call the agent makes (Stage 85).

Every chat model used to be built with NO request timeout, and with the client's default of two
automatic retries. A stalled model was indistinguishable from progress: the socket stayed open,
SSE keepalives flowed, nothing gave up. Seen in production:

* 2026-10-09, Lumen deepseek-v4-flash: HTTP 500 "Upstream error" after about 6 minutes on
  long-context calls, retried twice (6 + 6 + 6 min), stalling three turns 18+ minutes each.
* 2026-10-01, AnvilGPT qwen3.8:27b: its nginx gateway cut the call at 300 s (504), then the
  client retried twice more.
* A benchmark run wedged for 2 h 23 m in one model call.

So every client now gets ``timeout`` and ``max_retries`` from here, and a timeout reaches the turn
as :class:`LLMCallTimeout`, whose message says what happened ("model call timed out after 240 s")
instead of the client's bare "Request timed out.".

Settings:

``AGENT_LLM_TIMEOUT_SECONDS``
    Seconds a single model request may go without an answer. Default 240 for Lumen and OpenAI;
    280 for AnvilGPT, under its 300 s gateway so the error is ours and says so, rather than a
    504 page. Measured on the deployment (2026-10-19, 295 Lumen deepseek-v4-flash calls): the
    longest gap between two completions inside one turn was 44 s, p99 33 s, so 240 s is over
    five times the longest real call. ``0`` (or ``off``) sends no timeout, the old behaviour.
    It is httpx's timeout: for a streamed reply it bounds the silence between chunks, so a
    model that is still sending is never cut off; a call that sends nothing ends at it.
``AGENT_LLM_MAX_RETRIES``
    Automatic retries after a failed request (timeout, 5xx, 429, connection error). Default 1,
    so a stall costs at most two timeouts, not three. 0..5.

Deliberately NOT ``max_tokens``: a ceiling cuts a reasoning model's thinking short and returns
an empty answer (measured; AGENTS.md "Do not set max_tokens for a reasoning model").
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional
from urllib.parse import urlparse

DEFAULT_TIMEOUT_S = 240.0
# AnvilGPT's nginx answers 504 at 300 s whatever the client asks for.
ANVILGPT_TIMEOUT_S = 280.0
DEFAULT_MAX_RETRIES = 1
MAX_RETRIES_CAP = 5

_OFF = {"0", "off", "none", "false", "no"}


def timeout_seconds(provider: Optional[str] = None) -> Optional[float]:
    """The request timeout for ``provider``, or None when it is switched off."""
    raw = (os.getenv("AGENT_LLM_TIMEOUT_SECONDS") or "").strip().lower()
    default = ANVILGPT_TIMEOUT_S if (provider or "").strip().lower() == "anvilgpt" \
        else DEFAULT_TIMEOUT_S
    if not raw:
        return default
    if raw in _OFF:
        return None
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else None


def max_retries() -> int:
    raw = (os.getenv("AGENT_LLM_MAX_RETRIES") or "").strip()
    try:
        return max(0, min(MAX_RETRIES_CAP, int(raw))) if raw else DEFAULT_MAX_RETRIES
    except ValueError:
        return DEFAULT_MAX_RETRIES


def client_limits(provider: Optional[str] = None) -> Dict[str, Any]:
    """The keyword arguments every chat-model constructor gets: ``timeout`` and ``max_retries``.

    ``timeout=None`` is what the clients take to mean "no timeout", so it is passed as is.
    """
    return {"timeout": timeout_seconds(provider), "max_retries": max_retries()}


def _timeout_base() -> type:
    try:
        import openai
        return openai.APITimeoutError
    except Exception:  # pragma: no cover - langchain-openai depends on openai
        return TimeoutError


class LLMCallTimeout(_timeout_base()):  # type: ignore[misc]
    """A model call that got no answer within the request timeout, retries included.

    A subclass of the client's own ``APITimeoutError``, so anything that treats that as
    retryable still does. The constructor sets the attributes itself rather than calling the
    parent's: ``APIError.__init__`` has changed signature across openai releases (the Mac runs
    2.x, the deployed image 3.x), and this must not be the thing that breaks on an upgrade.
    """

    def __init__(self, message: str, *, seconds: Optional[float] = None,
                 model: Optional[str] = None, request: Any = None) -> None:
        Exception.__init__(self, message)
        self.message = message
        self.seconds = seconds
        self.model = model
        self.request = request
        self.body = None
        self.code = None
        self.param = None
        self.type = None

    def __str__(self) -> str:
        return self.message


def describe(seconds: Optional[float], model: Optional[str], base_url: Optional[str],
             retries: Optional[int]) -> str:
    host = urlparse(str(base_url)).hostname if base_url else None
    who = " ".join(x for x in (str(model or "").strip() or "the model",
                               f"({host})" if host else "") if x)
    tries = ""
    if retries:
        tries = f", after {retries} automatic retr{'y' if retries == 1 else 'ies'}"
    limit = f"{seconds:g} s" if seconds else "the client's limit"
    return (f"Model call timed out after {limit} with no answer from {who}{tries}. "
            "The model did not respond in time; retry the turn, or pick another model.")


def from_client_timeout(exc: BaseException, llm: Any) -> BaseException:
    """``exc`` (the client's timeout) restated as an :class:`LLMCallTimeout` for ``llm``."""
    if isinstance(exc, LLMCallTimeout):
        return exc
    seconds = getattr(llm, "request_timeout", None)
    if not isinstance(seconds, (int, float)):
        seconds = getattr(seconds, "read", None)   # an httpx.Timeout
    model = getattr(llm, "model_name", None) or getattr(llm, "model", None)
    base_url = getattr(llm, "openai_api_base", None)
    # Only the client's own timeout went through its retries; a stream that stalls after its
    # first chunk raises httpx's ReadTimeout from the reader, where nothing retries.
    retries = getattr(llm, "max_retries", None) if isinstance(exc, _timeout_base()) else None
    try:
        request = getattr(exc, "request", None)
    except RuntimeError:            # httpx raises, not AttributeError, when none was set
        request = None
    return LLMCallTimeout(describe(seconds if isinstance(seconds, (int, float)) else None,
                                   model, base_url, retries if isinstance(retries, int) else None),
                          seconds=seconds if isinstance(seconds, (int, float)) else None,
                          model=model, request=request)
