"""Shared plumbing for open-web search and fetch: caps, the hit envelope, and the per-turn budget.

Web access differs from every other retrieval path in this service in one way that matters: the
content is UNBOUNDED. A catalog connector returns a bounded record; a web page can be 400 KB of
navigation chrome, and a search engine will happily return ten of them. The efficient agents do
not solve that with a re-ranker after the fact — they keep the expensive operation (reading a
page) behind a cheap one (reading a snippet) and cap both in CODE rather than in the prompt.

This module holds the parts both stages need:

* ``WebHit`` — the metadata-only envelope a search returns. Deliberately has NO body field: a
  search result must not be able to smuggle page content into the model's context.
* the caps (env-tunable), so a turn cannot silently spend unbounded time or tokens on the network;
* the per-turn LEDGER, which counts searches/fetches and records every URL the run actually
  touched. That URL list is what lets the answer sanitizer distinguish a real source from a
  fabricated one — the failure mode this service has already been bitten by twice.
"""

from __future__ import annotations

import hashlib
import os
import threading
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .utils import get_logger

logger = get_logger("web_search")


# --- capability + caps ---------------------------------------------------------


def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(str(os.getenv(name, "")).strip() or default))
    except (TypeError, ValueError):
        return default


def web_enabled() -> bool:
    """Whether open-web access is available at all (kill switch: AGENT_WEB_ENABLED=false)."""
    return _env_flag("AGENT_WEB_ENABLED", True)


def max_searches_per_turn() -> int:
    """Query reformulation needs a few tries; unlimited tries is a runaway loop."""
    return _env_int("AGENT_WEB_MAX_SEARCHES_PER_TURN", 3, minimum=0)


def max_fetches_per_turn() -> int:
    """Efficient agents open 1-3 of ~10 results. This is the token-cost ceiling of the feature."""
    return _env_int("AGENT_WEB_MAX_FETCHES_PER_TURN", 3, minimum=0)


def search_snippet_chars() -> int:
    """Cap per search-result snippet.

    Intentionally ~300, NOT the 4000 of ``snippet_chars()`` used for catalog abstracts: the whole
    point of the discovery stage is that it is cheap. Snippets are a proxy for the document, read
    to decide what is worth fetching — not evidence to answer from.
    """
    return _env_int("AGENT_WEB_SNIPPET_CHARS", 300, minimum=80)


def web_timeout() -> int:
    """Per-request network timeout, matching the catalog connectors' default."""
    return _env_int("AGENT_WEB_TIMEOUT", 12, minimum=1)


def fetch_max_bytes() -> int:
    """Socket-level cap: how much of a response body is read at all.

    Enforced while streaming, so an enormous or hostile response is abandoned mid-download rather
    than buffered and then measured.
    """
    return _env_int("AGENT_WEB_MAX_BYTES", 2_000_000, minimum=10_000)


def fetch_max_chars() -> int:
    """Cap on extracted, on-topic text handed to the model — the real token-cost ceiling."""
    return _env_int("AGENT_WEB_FETCH_MAX_CHARS", 6000, minimum=500)


# --- the hit envelope ----------------------------------------------------------


@dataclass
class WebHit:
    """One search result: metadata only, by design.

    ``snippet`` is the engine's own extract, not page content we retrieved. There is no field for
    a body — fetching is a separate, budgeted step.
    """

    url: str
    title: str
    snippet: str
    provider: str
    published: Optional[str] = None
    rank: int = 0

    def doc_id(self) -> str:
        """Stable id derived from the canonical URL, so the same page dedupes across engines."""
        return "web-" + hashlib.sha1(canonical_url(self.url).encode("utf-8")).hexdigest()[:16]


# --- URL canonicalization -----------------------------------------------------

# Campaign/click trackers: present or absent, the page is the same document. Left in, they defeat
# dedupe and the URL allowlist (the model echoes a slightly different string than we recorded).
_TRACKING_PREFIXES = ("utm_", "pk_", "mtm_", "ga_")
_TRACKING_PARAMS = {
    "gclid", "fbclid", "msclkid", "dclid", "yclid", "igshid", "mc_cid", "mc_eid",
    "ref", "ref_src", "referrer", "source", "spm", "cmpid", "campaign_id", "_hsenc", "_hsmi",
}


def canonical_url(url: Any) -> str:
    """Normalized form of *url* for dedupe, caching, and allowlist comparison.

    Lowercases scheme/host, drops the fragment (never server-visible), strips tracking params, and
    removes a redundant default port. Everything else — path case, remaining query order — is left
    alone, because for many servers those are significant.
    """
    raw = str(url or "").strip()
    if not raw:
        return ""
    try:
        parts = urlsplit(raw)
    except ValueError:
        return raw
    host = (parts.hostname or "").lower()
    if not host:
        return raw
    if parts.port and not (
        (parts.scheme == "http" and parts.port == 80) or (parts.scheme == "https" and parts.port == 443)
    ):
        host = f"{host}:{parts.port}"
    kept = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k.lower() not in _TRACKING_PARAMS and not k.lower().startswith(_TRACKING_PREFIXES)
    ]
    return urlunsplit((parts.scheme.lower(), host, parts.path or "/", urlencode(kept), ""))


# --- per-turn budget ledger ---------------------------------------------------


@dataclass
class WebBudget:
    searches: int = 0
    fetches: int = 0
    urls: List[str] = field(default_factory=list)   # every URL this turn surfaced or read
    fetched: List[str] = field(default_factory=list)  # URLs whose CONTENT we actually retrieved


# A ContextVar rather than graph state: the tools run deep inside the LangChain executor, where
# SupervisorState is not reachable. Threads start with a fresh context, so concurrent requests are
# isolated -- but a gunicorn SYNC worker reuses its thread across requests, so a turn MUST call
# begin_turn() or the previous turn's spend would carry over and starve it.
_BUDGET: ContextVar[Optional[WebBudget]] = ContextVar("agent_web_budget", default=None)


def begin_turn() -> WebBudget:
    """Reset the ledger. Call once per user turn, before any peer runs."""
    budget = WebBudget()
    _BUDGET.set(budget)
    return budget


def budget() -> WebBudget:
    """The current ledger, created on demand.

    Lazy creation is deliberate: a missing begin_turn() must fail SAFE (a fresh budget) rather
    than crash a turn or hand out unlimited network access.
    """
    current = _BUDGET.get()
    if current is None:
        current = WebBudget()
        _BUDGET.set(current)
    return current


class Refusal(str):
    """A refusal that no retry within this turn can change: a spent budget or the kill switch.

    It IS its message, so every caller that treated the old return value as a string still
    does. It also carries the reason as a code, and ``fields()`` is what a tool result puts
    beside its ``error``. The turn log reads those fields (agent_runtime/turn_log.py,
    ``refused_for``) and answers every later call of the tool from this result instead of
    running it. Live, 2026-10-16 (sess-da4493c7): "web search budget exhausted for this turn
    (3/3)" came back again and again, because each new query was a new call and nothing in
    the result said the next one would fare the same.
    """

    reason: str

    def __new__(cls, message: str, reason: str) -> "Refusal":
        obj = super().__new__(cls, message)
        obj.reason = reason
        return obj

    def fields(self) -> Dict[str, Any]:
        return {"error": str(self), "reason": self.reason, "retryable": False,
                "refused_for": "turn"}


# The check and the increment are one step. A model that asks for three pages in one message
# has them run on parallel threads, and two threads that both read fetches=2 would both take
# the last fetch.
_CHARGE_LOCK = threading.Lock()


def charge_search() -> Optional[Refusal]:
    """Claim one search from the budget; returns the refusal when exhausted."""
    if not web_enabled():
        return Refusal("web access is disabled on this deployment (AGENT_WEB_ENABLED=false)",
                       "web_disabled")
    current = budget()
    limit = max_searches_per_turn()
    with _CHARGE_LOCK:
        if current.searches >= limit:
            return Refusal(f"web search budget exhausted for this turn ({current.searches}/{limit})",
                           "budget_exhausted")
        current.searches += 1
    return None


def charge_fetch() -> Optional[Refusal]:
    """Claim one page fetch from the budget; returns the refusal when exhausted."""
    if not web_enabled():
        return Refusal("web access is disabled on this deployment (AGENT_WEB_ENABLED=false)",
                       "web_disabled")
    current = budget()
    limit = max_fetches_per_turn()
    with _CHARGE_LOCK:
        if current.fetches >= limit:
            return Refusal(f"web fetch budget exhausted for this turn ({current.fetches}/{limit})",
                           "budget_exhausted")
        current.fetches += 1
    return None


def read_this_turn(url: Any) -> bool:
    """Whether this turn has already read the page at *url* (its canonical form)."""
    canonical = canonical_url(str(url or ""))
    return bool(canonical) and canonical in budget().fetched


def record_urls(urls: Any, *, fetched: bool = False) -> None:
    """Record URLs this turn legitimately surfaced (search) or read (fetch).

    Feeds the answer sanitizer's allowlist: a link may appear in an answer only if the run
    actually produced it, so an invented-but-plausible URL is stripped instead of shown.
    """
    current = budget()
    for url in urls if isinstance(urls, (list, tuple, set)) else [urls]:
        raw = str(url or "").strip()
        canonical = canonical_url(raw)
        if not canonical:
            continue
        # BOTH forms are citable. The model echoes the URL as it appeared in the tool result (raw),
        # while dedupe/caching key on the canonical form; the sanitizer compares exact strings, so
        # recording only one form would reject a perfectly legitimate citation.
        for form in (raw, canonical):
            if form and form not in current.urls:
                current.urls.append(form)
        if fetched and canonical not in current.fetched:
            current.fetched.append(canonical)


def allowed_urls() -> List[str]:
    """Every web URL this turn is permitted to cite (canonical form)."""
    return list(budget().urls)


def budget_snapshot() -> Dict[str, Any]:
    """Ledger state for traces/observability."""
    current = budget()
    return {
        "searches": current.searches,
        "searches_max": max_searches_per_turn(),
        "fetches": current.fetches,
        "fetches_max": max_fetches_per_turn(),
        "urls": len(current.urls),
        "fetched": len(current.fetched),
    }


__all__ = [
    "WebBudget",
    "WebHit",
    "Refusal",
    "allowed_urls",
    "begin_turn",
    "budget",
    "budget_snapshot",
    "canonical_url",
    "charge_fetch",
    "charge_search",
    "fetch_max_bytes",
    "fetch_max_chars",
    "max_fetches_per_turn",
    "max_searches_per_turn",
    "read_this_turn",
    "record_urls",
    "search_snippet_chars",
    "web_enabled",
    "web_timeout",
]
