"""Which I-GUIDE tier this agent talks to, as one switch instead of several hosts.

The platform comes in sets — a frontend, the backend that mints its cookies, and the OpenSearch
the backend stores state in — and the agent needs four things from them: where the browser
refreshes, where an unsigned-in visitor goes, where (in introspect mode) the agent asks who a
caller is, and which search cluster holds its own conversations. Setting those one at a time
invites the state this is meant to prevent: two pointing at dev, one left on prod, and a
verification that fails for a reason nobody can see.

The fourth is here because of exactly that. Dev's OpenSearch moved to a new host while
``OPENSEARCH_NODE`` stayed pinned to the old one, so the agent kept reading and writing a
cluster nobody maintained any more — still answering, still accepting writes, and by then out
of disk and unable to allocate a shard for a new index. Nothing failed loudly; it simply went
on talking to yesterday's machine. A tier that names the cluster makes that a single fact to
change instead of a variable somebody forgets.

    PLATFORM_TIER=dev | prod

Each explicit ``PLATFORM_*_URL`` still wins when set, because a tier table cannot anticipate a
staging host somebody stands up next month. The tier only supplies defaults.

The pairings below were verified live rather than assumed: each backend answers
``/api/refresh-token`` with 401 (its "no refresh cookie" reply), and each frontend answers
``/auth/login`` with a 302 to CILogon. They are public hostnames, not configuration secrets.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)

DEV = "dev"
PROD = "prod"

_TIERS: Dict[str, Dict[str, str]] = {
    DEV: {"backend": "https://backend-dev.i-guide.io", "frontend": "https://dev.i-guide.io",
          # Dev's cluster, moved here from 149.165.155.195 on 2026-09-18.
          "opensearch": "https://149.165.155.135:9200"},
    # No OpenSearch for prod until someone confirms which host it is. An empty string means
    # "this tier does not supply one", so a prod deployment keeps needing an explicit
    # OPENSEARCH_NODE rather than silently inheriting dev's cluster — which is the single worst
    # thing this table could do.
    PROD: {"backend": "https://backend.i-guide.io", "frontend": "https://platform.i-guide.io",
           "opensearch": ""},
}

_REFRESH_PATH = "/api/refresh-token"
_CHECK_PATH = "/api/check-tokens"
_SIGNIN_PATH = "/auth/login"


def current_tier() -> Optional[str]:
    """The configured tier, or None when the deployment names its URLs individually.

    An unrecognised value RAISES rather than falling back: picking the wrong platform silently
    is how a token gets verified against a backend that never minted it, and the resulting
    "invalid token" tells nobody what actually went wrong.
    """
    raw = str(os.getenv("PLATFORM_TIER") or "").strip().lower()
    if not raw:
        return None
    if raw not in _TIERS:
        raise ValueError(
            f"PLATFORM_TIER={raw!r} is not a tier. Expected one of: {', '.join(sorted(_TIERS))}.")
    return raw


def _from_tier(part: str, path: str) -> str:
    tier = current_tier()
    if not tier:
        return ""
    base = _TIERS[tier].get(part) or ""
    return f"{base}{path}" if base else ""


def _resolve(explicit_var: str, part: str, path: str) -> str:
    explicit = str(os.getenv(explicit_var) or "").strip()
    return explicit or _from_tier(part, path)


def refresh_url() -> str:
    """Where the BROWSER refreshes an aged-out access cookie. The agent never calls it."""
    return _resolve("PLATFORM_REFRESH_URL", "backend", _REFRESH_PATH)


def signin_url() -> str:
    """Where an unsigned-in visitor is sent."""
    return _resolve("PLATFORM_SIGNIN_URL", "frontend", _SIGNIN_PATH)


def check_tokens_url() -> str:
    """Where the AGENT asks the platform who a caller is, in introspect mode."""
    return _resolve("PLATFORM_CHECK_TOKENS_URL", "backend", _CHECK_PATH)


def opensearch_url() -> str:
    """The search cluster for this tier, or "" when the tier does not name one.

    ``OPENSEARCH_NODE`` still wins, as every explicit setting here does — a deployment pointing
    at a one-off cluster must not be overruled by a table. This is the default for a deployment
    that has only said which tier it is.
    """
    explicit = str(os.getenv("OPENSEARCH_NODE") or "").strip()
    if explicit:
        return explicit
    tier = current_tier()
    return (_TIERS[tier].get("opensearch") or "") if tier else ""


def opensearch_drift_warning() -> Optional[str]:
    """Said out loud when OPENSEARCH_NODE disagrees with the tier's own cluster.

    An explicit setting is allowed to win, but silently talking to a different cluster from the
    one the tier names is how the agent spent a day reading a decommissioned host: still
    reachable, still answering, and no longer the place anything else was writing to.
    """
    explicit = str(os.getenv("OPENSEARCH_NODE") or "").strip()
    if not explicit:
        return None
    try:
        tier = current_tier()
    except ValueError:
        return None
    expected = (_TIERS[tier].get("opensearch") or "") if tier else ""
    if expected and explicit.rstrip("/") != expected.rstrip("/"):
        return (f"OPENSEARCH_NODE={explicit} but PLATFORM_TIER={tier} names {expected}. "
                "The explicit setting wins; check it is deliberate.")
    return None


def consistency_warning() -> Optional[str]:
    """The half-switched states worth saying out loud at boot.

    The cookie NAME is the platform's own setting (``JWT_ACCESS_TOKEN_NAME``) and differs
    between tiers, so it does not move with ``PLATFORM_TIER``. A deployment pointed at prod
    while still expecting the dev cookie fails on every request, and the failure looks like a
    rejected token rather than a misconfiguration. This does not guess the right name — only
    names the disagreement.
    """
    tier = current_tier()
    if not tier:
        return None
    cookie = str(os.getenv("JWT_ACCESS_TOKEN_NAME") or "").strip().strip('"')
    if not cookie:
        return None
    looks_dev = cookie.endswith("-dev")
    if tier == PROD and looks_dev:
        return (f"PLATFORM_TIER=prod but JWT_ACCESS_TOKEN_NAME={cookie!r} looks like the dev "
                "tier's cookie. Every sign-in will be rejected until they agree.")
    if tier == DEV and not looks_dev:
        return (f"PLATFORM_TIER=dev but JWT_ACCESS_TOKEN_NAME={cookie!r} does not look like the "
                "dev tier's cookie. Check which platform actually mints it.")
    return None
