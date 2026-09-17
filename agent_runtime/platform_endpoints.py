"""Which I-GUIDE tier this agent talks to, as one switch instead of four URLs.

The platform comes in pairs — a frontend and the backend that mints its cookies — and the agent
needs three endpoints from them: where the browser refreshes, where an unsigned-in visitor goes,
and (in introspect mode) where the agent asks who a caller is. Setting those one at a time
invites the state this is meant to prevent: two pointing at dev, one left on prod, and a
verification that fails for a reason nobody can see.

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
    DEV: {"backend": "https://backend-dev.i-guide.io", "frontend": "https://dev.i-guide.io"},
    PROD: {"backend": "https://backend.i-guide.io", "frontend": "https://platform.i-guide.io"},
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
    return f"{_TIERS[tier][part]}{path}" if tier else ""


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
