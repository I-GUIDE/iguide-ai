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
from urllib.parse import quote
from typing import Dict, Optional

logger = logging.getLogger(__name__)

DEV = "dev"
PROD = "prod"

_TIERS: Dict[str, Dict[str, str]] = {
    DEV: {"backend": "https://backend-dev.i-guide.io", "frontend": "https://dev.i-guide.io",
          # Dev's cluster, moved here from 149.165.155.195 on 2026-09-18.
          "opensearch": "https://149.165.155.135:9200"},
    # Prod's cluster IS known — 149.165.155.195:9200, confirmed by the maintainer 2026-09-22 —
    # and is deliberately still not written here, because filling it in would arm two traps
    # that an empty string keeps disarmed. Measured the same day:
    #
    #   1. That cluster cannot accept a write. 55.3gb of 57.9gb used (95.5%), past the 95%
    #      flood-stage watermark, so 914 indices carry `read_only_allow_delete` and an index
    #      creation times out. READS still succeed, which is what makes it quiet: search keeps
    #      working while every conversation silently fails to save. Only 3.7gb of that disk is
    #      OpenSearch — the other ~51.6gb is something else on the box, so this is not a
    #      problem the agent can fix by deleting indices.
    #   2. Working around (1) by pinning OPENSEARCH_NODE to the dev cluster does not work
    #      either. An explicit node that disagrees with the tier makes opensearch_credentials()
    #      fall back to the UNTIERED pair, and that pair returns 401 against the dev cluster —
    #      so conversations would stop saving for a second, different reason.
    #
    # An empty string means "this tier does not supply one", so PLATFORM_TIER=prod still fails
    # loudly and demands an explicit OPENSEARCH_NODE. That is the right forcing function while
    # the above holds: better a deployment that refuses to start than one that answers
    # perfectly and remembers nothing. Fill this in once 195 has disk, and fix (2) first.
    #
    # The TOKEN side is only PARTLY blocked, and the halves are easy to get backwards.
    # Identity VERIFICATION is server-to-server — the agent forwards the cookie it received to
    # the backend's own /api/check-tokens (identity.py) — so no browser and no CORS is involved
    # and it would work against prod today. What is blocked is the browser's REFRESH.
    #
    # Measured 2026-09-22, OPTIONS /api/refresh-token with `Origin: https://agent.i-guide.io`:
    #
    #     backend-dev.i-guide.io -> Access-Control-Allow-Origin: https://agent.i-guide.io
    #     backend.i-guide.io     -> Access-Control-Allow-Origin: https://platform.i-guide.io
    #
    # Prod pins one origin and it is not this one, so the browser may not refresh an aged-out
    # access cookie from here. That failure is badly shaped: sign-in succeeds, the agent works,
    # and the session dies five minutes later at the first refusal — an expiry, not an error,
    # so nothing says why. Dev needed exactly this entry added before sign-in held there.
    #
    # Prod's redirect-whitelist.json is UNVERIFIED: it is not served as a static file (both
    # frontends answer that path with the Next.js app shell), so it could not be read from
    # outside. Dev's entry is reached via PLATFORM_REDIRECT_DOMAIN_ID=006.
    #
    # Both are platform-side config, not this repository's, and prod would also need a
    # JWT_ACCESS_TOKEN_NAME without the -dev suffix (consistency_warning() says so at boot).
    # Deployment stays on PLATFORM_TIER=dev until the refresh origin is added on prod.
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
    """Where an unsigned-in visitor is sent, and where they come back to.

    The platform's ``/auth/login`` takes ``redirect-domain-id`` and ``redirect-path`` and, after
    CILogon, returns the browser there instead of to the platform's own profile page. Without
    them someone sent from the agent signs in and lands on the platform, having lost whatever
    they were doing here.

    The domain id is NOT a URL: the frontend resolves it against its own
    ``redirect-whitelist.json``, so only hosts that file names can ever be redirect targets.
    That means the id is assigned by whoever maintains that file, which is why it is
    configuration here rather than a constant — and why this stays OFF until
    ``PLATFORM_REDIRECT_DOMAIN_ID`` is set. An unrecognised id is not an error at the far end;
    the frontend logs it and falls back to the profile page, so a wrong value degrades to
    today's behaviour rather than breaking sign-in.
    """
    base = _resolve("PLATFORM_SIGNIN_URL", "frontend", _SIGNIN_PATH)
    domain_id = str(os.getenv("PLATFORM_REDIRECT_DOMAIN_ID") or "").strip()
    if not base or not domain_id:
        return base
    # The frontend decodeURIComponent()s the path and requires it to start with "/" — anything
    # else falls back to the default, so encode it and keep it absolute.
    path = str(os.getenv("PLATFORM_REDIRECT_PATH") or "/").strip() or "/"
    if not path.startswith("/"):
        path = "/" + path
    joiner = "&" if "?" in base else "?"
    return (f"{base}{joiner}redirect-domain-id={quote(domain_id, safe='')}"
            f"&redirect-path={quote(path, safe='')}")


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


def search_tier() -> Optional[str]:
    """Which tier's SEARCH indices to use — ``SEARCH_TIER``, falling back to ``PLATFORM_TIER``.

    Separate from the platform tier because they answer different questions. ``PLATFORM_TIER``
    says which platform mints and verifies the tokens, and where this agent's own conversations
    live. ``SEARCH_TIER`` says which knowledge base to search. Wanting the dev platform against
    the prod knowledge base is an ordinary thing to want, and before this it meant editing index
    names by hand and remembering to put them back.

    Unset means "same tier as the platform", which is the safe default: one switch still moves
    everything unless someone deliberately splits them. An unrecognised value RAISES, for the
    same reason ``PLATFORM_TIER`` does — silently searching the wrong corpus produces confident
    answers from the wrong data, which is worse than an error.
    """
    raw = str(os.getenv("SEARCH_TIER") or "").strip().lower()
    if not raw:
        return current_tier()
    if raw not in _TIERS:
        raise ValueError(
            f"SEARCH_TIER={raw!r} is not a tier. Expected one of: {', '.join(sorted(_TIERS))}.")
    return raw


def tiered_env(name: str, default: Optional[str] = None,
               tier: Optional[str] = None) -> Optional[str]:
    """``<NAME>_<TIER>`` when the tier names one, otherwise ``<NAME>``.

    The general form of the rule the credential already follows, so a deployment that switches
    tiers switches everything the tier owns — hosts, credentials and INDEX NAMES — from one
    variable. Dev and prod do not agree on index names (dev's knowledge base is
    ``iguide-platform-embeddings-dev``, and the bare ``OPENSEARCH_INDEX`` still says
    ``new-opensearch-index``), and a wrong index is the quietest failure of the three: the
    cluster answers, the query succeeds, and search simply returns nothing.

    Only consulted when ``PLATFORM_TIER`` is set, and an empty tiered value counts as unset, so
    a half-written ``FOO_PROD=`` cannot blank out a working ``FOO``.
    """
    if tier is None:
        try:
            tier = current_tier()
        except ValueError:
            tier = None
    if tier:
        specific = os.getenv(f"{name}_{tier.upper()}")
        if specific:
            return specific
    return os.getenv(name, default)


def search_index() -> str:
    """The knowledge-base index for the SEARCH tier. See :func:`search_tier`.

    A bad ``SEARCH_TIER`` propagates rather than degrading to the untiered index: falling back
    would search the wrong corpus and answer confidently from it, which is the failure this
    whole switch exists to avoid.
    """
    return (tiered_env("OPENSEARCH_INDEX", tier=search_tier()) or "").strip()


def search_tier_note() -> Optional[str]:
    """Said at boot when search has been pointed at a different tier from the platform.

    Deliberate and supported, but not a thing to discover from surprising results.
    """
    try:
        platform, search = current_tier(), search_tier()
    except ValueError:
        return None
    if search and platform and search != platform:
        return (f"SEARCH_TIER={search} while PLATFORM_TIER={platform}: knowledge-base searches "
                f"use the {search} indices, identity and conversations use {platform}.")
    return None


def _node_overrides_tier() -> bool:
    """True when OPENSEARCH_NODE names a cluster the tier does not."""
    explicit = str(os.getenv("OPENSEARCH_NODE") or "").strip()
    if not explicit:
        return False
    try:
        tier = current_tier()
    except ValueError:
        return True
    expected = (_TIERS[tier].get("opensearch") or "") if tier else ""
    return bool(expected) and explicit.rstrip("/") != expected.rstrip("/")


def opensearch_credentials() -> tuple:
    """``(username, password)`` for this tier's cluster. Values are never logged.

    Precedence here is the REVERSE of the URL rule above, and deliberately so. For a URL the
    tier supplies a value and ``PLATFORM_*_URL`` overrides it, because a table cannot anticipate
    a staging host. For a credential the tier supplies no value at all — secrets do not live in
    this repository — so the tiered variable is simply the more specific name, and the bare
    ``OPENSEARCH_USERNAME`` / ``OPENSEARCH_PASSWORD`` are the un-tiered fallback.

    Getting that backwards would reintroduce the bug this is for: someone sets up per-tier
    credentials, an old bare ``OPENSEARCH_PASSWORD`` is still sitting in the file, and it
    silently pins every tier to one credential — which is exactly the half-switched state that
    let the agent keep talking to a decommissioned cluster.

    Either half is enough to select the tiered pair, so a username set for one tier and a
    password left in the bare variable does not silently mix two accounts.
    """
    try:
        tier = current_tier()
    except ValueError:
        tier = None
    # The credential follows the HOST that will actually be used, not the tier in the abstract.
    # An explicit OPENSEARCH_NODE pointing somewhere other than the tier's cluster means the
    # tier is not choosing the cluster, so it must not choose the credential either — the
    # untiered pair is the one that goes with the override.
    #
    # Found by testing what a restart would do: OPENSEARCH_NODE pinned the old cluster during a
    # migration while the tier supplied the NEW cluster's credential, and the container was one
    # restart away from a 401 that would have stopped conversations saving.
    if tier and not _node_overrides_tier():
        user = os.getenv(f"OPENSEARCH_USERNAME_{tier.upper()}")
        pwd = os.getenv(f"OPENSEARCH_PASSWORD_{tier.upper()}")
        if user or pwd:
            return (user or "").strip(), (pwd or "")
    return (os.getenv("OPENSEARCH_USERNAME") or "").strip(), (os.getenv("OPENSEARCH_PASSWORD") or "")


def opensearch_credential_warning() -> Optional[str]:
    """Said at boot when a tier is set but its credential is not tier-specific.

    Not an error — one host serving one tier forever is a perfectly good arrangement. But on a
    host that FLIPS tiers it is the trap: ``PLATFORM_TIER`` moves the cluster and leaves the
    password behind, and the 401 that follows reads as a network problem.
    """
    try:
        tier = current_tier()
    except ValueError:
        return None
    if not tier:
        return None
    if os.getenv(f"OPENSEARCH_USERNAME_{tier.upper()}") or os.getenv(f"OPENSEARCH_PASSWORD_{tier.upper()}"):
        return None
    if not (os.getenv("OPENSEARCH_USERNAME") or os.getenv("OPENSEARCH_PASSWORD")):
        return None
    return (f"PLATFORM_TIER={tier} but the OpenSearch credential is the un-tiered "
            f"OPENSEARCH_USERNAME/PASSWORD. Set OPENSEARCH_USERNAME_{tier.upper()} and "
            f"OPENSEARCH_PASSWORD_{tier.upper()} so switching tiers switches the credential too.")


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
                "The explicit setting wins, and the UNTIERED credential is used with it, "
                "because the tier's credential belongs to the tier's cluster.")
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
