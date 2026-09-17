"""Who is making this request, according to the I-GUIDE platform's own JWT.

The platform mints a one-hour HS256 access token carrying exactly ``{id, role}`` and sets it as
an httpOnly cookie scoped to ``.i-guide.io``. Because the agent is served from
``agent.i-guide.io`` — same registrable domain, and same ORIGIN as the map UI — that cookie
arrives here on its own, including on the ``<img>`` request for an inline map artifact. That is
why this file verifies a token rather than exchanging one, and why downloads need no signed URL.

Verification is LOCAL. The alternative, calling the backend's ``/api/check-tokens`` on every
request, buys only "the agent does not hold the signing secret" — and the secret is already in
this deployment's .env, so there is nothing left to buy. Local verification also removes a hard
dependency on the backend being up in order to answer a question.

Deliberate choices, each of which is a way this goes wrong if reversed
---------------------------------------------------------------------
* ``algorithms=["HS256"]`` is PINNED. A decoder that trusts the token's own ``alg`` header
  accepts ``none`` and will happily validate a forgery the attacker wrote.
* ``exp`` is REQUIRED. Without it a token issued once is valid forever.
* A missing or non-numeric ``role`` is REJECTED, never defaulted. Defaulting to 0 would make an
  unparseable token the most privileged caller on the system.
* Expiry raises a DIFFERENT exception from invalidity, because the client must be able to tell
  "refresh and retry" from "give up". Collapsing them means the UI either never refreshes or
  refreshes forever against a token that will never validate.

The role scale is the platform's, and it runs BACKWARDS: lower is more privileged
(``utils/utils.js``). SUPER_ADMIN 1, ADMIN 2, CONTENT_MODERATOR 3, UNRESTRICTED_CONTRIBUTOR 4,
TRUSTED_USER_PLUS 5, TRUSTED_USER 8, UNTRUSTED_USER 10.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import jwt
import requests

# Admit UNRESTRICTED_CONTRIBUTOR (4) and above. This excludes an ordinary TRUSTED_USER (8) — a
# normal .edu account — on purpose: the agent spends LLM budget and runs generated code in a
# sandbox, so access starts narrow. Widening it is this one number.
DEFAULT_MIN_ROLE = 4

# Clocks drift between the minting host and this one. Thirty seconds is small enough that it
# cannot meaningfully extend a one-hour token and large enough to absorb ordinary skew.
_LEEWAY_SECONDS = 30

_DEFAULT_COOKIE_NAME = "jwt-access-token"


class IdentityError(Exception):
    """Base class: something about the caller's identity is unusable."""


class TokenMissing(IdentityError):
    """No token was presented at all — the caller has not signed in."""


class TokenExpired(IdentityError):
    """Valid token, past its ``exp``. RECOVERABLE: refresh and retry."""


class TokenInvalid(IdentityError):
    """Signature, structure or claims are wrong. Not recoverable by refreshing."""


class InsufficientRole(IdentityError):
    """Authenticated, but this account is not permitted to use the agent."""

    def __init__(self, role: int, required: int) -> None:
        super().__init__(f"role {role} is not permitted (requires {required} or lower)")
        self.role = role
        self.required = required


class IdentityNotConfigured(IdentityError):
    """Token mode is on but the deployment cannot verify anything. Fails closed."""


@dataclass(frozen=True)
class User:
    """The whole of what the platform tells us about a caller."""

    id: str
    role: int

    def to_dict(self) -> dict:
        return {"id": self.id, "role": self.role}


def cookie_name() -> str:
    """The access-token cookie's name, which DIFFERS between tiers.

    Dev serves ``jwt-access-token-dev``. Reading it from the environment rather than hardcoding
    it is what lets the same image run against either tier.
    """
    return str(os.getenv("JWT_ACCESS_TOKEN_NAME") or _DEFAULT_COOKIE_NAME).strip()


def _secret() -> str:
    return str(os.getenv("JWT_ACCESS_TOKEN_SECRET") or "").strip()


def token_strict() -> bool:
    """Whether an unusable or absent identity is REJECTED, or merely noted.

    ``AGENT_TOKEN_STRICT=0`` is the migration window and nothing else: it lets identity flow and
    records get stamped with an owner while nothing is yet refused, so ownership can be
    backfilled before it starts gating. Delete it once the backfill is done — a permanently
    non-strict token mode is just dev mode wearing a costume.
    """
    return str(os.getenv("AGENT_TOKEN_STRICT") or "1").strip().lower() not in {
        "0", "false", "no", "off"}


def min_role() -> int:
    raw = str(os.getenv("AGENT_MIN_ROLE") or "").strip()
    if not raw:
        return DEFAULT_MIN_ROLE
    try:
        return int(raw)
    except ValueError:
        # A typo here would otherwise silently widen or close access. Refuse to guess.
        raise IdentityNotConfigured(f"AGENT_MIN_ROLE={raw!r} is not a number")


def _coerce_role(value: Any) -> int:
    """The backend stores roles as numbers but its own ``parseRole`` also accepts strings."""
    if isinstance(value, bool):          # bools are ints in Python; a bool role is nonsense
        raise TokenInvalid("token 'role' claim is a boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    raise TokenInvalid("token has no usable 'role' claim")


def decode_token(token: str) -> User:
    """Verify a platform access token and return its caller. Raises on anything unusable."""
    if not token or not str(token).strip():
        raise TokenMissing("no access token presented")
    secret = _secret()
    if not secret:
        raise IdentityNotConfigured(
            "JWT_ACCESS_TOKEN_SECRET is not set; this deployment cannot verify identity")
    try:
        claims = jwt.decode(
            str(token).strip(),
            secret,
            algorithms=["HS256"],          # pinned: never trust the token's own alg
            leeway=_LEEWAY_SECONDS,
            options={"require": ["exp"]},  # a token with no expiry would be eternal
        )
    except jwt.ExpiredSignatureError as exc:
        raise TokenExpired(str(exc) or "access token expired") from exc
    except jwt.InvalidTokenError as exc:
        # Covers a bad signature, a wrong algorithm, malformed segments and a missing `exp`.
        raise TokenInvalid(str(exc) or "invalid access token") from exc

    user_id = claims.get("id")
    if user_id is None or not str(user_id).strip():
        raise TokenInvalid("token has no 'id' claim")
    return User(id=str(user_id).strip(), role=_coerce_role(claims.get("role")))


# ---------------------------------------------------------------------------
# Verifying WITHOUT the signing secret
# ---------------------------------------------------------------------------
# Local verification needs this deployment to hold the platform's HS256 secret, and that secret
# mints a valid token for ANY user on the platform. On the dev tier it is already in this host's
# .env, so there was nothing left to protect and local verification cost nothing.
#
# Against PRODUCTION that reasoning inverts. This host runs LLM-generated code in a sandbox with
# a Docker socket, and putting the production signing secret on it would make a sandbox escape
# equivalent to minting tokens for every I-GUIDE account. So the agent can instead ASK the
# platform who the caller is: it forwards the cookie it received to the backend's own
# /api/check-tokens, which reads it with its own secret and its own cookie name and answers
# {id, role}. Nothing secret lives here, and the cookie name stops mattering too.
#
# The cost is a round-trip per request and a dependency on the backend being reachable. The
# round trip is cached for the token's own lifetime; the dependency is real and deliberate —
# failing closed when identity cannot be established is the correct direction to fail.

_LOCAL = "local"
_INTROSPECT = "introspect"

# Short on purpose. The cache exists to collapse a burst of requests carrying the SAME token into
# one backend call, not to keep a verdict alive: a longer window is a window in which a token
# revoked upstream still works here.
_INTROSPECT_TTL_SECONDS = 60
_INTROSPECT_MAX_ENTRIES = 512
_INTROSPECT_TIMEOUT = 8

# Keyed by a HASH of the token, never the token: this dict is the kind of thing that ends up in a
# heap dump or a debug print, and a raw access token in either is a credential leak.
_introspect_cache: Dict[str, Tuple[float, "User"]] = {}
_introspect_lock = threading.Lock()


def verify_mode() -> str:
    """How this deployment establishes identity: ``local`` (default) or ``introspect``."""
    raw = str(os.getenv("AGENT_TOKEN_VERIFY") or _LOCAL).strip().lower()
    if raw not in (_LOCAL, _INTROSPECT):
        raise IdentityNotConfigured(
            f"AGENT_TOKEN_VERIFY={raw!r} is not a mode. Expected {_LOCAL} or {_INTROSPECT}.")
    return raw


def _check_tokens_url() -> str:
    return str(os.getenv("PLATFORM_CHECK_TOKENS_URL") or "").strip()


def _cache_key(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> Optional["User"]:
    with _introspect_lock:
        hit = _introspect_cache.get(key)
        if not hit:
            return None
        expires, user = hit
        if expires <= time.time():
            _introspect_cache.pop(key, None)
            return None
        return user


def _cache_put(key: str, user: "User") -> None:
    with _introspect_lock:
        if len(_introspect_cache) >= _INTROSPECT_MAX_ENTRIES:
            # Drop the soonest-to-expire rather than an arbitrary entry, so a burst of new
            # tokens cannot evict the ones still actively in use.
            oldest = min(_introspect_cache, key=lambda k: _introspect_cache[k][0])
            _introspect_cache.pop(oldest, None)
        _introspect_cache[key] = (time.time() + _INTROSPECT_TTL_SECONDS, user)


def clear_introspection_cache() -> None:
    with _introspect_lock:
        _introspect_cache.clear()


def introspect_token(token: str) -> User:
    """Ask the platform who this caller is, forwarding the cookie exactly as received.

    Only FAILURES are distinguished by status, and they carry the same meanings the agent's own
    endpoints use, because they come from the same middleware: 401 is an expired token that a
    refresh will fix, 403 is anything else.
    """
    if not token or not str(token).strip():
        raise TokenMissing("no access token presented")
    url = _check_tokens_url()
    if not url:
        raise IdentityNotConfigured(
            "PLATFORM_CHECK_TOKENS_URL is not set; this deployment cannot verify identity")

    key = _cache_key(str(token).strip())
    cached = _cache_get(key)
    if cached is not None:
        return cached

    try:
        resp = requests.get(url, timeout=_INTROSPECT_TIMEOUT,
                            cookies={cookie_name(): str(token).strip()})
    except requests.RequestException as exc:
        # Fails CLOSED. An unreachable backend must not read as "nobody is signed in", which
        # would silently downgrade a token deployment to anonymous access.
        raise IdentityNotConfigured(f"could not reach the platform to verify identity: {exc}")

    if resp.status_code == 401:
        raise TokenExpired("access token expired")
    if resp.status_code == 403:
        raise TokenInvalid("the platform rejected this access token")
    if resp.status_code != 200:
        raise IdentityNotConfigured(
            f"the platform answered {resp.status_code} when asked to verify identity")
    try:
        claims = resp.json() or {}
    except ValueError as exc:
        raise IdentityNotConfigured(f"the platform's verify response was not JSON: {exc}")

    user_id = claims.get("id")
    if user_id is None or not str(user_id).strip():
        raise TokenInvalid("the platform returned no 'id' for this token")
    user = User(id=str(user_id).strip(), role=_coerce_role(claims.get("role")))
    _cache_put(key, user)
    return user


def identify(token: str) -> User:
    """Establish the caller, by whichever route this deployment is configured for."""
    return introspect_token(token) if verify_mode() == _INTROSPECT else decode_token(token)


def authorize(user: User, required: Optional[int] = None) -> None:
    """Role gate. Lower is more privileged, matching the platform's own comparison."""
    threshold = min_role() if required is None else required
    if user.role > threshold:
        raise InsufficientRole(user.role, threshold)


# ---------------------------------------------------------------------------
# The caller, visible to code that has no idea a request exists
# ---------------------------------------------------------------------------
# A ContextVar for the same reason the file store uses one: the file store and the memory layer
# need to know whose request this is, and threading a User through every call site would touch
# every tool. NOTE for anything that hands work to a thread — the streaming worker does — a
# ContextVar does NOT cross threads: copy the context (`contextvars.copy_context()`) and run the
# thread body inside it, exactly as graph_runtime does. Getting this wrong is silent: the owner
# simply reads as None in the worker and every record is written unowned.
_USER: ContextVar[Optional[User]] = ContextVar("agent_identity_user", default=None)


def set_user(user: Optional[User]) -> Any:
    return _USER.set(user)


def reset_user(token: Any) -> None:
    try:
        _USER.reset(token)
    except (ValueError, LookupError):
        # Set in a different context (a worker thread that copied this one). Nothing to reset.
        pass


def current_user() -> Optional[User]:
    return _USER.get()


def current_user_id() -> Optional[str]:
    user = _USER.get()
    return user.id if user else None
