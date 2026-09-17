"""Verifying the platform's access token, and the 401/403 split a client acts on.

Every assertion here is a way the feature fails open if it goes the other way. A decoder that
trusts the token's own `alg` accepts a forgery the caller wrote; one that defaults a missing
`role` to 0 makes an unparseable token the most privileged caller on the system; one that
answers 403 for an expired token leaves the UI unable to distinguish "refresh me" from "give up".
"""
from __future__ import annotations

import base64
import contextvars
import json
import sys
import threading
import time
from pathlib import Path

import jwt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime import identity as idm  # noqa: E402
import api.server as server  # noqa: E402

SECRET = "s" * 64
COOKIE = "jwt-access-token-dev"
SERVICE_KEY = "service-key"


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("JWT_ACCESS_TOKEN_SECRET", SECRET)
    monkeypatch.setenv("JWT_ACCESS_TOKEN_NAME", COOKIE)
    monkeypatch.delenv("AGENT_MIN_ROLE", raising=False)
    monkeypatch.delenv("AGENT_TOKEN_STRICT", raising=False)
    monkeypatch.delenv("AGENT_MODE", raising=False)
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.delenv("AGENT_CHAT_API_KEY", raising=False)


def token(*, id="u-1", role=4, exp_delta=3600, secret=SECRET, alg="HS256", drop=()):
    claims = {"id": id, "role": role, "exp": int(time.time()) + exp_delta}
    for k in drop:
        claims.pop(k, None)
    return jwt.encode(claims, secret, algorithm=alg)


def _b64(obj) -> str:
    return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()


def unsigned_token(*, id="u-1", role=1) -> str:
    """An `alg: none` token, hand-built — PyJWT will not encode one, but an attacker will.

    This is the forgery a decoder that reads the algorithm out of the token itself accepts.
    """
    claims = {"id": id, "role": role, "exp": int(time.time()) + 3600}
    return f"{_b64({'alg': 'none', 'typ': 'JWT'})}.{_b64(claims)}."


# --- decoding -------------------------------------------------------------------

def test_valid_token_yields_its_caller():
    user = idm.decode_token(token(id="cilogon|abc", role=2))
    assert (user.id, user.role) == ("cilogon|abc", 2)


def test_expired_is_its_own_error():
    """NOT TokenInvalid: the client refreshes on this one and gives up on the other."""
    with pytest.raises(idm.TokenExpired):
        idm.decode_token(token(exp_delta=-120))


def test_wrong_signature_is_invalid():
    with pytest.raises(idm.TokenInvalid):
        idm.decode_token(token(secret="other" * 16))


def test_unsigned_token_is_refused():
    """`alg: none` is the forgery a decoder that trusts the header's own alg will accept."""
    with pytest.raises(idm.TokenInvalid):
        idm.decode_token(unsigned_token())


def test_token_without_expiry_is_refused():
    """No `exp` means valid forever."""
    with pytest.raises(idm.TokenInvalid):
        idm.decode_token(token(drop=("exp",)))


def test_missing_token_is_its_own_error():
    for empty in ("", "   ", None):
        with pytest.raises(idm.TokenMissing):
            idm.decode_token(empty)


def test_garbage_is_invalid_not_a_crash():
    with pytest.raises(idm.TokenInvalid):
        idm.decode_token("not.a.jwt")


@pytest.mark.parametrize("bad_id", [None, "", "   "])
def test_token_without_an_id_is_refused(bad_id):
    with pytest.raises(idm.TokenInvalid):
        idm.decode_token(token(id=bad_id))


@pytest.mark.parametrize("bad_role", [None, "contributor", "", True, [4], {"role": 4}])
def test_unusable_role_is_refused_never_defaulted(bad_role):
    """Defaulting a missing role to 0 would make a broken token the most privileged caller."""
    with pytest.raises(idm.TokenInvalid):
        idm.decode_token(token(role=bad_role))


def test_numeric_string_role_is_accepted():
    """The backend's own parseRole accepts '4' as well as 4."""
    assert idm.decode_token(token(role="8")).role == 8


def test_clock_skew_leeway_does_not_reject_a_just_expired_token():
    idm.decode_token(token(exp_delta=-5))          # within leeway
    with pytest.raises(idm.TokenExpired):
        idm.decode_token(token(exp_delta=-120))    # well beyond it


def test_missing_secret_fails_closed():
    import os
    os.environ.pop("JWT_ACCESS_TOKEN_SECRET")
    with pytest.raises(idm.IdentityNotConfigured):
        idm.decode_token(token())


# --- the role gate --------------------------------------------------------------

@pytest.mark.parametrize("role", [1, 2, 3, 4])
def test_contributor_and_above_admitted(role):
    idm.authorize(idm.User(id="u", role=role))


@pytest.mark.parametrize("role", [5, 8, 10, 99])
def test_below_contributor_refused(role):
    with pytest.raises(idm.InsufficientRole) as exc:
        idm.authorize(idm.User(id="u", role=role))
    assert exc.value.role == role and exc.value.required == 4


def test_threshold_is_configurable(monkeypatch):
    monkeypatch.setenv("AGENT_MIN_ROLE", "8")
    idm.authorize(idm.User(id="u", role=8))
    with pytest.raises(idm.InsufficientRole):
        idm.authorize(idm.User(id="u", role=10))


def test_unparseable_threshold_fails_closed(monkeypatch):
    monkeypatch.setenv("AGENT_MIN_ROLE", "contributor")
    with pytest.raises(idm.IdentityNotConfigured):
        idm.min_role()


# --- the caller, across a thread -------------------------------------------------

def test_current_user_crosses_a_thread_only_with_a_copied_context():
    """The streaming worker runs in a thread. A ContextVar does not follow it on its own —

    this is the bug that already shipped once for the file-store session stamp, where the
    owner silently read as None inside the worker and every record was written unowned.
    """
    tok = idm.set_user(idm.User(id="u-42", role=3))
    try:
        naive, copied = {}, {}
        t = threading.Thread(target=lambda: naive.setdefault("id", idm.current_user_id()))
        t.start(); t.join()
        assert naive["id"] is None                      # the trap

        ctx = contextvars.copy_context()
        t = threading.Thread(target=lambda: ctx.run(
            lambda: copied.setdefault("id", idm.current_user_id())))
        t.start(); t.join()
        assert copied["id"] == "u-42"                   # the fix
    finally:
        idm.reset_user(tok)


# --- what the endpoint answers ---------------------------------------------------

def _post(cookies=None, headers=None):
    with server.app.test_client() as client:
        for name, value in (cookies or {}).items():
            client.set_cookie(name, value, domain="localhost")
        return client.post("/agent/chat", json={"userQuery": "hi"}, headers=headers or {})


def test_token_mode_asks_an_anonymous_caller_to_sign_in(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "token")
    res = _post()
    assert res.status_code == 403
    assert res.get_json()["reason"] == "not_signed_in"


def test_token_mode_answers_401_for_an_expired_token(monkeypatch):
    """401, not 403 — this is the signal that says 'refresh once, then retry'."""
    monkeypatch.setenv("AGENT_MODE", "token")
    res = _post(cookies={COOKIE: token(exp_delta=-3600)})
    assert res.status_code == 401
    assert res.get_json()["reason"] == "token_expired"


def test_token_mode_answers_403_for_a_forged_token(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "token")
    res = _post(cookies={COOKIE: token(secret="forged" * 12)})
    assert res.status_code == 403
    assert res.get_json()["reason"] == "token_invalid"


def test_token_mode_refuses_an_under_privileged_account(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "token")
    res = _post(cookies={COOKIE: token(role=8)})
    body = res.get_json()
    assert res.status_code == 403
    assert body["reason"] == "insufficient_role"
    assert body["role"] == 8 and body["requiredRole"] == 4
    # Distinct from the anonymous case: one is fixable by signing in, the other is not.
    assert body["reason"] != "not_signed_in"


def test_service_key_still_gets_in_without_a_jwt(monkeypatch):
    """The eval harness has no browser. Token mode must not lock out scripted callers.

    Exercises the guard rather than the endpoint on purpose: POSTing here would run a real turn
    against live OpenSearch and an LLM.
    """
    monkeypatch.setenv("AGENT_MODE", "token")
    monkeypatch.setenv("AGENT_CHAT_API_KEY", SERVICE_KEY)
    with server.app.test_request_context("/agent/chat", headers={"X-API-KEY": SERVICE_KEY}):
        assert server._require_user() is None           # no identity, and no rejection either


def test_a_bearer_service_key_is_not_mistaken_for_a_token(monkeypatch):
    """`Authorization: Bearer` carries the API key too; only a JWT-shaped value is a token."""
    monkeypatch.setenv("AGENT_MODE", "token")
    monkeypatch.setenv("AGENT_CHAT_API_KEY", SERVICE_KEY)
    with server.app.test_request_context(
            "/agent/chat", headers={"Authorization": f"Bearer {SERVICE_KEY}"}):
        # The point: the key is not even SEEN as a token, so no signature error is raised.
        assert server._extract_user_token() == ""
        assert server._require_user() is None


@pytest.mark.parametrize("mode", ["dev", "demo"])
def test_other_modes_identify_nobody(monkeypatch, mode):
    """Identity is token mode's business. dev and demo keep scoping by conversation."""
    monkeypatch.setenv("AGENT_MODE", mode)
    with server.app.test_request_context("/agent/chat"):
        assert server._require_user() is None


def test_non_strict_still_asks_a_browser_to_sign_in(monkeypatch):
    """Non-strict relaxes OWNERSHIP of pre-ownership records, not "who are you".

    Swallowing the identity error here dropped a signed-out visitor through to the API-key gate,
    which answered "Forbidden: invalid API key" — a message about a credential token mode gives
    them no way to enter, for a problem that is really "please sign in". Observed live.
    """
    monkeypatch.setenv("AGENT_MODE", "token")
    monkeypatch.setenv("AGENT_TOKEN_STRICT", "0")
    with server.app.test_request_context("/agent/chat"):
        with pytest.raises(idm.TokenMissing):
            server._require_user()
    with server.app.test_request_context(
            "/agent/chat", headers={"Cookie": f"{COOKIE}={token(role=10)}"}):
        with pytest.raises(idm.InsufficientRole):
            server._require_user()


def test_non_strict_still_lets_a_service_caller_through(monkeypatch):
    """The one exception: a real credential belonging to something with no browser."""
    monkeypatch.setenv("AGENT_MODE", "token")
    monkeypatch.setenv("AGENT_TOKEN_STRICT", "0")
    monkeypatch.setenv("AGENT_CHAT_API_KEY", SERVICE_KEY)
    with server.app.test_request_context(
            "/agent/chat", headers={"X-API-KEY": SERVICE_KEY,
                                    "Cookie": f"{COOKIE}={token(role=10)}"}):
        assert server._require_user() is None


def test_strict_is_the_default(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "token")
    with server.app.test_request_context("/agent/chat"):
        with pytest.raises(idm.IdentityError):
            server._require_user()
