"""Verifying identity WITHOUT holding the platform's signing secret.

The HS256 secret mints a valid token for any I-GUIDE account. This host runs LLM-generated code
in a sandbox with a Docker socket, so on the production tier a sandbox escape that found that
secret would be equivalent to minting tokens for everyone. Introspection removes the secret from
the equation: the agent forwards the cookie to the platform, which reads it with its own secret
and its own cookie name.

Everything below is about the ways that goes wrong. The one that matters most is an unreachable
backend: if "cannot verify" ever resolves to "nobody is signed in", a token deployment silently
degrades to anonymous access, which is the exact opposite of the point.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime import identity as idm  # noqa: E402

URL = "https://backend.i-guide.io/api/check-tokens"
TOKEN = "header.payload.signature"


class FakeResponse:
    def __init__(self, status, payload=None, text_body=None):
        self.status_code = status
        self._payload = payload
        self._text = text_body

    def json(self):
        if self._text is not None:
            raise ValueError("not json")
        return self._payload


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("AGENT_TOKEN_VERIFY", "introspect")
    monkeypatch.setenv("PLATFORM_CHECK_TOKENS_URL", URL)
    monkeypatch.setenv("JWT_ACCESS_TOKEN_NAME", "jwt-access-token")
    monkeypatch.delenv("AGENT_MIN_ROLE", raising=False)
    idm.clear_introspection_cache()
    yield
    idm.clear_introspection_cache()


def answer(monkeypatch, *responses):
    """Serve these responses in order, and record every call made."""
    calls = []

    def fake_get(url, timeout=None, cookies=None):
        calls.append({"url": url, "cookies": dict(cookies or {})})
        return responses[min(len(calls) - 1, len(responses) - 1)]

    monkeypatch.setattr(requests, "get", fake_get)
    return calls


# --- the mode switch ------------------------------------------------------------

def test_local_is_the_default(monkeypatch):
    monkeypatch.delenv("AGENT_TOKEN_VERIFY", raising=False)
    assert idm.verify_mode() == "local"


def test_an_unknown_mode_fails_closed(monkeypatch):
    monkeypatch.setenv("AGENT_TOKEN_VERIFY", "remote")
    with pytest.raises(idm.IdentityNotConfigured):
        idm.verify_mode()


def test_identify_routes_to_introspection(monkeypatch):
    calls = answer(monkeypatch, FakeResponse(200, {"id": "alice", "role": 3}))
    assert idm.identify(TOKEN).id == "alice"
    assert len(calls) == 1          # went to the platform, not to a local signature check


def test_identify_routes_to_local_verification(monkeypatch):
    """And local mode must NOT call out — that is the whole difference between them."""
    monkeypatch.setenv("AGENT_TOKEN_VERIFY", "local")
    monkeypatch.setenv("JWT_ACCESS_TOKEN_SECRET", "s" * 64)
    import jwt as pyjwt
    token = pyjwt.encode({"id": "bob", "role": 2, "exp": int(time.time()) + 60},
                         "s" * 64, algorithm="HS256")
    calls = answer(monkeypatch, FakeResponse(500))
    assert idm.identify(token).id == "bob"
    assert calls == []


# --- what the platform says -----------------------------------------------------

def test_the_caller_comes_from_the_platform(monkeypatch):
    answer(monkeypatch, FakeResponse(200, {"id": "cilogon|xyz", "role": "4"}))
    user = idm.introspect_token(TOKEN)
    assert (user.id, user.role) == ("cilogon|xyz", 4)


def test_the_cookie_is_forwarded_under_its_configured_name(monkeypatch):
    calls = answer(monkeypatch, FakeResponse(200, {"id": "a", "role": 1}))
    idm.introspect_token(TOKEN)
    assert calls[0]["cookies"] == {"jwt-access-token": TOKEN}
    assert calls[0]["url"] == URL


def test_401_is_expired_not_invalid(monkeypatch):
    """Same split the agent's own endpoints use — it comes from the same middleware."""
    answer(monkeypatch, FakeResponse(401))
    with pytest.raises(idm.TokenExpired):
        idm.introspect_token(TOKEN)


def test_403_is_invalid(monkeypatch):
    answer(monkeypatch, FakeResponse(403))
    with pytest.raises(idm.TokenInvalid):
        idm.introspect_token(TOKEN)


def test_a_reply_without_an_id_is_refused(monkeypatch):
    answer(monkeypatch, FakeResponse(200, {"role": 1}))
    with pytest.raises(idm.TokenInvalid):
        idm.introspect_token(TOKEN)


def test_a_reply_without_a_usable_role_is_refused(monkeypatch):
    """Never defaulted — a 0 here would be the most privileged caller on the system."""
    answer(monkeypatch, FakeResponse(200, {"id": "a"}))
    with pytest.raises(idm.TokenInvalid):
        idm.introspect_token(TOKEN)


# --- failing closed --------------------------------------------------------------

def test_an_unreachable_platform_is_not_anonymous_access(monkeypatch):
    """The failure that would matter most: 'cannot verify' must never mean 'not signed in'."""
    def boom(url, timeout=None, cookies=None):
        raise requests.ConnectionError("backend down")
    monkeypatch.setattr(requests, "get", boom)
    with pytest.raises(idm.IdentityNotConfigured):
        idm.introspect_token(TOKEN)


def test_an_unexpected_status_fails_closed(monkeypatch):
    answer(monkeypatch, FakeResponse(502))
    with pytest.raises(idm.IdentityNotConfigured):
        idm.introspect_token(TOKEN)


def test_a_non_json_reply_fails_closed(monkeypatch):
    answer(monkeypatch, FakeResponse(200, text_body="<html>gateway</html>"))
    with pytest.raises(idm.IdentityNotConfigured):
        idm.introspect_token(TOKEN)


def test_an_unconfigured_url_fails_closed(monkeypatch):
    monkeypatch.delenv("PLATFORM_CHECK_TOKENS_URL")
    with pytest.raises(idm.IdentityNotConfigured):
        idm.introspect_token(TOKEN)


def test_no_token_is_still_a_missing_token(monkeypatch):
    calls = answer(monkeypatch, FakeResponse(200, {"id": "a", "role": 1}))
    with pytest.raises(idm.TokenMissing):
        idm.introspect_token("")
    assert calls == []      # and does not waste a round trip asking about nothing


# --- the cache -------------------------------------------------------------------

def test_the_same_token_is_verified_once(monkeypatch):
    calls = answer(monkeypatch, FakeResponse(200, {"id": "a", "role": 1}))
    for _ in range(5):
        idm.introspect_token(TOKEN)
    assert len(calls) == 1


def test_different_tokens_are_verified_separately(monkeypatch):
    calls = answer(monkeypatch, FakeResponse(200, {"id": "a", "role": 1}))
    idm.introspect_token("a.b.c")
    idm.introspect_token("d.e.f")
    assert len(calls) == 2


def test_failures_are_never_cached(monkeypatch):
    """A cached rejection would keep refusing a caller who has since signed in again."""
    answer(monkeypatch, FakeResponse(403))
    for _ in range(2):
        with pytest.raises(idm.TokenInvalid):
            idm.introspect_token(TOKEN)
    calls = answer(monkeypatch, FakeResponse(200, {"id": "a", "role": 1}))
    assert idm.introspect_token(TOKEN).id == "a"
    assert len(calls) == 1


def test_the_cache_expires(monkeypatch):
    calls = answer(monkeypatch, FakeResponse(200, {"id": "a", "role": 1}))
    idm.introspect_token(TOKEN)
    # Capture the real clock FIRST: patching idm.time.time patches the shared module, so a
    # lambda calling time.time() would call itself.
    later = time.time() + 3600
    monkeypatch.setattr(idm.time, "time", lambda: later)
    idm.introspect_token(TOKEN)
    assert len(calls) == 2


def test_the_cache_never_holds_the_raw_token(monkeypatch):
    """This dict is exactly the thing that ends up in a heap dump or a debug print."""
    answer(monkeypatch, FakeResponse(200, {"id": "a", "role": 1}))
    idm.introspect_token(TOKEN)
    assert TOKEN not in idm._introspect_cache
    assert all(len(k) == 64 for k in idm._introspect_cache)     # sha256 hex


def test_the_cache_is_bounded(monkeypatch):
    answer(monkeypatch, FakeResponse(200, {"id": "a", "role": 1}))
    for i in range(idm._INTROSPECT_MAX_ENTRIES + 50):
        idm.introspect_token(f"t.{i}.x")
    assert len(idm._introspect_cache) <= idm._INTROSPECT_MAX_ENTRIES
