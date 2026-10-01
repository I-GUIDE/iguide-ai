"""Who may call the data-bearing routes, after the 2026-10-01 integration.

The credential model is prototype's: a verified platform user, the service key, or demo mode,
as ALTERNATIVES (``_require_user`` then ``_require_agent_chat_api_key(user)``). backend_swap's
M0 contribution that survives is coverage: ``/query`` and ``/query/batch`` — the full RAG
pipeline — never asked who the caller was, and now go through the same check as the agent routes.

backend_swap's fail-CLOSED rule (no key configured -> 500) did not survive: it cannot coexist with
prototype's dev and demo modes, and prototype documents an unset key as "auth disabled" outside
token mode, where identity refuses anonymous callers anyway. Pinned below as a decision, not an
accident, so changing it is deliberate.

``/agent/files/<id>/download`` is deliberately NOT key-gated: map layers load their GeoJSON from
``download_url`` directly, and a browser loading a map source cannot attach an X-API-KEY header.
It is owner-checked in token mode, and an unknown id is a 404 so the route is no existence oracle.

``/health`` stays open (it is the container healthcheck).
"""

from __future__ import annotations

import pytest

# Routes that must never serve data without a valid key.
PROTECTED = [
    ("POST", "/query"),
    ("POST", "/query/batch"),
    ("POST", "/agent/chat"),
    ("POST", "/agent/chat/stream"),
    ("POST", "/agent/files/upload"),
]

OPEN = [("GET", "/health")]


def _client(monkeypatch, *, api_key=None):
    if api_key is None:
        monkeypatch.delenv("AGENT_CHAT_API_KEY", raising=False)
    else:
        monkeypatch.setenv("AGENT_CHAT_API_KEY", api_key)
    import api.server as srv
    return srv.app.test_client()


def _call(client, method, path):
    return client.get(path) if method == "GET" else client.post(path, json={})


@pytest.mark.parametrize("method,path", PROTECTED)
def test_missing_key_is_rejected(monkeypatch, method, path):
    """A configured key with none presented must be a 403, not a served request."""
    client = _client(monkeypatch, api_key="s3cret")
    resp = _call(client, method, path)
    assert resp.status_code == 403, f"{method} {path} returned {resp.status_code}"


@pytest.mark.parametrize("method,path", PROTECTED)
def test_wrong_key_is_rejected(monkeypatch, method, path):
    client = _client(monkeypatch, api_key="s3cret")
    headers = {"X-API-KEY": "wrong"}
    resp = (client.get(path, headers=headers) if method == "GET"
            else client.post(path, json={}, headers=headers))
    assert resp.status_code == 403, f"{method} {path} returned {resp.status_code}"


def test_no_key_configured_leaves_dev_mode_open(monkeypatch):
    """Prototype's documented choice, kept by the integration: outside token mode an unset service
    key disables the key check. If this ever fails, the decision changed — update the module
    docstring and DEPLOYMENT.md with it."""
    client = _client(monkeypatch, api_key=None)
    resp = client.post("/query", json={})
    assert resp.status_code not in (403, 500), resp.status_code


def test_download_is_not_key_gated_and_is_no_existence_oracle(monkeypatch):
    """A configured key is NOT demanded here (the map client cannot send one), and an unknown id
    answers 404 rather than 403, so probing ids reveals nothing."""
    client = _client(monkeypatch, api_key="s3cret")
    assert client.get("/agent/files/does-not-exist/download").status_code == 404


def test_valid_key_passes_auth(monkeypatch):
    """A correct key gets past auth. Not a 403/500 — the handler's own outcome."""
    client = _client(monkeypatch, api_key="s3cret")
    resp = client.post("/query", json={}, headers={"X-API-KEY": "s3cret"})
    assert resp.status_code not in (403, 500)


def test_bearer_token_accepted(monkeypatch):
    client = _client(monkeypatch, api_key="s3cret")
    resp = client.post("/query", json={}, headers={"Authorization": "Bearer s3cret"})
    assert resp.status_code not in (403, 500)


@pytest.mark.parametrize("method,path", OPEN)
def test_open_routes_stay_open(monkeypatch, method, path):
    client = _client(monkeypatch, api_key="s3cret")
    assert _call(client, method, path).status_code == 200


def test_cors_is_not_wildcard(monkeypatch):
    """CORS(app) with no origins allowed any site to call the API from a browser."""
    monkeypatch.delenv("AGENT_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("ALLOWED_DOMAIN_LIST", raising=False)
    import api.server as srv
    assert srv._cors_origins() == []

    monkeypatch.setenv("AGENT_CORS_ORIGINS", "https://platform.i-guide.io, http://localhost:3000")
    assert srv._cors_origins() == ["https://platform.i-guide.io", "http://localhost:3000"]


def test_cors_falls_back_to_existing_allowed_domain_list(monkeypatch):
    """Deployments already carrying ALLOWED_DOMAIN_LIST keep working."""
    monkeypatch.delenv("AGENT_CORS_ORIGINS", raising=False)
    monkeypatch.setenv("ALLOWED_DOMAIN_LIST", '["https://dev.i-guide.io", "http://localhost"]')
    import api.server as srv
    assert srv._cors_origins() == ["https://dev.i-guide.io", "http://localhost"]
