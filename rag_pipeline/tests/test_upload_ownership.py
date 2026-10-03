"""Uploads need a credential, and carry the owner who made them.

`/agent/files/upload` had NO guard of any kind — not identity, not even the API key. Measured
against the deployed server before this fix: a keyless POST returned 200. Anyone who could reach
the host could put files in a store that the download endpoint then served.

The second half is what made token mode unfinishable: without an identity binding, every upload
was written `owner_id: None`, and under AGENT_TOKEN_STRICT=1 unowned means denied — so a user
could not download their own attachment. Strict mode was unreachable in practice, and the reason
was one missing line rather than anything in the design.
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import jwt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime import file_store  # noqa: E402
import api.server as server  # noqa: E402

SECRET = "s" * 64
COOKIE = "jwt-access-token-dev"
KEY = "service-key"


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("JWT_ACCESS_TOKEN_SECRET", SECRET)
    monkeypatch.setenv("JWT_ACCESS_TOKEN_NAME", COOKIE)
    for var in ("AGENT_MODE", "DEMO_MODE", "AGENT_CHAT_API_KEY", "AGENT_TOKEN_STRICT"):
        monkeypatch.delenv(var, raising=False)


def token(*, id="alice", role=4):
    return jwt.encode({"id": id, "role": role, "exp": int(time.time()) + 3600},
                      SECRET, algorithm="HS256")


def upload(headers=None, cookie=None):
    with server.app.test_client() as client:
        if cookie:
            client.set_cookie(COOKIE, cookie, domain="localhost")
        return client.post("/agent/files/upload",
                           data={"file": (io.BytesIO(b"payload"), "probe.txt")},
                           content_type="multipart/form-data", headers=headers or {})


# --- the credential ---------------------------------------------------------------

def test_a_keyless_upload_is_refused_when_a_key_is_configured(monkeypatch):
    """The hole: this returned 200 against the live server with no credential at all."""
    monkeypatch.setenv("AGENT_CHAT_API_KEY", KEY)
    assert upload().status_code == 403


def test_the_service_key_still_uploads(monkeypatch):
    monkeypatch.setenv("AGENT_CHAT_API_KEY", KEY)
    assert upload(headers={"X-API-KEY": KEY}).status_code == 200


def test_a_signed_in_user_uploads_without_a_key(monkeypatch):
    """A verified user IS a credential — and token mode hides the settings panel, so demanding
    a key as well refuses them for lacking one they cannot enter."""
    monkeypatch.setenv("AGENT_MODE", "token")
    monkeypatch.setenv("AGENT_CHAT_API_KEY", KEY)
    assert upload(cookie=token()).status_code == 200


def test_an_anonymous_upload_in_token_mode_asks_for_a_sign_in(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "token")
    res = upload()
    assert res.status_code == 403
    assert res.get_json()["reason"] == "not_signed_in"


def test_an_expired_token_gets_401_so_the_client_can_refresh(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "token")
    stale = jwt.encode({"id": "alice", "role": 4, "exp": int(time.time()) - 3600},
                       SECRET, algorithm="HS256")
    assert upload(cookie=stale).status_code == 401


def test_dev_mode_with_no_key_configured_is_unchanged():
    """A local checkout with no AGENT_CHAT_API_KEY must still work as it always has."""
    assert upload().status_code == 200


# --- the owner --------------------------------------------------------------------

def test_an_upload_records_who_made_it(monkeypatch):
    """The line whose absence made AGENT_TOKEN_STRICT=1 unreachable."""
    monkeypatch.setenv("AGENT_MODE", "token")
    body = upload(cookie=token(id="alice")).get_json()
    record = file_store.get_file_record(body["files"][0]["file_id"])
    assert record["owner_id"] == "alice"


def test_the_owner_can_download_their_own_upload_under_strict(monkeypatch):
    """End to end, and the case that was broken: upload signed in, then fetch it back with
    unowned records denied."""
    monkeypatch.setenv("AGENT_MODE", "token")
    monkeypatch.setenv("AGENT_TOKEN_STRICT", "1")
    file_id = upload(cookie=token(id="alice")).get_json()["files"][0]["file_id"]
    with server.app.test_client() as client:
        client.set_cookie(COOKIE, token(id="alice"), domain="localhost")
        res = client.get(f"/agent/files/{file_id}/download")
    assert res.status_code == 200 and res.data == b"payload"


def test_another_user_cannot_fetch_it(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "token")
    monkeypatch.setenv("AGENT_TOKEN_STRICT", "1")
    file_id = upload(cookie=token(id="alice")).get_json()["files"][0]["file_id"]
    with server.app.test_client() as client:
        client.set_cookie(COOKIE, token(id="bob"), domain="localhost")
        assert client.get(f"/agent/files/{file_id}/download").status_code == 404


def test_a_service_upload_is_unowned_not_misattributed(monkeypatch):
    """A caller with no browser has no user. It must not inherit the last one."""
    monkeypatch.setenv("AGENT_MODE", "token")
    monkeypatch.setenv("AGENT_CHAT_API_KEY", KEY)
    body = upload(headers={"X-API-KEY": KEY}).get_json()
    record = file_store.get_file_record(body["files"][0]["file_id"])
    assert record["owner_id"] is None
