"""Files belong to the user who made them, and the download endpoint enforces it.

Before this, `GET /agent/files/<id>/download` served any file to anyone holding an id — and
every answer publishes ids as download links, so each one was effectively a permanent public
URL. The tests that matter here are the negative ones.

Ownership is a SECOND axis, not a replacement for the conversation stamp: a user has many
conversations, and in dev/demo there is no user at all. Outside token mode nothing changes.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import jwt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime import file_store, identity as idm  # noqa: E402
import api.server as server  # noqa: E402

SECRET = "s" * 64
COOKIE = "jwt-access-token-dev"


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("JWT_ACCESS_TOKEN_SECRET", SECRET)
    monkeypatch.setenv("JWT_ACCESS_TOKEN_NAME", COOKIE)
    monkeypatch.delenv("AGENT_MODE", raising=False)
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.delenv("AGENT_TOKEN_STRICT", raising=False)
    monkeypatch.delenv("AGENT_CHAT_API_KEY", raising=False)
    yield tmp_path


def token(*, id="alice", role=4):
    return jwt.encode({"id": id, "role": role, "exp": int(time.time()) + 3600},
                      SECRET, algorithm="HS256")


def as_user(user_id, fn, role=4):
    tok = idm.set_user(idm.User(id=user_id, role=role))
    try:
        return fn()
    finally:
        idm.reset_user(tok)


def write(name="out.txt", body="x"):
    return file_store.create_output_file(name, body)


def fetch(file_id, cookie=None):
    with server.app.test_client() as client:
        if cookie:
            client.set_cookie(COOKIE, cookie, domain="localhost")
        return client.get(f"/agent/files/{file_id}/download")


# --- stamping -------------------------------------------------------------------

def test_a_file_records_who_made_it():
    rec = as_user("alice", lambda: write())
    assert rec["owner_id"] == "alice"


def test_without_identity_a_file_has_no_owner():
    """dev and demo identify nobody, and must keep working exactly as before."""
    assert write()["owner_id"] is None


def test_owner_is_independent_of_the_conversation():
    """Same user, two conversations: both files are theirs."""
    def two():
        a = file_store.set_session("thread-1")
        first = write("a.txt")
        file_store.reset_session(a)
        b = file_store.set_session("thread-2")
        second = write("b.txt")
        file_store.reset_session(b)
        return first, second
    first, second = as_user("alice", two)
    assert first["owner_id"] == second["owner_id"] == "alice"
    assert first["session"] != second["session"]


# --- who may read ---------------------------------------------------------------

def test_the_owner_may_read_their_own_file():
    rec = as_user("alice", lambda: write())
    assert as_user("alice", lambda: file_store.may_read(rec)) is True


def test_another_user_may_not():
    rec = as_user("alice", lambda: write())
    assert as_user("bob", lambda: file_store.may_read(rec)) is False
    assert as_user("bob", lambda: file_store.may_read(rec, allow_unowned=False)) is False


def test_unowned_is_the_caller_choice_not_the_stores():
    """1,325 records predate ownership. Reuse wants them; a browser download does not."""
    legacy = write()                       # written with nobody signed in
    assert as_user("alice", lambda: file_store.may_read(legacy, allow_unowned=True)) is True
    assert as_user("alice", lambda: file_store.may_read(legacy, allow_unowned=False)) is False


def test_with_no_caller_everything_is_readable():
    rec = as_user("alice", lambda: write())
    assert file_store.may_read(rec, allow_unowned=False) is True     # dev / demo / service


# --- lookups never surface someone else's file -----------------------------------

def test_find_files_hides_other_users_files():
    as_user("alice", lambda: write("alice-secret.txt"))
    as_user("bob", lambda: write("bob-secret.txt"))
    names = as_user("bob", lambda: [r["filename"] for r in
                                    file_store.find_files(session=None, limit=50)])
    assert "bob-secret.txt" in names
    assert "alice-secret.txt" not in names


def test_find_files_still_offers_the_legacy_pool():
    write("legacy-package.npz")             # unowned, as every pre-ownership record is
    names = as_user("alice", lambda: [r["filename"] for r in
                                      file_store.find_files(session=None, limit=50)])
    assert "legacy-package.npz" in names


# --- the endpoint ---------------------------------------------------------------

def test_download_is_unchanged_in_dev_mode(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "dev")
    rec = write()
    assert fetch(rec["file_id"]).status_code == 200


def test_owner_can_download_in_token_mode(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "token")
    rec = as_user("alice", lambda: write("mine.txt", "hello"))
    res = fetch(rec["file_id"], cookie=token(id="alice"))
    assert res.status_code == 200
    assert res.data == b"hello"


def test_another_user_gets_404_not_403(monkeypatch):
    """404 on purpose: a 403 confirms the id exists and makes this an enumeration oracle."""
    monkeypatch.setenv("AGENT_MODE", "token")
    rec = as_user("alice", lambda: write("mine.txt"))
    res = fetch(rec["file_id"], cookie=token(id="bob"))
    assert res.status_code == 404
    assert "forbidden" not in res.get_data(as_text=True).lower()


def test_anonymous_cannot_download_in_token_mode(monkeypatch):
    """The hole this closes: any id, no credential, any file."""
    monkeypatch.setenv("AGENT_MODE", "token")
    rec = as_user("alice", lambda: write("mine.txt"))
    res = fetch(rec["file_id"])
    assert res.status_code == 403
    assert res.get_json()["reason"] == "not_signed_in"


def test_expired_token_gets_401_here_too(monkeypatch):
    """An <img> that 401s can be retried after a refresh; a 403 tells the client to stop."""
    monkeypatch.setenv("AGENT_MODE", "token")
    rec = as_user("alice", lambda: write("mine.txt"))
    stale = jwt.encode({"id": "alice", "role": 4, "exp": int(time.time()) - 3600},
                       SECRET, algorithm="HS256")
    assert fetch(rec["file_id"], cookie=stale).status_code == 401


def test_unowned_denied_when_strict(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "token")
    rec = write()                                   # legacy, unowned
    assert fetch(rec["file_id"], cookie=token(id="alice")).status_code == 404


def test_unowned_allowed_during_the_migration(monkeypatch):
    """AGENT_TOKEN_STRICT=0 is what makes a backfill possible without an outage."""
    monkeypatch.setenv("AGENT_MODE", "token")
    monkeypatch.setenv("AGENT_TOKEN_STRICT", "0")
    rec = write()
    assert fetch(rec["file_id"], cookie=token(id="alice")).status_code == 200
