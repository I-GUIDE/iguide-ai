"""An upload belongs to the conversation that made it.

`/agent/files/upload` bound the caller, so token mode stamped `owner_id`, but it never bound the
conversation, and the map UI posted only the files. Measured through this route in dev mode on
2026-10-02, with a thread id sent in the query, the form and a header: the record was stored with
`session: null`. `find_files` reads a record with no session as the legacy pool that every
conversation shares, so:

- `list_conversation_files`, which lists only what this conversation wrote, never listed an
  upload, not even in the conversation that uploaded it;
- in dev and demo mode, where nobody is identified, every conversation could find an upload by
  name. In token mode the owner check still confined it to its user.

These pin the fix. The client sends its thread id with the upload, and the route binds it around
the write, the same way the chat routes bind it around a turn. An upload that names no
conversation is stored exactly as before, so an older client keeps working.

"By name" here is the store's own lookup, `find_files` and `resolve_file_ref`. The file tools'
bare-name read is a directory scan on prototype that never reads a record, so no stamp can scope
it. Stage 17 (`claude/read-by-name-scoping`) moves it onto `find_files`.
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import jwt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime import file_store  # noqa: E402
from agent_runtime.langchain_file_tools import list_conversation_files_tool  # noqa: E402
import api.server as server  # noqa: E402

SECRET = "s" * 64
COOKIE = "jwt-access-token-dev"


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("AGENT_PUBLIC_BASE_URL", "")
    monkeypatch.setenv("JWT_ACCESS_TOKEN_SECRET", SECRET)
    monkeypatch.setenv("JWT_ACCESS_TOKEN_NAME", COOKIE)
    for var in ("AGENT_MODE", "DEMO_MODE", "AGENT_CHAT_API_KEY", "AGENT_TOKEN_STRICT"):
        monkeypatch.delenv(var, raising=False)


def token(*, id="alice", role=4):
    return jwt.encode({"id": id, "role": role, "exp": int(time.time()) + 3600},
                      SECRET, algorithm="HS256")


def upload(*, form=None, query="", cookie=None, name="probe.txt"):
    with server.app.test_client() as client:
        if cookie:
            client.set_cookie(COOKIE, cookie, domain="localhost")
        res = client.post(f"/agent/files/upload{query}",
                          data={**(form or {}), "file": (io.BytesIO(b"payload"), name)},
                          content_type="multipart/form-data")
    assert res.status_code == 200, res.get_data(as_text=True)
    return res.get_json()["files"][0]


def stored(file_id):
    return file_store.get_file_record(file_id)


def in_conversation(session, fn, *args, **kwargs):
    token_ = file_store.set_session(session)
    try:
        return fn(*args, **kwargs)
    finally:
        file_store.reset_session(token_)


def listing(session):
    return json.loads(in_conversation(session, list_conversation_files_tool))


# --- the stamp ------------------------------------------------------------------------

@pytest.mark.parametrize("form,query", [
    ({"thread_id": "sess-a"}, ""),
    ({"threadId": "sess-a"}, ""),
    ({}, "?thread_id=sess-a"),
    ({}, "?threadId=sess-a"),
], ids=["form-snake", "form-camel", "query-snake", "query-camel"])
def test_an_upload_is_stamped_with_the_thread_it_names(form, query):
    """The two spellings `/agent/chat` accepts, in the form or the query string."""
    record = upload(form=form, query=query)

    assert stored(record["file_id"])["session"] == "sess-a"
    assert record["session"] == "sess-a", "the client can see which conversation it landed in"


def test_the_id_is_normalised_the_way_the_chat_route_normalises_it():
    """A turn binds the stripped id, so a stamp that kept the spaces would name no conversation."""
    record = upload(form={"thread_id": "  sess-a \n"})

    assert stored(record["file_id"])["session"] == "sess-a"


def test_the_form_wins_over_the_query():
    """The body is what the client composed for this upload; a query can come from a stale URL."""
    record = upload(form={"thread_id": "sess-form"}, query="?thread_id=sess-query")

    assert stored(record["file_id"])["session"] == "sess-form"


def test_the_binding_ends_with_the_request():
    """A gunicorn thread serves one request after another, and the conversation must not outlive
    the request that named it. The check has to come straight after that request: a second upload
    proves nothing, because it binds its own id, None included, before it writes."""
    upload(form={"thread_id": "sess-a"})

    assert file_store.current_session() is None, "the upload's conversation outlived its request"


def test_a_later_upload_with_no_thread_id_names_no_conversation():
    """The route binds on every request, even with nothing to bind, so an earlier caller's id
    cannot be inherited."""
    upload(form={"thread_id": "sess-a"})
    later = upload()

    assert stored(later["file_id"])["session"] is None


def test_token_mode_stamps_the_owner_and_the_conversation(monkeypatch):
    """Two axes (S9.3): whose file, and which of their conversations."""
    monkeypatch.setenv("AGENT_MODE", "token")
    record = stored(upload(form={"thread_id": "sess-a"}, cookie=token(id="alice"))["file_id"])

    assert (record["owner_id"], record["session"]) == ("alice", "sess-a")


# --- what the stamp is for ------------------------------------------------------------

def test_it_is_listed_in_the_conversation_that_uploaded_it():
    """The defect: this listing came back empty in the conversation that made the upload."""
    record = upload(form={"thread_id": "sess-a"})

    out = listing("sess-a")

    assert [(f["file_id"], f["kind"]) for f in out["files"]] == [(record["file_id"], "upload")]


def test_it_is_not_listed_in_another_conversation():
    upload(form={"thread_id": "sess-a"})

    out = listing("sess-b")

    assert out["count"] == 0
    assert "Nothing has been saved in this conversation yet" in out["note"]


def test_in_dev_mode_another_conversation_cannot_find_it_by_name():
    """Nobody is identified here, so the owner check passes everyone and only the conversation
    stamp separates them. Without it, sess-b found and resolved sess-a's upload by name."""
    record = upload(form={"thread_id": "sess-a"}, name="private_notes.txt")
    assert file_store.current_owner() is None, "dev mode: no caller, no owner check"

    assert in_conversation("sess-b", file_store.find_files, name="private_notes") == []
    with pytest.raises(ValueError, match="no stored file matches"):
        in_conversation("sess-b", file_store.resolve_file_ref, "private_notes.txt")

    path, found, _ = in_conversation("sess-a", file_store.resolve_file_ref, "private_notes.txt")
    assert found["file_id"] == record["file_id"]
    assert path.read_bytes() == b"payload"


# --- an upload that names no conversation ---------------------------------------------

def test_an_upload_with_no_thread_id_is_stored_as_before():
    """The fallback, pinned so that tightening it is a decision rather than an accident.

    There is no conversation to stamp. Minting one would hide the file, by name and from the
    listing, from the very conversation that later attaches it; refusing would break every
    client that does not send one yet. So it joins the legacy pool exactly as every upload did
    before: attachable by file id, findable by name, listed nowhere.
    """
    record = upload()
    assert stored(record["file_id"])["session"] is None

    assert listing("sess-a")["count"] == 0
    found = in_conversation("sess-b", file_store.find_files, name="probe")
    assert [r["file_id"] for r in found] == [record["file_id"]]


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_thread_id_is_no_thread_id(blank):
    """Not a conversation called "" that every blank-id client would then share."""
    record = upload(form={"thread_id": blank})

    assert stored(record["file_id"])["session"] is None
