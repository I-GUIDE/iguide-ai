"""Deleting a conversation deletes it, and only its owner can.

Before this there was no delete. The history pane's × removed the browser's IndexedDB copy, and
in token mode the list comes from the server, so the next refresh listed the conversation again:
the button did nothing a user could keep. `DELETE /agent/conversations/<id>` removes the
conversation document (agent memory and the client's stored view) and every recorded turn of it
in `chat_traces`, and leaves files alone.

The ownership rules are the GET route's, with one tightening: an unowned conversation, which the
migration window keeps readable by id, can never be deleted.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import jwt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import api.server as server  # noqa: E402
from rag_pipeline import memory_module as mm  # noqa: E402
from rag_pipeline.tests.test_memory_ownership import (FakeOpenSearch, _tokens,  # noqa: E402
                                                      as_user)

SECRET = "s" * 64
COOKIE = "jwt-access-token-dev"
KEY = "service-key"

# Platform ids are URLs, so a term query on the analysed `owner_id` matches nothing (AGENTS.md
# "Ownership"). Single-token ids like "alice" would pass against either field.
ALICE = "http://cilogon.org/serverE/users/137206"
BOB = "http://cilogon.org/serverE/users/999999"


class FakeStore(FakeOpenSearch):
    """The ownership fake, plus deletes, a second index for traces, and both kept near-real-time.

    A delete, like a write, is invisible to search until a refresh: the deleted conversation stays
    in a search result until then, while a GET by id already misses it. Without modelling that,
    a delete that does not ask for a refresh passes here and the client's immediate re-list shows
    the conversation it just deleted.
    """

    def __init__(self):
        super().__init__()
        self.traces: dict[str, dict] = {}
        self.pending_delete: set[str] = set()
        self.trace_failures: list = []

    def settle(self):
        for doc_id in self.pending_delete:
            self.docs.pop(doc_id, None)
            self.visible.discard(doc_id)
        self.pending_delete.clear()
        super().settle()

    def index(self, *, index, id, body, refresh=None, **_kw):
        if index == mm.TRACE_INDEX:
            self.traces[id] = dict(body)
            return
        self.pending_delete.discard(id)
        super().index(index=index, id=id, body=body, refresh=refresh)

    def get(self, *, index, id):
        if id in self.pending_delete:
            from opensearchpy import NotFoundError
            raise NotFoundError(404, "not found", {})
        return super().get(index=index, id=id)

    def search(self, *, index, body):
        if index != mm.TRACE_INDEX:
            return super().search(index=index, body=body)
        want = body["query"]["term"]["memory_id.keyword"]
        hits = [{"_id": k, "_source": dict(v)} for k, v in self.traces.items()
                if v.get("memory_id") == want]
        return {"hits": {"hits": hits[:body.get("size", 10)]}}

    def delete(self, *, index, id, refresh=None, **_kw):
        if id not in self.docs or id in self.pending_delete:
            from opensearchpy import NotFoundError
            raise NotFoundError(404, "not_found", {})
        if refresh:
            self.docs.pop(id)
            self.visible.discard(id)
        else:
            self.pending_delete.add(id)

    def delete_by_query(self, *, index, body, **_kw):
        assert index == mm.TRACE_INDEX
        term = body["query"]["term"]
        # Analysis modelled as for owner_id: a memory id is a UUID, which tokenises on its
        # hyphens, so a term on the bare field matches nothing and deletes nothing.
        if "memory_id.keyword" in term:
            want, tokenised = term["memory_id.keyword"], False
        else:
            want, tokenised = term["memory_id"], True
        if self.trace_failures:
            return {"deleted": 0, "failures": list(self.trace_failures)}
        doomed = [k for k, v in self.traces.items()
                  if v.get("memory_id") == want and (not tokenised or len(_tokens(want)) == 1)]
        for k in doomed:
            del self.traces[k]
        return {"deleted": len(doomed), "failures": []}


@pytest.fixture(autouse=True)
def store(monkeypatch):
    fake = FakeStore()
    monkeypatch.setattr(mm, "_get_opensearch_client", lambda: fake)
    monkeypatch.setenv("AGENT_MODE", "token")
    monkeypatch.setenv("JWT_ACCESS_TOKEN_SECRET", SECRET)
    monkeypatch.setenv("JWT_ACCESS_TOKEN_NAME", COOKIE)
    for var in ("DEMO_MODE", "AGENT_CHAT_API_KEY", "AGENT_TOKEN_STRICT", "AGENT_PERSISTENT_MEMORY"):
        monkeypatch.delenv(var, raising=False)
    return fake


def token(user_id):
    return jwt.encode({"id": user_id, "role": 4, "exp": int(time.time()) + 3600},
                      SECRET, algorithm="HS256")


def client_as(user_id=None):
    client = server.app.test_client()
    if user_id:
        client.set_cookie(COOKIE, token(user_id), domain="localhost")
    return client


def conversation(owner, name="Flood risk", turns=2):
    """A listable conversation, as the client leaves one: a snapshot, and a trace per turn."""
    mid = as_user(owner, lambda: mm.create_memory(name))
    as_user(owner, lambda: mm.save_session_snapshot(
        mid, {"title": name, "messages": [{"role": "user", "text": "secret question"}],
              "layers": []}))
    for n in range(turns):
        as_user(owner, lambda: mm.save_turn_trace(mid, thread_id="t", query=f"q{n}",
                                                   events=[{"type": "status"}], answer="a"))
    return mid


def traces_of(store, mid):
    return [k for k, v in store.traces.items() if v["memory_id"] == mid]


# --- the owner ---------------------------------------------------------------------

def test_the_owner_deletes_their_conversation(store):
    mid = conversation(ALICE)
    res = client_as(ALICE).delete(f"/agent/conversations/{mid}")
    assert res.status_code == 200
    assert res.get_json() == {"deleted": True, "memoryId": mid, "traces": 2}
    store.settle()
    assert mid not in store.docs
    assert traces_of(store, mid) == []


def test_it_is_gone_from_the_list_at_once(store):
    """The client re-lists the instant the delete returns; a delete is not searchable before a
    refresh, so without asking for one the row comes straight back."""
    keep = conversation(ALICE, "keep")
    gone = conversation(ALICE, "gone")
    client = client_as(ALICE)
    assert client.delete(f"/agent/conversations/{gone}").status_code == 200
    listed = client.get("/agent/conversations").get_json()["conversations"]
    assert [c["memoryId"] for c in listed] == [keep]


def test_after_deleting_nothing_of_it_is_readable(store):
    mid = conversation(ALICE)
    client = client_as(ALICE)
    client.delete(f"/agent/conversations/{mid}")
    assert client.get(f"/agent/conversations/{mid}").status_code == 404
    assert client.get(f"/agent/conversations/{mid}/traces").status_code == 404
    assert client.delete(f"/agent/conversations/{mid}").status_code == 404


def test_only_that_conversations_traces_go(store):
    mine = conversation(ALICE, "mine")
    other = conversation(ALICE, "other")
    theirs = conversation(BOB, "theirs")
    client_as(ALICE).delete(f"/agent/conversations/{mine}")
    assert traces_of(store, mine) == []
    assert len(traces_of(store, other)) == 2
    assert len(traces_of(store, theirs)) == 2


def test_orphaned_traces_would_be_readable_while_migrating(store, monkeypatch):
    """Why traces go too. With the conversation gone its traces have no owner to check, and
    AGENT_TOKEN_STRICT=0 lets anyone through to an unowned conversation's traces by id."""
    monkeypatch.setenv("AGENT_TOKEN_STRICT", "0")
    mid = conversation(ALICE)
    assert client_as(ALICE).delete(f"/agent/conversations/{mid}").status_code == 200
    body = client_as(BOB).get(f"/agent/conversations/{mid}/traces").get_json()
    assert not body.get("traces"), body


def test_a_trace_delete_that_fails_keeps_the_conversation(store):
    """Traces first, so a failure leaves an owned conversation that a retry can finish."""
    mid = conversation(ALICE)
    store.trace_failures = [{"cause": "version_conflict"}]
    assert client_as(ALICE).delete(f"/agent/conversations/{mid}").status_code == 500
    assert mid in store.docs and mid not in store.pending_delete
    store.trace_failures = []
    assert client_as(ALICE).delete(f"/agent/conversations/{mid}").status_code == 200


def test_the_deletion_is_logged_without_content(store, caplog):
    mid = conversation(ALICE)
    with caplog.at_level("INFO", logger=server.logger.name):
        client_as(ALICE).delete(f"/agent/conversations/{mid}")
    lines = [r.getMessage() for r in caplog.records if "deleted" in r.getMessage()]
    assert any(mid in line and ALICE in line for line in lines), lines
    assert not any("secret question" in r.getMessage() for r in caplog.records)


# --- everyone else -----------------------------------------------------------------

def test_another_user_gets_404_and_nothing_is_deleted(store):
    mid = conversation(ALICE)
    res = client_as(BOB).delete(f"/agent/conversations/{mid}")
    assert res.status_code == 404, "a 403 would confirm the id exists"
    assert res.get_json()["reason"] == "not_your_conversation"
    assert store.docs[mid]["owner_id"] == ALICE and mid not in store.pending_delete
    assert len(traces_of(store, mid)) == 2


def test_an_unknown_id_gets_404(store):
    conversation(ALICE)
    res = client_as(ALICE).delete("/agent/conversations/no-such-conversation")
    assert res.status_code == 404
    assert len(store.docs) == 1


def test_a_signed_out_caller_is_refused(store):
    mid = conversation(ALICE)
    res = client_as(None).delete(f"/agent/conversations/{mid}")
    assert res.status_code == 403
    assert res.get_json()["reason"] == "not_signed_in"
    assert mid in store.docs and len(traces_of(store, mid)) == 2


def test_the_service_key_cannot_delete(store, monkeypatch):
    """A service caller has no identity, so there is no owner to check it against."""
    monkeypatch.setenv("AGENT_CHAT_API_KEY", KEY)
    mid = conversation(ALICE)
    res = client_as(None).delete(f"/agent/conversations/{mid}", headers={"X-API-Key": KEY})
    assert res.status_code == 403
    assert res.get_json()["reason"] == "identity_required"
    assert mid in store.docs


@pytest.mark.parametrize("mode", ["dev", "demo"])
def test_a_mode_with_no_users_cannot_delete(store, monkeypatch, mode):
    monkeypatch.setenv("AGENT_MODE", mode)
    mid = conversation(ALICE)
    res = client_as(None).delete(f"/agent/conversations/{mid}")
    assert res.status_code == 403
    assert mid in store.docs


def test_an_unowned_conversation_is_never_deletable(store, monkeypatch):
    """The migration window keeps legacy conversations READABLE by id, not destroyable."""
    monkeypatch.setenv("AGENT_TOKEN_STRICT", "0")
    legacy = mm.create_memory("legacy")                      # no identity: unowned
    assert client_as(ALICE).get(f"/agent/conversations/{legacy}").status_code == 404  # no snapshot
    res = client_as(ALICE).delete(f"/agent/conversations/{legacy}")
    assert res.status_code == 404
    assert legacy in store.docs


def test_local_mode_answers_as_the_other_routes_do(store, monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "local")
    mid = conversation(ALICE)
    before = dict(store.docs)
    res = client_as(None).delete(f"/agent/conversations/{mid}")
    assert res.status_code == 404
    assert res.get_json()["reason"] == "persistent_memory_disabled"
    assert store.docs == before
