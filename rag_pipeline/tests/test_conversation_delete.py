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
        # An OVERWRITE is near-real-time too: until a refresh, search still returns the version
        # that was there before. The base fake hides an unrefreshed write instead, which for an
        # overwrite is the wrong way round: the tombstone that replaces a conversation would
        # vanish from the list at once whether or not the delete asked for a refresh (S9.14).
        self.stale: dict[str, dict] = {}

    def settle(self):
        for doc_id in self.pending_delete:
            self.docs.pop(doc_id, None)
            self.visible.discard(doc_id)
        self.pending_delete.clear()
        self.stale.clear()
        super().settle()

    def index(self, *, index, id, body, refresh=None, **_kw):
        if index == mm.TRACE_INDEX:
            self.traces[id] = dict(body)
            return
        self.pending_delete.discard(id)
        if refresh:
            self.stale.pop(id, None)
        elif id in self.visible and id in self.docs:
            self.stale.setdefault(id, dict(self.docs[id]))
        super().index(index=index, id=id, body=body, refresh=refresh)

    def get(self, *, index, id):
        if id in self.pending_delete:
            from opensearchpy import NotFoundError
            raise NotFoundError(404, "not found", {})
        return super().get(index=index, id=id)

    def search(self, *, index, body):
        if index != mm.TRACE_INDEX:
            docs, visible = self.docs, self.visible
            self.docs, self.visible = {**docs, **self.stale}, visible | set(self.stale)
            try:
                return super().search(index=index, body=body)
            finally:
                self.docs, self.visible = docs, visible
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
    # What is left is the tombstone (S9.14): a timestamp, with no owner and no content.
    assert set(store.docs[mid]) == {mm.TOMBSTONE_FIELD}
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


# --- a deleted conversation stays deleted (S9.14) ----------------------------------
# S9.13 left two ways back, both read from the code: another tab's next turn, and a turn already
# in flight. Measured against this fake before the fix, a `threadId`-only turn re-created the
# conversation in strict mode, and under AGENT_TOKEN_STRICT=0 so did a `memoryId` turn and the
# end-of-turn PUT. The conversation document is now overwritten with a tombstone, and every path
# that writes to a conversation id refuses one.

def _stub_agent(monkeypatch, during=None):
    """The agent, minus the model: records the history it was handed, and can run `during` mid-turn."""
    from agent_runtime import agent_chat_service as acs
    seen = []

    def run(user_input, *, chat_history=None, thread_id=None, **_kw):
        seen.append([m.get("content") for m in chat_history or []])
        return {"final_answer": "ok", "thread_id": thread_id, "available_skills": [],
                "route_trace": {}}

    def stream(user_input, *, chat_history=None, thread_id=None, trace_recorder=None, **_kw):
        seen.append([m.get("content") for m in chat_history or []])
        if trace_recorder:
            trace_recorder({"event": "status", "data": {}})
        yield {"event": "status", "data": {"stage": "working"}}
        if during:
            during()
        yield {"event": "completed", "data": {"answer": "ok", "thread_id": thread_id}}

    monkeypatch.setattr(acs, "run_agent_query", run)
    monkeypatch.setattr(acs, "stream_agent_query_events", stream)
    monkeypatch.setattr(acs, "_llm_for_request", lambda *a, **k: None)
    return seen


def chat(user, **body):
    return client_as(user).post("/agent/chat", json={"userQuery": "what did I ask?", **body})


def deleted(owner=ALICE):
    mid = conversation(owner)
    assert client_as(owner).delete(f"/agent/conversations/{mid}").status_code == 200
    return mid


def assert_still_deleted(store, mid):
    assert set(store.docs[mid]) == {mm.TOMBSTONE_FIELD}, store.docs[mid]
    assert traces_of(store, mid) == []
    assert client_as(ALICE).get("/agent/conversations").get_json()["conversations"] == []


def test_a_stale_tabs_first_turn_cannot_bring_it_back_but_a_new_one_starts(store, monkeypatch):
    """A tab whose first turn had not been answered names its conversation by `threadId` alone,
    as every first turn does. Before, that re-created the deleted conversation in strict mode."""
    seen = _stub_agent(monkeypatch)
    mid = deleted()
    res = chat(ALICE, threadId=mid)
    assert res.status_code == 404 and res.get_json()["reason"] == "not_your_conversation"
    assert seen == [], "the turn must be refused before the agent runs"
    assert_still_deleted(store, mid)
    # ...and the guard costs a first turn nothing: an id with nothing behind it is still created.
    assert chat(ALICE, threadId="sess-a-new-one").status_code == 200
    assert store.docs["sess-a-new-one"]["owner_id"] == ALICE


@pytest.mark.parametrize("strict", ["1", "0"])
def test_a_stale_tabs_next_turn_cannot_bring_it_back(store, monkeypatch, strict):
    """Under AGENT_TOKEN_STRICT=0 an ownerless id was let through, and the turn's
    get_or_create_memory re-created the conversation under the caller."""
    monkeypatch.setenv("AGENT_TOKEN_STRICT", strict)
    _stub_agent(monkeypatch)
    mid = deleted()
    res = chat(ALICE, memoryId=mid, threadId="sess-stale")
    assert res.status_code == 404
    assert_still_deleted(store, mid)


def test_the_streamed_turn_is_refused_too(store, monkeypatch):
    seen = _stub_agent(monkeypatch)
    mid = deleted()
    res = client_as(ALICE).post("/agent/chat/stream",
                                json={"userQuery": "and now?", "threadId": mid})
    body = res.get_data(as_text=True)
    assert "not_your_conversation" in body and seen == []
    assert_still_deleted(store, mid)


@pytest.mark.parametrize("strict", ["1", "0"])
def test_the_end_of_turn_save_cannot_bring_it_back(store, strict, monkeypatch):
    """A turn in flight on another tab ends with a PUT. Under AGENT_TOKEN_STRICT=0 it was let
    through and save_session_snapshot's NotFound branch indexed the conversation again."""
    monkeypatch.setenv("AGENT_TOKEN_STRICT", strict)
    mid = deleted()
    res = client_as(ALICE).put(f"/agent/conversations/{mid}",
                               json={"title": "back", "messages": [], "layers": []})
    assert res.status_code == 404
    assert_still_deleted(store, mid)


def test_a_caller_with_no_identity_cannot_bring_it_back(store, monkeypatch):
    """The service key skips assert_owner (no identity, nothing to check), so the store itself
    must refuse: before, its PUT re-created the conversation, unowned."""
    monkeypatch.setenv("AGENT_CHAT_API_KEY", KEY)
    mid = deleted()
    res = client_as(None).put(f"/agent/conversations/{mid}", headers={"X-API-Key": KEY},
                              json={"title": "back", "messages": [], "layers": []})
    assert res.status_code == 404
    assert_still_deleted(store, mid)


def test_a_turn_running_when_it_was_deleted_leaves_nothing_behind(store, monkeypatch):
    """The turn started before the delete, so no edge check can stop it. At its end it wrote a
    trace, an orphan whose parent has no owner; its answer must not land in the tombstone."""
    from agent_runtime import agent_chat_service as acs
    from rag_pipeline.tests.test_memory_ownership import as_user
    mid = conversation(ALICE)
    _stub_agent(monkeypatch, during=lambda: client_as(ALICE).delete(f"/agent/conversations/{mid}"))
    events = as_user(ALICE, lambda: list(acs.stream_agent_chat_events(
        user_input="still running", thread_id="sess-other-tab", memory_id=mid)))
    assert not any(e["event"] == "trace_saved" for e in events), events
    # The answer still reaches the tab, and its in-process session memory; not the store.
    saved = [e["data"] for e in events if e["event"] == "memory_saved"]
    assert all(d.get("scope") == "session" for d in saved), saved
    assert_still_deleted(store, mid)


def test_a_turn_naming_someone_elses_conversation_as_its_thread_is_refused(store, monkeypatch):
    """Not a delete case, found measuring one. The edge checked `memoryId`; the service keys
    memory on `memoryId or threadId`. Bob sending Alice's id as `threadId` alone had her history
    handed to his agent and his turn appended to her conversation, in strict mode."""
    from rag_pipeline.tests.test_memory_ownership import as_user
    seen = _stub_agent(monkeypatch)
    mid = conversation(ALICE)
    as_user(ALICE, lambda: mm.update_memory(mid, "alice's private question", "m1", "her answer", []))
    res = chat(BOB, threadId=mid)
    assert res.status_code == 404 and res.get_json()["reason"] == "not_your_conversation"
    assert seen == []
    assert [t["userQuery"] for t in store.docs[mid]["chat_history"]] == ["alice's private question"]


def test_a_store_outage_does_not_fail_a_first_turn(store, monkeypatch):
    """A guard against this stage's own edge read, so it passes on the base by design. Before it,
    a first turn's store read happened in the service, which fell back to in-process memory on
    failure. The edge check must not turn an outage into failed first turns, nor let the service
    make the read it could not."""
    seen = _stub_agent(monkeypatch)
    reads = []

    def down(**kw):
        reads.append(kw.get("id"))
        raise ConnectionError("cluster unreachable")

    monkeypatch.setattr(store, "get", down)
    res = chat(ALICE, threadId="sess-during-outage")
    assert res.status_code == 200 and len(seen) == 1
    assert reads == ["sess-during-outage"], "only the edge's read; the service must not retry it"
    assert "sess-during-outage" not in store.docs
