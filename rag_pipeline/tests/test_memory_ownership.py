"""A conversation belongs to the user who had it.

Before this, `get_or_create_memory(memory_id)` fetched by bare UUID with no owner check: anyone
holding an id read that transcript. Worse, the create half of get-or-create would have INDEXED
over a document it had just refused to read, replacing the owner's conversation with an empty
one — which is why ownership is asserted at the edge rather than guarded at each call.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime import identity as idm  # noqa: E402
from rag_pipeline import memory_module as mm  # noqa: E402


class FakeOpenSearch:
    """Enough OpenSearch to exercise ownership: get / index / update / search-by-term."""

    def __init__(self):
        self.docs: dict[str, dict] = {}

    def index(self, *, index, id, body):
        self.docs[id] = dict(body)

    def get(self, *, index, id):
        if id not in self.docs:
            from opensearchpy import NotFoundError
            raise NotFoundError(404, "not found", {})
        return {"_source": dict(self.docs[id])}

    def update(self, *, index, id, body):
        self.docs.setdefault(id, {}).update(body["doc"])

    def search(self, *, index, body):
        want = body["query"]["term"]["owner_id"]
        keep = body.get("_source")
        def project(doc):
            return {k: v for k, v in doc.items() if k in keep} if keep else dict(doc)
        hits = [{"_id": k, "_source": project(v)} for k, v in self.docs.items()
                if v.get("owner_id") == want]
        hits.sort(key=lambda h: self.docs[h["_id"]].get("updatedAt") or "", reverse=True)
        return {"hits": {"hits": hits[:body.get("size", 10)]}}


@pytest.fixture(autouse=True)
def store(monkeypatch):
    fake = FakeOpenSearch()
    monkeypatch.setattr(mm, "_get_opensearch_client", lambda: fake)
    monkeypatch.delenv("AGENT_TOKEN_STRICT", raising=False)
    return fake


def as_user(user_id, fn, role=4):
    token = idm.set_user(idm.User(id=user_id, role=role))
    try:
        return fn()
    finally:
        idm.reset_user(token)


# --- stamping -------------------------------------------------------------------

def test_a_conversation_records_its_owner(store):
    mid = as_user("alice", lambda: mm.create_memory("first"))
    assert store.docs[mid]["owner_id"] == "alice"


def test_without_identity_a_conversation_is_unowned(store):
    mid = mm.create_memory("anonymous")
    assert store.docs[mid]["owner_id"] is None


def test_get_or_create_claims_the_id_on_creation(store):
    """Otherwise the next caller inherits the conversation simply by knowing its id."""
    as_user("alice", lambda: mm.get_or_create_memory("known-id"))
    assert store.docs["known-id"]["owner_id"] == "alice"


# --- the gate -------------------------------------------------------------------

def test_the_owner_may_open_their_conversation(store):
    mid = as_user("alice", lambda: mm.create_memory("mine"))
    as_user("alice", lambda: mm.assert_owner(mid))          # no raise


def test_another_user_may_not(store):
    mid = as_user("alice", lambda: mm.create_memory("mine"))
    with pytest.raises(mm.MemoryAccessDenied):
        as_user("bob", lambda: mm.assert_owner(mid))


def test_refusing_does_not_destroy_the_conversation(store):
    """The get-or-create trap: a refused read must never fall through to an indexing CREATE."""
    mid = as_user("alice", lambda: mm.create_memory("precious"))
    as_user("alice", lambda: mm.update_memory(mid, "q", "m1", "a", []))
    before = list(store.docs[mid]["chat_history"])
    with pytest.raises(mm.MemoryAccessDenied):
        as_user("bob", lambda: mm.assert_owner(mid))
    assert store.docs[mid]["chat_history"] == before
    assert store.docs[mid]["owner_id"] == "alice"


def test_unowned_is_reachable_only_during_the_migration(store, monkeypatch):
    mm.create_memory("legacy")                              # unowned, as every old doc is
    legacy_id = next(iter(store.docs))
    monkeypatch.setenv("AGENT_TOKEN_STRICT", "0")
    as_user("alice", lambda: mm.assert_owner(legacy_id))    # allowed while migrating
    monkeypatch.setenv("AGENT_TOKEN_STRICT", "1")
    with pytest.raises(mm.MemoryAccessDenied):
        as_user("alice", lambda: mm.assert_owner(legacy_id))


def test_no_identity_means_nothing_to_enforce(store):
    mid = as_user("alice", lambda: mm.create_memory("mine"))
    mm.assert_owner(mid)                                    # dev / demo / service: no raise


# --- writes never move a conversation between users ------------------------------

def test_a_write_does_not_reassign_an_owned_conversation(store):
    mid = as_user("alice", lambda: mm.create_memory("mine"))
    as_user("bob", lambda: mm.update_memory(mid, "q", "m", "a", []))
    assert store.docs[mid]["owner_id"] == "alice"


def test_a_write_attributes_an_unowned_conversation(store):
    mm.create_memory("legacy")
    legacy_id = next(iter(store.docs))
    as_user("alice", lambda: mm.update_memory(legacy_id, "q", "m", "a", []))
    assert store.docs[legacy_id]["owner_id"] == "alice"


def test_a_write_stamps_updated_at(store):
    """Without it every conversation sorts equal and the user's list is arbitrary."""
    mid = as_user("alice", lambda: mm.create_memory("mine"))
    first = store.docs[mid]["updatedAt"]
    as_user("alice", lambda: mm.update_memory(mid, "q", "m", "a", []))
    assert store.docs[mid]["updatedAt"] >= first


# --- my conversations -----------------------------------------------------------

def test_listing_returns_only_my_conversations(store):
    as_user("alice", lambda: mm.create_memory("alice one"))
    as_user("alice", lambda: mm.create_memory("alice two"))
    as_user("bob", lambda: mm.create_memory("bob one"))
    names = as_user("alice", lambda: [c["conversationName"] for c in mm.list_memories()])
    assert sorted(names) == ["alice one", "alice two"]


def test_listing_without_identity_is_empty_not_everything(store):
    """The failure that would matter: no caller resolving to 'all conversations'."""
    as_user("alice", lambda: mm.create_memory("alice one"))
    assert mm.list_memories() == []


def test_listing_never_returns_transcripts(store):
    mid = as_user("alice", lambda: mm.create_memory("mine"))
    as_user("alice", lambda: mm.update_memory(mid, "secret question", "m", "secret answer", []))
    listed = as_user("alice", lambda: mm.list_memories())
    assert listed and all("chat_history" not in c for c in listed)
