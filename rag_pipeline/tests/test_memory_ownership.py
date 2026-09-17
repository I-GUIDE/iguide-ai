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
        # Real OpenSearch refuses to update a document that does not exist; a fake that quietly
        # creates one hides every code path that depends on the difference.
        if id not in self.docs:
            from opensearchpy import NotFoundError
            raise NotFoundError(404, "not found", {})
        self.docs[id].update(body["doc"])

    def search(self, *, index, body):
        # Analysis is modelled, not assumed. The previous fake read `term.owner_id` and
        # compared it to the stored value, which is what real OpenSearch does only for a
        # `keyword` field; on the `text` field this index actually has, a term query matches
        # TOKENS. So the fake passed while production listed nothing, for months. Here a term
        # query on the bare field matches only if the value is a single token, which is the
        # rule that makes the real failure reproducible: a platform id is a URL, and a URL is
        # not one token.
        query = body["query"]
        clauses = query["bool"]["filter"] if "bool" in query else [query]
        want, tokenised, must_exist = None, True, []
        for clause in clauses:
            if "term" in clause:
                term = clause["term"]
                if "owner_id.keyword" in term:
                    want, tokenised = term["owner_id.keyword"], False
                else:
                    want, tokenised = term["owner_id"], True
            elif "exists" in clause:
                must_exist.append(clause["exists"]["field"])
        keep = body.get("_source")
        def project(doc):
            return {k: v for k, v in doc.items() if k in keep} if keep else dict(doc)
        def matches(doc):
            stored = doc.get("owner_id")
            owned = stored == want and (not tokenised or len(_tokens(want)) == 1)
            return owned and all(doc.get(f) is not None for f in must_exist)
        hits = [{"_id": k, "_source": project(v)} for k, v in self.docs.items()
                if matches(v)]
        hits.sort(key=lambda h: self.docs[h["_id"]].get("updatedAt") or "", reverse=True)
        return {"hits": {"hits": hits[:body.get("size", 10)]}}


def _tokens(value: str) -> list:
    """Roughly what the standard analyser does to a string: split on non-alphanumerics."""
    import re
    return [t for t in re.split(r"[^A-Za-z0-9.]+", str(value)) if t]


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
    conversation("alice one", "alice")
    conversation("alice two", "alice")
    conversation("bob one", "bob")
    names = as_user("alice", lambda: [c["conversationName"] for c in mm.list_memories()])
    assert sorted(names) == ["alice one", "alice two"]


def test_listing_without_identity_is_empty_not_everything(store):
    """The failure that would matter: no caller resolving to 'all conversations'."""
    as_user("alice", lambda: mm.create_memory("alice one"))
    assert mm.list_memories() == []


# A conversation the CLIENT has stored its view of. Creating a memory is not enough to be
# listed: the list and the detail endpoint must agree, and the detail serves `session_snapshot`.
def conversation(name, owner):
    mid = as_user(owner, lambda: mm.create_memory(name))
    as_user(owner, lambda: mm.save_session_snapshot(mid, {"title": name, "messages": [], "layers": []}))
    return mid


# The shape a platform account ACTUALLY has. Every other test here uses "alice", which is a
# single token and therefore matches under any query — which is precisely why none of them
# caught the field this searched for months. The id is the thing under test.
PLATFORM_ID = "http://cilogon.org/serverE/users/137206"


def test_listing_finds_a_real_platform_id(store):
    """The live failure: three stored conversations, an empty list, and no error anywhere.

    `owner_id` is mapped `text` with a `keyword` subfield, so a term query on the bare field
    compares the whole id against single TOKENS of it. A CILogon URL tokenises to `http`,
    `cilogon.org`, `users`, `137206` — none of which is the id — so the filter matched nothing
    while saving, fetching by id and every ownership check kept working, because those go by
    document id and never search.
    """
    conversation("first", PLATFORM_ID)
    conversation("second", PLATFORM_ID)
    names = as_user(PLATFORM_ID, lambda: [c["conversationName"] for c in mm.list_memories()])
    assert sorted(names) == ["first", "second"]


def test_a_real_platform_id_still_excludes_other_people(store):
    other = "http://cilogon.org/serverE/users/999999"
    conversation("mine", PLATFORM_ID)
    conversation("theirs", other)
    assert as_user(PLATFORM_ID,
                   lambda: [c["conversationName"] for c in mm.list_memories()]) == ["mine"]
    assert as_user(other,
                   lambda: [c["conversationName"] for c in mm.list_memories()]) == ["theirs"]


def test_listing_never_returns_transcripts(store):
    mid = conversation("mine", "alice")
    as_user("alice", lambda: mm.update_memory(mid, "secret question", "m", "secret answer", []))
    listed = as_user("alice", lambda: mm.list_memories())
    assert listed and all("chat_history" not in c for c in listed)


def test_a_memory_with_no_client_snapshot_is_not_offered(store):
    """Listed must mean restorable. The two endpoints read different things.

    This index holds every memory, including ones the AGENT created mid-turn, while
    GET /agent/conversations/<id> serves `session_snapshot` and 404s without one. Listing the
    snapshot-less ones produced rows that rendered, reported their age, and did nothing when
    clicked — observed live as two `conversation-sess-...` entries that could not be opened.
    """
    conversation("openable", "alice")
    as_user("alice", lambda: mm.create_memory("agent-side only"))     # no snapshot ever stored
    names = as_user("alice", lambda: [c["conversationName"] for c in mm.list_memories()])
    assert names == ["openable"]
    # And the reason it must not be listed: there is nothing to give back.
    mid = as_user("alice", lambda: mm.create_memory("still nothing"))
    assert as_user("alice", lambda: mm.get_session_snapshot(mid)) is None
