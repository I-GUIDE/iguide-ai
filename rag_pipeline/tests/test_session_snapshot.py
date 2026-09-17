"""A restored conversation must come back as the conversation, not a text shell of it.

`chat_history` is the AGENT's memory — what was asked and answered, used to give the next turn
context. The user's conversation also has layers on the map, every file uploaded across the
session, a region and a model. Restored from chat_history alone, an answer that says "you can
see these features on the map" comes back beside an empty map, and the transcript lies.

The record is the client's (`sessionStore.ts`, written server-shaped on purpose), so the server
stores it rather than rebuilding it from tool output — and therefore treats it as data: capped,
and stripped of the fields the server owns.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime import identity as idm  # noqa: E402
from rag_pipeline import memory_module as mm  # noqa: E402
from rag_pipeline.tests.test_memory_ownership import FakeOpenSearch, as_user  # noqa: E402


@pytest.fixture(autouse=True)
def store(monkeypatch):
    fake = FakeOpenSearch()
    monkeypatch.setattr(mm, "_get_opensearch_client", lambda: fake)
    monkeypatch.delenv("AGENT_TOKEN_STRICT", raising=False)
    monkeypatch.delenv("AGENT_SESSION_SNAPSHOT_MAX_BYTES", raising=False)
    return fake


def a_conversation(**over):
    """The shape sessionStore.ts actually stores."""
    base = {
        "id": "sess-1", "title": "Flood risk in Champaign", "threadId": "thread-1",
        "createdAt": 1, "updatedAt": 2,
        "messages": [{"role": "user", "text": "buffer the rivers"},
                     {"role": "agent", "text": "Done — see the map."}],
        "layers": [
            # the usual case: a pointer, re-fetched through the live map_layer path
            {"kind": "geojson", "id": "l1", "name": "buffer",
             "sourceUrl": "https://agent.i-guide.io/agent/files/file_abc/download"},
            # delivered inline with no url behind it, so its geometry is the only copy
            {"kind": "geojson", "id": "l2", "name": "hospitals",
             "data": {"type": "FeatureCollection", "features": []}},
        ],
        "fileIds": ["file_abc", "file_def"],
        "region": {"bbox": [-88.3, 40.0, -88.1, 40.2]},
        "model": "gpt-4o", "provider": "openai",
    }
    base.update(over)
    return base


# --- round trip -----------------------------------------------------------------

def test_a_conversation_comes_back_whole():
    mid = as_user("alice", lambda: mm.create_memory("x"))
    as_user("alice", lambda: mm.save_session_snapshot(mid, a_conversation()))
    got = as_user("alice", lambda: mm.get_session_snapshot(mid))
    assert [l["id"] for l in got["layers"]] == ["l1", "l2"]
    assert got["layers"][0]["sourceUrl"].endswith("/agent/files/file_abc/download")
    assert got["layers"][1]["data"]["type"] == "FeatureCollection"
    assert got["region"]["bbox"] == [-88.3, 40.0, -88.1, 40.2]
    assert got["model"] == "gpt-4o"


def test_file_ids_survive():
    """Not cosmetic: the full file toolset attaches only `if input_file_ids`, so a conversation

    restored without them silently loses the ability to analyse its own uploads, and the agent
    will claim it cannot see files the transcript plainly shows.
    """
    mid = as_user("alice", lambda: mm.create_memory("x"))
    as_user("alice", lambda: mm.save_session_snapshot(mid, a_conversation()))
    assert as_user("alice", lambda: mm.get_session_snapshot(mid))["fileIds"] == \
        ["file_abc", "file_def"]


def test_missing_snapshot_reads_as_none_not_an_error():
    mid = as_user("alice", lambda: mm.create_memory("x"))
    assert as_user("alice", lambda: mm.get_session_snapshot(mid)) is None


def test_saving_creates_the_document_when_it_is_new(store):
    as_user("alice", lambda: mm.save_session_snapshot("brand-new", a_conversation()))
    assert store.docs["brand-new"]["owner_id"] == "alice"


# --- the snapshot is data, not authority ----------------------------------------

def test_a_snapshot_cannot_claim_an_owner(store):
    """The whole attack: hand yourself someone else's conversation by saying you own it."""
    mid = as_user("alice", lambda: mm.create_memory("mine"))
    as_user("alice", lambda: mm.save_session_snapshot(mid, a_conversation(owner_id="bob")))
    assert store.docs[mid]["owner_id"] == "alice"
    assert "owner_id" not in store.docs[mid]["session_snapshot"]


def test_a_snapshot_cannot_rewrite_the_agent_memory(store):
    mid = as_user("alice", lambda: mm.create_memory("mine"))
    as_user("alice", lambda: mm.update_memory(mid, "real question", "m", "real answer", []))
    as_user("alice", lambda: mm.save_session_snapshot(
        mid, a_conversation(chat_history=[{"userQuery": "forged", "answer": "forged"}])))
    assert store.docs[mid]["chat_history"] == [
        {"userQuery": "real question", "messageId": "m", "answer": "real answer", "elements": []}]


def test_an_oversized_conversation_is_refused_not_truncated(monkeypatch):
    """Coming back missing half its layers would look like data loss with no explanation."""
    monkeypatch.setenv("AGENT_SESSION_SNAPSHOT_MAX_BYTES", "2000")
    mid = as_user("alice", lambda: mm.create_memory("x"))
    huge = a_conversation(messages=[{"role": "agent", "text": "x" * 5000}])
    with pytest.raises(mm.SnapshotTooLarge):
        as_user("alice", lambda: mm.save_session_snapshot(mid, huge))


# --- the list stays a list ------------------------------------------------------

def test_a_rename_shows_up_in_the_conversation_list():
    mid = as_user("alice", lambda: mm.create_memory("untitled"))
    as_user("alice", lambda: mm.save_session_snapshot(mid, a_conversation(title="Flood work")))
    listed = as_user("alice", lambda: mm.list_memories())
    assert listed[0]["conversationName"] == "Flood work"


def test_listing_never_carries_a_whole_conversation():
    """The projection is explicit for exactly this: a new field must not leak by existing."""
    mid = as_user("alice", lambda: mm.create_memory("x"))
    as_user("alice", lambda: mm.save_session_snapshot(mid, a_conversation()))
    listed = as_user("alice", lambda: mm.list_memories())
    assert listed and all("session_snapshot" not in c and "chat_history" not in c
                          for c in listed)
