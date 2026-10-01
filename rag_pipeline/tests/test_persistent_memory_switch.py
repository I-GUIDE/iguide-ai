"""AGENT_PERSISTENT_MEMORY=0: a server that writes no conversation anywhere (M8.66).

The request's ``usePersistentMemory`` was the only switch, and the map UI always sends true. So
a local server started against the deployment-shaped ``.env``, which names the production
cluster, wrote every test turn into production's chat_memory and chat_traces. Two write paths
reach the conversation store from a request: the chat turn itself, gated by the request flag,
and ``PUT /agent/conversations/<id>``. The switch closes both, and unset changes nothing.
"""
from __future__ import annotations

import pytest

import api.server as server


def _normalized(**extra):
    return server._normalize_agent_chat_request({"user_input": "hi", **extra})


def test_unset_honours_the_request(monkeypatch):
    monkeypatch.delenv("AGENT_PERSISTENT_MEMORY", raising=False)
    assert _normalized()["use_persistent_memory"] is True
    assert _normalized(use_persistent_memory=False)["use_persistent_memory"] is False


@pytest.mark.parametrize("value", ["0", "false", "off", "no"])
def test_off_beats_a_client_that_asks_for_persistence(monkeypatch, value):
    monkeypatch.setenv("AGENT_PERSISTENT_MEMORY", value)
    assert _normalized(use_persistent_memory=True)["use_persistent_memory"] is False
    assert _normalized(usePersistentMemory=True)["use_persistent_memory"] is False


def test_off_refuses_to_save_a_conversation_rather_than_pretending(monkeypatch):
    def must_not_write(*args, **kwargs):
        raise AssertionError("a conversation was written with persistence off")

    monkeypatch.setenv("AGENT_PERSISTENT_MEMORY", "0")
    monkeypatch.setattr(server, "assert_memory_owner", lambda memory_id: None)
    monkeypatch.setattr(server, "save_session_snapshot", must_not_write)
    resp = server.app.test_client().put("/agent/conversations/abc", json={"messages": []})
    assert resp.status_code == 409, resp.get_data(as_text=True)
    assert resp.get_json()["reason"] == "persistence_disabled"


def test_unset_still_saves(monkeypatch):
    saved = []
    monkeypatch.delenv("AGENT_PERSISTENT_MEMORY", raising=False)
    monkeypatch.setattr(server, "assert_memory_owner", lambda memory_id: None)
    monkeypatch.setattr(server, "save_session_snapshot",
                        lambda memory_id, body: saved.append(memory_id) or {"memoryId": memory_id})
    resp = server.app.test_client().put("/agent/conversations/abc", json={"messages": []})
    assert resp.status_code == 200 and saved == ["abc"], resp.get_data(as_text=True)
