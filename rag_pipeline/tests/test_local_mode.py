"""AGENT_MODE=local: a developer's machine writes nothing to a shared store.

The guarantee exists because the manual version failed twice. On 2026-10-01 a local verification
run wrote seven conversations into the PRODUCTION OpenSearch cluster: the main checkout's .env
sets OPENSEARCH_NODE to prod, and nothing said so. The recipe recorded afterwards —
PLATFORM_TIER=dev with OPENSEARCH_NODE blank — still wrote, to the DEV cluster, because with the
explicit host blank the tier supplies one. Both failures were silent: a write that should not
happen does not error, it succeeds.

So the tests below are written against those two configurations specifically, and they assert
the store is never OPENED rather than that a write fails, because "it raised" and "it did not
touch the cluster" are not the same claim.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import api.server as server  # noqa: E402
from agent_runtime import deployment_mode, file_store  # noqa: E402
from rag_pipeline import memory_module  # noqa: E402

PROD_NODE = "https://149.165.155.195:9200"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("AGENT_MODE", "DEMO_MODE", "AGENT_CHAT_API_KEY", "PLATFORM_TIER", "SEARCH_TIER",
                 "OPENSEARCH_NODE", "AGENT_PUBLIC_BASE_URL", "FLASK_EMBEDDING_URL", "ANVILGPT_URL",
                 "OPENAI_BASE_URL", "VLLM_BASE_URL", "RS_EMBED_URL", "MCP_SERVER_URL",
                 "MINIO_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    # Never let a test reuse, or leave behind, a real client.
    monkeypatch.setattr(memory_module, "_OPENSEARCH_CLIENT", None)


# --- the mode itself ---------------------------------------------------------------------

def test_local_is_a_mode():
    assert "local" in deployment_mode.MODES


def test_local_turns_persistent_memory_off(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "local")
    assert deployment_mode.is_local()
    assert deployment_mode.persistent_memory_allowed() is False


@pytest.mark.parametrize("mode", ["dev", "demo", "token"])
def test_every_other_mode_keeps_it_on(monkeypatch, mode):
    monkeypatch.setenv("AGENT_MODE", mode)
    assert deployment_mode.persistent_memory_allowed() is True


def test_an_unknown_mode_still_raises(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "locla")
    with pytest.raises(ValueError):
        deployment_mode.current_mode()


# --- the store is never opened -----------------------------------------------------------

def _forbid_building_a_client(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("a conversation-store client was constructed in local mode")
    monkeypatch.setattr(memory_module, "OpenSearch", boom)


def test_the_incident_configuration_cannot_open_the_store(monkeypatch):
    """The 2026-10-01 case: prod named explicitly, as the main checkout's .env does."""
    monkeypatch.setenv("AGENT_MODE", "local")
    monkeypatch.setenv("OPENSEARCH_NODE", PROD_NODE)
    _forbid_building_a_client(monkeypatch)
    with pytest.raises(memory_module.PersistentMemoryDisabled):
        memory_module._get_opensearch_client()


def test_the_recorded_workaround_cannot_open_the_dev_cluster_either(monkeypatch):
    """The second failure: explicit host blank, so the dev tier would have supplied one."""
    monkeypatch.setenv("AGENT_MODE", "local")
    monkeypatch.setenv("PLATFORM_TIER", "dev")
    monkeypatch.setenv("OPENSEARCH_NODE", "")
    _forbid_building_a_client(monkeypatch)
    with pytest.raises(memory_module.PersistentMemoryDisabled):
        memory_module._get_opensearch_client()


def test_a_client_cached_earlier_is_not_handed_out(monkeypatch):
    """The guard sits AHEAD of the cache. Behind it, any client built before the mode was read
    — by an import, a warm-up, an earlier request — would sail straight past it."""
    sentinel = object()
    monkeypatch.setattr(memory_module, "_OPENSEARCH_CLIENT", sentinel)
    monkeypatch.setenv("AGENT_MODE", "local")
    with pytest.raises(memory_module.PersistentMemoryDisabled):
        memory_module._get_opensearch_client()


def test_every_store_operation_refuses(monkeypatch):
    """Snapshots are written by the map UI DIRECTLY, outside the per-request flag — which is
    why the guard has to live in the store rather than only where the flag is read."""
    monkeypatch.setenv("AGENT_MODE", "local")
    monkeypatch.setenv("OPENSEARCH_NODE", PROD_NODE)
    _forbid_building_a_client(monkeypatch)
    with pytest.raises(memory_module.PersistentMemoryDisabled):
        memory_module.save_session_snapshot("mem-1", {"messages": []})
    with pytest.raises(memory_module.PersistentMemoryDisabled):
        memory_module.get_session_snapshot("mem-1")


# --- the request flag --------------------------------------------------------------------

def test_a_request_asking_for_memory_is_overridden(monkeypatch):
    """The map UI hard-codes use_persistent_memory: true, so the client cannot be the switch."""
    monkeypatch.setenv("AGENT_MODE", "local")
    with server.app.test_request_context("/agent/chat"):
        out = server._normalize_agent_chat_request(
            {"query": "hi", "usePersistentMemory": True, "use_persistent_memory": True})
    assert out["use_persistent_memory"] is False


def test_outside_local_mode_the_request_still_decides(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "dev")
    with server.app.test_request_context("/agent/chat"):
        assert server._normalize_agent_chat_request(
            {"query": "hi", "usePersistentMemory": True})["use_persistent_memory"] is True
        assert server._normalize_agent_chat_request(
            {"query": "hi", "usePersistentMemory": False})["use_persistent_memory"] is False


# --- the endpoints answer as OFF, not as broken -----------------------------------------

def _never_called(*_a, **_k):
    raise AssertionError("the conversation store was reached in local mode")


@pytest.fixture()
def local_server(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "local")
    monkeypatch.setenv("OPENSEARCH_NODE", PROD_NODE)
    for name in ("save_session_snapshot", "get_session_snapshot", "assert_memory_owner",
                 "get_turn_trace", "list_turn_traces", "list_memories"):
        monkeypatch.setattr(server, name, _never_called)
    with server.app.test_client() as client:
        yield client


def test_saving_a_conversation_is_declined_not_failed(local_server):
    res = local_server.put("/agent/conversations/mem-1", json={"messages": []})
    assert res.status_code == 200, "the client keeps its own copy; this is not an error for it"
    assert res.get_json() == {"stored": False, "reason": "persistent_memory_disabled"}


def test_reading_a_conversation_finds_nothing(local_server):
    res = local_server.get("/agent/conversations/mem-1")
    assert res.status_code == 404
    assert res.get_json()["reason"] == "persistent_memory_disabled"


def test_traces_are_not_recorded(local_server):
    res = local_server.get("/agent/conversations/mem-1/traces")
    assert res.status_code == 404
    assert res.get_json()["reason"] == "persistent_memory_disabled"


def test_the_history_list_is_empty(local_server):
    res = local_server.get("/agent/conversations")
    assert res.status_code == 200
    assert res.get_json() == {"conversations": []}


def test_ui_config_says_so(local_server):
    body = local_server.get("/agent/ui-config").get_json()
    assert body["mode"] == "local"
    assert body["persistent_memory"] is False


def test_ui_config_reports_memory_on_elsewhere(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "dev")
    with server.app.test_client() as client:
        assert client.get("/agent/ui-config").get_json()["persistent_memory"] is True


# --- knowledge-base search keeps reading -------------------------------------------------

def test_kb_search_still_builds_its_own_client(monkeypatch):
    """Memory and search shared OPENSEARCH_NODE, so 'no memory' used to mean 'no search' too.
    Local mode closes the memory store only: search builds its own client and is untouched."""
    monkeypatch.setenv("AGENT_MODE", "local")
    monkeypatch.setenv("OPENSEARCH_NODE", PROD_NODE)
    from rag_pipeline.search import agent_kb
    assert agent_kb._os_client() is not None


# --- download links stay on this machine --------------------------------------------------

def test_public_base_url_is_ignored_locally(monkeypatch):
    monkeypatch.setenv("AGENT_PUBLIC_BASE_URL", "https://agent.i-guide.io")
    monkeypatch.setenv("AGENT_MODE", "local")
    assert file_store._public_base_url() == ""
    monkeypatch.setenv("AGENT_MODE", "dev")
    assert file_store._public_base_url() == "https://agent.i-guide.io"


# --- the boot banner ---------------------------------------------------------------------

def test_the_banner_names_what_a_laptop_will_still_reach(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "local")
    monkeypatch.setenv("OPENSEARCH_NODE", PROD_NODE)
    monkeypatch.setenv("ANVILGPT_URL", "http://localhost:11434/v1/chat/completions")
    monkeypatch.setenv("AGENT_PUBLIC_BASE_URL", "https://agent.i-guide.io")
    text = "\n".join(deployment_mode.local_mode_report())
    assert "persistent memory OFF" in text
    assert "[REMOTE] OPENSEARCH_NODE=" in text and "READ ONLY" in text
    assert "[local ] ANVILGPT_URL=" in text
    assert "[ignored] AGENT_PUBLIC_BASE_URL" in text


def test_the_banner_never_prints_a_credential(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "local")
    monkeypatch.setenv("OPENSEARCH_NODE", "https://admin:hunter2@149.165.155.195:9200")
    text = "\n".join(deployment_mode.local_mode_report())
    assert "hunter2" not in text and "admin:" not in text
    assert "***@149.165.155.195" in text


def test_no_banner_outside_local_mode(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "dev")
    monkeypatch.setenv("OPENSEARCH_NODE", PROD_NODE)
    assert deployment_mode.local_mode_report() == []
