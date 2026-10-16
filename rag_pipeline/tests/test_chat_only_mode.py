"""A chat-only client (the I-GUIDE platform) gets no map machinery; the map UI keeps all of it.

The platform calls /agent/chat/stream server-to-server and shows only the answer and the sources.
It dropped every map_layer event, and its users were told about layers they could not see.
``agent_runtime/client_profile.py`` turns the map parts off per request: the two map-only tools
are not bound, no map_layer event is sent, the forced map retry and the "already on your map"
note are skipped, the answer writer is told there is no map, and an answer that still talks
about one gets a marked correction.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime import client_profile as cp  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("AGENT_CHAT_ONLY_KEYS", raising=False)
    monkeypatch.delenv("AGENT_CHAT_ONLY_SOURCES", raising=False)


@pytest.fixture
def chat_only():
    token = cp.set_chat_only(True)
    yield
    cp.reset_chat_only(token)


# --- who is chat-only -------------------------------------------------------------------------

def test_the_body_flag_turns_it_on():
    assert cp.wants_chat_only({"chatOnly": True})
    assert cp.wants_chat_only({"chat_only": "true"})
    assert not cp.wants_chat_only({"chatOnly": False})
    assert not cp.wants_chat_only({})


def test_a_listed_key_turns_it_on(monkeypatch):
    monkeypatch.setenv("AGENT_CHAT_ONLY_KEYS", "platform-key, other")
    assert cp.wants_chat_only({}, presented_key="platform-key")
    assert not cp.wants_chat_only({}, presented_key="harness-key")


def test_a_listed_source_turns_it_on(monkeypatch):
    monkeypatch.setenv("AGENT_CHAT_ONLY_SOURCES", "149.165.154.167,149.165.155.135")
    assert cp.wants_chat_only({}, source_addresses=["149.165.154.167", "172.19.0.1"])
    assert not cp.wants_chat_only({}, source_addresses=["10.0.0.5", "172.19.0.1"])


def test_nothing_set_means_the_map_profile():
    assert cp.chat_only() is False


# --- what a chat-only turn turns off ----------------------------------------------------------

def _tool(name):
    return SimpleNamespace(name=name)


def test_the_map_only_tools_are_not_bound(monkeypatch, chat_only):
    import langchain.agents as la
    from agent_runtime import executor_factory as ef

    seen = {}

    def fake_create_agent(*, tools, **_kw):
        seen["tools"] = [t.name for t in tools]
        return "agent"

    monkeypatch.setattr(la, "create_agent", fake_create_agent)
    ef.build_agent_executor(llm=object(), preloaded_tools=[
        _tool("add_map_layer"), _tool("add_raster_layer"), _tool("admin_boundary"),
        _tool("execute_code")])
    assert "add_map_layer" not in seen["tools"] and "add_raster_layer" not in seen["tools"]
    # (Overpass was the kept example here; since 2026-10-16 a chat-only turn gets no Overpass
    # either: test_chat_only_platform_sources.)
    assert {"admin_boundary", "execute_code"} <= set(seen["tools"])


def test_the_map_profile_still_binds_them(monkeypatch):
    import langchain.agents as la
    from agent_runtime import executor_factory as ef

    seen = {}
    monkeypatch.setattr(la, "create_agent",
                        lambda *, tools, **_kw: seen.setdefault("tools", [t.name for t in tools]))
    ef.build_agent_executor(llm=object(), preloaded_tools=[_tool("add_map_layer")])
    assert "add_map_layer" in seen["tools"]


def _emitted_events(monkeypatch, on: bool):
    from agent_runtime import map_layers, streaming_trace as st

    monkeypatch.setattr(map_layers, "build_map_layers",
                        lambda name, output, **_kw: [{"kind": "map_layer", "id": "x"}])
    events = []
    token = cp.set_chat_only(on)
    try:
        with st.trace_context(lambda e: events.append(e), agent_dev=True):
            handler = st.active_callback_handler()
            handler.on_tool_start({"name": "overpass_search"}, "{}", run_id="r1")
            handler.on_tool_end('{"ok": true}', run_id="r1")
    finally:
        cp.reset_chat_only(token)
    return [e.get("event") for e in events]


def test_no_map_layer_event_is_sent_chat_only(monkeypatch):
    assert "map_layer" not in _emitted_events(monkeypatch, on=True)


def test_the_map_profile_still_sends_it(monkeypatch):
    assert "map_layer" in _emitted_events(monkeypatch, on=False)


def test_an_answer_that_talks_about_a_map_is_corrected(chat_only):
    from agent_runtime.supervisor.graph import _correct_artifact_claims

    out = _correct_artifact_claims("The tracts are shown on the map; zoom in to see them.")
    assert cp.CHAT_ONLY_CORRECTION in out
    assert "already on your interactive map" not in out


def test_an_answer_with_no_map_talk_is_untouched(chat_only):
    from agent_runtime.supervisor.graph import _correct_artifact_claims

    text = "There are 48 census tracts in Champaign County. Download: tracts.geojson"
    assert _correct_artifact_claims(text) == text


def test_the_map_profile_does_not_get_the_chat_only_correction():
    from agent_runtime.supervisor.graph import _correct_artifact_claims

    assert cp.CHAT_ONLY_CORRECTION not in _correct_artifact_claims("The layer is on the map.")


def _synthesis_prompt():
    from agent_runtime.supervisor.graph import default_synthesize_fn

    seen = {}

    class LLM:
        def invoke(self, prompt):
            seen["prompt"] = prompt
            return "ok"

    default_synthesize_fn(LLM())("q", [], None, None)
    return seen["prompt"]


def test_the_answer_writer_is_told_there_is_no_map(chat_only):
    assert cp.CHAT_ONLY_NOTE in _synthesis_prompt()


def test_the_map_profile_answer_writer_is_not():
    assert cp.CHAT_ONLY_NOTE not in _synthesis_prompt()


# --- the API edge -----------------------------------------------------------------------------

def test_the_request_edge_reads_flag_key_and_source(monkeypatch):
    import api.server as server

    monkeypatch.setenv("AGENT_CHAT_ONLY_SOURCES", "149.165.154.167")
    with server.app.test_request_context("/agent/chat/stream", method="POST",
                                         headers={"X-Real-IP": "149.165.154.167"}):
        assert server._request_chat_only({})
    with server.app.test_request_context("/agent/chat/stream", method="POST",
                                         headers={"X-Real-IP": "128.174.1.1"}):
        assert not server._request_chat_only({})
        assert server._request_chat_only({"chatOnly": True})


def test_the_result_says_which_profile_answered(chat_only):
    import api.server as server

    assert server._format_agent_chat_result({"answer": "a"})["chatOnly"] is True


def test_the_map_profile_result_has_no_chat_only_flag():
    import api.server as server

    assert "chatOnly" not in server._format_agent_chat_result({"answer": "a"})


def test_a_chat_only_peer_is_not_told_about_the_live_map(monkeypatch, chat_only):
    import langchain.agents as la
    from agent_runtime import executor_factory as ef
    from agent_runtime.supervisor.prompts import (ANALYSIS_WORKFLOW_PROMPT, CHAT_ONLY_DISPLAY_RULE,
                                                  VISUALIZATION_ROUTES_RULE)

    seen = {}
    monkeypatch.setattr(la, "create_agent",
                        lambda *, system_prompt, **_kw: seen.setdefault("prompt", system_prompt))
    ef.build_agent_executor(llm=object(), preloaded_tools=[],
                            system_prompt_override=ANALYSIS_WORKFLOW_PROMPT)
    assert CHAT_ONLY_DISPLAY_RULE in seen["prompt"]
    assert VISUALIZATION_ROUTES_RULE not in seen["prompt"]


def test_a_map_profile_peer_keeps_the_live_map_rule(monkeypatch):
    import langchain.agents as la
    from agent_runtime import executor_factory as ef
    from agent_runtime.supervisor.prompts import ANALYSIS_WORKFLOW_PROMPT, VISUALIZATION_ROUTES_RULE

    seen = {}
    monkeypatch.setattr(la, "create_agent",
                        lambda *, system_prompt, **_kw: seen.setdefault("prompt", system_prompt))
    ef.build_agent_executor(llm=object(), preloaded_tools=[],
                            system_prompt_override=ANALYSIS_WORKFLOW_PROMPT)
    assert VISUALIZATION_ROUTES_RULE in seen["prompt"]
