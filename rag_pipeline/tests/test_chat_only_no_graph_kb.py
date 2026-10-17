"""A chat-only client (the I-GUIDE platform) gets no graph search and no agent-KB search.

User, 2026-10-16: the platform's frontend reads its own element format, and hits from the Neo4j
graph search and the agent's extracted KB do not fit it. The tools are not bound, and the
backends refuse during a chat-only turn as well, because deterministic paths (the search sweep,
the KB join, popularity and by-id lookups) reach them without any tool call. The map UI keeps
both.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime import client_profile as cp  # noqa: E402

BLOCKED = ["neo4j_search", "neo4j_get_element_by_id", "neo4j_explore_related_nodes",
           "agent_kb_search", "get_kb_block", "kb_method_search", "get_method_contract"]


@pytest.fixture
def chat_only():
    token = cp.set_chat_only(True)
    yield
    cp.reset_chat_only(token)


def _bound_tools(monkeypatch, names):
    import langchain.agents as la
    from agent_runtime import executor_factory as ef

    seen = {}

    def fake_create_agent(*, tools, **_kw):
        seen["tools"] = [t.name for t in tools]
        return "agent"

    monkeypatch.setattr(la, "create_agent", fake_create_agent)
    ef.build_agent_executor(llm=object(),
                            preloaded_tools=[SimpleNamespace(name=n) for n in names])
    return set(seen["tools"])


def test_the_graph_and_kb_tools_are_on_the_blocked_list():
    # The list grew (web and Overpass, test_chat_only_platform_sources); these stay on it.
    assert set(BLOCKED) <= set(cp.CHAT_ONLY_BLOCKED_TOOLS)


def test_graph_and_kb_tools_are_not_bound_chat_only(monkeypatch, chat_only):
    bound = _bound_tools(monkeypatch, BLOCKED + ["keyword_search", "spatial_search"])
    assert not bound & set(BLOCKED)
    assert {"keyword_search", "spatial_search"} <= bound


def test_the_map_profile_keeps_them(monkeypatch):
    assert set(BLOCKED) <= _bound_tools(monkeypatch, BLOCKED)


# --- Neo4j: every query path refuses, and the cached driver is not poisoned -------------------

@pytest.mark.parametrize("module", ["rag_pipeline.search.neo4j", "rag_pipeline.search.agents"])
def test_no_cypher_runs_chat_only(monkeypatch, chat_only, module):
    import importlib

    mod = importlib.import_module(module)

    def no_driver(*_a, **_k):
        raise AssertionError("the Neo4j driver was reached in a chat-only turn")

    monkeypatch.setattr(mod, "_neo4j_driver", no_driver)
    assert mod._neo4j_run("MATCH (n) RETURN n", {}) == []


def test_a_chat_only_turn_does_not_switch_the_graph_off_for_the_next_turn(monkeypatch):
    """The driver is created once and cached; a chat-only check there would stick."""
    from rag_pipeline.search import neo4j as n4

    calls = []

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def run(self, cypher, **params):
            calls.append(cypher)
            return [{"n": 1}]

    monkeypatch.setattr(n4, "_neo4j_driver", lambda: SimpleNamespace(session=lambda **k: _Session()))
    monkeypatch.setattr(n4, "_neo4j_db", lambda: None)
    token = cp.set_chat_only(True)
    try:
        assert n4._neo4j_run("RETURN 1", {}) == []
    finally:
        cp.reset_chat_only(token)
    assert n4._neo4j_run("RETURN 1", {}) == [{"n": 1}]
    assert calls == ["RETURN 1"]


def test_the_text2cypher_search_is_skipped_before_its_model_call(monkeypatch, chat_only):
    from rag_pipeline.search import agents

    monkeypatch.setattr(agents, "_neo4j_driver", lambda: (_ for _ in ()).throw(AssertionError()))
    assert agents.get_neo4j_agent_results("most popular notebooks") == []


# --- agent KB: search, block reads and the join ------------------------------------------------

def test_agent_kb_search_returns_nothing_chat_only(chat_only):
    from rag_pipeline.search.agent_kb import agent_kb_search

    out = agent_kb_search("chicago crime")
    assert out["count"] == 0 and out["documents"] == []
    assert "chat-only" in out["note"]


def test_get_kb_block_is_not_found_chat_only(chat_only):
    from rag_pipeline.search.agent_kb import get_kb_block

    out = get_kb_block("abc::block::0")
    assert out["found"] is False and "chat-only" in out["note"]


def test_the_kb_join_leaves_the_results_alone_chat_only(chat_only):
    from rag_pipeline.search.agent_kb import attach_kb_to_documents

    docs = [{"id": "e1", "title": "A notebook"}]
    out = attach_kb_to_documents(docs)
    assert out["documents"] == docs and out["attached"] == 0 and out["actionable"] == []


def test_the_search_sweep_and_code_peers_treat_the_kb_as_off(monkeypatch, chat_only):
    from agent_runtime.supervisor import graph as sg

    monkeypatch.delenv("AGENT_ABLATE_KB", raising=False)
    assert sg.kb_ablated()


def test_the_map_profile_has_the_kb(monkeypatch):
    from agent_runtime.supervisor import graph as sg

    monkeypatch.delenv("AGENT_ABLATE_KB", raising=False)
    assert not sg.kb_ablated()
    assert cp.graph_search_allowed() and cp.agent_kb_allowed()
