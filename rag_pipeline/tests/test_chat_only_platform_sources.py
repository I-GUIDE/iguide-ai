"""A chat-only client (the I-GUIDE platform) searches only what it can show (user, 2026-10-16).

The platform draws every document in the agent's evidence as one of its knowledge elements. Web
pages and OpenGeoData records arrived as "undefined" cards with a broken thumbnail. So a chat-only
turn searches keyword, semantic and spatial (the platform's OpenSearch index) and OpenGeoData
(returned separately as `opengeodata_results`), and the evidence it returns holds only the
platform's own element types. The map UI is unchanged.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime import client_profile as cp  # noqa: E402

FOUR = ["keyword_search", "semantic_search", "spatial_search", "opengeodata_search"]


@pytest.fixture
def chat_only():
    token = cp.set_chat_only(True)
    yield
    cp.reset_chat_only(token)


# --- which search methods ----------------------------------------------------------------------

def test_unspecified_methods_become_the_four(chat_only):
    assert cp.chat_only_search_methods(None) == FOUR


def test_requested_methods_are_cut_to_the_four(chat_only):
    asked = ["web_search", "keyword_search", "overpass_search", "spatial_search", "neo4j_search"]
    assert cp.chat_only_search_methods(asked) == ["keyword_search", "spatial_search"]


def test_a_request_with_none_of_them_still_searches_the_platform(chat_only):
    assert cp.chat_only_search_methods(["web_search", "web_fetch"]) == FOUR


def test_the_map_profile_keeps_what_it_asked_for():
    assert cp.chat_only_search_methods(None) is None
    assert cp.chat_only_search_methods(["web_search"]) == ["web_search"]


def test_the_chat_service_applies_it(chat_only):
    from agent_runtime.agent_chat_service import _normalize_enabled_search_methods

    assert _normalize_enabled_search_methods(None) == FOUR
    assert _normalize_enabled_search_methods(["web_search", "semantic_search"]) == ["semantic_search"]


def test_the_web_fallback_does_not_fire_for_the_platform(monkeypatch, chat_only):
    from agent_runtime.supervisor import graph as sg

    def no_web(*_a, **_k):
        raise AssertionError("the web was reached on a chat-only turn")

    import rag_pipeline.search.web as web

    for name in ("web_search", "get_web_search_results", "search_web"):
        if hasattr(web, name):
            monkeypatch.setattr(web, name, no_web)
    assert sg._web_fallback_evidence("anything at all", cp.chat_only_search_methods(None)) == []


# --- which tools --------------------------------------------------------------------------------

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


def test_web_and_overpass_tools_are_not_bound_chat_only(monkeypatch, chat_only):
    bound = _bound_tools(monkeypatch, FOUR + ["web_search", "web_fetch", "overpass_search"])
    assert set(FOUR) <= bound
    assert not bound & {"web_search", "web_fetch", "overpass_search"}


def test_the_map_profile_keeps_web_and_overpass(monkeypatch):
    names = ["web_search", "web_fetch", "overpass_search"]
    assert set(names) <= _bound_tools(monkeypatch, names)


# --- what the platform is sent ---------------------------------------------------------------

def _doc(etype, title, nested=False):
    d = {"title": title, "element_type": etype, "doc_id": title}
    return {"document": d, "score": 1.0} if nested else d


EVIDENCE = [
    _doc("notebook", "nb"), _doc("dataset", "ds", nested=True), _doc("web", "github page"),
    _doc("opengeodata", "datacite record"), _doc("osm_feature", "a cafe"),
    _doc("agent_kb", "kb row"), _doc("resource", "untyped"), {"title": "no type"},
    {"title": "oer by resource-type", "resource-type": "OER"},
]


def test_only_platform_element_types_survive():
    kept = [cp._element_type(d) for d in cp.platform_elements_only(EVIDENCE)]
    assert kept == ["notebook", "dataset", "oer"]


def _payload():
    return {"answer": "a", "elements": copy.deepcopy(EVIDENCE),
            "opengeodata_results": [{"title": "datacite record", "element_type": "opengeodata"}],
            "agent_result": {"orchestration_result": {"evidence": copy.deepcopy(EVIDENCE),
                                                      "final_answer": "a"}}}


def test_the_platform_gets_only_its_elements(chat_only):
    import api.server as server

    payload = _payload()
    before = copy.deepcopy(payload)
    out = server._format_agent_chat_result(payload)
    ev = out["agent_result"]["orchestration_result"]["evidence"]
    assert [cp._element_type(d) for d in ev] == ["notebook", "dataset", "oer"]
    assert [cp._element_type(d) for d in out["elements"]] == ["notebook", "dataset", "oer"]
    assert out["count"] == 3
    # OpenGeoData still reaches the platform, in its own field.
    assert out["opengeodata_results"] == before["opengeodata_results"]
    assert out["agent_result"]["orchestration_result"]["final_answer"] == "a"
    # The turn's own payload (trace, history) is not narrowed.
    assert payload == before


def test_the_map_ui_gets_all_of_it():
    import api.server as server

    out = server._format_agent_chat_result(_payload())
    assert len(out["agent_result"]["orchestration_result"]["evidence"]) == len(EVIDENCE)
    assert len(out["elements"]) == len(EVIDENCE)


# --- the OpenSearch searches still run; the search agents, the agent KB and the web do not -------

def test_the_sweep_still_runs_keyword_semantic_and_spatial(monkeypatch, chat_only):
    """The three searches over the platform's OpenSearch index are what a chat-only turn keeps."""
    import rag_pipeline.search.agent_kb as akb
    import rag_pipeline.search.keyword as kw
    import rag_pipeline.search.semantic as sem
    import rag_pipeline.search.spatial as sp
    from agent_runtime.supervisor import graph as sg

    called = []

    def hit(name):
        def fn(query, size=10, **_k):
            called.append(name)
            return [{"_id": f"{name}-1", "_source": {"title": name, "resource-type": "notebook"}}]
        return fn

    monkeypatch.setattr(kw, "get_keyword_search_results", hit("keyword"))
    monkeypatch.setattr(sem, "semantic_search", hit("semantic"))
    monkeypatch.setattr(sp, "get_spatial_search_results", hit("spatial"))
    monkeypatch.setattr(sg, "_mentions_place", lambda q: True)

    def no_kb(*_a, **_k):
        raise AssertionError("the agent KB was searched on a chat-only turn")

    monkeypatch.setattr(akb, "agent_kb_search", no_kb)
    docs = sg._direct_search_sweep("notebooks about Chicago", cp.chat_only_search_methods(None))
    assert {"keyword", "semantic", "spatial"} <= set(called)
    assert docs, "the platform hits came back"


def test_the_search_agents_do_not_run_chat_only(monkeypatch, chat_only):
    from rag_pipeline.search import agents

    def boom(*_a, **_k):
        raise AssertionError("a search agent ran on a chat-only turn")

    monkeypatch.setattr(agents, "_os_client", boom)
    monkeypatch.setattr(agents, "_neo4j_driver", boom)
    assert agents.get_opensearch_agent_results("most cited datasets") == []
    assert agents.get_neo4j_agent_results("most popular notebooks") == []


def test_the_map_profile_still_has_the_search_agents():
    assert cp.search_agents_allowed()
