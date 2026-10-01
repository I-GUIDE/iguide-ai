"""enabledSearchMethods validation.

A name that did not match a tool EXACTLY used to be dropped in silence, leaving the agent with no
retrieval tools and an unexplained "no evidence" answer at HTTP 200.
"""

from __future__ import annotations

import pytest

from agent_runtime.search_methods import KNOWN_SEARCH_METHODS, normalize_search_methods


def test_none_means_all_methods():
    assert normalize_search_methods(None) is None


def test_valid_names_pass_through_deduped_and_ordered():
    assert normalize_search_methods(["keyword_search", "semantic_search"]) == \
        ["keyword_search", "semantic_search"]
    assert normalize_search_methods(["semantic", "semantic_search", "keyword"]) == \
        ["semantic_search", "keyword_search"]


def test_case_and_short_forms_are_accepted():
    assert normalize_search_methods(["Keyword_Search", "NEO4J", "OpenGeoData", "spatial"]) == \
        ["keyword_search", "neo4j_search", "opengeodata_search", "spatial_search"]
    assert normalize_search_methods("keyword_search, semantic") == \
        ["keyword_search", "semantic_search"]


def test_unknown_name_is_rejected_with_a_helpful_message():
    with pytest.raises(ValueError) as err:
        normalize_search_methods(["keyword_search", "nonsense_method"])
    msg = str(err.value)
    assert "nonsense_method" in msg                       # names the offender
    for method in ("keyword_search", "opengeodata_search"):
        assert method in msg                              # lists valid options


def test_empty_list_means_unspecified_not_no_retrieval():
    """[] previously stripped EVERY retrieval tool; that is a client bug, not an intent."""
    assert normalize_search_methods([]) is None
    assert normalize_search_methods(["", "  "]) is None
    assert normalize_search_methods("") is None


def test_bad_shape_is_rejected():
    with pytest.raises(ValueError):
        normalize_search_methods({"keyword_search": True})
    with pytest.raises(ValueError):
        normalize_search_methods(42)


def test_known_methods_match_the_actual_registry():
    """The validator's list must not drift from the tools that really exist."""
    from agent_runtime.langchain_granular_tools import make_langchain_granular_tools
    registry = {getattr(t, "name", "") for t in make_langchain_granular_tools(include_file_tools=False)}
    assert set(KNOWN_SEARCH_METHODS) <= registry


def test_api_layer_returns_400_for_an_unknown_method(monkeypatch):
    import api.server as srv
    monkeypatch.delenv("AGENT_CHAT_API_KEY", raising=False)
    client = srv.app.test_client()
    resp = client.post("/agent/chat", json={"userQuery": "hi", "enabledSearchMethods": ["keywrd"]})
    assert resp.status_code == 400
    assert "keywrd" in resp.get_json()["error"]
