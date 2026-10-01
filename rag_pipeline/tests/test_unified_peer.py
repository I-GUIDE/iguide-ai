"""The search+analyze merge, behind AGENT_UNIFIED_PEER.

The peers never ran concurrently — decide() returns one action per step — and every hard
failure came from state split across them. This collapses them into one agent with one context
and one tool list, leaving the supervisor as a verifier plus the router for the code peer.
"""

from __future__ import annotations

import json

import pytest

from agent_runtime.supervisor import graph as g


@pytest.fixture
def unified(monkeypatch):
    monkeypatch.setenv(g.UNIFIED_PEER_ENV, "1")


@pytest.fixture
def peered(monkeypatch):
    monkeypatch.delenv(g.UNIFIED_PEER_ENV, raising=False)


def test_the_flag_is_off_by_default(peered):
    assert g.unified_peer_enabled() is False


def test_the_supervisor_stops_offering_search_when_merged(unified):
    """There is no separate retrieval peer to route to — routing there would duplicate what
    the one agent already does, and split the context again."""
    # `search` is gone from the menu in every state — that is the whole point. (Which of the
    # remaining actions is offered still varies: analyze is withheld as an unproductive
    # back-to-back repeat, and done is withheld before anything has run.)
    for state in (
        {"evidence": [], "step": 0, "search_attempts": 0, "actions": []},
        {"evidence": [], "step": 1, "search_attempts": 0,
         "actions": ["analyze"], "analysis_results": {"summary": "x"}},
        {"evidence": [{"title": "d"}], "step": 2, "search_attempts": 1, "actions": ["analyze"]},
    ):
        assert "search" not in g._available_actions(state), state


def test_search_is_still_offered_when_peered(peered):
    state = {"evidence": [], "step": 0, "search_attempts": 0}
    assert "search" in g._available_actions(state)


# --- evidence, by tool name --------------------------------------------------

def test_only_retrieval_tools_become_evidence():
    """extract_documents_from_search_evidence keys on results/items/hits for ANY tool, so in a
    merged agent geocode_places({"results": [...]}) would silently become a cited document."""
    artifacts = {"tool_results": [
        {"name": "keyword_search",
         "content": json.dumps({"results": [{"title": "Flood risk dataset", "url": "u1"}]})},
        {"name": "geocode_places",
         "content": json.dumps({"results": [{"name": "Urbana", "lat": 40.1}]})},
        {"name": "overpass_search",
         "content": json.dumps({"items": [{"name": "Boneyard Creek"}]})},
        {"name": "admin_boundary",
         "content": json.dumps({"matched": [{"geoid": "17019"}], "results": [1]})},
    ]}
    docs = g._evidence_from_artifacts(artifacts)
    blob = json.dumps(docs, default=str)
    assert "Flood risk dataset" in blob
    assert "Urbana" not in blob and "Boneyard" not in blob and "17019" not in blob


def test_no_retrieval_tools_means_no_evidence():
    assert g._evidence_from_artifacts({"tool_results": [
        {"name": "embed_region", "content": json.dumps({"results": [1, 2]})}]}) == []
    assert g._evidence_from_artifacts({}) == []


def test_a_broken_harvest_never_fails_the_run(monkeypatch):
    import agent_runtime.supervisor.evidence_subgraph as es

    monkeypatch.setattr(es, "extract_documents_from_search_evidence",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert g._evidence_from_artifacts({"tool_results": [
        {"name": "keyword_search", "content": "{}"}]}) == []


# --- the merged tool list ----------------------------------------------------

def test_duplicate_tool_names_are_collapsed():
    """Naive concatenation yields 102 entries with 14 duplicate NAMES in the deployed config.
    ToolNode silently keeps the last; bind_tools ships all 102 to the provider."""
    class T:
        def __init__(self, name): self.name = name

    out = g._dedup_tools([T("a"), T("b"), T("a"), T("c"), T("b")])
    assert [t.name for t in out] == ["a", "b", "c"]


def test_the_state_keys_are_unchanged_by_the_merge():
    """evidence_quality.py and runtime_utils.py read analysis_results/evidence BY NAME; a
    rename degrades silently instead of raising. The merge changes who fills them, not
    what they are called."""
    import inspect

    src = inspect.getsource(g)
    assert '"analysis_results": clean' in src
    assert 'update["evidence"] = merged' in src


def test_the_merged_node_keeps_the_search_counters_alive():
    """_search_exhausted / refinement / the route trace all read these; a merged node that
    stops writing them silently changes routing behaviour."""
    import inspect

    src = inspect.getsource(g)
    for key in ("search_attempts", "searched_queries", "search_empty_streak"):
        assert f'update["{key}"]' in src, key


def test_done_is_not_legal_before_anything_has_run(unified):
    """MEASURED REGRESSION: with search removed from the menu, "Find flood risk datasets on
    I-GUIDE" went straight to done at step 0 and answered "I couldn't find any supporting
    material" without ever retrieving. In the peered shape `search` was the obvious opening
    move and carried that cue implicitly; merging deleted the cue along with the peer."""
    fresh = {"unified_peer": True, "evidence": [], "step": 0, "actions": []}
    assert "done" not in g._available_actions(fresh)
    assert "analyze" in g._available_actions(fresh)


def test_done_becomes_legal_once_a_pass_has_happened(unified):
    after = {"unified_peer": True, "evidence": [], "step": 1,
             "actions": ["analyze"], "analysis_results": {"summary": "x"}}
    assert "done" in g._available_actions(after)


def test_the_guard_does_not_touch_the_peered_shape(peered):
    fresh = {"unified_peer": False, "evidence": [], "step": 0, "actions": []}
    assert g._available_actions(fresh) == ["search", "analyze", "code", "done"]
