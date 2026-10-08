"""The peers that measure can also find the features they measure to.

Live, 2026-10-08 19:42 UTC on agent.i-guide.io: "Which schools are within 1 mile of [a drawn
box]?" The supervisor sent it to analyze. The analyze peer's 43 bound tools held no tool that
finds a feature by kind in an area: `overpass_search` existed, but only in the search peer's
toolset. So the peer did the one thing it could, geocoding school names it remembered (Hyde Park
schools, 15 km away, first) and then street intersections, 27 calls, until the recursion limit.
The code peer, which had no OSM tool either, found the Chicago Public Schools file by web search
and answered from that, so the answer covered one school system and missed the private,
Catholic and suburban schools OSM has in the same area.

Feature lookup produces its own input, like admin_boundary and dem_for_region, so it is bound
whether or not anything is uploaded, in both measuring peers.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_runtime import executor_factory as ef
from agent_runtime.supervisor import graph as g


class _Stop(Exception):
    pass


def _bound_names(monkeypatch, peer_fn):
    seen = {}

    def capture(**kwargs):
        seen["tools"] = [getattr(t, "name", "") for t in kwargs.get("preloaded_tools") or []]
        raise _Stop()

    monkeypatch.setattr(ef, "build_agent_executor", capture)
    with pytest.raises(_Stop):
        peer_fn("which schools are within 1 mile of this box?", [], {"thread_id": "t"})
    return seen["tools"]


def test_the_analyze_peer_can_find_features_with_nothing_uploaded(monkeypatch):
    names = _bound_names(monkeypatch, g.default_analyze_fn(llm=object(), include_mcp_tools=False))
    assert "overpass_search" in names


def test_the_code_peer_can_find_features_too(monkeypatch):
    monkeypatch.delenv("AGENT_CODE_PEER", raising=False)
    names = _bound_names(monkeypatch, g.default_code_fn(llm=object()))
    assert "overpass_search" in names


def _fake_overpass(n):
    def fake(feature, place=None, bbox=None, limit=60):
        feats = [{"osm_type": "node", "osm_id": i, "name": f"School {i}", "lat": 41.79,
                  "lon": -87.75, "feature_type": "amenity=school",
                  "tags": {"amenity": "school", "name": f"School {i}", "religion": "christian"},
                  "geometry": {"type": "Point", "coordinates": [-87.75, 41.79]}}
                 for i in range(n)]
        return {"query": {"feature": feature, "osm_filter": "amenity=school", "place": place,
                          "bbox": [-87.78, 41.77, -87.71, 41.82]},
                "count": len(feats), "features": feats[:limit]}
    return fake


@pytest.fixture
def store(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path))
    return tmp_path


def _tool():
    from agent_runtime.langchain_granular_tools import make_langchain_osm_tools

    (tool,) = make_langchain_osm_tools()
    return tool


def test_the_features_arrive_as_a_file_the_measuring_tools_can_read(monkeypatch, store):
    import rag_pipeline.search.overpass as ov
    from agent_runtime.file_store import resolve_file_id

    monkeypatch.setattr(ov, "overpass_search", _fake_overpass(3))
    out = json.loads(_tool().invoke({"feature": "school", "bbox": "-87.78,41.77,-87.71,41.82"}))
    assert out["count"] == 3 and out["file_id"]
    path = resolve_file_id(out["file_id"])
    fc = json.loads(Path(path).read_text(encoding="utf-8"))
    assert fc["type"] == "FeatureCollection" and len(fc["features"]) == 3
    props = fc["features"][0]["properties"]
    # The tags travel as columns, so a list can say which schools are religious or private.
    assert props["name"] == "School 0" and props["religion"] == "christian"
    assert props["osm_id"] == 0


def test_the_result_states_what_the_list_is_and_is_not(monkeypatch, store):
    import rag_pipeline.search.overpass as ov

    monkeypatch.setattr(ov, "overpass_search", _fake_overpass(3))
    out = json.loads(_tool().invoke({"feature": "school", "bbox": "-87.78,41.77,-87.71,41.82"}))
    statement = out["source_statement"]
    assert "OpenStreetMap" in statement and "amenity=school" in statement
    assert "not an official" in statement


def test_a_result_at_the_limit_says_it_may_be_cut(monkeypatch, store):
    import rag_pipeline.search.overpass as ov

    monkeypatch.setattr(ov, "overpass_search", _fake_overpass(10))
    out = json.loads(_tool().invoke({"feature": "school", "bbox": "-87.78,41.77,-87.71,41.82",
                                     "limit": 10}))
    assert out["count"] == 10 and "limit" in out.get("note", "")


def test_a_failed_query_writes_no_file(monkeypatch, store):
    import rag_pipeline.search.overpass as ov

    monkeypatch.setattr(ov, "overpass_search", lambda *a, **k: {
        "error": "overpass_failed", "message": "504", "features": [], "count": 0})
    out = json.loads(_tool().invoke({"feature": "school", "bbox": "-87.78,41.77,-87.71,41.82"}))
    assert out["error"] == "overpass_failed" and "file_id" not in out


def test_the_decider_is_told_analyze_can_find_features():
    from agent_runtime.capability_registry import describe

    line = describe("analyze").lower()
    assert "openstreetmap" in line and "school" in line
