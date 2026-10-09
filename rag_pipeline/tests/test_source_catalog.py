"""Stage 45: every data source is catalogued with what it covers and leaves out, and the answer
names the sources it used from the record.

docs/design-review-2026-10.md, flaw 6.
"""
from __future__ import annotations

import json

import pytest

from agent_runtime import source_catalog as sc
from agent_runtime.supervisor import graph as g


def test_every_entry_declares_its_coverage_and_licence():
    ids = [s.id for s in sc.CATALOG]
    assert len(ids) == len(set(ids))
    for s in sc.CATALOG:
        assert s.covers and s.excludes and s.extent and s.licence, s.id


def test_a_tool_belongs_to_one_source():
    tools = [t for s in sc.CATALOG for t in s.tools]
    assert len(tools) == len(set(tools))


def test_every_host_the_agent_may_fetch_from_is_catalogued():
    """A structural guard: a new allowlisted host without a catalogue entry would reach an
    answer with no statement of what it covers."""
    from agent_runtime.public_data_tools import DEFAULT_HOSTS

    catalogued = {h for s in sc.CATALOG for h in s.hosts}
    missing = [h for h, _ in DEFAULT_HOSTS if h not in catalogued]
    assert not missing, missing


def test_the_tools_that_fetch_external_data_are_catalogued():
    for tool in ("admin_boundary", "overpass_search", "geocode_places", "dem_for_region",
                 "embed_region", "embed_zones"):
        assert tool in sc._BY_TOOL, tool


def test_an_extent_is_checked_from_the_catalogue():
    mombasa = [39.6, -4.1, 39.7, -4.0]
    chicago = [-87.7, 41.8, -87.6, 41.9]
    assert sc.outside_extent("dem_for_region", mombasa).id == "usgs_3dep"
    assert sc.outside_extent("dem_for_region", chicago) is None
    assert sc.outside_extent("overpass_search", mombasa) is None     # global


def test_dem_for_region_refuses_outside_its_source_before_fetching(monkeypatch):
    from agent_runtime import terrain_tools as tt

    def boom(*a, **k):
        raise AssertionError("fetched a region its source does not cover")

    monkeypatch.setattr(tt, "_fetch_dem", boom)
    tool = next(t for t in tt.make_terrain_tools() if t.name == "dem_for_region")
    out = json.loads(tool.invoke({"bbox": [39.6, -4.1, 39.7, -4.0]}))
    assert out["ok"] is False and "USGS 3DEP" in out["error"]


def test_an_upload_is_named_by_its_filename(monkeypatch):
    monkeypatch.setattr("agent_runtime.file_store.get_file_record",
                        lambda fid: {"filename": "schools.geojson", "kind": "upload"})
    out = sc.source_of("execute_code", {"code": "...", "input_files": ["file_9aafc5a4f0cf"]}, "{}")
    assert out == [("schools.geojson (your upload)", ["schools.geojson"])]


def test_an_agent_output_is_not_a_source(monkeypatch):
    monkeypatch.setattr("agent_runtime.file_store.get_file_record",
                        lambda fid: {"filename": "buffer.geojson", "kind": "output"})
    assert sc.source_of("execute_code", {"input_files": ["file_x"]}, "{}") == []


def test_a_failed_result_is_not_a_source():
    pairs = [({"name": "overpass_search", "args": {}},
              {"name": "overpass_search", "tool_call_id": "a",
               "content": json.dumps({"error": "overpass_failed", "count": 0})})]
    assert sc.sources_of(pairs) == []


def test_every_peer_step_is_told_what_each_source_leaves_out():
    brief = g._turn_brief({"query": "schools within 1 mile"})
    assert "OpenStreetMap (via Overpass)" in brief and "leaves out" in brief
    assert "City of Chicago Data Portal" in brief and "not run by CPS" in brief
