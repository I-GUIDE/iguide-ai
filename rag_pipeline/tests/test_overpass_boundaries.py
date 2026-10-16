"""overpass_search returns a named boundary as one polygon with its area (stage 60).

Live, 2026-10-14: "Is Basco larger in area than 1st District of Palawan?" (MultiGlobeQA). The
query asked for nodes and ways only, and a boundary in OSM is a RELATION whose member ways
carry the outline in pieces. So the agent got 17 and 158 unnamed line segments, could measure
nothing, and answered from web pages. Offline: ``requests.post`` is patched.
"""
from __future__ import annotations

import json

import pytest
from pyproj import Geod
from shapely.geometry import box, shape

from rag_pipeline.search import overpass as ov


class _Resp:
    def __init__(self, payload, status=200):
        self.status_code, self._payload = status, payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"{self.status_code}")

    def json(self):
        return self._payload


def _way(coords, role="outer"):
    return {"type": "way", "role": role, "geometry": [{"lon": x, "lat": y} for x, y in coords]}


def _boundary(rel_id=1, name="Basco", members=None, kind="boundary"):
    return {"type": "relation", "id": rel_id,
            "tags": {"type": kind, "boundary": "administrative", "name": name, "admin_level": "6"},
            "members": members if members is not None else [
                # a 0.1° square, its outline split over two ways as OSM does
                _way([(0.0, 0.0), (0.1, 0.0), (0.1, 0.1)]),
                _way([(0.1, 0.1), (0.0, 0.1), (0.0, 0.0)]),
                # an enclave
                _way([(0.04, 0.04), (0.06, 0.04), (0.06, 0.06), (0.04, 0.06), (0.04, 0.04)],
                     role="inner"),
            ]}


def _patch(monkeypatch, responses):
    """Each post returns the next response; the queries sent are recorded."""
    sent = []

    def fake_post(url, data=None, headers=None, timeout=None):
        sent.append(data["data"])
        return responses.pop(0) if responses else _Resp({"elements": []})

    monkeypatch.setattr(ov.requests, "post", fake_post)
    monkeypatch.setattr(ov, "OVERPASS_ENDPOINTS", ["https://a/api", "https://b/api"])
    return sent


def test_a_boundary_relation_becomes_one_named_polygon_with_its_hole_and_area():
    feat = ov._element_to_feature(_boundary())
    assert feat["osm_type"] == "relation" and feat["name"] == "Basco"
    geom = shape(feat["geometry"])
    assert geom.geom_type == "Polygon" and len(geom.interiors) == 1
    expected = abs(Geod(ellps="WGS84").geometry_area_perimeter(
        box(0, 0, 0.1, 0.1).difference(box(0.04, 0.04, 0.06, 0.06)))[0]) / 1e6
    assert feat["area_km2"] == pytest.approx(expected, rel=1e-6)
    assert 110 < feat["area_km2"] < 125          # ~0.1° square at the equator, less the hole
    assert "sea" in feat["area_note"]             # OSM boundaries can run out to sea


def test_an_outline_with_a_gap_stays_lines_and_gets_no_area():
    rel = _boundary(members=[_way([(0.0, 0.0), (0.1, 0.0), (0.1, 0.1)]),
                             _way([(0.0, 0.1), (0.0, 0.05)])])
    feat = ov._element_to_feature(rel)
    assert feat["geometry"]["type"] == "MultiLineString"
    assert "area_km2" not in feat and feat["incomplete"]


def test_a_non_area_relation_is_the_lines_of_its_members():
    rel = {"type": "relation", "id": 7, "tags": {"type": "waterway", "waterway": "river",
                                                 "name": "Test River"},
           "members": [_way([(0, 0), (1, 0)]), _way([(1, 0), (2, 1)])]}
    feat = ov._element_to_feature(rel)
    assert feat["geometry"]["type"] == "LineString" and "area_km2" not in feat


def test_the_query_asks_for_relations_and_drops_their_member_ways(monkeypatch):
    sent = _patch(monkeypatch, [_Resp({"elements": [_boundary()]})])
    res = ov.overpass_search("boundary=administrative", bbox="0,0,1,1")
    q = sent[0]
    assert "relation[boundary=administrative]" in q
    assert "way(r.r)" in q                                   # member ways subtracted
    assert q.index(".r out geom") < q.index(".rest out geom")  # relations not crowded out
    assert res["count"] == 1 and res["features"][0]["area_km2"] > 0


def test_a_timed_out_query_reported_as_200_is_a_failure_not_an_empty_answer(monkeypatch):
    timeout = {"remark": 'runtime error: Query timed out in "query" at line 1', "elements": []}
    sent = _patch(monkeypatch, [_Resp(timeout), _Resp({"elements": [_boundary()]})])
    res = ov.overpass_search("boundary", bbox="0,0,1,1")
    assert len(sent) == 2 and res["count"] == 1               # the second mirror answered

    _patch(monkeypatch, [_Resp(timeout), _Resp(timeout)])
    res = ov.overpass_search("boundary", bbox="0,0,1,1")
    assert res["error"] == "overpass_failed" and "timed out" in res["message"]


def test_a_name_search_asks_for_relations_alone_first_and_stops_on_an_exact_name(monkeypatch):
    sent = _patch(monkeypatch, [_Resp({"elements": [
        _boundary(2, "Basco Port Area"), _boundary(1, "Basco")]})])
    res = ov.overpass_search("Basco", bbox="0,0,1,1")
    assert len(sent) == 1 and "node" not in sent[0]
    assert res["features"][0]["name"] == "Basco"              # the exact name first


def test_without_an_exact_relation_the_name_search_goes_on_to_nodes_and_ways(monkeypatch):
    poi = {"type": "node", "id": 9, "lat": 0.5, "lon": 0.5, "tags": {"name": "Basco Lighthouse"}}
    sent = _patch(monkeypatch, [_Resp({"elements": []}), _Resp({"elements": [poi]})])
    res = ov.overpass_search("Basco", bbox="0,0,1,1")
    assert len(sent) == 2 and "node[" in sent[1]
    assert res["features"][0]["name"] == "Basco Lighthouse"


def test_a_name_is_a_literal_inside_the_regex():
    assert ov._name_pattern('1st District (North).') == "1st District \\\\(North\\\\)\\\\."
    assert ov._name_pattern('Say "x"') == "Say x"


def test_boundary_words_map_to_the_boundary_key():
    assert ov._resolve_filter("boundary") == "boundary"
    assert ov._resolve_filter("administrative boundary") == "boundary=administrative"


def test_a_large_outline_is_simplified_for_display_but_its_area_is_not():
    import math
    ring = [(math.cos(t / 4000 * 2 * math.pi), math.sin(t / 4000 * 2 * math.pi))
            for t in range(4000)]
    rel = _boundary(members=[_way(ring + [ring[0]])])
    feat = ov._element_to_feature(rel)
    assert feat["vertices"] > 3000
    shown = ov.simplify_for_display(feat, max_vertices=500)
    assert shown["vertices"] <= 500 and shown["simplified"]["vertices_full"] == feat["vertices"]
    assert shown["area_km2"] == feat["area_km2"]
    assert shape(shown["full_geometry"]).equals(shape(feat["geometry"]))
    assert ov.simplify_for_display(shown | {"vertices": 10}) is not None  # small: untouched


def test_the_model_reads_the_simplified_outline_and_the_area_in_its_evidence(monkeypatch):
    from agent_runtime import langchain_granular_tools as G
    import math
    ring = [(math.cos(t / 4000 * 2 * math.pi), math.sin(t / 4000 * 2 * math.pi))
            for t in range(4000)]
    monkeypatch.setattr(ov, "_MAX_VERTICES", 500)
    monkeypatch.setattr(ov.simplify_for_display, "__defaults__", (500,))
    _patch(monkeypatch, [_Resp({"elements": [_boundary(members=[_way(ring + [ring[0]])])]})])
    out = json.loads(G.overpass_search_tool("boundary", bbox="-2,-2,2,2"))
    f = out["features"][0]
    assert "full_geometry" not in f and f["vertices"] <= 500 and f["simplified"]
    assert f"area {f['area_km2']} km²" in out["documents"][0]["contents"]
