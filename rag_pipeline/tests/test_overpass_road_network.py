"""overpass_search can fetch a whole drivable road network, as a file (stage 69).

Live, 2026-10-15 (local agent, deepseek-v4-flash): "the fastest drive across Chicago's Loop on
OpenStreetMap streets, following one-ways, posted maxspeed else 30 mph". The model called
overpass_search(feature='highway', bbox=..., limit=500) and got 34 features: 'highway' means
highway=motorway, 'roads' highway=primary and 'streets' highway=residential, a regex filter
had no '=' and became a NAME search, and the tool returned at most 500 features into the
model's context. Offline: ``requests.post`` is patched, nothing reaches Overpass.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_pipeline.search import overpass as ov

LOOP_BBOX = "-87.645,41.866,-87.618,41.889"


class _Resp:
    def __init__(self, payload, status=200):
        self.status_code, self._payload = status, payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"{self.status_code}")

    def json(self):
        return self._payload


def _grid(n=21, x0=-87.645, y0=41.866, step=0.001):
    """An n x n street grid as OSM ways, one way per block: 2*n*(n-1) ways (840 for n=21), well
    past the 500 the model-facing tool could return. Plus three signal nodes."""
    classes = ("residential", "secondary", "primary", "tertiary")
    els, wid = [], 1000
    for i in range(n):
        for j in range(n - 1):
            for horizontal in (True, False):
                a = (x0 + j * step, y0 + i * step) if horizontal else (x0 + i * step, y0 + j * step)
                b = (a[0] + step, a[1]) if horizontal else (a[0], a[1] + step)
                tags = {"highway": classes[(i + j) % 4], "name": f"{'W' if horizontal else 'N'} "
                        f"Street {i}"}
                if i % 2:
                    tags["oneway"] = "yes" if horizontal else "-1"
                if i % 3 == 0:
                    tags["maxspeed"] = "30 mph"
                els.append({"type": "way", "id": wid, "tags": tags, "nodes": [1, 2],
                            "geometry": [{"lon": a[0], "lat": a[1]}, {"lon": b[0], "lat": b[1]}]})
                wid += 1
    for k in range(3):
        els.append({"type": "node", "id": k + 1, "lon": x0, "lat": y0,
                    "tags": {"highway": "traffic_signals"}})
    return els


def _patch(monkeypatch, responses):
    sent = []

    def fake_post(url, data=None, headers=None, timeout=None):
        sent.append(data["data"])
        nxt = responses.pop(0) if responses else _Resp({"elements": []})
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    monkeypatch.setattr(ov.requests, "post", fake_post)
    monkeypatch.setattr(ov, "OVERPASS_ENDPOINTS", ["https://a/api"])
    return sent


@pytest.fixture
def store(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path))
    return tmp_path


# --------------------------------------------------------------------------- filters


@pytest.mark.parametrize("feature, expected", [
    # Raw Overpass filters pass through as written.
    ('highway~"^(primary|secondary)$"', 'highway~"^(primary|secondary)$"'),
    ("highway!=footway", "highway!=footway"),
    ('[highway~"^(a|b)$"][oneway=yes]', 'highway~"^(a|b)$"][oneway=yes'),
    ('"name"~"Loop",i', '"name"~"Loop",i'),
    ("amenity=school", "amenity=school"),
    # Every value of a key, said in OSM syntax.
    ("[highway]", "highway"),
    ("highway=*", "highway"),
    # A form models write that Overpass does not have.
    ("highway in (motorway, trunk, 'primary')", 'highway~"^(motorway|trunk|primary)$"'),
    # Unchanged: the words, bare keys and names.
    ("highway", "highway=motorway"),
    ("roads", "highway=primary"),
    ("streets", "highway=residential"),
    ("waterway", "waterway"),
    ("Basco", None),
    ("1st District of Palawan", None),
])
def test_filters_resolve(feature, expected):
    assert ov._resolve_filter(feature) == expected


@pytest.mark.parametrize("phrase", ["road network", "Road Network", "street network",
                                    "drivable roads", "drive network", "all streets"])
def test_network_phrases_ask_for_every_drivable_class(phrase):
    f = ov._resolve_filter(phrase)
    assert f == ov.DRIVE_NETWORK_FILTER and ov.is_road_network_filter(f)
    for cls in ("motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
                "residential", "living_street", "primary_link"):
        assert f"|{cls}|" in f or f"({cls}|" in f or f"|{cls})" in f
    assert "service" not in f and 'access!~"^(private|no)$"' in f


@pytest.mark.parametrize("osm_filter, network", [
    ("highway", True), ('highway~"^(primary|secondary)$"', True), ("highway!=footway", True),
    ("highway=primary", False), ("amenity=school", False), ("waterway", False), (None, False),
    ("!highway", False),
])
def test_what_counts_as_a_network(osm_filter, network):
    assert ov.is_road_network_filter(osm_filter) is network


def test_a_regex_filter_is_sent_as_a_tag_filter_not_a_name_search(monkeypatch):
    sent = _patch(monkeypatch, [_Resp({"elements": []})])
    ov.overpass_search('highway~"^(primary|secondary)$"', bbox=LOOP_BBOX)
    assert 'way[highway~"^(primary|secondary)$"](41.866,-87.645,41.889,-87.618)' in sent[0]
    assert '"name"~' not in sent[0]


def test_the_network_filter_is_valid_selector_text(monkeypatch):
    sent = _patch(monkeypatch, [_Resp({"elements": []})])
    ov.overpass_search("road network", bbox=LOOP_BBOX)
    assert ('way[highway~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|'
            'living_street|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link)$"]'
            '[area!~"^yes$"][access!~"^(private|no)$"][motor_vehicle!~"^(private|no)$"]'
            '[motorcar!~"^(private|no)$"](41.866,-87.645,41.889,-87.618)') in sent[0]


def test_a_one_class_word_says_it_is_one_class(monkeypatch):
    _patch(monkeypatch, [_Resp({"elements": _grid(3)[:4]})])
    out = ov.overpass_search("highway", bbox=LOOP_BBOX)
    warn = " ".join(out["diagnostics"].get("warnings", []))
    assert "highway=motorway only" in warn and "road network" in warn


# --------------------------------------------------------------------------- the network path


def _tools():
    from agent_runtime.langchain_granular_tools import (
        make_langchain_granular_tools, make_langchain_osm_tools)

    (peer,) = make_langchain_osm_tools()
    (search,) = [t for t in make_langchain_granular_tools() if t.name == "overpass_search"]
    return {"analyze/code": peer, "search": search}


@pytest.mark.parametrize("which", ["analyze/code", "search"])
def test_a_network_comes_back_whole_as_a_file_not_as_features(monkeypatch, store, which):
    from agent_runtime.file_store import resolve_file_id

    grid = _grid()
    sent = _patch(monkeypatch, [_Resp({"elements": grid})])
    raw = _tools()[which].invoke({"feature": "road network", "bbox": LOOP_BBOX})
    out = json.loads(raw)

    assert len(sent) == 1
    assert out["network"] is True and out["count"] == 840 and out["complete"] is True
    assert "features" not in out, "the ways go to the file, not into the model's context"
    assert len(raw) < 8000
    assert out["by_class"] == {"residential": 210, "secondary": 210, "primary": 210,
                               "tertiary": 210}
    assert out["oneway"] == {"both": 440, "forward": 200, "reverse": 200}
    assert out["maxspeed_tagged"] == 7 * 40 and out["maxspeed_values"] == {"30 mph": 280}
    assert 80 < out["total_km"] < 100              # 840 blocks of ~0.1 km
    assert "840 OpenStreetMap road ways" in out["summary"]
    assert "oneway" in out["how_to_route"] and "execute_code" in out["how_to_route"]
    assert "traffic signals" not in json.dumps(out)
    assert "3 point features" in out["diagnostics"]["warnings"][0]

    fc = json.loads(Path(resolve_file_id(out["file_id"])).read_text(encoding="utf-8"))
    assert len(fc["features"]) == 840
    assert {f["geometry"]["type"] for f in fc["features"]} == {"LineString"}
    props = fc["features"][len(fc["features"]) // 2]["properties"]
    for key in ("osm_id", "name", "highway", "oneway", "maxspeed", "length_m"):
        assert key in props
    assert any(f["properties"]["oneway"] == "-1" for f in fc["features"])

    ml = out["map_layer"]
    assert ml["url"] == out["download_url"] and ml["render"] == "network" and ml["count"] == 840
    assert out["citation_ids"] == ["osm:road_network"]


def test_the_network_layer_reaches_the_map_by_url(monkeypatch, store):
    from agent_runtime.map_layers import build_map_layers

    _patch(monkeypatch, [_Resp({"elements": _grid(5)})])
    out = _tools()["search"].invoke({"feature": "road network", "bbox": LOOP_BBOX})
    (layer,) = build_map_layers("overpass_search", out)
    assert layer["url"] and "geojson" not in layer and layer["count"] == 40


def test_the_same_network_asked_twice_is_one_layer(monkeypatch, store):
    _patch(monkeypatch, [_Resp({"elements": _grid(3)}), _Resp({"elements": _grid(3)})])
    tool = _tools()["analyze/code"]
    a = json.loads(tool.invoke({"feature": "road network", "bbox": LOOP_BBOX}))
    b = json.loads(tool.invoke({"feature": "road network", "bbox": LOOP_BBOX}))
    assert a["map_layer"]["id"] == b["map_layer"]["id"]


def test_a_failed_network_query_is_an_error_and_writes_no_file(monkeypatch, store):
    from agent_runtime import tool_bridge

    monkeypatch.setattr(tool_bridge, "_OVERPASS_RETRY_S", 0)
    _patch(monkeypatch, [RuntimeError("502 Bad Gateway")] * 40)
    out = json.loads(_tools()["analyze/code"].invoke({"feature": "road network",
                                                      "bbox": LOOP_BBOX}))
    assert out["error"] == "overpass_failed" and "file_id" not in out
    assert "no partial result" in out["message"]


def test_a_network_with_no_location_says_so(monkeypatch, store):
    out = json.loads(_tools()["search"].invoke({"feature": "road network"}))
    assert out["error"] == "no_location"


def test_a_single_class_is_still_returned_as_features(monkeypatch, store):
    _patch(monkeypatch, [_Resp({"elements": _grid(3)})])
    out = json.loads(_tools()["search"].invoke({"feature": "highway=primary",
                                                "bbox": LOOP_BBOX}))
    assert "network" not in out and out["count"] == 15 and out["features"]


def test_the_bridge_gets_the_network_filter_too(monkeypatch, store):
    from agent_runtime import tool_bridge

    sent = _patch(monkeypatch, [_Resp({"elements": _grid(3)})])
    out = tool_bridge._overpass_handler("road network", bbox=[-87.645, 41.866, -87.618, 41.889])
    assert out["osm_filter"] == ov.DRIVE_NETWORK_FILTER and out["count"] == 15
    assert "living_street" in sent[0]


def test_routing_instructions_forbid_snapping_and_name_the_strong_component(monkeypatch, store):
    """S69.1. Live, 2026-10-17: the corners of the Loop box sat on one-way stubs the box cut off,
    the model merged every pair of vertices within 5 m to get a path, and drove between Upper
    and Lower Wacker Drive eleven times (4.89 min against a legal 5.19). The instructions now
    forbid any tolerance join and say where to snap the endpoints instead."""
    _patch(monkeypatch, [_Resp({"elements": _grid(3)})])
    how = json.loads(_tools()["search"].invoke({"feature": "road network",
                                                "bbox": LOOP_BBOX}))["how_to_route"]
    assert "EXACT vertex coordinates" in how
    assert "NEVER merge, snap or buffer vertices" in how and "never add an edge" in how
    assert "LARGEST STRONGLY connected component" in how and "weakly" in how
    assert "say so instead of changing the network" in how
    from agent_runtime.langchain_granular_tools import _ROAD_NETWORK_NOTE

    assert "only at shared vertices" in _ROAD_NETWORK_NOTE
