"""Tests for the agent-side geocode_places tool (mocked Nominatim — no network)."""

from __future__ import annotations

import json

import rag_pipeline.search.opengeodata_new as og


def _fake_geocode(name):
    known = {
        "University of Illinois Urbana-Champaign": (-88.3, 40.0, -88.1, 40.2),
        "Michigan State University": (-84.5, 42.6, -84.4, 42.8),
    }
    return known.get(name)


def test_geocode_places_list_input(monkeypatch):
    monkeypatch.setattr(og, "geocode_place", _fake_geocode)
    from agent_runtime.langchain_granular_tools import geocode_places_tool
    out = json.loads(geocode_places_tool(
        ["University of Illinois Urbana-Champaign", "ORCID", "Michigan State University"]))
    assert out["count"] == 2
    assert out["not_found"] == ["ORCID"]
    uiuc = out["results"][0]
    assert uiuc["found"] and abs(uiuc["lat"] - 40.1) < 1e-6 and abs(uiuc["lon"] - (-88.2)) < 1e-6
    assert uiuc["bbox"] == [-88.3, 40.0, -88.1, 40.2]
    assert out["results"][1] == {"place": "ORCID", "found": False}


def test_geocode_places_string_inputs(monkeypatch):
    monkeypatch.setattr(og, "geocode_place", _fake_geocode)
    from agent_runtime.langchain_granular_tools import geocode_places_tool
    # comma-separated string
    out = json.loads(geocode_places_tool("Michigan State University, ORCID"))
    assert out["count"] == 1 and out["not_found"] == ["ORCID"]
    # JSON-encoded list string
    out2 = json.loads(geocode_places_tool('["Michigan State University"]'))
    assert out2["count"] == 1


def test_geocode_places_caps_input_and_never_raises(monkeypatch):
    calls = {"n": 0}

    def boom(name):
        calls["n"] += 1
        raise RuntimeError("nominatim down")
    monkeypatch.setattr(og, "geocode_place", boom)
    from agent_runtime.langchain_granular_tools import _GEOCODE_MAX_PLACES, geocode_places_tool
    out = json.loads(geocode_places_tool([f"place-{i}" for i in range(_GEOCODE_MAX_PLACES + 10)]))
    assert calls["n"] == _GEOCODE_MAX_PLACES            # capped
    assert out["count"] == 0                             # errors degrade to found=false
    assert "truncated" in out.get("note", "")


def test_geocode_tool_wired_into_code_and_analyze_peers(monkeypatch):
    """Both peers must expose geocode_places so named-place maps never ask the user."""
    from types import SimpleNamespace
    import agent_runtime.executor_factory as ef
    import agent_runtime.langchain_granular_tools as gt
    import agent_runtime.langchain_file_tools as ft
    import agent_runtime.supervisor_graph as sg

    monkeypatch.delenv("AGENT_CODE_EXEC", raising=False)
    monkeypatch.setattr(gt, "make_langchain_qgis_tools", lambda **k: [])
    monkeypatch.setattr(ft, "make_langchain_file_tools", lambda: [SimpleNamespace(name="read_text_file")])
    captured = {}

    def fake_build(**kwargs):
        captured["tools"] = [getattr(t, "name", "") for t in (kwargs.get("preloaded_tools") or [])]
        return object()
    monkeypatch.setattr(ef, "build_agent_executor", fake_build)
    monkeypatch.setattr(ef, "invoke_agent_with_payload_fallback", lambda *a, **k: {"messages": []})

    sg.default_code_fn()("bubble map of institutions", [], {"thread_id": None})
    assert "geocode_places" in captured["tools"]

    captured.clear()
    sg.default_analyze_fn(include_mcp_tools=False)("map institutions", [], {"thread_id": None})
    assert "geocode_places" in captured["tools"]


# Live, 2026-10-08: London–Paris was answered 340.0 km because geocode_places returned bbox
# centres; Greater London's sits 3.4 km from Nominatim's own point, and the great circle between
# Nominatim's points is 343.7 km. These are the real Nominatim answers for both names.
_NOMINATIM = {
    "London": {"lat": "51.5074456", "lon": "-0.1277653",
               "boundingbox": ["51.2867601", "51.6918741", "-0.5103751", "0.3340155"]},
    "Paris": {"lat": "48.8534951", "lon": "2.3483915",
              "boundingbox": ["48.8155755", "48.9021560", "2.2241220", "2.4697602"]},
}


class _Resp:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


class _Session:
    def __init__(self):
        self.calls = 0

    def get(self, url, params=None, headers=None):
        self.calls += 1
        hit = _NOMINATIM.get(params["q"])
        return _Resp([hit] if hit else [])


def _haversine(a, b):
    from math import asin, cos, radians, sin, sqrt
    (la1, lo1), (la2, lo2) = a, b
    h = sin(radians(la2 - la1) / 2) ** 2 + \
        cos(radians(la1)) * cos(radians(la2)) * sin(radians(lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(h))


def test_geocode_places_returns_nominatims_point_not_the_box_centre(monkeypatch):
    sess = _Session()
    monkeypatch.setattr(og, "session", lambda: sess)
    monkeypatch.setattr(og, "last_geocode_call", -10.0)
    for name in _NOMINATIM:
        og.geocode_cache.pop(name, None)
        og.geocode_point_cache.pop(name, None)
    from agent_runtime.langchain_granular_tools import geocode_places_tool
    try:
        out = json.loads(geocode_places_tool(["London", "Paris"]))
    finally:
        for name in _NOMINATIM:
            og.geocode_cache.pop(name, None)
            og.geocode_point_cache.pop(name, None)
    london, paris = out["results"]
    assert (london["lat"], london["lon"]) == (51.507446, -0.127765)
    assert london["point"] == "nominatim" and paris["point"] == "nominatim"
    assert london["bbox"] == [-0.5103751, 51.2867601, 0.3340155, 51.6918741]
    km = _haversine((london["lat"], london["lon"]), (paris["lat"], paris["lon"]))
    assert abs(km - 343.7) < 0.1, km
    assert sess.calls == 2                       # the point costs no extra request


def test_geocode_places_falls_back_to_the_box_centre_without_a_point(monkeypatch):
    monkeypatch.setattr(og, "geocode_place", _fake_geocode)
    monkeypatch.setattr(og, "geocode_place_point", lambda name: (0.0, 0.0))  # outside the box
    from agent_runtime.langchain_granular_tools import geocode_places_tool
    uiuc = json.loads(geocode_places_tool(["University of Illinois Urbana-Champaign"]))["results"][0]
    assert uiuc["point"] == "bbox_centre"
    assert abs(uiuc["lat"] - 40.1) < 1e-6 and abs(uiuc["lon"] - (-88.2)) < 1e-6
