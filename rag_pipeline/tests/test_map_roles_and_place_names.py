"""Three things the Spatial Analysis examples showed live on 2026-10-17.

1. Cooling centres, step 2 ("map the block groups coloured by the branch that serves them"): the
   2,158 block-group points came out as a heatmap. add_map_layer's render="auto" ignored the
   column it was given and switched every point set past 2,000 to a density surface.
2. Crash hot spots: hexagon H1852 at -87.622, 41.887 was described as "around Wrigleyville". It
   is in the Loop. No tool named the place; there was no tool that could.
3. Routes: a fastest path and the network it was found on were drawn with the same thin default
   line. A route now has its own render (a navigation-style cased line with its ends marked) and
   a road network its own (muted context lines beneath it).
"""
from __future__ import annotations

import json

import pytest

gpd = pytest.importorskip("geopandas")
from shapely.geometry import LineString, Point, box  # noqa: E402


@pytest.fixture()
def store(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "store"))
    monkeypatch.delenv("AGENT_PUBLIC_BASE_URL", raising=False)
    from agent_runtime.file_store import create_output_file_from_path

    def put(gdf, name):
        path = tmp_path / name
        gdf.to_file(path, driver="GeoJSON")
        return create_output_file_from_path(path, filename=name)["file_id"]
    return put


def _add_map_layer():
    from agent_runtime.langchain_geo_tools import make_langchain_geo_tools

    return {t.name: t for t in make_langchain_geo_tools()}["add_map_layer"]


def _block_groups(n=2158):
    branches = ["Budlong Woods", "Chicago Lawn", "Lincoln Park", "Lozano", "Portage-Cragin",
                "Whitney M. Young, Jr."]
    pts = [Point(-87.9 + (i % 50) * 0.01, 41.65 + (i // 50) * 0.01) for i in range(n)]
    return gpd.GeoDataFrame({"nearest_branch": [branches[i % 6] for i in range(n)],
                             "population": [500 + i for i in range(n)]},
                            geometry=pts, crs="EPSG:4326")


def test_auto_with_a_class_column_colours_by_it_however_many_points(store):
    fid = store(_block_groups(), "bg.geojson")
    out = json.loads(_add_map_layer().invoke({"file_id": fid, "render": "auto",
                                              "column": "nearest_branch"}))
    assert out["ok"], out
    ml = out["map_layer"]
    assert ml["render"] == "categories", "2,158 points with a branch column are not a heatmap"
    assert {e["label"] for e in ml["legend"]} == set(_block_groups()["nearest_branch"])


def test_auto_with_a_numeric_column_is_a_choropleth_and_without_one_still_a_heatmap(store):
    fid = store(_block_groups(), "bg2.geojson")
    add = _add_map_layer()
    num = json.loads(add.invoke({"file_id": fid, "render": "auto", "column": "population"}))
    assert num["ok"] and num["map_layer"]["render"] == "choropleth", num
    plain = json.loads(add.invoke({"file_id": fid, "render": "auto"}))
    assert plain["ok"] and plain["map_layer"]["render"] == "heatmap", plain


def test_the_tool_description_offers_categories_and_route():
    desc = _add_map_layer().description
    assert "'categories'" in desc and "'route'" in desc and "'network'" in desc


def test_route_and_network_draw_lines_and_refuse_anything_else(store):
    add = _add_map_layer()
    route = gpd.GeoDataFrame({"seq": [1, 2]}, crs="EPSG:4326",
                             geometry=[LineString([(-87.64, 41.878), (-87.63, 41.878)]),
                                       LineString([(-87.63, 41.878), (-87.624, 41.879)])])
    ok = json.loads(add.invoke({"file_id": store(route, "route.geojson"), "render": "route",
                                "name": "Fastest drive"}))
    assert ok["ok"] and ok["map_layer"]["render"] == "route", ok
    polys = gpd.GeoDataFrame({"a": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
    bad = json.loads(add.invoke({"file_id": store(polys, "p.geojson"), "render": "route"}))
    assert not bad["ok"] and "draws lines" in bad["error"]


def test_a_route_layer_keeps_its_render_through_the_map_layer_event(store):
    from agent_runtime.map_layers import build_map_layers

    route = gpd.GeoDataFrame({"seq": [1]}, crs="EPSG:4326",
                             geometry=[LineString([(-87.64, 41.878), (-87.624, 41.879)])])
    out = _add_map_layer().invoke({"file_id": store(route, "r.geojson"), "render": "route"})
    layers = build_map_layers("add_map_layer", out)
    assert [ml["render"] for ml in layers] == ["route"]


def test_reverse_geocode_names_each_point_from_nominatim(monkeypatch):
    from agent_runtime import langchain_granular_tools as g
    import rag_pipeline.search.opengeodata_new as og

    calls = []

    def fake(lat, lon, zoom=16):
        calls.append((lat, lon, zoom))
        return {"display_name": "North Lower Stetson Avenue, New East Side, Loop, Chicago",
                "address": {"road": "North Lower Stetson Avenue", "neighbourhood": "New East Side",
                            "suburb": "Loop", "city": "Chicago", "state": "Illinois"}}
    monkeypatch.setattr(og, "reverse_geocode_point", fake)
    out = json.loads(g.reverse_geocode_tool([{"lat": 41.8872, "lon": -87.6222}, [41.95, -87.65]]))
    assert out["ok"] and len(out["results"]) == 2
    first = out["results"][0]
    assert first["place"] == "New East Side" and first["address"]["suburb"] == "Loop"
    assert first["road"] == "North Lower Stetson Avenue"
    assert calls[1][:2] == (41.95, -87.65), "[lat, lon] pairs are read as lat, lon"
    assert "OpenStreetMap" in out["source"]
    # No lat/lon fields: the map UI would preview the queried points as a stray layer.
    assert first["point"] == [41.8872, -87.6222] and "lat" not in first and "lon" not in first


def test_reverse_geocode_refuses_what_is_not_a_coordinate():
    from agent_runtime import langchain_granular_tools as g

    out = json.loads(g.reverse_geocode_tool("not json"))
    assert not out["ok"]
    out = json.loads(g.reverse_geocode_tool([{"lat": 200, "lon": 0}]))
    assert not out["ok"], "a latitude of 200 is not a place"


def test_both_peers_and_the_synthesizer_carry_the_place_name_rule():
    from agent_runtime.supervisor import prompts as p

    assert p.PLACE_NAMES_RULE in p.ANALYSIS_WORKFLOW_PROMPT
    assert p.PLACE_NAMES_RULE in p.CODE_PEER_PROMPT
    assert "reverse_geocode" in p.PLACE_NAMES_RULE
    assert "9. PLACE NAMES" in p.SYNTHESIS_PROMPT


def test_reverse_geocode_is_bound_with_the_geocoder():
    from agent_runtime.langchain_granular_tools import make_langchain_geocode_tools

    assert [t.name for t in make_langchain_geocode_tools()] == ["geocode_places", "reverse_geocode"]
