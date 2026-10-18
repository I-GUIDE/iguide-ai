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


# --------------------------------------------------------------------------- colours
# The palette is the map client's (palette.ts). The agent sends a colour only where the user asked
# for one or the colour carries meaning, and says which colours it set.


def test_parse_colour_reads_names_hex_and_numbers():
    from agent_runtime.map_layers import parse_colour

    assert parse_colour("red") == [255, 0, 0, 255]
    assert parse_colour("#d7191c") == [215, 25, 28, 255]
    assert parse_colour("#d7191c80") == [215, 25, 28, 128]
    assert parse_colour([10, 20, 30]) == [10, 20, 30, 255]
    assert parse_colour([1.0, 0.5, 0.0]) == [255, 128, 0, 255]
    assert parse_colour("not-a-colour") is None and parse_colour([300, 0, 0]) is None


def test_categories_send_class_names_and_only_the_colours_asked_for(store):
    fid = store(_block_groups(60), "bg3.geojson")
    out = json.loads(_add_map_layer().invoke({
        "file_id": fid, "render": "categories", "column": "nearest_branch",
        "class_colors": {"Lozano": "orange"}}))
    assert out["ok"], out
    legend = {e["label"]: e.get("color") for e in out["map_layer"]["legend"]}
    assert legend["Lozano"] == [255, 165, 0, 255]
    assert all(c is None for k, c in legend.items() if k != "Lozano"), "the rest are the map's"
    assert out["colours_set"] == {"classes": {"Lozano": "orange"}}


def test_without_a_colour_the_answer_is_told_not_to_name_one(store):
    out = json.loads(_add_map_layer().invoke({"file_id": store(_block_groups(60), "bg4.geojson"),
                                              "render": "categories", "column": "nearest_branch"}))
    assert all("color" not in e for e in out["map_layer"]["legend"])
    assert "colours_set" not in out and "do not name colours" in out["colour_note"]


def test_a_layer_colour_reaches_the_map_layer_event(store):
    from agent_runtime.map_layers import build_map_layers

    out = _add_map_layer().invoke({"file_id": store(_block_groups(60), "bg5.geojson"),
                                   "render": "points", "color": "red"})
    r = json.loads(out)
    assert r["ok"] and r["colours_set"] == {"layer": "red"}
    assert [ml.get("color") for ml in build_map_layers("add_map_layer", out)] == [[255, 0, 0, 255]]


def test_colours_that_cannot_apply_are_refused_with_the_way_that_can(store):
    add = _add_map_layer()
    fid = store(_block_groups(60), "bg6.geojson")
    bad = json.loads(add.invoke({"file_id": fid, "render": "points", "color": "blurple"}))
    assert not bad["ok"] and "not a colour" in bad["error"]
    cat = json.loads(add.invoke({"file_id": fid, "render": "categories",
                                 "column": "nearest_branch", "color": "red"}))
    assert not cat["ok"] and "class_colors" in cat["hint"]
    unknown = json.loads(add.invoke({"file_id": fid, "render": "categories",
                                     "column": "nearest_branch", "class_colors": {"Nowhere": "red"}}))
    assert not unknown["ok"] and "Nowhere" in unknown["error"]
    chor = json.loads(add.invoke({"file_id": fid, "render": "choropleth", "column": "population",
                                  "color": "red"}))
    assert not chor["ok"] and "value scale" in chor["error"]


def test_meaningful_colours_stay_with_their_tool():
    """Hot spots red and cold spots blue are the statistic's own reading, not decoration."""
    from agent_runtime.analysis_spatial_stats_tools import _HOTSPOT_COLORS, _LISA_COLORS, _legend

    hot = _legend(_HOTSPOT_COLORS, ["Hot spot (p < 0.01)", "Not significant"])
    assert all(len(e["color"]) == 4 for e in hot)
    assert _legend(_LISA_COLORS, ["High-High"])[0]["color"][0] > 200, "High-High is red"


# --------------------------------------------------------------------------- recolouring
# Live 2026-10-17: asked to colour two branches, the code peer copied the layer's file, added a
# `color` column and drew the copy, and the map listed the service areas twice.


def _child_of(store, tmp_path, parent_id, gdf, name):
    """A file written from `parent_id`, as execute_code's lineage records it."""
    import json as _json

    from agent_runtime.file_store import _metadata_path, get_file_record

    child = store(gdf, name)
    rec = get_file_record(child)
    rec["lineage"] = {"produced_by": {"tool": "execute_code"}, "parents": [parent_id], "sources": []}
    _metadata_path(child).write_text(_json.dumps(rec), encoding="utf-8")
    return child


def _layer_ids(*results):
    from agent_runtime.map_layers import build_map_layers

    return [ml["id"] for out in results for ml in build_map_layers("add_map_layer", out, qa=False)]


def test_a_copy_that_only_adds_a_colour_column_restyles_its_parents_layer(store, tmp_path):
    add = _add_map_layer()
    bg = _block_groups(60)
    parent = store(bg, "assignment.geojson")
    first = add.invoke({"file_id": parent, "render": "categories", "column": "nearest_branch"})
    coloured = bg.assign(color=["black" if b == "Lozano" else "default" for b in bg["nearest_branch"]])
    child = _child_of(store, tmp_path, parent, coloured, "assignment_colored.geojson")
    again = add.invoke({"file_id": child, "render": "categories", "column": "nearest_branch",
                        "class_colors": {"Lozano": "black"}})
    assert json.loads(again)["ok"], again
    assert len(set(_layer_ids(first, again))) == 1, "one layer, restyled in place"
    assert parent in json.loads(again)["restyled"], "and the model is told to redraw the parent"


def test_a_copy_that_adds_a_result_column_is_its_own_layer(store, tmp_path):
    add = _add_map_layer()
    bg = _block_groups(60)
    parent = store(bg, "rates.geojson")
    first = add.invoke({"file_id": parent, "render": "categories", "column": "nearest_branch"})
    lisa = bg.assign(lisa_class=["High-High" if i % 2 else "Low-Low" for i in range(len(bg))])
    child = _child_of(store, tmp_path, parent, lisa, "rates_lisa.geojson")
    other = add.invoke({"file_id": child, "render": "categories", "column": "lisa_class"})
    assert len(set(_layer_ids(first, other))) == 2, "a new result keeps its own layer"
    assert "restyled" not in json.loads(other)


def test_a_coloured_file_with_no_recorded_parent_is_its_own_layer(store):
    add = _add_map_layer()
    bg = _block_groups(60)
    first = add.invoke({"file_id": store(bg, "a.geojson"), "render": "categories",
                        "column": "nearest_branch"})
    other = add.invoke({"file_id": store(bg.assign(color="red"), "b.geojson"),
                        "render": "categories", "column": "nearest_branch"})
    assert len(set(_layer_ids(first, other))) == 2


def test_the_tool_says_how_to_recolour():
    desc = _add_map_layer().description
    assert "RECOLOUR" in desc and "SAME file_id" in desc


def test_redrawing_the_styled_copy_add_map_layer_wrote_restyles_the_same_layer(store):
    """Live, second try: the model recoloured by drawing the file add_map_layer had returned
    for the first draw (its styled copy), not the file it drew from."""
    add = _add_map_layer()
    first = add.invoke({"file_id": store(_block_groups(60), "assigned.geojson"),
                        "render": "categories", "column": "nearest_branch"})
    copy_id = json.loads(first)["file_id"]
    again = add.invoke({"file_id": copy_id, "render": "categories", "column": "nearest_branch",
                        "class_colors": {"Lozano": "black", "Portage-Cragin": "hotpink"}})
    assert json.loads(again)["ok"], again
    assert len(set(_layer_ids(first, again))) == 1, "the recolour replaces the first layer"
    third = add.invoke({"file_id": json.loads(again)["file_id"], "render": "categories",
                        "column": "nearest_branch"})
    assert len(set(_layer_ids(first, again, third))) == 1, "and so does a copy of the copy"
