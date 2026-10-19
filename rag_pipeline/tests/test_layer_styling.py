"""add_map_layer decides what the map shows of the features, and says what it drew (Stage 80).

Live demo tests of the Ridgecrest aftershocks example answered "sized by magnitude, the mainshock
is the largest point" over 1,054 equal circles: point size was a constant in the client and
nothing could single a feature out. The tool now takes named options (size_column / size_range /
point_size, highlight, label_column, line_width / line_color), checks each against the data, and
its result's `drawn` states what the map will show.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

gpd = pytest.importorskip("geopandas")
from shapely.geometry import LineString, Point  # noqa: E402

RIDGECREST = (Path(__file__).resolve().parents[2] / "map-ui-prototype" / "public" / "demos"
              / "aftershocks" / "usgs_comcat_ridgecrest_2019-07.csv")


@pytest.fixture()
def store(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "store"))
    monkeypatch.delenv("AGENT_PUBLIC_BASE_URL", raising=False)
    from agent_runtime.file_store import create_output_file_from_path

    def put(obj, name):
        path = tmp_path / name
        if isinstance(obj, (str, Path)):
            path.write_bytes(Path(obj).read_bytes())
        else:
            obj.to_file(path, driver="GeoJSON")
        return create_output_file_from_path(path, filename=name)["file_id"]
    return put


def _add(**args):
    from agent_runtime.langchain_geo_tools import make_langchain_geo_tools
    tool = {t.name: t for t in make_langchain_geo_tools()}["add_map_layer"]
    out = tool.invoke(args)
    return json.loads(out), out


def _quakes():
    return gpd.GeoDataFrame(
        {"mag": [3.0, 7.1, 5.0, 4.2], "kind": ["after", "main", "after", "after"],
         "place": ["a", "Ridgecrest", "b", "c"]},
        geometry=[Point(-117.5, 35.7), Point(-117.6, 35.77), Point(-117.55, 35.72),
                  Point(-117.52, 35.71)], crs=4326)


def _built(out):
    from agent_runtime.map_layers import build_map_layers
    (ml,) = build_map_layers("add_map_layer", out)
    return ml


# --- the answer is told what is drawn -----------------------------------------------------

def test_without_options_the_result_says_one_size_and_the_layer_is_unchanged(store):
    r, out = _add(file_id=store(_quakes(), "q0.geojson"), render="points")
    assert r["ok"]
    assert "all points one size" in r["drawn"] and "no feature highlighted" in r["drawn"]
    assert "no labels" in r["drawn"] and r["drawn_note"]
    ml = _built(out)
    assert not {"size_by", "point_size", "highlight", "labels", "line_width", "line_color"} & set(ml)


def test_ridgecrest_sized_by_magnitude_with_the_mainshock_highlighted(store):
    """The demo's own CSV: GDAL reads `mag` as text, and it still sizes."""
    r, out = _add(file_id=store(RIDGECREST, "ridgecrest.csv"), render="points",
                  size_column="mag", highlight={"column": "mag", "gte": 7, "label": "mainshock"},
                  label_column="mag")
    assert r["ok"], r
    assert list(r)[:2] == ["ok", "drawn"]      # survives a tool output cut for length
    assert "points sized by mag (3 to 7.1 drawn 3–16 px" in r["drawn"]
    assert "1 feature highlighted (mag >= 7)" in r["drawn"] and "'mainshock'" in r["drawn"]
    assert "1 feature (highlighted only) labelled with mag" in r["drawn"]
    ml = _built(out)
    assert ml["size_by"] == {"column": "mag", "min": 3.0, "max": 7.1, "range": [3.0, 16.0]}
    assert ml["highlight"]["op"] == "gte" and ml["highlight"]["numeric"] is True
    assert ml["highlight"]["count"] == 1 and ml["highlight"]["label"] == "mainshock"
    assert ml["labels"]["scope"] == "highlighted" and ml["labels"]["items"][0]["text"] == "7.1"
    # The written layer carries numbers, so the client compares and sizes numerically.
    from agent_runtime.file_store import resolve_file_id
    written = json.loads(Path(resolve_file_id(r["file_id"])).read_text())
    assert isinstance(written["features"][0]["properties"]["mag"], float)


# --- size ---------------------------------------------------------------------------------

def test_size_range_and_point_size(store):
    fid = store(_quakes(), "q1.geojson")
    r, out = _add(file_id=fid, render="points", size_column="mag", size_range=[2, 20])
    assert r["ok"] and _built(out)["size_by"]["range"] == [2.0, 20.0]
    r, out = _add(file_id=fid, render="points", point_size=9)
    assert r["ok"] and "all points one size (9 px)" in r["drawn"]
    assert _built(out)["point_size"] == 9.0


@pytest.mark.parametrize("args, says", [
    ({"size_column": "nope"}, "not a column"),
    ({"size_column": "kind"}, "not numeric"),
    ({"size_column": "mag", "point_size": 5}, "both set the point size"),
    ({"size_range": [3, 16]}, "needs size_column"),
    ({"size_column": "mag", "size_range": [16, 3]}, "must grow"),
    ({"size_column": "mag", "size_range": [3]}, "[smallest_px, largest_px]"),
    ({"size_column": "mag", "size_range": [3, 400]}, "size in pixels"),
    ({"point_size": 0}, "size in pixels"),
])
def test_bad_size_options_are_answerable(store, args, says):
    r, _ = _add(file_id=store(_quakes(), "q2.geojson"), render="points", **args)
    assert not r["ok"] and says in r["error"] and r["hint"]


def test_a_constant_size_column_is_refused(store):
    g = _quakes()
    g["mag"] = 4.0
    r, _ = _add(file_id=store(g, "flat.geojson"), render="points", size_column="mag")
    assert not r["ok"] and "every point would be one size" in r["error"]


def test_size_on_a_heatmap_or_on_lines_is_refused(store):
    r, _ = _add(file_id=store(_quakes(), "q3.geojson"), render="heatmap", size_column="mag")
    assert not r["ok"] and "density" in r["error"]
    lines = gpd.GeoDataFrame({"w": [1, 2]}, geometry=[LineString([(0, 0), (1, 1)]),
                                                       LineString([(1, 1), (2, 0)])], crs=4326)
    r, _ = _add(file_id=store(lines, "l.geojson"), render="shapes", size_column="w")
    assert not r["ok"] and "applies to points" in r["error"]


def test_auto_with_a_size_draws_points_not_a_heatmap(store):
    import numpy as np
    n = 2500
    g = gpd.GeoDataFrame({"mag": np.linspace(1, 5, n)},
                         geometry=[Point(-117 + i * 1e-4, 35) for i in range(n)], crs=4326)
    r, out = _add(file_id=store(g, "many.geojson"), render="auto", size_column="mag")
    assert r["ok"] and _built(out)["render"] == "points"


# --- highlight ----------------------------------------------------------------------------

def test_text_highlight_with_its_own_colour(store):
    r, out = _add(file_id=store(_quakes(), "q4.geojson"), render="points",
                  highlight={"column": "kind", "equals": "main", "color": "red"})
    assert r["ok"] and "1 feature highlighted (kind == main)" in r["drawn"]
    hl = _built(out)["highlight"]
    assert hl["color"] == [255, 0, 0, 255] and hl["numeric"] is False and hl["label"] == "kind == main"


def test_highlight_in_a_list(store):
    r, _ = _add(file_id=store(_quakes(), "q5.geojson"), render="points",
                highlight={"column": "place", "in": ["a", "b"]})
    assert r["ok"] and "2 features highlighted (place in a, b)" in r["drawn"]


@pytest.mark.parametrize("hl, says", [
    ({"equals": "main"}, "highlight is {column"),
    ({"column": "kind"}, "exactly one condition"),
    ({"column": "kind", "equals": "main", "gte": 3}, "exactly one condition"),
    ({"column": "nope", "equals": 1}, "not a column"),
    ({"column": "kind", "gte": 3}, "compares numbers"),
    ({"column": "mag", "equals": "big"}, "is not a number"),
    ({"column": "mag", "gte": 9}, "matches no feature"),
    ({"column": "mag", "gte": 0}, "matches all 4 features"),
    ({"column": "kind", "equals": "main", "color": "notacolour"}, "is not a colour"),
])
def test_bad_highlights_are_answerable(store, hl, says):
    r, _ = _add(file_id=store(_quakes(), "q7.geojson"), render="points", highlight=hl)
    assert not r["ok"] and says in r["error"] and r["hint"]


def test_a_sample_keeps_the_highlighted_features(store):
    import numpy as np
    n = 400
    g = gpd.GeoDataFrame({"mag": np.r_[np.linspace(1, 4, n - 1), 7.1]},
                         geometry=[Point(-117 + i * 1e-4, 35) for i in range(n)], crs=4326)
    r, out = _add(file_id=store(g, "sampled.geojson"), render="points", max_points=50,
                  highlight={"column": "mag", "gte": 7})
    assert r["ok"] and r["sampled"] is True
    assert r["feature_count"] == 50
    assert _built(out)["highlight"]["count"] == 1


# --- labels -------------------------------------------------------------------------------

def test_labels_on_every_feature_when_few(store):
    r, out = _add(file_id=store(_quakes(), "q8.geojson"), render="points", label_column="place")
    assert r["ok"] and "4 features labelled with place" in r["drawn"]
    items = _built(out)["labels"]["items"]
    assert [i["text"] for i in items] == ["a", "Ridgecrest", "b", "c"]
    assert items[1]["position"] == [-117.6, 35.77]


def test_too_many_labels_are_refused_with_a_way_out(store):
    n = 60
    g = gpd.GeoDataFrame({"name": [f"p{i}" for i in range(n)], "v": list(range(n))},
                         geometry=[Point(i * 0.01, 0) for i in range(n)], crs=4326)
    fid = store(g, "sixty.geojson")
    r, _ = _add(file_id=fid, render="points", label_column="name")
    assert not r["ok"] and "at most 50" in r["error"] and "highlight" in r["hint"]
    r, out = _add(file_id=fid, render="points", label_column="name",
                  highlight={"column": "v", "gte": 55})
    assert r["ok"] and "5 features (highlighted only) labelled with name" in r["drawn"]
    assert len(_built(out)["labels"]["items"]) == 5


def test_a_polygon_label_sits_inside_it(store):
    from shapely.geometry import box
    g = gpd.GeoDataFrame({"name": ["Zone A"]}, geometry=[box(0, 0, 2, 1)], crs=4326)
    r, out = _add(file_id=store(g, "zone.geojson"), render="shapes", label_column="name")
    assert r["ok"]
    x, y = _built(out)["labels"]["items"][0]["position"]
    assert 0 < x < 2 and 0 < y < 1


def test_a_label_column_that_is_missing_is_refused(store):
    r, _ = _add(file_id=store(_quakes(), "q9.geojson"), render="points", label_column="nope")
    assert not r["ok"] and "not a column" in r["error"]


# --- lines --------------------------------------------------------------------------------

def test_line_width_and_colour(store):
    r, out = _add(file_id=store(_quakes(), "q10.geojson"), render="points",
                  line_width=3, line_color="black")
    assert r["ok"] and "lines/outlines 3 px" in r["drawn"]
    ml = _built(out)
    assert ml["line_width"] == 3.0 and ml["line_color"] == [0, 0, 0, 255]


@pytest.mark.parametrize("args, says", [
    ({"line_width": 0}, "size in pixels"),
    ({"line_width": 50}, "size in pixels"),
    ({"line_color": "nonsense"}, "is not a colour"),
])
def test_bad_line_options_are_answerable(store, args, says):
    r, _ = _add(file_id=store(_quakes(), "q11.geojson"), render="points", **args)
    assert not r["ok"] and says in r["error"]


def test_lines_on_a_heatmap_are_refused(store):
    r, _ = _add(file_id=store(_quakes(), "q12.geojson"), render="heatmap", line_width=2)
    assert not r["ok"] and "heatmap" in r["error"]


# --- the descriptor boundary --------------------------------------------------------------

def test_build_map_layer_drops_malformed_styling():
    from agent_runtime.map_layers import build_map_layer
    ml = build_map_layer("x", {"map_layer": {
        "url": "https://example.org/a.geojson", "render": "points",
        "size_by": {"column": "m", "min": "x", "max": 1, "range": [3, 16]},
        "highlight": {"column": "m", "op": "like", "value": 1},
        "labels": {"items": [{"position": [1], "text": "a"}]},
        "point_size": -1, "line_width": "wide", "line_color": "nope"}}, qa=False)
    assert not {"size_by", "point_size", "highlight", "labels", "line_width", "line_color"} & set(ml)


def test_choropleth_keeps_the_styling_columns(store):
    from shapely.geometry import box
    g = gpd.GeoDataFrame({"v": [1.0, 5.0, 9.0], "name": ["a", "b", "c"]},
                         geometry=[box(i, 0, i + 1, 1) for i in range(3)], crs=4326)
    r, out = _add(file_id=store(g, "chor.geojson"), render="choropleth", column="v",
                  highlight={"column": "name", "equals": "c"}, label_column="name")
    assert r["ok"], r
    from agent_runtime.file_store import resolve_file_id
    props = json.loads(Path(resolve_file_id(r["file_id"])).read_text())["features"][0]["properties"]
    assert {"v", "name"} <= set(props)
