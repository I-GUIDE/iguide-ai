"""Tests for the vector overlay / geometry tools (agent_runtime.analysis_overlay_tools).

Everything is built from tiny synthetic GeoDataFrames written to a tmp file store — no
network, no fixtures on disk. The invariants under test are the ones that break silently
in production: a result that is actually plottable (map_layer with a style_by column that
EXISTS in the written GeoJSON), a missing column that reports its candidates instead of a
stack trace, and metric work (buffer / simplify / area) that is not done in degrees.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

gpd = pytest.importorskip("geopandas")
pytest.importorskip("pyogrio")
from shapely.geometry import LineString, Point, Polygon, box  # noqa: E402

from agent_runtime.analysis_overlay_tools import make_overlay_tools  # noqa: E402

TOOL_NAMES = {"clip_layer", "dissolve_layer", "intersect_layers", "erase_layer",
              "buffer_layer", "simplify_layer", "geometry_summary"}


def _tools():
    return {t.name: t for t in make_overlay_tools()}


def _call(tool_name, **kwargs):
    return json.loads(_tools()[tool_name].invoke(kwargs))


def _written(result, storage_root):
    """Parse the GeoJSON the tool actually wrote (proves style_by is really in the file)."""
    root = Path(storage_root)
    matches = list((root / "outputs").glob(f"{result['file_id']}__*"))
    assert matches, f"no output written for {result['file_id']}"
    return json.loads(matches[0].read_text(encoding="utf-8"))


@pytest.fixture()
def store(monkeypatch, tmp_path):
    root = tmp_path / "store"
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(root))
    return root


@pytest.fixture()
def data(store, tmp_path):
    """Three adjacent tracts (2 states), a city box overlapping them, 2 points, 1 road."""
    work = tmp_path / "work"
    work.mkdir()

    def write(gdf, stem):
        path = work / f"{stem}.geojson"
        gdf.to_file(path, driver="GeoJSON")
        return str(path)

    tracts = gpd.GeoDataFrame(
        {"state": ["IL", "IL", "IN"], "pop": [100, 200, 300], "label": ["a", "b", "c"]},
        geometry=[box(-88.3, 40.0, -88.2, 40.1),
                  box(-88.2, 40.0, -88.1, 40.1),
                  box(-88.1, 40.0, -88.0, 40.1)],
        crs="EPSG:4326")
    city = gpd.GeoDataFrame({"city": ["Metropolis"]},
                            geometry=[box(-88.25, 40.02, -88.05, 40.08)], crs="EPSG:4326")
    points = gpd.GeoDataFrame({"kind": ["school", "school"], "students": [10, 20]},
                              geometry=[Point(-88.25, 40.05), Point(-88.05, 40.05)],
                              crs="EPSG:4326")
    road = gpd.GeoDataFrame({"road": ["Main"]},
                            geometry=[LineString([(-88.3, 40.05), (-88.0, 40.05)])],
                            crs="EPSG:4326")
    # A jagged ring whose extra vertices sit ~10 m apart, so a 500 m tolerance must drop them.
    jag = [(-88.30, 40.20), (-88.2999, 40.2001), (-88.2998, 40.2000), (-88.2997, 40.2001),
           (-88.20, 40.20), (-88.20, 40.25), (-88.30, 40.25)]
    jagged = gpd.GeoDataFrame({"zone": ["z"]}, geometry=[Polygon(jag)], crs="EPSG:4326")
    far = gpd.GeoDataFrame({"z": [1]}, geometry=[box(10.0, 10.0, 11.0, 11.0)], crs="EPSG:4326")
    return {
        "tracts": write(tracts, "tracts"),
        "city": write(city, "city"),
        "points": write(points, "points"),
        "road": write(road, "road"),
        "jagged": write(jagged, "jagged"),
        "far": write(far, "far"),
    }


# --- factory ---------------------------------------------------------------------

def test_factory_exposes_the_overlay_toolset():
    tools = _tools()
    assert set(tools) == TOOL_NAMES
    assert all(getattr(t, "metadata", {}).get("category") == "geo" for t in tools.values())
    for tool in tools.values():
        assert len(tool.description) > 80, f"{tool.name} needs a model-facing description"


# --- happy paths: mappable layer + real style_by ---------------------------------

def test_dissolve_returns_choropleth_whose_style_by_exists_in_the_output(data, store):
    r = _call("dissolve_layer", file_id=data["tracts"], by="state", statistic="sum")
    assert r["ok"] is True
    assert r["feature_count"] == 2                       # IL (2 tracts merged) + IN
    layer = r["map_layer"]
    assert layer["render"] == "choropleth"
    assert layer["source"] == "analysis" and layer["count"] == 2
    assert layer["url"] == r["download_url"]
    style_by = layer["style_by"]
    assert style_by == "pop"
    props = [f["properties"] for f in _written(r, store)["features"]]
    assert all(style_by in p for p in props), "choropleth style_by must exist in the GeoJSON"
    assert isinstance(props[0][style_by], (int, float))
    il = next(p for p in props if p["state"] == "IL")
    assert il["pop"] == 300 and il["feature_count"] == 2  # summed, and rows counted
    assert r["table"]["filename"].endswith(".csv")        # downloadable table behind the layer
    assert r["filename"] == "tracts_dissolved.geojson"    # purpose-derived download name


def test_dissolve_without_by_merges_everything(data):
    r = _call("dissolve_layer", file_id=data["tracts"])
    assert r["ok"] is True and r["feature_count"] == 1
    assert "__all__" not in r["columns"]                  # helper key must not leak


def test_dissolve_statistic_mean(data, store):
    r = _call("dissolve_layer", file_id=data["tracts"], by="state", statistic="mean")
    props = [f["properties"] for f in _written(r, store)["features"]]
    assert next(p for p in props if p["state"] == "IL")["pop"] == pytest.approx(150.0)


def test_clip_keeps_only_the_inside_and_remeasures_it(data, store):
    r = _call("clip_layer", target_file_id=data["tracts"], clip_file_id=data["city"])
    assert r["ok"] is True and r["on_map"] is True
    assert r["input_features"] == 3
    assert r["map_layer"]["style_by"] == "area_km2"
    props = [f["properties"] for f in _written(r, store)["features"]]
    assert all("area_km2" in p for p in props)
    # The clip window is 0.06 deg tall; every clipped remnant must be SMALLER than its tract.
    full = 0.1 * 0.1 * 111.0 * 111.0 / 1.5                # rough lower bound, tracts are ~85 km2
    assert all(p["area_km2"] < full for p in props)
    assert {p["state"] for p in props} <= {"IL", "IN"}     # target attributes survive


def test_intersect_carries_attributes_of_both_layers(data, store):
    r = _call("intersect_layers", left_file_id=data["tracts"], right_file_id=data["city"])
    assert r["ok"] is True
    props = [f["properties"] for f in _written(r, store)["features"]]
    assert all({"state", "pop", "city", "area_km2"} <= set(p) for p in props)
    assert r["map_layer"]["style_by"] == "area_km2"
    assert r["left_features"] == 3 and r["right_features"] == 1


def test_intersect_handles_points_against_polygons(data):
    """Mixed geometry dimensions must not fall over (overlay or the sjoin fallback)."""
    r = _call("intersect_layers", left_file_id=data["points"], right_file_id=data["city"])
    assert r["ok"] is True and r["feature_count"] == 2
    assert r["map_layer"]["render"] == "points"


def test_erase_removes_the_covered_part(data, store):
    r = _call("erase_layer", target_file_id=data["tracts"], erase_file_id=data["city"])
    assert r["ok"] is True and r["feature_count"] == 3
    props = [f["properties"] for f in _written(r, store)["features"]]
    assert all({"state", "pop"} <= set(p) for p in props)
    # Erasing the city band leaves less than the whole tract but not nothing.
    assert all(0 < p["area_km2"] < 100 for p in props)
    erased = _call("erase_layer", target_file_id=data["points"], erase_file_id=data["city"])
    assert erased["ok"] is True and erased["feature_count"] == 0   # both points were inside
    assert erased["on_map"] is False and "0 features" in erased["note"]


def test_geometry_summary_centroids_are_points_carrying_source_area(data, store):
    r = _call("geometry_summary", file_id=data["tracts"])
    assert r["ok"] is True and r["feature_count"] == 3
    assert r["map_layer"]["render"] == "points"
    assert r["map_layer"]["style_by"] == "area_km2"
    feats = _written(r, store)["features"]
    assert all(f["geometry"]["type"] == "Point" for f in feats)
    props = [f["properties"] for f in feats]
    assert all({"centroid_lon", "centroid_lat", "area_km2"} <= set(p) for p in props)
    lon, lat = props[0]["centroid_lon"], props[0]["centroid_lat"]
    assert -88.31 < lon < -87.99 and 39.99 < lat < 40.11
    assert r["table"]["row_count"] == 3


def test_geometry_summary_whole_layer_hull_and_bbox(data, store):
    hull = _call("geometry_summary", file_id=data["tracts"], output="convex_hull",
                 per_feature=False)
    assert hull["ok"] is True and hull["feature_count"] == 1
    assert hull["map_layer"]["render"] == "shapes"
    bbox = _call("geometry_summary", file_id=data["tracts"], output="bbox", per_feature=False)
    assert bbox["ok"] is True and bbox["feature_count"] == 1
    ring = _written(bbox, store)["features"][0]["geometry"]["coordinates"][0]
    xs = [c[0] for c in ring]
    assert min(xs) == pytest.approx(-88.3, abs=1e-6) and max(xs) == pytest.approx(-88.0, abs=1e-6)
    assert bbox["summary"]["bounds_wgs84"] == pytest.approx([-88.3, 40.0, -88.0, 40.1], abs=1e-6)


# --- bad input: ok=false with candidates -----------------------------------------

def test_bad_dissolve_column_lists_candidates_and_numerics(data):
    r = _call("dissolve_layer", file_id=data["tracts"], by="STATEFP")
    assert r["ok"] is False
    assert "STATEFP" in r["error"]
    assert set(r["candidates"]) == {"state", "pop", "label"}
    assert r["numeric_columns"] == ["pop"]                 # the usable statistic/shading column
    assert "STATEFP" not in r["numeric_columns"]
    assert r["hint"]


def test_bad_style_by_column_lists_candidates(data):
    r = _call("dissolve_layer", file_id=data["tracts"], by="state", style_by="population")
    assert r["ok"] is False and "population" in r["error"]
    assert "pop" in r["candidates"] and "pop" in r["numeric_columns"]


def test_bad_statistic_and_bad_output_mode_report_choices(data):
    stat = _call("dissolve_layer", file_id=data["tracts"], by="state", statistic="avg")
    assert stat["ok"] is False and "mean" in stat["candidates"]
    out = _call("geometry_summary", file_id=data["tracts"], output="middle")
    assert out["ok"] is False and "centroids" in out["candidates"]
    how = _call("intersect_layers", left_file_id=data["tracts"], right_file_id=data["city"],
                how="overlap")
    assert how["ok"] is False and "intersection" in how["candidates"]


def test_unknown_file_id_never_raises(data):
    r = _call("clip_layer", target_file_id="file_does_not_exist", clip_file_id=data["city"])
    assert r["ok"] is False and r["error"].startswith("ValueError")
    assert r["hint"]


def test_non_overlapping_inputs_report_zero_not_an_exception(data):
    r = _call("clip_layer", target_file_id=data["tracts"], clip_file_id=data["far"])
    assert r["ok"] is True and r["feature_count"] == 0
    assert r["on_map"] is False and "map_layer" not in r
    assert "extent" in r["note"]


# --- metric correctness: never degrees -------------------------------------------

def test_buffer_is_metric_not_degrees(data, store):
    """A 10 km buffer must be ~10 km wide, not 10 degrees (~1100 km)."""
    r = _call("buffer_layer", file_id=data["points"], distance=10, units="km")
    assert r["ok"] is True and r["feature_count"] == 2
    assert r["distance_m"] == 10000.0
    assert r["buffer_crs"].startswith("EPSG:326")           # a UTM zone, not 4326
    assert r["crs"] == "EPSG:4326"                          # output is lon/lat for the map
    props = [f["properties"] for f in _written(r, store)["features"]]
    # pi * 10km^2 = 314.16 km2; a degree buffer would be ~10^6 km2.
    assert all(p["area_km2"] == pytest.approx(314.16, rel=0.02) for p in props)
    assert r["total_area_km2"] == pytest.approx(628.3, rel=0.02)
    gj = _written(r, store)
    xs = [c[0] for f in gj["features"] for c in f["geometry"]["coordinates"][0]]
    # 10 km near 40N is ~0.117 deg of longitude: the ring must stay within a fraction of a degree.
    assert max(xs) - min(xs) < 1.0, "buffer appears to have been computed in degrees"
    assert r["map_layer"]["style_by"] == "area_km2"


def test_buffer_units_convert_and_degrees_are_refused(data):
    km = _call("buffer_layer", file_id=data["points"], distance=1, units="km")
    m = _call("buffer_layer", file_id=data["points"], distance=1000, units="m")
    mi = _call("buffer_layer", file_id=data["points"], distance=1, units="mi")
    assert km["distance_m"] == m["distance_m"] == 1000.0
    assert mi["distance_m"] == pytest.approx(1609.344)
    assert mi["total_area_km2"] > km["total_area_km2"]
    deg = _call("buffer_layer", file_id=data["points"], distance=10, units="degrees")
    assert deg["ok"] is False and "not a distance" in deg["error"]
    bad = _call("buffer_layer", file_id=data["points"], distance=0, units="km")
    assert bad["ok"] is False and "greater than 0" in bad["error"]
    unknown = _call("buffer_layer", file_id=data["points"], distance=5, units="parsecs")
    assert unknown["ok"] is False and "unknown distance units" in unknown["error"]


def test_buffer_dissolve_merges_overlapping_zones(data):
    apart = _call("buffer_layer", file_id=data["points"], distance=2, units="km", dissolve=True)
    together = _call("buffer_layer", file_id=data["points"], distance=15, units="km",
                     dissolve=True)
    assert apart["feature_count"] == 2                       # 2 km zones stay separate
    assert together["feature_count"] == 1                    # 15 km zones fuse into one
    assert together["dissolved"] is True


def test_simplify_tolerance_is_metres_and_drops_vertices(data):
    r = _call("simplify_layer", file_id=data["jagged"], tolerance_m=500)
    assert r["ok"] is True
    assert r["simplify_crs"].startswith("EPSG:326")          # metric CRS, not 4326
    assert r["tolerance_m"] == 500.0
    assert r["vertices_before"] > r["vertices_after"] > 0
    assert r["vertex_reduction_pct"] > 0
    # A 500 m tolerance read as 500 DEGREES would collapse the polygon entirely.
    assert r["feature_count"] == 1
    tiny = _call("simplify_layer", file_id=data["jagged"], tolerance_m=0.001)
    assert tiny["ok"] is True and tiny["vertices_after"] == tiny["vertices_before"]
    bad = _call("simplify_layer", file_id=data["jagged"], tolerance_m=0)
    assert bad["ok"] is False and "greater than 0" in bad["error"]


def test_simplify_keeps_shared_edges_when_asked(data):
    r = _call("simplify_layer", file_id=data["tracts"], tolerance_m=100, keep_topology=True)
    assert r["ok"] is True and r["keep_topology"] is True
    assert r["feature_count"] == 3
    assert "coverage" in r["method"] or "preserve_topology" in r["method"]


def test_line_layer_measures_length_in_km_not_degrees(data, store):
    r = _call("geometry_summary", file_id=data["road"], output="bbox")
    assert r["ok"] is True
    # 0.3 deg of longitude at 40N is ~25.6 km, definitely not 0.3.
    assert r["summary"]["total_length_km"] == pytest.approx(25.6, rel=0.05)


def test_every_layer_comes_back_in_wgs84(data):
    results = [
        _call("clip_layer", target_file_id=data["tracts"], clip_file_id=data["city"]),
        _call("dissolve_layer", file_id=data["tracts"], by="state"),
        _call("intersect_layers", left_file_id=data["tracts"], right_file_id=data["city"]),
        _call("erase_layer", target_file_id=data["tracts"], erase_file_id=data["city"]),
        _call("buffer_layer", file_id=data["points"], distance=1, units="km"),
        _call("simplify_layer", file_id=data["tracts"], tolerance_m=50),
        _call("geometry_summary", file_id=data["tracts"]),
    ]
    for r in results:
        assert r["ok"] is True, r
        assert r["crs"] == "EPSG:4326"
        layer = r["map_layer"]
        assert layer["render"] in {"heatmap", "choropleth", "points", "shapes"}
        assert layer["url"] and layer["count"] == r["feature_count"]
        if layer["render"] == "choropleth":
            assert layer["style_by"] in r["columns"]


def test_web_mercator_input_is_measured_in_utm_not_in_mercator_metres(store, tmp_path):
    """EPSG:3857 metres are not ground metres: at 40N they are ~1.3x too long."""
    work = tmp_path / "merc"
    work.mkdir()
    pts = gpd.GeoDataFrame({"n": [1]}, geometry=[Point(-88.25, 40.05)],
                           crs="EPSG:4326").to_crs("EPSG:3857")
    path = work / "merc_points.geojson"
    pts.to_file(path, driver="GeoJSON")
    r = _call("buffer_layer", file_id=str(path), distance=10, units="km")
    assert r["ok"] is True
    assert r["buffer_crs"].startswith("EPSG:326"), r["buffer_crs"]
    assert r["total_area_km2"] == pytest.approx(314.16, rel=0.02)
    assert "cos(latitude)" in r["note"]


def test_inputs_in_different_crs_are_aligned(store, tmp_path, data):
    """A 3857 target and a 4326 clip must still overlap after alignment."""
    work = tmp_path / "mixed"
    work.mkdir()
    tracts = gpd.read_file(data["tracts"]).to_crs("EPSG:3857")
    path = work / "tracts_3857.geojson"
    tracts.to_file(path, driver="GeoJSON")
    r = _call("clip_layer", target_file_id=str(path), clip_file_id=data["city"])
    assert r["ok"] is True and r["feature_count"] == 3
    assert r["crs"] == "EPSG:4326"


def test_accepts_csv_with_coordinates(store, tmp_path):
    csv = tmp_path / "sites.csv"
    csv.write_text("site,latitude,longitude\na,40.05,-88.25\nb,40.05,-88.05\n", encoding="utf-8")
    r = _call("buffer_layer", file_id=str(csv), distance=1, units="km")
    assert r["ok"] is True and r["feature_count"] == 2
    assert r["crs"] == "EPSG:4326"


def test_accepts_uploaded_shapefile_with_separate_sidecars(store, tmp_path):
    """The .shp and its sidecars arrive as SEPARATE uploads; staging must reunite them."""
    import glob

    from werkzeug.datastructures import FileStorage

    from agent_runtime.file_store import save_uploaded_file

    work = tmp_path / "shp"
    work.mkdir()
    gpd.GeoDataFrame({"state": ["IL", "IN"], "pop": [10, 20]},
                     geometry=[box(-88.3, 40.0, -88.2, 40.1), box(-88.2, 40.0, -88.1, 40.1)],
                     crs="EPSG:4326").to_file(work / "tracts.shp")

    def upload(path):
        with open(path, "rb") as fh:
            return save_uploaded_file(FileStorage(stream=fh, filename=Path(path).name))["file_id"]

    ids = {Path(c).suffix: upload(c) for c in sorted(glob.glob(str(work / "tracts.*")))}
    tools = {t.name: t for t in make_overlay_tools(list(ids.values()))}
    r = json.loads(tools["dissolve_layer"].invoke({"file_id": ids[".shp"], "by": "state"}))
    assert r["ok"] is True and r["feature_count"] == 2
    assert r["map_layer"]["style_by"] == "pop"
    assert r["filename"] == "tracts_dissolved.geojson"     # named from the ORIGINAL upload name


def test_map_layer_descriptor_is_accepted_by_build_map_layer(data):
    """The descriptor must survive the real SSE builder the client consumes."""
    from agent_runtime.map_layers import build_map_layer

    # statistic="mean" so the two dissolved groups DIFFER: the fixture's populations sum to
    # 300 for both IL (100+200) and IN (300), which makes a summed choropleth genuinely flat —
    # and layer_qa now refuses to ship a flat choropleth as a choropleth.
    r = _call("dissolve_layer", file_id=data["tracts"], by="state", statistic="mean",
              name="pop_by_state")
    built = build_map_layer("dissolve_layer", json.dumps(r))
    assert built is not None
    assert built["kind"] == "map_layer" and built["render"] == "choropleth"
    assert built["style_by"] == "pop" and built["source"] == "analysis"
    assert built["url"] == r["download_url"]
    assert r["filename"] == "pop_by_state.geojson"          # caller-supplied name is honoured
