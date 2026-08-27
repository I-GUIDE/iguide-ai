"""Tests for the aggregation / proximity / pattern tools.

Tiny synthetic GeoDataFrames only — no network, no fixtures on disk beyond ``tmp_path``. The
recurring assertions are the contract ones: a happy path returns ``ok=true`` with a ``map_layer``
whose ``style_by`` really exists in the written output, a bad column fails soft WITH candidates,
and every metric number (grid cell size, DBSCAN eps, nearest distance) is metres — not degrees.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

gpd = pytest.importorskip("geopandas")
from shapely.geometry import Point, box  # noqa: E402


def _tools():
    from agent_runtime.analysis_aggregate_tools import make_aggregate_tools
    return {t.name: t for t in make_aggregate_tools()}


def _call(tool_name: str, **kwargs):
    """Invoke a tool and parse its JSON string result."""
    return json.loads(_tools()[tool_name].func(**kwargs))


def _read_output(file_id: str):
    from agent_runtime.file_store import resolve_file_id
    return gpd.read_file(resolve_file_id(file_id))


@pytest.fixture(autouse=True)
def _store(monkeypatch, tmp_path):
    """Every artifact lands in a throwaway file store."""
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "store"))


@pytest.fixture()
def areas(tmp_path):
    """Two adjacent 0.1-degree squares west/east of -88.20."""
    path = tmp_path / "wards.geojson"
    gpd.GeoDataFrame(
        {"name": ["west", "east"]},
        geometry=[box(-88.30, 40.05, -88.20, 40.15), box(-88.20, 40.05, -88.10, 40.15)],
        crs="EPSG:4326",
    ).to_file(path, driver="GeoJSON")
    return str(path)


@pytest.fixture()
def incidents(tmp_path):
    """Three points in the west ward, one in the east, one far outside both."""
    path = tmp_path / "incidents.geojson"
    gpd.GeoDataFrame(
        {"val": [10, 20, 30, 100, 7], "cat": ["a", "a", "b", "b", "b"],
         "when": ["2024-01-05", "2024-01-19", "2024-02-02", "2024-02-20", "2024-02-27"]},
        geometry=[Point(-88.28, 40.10), Point(-88.26, 40.11), Point(-88.24, 40.12),
                  Point(-88.15, 40.10), Point(-87.00, 41.00)],
        crs="EPSG:4326",
    ).to_file(path, driver="GeoJSON")
    return str(path)


# --- factory ---------------------------------------------------------------------

def test_factory_shape():
    tools = _tools()
    assert set(tools) == {"count_points_in_areas", "aggregate_to_grid", "nearest_distance",
                          "cluster_points", "summary_statistics", "select_by_attribute"}
    assert all(getattr(t, "metadata", {}).get("category") == "geo" for t in tools.values())
    assert all((t.description or "").strip() for t in tools.values())


# --- count_points_in_areas: the entry-level workhorse ----------------------------

def test_count_points_in_areas_choropleth_and_csv(areas, incidents):
    from agent_runtime.map_layers import build_map_layer

    raw = _tools()["count_points_in_areas"].func(
        points_file_id=incidents, areas_file_id=areas, name="incidents_per_ward")
    res = json.loads(raw)
    assert res["ok"] is True, res
    assert res["feature_count"] == 2
    assert res["points_total"] == 5 and res["points_matched"] == 4 and res["points_unmatched"] == 1

    out = _read_output(res["file_id"])
    style_by = res["map_layer"]["style_by"]
    assert style_by == "point_count"
    assert style_by in out.columns                     # a choropleth must style by a real column
    assert out.crs.to_epsg() == 4326                   # web maps need lon/lat
    counts = dict(zip(out["name"], out["point_count"]))
    assert counts == {"west": 3, "east": 1}

    ml = res["map_layer"]
    assert ml["render"] == "choropleth" and ml["url"] and ml["count"] == 2
    layer = build_map_layer("count_points_in_areas", raw)   # -> the map_layer SSE event
    assert layer["kind"] == "map_layer" and layer["style_by"] == "point_count"
    assert layer["url"] == ml["url"] and layer["label"] == "incidents per ward"

    csv_text = Path(_csv_path(res)).read_text(encoding="utf-8")
    assert "point_count" in csv_text and "west" in csv_text


def _csv_path(res):
    from agent_runtime.file_store import resolve_file_id
    return resolve_file_id(res["csv"]["file_id"])


def test_artifacts_are_named_for_what_produced_them(areas, incidents, dense_points):
    """Several runs over one input must not all land as `<input>.geojson` in the download list."""
    counted = _call("count_points_in_areas", points_file_id=incidents, areas_file_id=areas)
    grid = _call("aggregate_to_grid", points_file_id=dense_points, cell_km=1.0, shape="hex")
    stats = _call("summary_statistics", file_id=incidents)
    assert counted["filename"] == "wards_by_area.geojson"
    assert counted["csv"]["filename"] == "wards_by_area_table.csv"
    assert grid["filename"] == "dense_hex_grid.geojson"
    assert stats["csv"]["filename"] == "incidents_summary.csv"
    assert stats["chart"]["filename"] == "incidents_summary.png"


def test_count_points_in_areas_statistic_needs_and_uses_a_value_column(areas, incidents):
    res = _call("count_points_in_areas", points_file_id=incidents, areas_file_id=areas,
                value_column="val", statistic="mean")
    assert res["ok"] is True and res["column"] == "mean_val"
    out = _read_output(res["file_id"])
    assert res["map_layer"]["style_by"] == "mean_val" and "mean_val" in out.columns
    means = dict(zip(out["name"], out["mean_val"]))
    assert means["west"] == pytest.approx(20.0) and means["east"] == pytest.approx(100.0)
    assert res["class_breaks"]                        # classes reported for the legend


def test_count_points_in_areas_bad_column_lists_numeric_candidates(areas, incidents):
    res = _call("count_points_in_areas", points_file_id=incidents, areas_file_id=areas,
                value_column="popultaion", statistic="sum")
    assert res["ok"] is False
    assert res["error"].startswith("ValueError:")
    assert "val" in res["numeric_columns"]             # candidates, not a truncated column dump
    assert res["hint"]


def test_count_points_in_areas_chains_without_clobbering_the_first_count(areas, incidents,
                                                                        tmp_path):
    """Aggregating a SECOND layer into the first run's output must keep both columns —
    overwriting `point_count` in place would silently destroy the result being compared."""
    from agent_runtime.file_store import resolve_file_id

    first = _call("count_points_in_areas", points_file_id=incidents, areas_file_id=areas)
    other = tmp_path / "other.geojson"
    gpd.GeoDataFrame({"v": [1]}, geometry=[Point(-88.15, 40.12)], crs="EPSG:4326").to_file(
        other, driver="GeoJSON")
    second = _call("count_points_in_areas", points_file_id=str(other),
                   areas_file_id=str(resolve_file_id(first["file_id"])))
    assert second["ok"] is True and second["count_column"] == "point_count_2"
    out = _read_output(second["file_id"])
    assert {"point_count", "point_count_2"} <= set(out.columns)
    assert second["map_layer"]["style_by"] == "point_count_2"
    assert dict(zip(out["name"], out["point_count"])) == {"west": 3, "east": 1}
    assert dict(zip(out["name"], out["point_count_2"])) == {"west": 0, "east": 1}


def test_count_points_in_areas_rejects_swapped_arguments(incidents):
    res = _call("count_points_in_areas", points_file_id=incidents, areas_file_id=incidents)
    assert res["ok"] is False and "not polygons" in res["error"]
    assert "nearest_distance" in res["hint"]           # points-to-points has its own tool


def test_count_points_in_areas_unsupported_statistic_lists_the_supported_ones(areas, incidents):
    res = _call("count_points_in_areas", points_file_id=incidents, areas_file_id=areas,
                statistic="mode")
    assert res["ok"] is False and "median" in res["hint"]


# --- aggregate_to_grid: metres, never degrees -----------------------------------

def _cluster_of(n, lon, lat, step=0.0005):
    return [Point(lon + i * step, lat + i * step) for i in range(n)]


@pytest.fixture()
def dense_points(tmp_path):
    path = tmp_path / "dense.geojson"
    pts = _cluster_of(20, -88.24, 40.11)
    gpd.GeoDataFrame({"val": list(range(20))}, geometry=pts, crs="EPSG:4326").to_file(
        path, driver="GeoJSON")
    return str(path)


@pytest.mark.parametrize("shape", ["hex", "square"])
def test_aggregate_to_grid_cells_are_kilometres_not_degrees(dense_points, shape):
    res = _call("aggregate_to_grid", points_file_id=dense_points, cell_km=1.0, shape=shape)
    assert res["ok"] is True, res
    assert res["points_binned"] == 20 and res["cells"] >= 1
    # The metric work happened in a PROJECTED crs, and the output is lon/lat.
    assert res["metric_crs"].startswith("EPSG:") and res["metric_crs"] != "EPSG:4326"
    assert res["crs"] == "EPSG:4326"

    out = _read_output(res["file_id"])
    assert res["map_layer"]["style_by"] == "point_count"
    assert "point_count" in out.columns and out.crs.to_epsg() == 4326
    assert out["point_count"].sum() == 20

    # A 1 km cell is ~0.01 degrees wide near latitude 40. Had the cell size been applied in
    # degrees, one cell would span 1000 degrees and swallow the planet.
    minx, miny, maxx, maxy = out.total_bounds
    assert 0.001 < (maxx - minx) < 0.1 and 0.001 < (maxy - miny) < 0.1
    # ...and its true area really is ~1 km^2 (0.866 km^2 for a hex of 1 km across).
    expected = 1.0 if shape == "square" else pytest.approx(0.866, rel=0.01)
    area_km2 = out.to_crs(out.estimate_utm_crs()).area.iloc[0] / 1e6
    assert area_km2 == pytest.approx(expected, rel=0.02)
    assert res["cell_area_km2"] == pytest.approx(area_km2, rel=0.02)


def test_aggregate_to_grid_power_user_knobs(dense_points):
    coarse = _call("aggregate_to_grid", points_file_id=dense_points, cell_km=5.0, shape="hex")
    fine = _call("aggregate_to_grid", points_file_id=dense_points, cell_km=0.2, shape="hex")
    assert coarse["ok"] and fine["ok"]
    assert coarse["cells"] <= fine["cells"]            # bigger cells cannot mean more cells
    valued = _call("aggregate_to_grid", points_file_id=dense_points, cell_km=1.0,
                   value_column="val", statistic="sum")
    assert valued["column"] == "sum_val"
    assert "sum_val" in _read_output(valued["file_id"]).columns


def test_aggregate_to_grid_cells_tile_the_points_they_counted(tmp_path):
    """Every point must fall inside exactly one returned cell — the hex lattice assignment is
    only correct if the cells it names actually cover the points it counted."""
    import numpy as np

    rng = np.random.default_rng(0)
    pts = [Point(-88.3 + x, 40.0 + y)
           for x, y in zip(rng.random(500) * 0.2, rng.random(500) * 0.2)]
    path = tmp_path / "spread.geojson"
    gpd.GeoDataFrame({"i": range(500)}, geometry=pts, crs="EPSG:4326").to_file(
        path, driver="GeoJSON")
    source = gpd.read_file(path)
    for shape in ("hex", "square"):
        res = _call("aggregate_to_grid", points_file_id=str(path), cell_km=0.8, shape=shape)
        assert res["ok"] is True and res["points_binned"] == 500
        cells = _read_output(res["file_id"])
        assert cells["point_count"].sum() == 500
        hit = gpd.sjoin(source, cells[["cell_id", "geometry"]], how="left", predicate="within")
        assert hit["cell_id"].isna().sum() == 0, f"{shape}: a counted point lies outside its cell"
        assert hit.index.duplicated().sum() == 0, f"{shape}: cells overlap"


def test_aggregate_to_grid_rejects_nonsense_cell_size(dense_points):
    for bad in (0, -3, "wide"):
        res = _call("aggregate_to_grid", points_file_id=dense_points, cell_km=bad)
        assert res["ok"] is False and "cell_km" in res["error"]
    res = _call("aggregate_to_grid", points_file_id=dense_points, shape="triangle")
    assert res["ok"] is False and "hex" in res["hint"]


# --- nearest_distance: real metres ----------------------------------------------

@pytest.fixture()
def schools(tmp_path):
    path = tmp_path / "schools.geojson"
    gpd.GeoDataFrame(
        {"name": ["near", "far"]},
        geometry=[Point(-88.00, 40.00), Point(-88.00, 40.90)],
        crs="EPSG:4326",
    ).to_file(path, driver="GeoJSON")
    return str(path)


@pytest.fixture()
def hospital(tmp_path):
    path = tmp_path / "hospital.geojson"
    gpd.GeoDataFrame({"hname": ["general"]}, geometry=[Point(-88.00, 40.00)],
                     crs="EPSG:4326").to_file(path, driver="GeoJSON")
    return str(path)


def test_nearest_distance_is_metres_not_degrees(schools, hospital):
    res = _call("nearest_distance", from_file_id=schools, to_file_id=hospital,
                to_label_column="hname")
    assert res["ok"] is True, res
    assert res["metric_crs"].startswith("EPSG:") and res["metric_crs"] != "EPSG:4326"
    out = _read_output(res["file_id"]).sort_values("distance_m")
    assert res["map_layer"]["style_by"] == "distance_m"
    assert "distance_m" in out.columns and res["map_layer"]["render"] == "points"
    d = sorted(out["distance_m"].tolist())
    assert d[0] == pytest.approx(0.0, abs=1.0)
    # 0.9 degrees of latitude is ~100 km. A degree-space distance would read 0.9.
    assert 98_000 < d[1] < 102_000
    assert out["distance_km"].max() == pytest.approx(d[1] / 1000.0, rel=1e-3)
    assert set(out["nearest_label"]) == {"general"}
    summary = res["distance_summary_km"]
    assert 98 < summary["max"] < 102 and summary["min"] == pytest.approx(0.0, abs=0.01)


def test_nearest_distance_max_km_leaves_far_features_unmatched(schools, hospital):
    res = _call("nearest_distance", from_file_id=schools, to_file_id=hospital, max_km=50,
                units="mi")
    assert res["ok"] is True and res["matched"] == 1 and res["unmatched"] == 1
    assert res["units"] == "mi" and res["distance_summary_mi"]["max"] == pytest.approx(0.0, abs=0.01)
    assert "distance_mi" in _read_output(res["file_id"]).columns


def test_nearest_distance_bad_label_column_lists_candidates(schools, hospital):
    res = _call("nearest_distance", from_file_id=schools, to_file_id=hospital,
                to_label_column="nmae")
    assert res["ok"] is False and "hname" in res["candidate_columns"]


# --- cluster_points: DBSCAN ------------------------------------------------------

@pytest.fixture()
def two_clusters(tmp_path):
    """Two tight groups ~8 km apart — one cluster each in metres, a single blob in degrees."""
    path = tmp_path / "clustered.geojson"
    pts = _cluster_of(6, -88.30, 40.10, step=0.0003) + _cluster_of(6, -88.20, 40.10, step=0.0003)
    gpd.GeoDataFrame({"i": list(range(12))}, geometry=pts, crs="EPSG:4326").to_file(
        path, driver="GeoJSON")
    return str(path)


def test_cluster_points_uses_metric_eps(two_clusters):
    res = _call("cluster_points", file_id=two_clusters, eps_km=0.5, min_samples=3)
    assert res["ok"] is True, res
    # eps=0.5 km separates the groups; eps interpreted as 0.5 DEGREES would merge them into one.
    assert res["n_clusters"] == 2 and res["n_noise"] == 0
    assert res["metric_crs"].startswith("EPSG:") and res["metric_crs"] != "EPSG:4326"
    out = _read_output(res["file_id"])
    assert res["map_layer"]["style_by"] == "cluster" and "cluster" in out.columns
    assert res["map_layer"]["render"] == "points"
    assert sorted(set(out["cluster"])) == [0, 1]
    summary = Path(_csv_path(res)).read_text(encoding="utf-8").strip().splitlines()
    assert summary[0].startswith("cluster,n,centroid_lon,centroid_lat,radius_km")
    assert len(summary) == 3                            # header + one row per cluster
    assert {c["n"] for c in res["clusters"]} == {6}


def test_cluster_points_noise_and_drop_noise(two_clusters):
    res = _call("cluster_points", file_id=two_clusters, eps_km=0.5, min_samples=8)
    assert res["ok"] is True and res["n_clusters"] == 0 and res["n_noise"] == 12
    assert any("min_samples" in n for n in res["notes"])
    dropped = _call("cluster_points", file_id=two_clusters, eps_km=0.5, min_samples=8,
                    drop_noise=True)
    assert dropped["ok"] is False and "noise" in dropped["error"]


def test_cluster_points_degrades_clearly_without_sklearn(monkeypatch, two_clusters):
    monkeypatch.setitem(sys.modules, "sklearn.cluster", None)   # -> ImportError on import
    res = _call("cluster_points", file_id=two_clusters)
    assert res["ok"] is False and "scikit-learn" in res["hint"]
    assert "aggregate_to_grid" in res["hint"]                   # a usable alternative


def test_cluster_points_rejects_bad_parameters(two_clusters):
    assert _call("cluster_points", file_id=two_clusters, eps_km=0)["ok"] is False
    assert _call("cluster_points", file_id=two_clusters, min_samples=0)["ok"] is False


# --- summary_statistics ----------------------------------------------------------

def test_summary_statistics_describes_every_numeric_column(incidents):
    res = _call("summary_statistics", file_id=incidents)
    assert res["ok"] is True, res
    assert res["filename"].endswith(".csv") and res["download_url"]
    stats = res["summary"]["columns"]["val"]
    assert stats["count"] == 5 and stats["min"] == 7 and stats["max"] == 100
    assert stats["mean"] == pytest.approx(33.4) and stats["median"] == 20
    assert res["chart"]["filename"].endswith(".png")
    from agent_runtime.file_store import resolve_file_id
    assert resolve_file_id(res["chart"]["file_id"]).stat().st_size > 0


def test_summary_statistics_group_by(incidents):
    res = _call("summary_statistics", file_id=incidents, column="val", by="cat")
    assert res["ok"] is True and res["summary"]["group_count"] == 2
    groups = {g["cat"]: g for g in res["summary"]["groups"]}
    assert groups["a"]["n"] == 2 and groups["a"]["mean_val"] == pytest.approx(15.0)
    assert groups["b"]["n"] == 3
    csv_text = Path(_csv_path(res)).read_text(encoding="utf-8")
    assert "mean_val" in csv_text


def test_summary_statistics_period_buckets_a_date_column(incidents):
    res = _call("summary_statistics", file_id=incidents, column="val", by="when", period="month")
    assert res["ok"] is True and res["summary"]["group_count"] == 2
    assert {g["when"] for g in res["summary"]["groups"]} == {"2024-01", "2024-02"}
    bad = _call("summary_statistics", file_id=incidents, column="val", by="cat", period="fortnight")
    assert bad["ok"] is False and "month" in bad["hint"]


def test_summary_statistics_bad_columns_list_candidates(incidents):
    res = _call("summary_statistics", file_id=incidents, column="valu")
    assert res["ok"] is False and "val" in res["numeric_columns"]
    grouped = _call("summary_statistics", file_id=incidents, by="categry")
    assert grouped["ok"] is False and "cat" in grouped["candidate_columns"]


# --- an entry-level upload: a spreadsheet of coordinates -------------------------

def test_a_plain_csv_of_coordinates_is_aggregated_like_any_layer(tmp_path, areas):
    """The no-tuning path for someone who uploaded a spreadsheet: read_vector derives point
    geometry from the latitude/longitude columns, so the workhorse tools just work."""
    csv = tmp_path / "sightings.csv"
    csv.write_text(
        "site,latitude,longitude,count\n"
        "a,40.10,-88.28,4\n"
        "b,40.11,-88.26,6\n"
        "c,40.10,-88.15,2\n",
        encoding="utf-8",
    )
    res = _call("count_points_in_areas", points_file_id=str(csv), areas_file_id=areas,
                value_column="count", statistic="sum")
    assert res["ok"] is True, res
    out = _read_output(res["file_id"])
    sums = dict(zip(out["name"], out["sum_count"]))
    assert sums == {"west": 10, "east": 2}
    assert res["map_layer"]["style_by"] == "sum_count"

    grid = _call("aggregate_to_grid", points_file_id=str(csv), cell_km=2.0)
    assert grid["ok"] is True and grid["points_binned"] == 3


# --- soft failure everywhere ------------------------------------------------------

def test_no_tool_raises_on_a_missing_file():
    tools = _tools()
    calls = {
        "count_points_in_areas": {"points_file_id": "nope", "areas_file_id": "nope"},
        "aggregate_to_grid": {"points_file_id": "nope"},
        "nearest_distance": {"from_file_id": "nope", "to_file_id": "nope"},
        "cluster_points": {"file_id": "nope"},
        "summary_statistics": {"file_id": "nope"},
    }
    for tool_name, kwargs in calls.items():
        res = json.loads(tools[tool_name].func(**kwargs))
        assert res["ok"] is False, tool_name
        assert res["error"] and res["hint"], tool_name

# --- select_by_attribute: the missing fundamental ---------------------------------
# Asked to "buffer the busiest cell", the analyze peer had no way to isolate one feature, so it
# buffered all 708 grid cells (4,504 overlapping polygons) and called it the busiest cell.

@pytest.fixture()
def grid(tmp_path):
    """Four cells carrying a point_count, the shape aggregate_to_grid emits."""
    path = tmp_path / "grid.geojson"
    gpd.GeoDataFrame(
        {"cell_id": ["a", "b", "c", "d"], "point_count": [3, 91, 12, 91],
         "ward": ["12", "12", "7", "7"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(0, 1, 1, 2), box(1, 1, 2, 2)],
        crs="EPSG:4326",
    ).to_file(path, driver="GeoJSON")
    return str(path)


def test_top_n_isolates_a_single_feature(grid):
    out = _call("select_by_attribute", file_id=grid, column="point_count", top_n=1,
                name="busiest")

    assert out["ok"] is True
    assert out["feature_count"] == 1 and out["features_in"] == 4
    assert out["criterion"] == "top 1 by point_count"
    # It must arrive as its own map layer, feedable straight into buffer_layer.
    assert out["on_map"] is True and out["map_layer"]["count"] == 1
    assert out["csv"]["file_id"]
    assert int(_read_output(out["file_id"])["point_count"].iloc[0]) == 91


def test_ascending_takes_the_smallest(grid):
    out = _call("select_by_attribute", file_id=grid, column="point_count", top_n=1,
                ascending=True)
    assert int(_read_output(out["file_id"])["point_count"].iloc[0]) == 3


def test_comparison_and_membership_select_the_right_rows(grid):
    assert _call("select_by_attribute", file_id=grid, column="point_count",
                 op=">=", value=12)["feature_count"] == 3
    # A value passed as a STRING against an int column must still compare numerically.
    assert _call("select_by_attribute", file_id=grid, column="point_count",
                 op=">=", value="12")["feature_count"] == 3
    assert _call("select_by_attribute", file_id=grid, column="ward",
                 op="in", value=["7"])["feature_count"] == 2
    assert _call("select_by_attribute", file_id=grid, column="point_count",
                 op="between", value=[10, 50])["feature_count"] == 1
    assert _call("select_by_attribute", file_id=grid, column="ward",
                 op="==", value="12")["feature_count"] == 2


def test_an_empty_selection_reports_the_real_range_instead_of_an_empty_layer(grid):
    out = _call("select_by_attribute", file_id=grid, column="point_count", op=">", value=10_000)

    assert out["ok"] is False
    assert out["matched"] == 0
    assert out["range"] == [3, 91]          # says what the column actually holds
    assert "empty layer" in out["error"]


def test_bad_column_and_bad_op_name_the_alternatives(grid):
    col = _call("select_by_attribute", file_id=grid, column="nope", top_n=1)
    assert col["ok"] is False and "point_count" in col["candidate_columns"]

    op = _call("select_by_attribute", file_id=grid, column="point_count", op="~=", value=1)
    assert op["ok"] is False and ">=" in op["accepted_ops"]


def test_missing_criterion_explains_both_ways_to_select(grid):
    out = _call("select_by_attribute", file_id=grid, column="point_count")
    assert out["ok"] is False
    assert "top_n" in out["error"] and "top_n" in out["hint"]
