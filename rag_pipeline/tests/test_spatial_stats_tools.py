"""Tests for the PySAL / GeoDa spatial-statistics tools.

Builds a real 8x8 polygon lattice with a KNOWN spatial structure -- one column with a strong
north-south gradient (so it must come back significantly clustered) and one that is spatially
random (so it must NOT) -- and runs the actual libpysal/esda/spreg/pygeoda code over it. No
mocking of the statistics: a mocked Moran's I would pass whatever the sign convention was.

Skipped when the spatial-stats stack is absent, exactly as the geo tools' tests skip without
geopandas.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

gpd = pytest.importorskip("geopandas")
pytest.importorskip("libpysal")
pytest.importorskip("esda")
np = pytest.importorskip("numpy")
from shapely.geometry import Point, Polygon, box  # noqa: E402
from werkzeug.datastructures import FileStorage  # noqa: E402

SIDE = 8  # 64 cells: enough for permutation inference, small enough to stay fast


def _tools(file_ids=None):
    from agent_runtime.analysis_spatial_stats_tools import make_spatial_stats_tools
    return {t.name: t for t in make_spatial_stats_tools(default_input_file_ids=file_ids)}


def _upload(path):
    from agent_runtime.file_store import save_uploaded_file
    with open(path, "rb") as fh:
        return save_uploaded_file(FileStorage(stream=fh, filename=Path(path).name))["file_id"]


def _lattice():
    """A square lattice whose `gradient` column rises with latitude and whose `noise` doesn't."""
    rng = np.random.default_rng(11)
    polys, rows = [], []
    for i in range(SIDE):
        for j in range(SIDE):
            polys.append(Polygon([(j, i), (j + 1, i), (j + 1, i + 1), (j, i + 1)]))
            rows.append({"name": f"cell_{i}_{j}", "row": i, "col": j})
    n = SIDE * SIDE
    lat = np.array([r["row"] for r in rows], dtype="float64")
    x1 = rng.normal(0, 1, n)
    frame = {
        "name": [r["name"] for r in rows],
        # strong, unambiguous spatial structure
        "gradient": lat * 10.0 + rng.normal(0, 1.0, n),
        # no spatial structure at all
        "noise": rng.normal(50, 10, n),
        "x1": x1,
        # y = x1 effect + a spatial component, so a spatial model should be preferred over OLS
        "y": 2.0 + 1.5 * x1 + 0.9 * lat + rng.normal(0, 1.0, n),
        "pop": rng.integers(100, 900, n).astype("float64"),
    }
    return gpd.GeoDataFrame(frame, geometry=polys, crs="EPSG:4326")


@pytest.fixture()
def lattice(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "store"))
    path = tmp_path / "lattice.geojson"
    _lattice().to_file(path, driver="GeoJSON")
    return _upload(path)


@pytest.fixture()
def points(monkeypatch, tmp_path):
    """A POINT layer, to prove contiguity is refused rather than silently mis-answered."""
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "store"))
    rng = np.random.default_rng(5)
    gdf = gpd.GeoDataFrame(
        {"v": rng.normal(0, 1, 30)},
        geometry=[Point(-88 + rng.uniform(0, 1), 40 + rng.uniform(0, 1)) for _ in range(30)],
        crs="EPSG:4326")
    path = tmp_path / "pts.geojson"
    gdf.to_file(path, driver="GeoJSON")
    return _upload(path)


# --- registration ---------------------------------------------------------------------


def test_factory_shape():
    tools = _tools()
    assert set(tools) == {"spatial_weights", "global_spatial_autocorrelation",
                          "local_moran_lisa", "local_getis_ord", "moran_scatterplot",
                          "spatial_regression", "regionalize"}
    assert all(getattr(t, "metadata", {}).get("category") == "geo" for t in tools.values())
    # The description is the only thing routing the model here; each must name its own job.
    assert all(len(t.description or "") > 120 for t in tools.values())


# --- weights --------------------------------------------------------------------------


def test_weights_queen_connectivity(lattice):
    r = json.loads(_tools()["spatial_weights"].invoke({"file_id": lattice}))
    assert r["ok"] is True
    conn = r["connectivity"]
    assert conn["n"] == SIDE * SIDE
    assert conn["islands"] == 0
    # Queen contiguity on a lattice: corner cells have 3 neighbours, interior cells 8.
    assert conn["min_neighbors"] == 3
    assert conn["max_neighbors"] == 8
    assert r["download_url"] and str(r["filename"]).endswith(".gal")
    assert sum(r["neighbor_count_histogram"].values()) == SIDE * SIDE


def test_weights_rook_is_sparser_than_queen(lattice):
    tools = _tools()
    queen = json.loads(tools["spatial_weights"].invoke({"file_id": lattice}))
    rook = json.loads(tools["spatial_weights"].invoke(
        {"file_id": lattice, "weights": "rook"}))
    assert rook["ok"] is True
    # Rook excludes diagonals, so interior cells have 4 neighbours rather than 8.
    assert rook["connectivity"]["max_neighbors"] == 4
    assert rook["connectivity"]["mean_neighbors"] < queen["connectivity"]["mean_neighbors"]


def test_weights_knn_gives_every_feature_k_neighbors(lattice):
    r = json.loads(_tools()["spatial_weights"].invoke(
        {"file_id": lattice, "weights": "knn", "k": 4}))
    assert r["ok"] is True
    assert r["connectivity"]["min_neighbors"] == r["connectivity"]["max_neighbors"] == 4
    assert str(r["filename"]).endswith(".gwt")


def test_weights_rejects_unknown_scheme(lattice):
    r = json.loads(_tools()["spatial_weights"].invoke(
        {"file_id": lattice, "weights": "telepathy"}))
    assert r["ok"] is False and "telepathy" in r["error"]


def test_contiguity_on_points_is_refused_with_a_route_out(points):
    """Queen contiguity is undefined for points; the error must say what to use instead."""
    r = json.loads(_tools()["spatial_weights"].invoke({"file_id": points}))
    assert r["ok"] is False
    assert "knn" in r["error"].lower() or "knn" in str(r.get("hint", "")).lower()


def test_knn_works_on_points(points):
    r = json.loads(_tools()["spatial_weights"].invoke(
        {"file_id": points, "weights": "knn", "k": 3}))
    assert r["ok"] is True and r["connectivity"]["max_neighbors"] == 3


# --- global autocorrelation -----------------------------------------------------------


def test_global_detects_the_planted_gradient(lattice):
    r = json.loads(_tools()["global_spatial_autocorrelation"].invoke(
        {"file_id": lattice, "column": "gradient"}))
    assert r["ok"] is True
    moran = r["results"]["morans_i"]
    assert moran["statistic"] > 0.5           # strong positive autocorrelation
    assert moran["p_value"] < 0.05
    assert "POSITIVE" in moran["interpretation"]
    # Geary's C is INVERTED: clustering pushes it BELOW its expectation of 1.
    geary = r["results"]["gearys_c"]
    assert geary["statistic"] < 1.0 and geary["p_value"] < 0.05
    assert "clustered" in geary["interpretation"]
    assert "local_moran_lisa" in r["next_step"]


def test_global_finds_nothing_in_spatial_noise(lattice):
    r = json.loads(_tools()["global_spatial_autocorrelation"].invoke(
        {"file_id": lattice, "column": "noise"}))
    assert r["ok"] is True
    assert r["results"]["morans_i"]["p_value"] > 0.05
    assert "no statistically significant" in r["results"]["morans_i"]["interpretation"]


def test_global_lists_candidates_for_a_bad_column(lattice):
    r = json.loads(_tools()["global_spatial_autocorrelation"].invoke(
        {"file_id": lattice, "column": "nope"}))
    assert r["ok"] is False
    assert "gradient" in r["numeric_columns"] and "noise" in r["numeric_columns"]


def test_getis_g_skipped_for_a_signed_variable(lattice):
    """Getis-Ord G is only defined for non-negative data; x1 is centred on zero."""
    r = json.loads(_tools()["global_spatial_autocorrelation"].invoke(
        {"file_id": lattice, "column": "x1"}))
    assert r["ok"] is True
    assert "getis_ord_g" not in r["results"]
    assert any("non-negative" in n for n in r["notes"])


# --- LISA -----------------------------------------------------------------------------


def test_lisa_classifies_and_delivers_a_categorical_layer(lattice):
    r = json.loads(_tools()["local_moran_lisa"].invoke(
        {"file_id": lattice, "column": "gradient"}))
    assert r["ok"] is True
    assert r["features_analyzed"] == SIDE * SIDE
    # A north-south gradient must produce both a hot end and a cold end.
    assert r["class_counts"].get("High-High", 0) > 0
    assert r["class_counts"].get("Low-Low", 0) > 0
    assert sum(r["class_counts"].values()) == SIDE * SIDE

    layer = r["map_layer"]
    # The whole point of render="categories": the client's numeric ramp would render a
    # string-valued style column as one flat colour.
    assert layer["render"] == "categories"
    assert layer["style_by"] == "lisa_class"
    assert layer["url"]
    labels = {entry["label"] for entry in layer["legend"]}
    assert labels == set(r["class_counts"])          # legend covers exactly what was assigned
    assert all(len(entry["color"]) == 4 for entry in layer["legend"])
    assert r["csv"]["download_url"]


def test_lisa_significance_level_is_honoured(lattice):
    tools = _tools()
    loose = json.loads(tools["local_moran_lisa"].invoke(
        {"file_id": lattice, "column": "gradient", "significance": 0.10}))
    strict = json.loads(tools["local_moran_lisa"].invoke(
        {"file_id": lattice, "column": "gradient", "significance": 0.01}))
    assert loose["significant_features"] >= strict["significant_features"]


def test_lisa_rejects_an_impossible_significance(lattice):
    r = json.loads(_tools()["local_moran_lisa"].invoke(
        {"file_id": lattice, "column": "gradient", "significance": 1.5}))
    assert r["ok"] is False and "significance" in r["error"]


# --- Getis-Ord Gi* --------------------------------------------------------------------


def test_getis_ord_finds_hot_and_cold_ends(lattice):
    r = json.loads(_tools()["local_getis_ord"].invoke(
        {"file_id": lattice, "column": "gradient"}))
    assert r["ok"] is True and r["statistic"] == "Gi*"
    assert any(k.startswith("Hot") for k in r["class_counts"])
    assert any(k.startswith("Cold") for k in r["class_counts"])
    assert r["map_layer"]["render"] == "categories"
    assert r["map_layer"]["style_by"] == "hotspot_class"
    # hottest is sorted by descending z, so it must not begin below the coldest.
    assert r["hottest"][0]["gi_z"] >= r["coldest"][0]["gi_z"]


def test_getis_ord_non_star_variant(lattice):
    r = json.loads(_tools()["local_getis_ord"].invoke(
        {"file_id": lattice, "column": "gradient", "star": False}))
    assert r["ok"] is True and r["statistic"] == "Gi"


def test_permutation_inference_starts_no_process_pool(lattice, monkeypatch):
    """esda.G_Local defaults to n_jobs=-1, a joblib/loky pool of one fresh interpreter per core,
    each forked from the agent process. Every esda statistic here runs in this process instead;
    Stage 26 in docs/agent-architecture-changes.md has why, and what that costs on large layers.
    esda imports Parallel inside the call, so patching joblib itself is patching it where it is
    used."""
    joblib = pytest.importorskip("joblib")

    def no_pool(*args, **kwargs):
        raise AssertionError("esda started a joblib process pool")

    monkeypatch.setattr(joblib, "Parallel", no_pool)
    tools = _tools()
    for tool, extra in (("local_getis_ord", {}), ("local_getis_ord", {"star": False}),
                        ("local_moran_lisa", {}), ("global_spatial_autocorrelation", {})):
        r = json.loads(tools[tool].invoke({"file_id": lattice, "column": "gradient", **extra}))
        assert r["ok"] is True, (tool, extra, r.get("error"))


# --- Moran scatterplot ----------------------------------------------------------------


def test_moran_scatterplot_writes_a_png(lattice):
    pytest.importorskip("matplotlib")
    r = json.loads(_tools()["moran_scatterplot"].invoke(
        {"file_id": lattice, "column": "gradient"}))
    assert r["ok"] is True
    assert str(r["filename"]).endswith(".png") and r["download_url"]
    assert r["morans_i"] > 0.5 and r["p_value"] < 0.05
    from agent_runtime.file_store import resolve_file_id
    assert Path(resolve_file_id(r["file_id"])).stat().st_size > 5000     # a real plot


# --- spatial regression ---------------------------------------------------------------


def test_regression_auto_prefers_a_spatial_model_when_dependence_exists(lattice):
    pytest.importorskip("spreg")
    r = json.loads(_tools()["spatial_regression"].invoke(
        {"file_id": lattice, "y_column": "y", "x_columns": ["x1"]}))
    assert r["ok"] is True
    # y carries a real spatial component, so the LM tests should reject plain OLS.
    assert r["model"] in {"lag", "error"}
    assert r["model_choice_reason"]
    assert r["diagnostics"]["lm_lag"]["p_value"] is not None
    names = [c["variable"] for c in r["coefficients"]]
    assert "CONSTANT" in names and "x1" in names
    # The spatial parameter must appear: spreg appends it to betas but not to name_x.
    assert len(r["coefficients"]) == 3
    assert any("rho" in n or "lambda" in n for n in names)
    x1 = next(c for c in r["coefficients"] if c["variable"] == "x1")
    assert x1["significant_at_05"] is True      # the planted effect is strong
    assert 1.0 < x1["coefficient"] < 2.0        # and recovered near its true 1.5
    assert r["map_layer"]["style_by"] == "residual"
    assert r["coefficients_csv"]["download_url"]


def test_regression_explicit_ols_stays_ols(lattice):
    pytest.importorskip("spreg")
    r = json.loads(_tools()["spatial_regression"].invoke(
        {"file_id": lattice, "y_column": "y", "x_columns": ["x1"], "model": "ols"}))
    assert r["ok"] is True and r["model"] == "ols"
    assert r["fit"]["r_squared"] is not None
    assert len(r["coefficients"]) == 2          # CONSTANT + x1, no spatial term


def test_regression_needs_explanatory_columns(lattice):
    r = json.loads(_tools()["spatial_regression"].invoke(
        {"file_id": lattice, "y_column": "y", "x_columns": []}))
    assert r["ok"] is False and "x_columns" in r["error"]


def test_regression_refuses_y_on_both_sides(lattice):
    r = json.loads(_tools()["spatial_regression"].invoke(
        {"file_id": lattice, "y_column": "y", "x_columns": ["y", "x1"]}))
    assert r["ok"] is False and "itself" in str(r.get("hint", "")) + r["error"]


# --- regionalization ------------------------------------------------------------------


def test_skater_builds_the_requested_number_of_regions(lattice):
    pytest.importorskip("pygeoda")
    r = json.loads(_tools()["regionalize"].invoke(
        {"file_id": lattice, "columns": ["gradient"], "n_regions": 4}))
    assert r["ok"] is True
    assert r["regions_found"] == 4
    assert sum(r["region_sizes"].values()) == SIDE * SIDE
    assert len(r["region_summary"]) == 4
    layer = r["map_layer"]
    assert layer["render"] == "categories" and layer["style_by"] == "region"
    assert len(layer["legend"]) == 4
    assert r["summary_csv"]["download_url"]


def test_regionalize_does_not_cry_island_on_a_connected_layer(lattice):
    """pygeoda's has_isolates is a METHOD; read un-called it is always truthy, which put a
    false island warning on every result."""
    pytest.importorskip("pygeoda")
    r = json.loads(_tools()["regionalize"].invoke(
        {"file_id": lattice, "columns": ["gradient"], "n_regions": 4}))
    assert r["ok"] is True
    assert not any("island" in n.lower() for n in (r.get("notes") or [])), r.get("notes")


def test_regionalize_rejects_more_regions_than_areas(lattice):
    pytest.importorskip("pygeoda")
    r = json.loads(_tools()["regionalize"].invoke(
        {"file_id": lattice, "columns": ["gradient"], "n_regions": 500}))
    assert r["ok"] is False and "fewer than" in r["error"]


def test_regionalize_needs_polygons(points):
    pytest.importorskip("pygeoda")
    r = json.loads(_tools()["regionalize"].invoke(
        {"file_id": points, "columns": ["v"], "n_regions": 3}))
    assert r["ok"] is False
    assert "POLYGON" in r["error"] or "polygon" in r["error"]


def test_maxp_requires_its_bound(lattice):
    pytest.importorskip("pygeoda")
    r = json.loads(_tools()["regionalize"].invoke(
        {"file_id": lattice, "columns": ["gradient"], "method": "maxp"}))
    assert r["ok"] is False and "min_bound" in r["error"]


def test_maxp_respects_the_bound(lattice):
    pytest.importorskip("pygeoda")
    r = json.loads(_tools()["regionalize"].invoke(
        {"file_id": lattice, "columns": ["gradient"], "method": "maxp",
         "bound_column": "pop", "min_bound": 3000}))
    assert r["ok"] is True and r["regions_found"] >= 2
    totals = [row["total_pop"] for row in r["region_summary"]]
    assert all(t >= 3000 for t in totals), totals


# --- the two correctness invariants the module documents -------------------------------


@pytest.fixture()
def lattice_with_gaps(monkeypatch, tmp_path):
    """The lattice with 12 cells missing their value, to pin the drop-then-build order."""
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "store"))
    gdf = _lattice()
    gdf.loc[gdf.index[:12], "gradient"] = None
    path = tmp_path / "gaps.geojson"
    gdf.to_file(path, driver="GeoJSON")
    return _upload(path)


def test_missing_values_are_dropped_before_the_weights_are_built(lattice_with_gaps):
    """The subtle bug this guards: subsetting AFTER building W leaves the matrix indexed to
    rows that are gone, pairing each area with the wrong neighbours."""
    r = json.loads(_tools()["local_moran_lisa"].invoke(
        {"file_id": lattice_with_gaps, "column": "gradient"}))
    assert r["ok"] is True
    assert r["features_analyzed"] == SIDE * SIDE - 12
    # W must be built on the SURVIVORS, not on the original row count.
    assert r["connectivity"]["n"] == SIDE * SIDE - 12
    assert sum(r["class_counts"].values()) == SIDE * SIDE - 12
    assert any("only correct order" in n for n in r["notes"])


def test_global_reports_the_same_reduced_n(lattice_with_gaps):
    r = json.loads(_tools()["global_spatial_autocorrelation"].invoke(
        {"file_id": lattice_with_gaps, "column": "gradient"}))
    assert r["ok"] is True
    assert r["features_analyzed"] == r["connectivity"]["n"] == SIDE * SIDE - 12


@pytest.fixture()
def lattice_with_island(monkeypatch, tmp_path):
    """A lattice plus one far-away detached cell, which queen contiguity leaves an island."""
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "store"))
    gdf = _lattice()
    far = gpd.GeoDataFrame(
        {"name": ["detached"], "row": [99], "col": [99], "gradient": [12.0], "noise": [50.0],
         "x1": [0.0], "y": [1.0], "pop": [500.0]},
        geometry=[Polygon([(50, 50), (51, 50), (51, 51), (50, 51)])], crs="EPSG:4326")
    path = tmp_path / "island.geojson"
    gpd.GeoDataFrame(pd_concat([gdf, far]), crs="EPSG:4326").to_file(path, driver="GeoJSON")
    return _upload(path)


def pd_concat(frames):
    import pandas as pd
    return pd.concat(frames, ignore_index=True)


def test_islands_are_reported_not_silently_called_insignificant(lattice_with_island):
    tools = _tools()
    w = json.loads(tools["spatial_weights"].invoke({"file_id": lattice_with_island}))
    assert w["ok"] is True and w["connectivity"]["islands"] == 1
    assert any("no neighbours" in n.lower() for n in w["notes"])

    r = json.loads(tools["local_moran_lisa"].invoke(
        {"file_id": lattice_with_island, "column": "gradient"}))
    assert r["ok"] is True
    # The island gets its own class rather than being folded into "Not significant".
    assert r["class_counts"].get("Island (no neighbours)") == 1
    assert "Island (no neighbours)" in {e["label"] for e in r["map_layer"]["legend"]}


# --- distance-based weights are measured in real metres --------------------------------


def test_distance_band_uses_real_kilometres(lattice):
    """The lattice spans whole DEGREES, so a 20 km band must find no neighbours at all
    while a very wide one connects everything -- proof the threshold is not in degrees."""
    tools = _tools()
    narrow = json.loads(tools["spatial_weights"].invoke(
        {"file_id": lattice, "weights": "distance_band", "threshold_km": 20}))
    assert narrow["ok"] is True
    assert narrow["connectivity"]["islands"] == SIDE * SIDE, narrow["connectivity"]

    wide = json.loads(tools["spatial_weights"].invoke(
        {"file_id": lattice, "weights": "distance_band", "threshold_km": 200}))
    assert wide["ok"] is True
    assert wide["connectivity"]["islands"] == 0
    assert wide["connectivity"]["mean_neighbors"] > narrow["connectivity"]["mean_neighbors"]


def test_distance_band_without_a_threshold_leaves_no_island(lattice):
    r = json.loads(_tools()["spatial_weights"].invoke(
        {"file_id": lattice, "weights": "distance_band"}))
    assert r["ok"] is True and r["connectivity"]["islands"] == 0
    assert any("no island" in n for n in r["notes"])


def test_default_distance_band_survives_a_rounding_disagreement(lattice, monkeypatch):
    """The no-island promise must not rest on rounding luck.

    min_threshold_distance returns the critical pair's distance as sqrt(d2), and DistanceBand
    admits a pair when d2 <= threshold**2. Squaring the rounded root can come back one ulp
    short. On Linux/x86-64 it does for this lattice and the band left an island, while
    macOS/arm64 rounds the other way, so the test above passes on a Mac with or without the
    pad. Shrinking the computed minimum by a relative 1e-12 (far more than one ulp, a thousandth
    of the tool's pad) reproduces the failure on every platform, so dropping the pad fails here
    on a Mac too.
    """
    from libpysal import weights

    real = weights.min_threshold_distance
    calls = []

    def short_by_rounding(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs) * (1.0 - 1e-12)

    monkeypatch.setattr(weights, "min_threshold_distance", short_by_rounding)
    r = json.loads(_tools()["spatial_weights"].invoke(
        {"file_id": lattice, "weights": "distance_band"}))
    assert calls, "the default band no longer comes from min_threshold_distance; revisit this test"
    assert r["ok"] is True and r["connectivity"]["islands"] == 0


def test_kernel_weights_build(lattice):
    r = json.loads(_tools()["spatial_weights"].invoke(
        {"file_id": lattice, "weights": "kernel", "k": 5}))
    assert r["ok"] is True and r["connectivity"]["n"] == SIDE * SIDE


def test_maxp_accepts_a_bound_column_that_is_also_a_clustering_variable(lattice):
    """The natural max-p call — "regions similar in population, each with >= N people" — names
    the same column twice. That duplicated it into the frame handed to pygeoda, where
    frame[name] is a DataFrame rather than a Series, so GetRealCol did DataFrame.to_list() and
    the tool died with an AttributeError that named nothing relevant. Reproduced on 3,265
    Illinois tracts and on this lattice."""
    tools = _tools([lattice])
    r = json.loads(tools["regionalize"].invoke({
        "file_id": lattice, "columns": ["pop"], "method": "maxp",
        "bound_column": "pop", "min_bound": 5000.0}))
    assert r.get("ok") is True, r.get("error")
    assert r.get("method") == "maxp"
    ml = r.get("map_layer") or {}
    assert ml.get("render") == "categories"
    # a categorical layer must carry its palette or the client cannot colour the classes
    assert len(ml.get("legend") or []) == r["regions_found"]


# --- a contiguity graph in separate parts is refused, never handed to pygeoda ------------
#
# On a disconnected contiguity graph pygeoda 0.1.3 SEGFAULTS (skater, redcap, schc) or spins
# forever (azp, max-p), and max-p spins on any layer whose total is below min_bound. So every
# test here that could hit one runs the call in a CHILD process: if a guard ever regresses, one
# test fails with the exit code and the suite lives.

_REPO = Path(__file__).resolve().parents[2]

_CHILD = """
import json, sys
from agent_runtime.analysis_spatial_stats_tools import make_spatial_stats_tools
regionalize = {t.name: t for t in make_spatial_stats_tools()}["regionalize"]
for args in json.loads(sys.argv[1]):
    print(regionalize.invoke(args), flush=True)
"""


def _regionalize_in_child(calls, timeout=120):
    """Run each regionalize call in one child process, and return their parsed results."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        p for p in (str(_REPO), os.environ.get("PYTHONPATH")) if p))
    try:
        # No cwd and close_fds=False keep CPython on posix_spawn on macOS. A fork() after PROJ
        # has run in this process can kill the child before exec, and that -11 would read here
        # as the tool crashing.
        proc = subprocess.run([sys.executable, "-c", _CHILD, json.dumps(calls)], env=env,
                              capture_output=True, text=True, timeout=timeout, close_fds=False)
    except subprocess.TimeoutExpired:
        pytest.fail(f"regionalize did not return within {timeout}s; azp and max-p spin forever "
                    "when no valid answer exists (a split graph, or a min_bound above the "
                    "layer's total)")
    results = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
    assert proc.returncode == 0, (
        f"the process running regionalize died with exit {proc.returncode} (-11 is SIGSEGV, "
        f"139 from a shell) after {len(results)} of {len(calls)} call(s); the next was "
        f"{calls[len(results)] if len(results) < len(calls) else None}\n{proc.stderr[-3000:]}")
    assert len(results) == len(calls), proc.stdout[-3000:]
    return results


@pytest.fixture()
def split_lattice(monkeypatch, tmp_path):
    """The 5x5 lattice from the crash report, built as -88.3 + c/100, so a cell's right edge
    (-88.29 + c/100) and its neighbour's left edge (-88.3 + (c+1)/100) can miss by one ulp.
    Three of the four column joins and one of the four row joins do, which leaves 8 blocks
    and no island: the graph the old has_isolates() check passed as clean.

    Written as a GeoPackage on purpose. GeoJSON writes the coordinates rounded, which turns
    -88.28999999999999 back into -88.29 and the lattice back into one block, and that would
    make every test on this fixture pass for nothing."""
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "store"))
    rng = np.random.default_rng(3)
    cells = [(r, c) for r in range(5) for c in range(5)]
    gdf = gpd.GeoDataFrame(
        {"name": [f"cell_{r}_{c}" for r, c in cells], "rate": rng.normal(10, 2, 25),
         "x1": rng.normal(0, 1, 25), "pop": rng.integers(100, 900, 25).astype("float64")},
        geometry=[box(-88.3 + c / 100, 40.0 + r / 100, -88.29 + c / 100, 40.01 + r / 100)
                  for r, c in cells],
        crs="EPSG:4326")
    path = tmp_path / "split_lattice.gpkg"
    gdf.to_file(path, driver="GPKG")
    return _upload(path)


@pytest.fixture()
def corner_pair(monkeypatch, tmp_path):
    """Two 3x3 blocks that meet at a single corner: two parts under rook, one under queen."""
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "store"))
    cells = [(r + off, c + off) for off in (0, 3) for r in range(3) for c in range(3)]
    gdf = gpd.GeoDataFrame(
        {"name": [f"cell_{r}_{c}" for r, c in cells],
         "v": np.random.default_rng(4).normal(0, 1, len(cells))},
        geometry=[box(c, r, c + 1, r + 1) for r, c in cells], crs="EPSG:4326")
    path = tmp_path / "corner_pair.geojson"
    gdf.to_file(path, driver="GeoJSON")
    return _upload(path)


def test_connected_parts_orders_largest_first_and_follows_one_way_edges():
    from agent_runtime.analysis_spatial_stats_tools import _connected_parts
    assert _connected_parts([[1], [0], [3], [2], []]) == [[0, 1], [2, 3], [4]]
    assert _connected_parts([[], [2], [1], [4], [3, 5], [4]]) == [[3, 4, 5], [1, 2], [0]]
    # an edge listed in one direction only still joins its two ends
    assert _connected_parts([[1], [], []]) == [[0, 1], [2]]


def test_regionalize_refuses_a_split_layer_instead_of_crashing(split_lattice):
    """The crash report, pinned: skater on this lattice killed the process with SIGSEGV, and
    the only pre-flight check, has_isolates(), was False. Every method is refused, and so is an
    n_regions at or above the number of parts, because the crash did not depend on it."""
    pytest.importorskip("pygeoda")
    report = {"file_id": split_lattice, "columns": ["rate", "x1"], "n_regions": 3}
    calls = [report, {**report, "n_regions": 10},
             *({**report, "method": m} for m in ("redcap", "schc", "azp")),
             {"file_id": split_lattice, "columns": ["rate"], "method": "maxp",
              "bound_column": "pop", "min_bound": 4000.0}]
    results = _regionalize_in_child(calls)
    for args, r in zip(calls, results):
        assert r["ok"] is False, (args, r)
        assert r["components"] == 8 and r["islands"] == 0, (args, r)
    r = results[0]
    assert r["component_sizes"] == [6, 4, 3, 3, 3, 2, 2, 2]
    assert "8 separate parts" in r["error"] and "no method or n_regions" in r["error"]
    # The way out is concrete: the areas to leave out to keep the largest part, by a column
    # select_by_attribute can match on.
    assert r["id_column"] == "name"
    assert len(r["outside_largest_part"]) == 25 - 6
    assert "select_by_attribute" in r["hint"] and "op='not_in'" in r["hint"]


def test_regionalize_refuses_an_island_rather_than_noting_it(lattice_with_island):
    """An island is a part of one. It used to earn a note, and then the same crash."""
    pytest.importorskip("pygeoda")
    (r,) = _regionalize_in_child(
        [{"file_id": lattice_with_island, "columns": ["gradient"], "n_regions": 4}])
    assert r["ok"] is False
    assert r["components"] == 2 and r["islands"] == 1
    assert r["component_sizes"] == [SIDE * SIDE, 1]
    assert r["outside_largest_part"] == ["detached"]
    assert "too small to regionalize" in r["hint"]


def test_a_rook_split_that_queen_joins_says_so(corner_pair):
    pytest.importorskip("pygeoda")
    rook, queen = _regionalize_in_child([
        {"file_id": corner_pair, "columns": ["v"], "n_regions": 3, "weights": "rook"},
        {"file_id": corner_pair, "columns": ["v"], "n_regions": 3, "weights": "queen"}])
    assert rook["ok"] is False and rook["components"] == 2
    assert "weights='queen' joins" in rook["hint"]
    # ...and the connected graph is not refused: the guard costs nothing on a usable layer.
    assert queen["ok"] is True and queen["regions_found"] == 3


@pytest.fixture()
def mainland_and_islands(monkeypatch, tmp_path):
    """A 4x4 mainland, a 3x3 island big enough to regionalize, and a 1x2 one that is not."""
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "store"))
    blocks = [(0, 0, 4, 4), (10, 10, 3, 3), (20, 20, 1, 2)]   # x0, y0, rows, cols
    cells = [(x0 + c, y0 + r) for x0, y0, rows, cols in blocks
             for r in range(rows) for c in range(cols)]
    gdf = gpd.GeoDataFrame(
        {"name": [f"cell_{x}_{y}" for x, y in cells],
         "v": np.random.default_rng(6).normal(0, 1, len(cells))},
        geometry=[box(x, y, x + 1, y + 1) for x, y in cells], crs="EPSG:4326")
    path = tmp_path / "mainland_and_islands.geojson"
    gdf.to_file(path, driver="GeoJSON")
    return _upload(path)


def test_a_split_rest_is_offered_part_by_part(mainland_and_islands):
    """The areas outside the largest part are one block only when there is one other part.
    Offering them all as one op='in' selection handed back another split, so with several
    parts each is listed, and the route selects one of them."""
    pytest.importorskip("pygeoda")
    (r,) = _regionalize_in_child(
        [{"file_id": mainland_and_islands, "columns": ["v"], "n_regions": 3}])
    assert r["ok"] is False and r["component_sizes"] == [16, 9, 2]
    assert [len(part) for part in r["smaller_parts"]] == [9, 2]
    assert "one list from smaller_parts" in r["hint"]

    from agent_runtime.analysis_aggregate_tools import make_aggregate_tools
    select = {t.name: t for t in make_aggregate_tools()}["select_by_attribute"]
    island = json.loads(select.invoke({"file_id": mainland_and_islands, "column": r["id_column"],
                                       "op": "in", "value": r["smaller_parts"][0]}))
    assert island["ok"] is True and island["feature_count"] == 9
    (done,) = _regionalize_in_child(
        [{"file_id": island["file_id"], "columns": ["v"], "n_regions": 2}])
    assert done["ok"] is True and done["features_analyzed"] == 9


def test_maxp_refuses_a_bound_the_whole_layer_cannot_reach(lattice):
    """max-p spins forever on a CONNECTED layer too, when no region can reach min_bound, and on
    one block that is exactly when the layer's total is below it (measured: the total itself
    returns one region, the total plus one never returns)."""
    pytest.importorskip("pygeoda")
    (r,) = _regionalize_in_child([{"file_id": lattice, "columns": ["gradient"],
                                   "method": "maxp", "bound_column": "pop",
                                   "min_bound": 1e9}])
    assert r["ok"] is False
    assert "than the whole layer holds" in r["error"] and r["bound_total"] < 1e9
    assert "lower min_bound" in r["hint"]


def test_id_column_skips_the_analysed_columns():
    """_prepare has already coerced an analysed column ("1,100" in the file is 1100 in the
    frame), so listing ids from it would give select_by_attribute values it cannot match."""
    from agent_runtime.analysis_spatial_stats_tools import _id_column
    frame = gpd.GeoDataFrame({"code": [1101, 1102, 1103], "tract": ["a", "b", "c"],
                              "share": [0.1, 0.2, 0.3]}, geometry=[Point(0, 0)] * 3)
    assert _id_column(frame, None, []) == "code"
    assert _id_column(frame, None, ["code"]) == "tract"
    # floats never qualify: a value that does not survive a trip through text cannot match
    assert _id_column(frame[["share", "geometry"]], None, []) is None


def test_spatial_weights_reports_the_split(lattice, split_lattice):
    """Islands are not the only way a graph falls apart; the diagnostics now say so."""
    tools = _tools()
    whole = json.loads(tools["spatial_weights"].invoke({"file_id": lattice}))
    assert whole["connectivity"]["components"] == 1
    assert "separate parts" not in whole["interpretation"]
    split = json.loads(tools["spatial_weights"].invoke({"file_id": split_lattice}))
    assert split["ok"] is True
    assert split["connectivity"]["islands"] == 0 and split["connectivity"]["components"] == 8
    assert "8 separate parts" in split["interpretation"]
