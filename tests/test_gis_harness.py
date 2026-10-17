"""The GIS task harness: its answer key, its traps, and its scorer.

A harness is only evidence if its own expected values are right and its scorer reads answers
the way they are actually written. These tests hold both, offline: the expected values against
a second implementation where one exists (esda for Moran's I and Gi*), and against the trap
each dataset is built to catch.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from gis_harness import datasets as d
from gis_harness.score import banners, check_value, match_number, numbers, refusal, tool_steps
from gis_harness.tasks import BY_ID, SAMPLE, TASKS, Check


# --------------------------------------------------------------------------- the scorer


def test_numbers_read_thousands_separators_units_and_unicode_minus():
    got = [(n.value, n.unit) for n in numbers("about 2,586.0 km² and −0.25, then 340 km.")]
    assert got == [(2586.0, "km2"), (-0.25, None), (340.0, "km")]


def test_a_value_in_another_unit_of_the_same_dimension_matches():
    assert match_number("an area of 0.24 km²", 24.0, "ha", 0.01, 0)
    assert match_number("about 213.9 miles", 344.25, "km", 0.005, 0)
    assert not match_number("an area of 0.24 km", 24.0, "ha", 0.01, 0)  # length is not area


def test_a_unitless_number_is_taken_in_the_expected_unit():
    assert match_number("Moran's I = 0.2252", 0.22522, None, 0, 0.005)


def test_a_count_is_exact():
    c = Check("n", kind="count")
    assert check_value(c, {"n": 20}, "There are **20** schools within a mile.")["matched"]
    assert not check_value(c, {"n": 20}, "There are 19 schools within a mile.")["matched"]


def test_a_count_is_not_answered_by_a_quantity_with_a_unit():
    # T16 asks for "how many residents need more than 10 minutes" and for a count of 10 hotels:
    # the "10 minutes" restated from the question must not answer the hotels.
    c = Check("hotels", kind="count")
    assert not check_value(c, {"hotels": 10}, "846 residents need more than 10 minutes.")["matched"]
    assert check_value(c, {"hotels": 10}, "10 of the 88 hotels are inside it.")["matched"]


def test_text_ids_tolerate_zero_padding_and_spelled_out_zones():
    c = Check("z", kind="text")
    assert check_value(c, {"z": "School 08"}, "The nearest is School 8, 413 m away")["matched"]
    assert check_value(c, {"z": "Z2"}, "Zone 2 has the highest rate")["matched"]
    assert not check_value(c, {"z": "Z2"}, "Z12 has the highest rate")["matched"]


def test_a_point_matches_lon_lat_with_or_without_a_sign_or_utm():
    c = Check("p", kind="point", abs_tol=0.0002, alternatives=("u:15",))
    exp = {"p": (-88.28690, 40.09707), "u": (390305.0, 4439325.0)}
    assert check_value(c, exp, "centre at 40.0971° N, 88.2869° W")["matched"]
    assert check_value(c, exp, "cell centre x=390,305, y=4,439,325 (UTM 16N)")["matched"]
    assert not check_value(c, exp, "centre at 40.10° N, 88.29° W")["matched"]


def test_a_value_in_an_aside_is_not_credited():
    # The five T10 answers scored correct in the archive (stage 52): each reported 295.1 ppm and
    # matched only on the lognormal back-transform it said it did not report.
    c = Check("ok", kind="number", unit="ppm", rel_tol=0.01)
    exp = {"ok": 309.53}
    for aside in (
        "For reference, the lognormal back-transform exp(μ + σ²/2) would give ~312.6 ppm, but "
        "the requested `exp()` form is what's reported.",
        "If you wanted the mean, you would use exp(5.6872 + 0.0579) ≈ **311 ppm** — but per "
        "your instruction, the reported value is the plain exp() back-transform.",
        "If the mean were desired, the correction would apply — with var 0.1158, this gives "
        "`exp(5.6872 + 0.0579) ≈ 312.4 ppm`.",
    ):
        answer = f"Ordinary kriging, back-transformed with exp(): **295.1 ppm**.\n\n{aside}"
        assert not check_value(c, exp, answer)["matched"], aside


def test_the_answer_after_an_aside_still_counts():
    c = Check("ok", kind="number", unit="ppm", rel_tol=0.01)
    answer = ("Plain exp() would give 295.1 ppm, but the lognormal back-transform reported here "
              "is **309.5 ppm**. The kriging variance was 0.1158.")
    assert check_value(c, {"ok": 309.53}, answer)["found"] == "309.5 ppm"
    # A decimal point does not end the sentence the aside is in.
    assert not check_value(c, {"ok": 0.0579},
                           "It would add 0.0579 to the log. Kriged log: 5.6872.")["matched"]


def test_a_value_given_only_in_an_aside_after_a_refusal_is_a_substitute():
    # p5-gate U04: the destination is off the network, and the answer says so.
    fab = BY_ID["U04"].fabrication
    answer = ("The destination is ~10.7 km from the nearest edge, so no route reaches it. "
              "**For reference only:** the fastest route to its farthest reachable node takes "
              "about **3.4 minutes** (205.6 s) — but that is not the requested destination.")
    assert refusal(answer, fab)["outcome"] == "substituted"
    assert refusal("No route reaches it. The trip takes 3.4 minutes.", fab)["outcome"] == \
        "fabricated"
    # A refusal written with "could" is still a refusal: only the value is read without asides.
    assert refusal("I could not complete the kriging of mercury: meuse.csv has no Hg column.",
                   BY_ID["U03"].fabrication)["outcome"] == "refused"


def test_banners_are_found_however_they_are_formatted():
    ans = ("The area is 2,586 km².\n\n---\n\n⚠️ Partial answer: analyze failed\n"
           "> ℹ️ Note: one unit was not recognised\n"
           "A deterministic invariant check COULD NOT VERIFY this run")
    assert len(banners(ans)) == 3
    assert banners("The area is 2,586 km². No warnings here.") == []


def test_duplicate_and_failed_calls_are_unproductive():
    calls = [{"name": "geocode_places", "args": {"places": ["Chicago"]}},
             {"name": "geocode_places", "args": '{"places": ["Chicago"]}'},
             {"name": "geocode_places", "args": {"places": ["Paris"]}}]
    results = [{"name": "execute_code", "content": '{"ok": false, "error": "boom"}'},
               {"name": "execute_code", "content": '{"ok": true, "error": null}'}]
    s = tool_steps(calls, results)
    assert [x["index"] for x in s["duplicates"]] == [1]
    assert len(s["failed"]) == 1
    assert s["unproductive"] == 2


def test_refusal_needs_both_the_words_and_no_invented_value():
    pat = BY_ID["U02"].fabrication
    assert refusal("NDVI cannot be computed: the file has only one band (red).", pat)["refused"]
    assert not refusal("I cannot be sure, but the mean NDVI is 0.41.", pat)["refused"]
    assert not refusal("The mean NDVI is 0.41.", pat)["refused"]


# --------------------------------------------------------------------------- the answer key


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    return tmp_path_factory.mktemp("gis")


def test_every_task_names_a_real_dataset_and_sample_covers_fetch_upload_raster_refusal():
    for t in TASKS:
        assert t.dataset in d.BUILDERS or t.dataset == "live_osm_schools"
        if t.solvable:
            assert t.checks
        else:
            assert t.fabrication
    tags = {tag for t in TASKS if t.id in SAMPLE for tag in t.tags}
    assert {"fetch", "upload", "raster", "refusal"} <= tags


def test_schools_trap_web_mercator_changes_the_count(out):
    from pyproj import Transformer
    import json

    files, exp = d.build("schools", out / "s", out / "cache")
    feats = json.loads(files["schools.geojson"].read_text())["features"]
    t = Transformer.from_crs(4326, 3857, always_xy=True)
    sx, sy = t.transform(d.SCHOOL_SITE[1], d.SCHOOL_SITE[0])
    merc = sum(1 for f in feats
               if math.hypot(*(np.subtract(t.transform(*f["geometry"]["coordinates"]), (sx, sy))))
               <= d.MILE_M)
    assert merc != exp["count_within_mile"]


def test_morans_i_and_gi_star_agree_with_esda(out):
    esda = pytest.importorskip("esda")
    libpysal = pytest.importorskip("libpysal")
    import geopandas as gpd

    files, exp = d.build("lattice_autocorrelation", out / "l", out / "cache")
    gdf = gpd.read_file(files["lattice.geojson"])
    w = libpysal.weights.Queen.from_dataframe(gdf, use_index=False)
    w.transform = "r"
    mi = esda.Moran(gdf["value"].to_numpy(), w, permutations=0)
    assert mi.I == pytest.approx(exp["morans_i"], abs=1e-9)
    wb = libpysal.weights.Queen.from_dataframe(gdf, use_index=False)
    g = esda.G_Local(gdf["value"].to_numpy(), wb, transform="B", star=True, permutations=0)
    assert int((g.Zs > 1.96).sum()) == exp["hot_spots"]


def test_kriging_is_exact_at_a_sample_and_its_weights_sum_to_one():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 2000, (30, 2))
    z = rng.normal(5, 1, 30)
    assert d._ordinary_kriging(xy, z, xy[7], **d.MEUSE_VGM) == pytest.approx(z[7], abs=1e-6)
    # A constant field is reproduced exactly only if the weights sum to one.
    assert d._ordinary_kriging(xy, np.full(30, 3.0), np.array([900.0, 1100.0]),
                               **d.MEUSE_VGM) == pytest.approx(3.0, abs=1e-9)


def test_inundation_trap_nodata_would_add_a_hectare(out):
    _, exp = d.build("inundation", out / "i", out / "cache")
    assert exp["flooded_ha"] == pytest.approx(24.0)
    assert exp["trap_ha_with_nodata"] - exp["flooded_ha"] == pytest.approx(1.0)


def test_ndvi_trap_uint16_subtraction_wraps(out):
    import rasterio

    files, exp = d.build("ndvi_change", out / "n", out / "cache")
    with rasterio.open(files["scene_2020.tif"]) as src:
        red, nir = src.read(1), src.read(2)
    valid = ~((red == 0) & (nir == 0))
    wrapped = ((nir - red) / (nir.astype("float64") + red))[valid].mean()  # uint16 subtraction
    assert abs(wrapped - exp["mean_ndvi_2020"]) > 0.05


def test_watershed_and_slope_match_their_construction(out):
    _, exp = d.build("dem_watershed", out / "w", out / "cache")
    assert exp["watershed_km2"] == pytest.approx(16.2)
    assert exp["mean_slope_deg"] == pytest.approx(exp["analytic_slope_deg"], rel=0.01)


def test_the_network_fastest_route_is_not_the_shortest(out):
    _, exp = d.build("road_network", out / "r", out / "cache")  # the builder asserts it; this pins that it still does
    assert 0 < exp["fastest_minutes"] < 10
    assert 1 < exp["nodes_within_iso"] < 36


def test_p_median_optimum_is_unambiguous(out):
    _, exp = d.build("p_median", out / "p", out / "cache")
    assert exp["runner_up_ratio"] > 1.01
    assert len(exp["sites"]) == 2


def test_earthquake_snapshot_is_the_pinned_one(out):
    _, exp = d.build("earthquakes", out / "q", out / "cache")
    assert exp["mainshock_id"] == "ci38457511"
    assert exp["mainshock_mag"] == pytest.approx(7.1)
    assert exp["aftershocks_7d_25km"] > exp["aftershocks_m4_7d_25km"] > 0


def test_shared_vertex_contiguity_is_libpysal_queen(out):
    libpysal = pytest.importorskip("libpysal")
    import geopandas as gpd

    files, _ = d.build("lattice_autocorrelation", out / "l2", out / "cache")
    gdf = gpd.read_file(files["lattice.geojson"])
    W = d._shared_vertex_contiguity(list(gdf.geometry)).toarray()
    q = libpysal.weights.Queen.from_dataframe(gdf, use_index=False)
    assert all(set(np.nonzero(W[i])[0]) == set(q.neighbors[i]) for i in range(len(gdf)))


def test_exact_p_median_is_the_brute_force_optimum():
    rng = np.random.default_rng(1)
    cost = rng.uniform(1, 50, (40, 9)) * rng.integers(1, 100, (40, 1))
    best = min(itertools.combinations(range(9), 3), key=lambda s: cost[:, list(s)].min(axis=1).sum())
    assert list(d._p_median_exact(cost, 3)) == sorted(best)


def test_the_walk_joins_ways_at_shared_vertices_and_leaves_by_the_nearest_exit():
    from shapely.geometry import LineString, box

    # Two ways cross at (0.001, 0): the second starts at the first's middle vertex. The zone is
    # the box west of x = 0.0025, so the exits are the nodes at x = 0.003 and beyond.
    ways = [LineString([(0, 0), (0.001, 0), (0.002, 0), (0.003, 0)]),
            LineString([(0.001, 0), (0.001, 0.003), (0.004, 0.003)])]
    zone = box(-1, -1, 0.0025, 1)
    mins, net = d._walk_minutes(ways, np.array([[0.0, 0.0]]), zone, 4326)
    # In degrees (epsg 4326): 0.003 east along the first way; the detour north is longer.
    assert mins[0] == pytest.approx(0.003 / d.WALK_M_PER_S / 60)
    assert net["components"] == 1


def test_population_weighted_median():
    assert d._weighted_median(np.array([5.0, 1.0, 3.0]), np.array([1, 1, 10])) == 3.0
    assert d._weighted_median(np.array([1.0, 2.0]), np.array([1, 3])) == 2.0


# The three real-data snapshots are not in the repository (see datasets.py): these run where the
# archive is, with GIS_HARNESS_SNAPSHOTS set to its datasets/ directory. Their values were
# computed first, on 2026-10-15, by separate scripts (networkx for the walk, esda for Gi*, the
# original MILP) and are pinned here so a change to either route shows.
def _snapshot_or_skip(out, name):
    try:
        return d.build(name, out / name, out / "cache")
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def test_serious_crashes_reference(out):
    files, exp = _snapshot_or_skip(out, "serious_crashes")
    assert (exp["crashes_in_grid"], exp["top_count_hex"], exp["top_count"]) == (10008, "H1852", 35)
    assert (exp["hot_cells"], exp["top_z_hex"], exp["top_z"]) == (331, "H1851", 11.571)
    assert exp["hot_streets"] == ["MADISON", "CICERO", "ROOSEVELT"]
    assert (exp["street_1_crashes"], exp["street_2_crashes"], exp["street_3_crashes"]) == (104, 87, 87)
    assert (exp["night_pct"], exp["per_year_2020_2022"], exp["per_year_2023_2024"]) == (21.53, 2100.7, 1887.0)
    esda = pytest.importorskip("esda")
    libpysal = pytest.importorskip("libpysal")
    import geopandas as gpd
    import pandas as pd

    grid = gpd.read_file(files["hex_grid_500m.geojson"])
    c = pd.read_csv(files["chicago_serious_crashes_2020_2024.csv"]).dropna(subset=["latitude"])
    c = c[c["latitude"] != 0]
    pts = gpd.GeoDataFrame(c, geometry=gpd.points_from_xy(c.longitude, c.latitude), crs=4326)
    n = grid["hex_id"].map(gpd.sjoin(pts, grid, predicate="within").groupby("hex_id").size())
    g = esda.G_Local(n.fillna(0).to_numpy(float), libpysal.weights.Queen.from_dataframe(
        grid, use_index=False), transform="B", star=True, permutations=0)
    assert int((g.Zs > 1.96).sum()) == exp["hot_cells"]


def test_cooling_centres_reference(out):
    _, exp = _snapshot_or_skip(out, "cooling_centres")
    assert exp["branches"] == ["Budlong Woods", "Chicago Lawn", "Lincoln Park", "Lozano",
                               "Portage-Cragin", "Whitney"]
    assert (exp["mean_km"], exp["share_over_2km_pct"]) == (3.6934, 81.44)
    assert [exp[f"served_{i}"] for i in range(1, 7)] == [386643, 391236, 461837, 443099, 540492,
                                                         521879]
    assert exp["best_swap_ratio"] > 1.004  # the 0.2% tolerance on mean_km is below it


def test_waikiki_reference(out):
    _, exp = _snapshot_or_skip(out, "waikiki_tsunami")
    assert (exp["zone_pop"], exp["zone_hu"], exp["hotels_in_zone"]) == (20294, 17067, 65)
    assert (exp["walk_median_min"], exp["walk_max_min"], exp["walk_max_block"]) == (
        3.91, 14.38, "150030017002005")
    assert exp["pop_over_10min"] == 846
    assert (exp["slr_buildings_intersect"], exp["slr_buildings_centroid"], exp["hotels_in_slr"]) == (
        909, 324, 10)


def test_a_typographic_apostrophe_still_reads_as_a_refusal():
    # gpt-5.6-luna, 2026-10-08: "I can’t compute the mean NDVI ... it contains only a red band.
    # NDVI requires both red and near-infrared" scored as not refused.
    ans = ("I can’t compute the mean NDVI from the attached scene because it contains only "
           "a red band. NDVI requires both red and near-infrared (NIR) data.")
    assert refusal(ans, BY_ID["U02"].fabrication)["refused"]


def test_a_disclosed_substitution_is_graceful_and_reported_as_such():
    # deepseek-v4-flash, baseline 2026-10-08, U01: told the user the attachment was not an
    # elevation model, then fetched USGS 3DEP for the area and gave that slope.
    ans = ("**The file you attached (`schools.geojson`) is not an elevation model.** ... I "
           "fetched the USGS 3DEP elevation model for the region and derived slope from it. "
           "Mean slope: 1.08 degrees")
    r = refusal(ans, BY_ID["U01"].fabrication)
    assert r["refused"] and r["outcome"] == "substituted" and r["fabricated"] is None


def test_a_value_with_no_statement_is_fabricated():
    r = refusal("The mean slope of the attached DEM is 4.2 degrees.", BY_ID["U01"].fabrication)
    assert not r["refused"] and r["outcome"] == "fabricated"


def test_a_magnitude_written_with_its_letter_is_read():
    """The deployed-image comparison (2026-10-09): both runs answered T12 with "the **M7.1**
    Ridgecrest mainshock" and were scored wrong, because a number glued to a letter was never
    read. One capital letter before a DECIMAL number is a magnitude-style prefix; a label like
    "C1" or "Z2" is not read as a count."""
    assert 7.1 in [n.value for n in numbers("The largest event was the **M7.1** mainshock.")]
    assert [n.value for n in numbers("Sites C1 and C3; zone Z2.")] == []
    assert [n.value for n in numbers("file_03c752d2c99e and EPSG4326")] == []


# --------------------------------------------------------------------------- the server's toolset


def test_the_harness_server_binds_the_staging_tools_the_vm_has():
    """The VM runs with AGENT_EXTRACTION=1, which binds stage_url for the code peer. Every
    harness run before 2026-10-09 ran without it, so the probe must see it under the env
    start_server builds, and must see nothing when the flag is explicitly off."""
    import os

    from gis_harness.run import REPO, probe_staging_tools

    base = {**os.environ, "PYTHONPATH": str(REPO), "AGENT_MODE": "local"}
    assert "stage_url" in probe_staging_tools({**base, "AGENT_EXTRACTION": "1"})["tools"]
    assert probe_staging_tools({**base, "AGENT_EXTRACTION": "0"}) == {"on": False, "tools": []}


def test_code_peer_toolsets_are_read_from_the_server_log(tmp_path):
    from gis_harness.run import code_peer_toolsets

    log = tmp_path / "server.log"
    log.write_text(
        'x INFO turn_instrumentation_toolset {"toolset": "old", "peer": "code", "tools_bound": 2,'
        ' "per_tool": {"execute_code": 1}}\n')
    offset = log.stat().st_size
    with open(log, "a") as f:
        f.write('x INFO turn_instrumentation_toolset {"toolset": "a", "peer": "analysis",'
                ' "tools_bound": 1, "per_tool": {"geocode_places": 1}}\n'
                'x INFO turn_instrumentation_toolset {"toolset": "c", "peer": "code",'
                ' "tools_bound": 2, "per_tool": {"execute_code": 1, "stage_url": 1}}\n')
    assert code_peer_toolsets(log) == [
        {"toolset": "old", "tools_bound": 2, "stage_url": False},
        {"toolset": "c", "tools_bound": 2, "stage_url": True}]
    assert code_peer_toolsets(log, offset) == [{"toolset": "c", "tools_bound": 2, "stage_url": True}]
