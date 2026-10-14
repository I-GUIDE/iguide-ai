"""The GIS task harness: its answer key, its traps, and its scorer.

A harness is only evidence if its own expected values are right and its scorer reads answers
the way they are actually written. These tests hold both, offline: the expected values against
a second implementation where one exists (esda for Moran's I and Gi*), and against the trap
each dataset is built to catch.
"""
from __future__ import annotations

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
