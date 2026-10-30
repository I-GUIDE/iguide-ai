"""The trap tests of 2026-10-23 (gis_harness X01-X08) and the fixes for the two that failed.

X01: a 40-row CSV, lat/lon swapped. The 20-line preview showed 19 rows and said nothing of the
rest; the model typed those rows into its code and answered 2 for 4. Now the preview counts the
file and says it is a preview, the swap is a check, and typed-in data is a pre-run finding.

X02: a GeoJSON in State Plane feet with no `crs`. Reported as EPSG:4326 beside bounds of a
million, measured as metres, 10.76x too large. Now the coordinates are checked against degrees
and the candidate CRSs, feet and metres both, are listed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime import code_watch  # noqa: E402
from agent_runtime.coord_sanity import crs_check, degree_like, latlon_swap_hint  # noqa: E402
from agent_runtime.langchain_file_tools import _csv_preview, _geojson_preview  # noqa: E402


def _stores_csv(n=40) -> str:
    rows = ["store,lat,lon"]
    for i in range(n):
        rows.append(f"S{i:02d},{-88.25 + i * 0.001:.5f},{40.10 + i * 0.001:.5f}")
    return "\n".join(rows) + "\n"


# ------------------------------------------------------------------ the CSV preview (X01)

def test_preview_counts_the_whole_file_and_says_it_is_a_preview():
    out = _csv_preview(_stores_csv(40), max_rows=20)
    assert len(out["rows"]) == 20 and out["header"] == ["store", "lat", "lon"]
    assert out["total_data_rows"] == 40 and out["data_rows_shown"] == 19
    assert out["truncated"] is True
    assert "19 of 40" in out["note"] and "never copy" in out["note"]


def test_a_file_that_fits_is_not_called_truncated():
    out = _csv_preview(_stores_csv(5), max_rows=20)
    assert out["total_data_rows"] == 5 and out["truncated"] is False and "note" not in out


def test_swapped_columns_are_flagged_from_every_row():
    out = _csv_preview(_stores_csv(40), max_rows=20)
    assert out["latlon_swapped"] == "likely"
    assert "Antarctica" in out["latlon_warning"]


def test_swap_beyond_90_is_certain():
    hint = latlon_swap_hint(["id", "latitude", "longitude"], [["a", "-120.5", "35.2"], ["b", "-121", "36"]])
    assert hint["latlon_swapped"] == "certain"


@pytest.mark.parametrize("rows", [
    [["a", "40.11", "-88.22"], ["b", "40.12", "-88.21"]],    # Champaign, as labelled
    [["a", "-77.85", "166.67"], ["b", "-77.84", "166.68"]],  # McMurdo really is in Antarctica
])
def test_correct_columns_are_left_alone(rows):
    assert latlon_swap_hint(["id", "lat", "lon"], rows) is None


# ------------------------------------------------------------------ typed-in data (X01)

TYPED = """import math
union = (40.1092, -88.2272)
stores = [
""" + "".join(f"    ('S{i:02d}', {40.10 + i * 0.001:.5f}, {-88.25 + i * 0.001:.5f}),\n" for i in range(19)) + """]
print(sum(1 for s in stores if math.dist(union, s[1:]) < 0.02))
"""


def test_a_table_typed_into_code_that_reads_no_file_is_flagged():
    found = code_watch.precheck(TYPED)
    rule = [f for f in found if f["rule"] == "typed_in_data"]
    assert rule and "19 records" in rule[0]["message"]
    assert "typed_in_data" in code_watch.PARAMETER_RULES  # shown whatever the run's timing


def test_code_that_reads_the_upload_is_not_flagged():
    code = "import pandas as pd\ndf = pd.read_csv('/work/inputs/stores.csv')\n" + TYPED
    assert not [f for f in code_watch.precheck(code) if f["rule"] == "typed_in_data"]


def test_a_few_constants_are_not_a_table():
    code = ("import math\npts = [(40.1, -88.2), (40.2, -88.3), (40.3, -88.4)]\n"
            "weights = [0.2, 0.3, 0.5]\nprint(pts, weights)\n")
    assert not [f for f in code_watch.precheck(code) if f["rule"] == "typed_in_data"]


# ------------------------------------------------------------------ CRS of a GeoJSON (X02)

PARCEL_BOUNDS = [1176000.0, 1900000.0, 1184344.0, 1901856.0]


def test_degrees_need_no_warning():
    assert degree_like([-88.3, 40.0, -88.1, 40.2])
    assert crs_check([-88.3, 40.0, -88.1, 40.2], "EPSG:4326") is None


def test_a_declared_projected_crs_is_consistent():
    assert crs_check(PARCEL_BOUNDS, "EPSG:3435", with_candidates=False) is None


@pytest.mark.parametrize("declared", [None, "EPSG:4326", "OGC:CRS84"])
def test_numbers_that_cannot_be_degrees_are_flagged(declared):
    pytest.importorskip("pyproj")
    warn = crs_check(PARCEL_BOUNDS, declared, with_candidates=False)
    assert warn and "Do not assume metres" in warn["crs_warning"]
    assert warn["coordinates_degree_like"] is False


def test_candidates_include_illinois_east_feet():
    pytest.importorskip("pyproj")
    warn = crs_check(PARCEL_BOUNDS, None)
    codes = [c["crs"] for c in warn.get("crs_candidates", [])]
    assert "EPSG:3435" in codes
    ill = next(c for c in warn["crs_candidates"] if c["crs"] == "EPSG:3435")
    lon, lat = ill["centre_lonlat"]
    assert -88.0 < lon < -87.5 and 41.6 < lat < 42.1  # downtown Chicago
    assert "foot" in (ill["unit"] or "").lower()


def test_geojson_preview_carries_the_warning():
    pytest.importorskip("pyproj")
    ring = [[1176000, 1900000], [1176300, 1900000], [1176300, 1900400], [1176000, 1900400],
            [1176000, 1900000]]
    text = json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"parcel": "P0"},
         "geometry": {"type": "Polygon", "coordinates": [ring]}}]})
    out = _geojson_preview(text, 12000)
    assert out["feature_count"] == 1 and out["bounds"][0] == 1176000
    assert "crs_warning" in out


# ------------------------------------------------------------------ the harness tasks

def test_trap_datasets_build_with_the_expected_answers(tmp_path):
    pytest.importorskip("pyproj")
    from gis_harness import datasets as D
    from gis_harness.tasks import BY_ID

    for tid in ("X01", "X02", "X03", "X04", "X05", "X06", "X07", "X08"):
        task = BY_ID[tid]
        files, expected = D.build(task.dataset, tmp_path / tid, tmp_path / "cache")
        assert all(p.exists() for p in files.values())
        assert all(c.name in expected for c in task.checks)
    _, e1 = D.build("trap_swapped_latlon", tmp_path / "a", tmp_path / "cache")
    assert e1["within_2km"] == 4
    _, e2 = D.build("trap_feet_no_crs", tmp_path / "b", tmp_path / "cache")
    assert 10 < e2["total_acres"] < 60
    assert e2["wrong_as_metres_acres"] / e2["total_acres"] == pytest.approx(10.76, rel=0.01)


def test_acres_are_a_unit_the_scorer_reads():
    from gis_harness.score import match_number

    assert match_number("The total is 25.3 acres.", 25.26, "acre", 0.01, 0.0)
    assert not match_number("The total is 271.94 acres.", 25.26, "acre", 0.01, 0.0)
