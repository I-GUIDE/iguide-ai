"""Steepness, and what lies under a given height — the two things a DEM is next asked for.

Both are arithmetic on a raster dem_for_region already fetched, which makes them cheap and
makes them easy to get quietly wrong: a slope computed in DEGREES of longitude rather than
metres is a number that changes with latitude while looking entirely plausible, and a bathtub
fill reads as a floodplain to anyone who is not told otherwise.

The arithmetic is checked against surfaces whose answer is known by construction — a ramp of
exactly 45 degrees, a flat plane, a half-submerged basin — rather than against whatever the
implementation happens to return.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from agent_runtime import terrain_tools


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("AGENT_PUBLIC_BASE_URL", "")
    return tmp_path


def _tools():
    return {t.name: t for t in terrain_tools.make_terrain_tools()}


def _raster(tmp_path, values, bbox):
    import rasterio
    from rasterio.transform import from_bounds

    from agent_runtime.file_store import create_output_file_from_path

    h, w = values.shape
    path = tmp_path / "dem.tif"
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32",
                       crs=terrain_tools._wgs84(), nodata=float("nan"),
                       transform=from_bounds(*bbox, w, h)) as dst:
        dst.write(values.astype("float32"), 1)
    return create_output_file_from_path(path, filename="dem.tif")["file_id"]


def _equator_box(n, metres_per_pixel):
    """A bbox at the equator whose pixels are *metres_per_pixel* across.

    On the equator cos(lat) is 1, so a degree of longitude is the full 111_320 m the tool uses
    and the expected slope can be worked out on paper. Away from it the x spacing shrinks, which
    is the whole reason the tool converts at all — tested separately below.
    """
    span = n * metres_per_pixel / 111_320.0
    return (0.0, 0.0, span, span * 110_540.0 / 110_540.0)


def test_slope_of_a_45_degree_ramp_is_45_degrees(store, tmp_path):
    """A ramp rising exactly one pixel-width per pixel is 45 degrees, whatever the pixel size.
    If the gradient were taken in degrees of longitude instead of metres this lands nowhere
    near 45, which is the failure that looks most like a plausible number."""
    n, res = 40, 30.0
    # Rows run north->south in the raster, columns west->east; the ramp climbs eastwards.
    values = np.tile(np.arange(n, dtype="float64") * res, (n, 1))
    rid = _raster(tmp_path, values, _equator_box(n, res))

    out = json.loads(_tools()["terrain_derivative"].func(raster_file_id=rid, kind="slope"))

    assert out["ok"] is True, out
    # The interior is exactly 45; the frame's edge uses a one-sided difference, so the mean
    # over the whole grid is compared loosely and the maximum tightly.
    assert out["max"] == pytest.approx(45.0, abs=0.5), out
    assert out["mean"] == pytest.approx(45.0, abs=1.0), out
    assert out["mean_percent"] == pytest.approx(100.0, abs=5.0), out


def test_a_flat_plane_has_no_slope_and_no_aspect(store, tmp_path):
    n = 20
    values = np.full((n, n), 250.0)
    rid = _raster(tmp_path, values, _equator_box(n, 30.0))

    slope = json.loads(_tools()["terrain_derivative"].func(raster_file_id=rid, kind="slope"))
    assert slope["max"] == pytest.approx(0.0, abs=1e-6)

    # Flat ground faces nowhere. Reporting 0 would draw a hard "due north" band over every
    # plain in the region, which reads as a real finding.
    aspect = json.loads(_tools()["terrain_derivative"].func(raster_file_id=rid, kind="aspect"))
    assert aspect["ok"] is True
    assert "mean" not in aspect or aspect.get("mean") is None


def test_slope_accounts_for_longitude_shrinking_with_latitude(store, tmp_path):
    """The same grid of degrees is a NARROWER strip of ground at 60N than at the equator, so
    the same rise over it is a STEEPER slope. A tool that never converted would report the two
    as identical."""
    n, res = 40, 30.0
    values = np.tile(np.arange(n, dtype="float64") * res, (n, 1))
    span = n * res / 111_320.0

    at_eq = _raster(tmp_path, values, (0.0, 0.0, span, span))
    out_eq = json.loads(_tools()["terrain_derivative"].func(raster_file_id=at_eq, kind="slope"))

    far_north = _raster(tmp_path, values, (0.0, 60.0, span, 60.0 + span))
    out_n = json.loads(_tools()["terrain_derivative"].func(raster_file_id=far_north,
                                                          kind="slope"))

    assert out_n["mean"] > out_eq["mean"], (out_eq["mean"], out_n["mean"])


def test_nodata_gets_no_slope_rather_than_an_invented_one(store, tmp_path):
    """np.gradient over NaN spreads it; filling the hole to keep the arithmetic alive would
    invent a gradient at the edge of the data. The hole has to come back."""
    n = 20
    values = np.tile(np.arange(n, dtype="float64") * 30.0, (n, 1))
    values[5:8, 5:8] = np.nan
    rid = _raster(tmp_path, values, _equator_box(n, 30.0))

    out = json.loads(_tools()["terrain_derivative"].func(raster_file_id=rid, kind="slope"))

    assert out["ok"] is True
    from agent_runtime.file_store import resolve_file_id
    import rasterio
    with rasterio.open(resolve_file_id(out["geotiff"]["file_id"])) as src:
        grid = src.read(1)
    assert np.isnan(grid[6, 6])


def test_unknown_kind_names_the_ones_that_exist(store, tmp_path):
    rid = _raster(tmp_path, np.full((10, 10), 1.0), _equator_box(10, 30.0))

    out = json.loads(_tools()["terrain_derivative"].func(raster_file_id=rid, kind="ruggedness"))

    assert out["ok"] is False
    assert "slope" in out["hint"] and "hillshade" in out["hint"]


# --- inundation ------------------------------------------------------------------------------

def test_half_the_basin_floods_at_the_midpoint(store, tmp_path):
    """Half the pixels are at 10 m and half at 30 m, so a 20 m level floods exactly half the
    area — a figure that can be checked without rerunning the implementation."""
    n = 40
    values = np.full((n, n), 30.0)
    values[:, : n // 2] = 10.0
    rid = _raster(tmp_path, values, _equator_box(n, 30.0))

    out = json.loads(_tools()["inundation_at_level"].func(raster_file_id=rid, level_m=20.0))

    assert out["ok"] is True, out
    assert out["flooded_fraction"] == pytest.approx(0.5, abs=0.01)
    assert out["mean_depth_m"] == pytest.approx(10.0, abs=0.01)
    assert out["max_depth_m"] == pytest.approx(10.0, abs=0.01)


def test_a_depth_above_the_regions_own_minimum(store, tmp_path):
    """The absolute elevation of a valley floor is usually not known, so the level can be given
    relative to it. 5 m above a 10 m floor must flood the 10 m half and nothing else."""
    n = 40
    values = np.full((n, n), 30.0)
    values[:, : n // 2] = 10.0
    rid = _raster(tmp_path, values, _equator_box(n, 30.0))

    out = json.loads(_tools()["inundation_at_level"].func(raster_file_id=rid,
                                                          depth_above_min_m=5.0))

    assert out["level_m"] == pytest.approx(15.0)
    assert out["flooded_fraction"] == pytest.approx(0.5, abs=0.01)
    assert "above the region minimum" in out["level_from"]


def test_a_level_below_everything_says_so_instead_of_drawing_nothing(store, tmp_path):
    """Nothing flooding is a real answer, but an empty layer with ok=true looks like a tool
    that failed quietly."""
    n = 20
    rid = _raster(tmp_path, np.full((n, n), 100.0), _equator_box(n, 30.0))

    out = json.loads(_tools()["inundation_at_level"].func(raster_file_id=rid, level_m=5.0))

    assert out["ok"] is True
    assert out["flooded_km2"] == 0.0
    assert "nothing is below" in out["verdict"]


def test_no_level_at_all_is_refused_with_both_ways_to_give_one(store, tmp_path):
    n = 10
    rid = _raster(tmp_path, np.full((n, n), 100.0), _equator_box(n, 30.0))

    out = json.loads(_tools()["inundation_at_level"].func(raster_file_id=rid))

    assert out["ok"] is False
    assert "level_m" in out["hint"] and "depth_above_min_m" in out["hint"]


def test_every_answer_says_it_is_not_a_floodplain(store, tmp_path):
    """The single most likely misreading of this tool, and the one with consequences: a bathtub
    fill presented as a regulatory floodplain. The disclaimer travels in the payload rather
    than only in the docstring, because the docstring is not what reaches the user."""
    n = 20
    values = np.full((n, n), 30.0)
    values[:, :10] = 10.0
    rid = _raster(tmp_path, values, _equator_box(n, 30.0))

    out = json.loads(_tools()["inundation_at_level"].func(raster_file_id=rid, level_m=20.0))

    assert "floodplain" in out["note"]
    assert "bathtub" in out["note"].lower()


def test_the_depth_raster_feeds_zonal_stats(store, tmp_path):
    """The reason inundation writes a depth GeoTIFF rather than only a picture: it is meant to
    go straight into zonal_stats_for_raster for a per-zone flooded figure."""
    from agent_runtime.file_store import create_output_file_from_path

    n = 40
    bbox = _equator_box(n, 30.0)
    values = np.full((n, n), 30.0)
    values[:, : n // 2] = 10.0
    rid = _raster(tmp_path, values, bbox)

    flooded = json.loads(_tools()["inundation_at_level"].func(raster_file_id=rid, level_m=20.0))

    x0, y0, x1, y1 = bbox
    zones = {"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {"GEOID": "west"},
        "geometry": {"type": "Polygon", "coordinates": [[
            [x0, y0], [(x0 + x1) / 2, y0], [(x0 + x1) / 2, y1], [x0, y1], [x0, y0]]]}}]}
    zp = tmp_path / "zones.geojson"
    zp.write_text(json.dumps(zones), encoding="utf-8")
    zid = create_output_file_from_path(zp, filename="zones.geojson")["file_id"]

    out = json.loads(_tools()["zonal_stats_for_raster"].func(
        raster_file_id=flooded["depth_geotiff"]["file_id"], polygons_file_id=zid,
        zone_id_field="GEOID", prefix="depth"))

    assert out["ok"] is True, out
    # The western zone is the flooded half, 10 m under throughout.
    assert out["preview"][0]["depth_mean"] == pytest.approx(10.0, abs=0.1)
