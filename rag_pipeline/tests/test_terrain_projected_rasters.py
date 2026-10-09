"""An uploaded PROJECTED raster (a UTM DEM) is measured in metres and drawn where it is.

Measured on prod 23cfd02, 2026-10-09, task T06: the terrain tools read a raster's transform
and never its CRS, so a 30 m DEM in EPSG:32616 was treated as degrees. terrain_derivative
reported ground_resolution_m 3,339,600 and slopes of about 5e-5 degrees, and its layer went
out with the UTM numbers as lon/lat bounds, so it was listed on the map and drew nowhere.
add_raster_layer separately failed with "CRS is invalid: None", because a one-shot PROJ repair
had gone stale on another thread.

The expected values here are worked out by hand, or taken from outside the tool: the lon/lat
of the T06 grid is what the agent's own sandbox code computed in that live run.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from agent_runtime import terrain_tools

# The T06 grid: EPSG:32616, 30 m pixels, 150 columns x 200 rows.
UTM = "EPSG:32616"
WEST, SOUTH, EAST, NORTH = 390000.0, 4434000.0, 394500.0, 4440000.0
RES = 30.0
COLS, ROWS = int((EAST - WEST) / RES), int((NORTH - SOUTH) / RES)
# Lon/lat of that grid, from transform_bounds in the agent's own sandbox code on prod (live
# run T06.f1). It is independent of the code under test.
T06_LONLAT = (-88.29058888874808, 40.049068897623876, -88.23682566908614, 40.10368947382861)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("AGENT_PUBLIC_BASE_URL", "")
    return tmp_path


def _tools():
    return {t.name: t for t in terrain_tools.make_terrain_tools()}


def _geo_tools():
    from agent_runtime.langchain_geo_tools import make_langchain_geo_tools
    return {t.name: t for t in make_langchain_geo_tools()}


def _write(tmp_path, values, crs=UTM, bounds=(WEST, SOUTH, EAST, NORTH), name="dem.tif",
           register=True):
    import rasterio
    from rasterio.transform import from_bounds

    terrain_tools._ensure_proj()
    h, w = values.shape
    path = tmp_path / name
    target = None if crs is None else rasterio.crs.CRS.from_string(crs)
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32",
                       crs=target, nodata=float("nan"),
                       transform=from_bounds(*bounds, w, h)) as dst:
        dst.write(values.astype("float32"), 1)
    if not register:
        return path
    from agent_runtime.file_store import create_output_file_from_path
    return create_output_file_from_path(path, filename=name)["file_id"]


def _east_ramp(rise_per_pixel, rows=ROWS, cols=COLS):
    """Elevation climbing *rise_per_pixel* metres per column, west to east."""
    return np.tile(200.0 + np.arange(cols, dtype="float64") * rise_per_pixel, (rows, 1))


# --- measured in metres ----------------------------------------------------------------

def test_slope_on_a_utm_grid_matches_the_hand_computed_value(store, tmp_path):
    """15 m of rise per 30 m pixel is atan(15/30) = 26.565 degrees. Read as degrees, the
    same grid gave about 5e-5 degrees on prod."""
    rid = _write(tmp_path, _east_ramp(15.0))

    out = json.loads(_tools()["terrain_derivative"].func(raster_file_id=rid, kind="slope"))

    assert out["ok"] is True, out
    expected = math.degrees(math.atan(15.0 / 30.0))
    assert expected == pytest.approx(26.565, abs=1e-3)
    # The projected step measured on the ellipsoid is 30 m to within UTM's own scale error
    # (about 0.03% here), so the slope agrees to a few thousandths of a degree.
    assert out["max"] == pytest.approx(expected, abs=0.05), out
    assert out["mean"] == pytest.approx(expected, abs=0.05), out
    assert out["ground_resolution_m"] == pytest.approx(30.0, abs=0.05), out
    assert out["crs"] == UTM
    assert out["native_bounds"] == [WEST, SOUTH, EAST, NORTH]


def test_flooded_area_on_a_utm_grid_is_in_square_kilometres(store, tmp_path):
    """150 x 200 pixels of 30 m is 27 km2. A level halfway up the ramp floods the western
    75 columns, so 13.5 km2. Read as degrees, a pixel was a 3,339 km square."""
    rid = _write(tmp_path, _east_ramp(1.0))       # 200 m .. 349 m

    out = json.loads(_tools()["inundation_at_level"].func(raster_file_id=rid,
                                                          level_m=200.0 + 74.0))

    assert out["ok"] is True, out
    assert out["region_km2"] == pytest.approx(27.0, rel=2e-3), out
    assert out["flooded_km2"] == pytest.approx(13.5, rel=2e-3), out
    assert out["flooded_fraction"] == pytest.approx(0.5, abs=1e-3), out


def test_web_mercator_metres_are_converted_to_ground_metres(store, tmp_path):
    """Web Mercator's "metres" are 1/cos(lat) too long. A 30-unit pixel at 40N is
    30 * cos(40) = 22.98 m on the ground. A tool that trusted the CRS's units would understate
    every slope by that factor."""
    y0 = 4_865_942.0                              # about 40.0N in EPSG:3857
    bounds = (-9_830_000.0, y0, -9_830_000.0 + 40 * 30.0, y0 + 40 * 30.0)
    rid = _write(tmp_path, _east_ramp(15.0, rows=40, cols=40), crs="EPSG:3857", bounds=bounds)

    out = json.loads(_tools()["terrain_derivative"].func(raster_file_id=rid, kind="slope"))

    assert out["ok"] is True, out
    ground = 30.0 * math.cos(math.radians(40.0))
    assert out["ground_resolution_m"] == pytest.approx(ground, rel=5e-3), out
    assert out["max"] == pytest.approx(math.degrees(math.atan(15.0 / ground)), abs=0.1), out


# --- drawn where it is -------------------------------------------------------------------

def test_derived_layer_lands_at_the_grids_real_lon_lat(store, tmp_path):
    rid = _write(tmp_path, _east_ramp(15.0))

    out = json.loads(_tools()["terrain_derivative"].func(raster_file_id=rid, kind="slope"))

    drawn = out["map_layer"]["bounds"]
    assert drawn == pytest.approx(list(T06_LONLAT), abs=2e-4), drawn   # under one pixel
    assert out["region_bbox"] == pytest.approx(list(T06_LONLAT), abs=1e-5)
    assert all(abs(v) <= 180 for v in drawn)


def test_derived_geotiff_keeps_the_sources_crs(store, tmp_path):
    """The slope of a UTM DEM was written as a file claiming EPSG:4326 with bounds of
    390000..394500, so anything that trusted the label misplaced it too."""
    import rasterio

    from agent_runtime.file_store import resolve_file_id

    rid = _write(tmp_path, _east_ramp(15.0))
    out = json.loads(_tools()["terrain_derivative"].func(raster_file_id=rid, kind="slope"))

    with rasterio.open(resolve_file_id(out["geotiff"]["file_id"])) as src:
        assert src.crs.to_epsg() == 32616
        assert tuple(src.bounds) == pytest.approx((WEST, SOUTH, EAST, NORTH))


def test_a_marked_pixel_is_drawn_at_its_own_lon_lat(store, tmp_path):
    """Placing the box right is not enough; the pixels inside it must line up too. UTM grid
    north here is turned about 0.8 degrees from true north, so an unwarped image stretched over
    its lon/lat box misplaces a corner by about 50 m while its centre stays put. A 3x3 NoData
    hole near the NE corner, read back from the drawn PNG through the layer's bounds, has to
    land within 20 m of where it really is."""
    from PIL import Image

    from agent_runtime.file_store import resolve_file_id

    values = _east_ramp(1.0)
    values[9:12, 139:142] = np.nan                # centred on (394215, 4439685)
    rid = _write(tmp_path, values)

    out = json.loads(_geo_tools()["add_raster_layer"].func(file_id=rid))
    assert out["ok"] is True, out
    rgba = np.asarray(Image.open(resolve_file_id(out["file_id"])))
    h, w = rgba.shape[:2]
    # The warped grid's empty corner wedges are transparent too; the hole is the transparent
    # patch a few pixels inside the frame.
    holes = [(r, c) for r, c in np.argwhere(rgba[..., 3] == 0)
             if 4 <= r < 20 and w - 20 <= c < w - 4]
    assert len(holes) >= 4, len(holes)
    r, c = np.mean(holes, axis=0)
    minlon, minlat, maxlon, maxlat = out["bounds"]
    lon = minlon + (c + 0.5) / w * (maxlon - minlon)
    lat = maxlat - (r + 0.5) / h * (maxlat - minlat)
    # pyproj's EPSG:32616 -> 4326 for (394215, 4439685), hard-coded.
    east_m = (lon - -88.24109549) * 111_320.0 * math.cos(math.radians(40.1))
    north_m = (lat - 40.10081632) * 110_540.0
    assert math.hypot(east_m, north_m) < 20.0, (east_m, north_m)


def test_add_raster_layer_places_a_utm_geotiff_without_bounds(store, tmp_path):
    rid = _write(tmp_path, _east_ramp(1.0))

    out = json.loads(_geo_tools()["add_raster_layer"].func(file_id=rid))

    assert out["ok"] is True, out
    assert out["bounds"] == pytest.approx(list(T06_LONLAT), abs=2e-4)
    assert out["map_layer"]["bounds"] == out["bounds"]


def test_zonal_stats_puts_lon_lat_polygons_on_a_utm_grid(store, tmp_path):
    """Zones arrive in lon/lat and the grid is in metres. The zones are reprojected onto the
    grid, so the western and eastern halves come out with their own means."""
    import geopandas as gpd
    from shapely.geometry import box

    from agent_runtime.file_store import create_output_file_from_path

    values = np.full((ROWS, COLS), 100.0)
    values[:, COLS // 2:] = 300.0
    rid = _write(tmp_path, values)
    mid_x = WEST + (COLS // 2) * RES
    zones = gpd.GeoDataFrame(
        {"GEOID": ["west", "east"]},
        geometry=[box(WEST + 300, SOUTH + 300, mid_x - 300, NORTH - 300),
                  box(mid_x + 300, SOUTH + 300, EAST - 300, NORTH - 300)],
        crs=UTM).to_crs(4326)
    zpath = tmp_path / "zones.geojson"
    zones.to_file(zpath, driver="GeoJSON")
    zid = create_output_file_from_path(zpath, filename="zones.geojson")["file_id"]

    out = json.loads(_tools()["zonal_stats_for_raster"].func(
        raster_file_id=rid, polygons_file_id=zid, zone_id_field="GEOID", prefix="elev"))

    assert out["ok"] is True, out
    by_zone = {p["zone"]: p for p in out["preview"]}
    assert by_zone["west"]["elev_mean"] == 100.0
    assert by_zone["east"]["elev_mean"] == 300.0
    assert by_zone["west"]["elev_coverage"] > 0.95
    assert out["raster_bounds"] == pytest.approx(list(T06_LONLAT), abs=1e-5)


# --- refused rather than guessed -------------------------------------------------------

def test_no_crs_and_coordinates_that_are_not_degrees_is_refused(store, tmp_path):
    """The original failure, minus the CRS tag: guessing degrees would bring it back."""
    rid = _write(tmp_path, _east_ramp(15.0), crs=None)

    out = json.loads(_tools()["terrain_derivative"].func(raster_file_id=rid, kind="slope"))

    assert out["ok"] is False
    assert "no CRS" in out["error"] and "not degrees" in out["error"]
    assert "EPSG" in out["hint"]


# --- PROJ ------------------------------------------------------------------------------

def test_a_second_proj_break_is_repaired_too(monkeypatch):
    """Prod 23cfd02: after one repair, a rasterio.Env on a worker thread broke PROJ again for
    the next fresh thread, and the one-shot flag refused to repair twice, so _wgs84() was None.
    The repair is now probed per call. Here the probe fails before each of two calls."""
    import rasterio._env

    states = iter([False, True, False, True])
    repairs = []
    monkeypatch.setattr(terrain_tools, "_proj_ok", lambda: next(states))
    monkeypatch.setattr(rasterio._env, "set_proj_data_search_path", repairs.append)

    assert terrain_tools._wgs84() is not None
    assert terrain_tools._wgs84() is not None
    assert len(repairs) == 2


_POISONED_RUN = r'''
import json, os, shutil, sqlite3, sys, tempfile, threading
import rasterio
from rasterio._env import set_proj_data_search_path

# The deployed container's database: proj.db with DATABASE.LAYOUT minor version 5.
src = os.path.join(os.path.dirname(rasterio.__file__), "proj_data", "proj.db")
d = tempfile.mkdtemp(); db = os.path.join(d, "proj.db"); shutil.copy(src, db)
con = sqlite3.connect(db)
con.execute("update metadata set value='5' where key='DATABASE.LAYOUT.VERSION.MINOR'")
con.commit(); con.close()
set_proj_data_search_path(d)
try:
    rasterio.crs.CRS.from_epsg(4326); broken = False
except Exception:
    broken = True

from agent_runtime import terrain_tools
from agent_runtime.file_store import create_output_file_from_path
fid = create_output_file_from_path(sys.argv[1], filename="dem.tif")["file_id"]
tools = {t.name: t for t in terrain_tools.make_terrain_tools()}
result = {}
def run():   # a worker thread, as the agent runs tools
    result["out"] = json.loads(tools["terrain_derivative"].func(raster_file_id=fid))
t = threading.Thread(target=run); t.start(); t.join()
print(json.dumps({"broken": broken, "out": result["out"]}))
'''


def test_terrain_derivative_works_under_the_deployed_proj_database(tmp_path):
    """The prod condition, reproduced faithfully: rasterio pointed at a proj.db whose layout is
    one minor version too old. Run in a fresh process, because PROJ caches its database and
    the condition cannot be re-created once a process has repaired it."""
    dem = _write(tmp_path, _east_ramp(15.0), register=False)
    env = dict(os.environ, AGENT_FILE_STORAGE_ROOT=str(tmp_path / "store"),
               AGENT_PUBLIC_BASE_URL="",
               PYTHONPATH=str(Path(terrain_tools.__file__).resolve().parents[1]))
    proc = subprocess.run([sys.executable, "-c", _POISONED_RUN, str(dem)], env=env,
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr[-2000:]
    got = json.loads(proc.stdout.strip().splitlines()[-1])

    assert got["broken"] is True                  # the condition really was reproduced
    out = got["out"]
    assert out["ok"] is True, out
    assert out["crs"] == UTM
    assert out["max"] == pytest.approx(math.degrees(math.atan(0.5)), abs=0.05)
    assert out["map_layer"]["bounds"] == pytest.approx(list(T06_LONLAT), abs=2e-4)
