"""A GeoTIFF reaches the map on the first call, not the fourth.

Measured live: `dem_for_region` produced a .tif and a .png, and it took FOUR tool calls to draw
one of them — add_map_layer on the tif ("unreadable vector/tabular source"), add_raster_layer on
the tif ("not an image"), add_map_layer on the png ("an image has no geometry"), and finally
add_raster_layer on the png. Every message was accurate and none of them was useful, because a
GeoTIFF is neither of the two things these tools knew about.

It is a raster, and it carries its own georeferencing — which is also why the bounds are read
from the file rather than restated by the caller. A draped image is positioned solely by its
bounds and nothing downstream can check them against the pixels, so a restated box draws a
plausible layer in the wrong place. That misregistration has already cost three rounds of
debugging once.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

rasterio = pytest.importorskip("rasterio")
import numpy as np  # noqa: E402

from agent_runtime import file_store  # noqa: E402
from agent_runtime.langchain_geo_tools import make_langchain_geo_tools  # noqa: E402

BOX = (-88.30, 40.05, -88.10, 40.20)      # minlon, minlat, maxlon, maxlat


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path))
    return tmp_path


def tools():
    return {t.name: t for t in make_langchain_geo_tools()}


def a_geotiff(tmp_path, crs="EPSG:4326", box=BOX) -> str:
    """A small real DEM-shaped raster, written through the file store."""
    from agent_runtime.terrain_tools import _wgs84
    path = tmp_path / "area_dem.tif"
    h = w = 16
    data = np.linspace(180.0, 260.0, h * w, dtype="float32").reshape(h, w)
    transform = rasterio.transform.from_bounds(*box, w, h)
    target = _wgs84() if crs == "EPSG:4326" else rasterio.crs.CRS.from_string(crs)
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1,
                       dtype="float32", crs=target, transform=transform) as dst:
        dst.write(data, 1)
    return file_store.create_output_file_from_path(str(path), filename="area_dem.tif")["file_id"]


# --- the first call works --------------------------------------------------------

def test_a_geotiff_drapes_with_no_bounds_given(tmp_path):
    """The whole point: the model's first instinct — pass the tif — now succeeds."""
    out = json.loads(tools()["add_raster_layer"].func(file_id=a_geotiff(tmp_path)))
    assert out.get("ok") is not False, out
    layer = out.get("layer") or out
    drawn = [round(float(v), 2) for v in (layer.get("bounds") or [])]
    assert drawn == [round(v, 2) for v in BOX]


def test_the_bounds_come_from_the_file_not_the_caller(tmp_path):
    """A restated box draws a plausible layer in the wrong place, and nothing downstream can
    tell. The file knows; the caller only remembers."""
    out = json.loads(tools()["add_raster_layer"].func(
        file_id=a_geotiff(tmp_path), bounds=[0.0, 0.0, 1.0, 1.0]))
    layer = out.get("layer") or out
    drawn = [round(float(v), 2) for v in (layer.get("bounds") or [])]
    assert drawn == [round(v, 2) for v in BOX]
    assert drawn != [0.0, 0.0, 1.0, 1.0]


def test_a_projected_raster_is_reported_in_lon_lat(tmp_path):
    """Web maps are 4326. Projected metres would land the layer off the map entirely."""
    fid = a_geotiff(tmp_path, crs="EPSG:3857",
                    box=(-9830000.0, 4870000.0, -9810000.0, 4890000.0))
    out = json.loads(tools()["add_raster_layer"].func(file_id=fid))
    layer = out.get("layer") or out
    minlon, minlat, maxlon, maxlat = [float(v) for v in layer["bounds"]]
    assert -180 <= minlon < maxlon <= 180 and -90 <= minlat < maxlat <= 90
    assert -89 < minlon < -87 and 39 < minlat < 41          # back in Illinois


# --- the redirect that used to be a dead end --------------------------------------

def test_add_map_layer_sends_a_raster_somewhere_useful(tmp_path):
    out = json.loads(tools()["add_map_layer"].func(file_id=a_geotiff(tmp_path)))
    assert out["ok"] is False
    assert "raster" in out["error"].lower()
    # The old message said "unreadable vector/tabular source" and hinted at shapefile sidecars,
    # which was true, unhelpful, and sent the model looking for a .shx that does not exist.
    assert "unreadable" not in out["error"].lower()
    assert "add_raster_layer" in out["hint"]
    assert "same file_id" in out["hint"]
    assert "do not need to supply" in out["hint"]           # and no bounds to invent


def test_an_ordinary_image_still_asks_for_the_dataset(tmp_path):
    """A PNG genuinely has no geometry; that redirect was already right and must not regress."""
    png = tmp_path / "plot.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    fid = file_store.create_output_file_from_path(str(png), filename="plot.png")["file_id"]
    out = json.loads(tools()["add_map_layer"].func(file_id=fid))
    assert out["ok"] is False and "no geometry" in out["error"]


def test_a_vector_file_is_untouched_by_any_of_this(tmp_path):
    gj = tmp_path / "pts.geojson"
    gj.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"n": 1},
         "geometry": {"type": "Point", "coordinates": [-88.2, 40.1]}}]}))
    fid = file_store.create_output_file_from_path(str(gj), filename="pts.geojson")["file_id"]
    out = json.loads(tools()["add_map_layer"].func(file_id=fid))
    assert out.get("ok") is not False, out
