"""A raster summarised inside each polygon, as columns the zone tools already take.

This is the join that lets elevation meet the satellite embeddings: embed_zones writes one row
per zone keyed by zone_id_field, fit_zone_model wants a polygon layer carrying the truth in an
attribute column, and nothing turned a raster into that column. dem_for_region -> here ->
fit_zone_model is the whole path, so the tests that matter are the ones about the JOIN holding:
the column names, the zone identity, and the honesty of a number computed from partial data.

Two failures are specifically guarded. A zone lying half outside the raster still produces a
mean, from whichever pixels happened to be inside — silently, unless coverage is reported, so
coverage is asserted rather than assumed. And src.crs is never read: the deployed container's
PROJ database is too old for rasterio, which is why dem_for_region reads only src.transform.
A summariser that reaches for the CRS would pass here and fail in production.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from agent_runtime import terrain_tools

# The raster frame every test works in. Small and rectangular so a zone can be placed to cover
# a known number of cells, and the arithmetic can be checked by hand rather than by rerunning
# the implementation.
BBOX = (-88.30, 40.00, -88.20, 40.10)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("AGENT_PUBLIC_BASE_URL", "")
    return tmp_path


def _tool():
    return {t.name: t for t in terrain_tools.make_terrain_tools()}["zonal_stats_for_raster"]


def _write_raster(tmp_path, values: np.ndarray, bbox=BBOX, nodata=None):
    """A real GeoTIFF on disk, registered in the file store, returning its file_id.

    Real rather than mocked because the point of this tool is what rasterio hands back — the
    masked read, the nodata fill and the transform — and a fake would assert our idea of that
    instead of the library's.

    The CRS comes from terrain_tools._wgs84() for the reason test_dem_for_region gives: this
    suite poisons PROJ when qgis is imported, so a literal "EPSG:4326" is order-dependent here
    in exactly the way the deployed container is permanently.
    """
    import rasterio
    from rasterio.transform import from_bounds

    from agent_runtime.file_store import create_output_file_from_path

    h, w = values.shape
    path = tmp_path / "dem.tif"
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1,
                       dtype="float32", crs=terrain_tools._wgs84(),
                       nodata=nodata if nodata is not None else float("nan"),
                       transform=from_bounds(*bbox, w, h)) as dst:
        dst.write(values.astype("float32"), 1)
    return create_output_file_from_path(path, filename="dem.tif")["file_id"]


def _write_zones(tmp_path, polys, ids=None, name="zones.geojson"):
    """A GeoJSON polygon layer with a GEOID field, registered in the store."""
    from agent_runtime.file_store import create_output_file_from_path

    feats = []
    for i, (x0, y0, x1, y1) in enumerate(polys):
        feats.append({
            "type": "Feature",
            "properties": {"GEOID": (ids[i] if ids else f"z{i}"), "other": i},
            "geometry": {"type": "Polygon", "coordinates": [[
                [x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]},
        })
    path = tmp_path / name
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}),
                    encoding="utf-8")
    return create_output_file_from_path(path, filename=name)["file_id"]


def _run(raster_id, zones_id, **kw):
    return json.loads(_tool().func(raster_file_id=raster_id, polygons_file_id=zones_id, **kw))


# The left half of the frame is 100 m, the right half 300 m. A zone over the left half must
# come back with mean exactly 100 — a mean of 200 would mean the zone mask is not being applied
# and the whole frame is being averaged instead, which is the failure that looks most like
# success.
def _split_values(n=20):
    values = np.full((n, n), 300.0)
    values[:, : n // 2] = 100.0
    return values


def test_zone_is_summarised_from_its_own_pixels(store, tmp_path):
    raster = _write_raster(tmp_path, _split_values())
    # A zone covering the left half only.
    zones = _write_zones(tmp_path, [(-88.30, 40.00, -88.255, 40.10)])

    out = _run(raster, zones, prefix="elev", zone_id_field="GEOID")

    assert out["ok"] is True
    assert out["zones_with_values"] == 1
    assert out["preview"][0]["elev_mean"] == 100.0
    assert out["preview"][0]["zone"] == "z0"


def test_columns_are_prefixed_so_two_rasters_do_not_collide(store, tmp_path):
    raster = _write_raster(tmp_path, _split_values())
    zones = _write_zones(tmp_path, [(-88.30, 40.00, -88.20, 40.10)])

    out = _run(raster, zones, prefix="slope")

    assert out["columns_added"] == ["slope_mean", "slope_min", "slope_max", "slope_relief",
                                    "slope_std", "slope_coverage", "slope_pixels"]
    assert out["map_layer"]["style_by"] == "slope_mean"


def test_relief_is_within_the_zone_not_across_the_region(store, tmp_path):
    """The zone spans both halves, so its own relief is 200 — the region's is also 200 here,
    so the zone is deliberately given a value the REGION does not have: a third band."""
    values = _split_values()
    values[0, :] = 1000.0                       # a high row only the top zone contains
    raster = _write_raster(tmp_path, values)
    zones = _write_zones(tmp_path, [(-88.30, 40.095, -88.20, 40.10),    # top row only
                                    (-88.30, 40.00, -88.20, 40.05)])    # bottom half only

    out = _run(raster, zones, prefix="elev", zone_id_field="GEOID")
    by_zone = {p["zone"]: p for p in out["preview"]}

    assert by_zone["z0"]["elev_mean"] == 1000.0          # the high row alone
    assert by_zone["z1"]["elev_relief"] == 200.0          # 300 - 100, not 1000 - 100


def test_partial_coverage_is_reported_not_hidden(store, tmp_path):
    """Half the zone hangs off the raster. The mean is still computed — from the pixels that
    were inside — so the only thing standing between that and a confidently wrong answer is
    coverage being stated."""
    raster = _write_raster(tmp_path, _split_values())
    zones = _write_zones(tmp_path, [(-88.35, 40.00, -88.25, 40.10)])   # extends west of frame

    out = _run(raster, zones, prefix="elev")

    assert out["ok"] is True
    assert out["zones_partially_covered"] == 1
    assert "warning" in out and "coverage" in out["warning"]


def test_nodata_pixels_do_not_drag_the_mean_down(store, tmp_path):
    """3DEP's sentinel is -999999, and a mean that includes it is catastrophically wrong rather
    than slightly wrong — the failure is loud, which is the only reason it would be caught."""
    values = np.full((20, 20), 100.0)
    values[:, :10] = -999999.0
    raster = _write_raster(tmp_path, values, nodata=-999999.0)
    zones = _write_zones(tmp_path, [(-88.30, 40.00, -88.20, 40.10)])

    out = _run(raster, zones, prefix="elev")

    assert out["preview"][0]["elev_mean"] == 100.0
    assert out["preview"][0]["elev_coverage"] == 0.5


def test_all_nodata_zone_is_refused_with_a_reason(store, tmp_path):
    values = np.full((20, 20), np.nan)
    raster = _write_raster(tmp_path, values)
    zones = _write_zones(tmp_path, [(-88.30, 40.00, -88.20, 40.10)])

    out = _run(raster, zones)

    assert out["ok"] is False
    assert "NoData" in out["error"]


def test_no_overlap_names_both_extents(store, tmp_path):
    """A dead end costs a whole turn, so the failure returns the two bounding boxes rather than
    only saying no."""
    raster = _write_raster(tmp_path, _split_values())
    zones = _write_zones(tmp_path, [(-70.0, 10.0, -69.9, 10.1)])

    out = _run(raster, zones)

    assert out["ok"] is False
    assert "raster_bounds" in out and "polygons_bounds" in out


def test_unknown_zone_id_field_lists_the_real_ones(store, tmp_path):
    raster = _write_raster(tmp_path, _split_values())
    zones = _write_zones(tmp_path, [(-88.30, 40.00, -88.20, 40.10)])

    out = _run(raster, zones, zone_id_field="TRACTCE")

    assert out["ok"] is False
    assert "GEOID" in out["available_fields"]


def test_output_layer_keeps_the_join_key_and_the_originals(store, tmp_path):
    """fit_zone_model matches this layer to the embed_zones CSV on zone_id_field. Dropping the
    original attributes would break that join while still producing a plausible-looking file."""
    raster = _write_raster(tmp_path, _split_values())
    zones = _write_zones(tmp_path, [(-88.30, 40.00, -88.255, 40.10)], ids=["17019000100"])

    out = _run(raster, zones, prefix="elev", zone_id_field="GEOID")

    from agent_runtime.file_store import resolve_file_id
    written = json.loads(open(resolve_file_id(out["geojson"]["file_id"]),
                              encoding="utf-8").read())
    props = written["features"][0]["properties"]
    assert props["GEOID"] == "17019000100"
    assert props["other"] == 0
    assert props["elev_mean"] == 100.0


def test_the_crs_is_never_read_from_the_raster(store, tmp_path, monkeypatch):
    """The deployed container's PROJ database is older than rasterio's PROJ accepts, so
    src.crs RAISES there and passes on a laptop. dem_for_region reads only src.transform for
    exactly this reason; this asserts the summariser inherited that discipline rather than
    reintroducing the production-only failure one module over.
    """
    import rasterio

    raster = _write_raster(tmp_path, _split_values())
    zones = _write_zones(tmp_path, [(-88.30, 40.00, -88.20, 40.10)])

    real_open = rasterio.open

    class _NoCRS:
        def __init__(self, inner):
            self._inner = inner

        def __enter__(self):
            self._ds = self._inner.__enter__()
            return self

        def __exit__(self, *a):
            return self._inner.__exit__(*a)

        @property
        def crs(self):
            raise RuntimeError("PROJ: database layout version 5 < 6 (as deployed)")

        def __getattr__(self, item):
            return getattr(self._ds, item)

    monkeypatch.setattr(rasterio, "open", lambda *a, **k: _NoCRS(real_open(*a, **k)))

    out = _run(raster, zones, prefix="elev")

    assert out["ok"] is True
    assert out["preview"][0]["elev_mean"] == 200.0
