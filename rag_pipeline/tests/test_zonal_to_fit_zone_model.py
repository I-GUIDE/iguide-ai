"""The whole point, end to end: dem_for_region -> zonal_stats_for_raster -> fit_zone_model.

zonal_stats_for_raster exists to produce the ONE thing that stood between elevation and the
satellite embeddings — a polygon layer carrying a per-zone number in an attribute column, which
is exactly what fit_zone_model already takes. Asserting that in a docstring is free; this file
runs it, because the join has four separate ways to be silently wrong:

  * the label column is named something fit_zone_model cannot find,
  * zone_id_field is dropped from the output, so the merge finds nothing,
  * the ids stringify differently on the two sides ("17019000100" vs 17019000100.0),
  * the geometry survives but not in a CRS the fitter can take centroids in.

Every one of those produces a file that looks correct and a fit that quietly sees zero rows.
The fit is run with a synthetic signal, so a failure here is a plumbing failure and not a
statement about whether embeddings predict terrain.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from agent_runtime import rs_embed_tools, terrain_tools

# Enough zones to clear the fitter's floor: it refuses under 12 with both a vector and a label,
# because spatial-block CV has to hold out whole blocks and still leave some to train on.
GRID = 4          # 4x4 = 16 zones
BBOX = (-88.40, 40.00, -88.00, 40.40)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("AGENT_PUBLIC_BASE_URL", "")
    return tmp_path


def _zone_grid():
    """A GRIDxGRID lattice of square zones tiling BBOX, with GEOIDs like the real thing."""
    x0, y0, x1, y1 = BBOX
    dx, dy = (x1 - x0) / GRID, (y1 - y0) / GRID
    feats = []
    for r in range(GRID):
        for c in range(GRID):
            a, b = x0 + c * dx, y0 + r * dy
            feats.append({
                "type": "Feature",
                "properties": {"GEOID": f"1701900{r}{c}00"},
                "geometry": {"type": "Polygon", "coordinates": [[
                    [a, b], [a + dx, b], [a + dx, b + dy], [a, b + dy], [a, b]]]},
            })
    return {"type": "FeatureCollection", "features": feats}


def _elevation_ramp(n=80):
    """Elevation rising west to east, so every zone column gets a distinct mean."""
    return np.tile(np.linspace(100.0, 900.0, n), (n, 1))


def test_zonal_output_is_what_fit_zone_model_takes(store, tmp_path):
    import rasterio
    from rasterio.transform import from_bounds

    from agent_runtime.file_store import create_output_file_from_path, resolve_file_id

    # --- a DEM over the zones, written the way dem_for_region writes one -------------------
    values = _elevation_ramp()
    tif = tmp_path / "dem.tif"
    with rasterio.open(tif, "w", driver="GTiff", height=values.shape[0], width=values.shape[1],
                       count=1, dtype="float32", crs=terrain_tools._wgs84(),
                       nodata=float("nan"),
                       transform=from_bounds(*BBOX, values.shape[1], values.shape[0])) as dst:
        dst.write(values.astype("float32"), 1)
    raster_id = create_output_file_from_path(tif, filename="dem.tif")["file_id"]

    zp = tmp_path / "zones.geojson"
    zp.write_text(json.dumps(_zone_grid()), encoding="utf-8")
    zones_id = create_output_file_from_path(zp, filename="zones.geojson")["file_id"]

    # --- summarise it per zone -------------------------------------------------------------
    zonal = {t.name: t for t in terrain_tools.make_terrain_tools()}["zonal_stats_for_raster"]
    summarised = json.loads(zonal.func(raster_file_id=raster_id, polygons_file_id=zones_id,
                                       zone_id_field="GEOID", prefix="elev"))
    assert summarised["ok"] is True, summarised
    assert summarised["zones_with_values"] == GRID * GRID

    # --- the vectors CSV embed_zones would have written -------------------------------------
    # Columns are `zone_id` plus e0..eN: that naming is the fitter's feature selector, not a
    # convention we are free to choose here. The features carry a synthetic signal so the fit
    # has something to find; this test is about the join, not about the science.
    labelled = json.loads(open(resolve_file_id(summarised["geojson"]["file_id"]),
                               encoding="utf-8").read())
    rows = ["zone_id,e0,e1,e2"]
    for feat in labelled["features"]:
        gid = feat["properties"]["GEOID"]
        elev = float(feat["properties"]["elev_mean"])
        rows.append(f"{gid},{elev / 1000.0},{(elev / 1000.0) ** 2},0.5")
    csv_path = tmp_path / "vectors.csv"
    csv_path.write_text("\n".join(rows), encoding="utf-8")
    vectors_id = create_output_file_from_path(csv_path, filename="vectors.csv")["file_id"]

    # --- fit, with no change to fit_zone_model whatsoever -----------------------------------
    fit = {t.name: t for t in rs_embed_tools.make_rs_embed_zonal_tools()}["fit_zone_model"]
    out = json.loads(fit.func(vectors_csv_file_id=vectors_id,
                              polygons_file_id=summarised["geojson"]["file_id"],
                              label_column="elev_mean", zone_id_field="GEOID", blocks=2))

    # The failure this guards is the merge finding nothing: fit_zone_model reports that as
    # "only 0 zones have both a vector and a label" rather than as a crash, so a plumbing
    # break would otherwise read as a modelling result.
    assert out.get("ok") is True, out
    assert out.get("zones_fitted", 0) >= 12, out


def test_the_label_column_is_offered_by_name_when_it_is_wrong(store, tmp_path):
    """fit_zone_model answers a bad label_column by listing the numeric columns it DOES have.
    That list is only useful if the zonal output put real numeric columns there, so this
    asserts the two halves agree about what was written."""
    import rasterio
    from rasterio.transform import from_bounds

    from agent_runtime.file_store import create_output_file_from_path

    values = _elevation_ramp()
    tif = tmp_path / "dem.tif"
    with rasterio.open(tif, "w", driver="GTiff", height=values.shape[0], width=values.shape[1],
                       count=1, dtype="float32", crs=terrain_tools._wgs84(),
                       nodata=float("nan"),
                       transform=from_bounds(*BBOX, values.shape[1], values.shape[0])) as dst:
        dst.write(values.astype("float32"), 1)
    raster_id = create_output_file_from_path(tif, filename="dem.tif")["file_id"]

    zp = tmp_path / "zones.geojson"
    zp.write_text(json.dumps(_zone_grid()), encoding="utf-8")
    zones_id = create_output_file_from_path(zp, filename="zones.geojson")["file_id"]

    zonal = {t.name: t for t in terrain_tools.make_terrain_tools()}["zonal_stats_for_raster"]
    summarised = json.loads(zonal.func(raster_file_id=raster_id, polygons_file_id=zones_id,
                                       zone_id_field="GEOID", prefix="elev"))

    csv_path = tmp_path / "vectors.csv"
    csv_path.write_text("zone_id,e0\n1701900000,0.1\n", encoding="utf-8")
    vectors_id = create_output_file_from_path(csv_path, filename="vectors.csv")["file_id"]

    fit = {t.name: t for t in rs_embed_tools.make_rs_embed_zonal_tools()}["fit_zone_model"]
    out = json.loads(fit.func(vectors_csv_file_id=vectors_id,
                              polygons_file_id=summarised["geojson"]["file_id"],
                              label_column="elevation", zone_id_field="GEOID"))

    assert out["ok"] is False
    assert "elev_mean" in out.get("numeric_columns", []), out
