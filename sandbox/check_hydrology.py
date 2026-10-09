"""Fail the image build if a baked hydrology library imports but cannot delineate a watershed.

verify_imports.py proves a package imports. That is not enough here: pysheds imports cleanly on
numpy 2.4 and then raises inside ``grid.accumulation`` on ``numpy.in1d``, which numpy 2.4
removed. The failure is at CALL time, in a routine every watershed needs, so the check has to
call it. The GIS harness found it the expensive way: on task T06 (slope + watershed) deepseek
spent 29 of 30 steps and about 2M tokens rewriting pysheds code around the error.

The DEM is T06's own, rebuilt from the formula in ``gis_harness/datasets.py`` ``dem_watershed``:
a V-valley with a cross ridge, 150 x 200 cells of 30 m, whose southern 120 rows drain along x
to the valley column and down it to the outlet. The basin is 150 x 120 cells = 16.2 km² by
construction, so the expected area needs no hydrology library to compute.

Two pysheds behaviours a model trips on, both measured on this DEM and both visible in T06's
transcripts, are why the check reads the area the way it does:

* The outlet is the lowest cell on the grid's edge, so its D8 direction is -2 (a pit), and
  ``catchment`` at a coordinate rounds a cell-centre coordinate to the next cell (x 15.5 cells
  becomes 16), landing on a hillslope. The check addresses the outlet by row and column.
* ``catchment`` leaves out the cells on the grid's border: 17,613 of 18,000 here (15.85 km²,
  the figure deepseek reported). ``accumulation`` counts them, 18,000 exactly. So the area is
  the accumulation at the outlet and must match exactly, and the catchment must come within
  3%, the harness's own tolerance.

Usage:
    python check_hydrology.py                        # built-in DEM; exit 1 on any failure
    python check_hydrology.py DEM.tif X Y KM2 [TOL]  # a given DEM and outlet (projected CRS)

Prints one JSON line per library. Adding a library is one function and one entry in CHECKS.
"""

import json
import os
import sys
import tempfile
import time

COLS, ROWS, RIDGE_ROWS, CELL = 150, 200, 120, 30.0
X0, YTOP = 390000.0, 4440000.0          # gis_harness DEM_X0 / DEM_YTOP: UTM 16N, Champaign
EXPECTED_CELLS = COLS * RIDGE_ROWS
CATCHMENT_TOL = 0.03


def synthetic_dem(path: str):
    """Write T06's DEM; return (outlet_x, outlet_y, expected_km2)."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    vc = COLS // 2
    x = X0 + (np.arange(COLS) + 0.5) * CELL
    y = YTOP - (np.arange(ROWS) + 0.5) * CELL
    X, Y = np.meshgrid(x, y)
    d = Y - (YTOP - ROWS * CELL)
    D = RIDGE_ROWS * CELL
    ns = np.where(d < D, 0.02 * d, 0.02 * D - 0.02 * (d - D))
    z = (200 + 0.10 * np.abs(X - x[vc]) + ns).astype("float32")
    with rasterio.open(path, "w", driver="GTiff", width=COLS, height=ROWS, count=1,
                       dtype="float32", crs="EPSG:32616",
                       transform=from_origin(X0, YTOP, CELL, CELL)) as dst:
        dst.write(z, 1)
    return float(x[vc]), float(YTOP - ROWS * CELL + CELL / 2), EXPECTED_CELLS * CELL * CELL / 1e6


def check_pysheds(dem_path: str, x: float, y: float, want: float, tol: float) -> dict:
    """The textbook pysheds sequence, plus two more of the routines numpy 2.4 broke."""
    start = time.monotonic()
    import numpy as np
    from pysheds.grid import Grid

    out = {"import_seconds": round(time.monotonic() - start, 1)}
    grid = Grid.from_raster(dem_path)
    dem = grid.read_raster(dem_path)
    conditioned = grid.resolve_flats(grid.fill_depressions(grid.fill_pits(dem)))
    fdir = grid.flowdir(conditioned)
    acc = grid.accumulation(fdir)                          # np.in1d
    col, row = grid.nearest_cell(x, y, snap="center")      # the cell that CONTAINS the outlet
    cell_km2 = abs(grid.affine.a * grid.affine.e) / 1e6
    area = float(np.asarray(acc)[row, col]) * cell_km2
    catch = grid.catchment(x=col, y=row, fdir=fdir, xytype="index")
    catch_area = int(np.asarray(catch).sum()) * cell_km2
    dist = grid.distance_to_outlet(x=col, y=row, fdir=fdir, xytype="index")   # np.in1d
    rivers = grid.extract_river_network(fdir, acc > 100)                      # np.in1d
    out.update(
        area_km2=area,
        catchment_km2=catch_area,
        river_features=len(rivers["features"]),
        max_flow_distance_cells=float(np.nanmax(np.where(np.isfinite(dist), dist, np.nan))),
        ok=(abs(area - want) <= tol * want
            and abs(catch_area - want) <= max(tol, CATCHMENT_TOL) * want
            and len(rivers["features"]) > 0),
    )
    out["numpy_in1d_shimmed"] = "pysheds_support" in (getattr(np.in1d, "__doc__", "") or "")
    cache = os.environ.get("NUMBA_CACHE_DIR", "")
    out["numba_cache_dir"] = cache
    out["numba_cache_seeded"] = bool(cache) and os.path.isdir(cache)
    return out


CHECKS = {"pysheds": check_pysheds}


def main(argv) -> int:
    if len(argv) >= 4:
        dem, x, y, want = argv[0], float(argv[1]), float(argv[2]), float(argv[3])
        tol = float(argv[4]) if len(argv) > 4 else 0.03
    else:
        dem = os.path.join(tempfile.mkdtemp(), "t06_dem.tif")
        x, y, want = synthetic_dem(dem)
        tol = 1e-9                                         # exact by construction
    failed = 0
    for name, fn in CHECKS.items():
        start = time.monotonic()
        row = {"library": name, "expected_km2": want}
        try:
            row.update(fn(dem, x, y, want, tol))
        except Exception as exc:
            row.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        row["seconds"] = round(time.monotonic() - start, 1)
        failed += not row["ok"]
        print(json.dumps(row))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
