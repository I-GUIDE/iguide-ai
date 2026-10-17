"""Pinned inputs for the GIS task harness, and the expected answers computed from them.

Every dataset here is either GENERATED from a fixed seed, or a pinned file whose bytes are
checked (the USGS catalog snapshot in `data/`, Meuse by SHA-256). Each expected value is
computed from those same bytes by an independent route: brute force, an analytic formula, or
a second library. None of it comes from the agent's own code, so a defect in the agent cannot
make its own answer key agree with it.

Each builder returns `(files, expected)`:
  files    -> {upload filename: path on disk}
  expected -> {name: value} plus `_why` notes on how each value was obtained

The synthetic data sit at real coordinates (Chicago, Champaign) because the failures worth
catching are frame failures: a distance taken in degrees, a buffer drawn in Web Mercator at
42 N (1.34x too long), an area summed over NoData. Several datasets carry a trap of that kind.
Each trap is named in the builder, with the wrong answer it would produce, so a score can say
which mistake was made and not just that one was.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

UTM16N = "EPSG:32616"
WGS84 = "EPSG:4326"
MILE_M = 1609.344

MEUSE_URL = ("https://raw.githubusercontent.com/mmaelicke/scikit-gstat/"
             "0690ef5ed1a7dc6ef30cda254719f651efed5447/skgstat/data/samples/meuse.txt")
MEUSE_SHA256 = "b27776bc1cad63c4bf308923c86a5a76a0a02566ac75984b018df2a477b52f64"
QUAKES_FILE = DATA / "usgs_comcat_ridgecrest_2019-07.csv"


def _geod():
    from pyproj import Geod
    return Geod(ellps="WGS84")


def _to_wgs(xs, ys):
    from pyproj import Transformer
    t = Transformer.from_crs(UTM16N, WGS84, always_xy=True)
    return t.transform(xs, ys)


def _write_geojson(path: Path, features: List[Dict[str, Any]]) -> None:
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))


def _point(lon: float, lat: float, props: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "Feature", "properties": props,
            "geometry": {"type": "Point", "coordinates": [round(lon, 7), round(lat, 7)]}}


def _utm_square(x0, y0, size) -> List[List[float]]:
    xs = [x0, x0 + size, x0 + size, x0, x0]
    ys = [y0, y0, y0 + size, y0 + size, y0]
    lon, lat = _to_wgs(xs, ys)
    return [[round(a, 7), round(b, 7)] for a, b in zip(lon, lat)]


def _write_tif(path: Path, bands: List[np.ndarray], *, x0: float, y0_top: float, cell: float,
               crs: str = UTM16N, nodata=None, descriptions=None, tags=None) -> None:
    import rasterio
    from rasterio.transform import from_origin

    arr = np.stack(bands)
    profile = {"driver": "GTiff", "height": arr.shape[1], "width": arr.shape[2],
               "count": arr.shape[0], "dtype": str(arr.dtype), "crs": crs,
               "transform": from_origin(x0, y0_top, cell, cell)}
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)
        for i, d in enumerate(descriptions or [], start=1):
            dst.set_band_description(i, d)
        if tags:
            dst.update_tags(**tags)


# --------------------------------------------------------------------------- T01


def area_distance(out: Path) -> Tuple[Dict[str, Path], Dict[str, Any]]:
    """Champaign County area (fetched by the agent) + a pinned great-circle distance.

    The area's reference is the Census Bureau's own AREALAND + AREAWATER for GEOID 17019
    (TIGERweb State_County layer 1, read 2026-10-08): 2,579,898,029 + 6,110,528 m^2. A polygon
    area measured on the ellipsoid from TIGER geometry lands within 0.1% of it (the 2026-10-08
    live turn answered 2,584.6 km^2). Land-only (2,579.9) is also within the 1% tolerance.
    Trap: the same polygon's area in EPSG:3857 is ~1.8x too large at 40 N.
    """
    g = _geod()
    a = (51.5080, -0.1281)   # Trafalgar Square
    b = (48.8530, 2.3499)    # Notre-Dame de Paris
    _, _, d = g.inv(a[1], a[0], b[1], b[0])
    return {}, {
        "area_km2": (2579898029 + 6110528) / 1e6,
        "distance_km": d / 1000.0,
        "_why": {"area_km2": "Census TIGERweb AREALAND+AREAWATER, GEOID 17019",
                 "distance_km": "pyproj Geod(WGS84).inv; a spherical haversine is within 0.3%"},
    }


# --------------------------------------------------------------------------- T02


SCHOOL_SITE = (41.8827, -87.6233)  # Millennium Park, Chicago


def schools(out: Path, seed: int = 2) -> Tuple[Dict[str, Path], Dict[str, Any]]:
    """Synthetic schools around a site; count within one mile by geodesic distance.

    Trap: a planar distance in EPSG:3857 at 41.9 N reads 1.343x the ground distance, so the
    schools placed at 1,250-1,500 m fall outside a "1609 m" Web Mercator buffer. Schools at
    1,590 and 1,625 m straddle the mile so rounding to "about 1.6 km" also shows.
    """
    rng = np.random.default_rng(seed)
    g = _geod()
    lat0, lon0 = SCHOOL_SITE
    dists = list(rng.uniform(250, 3200, 30)) + [1250, 1320, 1410, 1500, 1590, 1625, 1700]
    feats, inside = [], []
    for i, dist in enumerate(dists, start=1):
        az = float(rng.uniform(0, 360))
        lon, lat, _ = g.fwd(lon0, lat0, az, float(dist))
        name = f"School {i:02d}"
        _, _, true_d = g.inv(lon0, lat0, round(lon, 7), round(lat, 7))
        feats.append(_point(lon, lat, {"name": name, "school_id": i}))
        if true_d <= MILE_M:
            inside.append((name, true_d))
    path = out / "schools.geojson"
    _write_geojson(path, feats)
    nearest = min(inside, key=lambda t: t[1])
    return {"schools.geojson": path}, {
        "count_within_mile": len(inside),
        "names_within_mile": sorted(n for n, _ in inside),
        "nearest_school": nearest[0],
        "nearest_m": nearest[1],
        "_why": {"count_within_mile": "brute-force geodesic distance, pyproj Geod(WGS84)"},
    }


# --------------------------------------------------------------------------- T03


def zones_rates(out: Path, seed: int = 3) -> Tuple[Dict[str, Path], Dict[str, Any]]:
    """A 3x3 grid of 2 km zones with population, and incident points; rate per 1,000.

    Points sit at least 60 m inside their zone, so an edge drawn straight in degrees and the
    true UTM edge cannot disagree about any of them. Five points fall outside every zone.
    Trap: the zone with the most incidents is not the zone with the highest rate.
    """
    rng = np.random.default_rng(seed)
    x0, y0, size = 440000.0, 4630000.0, 2000.0
    pops = [5200, 830, 4100, 2600, 7400, 3900, 1500, 6100, 2800]
    counts = [41, 19, 22, 18, 66, 25, 9, 37, 14]
    feats, rows, rates = [], [], {}
    k = 0
    for idx in range(9):
        r, c = divmod(idx, 3)
        zx, zy = x0 + c * size, y0 + r * size
        zid = f"Z{idx + 1}"
        feats.append({"type": "Feature", "properties": {"zone_id": zid, "population": pops[idx]},
                      "geometry": {"type": "Polygon", "coordinates": [_utm_square(zx, zy, size)]}})
        px = rng.uniform(zx + 60, zx + size - 60, counts[idx])
        py = rng.uniform(zy + 60, zy + size - 60, counts[idx])
        lon, lat = _to_wgs(px, py)
        for a, b in zip(lon, lat):
            k += 1
            rows.append((k, round(float(b), 7), round(float(a), 7)))
        rates[zid] = counts[idx] / pops[idx] * 1000.0
    for _ in range(5):  # outside every zone
        lon, lat = _to_wgs([x0 - 900 - rng.uniform(0, 500)], [y0 + rng.uniform(0, 6000)])
        k += 1
        rows.append((k, round(float(lat[0]), 7), round(float(lon[0]), 7)))
    zpath, cpath = out / "zones.geojson", out / "incidents.csv"
    _write_geojson(zpath, feats)
    cpath.write_text("incident_id,lat,lon\n" + "".join(f"{i},{a},{b}\n" for i, a, b in rows))
    best = max(rates, key=rates.get)
    most = f"Z{int(np.argmax(counts)) + 1}"
    assert best != most
    return {"zones.geojson": zpath, "incidents.csv": cpath}, {
        "top_zone": best, "top_rate_per_1000": rates[best],
        "incidents_in_zones": int(sum(counts)), "most_incidents_zone": most,
        "_why": {"top_rate_per_1000": "counts placed by construction, >= 60 m from any edge"},
    }


# --------------------------------------------------------------------------- T04


ISO_MIN = 2.0


def road_network(out: Path, seed: int = 4) -> Tuple[Dict[str, Path], Dict[str, Any]]:
    """A 6x6 street grid (400 m blocks) with per-edge speeds; fastest path and a 2-min isochrone.

    Trap: the path with the shortest LENGTH is not the fastest one: a slow diagonal lane runs
    corner to corner, and a fast ring road runs round the edge. Travel time is length / speed,
    with each edge's length taken on the ellipsoid.
    """
    import networkx as nx

    rng = np.random.default_rng(seed)
    g = _geod()
    n, block, x0, y0 = 6, 400.0, 446000.0, 4636000.0
    node_xy = {(i, j): (x0 + i * block, y0 + j * block) for i in range(n) for j in range(n)}
    keys = list(node_xy)
    lons, lats = _to_wgs([node_xy[k][0] for k in keys], [node_xy[k][1] for k in keys])
    lonlat = {k: (round(float(a), 7), round(float(b), 7)) for k, a, b in zip(keys, lons, lats)}
    edges = []
    for (i, j) in node_xy:
        for di, dj in ((1, 0), (0, 1)):
            b = (i + di, j + dj)
            if b in node_xy:
                ring = (i in (0, n - 1) and di == 0) or (j in (0, n - 1) and dj == 0)
                speed = 70.0 if ring else float(rng.choice([20, 25, 30, 35, 40]))
                edges.append(((i, j), b, speed))
    for i in range(n - 1):  # the slow diagonal lane
        edges.append(((i, i), (i + 1, i + 1), 12.0))
    G = nx.Graph()
    feats = []
    for eid, (a, b, speed) in enumerate(edges, start=1):
        (lon1, lat1), (lon2, lat2) = lonlat[a], lonlat[b]
        _, _, length = g.inv(lon1, lat1, lon2, lat2)
        G.add_edge(a, b, minutes=length / (speed * 1000 / 60), length=length)
        feats.append({"type": "Feature", "properties": {"edge_id": eid, "speed_kph": speed},
                      "geometry": {"type": "LineString",
                                   "coordinates": [[lon1, lat1], [lon2, lat2]]}})
    src, dst = (0, 0), (n - 1, n - 1)
    fastest = nx.shortest_path_length(G, src, dst, weight="minutes")
    shortest_len_path = nx.shortest_path(G, src, dst, weight="length")
    fastest_path = nx.shortest_path(G, src, dst, weight="minutes")
    assert shortest_len_path != fastest_path
    reach = nx.single_source_dijkstra_path_length(G, src, weight="minutes")
    within = [k for k, t in reach.items() if t <= ISO_MIN]
    margin = min(abs(t - ISO_MIN) for t in reach.values())
    assert margin > 0.05, margin  # no node within 3 s of the cut
    path = out / "roads.geojson"
    _write_geojson(path, feats)
    return {"roads.geojson": path}, {
        "origin": lonlat[src], "destination": lonlat[dst],
        "fastest_minutes": fastest,
        "nodes_within_iso": len(within),
        "_why": {"fastest_minutes": "networkx Dijkstra on geodesic edge length / speed_kph"},
    }


# --------------------------------------------------------------------------- T05


def p_median(out: Path, seed: int = 5) -> Tuple[Dict[str, Path], Dict[str, Any]]:
    """Choose 2 of 8 candidate sites to minimise population-weighted distance to the nearest.

    Answered by enumerating all 28 pairs, with distances on the ellipsoid. The optimum beats
    the runner-up by more than 1%, so a correct method cannot land on the second pair through
    rounding.
    """
    g = _geod()
    for s in range(seed, seed + 200):
        rng = np.random.default_rng(s)
        dx = rng.uniform(0, 12000, 40) + 438000
        dy = rng.uniform(0, 12000, 40) + 4628000
        pop = rng.integers(200, 5000, 40)
        cx = rng.uniform(1000, 11000, 8) + 438000
        cy = rng.uniform(1000, 11000, 8) + 4628000
        dlon, dlat = _to_wgs(dx, dy)
        clon, clat = _to_wgs(cx, cy)
        dlon, dlat = np.round(dlon, 7), np.round(dlat, 7)
        clon, clat = np.round(clon, 7), np.round(clat, 7)
        D = np.zeros((40, 8))
        for i in range(40):
            for j in range(8):
                D[i, j] = g.inv(dlon[i], dlat[i], clon[j], clat[j])[2]
        scores = {}
        for a, b in itertools.combinations(range(8), 2):
            scores[(a, b)] = float((pop * np.minimum(D[:, a], D[:, b])).sum()) / 1000.0
        ranked = sorted(scores.items(), key=lambda kv: kv[1])
        if ranked[1][1] / ranked[0][1] > 1.01:
            break
    best, obj = ranked[0]
    dpath, cpath = out / "demand.csv", out / "candidates.csv"
    dpath.write_text("demand_id,lat,lon,population\n" + "".join(
        f"D{i + 1},{dlat[i]},{dlon[i]},{int(pop[i])}\n" for i in range(40)))
    cpath.write_text("site_id,lat,lon\n" + "".join(
        f"C{j + 1},{clat[j]},{clon[j]}\n" for j in range(8)))
    return {"demand.csv": dpath, "candidates.csv": cpath}, {
        "sites": sorted([f"C{best[0] + 1}", f"C{best[1] + 1}"]),
        "objective_person_km": obj,
        "runner_up_ratio": ranked[1][1] / ranked[0][1],
        "_why": {"sites": "exhaustive over all 28 pairs, geodesic distances"},
    }


# --------------------------------------------------------------------------- T06


DEM_X0, DEM_YTOP, DEM_CELL = 390000.0, 4440000.0, 30.0


def dem_watershed(out: Path) -> Tuple[Dict[str, Path], Dict[str, Any]]:
    """A V-valley DEM with a cross ridge: mean slope, and the area draining to the south outlet.

    z = 200 + 0.10*|x - x_valley| + 0.02*(distance below or above the ridge). Every cell
    south of the ridge drains along x to the valley, then south down the valley to the outlet
    (the cross-slope is steep enough that D8 moves along x, not diagonally, so no cell runs off
    the south edge first). The slope is atan(sqrt(0.10^2 + 0.02^2)) = 5.823 deg everywhere
    except the valley column and ridge rows. The southern basin is exactly 120 of 200 rows.
    """
    rows, cols, ridge_rows_from_south = 200, 150, 120
    vc = 75
    x = DEM_X0 + (np.arange(cols) + 0.5) * DEM_CELL
    y = DEM_YTOP - (np.arange(rows) + 0.5) * DEM_CELL
    X, Y = np.meshgrid(x, y)
    xv = x[vc]
    ybottom = DEM_YTOP - rows * DEM_CELL
    d = Y - ybottom
    D = ridge_rows_from_south * DEM_CELL
    ns = np.where(d < D, 0.02 * d, 0.02 * D - 0.02 * (d - D))
    z = (200 + 0.10 * np.abs(X - xv) + ns).astype("float32")
    path = out / "dem.tif"
    _write_tif(path, [z], x0=DEM_X0, y0_top=DEM_YTOP, cell=DEM_CELL)
    # Independent slope: Horn's method over interior cells.
    zp = z.astype("float64")
    c = DEM_CELL
    dzdx = ((zp[:-2, 2:] + 2 * zp[1:-1, 2:] + zp[2:, 2:]) -
            (zp[:-2, :-2] + 2 * zp[1:-1, :-2] + zp[2:, :-2])) / (8 * c)
    dzdy = ((zp[2:, :-2] + 2 * zp[2:, 1:-1] + zp[2:, 2:]) -
            (zp[:-2, :-2] + 2 * zp[:-2, 1:-1] + zp[:-2, 2:])) / (8 * c)
    horn = float(np.degrees(np.arctan(np.hypot(dzdx, dzdy))).mean())
    outlet_x, outlet_y = xv, ybottom + DEM_CELL / 2
    lon, lat = _to_wgs([outlet_x], [outlet_y])
    return {"dem.tif": path}, {
        "mean_slope_deg": horn,
        "analytic_slope_deg": math.degrees(math.atan(math.hypot(0.10, 0.02))),
        "watershed_km2": cols * ridge_rows_from_south * DEM_CELL ** 2 / 1e6,
        "outlet_utm": (outlet_x, outlet_y),
        "outlet_lonlat": (round(float(lon[0]), 6), round(float(lat[0]), 6)),
        "_why": {"watershed_km2": "by construction: 150 cols x 120 rows x 900 m^2",
                 "mean_slope_deg": "Horn 3x3 over interior cells; analytic 5.823 elsewhere"},
    }


# --------------------------------------------------------------------------- T07


def inundation(out: Path) -> Tuple[Dict[str, Path], Dict[str, Any]]:
    """A coastal ramp (0.02 m per m inland) with a NoData block; area at or below 5 m.

    Trap: the 10x10 NoData block (-9999) sits in the low ground. Counting it as "below 5 m"
    gives 25 ha instead of 24 ha.
    """
    rows, cols, cell = 100, 100, 10.0
    d = (rows - 1 - np.arange(rows) + 0.5) * cell  # distance from the south (sea) edge
    z = np.repeat((0.02 * d)[:, None], cols, axis=1).astype("float32")
    z[85:95, 40:50] = -9999.0
    path = out / "coast_dem.tif"
    _write_tif(path, [z], x0=DEM_X0, y0_top=DEM_YTOP, cell=cell, nodata=-9999.0)
    valid = z != -9999.0
    flooded = int(((z <= 5.0) & valid).sum())
    return {"coast_dem.tif": path}, {
        "flooded_ha": flooded * cell * cell / 1e4,
        "flooded_km2": flooded * cell * cell / 1e6,
        "trap_ha_with_nodata": int((z <= 5.0).sum()) * cell * cell / 1e4,
        "_why": {"flooded_ha": "count of valid cells with z <= 5 m, x 100 m^2"},
    }


# --------------------------------------------------------------------------- T08


def ndvi_change(out: Path, seed: int = 8) -> Tuple[Dict[str, Path], Dict[str, Any]]:
    """Two 2-band (red, NIR) uint16 scenes; mean NDVI each date and area that lost > 0.2.

    Traps: a water strip has red > NIR, so NIR - red computed in uint16 wraps to ~65,000 and
    NDVI comes out near 1 instead of negative. Pixels with value 0 are NoData in both dates.
    """
    rng = np.random.default_rng(seed)
    rows, cols, cell = 100, 100, 10.0

    def scene(veg_ndvi):
        red = rng.integers(400, 900, (rows, cols)).astype("float64")
        nir = red * (1 + veg_ndvi) / (1 - veg_ndvi)
        return red, nir

    ndvi20 = np.full((rows, cols), 0.62) + rng.normal(0, 0.03, (rows, cols))
    ndvi24 = ndvi20.copy()
    ndvi24[20:50, 30:60] -= 0.35   # cleared patch: 900 px = 9 ha
    ndvi24[60:70, 10:20] -= 0.12   # mild stress, below the threshold
    for nd in (ndvi20, ndvi24):
        nd[90:100, :] = -0.25      # water strip, red > NIR
    r20, n20 = scene(ndvi20)
    r24, n24 = scene(ndvi24)
    s20 = np.stack([r20, n20]).round().astype("uint16")
    s24 = np.stack([r24, n24]).round().astype("uint16")
    s20[:, 0:5, 0:5] = 0
    s24[:, 0:5, 0:5] = 0
    p20, p24 = out / "scene_2020.tif", out / "scene_2024.tif"
    _write_tif(p20, list(s20), x0=DEM_X0, y0_top=DEM_YTOP, cell=cell, nodata=0,
               descriptions=["red", "nir"])
    _write_tif(p24, list(s24), x0=DEM_X0, y0_top=DEM_YTOP, cell=cell, nodata=0,
               descriptions=["red", "nir"])

    def ndvi(s):
        r, n = s[0].astype("float64"), s[1].astype("float64")
        v = (n - r) / (n + r)
        v[(s[0] == 0) & (s[1] == 0)] = np.nan
        return v

    a, b = ndvi(s20), ndvi(s24)
    loss = (b - a) < -0.2
    return {"scene_2020.tif": p20, "scene_2024.tif": p24}, {
        "mean_ndvi_2020": float(np.nanmean(a)),
        "mean_ndvi_2024": float(np.nanmean(b)),
        "loss_ha": int(np.nansum(loss)) * cell * cell / 1e4,
        "_why": {"loss_ha": "float64 NDVI from the stored integers; NoData = both bands 0"},
    }


# --------------------------------------------------------------------------- T09


def lattice_autocorrelation(out: Path, seed: int = 9) -> Tuple[Dict[str, Path], Dict[str, Any]]:
    """A 10x10 lattice of 1 km cells with a planted cluster; Moran's I and Gi* hot spots.

    Moran's I: queen contiguity, row-standardised. Gi*: binary queen weights INCLUDING the cell
    itself, z-score from the analytic (normality) formula. Both computed here in numpy from
    their textbook formulas. tests/test_gis_harness.py checks them against esda.
    """
    for s in range(seed, seed + 500):
        rng = np.random.default_rng(s)
        n = 10
        v = rng.normal(50, 8, (n, n))
        v[2:5, 6:9] += 30
        v[7:9, 1:3] += 12
        W = _queen(n)
        I = _morans_i(v.ravel(), W)
        z = _gi_star_z(v.ravel(), W)
        if np.min(np.abs(z - 1.96)) > 0.1:
            break
    feats = []
    for idx in range(n * n):
        r, c = divmod(idx, n)
        feats.append({"type": "Feature",
                      "properties": {"cell_id": idx + 1, "value": round(float(v[r, c]), 4)},
                      "geometry": {"type": "Polygon", "coordinates": [
                          _utm_square(442000 + c * 1000, 4632000 + (n - 1 - r) * 1000, 1000)]}})
    path = out / "lattice.geojson"
    _write_geojson(path, feats)
    vals = np.array([f["properties"]["value"] for f in feats])
    I = _morans_i(vals, W)
    z = _gi_star_z(vals, W)
    return {"lattice.geojson": path}, {
        "morans_i": I,
        "hot_spots": int((z > 1.96).sum()),
        "_why": {"morans_i": "numpy, queen, row-standardised", "hot_spots": "Gi* z > 1.96"},
    }


def _queen(n: int) -> np.ndarray:
    W = np.zeros((n * n, n * n))
    for r in range(n):
        for c in range(n):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (dr or dc) and 0 <= r + dr < n and 0 <= c + dc < n:
                        W[r * n + c, (r + dr) * n + c + dc] = 1
    return W


def _morans_i(x: np.ndarray, W: np.ndarray) -> float:
    Wr = W / W.sum(axis=1, keepdims=True)
    zc = x - x.mean()
    return float(len(x) / Wr.sum() * (zc @ Wr @ zc) / (zc @ zc))


def _gi_star_z(x: np.ndarray, W) -> np.ndarray:
    """Getis-Ord Gi* z-scores, each unit in its own neighbourhood. W: dense or sparse, no self."""
    import scipy.sparse as sp

    n = len(x)
    Ws = sp.csr_matrix(W) + sp.identity(n, format="csr")
    xbar = x.mean()
    S = math.sqrt((x ** 2).mean() - xbar ** 2)
    wsum = np.asarray(Ws.sum(axis=1)).ravel()
    w2 = np.asarray(Ws.multiply(Ws).sum(axis=1)).ravel()
    num = Ws @ x - xbar * wsum
    den = S * np.sqrt((n * w2 - wsum ** 2) / (n - 1))
    return num / den


# --------------------------------------------------------------------------- T10


MEUSE_TARGET = (179500.0, 331500.0)
MEUSE_VGM = {"nugget": 0.05, "psill": 0.59, "range": 897.0}


def meuse(out: Path, cache: Path) -> Tuple[Dict[str, Path], Dict[str, Any]]:
    """Meuse zinc: IDW (power 2, all points) and ordinary kriging of log(zinc) at one point.

    Kriging uses the spherical model stated in the prompt (gstat's classic fit for log(zinc)),
    so the prediction is a linear solve with no fitting choice left to the agent. Fetched at
    run time from a pinned commit and checked by SHA-256; not committed.
    """
    import pandas as pd

    raw = _fetch_pinned(MEUSE_URL, MEUSE_SHA256, cache / "meuse.txt")
    df = pd.read_csv(raw)
    path = out / "meuse.csv"
    df[["x", "y", "cadmium", "copper", "lead", "zinc", "elev", "dist", "om"]].to_csv(
        path, index=False)
    xy = df[["x", "y"]].to_numpy(float)
    zn = df["zinc"].to_numpy(float)
    t = np.array(MEUSE_TARGET)
    h = np.hypot(*(xy - t).T)
    assert h.min() > 1
    w = 1 / h ** 2
    idw = float((w * zn).sum() / w.sum())
    ok_log = _ordinary_kriging(xy, np.log(zn), t, **MEUSE_VGM)
    return {"meuse.csv": path}, {
        "idw_zinc": idw, "ok_zinc": float(math.exp(ok_log)),
        "_why": {"ok_zinc": "OK system solved in numpy, Sph(0.05, 0.59, 897), exp() back"},
    }


def _sph(h, nugget, psill, range):  # noqa: A002 - gstat's own name
    h = np.asarray(h, float)
    g = np.where(h < range, nugget + psill * (1.5 * h / range - 0.5 * (h / range) ** 3),
                 nugget + psill)
    return np.where(h == 0, 0.0, g)


def _ordinary_kriging(xy, z, t, nugget, psill, range):  # noqa: A002
    n = len(z)
    H = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    A = np.ones((n + 1, n + 1))
    A[:n, :n] = _sph(H, nugget, psill, range)
    A[n, n] = 0
    b = np.ones(n + 1)
    b[:n] = _sph(np.hypot(*(xy - t).T), nugget, psill, range)
    lam = np.linalg.solve(A, b)
    return float(lam[:n] @ z)


def _fetch_pinned(url: str, sha256: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or hashlib.sha256(dest.read_bytes()).hexdigest() != sha256:
        import requests
        body = requests.get(url, timeout=60).content
        got = hashlib.sha256(body).hexdigest()
        if got != sha256:
            raise RuntimeError(f"{url}: sha256 {got} != pinned {sha256}")
        dest.write_bytes(body)
    return dest


# --------------------------------------------------------------------------- T11


LANDCOVER_CLASSES = {1: "water", 2: "wetland", 3: "grassland", 4: "cropland", 5: "forest"}


def suitability(out: Path, seed: int = 11) -> Tuple[Dict[str, Path], Dict[str, Any]]:
    """Weighted overlay of slope, road distance and land cover; suitable area and best cell.

    Suitable: slope < 8 deg, road distance <= 500 m, land cover not 1 (water) or 2 (wetland).
    Score: 0.5*(1 - slope/8) + 0.5*(1 - dist/500) over suitable cells. Trap: land cover is
    categorical, so resampling or averaging it changes which cells qualify.
    """
    rows, cols, cell = 120, 120, 10.0
    yy, xx = np.mgrid[0:rows, 0:cols]
    road_col, road_row = 30, 90
    dist = np.minimum(np.abs(xx - road_col), np.abs(yy - road_row)) * cell
    dist = dist.astype("float32")
    for s_ in range(seed, seed + 500):  # a seed whose best cell wins by a clear margin
        rng = np.random.default_rng(s_)
        slope = (5 + 4 * np.sin(xx / 17.0) * np.cos(yy / 23.0) + rng.normal(0, 0.4, (rows, cols)))
        slope = np.clip(slope, 0.3, None).astype("float32")
        lc = rng.choice([3, 4, 5], size=(rows, cols)).astype("uint8")
        lc[10:40, 50:100] = 1
        lc[60:80, 0:25] = 2
        ok = (slope < 8) & (dist <= 500) & (lc != 1) & (lc != 2)
        sc = np.sort(np.where(ok, 0.5 * (1 - slope.astype("float64") / 8)
                              + 0.5 * (1 - dist.astype("float64") / 500), -np.inf).ravel())
        if sc[-1] - sc[-2] > 2e-3:
            break
    paths = {"slope_deg.tif": out / "slope_deg.tif", "road_distance_m.tif": out / "road_distance_m.tif",
             "landcover.tif": out / "landcover.tif"}
    _write_tif(paths["slope_deg.tif"], [slope], x0=DEM_X0, y0_top=DEM_YTOP, cell=cell)
    _write_tif(paths["road_distance_m.tif"], [dist], x0=DEM_X0, y0_top=DEM_YTOP, cell=cell)
    # The legend travels with the raster (band description + one tag per class): a demo that
    # does not spell out the codes in its question still lets the agent find which are water
    # and wetland. Without it (2026-10-15) the agent had to guess, and guessed 4 and 5.
    _write_tif(paths["landcover.tif"], [lc], x0=DEM_X0, y0_top=DEM_YTOP, cell=cell,
               descriptions=["land cover class: " + ", ".join(f"{k} = {v}" for k, v in
                                                             LANDCOVER_CLASSES.items())],
               tags={f"CLASS_{k}": v for k, v in LANDCOVER_CLASSES.items()})
    s, d = slope.astype("float64"), dist.astype("float64")
    ok = (s < 8) & (d <= 500) & (lc != 1) & (lc != 2)
    score = np.where(ok, 0.5 * (1 - s / 8) + 0.5 * (1 - d / 500), -np.inf)
    flat = np.sort(score.ravel())[::-1]
    assert flat[0] - flat[1] > 1e-3
    r, c = np.unravel_index(int(np.argmax(score)), score.shape)
    bx, by = DEM_X0 + (c + 0.5) * cell, DEM_YTOP - (r + 0.5) * cell
    lon, lat = _to_wgs([bx], [by])
    return paths, {
        "suitable_ha": int(ok.sum()) * cell * cell / 1e4,
        "best_cell_utm": (bx, by),
        "best_cell_lonlat": (float(lon[0]), float(lat[0])),
        "best_score": float(flat[0]),
        "_why": {"suitable_ha": "boolean overlay on the stored float32 values"},
    }


# --------------------------------------------------------------------------- T12


def earthquakes(out: Path) -> Tuple[Dict[str, Path], Dict[str, Any]]:
    """USGS ComCat snapshot, Ridgecrest July 2019 (M>=3 within 100 km), fetched 2026-10-08.

    Aftershocks: events after the largest one, within 7 days (t0 < t <= t0 + 7 d) and 25 km
    (geodesic) of its epicentre. Public-domain US Government data.
    """
    import pandas as pd

    df = pd.read_csv(QUAKES_FILE)
    df["t"] = pd.to_datetime(df["time"], utc=True)
    main = df.loc[df["mag"].idxmax()]
    g = _geod()
    _, _, dist = g.inv(np.full(len(df), main["longitude"]), np.full(len(df), main["latitude"]),
                       df["longitude"].to_numpy(), df["latitude"].to_numpy())
    after = (df["t"] > main["t"]) & (df["t"] <= main["t"] + pd.Timedelta(days=7))
    sel = after & (dist <= 25000)
    return {QUAKES_FILE.name: QUAKES_FILE}, {
        "mainshock_mag": float(main["mag"]),
        "mainshock_id": str(main["id"]),
        "aftershocks_7d_25km": int(sel.sum()),
        "aftershocks_m4_7d_25km": int((sel & (df["mag"] >= 4)).sum()),
        "_why": {"aftershocks_7d_25km": "pandas over the pinned snapshot, geodesic distance"},
    }


# --------------------------------------------------------------------------- unsolvable


def single_band_red(out: Path, seed: int = 21) -> Tuple[Dict[str, Path], Dict[str, Any]]:
    """One band, described as red: NDVI needs a near-infrared band this file does not have."""
    rng = np.random.default_rng(seed)
    red = rng.integers(400, 900, (60, 60)).astype("uint16")
    path = out / "scene_red_only.tif"
    _write_tif(path, [red], x0=DEM_X0, y0_top=DEM_YTOP, cell=10.0, descriptions=["red"])
    return {"scene_red_only.tif": path}, {}


# --------------------------------------------------------------------------- Chicago crime (T13)
# A SNAPSHOT of three City of Chicago Data Portal datasets, pinned by SHA-256. The portal revises
# crime records daily, so a fresh download is a different input: the snapshot, not the URL, is the
# task's data. It is not in the repository (35 MB); `python -m gis_harness.datasets --snapshot
# chicago_crime` writes it under gis_harness/.cache/, and a copy lives in the git-excluded archive
# (gis_harness_runs_archive/datasets/). A download whose hash differs is refused, never used.
CHICAGO_DIR = "chicago_crime_2024"
CHICAGO_SNAPSHOT_AT = "2026-10-15T17:39:28Z"
CHICAGO_FILES = {
    "crimes_2024.csv": (
        "https://data.cityofchicago.org/resource/ijzp-q8t2.csv?$select=id,date,primary_type,"
        "description,location_description,arrest,domestic,community_area,latitude,longitude"
        "&$where=year=2024&$order=id&$limit=300000",
        "4691433aff6efea28aa631238ae470470b4aca381b7dd06904bf45b3ecd51478"),
    "community_areas.geojson": (
        "https://data.cityofchicago.org/api/geospatial/igwz-8jzy?method=export&format=GeoJSON",
        "0921a90f21ee433da800194c62044dbc6fc208f3a6828e2e358b382082ecd719"),
    "acs_2023_community_areas.csv": (
        "https://data.cityofchicago.org/resource/t68z-cikk.csv?$limit=100",
        "bfd610973f837019b1684d394378ab69fd8e8045d82fdd5ee80ff85f44e5b34f"),
}
CHICAGO_VIOLENT = ("HOMICIDE", "ASSAULT", "BATTERY", "ROBBERY", "CRIMINAL SEXUAL ASSAULT")
CHICAGO_REGIONS = 6


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pinned_snapshot(cache: Path, dirname: str, sha256s: Dict[str, str], missing: str) -> Path:
    """A pinned snapshot directory: the cache, else $GIS_HARNESS_SNAPSHOTS; every file hash-checked."""
    candidates = [cache / dirname]
    archive = os.environ.get("GIS_HARNESS_SNAPSHOTS")
    if archive:
        candidates.append(Path(archive) / dirname)
    for d in candidates:
        if all((d / name).exists() for name in sha256s):
            bad = [n for n, sha in sha256s.items() if _sha256(d / n) != sha]
            if bad:
                raise RuntimeError(f"{d}: {bad} do not match the pinned SHA-256; this is not the "
                                   "task's data")
            return d
    raise FileNotFoundError(
        f"the {dirname} snapshot is missing: copy it from the archive "
        f"(gis_harness_runs_archive/datasets/{dirname}, or set GIS_HARNESS_SNAPSHOTS). {missing}")


def _copy_snapshot(snap: Path, names, out: Path) -> Dict[str, Path]:
    files = {}
    for name in names:
        files[name] = out / name
        shutil.copyfile(snap / name, files[name])
    return files


def _chicago_snapshot(cache: Path) -> Path:
    return _pinned_snapshot(
        cache, CHICAGO_DIR, {n: sha for n, (_, sha) in CHICAGO_FILES.items()},
        "Or run `python -m gis_harness.datasets --snapshot chicago_crime` and accept that the "
        "portal's current data will not match the pinned hashes")


def chicago_expected(crimes: Path, areas: Path, acs: Path) -> Dict[str, Any]:
    """Violent crime per community area, independently of the agent: point-in-polygon on each
    incident's coordinates, ACS 2023 population, queen row-standardised Moran's I."""
    import geopandas as gpd
    import pandas as pd
    from esda.moran import Moran
    from libpysal.weights import Queen

    cr = pd.read_csv(crimes)
    v = cr[cr["primary_type"].isin(CHICAGO_VIOLENT)]
    no_xy = int(v["latitude"].isna().sum())
    v = v.dropna(subset=["latitude", "longitude"])
    pts = gpd.GeoDataFrame(v, geometry=gpd.points_from_xy(v["longitude"], v["latitude"]), crs=4326)
    ca = gpd.read_file(areas)
    ca["name"] = ca["community"].str.upper().str.strip()
    joined = gpd.sjoin(pts, ca[["name", "geometry"]], predicate="within", how="inner")
    counts = joined.groupby("name").size()
    pop = pd.read_csv(acs)
    pop["name"] = pop["community_area"].str.upper().str.strip()
    d = ca[["name", "geometry"]].merge(pop[["name", "total_population"]], on="name")
    d["violent"] = d["name"].map(counts).fillna(0).astype(int)
    d["rate"] = d["violent"] / d["total_population"] * 1000.0
    w = Queen.from_dataframe(d, use_index=True)
    w.transform = "r"
    moran = Moran(d["rate"].values, w, permutations=0)
    top = d.sort_values("rate", ascending=False).iloc[0]
    return {
        "citywide_rate": round(float(d["violent"].sum() / d["total_population"].sum() * 1000), 3),
        "top_area": str(top["name"]).title(),
        "top_rate": round(float(top["rate"]), 2),
        "morans_i": round(float(moran.I), 4),
        "regions": {"n_regions": CHICAGO_REGIONS, "n_areas": int(len(d))},
        "_why": (f"{len(cr)} incidents; {len(v) + no_xy} violent, {no_xy} without coordinates, "
                 f"{len(pts) - len(joined)} outside every community area, {len(joined)} counted; "
                 f"population {int(d['total_population'].sum())} (ACS 2023); snapshot "
                 f"{CHICAGO_SNAPSHOT_AT}"),
    }


def chicago_crime(out: Path, cache: Path) -> Tuple[Dict[str, Path], Dict[str, Any]]:
    files = _copy_snapshot(_chicago_snapshot(cache), CHICAGO_FILES, out)
    return files, chicago_expected(files["crimes_2024.csv"], files["community_areas.geojson"],
                                   files["acs_2023_community_areas.csv"])


def snapshot_chicago(cache: Path) -> Path:
    """Download the three files now. Their hashes will differ from the pinned ones once the
    portal revises a record; the task then needs new hashes and a re-check of its expected values."""
    import requests

    d = cache / CHICAGO_DIR
    d.mkdir(parents=True, exist_ok=True)
    for name, (url, sha) in CHICAGO_FILES.items():
        r = requests.get(url, headers={"User-Agent": "iguide-agent/1.0 (gis_harness snapshot)"},
                         timeout=300)
        r.raise_for_status()
        (d / name).write_bytes(r.content)
        got = _sha256(d / name)
        print(f"{name}: {got} {'(pinned)' if got == sha else '(DIFFERS from the pinned hash)'}")
    return d


# --------------------------------------------------------------------------- T14-T16: real data
# Three more snapshots pinned by SHA-256, taken 2026-10-16 for the Spatial Analysis examples
# (crash hot spots, library cooling centres, Waikīkī). Like T13's they are not in the repository;
# copies live in the git-excluded archive (gis_harness_runs_archive/datasets/<dir>). Unlike T13's
# they are DERIVED files (filtered, clipped, a grid built for the task), so there is no download
# command that reproduces their bytes: each block below says what was fetched and what was done
# to it, and a new snapshot means new hashes and re-checked expected values.

def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """The smallest value at which the cumulative weight reaches half the total."""
    o = np.argsort(values, kind="stable")
    cw = np.cumsum(np.asarray(weights, float)[o])
    return float(np.asarray(values)[o][np.searchsorted(cw, cw[-1] / 2.0)])


def _shared_vertex_contiguity(geoms) -> "Any":
    """Binary queen contiguity: two polygons are neighbours when they share a vertex as written
    (rounded to 1e-6 degrees). A second route to what libpysal's Queen computes; the test
    compares the two."""
    import scipy.sparse as sp
    from collections import defaultdict

    at = defaultdict(set)
    for i, g in enumerate(geoms):
        for poly in getattr(g, "geoms", [g]):
            for x, y in poly.exterior.coords:
                at[(round(x, 6), round(y, 6))].add(i)
    rows, cols = [], []
    for members in at.values():
        for a in members:
            for b in members:
                if a != b:
                    rows.append(a)
                    cols.append(b)
    n = len(geoms)
    W = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    W.data[:] = 1.0
    return W


# T14. Taken 2026-10-16T00:09Z.
#  chicago_serious_crashes_2020_2024.csv: City of Chicago Data Portal, Traffic Crashes - Crashes
#    (85ca-t3if), every crash with injuries_fatal > 0 or injuries_incapacitating > 0 dated
#    2020-01-01 to 2024-12-31: 10,076 rows, 68 of them without usable coordinates (67 blank, 1 at
#    0,0). Columns as the portal gives them.
#  hex_grid_500m.geojson: built for the task. Pointy-top hexagons 500 m across the flats in UTM
#    16N (EPSG:26916), kept where they intersect the city (T13's 77 community areas), numbered
#    H0001... row by row from the south-west, written in WGS84 to 6 decimals: 3,028 cells.
CRASH_DIR = "chicago_serious_crashes_2020_2024"
CRASH_FILES = {
    "chicago_serious_crashes_2020_2024.csv":
        "6b56a64aa3bad81ed7df1c91e4a5849904034f5d18778bc765c10e7fc43ed02d",
    "hex_grid_500m.geojson": "635738b77654e9eb13e3d48842eaad6ca27621318a10e08984afea19085b8ac3",
}
NIGHT_HOURS = (22, 23, 0, 1, 2, 3)


def crashes_expected(crashes: Path, hexes: Path) -> Dict[str, Any]:
    """Counts by point-in-polygon; Gi* with binary queen weights from shared vertices, each
    hexagon in its own neighbourhood, analytical z; the hot-spot streets from each crash's own
    street_name. The share at night and the yearly averages are over every row of the file."""
    import geopandas as gpd
    import pandas as pd

    c = pd.read_csv(crashes)
    grid = gpd.read_file(hexes)
    ok = (c["latitude"].notna() & c["longitude"].notna()
          & (c["latitude"] != 0) & (c["longitude"] != 0))
    pts = gpd.GeoDataFrame(c[ok], crs=4326, geometry=gpd.points_from_xy(
        c.loc[ok, "longitude"], c.loc[ok, "latitude"]))
    j = gpd.sjoin(pts, grid[["hex_id", "geometry"]], predicate="within", how="inner")
    counts = grid["hex_id"].map(j.groupby("hex_id").size()).fillna(0).to_numpy(float)
    z = _gi_star_z(counts, _shared_vertex_contiguity(list(grid.geometry)))
    top_n, top_z = int(np.argmax(counts)), int(np.argmax(z))
    hot = set(grid["hex_id"][z > 1.96])
    streets = j[j["hex_id"].isin(hot)]["street_name"].value_counts()
    # The answer is unambiguous: one busiest hexagon, one highest z, and a clear third street.
    assert (counts == counts[top_n]).sum() == 1 and (z == z[top_z]).sum() == 1
    assert streets.iloc[2] > streets.iloc[3]
    year = c["crash_date"].str[:4].astype(int)
    return {
        "crashes_in_grid": int(len(j)),
        "top_count_hex": str(grid["hex_id"][top_n]),
        "top_count": int(counts[top_n]),
        "hot_cells": int(len(hot)),
        "top_z_hex": str(grid["hex_id"][top_z]),
        "top_z": round(float(z[top_z]), 3),
        # "MADISON ST" is matched as MADISON: the agent may write "W. Madison Street".
        "hot_streets": [s.rsplit(" ", 1)[0] for s in streets.index[:3]],
        "street_1_crashes": int(streets.iloc[0]),
        "street_2_crashes": int(streets.iloc[1]),
        "street_3_crashes": int(streets.iloc[2]),
        "night_pct": round(float(c["crash_hour"].isin(NIGHT_HOURS).mean() * 100), 2),
        "per_year_2020_2022": round(float((year <= 2022).sum() / 3), 1),
        "per_year_2023_2024": round(float((year >= 2023).sum() / 2), 1),
        "_why": (f"{len(c)} rows, {int(ok.sum())} with coordinates, {len(j)} in a hexagon; "
                 f"{len(grid)} hexagons; Gi* over shared-vertex queen contiguity; streets "
                 f"{ {k: int(v) for k, v in streets.head(4).items()} }"),
    }


def serious_crashes(out: Path, cache: Path) -> Tuple[Dict[str, Path], Dict[str, Any]]:
    snap = _pinned_snapshot(cache, CRASH_DIR, CRASH_FILES, "The files are derived; see CRASH_FILES.")
    files = _copy_snapshot(snap, CRASH_FILES, out)
    return files, crashes_expected(files["chicago_serious_crashes_2020_2024.csv"],
                                   files["hex_grid_500m.geojson"])


# T15. Taken 2026-10-16T00:09Z.
#  cpl_branches.csv: City of Chicago Data Portal, Chicago Public Library locations (x8fc-8rcq):
#    branch, address, zip, and the latitude/longitude parsed from its location field; 82 branches.
#  chicago_blockgroup_population_centres_2020.csv: Census Bureau 2020 centres of population by
#    block group, Illinois (CenPop2020_Mean_BG17.txt), Cook County block groups whose centre
#    lies inside the city (T13's 77 community areas) with population > 0: 2,158 rows.
COOLING_DIR = "chicago_cooling_centres_2020"
COOLING_FILES = {
    "cpl_branches.csv": "f748488b60cbc3efdcda700c3d360c75b911a1caaf7e28caf1c1334939aaa2d7",
    "chicago_blockgroup_population_centres_2020.csv":
        "eb41315592e9546334bee26b8ece1d53cb6f93d87526cd4b18a2e564045203a1",
}
COOLING_OPEN = 6


def _p_median_exact(cost: np.ndarray, p: int) -> np.ndarray:
    """Exact p-median by MILP (scipy's HiGHS): open y_j, assign x_ij <= y_j, sum_j x_ij = 1.
    *cost* is demand x site, already weighted. Returns the indices of the open sites."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    n, m = cost.shape
    nx_ = n * m
    ii, jj = np.repeat(np.arange(n), m), np.tile(np.arange(m), n)
    link = n + np.arange(nx_)
    rows = np.concatenate([ii, link, link, np.full(m, n + nx_)])
    cols = np.concatenate([m + ii * m + jj, m + np.arange(nx_), jj, np.arange(m)])
    vals = np.concatenate([np.ones(nx_), np.ones(nx_), -np.ones(nx_), np.ones(m)])
    A = coo_matrix((vals, (rows, cols)), shape=(n + nx_ + 1, m + nx_)).tocsr()
    lb = np.concatenate([np.ones(n), np.full(nx_, -np.inf), [p]])
    ub = np.concatenate([np.ones(n), np.zeros(nx_), [p]])
    res = milp(np.concatenate([np.zeros(m), cost.ravel()]),
               constraints=LinearConstraint(A, lb, ub),
               integrality=np.concatenate([np.ones(m), np.zeros(nx_)]),
               bounds=Bounds(0, 1), options={"time_limit": 900})
    if res.status != 0:
        raise RuntimeError(f"p-median MILP did not solve to optimality: {res.message}")
    return np.sort(np.where(res.x[:m] > 0.5)[0])


def cooling_expected(branches: Path, blockgroups: Path) -> Dict[str, Any]:
    """Exact 6-median over the branches, straight-line metres in UTM 16N, population weights;
    then each block group is served by its nearest open branch. The best single swap away from
    the optimum is reported, so the tolerance on the mean distance can be set below it."""
    import geopandas as gpd
    import pandas as pd

    lib = pd.read_csv(branches)
    bg = pd.read_csv(blockgroups)
    L = gpd.GeoSeries(gpd.points_from_xy(lib.longitude, lib.latitude), crs=4326).to_crs(26916)
    B = gpd.GeoSeries(gpd.points_from_xy(bg.longitude, bg.latitude), crs=4326).to_crs(26916)
    km = np.hypot(B.x.to_numpy()[:, None] - L.x.to_numpy()[None, :],
                  B.y.to_numpy()[:, None] - L.y.to_numpy()[None, :]) / 1000.0
    pop = bg["population"].to_numpy(float)
    open_ = _p_median_exact(pop[:, None] * km, COOLING_OPEN)
    dmin = km[:, open_].min(axis=1)
    serving = open_[km[:, open_].argmin(axis=1)]
    obj = float((pop * dmin).sum())
    swap = min(float((pop * km[:, [s for s in open_ if s != a] + [k]].min(axis=1)).sum())
               for a in open_ for k in range(len(lib)) if k not in open_)
    names = {j: str(lib["branch"][j]) for j in open_}
    by_name = sorted(open_, key=lambda j: names[j])
    out = {
        # Matched by the name's first words: "Whitney M. Young, Jr." is also "Whitney Young".
        "branches": [names[j].split(",")[0].split(" M.")[0] for j in by_name],
        "mean_km": round(obj / pop.sum(), 4),
        "share_over_2km_pct": round(float(pop[dmin > 2].sum() / pop.sum() * 100), 2),
        "objective_person_km": round(obj, 1),
        "best_swap_ratio": round(swap / obj, 5),
        "_why": (f"{len(lib)} branches, {len(bg)} block groups, {int(pop.sum())} people; exact "
                 f"MILP; served {[(names[j], int(pop[serving == j].sum())) for j in by_name]}"),
    }
    for i, j in enumerate(by_name, 1):
        out[f"served_{i}"] = int(pop[serving == j].sum())
    return out


def cooling_centres(out: Path, cache: Path) -> Tuple[Dict[str, Path], Dict[str, Any]]:
    snap = _pinned_snapshot(cache, COOLING_DIR, COOLING_FILES,
                            "The files are derived; see COOLING_FILES.")
    files = _copy_snapshot(snap, COOLING_FILES, out)
    return files, cooling_expected(files["cpl_branches.csv"],
                                   files["chicago_blockgroup_population_centres_2020.csv"])


# T16. Taken 2026-10-16T02:24Z, clipped to Waikīkī, the Ala Wai and McCully: the box
# -157.850, 21.255 to -157.805, 21.300.
#  tsunami_evacuation_zone.geojson: Hawaii Statewide GIS Program, Hazards/MapServer/2 (Tsunami
#    Evacuation Zones; City and County of Honolulu), the Oʻahu zone, made valid and clipped.
#  census_blocks_2020.geojson: Census TIGERweb tigerWMS_Census2020 layer 10, Honolulu County
#    blocks intersecting the box: GEOID, POP100, HU100 and the internal point (561 blocks).
#  osm_streets.geojson, osm_hotels.geojson: OpenStreetMap via Overpass, data as of
#    2026-10-16T02:24:20Z (ODbL; attribution required). Streets: highway ways of the walkable
#    classes in the box, less foot=no (5,648 ways). Hotels: tourism=hotel nodes, and the centres
#    of ways and relations, inside the box (88).
#  slr_exposure_3_2ft.geojson: Hawaii Statewide GIS Program, Climate/MapServer/45 (SLR Exposure
#    Area, 3.2 ft scenario; SOEST, UH CGG, PacIOOS, Tetra Tech), the Oʻahu feature, clipped and
#    dissolved into one geometry.
#  buildings.geojson: City and County of Honolulu DPP, HoLIS building footprints (ArcGIS Online
#    "Building Footprints CCH"), footprints intersecting the box, numbered B00001... (9,843).
WAIKIKI_DIR = "waikiki_tsunami_2026"
WAIKIKI_FILES = {
    "tsunami_evacuation_zone.geojson":
        "f0c7d05c4245f712257c64a1648bc9c68a444b23ff238037c089bfb2cd3a6c91",
    "census_blocks_2020.geojson": "7e9f5b59fe724d7eb3de03eeab4692d80ef4da8f4de908d819a6275e33a9a6b4",
    "osm_hotels.geojson": "43d7d48e370f548b39311e28892c8a586663d7233f0685c38006956dea84285f",
    "osm_streets.geojson": "80e4466965e39fa144625e9084a0bc751acd94b35a57dfded0f7b59c1064ca8b",
    "slr_exposure_3_2ft.geojson": "a65e086b5893fc9c50602c4591457499048de47d06e44ccbd2a370df94a1d748",
    "buildings.geojson": "f4806a2b7976dae8b9d575bae69458f059891957ea1fe5595daf8e3a093dde47",
}
WAIKIKI_UTM = 32604
WALK_M_PER_S = 1.1


def _walk_minutes(lines, start_xy: np.ndarray, inside, epsg: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Shortest walk, in minutes, from each start to the nearest street node outside *inside*.

    The network is undirected; ways join where they share a vertex as written (to 1e-7 degrees).
    Each start snaps to its nearest node of the largest connected part. Lengths in *epsg*."""
    import geopandas as gpd
    import pandas as pd
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components, dijkstra
    from scipy.spatial import cKDTree
    from shapely import contains_xy

    key: Dict[Tuple[float, float], int] = {}
    lonlat: List[Tuple[float, float]] = []
    us, vs = [], []
    for g in lines:
        ids = []
        for x, y in g.coords:
            k = (round(x, 7), round(y, 7))
            if k not in key:
                key[k] = len(lonlat)
                lonlat.append(k)
            ids.append(key[k])
        for u, v in zip(ids[:-1], ids[1:]):
            if u != v:
                us.append(u)
                vs.append(v)
    ll = np.array(lonlat)
    p = gpd.GeoSeries(gpd.points_from_xy(ll[:, 0], ll[:, 1]), crs=4326).to_crs(epsg)
    xy = np.c_[p.x.to_numpy(), p.y.to_numpy()]
    us, vs = np.array(us), np.array(vs)
    e = pd.DataFrame({"u": np.r_[us, vs], "v": np.r_[vs, us],
                      "w": np.tile(np.hypot(*(xy[us] - xy[vs]).T), 2)})
    e = e.groupby(["u", "v"], as_index=False)["w"].min()
    G = sp.csr_matrix((e["w"], (e["u"], e["v"])), shape=(len(xy), len(xy)))
    ncomp, label = connected_components(G, directed=False)
    big = np.where(label == np.bincount(label).argmax())[0]
    exits = big[~contains_xy(inside, xy[big, 0], xy[big, 1])]
    metres = dijkstra(G, directed=False, indices=exits, min_only=True)
    _, k = cKDTree(xy[big]).query(start_xy)
    return metres[big[k]] / WALK_M_PER_S / 60.0, {"nodes": len(xy), "components": int(ncomp),
                                                  "largest": int(len(big))}


def waikiki_expected(d: Dict[str, Path]) -> Dict[str, Any]:
    """Blocks are in the zone when their internal point is. The walk is `_walk_minutes`; the
    median is population-weighted. Sea-level rise: buildings that intersect the exposure area,
    and those whose centroid is inside it, in UTM 4N; the WGS84 intersect count is in _why."""
    import geopandas as gpd
    import shapely

    U = WAIKIKI_UTM
    zone = gpd.read_file(d["tsunami_evacuation_zone.geojson"]).to_crs(U).union_all()
    blocks = gpd.read_file(d["census_blocks_2020.geojson"])
    ip = gpd.GeoSeries(gpd.points_from_xy(blocks.INTPTLON, blocks.INTPTLAT), crs=4326).to_crs(U)
    inz = ip.within(zone).to_numpy()
    hotels = gpd.read_file(d["osm_hotels.geojson"]).to_crs(U)
    streets = gpd.read_file(d["osm_streets.geojson"])
    walk = inz & (blocks["POP100"].to_numpy() > 0)
    mins, net = _walk_minutes(streets.geometry, np.c_[ip.x.to_numpy()[walk], ip.y.to_numpy()[walk]],
                              zone, U)
    pop = blocks["POP100"].to_numpy()[walk]
    far = int(np.argmax(mins))
    near = sorted(round(float(m), 2) for m in mins if 9.5 < m < 10.5)
    slr = gpd.read_file(d["slr_exposure_3_2ft.geojson"])
    bld = gpd.read_file(d["buildings.geojson"])
    S, Sw = slr.to_crs(U).union_all(), slr.union_all()
    shapely.prepare(S)
    shapely.prepare(Sw)
    bu = bld.to_crs(U)
    cen = bu.geometry.centroid
    return {
        "zone_pop": int(blocks["POP100"][inz].sum()),
        "zone_hu": int(blocks["HU100"][inz].sum()),
        "hotels_in_zone": int(hotels.within(zone).sum()),
        "walk_median_min": round(_weighted_median(mins, pop), 2),
        "walk_max_min": round(float(mins[far]), 2),
        "walk_max_block": str(blocks["GEOID"].to_numpy()[walk][far]),
        "pop_over_10min": int(pop[mins > 10].sum()),
        "slr_buildings_intersect": int(shapely.intersects(S, bu.geometry.values).sum()),
        "slr_buildings_centroid": int(shapely.contains_xy(S, cen.x.to_numpy(), cen.y.to_numpy()).sum()),
        "hotels_in_slr": int(shapely.contains_xy(S, hotels.geometry.x.to_numpy(),
                                                 hotels.geometry.y.to_numpy()).sum()),
        "_why": (f"{int(inz.sum())} of {len(blocks)} blocks in the zone; {int(walk.sum())} populated "
                 f"blocks walked; network {net}; walks within 0.5 min of 10: {near}; "
                 f"{int(shapely.intersects(Sw, bld.geometry.values).sum())} buildings intersect in "
                 f"WGS84; {len(hotels)} hotels"),
    }


def waikiki_tsunami(out: Path, cache: Path) -> Tuple[Dict[str, Path], Dict[str, Any]]:
    snap = _pinned_snapshot(cache, WAIKIKI_DIR, WAIKIKI_FILES,
                            "The files are derived; see WAIKIKI_FILES.")
    files = _copy_snapshot(snap, WAIKIKI_FILES, out)
    return files, waikiki_expected(files)


SNAPSHOT_BUILDERS = ("meuse", "chicago_crime", "serious_crashes", "cooling_centres",
                     "waikiki_tsunami")

BUILDERS = {
    "area_distance": area_distance,
    "schools": schools,
    "zones_rates": zones_rates,
    "road_network": road_network,
    "p_median": p_median,
    "dem_watershed": dem_watershed,
    "inundation": inundation,
    "ndvi_change": ndvi_change,
    "lattice_autocorrelation": lattice_autocorrelation,
    "meuse": meuse,
    "suitability": suitability,
    "earthquakes": earthquakes,
    "single_band_red": single_band_red,
    "chicago_crime": chicago_crime,
    "serious_crashes": serious_crashes,
    "cooling_centres": cooling_centres,
    "waikiki_tsunami": waikiki_tsunami,
}


def build(name: str, out: Path, cache: Path) -> Tuple[Dict[str, Path], Dict[str, Any]]:
    out.mkdir(parents=True, exist_ok=True)
    fn = BUILDERS[name]
    return fn(out, cache) if name in SNAPSHOT_BUILDERS else fn(out)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build or snapshot a harness dataset.")
    ap.add_argument("--snapshot", choices=["chicago_crime"], required=True)
    a = ap.parse_args()
    print(snapshot_chicago(Path(__file__).resolve().parent / ".cache"))

