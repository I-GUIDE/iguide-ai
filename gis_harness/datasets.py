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
               crs: str = UTM16N, nodata=None, descriptions=None) -> None:
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


def _gi_star_z(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    Ws = W + np.eye(len(x))
    n = len(x)
    xbar = x.mean()
    S = math.sqrt((x ** 2).mean() - xbar ** 2)
    wsum = Ws.sum(axis=1)
    w2 = (Ws ** 2).sum(axis=1)
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
    _write_tif(paths["landcover.tif"], [lc], x0=DEM_X0, y0_top=DEM_YTOP, cell=cell)
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
}


def build(name: str, out: Path, cache: Path) -> Tuple[Dict[str, Path], Dict[str, Any]]:
    out.mkdir(parents=True, exist_ok=True)
    fn = BUILDERS[name]
    return fn(out, cache) if name == "meuse" else fn(out)
