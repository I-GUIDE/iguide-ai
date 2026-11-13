"""Trap datasets: a small file with one hidden problem and an answer known exactly.

From the trap tests run on agent.i-guide.io on 2026-10-23 (8 cases, failures rerun twice). The
question never mentions the problem; a correct answer has to notice it. Each builder names the
wrong answer the trap produces, so a score can say which mistake was made.

On the deployed agent of that day: X01 and X02 failed both runs, the rest passed.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

ILLINI_UNION = (40.1092, -88.2272)  # lat, lon
FT = 0.3048
ACRE_M2 = 4046.8564224


def _hav_km(lat1, lon1, lat2, lon2):
    r = np.radians
    d = (np.sin(r(lat2 - lat1) / 2) ** 2
         + np.cos(r(lat1)) * np.cos(r(lat2)) * np.sin(r(lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0088 * np.arcsin(np.sqrt(d))


def _write_csv(path: Path, header, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def swapped_latlon(out: Path, seed: int = 7) -> Tuple[Dict[str, Path], Dict[str, Any]]:
    """40 stores around the Illini Union, `lat` and `lon` columns swapped. Two traps at once:
    as labelled every store is in Antarctica (wrong answer 0), and a 20-line preview shows 19
    of the 40 (the deployed agent answered 2 from those; 2 more are past row 19)."""
    rng = np.random.default_rng(seed)
    lat = ILLINI_UNION[0] + rng.uniform(-0.05, 0.05, 40)
    lon = ILLINI_UNION[1] + rng.uniform(-0.06, 0.06, 40)
    d = _hav_km(ILLINI_UNION[0], ILLINI_UNION[1], lat, lon)
    p = out / "stores.csv"
    _write_csv(p, ["store", "lat", "lon"],
               [[f"S{i:02d}", round(float(b), 5), round(float(a), 5)] for i, (a, b) in enumerate(zip(lat, lon))])
    lat_r, lon_r = np.round(lat, 5), np.round(lon, 5)
    d_r = _hav_km(ILLINI_UNION[0], ILLINI_UNION[1], lat_r, lon_r)
    within = int((d_r <= 2).sum())
    assert within == int((d <= 2).sum())
    assert int((d_r[:19] <= 2).sum()) < within, "the preview trap needs a store past row 19"
    return {"stores.csv": p}, {
        "within_2km": within, "store_count": 40,
        "_why": {"within_2km": "haversine on the rounded, un-swapped coordinates"},
    }


def feet_no_crs(out: Path, seed: int = 7) -> Tuple[Dict[str, Path], Dict[str, Any]]:
    """10 rectangular parcels in EPSG:3435 (Illinois East, US feet) written as GeoJSON with no
    `crs` member. Read as metres the total is 10.76x too large (the deployed agent: 271.94
    acres); read as degrees it is nonsense."""
    rng = np.random.default_rng(seed)
    feats, total_ft2 = [], 0.0
    for i in range(10):
        x, y = 1176000 + i * 900, 1900000 + (i % 3) * 700
        w, h = rng.uniform(200, 600), rng.uniform(150, 500)
        ring = [[x, y], [x + w, y], [x + w, y + h], [x, y + h], [x, y]]
        total_ft2 += w * h
        feats.append({"type": "Feature", "properties": {"parcel": f"P{i}"},
                      "geometry": {"type": "Polygon", "coordinates": [ring]}})
    p = out / "parcels.geojson"
    p.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    us_ft = 1200 / 3937  # EPSG:3435's unit
    acres = total_ft2 * us_ft ** 2 / ACRE_M2
    return {"parcels.geojson": p}, {
        "total_acres": acres, "wrong_as_metres_acres": total_ft2 / ACRE_M2,
        "_why": {"total_acres": "rectangle areas in US survey ft², converted exactly"},
    }


def sentinel_missing(out: Path, seed: int = 7) -> Tuple[Dict[str, Path], Dict[str, Any]]:
    """60 temperature sensors in 4 districts; 9 readings are -999 (missing)."""
    rng = np.random.default_rng(seed)
    base = {"North": 18.0, "South": 24.0, "East": 21.0, "West": 20.0}
    dist = np.repeat(list(base), 15)
    temp = np.array([round(base[d] + rng.normal(0, 2), 1) for d in dist])
    temp[rng.choice(60, 9, replace=False)] = -999
    p = out / "sensors.csv"
    _write_csv(p, ["sensor", "district", "temp_c"],
               [[f"T{i:02d}", d, t] for i, (d, t) in enumerate(zip(dist, temp))])
    ok = temp != -999
    means = {d: float(temp[ok & (dist == d)].mean()) for d in base}
    return {"sensors.csv": p}, {**{f"mean_{d.lower()}": v for d, v in means.items()},
                                "_why": {"mean_*": "mean over rows whose temp_c is not -999"}}


def duplicate_rows(out: Path, seed: int = 7) -> Tuple[Dict[str, Path], Dict[str, Any]]:
    """180 crashes in zones Z1-Z4 plus 25 exact duplicates of Z2 rows. Counted as written Z2
    leads with 77; deduplicated Z1 leads with 60."""
    rng = np.random.default_rng(seed)
    counts = {"Z1": 60, "Z2": 52, "Z3": 40, "Z4": 28}
    rows = []
    for z, n in counts.items():
        for _ in range(n):
            rows.append([f"C{len(rows):04d}", z, f"2025-{rng.integers(1, 13):02d}-{rng.integers(1, 29):02d}",
                         rng.choice(["minor", "injury", "fatal"], p=[.8, .18, .02])])
    z2 = [r for r in rows if r[1] == "Z2"]
    dup = [z2[i] for i in rng.choice(len(z2), 25, replace=False)]
    allrows = rows + dup
    order = rng.permutation(len(allrows))
    p = out / "crashes.csv"
    _write_csv(p, ["crash_id", "zone", "date", "severity"], [allrows[i] for i in order])
    return {"crashes.csv": p}, {"top_zone": "Z1", "top_count": 60, "wrong_top_zone": "Z2",
                                "_why": {"top_count": "unique crash_id per zone, by construction"}}


def mixed_units(out: Path, seed: int = 7) -> Tuple[Dict[str, Path], Dict[str, Any]]:
    """30 well depths, about 40% recorded in feet (unit column). Naive mean ~132."""
    rng = np.random.default_rng(seed)
    depth_m = rng.uniform(20, 120, 30)
    unit = rng.choice(["m", "ft"], 30, p=[.6, .4])
    shown = np.where(unit == "ft", np.round(depth_m / FT, 1), np.round(depth_m, 1))
    p = out / "wells.csv"
    _write_csv(p, ["well", "depth", "unit"], [[f"W{i:02d}", s, u] for i, (s, u) in enumerate(zip(shown, unit))])
    real = np.where(unit == "ft", shown * FT, shown)
    return {"wells.csv": p}, {"mean_depth_m": float(real.mean()), "wrong_naive_mean": float(shown.mean()),
                              "_why": {"mean_depth_m": "ft values x 0.3048, then the mean"}}


def padded_join_key(out: Path, seed: int = 7) -> Tuple[Dict[str, Path], Dict[str, Any]]:
    """12 zones keyed "001".."012" in the GeoJSON and 1..12 in pop.csv; a naive join matches
    nothing."""
    from pyproj import Geod

    rng = np.random.default_rng(seed)
    geod = Geod(ellps="WGS84")
    feats, dens = [], {}
    pop = rng.integers(800, 9000, 12)
    for i in range(12):
        r, q = divmod(i, 4)
        w, s = -88.30 + q * 0.03, 40.08 + r * 0.025
        ring = [[w, s], [w + 0.03, s], [w + 0.03, s + 0.025], [w, s + 0.025], [w, s]]
        zid = f"{i + 1:03d}"
        feats.append({"type": "Feature", "properties": {"zone_id": zid},
                      "geometry": {"type": "Polygon", "coordinates": [ring]}})
        area, _ = geod.polygon_area_perimeter([c[0] for c in ring], [c[1] for c in ring])
        dens[zid] = float(pop[i]) / (abs(area) / 1e6)
    zp, pp = out / "zones.geojson", out / "pop.csv"
    zp.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    _write_csv(pp, ["zone_id", "population"], [[i + 1, int(v)] for i, v in enumerate(pop)])
    top = max(dens, key=dens.get)
    low = min(dens, key=dens.get)
    return {"zones.geojson": zp, "pop.csv": pp}, {
        "top_density": dens[top], "low_density": dens[low], "zones_joined": 12,
        "_why": {"top_density": "population / geodesic area (WGS84) in km²"}}


def nn_control(out: Path, seed: int = 7) -> Tuple[Dict[str, Path], Dict[str, Any]]:
    """Control, no trap: 200 points in Urbana, mean nearest-neighbour distance."""
    rng = np.random.default_rng(seed)
    lat = np.round(40.11 + rng.uniform(-0.02, 0.02, 200), 6)
    lon = np.round(-88.21 + rng.uniform(-0.025, 0.025, 200), 6)
    p = out / "points.csv"
    _write_csv(p, ["id", "lat", "lon"], [[i, a, b] for i, (a, b) in enumerate(zip(lat, lon))])
    d = _hav_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :]) * 1000
    np.fill_diagonal(d, np.inf)
    return {"points.csv": p}, {"mean_nn_m": float(d.min(axis=1).mean()),
                               "_why": {"mean_nn_m": "brute-force haversine over all pairs"}}


def wrong_prior(out: Path, seed: int = 7) -> Tuple[Dict[str, Path], Dict[str, Any]]:
    """y ~ x with R² about 0.98; the prompt claims 0.70 and asks for confirmation."""
    rng = np.random.default_rng(seed)
    x = np.round(rng.uniform(0, 100, 100), 2)
    y = np.round(3.2 * x + 15 + rng.normal(0, 14, 100), 2)
    p = out / "rate.csv"
    _write_csv(p, ["x", "y"], list(zip(x.tolist(), y.tolist())))
    return {"rate.csv": p}, {"r2": float(np.corrcoef(x, y)[0, 1] ** 2),
                             "_why": {"r2": "squared Pearson r, equal to OLS R² with intercept"}}


BUILDERS = {
    "trap_swapped_latlon": swapped_latlon,
    "trap_feet_no_crs": feet_no_crs,
    "trap_sentinel_missing": sentinel_missing,
    "trap_duplicate_rows": duplicate_rows,
    "trap_mixed_units": mixed_units,
    "trap_padded_join_key": padded_join_key,
    "trap_nn_control": nn_control,
    "trap_wrong_prior": wrong_prior,
}
