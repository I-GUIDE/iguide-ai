"""Checks on coordinates that a file's own metadata cannot be trusted to make.

Two failures from the trap tests on agent.i-guide.io (2026-10-23), both repeated on rerun:

* **A GeoJSON in EPSG:3435 (Illinois East, US feet) with no ``crs`` member.** GDAL reads a
  GeoJSON with no CRS as EPSG:4326, so ``inspect_vector`` reported "EPSG:4326" next to bounds of
  1,176,000 / 1,900,000. The model saw that the numbers could not be degrees, decided they were
  metres, and reported 271.94 acres for parcels of 25.26 (the ft² to m² factor, 10.76x). The unit
  checks in ``sandbox_verify`` never ran: they need a CRS on the frame, and the area was computed
  from raw coordinates. :func:`crs_check` says the coordinates are not degrees, that the CRS is
  unknown, and which projected CRSs place the data inside their own area of use, with the unit
  of each, so feet and metres are both on the table.

* **lat and lon columns swapped.** As labelled, the points sat at latitude -88 (Antarctica);
  read the other way they were in Champaign. The model caught it once in two runs.
  :func:`latlon_swap_hint` makes that a check instead of a chance.

Neither function decides anything. Each returns a note for the model, which states the
assumption it takes or asks the user. Both are PROJ-light and never raise.
"""
from __future__ import annotations

import math
import re
import time
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence

# Same test as terrain_tools._degree_like: whether bounds could be lon/lat at all.
def degree_like(bounds: Optional[Sequence[float]]) -> bool:
    try:
        left, bottom, right, top = (float(v) for v in bounds)  # type: ignore[union-attr]
    except (TypeError, ValueError):
        return False
    return -180.5 <= left <= 180.5 and -180.5 <= right <= 180.5 and \
        -90.5 <= bottom <= 90.5 and -90.5 <= top <= 90.5


def _is_geographic(crs: Any) -> Optional[bool]:
    """True/False for a CRS we can read, None for no CRS at all."""
    if crs in (None, ""):
        return None
    try:
        from pyproj import CRS

        return bool(CRS.from_user_input(crs).is_geographic)
    except Exception:  # noqa: BLE001 - an unreadable CRS is treated as no CRS
        return None


# The families an upload is realistically in. All 5,291 projected EPSG CRSs take ~40 s to test
# (a Transformer costs ~8 ms to build); these 1,184 are US State Plane (NAD83, NAD83(2011)),
# UTM on WGS 84 / ETRS89, and a few national grids. Feet first: a feet CRS read as metres is
# the error this exists to stop, and the State Plane feet zones are where I-GUIDE's data live.
_FAMILIES = re.compile(
    r"^NAD83 / |^NAD83\(2011\) / |^WGS 84 / UTM zone|^ETRS89 / UTM zone|"
    r"^OSGB36 / British National Grid|^GDA2020 / MGA|^NZGD2000 / New Zealand Transverse|"
    r"^NAD83 / Conus Albers")


@lru_cache(maxsize=1)
def _projected_crs_index() -> List[Dict[str, Any]]:
    """Projected EPSG CRSs of the common families with an area of use: feet units first, then
    smallest area first."""
    try:
        from pyproj.database import query_crs_info
        from pyproj.enums import PJType

        infos = query_crs_info(auth_name="EPSG", pj_types=PJType.PROJECTED_CRS)
    except Exception:  # noqa: BLE001
        return []
    out = []
    for info in infos:
        aou = info.area_of_use
        if aou is None or info.deprecated or not _FAMILIES.search(info.name or ""):
            continue
        w, s, e, n = aou.west, aou.south, aou.east, aou.north
        span = (e - w if e >= w else e + 360 - w) * (n - s)
        feet = "(ftUS)" in info.name or "(ft)" in info.name
        out.append({"code": info.code, "name": info.name, "aou": (w, s, e, n), "span": span,
                    "feet": feet})
    out.sort(key=lambda r: (not r["feet"], r["span"]))
    return out


@lru_cache(maxsize=2048)
def _to_lonlat(code: str):
    from pyproj import Transformer

    return Transformer.from_crs(f"EPSG:{code}", "EPSG:4326", always_xy=True)


@lru_cache(maxsize=2048)
def _unit_name(code: str) -> Optional[str]:
    from pyproj import CRS

    axis = CRS.from_epsg(int(code)).axis_info
    return axis[0].unit_name if axis else None


def _inside(lon: float, lat: float, aou) -> bool:
    w, s, e, n = aou
    if not (s - 0.05 <= lat <= n + 0.05):
        return False
    if w <= e:
        return w - 0.05 <= lon <= e + 0.05
    return lon >= w - 0.05 or lon <= e + 0.05  # area crossing the antimeridian


def candidate_crs(bounds: Sequence[float], *, per_unit: int = 8,
                  budget_s: float = 12.0) -> List[Dict[str, Any]]:
    """Projected CRSs under which the centre of *bounds* lands inside that CRS's own area of
    use, up to *per_unit* zones for feet and for metres (NAD83 and NAD83(2011) of one zone count
    once). A guide, not an answer: State Plane zones share false origins, so T2's bounds fit 7
    feet zones from Arkansas South to Illinois East, and the location has to come from the user
    or the question. The first call builds transformers (~8 ms each), cached for the process;
    T2's search took 4 s cold."""
    try:
        left, bottom, right, top = (float(v) for v in bounds)
    except (TypeError, ValueError):
        return []
    cx, cy = (left + right) / 2.0, (bottom + top) / 2.0
    found: List[Dict[str, Any]] = []
    taken = {True: 0, False: 0}
    zones = set()
    deadline = time.monotonic() + budget_s
    for rec in _projected_crs_index():
        if time.monotonic() > deadline or all(n >= per_unit for n in taken.values()):
            break
        zone = rec["name"].split(" / ", 1)[-1]
        if taken[rec["feet"]] >= per_unit or zone in zones:
            continue
        try:
            lon, lat = _to_lonlat(rec["code"]).transform(cx, cy)
        except Exception:  # noqa: BLE001
            continue
        if not (math.isfinite(lon) and math.isfinite(lat)) or not _inside(lon, lat, rec["aou"]):
            continue
        try:
            unit = _unit_name(rec["code"])
        except Exception:  # noqa: BLE001
            unit = None
        taken[rec["feet"]] += 1
        zones.add(zone)
        found.append({"crs": f"EPSG:{rec['code']}", "name": rec["name"], "unit": unit,
                      "centre_lonlat": [round(lon, 5), round(lat, 5)]})
    return found


def crs_check(bounds: Optional[Sequence[float]], declared_crs: Any, *,
              with_candidates: bool = True) -> Optional[Dict[str, Any]]:
    """A warning when *declared_crs* is missing or geographic but *bounds* cannot be degrees.

    Returns None when nothing is wrong (or nothing can be said)."""
    if bounds is None or degree_like(bounds):
        return None
    geographic = _is_geographic(declared_crs)
    if geographic is False:
        return None  # a projected CRS is declared and the numbers are not degrees: consistent
    declared = "none" if geographic is None else str(declared_crs)
    out: Dict[str, Any] = {
        "crs_warning": (
            f"The coordinates (bounds {', '.join(f'{float(b):,.1f}' for b in bounds)}) cannot be "
            f"longitude/latitude, so the declared CRS ({declared}) is wrong or missing and the "
            "real CRS, and its unit, are unknown. Do not assume metres: a State Plane CRS in US "
            "feet gives areas 10.76x smaller than the same numbers read as metres. Ask the user "
            "which CRS the file is in, or state the assumption you take and give the result "
            "under each plausible unit. Set the CRS on the frame (set_crs, not to_crs) before "
            "measuring, so the unit checks can run."),
        "crs_declared": declared,
        "coordinates_degree_like": False,
    }
    if with_candidates:
        cands = candidate_crs(bounds)
        if cands:
            out["crs_candidates"] = cands
            out["crs_candidates_note"] = (
                "Every CRS listed places these numbers inside its own area of use; they differ in "
                "location and unit, so the numbers alone cannot pick one. Use the place the user "
                "or the question names, or ask.")
            units = sorted({c["unit"] for c in cands if c.get("unit")})
            if len(units) > 1:
                out["crs_candidate_units"] = units
    return out


# --------------------------------------------------------------------------- lat/lon columns

_LAT_NAMES = re.compile(r"^(lat|latitude|y|lat_dd|latitude_dd|point_y)$", re.I)
_LON_NAMES = re.compile(r"^(lon|lng|long|longitude|x|lon_dd|longitude_dd|point_x)$", re.I)


def _float(v: Any) -> Optional[float]:
    try:
        f = float(str(v).strip())
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# Share of "lat" values that must lie beyond ±90 before the columns count as swapped. One bad
# latitude in a correctly labelled file is a bad row, not a swap; calling it a swap would move
# every other point (Codex review of PR #181).
SWAP_SHARE = 0.9


class LatLonScan:
    """Running summary of a lat/lon column pair, fed one row at a time, so a large CSV is
    checked without holding its rows (Codex review of PR #181: the first version kept every
    parsed row of the upload only to count them)."""

    def __init__(self, header: Sequence[str]) -> None:
        self.lat_i = next((i for i, h in enumerate(header) if _LAT_NAMES.match(str(h).strip())), None)
        self.lon_i = next((i for i, h in enumerate(header) if _LON_NAMES.match(str(h).strip())), None)
        self.lat_name = header[self.lat_i] if self.lat_i is not None else None
        self.lon_name = header[self.lon_i] if self.lon_i is not None else None
        self.n_lat = self.n_lon = 0
        self.lat_beyond_90 = 0      # 90 < |lat| <= 180: could be a longitude
        self.lat_beyond_180 = 0     # cannot be a coordinate either way
        self.lon_beyond_90 = 0
        self.lat_min = self.lon_min = math.inf
        self.lat_max = self.lon_max = -math.inf

    def feed(self, row: Sequence[Any]) -> None:
        if self.lat_i is None or self.lon_i is None:
            return
        lat = _float(row[self.lat_i]) if len(row) > self.lat_i else None
        lon = _float(row[self.lon_i]) if len(row) > self.lon_i else None
        if lat is not None:
            self.n_lat += 1
            self.lat_min, self.lat_max = min(self.lat_min, lat), max(self.lat_max, lat)
            if abs(lat) > 180:
                self.lat_beyond_180 += 1
            elif abs(lat) > 90:
                self.lat_beyond_90 += 1
        if lon is not None:
            self.n_lon += 1
            self.lon_min, self.lon_max = min(self.lon_min, lon), max(self.lon_max, lon)
            if abs(lon) > 90:
                self.lon_beyond_90 += 1

    def hint(self) -> Optional[Dict[str, Any]]:
        """Three outcomes, besides nothing:
          * at least SWAP_SHARE of "lat" values beyond ±90 (and none beyond ±180), every "lon"
            a valid latitude: certainly swapped;
          * some "lat" values beyond ±90 but not most: those rows are invalid, not a swap;
          * every "lat" south of -60 (Antarctica) and every "lon" a valid latitude north of it:
            probably swapped, said as a likelihood."""
        if not self.n_lat or not self.n_lon:
            return None
        lat_name, lon_name = self.lat_name, self.lon_name
        bad = self.lat_beyond_90 + self.lat_beyond_180
        if (self.lat_beyond_180 == 0 and self.lon_beyond_90 == 0
                and self.lat_beyond_90 >= SWAP_SHARE * self.n_lat):
            return {"latlon_warning": (
                f"Column '{lat_name}' has values beyond ±90 in {self.lat_beyond_90} of "
                f"{self.n_lat} rows, which cannot be latitudes, and '{lon_name}' holds values "
                f"that fit as latitudes. The columns are swapped: read '{lon_name}' as latitude "
                f"and '{lat_name}' as longitude, and say so in the answer."),
                "latlon_swapped": "certain"}
        if bad:
            return {"latlon_warning": (
                f"{bad} of {self.n_lat} rows have a '{lat_name}' value beyond ±90, which is not "
                "a valid latitude. The other rows read correctly as labelled, so this is bad "
                "data in those rows, not swapped columns: leave the columns as they are, and "
                "exclude or report the invalid rows."),
                "latlon_invalid_rows": bad}
        if self.lat_max < -60 and self.lon_min >= -60 and self.lon_beyond_90 == 0:
            return {"latlon_warning": (
                f"As labelled, every point is south of 60°S (Antarctica): '{lat_name}' ranges "
                f"{self.lat_min:.4f} to {self.lat_max:.4f}. Read the other way round "
                f"('{lon_name}' as latitude) the points are at {self.lon_min:.4f} to "
                f"{self.lon_max:.4f}°N/S. The columns are probably swapped. Use the reading that "
                "places the data where the question does, and state which one you used."),
                "latlon_swapped": "likely"}
        return None


def latlon_swap_hint(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> Optional[Dict[str, Any]]:
    """:class:`LatLonScan` over *rows* (header excluded), in one pass."""
    scan = LatLonScan(header)
    if scan.lat_i is None or scan.lon_i is None:
        return None
    for row in rows:
        scan.feed(row)
    return scan.hint()


__all__ = ["LatLonScan", "candidate_crs", "crs_check", "degree_like", "latlon_swap_hint"]
