"""GeoJSON shape construction and CRS-to-WGS84 bounds. Pure, no I/O, no heavy imports.

Split out of ``spatial.py`` because that module builds a Flask app and loads a spaCy model at
import time — 1.2 s and a warning even when all the caller wanted was to turn two coordinate
pairs into an envelope. The dataset extractor needs exactly that and nothing else, and an
extractor should not depend on the search server to describe a bounding box.

``spatial.py`` re-exports ``infer_geo_shape`` so its existing caller is unaffected.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Generous but finite: anything outside this is not a longitude/latitude, so a bbox claiming it
# is the signature of un-reprojected coordinates rather than of an unusual place.
_LON_RANGE = (-180.0, 180.0)
_LAT_RANGE = (-90.0, 90.0)


def infer_geo_shape(coords_array: List[List[float]]) -> Dict[str, Any]:
    """
    Infer a GeoJSON shape from coordinate pairs:
      - 1 pair  => point
      - 2 pairs => envelope (normalized to top-left & bottom-right)
      - ≥3 pairs => polygon (auto-closed)
    """
    if not isinstance(coords_array, list) or not coords_array:
        raise ValueError("Coordinates must be a non-empty array of [lon, lat] pairs.")

    if len(coords_array) == 1:
        return {"type": "point", "coordinates": coords_array[0]}

    if len(coords_array) == 2:
        lon1, lat1 = coords_array[0]
        lon2, lat2 = coords_array[1]
        top_left = [min(lon1, lon2), max(lat1, lat2)]
        bottom_right = [max(lon1, lon2), min(lat1, lat2)]
        return {"type": "envelope", "coordinates": [top_left, bottom_right]}

    if coords_array[0] != coords_array[-1]:
        coords_array = [*coords_array, coords_array[0]]
    return {"type": "polygon", "coordinates": [coords_array]}


def plausible_wgs84(bounds: Sequence[float]) -> bool:
    """Do these look like lon/lat degrees at all?

    This is the check whose absence made the reprojection bug invisible. The index maps
    ``spatial-bounding-box-geojson`` as ``{type: geo_shape, ignore_malformed: true}``, so
    writing UTM metres into it does not fail the write — OpenSearch drops the field and the
    document indexes cleanly. The dataset is then simply absent from every spatial query, with
    nothing anywhere recording that it happened.
    """
    try:
        minx, miny, maxx, maxy = (float(v) for v in bounds)
    except (TypeError, ValueError):
        return False
    if not (_LON_RANGE[0] <= minx <= _LON_RANGE[1] and _LON_RANGE[0] <= maxx <= _LON_RANGE[1]):
        return False
    if not (_LAT_RANGE[0] <= miny <= _LAT_RANGE[1] and _LAT_RANGE[0] <= maxy <= _LAT_RANGE[1]):
        return False
    return minx <= maxx and miny <= maxy


def to_wgs84_bounds(bounds: Sequence[float], crs: Any,
                    *, is_raster: bool = False) -> Tuple[Optional[List[float]], str]:
    """Reproject ``(minx, miny, maxx, maxy)`` to EPSG:4326.

    Returns ``(bounds_or_None, note)``. **None is a real answer**: when the CRS is unknown or
    the transform cannot be trusted, the caller must emit NO bbox and record the note, because
    an absent bbox is merely incomplete while a wrong one is silently wrong — and, thanks to
    ``ignore_malformed``, indistinguishable from absent at query time anyway.

    ``rasterio.warp.transform_bounds`` is preferred for rasters (it densifies the edges, which
    matters for a large extent in a projected CRS); ``pyproj`` covers everything else. Both are
    already dependencies, and either being unavailable yields None rather than raw bounds.
    """
    try:
        raw = [float(v) for v in bounds]
    except (TypeError, ValueError):
        return None, "bounds are not numeric"
    if len(raw) != 4:
        return None, f"expected 4 bounds, got {len(raw)}"

    crs_text = str(crs or "").strip()
    if not crs_text or crs_text.lower() in {"none", "null"}:
        # Assuming 4326 here is exactly the bug: most files that omit a CRS are not in degrees.
        if plausible_wgs84(raw):
            return raw, "no CRS declared; bounds are within lon/lat range so used as EPSG:4326"
        return None, ("no CRS declared and bounds are outside lon/lat range, so they cannot be "
                      "interpreted as degrees; no bbox emitted")

    if _is_wgs84(crs_text):
        if plausible_wgs84(raw):
            return raw, ""
        return None, (f"CRS says {crs_text} but bounds {raw} are outside lon/lat range; "
                      f"no bbox emitted")

    if is_raster:
        try:
            from rasterio.warp import transform_bounds  # type: ignore
            out = list(transform_bounds(crs_text, "EPSG:4326", *raw, densify_pts=21))
            if plausible_wgs84(out):
                return out, f"reprojected from {crs_text} via rasterio"
            return None, f"reprojection from {crs_text} produced implausible bounds {out}"
        except Exception as exc:
            logger.debug("rasterio transform_bounds failed: %s", exc)

    try:
        from pyproj import Transformer  # type: ignore
        transformer = Transformer.from_crs(crs_text, "EPSG:4326", always_xy=True)
        out = list(transformer.transform_bounds(*raw))
        if plausible_wgs84(out):
            return out, f"reprojected from {crs_text} via pyproj"
        return None, f"reprojection from {crs_text} produced implausible bounds {out}"
    except Exception as exc:
        return None, f"could not reproject from {crs_text}: {type(exc).__name__}: {exc}"


def _is_wgs84(crs_text: str) -> bool:
    lowered = crs_text.strip().lower()
    return ("4326" in lowered or lowered in {"wgs84", "wgs 84", "epsg:4326", "crs84"}
            or "urn:ogc:def:crs:ogc:1.3:crs84" in lowered)


def bbox_geo_shape(bounds: Sequence[float], crs: Any,
                   *, is_raster: bool = False) -> Tuple[Optional[Dict[str, Any]], str]:
    """``(geo_shape_envelope_or_None, note)`` for an OpenSearch ``geo_shape`` field."""
    wgs84, note = to_wgs84_bounds(bounds, crs, is_raster=is_raster)
    if wgs84 is None:
        return None, note
    minx, miny, maxx, maxy = wgs84
    return infer_geo_shape([[minx, miny], [maxx, maxy]]), note


__all__ = ["infer_geo_shape", "to_wgs84_bounds", "bbox_geo_shape", "plausible_wgs84"]
