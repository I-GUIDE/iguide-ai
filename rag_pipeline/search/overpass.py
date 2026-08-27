"""Live OpenStreetMap retrieval via the Overpass API.

Unlike the knowledge-base backends, this returns real GEOMETRY (points / lines /
polygons) for ground-truth OSM features -- rivers, roads, hospitals, parks, dams,
power plants, ... -- inside a place or bounding box. It grounds spatial questions
in real-world infrastructure the I-GUIDE KB does not hold, and (because it emits
geometry) is directly plottable by a map client.

Location is resolved from an explicit ``bbox`` or, failing that, by geocoding a
``place`` name with the same cached/rate-limited Nominatim helper the geocode tool
uses -- the agent process has network; the code-exec sandbox does not.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Primary + community mirrors. The public instances are frequently overloaded (504),
# so we try them in order. Override/extend with OVERPASS_API_URL (comma-separated).
_DEFAULT_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
)
OVERPASS_ENDPOINTS = [
    e.strip() for e in os.getenv("OVERPASS_API_URL", ",".join(_DEFAULT_ENDPOINTS)).split(",") if e.strip()
]
_TIMEOUT = int(os.getenv("OVERPASS_TIMEOUT_SEC", "30"))
_MAX_FEATURES = int(os.getenv("OVERPASS_MAX_FEATURES", "80"))
# Overpass etiquette (and its nginx) require a descriptive User-Agent; the default
# python-requests UA is rejected with HTTP 406.
_HEADERS = {
    "User-Agent": os.getenv("OVERPASS_USER_AGENT", "iguide-agent/1.0 (overpass_search tool)"),
    "Accept": "application/json",
}

# Friendly feature words -> OSM tag filters. A raw ``key=value`` is accepted too.
_FEATURE_FILTERS: Dict[str, str] = {
    "cafe": "amenity=cafe", "coffee": "amenity=cafe",
    "restaurant": "amenity=restaurant",
    "school": "amenity=school", "university": "amenity=university",
    "hospital": "amenity=hospital", "clinic": "amenity=clinic",
    "pharmacy": "amenity=pharmacy", "fire station": "amenity=fire_station",
    "police": "amenity=police",
    "park": "leisure=park", "playground": "leisure=playground",
    "supermarket": "shop=supermarket", "grocery": "shop=supermarket",
    "river": "waterway=river", "stream": "waterway=stream", "canal": "waterway=canal",
    "dam": "waterway=dam", "weir": "waterway=weir",
    "water": "natural=water", "lake": "natural=water", "wetland": "natural=wetland",
    "forest": "landuse=forest", "wood": "natural=wood",
    "road": "highway=primary", "highway": "highway=motorway", "street": "highway=residential",
    "railway": "railway=rail", "rail": "railway=rail", "subway": "railway=subway",
    "bridge": "man_made=bridge",
    "power plant": "power=plant", "power line": "power=line", "substation": "power=substation",
    "building": "building", "school district": "amenity=school",
    "airport": "aeroway=aerodrome", "farmland": "landuse=farmland",
}

# Tags that indicate an area (closed way -> Polygon rather than LineString).
_AREA_TAG_KEYS = ("building", "landuse", "leisure", "natural", "amenity", "shop", "aeroway")
# Preference order for the human-readable primary tag.
_PRIMARY_TAG_KEYS = (
    "amenity", "shop", "leisure", "waterway", "natural", "landuse",
    "highway", "railway", "power", "man_made", "aeroway", "building",
)


def _resolve_filter(feature: str) -> Optional[str]:
    """Map a free-text feature word to an OSM ``key=value`` filter, or pass a raw one through."""
    f = (feature or "").strip()
    if not f:
        return None
    stripped = f.strip("[]")
    if "=" in stripped:                      # raw OSM filter, e.g. amenity=school
        return stripped
    low = f.lower()
    if low in _FEATURE_FILTERS:
        return _FEATURE_FILTERS[low]
    for word in (low, low.rstrip("s")):      # tolerate simple plurals
        if word in _FEATURE_FILTERS:
            return _FEATURE_FILTERS[word]
    # A single lowercase token with no spaces is treated as a bare OSM key (e.g. "waterway").
    if " " not in low and low.replace("_", "").isalpha():
        return low
    return None                              # -> caller falls back to a name-based search


def _resolve_bbox(place: Optional[str], bbox: Any) -> Optional[Tuple[float, float, float, float]]:
    if bbox:
        try:
            parts = ([float(x) for x in bbox.replace("[", "").replace("]", "").split(",")]
                     if isinstance(bbox, str) else [float(x) for x in bbox])
            if len(parts) == 4:
                return (parts[0], parts[1], parts[2], parts[3])
        except Exception:  # noqa: BLE001
            logger.warning("overpass: could not parse bbox %r", bbox)
    if place:
        try:
            from rag_pipeline.search.opengeodata_new import geocode_place
            return geocode_place(place)
        except Exception as exc:  # noqa: BLE001
            logger.warning("overpass: geocode failed for %r: %s", place, exc)
    return None


def _primary_tag(tags: Dict[str, Any]) -> str:
    for key in _PRIMARY_TAG_KEYS:
        if key in tags:
            return f"{key}={tags[key]}"
    return ""


def _element_to_feature(el: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    tags = el.get("tags") or {}
    etype = el.get("type")
    lon = lat = None
    geometry: Optional[Dict[str, Any]] = None

    if etype == "node" and el.get("lon") is not None:
        lon, lat = el["lon"], el["lat"]
        geometry = {"type": "Point", "coordinates": [lon, lat]}
    elif etype == "way":
        geom = el.get("geometry")
        if geom:
            coords = [[p["lon"], p["lat"]] for p in geom if p.get("lon") is not None]
            if len(coords) < 2:
                return None
            closed = len(coords) > 3 and coords[0] == coords[-1]
            area_like = any(k in tags for k in _AREA_TAG_KEYS)
            geometry = ({"type": "Polygon", "coordinates": [coords]}
                        if (closed and area_like)
                        else {"type": "LineString", "coordinates": coords})
            center = el.get("center") or {}
            lon = center.get("lon", sum(c[0] for c in coords) / len(coords))
            lat = center.get("lat", sum(c[1] for c in coords) / len(coords))
        else:
            center = el.get("center") or {}
            lon, lat = center.get("lon"), center.get("lat")
            if lon is None:
                return None
            geometry = {"type": "Point", "coordinates": [lon, lat]}
    else:
        return None

    return {
        "osm_type": etype,
        "osm_id": el.get("id"),
        "name": tags.get("name") or "(unnamed)",
        "lat": round(lat, 6) if lat is not None else None,
        "lon": round(lon, 6) if lon is not None else None,
        "feature_type": _primary_tag(tags),
        "tags": tags,
        "geometry": geometry,
    }


def overpass_search(
    feature: str,
    place: Optional[str] = None,
    bbox: Any = None,
    limit: int = _MAX_FEATURES,
) -> Dict[str, Any]:
    """Query live OSM features of ``feature`` type inside ``place`` (geocoded) or ``bbox``.

    Returns ``{"query", "count", "features": [{name, lat, lon, feature_type, tags,
    geometry}]}``. On a missing location or a transport failure, returns an
    ``{"error", "message", "features": [], "count": 0}`` payload instead of raising.
    """
    osm_filter = _resolve_filter(feature)
    region = _resolve_bbox(place, bbox)
    if region is None:
        return {
            "error": "no_location",
            "message": "Provide a `place` (e.g. 'Cook County, Illinois') or a `bbox` "
                       "as 'minLon,minLat,maxLon,maxLat' to bound the OSM query.",
            "features": [], "count": 0,
        }

    minlon, minlat, maxlon, maxlat = region
    bbox_str = f"{minlat},{minlon},{maxlat},{maxlon}"          # Overpass order: S,W,N,E
    limit = max(1, min(int(limit or _MAX_FEATURES), 500))

    if osm_filter:
        selectors = f"node[{osm_filter}]({bbox_str});\n      way[{osm_filter}]({bbox_str});"
    else:                                                       # name-based fallback
        safe = str(feature).replace('"', "")
        selectors = (f'node["name"~"{safe}",i]({bbox_str});\n'
                     f'      way["name"~"{safe}",i]({bbox_str});')

    query = f"[out:json][timeout:{_TIMEOUT}];\n(\n      {selectors}\n);\nout geom {limit};"

    data = None
    last_error: Optional[str] = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            resp = requests.post(endpoint, data={"data": query}, headers=_HEADERS, timeout=_TIMEOUT + 5)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            last_error = str(exc)
            logger.warning("overpass: %s failed (%s); trying next mirror", endpoint, exc)
    if data is None:
        return {"error": "overpass_failed", "message": last_error or "all Overpass mirrors failed",
                "features": [], "count": 0}

    features: List[Dict[str, Any]] = []
    for el in data.get("elements", []):
        feat = _element_to_feature(el)
        if feat:
            features.append(feat)
        if len(features) >= limit:
            break

    return {
        "query": {"feature": feature, "osm_filter": osm_filter, "place": place,
                  "bbox": [minlon, minlat, maxlon, maxlat]},
        "count": len(features),
        "features": features,
    }
