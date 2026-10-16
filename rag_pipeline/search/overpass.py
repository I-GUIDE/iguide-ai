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

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Public mirrors, tried in order until one answers. Override with OVERPASS_API_URL
# (comma-separated). The order comes from probes on 2026-10-08 (stage 40): the most reliable
# first, then two that fail in under a second (cheap to try), then the slower ones, and mail.ru
# last because it can hang for the full timeout. There are only four servers behind these six
# names: overpass-api.de is DNS round-robin over lz4 and z (it stays as a retry), and
# overpass.kumi.systems is a CNAME of overpass.private.coffee (yet each has answered while the
# other returned 500). maps.mail.ru is the one operator independent of both.
_DEFAULT_ENDPOINTS = (
    "https://z.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
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
    # Boundaries are relations in OSM (see _relation_geometry). The bare key matches every kind:
    # administrative, political (electoral districts), census, postal, protected areas.
    "boundary": "boundary", "boundaries": "boundary",
    "administrative boundary": "boundary=administrative",
    "admin boundary": "boundary=administrative",
}

# The drivable road network: every class a car is routed on, as OSMnx's "drive" network keeps
# them: motorway down to residential and living_street, with the _link ramps. Service roads
# (driveways, parking aisles, alleys), tracks, paths and anything tagged private or closed to
# cars are left out, as OSMnx leaves them out; `area=yes` drops pedestrian squares drawn as
# roads. One filter of several clauses, written the way _resolve_filter passes a raw
# `[a][b]` through: the selector wraps it in one pair of brackets.
DRIVE_CLASSES = ("motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
                 "residential", "living_street", "motorway_link", "trunk_link", "primary_link",
                 "secondary_link", "tertiary_link")
DRIVE_NETWORK_FILTER = (
    'highway~"^(' + "|".join(DRIVE_CLASSES) + ')$"]'
    '[area!~"^yes$"][access!~"^(private|no)$"]'
    '[motor_vehicle!~"^(private|no)$"][motorcar!~"^(private|no)$"'
)
# Live, 2026-10-15: asked for the fastest drive across Chicago's Loop, the model called
# overpass_search(feature='highway') and got the 34 motorway pieces, and no word could ask for
# the whole network. These phrases can.
for _phrase in ("road network", "roads network", "street network", "streets network",
                "drive network", "driving network", "drivable network", "drivable roads",
                "drivable road network", "drivable streets", "routable roads", "road graph",
                "street graph", "all roads", "all streets"):
    _FEATURE_FILTERS[_phrase] = DRIVE_NETWORK_FILTER

# Words that map to ONE road class. "highway" means highway=motorway on purpose (in US usage a
# highway is a major road, and the word has meant motorway since the tool was added), but a model
# that meant the OSM key gets a fraction of the streets and no sign of it. The result says so.
_NARROW_ROAD_WORDS = frozenset(("road", "roads", "highway", "highways", "street", "streets"))

# Keys accepted bare (every feature carrying the key). Any other single word is taken as a name.
_BARE_KEYS = frozenset((
    "amenity", "shop", "leisure", "waterway", "natural", "landuse", "highway", "railway",
    "power", "aeroway", "building", "boundary", "place", "tourism", "historic", "office",
    "craft", "emergency", "healthcare", "sport", "barrier", "route", "military", "water",
    "wetland", "bridge", "tunnel", "aerialway", "geological", "telecom", "harbour", "landmark",
))

# Tags that indicate an area (closed way -> Polygon rather than LineString).
_AREA_TAG_KEYS = ("building", "landuse", "leisure", "natural", "amenity", "shop", "aeroway",
                  "boundary", "place")
# Preference order for the human-readable primary tag.
_PRIMARY_TAG_KEYS = (
    "amenity", "shop", "leisure", "waterway", "natural", "landuse",
    "highway", "railway", "power", "man_made", "aeroway", "building", "boundary", "place",
)
# A relation outline larger than this is simplified for what the model and the map read; its
# area is measured on the full outline first, and `full_geometry` keeps it for files.
_MAX_VERTICES = int(os.getenv("OVERPASS_MAX_VERTICES", "1500"))


# One clause of a raw Overpass tag filter: `key`, `!key`, `key=v`, `key!=v`, `key~"re"`,
# `key!~"re"`, `"key"~"re",i`. Keys and values may be quoted.
_CLAUSE_RE = re.compile(
    r'''^\s*(?P<neg>!\s*)?(?P<key>"[^"]+"|[A-Za-z_][\w:.\-]*)\s*'''
    r'''(?:(?P<op>!=|!~|=|~)\s*(?P<val>"[^"]*"|[^\s"\[\]]+)(?:\s*,\s*i)?)?\s*$''')
# `highway in (motorway, trunk, primary)`, a form models write; Overpass has no `in`.
_IN_RE = re.compile(r"^\s*\[?\s*([A-Za-z_][\w:]*)\s+in\s*\(([^)]*)\)\s*\]?\s*$", re.I)


def filter_clauses(osm_filter: Optional[str]) -> List[Dict[str, Any]]:
    """The clauses of a resolved filter as ``{"neg", "key", "op", "value"}``, or [] if any
    clause is not plain Overpass tag-filter syntax."""
    if not osm_filter:
        return []
    out = []
    for part in str(osm_filter).strip().strip("[]").split("]["):
        m = _CLAUSE_RE.match(part)
        if not m:
            return []
        out.append({"neg": bool(m.group("neg")), "key": m.group("key").strip('"'),
                    "op": m.group("op"), "value": (m.group("val") or "").strip('"') or None})
    return out


def is_road_network_filter(osm_filter: Optional[str]) -> bool:
    """Whether a filter asks for a road NETWORK: the `highway` key with more than one value
    (bare, a regex or an exclusion), as opposed to one class such as highway=primary."""
    clauses = filter_clauses(osm_filter)
    return bool(clauses) and clauses[0]["key"] == "highway" and not clauses[0]["neg"] \
        and clauses[0]["op"] != "="


def _resolve_filter(feature: str) -> Optional[str]:
    """Map a free-text feature word to an OSM tag filter, or pass a raw one through.

    Raw Overpass filters pass through as written: `key=v`, `key!=v`, `key~"regex"`,
    `key!~"regex"`, several clauses as `[a][b]`, and a bracketed `[key]` or `key=*` for every
    value of a key. Before stage 69 only a filter containing `=` passed: `highway~"^(primary|
    secondary)$"` became a NAME search for that string and found nothing.
    """
    f = (feature or "").strip()
    if not f:
        return None
    m = _IN_RE.match(f)
    if m:
        values = [v.strip().strip("'\"") for v in m.group(2).split(",") if v.strip().strip("'\"")]
        if values:
            return f'{m.group(1)}~"^(' + "|".join(_name_pattern(v) for v in values) + ')$"'
    stripped = f.strip("[]")
    clauses = filter_clauses(stripped)
    if clauses:
        if len(clauses) == 1 and clauses[0]["op"] == "=" and clauses[0]["value"] == "*":
            return clauses[0]["key"]             # key=* -> every value of the key
        if f.startswith("[") or any(c["op"] for c in clauses):
            return stripped                      # raw filter, e.g. amenity=school, [highway]
    if "=" in stripped:                      # raw OSM filter, e.g. amenity=school
        return stripped
    low = f.lower()
    if low in _FEATURE_FILTERS:
        return _FEATURE_FILTERS[low]
    for word in (low, low.rstrip("s")):      # tolerate simple plurals
        if word in _FEATURE_FILTERS:
            return _FEATURE_FILTERS[word]
    # A bare OSM key ("waterway", "man_made") matches every feature with that key. Any other
    # single word is a NAME: before stage 60 every one-word string counted as a key, so
    # feature="Basco" queried the nonexistent key [basco] and found nothing, silently.
    if f == low and (low in _BARE_KEYS or ("_" in low or ":" in low)) and \
            low.replace("_", "").replace(":", "").isalpha():
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


def _name_pattern(name: str) -> str:
    """``name`` as a literal inside an Overpass QL regex string ("1st District (North)" etc.)."""
    out = []
    for ch in str(name).replace('"', ""):
        out.append("\\\\" + ch if ch in ".^$*+?()[]{}|\\" else ch)
    return "".join(out)


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
    elif etype == "relation":
        return _relation_to_feature(el)
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


def _relation_geometry(el: Dict[str, Any]) -> Tuple[Optional[Any], bool]:
    """(shapely geometry, complete) for a relation fetched with ``out geom``.

    A named boundary is never a single way in OSM: it is a relation whose member ways carry
    the outline in pieces, and usually only the relation carries the name. Before stage 60 the
    query asked for nodes and ways only, so "the boundary of Basco" came back as 17 unnamed
    segments and no polygon, and the agent could not measure an area at all.

    Multipolygon and boundary relations are rebuilt into (multi)polygons from their outer and
    inner members. Any other relation (a river, a route) becomes the lines of its members.
    ``complete`` is False when the outer members do not close into a ring, which happens when
    OSM has a gap; the members are then returned as lines, not guessed into a polygon.
    """
    from shapely.geometry import LineString, MultiLineString
    from shapely.ops import linemerge, polygonize, unary_union

    outer, inner = [], []
    for m in el.get("members") or []:
        if m.get("type") != "way":
            continue
        coords = [(p["lon"], p["lat"]) for p in (m.get("geometry") or [])
                  if p and p.get("lon") is not None]
        if len(coords) < 2:
            continue
        (inner if m.get("role") == "inner" else outer).append(LineString(coords))
    if not outer:
        return None, False
    if (el.get("tags") or {}).get("type") in ("multipolygon", "boundary"):
        faces = list(polygonize(unary_union(outer)))
        if faces:
            shape = unary_union(faces)
            if inner:
                holes = list(polygonize(unary_union(inner)))
                if holes:
                    shape = shape.difference(unary_union(holes))
            return shape, True
        return MultiLineString(outer), False
    merged = linemerge(outer)
    return merged, True


def _geodesic_area_km2(shape: Any) -> Optional[float]:
    try:
        from pyproj import Geod

        area, _ = Geod(ellps="WGS84").geometry_area_perimeter(shape)
        return round(abs(area) / 1e6, 4)
    except Exception as exc:  # noqa: BLE001 - an area is a bonus, never a failure
        logger.warning("overpass: geodesic area failed: %s", exc)
        return None


def _vertex_count(shape: Any) -> int:
    from shapely import get_num_coordinates

    return int(get_num_coordinates(shape))


def _relation_to_feature(el: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    from shapely.geometry import mapping

    tags = el.get("tags") or {}
    try:
        shape, complete = _relation_geometry(el)
    except Exception as exc:  # noqa: BLE001 - one malformed relation must not sink the query
        logger.warning("overpass: relation %s could not be assembled: %s", el.get("id"), exc)
        return None
    if shape is None or shape.is_empty:
        return None
    is_area = shape.geom_type in ("Polygon", "MultiPolygon")
    point = shape.representative_point() if is_area else shape.centroid
    feat: Dict[str, Any] = {
        "osm_type": "relation",
        "osm_id": el.get("id"),
        "name": tags.get("name") or "(unnamed)",
        "lat": round(point.y, 6),
        "lon": round(point.x, 6),
        "feature_type": _primary_tag(tags),
        "tags": tags,
        "geometry": mapping(shape),
        "vertices": _vertex_count(shape),
    }
    if is_area:
        # Measured on the ellipsoid from the FULL outline, before any simplification for display.
        feat["area_km2"] = _geodesic_area_km2(shape)
        if tags.get("boundary"):
            # Measured live, 2026-10-15: OSM's outline of Basco, Batanes is 690 km², its land
            # area 33.67 km². Philippine (and many other) boundaries in OSM run out to sea.
            feat["area_note"] = ("area of the boundary as mapped in OSM; a coastal boundary may "
                                 "include sea, so it can exceed the official land area")
    if not complete:
        feat["incomplete"] = ("the outer members do not close into a ring in OSM, so the outline "
                              "is returned as lines and no area is given")
    return feat


def simplify_for_display(feature: Dict[str, Any], max_vertices: int = _MAX_VERTICES) -> Dict[str, Any]:
    """A copy of ``feature`` whose geometry has at most about ``max_vertices`` vertices.

    A province or a coastline runs to tens of thousands of vertices, which would fill the
    model's context and the map's payload. The area was already measured on the full outline;
    the full geometry stays under ``full_geometry`` for anything that writes a file. Only
    features above the budget are touched.
    """
    n = feature.get("vertices")
    if not n or n <= max_vertices or not feature.get("geometry"):
        return feature
    from shapely.geometry import mapping, shape as to_shape

    full = to_shape(feature["geometry"])
    minx, miny, maxx, maxy = full.bounds
    tol = max(maxx - minx, maxy - miny) / 2000.0 or 1e-6
    simple = full
    for _ in range(12):
        simple = full.simplify(tol, preserve_topology=True)
        if _vertex_count(simple) <= max_vertices:
            break
        tol *= 2
    out = dict(feature)
    out["full_geometry"] = feature["geometry"]
    out["geometry"] = mapping(simple)
    out["vertices"] = _vertex_count(simple)
    out["simplified"] = {"vertices_full": n, "tolerance_deg": round(tol, 6),
                         "note": "outline simplified for display; area_km2 is from the full one"}
    return out


def _record_source(endpoint: str) -> None:
    """The mirror that answered, in the calling tool's lineage (agent_runtime/lineage.py)."""
    try:
        from agent_runtime import lineage
    except Exception:  # noqa: BLE001 - this module also runs outside the agent
        return
    lineage.record_source(source_id="osm_overpass", url=endpoint)


RAW_EXCERPT_CHARS = 2000


def _bad_remark(data: Dict[str, Any]) -> Optional[str]:
    remark = str(data.get("remark") or "")
    if "error" in remark.lower() or "timed out" in remark.lower():
        return remark[:300]
    return None


def _exchange(query: str, *, stream_cap: Optional[int] = None,
              http_timeout: Optional[float] = None) -> Dict[str, Any]:
    """Send ``query`` to the first mirror that answers well, and keep the evidence.

    Returns ``{"data", "error", "endpoint", "attempts": [...], "raw_excerpt", "bytes"}``. Every
    attempt is listed with its mirror, HTTP status, seconds and failure, so a caller (or the
    model, through ``debug``) can see WHY a result is what it is: which mirror answered, which
    ones timed out, and what the service itself said.
    """
    attempts: List[Dict[str, Any]] = []
    last_error: Optional[str] = None
    for endpoint in OVERPASS_ENDPOINTS:
        started = time.monotonic()
        attempt: Dict[str, Any] = {"endpoint": endpoint}
        try:
            resp = requests.post(endpoint, data={"data": query}, headers=_HEADERS,
                                 timeout=http_timeout or _TIMEOUT + 5)
            attempt["status"] = getattr(resp, "status_code", None)
            resp.raise_for_status()
            text = getattr(resp, "text", None)
            if stream_cap is not None and text is not None and len(text) > stream_cap:
                raise RuntimeError(f"response is {len(text)} bytes, over the {stream_cap}-byte cap")
            data = resp.json()
            bad = _bad_remark(data)
            if bad:
                # Overpass reports a query that ran out of time or memory as HTTP 200 with a
                # `remark` and an empty or PARTIAL element list. Taken as an answer, it reads as
                # "nothing here". Live, 2026-10-15, one mirror answered a province-sized name
                # search that way, with 1 element and "Query timed out".
                attempt["remark"] = bad
                raise RuntimeError(f"overpass remark: {bad}")
            attempt["seconds"] = round(time.monotonic() - started, 2)
            attempts.append(attempt)
            _record_source(endpoint)
            raw = text if isinstance(text, str) else json.dumps(data)
            return {"data": data, "error": None, "endpoint": endpoint, "attempts": attempts,
                    "raw_excerpt": raw[:RAW_EXCERPT_CHARS], "bytes": len(raw)}
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            last_error = str(exc)
            attempt.update({"seconds": round(time.monotonic() - started, 2),
                            "error": last_error[:300]})
            attempts.append(attempt)
            logger.warning("overpass: %s failed (%s); trying next mirror", endpoint, exc)
    return {"data": None, "error": last_error, "endpoint": None, "attempts": attempts,
            "raw_excerpt": None, "bytes": 0}


def _post(query: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """(response json, None) from the first mirror that answers, or (None, last error)."""
    ex = _exchange(query)
    return ex["data"], ex["error"]


# A raw query from sandboxed code (agent_runtime/tool_bridge.overpass_query). The text goes to a
# public mirror, so its length is capped like a fetched URL's: a long query is the channel data
# would leave by. Output is forced to JSON, and the server-side timeout and memory are capped.
RAW_QUERY_MAX_CHARS = int(os.getenv("OVERPASS_RAW_MAX_CHARS", "4000"))
RAW_TIMEOUT_MAX_S = int(os.getenv("OVERPASS_RAW_TIMEOUT_MAX", "90"))
RAW_MAXSIZE_MAX = int(os.getenv("OVERPASS_RAW_MAXSIZE_MAX", str(256 * 1024 * 1024)))
_SETTING_RE = re.compile(r"\[\s*(out|timeout|maxsize)\s*:\s*([^\]]*)\]", re.I)


class RawQueryRefused(ValueError):
    """The raw query broke a rule; the message says which."""


def prepare_raw_query(ql: str) -> str:
    """``ql`` with JSON output and capped timeout/memory, or RawQueryRefused.

    Overpass QL cannot write, so the rules are about cost and leakage, not about statements:
    the length cap above, JSON only (the caller parses it), ``[timeout:]`` at most
    RAW_TIMEOUT_MAX_S and ``[maxsize:]`` at most RAW_MAXSIZE_MAX, added when missing.
    """
    text = str(ql or "").strip()
    if not text:
        raise RawQueryRefused("the query is empty")
    if len(text) > RAW_QUERY_MAX_CHARS:
        raise RawQueryRefused(f"the query is {len(text)} characters; at most "
                              f"{RAW_QUERY_MAX_CHARS} are sent")
    settings = {m.group(1).lower(): m.group(2).strip() for m in _SETTING_RE.finditer(text)}
    if "out" in settings and settings["out"].lower() != "json":
        raise RawQueryRefused(f"[out:{settings['out']}] is not allowed; use [out:json]")
    try:
        timeout = min(int(settings.get("timeout") or _TIMEOUT), RAW_TIMEOUT_MAX_S)
        maxsize = min(int(settings.get("maxsize") or RAW_MAXSIZE_MAX), RAW_MAXSIZE_MAX)
    except ValueError:
        raise RawQueryRefused("[timeout:] and [maxsize:] take whole numbers") from None
    # Drop the caller's settings and put the capped ones in front: Overpass reads settings only
    # from the first statement, so a capped copy appended later would be ignored.
    body = _SETTING_RE.sub("", text).lstrip()
    if body.startswith(";"):
        body = body[1:].lstrip()
    return f"[out:json][timeout:{timeout}][maxsize:{maxsize}];\n{body}"


def raw_query(ql: str, *, max_bytes: int = 50 * 1024 * 1024) -> Dict[str, Any]:
    """Send a caller-written Overpass QL query (after ``prepare_raw_query``) and return the raw
    answer with the evidence: ``{"query", "data", "error", "endpoint", "attempts", "bytes",
    "raw_excerpt"}``. A query the service reports as timed out fails, mirror by mirror, and the
    attempts carry its remark."""
    query = prepare_raw_query(ql)
    server_timeout = int(re.search(r"\[timeout:(\d+)\]", query).group(1))
    ex = _exchange(query, stream_cap=max_bytes, http_timeout=server_timeout + 10)
    ex["query"] = query
    return ex


def _diagnostics(data: Optional[Dict[str, Any]], features: List[Dict[str, Any]],
                 osm_filter: Optional[str], feature: str) -> Dict[str, Any]:
    """What the answer is made of, and what looks wrong with it. Always returned, and small.

    Live, 2026-10-14: a boundary query returned 158 features and no error, and the agent took it
    as a result. 157 had no name: they were the pieces of outlines whose names live elsewhere.
    Nothing in the payload said so, and the model did not count. This counts for it.
    """
    elements = (data or {}).get("elements") or []
    by_type: Dict[str, int] = {}
    for el in elements:
        by_type[str(el.get("type"))] = by_type.get(str(el.get("type")), 0) + 1
    named = sum(1 for f in features if f.get("name") and f.get("name") != "(unnamed)")
    geoms: Dict[str, int] = {}
    for f in features:
        g = str((f.get("geometry") or {}).get("type"))
        geoms[g] = geoms.get(g, 0) + 1
    out: Dict[str, Any] = {"elements_by_type": by_type, "features": len(features),
                           "named": named, "geometry_types": geoms}
    if (data or {}).get("remark"):
        out["remark"] = str(data["remark"])[:300]
    warnings: List[str] = []
    if features and named == 0:
        warnings.append(f"none of the {len(features)} features has a name. Unnamed lines are "
                        "often the pieces of a larger outline (an OSM relation) whose name is on "
                        "the relation: ask for the place by name as `feature`.")
    elif len(features) >= 10 and named / len(features) < 0.1:
        warnings.append(f"only {named} of {len(features)} features have a name.")
    if not features and not osm_filter:
        warnings.append(f"nothing in the area has a name matching {feature!r}. Check the "
                        "spelling, the local-language name, or widen the area.")
    if str(feature or "").strip().lower() in _NARROW_ROAD_WORDS and osm_filter:
        warnings.append(f"{feature!r} was read as {osm_filter} only, one road class. For every "
                        "street a car can drive, pass feature='road network'; for chosen classes "
                        "pass a raw filter such as highway~\"^(primary|secondary|tertiary)$\".")
    incomplete = sum(1 for f in features if f.get("incomplete"))
    if incomplete:
        warnings.append(f"{incomplete} outline(s) do not close in OSM and came back as lines "
                        "with no area.")
    if warnings:
        out["warnings"] = warnings
    return out


def overpass_search(
    feature: str,
    place: Optional[str] = None,
    bbox: Any = None,
    limit: int = _MAX_FEATURES,
    max_limit: int = 500,
    debug: bool = False,
) -> Dict[str, Any]:
    """Query live OSM features of ``feature`` type inside ``place`` (geocoded) or ``bbox``.

    Returns ``{"query", "count", "features": [{name, lat, lon, feature_type, tags,
    geometry}], "diagnostics"}``. On a missing location or a transport failure, returns an
    ``{"error", "message", "features": [], "count": 0}`` payload instead of raising.

    ``debug=True`` adds ``debug``: the Overpass QL actually sent, every mirror attempt (status,
    seconds, error, the service's remark), and the start of the raw response. It is how a caller
    tells "OSM has nothing here" from "the query asked the wrong thing" from "the mirror failed".
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
    # 500 is this tool's cap, not Overpass's: a model reading 500 features is already a lot of
    # context. The code bridge, whose features go to a file, raises it (agent_runtime/tool_bridge).
    limit = max(1, min(int(limit or _MAX_FEATURES), int(max_limit or 500)))

    if osm_filter:
        sel = f"[{osm_filter}]"
    else:                                                       # name-based fallback
        sel = f'["name"~"{_name_pattern(feature)}",i]'

    # Relations first and on their own `out`, so a full page of ways cannot crowd them out; then
    # nodes and ways, minus the ways that are members of a relation just returned: those are the
    # unnamed pieces of an outline already delivered whole.
    query = (f"[out:json][timeout:{_TIMEOUT}];\n"
             f"relation{sel}({bbox_str})->.r;\n"
             f".r out geom {limit};\n"
             f"(node{sel}({bbox_str}); way{sel}({bbox_str});)->.nw;\n"
             f"(.nw; - way(r.r);)->.rest;\n"
             f".rest out geom {limit};")

    data = None
    exchanges: List[Tuple[str, Dict[str, Any]]] = []
    if not osm_filter:
        # A name is usually a place, and a place's outline is a relation. Ask for relations
        # alone first: over a province-sized box, a regex over every node and way as well ran
        # past Overpass's timeout (2026-10-15, "1st District of Palawan": 21 s for relations
        # alone, timed out for all three). A relation with exactly that name answers it.
        rel_only = f"[out:json][timeout:{_TIMEOUT}];\nrelation{sel}({bbox_str});\nout geom {limit};"
        ex = _exchange(rel_only)
        exchanges.append((rel_only, ex))
        first = ex["data"]
        want = str(feature).strip().casefold()
        if first and any(str((el.get("tags") or {}).get("name") or "").casefold() == want
                         for el in first.get("elements", [])):
            data = first
    last_error: Optional[str] = None
    if data is None:
        ex = _exchange(query)
        exchanges.append((query, ex))
        data, last_error = ex["data"], ex["error"]
    debug_block = None
    if debug:
        debug_block = {"exchanges": [
            {"query": q, "endpoint": e["endpoint"], "attempts": e["attempts"],
             "bytes": e["bytes"], "raw_excerpt": e["raw_excerpt"]} for q, e in exchanges]}
    if data is None:
        failed = {"error": "overpass_failed", "message": last_error or "all Overpass mirrors failed",
                  "features": [], "count": 0}
        if debug_block:
            failed["debug"] = debug_block
        return failed

    features: List[Dict[str, Any]] = []
    for el in data.get("elements", []):
        feat = _element_to_feature(el)
        if feat:
            features.append(feat)
    if not osm_filter:
        # A name search is a regex: "Basco" also finds "Basco Port". The exact name goes first.
        want = str(feature).strip().casefold()
        features.sort(key=lambda f: str(f.get("name") or "").casefold() != want)
    features = features[:limit]

    out = {
        "query": {"feature": feature, "osm_filter": osm_filter, "place": place,
                  "bbox": [minlon, minlat, maxlon, maxlat]},
        "count": len(features),
        "features": features,
        "diagnostics": _diagnostics(data, features, osm_filter, feature),
    }
    if debug_block:
        out["debug"] = debug_block
    return out
