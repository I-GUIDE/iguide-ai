"""Named US administrative areas -> a map layer and a zone file, with no upload.

Every polygon tool here (add_map_layer, embed_zones, fit_zone_model) starts from a file the
user attached, so "the embeddings for Champaign County" could only be answered by someone who
already had a boundary file and knew the GEOID inside it. This closes that gap: a name in,
a WGS84 GeoJSON out, already on the map and already shaped for `embed_zones`.

Source is the Census TIGERweb REST API, NOT Earth Engine. Three reasons: the agent container
has no `ee` and no Earth Engine credential (that lives only in the rs-embed service, under a
personal Google account); TIGERweb needs no credential at all; and it carries the one layer
Earth Engine's catalogue does not have — incorporated places, i.e. cities. `TIGER/*/Places`
does not exist in Earth Engine, so a city boundary is unavailable there at any level, and
GAUL/geoBoundaries stop at district (their ADM2 for Kenya is sub-counties; nothing is named
"Nairobi" at all).

US only. Everything here keys off Census FIPS.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent_runtime.map_layers import boundary_layer_id

logger = logging.getLogger(__name__)

TIGERWEB_URL = os.getenv(
    "AGENT_TIGERWEB_URL",
    "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb",
)
TIMEOUT_S = float(os.getenv("AGENT_TIGERWEB_TIMEOUT", "45") or 45)
# A state's tracts run to a few thousand; a whole-country query would be neither useful on a
# map nor affordable to embed. Truncation is REPORTED, never silent.
MAX_FEATURES = int(float(os.getenv("AGENT_TIGERWEB_MAX_FEATURES", "3000") or 3000))

# layer path, and the human word for what a GEOID at that level identifies
_LEVELS: Dict[str, Tuple[str, str]] = {
    "state": ("State_County/MapServer/0", "state"),
    "county": ("State_County/MapServer/1", "county"),
    "city": ("Places_CouSub_ConCity_SubMCD/MapServer/4", "incorporated place"),
    "cdp": ("Places_CouSub_ConCity_SubMCD/MapServer/5", "census designated place"),
}
_TRACTS_LAYER = "Tracts_Blocks/MapServer/0"
_BG_LAYER = "Tracts_Blocks/MapServer/1"

# TIGER's BASENAME is the bare name — "Champaign", never "Champaign County" — but people (and
# models writing tool arguments) say the full English name. Matching the literal string then
# fails, the LIKE fallback fails too (it also searches BASENAME), and the caller gets a dead
# end with no candidates. Watched live: gpt-oss:120b burned three tool calls guessing
# "Champaign County"/Illinois -> "Champaign County"/IL -> "Champaign"/IL before it landed.
# Strip the suffix that belongs to the level being asked for, and keep the original as a
# fallback so a place genuinely named e.g. "Township of Washington" still resolves.
_LEVEL_SUFFIXES: Dict[str, Tuple[str, ...]] = {
    "county": ("county", "parish", "borough", "census area", "municipality",
               "city and borough", "municipio"),
    "city": ("city", "town", "village", "borough", "municipality"),
    "cdp": ("cdp", "census designated place"),
}


def _name_variants(area_text: str, lvl: str) -> List[str]:
    """The name as given, plus the same name with this level's suffix removed."""
    variants = [area_text]
    low = area_text.lower()
    for suffix in _LEVEL_SUFFIXES.get(lvl, ()):
        if low.endswith(" " + suffix):
            trimmed = area_text[: -(len(suffix) + 1)].strip()
            if trimmed and trimmed not in variants:
                variants.append(trimmed)
    return variants


_states_cache: Optional[List[Dict[str, str]]] = None


def _sql_str(text: str) -> str:
    """A single-quoted SQL literal. The value reaches us from the model, so it is escaped."""
    return "'" + str(text or "").replace("'", "''") + "'"


def _query(layer: str, where: str, out_fields: str = "*", *,
           geometry: bool = True, limit: int = MAX_FEATURES) -> Dict[str, Any]:
    """One TIGERweb query as GeoJSON, or an error dict that says what to do about it."""
    import requests

    url = f"{TIGERWEB_URL}/{layer}/query"
    params = {
        "where": where,
        "outFields": out_fields,
        "returnGeometry": "true" if geometry else "false",
        "outSR": "4326",              # the client's map and every tool here are WGS84
        "f": "geojson",
        "resultRecordCount": str(int(limit) + 1),   # +1 so truncation is detectable
    }
    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001 - requests raises a family of these
        return {"error": f"could not reach the Census TIGERweb service at {TIGERWEB_URL}: "
                         f"{type(exc).__name__}",
                "hint": "This tool needs outbound HTTPS. Without it, ask the user to attach a "
                        "boundary file instead of inventing coordinates."}
    if resp.status_code >= 400:
        return {"error": f"TIGERweb returned HTTP {resp.status_code}", "detail": resp.text[:300]}
    try:
        data = resp.json()
    except ValueError:
        return {"error": "TIGERweb returned a non-JSON response", "detail": resp.text[:300]}
    # ArcGIS reports query errors with HTTP 200 and an `error` member.
    if isinstance(data, dict) and data.get("error"):
        return {"error": f"TIGERweb rejected the query: {str(data['error'])[:300]}"}
    feats = data.get("features") if isinstance(data, dict) else None
    if not isinstance(feats, list):
        return {"error": "TIGERweb returned no feature list"}
    return {"features": feats}


def _states() -> List[Dict[str, str]]:
    """[{fips, name, usps}] for the 50 states + DC + territories, fetched once per process."""
    global _states_cache
    if _states_cache is not None:
        return _states_cache
    res = _query(_LEVELS["state"][0], "1=1", "GEOID,NAME,STUSAB,STATE", geometry=False, limit=100)
    rows: List[Dict[str, str]] = []
    for f in res.get("features") or []:
        p = f.get("properties") or {}
        if p.get("STATE") and p.get("NAME"):
            rows.append({"fips": str(p["STATE"]), "name": str(p["NAME"]),
                         "usps": str(p.get("STUSAB") or "")})
    if rows:
        _states_cache = rows
    return rows


def resolve_state(text: str) -> Tuple[Optional[str], List[str]]:
    """(FIPS, candidates). Accepts 'IL', 'Illinois' or '17', case-insensitively."""
    raw = str(text or "").strip()
    if not raw:
        return None, []
    if re.fullmatch(r"\d{1,2}", raw):
        return raw.zfill(2), []
    low = raw.lower()
    rows = _states()
    for row in rows:
        if low in {row["name"].lower(), row["usps"].lower()}:
            return row["fips"], []
    near = [r["name"] for r in rows if low in r["name"].lower()]
    return None, near[:8]


def _write_layer(features: List[Dict[str, Any]], stem: str) -> Dict[str, Any]:
    """Persist a FeatureCollection to the file store; returns its record."""
    from agent_runtime.file_store import create_output_file_from_path

    fname = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")[:60] or "boundary"
    if not fname.endswith(".geojson"):
        fname += ".geojson"
    out = Path(tempfile.mkdtemp(prefix="admin_boundary_")) / fname
    out.write_text(json.dumps({"type": "FeatureCollection", "features": features}),
                   encoding="utf-8")
    return create_output_file_from_path(out, filename=fname)


def _bbox(features: List[Dict[str, Any]]) -> Optional[List[float]]:
    xs: List[float] = []
    ys: List[float] = []

    def walk(coords: Any) -> None:
        if isinstance(coords, (list, tuple)):
            if coords and isinstance(coords[0], (int, float)) and len(coords) >= 2:
                xs.append(float(coords[0]))
                ys.append(float(coords[1]))
                return
            for c in coords:
                walk(c)

    for f in features:
        walk((f.get("geometry") or {}).get("coordinates"))
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def make_admin_boundary_tools() -> List[Any]:
    """The `admin_boundary` StructuredTool. Needs no attached file — that is the point."""
    from langchain_core.tools import StructuredTool

    meta = {"category": "geo"}

    # Words that name a KIND of place. An `area` holding one of these is a mis-slotted level.
    _LEVEL_WORDS = {"city", "cities", "county", "counties", "state", "states", "place", "places",
                    "town", "towns", "municipality", "cdp", "tract", "tracts", "block_group",
                    "block_groups"}

    def admin_boundary(area: str, state: Optional[str] = None, level: str = "county",
                       subdivide: Optional[str] = None, name: Optional[str] = None) -> str:
        """Look up a US state, county or city BY NAME, draw it on the map, and return it as a
        polygon file other tools can use — no upload required.

        `area` is the PLACE NAME — "Urbana", "Champaign County", "Illinois". It is NOT the kind
        of place: that is `level`, and passing "city" as the area finds nothing. (`name` is
        something else again — the stem for the output filename.)

        This is how to answer "the embeddings for Champaign County" or "show me Cook County"
        when the user has attached nothing. `level`: "county" (default), "state", "city"
        (incorporated place) or "cdp". `state` accepts "IL", "Illinois" or "17" and should
        almost always be given — 16 US cities are named Springfield, and a bare name that
        matches more than one place is REFUSED with the candidates listed rather than guessed.

        `subdivide="tracts"` (or "block_groups") returns the census tracts INSIDE the named
        county instead of the county outline — that is the form `embed_zones` wants, one row
        per zone. The result carries `file_id` and `zone_id_field="GEOID"`, so the next call is
        embed_zones(file_id=..., zone_id_field="GEOID").

        US only; it reads the Census TIGERweb service.
        """
        # `area` holds the place NAME but reads like the KIND of place, and a model reading it
        # that way puts "city" in it. Observed live with gpt-oss:120b: four calls, three of them
        # area='city', before it stumbled onto area='Urbana'. `level` already carries the kind,
        # so an `area` holding a level word is unambiguous — recover instead of refusing, and
        # say so, because a silent correction teaches the caller nothing.
        swap_note = None
        if str(area or "").strip().lower() in _LEVEL_WORDS:
            candidate = str(name or "").strip()
            if candidate and candidate.lower() not in _LEVEL_WORDS:
                swap_note = (f"`area` was {area!r}, which is a kind of place rather than a name; "
                             f"used {candidate!r} as the place and kept it as the level. `area` "
                             f"takes the NAME.")
                level = level or area
                area, name = candidate, None
            else:
                return json.dumps({
                    "ok": False,
                    "error": f"`area` is the place NAME, not the kind of place — {area!r} is a "
                             f"level.",
                    "hint": "Call it as admin_boundary(area='Urbana', level='city', "
                            "state='Illinois'). `level` takes city/county/state/cdp; `name` is "
                            "only the output filename."})

        lvl = str(level or "county").strip().lower()
        if lvl in {"place", "town", "municipality"}:
            lvl = "city"
        if lvl not in _LEVELS:
            return json.dumps({"ok": False,
                               "error": f"unknown level {level!r}",
                               "hint": f"use one of: {', '.join(sorted(_LEVELS))}"})
        area_text = str(area or "").strip()
        if not area_text:
            return json.dumps({"ok": False, "error": "no area named",
                               "hint": "pass the name of a state, county or city"})

        # --- resolve the state qualifier ------------------------------------------------
        state_fips = None
        if state:
            state_fips, near = resolve_state(state)
            if state_fips is None:
                return json.dumps({"ok": False, "error": f"unknown state {state!r}",
                                   "hint": "use a name ('Illinois'), a USPS code ('IL') or a "
                                           "FIPS code ('17')",
                                   **({"did_you_mean": near} if near else {})})

        # BASENAME is the bare name; NAME carries the suffix ("Champaign" vs "Champaign
        # County", "Champaign city"), so matching NAME loses every county the user names
        # without saying "County".
        variants = _name_variants(area_text, lvl)
        name_match = " OR ".join(f"UPPER(BASENAME)={_sql_str(v.upper())}" for v in variants)
        clauses = [f"({name_match})"]
        if lvl == "state":
            clauses = [f"({name_match} OR UPPER(STUSAB)={_sql_str(area_text.upper())})"]
        elif state_fips:
            clauses.append(f"STATE={_sql_str(state_fips)}")
        found = _query(_LEVELS[lvl][0], " AND ".join(clauses),
                       "GEOID,NAME,BASENAME,STATE", geometry=True)
        if found.get("error"):
            return json.dumps({"ok": False, **found})
        feats = found["features"]

        # A city that is not incorporated is a CDP; trying that automatically saves a turn
        # spent discovering the distinction, which is a Census detail, not the user's problem.
        if not feats and lvl == "city":
            alt = _query(_LEVELS["cdp"][0], " AND ".join(clauses),
                         "GEOID,NAME,BASENAME,STATE", geometry=True)
            if alt.get("features"):
                feats = alt["features"]
                lvl = "cdp"

        if not feats:
            like = _query(_LEVELS[lvl][0],
                          "(" + " OR ".join(
                              f"UPPER(BASENAME) LIKE {_sql_str('%' + v.upper() + '%')}"
                              for v in variants) + ")"
                          + (f" AND STATE={_sql_str(state_fips)}" if state_fips else ""),
                          "NAME,STATE", geometry=False, limit=10)
            names = sorted({(f.get("properties") or {}).get("NAME")
                            for f in (like.get("features") or [])} - {None})
            return json.dumps({"ok": False,
                               "error": f"no {_LEVELS[lvl][1]} named {area_text!r}"
                                        + (f" in state {state_fips}" if state_fips else ""),
                               **({"did_you_mean": names[:8]} if names else
                                  {"hint": "check the spelling, or try level='city' / "
                                           "level='county'"})})

        def describe(f: Dict[str, Any]) -> Dict[str, Any]:
            p = f.get("properties") or {}
            return {"geoid": p.get("GEOID"), "name": p.get("NAME"), "state_fips": p.get("STATE")}

        # Ambiguity is REPORTED, not resolved by picking the first row. Verified against the
        # live service: 'Springfield' matches 16 incorporated places nationwide, and the first
        # of them is in none of the states anyone means.
        if len(feats) > 1 and not state_fips and lvl != "state":
            return json.dumps({
                "ok": False,
                "error": f"{len(feats)} places named {area_text!r} — say which state",
                "candidates": [describe(f) for f in feats[:12]],
                "hint": "call again with state=, e.g. state='IL'"}, default=str)

        matched = [describe(f) for f in feats]
        truncated = len(feats) > MAX_FEATURES
        feats = feats[:MAX_FEATURES]

        # --- optionally return what is INSIDE it, which is what embed_zones wants ---------
        zone_note = None
        if subdivide:
            sub = str(subdivide).strip().lower().rstrip("s").replace(" ", "_")
            layer = {"tract": _TRACTS_LAYER, "block_group": _BG_LAYER,
                     "blockgroup": _BG_LAYER}.get(sub)
            if layer is None:
                return json.dumps({"ok": False, "error": f"cannot subdivide into {subdivide!r}",
                                   "hint": "use subdivide='tracts' or 'block_groups'"})
            if lvl not in {"county", "state"}:
                return json.dumps({
                    "ok": False,
                    "error": ("subdivide is only defined for a county or a state; this is "
                              f"a {_LEVELS[lvl][1]}"),
                    "hint": "tracts nest inside counties, not inside city limits. Ask for the "
                            "county, or take the city outline and intersect it yourself."})
            if len(matched) > 1:
                return json.dumps({"ok": False,
                                   "error": f"{len(matched)} areas matched, so it is ambiguous "
                                            "which one to subdivide",
                                   "candidates": matched[:12]}, default=str)
            geoid = str(matched[0]["geoid"] or "")
            where = f"STATE={_sql_str(geoid[:2])}"
            if lvl == "county":
                where += f" AND COUNTY={_sql_str(geoid[2:5])}"
            inner = _query(layer, where, "GEOID,NAME,STATE,COUNTY", geometry=True)
            if inner.get("error"):
                return json.dumps({"ok": False, **inner})
            if not inner["features"]:
                return json.dumps({"ok": False,
                                   "error": f"no {sub}s found inside {matched[0]['name']}"})
            truncated = len(inner["features"]) > MAX_FEATURES
            feats = inner["features"][:MAX_FEATURES]
            zone_note = f"{len(feats)} {sub}s inside {matched[0]['name']}"

        stem = name or (zone_note and f"{matched[0]['name']}_{subdivide}") or \
            (matched[0]["name"] if len(matched) == 1 else f"{area_text}_{lvl}")
        try:
            rec = _write_layer(feats, str(stem))
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"ok": False,
                               "error": f"could not save the boundary: {type(exc).__name__}: {exc}"})

        label = zone_note or (matched[0]["name"] if len(matched) == 1
                              else f"{area_text} ({len(matched)})")
        result: Dict[str, Any] = {
            "ok": True,
            **({"note": swap_note} if swap_note else {}),
            "level": lvl,
            "matched": matched[:12],
            "feature_count": len(feats),
            "file_id": rec.get("file_id"),
            "download_url": rec.get("download_url"),
            "filename": rec.get("filename"),
            # Named so the next call does not have to guess the identifier column.
            "zone_id_field": "GEOID",
            "geoids": [str((f.get("properties") or {}).get("GEOID")) for f in feats[:50]],
            "bbox": _bbox(feats),
            "source": "US Census TIGERweb",
            "next_step": (f"embed_zones(file_id={rec.get('file_id')!r}, "
                          "zone_id_field='GEOID') embeds each of these polygons"),
            # Drawn as an OUTLINE: a boundary is a frame for whatever is analysed inside it,
            # and a filled polygon would hide the raster embed_zones puts underneath.
            # Keyed on the FILE, not on the label. embed_zones redraws these same polygons
            # with what it found inside them, and it knows this file_id — so an explicit id
            # lets it take this layer's place instead of adding a second outline of the same
            # city. Without one, build_map_layer invents `agent-<slug of the label>`, which
            # nothing downstream can reconstruct.
            "map_layer": {"url": rec.get("download_url"), "label": label, "render": "shapes",
                          "id": boundary_layer_id(str(rec.get("file_id") or "")),
                          "source": "analysis", "count": len(feats), "outline": True},
        }
        if truncated:
            result["truncated"] = (f"only the first {MAX_FEATURES} features were kept "
                                   f"(AGENT_TIGERWEB_MAX_FEATURES)")
        if zone_note and len(feats) > 8:
            # Measured, not guessed: Champaign County's 48 tracts planned 1140 tiles at the
            # default tile_px. embed_zones no longer caps itself, so that sweep now RUNS —
            # every one of those tiles is a request to the imagery provider. Say it here,
            # while the budget can still be chosen, rather than after the sweep is spent.
            result["coverage_hint"] = (
                f"embedding all {len(feats)} zones sweeps the whole county: a county of tracts "
                "planned ~1140 tiles in testing, and embed_zones is uncapped by default, so it "
                "will fetch every tile the zones touch. Pass max_tiles=<n> for a bounded, "
                "partial look, or zone_ids=[...] to embed a few zones properly and cheaply.")
        if len(matched) > 1:
            result["note"] = (f"{len(matched)} areas matched {area_text!r} in this state; all "
                              "are included")
        return json.dumps(result, default=str)

    return [StructuredTool.from_function(
        func=admin_boundary, name="admin_boundary", metadata=meta,
        description=(
            "THE tool for the boundary of a named US state, county or city — 'show me "
            "Champaign County on the map', 'the outline of Cook County, Illinois', 'the "
            "city limits of Boulder'. Prefer it over overpass_search / OpenStreetMap for "
            "these: OSM returns fragmentary boundary ways, this returns the one "
            "authoritative Census polygon with its GEOID. Puts the boundary on the map and "
            "returns it as a polygon file — NO upload needed — so it is also how to get a "
            "layer for embed_zones / add_map_layer when the user has attached nothing. "
            "`level`: 'county' (default), "
            "'state', 'city'. Always pass `state` when you know it — a bare name matching "
            "several places is refused with the candidates rather than guessed. "
            "`subdivide=\"tracts\"` returns the census tracts inside the named county, which "
            "is the many-zone input embed_zones is for. The result's `file_id` plus "
            "`zone_id_field='GEOID'` feed straight into embed_zones and fit_zone_model."))]


__all__ = ["make_admin_boundary_tools", "resolve_state", "TIGERWEB_URL"]
