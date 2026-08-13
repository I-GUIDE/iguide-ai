from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict, List, MutableMapping, Optional

import requests
from flask import Flask, jsonify, request
try:
    from flask_cors import CORS
except Exception:  # pragma: no cover - optional dependency
    def CORS(app):  # type: ignore
        return app
from opensearchpy import OpenSearch
from dotenv import load_dotenv

from .utils import get_logger, getenv, normalize_source_fields, safe_score, default_top_k
from ..state import EvidenceEntry, ensure_state_shapes, get_query_text, merge_retrieval

load_dotenv()

logger = get_logger("spatial_search")
app = Flask(__name__)
CORS(app)

try:
    import spacy

    nlp = spacy.load("en_core_web_sm")
except Exception as exc:  # pragma: no cover - optional dependency
    logger.warning("Failed to load Spacy model 'en_core_web_sm': %s", exc)
    nlp = None


@lru_cache(maxsize=1)
def _os_client() -> OpenSearch:
    node = getenv("OPENSEARCH_NODE")
    user = getenv("OPENSEARCH_USERNAME", required=False, default="")
    pwd = getenv("OPENSEARCH_PASSWORD", required=False, default="")
    use_ssl = node.lower().startswith("https")
    return OpenSearch(
        hosts=[node],
        http_auth=(user, pwd) if (user or pwd) else None,
        use_ssl=use_ssl,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        timeout=30,
        max_retries=2,
        retry_on_timeout=True,
    )


def _os_index(default: Optional[str] = None) -> str:
    index = default or getenv("OPENSEARCH_INDEX")
    if not index:
        raise RuntimeError("OPENSEARCH_INDEX must not be empty")
    return index


# infer_geo_shape now lives in the leaf module geo_shapes.py and is re-exported here so the
# existing caller below is unaffected. Moved because importing THIS module builds a Flask app
# and loads a spaCy model — 1.2s and a warning — which the dataset extractor should not pay to
# turn two coordinate pairs into an envelope.
from .geo_shapes import bbox_geo_shape, infer_geo_shape, to_wgs84_bounds  # noqa: F401,E402

def _format_search_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for hit in hits:
        doc_id = str(hit.get("_id", "") or "")
        source = normalize_source_fields(hit.get("_source", {}) or {}, doc_id)
        results.append(
            {
                "_id": doc_id,
                "_score": safe_score(hit.get("_score", 0.0)),
                "_source": source,
            }
        )
    return results


def _scroll_all_documents(search_body: Dict[str, Any], scroll_duration: str = "30s") -> List[Dict[str, Any]]:
    client = _os_client()
    response = client.search(index=_os_index(), body=search_body, scroll=scroll_duration)
    scroll_id = response.get("_scroll_id")

    collected: List[Dict[str, Any]] = []
    while True:
        hits = response.get("hits", {}).get("hits", [])
        if not hits:
            break
        collected.extend(hits)
        response = client.scroll(scroll_id=scroll_id, scroll=scroll_duration)
        scroll_id = response.get("_scroll_id")

    if scroll_id:
        try:
            client.clear_scroll(scroll_id=scroll_id)
        except Exception as exc:  # pragma: no cover - best effort cleanup
            logger.warning("Failed to clear OpenSearch scroll: %s", exc)

    return collected


def spatial_search(
    coords: Any,
    keyword: Optional[str] = None,
    relation: str = "INTERSECTS",
    limit: Any = "unlimited",
    element_type: Optional[str] = None,
) -> List[Dict[str, Any]] | Dict[str, Any]:
    try:
        coords_array = json.loads(coords) if isinstance(coords, str) else coords
        shape = infer_geo_shape(coords_array)
        logger.debug("Inferred geo shape: %s", shape)

        filters: List[Dict[str, Any]] = [
            {
                "geo_shape": {
                    "spatial-bounding-box-geojson": {
                        "shape": shape,
                        "relation": str(relation or "INTERSECTS").upper(),
                    }
                }
            }
        ]
        if element_type:
            filters.append({"term": {"resource-type": element_type}})

        bool_query: Dict[str, Any] = {"bool": {"filter": filters}}
        if keyword:
            bool_query["bool"]["must"] = [
                {
                    "multi_match": {
                        "query": keyword,
                        "fields": ["title^3", "authors^3", "tags^2", "contents", "contributor^3"],
                        "type": "best_fields",
                    }
                }
            ]

        search_body = {"query": bool_query, "track_total_hits": True}
        client = _os_client()
        if str(limit).isdigit():
            size = int(limit)
            response = client.search(index=_os_index(), body={**search_body, "size": size})
            hits = response.get("hits", {}).get("hits", [])
        else:
            hits = _scroll_all_documents(search_body)

        return _format_search_hits(hits)
    except Exception as exc:
        logger.error("Spatial search failed: %s", exc)
        return {"error": str(exc)}


@app.route("/search/spatial", methods=["GET", "OPTIONS"])
def spatial_search_endpoint():
    if request.method == "OPTIONS":
        return "", 200

    try:
        coords = request.args.get("coords")
        keyword = request.args.get("keyword")
        relation = request.args.get("relation", "INTERSECTS")
        limit = request.args.get("limit", "unlimited")
        element_type = request.args.get("element-type")

        if not coords:
            return jsonify({"error": "Missing required query parameter: coords"}), 400

        result = spatial_search(coords, keyword, relation, limit, element_type)
        if isinstance(result, dict) and "error" in result:
            status = 400 if "Missing" in result["error"] else 500
            return jsonify(result), status
        return jsonify(result)
    except Exception as exc:
        logger.error("Spatial search endpoint error: %s", exc)
        return jsonify({"error": str(exc)}), 500


# Physical-geography nouns. A named feature is only geocodable WITH its feature word: spaCy tags
# "Amazon" in "the Amazon basin" as LOC and stops there, and Google returns no result for a bare
# "Amazon" (it is hopelessly ambiguous), so the whole spatial search silently produced nothing for
# the vocabulary geospatial users actually type.
_FEATURE_WORDS = (
    "basin", "watershed", "river", "delta", "estuary", "valley", "range", "mountains", "mountain",
    "gulf", "bay", "sound", "strait", "lake", "lakes", "sea", "ocean", "reef", "desert", "forest",
    "plain", "plains", "plateau", "peninsula", "island", "islands", "aquifer", "canyon", "glacier",
    "prairie", "wetland", "wetlands", "coast", "county", "parish", "province", "region",
)

# Capitalized technical tokens that are never places. Candidate phrases are validated by the
# geocoder, so this only saves wasted lookups on the terms we know will fail.
_NOT_PLACES = {
    # coordinate systems / formats / tooling
    "utm", "epsg", "crs", "srs", "wgs", "nad", "dem", "dtm", "dsm", "gis", "api", "rest",
    "csv", "tsv", "json", "geojson", "cog", "tiff", "geotiff", "netcdf", "hdf", "zarr", "las",
    "laz", "shp", "shapefile", "gpkg", "wkt", "wkb",
    "gdal", "ogr", "proj", "pdal", "postgis", "qgis", "arcgis", "grass", "saga", "whitebox",
    "python", "r", "sql", "geopandas", "rasterio", "xarray", "numpy", "pandas",
    # standards bodies and mission/sensor names, which are organizations not places
    "stac", "ogc", "iso", "inspire", "nasa", "usgs", "noaa", "fema", "epa", "esa", "eumetsat",
    "modis", "landsat", "sentinel", "viirs", "srtm", "aster", "naip", "nhd", "nhdplus", "nlcd",
    "i-guide", "iguide", "cyberGIS".lower(), "hydroshare",
    # sentence filler that survives capitalization at the start of a query
    "part", "zone", "what", "which", "how", "where", "the", "a", "an", "and", "or", "for",
}


def _feature_phrase(doc, ent) -> Optional[str]:
    """``ent`` plus a following feature word, when there is one ("Amazon" -> "Amazon basin")."""
    tail = doc[ent.end] if ent.end < len(doc) else None
    if tail is not None and tail.text.lower() in _FEATURE_WORDS:
        return f"{ent.text} {tail.text}"
    return None


def _capitalized_candidates(user_query: str, limit: int = 3) -> List[str]:
    """Fallback place candidates from capitalization, for when NER finds nothing.

    The retrieval peer rewrites questions into keyword form ("UTM zone Champaign Illinois"), which
    strips the determiners and prepositions spaCy's NER leans on — it returned NO entities for that
    string while extracting both places from the original sentence. Candidates here are deliberately
    liberal because the geocoder is the arbiter: a non-place simply fails to resolve.
    """
    import re

    runs: List[str] = []
    for match in re.finditer(r"\b([A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,2})\b", user_query):
        phrase = match.group(1).strip()
        words = [w for w in phrase.split() if w.lower() not in _NOT_PLACES]
        if not words:
            continue
        candidate = " ".join(words)
        if len(candidate) < 3:
            continue
        # A lowercase feature word right after the capitalized run belongs to the place name:
        # "Amazon basin" geocodes, "Amazon" does not. Applied here too, so the fallback behaves the
        # same when spaCy's model is not installed.
        tail = user_query[match.end():].lstrip()
        next_word = tail.split()[0].strip(".,;:)!?") if tail else ""
        if next_word.lower() in _FEATURE_WORDS:
            longer = f"{candidate} {next_word}"
            if longer not in runs:
                runs.append(longer)
        if candidate not in runs:
            runs.append(candidate)
    # Longest first: "Champaign Illinois" is a better geocode than "Illinois" alone.
    runs.sort(key=lambda s: -len(s))
    return runs[:limit]


def extract_locations_from_query(user_query: str) -> List[str]:
    """Place names in *user_query*, best candidate first.

    A named entity is emitted WITH its feature word when it has one, because that is the form the
    geocoder can resolve. When NER yields nothing usable, capitalized phrases are offered instead.
    """
    if nlp is None:
        logger.debug("Spacy model unavailable; skipping spatial entity extraction.")
        return _capitalized_candidates(user_query)

    doc = nlp(user_query)
    ordered: List[str] = []
    for ent in doc.ents:
        if ent.label_ not in ("GPE", "LOC", "FAC"):
            continue
        phrase = _feature_phrase(doc, ent)
        # Prefer the fuller phrase, but keep the bare entity as a fallback candidate after it.
        for value in ((phrase, ent.text) if phrase else (ent.text,)):
            if value and value not in ordered:
                ordered.append(value)
    if not ordered:
        return _capitalized_candidates(user_query)
    return ordered


def get_bounding_box(location: str) -> Optional[Dict[str, Any]]:
    api_key = getenv("GOOGLE_MAPS_API_KEY", required=False, default="")
    if not api_key:
        logger.error("Google Maps API key not set.")
        return None

    geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": location, "key": api_key}

    try:
        response = requests.get(geocode_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        logger.error("Error fetching bounding box for %s: %s", location, exc)
        return None

    if not data.get("results"):
        logger.warning("No geocode results for location: %s", location)
        return None

    geometry = data["results"][0]["geometry"]
    bounds = geometry.get("bounds") or geometry.get("viewport")
    if not bounds:
        lat = geometry["location"]["lat"]
        lng = geometry["location"]["lng"]
        buffer = 0.1
        bounds = {
            "southwest": {"lat": lat - buffer, "lng": lng - buffer},
            "northeast": {"lat": lat + buffer, "lng": lng + buffer},
        }

    return {
        "type": "polygon",
        "coordinates": [
            [
                [bounds["southwest"]["lng"], bounds["southwest"]["lat"]],
                [bounds["northeast"]["lng"], bounds["southwest"]["lat"]],
                [bounds["northeast"]["lng"], bounds["northeast"]["lat"]],
                [bounds["southwest"]["lng"], bounds["northeast"]["lat"]],
                [bounds["southwest"]["lng"], bounds["southwest"]["lat"]],
            ]
        ],
    }


def resolve_query_bbox(user_query: str) -> Optional[Dict[str, Any]]:
    """The bounding box for the first candidate place in *user_query* that actually geocodes.

    Trying only ``locations[0]`` meant one unresolvable candidate ended the whole spatial search:
    "the Amazon basin" yielded the bare entity "Amazon", Google returned nothing for it, and the
    method reported no results rather than moving on to a candidate that would have resolved.
    """
    for candidate in extract_locations_from_query(user_query):
        bbox = _cached_bounding_box(candidate)
        if bbox:
            logger.info("Spatial search resolved %r -> %s", candidate, "bbox")
            return bbox
    return None


# Geocoding is a paid, rate-limited third-party call and the same places recur constantly. Failures
# are cached too, so an unresolvable candidate is not retried on every query that mentions it.
_BBOX_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}


def _cached_bounding_box(location: str) -> Optional[Dict[str, Any]]:
    key = str(location or "").strip().lower()
    if not key:
        return None
    if key in _BBOX_CACHE:
        return _BBOX_CACHE[key]
    bbox = get_bounding_box(location)
    if len(_BBOX_CACHE) < 512:
        _BBOX_CACHE[key] = bbox
    return bbox


def get_spatial_search_results(user_query: str, size: int = 10) -> List[Dict[str, Any]]:
    bounding_box = resolve_query_bbox(user_query)
    if not bounding_box:
        return []

    search_body = {
        "query": {
            "bool": {
                "should": [{"match": {"title": user_query}}],
                "filter": [
                    {
                        "geo_shape": {
                            "spatial-bounding-box-geojson": {"shape": bounding_box, "relation": "INTERSECTS"}
                        }
                    }
                ],
            }
        },
        "size": size,
    }

    try:
        response = _os_client().search(index=_os_index(), body=search_body)
        hits = response.get("hits", {}).get("hits", [])
        return _format_search_hits(hits)
    except Exception as exc:
        logger.error("Error performing NLP-based spatial search: %s", exc)
        return []


def _extract_spatial_config(state: MutableMapping[str, Any]) -> Dict[str, Any]:
    session_ctx = state.get("session_context") or {}
    config = session_ctx.get("spatial_search") or {}
    if not isinstance(config, dict):
        config = {}
    return config


def retrieve_spatial(state: MutableMapping[str, Any]) -> List[Dict[str, Any]]:
    """
    Retrieve spatially filtered documents using configuration from the shared state.
    
    If coordinates are not provided in the state, falls back to extracting locations
    from the query text using NLP and geocoding via Google Maps API.
    """
    ensure_state_shapes(state)
    config = _extract_spatial_config(state)

    coords = (
        config.get("coords")
        or state.get("session_context", {}).get("spatial_coords")
        or state.get("params", {}).get("spatial_coords")
    )
    
    # === FALLBACK: NLP + Geocoding if coords not in state ===
    if not coords:
        query_text = get_query_text(state) or ""
        if query_text:
            logger.debug("No coordinates in state; attempting NLP extraction from query")
            limit_value: Any = config.get("limit")
            if limit_value is None:
                limit_value = state.get("params", {}).get("top_k", default_top_k())
            
            # Use the NLP + Maps API flow
            result = get_spatial_search_results(query_text, size=limit_value)
            if result:
                logger.debug(f"Spatial retriever found {len(result)} docs via NLP+geocoding")
                return result
        
        logger.debug("Spatial retriever skipped: missing coordinates and no extractable location")
        return []

    # === DIRECT COORDS PATH ===
    keyword = config.get("keyword")
    if keyword is None:
        keyword = get_query_text(state) or None

    relation = config.get("relation", "INTERSECTS")
    if not isinstance(relation, str):
        relation = "INTERSECTS"

    limit_value: Any = config.get("limit")
    if limit_value is None:
        limit_value = state.get("params", {}).get("top_k", default_top_k())

    element_type = config.get("element_type")

    result = spatial_search(
        coords,
        keyword=keyword,
        relation=relation,
        limit=limit_value,
        element_type=element_type,
    )

    if isinstance(result, list):
        return result

    if isinstance(result, dict) and result.get("error"):
        logger.debug("Spatial retriever error suppressed: %s", result["error"])
    return []


def run_spatial_search(
    state: MutableMapping[str, Any],
    *,
    query: Optional[str] = None,
    limit: int = 10,
    max_total: Optional[int] = None,
    dedupe: bool = True,
    source: str = "spatial",
) -> List[EvidenceEntry]:
    ensure_state_shapes(state)
    actual_query = (query or get_query_text(state)).strip()
    if not actual_query:
        logger.debug("Spatial search skipped: empty query.")
        return []

    hits = get_spatial_search_results(actual_query, size=limit)
    if not hits:
        return []

    return merge_retrieval(
        state,
        source=source,
        hits=hits,
        limit=max_total,
        dedupe=dedupe,
    )


if __name__ == "__main__":  # pragma: no cover - CLI usage
    app.run(debug=True)


__all__ = ["spatial_search", "retrieve_spatial", "run_spatial_search", "get_spatial_search_results"]
