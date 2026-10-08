from __future__ import annotations

import json
import math
import os
import re
import time
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import logging
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple
from ..llm_utils import call_llm

import requests
from requests.adapters import HTTPAdapter, Retry

try:  # pragma: no cover - optional dependency
    from dotenv import dotenv_values
except Exception:  # pragma: no cover
    dotenv_values = None  # type: ignore

from .utils import get_logger
from ..state import EvidenceEntry, ensure_state_shapes, get_query_text, merge_retrieval
from .utils import default_top_k  # shared retrieval window

logger = get_logger("opengeodata_search")


@dataclass
class GeoAsset:
    id: str
    title: str
    abstract: Optional[str]
    keywords: List[str]
    bbox: Optional[Tuple[float, float, float, float]]
    datetime: Optional[Tuple[Optional[str], Optional[str]]]
    license: Optional[str]
    links: Dict[str, str]
    source: str
    provider: Optional[str]


class OpenGeoDataError(Exception):
    pass


class NLQueryError(Exception):
    pass


_API_BASE_ENV_VARS: Sequence[str] = (
    "VLLM_PROXY",
    "OPENAI_API_BASE",
    "OPENAI_BASE_URL",
    "ANVILGPT_URL",
    "API_BASE",
)
_API_KEY_ENV_VARS: Sequence[str] = ("VLLM_API_KEY", "OPENAI_API_KEY", "OPENAI_KEY", "ANVILGPT_KEY", "API_KEY")
_DEFAULT_PROVIDERS: Dict[str, Any] = {
    #"stac": ["https://planetarycomputer.microsoft.com/api/stac/v1"],
    "records": [],
    #"ckan": [("https://api.gsa.gov/technology/datagov/v3/action", None)],
    "cmr": True,
}
_DEFAULT_NL_MODEL = "Qwen/Qwen3.5-9B"


def _hydrate_api_credentials_from_env_files() -> None:
    if not dotenv_values:
        return
    env_candidates = [
        Path(__file__).resolve().parents[1] / ".env",
        Path(__file__).resolve().with_name(".env"),
        Path(__file__).resolve().parents[1] / "opengeodata_prototype" / ".env",
    ]
    target_keys = tuple(set(_API_BASE_ENV_VARS + _API_KEY_ENV_VARS))
    for env_path in env_candidates:
        if not env_path.exists():
            continue
        try:
            values = dotenv_values(env_path)
        except Exception:
            continue
        if not isinstance(values, dict):
            continue
        for key in target_keys:
            if key in os.environ:
                continue
            val = values.get(key)
            if val:
                os.environ[key] = val
        break


_hydrate_api_credentials_from_env_files()


def _get_env_value(names: Sequence[str]) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _session(timeout: int = 12) -> requests.Session:
    sess = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.4,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
    )
    sess.mount("https://", HTTPAdapter(max_retries=retries))
    sess.mount("http://", HTTPAdapter(max_retries=retries))
    original_request = sess.request

    def _request(method: str, url: str, **kwargs: Any):
        kwargs.setdefault("timeout", timeout)
        return original_request(method, url, **kwargs)

    sess.request = _request  # type: ignore[assignment]
    return sess


def _norm_bbox(bbox: Any) -> Optional[Tuple[float, float, float, float]]:
    if not bbox:
        return None
    if isinstance(bbox, dict) and "bbox" in bbox:
        bbox = bbox["bbox"]
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    if isinstance(bbox, dict) and bbox.get("type") == "Polygon":
        coords = bbox.get("coordinates", [[]])[0]
        xs = [coord[0] for coord in coords]
        ys = [coord[1] for coord in coords]
        return (min(xs), min(ys), max(xs), max(ys))
    return None


def _dt_range_from_props(props: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    start = props.get("start_datetime") or props.get("datetime")
    end = props.get("end_datetime") or start
    return (start, end)


def _parse_bbox_from_ckan_spatial(spatial: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    if not spatial:
        return None
    try:
        obj = json.loads(spatial)
        if isinstance(obj, dict):
            if "bbox" in obj:
                return _norm_bbox(obj["bbox"])
            if obj.get("type") in ("Polygon", "MultiPolygon"):
                return _norm_bbox(obj)
    except Exception:
        pass
    match = re.search(r"POLYGON\s*\(\((.*?)\)\)", spatial, re.IGNORECASE)
    if match:
        xs: List[float] = []
        ys: List[float] = []
        for part in match.group(1).split(","):
            coords = part.strip().split()
            if len(coords) >= 2:
                xs.append(float(coords[0]))
                ys.append(float(coords[1]))
        if xs and ys:
            return (min(xs), min(ys), max(xs), max(ys))
    return None


def search_stac(
    endpoint: str,
    q: Optional[str] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    time_range: Optional[Tuple[Optional[str], Optional[str]]] = None,
    collections: Optional[List[str]] = None,
    limit: int = 10,
) -> List[GeoAsset]:
    logger.info(f"STAC search called with: endpoint='{endpoint}', q='{q}', bbox={bbox}, time_range={time_range}, limit={limit}")
    try:
        from pystac_client import Client  # Imported lazily for optional dependency
    except ImportError:
        logger.warning("pystac_client not installed. STAC search will be skipped. Install with: pip install pystac-client")
        return []

    client = Client.open(endpoint)
    params: Dict[str, Any] = {"max_items": limit}
    if bbox:
        params["bbox"] = list(bbox)
    if time_range:
        start, end = time_range
        params["datetime"] = f"{start or '..'}/{end or '..'}"
    if collections:
        params["collections"] = collections

    supports_filter = False
    if hasattr(client, "conforms_to"):
        supports_filter = (
            client.conforms_to("https://api.stacspec.org/v1.0.0/item-search#filter")
            or client.conforms_to("http://www.opengis.net/spec/cql2/1.0/conf/cql2-text")
            or client.conforms_to("http://www.opengis.net/spec/cql2/1.0/conf/cql2-json")
        )
    if q and supports_filter:
        params["filter_lang"] = "cql2-text"
        params["filter"] = f"(title ILIKE '%{q}%') OR (description ILIKE '%{q}%')"

    try:
        search = client.search(**params)
    except Exception:
        params.pop("filter", None)
        params.pop("filter_lang", None)
        search = client.search(**params)

    assets: List[GeoAsset] = []
    for item in search.items():
        landing = getattr(item, "get_self_href", lambda: None)() or ""
        props = item.properties or {}
        keywords = set(props.get("keywords") or [])
        try:
            collection = item.get_collection()
        except Exception:
            collection = None
        if collection:
            try:
                for key in collection.keywords or []:
                    keywords.add(key)
            except Exception:
                pass
        license_value = props.get("license")
        if not license_value and collection:
            try:
                license_value = collection.license
            except Exception:
                pass
        assets.append(
            GeoAsset(
                id=item.id,
                title=props.get("title") or item.id,
                abstract=props.get("description"),
                keywords=list(keywords),
                bbox=_norm_bbox(getattr(item, "bbox", None)),
                datetime=_dt_range_from_props(props),
                license=license_value,
                links={"landing": landing, "api": endpoint},
                source="stac",
                provider=(
                    getattr(collection.providers[0], "name", None)
                    if collection and getattr(collection, "providers", None)
                    else None
                ),
            )
        )
    return assets


def search_ogc_records(
    base: str,
    q: Optional[str] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    time_range: Optional[Tuple[Optional[str], Optional[str]]] = None,
    limit: int = 10,
) -> List[GeoAsset]:
    logger.info(f"OGC Records search called with: endpoint='{base}', q='{q}', bbox={bbox}, time_range={time_range}, limit={limit}")
    sess = _session()
    payload: Dict[str, Any] = {"limit": limit}
    if q:
        payload["q"] = q
    if bbox:
        payload["bbox"] = list(bbox)
    if time_range:
        start, end = time_range
        payload["datetime"] = f"{start or '..'}/{end or '..'}"

    response = sess.post(f"{base.rstrip('/')}/search", json=payload, headers={"accept": "application/geo+json"})
    features: List[Dict[str, Any]] = []
    if response.ok:
        features = response.json().get("features", [])
    else:
        coll_resp = sess.get(f"{base.rstrip('/')}/collections", headers={"accept": "application/json"})
        collections = (coll_resp.json().get("collections", []) if coll_resp.ok else [])[:2]
        for coll in collections:
            page = sess.get(
                f"{base.rstrip('/')}/collections/{coll['id']}/items",
                params={"limit": limit},
                headers={"accept": "application/geo+json"},
            )
            if page.ok:
                features += page.json().get("features", [])

    assets: List[GeoAsset] = []
    for feature in features[:limit]:
        props = feature.get("properties", {}) or {}
        ex_bbox = feature.get("bbox") or props.get("bbox")
        start = props.get("datetime")
        end = props.get("end_datetime")
        links = props.get("links") or feature.get("links") or []
        landing = ""
        if isinstance(links, list) and links:
            landing = links[0].get("href", "") or ""
        assets.append(
            GeoAsset(
                id=str(feature.get("id") or props.get("identifier") or props.get("id")),
                title=props.get("title") or props.get("name") or str(feature.get("id")),
                abstract=props.get("description"),
                keywords=props.get("keywords") or [],
                bbox=_norm_bbox(ex_bbox),
                datetime=(start, end),
                license=props.get("license"),
                links={"landing": landing, "api": base},
                source="ogc-records",
                provider=props.get("publisher") or props.get("provider"),
            )
        )
    return assets


def search_ckan(base: str, api_key: Optional[str] = None, q: str = "", limit: int = 10) -> List[GeoAsset]:
    logger.info(f"CKAN search called with: endpoint='{base}', q='{q}', limit={limit}")
    sess = _session()
    params = {"q": q, "rows": limit}
    headers = {"X-Api-Key": api_key} if api_key else {}
    response = sess.get(f"{base.rstrip('/')}/package_search", params=params, headers=headers)
    response.raise_for_status()
    packages = response.json().get("result", {}).get("results", [])
    assets: List[GeoAsset] = []
    for pkg in packages:
        spatial = pkg.get("spatial")
        bbox = _parse_bbox_from_ckan_spatial(spatial) if spatial else None
        resources = pkg.get("resources") or []
        landing = pkg.get("url")
        if not landing and resources:
            landing = resources[0].get("url", "")
        tags = pkg.get("tags", [])
        if tags and isinstance(tags[0], dict):
            tags = [tag.get("name", "") for tag in tags]
        assets.append(
            GeoAsset(
                id=pkg["id"],
                title=pkg.get("title") or pkg.get("name"),
                abstract=pkg.get("notes"),
                keywords=[tag for tag in tags if tag],
                bbox=bbox,
                datetime=(pkg.get("temporal_start"), pkg.get("temporal_end")),
                license=pkg.get("license_title") or pkg.get("license_id"),
                links={"landing": landing, "api": f"{base.rstrip('/')}/package_show?id={pkg['id']}"},
                source="ckan",
                provider=(pkg.get("organization", {}) or {}).get("title"),
            )
        )
    return assets


def search_cmr_collections(
    q: Optional[str] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    time_range: Optional[Tuple[Optional[str], Optional[str]]] = None,
    limit: int = 10,
) -> List[GeoAsset]:
    logger.info(f"CMR search_cmr_collections called with: q='{q}', bbox={bbox}, time_range={time_range}, limit={limit}")
    sess = _session()
    params: Dict[str, Any] = {"page_size": limit, "include_has_granules": "true"}
    if q:
        params["keyword"] = q
        # # CMR supports multiple search strategies - try both keyword and text search
        # # Fix common typos: "lancover" -> "landcover", "land cover"
        # query_normalized = q.lower().strip()
        # if "lancover" in query_normalized:
        #     query_normalized = query_normalized.replace("lancover", "landcover")
        # params["keyword"] = query_normalized
        # # Also try as text search for better matching
        # params["text"] = query_normalized
    if bbox:
        params["bounding_box"] = ",".join(map(str, bbox))
    if time_range:
        start, end = time_range
        if start or end:
            params["temporal"] = f"{start or ''},{end or ''}"
    response = sess.get("https://cmr.earthdata.nasa.gov/search/collections.json", params=params)
    response.raise_for_status()
    entries = response.json().get("feed", {}).get("entry", []) or []

    assets: List[GeoAsset] = []
    for entry in entries:
        box = None
        if entry.get("boxes"):
            try:
                minlat, minlon, maxlat, maxlon = map(float, entry["boxes"][0].split())
                box = (minlon, minlat, maxlon, maxlat)
            except Exception:
                pass
        landing = ""
        for link in entry.get("links", []):
            rel = link.get("rel", "")
            if rel.endswith("/data#") or rel.endswith("/documentation#") or link.get("href"):
                landing = link.get("href", "")
                break
        assets.append(
            GeoAsset(
                id=entry["id"],
                title=entry.get("dataset_id") or entry.get("short_name") or entry["id"],
                abstract=entry.get("summary"),
                keywords=[
                    ", ".join(keyword.values()) if isinstance(keyword, dict) else str(keyword)
                    for keyword in (entry.get("science_keywords") or [])
                ],
                bbox=box,
                datetime=(entry.get("time_start"), entry.get("time_end")),
                license=None,
                links={"landing": landing, "api": "https://cmr.earthdata.nasa.gov/search/"},
                source="cmr",
                provider=entry.get("archive_center") or entry.get("data_center"),
            )
        )
    return assets


def discover(
    query: str = "",
    bbox: Optional[Tuple[float, float, float, float]] = None,
    time_range: Optional[Tuple[Optional[str], Optional[str]]] = None,
    limit: int = 6,
    providers: Optional[Dict[str, Any]] = None,
) -> List[GeoAsset]:
    providers = providers or dict(_DEFAULT_PROVIDERS)
    logger.info(f"OpenGeoData discover() called: query='{query}', limit={limit}, providers={list(providers.keys())}")
    results: List[GeoAsset] = []
    
    stac_count = 0
    for endpoint in providers.get("stac", []):
        try:
            stac_results = search_stac(endpoint, q=query, bbox=bbox, time_range=time_range, limit=limit)
            stac_count += len(stac_results)
            results += stac_results
            logger.info(f"OpenGeoData STAC provider '{endpoint}' returned {len(stac_results)} results")
        except Exception:
            logger.exception("STAC search failed for %s", endpoint)
    
    records_count = 0
    for endpoint in providers.get("records", []):
        try:
            records_results = search_ogc_records(endpoint, q=query, bbox=bbox, time_range=time_range, limit=limit)
            records_count += len(records_results)
            results += records_results
            logger.info(f"OpenGeoData OGC Records provider '{endpoint}' returned {len(records_results)} results")
        except Exception:
            logger.exception("OGC Records search failed for %s", endpoint)
    
    ckan_count = 0
    for base, api_key in providers.get("ckan", []):
        try:
            ckan_results = search_ckan(base, api_key=api_key, q=query, limit=limit)
            ckan_count += len(ckan_results)
            results += ckan_results
            logger.info(f"OpenGeoData CKAN provider '{base}' returned {len(ckan_results)} results")
        except Exception:
            logger.exception("CKAN search failed for %s", base)
    
    cmr_count = 0
    if providers.get("cmr"):
        try:
            cmr_results = search_cmr_collections(query, bbox=bbox, time_range=time_range, limit=limit)
            cmr_count = len(cmr_results)
            results += cmr_results
            logger.info(f"OpenGeoData CMR provider returned {cmr_count} results")
        except Exception:
            logger.exception("CMR search failed")
    
    logger.info(f"OpenGeoData discover() completed: total results={len(results)} (STAC:{stac_count}, Records:{records_count}, CKAN:{ckan_count}, CMR:{cmr_count})")

    max_results = limit * (
        bool(providers.get("stac"))
        + len(providers.get("records", []))
        + len(providers.get("ckan", []))
        + int(bool(providers.get("cmr")))
    )
    return results[: max(1, max_results)]


def score(
    asset: GeoAsset,
    query_terms: List[str],
    bbox: Optional[Tuple[float, float, float, float]] = None,
    time_range: Optional[Tuple[Optional[str], Optional[str]]] = None,
) -> float:
    text = " ".join(
        [
            asset.title or "",
            asset.abstract or "",
            " ".join(asset.keywords or []),
        ]
    ).lower()
    text_score = sum(1.0 for term in query_terms if term and term.lower() in text)
    spatiotemporal_score = 0.0
    if bbox and asset.bbox:
        ax1, ay1, ax2, ay2 = asset.bbox
        bx1, by1, bx2, by2 = bbox
        inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = inter_w * inter_h
        if inter > 0:
            area = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
            if area > 0:
                spatiotemporal_score += inter / area
    if time_range and asset.datetime and (asset.datetime[0] or asset.datetime[1]):
        spatiotemporal_score += 0.2
    license_bonus = 0.2 if (asset.license and "by" in asset.license.lower()) else 0.0
    return text_score + spatiotemporal_score + license_bonus


def _iso_date(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    match = re.match(r"^\s*(\d{4})-(\d{2})-(\d{2})", value.strip())
    if not match:
        return None
    return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"


def _valid_bbox(b: Optional[List[float]]) -> Optional[Tuple[float, float, float, float]]:
    if not b or len(b) < 4:
        return None
    x1, y1, x2, y2 = map(float, b[:4])
    if not (-180.0 <= x1 <= 180.0 and -180.0 <= x2 <= 180.0 and -90.0 <= y1 <= 90.0 and -90.0 <= y2 <= 90.0):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def get_q_bbox_timer_openai(
    user_query: str,
    *,
    current_date: str,
    api_base: Optional[str] = None,  # Ignored, uses llm_utils
    api_key: Optional[str] = None,    # Ignored, uses llm_utils
    model: Optional[str] = None,      # Ignored, uses llm_utils
    timeout: int = 20,                # Ignored
    default_bbox: Optional[Tuple[float,float,float,float]] = None,
    default_timer: Optional[Tuple[Optional[str],Optional[str]]] = None,
    max_retries: int = 2              # Ignored
) -> Tuple[str, Optional[Tuple[float,float,float,float]], Optional[Tuple[Optional[str],Optional[str]]]]:
    """Use same LLM as generation to parse NL query into (q, bbox, timer)."""
    
    prompt = f"""You are a geospatial query normalizer. Today is {current_date}.

Given a user query, extract:
- q: keyword query (no locations, no dates)
- bbox: [minlon, minlat, maxlon, maxlat] if location mentioned, else null
- timer: [start_date, end_date] if time mentioned, else [null, null]

Examples:
- "dams in illinois" → {{"q": "dams", "bbox": [-91.5, 37.0, -87.5, 42.5], "timer": [null, null]}}
- "recent landcover" → {{"q": "landcover", "bbox": null, "timer": ["2020-01-01", "2025-11-06"]}}

User query: {user_query}

Respond with ONLY JSON: {{"q":"...","bbox":[...]|null,"timer":[...]}}"""

    try:
        response = call_llm(prompt)
        data = json.loads(response)
        
        q = str(data.get("q", "")).strip()
        if not q:
            raise ValueError("Empty q")
        
        bbox = _valid_bbox(data.get("bbox"))
        timer_raw = data.get("timer")
        timer: Optional[Tuple[Optional[str], Optional[str]]] = None
        if isinstance(timer_raw, list) and len(timer_raw) >= 2:
            s = _iso_date(timer_raw[0])
            e = _iso_date(timer_raw[1])
            timer = (s, e)
        
        if bbox is None:
            bbox = default_bbox
        if (timer is None or (timer[0] is None and timer[1] is None)) and default_timer:
            timer = default_timer
        
        return q, bbox, timer
        
    except Exception as e:
        raise NLQueryError(f"NL parse failed: {e}")


def _asset_to_dict(asset: GeoAsset) -> Dict[str, Any]:
    data = asdict(asset)
    if data.get("bbox") is not None:
        data["bbox"] = list(data["bbox"])
    return data


def run_opengeodata(
    *,
    query: Optional[str] = None,
    bbox: Optional[List[float]] = None,
    timer: Optional[List[Optional[str]]] = None,
    limit: int = 10,
    providers: Optional[Dict[str, Any]] = None,
    nl: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run an OpenGeoData discovery query; returns ``{query, bbox, timer, count, assets}``.

    Delegates to the refactored engine in :mod:`opengeodata_new` (new sources — NASA CMR,
    Data.gov, Socrata — with LLM query normalization via the internal ``llm_utils.call_llm``).
    Kept as the stable module entry point so ``get_opengeodata_results`` / ``run_opengeodata_search``
    (and the tests that patch ``opengeodata.run_opengeodata``) keep working unchanged.
    """
    from .opengeodata_new import run_opengeodata as _run_engine

    return _run_engine(
        query=query, bbox=bbox, timer=timer, limit=limit, providers=providers, nl=nl,
    )


def _current_date_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _completion_url(api_base: str) -> str:
    base = (api_base or "").rstrip("/")
    lowered = base.lower()
    if lowered.endswith("/chat/completions") or lowered.endswith("/completions"):
        return base
    if lowered.endswith("/chat"):
        return f"{base}/completions"
    return f"{base}/chat/completions"


def _build_nl_block(
    query: str,
    *,
    session_ctx: Optional[Mapping[str, Any]],
    payload: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    ctx = session_ctx or {}
    og_config = ctx.get("opengeodata") if isinstance(ctx.get("opengeodata"), Mapping) else {}
    if not isinstance(og_config, Mapping):
        og_config = {}

    # Explicit override with full NL block
    explicit = og_config.get("nl")
    if isinstance(explicit, Mapping):
        block = dict(explicit)
    else:
        if og_config.get("disable_nl_parser"):
            return None
        block = {}

    block.setdefault(
        "model",
        og_config.get("nl_model")
        or os.getenv("OPENGEODATA_NL_MODEL")
        or os.getenv("VLLM_MODEL")
        or os.getenv("OPENAI_MODEL")
        or _DEFAULT_NL_MODEL,
    )
    block.setdefault("api_key", og_config.get("nl_api_key") or _get_env_value(_API_KEY_ENV_VARS))
    block.setdefault("api_base", og_config.get("nl_api_base") or _get_env_value(_API_BASE_ENV_VARS))

    block.setdefault("user_query", query)
    block.setdefault("current_date", og_config.get("current_date") or _current_date_iso())

    default_bbox = og_config.get("default_bbox")
    if default_bbox is None:
        default_bbox = payload.get("bbox")
    if default_bbox is not None and "default_bbox" not in block:
        block["default_bbox"] = default_bbox

    default_timer = og_config.get("default_timer")
    if default_timer is None:
        default_timer = payload.get("timer")
    if default_timer is not None and "default_timer" not in block:
        block["default_timer"] = default_timer

    if not block.get("model"):
        return None
    return block


def _maybe_attach_nl(
    payload: Dict[str, Any],
    *,
    query: str,
    session_ctx: Optional[Mapping[str, Any]],
) -> None:
    if payload.get("nl") is not None:
        return
    nl_block = _build_nl_block(query, session_ctx=session_ctx, payload=payload)
    if nl_block:
        payload["nl"] = nl_block


def _payload_from_context(
    query: str,
    *,
    limit: int,
    session_ctx: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"query": query, "limit": max(1, int(limit or 1))}
    ctx = session_ctx or {}

    # Accept either a dedicated opengeodata dict or flat keys for convenience.
    og_config = ctx.get("opengeodata") if isinstance(ctx.get("opengeodata"), Mapping) else {}
    if isinstance(og_config, Mapping):
        og_values = dict(og_config)
    else:
        og_values = {}

    # Allow shorthands like session_ctx["opengeodata_bbox"] = [...]
    if "bbox" not in og_values and "opengeodata_bbox" in ctx:
        og_values["bbox"] = ctx["opengeodata_bbox"]
    if "timer" not in og_values and "opengeodata_timer" in ctx:
        og_values["timer"] = ctx["opengeodata_timer"]

    for key in ("bbox", "timer", "providers", "nl"):
        value = og_values.get(key)
        if value is not None:
            payload[key] = value

    _maybe_attach_nl(payload, query=query, session_ctx=session_ctx)
    return payload


def _score_for_rank(rank: int) -> float:
    """
    Produce a descending score so downstream ranking logic continues to work.
    """
    base = max(rank + 1, 1)
    return round(1.0 / math.log2(base + 1), 6)


def _normalize_assets(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    assets = (raw or {}).get("assets") or []
    hits: List[Dict[str, Any]] = []
    skipped_count = 0
    for idx, asset in enumerate(assets):
        if not isinstance(asset, Mapping):
            skipped_count += 1
            logger.warning(f"OpenGeoData: Skipping asset at index {idx} - not a Mapping, type: {type(asset).__name__}")
            continue
        asset_id = str(asset.get("id") or f"opengeodata-{idx}")
        metadata = {
            "doc_id": asset_id,
            "title": asset.get("title") or "Open geospatial dataset",
            "contents": asset.get("abstract") or "No description provided.",
            "element_type": asset.get("element_type") or "opengeodata",
            "resource-type": asset.get("resource-type") or "opengeodata",
            "tags": asset.get("keywords") or [],
            "keywords": asset.get("keywords") or [],
            "bbox": asset.get("bbox"),
            "datetime": asset.get("datetime"),
            "links": asset.get("links"),
            "license": asset.get("license"),
            "provider": asset.get("provider"),
            "source_system": asset.get("source"),
        }
        hits.append(
            {
                "_id": asset_id,
                "_score": _score_for_rank(idx),
                "_source": metadata,
            }
        )
    if skipped_count > 0:
        logger.warning(f"OpenGeoData: Skipped {skipped_count} invalid assets out of {len(assets)} total")

    return hits


# --- discovery-intent gate -------------------------------------------------------
# External catalog search only makes sense when the user wants to FIND DATA. Running it for
# platform questions ("what are the related elements of X", "how to build the smart search") or
# for operations on an uploaded file produced confident-looking but unrelated datasets, because
# the catalogs match on incidental words. Relevance scoring alone cannot fix those: the terms do
# genuinely appear. So skip the search entirely unless the request looks like data discovery.
_EXTERNAL_DATA_INTENT_RE = re.compile(
    r"\bopen\s+data\b|\bexternal\s+(?:data|datasets?|sources?)\b|\bpublic\s+data\b"
    r"|\b(?:find|search|look\s+for|locate|discover|any|which|recommend|suggest|list|show)\b[^.?!]{0,60}"
    r"\b(?:datasets?|data|imagery|rasters?|satellite|layers?|catalogs?)\b"
    r"|\bdatasets?\s+(?:about|on|for|of|related\s+to|covering)\b"
    r"|\bdata\s+(?:about|on|for|sources?|availability)\b"
    r"|\bdata\.gov\b|\bsocrata\b|\bnasa\b|\bcmr\b|\bdatacite\b|\bzenodo\b|\bstac\b",
    re.I,
)
_PLATFORM_OR_TASK_RE = re.compile(
    r"\bknowledge\s+elements?\b|\brelated\s+elements?\b|\bmost\s+popular\b|\bsmart\s+search\b"
    r"|\bi-?guide\b|\bthis\s+(?:file|csv|dataset|geojson|shapefile|upload)\b|\buploaded\b"
    r"|\bhow\s+(?:to|do|can)\b|\bexplain\b|\bwhat\s+is\b|\bwho\s+(?:are|made)\b"
    r"|\bwrite\s+(?:code|a\s+script)\b|\bbuild\b|\bimplement\b|\bdebug\b"
    r"|\bbuffer\b|\bchoropleth\b|\bheat\s?map\b|\bplot\b|\brender\b|\bvisuali[sz]e\b"
    r"|\b(?:produce|draw|create|generate|make)\b[^.?!]{0,40}\b(?:map|chart|graph|figure|plot)\b"
    r"|\bcartographic\b|\bbubble\s+map\b|\bbasemap\b"
    r"|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.I,
)


def wants_external_data(query: str) -> bool:
    """Whether *query* is a request to DISCOVER external data (so catalog search is warranted).

    An explicit external-data cue always wins — "find open datasets about dams in Illinois" is
    discovery even though it also says "in". Otherwise a platform/task request (knowledge
    elements, how-to, code, an operation on an uploaded file, an element UUID) skips the search.
    """
    text = str(query or "")
    if _EXTERNAL_DATA_INTENT_RE.search(text):
        return True
    return not _PLATFORM_OR_TASK_RE.search(text)


def get_opengeodata_results(
    query: str,
    *,
    limit: Optional[int] = None,
    session_ctx: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    # Resolve the shared window HERE. _payload_from_context does `int(limit or 1)`, so a
    # None reaching it would silently become a single result rather than the window.
    if limit is None:
        limit = default_top_k()
    query = (query or "").strip()
    if not query:
        logger.info("OpenGeoData get_opengeodata_results: empty query, returning empty list")
        return []
    if not wants_external_data(query):
        logger.info("OpenGeoData skipped: %r is not an external-data discovery request", query[:80])
        return []

    payload = _payload_from_context(query, limit=limit, session_ctx=session_ctx)
    logger.info(f"OpenGeoData get_opengeodata_results: calling run_opengeodata with payload keys={list(payload.keys())}")

    try:
        data = run_opengeodata(**payload)
        logger.info(f"OpenGeoData run_opengeodata succeeded: found {data.get('count', 0)} assets")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "OpenGeoData NL query: %s bbox:%s timer:%s",
                data.get("query"),
                data.get("bbox"),
                data.get("timer"),
            )
    except OpenGeoDataError as exc:
        logger.error("OpenGeoData processing failed: %s", exc)
        return []
    except Exception as exc:
        logger.error("OpenGeoData unexpected error: %s", exc)
        return []

    if not isinstance(data, dict):
        logger.warning("OpenGeoData returned invalid payload.")
        return []

    hits = _normalize_assets(data)
    logger.info(f"OpenGeoData normalized {len(hits)} hits from {data.get('count', 0)} assets")
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("OpenGeoData hits: %s", [hit["_source"].get("title") for hit in hits])
    return hits


def retrieve_opengeodata(state: MutableMapping[str, Any]) -> List[Dict[str, Any]]:
    ensure_state_shapes(state)
    query = get_query_text(state)
    if not query:
        logger.info("OpenGeoData retriever skipped: empty query.")
        return []

    params = state.get("params") or {}
    try:
        limit = max(1, int(params.get("top_k", default_top_k())))
    except (TypeError, ValueError):
        limit = 8

    session_ctx = state.get("session_context") or {}
    logger.info(f"OpenGeoData retrieval started: query='{query}', limit={limit}")

    results = get_opengeodata_results(query, limit=limit, session_ctx=session_ctx)
    logger.info(f"OpenGeoData retrieval completed: found {len(results)} results")
    return results


def run_opengeodata_search(
    state: MutableMapping[str, Any],
    *,
    query: Optional[str] = None,
    limit: Optional[int] = None,
    max_total: Optional[int] = None,
    dedupe: bool = True,
    source: str = "opengeodata",
) -> List[EvidenceEntry]:
    ensure_state_shapes(state)
    actual_query = (query or get_query_text(state)).strip()
    if not actual_query:
        return []

    hits = get_opengeodata_results(actual_query, limit=limit, session_ctx=state.get("session_context"))
    if not hits:
        return []

    return merge_retrieval(
        state,
        source=source,
        hits=hits,
        limit=max_total,
        dedupe=dedupe,
    )


__all__ = [
    "OpenGeoDataError",
    "get_opengeodata_results",
    "retrieve_opengeodata",
    "run_opengeodata_search",
]
