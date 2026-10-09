"""The data sources the agent can reach, what each covers, and which one a result came from.

Before stage 45 a source was chosen by the model reading tool descriptions, and nothing recorded
what each source includes or leaves out (docs/design-review-2026-10.md, flaw 6). Live,
2026-10-08: "18 schools within 1 mile" came from one district's file (Chicago Public Schools)
while OpenStreetMap has 31, and the answer did not say which. `source` meant four different things
across tools (a service, imagery, a search method, a map-layer origin), and naming a source was
enforced in one place, for two tool families and lists of three or more rows.

This catalog is a list, and the review says so: a list of SOURCES, which change on the scale of
months, not of words, which change with every query. Each entry declares its coverage
(`covers`, `excludes`, `extent`) and its licence. The agent uses it three ways:

* `describe()` gives the peers and the decider each source's coverage, so a choice between OSM
  and a city portal is made knowing what each leaves out (a capability statement, not a rule);
* `outside_extent()` lets a tool refuse a region its source does not cover before calling it;
* `sources_of()` names the source of every result an answer used, and the answer's Sources line
  is rendered from that, deterministically, whatever the model wrote.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse


@dataclass(frozen=True)
class Source:
    id: str
    name: str                       # how the answer names it
    kind: str                       # features | boundaries | elevation | geocoding | ...
    covers: str
    excludes: str
    extent: str
    licence: str
    tools: Tuple[str, ...] = ()
    hosts: Tuple[str, ...] = ()
    bbox: Optional[Tuple[float, float, float, float]] = None   # lon/lat, when the extent is a box
    aliases: Tuple[str, ...] = ()   # other names an answer may use for it

    def line(self) -> str:
        return (f"{self.name} ({self.kind}): covers {self.covers}; leaves out {self.excludes}; "
                f"extent {self.extent}; licence {self.licence}")


# The United States and its territories, generously (Alaska's Aleutians cross 180°, so the box
# starts at -180). A box test is coarse: inside it, a source can still be empty, and each tool
# keeps its own empty-result check (3DEP's all-NoData refusal).
_US = (-180.0, 13.0, -64.0, 72.0)

CATALOG: Tuple[Source, ...] = (
    Source("census_tigerweb", "US Census TIGERweb", "boundaries",
           covers="US states, counties, incorporated places and census tracts, with Census "
                  "area figures (AREALAND, AREAWATER)",
           excludes="anything outside the United States; neighbourhoods and informal areas",
           extent="United States and territories", licence="US public domain",
           tools=("admin_boundary",), hosts=("tigerweb.geo.census.gov",), bbox=_US,
           aliases=("Census", "TIGER")),
    Source("osm_overpass", "OpenStreetMap (via Overpass)", "features",
           covers="features of any kind people have mapped (schools, hospitals, roads, ...), "
                  "public and private alike",
           excludes="whatever volunteers have not mapped; completeness varies by place, and "
                    "attributes are uneven",
           extent="global", licence="ODbL (© OpenStreetMap contributors)",
           tools=("overpass_search",), hosts=("overpass-api.de", "overpass.kumi.systems",
                                              "overpass.private.coffee", "maps.mail.ru",
                                              "z.overpass-api.de", "lz4.overpass-api.de"),
           aliases=("OpenStreetMap", "OSM", "Overpass")),
    Source("osm_nominatim", "OpenStreetMap Nominatim", "geocoding",
           covers="coordinates for place names and addresses",
           excludes="exact building positions for many addresses; ambiguous names resolve to "
                    "one candidate",
           extent="global", licence="ODbL (© OpenStreetMap contributors)",
           tools=("geocode_places",), hosts=("nominatim.openstreetmap.org",),
           aliases=("Nominatim", "OpenStreetMap")),
    Source("usgs_3dep", "USGS 3DEP", "elevation",
           covers="bare-earth elevation, 1 m to 30 m depending on place",
           excludes="anything outside the United States (the service answers with NoData); "
                    "buildings and vegetation heights",
           extent="United States and territories", licence="US public domain",
           tools=("dem_for_region",), hosts=("elevation.nationalmap.gov",), bbox=_US,
           aliases=("3DEP", "USGS")),
    Source("rs_embed", "Satellite foundation-model embeddings (rs-embed service)", "embeddings",
           covers="per-pixel or per-zone embedding vectors from the model named in the call",
           excludes="raw imagery values; anything a model was not trained to represent",
           extent="global, subject to each model's imagery source and dates",
           licence="per model and imagery source; see the result's provenance",
           tools=("embed_region", "embed_zones", "change_detection", "compare_regions"),
           aliases=("rs-embed",)),
    Source("chicago_data_portal", "City of Chicago Data Portal", "tabular",
           covers="City of Chicago datasets (e.g. Chicago Public Schools locations, crime "
                  "incidents)",
           excludes="anything outside the city's own operations: private and charter schools "
                    "not run by CPS, suburbs",
           extent="City of Chicago", licence="City of Chicago terms of use",
           hosts=("data.cityofchicago.org",), aliases=("Chicago Data Portal",)),
    Source("census_data", "US Census Bureau data", "tabular",
           covers="population and survey tables (decennial census, ACS)",
           excludes="anything outside the United States; figures between survey years",
           extent="United States", licence="US public domain",
           hosts=("api.census.gov", "www2.census.gov", "data.census.gov"), bbox=_US,
           aliases=("Census",)),
    Source("usgs_data", "USGS data services", "tabular",
           covers="USGS products fetched by URL (e.g. the earthquake catalog, ComCat)",
           excludes="anything a given product does not cover",
           extent="per product", licence="US public domain",
           hosts=("earthquake.usgs.gov", "www.usgs.gov", "waterservices.usgs.gov",
                  "tnmaccess.nationalmap.gov", "api.water.usgs.gov", "api.waterdata.usgs.gov",
                  "waterdata.usgs.gov"),
           aliases=("USGS",)),
    # Reached only through the code bridge's fetch (public_data_tools.BRIDGE_EXTRA_HOSTS).
    Source("planetary_computer", "Microsoft Planetary Computer (Sentinel-2 L2A)", "imagery",
           covers="Sentinel-2 surface reflectance scenes, 10-60 m, from 2015, with a STAC "
                  "catalogue and a cropping data API",
           excludes="cloud-covered pixels unless masked; days without a satellite pass",
           extent="global land", licence="Copernicus Sentinel data terms",
           hosts=("planetarycomputer.microsoft.com", "sentinel2l2a01.blob.core.windows.net"),
           aliases=("Planetary Computer", "Sentinel-2")),
    Source("osrm", "OSRM public router (OpenStreetMap roads)", "routing",
           covers="driving routes and times over OpenStreetMap's road network",
           excludes="live traffic; turn restrictions OSM lacks; the demo server's speeds are a "
                    "profile, not measurements",
           extent="global", licence="ODbL (© OpenStreetMap contributors)",
           hosts=("router.project-osrm.org",), aliases=("OSRM",)),
    Source("census_reporter", "Census Reporter (US Census ACS tables)", "tabular",
           covers="American Community Survey estimates and margins by geography, from the "
                  "Census Bureau's releases",
           excludes="anything the ACS does not tabulate; small-area estimates carry wide margins",
           extent="United States", licence="US public domain (data); Census Reporter terms",
           hosts=("api.censusreporter.org",), bbox=_US, aliases=("Census Reporter", "ACS")),
    Source("iguide_kb", "I-GUIDE platform knowledge base", "literature",
           covers="datasets, notebooks, publications and educational resources indexed by the "
                  "I-GUIDE platform",
           excludes="anything not contributed to the platform",
           extent="the platform's index", licence="per item",
           tools=("keyword_search", "semantic_search", "spatial_search", "neo4j_search",
                  "agent_kb_search", "get_kb_block"), aliases=("I-GUIDE",)),
    Source("web", "the open web (search results)", "web",
           covers="pages a web search returned for the query",
           excludes="anything the search did not return; page content is not verified",
           extent="global", licence="per page", tools=("web_search", "web_fetch")),
)

_BY_TOOL: Dict[str, Source] = {t: s for s in CATALOG for t in s.tools}
_BY_HOST: Dict[str, Source] = {h: s for s in CATALOG for h in s.hosts}


def describe(kinds: Optional[Iterable[str]] = None) -> str:
    """One line per source: what it covers and leaves out. For prompts."""
    wanted = set(kinds) if kinds else None
    return "\n".join(f"- {s.line()}" for s in CATALOG if wanted is None or s.kind in wanted)


def outside_extent(tool: str, bbox: Sequence[float]) -> Optional[Source]:
    """The source *tool* reads, when *bbox* lies wholly outside its declared extent."""
    src = _BY_TOOL.get(tool)
    if src is None or src.bbox is None or not bbox or len(bbox) != 4:
        return None
    w, s, e, n = (float(v) for v in bbox)
    W, S, E, N = src.bbox
    if e < W or w > E or n < S or s > N:
        return src
    return None


def _statement(src: Source) -> str:
    """A source as the answer names it: what it is, what it leaves out, and its licence. What
    it leaves out is what decides how far a count or a list from it reaches (18 schools from
    the city's file, 31 in OpenStreetMap)."""
    return f"{src.name}, which leaves out {src.excludes} ({src.licence})"


def _payload(content: Any) -> Dict[str, Any]:
    if isinstance(content, dict):
        return content
    text = str(content or "")
    if text.startswith("content="):
        import ast

        quote = text[8]
        end = text.find(f"{quote} name=", 9)
        try:
            text = ast.literal_eval(text[8:end + 1] if end > 0 else text[8:])
        except Exception:  # noqa: BLE001
            return {}
    try:
        val = json.loads(text)
    except (TypeError, ValueError):
        return {}
    return val if isinstance(val, dict) else {}


def _upload_record(file_id: str) -> Dict[str, Any]:
    try:
        from agent_runtime.file_store import get_file_record

        return get_file_record(file_id) or {}
    except Exception:  # noqa: BLE001
        return {}


def _is_output(rec: Dict[str, Any]) -> bool:
    return rec.get("kind") == "output" or str(rec.get("source") or "") == "agent"


def _upload_name(file_id: str, fallback: Optional[str] = None) -> Optional[str]:
    """The name a user's file goes by; None for a file the agent wrote, which is not a source."""
    rec = _upload_record(file_id)
    if _is_output(rec):
        return None
    return rec.get("filename") or rec.get("original_name") or fallback


def _args(args: Any) -> Dict[str, Any]:
    """A call's arguments, which some peers record as a JSON or Python-literal string."""
    if isinstance(args, dict):
        return args
    if isinstance(args, str) and args.lstrip().startswith("{"):
        import ast

        for parse in (json.loads, ast.literal_eval):
            try:
                val = parse(args)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(val, dict):
                return val
    return {}


_FILE_ID_RE = re.compile(r"^file_[0-9a-f]{6,}$")


def _file_ids_in(node: Any, out: List[str], depth: int = 0) -> None:
    if depth > 4:
        return
    if isinstance(node, str):
        if _FILE_ID_RE.match(node) and node not in out:
            out.append(node)
    elif isinstance(node, dict):
        for v in node.values():
            _file_ids_in(v, out, depth + 1)
    elif isinstance(node, (list, tuple)):
        for v in node:
            _file_ids_in(v, out, depth + 1)


def source_of(tool: str, args: Any, content: Any) -> List[Tuple[str, List[str]]]:
    """[(statement, names an answer could use for it)] for one tool result."""
    args = _args(args)
    out: List[Tuple[str, List[str]]] = []
    src = _BY_TOOL.get(tool)
    if src is not None:
        out.append((_statement(src), [src.name, *src.aliases]))
    payload = _payload(content)
    urls = [str(v) for k, v in {**args, **payload}.items()
            if k in ("url", "source", "origin") and isinstance(v, str) and v.startswith("http")]
    for url in urls:
        host = urlparse(url).netloc
        hs = _BY_HOST.get(host)
        if hs is not None:
            out.append((_statement(hs), [hs.name, *hs.aliases, host]))
        else:
            out.append((url, [host]))
    # Every file id the call was given, under whatever argument name the tool chose (`file_id`,
    # `raster_file_id`, `input_files`, and `path`, which the file inspector takes an id in).
    ids: List[str] = []
    _file_ids_in(args, ids)
    # A code run's result names the files it was given, with their filenames.
    staged = {str(f["file_id"]): f.get("filename") for f in payload.get("input_files") or []
              if isinstance(f, dict) and f.get("file_id")}
    if payload.get("file_id") in ids and payload.get("filename"):
        staged.setdefault(str(payload["file_id"]), payload["filename"])
    for fid in [*ids, *(i for i in staged if i not in ids)]:
        name = _upload_name(fid, staged.get(fid))
        if name:
            out.append((f"{name} (your upload)", [name]))
    # Data the code fetched itself through the tool bridge (agent_runtime/tool_bridge.py): the
    # run's figures came from those calls, so a figure linked to the run names their sources.
    for bc in payload.get("bridge_calls") or []:
        if isinstance(bc, dict) and bc.get("ok") and bc.get("name"):
            res = bc.get("result") if isinstance(bc.get("result"), dict) else {}
            for item in source_of(str(bc["name"]), bc.get("args"), res):
                if item not in out:
                    out.append(item)
    return out


def sources_of(pairs: Iterable[Tuple[Dict[str, Any], Dict[str, Any]]],
               used_call_ids: Optional[set] = None) -> List[Tuple[str, List[str]]]:
    """The distinct sources behind the (call, result) pairs, restricted to *used_call_ids* when
    given. A failed result is not a source."""
    seen: Dict[str, Tuple[str, List[str]]] = {}
    for call, res in pairs:
        cid = res.get("tool_call_id")
        if used_call_ids is not None and cid not in used_call_ids:
            continue
        # With no figure linking the answer to a result, only DATA sources count: a web search
        # or a knowledge-base search that found a dataset is how it was found, not what the
        # answer's figures came from (the answer cites evidence documents itself).
        src = _BY_TOOL.get(str(res.get("name") or ""))
        if used_call_ids is None and src is not None and src.kind in ("web", "literature"):
            continue
        payload = _payload(res.get("content"))
        if payload.get("ok") is False or payload.get("error"):
            continue
        for statement, names in source_of(str(res.get("name") or ""), call.get("args"),
                                          res.get("content")):
            seen.setdefault(statement, (statement, names))
    return list(seen.values())


def sources_line(answer: str, sources: List[Tuple[str, List[str]]]) -> Optional[str]:
    """The Sources line an answer gets: every source it used, named once. None when it used no
    data source. Always rendered, also when the answer already names one: what an answer says
    in prose depends on the model, and what this line says does not."""
    if not sources:
        return None
    return "**Sources:** " + "; ".join(s for s, _ in sources) + "."
