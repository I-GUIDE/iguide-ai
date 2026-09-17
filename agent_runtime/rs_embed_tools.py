"""Remote-sensing foundation-model embeddings as agent tools.

Proxies the rs-embed service (``examples/webapp/server.py`` in the rs-embed repo), which
owns the Earth Engine session, the model zoo and the heavy tensor work. These tools turn
its replies into the things this agent traffics in: file-store artifacts the user can
download, and ``map_layer`` descriptors that land on the interactive map.

Why a service instead of importing ``rs_embed`` here: it keeps torch / earthengine-api /
geemap out of the agent environment, keeps Earth Engine credentials in one place, and
reuses code that is tested in its own repo. Point ``RS_EMBED_URL`` at the service.

Georeferencing note. rs-embed's ``PointBuffer(buffer_m=N)`` footprint is a +/-N metre
square in **EPSG:3857**, not a geodesic square (Web Mercator metres run 1/cos(latitude)
long). Every geometry is therefore converted to an explicit bbox HERE, in 3857, and sent
as a bbox — so the raster we drape is bounded by exactly the region that was embedded
rather than by a reconstruction of it.
"""
from __future__ import annotations

import base64
import calendar
import hashlib
import itertools
import json
import logging
import os
import re
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from agent_runtime.tool_args import accept_null_defaults

logger = logging.getLogger(__name__)

RS_EMBED_URL = os.getenv("RS_EMBED_URL", "http://localhost:8077").rstrip("/")
# On-the-fly models download checkpoints on first use and run a transformer; the
# precomputed backends answer in seconds. Long, but bounded — a hung request is worse
# than a slow one because the turn never ends.
_TIMEOUT_S = float(os.getenv("RS_EMBED_TIMEOUT_S", "600"))
_DEFAULT_BUFFER_M = 2048          # matches the service's own point footprint
_MAX_MODELS_PER_CALL = 5
# How many saved packages list_embedding_packages will describe. Each one costs opening its .npz
# to read the manifest, and a listing long enough to need scrolling is not an answer anyway.
_PACKAGE_LIST_MAX = 15
# An identical embed_zones call, repeated inside one turn, replayed instead of re-swept. The
# model does this — the reported Champaign/Urbana turn swept one city twice — and a sweep is the
# most expensive thing here: one request to the imagery provider per tile, minutes of wall clock,
# and now that a named area is no longer capped, a duplicate costs a FULL second sweep rather
# than a cheap partial one. Keyed on every argument that determines the result, so a legitimate
# repeat (the same polygon for a different year, which is exactly the Change workflow) still runs.
_ZONE_MEMO: "OrderedDict[str, Tuple[float, str]]" = OrderedDict()
_ZONE_MEMO_TTL_S = float(os.getenv("RS_EMBED_ZONE_MEMO_TTL_S", "1800"))
_ZONE_MEMO_MAX = 32
_ZONE_SCOPE_SEQ = itertools.count(1)


def _as_list(value: Any) -> Optional[List[str]]:
    """One name or several, however the model wrote it.

    A parameter typed List[str] is rejected by pydantic BEFORE the function runs when the model
    passes a bare string — observed live as `ValidationError: models Input should be a valid
    list [input_value='gse', input_type=str]`, a whole wasted round trip for a request that was
    perfectly clear. Writing models="gse" for one model is the natural thing to write, so the
    signatures accept it and this normalises it. Comma-separated too: it is what a model reaches
    for next, and splitting it here is cheaper than another rejection.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value)]


def _iso_date(value: Optional[str], *, month_end: bool = False) -> Optional[str]:
    """Accept a month or a full date, return the ISO date the zones service demands.

    embed_region documents its window as MONTHS ("YYYY-MM") and the zones service wants full
    ISO dates, so a model that had just read embed_region passed "2022-06" here and got
    `SpecError: TemporalSpec.range expects ISO dates 'YYYY-MM-DD'`. It then retried with
    "2022-06-01" and succeeded — which is why one request produced two embed_zones calls, and
    why the first looked like a duplicate sweep rather than the failure it was.

    Two tools over the same imagery should not disagree about what a date looks like. A month
    widens to the whole month: the START to its first day, the END to its last, so "2022-06" to
    "2022-09" means June through September inclusive rather than stopping on the 1st.
    """
    text = str(value or "").strip()
    if not text:
        return None
    if re.fullmatch(r"\d{4}-\d{2}", text):
        year, month = (int(part) for part in text.split("-"))
        day = calendar.monthrange(year, month)[1] if month_end else 1
        return f"{year:04d}-{month:02d}-{day:02d}"
    return text


def _zone_memo_scope() -> Optional[str]:
    """A token for the TURN in flight, or None when there is no turn.

    Scoped to the streaming trace state, which is a ContextVar set once per request — so a
    replay can only ever serve the call it duplicates, never a later conversation. Process-global
    would have been wrong twice over: it would replay a result into a turn that never asked for
    it, and it silently coupled two tests in test_rs_embed_zonal that call this tool with the
    same arguments and different stubbed responses.

    None means no memo at all: without a trace state there is no turn to be inside — a CLI run,
    an eval, a unit test — and the duplicate this exists to stop happens inside one.
    """
    try:
        from agent_runtime.streaming_trace import _TRACE_STATE

        state = _TRACE_STATE.get()
        if state is None:
            return None
        # NOT id(): CPython reuses an address once the object is freed, so a later turn whose
        # state landed on a freed one's address would read the earlier turn's results — the
        # exact cross-turn replay this scoping exists to prevent, and a test caught it doing so.
        #
        # Stamped onto the state. A WeakKeyDictionary would be tidier — no mutation of another
        # module's object — but _TraceState is a plain @dataclass, so it generates __eq__ and is
        # therefore UNHASHABLE and cannot be a weak key. Holding a strong reference instead, to
        # keep an id alive, would pin a turn's sink and handler in memory for the whole TTL.
        #
        # This needs _TraceState to accept attributes, which it does today and would not under
        # `@dataclass(slots=True)`. test_the_scope_resolves_for_a_real_trace_state fails loudly
        # if that changes, rather than letting the memo quietly stop working.
        token = getattr(state, "_rs_zone_memo_scope", None)
        if token is None:
            token = f"turn{next(_ZONE_SCOPE_SEQ)}"
            setattr(state, "_rs_zone_memo_scope", token)
        return str(token)
    except Exception:  # noqa: BLE001 - the memo must never be the thing that breaks a call
        return None


def _zone_memo_key(**call: Any) -> str:
    """Every argument that determines the result, order-normalised so a reordered list of
    zone_ids is the same question rather than a new one."""
    norm = dict(call)
    for field in ("zone_ids", "sibling_file_ids"):
        if field in norm:
            norm[field] = sorted(str(v) for v in (norm[field] or []))
    blob = json.dumps(norm, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _zone_memo_get(key: str, *, now: Optional[float] = None) -> Optional[str]:
    """The stored result for an identical call in THIS turn. Expired entries drop on read."""
    scope = _zone_memo_scope()
    if scope is None:
        return None
    stamp = time.time() if now is None else now
    for k in [k for k, (ts, _) in _ZONE_MEMO.items() if stamp - ts > _ZONE_MEMO_TTL_S]:
        _ZONE_MEMO.pop(k, None)
    hit = _ZONE_MEMO.get(f"{scope}:{key}")
    if hit is None:
        return None
    _ZONE_MEMO.move_to_end(f"{scope}:{key}")
    return hit[1]


def _zone_memo_put(key: str, result: str, *, now: Optional[float] = None) -> None:
    """Remember a SUCCESSFUL result. A failure is not cached: the next call should retry it —
    a wedged service or an expired credential is exactly the case where attempt two works."""
    scope = _zone_memo_scope()
    if scope is None:
        return
    _ZONE_MEMO[f"{scope}:{key}"] = (time.time() if now is None else now, result)
    _ZONE_MEMO.move_to_end(f"{scope}:{key}")
    while len(_ZONE_MEMO) > _ZONE_MEMO_MAX:
        _ZONE_MEMO.popitem(last=False)


def _svc(path: str, payload: Optional[Dict[str, Any]] = None, *, method: str = "POST",
         timeout: Optional[float] = None) -> Dict[str, Any]:
    """Call the rs-embed service, or return an error dict that says what to do."""
    import requests

    url = f"{RS_EMBED_URL}{path}"
    timeout = float(timeout or _TIMEOUT_S)
    try:
        r = (requests.get(url, timeout=timeout) if method == "GET"
             else requests.post(url, json=payload or {}, timeout=timeout))
    except requests.exceptions.ConnectionError:
        return {"error": f"the rs-embed service is not reachable at {RS_EMBED_URL}",
                "hint": "Start it with: python -m uvicorn server:app --app-dir examples/webapp "
                        "--port 8077 (from the rs-embed repo), or set RS_EMBED_URL to where it runs. "
                        "Without it no embedding tool can run — say so rather than inventing values."}
    except requests.exceptions.Timeout:
        return {"error": f"the rs-embed service did not answer within {timeout:.0f}s",
                "hint": "On-the-fly models download checkpoints on first use. Retry, use a smaller "
                        "region, or use a precomputed model (gse / tessera / copernicus)."}
    if r.status_code >= 400:
        detail = ""
        try:
            detail = str(r.json().get("error") or "")[:400]
        except Exception:  # noqa: BLE001
            detail = r.text[:400]
        return {"error": f"rs-embed service returned HTTP {r.status_code}", "detail": detail}
    try:
        return r.json()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"rs-embed service returned unparseable output: {exc}"}


def _svc_upload(path: str, file_path: Any, *, field: str = "file",
                timeout: Optional[float] = None) -> Dict[str, Any]:
    """POST a file to the rs-embed service as multipart/form-data.

    The same failure branches as :func:`_svc`, worded the same: a caller cannot tell from the
    result which helper produced the error, and "the service is down, do not invent a value"
    has to read identically either way.
    """
    import requests

    url = f"{RS_EMBED_URL}{path}"
    timeout = float(timeout or _TIMEOUT_S)
    try:
        with open(file_path, "rb") as fh:
            r = requests.post(url, files={field: (Path(str(file_path)).name, fh,
                                                  "application/octet-stream")},
                              timeout=timeout)
    except FileNotFoundError:
        return {"error": f"the file to upload is gone: {file_path}"}
    except requests.exceptions.ConnectionError:
        return {"error": f"the rs-embed service is not reachable at {RS_EMBED_URL}",
                "hint": "Start it with: python -m uvicorn server:app --app-dir examples/webapp "
                        "--port 8077 (from the rs-embed repo), or set RS_EMBED_URL to where it runs. "
                        "Without it no embedding tool can run — say so rather than inventing values."}
    except requests.exceptions.Timeout:
        return {"error": f"the rs-embed service did not answer within {timeout:.0f}s",
                "hint": "A pooled-vector upload is a few kilobytes and normally answers at once, "
                        "so a timeout here means the service is wedged rather than busy."}
    if r.status_code >= 400:
        detail = ""
        try:
            body = r.json()
            detail = str(body.get("error") or body.get("detail") or "")[:400]
        except Exception:  # noqa: BLE001
            detail = r.text[:400]
        return {"error": f"rs-embed service returned HTTP {r.status_code}", "detail": detail}
    try:
        return r.json()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"rs-embed service returned unparseable output: {exc}"}


# --- geometry ------------------------------------------------------------------
def _mercator_square(lon: float, lat: float, buffer_m: float) -> List[float]:
    """The +/-buffer_m square around (lon, lat) measured in EPSG:3857, as a lon/lat bbox."""
    from pyproj import Transformer

    fwd = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    inv = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    x, y = fwd.transform(lon, lat)
    minlon, minlat = inv.transform(x - buffer_m, y - buffer_m)
    maxlon, maxlat = inv.transform(x + buffer_m, y + buffer_m)
    return [round(minlon, 6), round(minlat, 6), round(maxlon, 6), round(maxlat, 6)]


def _vector_extent(file_id: str) -> Optional[Dict[str, Any]]:
    """WGS84 bbox of an uploaded vector dataset, plus what its geometry IS.

    The geometry kind decides whether a bounding box is a fair reading of "this area": a
    point cloud has no shape to respect, a polygon layer does.
    """
    try:
        from agent_runtime.langchain_geo_tools import _resolve, read_vector

        path, _rec = _resolve(file_id)
        gdf = read_vector(path)
        if getattr(gdf, "crs", None) is not None:
            gdf = gdf.to_crs("EPSG:4326")
        b = [float(v) for v in gdf.total_bounds]
        if not all(v == v for v in b):
            return None
        kinds = {str(k) for k in gdf.geometry.geom_type.unique()}
        shaped = bool(kinds & {"Polygon", "MultiPolygon", "LineString", "MultiLineString"})
        fill = None
        if shaped:
            try:
                m = gdf.to_crs("EPSG:3857")
                total = float(m.geometry.area.sum())
                bx = m.total_bounds
                env = float(bx[2] - bx[0]) * float(bx[3] - bx[1])
                fill = round(total / env, 4) if env > 0 and total > 0 else None
            except Exception:  # noqa: BLE001
                fill = None
        return {"bounds": [round(v, 6) for v in b], "geom_kinds": sorted(kinds),
                "features": int(len(gdf)), "shaped": shaped, "fill_fraction": fill}
    except Exception:  # noqa: BLE001
        return None


def _bounds_of_file(file_id: str) -> Optional[List[float]]:
    """WGS84 bbox of an uploaded vector dataset, or None."""
    try:
        from agent_runtime.langchain_geo_tools import _resolve, read_vector

        path, _rec = _resolve(file_id)
        gdf = read_vector(path)
        if getattr(gdf, "crs", None) is not None:
            gdf = gdf.to_crs("EPSG:4326")
        b = [float(v) for v in gdf.total_bounds]
        return [round(v, 6) for v in b] if all(v == v for v in b) else None
    except Exception:  # noqa: BLE001
        return None


def _resolve_bbox(bbox: Optional[List[float]], lon: Optional[float], lat: Optional[float],
                  file_id: Optional[str], buffer_m: float,
                  polygon_extent_ok: bool = True) -> Any:
    """Return ``[minlon, minlat, maxlon, maxlat]`` or an error dict naming the options."""
    if bbox:
        vals = [float(v) for v in bbox]
        if len(vals) != 4:
            return {"error": f"bbox needs 4 numbers [minlon, minlat, maxlon, maxlat]; got {len(vals)}"}
        minlon, minlat, maxlon, maxlat = vals
        if minlon >= maxlon or minlat >= maxlat:
            return {"error": "bbox is empty or inverted",
                    "detail": f"got [{minlon}, {minlat}, {maxlon}, {maxlat}]; "
                              f"expected minlon < maxlon and minlat < maxlat"}
        return [round(v, 6) for v in vals]
    if lon is not None and lat is not None:
        return _mercator_square(float(lon), float(lat), buffer_m)
    if file_id:
        info = _vector_extent(file_id)
        if info and info["shaped"] and not polygon_extent_ok:
            # Observed: asked for "the embeddings for the area with geoid 17031330100", the
            # agent called BOTH embed_zones and this tool, and reported this one — because a
            # bbox tool that accepts a file_id looks like the direct answer. It embedded the
            # tract's bounding box: the tract fills 69% of it, so 46% of what was embedded was
            # outside the tract, most of it Lake Michigan. Name the tool that keeps the shape.
            pct = f"{info['fill_fraction'] * 100:.0f}%" if info.get("fill_fraction") else "part"
            return {"error": f"{file_id} contains {info['features']} "
                             f"{'/'.join(info['geom_kinds'])} feature(s), and this tool embeds a "
                             f"RECTANGLE — the shapes cover only {pct} of their bounding box, so "
                             f"the rest of the rectangle would be embedded too.",
                    "use_instead": "embed_zones",
                    "hint": "Use embed_zones(file_id=..., zone_id_field=...): it averages only "
                            "the pixels inside each shape AND returns a pixel-level PCA picture "
                            "cut to the same shape, so it answers both halves of the question. "
                            "Call this tool again with an explicit bbox=[minlon, minlat, maxlon, "
                            "maxlat] only when a rectangle is what you actually want — the image "
                            "will then be of that rectangle, wider than the shape, and the answer "
                            "should say so.",
                    "bounding_box": info["bounds"]}
        if info:
            return info["bounds"]
        return {"error": f"could not read a geographic extent from {file_id}",
                "hint": "Pass bbox=[minlon, minlat, maxlon, maxlat], or lon/lat for a point."}
    return {"error": "no region given",
            "hint": "Pass ONE of: bbox=[minlon, minlat, maxlon, maxlat] (what the map's Region "
                    "tool produces), lon+lat for a point, or file_id of an uploaded layer."}


def _geometry(bbox: List[float]) -> Dict[str, Any]:
    return {"type": "bbox", "minlon": bbox[0], "minlat": bbox[1],
            "maxlon": bbox[2], "maxlat": bbox[3]}


# --- artifacts -----------------------------------------------------------------
def _save_png(data_uri: str, stem: str) -> Optional[Dict[str, Any]]:
    """Persist a ``data:image/png;base64,...`` payload into the file store."""
    from agent_runtime.file_store import create_output_file_from_path

    if not isinstance(data_uri, str) or "base64," not in data_uri:
        return None
    raw = base64.b64decode(data_uri.split("base64,", 1)[1])
    out = Path(tempfile.mkdtemp(prefix="rsembed_")) / f"{stem}.png"
    out.write_bytes(raw)
    return create_output_file_from_path(out, filename=out.name)


def _raster_layer(rec: Dict[str, Any], bbox: List[float], label: str,
                  layer_id: Optional[str] = None, opacity: float = 0.85,
                  embedding: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """A ``map_layer`` descriptor the client drapes as a georeferenced image.

    ``layer_id`` is the layer's identity and should always be given: without it the id falls
    back to a slug of ``label``, which is a display string and moves when the wording does.

    ``embedding`` points at the vectors the picture was made FROM — the saved package, and which
    model inside it this layer draws. The raster is a 3-colour projection and answers nothing on
    its own; every later question about the layer ("predict from it", "compare it with that
    one") needs the real vectors. Carrying the pointer on the layer is what lets a later turn
    reach them: the id lives only in the process-local ledger otherwise, and the visible answer
    carries just a filename.
    """
    out = {"url": rec.get("download_url"), "label": label, "render": "raster",
           "bounds": bbox, "opacity": opacity, "source": "analysis"}
    if layer_id:
        out["id"] = layer_id
    if embedding:
        out["embedding"] = embedding
    return out


def _fetch_package(service_path: str, stem: str) -> Optional[Dict[str, Any]]:
    """Copy the service's .npz export into the agent file store so the user can actually get it."""
    import requests

    from agent_runtime.file_store import create_output_file_from_path

    try:
        r = requests.get(f"{RS_EMBED_URL}{service_path}", timeout=_TIMEOUT_S)
        if r.status_code >= 400 or not r.content:
            return None
        out = Path(tempfile.mkdtemp(prefix="rsembed_")) / f"{stem}.npz"
        out.write_bytes(r.content)
        return create_output_file_from_path(out, filename=out.name)
    except Exception:  # noqa: BLE001
        return None


def _region_tag(name: Optional[str], bbox: Optional[Any] = None) -> str:
    """A short, stable token that tells one run's layers apart from another's.

    Layer identity on the client is derived from the LABEL: the map UI builds
    ``artifact-<slugified label>`` and its ``putLayer`` REPLACES any existing layer with a
    matching id. A label built from the model alone therefore made a second region silently
    overwrite the first — two ``embed_region`` calls in one turn left a single raster on the
    map, with no error and nothing in the result to say a layer had been dropped.

    The caller's ``name`` is the tag when there is one; it already distinguishes the .npz and
    PNG artifacts, so the layer now agrees with them. Otherwise the region's centre, rounded,
    which keeps re-embedding the SAME region a replacement (identical tag) while letting
    different regions coexist.
    """
    if name and str(name).strip():
        return str(name).strip()
    try:
        b = [float(v) for v in bbox]  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    if len(b) < 4:
        return ""
    return f"{(b[1] + b[3]) / 2:.3f},{(b[0] + b[2]) / 2:.3f}"


def _round_bbox(bbox: Any) -> Optional[List[float]]:
    """A bbox as a stable key. Float noise must not look like a different region."""
    try:
        return [round(float(v), 6) for v in bbox]
    except (TypeError, ValueError):
        return None


def _layer_id(kind: str, hint: Any = None, /, **content: Any) -> str:
    """A layer's identity: a digest of everything that decides what the layer SHOWS.

    Different contents are different layers, so every input that changes the pixels or the
    features goes in the digest — the region, the model, the period, the parameters, the
    inputs it was computed from — and nothing else does.

    In particular the caller's ``name`` is NOT in it. A name is a label: the model picks a
    different one for the same place between turns ("Downtown Champaign - GSE" one turn,
    "Champaign downtown 1km box" the next), and an id that moved with the wording would turn
    one layer into two on every re-run. Keeping it out is what makes a re-run of the same
    request REPLACE its own layer instead of stacking a copy — and what lets a layer be
    renamed without becoming a different layer.

    ``hint`` is legibility only, for logs and the DOM. It is itself content-derived — a
    rounded centre, a file id — so it cannot drift while the content stands still. Uniqueness
    never rests on it: two layers with the same hint are still told apart by the digest.
    """
    blob = json.dumps(content, sort_keys=True, default=str)
    digest = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:10]
    bits = ["embed", _slug(kind)]
    if hint is not None and str(hint).strip():
        bits.append(_slug(str(hint)))
    bits.append(digest)
    return "-".join(bits)


def _layer_label(base: str, tag: str) -> str:
    """The TAG LEADS, because the layer list clips names with an ellipsis.

    The panel renders a name in ~158px and truncates the rest, so a tag appended after the
    description is exactly the part thrown away: two regions both read "gse embedding (PCA-R…"
    and were indistinguishable in the list even though their ids differed and both layers were
    on the map. The part that VARIES has to sit where truncation cannot reach it.

    ``base`` alone when there is nothing to disambiguate, so one-run labels stay clean.

    A word the TAG already carries is dropped from the front of ``base``. The model names its
    own layers, and it names them descriptively — "Urbana city — gse — Jun–Sep 2022" — so
    prepending that to "gse pixel embedding in zones" produced "…— gse — Jun–Sep 2022 — gse
    pixel embedding in zones", saying gse twice in a name the panel then clips. Only the
    leading token is considered: "gse" repeated is noise, but a "zones" or "change" later in
    the description is load-bearing and stays.
    """
    if not tag:
        return base
    head, _, rest = base.partition(" ")
    if head and rest and head.lower() in tag.lower().split():
        base = rest
    return f"{tag} \u2014 {base}"


# Pixels the shared PCA basis is FITTED on. Projection is never subsampled; this only
# bounds the SVD, which is O(pixels) in memory and would otherwise grow with region count.
_PCA_FIT_MAX_PX = 2_000_000


def _slug(text: str) -> str:
    keep = [c if c.isalnum() else "_" for c in str(text).lower()]
    return "".join(keep).strip("_")[:40] or "region"


def _model_error(models: List[str], available: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Reject unknown model names by NAMING the real ones (a bare rejection dead-ends)."""
    ids = {str(m.get("id")) for m in available}
    bad = [m for m in models if m not in ids]
    if not bad:
        return None
    pre = sorted(str(m.get("id")) for m in available if m.get("type") == "precomputed")
    otf = sorted(str(m.get("id")) for m in available if m.get("type") != "precomputed")
    return {"ok": False, "error": f"unknown model(s): {', '.join(bad)}",
            "precomputed_models": pre, "onthefly_models": otf,
            "hint": "Precomputed models answer in seconds; on-the-fly models run a "
                    "foundation model and are much slower on first use."}


# --- tools ---------------------------------------------------------------------
def _safe_path(record: Dict[str, Any]) -> Optional[Any]:
    """The stored path for a file record, or None when it cannot be resolved."""
    from agent_runtime.file_store import resolve_file_id

    try:
        return resolve_file_id(str(record.get("file_id")))
    except Exception:  # noqa: BLE001
        return None


def _where_when(path: Optional[Any]) -> Optional[tuple]:
    """A package's (region, months) as a comparable key, or None when it will not read.

    Used to tell apart the two kinds of duplicate name: several exports of the SAME region and
    period, which are interchangeable, and several different PLACES that happen to share the
    default export filename, which are not.
    """
    if path is None:
        return None
    from agent_runtime import rs_embed_head_domain as domain

    where = domain.package_region(domain.read_manifest(path))
    if not where:
        return None
    box = where.get("region_bbox")
    return (tuple(round(float(v), 4) for v in box) if box else None, where.get("months"))


def _candidate(record: Dict[str, Any]) -> Dict[str, Any]:
    """One package as a choice the model can actually make: its id, name, region and months."""
    from agent_runtime import rs_embed_head_domain as domain

    out: Dict[str, Any] = {"file_id": record.get("file_id"), "filename": record.get("filename")}
    path = _safe_path(record)
    if path is not None:
        out.update(domain.package_region(domain.read_manifest(path)))
    return {k: v for k, v in out.items() if v not in (None, "", [], {})}


def make_rs_embed_tools(default_input_file_ids: Optional[List[str]] = None) -> List[Any]:
    """Build the remote-sensing embedding StructuredTools."""
    from langchain_core.tools import StructuredTool

    meta = {"category": "geo"}

    def list_embedding_models() -> str:
        """List the remote-sensing foundation models available for embedding a region.

        Returns each model's id and whether it is `precomputed` (an existing global product,
        answers in seconds) or `onthefly` (runs the model on imagery now — minutes, and it
        downloads checkpoints the first time). Call this when unsure which model to name.
        """
        res = _svc("/api/models", method="GET")
        if res.get("error"):
            return json.dumps(res)
        models = res.get("models") or []
        return json.dumps({
            "ok": True,
            "precomputed": [m["id"] for m in models if m.get("type") == "precomputed"],
            "onthefly": [m["id"] for m in models if m.get("type") != "precomputed"],
            "detail": models,
            "note": "Precomputed models are the interactive choice; on-the-fly models are "
                    "worth the wait when you need a specific sensor or architecture.",
        })

    def embed_region(bbox: Optional[List[float]] = None, lon: Optional[float] = None,
                     lat: Optional[float] = None, file_id: Optional[str] = None,
                     models: Optional[Union[str, List[str]]] = None, start: str = "2022-06",
                     end: str = "2022-09", buffer_m: float = _DEFAULT_BUFFER_M,
                     name: Optional[str] = None) -> str:
        """Embed a RECTANGLE with remote-sensing foundation models and PUT THE RESULT ON THE MAP.

        This is also the entry point for SEGMENTATION into look-alike zones, CHANGE DETECTION
        between periods, per-pixel similarity and land-cover-style clustering: there is no separate
        tool for those, and each is written in code over the embedding package this saves — see
        COMPOSING FROM THE PACKAGE below.

        NAMED US AREA? Do not use this. "the embedding of Urbana" / "of Champaign County"
        wants the administrative boundary, and this tool embeds a box around a point — it
        takes in everything outside the city limits along with it. Call
        admin_boundary(area=..., state=..., level='city'|'county') and then
        embed_zones(file_id=..., zone_id_field='GEOID', model=..., start=..., end=...), which
        embeds the pixels INSIDE the polygon and accepts the same date range. Use embed_region
        for a bbox, a point with a buffer, or an uploaded file's extent.

        Each model returns a learned description of what the place looks like from space over
        the given months. The embedding grid is projected to 3 colours (PCA) and draped over
        the region as an interactive raster layer, so similar-looking ground reads as similar
        colour. Also saves a downloadable .npz holding the real vectors for reuse.

        Region: pass `bbox` [minlon, minlat, maxlon, maxlat] (what the map's Region tool gives),
        or `lon`+`lat` for a point (a `buffer_m` square around it). A `file_id` is accepted
        only for POINT layers; for polygons use embed_zones, which keeps their shape.
        `start`/`end` are months, "YYYY-MM".

        COMPOSING FROM THE PACKAGE — the second route, not the only one. segment_region,
        embedding_change and predict_for_region do the common cases in ONE call, server-side, on
        the NATIVE grid: prefer them when they fit, because the package exports a grid decimated
        to a cell budget (stride 2 at the default footprint), so clustering it in code clusters
        every second pixel. Compose when the tool cannot express what was asked — a k it does not
        take, a metric of your own, more than two periods, a per-pixel change surface — and say
        that the composed result is at the exported resolution.

        Some work has a tool and must not be written by hand. The trained heads live on the
        service and are never exported, so predict_from_package is the only route to them and
        list_prediction_heads says what has been trained. align_embedding_colors puts several
        regions on ONE shared colour basis; fitting a PCA per region and comparing the colours is
        the mistake it exists to prevent. list_embedding_packages finds a package saved in an
        earlier turn when the file_id is no longer to hand — it and predict_from_package both
        take a filename as well as an id.

        Stage the package into execute_code by passing `embedding_package.file_id` in
        `input_files` — a file_id an earlier TOOL produced works, not just an upload. Staging has
        a 200 MB budget across everything attached, and a package carrying a full grid can
        approach it, so stage the one package you need rather than several. Load it with numpy:

            grid__<model>    (D, H, W) float32 — the per-pixel embedding, north-up (row 0 = maxlat)
            pooled__<model>  (D,) float32      — one vector for the whole region
            meta             0-d ndarray of JSON — read it as json.loads(str(z["meta"])); plain
                             json.loads(z["meta"]) raises TypeError

        meta["models"] is a LIST, one entry per model, and grid_hw / grid_stride / grid_saved_hw
        are fields of that ENTRY, not of meta itself. A `grid_stride` above 1 means the saved grid
        is every Nth cell of the native grid_hw, so quote resolution from that entry's
        grid_saved_hw (equivalently grid__<model>.shape[1:]) and never from grid_hw.

        Then: k-means over the pixel axis for look-alike zones, 1 - cosine between two periods'
        pooled vectors for how much a place changed, per-pixel cosine for WHERE it changed.

        To deliver it, either polygonize and use add_map_layer — which gets a legend and clickable
        zones, and is the better answer when the classes matter — or save the array AS AN IMAGE and
        drape it with add_raster_layer, passing this call's `region_bbox` as bounds. For that
        second route the PNG must be the pixels themselves, one image pixel per grid cell:
        PIL.Image.fromarray(rgb).save(...) or plt.imsave(...). A matplotlib FIGURE is the wrong
        thing to drape — axes, margins, titles and a colorbar all become part of the layer, and
        the pixels no longer line up with the bounds, so the whole map is silently misregistered.

        Two things to say rather than let the map imply them: clusters are unlabelled, so the same
        number means nothing across separate runs, and a distance says THAT a place changed, not
        what changed.
        """
        box = _resolve_bbox(bbox, lon, lat, file_id, buffer_m, polygon_extent_ok=False)
        if isinstance(box, dict):
            return json.dumps({"ok": False, **box})

        avail = _svc("/api/models", method="GET")
        if avail.get("error"):
            return json.dumps({"ok": False, **avail})
        # _as_list, not a comprehension: a bare "gse" would iterate CHARACTERS and ask the
        # service for models g, s and e.
        chosen = (_as_list(models) or ["gse"])[:_MAX_MODELS_PER_CALL]
        bad = _model_error(chosen, avail.get("models") or [])
        if bad:
            return json.dumps(bad)

        res = _svc("/api/embed", {"geometry": _geometry(box), "start": start, "end": end,
                                  "models": chosen, "buffer_m": int(buffer_m)})
        if res.get("error"):
            return json.dumps({"ok": False, **res})

        layers, summaries, failed = [], [], []
        # Which model each entry of `layers` draws. The pointer can only be attached once the
        # package record exists, which is after this loop, and by then `rec` has been rebound.
        layer_models: List[str] = []
        region_tag = _region_tag(name, box)
        for r in res.get("results") or []:
            model = str(r.get("model"))
            if not r.get("ok"):
                failed.append({"model": model, "error": str(r.get("error"))[:300]})
                continue
            stem = f"{_slug(name or 'embedding')}_{model}_pca"
            rec = _save_png(str(r.get("image") or ""), stem)
            entry = {"model": model, "type": r.get("type"), "dim": r.get("dim"),
                     "grid": r.get("grid_hw"), "vector_norm": round(float(r.get("norm") or 0), 3)}
            # Omitted entirely on a service that does not send `meta`, rather than reported as
            # empty: "the run had no provenance" and "this deployment does not send it" are
            # different facts, and only one of them should look like a gap.
            prov = _provenance(r.get("meta"))
            if prov:
                entry["provenance"] = prov
            if rec:
                entry.update({"image_file_id": rec["file_id"], "download_url": rec.get("download_url")})
                layers.append(_raster_layer(
                    rec, box, _layer_label(f"{model} embedding (PCA-RGB)", region_tag),
                    _layer_id("pca", _region_tag(None, box), bbox=_round_bbox(box),
                              model=model, start=start, end=end)))
                layer_models.append(model)
            summaries.append(entry)

        pkg = res.get("package") or {}
        out: Dict[str, Any] = {
            "ok": bool(summaries), "region_bbox": box, "months": f"{start}..{end}",
            "models": summaries, "on_map": bool(layers),
            "compute": res.get("compute"),
            "note": "The colours are a 3-component PCA of the embedding, so they show which "
                    "areas resemble each other — they are NOT land-cover classes and the "
                    "colours are not comparable across separate runs.",
        }
        if failed:
            out["failed"] = failed
        if pkg:
            rec = _fetch_package(str(res.get("download_url") or ""),
                                 # NOT "embedding_vectors" for everything unnamed. That default
                                 # gave 32 of the 73 stored packages the same filename, so a
                                 # later turn naming one could not be answered — the ambiguity
                                 # refusal exists because of this line. Region, models and
                                 # period are all known here and make the name identify the
                                 # file, which a timestamp would not: unique is not the problem,
                                 # unidentifiable is.
                                 f"{_slug(name or region_tag or 'embedding')}"
                                 f"_{'-'.join(chosen)}_{start}_{end}_vectors")
            info: Dict[str, Any] = {"models_saved": pkg.get("models"),
                                    "pooled_vectors": True,
                                    "grids_saved": pkg.get("grids_saved") or []}
            if rec:
                info.update({"file_id": rec["file_id"], "filename": rec.get("filename"),
                             "download_url": rec.get("download_url"),
                             "size_bytes": rec.get("size_bytes")})
                # Each raster now says which vectors it came from and which model inside them
                # it draws, so "predict from that layer" is answerable in a later turn.
                for descriptor, layer_model in zip(layers, layer_models):
                    descriptor["embedding"] = {
                        "file_id": rec["file_id"], "filename": rec.get("filename"),
                        "model": layer_model, "months": f"{start}..{end}",
                        "models_in_package": pkg.get("models") or []}
            # The service DECIMATES an oversized grid rather than dropping it — the export cap
            # became a stride, recorded per entry as grid_stride — so a model missing from
            # grids_saved now means the export genuinely failed for it, not that the region was
            # too big. Say so: a
            # missing full-resolution grid is otherwise invisible until someone loads the file.
            else:
                # The package is the ONLY input to every composed operation — segmentation,
                # change, prediction — so losing it removes those capabilities for the turn.
                # Silence here read as "there is no package", and the model would go on to
                # describe clustering it could not do.
                info["unavailable"] = (
                    "the embedding ran, but its vector package could not be fetched from the "
                    "service, so there is no file to compose from")
                info["consequence"] = (
                    "clustering, change detection and prediction all need this file. Say the "
                    "map layer is here but the per-pixel work is not available this turn — do "
                    "not describe zones or distances you could not compute.")
            # A grid too large for the export budget is DECIMATED now, not dropped, so a model
            # missing from grids_saved means the export genuinely failed for it rather than that
            # the region was too big.
            dropped = [m["model"] for m in summaries
                       if m["model"] not in (pkg.get("grids_saved") or [])]
            if dropped:
                info["grid_missing_for"] = dropped
                info["why"] = ("no per-pixel grid came back for these models, so the file holds "
                               "their pooled vector only — enough for similarity, prediction and "
                               "comparison, not for per-pixel work. An oversized grid is "
                               "decimated rather than dropped (the manifest gives grid_stride), "
                               "so this is an export failure, not a size limit.")
            out["embedding_package"] = info
        # One descriptor per model; the client stacks them and the layer list toggles between.
        if layers:
            out["map_layer"] = layers[0]
            if len(layers) > 1:
                out["map_layers"] = layers
        return json.dumps(out)

    def segment_region(bbox: Optional[List[float]] = None, lon: Optional[float] = None,
                       lat: Optional[float] = None, file_id: Optional[str] = None,
                       k: int = 6, model: str = "gse", start: str = "2022-06",
                       end: str = "2022-09", buffer_m: float = _DEFAULT_BUFFER_M,
                       name: Optional[str] = None) -> str:
        """Segment a region into `k` look-alike zones from its embedding, ON THE MAP.

        Unsupervised land-cover-style segmentation: the embedding grid is clustered, so
        ground that looks alike from space gets the same colour. Returns the map layer plus
        a legend giving each cluster's share of the area. The clusters are discovered, not
        named — cluster 3 is not "forest" until someone looks.
        """
        box = _resolve_bbox(bbox, lon, lat, file_id, buffer_m)
        if isinstance(box, dict):
            return json.dumps({"ok": False, **box})
        if not 2 <= int(k) <= 10:
            return json.dumps({"ok": False, "error": f"k must be between 2 and 10; got {k}"})

        res = _svc("/api/segment", {"geometry": _geometry(box), "start": start, "end": end,
                                    "model": model, "k": int(k), "buffer_m": int(buffer_m)})
        if res.get("error"):
            return json.dumps({"ok": False, **res})
        rec = _save_png(str(res.get("image") or ""), f"{_slug(name or 'segments')}_{model}_k{k}")
        if not rec:
            return json.dumps({"ok": False, "error": "the service returned no segmentation image"})
        legend = [{"cluster": e.get("cluster"), "rgb": e.get("rgb"),
                   "share_pct": round(float(e.get("frac") or 0) * 100, 1)}
                  for e in (res.get("legend") or [])]
        return json.dumps({
            "ok": True, "region_bbox": box, "model": model, "k": int(k),
            "grid": res.get("grid_hw"), "legend": legend, "on_map": True,
            "image_file_id": rec["file_id"], "download_url": rec.get("download_url"),
            "map_layer": _raster_layer(
                rec, box, _layer_label(f"{model} segments (k={k})", _region_tag(name, box)),
                _layer_id("segments", _region_tag(None, box), bbox=_round_bbox(box),
                          model=model, k=int(k), start=start, end=end)),
            "note": "Clusters are unlabelled: they group similar-looking ground, and the "
                    "same number means nothing across separate runs.",
        })

    def embedding_change(bbox: Optional[List[float]] = None, lon: Optional[float] = None,
                         lat: Optional[float] = None, file_id: Optional[str] = None,
                         years: Optional[List[int]] = None, model: str = "gse",
                         buffer_m: float = _DEFAULT_BUFFER_M, name: Optional[str] = None) -> str:
        """Track how much a region CHANGED across years, from its embeddings.

        Embeds the region once per year and reports each year's distance from the baseline
        (the earliest year). A spike marks the year the place changed — new construction,
        clearing, flooding. Returns the per-year table as a CSV file plus the numbers.
        """
        box = _resolve_bbox(bbox, lon, lat, file_id, buffer_m)
        if isinstance(box, dict):
            return json.dumps({"ok": False, **box})
        yrs = sorted({int(y) for y in (years or [])})
        if len(yrs) < 2:
            return json.dumps({"ok": False, "error": "give at least two years",
                               "hint": "e.g. years=[2018, 2020, 2022, 2024]"})

        res = _svc("/api/change", {"geometry": _geometry(box), "years": yrs, "model": model,
                                   "buffer_m": int(buffer_m), "start": "2022-06", "end": "2022-09"})
        if res.get("error"):
            return json.dumps({"ok": False, **res})
        used = [int(y) for y in (res.get("years") or [])]
        dist = [float(d) for d in (res.get("distances") or [])]
        rows = list(zip(used, dist, strict=False))

        from agent_runtime.file_store import create_output_file_from_path

        out = Path(tempfile.mkdtemp(prefix="rsembed_")) / f"{_slug(name or 'change')}_{model}.csv"
        out.write_text("year,distance_from_baseline\n"
                       + "".join(f"{y},{d:.6f}\n" for y, d in rows), encoding="utf-8")
        rec = create_output_file_from_path(out, filename=out.name)
        peak = max(rows, key=lambda t: t[1]) if rows else None
        return json.dumps({
            "ok": True, "region_bbox": box, "model": model,
            "baseline_year": res.get("baseline"), "years": used,
            "distances": [round(d, 4) for d in dist],
            "largest_change_year": peak[0] if peak else None,
            "largest_change_distance": round(peak[1], 4) if peak else None,
            "csv_file_id": rec["file_id"], "download_url": rec.get("download_url"),
            "errors": res.get("errors") or [],
            "note": "Distance is 1 - cosine against the baseline year: 0 means indistinguishable. "
                    "It says THAT the place changed, not what changed.",
        })

    def compare_regions(bbox_a: List[float], bbox_b: List[float], model: str = "gse",
                        start: str = "2022-06", end: str = "2022-09") -> str:
        """Score how alike TWO regions look from space, using their embeddings.

        Returns cosine similarity (1.0 = indistinguishable) between the two regions'
        pooled embeddings. This is the retrieval primitive behind "find me somewhere
        that looks like this". Each bbox is [minlon, minlat, maxlon, maxlat].
        """
        boxes = []
        for label, raw in (("A", bbox_a), ("B", bbox_b)):
            box = _resolve_bbox(raw, None, None, None, _DEFAULT_BUFFER_M)
            if isinstance(box, dict):
                return json.dumps({"ok": False, "region": label, **box})
            boxes.append(box)
        res = _svc("/api/similarity", {"geometries": [_geometry(b) for b in boxes],
                                       "model": model, "start": start, "end": end,
                                       "buffer_m": int(_DEFAULT_BUFFER_M)})
        if res.get("error"):
            return json.dumps({"ok": False, **res})
        cos = res.get("cosine")
        return json.dumps({"ok": True, "model": model, "region_a_bbox": boxes[0],
                           "region_b_bbox": boxes[1], "cosine_similarity": cos,
                           "distance": res.get("distance"), "dim": res.get("dim"),
                           "reading": ("nearly identical" if isinstance(cos, (int, float)) and cos >= 0.95
                                       else "similar" if isinstance(cos, (int, float)) and cos >= 0.8
                                       else "different")})

    def list_prediction_heads() -> str:
        """List the pretrained downstream models that turn an embedding into a prediction."""
        res = _svc("/api/heads", method="GET")
        return json.dumps(res if res.get("error") else {"ok": True, **res})

    def predict_for_region(bbox: Optional[List[float]] = None, lon: Optional[float] = None,
                           lat: Optional[float] = None, file_id: Optional[str] = None,
                           models: Optional[Union[str, List[str]]] = None, start: str = "2022-06",
                           end: str = "2022-09", buffer_m: float = _DEFAULT_BUFFER_M) -> str:
        """Run a pretrained downstream head on a region's embedding to PREDICT a value.

        This is the "use the embedding" step: the region is embedded, then an already-trained
        head turns that vector into an estimate (e.g. crop presence). Call
        list_prediction_heads first to see what has been trained and how well it scored.
        """
        box = _resolve_bbox(bbox, lon, lat, file_id, buffer_m)
        if isinstance(box, dict):
            return json.dumps({"ok": False, **box})
        res = _svc("/api/predict", {"geometry": _geometry(box), "start": start, "end": end,
                                    "models": _as_list(models) or [],
                                    "buffer_m": int(buffer_m)})
        if res.get("error"):
            return json.dumps({"ok": False, **res})
        return json.dumps({"ok": True, "region_bbox": box, **res,
                           "note": "Each prediction carries the head's own validation score — "
                                   "quote it, because a confident number from a weak head is "
                                   "still a weak number."})

    def list_embedding_packages() -> str:
        """List the embedding packages already SAVED, newest first, with region and months.

        Every embed_region call saves the real vectors as an .npz. This says which ones exist,
        for which region and months, and which models each holds — so an embedding made in an
        earlier turn can be reused instead of paying for the region again. Use it whenever the
        user refers to an embedding, a layer or a region worked on earlier and the file_id is not
        to hand; then pass that file_id to predict_from_package or align_embedding_colors.

        SCOPE: this conversation's packages PLUS the deployment's shared ones — every package
        saved before files were attributed to a conversation, which is most of them. Another
        conversation's new package is not listed; a shared older one is. So match on the region
        and months rather than on the filename — the default export name was reused for every
        unnamed region, so one name covers many different places — and do not describe a package
        as "yours" or "the one from earlier" on the strength of its name alone. For what THIS
        conversation actually made, list_conversation_files is the authority.

        `has_head` says whether a pretrained head exists for that model, so a package that
        cannot be predicted from is visible as such before anything is attempted.
        """
        from agent_runtime import rs_embed_head_domain as domain
        from agent_runtime.file_store import find_files, resolve_file_id

        records = find_files(suffix=".npz", limit=_PACKAGE_LIST_MAX)
        if not records:
            return json.dumps({
                "ok": True, "packages": [], "count": 0,
                "note": "No embedding package has been saved. embed_region saves one each time "
                        "it runs; until then there is nothing to reuse."})

        # Best effort: the listing is still useful when the service is down, and saying "no head"
        # because the service was unreachable would be a false negative about the data.
        heads = _svc("/api/heads", method="GET")
        with_head = ({str(h.get("model")) for h in (heads.get("models") or [])}
                     if not heads.get("error") else None)

        packages = []
        for rec in records:
            try:
                path = resolve_file_id(str(rec.get("file_id")))
                vectors, _problems = domain.pooled_vectors(path)
                manifest = domain.read_manifest(path)
            except Exception:  # noqa: BLE001 - an unreadable package is listed, not fatal
                vectors, manifest = {}, {}
            entry = {"file_id": rec.get("file_id"), "filename": rec.get("filename"),
                     "models": sorted(vectors) or None,
                     "size_bytes": rec.get("size_bytes")}
            entry.update(domain.package_region(manifest))
            if with_head is not None and vectors:
                entry["has_head"] = sorted(set(vectors) & with_head) or []
            packages.append({k: v for k, v in entry.items() if v not in (None, "")})

        return json.dumps({
            "ok": True, "count": len(packages), "packages": packages,
            "showing": f"the {len(packages)} most recently saved"
                       if len(records) >= _PACKAGE_LIST_MAX else "all of them",
            "note": "Pass a file_id from here to predict_from_package to score an embedding "
                    "already paid for, or to align_embedding_colors to make two regions' "
                    "colours comparable. Neither re-embeds.",
        })

    def predict_from_package(file_id: str,
                             models: Optional[Union[str, List[str]]] = None) -> str:
        """Run the pretrained heads on an embedding you ALREADY have — no re-embedding.

        `file_id` takes EITHER the `embedding_package.file_id` from an embed_region result — from
        this turn or any earlier one — OR the package's filename, which is what an earlier answer
        surfaced if the id is no longer to hand ("downtown_champaign_1_km_box_vectors.npz", or
        just "champaign"). A layer on the map carries its own pointer in `embedding.file_id`, so
        "predict from that layer" resolves through the same argument. When a name matches more
        than one package the newest is used and the result lists the others under `also_matched`
        — say which one you used. list_embedding_packages shows what is there.

        The heads are trained weights that live on the service and are never exported, so this is
        the only route to them from here, but the EMBEDDING is already paid for: this fetches no
        imagery, spends no Earth Engine quota, and answers in milliseconds instead of minutes.
        Use it whenever a region has already been embedded — including when embedding is failing,
        since this path does not need Earth Engine at all.

        It scores the package's own pooled vectors, which is the same grid-then-average recipe
        the heads were trained on. Call list_prediction_heads first: coverage is narrow, and the
        result says plainly which models in the package have no head. `models` narrows the run to
        particular models in the package — pass one layer's model to score just that layer.

        This is ONE estimate for the whole region, not a per-pixel map, and the heads were
        fitted on a single crop in one state in one year. The result carries an
        `outside_training_domain` list whenever the package sits outside that — read it out
        rather than quoting the probability alone.
        """
        from agent_runtime import rs_embed_head_domain as domain
        from agent_runtime.file_store import resolve_file_ref

        fid = str(file_id or "").strip()
        if not fid:
            return json.dumps({"ok": False, "error": "no file_id or filename given",
                               "hint": "Pass embedding_package.file_id from an embed_region "
                                       "result, a layer's embedding.file_id, or the package's "
                                       "filename. list_embedding_packages shows what exists."})
        try:
            path, record, alternatives = resolve_file_ref(fid, suffix=".npz")
        except Exception as exc:  # noqa: BLE001
            return json.dumps({
                "ok": False, "error": f"cannot resolve {fid!r}: {type(exc).__name__}: {exc}"[:300],
                "hint": "Accepts a file_id, a layer's embedding.file_id, or the package's "
                        "filename. Call list_embedding_packages to see the saved packages with "
                        "their regions and months, then pass one of those file_ids."})

        if alternatives:
            chosen_where = _where_when(path)
            differing = [a for a in alternatives
                         if _where_when(_safe_path(a)) not in (None, chosen_where)]
            if differing:
                return json.dumps({
                    "ok": False,
                    "error": f"{len(alternatives) + 1} saved packages match {fid!r} and they "
                             "cover different regions or months, so which one to score is not "
                             "decided by the name",
                    "candidates": [_candidate(rec_a) for rec_a
                                   in [record, *alternatives][:8]],
                    "hint": "Pass the file_id of the one you mean. A layer on the map carries "
                            "its own in `embedding.file_id`; list_embedding_packages shows each "
                            "package's region and months. The default export name is reused for "
                            "every unnamed region, so a name often matches many places."})

        if not str(path).lower().endswith(".npz"):
            # A CSV is almost always embed_zones output, and the steer for it is a different
            # tool rather than a different file: these heads were fitted on ~2.6 km squares,
            # not on polygons, so zone rows should not be sent to them at all.
            zones = str(path).lower().endswith(".csv")
            return json.dumps({
                "ok": False,
                "error": f"{Path(str(path)).name} is not an embedding package (.npz)",
                "hint": ("This looks like the per-zone vector CSV from embed_zones. These heads "
                         "were fitted on ~2.6 km squares rather than on polygons, so zone rows "
                         "should not be scored by them — to predict a per-zone value, use "
                         "fit_zone_model with your own labels."
                         if zones else
                         "Pass embedding_package.file_id from an embed_region result — not the "
                         "PNG, and not the map layer.")})

        vectors, problems = domain.pooled_vectors(path)
        if not vectors:
            return json.dumps({
                "ok": False,
                "error": f"{Path(str(path)).name} holds no pooled embedding vectors",
                "problems": problems or None,
                "hint": "embed_region saves a package of pooled__<model> arrays — pass its "
                        "embedding_package.file_id. embed_zones does NOT write one: it writes a "
                        "CSV of per-zone vectors, and these heads were fitted on ~2.6 km squares "
                        "rather than on polygons, so they should not be applied to zone rows. To "
                        "predict a per-zone value, use fit_zone_model with your own labels."})

        heads = _svc("/api/heads", method="GET")
        if heads.get("error"):
            return json.dumps({"ok": False, **heads})
        # Every head's own width, so a right-width wrong-model vector is refused HERE rather
        # than being scored on the far side of an upload. Intersecting first also means a
        # package for a model with no head never travels at all.
        head_dim = {str(h.get("model")): h.get("dim") for h in (heads.get("models") or [])}
        head_score = {str(h.get("model")): h for h in (heads.get("models") or [])}

        wanted = {m.lower() for m in (_as_list(models) or [])}
        if wanted:
            missing = sorted(wanted - {m.lower() for m in vectors})
            vectors = {m: v for m, v in vectors.items() if m.lower() in wanted}
            if not vectors:
                return json.dumps({
                    "ok": False,
                    "error": f"this package holds no vectors for {', '.join(missing)}",
                    "hint": "Pass no `models` to score everything in the package, or one of the "
                            "models it does hold."})

        usable: List[str] = []
        not_scored: List[Dict[str, Any]] = []
        for model in sorted(vectors):
            if model not in head_dim:
                not_scored.append({"model": model, "why": "no pretrained head for this model"})
                continue
            refusal = domain.vector_refusal(model, vectors[model], head_dim.get(model))
            if refusal:
                entry: Dict[str, Any] = {"model": model, "why": refusal}
                # When the width is one several models share, say so: the reason this tool
                # trusts the package's key and not the array's shape is not self-evident.
                shared = domain.width_note(int(vectors[model].size))
                if shared:
                    entry["also"] = shared
                not_scored.append(entry)
                continue
            usable.append(model)

        if not usable:
            return json.dumps({
                "ok": False,
                "error": "nothing in this package can be scored by the available heads",
                "not_scored": not_scored,
                "heads_available": sorted(head_dim),
                "task": heads.get("task"), "label": heads.get("label"),
                "hint": "Embed the region with a model that has a head "
                        f"({', '.join(sorted(head_dim)) or 'none'}) and try again, or use "
                        "fit_zone_model to train a head on your own labels."})

        # Only the pooled keys go over the wire. The service reads the whole body into memory and
        # then uses nothing else, and packages in the store reach 216 MB because of their grids.
        packed = domain.repack_pooled(path, Path(tempfile.mkdtemp(prefix="rsembed_head_")) / "pooled.npz",
                                      usable)
        res = _svc_upload("/api/predict_package", packed["path"])
        if res.get("error"):
            return json.dumps({"ok": False, **res})

        manifest = domain.read_manifest(path)
        out: Dict[str, Any] = {"ok": True,
                               "package_file_id": str(record.get("file_id") or fid),
                               "package_filename": record.get("filename"),
                               "scored_models": usable}
        # A name can legitimately match several packages, and in practice it usually does: the
        # default export name is reused for every unnamed region, so one name can cover dozens
        # of different places. Duplicates of the SAME region and period are interchangeable and
        # picking the newest is harmless; candidates that differ in region or months are a
        # different question, and answering it by mtime would report a coin flip as a fact.
        # Those are refused before the upload, in _candidates_disagree above.
        if alternatives:
            out["also_matched"] = [_candidate(a) for a in alternatives[:5]]
            out["resolved_by"] = (f"{len(alternatives) + 1} packages match {fid!r} and all cover "
                                  "the same region and months; scored the newest")
        # Region and months come back onto the result because this tool takes no bbox argument:
        # without them neither the answer nor the action ledger records WHAT was predicted.
        out.update(domain.package_region(manifest))
        out.update({k: v for k, v in res.items() if v not in (None, "", [], {})})
        # /api/predict_package omits `region`, which is exactly the training-domain caveat.
        if not out.get("region") and heads.get("region"):
            out["region"] = heads["region"]
        if not_scored:
            out["not_scored"] = not_scored
        warnings = domain.domain_warnings(manifest)
        if warnings:
            out["outside_training_domain"] = warnings
        unverifiable = domain.unverifiable_domain(manifest)
        if unverifiable:
            out["domain_unverifiable"] = unverifiable
        if "dofa" in usable:
            out["pooling_note"] = (
                "The service's own re-embedding route puts dofa through the model's final norm "
                "layer on a square or point region, which is a different vector from the one "
                "saved in the package; the saved one is what the head was trained on. If a dofa "
                "number from that route is also in play, report the two as different recipes "
                "rather than reconciling them.")
        out["validation"] = {m: {k: head_score[m].get(k) for k in ("score", "score_name", "n")}
                             for m in usable if m in head_score}
        out["note"] = ("Scored the vectors already saved for this region, so no imagery was "
                       "fetched and no Earth Engine quota was spent. Quote each head's own "
                       "validation score with the prediction — a confident number from a weak "
                       "head is still a weak number — and read out any "
                       "`outside_training_domain` entry, because the heads cannot tell they are "
                       "being asked about another year or another place.")
        return json.dumps(out)

    def align_embedding_colors(file_ids: Union[str, List[str]], model: Optional[str] = None,
                               names: Optional[Union[str, List[str]]] = None) -> str:
        """Re-colour several already-embedded regions on ONE shared PCA basis, so the colours
        mean the same thing in every layer.

        Use this whenever two or more embedding rasters are being read against each other —
        "do these areas look alike?", "why are these maps different colours?", "put them side
        by side". Each embed_region call fits its OWN PCA and its OWN contrast stretch, so the
        same RGB in two layers encodes different directions in embedding space at different
        scales: each map is meaningful alone and none of them is comparable to another. This
        fits one basis and one stretch across all the regions at once and re-renders them,
        after which similar colour DOES mean similar ground across the layers.

        Pass the `file_id` of each region's embedding package — the .npz embed_region saves as
        `embedding_package.file_id`. They must all carry a grid for the same model; a package
        whose export genuinely failed holds only the pooled vector and has no pixels to
        re-colour, and this says so rather than quietly dropping it. A grid that was merely too
        LARGE is not that case: the service decimates it to a stride and still exports it.

        Costs nothing at the imagery provider: it reuses embeddings already paid for, so it is
        always cheaper than embedding the regions again, and it works on regions embedded in
        earlier turns. `names` labels the layers in order, one per file_id; without it a layer
        is named by its region's centre.
        """
        import numpy as np

        from agent_runtime.file_store import create_output_file_from_path, resolve_file_id

        ids = _as_list(file_ids) or []
        labels = _as_list(names) or []
        if len(ids) < 2:
            return json.dumps({
                "ok": False,
                "error": "a shared basis needs at least two embedding packages, got "
                         f"{len(ids)}",
                "hint": "Embed each region with embed_region and pass the "
                        "embedding_package.file_id of each."})

        loaded, problems = [], []
        for fid in ids:
            try:
                with np.load(resolve_file_id(fid), allow_pickle=True) as z:
                    keys = [k for k in z.files if k.startswith("grid__")]
                    entry = {"file_id": fid,
                             "grids": {k[len("grid__"):]: np.asarray(z[k], dtype=np.float32)
                                       for k in keys}}
                    try:
                        entry["meta"] = json.loads(str(z["meta"])) if "meta" in z.files else {}
                    except Exception:  # noqa: BLE001 - a package without readable meta is usable
                        entry["meta"] = {}
            except Exception as exc:  # noqa: BLE001
                problems.append({"file_id": fid, "error": f"{type(exc).__name__}: {exc}"[:200]})
                continue
            if not entry["grids"]:
                problems.append({"file_id": fid,
                                 "error": "this package holds the pooled vector only — its "
                                          "grid export failed, so there are no "
                                          "pixels to re-colour"})
                continue
            loaded.append(entry)

        if len(loaded) < 2:
            return json.dumps({"ok": False,
                               "error": "fewer than two packages could be read with a grid",
                               "packages_rejected": problems})

        shared = set(loaded[0]["grids"])
        for e in loaded[1:]:
            shared &= set(e["grids"])
        if model and model not in shared:
            return json.dumps({"ok": False,
                               "error": f"not every package holds a grid for {model!r}",
                               "models_in_common": sorted(shared),
                               "per_package": [{"file_id": e["file_id"],
                                                "models": sorted(e["grids"])} for e in loaded]})
        if not shared:
            return json.dumps({
                "ok": False,
                "error": "the packages share no model, so there is no common space to project",
                "hint": "A basis is only shared within one model — two models' embeddings are "
                        "different spaces and their colours were never comparable.",
                "per_package": [{"file_id": e["file_id"], "models": sorted(e["grids"])}
                                for e in loaded]})
        model = model or sorted(shared)[0]

        arrays = [np.nan_to_num(e["grids"][model], nan=0.0, posinf=0.0, neginf=0.0)
                  for e in loaded]
        dims = {int(a.shape[0]) for a in arrays}
        if len(dims) != 1:
            return json.dumps({"ok": False,
                               "error": f"the {model!r} grids disagree on dimensionality: "
                                        f"{sorted(dims)}"})

        # (pixels, dims) per region, then one basis over all of them at once.
        feats = [a.reshape(a.shape[0], -1).T.astype(np.float64) for a in arrays]
        stacked = np.concatenate(feats, axis=0)
        mu = stacked.mean(axis=0)
        # Fitting is subsampled by a deterministic stride on very large mosaics; the
        # PROJECTION always uses every pixel, so no region is rendered from a partial fit.
        step = max(1, int(np.ceil(stacked.shape[0] / _PCA_FIT_MAX_PX)))
        _u, sv, vt = np.linalg.svd(stacked[::step] - mu, full_matrices=False)
        comp = vt[:3]

        proj = [(f - mu) @ comp.T for f in feats]
        allp = np.concatenate(proj, axis=0)
        # Deterministic sign, mirroring the per-run renderer, so a rerun does not invert.
        signs = np.where(allp.sum(axis=0) < 0, -1.0, 1.0)
        proj = [p * signs for p in proj]
        allp = allp * signs
        # ONE stretch, over every region's pixels — this is what makes the colours comparable.
        lo = np.percentile(allp, 2.0, axis=0)
        hi = np.percentile(allp, 98.0, axis=0)

        from PIL import Image

        layers, regions = [], []
        # Two packages of the SAME region — one place at two periods, say — carry the same
        # bbox, so with no `names` both layers would take one tag and therefore one id, and
        # build_map_layers' per-call dedup would drop the second before it ever left the
        # process. Number the repeats instead.
        used_tags: Dict[str, int] = {}
        # The basis the PCA was ACTUALLY fitted on: packages drop out of `loaded` when they
        # cannot be read, hold no grid, or carry no bbox, and those that remain are what set
        # every layer's colours. Digesting the requested list instead would call two different
        # renderings the same layer.
        fitted_basis = sorted(str(e["file_id"]) for e in loaded)
        for idx, (entry, arr, p_) in enumerate(zip(loaded, arrays, proj)):
            h, w = int(arr.shape[1]), int(arr.shape[2])
            img = np.clip((p_ - lo) / (hi - lo + 1e-8), 0.0, 1.0).reshape(h, w, 3)
            geom = (entry["meta"].get("geometry") or {})
            try:
                bbox = [float(geom["minlon"]), float(geom["minlat"]),
                        float(geom["maxlon"]), float(geom["maxlat"])]
            except (KeyError, TypeError, ValueError):
                problems.append({"file_id": entry["file_id"],
                                 "error": "no bbox in the package meta, so the raster cannot "
                                          "be placed on the map"})
                continue
            tag = _region_tag(None, bbox)
            if labels and idx < len(labels) and str(labels[idx]).strip():
                tag = str(labels[idx]).strip()
            seen_before = used_tags.get(tag, 0)
            used_tags[tag] = seen_before + 1
            if seen_before:
                tag = f"{tag} ({seen_before + 1})"
            stem = f"{_slug(tag)}_{model}_shared_pca"
            out_png = Path(tempfile.mkdtemp(prefix="rsembed_shared_")) / f"{stem}.png"
            Image.fromarray((img * 255).astype(np.uint8)).save(out_png)
            rec = create_output_file_from_path(out_png, filename=out_png.name)
            # RE-COLOUR IN PLACE. This is the same embedding of the same region, period and
            # model as the layer embed_region already drew — only the colour basis changed — so
            # it takes that layer's identity and the client swaps it rather than stacking a
            # second raster on top. Emitting a new id put every year on the map TWICE, and the
            # duplicate was the misleading one: the per-run colours this tool exists to replace,
            # sitting next to the aligned ones with nothing to tell them apart but a label.
            #
            # Reconstructed from the package's OWN manifest rather than passed in, because the
            # embed may have happened in an earlier turn. `start`/`end` stay in the identity, so
            # one place at several periods — the documented case, and the case that produced
            # this bug report — remains several layers.
            superseded = None
            manifest = entry["meta"] if isinstance(entry.get("meta"), dict) else {}
            if manifest.get("start") and manifest.get("end"):
                superseded = _layer_id("pca", _region_tag(None, bbox), bbox=_round_bbox(bbox),
                                       model=model, start=manifest["start"],
                                       end=manifest["end"])
            layers.append(_raster_layer(
                rec, bbox, _layer_label(f"{model} embedding (shared PCA)", tag),
                # Falls back to an identity of its own when the manifest cannot say which layer
                # this supersedes: an extra layer is a worse map, but a WRONG replacement would
                # overwrite a raster of somewhere else. `package` is what makes this raster THIS
                # one, since bbox, model and basis are identical for every layer in the call.
                superseded or _layer_id("sharedpca", _region_tag(None, bbox),
                                        package=str(entry["file_id"]), bbox=_round_bbox(bbox),
                                        model=model, basis=fitted_basis),
                # A re-coloured raster points at the SAME vectors as the layer it replaces —
                # only the colours were refitted — so the pointer has to survive the re-render
                # or aligning the colours would cost the layer its data.
                embedding={"file_id": str(entry["file_id"]), "model": model,
                           "recoloured_on_shared_basis": True}))
            regions.append({"file_id": entry["file_id"], "label": tag, "bbox": bbox,
                            "grid": [h, w], "image_file_id": rec["file_id"],
                            "download_url": rec.get("download_url")})

        if not layers:
            return json.dumps({"ok": False,
                               "error": "no region could be placed on the map",
                               "packages_rejected": problems})

        var = np.asarray(sv, dtype=np.float64) ** 2
        out: Dict[str, Any] = {
            "ok": True, "model": model, "regions": regions,
            "pixels_used": int(stacked.shape[0]), "on_map": True,
            "variance_explained": [round(float(v), 4) for v in (var[:3] / var.sum())],
            "note": "These layers share ONE PCA basis and ONE contrast stretch, fitted across "
                    "all of them together, so a colour means the same thing in every one — "
                    "unlike the per-run rasters embed_region produces, which are each "
                    "normalised on their own pixels and are NOT comparable to each other. "
                    "They are still not land-cover classes.",
            # The answer used to describe these as new layers, which read as "now there are six"
            # when the map had three. Each re-colour takes over the layer it supersedes, so the
            # count does not change — say re-coloured, not added.
            "layers": "re-coloured in place: each takes over the layer the region already had "
                      "on the map, so no year is drawn twice",
        }
        if step > 1:
            out["basis_fitted_on"] = (f"every {step}th pixel ({_PCA_FIT_MAX_PX:,} cap); all "
                                      "pixels were projected and rendered")
        if problems:
            out["packages_rejected"] = problems
        out["map_layer"] = layers[0]
        if len(layers) > 1:
            out["map_layers"] = layers
        return json.dumps(out)

    return [
        StructuredTool.from_function(func=accept_null_defaults(list_embedding_models), name="list_embedding_models", metadata=meta),
        StructuredTool.from_function(func=accept_null_defaults(embed_region), name="embed_region", metadata=meta),
        StructuredTool.from_function(func=accept_null_defaults(segment_region), name="segment_region", metadata=meta),
        StructuredTool.from_function(func=accept_null_defaults(embedding_change), name="embedding_change", metadata=meta),
        StructuredTool.from_function(func=accept_null_defaults(compare_regions), name="compare_regions", metadata=meta),
        StructuredTool.from_function(func=accept_null_defaults(align_embedding_colors), name="align_embedding_colors", metadata=meta),
        StructuredTool.from_function(func=accept_null_defaults(list_prediction_heads), name="list_prediction_heads", metadata=meta),
        StructuredTool.from_function(func=accept_null_defaults(list_embedding_packages), name="list_embedding_packages", metadata=meta),
        StructuredTool.from_function(func=accept_null_defaults(predict_for_region), name="predict_for_region", metadata=meta),
        StructuredTool.from_function(func=accept_null_defaults(predict_from_package), name="predict_from_package", metadata=meta),
    ]




# --- zonal embeddings: pixels inside a polygon, aggregated ----------------------
_ZONAL_TIMEOUT_S = float(os.getenv("RS_EMBED_ZONAL_TIMEOUT_S", "900"))
# Distinct, colour-blind-safe hues for cluster classes (Okabe-Ito, alpha added).
_CLUSTER_COLORS = [
    [0, 114, 178, 190], [230, 159, 0, 190], [0, 158, 115, 190], [204, 121, 167, 190],
    [86, 180, 233, 190], [213, 94, 0, 190], [240, 228, 66, 190], [120, 120, 120, 190],
]


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
# The service sends its embedder's meta verbatim, with a comment saying it "carries the
# provenance a caller must not invent". Passing on only the numbers and dropping how they
# were produced re-creates exactly the problem that comment guards against: a follow-up like
# "what resolution was that?" still gets a confident answer, reconstructed from the defaults
# rather than read from the run. That is right until the day a default changes.
#
# Curated, not dumped. The keys below are the ones that change what a number MEANS; the
# embedder's own diagnostics (param_mean/std/absmax, device, batch_infer, batch_tokens_shape)
# and the model-side band aliases are debugging aids that would crowd the context and answer
# nothing a user asks.
_PROV_TOP = (
    "model", "model_key", "modality", "type", "backend", "source",
    "normalization", "pretrained", "layer_index", "image_size",
    "scale_m", "pixel_ground_m", "dims", "year",
    # How much of the footprint had no data — it changes how much a vector is worth, and a
    # mostly-empty zone otherwise reads exactly like a full one.
    "nodata_fraction",
    "grid_type", "grid_hw", "tokens_include_cls",
    "grid_orientation_policy", "grid_orientation_applied", "y_axis_direction",
)
# Flattened up from meta["sensor"], which is where the on-the-fly path puts them. `bands` is
# kept (a user does ask which bands); `bands_terramind` is the same list under model-side
# names and is dropped.
_PROV_SENSOR = ("collection", "scale_m", "cloudy_pct", "composite", "fill_value")
# Sentinel-2 has 13; past that a "band" list is a dimension list, not a sensor description.
_MAX_BAND_NAMES = 16


def _provenance(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """What the run actually used, flattened for reading, or {} when the service sent none.

    Flat on purpose: the model quotes these back to the user, and a nested sensor/temporal
    shape gets summarised into vagueness on the way. Absent keys are omitted rather than
    reported as null, so "not sent" never reads as "the run had no value for it".
    """
    if not isinstance(meta, dict) or not meta:
        return {}
    out: Dict[str, Any] = {}
    sensor = meta.get("sensor") if isinstance(meta.get("sensor"), dict) else {}
    for key in _PROV_SENSOR:
        if sensor.get(key) is not None:
            out[key] = sensor[key]
    for key in _PROV_TOP:                       # top level wins over the sensor block
        if meta.get(key) is not None:
            out[key] = meta[key]

    # `bands` lives in the sensor block for the on-the-fly path and at the top level for the
    # precomputed one, so both are consulted. A precomputed product names its 64 embedding
    # DIMENSIONS here (A00…A63), not spectral bands — listing those is noise, so past a
    # spectral-length list only the count is kept.
    bands = meta.get("bands") if meta.get("bands") is not None else sensor.get("bands")
    if bands is not None:
        bands = list(bands)
        if len(bands) <= _MAX_BAND_NAMES:
            out["bands"] = bands
        else:
            out["bands_count"] = len(bands)

    temporal = meta.get("temporal") if isinstance(meta.get("temporal"), dict) else {}
    start, end = temporal.get("start"), temporal.get("end")
    if start or end:
        out["date_range"] = f"{start or '?'}..{end or '?'}"
        if temporal.get("mode"):
            out["temporal_mode"] = temporal["mode"]

    # A caveat the numbers cannot show. If the grid was never oriented and the source's y-axis
    # direction is unknown, row order relative to north is unverified — which matters the
    # moment anyone reads the grid as a picture or joins it to coordinates.
    if out.get("grid_orientation_applied") is False and out.get("y_axis_direction") == "unknown":
        out["orientation_caveat"] = (
            "grid rows were not reoriented and the source reported no y-axis direction, so "
            "north-up is assumed, not verified")
    return out


def _zonal_service_body(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The optional temporal fields for a zonal request.

    A date RANGE when the caller gave one; the service falls back to `year` otherwise. Both
    are required together — half a range would silently become a whole-year composite, which
    is exactly the kind of quiet substitution that made "March to May" come back as 2025.
    """
    if payload.get("start") and payload.get("end"):
        return {"start": str(payload["start"]), "end": str(payload["end"])}
    return {}


def _dimension_keys(rows: List[Dict[str, Any]]) -> List[str]:
    """The eNNN columns, in dimension order.

    Sorted by the number, not the string: the service zero-pads today, so lexicographic
    order happens to agree, but a rename to unpadded ``e9``/``e10`` would silently transpose
    two dimensions of every vector — the kind of wrong answer nothing downstream can detect.
    Read from the first row that HAS them, because an uncovered zone carries none.
    """
    for row in rows:
        keys = [k for k in row if str(k).startswith("e") and str(k)[1:].isdigit()]
        if keys:
            return sorted(keys, key=lambda k: int(str(k)[1:]))
    return []


def run_zonal_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Per-zone embeddings from the rs-embed SERVICE, in the shape the tools already expect.

    This used to fork a subprocess under a second interpreter that had ``rs_embed``
    installed, because the library had no zones API: the tile sweep, the EPSG:3857 affine,
    the rasterising and the per-zone accumulation were all done here. Two things followed.
    The interpreter was located by a hardcoded path that existed on one laptop and nowhere
    else, so the deployed container answered "the rs_embed runtime is unavailable" for every
    request. And forking a process that already had torch loaded raised an OpenMP mutex
    failure that wedged the turn.

    ``rs_embed.embed_zones`` does the sweep now, so it belongs behind the service — where the
    model runtime, the Earth Engine credentials and the warm weights cache already are — and
    this is one HTTP call. The result keeps the old keys, so ``embed_zones`` and
    ``fit_zone_model`` downstream needed no edits.
    """
    if payload.get("mode") == "fit":
        # Fitting a ridge on vectors that already exist is pandas, numpy and a linear solve
        # over a CSV: no model runtime, no Earth Engine, nothing the service owns. It stays
        # in this process rather than crossing the boundary for nothing — which is why the
        # worker's fit path is scikit-learn-free: its k-means segfaults a process that has
        # torch loaded, and a segfault takes the worker down, not just the turn.
        from agent_runtime.rs_embed_zonal_worker import fit as _fit
        return _fit(payload)

    import geopandas as gpd

    id_field = payload.get("zone_id_field")
    model = str(payload.get("model") or "gse")
    year = int(payload.get("year") or 2022)
    try:
        gdf = gpd.read_file(payload.get("polygons_path"))
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        gdf = gdf.to_crs("EPSG:4326")
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"could not read the polygons: {exc}"}
    if id_field:
        if id_field not in gdf.columns:
            return {"ok": False,
                    "error": f"zone_id_field {id_field!r} is not in the layer",
                    "available_columns": [c for c in gdf.columns if c != "geometry"][:40],
                    "hint": "Name a column that identifies each polygon, or omit zone_id_field "
                            "to key zones by row number — but then use the same setting when "
                            "fitting, or the vectors and the labels will not meet."}
        # The id is the ONLY thing joining a returned vector back to a polygon — on the map,
        # in the CSV, and in fit_zone_model's merge. A column that cannot key uniquely does
        # not fail: it quietly paints one zone's vector onto every polygon that shares its
        # value, and puts the same feature row in both sides of a cross-validation split.
        blank = gdf[id_field].isna()
        if bool(blank.any()):
            return {"ok": False,
                    "error": f"{int(blank.sum())} of {len(gdf)} polygons have no value in "
                             f"{id_field!r}, so their vectors could not be joined back",
                    "hint": "Fill the column, pick another, or omit zone_id_field to key zones "
                            "by row number — but then use the same setting when fitting."}
        dup = gdf[id_field][gdf[id_field].duplicated(keep=False)]
        if len(dup):
            examples = sorted({str(v) for v in dup})[:4]
            return {"ok": False,
                    "error": f"{id_field!r} does not identify a polygon uniquely: {len(dup)} "
                             f"polygons share {dup.nunique()} value(s), e.g. {examples}",
                    "hint": "Pick a unique column, or omit zone_id_field to key zones by row "
                            "number — but then use the same setting when fitting.",
                    "available_columns": [c for c in gdf.columns if c != "geometry"][:40]}

    wanted = payload.get("zone_ids")
    if wanted:
        if not id_field:
            return {"ok": False,
                    "error": "zone_ids needs zone_id_field, or there is nothing to match them "
                             "against",
                    "hint": "Pass zone_id_field=<the column holding the identifier>, or drop "
                            "zone_ids to embed every polygon in the layer."}
        wanted = {str(z) for z in wanted}
        gdf = gdf[gdf[id_field].astype(str).isin(wanted)].reset_index(drop=True)
        if gdf.empty:
            return {"ok": False,
                    "error": f"none of the {len(wanted)} requested zone_ids are in "
                             f"{id_field!r}",
                    "hint": "Check the identifiers against the layer, or drop zone_ids to "
                            "embed every polygon."}
        missing = wanted - {str(v) for v in gdf[id_field]}
        if missing:
            # Embedding the ones that exist beats refusing the lot, but a silently short answer
            # is how "I asked for five and got three" goes unnoticed.
            payload = {**payload, "_missing_zone_ids": sorted(missing)[:10]}

    # Only the identifier travels with the geometry. The service keys zones by row index when
    # no field is named, and dropping columns cannot reorder rows, so the mapping is safe —
    # while a tract layer's forty attribute columns would otherwise be re-encoded into the
    # request body for nothing. The id goes as text: the round trip through JSON turns a
    # numeric id into a number and back, and 17031836500 that returns as "17031836500.0"
    # matches no polygon on the way home.
    keep = [id_field, "geometry"] if id_field else ["geometry"]
    sending = gdf[keep].copy()
    if id_field:
        sending[id_field] = sending[id_field].astype(str)
    body: Dict[str, Any] = {
        "zones_geojson": json.loads(sending.to_json()),
        "model": model, "year": year, "zone_id_field": id_field,
        "tile_px": int(payload.get("tile_px") or 256),
        # The pixels themselves, masked to the zones. A 64-number average does not answer
        # "what does this area look like to the model", and the service declines the render
        # rather than degrade it when the extent is too large for one request.
        "image": bool(payload.get("image", True)),
    }
    body.update(_zonal_service_body(payload))
    # `is not None`, not truthiness. max_tiles=0 is falsy, so it used to be dropped from the
    # body entirely, the service default of None applied, and the sweep ran COMPLETELY
    # UNCAPPED -- the exact opposite of what 0 asks for.
    if payload.get("max_tiles") is not None:
        body["max_tiles"] = int(payload["max_tiles"])
    res = _svc("/api/zones", body, timeout=_ZONAL_TIMEOUT_S)
    if res.get("error") or not res.get("ok"):
        out = {"ok": False,
               "error": str(res.get("error") or "the zones service returned no result")[:400]}
        # _svc's hint names the next action (start the service, point RS_EMBED_URL at it).
        # Dropping it turned a fixable failure into "unavailable" with nowhere to go.
        for key in ("hint", "detail"):
            if res.get(key):
                out[key] = res[key]
        return out

    meta = res.get("meta") or {}
    rows = res.get("rows") or []
    dim_keys = _dimension_keys(rows)
    zones: List[Dict[str, Any]] = []
    for row in rows:
        pixels = int(row.get("pixels") or 0)
        zones.append({
            "zone_id": str(row.get("zone_id")),
            "pixels": pixels,
            "area_km2": float(row.get("area_km2") or 0.0),
            "mean": [float(row.get(k) or 0.0) for k in dim_keys] if pixels else None,
        })
    covered = sum(1 for z in zones if z["pixels"])
    if covered:
        # One clustering implementation, shared with the standalone worker: the map's groups
        # are L2-normalised k-means over the zone vectors, 1-based because the layer treats
        # a missing group as "not embedded".
        from agent_runtime.rs_embed_zonal_worker import assign_look_alike_groups
        assign_look_alike_groups(zones, int(payload.get("clusters") or 0))

    out: Dict[str, Any] = {
        "ok": covered > 0,
        "model": str(meta.get("model") or model),
        "year": year,
        "dims": int(meta.get("dims") or len(dim_keys)),
        "bands": list(meta.get("bands") or []),
        "zone_id_field": meta.get("zone_id_field", id_field),
        # zones_total counts the polygons SENT, not the rows returned: a zone no tile reached
        # comes back with pixels == 0 and must still be accounted for.
        "zones_total": int(meta.get("zones_total") or len(zones)),
        "zones_with_pixels": int(meta.get("zones_with_pixels") or covered),
        # scale_m is EPSG:3857 metres; pixel_ground_m is what a pixel actually covers.
        "scale_m": meta.get("scale_m"),
        "pixel_ground_m": meta.get("pixel_ground_m"),
        # tiles_planned counts every cell of the bounding grid, most of which are empty;
        # tiles_needed counts only the cells a zone actually touches, which is the number a
        # cap should be read against.
        "tiles_planned": meta.get("tiles_planned"),
        "tiles_needed": meta.get("tiles_needed"),
        "tiles_fetched": meta.get("tiles_fetched"),
        "tiles_skipped_by_cap": int(meta.get("tiles_skipped_by_cap") or 0),
        "tiles_capped": bool(meta.get("tiles_capped")),
        "tile_errors": list(meta.get("tile_errors") or []),
        "pixel_size_warnings": list(meta.get("pixel_size_warnings") or []),
        # What the run actually used — imagery source, bands, compositing, dates, model
        # variant, grid orientation. Answering "at what resolution, from which collection?"
        # from here beats reconstructing it from the defaults.
        "provenance": _provenance(meta),
        "zones": zones,
        # The PCA-RGB picture of the pixels inside the shapes, rendered by the service
        # alongside the sweep and cut to the same polygons with the same rasterisation, so
        # the picture is of the pixels the vectors were computed from.
        "image": res.get("image") or {"error": "the service returned no pixel image"},
        "error": None if covered else "no zone received any pixels",
    }
    if payload.get("_missing_zone_ids"):
        out["zone_ids_not_found"] = payload["_missing_zone_ids"]
    if not covered:
        out["hint"] = ("Check that the polygons are where you think they are, that the model "
                       "has coverage for this year, and -- if you passed max_tiles -- that "
                       "it is not cutting the sweep short before it reaches them.")
    return out


def make_rs_embed_zonal_tools(default_input_file_ids: Optional[List[str]] = None) -> List[Any]:
    """Tools that aggregate per-pixel embeddings inside polygons."""
    from langchain_core.tools import StructuredTool

    meta = {"category": "geo"}

    def embed_zones(file_id: str, zone_id_field: Optional[str] = None, model: str = "gse",
                    year: int = 2022, clusters: int = 5, tile_px: int = 200,
                    max_tiles: Optional[int] = None, name: Optional[str] = None,
                    zone_ids: Optional[Union[str, List[str]]] = None,
                    sibling_file_ids: Optional[Union[str, List[str]]] = None,
                    start: Optional[str] = None, end: Optional[str] = None) -> str:
        """Embed one or many POLYGONS — the pixels INSIDE each shape — and map the result.

        This is also the tool for "the embedding of <a named place>": pair it with
        admin_boundary, which turns "Urbana" or "Champaign County" into the polygon file this
        takes. embed_region would embed a rectangle around the centroid instead, which is not
        the city.

        This is the tool for "the embedding of this area" whenever the area is a shape rather
        than a rectangle, whether the layer holds one feature or eight hundred. embed_region
        embeds a RECTANGLE, which for a lakefront polygon also takes in open water.

        Asked for ONE area out of many — "the embedding for geoid 17031330100" against a file
        of 801 tracts — pass `zone_ids=["17031330100"]` with `zone_id_field`. Do not extract
        the polygon into a new file first: that is four extra steps, and the sweep is bounded
        by the zones' own extent either way.

        Divides the polygons' area into pixels, embeds each pixel with a remote-sensing
        foundation model, and averages the pixels that fall INSIDE each polygon — so each
        zone (census tract, county, field, watershed, drawn box) gets one vector describing
        what it looks like from space. Works with any polygon layer: GeoJSON, shapefile,
        GeoPackage.

        `start`/`end` embed a DATE RANGE instead of the whole of `year` — pass both or neither.
        Use them whenever the user names a period: without them a request for March-May
        silently becomes a full-year composite. Either form works, the same as embed_region:
        a month ("2025-03") or a full date ("2025-03-01"). A month covers all of itself, so
        "2025-03" to "2025-05" is March through May inclusive.

        There is NO tile cap by default, so the sweep fetches every tile the polygons touch
        and the answer covers all of them. Each tile is one request to the imagery provider,
        and a large layer can need hundreds.

        DO NOT set `max_tiles` for an area the user named — a city, a county, a boundary from
        admin_boundary. A cap there produces a map that is WRONG IN A WAY THE PICTURE DOES NOT
        SHOW: the raster still fills its frame, the zones under the dropped tiles come back with
        `pixels == 0`, and only the `truncated` field says so. Capping a city at 20 of its 44
        tiles has already happened and the answer had to disown its own map. Set it only for an
        explicitly exploratory look at a large region, and say in the answer that the result is
        partial. Whatever the cap drops is reported as `truncated`.

        NOT an embedding package. This returns per-zone vectors as a CSV, where embed_region
        saves an .npz holding a per-pixel grid — so align_embedding_colors, predict_from_package
        and anything else that reads a package CANNOT consume this output. To put two areas on
        one shared colour basis today, embed each with embed_region and align those packages.

        Returns a CSV of per-zone vectors ready for machine learning (use fit_zone_model),
        and puts TWO things on the map: a PCA-RGB picture of the pixels themselves, cut to the
        shapes, and the zones grouped into `clusters` look-alike groups. Each zone
        also carries `pixels` and `area_km2` — its support — and because the pixel count is
        there, the per-zone SUM is recoverable exactly, so zones roll up to a coarser
        partition without error.
        """
        # One name or several, however they were written — a bare string here would iterate
        # characters into the zone filter and match nothing.
        zone_ids = _as_list(zone_ids)
        sibling_file_ids = _as_list(sibling_file_ids)
        # Every argument that determines the result. A repeat with ANY of them changed is a
        # different question and runs: same polygon, different year is the Change workflow.
        memo_key = _zone_memo_key(
            file_id=file_id, zone_id_field=zone_id_field, model=model, year=year,
            clusters=clusters, tile_px=tile_px, max_tiles=max_tiles, name=name,
            zone_ids=sorted(str(z) for z in (zone_ids or [])),
            sibling_file_ids=sorted(str(f) for f in (sibling_file_ids or [])),
            start=_iso_date(start), end=_iso_date(end, month_end=True))
        replayed = _zone_memo_get(memo_key)
        if replayed is not None:
            try:
                prior = json.loads(replayed)
            except ValueError:  # pragma: no cover - a stored result is our own json
                prior = None
            if isinstance(prior, dict):
                # Says so in the payload the model reads. The trace still shows two calls, and
                # an answer that describes two sweeps of one city would be wrong about what was
                # done — the layers and file_ids below are the FIRST call's, not a second set.
                prior["reused_earlier_run"] = (
                    "identical to a call already made in this conversation, so the tiles were "
                    "not fetched again — these are that run's layers and files, not new ones")
                return json.dumps(prior)

        tmp = None
        # Outside the guard below: that try/except exists because `_stage` is missing in some
        # builds, and its fallback branch does not re-import everything. map_layers has no
        # optional dependency, so putting this there left the name UNBOUND on exactly the path
        # the fallback takes — the layer then failed with "zones computed but not mapped".
        from agent_runtime.map_layers import boundary_layer_id
        try:
            import numpy as np

            from agent_runtime.file_store import create_output_file_from_path
            from agent_runtime.langchain_geo_tools import _stage, artifact_name  # type: ignore
        except Exception:  # pragma: no cover - import shape differs in some builds
            from agent_runtime.file_store import create_output_file_from_path
            import numpy as np
            from agent_runtime.langchain_geo_tools import artifact_name
            _stage = None  # type: ignore

        try:
            from agent_runtime.langchain_geo_tools import _stage_vector_source, _index_attached

            attached = _index_attached(default_input_file_ids)
            read_path, tmp = _stage_vector_source(file_id, sibling_file_ids, attached)
        except Exception as exc:  # noqa: BLE001
            logger.warning("embed_zones failed fast: could not read %s: %s", file_id, exc)
            return json.dumps({"ok": False, "error": f"could not read {file_id}: {exc}"})

        png_path = Path(tempfile.mkdtemp(prefix="rsembed_zonal_")) / artifact_name(
            name, "png", default=f"{model}_zone_pixels")
        res = run_zonal_worker({"polygons_path": str(read_path), "zone_id_field": zone_id_field,
                                "model": model, "year": int(year), "tile_px": int(tile_px),
                                "max_tiles": None if max_tiles is None else int(max_tiles),
                                # A month is widened to the whole month here rather than
                                # rejected: embed_region documents its window as "YYYY-MM", so
                                # a model that read that one passes months to this one.
                                "start": _iso_date(start),
                                "end": _iso_date(end, month_end=True),
                                "zone_ids": [str(z) for z in zone_ids] if zone_ids else None,
                                "clusters": max(2, min(int(clusters), len(_CLUSTER_COLORS))),
                                "image": True})
        if not res.get("ok"):
            # Logged, because the trace does not show tool RESULTS — only calls. A failure and
            # a success render identically as a single `embed_zones(...)` line, which is how a
            # fast failure followed by a retry read as a duplicate sweep for two rounds of
            # diagnosis. Until the trace carries results, the server log is the only place the
            # reason exists.
            logger.warning("embed_zones failed: %s",
                           json.dumps({k: v for k, v in res.items()
                                       if k in ("error", "detail", "hint")})[:400])
            return json.dumps({"ok": False, **{k: v for k, v in res.items() if k != "zones"}})

        zones = res["zones"]
        dims = int(res["dims"])
        with_px = [z for z in zones if z["pixels"]]

        # --- CSV of per-zone vectors: the artifact an ML step consumes ---
        stem = artifact_name(name, "csv", default=f"{model}_zone_embeddings")
        out_csv = Path(tempfile.mkdtemp(prefix="rsembed_zonal_")) / stem
        cols = ["zone_id", "pixels", "area_km2"] + [f"e{i:03d}" for i in range(dims)]
        lines = [",".join(cols)]
        for z in with_px:
            lines.append(",".join([str(z["zone_id"]), str(z["pixels"]), f"{z['area_km2']:.6f}"]
                                  + [f"{v:.6f}" for v in z["mean"]]))
        out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
        csv_rec = create_output_file_from_path(out_csv, filename=out_csv.name)

        # --- the map layer: look-alike groups. Groups are the one view of a 64-dim vector
        # worth looking at — a choropleth of a single dimension is a picture of an arbitrary
        # axis, and it looks like a result.
        layer = None
        cluster_note = None
        # What every zonal layer in this call is computed FROM. Keyed on the REQUEST, so a
        # swallowed tile error cannot re-identify a layer, and shared by all three so they
        # agree on what "the same run" means.
        # The EFFECTIVE period, not the raw arguments. _zonal_service_body sends a range only
        # when both ends are given and the service falls back to `year` otherwise, so half a
        # range and no range are the same imagery — digesting the arguments raw split one
        # composite across two layers, and left `year` deciding identity on runs that ignored it.
        _period = (("range", str(start), str(end)) if start and end else ("year", int(year)))
        zone_content = {"file": file_id, "model": model, "period": _period,
                        "tile_px": int(tile_px), "max_tiles": max_tiles,
                        "zone_id_field": zone_id_field,
                        # The data actually read is assembled from these too: _stage_vector_source
                        # reconstructs a shapefile from its siblings, so one file_id can name
                        # different geometry depending on what came with it.
                        "siblings": sorted(str(f) for f in sibling_file_ids) if sibling_file_ids else None,
                        # What the caller ASKED for. len(present) is the count that came back,
                        # which a swallowed tile error changes without changing the request.
                        "clusters": max(2, min(int(clusters), len(_CLUSTER_COLORS))),
                        "zone_ids": sorted(str(z) for z in zone_ids) if zone_ids else None}
        try:
            import geopandas as gpd

            gdf = gpd.read_file(read_path)
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            gdf = gdf.to_crs("EPSG:4326")
            # str() of the same values run_zonal_worker sent, so the ids match on the way back.
            ids = ([str(v) for v in gdf[zone_id_field]]
                   if zone_id_field and zone_id_field in gdf.columns
                   else [str(i) for i in range(len(gdf))])
            if zone_ids:
                wanted = {str(z) for z in zone_ids}
                rows = [i for i, zid in enumerate(ids) if zid in wanted]
                gdf = gdf.iloc[rows].reset_index(drop=True)
                ids = [ids[i] for i in rows]
            grouped = {z["zone_id"]: z for z in with_px if z.get("group")}
            keep = [i for i, zid in enumerate(ids) if zid in grouped]
            if keep:
                carry = ["geometry"] + [c for c in ("NAME",) if c in gdf.columns]
                sub = gdf.iloc[keep][carry].copy()
                sub["zone_id"] = [ids[i] for i in keep]
                sub["pixels"] = [grouped[ids[i]]["pixels"] for i in keep]
                sub["area_km2"] = [round(grouped[ids[i]]["area_km2"], 4) for i in keep]
                sub["look_alike_group"] = [f"group {grouped[ids[i]]['group']}" for i in keep]
                gj = Path(tempfile.mkdtemp(prefix="rsembed_zonal_")) / artifact_name(
                    name, "geojson", default=f"{model}_zone_groups")
                sub.to_file(gj, driver="GeoJSON")
                rec = create_output_file_from_path(gj, filename=gj.name)
                present = sorted({str(v) for v in sub["look_alike_group"]})
                # The REQUESTED extent, not sub's. `sub` holds only the zones that came back
                # with pixels, so a single failed tile shrinks it — and an id built on that
                # moves when coverage changes, stacking a duplicate of a layer that should
                # have replaced itself.
                zone_tag = _region_tag(name, gdf.total_bounds)
                if len(keep) == 1:
                    # "zone groups (k=1)" is a cluster analysis of one thing, and a legend with
                    # a single entry explains nothing. Say which PLACE it is: the model was
                    # adding a SECOND layer of the same polygon because this one did not read
                    # as the area it had asked about.
                    #
                    # Prefer the polygon's own NAME over the zone id. admin_boundary writes it
                    # and labels its layer with it, so "Urbana city — gse embedded" reads as the
                    # thing the user asked about where "gse embedded zone 1777005" does not.
                    place = str(name or "").strip()
                    if not place and "NAME" in getattr(sub, "columns", []):
                        place = str(sub["NAME"].iloc[0] or "").strip()
                    if not place:
                        place = f"zone {sub['zone_id'].iloc[0]}"
                    layer = {"url": rec.get("download_url"),
                             # TAKES THE PLACE OF the outline this polygon file already has on
                             # the map: admin_boundary keys its layer on the same file_id, so
                             # this redraws it with what the embedding found inside instead of
                             # stacking a second copy of the same city beside it.
                             #
                             # Only for ONE zone. The multi-zone branch below keeps its own
                             # k-bearing id on purpose — asking for 3 groups and then 6 is two
                             # analyses that must coexist — but a single zone has no clustering
                             # to tell apart, k is always 1, and the layer is the input polygon
                             # with three attributes added.
                             "id": boundary_layer_id(file_id),
                             # Through _layer_label so it gets the same de-duplication: the
                             # model names these layers itself and usually puts the model in
                             # the name, which would otherwise read "… — gse — … — gse embedded".
                             "label": _layer_label(f"{model} embedded", place),
                             # Outline, not fill: this layer sits over the pixel image of the
                             # same polygon, and a filled one covers the picture it frames.
                             "render": "shapes", "outline": True,
                             "source": "analysis", "count": 1}
                else:
                    layer = {"url": rec.get("download_url"),
                             # k belongs in the id because it is in the label: asking for 3
                             # groups and then 6 is two analyses of the same zones, and the
                             # second must not silently replace the first. A zone raster
                             # already keys on its k for the same reason.
                             "id": _layer_id("zonegroups", file_id, **zone_content),
                             "label": _layer_label(
                                 f"{model} zone groups (k={len(present)})", zone_tag),
                             "render": "categories", "style_by": "look_alike_group",
                             "source": "analysis", "count": len(keep),
                             "legend": [{"label": g,
                                         "color": _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)]}
                                        for i, g in enumerate(present)]}
        except Exception as exc:  # noqa: BLE001
            cluster_note = f"zones computed but not mapped: {type(exc).__name__}: {exc}"[:200]
        if layer is None and cluster_note is None:
            # The group layer is the only thing this tool puts on the map, so "no layer" is a
            # failed delivery, not a detail. Say it in the payload the model reads.
            cluster_note = ("no zone could be placed on the map: none of the embedded zone ids "
                            "matched a polygon in the layer. Say the vectors were computed but "
                            "not mapped — do not describe a map.")

        # One zone (or a few) is the commonest "embed this area" request, and a path to a CSV
        # does not answer it — which is why the model went looking for a tool that returned a
        # vector and found the bounding-box one.
        inline = []
        if len(with_px) <= 5:
            for z in with_px:
                vec = np.asarray(z["mean"], dtype=float)
                inline.append({"zone_id": z["zone_id"], "pixels": z["pixels"],
                               "area_km2": z["area_km2"], "dim": int(vec.size),
                               "vector_norm": round(float(np.linalg.norm(vec)), 4),
                               "vector_first_10": [round(float(v), 6) for v in vec[:10]]})

        out: Dict[str, Any] = {
            "ok": True, "model": model, "year": int(year), "dims": dims,
            "zones_total": res["zones_total"], "zones_with_pixels": res["zones_with_pixels"],
            # scale_m is EPSG:3857 metres; the ground figure is what a pixel really covers.
            "scale_m_mercator": res["scale_m"], "pixel_ground_m": res["pixel_ground_m"],
            "tiles_fetched": res["tiles_fetched"], "tiles_planned": res["tiles_planned"],
            "vectors_csv": {"file_id": csv_rec["file_id"], "filename": csv_rec.get("filename"),
                            "download_url": csv_rec.get("download_url"),
                            "size_bytes": csv_rec.get("size_bytes")},
            "support": {"pixels_min": min((z["pixels"] for z in with_px), default=0),
                        "pixels_median": int(np.median([z["pixels"] for z in with_px])) if with_px else 0,
                        "pixels_max": max((z["pixels"] for z in with_px), default=0)},
            "note": "Each zone's vector is the MEAN of the pixels inside it. `pixels` is its "
                    "support: a model fitted on small zones is extrapolating when applied to "
                    "a much larger one, because pooling averages away variance. These vectors "
                    "are a CSV, not an embedding package (.npz), so align_embedding_colors and "
                    "predict_from_package cannot read them — the two rasters this draws each "
                    "have their OWN PCA basis and are not comparable by colour.",
        }
        if res.get("tiles_capped"):
            # tiles_needed, not tiles_planned: the grid counts empty cells the sweep skips for
            # free, so "N of <grid>" read as lost coverage when nothing had been lost.
            out["truncated"] = (f"max_tiles={max_tiles} stopped the sweep after "
                                f"{res['tiles_fetched']} of the {res.get('tiles_needed')} tiles "
                                f"the zones touch; {res.get('tiles_skipped_by_cap')} tile(s) were "
                                f"never fetched — the zones under them have no pixels")
        if res.get("pixel_size_warnings"):
            out["pixel_size_warnings"] = res["pixel_size_warnings"]
        if inline:
            out["zone_vectors"] = inline
        if res.get("tile_errors"):
            out["tile_errors"] = res["tile_errors"]
        if cluster_note:
            out["cluster_note"] = cluster_note
        # Two views, both delivered: the pixel-level embedding masked to the shapes (what the
        # zone vectors are computed FROM), and the zones grouped by those vectors. Asking for
        # "the embedding of these polygons" and getting only a group colour hid the actual data.
        layers = []
        img = res.get("image") or {}
        rec_png = None
        if img.get("bounds"):
            if img.get("png"):                       # base64 from the service
                rec_png = _save_png(str(img["png"]), png_path.stem)
            elif img.get("path") and Path(img["path"]).exists():
                rec_png = create_output_file_from_path(Path(img["path"]),
                                                       filename=Path(img["path"]).name)
        if rec_png:
            out["pixel_image"] = {"file_id": rec_png["file_id"],
                                  "download_url": rec_png.get("download_url"),
                                  "size_bytes": rec_png.get("size_bytes"),
                                  "size_px": img.get("size_px"),
                                  "pixels_shown": img.get("pixels_shown"),
                                  "colour": img.get("colour")}
            layers.append(_raster_layer(
                rec_png, [float(v) for v in img["bounds"]],
                _layer_label(f"{model} pixel embedding in zones",
                             _region_tag(name, img["bounds"])),
                _layer_id("zonepixels", file_id, **zone_content)))
        elif img.get("error"):
            # Say why there is no picture, and what would get one — the vectors are unaffected
            # either way, and an unexplained absence reads as a failed analysis.
            out["image_note"] = " ".join(str(img[k]) for k in ("error", "hint") if img.get(k))
        if layer:
            layers.append(layer)
        if layers:
            out["map_layer"] = layers[0]
            if len(layers) > 1:
                out["map_layers"] = layers
            out["on_map"] = True
        rendered = json.dumps(out)
        # Only a SUCCESSFUL sweep is remembered. A failure should be retried, not replayed —
        # a wedged service or an expired credential is exactly the case where the second
        # attempt is the one that works.
        if out.get("ok"):
            _zone_memo_put(memo_key, rendered)
        return rendered


    def fit_zone_model(vectors_csv_file_id: str, polygons_file_id: str, label_column: str,
                       zone_id_field: Optional[str] = None, blocks: int = 5,
                       name: Optional[str] = None,
                       sibling_file_ids: Optional[List[str]] = None) -> str:
        """Predict a per-zone VALUE from zone embeddings, and map the prediction.

        Takes the CSV from embed_zones plus the polygon layer carrying the truth in
        `label_column` (tree canopy, yield, hardship index, ...), fits a ridge model and
        scores it with SPATIAL BLOCK cross-validation: folds are contiguous blocks of space,
        so no zone is scored by a model that trained on its neighbours.

        Reports the blocked score, the score a naive random split WOULD have claimed, and a
        predict-the-mean baseline — the gap between the first two is how much of an apparent
        result was just adjacency. Puts out-of-fold predictions on the map with residuals.
        """
        tmp = None
        try:
            from agent_runtime.file_store import create_output_file_from_path
            from agent_runtime.langchain_geo_tools import (_index_attached, _resolve,
                                                          _stage_vector_source, artifact_name)

            csv_path, _rec = _resolve(vectors_csv_file_id)
            attached = _index_attached(default_input_file_ids)
            poly_path, tmp = _stage_vector_source(polygons_file_id, sibling_file_ids, attached)
            gj = Path(tempfile.mkdtemp(prefix="rsembed_fit_")) / artifact_name(
                name, "geojson", default=f"{label_column}_predicted")

            # Fitting needs no model runtime and no Earth Engine, so it runs here rather
            # than behind the service — see run_zonal_worker's "fit" branch.
            res = run_zonal_worker({"mode": "fit", "vectors_csv": str(csv_path),
                                    "polygons_path": str(poly_path),
                                    "label_column": label_column,
                                    "zone_id_field": zone_id_field,
                                    "blocks": int(blocks), "out_geojson": str(gj)})
            if not res.get("ok"):
                return json.dumps({"ok": False, **res})

            rec = create_output_file_from_path(gj, filename=gj.name)
            blocked = res["spatial_block_cv"]
            naive = res.get("naive_random_split_cv") or {}
            out = {
                "ok": True, **{k: v for k, v in res.items() if k != "ok"},
                "predictions_file_id": rec["file_id"],
                "download_url": rec.get("download_url"),
                "on_map": True,
                "map_layer": {"url": rec.get("download_url"),
                              # polygons_file_id as well as the name: _region_tag has no
                              # bbox to fall back on here, so an unnamed run identified the
                              # layer by label_column alone and the same column over two
                              # different areas collided.
                              "id": _layer_id(
                                  "predicted", polygons_file_id,
                                  vectors=vectors_csv_file_id, polygons=polygons_file_id,
                                  column=label_column, zone_id_field=zone_id_field,
                                  siblings=sorted(str(f) for f in sibling_file_ids)
                                  if sibling_file_ids else None,
                                  blocks=int(blocks)),
                              "label": _layer_label(
                                  f"{label_column} predicted from embeddings",
                                  _region_tag(name)),
                              "render": "choropleth", "style_by": "predicted",
                              "source": "analysis", "count": res.get("zones_fitted")},
                "note": "r2/rmse are OUT-OF-FOLD under spatial block CV, so they estimate "
                        "performance in an area the model has not seen. Mapped values are "
                        "those out-of-fold predictions, not fitted values. Quote the blocked "
                        "score, not the random-split one.",
            }
            if isinstance(blocked.get("r2"), (int, float)) and blocked["r2"] <= 0:
                out["verdict"] = ("no skill: the model does no better than predicting the mean, "
                                  "so the embeddings do not explain this variable at this "
                                  "sample size. Report that, do not present the map as a result.")
            elif isinstance(naive.get("r2"), (int, float)) and \
                    naive["r2"] - blocked["r2"] > 0.15:
                out["verdict"] = (f"a random split would have claimed r2={naive['r2']}, versus "
                                  f"{blocked['r2']} when whole blocks are held out — most of "
                                  f"that apparent skill was spatial adjacency, not prediction.")
            return json.dumps(out)
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            if tmp:
                import shutil
                shutil.rmtree(tmp, ignore_errors=True)

    return [StructuredTool.from_function(func=accept_null_defaults(embed_zones), name="embed_zones", metadata=meta),
            StructuredTool.from_function(func=accept_null_defaults(fit_zone_model), name="fit_zone_model", metadata=meta)]


__all__ = ["make_rs_embed_tools", "make_rs_embed_zonal_tools", "run_zonal_worker", "RS_EMBED_URL"]
