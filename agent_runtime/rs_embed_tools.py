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
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

RS_EMBED_URL = os.getenv("RS_EMBED_URL", "http://localhost:8077").rstrip("/")
# On-the-fly models download checkpoints on first use and run a transformer; the
# precomputed backends answer in seconds. Long, but bounded — a hung request is worse
# than a slow one because the turn never ends.
_TIMEOUT_S = float(os.getenv("RS_EMBED_TIMEOUT_S", "600"))
_DEFAULT_BUFFER_M = 2048          # matches the service's own point footprint
_MAX_MODELS_PER_CALL = 5


def _svc(path: str, payload: Optional[Dict[str, Any]] = None, *, method: str = "POST") -> Dict[str, Any]:
    """Call the rs-embed service, or return an error dict that says what to do."""
    import requests

    url = f"{RS_EMBED_URL}{path}"
    try:
        r = (requests.get(url, timeout=_TIMEOUT_S) if method == "GET"
             else requests.post(url, json=payload or {}, timeout=_TIMEOUT_S))
    except requests.exceptions.ConnectionError:
        return {"error": f"the rs-embed service is not reachable at {RS_EMBED_URL}",
                "hint": "Start it with: python -m uvicorn server:app --app-dir examples/webapp "
                        "--port 8077 (from the rs-embed repo), or set RS_EMBED_URL to where it runs. "
                        "Without it no embedding tool can run — say so rather than inventing values."}
    except requests.exceptions.Timeout:
        return {"error": f"the rs-embed service did not answer within {_TIMEOUT_S:.0f}s",
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
                  file_id: Optional[str], buffer_m: float) -> Any:
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
        b = _bounds_of_file(file_id)
        if b:
            return b
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
                  opacity: float = 0.85) -> Dict[str, Any]:
    """A ``map_layer`` descriptor the client drapes as a georeferenced image."""
    return {"url": rec.get("download_url"), "label": label, "render": "raster",
            "bounds": bbox, "opacity": opacity, "source": "analysis"}


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
                     models: Optional[List[str]] = None, start: str = "2022-06",
                     end: str = "2022-09", buffer_m: float = _DEFAULT_BUFFER_M,
                     name: Optional[str] = None) -> str:
        """Embed a region with remote-sensing foundation models and PUT THE RESULT ON THE MAP.

        Each model returns a learned description of what the place looks like from space over
        the given months. The embedding grid is projected to 3 colours (PCA) and draped over
        the region as an interactive raster layer, so similar-looking ground reads as similar
        colour. Also saves a downloadable .npz holding the real vectors for reuse.

        Region: pass `bbox` [minlon, minlat, maxlon, maxlat] (what the map's Region tool gives),
        or `lon`+`lat` for a point (a `buffer_m` square around it), or `file_id` of an uploaded
        layer to use its extent. `start`/`end` are months, "YYYY-MM".
        """
        box = _resolve_bbox(bbox, lon, lat, file_id, buffer_m)
        if isinstance(box, dict):
            return json.dumps({"ok": False, **box})

        avail = _svc("/api/models", method="GET")
        if avail.get("error"):
            return json.dumps({"ok": False, **avail})
        chosen = [str(m) for m in (models or ["gse"])][:_MAX_MODELS_PER_CALL]
        bad = _model_error(chosen, avail.get("models") or [])
        if bad:
            return json.dumps(bad)

        res = _svc("/api/embed", {"geometry": _geometry(box), "start": start, "end": end,
                                  "models": chosen, "buffer_m": int(buffer_m)})
        if res.get("error"):
            return json.dumps({"ok": False, **res})

        layers, summaries, failed = [], [], []
        for r in res.get("results") or []:
            model = str(r.get("model"))
            if not r.get("ok"):
                failed.append({"model": model, "error": str(r.get("error"))[:300]})
                continue
            stem = f"{_slug(name or 'embedding')}_{model}_pca"
            rec = _save_png(str(r.get("image") or ""), stem)
            entry = {"model": model, "type": r.get("type"), "dim": r.get("dim"),
                     "grid": r.get("grid_hw"), "vector_norm": round(float(r.get("norm") or 0), 3)}
            if rec:
                entry.update({"image_file_id": rec["file_id"], "download_url": rec.get("download_url")})
                layers.append(_raster_layer(rec, box, f"{model} embedding (PCA-RGB)"))
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
                                 f"{_slug(name or 'embedding')}_vectors")
            info: Dict[str, Any] = {"models_saved": pkg.get("models"),
                                    "pooled_vectors": True,
                                    "grids_saved": pkg.get("grids_saved") or []}
            if rec:
                info.update({"file_id": rec["file_id"], "filename": rec.get("filename"),
                             "download_url": rec.get("download_url"),
                             "size_bytes": rec.get("size_bytes")})
            # The service caps which grids go into the export (300x300 cells). Say so: a
            # missing full-resolution grid is otherwise invisible until someone loads the file.
            dropped = [m["model"] for m in summaries
                       if m["model"] not in (pkg.get("grids_saved") or [])]
            if dropped:
                info["full_grid_omitted_for"] = dropped
                info["why"] = ("the per-model grid exceeded the export cap (300x300 cells), so the "
                               "file holds the pooled vector only — enough for similarity, "
                               "prediction and comparison, not for per-pixel work")
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
            "map_layer": _raster_layer(rec, box, f"{model} segments (k={k})"),
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
                           models: Optional[List[str]] = None, start: str = "2022-06",
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
                                    "models": [str(m) for m in (models or [])],
                                    "buffer_m": int(buffer_m)})
        if res.get("error"):
            return json.dumps({"ok": False, **res})
        return json.dumps({"ok": True, "region_bbox": box, **res,
                           "note": "Each prediction carries the head's own validation score — "
                                   "quote it, because a confident number from a weak head is "
                                   "still a weak number."})

    return [
        StructuredTool.from_function(func=list_embedding_models, name="list_embedding_models", metadata=meta),
        StructuredTool.from_function(func=embed_region, name="embed_region", metadata=meta),
        StructuredTool.from_function(func=segment_region, name="segment_region", metadata=meta),
        StructuredTool.from_function(func=embedding_change, name="embedding_change", metadata=meta),
        StructuredTool.from_function(func=compare_regions, name="compare_regions", metadata=meta),
        StructuredTool.from_function(func=list_prediction_heads, name="list_prediction_heads", metadata=meta),
        StructuredTool.from_function(func=predict_for_region, name="predict_for_region", metadata=meta),
    ]




# --- zonal embeddings: pixels inside a polygon, aggregated ----------------------
RS_EMBED_PYTHON = os.getenv("RS_EMBED_PYTHON", "")
_ZONAL_TIMEOUT_S = float(os.getenv("RS_EMBED_ZONAL_TIMEOUT_S", "900"))
# Path to a checkout of rs-embed whose `src` should win over whatever the interpreter's
# editable install points at — how to run a fixed version without touching that checkout.
RS_EMBED_SRC = os.getenv("RS_EMBED_SRC", "")
# Distinct, colour-blind-safe hues for cluster classes (Okabe-Ito, alpha added).
_CLUSTER_COLORS = [
    [0, 114, 178, 190], [230, 159, 0, 190], [0, 158, 115, 190], [204, 121, 167, 190],
    [86, 180, 233, 190], [213, 94, 0, 190], [240, 228, 66, 190], [120, 120, 120, 190],
]


def _zonal_python() -> Optional[str]:
    """Interpreter that can import rs_embed. The agent's own cannot (deliberately)."""
    if RS_EMBED_PYTHON:
        return RS_EMBED_PYTHON
    for guess in ("/Users/yfkang/Documents/Github/rs-embed/rsembed/bin/python",):
        if Path(guess).exists():
            return guess
    return None


def run_zonal_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run the zonal worker under the rs-embed interpreter; return its JSON result."""
    import subprocess

    py = _zonal_python()
    if not py:
        return {"ok": False,
                "error": "no interpreter with rs_embed installed",
                "hint": "Set RS_EMBED_PYTHON to a python that can `import rs_embed` "
                        "(the rs-embed repo's venv), then retry."}
    worker = str(Path(__file__).with_name("rs_embed_zonal_worker.py"))
    out = Path(tempfile.mkdtemp(prefix="rsembed_zonal_")) / "result.json"
    payload = {**payload, "out_path": str(out)}
    # The agent process has torch loaded, so a second OpenMP runtime in a child that
    # inherits its state raised "OMP: Error #179: pthread_mutex_init failed" and wedged the
    # turn. Pin the child to one OpenMP thread and let a duplicate runtime load.
    env = {**os.environ, "OMP_NUM_THREADS": "1", "KMP_INIT_AT_FORK": "FALSE",
           "KMP_DUPLICATE_LIB_OK": "TRUE"}
    if RS_EMBED_SRC:
        env["PYTHONPATH"] = RS_EMBED_SRC + os.pathsep + env.get("PYTHONPATH", "")
    try:
        proc = subprocess.run([py, worker], input=json.dumps(payload), text=True,
                              capture_output=True, timeout=_ZONAL_TIMEOUT_S, env=env)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"zonal embedding exceeded {_ZONAL_TIMEOUT_S:.0f}s",
                "hint": "Lower max_tiles, raise tile_px, or pass fewer zones."}
    if not out.exists():
        tail = (proc.stderr or proc.stdout or "")[-600:]
        return {"ok": False, "error": "the zonal worker produced no result", "detail": tail}
    return json.loads(out.read_text())


def make_rs_embed_zonal_tools(default_input_file_ids: Optional[List[str]] = None) -> List[Any]:
    """Tools that aggregate per-pixel embeddings inside polygons."""
    from langchain_core.tools import StructuredTool

    meta = {"category": "geo"}

    def embed_zones(file_id: str, zone_id_field: Optional[str] = None, model: str = "gse",
                    year: int = 2022, clusters: int = 5, tile_px: int = 200,
                    max_tiles: int = 24, name: Optional[str] = None,
                    sibling_file_ids: Optional[List[str]] = None) -> str:
        """Give every POLYGON its own satellite-embedding vector, and map the result.

        Divides the polygons' area into pixels, embeds each pixel with a remote-sensing
        foundation model, and averages the pixels that fall INSIDE each polygon — so each
        zone (census tract, county, field, watershed, drawn box) gets one vector describing
        what it looks like from space. Works with any polygon layer: GeoJSON, shapefile,
        GeoPackage.

        Returns a CSV of per-zone vectors ready for machine learning (use fit_zone_model),
        and puts the zones on the map grouped into `clusters` look-alike groups. Each zone
        also carries `pixels` and `area_km2` — its support — plus the raw SUM, so zones can
        be rolled up to a coarser partition exactly.
        """
        tmp = None
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
            return json.dumps({"ok": False, "error": f"could not read {file_id}: {exc}"})

        png_path = Path(tempfile.mkdtemp(prefix="rsembed_zonal_")) / artifact_name(
            name, "png", default=f"{model}_zone_pixels")
        res = run_zonal_worker({"polygons_path": str(read_path), "zone_id_field": zone_id_field,
                                "model": model, "year": int(year), "tile_px": int(tile_px),
                                "max_tiles": int(max_tiles),
                                "clusters": max(2, min(int(clusters), len(_CLUSTER_COLORS))),
                                "image": True, "out_png": str(png_path)})
        if not res.get("ok"):
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

        # --- the map layer: look-alike groups, clustered BY THE WORKER (sklearn must not
        # run in this process; it already has torch loaded and a second OpenMP runtime
        # crashed the turn). Groups are the one view of a 64-dim vector worth looking at —
        # a choropleth of a single dimension is a picture of an arbitrary axis.
        layer = None
        cluster_note = None
        try:
            import geopandas as gpd

            gdf = gpd.read_file(read_path)
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            gdf = gdf.to_crs("EPSG:4326")
            ids = ([str(v) for v in gdf[zone_id_field]]
                   if zone_id_field and zone_id_field in gdf.columns
                   else [str(i) for i in range(len(gdf))])
            grouped = {z["zone_id"]: z for z in with_px if z.get("group")}
            keep = [i for i, zid in enumerate(ids) if zid in grouped]
            if keep:
                sub = gdf.iloc[keep][["geometry"]].copy()
                sub["zone_id"] = [ids[i] for i in keep]
                sub["pixels"] = [grouped[ids[i]]["pixels"] for i in keep]
                sub["area_km2"] = [round(grouped[ids[i]]["area_km2"], 4) for i in keep]
                sub["look_alike_group"] = [f"group {grouped[ids[i]]['group']}" for i in keep]
                gj = Path(tempfile.mkdtemp(prefix="rsembed_zonal_")) / artifact_name(
                    name, "geojson", default=f"{model}_zone_groups")
                sub.to_file(gj, driver="GeoJSON")
                rec = create_output_file_from_path(gj, filename=gj.name)
                present = sorted({str(v) for v in sub["look_alike_group"]})
                layer = {"url": rec.get("download_url"),
                         "label": f"{model} zone groups (k={len(present)})",
                         "render": "categories", "style_by": "look_alike_group",
                         "source": "analysis", "count": len(keep),
                         "legend": [{"label": g,
                                     "color": _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)]}
                                    for i, g in enumerate(present)]}
        except Exception as exc:  # noqa: BLE001
            cluster_note = f"zones computed but not mapped: {type(exc).__name__}: {exc}"[:200]

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
                    "a much larger one, because pooling averages away variance.",
        }
        if res.get("tiles_capped"):
            out["truncated"] = (f"stopped after max_tiles={max_tiles} of {res['tiles_planned']} "
                                f"tiles — zones outside those tiles have no pixels")
        if res.get("pixel_size_warnings"):
            out["pixel_size_warnings"] = res["pixel_size_warnings"]
        if res.get("tile_errors"):
            out["tile_errors"] = res["tile_errors"]
        if cluster_note:
            out["cluster_note"] = cluster_note
        # Two views, both delivered: the pixel-level embedding masked to the shapes (what the
        # zone vectors are computed FROM), and the zones grouped by those vectors. Asking for
        # "the embedding of these polygons" and getting only a group colour hid the actual data.
        layers = []
        img = res.get("image") or {}
        if img.get("path") and Path(img["path"]).exists() and img.get("bounds"):
            rec_png = create_output_file_from_path(Path(img["path"]),
                                                   filename=Path(img["path"]).name)
            out["pixel_image"] = {"file_id": rec_png["file_id"],
                                  "download_url": rec_png.get("download_url"),
                                  "size_bytes": rec_png.get("size_bytes"),
                                  "size_px": img.get("size_px"),
                                  "pixels_shown": img.get("pixels_shown"),
                                  "colour": img.get("colour")}
            layers.append(_raster_layer(rec_png, [float(v) for v in img["bounds"]],
                                        f"{model} pixel embedding in zones"))
        elif img.get("error"):
            out["image_note"] = img["error"]
        if layer:
            layers.append(layer)
        if layers:
            out["map_layer"] = layers[0]
            if len(layers) > 1:
                out["map_layers"] = layers
            out["on_map"] = True
        return json.dumps(out)


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

            # Every numeric step runs in the worker's interpreter: sklearn in this process
            # meets a second OpenMP runtime alongside torch and takes the turn down.
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
                              "label": f"{label_column} predicted from embeddings",
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

    return [StructuredTool.from_function(func=embed_zones, name="embed_zones", metadata=meta),
            StructuredTool.from_function(func=fit_zone_model, name="fit_zone_model", metadata=meta)]


__all__ = ["make_rs_embed_tools", "make_rs_embed_zonal_tools", "run_zonal_worker", "RS_EMBED_URL"]
