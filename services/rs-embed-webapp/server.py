"""FastAPI backend for the rs-embed interactive web app.

Serves a Leaflet map UI (``index.html``) and three JSON endpoints that wrap the
rs-embed Python API:

* ``GET  /api/models``   — list available models (precomputed vs on-the-fly).
* ``POST /api/preview``  — fetch a Sentinel-2 RGB quicklook for the selected ROI
  (no model run) → the "S2 image first" popup-bubble step.
* ``POST /api/embed``    — run the chosen models on the ROI and return, per model,
  a PCA-RGB thumbnail of the grid embedding **and** pooled-vector stats.

Run (from the repo root, in the rsembed venv with GEE authenticated)::

    EARTHENGINE_PROJECT=ee-yfkang \
      rsembed/bin/python -m uvicorn examples.webapp.server:app --port 8000

Then open http://localhost:8000.

The ROI/temporal/model contract mirrors the demo notebook:
* point  → ``PointBuffer(lon, lat, buffer_m=2048)``  (a ~4 km box)
* bbox   → ``BBox(minlon, minlat, maxlon, maxlat)``
* a year-range [y0, y1] maps to ``TemporalSpec.year(...)`` for annual precomputed
  models and ``TemporalSpec.range("{y0}-06-01", "{y1}-09-01")`` for on-the-fly ones.
"""

from __future__ import annotations

import base64
import hashlib
import io
import math
import json
import os
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Reuse the demo helpers (pca_rgb / to_dhw / pooled_vector) shipped with examples/.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # examples/
import iguide_demo_helpers as H  # noqa: E402

# --- rs-embed + Earth Engine (initialized once, lazily) ---------------------
PRECOMPUTED = {"gse", "tessera", "copernicus"}
# On-the-fly models that consume a *time series* (multiple frames) rather than a
# single composite frame — either a static ModelInputSpec.temporal_mode == "multi"
# (agrifm/anysat) or a window-adaptive temporal_mode defaulting to "auto"→multi for
# multi-month windows (galileo/olmoearth/prithvi). Everything else takes one frame.
TIMESERIES = {"agrifm", "anysat", "galileo", "olmoearth", "prithvi"}
COPERNICUS_FIXED_YEAR = 2021  # copernicus product currently covers only 2021


def _input_kind(model: str) -> str | None:
    """Input contract for on-the-fly models: 'timeseries' | 'single' (None if precomputed)."""
    if model in PRECOMPUTED:
        return None
    return "timeseries" if model in TIMESERIES else "single"


# Saved embedding packages (.npz) live here; served back via /api/download.
DOWNLOADS = HERE / "downloads"
DOWNLOADS.mkdir(exist_ok=True)
ASSETS = HERE / "assets"
ASSETS.mkdir(exist_ok=True)  # holds logo.png etc., served at /assets

# Pretrained downstream heads (built by build_demo_cache.py --only heads).
_HEADS: dict[str, Any] = {}  # model -> loaded regressor (lazy)


def _heads_dir() -> Path:
    import iguide_demo_helpers as _H

    return _H.DemoCache.find(HERE).root / "heads"


def _heads_meta() -> dict[str, Any] | None:
    p = _heads_dir() / "heads_meta.json"
    return json.loads(p.read_text()) if p.exists() else None


def _load_head(model: str):
    if model in _HEADS:
        return _HEADS[model]
    meta = _heads_meta()
    task = (meta or {}).get("task", "maize_yield")
    import joblib

    p = _heads_dir() / f"{task}__{model}.pkl"
    if not p.exists():
        return None
    reg = joblib.load(p)
    _HEADS[model] = reg
    return reg


def _predict_one(reg, vec: np.ndarray, meta: dict, info: dict, model: str) -> dict:
    """Run one head; works for both regression and classification heads."""
    v = np.nan_to_num(np.asarray(vec, dtype=np.float32)).reshape(1, -1)
    if hasattr(reg, "predict_proba"):  # classification head
        proba = reg.predict_proba(v)[0]
        classes = list(getattr(reg, "classes_", [0, 1]))
        pidx = classes.index(1) if 1 in classes else len(classes) - 1
        p = float(proba[pidx])
        names = meta.get("classes", ["negative", "positive"])
        return {
            "model": model,
            "ok": True,
            "kind": "classification",
            "prediction": round(p, 3),
            "label_pred": names[1] if p >= 0.5 else names[0],
            "score": info.get("score"),
            "score_name": info.get("score_name", "accuracy"),
        }
    pred = float(reg.predict(v)[0])  # regression head
    lo, hi = info.get("y_min"), info.get("y_max")
    return {
        "model": model,
        "ok": True,
        "kind": "regression",
        "prediction": round(pred, 3),
        "in_range": bool(lo is None or (lo <= pred <= hi)),
        "score": info.get("score", info.get("r2")),
        "score_name": info.get("score_name", "R²"),
    }


GRID_SAVE_MAX_CELLS = 300 * 300  # cell budget for a grid written into the package


def _grid_stride(gh: int, gw: int, max_cells: int) -> int:
    """Smallest stride that brings a gh x gw grid under ``max_cells``.

    The package used to DROP a grid that did not fit, which made the export cap a silent
    capability cliff: a caller asking to cluster or difference an embedding got a file holding
    the pooled vector and no pixels, with nothing in the array to say why. The default footprint
    already exceeded it (a 2048 m buffer is a 4096 m square, 410x411 = 168,100 cells, 1.87x over),
    so the DEFAULT case exported nothing. Decimating keeps every package usable and turns the cap
    into a resolution, which the manifest then states.
    """
    s = 1
    while ((gh + s - 1) // s) * ((gw + s - 1) // s) > max_cells:
        s += 1
    return s


_ee_ready = False


def _ensure_ee() -> None:
    global _ee_ready
    if _ee_ready:
        return
    import ee

    project = os.environ.get("EARTHENGINE_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    ee.Initialize(project=project) if project else ee.Initialize()
    _ee_ready = True


def _rs():
    # Lazy access to the rs-embed API as a name->symbol dict. We resolve via
    # getattr over string names (not `from rs_embed import ...; return locals()`)
    # so ruff's F401 cannot flag the symbols as "unused" — the old form got its
    # imports auto-stripped by `ruff --fix` (pre-commit), collapsing this to an
    # empty dict and 500-ing every endpoint. Keep it getattr-based.
    import rs_embed

    names = (
        "BBox",
        "ExportConfig",
        "ExportTarget",
        "OutputSpec",
        "PointBuffer",
        "SensorSpec",
        "TemporalSpec",
        "export_batch",
        "get_embedding",
        "inspect_provider_patch",
        "list_models",
        "load_export",
    )
    return {n: getattr(rs_embed, n) for n in names}


# --- request models ---------------------------------------------------------
class Geometry(BaseModel):
    type: str  # "point" | "bbox"
    lon: float | None = None
    lat: float | None = None
    minlon: float | None = None
    minlat: float | None = None
    maxlon: float | None = None
    maxlat: float | None = None


class PreviewReq(BaseModel):
    geometry: Geometry
    start: str = "2022-06"  # "YYYY-MM" inclusive start month
    end: str = "2022-09"  # "YYYY-MM" inclusive end month
    buffer_m: int = 2048  # single-click point buffer (~4 km box)


class EmbedReq(PreviewReq):
    models: list[str] = []
    # Cell budget for the per-pixel grid in the export package. The grid is decimated to
    # fit rather than dropped, so raising this buys resolution at the cost of file size.
    grid_max_cells: int = 0  # 0 means GRID_SAVE_MAX_CELLS


class PredictReq(PreviewReq):
    models: list[str] = []  # embedding models to run through their heads


class SimilarityReq(BaseModel):
    geometries: list[Geometry] = []  # exactly two ROIs to compare
    start: str = "2022-06"
    end: str = "2022-09"
    buffer_m: int = 2048
    model: str = "gse"  # single model used to embed both ROIs


class SegmentReq(PreviewReq):
    model: str = "gse"  # model whose grid embedding is clustered
    k: int = 6  # number of k-means clusters (land-cover-like regions)


class ChangeReq(BaseModel):
    geometry: Geometry  # single ROI tracked across years
    buffer_m: int = 2048
    model: str = "gse"  # model used to embed each year
    years: list[int] = []  # years to compare; baseline = first


class PredictExampleReq(PreviewReq):
    # Corn-yield demo over several embeddings (heads framed as trained on SPAM / Illinois).
    models: list[str] = ["gse", "tessera", "olmoearth", "agrifm", "terramind", "thor"]


# --- helpers -----------------------------------------------------------------
def _spatial(rs: dict, g: Geometry, buffer_m: int = 2048):
    if g.type == "point":
        if g.lon is None or g.lat is None:
            raise ValueError("point geometry requires lon/lat")
        return rs["PointBuffer"](lon=float(g.lon), lat=float(g.lat), buffer_m=int(buffer_m))
    if g.type == "bbox":
        for k in ("minlon", "minlat", "maxlon", "maxlat"):
            if getattr(g, k) is None:
                raise ValueError("bbox geometry requires minlon/minlat/maxlon/maxlat")
        return rs["BBox"](
            minlon=float(g.minlon),
            minlat=float(g.minlat),
            maxlon=float(g.maxlon),
            maxlat=float(g.maxlat),
        )
    raise ValueError(f"unknown geometry type: {g.type!r}")


def _ym(s: str) -> tuple[int, int]:
    """Parse a 'YYYY-MM' string into (year, month)."""
    parts = str(s).split("-")
    return int(parts[0]), int(parts[1])


def _range_dates(start: str, end: str) -> tuple[str, str, int, int]:
    """('YYYY-MM','YYYY-MM') → (start_day, end_exclusive_day, start_year, end_year).

    The end month is inclusive, so the returned end is the first day of the
    month *after* ``end`` (rs-embed ranges are half-open ``[start, end)``).
    """
    y0, m0 = _ym(start)
    y1, m1 = _ym(end)
    if (y0, m0) > (y1, m1):
        (y0, m0), (y1, m1) = (y1, m1), (y0, m0)
    start_d = f"{y0:04d}-{m0:02d}-01"
    em, ey = m1 + 1, y1
    if em > 12:
        em, ey = 1, ey + 1
    return start_d, f"{ey:04d}-{em:02d}-01", y0, y1


def _temporal(rs: dict, model: str, start: str, end: str):
    start_d, end_excl, _y0, y1 = _range_dates(start, end)
    if model in PRECOMPUTED:
        if model == "copernicus":
            return rs["TemporalSpec"].year(COPERNICUS_FIXED_YEAR)
        return rs["TemporalSpec"].year(y1)  # latest year in range for annual products
    return rs["TemporalSpec"].range(start_d, end_excl)


def _png_b64(rgb_float_hwc: np.ndarray) -> str:
    """(H,W,3) float in [0,1] → base64 PNG data URI."""
    from PIL import Image

    arr = np.clip(np.asarray(rgb_float_hwc) * 255.0, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _s2_rgb_to_float(chw: np.ndarray) -> np.ndarray:
    """Raw S2 SR (B4,B3,B2) CHW → percentile-stretched (H,W,3) float for display."""
    x = np.asarray(chw, dtype=np.float32)
    if x.ndim != 3 or x.shape[0] < 3:
        raise ValueError(f"expected 3-band CHW, got {x.shape}")
    rgb = np.transpose(x[:3], (1, 2, 0))  # H,W,3
    lo = np.nanpercentile(rgb, 2, axis=(0, 1))
    hi = np.nanpercentile(rgb, 98, axis=(0, 1))
    return np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)


# --- app ---------------------------------------------------------------------
app = FastAPI(title="rs-embed web demo")
app.mount("/assets", StaticFiles(directory=str(ASSETS)), name="assets")


@app.get("/")
def index() -> FileResponse:
    # no-store so iterative edits always show on a plain refresh
    return FileResponse(str(HERE / "index.html"), headers={"Cache-Control": "no-store"})


@app.get("/api/models")
def api_models() -> Any:
    try:
        rs = _rs()
        ids = rs["list_models"]()
        models = [
            {
                "id": m,
                "type": "precomputed" if m in PRECOMPUTED else "onthefly",
                "input": _input_kind(m),
            }
            for m in sorted(ids)
        ]
        return {
            "models": models,
            "precomputed": sorted(PRECOMPUTED),
            "timeseries": sorted(TIMESERIES),
        }
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": repr(e)}, status_code=500)


@app.post("/api/preview")
def api_preview(req: PreviewReq) -> Any:
    """Sentinel-2 RGB quicklook for the ROI — no model is run."""
    try:
        _ensure_ee()
        rs = _rs()
        spatial = _spatial(rs, req.geometry, req.buffer_m)
        start_d, end_excl, _y0, _y1 = _range_dates(req.start, req.end)
        temporal = rs["TemporalSpec"].range(start_d, end_excl)
        sensor = rs["SensorSpec"](
            collection="COPERNICUS/S2_SR_HARMONIZED",
            bands=("B4", "B3", "B2"),
            scale_m=10,
            cloudy_pct=30,
            composite="median",
        )
        rep = rs["inspect_provider_patch"](
            spatial=spatial,
            temporal=temporal,
            sensor=sensor,
            backend="gee",
            name="s2_preview",
            return_array=True,
        )
        chw = rep.get("array_chw")
        if chw is None:
            return JSONResponse(
                {"error": "no imagery returned for this ROI/time."}, status_code=502
            )
        rgb = _s2_rgb_to_float(chw)
        return {
            "image": _png_b64(rgb),
            "shape": [int(s) for s in np.asarray(chw).shape],
            "ok": bool(rep.get("ok", True)),
        }
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse({"error": repr(e)}, status_code=500)


def _meta_json(meta: Any) -> dict[str, Any]:
    """An embedder's meta as plain JSON, or {} if it has none.

    Mirrors /api/zones, which already sends its meta verbatim: it carries the provenance a
    caller must not invent — which imagery, at what scale, composited how, over which dates,
    from which model variant. A caller that does not receive it does not stop answering "what
    resolution was that?"; it answers from the defaults, which is right until a default moves.

    default=str rather than letting a numpy scalar or an enum break the response: the point of
    meta is that it always arrives.
    """
    if not meta:
        return {}
    try:
        out = json.loads(json.dumps(dict(meta), default=str))
        return out if isinstance(out, dict) else {}
    except Exception:  # noqa: BLE001 - provenance is never worth failing the embedding over
        return {}


@app.post("/api/embed")
def api_embed(req: EmbedReq) -> Any:
    """Run each selected model → PCA-RGB thumbnail + pooled stats, and save a
    downloadable ``.npz`` embedding package for use in the notebook."""
    try:
        _ensure_ee()
        rs = _rs()
        spatial = _spatial(rs, req.geometry, req.buffer_m)

        models = list(req.models)
        results: list[dict[str, Any]] = []
        pkg: dict[str, np.ndarray] = {}  # arrays written into the package
        pkg_models: list[dict[str, Any]] = []  # per-model metadata for the manifest

        def _ok(m: str, grid: np.ndarray, meta: Any = None) -> None:
            vec = H.pooled_vector(grid)
            gh, gw = int(grid.shape[1]), int(grid.shape[2])
            mtype = "precomputed" if m in PRECOMPUTED else "onthefly"
            info = _meta_json(meta)
            pkg[f"pooled__{m}"] = vec.astype(np.float32)
            budget = int(getattr(req, "grid_max_cells", 0) or GRID_SAVE_MAX_CELLS)
            stride = _grid_stride(gh, gw, budget)
            sub = grid[:, ::stride, ::stride] if stride > 1 else grid
            pkg[f"grid__{m}"] = sub.astype(np.float32)
            saved = True
            pkg_models.append(
                {
                    "model": m,
                    "type": mtype,
                    "dim": int(vec.shape[0]),
                    "grid_hw": [gh, gw],
                    "grid_saved": bool(saved),
                    # The grid in the package is this much coarser than grid_hw. 1 means it is
                    # the native grid; anything higher and a consumer is looking at every Nth
                    # cell, which it has to know before quoting a per-pixel result.
                    "grid_stride": int(stride),
                    "grid_saved_hw": [int(sub.shape[1]), int(sub.shape[2])],
                    # In the manifest too: the .npz outlives this response, and a vector whose
                    # sensor and dates are unrecorded cannot be compared with a later one.
                    "meta": info,
                }
            )
            results.append(
                {
                    "model": m,
                    "type": mtype,
                    "ok": True,
                    "image": _png_b64(H.pca_rgb(grid)),
                    "dim": int(vec.shape[0]),
                    "grid_hw": [gh, gw],
                    "norm": float(np.linalg.norm(vec)),
                    "vector_preview": [round(float(v), 4) for v in vec[:32]],
                    "meta": info,
                }
            )

        def _err(m: str, msg: Any) -> None:
            results.append(
                {
                    "model": m,
                    "type": "precomputed" if m in PRECOMPUTED else "onthefly",
                    "ok": False,
                    "error": str(msg)[:400],
                }
            )

        # ── single model → get_embedding ; multiple → export_batch (shared fetch) ──
        if len(models) <= 1:
            compute = "get_embedding"
            for m in models:
                try:
                    emb = rs["get_embedding"](
                        m,
                        spatial=spatial,
                        temporal=_temporal(rs, m, req.start, req.end),
                        output=rs["OutputSpec"].grid(),
                        backend="auto",
                    )
                    _ok(m, H.to_dhw(emb.data), getattr(emb, "meta", None))
                except Exception as e:  # noqa: BLE001
                    _err(m, repr(e))
        else:
            compute = "export_batch"
            start_d, end_excl, _y0, _y1 = _range_dates(req.start, req.end)
            # one temporal for all models; gse/annual products convert range→year internally
            temporal = rs["TemporalSpec"].range(start_d, end_excl)
            tmp = DOWNLOADS / f"_export_{uuid.uuid4().hex[:8]}.npz"
            try:
                rs["export_batch"](
                    spatials=[spatial],
                    temporal=temporal,
                    models=models,
                    output=rs["OutputSpec"].grid(),
                    target=rs["ExportTarget"].combined(str(tmp)),
                    config=rs["ExportConfig"](
                        save_inputs=False,
                        save_embeddings=True,
                        continue_on_error=True,
                        show_progress=False,
                    ),
                    backend="auto",
                )
                er = rs["load_export"](str(tmp))
                # Read embeddings straight from the .npz arrays keyed by model.
                # (rs-embed's combined manifest can omit a model's entry even when
                # its `embeddings__<model>` array was written, so load_export alone
                # under-reports; the arrays are the source of truth.)
                with np.load(tmp, allow_pickle=True) as z:
                    by_model = {
                        k[len("embeddings__") :]: k for k in z.files if k.startswith("embeddings__")
                    }
                    for m in models:
                        arr = np.asarray(z[by_model[m]]) if m in by_model else None
                        if arr is None:  # fall back to load_export's view
                            mr = er.models.get(m)
                            arr = (
                                np.asarray(mr.embeddings)
                                if (mr and mr.embeddings is not None)
                                else None
                            )
                        if (
                            arr is not None
                            and arr.ndim >= 3
                            and arr.shape[0] >= 1
                            and np.isfinite(arr).any()
                        ):
                            # export_batch's per-model record, when it has one: the arrays
                            # are read straight from the .npz above, so this is the only place
                            # the batch path can pick provenance up.
                            _ok(m, H.to_dhw(arr[0]),  # (1,C,H,W) → (C,H,W)
                                getattr(er.models.get(m), "meta", None))
                        else:
                            mr = er.models.get(m)
                            _err(m, getattr(mr, "error", None) or "no embedding produced")
            finally:
                for p in (tmp, tmp.with_suffix(".json")):
                    try:
                        p.unlink()
                    except OSError:
                        pass

        download = None
        package = None
        if pkg:
            manifest = {
                "geometry": req.geometry.model_dump(),
                "start": req.start,
                "end": req.end,
                "buffer_m": req.buffer_m,
                "models": pkg_models,
                "compute": compute,
                "note": "Load with: import iguide_demo_helpers as H; H.load_embedding_package(path)",
            }
            pkg["meta"] = np.array(json.dumps(manifest))
            fname = f"rsembed_pkg_{uuid.uuid4().hex[:10]}.npz"
            np.savez_compressed(DOWNLOADS / fname, **pkg)
            download = f"/api/download/{fname}"
            package = {
                "filename": fname,
                "compute": compute,
                "models": [mm["model"] for mm in pkg_models],
                "grids_saved": [mm["model"] for mm in pkg_models if mm["grid_saved"]],
            }
        return {
            "results": results,
            "download_url": download,
            "package": package,
            "compute": compute,
        }
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse({"error": repr(e)}, status_code=500)


@app.get("/api/heads")
def api_heads() -> Any:
    """List available pretrained downstream heads (for the 'Use' tab)."""
    meta = _heads_meta()
    if not meta:
        return {
            "task": None,
            "models": [],
            "note": "No pretrained heads. Run: python build_demo_cache.py --only heads",
        }
    root, task = _heads_dir(), meta.get("task", "maize_yield")
    models = [
        {"model": m, **info}
        for m, info in meta.get("models", {}).items()
        if (root / f"{task}__{m}.pkl").exists()
    ]
    return {
        "task": task,
        "kind": meta.get("kind", "regression"),
        "label": meta.get("label"),
        "units": meta.get("units"),
        "region": meta.get("region"),
        "classes": meta.get("classes"),
        "models": models,
    }


@app.post("/api/predict")
def api_predict(req: PredictReq) -> Any:
    """ROI → embedding → pretrained head → predicted value, per selected model."""
    try:
        meta = _heads_meta()
        if not meta:
            return JSONResponse({"error": "no pretrained heads available"}, status_code=404)
        _ensure_ee()
        rs = _rs()
        spatial = _spatial(rs, req.geometry, req.buffer_m)
        results = []
        for m in req.models or []:
            reg = _load_head(m)
            info = meta.get("models", {}).get(m, {})
            if reg is None:
                results.append(
                    {"model": m, "ok": False, "error": "no pretrained head for this model"}
                )
                continue
            try:
                emb = rs["get_embedding"](
                    m,
                    spatial=spatial,
                    temporal=_temporal(rs, m, req.start, req.end),
                    output=rs["OutputSpec"].pooled(),
                    backend="auto",
                )
                results.append(_predict_one(reg, H.pooled_vector(emb.data), meta, info, m))
            except Exception as e:  # noqa: BLE001
                results.append({"model": m, "ok": False, "error": repr(e)[:300]})
        return {
            "task": meta.get("task"),
            "kind": meta.get("kind", "regression"),
            "label": meta.get("label"),
            "units": meta.get("units"),
            "region": meta.get("region"),
            "results": results,
        }
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse({"error": repr(e)}, status_code=500)


@app.post("/api/similarity")
def api_similarity(req: SimilarityReq) -> Any:
    """Two ROIs → one model → pooled vectors → cosine similarity.

    The retrieval primitive: the same pooled descriptor that powers nearest-
    neighbour search, here used to score how alike any two places look.
    """
    try:
        if len(req.geometries) != 2:
            return JSONResponse(
                {"error": "similarity needs exactly two regions (A and B)"}, status_code=400
            )
        _ensure_ee()
        rs = _rs()
        m = req.model
        temporal = _temporal(rs, m, req.start, req.end)
        vecs: list[np.ndarray] = []
        regions: list[dict[str, Any]] = []
        for i, g in enumerate(req.geometries):
            spatial = _spatial(rs, g, req.buffer_m)
            emb = rs["get_embedding"](
                m,
                spatial=spatial,
                temporal=temporal,
                output=rs["OutputSpec"].grid(),  # pool via nanmean (robust to fill/NaN)
                backend="auto",
            )
            vec = H.pooled_vector(emb.data)
            vecs.append(vec)
            regions.append(
                {
                    "label": ["A", "B"][i],
                    "dim": int(vec.shape[0]),
                    "norm": float(np.linalg.norm(vec)),
                }
            )
        a = H.l2_normalize(vecs[0])
        b = H.l2_normalize(vecs[1])
        cos = float(np.dot(a, b))
        return {
            "model": m,
            "cosine": round(cos, 4),
            "distance": round(1.0 - cos, 4),
            "dim": int(vecs[0].shape[0]),
            "regions": regions,
        }
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse({"error": repr(e)}, status_code=500)


@app.post("/api/segment")
def api_segment(req: SegmentReq) -> Any:
    """One ROI → grid embedding → k-means → a colourised land-cover mask.

    Unsupervised segmentation: the grid embedding is clustered into ``k`` groups
    and returned as an (H,W) RGB PNG, ready to drape over the ROI as a
    semi-transparent overlay.
    """
    try:
        _ensure_ee()
        rs = _rs()
        m = req.model
        k = max(2, min(10, int(req.k)))
        spatial = _spatial(rs, req.geometry, req.buffer_m)
        emb = rs["get_embedding"](
            m,
            spatial=spatial,
            temporal=_temporal(rs, m, req.start, req.end),
            output=rs["OutputSpec"].grid(),
            backend="auto",
        )
        grid = H.to_dhw(emb.data)
        labels, rgb = H.kmeans_landcover(grid, k=k)
        gh, gw = int(labels.shape[0]), int(labels.shape[1])
        # per-cluster pixel share (largest first) for a readable legend
        uniq, counts = np.unique(labels, return_counts=True)
        total = int(labels.size) or 1
        pal = (np.asarray(H.LANDCOVER_PALETTE) * 255).astype(int)
        legend = [
            {
                "cluster": int(c),
                "rgb": [int(v) for v in pal[int(c) % len(pal)]],
                "frac": round(int(cnt) / total, 3),
            }
            for c, cnt in sorted(
                zip(uniq.tolist(), counts.tolist(), strict=True), key=lambda t: -t[1]
            )
        ]
        return {
            "model": m,
            "k": k,
            "grid_hw": [gh, gw],
            "image": _png_b64(rgb),
            "legend": legend,
        }
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse({"error": repr(e)}, status_code=500)


@app.post("/api/change")
def api_change(req: ChangeReq) -> Any:
    """One ROI across several years → pooled embedding per year → change curve.

    Returns the cosine distance (``1 - cos``) of each year's pooled embedding from
    the baseline (first) year; a spike marks the year the place changed.
    """
    try:
        years = sorted({int(y) for y in req.years})
        if len(years) < 2:
            return JSONResponse({"error": "change needs at least two years"}, status_code=400)
        _ensure_ee()
        rs = _rs()
        m = req.model
        spatial = _spatial(rs, req.geometry, req.buffer_m)
        vecs: list[np.ndarray] = []
        used: list[int] = []
        errors: list[dict[str, Any]] = []
        for y in years:
            try:
                emb = rs["get_embedding"](
                    m,
                    spatial=spatial,
                    temporal=_temporal(rs, m, f"{y}-01", f"{y}-12"),
                    output=rs["OutputSpec"].grid(),  # pool via nanmean (robust to fill/NaN)
                    backend="auto",
                )
                vecs.append(H.pooled_vector(emb.data))
                used.append(y)
            except Exception as e:  # noqa: BLE001
                errors.append({"year": y, "error": repr(e)[:200]})
        if len(vecs) < 2:
            return JSONResponse(
                {"error": "fewer than two years produced an embedding", "errors": errors},
                status_code=502,
            )
        dist = H.change_curve(vecs, baseline_index=0, metric="cosine")
        return {
            "model": m,
            "baseline": used[0],
            "years": used,
            "distances": [round(float(d), 4) for d in dist],
            "errors": errors,
        }
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse({"error": repr(e)}, status_code=500)


def _demo_yield(model: str, vec: np.ndarray) -> float:
    """A deterministic, illustrative corn-yield (mt/ha) from a pooled embedding.

    Not a trained model: a fixed per-model random projection of the L2-normalised
    pooled vector, squashed into ~[8, 14] mt/ha. Same place+model → same value;
    different models differ slightly; nearby places read similarly. Lets the
    'Predict' app demo the *use* of embeddings before a real head exists.
    """
    v = H.l2_normalize(np.nan_to_num(np.asarray(vec, dtype=np.float32)))
    seed = int.from_bytes(hashlib.md5(model.encode()).digest()[:4], "big")  # stable per model
    w = np.random.default_rng(seed).standard_normal(v.shape[0]).astype(np.float32)
    w /= np.linalg.norm(w) + 1e-8
    raw = float(np.dot(w, v))  # ~[-1, 1] for unit vectors
    return round(float(np.clip(11.0 + 3.0 * np.tanh(raw * 1.5), 6.0, 16.0)), 2)


@app.post("/api/predict_example")
def api_predict_example(req: PredictExampleReq) -> Any:
    """Illustrative corn-yield demo: ROI → pooled embedding → a deterministic
    demo head (no training), per selected model. For showcasing how embeddings
    feed a downstream task before a real head is built."""
    try:
        _ensure_ee()
        rs = _rs()
        spatial = _spatial(rs, req.geometry, req.buffer_m)
        results = []
        for m in req.models or []:
            try:
                emb = rs["get_embedding"](
                    m,
                    spatial=spatial,
                    temporal=_temporal(rs, m, req.start, req.end),
                    output=rs["OutputSpec"].grid(),  # pool via nanmean (robust to fill/NaN)
                    backend="auto",
                )
                vec = H.pooled_vector(emb.data)
                results.append(
                    {
                        "model": m,
                        "ok": True,
                        "kind": "regression",
                        "prediction": _demo_yield(m, vec),
                        "demo": True,
                    }
                )
            except Exception as e:  # noqa: BLE001
                results.append({"model": m, "ok": False, "error": repr(e)[:200]})
        return {
            "task": "corn_yield_demo",
            "kind": "regression",
            "label": "corn yield",
            "units": "mt/ha",
            "region": "SPAM · Illinois",
            "demo": True,
            "results": results,
        }
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse({"error": repr(e)}, status_code=500)


@app.post("/api/predict_package")
async def api_predict_package(file: UploadFile = File(...)) -> Any:
    """Run an uploaded embedding package (.npz from /api/embed) through the heads."""
    try:
        meta = _heads_meta()
        if not meta:
            return JSONResponse({"error": "no pretrained heads available"}, status_code=404)
        z = np.load(io.BytesIO(await file.read()), allow_pickle=True)
        pooled = {
            k[len("pooled__") :]: np.asarray(z[k]) for k in z.files if k.startswith("pooled__")
        }
        if not pooled:
            return JSONResponse({"error": "no pooled vectors found in package"}, status_code=400)
        results = []
        for m, vec in pooled.items():
            reg = _load_head(m)
            info = meta.get("models", {}).get(m, {})
            if reg is None:
                results.append(
                    {"model": m, "ok": False, "error": "no pretrained head for this model"}
                )
                continue
            try:
                results.append(_predict_one(reg, np.asarray(vec), meta, info, m))
            except Exception as e:  # noqa: BLE001
                results.append({"model": m, "ok": False, "error": repr(e)[:300]})
        return {
            "task": meta.get("task"),
            "kind": meta.get("kind", "regression"),
            "label": meta.get("label"),
            "units": meta.get("units"),
            "results": results,
        }
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse({"error": repr(e)}, status_code=500)


@app.get("/api/download/{name}")
def api_download(name: str) -> Any:
    """Serve a previously saved embedding package (path-traversal safe)."""
    if not name.startswith("rsembed_pkg_") or "/" in name or "\\" in name:
        return JSONResponse({"error": "invalid package name"}, status_code=400)
    path = DOWNLOADS / name
    if not path.exists():
        return JSONResponse({"error": "package not found"}, status_code=404)
    return FileResponse(str(path), media_type="application/octet-stream", filename=name)


# --- zones: one vector per polygon -------------------------------------------------
# The agent used to do this itself, in a subprocess under a second interpreter, because the
# library had no zones API: it swept tiles, derived the 3857 affine, rasterised and aggregated
# by hand. rs_embed.embed_zones does all of that now, so the work belongs here — where the
# model runtime, the Earth Engine credentials and the warm HF cache already live — and the
# agent just posts polygons and gets vectors back.
class ZonesReq(BaseModel):
    zones_geojson: Dict[str, Any]        # a FeatureCollection, sent inline by the caller
    model: str = "gse"
    year: int = 2022
    # A DATE RANGE, which /api/embed has always accepted and this endpoint did not. Without
    # it "the clay embedding of Urbana for 2025-03-01..2025-05-01" could be answered only as a
    # rectangle (embed_region, which takes start/end) or as a whole-year composite over the
    # city polygon — the caller had to trade the boundary against the dates, silently.
    # Both must be given; either alone falls back to `year`.
    start: Optional[str] = None
    end: Optional[str] = None
    zone_id_field: Optional[str] = None
    tile_px: int = 256
    max_tiles: Optional[int] = None
    image: bool = True
    image_max_px: int = 4_000_000


_R_MERCATOR = 6378137.0        # the sphere EPSG:3857 — and the provider's grid — is defined on


def _to_merc(lon: float, lat: float) -> tuple:
    x = math.radians(lon) * _R_MERCATOR
    y = _R_MERCATOR * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def _to_lonlat(x: float, y: float) -> tuple:
    lon = math.degrees(x / _R_MERCATOR)
    lat = math.degrees(2 * math.atan(math.exp(y / _R_MERCATOR)) - math.pi / 2)
    return lon, lat


def _render_masked_pca(arr_dhw: Any, box_3857: tuple, gdf: Any) -> Dict[str, Any]:
    """PCA the grid to RGB, then cut it to the polygons. Kept apart from the fetch so the
    georeferencing can be tested without Earth Engine — which is where the bugs are.

    The affine comes from the REQUESTED EPSG:3857 bounds and the RETURNED shape. The grid
    arrives as a bare (D, H, W) array with no transform, and its ``scale_m`` is Web Mercator
    metres, which run 1/cos(latitude) longer than metres on the ground: reading them as ground
    metres offsets every zone boundary. The implied pixel size is reported so a mismatch is
    visible rather than silent.
    """
    import numpy as np
    from PIL import Image
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    x0, y0, x1, y1 = box_3857
    arr = H.to_dhw(arr_dhw)
    _d, h, w = arr.shape
    transform = from_bounds(x0, y0, x1, y1, w, h)
    rgb = H.pca_rgb(arr)
    shapes = [(geom, 1) for geom in gdf.to_crs("EPSG:3857").geometry if geom is not None]
    # all_touched=False, matching rs_embed.embed_zones own rasterisation: the answer quotes
    # a pixel count, and the picture has to be of THOSE pixels, not of a slightly fatter shape.
    mask = rasterize(shapes, out_shape=(h, w), transform=transform, fill=0, dtype="uint8",
                     all_touched=False)
    rgba = np.dstack([np.clip(np.asarray(rgb) * 255.0, 0, 255).astype(np.uint8),
                      (mask * 255).astype(np.uint8)])
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    lo_lon, lo_lat = _to_lonlat(x0, y0)
    hi_lon, hi_lat = _to_lonlat(x1, y1)
    return {
        "png": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
        "bounds": [round(lo_lon, 6), round(lo_lat, 6), round(hi_lon, 6), round(hi_lat, 6)],
        "size_px": [int(h), int(w)],
        "pixels_shown": int(mask.sum()),
        "pixel_size_m_mercator": [round((x1 - x0) / w, 3), round((y1 - y0) / h, 3)],
        "colour": "PCA of the embedding dimensions to RGB: similar colour means a similar "
                  "vector. The axes are arbitrary, so the colours carry no units.",
    }


def _zone_pixel_image(gdf: Any, model: str, temporal: Any, scale_m: Any,
                      max_px: int) -> Dict[str, Any]:
    """The pixels themselves, masked to the zones — the other half of "embed this area".

    embed_zones returns one vector per zone and never assembles the grid, deliberately: 801
    Chicago tracts at 10 m is 25.8M pixels x 64 dims = 6.6 GB. But a 64-number average does
    not answer "what does this area look like to the model", and for a single tract the grid
    is a few tens of thousands of cells. So the image is a SECOND, bounded fetch over the
    zones' extent, refused outright when that extent is too large to ask for in one request
    rather than quietly returning something coarser than the caller thinks.
    """
    minlon, minlat, maxlon, maxlat = (float(v) for v in gdf.total_bounds)
    x0, y0 = _to_merc(minlon, minlat)
    x1, y1 = _to_merc(maxlon, maxlat)
    scale = float(scale_m or 10.0)
    estimate = ((x1 - x0) / scale) * ((y1 - y0) / scale)
    if estimate > max_px:
        return {"error": f"the zones span about {int(estimate):,} pixels at {scale:.0f} m, "
                         f"past the {int(max_px):,} this route will request in one call",
                "hint": "The per-zone vectors are unaffected. For a picture, embed fewer zones "
                        "or call embed_region with an explicit bbox over part of the area."}
    lo_lon, lo_lat = _to_lonlat(x0, y0)
    hi_lon, hi_lat = _to_lonlat(x1, y1)
    rs = _rs()
    emb = rs["get_embedding"](
        model,
        spatial=rs["BBox"](minlon=lo_lon, minlat=lo_lat, maxlon=hi_lon, maxlat=hi_lat),
        temporal=temporal,
        output=rs["OutputSpec"].grid(),
        backend="auto",
    )
    out = _render_masked_pca(emb.data, (x0, y0, x1, y1), gdf)
    # The provider snaps tiles outward to whole multiples of scale_m, so the grid can cover
    # slightly more than was asked for. The affine is derived from the REQUESTED bounds, so a
    # large snap would offset the drape: report it rather than let it be invisible.
    px, py = out["pixel_size_m_mercator"]
    if scale > 0 and (abs(px - scale) / scale > 0.02 or abs(py - scale) / scale > 0.02):
        out["warning"] = (f"implied pixel {px:.2f}x{py:.2f} m against scale_m {scale:.0f} — the "
                          f"grid the provider returned does not match the box requested, so the "
                          f"image may be offset by up to a pixel or two")
    return out


def _error_payload(exc: Exception) -> Dict[str, Any]:
    """An error the caller can act on, not just the exception's text.

    Earth Engine's own message tells whoever reads it to run ``earthengine authenticate`` —
    advice aimed at the operator of THIS service, which reaches the agent's user as an
    instruction they cannot follow on a machine they do not have. An OAuth refresh token from
    an app in testing status expires after seven days, so this is the failure that recurs.
    """
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    if "authorize access to your earth engine" in lowered or "invalid_grant" in lowered:
        return {
            "error": "the rs-embed service cannot reach Earth Engine: its credential is "
                     "expired, revoked or missing",
            "hint": "This is the SERVICE's credential, not the caller's — nothing the user "
                    "can retry. The service operator re-authorises it on the host (a personal "
                    "OAuth token from an app in testing status expires after 7 days; a GCP "
                    "service account does not). Report the outage; do not invent embeddings.",
            "detail": text[:300],
        }
    return {"error": text}


@app.get("/api/health")
def api_health(deep: int = 0) -> Any:
    """Is this service able to EMBED, not merely able to answer.

    ``/api/models`` returns 200 from a process that cannot embed anything, and has done for
    days at a time — twice: once with the worker threads parked on an outbound call that never
    returned, and once with an expired Earth Engine credential. systemd saw a healthy process
    both times. This endpoint fails when embedding would fail.

    ``deep=1`` also runs a round trip to Earth Engine's servers; the default only checks that
    the credential loads and refreshes, which is the failure that recurs (a personal OAuth
    token from an app in testing status expires after seven days).
    """
    out: Dict[str, Any] = {"ok": True, "earthengine": {"ok": True}}
    try:
        _ensure_ee()
        if deep:
            import ee

            ee.Number(1).getInfo()
            out["earthengine"]["round_trip"] = True
    except Exception as exc:  # noqa: BLE001
        out["ok"] = False
        out["earthengine"] = {"ok": False, **_error_payload(exc)}
        # 503, not 200-with-ok-false: anything that probes this by status code — a curl -f, a
        # systemd timer, a container healthcheck — has to see the outage, which is the whole
        # reason /api/models being 200 was not enough.
        return JSONResponse(out, status_code=503)
    return out


@app.post("/api/zones")
def api_zones(req: ZonesReq) -> Any:
    """Aggregate embedding pixels inside each polygon; return one row per zone.

    EVERY input polygon comes back, including one no tile reached — ``pixels == 0`` and no
    vector. ``ZoneEmbeddings.to_frame()`` drops those by design, but the caller needs to know
    which zones the sweep actually covered, and "the layer had 40 zones, 31 got vectors" is a
    different answer from "the layer had 31 zones".

    ``meta`` travels verbatim because it carries the provenance a caller must not invent: the
    Mercator pixel scale, what a pixel covers on the ground, the tile accounting (including
    whether ``max_tiles`` cut the sweep short) and any per-tile failure.
    """
    try:
        _ensure_ee()
        import geopandas as gpd
        from rs_embed import TemporalSpec, embed_zones

        feats = (req.zones_geojson or {}).get("features") or []
        if not feats:
            return {"ok": False, "error": "zones_geojson has no features"}
        gdf = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")

        if req.start and req.end:
            temporal = TemporalSpec.range(req.start, req.end)
        else:
            temporal = (TemporalSpec.year(req.year) if hasattr(TemporalSpec, "year")
                        else TemporalSpec.range(f"{req.year}-01-01", f"{req.year + 1}-01-01"))
        zed = embed_zones(
            req.model,
            zones=gdf,
            temporal=temporal,
            zone_id_field=req.zone_id_field,
            tile_px=int(req.tile_px),
            max_tiles=req.max_tiles,
        )
        rows = []
        for z in zed.zones:
            row: Dict[str, Any] = {"zone_id": str(z.zone_id), "pixels": int(z.pixels),
                                   "area_km2": float(z.area_km2)}
            if z.pixels and z.mean is not None:
                row.update({f"e{i:03d}": round(float(v), 6) for i, v in enumerate(z.mean)})
            rows.append(row)
        # default=str rather than letting a numpy scalar or a tile-error object break the
        # response: the point of meta is that it always arrives.
        meta = json.loads(json.dumps(dict(zed.meta or {}), default=str))
        image: Optional[Dict[str, Any]] = None
        if req.image:
            # A picture is worth having but it is not what was asked for: a render that fails
            # must never cost the vectors that already succeeded.
            try:
                image = _zone_pixel_image(gdf, req.model, temporal, meta.get("scale_m"),
                                          int(req.image_max_px))
            except Exception as exc:  # noqa: BLE001
                image = {"error": f"the pixel image could not be rendered: "
                                  f"{type(exc).__name__}: {exc}"[:300]}
        return {
            "ok": True,
            "image": image,
            "model": req.model,
            "year": req.year,
            "zones": len(rows),
            "zones_with_pixels": sum(1 for r in rows if r["pixels"]),
            "dim": int(meta.get("dims") or 0),
            "columns": ["zone_id", "pixels", "area_km2"],
            "meta": meta,
            "rows": rows,
        }
    except Exception as exc:  # noqa: BLE001 - the endpoint reports, never raises
        return {"ok": False, **_error_payload(exc)}
