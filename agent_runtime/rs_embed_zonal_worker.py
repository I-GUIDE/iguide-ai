"""Per-zone remote-sensing embeddings: pixels inside a polygon, aggregated.

Runs under the rs-embed interpreter (it needs ``rs_embed`` + Earth Engine), invoked as a
subprocess by ``agent_runtime.rs_embed_tools``. Reads a JSON request on stdin, writes a JSON
result to ``out_path``.

The method, and why it is shaped this way:

* rs-embed returns a grid as a bare ``(D, H, W)`` array — no transform, no CRS, no bounds.
  The grid is ``scale_m`` metres per pixel in **EPSG:3857** (verified: a 0.02x0.01 degree bbox
  at 40.07N returns 224x146, matching the 3857 span / 10 m to within the outward snap, where a
  true-ground-metre reading would predict 170x111). So the affine is derived here from the
  REQUESTED 3857 tile bounds and the RETURNED shape, and the implied pixel size is checked
  against ``scale_m`` — a silent mismatch would offset every zone boundary.

* The AOI is walked in TILES and zone statistics accumulate as running sums. A cube is never
  assembled: Chicago's 801 tracts span ~46x56 km, which at 10 m is 25.8M pixels x 64 dims =
  6.6 GB, while the accumulator is one 64-vector per zone.

* Each zone records ``sum`` and ``pixels``, not just a mean. A mean cannot be re-aggregated —
  a mean of means is wrong when zones differ in size — whereas sums and counts roll up to any
  coarser partition exactly, and ``pixels`` is the honest measure of a zone's support.
"""
from __future__ import annotations

import json
import math
import sys
import warnings

warnings.filterwarnings("ignore")

_R = 6378137.0  # WGS84 semi-major, the EPSG:3857 sphere radius


def _to_merc(lon: float, lat: float) -> tuple[float, float]:
    return _R * math.radians(lon), _R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def _to_lonlat(x: float, y: float) -> tuple[float, float]:
    return math.degrees(x / _R), math.degrees(2 * math.atan(math.exp(y / _R)) - math.pi / 2)


def main() -> int:
    req = json.loads(sys.stdin.read())
    out_path = req["out_path"]
    try:
        mode = str(req.get("mode") or "zones")
        result = fit(req) if mode == "fit" else run(req)
    except Exception as exc:  # noqa: BLE001
        import traceback
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                  "traceback": traceback.format_exc()[-1500:]}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh)
    return 0 if result.get("ok") else 1


def run(req: dict) -> dict:
    # Validation first, on the light dependencies only: a bad column name should not need
    # Earth Engine to be importable to be reported.
    import geopandas as gpd
    import numpy as np
    from affine import Affine
    from rasterio.features import rasterize

    model = str(req.get("model") or "gse")
    tile_px = int(req.get("tile_px") or 200)
    max_tiles = int(req.get("max_tiles") or 24)
    year = int(req.get("year") or 2022)

    gdf = gpd.read_file(req["polygons_path"])
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    id_field = req.get("zone_id_field")
    if id_field and id_field not in gdf.columns:
        # Falling back to row numbers here looked harmless and was not: the caller asked for
        # 'GEOID' where the column is 'geoid', so zones came back keyed 0..n while the label
        # join expected geoids — and the two silently failed to meet. Name the real columns.
        return {"ok": False,
                "error": f"zone_id_field {id_field!r} is not a column in the polygons",
                "available_columns": [c for c in gdf.columns if c != "geometry"][:40],
                "hint": "Column names are case-sensitive. Omit zone_id_field to key zones by "
                        "row number instead — but then use the same setting when fitting."}
    if id_field:
        zone_ids = [str(v) for v in gdf[id_field]]
    else:
        zone_ids = [str(i) for i in range(len(gdf))]

    # Zone areas in an equal-area CRS — a zone's own size, independent of the pixel grid.
    areas = (gdf.to_crs("EPSG:5070").area / 1e6).tolist()

    from rs_embed import BBox, OutputSpec, TemporalSpec, get_embedding

    merc = gdf.to_crs("EPSG:3857")
    # Burn value 0 means "no zone", so zones are numbered from 1.
    shapes = [(geom, i + 1) for i, geom in enumerate(merc.geometry) if geom is not None and not geom.is_empty]
    if not shapes:
        return {"ok": False, "error": "no usable polygon geometry in the input"}

    minx, miny, maxx, maxy = merc.total_bounds
    scale = None  # discovered from the first response's meta

    # Probe one small tile to learn scale_m and the band count before planning the sweep.
    probe = _fetch_tile(get_embedding, BBox, TemporalSpec, OutputSpec, model, year,
                        minx, miny, min(minx + 1000.0, maxx), min(miny + 1000.0, maxy))
    scale = float(probe["scale_m"])
    dims = int(probe["dims"])
    bands = probe["bands"]

    step = tile_px * scale                       # tile side in 3857 metres
    # Snap the sweep origin to the scale grid so every tile's span is an exact multiple of
    # scale_m; the returned shape then matches the request and the affine is unambiguous.
    ox = math.floor(minx / scale) * scale
    oy = math.floor(miny / scale) * scale
    nx = int(math.ceil((maxx - ox) / step))
    ny = int(math.ceil((maxy - oy) / step))
    planned = nx * ny

    sums = np.zeros((len(gdf) + 1, dims), dtype=np.float64)
    counts = np.zeros(len(gdf) + 1, dtype=np.int64)

    # Optional pixel-level picture: the embedding INSIDE the shapes, which is the thing the
    # zone means are computed from. Colours come from projecting each pixel onto 3 axes fitted
    # on the zone means, so they are consistent across tiles and across the whole mosaic —
    # per-tile PCA would recolour every tile differently and the seams would read as data.
    want_image = bool(req.get("image", True))
    canvas = np.zeros((ny * tile_px, nx * tile_px, 4), dtype=np.uint8) if want_image else None
    proj_pixels: list = []   # (row0, col0, (D,h,w) values, mask) held only while sweeping

    tiles_done = 0
    tile_errors: list[dict] = []
    pixel_size_warnings: list[str] = []
    zone_index = {i + 1: i for i in range(len(gdf))}

    for ty in range(ny):
        for tx in range(nx):
            if tiles_done >= max_tiles:
                break
            x0, y0 = ox + tx * step, oy + ty * step
            x1, y1 = x0 + step, y0 + step
            if x0 > maxx or y0 > maxy:
                continue
            # Skip tiles no zone touches — most of a bounding box is usually empty.
            if not merc.sindex.query(_box(x0, y0, x1, y1), predicate="intersects").size:
                continue
            try:
                got = _fetch_tile(get_embedding, BBox, TemporalSpec, OutputSpec,
                                  model, year, x0, y0, x1, y1)
            except Exception as exc:  # noqa: BLE001
                tile_errors.append({"tile": [tx, ty], "error": f"{type(exc).__name__}: {exc}"[:200]})
                continue
            arr = got["array"]                     # (D, H, W)
            _d, h, w = arr.shape
            px = (x1 - x0) / w
            py = (y1 - y0) / h
            if abs(px - scale) / scale > 0.02 or abs(py - scale) / scale > 0.02:
                pixel_size_warnings.append(
                    f"tile {tx},{ty}: implied pixel {px:.2f}x{py:.2f} m vs scale_m {scale:.0f}")
            # North-up affine: rows run from the tile's TOP edge downwards.
            transform = Affine(px, 0.0, x0, 0.0, -py, y1)
            zmap = rasterize(shapes, out_shape=(h, w), transform=transform,
                             fill=0, all_touched=False, dtype="int32")
            present = np.unique(zmap)
            present = present[present > 0]
            if present.size:
                flat = arr.reshape(_d, h * w)
                zflat = zmap.reshape(h * w)
                for z in present:
                    m = zflat == z
                    n = int(m.sum())
                    if not n:
                        continue
                    vals = flat[:, m]
                    good = np.isfinite(vals).all(axis=0)
                    if not good.any():
                        continue
                    sums[z] += vals[:, good].sum(axis=1)
                    counts[z] += int(good.sum())
                if canvas is not None:
                    inside = zmap > 0
                    if inside.any():
                        row0 = (ny - 1 - ty) * tile_px   # canvas rows run north -> south
                        proj_pixels.append((row0, tx * tile_px, arr.copy(), inside))
            tiles_done += 1
        if tiles_done >= max_tiles:
            break

    zones = []
    for i, zid in enumerate(zone_ids):
        n = int(counts[i + 1])
        zones.append({
            "zone_id": zid,
            "pixels": n,
            "area_km2": round(float(areas[i]), 6),
            # sum + count, not a mean: a mean of means is wrong across unequal zones, while
            # sums roll up to any coarser partition exactly.
            "sum": [round(float(v), 6) for v in sums[i + 1]] if n else None,
            "mean": [round(float(v / n), 6) for v in sums[i + 1]] if n else None,
        })

    # Look-alike groups for the map. A 64-dim vector has no natural colour; clusters do.
    clusters = int(req.get("clusters") or 0)
    if clusters >= 2:
        with_px = [z for z in zones if z["pixels"]]
        if len(with_px) >= clusters:
            from sklearn.cluster import KMeans

            X = np.asarray([z["mean"] for z in with_px], dtype=np.float64)
            Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
            lab = KMeans(n_clusters=clusters, n_init=10, random_state=0).fit_predict(Xn)
            for z, v in zip(with_px, lab, strict=True):
                z["group"] = int(v) + 1

    image_info = None
    if canvas is not None and proj_pixels:
        try:
            from sklearn.decomposition import PCA

            basis = np.asarray([z["mean"] for z in zones if z["pixels"]], dtype=np.float64)
            if len(basis) >= 3:
                pca = PCA(n_components=3, random_state=0).fit(basis)
                comp = pca.components_
                # Deterministic sign, so a rerun does not invert the colours.
                for c in range(3):
                    if comp[c].sum() < 0:
                        comp[c] = -comp[c]
                vals_all = []
                for _r0, _c0, arr, inside in proj_pixels:
                    d = arr.shape[0]
                    flat = arr.reshape(d, -1)[:, inside.reshape(-1)]
                    vals_all.append((comp @ np.nan_to_num(flat)).T)
                stacked = np.concatenate(vals_all, axis=0)
                lo = np.percentile(stacked, 2, axis=0)
                hi = np.percentile(stacked, 98, axis=0)
                span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
                for (r0, c0, arr, inside), proj in zip(proj_pixels, vals_all, strict=True):
                    h, w = inside.shape
                    rgb = np.clip((proj - lo) / span, 0, 1) * 255.0
                    tile_rgba = np.zeros((h, w, 4), dtype=np.uint8)
                    tile_rgba[inside, :3] = rgb.astype(np.uint8)
                    tile_rgba[inside, 3] = 255          # outside the shapes stays transparent
                    # GEE snaps a tile OUTWARD to whole pixels, so a 250-pixel request can
                    # come back 251 rows. Those extra rows fall outside the tile that was
                    # asked for and would overlap the neighbour, so they are trimmed —
                    # the canvas grid is defined by the requested step, not by the response.
                    hh = min(h, canvas.shape[0] - r0)
                    ww = min(w, canvas.shape[1] - c0)
                    hh = min(hh, tile_px)
                    ww = min(ww, tile_px)
                    if hh > 0 and ww > 0:
                        canvas[r0:r0 + hh, c0:c0 + ww] = tile_rgba[:hh, :ww]
                from PIL import Image

                out_png = req.get("out_png")
                if out_png:
                    Image.fromarray(canvas).save(out_png)
                    wl, sl = _to_lonlat(ox, oy)
                    el, nl = _to_lonlat(ox + nx * step, oy + ny * step)
                    image_info = {"path": out_png,
                                  "bounds": [round(wl, 6), round(sl, 6), round(el, 6), round(nl, 6)],
                                  "size_px": [int(canvas.shape[1]), int(canvas.shape[0])],
                                  # counted from the written canvas, so it matches the
                                  # image rather than the pre-trim tile masks
                                  "pixels_shown": int((canvas[:, :, 3] > 0).sum()),
                                  "colour": "pixels projected onto 3 axes fitted on the zone "
                                            "means; consistent across the mosaic, not "
                                            "comparable to another run"}
        except Exception as exc:  # noqa: BLE001
            image_info = {"error": f"{type(exc).__name__}: {exc}"[:200]}
    proj_pixels.clear()

    covered = sum(1 for z in zones if z["pixels"])
    return {
        "ok": covered > 0,
        "model": model, "year": year, "dims": dims, "bands": bands,
        # scale_m is measured in EPSG:3857, where a metre is 1/cos(latitude) too long. Report
        # what a pixel actually covers on the ground so nobody reads 10 m and means 10 m.
        "scale_m": scale,
        "pixel_ground_m": round(scale * math.cos(math.radians(_to_lonlat(0, (miny + maxy) / 2)[1])), 3),
        "zone_id_field": id_field,
        "tiles_planned": planned, "tiles_fetched": tiles_done,
        "tiles_capped": planned > max_tiles,
        "tile_px": tile_px,
        "zones_total": len(zones), "zones_with_pixels": covered,
        "zones": zones,
        "image": image_info,
        "tile_errors": tile_errors[:10],
        "pixel_size_warnings": pixel_size_warnings[:5],
        "error": None if covered else "no zone received any pixels",
    }


def fit(req: dict) -> dict:
    """Ridge on zone vectors, scored by SPATIAL BLOCK cross-validation.

    Folds are contiguous blocks of space, so no zone is scored by a model that trained on its
    neighbours. On real Chicago tracts a random split claimed R2=+0.15 for a health outcome
    where blocked CV showed -0.91: the naive number was an artifact of adjacency, not skill.
    """
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold, KFold

    df = pd.read_csv(req["vectors_csv"])
    gdf = gpd.read_file(req["polygons_path"])
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    label = req["label_column"]
    if label not in gdf.columns:
        numeric = [c for c in gdf.columns
                   if c != "geometry" and pd.api.types.is_numeric_dtype(gdf[c])]
        return {"ok": False, "error": f"label_column {label!r} is not in the polygons",
                "numeric_columns": numeric[:40]}
    id_field = req.get("zone_id_field")
    if id_field and id_field not in gdf.columns:
        return {"ok": False, "error": f"zone_id_field {id_field!r} is not in the polygons",
                "available_columns": [c for c in gdf.columns if c != "geometry"][:40]}
    ids = ([str(v) for v in gdf[id_field]] if id_field else [str(i) for i in range(len(gdf))])
    gdf = gdf.assign(_zid=ids)
    df["zone_id"] = df["zone_id"].astype(str)
    m = gdf.merge(df, left_on="_zid", right_on="zone_id", how="inner")
    feat = [c for c in df.columns if c.startswith("e") and c[1:].isdigit()]
    m[label] = pd.to_numeric(m[label], errors="coerce")
    m = m.dropna(subset=[label] + feat)
    n = len(m)
    if n < 12:
        return {"ok": False, "error": f"only {n} zones have both a vector and a label",
                "hint": "Embed more zones before fitting; spatial-block CV holds out whole "
                        "blocks, so it needs enough zones to leave any for training."}

    X = m[feat].to_numpy(dtype=np.float64)
    y = m[label].to_numpy(dtype=np.float64)
    cent = m.to_crs("EPSG:5070").geometry.centroid
    XY = np.c_[cent.x.to_numpy(), cent.y.to_numpy()]
    blocks = int(max(2, min(int(req.get("blocks") or 5), n // 4)))
    groups = KMeans(n_clusters=blocks, n_init=10, random_state=0).fit_predict(XY)
    alphas = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)

    def _oof(splitter, g=None):
        out = np.full(n, np.nan)
        for tr, te in (splitter.split(X, y, g) if g is not None else splitter.split(X)):
            out[te] = RidgeCV(alphas=alphas).fit(X[tr], y[tr]).predict(X[te])
        return out

    oof = _oof(GroupKFold(n_splits=blocks), groups)
    naive = _oof(KFold(n_splits=blocks, shuffle=True, random_state=0))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    def _r2(p):
        return 1.0 - float(np.sum((y - p) ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((y - oof) ** 2)))
    base_rmse = float(np.sqrt(ss_tot / n))

    m["observed"] = np.round(y, 4)
    m["predicted"] = np.round(oof, 4)
    m["residual"] = np.round(y - oof, 4)
    keep = [c for c in ("geometry", "zone_id", "pixels", "area_km2",
                        "observed", "predicted", "residual") if c in m.columns]
    m[keep].to_file(req["out_geojson"], driver="GeoJSON")
    return {
        "ok": True, "label_column": label, "zones_fitted": n, "features": len(feat),
        "spatial_block_cv": {"blocks": blocks, "r2": round(_r2(oof), 4), "rmse": round(rmse, 4)},
        "naive_random_split_cv": {"r2": round(_r2(naive), 4)},
        "baseline_predict_the_mean": {"r2": 0.0, "rmse": round(base_rmse, 4)},
        "skill_vs_baseline_pct": round(100 * (1 - rmse / base_rmse), 1) if base_rmse else None,
        "observed_range": [round(float(y.min()), 4), round(float(y.max()), 4)],
        "support_pixels": ({"min": int(m["pixels"].min()), "median": int(m["pixels"].median()),
                            "max": int(m["pixels"].max())} if "pixels" in m else None),
    }


def _box(x0, y0, x1, y1):
    from shapely.geometry import box
    return box(x0, y0, x1, y1)


def _fetch_tile(get_embedding, BBox, TemporalSpec, OutputSpec, model, year,
                x0, y0, x1, y1) -> dict:
    """One tile: 3857 bounds in, ``(D,H,W)`` array + scale/bands out."""
    import numpy as np

    lon0, lat0 = _to_lonlat(x0, y0)
    lon1, lat1 = _to_lonlat(x1, y1)
    emb = get_embedding(
        model,
        spatial=BBox(minlon=lon0, minlat=lat0, maxlon=lon1, maxlat=lat1),
        temporal=TemporalSpec.year(year) if hasattr(TemporalSpec, "year")
        else TemporalSpec.range(f"{year}-01-01", f"{year + 1}-01-01"),
        output=OutputSpec.grid(),
        backend="auto",
    )
    data = emb.data
    arr = np.asarray(getattr(data, "values", data), dtype=np.float32)
    meta = emb.meta or {}
    return {"array": arr, "dims": int(arr.shape[0]),
            "scale_m": float(meta.get("scale_m") or 10),
            "bands": [str(b) for b in (meta.get("bands") or [])][:8]}


if __name__ == "__main__":
    raise SystemExit(main())
