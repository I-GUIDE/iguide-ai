"""Per-zone remote-sensing embeddings: pixels inside a polygon, aggregated.

What still runs from here, and what moved. The tile sweep in :func:`run` needs ``rs_embed``
and Earth Engine, so it runs under the rs-embed interpreter, reading a JSON request on stdin
and writing the result to ``out_path``; the agent no longer invokes it, because
``rs_embed.embed_zones`` now does the same sweep behind the rs-embed SERVICE. This module
remains the standalone/CLI implementation and the reference for the method below, and the
agent still calls two things in it directly, in its own process: :func:`fit` (pandas, numpy
and a linear solve — no model runtime) and :func:`assign_look_alike_groups`.

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


# --- numpy stand-ins for the scikit-learn pieces ---------------------------------
#
# These run in the AGENT's process, which already has torch loaded, and scikit-learn's
# k-means reaches OpenMP through its own compiled extension. Two OpenMP runtimes in one
# process is not a warning here: `KMeans.fit` SEGFAULTED the pytest process outright
# (macOS, anaconda scikit-learn beside torch), taking the whole run down rather than
# failing one test. It is the same clash that once wedged a turn through fork.
#
# So nothing in the agent-side path may import scikit-learn. These are the three things it
# was used for, in plain numpy: k-means, ridge with its alpha chosen by leave-one-out, and
# the two fold splitters. Each mirrors the scikit-learn behaviour it replaces closely enough
# that the reported numbers do not move — see the equivalence test, which runs scikit-learn
# in a FRESH subprocess (where torch is absent and it is safe) and compares.


def _kmeans_pp_centres(X, k, rng):
    """k-means++ seeding: spread the first centres out, so Lloyd's starts somewhere sane."""
    import numpy as np

    centres = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d2 = np.min(((X[:, None, :] - np.asarray(centres)[None, :, :]) ** 2).sum(-1), axis=1)
        total = float(d2.sum())
        if total <= 0:                      # every point already sits on a centre
            centres.append(X[rng.integers(len(X))])
            continue
        centres.append(X[rng.choice(len(X), p=d2 / total)])
    return np.asarray(centres, dtype=np.float64)


def kmeans_labels(X, k: int, seed: int = 0, restarts: int = 10, iters: int = 100):
    """Lloyd's algorithm with k-means++ starts; the lowest-inertia restart wins.

    Deterministic for a given seed, which matters more than it sounds: the map colours zones
    by group number, so a rerun that relabels the clusters looks like the analysis changed.
    """
    import numpy as np

    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    if k >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    best_labels, best_inertia = None, float("inf")
    for _ in range(restarts):
        centres = _kmeans_pp_centres(X, k, rng)
        labels = np.full(n, -1)
        for _ in range(iters):
            d2 = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
            new_labels = d2.argmin(1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for j in range(k):
                member = X[labels == j]
                if len(member):
                    centres[j] = member.mean(0)
        inertia = float(((X - centres[labels]) ** 2).sum())
        if inertia < best_inertia - 1e-12:
            best_inertia, best_labels = inertia, labels
    return best_labels


def ridge_loo_predict(X_train, y_train, X_test, alphas):
    """Ridge whose alpha is chosen by leave-one-out on the training fold, then predict.

    This is what ``RidgeCV(alphas=...)`` does by default, and it is closed-form: one SVD of
    the centred training matrix gives every alpha's LOO error at once, because the hat matrix
    is ``U diag(s^2/(s^2+a)) U'`` and the LOO residual is the plain residual over ``1 - h_ii``.
    No refitting, no inner loop.
    """
    import numpy as np

    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    n = len(X_train)
    x_bar = X_train.mean(0)
    y_bar = float(y_train.mean())
    U, s, Vt = np.linalg.svd(X_train - x_bar, full_matrices=False)
    Uy = U.T @ (y_train - y_bar)
    s2 = s ** 2
    best_alpha, best_err = float(alphas[0]), float("inf")
    for alpha in alphas:
        shrink = s2 / (s2 + float(alpha))
        fitted = U @ (shrink * Uy)
        # + 1/n for the intercept, which centring fitted implicitly.
        leverage = (U ** 2) @ shrink + 1.0 / n
        residual = (y_train - y_bar) - fitted
        err = float(np.mean((residual / np.clip(1.0 - leverage, 1e-9, None)) ** 2))
        if err < best_err:
            best_alpha, best_err = float(alpha), err
    coef = Vt.T @ ((s / (s2 + best_alpha)) * Uy)
    return (np.asarray(X_test, dtype=np.float64) - x_bar) @ coef + y_bar, best_alpha


def kfold_indices(n: int, splits: int, seed: int = 0):
    """``KFold(shuffle=True, random_state=seed)``: contiguous folds over a shuffled order.

    RandomState, not the newer Generator, so the permutation matches scikit-learn's exactly
    and the "what a random split would have claimed" number is reproducible against it.
    """
    import numpy as np

    order = np.random.RandomState(seed).permutation(n)
    sizes = [n // splits + (1 if i < n % splits else 0) for i in range(splits)]
    folds, start = [], 0
    for size in sizes:
        test = np.sort(order[start:start + size])
        folds.append((np.setdiff1d(np.arange(n), test), test))
        start += size
    return folds


def group_kfold_indices(groups, splits: int):
    """``GroupKFold``: whole groups per fold, biggest group first into the emptiest fold.

    No group is ever split across the train/test line — which is the entire point of blocking
    by space, since a zone scored by a model trained on its own neighbours is not held out.
    """
    import numpy as np

    groups = np.asarray(groups)
    unique, counts = np.unique(groups, return_counts=True)
    fold_of, load = {}, [0] * splits
    # argsort(counts)[::-1], not argsort(-counts): the two order ties differently, and a tie
    # is the common case when blocks come out even. This is scikit-learn's order.
    for idx in np.argsort(counts)[::-1]:
        target = int(np.argmin(load))
        fold_of[unique[idx]] = target
        load[target] += int(counts[idx])
    assigned = np.array([fold_of[g] for g in groups])
    folds = []
    for f in range(splits):
        test = np.flatnonzero(assigned == f)
        if len(test):
            folds.append((np.flatnonzero(assigned != f), test))
    return folds


def assign_look_alike_groups(zones: list, clusters: int) -> int:
    """Tag each covered zone with a 1-based look-alike ``group``; return the group count.

    A 64-dimension vector has no natural colour, so the map shows clusters instead. Vectors
    are L2-normalised first: without that, k-means splits on overall brightness — how much
    sunlit surface a zone has — rather than on what the surface is.

    Groups are 1-based because the map layer treats a missing group as "not embedded", and
    zones with no pixels are left untagged for the same reason. Mutates ``zones`` in place;
    ``clusters < 2`` means "do not group" and is a no-op.

    ``clusters`` is a ceiling, not a requirement: asking for five groups across three zones
    gives three. It used to give NONE, and since the group layer is the only thing embed_zones
    puts on the map, "embed this one polygon" — the commonest request there is — delivered a
    CSV and an empty map with nothing saying why.
    """
    import numpy as np

    with_px = [z for z in zones if z.get("pixels")]
    if clusters < 2 or not with_px:
        return 0
    k = min(int(clusters), len(with_px))
    if k < 2:
        # One covered zone: there is nothing to cluster, but it still belongs on the map.
        with_px[0]["group"] = 1
        return 1
    X = np.asarray([z["mean"] for z in with_px], dtype=np.float64)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    lab = kmeans_labels(Xn, k)
    for z, v in zip(with_px, lab, strict=True):
        z["group"] = int(v) + 1
    return int(len(set(lab)))


def run(req: dict) -> dict:
    # Validation first, on the light dependencies only: a bad column name should not need
    # Earth Engine to be importable to be reported.
    import geopandas as gpd
    import numpy as np
    from affine import Affine
    from rasterio.features import rasterize

    model = str(req.get("model") or "gse")
    tile_px = int(req.get("tile_px") or 200)
    # None means uncapped, matching embed_zones. `or 24` would have turned an explicit
    # "no cap" back into 24, and an explicit 0 into 24 as well.
    _mt = req.get("max_tiles")
    max_tiles = None if _mt is None else int(_mt)
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

    capped = False
    for ty in range(ny):
        for tx in range(nx):
            if max_tiles is not None and tiles_done >= max_tiles:
                capped = True
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
        if max_tiles is not None and tiles_done >= max_tiles:
            capped = True
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

    assign_look_alike_groups(zones, int(req.get("clusters") or 0))

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
        # Set where the sweep ACTUALLY stopped early, not `planned > max_tiles` -- that
        # compared the cap against the whole bounding grid, most of whose cells are empty and
        # skipped for free, so it reported truncation on sweeps that lost nothing.
        "tiles_capped": capped,
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
    groups = kmeans_labels(XY, blocks)
    alphas = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)

    def _oof(folds):
        out = np.full(n, np.nan)
        for tr, te in folds:
            out[te] = ridge_loo_predict(X[tr], y[tr], X[te], alphas)[0]
        return out

    block_folds = group_kfold_indices(groups, blocks)
    # Fewer distinct clusters than blocks means fewer folds, and the naive split has to use
    # the SAME count: the two r2 values are subtracted to decide whether apparent skill was
    # only adjacency, and a 5-fold naive score against a 4-fold blocked one moves that
    # difference by enough to flip the verdict on its own.
    blocks = len(block_folds)
    oof = _oof(block_folds)
    naive = _oof(kfold_indices(n, blocks, seed=0))
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
