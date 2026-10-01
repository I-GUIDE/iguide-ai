"""Elevation (DEM) for a region, draped on the map — the terrain counterpart to embed_region.

Deliberately NOT routed through the rs-embed service. That service holds the deployment's Earth
Engine credential, and Earth Engine is the only way it can reach imagery — so every embedding
call spends a personal Google credential that has expired twice. Elevation does not need it:
USGS 3DEP publishes the authoritative US DEM through an ArcGIS ImageServer that takes no key,
no token and no quota, and returns real float32 metres rather than a picture of them. So this
module talks to that directly and the agent gains a raster tool that costs nothing to run.

The price is coverage: 3DEP is the United States (plus territories). Outside it the server
answers with a full frame of NoData rather than an error, which would drape a blank layer over
the map and report a successful run — so an all-NoData response is detected here and refused
with the bbox that produced it.

zonal_stats_for_raster is the other half: it turns ANY raster here — the DEM above, or
anything else single-band — into per-zone columns on a polygon layer, which is the shape
fit_zone_model already takes. That is what lets elevation be read beside the satellite
embeddings rather than only looked at.

terrain_derivative and inundation_at_level are the rest of what a DEM is usually wanted
for — steepness, and what lies under a given height. Both are arithmetic on a raster this
module already fetched, so they add no dependency and, like dem_for_region, cost nothing to
run. inundation_at_level is a bathtub fill and says so in every answer: it is ground below
a height, and a regulatory floodplain is a different thing that this deployment cannot
produce.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

# Region resolution and the map-layer descriptor are shared plumbing, not embedding-specific:
# _resolve_bbox understands every way a region arrives (bbox, point+buffer, an uploaded file's
# extent) and _raster_layer builds the descriptor the client drapes. Reimplementing either here
# would be a second copy to keep in sync with the client's whitelist.
from agent_runtime.rs_embed_tools import (_layer_id, _layer_label, _raster_layer, _region_tag,
                                          _resolve_bbox, _round_bbox, _slug)
from agent_runtime.tool_args import accept_null_defaults

_3DEP_URL = ("https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation"
             "/ImageServer/exportImage")
# The frame requested from 3DEP. 512 is ~10 m per pixel over a 5 km box, which is 3DEP's own
# 1/3 arc-second resolution — asking for more returns interpolated pixels that look like detail
# and are not. The ceiling is what the map can drape without the PNG dominating the response.
_DEFAULT_SIZE = 512
_MAX_SIZE = 1536
_MIN_SIZE = 64
_TIMEOUT_S = float(os.getenv("AGENT_DEM_TIMEOUT_S", "60"))
_NODATA = -999999.0
# A square-metre area beyond which the requested frame cannot hold real detail. Not a refusal:
# the note says what the pixel size worked out to, so an answer cannot call a 2 km pixel "10 m".
_UA = "iguide-agent/1.0 (+https://iguide.illinois.edu)"


_PROJ_REPAIRED = False


def _wgs84() -> Any:
    """A WGS84 CRS for the GeoTIFF, repairing rasterio's PROJ lookup if it is broken. Or None.

    MEASURED on the deployed container, and it would have shipped: PROJ_LIB and PROJ_DATA are
    set to /usr/share/proj, whose proj.db is DATABASE.LAYOUT version 5, and rasterio's PROJ
    needs >= 6 — so CRS.from_epsg(4326) raises there while passing on a laptop. The system
    database is not a mistake; the QGIS worker subprocess needs exactly that copy, and it
    inherits our environment. So the repair must NOT touch os.environ.

    set_proj_data_search_path points the PROJ already loaded in THIS process at rasterio's own
    bundled database, leaving the environment every child inherits alone. It is a private
    rasterio API, hence the probe-first-then-repair shape and the None fallback: a GeoTIFF
    carrying a transform but no CRS is still readable as WGS84, and losing the CRS is a far
    smaller failure than losing the elevation request.
    """
    global _PROJ_REPAIRED
    import rasterio.crs

    try:
        return rasterio.crs.CRS.from_epsg(4326)
    except Exception:  # noqa: BLE001 - the broken-PROJ case this exists for
        pass
    if not _PROJ_REPAIRED:
        _PROJ_REPAIRED = True
        try:
            import rasterio
            from rasterio._env import set_proj_data_search_path

            set_proj_data_search_path(
                os.path.join(os.path.dirname(rasterio.__file__), "proj_data"))
        except Exception:  # noqa: BLE001
            return None
    try:
        return rasterio.crs.CRS.from_epsg(4326)
    except Exception:  # noqa: BLE001
        return None


def _frame_for(bbox: List[float], size: int) -> tuple:
    """``(width, height)`` in pixels whose aspect matches the box's DEGREE aspect.

    This is what keeps the raster the size of the region that was asked for. 3DEP honours the
    requested extent only when the frame's aspect matches the bbox's — in DEGREES, not on the
    ground. Ask for a square frame over a box that is square in METRES (which is what a drawn
    region and `_mercator_square` both give) and the server pads the short axis instead of
    distorting the pixels: measured, a 512x512 frame over a 1.307:1 box came back 466 m taller
    at each edge, while a 512x392 frame came back within 1.2 m of the request.

    The cost is pixels that are not square ON THE GROUND — at 40N one is ~1.3x taller in metres
    than it is wide. That is the normal shape of any EPSG:4326 raster, and every consumer here
    already reads the two spacings separately from the transform rather than assuming one.
    """
    dlon = abs(bbox[2] - bbox[0]) or 1.0
    dlat = abs(bbox[3] - bbox[1]) or 1.0
    long_side = max(_MIN_SIZE, min(int(size or _DEFAULT_SIZE), _MAX_SIZE))
    if dlon >= dlat:
        return long_side, max(_MIN_SIZE, min(_MAX_SIZE, round(long_side * dlat / dlon)))
    return max(_MIN_SIZE, min(_MAX_SIZE, round(long_side * dlon / dlat))), long_side


def _fetch_dem(bbox: List[float], size: int) -> Any:
    """The GeoTIFF bytes for *bbox*, or an error dict."""
    frame_w, frame_h = _frame_for(bbox, size)
    query = {
        "bbox": ",".join(str(v) for v in bbox),
        "bboxSR": 4326, "imageSR": 4326,
        "size": f"{frame_w},{frame_h}",
        "format": "tiff", "pixelType": "F32", "noData": _NODATA,
        "interpolation": "RSP_BilinearInterpolation",
        # f=image returns the raster itself; f=json returns a URL to fetch it from, which is a
        # second round trip and a second thing to fail.
        "f": "image",
    }
    url = f"{_3DEP_URL}?{urllib.parse.urlencode(query)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            body = resp.read()
    except Exception as exc:  # noqa: BLE001 - the network is the expected failure here
        return {"error": f"USGS 3DEP request failed: {type(exc).__name__}: {exc}"[:300]}
    # An ArcGIS error comes back as JSON with a 200, so a short response that parses as JSON is
    # a failure wearing a success. Without this the TIFF reader would raise something unrelated.
    if len(body) < 4096 and body[:1] in (b"{", b"["):
        try:
            return {"error": f"USGS 3DEP returned an error: {json.loads(body)}"[:300]}
        except Exception:  # noqa: BLE001
            pass
    return body


def _file_ref(rec: Dict[str, Any]) -> Dict[str, Any]:
    """A file the answer can offer: id, url AND NAME.

    The name is not decoration. The client harvests file records out of any tool result and
    falls back to the literal string "download" when `filename` is absent — so a download list
    read "download · download · download" while every one of these dicts knew its real name and
    simply did not send it.
    """
    return {"file_id": rec["file_id"], "filename": rec.get("filename"),
            "download_url": rec.get("download_url")}


def _stats(values: Any) -> Dict[str, Any]:
    import numpy as np

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {}
    lo, hi = float(finite.min()), float(finite.max())
    return {"min_m": round(lo, 2), "max_m": round(hi, 2),
            "mean_m": round(float(finite.mean()), 2),
            "relief_m": round(hi - lo, 2),
            "pixels_with_data": int(finite.size)}


def _ground_resolution_m(bbox: List[float], size: int) -> float:
    """Metres per pixel at the box's mid-latitude, so the answer can state it honestly."""
    mid_lat = (bbox[1] + bbox[3]) / 2.0
    width_m = (bbox[2] - bbox[0]) * 111_320.0 * math.cos(math.radians(mid_lat))
    height_m = (bbox[3] - bbox[1]) * 110_540.0
    return round(max(width_m, height_m) / max(size, 1), 2)


def _pixel_size_m(transform: Any, bbox: List[float]) -> float:
    """Metres per pixel read off the raster's OWN transform.

    Preferred over _ground_resolution_m wherever a real raster is in hand: that one divides the
    REQUESTED box by the REQUESTED size, and 3DEP honours neither exactly.
    """
    mid_lat = (bbox[1] + bbox[3]) / 2.0
    return round(max(abs(transform.a) * 111_320.0 * math.cos(math.radians(mid_lat)),
                     abs(transform.e) * 110_540.0), 2)


def _render(values: Any, path: Path) -> None:
    """A terrain-coloured PNG, ONE image pixel per DEM cell.

    Not a matplotlib figure: axes, margins and a colorbar would become part of the image, and a
    draped layer is positioned by its bounds — so the pixels would no longer line up with the
    ground. `embed_region`'s docstring warns about exactly this for composed layers; the same
    trap applies to anything drawn here.
    """
    import numpy as np
    import matplotlib
    from PIL import Image

    finite = values[np.isfinite(values)]
    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    span = (hi - lo) or 1.0
    norm = np.clip((values - lo) / span, 0.0, 1.0)
    # matplotlib.colormaps, not cm.get_cmap: that was deprecated for removal in 3.11 and the
    # deployed container runs 3.11.1, so the old call is a production AttributeError waiting on
    # an image nobody rendered locally.
    rgba = (matplotlib.colormaps["terrain"](np.nan_to_num(norm)) * 255).astype("uint8")
    # NoData must be TRANSPARENT, not the bottom of the ramp: a nodata hole coloured deep blue
    # reads as water, and the sea-level end of `terrain` is exactly that colour.
    rgba[..., 3] = np.where(np.isfinite(values), 255, 0)
    Image.fromarray(rgba, mode="RGBA").save(path)


def _clip_to(values: Any, transform: Any, file_id: str) -> Any:
    """*values* with everything outside the polygons in *file_id* set to NaN, or unchanged.

    Best effort by design: a geometry that cannot be read is a reason to return the rectangle,
    not to fail the elevation request the user actually asked for.
    """
    try:
        import numpy as np
        from rasterio.features import geometry_mask

        from agent_runtime.file_store import resolve_file_id

        raw = json.loads(Path(resolve_file_id(file_id)).read_text(encoding="utf-8"))
        feats = raw.get("features") if isinstance(raw, dict) else None
        geoms = [f["geometry"] for f in (feats or []) if isinstance(f, dict) and f.get("geometry")]
        if not geoms:
            if isinstance(raw, dict) and raw.get("type") in {"Polygon", "MultiPolygon"}:
                geoms = [raw]
        polys = [g for g in geoms if str(g.get("type")) in {"Polygon", "MultiPolygon"}]
        if not polys:
            return values
        outside = geometry_mask(polys, out_shape=values.shape, transform=transform,
                                invert=False, all_touched=True)
        return np.where(outside, np.nan, values)
    except Exception:  # noqa: BLE001 - the rectangle is still a correct answer
        return values



def _open_raster(raster_file_id: str, band: int = 1) -> Any:
    """``(values, transform, bounds)`` for a stored raster, or an error dict.

    Reads src.transform and never src.crs, for the reason _wgs84 documents: the deployed
    container's PROJ database is older than rasterio's PROJ accepts, so asking for a CRS these
    tools already know is a failure that only ever happens in production. The sentinel sweep is
    here rather than at each caller because a 3DEP raster can reach the store carrying -999999
    instead of NaN, and one tool remembering that while another forgets is how a mean ends up
    six orders of magnitude wrong.
    """
    import numpy as np
    import rasterio

    from agent_runtime.langchain_geo_tools import _resolve

    try:
        path, _rec = _resolve(raster_file_id)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"could not resolve {raster_file_id!r}: {exc}"[:250]}
    try:
        with rasterio.open(path) as src:
            if int(band) < 1 or int(band) > src.count:
                return {"error": f"band {band} does not exist",
                        "hint": f"this raster has {src.count} band(s)"}
            values = src.read(int(band), masked=True).astype("float64").filled(np.nan)
            transform = src.transform
            bounds = tuple(float(b) for b in src.bounds)
            nodata = src.nodata
    except Exception as exc:  # noqa: BLE001
        return {"error": f"could not read the raster: {type(exc).__name__}: {exc}"[:250]}
    if nodata is not None and np.isfinite(nodata):
        values[values == nodata] = np.nan
    values[values <= _NODATA + 1] = np.nan
    return values, transform, bounds


def _write_grid(path: Path, grid: Any, transform: Any) -> None:
    """A single-band float32 GeoTIFF, NaN for nodata, WGS84 when PROJ allows one."""
    import rasterio

    with rasterio.open(path, "w", driver="GTiff", height=grid.shape[0], width=grid.shape[1],
                       count=1, dtype="float32", crs=_wgs84(), transform=transform,
                       nodata=float("nan")) as dst:
        dst.write(grid.astype("float32"), 1)


def _render_grid(grid: Any, path: Path, cmap: str, vmin: Optional[float] = None) -> None:
    """One image pixel per cell, NoData transparent — the same contract _render holds.

    Not a matplotlib figure, for the reason _render gives: axes and margins would become part
    of an image the map positions by its bounds, so the pixels would stop lining up with the
    ground.
    """
    import numpy as np
    import matplotlib
    from PIL import Image

    finite = grid[np.isfinite(grid)]
    lo = float(vmin) if vmin is not None else (float(finite.min()) if finite.size else 0.0)
    hi = float(finite.max()) if finite.size else 1.0
    span = (hi - lo) or 1.0
    norm = np.clip((grid - lo) / span, 0.0, 1.0)
    rgba = (matplotlib.colormaps[cmap](np.nan_to_num(norm)) * 255).astype("uint8")
    rgba[..., 3] = np.where(np.isfinite(grid), 255, 0)
    Image.fromarray(rgba, mode="RGBA").save(path)


def _read_zones(path: str) -> Any:
    """The polygon layer as a GeoDataFrame in EPSG:4326, or an error dict.

    The rasters here are written in EPSG:4326 — dem_for_region asks 3DEP for imageSR=4326 — so
    the zones have to be in that frame to line up with a transform we never reproject. A layer
    declaring something else is converted; one declaring NOTHING is taken as 4326 rather than
    guessed at, which is what admin_boundary's output and every GeoJSON in this pipeline are.
    """
    try:
        import geopandas as gpd
    except Exception as exc:  # noqa: BLE001 - optional dependency
        return {"error": f"geopandas is not installed: {exc}"[:200]}
    try:
        gdf = gpd.read_file(path)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"could not read the polygon layer: {type(exc).__name__}: {exc}"[:300]}
    if gdf.empty:
        return {"error": "the polygon layer has no features"}
    try:
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(4326)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"could not reproject the polygons to EPSG:4326: {exc}"[:200],
                "hint": "supply the layer already in EPSG:4326; the rasters here are in it."}
    return gdf


def make_terrain_tools(*, default_input_file_ids: Optional[List[str]] = None) -> List[Any]:
    try:
        from langchain_core.tools import StructuredTool
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("LangChain is not installed.") from exc

    def dem_for_region(bbox: Optional[List[float]] = None, lon: Optional[float] = None,
                       lat: Optional[float] = None, file_id: Optional[str] = None,
                       buffer_m: float = 2500.0, name: Optional[str] = None,
                       size: int = _DEFAULT_SIZE, clip_to_shape: bool = True) -> str:
        """ELEVATION for a region, on the map as a raster layer, plus the GeoTIFF.

        Use for elevation, terrain, relief, altitude, "how high", "how hilly", a DEM or a
        hillshade over a place. Returns real metres — min, max, mean and relief — so a question
        about height is answered with numbers, not only a picture.

        Region, the same ways every other region tool takes one: `bbox`
        [minlon, minlat, maxlon, maxlat] (what the map's Region tool gives), `lon`+`lat` for a
        point with a `buffer_m` square around it, or a `file_id` whose extent to use. Unlike
        embed_region a POLYGON file is welcome: its bounding box is fetched and, with
        `clip_to_shape`, everything outside the shape is made transparent, so a county's
        elevation is cut to the county. Get a boundary from admin_boundary first when the user
        named a place rather than drawing one.

        Source is USGS 3DEP, which needs no credential and no quota — this call costs nothing,
        unlike the satellite-embedding tools. Coverage is the UNITED STATES and its territories;
        anywhere else comes back refused rather than blank, and there is no global DEM here.

        `size` is the frame in pixels per side (default 512, max 1536). The result states the
        metres per pixel it worked out to — quote that rather than 3DEP's nominal 10 m, which
        only holds when the box is small enough.
        """
        import numpy as np
        import rasterio
        from rasterio.io import MemoryFile

        from agent_runtime.file_store import create_output_file_from_path

        box = _resolve_bbox(bbox, lon, lat, file_id or (default_input_file_ids or [None])[0],
                            buffer_m, polygon_extent_ok=True)
        if isinstance(box, dict):
            return json.dumps({"ok": False, **box})
        px = max(_MIN_SIZE, min(int(size or _DEFAULT_SIZE), _MAX_SIZE))

        body = _fetch_dem(box, px)
        if isinstance(body, dict):
            return json.dumps({"ok": False, "region_bbox": box, **body})

        try:
            with MemoryFile(body) as mem, mem.open() as src:
                values = src.read(1, masked=True).astype("float64").filled(np.nan)
                # `src.transform` needs no PROJ; `src.crs` would parse one, and we already know
                # it — the request asked for imageSR=4326. Reading it back only creates a way
                # for a broken PROJ install to fail a request whose answer does not need it.
                transform = src.transform
                # THE EXTENT 3DEP ACTUALLY RETURNED, which is not the one we asked for. Asked
                # for a 512x512 frame over a box that is not square in degrees, the server pads
                # the shorter axis to keep its pixels square: measured on the deployed service,
                # a 0.0275-degree-tall request came back 0.0360 degrees tall — 466 m added at
                # each edge. Draping that image over the REQUESTED box squeezes it by 13% and
                # displaces every feature by up to 470 m, which is precisely the silent
                # misregistration this module's own _render docstring warns about. Only the
                # returned bounds describe these pixels.
                served = [round(float(v), 6) for v in src.bounds]
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"ok": False, "region_bbox": box,
                               "error": f"could not read the DEM 3DEP returned: {exc}"[:300]})

        # 3DEP fills a request outside its coverage with NoData and HTTP 200. Draping that
        # would put an invisible layer on the map and call the run a success.
        values[values <= _NODATA + 1] = np.nan
        if not np.isfinite(values).any():
            return json.dumps({
                "ok": False, "region_bbox": box,
                "error": "USGS 3DEP has no elevation data for this region — every pixel came "
                         "back NoData.",
                "hint": "3DEP covers the United States and its territories. For anywhere else "
                        "there is no DEM tool in this deployment; say so rather than "
                        "substituting another region."})

        if clip_to_shape and file_id:
            values = _clip_to(values, transform, file_id)
            if not np.isfinite(values).any():
                return json.dumps({
                    "ok": False, "region_bbox": box,
                    "error": "clipping to the shape left no pixels with data",
                    "hint": "Call again with clip_to_shape=false to get the bounding box."})

        stem = _slug(name or _region_tag(None, box) or "region") + "_dem"
        tmp = Path(tempfile.mkdtemp(prefix="dem_"))
        tif = tmp / f"{stem}.tif"
        crs = _wgs84()
        with rasterio.open(tif, "w", driver="GTiff", height=values.shape[0],
                           width=values.shape[1], count=1, dtype="float32",
                           crs=crs, transform=transform, nodata=float("nan")) as dst:
            dst.write(values.astype("float32"), 1)
        png = tmp / f"{stem}.png"
        _render(values, png)

        tif_rec = create_output_file_from_path(tif, filename=tif.name)
        png_rec = create_output_file_from_path(png, filename=png.name)

        layer = _raster_layer(
            png_rec, served, f"Elevation — {name or _region_tag(None, box)}",
            _layer_id("dem", _region_tag(None, box), bbox=_round_bbox(served), size=px,
                      clipped=bool(clip_to_shape and file_id)))

        out: Dict[str, Any] = {
            # region_bbox is what the pixels COVER, so every area and share computed from this
            # raster agrees with the layer on the map. requested_bbox is kept beside it because
            # the two differ, and an answer that says "the 4 km box you asked for" while the
            # data covers something else is wrong in a way nothing downstream can catch.
            "ok": True, "region_bbox": served, "requested_bbox": box,
            "source": "USGS 3DEP (no credential required)",
            "grid": [int(values.shape[0]), int(values.shape[1])],
            "ground_resolution_m": _pixel_size_m(transform, served),
            "units": "metres above sea level",
            "clipped_to_shape": bool(clip_to_shape and file_id),
            "geotiff": _file_ref(tif_rec),
            "image": _file_ref(png_rec),
            "on_map": True,
            "map_layer": layer,
            "note": "Colours are a relief ramp stretched between THIS region's own min and max, "
                    "so they are not comparable with another region's layer. The numbers are, "
                    "and the GeoTIFF holds the real metres for any further computation.",
        }
        if crs is None:
            # Said out loud rather than shipped as a silently unreferenced raster.
            out["geotiff"]["crs"] = ("not written — PROJ is misconfigured in this deployment; "
                                     "the pixels and the transform are correct and the grid is "
                                     "EPSG:4326")
        out.update(_stats(values))
        return json.dumps(out)

    def zonal_stats_for_raster(raster_file_id: str, polygons_file_id: str,
                               zone_id_field: Optional[str] = None, prefix: str = "value",
                               sibling_file_ids: Optional[List[str]] = None,
                               name: Optional[str] = None, all_touched: bool = True,
                               band: int = 1) -> str:
        """Summarise a RASTER inside each polygon, as new columns on the polygon layer.

        Turns a raster into per-zone numbers — `<prefix>_mean`, `_min`, `_max`, `_relief`,
        `_std`, and `_coverage`, the share of the zone that actually had data. Writes a GeoJSON
        carrying the original attributes plus those columns, and maps it as a choropleth.

        This is the bridge between the raster tools and the zone tools. `dem_for_region` returns
        a GeoTIFF of real metres; run it through here against the same polygons `embed_zones`
        used and the result is the layer `fit_zone_model` takes — so `label_column="elev_mean"`
        predicts terrain from embeddings, and elevation can be read beside them rather than only
        looked at.

        `prefix` names the columns: pass "elev" for elevation, "slope" for a slope raster, so two
        rasters summarised onto one layer do not overwrite each other. `zone_id_field` is the
        attribute identifying a zone (GEOID for census areas) and should be the SAME one given to
        embed_zones, or the two tables will not join.

        Check `<prefix>_coverage` before quoting a mean: a zone lying half outside the raster
        still gets one, computed from whichever pixels happened to fall inside.
        """
        import numpy as np

        tmp = None
        try:
            from rasterio.features import rasterize

            from agent_runtime.file_store import create_output_file_from_path
            from agent_runtime.langchain_geo_tools import (_index_attached, _stage_vector_source,
                                                           artifact_name)

            attached = _index_attached(default_input_file_ids)
            poly_path, tmp = _stage_vector_source(polygons_file_id, sibling_file_ids, attached)

            opened = _open_raster(raster_file_id, band=band)
            if isinstance(opened, dict):
                return json.dumps({"ok": False, **opened})
            values, transform, bounds_t = opened
            bounds = list(bounds_t)

            gdf = _read_zones(poly_path)
            if isinstance(gdf, dict):
                return json.dumps({"ok": False, **gdf})
            if zone_id_field and zone_id_field not in gdf.columns:
                return json.dumps({
                    "ok": False,
                    "error": f"no attribute named {zone_id_field!r} on this layer",
                    "available_fields": [str(c) for c in gdf.columns if c != "geometry"][:40]})

            shapes = [(geom, i + 1) for i, geom in enumerate(gdf.geometry) if geom is not None]
            if not shapes:
                return json.dumps({"ok": False, "error": "the polygon layer has no geometry"})
            zmap = rasterize(shapes, out_shape=values.shape, transform=transform, fill=0,
                             all_touched=bool(all_touched), dtype="int32")
            present = np.unique(zmap)
            present = present[present > 0]
            if present.size == 0:
                return json.dumps({
                    "ok": False,
                    "raster_bounds": [round(b, 6) for b in bounds],
                    "polygons_bounds": [round(float(b), 6) for b in gdf.total_bounds],
                    "error": "no polygon overlaps the raster — not one zone covered a pixel",
                    "hint": "the two are in different places, or the layer is not in EPSG:4326 "
                            "like the raster. Compare the two bounds above."})

            # One pass: sort the zone index once and slice, rather than build a boolean mask
            # per zone over the whole grid. embed_zones is built for 800-tract layers and this
            # is meant to be callable on the same ones.
            zflat, vflat = zmap.reshape(-1), values.reshape(-1)
            order = np.argsort(zflat, kind="stable")
            zsorted, vsorted = zflat[order], vflat[order]
            starts = np.searchsorted(zsorted, present, side="left")
            ends = np.searchsorted(zsorted, present, side="right")

            # Coverage is measured against the zone's OWN area, not against the pixels it got.
            # rasterize only draws inside the grid, so a zone hanging off the edge of the raster
            # simply has no pixels out there — count them and every such zone reports full
            # coverage while its mean comes from the half that happened to be inside. Measured:
            # a zone half outside the frame reported coverage 1.0. Pixel area and polygon area
            # are both in degrees here, so the ratio needs no projection and the latitude
            # distortion cancels within one zone. It stays APPROXIMATE — all_touched rounds a
            # zone outwards — hence the clamp and the 0.95 threshold rather than 0.999.
            pixel_area = abs(transform.a) * abs(transform.e)
            rows: List[Any] = []
            for z, lo, hi in zip(present, starts, ends):
                block = vsorted[lo:hi]
                finite = block[np.isfinite(block)]
                total = int(hi - lo)
                geom = gdf.geometry.iloc[int(z) - 1]
                expected = ((float(geom.area) / pixel_area)
                            if (geom is not None and pixel_area) else 0.0)
                covered = (min(1.0, float(finite.size) / expected) if expected > 0
                           else (float(finite.size) / total if total else 0.0))
                row: Dict[str, Any] = {
                    f"{prefix}_pixels": total,
                    f"{prefix}_coverage": round(covered, 4),
                }
                if finite.size:
                    lo_v, hi_v = float(finite.min()), float(finite.max())
                    row.update({f"{prefix}_mean": round(float(finite.mean()), 4),
                                f"{prefix}_min": round(lo_v, 4),
                                f"{prefix}_max": round(hi_v, 4),
                                f"{prefix}_relief": round(hi_v - lo_v, 4),
                                f"{prefix}_std": round(float(finite.std()), 4)})
                rows.append((int(z) - 1, row))

            cols = [f"{prefix}_{s}" for s in
                    ("mean", "min", "max", "relief", "std", "coverage", "pixels")]
            for c in cols:
                gdf[c] = np.nan
            for idx, row in rows:
                for c, v in row.items():
                    gdf.iat[idx, gdf.columns.get_loc(c)] = v

            with_data = sum(1 for _i, r in rows if f"{prefix}_mean" in r)
            if with_data == 0:
                return json.dumps({
                    "ok": False, "zone_count": int(len(gdf)),
                    "zones_overlapping_raster": int(present.size),
                    "error": "every overlapping zone was all-NoData — no zone has a value",
                    "hint": "the polygons sit over a hole in the raster. dem_for_region refuses "
                            "an all-NoData frame, so a DEM that got here has data somewhere; "
                            "check the extents overlap where that data actually is."})

            out_dir = Path(tempfile.mkdtemp(prefix="zonal_"))
            gj = out_dir / artifact_name(name, "geojson", default=f"{prefix}_by_zone")
            gdf.to_file(gj, driver="GeoJSON")
            rec = create_output_file_from_path(gj, filename=gj.name)

            preview = []
            for idx, row in rows[:10]:
                item = {k: v for k, v in row.items()
                        if k.endswith(("_mean", "_relief", "_coverage"))}
                if zone_id_field:
                    item["zone"] = str(gdf.iloc[idx][zone_id_field])
                preview.append(item)

            partial = sum(1 for _i, r in rows if r.get(f"{prefix}_coverage", 0.0) < 0.95)
            out: Dict[str, Any] = {
                "ok": True,
                "zone_count": int(len(gdf)),
                "zones_overlapping_raster": int(present.size),
                "zones_with_values": int(with_data),
                "zones_partially_covered": int(partial),
                "columns_added": cols,
                "zone_id_field": zone_id_field,
                "geojson": _file_ref(rec),
                "raster_bounds": [round(b, 6) for b in bounds],
                "on_map": True,
                "map_layer": {
                    "url": rec.get("download_url"),
                    "id": _layer_id("zonal", polygons_file_id, raster=raster_file_id,
                                    polygons=polygons_file_id, prefix=prefix,
                                    zone_id_field=zone_id_field),
                    "label": _layer_label(f"{prefix} by zone", _region_tag(name)),
                    "render": "choropleth", "style_by": f"{prefix}_mean",
                    "source": "analysis", "count": int(with_data)},
                "preview": preview,
                "note": f"{prefix}_mean is the column to fit on: pass it to fit_zone_model as "
                        f"label_column with the SAME zone_id_field embed_zones used. "
                        f"{prefix}_relief is max-min WITHIN a zone, which is not the "
                        f"region-wide relief dem_for_region reports.",
            }
            if partial:
                out["warning"] = (f"{partial} zone(s) are under 95% covered by the raster; their "
                                  f"means come from the pixels that were inside. Check "
                                  f"{prefix}_coverage before quoting one.")
            return json.dumps(out)
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]})
        finally:
            if tmp:
                import shutil
                shutil.rmtree(tmp, ignore_errors=True)

    def terrain_derivative(raster_file_id: str, kind: str = "slope",
                           name: Optional[str] = None, azimuth: float = 315.0,
                           altitude: float = 45.0) -> str:
        """SLOPE, aspect or hillshade from an elevation raster, on the map.

        Use for "how steep", "which way does it face", "show the terrain" — anything about the
        SHAPE of the ground rather than its height. Takes the GeoTIFF dem_for_region wrote.

        `kind` is one of:
          slope     — degrees from horizontal, 0 flat to 90 vertical. Also reported in percent,
                      which is what road and drainage standards are written in.
          aspect    — the compass direction the ground faces, 0-360 clockwise from north. Flat
                      ground has no aspect and comes back NoData rather than an arbitrary 0.
          hillshade — a shaded-relief picture for looking at, 0-255. It is not a measurement;
                      use slope for anything a number is wanted from.

        Slope is computed in METRES, from the pixel's real ground size at this latitude — a
        gradient taken in degrees of longitude would report a slope that changes with how far
        north the region is. Steepness on a resampled DEM is a property of the pixel size as
        much as the ground, so the result states the ground resolution it worked at.
        """
        import numpy as np

        try:
            kind = str(kind or "slope").strip().lower()
            if kind not in {"slope", "aspect", "hillshade"}:
                return json.dumps({
                    "ok": False, "error": f"unknown kind {kind!r}",
                    "hint": "kind is one of: slope, aspect, hillshade"})

            opened = _open_raster(raster_file_id)
            if isinstance(opened, dict):
                return json.dumps({"ok": False, **opened})
            values, transform, bounds = opened

            # Ground size of one pixel. The x spacing shrinks with the cosine of latitude and
            # the y spacing does not, so a single "resolution" would be wrong in one axis; both
            # are passed to the gradient separately.
            mid_lat = (bounds[1] + bounds[3]) / 2.0
            dx_m = abs(transform.a) * 111_320.0 * math.cos(math.radians(mid_lat))
            dy_m = abs(transform.e) * 110_540.0
            if dx_m <= 0 or dy_m <= 0:
                return json.dumps({"ok": False,
                                   "error": "the raster's pixel size works out to zero on the "
                                            "ground; it is not a geographic grid"})

            # np.gradient over NaN spreads it; filling with the nearest finite value would
            # invent a slope at the edge of the data. The holes are restored afterwards so a
            # NoData pixel has no slope rather than a fabricated one.
            holes = ~np.isfinite(values)
            filled = np.where(holes, float(np.nanmean(values)) if (~holes).any() else 0.0,
                              values)
            # Rows run north -> south (the transform's e is negative), so the row gradient is
            # already -dz/dnorth; negated here so dz_dy points north like a map reader expects.
            dz_dy, dz_dx = np.gradient(filled, dy_m, dx_m)
            dz_dy = -dz_dy

            if kind == "slope":
                grid = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
                grid[holes] = np.nan
                cmap, label, unit = "YlOrRd", "Slope", "degrees from horizontal"
            elif kind == "aspect":
                grid = (np.degrees(np.arctan2(dz_dy, -dz_dx)) + 360.0) % 360.0
                # Flat ground faces nowhere. Reporting 0 there would put a hard "due north"
                # band across every plain in the region.
                grid[np.hypot(dz_dx, dz_dy) < 1e-9] = np.nan
                grid[holes] = np.nan
                cmap, label, unit = "twilight", "Aspect", "degrees clockwise from north"
            else:
                az, alt = math.radians(360.0 - float(azimuth) + 90.0), math.radians(float(altitude))
                slope = np.arctan(np.hypot(dz_dx, dz_dy))
                aspect = np.arctan2(dz_dy, -dz_dx)
                grid = 255.0 * ((np.sin(alt) * np.cos(slope)) +
                                (np.cos(alt) * np.sin(slope) * np.cos(az - aspect)))
                grid = np.clip(grid, 0.0, 255.0)
                grid[holes] = np.nan
                cmap, label, unit = "gray", "Hillshade", "0-255, not a measurement"

            stem = _slug(name or _region_tag(None, list(bounds)) or "region") + f"_{kind}"
            tmp_dir = Path(tempfile.mkdtemp(prefix=f"{kind}_"))
            tif = tmp_dir / f"{stem}.tif"
            _write_grid(tif, grid, transform)
            png = tmp_dir / f"{stem}.png"
            _render_grid(grid, png, cmap)

            from agent_runtime.file_store import create_output_file_from_path
            tif_rec = create_output_file_from_path(tif, filename=tif.name)
            png_rec = create_output_file_from_path(png, filename=png.name)
            box = list(bounds)
            layer = _raster_layer(
                png_rec, box, f"{label} — {name or _region_tag(None, box)}",
                _layer_id(kind, _region_tag(None, box), bbox=_round_bbox(box)))

            finite = grid[np.isfinite(grid)]
            out: Dict[str, Any] = {
                "ok": True, "kind": kind, "region_bbox": [round(float(b), 6) for b in box],
                "units": unit,
                "ground_resolution_m": round(max(dx_m, dy_m), 2),
                "geotiff": _file_ref(tif_rec),
                "image": _file_ref(png_rec),
                "on_map": True, "map_layer": layer,
            }
            if finite.size:
                out.update({"min": round(float(finite.min()), 2),
                            "max": round(float(finite.max()), 2),
                            "mean": round(float(finite.mean()), 2)})
                if kind == "slope":
                    out["mean_percent"] = round(
                        float(np.tan(np.radians(finite.mean())) * 100.0), 2)
                    out["steep_over_15_deg_fraction"] = round(
                        float((finite > 15.0).mean()), 4)
            out["note"] = (f"Computed at {out['ground_resolution_m']} m per pixel. Slope is a "
                           f"property of that pixel size as much as of the ground — a coarser "
                           f"DEM flattens it — so quote the resolution beside the number. Pass "
                           f"the GeoTIFF to zonal_stats_for_raster with prefix='{kind}' for a "
                           f"per-zone figure.")
            return json.dumps(out)
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]})

    def inundation_at_level(raster_file_id: str, level_m: Optional[float] = None,
                            depth_above_min_m: Optional[float] = None,
                            name: Optional[str] = None) -> str:
        """What is under water at a given elevation — the flooded AREA and a layer showing it.

        A bathtub fill: every pixel at or below the level is called flooded. Use it for "what
        floods at 3 metres", sea-level rise, a reservoir level, or a quick exposure figure to
        pair with population.

        Give the level one of two ways: `level_m` is an absolute elevation above sea level;
        `depth_above_min_m` is that many metres above the region's own lowest point, which is
        the one to use for a river or a valley where the absolute figure is not known.

        THIS IS NOT A FLOOD MODEL. It takes no account of where the water comes from, whether
        it can reach a low spot at all, or of flow, drainage and defences — an inland hollow
        below the level is coloured exactly like the riverbank. It is honest as "ground below
        this height", which is a real question, and it is not a floodplain. Where a regulatory
        floodplain is what is wanted, say that this is not one.

        Returns the flooded area in km2, the share of the region, and a mask GeoTIFF to hand to
        zonal_stats_for_raster for a per-zone flooded fraction.
        """
        import numpy as np

        try:
            if level_m is None and depth_above_min_m is None:
                return json.dumps({
                    "ok": False,
                    "error": "no level given",
                    "hint": "pass level_m for an absolute elevation, or depth_above_min_m for "
                            "a depth above the region's own lowest point"})

            opened = _open_raster(raster_file_id)
            if isinstance(opened, dict):
                return json.dumps({"ok": False, **opened})
            values, transform, bounds = opened
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                return json.dumps({"ok": False,
                                   "error": "the raster has no elevation data at all"})

            lo, hi = float(finite.min()), float(finite.max())
            level = float(level_m) if level_m is not None else lo + float(depth_above_min_m)
            # A level outside the region's range is not an error — "nothing floods" is a real
            # answer — but it is stated rather than left to be inferred from a blank layer.
            verdict = None
            if level < lo:
                verdict = (f"nothing is below {round(level, 2)} m: the lowest ground in this "
                           f"region is {round(lo, 2)} m")
            elif level >= hi:
                verdict = (f"the whole region is at or below {round(level, 2)} m; its highest "
                           f"point is {round(hi, 2)} m")

            wet = np.isfinite(values) & (values <= level)
            mid_lat = (bounds[1] + bounds[3]) / 2.0
            px_m2 = (abs(transform.a) * 111_320.0 * math.cos(math.radians(mid_lat))) * \
                    (abs(transform.e) * 110_540.0)
            wet_km2 = float(wet.sum()) * px_m2 / 1_000_000.0
            land_km2 = float(np.isfinite(values).sum()) * px_m2 / 1_000_000.0

            # Depth of water, not a flag: a mask says where, this says how deep, and a mean
            # depth is what an exposure figure is usually multiplied by.
            depth = np.where(wet, level - values, np.nan)

            stem = _slug(name or _region_tag(None, list(bounds)) or "region") + "_inundation"
            tmp_dir = Path(tempfile.mkdtemp(prefix="inund_"))
            tif = tmp_dir / f"{stem}.tif"
            _write_grid(tif, depth, transform)
            png = tmp_dir / f"{stem}.png"
            _render_grid(depth, png, "Blues", vmin=0.0)

            from agent_runtime.file_store import create_output_file_from_path
            tif_rec = create_output_file_from_path(tif, filename=tif.name)
            png_rec = create_output_file_from_path(png, filename=png.name)
            box = list(bounds)
            layer = _raster_layer(
                png_rec, box, f"Under {round(level, 2)} m — {name or _region_tag(None, box)}",
                _layer_id("inundation", _region_tag(None, box), bbox=_round_bbox(box),
                          level=round(level, 3)))

            wet_depths = depth[np.isfinite(depth)]
            out: Dict[str, Any] = {
                "ok": True,
                "level_m": round(level, 2),
                "level_from": ("absolute" if level_m is not None
                               else f"{depth_above_min_m} m above the region minimum "
                                    f"({round(lo, 2)} m)"),
                "region_min_m": round(lo, 2), "region_max_m": round(hi, 2),
                "flooded_km2": round(wet_km2, 4),
                "region_km2": round(land_km2, 4),
                "flooded_fraction": round(wet_km2 / land_km2, 4) if land_km2 else 0.0,
                "mean_depth_m": (round(float(wet_depths.mean()), 2) if wet_depths.size else 0.0),
                "max_depth_m": (round(float(wet_depths.max()), 2) if wet_depths.size else 0.0),
                "depth_geotiff": _file_ref(tif_rec),
                "image": _file_ref(png_rec),
                "on_map": True, "map_layer": layer,
                "note": "A bathtub fill: ground at or below the level, with no account of where "
                        "water comes from, whether it could reach a hollow, or of drainage and "
                        "defences. Call it 'ground below this height', not a floodplain. Pass "
                        "the depth GeoTIFF to zonal_stats_for_raster with prefix='depth' for a "
                        "per-zone figure.",
            }
            if verdict:
                out["verdict"] = verdict
            return json.dumps(out)
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]})

    meta = {"category": "analysis"}
    return [StructuredTool.from_function(func=accept_null_defaults(dem_for_region), name="dem_for_region",
                                         metadata=meta),
            StructuredTool.from_function(func=accept_null_defaults(zonal_stats_for_raster),
                                         name="zonal_stats_for_raster",
                                         metadata=meta),
            StructuredTool.from_function(func=accept_null_defaults(terrain_derivative),
                                         name="terrain_derivative", metadata=meta),
            StructuredTool.from_function(func=accept_null_defaults(inundation_at_level),
                                         name="inundation_at_level", metadata=meta)]


__all__ = ["make_terrain_tools"]
