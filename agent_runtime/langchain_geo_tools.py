"""Vector / shapefile tools for the analysis and code agents.

Lets the agents READ, VISUALIZE, and ANALYZE uploaded vector data — in particular
Census TIGER/Line shapefiles, which arrive either as a ``.zip`` or as an extracted
set of separate files (``.shp`` + ``.shx`` + ``.dbf`` + ``.prj`` …).

The hard part is the extracted case: the file store saves each upload as
``uploads/<file_id>__<filename>``, so the sidecars are NOT co-located with the
``.shp`` under a shared basename — GDAL/pyogrio can't find them. ``_stage_vector_source``
reconstructs a readable shapefile by copying the ``.shp`` and its sibling uploads into
one temp dir under a common basename. The siblings are auto-discovered among the
conversation's attached files by matching the basename stem, so the model only has to
reference any ONE component (the ``.shp`` or a sidecar). A ``.zip`` is read directly via
GDAL's ``/vsizip/`` virtual filesystem (no extraction needed).

Backend: geopandas + pyogrio (already in the agent runtime image alongside system GDAL).
Heavy imports are deferred into the tool bodies so importing this module never fails the
agent boot. Every tool returns a JSON string and NEVER raises — on error it returns
``{"ok": false, "error": "..."}``.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Above this, a GeoJSON is too bulky to ship and parquet is written instead.
_GEOJSON_MAX_FEATURES = int(os.getenv("AGENT_GEOJSON_MAX_FEATURES", "60000"))
# Attributes carried on a density layer so clicking a point still says something.
_HEATMAP_KEEP_COLUMNS = int(os.getenv("AGENT_HEATMAP_KEEP_COLUMNS", "6"))
# A full city-scale point set is genuinely servable: 128,855 incidents measured at 35.9 MB
# with 6 attributes and 15.8 MB geometry-only, built in ~3s, and deck.gl draws that count
# without effort. So show the whole population by default and only sample well beyond it.
_MAP_LAYER_MAX_FEATURES = int(os.getenv("AGENT_MAP_LAYER_MAX_FEATURES", "150000"))
# Past this, a density layer drops attributes instead of being sampled: the render only needs
# position, and dropping them more than halves the payload (35.9 MB -> 15.8 MB).
_HEATMAP_LEAN_ABOVE = int(os.getenv("AGENT_HEATMAP_LEAN_ABOVE", "60000"))

_SELF_CONTAINED = {".geojson", ".json", ".gpkg", ".parquet", ".geoparquet", ".fgb", ".kml"}
# Components of an (extracted) ESRI shapefile set — any one of these is enough to reference it.
_SHAPE_PARTS = {".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".sbn", ".sbx", ".qpj", ".aih", ".ain"}


def _resolve(ref: str) -> Tuple[Path, Optional[Dict[str, Any]]]:
    """Resolve an uploaded file_id (preferred) or an on-disk path to (Path, record)."""
    from agent_runtime.file_store import get_file_record, resolve_file_id

    ref = str(ref or "").strip()
    if not ref:
        raise ValueError("empty file reference")
    record = get_file_record(ref)
    if record:
        return resolve_file_id(ref), record
    p = Path(ref).expanduser()
    if p.is_file():
        return p.resolve(), None
    raise ValueError(f"unknown file_id or path: {ref}")


def _true_name(path: Path, record: Optional[Dict[str, Any]]) -> str:
    """The original filename (on-disk name is mangled as <file_id>__<name>)."""
    if record and record.get("filename"):
        return str(record["filename"])
    name = path.name
    return name.split("__", 1)[1] if "__" in name else name


def _index_attached(file_ids: Optional[List[str]]) -> List[Dict[str, Any]]:
    """Resolve every conversation-attached file_id to ``{id, name, path}``.

    Unresolvable ids are skipped. Used to auto-discover shapefile sidecars by basename.
    """
    idx: List[Dict[str, Any]] = []
    for fid in file_ids or []:
        try:
            p, rec = _resolve(fid)
        except Exception:
            continue
        idx.append({"id": str(fid), "name": _true_name(p, rec), "path": p})
    return idx


def _stage_vector_source(ref: str, sibling_file_ids: Optional[List[str]],
                         attached: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, Optional[str]]:
    """Return ``(gdal_readable_path, tempdir_to_cleanup_or_None)``.

    - ``.zip``  -> ``/vsizip/<abs>`` (GDAL reads the zipped shapefile in place).
    - a shapefile component (``.shp`` OR a sidecar like ``.dbf``) -> reconstruct a readable
      shapefile by copying the ``.shp`` and every sibling into one temp dir under a shared
      basename. Siblings come from (a) explicit ``sibling_file_ids`` and (b) auto-discovery
      among the conversation's ``attached`` files sharing the same basename stem — so the
      model only has to reference any ONE component.
    - self-contained (.geojson/.gpkg/.parquet/…) -> the path as-is.
    """
    path, record = _resolve(ref)
    name = _true_name(path, record)
    ext = Path(name).suffix.lower()

    if ext == ".zip":
        return f"/vsizip/{path.resolve()}", None
    if ext in _SELF_CONTAINED:
        return str(path), None
    if ext in _SHAPE_PARTS:
        stem = Path(name).stem
        # dest_filename -> source Path, starting with the referenced component itself.
        members: Dict[str, Path] = {name: path}
        for sib in sibling_file_ids or []:  # explicit siblings the caller passed
            try:
                sp, srec = _resolve(sib)
            except Exception:
                continue
            members.setdefault(_true_name(sp, srec), sp)
        for entry in attached or []:  # auto-discover the rest of the set by basename
            if Path(entry["name"]).stem == stem:
                members.setdefault(entry["name"], entry["path"])
        shp_name = next((n for n in members if Path(n).suffix.lower() == ".shp"), None)
        if not shp_name:
            raise ValueError(
                f"no .shp found for '{name}'; attach the .shp together with its .shx/.dbf/.prj, "
                "or upload the shapefile as a single .zip")
        tmp = tempfile.mkdtemp(prefix="vec_")
        shp_stem = Path(shp_name).stem
        for n, sp in members.items():
            sext = Path(n).suffix.lower()
            if sext:
                shutil.copyfile(sp, Path(tmp) / f"{shp_stem}{sext}")
        return str(Path(tmp) / f"{shp_stem}.shp"), tmp
    # Unknown extension: let GDAL attempt to read it directly.
    return str(path), None


def artifact_name(stem: Optional[str], suffix: str, *, source: Optional[str] = None,
                  default: str = "output") -> str:
    """A short, purpose-bearing filename: ``<stem>.<suffix>``.

    Every run used to emit the same handful of fixed names (``vector.geojson``,
    ``vector_plot.png``, ``executed_code.py``), so a conversation ended up with several
    identical entries in its download list and no way to tell them apart. Prefer a name
    the caller supplies; otherwise fall back to the input file's own stem, which is
    already meaningful (``chicago_tracts.zip`` -> ``chicago_tracts.geojson``).
    """
    raw = (stem or "").strip() or Path(str(source or "")).stem or default
    slug = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_").lower()[:48] or default
    return f"{slug}.{suffix.lstrip('.')}"


def _epsg(crs: Any) -> Optional[str]:
    if crs is None:
        return None
    try:
        code = crs.to_epsg()
        if code:
            return f"EPSG:{code}"
    except Exception:
        pass
    try:
        return str(crs)
    except Exception:
        return None


# --- tabular (CSV/TSV) point data -------------------------------------------------
# gpd.read_file() on a CSV returns a plain DataFrame with NO geometry, so downstream calls fail
# obscurely (e.g. "'DataFrame' object has no attribute 'to_file'"). Build point geometry from
# coordinate columns instead, accepting decimal degrees OR DMS strings (24°07'12.0"N) — field
# datasets and paper supplements very often ship coordinates in DMS.
_LAT_KEYS = ("latitude", "lat", "y", "ycoord", "y_coord", "northing", "decimallatitude")
_LON_KEYS = ("longitude", "lon", "long", "lng", "x", "xcoord", "x_coord", "easting", "decimallongitude")
_DMS_RE = re.compile(
    r"""^\s*(-?\d+(?:\.\d+)?)\s*[°d:\s]\s*      # degrees
        (?:(\d+(?:\.\d+)?)\s*['m\u2032:\s]\s*)?  # minutes (optional)
        (?:(\d+(?:\.\d+)?)\s*["s\u2033]?\s*)?     # seconds (optional)
        ([NSEWnsew])?\s*$""",
    re.VERBOSE,
)


def parse_coordinate(value: Any) -> Optional[float]:
    """Decimal degrees from a number or a DMS/DM string; None when unparseable.

    Accepts ``24°07'12.0"N``, ``24 07 12 N``, ``121°14.5'E``, ``-97.1833`` and similar.
    A S/W hemisphere suffix negates the value.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    match = _DMS_RE.match(text.replace("\u2019", "'").replace("\u201d", '"'))
    if not match:
        return None
    deg, minutes, seconds, hemi = match.groups()
    try:
        dec = abs(float(deg)) + (float(minutes or 0) / 60.0) + (float(seconds or 0) / 3600.0)
    except (TypeError, ValueError):
        return None
    if str(deg).startswith("-") or (hemi or "").upper() in ("S", "W"):
        dec = -dec
    return dec


def _pick_coord_column(columns: Any, keys: Sequence[str]) -> Optional[str]:
    lookup = {str(c).strip().lower().replace(" ", "").replace("_", ""): c for c in columns}
    for key in keys:
        hit = lookup.get(key.replace("_", ""))
        if hit is not None:
            return hit
    for norm, original in lookup.items():       # substring fallback ("site_latitude")
        if any(norm.endswith(k.replace("_", "")) or k.replace("_", "") in norm for k in keys):
            return original
    return None


def dataframe_to_points(df: Any) -> Any:
    """Build an EPSG:4326 point GeoDataFrame from a table's coordinate columns.

    Raises ValueError when no usable coordinate columns/values are present, so callers can report a clear message instead of failing later on a missing ``.to_file``.
    """
    import geopandas as gpd

    lat_col = _pick_coord_column(df.columns, _LAT_KEYS)
    lon_col = _pick_coord_column(df.columns, _LON_KEYS)
    if lat_col is None or lon_col is None:
        raise ValueError(
            "tabular input has no recognizable coordinate columns; expected latitude/longitude "
            f"(or lat/lon, y/x). Columns present: {list(df.columns)[:12]}"
        )
    lats = [parse_coordinate(v) for v in df[lat_col]]
    lons = [parse_coordinate(v) for v in df[lon_col]]
    keep = [
        i for i, (la, lo) in enumerate(zip(lats, lons))
        if la is not None and lo is not None and -90 <= la <= 90 and -180 <= lo <= 180
    ]
    if not keep:
        raise ValueError(
            f"could not parse any coordinates from columns {lat_col!r}/{lon_col!r} "
            "(supported: decimal degrees or DMS such as 24°07'12.0\"N)"
        )
    sub = df.iloc[keep].copy()
    sub["latitude"] = [lats[i] for i in keep]
    sub["longitude"] = [lons[i] for i in keep]
    return gpd.GeoDataFrame(
        sub, geometry=gpd.points_from_xy(sub["longitude"], sub["latitude"]), crs="EPSG:4326"
    )


def read_vector(read_path: Any, layer: Optional[str] = None) -> Any:
    """Load any supported vector source as a GeoDataFrame.

    Falls back to tabular point construction when the source carries no geometry (a CSV/TSV of
    coordinates), which is why plain spreadsheets can be mapped/exported like any other layer.
    """
    import geopandas as gpd

    gdf = None
    # GDAL (read_file) cannot open (geo)parquet, and these tools WRITE parquet —
    # vector_spatial_join / reproject_vector emit it — so without this branch a tool's own
    # output is unreadable by every other tool in the set: an observed spatial join produced
    # 128,464 joined features that inspect_vector, render_map_image and pyqgis_layer_summary all
    # then refused as "not recognized as being in a supported file format".
    if str(read_path).lower().endswith((".parquet", ".geoparquet")):
        try:
            return gpd.read_parquet(read_path)
        except Exception:
            import pandas as pd

            return dataframe_to_points(pd.read_parquet(read_path))
    try:
        gdf = gpd.read_file(read_path, layer=layer) if layer else gpd.read_file(read_path)
    except Exception as exc:
        table_error = exc
        gdf = None
    else:
        table_error = None
    has_geom = (
        gdf is not None
        and hasattr(gdf, "geometry")
        and "geometry" in getattr(gdf, "columns", [])
        and not gdf.geometry.isna().all()
    )
    if has_geom:
        return gdf
    # No geometry: treat it as a table of coordinates.
    import pandas as pd

    frame = gdf
    if frame is None or not hasattr(frame, "columns"):
        suffix = str(read_path).lower()
        sep = "\t" if suffix.endswith((".tsv", ".tab")) else None
        try:
            frame = pd.read_csv(read_path, sep=sep, engine="python")
        except Exception as exc:
            raise ValueError(f"unreadable vector/tabular source: {table_error or exc}") from exc
    return dataframe_to_points(frame)

# Qualitative palette for render="categories". Distinct hues rather than a ramp: these are
# class NAMES with no order, so a sequential scale would imply one.
_CATEGORY_COLORS = [
    [215, 25, 28, 200], [44, 123, 182, 200], [253, 174, 97, 200], [171, 217, 233, 200],
    [26, 150, 65, 200], [123, 50, 148, 200], [255, 217, 47, 200], [166, 97, 26, 200],
    [230, 97, 158, 200], [102, 194, 165, 200], [140, 140, 140, 200], [8, 81, 156, 200],
]

def make_langchain_geo_tools(default_input_file_ids: Optional[List[str]] = None) -> List[Any]:
    """Build the vector/shapefile StructuredTools (geopandas-backed).

    ``default_input_file_ids`` is the conversation's attached file set. The tools use it to
    auto-discover shapefile sidecars (.shx/.dbf/.prj) by basename, so the model only has to
    reference the .shp (or any single component) — passing ``sibling_file_ids`` is optional.
    """
    from langchain_core.tools import StructuredTool

    _attached = _index_attached(default_input_file_ids)

    def _stage(ref, sibling_file_ids=None):
        return _stage_vector_source(ref, sibling_file_ids, _attached)

    _SIB = ("For an uploaded shapefile, just pass the .shp's file_id (or any single component) — "
            "the tool auto-finds the .shx/.dbf/.prj among the attached files. You may also pass "
            "the other components as sibling_file_ids, or upload the shapefile as a single .zip.")

    # Observed: after heatmap_image returned heatmap.png, the peer passed that PNG's file_id to
    # add_map_layer, got "unreadable vector/tabular source" plus a shapefile hint that did not
    # apply, and told the user an interactive heat map was impossible. An image genuinely cannot
    # become a layer — but the dataset it was drawn from can, so name it instead of dead-ending.
    _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg", ".pdf"}
    _MAPPABLE_EXTS = {".geojson", ".json", ".csv", ".tsv", ".shp", ".zip", ".gpkg",
                      ".parquet", ".geoparquet", ".gml", ".kml"}

    def _mappable_attached(exclude: Optional[str] = None) -> List[Dict[str, str]]:
        """Attached files add_map_layer can actually read, as {file_id, filename}."""
        out = []
        for a in _attached:
            name = str(a.get("name") or "")
            if a.get("id") == exclude or Path(name).suffix.lower() not in _MAPPABLE_EXTS:
                continue
            out.append({"file_id": str(a.get("id")), "filename": name})
        return out[:20]

    def _unmappable_input(file_id: str) -> Optional[Dict[str, Any]]:
        """An actionable redirect when *file_id* is an image, else None."""
        try:
            path, rec = _resolve(file_id)
        except Exception:
            return None
        name = _true_name(path, rec)
        if Path(name).suffix.lower() not in _IMAGE_EXTS:
            return None
        return {"ok": False,
                "error": f"{name} is an image, and an image cannot become an interactive map "
                         f"layer — it has no geometry to pan, zoom or click.",
                "hint": "Pass the DATASET this picture was drawn from instead. add_map_layer "
                        "reads GeoJSON, CSV/TSV with lat-lon columns, shapefile (.shp/.zip), "
                        "GeoPackage and GeoParquet. Keep the image as a download alongside it.",
                "mappable_file_ids": _mappable_attached(exclude=file_id)}

    def inspect_vector(file_id: str, sibling_file_ids: Optional[List[str]] = None,
                       layer: Optional[str] = None) -> str:
        """Read a vector dataset's metadata (CRS, extent, geometry type, feature count,
        attribute schema) WITHOUT loading all geometry. Accepts a TIGER/shapefile .zip,
        a .shp (+ sidecars), GeoJSON, GeoPackage, or GeoParquet. """
        tmp = None
        try:
            import pyogrio
            read_path, tmp = _stage(file_id, sibling_file_ids)
            # pyogrio/GDAL cannot open (geo)parquet, but these tools WRITE it and this
            # docstring promises to accept it — read those through geopandas instead of
            # reporting a tool's own output as an unsupported format.
            if str(read_path).lower().endswith((".parquet", ".geoparquet")):
                gdf = read_vector(read_path, layer)
                geom_types = sorted({str(t) for t in gdf.geometry.geom_type.unique()}) if hasattr(gdf, "geometry") else []
                tb = gdf.total_bounds
                return json.dumps({
                    "ok": True, "driver": "Parquet", "layers": [],
                    "feature_count": int(len(gdf)),
                    "geometry_type": geom_types[0] if len(geom_types) == 1 else (geom_types or None),
                    "crs": _epsg(getattr(gdf, "crs", None)),
                    "columns": [{"name": str(c), "dtype": str(gdf[c].dtype)} for c in gdf.columns],
                    "bounds": [float(x) for x in tb],
                }, default=str)
            try:
                layers = [str(n) for n in (pyogrio.list_layers(read_path)[:, 0])]
            except Exception:
                layers = []
            info = pyogrio.read_info(read_path, layer=layer) if layer else pyogrio.read_info(read_path)
            # fields/dtypes come back as numpy arrays — don't use `or []` (ambiguous truth value).
            _fields = info.get("fields")
            _dtypes = info.get("dtypes")
            fields = list(_fields) if _fields is not None else []
            dtypes = list(_dtypes) if _dtypes is not None else []
            cols = [{"name": str(f), "dtype": str(d)} for f, d in zip(fields, dtypes)]
            tb = info.get("total_bounds")
            payload = {
                "ok": True,
                "driver": info.get("driver"),
                "layers": layers,
                "feature_count": int(info.get("features") or 0),
                "geometry_type": info.get("geometry_type"),
                "crs": info.get("crs"),
                "columns": cols,
                "bounds": [float(x) for x in tb] if tb is not None else None,
            }
            # A CSV/TSV of coordinates has no geometry OF ITS OWN, so pyogrio reports none —
            # which reads as "not mappable". Report the geometry we can DERIVE from its
            # coordinate columns instead, so the caller knows the layer is usable as points.
            if not payload["geometry_type"]:
                try:
                    derived = read_vector(read_path, layer)
                    b = derived.total_bounds
                    payload.update({
                        "geometry_type": "Point",
                        "crs": "EPSG:4326",
                        "feature_count": int(len(derived)),
                        "bounds": [float(x) for x in b],
                        "geometry_source": "derived from coordinate columns "
                                           "(decimal degrees or DMS) — usable directly with "
                                           "render_map_image / vector_to_geojson / spatial tools",
                    })
                except Exception as derive_exc:
                    payload["geometry_note"] = (
                        f"no geometry in the file and none could be derived: {derive_exc}")
            return json.dumps(payload, default=str)
        except Exception as exc:
            return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}", "hint": _SIB})
        finally:
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)

    def render_map_image(file_id: str, column: Optional[str] = None,
                    sibling_file_ids: Optional[List[str]] = None, layer: Optional[str] = None,
                    max_features: int = 50000, cmap: str = "viridis", title: Optional[str] = None,
                    name: Optional[str] = None) -> str:
        """Render a vector dataset to a PNG map and return a downloadable file_id. Pass
        `column` for a choropleth. Large layers are downsampled to `max_features`.
        `name` is a short slug describing what the map shows (e.g. "chicago_rivers"); it
        becomes the download filename, so several maps in one conversation stay tellable
        apart. Defaults to the input file's own name. """
        tmp = None
        png = None
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import geopandas as gpd
            from agent_runtime.file_store import create_output_file_from_path

            read_path, tmp = _stage(file_id, sibling_file_ids)
            gdf = read_vector(read_path, layer)
            total = len(gdf)
            downsampled = total > max_features
            if downsampled:
                gdf = gdf.sample(int(max_features), random_state=0)
            fig, ax = plt.subplots(figsize=(9, 9))
            if column and column in gdf.columns:
                gdf.plot(column=column, legend=True, cmap=cmap, ax=ax, linewidth=0.3, edgecolor="#444")
            else:
                gdf.plot(ax=ax, color="#3aa9a0", edgecolor="#1c5a97", linewidth=0.4, alpha=0.85)
            ax.set_axis_off()
            ax.set_title(title or f"{total} features" + (f" (showing {len(gdf)})" if downsampled else ""))
            fd, png = tempfile.mkstemp(prefix="vec_plot_", suffix=".png")
            os.close(fd)
            fig.savefig(png, bbox_inches="tight", dpi=150)
            # A blank figure is worse than an error: it looks like a result. Check before
            # registering it as a download.
            try:
                from agent_runtime.layer_qa import inspect_image
                img_qa = inspect_image(str(png))
            except Exception:
                img_qa = {"ok": True}
            plt.close(fig)
            rec = create_output_file_from_path(
                png, filename=artifact_name(name, "png", source=str(read_path),
                                            default="vector_plot"))
            return json.dumps({
                "ok": True, "file_id": rec["file_id"], "filename": rec.get("filename"),
                "download_url": rec.get("download_url"),
                "plotted_features": int(len(gdf)), "total_features": int(total),
                "downsampled": bool(downsampled), "crs": _epsg(getattr(gdf, "crs", None)),
                **({"visual_warning": "; ".join(img_qa.get("problems") or [])}
                   if not img_qa.get("ok") else {}),
            }, default=str)
        except Exception as exc:
            return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}", "hint": _SIB})
        finally:
            if png and os.path.exists(png):
                try: os.remove(png)
                except OSError: pass
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)

    def vector_to_geojson(file_id: str, sibling_file_ids: Optional[List[str]] = None,
                          layer: Optional[str] = None, target_crs: str = "EPSG:4326",
                          name: Optional[str] = None) -> str:
        """Convert a vector dataset to GeoJSON (reprojected to target_crs, default WGS84)
        and return a downloadable file_id. `name` is a short slug describing the contents
        (e.g. "chicago_rivers"); it becomes the download filename and the map layer label.
        Defaults to the input file's own name. """
        tmp = None
        out = None
        try:
            import geopandas as gpd
            from agent_runtime.file_store import create_output_file_from_path

            read_path, tmp = _stage(file_id, sibling_file_ids)
            gdf = read_vector(read_path, layer)
            if target_crs and getattr(gdf, "crs", None) is not None:
                gdf = gdf.to_crs(target_crs)
            fname = artifact_name(name, "geojson", source=str(read_path),
                                  default="vector")
            out = Path(tempfile.mkdtemp(prefix="vec_gj_")) / fname
            gdf.to_file(out, driver="GeoJSON")
            rec = create_output_file_from_path(out, filename=fname)
            return json.dumps({
                "ok": True, "file_id": rec["file_id"], "filename": rec.get("filename"),
                "download_url": rec.get("download_url"),
                "feature_count": int(len(gdf)), "crs": _epsg(getattr(gdf, "crs", None)),
            }, default=str)
        except Exception as exc:
            return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}", "hint": _SIB})
        finally:
            if out and out.parent.exists():
                shutil.rmtree(out.parent, ignore_errors=True)
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)

    def reproject_vector(file_id: str, target_crs: str, sibling_file_ids: Optional[List[str]] = None,
                         layer: Optional[str] = None) -> str:
        """Reproject a vector dataset to target_crs (e.g. "EPSG:5070") and return a
        downloadable GeoParquet file_id for further analysis. """
        tmp = None
        out = None
        try:
            import geopandas as gpd
            from agent_runtime.file_store import create_output_file_from_path

            read_path, tmp = _stage(file_id, sibling_file_ids)
            gdf = read_vector(read_path, layer)
            if getattr(gdf, "crs", None) is None:
                return json.dumps({"ok": False, "error": "source has no CRS (.prj missing); cannot reproject"})
            gdf = gdf.to_crs(target_crs)
            out = Path(tempfile.mkdtemp(prefix="vec_rp_")) / "reprojected.parquet"
            gdf.to_parquet(out)
            rec = create_output_file_from_path(out, filename="reprojected.parquet")
            return json.dumps({
                "ok": True, "file_id": rec["file_id"], "filename": rec.get("filename"),
                "download_url": rec.get("download_url"),
                "feature_count": int(len(gdf)), "crs": _epsg(gdf.crs),
            }, default=str)
        except Exception as exc:
            return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}", "hint": _SIB})
        finally:
            if out and out.parent.exists():
                shutil.rmtree(out.parent, ignore_errors=True)
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)

    def vector_spatial_join(left_file_id: str, right_file_id: str, how: str = "inner",
                            predicate: str = "intersects",
                            left_siblings: Optional[List[str]] = None,
                            right_siblings: Optional[List[str]] = None,
                            name: Optional[str] = None) -> str:
        """Spatial-join two vector datasets (e.g. points-in-TIGER-polygons). how:
        inner|left|right; predicate: intersects|within|contains. Returns a downloadable
        GeoParquet file_id of the joined result. """
        t1 = t2 = None
        out = None
        try:
            import geopandas as gpd
            from agent_runtime.file_store import create_output_file_from_path

            lp, t1 = _stage(left_file_id, left_siblings)
            rp, t2 = _stage(right_file_id, right_siblings)
            left = read_vector(lp)
            right = read_vector(rp)
            note = None
            if getattr(left, "crs", None) is not None and getattr(right, "crs", None) is not None:
                if left.crs != right.crs:
                    right = right.to_crs(left.crs)
            else:
                note = "one or both inputs lack a CRS; join performed without reprojection"
            joined = gpd.sjoin(left, right, how=how, predicate=predicate)
            # GeoJSON unless the result is big enough that parquet earns its keep: a
            # .geojson artifact is readable by every other tool AND auto-loads on the
            # user's map, whereas parquet is a dead end for both.
            as_geojson = len(joined) <= _GEOJSON_MAX_FEATURES
            suffix = "geojson" if as_geojson else "parquet"
            fname = artifact_name(name, suffix, source=str(lp), default="spatial_join")
            out = Path(tempfile.mkdtemp(prefix="vec_sj_")) / fname
            if as_geojson:
                joined.to_crs("EPSG:4326").to_file(out, driver="GeoJSON")
            else:
                joined.to_parquet(out)
            rec = create_output_file_from_path(out, filename=fname)
            res = {"ok": True, "file_id": rec["file_id"], "filename": rec.get("filename"),
                   "download_url": rec.get("download_url"), "format": suffix,
                   "on_map": as_geojson,
                   "feature_count": int(len(joined)), "crs": _epsg(getattr(joined, "crs", None))}
            if note:
                res["note"] = note
            return json.dumps(res, default=str)
        except Exception as exc:
            return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}", "hint": _SIB})
        finally:
            for t in (t1, t2):
                if t:
                    shutil.rmtree(t, ignore_errors=True)
            if out and out.parent.exists():
                shutil.rmtree(out.parent, ignore_errors=True)

    meta = {"category": "geo"}
    def add_map_layer(file_id: str, render: str = "auto", column: Optional[str] = None,
                      name: Optional[str] = None, sibling_file_ids: Optional[List[str]] = None,
                      max_points: Optional[int] = None) -> str:
        """Put a dataset on the user's interactive map as a styled layer.

        `render`: "heatmap" (point density), "choropleth" (polygons shaded by a NUMERIC
        `column`), "categories" (shaded by a CLASS-NAME `column` — LISA classes, Gi* bands,
        region ids; a legend is built for you), "points"/"shapes", or "auto". Returns the layer descriptor
        the client plots, plus a downloadable GeoJSON file_id.
        """
        tmp = None
        try:
            from agent_runtime.file_store import create_output_file_from_path

            redirect = _unmappable_input(file_id)
            if redirect:
                return json.dumps(redirect)
            read_path, tmp = _stage(file_id, sibling_file_ids)
            gdf = read_vector(read_path)
            if getattr(gdf, "crs", None) is not None:
                gdf = gdf.to_crs("EPSG:4326")          # web maps are lon/lat
            geom_types = {str(t) for t in gdf.geometry.geom_type.unique()}
            is_point = geom_types and geom_types <= {"Point", "MultiPoint"}
            mode = (render or "auto").strip().lower()
            legend = None
            # The choropleth/heatmap branches below slice `gdf` down to the columns they need,
            # so hold the full frame: if the layer turns out to be meaningless, the alternatives
            # worth suggesting live in the columns that were dropped.
            full = gdf
            if mode == "auto":
                mode = "heatmap" if (is_point and len(gdf) > 2000) else ("points" if is_point else "shapes")
            if mode == "categories":
                # A class-NAME column (LISA's High-High, a Gi* band, a region id). Without a
                # legend the client has nothing to map names to colours and falls back to the
                # NUMERIC ramp, where Number("High-High") is NaN and every feature lands on one
                # flat fill — the exact bug this branch exists to prevent.
                if not column or column not in gdf.columns:
                    return json.dumps({"ok": False,
                                       "error": "render='categories' needs `column` — the class-name "
                                                "column to colour by",
                                       "columns": [c for c in gdf.columns if c != "geometry"][:40]})
                classes = [str(v) for v in gdf[column].dropna().unique()]
                if len(classes) > len(_CATEGORY_COLORS):
                    return json.dumps({"ok": False,
                                       "error": f"{column!r} has {len(classes)} distinct values; "
                                                f"categories renders at most {len(_CATEGORY_COLORS)}",
                                       "hint": "Use render='choropleth' for a numeric column, or bin "
                                               "the values first."})
                legend = [{"label": c, "color": _CATEGORY_COLORS[i]} for i, c in enumerate(sorted(classes))]
            if mode == "choropleth":
                if not column or column not in gdf.columns:
                    # List the NUMERIC columns: an arbitrary first-N slice hides the very
                    # column being looked for (a join count often lands last of 50+ fields).
                    import pandas as _pd
                    numeric = [c for c in gdf.columns
                               if c != "geometry" and _pd.api.types.is_numeric_dtype(gdf[c])]
                    return json.dumps({"ok": False,
                                       "error": f"choropleth needs a numeric `column` present in the data; "
                                                f"got {column!r}",
                                       "numeric_columns": numeric[:40]})
                gdf = gdf[[column, "geometry"]]
            elif mode == "heatmap":
                # Keep a few attributes so clicking a point says something — but past
                # _HEATMAP_LEAN_ABOVE drop them: a density render needs only position, and
                # shedding them is what makes the FULL population servable instead of a sample.
                keep = [c for c in gdf.columns if c == "geometry" or c == column]
                budget = 0 if len(gdf) > _HEATMAP_LEAN_ABOVE else _HEATMAP_KEEP_COLUMNS
                extra = [c for c in gdf.columns if c not in keep][:budget]
                gdf = gdf[[*keep, *extra]] if extra else gdf[keep]
            # Sample only beyond what is actually servable, and record it structurally so the
            # client can SHOW that a layer is partial — an answer that forgets to mention it
            # used to be the only signal, and the grounding audit passed such answers.
            ceiling = int(max_points) if max_points else _MAP_LAYER_MAX_FEATURES
            total = int(len(gdf))
            note = None
            sampled = total > ceiling and mode in {"heatmap", "points"}
            if sampled:
                note = (f"showing a random {ceiling} of {total} features "
                        f"({ceiling * 100 // total}%) — the layer on the map is a SAMPLE")
                gdf = gdf.sample(ceiling, random_state=0)
            fname = artifact_name(name, "geojson", source=str(read_path), default=f"{mode}_layer")
            out = Path(tempfile.mkdtemp(prefix="map_layer_")) / fname
            gdf.to_file(out, driver="GeoJSON")
            rec = create_output_file_from_path(out, filename=fname)
            # Look at what was actually written before calling it a visual. A choropleth over a
            # constant column, a styling column that did not survive the write, or geometry in
            # metres labelled EPSG:4326 all produce a map that is technically delivered and
            # visually meaningless — and the model cannot see the result to notice.
            try:
                from agent_runtime.layer_qa import inspect_geojson
                qa = inspect_geojson(str(out), render=mode, style_by=column, legend=legend)
            except Exception:
                qa = {"ok": True, "problems": []}
            if not qa.get("ok"):
                import pandas as _pd2
                # Offer columns that would actually WORK: numeric, varying, and not the one
                # that just failed. Listing the failing column back is not an alternative.
                usable = [c for c in full.columns
                          if c != "geometry" and c != column
                          and _pd2.api.types.is_numeric_dtype(full[c])
                          and full[c].nunique(dropna=True) > 1]
                varied = [c for c in full.columns
                          if c != "geometry" and c != column
                          and full[c].nunique(dropna=True) > 1]
                return json.dumps({
                    "ok": False,
                    "error": "the layer would render meaninglessly: "
                             + "; ".join(qa.get("problems") or []),
                    "render": mode, "column": column,
                    "numeric_columns_that_vary": usable[:40],
                    "categorical_columns_that_vary": [c for c in varied if c not in usable][:40],
                    "hint": "fix the column or the render and call add_map_layer again — the "
                            "file was written but is not worth showing as-is",
                })
            label = (name or Path(fname).stem).replace("_", " ")
            res = {"ok": True, "file_id": rec["file_id"], "filename": rec.get("filename"),
                   "download_url": rec.get("download_url"), "feature_count": int(len(gdf)),
                   "on_map": True,
                   # Consumed by agent_runtime.map_layers.build_map_layer -> `map_layer` SSE event.
                   "sampled": bool(sampled), "features_total": total,
                   "size_bytes": rec.get("size_bytes"),
                   "map_layer": {"url": rec.get("download_url"), "label": label, "render": mode,
                                 "style_by": column, "source": "analysis",
                                 "count": int(len(gdf)),
                                 # Only categorical layers carry one; the client keys its
                                 # 'categories' render off its presence.
                                 **({"legend": legend} if legend else {}),
                                 # Carried to the UI so a partial layer says so on screen.
                                 "sampled": bool(sampled), "total": total}}
            if note:
                res["note"] = note
            if qa.get("notes"):
                res["visual_notes"] = qa["notes"]
            return json.dumps(res, default=str)
        except Exception as exc:
            return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}", "hint": _SIB})
        finally:
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)

    return [
        StructuredTool.from_function(func=add_map_layer, name="add_map_layer", metadata=meta,
            description=("Put a dataset on the user's INTERACTIVE MAP as a styled layer they can pan, "
                         "zoom and click — this is how a map is delivered here, and it is what to use "
                         "when the user asks to see/plot/visualize data on the map. `render`: "
                         "'heatmap' for point density, 'choropleth' for polygons shaded by a numeric "
                         "`column` (e.g. the count from vector_spatial_join), 'points'/'shapes', or "
                         "'auto'. Accepts GeoJSON/shapefile/GeoPackage/GeoParquet/CSV-with-coordinates "
                         "by file_id; reprojects to WGS84 and samples very large point sets for "
                         "display. A PNG tool (render_map_image) is only for a static image someone wants "
                         "to download.")),
        StructuredTool.from_function(func=inspect_vector, name="inspect_vector", metadata=meta,
            description=("Read a vector / shapefile's metadata (CRS, extent, geometry type, feature "
                         "count, attribute columns) without loading all geometry. Handles a TIGER/Line "
                         "shapefile .zip, a .shp (+ sidecars), GeoJSON, GeoPackage, or GeoParquet by "
                         "file_id. " + _SIB)),
        StructuredTool.from_function(func=render_map_image, name="render_map_image", metadata=meta,
            description=("Draw a vector dataset into a STATIC PNG PICTURE (optionally shaded by "
                         "`column`) and return a downloadable file_id. The picture cannot be "
                         "panned, zoomed or clicked, so choose it when someone wants an IMAGE to "
                         "download, embed in a document or print. To show data on the user's "
                         "interactive map instead, use add_map_layer. Large layers auto-downsample. "
                         + _SIB)),
        StructuredTool.from_function(func=vector_to_geojson, name="vector_to_geojson", metadata=meta,
            description=("Convert a vector dataset to GeoJSON (reprojected to WGS84 by default) and "
                         "return a downloadable file_id, e.g. for web mapping. " + _SIB)),
        StructuredTool.from_function(func=reproject_vector, name="reproject_vector", metadata=meta,
            description=("Reproject a vector dataset to a target CRS (e.g. EPSG:5070 for equal-area "
                         "US analysis) and return a downloadable GeoParquet file_id. " + _SIB)),
        StructuredTool.from_function(func=vector_spatial_join, name="vector_spatial_join", metadata=meta,
            description=("Spatial-join two vector datasets (e.g. assign points to the TIGER polygon "
                         "they fall in). CRS is aligned automatically. Returns a downloadable "
                         "GeoParquet file_id.")),
    ]


__all__ = ["make_langchain_geo_tools"]
