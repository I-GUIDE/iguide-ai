"""Classic vector OVERLAY and GEOMETRY operations — the "Geoprocessing" menu every GIS ships.

Clip, dissolve, intersect, erase, buffer, simplify and centroid/hull/bbox summaries are the
operations an analyst reaches for constantly. Without them the agent has to WRITE CODE for
"keep only the tracts inside the city", which is slow, silently wrong about CRS (a degree
buffer is not a distance), and produces nothing the interactive map can plot.

Every tool here:

* accepts uploaded data by ``file_id`` — GeoJSON / shapefile (.zip or .shp + sidecars) /
  GeoPackage / GeoParquet / CSV-with-coordinates — via :func:`read_vector`;
* does all metric work (buffer distances, areas, lengths, simplify tolerance) in a projected,
  metre-based CRS chosen with ``estimate_utm_crs()`` — never in degrees;
* writes a WGS84 GeoJSON through ``create_output_file_from_path`` and returns a ``map_layer``
  descriptor so the result lands on the user's interactive map (see
  ``agent_runtime.map_layers.build_map_layer``);
* returns a JSON **string** and NEVER raises: failures come back as
  ``{"ok": false, "error": "...", "hint": ...}``, and a missing column error lists the
  candidate columns (including the NUMERIC ones) so the next call can succeed.

Heavy imports (geopandas / shapely / matplotlib) stay inside the tool bodies so importing
this module never breaks the agent boot.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent_runtime.langchain_geo_tools import (
    _MAP_LAYER_MAX_FEATURES,
    _epsg,
    _index_attached,
    _resolve,
    _stage_vector_source,
    artifact_name,
    read_vector,
)

# Metres per unit. Degrees are deliberately absent — see _distance_meters.
_UNITS_M: Dict[str, float] = {
    "m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0,
    "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0, "kilometre": 1000.0,
    "kilometres": 1000.0,
    "mi": 1609.344, "mile": 1609.344, "miles": 1609.344,
    "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
    "yd": 0.9144, "yard": 0.9144, "yards": 0.9144,
    "nmi": 1852.0, "nm": 1852.0,
}
_DEGREE_UNITS = {"deg", "degs", "degree", "degrees", "dd", "arcdeg"}

# Aggregations dissolve_layer accepts for the numeric columns it rolls up.
_STATISTICS = ("sum", "mean", "median", "min", "max", "count", "std", "var", "first", "last")
# Overlay flavours intersect_layers exposes to a power user (default: intersection).
_OVERLAY_HOWS = ("intersection", "union", "identity", "symmetric_difference", "difference")

# Equal-area metric fallback when estimate_utm_crs() cannot decide (empty / global extent).
_FALLBACK_METRIC_CRS = "EPSG:6933"
# Metre-based but NOT distance-true: Web Mercator's scale factor is 1/cos(latitude), so a
# "10 000 m" buffer drawn in EPSG:3857 at 40°N covers only ~7.7 km on the ground. These are
# re-projected to a local UTM zone before anything is measured.
_DISTORTED_METRE_EPSG = {3857, 3785, 3395, 900913, 102100, 102113, 54004}

_SIB = ("For an uploaded shapefile, pass the .shp's file_id (or any single component) — the "
        "sidecars (.shx/.dbf/.prj) are auto-discovered among the attached files. You may also "
        "pass them as sibling_file_ids, or upload the shapefile as a single .zip.")


# --- reading / naming -------------------------------------------------------------

def _origin_name(ref: str) -> Optional[str]:
    """The uploaded file's ORIGINAL filename, used to name derived artifacts.

    On disk an upload is ``uploads/<file_id>__<name>``, so the raw path stem would leak the
    file_id into every download name (``file_ab12cd__tracts_buffer.geojson``).
    """
    try:
        path, record = _resolve(ref)
    except Exception:  # noqa: BLE001 - naming must never fail a tool
        return None
    if record and record.get("filename"):
        return str(record["filename"])
    name = path.name
    return name.split("__", 1)[1] if "__" in name else name


def _default_stem(ref: str, op: str) -> str:
    stem = Path(_origin_name(ref) or "").stem
    return f"{stem}_{op}" if stem else op


# --- CRS helpers -----------------------------------------------------------------

def _is_metre_based(crs: Any) -> bool:
    try:
        units = [(ax.unit_name or "").lower() for ax in crs.axis_info[:2]]
    except Exception:  # noqa: BLE001
        return False
    return bool(units) and all(u in {"metre", "meter", "m"} for u in units)


def _as_metric(gdf: Any) -> Tuple[Any, Optional[str], Optional[str]]:
    """``(projected_gdf, metric_crs, note)`` — a metre-based CRS for measurement.

    Buffering / measuring in EPSG:4326 treats a degree as a unit of length, which is wrong
    everywhere and catastrophically wrong away from the equator, so metric work always
    happens here first.
    """
    note = None
    if getattr(gdf, "crs", None) is None:
        gdf = gdf.set_crs("EPSG:4326")
        note = "input had no CRS; assumed EPSG:4326 (lon/lat)"
    crs = gdf.crs
    try:
        code = crs.to_epsg()
    except Exception:  # noqa: BLE001
        code = None
    if not crs.is_geographic and _is_metre_based(crs):
        if code not in _DISTORTED_METRE_EPSG:
            return gdf, _epsg(crs), note
        note = "; ".join(x for x in [
            note, f"EPSG:{code} measures in metres but distorts distance by 1/cos(latitude); "
                  "measured in a local UTM zone instead"] if x)
    target: Any = None
    try:
        target = gdf.estimate_utm_crs()
    except Exception:  # noqa: BLE001 - empty or worldwide extents
        target = None
    if target is None:
        target = _FALLBACK_METRIC_CRS
        note = "; ".join(x for x in [note, f"UTM zone undetermined; used {_FALLBACK_METRIC_CRS}"] if x)
    return gdf.to_crs(target), _epsg(target), note


def _as_wgs84(gdf: Any) -> Any:
    """Web maps are lon/lat, so every returned layer is EPSG:4326."""
    if getattr(gdf, "crs", None) is None:
        return gdf.set_crs("EPSG:4326")
    return gdf.to_crs("EPSG:4326") if _epsg(gdf.crs) != "EPSG:4326" else gdf


def _align(left: Any, right: Any) -> Tuple[Any, Any, Optional[str]]:
    """Put ``right`` in ``left``'s CRS so an overlay compares the same coordinates."""
    note = None
    if getattr(left, "crs", None) is None:
        left = left.set_crs("EPSG:4326")
        note = "left input had no CRS; assumed EPSG:4326"
    if getattr(right, "crs", None) is None:
        right = right.set_crs("EPSG:4326")
        note = "; ".join(x for x in [note, "right input had no CRS; assumed EPSG:4326"] if x)
    if left.crs != right.crs:
        right = right.to_crs(left.crs)
    return left, right, note


def _distance_meters(distance: float, units: str) -> float:
    u = str(units or "m").strip().lower()
    if u in _DEGREE_UNITS:
        raise ValueError(
            "'degrees' is not a distance: 1 degree of longitude is 111 km at the equator and "
            "0 km at the pole. Pass units of km/m/mi/ft — the buffer is built in a projected "
            "UTM CRS and returned as lon/lat.")
    if u not in _UNITS_M:
        raise ValueError(f"unknown distance units {units!r}; use one of {sorted(set(_UNITS_M))}")
    try:
        d = float(distance)
    except (TypeError, ValueError):
        raise ValueError(f"distance must be a number, got {distance!r}") from None
    if not d > 0:
        raise ValueError(f"distance must be greater than 0, got {d}")
    return d * _UNITS_M[u]


# --- attribute helpers -----------------------------------------------------------

def _numeric_columns(gdf: Any) -> List[str]:
    import pandas as pd

    geom = getattr(gdf, "geometry", None)
    geom_name = getattr(geom, "name", "geometry")
    return [str(c) for c in gdf.columns
            if c != geom_name and pd.api.types.is_numeric_dtype(gdf[c])]


def _other_columns(gdf: Any) -> List[str]:
    numeric = set(_numeric_columns(gdf))
    geom_name = getattr(getattr(gdf, "geometry", None), "name", "geometry")
    return [str(c) for c in gdf.columns if c != geom_name and str(c) not in numeric]


def _column_error(gdf: Any, column: Any, role: str) -> str:
    """A missing-column failure that CARRIES the answer: every candidate, numerics flagged.

    An arbitrary first-N slice of the schema hid the very column being looked for (a join
    count lands last among 50+ TIGER fields), so list them all and separate the numerics.
    """
    numeric = _numeric_columns(gdf)
    return json.dumps({
        "ok": False,
        "error": f"KeyError: {role} column {column!r} is not in the layer",
        "candidates": [str(c) for c in gdf.columns
                       if c != getattr(getattr(gdf, "geometry", None), "name", "geometry")],
        "numeric_columns": numeric,
        "hint": (f"re-run with {role}=<one of candidates>. Numeric columns usable for "
                 f"statistics / choropleth shading: {numeric or 'none'}"),
    }, default=str)


_GENERIC_HINT = ("check the inputs with inspect_vector first — the usual causes are a wrong "
                 "file_id, a layer with no geometry, or two layers covering different areas. "
                 + _SIB)


def _fail(exc: BaseException, hint: str = _GENERIC_HINT) -> str:
    return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}", "hint": hint})


def _repair(gdf: Any) -> Any:
    """Best-effort fix for self-intersecting rings, which make overlays throw GEOS errors."""
    try:
        invalid = ~gdf.geometry.is_valid
        if bool(invalid.any()):
            fixed = gdf.copy()
            fixed.geometry = gdf.geometry.make_valid()
            return fixed
    except Exception:  # noqa: BLE001 - never block the operation on a repair attempt
        pass
    return gdf


def _clean(gdf: Any) -> Any:
    """Drop null / empty geometry rows an overlay leaves behind."""
    import warnings

    try:
        with warnings.catch_warnings():   # notna() warns loudly when empties are present
            warnings.filterwarnings("ignore", "GeoSeries.notna", UserWarning)
            keep = ~gdf.geometry.is_empty & gdf.geometry.notna()
        return gdf[keep]
    except Exception:  # noqa: BLE001
        return gdf


def _geom_kind(gdf: Any) -> str:
    """"points" | "lines" | "shapes" for the layer's dominant geometry."""
    try:
        types = {str(t) for t in gdf.geometry.geom_type.dropna().unique()}
    except Exception:  # noqa: BLE001
        return "shapes"
    if types and types <= {"Point", "MultiPoint"}:
        return "points"
    if types and types <= {"LineString", "MultiLineString", "LinearRing"}:
        return "lines"
    return "shapes"


def _vertex_count(gdf: Any) -> Optional[int]:
    try:
        import numpy as np
        import shapely

        return int(shapely.get_num_coordinates(np.asarray(gdf.geometry.values)).sum())
    except Exception:  # noqa: BLE001
        return None


def _add_metrics(gdf: Any) -> Tuple[Any, Optional[str], Optional[str]]:
    """Attach ``area_km2`` / ``length_km`` measured in a projected CRS (never degrees)."""
    metric, mcrs, note = _as_metric(gdf)
    out = gdf.copy()
    kind = _geom_kind(gdf)
    try:
        if kind == "shapes":
            out["area_km2"] = (metric.geometry.area / 1_000_000.0).round(6).values
            out["perimeter_km"] = (metric.geometry.length / 1000.0).round(6).values
        elif kind == "lines":
            out["length_km"] = (metric.geometry.length / 1000.0).round(6).values
    except Exception:  # noqa: BLE001 - metrics are a bonus, not the payload
        return gdf, mcrs, note
    return out, mcrs, note


# --- output ----------------------------------------------------------------------

def _write_table(df: Any, *, name: Optional[str], default: str, source: Optional[str]) -> Dict[str, Any]:
    """Persist the attribute table behind a layer as a downloadable .csv."""
    from agent_runtime.file_store import create_output_file_from_path

    # `name or default` as the STEM, not as artifact_name's fallback: passing only `source`
    # would name every result after its input ("tracts.csv" for both the dissolve and the
    # buffer of tracts.geojson), which is exactly the indistinguishable-downloads problem.
    fname = artifact_name(name or default, "csv", source=source, default=default)
    tmpdir = Path(tempfile.mkdtemp(prefix="ovl_csv_"))
    try:
        out = tmpdir / fname
        geom_name = getattr(getattr(df, "geometry", None), "name", "geometry")
        plain = df.drop(columns=[c for c in [geom_name] if c in df.columns])
        plain.to_csv(out, index=False)
        rec = create_output_file_from_path(out, filename=fname)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return {"file_id": rec["file_id"], "filename": rec.get("filename"),
            "download_url": rec.get("download_url"), "row_count": int(len(df))}


def _finish(gdf: Any, *, name: Optional[str], default: str, render: str = "auto",
            style_by: Optional[str] = None, label: Optional[str] = None,
            source: Optional[str] = None, extra: Optional[Dict[str, Any]] = None,
            table: bool = False, notes: Optional[List[Optional[str]]] = None) -> str:
    """Write ``gdf`` as WGS84 GeoJSON, register it, and describe it as a map layer."""
    from agent_runtime.file_store import create_output_file_from_path

    gdf = _as_wgs84(gdf)
    mode = (render or "auto").strip().lower()
    if mode == "auto":
        mode = _geom_kind(gdf)
        if mode == "lines":
            mode = "shapes"
    note_list = [n for n in (notes or []) if n]

    # A choropleth is only a choropleth if the shading column really is numeric and really
    # is in the written file — otherwise the client draws an unstyled blob.
    numeric = set(_numeric_columns(gdf))
    if style_by is not None and str(style_by) not in numeric:
        note_list.append(f"style_by {style_by!r} is not a numeric column of the result; layer left unstyled")
        style_by = None
    if mode == "choropleth" and style_by is None:
        mode = "shapes"
        note_list.append("no numeric column to shade by; rendered as plain shapes")
    if mode not in {"heatmap", "choropleth", "points", "shapes"}:
        note_list.append(f"unknown render {mode!r}; used 'shapes'")
        mode = "shapes"

    # `default` already carries the OPERATION ("tracts_clipped"), so it is the stem to use
    # when the model supplied no name — see _write_table for why source alone is not enough.
    fname = artifact_name(name or default, "geojson", source=source, default=default)
    tmpdir = Path(tempfile.mkdtemp(prefix="ovl_gj_"))
    try:
        out = tmpdir / fname
        if len(gdf) == 0:
            # to_file() on an empty frame is unreliable across drivers; an empty
            # FeatureCollection still gives the user a downloadable, valid answer.
            out.write_text(json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")
        else:
            gdf.to_file(out, driver="GeoJSON")
        rec = create_output_file_from_path(out, filename=fname)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    count = int(len(gdf))
    payload: Dict[str, Any] = {
        "ok": True,
        "file_id": rec["file_id"],
        "filename": rec.get("filename"),
        "download_url": rec.get("download_url"),
        "feature_count": count,
        "crs": _epsg(getattr(gdf, "crs", None)),
        "columns": [str(c) for c in gdf.columns
                    if c != getattr(getattr(gdf, "geometry", None), "name", "geometry")],
    }
    if extra:
        payload.update(extra)
    if table and count:
        try:
            payload["table"] = _write_table(gdf, name=(f"{name}_table" if name else None),
                                            default=f"{default}_table", source=source)
        except Exception as exc:  # noqa: BLE001 - the layer is the deliverable
            note_list.append(f"attribute csv not written: {type(exc).__name__}: {exc}")
    if count:
        if count > _MAP_LAYER_MAX_FEATURES:
            note_list.append(f"{count} features is a very large layer to draw; consider "
                             "dissolving or filtering before mapping")
        payload["on_map"] = True
        payload["map_layer"] = {
            "url": rec.get("download_url"),
            "label": label or (name or Path(fname).stem).replace("_", " "),
            "render": mode,
            "style_by": style_by,
            "source": "analysis",
            "count": count,
        }
    else:
        payload["on_map"] = False
        note_list.append("the operation produced 0 features — the inputs may not overlap, or "
                         "they may be in different areas; check both extents with inspect_vector")
    if note_list:
        payload["note"] = "; ".join(note_list)
    return json.dumps(payload, default=str)


def make_overlay_tools(default_input_file_ids: Optional[List[str]] = None) -> List[Any]:
    """Build the vector overlay / geometry StructuredTools (geopandas-backed).

    ``default_input_file_ids`` is the conversation's attached file set; it lets an uploaded
    shapefile be referenced by its ``.shp`` alone, with the sidecars auto-discovered.
    """
    from langchain_core.tools import StructuredTool

    _attached = _index_attached(default_input_file_ids)

    def _load(ref: str, siblings: Optional[List[str]] = None,
              layer: Optional[str] = None) -> Tuple[Any, str, Optional[str]]:
        """``(gdf, read_path, tempdir_to_cleanup)`` for any supported vector reference."""
        read_path, tmp = _stage_vector_source(ref, siblings, _attached)
        return read_vector(read_path, layer), read_path, tmp

    def _rm(*dirs: Optional[str]) -> None:
        for d in dirs:
            if d:
                shutil.rmtree(d, ignore_errors=True)

    # ---------------------------------------------------------------- clip
    def clip_layer(target_file_id: str, clip_file_id: str, keep_geom_type: bool = True,
                   name: Optional[str] = None, target_siblings: Optional[List[str]] = None,
                   clip_siblings: Optional[List[str]] = None) -> str:
        """Cookie-cut the target layer with the clip layer: keep only the parts of the
        target that fall INSIDE the clip boundary, trimming geometry at the edge. """
        t1 = t2 = None
        try:
            import geopandas as gpd

            target, tpath, t1 = _load(target_file_id, target_siblings)
            mask, _, t2 = _load(clip_file_id, clip_siblings)
            target, mask, crs_note = _align(_repair(target), _repair(mask))
            before = int(len(target))
            clipped = _clean(gpd.clip(target, mask, keep_geom_type=bool(keep_geom_type)))
            # Recompute size for the TRIMMED geometry (a clipped tract is smaller than the
            # tract), which is what makes "how much of each fell inside" answerable.
            clipped, mcrs, metric_note = _add_metrics(clipped)
            shade = next((c for c in ("area_km2", "length_km") if c in clipped.columns), None)
            return _finish(
                clipped, name=name, default=_default_stem(target_file_id, "clipped"),
                source=tpath, style_by=shade, table=True,
                label=(name or "clipped").replace("_", " "),
                extra={"operation": "clip", "input_features": before,
                       "features_dropped": before - int(len(clipped)),
                       "measure_crs": mcrs},
                notes=[crs_note, metric_note],
            )
        except Exception as exc:  # noqa: BLE001
            return _fail(exc)
        finally:
            _rm(t1, t2)

    # ------------------------------------------------------------- dissolve
    def dissolve_layer(file_id: str, by: Optional[str] = None, statistic: str = "sum",
                       style_by: Optional[str] = None, render: str = "auto",
                       name: Optional[str] = None, sibling_file_ids: Optional[List[str]] = None,
                       layer: Optional[str] = None) -> str:
        """Merge features that share a value of `by` into one feature each (all features into
        one when `by` is omitted), aggregating numeric columns with `statistic`. """
        tmp = None
        try:
            stat = str(statistic or "sum").strip().lower()
            if stat not in _STATISTICS:
                return json.dumps({"ok": False,
                                   "error": f"ValueError: unknown statistic {statistic!r}",
                                   "candidates": list(_STATISTICS),
                                   "hint": "pass statistic='sum' to total values, 'mean' to average them"})
            gdf, rpath, tmp = _load(file_id, sibling_file_ids, layer)
            if by is not None and str(by) not in [str(c) for c in gdf.columns]:
                return _column_error(gdf, by, "by")
            gdf = _clean(_repair(gdf))
            geom_name = getattr(gdf.geometry, "name", "geometry")
            group_col = str(by) if by is not None else None

            work = gdf.copy()
            work["feature_count"] = 1
            numeric = [c for c in _numeric_columns(work)
                       if c != group_col and c != "feature_count"]
            others = [c for c in _other_columns(work) if c != group_col]
            agg: Dict[str, str] = {c: stat for c in numeric}
            agg.update({c: "first" for c in others})   # keep names/labels readable
            agg["feature_count"] = "sum"               # how many rows each output merges

            key = group_col
            if key is None:                            # dissolve everything into one feature
                key = "__all__"
                work[key] = "all"
            dissolved = work.dissolve(by=key, aggfunc=agg, as_index=False)
            if group_col is None and key in dissolved.columns:
                dissolved = dissolved.drop(columns=[key])
            dissolved = dissolved.set_geometry(geom_name) if geom_name in dissolved.columns else dissolved

            if style_by is not None and str(style_by) not in [str(c) for c in dissolved.columns]:
                return _column_error(dissolved, style_by, "style_by")
            shade = str(style_by) if style_by else None
            if shade is None:
                candidates = [c for c in _numeric_columns(dissolved) if c != "feature_count"]
                shade = candidates[0] if candidates else "feature_count"
            mode = (render or "auto").strip().lower()
            if mode == "auto":
                # The point of a dissolve is to READ the aggregate, so shade by it when the
                # result is polygonal; points/lines stay unstyled marks.
                mode = "choropleth" if _geom_kind(dissolved) == "shapes" else "auto"
            return _finish(
                dissolved, name=name, default=_default_stem(file_id, "dissolved"),
                source=rpath, render=mode, style_by=shade,
                label=(name or (f"dissolved by {group_col}" if group_col else "dissolved")),
                table=True,
                extra={"operation": "dissolve", "by": group_col, "statistic": stat,
                       "input_features": int(len(gdf)),
                       "aggregated_columns": numeric, "carried_columns": others},
            )
        except Exception as exc:  # noqa: BLE001
            return _fail(exc)
        finally:
            _rm(tmp)

    # ------------------------------------------------------------ intersect
    def intersect_layers(left_file_id: str, right_file_id: str, how: str = "intersection",
                         keep_geom_type: bool = True, name: Optional[str] = None,
                         left_siblings: Optional[List[str]] = None,
                         right_siblings: Optional[List[str]] = None) -> str:
        """Geometric intersection of two layers: the overlapping pieces only, each carrying the
        attributes of BOTH inputs plus the overlap area. """
        t1 = t2 = None
        try:
            import geopandas as gpd

            mode = str(how or "intersection").strip().lower()
            if mode not in _OVERLAY_HOWS:
                return json.dumps({"ok": False, "error": f"ValueError: unknown how {how!r}",
                                   "candidates": list(_OVERLAY_HOWS),
                                   "hint": "how='intersection' keeps only the overlap"})
            left, lpath, t1 = _load(left_file_id, left_siblings)
            right, _, t2 = _load(right_file_id, right_siblings)
            left, right, crs_note = _align(_repair(left), _repair(right))
            try:
                result = gpd.overlay(left, right, how=mode, keep_geom_type=bool(keep_geom_type))
                method = f"overlay({mode})"
            except Exception as overlay_exc:  # noqa: BLE001
                if mode != "intersection":
                    raise
                # Mixed geometry dimensions (points vs polygons) can defeat overlay(); pair the
                # features with a spatial join and intersect them one by one instead.
                pairs = gpd.sjoin(left, right, how="inner", predicate="intersects")
                idx = pairs["index_right"].values
                geoms = gpd.GeoSeries(
                    [a.intersection(b) for a, b in
                     zip(pairs.geometry.values, right.geometry.iloc[idx].values)],
                    index=pairs.index, crs=left.crs)
                result = pairs.drop(columns=["index_right"]).set_geometry(geoms)
                method = f"sjoin+intersection (overlay unavailable: {type(overlay_exc).__name__})"
            result = _clean(result)
            result, mcrs, metric_note = _add_metrics(result)
            shade = "area_km2" if "area_km2" in result.columns else (
                "length_km" if "length_km" in result.columns else None)
            left_stem = Path(_origin_name(left_file_id) or "left").stem
            right_stem = Path(_origin_name(right_file_id) or "right").stem
            # Name the download after the operation actually performed, not always "x".
            joiner = "x" if mode == "intersection" else mode
            return _finish(
                result, name=name, default=f"{left_stem}_{joiner}_{right_stem}",
                source=lpath, style_by=shade, table=True,
                label=(name or mode).replace("_", " "),
                extra={"operation": mode, "left_features": int(len(left)),
                       "right_features": int(len(right)), "method": method,
                       "measure_crs": mcrs},
                notes=[crs_note, metric_note],
            )
        except Exception as exc:  # noqa: BLE001
            return _fail(exc)
        finally:
            _rm(t1, t2)

    # ---------------------------------------------------------------- erase
    def erase_layer(target_file_id: str, erase_file_id: str, name: Optional[str] = None,
                    target_siblings: Optional[List[str]] = None,
                    erase_siblings: Optional[List[str]] = None) -> str:
        """Punch the erase layer out of the target layer (target MINUS erase): whatever the
        erase shapes cover is removed, the rest of the target is kept with its attributes. """
        t1 = t2 = None
        try:
            target, tpath, t1 = _load(target_file_id, target_siblings)
            eraser, _, t2 = _load(erase_file_id, erase_siblings)
            target, eraser, crs_note = _align(_repair(target), _repair(eraser))
            before = int(len(target))
            # One combined mask, then a per-feature difference: this keeps the target's own
            # attributes untouched and works for points/lines/polygons alike, where
            # overlay(how="difference") insists on comparable geometry dimensions.
            mask = eraser.geometry.union_all()
            out = target.copy()
            out = out.set_geometry(target.geometry.difference(mask))
            out = _clean(out)
            out, mcrs, metric_note = _add_metrics(out)   # size of what SURVIVED the erase
            shade = next((c for c in ("area_km2", "length_km") if c in out.columns), None)
            return _finish(
                out, name=name, default=_default_stem(target_file_id, "erased"),
                source=tpath, style_by=shade, table=True,
                label=(name or "erased").replace("_", " "),
                extra={"operation": "erase", "input_features": before,
                       "features_dropped": before - int(len(out)),
                       "measure_crs": mcrs},
                notes=[crs_note, metric_note],
            )
        except Exception as exc:  # noqa: BLE001
            return _fail(exc)
        finally:
            _rm(t1, t2)

    # --------------------------------------------------------------- buffer
    def buffer_layer(file_id: str, distance: float = 1.0, units: str = "km",
                     dissolve: bool = False, name: Optional[str] = None,
                     sibling_file_ids: Optional[List[str]] = None,
                     layer: Optional[str] = None) -> str:
        """Grow every feature outward by a TRUE ground distance (default 1 km), producing
        zone polygons. Set dissolve=True to merge overlapping zones into one shape. """
        tmp = None
        try:
            meters = _distance_meters(distance, units)
            gdf, rpath, tmp = _load(file_id, sibling_file_ids, layer)
            gdf = _clean(_repair(gdf))
            metric, mcrs, crs_note = _as_metric(gdf)   # metres, never degrees
            buffered = metric.copy()
            buffered = buffered.set_geometry(metric.geometry.buffer(meters))
            if dissolve:
                import geopandas as gpd

                merged = buffered.geometry.union_all()
                buffered = gpd.GeoDataFrame({"buffer_m": [meters]},
                                            geometry=[merged], crs=buffered.crs)
                try:  # a dissolved buffer is usually explored as parts, not one blob
                    buffered = buffered.explode(index_parts=False).reset_index(drop=True)
                except Exception:  # noqa: BLE001
                    pass
            buffered["buffer_km"] = round(meters / 1000.0, 6)
            buffered["area_km2"] = (buffered.geometry.area / 1_000_000.0).round(6).values
            total_area = float(buffered.geometry.area.sum() / 1_000_000.0)
            return _finish(
                buffered, name=name,
                default=_default_stem(file_id, f"buffer_{int(meters)}m"),
                source=rpath, style_by="area_km2", table=True,
                label=(name or f"{distance:g} {units} buffer"),
                extra={"operation": "buffer", "distance": float(distance), "units": str(units),
                       "distance_m": meters, "buffer_crs": mcrs, "dissolved": bool(dissolve),
                       "total_area_km2": round(total_area, 6),
                       "input_features": int(len(gdf))},
                notes=[crs_note],
            )
        except Exception as exc:  # noqa: BLE001
            return _fail(exc, hint="distance units must be metric/imperial length (km, m, mi, "
                                   "ft) — never degrees. " + _SIB)
        finally:
            _rm(tmp)

    # ------------------------------------------------------------- simplify
    def simplify_layer(file_id: str, tolerance_m: float = 50.0, keep_topology: bool = True,
                       name: Optional[str] = None, sibling_file_ids: Optional[List[str]] = None,
                       layer: Optional[str] = None) -> str:
        """Generalize geometry by removing vertices finer than `tolerance_m` METRES (default 50),
        making a heavy boundary layer far lighter to draw while it still looks the same. """
        tmp = None
        try:
            tol = float(tolerance_m)
            if not tol > 0:
                raise ValueError(f"tolerance_m must be greater than 0 metres, got {tol}")
            gdf, rpath, tmp = _load(file_id, sibling_file_ids, layer)
            gdf = _clean(_repair(gdf))
            metric, mcrs, crs_note = _as_metric(gdf)   # tolerance is METRES on the ground
            before = _vertex_count(metric)
            method = "douglas_peucker"
            simplified = metric.copy()
            done = False
            if keep_topology and _geom_kind(metric) == "shapes" and hasattr(metric, "simplify_coverage"):
                try:
                    # Coverage simplification moves SHARED edges identically, so adjacent
                    # polygons cannot develop slivers or gaps between them.
                    simplified = simplified.set_geometry(metric.simplify_coverage(tol))
                    method, done = "simplify_coverage (shared edges preserved)", True
                except Exception:  # noqa: BLE001
                    done = False
            if not done:
                simplified = simplified.set_geometry(
                    metric.geometry.simplify(tol, preserve_topology=bool(keep_topology)))
                method = "douglas_peucker" + (" (preserve_topology)" if keep_topology else "")
            simplified = _clean(_repair(simplified))
            after = _vertex_count(simplified)
            reduction = (round(100.0 * (before - after) / before, 2)
                         if before and after is not None else None)
            return _finish(
                simplified, name=name, default=_default_stem(file_id, "simplified"),
                source=rpath, label=(name or "simplified").replace("_", " "),
                extra={"operation": "simplify", "tolerance_m": tol, "method": method,
                       "keep_topology": bool(keep_topology), "simplify_crs": mcrs,
                       "vertices_before": before, "vertices_after": after,
                       "vertex_reduction_pct": reduction},
                notes=[crs_note],
            )
        except Exception as exc:  # noqa: BLE001
            return _fail(exc)
        finally:
            _rm(tmp)

    # ------------------------------------------------------- geometry summary
    def geometry_summary(file_id: str, output: str = "centroids", per_feature: bool = True,
                         name: Optional[str] = None, sibling_file_ids: Optional[List[str]] = None,
                         layer: Optional[str] = None) -> str:
        """Derive simple geometry from a layer — centroids (default), convex hull, or bounding
        box — plus measured area/length, per feature or for the whole layer at once. """
        tmp = None
        try:
            import geopandas as gpd

            kinds = {"centroids": "centroids", "centroid": "centroids",
                     "convex_hull": "convex_hull", "hull": "convex_hull",
                     "bbox": "bbox", "envelope": "bbox", "bounding_box": "bbox"}
            kind = kinds.get(str(output or "centroids").strip().lower())
            if kind is None:
                return json.dumps({"ok": False, "error": f"ValueError: unknown output {output!r}",
                                   "candidates": ["centroids", "convex_hull", "bbox"],
                                   "hint": "output='centroids' for one point per feature, "
                                           "'convex_hull' or 'bbox' for an extent shape"})
            gdf, rpath, tmp = _load(file_id, sibling_file_ids, layer)
            gdf = _clean(_repair(gdf))
            metric, mcrs, crs_note = _as_metric(gdf)   # centroids/areas measured, not guessed
            bounds4326 = [float(x) for x in _as_wgs84(gdf).total_bounds]
            source_kind = _geom_kind(gdf)
            stats = {
                "input_features": int(len(gdf)),
                "geometry_types": sorted({str(t) for t in gdf.geometry.geom_type.dropna().unique()}),
                "measure_crs": mcrs,
                "bounds_wgs84": bounds4326,
                "total_area_km2": round(float(metric.geometry.area.sum() / 1_000_000.0), 6),
                "total_length_km": round(float(metric.geometry.length.sum() / 1000.0), 6),
            }

            # A bounding box is expected to be axis-aligned in the coordinates it is SHOWN in,
            # so build it in lon/lat (a UTM envelope back-projected comes out visibly skewed
            # and disagrees with the bounds_wgs84 reported alongside it). Centroids and hulls
            # stay metric, where they are geometrically meaningful.
            work = _as_wgs84(gdf) if kind == "bbox" else metric

            if per_feature:
                if kind == "centroids":
                    # A centroid loses the polygon's size, so carry it along: the point can then
                    # be drawn proportionally instead of as an undifferentiated dot.
                    if source_kind == "shapes":
                        work = work.assign(
                            area_km2=(metric.geometry.area / 1_000_000.0).round(6).values)
                    elif source_kind == "lines":
                        work = work.assign(
                            length_km=(metric.geometry.length / 1000.0).round(6).values)
                    geom = work.geometry.centroid
                elif kind == "convex_hull":
                    geom = work.geometry.convex_hull
                else:
                    geom = work.geometry.envelope
                result = work.copy().set_geometry(geom)
            else:
                merged = work.geometry.union_all()
                if kind == "centroids":
                    geom = merged.centroid
                elif kind == "convex_hull":
                    geom = merged.convex_hull
                else:
                    geom = merged.envelope
                result = gpd.GeoDataFrame({"features_summarized": [int(len(work))]},
                                          geometry=[geom], crs=work.crs)
            if kind != "centroids":
                result, _, _ = _add_metrics(_as_wgs84(result))
            else:
                pts = _as_wgs84(result)
                result = pts.copy()
                result["centroid_lon"] = pts.geometry.x.round(6).values
                result["centroid_lat"] = pts.geometry.y.round(6).values
            shade = next((c for c in ("area_km2", "length_km") if c in result.columns), None)
            return _finish(
                result, name=name, default=_default_stem(file_id, kind),
                source=rpath, render=("points" if kind == "centroids" else "shapes"),
                style_by=shade, table=True,
                label=(name or kind.replace("_", " ")),
                extra={"operation": kind, "per_feature": bool(per_feature),
                       "source_geometry": source_kind, "summary": stats},
                notes=[crs_note],
            )
        except Exception as exc:  # noqa: BLE001
            return _fail(exc)
        finally:
            _rm(tmp)

    meta = {"category": "geo"}
    return [
        StructuredTool.from_function(
            func=clip_layer, name="clip_layer", metadata=meta,
            description=("CLIP: cookie-cut one layer with another. Keeps only the parts of "
                         "`target_file_id` that lie inside the boundary of `clip_file_id`, "
                         "cutting geometry where it crosses the edge (e.g. 'keep only the roads "
                         "inside the county'). Attributes of the target are preserved; the clip "
                         "layer contributes only its shape, and each remaining piece gets its "
                         "recomputed `area_km2` / `length_km` (measured in a projected CRS) so "
                         "you can see how much of each feature fell inside. CRS is aligned "
                         "automatically. Returns a downloadable GeoJSON + .csv table and puts "
                         "the result on the user's interactive map. " + _SIB)),
        StructuredTool.from_function(
            func=dissolve_layer, name="dissolve_layer", metadata=meta,
            description=("DISSOLVE / group: merge features into bigger ones. Features sharing the "
                         "same value in column `by` become a single feature (all features become "
                         "one when `by` is omitted), numeric columns are rolled up with "
                         "`statistic` (sum|mean|median|min|max|count|std|var|first|last) and a "
                         "`feature_count` column records how many rows each output merges — e.g. "
                         "counties -> states with population summed. Polygon results are shaded "
                         "as a choropleth on the interactive map, and the aggregated table is "
                         "also returned as a .csv. If `by` names a column that does not exist, "
                         "the error lists every candidate column. " + _SIB)),
        StructuredTool.from_function(
            func=intersect_layers, name="intersect_layers", metadata=meta,
            description=("INTERSECT: keep only the geometry where two layers overlap, and give "
                         "each resulting piece the attributes of BOTH inputs plus its measured "
                         "`area_km2` (computed in a projected CRS). Use it to answer 'how much of "
                         "each tract is inside the floodplain'. `how` can also be union, identity, "
                         "symmetric_difference or difference for the other overlay flavours. "
                         "Returns a downloadable GeoJSON + .csv table and maps the result. " + _SIB)),
        StructuredTool.from_function(
            func=erase_layer, name="erase_layer", metadata=meta,
            description=("ERASE / difference: subtract one layer from another. Removes from "
                         "`target_file_id` everything covered by `erase_file_id` and keeps the "
                         "remainder with the target's own attributes (e.g. 'land area minus "
                         "water', 'parcels outside the protected zone'). Works for points, lines "
                         "and polygons, and the surviving pieces carry their recomputed "
                         "`area_km2` / `length_km`. Returns a downloadable GeoJSON + .csv table "
                         "and maps the result. " + _SIB)),
        StructuredTool.from_function(
            func=buffer_layer, name="buffer_layer", metadata=meta,
            description=("BUFFER: draw a zone of a given GROUND distance around every feature — "
                         "'everything within 500 m of a school'. `distance` with `units` "
                         "(km|m|mi|ft|yd|nmi, default 1 km) is measured in a projected UTM CRS "
                         "estimated from the data, so the zone is a true distance and NOT a "
                         "number of degrees; the polygons come back as lon/lat. dissolve=True "
                         "merges overlapping zones into one continuous area. Each zone carries "
                         "its `area_km2`. Returns a downloadable GeoJSON + .csv and maps the "
                         "result. " + _SIB)),
        StructuredTool.from_function(
            func=simplify_layer, name="simplify_layer", metadata=meta,
            description=("SIMPLIFY / generalize: thin out vertices so a heavy boundary layer "
                         "draws fast and looks the same. `tolerance_m` is a real distance in "
                         "METRES (default 50), applied in a projected CRS. keep_topology=True "
                         "(default) uses coverage simplification when available so neighbouring "
                         "polygons keep their shared edges and no slivers or gaps appear. Reports "
                         "vertex counts before/after. Returns a downloadable GeoJSON and maps the "
                         "result. " + _SIB)),
        StructuredTool.from_function(
            func=geometry_summary, name="geometry_summary", metadata=meta,
            description=("GEOMETRY SUMMARY: derive simple shapes from a layer — `output`="
                         "'centroids' (one point per feature, the default, mapped as points), "
                         "'convex_hull' (tightest containing outline) or 'bbox' (bounding box). "
                         "per_feature=True gives one output per input feature; per_feature=False "
                         "summarizes the WHOLE layer into a single shape. Also reports measured "
                         "total area/length and the WGS84 bounds, and returns the per-feature "
                         "measurements as a .csv. Good for labelling polygons, showing where a "
                         "dataset is, or reducing polygons to plottable points. " + _SIB)),
    ]


__all__ = ["make_overlay_tools"]
