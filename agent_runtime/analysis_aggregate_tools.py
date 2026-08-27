"""Aggregation, proximity and pattern statistics — the questions users actually ask of point data.

"How many incidents per neighborhood?", "where are they densest?", "how far is each school from
the nearest hospital?", "are these clustered?". Each tool takes uploaded file_id(s) plus plain
scalars, does the metric work in a PROJECTED CRS (never in degrees), writes an EPSG:4326 artifact
for the web map, and returns a ``map_layer`` descriptor that
:func:`agent_runtime.map_layers.build_map_layer` forwards to the client as the ``map_layer`` SSE
event. A ``.csv`` (and for :func:`summary_statistics` a ``.png``) rides along so the numbers behind
the picture are downloadable.

Conventions come from :mod:`agent_runtime.langchain_geo_tools` and its readers are reused verbatim
(``read_vector`` handles GeoJSON / shapefile-with-sidecars / GeoPackage / GeoParquet /
CSV-with-coordinates including DMS). Heavy imports stay inside the tool bodies so importing this
module can never fail the agent boot, and every tool returns a JSON string and NEVER raises: on
failure it returns ``{"ok": false, "error": "...", ...candidates/hint}``.

Two audiences are served by the same functions: the defaults answer the question with no tuning
(``count_points_in_areas(points, areas)`` just works), while ``predicate``, ``statistic``,
``cell_km``, ``units``, ``classes`` and ``period`` are there for someone who knows what they want.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from agent_runtime.langchain_geo_tools import (
    _MAP_LAYER_MAX_FEATURES,
    _epsg,
    _index_attached,
    _resolve,
    _stage_vector_source,
    artifact_name,
    read_vector,
)

# sjoin predicates that make sense for "which area is this point in" style questions.
_PREDICATES = ("within", "intersects", "contains", "covered_by", "covers", "touches", "crosses",
               "overlaps", "contains_properly")
# Aggregations offered to power users; "count" needs no value column, the rest do.
_STATISTICS = ("count", "sum", "mean", "median", "min", "max", "std")
_UNITS: Dict[str, float] = {"m": 1.0, "km": 1000.0, "mi": 1609.344, "ft": 0.3048, "nmi": 1852.0}
# Grouping periods for a datetime `by` column ("incidents by month").
_PERIODS: Dict[str, str] = {"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}
# Columns most likely to be a human-readable name for an area, tried in order.
_LABEL_HINTS = ("name", "namelsad", "label", "title", "area", "zone", "district", "neighborhood",
                "community", "county", "city", "town", "state", "region", "geoid", "id")

_SIB = ("For an uploaded shapefile, pass the .shp's file_id (or any single component) — the tool "
        "auto-finds the .shx/.dbf/.prj among the attached files. GeoJSON, GeoPackage, GeoParquet "
        "and a CSV of coordinates all work directly.")


# --- plumbing shared by every tool ------------------------------------------------


def _dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, default=str)


def _fail(exc: BaseException, hint: Optional[str] = None, **extra: Any) -> str:
    """The uniform soft failure: never raise out of a tool."""
    out: Dict[str, Any] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    out.update(extra)
    if hint:
        out["hint"] = hint
    return _dumps(out)


def _bad(message: str, hint: Optional[str] = None, **extra: Any) -> str:
    return _fail(ValueError(message), hint, **extra)


def _geom_name(gdf: Any) -> str:
    """The active geometry column's name; ``geometry`` when none is set (a bare table)."""
    try:
        return str(gdf.geometry.name)
    except Exception:
        return "geometry"


def _numeric_columns(gdf: Any) -> List[str]:
    import pandas as pd

    geom = _geom_name(gdf)
    return [str(c) for c in gdf.columns
            if c != geom and pd.api.types.is_numeric_dtype(gdf[c])]


def _attribute_columns(gdf: Any) -> List[str]:
    geom = _geom_name(gdf)
    return [str(c) for c in gdf.columns if c != geom]


def _missing_column(role: str, requested: Any, gdf: Any, *, numeric: bool = True) -> str:
    """Column-not-found, reported with the CANDIDATES — never a blind first-N slice.

    A truncated dump of "columns" is what hides the very field being looked for (a join count
    lands last among 50+ TIGER fields), so list the numeric columns when a number is wanted and
    the full attribute list otherwise.
    """
    if numeric:
        cands = _numeric_columns(gdf)
        return _bad(f"{role} column {requested!r} is not a numeric column in the data",
                    hint=("pick one of numeric_columns, or use statistic='count' which needs no "
                          "column at all"),
                    numeric_columns=cands[:40], numeric_column_count=len(cands))
    cands = _attribute_columns(gdf)
    return _bad(f"{role} column {requested!r} is not in the data",
                hint="pick one of candidate_columns",
                candidate_columns=cands[:60], column_count=len(cands))


@contextmanager
def _open_vector(ref: str, siblings: Optional[List[str]] = None,
                 attached: Optional[List[Dict[str, Any]]] = None,
                 layer: Optional[str] = None) -> Iterator[Any]:
    """Yield a GeoDataFrame for an uploaded file_id or a path, cleaning up any staging dir.

    ``_stage_vector_source`` is what makes an EXTRACTED shapefile (.shp uploaded apart from its
    .dbf/.shx) readable at all, so route every read through it rather than opening the path.
    """
    read_path, tmp = _stage_vector_source(ref, siblings, attached)
    try:
        yield read_vector(read_path, layer)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def _source_stem(ref: str) -> str:
    """The ORIGINAL upload's filename stem, a far better default artifact name than the
    on-disk one (the store saves ``outputs/<file_id>__<name>``, so the raw path stem would
    put a random id in every download name)."""
    try:
        path, record = _resolve(ref)
    except Exception:
        return ""
    name = (record or {}).get("filename") or path.name
    return Path(str(name)).stem


def _stem(name: Optional[str], source_stem: str, purpose: str) -> str:
    """A purpose-bearing artifact stem: the caller's ``name``, else ``<source>_<purpose>``.

    ``artifact_name`` prefers a caller stem over the source file, so without this every tool run
    on ``crashes.geojson`` would emit another ``crashes.geojson`` and the conversation's download
    list would be a column of identical names. ``crashes_hex_grid`` / ``crashes_clusters`` says
    which run produced what.
    """
    if name and str(name).strip():
        return str(name).strip()
    return f"{source_stem}_{purpose}" if source_stem else purpose


def _ensure_crs(gdf: Any, notes: List[str], label: str) -> Any:
    if getattr(gdf, "crs", None) is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
        notes.append(f"{label} carried no CRS (missing .prj?); assumed EPSG:4326 lon/lat")
    return gdf


def _to_metric(gdf: Any, notes: List[str], label: str) -> Tuple[Any, Optional[str]]:
    """Reproject to a metre-based CRS so distances/areas/eps are real metres.

    Degrees are NOT a length: 1 degree of longitude is 111 km at the equator and 0 at the pole,
    so every metric step here (grid size, DBSCAN eps, nearest distance, cell area) runs on an
    estimated local UTM zone instead.
    """
    crs = getattr(gdf, "crs", None)
    try:
        units = {str(a.unit_name).lower() for a in (crs.axis_info or [])}
        if crs.is_projected and units & {"metre", "meter", "m"}:
            return gdf, _epsg(crs)                      # already metric — leave it alone
    except Exception:
        pass
    try:
        target: Any = gdf.estimate_utm_crs()
    except Exception as exc:
        target = "EPSG:3857"
        notes.append(f"could not estimate a local UTM zone for {label} ({exc}); used EPSG:3857, "
                     "so metric results are approximate far from the equator")
    out = gdf.to_crs(target)
    return out, _epsg(out.crs)


def _as_points(gdf: Any, notes: List[str], label: str, *, single: bool = False) -> Any:
    """Point geometry for aggregation. Non-point inputs collapse to a representative point
    (inside the shape, unlike the centroid of a concave polygon or a multipart county)."""
    types = {str(t) for t in gdf.geom_type.dropna().unique()}
    allowed = {"Point"} if single else {"Point", "MultiPoint"}
    if types and types <= allowed:
        return gdf
    gdf = gdf.copy()
    gdf[_geom_name(gdf)] = gdf.representative_point()
    if types - allowed:
        notes.append(f"{label} is {'/'.join(sorted(types))}, not points; used one representative "
                     "point per feature")
    return gdf


def _numeric_series(gdf: Any, column: str, notes: List[str]) -> Optional[Any]:
    """The column as numbers, coercing strings ("1,203" from a CSV) when possible; None if not."""
    import pandas as pd

    if column not in gdf.columns:
        return None
    series = gdf[column]
    if pd.api.types.is_numeric_dtype(series):
        return series
    coerced = pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")
    if coerced.notna().sum() == 0:
        return None
    notes.append(f"column {column!r} was text; coerced to numbers "
                 f"({int(coerced.isna().sum())} value(s) unparseable)")
    return coerced


def _label_column(gdf: Any, explicit: Optional[str]) -> Optional[str]:
    """A human-readable identifier column for the CSV, guessed when not given."""
    if explicit:
        return explicit if explicit in gdf.columns else None
    lowered = {str(c).lower(): c for c in gdf.columns if c != _geom_name(gdf)}
    for hint in _LABEL_HINTS:
        for low, original in lowered.items():
            if low == hint or low.endswith(f"_{hint}") or hint in low:
                return original
    return None


def _unique_column(existing: Any, base: str) -> str:
    """``base``, or ``base_2``/``base_3``… when the input already carries that name.

    Aggregating a SECOND point layer into the areas produced by a first call would otherwise
    overwrite the earlier ``point_count`` in place, quietly destroying the previous result the
    user is comparing against.
    """
    taken = {str(c) for c in existing}
    if base not in taken:
        return base
    for n in range(2, 100):
        candidate = f"{base}_{n}"
        if candidate not in taken:
            return candidate
    return f"{base}_x"


def _num(value: Any) -> Any:
    """JSON-safe scalar: numpy types -> python, NaN/inf -> None."""
    try:
        if value is None:
            return None
        as_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(as_float) or math.isinf(as_float):
        return None
    if isinstance(value, bool):
        return bool(value)
    return int(as_float) if float(as_float).is_integer() and abs(as_float) < 1e15 else round(as_float, 6)


def _class_breaks(values: Any, classes: int) -> Optional[List[float]]:
    """Quantile break points, so the answer can describe the choropleth's classes."""
    import numpy as np

    try:
        arr = np.asarray(values, dtype="float64")
    except Exception:
        return None
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or int(classes) < 2:
        return None
    qs = np.linspace(0.0, 100.0, int(classes) + 1)
    return [round(float(b), 6) for b in np.unique(np.percentile(arr, qs))]


def _stats_row(series: Any) -> Dict[str, Any]:
    """count/missing/min/max/mean/median/std/sum for one numeric series."""
    clean = series.dropna()
    return {
        "count": int(clean.shape[0]),
        "missing": int(series.shape[0] - clean.shape[0]),
        "min": _num(clean.min()) if not clean.empty else None,
        "max": _num(clean.max()) if not clean.empty else None,
        "mean": _num(clean.mean()) if not clean.empty else None,
        "median": _num(clean.median()) if not clean.empty else None,
        "std": _num(clean.std()) if clean.shape[0] > 1 else None,
        "sum": _num(clean.sum()) if not clean.empty else None,
    }


def _write_geojson(gdf: Any, name: Optional[str], source: str, default: str) -> Dict[str, Any]:
    """Write EPSG:4326 GeoJSON through the file store (GeoJSON, never parquet: the map client
    and every other tool can read it, parquet is a dead end for both)."""
    from agent_runtime.file_store import create_output_file_from_path

    if getattr(gdf, "crs", None) is not None:
        gdf = gdf.to_crs("EPSG:4326")
    fname = artifact_name(name, "geojson", source=source, default=default)
    tmpdir = Path(tempfile.mkdtemp(prefix="agg_gj_"))
    try:
        out = tmpdir / fname
        gdf.to_file(out, driver="GeoJSON")
        return create_output_file_from_path(out, filename=fname)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _write_csv(frame: Any, name: Optional[str], source: str, default: str) -> Dict[str, Any]:
    from agent_runtime.file_store import create_output_file_from_path

    fname = artifact_name(name, "csv", source=source, default=default)
    tmpdir = Path(tempfile.mkdtemp(prefix="agg_csv_"))
    try:
        out = tmpdir / fname
        frame.to_csv(out, index=False)
        return create_output_file_from_path(out, filename=fname)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _asset(rec: Dict[str, Any]) -> Dict[str, Any]:
    """The nested artifact shape the supervisor walks for download links."""
    return {"file_id": rec["file_id"], "filename": rec.get("filename"),
            "download_url": rec.get("download_url")}


def _sample_for_map(gdf: Any, render: str, max_points: Optional[int]) -> Tuple[Any, bool, int, Optional[str]]:
    """Cap a point layer at what is actually servable, and say so STRUCTURALLY — an answer that
    forgets to mention a sample is otherwise the only signal the layer is partial."""
    total = int(len(gdf))
    ceiling = int(max_points) if max_points else _MAP_LAYER_MAX_FEATURES
    if total > ceiling and render in {"points", "heatmap"}:
        note = (f"showing a random {ceiling} of {total} features "
                f"({ceiling * 100 // total}%) — the layer on the map is a SAMPLE")
        return gdf.sample(ceiling, random_state=0), True, total, note
    return gdf, False, total, None


def _map_layer(rec: Dict[str, Any], label: str, render: str, style_by: Optional[str],
               count: int, *, sampled: bool = False, total: Optional[int] = None) -> Dict[str, Any]:
    """The descriptor ``agent_runtime.map_layers.build_map_layer`` turns into the SSE event."""
    return {"url": rec.get("download_url"), "label": label, "render": render,
            "style_by": style_by, "source": "analysis", "count": int(count),
            "sampled": bool(sampled), "total": int(total if total is not None else count)}


def _label_for(name: Optional[str], rec: Dict[str, Any]) -> str:
    return (name or Path(str(rec.get("filename") or "layer")).stem).replace("_", " ")


# Attribute selection — the plainest GIS operation there is, and the one the toolkit was
# missing. Asked to "buffer the busiest cell", the analyze peer had no way to isolate one
# feature, so it buffered all 708 grid cells (4,504 overlapping polygons covering the city)
# and reported it as a buffer "around the busiest grid cell".
_SELECT_OPS = ("==", "!=", ">", ">=", "<", "<=", "in", "not_in", "contains", "between",
               "isnull", "notnull")


def _apply_predicate(gdf: Any, column: str, op: str, value: Any,
                     notes: List[str]) -> Tuple[Optional[Any], str]:
    """Filter *gdf* by ``column op value``. Returns (selection, human criterion)."""
    import pandas as pd

    col = gdf[column]
    key = str(op).strip().lower()
    if key in ("isnull", "notnull"):
        mask = col.isna() if key == "isnull" else col.notna()
        return gdf[mask], f"{column} {key}"
    if key in ("in", "not_in"):
        wanted = value if isinstance(value, (list, tuple, set)) else [value]
        as_str = col.astype(str)
        mask = as_str.isin([str(v) for v in wanted])
        return gdf[mask if key == "in" else ~mask], f"{column} {key} {list(wanted)!r}"
    if key == "contains":
        mask = col.astype(str).str.contains(str(value), case=False, na=False)
        return gdf[mask], f"{column} contains {value!r}"
    if key == "between":
        pair = list(value) if isinstance(value, (list, tuple)) else []
        if len(pair) != 2:
            notes.append("between needs value=[low, high]")
            return gdf.iloc[0:0], f"{column} between {value!r}"
        lo, hi = sorted(pd.to_numeric(pd.Series(pair), errors="coerce").tolist())
        num = pd.to_numeric(col, errors="coerce")
        return gdf[num.between(lo, hi)], f"{column} between {lo} and {hi}"
    if key not in _SELECT_OPS:
        return None, f"{column} {op} {value!r}"

    # Compare numerically when both sides are numbers, textually otherwise, so
    # value="12" against an int column still matches instead of silently returning nothing.
    num_col = pd.to_numeric(col, errors="coerce")
    num_val = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if num_col.notna().any() and pd.notna(num_val):
        left, right = num_col, num_val
    else:
        left, right = col.astype(str), str(value)
    mask = {"==": left == right, "!=": left != right, ">": left > right,
            ">=": left >= right, "<": left < right, "<=": left <= right}[key]
    return gdf[mask.fillna(False) if hasattr(mask, "fillna") else mask], f"{column} {key} {value!r}"


def make_aggregate_tools(default_input_file_ids: Optional[List[str]] = None) -> List[Any]:
    """Build the aggregation / proximity / pattern StructuredTools.

    ``default_input_file_ids`` is the conversation's attached file set; it is used only to
    auto-discover shapefile sidecars by basename, exactly as the geo tools do, so the model
    can reference a single .shp component.
    """
    from langchain_core.tools import StructuredTool

    attached = _index_attached(default_input_file_ids)

    def _open(ref, siblings=None, layer=None):
        return _open_vector(ref, siblings, attached, layer)

    # --- 1. point-in-polygon aggregation: the entry-level workhorse ---------------

    def count_points_in_areas(points_file_id: str, areas_file_id: str,
                              value_column: Optional[str] = None, statistic: str = "count",
                              predicate: str = "within", area_label_column: Optional[str] = None,
                              classes: int = 5, name: Optional[str] = None,
                              points_siblings: Optional[List[str]] = None,
                              areas_siblings: Optional[List[str]] = None) -> str:
        """Aggregate points into areas (point-in-polygon) and map the result as a CHOROPLETH.

        Returns the polygons carrying `point_count` (plus `<statistic>_<value_column>` when a
        value column is given), a choropleth map layer styled by that column, and a CSV of
        area -> value. `statistic`: count|sum|mean|median|min|max|std. `predicate`:
        within|intersects|contains|... for the power user.
        """
        notes: List[str] = []
        try:
            import geopandas as gpd
            import pandas as pd

            stat = str(statistic or "count").strip().lower()
            if stat not in _STATISTICS:
                return _bad(f"unsupported statistic {statistic!r}",
                            hint=f"use one of {list(_STATISTICS)}")
            pred = str(predicate or "within").strip().lower()
            if pred not in _PREDICATES:
                return _bad(f"unsupported predicate {predicate!r}",
                            hint=f"use one of {list(_PREDICATES)}")

            with _open(points_file_id, points_siblings) as pts_in, \
                    _open(areas_file_id, areas_siblings) as areas_in:
                points = _ensure_crs(pts_in, notes, "points layer")
                areas = _ensure_crs(areas_in, notes, "areas layer")

            if len(points) == 0 or len(areas) == 0:
                return _bad("one of the inputs has no features "
                            f"(points={len(points)}, areas={len(areas)})",
                            hint="check the file_ids point at the layers you meant")
            area_types = {str(t) for t in areas.geom_type.dropna().unique()}
            if area_types and not (area_types & {"Polygon", "MultiPolygon"}):
                return _bad(f"areas_file_id is {'/'.join(sorted(area_types))}, not polygons",
                            hint="pass the POLYGON layer as areas_file_id (arguments may be "
                                 "swapped); for proximity between two point layers use "
                                 "nearest_distance, for density without polygons use "
                                 "aggregate_to_grid")

            points = _as_points(points, notes, "points layer").reset_index(drop=True)
            if points.crs != areas.crs:
                points = points.to_crs(areas.crs)

            values = None
            if stat != "count":
                if not value_column:
                    return _bad(f"statistic={stat!r} needs a value_column",
                                hint="pass a numeric value_column, or statistic='count'",
                                numeric_columns=_numeric_columns(points)[:40])
                values = _numeric_series(points, value_column, notes)
                if values is None:
                    return _missing_column("value", value_column, points, numeric=True)

            # Carry an explicit area key rather than trusting sjoin's index_right naming.
            areas = areas.reset_index(drop=True)
            areas["_area_key"] = range(len(areas))
            geom_left = _geom_name(points)
            left = points[[geom_left]].copy()
            if values is not None:
                left["_value"] = values.to_numpy()
            joined = gpd.sjoin(left, areas[["_area_key", _geom_name(areas)]],
                               how="inner", predicate=pred)

            reserved = [c for c in areas.columns if c != "_area_key"]
            count_col = _unique_column(reserved, "point_count")
            out_col = count_col if stat == "count" else _unique_column(
                reserved + [count_col],
                f"{stat}_{str(value_column).strip().lower().replace(' ', '_')}"[:60])
            counts = joined.groupby("_area_key").size()
            result = areas.copy()
            result[count_col] = counts.reindex(result["_area_key"]).fillna(0).astype("int64").to_numpy()
            if stat != "count":
                agg = joined.groupby("_area_key")["_value"].agg(stat)
                # Areas with no points stay NULL for mean/median/min/max (0 would be a lie);
                # a sum of nothing is genuinely 0.
                agg = agg.reindex(result["_area_key"])
                if stat == "sum":
                    agg = agg.fillna(0)
                result[out_col] = agg.to_numpy()
            result = result.drop(columns=["_area_key"])

            style_by = out_col
            base = _stem(name, _source_stem(areas_file_id), "by_area")
            rec = _write_geojson(result, base, _source_stem(areas_file_id), "points_in_areas")

            label_col = _label_column(result, area_label_column)
            key = str(label_col) if label_col else "area"
            table = pd.DataFrame({
                key: (result[label_col].astype(str).to_numpy() if label_col
                      else result.index.astype(str)),
                count_col: result[count_col].to_numpy(),
            })
            if stat != "count":
                table[out_col] = result[out_col].to_numpy()
            table = table.sort_values(style_by, ascending=False)
            csv_rec = _write_csv(table, f"{base}_table", _source_stem(areas_file_id),
                                 "points_in_areas_table")

            matched = int(joined.index.nunique())
            payload: Dict[str, Any] = {
                "ok": True, "file_id": rec["file_id"], "filename": rec.get("filename"),
                "download_url": rec.get("download_url"), "feature_count": int(len(result)),
                "crs": "EPSG:4326", "statistic": stat, "value_column": value_column,
                "column": style_by, "count_column": count_col, "predicate": pred,
                "areas_total": int(len(result)),
                "areas_with_points": int((result[count_col] > 0).sum()),
                "points_total": int(len(points)), "points_matched": matched,
                "points_unmatched": int(len(points) - matched),
                "top_areas": [
                    {"area": str(area), "point_count": int(n)}
                    for area, n in table[[key, count_col]].head(5).itertuples(index=False)
                ],
                "class_breaks": _class_breaks(result[style_by].to_numpy(), classes),
                "csv": _asset(csv_rec), "on_map": True,
                "map_layer": _map_layer(rec, _label_for(name, rec), "choropleth", style_by,
                                        len(result)),
            }
            if matched == 0:
                notes.append("no point fell in any area — check the two layers actually overlap, "
                             "or try predicate='intersects'")
            if notes:
                payload["notes"] = notes
            return _dumps(payload)
        except Exception as exc:
            return _fail(exc, _SIB)

    # --- 2. grid binning: density without any polygon layer ----------------------

    def aggregate_to_grid(points_file_id: str, cell_km: float = 1.0, shape: str = "hex",
                          value_column: Optional[str] = None, statistic: str = "count",
                          min_count: int = 1, classes: int = 5, name: Optional[str] = None,
                          siblings: Optional[List[str]] = None) -> str:
        """Bin points into a hexagonal or square grid and map the density as a CHOROPLETH.

        `cell_km` is the real cell width in kilometres (centre-to-centre for hexes, side length
        for squares), measured in a projected CRS — never in degrees. Returns the occupied cells
        with `point_count` and `per_km2`, a choropleth layer, and a CSV of cell -> value.
        """
        notes: List[str] = []
        try:
            import geopandas as gpd
            import numpy as np
            import pandas as pd
            from shapely.geometry import Polygon, box

            stat = str(statistic or "count").strip().lower()
            if stat not in _STATISTICS:
                return _bad(f"unsupported statistic {statistic!r}",
                            hint=f"use one of {list(_STATISTICS)}")
            kind = str(shape or "hex").strip().lower()
            if kind in ("hexagon", "hexbin", "h3"):
                kind = "hex"
            if kind in ("square", "rect", "rectangle", "grid", "box"):
                kind = "square"
            if kind not in ("hex", "square"):
                return _bad(f"unsupported shape {shape!r}", hint="use shape='hex' or 'square'")
            try:
                size_m = float(cell_km) * 1000.0
            except (TypeError, ValueError):
                return _bad(f"cell_km must be a number, got {cell_km!r}")
            if not (size_m > 0) or not math.isfinite(size_m):
                return _bad(f"cell_km must be a positive number of kilometres, got {cell_km!r}")

            with _open(points_file_id, siblings) as raw:
                points = _ensure_crs(raw, notes, "input layer")
            if len(points) == 0:
                return _bad("input layer has no features")
            # Project FIRST, then collapse any non-point geometry: a representative point taken
            # in the projected CRS is the one that lands in the right metric cell.
            metric, metric_crs = _to_metric(points, notes, "input layer")
            metric = _as_points(metric, notes, "input layer", single=True)

            geom = metric.geometry
            x = geom.x.to_numpy(dtype="float64")
            y = geom.y.to_numpy(dtype="float64")
            finite = np.isfinite(x) & np.isfinite(y)
            if not finite.any():
                return _bad("no finite coordinates in the input layer")
            if not finite.all():
                notes.append(f"dropped {int((~finite).sum())} feature(s) with missing coordinates")

            values = None
            if stat != "count":
                if not value_column:
                    return _bad(f"statistic={stat!r} needs a value_column",
                                hint="pass a numeric value_column, or statistic='count'",
                                numeric_columns=_numeric_columns(metric)[:40])
                values = _numeric_series(metric, value_column, notes)
                if values is None:
                    return _missing_column("value", value_column, metric, numeric=True)

            if kind == "square":
                col = np.floor(x[finite] / size_m).astype("int64")
                row = np.floor(y[finite] / size_m).astype("int64")
                cell_area_km2 = (size_m / 1000.0) ** 2
            else:
                # Pointy-top hex lattice: rows every 1.5R, odd rows offset half a width. A hex
                # cell is the Voronoi cell of that lattice, and since 1.5R > R the nearest centre
                # is always in row floor(y/dy) or the one above — two candidates, no full grid.
                radius = size_m / math.sqrt(3.0)
                dy = 1.5 * radius
                xf, yf = x[finite], y[finite]
                base = np.floor(yf / dy).astype("int64")
                best_d = None
                row = col = None
                for cand_row in (base, base + 1):
                    offset = np.where(cand_row % 2 == 0, 0.0, size_m / 2.0)
                    cand_col = np.rint((xf - offset) / size_m).astype("int64")
                    cx = cand_col * size_m + offset
                    cy = cand_row * dy
                    dist = (xf - cx) ** 2 + (yf - cy) ** 2
                    if best_d is None:
                        best_d, row, col = dist, cand_row, cand_col
                    else:
                        take = dist < best_d
                        best_d = np.where(take, dist, best_d)
                        row = np.where(take, cand_row, row)
                        col = np.where(take, cand_col, col)
                cell_area_km2 = (math.sqrt(3.0) / 2.0) * (size_m / 1000.0) ** 2

            frame = pd.DataFrame({"row": row, "col": col})
            if values is not None:
                frame["_value"] = values.to_numpy()[finite]
            grouped = frame.groupby(["row", "col"], sort=True)
            cells = grouped.size().rename("point_count").reset_index()
            out_col = "point_count" if stat == "count" else \
                f"{stat}_{str(value_column).strip().lower().replace(' ', '_')}"[:60]
            if stat != "count":
                agg = grouped["_value"].agg(stat).rename(out_col).reset_index()
                cells = cells.merge(agg, on=["row", "col"], how="left")
            floor_n = max(int(min_count or 1), 1)
            before = len(cells)
            cells = cells[cells["point_count"] >= floor_n]
            if cells.empty:
                return _bad(f"no grid cell reached min_count={floor_n}",
                            hint="lower min_count or raise cell_km so cells hold more points")
            if len(cells) < before:
                notes.append(f"dropped {before - len(cells)} cell(s) below min_count={floor_n}")

            r = cells["row"].to_numpy(dtype="int64")
            c = cells["col"].to_numpy(dtype="int64")
            if kind == "square":
                polys = [box(cc * size_m, rr * size_m, (cc + 1) * size_m, (rr + 1) * size_m)
                         for rr, cc in zip(r, c)]
                cx = (c + 0.5) * size_m
                cy = (r + 0.5) * size_m
            else:
                radius = size_m / math.sqrt(3.0)
                dy = 1.5 * radius
                offset = np.where(r % 2 == 0, 0.0, size_m / 2.0)
                cx = c * size_m + offset
                cy = r * dy
                corners = [(math.cos(math.radians(a)) * radius, math.sin(math.radians(a)) * radius)
                           for a in (90, 150, 210, 270, 330, 30)]
                polys = [Polygon([(px + dx_, py + dy_) for dx_, dy_ in corners])
                         for px, py in zip(cx, cy)]

            grid = gpd.GeoDataFrame(
                {
                    "cell_id": [f"{kind[0]}_{int(rr)}_{int(cc)}" for rr, cc in zip(r, c)],
                    "point_count": cells["point_count"].to_numpy(),
                },
                geometry=polys, crs=metric.crs,
            )
            if stat != "count":
                grid[out_col] = cells[out_col].to_numpy()
            grid["per_km2"] = (grid["point_count"] / cell_area_km2).round(4)
            grid["cell_km"] = float(cell_km)
            centroids = gpd.GeoSeries(gpd.points_from_xy(cx, cy), crs=metric.crs).to_crs("EPSG:4326")

            style_by = out_col if stat != "count" else "point_count"
            base = _stem(name, _source_stem(points_file_id), f"{kind}_grid")
            rec = _write_geojson(grid, base, _source_stem(points_file_id), f"{kind}_grid")

            table = pd.DataFrame({
                "cell_id": grid["cell_id"].to_numpy(),
                "point_count": grid["point_count"].to_numpy(),
                "per_km2": grid["per_km2"].to_numpy(),
                "centroid_lon": centroids.x.round(6).to_numpy(),
                "centroid_lat": centroids.y.round(6).to_numpy(),
            })
            if stat != "count":
                table[out_col] = grid[out_col].to_numpy()
            table = table.sort_values("point_count", ascending=False)
            csv_rec = _write_csv(table, f"{base}_table", _source_stem(points_file_id),
                                 f"{kind}_grid_table")

            payload: Dict[str, Any] = {
                "ok": True, "file_id": rec["file_id"], "filename": rec.get("filename"),
                "download_url": rec.get("download_url"), "feature_count": int(len(grid)),
                "crs": "EPSG:4326", "shape": kind, "cell_km": float(cell_km),
                "cell_area_km2": round(cell_area_km2, 6), "metric_crs": metric_crs,
                "statistic": stat, "value_column": value_column, "column": style_by,
                "cells": int(len(grid)), "points_total": int(len(points)),
                "points_binned": int(finite.sum()),
                "max_cell_count": int(grid["point_count"].max()),
                "mean_cell_count": _num(grid["point_count"].mean()),
                "max_per_km2": _num(grid["per_km2"].max()),
                "class_breaks": _class_breaks(grid[style_by].to_numpy(), classes),
                "csv": _asset(csv_rec), "on_map": True,
                "map_layer": _map_layer(rec, _label_for(name, rec), "choropleth", style_by,
                                        len(grid)),
            }
            if notes:
                payload["notes"] = notes
            return _dumps(payload)
        except Exception as exc:
            return _fail(exc, _SIB)

    # --- 3. proximity: distance to the nearest feature of another layer ----------

    def nearest_distance(from_file_id: str, to_file_id: str, max_km: Optional[float] = None,
                         units: str = "km", to_label_column: Optional[str] = None,
                         name: Optional[str] = None,
                         from_siblings: Optional[List[str]] = None,
                         to_siblings: Optional[List[str]] = None) -> str:
        """Distance from every feature in one layer to the NEAREST feature in another, in metres.

        Computed in a projected CRS (never degrees). Returns the "from" features carrying
        `distance_m` / `distance_km`, a map layer styled by distance, and a CSV. `max_km` caps
        the search (beyond it the distance is null); `units` (m|km|mi|ft|nmi) sets the reported
        summary units.
        """
        notes: List[str] = []
        try:
            import geopandas as gpd
            import pandas as pd

            unit = str(units or "km").strip().lower()
            if unit not in _UNITS:
                return _bad(f"unsupported units {units!r}", hint=f"use one of {list(_UNITS)}")
            limit_m = None
            if max_km is not None:
                try:
                    limit_m = float(max_km) * 1000.0
                except (TypeError, ValueError):
                    return _bad(f"max_km must be a number of kilometres, got {max_km!r}")
                if limit_m <= 0:
                    return _bad(f"max_km must be positive, got {max_km!r}")

            with _open(from_file_id, from_siblings) as src_in, \
                    _open(to_file_id, to_siblings) as dst_in:
                source = _ensure_crs(src_in, notes, "from layer")
                target = _ensure_crs(dst_in, notes, "to layer")
            if len(source) == 0 or len(target) == 0:
                return _bad(f"empty input (from={len(source)}, to={len(target)})",
                            hint="both layers need features to measure between")

            metric, metric_crs = _to_metric(source, notes, "from layer")
            target = target.to_crs(metric.crs)

            keep = [_geom_name(target)]
            renamed = None
            if to_label_column:
                if to_label_column not in target.columns:
                    return _missing_column("to_label", to_label_column, target, numeric=False)
                renamed = "nearest_label"
                target = target.rename(columns={to_label_column: renamed})
                keep = [renamed, _geom_name(target)]
            right = target[keep]

            # A clean RangeIndex keeps the tie de-duplication below unambiguous.
            left = metric.drop(columns=[c for c in ("index_left", "index_right", "distance_m",
                                                    "distance_km", "nearest_label")
                                        if c in metric.columns]).reset_index(drop=True)
            joined = gpd.sjoin_nearest(left, right, how="left", distance_col="distance_m",
                                       max_distance=limit_m)
            # Ties return one row per equidistant neighbour; keep the closest single match and
            # restore the caller's feature order.
            joined = joined.sort_values("distance_m", na_position="last")
            joined = joined[~joined.index.duplicated(keep="first")].reindex(left.index)
            joined = joined.drop(columns=[c for c in ("index_right",) if c in joined.columns])
            joined["distance_km"] = (joined["distance_m"] / 1000.0).round(4)
            if unit not in ("m", "km"):
                joined[f"distance_{unit}"] = (joined["distance_m"] / _UNITS[unit]).round(4)
            joined["distance_m"] = joined["distance_m"].round(2)

            render = "points" if {str(t) for t in joined.geom_type.dropna().unique()} <= {"Point", "MultiPoint"} \
                else "choropleth"
            layer_gdf, sampled, total, sample_note = _sample_for_map(joined, render, None)
            if sample_note:
                notes.append(sample_note)
            base = _stem(name, _source_stem(from_file_id), "nearest")
            rec = _write_geojson(layer_gdf, base, _source_stem(from_file_id), "nearest_distance")

            label_col = _label_column(joined, None)
            table = pd.DataFrame({
                (label_col or "feature"): (joined[label_col].to_numpy() if label_col
                                           else joined.index.astype(str)),
                "distance_m": joined["distance_m"].to_numpy(),
                "distance_km": joined["distance_km"].to_numpy(),
            })
            if renamed and renamed in joined.columns:
                table["nearest_label"] = joined[renamed].to_numpy()
            table = table.sort_values("distance_m", na_position="last")
            csv_rec = _write_csv(table, f"{base}_table", _source_stem(from_file_id),
                                 "nearest_distance_table")

            dist = joined["distance_m"].dropna()
            factor = _UNITS[unit]
            payload: Dict[str, Any] = {
                "ok": True, "file_id": rec["file_id"], "filename": rec.get("filename"),
                "download_url": rec.get("download_url"), "feature_count": int(len(layer_gdf)),
                "crs": "EPSG:4326", "metric_crs": metric_crs, "column": "distance_m",
                "units": unit, "max_km": (float(max_km) if max_km is not None else None),
                "from_features": int(len(source)), "to_features": int(len(target)),
                "matched": int(dist.shape[0]),
                "unmatched": int(len(joined) - dist.shape[0]),
                "distance_summary_" + unit: ({
                    "min": _num(dist.min() / factor), "mean": _num(dist.mean() / factor),
                    "median": _num(dist.median() / factor), "max": _num(dist.max() / factor),
                } if not dist.empty else None),
                "sampled": bool(sampled), "features_total": total,
                "csv": _asset(csv_rec), "on_map": True,
                "map_layer": _map_layer(rec, _label_for(name, rec), render, "distance_m",
                                        len(layer_gdf), sampled=sampled, total=total),
            }
            if dist.empty:
                notes.append("nothing was within range" + (f" of max_km={max_km}" if max_km else "")
                             + "; raise or drop max_km")
            if notes:
                payload["notes"] = notes
            return _dumps(payload)
        except Exception as exc:
            return _fail(exc, _SIB)

    # --- 4. pattern: DBSCAN clusters ---------------------------------------------

    def cluster_points(file_id: str, eps_km: float = 0.5, min_samples: int = 10,
                       drop_noise: bool = False, name: Optional[str] = None,
                       siblings: Optional[List[str]] = None) -> str:
        """Find spatial clusters (DBSCAN) among points and map them coloured by cluster id.

        `eps_km` is the neighbourhood radius in real kilometres and `min_samples` the minimum
        points that make a cluster; both are applied in a projected CRS. Points in no cluster get
        cluster = -1 (noise). Returns the classified points plus a CSV of cluster, n, centroid.
        """
        notes: List[str] = []
        try:
            import geopandas as gpd
            import numpy as np
            import pandas as pd
        except Exception as exc:
            return _fail(exc, "the geopandas stack is unavailable in this runtime")
        try:
            from sklearn.cluster import DBSCAN
        except Exception as exc:  # scikit-learn is optional in slimmer images
            return _fail(exc, "clustering needs scikit-learn, which is not installed in this "
                              "runtime: `pip install scikit-learn`. For density without it, use "
                              "aggregate_to_grid (hex/square bins) or count_points_in_areas.")
        try:
            try:
                eps_m = float(eps_km) * 1000.0
                min_pts = int(min_samples)
            except (TypeError, ValueError):
                return _bad(f"eps_km must be a number and min_samples an integer; got "
                            f"eps_km={eps_km!r}, min_samples={min_samples!r}")
            if not (eps_m > 0) or not math.isfinite(eps_m):
                return _bad(f"eps_km must be a positive number of kilometres, got {eps_km!r}")
            if min_pts < 1:
                return _bad(f"min_samples must be >= 1, got {min_samples!r}")

            with _open(file_id, siblings) as raw:
                points = _ensure_crs(raw, notes, "input layer")
            if len(points) == 0:
                return _bad("input layer has no features")
            metric, metric_crs = _to_metric(points, notes, "input layer")
            metric = _as_points(metric, notes, "input layer", single=True)

            xs = metric.geometry.x.to_numpy(dtype="float64")
            ys = metric.geometry.y.to_numpy(dtype="float64")
            finite = np.isfinite(xs) & np.isfinite(ys)
            if not finite.any():
                return _bad("no finite coordinates in the input layer")
            if not finite.all():
                notes.append(f"ignored {int((~finite).sum())} feature(s) with missing coordinates")
            coords = np.column_stack([xs[finite], ys[finite]])
            # eps is metres because coords are metres — a degree-space eps would mean 111 km
            # near the equator and a few km near the poles.
            labels_finite = DBSCAN(eps=eps_m, min_samples=min_pts).fit(coords).labels_
            labels = np.full(len(metric), -1, dtype="int64")
            labels[finite] = labels_finite

            classified = metric.copy()
            classified["cluster"] = labels
            classified["is_noise"] = labels < 0
            out = classified.to_crs("EPSG:4326")
            if drop_noise:
                out = out[out["cluster"] >= 0]
                if out.empty:
                    return _bad("every point was classified as noise, so nothing is left to map",
                                hint=f"raise eps_km (now {eps_km}) or lower min_samples "
                                     f"(now {min_samples})")

            render = "points"
            layer_gdf, sampled, total, sample_note = _sample_for_map(out, render, None)
            if sample_note:
                notes.append(sample_note)
            base = _stem(name, _source_stem(file_id), "clusters")
            rec = _write_geojson(layer_gdf, base, _source_stem(file_id), "clusters")

            # Cluster centroid + radius are metric means, so compute them in the projected CRS
            # and transform ALL centroids in one pass (a per-cluster to_crs is N transforms).
            members = classified[classified["cluster"] >= 0]
            rows: List[Dict[str, Any]] = []
            if not members.empty:
                stats = pd.DataFrame({
                    "cluster": members["cluster"].to_numpy(),
                    "x": members.geometry.x.to_numpy(dtype="float64"),
                    "y": members.geometry.y.to_numpy(dtype="float64"),
                })
                centres = stats.groupby("cluster")[["x", "y"]].mean()
                joined_xy = stats.join(centres, on="cluster", rsuffix="_c")
                joined_xy["_r"] = np.sqrt((joined_xy["x"] - joined_xy["x_c"]) ** 2
                                          + (joined_xy["y"] - joined_xy["y_c"]) ** 2)
                radii = joined_xy.groupby("cluster")["_r"].max()
                sizes = stats.groupby("cluster").size()
                lonlat = gpd.GeoSeries(
                    gpd.points_from_xy(centres["x"].to_numpy(), centres["y"].to_numpy()),
                    crs=classified.crs).to_crs("EPSG:4326")
                for pos, cid in enumerate(centres.index):
                    rows.append({"cluster": int(cid), "n": int(sizes.loc[cid]),
                                 "centroid_lon": round(float(lonlat.x.iloc[pos]), 6),
                                 "centroid_lat": round(float(lonlat.y.iloc[pos]), 6),
                                 "radius_km": round(float(radii.loc[cid]) / 1000.0, 4)})
            rows.sort(key=lambda r: r["n"], reverse=True)
            noise_n = int((labels < 0).sum())
            table = pd.DataFrame(rows or [{"cluster": None, "n": 0, "centroid_lon": None,
                                           "centroid_lat": None, "radius_km": None}])
            csv_rec = _write_csv(table, f"{base}_summary", _source_stem(file_id),
                                 "cluster_summary")

            payload: Dict[str, Any] = {
                "ok": True, "file_id": rec["file_id"], "filename": rec.get("filename"),
                "download_url": rec.get("download_url"), "feature_count": int(len(layer_gdf)),
                "crs": "EPSG:4326", "metric_crs": metric_crs, "column": "cluster",
                "eps_km": float(eps_km), "min_samples": min_pts,
                "points_total": int(len(points)), "n_clusters": len(rows),
                "n_noise": noise_n,
                "noise_share": round(noise_n / max(len(labels), 1), 4),
                "clusters": rows[:20],
                "largest_cluster": rows[0] if rows else None,
                "sampled": bool(sampled), "features_total": total,
                "csv": _asset(csv_rec), "on_map": True,
                "map_layer": _map_layer(rec, _label_for(name, rec), render, "cluster",
                                        len(layer_gdf), sampled=sampled, total=total),
            }
            if not rows:
                notes.append(f"no cluster met min_samples={min_pts} within eps_km={eps_km}; "
                             "raise eps_km or lower min_samples")
            if notes:
                payload["notes"] = notes
            return _dumps(payload)
        except Exception as exc:
            return _fail(exc, _SIB)

    # --- 5. plain descriptive statistics (with optional group-by) ----------------

    def summary_statistics(file_id: str, column: Optional[str] = None, by: Optional[str] = None,
                           statistic: str = "mean", period: Optional[str] = None,
                           chart: bool = True, top: int = 25, name: Optional[str] = None,
                           siblings: Optional[List[str]] = None) -> str:
        """count / min / max / mean / median (+ std, sum) for a dataset's numeric columns.

        With no `column`, describes every numeric column. With `by`, groups by that column first
        (`period` = day|week|month|quarter|year buckets a date column, e.g. incidents by month).
        Returns a compact JSON summary, a CSV of the table, and a PNG chart.
        """
        notes: List[str] = []
        try:
            import pandas as pd

            stat = str(statistic or "mean").strip().lower()
            if stat not in _STATISTICS:
                return _bad(f"unsupported statistic {statistic!r}",
                            hint=f"use one of {list(_STATISTICS)}")
            with _open(file_id, siblings) as raw:
                gdf = raw
                numeric_cols = _numeric_columns(gdf)
                attribute_cols = _attribute_columns(gdf)
                frame = pd.DataFrame({c: gdf[c] for c in attribute_cols})
                rows_total = int(len(gdf))
            if rows_total == 0:
                return _bad("input layer has no features")

            if column is not None and column not in numeric_cols:
                # Might still be a coercible text column (CSV numbers as strings).
                coerced = _numeric_series(gdf, column, notes) if column in frame.columns else None
                if coerced is None:
                    return _missing_column("value", column, gdf, numeric=True)
                frame[column] = coerced.to_numpy()
                numeric_cols = numeric_cols + [column]

            group_key = None
            if by:
                if by not in frame.columns:
                    return _missing_column("group-by", by, gdf, numeric=False)
                group_key = frame[by]
                if period:
                    freq = _PERIODS.get(str(period).strip().lower())
                    if not freq:
                        return _bad(f"unsupported period {period!r}",
                                    hint=f"use one of {list(_PERIODS)}")
                    parsed = pd.to_datetime(group_key, errors="coerce")
                    if parsed.notna().sum() == 0:
                        return _bad(f"column {by!r} could not be read as dates, so period="
                                    f"{period!r} cannot be applied",
                                    hint="pass a date/datetime column as `by`, or drop `period`")
                    group_key = parsed.dt.to_period(freq).astype(str)
                    notes.append(f"grouped {by!r} into {period} buckets")
                group_key = group_key.fillna("(missing)").astype(str)

            targets = [column] if column else numeric_cols
            summary: Dict[str, Any] = {"rows": rows_total, "numeric_columns": numeric_cols[:40]}
            chart_kind = None

            if group_key is None:
                if not targets:
                    return _bad("the dataset has no numeric column to summarize",
                                hint="pass `by` to count rows per category instead",
                                candidate_columns=attribute_cols[:60])
                records = []
                for col in targets:
                    row = _stats_row(frame[col])
                    records.append({"column": col, **row})
                table = pd.DataFrame(records)
                summary["columns"] = {r["column"]: {k: v for k, v in r.items() if k != "column"}
                                      for r in records}
                chart_kind = "hist" if (column and column in frame.columns) else "bar_columns"
            else:
                grouped = frame.groupby(group_key.to_numpy(), dropna=False, sort=True)
                sizes = grouped.size().rename("n")
                if targets:
                    col = targets[0]
                    if len(targets) > 1:
                        notes.append(f"grouped statistics reported for {col!r}; pass `column` to "
                                     "choose another")
                    agg = grouped[col].agg(["min", "max", "mean", "median", "std", "sum"])
                    agg.columns = [f"{a}_{col}" for a in agg.columns]
                    # concat on the shared group index, not positionally: a merge by position is
                    # exactly how group statistics end up attached to the wrong group.
                    table = pd.concat([sizes, agg], axis=1).reset_index(names=[by or "group"])
                    sort_col = f"{stat}_{col}" if f"{stat}_{col}" in table.columns else "n"
                    table = table.sort_values(sort_col, ascending=False)
                    summary["grouped_column"] = col
                    summary["group_statistic"] = stat
                else:
                    table = sizes.reset_index(names=[by or "group"]).sort_values(
                        "n", ascending=False)
                    notes.append("no numeric column present; reported row counts per group")
                summary["group_by"] = by
                summary["group_count"] = int(len(table))
                summary["groups"] = [
                    {k: (_num(v) if not isinstance(v, str) else v) for k, v in rec.items()}
                    for rec in table.head(max(int(top or 25), 1)).to_dict("records")
                ]
                chart_kind = "bar_groups"

            base = _stem(name, _source_stem(file_id), "summary")
            csv_rec = _write_csv(table, base, _source_stem(file_id), "summary_statistics")
            payload: Dict[str, Any] = {
                "ok": True, "file_id": csv_rec["file_id"], "filename": csv_rec.get("filename"),
                "download_url": csv_rec.get("download_url"),
                "feature_count": rows_total, "row_count": int(len(table)),
                "column": column, "by": by, "period": period, "statistic": stat,
                "summary": summary, "csv": _asset(csv_rec),
            }

            if chart:
                try:
                    import matplotlib
                    matplotlib.use("Agg")            # headless: no display in the runtime
                    import matplotlib.pyplot as plt
                    from agent_runtime.file_store import create_output_file_from_path

                    fig, ax = plt.subplots(figsize=(9, 5))
                    if chart_kind == "hist" and column:
                        ax.hist(frame[column].dropna().to_numpy(), bins=30, color="#3aa9a0",
                                edgecolor="#1c5a97")
                        ax.set_xlabel(str(column))
                        ax.set_ylabel("features")
                        ax.set_title(f"Distribution of {column} (n={rows_total})")
                    elif chart_kind == "bar_groups":
                        head = table.head(max(int(top or 25), 1))
                        value_col = next((c for c in head.columns
                                          if c.startswith(f"{stat}_")), "n")
                        labels = head[head.columns[0]].astype(str).to_numpy()
                        ax.barh(labels[::-1], head[value_col].to_numpy()[::-1], color="#3aa9a0")
                        ax.set_xlabel(str(value_col))
                        ax.set_title(f"{value_col} by {by}" + (f" ({period})" if period else ""))
                    else:
                        head = table.head(20)
                        ax.barh(head["column"].astype(str).to_numpy()[::-1],
                                head["mean"].fillna(0).to_numpy()[::-1], color="#3aa9a0")
                        ax.set_xlabel("mean")
                        ax.set_title("Mean of each numeric column")
                    fig.tight_layout()
                    png_dir = Path(tempfile.mkdtemp(prefix="agg_png_"))
                    try:
                        png_name = artifact_name(base, "png", source=_source_stem(file_id),
                                                 default="summary_chart")
                        png_path = png_dir / png_name
                        fig.savefig(png_path, dpi=150)
                        plt.close(fig)
                        png_rec = create_output_file_from_path(png_path, filename=png_name)
                        payload["chart"] = _asset(png_rec)
                    finally:
                        shutil.rmtree(png_dir, ignore_errors=True)
                except Exception as chart_exc:      # a missing chart must not lose the numbers
                    notes.append(f"chart not produced ({type(chart_exc).__name__}: {chart_exc}); "
                                 "the CSV and summary are complete")
            if notes:
                payload["notes"] = notes
            return _dumps(payload)
        except Exception as exc:
            return _fail(exc, _SIB)

    def select_by_attribute(file_id: str, column: Optional[str] = None, op: str = "==",
                            value: Optional[Any] = None, top_n: Optional[int] = None,
                            ascending: bool = False, render: str = "auto",
                            style_by: Optional[str] = None, name: Optional[str] = None,
                            sibling_file_ids: Optional[List[str]] = None,
                            layer: Optional[str] = None) -> str:
        """Keep only the features matching an attribute test, or the top/bottom N by a column.

        `column` + `top_n` ranks (ascending=True for the smallest); `column` + `op` + `value`
        compares. Returns the selection as its own layer plus a CSV, so it can be fed straight
        into buffer_layer / clip_layer instead of operating on the whole input.
        """
        try:
            with _open(file_id, sibling_file_ids, layer) as gdf:
                source = _source_stem(file_id)
                notes: List[str] = []
                total_in = int(len(gdf))
                if column is not None and column not in gdf.columns:
                    return _missing_column("attribute", column, gdf, numeric=False)

                if top_n is not None:
                    if column is None:
                        return _bad("top_n needs a `column` to rank by.",
                                    "Pass the column holding the value to rank on.",
                                    numeric_columns=_numeric_columns(gdf))
                    series = _numeric_series(gdf, column, notes)
                    if series is None:
                        return _missing_column("ranking", column, gdf, numeric=True)
                    n = max(int(top_n), 1)
                    order = series.sort_values(ascending=bool(ascending), na_position="last")
                    selected = gdf.loc[order.index[:n]]
                    criterion = f"{'bottom' if ascending else 'top'} {n} by {column}"
                else:
                    if column is None or (value is None
                                          and str(op).lower() not in ("isnull", "notnull")):
                        return _bad("select_by_attribute needs `column` plus either a `value` "
                                    "(with `op`) or `top_n`.",
                                    "Example: column='point_count', op='>=', value=50 — or "
                                    "column='point_count', top_n=1 for the single busiest feature.",
                                    columns=_attribute_columns(gdf),
                                    accepted_ops=list(_SELECT_OPS))
                    selected, criterion = _apply_predicate(gdf, column, str(op), value, notes)
                    if selected is None:
                        return _bad(f"unsupported op {op!r}.",
                                    "Use one of the accepted comparisons.",
                                    accepted_ops=list(_SELECT_OPS))

                if not len(selected):
                    # Never hand back an empty layer: say what the column actually holds so the
                    # next attempt can be right instead of mapping nothing.
                    probe: Dict[str, Any] = {"matched": 0}
                    if column in gdf.columns:
                        col = gdf[column]
                        numeric = _numeric_series(gdf, column, [])
                        if numeric is not None and len(numeric.dropna()):
                            probe["range"] = [_num(numeric.min()), _num(numeric.max())]
                        probe["example_values"] = [str(v) for v in col.dropna().unique()[:12]]
                    return _bad(f"{criterion} matched no features — not writing an empty layer.",
                                "Widen the test, or check the value against what the column holds.",
                                column=column, **probe)

                mode = (render or "auto").strip().lower()
                if mode == "auto":
                    kinds = {str(k) for k in selected.geometry.geom_type.unique()}
                    mode = "points" if kinds and kinds <= {"Point", "MultiPoint"} else "shapes"
                if style_by and style_by not in selected.columns:
                    notes.append(f"style_by {style_by!r} is not a column of the result; dropped")
                    style_by = None

                stem = _stem(name, source, "selected")
                mapped, sampled, total_out, sample_note = _sample_for_map(selected, mode, None)
                if sample_note:
                    notes.append(sample_note)
                rec = _write_geojson(mapped, name, source, stem)
                table = selected.drop(columns=[_geom_name(selected)], errors="ignore")
                csv_rec = _write_csv(table, name, source, f"{stem}_table")
                payload: Dict[str, Any] = {
                    "ok": True, "file_id": rec["file_id"], "filename": rec.get("filename"),
                    "download_url": rec.get("download_url"),
                    "feature_count": int(len(selected)), "features_in": total_in,
                    "criterion": criterion, "column": column, "csv": _asset(csv_rec),
                    "on_map": True,
                    "map_layer": _map_layer(rec, _label_for(name, rec), mode, style_by,
                                            int(len(mapped)), sampled=sampled, total=total_out),
                }
                if notes:
                    payload["notes"] = notes
                return _dumps(payload)
        except Exception as exc:
            return _fail(exc, _SIB)

    meta = {"category": "geo"}
    return [
        StructuredTool.from_function(
            func=select_by_attribute, name="select_by_attribute", metadata=meta,
            description=(
                "SELECT A SUBSET of a layer by its attributes — the everyday GIS 'select by "
                "attribute' / query, and the step that turns a whole layer into THE ONE FEATURE "
                "you meant. 'the busiest grid cell' is top_n=1 on the count column; 'cells with "
                "50+ incidents' is op='>=', value=50; 'only ward 12' is op='==', value=12. "
                "column + top_n ranks (ascending=True for the smallest), column + op + value "
                "compares: " + "|".join(_SELECT_OPS) + ". Returns the matching features as their "
                "own GeoJSON layer on the map plus a CSV, so you can feed the result straight "
                "into buffer_layer/clip_layer instead of operating on every feature. A test that "
                "matches nothing returns the column's real range and example values rather than "
                "an empty layer. " + _SIB)),
        StructuredTool.from_function(
            func=count_points_in_areas, name="count_points_in_areas", metadata=meta,
            description=(
                "Aggregate POINTS INTO AREAS (point-in-polygon) and put the result on the user's "
                "INTERACTIVE MAP as a choropleth: 'how many crashes per neighborhood', 'total "
                "population per district', 'average price per zip'. Pass the point layer and the "
                "polygon layer by file_id; the polygons come back carrying `point_count` (and "
                "`<statistic>_<value_column>` when you name a value column), plus a CSV of "
                "area -> value. statistic: count|sum|mean|median|min|max|std; predicate: "
                "within|intersects|... CRS is aligned automatically. " + _SIB)),
        StructuredTool.from_function(
            func=aggregate_to_grid, name="aggregate_to_grid", metadata=meta,
            description=(
                "Bin POINTS INTO A HEX OR SQUARE GRID and map the density as a choropleth — the "
                "answer to 'where are these densest?' when there is no polygon layer to aggregate "
                "into. `cell_km` is the real cell width in kilometres (computed in a projected "
                "CRS, never degrees), `shape` is 'hex' or 'square'. Returns the occupied cells "
                "with point_count and per_km2, plus a CSV. " + _SIB)),
        StructuredTool.from_function(
            func=nearest_distance, name="nearest_distance", metadata=meta,
            description=(
                "Measure how far each feature of one layer is from the NEAREST feature of another "
                "('distance from every school to the closest hospital'). Distances are true "
                "metres computed in a projected CRS. Returns the from-features carrying "
                "distance_m/distance_km, a map layer styled by distance, a CSV, and min/mean/"
                "median/max. Optional max_km caps the search; units sets the reported units. "
                + _SIB)),
        StructuredTool.from_function(
            func=cluster_points, name="cluster_points", metadata=meta,
            description=(
                "Detect spatial CLUSTERS / HOTSPOTS among points with DBSCAN and map them "
                "coloured by cluster id. `eps_km` is the neighbourhood radius in real kilometres "
                "and `min_samples` the minimum points per cluster; unclustered points are noise "
                "(cluster = -1). Returns the classified points plus a CSV of cluster, n, "
                "centroid and radius. Needs scikit-learn. " + _SIB)),
        StructuredTool.from_function(
            func=summary_statistics, name="summary_statistics", metadata=meta,
            description=(
                "Descriptive statistics for a dataset's attributes: count, min, max, mean, "
                "median, std, sum — for every numeric column by default, for one `column` when "
                "named, and per group when `by` is given (`period`=day|week|month|quarter|year "
                "buckets a date column). Returns a compact JSON summary plus a downloadable CSV "
                "and PNG chart. Use this for the numbers behind a map, not for mapping. " + _SIB)),
    ]


__all__ = ["make_aggregate_tools"]
