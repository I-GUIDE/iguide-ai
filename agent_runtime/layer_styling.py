"""What add_map_layer can ask the map to do with a layer's features, and what it then says it did.

Stage 80 (2026-10-18). Live demo tests of the Ridgecrest aftershocks example answered "sized by
magnitude, the mainshock is the largest point" over a map of 1,054 equal circles: point size was
a constant in the client (toDeckLayer.ts), and nothing could single out or label a feature. The
model asked for what the user asked for, the tool had no way to do it, and the answer described
the request instead of the map.

So the tool takes a small set of NAMED options, each validated against the data before anything
is written, and the client renders exactly those (map-ui-prototype toDeckLayer.ts):

* ``size_column`` (+ ``size_range``): per-point radius on a square-root scale;
  ``point_size``: one fixed radius.
* ``highlight``: one predicate on one column; matching features drawn larger, in a stand-out
  colour, on top, with a legend entry.
* ``label_column``: text labels, at most :data:`MAX_LABELS` of them.
* ``line_width`` / ``line_color``.

Free-form deck.gl or MapLibre style specs are deliberately not accepted: the client would have
to trust them, and the tool could not say what the map shows.

Every check here returns an answerable ``{ok: False, error, hint}`` rather than guessing, and
:func:`describe` builds the sentence the tool result carries about what will actually be drawn.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

# Pixels. Radius = lo + (hi - lo) * sqrt(t), t the value's place in [min, max] (toDeckLayer.ts
# pointRadiusOf); the ends are what the model may change.
SIZE_RANGE_DEFAULT: Tuple[float, float] = (3.0, 16.0)
_SIZE_MIN_PX, _SIZE_MAX_PX = 1.0, 40.0
_LINE_MIN_PX, _LINE_MAX_PX = 0.5, 20.0
# More labels than this is a wall of text over the map, not a key to it.
MAX_LABELS = 50
_LABEL_CHARS = 40
# Gold with a dark outline: reads over the default purple layers, the OrRd ramp and both basemaps.
HIGHLIGHT_COLOUR = [255, 193, 7, 255]
# A predicate, spelled the way a model writes one. Values are the op the client evaluates.
_OPS = {"equals": "eq", "eq": "eq", "==": "eq", "in": "in",
        "gte": "gte", ">=": "gte", "at_least": "gte", "min": "gte",
        "lte": "lte", "<=": "lte", "at_most": "lte", "max": "lte",
        "gt": "gt", ">": "gt", "lt": "lt", "<": "lt"}
_OP_TEXT = {"eq": "==", "in": "in", "gte": ">=", "lte": "<=", "gt": ">", "lt": "<"}


def _fail(error: str, hint: str, **extra: Any) -> Dict[str, Any]:
    return {"ok": False, "error": error, "hint": hint, **extra}


def _numeric(series) -> bool:
    import pandas as pd
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _numbers_in_text(gdf, col: str) -> bool:
    """Make a text column of numbers numeric, in place; True when it now is.

    GDAL reads every column of a CSV as text, so the Ridgecrest catalogue's `mag` arrived as
    "3.98". A column is converted only when EVERY non-empty value parses, so an id like "06"
    among names stays text. The written layer then carries numbers, which is what the client
    sizes and compares by.
    """
    import pandas as pd

    series = gdf[col]
    if _numeric(series):
        return True
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    present = series.dropna()
    present = present[present.astype(str).str.strip() != ""]
    if present.empty:
        return False
    parsed = pd.to_numeric(present, errors="coerce")
    if parsed.isna().any():
        return False
    gdf[col] = pd.to_numeric(series.where(series.astype(str).str.strip() != ""), errors="coerce")
    return True


def _columns(gdf) -> List[str]:
    return [c for c in gdf.columns if c != "geometry"][:40]


def _px(value: Any, name: str, lo: float, hi: float) -> Tuple[Optional[float], Optional[Dict]]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = float("nan")
    if not (math.isfinite(v) and lo <= v <= hi):
        return None, _fail(f"{name} is a size in pixels from {lo:g} to {hi:g}; got {value!r}",
                           f"Pass a number such as {name}=" + ("2" if "line" in name else "8") + ".")
    return v, None


def _parse_highlight(raw: Any) -> Tuple[Optional[Dict[str, Any]], Optional[Dict]]:
    """``{column, <op>: value[, color, label]}`` -> ``{column, op, value, color?, label?}``."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raw = None
    if not isinstance(raw, dict) or not raw.get("column"):
        return None, _fail("highlight is {column, equals|in|gte|lte|gt|lt: value}",
                           "For example highlight={'column': 'type', 'equals': 'mainshock'} or "
                           "{'column': 'mag', 'gte': 7}.")
    ops = [k for k in raw if k in _OPS]
    if len(ops) != 1:
        return None, _fail("highlight takes exactly one condition "
                           f"({', '.join(sorted(set(_OPS) - {'==', '>=', '<=', '>', '<'}))})",
                           "For example {'column': 'mag', 'gte': 7}; to single out several "
                           "values use {'column': 'type', 'in': ['a', 'b']}.")
    op = _OPS[ops[0]]
    out: Dict[str, Any] = {"column": str(raw["column"]), "op": op, "value": raw[ops[0]]}
    if op == "in" and not isinstance(out["value"], (list, tuple)):
        out["value"] = [out["value"]]
    if raw.get("color") is not None:
        from agent_runtime.map_layers import parse_colour
        rgba = parse_colour(raw["color"])
        if not rgba:
            return None, _fail(f"{raw['color']!r} (highlight colour) is not a colour",
                               "Use a colour name (red, gold) or hex (#d7191c), or leave it out "
                               "for the map's highlight gold.")
        out["color"] = rgba
    if raw.get("label"):
        out["label"] = str(raw["label"])[:80]
    return out, None


def _value_text(v: Any) -> str:
    if isinstance(v, float) and math.isfinite(v):
        return f"{v:g}"
    return str(v)


def highlight_mask(gdf, hl: Dict[str, Any]):
    """Which rows the highlight picks out — the SAME rule the client applies (toDeckLayer.ts
    ``highlightTest``): numbers compare as numbers on a numeric column, everything else as text."""
    import pandas as pd

    col = gdf[hl["column"]]
    op, value = hl["op"], hl["value"]
    if hl.get("numeric"):
        nums = pd.to_numeric(col, errors="coerce")
        if op == "in":
            return nums.isin([float(v) for v in value])
        v = float(value)
        return {"eq": nums == v, "gte": nums >= v, "lte": nums <= v,
                "gt": nums > v, "lt": nums < v}[op].fillna(False)
    text = col.map(lambda x: None if x is None or (isinstance(x, float) and math.isnan(x))
                   else _value_text(x))
    if op == "in":
        return text.isin([_value_text(v) for v in value])
    return text == _value_text(value)


def validate(gdf, *, mode: str, geom_types: set, size_column: Any = None, size_range: Any = None,
             point_size: Any = None, highlight: Any = None, label_column: Any = None,
             line_width: Any = None, line_color: Any = None) -> Tuple[Dict[str, Any], Optional[Dict]]:
    """Check every styling option against the data, before anything is written.

    Returns ``(spec, None)`` or ``({}, error)``. ``spec`` holds only the options that were set.
    """
    spec: Dict[str, Any] = {}
    is_point = bool(geom_types) and geom_types <= {"Point", "MultiPoint"}

    if size_column is not None or point_size is not None or size_range is not None:
        if size_column is not None and point_size is not None:
            return {}, _fail("size_column and point_size both set the point size",
                             "Pass size_column to size points by a value, or point_size for "
                             "one size, not both.")
        if size_range is not None and size_column is None:
            return {}, _fail("size_range needs size_column — it is the range values are scaled to",
                             "Pass size_column too, or point_size for one fixed size.")
        if mode == "heatmap":
            return {}, _fail("a heatmap draws density, not points, so it has no point size",
                             "Use render='points' to size the points themselves.")
        if not is_point:
            return {}, _fail(f"point size applies to points; this layer holds {sorted(geom_types)}",
                             "Drop size_column/point_size, or use line_width for lines and "
                             "outlines.")
    if point_size is not None:
        spec["point_size"], err = _px(point_size, "point_size", _SIZE_MIN_PX, _SIZE_MAX_PX)
        if err:
            return {}, err
    if size_column is not None:
        col = str(size_column)
        if col not in gdf.columns:
            return {}, _fail(f"size_column {col!r} is not a column of this layer",
                             "Pass one of the numeric columns.",
                             numeric_columns=[c for c in _columns(gdf) if _numeric(gdf[c])])
        if not _numbers_in_text(gdf, col):
            return {}, _fail(f"size_column {col!r} is not numeric, so it cannot set a size",
                             "Size by a numeric column; colour classes with render='categories'.",
                             numeric_columns=[c for c in _columns(gdf) if _numeric(gdf[c])])
        lo, hi = SIZE_RANGE_DEFAULT
        if size_range is not None:
            if isinstance(size_range, str):
                try:
                    size_range = json.loads(size_range)
                except ValueError:
                    size_range = None
            if not (isinstance(size_range, (list, tuple)) and len(size_range) == 2):
                return {}, _fail("size_range is [smallest_px, largest_px]",
                                 f"For example [{SIZE_RANGE_DEFAULT[0]:g}, {SIZE_RANGE_DEFAULT[1]:g}].")
            lo, err = _px(size_range[0], "size_range[0]", _SIZE_MIN_PX, _SIZE_MAX_PX)
            if err:
                return {}, err
            hi, err = _px(size_range[1], "size_range[1]", _SIZE_MIN_PX, _SIZE_MAX_PX)
            if err:
                return {}, err
            if hi <= lo:
                return {}, _fail("size_range must grow: [smallest_px, largest_px]",
                                 f"For example [{SIZE_RANGE_DEFAULT[0]:g}, {SIZE_RANGE_DEFAULT[1]:g}].")
        vals = gdf[col].astype(float)
        finite = vals[vals.map(math.isfinite)]
        if finite.empty:
            return {}, _fail(f"size_column {col!r} holds no numbers", "Pick a column with values.")
        if finite.min() == finite.max():
            return {}, _fail(f"size_column {col!r} is {finite.min():g} everywhere, so every point "
                             "would be one size",
                             "Use point_size for one size, or size by a column that varies.")
        spec["size_by"] = {"column": col, "range": [lo, hi]}

    if highlight is not None:
        hl, err = _parse_highlight(highlight)
        if err:
            return {}, err
        if hl["column"] not in gdf.columns:
            return {}, _fail(f"highlight column {hl['column']!r} is not a column of this layer",
                             "Pass one of the columns listed.", columns=_columns(gdf))
        numeric_col = _numbers_in_text(gdf, hl["column"])
        values = hl["value"] if hl["op"] == "in" else [hl["value"]]
        if hl["op"] in {"gte", "lte", "gt", "lt"} and not numeric_col:
            return {}, _fail(f"{_OP_TEXT[hl['op']]} compares numbers; {hl['column']!r} is not numeric",
                             "Use equals or in for a text column.")
        if numeric_col:
            try:
                [float(v) for v in values]
            except (TypeError, ValueError):
                return {}, _fail(f"{hl['column']!r} is numeric; {hl['value']!r} is not a number",
                                 "Compare it with a number, e.g. {'column': 'mag', 'gte': 7}.")
        hl["numeric"] = bool(numeric_col)
        n = int(highlight_mask(gdf, hl).sum())
        if n == 0:
            sample = sorted({_value_text(v) for v in gdf[hl["column"]].dropna().unique()})[:20]
            return {}, _fail(f"highlight {hl['column']} {_OP_TEXT[hl['op']]} {hl['value']!r} matches "
                             "no feature, so nothing would stand out",
                             "Check the value against the data; a highlight that matches nothing "
                             "is not drawn.",
                             **({"values": sample} if not numeric_col else
                                {"range": [float(gdf[hl["column"]].min()),
                                           float(gdf[hl["column"]].max())]}))
        if n == len(gdf):
            return {}, _fail(f"highlight matches all {n} features, so nothing would stand out",
                             "Tighten the condition so it picks out the features that matter.")
        spec["highlight"] = hl

    if label_column is not None:
        col = str(label_column)
        if col not in gdf.columns:
            return {}, _fail(f"label_column {col!r} is not a column of this layer",
                             "Pass one of the columns listed.", columns=_columns(gdf))
        spec["label_column"] = col

    if line_width is not None:
        if mode == "heatmap":
            return {}, _fail("a heatmap has no lines to set a width on",
                             "Use render='points' or 'shapes' for outlines.")
        spec["line_width"], err = _px(line_width, "line_width", _LINE_MIN_PX, _LINE_MAX_PX)
        if err:
            return {}, err
    if line_color is not None:
        if mode == "heatmap":
            return {}, _fail("a heatmap has no lines to colour",
                             "Pass `color` to tint the heatmap, or use render='points'.")
        from agent_runtime.map_layers import parse_colour
        rgba = parse_colour(line_color)
        if not rgba:
            return {}, _fail(f"{line_color!r} is not a colour",
                             "Use a colour name (black, white) or hex (#333333).")
        spec["line_color"] = rgba
    return spec, None


def columns_needed(spec: Dict[str, Any]) -> List[str]:
    """Columns the styling reads, so the render branches that slice the frame keep them."""
    out = []
    if spec.get("size_by"):
        out.append(spec["size_by"]["column"])
    if spec.get("highlight"):
        out.append(spec["highlight"]["column"])
    if spec.get("label_column"):
        out.append(spec["label_column"])
    return list(dict.fromkeys(out))     # one column may serve two options; select it once


def _label_text(v: Any) -> Optional[str]:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return None
    text = _value_text(v).strip()
    if not text:
        return None
    return text if len(text) <= _LABEL_CHARS else text[:_LABEL_CHARS - 1] + "…"


def finish(gdf, spec: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict]]:
    """The descriptor fields for the frame AS WRITTEN (after any sampling), or an error.

    Sizes, highlight counts and labels are taken from what the map will receive, so the
    sentence the tool returns is about the drawn map, not the request.
    """
    fields: Dict[str, Any] = {}
    sb = spec.get("size_by")
    if sb:
        vals = gdf[sb["column"]].astype(float)
        finite = vals[vals.map(math.isfinite)]
        fields["size_by"] = {"column": sb["column"], "min": float(finite.min()),
                             "max": float(finite.max()), "range": list(sb["range"]),
                             "missing": int(len(vals) - len(finite))}
    if spec.get("point_size") is not None:
        fields["point_size"] = spec["point_size"]
    hl_mask = None
    hl = spec.get("highlight")
    if hl:
        hl_mask = highlight_mask(gdf, hl)
        fields["highlight"] = {**hl, "color": hl.get("color") or list(HIGHLIGHT_COLOUR),
                               "label": hl.get("label") or highlight_phrase(hl),
                               "count": int(hl_mask.sum())}
    col = spec.get("label_column")
    if col:
        total = len(gdf)
        if total <= MAX_LABELS:
            rows, scope = gdf, "all"
        elif hl_mask is not None and int(hl_mask.sum()) <= MAX_LABELS:
            rows, scope = gdf[hl_mask.values], "highlighted"
        else:
            return {}, _fail(f"label_column would label {total} features; the map labels at most "
                             f"{MAX_LABELS}",
                             "Add a `highlight` that picks out the features worth naming — labels "
                             f"then go on those only (≤ {MAX_LABELS}) — or map a filtered subset.")
        items = []
        for value, geom in zip(rows[col], rows.geometry):
            text = _label_text(value)
            if text is None or geom is None or geom.is_empty:
                continue
            pt = geom if geom.geom_type == "Point" else geom.representative_point()
            items.append({"position": [round(pt.x, 6), round(pt.y, 6)], "text": text})
        if not items:
            return {}, _fail(f"label_column {col!r} is empty for every feature it would label",
                             "Pick a column with values.")
        fields["labels"] = {"column": col, "scope": scope, "items": items}
    if spec.get("line_width") is not None:
        fields["line_width"] = spec["line_width"]
    if spec.get("line_color") is not None:
        fields["line_color"] = spec["line_color"]
    return fields, None


def highlight_phrase(hl: Dict[str, Any]) -> str:
    value = hl["value"]
    shown = ", ".join(_value_text(v) for v in value) if hl["op"] == "in" else _value_text(value)
    return f"{hl['column']} {_OP_TEXT[hl['op']]} {shown}"


def describe(fields: Dict[str, Any], *, mode: str, is_point: bool, count: int) -> str:
    """One sentence on what the map will actually draw, for the answer writer to repeat."""
    parts: List[str] = []
    if is_point and mode != "heatmap":
        sb = fields.get("size_by")
        if sb:
            lo, hi = sb["range"]
            text = (f"points sized by {sb['column']} ({sb['min']:g} to {sb['max']:g} drawn "
                    f"{lo:g}–{hi:g} px, square-root scale)")
            if sb.get("missing"):
                text += f"; {sb['missing']} with no {sb['column']} drawn at {lo:g} px"
            parts.append(text)
        elif fields.get("point_size") is not None:
            parts.append(f"all points one size ({fields['point_size']:g} px)")
        else:
            parts.append("all points one size (no size_column was set)")
    hl = fields.get("highlight")
    if hl:
        n = hl["count"]
        rule = highlight_phrase(hl)
        legend = "" if hl["label"] == rule else f" '{hl['label']}'"
        parts.append(f"{n} feature{'' if n == 1 else 's'} highlighted ({rule}), drawn larger, "
                     f"on top and in the highlight colour, with a legend entry{legend}")
    else:
        parts.append("no feature highlighted")
    lb = fields.get("labels")
    if lb:
        n = len(lb["items"])
        where = "" if lb["scope"] == "all" else " (highlighted only)"
        parts.append(f"{n} feature{'' if n == 1 else 's'}{where} labelled with {lb['column']}")
    else:
        parts.append("no labels")
    if fields.get("line_width") is not None:
        parts.append(f"lines/outlines {fields['line_width']:g} px")
    if fields.get("line_color") is not None:
        parts.append("lines/outlines in the line_color set")
    return f"{count} feature{'' if count == 1 else 's'} drawn as {mode}: " + "; ".join(parts) + "."


DRAWN_NOTE = ("`drawn` is what the map shows. Describe sizes, highlights and labels only as it "
              "states them: a size, highlight or label it does not list is not on the map.")
