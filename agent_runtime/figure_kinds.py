"""What a figure IS, read from how the code computed it (stage 54).

Until stage 54 the checks decided what a number was after the fact, from the model's declaration
or the answer's prose. The validators bench (stage 52) measured what that costs: every firing of
the count check and of the declared-unit check was a false alarm, and so were five of the
projected-CRS and all_nan findings. Each class is a number whose kind the check guessed:
- "239 patches", "1,000,557 pixels" and "2,344 residents" compared with frame row counts, though
  `len(list_of_geoms)`, `valid.sum()` over a raster and `top['population']` count no frame's rows;
- `np.nanmean(ndvi)` declared with the unit "NDVI", which the unit library reads as a count of
  NDVI, then failed for being fractional;
- `print(gdf.area)` in degrees on a run that reprojects and measures again, and `.area > 0`, a
  sign test, which degrees answer the same as metres;
- `old_operator`, an OpenStreetMap tag that is empty after a join that matched, in a frame whose
  code never reads it.

This module reads the recorded code (a static AST pass, so a replay needs nothing but the
archived events) and says, for a figure, which operation produced it and of what:

* ``len(F)``, ``F.shape[0]``, a sum over a boolean mask of ``F``'s column, ``value_counts`` and
  ``groupby().size()`` are counts of ``F``'s ROWS; ``len(a_list)`` counts items, ``mask.sum()``
  over an array counts elements (pixels), ``ndimage.label`` counts labelled features,
  ``nunique`` counts distinct values;
* ``.mean()``, ``np.nanmean``, ``.median()``, ``.std()``, ``np.percentile`` are statistics;
* ``.area``, ``.length`` and ``.distance()`` are measurements in the units of the receiver's CRS;
* ``F.loc[i, 'col']``, ``row['col']`` are a value of a column; a quotient of two figures is a
  ratio. Scaling by a constant keeps the kind (``area / 1e6`` is still an area).

Anything else is ``unknown``, and an unknown kind never raises a banner: the checks that need a
kind (agent_runtime/declared_outputs.py, ``reclassify`` here) apply only when it is known. A
model's declaration is a hint; when it contradicts the computation, the computation wins.

Name resolution is deliberately simple: the nearest assignment above the use, in any scope. A
name rebound in a loop or a function resolves to whatever is textually nearest, which can be
wrong; the effect of a wrong resolution is a kind that is unknown or too general, and both are
silent by construction.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

PASS, FAIL, UNKNOWN = "pass", "fail", "cannot_determine"
DECLARED = "IGUIDE_OUTPUTS"

MEASURES = ("area", "length", "distance")
STATS = ("mean", "median", "std", "quantile", "sum", "min", "max")


@dataclass(frozen=True)
class Kind:
    """What a figure is. ``kind`` is one of: count, mean, median, std, quantile, sum, min, max,
    area, length, distance, ratio, value, unknown."""
    kind: str = "unknown"
    counts: Optional[str] = None     # count: rows | elements | items | distinct | features
    of: Optional[str] = None         # the frame, array, list or column it is computed from
    within: Optional[str] = None     # count of rows: the frame those rows were drawn from
    column: Optional[str] = None
    measure: Optional[str] = None    # a statistic of a measurement: area | length | distance
    via: str = ""                    # the operation that decided it

    @property
    def known(self) -> bool:
        return self.kind != "unknown"

    @property
    def is_count(self) -> bool:
        return self.kind == "count"

    @property
    def is_rows(self) -> bool:
        return self.kind == "count" and self.counts == "rows"

    @property
    def measured(self) -> Optional[str]:
        """area | length | distance when the figure is (a statistic of) a CRS measurement."""
        return self.kind if self.kind in MEASURES else self.measure

    def describe(self) -> str:
        if self.kind == "count":
            what = {"rows": f"rows of `{self.of}`" if self.of else "rows",
                    "elements": f"array elements of `{self.of}`" if self.of else "array elements",
                    "items": f"items of `{self.of}`" if self.of else "items",
                    "distinct": "distinct values", "features": "labelled features"}
            return f"a count of {what.get(self.counts or '', 'things')} ({self.via})"
        if self.kind in MEASURES:
            return f"{'an' if self.kind == 'area' else 'a'} {self.kind} measured in the CRS's units ({self.via})"
        col = f" of `{self.column}`" if self.column else (f" of `{self.of}`" if self.of else "")
        if self.kind == "value":
            return f"a value{col} ({self.via})"
        if self.kind == "ratio":
            return f"a ratio ({self.via})"
        if self.kind == "unknown":
            return "unknown"
        return f"a {self.kind}{col} ({self.via})"

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, "")}


UNKNOWN_KIND = Kind()


# --------------------------------------------------------------------------- values

@dataclass
class _V:
    """What an expression evaluates to, as far as the kinds care."""
    typ: Optional[str] = None        # frame|series|row|array|list|dict|tuple|geometry|number|str|groupby
    frame: Optional[str] = None      # the frame whose rows this is (a frame, its column, its row)
    base: Optional[str] = None       # the frame a subset's rows were drawn from
    boolean: bool = False            # a boolean series/array (a mask)
    measure: Optional[str] = None    # a series/array of area|length|distance measurements
    column: Optional[str] = None
    counts: bool = False             # a series of per-group row counts
    kind: Optional[Kind] = None      # a scalar figure
    items: Optional[List["_V"]] = None   # a tuple's elements
    const: Optional[float] = None    # a numeric constant
    projected: Optional[bool] = None # a frame reprojected with to_crs(<projected>)
    roots: Set[str] = field(default_factory=set)   # names a frame derives from


_FRAME_FUNCS = {"read_file", "read_parquet", "read_feather", "read_postgis", "read_csv",
                "read_excel", "read_json", "read_table", "GeoDataFrame", "DataFrame",
                "json_normalize", "from_features", "from_file", "from_postgis"}
_JOIN_FUNCS = {"sjoin", "sjoin_nearest", "overlay", "merge", "concat"}
_ROW_PRESERVING = {"to_crs", "copy", "set_crs", "set_geometry", "rename", "reset_index",
                   "set_index", "assign", "astype", "sort_values", "sort_index", "fillna",
                   "rename_geometry", "infer_objects", "convert_dtypes", "rename_axis"}
_ROW_SUBSET = {"head", "tail", "dropna", "drop_duplicates", "query", "sample", "nlargest",
               "nsmallest", "clip", "drop", "explode", "dissolve", "cx", "filter", "where",
               "mask"}
_BOOL_METHODS = {"isna", "notna", "isnull", "notnull", "isin", "between", "within", "intersects",
                 "contains", "touches", "overlaps", "crosses", "disjoint", "covered_by",
                 "covers", "duplicated", "eq", "ne", "gt", "lt", "ge", "le", "is_empty",
                 "is_valid", "startswith", "endswith", "any", "all"}
_NUMPY_BOOL = {"isfinite", "isnan", "isin", "logical_and", "logical_or", "logical_not",
               "logical_xor", "isinf", "isclose", "geometry_mask"}
_PASSTHROUGH = {"float", "int", "round", "abs", "float64", "float32", "int64", "int32", "floor",
                "ceil", "around", "rint", "trunc", "asarray", "array", "squeeze", "item",
                "nan_to_num", "Decimal"}
_STAT_FUNCS = {"mean": "mean", "nanmean": "mean", "average": "mean", "fmean": "mean",
               "median": "median", "nanmedian": "median",
               "std": "std", "nanstd": "std", "var": "std", "nanvar": "std", "stdev": "std",
               "pstdev": "std", "percentile": "quantile", "nanpercentile": "quantile",
               "quantile": "quantile", "nanquantile": "quantile",
               "sum": "sum", "nansum": "sum", "fsum": "sum",
               "min": "min", "amin": "min", "nanmin": "min", "max": "max", "amax": "max",
               "nanmax": "max"}
_ELEMENT = {"item", "first", "last", "squeeze"}
# Attributes of a frame that are not columns.
_FRAME_ATTRS = {"crs", "columns", "total_bounds", "bounds", "dtypes", "empty", "ndim", "plot",
                "explore", "sindex", "unary_union", "union_all", "is_valid", "geom_type",
                "has_z", "cx", "T", "axes", "attrs", "flags", "style", "active_geometry_name"}
_METRIC_ATTRS = {"area": "area", "length": "length"}
_METRIC_CALLS = {"distance": "distance", "hausdorff_distance": "distance",
                 "frechet_distance": "distance"}


def _const_number(node: ast.AST) -> Optional[float]:
    try:
        v = ast.literal_eval(node)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def _str_const(node: Optional[ast.AST]) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


class Analysis:
    """The static facts of one script."""

    def __init__(self, code: str) -> None:
        self.code = code or ""
        self.ok = False
        self.tree: Optional[ast.AST] = None
        self.parent: Dict[ast.AST, ast.AST] = {}
        self.assigns: Dict[str, List[Tuple[int, ast.AST, Any]]] = {}
        self.imports: Dict[str, str] = {}
        self.lines = self.code.splitlines()
        try:
            self.tree = ast.parse(self.code)
        except (SyntaxError, ValueError):
            return
        self.ok = True
        for node in ast.walk(self.tree):
            for child in ast.iter_child_nodes(node):
                self.parent[child] = node
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    self.imports[(a.asname or a.name).split(".")[0]] = (
                        a.name if a.asname else a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                for a in node.names:
                    self.imports[a.asname or a.name] = f"{node.module}.{a.name}"
            elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    self._bind(t, node.lineno, node.value, None)
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                self.assigns.setdefault(node.target.id, []).append((node.lineno, node, "aug"))
            elif isinstance(node, (ast.For, ast.comprehension)):
                self._bind(node.target, getattr(node, "lineno", 0) or
                           getattr(node.iter, "lineno", 0), node.iter, "for")
        for v in self.assigns.values():
            v.sort(key=lambda r: r[0])
        self._memo: Dict[Tuple[int, int], _V] = {}

    def _bind(self, target: ast.AST, line: int, value: ast.AST, how: Any) -> None:
        if isinstance(target, ast.Name):
            self.assigns.setdefault(target.id, []).append((line, value, how))
        elif isinstance(target, (ast.Tuple, ast.List)):
            for i, elt in enumerate(target.elts):
                if isinstance(elt, ast.Name):
                    self.assigns.setdefault(elt.id, []).append(
                        (line, value, ("for", i) if how == "for" else ("unpack", i)))

    # ------------------------------------------------------------------ names

    def qualname(self, func: ast.AST) -> Tuple[Optional[str], Optional[ast.AST]]:
        """(dotted name, receiver). A call on a module (`np.nanmean`, `gpd.read_file`) has a
        dotted name and no receiver; a method call (`gdf.area.sum`) has its receiver."""
        parts: List[str] = []
        node = func
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            if node.id in self.imports or not parts:
                head = self.imports.get(node.id, node.id)
                return ".".join([head, *reversed(parts)]), None
        if isinstance(func, ast.Attribute):
            return func.attr, func.value
        return None, None

    def assignment(self, name: str, line: int) -> Optional[Tuple[int, ast.AST, Any]]:
        best = None
        for row in self.assigns.get(name, []):
            if row[0] < line or (row[0] == line and best is None and row[2] == "for"):
                best = row
        return best

    def stable(self, name: str) -> bool:
        """Whether *name* keeps the same rows once bound: one binding, or later bindings that
        only reproject, copy or rename it."""
        rows = self.assigns.get(name, [])
        if not rows:
            return False
        for line, value, how in rows[1:]:
            if how is not None:
                return False
            if not self._row_preserving_of(value, name):
                return False
        return True

    def _row_preserving_of(self, node: ast.AST, name: str) -> bool:
        while isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr not in _ROW_PRESERVING:
                return False
            node = node.func.value
        return isinstance(node, ast.Name) and node.id == name

    # ------------------------------------------------------------------ evaluation

    def value(self, node: ast.AST, line: Optional[int] = None, depth: int = 0) -> _V:
        if node is None or depth > 30:
            return _V()
        line = line if line is not None else getattr(node, "lineno", 10 ** 9)
        key = (id(node), line)
        if key in self._memo:
            return self._memo[key]
        self._memo[key] = _V()           # a cycle resolves to nothing
        try:
            v = self._value(node, line, depth)
        except Exception:  # noqa: BLE001 - a kind is a hint; never fail the caller
            v = _V()
        self._memo[key] = v
        return v

    def _value(self, node: ast.AST, line: int, depth: int) -> _V:
        ev = lambda n: self.value(n, getattr(n, "lineno", line), depth + 1)  # noqa: E731
        c = _const_number(node) if isinstance(node, (ast.Constant, ast.UnaryOp, ast.BinOp)) else None
        if c is not None:
            return _V(typ="number", const=c)
        if isinstance(node, ast.Constant):
            return _V(typ="str" if isinstance(node.value, str) else None)
        if isinstance(node, ast.JoinedStr):
            return _V(typ="str")
        if isinstance(node, ast.Name):
            return self._name(node.id, line, depth)
        if isinstance(node, (ast.List, ast.ListComp, ast.Set, ast.SetComp, ast.GeneratorExp)):
            return _V(typ="list")
        if isinstance(node, (ast.Dict, ast.DictComp)):
            return _V(typ="dict")
        if isinstance(node, ast.Tuple):
            return _V(typ="tuple", items=[ev(e) for e in node.elts])
        if isinstance(node, ast.Compare):
            return self._compare(node, ev)
        if isinstance(node, ast.UnaryOp):
            inner = ev(node.operand)
            if isinstance(node.op, (ast.Invert, ast.Not)) and inner.boolean:
                return _V(typ=inner.typ, frame=inner.frame, base=inner.base, boolean=True)
            if isinstance(node.op, (ast.USub, ast.UAdd)):
                return inner
            return _V()
        if isinstance(node, ast.BoolOp):
            return _V()
        if isinstance(node, ast.BinOp):
            return self._binop(node, ev)
        if isinstance(node, ast.IfExp):
            a, b = ev(node.body), ev(node.orelse)
            if a.kind and b.kind and a.kind == b.kind:
                return a
            return _V()
        if isinstance(node, ast.Attribute):
            return self._attribute(node, ev)
        if isinstance(node, ast.Subscript):
            return self._subscript(node, ev)
        if isinstance(node, ast.Call):
            return self._call(node, ev, line)
        return _V()

    def _name(self, name: str, line: int, depth: int) -> _V:
        row = self.assignment(name, line)
        if row is None:
            return _V()
        at, value, how = row
        if how == "aug":
            return _V()
        v = self.value(value, at, depth + 1)
        if isinstance(how, tuple):
            idx = how[1]
            if how[0] == "unpack":
                v = (v.items or [])[idx] if v.items and idx < len(v.items) else _V()
            elif how[0] == "for":
                if v.typ == "iterrows" and idx == 1:
                    return _V(typ="row", frame=v.frame)
                return _V()
        elif how == "for":
            return _V()
        if v.typ == "frame":
            v = replace(v, frame=name, base=v.base or name, roots=set(v.roots) | {name})
        elif v.typ in ("list", "array", "dict"):
            v = replace(v, roots=set(v.roots) | {name})
        return v

    def _compare(self, node: ast.Compare, ev) -> _V:
        sides = [ev(node.left), *(ev(c) for c in node.comparators)]
        for s in sides:
            if s.typ in ("series", "array") or s.measure:
                return _V(typ=s.typ or "series", frame=s.frame, base=s.base, boolean=True)
        return _V(typ="bool")

    def _binop(self, node: ast.BinOp, ev) -> _V:
        a, b = ev(node.left), ev(node.right)
        if isinstance(node.op, (ast.BitAnd, ast.BitOr, ast.BitXor)) and (a.boolean or b.boolean):
            src = a if a.boolean else b
            return _V(typ=src.typ, frame=src.frame, base=src.base, boolean=True)
        # A series or array of measurements scaled stays a series of measurements.
        for x, y in ((a, b), (b, a)):
            if x.measure and y.const is not None and isinstance(node.op, (ast.Mult, ast.Div)):
                return _V(typ=x.typ, frame=x.frame, base=x.base, measure=x.measure)
        ka, kb = a.kind, b.kind
        if ka is None and a.measure and a.typ in (None, "number"):
            ka = Kind(a.measure, via=f".{a.measure}")
        if kb is None and b.measure and b.typ in (None, "number"):
            kb = Kind(b.measure, via=f".{b.measure}")
        for k, other in ((ka, b), (kb, a)):
            if k is not None and other.const is not None:
                if isinstance(node.op, (ast.Mult, ast.Div, ast.FloorDiv)):
                    if k.kind == "count":
                        # A count scaled is no longer a count: per thousand, a percentage.
                        return _V(typ="number", kind=Kind("ratio", via="count scaled"))
                    return _V(typ="number", kind=k)          # a unit conversion
                if isinstance(node.op, (ast.Add, ast.Sub)):
                    return _V(typ="number", kind=k)
                return _V()
        if ka is not None and kb is not None:
            if isinstance(node.op, (ast.Div, ast.FloorDiv)):
                return _V(typ="number", kind=Kind("ratio", via="quotient of two figures"))
            if isinstance(node.op, (ast.Add, ast.Sub)) and ka.kind == kb.kind:
                return _V(typ="number", kind=Kind(ka.kind, counts=ka.counts, measure=ka.measure,
                                                  via=f"{ka.via} {'+' if isinstance(node.op, ast.Add) else '-'} {kb.via}"))
        if a.typ in ("array", "series") or b.typ in ("array", "series"):
            src = a if a.typ in ("array", "series") else b
            return _V(typ=src.typ, frame=src.frame, base=src.base)
        return _V()

    def _attribute(self, node: ast.Attribute, ev) -> _V:
        q, recv = self.qualname(node)
        if recv is None:
            return _V()
        r = ev(recv)
        attr = node.attr
        if attr in _METRIC_ATTRS:
            m = _METRIC_ATTRS[attr]
            if r.typ in ("frame", "series", "geoseries"):
                return _V(typ="series", frame=r.frame, base=r.base, measure=m)
            if r.typ == "geometry":
                return _V(typ="number", kind=Kind(m, via=f".{attr}"))
            return _V(measure=m)                     # a scalar or a series; either way a measure
        if attr in ("geometry", "geom", "centroid", "boundary", "exterior", "envelope",
                    "convex_hull", "representative_point") and r.typ in ("frame", "series"):
            return _V(typ="series", frame=r.frame, base=r.base)
        if attr in ("values", "array") and r.typ == "series":
            return _V(typ="array", frame=r.frame, base=r.base, boolean=r.boolean,
                      measure=r.measure, column=r.column, counts=r.counts)
        if attr == "index" and r.typ in ("frame", "series"):
            return _V(typ="series", frame=r.frame, base=r.base)
        if attr == "shape":
            return _V(typ="shape", frame=r.frame if r.typ == "frame" else None,
                      base=r.base, column=r.typ)
        if attr == "size":
            if r.typ in ("series",) and r.frame:
                return _V(typ="number", kind=Kind("count", counts="rows", of=r.frame,
                                                  within=r.base, via=".size"))
            if r.typ == "array":
                return _V(typ="number", kind=Kind("count", counts="elements",
                                                  of=_first(r.roots), via=".size"))
            return _V()
        if attr in ("iloc", "loc", "at", "iat"):
            return _V(typ=f"{attr}:{r.typ}", frame=r.frame, base=r.base, boolean=r.boolean,
                      measure=r.measure, column=r.column, counts=r.counts)
        if attr == "str" and r.typ == "series":
            return _V(typ="strseries", frame=r.frame, base=r.base)
        if attr == "x" or attr == "y":
            if r.typ == "series":
                return _V(typ="series", frame=r.frame, base=r.base)
            return _V()
        if attr == "T" and r.typ == "array":
            return r
        if r.typ == "frame" and attr not in _FRAME_ATTRS | _ROW_PRESERVING | _ROW_SUBSET \
                and not attr.startswith("_"):
            # `F.col`: a column, read as an attribute.
            return _V(typ="series", frame=r.frame, base=r.base, column=attr)
        if r.typ == "row":
            return _V(typ="number", kind=Kind("value", column=attr, of=r.frame,
                                              via=f"row.{attr}"))
        return _V()

    def _subscript(self, node: ast.Subscript, ev) -> _V:
        r = ev(node.value)
        sl = node.slice
        key = _str_const(sl)
        if r.typ == "shape":
            idx = _const_number(sl)
            if idx == 0:
                if r.column == "frame":
                    return _V(typ="number", kind=Kind("count", counts="rows", of=r.frame,
                                                      within=r.base, via=".shape[0]"))
                if r.column in ("array", "series"):
                    return _V(typ="number", kind=Kind("count", counts="elements", via=".shape[0]"))
                return _V(typ="number", kind=Kind("count", via=".shape[0]"))
            return _V()
        if r.typ == "frame":
            if key is not None:
                return _V(typ="series", frame=r.frame, base=r.base, column=key)
            if isinstance(sl, (ast.List,)):
                return _V(typ="frame", base=r.base, roots=set(r.roots))
            m = ev(sl)
            if m.boolean or isinstance(sl, ast.Compare):
                return _V(typ="frame", base=r.base or r.frame, roots=set(r.roots))
            if isinstance(sl, ast.Slice):
                return _V(typ="frame", base=r.base or r.frame, roots=set(r.roots))
            return _V()
        if r.typ and r.typ.startswith(("iloc:", "loc:", "at:", "iat:")):
            acc, inner = r.typ.split(":", 1)
            col = None
            row_sel = sl
            if isinstance(sl, ast.Tuple) and len(sl.elts) == 2:
                row_sel, col = sl.elts[0], _str_const(sl.elts[1])
            if inner == "frame":
                if col is not None:
                    if isinstance(row_sel, ast.Slice) or ev(row_sel).boolean:
                        return _V(typ="series", frame=r.frame, base=r.base, column=col)
                    return _V(typ="number", kind=Kind("value", column=col, of=r.frame,
                                                      via=f".{acc}[row, '{col}']"))
                if isinstance(row_sel, ast.Slice) or ev(row_sel).boolean:
                    return _V(typ="frame", base=r.base or r.frame)
                if acc in ("iloc", "loc"):
                    return _V(typ="row", frame=r.frame)
                return _V()
            if inner in ("series", "array"):
                if isinstance(row_sel, ast.Slice) or ev(row_sel).boolean:
                    return _V(typ=inner, frame=r.frame, base=r.base, measure=r.measure,
                              column=r.column, counts=r.counts)
                return self._element(r, f".{acc}[]")
            return _V()
        if r.typ == "row":
            if key is not None:
                return _V(typ="number", kind=Kind("value", column=key, of=r.frame,
                                                  via=f"row['{key}']"))
            return _V()
        if r.typ in ("series", "array"):
            m = ev(sl)
            if m.boolean or isinstance(sl, (ast.Compare, ast.Slice)):
                return _V(typ=r.typ, frame=r.frame, base=r.base, measure=r.measure,
                          column=r.column, counts=r.counts, roots=set(r.roots))
            if r.typ == "array" and isinstance(sl, ast.Tuple):
                if any(isinstance(e, ast.Slice) for e in sl.elts):
                    return _V(typ="array", roots=set(r.roots), measure=r.measure)
            if key is not None and r.typ == "array":
                return _V(typ="array", roots=set(r.roots))     # an npz member
            return self._element(r, "[]")
        if r.typ == "tuple" and r.items:
            idx = _const_number(sl)
            if idx is not None and 0 <= int(idx) < len(r.items):
                return r.items[int(idx)]
        if r.typ == "dict" and key is not None:
            return _V()
        if r.typ == "groupby":
            return _V(typ="groupby", frame=r.frame, base=r.base, column=key)
        return _V()

    def _element(self, r: _V, via: str) -> _V:
        """One element of a series or array."""
        if r.counts:
            return _V(typ="number", kind=Kind("count", counts="rows", of=r.frame,
                                              within=r.base or r.frame, column=r.column,
                                              via=f"per-group count{via}"))
        if r.measure:
            return _V(typ="number", kind=Kind(r.measure, via=f".{r.measure}{via}"))
        if r.column and r.frame:
            return _V(typ="number", kind=Kind("value", column=r.column, of=r.frame,
                                              via=f"['{r.column}']{via}"))
        return _V()

    def _stat(self, r: _V, stat: str, via: str) -> _V:
        """A statistic over a series, array or list."""
        of = r.frame if r.typ in ("series", "frame") else _first(r.roots)
        if stat == "sum":
            if r.boolean:
                if r.frame:
                    return _V(typ="number", kind=Kind("count", counts="rows", of=r.frame,
                                                      within=r.base or r.frame,
                                                      via=f"{via} over a mask"))
                return _V(typ="number", kind=Kind("count", counts="elements", of=of,
                                                  via=f"{via} over a mask"))
            if r.counts:
                return _V(typ="number", kind=Kind("count", counts="rows", of=r.frame,
                                                  within=r.base or r.frame, via=f"{via} of counts"))
            if r.measure:
                return _V(typ="number", kind=Kind(r.measure, via=f".{r.measure}{via}"))
            if r.typ in ("series", "array", "list", "frame"):
                return _V(typ="number", kind=Kind("sum", column=r.column, of=of, via=via))
            return _V()
        if r.typ is None and not r.measure and via.startswith(("np.", "statistics.")) \
                and stat in ("mean", "median", "std", "quantile"):
            # numpy's and statistics' own statistic of something untraced is that statistic; a
            # sum is not, since a sum over a mask is a count.
            return _V(typ="number", kind=Kind(stat, of=_first(r.roots), via=via))
        if stat == "mean" and r.boolean:
            return _V(typ="number", kind=Kind("ratio", of=of, via=f"{via} of a mask"))
        if stat in ("min", "max"):
            if r.counts:
                return _V(typ="number", kind=Kind("count", counts="rows", of=r.frame,
                                                  within=r.base or r.frame, via=f"{via} of counts"))
            if r.measure:
                return _V(typ="number", kind=Kind(r.measure, via=f".{r.measure}{via}"))
            if r.typ in ("series", "array", "list"):
                return _V(typ="number", kind=Kind(stat, column=r.column, of=of, via=via))
            return _V()
        if r.typ in ("series", "array", "list", "frame") or r.measure:
            return _V(typ="number", kind=Kind(stat, column=r.column, of=of, measure=r.measure,
                                              via=via))
        return _V()

    def _call(self, node: ast.Call, ev, line: int) -> _V:
        q, recv = self.qualname(node.func)
        short = (q or "").rsplit(".", 1)[-1]
        args = node.args
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        if recv is None:
            # A function: a builtin, or one reached through an imported module.
            if short == "len" and args:
                return _V(typ="number", kind=self._len(ev(args[0]), args[0]))
            if short in ("sum",) and q == "sum" and args:
                if isinstance(args[0], ast.GeneratorExp):
                    return _V(typ="number", kind=Kind("count", counts="items",
                                                      via="sum(1 for ...)")
                              if _const_number(args[0].elt) == 1 else None)
                return self._stat(ev(args[0]), "sum", "sum()")
            if short in ("min", "max") and q in ("min", "max") and args:
                if len(args) == 1:
                    return self._stat(ev(args[0]), short, f"{short}()")
                ks = [ev(a).kind for a in args]
                if all(k is not None for k in ks) and len({k.kind for k in ks}) == 1:
                    return _V(typ="number", kind=ks[0])
                return _V()
            if short in _PASSTHROUGH and args and (q in _PASSTHROUGH or q.startswith(("numpy.",
                                                                                      "math."))):
                return ev(args[0])
            if q and q.startswith("numpy."):
                if short in _STAT_FUNCS and args:
                    return self._stat(ev(args[0]), _STAT_FUNCS[short], f"np.{short}")
                if short == "count_nonzero" and args:
                    inner = ev(args[0])
                    if inner.frame and inner.typ == "series":
                        return _V(typ="number", kind=Kind("count", counts="rows",
                                                          of=inner.frame,
                                                          within=inner.base or inner.frame,
                                                          via="np.count_nonzero"))
                    return _V(typ="number", kind=Kind("count", counts="elements",
                                                      of=_first(inner.roots),
                                                      via="np.count_nonzero"))
                if short in _NUMPY_BOOL:
                    inner = ev(args[0]) if args else _V()
                    return _V(typ="array" if inner.typ != "series" else "series",
                              frame=inner.frame if inner.typ == "series" else None,
                              base=inner.base, boolean=True, roots=set(inner.roots))
                if short == "unique":
                    return _V(typ="array", column="__distinct__")
                if short == "where" and len(args) == 3:
                    inner = ev(args[1])
                    return _V(typ="array", roots=set(inner.roots), measure=inner.measure)
                return _V(typ="array")
            if q and short in _NUMPY_BOOL and "rasterio" in q:
                return _V(typ="array", boolean=True)
            if q and short == "label" and ("ndimage" in q or "measure" in q or "skimage" in q):
                if "measure" in q and not _truthy(kw.get("return_num")):
                    return _V(typ="array")
                return _V(typ="tuple", items=[_V(typ="array"), _V(
                    typ="number", kind=Kind("count", counts="features",
                                            via=f"{short}() feature count"))])
            if q and short in _FRAME_FUNCS:
                return _V(typ="frame")
            if q and short in _JOIN_FUNCS:
                roots: Set[str] = set()
                for a in args[:2]:
                    roots |= ev(a).roots
                return _V(typ="frame", roots=roots)
            if q and short in ("Series",):
                return _V(typ="series")
            if q == "list" and args:
                inner = ev(args[0])
                return _V(typ="list", roots=set(inner.roots))
            if q in ("set", "frozenset"):
                return _V(typ="list", column="__distinct__")
            if q in ("dict",):
                return _V(typ="dict")
            if q in ("statistics.mean", "statistics.median", "statistics.stdev",
                     "statistics.fmean", "statistics.pstdev", "math.fsum") and args:
                return self._stat(ev(args[0]), _STAT_FUNCS.get(short, short), q)
            return _V()

        # A method call on a value.
        r = ev(recv)
        m = short
        if r.typ is None and m in _FRAME_FUNCS:
            return _V(typ="frame")                    # `pd.read_csv` with `pd` not imported here
        if r.typ == "frame":
            if m in _ROW_PRESERVING:
                proj = r.projected
                if m == "to_crs":
                    proj = self._projected(node)
                return _V(typ="frame", frame=r.frame, base=r.base, roots=set(r.roots),
                          projected=proj)
            if m in _ROW_SUBSET:
                return _V(typ="frame", base=r.base or r.frame, roots=set(r.roots),
                          projected=r.projected)
            if m in ("sjoin", "sjoin_nearest", "merge", "overlay", "join"):
                roots = set(r.roots)
                if args:
                    roots |= ev(args[0]).roots
                return _V(typ="frame", roots=roots)
            if m == "groupby":
                return _V(typ="groupby", frame=r.frame, base=r.base)
            if m == "iterrows":
                return _V(typ="iterrows", frame=r.frame)
            if m == "count":
                return _V(typ="series", frame=r.frame, base=r.base, counts=True)
            if m in ("to_numpy",):
                return _V(typ="array")
            if m in _STAT_FUNCS:
                return _V(typ="series")
            if m == "distance":
                return _V(typ="series", frame=r.frame, base=r.base, measure="distance")
            if m in _BOOL_METHODS:
                return _V(typ="series", frame=r.frame, base=r.base, boolean=True)
            if m == "estimate_utm_crs":
                return _V(typ="crs", projected=True)
            return _V()
        if r.typ == "groupby":
            if m in ("size", "count"):
                return _V(typ="series", frame=r.frame, base=r.base, counts=True, column=r.column)
            if m in ("agg", "aggregate", "sum", "mean", "median", "max", "min", "first",
                     "nunique", "std", "apply"):
                return _V(typ="series" if r.column else "frame")
            return _V()
        if r.typ in ("series", "array", "list", "strseries") or r.measure:
            typ = r.typ or "series"
            if m in _BOOL_METHODS or (r.typ == "strseries" and m in ("contains", "startswith",
                                                                     "endswith", "match")):
                if m in ("any", "all"):
                    return _V(typ="bool")
                return _V(typ="series" if typ == "strseries" else typ, frame=r.frame,
                          base=r.base, boolean=True)
            if m in _ROW_PRESERVING or m in ("dropna", "fillna", "clip", "abs", "round",
                                             "sort_values", "to_numpy", "tolist", "copy"):
                out_typ = "array" if m == "to_numpy" else ("list" if m == "tolist" else typ)
                return _V(typ=out_typ, frame=r.frame, base=r.base, boolean=r.boolean,
                          measure=r.measure, column=r.column, counts=r.counts,
                          roots=set(r.roots))
            if m in ("head", "tail", "nlargest", "nsmallest"):
                return _V(typ=typ, frame=r.frame, base=r.base, measure=r.measure,
                          column=r.column, counts=r.counts)
            if m in _ELEMENT:
                return self._element(r, f".{m}()")
            if m == "count":
                if r.typ == "series" and r.frame:
                    return _V(typ="number", kind=Kind("count", counts="rows", of=r.frame,
                                                      within=r.base or r.frame, column=r.column,
                                                      via=".count()"))
                return _V()
            if m == "nunique":
                return _V(typ="number", kind=Kind("count", counts="distinct", column=r.column,
                                                  of=r.frame, via=".nunique()"))
            if m == "unique":
                return _V(typ="array", column="__distinct__")
            if m == "value_counts":
                return _V(typ="series", frame=r.frame, base=r.base, counts=True, column=r.column)
            if m in _STAT_FUNCS:
                return self._stat(r, _STAT_FUNCS[m], f".{m}()")
            if m in ("idxmax", "idxmin", "argmax", "argmin"):
                return _V(typ="index")
            if m == "distance":
                return _V(typ="series", frame=r.frame, base=r.base, measure="distance")
            if m in ("astype",):
                return _V(typ=typ, frame=r.frame, base=r.base, measure=r.measure,
                          column=r.column, counts=r.counts, boolean=r.boolean)
            if m == "map" or m == "apply":
                return _V(typ=typ, frame=r.frame, base=r.base)
            if m == "reset_index":
                return _V(typ="frame") if r.counts or r.column else _V()
            if m == "sum" or m == "mean":
                return self._stat(r, m, f".{m}()")
            return _V()
        if r.typ == "geometry" and m == "distance":
            return _V(typ="number", kind=Kind("distance", via=".distance()"))
        if m in _METRIC_CALLS:
            return _V(measure=_METRIC_CALLS[m])
        if m == "read" and r.typ != "frame":
            return _V(typ="array")
        if m in ("item",) and r.kind is not None:
            return _V(typ="number", kind=r.kind)
        if r.typ == "number" and m in ("round", "item"):
            return r
        if r.typ == "dict" and m == "get":
            return _V()
        return _V()

    def _len(self, v: _V, arg: ast.AST) -> Kind:
        name = arg.id if isinstance(arg, ast.Name) else None
        if v.typ == "frame":
            return Kind("count", counts="rows", of=v.frame or name, within=v.base or v.frame,
                        via="len()")
        if v.typ == "series":
            if v.counts:
                return Kind("count", counts="distinct", column=v.column, of=v.frame,
                            via="len(value_counts)")
            return Kind("count", counts="rows", of=v.frame, within=v.base or v.frame,
                        via="len()")
        if v.typ in ("array", "list", "dict", "tuple"):
            if v.column == "__distinct__":
                return Kind("count", counts="distinct", via="len(unique)")
            return Kind("count", counts="items" if v.typ != "array" else "elements",
                        of=name or _first(v.roots), via="len()")
        # Whatever it is the length of, a length is a count; of what is not known, so it is not
        # compared with any frame.
        return Kind("count", of=name, via="len()")

    def _projected(self, call: ast.Call) -> Optional[bool]:
        """Whether a ``to_crs`` call reprojects to a projected CRS, from its literal argument."""
        arg = call.args[0] if call.args else None
        for k in call.keywords:
            if k.arg in ("crs", "epsg"):
                arg = k.value
                if k.arg == "epsg":
                    n = _const_number(arg)
                    return _crs_projected(f"EPSG:{int(n)}") if n is not None else None
        if arg is None:
            return None
        if isinstance(arg, ast.Name):
            row = self.assignment(arg.id, call.lineno)
            if row is None:
                return None
            arg = row[1]
        if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute) \
                and arg.func.attr == "estimate_utm_crs":
            return True
        n = _const_number(arg)
        if n is not None:
            return _crs_projected(f"EPSG:{int(n)}")
        s = _str_const(arg)
        return _crs_projected(s) if s else None

    # ------------------------------------------------------------------ outputs

    def declared(self) -> Dict[str, Tuple[Kind, int, str]]:
        """{name: (kind, line, source)} for every IGUIDE_OUTPUTS entry the script assigns."""
        out: Dict[str, Tuple[Kind, int, str]] = {}
        if not self.ok:
            return out

        def entry(key: Optional[str], spec: ast.AST, line: int) -> None:
            if key is None:
                return
            value = spec
            if isinstance(spec, ast.Dict):
                for k, v in zip(spec.keys, spec.values):
                    if _str_const(k) == "value":
                        value = v
                        break
                else:
                    return
            elif isinstance(spec, ast.Call) and getattr(spec.func, "id", None) == "dict":
                value = next((k.value for k in spec.keywords if k.arg == "value"), spec)
            v = self.value(value, line)
            kind = v.kind or (Kind(v.measure, via=f".{v.measure}") if v.measure
                              and v.typ in (None, "number") else UNKNOWN_KIND)
            out[key] = (kind, line, ast.get_source_segment(self.code, value) or "")

        def from_dict(d: ast.AST, line: int) -> None:
            if isinstance(d, ast.Dict):
                for k, v in zip(d.keys, d.values):
                    entry(_str_const(k), v, line)
            elif isinstance(d, ast.Call) and getattr(d.func, "id", None) == "dict":
                for k in d.keywords:
                    entry(k.arg, k.value, line)

        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == DECLARED:
                        from_dict(node.value, node.lineno)
                    elif isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) \
                            and t.value.id == DECLARED:
                        entry(_str_const(t.slice), node.value, node.lineno)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "update" and isinstance(node.func.value, ast.Name) \
                    and node.func.value.id == DECLARED:
                for a in node.args:
                    from_dict(a, node.lineno)
                for k in node.keywords:
                    entry(k.arg, k.value, node.lineno)
        return out

    def printed(self) -> List[Dict[str, Any]]:
        """Every figure a ``print`` call formats: its line, the literal text before it, and its
        kind. For the stages that link a printed number to the answer (plan step 5)."""
        out: List[Dict[str, Any]] = []
        if not self.ok:
            return out
        for node in ast.walk(self.tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "print"):
                continue
            label = ""
            for a in node.args:
                parts = a.values if isinstance(a, ast.JoinedStr) else [a]
                for p in parts:
                    if isinstance(p, ast.Constant) and isinstance(p.value, str):
                        label += p.value
                        continue
                    expr = p.value if isinstance(p, ast.FormattedValue) else p
                    v = self.value(expr, node.lineno)
                    kind = v.kind or (Kind(v.measure, via=f".{v.measure}") if v.measure else None)
                    if kind is not None or v.typ == "number":
                        out.append({"line": node.lineno, "label": label.strip()[-60:],
                                    "kind": (kind or UNKNOWN_KIND).to_dict()})
                    label = ""
        out.sort(key=lambda r: r["line"])
        return out

    def columns_read(self) -> Set[str]:
        """Every column name the code could read: each string constant (but a dict literal's
        keys, which build data) and each attribute name. An over-approximation, so a column it
        misses is one the code cannot name."""
        names: Set[str] = set()
        if not self.ok:
            return names
        keys = {id(k) for d in ast.walk(self.tree) if isinstance(d, ast.Dict)
                for k in d.keys if k is not None}
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if id(node) not in keys:      # a dict literal's key builds data, it reads none
                    names.add(node.value)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.keyword) and node.arg:
                names.add(node.arg)
        return names

    def join_kind(self, name: str) -> Optional[str]:
        """'inner' when every binding of *name* comes from an inner join (sjoin's and merge's
        default, overlay's intersection) or a subset of one; another `how` by name; None when
        *name* is not traceably a join result."""
        rows = self.assigns.get(name, [])
        if not rows:
            return None
        hows = set()
        for _line, value, how in rows:
            if how is not None:
                return None
            node = value
            while isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr in _ROW_PRESERVING | _ROW_SUBSET:
                node = node.func.value
            while isinstance(node, ast.Subscript):
                node = node.value
            if isinstance(node, ast.Name) and node.id == name:
                continue                              # a rebinding of itself
            if not isinstance(node, ast.Call):
                return None
            q, _recv = self.qualname(node.func)
            short = (q or "").rsplit(".", 1)[-1]
            if short not in ("sjoin", "sjoin_nearest", "merge", "overlay"):
                return None
            kw = {k.arg: k.value for k in node.keywords if k.arg}
            how_ = _str_const(kw.get("how")) if "how" in kw else None
            if short == "overlay":
                hows.add("inner" if how_ in (None, "intersection") else how_ or "?")
            else:
                hows.add(how_ or "inner")
        if not hows:
            return None
        return "inner" if hows == {"inner"} else ",".join(sorted(hows))

    # ------------------------------------------------------------------ measurement uses

    def op_nodes(self, code_line: str, op: str) -> List[ast.AST]:
        """The AST nodes of metric operation *op* on the source line the tracker recorded."""
        if not self.ok or not code_line:
            return []
        want = code_line.strip()[:160]
        lines = {i + 1 for i, text in enumerate(self.lines) if text.strip()[:160] == want}
        if not lines:
            return []
        out = []
        for node in ast.walk(self.tree):
            start = getattr(node, "lineno", None)
            if start is None or not any(start <= ln <= (node.end_lineno or start) for ln in lines):
                continue
            if op in _METRIC_ATTRS and isinstance(node, ast.Attribute) and node.attr == op:
                par = self.parent.get(node)
                if not (isinstance(par, ast.Call) and par.func is node):
                    out.append(node)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == op:
                out.append(node)
        return out

    def sinks(self, node: ast.AST, depth: int = 0, seen: Optional[Set[int]] = None) -> Set[str]:
        """Where a value goes. A sink is one of: print, declared, stored (a column or a
        container), returned, threshold (compared with a non-zero number), sign_test (compared
        with zero), test (truthiness), unused, unit_free (only its count is used), selector (it
        picks rows), unknown."""
        seen = seen if seen is not None else set()
        if depth > 12 or id(node) in seen:
            return {"unknown"}
        seen.add(id(node))
        cur = node
        for _ in range(40):
            p = self.parent.get(cur)
            if p is None:
                return {"unknown"}
            if isinstance(p, ast.Attribute):
                gp = self.parent.get(p)
                cur = gp if isinstance(gp, ast.Call) and gp.func is p else p
                continue
            if isinstance(p, ast.Subscript):
                if p.value is cur:
                    cur = p
                    continue
                return {"selector"}
            if isinstance(p, ast.Call):
                if p.func is cur:
                    cur = p
                    continue
                q, _recv = self.qualname(p.func)
                short = (q or "").rsplit(".", 1)[-1]
                if q == "print":
                    return {"print"}
                if q == "len":
                    return {"unit_free"}
                if q in ("min", "max", "sum", "sorted", "list", "tuple", "abs", "round", "float",
                         "int") or (q or "").startswith(("numpy.", "math.")) \
                        or short in _PASSTHROUGH:
                    cur = p
                    continue
                return {"unknown"}
            if isinstance(p, ast.keyword):
                return {"unknown"}
            if isinstance(p, ast.Compare):
                others = [x for x in [p.left, *p.comparators] if x is not cur]
                if others and all(_const_number(x) == 0 for x in others):
                    return {"sign_test"}
                return {"threshold"}
            if isinstance(p, (ast.BinOp, ast.UnaryOp, ast.FormattedValue, ast.JoinedStr,
                              ast.Tuple, ast.List, ast.Starred)):
                cur = p
                continue
            if isinstance(p, ast.BoolOp):
                return {"test"}
            if isinstance(p, (ast.If, ast.While, ast.Assert)) and getattr(p, "test", None) is cur:
                return {"test"}
            if isinstance(p, ast.IfExp):
                if p.test is cur:
                    return {"test"}
                cur = p
                continue
            if isinstance(p, ast.Dict):
                if self._in_declared(p):
                    return {"declared"}
                return {"stored"}
            if isinstance(p, (ast.Assign, ast.AnnAssign)):
                targets = p.targets if isinstance(p, ast.Assign) else [p.target]
                out: Set[str] = set()
                for t in targets:
                    if isinstance(t, ast.Name):
                        if t.id == DECLARED:
                            out.add("declared")
                            continue
                        loads = self._loads_after(t.id, p)
                        if not loads:
                            out.add("unused")
                        for ld in loads:
                            out |= self.sinks(ld, depth + 1, seen)
                    elif isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) \
                            and t.value.id == DECLARED:
                        out.add("declared")
                    else:
                        out.add("stored")
                return out or {"unused"}
            if isinstance(p, ast.Expr):
                return {"unused"}
            if isinstance(p, ast.Return):
                return {"returned"}
            return {"unknown"}
        return {"unknown"}

    def _in_declared(self, d: ast.Dict) -> bool:
        node: Optional[ast.AST] = d
        for _ in range(4):
            node = self.parent.get(node)
            if isinstance(node, ast.Assign):
                return any((isinstance(t, ast.Name) and t.id == DECLARED) or
                           (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                            and t.value.id == DECLARED) for t in node.targets)
            if node is None:
                return False
        return False

    def _loads_after(self, name: str, stmt: ast.AST) -> List[ast.Name]:
        """The reads of *name* after *stmt*, up to its next binding."""
        start = stmt.end_lineno or stmt.lineno
        later = [row[0] for row in self.assigns.get(name, []) if row[0] > start]
        stop = min(later) if later else 10 ** 9
        return [n for n in ast.walk(self.tree)
                if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)
                and start < n.lineno <= stop]

    def receiver_roots(self, op_node: ast.AST) -> Tuple[Set[str], Optional[bool]]:
        """The frame names an operation's receiver derives from, and whether it was reprojected
        to a projected CRS on the way."""
        recv = op_node.value if isinstance(op_node, ast.Attribute) else (
            op_node.func.value if isinstance(op_node, ast.Call)
            and isinstance(op_node.func, ast.Attribute) else None)
        return self._chain_roots(recv, getattr(op_node, "lineno", 10 ** 9), 0)

    def _chain_roots(self, node: Optional[ast.AST], line: int, depth: int) -> Tuple[Set[str], Optional[bool]]:
        projected: Optional[bool] = None
        while node is not None and depth < 12:
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if self.qualname(node.func)[1] is None:
                    return set(), projected          # a module's function (`gpd.read_file`)
                if node.func.attr == "to_crs" and projected is None:
                    projected = self._projected(node)
                node = node.func.value
            elif isinstance(node, (ast.Attribute,)):
                node = node.value
            elif isinstance(node, ast.Subscript):
                node = node.value
            elif isinstance(node, ast.Call):
                q, _r = self.qualname(node.func)
                if q and q.rsplit(".", 1)[-1] in ("GeoSeries", "GeoDataFrame") and node.args:
                    node = node.args[0]
                    continue
                return set(), projected
            elif isinstance(node, ast.Name):
                if node.id in self.imports:
                    return set(), projected
                roots = {node.id}
                row = self.assignment(node.id, line)
                if row is not None and row[2] is None:
                    sub, proj = self._chain_roots(row[1], row[0], depth + 1)
                    roots |= sub
                    if projected is None:
                        projected = proj
                return roots, projected
            else:
                return set(), projected
        return set(), projected

    def remeasured_projected(self, op_node: ast.AST, op: str,
                             projected_lines: Iterable[str] = ()) -> Optional[ast.AST]:
        """Another *op* in the script, on a receiver derived from the same frame and reprojected
        to a projected CRS: the run measured the quantity again, correctly."""
        roots, _ = self.receiver_roots(op_node)
        if not roots:
            return None
        recorded = {ln.strip()[:160] for ln in projected_lines if ln}
        for node in ast.walk(self.tree):
            if node is op_node:
                continue
            is_op = ((op in _METRIC_ATTRS and isinstance(node, ast.Attribute) and node.attr == op)
                     or (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                         and node.func.attr == op))
            if not is_op:
                continue
            other_roots, projected = self.receiver_roots(node)
            if not projected:
                text = self.lines[node.lineno - 1].strip()[:160] if node.lineno <= len(self.lines) else ""
                projected = text in recorded
            if projected and roots & other_roots:
                return node
        return None


def _first(names: Iterable[str]) -> Optional[str]:
    for n in sorted(names):
        return n
    return None


def _truthy(node: Optional[ast.AST]) -> bool:
    return isinstance(node, ast.Constant) and bool(node.value)


@lru_cache(maxsize=256)
def _crs_projected(text: str) -> Optional[bool]:
    try:
        from pyproj import CRS

        return bool(CRS.from_user_input(text).is_projected)
    except Exception:  # noqa: BLE001
        return None


@lru_cache(maxsize=64)
def analyze(code: str) -> Analysis:
    return Analysis(code or "")


def declared_kinds(code: Optional[str]) -> Dict[str, Kind]:
    if not code:
        return {}
    return {k: v[0] for k, v in analyze(code).declared().items()}


# --------------------------------------------------------------------------- the gate's findings

# What a metric operation's value does that cannot make a reported figure wrong.
_BENIGN = {"sign_test", "unused", "unit_free", "test"}


def measurement_reach(code: str, finding: Dict[str, Any],
                      metric_ops: Iterable[Dict[str, Any]] = ()) -> Tuple[bool, str]:
    """(reaches, why) for a projected_crs FAIL the operation tracker recorded.

    It does not reach the answer when its value is only compared with zero (`.area > 0` is the
    same in degrees as in metres), never used, or only printed by a run that measures the same
    quantity again on a reprojected copy of the same frame (an exploratory `print(gdf.area)`).
    Everything else reaches: a value compared with a non-zero threshold picks rows by a number in
    degrees, and a value the analysis cannot follow is assumed to reach."""
    op = str(finding.get("op") or "")
    line = str(finding.get("target") or "")
    if not code or not op:
        return True, ""
    a = analyze(code)
    nodes = a.op_nodes(line, op)
    if not nodes:
        return True, ""
    projected_lines = [str(o.get("code") or "") for o in metric_ops
                       if isinstance(o, dict) and o.get("projected") and o.get("op") == op]
    whys = []
    for node in nodes:
        sinks = a.sinks(node)
        if sinks <= _BENIGN:
            whys.append("only compared with zero" if "sign_test" in sinks
                        else "only tested for truth" if "test" in sinks
                        else "never used" if "unused" in sinks else "only counted")
            continue
        if sinks <= _BENIGN | {"print"}:
            again = a.remeasured_projected(node, op, projected_lines)
            if again is not None:
                whys.append(f"only printed, and measured again in a projected CRS on line "
                            f"{again.lineno}")
                continue
        return True, ""
    return False, "; ".join(dict.fromkeys(whys))


def optional_null_columns(code: str, finding: Dict[str, Any]) -> Tuple[bool, str]:
    """(ignorable, why) for an all_nan finding on a join result.

    Ignorable when the join matched rows and no line of the code names any of the empty
    columns: an optional attribute that is empty in the source (`old_operator`, `wikidata`),
    which no computed number read. A join whose index column is empty matched nothing, and a
    null column the code reads is a number computed from nothing; both stay."""
    columns = [str(c) for c in finding.get("columns") or []]
    target = str(finding.get("target") or "")
    if not columns or not code:
        return False, ""
    if any(c in ("index_right", "index_left") for c in columns):
        return False, ""
    a = analyze(code)
    if not a.ok:
        return False, ""
    read = a.columns_read() & set(columns)
    if read:
        return False, ""
    matched = finding.get("join_matched")
    how = a.join_kind(target)
    if matched is None:
        matched = how == "inner" or None
    if not matched:
        return False, ""
    via = "an inner join, so every row matched" if how == "inner" else "the join matched rows"
    return True, (f"{', '.join(columns)} empty after {via}, and no line of the code reads "
                  f"{'it' if len(columns) == 1 else 'them'}: an optional attribute, not a "
                  f"failed join")


def reclassify(report: Dict[str, Any], code: Optional[str]) -> Dict[str, Any]:
    """The gate's report with each sandbox finding that needs a figure's kind or its reach
    decided from the code (stage 54). A FAIL that does not apply becomes a pass and is listed in
    `kind_notes`; nothing is added. Never raises."""
    if not isinstance(report, dict):
        return report
    if not code:
        return _without_code(report)
    try:
        findings = list(report.get("findings") or [])
        ops = report.get("metric_ops") or []
        notes: List[Dict[str, Any]] = []
        changed = 0
        counts = dict(report.get("counts") or {})
        out = []
        for f in findings:
            if not isinstance(f, dict) or f.get("status") != FAIL:
                out.append(f)
                continue
            why = ""
            if f.get("check") == "projected_crs" and f.get("op"):
                reaches, why = measurement_reach(code, f, ops)
                if reaches:
                    why = ""
            elif f.get("check") == "all_nan":
                ignorable, why = optional_null_columns(code, f)
                if not ignorable:
                    why = ""
            if not why:
                out.append(f)
                continue
            changed += 1
            notes.append({"check": f.get("check"), "target": f.get("target"), "was": FAIL,
                          "why": why})
            out.append({**f, "status": PASS, "message": f"{f.get('message')} — not applied: {why}"})
            if counts:
                counts[FAIL] = max(0, counts.get(FAIL, 0) - 1)
                counts[PASS] = counts.get(PASS, 0) + 1
        if not changed:
            return report
        rep = {**report, "findings": out, "kind_notes": [*(report.get("kind_notes") or []),
                                                          *notes][:12]}
        if counts:
            rep["counts"] = counts
            if not report.get("findings_truncated"):
                rep["verdict"] = (FAIL if counts.get(FAIL) else UNKNOWN if counts.get(UNKNOWN)
                                  else PASS if counts.get(PASS) else report.get("verdict"))
        return rep
    except Exception:  # noqa: BLE001 - the kinds are a refinement; the report must survive
        return report


def _without_code(report: Dict[str, Any]) -> Dict[str, Any]:
    """No script to read (a run from a workspace file): an empty column after a join that
    matched is passed, as stage 46 did, because nothing says a number was computed from it."""
    if not any(isinstance(f, dict) and f.get("check") == "all_nan" and f.get("status") == FAIL
               and f.get("join_matched") for f in report.get("findings") or []):
        return report
    changed = 0
    out = []
    for f in report.get("findings") or []:
        if isinstance(f, dict) and f.get("check") == "all_nan" and f.get("status") == FAIL \
                and f.get("join_matched"):
            changed += 1
            f = {**f, "status": PASS}
        out.append(f)
    counts = dict(report.get("counts") or {})
    if counts:
        counts[FAIL] = max(0, counts.get(FAIL, 0) - changed)
        counts[PASS] = counts.get(PASS, 0) + changed
    rep = {**report, "findings": out, **({"counts": counts} if counts else {})}
    if counts and not report.get("findings_truncated"):
        rep["verdict"] = (FAIL if counts.get(FAIL) else UNKNOWN if counts.get(UNKNOWN)
                          else PASS if counts.get(PASS) else report.get("verdict"))
    return rep


__all__ = ["Kind", "Analysis", "analyze", "declared_kinds", "measurement_reach",
           "optional_null_columns", "reclassify", "UNKNOWN_KIND"]
