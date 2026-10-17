"""Watching a sandbox run while it executes: rule checks, a concurrent review, and where it stuck.

Production T16 (2026-10-17) ran six programs; three hit the 60 s limit and the model learned
only "execution timed out". The code was not wrong, it was slow, and the two slow lines were
recognisable before the first byte ran: a GeoSeries ``intersects`` against one 97,315-vertex
polygon that was never prepared (86 s as written, 7 s prepared) and a nested Python loop
snapping 23,382 nodes (8.4 s, 0.2 s with a KD-tree). This module is the three things that
would have caught them, in the order they act:

1. :func:`precheck` — **rules, before the run.** A syntax-tree scan for constructs that are
   slow on real geodata (unprepared predicates against one geometry, nested row loops, row-wise
   ``apply`` of geometry methods, unions or file reads inside loops) and for a stated parameter
   the code never uses ("buffer 2500 m" with only ``1000`` in the code). Deterministic and
   instant. It never blocks a run: the findings go back with the result, to the reviewer as
   leads, and are matched against the stack if the run times out.
2. :class:`RunWatch` — **a model review, during the run.** Once a run has been going for a few
   seconds (``AGENT_CODE_REVIEW_AFTER_S``) a reviewer model reads the code, the task, the input
   sizes, the rule findings and a live stack sample (where the program is right now), in a
   thread, while the container keeps computing. A run that finishes first waits for nothing
   and the review is dropped. A *critical* verdict stops the container and goes back to the
   code step as the run's result: what is wrong, on which line, and the fix. The code step
   (which owns the code) applies it; the reviewer never runs code of its own.
   A model can be wrong, and a wrong stop costs a whole run, so a verdict only stops anything
   after :func:`accept_verdict` checks it against things that cannot be made up: the quoted
   line must be in the code at the line cited, a performance claim must point at where the
   live stack actually is (or the run must be past half its limit), and a parameter claim
   must quote the task.
3. :func:`stack_report` — **where it was, on a timeout.** Just before a timed-out run is
   killed the host asks the program for its stack (faulthandler, through a ``sitecustomize``
   on the run's path), so the result names the line it was stuck on, with the matching rule
   finding.

``AGENT_CODE_REVIEW``: ``interrupt`` (default) reviews and stops on a critical verdict,
``observe`` reviews and reports but never stops a run, ``off`` skips the model entirely.
The rules and the stack report are always on: they cost nothing.
"""

from __future__ import annotations

import ast
import contextvars
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- settings


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default
    return value if value == value else default


def review_mode() -> str:
    raw = (os.getenv("AGENT_CODE_REVIEW") or "interrupt").strip().lower()
    if raw in {"0", "false", "no", "off", "none"}:
        return "off"
    if raw in {"observe", "report", "warn"}:
        return "observe"
    return "interrupt"


def review_after_s() -> float:
    return max(0.0, _env_float("AGENT_CODE_REVIEW_AFTER_S", 5.0))


def perf_floor_s() -> float:
    """How long a run must have been going before a PERFORMANCE verdict may stop it.

    Live 2026-10-17: the review stopped an ``sjoin`` at 7.7 s as too slow; the same join had
    finished in 12 s one run earlier. "This will not finish in time" cannot be known seconds in,
    so a performance verdict is held until the run has had this long and is then re-checked
    against a fresh stack sample (the program must still be in the cited code). Bugs and
    parameter errors are wrong at any speed and stop at once."""
    return max(0.0, _env_float("AGENT_CODE_REVIEW_PERF_FLOOR_S", 30.0))


def heartbeat_every_s() -> float:
    return max(1.0, _env_float("AGENT_CODE_PROGRESS_SECONDS", 5.0))


# --------------------------------------------------------------------------- stack dumps

RUNTIME_DIRNAME = ".iguide_rt"
STACK_FILENAME = "stack.txt"

# Installed as the run's sitecustomize: Python imports it at start-up from the first PYTHONPATH
# entry, before the program, so the program's own line numbers are untouched. SIGUSR1 then
# writes the Python stack to a file the host reads — faulthandler does it from the signal
# handler itself, so a program stuck inside GEOS or numpy answers too. The file, not stderr:
# the program's stderr is the model's to read, and live samples would litter it.
SITECUSTOMIZE = '''\
import os as _os
_path = _os.environ.get("IGUIDE_STACK_FILE")
if _path:
    try:
        import faulthandler as _fh, signal as _sig
        _IGUIDE_STACK_FH = open(_path, "a")
        _fh.register(_sig.SIGUSR1, file=_IGUIDE_STACK_FH, all_threads=False)
    except Exception:
        pass
'''

_FRAME = re.compile(r'File "([^"]+)", line (\d+) in (\S+)')


def parse_stack(text: str, *, script: str, line_offset: int = 0,
                code_lines: int = 0) -> List[Dict[str, Any]]:
    """The program's own frames from the LAST dump in *text*, innermost first.

    faulthandler prints most recent call first. Only frames in *script* are kept (library
    frames are where the time goes but not what the code step can change), mapped back through
    *line_offset* to the lines of the code as written; a frame outside them belongs to the
    invariant gate's prologue or epilogue and is dropped.
    """
    if not text:
        return []
    blocks = re.split(r"(?m)^(?:Current thread|Thread) 0x[0-9a-fA-F]+", text)
    last = blocks[-1] if blocks else text
    frames: List[Dict[str, Any]] = []
    first_lib: Optional[str] = None
    for match in _FRAME.finditer(last):
        path, line, func = match.group(1), int(match.group(2)), match.group(3)
        if os.path.basename(path) != os.path.basename(script):
            if first_lib is None and not frames:
                first_lib = f"{path.rsplit('site-packages/', 1)[-1]}:{line} in {func}"
            continue
        user_line = line - line_offset
        if user_line < 1 or (code_lines and user_line > code_lines):
            continue
        frames.append({"line": user_line, "function": func,
                       **({"inside": first_lib} if first_lib and not frames else {})})
    return frames


# --------------------------------------------------------------------------- rule checks

_PREDICATES = {"intersects", "within", "contains", "covers", "covered_by", "overlaps",
               "touches", "crosses", "disjoint", "contains_properly", "dwithin"}
_GEOM_METHODS = {"distance", "intersects", "within", "contains", "buffer", "intersection",
                 "union", "difference", "touches", "covers", "project", "interpolate"}
_READERS = {"read_file", "read_csv", "read_parquet", "read_excel", "open", "read_json"}
_UNIONS = {"unary_union", "union_all", "dissolve", "cascaded_union"}


@dataclass
class Finding:
    rule: str
    line: int
    message: str
    fix: str

    def as_dict(self) -> Dict[str, Any]:
        return {"rule": self.rule, "line": self.line, "message": self.message, "fix": self.fix}


def _call_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Call):
        fn = node.func
        if isinstance(fn, ast.Attribute):
            return fn.attr
        if isinstance(fn, ast.Name):
            return fn.id
    return None


def _is_loop(node: ast.AST) -> bool:
    return isinstance(node, (ast.For, ast.While, ast.ListComp, ast.SetComp, ast.DictComp,
                             ast.GeneratorExp))


def _small_literal_loop(node: ast.AST) -> bool:
    """`for x in (a, b, c)` / `for i in range(3)`: a handful of passes, not a row loop."""
    it = getattr(node, "iter", None)
    if isinstance(it, (ast.Tuple, ast.List, ast.Set)):
        return len(it.elts) <= 20
    if isinstance(it, ast.Call) and _call_name(it) == "range" and it.args:
        last = it.args[-1] if len(it.args) <= 2 else it.args[1]
        return isinstance(last, ast.Constant) and isinstance(last.value, int) and last.value <= 50
    return False


class _Scan(ast.NodeVisitor):
    def __init__(self) -> None:
        self.loops: List[ast.AST] = []
        self.found: List[Finding] = []
        self.prepared = False
        self.kdtree = False

    def _add(self, rule: str, node: ast.AST, message: str, fix: str) -> None:
        line = getattr(node, "lineno", 0) or 0
        if not any(f.rule == rule and f.line == line for f in self.found):
            self.found.append(Finding(rule, line, message, fix))

    def _row_loops(self) -> List[ast.AST]:
        return [n for n in self.loops if not _small_literal_loop(n)]

    def generic_visit(self, node: ast.AST) -> None:
        if _is_loop(node):
            outer = self._row_loops()
            if outer and not _small_literal_loop(node) and isinstance(node, (ast.For, ast.While)):
                self._add("nested_loop", node,
                          f"a loop inside the loop on line {outer[-1].lineno}: the work grows "
                          "with the product of both sizes, so tens of thousands of rows on each "
                          "side is hundreds of millions of passes",
                          "index one side once and query it: scipy.spatial.cKDTree for nearest "
                          "points, gpd.sjoin / gpd.sjoin_nearest for spatial matches, "
                          "DataFrame.merge for keys")
            self.loops.append(node)
            super().generic_visit(node)
            self.loops.pop()
            return
        super().generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node)
        if name in {"prepare", "prep"}:
            self.prepared = True
        if name in {"cKDTree", "KDTree", "BallTree", "STRtree", "sjoin", "sjoin_nearest"}:
            self.kdtree = True
        if name == "iterrows":
            self._add("iterrows", node,
                      "iterrows() builds a Series for every row; over thousands of features "
                      "this alone takes minutes",
                      "use vectorised columns, itertuples(), or a spatial join")
        if name in _PREDICATES and isinstance(node.func, ast.Attribute) and len(node.args) == 1:
            arg = node.args[0]
            single = (isinstance(arg, (ast.Name, ast.Subscript)) or _call_name(arg) in _UNIONS
                      or (isinstance(arg, ast.Attribute) and arg.attr in _UNIONS | {"geometry"}))
            if single:
                self._add("unprepared_predicate", node,
                          f".{name}() of many geometries against one geometry tests each one "
                          "against every vertex of it; against a polygon with tens of thousands "
                          "of vertices that takes minutes, and roughly ten times less prepared",
                          "shapely.prepare(geom) once before the call, or gpd.sjoin(points, "
                          "polygons_gdf, predicate=...) which indexes both sides; explode a "
                          "multi-part polygon and simplify it if the precision allows")
        if name == "apply" and any(isinstance(a, ast.Lambda) for a in node.args):
            lam = next(a for a in node.args if isinstance(a, ast.Lambda))
            calls = {_call_name(n) for n in ast.walk(lam.body)}
            if calls & _GEOM_METHODS:
                self._add("rowwise_geometry", node,
                          "a geometry method called row by row inside apply(): each call "
                          "crosses into GEOS separately",
                          "call the GeoSeries method on the whole column, or use sjoin / "
                          "sjoin_nearest")
        if self._row_loops():
            if name in _UNIONS:
                self._add("union_in_loop", node,
                          f"{name} inside a loop rebuilds the union on every pass",
                          "collect the pieces and union once after the loop")
            if name in _READERS:
                self._add("read_in_loop", node,
                          f"{name} inside a loop reads the file again on every pass",
                          "read it once before the loop")
            if name == "distance":
                self._add("distance_in_loop", node,
                          "distance() inside a Python loop: one GEOS call per pair",
                          "scipy.spatial.cKDTree(...).query for nearest points, or "
                          "gpd.sjoin_nearest(..., distance_col=...)")
        self.generic_visit(node)


# A number with a distance or time unit, as a person writes it in a request.
_STATED = re.compile(
    r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*"
    r"(km|kilomet(?:er|re)s?|m|met(?:er|re)s?|mi|miles?|ft|feet|minutes?|mins?|hours?|hrs?)\b",
    re.IGNORECASE)
_PARAM_WORDS = re.compile(r"\b(buffer|radius|within|distance|walk|drive|isochrone|catchment|"
                          r"threshold|band|cut-?off|range)\b", re.IGNORECASE)
_UNIT_EQUIV = {"km": [1000.0], "m": [0.001], "mi": [1609.344, 1.609344], "ft": [0.3048],
               "min": [60.0, 1 / 60.0], "h": [60.0, 3600.0]}


def _unit_key(unit: str) -> str:
    u = unit.lower()
    if u.startswith("k"):
        return "km"
    if u in {"mi", "mile", "miles"}:
        return "mi"
    if u in {"ft", "feet"}:
        return "ft"
    if u.startswith("min"):
        return "min"
    if u.startswith("h"):
        return "h"
    return "m"


def _numbers_in(code: str) -> List[float]:
    out: List[float] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            out.append(float(node.value))
    return out


def stated_parameters(brief: str) -> List[Tuple[float, str, str]]:
    """(value, unit key, the words) for each distance/time figure the task states next to a
    word that makes it a parameter (buffer, radius, within, ...)."""
    found: List[Tuple[float, str, str]] = []
    for match in _STATED.finditer(brief or ""):
        start = max(0, match.start() - 60)
        if not _PARAM_WORDS.search(brief[start:match.end() + 20]):
            continue
        try:
            value = float(match.group(1).replace(",", ""))
        except ValueError:
            continue
        if value <= 0:
            continue
        found.append((value, _unit_key(match.group(2)), match.group(0)))
    return found


def _parameter_findings(code: str, brief: str) -> List[Finding]:
    stated = stated_parameters(brief)
    if not stated:
        return []
    numbers = _numbers_in(code)
    out: List[Finding] = []
    for value, unit, words in stated:
        candidates = [value] + [value * f for f in _UNIT_EQUIV.get(unit, [])]
        if any(abs(n - c) <= 1e-6 * max(1.0, abs(c)) for n in numbers for c in candidates):
            continue
        out.append(Finding(
            "stated_parameter", 0,
            f"the task states “{words}”, and no number in the code equals it in any unit",
            f"use the stated value ({words}) or say in the result why a different one is used"))
    return out


def precheck(code: str, *, brief: str = "") -> List[Dict[str, Any]]:
    """Rule findings for *code* (see the module docstring). Never raises."""
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return []
    try:
        scan = _Scan()
        scan.visit(tree)
        found = list(scan.found)
        if scan.prepared:
            found = [f for f in found if f.rule != "unprepared_predicate"]
        if scan.kdtree:
            found = [f for f in found if f.rule not in {"distance_in_loop"}]
        found += _parameter_findings(code or "", brief)
        return [f.as_dict() for f in sorted(found, key=lambda f: (f.line or 10 ** 9, f.rule))]
    except Exception:  # noqa: BLE001 - a rule check must never cost a run
        logger.debug("precheck failed", exc_info=True)
        return []


# --------------------------------------------------------------------------- the stuck line

def enclosing_lines(code: str, line: int) -> List[int]:
    """Start lines of the statements and loops that contain *line*, outermost first."""
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if isinstance(node, (ast.stmt, ast.comprehension, ast.ListComp, ast.GeneratorExp,
                             ast.SetComp, ast.DictComp, ast.Call)) \
                and start is not None and end is not None and start <= line <= end:
            out.append(start)
    return sorted(set(out))


def stack_report(frames: Sequence[Dict[str, Any]], code: str,
                 findings: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Where a killed run was: the innermost line of the program, its text, and the rule
    finding that explains it when one covers that line or a loop around it."""
    if not frames:
        return None
    lines = (code or "").splitlines()
    top = frames[0]
    line = int(top["line"])
    report: Dict[str, Any] = {
        "line": line,
        "source": lines[line - 1].strip()[:200] if 0 < line <= len(lines) else "",
        "function": top.get("function"),
        "call_stack": [f"line {f['line']} in {f.get('function')}" for f in frames[:6]],
    }
    if top.get("inside"):
        report["inside"] = top["inside"]
    around = set(enclosing_lines(code, line)) | {line}
    match = [f for f in findings if f.get("line") in around]
    if match:
        report["explained_by"] = match[0]
    return report


# --------------------------------------------------------------------------- input sizes

_VECTOR_EXT = (".shp", ".zip", ".geojson", ".json", ".gpkg", ".parquet", ".geoparquet", ".fgb")
_HEAVY_VERTICES = 10_000
_SIZE_CACHE: Dict[Tuple[str, int, int], Optional[Dict[str, Any]]] = {}


def geometry_size(path: str, layer: Optional[str] = None, *,
                  max_features: int = 250_000) -> Optional[Dict[str, Any]]:
    """Vertex and part counts of a vector file: the size that decides what geometry code costs.

    A feature count says nothing about it. The SLR polygon of T16 was ONE feature with 97,315
    vertices in 474 parts, and ``inspect_vector`` reported "1 feature", so nothing told the
    model that testing 9,843 buildings against it would take minutes. None when the file is not
    a readable vector or has more than *max_features* features (reading every geometry of a
    huge layer is the cost this exists to warn about).
    """
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (str(path) + "|" + str(layer or ""), int(st.st_size), int(st.st_mtime_ns))
    if key in _SIZE_CACHE:
        return _SIZE_CACHE[key]
    result: Optional[Dict[str, Any]] = None
    try:
        import numpy as np
        import shapely

        if str(path).lower().endswith((".parquet", ".geoparquet")):
            import geopandas as gpd
            geoms = gpd.read_parquet(path, columns=None).geometry.values
        else:
            import pyogrio
            info = pyogrio.read_info(path, layer=layer) if layer else pyogrio.read_info(path)
            if int(info.get("features") or 0) > max_features:
                _SIZE_CACHE[key] = None
                return None
            if not info.get("geometry_type"):
                _SIZE_CACHE[key] = None
                return None
            frame = pyogrio.read_dataframe(path, layer=layer, columns=[]) if layer \
                else pyogrio.read_dataframe(path, columns=[])
            geoms = frame.geometry.values
        arr = np.asarray(geoms, dtype=object)
        verts = shapely.get_num_coordinates(arr)
        parts = shapely.get_num_geometries(arr)
        if len(verts) == 0:
            result = {"features": 0, "vertices_total": 0, "vertices_max": 0}
        else:
            i = int(np.argmax(verts))
            result = {"features": int(len(verts)), "vertices_total": int(verts.sum()),
                      "vertices_max": int(verts[i]), "parts_max": int(parts.max()),
                      "largest_feature_index": i}
            if result["vertices_max"] >= _HEAVY_VERTICES:
                result["note"] = (
                    f"feature {i} alone has {result['vertices_max']:,} vertices in "
                    f"{int(parts[i])} part(s): a predicate or overlay against it tests every "
                    "vertex for each other geometry. Prepare it (shapely.prepare), index with "
                    "gpd.sjoin, or explode/simplify it before testing many geometries against it")
    except Exception:  # noqa: BLE001 - a size report is never worth a failure
        logger.debug("geometry_size failed for %s", path, exc_info=True)
        result = None
    if len(_SIZE_CACHE) > 256:
        _SIZE_CACHE.clear()
    _SIZE_CACHE[key] = result
    return result


def describe_inputs(inputs: Sequence[Dict[str, Any]], *, budget_s: float = 8.0) -> List[Dict[str, Any]]:
    """Add feature and vertex counts to vector inputs ({name, path, size_mb}), within a budget."""
    started = time.monotonic()
    out: List[Dict[str, Any]] = []
    for item in inputs:
        entry = {k: v for k, v in item.items() if k != "path"}
        path = item.get("path")
        if path and str(path).lower().endswith(_VECTOR_EXT) \
                and time.monotonic() - started < budget_s:
            size = geometry_size(str(path))
            if size:
                entry.update({"features": size.get("features"),
                              "vertices": size.get("vertices_total"),
                              "vertices_max": size.get("vertices_max")})
                if size.get("note"):
                    entry["note"] = size["note"]
        out.append(entry)
    return out


# --------------------------------------------------------------------------- the reviewer

_REVIEW_SYSTEM = """You review a Python program that is RUNNING right now in a sandbox (2 CPUs, 4 GB, \
no network). It works on geospatial data for the task below. Your job is to decide whether it should \
be STOPPED now because it is critically wrong. Stopping costs the whole run, so say "critical" only \
for one of these, and only with a specific line:

- performance: it cannot finish within its time limit at these input sizes (an unindexed pairwise \
loop, a predicate against one huge unprepared geometry, a row-wise apply over many thousands of rows, \
file reads or unions inside a loop) AND the live stack shows it is in that code now;
- parameter: it uses a value the task states differently (a 1000 m buffer when the task says 2500 m, \
a different year, a different field), quoting the task;
- bug: it will produce a wrong result or crash later (distances in degrees, a join on the wrong key, \
positional vs label indexing after a filter, a unit error);
- data: it reads the wrong file or column for what the task asks.

Style, small inefficiencies, missing comments, and anything you are unsure of are "ok".

Answer with ONE JSON object and nothing else:
{"verdict": "ok" | "critical", "category": "performance" | "parameter" | "bug" | "data", \
"line": <line number in the numbered code>, "quote": "<exact text copied from that line>", \
"task_quote": "<exact words from the task, for a parameter problem>", \
"problem": "<one sentence>", "fix": "<what to change, concretely, with code if short>"}"""


def _numbered(code: str, limit: int = 400) -> str:
    lines = (code or "").splitlines()
    shown = [f"{i:4d}| {text}" for i, text in enumerate(lines[:limit], 1)]
    if len(lines) > limit:
        shown.append(f"…[{len(lines) - limit} more lines not shown]")
    return "\n".join(shown)


def review_prompt(*, code: str, brief: str, inputs: Sequence[Dict[str, Any]],
                  findings: Sequence[Dict[str, Any]], frames: Sequence[Dict[str, Any]],
                  elapsed: float, timeout: int, stdout_tail: str) -> str:
    parts = [f"TASK:\n{(brief or '(not given)')[:3000]}",
             f"TIME: running {elapsed:.0f} s of a {timeout} s limit."]
    if inputs:
        parts.append("INPUTS:\n" + "\n".join(
            f"- {i.get('name')}: {i.get('size_mb', '?')} MB"
            + (f", {i['features']:,} features" if i.get("features") else "")
            + (f", {i['vertices']:,} vertices" if i.get("vertices") else "")
            + (f" (largest feature {i['vertices_max']:,})" if i.get("vertices_max") else "")
            + (f". {i['note']}" if i.get("note") else "") for i in inputs[:12]))
    if frames:
        parts.append("LIVE STACK (where the program is right now, innermost first):\n" + "\n".join(
            f"- line {f['line']} in {f.get('function')}"
            + (f" (inside {f['inside']})" if f.get("inside") else "") for f in frames[:6]))
    else:
        parts.append("LIVE STACK: not available.")
    if findings:
        parts.append("RULE CHECKS (automatic and generic: they know the code, not these inputs' "
                     "sizes or timings; harmless at small sizes):\n" + "\n".join(
            f"- line {f.get('line') or '?'}: [{f['rule']}] {f['message']}" for f in findings[:8]))
    if stdout_tail.strip():
        parts.append("OUTPUT SO FAR (tail):\n" + stdout_tail[-1500:])
    parts.append("CODE:\n" + _numbered(code))
    return "\n\n".join(parts)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def accept_verdict(verdict: Any, *, code: str, brief: str, frames: Sequence[Dict[str, Any]],
                   elapsed: float, timeout: int) -> Tuple[bool, str]:
    """Whether a critical verdict may stop the run, and why not when it may not.

    Each test checks the verdict against something the reviewer cannot invent."""
    if not isinstance(verdict, dict) or str(verdict.get("verdict")).lower() != "critical":
        return False, "not critical"
    category = str(verdict.get("category") or "").lower()
    if category not in {"performance", "parameter", "bug", "data"}:
        return False, f"unknown category {category!r}"
    lines = (code or "").splitlines()
    try:
        line = int(verdict.get("line"))
    except (TypeError, ValueError):
        return False, "no line cited"
    if not 1 <= line <= len(lines):
        return False, f"line {line} is not in the code"
    quote = _norm(verdict.get("quote"))
    if len(quote) < 4:
        return False, "no quote from the cited line"
    window = _norm(" ".join(lines[max(0, line - 2):line + 1]))
    if quote not in window:
        return False, f"the quote is not on line {line}"
    if not _norm(verdict.get("fix")):
        return False, "no fix given"
    if category == "performance":
        if frames:
            now = int(frames[0]["line"])
            here = {int(f["line"]) for f in frames}
            if line not in here and line not in enclosing_lines(code, now):
                return False, (f"the program is at line {now}, not in the code at line {line} "
                               "the review calls slow")
        elif elapsed < 0.5 * timeout:
            return False, "no live stack, and the run is not yet past half its limit"
    if category == "parameter":
        task_quote = _norm(verdict.get("task_quote"))
        if len(task_quote) < 3 or task_quote.lower() not in _norm(brief).lower():
            return False, "the stated value is not quoted from the task"
    return True, ""


def default_reviewer(prompt: str) -> str:
    """One model call. The deployment's model unless AGENT_CODE_REVIEW_MODEL names another."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from agent_runtime.executor_factory import build_default_llm, build_llm

    model = (os.getenv("AGENT_CODE_REVIEW_MODEL") or "").strip()
    provider = (os.getenv("AGENT_CODE_REVIEW_PROVIDER") or "").strip()
    llm = build_llm(provider=provider or None, model=model or None) if (model or provider) \
        else build_default_llm()
    reply = llm.invoke([SystemMessage(content=_REVIEW_SYSTEM), HumanMessage(content=prompt)])
    content = getattr(reply, "content", reply)
    if isinstance(content, list):
        content = "".join(str(p.get("text", "")) if isinstance(p, dict) else str(p)
                          for p in content)
    return str(content or "")


def parse_verdict(text: str) -> Optional[Dict[str, Any]]:
    try:
        from agent_runtime.evidence_quality import _extract_json_object
        found = _extract_json_object(text)
    except Exception:  # noqa: BLE001
        found = None
    if found is None:
        match = re.search(r"\{.*\}", text or "", re.S)
        if match:
            try:
                found = json.loads(match.group(0))
            except ValueError:
                found = None
    return found if isinstance(found, dict) else None


# --------------------------------------------------------------------------- the watcher

@dataclass
class Interrupt:
    kind: str                      # "review" | "cancelled"
    message: str
    review: Optional[Dict[str, Any]] = None


@dataclass
class RunProbe:
    """What the supervision loop lets the watcher see of a live run."""
    sample_stack: Callable[[], List[Dict[str, Any]]]
    stdout_tail: Callable[[], str]


@dataclass
class RunWatch:
    code: str
    timeout: int
    label: str = ""
    brief: str = ""
    inputs: List[Dict[str, Any]] = field(default_factory=list)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    mode: str = field(default_factory=review_mode)
    review_after: float = field(default_factory=review_after_s)
    perf_floor: float = field(default_factory=perf_floor_s)
    heartbeat_every: float = field(default_factory=heartbeat_every_s)
    reviewer: Callable[[str], str] = default_reviewer
    emit: Optional[Callable[[str, Dict[str, Any]], None]] = None

    def __post_init__(self) -> None:
        self._probe: Optional[RunProbe] = None
        self._last_beat = -1e9
        self._thread: Optional[threading.Thread] = None
        self._done = threading.Event()
        self._result: Optional[Dict[str, Any]] = None
        self.review_state = "off" if self.mode == "off" else "waiting"
        self.review_record: Optional[Dict[str, Any]] = None
        self._held: Optional[Dict[str, Any]] = None     # a performance stop waiting for its floor
        if self.emit is None:
            from agent_runtime.streaming_trace import emit_trace_event
            self.emit = lambda event, data: emit_trace_event(event, data)

    # -- loop hooks ----------------------------------------------------------------------
    def start(self, probe: RunProbe) -> None:
        self._probe = probe
        self._send("code_progress", {"stage": "started", "elapsed_s": 0})
        if self.findings:
            self._send("code_review", {
                "stage": "rules", "verdict": "hints",
                "message": "Rule check: " + "; ".join(
                    f"line {f['line']}: {f['rule'].replace('_', ' ')}" if f.get("line")
                    else f["rule"].replace("_", " ") for f in self.findings[:4]),
                "findings": self.findings[:8]})

    def tick(self, elapsed: float) -> Optional[Interrupt]:
        if self.mode != "off" and self._thread is None and elapsed >= self.review_after:
            self._begin_review(elapsed)
        if elapsed - self._last_beat >= self.heartbeat_every:
            self._last_beat = elapsed
            self._send("code_progress", {"stage": "running", "elapsed_s": round(elapsed, 1)})
        if self._done.is_set() and self.review_record is None:
            return self._settle(elapsed)
        if self._held is not None and elapsed >= self.perf_floor:
            return self._confirm_held(elapsed)
        return None

    def _confirm_held(self, elapsed: float) -> Optional[Interrupt]:
        """At the floor: stop only if a fresh sample still has the program in the cited code."""
        record, self._held = self._held, None
        frames: List[Dict[str, Any]] = []
        if self._probe is not None:
            try:
                frames = self._probe.sample_stack()
            except Exception:  # noqa: BLE001
                frames = []
        ok, why_not = accept_verdict({"verdict": "critical", **record}, code=self.code,
                                     brief=self.brief, frames=frames, elapsed=elapsed,
                                     timeout=self.timeout)
        if not ok:
            record["not_acted_on"] = f"re-checked at {elapsed:.0f} s: {why_not}"
            self.review_state = "ok"
            self._send("code_review", {"stage": "done", "verdict": "ok", "stopped": False,
                                       "message": f"Review's slow-line flag dropped: {why_not}"[:300],
                                       "finding": record})
            return None
        record["stopped_run"] = True
        record["confirmed_at_s"] = round(elapsed, 1)
        self.review_state = "stopped"
        self._send("code_review", {
            "stage": "done", "verdict": "critical", "stopped": True,
            "message": f"Review stopped the run — line {record.get('line')}: "
                       f"{record.get('problem', '')}"[:300],
            "finding": record})
        return Interrupt("review", f"stopped by the code review: {record.get('problem', '')}",
                         review=record)

    def finish(self, *, ran_s: float, outcome: str) -> Optional[Dict[str, Any]]:
        """Called once the run ended (any way). Returns the review record for the result."""
        if self.review_record is None and self._thread is not None and not self._done.is_set():
            self.review_state = "dropped"
            self.review_record = {"state": "dropped",
                                  "note": "the run ended before the review answered"}
        self._send("code_progress", {"stage": "finished", "elapsed_s": round(ran_s, 1),
                                     "outcome": outcome})
        return self.review_record

    # -- internals -----------------------------------------------------------------------
    def _send(self, event: str, data: Dict[str, Any]) -> None:
        payload = {"label": self.label or "code", "timeout_s": self.timeout,
                   "review": self.review_state, **data}
        try:
            self.emit(event, payload)        # type: ignore[misc]
        except Exception:  # noqa: BLE001 - progress must never cost a run
            logger.debug("code progress emit failed", exc_info=True)

    def _begin_review(self, elapsed: float) -> None:
        frames = []
        stdout = ""
        if self._probe is not None:
            try:
                frames = self._probe.sample_stack()
            except Exception:  # noqa: BLE001
                frames = []
            try:
                stdout = self._probe.stdout_tail()
            except Exception:  # noqa: BLE001
                stdout = ""
        self._frames = frames
        self._review_started = elapsed
        self.review_state = "reviewing"
        self._send("code_review", {"stage": "started", "verdict": "pending",
                                   "message": "Reviewing the running code"})
        ctx = contextvars.copy_context()   # the turn's usage accounting goes with it

        def _work() -> None:
            try:
                # Sized here, in the review's thread: reading every geometry of an input can
                # take seconds, and the run is computing meanwhile either way.
                inputs = describe_inputs(self.inputs)
                prompt = review_prompt(code=self.code, brief=self.brief, inputs=inputs,
                                       findings=self.findings, frames=frames, elapsed=elapsed,
                                       timeout=self.timeout, stdout_tail=stdout)
                self.last_prompt = prompt
                text = ctx.run(self.reviewer, prompt)
                self._result = {"verdict": parse_verdict(text), "raw": (text or "")[:2000]}
            except Exception as exc:  # noqa: BLE001 - a failed review is no review
                self._result = {"verdict": None, "error": f"{type(exc).__name__}: {exc}"[:300]}
            finally:
                self._done.set()

        self._thread = threading.Thread(target=_work, name="code-review", daemon=True)
        self._thread.start()

    def _settle(self, elapsed: float) -> Optional[Interrupt]:
        result = self._result or {}
        verdict = result.get("verdict")
        took = round(elapsed - getattr(self, "_review_started", elapsed), 1)
        if verdict is None:
            self.review_state = "failed"
            self.review_record = {"state": "failed", "took_s": took,
                                  "error": result.get("error") or "the reply held no verdict"}
            self._send("code_review", {"stage": "done", "verdict": "unavailable",
                                       "message": "The code review did not answer"})
            return None
        ok, why_not = accept_verdict(verdict, code=self.code, brief=self.brief,
                                     frames=getattr(self, "_frames", []), elapsed=elapsed,
                                     timeout=self.timeout)
        record = {"state": "done", "took_s": took, "verdict": verdict.get("verdict"),
                  **{k: verdict.get(k) for k in ("category", "line", "quote", "task_quote",
                                                 "problem", "fix") if verdict.get(k)}}
        if str(verdict.get("verdict")).lower() == "critical" and not ok:
            record["not_acted_on"] = why_not
        if ok and self.mode == "interrupt" and str(verdict.get("category")).lower() == "performance" \
                and elapsed < self.perf_floor:
            record["held_until_s"] = self.perf_floor
            self._held = record
            self.review_state = "critical"
            self.review_record = record
            self._send("code_review", {
                "stage": "held", "verdict": "critical", "stopped": False,
                "message": f"Review flagged line {record.get('line')} as slow; the run is stopped "
                           f"if it is still there at {int(self.perf_floor)} s"[:300],
                "finding": record})
            return None
        if ok and self.mode == "interrupt":
            record["stopped_run"] = True
            self.review_state = "stopped"
            self.review_record = record
            self._send("code_review", {
                "stage": "done", "verdict": "critical", "stopped": True,
                "message": f"Review stopped the run — line {record.get('line')}: "
                           f"{record.get('problem', '')}"[:300],
                "finding": record})
            return Interrupt("review", f"stopped by the code review: {record.get('problem', '')}",
                             review=record)
        self.review_state = "critical" if ok else "ok"
        self.review_record = record
        self._send("code_review", {
            "stage": "done", "verdict": "critical" if ok else "ok", "stopped": False,
            "message": (f"Review flagged line {record.get('line')}: {record.get('problem', '')}"
                        if ok else "Review: no critical problem")[:300],
            "finding": record})
        return None


__all__ = ["Finding", "Interrupt", "RUNTIME_DIRNAME", "RunProbe", "RunWatch", "SITECUSTOMIZE",
           "STACK_FILENAME", "accept_verdict", "enclosing_lines", "parse_stack", "parse_verdict",
           "precheck", "review_mode", "review_prompt", "stack_report", "stated_parameters"]
