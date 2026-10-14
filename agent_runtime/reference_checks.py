"""A standard method's result, recomputed with a reference library (plan step 4b).

Two sources:

* **Code the agent wrote.** The sandbox records ordinary-kriging systems and esda Gi\\*/Moran
  calls and recomputes them before the run ends (agent_runtime/reference_sandbox.py). The
  result arrives on the tool result as `verification.reference`. Here the figures are tied to
  the run: a kriged value counts only if the run printed or declared it (or its exp()), so a
  kriging of `cadmium` the run never looked at is not a figure.
* **The built-in tools.** `local_getis_ord`, `global_spatial_autocorrelation`,
  `terrain_derivative` (slope) and `zonal_stats_for_raster` are recomputed here, in the agent
  process where the tools themselves ran, from the same input file:
  - Gi\\* with esda, binary weights, the focal area included;
  - Moran's I with esda, row-standardised weights;
  - slope with Horn's 3x3 method in numpy, on ground distances;
  - the zonal mean in numpy over a rasterio mask.

  The neighbour structure is built by the tool module's own `_weights_for`. It is not what is
  under test, and building it the same way keeps the two on the same areas.

A recomputation that disagrees becomes figure pairs (`reported`, `reference`). The answer's
writer is told before writing (`reconcile_note`). After it is written, an answer that states the
reported value and not the reference gets a finding (`findings`). Its kind is `KIND`, which the
benchmark decides (see KIND).
"""
from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

CHECK = "reference"
# A `problem` prints a banner; the plan admits a check to it only when the answers it marks are
# wrong at least 80% of the time over the scored archive. Measured (stage 53, validators bench
# with --reexec, 462 replayable turns, 77 wrong): 33 turns, all wrong.
KIND = "problem"
THRESHOLDS = (1.645, 1.96, 2.576)
# Size caps for recomputing a tool's result in the agent process.
MAX_FEATURES = 20000
MAX_PIXELS = 40_000_000

Resolver = Callable[[str], Any]


@dataclass
class Figure:
    label: str
    reported: float
    reference: float
    tol: float
    integer: bool = False
    context: Tuple[float, ...] = ()      # numbers the answer's sentence must also state

    @property
    def differs(self) -> bool:
        return abs(self.reported - self.reference) > self.tol


@dataclass
class Reference:
    method: str
    call_id: Optional[str]
    engine: str
    detail: str = ""
    figures: List[Figure] = field(default_factory=list)

    def disagreeing(self) -> List[Figure]:
        return [f for f in self.figures if f.differs]


# --------------------------------------------------------------------------- parsing

def _body(content: Any) -> Dict[str, Any]:
    if isinstance(content, dict):
        return content
    s = str(content or "").strip()
    if s.startswith("content='") or s.startswith('content="'):
        quote = s[8]
        end = s.find(f"{quote} name=", 9)
        try:
            s = ast.literal_eval(s[8:end + 1] if end > 0 else s[8:])
        except Exception:  # noqa: BLE001
            return {}
    if isinstance(s, str) and s.lstrip().startswith("{"):
        try:
            body = json.loads(s)
            return body if isinstance(body, dict) else {}
        except ValueError:
            return {}
    return {}


def _args(args: Any) -> Dict[str, Any]:
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        for parse in (json.loads, ast.literal_eval):
            try:
                v = parse(args)
                if isinstance(v, dict):
                    return v
            except Exception:  # noqa: BLE001
                continue
    return {}


_SHOWN_RE = re.compile(r"(?<![\w.])[-−]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _shown_numbers(body: Dict[str, Any]) -> List[Tuple[float, float]]:
    """(value, half a unit of its last shown digit) for every number the run printed or
    declared, with three or more significant digits: those are what a reader can be told."""
    out: List[Tuple[float, float]] = []
    texts = [str(body.get("stdout") or "")]
    for o in ((body.get("verification") or {}).get("outputs") or []):
        if isinstance(o, dict) and isinstance(o.get("value"), (int, float)):
            texts.append(repr(float(o["value"])))
    for text in texts:
        for m in _SHOWN_RE.finditer(text[:200000]):
            raw = m.group(0).replace("−", "-")
            try:
                v = float(raw)
            except ValueError:
                continue
            mant = raw.split("e")[0].split("E")[0]
            digits = re.sub(r"\D", "", mant).lstrip("0")
            if len(digits) < 3:
                continue
            dec = len(mant.split(".")[1]) if "." in mant else 0
            out.append((v, 0.5 * 10 ** (-dec) + 1e-12 * abs(v)))
    return out


def _was_shown(value: float, shown: Sequence[Tuple[float, float]]) -> bool:
    return math.isfinite(value) and any(abs(v - value) <= tol for v, tol in shown)


# --------------------------------------------------------------------------- the sandbox's records

def from_sandbox(content: Any, call_id: Optional[str] = None) -> List[Reference]:
    body = _body(content)
    records = (body.get("verification") or {}).get("reference") or []
    if not records:
        return []
    shown = _shown_numbers(body)
    out: List[Reference] = []
    for r in records:
        if not isinstance(r, dict) or r.get("status") != "recomputed":
            continue
        method = r.get("method")
        if method == "ordinary_kriging":
            ref = _kriging_from_record(r, shown, call_id)
        elif method == "getis_ord_gi_star":
            ref = _gi_from_counts(r.get("counts") or {}, r.get("cold_counts") or {},
                                  r.get("agent_z_max"), r.get("reference_z_max"), call_id,
                                  engine="esda (sandbox)",
                                  detail=f"the run's G_Local used transform "
                                         f"{r.get('transform')!r}, star={r.get('star')}")
        elif method == "morans_i":
            ref = Reference("Moran's I", call_id, "esda (sandbox)",
                            detail=f"the run's weights were transform {r.get('transform')!r}",
                            figures=[Figure("Moran's I", float(r["agent"]), float(r["reference"]),
                                            tol=5e-4)])
        else:
            continue
        if ref is not None and ref.figures:
            out.append(ref)
    return out


def _kriging_from_record(r: Dict[str, Any], shown: Sequence[Tuple[float, float]],
                         call_id: Optional[str]) -> Optional[Reference]:
    figs: List[Figure] = []
    for c in r.get("candidates") or []:
        try:
            a, ref = float(c["agent"]), float(c["reference"])
        except (KeyError, TypeError, ValueError):
            continue
        var = str(c.get("variable") or "the variable")
        # Both values are exact float64 computations from the same inputs (the run's is its own
        # weights times the vector, not what it printed), so only solver noise separates
        # them when the run is right. The nugget-on-the-diagonal bug moves a log value by
        # about 0.1%.
        tol = 1e-5 * max(abs(ref), 1e-9)
        if _was_shown(a, shown):
            figs.append(Figure(f"ordinary kriging of {var}", a, ref, tol))
        if abs(a) < 50 and abs(ref) < 50:
            ea, er = math.exp(a), math.exp(ref)
            if _was_shown(ea, shown):
                figs.append(Figure(f"exp() of ordinary kriging of {var}", ea, er,
                                   1e-5 * max(abs(er), 1e-9)))
    if not figs:
        return None
    params = r.get("pykrige_parameters") or {}
    pretty = ", ".join(f"{k} {float(v):.6g}" for k, v in params.items()
                       if isinstance(v, (int, float)))
    target = r.get("target") or []
    detail = (f"{r.get('model')} variogram ({pretty}) as the run's own matrix implies, at "
              f"({', '.join(f'{t:.6g}' for t in target)})")
    diag = r.get("diagonal_max")
    if isinstance(diag, (int, float)) and diag > 1e-12:
        detail += (f"; the run's semivariance matrix has {diag:.6g} on its diagonal, where "
                   f"gamma(0) must be 0")
    return Reference("ordinary kriging", call_id, str(r.get("engine") or "pykrige"),
                     detail=detail, figures=figs)


def _gi_from_counts(counts: Dict[str, Any], cold: Dict[str, Any], zmax_a: Any, zmax_r: Any,
                    call_id: Optional[str], *, engine: str, detail: str) -> Reference:
    figs: List[Figure] = []
    for t in THRESHOLDS:
        pair = counts.get(str(t))
        if isinstance(pair, (list, tuple)) and len(pair) == 2 and pair[0] is not None:
            figs.append(Figure(f"cells with Gi* z > {t}", float(pair[0]), float(pair[1]), 0.0,
                               integer=True, context=(t,)))
        pair = cold.get(str(t))
        if isinstance(pair, (list, tuple)) and len(pair) == 2 and pair[0] is not None:
            figs.append(Figure(f"cells with Gi* z < -{t}", float(pair[0]), float(pair[1]), 0.0,
                               integer=True, context=(-t,)))
    if isinstance(zmax_a, (int, float)) and isinstance(zmax_r, (int, float)):
        figs.append(Figure("the largest Gi* z", float(zmax_a), float(zmax_r), 1e-3))
    return Reference("Getis-Ord Gi*", call_id, engine, detail=detail, figures=figs)


# --------------------------------------------------------------------------- the built-in tools

def _default_resolver(ref: str) -> Any:
    from agent_runtime.langchain_geo_tools import _resolve

    return _resolve(ref)[0]


def _read_layer(file_id: str, resolve: Resolver):
    import geopandas as gpd

    return gpd.read_file(str(resolve(file_id)))


def _usable(gdf: Any, column: str):
    """The rows the spatial-statistics tools keep: a numeric value and a geometry."""
    import pandas as pd

    frame = gdf.copy()
    frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty]
    frame = frame.dropna(subset=[column])
    return frame.reset_index(drop=True)


def _weights(frame: Any, args: Dict[str, Any]):
    from agent_runtime.analysis_spatial_stats_tools import _weights_for

    kind = str(args.get("weights") or "queen").lower()
    if kind not in ("queen", "rook", "knn"):
        return None                      # distance and kernel weights have no binary reference
    cm = _weights_for(frame, kind, int(args.get("k") or 6), args.get("threshold_km"), "b", [])
    return cm


def _tool_gi(args: Dict[str, Any], body: Dict[str, Any], call_id: Optional[str],
             resolve: Resolver) -> Optional[Reference]:
    import warnings

    import esda
    import numpy as np

    column = args.get("column")
    if not column or not args.get("file_id"):
        return None
    frame = _usable(_read_layer(args["file_id"], resolve), column)
    if not 3 <= len(frame) <= MAX_FEATURES:
        return None
    # The tool says which transform it used. A result that names none predates stage 55, when
    # the tool row-standardised every Gi*. A non-binary transform the call ASKED for is the
    # caller's choice, not a method error, and there is no binary figure to hold it to.
    used = str(body.get("transform") or "R").upper()
    asked = str(args.get("transform") or "").strip().upper()
    if asked and asked != "B":
        return None
    cm = _weights(frame, args)
    if cm is None:
        return None
    star = bool(args.get("star", True))
    with cm as w, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = frame[column].to_numpy(dtype="float64")
        g = esda.G_Local(y, w, transform="B", star=star, permutations=0)
    z_ref = np.asarray(g.Zs, dtype=float).ravel()
    z_tool = _tool_z(body, resolve, len(frame))
    counts, cold = {}, {}
    hottest = [r.get("gi_z") for r in body.get("hottest") or [] if isinstance(r, dict)]
    coldest = [r.get("gi_z") for r in body.get("coldest") or [] if isinstance(r, dict)]
    zc = body.get("z_counts") if isinstance(body.get("z_counts"), dict) else {}
    for t in THRESHOLDS:
        if z_tool is not None:
            counts[str(t)] = [int(np.sum(z_tool > t)), int(np.sum(z_ref > t))]
            cold[str(t)] = [int(np.sum(z_tool < -t)), int(np.sum(z_ref < -t))]
            continue
        # Since stage 55 the result counts them itself.
        if isinstance(zc.get(f"z > {t}"), int) and isinstance(zc.get(f"z < -{t}"), int):
            counts[str(t)] = [zc[f"z > {t}"], int(np.sum(z_ref > t))]
            cold[str(t)] = [zc[f"z < -{t}"], int(np.sum(z_ref < -t))]
            continue
        # Without the per-area layer, the top and bottom ten still give an exact count when
        # the tenth is on the other side of the threshold.
        hz = [float(v) for v in hottest if isinstance(v, (int, float))]
        if hz and (len(hz) < 10 or hz[-1] <= t):
            counts[str(t)] = [sum(1 for v in hz if v > t), int(np.sum(z_ref > t))]
        cz = [float(v) for v in coldest if isinstance(v, (int, float))]
        if cz and (len(cz) < 10 or cz[-1] >= -t):
            cold[str(t)] = [sum(1 for v in cz if v < -t), int(np.sum(z_ref < -t))]
    zmax_tool = (float(np.nanmax(z_tool)) if z_tool is not None
                 else (max(hottest) if hottest and all(isinstance(v, (int, float))
                                                       for v in hottest) else None))
    return _gi_from_counts(counts, cold, zmax_tool, float(np.nanmax(z_ref)), call_id,
                           engine="esda",
                           detail=(("local_getis_ord used binary weights"
                                    if used == "B" else
                                    "local_getis_ord row-standardised its weights"
                                    if used == "R" else
                                    f"local_getis_ord used transform {used!r}")
                                   + "; the reference uses binary weights with the focal area "
                                     "included"))


def _tool_z(body: Dict[str, Any], resolve: Resolver, n: int):
    """The tool's per-area z-scores, from the layer it wrote, when it can be read."""
    import numpy as np

    fid = body.get("file_id")
    if not fid:
        return None
    try:
        layer = _read_layer(fid, resolve)
        z = layer["gi_z"].to_numpy(dtype=float)
        return z if z.size == n else None
    except Exception:  # noqa: BLE001
        return None


def _tool_moran(args: Dict[str, Any], body: Dict[str, Any], call_id: Optional[str],
                resolve: Resolver) -> Optional[Reference]:
    import warnings

    import esda

    stat = ((body.get("results") or {}).get("morans_i") or {}).get("statistic")
    column = args.get("column")
    if stat is None or not column or not args.get("file_id"):
        return None
    frame = _usable(_read_layer(args["file_id"], resolve), column)
    if not 3 <= len(frame) <= MAX_FEATURES:
        return None
    cm = _weights(frame, args)
    if cm is None:
        return None
    with cm as w, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w.transform = "r"
        m = esda.Moran(frame[column].to_numpy(dtype="float64"), w, permutations=0)
    return Reference("Moran's I", call_id, "esda", detail="row-standardised weights",
                     figures=[Figure("Moran's I", float(stat), float(m.I), 5e-4)])


def _ground_spacing(src: Any) -> Optional[Tuple[float, float]]:
    """Metres per pixel along x and y: from the CRS's own unit when projected, geodesically at
    the raster's centre when geographic, None when the CRS is unknown."""
    t = src.transform
    crs = src.crs
    if crs is None:
        return None
    if crs.is_geographic:
        from pyproj import Geod

        b = src.bounds
        lat, lon = (b.bottom + b.top) / 2.0, (b.left + b.right) / 2.0
        g = Geod(ellps="WGS84")
        dx = g.inv(lon, lat, lon + abs(t.a), lat)[2]
        dy = g.inv(lon, lat, lon, lat + abs(t.e))[2]
        return dx, dy
    factor = float(crs.linear_units_factor[1]) if crs.linear_units_factor else 1.0
    return abs(t.a) * factor, abs(t.e) * factor


def horn_slope_degrees(z: Any, dx: float, dy: float):
    """Horn (1981): the 3x3 weighted difference. Edges are NaN, and so is any pixel with a
    missing neighbour."""
    import numpy as np

    z = np.asarray(z, dtype=float)
    out = np.full(z.shape, np.nan)
    if z.shape[0] < 3 or z.shape[1] < 3:
        return out
    a, b, c = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    d, f = z[1:-1, :-2], z[1:-1, 2:]
    g, h, i = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * dx)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8.0 * dy)
    out[1:-1, 1:-1] = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    return out


def _tool_slope(args: Dict[str, Any], body: Dict[str, Any], call_id: Optional[str],
                resolve: Resolver) -> Optional[Reference]:
    import numpy as np
    import rasterio

    if str(args.get("kind") or "slope").lower() != "slope" or body.get("mean") is None:
        return None
    rid = args.get("raster_file_id")
    if not rid:
        return None
    with rasterio.open(str(resolve(rid))) as src:
        if src.width * src.height > MAX_PIXELS:
            return None
        spacing = _ground_spacing(src)
        if spacing is None:
            return None
        z = src.read(1, masked=True).astype("float64").filled(np.nan)
        nod = src.nodata
    if nod is not None and np.isfinite(nod):
        z[z == nod] = np.nan
    z[z < -1e5] = np.nan                  # the 3DEP sentinel the tools also sweep
    s = horn_slope_degrees(z, *spacing)
    finite = s[np.isfinite(s)]
    if not finite.size:
        return None
    ref_mean = float(finite.mean())
    # Horn's kernel smooths and central differences do not: on real terrain the two means
    # differ by a few percent, so only a difference a method choice cannot explain counts.
    return Reference("slope", call_id, "numpy (Horn 1981)",
                     detail=f"ground spacing {spacing[0]:.4g} x {spacing[1]:.4g} m",
                     figures=[Figure("mean slope (degrees)", float(body["mean"]), ref_mean,
                                     max(0.15 * ref_mean, 0.2))])


def _tool_zonal(args: Dict[str, Any], body: Dict[str, Any], call_id: Optional[str],
                resolve: Resolver) -> Optional[Reference]:
    import numpy as np
    import rasterio
    from rasterio.features import geometry_mask

    rid, pid = args.get("raster_file_id"), args.get("polygons_file_id")
    gj = (body.get("geojson") or {}).get("file_id") if isinstance(body.get("geojson"), dict) \
        else None
    prefix = str(args.get("prefix") or "value")
    if not (rid and pid and gj):
        return None
    try:
        out_layer = _read_layer(gj, resolve)
    except Exception:  # noqa: BLE001
        return None
    col = f"{prefix}_mean"
    if col not in out_layer.columns or len(out_layer) > MAX_FEATURES:
        return None
    zones = _read_layer(pid, resolve)
    with rasterio.open(str(resolve(rid))) as src:
        if src.width * src.height > MAX_PIXELS:
            return None
        band = int(args.get("band") or 1)
        vals = src.read(band, masked=True).astype("float64").filled(np.nan)
        if src.nodata is not None and np.isfinite(src.nodata):
            vals[vals == src.nodata] = np.nan
        vals[vals < -1e5] = np.nan
        transform, shape, rcrs = src.transform, (src.height, src.width), src.crs
    if rcrs is not None and zones.crs is not None and zones.crs != rcrs:
        zones = zones.to_crs(rcrs)
    figs: List[Figure] = []
    for i, geom in enumerate(zones.geometry):
        if i >= len(out_layer) or len(figs) >= 50 or geom is None or geom.is_empty:
            continue
        reported = out_layer[col].iloc[i]
        if reported is None or not np.isfinite(float(reported)):
            continue
        inside = ~geometry_mask([geom], out_shape=shape, transform=transform,
                                all_touched=bool(args.get("all_touched", True)))
        v = vals[inside]
        v = v[np.isfinite(v)]
        if not v.size:
            continue
        ref = float(v.mean())
        figs.append(Figure(f"zone {i + 1} mean", float(reported), ref,
                           max(1e-3 * abs(ref), 1e-4)))
    if not figs:
        return None
    return Reference("zonal mean", call_id, "numpy + rasterio mask",
                     detail="each zone masked on its own, in the raster's CRS", figures=figs)


_TOOLS = {
    "local_getis_ord": _tool_gi,
    "global_spatial_autocorrelation": _tool_moran,
    "terrain_derivative": _tool_slope,
    "zonal_stats_for_raster": _tool_zonal,
}


def from_tool(name: str, args: Any, content: Any, call_id: Optional[str] = None,
              resolve: Optional[Resolver] = None) -> Optional[Reference]:
    fn = _TOOLS.get(str(name or ""))
    if fn is None:
        return None
    body = _body(content)
    if not body.get("ok", True):
        return None
    try:
        return fn(_args(args), body, call_id, resolve or _default_resolver)
    except Exception:  # noqa: BLE001 - a reference that cannot run says nothing
        return None


def for_results(results: Iterable[Dict[str, Any]], calls: Iterable[Dict[str, Any]] = (),
                resolve: Optional[Resolver] = None) -> List[Reference]:
    """Every recomputation for a turn: `results` are `{tool, call_id, content}` (a turn log's
    tool_result events), `calls` give each call's args."""
    args_by_call = {c.get("call_id"): c.get("args") for c in calls or [] if c.get("call_id")}
    out: List[Reference] = []
    for res in results or []:
        if not isinstance(res, dict) or res.get("ok") is False or res.get("memo_of"):
            continue
        name = str(res.get("tool") or res.get("name") or "")
        cid = res.get("call_id") or res.get("tool_call_id")
        content = res.get("content")
        if name == "execute_code":
            out.extend(from_sandbox(content, cid))
        elif name in _TOOLS:
            ref = from_tool(name, args_by_call.get(cid), content, cid, resolve)
            if ref is not None:
                out.append(ref)
    return out


def for_log(log: Any, resolve: Optional[Resolver] = None) -> List[Reference]:
    if log is None:
        return []
    try:
        return for_results(log.results(), log.calls(), resolve)
    except Exception:  # noqa: BLE001
        return []


# --------------------------------------------------------------------------- what the turn sees

def _show(v: float, integer: bool = False) -> str:
    return str(int(round(v))) if integer else f"{v:,.6g}"


def reconcile_note(refs: Sequence[Reference]) -> str:
    bad = [r for r in refs if r.disagreeing()]
    if not bad:
        return ""
    lines = ["A reference library recomputed a standard method from the run's own inputs and "
             "got a DIFFERENT result:"]
    for r in bad[:4]:
        figs = r.disagreeing()[:3]
        pairs = "; ".join(f"{f.label}: the run {_show(f.reported, f.integer)}, the reference "
                          f"{_show(f.reference, f.integer)}" for f in figs)
        lines.append(f"- {r.method} ({r.engine}): {pairs}. {r.detail}.")
    lines.append("Report the reference value unless the user asked for the variant the run "
                 "computed, and say which one you report.")
    return "\n".join(lines)


def _sentences(answer: str) -> List[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n", answer or "") if s.strip()]


_NUM_RE = re.compile(r"(?<![\w.])[-−]?\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> List[float]:
    out = []
    for m in _NUM_RE.finditer(text or ""):
        try:
            out.append(float(m.group(0).replace(",", "").replace("−", "-")))
        except ValueError:
            continue
    return out


def _stated(answer: str, value: float, fig: Figure) -> bool:
    if fig.integer:
        for s in _sentences(answer):
            nums = _numbers(s)
            if any(abs(n - c) < 1e-9 for n in nums for c in fig.context) and \
                    any(abs(n - value) < 1e-9 for n in nums):
                return True
        return False
    from agent_runtime import facts as F

    target = F.Fact(id="reference", value=value, source="reference")
    # A claim only: "the tool returned slope 0" restates the tool, it is not the answer's figure.
    return any(F._matches(q, target) for q in F.quantities(answer or "")
               if F.is_claim(q) and (not q.unit or F._digits_shown(q.text) >= 3))


def _count_beside(answer: str, fig: Figure) -> Optional[float]:
    """A count the answer states next to the figure's threshold that is not the reference count:
    "15 cells are hot spots with Gi* z > 1.96" when the run printed a p-value class count and
    called it that (baseline T09, luna). None when there is none."""
    for s in _sentences(answer):
        nums = _numbers(s)
        if not any(abs(n - c) < 1e-9 for n in nums for c in fig.context):
            continue
        for n in nums:
            if n.is_integer() and n >= 0 and all(abs(n - c) > 1e-9 for c in fig.context) \
                    and abs(n - fig.reference) > 1e-9:
                return n
    return None


def findings(answer: str, refs: Sequence[Reference], *, kind: str) -> List[Any]:
    """A finding for each disagreeing figure the answer reports as the run computed it, without
    the reference value. A count is held to its threshold's sentence: any other count stated
    there is the run's, however the run arrived at it."""
    from agent_runtime import verdict as V

    out = []
    for r in refs:
        for f in r.disagreeing():
            if _stated(answer, f.reference, f):
                continue
            shown = f.reported if _stated(answer, f.reported, f) else (
                _count_beside(answer, f) if f.integer else None)
            if shown is None:
                continue
            out.append(V.Finding(
                CHECK, kind,
                f"{r.method}: the answer's {f.label} is {_show(shown, f.integer)}, but "
                f"recomputing it with {r.engine} from the same inputs gives "
                f"{_show(f.reference, f.integer)} ({r.detail})",
                evidence=[str(r.call_id)] if r.call_id else []))
            break                           # one finding per recomputation
    return out
