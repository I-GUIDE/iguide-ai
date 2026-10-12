"""What a run's declared numbers MEAN: unit, dimension, and where they were measured.

The sandbox gate reports each `IGUIDE_OUTPUTS` declaration as written, every metric operation
with the CRS its receiver was in, and the size of every frame (sandbox_verify.run_checks). This
module, in the agent process where a unit library is available, turns them into typed values:

    {name, value, unit, canonical, dimension, counted, measured_in_crs, source}

and into the findings that need units: a missing unit, a unit that does not parse, a count that
is fractional or negative or larger than every frame in the run, and a length or area declared
as measured in a geographic CRS. Nothing here reads a variable's or a column's NAME to decide
what a number is. Stage 43 deleted the vocabularies and name heuristics that did
(docs/design-review-2026-10.md, flaw 1).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from agent_runtime.units import ParsedUnit, parse_unit

PASS, FAIL, UNKNOWN = "pass", "fail", "cannot_determine"


def _finding(check: str, status: str, target: str, message: str, **extra: Any) -> Dict[str, Any]:
    return {"check": check, "status": status, "target": target, "message": message, **extra}


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f


def _crs_is_geographic(text: Any) -> Optional[bool]:
    try:
        from pyproj import CRS

        return bool(CRS.from_user_input(text).is_geographic)
    except Exception:  # noqa: BLE001 - an unreadable CRS is unknown, not geographic
        return None


def _measured_in(entry: Dict[str, Any], report: Dict[str, Any]) -> Tuple[Optional[Any], Optional[Dict]]:
    """Where a length or area was measured, and a finding when that is a geographic CRS.

    A CRS the declaration names wins: it is the run's statement about this number. Otherwise the
    operations decide: the projected CRSs metric operations ran in. With none, the number was
    computed some other way (a geodesic library, arithmetic on coordinates) and the gate's
    operation findings already cover the geopandas path.
    """
    name = entry.get("name")
    declared = entry.get("crs") or entry.get("measured_in")
    if declared:
        geo = _crs_is_geographic(declared)
        if geo:
            return declared, _finding(
                "measured_in", FAIL, str(name),
                f"declared as measured in {declared}, a GEOGRAPHIC CRS: a length or area "
                f"computed there is in degrees. Reproject before measuring.", crs=str(declared))
        return declared, None
    ops = [o for o in report.get("metric_ops") or [] if isinstance(o, dict)]
    projected = sorted({str(o.get("crs")) for o in ops if o.get("projected") and o.get("crs")})
    if len(projected) == 1:
        return projected[0], None
    return (projected or None), None


def evaluate(report: Dict[str, Any], code: Optional[str] = None
             ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(findings, typed outputs) for a sandbox report's declarations.

    *code* is the script that ran. Each declared value's KIND is read from it
    (agent_runtime/figure_kinds.py, stage 54): the operation that computed the value says whether
    it is a count and of what, a statistic, a measurement or a column's value. The declaration's
    unit is a hint; where the two disagree the computation wins. The checks that need a kind
    apply only when it is known:
    - a count is held to integers, non-negative, and to its frame's rows only when the code
      counted rows (a count of pixels, list items or labelled patches is not a frame's rows);
    - a length or area declared as measured in a geographic CRS fails only when the code
      measured it with a CRS-dependent operation.
    An unknown kind never fails: it is recorded on the typed output, and nothing is printed."""
    from agent_runtime import figure_kinds

    findings: List[Dict[str, Any]] = []
    outputs: List[Dict[str, Any]] = []
    sizes = {k: v for k, v in (report.get("frame_sizes") or {}).items() if isinstance(v, int)}
    analysis = figure_kinds.analyze(code) if code else None
    kinds = figure_kinds.declared_kinds(code)
    for entry in report.get("declared") or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name"))
        value = _number(entry.get("value"))
        if value is None:
            continue                       # a label (a CRS, a method name): not a measurement
        pu: ParsedUnit = parse_unit(entry.get("unit"))
        kind = kinds.get(name, figure_kinds.UNKNOWN_KIND)
        typed = {"name": name, "value": entry.get("value"), "unit": entry.get("unit"),
                 "canonical": pu.canonical, "dimension": pu.dimension, "counted": pu.counted,
                 "measured_in_crs": None, "source": "execute_code", "figure_kind": kind.to_dict()}
        if pu.kind == "missing":
            findings.append(_finding(
                "declared_units", FAIL, name,
                "unit is null: a number whose unit is unrecorded cannot be verified "
                "(25000 is right in metres, wrong in feet)"))
        elif pu.kind == "unparseable":
            # Loud, and not advisory: a unit nobody can read is a number nobody can check.
            findings.append(_finding(
                "declared_units", UNKNOWN, name,
                f"unit {entry.get('unit')!r} does not parse as a unit ({pu.error}); the number "
                f"cannot be checked", unit=str(entry.get("unit"))))
        else:
            # A plural word pint does not know names the things counted ("tracts", "schools");
            # a singular one names a dimensionless measure ("index", "ratio"). Grammar, not a
            # vocabulary: real units ("metres") were parsed by pint before this point.
            plural = pu.kind == "label" and str(pu.counted or "").lower().endswith("s") \
                and not str(pu.counted or "").lower().endswith("ss")
            declared_count = plural or pu.canonical == "count"
            # The declaration is a hint (stage 54): the count checks run on what the code
            # COUNTED. "NDVI (dimensionless)" parses as a count of NDVI, and np.nanmean(ndvi)
            # then failed as a fractional count; `len(patches)` and `valid.sum()` over a raster
            # were compared with frame row counts.
            counted = kind.is_count
            if declared_count and kind.known and not counted:
                typed["declared_as"] = "count"
                findings.append(_finding(
                    "declared_units", PASS, name,
                    f"declared as a count ({entry.get('unit')}), but the code computed "
                    f"{kind.describe()}: typed by the computation", unit=str(entry.get("unit"))))
            elif declared_count and not kind.known:
                findings.append(_finding(
                    "declared_units", PASS, name,
                    f"declared as a count ({entry.get('unit')}); how the code computed it is not "
                    f"known, so the count checks were not applied", unit=str(entry.get("unit"))))
            elif counted and not float(value).is_integer():
                findings.append(_finding("declared_units", FAIL, name,
                                         f"computed as {kind.describe()}, but the value is "
                                         f"fractional ({value})", unit=str(entry.get("unit"))))
            elif counted and value < 0:
                findings.append(_finding("declared_units", FAIL, name,
                                         f"a count of {pu.counted or 'things'} cannot be "
                                         f"negative ({value})", unit=str(entry.get("unit"))))
            else:
                findings.append(_finding(
                    "declared_units", PASS, name,
                    f"{value} {entry.get('unit')} "
                    f"({'a count of ' + pu.counted if pu.counted and pu.kind == 'label' else pu.dimension})",
                    unit=str(entry.get("unit"))))
            if counted and float(value).is_integer() and value >= 0:
                population = _count_population(name, value, kind, sizes, analysis)
                if population is not None:
                    findings.append(population)
            if pu.is_length or pu.is_area:
                crs, problem = _measured_in(entry, report)
                typed["measured_in_crs"] = crs
                # Only a figure the code MEASURED (.area, .length, .distance) depends on the
                # CRS it ran in; a geodesic distance or a value read from a column does not.
                if problem and kind.measured:
                    findings.append(problem)
        outputs.append(typed)
    return findings, outputs


def _count_population(name: str, value: float, kind: Any, sizes: Dict[str, int],
                      analysis: Any) -> Optional[Dict[str, Any]]:
    """A count of a frame's rows held to that frame's size, or None when the count is not of
    rows or the frame's size is not known.

    Until stage 54 every declared count was compared with EVERY frame and failed when it was
    larger than all of them: 239 patches (a list), 1,000,557 pixels (a raster mask) and 2,344
    residents (a column value) all "exceeded every frame". Now only `len(F)`, `F.shape[0]`, a
    mask over F's column and the like are compared, and with F's own size: the frame the rows
    were drawn from, when it keeps its rows for the whole run."""
    if not kind.is_rows or not sizes:
        return None
    for frame in (kind.within, kind.of):
        if not frame or frame not in sizes:
            continue
        if analysis is not None and not analysis.stable(frame):
            continue
        size = sizes[frame]
        if value > size:
            return _finding("count_population", FAIL, name,
                            f"{int(value)} is {kind.describe()}, but `{frame}` has {size} rows "
                            f"when the run ends, so the count cannot have been taken from it",
                            frames={frame: size})
        return _finding("count_population", PASS, name,
                        f"{int(value)} is {kind.describe()}, within the {size} rows of "
                        f"`{frame}`", frames={frame: size})
    return None


def merge_verdict(report: Dict[str, Any], extra: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The report's counts and verdict with *extra* findings folded in."""
    counts = dict(report.get("counts") or {})
    for f in extra:
        counts[f["status"]] = counts.get(f["status"], 0) + 1
    if counts.get(FAIL):
        verdict = FAIL
    elif counts.get(UNKNOWN):
        verdict = UNKNOWN
    elif counts.get(PASS):
        verdict = PASS
    else:
        verdict = report.get("verdict") or UNKNOWN
    return {"counts": counts, "verdict": verdict}
