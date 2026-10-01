"""Deterministic invariant checks that run INSIDE the sandbox, on the real objects.

Why in-sandbox rather than an AST pass. A source-text check can see that ``.buffer(25000)``
was called; it cannot see what CRS the frame was in when it happened, because that depends on
what the data actually loaded as. The failure this exists to catch —
``gdf.buffer(25000)`` on an EPSG:4326 frame, which silently buffers by 25000 *degrees* and
produces a number that looks like metres — is invisible to every static check and produces no
error. Only the live frame knows.

The module is executed as an epilogue appended to the user's code, and it must therefore be:

* **stdlib-only at import time** — geopandas/pandas are probed, never required, so a run that
  does not use them still gets its checks written;
* **incapable of failing the run** — every check is individually guarded, and the epilogue
  writes ``checks.json`` even when the checks themselves error. A verification step that can
  break a working analysis is worse than no verification;
* **explicit about not knowing.** Every check returns ``pass``, ``fail`` or
  ``cannot_determine``, and the third is reported, never silently treated as a pass. "The CRS
  is unknown" and "the CRS is correct" must not look the same to the reader.

Kept importable agent-side so it can be unit-tested directly; ``epilogue_source()`` returns
its own text for injection.
"""

from __future__ import annotations

import inspect
import json
import math
from types import ModuleType
from typing import Any, Dict, List, Optional

PASS = "pass"
FAIL = "fail"
UNKNOWN = "cannot_determine"

# Namespace-scoped signal for the coverage check; see run_checks.
_GEO_MODULES = frozenset({"geopandas", "rasterio", "shapely", "rioxarray", "xarray"})

CHECKS_FILENAME = "checks.json"
ENVIRONMENT_FILENAME = "environment.json"
DECLARED_FILENAME = "declared_outputs.json"


# --------------------------------------------------------------------------- #
# Individual checks. Each takes a live object and returns a finding dict.
# --------------------------------------------------------------------------- #

def _finding(check: str, status: str, target: str, message: str, **extra: Any) -> Dict[str, Any]:
    out = {"check": check, "status": status, "target": target, "message": message}
    out.update(extra)
    return out


def _crs_of(obj: Any) -> Any:
    try:
        return getattr(obj, "crs", None)
    except Exception:
        return None


def _is_projected(crs: Any) -> Optional[bool]:
    """True/False, or None when the CRS is absent or cannot be interpreted."""
    if crs is None:
        return None
    try:
        flag = getattr(crs, "is_projected", None)
        if isinstance(flag, bool):
            return flag
    except Exception:
        pass
    text = str(crs).strip().lower()
    if not text:
        return None
    # EPSG:4326 and friends are the common geographic cases; anything else is a guess, and a
    # guess must surface as cannot_determine rather than a confident pass.
    if "4326" in text or "epsg:4269" in text or "wgs 84" in text or "wgs84" in text:
        return False
    return None


def _crs_unit(crs: Any) -> Optional[str]:
    """The CRS's linear axis unit, lowercased, or None when unavailable."""
    try:
        info = getattr(crs, "axis_info", None) or []
        for axis in info:
            unit = getattr(axis, "unit_name", None)
            if isinstance(unit, str) and unit.strip():
                return unit.strip().lower()
    except Exception:
        pass
    return None


_UNIT_ALIASES = {
    "metre": "metres", "meter": "metres", "metres": "metres", "meters": "metres", "m": "metres",
    "us survey foot": "feet", "foot": "feet", "feet": "feet", "ft": "feet",
    "kilometre": "kilometres", "kilometer": "kilometres", "km": "kilometres",
    # AREAL units. Their absence was found by a live run: an agent declared `km²` for a buffer
    # area — correctly — and the gate answered "unrecognised unit 'km²'; not checked". A unit
    # the system asked for, got, and then could not read is worse than not asking.
    "m2": "square_metres", "m^2": "square_metres", "m²": "square_metres",
    "sq m": "square_metres", "sqm": "square_metres",
    "square metre": "square_metres", "square meter": "square_metres",
    "square metres": "square_metres", "square meters": "square_metres",
    "square_metres": "square_metres", "square_meters": "square_metres",
    "km2": "square_kilometres", "km^2": "square_kilometres", "km²": "square_kilometres",
    "sq km": "square_kilometres", "sqkm": "square_kilometres",
    "square kilometre": "square_kilometres", "square kilometer": "square_kilometres",
    "square kilometres": "square_kilometres", "square kilometers": "square_kilometres",
    "square_kilometres": "square_kilometres", "square_kilometers": "square_kilometres",
    # COUNTS. `count` was in the known set and `records` was not, so a live run declaring
    # {"value": 27824, "unit": "records"} — the natural word for what it was counting — came back
    # "unrecognised unit 'records'; not checked", and that single UNKNOWN downgraded a correct
    # answer to unverified. The vocabulary has to cover how the number is actually described, not
    # only the token we would have chosen.
    "record": "count", "records": "count", "row": "count", "rows": "count",
    "counts": "count", "n": "count", "number": "count", "observation": "count",
    "observations": "count", "feature": "count", "features": "count",
    "item": "count", "items": "count", "event": "count", "events": "count",
    "incident": "count", "incidents": "count", "occurrence": "count", "occurrences": "count",
    "hectare": "hectares", "hectares": "hectares", "ha": "hectares",
    "acre": "acres", "acres": "acres",
    "degree": "degrees", "degrees": "degrees", "deg": "degrees", "°": "degrees",
}


def _unit_matches(declared: Any, actual: Optional[str]) -> Optional[bool]:
    """True/False, or None when either side is unknown."""
    want = _UNIT_ALIASES.get(str(declared or "").strip().lower())
    got = _UNIT_ALIASES.get(str(actual or "").strip().lower())
    if not want or not got:
        return None
    return want == got


# Column names that mean "a distance or area was computed and stored here". A geographic frame
# carrying one of these is evidence that the number in it is in degrees.
_METRIC_COLUMN_HINTS = ("area", "length", "dist", "perimeter", "buffer", "radius", "km", "_m",
                        "acre", "hectare", "sqm", "sq_")


def _has_metric_column(frame: Any) -> Optional[str]:
    """A numeric column whose NAME says it holds a measurement, or None."""
    try:
        columns = list(frame.columns)
    except Exception:
        return None
    for col in columns:
        low = str(col).lower()
        if not any(h in low for h in _METRIC_COLUMN_HINTS):
            continue
        try:
            if frame[col].dtype.kind in "iuf":
                return str(col)
        except Exception:
            continue
    return None


def check_projected_crs(name: str, frame: Any) -> Dict[str, Any]:
    """A distance/area/buffer result is only meaningful in a PROJECTED CRS.

    The motivating replay: a 25 km buffer requested on an EPSG:4326 frame produced a figure
    reported as 21.5 km. No exception, no warning — just a wrong number with a plausible
    magnitude.
    """
    crs = _crs_of(frame)
    if crs is None:
        return _finding("projected_crs", UNKNOWN, name,
                        "frame has no CRS set, so distance/area results cannot be trusted")
    projected = _is_projected(crs)
    if projected is None:
        return _finding("projected_crs", UNKNOWN, name,
                        f"could not determine whether {crs!s} is projected", crs=str(crs))
    if projected:
        return _finding("projected_crs", PASS, name, f"projected CRS {crs!s}", crs=str(crs))

    # Geographic. Whether that is WRONG depends on what was computed from it, and a scan of
    # module scope cannot see that — which is exactly why the call-time contract guard exists.
    #
    # Failing on the mere presence of a 4326 frame fails the STANDARD CORRECT WORKFLOW: data
    # arrives in 4326 and you reproject it, so the input frame is still bound when the run ends.
    # Observed live — an agent reprojected to EPSG:32616, buffered correctly, produced areas
    # accurate to 0.16% of the analytic value, and the answer was stamped "⛔ invariant check
    # FAILED, numeric results are not verified". A ⛔ on a correct answer teaches the reader to
    # ignore ⛔.
    metric_column = _has_metric_column(frame)
    detail = (f" This frame holds a computed measurement ({metric_column!r}), which is therefore "
              f"in degrees." if metric_column else "")
    # Still a FAIL on its own. `gdf.buffer(25000)` on a 4326 frame produces a wrong GEOMETRY with
    # no numeric column at all, so keying the verdict on a measurement column would miss the
    # motivating case entirely. `run_checks` relaxes this only on positive evidence that the run
    # reprojected before measuring — see there.
    return _finding("projected_crs", FAIL, name,
                    f"{crs!s} is GEOGRAPHIC: distances and areas computed from this frame are "
                    f"in degrees, not metres.{detail} Reproject (e.g. .to_crs(3857) or a local "
                    f"UTM zone) before buffering or measuring.",
                    crs=str(crs), metric_column=metric_column)


def _looks_like_join_result(frame: Any) -> bool:
    """Whether this frame carries the fingerprint of a spatial/relational join."""
    try:
        columns = {str(c) for c in frame.columns}
    except Exception:
        return False
    return bool(columns & {"index_right", "index_left"}) or any(
        str(c).endswith(("_left", "_right")) for c in columns)


def check_not_all_nan(name: str, frame: Any) -> Dict[str, Any]:
    """An entirely-null column is a failed join or a failed parse wearing a result's shape.

    Checks EVERY column, not just ``select_dtypes("number")``. That was the first version and
    it missed the commonest case: pandas types an all-``None`` column as ``object``, so the
    column produced by an unmatched join — the exact thing this check exists for — was
    excluded from the check by its own dtype.

    **FAIL requires join evidence.** An all-null column is only provably a defect when the frame
    is a join result; in an INPUT it is ordinary data — a dataset with an optional column that
    happens to be empty (``apt_number``, ``middle_name``) is not broken. Reproduced: a correct
    run over such a frame was verdicted ``fail`` and blocked. Without that distinction the check
    fails correct runs, and a gate that fails correct runs gets switched off, which costs the
    unmatched-join detection this exists for.

    An unverifiable suspicion is still REPORTED — ``cannot_determine`` carries its message all
    the way to the user's caveat — it just does not claim the numbers are wrong.
    """
    try:
        columns = list(frame.columns)
    except Exception:
        return _finding("all_nan", UNKNOWN, name, "columns could not be inspected")
    if not columns:
        return _finding("all_nan", UNKNOWN, name, "frame has no columns")
    if len(frame) == 0:
        # NOT a fail. A filter that matches nothing and a spatial query with no hits are both
        # correct outcomes with zero rows, and calling them errors blocks a right answer. It is
        # still worth saying out loud, because an empty frame is also what a broken filter
        # produces, and the reader is the one who can tell.
        return _finding("all_nan", UNKNOWN, name,
                        "frame is empty (0 rows) — correct for a filter or query that matched "
                        "nothing, but also what a failed filter produces; confirm which")
    geom = _geometry_column(frame)
    bad: List[str] = []
    checked: List[str] = []
    for col in columns:
        if geom is not None and str(col) == geom:
            continue          # a null geometry is its own problem, not a null-column one
        if geom is None and str(col) == "geometry":
            continue
        try:
            checked.append(str(col))
            if bool(frame[col].isna().all()):
                bad.append(str(col))
        except Exception:
            continue
    if bad:
        if _looks_like_join_result(frame):
            return _finding("all_nan", FAIL, name,
                            f"column(s) entirely null: {', '.join(bad)} — this frame is a join "
                            f"result, so an all-null column means nothing matched; any count or "
                            f"ratio computed from it is wrong", columns=bad)
        if len(bad) == len(checked):
            return _finding("all_nan", FAIL, name,
                            f"EVERY non-geometry column is entirely null ({', '.join(bad)}), so "
                            f"an upstream step produced no data at all", columns=bad)
        # Some columns null, no join evidence, others populated. RECORDED but not alarming: a
        # dataset with an empty optional column (`apt_number`, `middle_name`) is ordinary, and a
        # correct run over one must be able to reach `pass`.
        #
        # This deliberately matches the call-site rule in `_check_one_arg`. Two different answers
        # to the same question in one module is how the confusion this check keeps causing starts.
        return _finding("all_nan", PASS, name,
                        f"populated, though {len(bad)} of {len(checked)} column(s) are entirely "
                        f"null ({', '.join(bad)}) — ordinary for an optional field, but check it "
                        f"if a number was computed from one of them", columns=bad)
    return _finding("all_nan", PASS, name, "no entirely-null columns")


def check_join_cardinality(name: str, frame: Any) -> Optional[Dict[str, Any]]:
    """Report what a spatial join actually did, so silent row inflation is visible.

    Never a FAIL on its own — duplication can be correct. It is reported so a count computed
    from the result can be judged, because a many-to-many join that quietly triples the rows
    turns 'incidents per area' into a number nobody can reproduce.
    """
    try:
        rows = int(len(frame))
    except Exception:
        return None
    detail: Dict[str, Any] = {"rows": rows}
    for col in ("index_right", "index_left"):
        if col in getattr(frame, "columns", []):
            try:
                detail["unmatched"] = int(frame[col].isna().sum())
                detail["duplicated_left"] = int(rows - frame.index.nunique())
            except Exception:
                pass
            return _finding("join_cardinality", PASS, name,
                            f"join result: {rows} rows, "
                            f"{detail.get('unmatched', '?')} unmatched, "
                            f"{detail.get('duplicated_left', '?')} duplicated index entries",
                            **detail)
    # Not a join result: report NOTHING rather than a cannot_determine. An "unknown" per
    # ordinary frame buries the findings that matter — the first real run emitted four
    # cannot_determine lines and two genuine failures, and the noise dominated.
    return None


def check_finite(name: str, value: Any) -> Dict[str, Any]:
    """A reported scalar must be a real number."""
    try:
        f = float(value)
    except Exception:
        return _finding("finite_value", UNKNOWN, name, "not a numeric scalar")
    if math.isnan(f) or math.isinf(f):
        return _finding("finite_value", FAIL, name, f"value is {f}")
    return _finding("finite_value", PASS, name, f"{f}")


# Names a run is expected to declare when it reports a number the answer will quote. The
# convention is cheap on the model's side and it is the only way a *unit* can be checked at
# all: 21500 is correct in metres and wrong in feet, and no amount of frame inspection can
# distinguish them.
DECLARED_OUTPUTS = "IGUIDE_OUTPUTS"

_KNOWN_UNITS = {"metres", "meters", "m", "kilometres", "kilometers", "km", "feet", "ft",
                "miles", "mi", "degrees", "deg", "count", "percent", "%", "ratio",
                "square_metres", "m2", "square_kilometres", "km2", "hectares", "acres",
                "index", "none", "dimensionless"}



def _count_finding(target: str, value: Any, unit: Any) -> Dict[str, Any]:
    """A declared count must be a non-negative whole number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return _finding("declared_units", UNKNOWN, target,
                        f"unit {unit} but the value is {type(value).__name__}, not a number",
                        unit=str(unit))
    try:
        if isinstance(value, float) and not float(value).is_integer():
            return _finding("declared_units", FAIL, target,
                            f"declared as a count but the value is fractional ({value})",
                            unit=str(unit))
        if value < 0:
            return _finding("declared_units", FAIL, target,
                            f"declared as a count but the value is negative ({value})",
                            unit=str(unit))
    except Exception:                                       # pragma: no cover - defensive
        return _finding("declared_units", UNKNOWN, target, f"unit {unit}; value not comparable",
                        unit=str(unit))
    return _finding("declared_units", PASS, target,
                    f"unit {unit}: a non-negative whole count ({int(value)})", unit=str(unit))


def check_count_population(outputs: Any, namespace: Dict[str, Any],
                           *, max_frames: int = 12) -> List[Dict[str, Any]]:
    """Report the population each declared count could have come from, and fail an impossible one.

    Motivated by a live run that answered a question about a 128,886-record dataset with counts
    computed from a 49,789-row spatially-joined subset — reporting THEFT as 9,993 where the file
    says 27,824. Every individual number was real; the POPULATION was different from the one the
    question named, and nothing in the report made that visible.

    This does not guess which frame is "the" population — that would be a false-positive
    generator. It records the frame sizes present, which is what lets a reader see 9,993-of-49,789
    and ask the right question, and it FAILS only the case that is impossible on any reading: a
    count larger than every frame in the run.
    """
    if not isinstance(outputs, dict) or not outputs:
        return []
    sizes: Dict[str, int] = {}
    for name, obj in list(namespace.items()):
        if name.startswith("_") or len(sizes) >= max_frames:
            continue
        try:
            if _looks_like_frame(obj):
                sizes[name] = int(len(obj))
        except Exception:
            continue
    if not sizes:
        return []

    largest = max(sizes.values())
    findings: List[Dict[str, Any]] = []
    for key, spec in list(outputs.items())[:24]:
        if not isinstance(spec, dict):
            continue
        unit = str(spec.get("unit") or "").strip().lower()
        if unit != "count" and _UNIT_ALIASES.get(unit) != "count":
            continue
        value = spec.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        summary = ", ".join(f"{n}={c}" for n, c in sorted(sizes.items())[:6])
        if value > largest:
            findings.append(_finding(
                "count_population", FAIL, str(key),
                f"declared count {int(value)} exceeds every frame in this run ({summary}), so it "
                f"cannot have been counted from any of them", frames=sizes))
        else:
            findings.append(_finding(
                "count_population", PASS, str(key),
                f"count {int(value)} is within the run's frames ({summary}) — confirm this is the "
                f"population the question asked about", frames=sizes))
    return findings


def check_declared_units(outputs: Any) -> List[Dict[str, Any]]:
    """Every numeric output the run declares must carry a unit and be in a plausible range.

    ``IGUIDE_OUTPUTS`` is expected to look like::

        IGUIDE_OUTPUTS = {"buffer_radius": {"value": 25000, "unit": "metres"},
                          "areas_covered": {"value": 77, "unit": "count",
                                            "min": 0, "max": 100}}

    A ``unit`` of ``None`` is a FAIL rather than an omission: the plan's rule is that a null
    unit blocks "verified", because the number most likely to be wrong is exactly the one whose
    unit nobody wrote down. ``min``/``max`` are optional and only checked when given — an
    invented plausible range would be a false positive generator.
    """
    findings: List[Dict[str, Any]] = []
    if outputs is None:
        return findings
    if not isinstance(outputs, dict):
        return [_finding("declared_units", UNKNOWN, DECLARED_OUTPUTS,
                         f"expected a dict, got {type(outputs).__name__}")]
    if not outputs:
        return findings
    for key, spec in list(outputs.items())[:24]:
        target = str(key)
        if not isinstance(spec, dict):
            findings.append(check_finite(target, spec))
            findings.append(_finding("declared_units", FAIL, target,
                                     "declared without a unit — give "
                                     "{'value': x, 'unit': 'metres'}"))
            continue
        value = spec.get("value")
        findings.append(check_finite(target, value))
        unit = spec.get("unit")
        if unit is None or str(unit).strip() == "":
            findings.append(_finding("declared_units", FAIL, target,
                                     "unit is null: a number whose unit is unrecorded cannot "
                                     "be verified (25000 is right in metres, wrong in feet)"))
        elif (str(unit).strip().lower() not in _KNOWN_UNITS
              and _UNIT_ALIASES.get(str(unit).strip().lower()) is None):
            # Checked against the ALIAS table as well as the literal set. `km²` — which is how a
            # model actually writes it, observed live — was in neither, so a correctly declared
            # unit came back "unrecognised; not checked" and downgraded the whole run.
            findings.append(_finding("declared_units", UNKNOWN, target,
                                     f"unrecognised unit {unit!r}; not checked", unit=str(unit)))
        elif _UNIT_ALIASES.get(str(unit).strip().lower()) == "count" or \
                str(unit).strip().lower() == "count":
            # A count is the one unit whose VALUE the gate can judge on its own: a negative or
            # fractional count is wrong whatever produced it. Recognising the unit and then not
            # checking it is how "unit count" passed for a value of -3.
            findings.append(_count_finding(target, value, unit))
        else:
            findings.append(_finding("declared_units", PASS, target, f"unit {unit}",
                                     unit=str(unit)))
        lo, hi = spec.get("min"), spec.get("max")
        try:
            f = float(value)
        except Exception:
            continue
        if lo is not None and f < float(lo):
            findings.append(_finding("output_bounds", FAIL, target,
                                     f"{f} is below the declared minimum {lo}"))
        elif hi is not None and f > float(hi):
            findings.append(_finding("output_bounds", FAIL, target,
                                     f"{f} is above the declared maximum {hi}"))
        elif lo is not None or hi is not None:
            findings.append(_finding("output_bounds", PASS, target,
                                     f"{f} within [{lo}, {hi}]"))
    return findings



# --------------------------------------------------------------------------- #
# Contract enforcement. The extracted unit says what it needs; this checks it.
# --------------------------------------------------------------------------- #

CONTRACTS_GLOBAL = "IGUIDE_CONTRACTS"
VIOLATIONS_GLOBAL = "_IGUIDE_CONTRACT_VIOLATIONS"
# Metric operations geopandas ran on a GEOGRAPHIC receiver during the run, recorded by
# install_operation_tracker; OP_TRACKING_GLOBAL says the tracker was live, which is what lets
# run_checks trust the ABSENCE of a record.
GEOGRAPHIC_OPS_GLOBAL = "_IGUIDE_GEOGRAPHIC_OPS"
OP_TRACKING_GLOBAL = "_IGUIDE_OP_TRACKING"
# The operations whose result is a distance, length or area, or is built from one. geopandas warns
# on these (and on `centroid`, which is a location and is not counted). buffer(0), the
# geometry-repair idiom, does not warn.
_METRIC_OPS = ("area", "length", "buffer", "distance", "dwithin", "hausdorff_distance",
               "frechet_distance", "interpolate", "sjoin_nearest")
_GEOGRAPHIC_WARNING = "Geometry is in a geographic CRS. Results from '"


def check_contract_arg(unit: str, invariant: Dict[str, Any], value: Any) -> Optional[Dict[str, Any]]:
    """Dispatch, descending one level into a container argument.

    A contract on ``catchments: List[gpd.GeoDataFrame]`` was a guaranteed no-op: the value is a
    list, ``_looks_like_frame`` rejects it, and the check returned None — not even
    cannot_determine. Nine shipped invariants on the e2SFCA family were decorative, and those
    are exactly the frames whose ``.area`` is taken.
    """
    if isinstance(value, (list, tuple)) and value:
        for index, item in enumerate(value[:8]):
            found = _check_one_arg(unit, invariant, item, suffix=f"[{index}]")
            if found is not None:
                return found
        return None
    if isinstance(value, dict) and value:
        for key, item in list(value.items())[:8]:
            found = _check_one_arg(unit, invariant, item, suffix=f"[{key!r}]")
            if found is not None:
                return found
        return None
    return _check_one_arg(unit, invariant, value)


def _check_one_arg(unit: str, invariant: Dict[str, Any], value: Any,
                   suffix: str = "") -> Optional[Dict[str, Any]]:
    """Check ONE declared invariant against ONE actual argument, or None if satisfied.

    Enforced at CALL time, by wrapping the imported unit, for a reason that is not incidental:
    an epilogue inspecting the namespace afterwards cannot know which frame was passed as which
    parameter. ``compute_accessibility(demand, supply)`` needs *demand* projected; a
    post-hoc scan sees two GeoDataFrames and has to guess. Wrapping the call removes the guess.
    """
    check = str(invariant.get("check") or "")
    target = str(invariant.get("target") or "?")
    where = f"{unit}({target}{suffix})"

    if check == "projected_crs":
        declared_unit = (invariant.get("args") or {}).get("unit")
        crs = _crs_of(value)
        if crs is None:
            if not _looks_like_frame(value):
                return None                     # not a frame; the contract does not apply
            return _finding(check, UNKNOWN, where,
                            f"{unit} declares {target} must be in a projected CRS "
                            f"(results in {declared_unit or 'metres'}), "
                            f"but the frame passed has no CRS set")
        projected = _is_projected(crs)
        if projected is False:
            return _finding(check, FAIL, where,
                            f"{unit} declares {target} must be in a PROJECTED CRS — its body "
                            f"performs a metric operation — but {crs!s} is geographic, so the "
                            f"result is in degrees. Reproject to a local projected CRS "
                            f"(a UTM or state-plane zone in {declared_unit or 'metres'}) "
                            f"before calling.", crs=str(crs))
        if projected is None:
            return _finding(check, UNKNOWN, where,
                            f"could not determine whether {crs!s} is projected", crs=str(crs))
        # Projected is not enough. A state-plane CRS in US survey feet satisfies "projected"
        # while making every length 3.28x wrong — the original degrees-vs-metres error class,
        # in feet, blessed by the check built to catch it.
        actual_unit = _crs_unit(crs)
        matches = _unit_matches(declared_unit, actual_unit)
        if matches is False:
            return _finding(check, FAIL, where,
                            f"{unit} declares {target} in {declared_unit}, but {crs!s} measures "
                            f"in {actual_unit} — every length and area from this frame is off by "
                            f"the unit ratio. Reproject to a CRS in {declared_unit}.",
                            crs=str(crs), crs_unit=actual_unit, declared_unit=str(declared_unit))
        if matches is None and declared_unit:
            return _finding(check, UNKNOWN, where,
                            f"{crs!s} is projected but its linear unit ({actual_unit or 'unknown'}) "
                            f"could not be compared to the declared {declared_unit}",
                            crs=str(crs))
        return None

    if check == "crs_equals":
        want = str((invariant.get("args") or {}).get("crs") or "")
        crs = _crs_of(value)
        if crs is None or not want:
            return None
        if str(crs).strip().lower() != want.strip().lower():
            return _finding(check, FAIL, where,
                            f"{unit} declares {target} must be {want}, got {crs!s}",
                            crs=str(crs), expected=want)
        return None

    if check == "reject_all_nan":
        if not _looks_like_frame(value):
            return None
        # Narrower than the module-scope check, and deliberately so. This invariant exists to
        # catch a FAILED UPSTREAM STEP -- "the caller learns which step broke, not just that the
        # end was NaN" -- so the signal is a frame that carries no data at all, not a frame with
        # one empty optional column.
        #
        # Reusing check_not_all_nan wholesale flagged any input whose optional column happened to
        # be empty (`apt_number`, `middle_name`), on every single call. A caveat that appears on
        # correct runs is a caveat nobody reads, and it would have arrived on the answer itself
        # now that cannot_determine reaches the user.
        try:
            columns = [c for c in list(value.columns) if str(c) != (_geometry_column(value) or "geometry")]
            rows = int(len(value))
        except Exception:
            return None
        if rows == 0:
            return _finding(check, UNKNOWN, where,
                            f"{unit} was called with an EMPTY {target} (0 rows), so any result "
                            f"is computed over no data — check the step that produced it")
        if columns:
            try:
                empty = [str(c) for c in columns if bool(value[c].isna().all())]
            except Exception:
                return None
            if len(empty) == len(columns):
                return _finding(check, FAIL, where,
                                f"every non-geometry column of {target} is entirely null when "
                                f"{unit} is called, so an upstream step produced nothing; any "
                                f"number derived from this is meaningless", columns=empty)
        return None

    return None


def install_contract_guards(namespace: Dict[str, Any], contracts: Dict[str, Any]) -> int:
    """Wrap library units named in *contracts* so their declared invariants are checked.

    Runs BEFORE the user's code, patching the module attribute — so the user's
    ``from iguide_methods.X.v_sha import symbol`` picks up the wrapped version. Violations are
    collected rather than raised: a contract breach means the NUMBER is wrong, and failing the
    run would destroy the evidence and the partial output the user might still want.

    Returns the number of units wrapped. Never raises: an unwrappable unit is simply unguarded,
    which is the pre-existing behaviour, and losing the guard must not lose the run.
    """
    import functools
    import importlib

    violations: List[Dict[str, Any]] = namespace.setdefault(VIOLATIONS_GLOBAL, [])
    wrapped = 0
    for unit_name, spec in (contracts or {}).items():
        if not isinstance(spec, dict):
            continue
        module_path = str(spec.get("module") or "")
        symbol = str(spec.get("symbol") or unit_name.split(".")[-1])
        invariants = [i for i in (spec.get("invariants") or []) if isinstance(i, dict)]
        if not (module_path and symbol and invariants):
            continue
        try:
            module = importlib.import_module(module_path)
            original = getattr(module, symbol)
        except Exception:
            continue
        if getattr(original, "_iguide_guarded", False):
            continue

        def make_guard(fn, unit, invs):
            @functools.wraps(fn)
            def guarded(*args, **kwargs):
                try:
                    import inspect
                    bound = inspect.signature(fn).bind_partial(*args, **kwargs)
                    supplied = dict(bound.arguments)
                except Exception:
                    supplied = {}
                for inv in invs:
                    target = str(inv.get("target") or "")
                    if target not in supplied:
                        continue
                    try:
                        found = check_contract_arg(unit, inv, supplied[target])
                    except Exception:
                        found = None
                    if found is not None:
                        violations.append(found)
                return fn(*args, **kwargs)

            guarded._iguide_guarded = True
            return guarded

        try:
            if isinstance(original, type):
                # A CLASS is guarded by patching its __init__, never by replacing the class.
                # functools.wraps on a class returns a plain function, which silently breaks
                # `isinstance(x, DoubleConv)` (arg 2 must be a type) and `class Sub(DoubleConv)`
                # (not an acceptable base type) -- and the units that carry classes here are
                # torch nn.Module layers, exactly the things that get subclassed and
                # isinstance-checked. Patching __init__ keeps the class identity intact and
                # still sees every constructor argument; a subclass calling super().__init__
                # is checked too.
                init = original.__dict__.get("__init__")
                if init is None:
                    continue           # inherited __init__: patching it would guard the base
                original.__init__ = make_guard(init, unit_name, invariants)
            else:
                setattr(module, symbol, make_guard(original, unit_name, invariants))
            wrapped += 1
        except Exception:
            continue
    return wrapped


def install_operation_tracker(namespace: Dict[str, Any]) -> bool:
    """Record every metric operation geopandas runs on a GEOGRAPHIC receiver.

    "Projected before measuring" is a property of the CALL, at the moment it runs, not of which
    frames happen to sit in EPSG:4326 when the run ends. Frame inventory was wrong both ways:
    - It stamped a correct run FAILED. The map UI, 2026-10-01: Champaign reprojected to 26916,
      buffered with the library's calculate_buffers, 165.04 km^2 against QGIS's 164.99. The
      untouched 4326 input was still bound, the measurement was a scalar, and no rescue fired.
    - A lineage rescue would have forgiven a run that reprojects and then measures the ORIGINAL
      anyway (the agent designer's counterexample).

    geopandas already names the operation. On a geographic receiver, ``area``, ``length``,
    ``buffer`` (not ``buffer(0)``), ``distance`` and friends warn "Geometry is in a geographic
    CRS. Results from '<op>' are likely incorrect", and the same call after ``to_crs`` is silent
    (verified in the deployed image, geopandas 1.1.4). ``warnings.warn`` is wrapped rather than
    ``showwarning`` hooked, because a script that does ``filterwarnings("ignore")``, as agents
    often do, never reaches ``showwarning``. The wrapper records first and then defers to the
    original, so what the run prints is unchanged.

    Not seen: shapely-level calls (``shapely.area(geoms)``) bypass geopandas. ``pyproj.Geod``
    measures geodesically in 4326, which is correct, and it never warns. Never raises.
    """
    import linecache
    import sys
    import warnings as _warnings

    ops = namespace.setdefault(GEOGRAPHIC_OPS_GLOBAL, [])
    current = _warnings.warn
    if getattr(current, "_iguide_tracked", False):
        # Already wrapped (a second gate in one process). Point it at THIS run's record: a
        # wrapper bound to the first namespace for good would record into a run that has ended.
        current._iguide_ops = ops
        namespace[OP_TRACKING_GLOBAL] = True
        return True
    original = current
    marker = _GEOGRAPHIC_WARNING

    def warn(message, category=None, stacklevel=1, source=None, **kwargs):
        try:
            text = str(message)
            ops = warn._iguide_ops
            if text.startswith(marker) and len(ops) < 50:
                op = text[len(marker):].split("'", 1)[0]
                code = ""
                frame = sys._getframe(1)
                while frame is not None and ("geopandas" in (frame.f_code.co_filename or "")
                                             or "pandas" in (frame.f_code.co_filename or "")):
                    frame = frame.f_back
                if frame is not None:
                    code = linecache.getline(frame.f_code.co_filename, frame.f_lineno).strip()
                ops.append({"op": op, "code": code[:160]})
        except Exception:
            pass
        return original(message, category, stacklevel + 1, source, **kwargs)

    try:
        warn._iguide_tracked = True
        warn._iguide_ops = ops
        _warnings.warn = warn
        namespace[OP_TRACKING_GLOBAL] = True
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def capture_environment() -> Dict[str, Any]:
    """The interpreter's own account of itself, recorded from INSIDE the sandbox.

    An agent-side `pip freeze` describes the agent's environment, not the container's — and
    the container is what produced the number. Only code running in the run can report the
    interpreter and the distributions that were actually importable, including whatever the
    per-session ``.deps`` directory contributed.

    Best-effort throughout: a run that cannot enumerate its packages should still record its
    Python version rather than nothing at all.
    """
    import sys

    env: Dict[str, Any] = {"python": sys.version.split()[0],
                           "python_full": sys.version.replace("\n", " "),
                           "platform": sys.platform,
                           "executable": sys.executable}
    packages: Dict[str, str] = {}
    try:
        from importlib import metadata as _md
        for dist in _md.distributions():
            try:
                name = (dist.metadata or {}).get("Name") or ""
                if name:
                    packages[str(name)] = str(dist.version or "")
            except Exception:
                continue
    except Exception as exc:
        env["packages_error"] = f"{type(exc).__name__}: {exc}"
    env["packages"] = dict(sorted(packages.items()))
    env["package_count"] = len(packages)
    return env


def _looks_like_frame(obj: Any) -> bool:
    return hasattr(obj, "columns") and hasattr(obj, "index") and hasattr(obj, "select_dtypes")


def _geometry_column(obj: Any) -> Optional[str]:
    """The ACTIVE geometry column, or None.

    This used to be ``"geometry" in obj.columns``. ``read_postgis`` defaults to ``geom``,
    geoparquet keeps whatever the writer used, and ``rename_geometry`` is routine — so a real
    GeoDataFrame was listed as inspected and reported clean while its buffer ran in degrees. Ask
    the object what its geometry column IS rather than guessing its name.
    """
    for probe in ("geometry",):
        try:
            name = getattr(getattr(obj, probe), "name", None)
            if isinstance(name, str) and name:
                return name
        except Exception:
            pass
    name = getattr(obj, "_geometry_column_name", None)
    if isinstance(name, str) and name:
        return name
    try:
        for col in list(getattr(obj, "columns", [])):
            if "geometry" in str(getattr(obj[col], "dtype", "")).lower():
                return str(col)
    except Exception:
        pass
    return None


def _has_geometry(obj: Any) -> bool:
    return _geometry_column(obj) is not None


def run_checks(namespace: Dict[str, Any], *, max_frames: int = 12) -> Dict[str, Any]:
    """Inspect every frame-like binding in *namespace* and return a findings report."""
    findings: List[Dict[str, Any]] = []
    inspected: List[str] = []

    # Contract violations recorded by the call-time guards go FIRST: they name the specific
    # unit and parameter that was misused, which is far more actionable than a frame-level
    # finding about a variable whose role the reader has to infer.
    for violation in (namespace.get(VIOLATIONS_GLOBAL) or []):
        if isinstance(violation, dict):
            findings.append(violation)

    # Geometry-bearing frames FIRST. Globals were walked in definition order and capped at
    # max_frames with a silent `continue`, so in a multi-step run the late output frames — the
    # ones a number is quoted from — were exactly the ones dropped, and the report said `pass`
    # with no sign of truncation.
    candidates: List[tuple] = []
    for name, obj in list(namespace.items()):
        if name.startswith("_"):
            continue
        try:
            if not _looks_like_frame(obj):
                continue
            candidates.append((0 if _has_geometry(obj) else 1, name, obj))
        except Exception:
            continue
    candidates.sort(key=lambda row: row[0])
    skipped = [name for _rank, name, _obj in candidates[max_frames:]]

    for _rank, name, obj in candidates[:max_frames]:
        inspected.append(name)
        for fn in (check_not_all_nan, check_join_cardinality):
            try:
                found = fn(name, obj)
                if found is not None:      # a check may decline to report; see join_cardinality
                    findings.append(found)
            except Exception as exc:            # a check must never break the run
                findings.append(_finding(fn.__name__, UNKNOWN, name, f"check errored: {exc}"))
        if _has_geometry(obj):
            try:
                findings.append(check_projected_crs(name, obj))
            except Exception as exc:
                findings.append(_finding("projected_crs", UNKNOWN, name, f"check errored: {exc}"))

    # A3: work done inside `def main()` leaves module scope empty, and the verdict was built
    # from finding counts alone — so one clean declared output scored `pass` with
    # `inspected: []`. A geospatial run whose frames were never reachable is UNVERIFIED.
    if not inspected:
        # Scoped to the namespace under inspection, NOT to sys.modules. `sys.modules` is
        # process-global: it reports geopandas as loaded because some *other* module imported
        # it, so a script doing pure arithmetic was called unverifiable. A bound module object
        # in the namespace being checked is the precise signal — the user's own
        # `import geopandas as gpd` at module scope — and it is testable.
        geo_bound = False
        for _name, _obj in list(namespace.items()):
            mod = getattr(_obj, "__name__", None) if isinstance(_obj, ModuleType) else None
            if mod and mod.split(".")[0] in _GEO_MODULES:
                geo_bound = True
                break
        if geo_bound:
            findings.append(_finding("coverage", UNKNOWN, "module scope",
                                     "a geospatial library is imported but no frame-like "
                                     "binding was reachable at module scope, so no frame was "
                                     "checked — work done inside a function cannot be "
                                     "verified; assign results to module-level names"))

    # Reproject-then-measure is the CORRECT workflow, and it necessarily leaves the original
    # geographic frame bound. A per-frame check cannot see that; run_checks can, because it sees
    # them all. So when some frame is projected AND carries a computed measurement, an unmeasured
    # geographic frame is an input that was reprojected — which is the right thing to have done,
    # not a caveat to put on the answer.
    reprojected = any(f.get("check") == "projected_crs" and f.get("status") == PASS
                      and _has_metric_column(namespace.get(f.get("target")))
                      for f in findings if isinstance(f, dict))
    if reprojected:
        for f in findings:
            # Only a geographic frame that holds NO measurement of its own. One that does is a
            # number computed in degrees regardless of what else the run got right.
            if (isinstance(f, dict) and f.get("check") == "projected_crs"
                    and f.get("status") == FAIL and not f.get("metric_column")):
                f["status"] = PASS
                f["message"] = (f"{f.get('crs', 'geographic CRS')} is geographic, but this run "
                                f"reprojected before measuring — the measurements live in a "
                                f"projected frame, so this is an untouched input.")

    # The OPERATION carries the verdict when the prologue's tracker was live: a metric operation
    # that ran on a geographic receiver is a FAIL, named by the line that ran it. Which frames
    # happen to sit in EPSG:4326 when the run ends no longer decides anything (see
    # install_operation_tracker). A geographic frame with no such operation is an input. One that
    # holds a measurement column no tracked operation produced is SUSPICIOUS, not proven wrong,
    # and a FAIL the gate cannot tie to a measurement reached the user as "⛔ not verified",
    # which teaches the reader to ignore ⛔. So it becomes a named cannot_determine.
    if namespace.get(OP_TRACKING_GLOBAL):
        metric_ops = [o for o in (namespace.get(GEOGRAPHIC_OPS_GLOBAL) or [])
                      if isinstance(o, dict) and o.get("op") in _METRIC_OPS]
        for f in findings:
            if not (isinstance(f, dict) and f.get("check") == "projected_crs"
                    and f.get("status") == FAIL and "op" not in f):
                continue
            if f.get("metric_column"):
                f["status"] = UNKNOWN
                f["message"] = (f"{f.get('crs', 'geographic CRS')} frame holds a measurement "
                                f"column ({f['metric_column']!r}) that no tracked operation "
                                f"produced in this run, so whether it is in degrees cannot be "
                                f"determined.")
            else:
                f["status"] = PASS
                f["message"] = (f"{f.get('crs', 'geographic CRS')} is geographic, but no metric "
                                f"operation ran on a geographic frame in this run: an input.")
        seen = set()
        for o in metric_ops:
            key = (o.get("op"), o.get("code"))
            if key in seen:
                continue
            seen.add(key)
            where = f" in `{o['code']}`" if o.get("code") else ""
            findings.append(_finding(
                "projected_crs", FAIL, o.get("code") or str(o.get("op")),
                f"'{o.get('op')}' ran on a GEOGRAPHIC CRS{where}, so its result is in degrees, "
                f"not metres. Reproject (e.g. .to_crs(3857) or a local UTM zone) before this "
                f"operation.", op=o.get("op")))

    # Declared numeric outputs, if the run published any. Checked outside the frame loop
    # because they are scalars the ANSWER will quote, not frames.
    declared = None
    try:
        declared = namespace.get(DECLARED_OUTPUTS)
        findings.extend(check_declared_units(declared))
    except Exception as exc:
        findings.append(_finding("declared_units", UNKNOWN, DECLARED_OUTPUTS,
                                 f"check errored: {exc}"))
    try:
        findings.extend(check_count_population(declared, namespace, max_frames=max_frames))
    except Exception as exc:
        findings.append(_finding("count_population", UNKNOWN, DECLARED_OUTPUTS,
                                 f"check errored: {exc}"))

    if skipped:
        # Never a silent cap: an uninspected frame is an unknown, not a pass.
        findings.append(_finding("coverage", UNKNOWN, ", ".join(skipped[:8]),
                                 f"{len(skipped)} frame-like binding(s) exceeded the inspection "
                                 f"budget of {max_frames} and were NOT checked",
                                 skipped=skipped[:24]))

    if not findings:
        # A run with no frames, no geospatial import and no declared outputs checked NOTHING, and
        # an empty report reached the reader as "cannot_determine (counts all zero) but its
        # findings were not retained" — which reads as evidence lost in transit. There was never
        # anything to retain. Saying so is the difference between "we tried and could not tell"
        # and "this run made no numeric claim to check", and in a multi-step turn the second is
        # usually a helper call that should not drag the answer to unverified.
        findings.append(_finding(
            "not_applicable", UNKNOWN, "this run",
            "nothing in this run was checkable: no frame-like binding, no geospatial import and "
            "no declared outputs. This is not a failed verification — publish IGUIDE_OUTPUTS to "
            "have the numbers you quote checked."))

    counts = {PASS: 0, FAIL: 0, UNKNOWN: 0}
    for f in findings:
        counts[f["status"]] = counts.get(f["status"], 0) + 1
    return {
        "schema": 1,
        "inspected": inspected,
        "findings": findings,
        "counts": counts,
        # The single field a reader should branch on, with precedence fail > unknown > pass.
        #
        # A single cannot_determine downgrades the whole run, even when everything else
        # passed. That is deliberate and it is the entire point: a frame with no CRS whose
        # null-check passes is NOT a verified result, and reporting `pass` there would let
        # exactly the wrong number through wearing a verified badge. The first version scored
        # `PASS if any passed`, which did precisely that.
        "verdict": (FAIL if counts[FAIL] else (UNKNOWN if counts[UNKNOWN] else
                                               (PASS if counts[PASS] else UNKNOWN))),
    }


def write_checks(namespace: Dict[str, Any], path: str = CHECKS_FILENAME) -> Dict[str, Any]:
    try:
        report = run_checks(namespace)
    except Exception as exc:
        report = {"schema": 1, "findings": [], "counts": {PASS: 0, FAIL: 0, UNKNOWN: 1},
                  "verdict": UNKNOWN, "error": f"{type(exc).__name__}: {exc}"}
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, default=str)
    except OSError:
        pass
    return report


_PROLOGUE = '''
# --- I-GUIDE invariant gate: registered BEFORE your code runs; does not change your results ---
{contracts_literal}
_IGUIDE_GATE_DONE = False


def _iguide_gate_body():
{body}
    # `locals()`, not a hand-listed tuple. The tuple form returned only
    # (run_checks, install_contract_guards) while the gate below also called
    # capture_environment() and read DECLARED_OUTPUTS -- both locals of THIS function, so both
    # raised NameError inside the gate's own `except Exception`, which reported the run as
    # `cannot_determine` with no findings. Indistinguishable from an honest "could not verify",
    # and it survived a four-exit-path probe that only checked that checks.json existed.
    # Returning the whole scope cannot drift as helpers are added.
    return dict(locals())


def _iguide_run_invariant_gate():
    global _IGUIDE_GATE_DONE
    if _IGUIDE_GATE_DONE:
        return
    _IGUIDE_GATE_DONE = True
    import json
    _h = _iguide_gate_body()
    _ns = dict(globals())
    try:
        _rep = _h["run_checks"](_ns)
    except Exception as _e:
        _rep = {{"schema": 1, "findings": [], "counts": {{"pass": 0, "fail": 0,
                "cannot_determine": 1}}, "verdict": "cannot_determine",
                "error": "%s: %s" % (type(_e).__name__, _e)}}
    try:
        with open({filename!r}, "w", encoding="utf-8") as _fh:
            json.dump(_rep, _fh, default=str)
    except OSError:
        pass
    try:
        with open({env_filename!r}, "w", encoding="utf-8") as _fh:
            json.dump(_h["capture_environment"](), _fh, default=str)
    except Exception:
        pass
    try:
        with open({declared_filename!r}, "w", encoding="utf-8") as _fh:
            json.dump(_ns.get(_h["DECLARED_OUTPUTS"]) or {{}}, _fh, default=str)
    except Exception:
        pass


# atexit, and registered from the PROLOGUE. An epilogue appended after the user's code is never
# even DEFINED when that code ends in `sys.exit(main())` — the standard script skeleton — so the
# run exited 0, discarded an already-recorded violation, and wrote no checks.json. Registering
# before the user's code runs is the only placement that survives every exit path.
try:
    import atexit as _iguide_atexit
    _iguide_atexit.register(_iguide_run_invariant_gate)
except Exception:
    pass

try:
    _iguide_install_contract_guards()
except Exception:
    pass

# Before the user's code, so every metric operation it runs is seen (install_operation_tracker).
try:
    _iguide_gate_body()['install_operation_tracker'](globals())
except Exception:
    pass

'''


def _inlined_helpers() -> str:
    """This module's own check functions, indented for injection into the sandbox."""
    parts: List[str] = []
    for obj in (_finding, _crs_of, _is_projected, _crs_unit, _unit_matches, check_projected_crs,
                check_not_all_nan, _looks_like_join_result, _has_metric_column,
                check_join_cardinality,
                check_finite, _count_finding, check_declared_units, check_count_population,
                capture_environment, check_contract_arg, _check_one_arg, _geometry_column,
                _looks_like_frame, _has_geometry, install_contract_guards,
                install_operation_tracker, run_checks):
        src = inspect.getsource(obj)
        parts.append("\n".join("    " + line if line.strip() else line
                               for line in src.splitlines()))
    return ("    PASS, FAIL, UNKNOWN = 'pass', 'fail', 'cannot_determine'\n"
            f"    DECLARED_OUTPUTS = {DECLARED_OUTPUTS!r}\n"
            f"    _KNOWN_UNITS = {_KNOWN_UNITS!r}\n"
            f"    _UNIT_ALIASES = {_UNIT_ALIASES!r}\n"
            f"    VIOLATIONS_GLOBAL = {VIOLATIONS_GLOBAL!r}\n"
            f"    GEOGRAPHIC_OPS_GLOBAL = {GEOGRAPHIC_OPS_GLOBAL!r}\n"
            f"    OP_TRACKING_GLOBAL = {OP_TRACKING_GLOBAL!r}\n"
            f"    _METRIC_OPS = {tuple(_METRIC_OPS)!r}\n"
            f"    _GEOGRAPHIC_WARNING = {_GEOGRAPHIC_WARNING!r}\n"
            f"    _GEO_MODULES = {set(_GEO_MODULES)!r}\n"
            f"    _METRIC_COLUMN_HINTS = {tuple(_METRIC_COLUMN_HINTS)!r}\n"
            "    import math\n"
            "    from types import ModuleType\n"
            "    from typing import Any, Dict, List, Optional\n" + "\n".join(parts))


def prologue_source(contracts: Optional[Dict[str, Any]] = None) -> str:
    """Everything the gate needs, injected BEFORE the user's code.

    Always emitted when the gate is enabled — not only when there are contracts to guard —
    because this is where the atexit registration lives, and that registration is what makes the
    gate survive ``sys.exit()``, an uncaught exception, and ``os._exit``-free early returns.

    Contracts are injected as a literal rather than read from the mounted registry: the mount is
    optional and the sandbox has no network, and a guard that silently fails to install reads
    exactly like a contract that passed.
    """
    import json as _json

    contracts = contracts or {}
    literal = f"{CONTRACTS_GLOBAL} = " + _json.dumps(contracts, default=str)
    installer = (
        "def _iguide_install_contract_guards():\n"
        f"    _iguide_gate_body()['install_contract_guards']"
        f"(globals(), {CONTRACTS_GLOBAL})\n"
        if contracts else
        "def _iguide_install_contract_guards():\n    return None\n")
    return _PROLOGUE.format(contracts_literal=literal + "\n\n\n" + installer,
                            body=_inlined_helpers(), filename=CHECKS_FILENAME,
                            env_filename=ENVIRONMENT_FILENAME,
                            declared_filename=DECLARED_FILENAME)


def epilogue_source() -> str:
    """A second, inline invocation appended AFTER the user's code.

    Belt and braces: the prologue's atexit registration is what guarantees the report, but an
    inline call runs the checks while the interpreter is still in a normal state, which produces
    better diagnostics when a check itself misbehaves. ``_IGUIDE_GATE_DONE`` makes the pair
    idempotent, so whichever fires first wins and the other is a no-op.
    """
    return (
        "\n\n# --- I-GUIDE invariant gate (inline; the prologue also registers it atexit) ---\n"
        "try:\n"
        "    _iguide_run_invariant_gate()\n"
        "except NameError:\n"
        # The prologue defines the gate. If only the epilogue was emitted, a bare
        # `except Exception: pass` silently wrote NO report -- indistinguishable from a gate
        # that was never enabled, which is the exact failure shape this gate exists to catch.
        # Record the refusal instead.
        "    try:\n"
        "        import json as _j, pathlib as _p\n"
        "        _p.Path(%r).write_text(_j.dumps({\n"
        "            'verdict': 'cannot_determine', 'inspected': [], 'findings': [{\n"
        "                'check': 'gate', 'status': 'cannot_determine', 'target': 'runtime',\n"
        "                'message': 'the invariant gate prologue was not installed, so nothing "
        "was checked'}],\n"
        "            'counts': {'pass': 0, 'fail': 0, 'cannot_determine': 1}}), "
        "encoding='utf-8')\n"
        "    except Exception:\n"
        "        pass\n"
        "except Exception:\n"
        "    pass\n" % CHECKS_FILENAME)


__all__ = ["run_checks", "write_checks", "epilogue_source", "capture_environment",
           "prologue_source", "install_contract_guards", "check_contract_arg",
           "install_operation_tracker", "CONTRACTS_GLOBAL", "VIOLATIONS_GLOBAL",
           "GEOGRAPHIC_OPS_GLOBAL", "OP_TRACKING_GLOBAL",
           "CHECKS_FILENAME", "ENVIRONMENT_FILENAME", "DECLARED_FILENAME",
           "PASS", "FAIL", "UNKNOWN", "DECLARED_OUTPUTS", "check_projected_crs",
           "check_not_all_nan", "check_join_cardinality", "check_finite",
           "check_declared_units"]
