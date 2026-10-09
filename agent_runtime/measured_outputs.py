"""Measured numbers, typed without the model declaring them.

The gate's unit and CRS checks used to run only on the numbers a script assigned to
`IGUIDE_OUTPUTS`. The tool description asked for that on every run, and across the GIS harness's
gate-on runs no model ever did it (0 of 96 reports), so the checks never ran. Whether a number
got checked depended on the model's habits, and that is the dependence the design programme set
out to remove (docs/design-review-2026-10.md).

Two things every run already produces carry the numbers instead:

* **What the run printed.** A model can only quote a number it has seen, and it sees only what
  the run prints. A figure printed with a unit the unit library parses is typed: "Area: 2,586.01
  km²", `area_km2=2586.0`, `{"distance_m": 412.5}`, "Watershed area (km²): 16.2".
* **What the measuring calls returned.** The sandbox records every `.area`, `.length`,
  `.distance` and `.hausdorff_distance` with the CRS it ran in, that CRS's linear unit, and the
  values (sandbox_verify._install_metric_op_recorder). That record is typed by the CRS, so it
  holds whatever the print looks like: a bare `print(a)`, a tuple, a table.

A printed length or area that equals a recorded measurement, after unit conversion and within
the digits printed, inherits that measurement's CRS. Two checks follow, and neither reads a
variable's or a column's name to decide what a number is:

* **measured_in**: the figure was measured in a geographic CRS, so it is in degrees.
* **printed_unit**: the figure equals the measurement only if the CRS's unit is taken to be
  metres, and it is not. EPSG:3435 is in US survey feet: ft² ÷ 1e6 printed as "km²" is 10.76x
  too large, and no frame shows it.

The declared-count checks (fractional, negative, more than every frame) are NOT run on printed
numbers. They judge a model's claim about a count. A printed number is the run's own output, and
"108785 people" is a sum, not a row count. The number scan (facts.py) already holds an answer's
counts to what the run printed.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from agent_runtime import facts as _facts
from agent_runtime.units import convert, parse_unit

PASS, FAIL = "pass", "fail"

# What travels back to the model inside `verification` (it is in the tool result): enough for
# the turn's typed facts, bounded so a 40-step loop does not cost the model 40 entries.
MAX_OUTPUTS = 16
# A figure is matched to a measurement only when it shows this many significant digits, so a
# coincidence is unlikely: "5 m" equals something in most runs; "2,586.01 km²" does not.
_MIN_SIGNIFICANT = 3
_MEASURES = ("[length]", "[length] ** 2")
_LINES, _NUMBERS = 2000, 400

# A unit in the label in front of the number, as the run printed it. Two shapes:
# - bracketed: "Area (km²): 16.2", "area [m]: 5";
# - the label's last word, joined to a word before it: `area_km2=`, "Watershed area km2:",
#   `"distance_m": 412.5`, and a snake_case key printed beside its value (`print('area_ha',
#   x)` gives "area_ha 103.85"). The word before has to be letters, so a parameter is not read
#   as the unit of the result after it ("within_1_mile: 19", "within 25 km: 673").
# Only lengths and areas are read this way. They are what the CRS checks are about, and other
# dimensions collide with common labels ("min=0" is a minimum, "C2:" a site name).
_BRACKET_LABEL_RE = re.compile(r"[(\[]\s*([^()\[\]]{1,24}?)\s*[)\]]\s*[\"']?\s*[:=]\s*$")
_TOKEN_LABEL_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]+[_ ]([A-Za-z²]{1,12}\d?)[\"']?\s*[:=]\s*$")
_KEY_LABEL_RE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]+_([A-Za-z²]{1,12}\d?)[\"']?,?\s+$")


def _label_unit(before: str) -> Optional[str]:
    for rx in (_BRACKET_LABEL_RE, _TOKEN_LABEL_RE, _KEY_LABEL_RE):
        m = rx.search(before)
        if m and parse_unit(m.group(1)).dimension in _MEASURES:
            return m.group(1)
    return None


def _significant(text: str) -> int:
    digits = re.sub(r"\D", "", text.split("e")[0].split("E")[0]).lstrip("0")
    return len(digits)


def printed_quantities(stdout: Any) -> List[Dict[str, Any]]:
    """Every number the run printed with a unit beside it: `{value, unit, dimension, text,
    line, decimals}`. A number with no unit is left out; it is still an untyped fact."""
    out: List[Dict[str, Any]] = []
    for line in str(stdout or "").splitlines()[:_LINES]:
        line = line[:400]
        for m in _facts._NUM_RE.finditer(line):
            if len(out) >= _NUMBERS:
                return out
            whole, frac = (m.group(2) or "").replace(",", ""), m.group(3) or ""
            try:
                value = float(whole + frac) * (-1 if m.group(1) else 1)
            except ValueError:
                continue
            if _facts._IDENTIFIER_BEFORE_RE.search(line[max(0, m.start() - 12):m.start()]):
                continue                                    # EPSG:32616, GEOID 17019
            if _facts._hemisphere(line, m.end(), value) is not None:
                continue                                    # 41.88 N is a latitude, not newtons
            unit = None
            um = _facts._UNIT_AFTER_RE.match(line, m.end())
            if um:
                candidate = um.group(1).rstrip(".,;:")
                if parse_unit(candidate).kind == "unit":
                    unit = candidate
            if unit is None:
                unit = _label_unit(line[:m.start()])
            if unit is None:
                continue
            pu = parse_unit(unit)
            out.append({"value": value, "unit": unit, "dimension": pu.dimension,
                        "counted": pu.counted, "text": m.group(0), "line": line.strip()[:120],
                        "decimals": len(frac) - 1 if frac else 0})
    return out


def _measurements(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Each recorded measurement's values in METRES (a projected CRS) or degrees (a geographic
    one), with the raw values kept for the as-if-metres comparison."""
    out = []
    for op in report.get("metric_ops") or []:
        if not isinstance(op, dict) or not isinstance(op.get("values"), dict):
            continue
        dim = "[length] ** 2" if op.get("op") == "area" else "[length]"
        power = 2 if dim == "[length] ** 2" else 1
        factor = op.get("crs_unit_factor")
        projected = op.get("projected")
        v = op["values"]
        stats = [(k, v[k]) for k in ("sum", "min", "max", "mean") if isinstance(v.get(k), (int, float))]
        stats += [(f"#{i}", x) for i, x in enumerate(v.get("each") or [])
                  if isinstance(x, (int, float))]
        metres = projected is True and isinstance(factor, (int, float)) and factor > 0
        out.append({"op": op.get("op"), "crs": op.get("crs"), "projected": projected,
                    "crs_unit": op.get("crs_unit"), "factor": factor if metres else None,
                    "dimension": dim, "code": op.get("code") or "",
                    "unit": ("m" if power == 1 else "m**2") if metres else None,
                    "stats": [(k, float(x) * (factor ** power if metres else 1.0), float(x))
                              for k, x in stats if math.isfinite(x)],
                    "n": v.get("n")})
    return out


def _close(printed: Dict[str, Any], candidate: Optional[float]) -> bool:
    if candidate is None or not math.isfinite(candidate):
        return False
    shown = 0.5 * 10 ** (-printed["decimals"])
    return abs(printed["value"] - candidate) <= max(shown, 1e-9 * abs(candidate))


def _link(p: Dict[str, Any], meas: List[Dict[str, Any]]) -> Tuple[Optional[Dict], Optional[Dict]]:
    """(the measurement this figure equals, or None; the measurement it equals only when the CRS
    unit is misread as metres, or None)."""
    pu = parse_unit(p["unit"])
    if pu.dimension not in _MEASURES or pu.counted or _significant(p["text"]) < _MIN_SIGNIFICANT:
        return None, None
    misread = None
    for m in meas:
        if m["dimension"] != pu.dimension:
            continue
        for stat, metric, raw in m["stats"]:
            if m["unit"] and _close(p, convert(metric, m["unit"], p["unit"])):
                return {**m, "stat": stat}, None
            # The same raw number read as metres: right when the CRS is in metres (already
            # matched above), a degree reading when it is geographic, a feet reading when not.
            as_metres = convert(raw, "m" if pu.dimension == "[length]" else "m**2", p["unit"])
            not_metres = m["projected"] is False or (m["factor"] and abs(m["factor"] - 1) > 1e-9)
            if misread is None and not_metres and _close(p, as_metres):
                misread = {**m, "stat": stat}
    return None, misread


def evaluate(report: Dict[str, Any], stdout: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(findings, typed outputs) for a run's printed figures and recorded measurements."""
    findings: List[Dict[str, Any]] = []
    printed_out: List[Dict[str, Any]] = []
    meas = _measurements(report or {})
    linked_ops = set()
    for p in printed_quantities(stdout):
        out = {"name": p["line"][:60], "value": p["value"], "unit": p["unit"],
               "source": "printed"}
        if p["dimension"] and p["dimension"] != "dimensionless":
            out["dimension"] = p["dimension"]
        hit, misread = _link(p, meas)
        target = f"{p['text']} {p['unit']}"
        if hit is not None:
            linked_ops.add(id(hit.get("stats")))
            out["measured_in_crs"] = hit["crs"]
            out["op"] = hit["op"]
            findings.append({"check": "measured_in", "status": PASS, "target": target,
                             "message": f"printed {target} is the {hit['op']} ({hit['stat']}) "
                                        f"measured in {hit['crs']}, unit {hit['crs_unit']}",
                             "code": hit["code"][:120]})
        elif misread is not None:
            linked_ops.add(id(misread.get("stats")))
            out["measured_in_crs"] = misread["crs"]
            out["op"] = misread["op"]
            if misread["projected"] is False:
                findings.append({
                    "check": "measured_in", "status": FAIL, "target": target,
                    "message": f"printed {target} is the {misread['op']} measured in "
                               f"{misread['crs']}, a GEOGRAPHIC CRS, so it is in degrees, not "
                               f"{p['unit']}. Reproject to a projected CRS before measuring.",
                    "code": misread["code"][:120]})
            elif misread["projected"] is True:
                findings.append({
                    "check": "printed_unit", "status": FAIL, "target": target,
                    "message": f"printed {target} equals the {misread['op']} measured in "
                               f"{misread['crs']} only if its unit were metres, but that CRS is "
                               f"in {misread['crs_unit']}. Convert from {misread['crs_unit']} "
                               f"before printing it as {p['unit']}.",
                    "code": misread["code"][:120]})
        printed_out.append(out)
    # A measurement nothing printed with a unit is still a typed fact: a model that prints
    # `print(a)` or a tuple had its number measured all the same.
    measured_out: List[Dict[str, Any]] = []
    for m in meas:
        if id(m["stats"]) in linked_ops or not m["unit"] or not m["stats"]:
            continue
        stats = dict((k, x) for k, x, _ in m["stats"])
        for key in ("sum", "min", "max") if (m.get("n") or 0) > 1 else ("sum",):
            if key in stats:
                measured_out.append({"name": f"{m['op']} {key}" if (m.get("n") or 0) > 1
                                     else m["op"], "value": stats[key], "unit": m["unit"],
                                     "dimension": m["dimension"], "measured_in_crs": m["crs"],
                                     "source": "measured", "op": m["op"]})
    # Linked and measured first: those carry a CRS. Then everything else printed with a unit.
    ranked = ([o for o in printed_out if o.get("measured_in_crs")] + measured_out
              + [o for o in printed_out if not o.get("measured_in_crs")])
    return findings, ranked[:MAX_OUTPUTS]
