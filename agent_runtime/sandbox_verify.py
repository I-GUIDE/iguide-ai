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

import json
import math
from typing import Any, Dict, List, Optional

PASS = "pass"
FAIL = "fail"
UNKNOWN = "cannot_determine"

CHECKS_FILENAME = "checks.json"


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
    return _finding("projected_crs", FAIL, name,
                    f"{crs!s} is GEOGRAPHIC: distances and areas computed from this frame are "
                    f"in degrees, not metres. Reproject (e.g. .to_crs(3857) or a local UTM "
                    f"zone) before buffering or measuring.", crs=str(crs))


def check_not_all_nan(name: str, frame: Any) -> Dict[str, Any]:
    """An entirely-null column is a failed join or a failed parse wearing a result's shape.

    Checks EVERY column, not just ``select_dtypes("number")``. That was the first version and
    it missed the commonest case: pandas types an all-``None`` column as ``object``, so the
    column produced by an unmatched join — the exact thing this check exists for — was
    excluded from the check by its own dtype.
    """
    try:
        columns = list(frame.columns)
    except Exception:
        return _finding("all_nan", UNKNOWN, name, "columns could not be inspected")
    if not columns:
        return _finding("all_nan", UNKNOWN, name, "frame has no columns")
    if len(frame) == 0:
        return _finding("all_nan", FAIL, name, "frame is empty (0 rows)")
    bad: List[str] = []
    for col in columns:
        if str(col) == "geometry":
            continue
        try:
            if bool(frame[col].isna().all()):
                bad.append(str(col))
        except Exception:
            continue
    if bad:
        return _finding("all_nan", FAIL, name,
                        f"column(s) entirely null: {', '.join(bad)} — usually an unmatched "
                        f"join or a failed parse, not a real result", columns=bad)
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


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def _looks_like_frame(obj: Any) -> bool:
    return hasattr(obj, "columns") and hasattr(obj, "index") and hasattr(obj, "select_dtypes")


def _has_geometry(obj: Any) -> bool:
    try:
        return "geometry" in list(getattr(obj, "columns", []))
    except Exception:
        return False


def run_checks(namespace: Dict[str, Any], *, max_frames: int = 12) -> Dict[str, Any]:
    """Inspect every frame-like binding in *namespace* and return a findings report."""
    findings: List[Dict[str, Any]] = []
    inspected: List[str] = []
    for name, obj in list(namespace.items()):
        if name.startswith("_") or len(inspected) >= max_frames:
            continue
        try:
            if not _looks_like_frame(obj):
                continue
        except Exception:
            continue
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


_EPILOGUE = '''

# --- I-GUIDE invariant gate (appended automatically; does not affect your results) ---
def _iguide_run_invariant_gate():
    import json as _json, math as _math
    _ns = dict(globals())
{body}
    try:
        _rep = run_checks(_ns)
    except Exception as _e:
        _rep = {{"schema": 1, "findings": [], "counts": {{"pass": 0, "fail": 0,
                "cannot_determine": 1}}, "verdict": "cannot_determine",
                "error": "%s: %s" % (type(_e).__name__, _e)}}
    try:
        with open({filename!r}, "w", encoding="utf-8") as _fh:
            _json.dump(_rep, _fh, default=str)
    except OSError:
        pass


try:
    _iguide_run_invariant_gate()
except Exception:
    pass
'''


def epilogue_source() -> str:
    """Self-contained checker source to append to sandboxed code.

    The check functions are inlined rather than imported: the method-library mount may be
    absent (nothing ingested yet) and the sandbox has no network, so an import-based epilogue
    would silently do nothing exactly when a library-free run most needs checking.
    """
    import inspect

    parts: List[str] = []
    for obj in (_finding, _crs_of, _is_projected, check_projected_crs, check_not_all_nan,
                check_join_cardinality, check_finite, _looks_like_frame, _has_geometry,
                run_checks):
        src = inspect.getsource(obj)
        parts.append("\n".join("    " + line if line.strip() else line
                               for line in src.splitlines()))
    body = ("    PASS, FAIL, UNKNOWN = 'pass', 'fail', 'cannot_determine'\n"
            "    from typing import Any, Dict, List, Optional\n" + "\n".join(parts))
    return _EPILOGUE.format(body=body, filename=CHECKS_FILENAME)


__all__ = ["run_checks", "write_checks", "epilogue_source", "CHECKS_FILENAME",
           "PASS", "FAIL", "UNKNOWN", "check_projected_crs", "check_not_all_nan",
           "check_join_cardinality", "check_finite"]
