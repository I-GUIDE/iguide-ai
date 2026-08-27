"""Temporal analysis tools for the analysis agent — the WHEN half of the toolkit.

``langchain_geo_tools`` can read, join and map a dataset but knows nothing about time, so
questions like "show only last July", "is this rising?", "which tracts got worse?" had no
tool at all. These five do:

    detect_time_column   which column holds the time, what span/granularity, how dirty
    filter_by_time       a temporal slice, straight onto the user's interactive map
    time_series          counts per period (+ hour-of-day / day-of-week profiles) as CSV + PNG
    compare_periods      per-area counts in two windows and a DIVERGING choropleth of the change
    temporal_hotspots    where activity concentrated in the latest period vs earlier ones

The hard part is that time arrives as *strings in whatever format the publisher liked* —
Chicago crime ships ``07/26/2026 08:00:00 PM``, ISO feeds ship ``2026-07-26T20:00:00Z``,
survey exports ship a bare ``2026`` or epoch milliseconds. ``parse_time_series`` tries a
ladder of strategies (already-datetime, epoch, inferred single format, pandas ``format="mixed"``,
then an explicit format list) and keeps whichever parsed the MOST rows, then every tool
reports the parse rate and how many rows were dropped — a silent 40% NaT is the classic way
a temporal answer ends up quietly wrong. Timestamps that carry a UTC offset are converted to
UTC and made tz-naive, so one dataset never mixes wall-clock and offset time; everything
downstream (windows, periods, hour-of-day) is therefore in UTC for such inputs.

Conventions copied from ``langchain_geo_tools``: heavy imports deferred into tool bodies,
every tool returns a JSON string and NEVER raises, metric work happens in a projected CRS
(``estimate_utm_crs``) and output ships as EPSG:4326 for the web map.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from agent_runtime.langchain_geo_tools import (  # reuse, do not reinvent
    _epsg,
    _index_attached,
    _resolve,
    _stage_vector_source,
    artifact_name,
    read_vector,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.tools import StructuredTool

# Same ceiling add_map_layer uses, so a temporal slice of a big point set behaves identically.
_MAP_MAX_FEATURES = int(os.getenv("AGENT_MAP_LAYER_MAX_FEATURES", "150000"))
# Rows sampled per column while *sniffing* for a time column (parsing 1M rows x 40 columns
# to answer "which one is the date?" is pure waste; the winner is then parsed in full).
_DETECT_SAMPLE = int(os.getenv("AGENT_TIME_DETECT_SAMPLE", "4000"))
# A column has to parse this fraction of its non-null values to count as a time column.
_MIN_PARSE_RATE = float(os.getenv("AGENT_TIME_MIN_PARSE_RATE", "0.6"))
# Above this many points, an "auto" temporal slice renders as density instead of marks.
_AUTO_HEATMAP_ABOVE = int(os.getenv("AGENT_TIME_HEATMAP_ABOVE", "5000"))

# Column names that suggest time. Used only to BREAK TIES / to justify trying a numeric
# column — the decision is made by how many values actually parse.
_TIME_NAME_HINTS = (
    "date", "time", "timestamp", "datetime", "_dt", "dt_", "year", "month", "day", "hour",
    "period", "when", "occur", "report", "observ", "record", "creat", "updat", "modif",
    "start", "end", "begin", "epoch", "collect", "sampl", "visit", "arrest", "incident",
)

# Tried in order, best-parse-rate wins. First entry is the Chicago-crime shape, which is
# also the shape most US open-data portals emit.
_EXPLICIT_FORMATS: Tuple[str, ...] = (
    "%m/%d/%Y %I:%M:%S %p",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y",
    "%m/%d/%y",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d",
    "%d-%b-%Y",
    "%d %b %Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%Y%m%d",
    "%Y-%m",
    "%Y",
)

_FREQ_ALIASES: Dict[str, str] = {
    "hour": "h", "hourly": "h", "h": "h", "hours": "h",
    "day": "D", "daily": "D", "d": "D", "days": "D", "date": "D",
    "week": "W", "weekly": "W", "w": "W", "weeks": "W",
    "month": "M", "monthly": "M", "m": "M", "months": "M",
    "quarter": "Q", "quarterly": "Q", "q": "Q",
    "year": "Y", "yearly": "Y", "annual": "Y", "annually": "Y", "y": "Y", "years": "Y",
}
_PROFILE_ALIASES: Dict[str, str] = {
    "hour_of_day": "hour_of_day", "hourofday": "hour_of_day", "hour-of-day": "hour_of_day",
    "hod": "hour_of_day", "time_of_day": "hour_of_day", "diurnal": "hour_of_day",
    "day_of_week": "day_of_week", "dayofweek": "day_of_week", "day-of-week": "day_of_week",
    "dow": "day_of_week", "weekday": "day_of_week", "weekly_profile": "day_of_week",
    "month_of_year": "month_of_year", "monthofyear": "month_of_year",
    "month-of-year": "month_of_year", "moy": "month_of_year", "seasonal": "month_of_year",
}
_LABEL_FMT = {"h": "%Y-%m-%d %H:00", "D": "%Y-%m-%d", "W": "%Y-%m-%d", "M": "%Y-%m", "Y": "%Y"}
_DOW_ORDER = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_MONTH_ORDER = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

_FREQ_HELP = ("hour | day | week | month | quarter | year, or a profile: "
              "hour_of_day | day_of_week | month_of_year")


# --------------------------------------------------------------------------- errors
class TimeColumnError(ValueError):
    """No usable time column. Carries the candidates so the caller can retry concretely."""

    def __init__(self, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.extra = extra


class ColumnError(ValueError):
    """A named column is absent/unusable. Carries candidate column lists."""

    def __init__(self, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.extra = extra


def _fail(exc: BaseException, **extra: Any) -> str:
    """The single failure envelope: ``{"ok": false, "error": "Type: msg", ...hints}``."""
    payload: Dict[str, Any] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    payload.update(getattr(exc, "extra", None) or {})
    payload.update(extra)
    return json.dumps(payload, default=str)


def _num(value: Any) -> Optional[float]:
    """A JSON-safe float: NaN/inf/None all collapse to ``None`` (``NaN`` is invalid JSON)."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if (out != out or out in (float("inf"), float("-inf"))) else out


# ------------------------------------------------------------------- time parsing
def _naive(series: Any) -> Any:
    """Force a parse result to tz-naive ``datetime64[ns]`` (mixed offsets come back as object)."""
    import pandas as pd

    if series is None:
        return None
    try:
        if isinstance(series.dtype, pd.DatetimeTZDtype):
            return series.dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:  # noqa: BLE001
        pass
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coerced = pd.to_datetime(series, errors="coerce", utc=True)
        return coerced.dt.tz_localize(None)
    except Exception:  # noqa: BLE001
        return pd.Series(pd.NaT, index=getattr(series, "index", None), dtype="datetime64[ns]")


def parse_time_series(values: Any) -> Tuple[Any, str]:
    """``(tz-naive datetime64 Series, method label)`` — the best of several strategies.

    Never raises: a hopeless column comes back as all-NaT with method ``"unparsed"`` so the
    caller can report a 0% parse rate instead of blowing up.
    """
    import pandas as pd

    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if pd.api.types.is_datetime64_any_dtype(series):
        return _naive(series), "already datetime"
    if len(series) == 0:
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]"), "unparsed"

    attempts: List[Tuple[str, Any]] = []
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        finite = numeric.dropna()
        if len(finite):
            lo, hi = float(finite.min()), float(finite.abs().max())
            if 1000 <= lo and hi <= 3000:  # a bare year column (2019, 2020, ...)
                text_years = finite.astype("int64").astype("string").reindex(series.index)
                attempts.append(("year number",
                                 lambda t=text_years: pd.to_datetime(t, format="%Y", errors="coerce")))
            elif 1e8 <= hi < 1e11:
                attempts.append(("epoch seconds",
                                 lambda n=numeric: pd.to_datetime(n, unit="s", errors="coerce")))
            elif 1e11 <= hi < 1e14:
                attempts.append(("epoch milliseconds",
                                 lambda n=numeric: pd.to_datetime(n, unit="ms", errors="coerce")))
    text = series.astype("string").str.strip().replace({"": None})
    attempts.append(("inferred single format", lambda: pd.to_datetime(text, errors="coerce")))
    attempts.append(("mixed formats", lambda: pd.to_datetime(text, errors="coerce", format="mixed")))
    attempts.extend(
        (f"format {fmt}", lambda f=fmt: pd.to_datetime(text, format=f, errors="coerce"))
        for fmt in _EXPLICIT_FORMATS
    )

    best, best_label, best_n = None, "unparsed", -1
    target = int(series.notna().sum())
    for label, attempt in attempts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = _naive(attempt())
        except Exception:  # noqa: BLE001 - a format that doesn't apply is not an error
            continue
        if parsed is None:
            continue
        hits = int(parsed.notna().sum())
        if hits > best_n:
            best, best_label, best_n = parsed, label, hits
        if best_n >= target:  # nothing left to win
            break
    if best is None or best_n <= 0:
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]"), "unparsed"
    return best, best_label


def _granularity(parsed: Any) -> Optional[str]:
    """The finest unit that is actually USED — a column of midnights is daily, not secondly."""
    valid = parsed.dropna()
    if valid.empty:
        return None
    acc = valid.dt
    midnight = bool((acc.hour == 0).all() and (acc.minute == 0).all() and (acc.second == 0).all())
    if midnight and bool((acc.month == 1).all() and (acc.day == 1).all()):
        return "year"
    if midnight and bool((acc.day == 1).all()):
        return "month"
    if midnight:
        return "day"
    if bool((acc.minute == 0).all() and (acc.second == 0).all()):
        return "hour"
    if bool((acc.second == 0).all()):
        return "minute"
    return "second"


def _span(parsed: Any) -> Dict[str, Any]:
    """min/max/duration of a parsed column, JSON-safe and empty-safe."""
    valid = parsed.dropna()
    if valid.empty:
        return {"start": None, "end": None, "days": None}
    start, end = valid.min(), valid.max()
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "days": round((end - start).total_seconds() / 86400.0, 3),
    }


def _profile(parsed: Any, column: str, method: str) -> Dict[str, Any]:
    """The per-column report shared by detect_time_column and every other tool."""
    total = int(len(parsed))
    ok = int(parsed.notna().sum())
    return {
        "column": column,
        "parse_method": method,
        "rows": total,
        "parsed_rows": ok,
        "failed_rows": total - ok,
        "parse_rate": round(ok / total, 4) if total else 0.0,
        "null_rate": round((total - ok) / total, 4) if total else 0.0,
        "granularity": _granularity(parsed),
        "span": _span(parsed),
    }


def _name_hint(column: str) -> bool:
    low = str(column).lower()
    return any(hint in low for hint in _TIME_NAME_HINTS)


def _candidate_columns(frame: Any) -> List[str]:
    """Columns worth *trying* to parse: datetimes, text, and time-named numerics.

    A numeric column without a time-ish name is skipped on purpose — "beat", "ward" and
    "population" all parse happily as epoch seconds and would outrank the real date.
    """
    import pandas as pd

    out: List[str] = []
    for col in frame.columns:
        if col == getattr(frame, "_geometry_column_name", "geometry") or col == "geometry":
            continue
        series = frame[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            out.append(col)
        elif pd.api.types.is_bool_dtype(series):
            continue
        elif pd.api.types.is_numeric_dtype(series):
            if _name_hint(col):
                out.append(col)
        else:
            out.append(col)
    return out


def _rank_time_columns(frame: Any, limit: int = 6) -> List[Dict[str, Any]]:
    """Profile every plausible column on a sample and return the good ones, best first."""
    sample = frame if len(frame) <= _DETECT_SAMPLE else frame.head(_DETECT_SAMPLE)
    scored: List[Tuple[float, int, Dict[str, Any]]] = []
    for col in _candidate_columns(sample):
        parsed, method = parse_time_series(sample[col])
        report = _profile(parsed, str(col), method)
        if report["parse_rate"] < _MIN_PARSE_RATE or report["parsed_rows"] == 0:
            continue
        report["name_suggests_time"] = _name_hint(col)
        report["sample_values"] = [str(v) for v in sample[col].dropna().head(3).tolist()]
        scored.append((report["parse_rate"], 1 if report["name_suggests_time"] else 0, report))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored[:limit]]


def _resolve_time_column(frame: Any, time_column: Optional[str]) -> Tuple[Any, Dict[str, Any]]:
    """``(parsed datetime Series, report)``; raises TimeColumnError with candidates.

    ``time_column=None`` auto-detects. A *named* column that is present but unparseable is
    reported differently from one that is absent, because the fix differs.
    """
    columns = [str(c) for c in frame.columns if c != "geometry"]
    if time_column:
        wanted = str(time_column)
        if wanted not in frame.columns:
            lower = {c.lower(): c for c in columns}
            if wanted.lower() in lower:  # forgiving on case, since portals shout their headers
                wanted = lower[wanted.lower()]
            else:
                ranked = _rank_time_columns(frame)
                cands = [r["column"] for r in ranked]
                raise TimeColumnError(
                    f"time column {time_column!r} is not in this dataset",
                    candidates=cands or columns[:40],
                    time_column_candidates=cands,
                    columns=columns[:60],
                )
        parsed, method = parse_time_series(frame[wanted])
        report = _profile(parsed, wanted, method)
        if report["parsed_rows"] == 0:
            ranked = _rank_time_columns(frame)
            cands = [r["column"] for r in ranked]
            raise TimeColumnError(
                f"no value in column {wanted!r} could be parsed as a date/time",
                candidates=cands or columns[:40],
                time_column_candidates=cands,
                columns=columns[:60],
                sample_values=[str(v) for v in frame[wanted].dropna().head(5).tolist()],
            )
        return parsed, report

    ranked = _rank_time_columns(frame)
    if not ranked:
        raise TimeColumnError(
            "no time/date column could be detected in this dataset",
            candidates=columns[:40],
            columns=columns[:60],
            hint="pass time_column explicitly, or run detect_time_column to see what is there",
        )
    chosen = ranked[0]["column"]
    parsed, method = parse_time_series(frame[chosen])  # re-parse in FULL (ranking used a sample)
    report = _profile(parsed, chosen, method)
    report["auto_detected"] = True
    report["other_candidates"] = [r["column"] for r in ranked[1:]]
    return parsed, report


def _parse_bound(text: Any, *, is_end: bool) -> Any:
    """A window edge from human text, honouring its PRECISION.

    ``"2026"`` -> Jan 1 .. Dec 31 23:59:59, ``"2026-07"`` -> that month, ``"2026-07-26"``
    -> that whole day. Without this, ``end="2026-07-26"`` would silently drop every event
    after midnight of the 26th.
    """
    import pandas as pd

    raw = str(text).strip()
    if not raw:
        raise ValueError("empty date bound")
    try:
        period = pd.Period(raw)
        return period.end_time if is_end else period.start_time
    except Exception:  # noqa: BLE001
        pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stamp = pd.to_datetime(raw, errors="coerce")
        if stamp is None or pd.isna(stamp):
            stamp = pd.to_datetime(raw, errors="coerce", format="mixed")
    if stamp is None or pd.isna(stamp):
        raise ValueError(
            f"could not read {text!r} as a date; try 2026-07-26, 2026-07, 2026, "
            "or a range like '2026-01-01..2026-06-30'")
    if getattr(stamp, "tzinfo", None) is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    if is_end and stamp == stamp.normalize():  # date-only text means the WHOLE day
        return stamp + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return stamp


_RANGE_SEPARATORS = ("..", "…", " to ", " through ", " until ", " - ", " – ", " — ", ";", "|")


def _parse_period(text: Any, label: str) -> Tuple[Any, Any]:
    """``"2026-07"`` / ``"2026"`` / ``"2026-01-01..2026-06-30"`` -> inclusive (start, end)."""
    raw = str(text or "").strip()
    if not raw:
        raise ValueError(f"{label} is empty; give a period like '2026-07' or '2026-01-01..2026-06-30'")
    for sep in _RANGE_SEPARATORS:
        if sep in raw:
            left, _, right = raw.partition(sep)
            start = _parse_bound(left, is_end=False)
            end = _parse_bound(right, is_end=True)
            break
    else:
        start = _parse_bound(raw, is_end=False)
        end = _parse_bound(raw, is_end=True)
    if end < start:
        start, end = end, start
    return start, end


def _normalize_freq(freq: Any) -> Tuple[str, str]:
    """``("period", pandas_code)`` or ``("profile", name)``; raises with the accepted list."""
    key = str(freq or "month").strip().lower().replace(" ", "_")
    if key in _FREQ_ALIASES:
        return "period", _FREQ_ALIASES[key]
    if key in _PROFILE_ALIASES:
        return "profile", _PROFILE_ALIASES[key]
    raise ValueError(f"unknown freq {freq!r}; accepted: {_FREQ_HELP}")


def _period_labels(parsed: Any, code: str) -> Tuple[Any, Any]:
    """``(period_start datetimes, printable labels)`` for a chronological frequency."""
    periods = parsed.dt.to_period(code)
    start = periods.dt.start_time
    if code == "Q":
        labels = start.dt.year.astype("Int64").astype("string") + "Q" + start.dt.quarter.astype("Int64").astype("string")
    else:
        labels = start.dt.strftime(_LABEL_FMT.get(code, "%Y-%m-%d"))
    return start, labels


# ---------------------------------------------------------------------------- io
def _read_frame(ref: str, siblings: Optional[List[str]], attached: Optional[List[Dict[str, Any]]],
                *, layer: Optional[str] = None, need_geometry: bool = True) -> Tuple[Any, str]:
    """``(DataFrame|GeoDataFrame, source_name)`` for any supported upload.

    ``need_geometry=False`` lets the non-spatial tool (time_series) work on a plain
    spreadsheet with no coordinates at all, which read_vector alone refuses.
    """
    path, record = _resolve(ref)
    source = str((record or {}).get("filename") or path.name)
    read_path, tmp = _stage_vector_source(ref, siblings, attached)
    try:
        try:
            frame = read_vector(read_path, layer)
        except Exception:
            if need_geometry:
                raise
            frame = _read_plain_table(read_path)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    if need_geometry and "geometry" not in getattr(frame, "columns", []):
        raise ValueError(f"{source} has no geometry and no coordinate columns to derive it from")
    return frame, source


def _read_plain_table(read_path: Any) -> Any:
    """Last resort for a geometry-less table (CSV/TSV/parquet/Excel)."""
    import pandas as pd

    low = str(read_path).lower()
    if low.endswith((".parquet", ".geoparquet")):
        return pd.read_parquet(read_path)
    if low.endswith((".xlsx", ".xls")):
        return pd.read_excel(read_path)
    sep = "\t" if low.endswith((".tsv", ".tab")) else None
    return pd.read_csv(read_path, sep=sep, engine="python")


def _stringify(gdf: Any) -> Any:
    """ISO-string every datetime/period column — GeoJSON has no date type, and NaT/NaN
    would otherwise be written as the invalid JSON literal ``NaN``."""
    import pandas as pd

    out = gdf.copy()
    for col in out.columns:
        if col == "geometry":
            continue
        series = out[col]
        if pd.api.types.is_datetime64_any_dtype(series) or str(series.dtype).startswith("period"):
            out[col] = series.astype("string").where(series.notna(), None)
        elif pd.api.types.is_float_dtype(series) and series.isna().any():
            out[col] = series.astype(object).where(series.notna(), None)
    return out


def _publish_geojson(gdf: Any, filename: str) -> Dict[str, Any]:
    from agent_runtime.file_store import create_output_file_from_path

    tmpdir = Path(tempfile.mkdtemp(prefix="temporal_gj_"))
    try:
        out = tmpdir / filename
        _stringify(gdf).to_file(out, driver="GeoJSON")
        return create_output_file_from_path(out, filename=filename)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _publish_table(frame: Any, filename: str) -> Dict[str, Any]:
    from agent_runtime.file_store import create_output_file_from_path

    tmpdir = Path(tempfile.mkdtemp(prefix="temporal_csv_"))
    try:
        out = tmpdir / filename
        frame.to_csv(out, index=False)
        return create_output_file_from_path(out, filename=filename)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _publish_figure(fig: Any, filename: str) -> Dict[str, Any]:
    from agent_runtime.file_store import create_output_file_from_path

    tmpdir = Path(tempfile.mkdtemp(prefix="temporal_png_"))
    try:
        out = tmpdir / filename
        fig.savefig(out, bbox_inches="tight", dpi=150)
        return create_output_file_from_path(out, filename=filename)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _asset(record: Dict[str, Any]) -> Dict[str, Any]:
    return {"file_id": record["file_id"], "filename": record.get("filename"),
            "download_url": record.get("download_url")}


def _to_metric(gdf: Any) -> Tuple[Any, str, Optional[str]]:
    """``(projected GeoDataFrame, epsg, note)`` — metric work NEVER happens in degrees.

    Cell sizes and distances are metres, so the frame is moved into its own UTM zone first
    (falling back to Web Mercator only if the extent defeats ``estimate_utm_crs``).
    """
    note = None
    work = gdf
    if getattr(work, "crs", None) is None:
        work = work.set_crs("EPSG:4326")
        note = "input had no CRS; assumed EPSG:4326 (lon/lat)"
    try:
        target = work.estimate_utm_crs()
    except Exception:  # noqa: BLE001 - global/empty extents have no single UTM zone
        target = "EPSG:3857"
        note = ((note + "; ") if note else "") + "extent spans too much for one UTM zone; used EPSG:3857"
    projected = work.to_crs(target)
    return projected, (_epsg(projected.crs) or str(target)), note


def _points_of(gdf: Any) -> Tuple[Any, Optional[str]]:
    """Point geometry for gridding: non-point inputs collapse to a representative point."""
    kinds = {str(t) for t in gdf.geometry.geom_type.dropna().unique()}
    if kinds and kinds <= {"Point", "MultiPoint"}:
        return gdf, None
    out = gdf.copy()
    out["geometry"] = gdf.geometry.representative_point()
    return out, f"input geometry is {sorted(kinds)}; each feature reduced to a representative point"


def _label_column(gdf: Any, requested: Optional[str] = None) -> Optional[str]:
    """A human name for an area: honour the caller, else the first name-ish string column."""
    if requested:
        if requested in gdf.columns:
            return requested
        lower = {str(c).lower(): c for c in gdf.columns}
        if str(requested).lower() in lower:
            return lower[str(requested).lower()]
        raise ColumnError(
            f"areas_label_column {requested!r} is not in the areas dataset",
            candidates=[str(c) for c in gdf.columns if c != "geometry"][:40],
            columns=[str(c) for c in gdf.columns if c != "geometry"][:60],
        )
    preferred = ("name", "namelsad", "name10", "label", "area", "zone", "neighborhood",
                 "community", "district", "tract", "geoid", "id")
    lower = {str(c).lower(): c for c in gdf.columns}
    for key in preferred:
        if key in lower and lower[key] != "geometry":
            return lower[key]
    for col in gdf.columns:
        if col != "geometry" and gdf[col].dtype == object:
            return col
    return None


def _numeric_columns(frame: Any) -> List[str]:
    import pandas as pd

    return [str(c) for c in frame.columns
            if c != "geometry" and pd.api.types.is_numeric_dtype(frame[c])]


def _sample_for_map(gdf: Any, ceiling: int) -> Tuple[Any, bool, int]:
    total = int(len(gdf))
    if total <= ceiling:
        return gdf, False, total
    return gdf.sample(int(ceiling), random_state=0), True, total


# ----------------------------------------------------------------------- factory
def make_temporal_tools(default_input_file_ids: Optional[List[str]] = None) -> List["StructuredTool"]:
    """Build the temporal StructuredTools (pandas + geopandas backed).

    ``default_input_file_ids`` is the conversation's attached file set, used only to
    auto-discover shapefile sidecars (.shx/.dbf/.prj) by basename — exactly as
    ``make_langchain_geo_tools`` does, so a user who uploaded an extracted shapefile can
    reference any single component.
    """
    from langchain_core.tools import StructuredTool

    _attached = _index_attached(default_input_file_ids)

    def _load(ref: str, siblings: Optional[List[str]] = None, *, layer: Optional[str] = None,
              need_geometry: bool = True) -> Tuple[Any, str]:
        return _read_frame(ref, siblings, _attached, layer=layer, need_geometry=need_geometry)

    # -------------------------------------------------------------- detect
    def detect_time_column(file_id: str, time_column: Optional[str] = None,
                           sibling_file_ids: Optional[List[str]] = None,
                           layer: Optional[str] = None) -> str:
        """Find the time/date column(s) in a dataset and report the parsed span,
        granularity, parse method and how many rows FAILED to parse. Run this first when a
        temporal question arrives and the time field is unknown or messy. Timestamps with a
        UTC offset are normalised to UTC. """
        try:
            frame, source = _load(file_id, sibling_file_ids, layer=layer, need_geometry=False)
            columns = [str(c) for c in frame.columns if c != "geometry"]
            if time_column:
                parsed, report = _resolve_time_column(frame, time_column)
                ranked = [report]
            else:
                ranked = _rank_time_columns(frame)
            payload: Dict[str, Any] = {
                "ok": True,
                "source": source,
                "row_count": int(len(frame)),
                "found": bool(ranked),
                "time_column": ranked[0]["column"] if ranked else None,
                "candidates": ranked,
                "spatial": "geometry" in getattr(frame, "columns", []),
                "columns": columns[:60],
            }
            if not ranked:
                payload["hint"] = (
                    "no column parsed as a date/time; if a time field exists under an unusual "
                    "name or encoding, pass it as time_column to see the parse attempt")
            else:
                best = ranked[0]
                payload["note"] = (
                    f"{best['column']}: {best['parsed_rows']}/{best['rows']} rows parsed "
                    f"({best['parse_method']}), granularity {best['granularity']}, "
                    f"{best['span']['start']} -> {best['span']['end']}")
                if best["failed_rows"]:
                    payload["warning"] = (
                        f"{best['failed_rows']} row(s) ({best['null_rate']:.1%}) have no usable "
                        "timestamp and will be EXCLUDED from every temporal result")
            return json.dumps(payload, default=str)
        except Exception as exc:  # noqa: BLE001
            return _fail(exc)

    # -------------------------------------------------------------- filter
    def filter_by_time(file_id: str, start: Optional[str] = None, end: Optional[str] = None,
                       time_column: Optional[str] = None, render: str = "auto",
                       style_by: Optional[str] = None, name: Optional[str] = None,
                       sibling_file_ids: Optional[List[str]] = None,
                       layer: Optional[str] = None, max_features: Optional[int] = None) -> str:
        """Keep only the features inside a time window and put them on the user's
        interactive map. `start`/`end` accept 2026-07-26, 2026-07, 2026 or a full timestamp;
        either may be omitted for an open-ended window. An end given as a plain date covers
        that WHOLE day. """
        try:
            import pandas as pd

            gdf, source = _load(file_id, sibling_file_ids, layer=layer)
            parsed, report = _resolve_time_column(gdf, time_column)
            available = _span(parsed)
            lower = _parse_bound(start, is_end=False) if start else None
            upper = _parse_bound(end, is_end=True) if end else None
            if lower is not None and upper is not None and upper < lower:
                lower, upper = upper, lower

            mask = parsed.notna()
            if lower is not None:
                mask &= parsed >= lower
            if upper is not None:
                mask &= parsed <= upper
            subset = gdf.loc[mask.fillna(False).to_numpy()].copy()

            if style_by:
                if style_by not in subset.columns:
                    raise ColumnError(
                        f"style_by {style_by!r} is not a column in this dataset",
                        candidates=_numeric_columns(gdf), numeric_columns=_numeric_columns(gdf),
                        columns=[str(c) for c in gdf.columns if c != "geometry"][:60])
                if not pd.api.types.is_numeric_dtype(subset[style_by]):
                    raise ColumnError(
                        f"style_by {style_by!r} is not numeric, so it cannot shade a layer",
                        candidates=_numeric_columns(gdf), numeric_columns=_numeric_columns(gdf))

            window = {"start": lower.isoformat() if lower is not None else None,
                      "end": upper.isoformat() if upper is not None else None}
            base = {
                "ok": True, "source": source, "time_column": report["column"],
                "parse": report, "window": window, "available_span": available,
                "matched": int(len(subset)), "input_features": int(len(gdf)),
                "excluded_out_of_window": int(len(gdf) - len(subset) - report["failed_rows"]),
                "excluded_unparsed_time": int(report["failed_rows"]),
            }
            if subset.empty:
                base.update({
                    "feature_count": 0, "on_map": False,
                    "note": ("no feature falls in that window, so nothing was added to the map; "
                             f"this dataset covers {available['start']} -> {available['end']}"),
                })
                return json.dumps(base, default=str)

            if getattr(subset, "crs", None) is not None:
                subset = subset.to_crs("EPSG:4326")           # web maps are lon/lat
            else:
                subset = subset.set_crs("EPSG:4326")
                base["crs_note"] = "input had no CRS; assumed EPSG:4326"

            kinds = {str(t) for t in subset.geometry.geom_type.dropna().unique()}
            is_point = bool(kinds) and kinds <= {"Point", "MultiPoint"}
            mode = str(render or "auto").strip().lower()
            if mode == "auto":
                if not is_point:
                    mode = "shapes"
                elif style_by or len(subset) <= _AUTO_HEATMAP_ABOVE:
                    mode = "points"
                else:
                    mode = "heatmap"
            if mode not in {"heatmap", "choropleth", "points", "shapes"}:
                raise ValueError(f"render must be heatmap|choropleth|points|shapes, got {render!r}")
            if mode == "choropleth" and not style_by:
                raise ColumnError(
                    "render='choropleth' needs style_by set to a numeric column",
                    candidates=_numeric_columns(subset), numeric_columns=_numeric_columns(subset))

            ceiling = int(max_features) if max_features else _MAP_MAX_FEATURES
            subset, sampled, total = _sample_for_map(subset, ceiling)
            fname = artifact_name(name, "geojson", source=source, default="time_window")
            record = _publish_geojson(subset, fname)
            label_bits = [w for w in (window["start"], window["end"]) if w]
            label = (name or "").replace("_", " ").strip() or (
                " to ".join(b[:10] for b in label_bits) if label_bits else "time window")
            base.update({
                "file_id": record["file_id"], "filename": record.get("filename"),
                "download_url": record.get("download_url"),
                "feature_count": int(len(subset)), "features_total": total,
                "sampled": bool(sampled), "on_map": True,
                "crs": _epsg(getattr(subset, "crs", None)),
                "map_layer": {"url": record.get("download_url"), "label": label, "render": mode,
                              "style_by": style_by, "source": "analysis",
                              "count": int(len(subset)), "sampled": bool(sampled),
                              "total": total},
            })
            if sampled:
                base["note"] = (f"showing a random {len(subset)} of {total} matching features — "
                                "the layer on the map is a SAMPLE")
            return json.dumps(base, default=str)
        except Exception as exc:  # noqa: BLE001
            return _fail(exc)

    # ---------------------------------------------------------- time series
    def time_series(file_id: str, freq: str = "month", time_column: Optional[str] = None,
                    by: Optional[str] = None, top_n: int = 8, name: Optional[str] = None,
                    sibling_file_ids: Optional[List[str]] = None,
                    layer: Optional[str] = None) -> str:
        """Count records per time period and return a CSV of the table plus a PNG chart.
        `freq`: hour|day|week|month|quarter|year, or a cyclical profile hour_of_day |
        day_of_week | month_of_year. `by` splits the series by a category column (top_n
        categories kept). NON-SPATIAL: this returns a chart and a table, NOT a map layer. """
        fig = None
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import pandas as pd

            frame, source = _load(file_id, sibling_file_ids, layer=layer, need_geometry=False)
            parsed, report = _resolve_time_column(frame, time_column)
            kind, code = _normalize_freq(freq)

            group_col = None
            if by:
                if by in frame.columns:
                    group_col = by
                else:
                    lower = {str(c).lower(): c for c in frame.columns}
                    group_col = lower.get(str(by).lower())
                if group_col is None:
                    cats = [str(c) for c in frame.columns
                            if c != "geometry" and frame[c].nunique(dropna=True) <= 50]
                    raise ColumnError(
                        f"by {by!r} is not a column in this dataset",
                        candidates=cats or [str(c) for c in frame.columns if c != "geometry"][:40],
                        category_columns=cats[:40], numeric_columns=_numeric_columns(frame),
                        columns=[str(c) for c in frame.columns if c != "geometry"][:60])

            valid = parsed.notna()
            times = parsed[valid]
            if times.empty:
                raise ValueError(f"no parseable timestamp in column {report['column']!r}")

            if kind == "period":
                period_start, labels = _period_labels(times, code)
                order = pd.DataFrame({"period": labels, "_start": period_start})
                order = order.drop_duplicates("period").sort_values("_start")
                index_order = order["period"].tolist()
                axis_label = {"h": "hour", "D": "day", "W": "week (starting)",
                              "M": "month", "Q": "quarter", "Y": "year"}[code]
                chart_kind = "line"
                keys = labels
            elif code == "hour_of_day":
                keys = times.dt.hour.astype(int).map(lambda h: f"{h:02d}")
                index_order = [f"{h:02d}" for h in range(24)]
                axis_label, chart_kind = "hour of day (local clock time)", "bar"
            elif code == "day_of_week":
                keys = times.dt.day_name()
                index_order = list(_DOW_ORDER)
                axis_label, chart_kind = "day of week", "bar"
            else:  # month_of_year
                keys = times.dt.month.map(lambda m: _MONTH_ORDER[int(m) - 1])
                index_order = list(_MONTH_ORDER)
                axis_label, chart_kind = "month of year", "bar"

            work = pd.DataFrame({"period": list(keys)})
            note_by = None
            if group_col is not None:
                cats = frame.loc[valid.to_numpy(), group_col].astype("string").fillna("(missing)")
                keep = cats.value_counts().head(max(1, int(top_n))).index.tolist()
                dropped = int(cats.nunique() - len(keep))
                work["group"] = [c if c in keep else "other" for c in cats]
                table = (work.pivot_table(index="period", columns="group", aggfunc="size",
                                          fill_value=0)
                         .reindex(index_order, fill_value=0))
                table.columns = [str(c) for c in table.columns]
                table = table.reindex(columns=sorted(table.columns,
                                                     key=lambda c: (c == "other", c)))
                table["total"] = table.sum(axis=1)
                if dropped > 0:
                    note_by = (f"{group_col}: top {len(keep)} categories kept, the remaining "
                               f"{dropped} folded into 'other'")
            else:
                counts = work["period"].value_counts().reindex(index_order, fill_value=0)
                table = pd.DataFrame({"count": counts.astype(int)})
                table.index.name = "period"

            table = table.fillna(0)
            out_table = table.reset_index().rename(columns={"index": "period"})
            csv_name = artifact_name(name, "csv", source=source, default=f"time_series_{code.lower()}")
            csv_rec = _publish_table(out_table, csv_name)

            fig, ax = plt.subplots(figsize=(10, 4.6))
            plot_frame = table.drop(columns=["total"]) if "total" in table.columns else table
            if chart_kind == "line":
                plot_frame.plot(ax=ax, marker="o", linewidth=1.6, markersize=3.5)
            else:
                plot_frame.plot(kind="bar", ax=ax, stacked=group_col is not None,
                                color=None if group_col is not None else "#3aa9a0",
                                edgecolor="#1c5a97", linewidth=0.4, width=0.82)
            ax.set_xlabel(axis_label)
            ax.set_ylabel("records")
            title = (name or Path(source).stem).replace("_", " ")
            ax.set_title(f"{title} — records per {axis_label}"
                         + (f" by {group_col}" if group_col is not None else ""))
            ax.grid(axis="y", alpha=0.25)
            if group_col is None and ax.get_legend():
                ax.get_legend().remove()
            if len(plot_frame) > 14:
                step = max(1, len(plot_frame) // 14)
                ax.set_xticks(range(0, len(plot_frame), step))
                ax.set_xticklabels(list(plot_frame.index)[::step], rotation=45, ha="right")
            else:
                ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            png_rec = _publish_figure(fig, artifact_name(name, "png", source=source,
                                                         default=f"time_series_{code.lower()}"))

            totals = table["total"] if "total" in table.columns else table["count"]
            peak_period = str(totals.idxmax()) if len(totals) else None
            preview_rows = out_table.head(60).to_dict(orient="records")
            payload = {
                "ok": True, "spatial": False,
                "note": ("NON-SPATIAL result: a table (CSV) and a chart (PNG). Nothing was added "
                         "to the map — use filter_by_time, compare_periods or temporal_hotspots "
                         "for a map layer."),
                "source": source, "time_column": report["column"], "parse": report,
                "freq": str(freq), "freq_resolved": code, "chart_kind": chart_kind,
                "by": group_col, "periods": int(len(table)),
                "records_counted": int(totals.sum()),
                "excluded_unparsed_time": int(report["failed_rows"]),
                "span": _span(times),
                "peak": {"period": peak_period,
                         "count": int(totals.max()) if len(totals) else 0},
                "mean_per_period": _num(totals.mean()) if len(totals) else None,
                # Top-level file_id is the CSV (the table behind the chart).
                "file_id": csv_rec["file_id"], "filename": csv_rec.get("filename"),
                "download_url": csv_rec.get("download_url"),
                "row_count": int(len(out_table)),
                "csv": _asset(csv_rec), "chart": _asset(png_rec),
                "columns": [str(c) for c in out_table.columns],
                "series": preview_rows,
                "series_truncated": len(out_table) > len(preview_rows),
            }
            if note_by:
                payload["by_note"] = note_by
            if report["failed_rows"]:
                payload["warning"] = (f"{report['failed_rows']} row(s) had no usable timestamp "
                                      "and are not counted anywhere in this series")
            return json.dumps(payload, default=str)
        except Exception as exc:  # noqa: BLE001
            return _fail(exc)
        finally:
            if fig is not None:
                try:
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                except Exception:  # noqa: BLE001
                    pass

    # ------------------------------------------------------- compare periods
    def compare_periods(file_id: str, areas_file_id: str, period_a: str, period_b: str,
                        time_column: Optional[str] = None, predicate: str = "within",
                        areas_label_column: Optional[str] = None, name: Optional[str] = None,
                        sibling_file_ids: Optional[List[str]] = None,
                        areas_sibling_file_ids: Optional[List[str]] = None) -> str:
        """Count records per area in TWO time windows and map the change between them as a
        DIVERGING choropleth (styled by `change`, centred on zero). `period_a`/`period_b`
        accept 2026-07, 2026, or a range like 2026-01-01..2026-06-30. Also returns a CSV of
        area / a / b / change / pct_change. """
        try:
            import geopandas as gpd
            import pandas as pd

            events, source = _load(file_id, sibling_file_ids)
            areas, areas_source = _load(areas_file_id, areas_sibling_file_ids)
            parsed, report = _resolve_time_column(events, time_column)
            a_start, a_end = _parse_period(period_a, "period_a")
            b_start, b_end = _parse_period(period_b, "period_b")
            pred = str(predicate or "within").strip().lower()
            if pred not in {"within", "intersects", "contains"}:
                raise ValueError(f"predicate must be within|intersects|contains, got {predicate!r}")

            notes: List[str] = []
            if getattr(areas, "crs", None) is None:
                areas = areas.set_crs("EPSG:4326")
                notes.append("areas had no CRS; assumed EPSG:4326")
            if getattr(events, "crs", None) is None:
                events = events.set_crs("EPSG:4326")
                notes.append("events had no CRS; assumed EPSG:4326")
            events = events.to_crs(areas.crs)

            label_col = _label_column(areas, areas_label_column)
            areas = areas.reset_index(drop=True)
            areas["_area_idx"] = range(len(areas))
            frames = areas[["_area_idx", "geometry"]]

            def _count(lo: Any, hi: Any) -> Any:
                mask = parsed.notna() & (parsed >= lo) & (parsed <= hi)
                subset = events.loc[mask.fillna(False).to_numpy()]
                if subset.empty:
                    return pd.Series(0, index=areas["_area_idx"], dtype="int64")
                joined = gpd.sjoin(subset[["geometry"]], frames, how="inner", predicate=pred)
                counts = joined["_area_idx"].value_counts()
                return counts.reindex(areas["_area_idx"], fill_value=0).astype("int64")

            count_a = _count(a_start, a_end)
            count_b = _count(b_start, b_end)

            keep = ["_area_idx", "geometry"] + ([label_col] if label_col else [])
            out = areas[keep].copy()
            out["count_a"] = count_a.to_numpy()
            out["count_b"] = count_b.to_numpy()
            out["change"] = (out["count_b"] - out["count_a"]).astype("int64")
            # pct_change has NO meaning against a zero baseline — leave it null rather than
            # emitting inf, which would blow up both the CSV and the colour ramp.
            pct = (out["change"] / out["count_a"].replace(0, pd.NA)) * 100.0
            out["pct_change"] = pd.to_numeric(pct, errors="coerce").round(2)
            out = out.drop(columns=["_area_idx"])
            out = out.to_crs("EPSG:4326")                      # web maps are lon/lat

            csv_frame = pd.DataFrame({
                "area": (out[label_col].astype("string") if label_col
                         else pd.Series([f"area_{i}" for i in range(len(out))])),
                "a": out["count_a"].to_numpy(),
                "b": out["count_b"].to_numpy(),
                "change": out["change"].to_numpy(),
                "pct_change": out["pct_change"].to_numpy(),
            })
            csv_rec = _publish_table(csv_frame, artifact_name(name, "csv", source=areas_source,
                                                              default="period_comparison"))
            gj_name = artifact_name(name, "geojson", source=areas_source, default="period_change")
            gj_rec = _publish_geojson(out, gj_name)

            ranked = csv_frame.sort_values("change", ascending=False)
            top_up = [{"area": str(r["area"]), "a": int(r["a"]), "b": int(r["b"]),
                       "change": int(r["change"]), "pct_change": _num(r["pct_change"])}
                      for _, r in ranked.head(5).iterrows() if int(r["change"]) > 0]
            top_down = [{"area": str(r["area"]), "a": int(r["a"]), "b": int(r["b"]),
                         "change": int(r["change"]), "pct_change": _num(r["pct_change"])}
                        for _, r in ranked.tail(5).iloc[::-1].iterrows() if int(r["change"]) < 0]
            extreme = int(max(abs(int(out["change"].min())), abs(int(out["change"].max())))) if len(out) else 0
            label = (name or "").replace("_", " ").strip() or (
                f"change {a_start.date()}..{a_end.date()} -> {b_start.date()}..{b_end.date()}")
            payload = {
                "ok": True, "source": source, "areas_source": areas_source,
                "time_column": report["column"], "parse": report, "predicate": pred,
                "period_a": {"start": a_start.isoformat(), "end": a_end.isoformat(),
                             "total": int(out["count_a"].sum())},
                "period_b": {"start": b_start.isoformat(), "end": b_end.isoformat(),
                             "total": int(out["count_b"].sum())},
                "area_label_column": label_col,
                "file_id": gj_rec["file_id"], "filename": gj_rec.get("filename"),
                "download_url": gj_rec.get("download_url"),
                "feature_count": int(len(out)), "areas": int(len(out)),
                "areas_increased": int((out["change"] > 0).sum()),
                "areas_decreased": int((out["change"] < 0).sum()),
                "areas_unchanged": int((out["change"] == 0).sum()),
                "net_change": int(out["change"].sum()),
                "excluded_unparsed_time": int(report["failed_rows"]),
                "csv": _asset(csv_rec), "geojson": _asset(gj_rec),
                "top_increases": top_up, "top_decreases": top_down,
                "crs": _epsg(getattr(out, "crs", None)),
                "method": (f"records counted per area with a spatial join (predicate={pred}); "
                           "change = count in period_b minus count in period_a; pct_change is "
                           "null where period_a was zero (no baseline to divide by)"),
                "map_layer": {"url": gj_rec.get("download_url"), "label": label,
                              "render": "choropleth", "style_by": "change",
                              "source": "analysis", "count": int(len(out)),
                              # Hints for a diverging ramp: 0 is the neutral middle.
                              "diverging": True, "midpoint": 0,
                              "domain": [-extreme, extreme]},
                "on_map": True,
                "note": ("diverging choropleth: negative change (fewer in period_b) and positive "
                         "change (more in period_b) diverge from zero"),
            }
            if notes:
                payload["crs_note"] = "; ".join(notes)
            if report["failed_rows"]:
                payload["warning"] = (f"{report['failed_rows']} record(s) had no usable timestamp "
                                      "and are counted in neither period")
            return json.dumps(payload, default=str)
        except Exception as exc:  # noqa: BLE001
            return _fail(exc)

    # ------------------------------------------------------ temporal hotspots
    def temporal_hotspots(file_id: str, freq: str = "month", cell_km: float = 1.0,
                          time_column: Optional[str] = None, min_events: int = 1,
                          name: Optional[str] = None,
                          sibling_file_ids: Optional[List[str]] = None,
                          layer: Optional[str] = None) -> str:
        """Show WHERE activity concentrated in the most recent period compared with the
        average of the earlier ones: a square grid (`cell_km` kilometres on a side, measured
        in a metric projection) shaded by the shift. Returns a diverging choropleth styled by
        `shift`, plus a CSV of every cell. """
        try:
            import geopandas as gpd
            import pandas as pd
            from shapely.geometry import box

            gdf, source = _load(file_id, sibling_file_ids, layer=layer)
            parsed, report = _resolve_time_column(gdf, time_column)
            kind, code = _normalize_freq(freq)
            if kind != "period":
                raise ValueError(
                    f"temporal_hotspots needs a chronological freq (hour|day|week|month|quarter|year), "
                    f"not the cyclical profile {freq!r}")
            try:
                size_km = float(cell_km)
            except (TypeError, ValueError):
                raise ValueError(f"cell_km must be a number in kilometres, got {cell_km!r}")
            if not (size_km > 0):
                raise ValueError(f"cell_km must be greater than 0, got {cell_km!r}")

            valid = parsed.notna().to_numpy()
            events = gdf.loc[valid].copy()
            times = parsed[valid]
            if events.empty:
                raise ValueError(f"no parseable timestamp in column {report['column']!r}")

            events, geom_note = _points_of(events)
            # METRIC WORK IN A METRIC CRS: cell edges are metres, so grid in UTM and only
            # convert the finished cells back to lon/lat. Never grid or buffer in degrees.
            metric, metric_epsg, crs_note = _to_metric(events)
            size_m = size_km * 1000.0
            xs = metric.geometry.x.to_numpy()
            ys = metric.geometry.y.to_numpy()
            import numpy as np

            ix = np.floor(xs / size_m).astype("int64")
            iy = np.floor(ys / size_m).astype("int64")

            period_start, labels = _period_labels(times, code)
            table = pd.DataFrame({"ix": ix, "iy": iy,
                                  "period": labels.to_numpy(),
                                  "period_start": period_start.to_numpy()})
            order = (table[["period", "period_start"]].drop_duplicates("period")
                     .sort_values("period_start")["period"].tolist())
            if len(order) < 2:
                raise ValueError(
                    f"only one {code!r} period ({order[0] if order else 'none'}) in this data, so "
                    "there is nothing earlier to compare the latest period against")
            latest, earlier = order[-1], order[:-1]

            counts = (table.groupby(["ix", "iy", "period"]).size()
                      .unstack("period", fill_value=0).reindex(columns=order, fill_value=0))
            latest_counts = counts[latest].astype("int64")
            # Mean over ALL earlier periods including the zeros: a cell that only lit up in
            # the latest period has a baseline of 0, which is the honest comparison.
            baseline = counts[earlier].mean(axis=1)
            shift = latest_counts - baseline
            pct = (shift / baseline.replace(0, pd.NA)) * 100.0
            total_events = counts.sum(axis=1).astype("int64")

            cells = pd.DataFrame({
                "cell_id": [f"{a}_{b}" for a, b in counts.index],
                "ix": [a for a, _ in counts.index],
                "iy": [b for _, b in counts.index],
                "latest_count": latest_counts.to_numpy(),
                "baseline_mean": baseline.round(3).to_numpy(),
                "shift": shift.round(3).to_numpy(),
                "pct_shift": pd.to_numeric(pct, errors="coerce").round(2).to_numpy(),
                "total_events": total_events.to_numpy(),
            })
            floor = max(1, int(min_events))
            cells = cells[cells["total_events"] >= floor]
            if cells.empty:
                raise ValueError(f"no grid cell has at least min_events={floor} record(s); "
                                 "lower min_events or increase cell_km")

            geometry = [box(a * size_m, b * size_m, (a + 1) * size_m, (b + 1) * size_m)
                        for a, b in zip(cells["ix"], cells["iy"])]
            grid = gpd.GeoDataFrame(cells.drop(columns=["ix", "iy"]), geometry=geometry,
                                    crs=metric_epsg)
            # Cell centres come from the indices (exact by construction) and are transformed as
            # arrays, so no centroid is ever taken in degrees.
            from pyproj import Transformer

            to_wgs84 = Transformer.from_crs(metric_epsg, "EPSG:4326", always_xy=True)
            lon, lat = to_wgs84.transform((cells["ix"].to_numpy() + 0.5) * size_m,
                                          (cells["iy"].to_numpy() + 0.5) * size_m)
            grid = grid.to_crs("EPSG:4326")                    # web maps are lon/lat
            grid["lon"] = np.round(lon, 6)
            grid["lat"] = np.round(lat, 6)

            csv_frame = pd.DataFrame(grid.drop(columns="geometry"))
            csv_rec = _publish_table(csv_frame, artifact_name(name, "csv", source=source,
                                                              default="temporal_hotspots"))
            gj_rec = _publish_geojson(grid, artifact_name(name, "geojson", source=source,
                                                          default="temporal_hotspots"))

            ranked = grid.sort_values("shift", ascending=False)
            def _cell_rows(rows: Any) -> List[Dict[str, Any]]:
                return [{"cell_id": str(r["cell_id"]), "lon": _num(r["lon"]), "lat": _num(r["lat"]),
                         "latest_count": int(r["latest_count"]),
                         "baseline_mean": _num(r["baseline_mean"]), "shift": _num(r["shift"]),
                         "pct_shift": _num(r["pct_shift"])} for _, r in rows.iterrows()]

            extreme = _num(grid["shift"].abs().max()) or 0.0
            label = (name or "").replace("_", " ").strip() or f"activity shift ({latest} vs earlier)"
            payload = {
                "ok": True, "source": source, "time_column": report["column"], "parse": report,
                "freq": str(freq), "freq_resolved": code,
                "cell_km": size_km, "cell_size_m": size_m,
                "metric_crs": metric_epsg, "grid_crs": metric_epsg,
                "periods": order, "latest_period": latest, "baseline_periods": earlier,
                "cells": int(len(grid)), "feature_count": int(len(grid)),
                "events_used": int(total_events.sum()),
                "excluded_unparsed_time": int(report["failed_rows"]),
                "cells_up": int((grid["shift"] > 0).sum()),
                "cells_down": int((grid["shift"] < 0).sum()),
                "file_id": gj_rec["file_id"], "filename": gj_rec.get("filename"),
                "download_url": gj_rec.get("download_url"),
                "csv": _asset(csv_rec), "geojson": _asset(gj_rec),
                "top_emerging": _cell_rows(ranked.head(5)),
                "top_cooling": _cell_rows(ranked.tail(5).iloc[::-1]),
                "crs": _epsg(getattr(grid, "crs", None)),
                "method": (
                    f"records were binned into square cells {size_km} km on a side in {metric_epsg} "
                    f"(a metric projection, NOT degrees) and counted per {code!r} period; "
                    f"shift = count in the latest period ({latest}) minus the mean count across the "
                    f"{len(earlier)} earlier period(s) {earlier[0]}..{earlier[-1]}, where a period "
                    "with no records in a cell counts as 0; pct_shift is null when the baseline was 0"),
                "map_layer": {"url": gj_rec.get("download_url"), "label": label,
                              "render": "choropleth", "style_by": "shift",
                              "source": "analysis", "count": int(len(grid)),
                              "diverging": True, "midpoint": 0,
                              "domain": [-extreme, extreme]},
                "on_map": True,
                "note": ("diverging choropleth of the shift: positive = more activity in the latest "
                         "period than its own earlier average, negative = less"),
            }
            for extra_note in (geom_note, crs_note):
                if extra_note:
                    payload.setdefault("notes", []).append(extra_note)
            if report["failed_rows"]:
                payload["warning"] = (f"{report['failed_rows']} record(s) had no usable timestamp "
                                      "and were excluded from the grid")
            return json.dumps(payload, default=str)
        except Exception as exc:  # noqa: BLE001
            return _fail(exc)

    meta = {"category": "geo"}
    return [
        StructuredTool.from_function(
            func=detect_time_column, name="detect_time_column", metadata=meta,
            description=("Find which column holds the DATE/TIME in a dataset and report the parsed "
                         "span (min -> max), granularity (year/month/day/hour/...), the parse method "
                         "used and how many rows failed to parse. Handles messy string formats such "
                         "as '07/26/2026 08:00:00 PM', ISO timestamps, bare years and epoch numbers. "
                         "Call this FIRST for any 'when / trend / recent / since' question before "
                         "filtering or charting by time.")),
        StructuredTool.from_function(
            func=filter_by_time, name="filter_by_time", metadata=meta,
            description=("Slice a dataset to a time window and put just that window on the user's "
                         "INTERACTIVE MAP (points/shapes, or heatmap for very large point sets). "
                         "start/end accept '2026-07-26', '2026-07', '2026' or a full timestamp, and "
                         "either can be omitted for an open-ended window; a plain date as `end` "
                         "covers that whole day. Returns a downloadable GeoJSON of the slice and "
                         "reports how many records were dropped for an unparseable time.")),
        StructuredTool.from_function(
            func=time_series, name="time_series", metadata=meta,
            description=("Count records per time period and return a CSV table plus a PNG line/bar "
                         "chart — use it for 'is this rising or falling?', 'what time of day?', "
                         "'which months?'. freq: hour|day|week|month|quarter|year, or a cyclical "
                         "profile hour_of_day|day_of_week|month_of_year. `by` splits the series by a "
                         "category column. This output is NON-SPATIAL: it produces a chart and a "
                         "table, not a map layer.")),
        StructuredTool.from_function(
            func=compare_periods, name="compare_periods", metadata=meta,
            description=("Compare two time windows per AREA: counts records inside each polygon of "
                         "an areas layer (tracts, neighborhoods, counties) in period_a and period_b "
                         "and maps the change as a DIVERGING choropleth centred on zero, styled by "
                         "the `change` column. Periods accept '2026-07', '2026' or ranges like "
                         "'2026-01-01..2026-06-30'. Also returns a CSV of area / a / b / change / "
                         "pct_change. Use it for 'which areas got better or worse?'.")),
        StructuredTool.from_function(
            func=temporal_hotspots, name="temporal_hotspots", metadata=meta,
            description=("Find WHERE activity concentrated in the most recent period compared with "
                         "the average of the earlier ones, without needing an areas layer: bins the "
                         "records into a square grid (`cell_km` kilometres on a side, computed in a "
                         "metric projection) and maps the per-cell shift as a diverging choropleth "
                         "styled by `shift`. States its method in the result and also returns a CSV "
                         "of every cell. Use it for 'where is this emerging / cooling off?'.")),
    ]


__all__ = ["make_temporal_tools", "parse_time_series"]
