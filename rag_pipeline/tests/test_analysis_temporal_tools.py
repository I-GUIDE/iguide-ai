"""Tests for the temporal analysis tools (agent_runtime/analysis_temporal_tools.py).

Tiny synthetic GeoDataFrames only — no network, no big fixtures. The interesting cases are
the ones that used to bite: a Chicago-style ``07/26/2026 08:00:00 PM`` string column, rows
that do NOT parse at all, a column name that does not exist, and grid cells that must be
1 km on a side in metres rather than 1 degree.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

gpd = pytest.importorskip("geopandas")
pytest.importorskip("pyogrio")
pytest.importorskip("matplotlib")
import pandas as pd  # noqa: E402
from shapely.geometry import Point, box  # noqa: E402
from werkzeug.datastructures import FileStorage  # noqa: E402

# Two clusters ~3 km apart in Chicago, so a 1 km grid separates them.
_WEST = (-87.700, 41.900)
_EAST = (-87.660, 41.900)


def _tools():
    from agent_runtime.analysis_temporal_tools import make_temporal_tools
    return {t.name: t for t in make_temporal_tools()}


def _upload(path: Path) -> str:
    from agent_runtime.file_store import save_uploaded_file
    with open(path, "rb") as fh:
        return save_uploaded_file(FileStorage(stream=fh, filename=Path(path).name))["file_id"]


def _read_output(file_id: str):
    """Read a published output artifact back off disk (asserting on the REAL file)."""
    from agent_runtime.file_store import resolve_file_id
    return resolve_file_id(file_id)


def _call(tool, **kwargs):
    return json.loads(tool.invoke(kwargs))


@pytest.fixture()
def store(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "store"))
    work = tmp_path / "work"
    work.mkdir()
    return work


def _incident_frame() -> gpd.GeoDataFrame:
    """12 incidents in Chicago-crime style: ``MM/DD/YYYY hh:mm:ss AM/PM`` strings.

    Design: Jan/Feb/Mar 2026 are quiet, July 2026 is busy AND shifted east, so
    time_series shows a rise, compare_periods shows +east/-west, and temporal_hotspots
    puts the emerging cell in the east.
    """
    rows = [
        # ---- period A: Jan-Mar 2026, mostly WEST
        ("01/05/2026 09:00:00 AM", "theft", *_WEST, 4.0),
        ("01/20/2026 11:00:00 PM", "theft", *_WEST, 6.0),
        ("02/10/2026 08:00:00 AM", "assault", *_WEST, 2.0),
        ("03/03/2026 02:00:00 PM", "theft", *_EAST, 7.0),
        # ---- period B: July 2026, mostly EAST and more of it
        ("07/01/2026 10:00:00 PM", "theft", *_EAST, 9.0),
        ("07/04/2026 08:00:00 PM", "theft", *_EAST, 5.0),
        ("07/12/2026 09:00:00 PM", "assault", *_EAST, 3.0),
        ("07/18/2026 07:00:00 AM", "theft", *_EAST, 8.0),
        ("07/26/2026 08:00:00 PM", "assault", *_EAST, 1.0),
        ("07/28/2026 08:00:00 PM", "theft", *_WEST, 2.0),
        # ---- two rows whose time is unusable: they must be REPORTED, never silently kept
        ("not a date", "theft", *_EAST, 5.0),
        (None, "theft", *_WEST, 5.0),
    ]
    frame = pd.DataFrame(rows, columns=["occurred_on", "kind", "lon", "lat", "value"])
    return gpd.GeoDataFrame(
        frame,
        geometry=[Point(x, y) for x, y in zip(frame["lon"], frame["lat"])],
        crs="EPSG:4326",
    )


@pytest.fixture()
def incidents(store) -> str:
    path = store / "incidents.geojson"
    _incident_frame().to_file(path, driver="GeoJSON")
    return _upload(path)


@pytest.fixture()
def areas(store) -> str:
    """Two adjacent rectangles: 'West side' holds _WEST, 'East side' holds _EAST."""
    path = store / "areas.geojson"
    gpd.GeoDataFrame(
        {"name": ["West side", "East side"]},
        geometry=[box(-87.72, 41.88, -87.68, 41.92), box(-87.68, 41.88, -87.64, 41.92)],
        crs="EPSG:4326",
    ).to_file(path, driver="GeoJSON")
    return _upload(path)


# --------------------------------------------------------------------------- factory
def test_factory_shape():
    tools = _tools()
    assert set(tools) == {"detect_time_column", "filter_by_time", "time_series",
                          "compare_periods", "temporal_hotspots"}
    assert all(getattr(t, "metadata", {}).get("category") == "geo" for t in tools.values())
    assert all((t.description or "").strip() for t in tools.values())


# ----------------------------------------------------------------------- parsing
def test_parse_time_series_handles_chicago_format():
    from agent_runtime.analysis_temporal_tools import parse_time_series

    parsed, method = parse_time_series(pd.Series(["07/26/2026 08:00:00 PM", "01/05/2026 09:00:00 AM"]))
    assert list(parsed) == [pd.Timestamp("2026-07-26 20:00"), pd.Timestamp("2026-01-05 09:00")]
    assert method != "unparsed"


def test_parse_time_series_never_raises_on_garbage():
    from agent_runtime.analysis_temporal_tools import parse_time_series

    parsed, method = parse_time_series(pd.Series(["nope", "", None]))
    assert parsed.isna().all() and method == "unparsed"


def test_parse_time_series_reads_epoch_and_years():
    from agent_runtime.analysis_temporal_tools import parse_time_series

    epoch, _ = parse_time_series(pd.Series([1_767_225_600, 1_769_904_000]))
    assert epoch.dt.year.tolist() == [2026, 2026]
    years, _ = parse_time_series(pd.Series([2019, 2020, 2021]))
    assert years.dt.year.tolist() == [2019, 2020, 2021]


def test_parse_time_series_never_reads_codes_as_years():
    """Beats, ids and clock times are numbers, not dates, whether typed or CSV text.

    The deployed failure: pandas 3 parses text at microsecond resolution, so "1234" became the
    year 1234 where pandas 2 had returned NaT. Chicago's real beats run 111-2535, so the year
    branch declines them, and they used to fall through to the text ladder instead.
    """
    from agent_runtime.analysis_temporal_tools import parse_time_series

    chicago_beats = [111, 112, 1011, 1834, 2535, 421, 631, 1124] * 5
    for values in ([1234, 1235, 1236],      # the failing fixture: inside the year branch's range
                   chicago_beats,           # below 1000: only the text fall-through could take them
                   [1001, 1002, 1003],      # incident ids
                   [830, 1415, 2359, 5]):   # HHMM clock times
        for series in (pd.Series(values), pd.Series([str(v) for v in values])):
            parsed, method = parse_time_series(series)
            assert method == "unparsed" and parsed.isna().all(), (values[:4], str(series.dtype), method)


def test_parse_time_series_reads_yyyymmdd_numbers():
    """An 8-digit date stored as a number still parses, typed or as CSV text, by an explicit rule."""
    from agent_runtime.analysis_temporal_tools import parse_time_series

    expected = [pd.Timestamp("2019-01-01"), pd.Timestamp("2019-02-15"), pd.Timestamp("2020-12-31")]
    for series in (pd.Series([20190101, 20190215, 20201231]),
                   pd.Series(["20190101", "20190215", "20201231"])):
        parsed, method = parse_time_series(series)
        assert list(parsed) == expected and method == "YYYYMMDD number"
    parsed, _ = parse_time_series(pd.Series([20190101, 20191340]))   # month 13 is a failed row
    assert parsed.iloc[0] == pd.Timestamp("2019-01-01") and pd.isna(parsed.iloc[1])


def test_parse_time_series_reads_numbers_that_arrive_as_text():
    """GDAL hands every CSV field over as text: a year column, one with a gap, epoch seconds."""
    from agent_runtime.analysis_temporal_tools import parse_time_series

    years, method = parse_time_series(pd.Series(["2019", "2020", "2021"]))
    assert method == "year number" and years.dt.year.tolist() == [2019, 2020, 2021]
    gappy, method = parse_time_series(pd.Series(["2019.0", "", "2021.0"]))   # how to_csv writes NaN
    assert method == "year number" and gappy.dt.year.tolist()[::2] == [2019, 2021]
    assert pd.isna(gappy.iloc[1])
    epoch, method = parse_time_series(pd.Series(["1767225600", "1769904000"]))
    assert method == "epoch seconds" and epoch.dt.year.tolist() == [2026, 2026]


def test_parse_time_series_year_window_does_not_move_with_pandas():
    """1678-2262 on every pandas version: the window pandas 2 enforced implicitly, now explicit."""
    from agent_runtime.analysis_temporal_tools import parse_time_series

    edges, method = parse_time_series(pd.Series([1677, 1678, 2262, 2263]))
    assert method == "year number"
    assert [None if pd.isna(v) else v.year for v in edges] == [None, 1678, 2262, None]
    text, _ = parse_time_series(pd.Series(["1556-01-23", "1700-01-26"]))   # text is held to it too
    assert pd.isna(text.iloc[0]) and text.iloc[1] == pd.Timestamp("1700-01-26")


def test_parse_time_series_converts_mixed_offsets_to_utc():
    """Offsets that differ within one column, as a US feed's do either side of daylight saving.

    The deployed failure: pandas 3 raises "Mixed timezones detected" where pandas 2 returned
    objects, so every text strategy failed and the column read as 0% parsed.
    """
    from agent_runtime.analysis_temporal_tools import parse_time_series

    dst, method = parse_time_series(pd.Series(["2026-01-05T09:00:00-06:00", "2026-07-26T20:00:00-05:00"]))
    assert method == "inferred single format"
    assert list(dst) == [pd.Timestamp("2026-01-05 15:00"), pd.Timestamp("2026-07-27 01:00")]
    mixed, _ = parse_time_series(pd.Series(["2026-07-26T20:00:00Z", "2026-01-05T09:00:00+01:00"]))
    assert list(mixed) == [pd.Timestamp("2026-07-26 20:00"), pd.Timestamp("2026-01-05 08:00")]
    assert dst.dt.tz is None and mixed.dt.tz is None
    # Text with no offset beside text with one is read as UTC, as pandas 2 always did; pandas 3
    # raised here too, and the unfixed parser kept only the rows the first format fitted.
    both, _ = parse_time_series(pd.Series(["2026-01-05 09:00:00", "2026-07-26T20:00:00-05:00"]))
    assert list(both) == [pd.Timestamp("2026-01-05 09:00"), pd.Timestamp("2026-07-27 01:00")]


def test_parse_time_series_utc_leaves_other_columns_as_they_were():
    """One offset is still converted to UTC, and text without an offset keeps its clock time."""
    from agent_runtime.analysis_temporal_tools import parse_time_series

    single, _ = parse_time_series(pd.Series(["2026-01-05T09:00:00-06:00", "2026-02-05T09:00:00-06:00"]))
    assert list(single) == [pd.Timestamp("2026-01-05 15:00"), pd.Timestamp("2026-02-05 15:00")]
    naive, method = parse_time_series(pd.Series(["2026-01-05T09:00:00", "2026-07-26T20:00:00"]))
    assert method == "inferred single format"
    assert list(naive) == [pd.Timestamp("2026-01-05 09:00"), pd.Timestamp("2026-07-26 20:00")]


# ----------------------------------------------------------------- detect_time_column
def test_detect_time_column(incidents):
    res = _call(_tools()["detect_time_column"], file_id=incidents)
    assert res["ok"] is True and res["found"] is True
    assert res["time_column"] == "occurred_on"
    best = res["candidates"][0]
    assert best["rows"] == 12 and best["parsed_rows"] == 10 and best["failed_rows"] == 2
    assert best["granularity"] == "hour"                      # every value is on the hour
    assert best["span"]["start"].startswith("2026-01-05")
    assert best["span"]["end"].startswith("2026-07-28")
    assert res["warning"] and "2 row" in res["warning"]       # dirty rows are surfaced
    # 'kind' is text but parses as nothing, so it must not be offered as a time column.
    assert "kind" not in [c["column"] for c in res["candidates"]]


def test_detect_time_column_bad_column_lists_candidates(incidents):
    res = _call(_tools()["detect_time_column"], file_id=incidents, time_column="when_ever")
    assert res["ok"] is False
    assert "occurred_on" in res["candidates"]
    assert "occurred_on" in res["time_column_candidates"]


# -------------------------------------------------------------------- filter_by_time
def test_filter_by_time_happy_path_map_layer(incidents):
    res = _call(_tools()["filter_by_time"], file_id=incidents,
                start="2026-07-01", end="2026-07-26", style_by="value", name="july_window")
    assert res["ok"] is True
    assert res["matched"] == 5                                # 07/01..07/26 inclusive of the 26th
    assert res["excluded_unparsed_time"] == 2
    layer = res["map_layer"]
    assert layer["render"] == "points" and layer["source"] == "analysis"
    assert layer["url"] == res["download_url"] and layer["count"] == 5
    # the styling column must EXIST in the written output, not just be named in the descriptor
    out = gpd.read_file(_read_output(res["file_id"]))
    assert layer["style_by"] in out.columns
    assert len(out) == 5
    assert res["crs"] == "EPSG:4326"                           # web maps need lon/lat
    assert Path(res["filename"]).name == "july_window.geojson"


def test_filter_by_time_plain_end_date_covers_whole_day(incidents):
    """end='2026-07-26' must keep the 08:00 PM record on the 26th, not drop it at midnight."""
    res = _call(_tools()["filter_by_time"], file_id=incidents, start="2026-07-26", end="2026-07-26")
    assert res["ok"] is True and res["matched"] == 1
    assert res["window"]["end"].startswith("2026-07-26T23:59")


def test_filter_by_time_month_and_open_ended(incidents):
    tools = _tools()
    month = _call(tools["filter_by_time"], file_id=incidents, start="2026-07")
    assert month["ok"] is True and month["matched"] == 6       # all of July
    everything = _call(tools["filter_by_time"], file_id=incidents)
    assert everything["matched"] == 10                        # the 2 unparseable rows are out


def test_filter_by_time_bad_time_column(incidents):
    res = _call(_tools()["filter_by_time"], file_id=incidents, time_column="nope")
    assert res["ok"] is False
    assert "TimeColumnError" in res["error"]
    assert "occurred_on" in res["candidates"]


def test_filter_by_time_bad_style_column_lists_numeric(incidents):
    res = _call(_tools()["filter_by_time"], file_id=incidents, style_by="valu")
    assert res["ok"] is False
    assert "value" in res["numeric_columns"] and "value" in res["candidates"]


def test_filter_by_time_empty_window_is_not_a_crash(incidents):
    res = _call(_tools()["filter_by_time"], file_id=incidents, start="1990", end="1991")
    assert res["ok"] is True and res["feature_count"] == 0
    assert res["on_map"] is False and "map_layer" not in res
    assert res["available_span"]["start"].startswith("2026-01-05")


# ----------------------------------------------------------------------- time_series
def test_time_series_csv_and_png(incidents):
    res = _call(_tools()["time_series"], file_id=incidents, freq="month", name="monthly")
    assert res["ok"] is True
    assert res["spatial"] is False and "map_layer" not in res
    assert "NON-SPATIAL" in res["note"]
    assert res["freq_resolved"] == "M"
    assert res["records_counted"] == 10
    assert res["peak"] == {"period": "2026-07", "count": 6}
    counts = {row["period"]: row["count"] for row in res["series"]}
    assert counts == {"2026-01": 2, "2026-02": 1, "2026-03": 1, "2026-07": 6}
    table = pd.read_csv(_read_output(res["csv"]["file_id"]))
    assert list(table.columns) == ["period", "count"]
    assert int(table["count"].sum()) == 10
    png = Path(_read_output(res["chart"]["file_id"]))
    assert png.suffix == ".png" and png.stat().st_size > 1000   # a real rendered chart
    assert res["file_id"] == res["csv"]["file_id"]              # top-level asset is the table


def test_time_series_profiles(incidents):
    tools = _tools()
    hod = _call(tools["time_series"], file_id=incidents, freq="hour_of_day")
    assert hod["ok"] is True and hod["chart_kind"] == "bar"
    assert hod["periods"] == 24                                # empty hours are kept as zeros
    hours = {row["period"]: row["count"] for row in hod["series"]}
    # 8 PM must land in hour 20, not hour 8 — the AM/PM half of the format is load-bearing.
    assert hours["20"] == 3 and hours["22"] == 1 and hours["23"] == 1
    assert hours["09"] == 1 and hours["08"] == 1 and hours["00"] == 0
    dow = _call(tools["time_series"], file_id=incidents, freq="day_of_week")
    assert [row["period"] for row in dow["series"]][:2] == ["Monday", "Tuesday"]
    assert sum(row["count"] for row in dow["series"]) == 10


def test_time_series_grouped_by_category(incidents):
    res = _call(_tools()["time_series"], file_id=incidents, freq="month", by="kind")
    assert res["ok"] is True and res["by"] == "kind"
    table = pd.read_csv(_read_output(res["csv"]["file_id"]))
    assert {"period", "assault", "theft", "total"} <= set(table.columns)
    assert int(table["total"].sum()) == 10


def test_time_series_bad_by_column_lists_candidates(incidents):
    res = _call(_tools()["time_series"], file_id=incidents, by="category")
    assert res["ok"] is False and "ColumnError" in res["error"]
    assert "kind" in res["candidates"]


def test_time_series_bad_freq(incidents):
    res = _call(_tools()["time_series"], file_id=incidents, freq="fortnight")
    assert res["ok"] is False and "hour_of_day" in res["error"]


def test_time_series_works_without_geometry(store):
    """A plain CSV with no coordinates is still chartable — time_series is non-spatial."""
    path = store / "plain.csv"
    pd.DataFrame({"reported_date": ["2026-01-02", "2026-01-09", "2026-02-03"],
                  "kind": ["a", "b", "a"]}).to_csv(path, index=False)
    res = _call(_tools()["time_series"], file_id=_upload(path), freq="month")
    assert res["ok"] is True and res["records_counted"] == 3
    assert res["time_column"] == "reported_date"


# -------------------------------------------------------------------- compare_periods
def test_compare_periods_diverging_choropleth(incidents, areas):
    res = _call(_tools()["compare_periods"], file_id=incidents, areas_file_id=areas,
                period_a="2026-01-01..2026-03-31", period_b="2026-07", name="q1_vs_july")
    assert res["ok"] is True
    layer = res["map_layer"]
    assert layer["render"] == "choropleth" and layer["style_by"] == "change"
    assert layer["diverging"] is True and layer["midpoint"] == 0
    out = gpd.read_file(_read_output(res["file_id"]))
    # A choropleth MUST style by a numeric column that exists in the output.
    assert layer["style_by"] in out.columns
    assert pd.api.types.is_numeric_dtype(out[layer["style_by"]])
    assert res["feature_count"] == 2 and len(out) == 2

    per_area = out.set_index("name")
    assert per_area.loc["West side", "count_a"] == 3 and per_area.loc["West side", "count_b"] == 1
    assert per_area.loc["East side", "count_a"] == 1 and per_area.loc["East side", "count_b"] == 5
    assert per_area.loc["West side", "change"] == -2
    assert per_area.loc["East side", "change"] == 4
    assert res["areas_increased"] == 1 and res["areas_decreased"] == 1
    assert res["top_increases"][0]["area"] == "East side"
    assert res["top_decreases"][0]["area"] == "West side"

    table = pd.read_csv(_read_output(res["csv"]["file_id"]))
    assert list(table.columns) == ["area", "a", "b", "change", "pct_change"]
    east = table.set_index("area").loc["East side"]
    assert east["change"] == 4 and east["pct_change"] == pytest.approx(400.0)
    assert "period_b minus" in res["method"]


def test_compare_periods_zero_baseline_has_null_pct(incidents, areas):
    """period_a with no records anywhere: pct_change has no baseline, so it must be null."""
    res = _call(_tools()["compare_periods"], file_id=incidents, areas_file_id=areas,
                period_a="2025", period_b="2026-07")
    assert res["ok"] is True
    table = pd.read_csv(_read_output(res["csv"]["file_id"]))
    assert table["a"].sum() == 0
    assert table["pct_change"].isna().all()
    assert res["top_increases"][0]["pct_change"] is None


def test_compare_periods_output_is_strict_json(incidents, areas):
    """NaN is not valid JSON: a null pct_change must serialise as `null` in the .geojson."""
    res = _call(_tools()["compare_periods"], file_id=incidents, areas_file_id=areas,
                period_a="2025", period_b="2026-07")
    raw = json.loads(Path(_read_output(res["file_id"])).read_text(encoding="utf-8"))
    props = [f["properties"] for f in raw["features"]]
    assert all(p["pct_change"] is None for p in props)
    assert {"count_a", "count_b", "change", "pct_change", "name"} <= set(props[0])


def test_compare_periods_bad_period_text(incidents, areas):
    res = _call(_tools()["compare_periods"], file_id=incidents, areas_file_id=areas,
                period_a="whenever", period_b="2026-07")
    assert res["ok"] is False and "2026-07-26" in res["error"]  # the error shows the accepted forms


# ------------------------------------------------------------------ temporal_hotspots
def test_temporal_hotspots_shift_choropleth(incidents):
    res = _call(_tools()["temporal_hotspots"], file_id=incidents, freq="month", cell_km=1.0)
    assert res["ok"] is True
    layer = res["map_layer"]
    assert layer["render"] == "choropleth" and layer["style_by"] == "shift"
    assert layer["diverging"] is True
    out = gpd.read_file(_read_output(res["file_id"]))
    assert layer["style_by"] in out.columns
    assert pd.api.types.is_numeric_dtype(out[layer["style_by"]])
    assert res["latest_period"] == "2026-07"
    assert res["baseline_periods"] == ["2026-01", "2026-02", "2026-03"]
    # The east cluster fires in July only -> it is the emerging cell.
    top = res["top_emerging"][0]
    assert top["latest_count"] == 5 and top["shift"] > 0
    assert top["lon"] == pytest.approx(_EAST[0], abs=0.01)
    assert res["cells"] == len(out) == 2
    assert res["events_used"] == 10
    assert "metric projection" in res["method"] and "1.0 km" in res["method"]
    table = pd.read_csv(_read_output(res["csv"]["file_id"]))
    assert {"cell_id", "latest_count", "baseline_mean", "shift", "pct_shift", "lon", "lat"} <= set(table.columns)


def test_temporal_hotspots_cells_are_metric_not_degrees(incidents):
    """cell_km=1 must mean 1 kilometre, measured in a projected CRS — never 1 degree."""
    res = _call(_tools()["temporal_hotspots"], file_id=incidents, freq="month", cell_km=1.0)
    assert res["ok"] is True
    assert res["metric_crs"].startswith("EPSG:326")            # a real UTM zone, not 4326
    assert res["cell_size_m"] == 1000.0
    cells = gpd.read_file(_read_output(res["file_id"]))
    assert cells.crs.to_epsg() == 4326                          # output ships as lon/lat

    metric = cells.to_crs(res["metric_crs"])
    one = metric.geometry.iloc[0]
    minx, miny, maxx, maxy = one.bounds
    assert (maxx - minx) == pytest.approx(1000.0, rel=0.001)    # 1 km wide in metres
    assert (maxy - miny) == pytest.approx(1000.0, rel=0.001)
    # In degrees a 1 km cell is ~0.012 deg of longitude at this latitude. A degree-based
    # buffer would have produced a ~1.0 deg box (~80x too big), so bound it hard.
    lon_minx, _, lon_maxx, _ = cells.geometry.iloc[0].bounds
    assert (lon_maxx - lon_minx) < 0.02


def test_temporal_hotspots_cell_km_scales(incidents):
    """A coarser cell_km really produces coarser cells (the clusters are 3.3 km apart)."""
    res = _call(_tools()["temporal_hotspots"], file_id=incidents, freq="month", cell_km=10.0)
    assert res["ok"] is True and res["cell_size_m"] == 10000.0
    metric = gpd.read_file(_read_output(res["file_id"])).to_crs(res["metric_crs"])
    minx, _, maxx, _ = metric.geometry.iloc[0].bounds
    assert (maxx - minx) == pytest.approx(10000.0, rel=0.001)
    # The grid is aligned to the projection origin, so a coarser grid merges neighbours but
    # can never split them: 10 km swallows both clusters into one cell.
    assert res["cells"] == 1
    assert res["top_emerging"][0]["latest_count"] == 6


def test_temporal_hotspots_single_period_is_explained(incidents):
    res = _call(_tools()["temporal_hotspots"], file_id=incidents, freq="year")
    assert res["ok"] is False
    assert "nothing earlier to compare" in res["error"]


def test_temporal_hotspots_rejects_bad_arguments(incidents):
    tools = _tools()
    bad_cell = _call(tools["temporal_hotspots"], file_id=incidents, cell_km=0)
    assert bad_cell["ok"] is False and "cell_km" in bad_cell["error"]
    profile = _call(tools["temporal_hotspots"], file_id=incidents, freq="hour_of_day")
    assert profile["ok"] is False and "chronological" in profile["error"]


# --------------------------------------------------------------- tabular (CSV) input
def test_csv_with_coordinates_flows_through(store):
    """The real target shape: a CSV of Chicago-style strings plus Latitude/Longitude."""
    path = store / "crimes.csv"
    pd.DataFrame({
        "ID": [1, 2, 3],
        "Date": ["07/26/2026 08:00:00 PM", "07/27/2026 01:00:00 AM", "01/02/2026 08:00:00 AM"],
        "Latitude": [41.900, 41.901, 41.902],
        "Longitude": [-87.660, -87.661, -87.662],
        "Beat": [1234, 1235, 1236],           # numeric, no time-ish name -> never a candidate
    }).to_csv(path, index=False)
    file_id = _upload(path)
    tools = _tools()

    detected = _call(tools["detect_time_column"], file_id=file_id)
    assert detected["time_column"] == "Date" and detected["spatial"] is True
    assert "Beat" not in [c["column"] for c in detected["candidates"]]

    sliced = _call(tools["filter_by_time"], file_id=file_id, start="2026-07")
    assert sliced["ok"] is True and sliced["matched"] == 2
    assert sliced["map_layer"]["render"] == "points"
    assert len(gpd.read_file(_read_output(sliced["file_id"]))) == 2


def test_csv_beat_cannot_outrank_a_date_with_a_blank_row(store):
    """The deployed failure in full: one blank date, and the beats became THE time column.

    GDAL hands the blank cell over as "", which counts against Date: 3 of 4 rows parse. On
    pandas 3, before the name gate covered a CSV's numbers, the beats parsed 4 of 4 as the years
    1234-1237 and outranked it. The test above, whose Date has no gap, could only ever catch the
    beats as a runner-up, because there the name hint broke a 1.0 against 1.0 tie.
    """
    path = store / "crimes_with_gap.csv"
    pd.DataFrame({
        "ID": [1, 2, 3, 4],
        "Date": ["07/26/2026 08:00:00 PM", "07/27/2026 01:00:00 AM", None,
                 "01/02/2026 08:00:00 AM"],
        "Latitude": [41.900, 41.901, 41.902, 41.903],
        "Longitude": [-87.660, -87.661, -87.662, -87.663],
        "Beat": [1234, 1235, 1236, 1237],
    }).to_csv(path, index=False)
    detected = _call(_tools()["detect_time_column"], file_id=_upload(path))
    assert detected["ok"] is True and detected["time_column"] == "Date"
    assert "Beat" not in [c["column"] for c in detected["candidates"]]
    date = detected["candidates"][0]
    assert date["parsed_rows"] == 3 and date["failed_rows"] == 1   # the blank is reported, not hidden


def test_csv_code_columns_are_never_time_candidates(store):
    """Every kind of code column a crime CSV carries, read the way GDAL reads them: as text.

    On the deployed image (pandas 3) the unfixed tool picked incident_id as THE time column,
    and the IUCR codes tied this Date column, which has one dirty row.
    """
    path = store / "codes.csv"
    pd.DataFrame({
        "incident_id": [1001, 1002, 1003, 1004, 1005],    # time-ish name, so it IS tried, and fails
        "Date": ["07/26/2026 08:00:00 PM", "07/27/2026 01:00:00 AM", "01/02/2026 08:00:00 AM",
                 "02/03/2026 09:30:00 AM", "not a date"],
        "IUCR": ["0486", "0820", "1310", "041A", "2820"],  # mostly digits, so still text
        "Beat": [111, 1834, 2535, 421, 1124],
        "Ward": [1, 2, 3, 4, 5],
        "Latitude": [41.90, 41.91, 41.92, 41.93, 41.94],
        "Longitude": [-87.66, -87.67, -87.68, -87.69, -87.70],
    }).to_csv(path, index=False)
    res = _call(_tools()["detect_time_column"], file_id=_upload(path))
    assert res["ok"] is True and res["time_column"] == "Date"
    assert [c["column"] for c in res["candidates"]] == ["Date"]


@pytest.mark.parametrize("column, suffix", [
    pytest.param("yr", ".csv", id="csv-yr"),             # the text ladder read it with no hint
    pytest.param("YRBUILT", ".csv", id="csv-YRBUILT"),   # run-together, so the hint is a substring
    pytest.param("yr", ".geojson", id="geojson-yr"),     # typed integers: never even tried before
])
def test_year_column_named_yr_is_a_time_column(store, column, suffix):
    """A year column's name is all that gets it past the name gate, so yr has to be a hint.

    These beats sit inside 1678-2262, so the year window cannot be what keeps them out: only the
    gate does, and the year column faces the same gate. Without the hint, both CSV shapes came
    back with no time column at all, where the text ladder used to read them as years.
    """
    path = store / f"annual{suffix}"
    frame = pd.DataFrame({
        column: [2019, 2020, 2021],
        "Beat": [1834, 2011, 2012],
        "Latitude": [41.900, 41.901, 41.902],
        "Longitude": [-87.660, -87.661, -87.662],
    })
    if suffix == ".csv":
        frame.to_csv(path, index=False)
    else:
        gpd.GeoDataFrame(frame, geometry=gpd.points_from_xy(frame["Longitude"], frame["Latitude"]),
                         crs="EPSG:4326").to_file(path, driver="GeoJSON")
    res = _call(_tools()["detect_time_column"], file_id=_upload(path))
    assert res["ok"] is True and res["time_column"] == column
    assert [c["column"] for c in res["candidates"]] == [column]
    best = res["candidates"][0]
    assert best["parse_method"] == "year number" and best["granularity"] == "year"


def test_csv_year_column_is_still_a_time_column(store):
    """A CSV's year column is text holding numbers too; its NAME is what keeps it detectable."""
    path = store / "annual.csv"
    pd.DataFrame({
        "Year": [2019, 2020, 2021],
        "Beat": [1234, 1235, 1236],
        "Latitude": [41.900, 41.901, 41.902],
        "Longitude": [-87.660, -87.661, -87.662],
    }).to_csv(path, index=False)
    res = _call(_tools()["detect_time_column"], file_id=_upload(path))
    assert res["ok"] is True and res["time_column"] == "Year"
    assert [c["column"] for c in res["candidates"]] == ["Year"]
    best = res["candidates"][0]
    assert best["parse_method"] == "year number" and best["granularity"] == "year"
    assert best["span"]["start"].startswith("2019-01-01") and best["span"]["end"].startswith("2021-01-01")


def test_csv_with_daylight_saving_offsets_is_a_time_column(store):
    """A US feed's local times: -06:00 in winter, -05:00 in summer, read as text like every CSV field.

    On the deployed image (pandas 3) the unfixed tool reported that no time column existed.
    """
    path = store / "dst.csv"
    pd.DataFrame({
        "reported_at": ["2026-01-05T09:00:00-06:00", "2026-03-10T23:30:00-05:00",
                        "2026-07-26T20:00:00-05:00", "2026-11-02T08:00:00-06:00"],
        "Latitude": [41.90, 41.91, 41.92, 41.93],
        "Longitude": [-87.66, -87.67, -87.68, -87.69],
    }).to_csv(path, index=False)
    file_id = _upload(path)
    tools = _tools()

    detected = _call(tools["detect_time_column"], file_id=file_id)
    assert detected["ok"] is True and detected["time_column"] == "reported_at"
    best = detected["candidates"][0]
    assert best["parsed_rows"] == 4 and best["failed_rows"] == 0
    assert best["span"]["start"].startswith("2026-01-05T15:00")     # in UTC
    # Windows apply in UTC too: 23:30 at -05:00 on March 10 is 04:30 UTC on March 11.
    sliced = _call(tools["filter_by_time"], file_id=file_id, start="2026-03-11", end="2026-03-11")
    assert sliced["ok"] is True and sliced["matched"] == 1


def test_unknown_file_id_fails_cleanly():
    for name, kwargs in (
        ("detect_time_column", {"file_id": "file_nope"}),
        ("filter_by_time", {"file_id": "file_nope"}),
        ("time_series", {"file_id": "file_nope"}),
        ("temporal_hotspots", {"file_id": "file_nope"}),
    ):
        res = _call(_tools()[name], **kwargs)
        assert res["ok"] is False and "file_nope" in res["error"]
