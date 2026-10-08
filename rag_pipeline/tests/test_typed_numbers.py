"""Stage 43: a number carries its unit and where it was measured; nothing infers them.

docs/design-review-2026-10.md, flaw 1. The gate used to decide what a number was from two unit
vocabularies (113 + 25 entries that disagreed on 9 tokens), from column names (`area`, `dist`,
`_m`), and from a variable's name (`num_`, `count`). Each was extended after a correct answer
was flagged. These tests hold the replacement: a unit library parses any unit; a count is a
count of something; the operations say where a length was measured.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("pint")

from agent_runtime import declared_outputs, sandbox_verify as sv, turn_log
from agent_runtime.units import convert, parse_unit, typed_value


# --------------------------------------------------------------------------- any unit parses

@pytest.mark.parametrize("unit,dimension", [
    ("km²", "[length] ** 2"), ("km^2", "[length] ** 2"), ("km2", "[length] ** 2"),
    ("sq km", "[length] ** 2"), ("square_kilometers", "[length] ** 2"),
    ("square miles", "[length] ** 2"), ("mi2", "[length] ** 2"), ("m2", "[length] ** 2"),
    ("ha", "[length] ** 2"), ("acres", "[length] ** 2"), ("ft2", "[length] ** 2"),
    ("metres", "[length]"), ("kilometres", "[length]"), ("nautical_mile", "[length]"),
    ("furlongs", "[length]"), ("km/h", "[length] / [time]"), ("minutes", "[time]"),
    ("degrees", "dimensionless"), ("ppm", "dimensionless"), ("%", "dimensionless"),
])
def test_a_unit_parses_in_whatever_spelling_the_model_chose(unit, dimension):
    """Every one of these was, or would have been, "unrecognised unit; not checked"."""
    p = parse_unit(unit)
    assert p.kind == "unit" and p.dimension == dimension, p


@pytest.mark.parametrize("unit", ["schools", "records", "points", "pixels", "tracts", "index"])
def test_a_word_the_library_does_not_know_names_what_is_counted(unit):
    """`points` and `pixels` are printer's units in pint; read as lengths, a count of points
    would be 1/72 inch. pint's own Printer group says which units those are."""
    p = parse_unit(unit)
    assert p.kind == "label" and p.dimension == "dimensionless" and p.counted == unit


def test_a_counted_factor_inside_a_compound_keeps_its_dimension():
    assert parse_unit("person*km").dimension == "[length]"
    assert parse_unit("person-km").counted == "person"
    assert parse_unit("schools per km2").dimension == "1 / [length] ** 2"
    assert parse_unit("per 1,000 residents").dimension == "dimensionless"


@pytest.mark.parametrize("unit", ["km/hr^^", "EPSG:4326", "25 m", "12"])
def test_what_is_not_a_unit_fails_loudly(unit):
    assert parse_unit(unit).kind == "unparseable"


def test_conversion_comes_from_the_library():
    assert convert(2586.0, "km²", "sq mi") == pytest.approx(998.46, rel=1e-4)
    assert convert(1, "mi", "m") == pytest.approx(1609.344)
    assert convert(1, "km", "min") is None                        # different dimensions


def test_a_tool_cannot_emit_a_unit_that_does_not_parse():
    assert typed_value("a", 1.0, "km^2", measured_in_crs="EPSG:32616")["dimension"] == "[length] ** 2"
    with pytest.raises(ValueError):
        typed_value("a", 1.0, "km/hr^^")


# --------------------------------------------------------------------------- the lists are gone

@pytest.mark.parametrize("name", ["_UNIT_ALIASES", "_KNOWN_UNITS", "_METRIC_COLUMN_HINTS",
                                  "_has_metric_column", "_inferred_count",
                                  "check_declared_units", "check_count_population"])
def test_the_vocabularies_and_name_heuristics_are_deleted(name):
    """A structural guard: re-adding one of these is re-opening the patch loop."""
    assert not hasattr(sv, name)


def test_no_column_or_variable_name_reaches_a_verdict():
    """`pop_male` matched the `_m` hint, `district_id` matched `dist`, and `num_` decided
    whether a word was a count. The same outputs under any names now score the same."""
    a = {"area": {"value": 2.5, "unit": "km^2"}, "num_schools": {"value": 3, "unit": "schools"}}
    b = {"x1": {"value": 2.5, "unit": "km^2"}, "pop_male": {"value": 3, "unit": "schools"}}
    va, vb = (declared_outputs.evaluate(sv.run_checks({sv.DECLARED_OUTPUTS: o})) for o in (a, b))
    assert [f["status"] for f in va[0]] == [f["status"] for f in vb[0]]


# --------------------------------------------------------------------------- where it was measured

gpd = pytest.importorskip("geopandas")
from shapely.geometry import Point  # noqa: E402


def _run_script(tmp_path, monkeypatch, code):
    """Through the assembled prologue and epilogue, as the sandbox runs it."""
    from agent_runtime.code_execution import _read_checks

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "script.py"
    path.write_text(sv.prologue_source({}) + code + sv.epilogue_source(), encoding="utf-8")
    ns = {"__name__": "__main__"}
    exec(compile(path.read_text(), str(path), "exec"), ns)
    sv_state = __import__("sys").__dict__.get("_iguide_tracker_state")
    return _read_checks(tmp_path), sv_state


SCRIPT = '''
import geopandas as gpd
from shapely.geometry import Point
g = gpd.GeoDataFrame({"name": ["a", "b"]}, geometry=[Point(-88.2, 40.1), Point(-87.6, 41.9)],
                     crs=4326)
p = g.to_crs(32616)
area = float(p.buffer(1000).area.sum() / 1e6)
IGUIDE_OUTPUTS = {"total_area": {"value": area, "unit": "km²"},
                  "n": {"value": 2, "unit": "schools"},
                  "crs": {"value": "EPSG:32616", "unit": "crs"}}
'''


def test_a_measurement_carries_the_crs_its_operation_ran_in(tmp_path, monkeypatch):
    report, _ = _run_script(tmp_path, monkeypatch, SCRIPT)
    assert report["verdict"] == "pass", report["findings"]
    by_name = {o["name"]: o for o in report["outputs"]}
    assert by_name["total_area"]["measured_in_crs"] == "EPSG:32616"
    assert by_name["total_area"]["dimension"] == "[length] ** 2"
    assert by_name["n"]["counted"] == "schools"


def test_a_length_declared_in_a_geographic_crs_fails():
    rep = sv.run_checks({sv.DECLARED_OUTPUTS: {"d": {"value": 0.02, "unit": "km",
                                                       "crs": "EPSG:4326"}}})
    findings, _ = declared_outputs.evaluate(rep)
    assert any(f["check"] == "measured_in" and f["status"] == "fail" for f in findings)


def test_typed_outputs_become_facts_in_the_turn_log(tmp_path, monkeypatch):
    report, _ = _run_script(tmp_path, monkeypatch, SCRIPT)
    log = turn_log.new_log()
    log.record_result(peer="code", run=None, name="execute_code", args={"code": "x"},
                      call_id="c1", content=json.dumps({"ok": True, "verification": report}))
    facts = {f["name"]: f for f in log.facts()}
    assert facts["total_area"]["measured_in_crs"] == "EPSG:32616"
    assert facts["total_area"]["call_id"] == "c1"


def test_a_degree_buffer_still_fails_through_the_whole_path(tmp_path, monkeypatch):
    report, _ = _run_script(tmp_path, monkeypatch, SCRIPT + "bad = g.buffer(0.01)\n")
    assert report["verdict"] == "fail"
    assert any("g.buffer(0.01)" in f["message"] for f in report["findings"])


def test_without_the_unit_library_declarations_are_unknown_not_passed(tmp_path, monkeypatch):
    """An image built without pint must not pass a null unit by skipping the check."""
    from agent_runtime import code_execution

    (tmp_path / "checks.json").write_text(json.dumps({
        "verdict": "pass", "counts": {"pass": 1}, "findings": [],
        "declared": [{"name": "r", "value": 25000, "unit": None}]}))
    monkeypatch.setattr(declared_outputs, "evaluate", lambda r: (_ for _ in ()).throw(
        ImportError("No module named 'pint'")))
    report = code_execution._read_checks(tmp_path)
    assert report["verdict"] == "cannot_determine"
    assert "pint" in report["findings"][0]["message"]
