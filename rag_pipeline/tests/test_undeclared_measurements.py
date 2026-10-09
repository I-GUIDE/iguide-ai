"""Measured numbers reach the gate without the model declaring them.

The gate's unit and CRS checks ran only on numbers a script assigned to `IGUIDE_OUTPUTS`, and in
the GIS harness no model ever did (0 of 96 gate reports), so those checks never ran. Two routes
now carry measured numbers to the gate without any declaration or prompt:

* what the run PRINTED with a unit the unit library parses ("Area: 2,586.01 km²",
  `area_km2=2586.0`, `{"distance_m": 412.5}`);
* what the MEASURING CALLS returned (`.area`, `.length`, `.distance`), recorded in the sandbox with
  the CRS and the CRS's linear unit, so a printed figure inherits the CRS it was measured in.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("pint")

from agent_runtime import facts as turn_facts, measured_outputs, sandbox_verify as sv, turn_log


# --------------------------------------------------------------------------- printed figures

@pytest.mark.parametrize("line,value,unit", [
    ("Area: 2,586.01 km²", 2586.01, "km²"),
    ("Total population-weighted distance: 382254.14 person-km", 382254.14, "person-km"),
    ("area_km2=2586.008615849", 2586.008615849, "km2"),
    ('{"distance_m": 412.52746200242365, "name": "School 08"}', 412.52746200242365, "m"),
    ("Watershed area (km²): 16.2", 16.2, "km²"),
    ("total valid area ha: 99.0", 99.0, "ha"),
])
def test_a_printed_figure_carries_the_unit_printed_with_it(line, value, unit):
    got = measured_outputs.printed_quantities(line)
    assert [(q["value"], q["unit"]) for q in got] == [(value, unit)]


@pytest.mark.parametrize("line,value", [
    ("TOTAL schools within 1 mile: 19", 19),     # the unit belongs to the 1, not the 19
    ("Events within 7 days AND within 25 km: 673", 673),
    ("band 2: min=0, max=5343, dtype=uint16", 0),  # min is a minimum, not minutes
    ("C1 + C2: 516771.47", 516771.47),           # a site label, not coulombs squared
    ("crs EPSG:32616", 32616),                   # an identifier
    ("(199,75) acc=18000 elev=200.3", 18000),
    ("nearest school 41.8796 N, 87.6261 W", 41.8796),  # a latitude, not newtons
])
def test_a_label_is_not_read_as_the_unit_of_a_number_it_does_not_measure(line, value):
    got = measured_outputs.printed_quantities(line)
    assert not [q for q in got if q["value"] == value], got


# --------------------------------------------------------------------------- through the sandbox

gpd = pytest.importorskip("geopandas")


def _run(tmp_path, monkeypatch, code):
    """The assembled prologue + code + epilogue, as the sandbox runs it, and the agent-side read
    of its report with the run's stdout."""
    import contextlib
    import io

    from agent_runtime.code_execution import _read_checks

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "script.py"
    path.write_text(sv.prologue_source({}) + code + sv.epilogue_source(), encoding="utf-8")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(path.read_text(), str(path), "exec"), {"__name__": "__main__"})
    return _read_checks(tmp_path, out.getvalue()), out.getvalue()


COUNTY = '''
import geopandas as gpd
from shapely.geometry import box
g = gpd.GeoDataFrame({"name": ["c"]}, geometry=[box(-88.46, 39.88, -87.93, 40.40)], crs=4326)
'''


def test_a_printed_area_inherits_the_crs_it_was_measured_in(tmp_path, monkeypatch):
    """No IGUIDE_OUTPUTS anywhere: the printed figure is typed, and the area call it came from
    says where it was measured."""
    report, stdout = _run(tmp_path, monkeypatch, COUNTY + '''
a = g.to_crs(5070).area.sum()
print(f"Area: {a / 1e6:,.2f} km²")
''')
    assert report["verdict"] == "pass", report["findings"]
    printed = [o for o in report["outputs"] if o.get("source") == "printed"]
    assert printed and printed[0]["unit"] == "km²"
    assert printed[0]["measured_in_crs"] == "EPSG:5070"


def test_a_measurement_reaches_the_gate_even_when_the_print_has_no_unit(tmp_path, monkeypatch):
    """A model that prints a bare number (or a tuple) still had its measurement recorded."""
    report, _ = _run(tmp_path, monkeypatch, COUNTY + '''
a = g.to_crs(5070).area.sum()
print(a)
''')
    measured = [o for o in report["outputs"] if o.get("source") == "measured"]
    assert measured and measured[0]["measured_in_crs"] == "EPSG:5070"
    assert measured[0]["unit"] == "m**2"


def test_a_feet_crs_printed_as_metres_fails(tmp_path, monkeypatch):
    """EPSG:3435 is in US survey feet. Dividing ft² by 1e6 and calling it km² is a factor of
    10.76 off, and nothing in a frame shows it: the unit has to be compared with the CRS."""
    report, _ = _run(tmp_path, monkeypatch, COUNTY + '''
a = g.to_crs(3435).area.sum()
print(f"Area: {a / 1e6:,.2f} km²")
''')
    assert report["verdict"] == "fail"
    bad = [f for f in report["findings"] if f["check"] == "printed_unit"]
    assert bad and "foot" in bad[0]["message"].lower()


def test_the_same_feet_crs_converted_correctly_passes(tmp_path, monkeypatch):
    report, _ = _run(tmp_path, monkeypatch, COUNTY + '''
a = g.to_crs(3435).area.sum() * 0.3048006096**2
print(f"Area: {a / 1e6:,.2f} km²")
''')
    assert report["verdict"] == "pass", report["findings"]
    assert not [f for f in report["findings"] if f["check"] == "printed_unit"]


def test_a_printed_figure_measured_in_degrees_is_named(tmp_path, monkeypatch):
    report, _ = _run(tmp_path, monkeypatch, COUNTY + '''
import warnings
from shapely.geometry import Point
warnings.filterwarnings("ignore")
d = g.distance(Point(-87.0, 41.0)).min()
print(f"Distance to the gauge: {d:.4f} m")
''')
    assert report["verdict"] == "fail"
    named = [f for f in report["findings"] if f["check"] == "measured_in"]
    assert named and named[0]["status"] == "fail" and " m" in named[0]["target"]
    assert "EPSG:4326" in named[0]["message"]


def test_the_model_sees_a_bounded_record(tmp_path, monkeypatch):
    """`verification` is inside the tool result the model reads: the recorded values stay in
    checks.json, and at most a few typed outputs travel back."""
    report, _ = _run(tmp_path, monkeypatch, COUNTY + '''
p = g.to_crs(5070)
for i in range(40):
    print(f"step {i}: {p.buffer(100 * (i + 1)).area.sum() / 1e6:.3f} km²")
''')
    assert len(report["outputs"]) <= measured_outputs.MAX_OUTPUTS
    assert all("values" not in o for o in report.get("metric_ops") or [])


# --------------------------------------------------------------------------- into the answer

def test_an_answer_figure_resolves_to_a_typed_measured_fact(tmp_path, monkeypatch):
    report, stdout = _run(tmp_path, monkeypatch, COUNTY + '''
a = g.to_crs(5070).area.sum()
print(f"Area: {a / 1e6:,.2f} km²")
''')
    log = turn_log.new_log()
    content = json.dumps({"ok": True, "stdout": stdout, "verification": report})
    log.record_result(peer="code", run=None, name="execute_code", args={"code": "x"},
                      call_id="c1", content=content)
    printed = next(o for o in report["outputs"] if o.get("source") == "printed")
    fs = turn_facts.build(log=log, results=[{"name": "execute_code", "content": content}])
    answer = f"The county covers {printed['value']:,.1f} km²."
    hit = turn_facts.resolve(answer, fs)[0]
    assert hit.fact is not None and hit.fact.typed
    assert hit.fact.measured_in_crs == "EPSG:5070"


def test_a_count_check_is_not_run_on_a_printed_count():
    """A printed "108785 people" is a sum, not a row count: the declared-count checks (fractional,
    larger than every frame) would fail correct runs, so they stay with declarations."""
    report = {"verdict": "pass", "counts": {"pass": 1}, "findings": [],
              "frame_sizes": {"demand": 40}, "metric_ops": []}
    findings, outputs = measured_outputs.evaluate(report, "Total population: 108785 people\n"
                                                          "Mean: 4.5 schools\n")
    assert not [f for f in findings if f["status"] != "pass"]
