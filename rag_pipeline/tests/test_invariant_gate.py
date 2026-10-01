"""The invariant gate: deterministic checks on the LIVE objects a run produced.

The class of error this exists for produces no exception and a plausible number. A 25 km
buffer requested on an EPSG:4326 frame buffers by 25000 *degrees*; the run exits 0, prints a
figure, and every static check passes. Only the frame itself knows its CRS, which is why the
checks run inside the sandbox rather than over source text.

Two properties are asserted throughout:

* **no false positives** — a correct run must come back clean, or the gate gets ignored;
* **"cannot determine" is never silently a pass** — an unknown CRS and a correct CRS must not
  produce the same verdict, because the whole point is that a confident wrong number is worse
  than an admitted unknown.
"""

from __future__ import annotations

import ast
import json

import pytest

from agent_runtime.sandbox_verify import (prologue_source, check_declared_units, FAIL, PASS, UNKNOWN, check_join_cardinality,
                                          check_not_all_nan, check_projected_crs,
                                          epilogue_source, run_checks)

gpd = pytest.importorskip("geopandas")
pd = pytest.importorskip("pandas")
from shapely.geometry import Point  # noqa: E402


def _geo(crs="EPSG:4326", n=2):
    return gpd.GeoDataFrame({"v": list(range(n))},
                            geometry=[Point(-87.6 - i / 100, 41.9) for i in range(n)], crs=crs)


def _status(findings, target, check):
    for f in findings:
        if f["target"] == target and f["check"] == check:
            return f["status"]
    return None


# ------------------------------------------------------------------ CRS: the motivating bug

def test_a_geographic_frame_fails_the_crs_check():
    assert check_projected_crs("gdf", _geo("EPSG:4326"))["status"] == FAIL


def test_a_projected_frame_passes():
    assert check_projected_crs("gdf", _geo("EPSG:4326").to_crs(3857))["status"] == PASS


def test_a_utm_frame_passes():
    assert check_projected_crs("gdf", _geo("EPSG:4326").to_crs(32616))["status"] == PASS


def test_a_frame_with_no_crs_is_unknown_not_a_pass():
    """An unset CRS means distances cannot be trusted — but it is not proof they are wrong."""
    frame = _geo("EPSG:4326")
    frame.crs = None
    assert check_projected_crs("gdf", frame)["status"] == UNKNOWN


def test_the_failure_message_says_how_to_fix_it():
    msg = check_projected_crs("gdf", _geo("EPSG:4326"))["message"]
    assert "degrees" in msg and "to_crs" in msg


# ------------------------------------------------------------------ null columns

def test_an_object_dtype_all_null_column_is_caught():
    """The first version used select_dtypes("number"), which excludes an all-None column
    because pandas types it as object — exactly what an unmatched join produces.

    The frame carries ``index_right`` because that is what the docstring above describes — an
    unmatched join. Without join evidence the same shape is an ordinary dataset with an empty
    optional column, and a correct run over one has to be able to reach ``pass``; the dtype
    detection this test is about is identical either way."""
    frame = pd.DataFrame({"a": [None, None], "b": [1, 2], "index_right": [0, 1]})
    out = check_not_all_nan("joined", frame)
    assert out["status"] == FAIL and "a" in out["columns"]


def test_an_all_null_column_in_a_JOIN_result_is_a_hard_failure():
    """With join evidence there is no ambiguity: nothing matched, so any count or ratio computed
    from the result is wrong. This is the case the check exists for, and it must stay a fail."""
    frame = pd.DataFrame({"a": [1, 2], "index_right": [0, 1], "joined": [None, None]})
    out = check_not_all_nan("joined", frame)
    assert out["status"] == FAIL
    assert "nothing matched" in out["message"]


def test_merge_suffixes_also_count_as_join_evidence():
    """``pd.merge`` does not add ``index_right``; it adds ``_left``/``_right`` suffixes on
    collisions. Keying only on the sjoin marker would miss the commonest unmatched merge."""
    frame = pd.DataFrame({"pop_left": [1, 2], "count_right": [None, None]})
    assert check_not_all_nan("merged", frame)["status"] == FAIL


def test_a_numeric_all_nan_column_is_caught():
    frame = pd.DataFrame({"a": [float("nan")] * 3, "b": [1, 2, 3], "index_left": [0, 1, 2]})
    assert check_not_all_nan("df", frame)["status"] == FAIL


def test_a_sparse_optional_column_is_recorded_without_downgrading_the_run():
    """A dataset with an empty optional column (``apt_number``, ``middle_name``) is ordinary, and
    a correct run over one must reach ``pass``. The observation is still recorded in the finding,
    so anyone reading checks.json sees it — it just does not claim the numbers are unverified.

    Deliberately the same rule as the call site in ``_check_one_arg``. Two different answers to
    the same question in one module is how the confusion this check keeps causing starts."""
    out = check_not_all_nan("tracts", pd.DataFrame({"tract": [1, 2], "apt_no": [None, None]}))
    assert out["status"] == PASS
    assert out["columns"] == ["apt_no"], "the observation must still be recorded"


def test_every_column_null_is_a_failure_even_without_join_evidence():
    """Nothing came through at all. That needs no join marker to be unambiguous."""
    out = check_not_all_nan("df", pd.DataFrame({"a": [None, None], "b": [None, None]}))
    assert out["status"] == FAIL and "EVERY non-geometry column" in out["message"]


def test_an_empty_frame_is_reported_but_not_called_wrong():
    """A filter that matches nothing and a spatial query with no hits are both CORRECT outcomes
    with zero rows. Calling them errors blocks a right answer — and it is also what a broken
    filter produces, so it is still said out loud."""
    out = check_not_all_nan("df", pd.DataFrame({"a": []}))
    assert out["status"] == UNKNOWN
    assert "matched nothing" in out["message"] and "failed filter" in out["message"]


def test_partial_nulls_are_not_a_failure():
    """Real data has gaps; only an ENTIRELY null column is the failed-join signature."""
    frame = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})
    assert check_not_all_nan("df", frame)["status"] == PASS


def test_a_null_geometry_column_does_not_trip_the_null_check():
    frame = _geo()
    frame["geometry"] = None
    assert check_not_all_nan("gdf", frame)["status"] == PASS


# ------------------------------------------------------------------ join cardinality

def test_a_join_result_reports_its_cardinality():
    frame = pd.DataFrame({"a": [1, 2, 2], "index_right": [10, 11, None]})
    out = check_join_cardinality("joined", frame)
    assert out is not None and out["rows"] == 3 and out["unmatched"] == 1


def test_a_non_join_frame_reports_nothing_rather_than_unknown():
    """An 'unknown' per ordinary frame buries the findings that matter: the first real run
    emitted four cannot_determine lines against two genuine failures."""
    assert check_join_cardinality("df", pd.DataFrame({"a": [1]})) is None


# ------------------------------------------------------------------ the driver

def test_a_correct_run_produces_no_failures():
    """No false positives, or the gate gets ignored."""
    report = run_checks({"gdf": _geo().to_crs(3857), "df": pd.DataFrame({"a": [1, 2]})})
    assert report["verdict"] == PASS
    assert report["counts"][FAIL] == 0


def test_a_wrong_run_fails_the_whole_report():
    report = run_checks({"gdf": _geo("EPSG:4326")})
    assert report["verdict"] == FAIL


def test_one_unknown_downgrades_the_whole_verdict():
    """The asymmetry that matters. A frame with no CRS still passes the null check, and the
    first version scored `PASS if anything passed` — so a result nobody could verify came back
    wearing a verified badge. Precedence is fail > cannot_determine > pass."""
    frame = _geo()
    frame.crs = None
    report = run_checks({"gdf": frame})
    assert report["counts"][PASS] >= 1, "something did pass, which is the point of the case"
    assert report["verdict"] == UNKNOWN


def test_a_failure_outranks_an_unknown():
    frame_unknown = _geo()
    frame_unknown.crs = None
    report = run_checks({"unknown": frame_unknown, "bad": _geo("EPSG:4326")})
    assert report["verdict"] == FAIL


def test_non_frame_bindings_are_ignored():
    report = run_checks({"x": 5, "s": "text", "fn": len, "gdf": _geo().to_crs(3857)})
    assert report["inspected"] == ["gdf"]


def test_a_check_that_raises_cannot_break_the_report():
    class Hostile:
        columns = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
        index = ()

        def select_dtypes(self, **kw):
            raise RuntimeError("boom")

    run_checks({"bad": Hostile()})   # must not raise


def test_underscore_names_are_skipped():
    """The epilogue's own locals must not be inspected as if they were results."""
    report = run_checks({"_internal": _geo("EPSG:4326")})
    assert report["inspected"] == []


# ------------------------------------------------------------------ the injected epilogue

def test_the_injected_gate_is_self_contained(tmp_path, monkeypatch):
    """It runs in the sandbox where the method library may be absent and there is no network,
    so it must not depend on importing anything of ours.

    The gate body moved into the PROLOGUE so that its atexit registration exists before the
    user's code runs — a script ending in ``sys.exit(main())`` never even defines an appended
    epilogue. The assembled pair is what the sandbox executes, so it is what gets tested.
    """
    src = prologue_source(None) + epilogue_source()
    # Checked against the parsed IMPORT statements, not against the text. The gate inlines its
    # own helpers, docstrings and all, and one of those docstrings mentions
    # `from iguide_methods...` as prose -- a substring assertion fails on the explanation while
    # saying nothing about what the code actually imports.
    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"agent_runtime", "iguide_methods", "geopandas", "pandas"}, imported

    # The run's own code is what fails it now: a degree buffer, seen as the call happens. A bare
    # 4326 frame that nothing measures is an input (see the operation-tracker tests below).
    import warnings

    monkeypatch.setattr(warnings, "warn", warnings.warn)
    monkeypatch.chdir(tmp_path)
    ns = {"gdf": _geo("EPSG:4326"), "__name__": "__main__"}
    gate = prologue_source(None)
    exec(compile(gate + "bad = gdf.buffer(0.01)\n" + epilogue_source(), "<script>", "exec"), ns)
    report = json.loads((tmp_path / "checks.json").read_text())
    assert report["verdict"] == FAIL
    assert any(f.get("op") == "buffer" and f["status"] == FAIL for f in report["findings"])


def test_the_epilogue_writes_checks_even_with_nothing_to_check(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exec(compile(epilogue_source(), "<epilogue>", "exec"), {"x": 1})
    assert (tmp_path / "checks.json").is_file()


def test_the_epilogue_cannot_raise(tmp_path, monkeypatch):
    """A verification step that breaks a working analysis is worse than none."""
    monkeypatch.chdir(tmp_path)

    class Hostile:
        def __getattr__(self, name):
            raise RuntimeError("boom")

    exec(compile(epilogue_source(), "<epilogue>", "exec"), {"bad": Hostile()})


# ------------------------------------------------------------------ wiring

def test_the_gate_is_on_by_default(monkeypatch):
    """The error it catches produces a plausible number and no exception, so a gate that is
    off by default protects nobody."""
    from agent_runtime.code_execution import invariant_gate_enabled

    monkeypatch.delenv("AGENT_INVARIANT_GATE", raising=False)
    assert invariant_gate_enabled() is True
    monkeypatch.setenv("AGENT_INVARIANT_GATE", "0")
    assert invariant_gate_enabled() is False


def test_verification_reaches_the_tool_result():
    """Surfaced INSIDE the tool result so the model can reproject and re-run in-loop, rather
    than caveating a wrong number after the fact."""
    from agent_runtime.code_execution import ExecResult

    payload = ExecResult(0, verification={"verdict": FAIL, "findings": []}).to_dict()
    assert payload["verification"]["verdict"] == FAIL


def test_an_absent_report_is_empty_not_a_synthetic_pass(tmp_path):
    from agent_runtime.code_execution import _read_checks

    assert _read_checks(tmp_path) == {}


def test_the_reader_drops_passes_but_keeps_failures(tmp_path):
    """The model needs the verdict and what failed, not every pass."""
    from agent_runtime.code_execution import _read_checks

    (tmp_path / "checks.json").write_text(json.dumps({
        "verdict": FAIL, "counts": {PASS: 3, FAIL: 1, UNKNOWN: 0}, "inspected": ["g"],
        "findings": [{"status": PASS, "check": "all_nan", "target": "g", "message": "ok"},
                     {"status": FAIL, "check": "projected_crs", "target": "g", "message": "bad"}]}))
    out = _read_checks(tmp_path)
    assert out["verdict"] == FAIL
    assert [f["status"] for f in out["findings"]] == [FAIL]


def test_a_corrupt_report_does_not_raise(tmp_path):
    from agent_runtime.code_execution import _read_checks

    (tmp_path / "checks.json").write_text("{not json")
    assert _read_checks(tmp_path) == {}


# ------------------------------------------------------------------ the gate BLOCKS the answer

def _run_ctx(verdict, findings=(), stdout="AREA: 0.196\n"):
    """An execute_code tool result as it actually arrives: a JSON STRING inside a tool call."""
    return {"code_result": {"tool_calls": [{"name": "execute_code", "result": json.dumps({
        "ok": True, "exit_code": 0, "stdout": stdout,
        "verification": {"verdict": verdict, "findings": list(findings)}})}]}}


_CRS_FINDING = {"status": FAIL, "check": "projected_crs", "target": "buffered",
                "message": "EPSG:4326 is GEOGRAPHIC: reproject before measuring."}


def test_gate_failures_are_found_through_a_json_string_tool_result():
    """Tool results arrive as JSON strings inside ToolMessages, not as dicts — indexing a
    fixed path instead of walking would silently find nothing."""
    from agent_runtime.supervisor.graph import _gate_failures

    assert len(_gate_failures(_run_ctx(FAIL, [_CRS_FINDING]))) == 1
    assert _gate_failures(_run_ctx(PASS)) == []


def test_a_number_in_an_UNVERIFIED_record_is_no_longer_treated_as_grounded():
    """The hole this closes. Rule (2) drops a disputed number when it appears in the execution
    record — but a wrong number appears there too. `AREA: 0.196` is right in stdout, so the
    gate would say "degrees squared" and the reconciliation would answer "it is in the record."
    """
    from agent_runtime.supervisor.graph import _reconcile_audit_with_artifacts

    audit = {"hallucination_detected": True, "severity": "high",
             "issues": [{"claim": "the area is 0.196", "reason": "units unclear"}]}
    out = _reconcile_audit_with_artifacts(audit, artifacts=[],
                                          execution_context=_run_ctx(FAIL, [_CRS_FINDING]))
    assert out["hallucination_detected"] is True
    assert out["severity"] == "high"
    assert out.get("invariant_gate") == "fail"


def test_a_number_in_a_VERIFIED_record_is_still_cleared():
    """The M6a false-positive suppression must survive: 5/5 -> 0/5 was the whole point."""
    from agent_runtime.supervisor.graph import _reconcile_audit_with_artifacts

    audit = {"hallucination_detected": True, "severity": "high",
             "issues": [{"claim": "there were 21500 incidents", "reason": "not in evidence"}]}
    out = _reconcile_audit_with_artifacts(
        audit, artifacts=[], execution_context=_run_ctx(PASS, stdout="incidents: 21500\n"))
    assert out["hallucination_detected"] is False


def test_the_gate_flags_even_when_the_auditor_found_nothing():
    """Deterministic beats prose: the gate knows a distance was computed in degrees, and an
    auditor reading the answer text has no way to."""
    from agent_runtime.supervisor.graph import _reconcile_audit_with_artifacts

    silent = {"hallucination_detected": False, "severity": "none", "issues": []}
    out = _reconcile_audit_with_artifacts(silent, artifacts=[],
                                          execution_context=_run_ctx(FAIL, [_CRS_FINDING]))
    assert out["hallucination_detected"] is True
    assert "not verified" in out["summary"]


def test_the_gate_issue_carries_the_remedy_not_just_a_complaint():
    from agent_runtime.supervisor.graph import _reconcile_audit_with_artifacts

    out = _reconcile_audit_with_artifacts({"hallucination_detected": False, "severity": "none",
                                           "issues": []}, artifacts=[],
                                          execution_context=_run_ctx(FAIL, [_CRS_FINDING]))
    assert "reproject" in out["issues"][0]["reason"].lower()


def test_a_clean_run_with_no_audit_is_left_completely_alone():
    from agent_runtime.supervisor.graph import _reconcile_audit_with_artifacts

    silent = {"hallucination_detected": False, "severity": "none", "issues": []}
    assert _reconcile_audit_with_artifacts(silent, artifacts=[],
                                           execution_context=_run_ctx(PASS)) is silent


# ------------------------------------------------------------------ declared units + bounds

def _declared(outputs):
    from agent_runtime.sandbox_verify import DECLARED_OUTPUTS
    return run_checks({DECLARED_OUTPUTS: outputs})


def test_a_declared_output_with_a_unit_passes():
    rep = _declared({"radius": {"value": 25000, "unit": "metres", "min": 0, "max": 1e6}})
    assert rep["verdict"] == PASS


def test_a_null_unit_blocks_verification():
    """The plan's rule: a null unit blocks 'verified', because the number most likely to be
    wrong is the one whose unit nobody wrote down. 25000 is right in metres, wrong in feet."""
    assert _declared({"radius": {"value": 25000, "unit": None}})["verdict"] == FAIL


def test_a_bare_number_with_no_unit_at_all_fails():
    assert _declared({"radius": 25000})["verdict"] == FAIL


def test_a_value_outside_its_declared_bounds_fails():
    assert _declared({"pct": {"value": 140, "unit": "percent", "min": 0, "max": 100}})["verdict"] == FAIL


def test_bounds_are_only_checked_when_declared():
    """Inventing a plausible range would generate false positives."""
    assert _declared({"x": {"value": 1e12, "unit": "metres"}})["verdict"] == PASS


def test_a_nan_output_fails():
    assert _declared({"mean": {"value": float("nan"), "unit": "metres"}})["verdict"] == FAIL


def test_an_unrecognised_unit_is_unknown_not_a_failure():
    assert _declared({"x": {"value": 1, "unit": "furlongs"}})["verdict"] == UNKNOWN


def test_no_declared_outputs_adds_no_findings():
    """Most runs will not declare any; that must not itself downgrade the verdict."""
    rep = run_checks({"gdf": _geo().to_crs(3857)})
    assert rep["verdict"] == PASS


def test_declared_outputs_survive_the_epilogue(tmp_path, monkeypatch):
    from agent_runtime.sandbox_verify import DECLARED_OUTPUTS

    monkeypatch.chdir(tmp_path)
    ns = {DECLARED_OUTPUTS: {"radius": {"value": 25000, "unit": None}}, "__name__": "__main__"}
    exec(compile(prologue_source(None) + epilogue_source(), "<script>", "exec"), ns)
    report = json.loads((tmp_path / "checks.json").read_text())
    assert report["verdict"] == FAIL
    assert any(f["check"] == "declared_units" for f in report["findings"])


# ------------------------------------------------------------------ the gate cannot go quiet

def test_the_epilogue_alone_records_that_nothing_was_checked(tmp_path, monkeypatch):
    """The epilogue calls a gate the PROLOGUE defines. If only the epilogue is ever emitted,
    the call raises NameError — and a bare ``except Exception: pass`` around it wrote no report
    at all, which is indistinguishable from a gate that was deliberately disabled.

    An absent report must never be the way "we checked nothing" is communicated.
    """
    monkeypatch.chdir(tmp_path)
    ns: dict = {"gdf": _geo("EPSG:4326"), "__name__": "__main__"}
    exec(compile(epilogue_source(), "<epilogue>", "exec"), ns)
    report = json.loads((tmp_path / "checks.json").read_text())
    assert report["verdict"] == UNKNOWN
    assert any("prologue was not installed" in f["message"] for f in report["findings"])


@pytest.mark.parametrize("tail", [
    "sys.exit(main())",          # the standard script skeleton
    "main()\nexit()",
    "main()\nraise RuntimeError('boom')",
    "main()",
])
def test_every_exit_path_writes_a_report_with_no_internal_error(tmp_path, tail):
    """Three separate NameErrors have now been swallowed by the gate's own ``except`` and
    surfaced as a bare ``cannot_determine``: ``math as _math``, ``ModuleType``, and a
    ``_iguide_gate_body`` that returned two of the four names the gate used — which silently
    stopped ``environment.json`` and ``declared_outputs.json`` being written at all, leaving a
    replay with nothing to compare.

    Asserting only that checks.json EXISTS does not catch any of them; the absence of an
    ``error`` key is what does. This runs a real subprocess so the exit paths are real.
    """
    import subprocess
    import sys as _sys

    body = ("import sys\n"
            "import geopandas as gpd\n"
            "from shapely.geometry import Point\n"
            "def main():\n"
            "    g = gpd.GeoDataFrame({'a': [1]}, geometry=[Point(0, 0)], crs='EPSG:4326')\n"
            "    g['b'] = g.buffer(0.2)\n"
            "    return 0\n")
    (tmp_path / "run.py").write_text(
        prologue_source(None) + body + tail + "\n" + epilogue_source(), encoding="utf-8")
    subprocess.run([_sys.executable, "run.py"], cwd=tmp_path, capture_output=True, text=True)

    report = json.loads((tmp_path / "checks.json").read_text())
    assert "error" not in report, f"the gate errored internally: {report.get('error')}"
    assert (tmp_path / "environment.json").is_file(), "no env capture = no reproducibility"
    # Work done inside main() leaves module scope empty, and the coverage finding still SAYS SO.
    # But the degree buffer inside it is no longer invisible: the operation tracker sees the
    # call wherever it runs, so the verdict is the FAIL the code deserves, not an unknown.
    assert any(f["check"] == "coverage" for f in report["findings"])
    assert report["verdict"] == FAIL
    assert any(f.get("op") == "buffer" and f["status"] == FAIL for f in report["findings"])


def test_a_run_with_no_geospatial_library_can_still_pass(tmp_path, monkeypatch):
    """The coverage guard keyed on ``sys.modules``, which is process-global: geopandas being
    loaded by some unrelated module made every pure-arithmetic run unverifiable. Flooding the
    channel with unknowns is how a real one stops being read."""
    from agent_runtime.sandbox_verify import DECLARED_OUTPUTS

    monkeypatch.chdir(tmp_path)
    ns = {DECLARED_OUTPUTS: {"n": {"value": 3, "unit": "count"}}, "__name__": "__main__"}
    exec(compile(prologue_source(None) + epilogue_source(), "<script>", "exec"), ns)
    assert json.loads((tmp_path / "checks.json").read_text())["verdict"] == PASS


def test_a_geospatial_run_whose_frames_are_unreachable_cannot_pass(tmp_path, monkeypatch):
    """The same guard, in the case it exists for: a bound geo module and no frame to inspect."""
    import geopandas

    monkeypatch.chdir(tmp_path)
    ns = {"gpd": geopandas, "main": lambda: None, "__name__": "__main__"}
    exec(compile(prologue_source(None) + epilogue_source(), "<script>", "exec"), ns)
    report = json.loads((tmp_path / "checks.json").read_text())
    assert report["verdict"] == UNKNOWN
    assert any(f["check"] == "coverage" for f in report["findings"])


# ------------------------------------------------- the frame the number came from (A4, A7)

def test_a_geometry_column_that_is_not_named_geometry_is_still_found():
    """``_has_geometry`` tested for a column literally named ``geometry``. GeoPandas lets the
    active geometry be any column — ``geom``, ``the_geom``, ``centroid`` are all common, and
    PostGIS exports default to ``geom`` — so a frame carrying real geometry in a differently
    named column was never CRS-checked at all, and reported clean."""
    from agent_runtime.sandbox_verify import _geometry_column, check_projected_crs

    gdf = _geo("EPSG:4326").rename_geometry("geom")
    assert _geometry_column(gdf) == "geom"
    assert check_projected_crs("gdf", gdf)["status"] == FAIL


def test_geometry_bearing_frames_are_inspected_before_plain_ones():
    """The frame budget is 12 and globals were walked in DEFINITION order, so in a multi-step
    run the late output frames — the ones a number is quoted from — were exactly the ones
    dropped, and the report said ``pass`` with no sign of truncation."""
    ns = {f"df{i}": _pd().DataFrame({"a": [1, 2]}) for i in range(14)}
    ns["result_gdf"] = _geo("EPSG:4326")          # defined LAST, would have been cut
    report = run_checks(ns)
    assert report["verdict"] == FAIL
    assert "result_gdf" in report["inspected"]


def test_a_truncated_inspection_says_so():
    """A silent cap is the failure mode this whole module exists to prevent."""
    ns = {f"df{i}": _pd().DataFrame({"a": [1, 2]}) for i in range(20)}
    report = run_checks(ns)
    coverage = [f for f in report["findings"] if f["check"] == "coverage"]
    assert coverage and "NOT checked" in coverage[0]["message"]
    assert report["verdict"] == UNKNOWN


def _pd():
    import pandas
    return pandas


# ------------------------------------------------- projected is not the same as metres (A6)

@pytest.mark.parametrize("epsg,declared,expected", [
    ("EPSG:32616", "metres", PASS),     # UTM 16N, metres
    ("EPSG:3857", "metres", PASS),      # web mercator, metres
    # Illinois East state plane in US SURVEY FEET. `is_projected` is True, so the CRS check
    # passed and a buffer declared in metres was silently 3.28x too large. "Projected" answers
    # a different question than "in the unit you declared".
    ("EPSG:3435", "metres", FAIL),
    ("EPSG:3435", "feet", PASS),
])
def test_a_projected_crs_in_the_wrong_unit_fails_a_declared_unit(epsg, declared, expected):
    from agent_runtime.sandbox_verify import _crs_unit, _unit_matches

    gdf = _geo(epsg)
    actual = _crs_unit(gdf.crs)
    matched = _unit_matches(declared, actual)
    assert matched is not None, f"{epsg} axis unit was unreadable ({actual!r})"
    assert (PASS if matched else FAIL) == expected


# ------------------------------------------------- contracts on containers (A9)

def test_a_contract_descends_into_a_list_of_frames():
    """``e2sfca(catchments)`` takes a LIST of frames. The guard bound the argument and tested it
    with ``_looks_like_frame``, which a list is not, so the invariant declared on that parameter
    was skipped — silently, for every unit whose interface is a collection."""
    from agent_runtime.sandbox_verify import check_contract_arg

    inv = {"check": "projected_crs", "target": "catchments"}
    found = check_contract_arg("e2sfca", inv, [_geo("EPSG:4326"), _geo("EPSG:4326")])
    assert found and found["status"] == FAIL
    assert "catchments[0]" in found["target"], found["target"]


def test_a_contract_descends_into_a_dict_of_frames():
    from agent_runtime.sandbox_verify import check_contract_arg

    inv = {"check": "projected_crs", "target": "layers"}
    found = check_contract_arg("overlay", inv, {"tracts": _geo("EPSG:4326")})
    assert found and found["status"] == FAIL
    assert "layers['tracts']" in found["target"] or "layers[tracts]" in found["target"]


def test_a_container_of_correct_frames_passes():
    """No false positives, or the guard gets switched off."""
    from agent_runtime.sandbox_verify import check_contract_arg

    inv = {"check": "projected_crs", "target": "catchments"}
    assert check_contract_arg("e2sfca", inv, [_geo("EPSG:32616")]) is None


def test_no_name_in_the_inlined_gate_is_unbound():
    """The structural guard against the bug this module keeps producing.

    ``_inlined_helpers()`` copies selected functions into the sandbox script by source text. Any
    module-level name one of them references — a sibling helper, a constant, an import — is NOT
    carried along, so it raises ``NameError`` inside the gate's own ``except``, and the run is
    reported ``cannot_determine`` with a plausible-looking message. It has happened six times:
    ``math as _math``, ``ModuleType``, ``_GEO_MODULES``, ``DECLARED_OUTPUTS``,
    ``capture_environment``, and ``_looks_like_join_result``.

    Every previous fix was whack-a-mole, and each one was found by a run that happened to
    exercise that path — the sparse-column case above was found in a live container, not by the
    suite. Binding analysis over the generated source catches all of them at once, including the
    next one.
    """
    import builtins

    body = next(n for n in ast.parse(prologue_source(None)).body
                if isinstance(n, ast.FunctionDef) and n.name == "_iguide_gate_body")
    bound = set(dir(builtins)) | {"__name__", "__file__"}
    for node in ast.walk(body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
    loaded = {n.id for n in ast.walk(body)
              if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    assert not (loaded - bound), (
        f"these names are referenced by the inlined gate but never bound in it, so they will "
        f"raise NameError inside the sandbox and be reported as cannot_determine: "
        f"{sorted(loaded - bound)}")


@pytest.mark.parametrize("body,expect_clean", [
    # A correct run over a real dataset whose OPTIONAL column is empty. Must not be flagged.
    ("""
import geopandas as gpd
from shapely.geometry import Point
tracts = gpd.GeoDataFrame({'tract': [1, 2], 'apt_number': [None, None]},
                          geometry=[Point(447000, 4636000), Point(448000, 4637000)],
                          crs='EPSG:32616')
tracts['area_m2'] = tracts.buffer(500).area
""", True),
    # An unmatched spatial join. Must be flagged.
    ("""
import geopandas as gpd
from shapely.geometry import Point
left = gpd.GeoDataFrame({'a': [1, 2]}, geometry=[Point(0, 0), Point(1, 1)], crs='EPSG:32616')
right = gpd.GeoDataFrame({'val': [9]}, geometry=[Point(900, 900)], crs='EPSG:32616')
joined = gpd.sjoin(left, right, how='left', predicate='intersects')
""", False),
])
def test_the_null_check_discriminates_in_a_real_run(tmp_path, body, expect_clean):
    """Runs the assembled script in a subprocess, so every check executes against live frames.

    This is the shape that caught ``_looks_like_join_result``: the parametrized exit-path test
    above exercises the coverage path only, so a NameError in the null check survived it."""
    import subprocess
    import sys as _sys

    (tmp_path / "run.py").write_text(prologue_source(None) + body + epilogue_source(),
                                     encoding="utf-8")
    subprocess.run([_sys.executable, "run.py"], cwd=tmp_path, capture_output=True, text=True)
    report = json.loads((tmp_path / "checks.json").read_text())

    errored = [f for f in report["findings"] if "check errored" in f.get("message", "")]
    assert not errored, f"a check raised inside its own guard: {errored}"

    if expect_clean:
        assert report["verdict"] == PASS, [f for f in report["findings"] if f["status"] != PASS]
    else:
        assert report["verdict"] == FAIL
        assert any(f["check"] == "all_nan" and f["status"] == FAIL for f in report["findings"])


# ------------------------------------------- reproject-then-measure is the CORRECT workflow

def test_an_input_frame_that_was_reprojected_before_measuring_is_not_a_failure():
    """Found by a live prototype run, and it is the flooding case.

    The agent reprojected three Chicago points to EPSG:32616, buffered by 25 km, and produced
    areas accurate to 0.16% of the analytic value — a completely correct run. The answer was
    stamped "⛔ A deterministic invariant check FAILED, numeric results are not verified",
    because the untouched 4326 input frame was still bound at module scope.

    Data arrives in 4326 and you reproject it, so an input frame in a geographic CRS is present
    in almost every correct geospatial script. Failing on it fails the standard workflow, and a
    ⛔ on a right answer teaches the reader to ignore ⛔.
    """
    ns = {"gdf_wgs84": _geo("EPSG:4326"),
          "gdf_utm": _geo("EPSG:32616").assign(area_km2=[1960.34, 1960.34])}
    report = run_checks(ns)
    assert report["verdict"] == PASS, [f for f in report["findings"] if f["status"] != PASS]
    assert "reprojected before measuring" in _find(report, "gdf_wgs84", "projected_crs")["message"]


def test_a_bare_geographic_frame_still_fails():
    """The relaxation is driven by POSITIVE evidence of reprojection, not by the absence of a
    measurement column. `gdf.buffer(25000)` on a 4326 frame produces a wrong GEOMETRY and no
    numeric column at all, so keying on a measurement column would miss the motivating case."""
    assert run_checks({"gdf": _geo("EPSG:4326")})["verdict"] == FAIL


def test_a_measurement_computed_in_degrees_fails_even_beside_a_projected_frame():
    """A number computed in a geographic CRS is wrong regardless of what else the run got right.
    Only a frame carrying no measurement of its own can be an untouched input."""
    ns = {"bad": _geo("EPSG:4326").assign(area_km2=[0.196, 0.196]),
          "good": _geo("EPSG:32616").assign(area_km2=[1960.34, 1960.34])}
    report = run_checks(ns)
    assert report["verdict"] == FAIL
    assert _find(report, "bad", "projected_crs")["metric_column"] == "area_km2"


# ------------------------------------------- the OPERATION decides, not the frame inventory

@pytest.fixture()
def tracked(monkeypatch):
    """The prologue's operation tracker, live for one test; warnings.warn restored after."""
    import warnings

    from agent_runtime.sandbox_verify import install_operation_tracker

    monkeypatch.setattr(warnings, "warn", warnings.warn)
    ns = {}
    assert install_operation_tracker(ns)
    return ns


def _run(ns, tmp_path, code):
    """From a real file, as the sandbox runs it, so the FAIL can quote the line."""
    path = tmp_path / "script.py"
    path.write_text(code, encoding="utf-8")
    exec(compile(code, str(path), "exec"), ns)
    return run_checks(ns)


def test_a_reprojected_input_measured_into_a_scalar_passes(tracked, tmp_path):
    """The map UI, 2026-10-01. Champaign reprojected to 26916, buffered by 2 km, 165.04 km^2
    against QGIS's 164.99, and the answer was stamped ⛔ FAILED. The untouched 4326 input was
    still bound and the measurement was a scalar, so no frame-level rescue fired."""
    tracked["boundary"] = _geo("EPSG:4326")
    report = _run(tracked, tmp_path, "projected = boundary.to_crs('EPSG:26916')\n"
                           "buffered = projected.buffer(2000)\n"
                           "area_km2 = float(buffered.area.sum() / 1e6)\n")
    assert report["verdict"] != FAIL, [f for f in report["findings"] if f["status"] == FAIL]
    assert _status(report["findings"], "boundary", "projected_crs") == PASS


def test_measuring_the_original_after_reprojecting_fails(tracked, tmp_path):
    """The agent designer's counterexample to frame lineage: reproject, then measure the WRONG
    variable. Seen at the call, it is a degree measurement wherever its result lands."""
    tracked["boundary"] = _geo("EPSG:4326")
    report = _run(tracked, tmp_path, "projected = boundary.to_crs('EPSG:26916')\n"
                           "area = float(boundary.area.sum())\n")
    assert report["verdict"] == FAIL
    failed = [f for f in report["findings"] if f["status"] == FAIL]
    assert failed and failed[0]["op"] == "area"
    assert "boundary.area.sum()" in failed[0]["message"], "the FAIL names the line"


def test_a_buffer_in_degrees_fails_even_when_warnings_are_silenced(tracked, tmp_path):
    """Agents routinely start with filterwarnings('ignore'), and that must not blind the gate."""
    tracked["gdf"] = _geo("EPSG:4326")
    report = _run(tracked, tmp_path, "import warnings\nwarnings.filterwarnings('ignore')\n"
                           "bad = gdf.buffer(0.01)\n")
    assert report["verdict"] == FAIL
    assert any(f.get("op") == "buffer" and f["status"] == FAIL for f in report["findings"])


def test_a_centroid_or_a_zero_buffer_in_4326_is_not_a_measurement(tracked, tmp_path):
    """A centroid is a location, and buffer(0) is the geometry-repair idiom (geopandas does not
    even warn on it). Neither is a distance."""
    tracked["gdf"] = _geo("EPSG:4326")
    report = _run(tracked, tmp_path, "labels = gdf.centroid\nrepaired = gdf.buffer(0)\n")
    assert report["verdict"] != FAIL, [f for f in report["findings"] if f["status"] == FAIL]


def test_an_unexplained_metric_column_in_4326_is_unknown_not_a_failure(tracked, tmp_path):
    """Suspicious, not proven: no tracked operation produced it, and a FAIL the gate cannot tie
    to a measurement reaches the user as ⛔."""
    tracked["gdf"] = _geo("EPSG:4326").assign(area_km2=[0.196, 0.196])
    report = _run(tracked, tmp_path, "x = 1\n")
    assert _status(report["findings"], "gdf", "projected_crs") == UNKNOWN


def test_without_the_tracker_the_frame_rules_still_apply():
    """run_checks called on its own (no live tracker) keeps the old semantics."""
    assert run_checks({"gdf": _geo("EPSG:4326")})["verdict"] == FAIL


def test_the_ui_script_passes_through_the_assembled_gate(tmp_path, monkeypatch):
    """The prologue, the shape of the script the UI run executed, and the epilogue: what the
    sandbox actually runs."""
    import warnings

    monkeypatch.setattr(warnings, "warn", warnings.warn)
    monkeypatch.chdir(tmp_path)
    _geo("EPSG:4326").to_file(tmp_path / "boundary.geojson", driver="GeoJSON")
    code = ("import geopandas as gpd\n"
            "boundary = gpd.read_file('boundary.geojson')\n"
            "projected = boundary.to_crs('EPSG:26916')\n"
            "buffered = projected.buffer(2000)\n"
            "area_km2 = float(buffered.area.sum() / 1_000_000)\n"
            "IGUIDE_OUTPUTS = {'buffered_area': {'value': area_km2, 'unit': 'km2'}}\n")
    src = prologue_source(None) + code + epilogue_source()
    exec(compile(src, "<script>", "exec"), {"__name__": "__main__"})
    report = json.loads((tmp_path / "checks.json").read_text())
    assert report["verdict"] != FAIL, report["findings"]


def _find(report, target, check):
    return next(f for f in report["findings"]
                if f["target"] == target and f["check"] == check)


# ------------------------------------------- units a model actually writes

@pytest.mark.parametrize("unit", ["km²", "km2", "square kilometres", "sq km",
                                  "m²", "m2", "square metres", "hectares", "acres"])
def test_an_areal_unit_is_recognised(unit):
    """`km²` is how a model writes it, observed live — and it was in neither the known-unit set
    nor the alias table, so a correctly declared unit came back "unrecognised; not checked" and
    downgraded the whole run. A unit the system asked for, received, and then could not read is
    worse than not having asked."""
    findings = check_declared_units({"area": {"value": 2790.47, "unit": unit}})
    statuses = {f["status"] for f in findings if f["check"] == "declared_units"}
    assert UNKNOWN not in statuses, f"{unit!r} was not recognised"


def test_a_genuinely_unknown_unit_is_still_flagged():
    """Widening the vocabulary must not turn it into "accept anything"."""
    findings = check_declared_units({"x": {"value": 1, "unit": "bananas"}})
    assert any(f["status"] == UNKNOWN for f in findings)


# ------------------------------------------------------------- counts are checkable numbers

@pytest.mark.parametrize("unit", ["records", "record", "rows", "count", "counts", "n",
                                  "observations", "features", "incidents", "events", "items"])
def test_the_words_people_actually_use_for_a_count_are_recognised(unit):
    """`count` was in the known set and `records` was not, so a live run declaring
    {'value': 27824, 'unit': 'records'} — the natural word for what it was counting — came back
    "unrecognised unit 'records'; not checked", and that single unknown downgraded a correct
    answer to unverified."""
    from agent_runtime.sandbox_verify import check_declared_units

    findings = check_declared_units({"n_rows": {"value": 27824, "unit": unit}})
    statuses = {f["status"] for f in findings if f["check"] == "declared_units"}
    assert statuses == {"pass"}, (unit, findings)


def test_recognising_a_count_is_not_the_same_as_checking_it():
    """A negative or fractional count is wrong whatever produced it. Recognising the unit and then
    passing is how 'unit count' would score a pass for a value of -3."""
    from agent_runtime.sandbox_verify import check_declared_units

    negative = check_declared_units({"n": {"value": -3, "unit": "records"}})
    assert any(f["check"] == "declared_units" and f["status"] == "fail" for f in negative)

    fractional = check_declared_units({"n": {"value": 12.5, "unit": "count"}})
    assert any(f["check"] == "declared_units" and f["status"] == "fail" for f in fractional)

    whole = check_declared_units({"n": {"value": 27824, "unit": "records"}})
    assert any(f["check"] == "declared_units" and f["status"] == "pass" for f in whole)


def test_a_whole_float_count_is_accepted():
    """`int(len(df))` is the common form, but `df.shape[0] * 1.0` reaches here as 27824.0."""
    from agent_runtime.sandbox_verify import check_declared_units

    findings = check_declared_units({"n": {"value": 27824.0, "unit": "records"}})
    assert any(f["check"] == "declared_units" and f["status"] == "pass" for f in findings)


# ------------------------------------------------------------- which population was counted

def test_a_count_larger_than_every_frame_in_the_run_fails():
    """The case that is impossible on any reading. Motivated by a live run that answered a
    question about 128,886 records with counts from a 49,789-row spatially-joined subset."""
    pd = pytest.importorskip("pandas")

    from agent_runtime.sandbox_verify import check_count_population

    namespace = {"df": pd.DataFrame({"a": range(40000)})}
    findings = check_count_population({"total": {"value": 128886, "unit": "records"}}, namespace)
    assert [f["status"] for f in findings] == ["fail"]
    assert "exceeds every frame" in findings[0]["message"]


def test_a_plausible_count_reports_the_population_it_came_from():
    """This does not guess which frame is 'the' population — that would generate false positives.
    It records the sizes present, which is what lets a reader see 9,993-of-49,789 and ask the
    right question."""
    pd = pytest.importorskip("pandas")

    from agent_runtime.sandbox_verify import check_count_population

    namespace = {"df": pd.DataFrame({"a": range(128886)})}
    findings = check_count_population({"theft": {"value": 27824, "unit": "records"}}, namespace)
    assert findings[0]["status"] == "pass"
    assert "df=128886" in findings[0]["message"]
    assert findings[0]["frames"] == {"df": 128886}


def test_a_non_count_output_is_not_population_checked():
    """A buffer radius in metres has no population, and comparing it to a row count would be
    nonsense that fires on every geospatial run."""
    pd = pytest.importorskip("pandas")

    from agent_runtime.sandbox_verify import check_count_population

    namespace = {"df": pd.DataFrame({"a": range(10)})}
    assert check_count_population({"radius": {"value": 25000, "unit": "metres"}}, namespace) == []


def test_population_checking_needs_a_frame_to_compare_against():
    from agent_runtime.sandbox_verify import check_count_population

    assert check_count_population({"n": {"value": 5, "unit": "count"}}, {}) == []


# ------------------------------------------------------------- nothing to check is not a failure

def test_a_run_that_checked_nothing_says_so_explicitly():
    """An empty report reached the reader as "cannot_determine (counts all zero) but its findings
    were not retained" — which reads as evidence lost in transit. There was never anything to
    retain, and the two need different responses."""
    from agent_runtime.sandbox_verify import run_checks

    report = run_checks({"x": 1, "y": "text"})
    assert report["verdict"] == "cannot_determine"
    assert [f["check"] for f in report["findings"]] == ["not_applicable"]
    assert "nothing in this run was checkable" in report["findings"][0]["message"]
    assert any(report["counts"].values()), "an all-zero count is what made this unreadable"


def test_the_synthesised_message_distinguishes_nothing_checked_from_evidence_lost():
    from agent_runtime.supervisor.graph import _gate_issues_from

    nothing = _gate_issues_from({"verdict": "cannot_determine",
                                 "counts": {"pass": 0, "fail": 0, "cannot_determine": 0},
                                 "findings": []})
    assert "checked nothing in this run" in nothing[0]["message"]
    assert "not retained" not in nothing[0]["message"]

    truncated = _gate_issues_from({"verdict": "cannot_determine",
                                   "counts": {"pass": 2, "fail": 0, "cannot_determine": 1},
                                   "findings": []})
    assert "not retained" in truncated[0]["message"]


# ------------------------------------------- agent side: what the run's output can tell us

def _gated_run(tmp_path, monkeypatch, *, stdout, declared):
    """A run whose sandbox wrote a passing checks.json and the given declared outputs."""
    from agent_runtime import code_execution as ce

    monkeypatch.setenv("AGENT_CODE_EXEC_WORK_ROOT", str(tmp_path))

    class Probe(ce.LocalSubprocessExecutor):
        def _run(self, work, timeout, dependencies=None, deps_cache=None, entrypoint=None):
            (work / "checks.json").write_text(json.dumps(
                {"verdict": PASS, "counts": {PASS: 1, FAIL: 0, UNKNOWN: 0}, "findings": []}))
            (work / "declared_outputs.json").write_text(json.dumps(declared))
            (work / "environment.json").write_text(json.dumps({"python": "3.11"}))
            (work / "buffer.geojson").write_text('{"type": "FeatureCollection", "features": []}')
            return 0, stdout, "", False, None

    return Probe().execute("x = 1")


def test_a_printed_iguide_outputs_is_named_not_quietly_unchecked(tmp_path, monkeypatch):
    """The map UI, 2026-10-01: `print('IGUIDE_OUTPUTS =', {...})`, so declared_outputs.json was
    {} and nothing was checked, with no word about why."""
    result = _gated_run(tmp_path, monkeypatch, declared={},
                        stdout="IGUIDE_OUTPUTS = {'buffered_area': {'value': 165.0, 'unit': 'km2'}}")
    assert result.verification["verdict"] == UNKNOWN
    named = [f for f in result.verification["findings"] if f["check"] == "declared_outputs"]
    assert named and "printed, not assigned" in named[0]["message"]


def test_an_assigned_iguide_outputs_is_left_alone(tmp_path, monkeypatch):
    result = _gated_run(tmp_path, monkeypatch, stdout="IGUIDE_OUTPUTS printed for the log too",
                        declared={"buffered_area": {"value": 165.0, "unit": "km2"}})
    assert result.verification["verdict"] == PASS


def test_the_gates_own_files_are_not_offered_as_downloads(tmp_path, monkeypatch):
    """Provenance, not results: the map UI listed environment.json and declared_outputs.json
    again after every run. They stay in the workspace for the run record."""
    result = _gated_run(tmp_path, monkeypatch, stdout="", declared={})
    names = {a.get("filename") or a.get("name") for a in result.artifacts}
    assert "buffer.geojson" in names, names
    assert not {"environment.json", "declared_outputs.json", "checks.json"} & names, names
