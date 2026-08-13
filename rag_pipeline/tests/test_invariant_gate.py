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

import json

import pytest

from agent_runtime.sandbox_verify import (FAIL, PASS, UNKNOWN, check_join_cardinality,
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
    because pandas types it as object — exactly what an unmatched join produces."""
    frame = pd.DataFrame({"a": [None, None], "b": [1, 2]})
    out = check_not_all_nan("joined", frame)
    assert out["status"] == FAIL and "a" in out["columns"]


def test_a_numeric_all_nan_column_is_caught():
    frame = pd.DataFrame({"a": [float("nan")] * 3, "b": [1, 2, 3]})
    assert check_not_all_nan("df", frame)["status"] == FAIL


def test_an_empty_frame_fails():
    assert check_not_all_nan("df", pd.DataFrame({"a": []}))["status"] == FAIL


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

def test_the_epilogue_is_self_contained(tmp_path, monkeypatch):
    """It runs in the sandbox where the method library may be absent and there is no network,
    so it must not depend on importing anything of ours."""
    src = epilogue_source()
    assert "import agent_runtime" not in src
    assert "iguide_methods" not in src

    monkeypatch.chdir(tmp_path)
    ns = {"gdf": _geo("EPSG:4326")}
    exec(compile(src, "<epilogue>", "exec"), ns)
    report = json.loads((tmp_path / "checks.json").read_text())
    assert report["verdict"] == FAIL
    assert _status(report["findings"], "gdf", "projected_crs") == FAIL


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
    ns = {DECLARED_OUTPUTS: {"radius": {"value": 25000, "unit": None}}}
    exec(compile(epilogue_source(), "<epilogue>", "exec"), ns)
    report = json.loads((tmp_path / "checks.json").read_text())
    assert report["verdict"] == FAIL
    assert any(f["check"] == "declared_units" for f in report["findings"])
