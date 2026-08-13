"""The extracted contract, enforced at run time — not merely displayed to the model.

The gap this closes. ``UnitContract`` has always carried ``crs_expectation`` and
``declared_unit``, and ``infer_units_and_crs`` has always set ``crs_expectation="projected"`` on
a GeoDataFrame parameter whose body performs a metric operation. None of it was ever converted
into an ``InvariantSpec``, and nothing at run time read it: measured before this change, **zero
invariants existed in the whole 203-unit registry**. "Contract-bearing structure" meant
"structure the model is told about".

Two halves:

* ``contract_invariants`` turns the inferred expectation into an enforceable spec that names the
  PARAMETER it constrains;
* a prologue wraps the imported unit so the invariant is checked against the ACTUAL argument at
  call time. That ordering is load-bearing — an epilogue inspecting the namespace afterwards
  cannot know which frame was passed as which parameter. ``compute_accessibility(demand,
  supply)`` needs *demand* projected; a post-hoc scan sees two GeoDataFrames and must guess.

Violations are collected, never raised: a breach means the NUMBER is wrong, and killing the run
would destroy the evidence and any partial output.
"""

from __future__ import annotations

import ast
import json

import pytest

from agent_runtime.sandbox_verify import (CONTRACTS_GLOBAL, FAIL, PASS, UNKNOWN,
                                          VIOLATIONS_GLOBAL, check_contract_arg,
                                          install_contract_guards, prologue_source, run_checks)
from extractors.analysis.signatures import contract_invariants, contract_params

gpd = pytest.importorskip("geopandas")
from shapely.geometry import Point  # noqa: E402


def _geo(crs="EPSG:4326"):
    return gpd.GeoDataFrame({"v": [1]}, geometry=[Point(-87.6, 41.9)], crs=crs)


def _invariants(source):
    node = ast.parse(source).body[0]
    params = contract_params(node, ast.get_docstring(node) or "")
    return contract_invariants(params, node)


# ------------------------------------------------------------------ extraction half

def test_a_metric_operation_produces_a_projected_crs_invariant():
    """The signal existed for 20 corpus params and became an InvariantSpec for none of them."""
    invs = _invariants("def buffer_it(gdf, radius):\n    return gdf.buffer(radius)\n")
    projected = [i for i in invs if i.check == "projected_crs"]
    assert len(projected) == 1
    assert projected[0].target == "gdf"
    assert projected[0].args["unit"] == "metres"
    assert "buffer" in projected[0].evidence


def test_the_invariant_names_the_parameter_not_the_function():
    """So the runtime wrapper can check the actual argument instead of guessing which frame."""
    invs = _invariants("def measure(layer, other):\n    return layer.distance(other)\n")
    assert all(i.target in {"layer", "other"} for i in invs)


def test_no_metric_operation_means_no_crs_invariant():
    invs = _invariants("def rename(gdf):\n    return gdf.rename(columns={'a': 'b'})\n")
    assert [i for i in invs if i.check == "projected_crs"] == []


def test_a_frame_parameter_gets_a_null_check():
    invs = _invariants("def summarise(gdf):\n    return gdf.head()\n")
    assert any(i.check == "reject_all_nan" for i in invs)


def test_a_non_frame_parameter_gets_no_invariants():
    invs = _invariants("def add(a, b):\n    return a + b\n")
    assert invs == []


def test_extractors_attach_the_invariants_to_the_unit(tmp_path):
    """Wired in both extractors, or the contract is enforceable in principle only."""
    from extractors.base import EMIT_LIBRARY, EMIT_OPENSEARCH, ExtractContext
    from extractors.code_extractor import CodeExtractor

    path = tmp_path / "m.py"
    path.write_text("import geopandas as gpd\n\n"
                    "def buffer_it(gdf, radius):\n    return gdf.buffer(radius)\n",
                    encoding="utf-8")
    ctx = ExtractContext(element_id="c1", element_type="code", fields={"title": "M"},
                         targets=[EMIT_OPENSEARCH, EMIT_LIBRARY])
    unit = [a for a in CodeExtractor().extract(str(path), ctx=ctx).assets
            if getattr(a, "unit", None)][0]
    checks = {i["check"] for i in unit.unit["invariants"]}
    assert "projected_crs" in checks


# ------------------------------------------------------------------ the check itself

PROJECTED_INV = {"check": "projected_crs", "target": "gdf", "args": {"unit": "metres"}}


def test_a_geographic_argument_fails_the_declared_invariant():
    found = check_contract_arg("buffer_it", PROJECTED_INV, _geo("EPSG:4326"))
    assert found and found["status"] == FAIL
    assert "buffer_it declares gdf" in found["message"]
    assert "Reproject before calling" in found["message"]


def test_a_projected_argument_satisfies_it():
    assert check_contract_arg("buffer_it", PROJECTED_INV, _geo().to_crs(3857)) is None


def test_a_missing_crs_is_unknown_not_a_pass():
    frame = _geo()
    frame.crs = None
    found = check_contract_arg("buffer_it", PROJECTED_INV, frame)
    assert found and found["status"] == UNKNOWN


def test_a_non_frame_argument_is_not_second_guessed():
    """The contract constrains a frame; a number passed there is a different problem."""
    assert check_contract_arg("buffer_it", PROJECTED_INV, 42) is None


def test_crs_equals_compares_exactly():
    inv = {"check": "crs_equals", "target": "gdf", "args": {"crs": "EPSG:4326"}}
    assert check_contract_arg("f", inv, _geo("EPSG:4326")) is None
    bad = check_contract_arg("f", inv, _geo().to_crs(3857))
    assert bad and bad["status"] == FAIL and bad["expected"] == "EPSG:4326"


def test_reject_all_nan_only_reports_real_failures():
    """"This frame is fine" on every call would bury the findings that matter."""
    pd = pytest.importorskip("pandas")
    inv = {"check": "reject_all_nan", "target": "df"}
    assert check_contract_arg("f", inv, pd.DataFrame({"a": [1, 2]})) is None
    bad = check_contract_arg("f", inv, pd.DataFrame({"a": [None, None]}))
    assert bad and bad["status"] == FAIL


def test_an_unknown_check_name_is_ignored_rather_than_guessed():
    assert check_contract_arg("f", {"check": "not_a_check", "target": "x"}, _geo()) is None


# ------------------------------------------------------------------ the guard wrapper

class _FakeModule:
    """Stands in for a mounted library module."""


def _install(invariants, fn):
    import sys
    import types

    module = types.ModuleType("fake_lib_mod")
    module.target_fn = fn
    sys.modules["fake_lib_mod"] = module
    ns = {}
    installed = install_contract_guards(ns, {"target_fn": {
        "module": "fake_lib_mod", "symbol": "target_fn", "invariants": invariants}})
    return module, ns, installed


def test_the_guard_checks_the_actual_argument_at_call_time():
    module, ns, installed = _install([PROJECTED_INV], lambda gdf, radius=1: "result")
    assert installed == 1
    assert module.target_fn(_geo("EPSG:4326"), 25000) == "result", "the call must still work"
    violations = ns[VIOLATIONS_GLOBAL]
    assert len(violations) == 1 and violations[0]["status"] == FAIL


def test_a_satisfied_contract_records_nothing():
    module, ns, _ = _install([PROJECTED_INV], lambda gdf, radius=1: "ok")
    module.target_fn(_geo().to_crs(3857), 25000)
    assert ns[VIOLATIONS_GLOBAL] == []


def test_the_guard_checks_by_PARAMETER_NAME_not_position():
    """The reason for wrapping at all: `f(a, b)` where only `b` is constrained."""
    inv = {"check": "projected_crs", "target": "second", "args": {}}
    module, ns, _ = _install([inv], lambda first, second: "ok")
    module.target_fn(_geo().to_crs(3857), _geo("EPSG:4326"))     # only `second` is geographic
    assert len(ns[VIOLATIONS_GLOBAL]) == 1
    assert ns[VIOLATIONS_GLOBAL][0]["target"] == "target_fn(second)"


def test_a_keyword_argument_is_checked_too():
    module, ns, _ = _install([PROJECTED_INV], lambda gdf=None, radius=1: "ok")
    module.target_fn(gdf=_geo("EPSG:4326"))
    assert len(ns[VIOLATIONS_GLOBAL]) == 1


def test_an_unsupplied_parameter_is_not_checked():
    module, ns, _ = _install([PROJECTED_INV], lambda gdf=None: "ok")
    module.target_fn()
    assert ns[VIOLATIONS_GLOBAL] == []


def test_a_violation_never_raises():
    """A breach means the number is wrong; killing the run would destroy the evidence and any
    partial output the user might still want."""
    module, _ns, _ = _install([PROJECTED_INV], lambda gdf: "computed anyway")
    assert module.target_fn(_geo("EPSG:4326")) == "computed anyway"


def test_a_guard_is_not_installed_twice():
    module, ns, _ = _install([PROJECTED_INV], lambda gdf: "ok")
    again = install_contract_guards(ns, {"target_fn": {
        "module": "fake_lib_mod", "symbol": "target_fn", "invariants": [PROJECTED_INV]}})
    assert again == 0
    module.target_fn(_geo("EPSG:4326"))
    assert len(ns[VIOLATIONS_GLOBAL]) == 1, "double-wrapping would double-report"


def test_a_unit_with_no_invariants_is_not_wrapped():
    _module, ns, installed = _install([], lambda gdf: "ok")
    assert installed == 0


def test_an_unimportable_module_leaves_the_run_unguarded_not_broken():
    ns = {}
    installed = install_contract_guards(ns, {"x": {
        "module": "no.such.module", "symbol": "x", "invariants": [PROJECTED_INV]}})
    assert installed == 0


def test_a_function_whose_signature_cannot_be_read_still_runs():
    module, ns, _ = _install([PROJECTED_INV], print)   # builtins have no bindable signature
    module.target_fn("anything")


# ------------------------------------------------------------------ the injected prologue

def test_no_contracts_means_no_prologue():
    """A run importing no library unit pays nothing."""
    assert prologue_source(None) == ""
    assert prologue_source({}) == ""


def test_the_prologue_compiles_and_is_self_contained():
    src = prologue_source({"f": {"module": "m", "symbol": "f", "invariants": [PROJECTED_INV]}})
    compile(src, "<prologue>", "exec")
    assert "import agent_runtime" not in src, "the sandbox cannot import our packages"
    assert CONTRACTS_GLOBAL in src


def test_the_contracts_are_injected_as_a_literal():
    """Read from the mounted registry instead, a guard could silently fail to install — and a
    guard that does not install reads exactly like a contract that passed."""
    src = prologue_source({"f": {"module": "m", "symbol": "f", "invariants": [PROJECTED_INV]}})
    assert "projected_crs" in src
    assert "load_registry" not in src


def test_the_prologue_cannot_break_a_run():
    src = prologue_source({"f": {"module": "does.not.exist", "symbol": "f",
                                "invariants": [PROJECTED_INV]}})
    exec(compile(src, "<prologue>", "exec"), {})


# ------------------------------------------------------------------ reaching the verdict

def test_violations_reach_the_checks_report():
    report = run_checks({VIOLATIONS_GLOBAL: [
        {"check": "projected_crs", "status": FAIL, "target": "f(gdf)", "message": "geographic"}]})
    assert report["verdict"] == FAIL
    assert any(f["target"] == "f(gdf)" for f in report["findings"])


def test_contract_findings_come_before_frame_findings():
    """A contract finding names the unit AND the parameter, which is more actionable than a
    frame-level note about a variable whose role the reader has to infer."""
    report = run_checks({
        VIOLATIONS_GLOBAL: [{"check": "projected_crs", "status": FAIL,
                             "target": "f(gdf)", "message": "geographic"}],
        "some_frame": _geo("EPSG:4326"),
    })
    assert report["findings"][0]["target"] == "f(gdf)"


def test_no_violations_leaves_a_clean_run_clean():
    report = run_checks({VIOLATIONS_GLOBAL: [], "gdf": _geo().to_crs(3857)})
    assert report["verdict"] == PASS


# ------------------------------------------------------------------ resolution from code

def test_contracts_are_resolved_from_the_import_line(monkeypatch):
    import agent_runtime.method_library as ml
    from agent_runtime.code_execution import contracts_for_code

    monkeypatch.setattr(ml, "load_registry", lambda: {
        "ke_x.buffer_it": {"module": "iguide_methods.ke_x.v_abc",
                           "library_symbol": "buffer_it",
                           "invariants": [PROJECTED_INV]}})
    out = contracts_for_code("from iguide_methods.ke_x.v_abc import buffer_it\n")
    assert "buffer_it" in out
    assert out["buffer_it"]["invariants"] == [PROJECTED_INV]


def test_code_importing_nothing_resolves_no_contracts():
    from agent_runtime.code_execution import contracts_for_code

    assert contracts_for_code("import geopandas as gpd\nprint(1)\n") == {}


def test_a_unit_with_no_invariants_is_not_injected(monkeypatch):
    import agent_runtime.method_library as ml
    from agent_runtime.code_execution import contracts_for_code

    monkeypatch.setattr(ml, "load_registry", lambda: {
        "ke_x.plain": {"module": "iguide_methods.ke_x.v_abc", "library_symbol": "plain",
                       "invariants": []}})
    assert contracts_for_code("from iguide_methods.ke_x.v_abc import plain\n") == {}
