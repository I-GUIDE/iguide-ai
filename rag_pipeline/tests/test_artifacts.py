"""The reproducible artifact: what a run records so it can be replayed and COMPARED.

The system's claim is that the deliverable is a runnable, verified artifact and the answer is a
byproduct. Four things have to be recorded for that to mean anything, and the interesting ones
are the two that are easy to get almost right:

* the image by **digest**, not by tag. ``python:3.11-slim`` resolves to different bytes next
  month, so an artifact holding the tag records nothing about the environment it ran in.
* the declared output **values**, not the gate's findings about them. A manifest holding
  "unit is null" instead of ``25000`` keeps the complaint and loses the measurement, leaving a
  re-run nothing to compare against.

The replay itself (``scripts/rerun_artifact.py``) needs Docker, so its end-to-end behaviour is
verified by hand and recorded in the DEVLOG; what is unit-tested here is everything that
decides *what gets written*.
"""

from __future__ import annotations

import json

import pytest

from agent_runtime import artifacts


# ------------------------------------------------------------------ library provenance

def test_a_pinned_library_import_records_its_slice_sha():
    """The v_<sha> segment records WHICH version of an extracted unit produced the result, so a
    replay after a re-ingest imports the same code rather than a newer namesake."""
    units = artifacts.library_units_used(
        "from iguide_methods.ke_cca9b545_chicago.v_45b703eec714 import load_chicago_crime_data")
    assert units == [{"symbol": "load_chicago_crime_data",
                      "module": "iguide_methods.ke_cca9b545_chicago.v_45b703eec714",
                      "slice_sha": "45b703eec714"}]


def test_multiple_symbols_on_one_import_line_are_all_recorded():
    units = artifacts.library_units_used(
        "from iguide_methods.ke_x.v_abc import load_data, plot_map\n")
    assert {u["symbol"] for u in units} == {"load_data", "plot_map"}
    assert all(u["slice_sha"] == "abc" for u in units)


def test_ordinary_imports_are_not_recorded_as_library_units():
    code = "import geopandas as gpd\nfrom shapely.geometry import Point\n"
    assert artifacts.library_units_used(code) == []


def test_an_unpinned_library_import_records_an_empty_sha_rather_than_guessing():
    units = artifacts.library_units_used("from iguide_methods.ke_x import load_data")
    assert units and units[0]["slice_sha"] == ""


# ------------------------------------------------------------------ the manifest

def _manifest(**kw):
    base = dict(code="print(1)", work=None, image="python:3.11-slim", backend="local")
    base.update(kw)
    return artifacts.build_manifest(**base)


def test_a_local_run_records_no_digest_because_there_is_no_image_to_pin():
    assert _manifest(backend="local")["image_digest"] is None


def test_verified_is_true_only_when_the_gate_passed():
    assert _manifest(verification={"verdict": "pass"})["verified"] is True
    assert _manifest(verification={"verdict": "fail"})["verified"] is False
    assert _manifest(verification={"verdict": "cannot_determine"})["verified"] is False
    assert _manifest(verification={})["verified"] is False, (
        "no verification at all must not read as verified")


def test_the_code_hash_changes_with_the_code():
    a = _manifest(code="print(1)")["code_sha256"]
    b = _manifest(code="print(2)")["code_sha256"]
    assert a != b and len(a) == 64


def test_the_manifest_records_the_declared_values():
    outputs = {"radius": {"value": 25000, "unit": "metres"}}
    assert _manifest(declared_outputs=outputs)["declared_outputs"] == outputs


def test_an_unresolvable_digest_is_recorded_as_none_not_omitted(monkeypatch):
    """"We could not pin this" is information a replay needs; a missing key hides it."""
    monkeypatch.setattr(artifacts, "resolve_image_digest", lambda image: None)
    m = _manifest(backend="docker")
    assert "image_digest" in m and m["image_digest"] is None


def test_a_reference_that_is_already_a_digest_is_passed_through():
    ref = "python@sha256:" + "a" * 64
    assert artifacts.resolve_image_digest(ref) == ref


def test_an_empty_image_resolves_to_none():
    assert artifacts.resolve_image_digest("") is None


# ------------------------------------------------------------------ inputs

def test_inputs_are_recorded_with_a_hash(tmp_path):
    (tmp_path / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    rows = artifacts.collect_inputs(tmp_path, staged=["data.csv"])
    assert len(rows) == 1 and len(rows[0]["sha256"]) == 64 and rows[0]["bytes"] == 8


def test_recorded_provenance_is_preserved_and_merged(tmp_path):
    """A staging tool records origin_url/sha256; that must survive into the manifest."""
    (tmp_path / "inputs.jsonl").write_text(json.dumps(
        {"name": "data.csv", "origin_url": "https://example.org/data.csv"}) + "\n",
        encoding="utf-8")
    (tmp_path / "data.csv").write_text("x", encoding="utf-8")
    row = artifacts.collect_inputs(tmp_path, staged=["data.csv"])[0]
    assert row["origin_url"] == "https://example.org/data.csv"
    assert "sha256" in row


def test_an_input_whose_file_is_gone_is_still_listed(tmp_path):
    """Dropping it would make the artifact look self-contained when it is not."""
    (tmp_path / "inputs.jsonl").write_text(
        json.dumps({"name": "vanished.tif", "origin_url": "s3://b/k"}) + "\n", encoding="utf-8")
    names = [r["name"] for r in artifacts.collect_inputs(tmp_path, staged=[])]
    assert "vanished.tif" in names


def test_no_inputs_is_an_empty_list_not_an_error(tmp_path):
    assert artifacts.collect_inputs(tmp_path, staged=[]) == []


# ------------------------------------------------------------------ emission

def test_emit_writes_the_three_files(tmp_path):
    (tmp_path / "checks.json").write_text(json.dumps({"verdict": "pass", "findings": []}))
    (tmp_path / "declared_outputs.json").write_text(
        json.dumps({"area": {"value": 42, "unit": "square_metres"}}))
    (tmp_path / "environment.json").write_text(json.dumps({"python": "3.11.15",
                                                           "package_count": 13}))
    manifest = artifacts.emit(code="print(1)", work=tmp_path, image="python:3.11-slim",
                              backend="local", dependencies=["geopandas"], tier="standard")
    for name in (artifacts.RUN_FILENAME, artifacts.MANIFEST_FILENAME, artifacts.INPUTS_FILENAME):
        assert (tmp_path / name).is_file(), name
    assert manifest["declared_outputs"]["area"]["value"] == 42
    assert manifest["environment"]["python"] == "3.11.15"
    assert manifest["verified"] is True


def test_run_py_is_the_model_source_without_the_injected_epilogue(tmp_path):
    """The persisted source must be exactly what the model wrote — the gate's epilogue is
    appended to the executed script only, never to the artifact."""
    code = "print('hello')\n"
    artifacts.emit(code=code, work=tmp_path, image="", backend="local")
    written = (tmp_path / artifacts.RUN_FILENAME).read_text(encoding="utf-8")
    assert written == code
    assert "invariant" not in written.lower()


def test_emission_failure_never_breaks_a_successful_run(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "build_manifest",
                        lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    assert artifacts.emit(code="x", work=tmp_path, image="", backend="local") == {}


def test_emit_is_skippable_by_env(monkeypatch):
    monkeypatch.setenv("AGENT_ARTIFACT_EMIT", "0")
    assert artifacts.artifacts_enabled() is False
    monkeypatch.delenv("AGENT_ARTIFACT_EMIT", raising=False)
    assert artifacts.artifacts_enabled() is True, "a record that cannot be rebuilt later must default on"


# ------------------------------------------------------------------ replay comparison

def test_float_comparison_tolerates_the_last_bit():
    """Exact equality would flag a re-ordered sum and train everyone to ignore the script."""
    from scripts.rerun_artifact import _same_value

    assert _same_value(3920685613.182, 3920685613.182000001)
    assert not _same_value(3920685613.182, 4200000000.0)


def test_comparison_unwraps_the_declared_spec():
    from scripts.rerun_artifact import _value_of

    assert _value_of({"value": 25000, "unit": "metres"}) == 25000
    assert _value_of(25000) == 25000
