"""agent_runtime.claim_checker: disputed claims checked by code against the turn's own files.

Offline: the check writer is a stub that returns a fixed plan, and the checks run through the
local subprocess backend. The recorded Chicago turns are replayed from the archive when present.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_runtime import claim_checker as cc
from agent_runtime import verdict as V
from agent_runtime.code_execution import LocalSubprocessExecutor

ARCHIVE = Path("/Users/yfkang/i-guide-platform-flask-servers/gis_harness_runs_archive/claim_check")


class _HostRunner:
    """Runs the check script with this interpreter in a scratch folder holding the inputs.

    The local subprocess backend uses a minimal interpreter without geopandas; the recorded
    checks read GeoJSON, which the deployed sandbox image can."""

    def __init__(self, tmp):
        self.tmp = Path(tmp)

    def execute(self, code, *, input_files=(), **kw):
        import shutil
        import subprocess
        import sys

        for f in input_files:
            shutil.copyfile(f["source"], self.tmp / f["dest"])
        (self.tmp / "script.py").write_text(code)
        p = subprocess.run([sys.executable, "script.py"], cwd=self.tmp, capture_output=True,
                           text=True, timeout=300)
        return type("R", (), {"stdout": p.stdout, "stderr": p.stderr, "error": None})()


class _Writer:
    """Returns a fixed plan, as the model would."""

    def __init__(self, plan):
        self.plan, self.prompts = plan, []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return type("M", (), {"content": json.dumps(self.plan)})()


@pytest.fixture()
def files(tmp_path):
    rates = tmp_path / "rates.csv"
    rates.write_text("community,rate\nFULLER PARK,118.0\nENGLEWOOD,100.1\nOHARE,12.0\n")
    regions = tmp_path / "regions.csv"
    regions.write_text("community,region\nFULLER PARK,6\nENGLEWOOD,6\nOHARE,1\n")
    return [{"file_id": "a", "name": "01_rates.csv", "path": str(rates)},
            {"file_id": "b", "name": "02_regions.csv", "path": str(regions)}]


SETUP = ("import pandas as pd, re\n"
         "rates = pd.read_csv('01_rates.csv'); regions = pd.read_csv('02_regions.csv')\n"
         "norm = lambda s: re.sub(r'[^A-Z0-9]', '', str(s).upper())\n"
         "def member(r, name):\n"
         "    got = regions[regions.community.map(norm) == norm(name)].region.tolist()\n"
         "    return (got and got[0] == r), f'{name} is in region {got[0] if got else None}'\n")


def test_rankings_and_memberships_are_decided_by_running_code(files):
    claims = ["Highest-rate community area: Fuller Park", "Region 6 includes Englewood",
              "Region 1 includes Fuller Park", "Region 1 includes O'Hare"]
    plan = {"setup": SETUP, "checks": [
        {"i": 1, "checkable": True, "code": "top = rates.sort_values('rate').iloc[-1]\n"
                                            "ok = norm(top.community) == norm('Fuller Park')\n"
                                            "detail = f'max rate {top.rate} in {top.community}'"},
        {"i": 2, "checkable": True, "code": "ok, detail = member(6, 'Englewood')"},
        {"i": 3, "checkable": True, "code": "ok, detail = member(1, 'Fuller Park')"},
        {"i": 4, "checkable": True, "code": "ok, detail = member(1, \"O'Hare\")"}]}
    writer = _Writer(plan)
    v = cc.check_claims(claims, files, writer, executor=LocalSubprocessExecutor())
    assert [x["verdict"] for x in v] == ["supported", "supported", "contradicted", "supported"]
    assert "118.0" in v[0]["detail"] and "region 6" in v[2]["detail"]
    assert "01_rates.csv" in writer.prompts[0] and "community:" in writer.prompts[0]


def test_what_cannot_be_checked_is_unverified_not_contradicted(files):
    plan = {"setup": SETUP, "checks": [
        {"i": 1, "checkable": False, "why": "no file holds 2023 figures"},
        {"i": 2, "checkable": True, "code": "ok = rates.nope.max() > 0"}]}
    v = cc.check_claims(["Crime fell since 2023", "Rates are positive"], files, _Writer(plan),
                        executor=LocalSubprocessExecutor())
    assert [x["verdict"] for x in v] == ["unverified", "unverified"]
    assert "2023" in v[0]["detail"] and "AttributeError" in v[1]["detail"]


def test_a_broken_setup_leaves_every_claim_unverified(files):
    plan = {"setup": "raise RuntimeError('cannot load')",
            "checks": [{"i": 1, "checkable": True, "code": "ok = True"}]}
    v = cc.check_claims(["x"], files, _Writer(plan), executor=LocalSubprocessExecutor())
    assert v[0]["verdict"] == "unverified" and "setup failed" in v[0]["detail"]


def test_no_files_means_unverified_without_calling_the_model():
    writer = _Writer({})
    v = cc.check_claims(["Fuller Park is highest"], [], writer)
    assert v[0]["verdict"] == "unverified" and writer.prompts == []


def test_a_clipped_output_loses_only_the_tail():
    class Clipped:
        def execute(self, code, **kw):
            line = cc.RESULT_MARK + json.dumps({"1": {"ok": True, "detail": "d"}})
            return type("R", (), {"stdout": line + "\n" + cc.RESULT_MARK + '{"2": {"ok": tr',
                                  "error": None, "stderr": ""})()
    out = cc.run_checks({"checks": [{"i": 1, "checkable": True, "code": "ok=1"}]}, [], Clipped())
    assert out == {"1": {"ok": True, "detail": "d"}}


def test_apply_withdraws_marks_and_keeps_evidence():
    audit = {"severity": "high", "hallucination_detected": True, "issues": [
        {"claim": "Highest-rate community area: Fuller Park", "reason": "not in record"},
        {"claim": "Region 1 includes Austin", "reason": "not in record"},
        {"claim": "Crime fell", "reason": "not in record"},
        {"claim": "computed value from `gdf`", "source": "invariant_gate"}]}
    verdicts = [
        {"claim": "Highest-rate community area: Fuller Park", "verdict": "supported", "detail": "max"},
        {"claim": "Region 1 includes Austin", "verdict": "contradicted",
         "detail": "Austin is in region 2"},
        {"claim": "Crime fell", "verdict": "unverified", "detail": "no earlier year"}]
    assert cc.claims_to_check(audit) == [v["claim"] for v in verdicts]
    out = cc.apply(audit, verdicts)
    claims = [i["claim"] for i in out["issues"]]
    assert "Highest-rate community area: Fuller Park" not in claims
    austin = next(i for i in out["issues"] if i["claim"] == "Region 1 includes Austin")
    assert austin["status"] == "fail" and "region 2" in austin["reason"]
    assert out["claim_check"] == {"supported": 1, "contradicted": 1, "unverified": 1}


def test_all_supported_clears_the_audit():
    audit = {"severity": "high", "hallucination_detected": True,
             "issues": [{"claim": "Highest-rate community area: Fuller Park"}]}
    out = cc.apply(audit, [{"claim": "Highest-rate community area: Fuller Park",
                            "verdict": "supported", "detail": "max 118.0"}])
    assert out["issues"] == [] and out["severity"] == "none" and not out["hallucination_detected"]
    assert V.from_audit(out, numeric_claims_resolved=True) == []


def test_the_banner_shows_evidence_for_contradicted_and_not_verified_for_the_rest():
    audit = cc.apply({"severity": "high", "issues": [
        {"claim": "Region 1 includes Austin"}, {"claim": "Crime fell"}]},
        [{"claim": "Region 1 includes Austin", "verdict": "contradicted", "detail": "Austin is in region 2"},
         {"claim": "Crime fell", "verdict": "unverified", "detail": "no earlier year"}])
    audit["severity"] = "high"
    f = V.from_audit(audit, numeric_claims_resolved=True)
    kinds = {x.kind: x.message for x in f}
    assert V.PROBLEM in kinds and "Austin is in region 2" in kinds[V.PROBLEM]
    assert V.UNVERIFIABLE in kinds and kinds[V.UNVERIFIABLE].startswith("not verified")


def test_off_by_default(monkeypatch):
    from agent_runtime.supervisor.graph import _check_disputed_claims

    monkeypatch.delenv(cc.ENABLE_ENV, raising=False)
    audit = {"severity": "high", "issues": [{"claim": "x"}]}
    assert _check_disputed_claims(audit, {}, {}, None, llm=None) is audit


def test_turn_files_keeps_tables_and_vectors_named_in_the_results(tmp_path):
    keep = tmp_path / "rates.geojson"; keep.write_text("{}")
    skip = tmp_path / "map.png"; skip.write_bytes(b"\x89PNG")
    paths = {"file_aaaaaaaaaaaa": (keep, "rates.geojson"), "file_bbbbbbbbbbbb": (skip, "map.png")}
    got = cc.turn_files([{"tool_results": "wrote file_aaaaaaaaaaaa and file_bbbbbbbbbbbb"}],
                        resolve=lambda fid: paths.get(fid, (None, None)))
    assert [g["file_id"] for g in got] == ["file_aaaaaaaaaaaa"] and got[0]["name"] == "01_rates.geojson"


# ----------------------------------------------------------------------- the recorded Chicago turns

@pytest.mark.parametrize("turn", ["turnA", "turnB"])
def test_the_recorded_chicago_turn_replays_to_the_ground_truth(turn, tmp_path):
    d = ARCHIVE / turn
    if not (d / "plan.json").exists():
        pytest.skip("the recorded Chicago turns are in the git-excluded archive")
    claims = json.loads((d / "claims.json").read_text())
    files = [{"file_id": f.stem, "name": f.name, "path": str(f)} for f in sorted(d.glob("*"))
             if f.suffix in (".geojson", ".csv")]
    plan = json.loads((d / "plan.json").read_text())
    v = cc.check_claims(claims, files, _Writer(plan), executor=_HostRunner(tmp_path))
    if turn == "turnA":
        assert v[0]["verdict"] == "supported" and "FULLER PARK" in v[0]["detail"].upper()
    else:
        truth = json.loads((d / "truth.json").read_text())
        assert all((x["verdict"] == "supported") == truth[x["claim"]] for x in v)
        assert sum(x["verdict"] == "contradicted" for x in v) == 21
