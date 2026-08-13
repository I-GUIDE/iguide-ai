"""The last link: a gate verdict has to reach the ANSWER, or the gate is decoration.

Everything else about the invariant gate can be right — the check fires, the verdict is `fail`,
`checks.json` is written — and the user still gets a confidently wrong number, because the path
from the tool result to the answer's grounding caveat is a tree walk over JSON-inside-JSON with
a depth limit, a finding cap, and a visited set. Two defects lived in exactly that gap:

* the visited set held ``id(node)`` across the whole traversal while each ``json.loads`` graph
  was reachable only from its call frame, so CPython recycled addresses between branches;
* the walk keyed on ``status == "fail"`` findings rather than on the verdict, so a truncated
  report and every ``cannot_determine`` run passed through as verified.

Both are invisible to a single-execute_code test. The matrix below is the point: N results with
the failure at each position, which is the shape a real turn has — the tool description tells
the model to fix and re-run, so multiple executions per turn is the normal case, not the edge.
"""

from __future__ import annotations

import json

import pytest

from agent_runtime.supervisor.graph import (_gate_failures, _gate_issues_from,
                                            _reconcile_audit_with_artifacts)

PASSING = {"verdict": "pass", "inspected": ["gdf"], "counts": {"pass": 3},
           "findings": [{"check": "projected_crs", "status": "pass", "target": "gdf",
                         "message": "projected CRS EPSG:32616"}]}
FAILING = {"verdict": "fail", "inspected": ["gdf"], "counts": {"fail": 1},
           "findings": [{"check": "projected_crs", "status": "fail", "target": "gdf",
                         "message": "EPSG:4326 is GEOGRAPHIC"}]}


def _tool_result(report: dict) -> str:
    """A real execute_code payload: the report arrives as a JSON string inside a ToolMessage."""
    return json.dumps({"ok": True, "stdout": "done\n", "exit_code": 0,
                       "artifacts": [{"name": "map.png"}], "verification": report})


# ------------------------------------------------------------------ the id()-reuse matrix

@pytest.mark.parametrize("total", [1, 2, 3, 5, 8])
def test_a_failure_is_found_at_every_position_among_passing_runs(total):
    """Replayed against the pre-fix walk, roughly half of the (total x position) combinations
    returned ZERO gate findings — 9/19 and 14/21 on two runs of the same matrix. The count
    varies because it depends on the allocator reusing an address, which is precisely what makes
    the bug nasty: it is load-dependent, so it cannot be reproduced on demand and a single-case
    test can pass on the very run that would ship a wrong number.

    The most natural sequence of all was among the misses: a passing data load followed by a
    failing measurement."""
    for position in range(total):
        results = [_tool_result(FAILING if i == position else PASSING) for i in range(total)]
        ctx = {"messages": [{"role": "tool", "content": r} for r in results]}
        found = _gate_failures(ctx)
        assert found, f"missed a fail at {position + 1} of {total}"
        assert any(f.get("check") == "projected_crs" for f in found)


def test_all_passing_runs_produce_no_issue():
    """The other half of the matrix: no false positives, or the caveat becomes noise."""
    ctx = {"messages": [{"role": "tool", "content": _tool_result(PASSING)} for _ in range(8)]}
    assert _gate_failures(ctx) == []


def test_several_failures_are_all_reported():
    ctx = {"messages": [{"role": "tool", "content": _tool_result(FAILING)} for _ in range(4)]}
    assert len(_gate_failures(ctx)) == 4


# ------------------------------------------------------------------ verdict, not findings

def test_a_truncated_report_still_yields_an_issue():
    """``_read_checks`` caps findings at 12. The judgement can outlive its evidence, and a
    verdict with an empty findings list must not read as "nothing was wrong"."""
    issues = _gate_issues_from({"verdict": "fail", "findings": [], "counts": {"fail": 3}})
    assert len(issues) == 1
    assert "unverified" in issues[0]["message"]
    assert issues[0]["status"] == "fail"


def test_cannot_determine_is_not_a_pass():
    """An unverifiable number and a verified one must not reach the answer the same way. Keying
    on fail findings alone meant an all-unknown run was actively relabelled "Grounded"."""
    issues = _gate_issues_from(
        {"verdict": "cannot_determine",
         "findings": [{"check": "coverage", "status": "cannot_determine",
                       "target": "module scope", "message": "no frame was reachable"}]})
    assert len(issues) == 1 and issues[0]["status"] == "cannot_determine"


def test_a_pass_yields_nothing():
    assert _gate_issues_from(PASSING) == []
    assert _gate_issues_from({}) == []


# ------------------------------------------------------------------ severity reaches the audit

def _reconciled(report: dict):
    ctx = {"messages": [{"role": "tool", "content": _tool_result(report)}]}
    audit = {"hallucination_detected": False, "severity": "none", "issues": [],
             "summary": "Grounded: flagged claims are supported by the execution record."}
    return _reconcile_audit_with_artifacts(audit, [{"name": "map.png"}], ctx)


def test_a_gate_failure_overrides_a_grounded_audit():
    """The LLM auditor saw a number in the execution record and called it supported. It IS in
    the record — and it is wrong. Only the deterministic gate knows that."""
    out = _reconciled(FAILING)
    assert out["hallucination_detected"] is True
    assert out["severity"] == "high"
    assert out["invariant_gate"] == "fail"
    # The REMEDY has to travel, not just the complaint -- the model needs "reproject", and it is
    # the gate's message that carries it.
    assert any("GEOGRAPHIC" in i["reason"] for i in out["issues"])
    assert "not verified" in out["summary"]


def test_an_unverifiable_run_is_flagged_at_medium_not_high():
    """"We could not check this" is weaker evidence than "this is wrong", and collapsing the two
    would make the high-severity signal meaningless."""
    out = _reconciled({"verdict": "cannot_determine", "counts": {"cannot_determine": 1},
                       "findings": [{"check": "coverage", "status": "cannot_determine",
                                     "target": "module scope", "message": "unreachable"}]})
    assert out["severity"] == "medium", "an unchecked run is not a detected error"
    assert out["invariant_gate"] == "cannot_determine"
    assert "not the same as them being wrong" in out["summary"]


def test_a_clean_run_keeps_its_grounded_verdict():
    out = _reconciled(PASSING)
    assert out is not None and out["severity"] == "none"
    assert "invariant_gate" not in out and out["hallucination_detected"] is False
