"""A claim whose part of the record was cut is "unseen": not verified, never an over-reach.

Live, 2026-10-15 (Chicago demo, LISA step): the auditor called the LISA cluster counts
"unsupported due to the truncated execution record, constituting minor over-reach", as its
prompt then told it to. The current turn's record also gets a larger budget now.
"""
from __future__ import annotations

import json

from agent_runtime import claim_checker as cc
from agent_runtime import evidence_quality as eq
from agent_runtime import verdict as V


def _audit(rows, severity="low", summary="minor over-reach"):
    return eq._recompute_verdict({"claim_ledger": rows, "severity": severity, "summary": summary})


def test_unseen_rows_are_not_issues_and_the_summary_does_not_call_them_over_reach():
    out = _audit([
        {"claim": "Moran's I = 0.5646", "evidence_quote": "I = 0.5646", "status": "supported"},
        {"claim": "12 areas are High-High", "evidence_quote": "none", "status": "unseen"}])
    assert out["issues"] == [] and out["severity"] == "none" and not out["hallucination_detected"]
    assert out["unverified_claims"] == ["12 areas are High-High"]
    assert "over-reach" not in out["summary"] and "cut" in out["summary"]


def test_an_absent_row_that_blames_the_cut_is_unseen():
    out = _audit([{"claim": "Englewood is High-High",
                   "evidence_quote": "none — the execution record was truncated", "status": "absent"}])
    assert out["issues"] == [] and out["unverified_claims"] == ["Englewood is High-High"]


def test_a_real_absence_is_still_an_issue_beside_an_unseen_one():
    out = _audit([{"claim": "Published in Nature", "evidence_quote": "none", "status": "absent"},
                  {"claim": "12 areas are High-High", "evidence_quote": "none", "status": "unseen"}])
    assert [i["claim"] for i in out["issues"]] == ["Published in Nature"]
    assert out["unverified_claims"] == ["12 areas are High-High"]


def test_the_banner_says_not_verified_for_unseen_claims():
    f = V.from_audit({"severity": "none", "issues": [],
                      "unverified_claims": ["12 areas are High-High"]}, numeric_claims_resolved=True)
    assert [x.kind for x in f] == [V.UNVERIFIABLE] and "was cut" in f[0].message


def test_the_checker_takes_unseen_claims_and_settles_them():
    audit = {"severity": "none", "issues": [],
             "unverified_claims": ["12 areas are High-High", "Englewood is Low-Low"]}
    assert cc.claims_to_check(audit) == ["12 areas are High-High", "Englewood is Low-Low"]
    out = cc.apply(audit, [
        {"claim": "12 areas are High-High", "verdict": "supported", "detail": "12 HH"},
        {"claim": "Englewood is Low-Low", "verdict": "contradicted", "detail": "Englewood is HH"}])
    assert out["unverified_claims"] == []
    assert out["issues"][0]["claim"] == "Englewood is Low-Low" and out["severity"] == "high"
    f = V.from_audit(out, numeric_claims_resolved=True)
    assert any(x.kind == V.PROBLEM and "Englewood is HH" in x.message for x in f)


def test_the_current_turn_gets_the_larger_budget():
    long_turn = ["execute_code(...) -> " + "LISA cluster per area: " + "x" * 30000]
    rendered = eq._format_execution_context({"this_turn": long_turn})
    assert len(rendered) > 30000 and eq._TURN_RECORD_MAX_CHARS == 32000
    too_long = ["y" * 50000]
    assert "chars elided" in eq._format_execution_context({"this_turn": too_long})
