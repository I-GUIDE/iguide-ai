"""Stage 44: the answer is held to the turn's facts number by number, under one verdict.

docs/design-review-2026-10.md, flaws 2 and 7. Each test holds a class, not an incident.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("pint")

from agent_runtime import facts as F
from agent_runtime import verdict as V
from agent_runtime.supervisor import graph as g
from agent_runtime.supervisor.graph import run_supervisor

from rag_pipeline.tests.test_regrounding_turn_record import scripted_producers


def _record(*stdouts):
    return [{"name": "execute_code", "tool_call_id": f"c{i}",
             "content": json.dumps({"ok": True, "stdout": s})} for i, s in enumerate(stdouts)]


# --------------------------------------------------------------------------- resolution

@pytest.mark.parametrize("stated", ["2,584.6 km²", "2,585 km²", "≈ 998 square miles",
                                    "258,462 ha", "about 2,580 km²", "2584.62 km2"])
def test_a_figure_resolves_in_any_unit_and_at_the_precision_it_is_shown(stated):
    fs = F.build(results=_record("Area: 2584.62 km²\n"))
    (r,) = [r for r in F.resolve(f"The county is {stated}.", fs) if F.is_claim(r.quantity)]
    assert r.resolved, stated


@pytest.mark.parametrize("stated", ["2,600.4 km²", "2,590 km²", "1,200 square miles"])
def test_a_figure_that_differs_beyond_its_shown_precision_does_not(stated):
    fs = F.build(results=_record("Area: 2584.62 km²\n"))
    (r,) = [r for r in F.resolve(f"The county is {stated}.", fs) if F.is_claim(r.quantity)]
    assert not r.resolved, stated


def test_a_small_bare_number_does_not_ground_a_figure_with_a_unit():
    """`"n": 5` in a record is not evidence for "a 5-hour drive"."""
    fs = F.build(results=[{"name": "t", "tool_call_id": "c", "content": json.dumps({"n": 5})}])
    (r,) = [r for r in F.resolve("A 5-hour drive separates them.", fs)]
    assert not r.resolved


def test_identifiers_years_list_markers_code_and_links_are_not_claims():
    text = ("1. The boundary (GEOID 17019, EPSG:26916) is from 2024.\n"
            "```python\nx = 460\n```\n[the file](/agent/files/file_2272c8426ec9/download)")
    claims = [q.text for q in F.quantities(text) if F.is_claim(q)]
    assert claims == [], claims


def test_the_question_and_the_evidence_count_as_record():
    fs = F.build(query="Which schools are within 1 mile of 41.9231 N?",
                 evidence=[{"title": "School census", "contents": "2,156 schools in 2023"}])
    res = F.resolve("Within 1 mile of 41.9231 N: of the 2,156 schools counted.", fs)
    assert all(r.resolved for r in res if F.is_claim(r.quantity))


# --------------------------------------------------------------------------- the cut

@pytest.mark.parametrize("memorised", [
    "Its population is about 205,865 people.",            # not travel: stage 39 never saw it
    "The road distance is roughly 460 km.",
    "The Eurostar takes about 2 hours 16 minutes.",
])
def test_a_figure_no_bound_tool_produces_is_cut_whatever_it_is_about(memorised):
    fs = F.build(results=_record("area 2584.62 km2\n"))
    answer = f"The county is 2,584.6 km². {memorised}"

    def producers(sents):
        return {s: None for s in sents}               # no bound tool produces any of them

    out, dropped, kept, _, _ = g._cut_unproducible_figures(answer, fs, producers)
    assert out == "The county is 2,584.6 km².", out
    assert dropped and not kept


def test_a_figure_a_bound_tool_could_produce_is_kept_and_reported():
    fs = F.build(results=_record("area 2584.62 km2\n"))
    answer = "The county is 2,584.6 km². Its population is about 205,865 people."
    out, dropped, kept, _, _ = g._cut_unproducible_figures(
        answer, fs, lambda sents: {s: "fetch_public_data" for s in sents})
    assert out == answer and not dropped
    assert [r.quantity.text for r in kept] == ["205,865"]


# --------------------------------------------------------------------------- one verdict

def test_no_findings_no_banner():
    assert V.render("The answer.", []) == "The answer."


def test_one_banner_however_many_checks_spoke():
    findings = [V.Finding("gate", V.UNVERIFIABLE, "a"), V.Finding("peer", V.UNVERIFIABLE, "b"),
                V.Finding("correction", V.PROBLEM, "c"), V.Finding("number_scan", V.NOTE, "d")]
    out = V.render("The answer.", findings)
    assert out.count("---") == 1 and out.count("⚠️") == 1 and "ℹ️" not in out
    assert out.index("Problem: c") < out.index("Not checked: a")


def test_could_not_check_is_never_called_a_detection():
    out = V.render("x", [V.Finding("gate", V.UNVERIFIABLE, "a unit did not parse")])
    assert "⚠️" not in out and "hallucination" not in out.lower()
    assert "not the same as wrong" in out


def test_a_failed_audit_is_reported_as_not_checked_not_as_clean():
    """It used to return severity "unknown", which raised no flag: silent, like a clean audit."""
    findings = V.from_audit({"severity": "unknown", "summary": "Audit LLM call failed."},
                            numeric_claims_resolved=True)
    assert [f.kind for f in findings] == [V.UNVERIFIABLE]


def test_the_audit_does_not_judge_numbers():
    """A numeric claim is the number scan's; the audit's sampled verdict on it is not used."""
    audit = {"severity": "high", "issues": [{"claim": "roughly 460 km by road", "reason": "x"},
                                             {"claim": "borders six counties", "reason": "x"}]}
    msgs = [f.message for f in V.from_audit(audit, numeric_claims_resolved=True)]
    assert len(msgs) == 1 and "six counties" in msgs[0]


# --------------------------------------------------------------------------- the whole turn

SIDE_RUNS = [
    # a directory listing, a lookup that matched nothing, a failed first attempt
    {"verdict": "cannot_determine", "counts": {"cannot_determine": 1},
     "findings": [{"check": "not_applicable", "status": "cannot_determine", "target": "this run",
                   "message": "nothing in this run was checkable"}]},
    {"verdict": "cannot_determine", "counts": {"cannot_determine": 1},
     "findings": [{"check": "all_nan", "status": "cannot_determine", "target": "unnamed",
                   "message": "frame is empty"}]},
]


def _turn(answer, results, **kw):
    def analyze(q, ev, st):
        return {"summary": "done",
                "tool_calls": [{"name": r["name"], "args": {}, "id": r["tool_call_id"]}
                               for r in results],
                "tool_results": results}

    box = {"i": 0}

    def decide(s, d):
        box["i"] += 1
        return "analyze" if box["i"] == 1 else "done"

    return run_supervisor("schools within 1 mile", llm=lambda p: "ok", thread_id="sess-verdict",
                          decide_fn=decide, analyze_fn=analyze, search_fn=lambda q, s: [],
                          synthesize_fn=lambda *a, **k: answer, do_rerank=False, do_audit=False,
                          **kw)


def test_side_runs_do_not_speak_for_the_answer():
    """Live 2026-10-08 21:34 UTC (sess-07bc717f): a correct 23-school answer showed COULD NOT
    VERIFY whose findings all came from side runs. The run that produced the numbers raised
    none, and it is the one the answer uses."""
    main = {"name": "execute_code", "tool_call_id": "main",
            "content": json.dumps({"ok": True, "stdout": "schools within 1 mile: 23\n"
                                                         "nearest: 412.7 m\n",
                                   "verification": {"verdict": "pass", "counts": {"pass": 4},
                                                    "findings": []}})}
    side = [{"name": "execute_code", "tool_call_id": f"side{i}",
             "content": json.dumps({"ok": True, "stdout": "x", "verification": rep})}
            for i, rep in enumerate(SIDE_RUNS)]
    state = _turn("There are 23 schools within 1 mile; the nearest is 412.7 m away.",
                  [*side, main])
    assert "could be checked" not in state["final_answer"], state["final_answer"]
    assert state["verdict"]["status"] == "verified"


def test_the_run_the_answer_uses_still_speaks():
    main = {"name": "execute_code", "tool_call_id": "main",
            "content": json.dumps({"ok": True, "stdout": "area 0.196\n",
                                   "verification": {"verdict": "fail", "counts": {"fail": 1},
                                                    "findings": [{"check": "projected_crs",
                                                                  "status": "fail",
                                                                  "target": "gdf.area",
                                                                  "message": "in degrees"}]}})}
    state = _turn("The area is 0.196.", [main])
    assert state["verdict"]["status"] == "problem", state["verdict"]


def test_the_writer_is_shown_the_typed_facts(monkeypatch):
    seen = {}

    def synth(q, ev, ar, cr, ch, note=None):
        seen["note"] = note or ""
        return "The buffer covers 6.27 km²."

    typed = {"name": "buffer_layer", "tool_call_id": "b1",
             "content": json.dumps({"ok": True, "outputs": [
                 {"name": "total_area", "value": 6.273, "unit": "km^2",
                  "dimension": "[length] ** 2", "measured_in_crs": "EPSG:32616"}]})}

    def analyze(q, ev, st):
        from agent_runtime import turn_log
        log, run = turn_log.active(), turn_log.active_run()
        log.record_result(peer="analyze", run=run, name="buffer_layer", args={},
                          call_id="b1", content=typed["content"])
        return {"summary": "done"}

    box = {"i": 0}
    state = run_supervisor("buffer area", llm=lambda p: "ok", thread_id="sess-facts",
                           decide_fn=lambda s, d: "analyze" if not s.get("actions") else "done",
                           analyze_fn=analyze, search_fn=lambda q, s: [], synthesize_fn=synth,
                           do_rerank=False, do_audit=False)
    assert "total_area = 6.273 km^2 (measured in EPSG:32616)" in seen["note"]
    assert state["verdict"]["numbers"][0]["fact"] is not None
