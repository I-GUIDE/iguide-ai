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
                V.Finding("correction", V.PROBLEM, "c"),
                V.Finding("number_scan", V.NOTE, "1 statement(s) were left out", evidence=["d"])]
    out = V.render("The answer.", findings)
    assert out.count("---") == 1 and out.count("⚠️") == 1 and "ℹ️" not in out
    assert out.index("Problem: c") < out.index("Incomplete: b")
    assert "One statement with a figure no tool here can produce was left out." in out
    # "Not checked" items stay in the verdict, not in the text (stage 46).
    assert "- Problem: c" in out and "Not checked" not in out


def test_could_not_check_is_never_called_a_detection():
    """Stage 46: "could not check" no longer prints at all. Across the archive it sat on 22
    correct answers and 0 wrong ones in the stack's runs. It stays in the verdict payload."""
    findings = [V.Finding("gate", V.UNVERIFIABLE, "a unit did not parse")]
    out = V.render("x", findings)
    assert out == "x"
    assert V.status(findings) == "unverified"


def test_a_failed_peer_still_says_the_answer_is_incomplete():
    out = V.render("x", V.from_peer_failures([{"peer": "code"}]))
    assert "may be incomplete" in out and "code execution failed" in out and "⚠️" not in out


def test_a_cut_is_one_quiet_line_with_the_full_count():
    out = V.render("x", [V.Finding("number_scan", V.NOTE, "8 statement(s) were left out",
                                   evidence=list("abcdef"))])
    assert out.endswith("_8 statements with figures no tool here can produce were left out._")
    assert "⚠️" not in out and "ℹ️" not in out


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


# --------------------------------------------------------------------------- the first harness run
# Stage 44's first gate-on harness run (deepseek-v4-flash, 17 tasks) put a banner on 10 of 11
# correct answers. Replaying the scan over 136 recorded answers found the classes below.

def test_the_question_s_coordinates_ground_the_answer_s():
    """`0.1281 W` in the question was read as 0.1281 watts and `41.8827 N` as newtons, so the
    answer's "0.1281° W" resolved nowhere and its sentence was cut."""
    fs = F.build(query="between Trafalgar Square (51.5080 N, 0.1281 W) and Notre-Dame")
    res = F.resolve("Trafalgar Square (51.5080° N, 0.1281° W) is the start.", fs)
    assert [r.resolved for r in res if F.is_claim(r.quantity)] == [True, True]


def test_a_hemisphere_is_a_sign():
    fs = F.build(results=_record("longitude -117.599333\n"))
    (r,) = [r for r in F.resolve("The epicentre is at 117.5993°W.", fs) if F.is_claim(r.quantity)]
    assert r.resolved
    (r,) = [r for r in F.resolve("The epicentre is at 117.5993°E.", fs) if F.is_claim(r.quantity)]
    assert not r.resolved


@pytest.mark.parametrize("record,stated", [
    ("area hectares: 24.0\n", "The flooded area is 24.0 hectares."),
    ('{"watershed_area_km2": 16.2}', "The watershed is 16.20 km²."),
    ('{"nearest_sample_distance_m": 69.28924880528002}', "The nearest sample is 69.29 m away."),
])
def test_a_recorded_number_is_named_by_the_words_beside_it(record, stated):
    """The label used to be the first 60 characters of the whole output, so "24.0" never saw
    the "hectares" written just before it."""
    fs = F.build(results=_record(record))
    (r,) = [r for r in F.resolve(stated, fs) if F.is_claim(r.quantity)]
    assert r.resolved, stated


def test_a_designation_glued_to_a_number_is_not_a_claim():
    assert [q.text for q in F.quantities("Reprojected to UTM zone 16N first.")
            if F.is_claim(q)] == []


def test_a_difference_of_two_grounded_figures_resolves_to_both():
    fs = F.build(results=_record("C1,C3 382254.14\nC2,C3 418879.72\n"))
    answer = ("C1 and C3 give 382,254 person-km; C2 and C3 give 418,880 person-km. "
              "The best pair wins by about 36,626 person-km.")
    res = [r for r in F.resolve(answer, fs) if F.is_claim(r.quantity)]
    assert all(r.resolved for r in res)
    assert {f.value for f in res[-1].parts} == {382254.14, 418879.72}


def test_a_short_figure_is_not_resolved_by_a_coincidental_ratio():
    """Replayed, ratios matched mostly by coincidence ("0.3°" as 118 ÷ 470)."""
    fs = F.build(results=_record("cells 118\ntotal 470\n"))
    res = [r for r in F.resolve("118 of 470 cells qualify; the best is flat (0.3°).", fs)
           if F.is_claim(r.quantity)]
    assert [r.resolved for r in res] == [True, True, False]


def test_a_run_with_nothing_geospatial_is_not_reported_as_unchecked():
    """`not_applicable` means the gate's checks had nothing to apply to; `coverage` means
    geospatial work it could not reach. Only the second is "not checked"."""
    def audit(check):
        return {"invariant_gate": "cannot_determine",
                "issues": [{"source": "invariant_gate", "status": "cannot_determine",
                            "check": check, "claim": "computed value from `this run`",
                            "reason": f"invariant gate ({check}): x"}]}
    assert V.from_gate(audit("not_applicable")) == []
    assert [f.kind for f in V.from_gate(audit("coverage"))] == [V.UNVERIFIABLE]


def test_a_gate_note_is_not_printed():
    out = V.render("x", [V.Finding("gate", V.NOTE, "an optional column is empty")])
    assert out == "x"


def test_an_unresolved_figure_does_not_make_side_runs_speak():
    """The first harness run widened the gate's scope to every run whenever one figure resolved
    nowhere, and a side run's finding then spoke for a correct answer."""
    main = {"name": "execute_code", "tool_call_id": "main",
            "content": json.dumps({"ok": True, "stdout": "schools within 1 mile: 23\n",
                                   "verification": {"verdict": "pass", "counts": {"pass": 4},
                                                    "findings": []}})}
    side = {"name": "execute_code", "tool_call_id": "side",
            "content": json.dumps({"ok": True, "stdout": "x",
                                   "verification": SIDE_RUNS[1]})}
    state = _turn("There are 23 schools within 1 mile; Earth's radius is 6,371.0088 km.",
                  [side, main])
    assert not any(f["check"] == "gate" for f in state["verdict"]["findings"]), state["verdict"]


# --------------------------------------------------------------------------- stage 46

def test_a_decimal_point_does_not_end_a_sentence():
    """The sentence start was the last "." of any kind, so "1,609.344 m … 41.8827° N" split
    into "344 meters) … 8827° N". The fragment resolved to nothing and was cut, which in
    p5-gate T02 removed the headline answer itself."""
    answer = ("**20 schools** are within 1 mile (1,609.344 meters) of the site at 41.8827° N, "
              "87.6233° W. The nearest is 412.5 m away.")
    qs = F.quantities(answer)
    head = [q for q in qs if q.text.startswith(("20", "1,609", "41.8827", "87.6233"))]
    assert len(head) == 4
    assert all(q.sentence.startswith("**20 schools**") for q in head)
    assert [q for q in qs if q.text.startswith("412.5")][0].sentence.rstrip(".") == "The nearest is 412.5 m away"


def test_the_whole_headline_survives_the_scan():
    fs = F.build(results=_record("count 20 nearest 412.53 m radius 1609.344"),
                 query="How many schools are within 1 mile of 41.8827 N, 87.6233 W?")
    answer = ("**20 schools** are within 1 mile (1,609.344 meters) of the site at 41.8827° N, "
              "87.6233° W. The nearest is 412.5 m away.")
    assert all(r.resolved for r in F.resolve(answer, fs) if F.is_claim(r.quantity))


def test_a_unit_conversion_of_a_grounded_figure_resolves():
    """p4-gate U04: "≈ 3.43 minutes (0.0571 hours, ~206 seconds)" lost both conversions. The
    conversion of 3.43 as shown carries its rounding: 0.05717 h ± 0.00015 h covers 0.0571."""
    fs = F.build(results=_record("fastest_minutes: 3.4298"))
    res = [r for r in F.resolve("Fastest: ≈ 3.43 minutes (0.0571 hours, ~206 seconds).", fs)
           if F.is_claim(r.quantity)]
    assert [r.resolved for r in res] == [True, True, True]
    assert [r.fact.source for r in res[1:]] == ["converted", "converted"]


def test_a_recalled_figure_in_another_unit_stays_unresolved():
    """The true catch the cut exists for (p5-fixed-gate T01): a remembered Census figure."""
    fs = F.build(results=_record("area_km2: 2584.6234"))
    answer = ("The area is 2,584.62 km². This is consistent with the official Census figure of "
              "approximately 997.5 sq mi (≈ 2,583.5 km²).")
    res = [r for r in F.resolve(answer, fs) if F.is_claim(r.quantity)]
    assert [r.resolved for r in res] == [True, False, False]


def test_a_short_sum_resolves_when_its_operands_are_in_its_sentence():
    """p5-fixed-gate T10 cut a correct "total sill 0.64" (0.05 + 0.59): two significant digits
    were below the derivation floor."""
    fs = F.build(query="spherical variogram (nugget 0.05, partial sill 0.59, range 897 m)")
    res = [r for r in F.resolve("Nugget 0.05, partial sill 0.59 (total sill = 0.64), "
                                "range 897 m.", fs) if F.is_claim(r.quantity)]
    assert all(r.resolved for r in res)
    assert res[2].fact.source == "derived"


def test_a_short_sum_of_operands_elsewhere_does_not_resolve():
    fs = F.build(query="nugget 0.05, partial sill 0.59")
    answer = "The nugget is 0.05 and the partial sill 0.59. Separately, the ratio was 0.64."
    res = [r for r in F.resolve(answer, fs) if F.is_claim(r.quantity)]
    assert [r.resolved for r in res] == [True, True, False]


def test_numbers_inside_a_list_are_recorded():
    """`"region_bbox": [-87.65221, 41.855, …]` was dropped, so an answer restating its own
    bounding box (deployed-23cfd02-gate U01) resolved to nothing."""
    fs = F.build(results=[{"name": "dem_for_region", "tool_call_id": "c1", "content": json.dumps(
        {"ok": True, "region_bbox": [-87.65221, 41.855, -87.59659, 41.9091]})}])
    res = [r for r in F.resolve("The DEM covers lon −87.652 to −87.597, lat 41.855 to 41.909.",
                                fs) if F.is_claim(r.quantity)]
    assert all(r.resolved for r in res)
    assert any(f.label.endswith("region_bbox: ") for f in fs.facts)
