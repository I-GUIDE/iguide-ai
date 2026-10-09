"""Memorised travel figures must not reach the user just because the audit did not flag them.

Stage 37 cuts a road or rail figure only when the LLM audit flags it. In 2 of 2 local replays of
"what is the area of Champaign County, Illinois, and how far is London from Paris?"
(AGENT_MODE=local, Lumen deepseek-v4-flash, 2026-10-08) the answer still shipped memorised travel
figures with no caveat:

* run 1, "roughly 490 km by rail and ~450 km by road": the auditor flagged it, and
  `_reconcile_audit_with_artifacts` removed every issue. Rule (2) matched the disputed number as a
  SUBSTRING of the whole JSON record, so "450" is "in the record" when a coordinate or an id holds
  those three digits.
* run 2, "Driving is roughly 450–470 km": the auditor passed it.

So: (1) disputed numbers match on numeric boundaries, (2) a deterministic scan of the answer cuts
travel sentences whose distance or time is in no tool result when no routing tool is bound, and
(3) a lead-in whose list was cut entirely goes with it.

Fully stubbed, like test_regrounding_turn_record.py: no network, no keys.
"""

from __future__ import annotations

import json

import pytest

from rag_pipeline.tests.test_regrounding_turn_record import (
    ANSWER_1, FIRST_PASS, GROUNDED, ROUTING_AUDIT, SECOND_PASS, _run, scripted_producers)

from agent_runtime.supervisor import graph as g

# The two local answers, in their structure. The figures are the ones that shipped.
RUN_1 = (
    "## Champaign County, Illinois — Area\n"
    "- **2,584.6 km²** (≈ **998 square miles**), GEOID 17019\n\n"
    "## London to Paris — Distance\n"
    "- **340.0 km** (≈ **211 miles**) as the crow flies\n\n"
    "This is the straight-line distance between the two city centres. Actual travel is longer: "
    "roughly 490 km by rail and ~450 km by road.\n\n"
    "The Champaign County boundary is on your interactive map.")
RUN_2 = (
    "**Champaign County** covers **2,584.6 km²** (998 square miles).\n\n"
    "**London to Paris** is **340.0 km** in a straight line between the geocoded centres. "
    "Driving is roughly 450–470 km, and the Eurostar takes about 2 hours 15 minutes.\n\n"
    "The boundary is on your interactive map.")
# Item 3: every list item under the lead-in is a memorised travel figure.
LEAD_IN = (
    "**London to Paris** is **340.0 km** in a straight line.\n\n"
    "Actual travel distances are longer:\n"
    "- **By rail** (Eurostar): about 490 km\n"
    "- **By road**: roughly 450 km\n\n"
    "The boundary is on your interactive map.")

_ADMIN = FIRST_PASS["tool_results"][0]["content"]


def _record(*results):
    return {"analysis_results": {"summary": "done",
                                 "tool_results": [{"name": n, "tool_call_id": f"c{i}", "content": c}
                                                  for i, (n, c) in enumerate(results)]}}


def _flag(claim):
    return {"verdict": "partially_supported", "severity": "high", "hallucination_detected": True,
            "summary": "not in the record",
            "issues": [{"claim": claim, "reason": "absent from every tool result"}]}


# --- 1. rule (2) and rule (2') match a number on numeric boundaries ------------------------------

@pytest.mark.parametrize("content", [
    # a coordinate holds the three digits
    json.dumps({"results": [{"place": "Paris", "lat": 48.84502, "lon": 2.346941}]}),
    # so does an id
    json.dumps({"ok": True, "file_id": "file_2272c8450ec9"}),
    # and a larger number
    json.dumps({"ok": True, "stdout": "population: 14502\n"}),
])
def test_a_disputed_number_inside_another_number_is_not_in_the_record(content):
    ctx = _record(("geocode_places", content))
    assert "450" in json.dumps(ctx)                       # the old substring match accepted it
    out = g._reconcile_audit_with_artifacts(_flag("~450 km by road"), [], execution_context=ctx)
    assert out["issues"], out
    assert out["hallucination_detected"] is True


@pytest.mark.parametrize("stdout", ["road length: 450 km\n", "length_km=450.2\n",
                                    "total: 1450\n", "d=-450\n", "n\n450\n"])
def test_a_disputed_number_that_is_in_the_record_is_still_reconciled(stdout):
    ctx = _record(("execute_code", json.dumps({"ok": True, "stdout": stdout})))
    claim = "1,450 segments" if "1450" in stdout else "450 km of road"
    out = g._reconcile_audit_with_artifacts(_flag(claim), [], execution_context=ctx)
    assert not out["issues"], (stdout, out)


def test_rule_2_prime_matches_on_boundaries_too():
    """Under a gate verdict only ungated tool results are checkable — on boundaries as well."""
    report = {"verdict": "cannot_determine", "counts": {"pass": 6, "cannot_determine": 1},
              "findings": [{"check": "coverage", "status": "cannot_determine",
                            "target": "this run", "message": "no frame at module scope"}]}
    fail_exec = json.dumps({"ok": True, "stdout": "AREA: 0.196\n", "verification": report})
    geocode = json.dumps({"results": [{"place": "Paris", "lat": 48.84502}]})
    ctx = _record(("execute_code", fail_exec), ("geocode_places", geocode))
    assert g._gate_failures(ctx)
    out = g._reconcile_audit_with_artifacts(_flag("~450 km by road"), [], execution_context=ctx)
    assert any(i.get("claim") == "~450 km by road" for i in out["issues"]), out
    # the same number on its own in the ungated result is still accepted
    ctx = _record(("execute_code", fail_exec),
                  ("admin_boundary", json.dumps({"ok": True, "perimeter_km": 450})))
    out = g._reconcile_audit_with_artifacts(_flag("~450 km by road"), [], execution_context=ctx)
    assert not any(i.get("claim") == "~450 km by road" for i in out["issues"]), out


def test_a_decimal_tail_is_not_a_claim_number():
    """"0.450" is not the number 450; extracting its tail is what let a coordinate match it."""
    assert g._claim_numbers("an index of 0.450") == []
    assert g._claim_numbers("2,584.6 km² and 340.0 km") == ["2584", "340"]


# --- 2. the answer-level scan --------------------------------------------------------------------

def test_run_1_the_flag_is_no_longer_reconciled_away(monkeypatch, ledger):
    """The auditor flagged the travel figure; the record's coordinates must not ground it."""
    geocode = json.dumps({"results": [
        {"place": "London", "found": True, "lat": 51.507446, "lon": -0.127765},
        {"place": "Paris", "found": True, "lat": 48.84502, "lon": 2.349014}]})
    first = {**FIRST_PASS, "tool_results": [
        FIRST_PASS["tool_results"][0],
        {"name": "geocode_places", "tool_call_id": "c2", "content": geocode},
        *FIRST_PASS["tool_results"][2:]]}
    state, seen = _run(monkeypatch, [_flag("~450 km by road"), GROUNDED],
                       passes=(first, SECOND_PASS), answers=(RUN_1,))
    final = state["final_answer"]
    assert "450" not in final and "490" not in final, final
    assert "340.0 km" in final and "2,584.6 km²" in final and "GEOID 17019" in final
    # Stage 42: a FLAGGED claim no bound tool produces is cut by the producer check, with its
    # general note; the routing note is the unflagged scan's (run 2 below).
    assert "no tool here can produce" in final
    assert state["actions"].count("analyze") == 1, state["actions"]


def test_run_2_unflagged_travel_figures_are_cut(monkeypatch, ledger):
    state, _ = _run(monkeypatch, [GROUNDED], answers=(RUN_2,))
    final = state["final_answer"]
    assert "450" not in final and "470" not in final and "2 hours" not in final, final
    assert "**London to Paris** is **340.0 km** in a straight line between the geocoded centres." \
        in final
    assert "2,584.6 km²" in final and "interactive map" in final
    # Stage 44: the scan's own note, one general sentence for every cut class.
    assert "no tool here can produce" in final
    assert "Grounding check" not in final, final
    assert state["actions"].count("analyze") == 1


def test_the_scan_leaves_travel_figures_alone_when_a_routing_tool_is_bound(monkeypatch, ledger):
    first = {**FIRST_PASS, "bound_tools": ["execute_code", "network_route_distance"],
             "bound_tool_docs": {"execute_code": "run code",
                                 "network_route_distance": "road distance between points"}}
    state, _ = _run(monkeypatch, [GROUNDED], passes=(first, SECOND_PASS), answers=(RUN_2,))
    assert "450–470 km" in state["final_answer"]


def test_a_travel_figure_a_tool_computed_is_kept(monkeypatch, ledger):
    exec_ok = json.dumps({"ok": True, "stdout": "London to Paris: 340.0 km\n"
                                                "road distance (network): 463.8 km\n"
                                                "drive time: 5.2 h\n"})
    first = {**FIRST_PASS, "tool_results": [*FIRST_PASS["tool_results"][:3],
                                            {**FIRST_PASS["tool_results"][3], "content": exec_ok}]}
    answer = ("London to Paris is 340.0 km in a straight line. By road it is about 464 km, "
              "a drive of roughly 5.2 h.")
    state, _ = _run(monkeypatch, [GROUNDED], passes=(first, SECOND_PASS), answers=(answer,))
    assert "about 464 km" in state["final_answer"], state["final_answer"]
    assert "no routing tool" not in state["final_answer"]


def _scan(text, *contents, bound=()):
    """Stage 44: the general pass, `_cut_unproducible_figures`, over a record of *contents*, with
    the scripted producer model (test_regrounding_turn_record.scripted_producers)."""
    from agent_runtime import facts

    fs = facts.build(results=[{"name": "execute_code", "tool_call_id": f"c{i}", "content": c}
                              for i, c in enumerate(contents)])
    docs = {t: t for t in bound}
    out, cut, _kept, _res, _ = g._cut_unproducible_figures(
        text, fs, lambda sents: scripted_producers(sents, docs))
    return out, cut


@pytest.mark.parametrize("text", [
    "Population growth and urban expansion are the main driving factors behind 450 km² of loss.",
    "The road network layer holds 3,400 km of roads.",
    "London is 340.0 km from Paris as the crow flies.",
    "The great-circle distance is 340.0 km.",
    "Saved to Google Drive in 12 minutes.",
    "We train the model for 20 minutes.",
])
def test_the_scan_leaves_ordinary_sentences_alone(text):
    out, cut = _scan(text, json.dumps({"stdout": "network length 3400.0 km; d=340.0 km"}))
    assert out == text and cut == [], (text, out)


@pytest.mark.parametrize("text, gone", [
    ("Driving is roughly 450–470 km.", "450"),
    ("It is roughly 490 km by rail and ~450 km by road.", "490"),
    ("The Eurostar takes about 2 hours 15 minutes.", "2 hours"),
    ("A 5-hour drive separates them.", "5-hour"),
    ("Flights take about 1 h 15 min.", "1 h"),
    ("The road distance is about 460 km.", "460"),
    ("Travel time by train is roughly 2.5 hours.", "2.5"),
])
def test_the_scan_cuts_travel_figures_in_no_tool_result(text, gone):
    # small numbers that ARE in the record as counts or coordinates do not ground a time
    record = json.dumps({"count": 2, "feature_count": 1, "n": 5, "lat": 48.84502,
                         "stdout": "340.0 km"})
    out, cut = _scan("London is 340.0 km from Paris. " + text, record)
    assert gone not in out and cut, (text, out)
    assert out.startswith("London is 340.0 km from Paris.")


def test_the_scan_cuts_only_the_parenthetical_segment():
    out, cut = _scan("London is 340.0 km from Paris (about 460 km by road; 343.6 km between "
                     "centres).", json.dumps({"stdout": "340.0 km, 343.6 km"}))
    assert out == "London is 340.0 km from Paris (343.6 km between centres).", out


def test_the_scan_skips_code_blocks():
    text = "Here is the code:\n\n```python\n# a 450 km drive\nprint(1)\n```\n\nDone."
    out, cut = _scan(text, "{}")
    assert out == text and not cut


def test_a_flagged_routing_claim_the_scan_cut_no_longer_warns(monkeypatch, ledger):
    """The auditor paraphrased the claim, so Stage 37 could not place it. Once the scan cut the
    sentence, the issue refers to text the user will not see, and must not leave a caveat."""
    audit = {**ROUTING_AUDIT, "issues": [{"claim": "the driving distance is about 460 km",
                                          "reason": "absent"}]}
    state, _ = _run(monkeypatch, [audit, GROUNDED])
    final = state["final_answer"]
    assert "460" not in final, final
    assert "Grounding check" not in final, final
    assert state["actions"].count("analyze") == 1


def test_the_live_answer_still_reads_well(monkeypatch, ledger):
    state, _ = _run(monkeypatch, [GROUNDED], answers=(ANSWER_1,))
    final = state["final_answer"]
    assert "460" not in final and "Eurostar" not in final, final
    assert "- **340.0 km** (≈ **211 miles**) as the crow flies" in final
    assert "The Champaign County boundary was also placed on your interactive map." in final


# --- 3. a lead-in whose list was cut goes with it ------------------------------------------------

def test_an_orphaned_lead_in_is_removed(monkeypatch, ledger):
    state, _ = _run(monkeypatch, [GROUNDED], answers=(LEAD_IN,))
    final = state["final_answer"]
    assert "Actual travel distances are longer" not in final, final
    assert "490" not in final and "450" not in final
    assert "**London to Paris** is **340.0 km** in a straight line." in final
    assert "interactive map" in final


def test_a_lead_in_keeps_its_surviving_items():
    text = ("Distances:\n- **340.0 km** straight line\n- **By road**: roughly 450 km\n\nDone.")
    out, cut = _scan(text, json.dumps({"stdout": "340.0 km"}))
    assert out.startswith("Distances:\n- **340.0 km** straight line"), out
    assert "450" not in out


def test_an_orphaned_lead_in_after_other_sentences_loses_only_itself():
    text = ("This is the straight-line distance. Actual travel distances are longer:\n"
            "- about 490 km by rail\n- roughly 450 km by road\n\nDone.")
    out, _ = _scan(text, "{}")
    assert out.startswith("This is the straight-line distance."), out
    assert "longer" not in out and "Done." in out


def test_drop_claims_removes_an_orphaned_lead_in_too():
    text = "Actual travel distances are longer:\n- roughly 460 km by road\n\nDone."
    out, dropped = g._drop_claims(text, ["roughly 460 km by road"])
    assert dropped and out == "Done.", out


def test_a_lead_in_that_never_had_a_list_is_kept():
    text = "Here is the result:\n\nThe Eurostar takes about 2 hours 15 minutes. Done."
    out, _ = _scan(text, "{}")
    assert out.startswith("Here is the result:"), out


@pytest.fixture
def ledger(monkeypatch):
    import agent_runtime.session_memory as sm

    rows = []
    monkeypatch.setattr(sm, "get_session_actions", lambda tid: [])
    monkeypatch.setattr(sm, "append_session_actions", lambda tid, new: rows.extend(new))
    return rows


# --- 1b. a peer's summary is not part of the record rule (2) checks ------------------------------

def test_a_number_only_in_the_peers_summary_is_not_in_the_record():
    """Local replay 3, 2026-10-08: rule (2) removed "roughly 490 km by road" and "about 340 km via
    the eurostar rail line", logged as found @analysis_results.summary — the analyze peer's own
    prose, which repeated the memorised figures. No tool result held either number."""
    ctx = _record(("geocode_places", json.dumps({"results": [{"place": "Paris", "lat": 48.853}]})))
    ctx["analysis_results"]["summary"] = ("London–Paris is 343.7 km great-circle; by road it is "
                                          "roughly 490 km.")
    out = g._reconcile_audit_with_artifacts(_flag("roughly 490 km by road"), [],
                                            execution_context=ctx)
    assert out["issues"], out
    # a number a tool result holds is still reconciled when the summary repeats it
    ctx = _record(("execute_code", json.dumps({"ok": True, "stdout": "d = 343.7 km\n"})))
    ctx["analysis_results"]["summary"] = "the distance is 343.7 km"
    out = g._reconcile_audit_with_artifacts(_flag("343.7 km apart"), [], execution_context=ctx)
    assert not out["issues"], out
