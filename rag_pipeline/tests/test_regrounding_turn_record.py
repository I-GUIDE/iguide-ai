"""A re-grounding pass must not cost the turn its record, and must not chase claims no tool makes.

Live, 2026-10-08 19:12 UTC (thread sess-1e8e3edd-…, Lumen deepseek-v4-flash): "what is the area
of Champaign County, Illinois, and how far is London from Paris?" was answered correctly —
2,584.6 km² and 340.0 km — after admin_boundary, geocode_places and two execute_code calls. The
synthesizer added "roughly 460 km by road or ~340 km by the Eurostar rail line" from memory; the
audit flagged both; the re-grounding pass re-ran analyze, which ran ONE execute_code. Then:

* `analysis_results` held only that one call, so the auditor was shown no admin_boundary result
  and flagged "GEOID 17019" as unsupported, and the ledger recorded 1 row instead of 4;
* the re-run could never succeed — there is no routing tool — and cost ~28 s of a 72 s turn;
* the final answer restated the road/rail figures in a "What I could NOT establish" section and
  called them "the earlier rejected answer";
* `square_miles` was an unrecognised unit, so the banner read "COULD NOT VERIFY".

Fully stubbed, like test_supervisor_graph.py: no network, no keys.
"""

from __future__ import annotations

import json

import pytest

from agent_runtime.supervisor_graph import run_supervisor

QUERY = "what is the area of Champaign County, Illinois, and how far is London from Paris?"

_ADMIN = json.dumps({"ok": True, "level": "county",
                     "matched": [{"geoid": "17019", "name": "Champaign County"}],
                     "feature_count": 1, "file_id": "file_2272c8426ec9",
                     "geoids": ["17019"], "source": "US Census TIGERweb"})
_GEOCODE = json.dumps({"results": [
    {"place": "London", "found": True, "lat": 51.489317, "lon": -0.08818},
    {"place": "Paris", "found": True, "lat": 48.858866, "lon": 2.346941}],
    "not_found": [], "count": 2})
_EXEC_FAIL = json.dumps({"ok": False, "exit_code": 1, "stdout": "",
                         "stderr": "DataSourceError: file_2272c8426ec9: No such file"})
_EXEC_OK = json.dumps({"ok": True, "exit_code": 0,
                       "stdout": "Champaign County area: 2584.62 km2 = 997.93 mi2\n"
                                 "London to Paris: 340.0 km = 211.3 mi\n"})


def _call(name, cid, args=None):
    return {"name": name, "args": args or {}, "id": cid}


def _result(name, cid, content):
    return {"name": name, "tool_call_id": cid, "content": content}


FIRST_PASS = {
    "summary": "Both computations completed successfully.",
    "tool_calls": [_call("admin_boundary", "c1", {"area": "Champaign County"}),
                   _call("geocode_places", "c2", {"places": ["London", "Paris"]}),
                   _call("execute_code", "c3"), _call("execute_code", "c4")],
    "tool_results": [_result("admin_boundary", "c1", _ADMIN),
                     _result("geocode_places", "c2", _GEOCODE),
                     _result("execute_code", "c3", _EXEC_FAIL),
                     _result("execute_code", "c4", _EXEC_OK)],
}
SECOND_PASS = {
    "summary": "Both values are computed and grounded in tool results.",
    "tool_calls": [_call("execute_code", "c5")],
    "tool_results": [_result("execute_code", "c5", _EXEC_OK)],
}

# The live first answer, verbatim in its structure.
ANSWER_1 = (
    "## Champaign County, Illinois — Area\n"
    "Using the US Census TIGER/Line boundary (GEOID 17019), computed in EPSG:26916:\n"
    "- **2,584.6 km²** (≈ **998 square miles**)\n\n"
    "## London to Paris — Distance\n"
    "Computed as the great-circle (haversine) distance between the geocoded city centers:\n"
    "- **340.0 km** (≈ **211 miles**) as the crow flies\n\n"
    "Note: this is the straight-line distance between the two city centers. Actual travel "
    "distance is longer — roughly 460 km by road or ~340 km by the Eurostar rail line (which "
    "follows a more direct route than driving). The Champaign County boundary was also placed "
    "on your interactive map.")

ROUTING_AUDIT = {
    "verdict": "partially_supported", "severity": "high", "hallucination_detected": True,
    "summary": "The road and rail figures are not in any tool result.",
    "issues": [{"claim": "roughly 460 km by road",
                "reason": "no span in the evidence or execution record covers this claim"},
               {"claim": "~340 km by the Eurostar rail line",
                "reason": "no span in the evidence or execution record covers this claim"}]}
COUNTIES_AUDIT = {
    "verdict": "partially_supported", "severity": "high", "hallucination_detected": True,
    "summary": "The neighbouring counties are not in any tool result.",
    "issues": [{"claim": "Champaign County borders six counties",
                "reason": "absent - no tool result lists neighbouring counties"}]}
GROUNDED = {"verdict": "supported", "severity": "none", "summary": "grounded", "issues": []}


def _fake_llm(prompt: str) -> str:
    return "x"


def _scripted(seq):
    box = {"i": 0}

    def decide(state, distilled):
        i = box["i"]
        box["i"] += 1
        return seq[i] if i < len(seq) else "done"

    return decide


@pytest.fixture
def ledger(monkeypatch):
    """Capture what the turn writes to the thread ledger, and start the thread empty."""
    import agent_runtime.session_memory as sm

    rows = []
    monkeypatch.setattr(sm, "get_session_actions", lambda tid: [])
    monkeypatch.setattr(sm, "append_session_actions", lambda tid, new: rows.extend(new))
    return rows


def _run(monkeypatch, audits, *, passes=(FIRST_PASS, SECOND_PASS), answers=(ANSWER_1,),
         code_peer=False, decisions=None):
    """Drive the real graph with scripted audits, peer passes and synthesized answers."""
    from agent_runtime.supervisor import graph as g

    seen = {"audits": [], "analyze": [], "answers": 0}

    def fake_audit(q, answer, evidence, *, llm=None, execution_context=None):
        seen["audits"].append({"answer": answer, "execution_context": execution_context})
        i = len(seen["audits"]) - 1
        return json.loads(json.dumps(audits[min(i, len(audits) - 1)]))

    monkeypatch.setattr(g, "audit_answer_grounding", fake_audit)

    # Stage 42: whether a bound tool produces a flagged claim is the producer check's answer (a
    # model call in production). Scripted here as a model that knows only routing needs a
    # routing tool; the graph's reaction to that answer is what these tests hold.
    def fake_producers(claims, tool_docs, llm):
        routing = any(g._ROUTING_TOOL_RE.search(t) for t in tool_docs)
        return {c: (None if g._ROUTING_CLAIM_RE.search(c) and not routing else "execute_code")
                for c in claims}

    monkeypatch.setattr(g, "_producing_tools", fake_producers)

    def peer(q, ev, st):
        seen["analyze"].append(list(st.get("grounding_gaps") or []))
        return json.loads(json.dumps(passes[min(len(seen["analyze"]) - 1, len(passes) - 1)]))

    def synthesize(q, ev, ar, cr, ch, pa=None):
        i = seen["answers"]
        seen["answers"] += 1
        return answers[min(i, len(answers) - 1)]

    kw = {"code_fn": peer} if code_peer else {"analyze_fn": peer}
    state = run_supervisor(
        QUERY, llm=_fake_llm, thread_id="sess-test",
        decide_fn=_scripted(decisions or ["code" if code_peer else "analyze", "done", "done",
                                          "done"]),
        search_fn=lambda q, s: [], synthesize_fn=synthesize, do_rerank=False, **kw)
    return state, seen


# --- 1. the re-grounding pass keeps the turn's earlier tool results ----------------------------

def test_a_second_analyze_pass_adds_to_the_turn_record_instead_of_replacing_it(monkeypatch, ledger):
    state, seen = _run(monkeypatch, [COUNTIES_AUDIT, GROUNDED])
    assert state["actions"].count("analyze") == 2, state["actions"]
    names = [r["name"] for r in state["analysis_results"]["tool_results"]]
    assert names == ["admin_boundary", "geocode_places", "execute_code", "execute_code",
                     "execute_code"], names
    # the newest pass still speaks for the peer
    assert state["analysis_results"]["summary"] == SECOND_PASS["summary"]


def test_the_audit_after_a_re_run_sees_the_first_pass(monkeypatch, ledger):
    """The GEOID false positive: admin_boundary's 17019 must be in the record the auditor reads."""
    state, seen = _run(monkeypatch, [COUNTIES_AUDIT, GROUNDED])
    assert len(seen["audits"]) == 2
    second = json.dumps(seen["audits"][1]["execution_context"], default=str)
    assert "17019" in second
    assert "admin_boundary" in second and "geocode_places" in second


def test_the_ledger_records_every_run_of_the_turn(monkeypatch, ledger):
    _run(monkeypatch, [COUNTIES_AUDIT, GROUNDED])
    tools = sorted({str(r.get("tool")) for r in ledger})
    assert {"admin_boundary", "geocode_places", "execute_code"} <= set(tools), tools
    assert len(ledger) >= 3, ledger


def test_a_peer_that_replays_its_earlier_calls_is_not_double_counted(monkeypatch, ledger):
    """PeerSession falls back to the whole thread when its prefix guard fails, so a second pass
    can hand back the first pass's calls again. Same id = same call."""
    replay = {"summary": "again",
              "tool_calls": [*FIRST_PASS["tool_calls"], *SECOND_PASS["tool_calls"]],
              "tool_results": [*FIRST_PASS["tool_results"], *SECOND_PASS["tool_results"]]}
    state, _ = _run(monkeypatch, [COUNTIES_AUDIT, GROUNDED], passes=(FIRST_PASS, replay))
    ids = [r["tool_call_id"] for r in state["analysis_results"]["tool_results"]]
    assert ids == ["c1", "c2", "c3", "c4", "c5"], ids


def test_the_code_peer_keeps_its_earlier_runs_too(monkeypatch, ledger):
    state, _ = _run(monkeypatch, [GROUNDED], code_peer=True,
                    decisions=["code", "search", "code", "done"])
    assert state["actions"].count("code") == 2, state["actions"]
    ids = [r["tool_call_id"] for r in state["code_result"]["tool_results"]]
    assert ids == ["c1", "c2", "c3", "c4", "c5"], ids


def test_merge_keeps_rows_without_ids_and_drops_a_stale_error():
    from agent_runtime.supervisor import graph as g

    a = {"summary": "one", "tool_results": [{"name": "execute_code", "content": "{}"}],
         "error": "boom", "on_map": True}
    b = {"summary": "two", "tool_results": [{"name": "execute_code", "content": "{}"}]}
    merged = g._merge_peer_result(a, b)
    assert len(merged["tool_results"]) == 2          # no id: two calls, however alike
    assert "error" not in merged                     # the latest pass did not fail
    assert merged["on_map"] is True and merged["summary"] == "two"
    assert g._merge_peer_result(None, b) == b


# --- 3. claims no bound tool can produce are dropped, not re-run --------------------------------

def test_routing_claims_are_dropped_and_nothing_re_runs(monkeypatch, ledger):
    state, seen = _run(monkeypatch, [ROUTING_AUDIT, GROUNDED])
    assert state["actions"].count("analyze") == 1, state["actions"]
    assert not state.get("grounding_retries")
    final = state["final_answer"]
    assert "460" not in final and "Eurostar" not in final, final
    assert "340.0 km" in final and "2,584.6 km²" in final
    assert "straight-line distance" in final and "interactive map" in final
    # no warning: every flagged claim is gone from the answer
    assert "Grounding check" not in final and "COULD NOT VERIFY" not in final, final
    assert not state["audit"].get("issues"), state["audit"]


def test_a_routing_claim_with_a_routing_tool_bound_still_re_runs(monkeypatch, ledger):
    """The rule is "no BOUND tool can produce it", not "routing is never computable"."""
    first = {**FIRST_PASS, "bound_tools": ["execute_code", "network_route_distance"],
             "bound_tool_docs": {"execute_code": "run code",
                                 "network_route_distance": "road distance between points"}}
    state, seen = _run(monkeypatch, [ROUTING_AUDIT, GROUNDED], passes=(first, SECOND_PASS))
    assert state["actions"].count("analyze") == 2, state["actions"]
    assert "bound_tools" not in state["analysis_results"]


def test_only_producible_claims_are_sent_back(monkeypatch, ledger):
    mixed = {**ROUTING_AUDIT, "issues": [*ROUTING_AUDIT["issues"], *COUNTIES_AUDIT["issues"]]}
    answer = ANSWER_1 + " Champaign County borders six counties."
    state, seen = _run(monkeypatch, [mixed, GROUNDED], answers=(answer, ANSWER_1))
    assert state["actions"].count("analyze") == 2
    assert seen["analyze"][1] == ["Champaign County borders six counties"], seen["analyze"]


def test_an_unproducible_claim_that_cannot_be_located_is_not_re_run(monkeypatch, ledger):
    """Paraphrased by the auditor, so it cannot be cut out — the caveat stays, the re-run does not
    happen, because no re-run can produce a driving route.

    The claim carries no figure: since Stage 38 the answer scan cuts a travel sentence whose
    figure is in no tool result, and drops a flagged claim whose figures it cut (see
    test_routing_figures_scan.py), so a paraphrased "about 460 km" no longer reaches here."""
    audit = {**ROUTING_AUDIT, "issues": [{"claim": "the driving route goes via the A26",
                                          "reason": "absent"}]}
    answer = ANSWER_1 + " The quickest drive follows the A26 motorway."
    state, _ = _run(monkeypatch, [audit, GROUNDED], answers=(answer,))
    assert state["actions"].count("analyze") == 1
    assert "Grounding check" in state["final_answer"]


@pytest.mark.parametrize("claim", ["roughly 460 km by road", "~340 km by the Eurostar rail line",
                                   "a 5 h drive", "about 2 hours by train", "the driving route",
                                   "a travel time of 45 minutes", "a 3 hour flight"])
def test_routing_claims_are_recognised(claim):
    from agent_runtime.supervisor import graph as g
    assert g._unproducible_capability(claim, []) == "routing", claim


@pytest.mark.parametrize("claim", ["340.0 km as the crow flies", "GEOID 17019",
                                   "the great-circle distance", "2,584.6 km²",
                                   "the road network layer has 1,204 segments"])
def test_ordinary_claims_are_not_routing(claim):
    from agent_runtime.supervisor import graph as g
    assert g._unproducible_capability(claim, []) is None, claim


# --- dropping a claim from the text -------------------------------------------------------------

def test_drop_claims_removes_the_sentence_and_keeps_its_neighbours():
    from agent_runtime.supervisor import graph as g
    out, dropped = g._drop_claims(ANSWER_1, ["roughly 460 km by road",
                                             "~340 km by the Eurostar rail line"])
    assert dropped == ["roughly 460 km by road", "~340 km by the Eurostar rail line"]
    assert "460" not in out and "Eurostar" not in out
    assert "Note: this is the straight-line distance between the two city centers." in out
    assert "The Champaign County boundary was also placed on your interactive map." in out
    assert "- **340.0 km** (≈ **211 miles**) as the crow flies" in out


def test_drop_claims_removes_one_parenthetical_segment():
    from agent_runtime.supervisor import graph as g
    text = "It is 340 km apart (roughly 460 km by road; 343.6 km between centres). Done."
    out, dropped = g._drop_claims(text, ["roughly 460 km by road"])
    assert out == "It is 340 km apart (343.6 km between centres). Done."
    out, _ = g._drop_claims("It is 340 km apart (roughly 460 km by road). Done.",
                            ["roughly 460 km by road"])
    assert out == "It is 340 km apart. Done."


def test_drop_claims_removes_a_list_item_and_an_emptied_section():
    from agent_runtime.supervisor import graph as g
    text = ("## Distance\n- **340.0 km** straight line\n\n## Travel\n"
            "- **~460 km** by road\n\n## Map\nThe boundary is on your map.")
    out, dropped = g._drop_claims(text, ["~460 km by road"])
    assert dropped == ["~460 km by road"]
    assert "460" not in out and "## Travel" not in out
    assert "## Distance\n- **340.0 km** straight line" in out and "## Map" in out


def test_drop_claims_leaves_text_alone_when_the_claim_is_not_there():
    from agent_runtime.supervisor import graph as g
    out, dropped = g._drop_claims(ANSWER_1, ["the driving distance is about 460 km"])
    assert out == ANSWER_1 and dropped == []
    out, dropped = g._drop_claims("e.g. one. Two.", ["x"])     # too short to locate safely
    assert dropped == []


# --- 4. the final answer does not restate removed claims or mention a rejected answer ----------

ANSWER_2 = (
    "## Champaign County, Illinois — Area\n- **2,584.6 km²** (GEOID 17019)\n\n"
    "## London to Paris — Distance\n- **340.0 km** as the crow flies\n\n"
    "## What I could NOT establish\n"
    "I did **not** compute the road distance (~460 km) or the Eurostar rail distance (~340 km) — "
    "those figures were in the earlier rejected answer but were not grounded in any tool result. "
    "I am therefore **not** restating them.")

ROUTING_AUDIT_2 = {**ROUTING_AUDIT, "issues": [
    {"claim": "road distance (~460 km)", "reason": "no span covers this claim"},
    {"claim": "Eurostar rail distance (~340 km)", "reason": "no span covers this claim"}]}


def test_after_a_re_run_the_answer_does_not_restate_removed_claims(monkeypatch, ledger):
    answer_1 = ANSWER_1 + " Champaign County borders six counties."
    state, _ = _run(monkeypatch, [COUNTIES_AUDIT, ROUTING_AUDIT_2], answers=(answer_1, ANSWER_2))
    final = state["final_answer"]
    assert "460" not in final, final
    assert "rejected" not in final.lower(), final
    assert "2,584.6 km²" in final and "340.0 km" in final
    # Known limit: the cut is per sentence, so a follow-on like "I am therefore not restating
    # them." can stay behind. The directive no longer asks for this section at all.


def test_a_mention_of_the_rejected_draft_is_removed_after_a_re_run(monkeypatch, ledger):
    answer_2 = ("- **340.0 km** as the crow flies.\n\nThe earlier rejected answer included "
                "figures I could not verify. The boundary is on your map.")
    answer_1 = ANSWER_1 + " Champaign County borders six counties."
    state, _ = _run(monkeypatch, [COUNTIES_AUDIT, GROUNDED], answers=(answer_1, answer_2))
    final = state["final_answer"]
    assert "rejected" not in final.lower(), final
    assert "The boundary is on your map." in final and "340.0 km" in final


def test_the_directive_does_not_invite_restating_or_mentioning_a_draft():
    from agent_runtime.supervisor import graph as g
    note = g._reground_note({"grounding_gaps": ["roughly 460 km by road"]})
    assert "rejected" not in note.lower()
    assert "not even to disown them" in note
    assert "never saw" in note


# --- 2. the units gate reads the units a run actually writes ------------------------------------

def _unit_findings(outputs):
    """Stage 43: units are judged agent-side with a unit library, from the sandbox's report."""
    from agent_runtime import declared_outputs
    from agent_runtime.sandbox_verify import DECLARED_OUTPUTS, run_checks

    rep = run_checks({DECLARED_OUTPUTS: outputs})
    extra, _ = declared_outputs.evaluate(rep)
    return [f for f in [*rep["findings"], *extra]
            if f["check"] in ("declared_units", "finite_value", "measured_in")]


@pytest.mark.parametrize("unit", ["square_miles", "square miles", "mi2", "mi²", "sq mi", "km2",
                                  "acres", "hectares", "m", "km", "mi", "miles", "mile", "ft",
                                  "sq ft", "square_feet", "ft2", "yards", "nautical miles"])
def test_common_area_and_length_units_are_recognised(unit):
    from agent_runtime.sandbox_verify import PASS
    out = _unit_findings({"v": {"value": 997.93, "unit": unit}})
    statuses = {f["check"]: f["status"] for f in out}
    assert statuses.get("declared_units") == PASS, (unit, out)


def test_the_live_runs_declared_outputs_all_pass():
    from agent_runtime.sandbox_verify import PASS
    for outputs in (
            {"champaign_area_km2": {"value": 2584.62, "unit": "km2"},
             "champaign_area_mi2": {"value": 997.93, "unit": "mi2"},
             "london_paris_km": {"value": 340.0, "unit": "km"},
             "london_paris_mi": {"value": 211.3, "unit": "mi"}},
            {"champaign_area_km2": {"value": 2584.62, "unit": "km2"},
             "champaign_area_mi2": {"value": 997.93, "unit": "square_miles"},
             "london_paris_km": {"value": 340.0, "unit": "km"},
             "london_paris_mi": {"value": 211.3, "unit": "miles"}}):
        bad = [f for f in _unit_findings(outputs) if f["status"] != PASS]
        assert not bad, bad


def test_a_unit_that_does_not_parse_is_not_quietly_passed():
    """Stage 43: units parse with a unit library, so `furlongs2`, `square_miles` and `schools`
    no longer reach here (test_invariant_gate.py holds that). What does is a unit nobody can
    read, and that is a number nobody can check: it says COULD NOT VERIFY, without calling it
    a hallucination."""
    from agent_runtime.supervisor import graph as g
    report = {"verdict": "cannot_determine", "counts": {"pass": 6, "cannot_determine": 1},
              "findings": [{"check": "declared_units", "status": "cannot_determine",
                            "target": "rate", "unit": "km/hr^^",
                            "message": "unit 'km/hr^^' does not parse as a unit; the number "
                                       "cannot be checked"}]}
    ctx = {"analysis_results": {"tool_results": [_result(
        "execute_code", "c9", json.dumps({"ok": True, "verification": report}))]}}
    audit = g._reconcile_audit_with_artifacts(GROUNDED, [], execution_context=ctx)
    assert audit["invariant_gate"] == "cannot_determine"
    note = g._apply_grounding_caveat("The rate is 12.", audit)
    assert "COULD NOT VERIFY" in note and "km/hr^^" in note
    assert "hallucination" not in note.lower()


def test_a_gate_unknown_does_not_disable_reconciling_a_number_another_tool_returned():
    """GEOID 17019 came from admin_boundary, not from code the gate checks. An unknown about a
    computed value says nothing about it."""
    from agent_runtime.supervisor import graph as g
    report = {"verdict": "cannot_determine", "counts": {"pass": 6, "cannot_determine": 1},
              "findings": [{"check": "coverage", "status": "cannot_determine",
                            "target": "this run", "message": "no frame at module scope"}]}
    ctx = {"analysis_results": {"tool_results": [
        _result("admin_boundary", "c1", _ADMIN),
        _result("execute_code", "c4", json.dumps({"ok": True, "stdout": "area 2584.62",
                                                  "verification": report}))]}}
    flagged = {"severity": "high", "hallucination_detected": True, "summary": "s",
               "issues": [{"claim": "GEOID 17019", "reason": "no span covers this claim"},
                          {"claim": "area 2584.62 km2", "reason": "no span covers this claim"}]}
    audit = g._reconcile_audit_with_artifacts(flagged, [], execution_context=ctx)
    claims = [i.get("claim") for i in audit["issues"]]
    assert "GEOID 17019" not in claims, claims
    # a number only the gated code produced is still not amnestied by the record
    assert "area 2584.62 km2" in claims, claims


def test_the_analyze_peer_reports_what_it_had_bound(monkeypatch):
    """The routing rule asks what the peer COULD call, so the real peer must say."""
    import agent_runtime.executor_factory as ef
    from agent_runtime.supervisor.graph import default_analyze_fn

    monkeypatch.setattr(ef, "build_agent_executor", lambda **kw: object())
    monkeypatch.setattr(ef, "invoke_agent_with_payload_fallback",
                        lambda *a, **k: {"messages": []})
    out = default_analyze_fn(include_mcp_tools=False, code_exec=True)(
        "area of Champaign County", [], {"thread_id": "t"})
    assert "execute_code" in out["bound_tools"] and "geocode_places" in out["bound_tools"]
    assert not any("rout" in t for t in out["bound_tools"])


def test_any_other_gate_unknown_still_says_could_not_verify():
    """Guard: only the unit-name case is quieted. A frame the gate could not inspect is a real
    unknown about the numbers."""
    from agent_runtime.supervisor import graph as g
    report = {"verdict": "cannot_determine", "counts": {"pass": 3, "cannot_determine": 2},
              "findings": [{"check": "declared_units", "status": "cannot_determine",
                            "target": "a", "message": "unrecognised unit 'furlongs'; not checked"},
                           {"check": "coverage", "status": "cannot_determine",
                            "target": "this run", "message": "no frame at module scope"}]}
    ctx = {"analysis_results": {"tool_results": [_result(
        "execute_code", "c9", json.dumps({"ok": True, "verification": report}))]}}
    audit = g._reconcile_audit_with_artifacts(GROUNDED, [], execution_context=ctx)
    assert audit["hallucination_detected"] is True
    assert "COULD NOT VERIFY" in g._apply_grounding_caveat("x", audit)


def test_an_advisory_gate_unknown_is_a_note_not_could_not_verify():
    """A finding the gate marks advisory is "left unchecked", not a reason to say COULD NOT
    VERIFY. (Stage 43's gate no longer emits the two advisory classes this was written for; the
    flag stays the general mechanism.)"""
    from agent_runtime.supervisor import graph as g
    report = {"verdict": "cannot_determine", "counts": {"pass": 9, "cannot_determine": 2},
              "findings": [
                  {"check": "projected_crs", "status": "cannot_determine",
                   "target": "schools_within_4326", "advisory": True,
                   "message": "EPSG:4326 frame holds a measurement column ('distance_m') …"},
                  {"check": "coverage", "status": "cannot_determine", "target": "x",
                   "advisory": True, "message": "left unchecked"}]}
    ctx = {"analysis_results": {"tool_results": [_result(
        "execute_code", "c9", json.dumps({"ok": True, "verification": report}))]}}
    audit = g._reconcile_audit_with_artifacts(GROUNDED, [], execution_context=ctx)
    assert audit["hallucination_detected"] is False and audit["severity"] == "low"
    note = g._apply_grounding_caveat("18 schools.", audit)
    assert note.count("ℹ️") == 1 and "COULD NOT VERIFY" not in note
    assert "distance_m" in note
