"""The decider is told whether the thing asked for is already on the user's map.

Measured live, twice: asked for a DEM, the supervisor ran analyze — which fetched the DEM and
drew it — and then routed to `code`, which fetched the same DEM again and drew a second copy.
In the sweep that second pass cost 266 seconds and 16 execute_code iterations to redo work one
tool call had already done.

Nothing in the decision payload answered "is the deliverable delivered?". `has_analysis` says a
peer RAN; `artifacts_produced` lists images; neither says a layer reached the map. The detector
already existed (`_map_delivered_this_turn`, which asks the delivery boundary per tool result and
requires the tool to have succeeded) — the decider simply never saw it.

Deliberately a SIGNAL plus a rule, not a veto. Plenty of requests legitimately need code after a
successful analyze ("map it, then compute the stats"), and hard-vetoing `code` whenever a layer
exists would break those.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime.supervisor import graph as sg  # noqa: E402


def delivered_analysis():
    """An analyze result whose tool call genuinely put a layer on the map."""
    return {"summary": "Elevation for Urbana.",
            "tool_calls": [{"name": "dem_for_region"}],
            "tool_results": [{"name": "dem_for_region", "ok": True,
                              # The real descriptor shape: single layers live under `map_layer`.
                              "content": {"ok": True, "map_layer": {
                                  "kind": "map_layer", "id": "dem-1", "name": "Elevation",
                                  "render": "raster",
                                  "bounds": [-88.3, 40.0, -88.1, 40.2],
                                  "url": "https://agent.i-guide.io/agent/files/f1/download"}}}]}


def failed_analysis():
    return {"summary": "", "tool_calls": [{"name": "admin_boundary"}],
            "tool_results": [{"name": "admin_boundary", "ok": False,
                              "content": {"ok": False, "error": "no county named 'city'"}}]}


# --- the signal ------------------------------------------------------------------

def test_a_delivered_layer_is_reported():
    d = sg._distill({"query": "Add the DEM layer for Urbana",
                     "analysis_results": delivered_analysis()})
    assert d["map_layer_delivered"] is True


def test_nothing_delivered_reads_false():
    d = sg._distill({"query": "Add the DEM layer for Urbana", "analysis_results": None})
    assert d["map_layer_delivered"] is False


def test_a_failed_tool_did_not_deliver():
    """The distinction that matters: a peer RAN is not a deliverable EXISTS. admin_boundary
    returns ok:false on its error paths, and calling that 'delivered' is how an earlier version
    suppressed its own corrective retry."""
    d = sg._distill({"query": "boundary of Urbana", "analysis_results": failed_analysis()})
    assert d["has_analysis"] is True          # it ran
    assert d["map_layer_delivered"] is False  # it delivered nothing


def test_it_is_distinct_from_the_signals_that_already_existed():
    d = sg._distill({"query": "q", "analysis_results": delivered_analysis()})
    assert d["has_analysis"] is True
    assert d["artifacts_produced"] == []      # no IMAGE artifact, yet a layer is on the map
    assert d["map_layer_delivered"] is True


# --- and the decider is told what it means ----------------------------------------

def test_the_decider_prompt_says_what_to_do_with_it():
    captured = {}

    class Recorder:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return '{"next": "done", "reason": "the layer is already drawn"}'

    sg.default_decide_fn(llm=Recorder())(
        {"query": "Add the DEM layer for Urbana"},
        {"available_actions": ["analyze", "code", "done"], "map_layer_delivered": True})
    prompt = captured["prompt"]
    assert "map_layer_delivered" in prompt
    assert "redoing work a tool has already done" in prompt
    assert "second copy of the same layer" in prompt


def test_code_is_still_offered_after_a_successful_analyze():
    """A signal, not a veto. 'Map it, then compute the statistics' must stay possible."""
    actions = sg._available_actions({"actions": ["analyze"],
                                     "analysis_results": delivered_analysis()})
    assert "code" in actions and "done" in actions


# --- the structural gap the boolean was papering over ------------------------------

def test_the_decider_now_sees_this_turns_ledger():
    """The rows were always kept — they are what the trace renders as
    "dem_for_region(...) -> 1 layer on the map". But `_ledger_lines` had exactly two consumers,
    the answering model and the grounding auditor, and the decider was not one of them. It saw
    counts and flags about the current turn and the ledger only of PREVIOUS turns, so it could
    not tell that the tool it was about to route to had already run.
    """
    state = {"query": "Add the DEM layer for Urbana",
             "analysis_results": delivered_analysis(),
             "action_rows": [{"tool": "dem_for_region", "args": {"size": 512},
                              "facts": {"layers": 1}, "outputs": "area_dem.tif",
                              "file_id": "file_abc"}]}
    lines = sg._distill(state, for_decision=True)["this_turn"]
    assert any("dem_for_region" in l for l in lines)
    assert any("area_dem.tif" in l for l in lines)


def test_a_failed_call_reads_as_not_run_in_that_ledger():
    """Same rule the answerer and auditor follow: a failure must never read as done work."""
    state = {"query": "boundary", "action_rows": [
        {"tool": "admin_boundary", "args": {"area": "city"}, "failed": True,
         "error": "no county named 'city'"}]}
    lines = sg._distill(state, for_decision=True)["this_turn"]
    assert any("FAILED" in l and "DID NOT RUN" in l for l in lines)


def test_the_note_tells_the_decider_what_the_ledger_means():
    note = sg._distill({"query": "q", "action_rows": [{"tool": "x"}]},
                       for_decision=True)["this_turn_note"]
    assert "already done" in note and "second copy" in note


def test_the_client_payload_does_not_carry_it():
    """Decision-only, like prior_turns: the client payload is a per-turn record and the trace
    already renders these rows itself."""
    state = {"query": "q", "action_rows": [{"tool": "dem_for_region"}]}
    assert "this_turn" not in sg._distill(state)
    assert "this_turn" in sg._distill(state, for_decision=True)
