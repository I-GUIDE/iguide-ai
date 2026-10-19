"""The browser reports what is on the user's map, and the agent believes it (Stage 81).

Two live failures on 2026-10-18, both from the agent guessing the map from its own ledger of
layers it had SENT:

1. Both suitability layers failed in the browser (401 token_expired). The next answer said the
   score was "available on your interactive map" beside an empty map.
2. Schools demo, step 2, "Show the site, its 1-mile buffer and the schools inside it on the map":
   after step 1 had drawn a schools-only layer the router's first decision was 'done' in 2 of 3
   runs — the buffer had been computed, and that read as "on the map".
"""
from __future__ import annotations

import pytest

from agent_runtime import map_state as ms
from agent_runtime import session_memory as sm
from agent_runtime.supervisor import graph as g

STEP2 = "Show the site, its 1-mile buffer and the schools inside it on the map."
STEP1 = ("How many schools are within 1 mile of the site at 41.8827 N, 87.6233 W? Which one is "
         "nearest, and how far is it in metres?")

SCHOOLS = {"id": "schools", "label": "Schools within 1 mile", "kind": "geojson",
           "render": "geojson", "features": 20, "visible": True, "source": "analysis"}
SITE = {"id": "site", "label": "Site (41.8827, -87.6233)", "kind": "geojson", "features": 1}
BUFFER = {"id": "buf", "label": "1-mile buffer around site", "kind": "geojson", "features": 1}


@pytest.fixture(autouse=True)
def clean():
    sm.reset_all()
    yield
    sm.reset_all()


def _report(*layers, undrawn=()):
    return ms.normalize({"layers": list(layers), "undrawn": list(undrawn),
                         "viewport": [-87.65, 41.86, -87.6, 41.9]})


# --- the report itself ----------------------------------------------------------------------

def test_no_field_is_none_and_an_empty_map_is_not():
    """An older client and a client with an empty map must stay distinguishable."""
    assert ms.normalize(None) is None
    assert ms.normalize("layers") is None
    assert ms.normalize({}) == {"layers": [], "undrawn": []}


def test_the_report_is_capped_and_says_so():
    raw = {"layers": [{"label": f"layer {i}", "features": i} for i in range(ms.MAX_LAYERS + 7)],
           "undrawn": [{"label": "x", "reason": "HTTP 401"}],
           "truncated": {"layers": 90}}
    out = ms.normalize(raw)
    assert len(out["layers"]) == ms.MAX_LAYERS
    assert out["truncated"] == {"layers": 90, "undrawn": 1}
    assert "capped: 40 of 90 layers" in "\n".join(ms.render_lines(out))


def test_garbage_is_dropped_not_rendered():
    out = ms.normalize({"layers": [{"label": ""}, "x", {"label": "ok", "features": "NaN",
                                                         "opacity": 7, "geometry": [1, 2]}],
                        "viewport": [1, 2, "nan", 4]})
    assert out["layers"] == [{"label": "ok", "opacity": 1.0}]
    assert "viewport" not in out


# --- which items a request asks to see, and which a layer shows -----------------------------

@pytest.mark.parametrize("query,items", [
    (STEP2, ["site", "buffer"]),
    (STEP1, []),                                          # a question, not a request to draw
    ("Show the schools within 1 mile of the site on the map", []),   # the site only locates them
    ("Show on the map the site and its buffer", ["site", "buffer"]),
    ("Add the 2 km buffer to the map", ["buffer"]),
    ("What is the buffer distance?", []),
    ("Is the route on the map?", []),
])
def test_requested_items(query, items):
    assert ms.requested_items(query) == items


@pytest.mark.parametrize("label,items", [
    ("1-mile buffer around site", ["buffer"]),            # the site only locates the buffer
    ("site_1mi_buffer", ["buffer"]),
    ("Site (41.8827, -87.6233)", ["site"]),
    ("Schools within 1 mile of the site", []),
    ("schools_within_1mile", []),
])
def test_layer_items_read_the_head_of_the_label(label, items):
    assert ms.layer_items({"label": label}) == items


# --- the rule, in a whole supervisor turn ---------------------------------------------------

def _turn(map_state, query=STEP2):
    """One supervisor turn whose decider says 'done' at once, as the live router did."""
    ran = {"analyze": 0, "map_missing": None, "notes": []}

    def analyze(q, ev, state):
        ran["analyze"] += 1
        ran["map_missing"] = state.get("map_missing")
        return {"summary": "drew the layers"}

    def synth(q, ev, ar, cr, ch=None, note=None):
        ran["notes"].append(note)
        return "answer"

    graph = g.build_supervisor_graph(
        search_fn=lambda q, s: [], analyze_fn=analyze, synthesize_fn=synth,
        decide_fn=lambda *a: g.Decision("done"), do_rerank=False, do_audit=False)
    state = {"query": query, "thread_id": "t-map", "chat_history": [
        {"role": "user", "content": STEP1}, {"role": "assistant", "content": "20 schools"}],
        "evidence": [], "actions": []}
    if map_state is not None:
        state["map_state"] = map_state
    out = graph.invoke(state)
    return ran, out


def test_schools_only_on_the_map_is_not_done():
    ran, out = _turn(_report(SCHOOLS))
    assert ran["analyze"] == 1, "the site and buffer are not on the map: the peer must draw them"
    assert ran["map_missing"] == ["site", "buffer"]
    assert out["actions"][0] == "analyze"


def test_all_three_on_the_map_is_done():
    ran, out = _turn(_report(SCHOOLS, SITE, BUFFER))
    assert ran["analyze"] == 0
    assert out["actions"] == ["done"]


def test_without_a_report_nothing_changes():
    ran, out = _turn(None)
    assert ran["analyze"] == 0 and out["actions"] == ["done"]


def test_a_question_is_left_to_the_router():
    ran, out = _turn(_report(SCHOOLS), query=STEP1)
    assert ran["analyze"] == 0


def test_the_peer_is_told_what_to_draw():
    text = ms.draw_directive(["site", "buffer"])
    assert "site, buffer" in text and "NOT ON IT" in text


# --- a layer that failed to load is not on the map ------------------------------------------

SUIT_ROWS = [{"tool": "suitability_score", "args": {"area": "Champaign"},
              "facts": {"cells": 10385}, "map_layer": ["Suitability score", "Best cell"]}]


def test_a_failed_layer_reads_as_not_on_the_map_for_the_writer():
    sm.append_session_actions("t-map", SUIT_ROWS)
    report = _report(undrawn=[{"label": "Suitability score", "reason": "HTTP 401, token_expired"},
                              {"label": "Best cell", "reason": "HTTP 401, token_expired"}])
    ran, _ = _turn(report, query="Where is the best cell?")
    note = ran["notes"][-1]
    assert ms.MAP_STATE_HEADING in note
    assert "on the map as" not in note, "the ledger must not say a failed layer is on the map"
    assert "still on the user's map" not in note
    assert "'Suitability score' failed to load in the browser (HTTP 401, token_expired)" in note
    assert "the map has no layers on it" in note


def test_a_failed_layer_loses_the_auditors_map_amnesty():
    """`_map_delivered_earlier` reads the same rows: with the layer gone from the browser, a
    claim that it is on the map stays flagged."""
    rows = ms.reconcile_rows(SUIT_ROWS, _report(undrawn=[{"label": "Suitability score",
                                                          "reason": "HTTP 401"}]))
    assert not g._map_delivered_earlier(rows)
    assert "map_layer" not in rows[0]
    assert rows[0]["map_layer_absent"] == [
        {"label": "Suitability score", "why": "failed to load: HTTP 401"},
        {"label": "Best cell", "why": "not in the browser"}]
    assert "NOT on it" in g._ledger_line(rows[0])
    assert SUIT_ROWS[0]["map_layer"] == ["Suitability score", "Best cell"], "ledger untouched"


def test_a_layer_still_on_the_map_keeps_its_row():
    rows = ms.reconcile_rows(SUIT_ROWS, _report({"label": "Suitability score", "features": 10385}))
    assert rows[0]["map_layer"] == ["Suitability score"]
    assert g._map_delivered_earlier(rows)


def test_the_router_sees_the_report():
    sm.append_session_actions("t-map", SUIT_ROWS)
    state = {"query": "q", "thread_id": "t-map", "evidence": [], "actions": [],
             "map_state": _report(SCHOOLS)}
    payload = g._distill(state, for_decision=True)
    lines = "\n".join(payload["user_map_now"])
    assert "'Schools within 1 mile' (geojson, 20 features, analysis)" in lines
    assert "'Suitability score'" in lines and "NOT on the map" in lines
    assert "overrides prior_turns_in_this_conversation" in payload["user_map_now_note"]
    # and the prior-turn rows it is shown no longer claim the layer
    assert not any(r.get("map_layer") for r in payload["prior_turns_in_this_conversation"])


def test_without_a_report_the_note_is_byte_identical():
    sm.append_session_actions("t-map", SUIT_ROWS)
    state = {"query": "q", "thread_id": "t-map"}
    assert g._ledger_note_for(state) == g._prior_actions_note(SUIT_ROWS)
    assert "user_map_now" not in g._distill({**state, "evidence": [], "actions": []},
                                            for_decision=True)


def test_an_empty_ledger_still_carries_the_report():
    note = g._prior_actions_note([], _report(SCHOOLS))
    assert note.startswith(ms.MAP_STATE_HEADING)
    assert g._prior_actions_note([]) is None


# --- the request field reaches the supervisor -----------------------------------------------

def test_server_accepts_map_state_and_the_service_carries_it(monkeypatch):
    import api.server as server
    from agent_runtime import agent_chat_service as svc

    norm = server._normalize_agent_chat_request(
        {"user_input": "hi", "map_state": {"layers": [SCHOOLS]}})
    assert norm["map_state"] == {"layers": [SCHOOLS]}
    assert server._normalize_agent_chat_request({"user_input": "hi"})["map_state"] is None

    seen = {}

    def fake_run(query, **kw):
        seen.update(kw)
        return {"final_answer": "ok"}

    monkeypatch.setattr(svc, "run_agent_query", fake_run)
    svc.run_agent_chat(user_input="hi", use_persistent_memory=False,
                       map_state={"layers": [SCHOOLS]})
    assert seen["map_state"]["layers"][0]["label"] == "Schools within 1 mile"
    svc.run_agent_chat(user_input="hi", use_persistent_memory=False)
    assert seen["map_state"] is None


def test_run_supervisor_puts_it_in_the_state_only_when_sent(monkeypatch):
    seen = []

    class _G:
        def invoke(self, initial, config):
            seen.append(dict(initial))
            return {}

    monkeypatch.setattr(g, "build_supervisor_graph", lambda **kw: _G())
    g.run_supervisor("q", map_state={"layers": []})
    g.run_supervisor("q")
    assert seen[0]["map_state"] == {"layers": []}
    assert "map_state" not in seen[1]
