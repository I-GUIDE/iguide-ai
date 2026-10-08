"""Stage 42: one append-only record per turn, progress instead of counts, a plan in state.

docs/design-review-2026-10.md has the reasons (flaws 3, 4 and 5). Each test here holds a
property, not an incident: they are written so that the class of failure is closed whatever
the query or the model.
"""
from __future__ import annotations

import json

import pytest

from agent_runtime import turn_log
from agent_runtime.supervisor import graph as g
from agent_runtime.supervisor.graph import run_supervisor


def _fake_llm(prompt):  # noqa: ANN001 - the graph only needs a callable here
    return "ok"


def _scripted(seq):
    it = iter(seq)

    def decide(state, distilled):
        try:
            return next(it)
        except StopIteration:
            return "done"
    return decide


def _run(**kw):
    kw.setdefault("search_fn", lambda q, s: [])
    kw.setdefault("synthesize_fn", lambda *a, **k: "final")
    return run_supervisor("compute the thing", llm=_fake_llm, thread_id="sess-log",
                          do_rerank=False, do_audit=False, **kw)


# --------------------------------------------------------------------------- progress


def test_an_alternation_stops_when_a_peer_stops_adding():
    """analyze, code, analyze, code ... never repeats back to back, so the adjacency guard
    let it run to the per-peer cap. A peer whose last run added nothing is not run again."""
    runs = {"analyze": 0, "code": 0}

    def analyze(q, ev, st):
        runs["analyze"] += 1
        return {"summary": "the same summary"}           # new once, then nothing new

    def code(q, ev, st):
        runs["code"] += 1
        return {"answer": f"code answer {runs['code']}"}

    state = _run(decide_fn=_scripted(["analyze", "code", "analyze", "code", "analyze", "done"]),
                 analyze_fn=analyze, code_fn=code)
    assert runs["analyze"] == 2, state["actions"]


def test_a_queued_need_does_not_bypass_the_progress_rule():
    """A peer's request used to be fulfilled before the decider was consulted and outside the
    repeat guard. A need for a peer that just added nothing is dropped."""
    runs = {"analyze": 0, "code": 0}

    def analyze(q, ev, st):
        runs["analyze"] += 1
        return {"summary": ""}                            # adds nothing, ever

    def code(q, ev, st):
        runs["code"] += 1
        return {"answer": "", "needs": [{"capability": "analyze", "reason": f"need {runs['code']}"}]}

    _run(decide_fn=_scripted(["analyze", "code", "code", "done"]),
         analyze_fn=analyze, code_fn=code)
    assert runs["analyze"] == 1


def test_a_request_for_what_is_missing_counts_as_progress():
    """Asking for a capability says what blocks the peer: the needs-driven re-run still runs."""
    calls = {"code": 0}

    def code(q, ev, st):
        calls["code"] += 1
        if calls["code"] == 1:
            return {"needs": ["search"]}
        return {"answer": "done-code"}

    state = _run(decide_fn=lambda s, d: "code" if not s.get("actions") else "done",
                 code_fn=code, search_fn=lambda q, s: [{"title": "doc", "url": "u"}])
    assert calls["code"] == 2, state["actions"]


# --------------------------------------------------------------------------- one record


def test_tool_records_come_from_the_turn_log():
    """A call the middleware recorded is in the analyze slot even when the peer's own return
    value omits it: the log, not the peer's report, is the record."""

    def analyze(q, ev, st):
        log, run = turn_log.active(), turn_log.active_run()
        log.record_call(peer="analyze", run=run, name="admin_boundary",
                        args={"area": "Champaign"}, call_id="c1")
        log.record_result(peer="analyze", run=run, name="admin_boundary",
                          args={"area": "Champaign"}, call_id="c1",
                          content=json.dumps({"ok": True, "geoid": "17019"}))
        return {"summary": "fetched the county"}          # reports no tool records

    state = _run(decide_fn=_scripted(["analyze", "done"]), analyze_fn=analyze)
    names = [r["name"] for r in state["analysis_results"]["tool_results"]]
    assert names == ["admin_boundary"]
    assert state["turn_log_summary"]["new_results"] >= 1


def test_the_ledger_is_written_when_synthesis_fails():
    """It used to be written only at the end of a successful synthesis, so a turn that could
    not compose its answer left no record of what it had done."""
    from agent_runtime import session_memory

    thread = "sess-ledger-on-failure"

    def analyze(q, ev, st):
        return {"summary": "s", "tool_calls": [{"name": "admin_boundary", "args": {"area": "X"},
                                                "id": "c9"}],
                "tool_results": [{"name": "admin_boundary", "tool_call_id": "c9",
                                  "content": json.dumps({"ok": True, "level": "county",
                                                         "feature_count": 1})}]}

    def synth(*a, **k):
        raise RuntimeError("model unreachable")

    run_supervisor("q", llm=_fake_llm, thread_id=thread, do_rerank=False, do_audit=False,
                   decide_fn=_scripted(["analyze", "done"]), analyze_fn=analyze,
                   search_fn=lambda q, s: [], synthesize_fn=synth)
    rows = session_memory.get_session_actions(thread)
    assert any(r.get("tool") == "admin_boundary" for r in rows), rows


def test_a_failed_synthesis_does_not_route_back():
    """A `reground` flag from an earlier pass survived a failed composition and sent the graph
    back to the supervisor, where the next composition failed the same way."""
    state = {"query": "q", "reground": True, "grounding_gaps": ["x"], "actions": ["analyze"],
             "evidence": [], "turn_log_id": turn_log.new_log().id}

    def synth(*a, **k):
        raise RuntimeError("model unreachable")

    graph = g.build_supervisor_graph(llm=_fake_llm, decide_fn=_scripted(["done"]),
                                     search_fn=lambda q, s: [], synthesize_fn=synth,
                                     do_rerank=False, do_audit=False)
    out = graph.invoke({**state, "step": 0, "max_steps": 8, "needs": []})
    assert out.get("reground") is False
    assert out["actions"].count("done") == 1, out["actions"]


# --------------------------------------------------------------------------- the plan


def test_the_plan_is_in_state_and_every_peer_step_is_given_it():
    seen = {}

    def decide(state, distilled):
        if state.get("actions"):
            return "done"
        d = g.Decision("analyze")
        d.plan = {"goal": "area and distance",
                  "subgoals": [{"id": "s1", "text": "fetch the county boundary", "status": "doing"},
                               {"id": "s2", "text": "measure the distance", "status": "todo"}]}
        d.subgoal = "s1"
        return d

    def analyze(q, ev, st):
        seen["brief"] = turn_log.active_brief()
        log, run = turn_log.active(), turn_log.active_run()
        log.record_result(peer="analyze", run=run, name="admin_boundary", args={}, call_id="c1",
                          content='{"ok": true, "geoid": "17019"}')
        return {"summary": "boundary fetched"}

    state = _run(decide_fn=decide, analyze_fn=analyze)
    assert "fetch the county boundary" in seen["brief"] and "compute the thing" in seen["brief"]
    s1 = next(x for x in state["plan"]["subgoals"] if x["id"] == "s1")
    assert s1["produced"] >= 1                     # measured from the log, not declared


def test_a_subgoal_whose_run_produced_nothing_is_marked_blocked():
    def decide(state, distilled):
        if state.get("actions"):
            return "done"
        d = g.Decision("analyze")
        d.plan = {"goal": "g", "subgoals": [{"id": "s1", "text": "t", "status": "doing"}]}
        d.subgoal = "s1"
        return d

    state = _run(decide_fn=decide, analyze_fn=lambda q, ev, st: {"summary": ""})
    assert state["plan"]["subgoals"][0]["status"] == "blocked"


def test_the_decider_parses_a_plan_and_stays_a_string():
    class LLM:
        def invoke(self, prompt):
            return json.dumps({"next": "analyze", "reason": "r",
                               "plan": {"goal": "g", "subgoals": [{"id": "s1", "text": "t"}]},
                               "subgoal": "s1"})

    out = g.default_decide_fn(llm=LLM())({"query": "q"}, {"available_actions": ["analyze", "done"]})
    assert out == "analyze" and out.plan["goal"] == "g" and out.subgoal == "s1"


# --------------------------------------------------------------------------- re-run predicate


class _ProducerLLM:
    def __init__(self, answer):
        self.answer = answer

    def invoke(self, prompt):
        return self.answer


def test_a_claim_is_producible_only_by_a_tool_that_is_bound():
    docs = {"execute_code": "run Python", "admin_boundary": "fetch a boundary"}
    out = g._producing_tools(["460 km by road", "the county area"], docs,
                             _ProducerLLM('{"1": "osrm_route", "2": "execute_code"}'))
    assert out == {"460 km by road": None, "the county area": "execute_code"}


def test_an_unanswerable_producer_check_leaves_the_claim_producible():
    """Unknown is not "no": the re-run is allowed, and the progress rule bounds it."""
    docs = {"execute_code": "run Python"}
    assert g._producing_tools(["x"], docs, _ProducerLLM("not json"))["x"] == "unknown"
    assert g._producing_tools(["x"], {}, _ProducerLLM('{"1": "NONE"}'))["x"] == "unknown"


# --------------------------------------------------------------------------- the log itself


@pytest.mark.parametrize("content,status,ok", [
    ('{"ok": false, "error": "boom"}', None, False),
    ('{"error": "overpass_failed", "message": "504", "count": 0}', None, False),
    ('{"ok": true, "exit_code": 1, "stderr": "Traceback"}', None, False),
    ('{"ok": true, "timed_out": true}', None, False),
    ("Error: ValueError: bad", None, False),
    ("anything", "error", False),
    ('{"ok": true, "error": null, "count": 3}', None, True),
    ('{"results": [], "found": false}', None, True),
])
def test_results_are_read_whatever_shape_the_tool_chose(content, status, ok):
    assert turn_log.parse_result(content, status)["ok"] is ok


def test_two_failures_that_differ_only_in_particulars_are_one_class():
    a = turn_log.error_class("DataSourceError: '/work/uploads/file_2272c8426ec9.geojson': No such file")
    b = turn_log.error_class("DataSourceError: '/work/Champaign_County.geojson': No such file")
    c = turn_log.error_class("KeyError: 'area_km2'")
    assert a == b and a != c
