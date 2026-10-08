"""A peer that failed and was then replaced by one that delivered is not a partial answer.

Live, 2026-10-08 19:42 UTC on agent.i-guide.io: the analyze peer hit LangGraph's recursion limit
after 27 geocoding calls, the supervisor sent the work to the code peer, and the code peer
answered the question in full (18 schools, each with its distance, on the map). The reply still
ended "⚠️ Partial answer: analysis failed during this turn, so this reply is based on what
completed before the failure." Neither half of that sentence was true: the reply was not based on
what completed before the failure, it was based on what completed after it.

The note stays wherever it IS true: no later peer ran, the later peer failed too, or it came back
with nothing.
"""
from __future__ import annotations

from langgraph.errors import GraphRecursionError

from agent_runtime.supervisor import graph as g


def _boom(*_a, **_k):
    raise GraphRecursionError("Recursion limit of 60 reached")


def _code_answers(_q, _ev, _st):
    return {"answer": "18 schools, each with its distance", "executed": True}


def _scripted(seq):
    box = {"i": 0}

    def decide(_state, _distilled):
        i = box["i"]
        box["i"] += 1
        return seq[i] if i < len(seq) else "done"

    return decide


def _run(route, **kwargs):
    kwargs.setdefault("synthesize_fn", lambda *a, **k: "the answer")
    kwargs.setdefault("do_rerank", False)
    kwargs.setdefault("search_fn", lambda q, s: [])
    return g.run_supervisor("q", decide_fn=_scripted(route), **kwargs)


def test_a_failed_analyze_answered_by_code_carries_no_banner():
    out = _run(["analyze", "code", "done"], analyze_fn=_boom, code_fn=_code_answers)
    assert out["actions"][:2] == ["analyze", "code"]
    assert "Partial answer" not in out["final_answer"], out["final_answer"]
    # The failure is still on the record for the trace and the ledger; only the claim goes.
    assert [f["peer"] for f in out.get("peer_failures") or []] == ["analyze"]


def test_the_banner_stays_when_the_later_peer_fails_too():
    out = _run(["analyze", "code", "done"], analyze_fn=_boom, code_fn=_boom)
    assert "Partial answer" in out["final_answer"]


def test_the_banner_stays_when_the_later_peer_returns_nothing():
    out = _run(["analyze", "code", "done"], analyze_fn=_boom,
               code_fn=lambda q, ev, st: {"answer": "", "executed": False})
    assert "Partial answer" in out["final_answer"]


def test_the_banner_stays_when_the_failure_came_last():
    """Code answered, THEN analyze failed: what the analysis was asked for never arrived."""
    out = _run(["code", "analyze", "done"], analyze_fn=_boom, code_fn=_code_answers)
    assert "Partial answer" in out["final_answer"]


def test_a_failure_with_no_position_keeps_its_banner():
    """A failure recorded before positions were (an older state, a hand-built one) is not
    assumed resolved."""
    state = {"actions": ["analyze", "code"], "code_result": {"answer": "x", "executed": True},
             "peer_failures": [{"peer": "analyze", "error": "boom", "fatal": False}]}
    assert g._unresolved_peer_failures(state) == state["peer_failures"]
