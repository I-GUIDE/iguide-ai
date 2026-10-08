"""A dead peer is not a dead turn — for every peer, not just the code one.

code_node grew a handler because "a peer that hit the recursion limit raised straight out of
the supervisor graph and the user got an SSE error with no answer". analysis_node and
search_node had the same gap and did not get one. That matters most for ANALYZE: it is the peer
the router sends code-shaped work to, and it invokes the model up to five times per turn (the
initial run plus the stuck, map-not-delivered, model-mismatch and execution-honesty retries),
so it has the most chances to raise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langgraph.errors import GraphRecursionError

from agent_runtime.supervisor import graph as g


def _boom(*_a, **_k):
    raise GraphRecursionError("Recursion limit of 60 reached")


def _some_evidence(_q, _state):
    return [{"title": "doc", "content": "something useful"}]


def _scripted(seq):
    """Force the route. Without this the supervisor never reaches analyze on these fixtures,
    the turn survives because the peer never ran, and the test proves nothing."""
    box = {"i": 0}

    def decide(_state, _distilled):
        i = box["i"]
        box["i"] += 1
        return seq[i] if i < len(seq) else "done"

    return decide


def _run(route, **kwargs):
    kwargs.setdefault("synthesize_fn", lambda *a, **k: "an answer")
    kwargs.setdefault("do_rerank", False)
    return g.run_supervisor("q", decide_fn=_scripted(route), **kwargs)


@pytest.mark.parametrize("peer,route,kwargs", [
    ("search", ["search", "done"], {"search_fn": _boom}),
    ("analyze", ["analyze", "done"], {"analyze_fn": _boom}),
    ("code", ["code", "done"], {"code_fn": _boom}),
])
def test_a_raising_peer_does_not_kill_the_turn(peer, route, kwargs):
    out = _run(route, **kwargs)
    assert out["actions"][0] == peer, f"the fixture never routed to {peer}"
    assert out.get("final_answer"), f"{peer} raising lost the whole turn"


def test_a_dead_analyze_peer_still_lets_the_others_answer():
    """The turn must be answerable from what the surviving peers produced."""
    out = _run(["search", "analyze", "done"], search_fn=_some_evidence, analyze_fn=_boom,
               synthesize_fn=lambda *a, **k: "answered from search")
    assert out["actions"] == ["search", "analyze", "done"]
    # The integration keeps backend_swap's deterministic disclosure (`_peer_failure_note`): when a
    # peer failed, the answer says so. The synthesizer's own text must still LEAD it.
    assert out["final_answer"].startswith("answered from search")
    assert "failed during this turn" in out["final_answer"], "the dead analyze peer went unmentioned"
    assert {d["title"] for d in out["evidence"]} == {"doc"}, "search's work was lost too"


def test_the_failure_is_recorded_where_synthesis_can_see_it():
    """Degrading silently would let the answerer describe an analysis that never ran."""
    out = _run(["analyze", "done"], analyze_fn=_boom)
    results = out.get("analysis_results") or {}
    assert "GraphRecursionError" in str(results.get("error", "")), results


def test_a_dead_search_peer_counts_its_attempt():
    """Otherwise the exhaustion logic never advances and the supervisor keeps re-routing to a
    peer that raises every time."""
    out = _run(["search", "done"], search_fn=_boom)
    assert out.get("search_attempts", 0) >= 1
    assert out.get("search_empty_streak", 0) >= 1


def test_the_search_handler_writes_only_declared_state_keys():
    """SupervisorState has no error channel for search; an unknown key would make the handler
    raise the very error it exists to absorb."""
    out = _run(["search", "done"], search_fn=_boom)
    assert set(out) <= set(g.SupervisorState.__annotations__), (
        set(out) - set(g.SupervisorState.__annotations__))
