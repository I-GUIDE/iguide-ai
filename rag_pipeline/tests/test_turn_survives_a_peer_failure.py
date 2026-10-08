"""A turn that loses one capability must still deliver what the others produced.

Reproduced before the fix, from a real server log: a search peer that had already merged 20
documents into supervisor state raised on its next call, and the exception unwound the whole
``graph.invoke``. The user got a raw Python exception string and nothing else — no answer, no
evidence, no record of having asked.

The capability to degrade was already there. ``synthesize_node`` can answer from partial
evidence, from conversation history, or state an honest insufficiency; the sibling orchestration
arm has caught sub-agent exceptions since it was written (``legacy/graph_nodes.py:265-283``).
Nothing was missing except a ``try`` at the peer boundary, so the exception aborted the graph
before ``synthesize`` was ever routed to.

Four boundaries are asserted here, because containing only the obvious one moves the crash a node
later and buys nothing — which is what happened on the first attempt: peers were contained, and
the run then died in ``synthesize``, because the failure being survived is *the model being
unreachable* and synthesis calls the model too.
"""

from __future__ import annotations

import pytest

from agent_runtime.supervisor import graph as G

DOCS = [{"doc_id": f"d{i}", "title": f"Chicago crime notebook {i}",
         "url": f"https://platform.i-guide.io/e/{i}", "contents": "crime data"}
        for i in range(20)]

CRASH = "claude CLI was killed by signal 11 (model=sonnet) after 3 attempt(s)"


def _boom(*_a, **_k):
    raise RuntimeError(CRASH)


def _ok_search(_q, _state):
    return {"documents": list(DOCS)}


def _run(**kw):
    kw.setdefault("search_fn", _ok_search)
    kw.setdefault("analyze_fn", _boom)
    kw.setdefault("code_fn", _boom)
    return G.run_supervisor("What Chicago crime data does the platform have?", **kw)


# ------------------------------------------------------------------ the turn survives

def test_a_search_peer_that_raises_does_not_destroy_the_turn():
    """The reported case. Before: `evidence in state: 10` → `RAISED` → `synthesize ran: False`."""
    out = _run(search_fn=_boom)
    assert (out.get("final_answer") or "").strip(), "no answer was produced at all"
    assert any(f["peer"] == "search" for f in (out.get("peer_failures") or []))


def test_evidence_gathered_before_the_failure_is_kept():
    """Retrieval is the expensive half of a turn and it usually survives the failure. Throwing it
    away costs the user the work that actually completed."""
    calls = {"n": 0}

    def flaky(q, state):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"documents": list(DOCS)}
        raise RuntimeError(CRASH)

    out = _run(search_fn=flaky)
    assert out.get("evidence"), "the documents from the successful sweep were discarded"
    answer = out.get("final_answer") or ""
    assert answer.strip()


def test_synthesis_itself_is_contained():
    """The failure being survived is the model being unreachable, and synthesis calls the model.
    Containing the peers and leaving synthesis exposed just moves the crash one node later."""
    out = _run(synthesize_fn=_boom)
    answer = out.get("final_answer") or ""
    assert answer.strip(), "a failing synthesizer must still yield an answer"
    assert any(f["peer"] == "synthesize" for f in (out.get("peer_failures") or []))


def test_the_decider_is_contained_too():
    """The decider calls the model, so it fails FIRST when the backend is down — before any peer
    has run, which would make peer containment moot in exactly the case it exists for."""
    out = _run(decide_fn=_boom)
    assert (out.get("final_answer") or "").strip()


def test_a_total_backend_outage_still_produces_an_answer():
    """Decider, every peer, and synthesis all failing is the shape of an expired token or a dead
    endpoint. It must degrade, not crash."""
    out = _run(decide_fn=_boom, search_fn=_boom, analyze_fn=_boom, code_fn=_boom,
               synthesize_fn=_boom)
    assert (out.get("final_answer") or "").strip()


# ------------------------------------------------------------------ and it says so

def test_a_partial_answer_says_it_is_partial():
    """An answer built after a capability failed is real but incomplete. Presenting it as a normal
    answer hides that something the user asked for did not run."""
    out = _run(analyze_fn=_boom, decide_fn=lambda *_a, **_k: "analyze")
    answer = out.get("final_answer") or ""
    if any(f["peer"] == "analyze" for f in (out.get("peer_failures") or [])):
        assert "Partial answer" in answer or "could not compose" in answer


def test_a_failed_lookup_is_not_reported_as_an_empty_corpus():
    """The load-bearing honesty case. "The knowledge base has no matching content" is a factual
    claim about the corpus; when the SEARCH RAISED, nothing was looked up and the system has no
    basis for it. Reproduced: a crashed search peer produced exactly that sentence."""
    out = _run(search_fn=_boom)
    answer = (out.get("final_answer") or "").lower()
    assert "knowledge base has no" not in answer
    assert "no matching content" not in answer
    assert "not evidence that the platform has nothing" in answer


def test_a_search_that_genuinely_finds_nothing_still_says_so():
    """The other half: when the lookup DID run and the corpus has nothing, saying so is true and
    must not be softened into a failure message."""
    out = _run(search_fn=lambda _q, _s: {"documents": []})
    answer = (out.get("final_answer") or "").lower()
    assert "knowledge base has no" in answer or "couldn't find any supporting evidence" in answer


def test_the_deterministic_answer_hands_back_the_documents():
    """When the model cannot compose prose, the retrieved documents are still the useful part."""
    calls = {"n": 0}

    def once_then_die(q, state):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"documents": list(DOCS)}
        raise RuntimeError(CRASH)

    out = _run(search_fn=once_then_die, synthesize_fn=_boom)
    answer = out.get("final_answer") or ""
    assert "Chicago crime notebook" in answer, "the retrieved titles were not handed back"
    assert "https://platform.i-guide.io/e/" in answer


# ------------------------------------------------------------------ it stops trying

def test_a_failing_peer_is_not_retried_until_max_steps(monkeypatch):
    """Without a budget a broken peer is re-routed to until max_steps, paying its full latency
    each time to fail identically. The loop already tolerates 8 unproductive steps; it must not
    also tolerate 8 broken ones."""
    monkeypatch.setenv("AGENT_PEER_ERROR_BUDGET", "2")
    calls = {"n": 0}

    def counting_boom(*_a, **_k):
        calls["n"] += 1
        raise RuntimeError(CRASH)

    _run(search_fn=counting_boom, decide_fn=lambda *_a, **_k: "search")
    assert calls["n"] <= 3, f"a broken peer ran {calls['n']} times"


def test_an_unavailable_model_backend_stops_the_run_immediately(monkeypatch):
    """A fatal failure is different from a flaky one: every remaining peer would pay its full
    latency to fail the same way."""
    from rag_pipeline.llm_claude_cli import ClaudeCliUnavailable

    calls = {"n": 0}

    def expired(*_a, **_k):
        calls["n"] += 1
        raise ClaudeCliUnavailable("401 OAuth access token has expired.")

    out = _run(search_fn=expired, decide_fn=lambda *_a, **_k: "search")
    assert calls["n"] == 1, "a dead backend must not be asked twice"
    assert any(f.get("fatal") for f in (out.get("peer_failures") or []))


@pytest.mark.parametrize("raw,expected", [("1", 1), ("2", 2), ("8", 8),
                                          ("99", 8), ("0", 1), ("nope", 2)])
def test_the_error_budget_is_clamped(raw, expected, monkeypatch):
    monkeypatch.setenv("AGENT_PEER_ERROR_BUDGET", raw)
    assert G._peer_error_budget() == expected


# ------------------------------------------------------------------ state is recoverable

def test_each_run_gets_its_own_checkpoint_namespace():
    """The graph is compiled with a checkpointer so partial state survives the invoke that made
    it. Binding the CONVERSATION thread instead would make a second turn resume the first —
    inheriting its `step`, `actions` and `evidence`, so a follow-up would start at step 8 and
    route straight to `done`. Recoverability and cross-turn resumption are different features."""
    first = _run(thread_id="conv-1")
    second = _run(thread_id="conv-1")
    assert first["checkpoint_thread_id"] != second["checkpoint_thread_id"]
    assert second.get("step", 0) <= first.get("step", 0) + 1, (
        "the second turn inherited the first turn's step counter")
    assert "conv-1" in first["checkpoint_thread_id"], (
        "the conversation is still identifiable in the checkpoint key")
