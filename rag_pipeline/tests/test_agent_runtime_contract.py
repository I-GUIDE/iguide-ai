"""Characterization tests locking the public agent-runtime contract.

These stub the orchestrator-invocation seam (no real LLM or search backends)
so they exercise the *contract assembly* in ``graph_runtime`` -- the response
dict keys and the streaming events -- which must survive the LangGraph
migration.  Intermediate SSE event *names* are allowed to change across the
migration; what is asserted here is:

* ``run_agent_query`` returns the documented top-level keys + final answer.
* ``stream_agent_query_events`` reaches a terminal ``completed`` event whose
  payload carries the response, and emits ordered execution-state stages.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import agent_runtime.graph_runtime as gr


# ---------------------------------------------------------------------------
# Canned orchestrator result (shape returned by create_agent: {"messages": [...]})
# ---------------------------------------------------------------------------

FINAL_ANSWER = "Here are the flood datasets you asked about."


def _human(content):
    return SimpleNamespace(content=content, type="human", tool_calls=[])


def _ai(content, tool_calls=None):
    return SimpleNamespace(content=content, type="ai", tool_calls=tool_calls or [])


def _tool(name, content, call_id):
    return SimpleNamespace(content=content, name=name, tool_call_id=call_id, type="tool", tool_calls=[])


def _decider(sequence):
    """A decide_fn that walks a fixed sequence of actions, then stops."""
    steps = list(sequence)

    def decide(state, distilled):
        return steps.pop(0) if steps else "done"

    return decide


def _canned_supervisor_state():
    """What `run_supervisor` hands back to its wrapper.

    This used to be the RAW executor shape (`{"messages": [...]}`) stubbed into the
    agents-as-tools arm, which did its own conversion. That arm is gone; the canned value now
    sits at the seam that remains. The messages are kept underneath because the stream tests
    read the tool call out of them.
    """
    return {
        "messages": [
            _human("What datasets exist for floods?"),
            _ai("", tool_calls=[{"name": "search_agent_evidence", "args": {"query": "floods"},
                                 "id": "c1"}]),
            _tool("search_agent_evidence", '{"search_agent_summary": "found 2 docs"}', "c1"),
            _ai(FINAL_ANSWER),
        ],
        "final_answer": FINAL_ANSWER,
        "audit": {},
    }


@pytest.fixture()
def stub_orchestrator(monkeypatch):
    """Replace the orchestrate-node invocation seam with canned data.

    These tests characterise the RESPONSE and STREAM contract — what a caller and an SSE client
    receive — not how the orchestration reaches its answer. They used to stub the
    agents-as-tools arm and pin `AGENT_SUPERVISOR=0`; that arm was removed, so they stub the
    one seam that remains. The contract under test did not change with it, which is the point:
    the shape a client sees should survive the graph behind it being replaced.
    """
    import agent_runtime.supervisor.graph as sg

    # Stubbed BELOW run_supervisor_orchestration, not in place of it: that wrapper emits the
    # orchestrate node_started/node_completed pair the lifecycle test asserts on, so replacing
    # it would quietly delete the thing under test. Canning `run_supervisor` leaves the real
    # wrapper, the real trace emission and the real state mapping in the path.
    def fake_run_supervisor(query, **kwargs):
        return _canned_supervisor_state()

    monkeypatch.setattr(sg, "run_supervisor", fake_run_supervisor)
    for name in ("default_search_fn", "default_analyze_fn", "default_code_fn"):
        monkeypatch.setattr(sg, name, lambda **kwargs: (lambda *a, **k: None))
    return None


# ---------------------------------------------------------------------------
# Non-streaming contract
# ---------------------------------------------------------------------------

class _FakeLLM:
    def __init__(self, text):
        self._text = text

    def invoke(self, messages):
        return SimpleNamespace(content=self._text)


def test_triage_fast_paths_trivial_queries():
    from agent_runtime.orchestrator_graph import is_trivial_query

    assert is_trivial_query("hi")
    assert is_trivial_query("Hello!")
    assert is_trivial_query("what can you do?")
    assert not is_trivial_query("What datasets exist for floods?")
    assert not is_trivial_query("find crime hotspots in Chicago")


def test_fast_path_answers_without_orchestrator(monkeypatch):
    import agent_runtime.strategy as strat

    # orchestrate_node resolves the path via the strategy registry; if it ran, this
    # blows up — proving the trivial query was fast-pathed and never orchestrated.
    def explode(*args, **kwargs):
        raise AssertionError("orchestrate path must not run for a greeting")

    monkeypatch.setattr(strat, "get_orchestration_strategy", explode)

    result = gr.run_agent_query("hi", llm=_FakeLLM("Hello! I'm the I-GUIDE assistant."))
    assert result["final_answer"].startswith("Hello!")
    assert result["orchestration_result"] is None


def test_run_agent_query_response_contract(stub_orchestrator):
    result = gr.run_agent_query("What datasets exist for floods?")

    # Documented top-level keys
    assert "orchestration_result" in result
    assert "route_trace" in result
    assert "available_skills" in result
    assert result.get("final_answer") == FINAL_ANSWER
    # checkpointer is set by default -> a thread id is always resolved
    assert isinstance(result.get("thread_id"), str) and result["thread_id"]

    # route_trace is a dict that reflects the called tool
    route_trace = result["route_trace"]
    assert isinstance(route_trace, dict)
    assert "route" in route_trace
    assert "search_agent_evidence" in (route_trace.get("called_tools") or [])


# ---------------------------------------------------------------------------
# Streaming contract
# ---------------------------------------------------------------------------

def _collect_events(query="What datasets exist for floods?"):
    return list(gr.stream_agent_query_events(query))


def test_stream_reaches_terminal_completed_with_payload(stub_orchestrator):
    events = _collect_events()
    names = [e["event"] for e in events]

    assert "completed" in names, f"no terminal completed event; saw {names}"
    completed = [e for e in events if e["event"] == "completed"][-1]["data"]
    assert completed.get("final_answer") == FINAL_ANSWER
    assert "route_trace" in completed
    assert "available_skills" in completed
    assert isinstance(completed.get("thread_id"), str) and completed["thread_id"]

    # final_answer event is emitted with the answer
    final_events = [e for e in events if e["event"] == "final_answer"]
    assert final_events and final_events[-1]["data"].get("answer") == FINAL_ANSWER


def test_stream_emits_ordered_execution_state_stages(stub_orchestrator):
    events = _collect_events()
    stages = [e["data"].get("stage") for e in events if e["event"] == "status"]

    # Execution-state references must appear in order (names may evolve, but
    # the lifecycle started -> initialized -> agent started -> agent completed
    # must be observable for the UI to reflect progress).
    for expected in ("started", "initialized", "orchestration_agent_started", "orchestration_agent_completed"):
        assert expected in stages, f"missing stage {expected!r}; saw {stages}"
    assert stages.index("started") < stages.index("orchestration_agent_completed")


def test_stream_emits_graph_node_lifecycle_events(stub_orchestrator):
    events = _collect_events()
    node_stages = {
        e["data"].get("stage")
        for e in events
        if e["event"] in {"node_started", "node_completed"}
    }
    assert "triage" in node_stages
    assert "orchestrate" in node_stages


# ---------------------------------------------------------------------------
# Fix #2/#3: shared, deduplicated evidence store
# ---------------------------------------------------------------------------

def test_a_throwing_search_peer_does_not_kill_the_turn(monkeypatch):
    """Replaces two tests of `make_search_agent_evidence_tool`, the agents-as-tools search tool
    removed with that arm. Dedup, the other property they covered, is tested on the supervisor
    arm already (test_supervisor_graph: evidence accumulates and dedups across searches). This
    one was not, and it is the half that matters: a dead peer must cost its own result, not the
    whole turn including evidence already gathered.
    """
    from agent_runtime.supervisor import graph as sg

    def exploding_search(query, state):
        raise RuntimeError("the search backend is down")

    # do_audit=False is load-bearing, not tidiness: the grounding audit builds a default LLM
    # and calls it. Left on, this unit test reached the live provider — which is how it turned a
    # 3-minute suite into 25 minutes and left the live spatial e2e test failing behind it, while
    # passing on its own. A test that touches the network is not a unit test, and one that
    # spends a shared rate limit breaks tests it never mentions.
    out = sg.run_supervisor(
        "find flood datasets",
        search_fn=exploding_search,
        analyze_fn=lambda q, ev, st: {"summary": "analysed anyway", "tool_calls": [],
                                      "tool_results": []},
        code_fn=lambda q, ev, st: None,
        synthesize_fn=lambda *a, **k: "an answer",
        decide_fn=_decider(["search", "analyze", "done"]),
        do_audit=False,
        max_steps=4,
    )
    assert out.get("final_answer") == "an answer"


def test_agent_dev_off_emits_status_only(stub_orchestrator, monkeypatch):
    monkeypatch.delenv("AGENT_DEV", raising=False)
    events = _collect_events()
    names = {e["event"] for e in events}

    # Detail-tier events suppressed...
    assert "tool_call" not in names
    assert "llm_interaction" not in names
    assert "route_trace" not in names
    assert "decision" not in names
    # ...but status + answer references remain.
    assert "completed" in names
    assert "final_answer" in names
    assert "status" in names


def test_agent_dev_on_emits_detail(stub_orchestrator, monkeypatch):
    monkeypatch.setenv("AGENT_DEV", "true")
    events = _collect_events()
    names = {e["event"] for e in events}

    # Dev-tier post-run summary (route trace + routing decision). Per-step
    # tool_call/tool_result/llm_interaction now come from the LIVE callback handler
    # during the run (not a post-run replay), which the stubbed orchestrator here
    # does not exercise.
    assert "route_trace" in names
    assert "decision" in names
    assert "completed" in names


def test_agent_dev_request_flag_overrides_env(stub_orchestrator, monkeypatch):
    # env OFF but per-request flag True -> detail events appear
    monkeypatch.delenv("AGENT_DEV", raising=False)
    names_on = {e["event"] for e in gr.stream_agent_query_events("q", agent_dev=True)}
    assert "route_trace" in names_on and "decision" in names_on

    # env ON but per-request flag False -> detail suppressed
    monkeypatch.setenv("AGENT_DEV", "true")
    names_off = {e["event"] for e in gr.stream_agent_query_events("q", agent_dev=False)}
    assert "route_trace" not in names_off and "decision" not in names_off
    assert "completed" in names_off  # status tier still present


# ---------------------------------------------------------------------------
# Robustness: tool failures return a result (never leave a dangling tool_call)
# ---------------------------------------------------------------------------

def test_fallback_does_not_retry_on_tool_ordering_400():
    from agent_runtime.executor_factory import invoke_agent_with_payload_fallback

    class _OrderingErrExecutor:
        def __init__(self):
            self.calls = 0

        def invoke(self, payload, config=None):
            self.calls += 1
            raise RuntimeError(
                "Error code: 400 - invalid_request_error: An assistant message with "
                "'tool_calls' must be followed by tool messages responding to each "
                "'tool_call_id'."
            )

    ex = _OrderingErrExecutor()
    with pytest.raises(RuntimeError):
        invoke_agent_with_payload_fallback(
            ex, query="hi", chat_history=None,
            config={"configurable": {"thread_id": "t"}},
        )
    assert ex.calls == 1, "a tool-ordering 400 must not trigger a legacy-payload retry"


# ---------------------------------------------------------------------------
# Phase 2: history self-healing (tool_call / tool-message repair)
# ---------------------------------------------------------------------------

def _ai_tc(call_id, name="search_agent_evidence"):
    return SimpleNamespace(content="", type="ai", tool_calls=[{"name": name, "args": {}, "id": call_id}])


def _toolmsg(call_id, content="{}"):
    return SimpleNamespace(content=content, type="tool", name="search_agent_evidence", tool_call_id=call_id)


def test_repair_drops_dangling_tool_call():
    from agent_runtime.runtime_utils import repair_tool_call_sequence

    msgs = [_human("hi"), _ai_tc("call_X")]  # tool_call never answered
    fixed, changed = repair_tool_call_sequence(msgs)
    assert changed is True
    assert fixed == [msgs[0]]  # dangling assistant message dropped


def test_repair_drops_orphan_tool_message():
    from agent_runtime.runtime_utils import repair_tool_call_sequence

    msgs = [_human("hi"), _toolmsg("call_ghost"), _ai("answer")]  # tool msg with no AI tool_call
    fixed, changed = repair_tool_call_sequence(msgs)
    assert changed is True
    assert _toolmsg("call_ghost") not in fixed
    assert msgs[0] in fixed and msgs[2] in fixed


def test_repair_passes_through_valid_history():
    from agent_runtime.runtime_utils import repair_tool_call_sequence

    msgs = [_human("hi"), _ai_tc("call_A"), _toolmsg("call_A"), _ai("final answer")]
    fixed, changed = repair_tool_call_sequence(msgs)
    assert changed is False
    assert fixed is msgs  # unchanged -> same object


def test_history_repair_middleware_sanitizes_request():
    from agent_runtime.executor_factory import _make_history_repair_middleware

    mw = _make_history_repair_middleware()

    class _Req:
        def __init__(self, messages):
            self.messages = messages

        def override(self, **kw):
            return _Req(kw.get("messages", self.messages))

    captured = {}

    def handler(req):
        captured["messages"] = req.messages
        return "ok"

    dangling = [_human("hi"), _ai_tc("call_X")]
    result = mw.wrap_model_call(_Req(dangling), handler)
    assert result == "ok"
    assert captured["messages"] == [dangling[0]]  # repaired before model call


# ---------------------------------------------------------------------------
# Supervisor arm contract (the legacy tests pin AGENT_SUPERVISOR=0; this guards
# the DEFAULT supervisor arm through build_orchestrator_graph end-to-end).
# ---------------------------------------------------------------------------

@pytest.fixture()
def stub_supervisor(monkeypatch):
    """Replace run_supervisor (and the peer-fn builders) with canned data so the
    supervisor arm of orchestrate_node runs without an LLM/backends.

    orchestrate_node does `from agent_runtime.supervisor_graph import run_supervisor,
    default_*_fn` at call time, so these are patched on the supervisor_graph module.
    """
    import agent_runtime.supervisor_graph as sg

    monkeypatch.setenv("AGENT_SUPERVISOR", "1")

    def fake_run_supervisor(query, **kwargs):
        return {"final_answer": FINAL_ANSWER, "audit": {}, "evidence": [], "actions": ["search", "done"]}

    monkeypatch.setattr(sg, "run_supervisor", fake_run_supervisor)
    monkeypatch.setattr(sg, "default_search_fn", lambda **k: (lambda *a, **kw: []))
    monkeypatch.setattr(sg, "default_analyze_fn", lambda **k: (lambda *a, **kw: {}))
    monkeypatch.setattr(sg, "default_code_fn", lambda **k: (lambda *a, **kw: {}))
    return None


def test_supervisor_arm_response_contract(stub_supervisor):
    result = gr.run_agent_query("What datasets exist for floods?")
    assert "orchestration_result" in result
    assert "route_trace" in result
    assert "available_skills" in result
    assert result.get("final_answer") == FINAL_ANSWER
    assert isinstance(result.get("thread_id"), str) and result["thread_id"]
    # supervisor advertises the peer names
    assert set(result.get("available_agents") or result.get("available_agent_names") or []) >= {"search", "analyze", "code"} \
        or "search" in str(result.get("route_trace"))


def test_supervisor_arm_stream_reaches_completed(stub_supervisor):
    events = list(gr.stream_agent_query_events("What datasets exist for floods?"))
    names = [e["event"] for e in events]
    assert "completed" in names, f"no terminal completed; saw {names}"
    completed = [e for e in events if e["event"] == "completed"][-1]["data"]
    assert completed.get("final_answer") == FINAL_ANSWER
    node_stages = {e["data"].get("stage") for e in events if e["event"] in {"node_started", "node_completed"}}
    assert "orchestrate" in node_stages


def test_greeting_plus_identity_question_fast_paths():
    """'Hi who are you' previously fell through to retrieval (a bare trivial phrase was required)
    and came back as a no-evidence refusal."""
    from agent_runtime.orchestrator_graph import is_trivial_query
    for q in ("Hi who are you", "hello, what can you do?", "hey who made you",
              "who are you", "what is I-GUIDE?", "thanks"):
        assert is_trivial_query(q), q
    for q in ("hi, find datasets about floods", "what is a shapefile",
              "explain the buffer workflow"):
        assert not is_trivial_query(q), q
