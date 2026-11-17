"""Stop ends the turn at the supervisor's next step, and no model request follows it (Stage 95).

Measured 2026-11-17 on the default client (``gis_harness/fault_checks.py``, scenario ``stop``):
Stop pressed 3 s into a stalled code-peer call answered 200, then the call ran to its 45 s
timeout, the openai SDK sent its automatic retry, the peer failed with ``LLMCallTimeout``, the
supervisor routed to synthesis, and synthesis answered from general knowledge: two model requests
and 91 s after the person had stopped the turn. The cancellation check lived only in the
streaming callback handler that peers attach; the supervisor's own steps and the SDK's retries
never passed through it.

Two checks close it, each tested here:

* every node of the supervisor graph starts with a Stop check, and a peer that fails after Stop
  raises ``TurnCancelled`` instead of recording a failure (``_stoppable``, ``_run_peer``);
* every ``ChatOpenAI`` built here sends through an httpx client whose request hook refuses a
  request for a stopped turn, retries included (``_stoppable_http_client``).
"""
from __future__ import annotations

import socket
import threading
import time

import pytest

from agent_runtime import turn_control
from agent_runtime.llm_timeout import LLMCallTimeout
from agent_runtime.supervisor import graph as g


@pytest.fixture
def turn():
    handle = turn_control.begin(None)
    try:
        yield handle
    finally:
        turn_control.end(handle)


def _graph(model_calls, code_fn, decisions=("code", "done")):
    """A supervisor graph whose every model-calling seam records the call."""
    queue = list(decisions)

    def decide(*_a, **_k):
        model_calls.append("decide")
        return queue.pop(0) if queue else "done"

    def synthesize(*_a):
        model_calls.append("synthesize")
        return "answer"

    class _Model:
        """The model the synthesis node calls directly (the general-knowledge answer)."""

        def invoke(self, _prompt):
            model_calls.append("general_answer")
            return "general answer"

    return g.build_supervisor_graph(
        search_fn=lambda q, s: [], analyze_fn=lambda *a: {"summary": "s"},
        code_fn=code_fn, synthesize_fn=synthesize, decide_fn=decide, llm=_Model(),
        do_rerank=False, do_audit=False)


def _state(thread):
    return {"query": "Write and run Python code that sums the first 30 Fibonacci numbers.",
            "thread_id": thread, "actions": [], "evidence": []}


def test_a_peer_that_fails_after_stop_ends_the_turn_with_no_further_model_call(turn):
    """The measured sequence: Stop during the code peer's model call, which then times out."""
    calls = []

    def code(*_a):
        calls.append("code")
        turn_control.request_cancel(turn.id, None)      # Stop, pressed mid-call
        raise LLMCallTimeout("Model call timed out after 45 s", seconds=45)

    graph = _graph(calls, code)
    with turn_control.bound(turn), pytest.raises(turn_control.TurnCancelled):
        graph.invoke(_state("t-stop-peer-fails"))
    assert calls == ["decide", "code"], f"model calls after Stop: {calls[2:]}"


def test_a_peer_that_returns_after_stop_ends_the_turn_before_the_next_decision(turn):
    """A peer whose run completes after Stop (no model call left to notice it)."""
    calls = []

    def code(*_a):
        calls.append("code")
        turn_control.request_cancel(turn.id, None)
        return {"answer": "832039", "executed": True, "tool_calls": [], "tool_results": []}

    graph = _graph(calls, code)
    with turn_control.bound(turn), pytest.raises(turn_control.TurnCancelled):
        graph.invoke(_state("t-stop-peer-returns"))
    assert calls == ["decide", "code"], f"model calls after Stop: {calls[2:]}"


def test_a_turn_stopped_before_it_starts_makes_no_model_call(turn):
    calls = []
    turn_control.request_cancel(turn.id, None)
    graph = _graph(calls, lambda *a: calls.append("code"))
    with turn_control.bound(turn), pytest.raises(turn_control.TurnCancelled):
        graph.invoke(_state("t-stop-before"))
    assert calls == []


def test_a_peer_failure_without_stop_still_degrades_to_synthesis():
    """The containment the check must not break: no Stop, a failed peer, an answer."""
    calls = []

    def code(*_a):
        calls.append("code")
        raise LLMCallTimeout("Model call timed out after 45 s", seconds=45)

    out = _graph(calls, code).invoke(_state("t-no-stop"))
    assert calls[0:2] == ["decide", "code"] and calls[-1] in ("synthesize", "general_answer")
    assert out["final_answer"]


def test_every_node_of_the_graph_checks_for_stop():
    import inspect

    src = inspect.getsource(g.build_supervisor_graph)
    adds = [line.strip() for line in src.splitlines() if "builder.add_node(" in line]
    assert adds and all("_stoppable(" in line for line in adds), adds


class _StallingServer:
    """Accepts model requests and never answers; counts how many arrive."""

    def __init__(self):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.accepted = []
        self._conns = []
        threading.Thread(target=self._serve, daemon=True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.sock.getsockname()[1]}/v1"

    def _serve(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            self.accepted.append(time.monotonic())
            self._conns.append(conn)

    def close(self):
        for c in self._conns:
            c.close()
        self.sock.close()


def test_the_sdk_does_not_retry_a_model_request_after_stop(turn, monkeypatch):
    """The stalled call's own retry, inside one invoke, is refused once the turn is stopped."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from agent_runtime.executor_factory import _reasoning_preserving_chat_openai

    server = _StallingServer()
    try:
        llm = _reasoning_preserving_chat_openai()(
            model="m", base_url=server.url, timeout=1.0, max_retries=2)
        timer = threading.Timer(0.3, turn_control.request_cancel, (turn.id, None))
        timer.start()
        with turn_control.bound(turn), pytest.raises(turn_control.TurnCancelled):
            llm.invoke("hi")
        timer.join()
        time.sleep(0.2)
        assert len(server.accepted) == 1, f"{len(server.accepted)} requests; retries after Stop"
    finally:
        server.close()


def test_without_stop_the_same_client_still_retries(monkeypatch):
    """Stage 85's retry is intact for a turn nobody stopped."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from agent_runtime.executor_factory import _reasoning_preserving_chat_openai

    server = _StallingServer()
    try:
        llm = _reasoning_preserving_chat_openai()(
            model="m", base_url=server.url, timeout=0.5, max_retries=1)
        with pytest.raises(LLMCallTimeout):
            llm.invoke("hi")
        time.sleep(0.2)
        assert len(server.accepted) == 2
    finally:
        server.close()
