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
* every ``ChatOpenAI`` built here sends through an httpx transport that refuses a request for a
  stopped turn, retries included, and aborts the one in flight (``agent_runtime.stoppable_http``,
  Stage 95.1).
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


_OK_BODY = (b'{"id":"c1","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,'
            b'"message":{"role":"assistant","content":"OK"},"finish_reason":"stop"}],'
            b'"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}')
_SSE_HEAD = (b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n"
             b"transfer-encoding: chunked\r\n\r\n")


def _sse_chunk(text):
    data = (b'data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m",'
            b'"choices":[{"index":0,"delta":{"content":"' + text + b'"},"finish_reason":null}]}\n\n')
    return b"%x\r\n%s\r\n" % (len(data), data)


class _StallingServer:
    """A model endpoint that counts requests and records when the client hangs up.

    ``mode``: ``stall`` never answers; ``stream`` sends the headers and one SSE chunk of a
    streamed completion, then stalls; ``ok`` answers every request on a kept-alive connection.
    """

    def __init__(self, mode="stall"):
        self.mode = mode
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.accepted = []
        self.requests = 0
        self.hangups = []
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
            threading.Thread(target=self._talk, args=(conn,), daemon=True).start()

    def _talk(self, conn):
        buf = b""
        try:
            while True:
                data = conn.recv(65536)
                if not data:
                    self.hangups.append(time.monotonic())
                    return
                buf += data
                while b"\r\n\r\n" in buf:
                    head, _, rest = buf.partition(b"\r\n\r\n")
                    length = 0
                    for line in head.split(b"\r\n"):
                        if line.lower().startswith(b"content-length:"):
                            length = int(line.split(b":", 1)[1])
                    if len(rest) < length:
                        break
                    buf = rest[length:]
                    self.requests += 1
                    if self.mode == "ok":
                        conn.sendall(b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n"
                                     b"content-length: %d\r\n\r\n%s" % (len(_OK_BODY), _OK_BODY))
                    elif self.mode == "stream":
                        conn.sendall(_SSE_HEAD + _sse_chunk(b"Hel"))
        except OSError:
            self.hangups.append(time.monotonic())

    def close(self):
        for c in self._conns:
            try:
                c.close()
            except OSError:
                pass
        self.sock.close()


def _client(server, **kw):
    from agent_runtime.executor_factory import _reasoning_preserving_chat_openai

    return _reasoning_preserving_chat_openai()(model="m", base_url=server.url, **kw)


def _stop_after(turn, seconds):
    timer = threading.Timer(seconds, turn_control.request_cancel, (turn.id, None))
    timer.start()
    return timer


def test_stop_interrupts_the_model_call_in_flight(turn, monkeypatch):
    """The measured fault of Stage 95's own run: the stalled call ran 42 s past Stop."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    server = _StallingServer("stall")
    try:
        llm = _client(server, timeout=30.0, max_retries=2)
        timer = _stop_after(turn, 0.3)
        t0 = time.monotonic()
        with turn_control.bound(turn), pytest.raises(turn_control.TurnCancelled):
            llm.invoke("hi")
        elapsed = time.monotonic() - t0
        timer.join()
        assert elapsed < 1.5, f"the call ran {elapsed:.1f} s; Stop was at 0.3 s"
        deadline = time.monotonic() + 2
        while not server.hangups and time.monotonic() < deadline:
            time.sleep(0.02)
        assert server.hangups, "the provider's connection was left open after Stop"
        assert len(server.accepted) == 1, f"{len(server.accepted)} requests; retries after Stop"
    finally:
        server.close()


def test_stop_interrupts_a_streamed_answer_between_chunks(turn, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    server = _StallingServer("stream")
    try:
        llm = _client(server, timeout=30.0, max_retries=2, streaming=True)
        got = []
        t0 = time.monotonic()
        with turn_control.bound(turn), pytest.raises(turn_control.TurnCancelled):
            for chunk in llm.stream("hi"):
                got.append(chunk.content)
                if not turn.cancelled:
                    _stop_after(turn, 0.3)
        assert "Hel" in "".join(got), got
        assert time.monotonic() - t0 < 1.5
        assert len(server.accepted) == 1
    finally:
        server.close()


def test_the_sdk_does_not_retry_a_model_request_after_stop(turn, monkeypatch):
    """A request the SDK would send after Stop is refused before it reaches the provider."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    server = _StallingServer("stall")
    try:
        llm = _client(server, timeout=1.0, max_retries=2)
        turn_control.request_cancel(turn.id, None)
        with turn_control.bound(turn), pytest.raises(turn_control.TurnCancelled):
            llm.invoke("hi")
        time.sleep(0.2)
        assert server.accepted == []
    finally:
        server.close()


@pytest.mark.parametrize("in_turn", [False, True])
def test_without_stop_the_same_client_still_times_out_and_retries(in_turn, monkeypatch):
    """Stage 85's timeout and retry are intact, on the worker path and off it."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    server = _StallingServer("stall")
    handle = turn_control.begin(None) if in_turn else None
    try:
        llm = _client(server, timeout=0.5, max_retries=1)
        with turn_control.bound(handle), pytest.raises(LLMCallTimeout):
            llm.invoke("hi")
        time.sleep(0.2)
        assert len(server.accepted) == 2
    finally:
        turn_control.end(handle)
        server.close()


@pytest.mark.parametrize("in_turn", [False, True])
def test_answers_arrive_and_the_connection_is_reused(in_turn, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    server = _StallingServer("ok")
    handle = turn_control.begin(None) if in_turn else None
    try:
        llm = _client(server, timeout=5.0, max_retries=0)
        with turn_control.bound(handle):
            answers = [llm.invoke("hi").content for _ in range(3)]
        assert answers == ["OK", "OK", "OK"]
        assert server.requests == 3 and len(server.accepted) == 1, (
            f"{len(server.accepted)} connections for 3 calls; the lane was not reused")
    finally:
        turn_control.end(handle)
        server.close()


def _openai_http_library():
    import importlib

    import openai

    return importlib.import_module(openai.DefaultHttpxClient.__mro__[1].__module__.split(".")[0])


def test_the_lane_can_shut_its_socket():
    """The private pool backend is where the abort hooks in; an httpx that moves it fails here
    rather than silently leaving a stopped request's worker blocked to its timeout."""
    from agent_runtime import stoppable_http

    lane = stoppable_http.build(_openai_http_library().__name__).StoppableTransport()._take()
    try:
        assert lane.abortable
    finally:
        lane.close()


def test_the_transport_is_from_the_library_openai_sends_with():
    """openai 3.x sends through httpx2; an httpx transport there fails every request."""
    from agent_runtime.stoppable_http import stoppable_http_client

    assert isinstance(stoppable_http_client()._transport, _openai_http_library().BaseTransport)
