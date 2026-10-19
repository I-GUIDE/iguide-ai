"""The agent's own model client (stage 88), against a stub OpenAI-compatible server.

Two kinds of test. The client's own guarantees — a stalled call ends at its deadline, Stop ends
a call mid-flight, retries are bounded and only for what is worth retrying, streaming — run
against a local HTTP server that can stall, fail or stream. And parity: for the same messages and
tools, the request ``ChatOpenAI`` sends and the request ``OwnChatModel`` sends are compared as
the server received them, and the replies are compared as the agent sees them (content, tool
calls, reasoning carry-over, the usage the cost accounting records). A harness run on the two
clients then compares the client, not a changed prompt.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List

import pytest

from agent_runtime import model_client, turn_control
from agent_runtime.model_client import ChatClient, ModelCallError, ModelCallTimeout

PLAN = ("The user wants the county's area. I should call admin_boundary for Champaign "
        "County, then measure it in an equal-area CRS.")


def _reply(*, content: Any = "done", reasoning: str = "", tool_call: bool = False,
           model: str = "deepseek-v4-flash") -> Dict[str, Any]:
    message: Dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_call:
        message["tool_calls"] = [{"id": "call_1", "type": "function", "function": {
            "name": "admin_boundary", "arguments": '{"area": "Champaign County"}'}}]
    return {"id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": model,
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if tool_call else "stop"}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150,
                      "completion_tokens_details": {"reasoning_tokens": 12},
                      "prompt_tokens_details": {"cached_tokens": 64}}}


class Stub:
    """A chat/completions server driven by a script: one entry per request, the last repeats."""

    def __init__(self) -> None:
        self.script: List[Dict[str, Any]] = [{"json": _reply()}]
        self.bodies: List[Dict[str, Any]] = []
        self.disconnects = 0
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                pass

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                n = len(stub.bodies)
                stub.bodies.append(body)
                step = stub.script[min(n, len(stub.script) - 1)]
                if step.get("stall"):
                    # Hold the socket open with nothing sent; notice when the client hangs up.
                    end = time.monotonic() + step["stall"]
                    self.connection.settimeout(0.05)
                    while time.monotonic() < end:
                        try:
                            if self.connection.recv(1) == b"":
                                stub.disconnects += 1
                                return
                        except TimeoutError:
                            continue
                        except OSError:
                            stub.disconnects += 1
                            return
                    return
                if "stream" in step:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for item in step["stream"]:
                        if item == "stall":
                            time.sleep(step.get("stall_s", 5))
                            return
                        self.wfile.write(f"data: {json.dumps(item)}\n\n".encode())
                        self.wfile.flush()
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    self.close_connection = True
                    return
                status = step.get("status", 200)
                payload = json.dumps(step.get("json", {"error": {"message": "boom"}})).encode()
                self.send_response(status)
                for k, v in (step.get("headers") or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture()
def stub():
    s = Stub()
    yield s
    s.close()


def _body() -> Dict[str, Any]:
    return {"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "hi"}]}


def _call_threads() -> int:
    return sum(1 for t in threading.enumerate() if t.name in ("model-call", "model-stream"))


# --- the client's guarantees --------------------------------------------------------------------

def test_a_stalled_call_ends_at_its_timeout(stub):
    stub.script = [{"stall": 30}]
    client = ChatClient(stub.base_url, "k", timeout=1.0, max_retries=0)
    t0 = time.monotonic()
    with pytest.raises(ModelCallTimeout) as err:
        client.complete(_body())
    took = time.monotonic() - t0
    assert 0.9 <= took < 2.0, took
    assert err.value.attempts == 1
    # The socket was shut, not abandoned: the server saw the hang-up and the helper is gone.
    deadline = time.monotonic() + 2
    while (stub.disconnects < 1 or _call_threads()) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert stub.disconnects == 1
    assert _call_threads() == 0


def test_a_stalled_call_is_retried_once_then_fails(stub):
    stub.script = [{"stall": 30}]
    client = ChatClient(stub.base_url, "k", timeout=0.6, max_retries=1)
    t0 = time.monotonic()
    with pytest.raises(ModelCallTimeout) as err:
        client.complete(_body())
    assert time.monotonic() - t0 < 3.0
    assert err.value.attempts == 2
    assert len(stub.bodies) == 2


def test_the_timeout_comes_from_the_shared_env_setting(stub, monkeypatch):
    monkeypatch.setenv(model_client.TIMEOUT_ENV, "0.7")
    monkeypatch.setenv(model_client.RETRIES_ENV, "0")
    stub.script = [{"stall": 30}]
    t0 = time.monotonic()
    with pytest.raises(ModelCallTimeout):
        ChatClient(stub.base_url, "k").complete(_body())
    assert time.monotonic() - t0 < 1.5


def test_stop_ends_a_call_mid_flight(stub):
    stub.script = [{"stall": 30}]
    client = ChatClient(stub.base_url, "k", timeout=60, max_retries=2)
    handle = turn_control.begin(owner=None)
    outcome: Dict[str, Any] = {}

    def run() -> None:
        with turn_control.bound(handle):
            t0 = time.monotonic()
            try:
                client.complete(_body())
            except BaseException as exc:  # noqa: BLE001
                outcome["exc"] = exc
            outcome["took"] = time.monotonic() - t0

    worker = threading.Thread(target=run)
    worker.start()
    time.sleep(0.5)
    assert turn_control.request_cancel(handle.id, None) == "cancelled"
    worker.join(5)
    turn_control.end(handle)
    assert isinstance(outcome.get("exc"), turn_control.TurnCancelled)
    assert outcome["took"] < 1.5
    assert len(stub.bodies) == 1, "a stopped call is not retried"


def test_a_server_error_is_retried_and_the_retry_answers(stub):
    stub.script = [{"status": 500, "json": {"error": {"message": "Upstream error"}}},
                   {"json": _reply(content="ok")}]
    data = ChatClient(stub.base_url, "k", timeout=5, max_retries=1).complete(_body())
    assert data["choices"][0]["message"]["content"] == "ok"
    assert len(stub.bodies) == 2


def test_a_rate_limit_waits_its_retry_after(stub):
    stub.script = [{"status": 429, "headers": {"Retry-After": "0"},
                    "json": {"error": {"message": "slow down"}}}, {"json": _reply()}]
    ChatClient(stub.base_url, "k", timeout=5, max_retries=1).complete(_body())
    assert len(stub.bodies) == 2


def test_a_bad_request_is_not_retried_and_keeps_the_providers_message(stub):
    stub.script = [{"status": 400, "json": {"error": {"message": "Unsupported value: temperature"}}}]
    with pytest.raises(ModelCallError) as err:
        ChatClient(stub.base_url, "k", timeout=5, max_retries=3).complete(_body())
    assert len(stub.bodies) == 1
    assert err.value.status == 400
    assert str(err.value).startswith("Error code: 400 - ")
    assert "Unsupported value: temperature" in str(err.value)


def test_retries_are_bounded(stub):
    stub.script = [{"status": 503, "json": {"error": {"message": "busy"}}}]
    with pytest.raises(ModelCallError) as err:
        ChatClient(stub.base_url, "k", timeout=5, max_retries=2).complete(_body())
    assert len(stub.bodies) == 3 and err.value.attempts == 3


def test_a_connection_refused_is_an_error_not_a_hang():
    client = ChatClient("http://127.0.0.1:9/v1", "k", timeout=2, max_retries=0)
    with pytest.raises(ModelCallError):
        client.complete(_body())


def test_the_hard_abort_hook_is_in_place():
    """httpx 0.28 gives no argument for the pool's network backend; it is replaced by attribute.
    If an httpx upgrade renames it, this fails instead of aborts silently degrading."""
    assert model_client._Lease().abortable


def _chunk(delta: Dict[str, Any], finish: Any = None, usage: Any = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"id": "c", "object": "chat.completion.chunk", "model": "m",
                           "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage:
        out = {**out, "choices": [], "usage": usage}
    return out


def test_streaming_yields_chunks_and_usage(stub):
    stub.script = [{"stream": [_chunk({"role": "assistant", "content": "Hel"}),
                               _chunk({"content": "lo"}, finish="stop"),
                               _chunk({}, usage={"prompt_tokens": 5, "completion_tokens": 2})]}]
    chunks = list(ChatClient(stub.base_url, "k", timeout=5).stream(_body()))
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"]) \
        == "Hello"
    assert chunks[-1]["usage"]["completion_tokens"] == 2
    assert stub.bodies[0]["stream"] is True
    assert stub.bodies[0]["stream_options"] == {"include_usage": True}


def test_a_stream_that_goes_silent_ends_at_its_idle_timeout_without_a_retry(stub):
    stub.script = [{"stream": [_chunk({"content": "partial"}), "stall"], "stall_s": 30}]
    client = ChatClient(stub.base_url, "k", timeout=0.8, max_retries=2)
    got: List[Any] = []
    t0 = time.monotonic()
    with pytest.raises(ModelCallTimeout):
        for chunk in client.stream(_body()):
            got.append(chunk)
    assert len(got) == 1 and time.monotonic() - t0 < 2.5
    assert len(stub.bodies) == 1, "text already handed over is never replayed by a retry"


# --- the LangChain adapter: parity with ChatOpenAI -------------------------------------------

lc_openai = pytest.importorskip("langchain_openai")


def _models(stub, monkeypatch, **kwargs):
    from agent_runtime import executor_factory as ef
    from agent_runtime.model_client_langchain import OwnChatModel

    monkeypatch.delenv(model_client.CLIENT_ENV, raising=False)
    legacy_cls = ef._reasoning_preserving_chat_openai()
    args = {"model": "deepseek-v4-flash", "api_key": "k", "base_url": stub.base_url,
            "temperature": 0.0, **kwargs}
    return legacy_cls(max_retries=0, **args), OwnChatModel(max_retries=0, **args)


def _conversation():
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    return [
        SystemMessage(content="You are a GIS analyst."),
        HumanMessage(content="Area of Champaign County, please."),
        AIMessage(content=PLAN, tool_calls=[{"name": "admin_boundary", "id": "call_1",
                                             "args": {"area": "Champaign County",
                                                      "note": "Ünïcode ok"}}]),
        ToolMessage(content='{"ok": true, "area_km2": 2582.0}', tool_call_id="call_1"),
        AIMessage(content="", tool_calls=[{"name": "measure", "id": "call_2", "args": {}}]),
        ToolMessage(content="2582 km2", tool_call_id="call_2", name="measure"),
        HumanMessage(content=[{"type": "text", "text": "And in square miles?"}]),
    ]


def _tools():
    from langchain_core.tools import tool

    @tool
    def admin_boundary(area: str, subdivide: str = "") -> str:
        """Fetch an administrative boundary by name."""
        return area

    @tool
    def measure(layer: str = "") -> str:
        """Measure a layer's area."""
        return layer

    return [admin_boundary, measure]


def test_the_request_body_is_the_same_as_chatopenai_sends(stub, monkeypatch):
    legacy, own = _models(stub, monkeypatch)
    for model in (legacy, own):
        model.bind_tools(_tools()).invoke(_conversation())
    sent_legacy, sent_own = stub.bodies
    assert sent_own == sent_legacy


@pytest.mark.parametrize("choice", ["any", "measure", True, "auto"])
def test_tool_choice_is_translated_as_chatopenai_does(stub, monkeypatch, choice):
    legacy, own = _models(stub, monkeypatch)
    for model in (legacy, own):
        model.bind_tools(_tools(), tool_choice=choice).invoke("hi")
    assert stub.bodies[0]["tool_choice"] == stub.bodies[1]["tool_choice"]


def test_openai_effort_and_no_temperature_are_sent_as_chatopenai_sends(stub, monkeypatch):
    from agent_runtime import executor_factory as ef
    from agent_runtime.model_client_langchain import OwnChatModel

    monkeypatch.delenv(model_client.CLIENT_ENV, raising=False)
    args = {"model": "gpt-5.6-luna", "api_key": "k", "base_url": stub.base_url,
            "reasoning_effort": "none"}
    ef._reasoning_preserving_chat_openai()(max_retries=0, **args).invoke("hi")
    OwnChatModel(**args).invoke("hi")
    assert stub.bodies[0] == stub.bodies[1]
    assert "temperature" not in stub.bodies[1] and stub.bodies[1]["reasoning_effort"] == "none"


@pytest.mark.parametrize("reply", [
    _reply(content=None, reasoning=PLAN, tool_call=True),        # carried over into content
    _reply(content="Calling it now.", reasoning=PLAN, tool_call=True),  # own text kept
    _reply(content="The area is 2,582 km².", reasoning=PLAN),    # final answer: never promoted
    _reply(content=None, tool_call=True),                        # OpenAI: no reasoning on the wire
    _reply(content="plain"),
])
def test_the_reply_reads_the_same_on_both_clients(stub, monkeypatch, reply):
    stub.script = [{"json": reply}]
    legacy, own = _models(stub, monkeypatch)
    a = legacy.bind_tools(_tools()).invoke("hi")
    b = own.bind_tools(_tools()).invoke("hi")
    assert b.content == a.content
    assert b.tool_calls == a.tool_calls
    assert b.additional_kwargs.get("reasoning_content") == a.additional_kwargs.get(
        "reasoning_content")
    assert b.usage_metadata == a.usage_metadata
    assert b.response_metadata.get("finish_reason") == a.response_metadata.get("finish_reason")


def test_the_carried_over_reasoning_is_sent_back_as_content_not_as_a_field(stub, monkeypatch):
    """deepseek refuses reasoning_content in input messages; the plan rides in content."""
    stub.script = [{"json": _reply(content=None, reasoning=PLAN, tool_call=True)}]
    _, own = _models(stub, monkeypatch)
    step = own.bind_tools(_tools()).invoke("hi")
    from langchain_core.messages import HumanMessage, ToolMessage

    own.bind_tools(_tools()).invoke([HumanMessage(content="hi"), step,
                                     ToolMessage(content="ok", tool_call_id="call_1")])
    replayed = stub.bodies[1]["messages"][1]
    assert replayed["content"] == PLAN
    assert "reasoning_content" not in replayed


def test_cost_accounting_records_the_same_call_on_both_clients(stub, monkeypatch):
    """Stage 59's handler prices a call from the model id and recognises Lumen by base URL."""
    from agent_runtime import streaming_trace

    monkeypatch.setenv("LUMEN_URL", stub.base_url)
    legacy, own = _models(stub, monkeypatch)
    recorded = []
    for model in (legacy, own):
        with streaming_trace.usage_collection() as handler:
            model.invoke("hi")
            recorded.append(list(handler.calls))
    assert recorded[0] == recorded[1]
    call = recorded[1][0]
    assert call["model"] == "deepseek-v4-flash" and call["provider"] == "lumen"
    assert (call["input_tokens"], call["output_tokens"]) == (120, 30)
    assert call["reasoning_tokens"] == 12 and call["cached_input_tokens"] == 64


def test_the_adapter_streams_with_the_carry_over(stub, monkeypatch):
    stub.script = [{"stream": [
        _chunk({"role": "assistant", "reasoning_content": PLAN[:40]}),
        _chunk({"reasoning_content": PLAN[40:]}),
        _chunk({"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {
            "name": "admin_boundary", "arguments": '{"area": '}}]}),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"Champaign County"}'}}]},
               finish="tool_calls"),
        _chunk({}, usage={"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13}),
    ]}]
    _, own = _models(stub, monkeypatch)
    msg = None
    for part in own.bind_tools(_tools()).stream("hi"):
        msg = part if msg is None else msg + part
    assert msg.content == PLAN
    assert msg.tool_calls[0]["name"] == "admin_boundary"
    assert msg.tool_calls[0]["args"] == {"area": "Champaign County"}
    assert msg.usage_metadata["output_tokens"] == 4


def test_the_adapters_timeout_surfaces_through_invoke(stub, monkeypatch):
    stub.script = [{"stall": 30}]
    _, own = _models(stub, monkeypatch, timeout=0.6)
    t0 = time.monotonic()
    with pytest.raises(ModelCallTimeout):
        own.invoke("hi")
    assert time.monotonic() - t0 < 1.5


def test_a_create_agent_turn_runs_the_same_on_both_clients(stub, monkeypatch):
    """A tool call, its result, then the answer: the agent loop sees the same messages."""
    from langchain.agents import create_agent

    stub.script = [{"json": _reply(content=None, reasoning=PLAN, tool_call=True)},
                   {"json": _reply(content="2,582 km².")}] * 2
    legacy, own = _models(stub, monkeypatch)
    runs = []
    for model in (legacy, own):
        agent = create_agent(model=model, tools=_tools(), system_prompt="be brief")
        out = agent.invoke({"messages": [{"role": "user", "content": "area?"}]})
        runs.append([(m.type, m.content, getattr(m, "tool_calls", None))
                     for m in out["messages"]])
    stub.script = stub.script  # (two replies per run, consumed in order)
    assert runs[0] == runs[1]
    assert stub.bodies[0] == stub.bodies[2] and stub.bodies[1] == stub.bodies[3]


# --- the flag ---------------------------------------------------------------------------------

def test_the_flag_picks_the_client_and_defaults_to_langchain(monkeypatch):
    from agent_runtime import executor_factory as ef

    monkeypatch.setenv("AGENT_LLM_PROVIDER", "lumen")
    monkeypatch.setenv("LUMEN_API_KEY", "k")
    monkeypatch.setattr(ef, "_lumen_fallback", lambda: None)
    monkeypatch.delenv(model_client.CLIENT_ENV, raising=False)
    assert type(ef.build_default_llm()).__name__ == "ReasoningPreservingChatOpenAI"
    monkeypatch.setenv(model_client.CLIENT_ENV, "own")
    llm = ef.build_default_llm()
    assert type(llm).__name__ == "OwnChatModel"
    assert llm.model_name == "deepseek-v4-flash" and llm.temperature == 0.0
    monkeypatch.setenv("AGENT_MODE", "local")
    picked = ef.build_llm(provider="lumen", model="gemma-4-31b-it")
    assert type(picked).__name__ == "OwnChatModel" and picked.model_name == "gemma-4-31b-it"
    monkeypatch.setenv(model_client.CLIENT_ENV, "anything-else")
    assert type(ef.build_default_llm()).__name__ == "ReasoningPreservingChatOpenAI"


def test_the_description_names_the_client(monkeypatch):
    from agent_runtime import executor_factory as ef

    monkeypatch.delenv(model_client.CLIENT_ENV, raising=False)
    assert ef.active_llm_description()["client"] == "langchain"
    monkeypatch.setenv(model_client.CLIENT_ENV, "own")
    assert ef.active_llm_description()["client"] == "own"
