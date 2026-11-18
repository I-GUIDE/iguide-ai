"""The code peer's own tool-calling loop (stage 94), beside the create_agent graph it replaces.

Two kinds of test. The loop's own rules: an explicit step limit that ends with a message, the
progress rule and the repeat memo read from the turn log, Stop between steps, the brief in the
system message, the budget trim, a bounded thread history, and tool failures worded as the graph
words them. And parity: the same scripted model and the same tools, run once through the real
``create_agent`` graph with the real middleware and once through the loop, produce the same flat
``tool_calls`` / ``tool_results`` / ``answer`` the code node returns.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, List

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool, tool
from langgraph.checkpoint.memory import InMemorySaver

from agent_runtime import code_loop, executor_factory as ef, streaming_trace as st, turn_control, turn_log
from agent_runtime.code_loop import OwnPeerSession, ThreadStore
from agent_runtime.tool_args import accept_null_defaults


class ToolAwareFake(GenericFakeChatModel):
    model_name: str = "deepseek-v4-flash"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
        return self


class RecordingFake(ToolAwareFake):
    """Keeps every request, to see what the model was given."""

    seen: List[List[Any]] = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
        RecordingFake.seen.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


CALLS = {"geocode": 0, "boom": 0}


@tool
def geocode(places: list) -> str:
    """Geocode places."""
    CALLS["geocode"] += 1
    return json.dumps({"ok": True, "results": [{"place": places[0], "lat": 41.8, "lon": -87.7}]})


@tool
def boom(what: str) -> str:
    """Always fails."""
    CALLS["boom"] += 1
    raise RuntimeError(f"no {what}")


def _boundary(area: str, level: str = "county") -> str:
    """A boundary, with an optional level."""
    return f"{area}/{level}"


boundary = StructuredTool.from_function(func=accept_null_defaults(_boundary), name="boundary",
                                        description="A boundary.")

TOOLS = [geocode, boom, boundary]


@pytest.fixture(autouse=True)
def _reset():
    CALLS.update(geocode=0, boom=0)
    RecordingFake.seen = []
    code_loop.THREADS.clear()
    yield
    code_loop.THREADS.clear()


def _call(name: str, args: Dict[str, Any], i: int) -> Dict[str, Any]:
    return {"name": name, "args": args, "id": f"call_{i}", "type": "tool_call"}


def _session(replies, *, thread="t1", cls=ToolAwareFake, tools=TOOLS, prompt="test") -> OwnPeerSession:
    return OwnPeerSession(cls(messages=iter(replies)), tools, prompt, thread)


def _in_turn(fn, *, brief=None):
    log = turn_log.new_log()
    with turn_log.bind(log, "code", brief) as b:
        return fn(), log, b.run


# ------------------------------------------------------------------ the flat shape, and parity

def test_a_tool_call_its_result_and_the_answer_have_the_flat_shape():
    replies = [AIMessage(content="", tool_calls=[_call("geocode", {"places": ["Urbana"]}, 1)]),
               AIMessage(content="Urbana is at 41.8, -87.7.")]
    run, _, _ = _in_turn(lambda: _session(replies).run("where is Urbana"))
    assert run.answer == "Urbana is at 41.8, -87.7."
    assert run.artifacts["tool_calls"] == [{"name": "geocode", "args": {"places": ["Urbana"]},
                                            "id": "call_1"}]
    [res] = run.artifacts["tool_results"]
    assert (res["name"], res["tool_call_id"]) == ("geocode", "call_1")
    assert json.loads(res["content"])["ok"] is True
    assert isinstance(run.resp, dict) and len(run.resp["messages"]) == 4
    assert CALLS["geocode"] == 1


def test_the_graph_and_the_loop_return_the_same_record_for_the_same_script():
    replies = [AIMessage(content="plan", tool_calls=[_call("geocode", {"places": ["Urbana"]}, 1),
                                                     _call("boundary", {"area": "Champaign"}, 2)]),
               AIMessage(content="", tool_calls=[_call("geocode", {"places": ["Urbana"]}, 3)]),
               AIMessage(content="done")]

    def graph_run():
        agent = create_agent(model=ToolAwareFake(messages=iter(replies)), tools=TOOLS,
                             system_prompt="test", middleware=ef._default_middleware(),
                             checkpointer=InMemorySaver())
        session = ef.open_peer_session(agent, {"configurable": {"thread_id": "g"}})
        return session.run("q"), session

    def loop_run():
        session = _session(replies, thread="l")
        return session.run("q"), session

    (g_run, g_sess), _, _ = _in_turn(graph_run)
    CALLS.update(geocode=0)
    (l_run, l_sess), _, _ = _in_turn(loop_run)
    assert l_run.answer == g_run.answer == "done"
    assert l_run.artifacts["tool_calls"] == g_run.artifacts["tool_calls"]
    strip = lambda rows: [(r["name"], r["tool_call_id"], r["content"]) for r in rows]  # noqa: E731
    assert strip(l_run.artifacts["tool_results"]) == strip(g_run.artifacts["tool_results"])
    assert l_sess.turn_artifacts == g_sess.turn_artifacts
    # The third call repeats the first: answered from the turn log under both loops.
    assert "Same call, same arguments as call_1" in l_run.artifacts["tool_results"][2]["content"]
    assert CALLS["geocode"] == 1


# ------------------------------------------------------------------ the loop's own rules

def test_the_step_limit_ends_the_run_with_a_message_and_keeps_its_results(monkeypatch):
    monkeypatch.setenv(code_loop.MAX_STEPS_ENV, "3")
    replies = [AIMessage(content="", tool_calls=[_call("geocode", {"places": [f"P{i}"]}, i)])
               for i in range(1, 10)]
    run, log, _ = _in_turn(lambda: _session(replies).run("q"))
    assert run.answer == code_loop.step_limit_text(3)
    assert [c["args"]["places"] for c in run.artifacts["tool_calls"]] == [["P1"], ["P2"], ["P3"]]
    assert len(run.artifacts["tool_results"]) == 3 and CALLS["geocode"] == 3
    assert len(log.results()) == 3, "every result the run made is in the turn's record"


def test_the_progress_rule_ends_a_run_that_repeats_itself():
    replies = [AIMessage(content="", tool_calls=[_call("geocode", {"places": ["Urbana"]}, i)])
               for i in range(1, 10)]
    run, _, _ = _in_turn(lambda: _session(replies).run("q"))
    assert run.answer == ef.progress_stop_text()
    # Step 1 ran; steps 2 and 3 were answered from the log and added nothing; no step 4.
    assert CALLS["geocode"] == 1
    results = run.artifacts["tool_results"]
    assert len(results) == 3
    assert all("Same call, same arguments as call_1" in r["content"] for r in results[1:])
    repeat = [m for m in run.resp["messages"] if isinstance(m, ToolMessage)][1]
    assert repeat.additional_kwargs.get("repeat_of") == "call_1"


def test_an_unknown_tool_and_a_tool_exception_read_as_under_the_graph():
    replies = [AIMessage(content="", tool_calls=[_call("nope", {}, 1), _call("boom", {"what": "x"}, 2)]),
               AIMessage(content="gave up")]
    run, log, _ = _in_turn(lambda: _session(replies).run("q"))
    unknown, failed = run.artifacts["tool_results"]
    assert unknown["content"] == "Error: nope is not a valid tool, try one of [geocode, boom, boundary]."
    assert failed["content"] == "Error: RuntimeError('no x')\n Please fix your mistakes."
    messages = [m for m in run.resp["messages"] if isinstance(m, ToolMessage)]
    assert [m.status for m in messages] == ["error", "error"]
    assert [r.get("ok") for r in log.results()] == [False], "the exception is in the turn's record"


def test_an_unparseable_tool_call_is_answered_with_its_error():
    bad = {"name": "geocode", "args": "{not json", "id": "call_1", "error": "Expecting value"}
    replies = [AIMessage(content="", invalid_tool_calls=[bad]), AIMessage(content="fixed")]
    run, _, _ = _in_turn(lambda: _session(replies).run("q"))
    assert run.answer == "fixed"
    [msg] = [m for m in run.resp["messages"] if isinstance(m, ToolMessage)]
    assert msg.tool_call_id == "call_1" and msg.status == "error"
    assert msg.content == "Error: Expecting value\n Please fix your mistakes."


def test_null_for_an_optional_argument_means_the_default():
    replies = [AIMessage(content="", tool_calls=[_call("boundary", {"area": "Champaign", "level": None}, 1)]),
               AIMessage(content="ok")]
    run, _, _ = _in_turn(lambda: _session(replies).run("q"))
    assert run.artifacts["tool_results"][0]["content"] == "Champaign/county"


def test_stop_between_steps_raises_turn_cancelled_and_keeps_what_ran():
    replies = [AIMessage(content="", tool_calls=[_call("geocode", {"places": ["A"]}, 1)]),
               AIMessage(content="", tool_calls=[_call("geocode", {"places": ["B"]}, 2)]),
               AIMessage(content="never")]
    handle = turn_control.begin(owner=None)
    outcome: Dict[str, Any] = {}

    count = {"calls": 0}

    class StoppingFake(ToolAwareFake):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
            out = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            count["calls"] += 1
            if count["calls"] == 2:
                turn_control.request_cancel(handle.id, None)   # Stop pressed during step 2
            return out

    def run() -> None:
        with turn_control.bound(handle):
            try:
                _in_turn(lambda: _session(replies, cls=StoppingFake).run("q"))
            except BaseException as exc:  # noqa: BLE001
                outcome["exc"] = exc

    worker = threading.Thread(target=run)
    worker.start()
    worker.join(5)
    turn_control.end(handle)
    assert isinstance(outcome.get("exc"), turn_control.TurnCancelled)
    assert CALLS["geocode"] == 1, "the first call ran; the second's tool call never started"
    history = code_loop.THREADS.get("t1")
    assert [type(m).__name__ for m in history] == ["HumanMessage", "AIMessage", "ToolMessage",
                                                   "AIMessage"]


def test_the_brief_is_in_the_system_message_on_every_call():
    replies = [AIMessage(content="", tool_calls=[_call("geocode", {"places": ["A"]}, 1)]),
               AIMessage(content="done")]
    _in_turn(lambda: _session(replies, cls=RecordingFake, prompt="PROMPT").run("q"),
             brief="Task (the user's request, verbatim): where is A")
    assert len(RecordingFake.seen) == 2
    for request in RecordingFake.seen:
        system = request[0]
        assert isinstance(system, SystemMessage)
        assert system.content.startswith("PROMPT" + ef._BRIEF_HEADER)
        assert "where is A" in system.content
        assert sum(isinstance(m, SystemMessage) for m in request) == 1


def test_the_history_persists_across_runs_on_the_thread_and_is_bounded():
    first = [AIMessage(content="", tool_calls=[_call("geocode", {"places": ["A"]}, 1)]),
             AIMessage(content="first")]
    _in_turn(lambda: _session(first, thread="k").run("one"))
    second = [AIMessage(content="second")]
    run, _, _ = _in_turn(lambda: _session(second, thread="k", cls=RecordingFake).run("two"))
    # The second run was shown the first run's messages, and reports only its own.
    assert [type(m).__name__ for m in RecordingFake.seen[0]] == [
        "SystemMessage", "HumanMessage", "AIMessage", "ToolMessage", "AIMessage", "HumanMessage"]
    assert run.artifacts["tool_calls"] == [] and run.answer == "second"
    assert len(run.resp["messages"]) == 6, "resp is the whole thread, as the graph's was"

    store = ThreadStore(max_threads=2)
    for key in ("a", "b", "c"):
        store.put(key, [HumanMessage(content=key)])
    assert store.get("a") == [] and store.get("c")[0].content == "c"


def test_a_run_with_no_thread_key_keeps_nothing():
    replies = [AIMessage(content="x")]
    run, _, _ = _in_turn(lambda: _session(replies, thread=None).run("q"))
    assert run.answer == "x"
    assert code_loop.THREADS.get("t1") == []


def test_the_history_sent_is_trimmed_to_the_budget_but_the_stored_history_is_not(monkeypatch):
    monkeypatch.setattr(ef, "_derive_budget", lambda request: 60)
    long = [AIMessage(content="", tool_calls=[_call("geocode", {"places": [f"P{i}" * 40]}, i)])
            for i in range(1, 5)] + [AIMessage(content="done")]
    run, _, _ = _in_turn(lambda: _session(long, cls=RecordingFake).run("q"))
    assert run.answer == "done"
    sent = RecordingFake.seen[-1]
    assert len(sent) < 10, "the last request was trimmed"
    assert isinstance(sent[-1], (HumanMessage, ToolMessage))
    assert len(code_loop.THREADS.get("t1")) == 10, "the thread keeps every message"


def test_trace_events_and_cost_are_recorded_through_the_same_callbacks():
    replies = [AIMessage(content="", tool_calls=[_call("geocode", {"places": ["A"]}, 1)]),
               AIMessage(content="done")]
    events: List[Dict[str, Any]] = []
    with st.trace_context(lambda e: events.append(e), agent_dev=True):
        handler = st._USAGE_HANDLER.get()
        _in_turn(lambda: _session(replies).run("q"))
    kinds = [e["event"] for e in events]
    assert kinds.count("llm_start") == 2
    assert "tool_call" in kinds and "tool_result" in kinds
    assert kinds.count("llm_usage") == 2
    assert len(handler.calls) == 2, "the configure-hook cost handler saw both model calls"


def test_the_toolset_catalog_and_per_call_instrumentation_are_logged(monkeypatch, caplog):
    """The harness reads `turn_instrumentation_toolset` with peer == "code" to check the code
    peer's toolset; the own loop logs it under the same thread key the graph would."""
    import logging

    monkeypatch.delenv(ef.INSTRUMENTATION_ENV, raising=False)
    ef._TOOLSET_LOGGED.clear()
    replies = [AIMessage(content="", tool_calls=[_call("geocode", {"places": ["A"]}, 1)]),
               AIMessage(content="done")]
    with caplog.at_level(logging.INFO, logger="agent_runtime.executor_factory"):
        _in_turn(lambda: _session(replies, thread="sess-9::code").run("q"))
    catalog = [json.loads(r.message.split(" ", 1)[1]) for r in caplog.records
               if r.message.startswith("turn_instrumentation_toolset ")]
    assert len(catalog) == 1 and catalog[0]["peer"] == "code"
    assert set(catalog[0]["per_tool"]) == {"geocode", "boom", "boundary"}
    calls = [json.loads(r.message.split(" ", 1)[1]) for r in caplog.records
             if r.message.startswith("turn_instrumentation ")]
    assert [c["peer"] for c in calls] == ["code", "code"]
    assert calls[0]["session"] == "sess-9" and calls[0]["tools_called"] == ["geocode"]
    assert calls[1]["tools_called"] == [] and calls[1]["messages"] == 3  # human, ai, tool


# ------------------------------------------------------------------ the switch

def test_the_flag_defaults_to_langchain(monkeypatch):
    monkeypatch.delenv(code_loop.LOOP_ENV, raising=False)
    assert code_loop.loop_choice() == "langchain" and not code_loop.selected()
    monkeypatch.setenv(code_loop.LOOP_ENV, "own")
    assert code_loop.selected()
    monkeypatch.setenv(code_loop.LOOP_ENV, "graph")
    assert not code_loop.selected()
    monkeypatch.setenv(code_loop.MAX_STEPS_ENV, "x")
    assert code_loop.max_steps() == code_loop.DEFAULT_MAX_STEPS


class _GraphBuilt(Exception):
    pass


def _run_code_peer(monkeypatch, llm):
    import langchain.agents

    import agent_runtime.supervisor.graph as graph

    def stop(**kwargs):
        raise _GraphBuilt()

    monkeypatch.setattr(langchain.agents, "create_agent", stop)
    monkeypatch.setenv("AGENT_SKILLS_ENABLED", "0")
    fn = graph.default_code_fn(llm=llm, code_exec=False, input_file_ids=None, code_peer="langchain")
    log = turn_log.new_log()
    with turn_log.bind(log, "code", "Task: q"):
        return fn("q", [], {"thread_id": "sess-1"})


def test_default_code_fn_builds_the_graph_by_default_and_the_loop_when_selected(monkeypatch):
    monkeypatch.delenv(code_loop.LOOP_ENV, raising=False)
    with pytest.raises(_GraphBuilt):
        _run_code_peer(monkeypatch, ToolAwareFake(messages=iter([AIMessage(content="hi")])))

    monkeypatch.setenv(code_loop.LOOP_ENV, "own")
    result = _run_code_peer(monkeypatch, RecordingFake(messages=iter([AIMessage(content="hi")])))
    assert result["answer"] == "hi"
    assert result["tool_calls"] == [] and result["tool_results"] == []
    system = RecordingFake.seen[0][0]
    assert "Task: q" in system.content, "the brief reached the system message"
    history = code_loop.THREADS.get(ef.child_thread_id("sess-1", "code"))
    assert [type(m).__name__ for m in history] == ["HumanMessage", "AIMessage"]


def test_open_session_prepares_the_tools_the_graph_would_bind(monkeypatch):
    seen: Dict[str, Any] = {}

    def prepare(tools, **kwargs):
        seen["tools"] = list(tools)
        seen["kwargs"] = kwargs
        return [geocode]

    session = code_loop.open_session(llm=ToolAwareFake(messages=iter([])), tools=TOOLS,
                                     system_prompt_override="P", thread_key="k",
                                     skill_roots=["/nowhere"], prepare=prepare)
    assert seen["tools"] == TOOLS and seen["kwargs"] == {"skill_roots": ["/nowhere"]}
    assert [t.name for t in session.tools] == ["geocode"]
    assert session.system_prompt == "P"


# ------------------------------------------------------------------ check 3: a runaway code peer
#
# Four of the seven step-limit overruns in the harness archive (683 turns, 2026-11-17) were the
# code peer (T06 once, T08L three times). This drives a code peer that never stops calling tools
# through the REAL path the code node takes: default_code_fn, whichever loop it builds, and the
# supervisor's _run_peer, with only the tool list replaced. Every call returns something new, so
# the progress rule does not end the run first: only the step limit can.

def _runaway(monkeypatch, loop: str):
    import agent_runtime.supervisor.graph as graph

    counter = {"n": 0}

    @tool
    def measure(i: int) -> str:
        """Measure something new."""
        counter["n"] += 1
        return json.dumps({"ok": True, "value": i * 7, "n": counter["n"]})

    replies = [AIMessage(content="", tool_calls=[_call("measure", {"i": i}, i)])
               for i in range(1, 200)]
    monkeypatch.setenv(code_loop.LOOP_ENV, loop)
    monkeypatch.setenv("AGENT_RECURSION_LIMIT", "25")       # the graph's floor
    monkeypatch.setenv(code_loop.MAX_STEPS_ENV, "12")       # the same model calls, roughly
    monkeypatch.setenv("AGENT_SKILLS_ENABLED", "0")
    monkeypatch.setattr(ef, "prepare_peer_tools", lambda tools, **kw: [measure])
    fn = graph.default_code_fn(llm=ToolAwareFake(messages=iter(replies)), code_exec=False,
                               input_file_ids=None, code_peer="langchain")
    state = {"thread_id": f"runaway-{loop}", "query": "q"}
    log = turn_log.new_log()
    with turn_log.bind(log, "code", "Task: q"):
        result, failure = graph._run_peer("code", lambda: fn("q", [], state), state)
    return result, failure, counter["n"], log


def test_a_runaway_code_peer_is_a_failure_under_the_graph(monkeypatch):
    result, failure, calls, log = _runaway(monkeypatch, "langchain")
    assert result is None
    assert "recursion limit" in failure["error"].lower()
    assert calls >= 10
    # What survives is the turn log's record, not the peer's result (stage 42).
    assert len(log.results("code")) == calls


def test_a_runaway_code_peer_is_an_answer_under_the_own_loop(monkeypatch):
    result, failure, calls, log = _runaway(monkeypatch, "own")
    assert failure is None
    assert result["answer"] == code_loop.step_limit_text(12)
    assert calls == 12
    assert len(result["tool_calls"]) == 12 and len(result["tool_results"]) == 12
    assert len(log.results("code")) == 12


# ------------------------------------------------------------------ the wire: identical requests

def test_both_loops_send_byte_identical_requests_for_the_same_conversation():
    """The comparison and record-and-replay both assume the model is sent the same bytes under
    either loop. Found by this test's first version: create_agent names every stored reply
    "code_agent" and that name is sent; the loop sent none."""
    import threading as _threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from agent_runtime.model_client_langchain import OwnChatModel

    script = [
        {"content": "Looking it up.", "tool_calls": [{"id": "call_1", "type": "function", "function": {
            "name": "geocode", "arguments": '{"places": ["Urbana"]}'}}]},
        {"content": None, "tool_calls": [{"id": "call_2", "type": "function", "function": {
            "name": "geocode", "arguments": '{"places": ["Champaign"]}'}}]},
        {"content": "Done."},
    ]
    bodies: Dict[str, List[Any]] = {"graph": [], "own": []}
    state = {"arm": "graph", "i": 0}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # noqa: ANN002
            pass

        def do_POST(self):  # noqa: N802
            bodies[state["arm"]].append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            msg = {**script[state["i"] % len(script)], "role": "assistant"}
            state["i"] += 1
            out = json.dumps({"id": "x", "object": "chat.completion", "model": "deepseek-v4-flash",
                              "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
                              "usage": {"prompt_tokens": 10, "completion_tokens": 2,
                                        "total_tokens": 12}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    _threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/v1"

    def model():
        return OwnChatModel(model="deepseek-v4-flash", api_key="k", base_url=base,
                            temperature=0.0, max_retries=0)

    try:
        state.update(arm="graph", i=0)
        agent = create_agent(model=model(), tools=[geocode], system_prompt="PROMPT",
                             middleware=ef._default_middleware(), checkpointer=InMemorySaver(),
                             name="code_agent")
        _in_turn(lambda: ef.open_peer_session(
            agent, {"configurable": {"thread_id": "g::code"}}).run("where is Urbana"))
        state.update(arm="own", i=0)
        _in_turn(lambda: OwnPeerSession(model(), [geocode], "PROMPT", "o::code").run("where is Urbana"))
    finally:
        server.shutdown()
    assert len(bodies["graph"]) == len(bodies["own"]) == 3
    for g, o in zip(bodies["graph"], bodies["own"]):
        assert json.dumps(g, sort_keys=True) == json.dumps(o, sort_keys=True)
