"""A repeated call is answered from the turn's event log, and a run that stops adding ends.

The live turn behind this (agent.i-guide.io, 2026-10-08 19:42 UTC, Lumen deepseek-v4-flash):
the analyze peer geocoded the same ten street intersections at seq 16, then issued the
IDENTICAL call fourteen more times until LangGraph's recursion limit of 60 stopped the run.
Stage 38 answered a repeat only within the latest executed step and ended the run at the fourth
identical ask. Stage 42 replaces both counts with the turn log: an identical call is answered
from the log for the whole turn, across peer runs, while nothing new has happened since; and two
consecutive steps that add nothing end the run, whatever the calls were.

These drive the REAL create_agent with the real middleware stack, inside a supervisor turn
(`turn_log.bind`), because the middleware reads the active log that the peer node binds.
"""
from __future__ import annotations

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from agent_runtime import executor_factory as ef
from agent_runtime import turn_log


class ToolAwareFake(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
        return self


class RecordingFake(ToolAwareFake):
    """Keeps the system message of every request, to see what the model was given."""

    seen_system: list = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
        RecordingFake.seen_system.append(
            next((m.content for m in messages if isinstance(m, SystemMessage)), ""))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


CALLS = {"geocode": 0, "listing": 0, "write": 0, "open": 0}


@tool
def geocode(places: list) -> str:
    """Geocode places."""
    CALLS["geocode"] += 1
    return '{"results": [{"place": "%s", "lat": 41.8, "lon": -87.74}]}' % places[0]


@tool
def listing() -> str:
    """List files."""
    CALLS["listing"] += 1
    return f'{{"files": {CALLS["write"]}}}'


@tool
def write(name: str) -> str:
    """Write a file."""
    CALLS["write"] += 1
    return f'{{"ok": true, "file_id": "file_{CALLS["write"]:012x}"}}'


@tool
def open_boundary(path: str) -> str:
    """Open a file by path."""
    CALLS["open"] += 1
    return f'{{"ok": false, "error": "DataSourceError: \'/work/{path}\': No such file or directory"}}'


@pytest.fixture(autouse=True)
def _reset():
    CALLS.update(geocode=0, listing=0, write=0, open=0)
    RecordingFake.seen_system = []


def _call(name, args, i):
    return {"name": name, "args": args, "id": f"call_{i}", "type": "tool_call"}


def _agent(replies, checkpointer=None, model_cls=ToolAwareFake):
    return create_agent(
        model=model_cls(messages=iter(replies)),
        tools=[geocode, listing, write, open_boundary],
        system_prompt="test",
        middleware=ef._default_middleware(),
        checkpointer=checkpointer,
    )


def _invoke(agent, log, peer="analyze", brief=None, config=None, text="q"):
    with turn_log.bind(log, peer, brief) as b:
        out = agent.invoke({"messages": [HumanMessage(content=text)]}, config=config)
    return out, b.run


def _tool_messages(out):
    return [m for m in out["messages"] if isinstance(m, ToolMessage)]


def _final(out):
    return out["messages"][-1].content


@pytest.mark.parametrize("checkpointed", [False, True])
def test_an_identical_call_is_answered_from_the_log_and_the_loop_ends(checkpointed):
    """The incident's shape: the same call over and over. It runs once, the repeats get the
    earlier result and say so, and the run ends after two steps that added nothing instead of
    at the recursion limit."""
    same = {"places": ["Cicero Ave 51st St"]}
    log = turn_log.new_log()
    agent = _agent([AIMessage(content="", tool_calls=[_call("geocode", same, i)])
                    for i in range(1, 15)] + [AIMessage(content="done")],
                   checkpointer=InMemorySaver() if checkpointed else None)
    out, run = _invoke(agent, log, config={"configurable": {"thread_id": "t::analysis"}})

    assert CALLS["geocode"] == 1
    first, *repeats = _tool_messages(out)
    assert "41.8" in first.content and "Same call" not in first.content
    assert len(repeats) == turn_log.PROGRESS_LIMIT
    for r in repeats:
        assert "41.8" in r.content and "Same call" in r.content
        assert r.additional_kwargs.get("repeat_of") == "call_1"
    assert "Stopped" in _final(out)
    assert log.run_was_productive(run)        # the first step did add something


def test_different_arguments_still_run():
    log = turn_log.new_log()
    agent = _agent([
        AIMessage(content="", tool_calls=[_call("geocode", {"places": ["a"]}, 1)]),
        AIMessage(content="", tool_calls=[_call("geocode", {"places": ["b"]}, 2)]),
        AIMessage(content="done"),
    ])
    out, _ = _invoke(agent, log)
    assert CALLS["geocode"] == 2
    assert all("Same call" not in m.content for m in _tool_messages(out))


def test_argument_order_does_not_make_a_call_new():
    log = turn_log.new_log()
    agent = _agent([
        AIMessage(content="", tool_calls=[_call("geocode", {"places": ["a"], "k": 1}, 1)]),
        AIMessage(content="", tool_calls=[_call("geocode", {"k": 1, "places": ["a"]}, 2)]),
        AIMessage(content="done"),
    ])
    _invoke(agent, log)
    assert CALLS["geocode"] == 1


def test_a_listing_after_a_write_runs_again():
    """Stage 38's reason for a narrow window still holds: once something new has happened, the
    same call can legitimately answer differently, so it runs."""
    log = turn_log.new_log()
    agent = _agent([
        AIMessage(content="", tool_calls=[_call("listing", {}, 1)]),
        AIMessage(content="", tool_calls=[_call("write", {"name": "a"}, 2)]),
        AIMessage(content="", tool_calls=[_call("listing", {}, 3)]),
        AIMessage(content="done"),
    ])
    out, _ = _invoke(agent, log)
    assert CALLS["listing"] == 2
    assert '"files": 1' in _tool_messages(out)[-1].content


def test_parallel_siblings_of_a_step_count_as_already_returned():
    a, b = {"places": ["a"]}, {"places": ["b"]}
    log = turn_log.new_log()
    agent = _agent([
        AIMessage(content="", tool_calls=[_call("geocode", a, 1), _call("geocode", b, 2)]),
        AIMessage(content="", tool_calls=[_call("geocode", a, 3), _call("geocode", b, 4)]),
        AIMessage(content="done"),
    ])
    _invoke(agent, log)
    assert CALLS["geocode"] == 2


def test_an_alternation_is_caught_too():
    """A-B-A-B never repeats the latest step, which is why stage 38's window let it through.
    The second cycle adds nothing, so it is answered from the log and the run ends."""
    a, b = {"places": ["a"]}, {"places": ["b"]}
    log = turn_log.new_log()
    replies = []
    for i in range(6):
        replies.append(AIMessage(content="", tool_calls=[
            _call("geocode", a if i % 2 == 0 else b, i + 1)]))
    agent = _agent(replies + [AIMessage(content="done")])
    out, _ = _invoke(agent, log)
    assert CALLS["geocode"] <= 3
    assert "Stopped" in _final(out)


def test_a_repeat_in_a_later_run_of_the_turn_is_answered_from_the_log():
    """Every retry observation and every peer re-run started stage 38's window afresh. The
    log belongs to the turn, so a later run, even of another peer, finds the answer."""
    same = {"places": ["a"]}
    log = turn_log.new_log()
    first = _agent([AIMessage(content="", tool_calls=[_call("geocode", same, 1)]),
                    AIMessage(content="first")])
    _invoke(first, log, peer="analyze")
    second = _agent([AIMessage(content="", tool_calls=[_call("geocode", same, 2)]),
                     AIMessage(content="second")])
    out, _ = _invoke(second, log, peer="code")
    assert CALLS["geocode"] == 1
    assert _tool_messages(out)[-1].additional_kwargs.get("repeat_of") == "call_1"


def test_varied_calls_failing_the_same_way_end_the_run():
    """Guessing filenames: every call differs, so no memo matches, and a step count would let
    it run to the recursion limit. Each failure is the same error class, so after the first
    one nothing new is learned, and the run ends."""
    log = turn_log.new_log()
    agent = _agent([AIMessage(content="", tool_calls=[_call("open_boundary", {"path": f"f{i}.geojson"}, i)])
                    for i in range(1, 12)] + [AIMessage(content="done")])
    out, run = _invoke(agent, log)
    assert CALLS["open"] == 1 + turn_log.PROGRESS_LIMIT
    assert "Stopped" in _final(out)
    assert not log.run_was_productive(run)


def test_a_new_error_is_progress_while_debugging():
    """Fix, re-run, a different error: that is debugging, and it continues."""
    log = turn_log.new_log()
    errors = iter(["NameError: x", "KeyError: 'area'", "ok"])

    @tool
    def run_code(code: str) -> str:
        """Run code."""
        e = next(errors)
        return '{"ok": true, "stdout": "42"}' if e == "ok" else f'{{"ok": false, "error": "{e}"}}'

    agent = create_agent(
        model=ToolAwareFake(messages=iter(
            [AIMessage(content="", tool_calls=[_call("run_code", {"code": f"v{i}"}, i)])
             for i in range(1, 4)] + [AIMessage(content="done: 42")])),
        tools=[run_code], system_prompt="t", middleware=ef._default_middleware())
    out, _ = _invoke(agent, log)
    assert _final(out) == "done: 42"


FLAKY = {"n": 0}


@tool
def flaky(feature: str) -> str:
    """A service that fails, then answers."""
    FLAKY["n"] += 1
    if FLAKY["n"] == 1:
        return '{"error": "overpass_failed", "message": "504 Gateway Timeout", "count": 0}'
    return '{"count": 56}'


def test_a_call_that_failed_is_run_again_once():
    """Every Overpass mirror answered 504 in stage 38's final replay; an identical retry is how
    an outage gets through, so a failure is retried once before it becomes the answer."""
    FLAKY["n"] = 0
    log = turn_log.new_log()
    agent = create_agent(
        model=ToolAwareFake(messages=iter([
            AIMessage(content="", tool_calls=[_call("flaky", {"feature": "school"}, 1)]),
            AIMessage(content="", tool_calls=[_call("flaky", {"feature": "school"}, 2)]),
            AIMessage(content="done"),
        ])),
        tools=[flaky], system_prompt="t", middleware=ef._default_middleware())
    out, _ = _invoke(agent, log)
    assert FLAKY["n"] == 2
    assert _tool_messages(out)[-1].content == '{"count": 56}'


def test_a_failure_repeated_past_its_retry_is_answered_from_the_log():
    log = turn_log.new_log()
    same = {"path": "Champaign_County.geojson"}
    agent = _agent([AIMessage(content="", tool_calls=[_call("open_boundary", same, i)])
                    for i in range(1, 6)] + [AIMessage(content="done")])
    _invoke(agent, log)
    assert CALLS["open"] == 1 + turn_log.FAILED_RETRIES


def test_a_long_result_is_cut_and_says_where_the_rest_is():
    long = "x" * 10_000

    @tool
    def big() -> str:
        """Big."""
        return long

    log = turn_log.new_log()
    agent = create_agent(
        model=ToolAwareFake(messages=iter([
            AIMessage(content="", tool_calls=[_call("big", {}, 1)]),
            AIMessage(content="", tool_calls=[_call("big", {}, 2)]),
            AIMessage(content="done"),
        ])),
        tools=[big], system_prompt="t", middleware=ef._default_middleware())
    out, _ = _invoke(agent, log)
    repeat = _tool_messages(out)[-1]
    assert len(repeat.content) < 4_000
    assert "call_1" in repeat.content


def test_the_task_and_plan_ride_in_the_system_message():
    """On a retry the latest human message is a bare observation, and the context budget keeps
    only that message plus what recent tail fits, so the task itself could be trimmed away. The
    brief is in the system message, which is never trimmed."""
    log = turn_log.new_log()
    agent = _agent([AIMessage(content="", tool_calls=[_call("geocode", {"places": ["a"]}, 1)]),
                    AIMessage(content="done")], model_cls=RecordingFake)
    brief = "Task (the user's request, verbatim): schools within 1 mile of the site"
    _invoke(agent, log, brief=brief, text="The last call failed; try again.")
    assert RecordingFake.seen_system and all(brief in s for s in RecordingFake.seen_system)


def test_outside_a_supervisor_turn_every_call_runs():
    """No active log, no memo: a bare create_agent (a script, a test) behaves as before."""
    same = {"places": ["a"]}
    agent = _agent([AIMessage(content="", tool_calls=[_call("geocode", same, i)])
                    for i in range(1, 4)] + [AIMessage(content="done")])
    agent.invoke({"messages": [HumanMessage(content="q")]})
    assert CALLS["geocode"] == 3
