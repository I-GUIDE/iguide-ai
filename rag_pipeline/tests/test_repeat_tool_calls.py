"""A peer that re-issues the call it just made gets the earlier result back, not a re-run.

The live turn this is from (agent.i-guide.io, 2026-10-08 19:42 UTC, Lumen deepseek-v4-flash):
the analyze peer geocoded the same ten street intersections at seq 16, then issued the
IDENTICAL call at seq 17, 18, ... 30 (fourteen more), until LangGraph's recursion limit of 60
stopped the run. Each repeat ran the tool again and handed back the same JSON with nothing to
say it was a repeat, so the model's context looked exactly as it had the step before, and at
temperature 0 it chose the same call again.

These drive the REAL create_agent with the real middleware stack, because the middleware reads
the graph state the ToolNode hands it, which a stubbed handler would not have.
"""
from __future__ import annotations

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from agent_runtime import executor_factory as ef


class ToolAwareFake(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
        return self


CALLS = {"geocode": 0, "listing": 0}


@tool
def geocode(places: list) -> str:
    """Geocode places."""
    CALLS["geocode"] += 1
    return '{"results": [{"place": "%s", "lat": 41.8, "lon": -87.74}]}' % places[0]


@tool
def listing() -> str:
    """List files."""
    CALLS["listing"] += 1
    return f'{{"files": {CALLS["listing"]}}}'


@pytest.fixture(autouse=True)
def _reset():
    CALLS.update(geocode=0, listing=0)


def _call(name, args, i):
    return {"name": name, "args": args, "id": f"call_{i}", "type": "tool_call"}


def _agent(replies, checkpointer=None):
    return create_agent(
        model=ToolAwareFake(messages=iter(replies)),
        tools=[geocode, listing],
        system_prompt="test",
        middleware=ef._default_middleware(),
        checkpointer=checkpointer,
    )


def _tool_messages(out):
    return [m for m in out["messages"] if isinstance(m, ToolMessage)]


@pytest.mark.parametrize("checkpointed", [False, True])
def test_an_identical_call_back_to_back_is_not_run_again(checkpointed):
    """Checkpointed too, as the peers are: the repeat mark has to survive serialisation, or
    the second repeat would read the first as new work and run."""
    same = {"places": ["Cicero Ave 51st St"]}
    agent = _agent([
        AIMessage(content="", tool_calls=[_call("geocode", same, 1)]),
        AIMessage(content="", tool_calls=[_call("geocode", same, 2)]),
        AIMessage(content="", tool_calls=[_call("geocode", same, 3)]),
        AIMessage(content="done"),
    ], checkpointer=InMemorySaver() if checkpointed else None)
    out = agent.invoke({"messages": [HumanMessage(content="q")]},
                       config={"configurable": {"thread_id": "t::analysis"}})

    assert CALLS["geocode"] == 1
    first, second, third = _tool_messages(out)
    assert "41.8" in first.content and "same call" not in first.content
    # The repeat carries the earlier result AND says what it is.
    for repeat in (second, third):
        assert "41.8" in repeat.content
        assert "same call" in repeat.content
        assert repeat.additional_kwargs.get("repeat_of") == "call_1"
    assert "2 times" in second.content and "3 times" in third.content


def test_different_arguments_still_run():
    agent = _agent([
        AIMessage(content="", tool_calls=[_call("geocode", {"places": ["a"]}, 1)]),
        AIMessage(content="", tool_calls=[_call("geocode", {"places": ["b"]}, 2)]),
        AIMessage(content="done"),
    ])
    out = agent.invoke({"messages": [HumanMessage(content="q")]})
    assert CALLS["geocode"] == 2
    assert all("same call" not in m.content for m in _tool_messages(out))


def test_argument_order_does_not_make_a_call_new():
    agent = _agent([
        AIMessage(content="", tool_calls=[_call("geocode", {"places": ["a"]}, 1)]),
        AIMessage(content="", tool_calls=[_call("geocode", {"places": ["a"]}, 2)]),
        AIMessage(content="done"),
    ])
    agent.invoke({"messages": [HumanMessage(content="q")]})
    assert CALLS["geocode"] == 1


def test_a_call_after_new_work_runs_again():
    """`list_conversation_files` after a write is the case this protects: the same call with
    the same (no) arguments legitimately answers differently once another tool has run. So a
    repeat is only short-circuited when nothing new has run since the earlier result."""
    agent = _agent([
        AIMessage(content="", tool_calls=[_call("listing", {}, 1)]),
        AIMessage(content="", tool_calls=[_call("geocode", {"places": ["a"]}, 2)]),
        AIMessage(content="", tool_calls=[_call("listing", {}, 3)]),
        AIMessage(content="done"),
    ])
    out = agent.invoke({"messages": [HumanMessage(content="q")]})
    assert CALLS["listing"] == 2
    assert '"files": 2' in _tool_messages(out)[-1].content


def test_parallel_siblings_of_the_last_step_count_as_already_returned():
    """A step that issued two calls and then re-issued both: neither is new."""
    a, b = {"places": ["a"]}, {"places": ["b"]}
    agent = _agent([
        AIMessage(content="", tool_calls=[_call("geocode", a, 1), _call("geocode", b, 2)]),
        AIMessage(content="", tool_calls=[_call("geocode", a, 3), _call("geocode", b, 4)]),
        AIMessage(content="done"),
    ])
    agent.invoke({"messages": [HumanMessage(content="q")]})
    assert CALLS["geocode"] == 2


def test_a_new_invocation_on_the_same_thread_starts_a_new_run():
    """Peer threads outlive the turn and the supervisor re-invokes them with an observation
    (map retry, mismatch retry). A call repeated after that new human message is a new run's
    call and runs."""
    same = {"places": ["a"]}
    saver = InMemorySaver()
    agent = _agent([
        AIMessage(content="", tool_calls=[_call("geocode", same, 1)]),
        AIMessage(content="first"),
        AIMessage(content="", tool_calls=[_call("geocode", same, 2)]),
        AIMessage(content="second"),
    ], checkpointer=saver)
    cfg = {"configurable": {"thread_id": "t::analysis"}}
    agent.invoke({"messages": [HumanMessage(content="q1")]}, config=cfg)
    agent.invoke({"messages": [HumanMessage(content="q2")]}, config=cfg)
    assert CALLS["geocode"] == 2


def test_a_long_result_is_cut_and_says_where_the_rest_is():
    long = "x" * 10_000

    @tool
    def big() -> str:
        """Big."""
        return long

    agent = create_agent(
        model=ToolAwareFake(messages=iter([
            AIMessage(content="", tool_calls=[_call("big", {}, 1)]),
            AIMessage(content="", tool_calls=[_call("big", {}, 2)]),
            AIMessage(content="done"),
        ])),
        tools=[big], system_prompt="t", middleware=ef._default_middleware())
    out = agent.invoke({"messages": [HumanMessage(content="q")]})
    repeat = _tool_messages(out)[-1]
    assert len(repeat.content) < 4_000
    assert "call_1" in repeat.content


def test_a_model_that_ignores_the_observation_ends_the_run():
    """Measured against Lumen deepseek-v4-flash on the incident's own history: told that the
    call was a repeat, it repeated it again in 2 of 3 trials after one repeat and 3 of 3 after
    four. So the observation alone does not end the loop. The run ends at the fourth identical
    ask (the observation having been ignored twice), with an error that names the call, instead
    of fourteen steps later at the recursion limit."""
    same = {"places": ["Cicero Ave 51st St"]}
    agent = _agent([AIMessage(content="", tool_calls=[_call("geocode", same, i)])
                    for i in range(1, 9)] + [AIMessage(content="done")])
    with pytest.raises(ef.RepeatedToolCallError) as err:
        agent.invoke({"messages": [HumanMessage(content="q")]})
    assert CALLS["geocode"] == 1
    assert "geocode" in str(err.value) and "4 times" in str(err.value)


def test_the_stop_reaches_the_supervisor_as_a_peer_failure():
    """Through the same path a recursion-limit error takes, so the turn goes on to another
    peer, and the failure record says what happened."""
    from agent_runtime.supervisor import graph as g

    def analyze(_q, _ev, _st):
        raise ef.RepeatedToolCallError("geocode_places was called with the same arguments 4 times")

    box = {"i": 0}

    def decide(_s, _d):
        box["i"] += 1
        return "analyze" if box["i"] == 1 else "done"

    out = g.run_supervisor("q", decide_fn=decide, analyze_fn=analyze, search_fn=lambda q, s: [],
                           synthesize_fn=lambda *a, **k: "answer", do_rerank=False)
    (failure,) = out["peer_failures"]
    assert failure["peer"] == "analyze" and "RepeatedToolCallError" in failure["error"]
    assert not failure["fatal"]


FLAKY = {"n": 0}


@tool
def flaky(feature: str) -> str:
    """A service that fails, then answers."""
    FLAKY["n"] += 1
    if FLAKY["n"] == 1:
        return '{"error": "overpass_failed", "message": "504 Gateway Timeout", "count": 0}'
    return '{"count": 56}'


def test_a_call_that_failed_is_run_again():
    """Seen in the final local replay: every Overpass mirror answered 504, the peer retried the
    identical call, and the guard handed back the failure saying nothing had changed. For an
    outage that is false; a retry is how it gets through."""
    FLAKY["n"] = 0
    agent = create_agent(
        model=ToolAwareFake(messages=iter([
            AIMessage(content="", tool_calls=[_call("flaky", {"feature": "school"}, 1)]),
            AIMessage(content="", tool_calls=[_call("flaky", {"feature": "school"}, 2)]),
            AIMessage(content="done"),
        ])),
        tools=[flaky], system_prompt="t", middleware=ef._default_middleware())
    out = agent.invoke({"messages": [HumanMessage(content="q")]})
    assert FLAKY["n"] == 2
    assert _tool_messages(out)[-1].content == '{"count": 56}'
