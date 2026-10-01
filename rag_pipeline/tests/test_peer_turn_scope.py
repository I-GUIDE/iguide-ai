"""A peer's own work must be scoped to the turn that did it.

A peer thread OUTLIVES the turn. The supervisor graph is compiled without a checkpointer, but
each peer's create_agent graph shares the process-global DEFAULT_CHECKPOINTER under a stable
child id, and the map UI sends a required thread_id. So extract_search_artifacts walks every
prior turn's messages and four verifiers that ask "what happened THIS turn?" are handed the
whole conversation.

Of 50 test files, none drove two turns against a stable peer thread — which is why this shipped
and why two in-code comments and AGENTS.md all asserted the opposite ("the search peer starts on
a fresh thread every turn"). Under multiple workers that claim is intermittently TRUE, since the
checkpointer is process-global, so it was observed rather than invented.

These tests drive the REAL create_agent graph, the REAL BoundedInMemorySaver and the real invoke
helper. The existing peer doubles cannot cover this: they fake a non-cumulative thread with
hand-built messages whose id is None, so the prefix guard declines to slice and they pass either
way. A green existing suite is not verification here.
"""
from __future__ import annotations

import json
from typing import Any, List

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from agent_runtime.executor_factory import BoundedInMemorySaver, PeerSession
from agent_runtime.runtime_utils import extract_search_artifacts


class _Scripted(BaseChatModel):
    """Replays a fixed script, advancing ACROSS turns like a real conversation."""

    script: List[Any] = []
    seen: List[int] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kw):        # create_agent binds; keep replaying the script
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw) -> ChatResult:
        i = len(self.seen)
        self.seen.append(i)
        return ChatResult(generations=[ChatGeneration(
            message=self.script[min(i, len(self.script) - 1)])])


@tool
def execute_code(code: str) -> str:
    """Run code."""
    return json.dumps({"ok": True, "exit_code": 0, "stdout": "42"})


@tool
def regionalize(file_id: str) -> str:
    """Cluster contiguous regions."""
    return json.dumps({"ok": False, "error": "ValueError: needs a weights matrix"})


@tool
def summary_statistics(file_id: str) -> str:
    """Summarise a table."""
    return json.dumps({"ok": True, "rows": 10})


def _call(name, args, cid):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": cid}])


def _harness(script, tools):
    model = _Scripted(script=script)
    from langchain.agents import create_agent

    ex = create_agent(model=model, tools=tools, checkpointer=BoundedInMemorySaver())
    return ex, {"configurable": {"thread_id": "conv::peer"}}, model


# --- the mechanism, against the real stack -----------------------------------------------

def test_a_prior_turns_tool_call_is_not_this_turns():
    """The headline defect: turn 2 ships a bare code fence and looks like it ran code."""
    ex, cfg, _ = _harness([
        _call("execute_code", {"code": "print(42)"}, "t1"),
        AIMessage(content="Ran it; output was 42."),
        AIMessage(content="Here you go:\n```python\nprint(1)\n```"),
    ], [execute_code])

    t1 = PeerSession(ex, cfg).run("run print(42)")
    assert [c["name"] for c in t1.artifacts["tool_calls"]] == ["execute_code"]

    t2 = PeerSession(ex, cfg).run("show me some code")
    assert len(t2.resp["messages"]) > len(t1.resp["messages"]), "the thread must accumulate"

    # the defect, still reproducible through the unscoped extractor
    unscoped = extract_search_artifacts(t2.resp)["tool_calls"]
    assert [c["name"] for c in unscoped] == ["execute_code"], "control: defect not reproduced"

    # the fix
    assert t2.artifacts["tool_calls"] == []
    assert t2.artifacts["tool_results"] == []


def test_the_answer_is_scoped_too():
    ex, cfg, _ = _harness([
        AIMessage(content="First answer."),
        AIMessage(content="Second answer."),
    ], [execute_code])
    assert PeerSession(ex, cfg).run("q1").answer == "First answer."
    assert PeerSession(ex, cfg).run("q2").answer == "Second answer."


def test_a_later_turn_does_not_harvest_the_earlier_turns_result():
    """The search peer's evidence harvest reads tool_results the same way."""
    ex, cfg, _ = _harness([
        _call("summary_statistics", {"file_id": "f1"}, "s1"),
        AIMessage(content="10 rows."),
        AIMessage(content="Nothing to look up for that."),
    ], [summary_statistics])

    PeerSession(ex, cfg).run("summarise f1")
    t2 = PeerSession(ex, cfg).run("what is a shapefile?")
    assert t2.artifacts["tool_results"] == []


# --- per INVOCATION, not per turn ---------------------------------------------------------

def test_a_retry_within_one_turn_is_not_double_counted():
    """default_analyze_fn invokes the same thread up to 3x and CONCATENATES the slices.

    A per-turn watermark would re-yield the first invocation on every retry, so two failures of
    one tool would render as four and trip _TOOL_FAIL_REPEATS on their own.
    """
    ex, cfg, _ = _harness([
        _call("regionalize", {"file_id": "f1"}, "r1"),
        _call("regionalize", {"file_id": "f1"}, "r2"),
        AIMessage(content="regionalize keeps failing."),
        _call("summary_statistics", {"file_id": "f1"}, "s1"),
        AIMessage(content="Used summary statistics instead."),
    ], [regionalize, summary_statistics])

    session = PeerSession(ex, cfg)          # ONE session for the whole turn
    first = session.run("cluster f1")
    retry = session.run("regionalize failed repeatedly this turn; try another route")

    assert [c["name"] for c in first.artifacts["tool_calls"]] == ["regionalize", "regionalize"]
    assert [c["name"] for c in retry.artifacts["tool_calls"]] == ["summary_statistics"], \
        "the retry must report only its OWN calls"
    # the turn as a whole, with nothing counted twice
    assert [c["name"] for c in session.turn_artifacts["tool_calls"]] == \
        ["regionalize", "regionalize", "summary_statistics"]


# --- the guard -----------------------------------------------------------------------------

def test_a_rewritten_history_falls_back_instead_of_cutting_this_turn():
    """If something rewrote the thread, prefer today's behaviour over losing real records."""
    from agent_runtime.executor_factory import _prefix_is_intact

    class _M:
        def __init__(self, i):
            self.id = i

    assert _prefix_is_intact([_M("a"), _M("b"), _M("c")], ["a", "b"])
    assert not _prefix_is_intact([_M("z"), _M("b")], ["a", "b"])       # rewritten
    assert not _prefix_is_intact([_M(None), _M(None)], [None])         # the old test doubles
    assert not _prefix_is_intact([_M("a")], [])                        # nothing seen yet
    assert not _prefix_is_intact([_M("a")], ["a", "b"])                # shorter than the mark


def test_an_unknowable_thread_is_simply_unscoped():
    from agent_runtime.executor_factory import peer_thread_messages

    class _NoState:
        pass

    assert peer_thread_messages(_NoState(), {"configurable": {"thread_id": "x"}}) == []
    assert peer_thread_messages(None, None) == []


def test_a_first_turn_needs_no_watermark():
    ex, cfg, _ = _harness([AIMessage(content="Hello.")], [execute_code])
    session = PeerSession(ex, cfg)
    assert session._seen == 0
    assert session.run("hi").answer == "Hello."
