"""Every model call inside a traced turn reports its token usage, wired or not.

The supervisor's decider, synthesis and audit call the model directly, with no callbacks
passed. Before `UsageCallbackHandler`, those calls reported nothing, so a turn's cost counted
only the peers. These tests call a model the way those nodes do: bare `invoke`, no config.
"""
from __future__ import annotations

import contextvars
import threading

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from agent_runtime.streaming_trace import emit_trace_event, trace_context


def _model(usage=None):
    msg = AIMessage(content="ok", usage_metadata=usage) if usage else AIMessage(content="ok")
    return GenericFakeChatModel(messages=iter([msg]))


def _usage_events(events):
    return [e["data"] for e in events if e.get("event") == "llm_usage"]


def test_a_bare_invoke_reports_its_usage():
    events = []
    usage = {"input_tokens": 1200, "output_tokens": 34, "total_tokens": 1234,
             "input_token_details": {"cache_read": 1000},
             "output_token_details": {"reasoning": 20}}
    with trace_context(events.append, agent_dev=True):
        _model(usage).invoke("hello")
    got = _usage_events(events)
    assert len(got) == 1
    assert got[0]["input_tokens"] == 1200
    assert got[0]["output_tokens"] == 34
    assert got[0]["cached_input_tokens"] == 1000
    assert got[0]["reasoning_tokens"] == 20


def test_a_call_on_a_worker_thread_that_copied_the_context_reports():
    events = []
    usage = {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
    with trace_context(events.append, agent_dev=True):
        ctx = contextvars.copy_context()
        t = threading.Thread(target=lambda: ctx.run(_model(usage).invoke, "x"))
        t.start()
        t.join()
    assert [e["input_tokens"] for e in _usage_events(events)] == [7]


def test_missing_usage_is_reported_as_absent_not_zero():
    events = []
    with trace_context(events.append, agent_dev=True):
        _model().invoke("hello")
    got = _usage_events(events)
    assert len(got) == 1
    assert got[0].get("usage") == "absent"
    assert "input_tokens" not in got[0]


def test_outside_a_traced_turn_nothing_is_emitted_and_nothing_breaks():
    # No trace context: the handler is not installed, the call still works.
    assert _model({"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}).invoke("x").content == "ok"
    emit_trace_event("llm_usage", {})  # no active stream: a no-op


def test_usage_is_detail_tier():
    # A cost is a developer's question; a status-only client does not get it.
    events = []
    with trace_context(events.append, agent_dev=False):
        _model({"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}).invoke("x")
    assert _usage_events(events) == []
