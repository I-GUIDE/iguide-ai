"""Per-turn instrumentation: what each model call cost and which tools it used.

These drive the REAL create_agent with the real middleware, because the two defects this
instrumentation is most likely to have are both invisible to a stubbed handler:

  1. `ModelResponse.result` is a LIST of messages, not a message. Reading usage off the
     response (or off `result` directly) yields None on every call, and the instrumentation
     reports nothing forever while looking healthy.
  2. The thread id is reachable only through langchain's active-config contextvar, which only
     exists during a real invocation.
"""
from __future__ import annotations

import json
import logging

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from agent_runtime import executor_factory as ef


class ToolAwareFake(GenericFakeChatModel):
    """GenericFakeChatModel raises NotImplementedError on bind_tools, which create_agent calls
    whenever tools are supplied — and half of these cases exist precisely to bind tools."""

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
        return self


@tool
def alpha(city: str) -> str:
    """Look up a city."""
    return city


@tool
def beta(n: int) -> str:
    """Double a number."""
    return str(n * 2)


def _agent(reply, tools=None, extra_middleware=None):
    middleware = [ef._make_instrumentation_middleware()] + list(extra_middleware or [])
    replies = reply if isinstance(reply, list) else [reply]
    return create_agent(
        model=ToolAwareFake(messages=iter(replies)),
        tools=list(tools or []),
        system_prompt="you are a test",
        middleware=middleware,
    )


def _lines(caplog, prefix="turn_instrumentation "):
    out = []
    for rec in caplog.records:
        msg = rec.getMessage()
        if msg.startswith(prefix):
            out.append(json.loads(msg[len(prefix):]))
    return out


def _run(caplog, reply, tools=None, thread="T-1::analysis"):
    caplog.set_level(logging.INFO, logger="agent_runtime.executor_factory")
    agent = _agent(reply, tools=tools)
    config = {"configurable": {"thread_id": thread}} if thread else None
    agent.invoke({"messages": [HumanMessage(content="a question")]}, config=config)


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.delenv(ef.INSTRUMENTATION_ENV, raising=False)
    ef._TOOLSET_LOGGED.clear()
    ef._SCHEMA_COST_CACHE.clear()


def test_records_provider_usage_from_messages_inside_result(caplog):
    """The load-bearing one: real input tokens must be captured.

    Fails against the natural-but-wrong implementations — `response.usage_metadata` and
    `getattr(response.result, 'usage_metadata')` are both None/absent here.
    """
    reply = AIMessage(content="done", usage_metadata={
        "input_tokens": 9000, "output_tokens": 12, "total_tokens": 9012})
    _run(caplog, reply)
    lines = _lines(caplog)
    assert len(lines) == 1
    assert lines[0]["real_input_tokens"] == 9000
    assert lines[0]["output_tokens"] == 12
    assert "usage" not in lines[0]


def test_undercount_ratio_exposes_the_estimator_gap(caplog):
    """The ratio is the whole point: >1 means we undercounted, the direction that causes a 400."""
    reply = AIMessage(content="done", usage_metadata={
        "input_tokens": 8000, "output_tokens": 1, "total_tokens": 8001})
    _run(caplog, reply, tools=[alpha, beta])
    line = _lines(caplog)[0]
    est = line["est_message_tokens"] + line["schema_tokens"] + line["system_tokens"]
    assert est > 0
    assert line["undercount_ratio"] == pytest.approx(round(8000 / est, 3))
    assert line["undercount_ratio"] > 1  # the fake reports far more than the estimate


def test_ratio_denominator_includes_the_system_prompt(caplog):
    """The provider counts the system prompt, so the denominator must too.

    The magnitude turned out to be smaller than the guess that prompted this: the deployed
    default executor's prompt is 95 tokens, so including it moved the live ratio only 0.736 ->
    0.720. The fix is still required — a peer's prompt is an order of magnitude larger, and a
    denominator that does not mirror what the provider counted is not a measurement — but the
    real finding underneath was the SCHEMA estimate, which overstates by ~40% (4,397 estimated
    against ~3,130 actual for the same 24 tools)."""
    reply = AIMessage(content="ok", usage_metadata={
        "input_tokens": 500, "output_tokens": 1, "total_tokens": 501})
    _run(caplog, reply, tools=[alpha])
    line = _lines(caplog)[0]
    assert line["system_tokens"] > 0, "the system prompt must be counted, not assumed absent"
    naive = line["est_message_tokens"] + line["schema_tokens"]
    honest = naive + line["system_tokens"]
    assert line["undercount_ratio"] == pytest.approx(round(500 / honest, 3))
    assert line["undercount_ratio"] != pytest.approx(round(500 / naive, 3))


def test_absent_usage_is_labelled_not_silently_dropped(caplog):
    """A provider that reports no usage must be distinguishable from a read failure."""
    _run(caplog, AIMessage(content="done"))
    line = _lines(caplog)[0]
    assert line["usage"] == "absent"
    assert "real_input_tokens" not in line


def test_session_and_peer_are_split_from_the_child_thread_id(caplog):
    """Peer threads are '{thread}::{label}': prefix = CONVERSATION id, suffix = peer name.

    Named `session`, not `turn` — live lines carry "sess-4fa362d4-..." which every turn of a
    conversation shares, so it cannot group a turn and must not claim to.
    """
    _run(caplog, AIMessage(content="x"), thread="sess-abc::code")
    line = _lines(caplog)[0]
    assert line["session"] == "sess-abc"
    assert line["peer"] == "code"
    assert "turn" not in line, "the field must not claim to identify a turn"


def test_calls_are_sequenced_per_conversation_and_peer(caplog):
    """Without an ordinal, the lines of a multi-turn conversation cannot even be ordered."""
    caplog.set_level(logging.INFO, logger="agent_runtime.executor_factory")
    ef._CALL_SEQ.clear()
    for _ in range(3):
        agent = _agent(AIMessage(content="x"))
        agent.invoke({"messages": [HumanMessage(content="q")]},
                     config={"configurable": {"thread_id": "sess-1::analysis"}})
    other = _agent(AIMessage(content="x"))
    other.invoke({"messages": [HumanMessage(content="q")]},
                 config={"configurable": {"thread_id": "sess-1::code"}})
    lines = _lines(caplog)
    assert [ln["seq"] for ln in lines[:3]] == [1, 2, 3]
    assert lines[3]["peer"] == "code" and lines[3]["seq"] == 1, "a different peer counts separately"


def test_a_rejected_call_still_logs(caplog):
    """The provider 400 is the event this instrumentation most needs to capture.

    It used to log only after a SUCCESSFUL handler call, so a rejection produced no line —
    censoring the sample exactly at the tail that matters.
    """
    caplog.set_level(logging.INFO, logger="agent_runtime.executor_factory")

    @wrap_model_call
    def explode(request, handler):
        raise RuntimeError("Input tokens exceed the configured limit of 922000 tokens")

    agent = create_agent(model=ToolAwareFake(messages=iter([AIMessage(content="never")])),
                         tools=[alpha], system_prompt="s",
                         middleware=[ef._make_instrumentation_middleware(), explode])
    with pytest.raises(Exception):
        agent.invoke({"messages": [HumanMessage(content="q")]},
                     config={"configurable": {"thread_id": "sess-x::analysis"}})
    lines = _lines(caplog)
    assert len(lines) == 1, "the rejected call must still produce a line"
    assert lines[0]["error"] == "RuntimeError"
    assert "922000" in lines[0]["error_detail"]
    assert lines[0]["tools_bound"] == 1          # the payload it was rejected for is recorded
    assert "real_input_tokens" not in lines[0]


# Measured 2026-09-02 by probing every chat model the deployed key can see with a deliberately
# oversized request and reading the limit out of the provider's rejection. These are the ground
# truth this table exists to encode, so the test is the measurement, not a restatement of code.
MEASURED_WINDOWS = {
    "gpt-5.6-luna": 922_000, "gpt-5.6-sol": 922_000, "gpt-5.6-terra": 922_000,
    "gpt-5.5": 922_000, "gpt-5.4": 922_000,
    "gpt-5.4-mini": 272_000, "gpt-5.2": 272_000, "gpt-5-2025-08-07": 272_000,
    "gpt-5-mini-2025-08-07": 272_000, "gpt-5-nano-2025-08-07": 272_000,
    "gpt-4.1-2025-04-14": 1_047_576, "gpt-4.1-mini-2025-04-14": 1_047_576,
    "gpt-4.1-nano-2025-04-14": 1_047_576,
    "gpt-4o-2024-11-20": 128_000, "gpt-4o-2024-08-06": 128_000, "gpt-4o-mini": 128_000,
    "o4-mini-2025-04-16": 200_000, "o3-2025-04-16": 200_000, "o3-mini-2025-01-31": 200_000,
}


@pytest.mark.parametrize("model_id,window", sorted(MEASURED_WINDOWS.items()))
def test_every_measured_window_resolves_correctly(model_id, window):
    """Three of the table's original entries were wrong, two of them badly:
    gpt-4.1 was assumed 128,000 against a real 1,047,576 (8.2x), o4-mini 128,000 against
    200,000, and gpt-5.6-luna - the deployed default - took the 65,536 floor against 922,000."""
    class M:
        model_name = model_id

    assert ef._model_context_window(M()) == window


def test_the_mini_variant_does_not_inherit_its_family_window():
    """The specific trap prefix matching creates here: gpt-5.4-mini (272,000) extends
    gpt-5.4 (922,000). Reversed, every mini call is handed 3.4x its real window - a 400."""
    class Mini:
        model_name = "gpt-5.4-mini"

    class Full:
        model_name = "gpt-5.4"

    assert ef._model_context_window(Mini()) == 272_000
    assert ef._model_context_window(Full()) == 922_000


def test_an_unlisted_gpt5_lands_on_the_conservative_family_size():
    """A future gpt-5.x must waste context, not overrun: the bare gpt-5 entry is last in the
    family and carries 272,000, the SMALLER of the two observed tiers."""
    class M:
        model_name = "gpt-5.9-unreleased"

    assert ef._model_context_window(M()) == 272_000


def test_an_unlisted_model_warns_instead_of_silently_taking_the_floor(caplog):
    caplog.set_level(logging.WARNING, logger="agent_runtime.executor_factory")
    ef._UNKNOWN_WINDOW_WARNED.clear()

    class M:
        model_name = "some-brand-new-model-9"

    assert ef._model_context_window(M()) == ef._DEFAULT_WINDOW
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("not configured" in w and "some-brand-new-model-9" in w for w in warnings)
    # ...once per process, not once per call
    caplog.clear()
    ef._model_context_window(M())
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_a_specific_id_wins_over_its_family_prefix():
    """Ordering is load-bearing: gpt-5.6-luna must not be captured by a broader gpt-5 entry."""
    seen = set()
    for prefix, _ in ef._MODEL_WINDOWS:
        assert not any(prefix.startswith(p) for p in seen), \
            f"{prefix} is shadowed by an earlier, broader prefix"
        seen.add(prefix)


def test_supervisor_thread_without_a_label_is_named(caplog):
    _run(caplog, AIMessage(content="x"), thread="thread-abc")
    assert _lines(caplog)[0]["peer"] == "supervisor"


def test_bound_and_called_tools_are_both_recorded(caplog):
    """Bound-vs-called is what lets a human assign absent / present-not-chosen later."""
    # Two replies: a tool call, then the answer — a tool call sends the agent round the loop
    # for a second model call, so this also pins that each call gets its own line.
    replies = [AIMessage(content="", tool_calls=[{"name": "alpha", "args": {"city": "Urbana"},
                                                  "id": "c1"}]),
               AIMessage(content="Urbana")]
    _run(caplog, replies, tools=[alpha, beta])
    lines = _lines(caplog)
    assert len(lines) == 2
    assert [ln["tools_bound"] for ln in lines] == [2, 2]
    assert lines[0]["tools_called"] == ["alpha"]
    assert lines[1]["tools_called"] == []      # present-not-chosen on the second call
    # The second call carries the tool result, so it must cost more than the first.
    assert lines[1]["est_message_tokens"] > lines[0]["est_message_tokens"]


def test_toolset_catalogue_carries_per_tool_cost_and_is_logged_once(caplog):
    """Per-tool cost is what makes a pruning decision possible; once per fingerprint keeps it cheap."""
    caplog.set_level(logging.INFO, logger="agent_runtime.executor_factory")
    mw = ef._make_instrumentation_middleware()
    for _ in range(3):
        agent = create_agent(model=ToolAwareFake(messages=iter([AIMessage(content="x")])),
                             tools=[alpha, beta], system_prompt="s", middleware=[mw])
        agent.invoke({"messages": [HumanMessage(content="q")]},
                     config={"configurable": {"thread_id": "T::analysis"}})

    catalogue = _lines(caplog, "turn_instrumentation_toolset ")
    assert len(catalogue) == 1, "the catalogue must not repeat per call"
    assert set(catalogue[0]["per_tool"]) == {"alpha", "beta"}
    assert all(v > 0 for v in catalogue[0]["per_tool"].values())
    assert catalogue[0]["schema_tokens"] == sum(catalogue[0]["per_tool"].values())
    # ...while the per-call line still appears every time, keyed to that catalogue.
    calls = _lines(caplog)
    assert len(calls) == 3
    assert {c["toolset"] for c in calls} == {catalogue[0]["toolset"]}


def test_schema_cost_is_cached_per_fingerprint(caplog):
    """Rendering 86 schemas on every model call is real CPU on the hot path."""
    renders = {"n": 0}
    real = ef._schema_cost

    total_a, per_a = real([alpha, beta])
    assert total_a > 0 and len(per_a) == 2
    assert ef._toolset_fingerprint(["alpha", "beta"]) in ef._SCHEMA_COST_CACHE

    import langchain_core.utils.function_calling as fc
    original = fc.convert_to_openai_tool

    def counting(*a, **k):
        renders["n"] += 1
        return original(*a, **k)

    fc.convert_to_openai_tool = counting
    try:
        for _ in range(5):
            assert real([alpha, beta]) == (total_a, per_a)
        assert renders["n"] == 0, "a cached fingerprint must not re-render"
    finally:
        fc.convert_to_openai_tool = original


def test_fingerprint_is_order_independent_but_membership_sensitive():
    assert ef._toolset_fingerprint(["a", "b"]) == ef._toolset_fingerprint(["b", "a"])
    assert ef._toolset_fingerprint(["a", "b"]) != ef._toolset_fingerprint(["a", "b", "c"])


def test_env_switch_silences_it(caplog, monkeypatch):
    monkeypatch.setenv(ef.INSTRUMENTATION_ENV, "0")
    _run(caplog, AIMessage(content="x"), tools=[alpha])
    assert _lines(caplog) == []
    assert _lines(caplog, "turn_instrumentation_toolset ") == []


def test_a_broken_recorder_never_breaks_the_turn(caplog, monkeypatch):
    """Instrumentation is diagnostics: it must not be able to fail a user's request."""
    monkeypatch.setattr(ef, "_schema_cost", lambda tools: (_ for _ in ()).throw(RuntimeError("boom")))
    caplog.set_level(logging.INFO, logger="agent_runtime.executor_factory")
    agent = _agent(AIMessage(content="still answered"), tools=[alpha])
    out = agent.invoke({"messages": [HumanMessage(content="q")]},
                       config={"configurable": {"thread_id": "T::analysis"}})
    assert out["messages"][-1].content == "still answered"


def test_instrumentation_is_innermost_in_the_default_stack():
    """First handler is outermost, so instrumentation must be LAST to see the final payload."""
    names = [m.name for m in ef._default_middleware()]
    assert names == ["repair_history", "budget_context", "instrument"], names
