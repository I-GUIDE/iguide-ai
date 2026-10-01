"""A ceiling on the outgoing model payload.

Nothing measured it before: a peer thread accumulates every tool result for the life of a
conversation (BoundedInMemorySaver caps THREADS, not messages inside one), and the deployed
default has no fallback, so the overflow reached the user as a raw provider error —
"Input length (66,275) exceeds model's maximum context length (65,536)". One clay turn
reached 199,605.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately

from agent_runtime import executor_factory as ef


class _Model:
    def __init__(self, model_name="gpt-oss:120b"):
        self.model_name = model_name


class _Req:
    """The shape a real ModelRequest has — and the reason the accounting bug survived.

    `messages` is annotated "excluding system message" upstream, and `tools` is a sibling
    field: both the system prompt and every tool schema are added downstream in langchain's
    model node. The old stub carried ONLY `messages`, and it put a SystemMessage inside them,
    so a middleware that pinned SystemMessages and counted only `messages` looked correct here
    while being a no-op that undercounted the real payload by the larger half.
    """

    def __init__(self, messages, *, system_message=None, tools=None, model=None):
        self.messages = messages
        self.system_message = system_message
        self.tools = list(tools or [])
        self.model = model or _Model()

    def override(self, messages):
        return _Req(messages, system_message=self.system_message,
                    tools=self.tools, model=self.model)


def _send(messages, budget, monkeypatch, **req_kw):
    """Drive the middleware with an EXPLICIT ceiling, via the env override."""
    monkeypatch.setenv(ef.CONTEXT_BUDGET_ENV, str(budget))
    mw = ef._make_context_budget_middleware()
    seen = {}
    mw.wrap_model_call(_Req(list(messages), **req_kw),
                       lambda r: seen.setdefault("out", r.messages))
    return seen["out"]


def _geojson_blob(tracts=12):
    """A tool result shaped like what this agent actually moves around.

    The fixture used to be "x" * 4000, which is the single most FAVOURABLE content possible for
    a chars/4 estimator: measured, count_tokens_approximately OVERcounts it 2.01x. Real payloads
    go the other way — GeoJSON boundaries undercount 1.72x and per-zone embedding vectors 2.25x
    (o200k_base) — so the budget believes it is under while the provider sees up to 2.25x more.
    A test built on "x" * n can never exercise that, which is why it passed while the estimator
    was the actual overflow cause.
    """
    import json
    import random

    random.seed(7)
    return json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature",
         "properties": {"GEOID": f"170190{i:04d}", "NAME": f"Census Tract {i}"},
         "geometry": {"type": "Polygon", "coordinates": [[
             [-88.3 + random.random() / 100, 40.1 + random.random() / 100]
             for _ in range(40)]]}}
        for i in range(tracts)]})


def _fat_thread(pairs=40, chars=4000, realistic=True):
    body = _geojson_blob() if realistic else "x" * chars
    msgs = [SystemMessage("you are an agent"), HumanMessage("the original question")]
    for i in range(pairs):
        msgs.append(AIMessage(content="", tool_calls=[{"name": "t", "id": f"c{i}", "args": {}}]))
        msgs.append(ToolMessage(content=body, name="t", tool_call_id=f"c{i}"))
    msgs.append(HumanMessage("what resolution was that?"))
    return msgs


def test_the_estimator_undercounts_the_payloads_this_agent_actually_moves():
    """Pins the mechanism, so the budget's headroom is never mistaken for real headroom.

    Not a hypothesis: geospatial tool results tokenize far worse than chars/4 predicts, and the
    budget middleware is built on that estimator. Skipped where tiktoken is unavailable, since
    the point is the comparison against a real tokenizer.
    """
    tiktoken = pytest.importorskip("tiktoken")
    enc = tiktoken.get_encoding("o200k_base")

    blob = _geojson_blob(30)
    approx = count_tokens_approximately([("user", blob)])
    real = len(enc.encode(blob))
    assert approx < real, "GeoJSON must undercount; if this flips, revisit the budget maths"
    assert real / approx > 1.5, f"undercount only {real / approx:.2f}x — recheck the fixture"

    # and the reason the old fixture hid it
    filler = "x" * 4000
    assert count_tokens_approximately([("user", filler)]) > len(enc.encode(filler)), \
        "a filler string OVERcounts, so a thread built from it cannot exercise the undercount"


def test_an_oversized_thread_is_brought_under_budget(monkeypatch):
    msgs = _fat_thread()
    assert count_tokens_approximately(msgs) > 5000
    out = _send(msgs, 5000, monkeypatch)
    assert count_tokens_approximately(out) <= 5000
    assert len(out) < len(msgs)


def test_the_current_question_always_survives(monkeypatch):
    """The system prompt is not in `messages` at all, so it cannot be trimmed — and the old
    "pin every SystemMessage" behaviour was a no-op. What must survive is the last HUMAN
    message: the question, whose loss makes the call pointless."""
    out = _send(_fat_thread(), 3000, monkeypatch,
                system_message=SystemMessage("you are an agent"))
    humans = [m for m in out if isinstance(m, HumanMessage)]
    assert humans, "the question must survive"
    assert humans[-1].content == "what resolution was that?"


def test_the_system_prompt_and_tool_schemas_are_counted(monkeypatch):
    """They are the larger half of the analyze peer's request and were counted as zero."""
    tool = {"type": "function", "function": {
        "name": "t", "description": "d" * 3000, "parameters": {"type": "object", "properties": {}}}}
    bare = ef._overhead_tokens(_Req([]))
    loaded = ef._overhead_tokens(_Req([], system_message=SystemMessage("s" * 4000), tools=[tool]))
    assert bare == 0
    assert loaded > 1000, f"overhead measured only {loaded} tokens"


def test_the_ceiling_is_derived_per_model(monkeypatch):
    """A flat budget was wrong in both directions: it discarded context that fits in gpt-4o's
    128k while leaving gpt-oss's 65,536 to be overrun by the schemas nothing counted."""
    monkeypatch.delenv(ef.CONTEXT_BUDGET_ENV, raising=False)
    monkeypatch.delenv("AGENT_MODEL_CONTEXT_WINDOW", raising=False)
    small = ef._derive_budget(_Req([], model=_Model("gpt-oss:120b")))
    large = ef._derive_budget(_Req([], model=_Model("gpt-4o-2024-11-20")))
    assert large > small, f"gpt-4o ({large}) must get more room than gpt-oss ({small})"
    assert small < 65_536 and large < 128_000, "the output reserve must come off the top"


def test_the_bound_tools_shrink_the_ceiling(monkeypatch):
    """Per PEER, not just per model: the analyze peer binds far more schemas than the code peer."""
    monkeypatch.delenv(ef.CONTEXT_BUDGET_ENV, raising=False)
    fat = [{"type": "function", "function": {
        "name": f"t{i}", "description": "d" * 2000,
        "parameters": {"type": "object", "properties": {}}}} for i in range(20)]
    assert ef._derive_budget(_Req([], tools=fat)) < ef._derive_budget(_Req([]))


def test_an_unknown_model_gets_the_observed_conservative_window(monkeypatch):
    """Guessing high produces the 400 this mechanism exists to prevent."""
    monkeypatch.delenv("AGENT_MODEL_CONTEXT_WINDOW", raising=False)
    assert ef._model_context_window(_Model("some-new-model")) == ef._DEFAULT_WINDOW
    monkeypatch.setenv("AGENT_MODEL_CONTEXT_WINDOW", "200000")
    assert ef._model_context_window(_Model("some-new-model")) == 200_000


def test_the_oldest_messages_go_first(monkeypatch):
    """A follow-up is nearly always about recent work, and the turn ledger re-injects the
    older facts at ~130 tokens instead of the thousands the raw tool results cost."""
    # Sized from measurement: one realistic GeoJSON tool result is ~5,366 approx tokens, ~9,658
    # after the payload-safety markup, so a ceiling has to be tens of thousands for ANY of them
    # to fit. The old 6,000 was sized for "x" * 4000 filler and now keeps nothing but the
    # question — which is correct behaviour, and no longer tests ordering.
    out = _send(_fat_thread(pairs=20), 34_000, monkeypatch)
    kept_ids = {getattr(m, "tool_call_id", None) for m in out}
    assert "c0" not in kept_ids            # oldest dropped
    assert "c19" in kept_ids               # newest kept


def test_a_thread_under_budget_is_passed_through_untouched(monkeypatch):
    msgs = [SystemMessage("s"), HumanMessage("hello")]
    assert _send(msgs, 5000, monkeypatch) == msgs


def test_the_trimmed_list_has_no_orphaned_tool_results(monkeypatch):
    """A ToolMessage whose AIMessage tool_call was trimmed away is a 400 on most providers."""
    out = _send(_fat_thread(), 3000, monkeypatch)
    call_ids = {c["id"] for m in out for c in (getattr(m, "tool_calls", None) or [])}
    for m in out:
        tcid = getattr(m, "tool_call_id", None)
        if tcid is not None:
            assert tcid in call_ids, f"orphaned tool result {tcid}"


def test_trimming_can_be_disabled(monkeypatch):
    """A NEGATIVE value disables it. 0 / unset now means "derive from the model", so the old
    "0 disables" reading would have silently switched trimming off for every deployment that
    had set it to zero."""
    msgs = _fat_thread()
    monkeypatch.setenv(ef.CONTEXT_BUDGET_ENV, "-1")
    mw = ef._make_context_budget_middleware()
    seen = {}
    mw.wrap_model_call(_Req(list(msgs)), lambda r: seen.setdefault("out", r.messages))
    assert len(seen["out"]) == len(msgs)


def test_the_budget_env_tolerates_junk(monkeypatch):
    """Junk falls back to derivation rather than to a flat number or a crash."""
    monkeypatch.setenv(ef.CONTEXT_BUDGET_ENV, "not-a-number")
    assert ef._context_budget() == 0                      # 0 == derive
    monkeypatch.delenv("AGENT_MODEL_CONTEXT_WINDOW", raising=False)
    assert ef._derive_budget(_Req([])) > 4_000


def test_the_counter_is_the_approximate_one_not_the_model_one():
    """ChatOpenAI.get_num_tokens_from_messages raises NotImplementedError for the AnvilGPT
    model ids this deployment defaults to, so an exact counter breaks the default provider."""
    import inspect

    src = inspect.getsource(ef._make_context_budget_middleware)
    assert "count_tokens_approximately" in src
    assert "get_num_tokens" not in src


def test_it_is_wired_into_every_agent():
    """Assert on the real stack, not on source text.

    This used to grep build_agent_executor for the factory call, which would pass just as
    happily on a stack that was assembled and then never handed to create_agent.
    """
    import inspect

    names = [m.name for m in ef._default_middleware()]
    assert "budget_context" in names
    # Repair must run FIRST so the budgeter always sees a coherent tool-call list. First
    # handler is the outermost wrapper, so list order is wrapping order.
    assert names.index("repair_history") < names.index("budget_context")
    # ...and the stack the executor actually passes is that one.
    assert "middleware=_default_middleware()" in inspect.getsource(ef.build_agent_executor)


def test_counting_failures_never_break_the_call(monkeypatch):
    """A trimmer that raises is worse than one that does nothing."""
    import langchain_core.messages.utils as u

    monkeypatch.setattr(ef, "DEFAULT_CONTEXT_BUDGET", 10)
    monkeypatch.delenv(ef.CONTEXT_BUDGET_ENV, raising=False)
    monkeypatch.setattr(u, "count_tokens_approximately",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    msgs = [SystemMessage("s"), HumanMessage("q")]
    mw = ef._make_context_budget_middleware()
    seen = {}
    mw.wrap_model_call(_Req(list(msgs)), lambda r: seen.setdefault("out", r.messages))
    assert seen["out"] == msgs
