"""The retrieval window is one knob, resolved at call time.

Two failure modes this pins, both of which actually happened:

1. **The window was sixteen literals.** `8` appeared independently in keyword.py,
   semantic.py, core.py, neo4j.py, spatial.py (x2), opengeodata.py (x3), five tool
   signatures, and `_direct_search_sweep` — plus a `12` in two underlying functions. The
   agent never overrode any of them, so recall was pinned at 22/37 while 29/37 was
   available at k=20.

2. **A function default freezes at import.** `limit: int = default_top_k()` would bind once
   and ignore the environment forever. Every site must therefore take `None` and resolve
   inside. This is not hypothetical: the first draft of the window change did exactly that
   and measured "no improvement" because the value never moved.

A third, subtler one: `opengeodata._payload_from_context` does `int(limit or 1)`, so a None
arriving there becomes a single result rather than the window. Pinned below.
"""

from __future__ import annotations

import pytest

from rag_pipeline.search.utils import default_top_k


def test_default_window_is_20():
    """20 recovers 7 of the 15 expected elements that k=8 dropped (22/37 -> 29/37)."""
    assert default_top_k() == 20


def test_env_override(monkeypatch):
    monkeypatch.setenv("AGENT_SEARCH_TOP_K", "50")
    assert default_top_k() == 50


def test_bad_value_falls_back(monkeypatch):
    monkeypatch.setenv("AGENT_SEARCH_TOP_K", "not-a-number")
    assert default_top_k() == 20


def test_window_is_clamped(monkeypatch):
    monkeypatch.setenv("AGENT_SEARCH_TOP_K", "9999")
    assert default_top_k() == 100
    monkeypatch.setenv("AGENT_SEARCH_TOP_K", "0")
    assert default_top_k() == 1


def test_resolution_happens_at_call_time(monkeypatch):
    """The regression that made the first attempt measure nothing."""
    monkeypatch.setenv("AGENT_SEARCH_TOP_K", "11")
    assert default_top_k() == 11
    monkeypatch.setenv("AGENT_SEARCH_TOP_K", "33")
    assert default_top_k() == 33, "value was frozen — a default arg bound it at import"


def test_safe_int_follows_the_window(monkeypatch):
    from agent_runtime.langchain_granular_tools import _safe_int
    monkeypatch.setenv("AGENT_SEARCH_TOP_K", "25")
    assert _safe_int(None) == 25          # tool called without an explicit limit
    assert _safe_int(5) == 5              # model asked for fewer: honoured
    assert _safe_int(9999) == 100         # and still clamped


@pytest.mark.parametrize("tool_name", [
    "keyword_search_tool", "semantic_search_tool", "neo4j_search_tool",
    "spatial_search_tool", "agent_kb_search_tool", "opengeodata_search_tool",
])
def test_no_tool_signature_freezes_the_window(tool_name):
    """Every retrieval tool must default `limit` to None, not to an integer."""
    import inspect
    from agent_runtime import langchain_granular_tools as g
    fn = getattr(g, tool_name)
    sig = inspect.signature(getattr(fn, "__wrapped__", fn))
    assert "limit" in sig.parameters, f"{tool_name} has no limit parameter"
    default = sig.parameters["limit"].default
    assert default is None, (
        f"{tool_name} freezes limit at {default!r}; it must be None so the shared window "
        "resolves per call"
    )


def test_underlying_search_functions_take_none():
    """keyword/semantic had their OWN signature defaults of 12 — a third window."""
    import inspect
    from rag_pipeline.search.keyword import get_keyword_search_results
    from rag_pipeline.search.semantic import semantic_search
    assert inspect.signature(get_keyword_search_results).parameters["size"].default is None
    assert inspect.signature(semantic_search).parameters["size"].default is None


def test_opengeodata_none_becomes_the_window_not_one(monkeypatch):
    """_payload_from_context does `int(limit or 1)`: a None reaching it means ONE result."""
    import rag_pipeline.search.opengeodata as og
    monkeypatch.setenv("AGENT_SEARCH_TOP_K", "17")
    seen = {}
    original = og._payload_from_context

    def spy(query, *, limit, session_ctx=None, **kw):
        seen["limit"] = limit
        return original(query, limit=limit, session_ctx=session_ctx, **kw)

    monkeypatch.setattr(og, "_payload_from_context", spy)
    og.get_opengeodata_results("satellite imagery of wildfires in California")
    assert seen.get("limit") == 17, f"limit collapsed to {seen.get('limit')!r}"


def test_supervisor_top_k_is_a_separate_knob(monkeypatch):
    """Retrieve wide, rerank, show few. Conflating these scales prompt cost with k."""
    from agent_runtime.supervisor.graph import _default_top_k
    monkeypatch.setenv("AGENT_SEARCH_TOP_K", "50")
    monkeypatch.delenv("AGENT_SUPERVISOR_TOP_K", raising=False)
    assert default_top_k() == 50
    assert _default_top_k() == 8, "the answer-prompt cap must not follow the retrieval window"


def test_direct_search_sweep_does_not_freeze_k():
    import inspect
    from agent_runtime.supervisor.graph import _direct_search_sweep
    assert inspect.signature(_direct_search_sweep).parameters["k"].default is None
