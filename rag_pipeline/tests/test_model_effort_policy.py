"""Which reasoning_effort values may be sent, per model, WITH function tools attached.

The table under test was probed against the live API (tools bound, three efforts per model,
repeated). These tests pin the shape of the policy, not the API: if the API changes, the
table changes and these move with it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime.executor_factory import (  # noqa: E402
    REASONING_EFFORTS,
    effort_options,
    list_available_models,
    required_effort,
    resolve_effort,
    supports_reasoning_effort,
)


def test_gpt4o_never_receives_the_argument():
    """gpt-4o answers 'Unrecognized request argument supplied: reasoning_effort'."""
    for m in ("gpt-4o-2024-11-20", "gpt-4o-mini", "gpt-4.1-2025-04-14"):
        assert effort_options(m) == []
        assert supports_reasoning_effort(m) is False
        assert resolve_effort(m, "high") is None
        assert resolve_effort(m, "none") is None


def test_gpt56_requires_none_even_when_nothing_was_asked_for():
    """The observed failure: with tools and no effort, gpt-5.6-* refuses outright —
    'Function tools with reasoning_effort are not supported ... or set it to none'."""
    for m in ("gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"):
        assert required_effort(m) == "none"
        assert resolve_effort(m, None) == "none", "must be supplied unasked"
        assert resolve_effort(m, "high") == "none", "a real level is refused; coerce, not 400"


def test_models_that_accept_only_none_drop_a_real_level():
    for m in ("gpt-5.5", "gpt-5.4", "gpt-5.4-mini"):
        assert effort_options(m) == ["none"]
        assert resolve_effort(m, "medium") is None
        assert resolve_effort(m, "none") == "none"


def test_the_two_models_that_can_actually_think_harder_with_tools():
    assert resolve_effort("gpt-5.2", "high") == "high"
    assert resolve_effort("gpt-5.2", "none") == "none"
    # o4-mini takes any level but rejects 'none' ("Supported values are: 'low', ... 'xhigh'").
    assert resolve_effort("o4-mini-2025-04-16", "high") == "high"
    assert "none" not in effort_options("o4-mini-2025-04-16")
    assert resolve_effort("o4-mini-2025-04-16", "none") is None


def test_an_unknown_model_degrades_to_no_effort_rather_than_a_400():
    assert effort_options("gpt-9-future") == []
    assert resolve_effort("gpt-9-future", "high") is None


def test_ids_that_cannot_serve_a_tool_using_agent_are_not_offered():
    """gpt-5.5-pro is 'not a chat model'; gpt-5.3-chat-latest 'has been deprecated'.
    Offering them put a guaranteed failure in the picker."""
    openai = [p for p in list_available_models(timeout=1)["providers"]
              if p["provider"] == "openai"][0]
    assert "gpt-5.5-pro" not in openai["models"]
    assert "gpt-5.3-chat-latest" not in openai["models"]


def test_catalogue_gives_the_picker_per_model_values():
    openai = [p for p in list_available_models(timeout=1)["providers"]
              if p["provider"] == "openai"][0]
    opts = openai["effort_options"]
    assert opts["gpt-5.6-luna"] == ["none"]
    assert opts["gpt-5.2"] == ["none", "low", "medium", "high", "xhigh"]
    assert "gpt-4o-2024-11-20" not in opts, "no options means no control at all"
    assert openai["effort_required"]["gpt-5.6-sol"] == "none"
    # every advertised value is one the request validator accepts
    for values in opts.values():
        assert set(values) <= set(REASONING_EFFORTS)
