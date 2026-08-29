"""The ablation switch has to remove the knowledge base from EVERY path it reaches the agent by.

This file exists because the first ablation did not. It was built on `enabled_search_methods`,
which filters the search peer only — the code and analyze peers hold `_CODE_PEER_KB_TOOLS`
"deliberately independent of the request's enabled_search_methods", and `_direct_search_sweep`
unions the KB in deterministically so it does not depend on the model electing a tool. All three
are correct for serving. Together they made a no-KB arm a second with-KB arm, and it reported
its answers as CORRECT — a measurement that looks like a result.

So each test below pins one path that leaked.
"""

from __future__ import annotations

import pytest

from agent_runtime.supervisor import graph


def test_the_switch_is_off_by_default(monkeypatch):
    """An experiment control, not a feature. The deployed answer is always no."""
    monkeypatch.delenv("AGENT_ABLATE_KB", raising=False)
    assert graph.kb_ablated() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_the_switch_reads_the_usual_truthy_values(monkeypatch, value):
    monkeypatch.setenv("AGENT_ABLATE_KB", value)
    assert graph.kb_ablated() is True


def test_the_switch_is_read_per_call_not_captured_at_import(monkeypatch):
    """A harness sets it between runs in one process; a value captured at import would apply
    the first arm's setting to every arm after it."""
    monkeypatch.setenv("AGENT_ABLATE_KB", "1")
    assert graph.kb_ablated() is True
    monkeypatch.setenv("AGENT_ABLATE_KB", "0")
    assert graph.kb_ablated() is False


def test_the_code_and_analyze_peers_lose_their_kb_grant(monkeypatch):
    """The path that broke the first sweep. These peers are granted the KB tools regardless of
    the request's `enabled_search_methods`, so filtering that alone leaves them intact."""
    monkeypatch.delenv("AGENT_ABLATE_KB", raising=False)
    assert graph._peer_kb_tools() == graph._CODE_PEER_KB_TOOLS
    monkeypatch.setenv("AGENT_ABLATE_KB", "1")
    assert graph._peer_kb_tools() == set()


def test_the_grant_is_taken_from_the_function_not_the_constant():
    """Both peer sites must call `_peer_kb_tools()`. Reading the constant directly at either one
    would leave that peer holding the KB through an ablation — which is exactly what happened,
    silently, in both places at once."""
    import inspect

    source = inspect.getsource(graph)
    body = source.split("def _peer_kb_tools", 1)[1]
    assert body.count("enabled_search_methods=sorted(_peer_kb_tools())") == 2
    assert "enabled_search_methods=sorted(_CODE_PEER_KB_TOOLS)" not in body


def test_the_deterministic_sweep_drops_both_of_its_kb_arms():
    """`_direct_search_sweep` unions the KB in without the model electing anything, so it is a
    second leak with no tool call to show for it. Two arms: extracted blocks, and the method
    library rendered as evidence."""
    import inspect

    sweep = inspect.getsource(graph._direct_search_sweep)
    assert 'permitted("agent_kb_search") and not kb_ablated()' in sweep
    assert "if not kb_ablated():\n                docs.extend(_method_units_as_documents" in sweep
