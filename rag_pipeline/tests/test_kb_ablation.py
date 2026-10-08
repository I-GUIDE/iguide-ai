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
    # Prefix match: the code peer unions in prototype's web_search/web_fetch, so its call reads
    # `sorted(_peer_kb_tools() | {"web_search"})`. What matters is the function, not the constant.
    assert body.count("enabled_search_methods=sorted(_peer_kb_tools()") == 2
    assert "enabled_search_methods=sorted(_CODE_PEER_KB_TOOLS)" not in body


def test_the_deterministic_sweep_drops_both_of_its_kb_arms():
    """`_direct_search_sweep` unions the KB in without the model electing anything, so it is a
    second leak with no tool call to show for it. Two arms: extracted blocks, and the method
    library rendered as evidence."""
    import inspect

    sweep = inspect.getsource(graph._direct_search_sweep)
    assert 'permitted("agent_kb_search") and not kb_ablated()' in sweep
    assert "if not kb_ablated():\n                docs.extend(_method_units_as_documents" in sweep


# --------------------------------------------------- the harness must not change the treatment

def test_the_ab_harness_instrumentation_is_signature_transparent():
    """The counter that measures KB use must not alter the tools it counts.

    LangChain infers a tool's argument schema from the wrapped function's SIGNATURE. A wrapper
    that copied only `__name__` and `__doc__` left `(*a, **kw)`, so `agent_kb_search` was
    advertised to the model with NO parameters and the peer turn failed as soon as it called
    one. Because only the with-KB arm calls those tools, only that arm broke — producing
    no_kb 8/8 against with_kb 0/8, a perfectly clean and completely false result.
    """
    import inspect
    import sys

    sys.argv = ["ab_kb_problems"]
    from agent_runtime import langchain_granular_tools as tools_module
    from scripts.ab_kb_problems import _ToolCounter

    for attr in _ToolCounter.NAMES:
        if not hasattr(tools_module, attr):
            continue
        before = inspect.signature(getattr(tools_module, attr))
        with _ToolCounter():
            during = inspect.signature(getattr(tools_module, attr))
        after = inspect.signature(getattr(tools_module, attr))
        assert str(before) == str(during) == str(after), attr
        assert "*a" not in str(during), f"{attr} lost its parameters to the wrapper"


def test_the_counter_restores_the_originals():
    """A leaked wrapper would follow the process into every later run."""
    import sys

    sys.argv = ["ab_kb_problems"]
    from agent_runtime import langchain_granular_tools as tools_module
    from scripts.ab_kb_problems import _ToolCounter

    original = tools_module.agent_kb_search_tool
    with _ToolCounter():
        assert tools_module.agent_kb_search_tool is not original
    assert tools_module.agent_kb_search_tool is original


def test_importing_the_harness_does_not_reconfigure_the_process():
    """A module that mutates `os.environ` on import is a landmine for every test after it.

    This one loaded the platform `.env` at module scope, so importing it — which a test does,
    to check the instrumentation — supplied OPENSEARCH_NODE, NEO4J_* and the rest to the whole
    pytest process. Two unrelated tests asserting what happens when a lookup FAILS then passed
    alone and failed in a full run, with the cause three files away.
    """
    import inspect
    import sys

    sys.argv = ["ab_kb_problems"]
    from scripts import ab_kb_problems

    module_level = "".join(
        line for line in inspect.getsource(ab_kb_problems).splitlines(keepends=True)
        if line and not line[0].isspace())
    assert "load_dotenv(" not in module_level
    assert "os.environ[" not in module_level
    assert "os.environ.setdefault(" not in module_level
