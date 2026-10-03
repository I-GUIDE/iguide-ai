"""What the search peer binds from the MCP server, read off its real tool assembly.

The search peer passed no MCP module list, and ``make_langchain_mcp_tools`` reads that as every
tool the server registers. MCP is on for an API request by default, so the deployed search peer
bound 39 tools and 14 of them were MCP tools: 2,248 of its 7,183 schema tokens (o200k), and about
1,584 real input tokens on every search model call. The deployed journal holds 166 search calls,
and none called an MCP tool. ``SEARCH_AGENT_PROMPT`` names one MCP tool, rule 8's
``mcp_fetch_element_source``, so search now binds ``element_tools`` by default, the way analyze
binds ``spatial_analysis_tools``.

The server's listing is stubbed with what a server built from this checkout registers, which is
every ``@mcp_tool`` function under ``MCP_server/tools``. On 2026-10-03 the deployed server listed
the same 16 names. Nothing here reaches the network.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[2] / "MCP_server" / "tools"


def _is_mcp_tool(decorator) -> bool:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    return "mcp_tool" in (getattr(target, "id", None), getattr(target, "attr", None))


def _server_tools() -> dict:
    """``{module: [tool names]}``, as ``MCP_server/server.py`` registers them at startup."""
    out = {}
    for path in sorted(TOOLS_DIR.glob("*.py")):
        if path.name.startswith("__"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        out[path.stem] = [node.name for node in tree.body
                          if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                          and any(_is_mcp_tool(d) for d in node.decorator_list)]
    return out


def _bound_from(module: str) -> set:
    """The agent-side names of *module*'s tools, after the agent's own unbind list."""
    from agent_runtime.langchain_mcp_tools import _is_unbound_mcp_tool

    return {f"mcp_{name}" for name in _server_tools()[module] if not _is_unbound_mcp_tool(name)}


@pytest.fixture(autouse=True)
def _fresh_mcp(monkeypatch):
    from agent_runtime import langchain_mcp_tools as mcp

    monkeypatch.delenv("AGENT_MCP_UNBIND", raising=False)
    monkeypatch.setenv("MCP_CACHE_TTL_SECONDS", "0")
    mcp.clear_mcp_cache()
    yield
    mcp.clear_mcp_cache()


def _serve(monkeypatch, names) -> None:
    """Answer the MCP listing with *names*, as the server's ``list_tools`` would."""
    from agent_runtime import langchain_mcp_tools as mcp

    listing = [SimpleNamespace(name=n, description=f"{n} (stub)", inputSchema={}) for n in names]

    async def list_tools(url):
        return listing

    monkeypatch.setattr(mcp, "_remote_mcp_list_tools_async", list_tools)


@pytest.fixture
def server(monkeypatch):
    tools = _server_tools()
    names = [n for module_tools in tools.values() for n in module_tools]
    # Not vacuous: the server lists element_tools' tool AND tools a scope has to leave out.
    assert "fetch_element_source" in names and len(names) > len(tools["element_tools"]), names
    _serve(monkeypatch, names)


@pytest.fixture
def unreachable(monkeypatch):
    from agent_runtime import langchain_mcp_tools as mcp

    async def refuse(url):
        raise ConnectionError("no MCP server in tests")

    monkeypatch.setattr(mcp, "_remote_mcp_list_tools_async", refuse)


class _Built(Exception):
    """Raised in place of create_agent: the search peer's final tool list is in hand."""


def _search_binds(monkeypatch, **flags) -> set:
    """The tool names the supervisor's search peer hands to ``create_agent``."""
    import langchain.agents

    from agent_runtime.supervisor import graph

    def stop(**kwargs):
        raise _Built([str(getattr(t, "name", "")) for t in kwargs.get("tools") or []])

    monkeypatch.setattr(langchain.agents, "create_agent", stop)
    with pytest.raises(_Built) as built:
        graph.default_search_fn(llm=object(), **flags)("probe", {"thread_id": "probe"})
    return set(built.value.args[0])


def _mcp(names) -> set:
    return {n for n in names if n.startswith("mcp_")}


def test_search_binds_only_element_tools_by_default(monkeypatch, server):
    """An API request's default: MCP on and no module list."""
    bound = _mcp(_search_binds(monkeypatch, include_mcp_tools=True))
    assert bound == {"mcp_fetch_element_source"}, (
        f"the search peer binds {sorted(bound)} from the MCP server. Before element_tools was its "
        "default it bound all 14 remote tools and called none of them in 166 deployed calls.")


def test_the_local_fallback_binds_the_same_scope(monkeypatch, unreachable):
    bound = _mcp(_search_binds(monkeypatch, include_mcp_tools=True))
    assert bound == {"mcp_fetch_element_source"}


def test_mcp_off_binds_no_mcp_tool(monkeypatch, server):
    assert not _mcp(_search_binds(monkeypatch, include_mcp_tools=False))


def test_a_request_naming_modules_still_gets_them(monkeypatch, server):
    """``mcpModules`` replaces the default, for search as for analyze. The reference client
    sends one, so this is what it binds."""
    bound = _mcp(_search_binds(monkeypatch, include_mcp_tools=True, mcp_modules=["data_tools"]))
    assert bound == _bound_from("data_tools") != set()


def test_a_scope_the_server_does_not_serve_binds_nothing(monkeypatch):
    """When no remote tool matched, scoping used to keep the WHOLE list. A server without
    element_tools would then have handed search the other 13 tools again."""
    names = [n for tools in _server_tools().values() for n in tools if n != "fetch_element_source"]
    _serve(monkeypatch, names)
    assert not _mcp(_search_binds(monkeypatch, include_mcp_tools=True))


def test_search_tools_alone_binds_nothing(monkeypatch, server):
    """The scope that looks like search's: its one tool, search_external_resources, is unbound
    by default, so nothing matches. It bound all 14 remote tools, measured in agent-api against
    the deployed server on 2026-10-03."""
    pytest.importorskip("ddgs")  # the module imports it; unimportable, it would not be resolved
    from agent_runtime.langchain_mcp_tools import make_langchain_mcp_tools

    assert make_langchain_mcp_tools(include_modules=["search_tools"]) == []
    assert not _mcp(_search_binds(monkeypatch, include_mcp_tools=True,
                                  mcp_modules=["search_tools"]))


def test_analyze_keeps_its_scope(server):
    from agent_runtime.langchain_mcp_tools import make_langchain_mcp_tools

    bound = {t.name for t in make_langchain_mcp_tools(include_modules=["spatial_analysis_tools"])}
    assert bound == _bound_from("spatial_analysis_tools") != set()


def test_search_binds_the_mcp_tools_its_prompt_names_and_no_others(monkeypatch, server):
    """Both directions. A tool the prompt names and the peer lacks is a call that cannot work,
    which rule 8 was until the name gained its prefix. A tool the peer binds and nothing asks for
    is the 14-tool surface this replaced. Widening the scope means naming the tools in the prompt
    and deciding whether the decider's search line should change too."""
    from agent_runtime.prompts import SEARCH_AGENT_PROMPT

    named = set(re.findall(r"`(mcp_[a-z0-9_]+)`", SEARCH_AGENT_PROMPT))
    assert named, "SEARCH_AGENT_PROMPT no longer names an MCP tool; drop the scope with it"
    assert _mcp(_search_binds(monkeypatch, include_mcp_tools=True)) == named
