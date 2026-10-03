"""No peer is offered a tool that runs stored code inside the agent's own process.

``kb_run_geofunction`` loaded a knowledge-base block with ``get_kb_block``, cut one function out
of it with ``ast``, and ``exec()``'d it right here. Block source is notebook code that third
parties submit to the platform, and "here" is ``agent-api``, which runs as root with the host's
Docker socket mounted: whatever reached that ``exec`` could drive the host's Docker daemon. It was
bound to the analysis peer unconditionally, with no flag and no upload gate, and the only thing
keeping it unreachable was an empty knowledge base, which the extraction work is about to fill.

A name check alone would only stop this one function coming back, so there are two layers:

* **The regression.** No peer is offered the tool, the geo toolset keeps its three tools that do
  not run stored source, nothing in runtime code or prompts names it, and the function itself
  refuses to run unless the dev-only ``local`` backend is selected.
* **The class.** No tool any peer is offered can reach ``exec``/``eval`` (or a module/path
  runner) by name through this repo's code, followed transitively across module globals,
  function-local imports, closures, ``functools.wraps`` chains and nested functions. Running code
  in ANOTHER process is out of scope by design: that is how ``execute_code`` reaches its per-run
  sandbox container, the one sanctioned route for stored code.

The scan cannot see through an attribute call on an object it cannot type (``obj.run(src)``).
The positive controls at the bottom pin what it DOES see, so an interpreter upgrade that renamed
an opcode cannot quietly turn every assertion above them into a pass.
"""
from __future__ import annotations

import dis
import functools
import importlib
import importlib.util
import inspect
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# ---------------------------------------------------------------------------------------------
# The scan
# ---------------------------------------------------------------------------------------------

#: Builtins that run a string as code in the current process.
_SINK_NAMES = frozenset({"exec", "eval"})
#: Attribute calls that run a module or a file in the current process (importlib, runpy).
_SINK_ATTRS = frozenset({"exec_module", "run_path", "run_module"})
_CONST_LOADS = frozenset({"LOAD_CONST", "LOAD_SMALL_INT"})


def _is_repo_function(fn: types.FunctionType) -> bool:
    try:
        path = Path(fn.__code__.co_filename).resolve()
    except Exception:  # noqa: BLE001 - a code object with no usable filename is not ours
        return False
    return REPO in path.parents and "site-packages" not in path.parts


def _functions_in(value) -> list:
    """The repo-defined Python functions that a value reached by name stands for."""
    if isinstance(value, functools.partial):
        return _functions_in(value.func)
    if isinstance(value, (staticmethod, classmethod)):
        value = value.__func__
    if isinstance(value, types.MethodType):
        value = value.__func__
    if isinstance(value, types.FunctionType):
        return [value] if _is_repo_function(value) else []
    if isinstance(value, type):  # a class defined here: every function it defines
        out: list = []
        for member in vars(value).values():
            if isinstance(member, property):
                member = member.fget
            if member is not None and not isinstance(member, type):
                out += _functions_in(member)
        return out
    return []


def _imported_module(fn: types.FunctionType, ops: list, i: int):
    """The module that the ``IMPORT_NAME`` at ``ops[i]`` loads, relative imports resolved."""
    level = ops[i - 2].argval if i >= 2 and ops[i - 2].opname in _CONST_LOADS else 0
    name = ops[i].argval or ""
    try:
        if level:
            package = fn.__globals__.get("__package__") or fn.__module__.rpartition(".")[0]
            name = importlib.util.resolve_name("." * level + name, package)
        return importlib.import_module(name)
    except Exception:  # noqa: BLE001 - an optional dependency that is absent has no code to scan
        return None


def _in_process_sinks(fn) -> list:
    """Every reachable call that runs code in THIS process, as ``(sink, "file:line", via)``.

    A tool that only DEFINES an exec-ing function and registers it counts too: nested code is
    scanned as part of the function that defines it, because that registered function is the
    one the next tool build binds.
    """
    found: list = []
    seen: set = set()
    stack = [(f, ()) for f in _functions_in(fn)]
    while stack:
        f, via = stack.pop()
        if f.__code__ in seen:
            continue
        seen.add(f.__code__)
        here = via + (f"{f.__module__}.{f.__qualname__}",)
        stack += [(g, here) for g in _functions_in(getattr(f, "__wrapped__", None))]
        for cell in f.__closure__ or ():
            try:
                held = cell.cell_contents
            except ValueError:  # an empty cell
                continue
            stack += [(g, here) for g in _functions_in(held)]
        codes = [f.__code__]
        while codes:
            code = codes.pop()
            codes += [c for c in code.co_consts if isinstance(c, types.CodeType)]
            ops = list(dis.get_instructions(code))
            module = None  # what the last IMPORT_NAME loaded, for the IMPORT_FROMs after it
            for i, op in enumerate(ops):
                line = getattr(getattr(op, "positions", None), "lineno", None)
                where = f"{Path(code.co_filename).name}:{line}"
                if op.opname in ("LOAD_GLOBAL", "LOAD_NAME"):
                    if op.argval in _SINK_NAMES:
                        found.append((f"{op.argval}()", where, here))
                    else:
                        stack += [(g, here) for g in _functions_in(f.__globals__.get(op.argval))]
                elif op.opname in ("LOAD_ATTR", "LOAD_METHOD"):
                    prev = ops[i - 1] if i else None
                    via_builtins = (op.argval in _SINK_NAMES and prev is not None
                                    and prev.argval == "builtins")
                    if op.argval in _SINK_ATTRS or via_builtins:
                        found.append((f".{op.argval}()", where, here))
                elif op.opname == "IMPORT_NAME":
                    module = _imported_module(f, ops, i)
                elif op.opname == "IMPORT_FROM" and module is not None:
                    stack += [(g, here) for g in _functions_in(getattr(module, op.argval, None))]
    return found


def _tool_functions(tool) -> list:
    """What actually runs when a model calls ``tool``: its func/coroutine, or a subclass's methods."""
    out: list = []
    for attr in ("func", "coroutine"):
        out += _functions_in(getattr(tool, attr, None))
    return out or _functions_in(type(tool))


def _tool_sinks(tool) -> list:
    return [s for f in _tool_functions(tool) for s in _in_process_sinks(f)]


# ---------------------------------------------------------------------------------------------
# What each peer is actually offered
# ---------------------------------------------------------------------------------------------

_STATE = {"query": "q", "thread_id": "t-no-exec"}
# Pinned per configuration so a shell that exports the unified-peer default cannot relabel one.
_SPLIT = {**_STATE, "unified_peer": False}
_UNIFIED = {**_STATE, "unified_peer": True}


def _peer_builds() -> dict:
    from agent_runtime.supervisor import graph as g

    return {
        # build_supervisor_graph's own default for the search peer: no MCP tools
        "search": lambda: g.default_search_fn(llm=object())("q", dict(_STATE)),
        # what the API passes: include_mcp_tools defaults ON there, with no module list
        "search+mcp": lambda: g.default_search_fn(llm=object(), include_mcp_tools=True)(
            "q", dict(_STATE)),
        "analyze": lambda: g.default_analyze_fn(llm=object(), code_exec=True)(
            "q", [], dict(_SPLIT)),
        "analyze+upload": lambda: g.default_analyze_fn(
            llm=object(), code_exec=True, input_file_ids=["file_abc"])("q", [], dict(_SPLIT)),
        "analyze+unified": lambda: g.default_analyze_fn(llm=object(), code_exec=True)(
            "q", [], dict(_UNIFIED)),
        "code": lambda: g.default_code_fn(llm=object(), code_exec=True, code_peer="langchain")(
            "q", [], dict(_STATE)),
        "code+upload": lambda: g.default_code_fn(
            llm=object(), code_exec=True, code_peer="langchain", input_file_ids=["file_abc"])(
            "q", [], dict(_STATE)),
    }


PEERS = ("search", "search+mcp", "analyze", "analyze+upload", "analyze+unified",
         "code", "code+upload")


class _Built(Exception):
    """Raised in place of creating the agent: the final tool list has been captured."""


@pytest.fixture(scope="module")
def offered():
    """``{peer configuration: [tools]}``, captured where every peer hands its FINAL list over."""
    import langchain.agents as agents

    from agent_runtime import langchain_mcp_tools as mcp

    def _capture(**kw):
        raise _Built(list(kw.get("tools") or []))

    captured: dict = {}
    with pytest.MonkeyPatch.context() as mp:
        # The local import fallback, always. It is the configuration in which MCP tools run IN
        # this process, and a dev server answering on :8000 must not change what is checked.
        mp.setattr(mcp, "_make_remote_mcp_tools", lambda url: [])
        mp.setattr(agents, "create_agent", _capture)
        mcp.clear_mcp_cache()
        try:
            for label, build in _peer_builds().items():
                try:
                    build()
                except _Built as done:
                    captured[label] = done.args[0]
        finally:
            mcp.clear_mcp_cache()
    return captured


def _names(tools) -> set:
    return {str(getattr(t, "name", "")) for t in tools}


#: In-process paths that exist today and are NOT this change's to fix, with why. The scan skips
#: these names; ``test_the_known_gaps_are_still_real`` fails once one stops being real, so a
#: closed gap cannot linger here. ``kb_run_geofunction`` may never be added.
KNOWN_IN_PROCESS_PATHS = {
    "mcp_create_notebook_workflow_tool": (
        "Only under the MCP local-import fallback (the MCP server unreachable when tools are "
        "built) with MCP tools on and no module list, which is the API's default for the search "
        "peer. It registers a generated tool whose body exec()s notebook-derived source, and the "
        "next tool build binds that tool in this process. Remotely, the same code runs in the MCP "
        "server's container instead. Found by this guard; not fixed by the kb_run_geofunction "
        "change."),
}


def test_every_peer_configuration_was_captured(offered):
    """A configuration that never reached the agent constructor would make every check vacuous."""
    assert set(offered) == set(PEERS)
    assert all(offered[p] for p in PEERS)


@pytest.mark.parametrize("peer", PEERS)
def test_kb_run_geofunction_is_offered_to_no_peer(offered, peer):
    assert "kb_run_geofunction" not in _names(offered[peer]), (
        f"the {peer} peer is offered kb_run_geofunction, which exec()s third-party knowledge-base "
        "source inside agent-api (root, host Docker socket). KB code runs through get_kb_block + "
        "execute_code, in the sandbox.")


@pytest.mark.parametrize("peer", PEERS)
def test_the_scan_can_see_what_every_offered_tool_runs(offered, peer):
    """A tool the scan cannot resolve to code is a tool it silently passes."""
    blind = sorted(_names([t for t in offered[peer] if not _tool_functions(t)]))
    assert not blind, f"cannot resolve the code behind {blind}; teach _tool_functions about them"


@pytest.mark.parametrize("peer", PEERS)
def test_no_tool_a_peer_is_offered_runs_code_in_this_process(offered, peer):
    offenders = {}
    for tool in offered[peer]:
        name = str(getattr(tool, "name", ""))
        if name in KNOWN_IN_PROCESS_PATHS:
            continue
        for sink, where, via in _tool_sinks(tool):
            offenders.setdefault(name, f"{sink} at {where} via {' -> '.join(via)}")
    assert not offenders, (
        f"the {peer} peer is offered tools that run code inside agent-api, which is root with the "
        "host's Docker socket. Route them through the sandbox (agent_runtime/code_execution.py) "
        "or do not bind them:\n" + "\n".join(f"  {n}: {w}" for n, w in sorted(offenders.items())))


def test_the_known_gaps_are_still_real(offered):
    assert "kb_run_geofunction" not in KNOWN_IN_PROCESS_PATHS
    if not any(n.startswith("mcp_") for n in _names(offered["search+mcp"])):
        pytest.skip("no MCP tool module imported in this environment; nothing to compare against")
    still_real = {str(getattr(t, "name", "")) for tools in offered.values() for t in tools
                  if str(getattr(t, "name", "")) in KNOWN_IN_PROCESS_PATHS and _tool_sinks(t)}
    stale = sorted(set(KNOWN_IN_PROCESS_PATHS) - still_real)
    assert not stale, (
        f"{stale} no longer reach in-process execution from any peer. Delete them from "
        "KNOWN_IN_PROCESS_PATHS so the guard covers them again.")


# ---------------------------------------------------------------------------------------------
# The regression itself
# ---------------------------------------------------------------------------------------------

def test_the_geo_toolset_keeps_the_tools_that_do_not_run_stored_source():
    """The fix removes one tool, not the family: the filter and both renderers stay bound."""
    from extractors.geo_handles import make_geo_analysis_tools

    names = _names(make_geo_analysis_tools())
    assert "kb_run_geofunction" not in names
    assert {"kb_select_rows", "heatmap_image", "choropleth_image"} <= names


def test_what_can_you_do_does_not_advertise_it():
    """The capability inventory is read from the same factories, so it must agree."""
    from agent_runtime.capabilities import collect_capability_inventory

    names = {t["name"] for t in collect_capability_inventory(include_mcp_tools=False)["tools"]}
    assert "kb_run_geofunction" not in names
    assert {"kb_select_rows", "heatmap_image", "choropleth_image"} <= names


#: Where the deployment's own code and prompts live. The function is defined in
#: extractors/geo_handles.py and may be called by hand from extractors/examples/; nothing else
#: here may name it, whether as a direct call a tool scan cannot see or a prompt that would send
#: the model to a tool it does not have.
_RUNTIME_DIRS = ("agent_runtime", "api", "rag_pipeline", "extractors", "MCP_server", "skills")
_MAY_NAME_IT = (Path("extractors/geo_handles.py"), Path("extractors/examples"),
                Path("rag_pipeline/tests"))


def test_no_runtime_code_or_prompt_names_it():
    hits = []
    for top in _RUNTIME_DIRS:
        for path in sorted((REPO / top).rglob("*")):
            if path.suffix not in {".py", ".md"} or not path.is_file():
                continue
            rel = path.relative_to(REPO)
            if any(rel == ok or ok in rel.parents for ok in _MAY_NAME_IT):
                continue
            if "kb_run_geofunction" in path.read_text(encoding="utf-8", errors="ignore"):
                hits.append(str(rel))
    assert not hits, f"runtime code or prompts name kb_run_geofunction: {hits}"


_PLANTED = '''
def planted(x: int) -> int:
    return x + 1
'''


def _plant_block(monkeypatch, code: str) -> list:
    """Make every block id resolve to ``code``; return the ids that were actually read."""
    import rag_pipeline.search.agent_kb as kb

    reads: list = []

    def _get_kb_block(doc_id, **_kw):
        reads.append(doc_id)
        return {"found": True, "source": {"extracted": {"block": {"code": code}}}}

    monkeypatch.setattr(kb, "get_kb_block", _get_kb_block)
    return reads


@pytest.mark.parametrize("backend", [None, "docker", " DOCKER ", "disabled", ""])
def test_the_function_refuses_outside_the_local_backend(monkeypatch, backend):
    """Fail closed even for a direct call. The deployment pins the docker backend."""
    from extractors.geo_handles import kb_run_geofunction

    reads = _plant_block(monkeypatch, _PLANTED)
    if backend is None:
        monkeypatch.delenv("AGENT_CODE_EXEC_BACKEND", raising=False)
    else:
        monkeypatch.setenv("AGENT_CODE_EXEC_BACKEND", backend)
    out = json.loads(kb_run_geofunction("blk_planted", "planted", '{"x": 41}'))
    assert "disabled" in out.get("error", ""), out
    assert "result" not in out
    assert reads == [], "the block was read: the refusal must come before anything stored loads"


def test_the_dev_backend_still_runs_it_for_the_demo(monkeypatch):
    """The gate is the only thing that changed; the hand-run demo keeps working."""
    from extractors.geo_handles import kb_run_geofunction

    reads = _plant_block(monkeypatch, _PLANTED)
    monkeypatch.setenv("AGENT_CODE_EXEC_BACKEND", "local")
    out = json.loads(kb_run_geofunction("blk_planted", "planted", '{"x": 41}'))
    assert out.get("result") == 42, out
    assert reads == ["blk_planted"]


# ---------------------------------------------------------------------------------------------
# Positive controls: what the scan must see, so the guards above cannot pass vacuously
# ---------------------------------------------------------------------------------------------

def _runs_source(src: str) -> None:
    exec(src, {})  # noqa: S102 - the sink, one call away from the control below


def _calls_a_helper(src: str) -> None:
    _runs_source(src)


def _runs_a_file(path: str) -> None:
    spec = importlib.util.spec_from_file_location("planted", path)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))


def _through_a_local_import(doc_id: str) -> str:
    from extractors.geo_handles import kb_run_geofunction as run

    return run(doc_id, "f")


def _defines_an_exec_ing_function() -> object:
    def registered(src: str) -> None:
        eval(src)  # noqa: S307 - the shape create_notebook_workflow_tool has

    return registered


def _controls() -> dict:
    from langchain_core.tools import StructuredTool

    from agent_runtime.tool_args import accept_null_defaults
    from extractors import geo_handles as gh

    return {
        "the function itself": gh.kb_run_geofunction,
        "a functools.wraps wrapper": accept_null_defaults(gh.kb_run_geofunction),
        "a closure over it": gh.make_file_handle_tool(gh.kb_run_geofunction),
        "a StructuredTool": StructuredTool.from_function(
            func=gh.kb_run_geofunction, name="kb_run_geofunction", description="control"),
        "a module-level helper": _calls_a_helper,
        "importlib exec_module": _runs_a_file,
        "a function-local import": _through_a_local_import,
        "a nested definition": _defines_an_exec_ing_function,
        "a partial": functools.partial(_calls_a_helper, "1"),
    }


_CONTROL_LABELS = ("the function itself", "a functools.wraps wrapper", "a closure over it",
                   "a StructuredTool", "a module-level helper", "importlib exec_module",
                   "a function-local import", "a nested definition", "a partial")


@pytest.mark.parametrize("label", _CONTROL_LABELS)
def test_the_scan_sees_in_process_execution(label):
    subject = _controls()[label]
    sinks = _tool_sinks(subject) if hasattr(subject, "func") and hasattr(subject, "name") \
        else [s for f in _functions_in(subject) for s in _in_process_sinks(f)]
    assert sinks, f"the scan missed in-process execution through {label}"


def test_the_sandbox_route_is_not_mistaken_for_it():
    """execute_code reaches a per-run container; flagging it would make the guard unusable."""
    from agent_runtime.langchain_exec_tools import make_code_execution_tools

    tool = next(t for t in make_code_execution_tools() if t.name == "execute_code")
    assert _tool_functions(tool), "execute_code resolved to no code; the negative control is void"
    assert not _tool_sinks(tool)
