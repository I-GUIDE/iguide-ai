"""The code peer must not be told to use a tool it does not have.

5f828f4 replaced an always-failing instruction (call a network loader inside a --network none
sandbox) with a route through `write_output_file` — a tool that is not bound on this peer — and
`web_fetch`, which returns a page's on-topic passages rather than the bytes of a dataset. Both
the prompt and the Chicago skill prescribed it. An impossible instruction had been swapped for
a differently impossible one, and nothing could catch that because no test compared what the
prompt NAMES against what the peer HAS.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO = Path(__file__).resolve().parents[2]


def _code_peer_tools(input_file_ids=None):
    """The tool names actually bound on the code peer, captured at the executor boundary."""
    import agent_runtime.executor_factory as ef
    from agent_runtime.supervisor import graph as g

    captured = {}
    original = ef.build_agent_executor

    def spy(*_a, **kw):
        captured["names"] = [getattr(t, "name", "") for t in (kw.get("preloaded_tools") or [])]
        raise RuntimeError("captured")

    ef.build_agent_executor = spy
    try:
        g.default_code_fn(llm=object(), code_exec=True,
                          input_file_ids=input_file_ids)("q", [], {"thread_id": "t"})
    except Exception:
        pass
    finally:
        ef.build_agent_executor = original
    return set(captured.get("names") or [])


def _tools_the_peer_can_have():
    """The union over both binding states.

    Six vector tools are gated on an attachment — 29 tools on a no-upload turn, 59 with one —
    so naming them is legitimate; the prompt is shared across both. The contract worth pinning
    is that it may not name a tool the peer can NEVER have.
    """
    return _code_peer_tools(None) | _code_peer_tools(["file_abc"])


# Names that look like tools but are not: python builtins, module names, file names.
_NOT_TOOLS = {"input_files", "element_id", "doc_id", "result_png", "bbox_inches",
              "network_none", "plt_show", "plt_savefig"}


def _tool_names_in(text: str) -> set:
    """Backticked snake_case identifiers that read as tool calls."""
    out = set()
    for name in re.findall(r"`([a-z][a-z0-9_]{3,})(?:\(|`)", text):
        if name in _NOT_TOOLS or "_" not in name:
            continue
        out.add(name)
    return out


def test_the_code_peer_prompt_names_no_tool_it_lacks():
    from agent_runtime.supervisor import prompts

    bound = _tools_the_peer_can_have()
    assert bound, "failed to capture the peer's tools"
    named = _tool_names_in(prompts.CODE_PEER_PROMPT)
    missing = sorted(n for n in named if n not in bound and n in _KNOWN_TOOLS)
    assert not missing, f"the prompt tells the peer to use unbound tools: {missing}"


def test_the_chicago_skill_names_no_tool_the_peer_lacks():
    bound = _tools_the_peer_can_have()
    skill = (REPO / "skills/chicago-crime-analysis/SKILL.md").read_text()
    missing = sorted(n for n in _tool_names_in(skill) if n not in bound and n in _KNOWN_TOOLS)
    assert not missing, f"the skill tells the peer to use unbound tools: {missing}"


def test_web_fetch_is_not_presented_as_a_way_to_download_data():
    """It returns on-topic passages. Prescribing it for a dataset produces a truncated file or
    none at all, and the peer cannot tell which."""
    from agent_runtime.supervisor import prompts

    skill = (REPO / "skills/chicago-crime-analysis/SKILL.md").read_text()
    for text in (prompts.CODE_PEER_PROMPT, skill):
        low = text.lower()
        if "web_fetch" not in low:
            continue
        assert "passages" in low or "not a dataset" in low, (
            "web_fetch is named without saying it cannot supply a dataset")


# Every tool name this repo actually defines, so a prose word that merely looks like a tool
# does not fail the test above.
def _known_tools():
    import importlib
    import pkgutil

    import agent_runtime
    names = set()
    for mod in pkgutil.iter_modules(agent_runtime.__path__):
        if mod.ispkg:
            continue
        try:
            m = importlib.import_module(f"agent_runtime.{mod.name}")
        except Exception:
            continue
        for attr, fn in vars(m).items():
            if not (attr.startswith("make_") and attr.endswith("tools") and callable(fn)):
                continue
            try:
                names.update(getattr(t, "name", "") for t in fn())
            except Exception:
                continue
    return {n for n in names if n}


_KNOWN_TOOLS = _known_tools()


def test_the_known_tool_inventory_is_not_empty():
    """Guards the two tests above: an empty inventory would make them vacuous."""
    assert len(_KNOWN_TOOLS) > 30, len(_KNOWN_TOOLS)
    assert "write_output_file" in _KNOWN_TOOLS
