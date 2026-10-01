"""Evidence describes only what its reader can act on (M8.64).

Every consumer used to get one rendering of an extracted method: "the import line works verbatim
inside execute_code (the library is mounted read-only)", and for a loader "FIRST call
stage_element". Only the LangChain code peer holds all of that. The analyse peer binds no staging
tools. The CLI code peers (claude_peer, opencode_peer) hold no execute_code, no staging tool and
no library mount, deliberately: their container keeps network access and the model credential, a
trust tier below the sandbox (extraction review, D1).

The rows live in capability_registry.EVIDENCE_CONSUMERS. These tests hold the renderer to the rows,
and each peer's row to the tools its builder actually binds. conftest turns the bundle ON, which is
the only state in which any of this differs: off, every consumer gets the reference view.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
IMPORT = "from iguide_methods.ke_b1fa548b.v_abc123 import load_crimes"
EXTRACTED = {"units": [{"symbol": "load_crimes", "signature": "def load_crimes(staged_path: str)",
                        "import_line": IMPORT, "requirements": ["pandas"]}],
             "unit_count": 5}
METHOD_DOC = {"doc_id": "method::load_crimes", "title": "load_crimes — callable method",
              "contents": ("def load_crimes(staged_path: str)\nRead the crime table.\n"
                           f"import: {IMPORT}\nrequires: pandas"),
              "source": "method_library", "resource_type": "MethodUnit", "import_line": IMPORT}
# Everything that tells a reader it can run, stage or search for a library method.
ACTIONABLE = ("iguide_methods", "import:", "RUNNABLE", "mounted", "stage_element",
              "kb_method_search")


def _render(consumer):
    from agent_runtime.supervisor.evidence_subgraph import _render_extracted

    return _render_extracted(EXTRACTED, "b1fa548b", consumer=consumer)


def test_the_code_peer_gets_everything_it_can_use():
    text = _render("code_peer")
    assert IMPORT in text and "stage_element" in text and "kb_method_search" in text


def test_the_analyze_peer_is_not_told_to_stage():
    text = _render("analyze_peer")
    assert IMPORT in text and "kb_method_search" in text
    assert "stage_element" not in text, "default_analyze_fn binds no staging tools"


def test_a_cli_peer_gets_the_reference_view_with_the_bundle_on():
    text = _render("cli_peer")
    assert "load_crimes" in text, "the method is still named, as a reference"
    assert not [t for t in ACTIONABLE if t in text], text


def test_an_unknown_consumer_is_told_nothing_it_might_not_have():
    assert not [t for t in ACTIONABLE if t in _render("a_peer_added_later")]


def test_a_method_document_loses_its_import_line_for_a_cli_peer():
    from agent_runtime.supervisor.evidence_subgraph import _format_documents

    cli = _format_documents([METHOD_DOC], consumer="cli_peer")
    assert "load_crimes" in cli and "requires: pandas" in cli
    assert IMPORT not in cli and "callable" not in cli, cli
    assert IMPORT in _format_documents([METHOD_DOC], consumer="code_peer")


def test_the_answer_view_is_unchanged_by_the_table():
    """Which cut the answerer reads is B3's to settle, not this change's."""
    from agent_runtime.supervisor.evidence_subgraph import _format_documents

    doc = {"doc_id": "b1fa548b", "title": "Chicago crime", "contents": "A notebook.",
           "extracted": EXTRACTED}
    assert (_format_documents([doc, METHOD_DOC])
            == _format_documents([doc, METHOD_DOC], consumer="code_peer"))


# ------------------------------------------------------------------ the CLI brief, end to end

def test_the_cli_brief_names_the_methods_and_offers_none_of_them():
    from agent_runtime.opencode_peer import _build_peer_prompt

    doc = {"doc_id": "b1fa548b", "title": "Chicago crime", "contents": "A notebook.",
           "extracted": EXTRACTED}
    prompt = _build_peer_prompt("map thefts by beat", [doc, METHOD_DOC], None)
    assert "load_crimes" in prompt
    assert not [t for t in (IMPORT, "RUNNABLE", "stage_element", "kb_method_search")
                if t in prompt], prompt


def test_library_code_from_the_analyze_peer_comes_with_a_warning():
    """The second route: analysis_results carries the analyse peer's own tool calls, which may
    import the library, into the CLI brief verbatim."""
    from agent_runtime.opencode_peer import _build_peer_prompt

    results = {"tool_calls": [{"name": "execute_code",
                               "args": {"code": f"{IMPORT}\nload_crimes(path)"}}]}
    assert "not available in this container" in _build_peer_prompt("map thefts", None, results)
    plain = {"tool_calls": [{"name": "execute_code", "args": {"code": "import pandas"}}]}
    assert "not available in this container" not in _build_peer_prompt("map", None, plain)


def test_a_failed_library_import_alone_also_brings_the_warning():
    """The case a user actually hits (the agent designer's check): the package name appears ONLY
    in a tool RESULT, the traceback of an import that failed, not in any code the brief shows."""
    from agent_runtime.opencode_peer import _build_peer_prompt

    results = {"tool_calls": [{"name": "execute_code", "args": {"code": "run()"}}],
               "tool_results": [{"name": "execute_code", "content":
                                 "ModuleNotFoundError: No module named 'iguide_methods'"}]}
    assert "not available in this container" in _build_peer_prompt("map thefts", None, results)


# ------------------------------------------------------------------ the rows match the builders

GRAPH = (REPO / "agent_runtime" / "supervisor" / "graph.py").read_text(encoding="utf-8")


def _builder(name: str) -> str:
    start = GRAPH.index(f"def {name}(")
    following = re.search(r"^def ", GRAPH[start + 1:], flags=re.M)
    return GRAPH[start:start + 1 + following.start()] if following else GRAPH[start:]


@pytest.mark.parametrize("consumer,builder", [("code_peer", "default_code_fn"),
                                              ("analyze_peer", "default_analyze_fn")])
def test_each_peer_row_matches_what_its_builder_binds(consumer, builder):
    """A declared row that drifts from the bindings is the original defect again, one level up."""
    from agent_runtime.capability_registry import (EVIDENCE_CONSUMERS, RUN_LIBRARY,
                                                   SEARCH_METHODS, STAGE_INPUTS)

    row, src = EVIDENCE_CONSUMERS[consumer], _builder(builder)
    assert (STAGE_INPUTS in row) == ("make_langchain_staging_tools(" in src), consumer
    assert (RUN_LIBRARY in row) == ("make_code_execution_tools(" in src), consumer
    assert (SEARCH_METHODS in row) == ("make_langchain_granular_tools(" in src), consumer


def test_each_renderer_call_names_its_consumer():
    """The analyse and code peers render with their own rows; the default is the answerer's."""
    assert "_format_documents(evidence, consumer='analyze_peer')" in _builder("default_analyze_fn")
    assert "_format_documents(evidence, consumer='code_peer')" in _builder("default_code_fn")
    peer = (REPO / "agent_runtime" / "opencode_peer.py").read_text(encoding="utf-8")
    assert '_format_documents(evidence, consumer="cli_peer")' in peer
