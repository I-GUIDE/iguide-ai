"""What AGENT_EXTRACTION=off means, pinned surface by surface.

The extraction bundle is OFF by default (agent_runtime/extraction_flag.py), so integrating it into
prototype changes nothing in production until someone turns it on. "Nothing" has two halves, and
both are tested here: the tools, mount, gate and artifacts are absent, AND no prompt, tool
description or capability clause mentions them. The second half matters as much as the first: a
peer told to call `kb_method_search` without having it guessed the package name and wrote
`from method_library import ...`, which fails.

conftest turns the flag ON for the rest of the suite; every test here sets it explicitly.
"""
from __future__ import annotations

import pytest

EXTRACTION_TOOLS = {"kb_method_search", "get_method_contract",
                    "stage_element", "stage_url", "list_staged_inputs"}


@pytest.fixture()
def off(monkeypatch):
    monkeypatch.delenv("AGENT_EXTRACTION", raising=False)
    monkeypatch.delenv("AGENT_INVARIANT_GATE", raising=False)
    monkeypatch.delenv("AGENT_ARTIFACT_EMIT", raising=False)
    return monkeypatch


@pytest.fixture()
def library(tmp_path, monkeypatch):
    """A built library on disk, so 'not mounted' cannot pass merely because none exists."""
    (tmp_path / "iguide_methods").mkdir()
    monkeypatch.setenv("AGENT_METHOD_LIBRARY_DIR", str(tmp_path))
    return tmp_path


def _tool_names():
    from agent_runtime.langchain_granular_tools import (make_langchain_granular_tools,
                                                         make_langchain_staging_tools)

    names = {t.name for t in make_langchain_granular_tools()}
    names |= {t.name for t in make_langchain_staging_tools(session_id="flag-test")}
    return names


def test_the_flag_defaults_off(off):
    from agent_runtime.extraction_flag import extraction_enabled

    assert extraction_enabled() is False


def test_off_offers_none_of_the_extraction_tools(off):
    assert not (_tool_names() & EXTRACTION_TOOLS)


def test_on_offers_all_of_them(monkeypatch):
    monkeypatch.setenv("AGENT_EXTRACTION", "1")
    assert EXTRACTION_TOOLS <= _tool_names()


def test_off_mounts_no_library_even_when_one_is_built(off, library, tmp_path):
    from agent_runtime.code_execution import METHOD_LIBRARY_MOUNT, DockerCodeExecutor

    argv = " ".join(DockerCodeExecutor().build_argv(tmp_path, "x"))
    assert METHOD_LIBRARY_MOUNT not in argv

    off.setenv("AGENT_EXTRACTION", "1")
    argv = " ".join(DockerCodeExecutor().build_argv(tmp_path, "x"))
    assert f"{library}:{METHOD_LIBRARY_MOUNT}:ro" in argv, "the ON half of this test is not real"


def test_off_runs_no_gate_and_emits_no_artifacts_by_default(off):
    from agent_runtime.artifacts import artifacts_enabled
    from agent_runtime.code_execution import invariant_gate_enabled

    assert invariant_gate_enabled() is False
    assert artifacts_enabled() is False


def test_their_own_switches_still_override_the_bundle(off):
    from agent_runtime.artifacts import artifacts_enabled
    from agent_runtime.code_execution import invariant_gate_enabled

    off.setenv("AGENT_INVARIANT_GATE", "1")
    off.setenv("AGENT_ARTIFACT_EMIT", "1")
    assert invariant_gate_enabled() is True and artifacts_enabled() is True


def test_off_prompts_never_mention_the_tools(off):
    from agent_runtime import prompts

    for text in (prompts.search_agent_prompt(), prompts.code_agent_prompt()):
        assert "kb_method_search" not in text and "get_method_contract" not in text
    assert prompts.code_agent_prompt() == prompts.CODE_AGENT_PROMPT


def test_off_the_supervisor_is_not_told_about_them(off):
    from agent_runtime.capability_registry import describe

    for cap in ("analyze", "code"):
        text = describe(cap)
        assert "method library" not in text and "staging" not in text, text


def test_off_execute_code_does_not_describe_the_gate_or_the_library(off):
    from agent_runtime.langchain_exec_tools import make_code_execution_tools

    desc = make_code_execution_tools()[0].description
    assert "iguide_methods" not in desc and "IGUIDE_OUTPUTS" not in desc and "kb_method_search" not in desc


def test_off_the_sweep_never_reads_the_library(off, monkeypatch):
    import agent_runtime.method_library as ml
    from agent_runtime.supervisor import graph as g

    def boom(*a, **k):
        raise AssertionError("the sweep searched the method library with extraction off")

    monkeypatch.setattr(ml, "search_methods", boom)
    docs = g._direct_search_sweep("buffer a road network", ["kb_method_search"])
    assert not [d for d in docs if d.get("source") == "method_library"]


# ------------------------------------------------------------------ what the evidence says
#
# Three surfaces that came with the merge described the bundle with it off: every agent_kb_search
# hit with a unit carried its import line, the sweep's per-turn KB join attached units to
# documents, and the evidence view said the library was mounted and told the model to
# `stage_element` and to call `kb_method_search`. Off, a unit is still NAMED (it is true of the
# element in any deployment); nothing says how to import it.

UNIT = {
    "library_symbol": "load_crimes",
    "library_module": "iguide_methods.ke_b1fa548b.v_abc123",
    "slice_sha": "abc123",
    "signature": "def load_crimes(staged_path: str)",
    "doc_summary": "Read the crime table.",
    "requirements": {"pip": ["pandas"]},
    "callability": {"verdict": "callable", "reason": ""},
    "import_line": "from iguide_methods.ke_b1fa548b.v_abc123 import load_crimes",
}
# Every phrase that tells a model some part of the bundle is there.
BUNDLE_TERMS = ("iguide_methods", "import:", "kb_method_search", "get_method_contract",
                "stage_element", "mounted", "RUNNABLE")


def _hit():
    return {"_index": "iguide_agent_method_units", "_id": "b1fa548b::load_crimes",
            "_source": {"doc_id": "b1fa548b::load_crimes", "title": "load_crimes",
                        "extracted": {"parent_doc_id": "b1fa548b", "unit": dict(UNIT)}}}


def test_off_a_kb_hit_names_the_unit_but_not_how_to_import_it(off):
    from rag_pipeline.search.agent_kb import normalize_hit

    method = normalize_hit(_hit(), "keyword")["method"]
    assert method["signature"] == UNIT["signature"], "the reference half is true in any deployment"
    assert not {"import_line", "slice_sha", "callable"} & set(method), method

    off.setenv("AGENT_EXTRACTION", "1")
    assert normalize_hit(_hit(), "keyword")["method"]["import_line"] == UNIT["import_line"]


def _extracted():
    return {"units": [{"symbol": "load_crimes", "signature": UNIT["signature"],
                       "import_line": UNIT["import_line"], "requirements": ["pandas"]}],
            "unit_count": 5}


def test_off_the_evidence_view_never_points_at_the_bundle(off):
    from agent_runtime.supervisor.evidence_subgraph import _render_extracted

    text = _render_extracted(_extracted(), "b1fa548b")
    assert "load_crimes" in text, "the method is still named, as a reference"
    assert not [t for t in BUNDLE_TERMS if t in text], text

    off.setenv("AGENT_EXTRACTION", "1")
    on = _render_extracted(_extracted(), "b1fa548b")
    assert UNIT["import_line"] in on and "stage_element" in on and "kb_method_search" in on


def test_off_a_fetched_kb_doc_carries_no_import_line(off, monkeypatch):
    """get_kb_block returns the RAW stored doc, so a unit doc carried its import line and library
    module whatever the flag said. Found while driving the map UI with the bundle off (M8.67)."""
    from rag_pipeline.search import agent_kb

    class Client:
        def get(self, index, id):
            return {"found": True, "_source": _hit()["_source"]}

    monkeypatch.setattr(agent_kb, "_os_client", lambda: Client())
    monkeypatch.setenv("AGENT_KB_BACKEND", "opensearch")
    unit = agent_kb.get_kb_block("b1fa548b::load_crimes")["source"]["extracted"]["unit"]
    assert unit["signature"] == UNIT["signature"], "what the unit IS stays"
    assert not {"import_line", "library_module", "slice_sha", "callability"} & set(unit), unit

    off.setenv("AGENT_EXTRACTION", "1")
    unit = agent_kb.get_kb_block("b1fa548b::load_crimes")["source"]["extracted"]["unit"]
    assert unit["import_line"] == UNIT["import_line"], "the ON half of this test is not real"


def test_off_the_sweep_does_not_join_the_kb(off, monkeypatch):
    """Recorded rather than raised: the call site swallows exceptions, so a join that raised
    would look exactly like a join that never ran."""
    import rag_pipeline.search.agent_kb as kb
    from agent_runtime.supervisor import graph as g

    calls = []

    def record(docs, **kwargs):
        calls.append(len(docs))
        return {"documents": docs, "attached": 0, "folded": 0, "actionable": []}

    monkeypatch.setattr(kb, "attach_kb_to_documents", record)
    g._direct_search_sweep("buffer a road network", ["kb_method_search"])
    assert not calls

    off.setenv("AGENT_EXTRACTION", "1")
    g._direct_search_sweep("buffer a road network", ["kb_method_search"])
    assert calls, "the ON half of this test is not real"
