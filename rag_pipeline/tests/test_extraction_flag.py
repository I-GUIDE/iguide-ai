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
