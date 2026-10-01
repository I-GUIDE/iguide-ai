"""The code peer can look up an API before writing against it.

Its failure mode is plausible code against a misremembered API — a keyword argument that moved,
a function that returns a tuple now. The knowledge base holds I-GUIDE's own content, not
pysal's or geopandas', and the sandbox has no network, so there was no way to check: the peer
had to guess and discover the mistake by running it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import agent_runtime.executor_factory as ef
from agent_runtime.supervisor import graph as g


def _bound(monkeypatch, input_file_ids=None):
    captured = {}

    def _fake_build(**kw):
        captured["tools"] = [str(getattr(t, "name", "")) for t in (kw.get("preloaded_tools") or [])]
        raise RuntimeError("far enough")

    monkeypatch.setattr(ef, "build_agent_executor", _fake_build)
    try:
        g.default_code_fn(llm=object(), input_file_ids=input_file_ids)(
            "q", [], {"query": "q", "thread_id": "t1"})
    except Exception:
        pass
    return set(captured.get("tools", []))


def test_the_code_peer_can_search_the_web(monkeypatch):
    assert "web_search" in _bound(monkeypatch)


def test_and_can_read_what_it_finds(monkeypatch):
    """Finding a page and being unable to open it is not a capability — the factory's own rule.
    Pinned here because the two are bound through separate gates and could drift apart."""
    names = _bound(monkeypatch)
    assert "web_fetch" in names


def test_both_survive_the_upload_gate(monkeypatch):
    """Attaching a file swaps in a much larger tool set; the web pair must not fall out."""
    names = _bound(monkeypatch, input_file_ids=["file_abc"])
    assert {"web_search", "web_fetch"} <= names


def test_the_kb_tools_are_still_there(monkeypatch):
    """They share one factory call with the web pair, so a mistake in the name filter would
    take these with it."""
    assert {"agent_kb_search", "get_kb_block"} <= _bound(monkeypatch)


def test_the_retrieval_family_is_not_bound_wholesale(monkeypatch):
    """Four names out of twenty-two. keyword_search and semantic_search belong to the search
    peer; binding the family would put its whole schema cost on every code turn."""
    names = _bound(monkeypatch)
    assert "keyword_search" not in names
    assert "semantic_search" not in names
    assert "overpass_search" not in names


def test_the_prompt_says_to_look_up_before_writing(monkeypatch):
    """A bound tool the prompt never mentions does not get used, and the one thing the peer
    must know is that the lookup happens BEFORE execute_code, not inside it."""
    from agent_runtime.supervisor.prompts import CODE_PEER_PROMPT

    assert "web_search" in CODE_PEER_PROMPT and "web_fetch" in CODE_PEER_PROMPT
    assert "NO network" in CODE_PEER_PROMPT
