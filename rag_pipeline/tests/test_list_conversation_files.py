"""The agent could not answer "what have you saved?" — it had no tool for it.

It answered from the transcript instead, and the transcript is not a record: an earlier turn's
links get trimmed out of context, so a conversation that had produced four files was told it had
produced one, with the other three described as "not available in the current verifiable
records". The grounding audit caught it, which is the only reason it was visible at all.

These tests pin the two halves of the fix: the listing exists, and it is scoped to the
conversation asking — including the part that scoping alone gets wrong, since find_files' default
deliberately shows unowned legacy records so saved embeddings stay reusable.
"""

from __future__ import annotations

import json

import pytest

from agent_runtime import file_store
from agent_runtime.langchain_file_tools import (list_conversation_files_tool,
                                                make_langchain_file_tools)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("AGENT_PUBLIC_BASE_URL", "")
    return tmp_path


def _write(session, name, text="x"):
    token = file_store.set_session(session)
    try:
        return file_store.create_output_file(name, text)
    finally:
        file_store.reset_session(token)


def _listing(session, **kw):
    token = file_store.set_session(session)
    try:
        return json.loads(list_conversation_files_tool(**kw))
    finally:
        file_store.reset_session(token)


def test_it_lists_everything_this_conversation_made(store):
    _write("sess-a", "Savoy_village.geojson")
    _write("sess-a", "savoy_village_gse_embedding.csv")
    _write("sess-a", "savoy_village_gse_embedding.png")

    out = _listing("sess-a")

    assert out["ok"] is True
    assert out["count"] == 3, "the whole point: not just the one the model happened to remember"
    assert {f["filename"] for f in out["files"]} == {
        "Savoy_village.geojson",
        "savoy_village_gse_embedding.csv",
        "savoy_village_gse_embedding.png",
    }
    assert all(f["file_id"] and f["download_url"] for f in out["files"])


def test_another_conversation_is_not_listed(store):
    _write("sess-a", "mine.geojson")
    _write("sess-b", "theirs.geojson")

    assert [f["filename"] for f in _listing("sess-a")["files"]] == ["mine.geojson"]
    assert [f["filename"] for f in _listing("sess-b")["files"]] == ["theirs.geojson"]


def test_legacy_files_are_reusable_but_not_claimed_as_ours(store):
    """The two scopes pull opposite ways, so both are pinned here.

    An unstamped record predates sessions. find_files must keep offering it — 82 saved embedding
    packages the demo reuses have no owner — while the conversation listing must not report it as
    something this conversation saved.
    """
    _write(None, "legacy_vectors.npz")          # written before sessions existed
    _write("sess-a", "mine.geojson")

    token = file_store.set_session("sess-a")
    try:
        reusable = {r["filename"] for r in file_store.find_files(limit=50)}
        owned = {r["filename"] for r in file_store.find_files(limit=50, include_unowned=False)}
    finally:
        file_store.reset_session(token)

    assert reusable == {"legacy_vectors.npz", "mine.geojson"}
    assert owned == {"mine.geojson"}
    assert [f["filename"] for f in _listing("sess-a")["files"]] == ["mine.geojson"]


def test_an_empty_conversation_says_so_rather_than_showing_everyone_elses(store):
    _write("sess-other", "not-yours.geojson")
    _write(None, "legacy.npz")

    out = _listing("sess-a")

    assert out["count"] == 0
    assert "Nothing has been saved in this conversation yet" in out["note"]


def test_an_unbound_request_says_the_scope_is_unknown(store):
    """A CLI run binds no session. Reporting "no files" would be a different kind of wrong."""
    _write(None, "legacy.npz")

    out = json.loads(list_conversation_files_tool())

    assert "scope_unknown" in out


def test_a_name_narrows_the_listing(store):
    _write("sess-a", "Savoy_village.geojson")
    _write("sess-a", "urbana_vectors.npz")

    assert [f["filename"] for f in _listing("sess-a", name="savoy")["files"]] == [
        "Savoy_village.geojson"], "matching is a case-insensitive substring"


def test_the_listing_is_truncated_out_loud(store):
    for i in range(5):
        _write("sess-a", f"f{i}.txt")

    out = _listing("sess-a", limit=3)

    assert out["count"] == 3
    assert "truncated" in out, "a silent cut here reads as a complete answer, which is the bug"


def test_the_tool_is_registered_and_reachable(store):
    """Registered is not the same as reachable: tool_policy filters against FILE_TOOL_NAMES, so a
    name missing from that set is documented and stripped for every intent."""
    from agent_runtime.graph_state import FILE_TOOL_NAMES
    from agent_runtime.tool_policy import select_allowed_tools

    assert "list_conversation_files" in {t.name for t in make_langchain_file_tools()}
    assert "list_conversation_files" in FILE_TOOL_NAMES
    for intent in ("analysis_task", "code_task", "general_discovery", "hybrid"):
        assert "list_conversation_files" in select_allowed_tools(
            intent, ["keyword_search", "list_conversation_files"]), intent


# --- the tool has to reach the peer that answers the question ------------------------------
#
# It did not. The full file toolset is attached only `if input_file_ids` — only on an upload
# turn — and this conversation uploaded nothing: it made a boundary and an embedding. So the
# analyse peer had no file tool at all, wrote `import os; os.listdir('.')` in execute_code, and
# listed the sandbox working directory. The saved script it wrote to do it then appeared in the
# answer as one of the conversation's artifacts.

def _peer_tools(monkeypatch, which, input_file_ids=None):
    import agent_runtime.executor_factory as ef
    from agent_runtime.supervisor import graph as g

    captured = {}

    def _fake_build(**kw):
        captured["tools"] = [str(getattr(t, "name", ""))
                             for t in (kw.get("preloaded_tools") or [])]
        raise RuntimeError("far enough")

    monkeypatch.setattr(ef, "build_agent_executor", _fake_build)
    factory = g.default_analyze_fn if which == "analyze" else g.default_code_fn
    try:
        factory(llm=object(), input_file_ids=input_file_ids)(
            "what files have you saved?", [], {"query": "q", "thread_id": "t1"})
    except Exception:  # noqa: BLE001 - the spy raises to stop before the LLM call
        pass
    return set(captured.get("tools", []))


@pytest.mark.parametrize("which", ["analyze", "code"])
def test_the_peer_has_the_listing_with_no_upload(monkeypatch, which):
    """The regression itself: no upload is the normal case, not the exceptional one."""
    assert "list_conversation_files" in _peer_tools(monkeypatch, which)


@pytest.mark.parametrize("which", ["analyze", "code"])
def test_and_still_has_it_when_a_file_is_attached(monkeypatch, which):
    """The upload path adds the full toolset, which carries this tool too — exactly once."""
    assert "list_conversation_files" in _peer_tools(monkeypatch, which,
                                                    input_file_ids=["file_abc"])


def test_the_upload_path_does_not_bind_it_twice(monkeypatch):
    import agent_runtime.executor_factory as ef
    from agent_runtime.supervisor import graph as g

    captured = {}
    monkeypatch.setattr(ef, "build_agent_executor",
                        lambda **kw: captured.setdefault(
                            "t", [str(getattr(t, "name", ""))
                                  for t in (kw.get("preloaded_tools") or [])]))
    try:
        g.default_analyze_fn(llm=object(), input_file_ids=["file_abc"])(
            "q", [], {"query": "q", "thread_id": "t1"})
    except Exception:  # noqa: BLE001
        pass
    assert captured.get("t", []).count("list_conversation_files") == 1


@pytest.mark.parametrize("prompt_name", ["ANALYSIS_WORKFLOW_PROMPT", "CODE_PEER_PROMPT"])
def test_the_prompt_says_to_ask_the_store_not_the_sandbox(prompt_name):
    """A bound tool the prompt never mentions does not get used — and here the peer had a
    working alternative it preferred. The rule has to name that alternative to rule it out."""
    from agent_runtime.supervisor import prompts

    text = getattr(prompts, prompt_name)
    assert "list_conversation_files" in text
    assert "working directory" in text
