"""Another user's ``threadId`` names none of your agent state (S9.15).

S9.14 made the chat edge check the conversation document the service keys memory on. A turn that
sends its OWN ``memoryId`` passes that check whatever ``threadId`` it sends, and the agent keeps
everything between turns under the ``threadId``: the peers' checkpointed threads, session memory,
session files, the action ledger, the sandbox workspace and the lineage graph. The map UI prints
that id in every answer.

Measured on the base (``28294473``), token mode, strict. Alice ran two turns on T with a file;
Bob then sent ``{memoryId: his own, threadId: T}`` and got 200. His analysis peer was handed
Alice's two questions with her file's id and name, his input listed her file id, and his turn
saw her workspace file, her ledger rows and her lineage. With ``AGENT_PERSISTENT_MEMORY=0`` (or
during a store outage, which falls back to it) the edge has no document to check, and a
``threadId``-only turn also got her chat history. Her file's CONTENT stayed refused: the file
store checks owners itself.

The model is stubbed. ``_agent`` does with the thread id it is handed what the real graph does
with it: the analysis peer runs a real ``create_agent`` graph on the real ``DEFAULT_CHECKPOINTER``
under ``agent_config(child_thread_id(thread_id, "analysis"))`` (supervisor/graph.py, analyze
node), the code peer lists ``session_workspace_listing(child_thread_id(thread_id, "codeexec"))``,
routing reads ``get_session_actions(thread_id)`` (``_prior_actions``), and tools record into
``lineage.active()``.
"""
from __future__ import annotations

import io
import json
import uuid

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from werkzeug.datastructures import FileStorage

from agent_runtime import thread_scope
from rag_pipeline.tests.test_conversation_delete import (ALICE, BOB, client_as,  # noqa: F401
                                                         store)
from rag_pipeline.tests.test_memory_ownership import as_user


class _Peer(BaseChatModel):
    """The analysis peer's model: records what it was shown, and answers."""

    seen: list = []

    @property
    def _llm_type(self):
        return "recording"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.seen.append([str(m.content) for m in messages])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="peer ok"))])


@pytest.fixture
def agent(monkeypatch, tmp_path):
    """Run turns; returns the list of what each turn's agent received."""
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "store"))
    monkeypatch.setenv("AGENT_CODE_EXEC_WORK_ROOT", str(tmp_path / "work"))
    from langchain.agents import create_agent

    from agent_runtime import agent_chat_service as acs
    from agent_runtime import lineage
    from agent_runtime import session_memory as sm
    from agent_runtime.code_execution import session_workspace_dir, session_workspace_listing
    from agent_runtime.executor_factory import (DEFAULT_CHECKPOINTER, agent_config, child_thread_id,
                                                resolve_thread_id)

    for d in (sm._STORE, sm._FILES, sm._ACTIONS, lineage._GRAPHS):
        d.clear()
    model = _Peer(seen=[])
    peer = create_agent(model=model, tools=[], checkpointer=DEFAULT_CHECKPOINTER)
    turns = []

    def run(user_input, *, chat_history=None, thread_id=None, **_kw):
        thread_id = resolve_thread_id(thread_id, DEFAULT_CHECKPOINTER)   # as graph_runtime does
        tag = user_input.split()[0]
        seen = {"thread_id": thread_id, "input": user_input,
                "history": [m.get("content") for m in chat_history or []]}
        model.seen.clear()
        peer.invoke({"messages": [{"role": "user", "content": user_input}]},
                    agent_config(child_thread_id(thread_id, "analysis")))
        seen["peer"] = model.seen[-1]
        ws = child_thread_id(thread_id, "codeexec")
        seen["workspace"] = [i["name"] for i in session_workspace_listing(ws)]
        seen["ledger"] = [a.get("tool") for a in sm.get_session_actions(thread_id)]
        g = lineage.active()
        seen["lineage"] = sorted(g.nodes) if g is not None else []
        # What a real turn leaves behind for the next one.
        (session_workspace_dir(ws) / f"{tag}_result.csv").write_text("x\n1\n")
        sm.append_session_actions(thread_id, [{"tool": f"{tag}_tool", "ok": True}])
        if g is not None:
            g.add_node(f"src:{tag}", "source")
        turns.append(seen)
        return {"final_answer": f"{tag}'s answer", "thread_id": thread_id,
                "available_skills": [], "route_trace": {}}

    monkeypatch.setattr(acs, "run_agent_query", run)
    monkeypatch.setattr(acs, "_llm_for_request", lambda *a, **k: None)
    return turns


def post(user, query, **body):
    res = client_as(user).post("/agent/chat", json={"userQuery": query, **body})
    return res.status_code, res.get_json()


def upload(user, thread):
    from agent_runtime import file_store as fs
    tok = fs.set_session(thread)
    try:
        return as_user(user, lambda: fs.save_uploaded_file(FileStorage(
            io.BytesIO(b"id,income\n1,90000\n"), filename="alice_payroll.csv")))["file_id"]
    finally:
        fs.reset_session(tok)


def new_id():
    return f"sess-{uuid.uuid4()}"


def alice_works_on(thread):
    fid = upload(ALICE, thread)
    assert post(ALICE, "alice what is my payroll total?", threadId=thread, fileIds=[fid])[0] == 200
    return fid


def nothing_of_alices(seen, fid):
    blob = json.dumps(seen)
    assert "alice" not in blob and fid not in blob, json.dumps(seen, indent=1)


# --- the hole ------------------------------------------------------------------------------

def test_bobs_own_memory_id_with_alices_thread_reaches_none_of_her_state(store, agent):
    """The measured case: every store the agent keys on the thread id handed Bob's turn
    Alice's. The edge passes this turn, correctly, because the memoryId IS his."""
    t, bob = new_id(), new_id()
    fid = alice_works_on(t)
    assert post(BOB, "bob hello", threadId=bob)[0] == 200
    status, body = post(BOB, "bob what did we do?", memoryId=bob, threadId=t)
    assert status == 200
    seen = agent[-1]
    assert seen["thread_id"] != agent[0]["thread_id"]
    assert seen["peer"] == ["bob what did we do?"], "Alice's checkpointed peer thread"
    assert seen["workspace"] == [], "Alice's sandbox workspace"
    assert seen["ledger"] == [], "Alice's action ledger"
    assert seen["lineage"] == [], "Alice's lineage graph"
    nothing_of_alices(seen, fid)
    assert body["threadId"] == t, "the client keeps the id it sent"


@pytest.mark.parametrize("why", ["persistence off", "store outage"])
def test_with_no_document_to_check_a_thread_only_turn_gets_none_of_her_history(
        store, agent, monkeypatch, why):
    """With no conversation document the edge cannot check a threadId at all, and session memory
    is the turn's only history. Before, Bob's first turn got Alice's questions and answers."""
    if why == "persistence off":
        monkeypatch.setenv("AGENT_PERSISTENT_MEMORY", "0")
    else:
        def down(**_kw):
            raise ConnectionError("cluster unreachable")
        monkeypatch.setattr(store, "get", down)
    t = new_id()
    fid = alice_works_on(t)
    assert post(ALICE, "alice and the median?", threadId=t)[0] == 200
    assert agent[-1]["history"] == ["alice what is my payroll total?", "alice's answer"], \
        "control: Alice's session memory carries her conversation"
    status, _ = post(BOB, "bob what did we do?", threadId=t)
    assert status == 200
    assert agent[-1]["history"] == []
    nothing_of_alices(agent[-1], fid)


def test_bobs_turn_leaves_nothing_in_alices_state(store, agent):
    """The other direction: what Bob's turn wrote under her thread id was in her next turn."""
    t, bob = new_id(), new_id()
    alice_works_on(t)
    post(BOB, "bob hello", threadId=bob)
    post(BOB, "bob injected", memoryId=bob, threadId=t)
    assert post(ALICE, "alice next", threadId=t, memoryId=t)[0] == 200
    blob = json.dumps(agent[-1])
    assert "bob" not in blob, json.dumps(agent[-1], indent=1)


# --- what must not change ------------------------------------------------------------------

@pytest.mark.parametrize("persist", ["1", "0"])
def test_alice_keeps_her_own_conversation(store, agent, monkeypatch, persist):
    """Scoping changes whose state it is, not whether a conversation has one."""
    monkeypatch.setenv("AGENT_PERSISTENT_MEMORY", persist)
    t = new_id()
    fid = alice_works_on(t)
    body = dict(threadId=t) if persist == "0" else dict(threadId=t, memoryId=t)
    assert post(ALICE, "alice and the median?", **body)[0] == 200
    first, second = agent
    assert second["thread_id"] == first["thread_id"]
    assert second["peer"][0].startswith("alice what is my payroll total?")
    assert second["workspace"] == ["alice_result.csv"]
    assert second["ledger"] == ["alice_tool"]
    assert second["lineage"] == ["src:alice"]
    assert fid in second["input"] and "alice_payroll.csv" in second["input"]
    assert second["history"] == ["alice what is my payroll total?", "alice's answer"]


def test_with_nobody_signed_in_the_key_is_the_clients_id(store, agent):
    """Dev and demo identify nobody, and the service key is nobody: their keys are unchanged."""
    from agent_runtime import agent_chat_service as acs
    acs.run_agent_chat(user_input="dev hello", thread_id="dev-thread-1",
                       use_persistent_memory=False)
    assert agent[-1]["thread_id"] == "dev-thread-1"


def test_the_service_key_cannot_spell_out_a_users_key(store, agent):
    """An unsigned caller that sends a scoped key gets it scoped again, not the user's state."""
    from agent_runtime import agent_chat_service as acs
    t = new_id()
    alice_works_on(t)
    alices_key = agent[0]["thread_id"]
    acs.run_agent_chat(user_input="svc what did we do?", thread_id=alices_key,
                       use_persistent_memory=False)
    assert agent[-1]["thread_id"] != alices_key
    assert agent[-1]["workspace"] == [] and agent[-1]["ledger"] == []


def test_a_signed_in_turn_with_no_id_keeps_its_state_on_the_next(store, agent):
    """A guard on this stage's own change, so it passes on the base. The graph generates an id
    when given none, and it would be UNSCOPED: the next turn, sending it back, would be scoped and
    find nothing of the first. So the service generates it for a signed-in turn."""
    from agent_runtime import agent_chat_service as acs
    first = as_user(ALICE, lambda: acs.run_agent_chat(user_input="alice one",
                                                      use_persistent_memory=False))
    client = first["thread_id"]
    assert client and thread_scope.SEP not in client
    as_user(ALICE, lambda: acs.run_agent_chat(user_input="alice two", thread_id=client,
                                              use_persistent_memory=False))
    assert agent[1]["thread_id"] == agent[0]["thread_id"]
    assert agent[1]["ledger"] == ["alice_tool"]


def test_the_client_never_sees_the_scoped_key(store, monkeypatch, tmp_path):
    """The map UI adopts a thread_id from ANY event as the conversation's id, so one scoped key
    in the stream would come back as the next turn's threadId and be scoped twice. Driven through
    the real graph_runtime stream, with only the orchestrator graph stubbed."""
    from agent_runtime import graph_runtime
    from agent_runtime import agent_chat_service as acs
    got = []

    class _Graph:
        def invoke(self, state):
            got.append(state["thread_id"])
            return {"final_answer": "ok", "orchestration_result": {}}

    monkeypatch.setattr(graph_runtime, "build_orchestrator_graph", lambda **kw: _Graph())
    monkeypatch.setattr(acs, "_llm_for_request", lambda *a, **k: None)
    t = new_id()
    res = client_as(ALICE).post("/agent/chat/stream", json={"userQuery": "hi", "threadId": t})
    frames = res.get_data(as_text=True)
    assert got and got[0] != t and got[0].startswith(t), got
    assert got[0] not in frames, "a scoped key reached the client"
    assert f'"threadId": "{t}"' in frames or f'"threadId":"{t}"' in frames
    assert f'"thread_id": "{t}"' in frames or f'"thread_id":"{t}"' in frames


# --- the key itself ------------------------------------------------------------------------

def test_two_users_one_id_two_keys_and_both_unscope_to_it():
    a = as_user(ALICE, lambda: thread_scope.scoped("sess-1"))
    b = as_user(BOB, lambda: thread_scope.scoped("sess-1"))
    assert len({a, b, "sess-1"}) == 3
    assert ALICE not in a, "the key carries a digest, not the platform id"
    for key in (a, b, f"{a}::analysis"):
        assert thread_scope.unscoped(key).split("::")[0] == "sess-1"
    assert thread_scope.unscoped(f"{a}::analysis") == "sess-1::analysis"
    assert thread_scope.scoped(None) is None and thread_scope.unscoped(None) is None
    assert thread_scope.scoped("plain") == "plain"
    assert thread_scope.unscoped(thread_scope.scoped(a)) == a
