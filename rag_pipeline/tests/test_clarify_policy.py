"""AGENT_CLARIFY_POLICY: never (as before), assume (name the assumption), ask (a gate first).

Offline: the gate's model is a stub, the peers are stubs.
"""
from __future__ import annotations

import json

import pytest

from agent_runtime import clarify_policy as cp
from agent_runtime import session_memory as sm
from agent_runtime.supervisor import graph as g


class _LLM:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return type("M", (), {"content": json.dumps(self.reply)})()


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(cp.POLICY_ENV, raising=False)
    sm.reset_all()
    yield
    sm.reset_all()


_GATE = {}


@pytest.fixture(autouse=True)
def _gate_model(monkeypatch):
    """The gate builds the default model; here that is whatever _graph was handed."""
    from agent_runtime import executor_factory

    monkeypatch.setattr(executor_factory, "build_default_llm", lambda: _GATE["llm"])


def _graph(llm, synth=lambda *a, **k: "the answer"):
    _GATE["llm"] = llm
    return g.build_supervisor_graph(
        search_fn=lambda q, s: [], synthesize_fn=synth,
        decide_fn=lambda *a, **k: ("done", "why"), do_rerank=False, do_audit=False)


def _run(graph, thread="t1", history=None):
    return graph.invoke({"query": "Which grid cell contains Erl both in h3 and s2 formats?",
                         "thread_id": thread, "chat_history": history or [], "actions": [],
                         "evidence": [{"doc_id": "d1", "title": "t", "contents": "c"}]})


def test_unset_is_never_and_adds_nothing():
    assert cp.policy() == "never" and cp.writer_rule() is None
    llm = _LLM({"ask": True, "question": "Which resolution?"})
    out = _run(_graph(llm))
    assert out["final_answer"] == "the answer" and llm.prompts == []   # no gate


def test_an_unknown_value_is_never(monkeypatch):
    monkeypatch.setenv(cp.POLICY_ENV, "sometimes")
    assert cp.policy() == "never"


def test_assume_tells_the_writer_and_never_gates(monkeypatch):
    monkeypatch.setenv(cp.POLICY_ENV, "assume")
    assert "SAY which assumption" in cp.writer_rule()
    llm = _LLM({"ask": True, "question": "Which resolution?"})
    assert _run(_graph(llm))["final_answer"] == "the answer" and llm.prompts == []


def test_ask_returns_the_question_and_does_no_work(monkeypatch):
    monkeypatch.setenv(cp.POLICY_ENV, "ask")
    ran = []
    llm = _LLM({"ask": True, "missing": "H3 resolution and S2 level",
                "question": "Which H3 resolution and S2 level do you want?"})
    out = _run(_graph(llm, synth=lambda *a, **k: ran.append(1) or "work"))
    assert out["final_answer"] == "Which H3 resolution and S2 level do you want?"
    assert ran == [] and out["clarify"]["ask"] is True
    assert "Erl" in llm.prompts[0]


def test_ask_proceeds_when_the_gate_sees_nothing_open(monkeypatch):
    monkeypatch.setenv(cp.POLICY_ENV, "ask")
    out = _run(_graph(_LLM({"ask": False, "question": ""})))
    assert out["final_answer"] == "the answer" and out["clarify"]["ask"] is False


def test_it_asks_at_most_once_per_conversation(monkeypatch):
    monkeypatch.setenv(cp.POLICY_ENV, "ask")
    llm = _LLM({"ask": True, "question": "Which one?"})
    assert _run(_graph(llm), thread="t2")["final_answer"] == "Which one?"
    again = _run(_graph(llm), thread="t2", history=[{"userQuery": "q", "answer": "Which one?"}])
    assert again["final_answer"] == "the answer" and len(llm.prompts) == 1


def test_a_broken_gate_proceeds(monkeypatch):
    monkeypatch.setenv(cp.POLICY_ENV, "ask")

    class Boom:
        def invoke(self, prompt):
            raise RuntimeError("model down")

    assert _run(_graph(Boom()))["final_answer"] == "the answer"


def test_ask_true_without_a_question_does_not_ask(monkeypatch):
    monkeypatch.setenv(cp.POLICY_ENV, "ask")
    assert _run(_graph(_LLM({"ask": True, "question": ""})))["final_answer"] == "the answer"


def test_the_gate_sees_the_conversation_and_the_attached_files(monkeypatch):
    monkeypatch.setattr(cp, "_file_names", lambda thread: ["schools.geojson"])
    llm = _LLM({"ask": False})
    cp.gate("What is the mean slope of the elevation model I attached?",
            [{"userQuery": "hello", "answer": "hi"}], "t3", llm)
    assert "schools.geojson" in llm.prompts[0] and "User: hello" in llm.prompts[0]


def test_the_writer_rule_reaches_the_synthesis_prompt(monkeypatch):
    monkeypatch.setenv(cp.POLICY_ENV, "assume")
    seen = []

    class Writer:
        def invoke(self, prompt):
            seen.append(prompt)
            return type("M", (), {"content": "ok"})()

    g.default_synthesize_fn(llm=Writer())("q", [], {"summary": "s"}, None)
    assert cp.ASSUME_RULE in seen[0]
