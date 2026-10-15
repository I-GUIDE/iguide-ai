"""Answer validation can be turned off: per request (`validation: false`) or for the server
(`AGENT_VALIDATION=0`). An unchecked answer says it is unchecked; it is never "verified".

Measured on the Chicago SKATER step, 2026-10-15: checking the answer took 4 min 55 s of 7 min
48 s, mostly two grounding audits and a second analysis run for one disputed sentence.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_runtime import client_profile as cp
from agent_runtime import verdict as V


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(cp.VALIDATION_ENV, raising=False)


@pytest.fixture
def validation_off():
    token = cp.set_validation(False)
    yield
    cp.reset_validation(token)


# --- the switch -------------------------------------------------------------------------------

def test_on_by_default():
    assert cp.validation_enabled() is True


@pytest.mark.parametrize("raw", ["0", "false", "off", "no", "OFF"])
def test_the_server_default_turns_it_off(monkeypatch, raw):
    monkeypatch.setenv(cp.VALIDATION_ENV, raw)
    assert cp.validation_enabled() is False


def test_the_request_wins_over_the_server_default(monkeypatch):
    monkeypatch.setenv(cp.VALIDATION_ENV, "0")
    token = cp.set_validation(True)
    try:
        assert cp.validation_enabled() is True
    finally:
        cp.reset_validation(token)


@pytest.mark.parametrize("body,expected", [
    ({"validation": False}, False), ({"validate": "false"}, False), ({"validation": "off"}, False),
    ({"validation": True}, True), ({}, None), ({"validation": None}, None),
    ({"validation": "maybe"}, None)])
def test_the_request_field(body, expected):
    assert cp.requested_validation(body) is expected


# --- the request edge and the result ------------------------------------------------------------

def test_the_result_says_it_was_not_checked(validation_off):
    import api.server as server

    assert server._format_agent_chat_result({"answer": "a"})["validation"] is False


def test_a_checked_result_carries_no_flag():
    import api.server as server

    assert "validation" not in server._format_agent_chat_result({"answer": "a"})


# --- the peers are not offered the audit tool ---------------------------------------------------

def _bound(monkeypatch):
    import langchain.agents as la
    from agent_runtime import executor_factory as ef

    seen = {}
    monkeypatch.setattr(la, "create_agent",
                        lambda *, tools, **_kw: seen.setdefault("tools", [t.name for t in tools]))
    ef.build_agent_executor(llm=object(), preloaded_tools=[
        SimpleNamespace(name="audit_answer_grounding"), SimpleNamespace(name="keyword_search")])
    return seen["tools"]


def test_the_audit_tool_is_not_bound_when_off(monkeypatch, validation_off):
    tools = _bound(monkeypatch)
    assert "audit_answer_grounding" not in tools and "keyword_search" in tools


def test_it_is_bound_when_on(monkeypatch):
    assert "audit_answer_grounding" in _bound(monkeypatch)


# --- the turn -----------------------------------------------------------------------------------

ANALYSIS = {"summary": "Counted crimes per area.", "tool_calls": [{"name": "execute_code"}],
            "tool_results": [{"name": "execute_code", "ok": True,
                              "content": {"ok": True, "stdout": "citywide rate 30.35"}}]}
ANSWER = "The citywide rate is 30.35 per 1,000. Fuller Park leads at 118.0 per 1,000."


def _turn(monkeypatch):
    """A turn whose analysis ran and whose answer states a figure the record lacks (118.0).
    Validation on would audit it and could send the work back; off, neither may happen."""
    from agent_runtime.supervisor import graph as g

    audits, analyses = [], []
    monkeypatch.setattr(g, "audit_answer_grounding",
                        lambda *a, **k: audits.append(1) or {"severity": "none", "issues": []})

    def analyze(*_a, **_k):
        analyses.append(1)
        return ANALYSIS

    decided = []

    def decide(state, _payload):
        decided.append(1)
        return "analyze" if not state.get("actions") else "done"

    graph = g.build_supervisor_graph(
        search_fn=lambda q, s: [], analyze_fn=analyze, synthesize_fn=lambda *a: ANSWER,
        decide_fn=decide, do_rerank=False)
    out = graph.invoke({"query": "rate per 1,000 by community area", "thread_id": "t-valid",
                        "actions": [], "evidence": []})
    return out, audits, analyses


def test_off_the_turn_runs_no_check_and_says_so(monkeypatch, validation_off):
    out, audits, analyses = _turn(monkeypatch)
    assert audits == [], "the grounding audit ran"
    assert len(analyses) == 1, "a re-grounding pass re-ran the analysis"
    assert "118.0" in out["final_answer"], "the number scan cut a statement"
    assert out["final_answer"].rstrip().endswith(f"*{V.VALIDATION_OFF_LINE}*")
    findings = out["verdict"]["findings"]
    assert [f["check"] for f in findings] == ["validation"]
    assert V.status([V.Finding(**f) for f in findings]) == "unverified"


def test_on_the_same_turn_is_audited(monkeypatch):
    out, audits, _ = _turn(monkeypatch)
    assert audits, "control: with validation on the audit runs"
    assert V.VALIDATION_OFF_LINE not in out["final_answer"]


def test_the_unchecked_line_is_quiet_not_a_warning():
    rendered = V.render("Answer.", [V.Finding("validation", V.UNVERIFIABLE,
                                              cp.VALIDATION_OFF_FINDING)])
    assert "Check this answer" not in rendered and V.VALIDATION_OFF_LINE in rendered
