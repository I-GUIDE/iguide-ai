"""Stage 92: the answer is composed while the supervisor decides whether the turn is done.

Live on 2026-11-13 (Lumen deepseek-v4-flash), "area of Champaign County" spent 1.7 s on the
decider's `done` and then 2.7 s on the writer, one after the other. The writer's call now starts
when the decider's does, and synthesize keeps the draft only if its own inputs are identical.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime import client_profile  # noqa: E402
from agent_runtime.supervisor import speculation  # noqa: E402
from agent_runtime.supervisor.graph import run_supervisor  # noqa: E402


def _fake_llm(prompt: str) -> str:
    return "ok"


class Writer:
    """A synthesize_fn that records which thread called it, when, and with what."""

    def __init__(self, delay: float = 0.0):
        self.calls = []
        self.delay = delay
        self.lock = threading.Lock()

    def __call__(self, q, ev, ar, cr, ch, pa=None):
        start = time.monotonic()
        time.sleep(self.delay)
        with self.lock:
            self.calls.append({"thread": threading.current_thread().name, "start": start,
                               "code": (cr or {}).get("answer"), "analysis": (ar or {}).get("summary")})
        return f"final:{(cr or {}).get('answer', '')}|{(ar or {}).get('summary', '')}"


def _decider(script, sleep_after_peer=0.0, log=None):
    box = {"i": 0}

    def decide(state, distilled):
        if state.get("actions"):
            if log is not None:
                log.append(("decide_start", time.monotonic()))
            time.sleep(sleep_after_peer)
        nxt = script[min(box["i"], len(script) - 1)]
        box["i"] += 1
        if log is not None:
            log.append(("decide_end", time.monotonic()))
        return nxt

    return decide


def _code_fn(q, ev, st):
    return {"answer": "code-answer", "code_result": {"x": 1}}


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    monkeypatch.setenv(speculation.ENV, "1")
    token = client_profile.set_validation(False)
    yield
    client_profile.reset_validation(token)


def test_the_draft_is_written_while_the_decider_decides_and_is_used():
    writer, log = Writer(delay=0.2), []
    state = run_supervisor(
        "write code", llm=_fake_llm, decide_fn=_decider(["code", "done"], 0.4, log),
        search_fn=lambda q, s: [], code_fn=_code_fn, synthesize_fn=writer, do_audit=False)
    assert state["final_answer"].startswith("final:code-answer")
    assert len(writer.calls) == 1                                   # written once, not twice
    call = writer.calls[0]
    assert call["thread"].startswith("speculative-synth")           # by the draft
    decide_start = next(t for k, t in log if k == "decide_start")
    decide_end = [t for k, t in log if k == "decide_end"][-1]
    assert decide_start - 0.05 <= call["start"] < decide_end        # during the decision


def test_a_decider_that_continues_drops_the_draft_and_the_answer_uses_the_new_work():
    writer = Writer()

    def analyze_fn(q, ev, st):
        return {"summary": "analysis-summary"}

    state = run_supervisor(
        "write code then analyse", llm=_fake_llm, decide_fn=_decider(["code", "analyze", "done"]),
        search_fn=lambda q, s: [], code_fn=_code_fn, analyze_fn=analyze_fn,
        synthesize_fn=writer, do_audit=False)
    # The draft after `code` lacked the analysis; the answer must include it.
    assert "analysis-summary" in state["final_answer"]
    assert any(c["analysis"] == "analysis-summary" for c in writer.calls)


def test_switched_off_the_writer_runs_once_in_the_turn_itself(monkeypatch):
    monkeypatch.setenv(speculation.ENV, "0")
    writer = Writer()
    state = run_supervisor(
        "write code", llm=_fake_llm, decide_fn=_decider(["code", "done"]),
        search_fn=lambda q, s: [], code_fn=_code_fn, synthesize_fn=writer, do_audit=False)
    assert state["final_answer"].startswith("final:code-answer")
    assert len(writer.calls) == 1 and not writer.calls[0]["thread"].startswith("speculative-synth")


def test_no_draft_with_validation_on():
    token = client_profile.set_validation(True)
    try:
        writer = Writer()
        run_supervisor(
            "write code", llm=_fake_llm, decide_fn=_decider(["code", "done"]),
            search_fn=lambda q, s: [], code_fn=_code_fn, synthesize_fn=writer, do_audit=False)
        assert writer.calls and not any(c["thread"].startswith("speculative-synth")
                                        for c in writer.calls)
    finally:
        client_profile.reset_validation(token)


# ------------------------------------------------------------------ the draft store itself

def test_a_draft_is_taken_only_for_identical_inputs():
    args = ("q", [{"doc_id": "a"}], {"summary": "s"}, None, [], "note")
    assert speculation.start("turn-1", args, lambda *a: "draft")
    assert speculation.take("turn-1", ("q", [{"doc_id": "b"}], {"summary": "s"}, None, [], "note")) is None
    assert speculation.start("turn-2", args, lambda *a: "draft")
    assert speculation.take("turn-2", args) == "draft"
    assert speculation.take("turn-2", args) is None                  # taken once


def test_dropped_and_failed_drafts_are_not_used():
    args = ("q", [], None, None, [], None)
    speculation.start("turn-3", args, lambda *a: "draft")
    speculation.drop("turn-3")
    assert speculation.take("turn-3", args) is None

    def boom(*a):
        raise RuntimeError("model down")

    speculation.start("turn-4", args, boom)
    assert speculation.take("turn-4", args) is None


def test_inputs_that_cannot_be_compared_never_start_a_draft():
    class Opaque:
        pass

    assert speculation.fingerprint(("q", [Opaque()])) is None
    assert not speculation.start("turn-5", ("q", [Opaque()]), lambda *a: "draft")
