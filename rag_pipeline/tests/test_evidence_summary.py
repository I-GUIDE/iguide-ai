"""The decider is told what the evidence SAYS, not only how much of it there is.

`topical_coverage` and `top_score` are cheap, deterministic and cannot be talked into anything —
but they are lexical. They cannot separate PySAL accessibility notebooks from DEM sources when
both mention "elevation". Measured live on a self-hosted model: two full search rounds where the
second added nothing, because "is this enough?" was being answered from counts.

The two properties that keep this from making things worse are the ones most of these tests are
about: the summary DESCRIBES rather than ruling on sufficiency, and it never replaces the
deterministic signals it sits beside.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime.supervisor import graph as sg  # noqa: E402


class FakeLLM:
    def __init__(self, reply="These are PySAL accessibility notebooks; none cover elevation."):
        self.reply = reply
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.reply


DOCS = [
    {"title": "PySAL Access", "contents": "Two-step floating catchment area accessibility."},
    {"title": "Chicago hospitals", "contents": "Hospital point locations for Cook County."},
]


@pytest.fixture(autouse=True)
def on(monkeypatch):
    monkeypatch.delenv("AGENT_EVIDENCE_SUMMARY", raising=False)


# --- it produces something useful -------------------------------------------------

def test_the_summary_is_written_from_the_documents():
    llm = FakeLLM()
    out = sg._summarize_evidence(llm, "hospital accessibility", DOCS)
    assert out == llm.reply
    prompt = llm.prompts[0]
    assert "hospital accessibility" in prompt
    assert "PySAL Access" in prompt and "floating catchment" in prompt


def test_the_prompt_forbids_a_verdict():
    """The decider owns 'is this enough'. A summary that answered it would collapse two
    independent checks into one and hand a weak model's opinion the final word."""
    llm = FakeLLM()
    sg._summarize_evidence(llm, "q", DOCS)
    prompt = llm.prompts[0].lower()
    assert "do not" in prompt and "sufficient" in prompt
    assert "describe only" in prompt


def test_it_reaches_the_decider_beside_the_lexical_signals():
    state = {"query": "hospital accessibility", "evidence": DOCS,
             "evidence_summary": "Accessibility notebooks; no elevation data."}
    distilled = sg._distill(state)
    assert distilled["evidence_summary"] == "Accessibility notebooks; no elevation data."
    # Beside, NOT instead of: a wrong summary must be something the decider can disagree with.
    assert "topical_coverage" in distilled and "document_count" in distilled
    assert "evidence_titles" in distilled


def test_the_decider_is_told_how_to_weigh_it():
    captured = {}

    class Recorder:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return '{"next": "done", "reason": "enough"}'

    decide = sg.default_decide_fn(llm=Recorder())
    decide({"query": "compute a DEM"}, {"available_actions": ["search", "done"],
                                        "evidence_summary": "No DEM sources found."})
    prompt = captured["prompt"]
    assert "evidence_summary" in prompt
    assert "names a specific gap" in prompt           # not "the count looks small"
    assert "retrieval cannot help" in prompt          # tool work is not a literature question


# --- it never becomes a precondition ----------------------------------------------

def test_no_llm_no_summary():
    assert sg._summarize_evidence(None, "q", DOCS) is None


def test_no_documents_no_call():
    llm = FakeLLM()
    assert sg._summarize_evidence(llm, "q", []) is None
    assert llm.prompts == []      # and no wasted round trip


def test_a_failing_model_does_not_break_the_turn():
    class Broken:
        def invoke(self, prompt):
            raise RuntimeError("model is down")

    assert sg._summarize_evidence(Broken(), "q", DOCS) is None


def test_an_empty_reply_is_none_not_an_empty_string():
    assert sg._summarize_evidence(FakeLLM(reply="   "), "q", DOCS) is None


def test_it_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("AGENT_EVIDENCE_SUMMARY", "0")
    llm = FakeLLM()
    assert sg._summarize_evidence(llm, "q", DOCS) is None
    assert llm.prompts == []      # off means no cost, not a discarded result


def test_the_summary_is_capped():
    """It rides in every later decision prompt, so an unbounded one crowds out the rest."""
    out = sg._summarize_evidence(FakeLLM(reply="x" * 5000), "q", DOCS)
    assert out is not None and len(out) <= sg._EVIDENCE_SUMMARY_MAX_CHARS


def test_only_a_bounded_slice_of_each_document_is_sent():
    llm = FakeLLM()
    sg._summarize_evidence(llm, "q", [{"title": "Big", "contents": "y" * 9000}])
    assert len(llm.prompts[0]) < 9000


def test_a_missing_summary_is_simply_absent():
    """Every deployment before this one has no summary; the decider must still work."""
    distilled = sg._distill({"query": "q", "evidence": DOCS})
    assert distilled["evidence_summary"] is None
    assert distilled["document_count"] == 2
