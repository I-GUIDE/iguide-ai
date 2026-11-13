"""Stage 93: the answer writer streams, and the draft reaches the client while it is written.

The turn still returns the whole text, and the checks after the writer see exactly what they
saw. (The capability-answer cache that shared this stage on the `fast-replies` branch was left
out: see the Stage 93 entry in docs/agent-architecture-changes.md.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime import streaming_trace  # noqa: E402
from agent_runtime.supervisor import graph  # noqa: E402


# ------------------------------------------------------------------ the streamed writer

class Chunk:
    def __init__(self, content):
        self.content = content


class StreamingModel:
    stream_usage = False

    def __init__(self, pieces, fail_at=None):
        self.pieces, self.fail_at, self.stream_kwargs, self.invoked = pieces, fail_at, None, 0

    def stream(self, prompt, **kwargs):
        self.stream_kwargs = kwargs
        for i, p in enumerate(self.pieces):
            if self.fail_at == i:
                raise RuntimeError("stream broke")
            yield Chunk(p)

    def invoke(self, prompt):
        self.invoked += 1
        return Chunk("invoked answer")


@pytest.fixture
def drafts(monkeypatch):
    seen = []
    monkeypatch.setattr(streaming_trace, "emit_answer_draft", lambda t: seen.append(t))
    monkeypatch.setattr(graph, "_DRAFT_EVERY_S", 0.0)
    monkeypatch.setenv("AGENT_STREAM_ANSWER", "1")
    return seen


def _write(model):
    return graph.default_synthesize_fn(llm=model)("q", [], {"summary": "s"}, None, [], None)


def test_the_answer_streams_with_its_spacing_and_reports_usage(drafts):
    model = StreamingModel(["The", " area", " is", " 2,584.6 km²."])
    assert _write(model) == "The area is 2,584.6 km²."
    assert model.stream_kwargs == {"stream_usage": True}  # token counts for the coin budget
    assert drafts[0] == "The" and drafts[-1] == "The area is 2,584.6 km²."
    assert model.invoked == 0


def test_a_stream_that_fails_before_any_text_falls_back_to_the_ordinary_call(drafts):
    model = StreamingModel(["The"], fail_at=0)
    assert _write(model) == "invoked answer" and model.invoked == 1


def test_a_stream_that_fails_part_way_raises_as_the_ordinary_call_would(drafts):
    with pytest.raises(RuntimeError):
        _write(StreamingModel(["The", " area"], fail_at=1))


def test_switched_off_or_without_stream_usage_the_writer_is_called_once(drafts, monkeypatch):
    monkeypatch.setenv("AGENT_STREAM_ANSWER", "0")
    model = StreamingModel(["The"])
    assert _write(model) == "invoked answer" and model.stream_kwargs is None
    monkeypatch.setenv("AGENT_STREAM_ANSWER", "1")

    class NoUsage:  # a model that streams but cannot report usage on a stream
        stream_kwargs, invoked = None, 0

        def stream(self, prompt, **kwargs):
            self.stream_kwargs = kwargs
            yield Chunk("The")

        def invoke(self, prompt):
            return Chunk("invoked answer")

    plain = NoUsage()
    assert _write(plain) == "invoked answer" and plain.stream_kwargs is None
    assert drafts == []


def test_a_draft_goes_to_the_client_and_is_not_recorded():
    sent, recorded = [], []
    state = streaming_trace._TraceState(sink=sent.append, handler=None, agent_role="agent")
    state.recorder = recorded.append
    token = streaming_trace._TRACE_STATE.set(state)
    try:
        streaming_trace.emit_answer_draft("The area")
    finally:
        streaming_trace._TRACE_STATE.reset(token)
    assert sent == [{"event": "answer_draft", "data": {"text": "The area"}}]
    assert recorded == []
