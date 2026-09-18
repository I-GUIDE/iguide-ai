"""The durable record of a turn, as opposed to what a viewer happened to watch.

The client's snapshot already stores a trace, but a RENDERED one: tool names with arguments cut
at the display cap, results as headlines like "1 feature · 0.4s". That is right for a person
reading it and useless for re-running the turn. These events are the ones actually emitted, with
their arguments and outcomes intact, which is what reproducing a failure needs and what a
benchmark case is built from.

The assertion that matters most is the first: the recorder must not inherit `agent_dev`. Detail
events are suppressed for a viewer who did not ask for them, and if the record inherited that,
the turns worth studying — the ones nobody was watching closely — would be the ones stored
empty.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime import identity as idm  # noqa: E402
from agent_runtime.streaming_trace import emit_trace_event, trace_context  # noqa: E402
from rag_pipeline import memory_module as mm  # noqa: E402


class FakeOpenSearch:
    def __init__(self):
        self.docs: dict[str, dict] = {}
        self.calls: list[dict] = []

    def index(self, *, index, id, body, refresh=None, request_timeout=None, **_kw):
        # Both recorded, because both are assertions the tests make: a trace must NOT ask the
        # cluster to make it searchable, and must NOT be allowed to block indefinitely.
        self.calls.append({"refresh": refresh, "request_timeout": request_timeout})
        self.docs[id] = dict(body)

    def get(self, *, index, id):
        if id not in self.docs:
            from opensearchpy import NotFoundError
            raise NotFoundError(404, "not found", {})
        return {"_source": dict(self.docs[id])}

    def search(self, *, index, body):
        want = body["query"]["term"]["memory_id.keyword"]
        keep = body.get("_source")
        hits = [{"_id": k, "_source": ({x: v[x] for x in keep if x in v} if keep else dict(v))}
                for k, v in self.docs.items() if v.get("memory_id") == want]
        hits.sort(key=lambda h: self.docs[h["_id"]].get("createdAt") or "", reverse=True)
        return {"hits": {"hits": hits[:body.get("size", 10)]}}


@pytest.fixture(autouse=True)
def store(monkeypatch):
    fake = FakeOpenSearch()
    monkeypatch.setattr(mm, "_get_opensearch_client", lambda: fake)
    monkeypatch.delenv("AGENT_TRACE_MAX_BYTES", raising=False)
    return fake


def as_user(user_id, fn, role=4):
    token = idm.set_user(idm.User(id=user_id, role=role))
    try:
        return fn()
    finally:
        idm.reset_user(token)


# --- the recorder sees everything the viewer does not ------------------------------

@pytest.mark.parametrize("agent_dev", [True, False])
def test_the_recorder_ignores_the_viewers_verbosity(agent_dev):
    """`agent_dev` decides what is STREAMED. It must not decide what is KEPT."""
    streamed, recorded = [], []
    with trace_context(streamed.append, agent_dev=agent_dev, recorder=recorded.append):
        emit_trace_event("status", {"stage": "started"})
        emit_trace_event("tool_call", {"name": "dem_for_region",
                                       "args": {"size": 512, "clip_to_shape": True}})
        emit_trace_event("tool_result", {"name": "dem_for_region", "outcome": "1 layer"})

    kinds = [e["event"] for e in recorded]
    assert kinds == ["status", "tool_call", "tool_result"], "the record is always complete"
    # ...while the stream still honours the setting it exists for.
    streamed_kinds = [e["event"] for e in streamed]
    assert ("tool_call" in streamed_kinds) is agent_dev


def test_the_recorder_keeps_full_tool_arguments():
    """The point of the exercise: the rendered trace truncates these, this one does not."""
    recorded = []
    args = {"bbox": [-88.7, 39.8, -87.9, 40.4], "size": 512, "clip_to_shape": True,
            "name": None, "lon": None, "lat": None}
    with trace_context(lambda _e: None, agent_dev=False, recorder=recorded.append):
        emit_trace_event("tool_call", {"name": "dem_for_region", "args": args})
    assert recorded[0]["data"]["args"] == args


def test_a_failing_sink_does_not_lose_the_record():
    """A broken client stream must not cost the diagnostics for the turn."""
    recorded = []

    def exploding(_event):
        raise RuntimeError("client went away")

    with trace_context(exploding, agent_dev=True, recorder=recorded.append):
        emit_trace_event("tool_call", {"name": "x"})
    assert len(recorded) == 1


def test_a_failing_recorder_does_not_break_the_stream():
    """And the reverse: diagnostics are never worth the answer."""
    streamed = []

    def exploding(_event):
        raise RuntimeError("recorder is broken")

    with trace_context(streamed.append, agent_dev=True, recorder=exploding):
        emit_trace_event("tool_call", {"name": "x"})
    assert len(streamed) == 1


# --- storing and reading back ------------------------------------------------------

def _events(n=3):
    return [{"event": "tool_call", "data": {"name": f"tool_{i}", "args": {"i": i}}}
            for i in range(n)]


def test_a_stored_trace_comes_back_whole():
    res = as_user("alice", lambda: mm.save_turn_trace(
        "mem-1", thread_id="mem-1", query="how high is it?", events=_events(),
        answer="189 m", model="gpt-oss:120b", provider="anvilgpt"))
    assert res["stored"] is True and res["eventCount"] == 3

    got = as_user("alice", lambda: mm.get_turn_trace(res["traceId"]))
    assert got["query"] == "how high is it?"
    assert got["model"] == "gpt-oss:120b" and got["provider"] == "anvilgpt"
    assert [e["data"]["name"] for e in got["events"]] == ["tool_0", "tool_1", "tool_2"]
    assert got["owner_id"] == "alice"


def test_the_summary_list_withholds_the_events():
    """A conversation's traces are large; choosing which turn to open must not download them."""
    as_user("alice", lambda: mm.save_turn_trace("mem-1", thread_id="t", query="one",
                                                events=_events(50)))
    listed = as_user("alice", lambda: mm.list_turn_traces("mem-1"))
    assert len(listed) == 1
    assert listed[0]["event_count"] == 50
    assert "events" not in listed[0]
    assert "events" in as_user("alice", lambda: mm.list_turn_traces("mem-1", include_events=True))[0]


def test_traces_of_other_conversations_are_not_listed():
    as_user("alice", lambda: mm.save_turn_trace("mem-1", thread_id="t", query="mine",
                                                events=_events()))
    as_user("alice", lambda: mm.save_turn_trace("mem-2", thread_id="t", query="other",
                                                events=_events()))
    assert [t["query"] for t in as_user("alice", lambda: mm.list_turn_traces("mem-1"))] == ["mine"]


def test_a_store_failure_never_costs_the_turn(monkeypatch, store):
    """Diagnostics are the least important thing happening; they must fail alone."""
    def boom(**_kw):
        raise RuntimeError("opensearch is down")
    monkeypatch.setattr(store, "index", boom)
    res = as_user("alice", lambda: mm.save_turn_trace("mem-1", thread_id="t", query="q",
                                                      events=_events()))
    assert res["stored"] is False and "opensearch is down" in res["error"]


# --- the size ceiling --------------------------------------------------------------

def test_an_oversized_turn_keeps_its_ends_and_says_what_went(monkeypatch):
    """Trimmed from the MIDDLE.

    The tail holds the outcome and the head holds the question; a turn that blew the limit did
    so in between, which is usually a retry loop repeating itself. Silently dropping either end
    would leave a record that reads as complete and is not.
    """
    monkeypatch.setenv("AGENT_TRACE_MAX_BYTES", "4000")
    res = as_user("alice", lambda: mm.save_turn_trace(
        "mem-1", thread_id="t", query="q", events=_events(400)))
    assert res["stored"] is True
    assert res["dropped"] > 0, "it should have had to drop something"

    got = as_user("alice", lambda: mm.get_turn_trace(res["traceId"]))
    names = [e["data"]["name"] for e in got["events"]]
    assert names[0] == "tool_0", "the question end survives"
    assert names[-1] == "tool_399", "the outcome end survives"
    assert got["dropped_count"] == res["dropped"], "and the record says how much went"


def test_a_turn_within_the_limit_is_untouched():
    res = as_user("alice", lambda: mm.save_turn_trace("mem-1", thread_id="t", query="q",
                                                      events=_events(5)))
    assert res["dropped"] == 0
    assert len(as_user("alice", lambda: mm.get_turn_trace(res["traceId"]))["events"]) == 5


# --- failing fast is part of failing alone -----------------------------------------

def test_a_trace_write_does_not_wait_to_be_searchable(store):
    """`refresh="wait_for"` is right for the snapshot and wrong here.

    The client re-lists conversations the instant a turn ends, so a snapshot has to be
    searchable before its save returns. Nothing lists traces — they are read later, by someone
    debugging — so waiting buys nothing. It costs, though: measured against a RED `chat_traces`
    index on a cluster that had run out of disk, `wait_for` blocked until the 30-second client
    timeout, once per turn, after the answer had already gone out.
    """
    as_user("alice", lambda: mm.save_turn_trace("mem-1", thread_id="t", query="q",
                                                events=_events()))
    assert store.calls[-1]["refresh"] is None


def test_a_trace_write_is_time_bounded(store, monkeypatch):
    """A sick cluster should cost a turn seconds, not the client default."""
    as_user("alice", lambda: mm.save_turn_trace("mem-1", thread_id="t", query="q",
                                                events=_events()))
    assert store.calls[-1]["request_timeout"] == 5.0

    monkeypatch.setenv("AGENT_TRACE_TIMEOUT_SECONDS", "1.5")
    as_user("alice", lambda: mm.save_turn_trace("mem-2", thread_id="t", query="q",
                                                events=_events()))
    assert store.calls[-1]["request_timeout"] == 1.5


def test_a_slow_store_does_not_propagate(store, monkeypatch):
    """The turn is already answered by the time this runs; a timeout must stay contained."""
    def slow(**_kw):
        raise TimeoutError("read timed out")
    monkeypatch.setattr(store, "index", slow)
    res = as_user("alice", lambda: mm.save_turn_trace("mem-1", thread_id="t", query="q",
                                                      events=_events()))
    assert res["stored"] is False and "timed out" in res["error"]
