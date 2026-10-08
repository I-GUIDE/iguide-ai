"""What survives a turn that ends in an exception: the record of it, and a readable reason.

Two defects, both at the very edge of the system where nobody looks.

**The turn was erased.** ``update_memory`` and ``append_session_turn`` sit *after* the ``for``
loop that consumes the event stream. A generator that raises unwinds its consumer's loop, so an
exception skipped every line of recording — the user's own question vanished from the
conversation. The next turn was then built from a history with a hole in it, and a follow-up like
"why did that fail?" referred to something that was no longer there.

**The reason was a stack trace.** The stream emitted ``{"error": str(e)}`` verbatim, so a user
could be shown ``claude CLI was killed by signal 11 (model=sonnet) with no diagnostic output
after 3 attempt(s)`` — accurate, and useless to them. It names an internal tool, offers no
action, and reads like a crash report because it is one.
"""

from __future__ import annotations

import ast

import pytest


# ------------------------------------------------------------------ the turn is recorded

@pytest.fixture()
def service(monkeypatch):
    from agent_runtime import agent_chat_service as A

    recorded: dict = {}
    monkeypatch.setattr(A, "append_session_turn",
                        lambda tid, q, a: recorded.update(thread=tid, question=q, answer=a))
    monkeypatch.setattr(A, "append_session_files", lambda *a, **k: None)
    monkeypatch.setattr(A, "get_session_files", lambda *a, **k: [])
    return A, recorded


def _failing_stream(*_a, **_k):
    yield {"event": "status", "data": {"status": "Started"}}
    yield {"event": "node", "data": {"stage": "search"}}
    raise RuntimeError("claude CLI was killed by signal 11 after 3 attempt(s)")


def test_a_failed_turn_still_records_the_question(service, monkeypatch):
    A, recorded = service
    monkeypatch.setattr(A, "stream_agent_query_events", _failing_stream)

    with pytest.raises(RuntimeError):
        list(A.stream_agent_chat_events(user_input="What Chicago crime data exists?",
                                        use_persistent_memory=False))

    assert recorded, "the turn was erased from history by the exception"
    assert recorded["question"] == "What Chicago crime data exists?"


def test_the_recorded_answer_says_what_went_wrong(service, monkeypatch):
    """A blank answer in history is indistinguishable from a question nobody answered. The record
    should say the request failed, so a later turn reading the history has the context."""
    A, recorded = service
    monkeypatch.setattr(A, "stream_agent_query_events", _failing_stream)

    with pytest.raises(RuntimeError):
        list(A.stream_agent_chat_events(user_input="q", use_persistent_memory=False))

    assert "could not be completed" in recorded["answer"]
    assert "signal 11" in recorded["answer"], "the reason must survive into the record"


def test_the_exception_still_reaches_the_caller(service, monkeypatch):
    """Recording first must not swallow the failure — the API layer's error handling depends on
    seeing it."""
    A, _ = service
    monkeypatch.setattr(A, "stream_agent_query_events", _failing_stream)

    with pytest.raises(RuntimeError, match="signal 11"):
        list(A.stream_agent_chat_events(user_input="q", use_persistent_memory=False))


def test_events_emitted_before_the_failure_are_still_delivered(service, monkeypatch):
    """The client has already rendered them; discarding them would rewind the user's view."""
    A, _ = service
    monkeypatch.setattr(A, "stream_agent_query_events", _failing_stream)

    seen = []
    with pytest.raises(RuntimeError):
        for event in A.stream_agent_chat_events(user_input="q", use_persistent_memory=False):
            seen.append(event.get("event"))
    assert "node" in seen


def test_a_successful_turn_is_unaffected(service, monkeypatch):
    """The guard must be invisible when nothing fails."""
    A, recorded = service

    def good_stream(*_a, **_k):
        yield {"event": "status", "data": {}}
        yield {"event": "completed",
               "data": {"final_answer": "Chicago has 3 crime datasets.", "thread_id": "t1"}}

    monkeypatch.setattr(A, "stream_agent_query_events", good_stream)
    events = list(A.stream_agent_chat_events(user_input="q", use_persistent_memory=False))
    assert events[-1]["event"] == "response"
    assert "3 crime datasets" in recorded["answer"]


# ------------------------------------------------------------------ the reason is readable

def _classifier():
    """Loaded from source so importing it does not boot Flask."""
    src = open("api/server.py", encoding="utf-8").read()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "_classify_stream_error")
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<probe>", "exec"), ns)
    return ns["_classify_stream_error"]


class _Unavailable(Exception):
    pass


_Unavailable.__name__ = "ClaudeCliUnavailable"


@pytest.mark.parametrize("exc,code,retryable", [
    (RuntimeError("claude CLI was killed by signal 11 (model=sonnet) with no diagnostic "
                  "output after 3 attempt(s)"), "llm_transient", True),
    (RuntimeError("claude CLI upstream API returned 429 (model=sonnet), which is transient"),
     "llm_transient", True),
    (_Unavailable("Failed to authenticate. API Error: 401 OAuth access token has expired. "
                  "Re-authenticate to continue."), "llm_unauthenticated", False),
    (ConnectionError("Connection refused to 149.165.155.195:9200"), "backend_unreachable", True),
    (KeyError("documents"), "internal_error", True),
])
def test_an_exception_becomes_an_actionable_message(exc, code, retryable):
    result = _classifier()(exc)
    assert result["code"] == code
    assert result["retryable"] is retryable
    assert "claude" not in result["error"].lower(), (
        "the user-facing message must not name an internal tool")
    assert len(result["error"]) > 40, "a message with no explanation is not an improvement"


def test_the_raw_text_is_kept_for_whoever_is_debugging():
    """Genericising the message must not destroy the diagnosis — it moves to `detail`."""
    result = _classifier()(RuntimeError("killed by signal 11"))
    assert "signal 11" in result["detail"]


def test_an_unauthenticated_backend_is_not_advertised_as_retryable():
    """Telling a user to retry an expired credential sends them into a loop that cannot succeed."""
    result = _classifier()(_Unavailable("401 OAuth access token has expired. Re-authenticate."))
    assert result["retryable"] is False
    assert "retrying will not help" in result["error"].lower()
