"""SSE keepalive: quiet agent stretches must not leave the stream byte-silent
(Node fetch kills a body silent for 300s -> "Body Timeout Error")."""

from __future__ import annotations


def test_stream_emits_keepalive_during_quiet_stretch(monkeypatch):
    import time
    import agent_runtime.graph_runtime as gr
    import agent_runtime.strategy as strat

    monkeypatch.setenv("AGENT_STREAM_HEARTBEAT_SECONDS", "1")   # min clamp is 1s

    def slow_strategy(kind=None):
        def run(query, chat_history, cfg):
            time.sleep(2.4)   # a "long LLM/sandbox run" with no events
            return {"orchestration_result": {"messages": []}, "final_answer": "done",
                    "available_agent_names": []}
        return run
    monkeypatch.setattr(strat, "get_orchestration_strategy", slow_strategy)

    events = list(gr.stream_agent_query_events("slow question"))
    names = [e.get("event") for e in events]
    assert names.count("keepalive") >= 1                     # heartbeat fired while quiet
    assert "completed" in names                              # terminal payload still delivered
    assert names.index("keepalive") < names.index("completed")


def test_keepalive_rendered_as_sse_comment(monkeypatch):
    """The API layer must render keepalive as an SSE comment line (ignored by parsers),
    not drop it (unmapped events yield nothing) and not a named event (would pollute UIs)."""
    import api.server as srv

    def fake_stream(**kwargs):
        yield {"event": "keepalive", "data": {}}
        yield {"event": "response", "data": {"answer": "hi", "thread_id": "t"}}
    monkeypatch.setattr(srv, "stream_agent_chat_events", fake_stream)
    monkeypatch.delenv("AGENT_CHAT_API_KEY", raising=False)

    client = srv.app.test_client()
    resp = client.post("/agent/chat/stream", json={"userQuery": "hi"})
    body = resp.get_data(as_text=True)
    assert ": keepalive\n\n" in body                         # comment line present
    assert "event: keepalive" not in body                    # not a named event
    assert "event: result" in body                           # terminal result still emitted
