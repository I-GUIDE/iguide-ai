"""Record and replay of the own client's model calls (stage 96), against a local stub server.

The guarantees: a recorded request is answered with no network at all; ids minted per run are
numbered, so a replayed answer names THIS run's file; durations and timestamps do not break a
match; strict replay refuses a new request and replay_or_live sends it live without writing it;
repeats replay in recorded order; streams replay; a deployment is never recorded; no header or
key reaches the cassette; and a replayed call is marked as such on its usage event.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

import pytest

from agent_runtime import model_cassette
from agent_runtime.model_client import ChatClient, ModelCallError


def _reply(content: Any = "done", tool_args: Any = None) -> Dict[str, Any]:
    message: Dict[str, Any] = {"role": "assistant", "content": content}
    if tool_args is not None:
        message["tool_calls"] = [{"id": "call_1", "type": "function", "function": {
            "name": "add_map_layer", "arguments": json.dumps(tool_args)}}]
    return {"id": "chatcmpl-abcdef123456", "object": "chat.completion", "model": "deepseek-v4-flash",
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}}


class Stub:
    def __init__(self) -> None:
        self.replies: List[Any] = [_reply()]
        self.requests = 0
        self.headers: List[Dict[str, str]] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a: Any) -> None:
                pass

            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.headers.append(dict(self.headers))
                reply = stub.replies[min(stub.requests, len(stub.replies) - 1)]
                stub.requests += 1
                if body.get("stream"):
                    chunks = reply
                    data = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks)
                    data += b"data: [DONE]\n\n"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                out = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self) -> None:
        self.server.shutdown()


@pytest.fixture
def stub():
    s = Stub()
    yield s
    s.close()


@pytest.fixture(autouse=True)
def _local_mode(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "local")
    model_cassette._ACTIVE.clear()
    yield
    model_cassette._ACTIVE.clear()


def _use(monkeypatch, directory: Path, mode: str) -> None:
    monkeypatch.setenv(model_cassette.DIR_ENV, str(directory))
    monkeypatch.setenv(model_cassette.MODE_ENV, mode)
    model_cassette._ACTIVE.clear()


def _body(file_id: str = "file_aaaaaaaaaaaa", duration: float = 1.25,
          when: str = "2026-11-17T10:00:00Z", text: str = "where is Urbana") -> Dict[str, Any]:
    tool_result = json.dumps({"ok": True, "file_id": file_id, "duration_s": duration,
                              "finished_at": when, "path": f"/tmp/agent-work/{file_id}.geojson"})
    return {"model": "deepseek-v4-flash", "temperature": 0.0, "messages": [
        {"role": "system", "content": "You are the code peer."},
        {"role": "user", "content": text},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_0", "type": "function",
            "function": {"name": "admin_boundary", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_0", "content": tool_result}]}


def test_a_recorded_request_is_replayed_with_no_network(stub, monkeypatch, tmp_path):
    _use(monkeypatch, tmp_path, "record")
    recorded = ChatClient(stub.base_url, "sk-secret-key", max_retries=0).complete(_body())
    assert stub.requests == 1
    stub.close()                                   # nothing listens any more
    _use(monkeypatch, tmp_path, "replay")
    replayed = ChatClient(stub.base_url, "sk-secret-key", max_retries=0, timeout=2).complete(_body())
    assert replayed.pop("x_replayed") is True
    assert replayed == recorded


def test_this_runs_ids_paths_durations_and_times_still_match(stub, monkeypatch, tmp_path):
    stub.replies = [_reply(content=None, tool_args={"file_id": "file_aaaaaaaaaaaa", "render": "auto"})]
    _use(monkeypatch, tmp_path, "record")
    ChatClient(stub.base_url, "k", max_retries=0).complete(_body())
    _use(monkeypatch, tmp_path, "replay")
    live = _body(file_id="file_bbbbbbbbbbbb", duration=7.9, when="2026-11-18T03:21:09Z")
    replayed = ChatClient(stub.base_url, "k", max_retries=0).complete(live)
    args = json.loads(replayed["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert args == {"file_id": "file_bbbbbbbbbbbb", "render": "auto"}, \
        "the answer names the live run's file, not the recorded run's"
    assert stub.requests == 1


def test_strict_replay_refuses_a_new_request(stub, monkeypatch, tmp_path):
    _use(monkeypatch, tmp_path, "record")
    ChatClient(stub.base_url, "k", max_retries=0).complete(_body())
    _use(monkeypatch, tmp_path, "replay")
    with pytest.raises(ModelCallError, match="cassette miss"):
        ChatClient(stub.base_url, "k", max_retries=0).complete(_body(text="where is Peoria"))
    assert stub.requests == 1
    lines = [json.loads(l) for l in (tmp_path / model_cassette.LOG_NAME).read_text().splitlines()]
    assert [(e["hit"], e["served"]) for e in lines] == [(None, "live"), (False, "refused")]
    assert lines[-1]["caller"].startswith("You are the code peer")


def test_replay_or_live_sends_a_miss_live_and_keeps_the_cassette_clean(stub, monkeypatch, tmp_path):
    _use(monkeypatch, tmp_path, "record")
    ChatClient(stub.base_url, "k", max_retries=0).complete(_body())
    entries = sorted(p.name for p in tmp_path.glob("*.json"))
    _use(monkeypatch, tmp_path, "replay_or_live")
    client = ChatClient(stub.base_url, "k", max_retries=0)
    assert client.complete(_body()).get("x_replayed") is True
    assert "x_replayed" not in client.complete(_body(text="where is Peoria"))
    assert stub.requests == 2
    assert sorted(p.name for p in tmp_path.glob("*.json")) == entries
    s = model_cassette.summarize(tmp_path)
    assert (s["calls"], s["hits"], s["hit_rate"]) == (2, 1, 0.5)
    [turn] = s["turns"].values()
    assert turn["first_miss"]["call"] == 2


def test_repeated_requests_replay_in_recorded_order(stub, monkeypatch, tmp_path):
    stub.replies = [_reply("first"), _reply("second")]
    _use(monkeypatch, tmp_path, "record")
    client = ChatClient(stub.base_url, "k", max_retries=0)
    client.complete(_body())
    client.complete(_body())
    _use(monkeypatch, tmp_path, "replay")
    client = ChatClient(stub.base_url, "k", max_retries=0)
    got = [client.complete(_body())["choices"][0]["message"]["content"] for _ in range(3)]
    assert got == ["first", "second", "second"]


def test_a_stream_is_recorded_and_replayed(stub, monkeypatch, tmp_path):
    chunks = [{"id": "c", "choices": [{"index": 0, "delta": {"content": "Hel"}}]},
              {"id": "c", "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}]},
              {"id": "c", "choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}]
    stub.replies = [chunks]
    _use(monkeypatch, tmp_path, "record")
    live = list(ChatClient(stub.base_url, "k", max_retries=0).stream(_body()))
    _use(monkeypatch, tmp_path, "replay")
    replayed = list(ChatClient(stub.base_url, "k", max_retries=0).stream(_body()))
    assert [{k: v for k, v in c.items() if k != "x_replayed"} for c in replayed] == live
    assert all(c["x_replayed"] for c in replayed) and stub.requests == 1


@pytest.mark.parametrize("mode", ["token", "demo"])
def test_a_deployment_is_never_recorded(stub, monkeypatch, tmp_path, mode):
    monkeypatch.setenv("AGENT_MODE", mode)
    if mode == "token":
        monkeypatch.setenv("AGENT_TOKEN_SECRET", "x" * 32)
    _use(monkeypatch, tmp_path, "record")
    assert model_cassette.active() is None
    ChatClient(stub.base_url, "k", max_retries=0).complete(_body())
    assert list(tmp_path.iterdir()) == []


def test_no_header_or_key_reaches_the_cassette(stub, monkeypatch, tmp_path):
    _use(monkeypatch, tmp_path, "record")
    ChatClient(stub.base_url, "sk-secret-key-123", max_retries=0).complete(_body())
    assert stub.headers[0]["Authorization"] == "Bearer sk-secret-key-123"
    for path in tmp_path.iterdir():
        text = path.read_text()
        assert "sk-secret-key-123" not in text and "Authorization" not in text


def test_the_usage_event_says_the_call_was_replayed(stub, monkeypatch, tmp_path):
    from agent_runtime import streaming_trace as st
    from agent_runtime.model_client_langchain import OwnChatModel

    _use(monkeypatch, tmp_path, "record")
    model = OwnChatModel(model="deepseek-v4-flash", api_key="k", base_url=stub.base_url,
                         temperature=0.0, max_retries=0)
    model.invoke("hi")
    _use(monkeypatch, tmp_path, "replay")
    model = OwnChatModel(model="deepseek-v4-flash", api_key="k", base_url=stub.base_url,
                         temperature=0.0, max_retries=0)
    events: List[Dict[str, Any]] = []
    with st.trace_context(lambda e: events.append(e), agent_dev=True):
        model.invoke("hi")
    usage = [e["data"] for e in events if e["event"] == "llm_usage"]
    assert len(usage) == 1 and usage[0].get("replayed") is True
    assert stub.requests == 1


def test_normalizing_numbers_tokens_in_order_and_blanks_measurements():
    a, numbering_a = model_cassette.normalize(_body("file_aaaaaaaaaaaa", 1.0, "2026-01-01T00:00:00Z"))
    b, numbering_b = model_cassette.normalize(_body("file_cccccccccccc", 99.5, "2027-05-05T12:00:00Z"))
    assert a == b
    assert list(numbering_a) == ["file_aaaaaaaaaaaa", "/tmp/agent-work/file_aaaaaaaaaaaa.geojson"]
    assert model_cassette.request_key(a) == model_cassette.request_key(b)
    c, _ = model_cassette.normalize(_body(text="where is Peoria"))
    assert model_cassette.request_key(c) != model_cassette.request_key(a)


def test_a_miss_can_be_diffed_against_the_nearest_recording(stub, monkeypatch, tmp_path):
    _use(monkeypatch, tmp_path, "record")
    ChatClient(stub.base_url, "k", max_retries=0).complete(_body())
    _use(monkeypatch, tmp_path, "replay_or_live")
    ChatClient(stub.base_url, "k", max_retries=0).complete(_body(text="where is Peoria"))
    [d] = model_cassette.diff_misses(tmp_path)
    assert "Urbana" in d["recorded"] and "Peoria" in d["live"]
    assert d["recorded"][:40] == d["live"][:40]


def test_a_sandbox_run_time_does_not_break_a_match():
    """The first live replay's misses: execute_code's `ran_seconds` (1.4 recorded, 1.3 live)."""
    def body(ran: float) -> Dict[str, Any]:
        result = json.dumps({"ok": True, "ran_seconds": ran, "time_limit_s": 180, "exit_code": 0,
                             "summary": {"mean_ms": ran * 1000}})
        return {"model": "m", "messages": [{"role": "tool", "tool_call_id": "c", "content": result}]}
    a, _ = model_cassette.normalize(body(1.4))
    b, _ = model_cassette.normalize(body(1.3))
    assert a == b
    c, _ = model_cassette.normalize({"model": "m", "messages": [
        {"role": "tool", "tool_call_id": "c", "content": json.dumps({"exit_code": 1})}]})
    assert c != a, "values that are not times still count"


def test_rekey_brings_an_old_cassette_up_to_the_current_rules(stub, monkeypatch, tmp_path):
    _use(monkeypatch, tmp_path, "record")
    ChatClient(stub.base_url, "k", max_retries=0).complete(_body())
    [path] = list(tmp_path.glob("*.json"))
    entry = json.loads(path.read_text())
    # As an older rule set would have stored it: a run time left in, keyed on that.
    entry["request"] = entry["request"].replace('\\"duration_s\\": 0', '\\"duration_s\\": 1.25')
    entry["key"] = model_cassette.request_key(entry["request"])
    path.unlink()
    (tmp_path / f"{entry['key']}.json").write_text(json.dumps(entry))
    assert model_cassette.rekey(tmp_path) == {"rekeyed": 1, "merged": 0, "unchanged": 0}
    _use(monkeypatch, tmp_path, "replay")
    assert ChatClient(stub.base_url, "k", max_retries=0).complete(_body(duration=9.0))["x_replayed"]


def test_the_calls_an_answer_used_keep_their_order():
    """The auditor's lineage lines follow this order; a set of strings iterates differently in
    every process, so two identical runs sent the auditor different prompts (found by replay)."""
    from agent_runtime.supervisor.graph import _used_calls

    numbers = [{"call_id": "call_b"}, {"call_id": "call_a"}, {"call_id": "call_b"}]
    pairs = [({}, {"tool_call_id": "call_c",
                   "content": json.dumps({"ok": True, "map_layer": {"id": "layer_1"}})})]
    assert _used_calls(pairs, numbers) == ["call_b", "call_a", "call_c"]
