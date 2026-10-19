"""Every model call has a request timeout and a retry cap, and a timeout says what it was (Stage 85).

Before this, every chat model was built with no timeout and the client's two automatic retries.
Production, 2026-10-09: Lumen deepseek answered HTTP 500 after about 6 minutes on long-context
calls, retried twice, and three turns stalled 18+ minutes each. A stalled model looked exactly like
progress: the socket stayed open and SSE keepalives kept flowing.

Three things are pinned:

* the limits reach every construction site in executor_factory (a source scan, so a new site
  cannot be added without them), and the built clients really carry them;
* a call to a server that never answers ends at the timeout, retries included, with an
  ``LLMCallTimeout`` that names the limit and the model; streamed and async calls too;
* the supervisor treats that as fatal: it answers with what it has instead of routing back to
  the stalled peer.

The stalled server is a socket on 127.0.0.1 that this file opens; nothing leaves the machine.
"""
from __future__ import annotations

import ast
import asyncio
import inspect
import socket
import threading
import time
from pathlib import Path

import pytest

from agent_runtime import executor_factory as ef
from agent_runtime import llm_timeout
from agent_runtime.llm_timeout import LLMCallTimeout


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("AGENT_LLM_TIMEOUT_SECONDS", "AGENT_LLM_MAX_RETRIES", "AGENT_LLM_PROVIDER",
                "ANVILGPT_URL", "ANVILGPT_MODEL", "LUMEN_URL", "LUMEN_MODEL", "OPENAI_BASE_URL",
                "OPENAI_CHAT_MODEL", "OPENAI_MODEL", "VLLM_API_KEY", "VLLM_MODEL", "VLLM_PROXY",
                "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "LLM_PROVIDER",
                "AGENT_LUMEN_FAILOVER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_KEY", "sk-test")
    monkeypatch.setenv("ANVILGPT_KEY", "sk-test-anvil")
    monkeypatch.setenv("LUMEN_API_KEY", "sk_test_not_a_real_key")
    monkeypatch.setenv("AGENT_MODE", "dev")             # Lumen is offered in dev mode only


# ------------------------------------------------------------------ settings

def test_defaults_are_provider_aware():
    assert llm_timeout.timeout_seconds("lumen") == 240.0
    assert llm_timeout.timeout_seconds("openai") == 240.0
    assert llm_timeout.timeout_seconds(None) == 240.0
    # Under AnvilGPT's 300 s nginx gateway, so the error is ours and says so.
    assert llm_timeout.timeout_seconds("anvilgpt") < 300
    assert llm_timeout.max_retries() == 1


@pytest.mark.parametrize("raw,expected", [("90", 90.0), ("12.5", 12.5), ("0", None),
                                          ("off", None), ("-3", None), ("nope", 240.0)])
def test_the_timeout_can_be_set_or_switched_off(monkeypatch, raw, expected):
    monkeypatch.setenv("AGENT_LLM_TIMEOUT_SECONDS", raw)
    assert llm_timeout.timeout_seconds("lumen") == expected


def test_an_explicit_timeout_applies_to_anvilgpt_too(monkeypatch):
    monkeypatch.setenv("AGENT_LLM_TIMEOUT_SECONDS", "120")
    assert llm_timeout.timeout_seconds("anvilgpt") == 120.0


@pytest.mark.parametrize("raw,expected", [("0", 0), ("1", 1), ("3", 3), ("99", 5), ("-1", 0),
                                          ("x", 1), ("", 1)])
def test_retries_are_configurable_and_clamped(monkeypatch, raw, expected):
    monkeypatch.setenv("AGENT_LLM_MAX_RETRIES", raw)
    assert llm_timeout.max_retries() == expected


# ------------------------------------------------------------------ every construction site

_CONSTRUCTORS = {"ChatOpenAI", "ChatAnthropic", "factory"}


def _calls_without_limits():
    """Model-client constructions in executor_factory that do not spread client_limits(...)."""
    tree = ast.parse(Path(inspect.getsourcefile(ef)).read_text())
    found, missing = 0, []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name not in _CONSTRUCTORS:
            continue
        found += 1
        spread = [k.value for k in node.keywords if k.arg is None]
        if not any(isinstance(v, ast.Call) and getattr(v.func, "id", "") == "client_limits"
                   for v in spread):
            missing.append(f"line {node.lineno}: {name}(...)")
    return found, missing


def test_no_client_is_constructed_without_the_limits():
    found, missing = _calls_without_limits()
    # 5 ChatOpenAI, 1 ChatAnthropic, 1 bearer-token SDK client factory as of Stage 85. A drop
    # means the scan stopped seeing them, not that they went away.
    assert found >= 7, f"only {found} constructions seen; the scan is broken"
    assert not missing, f"model clients built without client_limits(): {missing}"


def _limits_of(llm):
    timeout = getattr(llm, "request_timeout", None)
    if timeout is None:
        timeout = getattr(llm, "default_request_timeout", None)
    client = getattr(llm, "root_client", None) or getattr(llm, "_client", None)
    return timeout, llm.max_retries, client.timeout, client.max_retries


@pytest.mark.parametrize("provider,expected", [("lumen", 240.0), ("anvilgpt", 280.0),
                                               (None, 240.0)])
def test_the_default_model_carries_the_limits(monkeypatch, provider, expected):
    if provider:
        monkeypatch.setenv("AGENT_LLM_PROVIDER", provider)
    monkeypatch.setattr(ef, "_lumen_fallback", lambda: None)
    assert _limits_of(ef.build_default_llm()) == (expected, 1, expected, 1)


def test_the_lumen_failover_model_carries_the_limits(monkeypatch):
    """Stage 74 moves the default to gpt-5.6-luna when the Lumen pool runs low; that client is
    built through build_llm and must not be the one that can hang."""
    monkeypatch.setenv("AGENT_LLM_TIMEOUT_SECONDS", "60")
    monkeypatch.setattr(ef, "_lumen_settings", lambda: None)
    monkeypatch.setattr(ef, "_lumen_fallback",
                        lambda: {"provider": "openai", "model": "gpt-5.6-luna"})
    assert _limits_of(ef.build_default_llm()) == (60.0, 1, 60.0, 1)


@pytest.mark.parametrize("provider,model,expected", [
    ("openai", "gpt-4o-2024-11-20", 240.0),
    ("openai", "gpt-5.6-luna", 240.0),
    ("lumen", "deepseek-v4-flash", 240.0),
    ("anvilgpt", "qwen3.8:27b", 280.0),
])
def test_every_picked_model_carries_the_limits(monkeypatch, provider, model, expected):
    monkeypatch.setenv("AGENT_LLM_MAX_RETRIES", "0")
    assert _limits_of(ef.build_llm(provider=provider, model=model)) == (expected, 0, expected, 0)


def test_anthropic_carries_the_limits(monkeypatch):
    pytest.importorskip("langchain_anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    llm = ef.build_llm(provider="anthropic", model="claude-sonnet-5")
    assert _limits_of(llm) == (240.0, 1, 240.0, 1)


def test_the_bearer_token_clients_keep_the_limits(monkeypatch):
    """A subscription token swaps the SDK clients out after construction; the swapped-in ones
    must carry the same limits, or the swap silently undoes them."""
    pytest.importorskip("langchain_anthropic")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat-test")
    monkeypatch.setenv("AGENT_LLM_TIMEOUT_SECONDS", "77")
    llm = ef.build_llm(provider="anthropic", model="claude-sonnet-5")
    for client in (llm._client, llm._async_client):
        assert (client.timeout, client.max_retries) == (77.0, 1)


def test_no_max_tokens_is_sent_to_a_reasoning_model():
    """The timeout is the lever, not max_tokens: a ceiling truncates a reasoning model's thinking
    into an empty answer (measured; AGENTS.md)."""
    for prov, model in (("lumen", "deepseek-v4-flash"), ("anvilgpt", "qwen3.8:27b")):
        assert ef.build_llm(provider=prov, model=model).max_tokens is None


# ------------------------------------------------------------------ a stalled call ends

class _StalledServer:
    """Accepts connections, reads the request, and answers nothing (or one SSE chunk) — the
    shape of the 2026-10-09 Lumen stall, where nothing came back for six minutes."""

    def __init__(self, *, first_chunk: bool = False):
        self.first_chunk = first_chunk
        self.requests = 0
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self._sock.settimeout(0.2)
        self._stop = threading.Event()
        self._conns = []
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._sock.getsockname()[1]}/v1"

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                continue
            self._conns.append(conn)
            conn.settimeout(2)
            try:
                data = b""
                while b"\r\n\r\n" not in data:
                    data += conn.recv(65536)
                self.requests += 1
                if self.first_chunk:
                    conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                                 b"Transfer-Encoding: chunked\r\n\r\n")
                    body = (b'data: {"id":"x","object":"chat.completion.chunk","created":0,'
                            b'"model":"m","choices":[{"index":0,"delta":{"role":"assistant",'
                            b'"content":"Th"},"finish_reason":null}]}\n\n')
                    conn.sendall(f"{len(body):x}\r\n".encode() + body + b"\r\n")
            except OSError:
                pass                                   # then say nothing more, ever

    def close(self):
        self._stop.set()
        for conn in self._conns:
            try:
                conn.close()
            except OSError:
                pass
        self._sock.close()
        self._thread.join(timeout=2)


@pytest.fixture
def stalled(monkeypatch):
    servers = []

    def make(**kw):
        server = _StalledServer(**kw)
        servers.append(server)
        monkeypatch.setenv("OPENAI_BASE_URL", server.base_url)
        monkeypatch.setenv("AGENT_LLM_TIMEOUT_SECONDS", "0.5")
        return server

    yield make
    for server in servers:
        server.close()


def _llm():
    return ef.build_llm(provider="openai", model="gpt-4o-2024-11-20")


def test_a_call_that_never_answers_ends_at_the_timeout(stalled):
    server = stalled()
    started = time.monotonic()
    with pytest.raises(LLMCallTimeout) as info:
        _llm().invoke("hi")
    elapsed = time.monotonic() - started
    # Two attempts of 0.5 s plus the client's backoff (<= 1 s at the first retry).
    assert elapsed < 6, f"took {elapsed:.1f} s; the timeout did not bound the call"
    assert server.requests == 2, "max_retries=1 should mean exactly one retry"
    message = str(info.value)
    assert "timed out after 0.5 s" in message
    assert "gpt-4o-2024-11-20" in message and "127.0.0.1" in message
    assert "1 automatic retry" in message
    assert info.value.seconds == 0.5


def test_it_is_still_the_clients_timeout_error(stalled):
    """Anything that already treats openai.APITimeoutError as retryable keeps doing so."""
    import openai

    stalled()
    with pytest.raises(openai.APITimeoutError):
        _llm().invoke("hi")


def test_no_retry_means_one_wait(stalled, monkeypatch):
    server = stalled()
    monkeypatch.setenv("AGENT_LLM_MAX_RETRIES", "0")
    started = time.monotonic()
    with pytest.raises(LLMCallTimeout) as info:
        _llm().invoke("hi")
    assert time.monotonic() - started < 3
    assert server.requests == 1
    assert "retr" not in str(info.value).split(". ")[0]


def test_a_stream_that_stalls_mid_reply_ends_too(stalled, monkeypatch):
    """A streamed reply bounds the silence between chunks: one chunk, then nothing."""
    stalled(first_chunk=True)
    monkeypatch.setenv("AGENT_LLM_MAX_RETRIES", "0")
    chunks = []
    with pytest.raises(LLMCallTimeout):
        for chunk in _llm().stream("hi"):
            chunks.append(chunk)
    assert chunks and chunks[0].content == "Th"


def test_a_timeout_with_no_request_attached_is_still_restated():
    """httpx raises RuntimeError, not AttributeError, from .request when none was set."""
    import httpx

    exc = llm_timeout.from_client_timeout(httpx.ReadTimeout("timed out"), _llm())
    assert isinstance(exc, LLMCallTimeout) and exc.request is None
    assert "timed out after 240 s" in str(exc) and "retr" not in str(exc).split(". ")[0]


def test_an_async_call_ends_at_the_timeout(stalled, monkeypatch):
    stalled()
    monkeypatch.setenv("AGENT_LLM_MAX_RETRIES", "0")
    with pytest.raises(LLMCallTimeout):
        asyncio.run(_llm().ainvoke("hi"))


def test_a_bound_tool_model_ends_at_the_timeout_too(stalled, monkeypatch):
    """Peers call the model through bind_tools(); the binding must not route around the
    subclass that restates the error."""
    from langchain_core.tools import tool

    @tool
    def ping(x: str) -> str:
        """Echo."""
        return x

    stalled()
    monkeypatch.setenv("AGENT_LLM_MAX_RETRIES", "0")
    with pytest.raises(LLMCallTimeout):
        _llm().bind_tools([ping]).invoke("hi")


# ------------------------------------------------------------------ the turn does not hang

def test_the_supervisor_stops_rerouting_after_a_model_timeout(monkeypatch):
    """2026-10-09: after an 18-minute stall the analyze peer failed and the supervisor routed
    straight back to analyze. A timed-out model is as fatal as an unreachable one: every peer
    calls the same model, so the turn answers with what it has."""
    from agent_runtime.supervisor import graph as G

    monkeypatch.setenv("AGENT_PEER_ERROR_BUDGET", "8")
    calls = {"n": 0}
    message = llm_timeout.describe(240, "deepseek-v4-flash",
                                   "https://lumen.ncsa.illinois.edu/v1", 1)

    def stalled_search(*_a, **_k):
        calls["n"] += 1
        raise LLMCallTimeout(message, seconds=240, model="deepseek-v4-flash")

    out = G.run_supervisor("What Chicago crime data does the platform have?",
                           search_fn=stalled_search, analyze_fn=stalled_search,
                           code_fn=stalled_search, decide_fn=lambda *_a, **_k: "search")
    assert calls["n"] == 1, f"the stalled model was asked {calls['n']} times"
    failures = out.get("peer_failures") or []
    assert any(f.get("fatal") for f in failures)
    assert any("timed out after 240 s" in f["error"] for f in failures)
    assert (out.get("final_answer") or "").strip(), "the turn must still answer"


def test_the_failure_text_is_what_the_supervisor_records():
    from agent_runtime.supervisor import graph as G

    exc = LLMCallTimeout(llm_timeout.describe(280, "qwen3.8:27b",
                                              "https://anvilgpt.rcac.purdue.edu/api", 1))
    assert G._is_fatal_peer_error(exc)
    _, failure = G._run_peer("analyze", lambda: (_ for _ in ()).throw(exc), {"actions": []})
    assert failure["error"].startswith("LLMCallTimeout: Model call timed out after 280 s")
    assert "qwen3.8:27b (anvilgpt.rcac.purdue.edu)" in failure["error"]
    # An ordinary error is still not fatal.
    assert not G._is_fatal_peer_error(RuntimeError("boom"))
