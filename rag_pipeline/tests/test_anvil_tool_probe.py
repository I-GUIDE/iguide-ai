"""The picker hides AnvilGPT models that cannot make a structured tool call, and only those.

Every response body below is one AnvilGPT actually returned on 2026-10-04 (trimmed). No test
here reaches the network: the probe's POST is a parameter, and the roster GET is replaced.
"""
import json
import threading
import time

import pytest

from agent_runtime import anvil_tool_probe as atp

BASE = "https://anvil.example/api"

REFUSED_AUTO = ('{"detail":"litellm.BadRequestError: OpenAIException - \\"auto\\" tool choice '
                'requires --enable-auto-tool-choice and --tool-call-parser to be set. Received '
                'Model Group=qwen3:4b\\nAvailable Model Group Fallbacks=None"}')
REFUSED_REQUIRED = ('{"detail":"litellm.BadRequestError: OpenAIException - '
                    'tool_choice=\\"required\\" requires --tool-call-parser to be set."}')
OLLAMA_NO_TOOLS = '{"error":"registry.ollama.ai/library/codegemma:latest does not support tools"}'
# Purdue's Ollama backend was down during the sweep. LiteLLM still answers 400.
OLLAMA_DOWN = ('{"detail":"litellm.APIConnectionError: OllamaException - Cannot connect to host '
               'ollama.erik-ai.svc.cluster.local:80 ssl:<ssl.SSLContext object at 0x7f>"}')


def _ok(message):
    return json.dumps({"choices": [{"message": message, "finish_reason": "stop"}]})


CALL = _ok({"role": "assistant", "content": None, "tool_calls": [{
    "id": "c1", "type": "function",
    "function": {"name": "lookup_code", "arguments": '{"key": "alpha"}'}}]})
TEXT_CALL = _ok({"role": "assistant", "content":
                 '<tool_call>\n{"name": "lookup_code", "arguments": {"key": "alpha"}}\n</tool_call>'})


def _settle(limit=2.0):
    """Wait for background probes to record their answers (bounded)."""
    end = time.time() + limit
    while atp._INFLIGHT and time.time() < end:
        time.sleep(0.01)
    assert not atp._INFLIGHT, "a probe did not finish"


@pytest.fixture(autouse=True)
def _fresh_cache():
    atp._reset_for_tests()
    yield
    atp._reset_for_tests()


@pytest.mark.parametrize("status,text,kind", [
    (200, CALL, "call"),
    (200, TEXT_CALL, "no_call"),          # a call written as text cannot drive the agent
    (400, REFUSED_AUTO, "refused"),
    (400, REFUSED_REQUIRED, "refused"),
    (400, OLLAMA_NO_TOOLS, "refused"),
    (400, OLLAMA_DOWN, "failed"),         # an outage, not an answer about tools
    (401, '{"detail":"Not authenticated"}', "failed"),
    (404, '{"detail":"Model not found"}', "failed"),
    (429, '{"detail":"rate limited"}', "failed"),
    (502, "<html>Bad Gateway</html>", "failed"),
    (200, "not json", "failed"),
])
def test_only_an_answer_about_tool_calling_is_a_verdict(status, text, kind):
    assert atp.classify(status, text)[0] == kind


def test_a_400_is_not_read_as_a_refusal_unless_it_names_tool_calling():
    """The Ollama outage arrived as HTTP 400 for seven models. Hiding on status alone would
    have emptied most of the picker during an outage."""
    assert atp.classify(400, OLLAMA_DOWN)[0] == "failed"
    assert atp.probe(BASE, "k", "llama3.2:latest", post=lambda *a: (400, OLLAMA_DOWN))[0] is None


def test_a_refusal_hides_at_once_and_says_why():
    tools, reason = atp.probe(BASE, "k", "qwen3:4b", post=lambda *a: (400, REFUSED_AUTO))
    assert tools is False
    assert reason == ("AnvilGPT refuses tool calls for it (its server runs without a "
                      "tool-call parser)"), "read in the picker: words, not the JSON body"


def test_an_answer_without_a_call_is_asked_twice_before_it_counts():
    replies = iter([(200, TEXT_CALL), (200, CALL)])
    assert atp.probe(BASE, "k", "m", post=lambda *a: next(replies))[0] is True
    replies = iter([(200, TEXT_CALL), (200, TEXT_CALL)])
    tools, reason = atp.probe(BASE, "k", "m", post=lambda *a: next(replies))
    assert tools is False and reason.endswith("twice")


def test_a_timeout_is_not_an_answer():
    def boom(*a):
        raise TimeoutError("read timed out")
    assert atp.probe(BASE, "k", "m", post=boom) == (None, "probe failed: TimeoutError")


def _fake_post(table, calls=None, delay=0.0):
    def post(base, key, model):
        if calls is not None:
            calls.append(model)
        if delay:
            time.sleep(delay)
        return table[model]
    return post


def test_verdicts_are_cached_and_reused():
    calls = []
    post = _fake_post({"qwen3:4b": (400, REFUSED_AUTO), "qwen3.8:27b": (200, CALL)}, calls)
    first = atp.tool_support(["qwen3:4b", "qwen3.8:27b"], base_url=BASE, api_key="k",
                             wait=2.0, post=post)
    assert first["qwen3:4b"].tools is False and first["qwen3.8:27b"].tools is True
    again = atp.tool_support(["qwen3:4b", "qwen3.8:27b"], base_url=BASE, api_key="k",
                             wait=2.0, post=post)
    assert again == first
    assert sorted(calls) == ["qwen3.8:27b", "qwen3:4b"], "one probe per model, then the cache"


def test_a_failed_reprobe_keeps_the_last_answer(monkeypatch):
    post = _fake_post({"qwen3:4b": (400, REFUSED_AUTO)})
    atp.tool_support(["qwen3:4b"], base_url=BASE, api_key="k", wait=2.0, post=post)
    # The answer expires, and the next probe meets an outage.
    atp._ENTRIES[(BASE, "qwen3:4b")].next_probe_at = 0.0
    down = _fake_post({"qwen3:4b": (400, OLLAMA_DOWN)})
    atp.tool_support(["qwen3:4b"], base_url=BASE, api_key="k", wait=2.0, post=down)
    _settle()
    after = atp.tool_support(["qwen3:4b"], base_url=BASE, api_key="k", wait=0.0, post=down)
    assert after["qwen3:4b"].tools is False, "an outage must not resurrect a refused model"
    entry = atp._ENTRIES[(BASE, "qwen3:4b")]
    assert entry.next_probe_at - time.time() == pytest.approx(atp.RETRY_AFTER_S, abs=5)


def test_a_slow_probe_does_not_hold_the_picker():
    """A dead backend takes up to the probe timeout. The catalogue waits only `wait`, answers
    "unknown" for the rest, and a later call sees the verdict."""
    post = _fake_post({"slow": (400, REFUSED_AUTO)}, delay=0.4)
    t0 = time.time()
    first = atp.tool_support(["slow"], base_url=BASE, api_key="k", wait=0.05, post=post)
    assert time.time() - t0 < 0.3
    assert first["slow"].tools is None
    time.sleep(0.6)
    assert atp.tool_support(["slow"], base_url=BASE, api_key="k", wait=0.0,
                            post=post)["slow"].tools is False


def test_a_fast_refusal_is_not_queued_behind_hung_probes():
    """Measured on the live roster during an Ollama outage: with 8 slots, the hung probes held
    them all and qwen3:4b's 0.2 s refusal waited behind them, so the first catalogue hid
    nothing. Every model of a roster this size is probed at once."""
    table = {f"ollama-{i}": (400, OLLAMA_DOWN) for i in range(12)}
    table["qwen3:4b"] = (400, REFUSED_AUTO)

    def post(base, key, model):
        if model != "qwen3:4b":
            time.sleep(1.0)            # the outage: an answer only after a long wait
        return table[model]

    first = atp.tool_support(list(table), base_url=BASE, api_key="k", wait=0.5, post=post)
    assert first["qwen3:4b"].tools is False


def test_a_reprobe_runs_in_the_background_while_the_last_answer_stands():
    """During an outage the failed probes retry every RETRY_AFTER_S. Had each retry been
    waited for, one page load in every ten minutes would take the full wait."""
    post = _fake_post({"qwen3:4b": (400, REFUSED_AUTO)})
    atp.tool_support(["qwen3:4b"], base_url=BASE, api_key="k", wait=2.0, post=post)
    atp._ENTRIES[(BASE, "qwen3:4b")].next_probe_at = 0.0          # the verdict has expired
    slow = _fake_post({"qwen3:4b": (400, REFUSED_AUTO)}, delay=0.5)
    t0 = time.time()
    again = atp.tool_support(["qwen3:4b"], base_url=BASE, api_key="k", wait=2.0, post=slow)
    assert time.time() - t0 < 0.3
    assert again["qwen3:4b"].tools is False, "the last answer stands while it is re-asked"


def test_a_probe_in_flight_is_not_started_twice():
    calls = []
    gate = threading.Event()

    def post(base, key, model):
        calls.append(model)
        gate.wait(2)
        return (200, CALL)

    atp.tool_support(["m"], base_url=BASE, api_key="k", wait=0.0, post=post)
    atp.tool_support(["m"], base_url=BASE, api_key="k", wait=0.0, post=post)
    gate.set()
    time.sleep(0.2)
    assert calls == ["m"]


def test_schedule_false_reports_only_what_is_known():
    calls = []
    out = atp.tool_support(["m"], base_url=BASE, api_key="k", wait=1.0, schedule=False,
                           post=_fake_post({"m": (200, CALL)}, calls))
    assert out["m"].tools is None and calls == []


# --- the catalogue -----------------------------------------------------------------------

class _Roster:
    status_code = 200

    def __init__(self, ids):
        self._ids = ids

    def raise_for_status(self):
        return None

    def json(self):
        return {"data": [{"id": i} for i in self._ids]}


def _catalogue(monkeypatch, *, roster=None, table=None):
    import requests

    from agent_runtime import executor_factory as ef

    monkeypatch.setenv("ANVILGPT_KEY", "test-key")
    monkeypatch.delenv("ANVILGPT_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)

    def get(url, **kwargs):
        if roster is None:
            raise requests.ConnectionError("tests make no network calls")
        return _Roster(roster)

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(atp, "_post", _fake_post(table or {}))
    cat = ef.list_available_models(timeout=0.01, probe_wait=2.0)
    return next(p for p in cat["providers"] if p["provider"] == "anvilgpt")


def test_the_picker_leaves_out_what_cannot_call_and_says_why(monkeypatch):
    anvil = _catalogue(monkeypatch,
                       roster=["qwen3:4b", "qwen3.8:27b", "llama3.2:latest", "qwen2.5:7b"],
                       table={"qwen3:4b": (400, REFUSED_AUTO), "qwen2.5:7b": (400, REFUSED_AUTO),
                              "qwen3.8:27b": (200, CALL), "llama3.2:latest": (400, OLLAMA_DOWN)})
    assert anvil["models"] == ["llama3.2:latest", "qwen3.8:27b"], \
        "the unreachable model stays: an outage is not an answer about tools"
    assert sorted(anvil["hidden"]) == ["qwen2.5:7b", "qwen3:4b"]
    assert "tool-call parser" in anvil["hidden"]["qwen3:4b"]


def test_nothing_is_hidden_until_something_has_answered(monkeypatch):
    anvil = _catalogue(monkeypatch, roster=["qwen3.8:27b"], table={"qwen3.8:27b": (200, CALL)})
    assert anvil["models"] == ["qwen3.8:27b"] and "hidden" not in anvil


def test_a_failed_roster_fetch_starts_no_probes_but_applies_known_answers(monkeypatch):
    from agent_runtime.executor_factory import normalize_openai_base_url

    base = normalize_openai_base_url("https://anvilgpt.rcac.purdue.edu/api/chat/completions")
    atp._ENTRIES[(base, "qwen3-coder:30b")] = atp._Entry(
        tools=False, reason="the server refuses tool calls (test)", next_probe_at=time.time() + 60)
    anvil = _catalogue(monkeypatch, roster=None, table={})
    assert anvil["stale"] is True
    assert anvil["models"] == ["qwen3.8:27b"]
    assert "qwen3-coder:30b" in anvil["hidden"]
    assert not atp._INFLIGHT, "the host is down; probing it would only queue timeouts"


def test_the_fallback_offers_only_models_measured_calling_tools():
    from agent_runtime.executor_factory import _ANVIL_FALLBACK_MODELS

    for model in ("qwen3:4b", "qwen2.5:7b", "qwen3:32b", "qwen3-vl:32b"):
        assert model not in _ANVIL_FALLBACK_MODELS
