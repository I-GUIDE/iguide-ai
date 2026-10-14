"""Text2Cypher asks the process-default model, with parameters that model accepts (Stage 58).

On the VM every Text2Cypher call was an OpenAI 400 from 2026-10-13: the search agent POSTed to
OpenAI with its own model choice (OPENAI_CHAT_MODEL=gpt-5.6-luna) and ``max_tokens``, which that
model refuses, while the agent itself ran on Lumen. The graph search fell back to keyword matching
without saying so. These tests intercept the OpenAI SDK, so nothing leaves the machine.
"""
from __future__ import annotations

import pytest

from rag_pipeline.search import agents

CYPHER_JSON = (
    '{"cypher": "MATCH (r) WHERE toLower(coalesce(r.title,\'\')) CONTAINS toLower($q) '
    'RETURN r AS node, 1.0 AS score LIMIT $limit", "params": {"q": "flood", "limit": 5}}'
)


@pytest.fixture
def sent(monkeypatch):
    """Capture every chat-completions request the OpenAI SDK would send; answer with JSON.

    Also fails the raw ``requests.post`` route the search agent used to take, so a regression
    reports where it was going instead of reaching the network.
    """
    from openai.resources.chat.completions import Completions
    from openai.types.chat import ChatCompletion
    import requests

    calls = []

    class RawResponse:
        """What ``with_raw_response.create`` hands LangChain: headers plus ``parse()``."""

        headers: dict = {}

        def __init__(self, completion):
            self._completion = completion

        def parse(self):
            return self._completion

    def fake_create(self, **kwargs):
        kwargs.pop("extra_headers", None)  # the SDK's raw-response marker
        calls.append({"base_url": str(self._client.base_url), **kwargs})
        completion = ChatCompletion.model_validate({
            "id": "c1", "object": "chat.completion", "created": 0, "model": kwargs["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": CYPHER_JSON}}],
        })
        return RawResponse(completion)

    def raw_post(url, *args, **kwargs):
        body = kwargs.get("json") or {}
        raise AssertionError(
            f"search agent bypassed the default model: POST {url} model={body.get('model')!r} "
            f"params={sorted(k for k in body if k not in ('messages',))}")

    monkeypatch.setattr(Completions, "create", fake_create)
    monkeypatch.setattr(requests, "post", raw_post)
    for name in ("LLM_PROVIDER", "VLLM_PROXY", "VLLM_MODEL", "VLLM_API_KEY", "ANVILGPT_URL",
                 "ANVILGPT_MODEL", "ANVILGPT_KEY", "OPENAI_BASE_URL", "LUMEN_URL"):
        monkeypatch.delenv(name, raising=False)
    # The VM's shape: the agent on Lumen, OpenAI still configured with a gpt-5.6 model.
    monkeypatch.setenv("OPENAI_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-5.6-luna")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.6-luna")
    return calls


def test_text2cypher_follows_the_deployment_provider(sent, monkeypatch):
    """AGENT_LLM_PROVIDER=lumen: Text2Cypher goes to Lumen's default model, not to OpenAI."""
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "lumen")
    monkeypatch.setenv("LUMEN_API_KEY", "lumen-test")
    monkeypatch.delenv("LUMEN_MODEL", raising=False)

    cypher, params = agents._text2cypher("flood notebooks", "(schema)", 5)

    assert "MATCH (r)" in cypher and params == {"q": "flood", "limit": 5}
    (call,) = sent
    assert call["base_url"].startswith("https://lumen.ncsa.illinois.edu/v1")
    assert call["model"] == "deepseek-v4-flash"
    # A ceiling cuts a reasoning model's thinking into an empty answer.
    assert "max_tokens" not in call and "max_completion_tokens" not in call
    assert call["timeout"] == agents._LLM_TIMEOUT_SEC


def test_an_openai_gpt56_default_gets_parameters_it_accepts(sent, monkeypatch):
    """With OpenAI as the default, gpt-5.6-luna must not see ``max_tokens``, and temperature 0
    only rides with ``reasoning_effort='none'``. Both were measured against the API on
    2026-10-13: ``max_tokens`` → 400 unsupported_parameter; temperature 0 alone → 400
    unsupported_value; temperature 0 + effort 'none' → 200."""
    monkeypatch.delenv("AGENT_LLM_PROVIDER", raising=False)

    agents._agent_generate_opensearch_body("floods near Houston", "(fields)", 5)

    (call,) = sent
    assert call["base_url"].startswith("https://api.openai.com/v1")
    assert call["model"] == "gpt-5.6-luna"
    assert "max_tokens" not in call
    if "temperature" in call:
        assert call.get("reasoning_effort") == "none"


def test_graph_search_uses_text2cypher_hits_instead_of_falling_back(sent, monkeypatch):
    """The dispatcher returns the Text2Cypher rows; the keyword arm is not reached."""
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "lumen")
    monkeypatch.setenv("LUMEN_API_KEY", "lumen-test")
    monkeypatch.setenv("USE_TEXT2CYPHER", "true")
    monkeypatch.setattr(agents, "detect_pattern", lambda q: None)
    monkeypatch.setattr(agents, "get_comprehensive_schema", lambda: "(schema)")
    ran = []

    def fake_run(cypher, params):
        ran.append((cypher, params))
        return [{"node": {"id": "abc", "title": "Flood notebook"}, "score": 1.0}]

    monkeypatch.setattr(agents, "_neo4j_run", fake_run)

    import rag_pipeline.search.neo4j as n4

    def keyword_arm(*a, **k):
        raise AssertionError("fell back to the keyword arm")

    monkeypatch.setattr(n4, "get_neo4j_search_results", keyword_arm)

    hits = agents.get_neo4j_agent_results("flood notebooks", limit=5)

    assert len(sent) == 1 and len(ran) == 1
    assert hits and hits[0]["_source"]["title"] == "Flood notebook"
