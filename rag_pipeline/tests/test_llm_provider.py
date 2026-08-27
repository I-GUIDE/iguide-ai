"""Which LLM a run actually uses — provider selection must be explicit and inspectable.

A silent fallback to the default provider looks exactly like success: the request works, the
answer arrives, and nothing says it came from a different model than intended. So selection is
opt-in via AGENT_LLM_PROVIDER, and `active_llm_description` reports the resolved choice.
"""

from __future__ import annotations

import pytest

from agent_runtime.executor_factory import (
    _anvilgpt_settings,
    active_llm_description,
    normalize_openai_base_url,
)

ANVIL_CHAT_URL = "https://anvilgpt.rcac.purdue.edu/api/chat/completions"


@pytest.fixture()
def clean_env(monkeypatch):
    for var in ("AGENT_LLM_PROVIDER", "ANVILGPT_KEY", "ANVILGPT_URL", "ANVILGPT_MODEL",
                "VLLM_MODEL", "VLLM_PROXY", "VLLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-2024-11-20")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_KEY", "sk-test")


def test_anvilgpt_is_off_unless_explicitly_selected(clean_env, monkeypatch):
    """Configuring the variables must not be enough to move every request onto another model."""
    monkeypatch.setenv("ANVILGPT_KEY", "k")
    monkeypatch.setenv("ANVILGPT_MODEL", "qwen3.6:27b")
    assert _anvilgpt_settings() is None
    assert active_llm_description()["provider"] == "openai"


def test_selecting_anvilgpt_resolves_model_and_base_url(clean_env, monkeypatch):
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "anvilgpt")
    monkeypatch.setenv("ANVILGPT_KEY", "k")
    monkeypatch.setenv("ANVILGPT_URL", ANVIL_CHAT_URL)
    monkeypatch.setenv("ANVILGPT_MODEL", "qwen3.6:27b")

    cfg = _anvilgpt_settings()
    assert cfg["model"] == "qwen3.6:27b"
    # Open WebUI serves chat at /api/chat/completions, so the OpenAI-compatible base is /api
    assert cfg["base_url"] == "https://anvilgpt.rcac.purdue.edu/api"
    assert "max_tokens" not in cfg, "no invented ceiling — a cap truncates reasoning"

    desc = active_llm_description()
    assert desc["provider"] == "anvilgpt" and desc["model"] == "qwen3.6:27b"


def test_no_token_ceiling_is_imposed(clean_env, monkeypatch):
    """A cap would truncate reasoning: at max_tokens=20 the model returned content=None,
    having spent the whole budget on reasoning_content. Unset, it completes normally."""
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "anvilgpt")
    monkeypatch.setenv("ANVILGPT_KEY", "k")
    assert "max_tokens" not in _anvilgpt_settings()


def test_a_missing_key_fails_with_the_fix_not_a_traceback(clean_env, monkeypatch):
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "anvilgpt")
    with pytest.raises(RuntimeError) as err:
        _anvilgpt_settings()
    assert "anvilgpt.rcac.purdue.edu" in str(err.value), "say where to get a key"


def test_the_model_defaults_to_the_verified_id(clean_env, monkeypatch):
    """qwen3.6:27b is the id AnvilGPT actually serves — NOT the HuggingFace Qwen/... form."""
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "anvilgpt")
    monkeypatch.setenv("ANVILGPT_KEY", "k")
    assert _anvilgpt_settings()["model"] == "qwen3.6:27b"


def test_open_webui_url_normalisation():
    assert normalize_openai_base_url(ANVIL_CHAT_URL) == "https://anvilgpt.rcac.purdue.edu/api"
