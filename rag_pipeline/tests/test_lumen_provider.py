"""NCSA Lumen is offered in dev and local mode only, and is refused everywhere else.

Every call spends the Lumen coins of whoever created LUMEN_API_KEY, so a token-mode deployment
must not hand it to every platform user who signs in, and leaving it out of the picker alone
would still let a crafted request through. No test here reaches the network.
"""
from types import SimpleNamespace

import pytest

from agent_runtime import executor_factory as ef

CATALOGUE = {"data": [
    {"id": "nemotron-3-super-120b-a12b", "input_modalities": ["text"],
     "output_modalities": ["text"], "status": "ok", "max_model_len": 262144},
    {"id": "gemma-4-31b-it", "input_modalities": ["text"], "output_modalities": ["text"],
     "status": "ok", "max_model_len": 46790},
    {"id": "granite-speech-4.1-2b-plus", "input_modalities": ["audio"],
     "output_modalities": ["text"], "status": "ok"},
    {"id": "deepseek-v4-flash", "input_modalities": ["text"], "output_modalities": ["text"],
     "status": "ok"},
    {"id": "broken-model", "input_modalities": ["text"], "output_modalities": ["text"],
     "status": "down"},
]}


@pytest.fixture(autouse=True)
def _lumen_env(monkeypatch):
    for var in ("ANVILGPT_KEY", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "LUMEN_URL",
                "LUMEN_MODEL", "DEMO_MODE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LUMEN_API_KEY", "sk_test_not_a_real_key")
    monkeypatch.setenv("OPENAI_KEY", "sk-test")


def _catalogue(monkeypatch, *, fail=False):
    import requests

    seen = []

    class Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return CATALOGUE

    def get(url, headers=None, timeout=None):
        seen.append((url, (headers or {}).get("Authorization")))
        if fail or "lumen" not in url:
            raise requests.ConnectionError("no network in tests")
        return Resp()

    monkeypatch.setattr(requests, "get", get)
    cat = ef.list_available_models(timeout=0.01)
    return next((p for p in cat["providers"] if p["provider"] == "lumen"), None), seen


@pytest.mark.parametrize("mode", ["dev", "local"])
def test_dev_and_local_offer_lumen_text_models_only(monkeypatch, mode):
    monkeypatch.setenv("AGENT_MODE", mode)
    lumen, seen = _catalogue(monkeypatch)
    assert lumen["configured"] is True and "stale" not in lumen
    assert lumen["models"] == ["deepseek-v4-flash", "gemma-4-31b-it", "nemotron-3-super-120b-a12b"], \
        "the speech model and the model whose backend is down are not offered"
    assert "coins" in lumen["caveat"]
    assert seen[0] == ("https://lumen.ncsa.illinois.edu/v1/models", "Bearer sk_test_not_a_real_key")


@pytest.mark.parametrize("mode", ["token", "demo"])
def test_other_modes_do_not_list_lumen_at_all(monkeypatch, mode):
    monkeypatch.setenv("AGENT_MODE", mode)
    lumen, seen = _catalogue(monkeypatch)
    assert lumen is None
    assert not any("lumen" in url for url, _ in seen), "not even asked"


def test_without_a_key_it_is_listed_as_needing_one(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "dev")
    monkeypatch.delenv("LUMEN_API_KEY")
    lumen, seen = _catalogue(monkeypatch)
    assert lumen["configured"] is False and lumen["needs"] == "LUMEN_API_KEY"
    assert lumen["models"] == list(ef._LUMEN_FALLBACK_MODELS)
    assert not any("lumen" in url for url, _ in seen)


def test_an_unreachable_catalogue_offers_known_ids_and_says_so(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "dev")
    lumen, _ = _catalogue(monkeypatch, fail=True)
    assert lumen["stale"] is True and lumen["models"] == list(ef._LUMEN_FALLBACK_MODELS)


@pytest.mark.parametrize("mode", ["dev", "local"])
def test_build_llm_points_at_lumen_in_dev_and_local(monkeypatch, mode):
    monkeypatch.setenv("AGENT_MODE", mode)
    llm = ef.build_llm(provider="lumen", model="gemma-4-31b-it")
    assert llm.model_name == "gemma-4-31b-it"
    assert str(llm.openai_api_base).rstrip("/") == "https://lumen.ncsa.illinois.edu/v1"
    assert llm.temperature == 0.0
    assert type(llm).__name__ == "ReasoningPreservingChatOpenAI", \
        "these are reasoning models; their thinking must survive a tool-calling step"
    assert ef.build_llm(provider="lumen").model_name == "deepseek-v4-flash"


@pytest.mark.parametrize("mode", ["token", "demo"])
def test_a_request_for_lumen_is_refused_outside_dev_and_local(monkeypatch, mode):
    monkeypatch.setenv("AGENT_MODE", mode)
    with pytest.raises(ValueError, match="dev and local mode only"):
        ef.build_llm(provider="lumen", model="deepseek-v4-flash")


def test_a_missing_key_says_where_to_get_one(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "dev")
    monkeypatch.delenv("LUMEN_API_KEY")
    with pytest.raises(ValueError, match="LUMEN_API_KEY"):
        ef.build_llm(provider="lumen")


def test_reasoning_effort_is_not_sent_to_lumen(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "dev")
    llm = ef.build_llm(provider="lumen", model="deepseek-v4-flash", reasoning_effort="high")
    assert getattr(llm, "reasoning_effort", None) is None


@pytest.mark.parametrize("model,window", [("gemma-4-31b-it", 46_790),
                                          ("deepseek-v4-flash", 511_994),
                                          ("nemotron-3-super-120b-a12b", 262_144),
                                          ("ornith-1.0-35b", 262_138)])
def test_each_lumen_model_gets_the_window_lumen_reports(monkeypatch, model, window):
    """gemma's 46,790 is BELOW the 65,536 floor an unlisted model gets: unlisted, every long
    turn on it would overrun the window."""
    monkeypatch.delenv("AGENT_MODEL_CONTEXT_WINDOW", raising=False)
    assert ef._model_context_window(SimpleNamespace(model_name=model)) == window


def test_the_gate_never_breaks_the_catalogue(monkeypatch):
    """Applied to a tree older than local mode, the gate read deployment_mode.LOCAL, which did
    not exist. The AttributeError took the whole model picker down, OpenAI included."""
    from agent_runtime import deployment_mode

    monkeypatch.setenv("AGENT_MODE", "dev")
    monkeypatch.delattr(deployment_mode, "LOCAL", raising=False)
    lumen, _ = _catalogue(monkeypatch)
    assert lumen is not None and lumen["models"]

    def boom():
        raise RuntimeError("mode table broken")

    monkeypatch.setattr(deployment_mode, "current_mode", boom)
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(
        requests.ConnectionError("offline")))
    cat = ef.list_available_models(timeout=0.01)
    assert [p["provider"] for p in cat["providers"]][:2] == ["openai", "anvilgpt"]
    assert all(p["provider"] != "lumen" for p in cat["providers"])


@pytest.mark.parametrize("mode", ["token", "dev"])
def test_the_operator_can_make_lumen_the_default_in_any_mode(monkeypatch, mode):
    """AGENT_LLM_PROVIDER=lumen is the deployment's own choice for requests that name no model,
    so it holds in token mode too. Picking Lumen per request is still dev and local only."""
    monkeypatch.setenv("AGENT_MODE", mode)
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "lumen")
    monkeypatch.setenv("LUMEN_MODEL", "deepseek-v4-flash")
    llm = ef.build_default_llm()
    assert llm.model_name == "deepseek-v4-flash"
    assert str(llm.openai_api_base).rstrip("/") == "https://lumen.ncsa.illinois.edu/v1"
    assert type(llm).__name__ == "ReasoningPreservingChatOpenAI"
    active = ef.active_llm_description()
    assert active["provider"] == "lumen" and active["model"] == "deepseek-v4-flash"
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(
        requests.ConnectionError("offline")))
    assert ef.list_available_models(timeout=0.01)["default"] == {
        "provider": "lumen", "model": "deepseek-v4-flash"}
    if mode == "token":
        with pytest.raises(ValueError, match="dev and local mode only"):
            ef.build_llm(provider="lumen", model="nemotron-3-super-120b-a12b")


def test_the_default_without_a_key_names_it(monkeypatch):
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "lumen")
    monkeypatch.delenv("LUMEN_API_KEY")
    with pytest.raises(RuntimeError, match="LUMEN_API_KEY"):
        ef.build_default_llm()


def test_setting_the_key_alone_does_not_move_the_default(monkeypatch):
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-5.6-luna")
    assert ef.active_llm_description()["provider"] == "openai"
