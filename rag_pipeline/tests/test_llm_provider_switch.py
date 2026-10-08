"""LLM_PROVIDER dispatch, and the claude-cli backend's failure handling.

subprocess is stubbed throughout, so these run with no credentials and no network — the
point is the dispatch and the error translation, not Anthropic's uptime.

The behaviour worth pinning: the CLI exits NON-ZERO while still emitting the JSON that
explains why. Judging the exit code before parsing that JSON turns "not logged in" into an
unreadable dump of usage counters, which is what the first version of this module did.
"""

from __future__ import annotations

import json

import pytest

from rag_pipeline import llm_claude_cli as cc


class _Proc:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


def _result_json(result, *, is_error=False, subtype="success"):
    return json.dumps({"type": "result", "subtype": subtype, "is_error": is_error,
                       "result": result, "usage": {"input_tokens": 1}})


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("LLM_PROVIDER", "CLAUDE_CLI_MODEL", "CLAUDE_CLI_BARE",
                "ANTHROPIC_API_KEY", "CLAUDE_CLI_MAX_BUDGET_USD",
                "AGENT_DEPLOYED", "KUBERNETES_SERVICE_HOST"):
        monkeypatch.delenv(var, raising=False)
    cc._last_model = None


def _stub(monkeypatch, proc, capture=None):
    def fake_run(argv, **kwargs):
        if capture is not None:
            capture.extend(argv)
        return proc
    monkeypatch.setattr(cc.subprocess, "run", fake_run)
    monkeypatch.setattr(cc.shutil, "which", lambda name: "/usr/bin/claude")


# --------------------------------------------------------------------------- dispatch

def test_provider_not_selected_by_default():
    assert cc.is_selected() is False


@pytest.mark.parametrize("value", ["claude-cli", "claude_cli", "claude", "CLAUDE-CLI"])
def test_provider_selection_accepts_aliases(monkeypatch, value):
    monkeypatch.setenv("LLM_PROVIDER", value)
    assert cc.is_selected() is True


def test_call_llm_routes_to_claude_cli(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "claude-cli")
    _stub(monkeypatch, _Proc(stdout=_result_json("routed")))
    from rag_pipeline.llm_utils import call_llm
    assert call_llm("hi") == "routed"


def test_call_llm_does_not_route_when_unset(monkeypatch):
    """The openai/vllm path must be untouched when the provider is not selected."""
    def explode(*a, **k):
        raise AssertionError("claude-cli was called despite LLM_PROVIDER being unset")
    monkeypatch.setattr(cc, "call", explode)
    monkeypatch.setenv("OPENAI_KEY", "sk-test")
    import requests
    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("network blocked")))
    from rag_pipeline.llm_utils import call_llm
    with pytest.raises(Exception):
        call_llm("hi")  # fails at the network, NOT via claude-cli


def test_registered_callable_still_wins(monkeypatch):
    """register_llm_callable is the test seam and must short-circuit everything."""
    monkeypatch.setenv("LLM_PROVIDER", "claude-cli")
    from rag_pipeline import llm_utils
    monkeypatch.setattr(llm_utils, "_llm_callable", lambda p: "stubbed")
    try:
        assert llm_utils.call_llm("hi") == "stubbed"
    finally:
        monkeypatch.setattr(llm_utils, "_llm_callable", None)


# --------------------------------------------------------------------------- model + argv

def test_model_defaults_to_sonnet():
    assert cc.model() == "sonnet"


def test_model_is_overridable_and_recorded(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "claude-cli")
    monkeypatch.setenv("CLAUDE_CLI_MODEL", "haiku")
    _stub(monkeypatch, _Proc(stdout=_result_json("ok")))
    assert cc.call("hi") == "ok"
    # attributability: a number produced under one model must not be mistaken for another
    assert cc.last_model() == "haiku"


def test_bare_follows_the_available_credential(monkeypatch):
    """--bare cannot read OAuth, so it may only be the default when an API key exists."""
    assert cc.use_bare() is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert cc.use_bare() is True
    monkeypatch.setenv("CLAUDE_CLI_BARE", "0")
    assert cc.use_bare() is False


def test_argv_includes_bare_and_budget_when_configured(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "claude-cli")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("CLAUDE_CLI_MAX_BUDGET_USD", "0.50")
    argv = []
    _stub(monkeypatch, _Proc(stdout=_result_json("ok")), capture=argv)
    cc.call("hi")
    assert "--bare" in argv
    assert "--output-format" in argv and "json" in argv
    assert argv[argv.index("--max-budget-usd") + 1] == "0.50"


# --------------------------------------------------------------------------- failures

@pytest.mark.parametrize("detail", [
    "Not logged in · Please run /login",
    "Failed to authenticate. API Error: 401 OAuth access token has expired.",
])
def test_auth_failure_is_actionable_even_on_nonzero_exit(monkeypatch, detail):
    """THE REGRESSION THIS FILE EXISTS FOR: exit!=0 but the JSON holds the reason."""
    monkeypatch.setenv("LLM_PROVIDER", "claude-cli")
    _stub(monkeypatch, _Proc(stdout=_result_json(detail, is_error=True), returncode=1))
    with pytest.raises(cc.ClaudeCliUnavailable) as err:
        cc.call("hi")
    msg = str(err.value)
    assert "ANTHROPIC_API_KEY" in msg and "/login" in msg      # both remediation paths
    assert "LLM_PROVIDER=vllm" in msg                          # and the way out


def test_missing_executable_names_the_fix(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "claude-cli")
    monkeypatch.setattr(cc.shutil, "which", lambda name: None)
    with pytest.raises(cc.ClaudeCliUnavailable, match="not on PATH"):
        cc.call("hi")


def test_non_auth_error_is_reported_verbatim(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "claude-cli")
    _stub(monkeypatch, _Proc(stdout=_result_json("model overloaded", is_error=True,
                                                 subtype="error_api"), returncode=1))
    with pytest.raises(RuntimeError) as err:
        cc.call("hi")
    assert "model overloaded" in str(err.value)
    assert not isinstance(err.value, cc.ClaudeCliUnavailable)


def test_non_json_output_is_still_returned(monkeypatch):
    """An older CLI printing plain text should degrade, not fail."""
    monkeypatch.setenv("LLM_PROVIDER", "claude-cli")
    _stub(monkeypatch, _Proc(stdout="plain text answer"))
    assert cc.call("hi") == "plain text answer"


def test_timeout_names_the_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "claude-cli")
    monkeypatch.setattr(cc.shutil, "which", lambda n: "/usr/bin/claude")

    def fake_run(argv, **kwargs):
        raise cc.subprocess.TimeoutExpired(cmd="claude", timeout=1)
    monkeypatch.setattr(cc.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="timed out"):
        cc.call("hi")


# --------------------------------------------------------------------------- deployment guard

@pytest.mark.parametrize("marker", ["AGENT_DEPLOYED", "KUBERNETES_SERVICE_HOST"])
def test_refuses_to_run_in_a_deployment(monkeypatch, marker):
    """A personal subscription must never serve platform users."""
    monkeypatch.setenv("LLM_PROVIDER", "claude-cli")
    monkeypatch.setenv(marker, "1")
    with pytest.raises(RuntimeError, match="development-only"):
        cc.check_not_deployed()


def test_deployment_guard_is_inert_when_provider_unselected(monkeypatch):
    monkeypatch.setenv("AGENT_DEPLOYED", "1")
    cc.check_not_deployed()  # must not raise
