"""DEMO_MODE opens the deployment so a link can be handed to an audience.

It does two things that have to stay in step: it stops enforcing the API key on every agent
endpoint, and it tells the browser to hide its connection settings — because a page with no key
to enter should not show a field demanding one, and a page that hides the field must not then be
rejected for having no key.

The direction of every default here is deliberate. Off unless explicitly switched on, and any
ambiguity resolves to CLOSED: an accidentally open deployment spends this project's Earth Engine
quota and LLM budget for whoever finds the URL.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import api.server as server  # noqa: E402

KEY = "s3cret-key"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("AGENT_MODE", raising=False)
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.delenv("AGENT_CHAT_API_KEY", raising=False)


def _check(headers: dict | None = None):
    """Run the guard the endpoints run, inside a request carrying these headers."""
    with server.app.test_request_context("/agent/chat", headers=headers or {}):
        server._require_agent_chat_api_key()


def _ui_config() -> dict:
    with server.app.test_client() as client:
        res = client.get("/agent/ui-config")
        assert res.status_code == 200
        return res.get_json()


# --- the switch itself ----------------------------------------------------------
@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " true "])
def test_the_spellings_that_turn_it_on(monkeypatch, value):
    monkeypatch.setenv("DEMO_MODE", value)
    assert server._demo_mode() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe", "demo"])
def test_everything_else_leaves_it_off(monkeypatch, value):
    """Anything unrecognised reads as OFF. A typo must not open the deployment."""
    monkeypatch.setenv("DEMO_MODE", value)
    assert server._demo_mode() is False


def test_off_when_the_variable_is_absent():
    assert server._demo_mode() is False


# --- what it does to the key ----------------------------------------------------
def test_a_configured_key_is_enforced_when_demo_mode_is_off(monkeypatch):
    monkeypatch.setenv("AGENT_CHAT_API_KEY", KEY)
    with pytest.raises(PermissionError):
        _check()
    with pytest.raises(PermissionError):
        _check({"X-API-KEY": "wrong"})
    _check({"X-API-KEY": KEY})           # the real key still works
    _check({"Authorization": f"Bearer {KEY}"})


def test_demo_mode_lets_a_request_through_with_no_key(monkeypatch):
    """The key stays CONFIGURED. A demo should not require unsetting the secret and

    remembering to put it back — that is how a deployment ends up permanently open.
    """
    monkeypatch.setenv("AGENT_CHAT_API_KEY", KEY)
    monkeypatch.setenv("DEMO_MODE", "true")
    _check()
    _check({"X-API-KEY": "wrong"})       # not merely optional: not checked at all


def test_no_key_configured_is_open_with_or_without_demo_mode(monkeypatch):
    """Unchanged behaviour: an unset key has always meant no auth."""
    _check()
    monkeypatch.setenv("DEMO_MODE", "true")
    _check()


# --- what it tells the browser --------------------------------------------------
def test_ui_config_says_a_key_is_required_when_one_is_enforced(monkeypatch):
    monkeypatch.setenv("AGENT_CHAT_API_KEY", KEY)
    assert _ui_config() == {"mode": "dev", "demo_mode": False, "api_key_required": True}


def test_ui_config_stops_asking_for_a_key_in_demo_mode(monkeypatch):
    monkeypatch.setenv("AGENT_CHAT_API_KEY", KEY)
    monkeypatch.setenv("DEMO_MODE", "true")
    assert _ui_config() == {"mode": "demo", "demo_mode": True, "api_key_required": False}


def test_ui_config_needs_no_key_of_its_own(monkeypatch):
    """Unauthenticated on purpose: a client that cannot authenticate is exactly the one that

    needs to ask whether it has to.
    """
    monkeypatch.setenv("AGENT_CHAT_API_KEY", KEY)
    with server.app.test_client() as client:
        assert client.get("/agent/ui-config").status_code == 200


def test_ui_config_never_returns_the_key_itself(monkeypatch):
    monkeypatch.setenv("AGENT_CHAT_API_KEY", KEY)
    monkeypatch.setenv("DEMO_MODE", "true")
    assert KEY not in str(_ui_config())


# --- files belong to the conversation that made them ---------------------------
@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "fs"))
    from agent_runtime import file_store

    return file_store


def test_a_file_records_the_conversation_that_wrote_it(store, tmp_path):
    """The record had seven fields and none of them said who made it, so every session saw

    every file: 48 packages sharing one filename came from many different conversations, and
    list_embedding_packages could not be scoped to the caller.
    """
    token = store.set_session("sess-alpha")
    try:
        rec = store.create_output_file("a.txt", "x")
    finally:
        store.reset_session(token)
    assert rec["session"] == "sess-alpha"


def test_a_lookup_sees_its_own_conversation(store):
    a = store.set_session("sess-alpha")
    store.create_output_file("mine.txt", "x")
    store.reset_session(a)

    b = store.set_session("sess-beta")
    store.create_output_file("theirs.txt", "x")
    try:
        names = {f["filename"] for f in store.find_files()}
        assert "theirs.txt" in names
        assert "mine.txt" not in names, "another conversation's file must not be visible"
    finally:
        store.reset_session(b)


def test_files_from_before_sessions_existed_stay_visible(store):
    """1,325 records predate this field. Hiding them all would break every reuse the demo

    depends on, so an unstamped record belongs to everyone.
    """
    rec = store.create_output_file("legacy.txt", "x")     # no session bound
    assert rec.get("session") is None
    token = store.set_session("sess-alpha")
    try:
        assert "legacy.txt" in {f["filename"] for f in store.find_files()}
    finally:
        store.reset_session(token)


def test_searching_every_conversation_is_possible_but_explicit(store):
    a = store.set_session("sess-alpha")
    store.create_output_file("mine.txt", "x")
    store.reset_session(a)
    b = store.set_session("sess-beta")
    try:
        assert "mine.txt" not in {f["filename"] for f in store.find_files()}
        assert "mine.txt" in {f["filename"] for f in store.find_files(session=None)}
    finally:
        store.reset_session(b)


def test_no_session_bound_sees_everything(store):
    """A CLI run or a test has no conversation, and scoping it to nothing would show nothing."""
    a = store.set_session("sess-alpha")
    store.create_output_file("mine.txt", "x")
    store.reset_session(a)
    assert "mine.txt" in {f["filename"] for f in store.find_files()}


def test_the_session_survives_the_worker_thread():
    """The agent runs in a worker thread (graph_runtime), and ContextVars do NOT cross threads.

    Bound at the request edge and read in tool code, the session came out None on the real path
    while every unit test passed — the tests all ran on one thread. copy_context() on the
    caller's side is what carries it over.
    """
    import contextvars
    import threading

    from agent_runtime import file_store

    seen = []
    token = file_store.set_session("sess-worker")
    try:
        def work():
            seen.append(file_store.current_session())

        # what the code does now: snapshot here, run there
        ctx = contextvars.copy_context()
        t = threading.Thread(target=lambda: ctx.run(work))
        t.start(); t.join()

        # and what it did before, for contrast
        plain = []
        t2 = threading.Thread(target=lambda: plain.append(file_store.current_session()))
        t2.start(); t2.join()
    finally:
        file_store.reset_session(token)

    assert seen == ["sess-worker"], "the worker must see the request's session"
    assert plain == [None], "and without copy_context it does not — this is the bug"


# --- the model a demo answers with --------------------------------------------------------
#
# Demo mode hides the settings panel, which is the only control over the model. So the choice
# has to be made server-side, and it has to WIN: a value left in a returning visitor's
# localStorage would otherwise pin them to a model they can neither see nor change.

def _extract(monkeypatch, body, demo):
    monkeypatch.setenv("DEMO_MODE", "true" if demo else "false")
    import importlib

    import api.server as srv
    importlib.reload(srv)
    with srv.app.test_request_context(json={"userQuery": "hi", **body}):
        return srv._normalize_agent_chat_request(srv.request.get_json())


def test_demo_mode_answers_with_luna(monkeypatch):
    got = _extract(monkeypatch, {}, demo=True)
    assert got["llm_model"] == "gpt-5.6-luna"
    assert got["llm_provider"] == "openai"


def test_demo_mode_overrides_a_model_the_client_still_sends(monkeypatch):
    """The client cannot show the picker in demo mode, so a stored value is stale by
    definition — honouring it hands a visitor a model with no way to change it."""
    got = _extract(monkeypatch, {"model": "gpt-4o-2024-11-20", "provider": "openai"}, demo=True)
    assert got["llm_model"] == "gpt-5.6-luna"


def test_outside_demo_mode_the_request_still_chooses(monkeypatch):
    got = _extract(monkeypatch, {"model": "claude-sonnet-5", "provider": "anthropic"}, demo=False)
    assert got["llm_model"] == "claude-sonnet-5"
    assert got["llm_provider"] == "anthropic"


def test_outside_demo_mode_silence_still_means_the_configured_default(monkeypatch):
    got = _extract(monkeypatch, {}, demo=False)
    assert got["llm_model"] is None, "None lets executor_factory resolve OPENAI_CHAT_MODEL"
    assert got["llm_provider"] is None


def test_the_demo_model_is_configurable(monkeypatch):
    monkeypatch.setenv("DEMO_MODEL", "gpt-5.6-sol")
    monkeypatch.setenv("DEMO_MODEL_PROVIDER", "openai")
    got = _extract(monkeypatch, {}, demo=True)
    assert got["llm_model"] == "gpt-5.6-sol"


def test_it_does_not_fall_through_to_the_deployment_default(monkeypatch):
    """A demo is handed to an audience; "whatever OPENAI_CHAT_MODEL happens to be" is not a
    demo decision, so the fallback is a named constant rather than that env var."""
    monkeypatch.delenv("DEMO_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-4o-2024-11-20")
    got = _extract(monkeypatch, {}, demo=True)
    assert got["llm_model"] == "gpt-5.6-luna"


# --- token mode hides the same panel, and must not honour a stale model -------------

def _extract_mode(monkeypatch, body, mode):
    """Same as _extract, but selecting AGENT_MODE rather than the legacy DEMO_MODE flag."""
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.setenv("AGENT_MODE", mode)
    import importlib

    import api.server as srv
    importlib.reload(srv)
    with srv.app.test_request_context(json={"userQuery": "hi", **body}):
        return srv._normalize_agent_chat_request(srv.request.get_json())


def test_token_mode_drops_a_model_the_client_still_sends(monkeypatch):
    """The trap demo mode already guards against, reached through the other hidden panel.

    The only control that can change the model is the settings panel, and token mode hides it
    too — the credential is a cookie, so there is nothing to paste. A value left in a returning
    visitor's localStorage then pins them to a model they can neither see nor change. Observed
    live: token mode was switched on while browsers still held `gpt-oss:120b`, the deployment
    default moved to gpt-5.6-luna, and every request from the map UI kept asking for the old one.
    """
    got = _extract_mode(monkeypatch, {"model": "gpt-oss:120b", "provider": "anvilgpt"}, "token")
    assert got["llm_model"] is None, "the stale client model must be dropped"
    assert got["llm_provider"] is None


def test_token_mode_defers_to_the_deployment_rather_than_naming_a_model(monkeypatch):
    """Dropped, NOT forced to a fixed id — unlike demo.

    Demo names its own model because "whatever this deployment happens to be set to" is not a
    demo decision. Token mode passes None so the agent uses its configured default, which is
    what makes changing the deployment's model actually change what signed-in users get.
    """
    monkeypatch.setenv("DEMO_MODEL", "gpt-5.6-sol")
    got = _extract_mode(monkeypatch, {}, "token")
    assert got["llm_model"] is None, "token mode must not inherit the DEMO_MODEL choice"


def test_dev_mode_still_honours_the_clients_choice(monkeypatch):
    """Dev shows the settings panel, so a model named by the request is a live choice."""
    got = _extract_mode(monkeypatch, {"model": "claude-sonnet-5", "provider": "anthropic"}, "dev")
    assert got["llm_model"] == "claude-sonnet-5"
    assert got["llm_provider"] == "anthropic"
