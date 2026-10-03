"""One named mode decides who a deployment is for; three booleans did not.

The load-bearing property is BACKWARD COMPATIBILITY: the running server is configured with the
old ``DEMO_MODE`` flag and no ``AGENT_MODE`` at all, so every assertion about that combination
is an assertion about production behaviour, not a style preference.

The other direction that matters is that an unknown mode RAISES. This switch selects security
behaviour, and a typo resolving to a working mode is exactly the failure nobody notices on a
public host.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime import deployment_mode as dm  # noqa: E402
import api.server as server  # noqa: E402


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("AGENT_MODE", raising=False)
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.delenv("AGENT_CHAT_API_KEY", raising=False)


def _ui_config() -> dict:
    with server.app.test_client() as client:
        res = client.get("/agent/ui-config")
        assert res.status_code == 200
        return res.get_json()


# --- the selector ---------------------------------------------------------------

def test_default_is_dev():
    """Nothing set: the safe, ordinary deployment. Not demo, not token."""
    assert dm.current_mode() == dm.DEV


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_legacy_demo_flag_still_selects_demo(monkeypatch, value):
    """The DEPLOYED server is configured this way. If this breaks, production changed."""
    monkeypatch.setenv("DEMO_MODE", value)
    assert dm.current_mode() == dm.DEMO
    assert dm.is_demo() is True


@pytest.mark.parametrize("mode", ["dev", "demo", "token"])
def test_explicit_mode_wins(monkeypatch, mode):
    monkeypatch.setenv("AGENT_MODE", mode)
    assert dm.current_mode() == mode


def test_explicit_mode_beats_the_legacy_flag(monkeypatch):
    """A deployment migrating to AGENT_MODE must not be dragged back by a stale DEMO_MODE."""
    monkeypatch.setenv("AGENT_MODE", "dev")
    monkeypatch.setenv("DEMO_MODE", "true")
    assert dm.current_mode() == dm.DEV
    assert dm.is_demo() is False


@pytest.mark.parametrize("bad", ["tokne", "production", "on", "1", "demo_mode"])
def test_unknown_mode_raises_rather_than_falling_back(monkeypatch, bad):
    monkeypatch.setenv("AGENT_MODE", bad)
    with pytest.raises(ValueError) as exc:
        dm.current_mode()
    assert bad in str(exc.value)          # says what was wrong
    assert "dev, demo, token" in str(exc.value)   # and what was expected


def test_modes_are_mutually_exclusive(monkeypatch):
    for mode, checks in [(dm.DEV, (True, False, False)),
                         (dm.DEMO, (False, True, False)),
                         (dm.TOKEN, (False, False, True))]:
        monkeypatch.setenv("AGENT_MODE", mode)
        assert (dm.is_dev(), dm.is_demo(), dm.is_token()) == checks


# --- the boot warning -----------------------------------------------------------

def test_demo_warns_at_boot(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "demo")
    warning = dm.boot_warning()
    assert warning and "open to anyone" in warning


@pytest.mark.parametrize("mode", ["dev", "token"])
def test_gated_modes_do_not_warn(monkeypatch, mode):
    monkeypatch.setenv("AGENT_MODE", mode)
    assert dm.boot_warning() is None


# --- what the browser is told ---------------------------------------------------

def test_ui_config_reports_the_mode(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "token")
    assert _ui_config()["mode"] == "token"


def test_ui_config_keeps_demo_mode_for_old_bundles(monkeypatch):
    """A page holding a pre-mode bundle reads `demo_mode`; dropping it blanks its settings."""
    monkeypatch.setenv("AGENT_MODE", "demo")
    cfg = _ui_config()
    assert cfg["demo_mode"] is True and cfg["mode"] == "demo"


def test_api_key_still_required_in_dev(monkeypatch):
    """The mode does NOT decide the key: dev with a key configured still enforces it."""
    monkeypatch.setenv("AGENT_MODE", "dev")
    monkeypatch.setenv("AGENT_CHAT_API_KEY", "s3cret")
    assert _ui_config()["api_key_required"] is True


def test_api_key_not_required_in_demo(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "demo")
    monkeypatch.setenv("AGENT_CHAT_API_KEY", "s3cret")
    assert _ui_config()["api_key_required"] is False


def test_token_mode_still_honours_the_service_key(monkeypatch):
    """Token mode does not drop the service credential — the eval harness has no browser."""
    monkeypatch.setenv("AGENT_MODE", "token")
    monkeypatch.setenv("AGENT_CHAT_API_KEY", "s3cret")
    assert _ui_config()["api_key_required"] is True
    with server.app.test_request_context("/agent/chat", headers={"X-API-KEY": "s3cret"}):
        server._require_agent_chat_api_key()          # accepted
    with server.app.test_request_context("/agent/chat", headers={"X-API-KEY": "wrong"}):
        with pytest.raises(PermissionError):
            server._require_agent_chat_api_key()
