"""One switch for which platform this agent talks to.

The platform comes in pairs — a frontend and the backend that mints its cookies — and the agent
needs three endpoints from them. Setting those one at a time invites the state this exists to
prevent: two pointing at dev, one left on prod, and a verification that fails for a reason
nobody can see from the outside.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime import platform_endpoints as pe  # noqa: E402
from agent_runtime import identity as idm  # noqa: E402


@pytest.fixture(autouse=True)
def env(monkeypatch):
    for var in ("PLATFORM_TIER", "PLATFORM_REFRESH_URL", "PLATFORM_SIGNIN_URL",
                "PLATFORM_CHECK_TOKENS_URL", "JWT_ACCESS_TOKEN_NAME"):
        monkeypatch.delenv(var, raising=False)


# --- the switch -----------------------------------------------------------------

def test_no_tier_means_no_derived_urls():
    """A deployment naming its URLs individually must not have a tier invented for it."""
    assert pe.current_tier() is None
    assert (pe.refresh_url(), pe.signin_url(), pe.check_tokens_url()) == ("", "", "")


def test_dev_resolves_the_whole_set(monkeypatch):
    monkeypatch.setenv("PLATFORM_TIER", "dev")
    assert pe.refresh_url() == "https://backend-dev.i-guide.io/api/refresh-token"
    assert pe.check_tokens_url() == "https://backend-dev.i-guide.io/api/check-tokens"
    assert pe.signin_url() == "https://dev.i-guide.io/auth/login"


def test_prod_resolves_the_whole_set(monkeypatch):
    monkeypatch.setenv("PLATFORM_TIER", "prod")
    assert pe.refresh_url() == "https://backend.i-guide.io/api/refresh-token"
    assert pe.check_tokens_url() == "https://backend.i-guide.io/api/check-tokens"
    assert pe.signin_url() == "https://platform.i-guide.io/auth/login"


def test_one_switch_moves_every_endpoint(monkeypatch):
    """The whole point: no half-switched state where two say dev and one says prod."""
    monkeypatch.setenv("PLATFORM_TIER", "dev")
    dev = (pe.refresh_url(), pe.check_tokens_url(), pe.signin_url())
    monkeypatch.setenv("PLATFORM_TIER", "prod")
    prod = (pe.refresh_url(), pe.check_tokens_url(), pe.signin_url())
    assert all(d != p for d, p in zip(dev, prod))
    assert all("-dev" in d or "dev." in d for d in dev)
    assert not any("dev" in p for p in prod)


@pytest.mark.parametrize("bad", ["development", "production", "staging", "DEV1", "true"])
def test_an_unknown_tier_raises(monkeypatch, bad):
    """Picking the wrong platform silently verifies tokens against a backend that never minted
    them, and the resulting 'invalid token' explains nothing."""
    monkeypatch.setenv("PLATFORM_TIER", bad)
    with pytest.raises(ValueError) as exc:
        pe.current_tier()
    assert "dev, prod" in str(exc.value)


def test_case_and_whitespace_are_forgiven(monkeypatch):
    monkeypatch.setenv("PLATFORM_TIER", "  PROD ")
    assert pe.current_tier() == "prod"


# --- explicit still wins ---------------------------------------------------------

def test_an_explicit_url_beats_the_tier(monkeypatch):
    """A tier table cannot anticipate a staging host somebody stands up next month."""
    monkeypatch.setenv("PLATFORM_TIER", "dev")
    monkeypatch.setenv("PLATFORM_CHECK_TOKENS_URL", "https://staging.example.org/api/check-tokens")
    assert pe.check_tokens_url() == "https://staging.example.org/api/check-tokens"
    assert pe.refresh_url() == "https://backend-dev.i-guide.io/api/refresh-token"   # untouched


def test_identity_reads_the_resolved_url(monkeypatch):
    """The resolver is not decorative: introspection has to actually use it."""
    monkeypatch.setenv("PLATFORM_TIER", "prod")
    assert idm._check_tokens_url() == "https://backend.i-guide.io/api/check-tokens"


# --- the half-switched state it cannot fix, but must name -------------------------

def test_prod_tier_with_a_dev_cookie_warns(monkeypatch):
    """The cookie name is the platform's own setting and does not move with the tier. Mismatched,
    every request fails and the failure looks like a rejected token, not a misconfiguration."""
    monkeypatch.setenv("PLATFORM_TIER", "prod")
    monkeypatch.setenv("JWT_ACCESS_TOKEN_NAME", "jwt-access-token-dev")
    warning = pe.consistency_warning()
    assert warning and "dev" in warning and "rejected" in warning


def test_dev_tier_with_a_prod_cookie_warns(monkeypatch):
    monkeypatch.setenv("PLATFORM_TIER", "dev")
    monkeypatch.setenv("JWT_ACCESS_TOKEN_NAME", "jwt-access-token")
    assert pe.consistency_warning() is not None


def test_a_matching_pair_is_quiet(monkeypatch):
    monkeypatch.setenv("PLATFORM_TIER", "dev")
    monkeypatch.setenv("JWT_ACCESS_TOKEN_NAME", '"jwt-access-token-dev"')   # quoted, as in .env
    assert pe.consistency_warning() is None


def test_no_tier_no_warning(monkeypatch):
    monkeypatch.setenv("JWT_ACCESS_TOKEN_NAME", "jwt-access-token-dev")
    assert pe.consistency_warning() is None
