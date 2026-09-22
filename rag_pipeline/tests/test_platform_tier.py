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


# --- the tier names its OpenSearch too --------------------------------------------

def test_the_dev_tier_names_its_cluster(monkeypatch):
    """Dev's OpenSearch moved hosts; the tier is where that fact belongs.

    It used to live only in OPENSEARCH_NODE, so when the cluster moved the variable stayed
    pinned to the old host — which kept answering, and kept accepting writes, while everything
    else in the tier had moved on. Nothing failed loudly.
    """
    monkeypatch.delenv("OPENSEARCH_NODE", raising=False)
    monkeypatch.setenv("PLATFORM_TIER", "dev")
    assert pe.opensearch_url() == "https://149.165.155.135:9200"


def test_prod_inherits_nothing(monkeypatch):
    """The worst thing this table could do is point production at dev's cluster."""
    monkeypatch.delenv("OPENSEARCH_NODE", raising=False)
    monkeypatch.setenv("PLATFORM_TIER", "prod")
    assert pe.opensearch_url() == ""


def test_an_explicit_host_still_wins(monkeypatch):
    monkeypatch.setenv("PLATFORM_TIER", "dev")
    monkeypatch.setenv("OPENSEARCH_NODE", "https://one-off:9200")
    assert pe.opensearch_url() == "https://one-off:9200"


def test_drift_is_said_out_loud(monkeypatch):
    """Winning quietly is the failure mode. It may win, but it has to announce itself."""
    monkeypatch.setenv("PLATFORM_TIER", "dev")
    monkeypatch.setenv("OPENSEARCH_NODE", "https://149.165.155.195:9200")
    warning = pe.opensearch_drift_warning()
    assert warning and "149.165.155.195" in warning and "149.165.155.135" in warning


@pytest.mark.parametrize("node", ["https://149.165.155.135:9200", "https://149.165.155.135:9200/"])
def test_agreeing_with_the_tier_is_silent(monkeypatch, node):
    monkeypatch.setenv("PLATFORM_TIER", "dev")
    monkeypatch.setenv("OPENSEARCH_NODE", node)
    assert pe.opensearch_drift_warning() is None


def test_no_tier_means_no_opinion(monkeypatch):
    monkeypatch.delenv("PLATFORM_TIER", raising=False)
    monkeypatch.setenv("OPENSEARCH_NODE", "https://anything:9200")
    assert pe.opensearch_drift_warning() is None


# --- the credential moves with the tier -------------------------------------------

def test_a_tiered_credential_beats_the_bare_one(monkeypatch):
    """The REVERSE of the URL rule, and deliberately.

    For a URL the tier supplies a value and the explicit variable overrides it. For a
    credential the tier supplies no value at all — secrets are not in this repository — so the
    tiered name is merely the more specific one. Getting it backwards would let an old bare
    OPENSEARCH_PASSWORD silently pin every tier to one account, which is the half-switched
    state this module exists to prevent.
    """
    monkeypatch.setenv("PLATFORM_TIER", "dev")
    monkeypatch.setenv("OPENSEARCH_USERNAME", "shared")
    monkeypatch.setenv("OPENSEARCH_PASSWORD", "shared-pw")
    monkeypatch.setenv("OPENSEARCH_USERNAME_DEV", "dev-user")
    monkeypatch.setenv("OPENSEARCH_PASSWORD_DEV", "dev-pw")
    assert pe.opensearch_credentials() == ("dev-user", "dev-pw")


def test_flipping_the_tier_flips_the_credential(monkeypatch):
    """The whole point on a host that switches tiers."""
    monkeypatch.setenv("OPENSEARCH_USERNAME_DEV", "dev-user")
    monkeypatch.setenv("OPENSEARCH_PASSWORD_DEV", "dev-pw")
    monkeypatch.setenv("OPENSEARCH_USERNAME_PROD", "prod-user")
    monkeypatch.setenv("OPENSEARCH_PASSWORD_PROD", "prod-pw")
    monkeypatch.setenv("PLATFORM_TIER", "dev")
    assert pe.opensearch_credentials() == ("dev-user", "dev-pw")
    monkeypatch.setenv("PLATFORM_TIER", "prod")
    assert pe.opensearch_credentials() == ("prod-user", "prod-pw")


def test_half_a_tiered_pair_does_not_mix_accounts(monkeypatch):
    """A username for the tier and a password from the bare variable would be two accounts."""
    monkeypatch.setenv("PLATFORM_TIER", "dev")
    monkeypatch.setenv("OPENSEARCH_USERNAME", "shared")
    monkeypatch.setenv("OPENSEARCH_PASSWORD", "shared-pw")
    monkeypatch.setenv("OPENSEARCH_USERNAME_DEV", "dev-user")
    monkeypatch.delenv("OPENSEARCH_PASSWORD_DEV", raising=False)
    user, pwd = pe.opensearch_credentials()
    assert user == "dev-user" and pwd == "", "the tiered pair is selected whole, or not at all"


def test_the_bare_credential_still_works_untiered(monkeypatch):
    monkeypatch.delenv("PLATFORM_TIER", raising=False)
    monkeypatch.setenv("OPENSEARCH_USERNAME", "shared")
    monkeypatch.setenv("OPENSEARCH_PASSWORD", "shared-pw")
    assert pe.opensearch_credentials() == ("shared", "shared-pw")
    assert pe.opensearch_credential_warning() is None


def test_an_untiered_credential_under_a_tier_is_flagged(monkeypatch):
    """Not an error — one host, one tier, forever is fine. A trap only when the host flips."""
    monkeypatch.setenv("PLATFORM_TIER", "dev")
    monkeypatch.setenv("OPENSEARCH_USERNAME", "shared")
    monkeypatch.delenv("OPENSEARCH_USERNAME_DEV", raising=False)
    monkeypatch.delenv("OPENSEARCH_PASSWORD_DEV", raising=False)
    warning = pe.opensearch_credential_warning()
    assert warning and "OPENSEARCH_USERNAME_DEV" in warning


def test_no_credential_at_all_is_not_a_warning(monkeypatch):
    """Nothing configured is a different problem, and the client's own error says it better."""
    monkeypatch.setenv("PLATFORM_TIER", "dev")
    for name in ("OPENSEARCH_USERNAME", "OPENSEARCH_PASSWORD",
                 "OPENSEARCH_USERNAME_DEV", "OPENSEARCH_PASSWORD_DEV"):
        monkeypatch.delenv(name, raising=False)
    assert pe.opensearch_credential_warning() is None


# --- SEARCH_TIER: which corpus, independent of which platform ----------------------

def test_search_follows_the_platform_tier_by_default(monkeypatch):
    """One switch still moves everything unless someone deliberately splits them."""
    monkeypatch.delenv("SEARCH_TIER", raising=False)
    monkeypatch.setenv("PLATFORM_TIER", "dev")
    monkeypatch.setenv("OPENSEARCH_INDEX", "new-opensearch-index")
    monkeypatch.setenv("OPENSEARCH_INDEX_DEV", "iguide-platform-embeddings-dev")
    assert pe.search_tier() == "dev"
    assert pe.search_index() == "iguide-platform-embeddings-dev"


def test_search_can_be_pointed_at_the_other_tier(monkeypatch):
    """The dev platform against the prod knowledge base is an ordinary thing to want."""
    monkeypatch.setenv("PLATFORM_TIER", "dev")
    monkeypatch.setenv("SEARCH_TIER", "prod")
    monkeypatch.setenv("OPENSEARCH_INDEX", "new-opensearch-index")
    monkeypatch.setenv("OPENSEARCH_INDEX_DEV", "iguide-platform-embeddings-dev")
    assert pe.search_index() == "new-opensearch-index", "must NOT pick up the dev index"
    # ...and identity is untouched by it.
    assert pe.current_tier() == "dev"
    assert pe.check_tokens_url().startswith("https://backend-dev.i-guide.io")


def test_a_split_is_said_out_loud(monkeypatch):
    monkeypatch.setenv("PLATFORM_TIER", "dev")
    monkeypatch.setenv("SEARCH_TIER", "prod")
    note = pe.search_tier_note()
    assert note and "SEARCH_TIER=prod" in note and "PLATFORM_TIER=dev" in note


def test_no_split_is_silent(monkeypatch):
    monkeypatch.setenv("PLATFORM_TIER", "dev")
    monkeypatch.setenv("SEARCH_TIER", "dev")
    assert pe.search_tier_note() is None


def test_an_unknown_search_tier_refuses_rather_than_guessing(monkeypatch):
    """Falling back would search the wrong corpus and answer confidently from it."""
    monkeypatch.setenv("PLATFORM_TIER", "dev")
    monkeypatch.setenv("SEARCH_TIER", "staging")
    monkeypatch.setenv("OPENSEARCH_INDEX", "new-opensearch-index")
    with pytest.raises(ValueError):
        pe.search_index()


def test_an_empty_tiered_value_does_not_blank_a_working_one(monkeypatch):
    """A half-written FOO_PROD= must not erase FOO."""
    monkeypatch.setenv("PLATFORM_TIER", "prod")
    monkeypatch.delenv("SEARCH_TIER", raising=False)
    monkeypatch.setenv("OPENSEARCH_INDEX", "new-opensearch-index")
    monkeypatch.setenv("OPENSEARCH_INDEX_PROD", "")
    assert pe.search_index() == "new-opensearch-index"


def test_an_overridden_node_takes_the_untiered_credential(monkeypatch):
    """A credential must never be paired with a host it does not belong to.

    Caught by testing what a restart would do during the cluster migration: OPENSEARCH_NODE
    pinned the old cluster while the tier supplied the NEW cluster's credential, and the next
    restart would have 401'd and stopped conversations saving.
    """
    monkeypatch.setenv("PLATFORM_TIER", "dev")
    monkeypatch.setenv("OPENSEARCH_NODE", "https://149.165.155.195:9200")   # not the tier's
    monkeypatch.setenv("OPENSEARCH_USERNAME", "old-user")
    monkeypatch.setenv("OPENSEARCH_PASSWORD", "old-pw")
    monkeypatch.setenv("OPENSEARCH_USERNAME_DEV", "new-user")
    monkeypatch.setenv("OPENSEARCH_PASSWORD_DEV", "new-pw")
    assert pe.opensearch_credentials() == ("old-user", "old-pw")


def test_a_node_agreeing_with_its_tier_uses_the_tiered_credential(monkeypatch):
    monkeypatch.setenv("PLATFORM_TIER", "dev")
    monkeypatch.setenv("OPENSEARCH_NODE", "https://149.165.155.135:9200")   # IS the tier's
    monkeypatch.setenv("OPENSEARCH_USERNAME", "old-user")
    monkeypatch.setenv("OPENSEARCH_PASSWORD", "old-pw")
    monkeypatch.setenv("OPENSEARCH_USERNAME_DEV", "new-user")
    monkeypatch.setenv("OPENSEARCH_PASSWORD_DEV", "new-pw")
    assert pe.opensearch_credentials() == ("new-user", "new-pw")


def test_no_explicit_node_uses_the_tiered_credential(monkeypatch):
    monkeypatch.delenv("OPENSEARCH_NODE", raising=False)
    monkeypatch.setenv("PLATFORM_TIER", "dev")
    monkeypatch.setenv("OPENSEARCH_USERNAME_DEV", "new-user")
    monkeypatch.setenv("OPENSEARCH_PASSWORD_DEV", "new-pw")
    assert pe.opensearch_credentials() == ("new-user", "new-pw")


# --- coming back here after signing in ---------------------------------------------

def test_signin_url_is_unchanged_until_a_domain_id_is_configured(monkeypatch):
    """OFF by default, because the far end has to know the id before it means anything.

    The frontend resolves `redirect-domain-id` against its own redirect-whitelist.json. Until
    that file names the agent, sending the parameter achieves nothing — so this stays exactly
    as it was rather than emitting a parameter that only shows up in someone's warning log.
    """
    monkeypatch.setenv("PLATFORM_TIER", "dev")
    monkeypatch.delenv("PLATFORM_REDIRECT_DOMAIN_ID", raising=False)
    assert pe.signin_url() == "https://dev.i-guide.io/auth/login"


def test_signin_url_carries_the_return_trip_when_configured(monkeypatch):
    monkeypatch.setenv("PLATFORM_TIER", "dev")
    monkeypatch.setenv("PLATFORM_REDIRECT_DOMAIN_ID", "agent")
    monkeypatch.delenv("PLATFORM_REDIRECT_PATH", raising=False)
    assert pe.signin_url() == (
        "https://dev.i-guide.io/auth/login?redirect-domain-id=agent&redirect-path=%2F")


def test_the_return_path_is_forced_absolute_and_encoded(monkeypatch):
    """The frontend decodeURIComponent()s it and requires a leading "/" — a relative path is
    silently dropped for the platform's own profile page, which is the bug this avoids."""
    monkeypatch.setenv("PLATFORM_TIER", "dev")
    monkeypatch.setenv("PLATFORM_REDIRECT_DOMAIN_ID", "agent")
    monkeypatch.setenv("PLATFORM_REDIRECT_PATH", "rs")
    assert pe.signin_url().endswith("redirect-path=%2Frs")


def test_prod_signs_in_at_the_platform_not_dev(monkeypatch):
    """The pairing the tier exists to keep straight."""
    monkeypatch.delenv("PLATFORM_SIGNIN_URL", raising=False)
    monkeypatch.delenv("PLATFORM_REDIRECT_DOMAIN_ID", raising=False)
    monkeypatch.setenv("PLATFORM_TIER", "prod")
    assert pe.signin_url() == "https://platform.i-guide.io/auth/login"
    monkeypatch.setenv("PLATFORM_TIER", "dev")
    assert pe.signin_url() == "https://dev.i-guide.io/auth/login"
