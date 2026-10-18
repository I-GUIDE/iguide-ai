"""The default model leaves Lumen while its pool runs low, and each user has a daily budget.

Two rules from the user (2026-10-16):
- When 90% of the Lumen coin pool (100 coins) is used, the process default switches to OpenAI
  gpt-5.6-luna, and it returns to Lumen once usage is back below 80%. The gap stops the default
  flipping on every turn while the balance hovers at one threshold.
- Each signed-in user may spend at most 3 Lumen coins and, separately, $3 of OpenAI per day,
  the day ending at midnight US Central. Over either, the next turn is refused.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime import file_store  # noqa: E402
from agent_runtime import lumen_failover as lf  # noqa: E402
from agent_runtime import usage_limit as ul  # noqa: E402

CENTRAL = ZoneInfo("America/Chicago")


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(file_store, "storage_root", lambda: tmp_path)
    for name in ("AGENT_LUMEN_FAILOVER", "AGENT_LUMEN_POOL_COINS", "AGENT_LUMEN_SWITCH_AT",
                 "AGENT_LUMEN_RETURN_AT", "AGENT_DAILY_COIN_LIMIT", "AGENT_DAILY_USD_LIMIT",
                 "AGENT_USAGE_TZ", "AGENT_LUMEN_FALLBACK_MODEL", "AGENT_LUMEN_FALLBACK_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "lumen")
    lf._reset()
    yield
    lf._reset()


# --- failover ------------------------------------------------------------------------------

def test_used_fraction_reads_the_balance_against_a_100_coin_pool():
    assert lf.used_fraction(99.93) == pytest.approx(0.0007)
    assert lf.used_fraction(9.9) == pytest.approx(0.901)
    assert lf.used_fraction(-2) == 0.0          # unlimited pool never fails over
    assert lf.used_fraction(None) is None       # unknown keeps the current side


def test_switches_at_90_percent_used_and_returns_below_80():
    assert lf.observe(15.0) is False   # 85% used, still on Lumen
    assert lf.observe(9.0) is True     # 91% used: fail over
    assert lf.observe(15.0) is True    # 85% used: stays failed over (no flapping)
    assert lf.observe(None) is True    # unknown balance keeps the side
    assert lf.observe(21.0) is False   # 79% used: back to Lumen


def test_the_side_survives_a_restart_inside_the_band(tmp_path):
    assert lf.observe(5.0) is True
    assert json.loads((tmp_path / "lumen_failover.json").read_text())["failed_over"] is True
    lf._reset()                        # a process restart
    assert lf.observe(15.0) is True    # 85% used: still failed over after the restart


def test_the_balance_is_read_at_most_once_a_minute(monkeypatch):
    from agent_runtime import lumen_coins

    reads = []
    monkeypatch.setattr(lumen_coins, "coins_available", lambda: (reads.append(1), 5.0)[1])
    assert lf.failed_over() is True
    assert lf.failed_over() is True
    assert len(reads) == 1


def test_only_a_lumen_default_can_fail_over(monkeypatch):
    from agent_runtime import lumen_coins

    monkeypatch.setattr(lumen_coins, "coins_available", lambda: 1.0)
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "openai")
    assert lf.failed_over() is False
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "lumen")
    monkeypatch.setenv("AGENT_LUMEN_FAILOVER", "0")
    assert lf.failed_over() is False


def test_the_default_model_is_luna_while_failed_over(monkeypatch):
    from agent_runtime import executor_factory as ef

    monkeypatch.setenv("LUMEN_API_KEY", "lumen-test-key")
    monkeypatch.setenv("OPENAI_KEY", "openai-test-key")
    monkeypatch.setattr(lf, "failed_over", lambda: True)
    llm = ef.build_default_llm()
    assert llm.model_name == "gpt-5.6-luna"
    assert "lumen" not in str(getattr(llm, "openai_api_base", "") or "")
    desc = ef.active_llm_description()
    assert desc["provider"] == "openai" and desc["model"] == "gpt-5.6-luna"
    monkeypatch.setattr(lf, "failed_over", lambda: False)
    assert ef.active_llm_description()["provider"] == "lumen"


# --- daily limit ---------------------------------------------------------------------------

ALICE = "http://cilogon.org/serverE/users/137206"
NOON = datetime(2026, 10, 16, 12, 0, tzinfo=CENTRAL)


def test_coins_and_dollars_are_limited_separately():
    ul.record(ALICE, {"cost_coins": 2.9, "cost_usd": 2.9}, now=NOON)
    assert ul.refusal(ALICE, now=NOON) is None
    ul.record(ALICE, {"cost_coins": 0.2, "cost_usd": None}, now=NOON)
    refused = ul.refusal(ALICE, now=NOON)
    assert refused["reason"] == "daily_limit" and "of 3 credits" in refused["error"]
    assert "Lumen" not in refused["error"] and "coin" not in refused["error"]
    assert "OpenAI" not in refused["error"]
    assert refused["spent"]["usd"] == pytest.approx(2.9)


def test_the_day_resets_at_midnight_central():
    late = datetime(2026, 10, 16, 23, 59, tzinfo=CENTRAL)
    after = datetime(2026, 10, 17, 0, 1, tzinfo=CENTRAL)
    ul.record(ALICE, {"cost_usd": 3.5}, now=late)
    assert ul.refusal(ALICE, now=late)["error"] == (
        "You have reached today's usage limit ($3.50 of $3). It resets at midnight US Central time.")
    assert ul.refusal(ALICE, now=after) is None
    assert ul.resets_at(late).startswith("2026-10-17T00:00:00-05:00")


def test_nobody_signed_in_is_never_limited_or_recorded(tmp_path):
    assert ul.record(None, {"cost_usd": 100}) is None
    assert ul.refusal(None) is None
    assert not (tmp_path / "usage_ledger").exists()


def test_the_ledger_holds_a_digest_not_the_user_id(tmp_path):
    ul.record(ALICE, {"cost_coins": 0.01}, now=NOON)
    text = (tmp_path / "usage_ledger" / "2026-10-16.json").read_text()
    assert "cilogon" not in text and "137206" not in text


def test_a_limit_of_zero_disables_it(monkeypatch):
    monkeypatch.setenv("AGENT_DAILY_USD_LIMIT", "0")
    ul.record(ALICE, {"cost_usd": 50}, now=NOON)
    assert ul.refusal(ALICE, now=NOON) is None


def test_a_turn_charges_the_signed_in_user_and_reports_the_day(monkeypatch):
    from agent_runtime import identity
    from agent_runtime.agent_chat_service import _charge_turn

    token = identity.set_user(identity.User(id=ALICE, role=4))
    try:
        out = _charge_turn({"llm_calls": 2, "cost_coins": 0.5, "cost_usd": None})
    finally:
        identity.reset_user(token)
    assert out["today"]["coins"] == pytest.approx(0.5) and out["today"]["turns"] == 1
    assert out["today"]["limits"] == {"coins": 3.0, "usd": 3.0}


def test_the_routes_refuse_a_spent_user_before_running_a_turn(monkeypatch):
    import api.server as server
    from agent_runtime import identity

    for _ in range(2):
        ul.record(ALICE, {"cost_coins": 2.0})
    user = identity.User(id=ALICE, role=4)
    monkeypatch.setattr(server, "_require_user", lambda: user)
    monkeypatch.setattr(server, "_require_agent_chat_api_key", lambda *_a, **_k: None)
    ran = []
    monkeypatch.setattr(server, "run_agent_chat", lambda **kw: ran.append(kw))
    client = server.app.test_client()
    r = client.post("/agent/chat", json={"userQuery": "hello"})
    assert r.status_code == 429 and r.get_json()["reason"] == "daily_limit"
    s = client.post("/agent/chat/stream", json={"userQuery": "hello"})
    body = s.get_data(as_text=True)
    assert "event: error" in body and "daily_limit" in body
    assert not ran
