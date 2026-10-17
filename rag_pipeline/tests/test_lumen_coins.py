"""A turn on NCSA Lumen reports its cost in coins, at Lumen's own published rate.

Lumen bills coins, not dollars, so Stage 59 left every Lumen turn at ``cost_usd: null``, which on
the VM's default model meant every turn. Lumen publishes each model's coin rate on
``GET /v1/models/<id>`` and the account's balance on ``GET /v1/usage``. A call is Lumen's when it
went to Lumen's endpoint, never by model id alone: ``gpt-4o`` exists on both.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime import lumen_coins as lc  # noqa: E402
from agent_runtime import model_pricing as mp  # noqa: E402
from agent_runtime import streaming_trace as st  # noqa: E402

LUMEN = "https://lumen.ncsa.illinois.edu/v1"
RATE = {"input_cost_per_million": 0.161857, "output_cost_per_million": 0.37146}


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    lc._reset_cache()
    monkeypatch.delenv("LUMEN_URL", raising=False)
    monkeypatch.delenv("AGENT_MODEL_PRICES", raising=False)
    monkeypatch.setenv("LUMEN_API_KEY", "test-key")
    yield
    lc._reset_cache()


@pytest.fixture
def lumen(monkeypatch):
    """A fake Lumen: models carry RATE, the account has 99.5 coins. Counts lookups."""
    seen = []

    def fake_get(path):
        seen.append(path)
        if path == "/usage":
            return {"coins_available": 99.5}
        if path.startswith("/models/"):
            return dict(RATE)
        return None

    monkeypatch.setattr(lc, "_get", fake_get)
    return seen


def test_a_call_is_lumens_by_its_endpoint():
    assert lc.is_lumen(LUMEN)
    assert lc.is_lumen("https://lumen.ncsa.illinois.edu/v1/")
    assert not lc.is_lumen("https://api.openai.com/v1")
    assert not lc.is_lumen(None)


def test_coins_follow_lumens_published_formula(lumen):
    # The 2026-10-14 one-line test turn's first call sizes: 61,809 in, 3,312 out (whole turn).
    got = lc.call_coins("deepseek-v4-flash", 61_809, 3_312)
    assert got == pytest.approx(61_809 / 1e6 * 0.161857 + 3_312 / 1e6 * 0.37146)


def test_a_rate_is_looked_up_once_and_a_failure_is_remembered(lumen, monkeypatch):
    lc.call_coins("deepseek-v4-flash", 1, 1)
    lc.call_coins("deepseek-v4-flash", 2, 2)
    assert lumen.count("/models/deepseek-v4-flash") == 1
    monkeypatch.setattr(lc, "_get", lambda path: (lumen.append(path), None)[1])
    assert lc.call_coins("unknown-model", 1, 1) is None
    assert lc.call_coins("unknown-model", 1, 1) is None
    assert lumen.count("/models/unknown-model") == 1


def test_no_key_means_no_lookup(monkeypatch):
    monkeypatch.delenv("LUMEN_API_KEY", raising=False)
    assert lc.coin_rate("deepseek-v4-flash") is None
    assert lc.coins_available() is None


def _end_call(handler, run, model, base, tin, tout):
    serialized = {"kwargs": {"openai_api_base": base, "openai_api_key": {"lc": 1, "type": "secret"}}}
    handler.on_chat_model_start(serialized, [], run_id=run, invocation_params={"model": model})
    message = SimpleNamespace(usage_metadata={"input_tokens": tin, "output_tokens": tout})
    handler.on_llm_end(SimpleNamespace(generations=[[SimpleNamespace(message=message)]],
                                       llm_output={}), run_id=run)


def test_each_lumen_call_carries_coins_and_the_turn_reports_the_balance(lumen):
    with st.usage_collection() as handler:
        _end_call(handler, 1, "deepseek-v4-flash", LUMEN, 1_000_000, 100_000)
        _end_call(handler, 2, "gpt-5.6-luna", "https://api.openai.com/v1", 1000, 100)
        usage = st.turn_usage()
    lumen_call, openai_call = handler.calls
    assert lumen_call["provider"] == "lumen"
    assert lumen_call["cost_coins"] == pytest.approx(0.161857 + 0.1 * 0.37146)
    assert lumen_call["cost_usd"] is None
    assert "cost_coins" not in openai_call and openai_call["cost_usd"] == pytest.approx(0.00032)
    assert usage["cost_coins"] == pytest.approx(0.161857 + 0.037146)
    assert usage["coin_priced_calls"] == 1 and usage["priced_calls"] == 1
    assert usage["cost_usd"] == pytest.approx(0.00032)
    assert usage["unpriced_models"] == []
    assert usage["coins_available"] == 99.5


def test_the_same_model_id_on_lumen_is_billed_in_coins_not_at_openai_dollar_rates(lumen):
    with st.usage_collection() as handler:
        _end_call(handler, 1, "gpt-4o", LUMEN, 1_000_000, 0)
        usage = st.turn_usage()
    assert usage["cost_usd"] is None
    assert usage["cost_coins"] == pytest.approx(0.161857)


def test_no_lumen_call_means_no_balance_lookup(lumen):
    with st.usage_collection() as handler:
        _end_call(handler, 1, "gpt-4o", "https://api.openai.com/v1", 10, 1)
        usage = st.turn_usage()
    assert "/usage" not in lumen and "coins_available" not in usage


def test_the_key_and_the_endpoint_are_never_recorded(lumen):
    with st.usage_collection() as handler:
        _end_call(handler, 1, "deepseek-v4-flash", LUMEN, 10, 1)
    text = repr(handler.calls)
    assert "lumen.ncsa" not in text and "secret" not in text and "openai_api" not in text


def test_the_history_entry_keeps_coins_as_floats():
    usage = mp.summarize([{"model": "deepseek-v4-flash", "provider": "lumen", "input_tokens": 10,
                           "output_tokens": 1, "cost_coins": 0.0000020}], coins_available=99.0)
    kept = mp.memory_summary(usage)
    assert isinstance(kept["cost_coins"], float) and kept["coins_available"] == 99.0
    assert kept["coin_priced_calls"] == 1 and "cost_usd" not in kept
    assert kept["unpriced_models"] == []
