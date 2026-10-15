"""Every turn reports what its model calls cost, per call and in total, and keeps it.

Each model call already emitted an ``llm_usage`` trace event with the provider's token counts,
but nothing priced them outside the GIS harness, and nothing kept them with the conversation. A
turn now carries ``usage``: each call with its ``cost_usd``, a per-model breakdown, and the
turn's totals. It is returned with the answer, stored on the conversation's history entry in a
type-stable compact form, and stored whole on the turn's trace.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime import model_pricing as mp  # noqa: E402
from agent_runtime import streaming_trace as st  # noqa: E402
from rag_pipeline import memory_module as mm  # noqa: E402


@pytest.fixture(autouse=True)
def _no_override(monkeypatch):
    monkeypatch.delenv("AGENT_MODEL_PRICES", raising=False)


def test_a_priced_call_bills_cached_input_at_the_cached_rate():
    # gpt-5.6-luna: 0.20 input, 0.02 cached, 1.20 output, USD per million tokens.
    got = mp.call_cost("gpt-5.6-luna", 1000, 500, cached_input_tokens=200)
    assert got == pytest.approx((800 * 0.20 + 200 * 0.02 + 500 * 1.20) / 1e6)


def test_an_unpriced_model_costs_none_not_zero():
    assert mp.call_cost("deepseek-v4-flash", 1000, 500) is None
    assert mp.call_cost("some-model-nobody-priced", 1000, 500) is None


def test_a_provider_prefix_still_finds_the_rate():
    assert mp.call_cost("openai:gpt-4o", 1_000_000, 0) == pytest.approx(2.50)


def test_a_turn_mixing_priced_and_unpriced_calls_says_how_much_the_cost_covers():
    s = mp.summarize([
        {"model": "gpt-5.6-luna", "input_tokens": 1000, "output_tokens": 100},
        {"model": "deepseek-v4-flash", "input_tokens": 5000, "output_tokens": 300},
        {"model": "deepseek-v4-flash", "usage": "absent"},
    ])
    assert s["llm_calls"] == 3
    assert s["input_tokens"] == 6000 and s["output_tokens"] == 400
    assert s["cost_usd"] == pytest.approx((1000 * 0.20 + 100 * 1.20) / 1e6)
    assert s["priced_calls"] == 1
    assert s["unpriced_models"] == ["deepseek-v4-flash"]
    assert s["usage_absent"] == 1
    assert [c["cost_usd"] for c in s["calls"]][:2] == [pytest.approx(0.00032), None]
    assert {m["model"]: m["calls"] for m in s["by_model"]} == {"gpt-5.6-luna": 1,
                                                              "deepseek-v4-flash": 2}


def test_a_turn_with_no_priced_call_reports_none():
    s = mp.summarize([{"model": "deepseek-v4-flash", "input_tokens": 10, "output_tokens": 1}])
    assert s["cost_usd"] is None and s["priced_calls"] == 0


def test_an_operator_can_price_a_model_and_a_bad_override_is_ignored(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL_PRICES",
                       json.dumps({"deepseek-v4-flash": {"input": 1.0, "output": 2.0}}))
    assert mp.call_cost("deepseek-v4-flash", 1_000_000, 1_000_000) == pytest.approx(3.0)
    monkeypatch.setenv("AGENT_MODEL_PRICES", "{not json")
    assert mp.call_cost("deepseek-v4-flash", 1, 1) is None
    assert mp.call_cost("gpt-4o", 1_000_000, 0) == pytest.approx(2.50)


def test_the_listed_calls_are_bounded_but_the_totals_are_not(monkeypatch):
    monkeypatch.setattr(mp, "MAX_LISTED_CALLS", 3)
    s = mp.summarize([{"model": "gpt-4o", "input_tokens": 1, "output_tokens": 1}] * 5)
    assert s["llm_calls"] == 5 and len(s["calls"]) == 3 and s["calls_truncated"] == 2
    assert s["input_tokens"] == 5


def _llm_result(input_tokens, output_tokens, cached=0):
    message = SimpleNamespace(usage_metadata={
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "input_token_details": {"cache_read": cached} if cached else {}})
    return SimpleNamespace(generations=[[SimpleNamespace(message=message)]], llm_output={})


def test_each_call_event_carries_its_cost_and_the_turn_sums_them():
    with st.usage_collection() as handler:
        for run, (model, tin, tout) in enumerate([("gpt-5.6-luna", 1000, 100),
                                                  ("deepseek-v4-flash", 2000, 50)]):
            handler.on_chat_model_start({}, [], run_id=run,
                                        invocation_params={"model": model})
            handler.on_llm_end(_llm_result(tin, tout), run_id=run)
        usage = st.turn_usage()
    assert [c["cost_usd"] for c in handler.calls] == [pytest.approx(0.00032), None]
    assert usage["llm_calls"] == 2 and usage["cost_usd"] == pytest.approx(0.00032)
    assert usage["unpriced_models"] == ["deepseek-v4-flash"]
    assert st.turn_usage() is None   # outside the turn there is nothing to report


def test_a_nested_collection_reuses_the_turn_handler():
    with st.usage_collection() as outer:
        with st.usage_collection() as inner:
            assert inner is outer


def test_the_history_entry_keeps_a_type_stable_summary():
    usage = mp.summarize([{"model": "gpt-5.6-luna", "input_tokens": 1000, "output_tokens": 100},
                          {"model": "deepseek-v4-flash", "input_tokens": 10, "output_tokens": 1}])
    kept = mp.memory_summary(usage)
    assert kept["llm_calls"] == 2 and isinstance(kept["input_tokens"], int)
    assert isinstance(kept["cost_usd"], float)
    assert kept["models"] == ["gpt-5.6-luna", "deepseek-v4-flash"]
    assert "calls" not in kept and "by_model" not in kept
    # No model id may become a field name: it would be a nested object path in the mapping.
    assert all("." not in k and "-" not in k for k in kept)
    unpriced = mp.memory_summary(mp.summarize([{"model": "deepseek-v4-flash",
                                                "input_tokens": 1, "output_tokens": 1}]))
    assert "cost_usd" not in unpriced   # left out, not null and not 0
    assert mp.memory_summary(mp.summarize([])) is None


class _Store:
    def __init__(self):
        self.docs = {"m1": {"chat_history": [], "owner_id": "u1"}}
        self.updates, self.indexed = [], []

    def get(self, *, index, id, **_kw):
        return {"_id": id, "_source": dict(self.docs[id])}

    def update(self, *, index, id, body, **_kw):
        self.updates.append(body)
        self.docs[id].update(body["doc"])

    def index(self, *, index, id, body, **_kw):
        self.indexed.append(body)


def test_update_memory_stores_the_turn_cost_on_its_history_entry(monkeypatch):
    store = _Store()
    monkeypatch.setattr(mm, "_get_opensearch_client", lambda: store)
    usage = mp.summarize([{"model": "gpt-4o", "input_tokens": 100, "output_tokens": 10}])
    mm.update_memory("m1", user_query="q", message_id="x", answer="a", elements=[], usage=usage)
    entry = store.docs["m1"]["chat_history"][-1]
    assert entry["usage"]["cost_usd"] == pytest.approx((100 * 2.5 + 10 * 10) / 1e6)
    mm.update_memory("m1", user_query="q2", message_id="y", answer="b", elements=[])
    assert "usage" not in store.docs["m1"]["chat_history"][-1]


def test_the_trace_keeps_the_whole_cost_record(monkeypatch):
    store = _Store()
    monkeypatch.setattr(mm, "_get_opensearch_client", lambda: store)
    usage = mp.summarize([{"model": "gpt-4o", "input_tokens": 100, "output_tokens": 10}])
    out = mm.save_turn_trace("m1", thread_id="t", query="q", events=[{"event": "x"}], usage=usage)
    assert out["stored"]
    stored = store.indexed[-1]["usage"]
    # An unmapped field: stored as JSON text so its shape can never break the mapping.
    assert isinstance(stored, str)
    assert json.loads(stored)["calls"][0]["model"] == "gpt-4o"


def test_the_api_result_forwards_usage():
    import api.server as server

    usage = {"llm_calls": 1, "cost_usd": 0.001}
    assert server._format_agent_chat_result({"answer": "a", "usage": usage})["usage"] == usage
    assert "usage" not in server._format_agent_chat_result({"answer": "a"})
