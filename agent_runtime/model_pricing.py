"""What a model call cost, from the provider's own token counts.

Every model call already reports its tokens as an ``llm_usage`` trace event
(``streaming_trace.UsageCallbackHandler``). This module turns those counts into dollars, per call
and per turn, so a turn's cost can be returned with its answer and kept with the conversation.

The price table lives here and not in a JSON file because ``.dockerignore`` excludes ``*.json``:
a table shipped as JSON would be missing from the agent image and every turn would read as
unpriced. ``gis_harness/run.py`` reads this module too, so the harness and the server cannot
disagree about what a run cost.

A model with no per-token dollar price reports ``cost_usd: None``, never ``0``. NCSA Lumen spends
the key owner's coins instead, which have no dollar rate; its calls are priced in coins
(``cost_coins``) at the rate Lumen publishes per model (``agent_runtime/lumen_coins.py``).
AnvilGPT is Purdue's allocation and has neither. Calling an unpriced model "free" would make a
turn on it look cheaper than a priced one when nobody knows that it is. An operator who has a
dollar rate can supply it with ``AGENT_MODEL_PRICES``.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional

logger = logging.getLogger(__name__)

# USD per MILLION tokens. OpenAI rates as recorded in claude/benchmark's pricing.json (verified
# 2026-09-17 against developers.openai.com/api/docs/pricing); gpt-4o from the same page's
# long-standing rate. Moved here from gis_harness/pricing.json on 2026-10-14.
PRICES: Dict[str, Dict[str, Optional[float]]] = {
    "gpt-5.6-luna": {"input": 0.20, "cached_input": 0.02, "output": 1.20},
    "gpt-4o-2024-11-20": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "deepseek-v4-flash": {"input": None, "cached_input": None, "output": None},
}

# How many calls one turn's summary lists individually. The totals always cover every call; a
# runaway turn should not put thousands of rows into a response or a trace.
MAX_LISTED_CALLS = 200


def prices() -> Dict[str, Dict[str, Optional[float]]]:
    """The table, with ``AGENT_MODEL_PRICES`` (a JSON object of the same shape) laid over it.

    A malformed override is logged and ignored rather than raised: a typo in a price must not
    stop the agent answering."""
    table = {k: dict(v) for k, v in PRICES.items()}
    raw = (os.getenv("AGENT_MODEL_PRICES") or "").strip()
    if not raw:
        return table
    try:
        extra = json.loads(raw)
        if not isinstance(extra, dict):
            raise ValueError("expected a JSON object of {model: {input, cached_input, output}}")
        for model, rate in extra.items():
            if not isinstance(rate, dict):
                raise ValueError(f"{model}: expected an object of rates")
            table[str(model)] = {k: (None if rate.get(k) is None else float(rate[k]))
                                 for k in ("input", "cached_input", "output")}
    except Exception as exc:  # noqa: BLE001 - a bad override must not break a turn
        logger.warning("AGENT_MODEL_PRICES ignored: %s", exc)
    return table


def rate_for(model: Optional[str],
             table: Optional[Mapping[str, Mapping[str, Optional[float]]]] = None
             ) -> Optional[Mapping[str, Optional[float]]]:
    """The rate for a model id, or None. Exact id first, then without a ``provider:`` prefix."""
    table = prices() if table is None else table
    name = str(model or "").strip()
    if not name:
        return None
    if name in table:
        return table[name]
    bare = name.split(":", 1)[1] if ":" in name and "/" not in name.split(":", 1)[0] else ""
    return table.get(bare) if bare else None


def _int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def call_cost(model: Optional[str], input_tokens: Any, output_tokens: Any,
              cached_input_tokens: Any = 0,
              table: Optional[Mapping[str, Mapping[str, Optional[float]]]] = None) -> Optional[float]:
    """Dollars for one call, or None when the model has no price.

    Cached input is billed at the cached rate and the rest at the input rate, so a cached token
    is counted once. A model without a separate cached rate bills cached tokens as input."""
    rate = rate_for(model, table)
    if not rate or rate.get("input") is None or rate.get("output") is None:
        return None
    cached = min(_int(cached_input_tokens), _int(input_tokens))
    fresh = _int(input_tokens) - cached
    cached_rate = rate.get("cached_input")
    cost = (fresh * float(rate["input"])
            + cached * float(rate["input"] if cached_rate is None else cached_rate)
            + _int(output_tokens) * float(rate["output"])) / 1e6
    return round(cost, 6)


def summarize(calls: Iterable[Mapping[str, Any]],
              coins_available: Optional[float] = None) -> Dict[str, Any]:
    """One turn's cost from its ``llm_usage`` payloads, in dollars and in Lumen coins.

    Two currencies, kept apart because they are different money: ``cost_usd`` sums the calls
    with a dollar price (OpenAI), ``cost_coins`` the calls to NCSA Lumen, priced at Lumen's own
    published per-model coin rate (``lumen_coins``). Each is None only when NO call was priced
    in it, and ``priced_calls`` / ``coin_priced_calls`` say how many calls each covers. A model
    with neither price is listed in ``unpriced_models``. ``usage_absent`` counts calls whose
    provider sent no token counts at all, a different fact from a call that used none.
    ``coins_available``, when given, is the Lumen account's balance after the turn.
    """
    table = prices()
    rows: List[Dict[str, Any]] = []
    totals = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0,
              "reasoning_tokens": 0}
    by_model: Dict[str, Dict[str, Any]] = {}
    cost, priced, absent = 0.0, 0, 0
    coins, coin_priced = 0.0, 0
    unpriced: set = set()
    for call in calls or []:
        if not isinstance(call, Mapping):
            continue
        model = str(call.get("model") or "unknown")
        if call.get("usage") == "absent":
            absent += 1
        tin, tout = _int(call.get("input_tokens")), _int(call.get("output_tokens"))
        tcached, treason = _int(call.get("cached_input_tokens")), _int(call.get("reasoning_tokens"))
        c = call.get("cost_usd")
        if c is None and call.get("usage") != "absent" and call.get("provider") != "lumen":
            # A Lumen call is billed in coins, never at an OpenAI dollar rate for the same id.
            c = call_cost(model, tin, tout, tcached, table)
        cc = call.get("cost_coins")
        for key, value in (("input_tokens", tin), ("cached_input_tokens", tcached),
                           ("output_tokens", tout), ("reasoning_tokens", treason)):
            totals[key] += value
        m = by_model.setdefault(model, {"model": model, "calls": 0, "input_tokens": 0,
                                        "output_tokens": 0, "cost_usd": None, "cost_coins": None})
        m["calls"] += 1
        m["input_tokens"] += tin
        m["output_tokens"] += tout
        if c is not None:
            priced += 1
            cost += float(c)
            m["cost_usd"] = round((m["cost_usd"] or 0.0) + float(c), 6)
        if cc is not None:
            coin_priced += 1
            coins += float(cc)
            m["cost_coins"] = round((m["cost_coins"] or 0.0) + float(cc), 8)
        if c is None and cc is None and call.get("usage") != "absent":
            unpriced.add(model)
        if len(rows) < MAX_LISTED_CALLS:
            row = {"model": model, "input_tokens": tin, "output_tokens": tout,
                   "cost_usd": None if c is None else round(float(c), 6)}
            if call.get("provider"):
                row["provider"] = call["provider"]
            if cc is not None:
                row["cost_coins"] = round(float(cc), 8)
            if tcached:
                row["cached_input_tokens"] = tcached
            if treason:
                row["reasoning_tokens"] = treason
            for key in ("agent", "node"):
                if call.get(key):
                    row[key] = call[key]
            rows.append(row)
    n = sum(m["calls"] for m in by_model.values())
    return {
        "llm_calls": n,
        **totals,
        "cost_usd": round(cost, 6) if priced else None,
        "priced_calls": priced,
        "cost_coins": round(coins, 8) if coin_priced else None,
        "coin_priced_calls": coin_priced,
        "unpriced_models": sorted(unpriced),
        "usage_absent": absent,
        "by_model": sorted(by_model.values(), key=lambda m: -m["calls"]),
        "calls": rows,
        "calls_truncated": max(0, n - len(rows)),
        **({"coins_available": float(coins_available)} if coins_available is not None else {}),
    }


def memory_summary(usage: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """The part of a turn's cost kept on the conversation's ``chat_history`` entry.

    ``chat_memory`` maps new fields dynamically, so every value here keeps one type for good:
    counts are ints, ``cost_usd`` is a float or left out, and the model ids are a list of strings
    and never keys (a key like ``gpt-5.6-luna`` would become a nested object path). The
    per-call rows stay in the turn's trace, which stores them as text."""
    if not isinstance(usage, Mapping) or not usage.get("llm_calls"):
        return None
    out: Dict[str, Any] = {k: _int(usage.get(k)) for k in
                           ("llm_calls", "input_tokens", "cached_input_tokens", "output_tokens",
                            "reasoning_tokens", "priced_calls", "coin_priced_calls")}
    for key in ("cost_usd", "cost_coins", "coins_available"):
        if usage.get(key) is not None:
            out[key] = float(usage[key])
    out["models"] = [str(m.get("model")) for m in usage.get("by_model") or [] if m.get("model")]
    out["unpriced_models"] = [str(m) for m in usage.get("unpriced_models") or []]
    return out


__all__ = ["PRICES", "prices", "rate_for", "call_cost", "summarize", "memory_summary"]
