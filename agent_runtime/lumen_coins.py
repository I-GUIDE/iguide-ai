"""What a call to NCSA Lumen cost in Lumen's own currency, coins, and what the account has left.

Lumen spends coins, not dollars, and publishes each model's rate in coins per million tokens on
``GET /v1/models/<id>`` (``input_cost_per_million``, ``output_cost_per_million``). Its own help
page gives the formula, ``input_tokens / 1e6 * input_rate + output_tokens / 1e6 * output_rate``,
with no separate cached-input rate. ``GET /v1/usage`` returns the account's remaining balance as
``coins_available`` (``-2`` for an unlimited pool, ``null`` for none).

Rates are read live rather than copied into a table, because Lumen sets and can change them,
and a copied rate drifts silently. Each model's rate is cached for six hours, so a turn of a
dozen calls costs one lookup at most. A failed lookup is cached for ten minutes as "unknown", so
an unreachable catalogue cannot add a timeout to every call. Every function here returns None
rather than raising: a cost figure must never cost the turn.

A call counts as Lumen's only when it went to Lumen's endpoint (the model's base URL), never by
model id alone. A model id like ``gpt-4o`` can exist on Lumen and on OpenAI, and the two bill
different accounts in different currencies.
"""
from __future__ import annotations

import logging
import os
import time
from threading import Lock
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

RATE_TTL_S = 6 * 3600
FAILURE_TTL_S = 600
TIMEOUT_S = 5.0

_cache: Dict[str, Tuple[float, Optional[Dict[str, float]]]] = {}
_lock = Lock()


def _base_url() -> str:
    try:
        from agent_runtime.executor_factory import LUMEN_DEFAULT_URL
    except Exception:  # noqa: BLE001
        LUMEN_DEFAULT_URL = "https://lumen.ncsa.illinois.edu/v1"
    return (os.getenv("LUMEN_URL") or LUMEN_DEFAULT_URL).rstrip("/")


def is_lumen(base_url: Optional[str]) -> bool:
    """Whether a model's base URL is Lumen's endpoint (same host as LUMEN_URL or the default)."""
    if not base_url:
        return False
    try:
        return urlparse(str(base_url)).hostname == urlparse(_base_url()).hostname
    except Exception:  # noqa: BLE001
        return False


def _get(path: str) -> Optional[Dict[str, Any]]:
    key = os.getenv("LUMEN_API_KEY")
    if not key:
        return None
    try:
        import requests

        r = requests.get(f"{_base_url()}{path}", headers={"Authorization": f"Bearer {key}"},
                         timeout=TIMEOUT_S)
        if r.status_code != 200:
            logger.info("Lumen %s answered HTTP %s", path, r.status_code)
            return None
        data = r.json()
        return data if isinstance(data, dict) else None
    except Exception as exc:  # noqa: BLE001 - a lookup failure is reported as None
        logger.info("Lumen %s lookup failed: %s", path, type(exc).__name__)
        return None


def coin_rate(model: Optional[str]) -> Optional[Dict[str, float]]:
    """``{"input": coins/1M, "output": coins/1M}`` for a Lumen model, or None if unknown."""
    name = str(model or "").strip()
    if not name:
        return None
    now = time.monotonic()
    with _lock:
        hit = _cache.get(name)
        if hit and now < hit[0]:
            return hit[1]
    data = _get(f"/models/{name}") or {}
    rate: Optional[Dict[str, float]] = None
    try:
        if data.get("input_cost_per_million") is not None and data.get("output_cost_per_million") is not None:
            rate = {"input": float(data["input_cost_per_million"]),
                    "output": float(data["output_cost_per_million"])}
    except (TypeError, ValueError):
        rate = None
    with _lock:
        _cache[name] = (now + (RATE_TTL_S if rate else FAILURE_TTL_S), rate)
    return rate


def call_coins(model: Optional[str], input_tokens: Any, output_tokens: Any) -> Optional[float]:
    """Coins for one Lumen call, by Lumen's published formula, or None if the rate is unknown."""
    rate = coin_rate(model)
    if not rate:
        return None
    try:
        tin = max(0, int(input_tokens or 0))
        tout = max(0, int(output_tokens or 0))
    except (TypeError, ValueError):
        return None
    return round(tin / 1e6 * rate["input"] + tout / 1e6 * rate["output"], 8)


def coins_available() -> Optional[float]:
    """The Lumen account's remaining coins, or None if unknown. ``-2`` means unlimited.

    The balance belongs to the ACCOUNT that owns the key, so it also falls with that account's
    other keys and its browser chat."""
    data = _get("/usage")
    if not data or data.get("coins_available") is None:
        return None
    try:
        return float(data["coins_available"])
    except (TypeError, ValueError):
        return None


def _reset_cache() -> None:
    """For tests."""
    with _lock:
        _cache.clear()


__all__ = ["is_lumen", "coin_rate", "call_coins", "coins_available", "RATE_TTL_S"]
