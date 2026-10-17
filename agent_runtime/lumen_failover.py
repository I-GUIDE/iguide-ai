"""Move the default model off NCSA Lumen while its coin pool runs low, and back when it refills.

The user's rule (2026-10-16): when 90% of the Lumen pool is used, the process default switches
to OpenAI gpt-5.6-luna; once usage is back below 80% (a refill tops the pool up), it returns to
Lumen. The gap between the two thresholds is deliberate: a single threshold would flip the
default on every turn while the balance hovered at it.

Only the PROCESS default is affected (``AGENT_LLM_PROVIDER=lumen``, a turn that names no model).
A request that picks its own model gets that model.

Lumen's API reports the remaining balance (``coins_available``) but not the pool's size, so the
size is configured: ``AGENT_LUMEN_POOL_COINS`` (100, the pool this deployment has). The balance
is read at most once a minute, and after every Lumen turn (``observe``). The current side is
kept in a small file beside the agent's other files, so a restart in the 80-90% band neither
returns to Lumen early nor leaves it early. An unknown balance (Lumen unreachable, no key)
keeps whatever side is current. An unlimited pool (``-2``) never fails over.

Settings: ``AGENT_LUMEN_FAILOVER=0`` disables it. ``AGENT_LUMEN_SWITCH_AT`` (0.90) and
``AGENT_LUMEN_RETURN_AT`` (0.80) are fractions of the pool USED.
``AGENT_LUMEN_FALLBACK_PROVIDER`` / ``AGENT_LUMEN_FALLBACK_MODEL`` default to openai and
gpt-5.6-luna.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CHECK_TTL_S = 60.0

_lock = Lock()
_checked_at = 0.0
_state: Optional[bool] = None   # None until read from the state file


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name) or default)
    except (TypeError, ValueError):
        return default


def enabled() -> bool:
    if (os.getenv("AGENT_LUMEN_FAILOVER") or "1").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return (os.getenv("AGENT_LLM_PROVIDER") or "").strip().lower() == "lumen"


def pool_coins() -> float:
    return _float_env("AGENT_LUMEN_POOL_COINS", 100.0)


def switch_at() -> float:
    return _float_env("AGENT_LUMEN_SWITCH_AT", 0.90)


def return_at() -> float:
    return _float_env("AGENT_LUMEN_RETURN_AT", 0.80)


def fallback() -> Dict[str, str]:
    return {"provider": (os.getenv("AGENT_LUMEN_FALLBACK_PROVIDER") or "openai").strip(),
            "model": (os.getenv("AGENT_LUMEN_FALLBACK_MODEL") or "gpt-5.6-luna").strip()}


def used_fraction(balance: Optional[float]) -> Optional[float]:
    """How much of the pool is used, 0..1, or None if unknown. Unlimited (-2) counts as 0."""
    if balance is None:
        return None
    if balance == -2:
        return 0.0
    pool = pool_coins()
    if pool <= 0:
        return None
    return min(1.0, max(0.0, 1.0 - float(balance) / pool))


def _state_path() -> Optional[Path]:
    try:
        from agent_runtime.file_store import storage_root

        return Path(storage_root()) / "lumen_failover.json"
    except Exception:  # noqa: BLE001
        return None


def _load() -> bool:
    global _state
    if _state is None:
        _state = False
        path = _state_path()
        try:
            if path and path.exists():
                _state = bool(json.loads(path.read_text()).get("failed_over"))
        except Exception:  # noqa: BLE001 - an unreadable state file starts on Lumen
            _state = False
    return _state


def _save(failed_over: bool, used: Optional[float]) -> None:
    path = _state_path()
    if not path:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"failed_over": failed_over, "used": used,
                                   "at": datetime.now(timezone.utc).isoformat()}))
        tmp.replace(path)
    except Exception:  # noqa: BLE001 - the in-process state still holds
        logger.warning("Could not persist the Lumen failover state", exc_info=True)


def observe(balance: Optional[float]) -> bool:
    """Apply one balance reading and return whether the default is now failed over."""
    global _state
    with _lock:
        current = _load()
        used = used_fraction(balance)
        if used is None:
            return current
        if not current and used >= switch_at():
            _state = True
            logger.warning("Lumen pool %.0f%% used (%.2f of %.0f coins left): default model "
                           "switches to %s", used * 100, balance, pool_coins(), fallback())
            _save(True, used)
        elif current and used <= return_at():
            _state = False
            logger.warning("Lumen pool back to %.0f%% used (%.2f coins left): default model "
                           "returns to Lumen", used * 100, balance)
            _save(False, used)
        return bool(_state)


def failed_over() -> bool:
    """Whether the process default should be the fallback right now. Cheap: reads Lumen's
    balance at most once a minute."""
    global _checked_at
    if not enabled():
        return False
    now = time.monotonic()
    with _lock:
        due = now - _checked_at >= CHECK_TTL_S
        if due:
            _checked_at = now
        current = _load()
    if not due:
        return current
    try:
        from agent_runtime import lumen_coins

        return observe(lumen_coins.coins_available())
    except Exception:  # noqa: BLE001 - a failed check keeps the current side
        return current


def status() -> Dict[str, Any]:
    """For logs and the turn's usage record."""
    return {"failed_over": bool(_load()), "pool_coins": pool_coins(),
            "switch_at": switch_at(), "return_at": return_at(), **{f"fallback_{k}": v
                                                                    for k, v in fallback().items()}}


def _reset() -> None:
    """For tests."""
    global _state, _checked_at
    with _lock:
        _state, _checked_at = None, 0.0


__all__ = ["enabled", "failed_over", "observe", "used_fraction", "status", "fallback"]
