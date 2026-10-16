"""Each signed-in user may spend at most 3 Lumen coins and, separately, $3 of OpenAI per day.

The user's rule (2026-10-16). The two currencies are counted apart, because they are different
money: hitting either limit refuses the user's next turn until the day resets at midnight US
Central (``America/Chicago``). A turn already running finishes even if it crosses the limit,
since its cost is known only when it ends; the refusal applies from the next turn.

Only signed-in users are limited: a caller identified by nobody (the service key, dev and
local modes, the harness) has no one to charge. What a turn cost comes from its ``usage``
record (``model_pricing.summarize``), so the limit counts exactly what the turn reported.

The ledger is one small JSON file per day under the agent's file store, keyed by a digest of
the user id, not the id itself, and written atomically under a lock. The deployment runs one
gunicorn worker, so a process lock is the whole story. Files older than a month are pruned.

Settings: ``AGENT_DAILY_COIN_LIMIT`` (3) and ``AGENT_DAILY_USD_LIMIT`` (3); a value of 0 or
less disables that limit. ``AGENT_USAGE_TZ`` (America/Chicago) sets the day boundary.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

KEEP_DAYS = 31
_lock = Lock()


def _limit(name: str, default: float) -> Optional[float]:
    try:
        value = float(os.getenv(name) or default)
    except (TypeError, ValueError):
        value = default
    return value if value > 0 else None


def coin_limit() -> Optional[float]:
    return _limit("AGENT_DAILY_COIN_LIMIT", 3.0)


def usd_limit() -> Optional[float]:
    return _limit("AGENT_DAILY_USD_LIMIT", 3.0)


def _tz():
    from zoneinfo import ZoneInfo

    try:
        return ZoneInfo(os.getenv("AGENT_USAGE_TZ") or "America/Chicago")
    except Exception:  # noqa: BLE001 - an unknown zone falls back to the agreed one
        return ZoneInfo("America/Chicago")


def today(now: Optional[datetime] = None) -> date:
    return (now or datetime.now(_tz())).astimezone(_tz()).date()


def resets_at(now: Optional[datetime] = None) -> str:
    """The next midnight in the limit's time zone, as ISO 8601 with its offset."""
    tz = _tz()
    d = today(now) + timedelta(days=1)
    return datetime(d.year, d.month, d.day, tzinfo=tz).isoformat()


def _key(owner: str) -> str:
    return hashlib.sha256(owner.encode("utf-8")).hexdigest()[:32]


def _dir() -> Path:
    from agent_runtime.file_store import storage_root

    return Path(storage_root()) / "usage_ledger"


def _path(day: date) -> Path:
    return _dir() / f"{day.isoformat()}.json"


def _read(day: date) -> Dict[str, Dict[str, Any]]:
    try:
        data = json.loads(_path(day).read_text())
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:  # noqa: BLE001 - an unreadable ledger is logged, never fatal
        logger.warning("usage ledger for %s unreadable", day, exc_info=True)
        return {}


def spent(owner: Optional[str], now: Optional[datetime] = None) -> Dict[str, float]:
    """What this user has spent today: ``{"coins", "usd", "turns"}`` (zeros if nothing)."""
    row = _read(today(now)).get(_key(owner)) if owner else None
    row = row or {}
    return {"coins": float(row.get("coins") or 0.0), "usd": float(row.get("usd") or 0.0),
            "turns": int(row.get("turns") or 0)}


def record(owner: Optional[str], usage: Optional[Mapping[str, Any]],
           now: Optional[datetime] = None) -> Optional[Dict[str, float]]:
    """Add one turn's cost to the user's day. Returns the day's new totals, or None."""
    if not owner or not isinstance(usage, Mapping):
        return None
    coins = float(usage.get("cost_coins") or 0.0)
    usd = float(usage.get("cost_usd") or 0.0)
    day = today(now)
    try:
        with _lock:
            data = _read(day)
            row = data.setdefault(_key(owner), {"coins": 0.0, "usd": 0.0, "turns": 0})
            row["coins"] = round(float(row.get("coins") or 0.0) + coins, 8)
            row["usd"] = round(float(row.get("usd") or 0.0) + usd, 8)
            row["turns"] = int(row.get("turns") or 0) + 1
            path = _path(day)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data))
            tmp.replace(path)
            _prune(day)
            return {"coins": row["coins"], "usd": row["usd"], "turns": row["turns"]}
    except Exception:  # noqa: BLE001 - accounting must never cost the turn
        logger.warning("could not record usage for a turn", exc_info=True)
        return None


def _prune(day: date) -> None:
    cutoff = day - timedelta(days=KEEP_DAYS)
    try:
        for f in _dir().glob("*.json"):
            try:
                if date.fromisoformat(f.stem) < cutoff:
                    f.unlink()
            except ValueError:
                continue
    except Exception:  # noqa: BLE001
        pass


def refusal(owner: Optional[str], now: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
    """Why this user may not start a turn now, or None if they may."""
    if not owner:
        return None
    used = spent(owner, now)
    over = []
    if coin_limit() is not None and used["coins"] >= coin_limit():
        over.append(f"{used['coins']:.2f} of {coin_limit():g} Lumen coins")
    if usd_limit() is not None and used["usd"] >= usd_limit():
        over.append(f"${used['usd']:.2f} of ${usd_limit():g} of OpenAI usage")
    if not over:
        return None
    when = resets_at(now)
    return {
        "error": ("You have reached today's usage limit ("
                  + " and ".join(over) + "). It resets at midnight US Central time."),
        "reason": "daily_limit",
        "spent": used,
        "limits": {"coins": coin_limit(), "usd": usd_limit()},
        "resets_at": when,
    }


__all__ = ["coin_limit", "usd_limit", "spent", "record", "refusal", "resets_at", "today"]
