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


# --- per-user overrides, set by an admin ------------------------------------------------------
#
# An admin can give one user a different daily limit in either currency, or none, and can clear
# that user's spending for the day. Overrides live beside the day files, keyed by the same
# digest. A value is a number >= 0 (0 blocks that currency), or UNLIMITED; a currency with no
# override follows the deployment default.

UNLIMITED = "unlimited"
_UNLIMITED_STORED = -1.0


def _overrides_path() -> Path:
    return _dir() / "overrides.json"


def _read_overrides() -> Dict[str, Dict[str, Any]]:
    try:
        data = json.loads(_overrides_path().read_text())
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:  # noqa: BLE001
        logger.warning("usage overrides unreadable", exc_info=True)
        return {}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


def override_for(owner: Optional[str]) -> Dict[str, Any]:
    """The admin-set limits for this user, in the API's terms: ``{"coins": n | "unlimited"}``,
    only the currencies that are overridden."""
    if not owner:
        return {}
    row = _read_overrides().get(_key(owner)) or {}
    out: Dict[str, Any] = {}
    for cur in ("coins", "usd"):
        if cur in row:
            out[cur] = UNLIMITED if float(row[cur]) == _UNLIMITED_STORED else float(row[cur])
    if row.get("set_at"):
        out["set_at"] = row["set_at"]
    return out


def limits_for(owner: Optional[str]) -> Dict[str, Optional[float]]:
    """The limits that apply to this user now. None means no limit in that currency."""
    limits: Dict[str, Optional[float]] = {"coins": coin_limit(), "usd": usd_limit()}
    for cur, value in override_for(owner).items():
        if cur in limits:
            limits[cur] = None if value == UNLIMITED else float(value)
    return limits


def parse_limit(value: Any) -> Any:
    """An admin's limit value: None (back to the default), "unlimited", or a number >= 0.
    Raises ValueError for anything else."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == UNLIMITED:
        return UNLIMITED
    if isinstance(value, bool):
        raise ValueError("a limit is a number, null or \"unlimited\"")
    number = float(value)
    if number != number or number < 0 or number == float("inf"):
        raise ValueError("a limit must be a finite number >= 0, null or \"unlimited\"")
    return number


def set_override(owner: str, *, coins: Any = ..., usd: Any = ...,
                 now: Optional[datetime] = None) -> Dict[str, Any]:
    """Set (or clear, with None) one user's limits. ``...`` leaves a currency as it is."""
    stamp = (now or datetime.now(_tz())).isoformat()
    with _lock:
        data = _read_overrides()
        row = dict(data.get(_key(owner)) or {})
        for cur, value in (("coins", coins), ("usd", usd)):
            if value is ...:
                continue
            if value is None:
                row.pop(cur, None)
            else:
                row[cur] = _UNLIMITED_STORED if value == UNLIMITED else float(value)
        if any(k in row for k in ("coins", "usd")):
            row["set_at"] = stamp
            data[_key(owner)] = row
        else:
            data.pop(_key(owner), None)
        _write_json(_overrides_path(), data)
    return override_for(owner)


def reset_today(owner: str, now: Optional[datetime] = None) -> None:
    """Forget what this user has spent today, so their next turn is allowed again."""
    day = today(now)
    with _lock:
        data = _read(day)
        if data.pop(_key(owner), None) is not None:
            _write_json(_path(day), data)


def history(owner: Optional[str], days: int = 7,
            now: Optional[datetime] = None) -> list:
    """This user's spending for the last ``days`` days, newest first, days with none left out."""
    if not owner:
        return []
    out = []
    d = today(now)
    for i in range(max(1, days)):
        day = d - timedelta(days=i)
        row = _read(day).get(_key(owner))
        if row:
            out.append({"day": day.isoformat(), "coins": float(row.get("coins") or 0.0),
                        "usd": float(row.get("usd") or 0.0), "turns": int(row.get("turns") or 0)})
    return out


def refusal(owner: Optional[str], now: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
    """Why this user may not start a turn now, or None if they may."""
    if not owner:
        return None
    used = spent(owner, now)
    limits = limits_for(owner)
    over = []
    # Said to the user, so in the user's terms: credits, not the provider or its coins (user,
    # 2026-10-18: "(x of 3 credits)", no Lumen). The logs and the admin endpoint keep the names.
    if limits["coins"] is not None and used["coins"] >= limits["coins"]:
        over.append(f"{used['coins']:.2f} of {limits['coins']:g} credits")
    if limits["usd"] is not None and used["usd"] >= limits["usd"]:
        over.append(f"${used['usd']:.2f} of ${limits['usd']:g}")
    if not over:
        return None
    when = resets_at(now)
    return {
        "error": ("You have reached today's usage limit ("
                  + " and ".join(over) + "). It resets at midnight US Central time."),
        "reason": "daily_limit",
        "spent": used,
        "limits": limits,
        "resets_at": when,
    }


__all__ = ["coin_limit", "usd_limit", "spent", "record", "refusal", "resets_at", "today",
           "UNLIMITED", "override_for", "limits_for", "parse_limit", "set_override",
           "reset_today", "history"]
