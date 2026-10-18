"""Each signed-in user may spend at most 3 credits per day.

A credit is one Lumen coin or one dollar of OpenAI: the day's credits are the coins and the
dollars added together (user, 2026-10-18). They used to be two limits, 3 coins and, separately,
$3 (2026-10-16), but the default model moves between Lumen and OpenAI by itself when the Lumen
pool runs low (lumen_failover), so a user could not know which budget a turn would spend, and
two budgets gave them 6 a day. One budget in one unit is what the user sees. Reaching it refuses
the user's next turn until the day resets at midnight US Central (``America/Chicago``). A turn
already running finishes even if it crosses the limit, since its cost is known only when it ends;
the refusal applies from the next turn.

Only signed-in users are limited: a caller identified by nobody (the service key, dev and
local modes, the harness) has no one to charge. What a turn cost comes from its ``usage``
record (``model_pricing.summarize``), so the limit counts exactly what the turn reported.

The ledger is one small JSON file per day under the agent's file store, keyed by a digest of
the user id, not the id itself, and written atomically under a lock. The deployment runs one
gunicorn worker, so a process lock is the whole story. Files older than a month are pruned.

Settings: ``AGENT_DAILY_CREDIT_LIMIT`` (3); a value of 0 or less disables the limit.
``AGENT_USAGE_TZ`` (America/Chicago) sets the day boundary. The ledger still records coins and
dollars apart, for the admin view.
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


def credit_limit() -> Optional[float]:
    return _limit("AGENT_DAILY_CREDIT_LIMIT", 3.0)


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
    """What this user has spent today: ``{"credits", "coins", "usd", "turns"}`` (zeros if
    nothing). ``credits`` is coins plus dollars."""
    row = _read(today(now)).get(_key(owner)) if owner else None
    return _totals(row or {})


def _totals(row: Mapping[str, Any]) -> Dict[str, Any]:
    coins, usd = float(row.get("coins") or 0.0), float(row.get("usd") or 0.0)
    return {"credits": round(coins + usd, 8), "coins": coins, "usd": usd,
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
            return _totals(row)
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
# An admin can give one user a different daily credit limit, or none, and can clear that user's
# spending for the day. Overrides live beside the day files, keyed by the same digest. A value is
# a number >= 0 (0 blocks the user), or UNLIMITED; no override follows the deployment default.
# A row written before credits (2026-10-18) holds "coins"/"usd"; its "coins" is read as credits.

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
    """The admin-set limit for this user, in the API's terms: ``{"credits": n | "unlimited"}``,
    or ``{}`` when the default applies."""
    if not owner:
        return {}
    row = _read_overrides().get(_key(owner)) or {}
    out: Dict[str, Any] = {}
    value = row.get("credits", row.get("coins"))
    if value is not None:
        out["credits"] = UNLIMITED if float(value) == _UNLIMITED_STORED else float(value)
    if out and row.get("set_at"):
        out["set_at"] = row["set_at"]
    return out


def limits_for(owner: Optional[str]) -> Dict[str, Optional[float]]:
    """The limit that applies to this user now: ``{"credits": n}``, None meaning no limit."""
    value = override_for(owner).get("credits")
    if value is None:
        return {"credits": credit_limit()}
    return {"credits": None if value == UNLIMITED else float(value)}


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


def set_override(owner: str, *, credits: Any, now: Optional[datetime] = None) -> Dict[str, Any]:
    """Set one user's daily credit limit, or clear it (None) to return to the default."""
    stamp = (now or datetime.now(_tz())).isoformat()
    with _lock:
        data = _read_overrides()
        if credits is None:
            data.pop(_key(owner), None)
        else:
            data[_key(owner)] = {"credits": _UNLIMITED_STORED if credits == UNLIMITED
                                 else float(credits), "set_at": stamp}
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
            out.append({"day": day.isoformat(), **_totals(row)})
    return out


def refusal(owner: Optional[str], now: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
    """Why this user may not start a turn now, or None if they may."""
    if not owner:
        return None
    used = spent(owner, now)
    limits = limits_for(owner)
    if limits["credits"] is None or used["credits"] < limits["credits"]:
        return None
    when = resets_at(now)
    # In the user's terms: credits, not the provider that was spent (user, 2026-10-18).
    return {
        "error": (f"You have reached today's usage limit ({used['credits']:.2f} of "
                  f"{limits['credits']:g} credits). It resets at midnight US Central time."),
        "reason": "daily_limit",
        "spent": used,
        "limits": limits,
        "resets_at": when,
    }


__all__ = ["credit_limit", "spent", "record", "refusal", "resets_at", "today",
           "UNLIMITED", "override_for", "limits_for", "parse_limit", "set_override",
           "reset_today", "history"]
