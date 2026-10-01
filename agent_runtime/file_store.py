from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

DEFAULT_STORAGE_DIR = "agent_chat_files"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_configured_root(value: str) -> Path:
    configured = Path(value).expanduser()
    if configured.is_absolute():
        return configured.resolve()
    return (_repo_root() / configured).resolve()


def storage_root() -> Path:
    configured = os.getenv("AGENT_FILE_STORAGE_ROOT")
    if configured:
        root = _resolve_configured_root(configured)
        root.mkdir(parents=True, exist_ok=True)
        return root

    candidates = [(_repo_root() / DEFAULT_STORAGE_DIR).resolve()]

    last_error: Optional[Exception] = None
    for root in candidates:
        try:
            root.mkdir(parents=True, exist_ok=True)
            return root
        except OSError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error

    raise RuntimeError("Unable to determine agent file storage root.")


def _uploads_dir() -> Path:
    path = storage_root() / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _outputs_dir() -> Path:
    path = storage_root() / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _metadata_dir() -> Path:
    path = storage_root() / "metadata"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _metadata_path(file_id: str) -> Path:
    return _metadata_dir() / f"{file_id}.json"


def _build_download_url(file_id: str) -> str:
    return f"/agent/files/{file_id}/download"


def _public_base_url() -> str:
    """Optional absolute origin for download URLs (e.g. ``http://149.165.147.219:3500``).

    Unset (default): ``download_url`` stays HOST-RELATIVE and clients resolve it against the
    origin they call — robust across port mappings and proxies. Set ``AGENT_PUBLIC_BASE_URL``
    when clients can't do that resolution and need ready-to-use absolute URLs.
    """
    base = (os.getenv("AGENT_PUBLIC_BASE_URL") or "").strip().rstrip("/")
    return base if base.lower().startswith(("http://", "https://")) else ""


def _with_public_url(record: Dict[str, Any]) -> Dict[str, Any]:
    """Absolutize the record's download_url for EMISSION only. Persisted metadata always keeps
    the host-relative path, so records stay valid if the public base URL changes later."""
    base = _public_base_url()
    url = record.get("download_url")
    if base and isinstance(url, str) and url.startswith("/"):
        record = dict(record)
        record["download_url"] = f"{base}{url}"
    return record


# ---------------------------------------------------------------------------
# Retention / TTL sweep
# ---------------------------------------------------------------------------
# The store has no other GC: uploads + generated outputs accumulate forever. An OPT-IN TTL
# sweep deletes data files (and their metadata) older than AGENT_FILE_RETENTION_DAYS. It is
# DISABLED by default (unset / <= 0) so existing deployments keep their current behavior until
# an operator opts in — deleting user files is irreversible. Age is measured from each file's
# last-modified time. The sweep runs opportunistically (throttled) when files are created, so
# no background scheduler is needed; it can also be invoked directly (cron / CLI).

logger = logging.getLogger(__name__)

_SECONDS_PER_DAY = 86400.0
_DEFAULT_SWEEP_INTERVAL_SECONDS = 3600.0
_LAST_SWEEP = {"t": 0.0}
_SWEEP_LOCK = threading.Lock()


def _retention_seconds() -> Optional[float]:
    """TTL in seconds from AGENT_FILE_RETENTION_DAYS, or None when retention is disabled
    (unset, non-numeric, or <= 0). None => keep files forever (the default)."""
    raw = os.getenv("AGENT_FILE_RETENTION_DAYS")
    if raw is None:
        return None
    try:
        days = float(raw)
    except (TypeError, ValueError):
        return None
    return days * _SECONDS_PER_DAY if days > 0 else None


def _sweep_interval_seconds() -> float:
    try:
        return max(60.0, float(os.getenv("AGENT_FILE_SWEEP_INTERVAL_SECONDS",
                                         str(_DEFAULT_SWEEP_INTERVAL_SECONDS))))
    except (TypeError, ValueError):
        return _DEFAULT_SWEEP_INTERVAL_SECONDS


def sweep_expired_files(max_age_days: Optional[float] = None, *, now: Optional[float] = None) -> Dict[str, int]:
    """Delete stored files (uploads + outputs) and their metadata older than the TTL.

    Age is each file's mtime. Deleting an expired data file also removes its ``metadata/<id>.json``
    sidecar; an expired metadata json whose data file is already gone is removed too (a present,
    non-expired data file is never orphaned). TTL comes from *max_age_days* else
    AGENT_FILE_RETENTION_DAYS; <= 0 / unset disables the sweep (no-op). Never raises — a sweep
    must not break an upload or a request. Returns counts: {scanned, deleted, freed_bytes, errors}.
    """
    ttl = (max_age_days * _SECONDS_PER_DAY) if (max_age_days and max_age_days > 0) else _retention_seconds()
    result = {"scanned": 0, "deleted": 0, "freed_bytes": 0, "errors": 0}
    if not ttl:
        return result
    cutoff = (time.time() if now is None else now) - ttl
    try:
        root = storage_root()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("File sweep skipped: storage root unavailable (%s)", exc)
        return result

    def _expired(p: Path) -> bool:
        try:
            return p.stat().st_mtime < cutoff
        except OSError:
            return False

    # 1) expired DATA files (+ their metadata sidecar)
    for sub in ("uploads", "outputs"):
        d = root / sub
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if not p.is_file():
                continue
            result["scanned"] += 1
            if not _expired(p):
                continue
            try:
                size = p.stat().st_size
                p.unlink()
                result["deleted"] += 1
                result["freed_bytes"] += size
            except OSError:
                result["errors"] += 1
                continue
            file_id = p.name.split("__", 1)[0]
            meta = root / "metadata" / f"{file_id}.json"
            try:
                if meta.is_file():
                    result["freed_bytes"] += meta.stat().st_size
                    meta.unlink()
                    result["deleted"] += 1
            except OSError:
                pass

    # 2) expired ORPHAN metadata (its data file is already gone) — never orphan a present file
    meta_dir = root / "metadata"
    if meta_dir.is_dir():
        for mp in meta_dir.glob("*.json"):
            if not _expired(mp):
                continue
            file_id = mp.stem
            data_present = any(
                next((root / sub).glob(f"{file_id}__*"), None) is not None
                for sub in ("uploads", "outputs")
            )
            if data_present:
                continue
            try:
                mp.unlink()
                result["deleted"] += 1
            except OSError:
                result["errors"] += 1

    if result["deleted"]:
        logger.info("File sweep removed %d item(s), freed %d bytes from %s",
                    result["deleted"], result["freed_bytes"], root)
    return result


def maybe_sweep_expired_files() -> None:
    """Opportunistic, throttled sweep — at most once per AGENT_FILE_SWEEP_INTERVAL_SECONDS per
    process. Called from the file-creation paths so retention is enforced without a scheduler.
    No-op when retention is disabled. Never raises."""
    if _retention_seconds() is None:
        return
    nowt = time.time()
    with _SWEEP_LOCK:
        if nowt - _LAST_SWEEP["t"] < _sweep_interval_seconds():
            return
        _LAST_SWEEP["t"] = nowt
    try:
        sweep_expired_files()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Opportunistic file sweep failed: %s", exc)


def get_file_record(file_id: str) -> Optional[Dict[str, Any]]:
    if not file_id:
        return None
    meta_path = _metadata_path(str(file_id).strip())
    if not meta_path.exists():
        return None
    return _with_public_url(json.loads(meta_path.read_text(encoding="utf-8")))


def require_file_record(file_id: str) -> Dict[str, Any]:
    record = get_file_record(file_id)
    if not record:
        raise ValueError(f"unknown file_id: {file_id}")
    return record


def resolve_file_id(file_id: str) -> Path:
    record = require_file_record(file_id)
    path = _record_path(record)
    if not path.exists():
        raise ValueError(
            f"file for file_id does not exist: {file_id}; "
            f"expected under storage root: {storage_root()}"
        )
    return path


# Which conversation is writing. A ContextVar because the store is called from deep inside tool
# code that has no idea what a session is — the same shape the streaming trace state uses, set
# once per request at the edge. JWT will replace WHERE this value comes from, not what it does.
_SESSION: ContextVar[Optional[str]] = ContextVar("agent_file_store_session", default=None)
# Sentinel: `session=None` means "search every conversation", which is different from
# "the caller did not say", and a default of None could not tell them apart.
_UNSET = object()


def set_session(session_id: Optional[str]) -> Any:
    """Bind the conversation for this request. Returns a token for ContextVar.reset."""
    return _SESSION.set(str(session_id).strip() or None if session_id else None)


def reset_session(token: Any) -> None:
    try:
        _SESSION.reset(token)
    except Exception:  # noqa: BLE001 - a stale token must not break a response
        pass


def current_session() -> Optional[str]:
    return _SESSION.get()


# ---------------------------------------------------------------------------
# WHOSE file this is (token mode), as distinct from WHICH CONVERSATION wrote it
# ---------------------------------------------------------------------------
# Two independent axes, and conflating them loses one of them: a user has many conversations,
# and in dev/demo there is no user at all. `session` keeps answering "which conversation", and
# `owner_id` answers "whose". Outside token mode `owner_id` is None on every new record and
# nothing changes — scoping stays exactly the per-conversation behaviour it is today.
#
# The value comes from the identity ContextVar rather than a second one of our own, so there is
# one place a caller is established and one place it can be wrong.


def current_owner() -> Optional[str]:
    """The signed-in user's id, or None when this deployment does not identify anyone."""
    try:
        from agent_runtime import identity
    except Exception:  # noqa: BLE001 - identity is optional; the store predates it
        return None
    return identity.current_user_id()


def record_owner(record: Dict[str, Any]) -> Optional[str]:
    value = (record or {}).get("owner_id")
    return str(value).strip() or None if value else None


def may_read(record: Dict[str, Any], *, allow_unowned: bool = True) -> bool:
    """Whether the CURRENT caller may read this record.

    With no caller — dev, demo, or a service request — this is always True and the store behaves
    exactly as it did before ownership existed.

    ``allow_unowned`` decides the one genuinely awkward case: 1,325 records predate ownership
    and cannot be attributed to anyone. Treating them as readable keeps every existing reuse
    working; treating them as denied is the safe reading for a browser download. The two callers
    want different answers, so neither is hardcoded here.
    """
    caller = current_owner()
    if not caller:
        return True
    owner = record_owner(record)
    if owner is None:
        return allow_unowned
    return owner == caller


def find_files(name: Optional[str] = None, *, suffix: Optional[str] = None,
               kind: Optional[str] = None, limit: int = 20,
               session: Any = _UNSET,
               include_unowned: bool = True) -> List[Dict[str, Any]]:
    """Stored file records, newest first, optionally narrowed by name / extension / kind.

    A ``file_id`` has been the store's only handle, and a ``file_id`` is exactly what a later
    turn does not have: an answer surfaces an artifact by FILENAME with a download link, and the
    id survives only in the process-local action ledger, which a restart discards. So a turn
    asked to "predict from the package you saved for Champaign" had no way to reach a file it
    could plainly see. This is the lookup that closes that gap.

    ``name`` matches as a case-insensitive substring, so a filename remembered imprecisely still
    finds its file. Ordering is by mtime, because records carry no timestamp of their own; ties
    break on file_id so the result is deterministic.

    ``include_unowned=False`` narrows the result to records this conversation actually wrote.
    The default keeps unstamped records visible, which is right for REUSE — a saved embedding
    package is worth offering whoever asks — and wrong for a question about this conversation,
    where "the files you made" must not be answered with 1,325 files from everyone else.

    There is no index — records are one json file each — so this scans the metadata directory,
    which is the same scan ``create_output_file_from_path`` already does to honour ``overwrite``.
    """
    needle = (name or "").strip().lower()
    want_suffix = (suffix or "").strip().lower()
    # THIS conversation's files by default. A record written before sessions existed has no
    # `session` and stays visible to everyone: the store holds 1,325 of them and hiding the lot
    # would break every reuse the demo depends on. Pass session=None to search across all.
    want_session = current_session() if session is _UNSET else session
    out: List[Dict[str, Any]] = []
    for meta_path in _metadata_dir().glob("*.json"):
        try:
            record = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - a half-written record must not break the lookup
            continue
        filename = str(record.get("filename") or "")
        if kind and str(record.get("kind") or "") != kind:
            continue
        if want_suffix and not filename.lower().endswith(want_suffix):
            continue
        if needle and needle not in filename.lower():
            continue
        owner = record.get("session")
        if want_session and owner and owner != want_session:
            continue
        if want_session and not owner and not include_unowned:
            continue
        # Another user's file is never a candidate, in any conversation. Unowned records stay
        # visible: they are the legacy reuse pool, they were created by a deployment that
        # identified nobody, and hiding them would break the reuse this lookup exists for.
        if not may_read(record, allow_unowned=True):
            continue
        try:
            path = _record_path(record)
            if not path.exists():
                continue
            stamped = {**record, "_mtime": path.stat().st_mtime}
        except Exception:  # noqa: BLE001
            continue
        out.append(stamped)
    out.sort(key=lambda r: (r.get("_mtime") or 0, str(r.get("file_id"))), reverse=True)
    return [_with_public_url({k: v for k, v in r.items() if k != "_mtime"})
            for r in out[:max(1, int(limit))]]


def resolve_file_ref(ref: str, *, suffix: Optional[str] = None
                     ) -> Tuple[Path, Dict[str, Any], List[Dict[str, Any]]]:
    """A path for a file named either by its ``file_id`` or by its filename.

    Returns the path, the record it resolved to, and any OTHER records that matched the same
    name. The caller is expected to say which one it used whenever that list is non-empty:
    "the Champaign package" can legitimately name several files, and silently taking the newest
    would be a guess reported as a fact.
    """
    text = str(ref or "").strip()
    if not text:
        raise ValueError("no file_id or filename given")
    try:
        return resolve_file_id(text), require_file_record(text), []
    except Exception:  # noqa: BLE001 - not an id, so try it as a name
        pass
    # A high limit on purpose: the CALLER reports how many matched, and find_files' default page
    # size of 20 made that count the page size rather than the truth — "20 saved packages match"
    # where 32 did. The caller still shows only a handful.
    matches = find_files(name=text, suffix=suffix, limit=500)
    if not matches:
        raise ValueError(f"no stored file matches {text!r}")
    first = matches[0]
    return resolve_file_id(str(first["file_id"])), first, matches[1:]


def _write_record(record: Dict[str, Any]) -> Dict[str, Any]:
    _metadata_path(record["file_id"]).write_text(json.dumps(record, ensure_ascii=True, indent=2), encoding="utf-8")
    return record


def _record_path(record: Dict[str, Any]) -> Path:
    root = storage_root()
    relative_path = record.get("relative_path")
    if relative_path:
        return (root / str(relative_path)).resolve()

    stored_path = record.get("path")
    if stored_path:
        candidate = Path(str(stored_path)).expanduser()
        if candidate.is_absolute() and candidate.exists():
            return candidate.resolve()
        if not candidate.is_absolute():
            relative_candidate = (root / candidate).resolve()
            if relative_candidate.exists():
                return relative_candidate

    filename = record.get("filename")
    file_id = record.get("file_id")
    kind = record.get("kind") or "upload"
    if filename and file_id:
        folder = "outputs" if kind == "output" else "uploads"
        return (root / folder / f"{file_id}__{filename}").resolve()

    if stored_path:
        candidate = Path(str(stored_path)).expanduser()
        return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    raise ValueError(f"file record missing path for file_id: {record.get('file_id')}")


def save_uploaded_file(file_storage: FileStorage) -> Dict[str, Any]:
    maybe_sweep_expired_files()
    original_name = secure_filename(file_storage.filename or "upload.bin") or "upload.bin"
    file_id = f"file_{uuid4().hex[:12]}"
    relative_path = Path("uploads") / f"{file_id}__{original_name}"
    target = storage_root() / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    file_storage.save(target)
    record = {
        "file_id": file_id,
        "filename": original_name,
        "kind": "upload",
        "session": current_session(),
        "owner_id": current_owner(),
        "path": str(relative_path),
        "relative_path": str(relative_path),
        "size_bytes": target.stat().st_size,
        "download_url": _build_download_url(file_id),
    }
    return _with_public_url(_write_record(record))


def create_output_file(filename: str, content: str, overwrite: bool = False) -> Dict[str, Any]:
    maybe_sweep_expired_files()
    safe_name = secure_filename(filename or "agent_output.txt") or "agent_output.txt"
    existing_record: Optional[Dict[str, Any]] = None

    if overwrite:
        for meta_path in _metadata_dir().glob("*.json"):
            try:
                record = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if record.get("kind") == "output" and record.get("filename") == safe_name:
                existing_record = record
                break

    if existing_record:
        file_id = str(existing_record["file_id"])
    else:
        file_id = f"file_{uuid4().hex[:12]}"

    relative_path = Path("outputs") / f"{file_id}__{safe_name}"
    target = storage_root() / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content or "", encoding="utf-8")
    record = {
        "file_id": file_id,
        "filename": safe_name,
        "kind": "output",
        # Which conversation produced this. Absent on everything written before this existed,
        # and find_files treats that absence as "visible to all" so the demo's existing 1,325
        # files stay reachable rather than vanishing.
        "session": current_session(),
        "owner_id": current_owner(),
        "path": str(relative_path),
        "relative_path": str(relative_path),
        "size_bytes": target.stat().st_size,
        "download_url": _build_download_url(file_id),
    }
    return _with_public_url(_write_record(record))


def create_output_file_from_path(
    source_path: str | Path,
    filename: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    maybe_sweep_expired_files()
    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"source file does not exist: {source}")

    safe_name = secure_filename(filename or source.name or "agent_output.bin") or "agent_output.bin"
    existing_record: Optional[Dict[str, Any]] = None

    if overwrite:
        for meta_path in _metadata_dir().glob("*.json"):
            try:
                record = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if record.get("kind") == "output" and record.get("filename") == safe_name:
                existing_record = record
                break

    if existing_record:
        file_id = str(existing_record["file_id"])
    else:
        file_id = f"file_{uuid4().hex[:12]}"

    relative_path = Path("outputs") / f"{file_id}__{safe_name}"
    target = storage_root() / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    record = {
        "file_id": file_id,
        "filename": safe_name,
        "kind": "output",
        # Which conversation produced this. Absent on everything written before this existed,
        # and find_files treats that absence as "visible to all" so the demo's existing 1,325
        # files stay reachable rather than vanishing.
        "session": current_session(),
        "owner_id": current_owner(),
        "path": str(relative_path),
        "relative_path": str(relative_path),
        "size_bytes": target.stat().st_size,
        "download_url": _build_download_url(file_id),
    }
    return _with_public_url(_write_record(record))


__all__ = [
    "create_output_file",
    "create_output_file_from_path",
    "find_files",
    "resolve_file_ref",
    "get_file_record",
    "maybe_sweep_expired_files",
    "require_file_record",
    "resolve_file_id",
    "save_uploaded_file",
    "storage_root",
    "sweep_expired_files",
]
