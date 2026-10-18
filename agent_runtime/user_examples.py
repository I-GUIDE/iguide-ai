"""Examples users author for the map UI's Spatial Analysis tab (architecture stage 73).

The built-in examples are a generated catalogue served as static files
(``map-ui-prototype/src/demoCatalog.json``). These are the other kind: a signed-in user writes
one (title, description, category, tags, data files, ordered steps), it is stored here under
their account, and it follows them to another browser. It is private until they share it.

Three ways to see one, and only these:

- **owner**: the user who created it. Only the owner edits or deletes it, or makes a share link.
- **link**: the owner made a share link. Another signed-in user who opens the link gets the
  example in their "Shared with me" list. That entry remembers the token it was opened with, so
  revoking the link (which drops the token) takes the example away from everyone it reached.
- **published**: an admin (role at or below ``AGENT_EXAMPLES_PUBLISH_ROLE``, default 2 =
  Admin) published it, and it is listed for every caller, beside the built-in examples.

Anything else answers as if the id did not exist (``NotFound``, a 404), as files and
conversations do: a 403 would confirm the id and make the routes an enumeration oracle.

Where it lives: ``<AGENT_FILE_STORAGE_ROOT>/examples/``, on the same volume as the file store,
so it survives a rebuild and works in every mode, local included::

    defs/<example_id>.json        the definition
    files/<example_id>/<name>     its data files
    shared/<user digest>.json     one user's "Shared with me" list

The data files are NOT file-store records. Running an example uploads them into the runner's
own conversation through the ordinary upload route, so the file store's owner and conversation
scoping is untouched: a shared example never lets anyone read a file-store record they could
not read before.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, Iterator, List, Optional, Tuple

from werkzeug.utils import secure_filename

try:  # POSIX only; on Windows the in-process lock is all there is
    import fcntl
except ImportError:  # pragma: no cover - not exercised on the deployment
    fcntl = None  # type: ignore[assignment]

from agent_runtime.file_store import storage_root

# --- limits -------------------------------------------------------------------------------
# Every limit is an env var so a deployment can move it without a release. The defaults are
# sized from the built-in catalogue: its largest case is 5 files and 4 steps, and its largest
# data set is a few MB.

MAX_TITLE = 120
MAX_DESCRIPTION = 2000
MAX_CATEGORY = 60
MAX_TAGS = 12
MAX_TAG = 40
MAX_LABEL = 120
MAX_PROMPT = 4000
CHUNK = 1024 * 1024


def _int_env(name: str, default: int) -> int:
    try:
        value = int(str(os.getenv(name) or "").strip() or default)
    except ValueError:
        return default
    return value if value > 0 else default


def max_examples_per_user() -> int:
    return _int_env("AGENT_EXAMPLES_MAX_PER_USER", 50)


def max_steps() -> int:
    return _int_env("AGENT_EXAMPLES_MAX_STEPS", 20)


def max_files() -> int:
    return _int_env("AGENT_EXAMPLES_MAX_FILES", 20)


def max_bytes() -> int:
    """The total size of one example's data files."""
    return _int_env("AGENT_EXAMPLES_MAX_MB", 100) * 1024 * 1024


def publish_role() -> int:
    """The role an admin needs to publish (lower is more privileged, as the platform counts)."""
    return _int_env("AGENT_EXAMPLES_PUBLISH_ROLE", 2)


def limits() -> Dict[str, int]:
    return {"maxExamples": max_examples_per_user(), "maxSteps": max_steps(),
            "maxFiles": max_files(), "maxBytes": max_bytes()}


# --- errors -------------------------------------------------------------------------------

class ExampleError(Exception):
    status = 400


class Invalid(ExampleError):
    status = 400


class NotFound(ExampleError):
    status = 404


class Forbidden(ExampleError):
    """Only for what the CALLER may do (publish), never for whether an example exists."""
    status = 403


class SignInRequired(Forbidden):
    """The deployment identified nobody (dev, demo, a service caller): nothing to own with."""


class LimitReached(ExampleError):
    status = 413


# --- paths and locking -----------------------------------------------------------------------

_ID_RE = re.compile(r"^ex_[0-9a-f]{16}$")
_LOCK = threading.Lock()


def _root() -> Path:
    path = storage_root() / "examples"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _defs_dir() -> Path:
    path = _root() / "defs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _files_dir(example_id: str) -> Path:
    return _root() / "files" / example_id


def _shared_path(user_id: str) -> Path:
    path = _root() / "shared"
    path.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()[:32]
    return path / f"{digest}.json"


def _def_path(example_id: str) -> Path:
    if not _ID_RE.match(str(example_id or "")):
        raise NotFound("unknown example")
    return _defs_dir() / f"{example_id}.json"


@contextmanager
def _locked() -> Iterator[None]:
    """One writer at a time: threads by the lock, gunicorn workers by flock on a lock file."""
    with _LOCK:
        if fcntl is None:
            yield
            return
        with open(_root() / ".lock", "a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _write_json(path: Path, value: Any) -> None:
    """Atomic: a reader sees the old file or the new one, never half of one."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=True, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _read_def(example_id: str) -> Dict[str, Any]:
    path = _def_path(example_id)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise NotFound("unknown example") from None
    except ValueError:
        raise NotFound("unknown example") from None


def _all_defs() -> List[Dict[str, Any]]:
    out = []
    for path in _defs_dir().glob("ex_*.json"):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001 - one unreadable definition must not hide the rest
            continue
    return out


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --- validation ---------------------------------------------------------------------------

def _text(value: Any, field: str, limit: int, *, required: bool = False) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise Invalid(f"{field} must be text")
    value = value.strip()
    if required and not value:
        raise Invalid(f"{field} is required")
    if len(value) > limit:
        raise Invalid(f"{field} is longer than {limit} characters")
    return value


def validate_definition(data: Any) -> Dict[str, Any]:
    """The fields a user may set, cleaned. Raises ``Invalid`` naming the first bad field."""
    if not isinstance(data, dict):
        raise Invalid("the example must be a JSON object")
    title = _text(data.get("title"), "title", MAX_TITLE, required=True)
    description = _text(data.get("description", data.get("blurb")), "description", MAX_DESCRIPTION)
    # Free text, not a fixed list: the categories live in the client (DEMO_CATEGORIES), and a
    # second copy here is a copy that drifts. The panel files an unknown one under its own name.
    category = _text(data.get("category"), "category", MAX_CATEGORY, required=True)

    raw_tags = data.get("tags") or []
    if isinstance(raw_tags, str):
        raw_tags = raw_tags.split(",")
    if not isinstance(raw_tags, list):
        raise Invalid("tags must be a list")
    tags: List[str] = []
    for tag in raw_tags:
        tag = _text(tag, "a tag", MAX_TAG)
        if tag and tag not in tags:
            tags.append(tag)
    if len(tags) > MAX_TAGS:
        raise Invalid(f"at most {MAX_TAGS} tags")

    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise Invalid("an example needs at least one step")
    if len(raw_steps) > max_steps():
        raise Invalid(f"at most {max_steps()} steps")
    steps = []
    for n, step in enumerate(raw_steps, 1):
        if not isinstance(step, dict):
            raise Invalid(f"step {n} must have a label and a question")
        prompt = _text(step.get("prompt"), f"step {n}'s question", MAX_PROMPT, required=True)
        # A step with no label is labelled by its question, cut short, so the list stays readable.
        label = _text(step.get("label"), f"step {n}'s label", MAX_LABEL) or prompt[:60]
        steps.append({"label": label, "prompt": prompt})
    return {"title": title, "description": description, "category": category,
            "tags": tags, "steps": steps}


def _safe_name(name: str) -> str:
    safe = secure_filename(name or "")
    if not safe:
        raise Invalid(f"{name!r} is not a usable file name")
    return safe


# --- who may see what ---------------------------------------------------------------------

def _uid(user: Any) -> Optional[str]:
    value = getattr(user, "id", None)
    return str(value) if value else None


def is_admin(user: Any) -> bool:
    role = getattr(user, "role", None)
    return isinstance(role, int) and not isinstance(role, bool) and role <= publish_role()


def _is_owner(example: Dict[str, Any], user: Any) -> bool:
    uid = _uid(user)
    return bool(uid) and example.get("owner_id") == uid


def _shared_list(user_id: str) -> List[Dict[str, Any]]:
    try:
        value = json.loads(_shared_path(user_id).read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return []
    return [e for e in value if isinstance(e, dict)] if isinstance(value, list) else []


def _reached_by_link(example: Dict[str, Any], user: Any) -> bool:
    token = example.get("share_token")
    uid = _uid(user)
    if not token or not uid:
        return False
    return any(e.get("id") == example.get("id") and secrets.compare_digest(str(e.get("token") or ""), token)
               for e in _shared_list(uid))


def _can_view(example: Dict[str, Any], user: Any) -> bool:
    return bool(example.get("published")) or _is_owner(example, user) or _reached_by_link(example, user)


def _owned(example_id: str, user: Any) -> Dict[str, Any]:
    """The example, if the caller owns it. Anyone else gets NotFound, even one who can see it:
    "you may look but not edit" is a 403 they cannot act on, and the UI never offers it."""
    example = _read_def(example_id)
    if not _is_owner(example, user):
        raise NotFound("unknown example")
    return example


def visibility(example: Dict[str, Any]) -> str:
    if example.get("published"):
        return "published"
    return "link" if example.get("share_token") else "private"


def view(example: Dict[str, Any], user: Any) -> Dict[str, Any]:
    """What a client is sent. The owner's id never leaves the server (a platform id is a CILogon
    URL, which names a person), and only the owner sees the share token."""
    mine = _is_owner(example, user)
    eid = example["id"]
    body = {
        "id": eid,
        "source": "user",
        "title": example.get("title", ""),
        "blurb": example.get("description", ""),
        "category": example.get("category", ""),
        "tags": list(example.get("tags") or []),
        "steps": list(example.get("steps") or []),
        "files": [{"name": f["name"], "bytes": int(f.get("bytes") or 0),
                   "url": f"/agent/examples/{eid}/files/{f['name']}"}
                  for f in example.get("files") or []],
        "mine": mine,
        "visibility": visibility(example),
        "published": bool(example.get("published")),
        "createdAt": example.get("created_at"),
        "updatedAt": example.get("updated_at"),
    }
    if mine:
        body["shareToken"] = example.get("share_token")
        if example.get("unpublished_by_edit"):
            body["unpublishedByEdit"] = example["unpublished_by_edit"]
    if example.get("copied_from"):
        body["copiedFrom"] = example["copied_from"]
    return body


# --- files --------------------------------------------------------------------------------

def _save_stream(stream: BinaryIO, target: Path, budget: int) -> int:
    """Copy at most ``budget`` bytes. Over it, nothing is kept and LimitReached says by how much
    the limit was exceeded in terms a user can act on (the limit, not the byte count read)."""
    written = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(target, "wb") as out:
            while True:
                chunk = stream.read(CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > budget:
                    raise LimitReached(
                        f"an example's data may total at most {max_bytes() // (1024 * 1024)} MB")
                out.write(chunk)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return written


def _store_files(example_id: str, kept: List[Dict[str, Any]],
                 uploads: Iterable[Tuple[str, BinaryIO]]) -> List[Dict[str, Any]]:
    """Write the new uploads beside the kept files, enforcing the count and total size.
    A new file with a kept file's name replaces it. On any failure the new files are removed."""
    uploads = list(uploads)
    names = [_safe_name(name) for name, _ in uploads]
    if len(set(names)) != len(names):
        raise Invalid("two uploaded files have the same name")
    kept = [f for f in kept if f["name"] not in names]
    if len(kept) + len(names) > max_files():
        raise LimitReached(f"an example may have at most {max_files()} files")
    budget = max_bytes() - sum(int(f.get("bytes") or 0) for f in kept)
    folder = _files_dir(example_id)
    staged: List[Tuple[Path, Path, str, int]] = []
    try:
        for name, (_, stream) in zip(names, uploads):
            tmp = folder / f".upload.{secrets.token_hex(6)}.{name}"
            size = _save_stream(stream, tmp, budget)
            budget -= size
            staged.append((tmp, folder / name, name, size))
    except BaseException:
        for tmp, _, _, _ in staged:
            tmp.unlink(missing_ok=True)
        raise
    for tmp, final, _, _ in staged:
        os.replace(tmp, final)
    return kept + [{"name": name, "bytes": size} for _, _, name, size in staged]


def _prune_files(example_id: str, keep: Iterable[str]) -> None:
    keep = set(keep)
    folder = _files_dir(example_id)
    if not folder.is_dir():
        return
    for path in folder.iterdir():
        if path.is_file() and path.name not in keep and not path.name.startswith(".upload."):
            path.unlink(missing_ok=True)


def file_path(example_id: str, name: str, user: Any) -> Path:
    """A data file of an example the caller can see."""
    example = _read_def(example_id)
    if not _can_view(example, user):
        raise NotFound("unknown example")
    if name not in {f["name"] for f in example.get("files") or []}:
        raise NotFound("unknown file")
    path = (_files_dir(example_id) / name).resolve()
    if path.parent != _files_dir(example_id).resolve() or not path.is_file():
        raise NotFound("unknown file")
    return path


# --- operations ---------------------------------------------------------------------------

def _require_user(user: Any) -> str:
    uid = _uid(user)
    if not uid:
        raise SignInRequired("sign in to create or change your own examples")
    return uid


def create(user: Any, data: Any, uploads: Iterable[Tuple[str, BinaryIO]] = ()) -> Dict[str, Any]:
    uid = _require_user(user)
    clean = validate_definition(data)
    with _locked():
        count = sum(1 for e in _all_defs() if e.get("owner_id") == uid)
        if count >= max_examples_per_user():
            raise LimitReached(f"you have {count} examples, the most one account may keep; "
                               "delete one to add another")
        example_id = f"ex_{secrets.token_hex(8)}"
        try:
            files = _store_files(example_id, [], uploads)
        except BaseException:
            shutil.rmtree(_files_dir(example_id), ignore_errors=True)
            raise
        now = _now()
        example = {"id": example_id, "owner_id": uid, **clean, "files": files,
                   "share_token": None, "published": False,
                   "created_at": now, "updated_at": now}
        _write_json(_def_path(example_id), example)
    return example


def update(user: Any, example_id: str, data: Any, keep_files: Optional[Iterable[str]] = None,
           uploads: Iterable[Tuple[str, BinaryIO]] = ()) -> Dict[str, Any]:
    """Replace the editable fields. ``keep_files`` names the existing files to keep (None keeps
    all); ``uploads`` adds or replaces files.

    Editing a published example unpublishes it unless the editor may publish: an admin approved
    what it said, and an edit afterwards would otherwise change what everyone sees without that
    approval."""
    _require_user(user)
    clean = validate_definition(data)
    with _locked():
        example = _owned(example_id, user)
        current = list(example.get("files") or [])
        kept = current if keep_files is None else [f for f in current if f["name"] in set(keep_files)]
        files = _store_files(example_id, kept, uploads)
        _prune_files(example_id, [f["name"] for f in files])
        example.update(clean)
        example["files"] = files
        example["updated_at"] = _now()
        if example.get("published") and not is_admin(user):
            example["published"] = False
            example["unpublished_by_edit"] = example["updated_at"]
        _write_json(_def_path(example_id), example)
    return example


def delete(user: Any, example_id: str) -> None:
    with _locked():
        _owned(example_id, user)
        _def_path(example_id).unlink(missing_ok=True)
        shutil.rmtree(_files_dir(example_id), ignore_errors=True)


def get(user: Any, example_id: str) -> Dict[str, Any]:
    example = _read_def(example_id)
    if not _can_view(example, user):
        raise NotFound("unknown example")
    return example


def list_for(user: Any) -> Dict[str, List[Dict[str, Any]]]:
    """Three groups, each example in exactly one: mine, then published, then shared with me.
    A shared entry whose link was revoked, or whose example was deleted, is left out."""
    uid = _uid(user)
    defs = {e["id"]: e for e in _all_defs() if isinstance(e.get("id"), str)}
    mine = [e for e in defs.values() if uid and e.get("owner_id") == uid]
    published = [e for e in defs.values() if e.get("published") and e not in mine]
    shared = []
    if uid:
        for entry in _shared_list(uid):
            example = defs.get(entry.get("id"))
            if (example and example not in mine and example not in published
                    and example.get("share_token")
                    and secrets.compare_digest(str(entry.get("token") or ""), example["share_token"])):
                shared.append(example)
    newest = lambda e: e.get("updated_at") or ""  # noqa: E731
    return {"mine": sorted(mine, key=newest, reverse=True),
            "published": sorted(published, key=newest, reverse=True),
            "shared": shared}


def share(user: Any, example_id: str, *, rotate: bool = False) -> str:
    """The example's share token, made on first use. ``rotate`` replaces it, which cuts off
    everyone the old link reached."""
    with _locked():
        example = _owned(example_id, user)
        if rotate or not example.get("share_token"):
            example["share_token"] = secrets.token_urlsafe(18)
            example["updated_at"] = _now()
            _write_json(_def_path(example_id), example)
        return example["share_token"]


def unshare(user: Any, example_id: str) -> None:
    with _locked():
        example = _owned(example_id, user)
        if example.get("share_token"):
            example["share_token"] = None
            example["updated_at"] = _now()
            _write_json(_def_path(example_id), example)


def open_shared(user: Any, token: str) -> Dict[str, Any]:
    """Open a share link: the example it names, added to the caller's "Shared with me" list
    (unless it is their own). An unknown or revoked token is NotFound."""
    uid = _require_user(user)
    token = str(token or "").strip()
    if not token or len(token) > 64:
        raise NotFound("this link does not open an example; it may have been revoked")
    with _locked():
        example = next((e for e in _all_defs() if e.get("share_token")
                        and secrets.compare_digest(str(e["share_token"]), token)), None)
        if example is None:
            raise NotFound("this link does not open an example; it may have been revoked")
        if example.get("owner_id") != uid:
            entries = [e for e in _shared_list(uid) if e.get("id") != example["id"]]
            entries.insert(0, {"id": example["id"], "token": token, "added_at": _now()})
            # Bounded like everything else: a list of every link one account ever opened.
            _write_json(_shared_path(uid), entries[:200])
    return example


def forget_shared(user: Any, example_id: str) -> None:
    """Take an example off the caller's "Shared with me" list. Its owner keeps it."""
    uid = _require_user(user)
    with _locked():
        entries = _shared_list(uid)
        kept = [e for e in entries if e.get("id") != example_id]
        if len(kept) == len(entries):
            raise NotFound("unknown example")
        _write_json(_shared_path(uid), kept)


def copy(user: Any, example_id: str) -> Dict[str, Any]:
    """A private copy, owned by the caller, of an example they can see (files included)."""
    source = get(user, example_id)
    folder = _files_dir(example_id)
    handles = []
    try:
        for f in source.get("files") or []:
            handles.append((f["name"], open(folder / f["name"], "rb")))
        data = {k: source.get(k) for k in ("title", "description", "category", "tags", "steps")}
        example = create(user, data, handles)
    except FileNotFoundError:
        raise NotFound("a file of this example is missing") from None
    finally:
        for _, handle in handles:
            handle.close()
    with _locked():
        example["copied_from"] = example_id
        _write_json(_def_path(example["id"]), example)
    return example


def set_published(user: Any, example_id: str, published: bool) -> Dict[str, Any]:
    """Publish (admins only) or unpublish (admins, or the owner withdrawing their own)."""
    _require_user(user)
    with _locked():
        example = _read_def(example_id)
        admin = is_admin(user)
        if not _can_view(example, user):
            raise NotFound("unknown example")
        if published and not admin:
            raise Forbidden("only an admin can publish an example to everyone")
        if not published and not (admin or _is_owner(example, user)):
            raise Forbidden("only an admin or the example's owner can unpublish it")
        example["published"] = bool(published)
        if published:
            example["published_at"] = _now()
            example["published_by"] = _uid(user)
            example.pop("unpublished_by_edit", None)
        example["updated_at"] = _now()
        _write_json(_def_path(example_id), example)
    return example
