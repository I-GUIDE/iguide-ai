"""Put a platform dataset's bytes where the sandbox can read them — agent-side.

Extraction generates 44 dataset loaders shaped ``def load_x(staged_path)``, and until now nothing
could give them a ``staged_path``: the only staging that existed resolved a user-uploaded
``file_id``. So an entire element type was readable and not runnable — the agent could describe a
dataset's schema and could not open the file.

**Why this runs agent-side and not in the sandbox.** The sandbox runs ``--network none``, and that
is a security property rather than a limitation to work around. MinIO keys, the OpenSearch
credentials and the platform's private subnet all live in the agent process; putting a fetch
inside the container would mean putting credentials inside it too. So the fetch happens here, and
only the resulting BYTES land in the workspace that is bind-mounted at ``/work``. The container
gains a file and learns nothing about where it came from or how to get another.

**Provenance is the same mechanism, not an extra one.** Every staged file appends a line to
``inputs.jsonl`` in the workspace — origin, sha256, size, content type, fetch time. That file is
what makes a run re-executable: ``scripts/rerun_artifact.py`` can re-stage by recorded hash and
verify it got the same bytes.

The URL comes from a model, so ``_assert_fetchable`` is the load-bearing function in this module.
An agent that can be steered into fetching ``http://169.254.169.254/`` or ``http://10.0.147.52:7687``
is a credential-exfiltration path with extra steps.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import socket
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

INPUTS_DIRNAME = "inputs"
MANIFEST_NAME = "inputs.jsonl"

# The container path the sandbox sees. `session_work_dir()` is bind-mounted at /work, so a file
# written to <work>/inputs/x.csv is /work/inputs/x.csv inside — and that is the string a generated
# loader needs, not the host path.
CONTAINER_WORK = "/work"

#: Set ``AGENT_STAGING_ALLOW_PRIVATE=1`` to permit private-range hosts. Off by default, and it
#: should stay off anywhere the agent holds credentials.
_ALLOW_PRIVATE_ENV = "AGENT_STAGING_ALLOW_PRIVATE"

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class StagingError(RuntimeError):
    """A staging request that was refused or failed, with a machine-readable ``kind``."""

    def __init__(self, message: str, *, kind: str = "error"):
        super().__init__(message)
        self.kind = kind


def _allow_private() -> bool:
    return str(os.getenv(_ALLOW_PRIVATE_ENV, "")).strip().lower() in {"1", "true", "yes"}


def _addresses_for(host: str) -> List[str]:
    try:
        return sorted({info[4][0] for info in socket.getaddrinfo(host, None)})
    except OSError as exc:
        raise StagingError(f"could not resolve host {host!r}: {exc}", kind="unresolvable") from exc


def _assert_fetchable(url: str) -> str:
    """Refuse anything that is not a public http(s) resource. Returns the URL unchanged.

    The URL is chosen by a language model, and this process holds MinIO keys, OpenSearch
    credentials and a route into the platform's private subnet. Every rejection below is a real
    reachable target from here, not a hypothetical:

    * ``file://`` / ``gs://`` / other schemes — read the agent's own disk;
    * ``169.254.169.254`` — the cloud metadata service, which hands out instance credentials;
    * ``127.0.0.1`` / ``::1`` — the agent's own API, which is what this tool would be used to
      attack from the inside;
    * ``10.0.147.52:7687`` — the platform's Neo4j, on the private subnet this host can reach.

    Resolution happens BEFORE the fetch and every returned address must pass, because a hostname
    that resolves to a public address for one lookup and a private one for the next is the whole
    trick behind DNS rebinding.
    """
    parsed = urlparse(str(url or "").strip())
    if parsed.scheme not in ("http", "https"):
        raise StagingError(
            f"only http(s) URLs can be staged, not {parsed.scheme or 'a bare path'!r}",
            kind="scheme")
    if not parsed.hostname:
        raise StagingError("the URL has no host", kind="scheme")
    if _allow_private():
        return url

    for address in _addresses_for(parsed.hostname):
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:                                  # pragma: no cover - defensive
            raise StagingError(f"host {parsed.hostname!r} resolved to something that is not an "
                               f"IP address: {address!r}", kind="blocked")
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                or ip.is_multicast or ip.is_unspecified):
            raise StagingError(
                f"refusing to stage from {parsed.hostname!r}: it resolves to {address}, which is "
                f"a private, loopback or link-local address. This agent can reach the platform's "
                f"internal services and its own API, so fetching one on request would be a "
                f"credential-exfiltration path. Set {_ALLOW_PRIVATE_ENV}=1 only in an environment "
                f"holding no credentials.", kind="blocked")
    return url


def safe_filename(name: str, *, fallback: str = "input.bin") -> str:
    """A plain filename that cannot escape the inputs directory.

    Path separators, ``..`` and every other unexpected character are replaced rather than
    rejected, because the name usually comes from a URL and being strict here would fail on
    ordinary files for no safety gain. Containment is asserted again after joining.
    """
    base = os.path.basename(str(name or "").strip().split("?")[0].split("#")[0])
    cleaned = _SAFE_NAME.sub("_", base).strip("._") or fallback
    return cleaned[:120]


def inputs_dir(session_id: str) -> Path:
    from agent_runtime.code_execution import session_work_dir

    path = session_work_dir(session_id) / INPUTS_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _destination(session_id: str, filename: str) -> Path:
    directory = inputs_dir(session_id)
    target = (directory / safe_filename(filename)).resolve()
    if target.parent != directory.resolve():
        raise StagingError("the destination escapes the inputs directory", kind="unsafe_path")
    return target


def _container_path(session_id: str, target: Path) -> str:
    """Where the staged file appears INSIDE the sandbox."""
    return f"{CONTAINER_WORK}/{INPUTS_DIRNAME}/{target.name}"


def record_input(session_id: str, entry: Dict[str, Any]) -> None:
    """Append one provenance line to the session's ``inputs.jsonl``.

    Append-only and one JSON object per line, so a partially-written run still yields every input
    that completed — a re-run needs to know what it HAD, not only what a tidy final state says.
    """
    from agent_runtime.code_execution import session_work_dir

    manifest = session_work_dir(session_id) / MANIFEST_NAME
    try:
        with open(manifest, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
    except OSError as exc:                                  # pragma: no cover - defensive
        logger.warning("could not record staged input for %s: %s", session_id, exc)


def staged_inputs(session_id: str) -> List[Dict[str, Any]]:
    """Everything staged into this session so far, oldest first."""
    from agent_runtime.code_execution import session_work_dir

    manifest = session_work_dir(session_id) / MANIFEST_NAME
    if not manifest.is_file():
        return []
    out: List[Dict[str, Any]] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _staged_result(session_id: str, target: Path, source: Any, origin: str) -> Dict[str, Any]:
    entry = {
        "staged_path": _container_path(session_id, target),
        "host_path": str(target),
        "filename": target.name,
        "origin": origin,
        "bytes": getattr(source, "bytes", None),
        "sha256": getattr(source, "sha256", ""),
        "content_type": getattr(source, "content_type", ""),
        "fetched_at": getattr(source, "fetched_at", ""),
        "backend": getattr(source, "backend", ""),
    }
    record_input(session_id, entry)
    return entry


def stage_url(url: str, session_id: str, *, filename: str = "") -> Dict[str, Any]:
    """Fetch a public http(s) URL into the session workspace. Returns the provenance entry."""
    from extractors.sources import SourceError, fetch_url

    _assert_fetchable(url)
    name = filename or os.path.basename(urlparse(url).path) or "input.bin"
    target = _destination(session_id, name)
    try:
        source = fetch_url(url, target, element_id=session_id)
    except SourceError as exc:
        raise StagingError(f"{exc}", kind=getattr(exc, "kind", "error")) from exc
    return _staged_result(session_id, Path(source.local_path), source, url)


def stage_object(bucket: str, key: str, session_id: str, *,
                 filename: str = "") -> Dict[str, Any]:
    """Fetch a MinIO/S3 object into the session workspace.

    The credentials used here stay in this process. Only the bytes cross into the container.
    """
    from extractors.sources import SourceError, fetch_object

    if not str(bucket or "").strip() or not str(key or "").strip():
        raise StagingError("both a bucket and a key are required", kind="bad_request")
    target = _destination(session_id, filename or os.path.basename(key) or "object.bin")
    try:
        source = fetch_object(bucket, key, target, element_id=session_id)
    except SourceError as exc:
        raise StagingError(f"{exc}", kind=getattr(exc, "kind", "error")) from exc
    return _staged_result(session_id, Path(source.local_path), source, f"{bucket}/{key}")


def stage_element(element_id: str, session_id: str, *, metadata: Optional[Dict] = None,
                  filename: str = "") -> Dict[str, Any]:
    """Stage the file a PLATFORM ELEMENT points at, by its id.

    This is the one that closes the loop the rest of the extraction work opened: a dataset element
    surfaced by search carries a generated loader whose only parameter is a staged path, and this
    is what produces that path. The link field is resolved through ``sources.source_link``, which
    knows both of the platform's naming vocabularies — the graph's ``external_link`` and the REST
    API's type-suffixed ``external-link-publication`` and friends.
    """
    from extractors.sources import source_link

    record = metadata
    if record is None:
        record = _element_metadata(element_id)
    if not record:
        raise StagingError(f"no platform record found for element {element_id!r}",
                           kind="not_found")

    bucket = str(record.get("bucket") or "").strip()
    key = str(record.get("key") or "").strip()
    if bucket and key:
        return stage_object(bucket, key, session_id, filename=filename)

    url = source_link(record)
    if not url:
        raise StagingError(
            f"element {element_id!r} records no direct file link. It may be a portal pointer "
            f"rather than a deposited dataset, in which case there is no file to stage.",
            kind="no_source")
    return stage_url(url, session_id, filename=filename)


_FULL_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_SHORT_ID = re.compile(r"^[0-9a-f]{6,12}$", re.I)
_id_index: Dict[str, str] = {}


def _element_index() -> Dict[str, str]:
    """Map of 8-character id prefix -> full platform UUID, fetched once per process.

    Built lazily and only when a short id actually arrives, so the common path — a caller that
    already has the full id — costs nothing.
    """
    global _id_index
    if _id_index:
        return _id_index
    try:
        import requests

        backend = os.getenv("IGUIDE_BACKEND_URL", "https://backend.i-guide.io").rstrip("/")
        # `size`, not `limit`: `limit` is accepted and ignored, returning the default page of 10,
        # which would silently resolve only the first ten elements on the platform.
        resp = requests.get(f"{backend}/api/elements", params={"size": 2000}, timeout=60)
        if resp.status_code == 200:
            for element in (resp.json() or {}).get("elements") or []:
                full = str(element.get("id") or "")
                if full:
                    _id_index.setdefault(full[:8].lower(), full)
    except Exception:
        pass
    return _id_index


def resolve_element_id(element_id: str) -> str:
    """Accept the 8-character id the rest of this system uses, not only the full UUID.

    The short id is the canonical form everywhere in extraction — `doc_ids`, the method-library
    registry, the Postgres record and every `provenance.element_id` — because that is what the
    ingest pipeline stores. The REST endpoint takes only the full UUID, so `/api/elements/265e6957`
    404s and staging reported "no platform record found".

    That is not a cosmetic mismatch. The knowledge base emits, at the point of use, `FIRST call
    stage_element("265e6957") to obtain staged_path` — the id it holds is the short one — so the
    KB was steering the agent into a call that could never succeed. Measured: on a two-dataset
    proximity problem the agent spent 2h07m and 21 KB tool calls following that instruction and
    returned no answer, while the same agent with the KB ablated staged the files and answered
    correctly in 15 minutes.
    """
    candidate = (element_id or "").strip()
    if not candidate or _FULL_ID.match(candidate):
        return candidate
    if not _SHORT_ID.match(candidate):
        return candidate
    return _element_index().get(candidate[:8].lower(), candidate)


def _element_metadata(element_id: str) -> Dict[str, Any]:
    """The platform's record for an element, from the public REST API.

    Not the graph: ``platform_graph`` exposes ``elements(type)`` for listing and has no
    single-element getter, and the REST endpoint needs no credentials — which matters because this
    runs on every staging request, including in environments with no Neo4j tunnel.
    """
    try:
        import requests

        backend = os.getenv("IGUIDE_BACKEND_URL", "https://backend.i-guide.io").rstrip("/")
        resp = requests.get(f"{backend}/api/elements/{resolve_element_id(element_id)}", timeout=30)
        if resp.status_code == 200 and isinstance(resp.json(), dict):
            return resp.json()
    except Exception:
        pass
    return {}


__all__ = ["stage_url", "stage_object", "stage_element", "staged_inputs", "record_input",
           "resolve_element_id",
           "inputs_dir", "safe_filename", "StagingError", "MANIFEST_NAME", "INPUTS_DIRNAME"]
