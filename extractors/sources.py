"""Resolve a knowledge element's real source file, wherever it lives.

Extractors take a local path; the corpus does not. Notebooks and code live on GitHub,
user-uploaded datasets live in MinIO, and some elements only carry a direct link. This is the
one place that turns platform metadata into a local file plus the provenance record that makes
the fetch reproducible.

Three backends, no new fetching code — each lifts a pattern that already worked somewhere:

  GitHub   ``agent_runtime.element_resolver.github_find_raw_url`` (default-branch resolution
           and a recursive tree walk, so a notebook in a subdirectory is found)
  MinIO/S3 the boto3 ``signature_version='s3v4'`` client from the metadata-extraction server
  HTTP     a plain streamed GET

**Credentials never enter the sandbox.** Resolution and fetching happen agent-side; what
reaches the sandbox is a staged local path. That is why ``--network none`` can stay closed.

Every resolution returns a :class:`ResolvedSource`, and that single record is the provenance
for both the index document and the artifact manifest — so "where did this come from" has one
answer, not two that can disagree.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote, unquote, urlparse

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60
MAX_BYTES = int(os.getenv("EXTRACTOR_MAX_SOURCE_BYTES", str(64 * 1024 * 1024)))


class SourceError(RuntimeError):
    """Typed failure so a caller can tell 'not found' from 'server error' from 'too big'.

    The webhook used to surface any of these as a 500 with a stack trace; an element whose
    repo was renamed is a data problem, not a server fault.
    """

    def __init__(self, message: str, *, kind: str = "error", element_id: str = ""):
        super().__init__(message)
        self.kind = kind              # not_found | forbidden | too_large | network | unsupported
        self.element_id = element_id


@dataclass
class ResolvedSource:
    """What was fetched, from where, and proof it is byte-identical on a re-run."""
    local_path: str = ""
    origin_url: str = ""
    bucket: str = ""
    key: str = ""
    sha256: str = ""
    bytes: int = 0
    content_type: str = ""
    fetched_at: str = ""
    commit_sha: str = ""
    ref: str = ""                     # branch or tag the URL pins, when known
    backend: str = ""                 # github | minio | http | local
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_provenance(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in ("", 0, {}, None)}


# --------------------------------------------------------------------------- #
# GitHub
# --------------------------------------------------------------------------- #

_BLOB_RE = re.compile(r"^https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)$", re.I)


def raw_url_from_blob(url: str) -> Optional[str]:
    """Convert a github.com /blob/<ref>/<path> URL to its raw.githubusercontent equivalent.

    Preferred over a tree walk when the element carries ``notebook-url``, because the blob URL
    already names the REF the curator linked. Guessing the default branch instead silently
    resolves to a different commit than the one the element was published against.
    """
    m = _BLOB_RE.match((url or "").strip())
    if not m:
        return None
    owner, repo, ref, path = m.groups()
    safe = "/".join(quote(unquote(p)) for p in path.split("/"))
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{safe}"


def resolve_github_url(metadata: Dict[str, Any]) -> Optional[str]:
    """Best raw URL for an element's source file, or None.

    Order matters: the curator-supplied blob URL pins a ref, the repo+path pair needs the
    default branch looked up, and the bare tree walk is the last resort for a filename with no
    directory. Measured across the live corpus in ``scripts/measure_source_fetchability.py``.
    """
    for key in ("notebook-url", "notebook_url", "code-url", "url"):
        raw = raw_url_from_blob(str(metadata.get(key) or ""))
        if raw:
            return raw

    # `github-repo-link` is what a CODE element carries — measured on the live corpus, 36 of 36
    # of them have it and none has any of the other spellings. Its absence here made every code
    # element resolve to "no source", which is the entire reason the library holds zero code
    # units despite CodeExtractor working.
    repo = str(metadata.get("notebook-repo") or metadata.get("notebook_repo")
               or metadata.get("github-repo") or metadata.get("github-repo-link")
               or metadata.get("github_repo_link") or metadata.get("github_url") or "").strip()
    rel = str(metadata.get("notebook-file") or metadata.get("notebook_file")
              or metadata.get("file-path") or "").strip()
    if not repo:
        return None

    if rel:
        # A repo-relative path is authoritative when present; try it directly on the default
        # branch before paying for a recursive tree walk.
        owner_repo = _owner_repo(repo)
        if owner_repo:
            owner, name = owner_repo
            ref = metadata.get("ref") or _default_branch(owner, name) or "main"
            safe = "/".join(quote(p) for p in rel.split("/"))
            return f"https://raw.githubusercontent.com/{owner}/{name}/{ref}/{safe}"

    try:
        from agent_runtime.element_resolver import github_find_raw_url
        return github_find_raw_url(repo, rel or "")
    except Exception as exc:
        logger.debug("github tree walk failed for %s: %s", repo, exc)
        return None


def _owner_repo(repo_url: str):
    m = re.match(r"^https?://github\.com/([^/]+)/([^/#?]+)", (repo_url or "").strip(), re.I)
    if not m:
        return None
    return m.group(1), m.group(2).removesuffix(".git")


def _default_branch(owner: str, repo: str, *, timeout: int = 30) -> Optional[str]:
    import requests

    try:
        from agent_runtime.element_resolver import _gh_headers
        headers = _gh_headers()
    except Exception:
        headers = {}
    try:
        r = requests.get(f"https://api.github.com/repos/{owner}/{repo}",
                         headers=headers, timeout=timeout)
        if r.ok:
            return r.json().get("default_branch")
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------- #
# MinIO / S3
# --------------------------------------------------------------------------- #

def _s3_client():
    """The boto3 client the metadata-extraction server already uses.

    Credentials are read agent-side only and are never passed into the execution sandbox.
    """
    import boto3
    from botocore.client import Config

    endpoint = (os.getenv("MINIO_ENDPOINT") or os.getenv("AWS_S3_ENDPOINT") or "").strip()
    key = (os.getenv("MINIO_AWS_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID") or "").strip()
    secret = (os.getenv("MINIO_AWS_SECRET_ACCESS_KEY")
              or os.getenv("AWS_SECRET_ACCESS_KEY") or "").strip()
    if not (key and secret):
        raise SourceError("MinIO credentials are not configured "
                          "(MINIO_AWS_ACCESS_KEY_ID / MINIO_AWS_SECRET_ACCESS_KEY)",
                          kind="forbidden")
    kwargs: Dict[str, Any] = {"aws_access_key_id": key, "aws_secret_access_key": secret,
                              "config": Config(signature_version="s3v4")}
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    return boto3.client("s3", **kwargs)


def fetch_object(bucket: str, key: str, dest: Path, *, element_id: str = "") -> ResolvedSource:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        _s3_client().download_file(bucket, key, str(dest))
    except SourceError:
        raise
    except Exception as exc:
        kind = "not_found" if "404" in str(exc) or "NoSuchKey" in str(exc) else "network"
        raise SourceError(f"MinIO download failed for {bucket}/{key}: {exc}",
                          kind=kind, element_id=element_id) from exc
    return _finish(dest, bucket=bucket, key=key, backend="minio")


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

def fetch_url(url: str, dest: Path, *, element_id: str = "",
              timeout: int = DEFAULT_TIMEOUT) -> ResolvedSource:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            if r.status_code == 404:
                raise SourceError(f"source not found: {url}", kind="not_found",
                                  element_id=element_id)
            if r.status_code in (401, 403):
                raise SourceError(f"source forbidden ({r.status_code}): {url}",
                                  kind="forbidden", element_id=element_id)
            r.raise_for_status()
            content_type = (r.headers.get("Content-Type") or "").split(";")[0].strip()
            total = 0
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(chunk_size=65536):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > MAX_BYTES:
                        # Streamed and capped rather than read into memory: one oversized
                        # raster must not take the ingest process down with it.
                        fh.close()
                        dest.unlink(missing_ok=True)
                        raise SourceError(f"source exceeds {MAX_BYTES} bytes: {url}",
                                          kind="too_large", element_id=element_id)
                    fh.write(chunk)
    except SourceError:
        raise
    except Exception as exc:
        raise SourceError(f"fetch failed for {url}: {exc}", kind="network",
                          element_id=element_id) from exc

    src = _finish(dest, origin_url=url, backend="github" if "raw.githubusercontent" in url else "http")
    src.content_type = content_type or src.content_type
    m = re.match(r"^https://raw\.githubusercontent\.com/[^/]+/[^/]+/([^/]+)/", url)
    if m:
        src.ref = m.group(1)
    return src


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def resolve_and_fetch(metadata: Dict[str, Any], dest_dir: Path, *,
                      element_id: str = "", filename: str = "") -> ResolvedSource:
    """Fetch an element's source into *dest_dir* and return its provenance.

    Raises :class:`SourceError` with a ``kind`` rather than returning None, so a caller can
    record WHY an element was skipped instead of reporting an undifferentiated failure count.
    """
    eid = element_id or str(metadata.get("id") or "")
    dest_dir = Path(dest_dir)

    bucket = str(metadata.get("bucket") or "").strip()
    key = str(metadata.get("key") or "").strip()
    if bucket and key:
        return fetch_object(bucket, key, dest_dir / (filename or Path(key).name), element_id=eid)

    url = resolve_github_url(metadata)
    if not url:
        for k in ("direct-download-link", "direct_download_link", "external-link", "url"):
            candidate = str(metadata.get(k) or "").strip()
            if candidate.startswith(("http://", "https://")):
                url = candidate
                break
    if not url:
        raise SourceError("element carries no resolvable source "
                          "(no bucket/key, no notebook-repo/url, no direct link)",
                          kind="unsupported", element_id=eid)

    name = filename or Path(urlparse(url).path).name or f"{eid or 'source'}.bin"
    return fetch_url(url, dest_dir / name, element_id=eid)


def _finish(dest: Path, *, origin_url: str = "", bucket: str = "", key: str = "",
            backend: str = "") -> ResolvedSource:
    import datetime as _dt
    import mimetypes

    data = dest.read_bytes()
    return ResolvedSource(
        local_path=str(dest),
        origin_url=origin_url,
        bucket=bucket,
        key=key,
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
        content_type=mimetypes.guess_type(dest.name)[0] or "",
        fetched_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        backend=backend or "local",
    )


__all__ = ["ResolvedSource", "SourceError", "resolve_and_fetch", "resolve_github_url",
           "raw_url_from_blob", "fetch_url", "fetch_object"]
