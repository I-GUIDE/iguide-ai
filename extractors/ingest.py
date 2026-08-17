"""Ingest harness — two entry points feeding the same manifest + emitters.

GitHub path  (code + notebook):  ingest_from_github(url, ...)   [IMPLEMENTED — dry-run]
Upload path  (dataset + publication, via webhook):  ingest_uploaded_file(path, ...)  [stub]

``ingest_from_github`` materializes the source (git clone, or a local path for
testing), classifies files, runs the implemented extractors (gracefully skipping
stub extractors), and merges everything into a ``UnifiedManifest``. Emitter fan-out
is not wired yet (design doc §10 step 2) — this returns the manifest as a dry run.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from typing import Callable, List, Optional, Sequence, Tuple

from .base import (
    KIND_CODE_BLOCK,
    KIND_NOTEBOOK_BLOCK,
    ExtractContext,
    Extractor,
    ExtractionResult,
    VALID_TARGETS,
)
from .doc_ids import repo_id
from .fileclass import classify_github, classify_upload
from .manifest import UnifiedManifest


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _materialize_source(url: str, ref: str) -> Tuple[str, str, Callable[[], None]]:
    """Return (repo_dir, commit_sha, cleanup). Uses a local path as-is if it exists,
    else shallow-clones the repo into a temp dir."""
    if os.path.exists(url):
        p = os.path.abspath(url)
        repo_dir = p if os.path.isdir(p) else os.path.dirname(p)
        return repo_dir, "", lambda: None

    tmp = tempfile.mkdtemp(prefix="iguide_ingest_")
    cmd = ["git", "clone", "--depth", "1"]
    if ref:
        cmd += ["--branch", ref]
    cmd += [url, tmp]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    try:
        sha = subprocess.run(["git", "-C", tmp, "rev-parse", "HEAD"],
                             check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        sha = ""
    return tmp, sha, lambda: shutil.rmtree(tmp, ignore_errors=True)


def _run_extractor(extractor: Extractor, files: List[str], ctx: ExtractContext,
                   manifest: UnifiedManifest) -> None:
    """Run ``extractor`` over ``files``, merge into manifest. Stub extractors
    (NotImplementedError) are skipped with a recorded warning."""
    combined = ExtractionResult()
    for f in files:
        try:
            res = extractor.extract(f, ctx=ctx)
        except NotImplementedError as exc:
            combined.warnings.append(f"{extractor.name} extractor not implemented: {exc}")
            break
        except Exception as exc:  # one bad file shouldn't sink the run
            combined.warnings.append(f"{extractor.name} failed on {f}: {type(exc).__name__}: {exc}")
            continue
        combined.assets.extend(res.assets)
        combined.edges.extend(res.edges)
        combined.warnings.extend(res.warnings)
        if combined.skill is None and res.skill is not None:
            combined.skill = res.skill
        elif res.skill is not None:
            combined.warnings.append(f"multiple skills; kept '{combined.skill.name}', dropped '{res.skill.name}'")
    manifest.add_result(extractor.name, combined)


def ingest_from_github(
    url: str,
    *,
    ref: str = "",
    targets: Sequence[str] = VALID_TARGETS,
    element_id: str = "",
    dry_run: bool = False,
    reingest: bool = False,
) -> UnifiedManifest:
    """Clone ``url`` (or use a local path), run notebook + code extractors, and emit.

    Previously this returned the manifest WITHOUT calling ``_fan_out``, so both callers
    that reach it -- ``extractors.cli`` and the MCP ``ingest_github_repo`` tool -- extracted
    and then silently discarded everything, while accepting a ``--targets`` argument that
    implied otherwise. ``ingest_submission`` (the webhook) was the only path that persisted.

    ``element_id`` anchors every derived doc_id on the platform element. Without it ids
    anchor on ``repo_id`` instead, which seeds unanchored docs into the agent KB that no
    element can ever claim -- so emitting requires either an ``element_id`` or an explicit
    ``dry_run``.
    """
    from .notebook_extractor import NotebookExtractor
    from .code_extractor import CodeExtractor

    if not dry_run and not element_id:
        raise ValueError(
            "ingest_from_github would emit docs anchored on repo_id, which no platform "
            "element can claim. Pass element_id=... to anchor them, or dry_run=True to "
            "inspect the manifest without emitting."
        )

    rid = repo_id(url)
    repo_dir, commit_sha, cleanup = _materialize_source(url, ref)
    ctx = ExtractContext(source_url=url, repo_id=rid, commit_sha=commit_sha,
                         element_id=element_id,
                         targets=tuple(targets), reingest=reingest, extra={"repo_dir": repo_dir})
    manifest = UnifiedManifest(repo_id=rid, source_url=url, commit_sha=commit_sha, cloned_at=_now_iso())
    try:
        buckets = classify_github(repo_dir)
        if buckets.get(KIND_NOTEBOOK_BLOCK):
            _run_extractor(NotebookExtractor(), buckets[KIND_NOTEBOOK_BLOCK], ctx, manifest)
        if buckets.get(KIND_CODE_BLOCK):
            _run_extractor(CodeExtractor(), buckets[KIND_CODE_BLOCK], ctx, manifest)
        if not dry_run:
            _fan_out(manifest, ctx.targets)
    finally:
        cleanup()
    return manifest


def ingest_submission(submission) -> UnifiedManifest:
    """Canonical entry: ingest ONE knowledge-element submission (from the webhook).

    Routes by ``element_type`` to the right extractor, anchors all derived doc_ids on
    the platform ``element_id``, and inherits the form ``fields`` into the docs.
    GitHub source (notebook/code) is cloned; file source (dataset/publication) is
    read from ``file_path``. Returns a dry-run UnifiedManifest (emitter fan-out is
    design-doc §10 step 2).
    """
    from .submission import GITHUB_TYPES
    from .fileclass import classify_github  # local import to keep module load light

    submission.validate()
    etype = submission.element_type
    rid = repo_id(submission.github_url) if submission.github_url else ""
    ctx = ExtractContext(
        element_id=submission.element_id,
        element_type=etype,
        source_url=submission.github_url or submission.file_path or submission.key,
        repo_id=rid,
        fields=submission.fields,
        targets=tuple(submission.targets),
    )
    manifest = UnifiedManifest(
        element_id=submission.element_id, element_type=etype,
        repo_id=rid, source_url=ctx.source_url, cloned_at=_now_iso(),
    )

    if etype in GITHUB_TYPES:
        repo_dir, commit_sha, cleanup = _materialize_source(submission.github_url, submission.ref)
        ctx.commit_sha = commit_sha
        ctx.extra["repo_dir"] = repo_dir
        manifest.commit_sha = commit_sha
        try:
            if etype == "notebook":
                from .notebook_extractor import NotebookExtractor
                if submission.notebook_file:
                    files = [os.path.join(repo_dir, submission.notebook_file)]
                else:
                    files = classify_github(repo_dir).get(KIND_NOTEBOOK_BLOCK, [])
                _run_extractor(NotebookExtractor(), files, ctx, manifest)
            else:  # code
                from .code_extractor import CodeExtractor
                files = classify_github(repo_dir).get(KIND_CODE_BLOCK, [])
                _run_extractor(CodeExtractor(), files, ctx, manifest)
        finally:
            cleanup()
    elif etype == "dataset":
        from .data_extractor import DataExtractor
        _run_extractor(DataExtractor(), [submission.file_path or submission.key], ctx, manifest)
    elif etype == "publication":
        from .publication_extractor import PublicationExtractor
        _run_extractor(PublicationExtractor(), [submission.file_path or submission.key], ctx, manifest)

    _fan_out(manifest, ctx.targets)
    return manifest


def _fan_out(manifest: UnifiedManifest, targets: Sequence[str]) -> None:
    """Route the manifest to emitters. Failures are recorded as warnings, never raised —
    extraction already succeeded.

    **The order is load-bearing, and it used to be wrong.** ``library_emitter`` is what assigns
    each unit its ``library_module`` (``library_emitter.py:304``, mutating the live unit dict),
    and the OpenSearch document advertises that module as the unit's import line. With OpenSearch
    emitting first, every indexed unit was built before its module path existed, so no document
    could ever carry an importable line — measured 0 of 130 on the corpus.

    That was invisible until the document started carrying the contract at all: before then it
    held neither the module nor anything else, so there was no field to notice was empty. Library
    first, then the index that describes it.
    """
    if "library" in targets:
        try:
            from .emitters import library_emitter
            summary = library_emitter.emit(manifest)
            if summary.get("written"):
                manifest.warnings.append(
                    f"[library] wrote {len(summary['written'])} unit module(s); "
                    f"registry now {summary.get('registry_size')} symbol(s)")
            if summary.get("skipped"):
                manifest.warnings.append(f"[library] skipped {len(summary['skipped'])} unit(s)")
        except Exception as exc:
            manifest.warnings.append(f"[library] emit failed: {type(exc).__name__}: {exc}")

    if "mcp" in targets:
        try:
            from .emitters import mcp_emitter
            summary = mcp_emitter.emit(manifest)
            if summary.get("written"):
                manifest.warnings.append(f"[mcp] wrote workflow manifests: {summary['written']}")
        except Exception as exc:
            manifest.warnings.append(f"[mcp] emit failed: {type(exc).__name__}: {exc}")

    # Last, because it mirrors what the emitters above produced — including the library module
    # path each unit's import line is built from.
    if "opensearch" in targets:
        try:
            from .emitters import opensearch_emitter
            summary = opensearch_emitter.emit(manifest)
            manifest.warnings.append(
                f"[kb:{summary.get('backend')}] indexed {summary.get('indexed')} docs "
                f"into {list(summary.get('indices', {}))}")
        except Exception as exc:
            manifest.warnings.append(f"[kb] emit failed: {type(exc).__name__}: {exc}")

    if "skill" in targets:
        try:
            from .emitters import skill_emitter
            summary = skill_emitter.emit(manifest)
            if summary.get("written"):
                manifest.warnings.append(f"[skill] wrote {summary['written']} (discoverable={summary.get('discoverable')})")
        except Exception as exc:
            manifest.warnings.append(f"[skill] emit failed: {type(exc).__name__}: {exc}")


def ingest_uploaded_file(
    path: str,
    *,
    element_id: Optional[str] = None,
    kind: Optional[str] = None,
    targets: Sequence[str] = ("opensearch",),
) -> UnifiedManifest:
    """Convenience wrapper: build a Submission for a dataset/publication file and
    route through ``ingest_submission``. ``kind`` defaults to ``classify_upload``."""
    from .submission import Submission

    etype = kind or classify_upload(path)
    sub = Submission(
        element_id=element_id or os.path.basename(path),
        element_type=etype,
        file_path=path,
        targets=list(targets),
    )
    return ingest_submission(sub)


__all__ = ["ingest_submission", "ingest_from_github", "ingest_uploaded_file"]
