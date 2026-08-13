"""Emit a run as a reproducible artifact, and pin everything it depended on.

The deliverable of this system is meant to be a *runnable, verified artifact* — the answer is
a byproduct of running it. That claim needs four things recorded, and each was absent:

  the code        the exact source that ran, not a reconstruction from the answer
  the environment the container's OWN account of its interpreter and packages, captured
                  in-sandbox, plus the image by **digest** rather than by tag
  the inputs      every staged file with a sha256, so a re-run can prove it read the same bytes
  the outputs     the numbers the answer quoted, with their declared units, plus the gate's
                  verdict — so a re-run has something to *compare*, not just repeat

A tag is not a pin. ``python:3.11-slim`` resolves to different bytes next month, so an artifact
recording the tag records nothing about the environment; ``image_digest`` is what makes the
re-run meaningful.

Pure and I/O-light: builds a dict and writes files. ``scripts/rerun_artifact.py`` consumes it.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"
RUN_FILENAME = "run.py"
INPUTS_FILENAME = "inputs.jsonl"
ARTIFACT_SCHEMA = 1


def artifacts_enabled() -> bool:
    """Whether to emit an artifact bundle per substantive run. On by default.

    Cheap — a few small files beside a workspace that already exists — and the thing it buys
    (a re-runnable record) cannot be reconstructed after the fact, so defaulting it off would
    mean the artifact is missing exactly when someone wants it.
    """
    return (os.getenv("AGENT_ARTIFACT_EMIT", "1") or "").strip().lower() not in {
        "0", "false", "no", "off"}


def resolve_image_digest(image: str) -> Optional[str]:
    """``repo@sha256:…`` for a local image tag, or None.

    Prefers a RepoDigest (what the registry serves, so another machine can pull the same
    bytes) and falls back to the local image Id. Returns None rather than guessing when
    neither is available — a manifest claiming a digest it did not verify is worse than one
    that admits the image was unpinned.
    """
    image = (image or "").strip()
    if not image:
        return None
    if "@sha256:" in image:
        return image
    try:
        proc = subprocess.run(
            ["docker", "image", "inspect", image,
             "--format", "{{json .RepoDigests}}|{{.Id}}"],
            capture_output=True, text=True, timeout=30)
    except Exception as exc:
        logger.debug("could not inspect %s: %s", image, exc)
        return None
    if proc.returncode != 0:
        return None
    raw = (proc.stdout or "").strip()
    digests_json, _, image_id = raw.partition("|")
    try:
        digests = json.loads(digests_json) or []
    except ValueError:
        digests = []
    if digests:
        return str(digests[0])
    return image_id.strip() or None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


_IMPORT_RE = re.compile(r"^\s*from\s+(iguide_methods\.[\w.]+)\s+import\s+([\w, ]+)", re.M)


def library_units_used(code: str) -> List[Dict[str, str]]:
    """Method-library units the code imports, with their version-pinned module path.

    The ``v_<slice_sha>`` segment is the whole point: it records WHICH version of an extracted
    unit produced the result, so a re-run after a re-ingest imports the same code rather than
    a newer one that happens to share a name.
    """
    out: List[Dict[str, str]] = []
    for module, symbols in _IMPORT_RE.findall(code or ""):
        sha = ""
        for part in module.split("."):
            if part.startswith("v_"):
                sha = part[2:]
        for symbol in (s.strip() for s in symbols.split(",")):
            if symbol:
                out.append({"symbol": symbol, "module": module, "slice_sha": sha})
    return out


def build_manifest(*, code: str, work: Path, image: str, backend: str,
                   dependencies: Optional[List[str]] = None,
                   tier: Optional[str] = None,
                   inputs: Optional[List[Dict[str, Any]]] = None,
                   verification: Optional[Dict[str, Any]] = None,
                   environment: Optional[Dict[str, Any]] = None,
                   declared_outputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Everything needed to re-run this and compare the result."""
    return {
        "schema": ARTIFACT_SCHEMA,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "backend": backend,
        "image": image,
        # None is recorded explicitly: "we could not pin this" is information a re-run needs.
        "image_digest": resolve_image_digest(image) if backend == "docker" else None,
        "tier": tier,
        "dependencies": list(dependencies or []),
        "code_sha256": hashlib.sha256((code or "").encode("utf-8")).hexdigest(),
        "library_units": library_units_used(code),
        "inputs": list(inputs or []),
        "environment": environment or {},
        "declared_outputs": declared_outputs or {},
        "verification": verification or {},
        # The single field a reader should branch on before quoting any number from this run.
        "verified": bool((verification or {}).get("verdict") == "pass"),
    }


def read_json(work: Path, name: str) -> Dict[str, Any]:
    path = Path(work) / name
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def collect_inputs(work: Path, staged: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Staged input files with a sha256 each, merged with any provenance already recorded.

    A re-run must be able to assert it read the *same bytes*, not merely a file with the same
    name — which is why the hash matters more than the path.
    """
    work = Path(work)
    recorded: Dict[str, Dict[str, Any]] = {}
    existing = work / INPUTS_FILENAME
    if existing.is_file():
        try:
            for line in existing.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if isinstance(row, dict) and row.get("name"):
                    recorded[str(row["name"])] = row
        except (OSError, ValueError):
            pass

    out: List[Dict[str, Any]] = []
    for name in sorted(set(staged or ())):
        path = work / name
        row: Dict[str, Any] = dict(recorded.get(name) or {})
        row["name"] = name
        if path.is_file():
            try:
                row["sha256"] = _sha256_file(path)
                row["bytes"] = path.stat().st_size
            except OSError:
                pass
        out.append(row)
    # Inputs staged by a tool that recorded provenance but whose file is gone still belong in
    # the record: dropping them would make the artifact look self-contained when it is not.
    for name, row in recorded.items():
        if name not in {r["name"] for r in out}:
            out.append(row)
    return out


def emit(*, code: str, work: Path, image: str, backend: str,
         dependencies: Optional[List[str]] = None, tier: Optional[str] = None,
         staged: Optional[List[str]] = None,
         verification: Optional[Dict[str, Any]] = None,
         dest: Optional[Path] = None) -> Dict[str, Any]:
    """Write ``run.py``, ``manifest.json`` and ``inputs.jsonl`` into *dest* (default: work).

    Never raises: an artifact-emission failure must not fail a successful analysis.
    """
    work = Path(work)
    target = Path(dest or work)
    try:
        environment = read_json(work, "environment.json")
        checks = read_json(work, "checks.json")
        # The VALUES the run published, not the gate's findings about them. A manifest holding
        # "unit is null" instead of 25000 records the complaint and loses the measurement, so a
        # re-run would have nothing to compare against.
        declared = read_json(work, "declared_outputs.json")
        manifest = build_manifest(
            code=code, work=work, image=image, backend=backend,
            dependencies=dependencies, tier=tier,
            inputs=collect_inputs(work, staged),
            verification=verification or {"verdict": checks.get("verdict")} if checks else {},
            environment=environment, declared_outputs=declared)
        target.mkdir(parents=True, exist_ok=True)
        (target / RUN_FILENAME).write_text(code or "", encoding="utf-8")
        (target / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, default=str),
                                                encoding="utf-8")
        with open(target / INPUTS_FILENAME, "w", encoding="utf-8") as fh:
            for row in manifest["inputs"]:
                fh.write(json.dumps(row, default=str) + "\n")
        return manifest
    except Exception as exc:
        logger.warning("artifact emission failed: %s", exc)
        return {}


__all__ = ["emit", "build_manifest", "resolve_image_digest", "library_units_used",
           "collect_inputs", "artifacts_enabled", "MANIFEST_FILENAME", "RUN_FILENAME",
           "INPUTS_FILENAME", "ARTIFACT_SCHEMA"]
