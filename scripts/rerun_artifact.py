"""Re-run an emitted artifact in a clean container and compare what it produced.

This is the script the reproducibility claim rests on. Everything else in the artifact layer
is bookkeeping until something actually replays it and says whether the numbers came back the
same.

What it enforces, and why each matters:

* **the image by digest, not by tag.** ``python:3.11-slim`` resolves to different bytes next
  month. A re-run on the tag is a different experiment wearing the same name; a re-run that
  cannot obtain the recorded digest says so and reports ``image_mismatch`` rather than
  quietly proceeding.
* **the input bytes.** Inputs are re-hashed and compared to the manifest. A file with the
  right name and different contents is the most convincing way to reproduce a wrong number.
* **the declared outputs.** ``IGUIDE_OUTPUTS`` is compared value by value. A run that
  declared nothing can be *repeated* but not *verified*, and this reports that difference
  instead of printing a reassuring "ok".

Exit codes: 0 identical · 1 differed · 2 could not run.

Usage
-----
    python scripts/rerun_artifact.py <artifact-dir>
    python scripts/rerun_artifact.py <artifact-dir> --allow-tag   # digest unavailable locally
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from agent_runtime.artifacts import (INPUTS_FILENAME, MANIFEST_FILENAME,  # noqa: E402
                                     RUN_FILENAME)
from agent_runtime.sandbox_verify import (CHECKS_FILENAME,  # noqa: E402
                                          ENVIRONMENT_FILENAME, DECLARED_OUTPUTS)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(artifact: Path) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]]]:
    manifest = json.loads((artifact / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    code = (artifact / RUN_FILENAME).read_text(encoding="utf-8")
    inputs: List[Dict[str, Any]] = []
    inputs_path = artifact / INPUTS_FILENAME
    if inputs_path.is_file():
        for line in inputs_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                inputs.append(json.loads(line))
    return manifest, code, inputs


def _image_available(ref: str) -> bool:
    try:
        return subprocess.run(["docker", "image", "inspect", ref],
                              capture_output=True, timeout=30).returncode == 0
    except Exception:
        return False


def _declared_from_checks(work: Path) -> Dict[str, Any]:
    """Read back what the run declared, via the epilogue's own report."""
    path = work / "declared_outputs.json"
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
    return {}


_CAPTURE = f'''

# --- rerun_artifact: publish declared outputs so the replay can be COMPARED, not just rerun ---
try:
    import json as _rj
    with open("declared_outputs.json", "w", encoding="utf-8") as _rf:
        _rj.dump(globals().get({DECLARED_OUTPUTS!r}) or {{}}, _rf, default=str)
except Exception:
    pass
'''


def _value_of(spec: Any) -> Any:
    return spec.get("value") if isinstance(spec, dict) else spec


def _same_value(old: Any, new: Any) -> bool:
    """Equal, with a relative tolerance for floats.

    Exact float equality would report a difference for the last bit of a sum whose order
    changed — a false alarm that would train everyone to ignore this script. 1e-9 relative is
    tight enough that a real change in a computed geospatial quantity still shows.
    """
    if old == new:
        return True
    try:
        a, b = float(old), float(new)
    except (TypeError, ValueError):
        return False
    if a == b:
        return True
    scale = max(abs(a), abs(b), 1e-30)
    return abs(a - b) / scale <= 1e-9


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("artifact")
    ap.add_argument("--allow-tag", action="store_true",
                    help="proceed on the image TAG when the recorded digest is unavailable")
    ap.add_argument("--keep", action="store_true", help="keep the replay work dir")
    args = ap.parse_args()

    artifact = Path(args.artifact)
    if not (artifact / MANIFEST_FILENAME).is_file():
        print(f"no {MANIFEST_FILENAME} in {artifact}")
        return 2
    manifest, code, inputs = _load(artifact)

    print(f"artifact   {artifact}")
    print(f"created    {manifest.get('created_at')}")
    print(f"backend    {manifest.get('backend')}")
    print(f"verified   {manifest.get('verified')} "
          f"(gate verdict {(manifest.get('verification') or {}).get('verdict')})")
    units = manifest.get("library_units") or []
    if units:
        print(f"library    {len(units)} unit(s): "
              + ", ".join(f"{u['symbol']}@{u.get('slice_sha') or '?'}" for u in units[:4]))

    # ---- the image, by digest ----
    digest = manifest.get("image_digest")
    tag = manifest.get("image") or ""
    if digest and _image_available(digest):
        image = digest
        print(f"image      {digest}  (pinned by digest)")
    elif args.allow_tag and tag and _image_available(tag):
        image = tag
        print(f"image      {tag}  (TAG ONLY — this is a different experiment if the tag moved)")
    else:
        print(f"\ncannot obtain the recorded image.\n  digest: {digest}\n  tag:    {tag}\n"
              f"Pull the digest, or pass --allow-tag to accept an unpinned replay.")
        return 2

    # ---- inputs, by hash ----
    work = Path(tempfile.mkdtemp(prefix="rerun_"))
    mismatched: List[str] = []
    missing: List[str] = []
    for row in inputs:
        name = str(row.get("name") or "")
        if not name:
            continue
        source = artifact / name
        if not source.is_file():
            missing.append(name)
            continue
        shutil.copy2(source, work / Path(name).name)
        want = row.get("sha256")
        if want and _sha256(work / Path(name).name) != want:
            mismatched.append(name)
    if inputs:
        print(f"inputs     {len(inputs)} recorded, {len(missing)} missing, "
              f"{len(mismatched)} hash mismatch")
    if mismatched:
        print(f"\nINPUT BYTES DIFFER: {mismatched}\nA replay on different bytes proves nothing.")
        return 1

    # ---- run ----
    # The epilogue is appended here as well: without it the replay produces no checks.json and
    # the gate verdict reads "original pass -> replay None", which looks like a regression and
    # is really just a missing check.
    from agent_runtime.sandbox_verify import epilogue_source
    (work / "script.py").write_text(code + epilogue_source() + _CAPTURE, encoding="utf-8")
    deps = manifest.get("dependencies") or []
    if deps:
        print(f"deps       installing {deps}")
        inst = subprocess.run(
            ["docker", "run", "--rm", "-v", f"{work}:/work:rw", "--memory", "1g",
             image, "pip", "install", "--no-cache-dir", "--target", "/work/.deps", *deps],
            capture_output=True, text=True, timeout=900)
        if inst.returncode != 0:
            print("dependency install failed:\n" + (inst.stderr or "")[-600:])
            return 2

    proc = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", "--memory", "2g",
         "-v", f"{work}:/work:rw", "--env", "PYTHONPATH=/work/.deps",
         "-w", "/work", image, "python", "/work/script.py"],
        capture_output=True, text=True, timeout=1800)
    print(f"\nreplay exit {proc.returncode}")
    if proc.returncode != 0:
        print((proc.stderr or "")[-700:])

    # ---- compare ----
    original = manifest.get("declared_outputs") or {}
    replay = _declared_from_checks(work)
    checks = {}
    if (work / CHECKS_FILENAME).is_file():
        try:
            checks = json.loads((work / CHECKS_FILENAME).read_text(encoding="utf-8"))
        except ValueError:
            checks = {}

    print(f"gate       original {(manifest.get('verification') or {}).get('verdict')} "
          f"-> replay {checks.get('verdict')}")

    if not replay:
        print("\nREPEATED, NOT VERIFIED: the run declared no IGUIDE_OUTPUTS, so there is "
              "nothing to compare. Declare the numbers the answer quotes to make this "
              "artifact checkable.")
        return 0 if proc.returncode == 0 else 1

    differences: List[str] = []
    no_baseline: List[str] = []
    for key in sorted(set(replay) | set(original)):
        new = _value_of(replay.get(key))
        old = _value_of(original.get(key))
        if key not in original:
            no_baseline.append(key)
            print(f"  {key}: replay={new}   (no baseline in the manifest)")
            continue
        if key not in replay:
            differences.append(f"{key}: missing from the replay (was {old})")
            print(f"  {key}: MISSING from the replay   original={old}")
            continue
        same = _same_value(old, new)
        print(f"  {key}: replay={new}   original={old}   {'==' if same else '!= DIFFERS'}")
        if not same:
            differences.append(f"{key}: {old} -> {new}")

    original_verdict = (manifest.get("verification") or {}).get("verdict")
    if original_verdict and checks.get("verdict") and checks["verdict"] != original_verdict:
        differences.append(f"gate verdict {original_verdict} -> {checks['verdict']}")

    if differences:
        print(f"\nDIFFERED ({len(differences)}):")
        for d in differences:
            print(f"  - {d}")
        return 1
    if no_baseline and not [k for k in replay if k in original]:
        print("\nREPEATED, NOT VERIFIED: nothing in the replay had a recorded baseline.")
        return 0 if proc.returncode == 0 else 1

    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print(f"\nwork dir kept at {work}")
    print("\nreplay completed" + (" (outputs identical)" if not differences else ""))
    return 0 if proc.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
