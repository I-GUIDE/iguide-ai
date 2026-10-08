#!/usr/bin/env python3
"""Walk every seam of the system and say which ones hold.

    python scripts/smoke_end_to_end.py              # everything reachable
    python scripts/smoke_end_to_end.py --offline    # skip the cluster and the API
    python scripts/smoke_end_to_end.py --api http://127.0.0.1:3500 --key "$AGENT_CHAT_API_KEY"

Why this exists rather than a checklist in a document: nearly every failure this system has
produced was a step that reported success while the thing it was meant to establish was false.
A checklist inherits that problem — you tick "library builds" without noticing that the units it
built cannot be imported. So each check here asserts an OUTCOME, prints the number it measured,
and a check that cannot run is reported as SKIP with the reason. **A skip is never a pass.**

Exit code is the number of failures, so it composes with CI and with `&&`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
_results: list = []


def record(stage: str, status: str, detail: str) -> None:
    _results.append((stage, status, detail))
    mark = {PASS: "  ok ", FAIL: "  FAIL", SKIP: "  skip"}[status]
    print(f"{mark}  {stage:<34} {detail}")


def section(name: str) -> None:
    print(f"\n{name}")
    print("-" * 78)


# --------------------------------------------------------------------------- 1. configuration

def check_config(offline: bool) -> None:
    section("1. configuration — the settings whose absence fails quietly")
    from agent_runtime.code_execution import method_library_dir

    from agent_runtime.extraction_flag import extraction_enabled

    # A pre-flight as much as a check: run it BEFORE turning the bundle on for the server. This
    # process forces the bundle on so sections 2-4 test what a turn would get once it is.
    on = extraction_enabled()
    record("extraction bundle", PASS if on else SKIP,
           "on" if on else "off for the server (AGENT_EXTRACTION unset); forced on for this "
                           "pre-flight only")
    os.environ["AGENT_EXTRACTION"] = "1"

    lib = method_library_dir()
    record("method library resolves", PASS if lib else FAIL,
           str(lib) if lib else "AGENT_METHOD_LIBRARY_DIR unset and no storage_root copy — "
                                "kb_method_search will report an empty library")
    # Docker-out-of-Docker: inside a container, the library mount's source is resolved by the
    # HOST's daemon, and the default path is on a named volume the host does not have. The sandbox
    # check in section 4 proves it either way; this says why before it runs.
    if Path("/.dockerenv").exists() and (os.getenv("AGENT_CODE_EXEC_BACKEND") or "docker") == "docker":
        explicit = bool((os.getenv("AGENT_METHOD_LIBRARY_DIR") or "").strip())
        record("library path visible to the host", PASS if explicit else FAIL,
               "AGENT_METHOD_LIBRARY_DIR set (it must be bind-mounted at the identical path)"
               if explicit else "default path is container-local: the sandbox will get an EMPTY "
                                "library (include docker-compose.extraction.yml)")

    backend = (os.getenv("AGENT_KB_BACKEND") or "local").strip()
    cluster = bool(os.getenv("OPENSEARCH_NODE"))
    if backend == "opensearch":
        record("agent KB backend", PASS, "opensearch")
    elif cluster:
        record("agent KB backend", FAIL,
               "local, but OPENSEARCH_NODE is set — the indexed corpus is NOT being searched")
    else:
        record("agent KB backend", SKIP, "local, and no cluster configured")

    provider = (os.getenv("LLM_PROVIDER") or "").strip()
    record("LLM provider", PASS if provider else SKIP,
           provider or "unset — the agent path will fail to build a model")
    if provider == "claude-cli":
        record("provider is dev-only", SKIP,
               "claude-cli must never back a deployed server; fine locally")

    image = (os.getenv("AGENT_CODE_EXEC_IMAGE") or "").strip()
    record("sandbox image pinned", PASS if "@sha256:" in image else SKIP,
           image or "unset — falls back to a tag, so artifacts record no environment")


# --------------------------------------------------------------------------- 2. the library

def check_library() -> None:
    section("2. method library — does what was extracted actually import")
    from agent_runtime.method_library import library_summary, load_registry

    summary = library_summary()
    reg = load_registry()
    if not summary["units"]:
        record("library built", FAIL, "0 units — run scripts/build_method_library.py")
        return
    record("library built", PASS,
           f"{summary['units']} units from {summary['elements']} elements "
           f"({summary['kinds']})")

    # The direction that matters is one-way. Slices are content-addressed, so re-extracting an
    # edited function MINTS a new `v_<sha>.py` and leaves the old one importable — that is what
    # makes "which version" in an artifact's provenance a resolvable question. So more modules
    # than units is normal (measured: 617 modules for 551 units, 66 superseded versions, 129
    # modules for 92 units in the most-re-extracted element).
    #
    # A registry unit whose module is MISSING is the real fault, and it is what the old wording
    # described ("a stale entry advertises a contract for code that is gone") while the assertion
    # tested equality — so it failed on healthy version history and would also have failed, with
    # the same message, on the genuine problem.
    root = Path(summary["root"]) / "iguide_methods"
    modules = len(list(root.rglob("v_*.py")))
    missing = []
    for key, entry in reg.items():
        if not isinstance(entry, dict) or entry.get("alias_for") or entry.get("ambiguous"):
            continue
        module = str(entry.get("module") or "")
        if not module:
            continue
        relative = module.split(".", 1)[-1].replace(".", "/") + ".py"
        if not (root / relative).is_file():
            missing.append(key)
    record("every registry unit has its module on disk", PASS if not missing else FAIL,
           f"{summary['units']} units, {modules} modules on disk "
           f"({modules - summary['units']} superseded versions kept by design)"
           if not missing else
           f"{len(missing)} unit(s) advertise a module that is gone, e.g. {missing[:3]}")

    invariants = sum(1 for v in reg.values()
                     if isinstance(v, dict) and not v.get("alias_for") and v.get("invariants"))
    record("units with a contract", PASS if invariants else FAIL,
           f"{invariants} of {summary['units']} carry an enforceable invariant")

    # Import a sample in a SUBPROCESS with only the library on the path. Importing in-process
    # would let this repo's own site-packages mask a missing declared dependency.
    sample = [v for v in reg.values()
              if isinstance(v, dict) and not v.get("alias_for") and v.get("module")][:25]
    undeclared, expected = [], []
    for entry in sample:
        code = f"import {entry['module']} as m; getattr(m, {entry['library_symbol']!r})"
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                              cwd=str(root.parent), timeout=120)
        if proc.returncode == 0:
            continue
        tail = (proc.stderr or "").strip().splitlines()[-1][:70]
        # A unit that needs cartopy and SAYS SO is correct: the caller installs declared
        # requirements, and this host has not. What is a defect is a module the unit needs and
        # never declared — the contract is then incomplete, and the failure lands in the sandbox
        # at import time with no way for the caller to have prevented it.
        missing = tail.split("'")[1] if "No module named" in tail and "'" in tail else ""
        declared = {str(d).split("[")[0].split("==")[0].strip().lower().replace("-", "_")
                    for d in ((entry.get("requirements") or {}).get("pip") or [])}
        # Resolve the IMPORT name to its DISTRIBUTION name before comparing. `import ee` is
        # satisfied by `earthengine-api`, `sklearn` by `scikit-learn`, `PIL` by `pillow` — the two
        # namespaces differ by design, and translating between them is exactly what pkgmap exists
        # for. Comparing them directly reported 6 modules across ~30 units as undeclared when
        # every one of them was declared correctly. (Written here first, then found; the check was
        # wrong and the extraction was right.)
        try:
            from extractors.pkgmap import requirements_from_source

            resolved = (requirements_from_source(f"import {missing}\n").get("pip") or [missing]) \
                if missing else []
        except Exception:
            resolved = [missing] if missing else []
        candidates = {str(r).split("[")[0].split("==")[0].strip().lower().replace("-", "_")
                      for r in resolved} | ({missing.lower().replace("-", "_")} if missing else set())
        if candidates & declared:
            expected.append((entry["library_symbol"], missing))
        else:
            undeclared.append((entry["library_symbol"], tail))
    ran = len(sample) - len(undeclared) - len(expected)
    detail = f"{ran}/{len(sample)} import on a bare host"
    if expected:
        detail += f"; {len(expected)} need declared deps not installed here " \
                  f"(e.g. {expected[0][0]} needs {expected[0][1]}) — expected"
    record("units import standalone", PASS if not undeclared else FAIL,
           detail + ("" if not undeclared
                     else f" | UNDECLARED: {undeclared[0][0]}: {undeclared[0][1]}"))


# --------------------------------------------------------------------------- 3. retrieval

def check_retrieval(offline: bool) -> None:
    section("3. retrieval — can the agent find what was extracted")
    from agent_runtime.method_library import search_methods

    hits = search_methods("buffer geometries by a distance", limit=10)
    found = [h["symbol"].split(".")[-1] for h in hits]
    ok = "calculate_buffers" in found
    record("method search finds a known unit", PASS if ok else FAIL,
           f"calculate_buffers at rank {found.index('calculate_buffers') + 1}" if ok
           else f"not in top 10; got {found[:3]}")

    if offline or not os.getenv("OPENSEARCH_NODE"):
        record("agent KB search", SKIP, "no cluster configured")
        return
    try:
        from rag_pipeline.search.agent_kb import agent_kb_search

        out = agent_kb_search("spatial accessibility catchment", size=8)
        n, backend = out.get("count", 0), out.get("backend")
        record("agent KB search", PASS if n else FAIL,
               f"{n} documents from the {backend} backend"
               + (f" — {out['note']}" if out.get("note") else ""))
    except Exception as exc:
        record("agent KB search", FAIL, f"{type(exc).__name__}: {exc}"[:70])


# --------------------------------------------------------------------------- 4. sandbox + gate

GOOD = """
import geopandas as gpd
from shapely.geometry import Point
{imp}
pts = gpd.GeoDataFrame({{'id': [1, 2, 3]}},
        geometry=[Point(-87.60, 41.90), Point(-87.68, 41.85), Point(-87.72, 41.95)],
        crs='EPSG:4326')
utm = pts.to_crs('EPSG:32616')
out = calculate_buffers(utm, 25000)
out['area_km2'] = out.geometry.area / 1e6
IGUIDE_OUTPUTS = {{'total_area': {{'value': float(out['area_km2'].sum()), 'unit': 'km2'}}}}
print('total', round(out['area_km2'].sum(), 2))
"""

BAD = GOOD.replace("utm = pts.to_crs('EPSG:32616')", "utm = pts        # NOT reprojected")


def _first_exception(stderr: str) -> str:
    """The exception line from a traceback, or "" — what actually stopped the script."""
    for line in reversed((stderr or "").strip().splitlines()):
        stripped = line.strip()
        if stripped and re.match(r"^[A-Za-z_][A-Za-z0-9_.]*(Error|Exception|Exit)\b", stripped):
            return stripped[:120]
    return ""


def check_sandbox() -> None:
    section("4. sandbox, contract guard and invariant gate")
    os.environ.setdefault("AGENT_INVARIANT_GATE", "1")
    from agent_runtime.code_execution import contracts_for_code, get_code_executor
    from agent_runtime.method_library import get_contract

    contract = get_contract("calculate_buffers")
    imp = contract.get("import_line")
    if not imp:
        record("sandbox", SKIP, "calculate_buffers not in the library")
        return

    executor = get_code_executor()
    if executor.backend not in ("docker", "local"):
        record("sandbox", FAIL, f"executor disabled (backend={executor.backend}) — "
                                f"docker unavailable and it is NOT silently downgraded")
        return
    record("executor backend", PASS if executor.backend == "docker" else SKIP,
           executor.backend + ("" if executor.backend == "docker" else " — UNSANDBOXED, dev only"))

    good = GOOD.format(imp=imp)
    resolved = sorted(contracts_for_code(good))
    record("import resolves its contract", PASS if resolved else FAIL,
           f"{resolved}" if resolved else "no contract installed — the guard is silently off")

    for label, code, want in (("correct run verdicts pass", good, "pass"),
                              ("degrees buffer verdicts fail", BAD.format(imp=imp), "fail")):
        started = time.time()
        result = executor.execute(code, session="smoke", dependencies=[])
        report = result.verification or {}
        verdict = report.get("verdict")
        if report.get("error"):
            record(label, FAIL, f"the gate itself errored: {report['error'][:60]}")
            continue
        # A run that never reached the gate is not a gate result. Reporting its
        # `cannot_determine` as a gate failure sent me looking at sandbox_verify for what was
        # really an unpinned image: AGENT_CODE_EXEC_IMAGE unset falls back to python:3.11-slim,
        # which has no geopandas, so the script died on line 1 and the gate correctly said it
        # could not determine anything. Two different problems, one message.
        crashed = _first_exception(result.stderr or "")
        if crashed and not (report.get("findings") or []):
            record(label, FAIL,
                   f"the run failed BEFORE the gate — {crashed}. The gate's "
                   f"{verdict!r} is correct: nothing ran for it to inspect."
                   + ("  Set AGENT_CODE_EXEC_IMAGE to an image with the geospatial stack "
                      "(iguide-codeexec) — it is unset, so the fallback tag is in use."
                      if "ModuleNotFoundError" in crashed and not os.getenv(
                          "AGENT_CODE_EXEC_IMAGE") else ""))
            continue
        detail = f"verdict={verdict} in {time.time() - started:.0f}s"
        if verdict == want:
            record(label, PASS, detail + f" | {(result.stdout or '').strip()[:34]}")
        else:
            offenders = [f["check"] for f in (report.get("findings") or [])
                         if f.get("status") != "pass"]
            record(label, FAIL, detail + f" (wanted {want}) {offenders[:3]}")


# --------------------------------------------------------------------------- 5. the live API

def check_api(base: str, key: str) -> None:
    section("5. live API — auth and one real turn")
    try:
        import requests
    except ImportError:
        record("api", SKIP, "requests not installed")
        return

    try:
        health = requests.get(f"{base}/health", timeout=10)
        record("health", PASS if health.status_code == 200 else FAIL, str(health.status_code))
    except Exception as exc:
        record("api reachable", SKIP, f"{type(exc).__name__} — is it running?")
        return

    try:
        unauth = requests.post(f"{base}/agent/chat/stream", json={"userQuery": "hi"}, timeout=20)
        blocked = unauth.status_code in (401, 403, 500)
        record("rejects an unauthenticated call", PASS if blocked else FAIL,
               f"{unauth.status_code}" + ("" if blocked else " — the endpoint is OPEN"))
    except Exception as exc:
        record("rejects an unauthenticated call", SKIP, type(exc).__name__)

    if not key:
        record("one real turn", SKIP, "no API key given (--key)")
        return
    started = time.time()
    try:
        resp = requests.post(f"{base}/agent/chat/stream",
                             headers={"X-API-KEY": key, "Content-Type": "application/json"},
                             json={"userQuery": "Reply with exactly: OK", "session_id": "smoke"},
                             timeout=600, stream=True)
        body = "".join(chunk for chunk in resp.iter_content(chunk_size=None, decode_unicode=True))
    except Exception as exc:
        record("one real turn", FAIL, f"{type(exc).__name__}: {exc}"[:70])
        return
    elapsed = time.time() - started
    if '"error"' in body and "event: answer" not in body:
        reason = body[body.find('"error"'):][:90]
        record("one real turn", FAIL, f"stream errored after {elapsed:.0f}s: {reason}")
    elif "event: result" in body or "event: answer" in body:
        record("one real turn", PASS, f"answered in {elapsed:.0f}s")
    else:
        record("one real turn", FAIL, f"stream ended with no answer after {elapsed:.0f}s")


# --------------------------------------------------------------------------- 6. reproducibility

def check_replay() -> None:
    section("6. reproducibility — can a recorded run be re-run")
    manifests = sorted(REPO.glob("**/artifact/manifest.json"))
    if not manifests:
        record("artifact replay", SKIP,
               "no artifact recorded yet — run a code-executing turn first, then re-run this")
        return
    latest = manifests[-1]
    data = json.loads(latest.read_text())
    record("artifact records an image digest", PASS if data.get("image_digest") else FAIL,
           str(data.get("image_digest"))[:52])
    record("artifact records declared values", PASS if data.get("declared_outputs") else FAIL,
           f"{len(data.get('declared_outputs') or {})} declared output(s)")
    proc = subprocess.run([sys.executable, "scripts/rerun_artifact.py", str(latest.parent)],
                          capture_output=True, text=True, cwd=str(REPO), timeout=900)
    record("replay reproduces the numbers", PASS if proc.returncode == 0 else FAIL,
           (proc.stdout or proc.stderr or "").strip().splitlines()[-1][:64] if
           (proc.stdout or proc.stderr) else f"exit {proc.returncode}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true", help="skip the cluster and the live API")
    ap.add_argument("--api", default=os.getenv("SMOKE_API", "http://127.0.0.1:3500"))
    ap.add_argument("--key", default=os.getenv("AGENT_CHAT_API_KEY", ""))
    ap.add_argument("--skip-sandbox", action="store_true", help="skip the docker checks")
    args = ap.parse_args()

    # .env carries the cluster and the token. Loaded with python-dotenv, never by sourcing:
    # the password contains a shell metacharacter and `set -a; . .env` expands it.
    for candidate in (REPO / ".env", Path("/Users/yfkang/i-guide-platform-flask-servers/.env")):
        if candidate.is_file():
            try:
                from dotenv import load_dotenv

                load_dotenv(candidate, override=False)
                break
            except ImportError:
                pass

    print(f"I-GUIDE end-to-end smoke test — {REPO}")
    check_config(args.offline)
    check_library()
    check_retrieval(args.offline)
    if not args.skip_sandbox:
        check_sandbox()
    if not args.offline:
        check_api(args.api, args.key)
    check_replay()

    failed = [r for r in _results if r[1] == FAIL]
    skipped = [r for r in _results if r[1] == SKIP]
    print("\n" + "=" * 78)
    print(f"{len(_results) - len(failed) - len(skipped)} passed, {len(failed)} failed, "
          f"{len(skipped)} skipped")
    if skipped:
        # Stated, not buried. A skipped check is an unanswered question, and the whole point of
        # this script is that an unanswered question must not look like a satisfied one.
        print("\nNOT CHECKED (each of these is unverified, not fine):")
        for stage, _s, detail in skipped:
            print(f"  - {stage}: {detail}")
    if failed:
        print("\nFAILED:")
        for stage, _s, detail in failed:
            print(f"  - {stage}: {detail}")
    return len(failed)


if __name__ == "__main__":
    raise SystemExit(main())
