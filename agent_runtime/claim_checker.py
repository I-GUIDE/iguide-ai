"""Check the auditor's disputed claims against the data the turn produced, with code.

The grounding auditor (an LLM) reads the answer beside an EXCERPT of the turn's record, and
reconciliation (rule-based) withdraws what it can prove from that record: numbers that appear in
it, files and maps that were produced, notes the turn was shown. What survives both is shown to
the user as a problem. Two kinds of claim survive wrongly, in opposite directions:

* A RANKING. "Highest-rate community area: Fuller Park" (Chicago demo, step 1, 2026-10-15) was
  right: Fuller Park is the maximum of all 77 rates. The auditor saw only the top five printed,
  and no rule handles a claim without a number, so a correct answer carried a warning.
* A MEMBERSHIP. "Region 1 includes Austin" and 77 more (step 3) named real areas, but several
  were in the wrong region: the regions layer had no names and the agent rebuilt them in code.
  A rule that cleared any claim whose names appear in the record would have passed them.

Neither can be settled by reading. Both can by computing: the maximum of a column, the region a
polygon falls in. So for each disputed claim a model WRITES a check against the turn's own files,
the sandbox RUNS it, and the result decides (the tool-interactive pattern, CRITIC, arXiv
2305.11738): supported claims are withdrawn, contradicted ones stay with the computed evidence,
and a claim that cannot be checked is reported as not verified rather than as a problem.

The checks run in the code sandbox (no network), like any code the agent writes. Off unless
AGENT_CLAIM_CHECK=1.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

ENABLE_ENV = "AGENT_CLAIM_CHECK"
MAX_CLAIMS = int(os.getenv("AGENT_CLAIM_CHECK_MAX_CLAIMS", "80"))
MAX_FILES = 8
MAX_FILE_BYTES = 80 * 1024 * 1024
TABLE_EXT = (".csv", ".geojson", ".json", ".gpkg", ".parquet")
RESULT_MARK = "CLAIMCHECK_JSON="
_FILE_ID_RE = re.compile(r"\bfile_[0-9a-f]{12}\b")

WRITER_PROMPT = """You verify claims from a GIS analysis by writing Python checks against the data
files the analysis produced or was given. You do NOT judge from memory or by reading: every
verdict must come from computing on these files.

Files in the working directory (read them by these exact names):
{files}

Claims to check (each was disputed by a reviewer who could not see the full data):
{claims}

Write JSON only, of this shape:
{{"setup": "<python run once: imports and loading the files you need into variables>",
  "checks": [{{"i": <claim number>, "checkable": true|false, "why": "<only when not checkable>",
              "code": "<python that sets ok = True/False and detail = '<the values compared>'>"}}]}}

Rules:
- pandas, geopandas, numpy and shapely are available; there is no network.
- Each check's code sets two variables: `ok` (True if the data supports the claim, False if it
  contradicts it) and `detail` (one short sentence with the actual values you compared).
- Compare names after upper-casing and removing everything but letters and digits: data often
  spells a name differently from prose ("O'Hare" is "OHARE" in the Chicago boundaries).
- A claim about a ranking ("highest", "most", "lowest") is checked over ALL rows, not a sample.
- If a layer has no name column, attach names by location from a file that has them (spatial join
  on each polygon's representative point).
- Claims of the same form (e.g. many "Region k includes X") should share helpers defined in setup.
- If the files cannot settle a claim, set "checkable": false and say why. Do not guess.
"""


def enabled() -> bool:
    return (os.getenv(ENABLE_ENV) or "").strip().lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------- the turn's files

def turn_files(sources: Sequence[Any], thread_id: Optional[str] = None,
               resolve: Optional[Callable[[str], Tuple[Optional[Path], Optional[str]]]] = None
               ) -> List[Dict[str, Any]]:
    """Tabular and vector files the turn produced (ids named in its results) or was given
    (the conversation's uploads), resolved to local paths. Newest ids last; capped."""
    ids: List[str] = []
    try:
        from agent_runtime.session_memory import get_session_files

        ids.extend(get_session_files(thread_id) or [])
    except Exception:  # noqa: BLE001
        pass
    for src in sources:
        try:
            ids.extend(_FILE_ID_RE.findall(json.dumps(src, default=str)))
        except Exception:  # noqa: BLE001
            continue
    resolve = resolve or _resolve
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for fid in ids:
        if fid in seen:
            continue
        seen.add(fid)
        path, name = resolve(fid)
        if not path or not Path(path).is_file():
            continue
        name = name or Path(path).name
        if not name.lower().endswith(TABLE_EXT) or Path(path).stat().st_size > MAX_FILE_BYTES:
            continue
        out.append({"file_id": fid, "name": _safe_name(name, len(out)), "path": str(path)})
    return out[-MAX_FILES:]


def _resolve(fid: str) -> Tuple[Optional[Path], Optional[str]]:
    try:
        from agent_runtime.file_store import get_file_record, resolve_file_id

        rec = get_file_record(fid) or {}
        return resolve_file_id(fid), rec.get("filename")
    except Exception:  # noqa: BLE001
        return None, None


def _safe_name(name: str, k: int) -> str:
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name).name).strip("._") or f"file{k}"
    return f"{k + 1:02d}_{base}"


def describe(files: Sequence[Dict[str, Any]], max_cols: int = 30) -> str:
    """What the check writer sees of each file: shape, columns with types, three rows."""
    lines = []
    for f in files:
        try:
            p = f["path"]
            if p.lower().endswith(".csv"):
                import pandas as pd

                df = pd.read_csv(p, nrows=200000)
                geom = ""
            else:
                import geopandas as gpd

                df = gpd.read_file(p)
                geom = f", geometry {sorted(set(df.geom_type.dropna()))}, crs {df.crs}"
            cols = [c for c in df.columns if c != "geometry"][:max_cols]
            types = ", ".join(f"{c}:{str(df[c].dtype)}" for c in cols)
            sample = df[cols].head(3).to_dict(orient="records")
            lines.append(f"- {f['name']}: {len(df)} rows{geom}; columns {types}; first rows "
                         f"{json.dumps(sample, default=str)[:600]}")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"- {f['name']}: could not be read ({type(exc).__name__})")
    return "\n".join(lines)


# --------------------------------------------------------------------------- write and run

def write_checks(claims: Sequence[str], files: Sequence[Dict[str, Any]], llm: Any) -> Dict[str, Any]:
    prompt = WRITER_PROMPT.format(
        files=describe(files),
        claims="\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims)))
    raw = llm.invoke(prompt)
    text = getattr(raw, "content", raw)
    if isinstance(text, list):
        text = " ".join(str(p.get("text", p)) if isinstance(p, dict) else str(p) for p in text)
    from agent_runtime.evidence_quality import _extract_json_object

    plan = _extract_json_object(str(text)) or {}
    return plan if isinstance(plan, dict) else {}


def build_script(plan: Dict[str, Any]) -> str:
    """One program: the setup once, then every check in its own try, results as one JSON line."""
    checks = [c for c in plan.get("checks") or [] if isinstance(c, dict) and c.get("checkable")
              and str(c.get("code") or "").strip()]
    payload = json.dumps({"setup": str(plan.get("setup") or ""),
                          "checks": [{"i": int(c.get("i") or 0), "code": str(c["code"])}
                                     for c in checks]})
    return f'''import json
_P = json.loads({payload!r})
_G = {{"__name__": "__claimcheck__"}}
_R = {{}}
try:
    exec(_P["setup"], _G)
    _setup_error = None
except Exception as _e:
    _setup_error = f"{{type(_e).__name__}}: {{_e}}"[:300]
for _c in _P["checks"]:
    if _setup_error:
        _R[str(_c["i"])] = {{"error": "setup failed: " + _setup_error}}
        continue
    _L = {{}}
    try:
        exec(_c["code"], _G, _L)
        _ok = _L.get("ok", _G.get("ok"))
        _R[str(_c["i"])] = {{"ok": None if _ok is None else bool(_ok),
                             "detail": str(_L.get("detail", _G.get("detail", "")))[:160]}}
    except Exception as _e:
        _R[str(_c["i"])] = {{"error": f"{{type(_e).__name__}}: {{_e}}"[:200]}}
# One line per claim: the sandbox clips long output, and a single JSON line clipped anywhere
# lost every verdict (77 of 77 "unverified" in one run, 2026-10-15). Clipped, this loses the tail.
for _i, _v in _R.items():
    print({RESULT_MARK!r} + json.dumps({{_i: _v}}))
'''


def run_checks(plan: Dict[str, Any], files: Sequence[Dict[str, Any]],
               executor: Any = None, timeout: int = 180) -> Dict[str, Dict[str, Any]]:
    if executor is None:
        from agent_runtime.code_execution import get_code_executor

        executor = get_code_executor()
    script = build_script(plan)
    res = executor.execute(script, timeout=timeout, label="claim check",
                           input_files=[{"source": f["path"], "dest": f["name"]} for f in files])
    out = getattr(res, "stdout", "") or ""
    results: Dict[str, Dict[str, Any]] = {}
    for line in out.splitlines():
        if not line.startswith(RESULT_MARK):
            continue
        try:
            results.update(json.loads(line[len(RESULT_MARK):]))
        except ValueError:
            continue                       # a line cut by the output clip: that claim only
    if not results:
        err = getattr(res, "error", None) or (getattr(res, "stderr", "") or "")[-300:]
        return {"_run_error": {"error": str(err) or "no result from the check run"}}
    return results


# --------------------------------------------------------------------------- verdicts

def check_claims(claims: Sequence[str], files: Sequence[Dict[str, Any]], llm: Any,
                 executor: Any = None) -> List[Dict[str, Any]]:
    """One verdict per claim: supported | contradicted | unverified, with the evidence."""
    claims = list(claims)[:MAX_CLAIMS]
    if not claims:
        return []
    if not files:
        return [{"claim": c, "verdict": "unverified", "detail": "the turn left no data file "
                 "to check it against"} for c in claims]
    try:
        plan = write_checks(claims, files, llm)
    except Exception as exc:  # noqa: BLE001 - a failed writer leaves every claim unverified
        logger.warning("claim check: writer failed: %s", exc)
        plan = {}
    by_i = {int(c.get("i") or 0): c for c in plan.get("checks") or [] if isinstance(c, dict)}
    results = run_checks(plan, files, executor) if any(
        c.get("checkable") for c in by_i.values()) else {}
    run_error = (results.get("_run_error") or {}).get("error")
    verdicts = []
    for i, claim in enumerate(claims, 1):
        spec, got = by_i.get(i), results.get(str(i)) or {}
        if not spec or not spec.get("checkable"):
            why = (spec or {}).get("why") or "no check could be written for it"
            verdicts.append({"claim": claim, "verdict": "unverified", "detail": str(why)[:300]})
        elif got.get("ok") is True:
            verdicts.append({"claim": claim, "verdict": "supported", "detail": got.get("detail", "")})
        elif got.get("ok") is False:
            verdicts.append({"claim": claim, "verdict": "contradicted",
                             "detail": got.get("detail", ""), "code": spec.get("code")})
        else:
            verdicts.append({"claim": claim, "verdict": "unverified",
                             "detail": got.get("error") or run_error or "the check did not run"})
    return verdicts


def apply(audit: Optional[Dict[str, Any]], verdicts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The audit with supported claims withdrawn, contradicted ones carrying their evidence, and
    unverifiable ones marked so the banner calls them not verified, not a problem."""
    audit = dict(audit or {})
    by_claim = {v["claim"]: v for v in verdicts}
    kept, withdrawn = [], []
    for it in audit.get("issues") or []:
        claim = str(it.get("claim") if isinstance(it, dict) else it)
        v = by_claim.get(claim)
        if v is None:
            kept.append(it)
        elif v["verdict"] == "supported":
            withdrawn.append({"claim": claim[:200], "detail": v["detail"]})
        elif v["verdict"] == "contradicted":
            kept.append({**(it if isinstance(it, dict) else {"claim": claim}),
                         "reason": f"checked against the turn's data: {v['detail']}",
                         "source": "claim_check", "status": "fail"})
        else:
            kept.append({**(it if isinstance(it, dict) else {"claim": claim}),
                         "source": "claim_check", "status": "unverified",
                         "check_note": v["detail"]})
    if withdrawn:
        logger.info("claim check withdrew %d issue(s): %s", len(withdrawn), withdrawn[:10])
    # Claims the auditor could not see: checked like the rest. Supported ones are dropped,
    # contradicted ones become issues (the auditor did not raise them, the check did).
    still_unseen = []
    for claim in audit.get("unverified_claims") or []:
        v = by_claim.get(str(claim))
        if v is None or v["verdict"] == "unverified":
            still_unseen.append(claim)
        elif v["verdict"] == "contradicted":
            kept.append({"claim": str(claim), "reason": f"checked against the turn's data: {v['detail']}",
                         "source": "claim_check", "status": "fail"})
        else:
            withdrawn.append({"claim": str(claim)[:200], "detail": v["detail"]})
    audit["unverified_claims"] = still_unseen
    real = [k for k in kept if not (isinstance(k, dict) and k.get("status") == "unverified")]
    out = {**audit, "issues": kept, "claim_check": {
        "supported": len(withdrawn), "contradicted": sum(1 for v in verdicts if v["verdict"] == "contradicted"),
        "unverified": sum(1 for v in verdicts if v["verdict"] == "unverified")}}
    if not real:
        out["hallucination_detected"] = False
        if not kept:
            out["severity"] = "none"
    elif any(isinstance(k, dict) and k.get("source") == "claim_check" and k.get("status") == "fail"
             for k in kept):
        # A contradiction found by computing is a problem whatever the auditor rated the turn.
        out["hallucination_detected"] = True
        out["severity"] = "high"
    return out


def claims_to_check(audit: Optional[Dict[str, Any]]) -> List[str]:
    """The auditor's own disputed claims, and the ones it could not see (``unverified_claims``,
    "unseen" in its ledger: their part of the record was cut). Not the invariant gate's, which
    is deterministic already."""
    out = [str(c).strip() for c in (audit or {}).get("unverified_claims") or [] if str(c).strip()]
    for it in (audit or {}).get("issues") or []:
        if isinstance(it, dict) and it.get("source") in ("invariant_gate", "claim_check"):
            continue
        claim = str(it.get("claim") if isinstance(it, dict) else it).strip()
        if claim:
            out.append(claim)
    return out


__all__ = ["ENABLE_ENV", "apply", "build_script", "check_claims", "claims_to_check", "describe",
           "enabled", "run_checks", "turn_files", "write_checks"]
