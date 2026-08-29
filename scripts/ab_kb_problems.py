#!/usr/bin/env python3
"""Does the extracted knowledge base change what the agent gets RIGHT?

    python scripts/ab_kb_problems.py --all
    python scripts/ab_kb_problems.py --problem p1_streets --arm with_kb

Four analysis problems over two datasets that are genuinely in the corpus and genuinely
cached, run twice each. The two arms differ in **exactly one thing** — whether
`agent_kb_search` and `kb_method_search` are in `enabled_search_methods`. Same model, same
prompts, same sandbox, same platform search, same web access, same staging tools. Both arms can
reach the same bytes; only one can see what extraction found inside them.

Grading is against numbers computed independently by `scripts/ab_kb_ground_truth.py`, not
against a judgement of the prose. Two of the problems have **diagnostic wrong answers**: for
p1_streets, 47,634 is what you get by taking the buffer as feet in the streets' own EPSG:3435
and 68,419 by buffering in degrees, so a reported count says which mistake was made rather than
merely that one was. That is the point — "the agent did worse" is not a finding, "the agent
measured in feet" is.

Development only: `LLM_PROVIDER=claude-cli` keeps a run at zero API spend, and the model is
recorded in every result because a number produced under a different model is not comparable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Set before the runtime is imported.
os.environ.setdefault("AGENT_FILE_STORAGE_ROOT", str(REPO / "agent_chat_files"))
os.environ.setdefault("AGENT_KB_BACKEND", "local")
os.environ["AGENT_CODE_EXEC"] = "1"
os.environ.setdefault("AGENT_CODE_EXEC_BACKEND", "docker")
# Stays off. It routes ingested notebook source into a bare exec() in a credentialed process.
os.environ["AGENT_ALLOW_WORKFLOW_EXEC"] = "0"
os.environ.setdefault("AGENT_SKILLS_ENABLED", "0")
os.environ.setdefault("LLM_PROVIDER", "claude-cli")
os.environ.setdefault("CLAUDE_CLI_MODEL", "sonnet")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env", override=False)

# Everything both arms get. Platform search, the open web, code execution and staging are NOT
# the variable under test: without them the no-KB arm could not reach the data at all and the
# experiment would measure file access rather than extracted knowledge.
COMMON_TOOLS = ["keyword_search", "semantic_search", "spatial_search", "neo4j_search",
                "web_search"]
# The variable. Readers (`get_kb_block`, `get_method_contract`) follow their search tool
# automatically — see the companion_of map in langchain_granular_tools.
KB_TOOLS = ["agent_kb_search", "kb_method_search"]

ARMS = {"no_kb": COMMON_TOOLS, "with_kb": COMMON_TOOLS + KB_TOOLS}

PROBLEMS: Dict[str, Dict[str, Any]] = {
    "p1_streets": {
        "query": (
            "Using the I-GUIDE platform's Chicago Crime data 2026 dataset (element 265e6957) "
            "and the Chicago Major Streets dataset (element b32bec3e), determine how many "
            "crime incidents occurred within 100 metres of a major street. Report the count, "
            "the total number of incidents you measured against, and the CRS you performed "
            "the distance measurement in."),
        "why": ("The streets are stored in EPSG:3435 — Illinois State Plane, US survey FEET — "
                "and the crime points in EPSG:4326 degrees. Neither is safe to buffer in."),
        "checks": [
            {"label": "crimes within 100 m", "correct": 72658, "tolerance": 0.02,
             "wrong": {47634: "measured in feet in EPSG:3435",
                       68419: "buffered in degrees"}},
            {"label": "points measured against", "correct": 128855, "tolerance": 0.01,
             "wrong": {20000: "read only the first 20,000 rows"}},
        ],
    },
    "p2_arrest_shift": {
        "query": (
            "Using the I-GUIDE platform's Chicago Crime data 2026 dataset (element 265e6957), "
            "report the five most common primary crime types with their counts. Then report "
            "the five most common primary crime types among only those incidents that "
            "resulted in an arrest, and say how the ranking changes."),
        "why": "Two rankings over the same file; the arrest subset reorders them.",
        "checks": [
            {"label": "THEFT overall", "correct": 27824, "tolerance": 0.01, "wrong": {}},
            {"label": "BATTERY overall", "correct": 23885, "tolerance": 0.01, "wrong": {}},
            {"label": "BATTERY among arrests", "correct": 4532, "tolerance": 0.01, "wrong": {}},
            {"label": "NARCOTICS among arrests", "correct": 3439, "tolerance": 0.01,
             "wrong": {}},
        ],
    },
    "p3_busiest_month": {
        "query": (
            "Using the I-GUIDE platform's Chicago Crime data 2026 dataset (element 265e6957), "
            "identify the calendar month with the most reported incidents, and report the "
            "three most common primary crime types within that month with their counts."),
        "why": "Date parsing plus a grouped ranking; the file spans 2026-01 to 2026-07.",
        "checks": [
            {"label": "incidents in the busiest month", "correct": 20991, "tolerance": 0.01,
             "wrong": {}},
            {"label": "THEFT that month", "correct": 4599, "tolerance": 0.01, "wrong": {}},
        ],
    },
    "p4_negative_control": {
        "query": (
            "Using only methods that already exist in the I-GUIDE platform's extracted method "
            "library, compute a monthly NDVI time series for the Amazon basin from Sentinel-2 "
            "imagery for 2024. If the library does not contain what is needed, say so "
            "explicitly rather than substituting something else."),
        "why": ("The corpus has no Sentinel-2 NDVI method. A KB that makes the agent claim "
                "reuse it cannot support is worse than no KB."),
        "checks": [],
        "expect_refusal": True,
    },
}


def _numbers(text: str) -> List[int]:
    """Every integer in the answer, with thousands separators normalised."""
    return [int(m.replace(",", "")) for m in re.findall(r"\d[\d,]{2,}", text or "")]


def grade(problem: Dict[str, Any], answer: str) -> Dict[str, Any]:
    found = set(_numbers(answer))
    results = []
    for check in problem.get("checks") or []:
        correct = check["correct"]
        tol = max(1, int(correct * check.get("tolerance", 0.01)))
        hit = any(abs(n - correct) <= tol for n in found)
        diagnosed = [why for wrong, why in (check.get("wrong") or {}).items()
                     if any(abs(n - wrong) <= max(1, int(wrong * 0.005)) for n in found)]
        results.append({"label": check["label"], "correct": correct, "found": hit,
                        "diagnosis": diagnosed})
    out: Dict[str, Any] = {"checks": results,
                           "passed": sum(1 for r in results if r["found"]),
                           "total": len(results)}
    if problem.get("expect_refusal"):
        low = (answer or "").lower()
        refused = any(p in low for p in (
            "does not contain", "no such method", "not available in the library",
            "library does not", "no method", "cannot be done with", "not present",
            "does not have", "no ndvi", "unable to find"))
        out["refused_correctly"] = bool(refused)
    return out


def run_one(problem_id: str, arm: str, timeout_note: str = "") -> Dict[str, Any]:
    from agent_runtime.graph_runtime import run_agent_query

    problem = PROBLEMS[problem_id]
    t0 = time.time()
    error = None
    res: Dict[str, Any] = {}
    try:
        res = run_agent_query(
            problem["query"],
            use_supervisor=True,
            code_exec=True,
            include_mcp_tools=False,
            enabled_search_methods=ARMS[arm],
            # A fresh thread per (problem, arm): a shared checkpointer would let the second arm
            # read the first one's artifacts and turn an ablation into a continuation.
            thread_id=f"ab_{problem_id}_{arm}_{int(t0)}",
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"[:400]
    elapsed = time.time() - t0

    answer = str(res.get("final_answer") or "")
    trace = res.get("route_trace") or {}
    called = trace.get("called_tools") if isinstance(trace, dict) else None
    called = [str(c) for c in (called or [])]
    return {
        "problem": problem_id,
        "arm": arm,
        "model": os.getenv("CLAUDE_CLI_MODEL", "?"),
        "provider": os.getenv("LLM_PROVIDER", "?"),
        "elapsed_s": round(elapsed, 1),
        "error": error,
        "final_answer": answer,
        "called_tools": called,
        # Did the arm actually USE the thing under test? A with_kb run that never called a KB
        # tool is not evidence about the KB, and averaging it in would hide that.
        "used_kb_tools": sorted({c for c in called
                                 if c in {"agent_kb_search", "get_kb_block",
                                          "kb_method_search", "get_method_contract"}}),
        "ran_code": any(c == "execute_code" for c in called),
        "staged": any(c.startswith("stage_") for c in called),
        "grounding_audit": res.get("grounding_audit"),
        "grade": grade(problem, answer),
        "note": timeout_note,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", action="append", choices=sorted(PROBLEMS))
    ap.add_argument("--arm", action="append", choices=sorted(ARMS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=str(REPO / "outputs" / "ab_kb_problems.json"))
    args = ap.parse_args()

    problems = args.problem or (sorted(PROBLEMS) if args.all else ["p1_streets"])
    arms = args.arm or ["no_kb", "with_kb"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if out_path.is_file():
        try:
            existing = json.loads(out_path.read_text())["runs"]
        except (OSError, ValueError, KeyError):
            existing = []

    runs = list(existing)
    for problem_id in problems:
        for arm in arms:
            print(f"\n{'=' * 78}\n{problem_id}  [{arm}]  tools={ARMS[arm]}\n{'=' * 78}",
                  flush=True)
            row = run_one(problem_id, arm)
            runs = [r for r in runs
                    if not (r["problem"] == problem_id and r["arm"] == arm)] + [row]
            grade_row = row["grade"]
            print(f"  {row['elapsed_s']}s   tools={len(row['called_tools'])}   "
                  f"kb={row['used_kb_tools'] or '-'}   code={row['ran_code']}", flush=True)
            if row["error"]:
                print(f"  ERROR {row['error']}", flush=True)
            for check in grade_row["checks"]:
                mark = "OK  " if check["found"] else "MISS"
                extra = f"   <- {'; '.join(check['diagnosis'])}" if check["diagnosis"] else ""
                print(f"  {mark} {check['label']:<34}{check['correct']:>10,}{extra}", flush=True)
            if "refused_correctly" in grade_row:
                print(f"  {'OK  ' if grade_row['refused_correctly'] else 'MISS'} "
                      f"refused a capability it does not have", flush=True)
            out_path.write_text(json.dumps({"runs": runs}, indent=2, default=str) + "\n")

    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
