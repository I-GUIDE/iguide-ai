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
import functools
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

def configure() -> None:
    """Environment for a real run. Called from `main`, NEVER at import.

    This used to run at module scope, which made importing the harness — as a test does, to
    check the instrumentation — load the platform `.env` into the whole pytest process and set
    OPENSEARCH_NODE, NEO4J_* and the rest. Two unrelated tests that assert what happens when a
    lookup FAILS then passed alone and failed in a full run, because the credentials they assume
    are absent had been supplied by an import three files away.
    """
    from dotenv import load_dotenv

    os.environ.setdefault("AGENT_FILE_STORAGE_ROOT", str(REPO / "agent_chat_files"))
    os.environ["AGENT_CODE_EXEC"] = "1"
    os.environ.setdefault("AGENT_CODE_EXEC_BACKEND", "docker")
    # Stays off. It routes ingested notebook source into a bare exec() in a credentialed process.
    os.environ["AGENT_ALLOW_WORKFLOW_EXEC"] = "0"
    os.environ.setdefault("AGENT_SKILLS_ENABLED", "0")
    os.environ.setdefault("LLM_PROVIDER", "claude-cli")
    os.environ.setdefault("CLAUDE_CLI_MODEL", "sonnet")

    # This worktree has no `.env`; the platform credentials live in the main checkout. Without
    # them OPENSEARCH_NODE, FLASK_EMBEDDING_URL and Neo4j are unset, the supervisor's search peer
    # fails, and BOTH arms degrade to a partial answer — which would be recorded as a result.
    # override=False so an explicitly exported variable still wins.
    for candidate in (REPO / ".env", REPO.parent / "i-guide-platform-flask-servers" / ".env"):
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            break

    # Inherited from a DEPLOYMENT .env and wrong locally: every download_url would be built
    # against the remote host, so a locally produced artifact 404s with `unknown file_id`.
    os.environ["AGENT_PUBLIC_BASE_URL"] = ""
    # The platform's OpenSearch is the WEBSITE's cluster. Both arms read it for keyword/semantic/
    # spatial search, which is fair and read-only; the agent KB is served from the local Postgres
    # record, so nothing in this experiment writes to the platform.
    os.environ["AGENT_KB_BACKEND"] = "local"


# Everything both arms get. Platform search, the open web, code execution and staging are NOT
# the variable under test: without them the no-KB arm could not reach the data at all and the
# experiment would measure file access rather than extracted knowledge.
COMMON_TOOLS = ["keyword_search", "semantic_search", "spatial_search", "neo4j_search",
                "web_search"]
KB_TOOLS = ["agent_kb_search", "kb_method_search"]

# `enabled_search_methods` alone is NOT the ablation, and building on it cost a whole sweep. It
# filters the search peer; the code and analyze peers are granted the KB tools "deliberately
# independent of the request's enabled_search_methods", and `_direct_search_sweep` unions the KB
# in without the model electing anything. The no-KB arm called `agent_kb_search` once and
# answered both checks correctly — a control that was really a second treatment, reporting
# success. `AGENT_ABLATE_KB` closes all four paths; see `supervisor.graph.kb_ablated`.
ARMS = {
    "no_kb":   {"tools": COMMON_TOOLS, "ablate": True},
    "with_kb": {"tools": COMMON_TOOLS + KB_TOOLS, "ablate": False},
}

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


class _ToolCounter:
    """Counts calls to the tools under test, by wrapping them where they are defined.

    `route_trace.called_tools` records the SUPERVISOR's peer delegations —
    `search_agent_evidence`, `code_agent_answer` — not the tools those peers then call. Reading
    it for "did this arm use the knowledge base" gave `kb=-` on a run whose answer was exactly
    right, which is the wrong layer entirely: the same gap the plan records as "search-peer
    inner tool-call logging".

    Patching the module attribute is enough because `Tool(func=…)` resolves it when the graph is
    built, and the graph is built per run.
    """

    NAMES = ("agent_kb_search_tool", "get_kb_block_tool", "kb_method_search_tool",
             "get_method_contract_tool")

    def __init__(self):
        self.calls: Dict[str, int] = {}
        self._saved: Dict[str, Any] = {}

    def _wrap(self, module, attr):
        """`functools.wraps`, and it is load-bearing — not tidiness.

        LangChain infers a tool's argument schema from the wrapped function's SIGNATURE. A
        hand-rolled wrapper copying only `__name__` and `__doc__` left `(*a, **kw)`, so
        `agent_kb_search` was advertised to the model with no parameters at all and the peer
        turn failed the moment it tried to call one. Only the with-KB arm calls these tools, so
        only that arm broke: 8/8 versus 0/8, a perfectly clean result produced entirely by the
        instrumentation. `functools.wraps` sets `__wrapped__`, which `inspect.signature` follows.
        """
        original = getattr(module, attr)

        @functools.wraps(original)
        def counted(*a, **kw):
            self.calls[attr] = self.calls.get(attr, 0) + 1
            return original(*a, **kw)

        setattr(module, attr, counted)
        self._saved[attr] = (module, original)

    def __enter__(self):
        from agent_runtime import langchain_granular_tools

        for attr in self.NAMES:
            if hasattr(langchain_granular_tools, attr):
                self._wrap(langchain_granular_tools, attr)
        # `execute_code` is a closure inside `make_code_execution_tools`, so there is no module
        # attribute to wrap. It does not need one: the orchestration result carries `code_result`
        # when the sandbox ran, which is a fact about the run rather than an inference from it.
        return self

    def __exit__(self, *exc):
        for attr, (module, original) in self._saved.items():
            setattr(module, attr, original)
        return False


class _WarnCounter:
    """Counts the two degradations that would otherwise be invisible in a result.

    `claude-cli` tool calling is prompt-enforced, not API-enforced, so a turn can come back as
    prose and the envelope leaks into the answer. `ChatClaudeCli.malformed_replies` exists for
    this but `bind_tools` returns a NEW instance, so the counter is scattered across objects and
    unreadable from here. The log line is the reliable signal.

    A run with malformed replies is not evidence about the knowledge base — it is evidence about
    the shim — and averaging it in would let a harness artefact read as a KB result.
    """

    def __init__(self):
        self.malformed = 0
        self.search_failed = 0

    def __enter__(self):
        import logging

        self._handler = logging.Handler()
        self._handler.emit = self._emit
        logging.getLogger().addHandler(self._handler)
        logging.getLogger().setLevel(logging.WARNING)
        return self

    def _emit(self, record):
        text = str(getattr(record, "msg", ""))
        if "was not JSON" in text:
            self.malformed += 1
        if "peer search failed" in text:
            self.search_failed += 1

    def __exit__(self, *exc):
        import logging

        logging.getLogger().removeHandler(self._handler)
        return False


def run_one(problem_id: str, arm: str, timeout_note: str = "") -> Dict[str, Any]:
    from agent_runtime.graph_runtime import run_agent_query

    problem = PROBLEMS[problem_id]
    os.environ["AGENT_ABLATE_KB"] = "1" if ARMS[arm]["ablate"] else "0"
    t0 = time.time()
    error = None
    res: Dict[str, Any] = {}
    counter = _WarnCounter()
    tools_used = _ToolCounter()
    try:
      with counter, tools_used:
        res = run_agent_query(
            problem["query"],
            use_supervisor=True,
            code_exec=True,
            include_mcp_tools=False,
            enabled_search_methods=ARMS[arm]["tools"],
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
        "ablated": ARMS[arm]["ablate"],
        "model": os.getenv("CLAUDE_CLI_MODEL", "?"),
        "provider": os.getenv("LLM_PROVIDER", "?"),
        "elapsed_s": round(elapsed, 1),
        "error": error,
        "final_answer": answer,
        "called_tools": called,
        # Did the arm actually USE the thing under test? A with_kb run that never called a KB
        # tool is not evidence about the KB, and averaging it in would hide that. Counted at the
        # tool itself, because `called_tools` records peer delegations one layer up — reading it
        # reported `kb=-` on a run whose answer was exactly right.
        "kb_tool_calls": dict(tools_used.calls),
        "used_kb": bool(tools_used.calls),
        "ran_code": bool((res.get("orchestration_result") or {}).get("code_result")),
        "grounding_audit": res.get("grounding_audit"),
        # Harness health, kept beside the result so a shim failure is never read as a KB result.
        "malformed_llm_replies": counter.malformed,
        "peer_search_failures": counter.search_failed,
        "grade": grade(problem, answer),
        "note": timeout_note,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", action="append", choices=sorted(PROBLEMS))
    ap.add_argument("--arm", action="append", choices=sorted(ARMS))
    ap.add_argument("--all", action="store_true")
    # Replicates, because one run per cell cannot tell an effect from the shim: the same
    # arm on the same problem passed 2/2 and then missed 2/2, and the difference was six
    # malformed LLM replies rather than anything about the knowledge base.
    ap.add_argument("--replicates", type=int, default=1)
    ap.add_argument("--out", default=str(REPO / "outputs" / "ab_kb_problems.json"))
    args = ap.parse_args()
    configure()

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
    for rep in range(max(1, args.replicates)):
      for problem_id in problems:
        for arm in arms:
            print(f"\n{'=' * 78}\n{problem_id}  [{arm}]  "
                  f"ablate_kb={ARMS[arm]['ablate']}  tools={ARMS[arm]['tools']}\n{'=' * 78}",
                  flush=True)
            row = run_one(problem_id, arm)
            row["replicate"] = rep
            runs = [r for r in runs
                    if not (r["problem"] == problem_id and r["arm"] == arm
                            and r.get("replicate") == rep)] + [row]
            grade_row = row["grade"]
            print(f"  {row['elapsed_s']}s   tools={len(row['called_tools'])}   "
                  f"kb={row['kb_tool_calls'] or '-'}   code={row['ran_code']}",
                  flush=True)
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
