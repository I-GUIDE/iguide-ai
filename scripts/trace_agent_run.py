#!/usr/bin/env python3
"""Record every step an agent turn takes, with and without the knowledge base.

    python scripts/trace_agent_run.py --case flood --arm both
    python scripts/trace_agent_run.py --case e2sfca --arm with_kb

The A/B harness records the ANSWER and counts of KB calls. That is enough to score a run and
useless for understanding one: it cannot say what the agent searched for, what came back, which
method it read the contract of, or what it ran. Every behavioural claim in the last few days —
"it searched the library, understood it, and wrote its own code anyway" — was inferred from
counts and answer text rather than observed.

So this wraps ``BaseTool.run``, which is the one hook every tool passes through no matter how it
was constructed: the granular tools are module-level functions, ``execute_code`` is a closure
inside a factory, and the staging tools are built per session. Wrapping the factories would have
missed two of the three.

Each step records the tool, its arguments, how long it took, and a bounded digest of the result —
never the whole payload, because a single `agent_kb_search` returns thousands of tokens and the
point is a readable trace.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MAX_ARG_CHARS = 400
MAX_RESULT_CHARS = 700


def _digest(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + f" …[+{len(text) - limit}]"


class ToolTracer:
    """Every tool call, in order, with arguments and a bounded result digest."""

    def __init__(self):
        self.steps: List[Dict[str, Any]] = []
        self._original = None

    def __enter__(self):
        from langchain_core.tools import BaseTool

        self._original = BaseTool.run
        tracer = self

        def traced(self, tool_input, *a, **kw):
            started = time.time()
            record: Dict[str, Any] = {
                "n": len(tracer.steps) + 1,
                "tool": getattr(self, "name", type(self).__name__),
                "args": _digest(tool_input, MAX_ARG_CHARS),
            }
            tracer.steps.append(record)
            try:
                result = tracer._original(self, tool_input, *a, **kw)
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"[:200]
                record["seconds"] = round(time.time() - started, 1)
                raise
            record["seconds"] = round(time.time() - started, 1)
            record["result_chars"] = len(result) if isinstance(result, str) else None
            record["result"] = _digest(result, MAX_RESULT_CHARS)
            return result

        BaseTool.run = traced
        return self

    def __exit__(self, *exc):
        from langchain_core.tools import BaseTool

        BaseTool.run = self._original
        return False


CASES = {
    "e2sfca": {
        "query": (
            "The I-GUIDE platform holds a publication describing how to measure spatial "
            "accessibility of COVID-19 healthcare resources in Illinois using the Enhanced "
            "Two-Step Floating Catchment Area method. Report the travel-time catchment bands "
            "in minutes, the distance-decay weight for each band, and the exact import line "
            "for the platform's callable E2SFCA implementation."),
        "why": ("The answer exists only as extracted cross-document structure: the paper's "
                "parameters in one element, the callable implementation in another."),
        "truth": {"bands_minutes": [10, 20, 30], "weights": [1.0, 0.68, 0.22],
                  "import_contains": "iguide_methods"},
    },
    "flood": {
        "query": (
            "Using the I-GUIDE platform's 'Probabilistic Flood Inundation Mapping using "
            "Physics-Aware Spatial AI' element (05269a1a) and its associated dataset "
            "(element f49f395e), report how many dates are listed in the test split file, "
            "how many Dynamic World raster files the dataset contains, and the coordinate "
            "reference system of those rasters."),
        "why": ("The most complete runnable bundle in the corpus: ten callable units against a "
                "staged 16 MB archive holding exactly the files they read, including trained "
                "checkpoints."),
        # Read directly from the archive before any run, so the trace can be judged rather than
        # admired: 322 entries, of which 314 are Dynamic World rasters, all EPSG:4326.
        "truth": {"test_dates": 24, "dw_rasters": 314, "crs": "EPSG:4326"},
    },
}


def run(case_id: str, arm: str) -> Dict[str, Any]:
    from scripts.ab_kb_problems import ARMS as _ARMS, configure

    # The skills arm gets the same tools as with_kb; only the skill registry differs.
    ARMS = {**_ARMS, "with_kb_skills": _ARMS["with_kb"]}

    configure()
    os.environ["AGENT_KB_DB"] = "1"
    os.environ["AGENT_ABLATE_KB"] = "1" if ARMS[arm]["ablate"] else "0"
    # A third arm: the knowledge base AND the generated SKILL.md for the element in question.
    # Both A/B harnesses set AGENT_SKILLS_ENABLED=0, so every measurement so far ran with this
    # path off — and the flood trace is the argument for turning it on. Twelve of its
    # thirty-three calls were spent reading the library's SOURCE and searching the filesystem
    # for a dataset nothing had staged, which is precisely what the skill's Run section says to
    # do instead.
    if arm == "with_kb_skills":
        os.environ["AGENT_SKILLS_ENABLED"] = "1"
        extra = os.environ.get("TRACE_SKILL_PATHS", "")
        if extra:
            os.environ["AGENT_SKILL_PATHS"] = extra
    else:
        os.environ["AGENT_SKILLS_ENABLED"] = "0"

    from agent_runtime.graph_runtime import run_agent_query

    case = CASES[case_id]
    tracer = ToolTracer()
    started = time.time()
    error = None
    res: Dict[str, Any] = {}
    try:
        with tracer:
            res = run_agent_query(
                case["query"], use_supervisor=True, code_exec=True, include_mcp_tools=False,
                enabled_search_methods=ARMS[arm]["tools"],
                thread_id=f"trace_{case_id}_{arm}_{int(started)}")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"[:300]

    return {
        "case": case_id, "arm": arm, "ablated": ARMS[arm]["ablate"],
        "skills": os.environ.get("AGENT_SKILLS_ENABLED") == "1",
        "query": case["query"], "why": case["why"], "truth": case.get("truth"),
        "elapsed_s": round(time.time() - started, 1), "error": error,
        "model": os.getenv("CLAUDE_CLI_MODEL", "?"),
        "steps": tracer.steps,
        "final_answer": str(res.get("final_answer") or ""),
        "peer_route": [str(c) for c in ((res.get("route_trace") or {}).get("called_tools") or [])],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", action="append", choices=sorted(CASES))
    ap.add_argument("--arm", action="append",
                    choices=["no_kb", "with_kb", "with_kb_skills"])
    ap.add_argument("--out", default=str(REPO / "outputs" / "agent_traces.json"))
    args = ap.parse_args()

    cases = args.case or sorted(CASES)
    arms = args.arm or ["no_kb", "with_kb"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traces = []
    if out_path.is_file():
        try:
            traces = json.loads(out_path.read_text())["traces"]
        except (OSError, ValueError, KeyError):
            traces = []

    for case_id in cases:
        for arm in arms:
            print(f"\n{'=' * 78}\n{case_id}  [{arm}]\n{'=' * 78}", flush=True)
            trace = run(case_id, arm)
            traces = [t for t in traces
                      if not (t["case"] == case_id and t["arm"] == arm)] + [trace]
            print(f"  {trace['elapsed_s']}s   {len(trace['steps'])} tool call(s)", flush=True)
            for step in trace["steps"]:
                print(f"   {step['n']:>2}. {step['tool']:<26}{step.get('seconds', 0):>7.1f}s  "
                      f"{step['args'][:78]}", flush=True)
            out_path.write_text(json.dumps({"traces": traces}, indent=2, default=str) + "\n")

    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
