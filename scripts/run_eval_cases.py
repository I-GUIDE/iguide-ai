"""Run GeoPathfinder eval cases end-to-end through the agent runtime and record outcomes.

Local dev settings: agent-KB = local (the eval corpus built by build_eval_kb.py),
search restricted to the agent-KB, code execution = local backend. Captures the
supervisor route, produced artifacts, citations, and a reuse-vs-generation signal.

Usage:
    python scripts/run_eval_cases.py --case 1b-crime
    python scripts/run_eval_cases.py --all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
# override=False: an explicitly exported variable must WIN over the file. With
# override=True a stale .env value (e.g. a decommissioned FLASK_EMBEDDING_URL)
# silently replaced whatever the caller set, making a per-run override impossible
# and the reason invisible.
from dotenv import load_dotenv
load_dotenv(REPO / ".env")  # OPENAI_* for the LLM

# Dev execution flags (set before importing the runtime).
os.environ["AGENT_KB_BACKEND"] = "local"
os.environ.setdefault("AGENT_FILE_STORAGE_ROOT", str(REPO / "agent_chat_files"))
os.environ["AGENT_CODE_EXEC"] = "1"
os.environ.setdefault("AGENT_CODE_EXEC_BACKEND", os.getenv("AGENT_CODE_EXEC_BACKEND", "local"))
os.environ["AGENT_ALLOW_WORKFLOW_EXEC"] = "1"
# The reuse cases must reuse EXTRACTED KB blocks (get_kb_block), not the hand-authored
# skills/chicago-crime-analysis/SKILL.md (which imports a non-existent module). Skills
# stay off by default here; case 5 (publication-guided) re-enables with the extracted skill.
os.environ.setdefault("AGENT_SKILLS_ENABLED", "0")

# Executable case studies anchor on the crime notebook (cca9b545), which is genuinely
# self-contained (Socrata API + raw-GitHub community areas). The heat cluster stays in the
# KB for grounding/precision (build_eval_kb.py) but reads non-fetchable local repo files,
# so it is not used for execution. Coverage gradient: exact-reuse -> parametric ->
# compositional -> negative control.
CASES = {
    "1-exact-choropleth": ("Map the number of crimes per Chicago community area as a choropleth.", ["cca9b545"]),
    "2-parametric-theft": ("Map the number of THEFT crimes per Chicago community area as a choropleth.", ["cca9b545"]),
    "3-compositional-heatmap": ("Show me a heat map of violent crime cases in Chicago.", ["cca9b545"]),
    # New-domain generalization (distinct data-acquisition patterns, not crime):
    "W-weather": ("Using the Open-Meteo weather workflow, fetch recent daily temperatures and plot the time series.", ["afbee4bd"]),
    "G-gwr": ("Run the geographically weighted regression demo on its example dataset and map a result.", ["d8926bb3"]),
    "O-osm": ("Look up a place and build its street network with OSMnx, then summarize it.", ["8a77279b"]),
    "7-negative-control": ("Forecast next month's Chicago crime counts using an LSTM deep-learning model.", []),
}


def _outputs_dir() -> Path:
    return Path(os.environ["AGENT_FILE_STORAGE_ROOT"]) / "outputs"


def _snapshot(d: Path):
    return {p.name: p.stat().st_mtime for p in d.glob("*")} if d.exists() else {}


def run_one(cid: str, query: str, expected: list) -> dict:
    from agent_runtime.graph_runtime import run_agent_query

    before = _snapshot(_outputs_dir())
    t0 = time.time()
    res = run_agent_query(
        query,
        use_supervisor=True,
        code_exec=True,
        include_mcp_tools=False,
        enabled_search_methods=["agent_kb_search"],
        thread_id=f"eval_{cid}",
    )
    dt = time.time() - t0
    after = _snapshot(_outputs_dir())
    new_files = sorted(n for n in after if n not in before or after[n] != before.get(n))

    orch = res.get("orchestration_result") or {}
    trace = res.get("route_trace") or {}
    blob = json.dumps(res, default=str)
    summary = {
        "case": cid,
        "query": query,
        "elapsed_s": round(dt, 1),
        "final_answer": (res.get("final_answer") or "")[:1200],
        "grounding_audit": res.get("grounding_audit"),
        "new_artifacts": new_files,
        "cited_expected": [e for e in expected if e[:8] in blob],
        "supervisor_actions": (trace.get("supervisor_actions") if isinstance(trace, dict) else None),
        "called_tools": (trace.get("called_tools") if isinstance(trace, dict) else None),
        "code_result": (json.dumps(orch.get("code_result"), default=str)[:2500] if isinstance(orch, dict) else None),
        "analysis_results": (json.dumps(orch.get("analysis_results"), default=str)[:1200] if isinstance(orch, dict) else None),
    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all:
        cids = list(CASES)
    elif args.case:
        cids = [c.strip() for c in args.case.split(",") if c.strip()]
    else:
        cids = ["3-compositional-heatmap"]

    print(f"flags: CODE_EXEC={os.environ['AGENT_CODE_EXEC']} "
          f"BACKEND={os.environ['AGENT_CODE_EXEC_BACKEND']} "
          f"WORKFLOW_EXEC={os.environ['AGENT_ALLOW_WORKFLOW_EXEC']} KB=local\n")
    results = []
    for cid in cids:
        if cid not in CASES:
            print(f"unknown case {cid}; known: {list(CASES)}"); continue
        q, expected = CASES[cid]
        print("=" * 72)
        print(f"CASE {cid}: {q}")
        print("=" * 72)
        try:
            s = run_one(cid, q, expected)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            s = {"case": cid, "error": f"{type(exc).__name__}: {exc}"}
        results.append(s)
        print(json.dumps(s, indent=2, default=str)[:3000])

    out = REPO / "outputs" / "eval_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
