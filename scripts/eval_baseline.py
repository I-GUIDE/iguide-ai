"""Retrieval-only ABLATION of GeoPathfinder over the ten-task benchmark.

Same index, prompts, expected sources, rubric, and backbone model as eval_record.py,
but DISABLES the action layer to isolate whether extraction, tools, skills, code
execution, and orchestration add value beyond retrieval:

  - Smart Search keyword / semantic / spatial / graph retrieval ........ ENABLED
  - answer synthesis .................................................... ENABLED
  - agent-KB extraction records (agent_kb_search / get_kb_block) ........ DISABLED
  - MCP tools .......................................................... DISABLED
  - skills ............................................................. DISABLED
  - generated-code execution + workflow execution ...................... DISABLED
  - artifact generation ................................................ (none possible)

Records to outputs/eval_records/baseline/<tid>/ (does NOT touch the GeoPathfinder
records). Score with the SAME eval_common.normalize rubric so both systems are
judged identically.

Usage:
  python scripts/eval_baseline.py --all
  python scripts/eval_baseline.py --tasks T2,T4
"""
from __future__ import annotations
import argparse, json, os, shutil, sys, time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
# override=False: an explicitly exported variable must WIN over the file. With
# override=True a stale .env value (e.g. a decommissioned FLASK_EMBEDDING_URL)
# silently replaced whatever the caller set, making a per-run override impossible
# and the reason invisible.
from dotenv import load_dotenv; load_dotenv(REPO / ".env")

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", default=""); ap.add_argument("--all", action="store_true")
ap.add_argument("--model", default="gpt-5.4-mini")
args, _ = ap.parse_known_args()

# Backbone-model parity with the GeoPathfinder benchmark.
os.environ["OPENAI_CHAT_MODEL"] = args.model
os.environ.pop("OPENAI_MODEL", None); os.environ.pop("VLLM_MODEL", None)
# Ablation: turn the entire action layer OFF.
os.environ["AGENT_CODE_EXEC"] = "0"
os.environ["AGENT_ALLOW_WORKFLOW_EXEC"] = "0"
os.environ.setdefault("AGENT_FILE_STORAGE_ROOT", str(REPO / "agent_chat_files"))
OUT = Path(os.environ["AGENT_FILE_STORAGE_ROOT"]) / "outputs"

import eval_common  # noqa: E402
eval_common.RECDIR = REPO / "outputs" / "eval_records" / "baseline"  # don't clobber GP records
from eval_common import (  # noqa: E402
    TASKS, TASK_META, evidence_list, peer_calls, tool_outputs, real_artifacts, write_record)

# Retrieval-only substrate: Smart Search methods, NO agent-KB.
# (Must use the exact granular-tool NAMES, which carry the "_search" suffix.)
RETRIEVAL_ONLY_METHODS = ["keyword_search", "semantic_search", "spatial_search", "neo4j_search"]
# A non-existent skills root => empty SkillRegistry => skills disabled.
NO_SKILLS_ROOT = [str(REPO / "_ablation_no_skills")]


def run_task(tid, prompt, exp_ids, geo, llm):
    from agent_runtime.graph_runtime import run_agent_query
    before = {p.name for p in OUT.glob("*")} if OUT.exists() else set()
    t0 = time.time()
    res = run_agent_query(
        prompt, llm=llm,
        include_mcp_tools=False,                         # no MCP tools
        enabled_search_methods=RETRIEVAL_ONLY_METHODS,   # no agent_kb_search / get_kb_block
        code_exec=False,                                 # no generated-code execution
        skill_roots=NO_SKILLS_ROOT,                      # no skills
        thread_id="base_" + tid,
    )
    dt = round(time.time() - t0)
    new = sorted(({p.name for p in OUT.glob("*")} if OUT.exists() else set()) - before)
    arts = [n for n in new if n.endswith((".png", ".csv", ".geojson", ".parquet", ".json", ".html", ".txt"))]
    orch = res.get("orchestration_result") or {}; rt = res.get("route_trace") or {}
    final = res.get("final_answer") or ""
    blob = json.dumps(res, default=str)
    rec = {
        "task_id": tid, "model": args.model, "elapsed_s": dt, "config": "retrieval_only",
        "prompt": prompt, "final_answer": final,
        "retrieved_evidence": evidence_list(orch),
        "execution_trace": {"supervisor_decisions": rt.get("supervisor_actions"),
                            "peer_tool_calls": peer_calls(orch)},
        "artifact_lineage": tool_outputs(orch),
        "output_artifacts": arts,
        "grounding_audit": res.get("grounding_audit"),
        "grounded_on_expected_source_elements": [e for e in exp_ids if e in blob],
    }
    d = eval_common.RECDIR / tid; d.mkdir(parents=True, exist_ok=True)
    for n in real_artifacts(arts):  # should be empty (execution disabled)
        try: shutil.copyfile(OUT / n, d / n)
        except Exception: pass
    rec = write_record(rec, tid)
    o = rec["outcome"]
    print(f"{tid}: {dt}s | status={o['task_status']} | retr={o['retrieval_success']} "
          f"exec={o['execution_success']} art_prod={o['artifact_produced']} "
          f"| grounded={rec['grounded_on_expected_source_elements']}")
    return rec


def main():
    ids = list(TASKS) if args.all else [t.strip().upper() for t in args.tasks.split(",") if t.strip()]
    if not ids: ids = list(TASKS)
    from agent_runtime.executor_factory import build_default_llm
    llm = build_default_llm()
    eval_common.RECDIR.mkdir(parents=True, exist_ok=True)
    print(f"retrieval-only ablation: model={args.model} index={os.environ.get('OPENSEARCH_INDEX')} "
          f"-> {eval_common.RECDIR}\n")
    tally = {}
    for tid in ids:
        if tid not in TASKS:
            print(f"unknown task {tid}"); continue
        prompt, exp, geo = TASKS[tid]
        try:
            rec = run_task(tid, prompt, exp, geo, llm)
            tally[tid] = rec["outcome"]["task_status"]
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"{tid}: ERROR {type(e).__name__}: {e}")
    print("\n=== retrieval-only tally ===", dict(Counter(tally.values())))
    print("per task:", tally)


if __name__ == "__main__":
    main()
