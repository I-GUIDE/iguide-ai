"""Per-run evaluation recorder + outcome normalizer for the GeoPathfinder benchmark.

Runs each task through the REAL agent (run_agent_query, default deployment config) and saves a
rich, normalized record to outputs/eval_records/<task_id>/ (record.json + record.md + artifacts),
plus a paper-ready outputs/eval_records/SUMMARY.md. Shared task/normalization logic lives in
scripts/eval_common.py; for the FULL streaming agent trace (verbatim decisions/prompts/tool I/O)
use scripts/eval_trace.py.

Usage:
  python scripts/eval_record.py --tasks T3,T4              # (re)run a subset through the agent
  python scripts/eval_record.py --all [--model gpt-5.4-mini]
  python scripts/eval_record.py --reprocess               # recompute outcome for ALL existing records (offline)
  python scripts/eval_record.py --reprocess --tasks T1,T7
  python scripts/eval_record.py --summary-only            # just (re)write SUMMARY.md
"""
from __future__ import annotations
import argparse, json, os, shutil, sys, time
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
ap.add_argument("--reprocess", action="store_true", help="recompute outcome from existing record.json (no agent run)")
ap.add_argument("--summary-only", action="store_true", help="only (re)write SUMMARY.md")
args, _ = ap.parse_known_args()

_LIVE = not (args.reprocess or args.summary_only)
if _LIVE:
    os.environ["OPENAI_CHAT_MODEL"] = args.model; os.environ.pop("OPENAI_MODEL", None); os.environ.pop("VLLM_MODEL", None)
    os.environ["AGENT_CODE_EXEC"] = "1"; os.environ.setdefault("AGENT_CODE_EXEC_BACKEND", "local")
    os.environ["AGENT_KB_BACKEND"] = "local"; os.environ.setdefault("AGENT_FILE_STORAGE_ROOT", str(REPO / "agent_chat_files"))
    os.environ["AGENT_ALLOW_WORKFLOW_EXEC"] = "1"
OUT = Path(os.environ.get("AGENT_FILE_STORAGE_ROOT", str(REPO / "agent_chat_files"))) / "outputs"

from eval_common import (  # noqa: E402
    TASKS, TASK_META, RECDIR, evidence_list, peer_calls, tool_outputs,
    real_artifacts, write_record, write_summary, normalize)


def run_task(tid, prompt, exp_ids, geo, llm):
    from agent_runtime.graph_runtime import run_agent_query
    from agent_runtime.langchain_mcp_tools import mcp_tools_enabled
    before = {p.name for p in OUT.glob("*")} if OUT.exists() else set()
    t0 = time.time()
    res = run_agent_query(prompt, llm=llm, include_mcp_tools=mcp_tools_enabled(), thread_id="rec_" + tid)
    dt = round(time.time() - t0)
    new = sorted(({p.name for p in OUT.glob("*")} if OUT.exists() else set()) - before)
    arts = [n for n in new if n.endswith((".png", ".csv", ".geojson", ".parquet", ".json", ".html", ".txt"))]
    orch = res.get("orchestration_result") or {}; rt = res.get("route_trace") or {}
    final = res.get("final_answer") or ""
    blob = json.dumps(res, default=str)
    rec = {
        "task_id": tid, "model": args.model, "elapsed_s": dt,
        "prompt": prompt, "final_answer": final,
        "retrieved_evidence": evidence_list(orch),
        "execution_trace": {"supervisor_decisions": rt.get("supervisor_actions"),
                            "peer_tool_calls": peer_calls(orch)},
        "artifact_lineage": tool_outputs(orch),
        "output_artifacts": arts,
        "grounding_audit": res.get("grounding_audit"),
        "grounded_on_expected_source_elements": [e for e in exp_ids if e in blob],
    }
    d = RECDIR / tid; d.mkdir(parents=True, exist_ok=True)
    for n in real_artifacts(arts):  # copy only real deliverables; cache recorded by name only
        try: shutil.copyfile(OUT / n, d / n)
        except Exception: pass
    rec = write_record(rec, tid)
    o = rec["outcome"]
    print(f"{tid}: {dt}s | status={o['task_status']} | retr={o['retrieval_success']} exec={o['execution_success']} "
          f"| artifacts={[p['name'] for p in rec['artifact_paths']]} | grounded={rec['grounded_on_expected_source_elements']}")
    return rec


def reprocess(tid):
    p = RECDIR / tid / "record.json"
    if not p.exists():
        print(f"{tid}: no existing record.json — skip (run live first)"); return
    rec = json.loads(p.read_text(encoding="utf-8"))
    if tid in TASKS:
        rec.setdefault("prompt", TASKS[tid][0])
        rec["grounded_on_expected_source_elements"] = rec.get("grounded_on_expected_source_elements") or []
    rec = write_record(rec, tid)
    o = rec["outcome"]
    print(f"{tid}: reprocessed | status={o['task_status']} | retr={o['retrieval_success']} exec={o['execution_success']} "
          f"| artifacts={[x['name'] for x in rec['artifact_paths']]}")
    return rec


def main():
    if args.summary_only:
        write_summary(args.model); return

    ids = list(TASKS) if args.all else [t.strip().upper() for t in args.tasks.split(",") if t.strip()]
    if not ids:
        ids = list(TASK_META)

    if args.reprocess:
        print(f"reprocess (offline) -> {RECDIR}\n")
        for tid in ids:
            if tid not in TASK_META:
                print(f"unknown task {tid}"); continue
            reprocess(tid)
        write_summary(args.model); return

    from agent_runtime.executor_factory import build_default_llm
    llm = build_default_llm()
    print(f"recorder: model={args.model} backend={os.environ['AGENT_CODE_EXEC_BACKEND']} -> {RECDIR}\n")
    for tid in ids:
        if tid not in TASKS:
            print(f"unknown task {tid}; known: {list(TASKS)}"); continue
        prompt, exp, geo = TASKS[tid]
        try:
            run_task(tid, prompt, exp, geo, llm)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"{tid}: ERROR {type(e).__name__}: {e}")
    write_summary(args.model)


if __name__ == "__main__":
    main()
