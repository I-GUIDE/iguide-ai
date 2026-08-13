"""Full-trace recorder for the GeoPathfinder benchmark.

Runs a case through the REAL agent via the STREAMING entry point with detail-tier events on
(agent_dev=True), buffering the complete event log: every supervisor decision WITH its stated
reason, each peer's LLM messages, and full tool call args + (large-capped) tool outputs. Writes,
per task, to outputs/eval_records/<task_id>/:
  - trace.json : the raw ordered event list (verbatim).
  - trace.md   : a human-readable chronological trace (decision timeline + per-stage LLM/tool I/O).
  - record.json / record.md / artifacts : the same normalized record eval_record.py produces.

Requires (vs. eval_record.py): the supervisor `reason` is now emitted as a `supervisor_decision`
trace event (agent_runtime/supervisor_graph.py) and detail-text limits are raised via
AGENT_TRACE_TEXT_LIMIT / AGENT_TRACE_JSON_LIMIT (set below before importing the runtime).

Usage:
  python scripts/eval_trace.py --tasks T4
  python scripts/eval_trace.py --tasks CRIME_HEATMAP --model gpt-5.4-mini
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
ap.add_argument("--tasks", default="T4")
ap.add_argument("--model", default="gpt-5.4-mini")
ap.add_argument("--text-limit", default="20000", help="max chars per LLM message / tool result in the trace")
args, _ = ap.parse_known_args()

os.environ["OPENAI_CHAT_MODEL"] = args.model; os.environ.pop("OPENAI_MODEL", None); os.environ.pop("VLLM_MODEL", None)
os.environ["AGENT_CODE_EXEC"] = "1"; os.environ.setdefault("AGENT_CODE_EXEC_BACKEND", "local")
os.environ["AGENT_KB_BACKEND"] = "local"; os.environ.setdefault("AGENT_FILE_STORAGE_ROOT", str(REPO / "agent_chat_files"))
os.environ["AGENT_ALLOW_WORKFLOW_EXEC"] = "1"
# Capture detail-tier events + raise the per-event text caps for a faithful trace.
os.environ["AGENT_DEV"] = "1"
os.environ["AGENT_TRACE_TEXT_LIMIT"] = str(args.text_limit)
os.environ["AGENT_TRACE_JSON_LIMIT"] = str(max(int(args.text_limit), 20000))
OUT = Path(os.environ["AGENT_FILE_STORAGE_ROOT"]) / "outputs"

from eval_common import (  # noqa: E402
    TASKS, RECDIR, evidence_list, peer_calls, tool_outputs, real_artifacts, write_record)

# Events that carry no narrative value in the rendered trace.
_SKIP = {"status", "llm_start", "search_complete"}


def _fmt_args(name, a):
    """Render tool args; show execute_code's code in a fenced block."""
    if isinstance(a, dict):
        if name == "execute_code" and a.get("code"):
            rest = {k: v for k, v in a.items() if k != "code"}
            head = (f" _{json.dumps(rest, default=str)}_" if rest else "")
            return head + "\n\n```python\n" + str(a["code"]) + "\n```"
        return " " + json.dumps(a, ensure_ascii=False, default=str)
    return " " + str(a)


def render_trace_md(tid, prompt, model, dt, events, final_answer):
    decisions = [e for e in events if e.get("event") == "supervisor_decision"]
    md = [f"# {tid} — full agent trace (model={model}, {dt}s)", "",
          "## Prompt", prompt, "",
          "## Supervisor decision timeline"]
    if decisions:
        for i, e in enumerate(decisions, 1):
            d = e.get("data") or {}
            md.append(f"{i}. **{d.get('next')}** — {d.get('reason') or '(no reason given)'}")
    else:
        md.append("- (no LLM supervisor_decision events — heuristic routing or non-dev run)")
    md += ["", "## Final answer", (final_answer or "")[:4000], "",
           "## Chronological trace",
           "_Detail events are attributed to the most recent stage (node). Tool-call args are "
           "verbatim; LLM messages and tool results are capped at the configured text limit._", ""]

    stage = "supervisor"; stage_n = 0
    for e in events:
        ev, d = e.get("event"), (e.get("data") or {})
        if ev in _SKIP:
            continue
        if ev == "node_started":
            stage = d.get("stage") or stage; stage_n += 1
            md.append(f"\n### [{stage_n}] ▶ {stage}")
            continue
        if ev == "node_completed":
            md.append(f"_↳ {d.get('stage') or stage} complete_")
            continue
        if ev == "supervisor_decision":
            md.append(f"\n🧭 **supervisor → `{d.get('next')}`** — {d.get('reason') or '(no reason)'}")
            continue
        if ev == "decider_fallback":
            md.append("\n⚠️ supervisor used the heuristic fallback (LLM decider output unparseable)")
            continue
        if ev == "llm_interaction":
            txt = d.get("content") or d.get("message") or ""
            if txt.strip():
                md.append(f"💬 _[{stage}] LLM:_ {txt}")
            continue
        if ev == "tool_call":
            md.append(f"🔧 _[{stage}]_ `{d.get('name')}`{_fmt_args(d.get('name'), d.get('args'))}")
            continue
        if ev in ("tool_result", "tool_error"):
            label = "result" if ev == "tool_result" else "ERROR"
            md.append(f"↳ _{d.get('tool_name') or d.get('name') or 'tool'} {label}:_ {d.get('content') or d.get('message') or ''}")
            continue
        if ev == "llm_error":
            md.append(f"✗ LLM error: {d.get('message')}")
            continue
        # route_trace / decision / final_answer / completed handled elsewhere
    md += ["", "## Outcome", "- see [`record.md`](record.md) for the normalized outcome block + correctness checks."]
    return "\n".join(md)


def run_trace(tid, prompt, exp_ids):
    from agent_runtime.graph_runtime import stream_agent_query_events
    from agent_runtime.langchain_mcp_tools import mcp_tools_enabled
    before = {p.name for p in OUT.glob("*")} if OUT.exists() else set()
    events, completed, final_answer = [], None, ""
    t0 = time.time()
    for ev in stream_agent_query_events(
            prompt, agent_dev=True, include_mcp_tools=mcp_tools_enabled(), thread_id="trace_" + tid):
        if not isinstance(ev, dict):
            continue
        kind = ev.get("event")
        if kind == "completed":
            completed = ev.get("data") or {}
            continue
        if kind == "final_answer":
            final_answer = (ev.get("data") or {}).get("answer") or final_answer
        if kind == "error":
            events.append(ev); continue
        events.append(ev)
    dt = round(time.time() - t0)

    resp = completed or {}
    orch = resp.get("orchestration_result") or {}
    rt = resp.get("route_trace") or {}
    final = resp.get("final_answer") or final_answer or ""
    new = sorted(({p.name for p in OUT.glob("*")} if OUT.exists() else set()) - before)
    arts = [n for n in new if n.endswith((".png", ".csv", ".geojson", ".parquet", ".json", ".html", ".txt"))]
    blob = json.dumps(resp, default=str)

    rec = {
        "task_id": tid, "model": args.model, "elapsed_s": dt,
        "prompt": prompt, "final_answer": final,
        "retrieved_evidence": evidence_list(orch),
        "execution_trace": {"supervisor_decisions": rt.get("supervisor_actions"),
                            "peer_tool_calls": peer_calls(orch)},
        "artifact_lineage": tool_outputs(orch),
        "output_artifacts": arts,
        "grounding_audit": resp.get("grounding_audit"),
        "grounded_on_expected_source_elements": [e for e in exp_ids if e in blob],
    }
    d = RECDIR / tid; d.mkdir(parents=True, exist_ok=True)
    for n in real_artifacts(arts):
        try: shutil.copyfile(OUT / n, d / n)
        except Exception: pass
    # write the trace artifacts FIRST so write_record's record.md can link to them
    (d / "trace.json").write_text(json.dumps({"task_id": tid, "model": args.model, "elapsed_s": dt,
                                              "prompt": prompt, "events": events}, indent=2, default=str), encoding="utf-8")
    (d / "trace.md").write_text(render_trace_md(tid, prompt, args.model, dt, events, final), encoding="utf-8")
    rec = write_record(rec, tid)

    o = rec["outcome"]
    n_dec = sum(1 for e in events if e.get("event") == "supervisor_decision")
    n_tool = sum(1 for e in events if e.get("event") == "tool_call")
    print(f"{tid}: {dt}s | status={o['task_status']} | events={len(events)} (decisions={n_dec}, tool_calls={n_tool}) "
          f"| artifacts={[p['name'] for p in rec['artifact_paths']]}")
    return rec


def main():
    from agent_runtime.langchain_mcp_tools import mcp_tools_enabled  # warm import / surface errors early
    ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"trace recorder: model={args.model} text_limit={args.text_limit} mcp={mcp_tools_enabled()} -> {RECDIR}\n")
    for tid in ids:
        key = tid if tid in TASKS else tid.upper()
        if key not in TASKS:
            print(f"unknown task {tid}; known: {list(TASKS)}"); continue
        prompt, exp, _ = TASKS[key]
        try:
            run_trace(key, prompt, exp)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"{tid}: ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
