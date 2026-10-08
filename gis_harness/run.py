"""Run GIS tasks against a local agent server and score them.

    python -m gis_harness.run --start-server 5301 --model lumen:deepseek-v4-flash --tasks sample
    python -m gis_harness.run --base-url http://localhost:5301 --model openai:gpt-5.6-luna --tasks all
    python -m gis_harness.run --summarise gis_harness/runs/<label>

Each run writes gis_harness/runs/<label>/<model>/<task>.json (the score, the answer, the tool
calls, the token usage) and <task>.events.jsonl (every SSE event), then summary.json and a
printed table with one column block per model. Scores for different models are never pooled.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import datasets
from .client import run_turn, upload
from .score import score
from .tasks import BY_ID, SAMPLE, TASKS, live_osm_schools

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PRICING = json.loads((HERE / "pricing.json").read_text())["models"]


def start_server(port: int, log_path: Path) -> subprocess.Popen:
    """A local agent server that cannot write to shared infrastructure, or pay for geocoding.

    AGENT_MODE=local closes the conversation store and keeps KB search read-only (AGENTS.md,
    "Run it"). GOOGLE_MAPS_API_KEY is blanked because KB spatial search geocodes through
    Google, a metered call no task needs. The trace limits are raised so duplicate-call
    detection compares whole arguments, not their first 3,000 characters.
    """
    env = {**os.environ, "PYTHONPATH": str(REPO), "PORT": str(port), "AGENT_MODE": "local",
           "AGENT_CHAT_API_KEY": "", "GOOGLE_MAPS_API_KEY": "",
           "AGENT_TRACE_JSON_LIMIT": "20000", "AGENT_TRACE_TEXT_LIMIT": "20000",
           "AGENT_TRACE_SSE_TEXT_LIMIT": "20000"}
    # Appended, not truncated: a --resume run starts a new server on the same label, and a
    # truncated log lost the main run's evidence (2026-10-08, phase 2's after-run).
    log = open(log_path, "a")
    proc = subprocess.Popen([sys.executable, str(REPO / "api" / "server.py")], env=env,
                            stdout=log, stderr=subprocess.STDOUT, cwd=str(REPO))
    import requests
    for _ in range(120):
        try:
            if requests.get(f"http://localhost:{port}/agent/models", timeout=5).ok:
                return proc
        except Exception:  # noqa: BLE001
            pass
        if proc.poll() is not None:
            raise RuntimeError(f"server exited; see {log_path}")
        time.sleep(1)
    proc.terminate()
    raise RuntimeError(f"server did not answer on :{port}; see {log_path}")


def cost_usd(usage: List[Dict[str, Any]]) -> Dict[str, Any]:
    tin = sum(u.get("input_tokens") or 0 for u in usage)
    tout = sum(u.get("output_tokens") or 0 for u in usage)
    tcached = sum(u.get("cached_input_tokens") or 0 for u in usage)
    cost, unpriced = 0.0, set()
    for u in usage:
        rate = PRICING.get(u.get("model") or "")
        if not rate or rate.get("input") is None:
            unpriced.add(u.get("model"))
            continue
        cached = u.get("cached_input_tokens") or 0
        fresh = (u.get("input_tokens") or 0) - cached
        cost += (fresh * rate["input"] + cached * rate["cached_input"]
                 + (u.get("output_tokens") or 0) * rate["output"]) / 1e6
    return {"llm_calls": len(usage), "input_tokens": tin, "cached_input_tokens": tcached,
            "output_tokens": tout, "usage_absent": sum(1 for u in usage if u.get("usage")),
            "cost_usd": None if unpriced and not cost else round(cost, 4),
            "unpriced_models": sorted(m for m in unpriced if m)}


def expected_for(task, data_dir: Path, cache: Path):
    if task.dataset == "live_osm_schools":
        return {}, live_osm_schools()
    return datasets.build(task.dataset, data_dir, cache)


def run_one(task_id: str, provider: str, model: str, base_url: str, out: Path,
            data_dir: Path, cache: Path, trial: int = 0, resume: bool = False) -> Optional[Dict[str, Any]]:
    """One task, one trial. A harness failure (a live reference that cannot be fetched, an
    upload error) is recorded for that task and does not end the run: on 2026-10-08 every
    Overpass mirror refused the live reference for one trial, and the exception took the whole
    102-task run down at task 100, with no summary."""
    stem = f"{task_id}" + (f".t{trial}" if trial else "")
    if resume and (out / f"{stem}.json").exists():
        return json.loads((out / f"{stem}.json").read_text())
    try:
        return _run_one(task_id, provider, model, base_url, out, cache, trial)
    except Exception as exc:  # noqa: BLE001
        record = {"task": task_id, "trial": trial, "provider": provider, "model": model,
                  "harness_error": f"{type(exc).__name__}: {exc}"[:500]}
        (out / f"{stem}.harness-error.txt").write_text(record["harness_error"])
        print(f"{provider}:{model} {task_id} HARNESS ERROR {record['harness_error'][:160]}",
              flush=True)
        return None


def _run_one(task_id: str, provider: str, model: str, base_url: str, out: Path,
             cache: Path, trial: int) -> Dict[str, Any]:
    task = BY_ID[task_id]
    # Per task and trial: T02 and U01 share a dataset, and runs in flight must not share files.
    files, expected = expected_for(task, out / "_data" / f"{task_id}.t{trial}", cache)
    thread = f"harness-{task_id}-{model.replace(':', '_')}-{int(time.time())}-{trial}"
    file_ids = upload(base_url, thread, files) if files else []
    query = task.prompt(expected)
    stem = f"{task_id}" + (f".t{trial}" if trial else "")
    turn = run_turn(base_url, query, provider=provider, model=model, file_ids=file_ids,
                    thread_id=thread, raw_log=out / f"{stem}.events.jsonl")
    s = score(task, expected, turn)
    record = {"task": task_id, "title": task.title, "trial": trial, "provider": provider,
              "model": model, "query": query, "files": sorted(files),
              "expected": {k: v for k, v in expected.items() if k != "_why"},
              "expected_why": expected.get("_why"), "score": s.to_dict(),
              "answer": turn["answer"], "seconds": turn["seconds"], "route": turn["route"],
              "audit_severity": turn["audit_severity"], "map_layers": turn["map_layers"],
              "tool_calls": [{"name": c["name"], "agent": c.get("agent")} for c in turn["tool_calls"]],
              "usage": cost_usd(turn["usage"]), "error": turn["error"], "thread_id": thread}
    (out / f"{stem}.json").write_text(_scrub(json.dumps(record, indent=2, default=str)))
    return record


def _scrub(text: str) -> str:
    """Records are committed to a public repository as baselines: no local paths in them."""
    for path, mark in ((str(REPO), "<repo>"), (os.path.expanduser("~"), "<home>")):
        text = text.replace(path, mark)
    return text


def rescore(path: Path) -> Dict[str, Any]:
    """Re-score a stored task from its own event stream, with no model call.

    The record keeps the expected values it was scored against, and `<task>.events.jsonl`
    every SSE event, so a fix to score.py applies to past runs exactly.
    """
    from .client import _reduce

    record = json.loads(path.read_text())
    events = path.with_name(path.name[: -len(".json")] + ".events.jsonl")
    if not events.exists():
        return record
    turn = {"answer": None, "tool_calls": [], "tool_results": [], "tool_errors": [],
            "usage": [], "map_layers": 0, "events": 0, "route": None, "error": None,
            "audit_severity": None}
    for line in events.read_text().splitlines():
        ev = json.loads(line)
        _reduce(turn, ev["event"], ev["data"])
    turn["error"] = record.get("error") or turn["error"]
    record["score"] = score(BY_ID[record["task"]], record["expected"], turn).to_dict()
    record["usage"] = cost_usd(turn["usage"])
    path.write_text(_scrub(json.dumps(record, indent=2, default=str)))
    return record


def compare(base_dir: Path, new_dir: Path, trials: Optional[int] = None) -> Dict[str, Any]:
    """Per-model metrics for two runs, over the task trials both have. A baseline with three
    trials compared against an after-run with two compares only trials 0 and 1 of each."""
    def load(d: Path) -> List[Dict[str, Any]]:
        out = []
        for p in sorted(d.rglob("*.json")):
            if p.name == "summary.json" or "_data" in p.parts:
                continue
            r = rescore(p) if p.with_name(p.name[:-5] + ".events.jsonl").exists() else \
                json.loads(p.read_text())
            out.append(r)
        return out

    base, new = load(base_dir), load(new_dir)
    key = lambda r: (r["provider"], r["model"], r["task"], r["trial"])  # noqa: E731
    both = {key(r) for r in base} & {key(r) for r in new}
    if trials is not None:
        both = {k for k in both if k[3] < trials}
    sb = summarise([r for r in base if key(r) in both])
    sn = summarise([r for r in new if key(r) in both])
    rows = {}
    for model in sorted(set(sb) | set(sn)):
        a, b = sb.get(model, {}), sn.get(model, {})
        rows[model] = {k: (a.get(k), b.get(k)) for k in (
            "tasks", "correct", "refused_gracefully", "strict", "zero_unproductive",
            "unproductive_steps", "duplicate_calls", "failed_calls", "banners_on_correct",
            "banners_total", "source_named", "seconds_total", "llm_calls", "input_tokens",
            "output_tokens", "cost_usd")}
        changed = {}
        for t in sorted(set(a.get("per_task", {})) | set(b.get("per_task", {}))):
            ra, rb = a.get("per_task", {}).get(t), b.get("per_task", {}).get(t)
            if (ra or "").split(" [")[0] != (rb or "").split(" [")[0] or \
                    ("unprod" in (ra or "")) != ("unprod" in (rb or "")):
                changed[t] = (ra, rb)
        rows[model]["changed_tasks"] = changed
    return rows


def summarise(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        by_model.setdefault(f"{r['provider']}:{r['model']}", []).append(r)
    out = {}
    for m, rs in by_model.items():
        solv = [r for r in rs if r["score"]["solvable"]]
        uns = [r for r in rs if not r["score"]["solvable"]]
        correct = [r for r in solv if r["score"]["correct"]]
        costs = [r["usage"]["cost_usd"] for r in rs if r["usage"]["cost_usd"] is not None]
        out[m] = {
            "tasks": len(rs),
            "correct": f"{len(correct)}/{len(solv)}",
            "refused_gracefully": f"{sum(1 for r in uns if r['score']['refusal']['refused'])}/{len(uns)}",
            "of_which_substituted": sum(1 for r in uns
                                        if r["score"]["refusal"].get("outcome") == "substituted"),
            "strict": f"{sum(1 for r in rs if r['score']['strict'])}/{len(rs)}",
            "zero_unproductive": f"{sum(1 for r in rs if r['score']['productive'])}/{len(rs)}",
            "unproductive_steps": sum(r["score"]["steps"]["unproductive"] for r in rs),
            "duplicate_calls": sum(len(r["score"]["steps"]["duplicates"]) for r in rs),
            "failed_calls": sum(len(r["score"]["steps"]["failed"]) for r in rs),
            "banners_on_correct": sum(1 for r in correct if r["score"]["banners"]),
            "banners_total": sum(len(r["score"]["banners"]) for r in rs),
            "source_named": f"{sum(1 for r in rs if r['score']['source'])}/{len(rs)}",
            "errors": sum(1 for r in rs if r["error"]),
            "seconds_total": round(sum(r["seconds"] or 0 for r in rs), 1),
            "llm_calls": sum(r["usage"]["llm_calls"] for r in rs),
            "input_tokens": sum(r["usage"]["input_tokens"] for r in rs),
            "output_tokens": sum(r["usage"]["output_tokens"] for r in rs),
            "cost_usd": round(sum(costs), 4) if costs else None,
            "per_task": {r["task"] + (f".t{r['trial']}" if r["trial"] else ""): _row(r) for r in rs},
        }
    return out


def _row(r: Dict[str, Any]) -> str:
    s = r["score"]
    if s["solvable"]:
        res = "OK " if s["correct"] else "BAD"
        miss = [c["name"] for c in s["checks"] if not c["matched"]]
        res += f" miss={','.join(miss)}" if miss else ""
    else:
        ref = s["refusal"]
        outcome = ref.get("outcome") or ("refused" if ref["refused"] else "fabricated")
        res = {"refused": "REFUSED", "substituted": f"SUBSTITUTED({ref.get('value_given')})",
               "fabricated": f"FABRICATED({ref.get('fabricated')})"}.get(outcome, "UNCLEAR")
    flags = []
    if s["steps"]["unproductive"]:
        flags.append(f"unprod={s['steps']['unproductive']}")
    if s["banners"]:
        flags.append(f"banners={len(s['banners'])}")
    if not s["source"]:
        flags.append("no-source")
    if r["error"]:
        flags.append("ERROR")
    return f"{res} [{' '.join(flags)}] {r['seconds']}s"


def print_summary(summary: Dict[str, Any]) -> None:
    for m, s in summary.items():
        print(f"\n== {m}")
        for k, v in s.items():
            if k != "per_task":
                print(f"  {k:22s} {v}")
        for t, row in s["per_task"].items():
            print(f"    {t:6s} {row}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--start-server", type=int, metavar="PORT")
    ap.add_argument("--model", action="append", default=[], help="provider:model, repeatable")
    ap.add_argument("--tasks", default="sample", help="sample | all | comma-separated ids")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--label", default=None)
    ap.add_argument("--parallel", type=int, default=1, help="tasks in flight at once, per run")
    ap.add_argument("--summarise", type=Path, default=None, help="re-summarise a finished run dir")
    ap.add_argument("--resume", action="store_true", help="skip tasks this label already has")
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("BASE", "NEW"),
                    help="per-model metrics of two finished runs over the trials both have")
    ap.add_argument("--compare-trials", type=int, default=None)
    a = ap.parse_args(argv)

    if a.compare:
        rows = compare(a.compare[0], a.compare[1], a.compare_trials)
        for model, r in rows.items():
            print(f"\n== {model}   (base -> new)")
            for k, v in r.items():
                if k != "changed_tasks":
                    print(f"  {k:22s} {v[0]!s:>12} -> {v[1]!s}")
            for t, (x, y) in r["changed_tasks"].items():
                print(f"    {t:8s} {x}  ->  {y}")
        return 0

    if a.summarise:
        recs = [rescore(p) for p in sorted(a.summarise.rglob("*.json"))
                if p.name != "summary.json" and "_data" not in p.parts]
        summary = summarise(recs)
        (a.summarise / "summary.json").write_text(json.dumps(summary, indent=2))
        print_summary(summary)
        return 0

    ids = (SAMPLE if a.tasks == "sample" else [t.id for t in TASKS] if a.tasks == "all"
           else [x.strip() for x in a.tasks.split(",") if x.strip()])
    unknown = [i for i in ids if i not in BY_ID]
    if unknown:
        ap.error(f"unknown task ids: {unknown}")
    if not a.model:
        ap.error("--model provider:model is required")
    label = a.label or time.strftime("%Y%m%d-%H%M%S")
    root = HERE / "runs" / label
    root.mkdir(parents=True, exist_ok=True)
    data_dir, cache = root, HERE / ".cache"

    proc = None
    base = a.base_url
    if a.start_server:
        proc = start_server(a.start_server, root / "server.log")
        base = f"http://localhost:{a.start_server}"
    if not base:
        ap.error("--base-url or --start-server is required")
    try:
        jobs = []
        with cf.ThreadPoolExecutor(max_workers=max(1, a.parallel) * len(a.model)) as ex:
            for spec in a.model:
                provider, _, model = spec.partition(":")
                out = root / f"{provider}_{model.replace(':', '_')}"
                out.mkdir(parents=True, exist_ok=True)
                for trial in range(a.trials):
                    for tid in ids:
                        jobs.append(ex.submit(run_one, tid, provider, model, base, out,
                                              data_dir, cache, trial, a.resume))
            records = []
            for j in cf.as_completed(jobs):
                r = j.result()
                if r is None:
                    continue
                print(f"{r['provider']}:{r['model']} {r['task']} {_row(r)}", flush=True)
                records.append(r)
    finally:
        if proc:
            proc.terminate()
    summary = summarise(records)
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
