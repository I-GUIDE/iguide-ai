"""Does asking the user help? Three clarification policies, a simulated user, one task set.

    python -m gis_harness.clarify --tasks-file TASKS.json --label clarify-1 \\
        --policies never,assume,ask --model lumen:deepseek-v4-flash [--port 5340]

Each policy gets its own local server (AGENT_CLARIFY_POLICY, agent_runtime/clarify_policy.py),
and every task is run as a CONVERSATION: the question; then, while the agent's reply asks the
user something, a simulated user answers from what that task's user knows (`user_knows`), at
most `--max-followups` times. The simulated user is the same model as the agent and is told to
reveal only what is asked, and to say "no preference, use your judgment" for anything it does
not know, which is how a needless question shows up.

What is measured per conversation: whether the final answer is right (strictly: its headline
value; leniently: anywhere in it, and for grid cells a finer cell whose parent is the gold
one), whether the agent asked, whether the task needed a question (`ambiguous`), turns, time.

Task file (kept outside the repo when it holds third-party benchmark items):
[{"id", "ambiguous": bool, "user_knows": str,
  "catalogue": "T02"  -> a gis_harness.tasks task, scored by score.score; OR
  "query": str, "gold": {...}, "check": "h3" | "s2" | "h3s2" | "geohash" | "distance_km"
                                       | "direction"}]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import client, datasets
from .run import REPO, start_server

SIM_USER_PROMPT = (
    "You are role-playing the person who asked a GIS assistant a question.\n"
    "Your question was:\n{query}\n\n"
    "What you know (reveal only what the assistant asks for, in your own words):\n{knows}\n\n"
    "The assistant just replied:\n---\n{reply}\n---\n\n"
    "If the reply ASKS you something or asks you to choose BEFORE giving an answer, write the "
    "user's reply in one or two short sentences, using only what you know. If it asks for "
    "something you do not know, say you have no preference and it should use its best "
    "judgment. If the reply gives an answer (even one that states assumptions, offers "
    "alternatives, or refuses because something is missing), you are done.\n"
    "Output JSON only: {{\"done\": true}} or {{\"done\": false, \"reply\": \"...\"}}"
)


# --------------------------------------------------------------------------- scoring

def _h3_res(h: int) -> int:
    return (h >> 52) & 0xF


def _h3_parent(h: int, res: int) -> int:
    h = (h & ~(0xF << 52)) | (res << 52)
    for r in range(res + 1, 16):
        h |= 7 << ((15 - r) * 3)
    return h


def _s2_level(i: int) -> int:
    lsb = (i & -i).bit_length() - 1
    return 30 - lsb // 2


def _s2_parent(i: int, level: int) -> int:
    lsb = 1 << (2 * (30 - level))
    return (i & -lsb) | lsb


def _h3_cells(text: str) -> List[int]:
    out = []
    for m in re.findall(r"\b8[0-9a-f]{14}\b", text):
        out.append(int(m, 16))
    for m in re.findall(r"\b\d{17,19}\b", text):
        v = int(m)
        if (v >> 59) & 0xF == 1:            # H3 cell mode bit
            out.append(v)
    return out


def _s2_cells(text: str) -> List[int]:
    out = []
    for m in re.findall(r"\b\d{15,20}\b", text):
        v = int(m)
        if v < 2 ** 64 and (v >> 59) & 0xF != 1:
            out.append(v)
    for m in re.findall(r"\b(?:0x)?([0-9a-f]{1,16})\b", text):
        if len(m) >= 3 and re.search(r"[a-f]", m) and not m.startswith("8"):
            out.append(int(m.ljust(16, "0"), 16))    # an S2 token is the id without trailing 0s
    return out


def _grid_match(cells: List[int], gold: int, parent, level_of) -> Tuple[bool, bool]:
    """(the gold cell itself is stated FIRST, some stated cell lies inside the gold one)."""
    strict = bool(cells) and cells[0] == gold
    lvl = level_of(gold)
    lenient = any(c == gold or (level_of(c) >= lvl and parent(c, lvl) == gold) for c in cells)
    return strict, lenient


_KM = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(?:km|kilomet)", re.I)
_DIRS = ["northeast", "northwest", "southeast", "southwest", "north", "south", "east", "west"]


def _km_tolerance(gold: float) -> float:
    floor = 0.5 if gold < 50 else 5.0 if gold < 500 else 10.0   # MultiGlobeQA Table 12
    return max(floor, 0.05 * gold)


def score_answer(task: Dict[str, Any], answer: str) -> Dict[str, Any]:
    text = (answer or "").replace(" ", " ")
    low = text.lower()
    gold, check = task.get("gold") or {}, task.get("check")
    if check == "h3":
        s, l = _grid_match(_h3_cells(low), int(gold["h3"]), _h3_parent, _h3_res)
    elif check == "s2":
        s, l = _grid_match(_s2_cells(low), int(gold["s2"]), _s2_parent, _s2_level)
    elif check == "h3s2":
        hs, hl = _grid_match(_h3_cells(low), int(gold["h3"]), _h3_parent, _h3_res)
        ss, sl = _grid_match(_s2_cells(low), int(gold["s2"]), _s2_parent, _s2_level)
        s, l = hs and ss, hl and sl
    elif check == "geohash":
        found = re.findall(r"\b[0-9bcdefghjkmnpqrstuvwxyz]{6}\b", low)
        s, l = bool(found) and found[0] == gold["geohash"], gold["geohash"] in found
    elif check == "distance_km":
        vals = [float(v.replace(",", "")) for v in _KM.findall(text)]
        tol = _km_tolerance(float(gold["km"]))
        ok = [abs(v - float(gold["km"])) <= tol for v in vals]
        s, l = bool(ok) and ok[0], any(ok)
    elif check == "direction":
        flat = re.sub(r"\b(north|south)[- ](east|west)\b", r"\1\2", low)
        hits = re.findall(r"\b(" + "|".join(_DIRS) + r")\b", flat)
        want = gold["direction"].lower().replace("-", "").replace(" ", "")
        s, l = bool(hits) and hits[0] == want, want in hits
    else:
        raise ValueError(f"unknown check {check!r}")
    return {"correct": bool(s), "lenient": bool(l)}


# --------------------------------------------------------------------------- conversation

def _sim_user(llm: Any, query: str, knows: str, reply: str) -> Dict[str, Any]:
    from agent_runtime.evidence_quality import _extract_json_object

    raw = llm.invoke(SIM_USER_PROMPT.format(query=query, knows=knows or "(nothing more)",
                                            reply=(reply or "")[:4000]))
    text = getattr(raw, "content", raw)
    data = _extract_json_object(str(text)) or {}
    if data.get("done") is False and str(data.get("reply") or "").strip():
        return {"done": False, "reply": str(data["reply"]).strip()}
    return {"done": True}


def _gate_fired(events_path: Path) -> Optional[bool]:
    """The gate's own decision, from its node event (None: no gate ran this turn)."""
    try:
        for line in events_path.read_text().splitlines():
            ev = json.loads(line)
            data = ev.get("data") or {}
            if ev.get("event") == "node" and data.get("stage") == "clarify":
                detail = data.get("detail") or {}
                return bool((detail.get("detail") or detail).get("ask"))
    except Exception:  # noqa: BLE001
        return None
    return None


def converse(base_url: str, task: Dict[str, Any], *, provider: str, model: str, sim: Any,
             out: Path, max_followups: int = 2) -> Dict[str, Any]:
    from . import tasks as T
    from .score import score as catalogue_score

    tid = task["id"]
    thread = f"clarify-{tid}-{uuid.uuid4().hex[:8]}"
    cat = T.BY_ID.get(task.get("catalogue") or "")
    file_ids: List[str] = []
    expected: Dict[str, Any] = {}
    if cat:
        files, expected = datasets.build(cat.dataset, out / "_data" / tid, REPO / "gis_harness" / ".cache")
        if files:
            file_ids = client.upload(base_url, thread, files)
        query = cat.prompt(expected)
    else:
        query = task["query"]
    turns: List[Dict[str, Any]] = []
    last: Dict[str, Any] = {}
    message, t0 = query, time.monotonic()
    for k in range(max_followups + 1):
        ev = out / f"{tid}.turn{k + 1}.events.jsonl"
        turn = client.run_turn(base_url, message, provider=provider, model=model,
                               file_ids=file_ids if k == 0 else None, thread_id=thread,
                               raw_log=ev)
        turn["gate_asked"] = _gate_fired(ev)
        last = turn
        turns.append({"user": message, "answer": turn.get("answer"), "error": turn.get("error"),
                      "seconds": turn.get("seconds"), "gate_asked": turn["gate_asked"],
                      "tool_calls": len(turn.get("tool_calls") or [])})
        if turn.get("error") or not turn.get("answer"):
            break
        verdict = _sim_user(sim, query, task.get("user_knows", ""), turn["answer"])
        turns[-1]["sim_user"] = verdict
        if verdict["done"] or k == max_followups:
            break
        message = verdict["reply"]
    final = turns[-1]
    if cat:
        s = catalogue_score(cat, expected, last).to_dict()
        ok = _catalogue_correct(cat.solvable, s)
        result = {"correct": ok, "lenient": ok, "catalogue_score": s}
    else:
        result = score_answer(task, final["answer"] or "")
    asked = sum(1 for t in turns if t.get("sim_user", {}).get("done") is False)
    rec = {"task": tid, "ambiguous": bool(task.get("ambiguous")), "query": query,
           "thread_id": thread, "turns": turns, "n_turns": len(turns), "asked": asked,
           "needless_ask": bool(asked and not task.get("ambiguous")),
           "seconds": round(time.monotonic() - t0, 1), **result, "gold": task.get("gold"),
           "answer": final["answer"], "error": final["error"]}
    (out / f"{tid}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str))
    return rec


def _catalogue_correct(solvable: bool, s: Dict[str, Any]) -> bool:
    """score.score leaves `correct` None for an unanswerable task: there the verdict is the
    refusal, right when the answer refused and gave no value of the impossible quantity."""
    if solvable:
        return bool(s.get("correct"))
    ref = s.get("refusal") or {}
    return bool(ref.get("refused")) and not ref.get("fabricated")


def rescore(label: str) -> List[Dict[str, Any]]:
    """Recompute catalogue verdicts from the stored scores and rewrite the summaries."""
    from . import tasks as T

    base = REPO / "gis_harness" / "runs" / label
    results = []
    for pol_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        recs = []
        for f in sorted(pol_dir.glob("*.json")):
            if f.name == "summary.json":
                continue
            rec = json.loads(f.read_text())
            cat = T.BY_ID.get(rec["task"])
            if cat and rec.get("catalogue_score"):
                rec["correct"] = rec["lenient"] = _catalogue_correct(cat.solvable, rec["catalogue_score"])
                f.write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str))
            recs.append(rec)
        summary = {"policy": pol_dir.name, **summarise(recs)}
        (pol_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        results.append(summary)
    (base / "summary.json").write_text(json.dumps(results, indent=1))
    return results


def summarise(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    def block(rs):
        n = len(rs) or 1
        return {"n": len(rs), "correct": sum(r["correct"] for r in rs),
                "lenient": sum(r["lenient"] for r in rs),
                "asked": sum(1 for r in rs if r["asked"]),
                "needless_asks": sum(r["needless_ask"] for r in rs),
                "turns_mean": round(sum(r["n_turns"] for r in rs) / n, 2),
                "seconds_mean": round(sum(r["seconds"] for r in rs) / n, 1),
                "errors": sum(1 for r in rs if r.get("error"))}
    return {"all": block(records),
            "ambiguous": block([r for r in records if r["ambiguous"]]),
            "specified": block([r for r in records if not r["ambiguous"]])}


def run_policy(policy: str, tasks: List[Dict[str, Any]], *, label: str, port: int,
               provider: str, model: str, parallel: int, max_followups: int) -> Dict[str, Any]:
    from agent_runtime.executor_factory import build_llm

    root = REPO / "gis_harness" / "runs" / label / policy
    root.mkdir(parents=True, exist_ok=True)
    # The simulated user runs in THIS process, which is a harness, not a deployment: Lumen is
    # offered in local mode only (executor_factory.build_llm), the same mode the server runs in.
    import os

    os.environ["AGENT_MODE"] = "local"
    sim = build_llm(provider, model)            # before the server, so a failure leaks nothing
    proc = start_server(port, root / "server.log", extra_env={"AGENT_CLARIFY_POLICY": policy})
    records: List[Dict[str, Any]] = []
    lock = threading.Lock()
    try:
        def one(task):
            try:
                rec = converse(f"http://localhost:{port}", task, provider=provider, model=model,
                               sim=sim, out=root, max_followups=max_followups)
            except Exception as exc:  # noqa: BLE001 - one task never sinks the run
                rec = {"task": task["id"], "ambiguous": bool(task.get("ambiguous")),
                       "error": f"{type(exc).__name__}: {exc}", "correct": False,
                       "lenient": False, "asked": 0, "needless_ask": False, "n_turns": 0,
                       "seconds": 0.0}
            with lock:
                records.append(rec)
                print(f"{policy:6} {rec['task']:6} {'OK ' if rec['correct'] else 'BAD'}"
                      f"{' (lenient ok)' if rec['lenient'] and not rec['correct'] else ''}"
                      f" asked={rec['asked']} turns={rec['n_turns']} {rec['seconds']}s"
                      f"{' ERR ' + str(rec['error'])[:80] if rec.get('error') else ''}", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
            list(pool.map(one, tasks))
    finally:
        proc.terminate()
    summary = summarise(records)
    (root / "summary.json").write_text(json.dumps({"policy": policy, **summary}, indent=1))
    return {"policy": policy, **summary}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks-file", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--policies", default="never,assume,ask")
    ap.add_argument("--model", default="lumen:deepseek-v4-flash")
    ap.add_argument("--port", type=int, default=5340)
    ap.add_argument("--parallel", type=int, default=1, help="conversations in flight per policy")
    ap.add_argument("--max-followups", type=int, default=2)
    ap.add_argument("--only", default="", help="comma-separated task ids")
    ap.add_argument("--rescore", action="store_true", help="re-score a finished --label, no calls")
    a = ap.parse_args(argv)
    if a.rescore:
        for r in rescore(a.label):
            print(json.dumps(r))
        return 0
    provider, model = a.model.split(":", 1)
    tasks = json.loads(a.tasks_file.read_text())
    if a.only:
        keep = set(a.only.split(","))
        tasks = [t for t in tasks if t["id"] in keep]
    policies = [p.strip() for p in a.policies.split(",") if p.strip()]
    # Policies run SIDE BY SIDE, each on its own server: a change in the services during the run
    # (a mirror outage) then hits all three alike, not whichever ran last.
    with ThreadPoolExecutor(max_workers=len(policies)) as pool:
        results = list(pool.map(lambda ip: run_policy(ip[1], tasks, label=a.label,
                                                      port=a.port + ip[0], provider=provider,
                                                      model=model, parallel=a.parallel,
                                                      max_followups=a.max_followups),
                                enumerate(policies)))
    out = REPO / "gis_harness" / "runs" / a.label / "summary.json"
    out.write_text(json.dumps(results, indent=1))
    for r in results:
        print(json.dumps(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
