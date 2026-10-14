"""Replay the number scan offline over recorded answers, with no model calls.

    python -m gis_harness.replay_number_scan RUN_DIR [RUN_DIR ...] [--show] [--json OUT]

RUN_DIR is a harness or archived run folder holding `<turn>.json` (with `query` and `answer`)
and `<turn>.events.jsonl`. The fact set is rebuilt from the recorded tool results and the query.
The turn's typed log is not in the record, so this is slightly stricter than the live scan.

It reports, per run, the stated figures the scan would leave unresolved. In the live pipeline
each of those is either cut (no tool could produce it) or listed as not checked. Run it before
and after a change to `agent_runtime/facts.py`, over answers the scan never touched (runs made
without it), to see how many correct statements the change stops cutting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from agent_runtime import facts as turn_facts


def tool_results(events_path: Path) -> List[Dict[str, Any]]:
    """`{name, content}` for every tool result in a recorded event stream.

    Harness streams carry them as `analysis` events with the content in `data.detail`; stored
    chat traces carry them as `tool_result` events with the content in `data`."""
    out: List[Dict[str, Any]] = []
    for line in events_path.read_text().splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        d = e.get("data") or {}
        if not isinstance(d, dict):
            continue
        det = d.get("detail") if isinstance(d.get("detail"), dict) else None
        if e.get("event") == "analysis" and d.get("type") == "tool_result" and det:
            out.append({"name": det.get("tool_name") or det.get("name"),
                        "content": det.get("content")})
        elif e.get("event") == "tool_result" and d.get("kind") == "tool_result":
            out.append({"name": d.get("tool_name") or d.get("name"), "content": d.get("content")})
    return out


def _starts_mid_number(answer: str, q: Any) -> bool:
    sent = (q.sentence or "").lstrip()
    i = answer.find(sent) if sent else -1
    return i >= 2 and answer[i - 1] == "." and answer[i - 2].isdigit() and sent[:1].isdigit()


def replay(run_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for tj in sorted(run_dir.rglob("*.json")):
        if tj.name.endswith(".bounds.json") or tj.name == "summary.json" or "layers" in tj.parts:
            continue
        ev = tj.with_name(tj.name[:-len(".json")] + ".events.jsonl")
        if not ev.exists():
            continue
        try:
            turn = json.loads(tj.read_text())
        except ValueError:
            continue
        answer, query = turn.get("answer") or "", turn.get("query") or ""
        if not answer:
            continue
        fs = turn_facts.build(results=tool_results(ev), query=query)
        res = turn_facts.resolve(answer, fs)
        claims = [r for r in res if turn_facts.is_claim(r.quantity)]
        unresolved = [r for r in claims if not r.resolved]
        # A sentence that starts inside a number ("8827° N, …" out of "41.8827° N"): what the
        # pre-stage-46 splitter produced, and what a cut then removed in pieces.
        split = [r for r in claims if _starts_mid_number(answer, r.quantity)]
        rows.append({"run": run_dir.name, "turn": tj.stem, "claims": len(claims),
                     "unresolved": len(unresolved), "split_sentences": len(split),
                     "figures": [{"text": r.quantity.text, "sentence": r.quantity.sentence}
                                 for r in unresolved]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--show", action="store_true", help="print every unresolved figure")
    ap.add_argument("--json", help="write every row to this file")
    a = ap.parse_args()
    allrows: List[Dict[str, Any]] = []
    for r in a.runs:
        rows = replay(Path(r))
        allrows += rows
        c = sum(x["claims"] for x in rows)
        u = sum(x["unresolved"] for x in rows)
        sp = sum(x["split_sentences"] for x in rows)
        print(f"{Path(r).name:28} turns {len(rows):3}  figures {c:5}  unresolved {u:4} "
              f"({100 * u / max(c, 1):.1f}%)  in a split sentence {sp:4}")
        if a.show:
            for x in rows:
                for f in x["figures"]:
                    print(f"    {x['turn']:8} {f['text']:>14} | {f['sentence'][:100]}")
    if a.json:
        Path(a.json).write_text(json.dumps(allrows, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
