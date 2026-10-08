#!/usr/bin/env python3
"""Render recorded agent traces as a side-by-side document.

    python scripts/report_agent_traces.py
    python scripts/report_agent_traces.py --out outputs/agent_traces.md

Reads `outputs/agent_traces.json` and writes Markdown: for each case, the question, the ground
truth read from the data before any run, then every step both arms took in order, then the two
answers judged against that truth.

The step list is the point. Counts of tool calls said the with-KB arm "searched the library and
wrote its own code anyway"; only the ordered arguments and results show whether that is what
happened.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

KB_TOOLS = {"agent_kb_search", "get_kb_block", "kb_method_search", "get_method_contract"}


def _check(answer: str, truth: Dict[str, Any]) -> List[str]:
    """Judge the answer against values read from the data, not from another model."""
    out = []
    text = " ".join((answer or "").split())
    for key, value in (truth or {}).items():
        if isinstance(value, list):
            hit = all(str(v) in text for v in value)
            out.append(f"{'OK  ' if hit else 'MISS'} {key}: {value}")
        elif isinstance(value, (int, float)):
            hit = re.search(rf"(?<![\d.]){re.escape(str(value))}(?![\d])", text) is not None
            out.append(f"{'OK  ' if hit else 'MISS'} {key}: {value}")
        else:
            hit = str(value).lower() in text.lower()
            out.append(f"{'OK  ' if hit else 'MISS'} {key}: {value}")
    return out


def render(trace: Dict[str, Any]) -> str:
    arm = "WITHOUT the knowledge base" if trace["ablated"] else "WITH the knowledge base"
    lines = [f"### {arm}", ""]
    kb_calls = sum(1 for s in trace["steps"] if s["tool"] in KB_TOOLS)
    lines.append(f"`{trace['elapsed_s']}s` · {len(trace['steps'])} tool calls · "
                 f"{kb_calls} of them knowledge-base calls")
    if trace.get("error"):
        lines.append(f"\n**error:** {trace['error']}")
    lines += ["", "| # | tool | s | argument | what came back |",
              "|---|---|---|---|---|"]
    for step in trace["steps"]:
        mark = "**" if step["tool"] in KB_TOOLS else ""
        args = str(step.get("args", "")).replace("|", "\\|")[:150]
        result = str(step.get("error") or step.get("result", "")).replace("|", "\\|")[:190]
        lines.append(f"| {step['n']} | {mark}{step['tool']}{mark} | {step.get('seconds', 0):.0f} "
                     f"| `{args}` | {result} |")
    lines += ["", "**Answer**", "", "> " + (trace["final_answer"] or "(none)")
              .strip().replace("\n", "\n> ")[:1800], ""]
    if trace.get("truth"):
        lines += ["**Judged against the data**", ""]
        lines += [f"- `{c}`" for c in _check(trace["final_answer"], trace["truth"])]
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", default=str(REPO / "outputs" / "agent_traces.json"))
    ap.add_argument("--out", default=str(REPO / "outputs" / "agent_traces.md"))
    args = ap.parse_args()

    path = Path(args.traces)
    if not path.is_file():
        print(f"no traces at {path}; run scripts/trace_agent_run.py first")
        return 2
    traces = json.loads(path.read_text())["traces"]

    doc = ["# What the agent does, step by step, with and without the knowledge base", "",
           "Recorded by wrapping `BaseTool.run`, the one hook every tool passes through — the",
           "granular tools are module-level functions, `execute_code` is a closure inside a",
           "factory, and the staging tools are built per session, so wrapping the factories",
           "would have missed two of the three.",
           "",
           "The two arms differ in exactly one thing: whether the knowledge base exists.",
           "`AGENT_ABLATE_KB` removes it from all four paths it otherwise reaches the agent by —",
           "the search peer's tool list, the code and analyze peers' standing grant, and the two",
           "arms of the deterministic evidence sweep.",
           "",
           f"Model: `{traces[0].get('model')}` via the development CLI provider.", ""]

    for case in sorted({t["case"] for t in traces}):
        rows = [t for t in traces if t["case"] == case]
        first = rows[0]
        doc += [f"## Case: `{case}`", "", f"**Question.** {first['query']}", "",
                f"**Why this case.** {first['why']}", ""]
        if first.get("truth"):
            doc += ["**Ground truth**, read directly from the data before any run:", ""]
            doc += [f"- `{k}` = `{v}`" for k, v in first["truth"].items()]
            doc.append("")
        for arm in ("no_kb", "with_kb"):
            trace = next((t for t in rows if t["arm"] == arm), None)
            if trace:
                doc += [render(trace), ""]

    out = Path(args.out)
    out.write_text("\n".join(doc) + "\n")
    print(f"wrote {out}  ({out.stat().st_size / 1000:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
