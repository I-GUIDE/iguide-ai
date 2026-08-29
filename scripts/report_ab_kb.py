#!/usr/bin/env python3
"""Summarise the KB ablation into something a reader can check.

    python scripts/report_ab_kb.py

Reads `outputs/ab_kb_problems.json` and prints, per problem, what each arm got and — where a
wrong number is diagnostic — WHICH mistake it made. Also prints the harness-health columns,
because a run with malformed LLM replies is evidence about the shim rather than about the
knowledge base, and averaging it in would let a harness artefact read as a result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    path = REPO / "outputs" / "ab_kb_problems.json"
    if not path.is_file():
        print(f"no results at {path}; run scripts/ab_kb_problems.py --all")
        return 2
    runs = json.loads(path.read_text())["runs"]
    by = {(r["problem"], r["arm"]): r for r in runs}
    problems = sorted({r["problem"] for r in runs})

    from scripts.ab_kb_problems import PROBLEMS

    totals = {"no_kb": [0, 0], "with_kb": [0, 0]}
    print(f"{len(runs)} runs · model={runs[0].get('model')} "
          f"provider={runs[0].get('provider')}\n")

    for problem in problems:
        meta = PROBLEMS.get(problem, {})
        print("=" * 92)
        print(f"{problem}")
        if meta.get("why"):
            print(f"  {meta['why']}")
        print("=" * 92)
        for arm in ("no_kb", "with_kb"):
            row = by.get((problem, arm))
            if not row:
                print(f"  {arm:<9} (not run)")
                continue
            grade = row["grade"]
            passed, total = grade["passed"], grade["total"]
            if total:
                totals[arm][0] += passed
                totals[arm][1] += total
            health = []
            if row.get("malformed_llm_replies"):
                health.append(f"{row['malformed_llm_replies']} malformed LLM replies")
            if row.get("peer_search_failures"):
                health.append(f"{row['peer_search_failures']} search failures")
            if row.get("error"):
                health.append(f"ERROR {row['error'][:60]}")
            print(f"  {arm:<9} {passed}/{total} correct   {row['elapsed_s']:>6.0f}s   "
                  f"code={'yes' if row.get('ran_code') else 'no ':<3}  "
                  f"kb={row.get('kb_tool_calls') or '-'}")
            if health:
                print(f"            ! {'; '.join(health)}")
            for check in grade["checks"]:
                mark = "OK  " if check["found"] else "MISS"
                why = f"   <- {'; '.join(check['diagnosis'])}" if check["diagnosis"] else ""
                print(f"            {mark} {check['label']:<32}{check['correct']:>10,}{why}")
            if "refused_correctly" in grade:
                print(f"            {'OK  ' if grade['refused_correctly'] else 'MISS'} "
                      f"refused a capability the library does not have")
        print()

    print("=" * 92)
    for arm in ("no_kb", "with_kb"):
        got, total = totals[arm]
        pct = f"{got / total:.0%}" if total else "n/a"
        print(f"  {arm:<9} {got}/{total} numeric checks correct  ({pct})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
