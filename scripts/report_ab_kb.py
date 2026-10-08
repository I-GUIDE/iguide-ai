#!/usr/bin/env python3
"""Summarise the KB ablation into something a reader can check.

    python scripts/report_ab_kb.py

Reads `outputs/ab_kb_problems.json` and prints, per problem and arm, every replicate: what it
got, and — where a wrong number is diagnostic — WHICH mistake it made.

Replicates are printed individually rather than averaged. One run per cell cannot tell an effect
from the harness: the same arm on the same problem answered 2/2 and then 0/2, and the difference
was six malformed LLM replies. An average over two such runs would report 50% and say nothing.
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
    if not runs:
        print("no runs recorded")
        return 2

    sys.argv = ["report"]
    from scripts.ab_kb_problems import PROBLEMS

    problems = sorted({r["problem"] for r in runs})
    totals = {"no_kb": [0, 0], "with_kb": [0, 0]}
    degraded = {"no_kb": 0, "with_kb": 0}
    ran_code = {"no_kb": 0, "with_kb": 0}
    counted = {"no_kb": 0, "with_kb": 0}

    print(f"{len(runs)} runs · model={runs[0].get('model')} "
          f"provider={runs[0].get('provider')}\n")

    for problem in problems:
        meta = PROBLEMS.get(problem, {})
        print("=" * 96)
        print(problem)
        if meta.get("why"):
            print(f"  {meta['why']}")
        print("=" * 96)
        for arm in ("no_kb", "with_kb"):
            reps = sorted((r for r in runs if r["problem"] == problem and r["arm"] == arm),
                          key=lambda r: r.get("replicate", 0))
            if not reps:
                print(f"  {arm:<9} (not run)")
                continue
            for row in reps:
                grade = row["grade"]
                passed, total = grade["passed"], grade["total"]
                counted[arm] += 1
                if total:
                    totals[arm][0] += passed
                    totals[arm][1] += total
                if row.get("ran_code"):
                    ran_code[arm] += 1
                bad = (row.get("malformed_llm_replies") or 0)
                if bad or row.get("error"):
                    degraded[arm] += 1
                tag = f"{arm} #{row.get('replicate', 0)}"
                print(f"  {tag:<12} {passed}/{total} correct   {row['elapsed_s']:>6.0f}s   "
                      f"code={'yes' if row.get('ran_code') else 'no ':<3}  "
                      f"kb={row.get('kb_tool_calls') or '-'}")
                flags = []
                if bad:
                    flags.append(f"{bad} malformed LLM replies — a degraded turn, not a result")
                if row.get("error"):
                    flags.append(f"ERROR {row['error'][:60]}")
                for flag in flags:
                    print(f"               ! {flag}")
                for check in grade["checks"]:
                    mark = "OK  " if check["found"] else "MISS"
                    why = f"   <- {'; '.join(check['diagnosis'])}" if check["diagnosis"] else ""
                    print(f"               {mark} {check['label']:<30}"
                          f"{check['correct']:>10,}{why}")
                if "refused_correctly" in grade:
                    print(f"               {'OK  ' if grade['refused_correctly'] else 'MISS'} "
                          f"refused a capability the library does not have")
        print()

    print("=" * 96)
    for arm in ("no_kb", "with_kb"):
        got, total = totals[arm]
        pct = f"{got / total:.0%}" if total else "n/a"
        print(f"  {arm:<9} {got}/{total} numeric checks ({pct})   "
              f"reached the sandbox in {ran_code[arm]}/{counted[arm]} runs   "
              f"{degraded[arm]}/{counted[arm]} runs degraded")
    print("\n  A degraded run is evidence about the dev LLM shim, not about the knowledge base.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
