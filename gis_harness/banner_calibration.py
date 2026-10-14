"""How often each check's finding lands on a correct answer versus a wrong one.

    python -m gis_harness.banner_calibration RUN_DIR [RUN_DIR ...]

Reads every turn's recorded verdict (`findings`, from stage 44 on) and its harness score. A
finding on a correct answer is a false alarm if it is printed; one on a wrong answer is a catch.
The second table re-renders each turn under `agent_runtime.verdict.shown`, so it says how many
correct answers would still carry a banner under the current policy, and how many wrong answers
it would still mark. A check belongs in the banner only while its precision stays above a bar.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent_runtime import verdict as V

_FINDINGS_RE = re.compile(r'"findings": (\[.*?\]), "(?:numbers|status)"')


def _good(score: Dict[str, Any]) -> Optional[bool]:
    if score.get("solvable"):
        return score.get("correct")
    if score.get("refusal") is not None:
        return bool(score["refusal"])
    return score.get("correct")


def turns(run_dir: Path) -> List[Tuple[str, bool, List[V.Finding]]]:
    out = []
    for tj in sorted(run_dir.rglob("*.json")):
        ev = tj.with_name(tj.name[:-len(".json")] + ".events.jsonl")
        if not ev.exists():
            continue
        try:
            score = json.loads(tj.read_text()).get("score") or {}
        except ValueError:
            continue
        good = _good(score)
        if good is None:
            continue
        found = _FINDINGS_RE.findall(ev.read_text())
        if not found:
            continue                     # a run recorded before stage 44 has no verdict
        fs = [V.Finding(check=f.get("check", ""), kind=f.get("kind", ""),
                        message=f.get("message", ""), evidence=f.get("evidence") or [])
              for f in json.loads(found[-1])]
        out.append((f"{run_dir.name}/{tj.stem}", bool(good), fs))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    rows = [t for r in a.runs for t in turns(Path(r))]
    by_kind: collections.Counter = collections.Counter()
    for _, good, fs in rows:
        for key in {(f.check, f.kind) for f in fs}:
            by_kind[(key, good)] += 1
    print(f"{len(rows)} turns with a recorded verdict\n")
    print(f"{'check':14} {'kind':13} {'on correct':>10} {'on wrong':>9}  printed now")
    for key in sorted({k for k, _ in by_kind}):
        probe = V.Finding(key[0], key[1], "")
        where = ("banner" if V.shown(probe) else
                 "quiet line" if key == ("number_scan", V.NOTE) else "verdict only")
        print(f"{key[0]:14} {key[1]:13} {by_kind[(key, True)]:10} {by_kind[(key, False)]:9}  "
              f"{where}")
    before = collections.Counter((good, bool(fs)) for _, good, fs in rows)
    after = collections.Counter((good, any(V.shown(f) for f in fs)) for _, good, fs in rows)
    print("\nanswers carrying a banner        correct  wrong")
    print(f"  every finding printed (stage 44) {before[(True, True)]:6} {before[(False, True)]:6}")
    print(f"  problems and failed peers (46)   {after[(True, True)]:6} {after[(False, True)]:6}")
    print(f"  answers in total                 {sum(1 for _, g, _ in rows if g):6} "
          f"{sum(1 for _, g, _ in rows if not g):6}")


if __name__ == "__main__":
    main()
