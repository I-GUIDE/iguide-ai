"""How many functions in the real notebook corpus are independently callable?

This is the headline number of the extraction restructure. If it is low, the analyzer is
telling the truth about the notebooks and the composition story needs parameterisation work
before it needs more plumbing — so it is worth measuring before building the library on top.

    python scripts/measure_callable_units.py [--notebooks DIR] [--json OUT]

Reads notebooks with the extractor's own R1 transform so the measurement matches what
ingestion would actually see, rather than a simplified reimplementation.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DEFAULT_NOTEBOOKS = Path("/Users/yfkang/i-guide-platform-flask-servers/agent_chat_files/eval_notebooks")


def notebook_module_source(path: Path) -> Tuple[str, Dict[str, Any]]:
    """Concatenate a notebook's transformed code cells into one module, R1-style.

    Mirrors ``NotebookExtractor._build_module_source``: only cells that PARSE are included,
    which is the per-function promotion premise — one bad cell must not lose the notebook.
    """
    import nbformat
    from extractors.r1_ipython_frontend import transform_cell

    nb = nbformat.read(str(path), as_version=4)
    meta = nb.get("metadata") or {}
    kernel = ((meta.get("kernelspec") or {}).get("language")
              or (meta.get("language_info") or {}).get("name") or "").lower()

    parts: List[str] = []
    stats = {"cells": 0, "parsed": 0, "failed": 0, "kernel": kernel or "unknown",
             "r_magic": False}
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source") or ""
        if not src.strip():
            continue
        stats["cells"] += 1
        if "%%R" in src or "%R " in src or "rpy2" in src:
            stats["r_magic"] = True
        try:
            transformed, note = transform_cell(src)
        except Exception:
            transformed, note = src, "transform_failed"
        try:
            import ast
            ast.parse(transformed)
            parts.append(transformed)
            stats["parsed"] += 1
        except SyntaxError:
            stats["failed"] += 1
    return "\n\n".join(parts), stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebooks", default=str(DEFAULT_NOTEBOOKS))
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    from extractors.analysis import analyze_module

    root = Path(args.notebooks)
    files = sorted(root.glob("*.ipynb"))
    if not files:
        print(f"no notebooks under {root}")
        return 1

    print(f"{len(files)} notebook(s) from {root}\n")
    hdr = f"{'notebook':<40}{'kernel':<9}{'cells':>6}{'bad':>5}{'units':>7}{'callable':>10}"
    print(hdr)
    print("-" * len(hdr))

    tot_units = tot_callable = 0
    blocked = collections.Counter()
    per_nb: Dict[str, Any] = {}
    nb_with_unit = 0

    for f in files:
        try:
            src, stats = notebook_module_source(f)
        except Exception as exc:
            print(f"{f.stem[:38]:<40}{'ERR':<9}  {type(exc).__name__}")
            continue
        verdicts, _scope, summary = analyze_module(src)
        n, ok = summary.get("total", 0), summary.get("callable", 0)
        tot_units += n
        tot_callable += ok
        if ok:
            nb_with_unit += 1
        for name, cnt in (summary.get("blocked_by") or {}).items():
            blocked[name] += cnt
        flag = " (R!)" if stats["r_magic"] or stats["kernel"] not in ("python", "unknown") else ""
        print(f"{f.stem[:38]:<40}{stats['kernel'][:8]:<9}{stats['cells']:>6}"
              f"{stats['failed']:>5}{n:>7}{ok:>10}{flag}")
        per_nb[f.stem] = {"kernel": stats["kernel"], "cells": stats["cells"],
                          "unparsed_cells": stats["failed"], "units": n, "callable": ok,
                          "r_magic": stats["r_magic"],
                          "needs_globals": summary.get("needs_globals", [])}

    print("-" * len(hdr))
    pct = (100 * tot_callable / tot_units) if tot_units else 0
    nb_pct = (100 * nb_with_unit / len(files)) if files else 0
    print(f"{'TOTAL':<40}{'':<9}{'':>6}{'':>5}{tot_units:>7}{tot_callable:>10}")
    print()
    print(f"  callable ratio: {tot_callable} of {tot_units} functions ({pct:.0f}%)")
    print(f"  notebooks contributing >=1 callable unit: {nb_with_unit}/{len(files)} ({nb_pct:.0f}%)")

    # THE limiting factor, and it is not the one the analyzer was built for. A notebook that
    # defines no function at all yields no callable unit no matter how clean its globals are:
    # per-function promotion has nothing to promote. Report it as a first-class number so the
    # roadmap is driven by supply, not by the blocker histogram alone.
    scriptish = [n for n, d in per_nb.items() if d["units"] == 0]
    concentration = sorted(((d["units"], n) for n, d in per_nb.items()), reverse=True)[:2]
    print()
    print(f"  SUPPLY: {len(scriptish)}/{len(files)} notebooks define ZERO functions "
          f"(script-style, straight-line cells)")
    if concentration and tot_units:
        top_n = sum(u for u, _ in concentration)
        print(f"  CONCENTRATION: the top 2 notebooks supply {top_n}/{tot_units} units "
              f"({100 * top_n / tot_units:.0f}%)")
    print()
    print("  top blockers (hidden-global class):")
    if blocked:
        for name, cnt in blocked.most_common(12):
            print(f"    {cnt:>4}x  {name}")
    else:
        print("    (none)")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "notebooks": len(files), "units": tot_units, "callable": tot_callable,
            "callable_pct": round(pct, 1), "notebooks_with_unit": nb_with_unit,
            "notebooks_with_zero_functions": [n for n, d in per_nb.items() if d["units"] == 0],
            "blocked_by": dict(blocked.most_common()), "per_notebook": per_nb,
        }, indent=2))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
