"""Rank quality for `kb_method_search`: R@1 / R@3 / R@8 / MRR over the built method library.

WHY THIS EXISTS. Every ranking change to ``agent_runtime.method_library`` so far has been judged
by a number produced in a throwaway script and then quoted from memory — "R@1 5 of 8" appears in
this project's notes with no harness behind it. That is not a measurement, it is a recollection.
Two ranking changes were already made on that basis, and one of them (an IDF floor) had to be
reverted after it turned out to have moved the number the wrong way.

WHAT IT MEASURES, AND WHAT IT DOES NOT. Each case is a natural-language question paired with the
symbol(s) that should answer it. That measures **ranking among the candidates the library
already contains**. It says nothing about coverage, because the questions were written by reading
the library — I chose questions I knew were answerable. A rising score here means "the right unit
is easier to find", never "the library covers more".

Several cases accept more than one symbol on purpose. ``spatial_join_and_count`` exists twice
under different elements with the same docstring; ``e2sfca`` and ``ae2sfca`` are the plain and
adjusted forms of one method. A benchmark that demanded one specific answer there would be
measuring an arbitrary tie-break rather than retrieval.

    python scripts/eval_method_retrieval.py
    python scripts/eval_method_retrieval.py --show-misses
    python scripts/eval_method_retrieval.py --json outputs/method_retrieval.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# (question, any-of these bare symbol names is correct)
CASES: Sequence[tuple] = (
    ("how do I buffer a GeoDataFrame by a distance in metres",
     ("calculate_buffers",)),
    ("count how many points fall inside each polygon",
     ("spatial_join_and_count",)),
    ("compute two-step floating catchment area accessibility",
     ("e2sfca", "ae2sfca")),
    ("aggregate an accessibility measure to larger units by area share",
     ("aggregate_ratios_area", "aggregate_ratios_centroid")),
    ("sum an attribute proportionally to the overlapping area of two layers",
     ("sum_attribute_proportional_to_area",)),
    ("resource to population ratio for each catchment",
     ("catchment_ratios_area", "catchment_ratios_centroid")),
    ("energy balance ratio from uncorrected flux measurements",
     ("ebr_of",)),
    ("de-accumulate cumulative ERA5-Land hourly fluxes",
     ("deaccumulate",)),
    ("area-weighted permeability and porosity catchment attributes",
     ("camels_geology_attrs",)),
    ("run GeoShapley to explain a spatial model",
     ("run_geoshapley",)),
    ("visualise patch footprints on top of a raster",
     ("visualize_patch_grid_on_raster",)),
    ("overlap of each buffer with the regions around it",
     ("calculate_primary_regions", "calculate_secondary_regions")),
)


def _bare(symbol: str) -> str:
    return str(symbol or "").rsplit(".", 1)[-1]


def evaluate(limit: int = 8) -> Dict[str, Any]:
    from agent_runtime.method_library import library_summary, search_methods

    summary = library_summary()
    rows: List[Dict[str, Any]] = []
    for question, expected in CASES:
        hits = search_methods(question, limit=limit) or []
        names = [_bare(h.get("symbol")) for h in hits]
        rank = next((i + 1 for i, n in enumerate(names) if n in expected), None)
        rows.append({"question": question, "expected": list(expected), "rank": rank,
                     "returned": names[:limit]})

    n = len(rows) or 1
    def at(k):
        return sum(1 for r in rows if r["rank"] and r["rank"] <= k)
    return {
        "library": {"units": summary["units"], "elements": summary["elements"],
                    "root": summary.get("root")},
        "cases": n, "limit": limit,
        "recall_at_1": at(1), "recall_at_3": at(3), "recall_at_8": at(min(8, limit)),
        "mrr": round(sum(1.0 / r["rank"] for r in rows if r["rank"]) / n, 4),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--show-misses", action="store_true",
                    help="print what was returned instead, for every case not ranked 1")
    ap.add_argument("--json", help="also write the full result here")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

    result = evaluate(limit=args.limit)
    lib = result["library"]
    print(f"library: {lib['units']} units from {lib['elements']} elements")
    print(f"  root : {lib['root']}")
    if not lib["units"]:
        print("\nNo library on disk. Build one first:\n"
              "  python scripts/build_method_library.py --type notebook --relevance-gate")
        return 2

    n = result["cases"]
    print(f"\n{n} cases, window {result['limit']}")
    print(f"  R@1 {result['recall_at_1']}/{n}"
          f"   R@3 {result['recall_at_3']}/{n}"
          f"   R@8 {result['recall_at_8']}/{n}"
          f"   MRR {result['mrr']}")

    print(f"\n{'rank':<6}{'question':<58}expected")
    print("-" * 100)
    for row in result["rows"]:
        rank = str(row["rank"]) if row["rank"] else "MISS"
        print(f"{rank:<6}{row['question'][:56]:<58}{'|'.join(row['expected'])}")

    if args.show_misses:
        for row in result["rows"]:
            if row["rank"] == 1:
                continue
            print(f"\n  {row['question']}")
            print(f"    wanted : {'|'.join(row['expected'])}  (rank {row['rank'] or 'MISS'})")
            for i, name in enumerate(row["returned"], 1):
                mark = "<--" if name in row["expected"] else ""
                print(f"    {i:>2}. {name} {mark}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
