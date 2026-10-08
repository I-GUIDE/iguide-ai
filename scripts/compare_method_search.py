#!/usr/bin/env python3
"""Can the agent find its methods better through Postgres than through the JSON registry?

    python scripts/compare_method_search.py --out outputs/method_search_compare.json

Both arms rank the SAME 840 units and return the same result shape, so this measures the store
and the ranking, not the corpus:

  registry   agent_runtime.method_library.search_methods — reads the 2.5 MB `_registry.json`,
             re-derives IDF over it, sums weighted token overlap. What the agent uses today.
  postgres   extractors.kb_db.search_units — GIN index over a weighted tsvector, ts_rank.

Three measurements, each with an objective answer rather than a judgement:

  A. known-item — query with a unit's own doc_summary; where does that unit rank? A store that
     cannot return a method when handed its own description will not find it from a question.
  B. by element title — query with the parent element's title; is any unit from it in the top 5?
     `element_title` is absent from all 840 provenance records, so the registry's `element`
     field falls back to a hex id and its 1.0 weight is dead. It still scores 92.6%, because
     the registry KEY embeds a slug of the title — truncated to ~50 characters, so
     "…Mapping using Physics-Aware Spatial AI" is stored as "…mapping_using_phy". Measure the
     gap rather than assuming it: a first draft of this file asserted the registry "cannot
     answer at all", which the numbers disproved.
  C. latency — per query, both the warm path and the one the agent actually calls.

Nothing here writes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _identity(hit: Dict[str, Any]) -> tuple:
    """(element, bare symbol) — the one identity both arms can produce."""
    return (str(hit.get("element_id") or ""),
            str(hit.get("symbol") or "").rsplit(".", 1)[-1])


def _rank_of(hits: List[Dict[str, Any]], want: tuple) -> int:
    for i, hit in enumerate(hits, start=1):
        if _identity(hit) == want:
            return i
    return 0


def _report(name: str, ranks: List[int], k_values=(1, 5, 10)) -> Dict[str, Any]:
    n = len(ranks)
    found = [r for r in ranks if r]
    out = {"arm": name, "queries": n,
           "mrr": round(sum(1 / r for r in found) / n, 4) if n else 0.0}
    for k in k_values:
        out[f"recall@{k}"] = round(sum(1 for r in found if r <= k) / n, 4) if n else 0.0
    out["never_found"] = n - len(found)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "outputs" / "method_search_compare.json"))
    ap.add_argument("--limit", type=int, default=10, help="result window")
    ap.add_argument("--sample", type=int, default=250, help="known-item queries")
    ap.add_argument("--dsn", default=None)
    args = ap.parse_args()

    from agent_runtime import method_library
    from extractors import kb_db

    registry = method_library.load_registry()
    real = {k: v for k, v in registry.items()
            if isinstance(v, dict) and v.get("signature")
            and not v.get("alias_for") and not v.get("ambiguous")}
    print(f"registry: {len(real)} rankable units")

    with kb_db.connect(args.dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM unit WHERE verdict = 'callable'")
            print(f"postgres: {cur.fetchone()[0]} rankable units")

        # ---- A. known-item ---------------------------------------------------------
        cases = []
        for key, entry in sorted(real.items()):
            summary = (entry.get("doc_summary") or "").strip()
            element = str((entry.get("provenance") or {}).get("element_id") or "")
            symbol = str(entry.get("library_symbol") or "").rsplit(".", 1)[-1]
            if len(summary) >= 20 and element and symbol:
                cases.append({"query": summary, "want": (element, symbol)})
        step = max(1, len(cases) // max(1, args.sample))
        cases = cases[::step][: args.sample]
        print(f"\nA. known-item over {len(cases)} units with a usable summary")

        ranks = {"registry": [], "postgres": []}
        times = {"registry": [], "registry_cold": [], "postgres": []}
        for case in cases:
            t0 = time.perf_counter()
            hits = method_library.search_methods(case["query"], limit=args.limit,
                                                 registry=registry)
            times["registry"].append(time.perf_counter() - t0)
            ranks["registry"].append(_rank_of(hits, case["want"]))

            # The path the AGENT takes. `search_methods()` with no registry argument re-reads
            # the 2.5 MB file and re-derives IDF over it on every call, deliberately: "ingest
            # runs in a different process and a long-lived agent server would otherwise serve a
            # registry from before the last extraction". Correct, and it is what the tool costs.
            t0 = time.perf_counter()
            method_library.search_methods(case["query"], limit=args.limit)
            times["registry_cold"].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            hits = kb_db.search_units(conn, case["query"], limit=args.limit)
            times["postgres"].append(time.perf_counter() - t0)
            ranks["postgres"].append(_rank_of(hits, case["want"]))

        known_item = [_report(arm, ranks[arm]) for arm in ("registry", "postgres")]
        for row in known_item:
            print(f"   {row['arm']:<10} MRR {row['mrr']:.3f}   "
                  f"r@1 {row['recall@1']:.3f}  r@5 {row['recall@5']:.3f}  "
                  f"r@10 {row['recall@10']:.3f}   never found {row['never_found']}")

        # ---- B. by element title ----------------------------------------------------
        with conn.cursor() as cur:
            cur.execute("""
                SELECT e.id, e.title, count(*) FROM unit u JOIN element e ON e.id = u.element_id
                 WHERE u.verdict = 'callable' AND e.title <> ''
                 GROUP BY e.id, e.title HAVING count(*) > 0 ORDER BY e.id
            """)
            titled = cur.fetchall()
        print(f"\nB. by element title over {len(titled)} elements that have callable units")

        by_title = {"registry": 0, "postgres": 0}
        for element_id, title, _n in titled:
            hits = method_library.search_methods(title, limit=5, registry=registry)
            if any(str(h.get("element_id") or "") == element_id for h in hits):
                by_title["registry"] += 1
            hits = kb_db.search_units(conn, title, limit=5)
            if any(str(h.get("element_id") or "") == element_id for h in hits):
                by_title["postgres"] += 1
        total = len(titled) or 1
        for arm in ("registry", "postgres"):
            print(f"   {arm:<10} a unit from the right element in the top 5: "
                  f"{by_title[arm]}/{total} ({by_title[arm] / total:.1%})")

        latency = {arm: {"median_ms": round(statistics.median(times[arm]) * 1000, 2),
                         "p95_ms": round(sorted(times[arm])[int(len(times[arm]) * 0.95) - 1]
                                         * 1000, 2)}
                   for arm in times}
        print(f"\nC. latency over {len(cases)} queries")
        for arm in ("registry", "registry_cold", "postgres"):
            print(f"   {arm:<10} median {latency[arm]['median_ms']:>8.2f} ms   "
                  f"p95 {latency[arm]['p95_ms']:>8.2f} ms")

    payload = {"units_registry": len(real), "known_item": known_item,
               "by_element_title": {"elements": total, **by_title},
               "latency": latency, "window": args.limit}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
