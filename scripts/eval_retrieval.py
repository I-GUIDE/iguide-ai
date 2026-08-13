"""LLM-free retrieval measurement over the GeoPathfinder benchmark.

Answers one question with no model in the loop: **for each retrieval method and each
window size k, how many of a task's expected elements come back?**

Why this exists separately from ``run_eval_cases.py``: that harness measures end-to-end
agent behavior, so a retrieval regression can hide behind a good answer and a retrieval
improvement can be masked by the model failing to use it. This script isolates the
retrieval layer, is deterministic, costs nothing, and runs in seconds — so it is the
instrument M1.2 (retrieval window) and M4 (KB reachability) are judged by.

Scores against the FULL expected set in ``eval_common.TASKS[tid][1]``, not the
``primary_ids`` subset (see M0.3 in docs/DEVLOG.md).

Requires the live cluster: ``kb_store.local_search`` is token-overlap counting, not
BM25, so offline numbers do not transfer. Semantic and agent_kb arms additionally need
a reachable embedding server; each arm reports its own availability rather than
silently scoring zero.

Usage
-----
    python scripts/eval_retrieval.py                       # default k=8,20,50
    python scripts/eval_retrieval.py --k 8,20 --methods keyword,union
    python scripts/eval_retrieval.py --json outputs/retrieval_$(date +%F).json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from dotenv import load_dotenv  # noqa: E402

# The repo's scripts load .env; this worktree may not have one, so fall back to the
# primary checkout's file rather than failing with a confusing missing-env error.
for candidate in (REPO / ".env", Path("/Users/yfkang/i-guide-platform-flask-servers/.env")):
    if candidate.exists():
        load_dotenv(candidate)
        break

import eval_common as ec  # noqa: E402

DEFAULT_KS = [8, 20, 50]
DEFAULT_METHODS = ["keyword", "semantic", "union", "union+agent_kb"]


# --------------------------------------------------------------------------- #
# Retrieval arms. Each returns an ordered list of doc ids, or None if the arm's
# backend is unavailable — None and "returned nothing" must not look the same.
# --------------------------------------------------------------------------- #

def _ids(hits: Any) -> List[str]:
    out = []
    for h in hits or []:
        src = h.get("_source") if isinstance(h, dict) and isinstance(h.get("_source"), dict) else h
        if not isinstance(src, dict):
            continue
        doc_id = src.get("doc_id") or src.get("id") or (h.get("_id") if isinstance(h, dict) else None)
        if doc_id:
            out.append(str(doc_id))
    return out


def arm_keyword(query: str, k: int) -> Optional[List[str]]:
    from rag_pipeline.search.keyword import get_keyword_search_results
    return _ids(get_keyword_search_results(query, size=k))


def arm_semantic(query: str, k: int) -> Optional[List[str]]:
    from rag_pipeline.search.semantic import semantic_search
    hits = semantic_search(query, size=k)
    # semantic_search returns [] both when the embedder is down and when nothing
    # matched; treat a total blank as unavailable so it is reported, not scored 0.
    return _ids(hits) if hits else None


def arm_neo4j(query: str, k: int) -> Optional[List[str]]:
    try:
        from rag_pipeline.search.neo4j import get_neo4j_search_results
        hits = get_neo4j_search_results(query, limit=k)
    except Exception:
        return None
    return _ids(hits) if hits else None


def arm_agent_kb(query: str, k: int) -> Optional[List[str]]:
    """Sub-document evidence, scored by the ELEMENT each block belongs to.

    Two traps, both hit here before this was written properly:

    * ``agent_kb_search`` returns ``{"documents": [...]}``, not a list of hits. Passing the
      dict to ``_ids`` iterates its KEYS, yields nothing, and returns ``[]`` — which scores as
      a hard 0 instead of being reported unavailable. (The same wrong key silently disabled
      this arm in the supervisor's sweep; see M4.3.)
    * a block's ``doc_id`` is ``<element>::block::<n>``, but the benchmark's expected ids are
      ELEMENT ids. Scoring raw block ids would report 0/37 for a perfectly working arm, since
      no block id can ever prefix-match an element id.
    """
    try:
        from rag_pipeline.search.agent_kb import agent_kb_search
        payload = agent_kb_search(query, size=k) or {}
    except Exception:
        return None
    docs = payload.get("documents") or []
    if not docs:
        return None
    out, seen = [], set()
    for d in docs:
        parent = str((d or {}).get("parent_doc_id") or "")
        if parent and parent not in seen:
            seen.add(parent)
            out.append(parent)
    return out or None


def _rrf(rankings: List[List[str]], k_rrf: int = 60) -> List[str]:
    """Reciprocal-rank fusion — the same fusion the agent's reranker uses, so a
    union number here is comparable to what the agent would see."""
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank)
    return [d for d, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


# Arms that COLLECT rather than rank. For these, k is the window given to each underlying
# arm, and recall is scored over everything collected — because that set is the ceiling on
# what the agent can possibly cite: the sweep merges every arm's hits, the reranker ORDERS
# them, and only then does the supervisor show its top few. Truncating a collection arm to k
# measures the arbitrary arm that happened to be concatenated first (union@8 read 22/37 —
# exactly keyword's own 22 — while the collected set held 29).
COLLECTION_ARMS = {"union", "union+agent_kb"}


def _set_union(parts: List[List[str]], k: int) -> List[str]:
    """Concatenate arms and dedupe, preserving first-seen order — what the agent does.

    ``_direct_search_sweep`` collects each arm's hits and merges them with ``_merge_dedup``;
    it does NOT rank-fuse and then truncate. Modelling union as RRF-then-cut made this
    instrument report a number the system never produces, and a PESSIMISTIC one: measured over
    the 37 expected ids, RRF@8 gave 25/37 where the set union gives 29/37, and RRF@20 gave
    29/37 against 31/37. The arms are genuinely complementary (at k=8: 19 found by both, 3 by
    keyword only, 7 by semantic only), so truncating a fused ranking to k discards correct
    hits that the agent in fact keeps and reranks.
    """
    out: List[str] = []
    seen = set()
    for ranking in parts:
        for doc_id in ranking[:k]:
            if doc_id not in seen:
                seen.add(doc_id)
                out.append(doc_id)
    return out


def arm_union(query: str, k: int) -> Optional[List[str]]:
    """The agent's actual behaviour: every arm's top-k, merged and deduped."""
    parts = [r for r in (arm_keyword(query, k), arm_semantic(query, k)) if r]
    return _set_union(parts, k) if parts else None


def arm_union_rrf(query: str, k: int) -> Optional[List[str]]:
    """Rank-fused and truncated to k. Kept to make the gap above reproducible, and as the
    right model for a ranked-list consumer — but it is NOT what the sweep does."""
    parts = [r for r in (arm_keyword(query, k), arm_semantic(query, k)) if r]
    return _rrf(parts)[:k] if parts else None


def arm_union_kb(query: str, k: int) -> Optional[List[str]]:
    parts = [r for r in (arm_keyword(query, k), arm_semantic(query, k), arm_agent_kb(query, k)) if r]
    return _set_union(parts, k) if parts else None


def arm_union_neo4j(query: str, k: int) -> Optional[List[str]]:
    """Kept so the negative result stays reproducible, and NOT part of ``union``.

    Once the graph arm returned real platform UUIDs it scored 20/37 alone — but every one of
    those 20 was already found by keyword (29/37), and it contributed **0 unique** elements.
    Fusing a strict subset into a fixed top-k window can only evict correct hits: measured
    union@20 fell 29/37 -> 26/37. The graph earns its keep through traversal (related
    elements, contributor/collection edges), not through lexical recall.
    """
    parts = [r for r in (arm_keyword(query, k), arm_semantic(query, k), arm_neo4j(query, k)) if r]
    return _rrf(parts)[:k] if parts else None


ARMS: Dict[str, Callable[[str, int], Optional[List[str]]]] = {
    "keyword": arm_keyword,
    "semantic": arm_semantic,
    "neo4j": arm_neo4j,
    "agent_kb": arm_agent_kb,
    "union": arm_union,
    "union_rrf": arm_union_rrf,
    "union+agent_kb": arm_union_kb,
    "union+neo4j": arm_union_neo4j,
}


# --------------------------------------------------------------------------- #
# Metrics. Expected ids are 8-char prefixes of full UUIDs, so match on prefix.
# --------------------------------------------------------------------------- #

def _hit_ranks(retrieved: List[str], expected: List[str]) -> Dict[str, Optional[int]]:
    return {e: next((i + 1 for i, d in enumerate(retrieved) if d.startswith(e)), None)
            for e in expected}


def score(retrieved: List[str], expected: List[str], k: int) -> Dict[str, Any]:
    ranks = _hit_ranks(retrieved[:k], expected)
    found = [e for e, r in ranks.items() if r is not None]
    first = min((r for r in ranks.values() if r), default=None)
    return {
        "recall": f"{len(found)}/{len(expected)}",
        "n_found": len(found),
        "n_expected": len(expected),
        "precision": round(len(found) / k, 3) if k else 0.0,
        "mrr": round(1.0 / first, 3) if first else 0.0,
        "ranks": {e: r for e, r in ranks.items()},
        "missing": [e for e, r in ranks.items() if r is None],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", default=",".join(str(k) for k in DEFAULT_KS))
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    ap.add_argument("--tasks", default="", help="comma-separated task ids; default all")
    ap.add_argument("--json", default="", help="write the full result to this path")
    args = ap.parse_args()

    ks = [int(x) for x in args.k.split(",") if x.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in ARMS]
    if unknown:
        ap.error(f"unknown method(s) {unknown}; choose from {sorted(ARMS)}")

    tids = ([t.strip() for t in args.tasks.split(",") if t.strip()]
            or [t for t in ec.TASKS if ec.TASKS[t][1]])

    print(f"index={os.getenv('OPENSEARCH_INDEX')}  tasks={len(tids)}  "
          f"k={ks}  methods={methods}\n")

    results: Dict[str, Any] = {"index": os.getenv("OPENSEARCH_INDEX"),
                              "ks": ks, "methods": methods, "tasks": {}}
    unavailable: set = set()

    # header
    w = 15
    print("TASK".ljust(w) + "exp  " + "".join(f"{m}@{k}".ljust(13) for m in methods for k in ks))
    print("-" * (w + 5 + 13 * len(methods) * len(ks)))

    totals: Dict[Tuple[str, int], List[int]] = {(m, k): [0, 0] for m in methods for k in ks}

    for tid in tids:
        prompt, expected = ec.TASKS[tid][0], list(ec.TASKS[tid][1])
        row = {"expected": expected, "arms": {}}
        cells = []
        for m in methods:
            retrieved = None
            for k in ks:
                # Re-retrieve per k for collection arms: their result depends on the per-arm
                # window, so reusing a single max(ks) fetch would report the same number for
                # every k and hide the window's effect entirely.
                if retrieved is None or m in COLLECTION_ARMS:
                    retrieved = ARMS[m](prompt, k if m in COLLECTION_ARMS else max(ks))
                if retrieved is None:
                    unavailable.add(m)
                    cells.append("—".ljust(13))
                    continue
                # A collection arm is scored over its whole merged set; k was the per-arm
                # window, not an output cap. See COLLECTION_ARMS.
                window = len(retrieved) if m in COLLECTION_ARMS else k
                s = score(retrieved, expected, window)
                row["arms"][f"{m}@{k}"] = s
                totals[(m, k)][0] += s["n_found"]
                totals[(m, k)][1] += s["n_expected"]
                cells.append(f"{s['recall']} m{s['mrr']:.2f}".ljust(13))
        results["tasks"][tid] = row
        print(tid.ljust(w) + str(len(expected)).ljust(5) + "".join(cells))

    print("-" * (w + 5 + 13 * len(methods) * len(ks)))
    print("TOTAL".ljust(w) + "".ljust(5) +
          "".join((f"{n}/{d}" if d else "—").ljust(13) for (m, k), (n, d) in totals.items()))
    print()
    results["totals"] = {f"{m}@{k}": {"recall": f"{n}/{d}",
                                      "pct": round(100 * n / d, 1) if d else None}
                         for (m, k), (n, d) in totals.items()}
    for key, val in results["totals"].items():
        if val["pct"] is not None:
            print(f"  recall {key:<20} = {val['recall']:<8} ({val['pct']}%)")

    if unavailable:
        print(f"\n  UNAVAILABLE (reported, not scored 0): {sorted(unavailable)}")
        if "semantic" in unavailable:
            print("  semantic needs a reachable FLASK_EMBEDDING_URL; .env may point at a dead host.")
        if "agent_kb" in unavailable:
            print("  agent_kb needs the iguide_agent_* indices (M1.3) or AGENT_KB_BACKEND=local data.")

    # Elements no arm reached at the LARGEST k tested. Distinguish carefully: these are
    # candidates for indexing gaps, but only a sweep at a much larger k can tell a true
    # gap from a ranking problem. Raising --k is how you separate them.
    never = sorted({e for t in results["tasks"].values()
                    for arm in t["arms"].values() for e in arm["missing"]
                    if all(e in a["missing"] for a in t["arms"].values())})
    if never:
        print(f"\n  missed by every arm at every k tested ({len(never)}): {never}")
        print(f"  -> a bigger window may still recover some of these; re-run with a "
              f"larger --k than {max(ks)} to tell a ranking problem from an indexing gap.")
    results["missed_at_all_k"] = never

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
