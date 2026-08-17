#!/usr/bin/env python3
"""Daily corpus-graph refresh: idempotent, id-stable, and loud when the corpus shrinks.

    python scripts/graph_daily.py --details <records.jsonl>            # from a cached dump
    python scripts/graph_daily.py --details ... --summarize            # + LLM reports for changed
    python scripts/graph_daily.py --details ... --force                # rebuild even if unchanged

WHY DAILY REBUILD AND NOT ONLINE INCREMENTAL — measured on this corpus:

    louvain partition                41 ms
    whole rebuild compute         2,628 ms   (embedding is 2,003 of it)
    fetch 750 records over REST 1,234,000 ms  (rate limit 500/600s)
    15 community reports          199,000 ms

Incremental community detection would optimise **41 ms of a 2.6 second job**. Fetching costs 470x
the entire compute, and the LLM reports cost 76x it. So the things worth making incremental are
the fetch and the summaries — never the clustering. Rebuilding from scratch is also strictly more
correct: an incrementally-maintained partition drifts from the one you would get by rebuilding,
and nothing would tell you.

THREE GUARDS, each from a failure this project actually produced:

1. **A partial fetch fails the run.** A previous corpus pass hit a 429 mid-listing, produced 134
   of 750 items, and looked clean. `--min-elements` and the shrink check make that impossible to
   report as success.
2. **Change detection hashes content, not `updated-at`.** That field is set on 249 of 750
   elements (33%); a cursor over it would ignore two thirds of the corpus.
3. **Community ids are reconciled, not re-enumerated.** Louvain numbers from zero each run and the
   partition genuinely moves — 29 citation edges (0.5% of edges) shifted this corpus from 15
   communities to 16. Without reconciliation a nightly job renames most communities most nights.

Exit codes: 0 success (including a no-change no-op), 1 a guard tripped, 2 a hard failure.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from agent_runtime.corpus_graph import (  # noqa: E402
    build_graph, citation_edges, collection_edges, community_profiles, corpus_fingerprint,
    curated_edges, fuse, knn_edges, load_elements, reconcile_communities,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", required=True)
    ap.add_argument("--out", default=str(REPO / "outputs" / "corpus_graph"))
    ap.add_argument("--corpus-dir", default=str(REPO / ".corpus_cache"))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--min-elements", type=int, default=700,
                    help="fail if fewer elements than this are loaded; a partial fetch must not "
                         "silently produce a smaller graph")
    ap.add_argument("--max-shrink", type=float, default=0.10,
                    help="fail if elements or edges drop by more than this fraction")
    ap.add_argument("--summarize", action="store_true",
                    help="regenerate LLM reports for CHANGED communities only")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    state_path = outdir / "daily_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    started = time.time()
    log = {"started_at": int(started), "resolution": args.resolution, "seed": args.seed}

    # ---------------------------------------------------------------- load + guard
    elements = load_elements(args.details)
    log["elements"] = len(elements)
    if len(elements) < args.min_elements:
        print(f"FAIL: loaded {len(elements)} elements, below --min-elements "
              f"{args.min_elements}. A partial corpus must not become a smaller graph.")
        return 1
    prev_n = state.get("elements")
    if prev_n and len(elements) < prev_n * (1 - args.max_shrink):
        print(f"FAIL: elements {prev_n} -> {len(elements)}, a drop of "
              f"{(1-len(elements)/prev_n):.1%} (limit {args.max_shrink:.0%}). "
              f"Investigate before accepting; pass --force to override.")
        if not args.force:
            return 1

    # ---------------------------------------------------------------- change detection
    fp = corpus_fingerprint(elements)
    log["fingerprint"] = fp
    if not args.force and state.get("fingerprint") == fp:
        print(f"no change: fingerprint {fp} matches the last run "
              f"({state.get('finished_iso','?')}). Nothing rebuilt.")
        log.update(outcome="noop", finished_iso=None)
        return 0
    if state.get("fingerprint"):
        print(f"corpus changed: {state['fingerprint']} -> {fp}")

    # ---------------------------------------------------------------- rebuild
    ids = [e.id for e in elements]
    id_by_prefix = {}
    listing = Path(args.corpus_dir) / "_elements_notebook.json"
    if listing.exists():
        data = json.loads(listing.read_text(encoding="utf-8"))
        els = data.get("elements") if isinstance(data, dict) else data
        id_by_prefix = {str(e.get("id"))[:8]: str(e.get("id")) for e in (els or []) if e.get("id")}

    edges = fuse(curated_edges(args.details, ids), collection_edges(args.details, ids),
                 knn_edges(elements, k=args.k),
                 citation_edges(args.corpus_dir, id_by_prefix, ids) if id_by_prefix else [])
    graph = build_graph(elements, edges)
    log["edges"] = graph.number_of_edges()

    prev_e = state.get("edges")
    if prev_e and graph.number_of_edges() < prev_e * (1 - args.max_shrink):
        print(f"FAIL: edges {prev_e} -> {graph.number_of_edges()}, a drop of "
              f"{(1-graph.number_of_edges()/prev_e):.1%} (limit {args.max_shrink:.0%}).")
        if not args.force:
            return 1

    from networkx.algorithms import community as nx_community

    parts = nx_community.louvain_communities(graph, weight="weight",
                                             resolution=args.resolution, seed=args.seed)
    parts = [sorted(p) for p in sorted(parts, key=lambda p: (-len(p), min(p)))]

    prev_members = {int(k): v for k, v in (state.get("members") or {}).items()}
    if prev_members:
        membership, events = reconcile_communities(prev_members, parts)
    else:
        membership = {n: i for i, p in enumerate(parts) for n in p}
        events = [{"community": i, "event": "new", "size": len(p)} for i, p in enumerate(parts)]

    profiles = community_profiles(graph, membership)
    log["communities"] = len(profiles)

    # ---------------------------------------------------------------- report the deltas
    tally = {}
    for ev in events:
        tally[ev["event"]] = tally.get(ev["event"], 0) + 1
    isolated = [n for n, d in graph.degree() if d == 0]
    print(f"\n  elements     {len(elements)}" + (f"  (was {prev_n})" if prev_n else ""))
    print(f"  edges        {graph.number_of_edges()}" + (f"  (was {prev_e})" if prev_e else ""))
    print(f"  communities  {len(profiles)}  " +
          " · ".join(f"{v} {k}" for k, v in sorted(tally.items())))
    print(f"  isolated     {len(isolated)}")
    for ev in events:
        if ev["event"] == "carried" and (ev.get("gained") or ev.get("lost")):
            print(f"    C{ev['community']:<3} carried  +{ev['gained']} -{ev['lost']} "
                  f"(jaccard {ev['jaccard']})")
        elif ev["event"] != "carried":
            print(f"    C{ev['community']:<3} {ev['event'].upper()}"
                  + (f" -> {ev.get('into')}" if ev.get("into") else f"  size {ev['size']}"))

    # ---------------------------------------------------------------- persist
    changed = sorted({ev["community"] for ev in events
                      if ev["event"] != "carried" or ev.get("gained") or ev.get("lost")})
    log["changed_communities"] = changed

    from scripts.build_corpus_graph import community_layout  # reuse the same layout

    pos = community_layout(graph, membership, seed=args.seed)
    by_id = {e.id: e for e in elements}
    order = sorted(graph.nodes())
    idx = {nid: i for i, nid in enumerate(order)}
    payload = {
        "stats": {"elements": len(elements), "edges": graph.number_of_edges(),
                  "communities": len(profiles), "isolated": len(isolated),
                  "modularity": round(nx_community.modularity(
                      graph, [set(p) for p in parts], weight="weight"), 4),
                  "resolution": args.resolution, "seed": args.seed, "k": args.k,
                  "fingerprint": fp},
        "nodes": [{"id": n, "t": by_id[n].title, "rt": by_id[n].resource_type,
                   "c": membership[n], "x": round(pos[n][0], 3), "y": round(pos[n][1], 3),
                   "d": graph.degree(n), "tags": list(by_id[n].tags)[:8],
                   "by": by_id[n].contributor, "clicks": by_id[n].click_count} for n in order],
        "links": sorted(({"s": idx[a], "t": idx[b], "w": round(float(d.get("weight", 1)), 4),
                          "r": d.get("rel", ""), "sup": d.get("support", 1)}
                         for a, b, d in graph.edges(data=True)),
                        key=lambda e: (e["s"], e["t"])),
        "communities": [{k: v for k, v in p.items() if k != "members"} for p in profiles],
        "events": events,
    }

    # Carry forward reports for communities that did not change; only changed ones are re-written.
    prior_reports = {}
    rp = outdir / "community_reports.json"
    if rp.exists():
        prior_reports = json.loads(rp.read_text(encoding="utf-8"))
    payload["reports"] = {k: v for k, v in prior_reports.items() if int(k) not in changed}
    (outdir / "graph.json").write_text(json.dumps(payload), encoding="utf-8")
    (outdir / "communities.json").write_text(json.dumps(profiles, indent=1), encoding="utf-8")

    if args.summarize and changed:
        print(f"\n  regenerating reports for {len(changed)} changed community(ies) "
              f"— {len(payload['reports'])} carried forward unchanged")
        r = subprocess.run([sys.executable, str(REPO / "scripts" / "summarize_communities.py"),
                            "--graph", str(outdir)], cwd=str(REPO))
        if r.returncode != 0:
            print("  WARNING: summarisation reported failures; graph is still written")
    elif args.summarize:
        print("\n  no community changed; no LLM calls made")

    state_path.write_text(json.dumps({
        "fingerprint": fp, "elements": len(elements), "edges": graph.number_of_edges(),
        "communities": len(profiles),
        "members": {str(p["community"]): p["members"] for p in profiles},
        "finished_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "seconds": round(time.time() - started, 1),
    }, indent=1), encoding="utf-8")
    print(f"\ndone in {time.time()-started:.1f}s · state at {state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
