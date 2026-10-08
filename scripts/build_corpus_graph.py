#!/usr/bin/env python3
"""Build the corpus knowledge graph and emit a self-contained payload for visualisation.

    python scripts/build_corpus_graph.py --details <g1_details.jsonl> --out outputs/corpus_graph
    python scripts/build_corpus_graph.py --details ... --resolution 1.5 --k 10

Layout is computed HERE, not in the browser. 750 nodes and ~9k edges settle slowly and
nondeterministically in a client-side force simulation, so two people opening the page would see
different pictures of the same graph — which for a reproducibility-focused system is the wrong
default. The layout is community-aware: communities are placed on a circle by size, then nodes
relax within their own community, so the picture shows the partition instead of a hairball.

Every number printed is measured from the graph just built. `--assert-stable` re-runs the whole
partition a second time and fails if the membership differs, because the entire value of the
canonical insertion order in corpus_graph.build_graph is that this can never happen quietly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from agent_runtime.corpus_graph import (  # noqa: E402
    build_corpus_graph, build_graph, fuse, partition,
)


def community_layout(graph, membership, *, seed: int = 17, spread: float = 11.0):
    """Communities on a circle sized by membership; nodes relaxed inside their own community.

    A single global spring layout over 9k edges puts everything in one blob — the kNN layer
    connects all 750 nodes, so there is no visual separation to find. Laying each community out
    independently and then translating it makes the partition legible, which is the thing being
    visualised.
    """
    import math

    import networkx as nx
    import numpy as np

    groups: dict = {}
    for node, cid in membership.items():
        groups.setdefault(cid, []).append(node)
    order = sorted(groups, key=lambda c: (-len(groups[c]), c))

    pos: dict = {}
    n = len(order)
    for rank, cid in enumerate(order):
        members = sorted(groups[cid])
        angle = 2.0 * math.pi * rank / max(1, n)
        radius = spread * (1.0 + 0.14 * math.log1p(rank))
        cx, cy = radius * math.cos(angle), radius * math.sin(angle)

        sub = graph.subgraph(members)
        if sub.number_of_nodes() == 1:
            local = {members[0]: np.array([0.0, 0.0])}
        else:
            # k scales the ideal edge length; seeded and iteration-capped so it replays.
            local = nx.spring_layout(sub, weight="weight", seed=seed, iterations=120,
                                     k=1.6 / math.sqrt(sub.number_of_nodes()))
        scale = 2.6 + 1.5 * math.log1p(len(members))
        for node, (x, y) in local.items():
            pos[node] = (cx + float(x) * scale, cy + float(y) * scale)

    for node in graph.nodes():
        pos.setdefault(node, (0.0, 0.0))
    return pos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", required=True,
                    help="JSONL/JSON of platform element records (must carry related-elements)")
    ap.add_argument("--corpus-dir", default=str(REPO / ".corpus_cache"),
                    help="cached .ipynb dir, for CITES/USES edges")
    ap.add_argument("--listing", default="",
                    help="element listing JSON used to map an 8-char filename prefix to an id")
    ap.add_argument("--out", default=str(REPO / "outputs" / "corpus_graph"))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--assert-stable", action="store_true",
                    help="re-partition and fail if membership differs")
    args = ap.parse_args()

    id_by_prefix = {}
    listing = args.listing or str(Path(args.corpus_dir) / "_elements_notebook.json")
    if Path(listing).exists():
        data = json.loads(Path(listing).read_text(encoding="utf-8"))
        els = data.get("elements") if isinstance(data, dict) else data
        id_by_prefix = {str(e.get("id"))[:8]: str(e.get("id")) for e in (els or []) if e.get("id")}

    print(f"building from {args.details} …")
    cg = build_corpus_graph(args.details, corpus_dir=args.corpus_dir, id_by_prefix=id_by_prefix,
                            k=args.k, resolution=args.resolution, seed=args.seed)
    s = cg.stats
    print(f"  elements           {s['elements']}")
    print(f"  edges              {s['edges']}")
    for name, n in s["layer_pairs"].items():
        print(f"    {name:<14} {n} pairs")
    print(f"  components         {s['components']}")
    print(f"  isolated           {s['isolated']}  "
          f"(without the kNN layer: {s['isolated_without_knn']})")
    print(f"  communities        {s['communities']}  modularity {s['modularity']}")
    print(f"  single-layer edges {s['single_layer_edge_fraction']:.1%} "
          f"— this is a union, not a consensus")

    if args.assert_stable:
        again = partition(build_graph(cg.elements, cg.edges),
                          resolution=args.resolution, seed=args.seed)
        if again != cg.membership:
            moved = sum(1 for k, v in again.items() if cg.membership.get(k) != v)
            print(f"  FAIL: partition is not reproducible — {moved} node(s) moved")
            return 1
        print("  partition replays identically ✓")

    pos = community_layout(cg.graph, cg.membership, seed=args.seed)
    by_id = {e.id: e for e in cg.elements}

    nodes = []
    for nid in sorted(cg.graph.nodes()):
        el = by_id[nid]
        x, y = pos[nid]
        nodes.append({
            "id": nid, "t": el.title, "rt": el.resource_type,
            "c": cg.membership[nid], "x": round(x, 3), "y": round(y, 3),
            "d": cg.graph.degree(nid), "tags": list(el.tags)[:8],
            "by": el.contributor, "clicks": el.click_count,
        })

    idx = {nid: i for i, nid in enumerate(sorted(cg.graph.nodes()))}
    links = []
    for a, b, data in cg.graph.edges(data=True):
        links.append({"s": idx[a], "t": idx[b], "w": round(float(data.get("weight", 1.0)), 4),
                      "r": data.get("rel", ""), "sup": data.get("support", 1)})
    links.sort(key=lambda e: (e["s"], e["t"]))

    profiles = [{k: v for k, v in p.items() if k != "members"} for p in cg.profiles]

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {"stats": s, "nodes": nodes, "links": links, "communities": profiles}
    (outdir / "graph.json").write_text(json.dumps(payload), encoding="utf-8")
    (outdir / "communities.json").write_text(json.dumps(cg.profiles, indent=1), encoding="utf-8")
    print(f"\nwrote {outdir/'graph.json'} "
          f"({(outdir/'graph.json').stat().st_size/1024:.0f} KB) and communities.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
