"""The corpus knowledge graph: elements as nodes, measured relatedness as edges.

WHAT THIS IS, AND WHAT IT DELIBERATELY IS NOT
---------------------------------------------
It is a graph over the 750 platform knowledge elements, partitioned into communities. It is
**not** built from the extraction pipeline's provenance edges, and that is a measured decision
rather than an oversight.

Extraction emits 4,212 edges over the corpus, of which 4,071 (96.7%) satisfy
``dst.startswith(src + "::")`` — a pure function of the id, already recoverable by
``doc_ids.parent_doc_id`` — and 0 cross an element boundary. Adding every extraction-native
layer on top of the layers below moved the graph by **141 element pairs on a 15,711-pair
baseline (0.90%)**, rescued **0** isolated elements, changed Louvain modularity by ≤0.0025, and
12 of 15 randomly sampled new pairs read as spurious. The citation edges from
``extractors/analysis/citations.py`` are correct and high-precision and add **zero** pairs the
layers below do not already connect.

So the layers here are the ones that were measured to carry relatedness:

``RELATED_TO``      the platform's own curated links. 592 undirected pairs, 100% reciprocal and
                    untyped. Survives hub removal (top-10 hubs removed keeps 78% of pairs) and
                    has clustering coefficient 0.47 — a real graph, not a star. Note 0 of 160
                    ``map`` elements appear in it at all.
``IN_COLLECTION``   5 collections over 68 elements. Degree-normalised, because the largest
                    collection alone would otherwise contribute 61% of collection pairs as one
                    clique.
``SIMILAR_TO``      embedding kNN over title+tags+contents. The load-bearing layer, and the
                    honest caveat: it is why the graph has no isolated nodes. Strip it and 103
                    of 750 are isolated; strip to curated-only and 312 of 750 are.
``CITES``/``USES``  notebook→element references parsed from notebook text. Kept because they are
                    free, exact, and human-authored, not because they add reach.

Validation that justifies the whole thing: held-out curated pairs land in the same community
**82.9%** of the time against 11.0% for random pairs and 11.4% for degree-matched controls
(resolution 1.0, 10 folds) — a 7.6x lift. kNN alone, which never sees curated data, reaches
74.2%, and a graph with every curated edge removed still recovers 78.4% of them.

REPRODUCIBILITY
---------------
``seed=`` alone does NOT make Louvain reproducible; it pins the algorithm's randomness but not
the order it visits nodes, which follows insertion order. Measured: 9 insertion orders produce 9
distinct partitions at every size from 60 nodes to 750 nodes / 9,112 edges. Every graph here is
therefore built through :func:`build_graph`, which sorts nodes and edges before inserting. See
``test_deployment_contract.py::test_a_seed_alone_does_NOT_make_the_partition_reproducible``.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "Element", "Edge", "LAYER_WEIGHTS", "load_elements", "curated_edges", "collection_edges",
    "knn_edges", "citation_edges", "fuse", "build_graph", "partition", "community_profiles",
    "CorpusGraph", "build_corpus_graph",
]

# Base confidence that a layer asserts *relatedness*. These are not tuned — the validation
# above was re-run with uniform weight=1.0 and still gave 79.5% vs 12.7% (6.2x), so the result
# does not depend on them. They are here to express ordering, not to be a fitted model.
LAYER_WEIGHTS: Dict[str, float] = {
    "RELATED_TO": 1.00,      # a human asserted it
    "CITES": 1.00,           # a human wrote a URL naming this exact element
    "USES": 1.00,            # ...and downloaded its data
    "IN_COLLECTION": 0.50,   # curated grouping, but degree-normalised below
    "SIMILAR_TO": 0.50,      # semantic proximity; real, but not an assertion by anyone
}


@dataclass(frozen=True)
class Element:
    id: str
    title: str
    resource_type: str
    tags: Tuple[str, ...] = ()
    contributor: str = ""
    authors: Tuple[str, ...] = ()
    contents: str = ""
    click_count: int = 0

    @property
    def text(self) -> str:
        """The field used for embedding. Title and tags are repeated deliberately: `contents`
        has a median length of 353 chars and a max of 4,938, so without repetition a long
        description drowns the title in a mean-pooled vector."""
        tags = " ".join(self.tags)
        return f"{self.title}. {tags}. {self.title}. {self.contents}".strip()


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    rel: str
    weight: float = 1.0
    evidence: str = ""

    def key(self) -> Tuple[str, str]:
        """Undirected identity. The curated relation is 100% reciprocal, so treating edges as
        directed would double every pair and silently double its weight in the partition."""
        return (self.src, self.dst) if self.src <= self.dst else (self.dst, self.src)


# --------------------------------------------------------------------------- loading

def load_elements(details_path: str | Path, *, require_public: bool = True) -> List[Element]:
    """Read platform element records from a JSONL or JSON dump.

    ``require_public`` drops anything not marked public. It defaults to on and is not merely a
    filter: private elements must never enter a graph that the agent reads from, and the check
    belongs at the boundary where records enter rather than at the point of use.
    """
    p = Path(details_path)
    raw = p.read_text(encoding="utf-8")
    if p.suffix == ".jsonl":
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        loaded = json.loads(raw)
        records = loaded.get("elements", loaded) if isinstance(loaded, dict) else loaded

    out: List[Element] = []
    dropped_private = 0
    seen: set = set()
    for r in records:
        eid = str(r.get("id") or "").strip()
        if not eid or eid in seen:
            continue
        if require_public and str(r.get("visibility") or "").lower() != "public":
            dropped_private += 1
            continue
        seen.add(eid)
        contributor = r.get("contributor") or {}
        if isinstance(contributor, dict):
            contributor = contributor.get("name") or contributor.get("id") or ""
        out.append(Element(
            id=eid,
            title=str(r.get("title") or "").strip(),
            resource_type=str(r.get("resource-type") or "unknown").strip().lower(),
            tags=tuple(sorted({str(t).strip() for t in (r.get("tags") or []) if str(t).strip()})),
            contributor=str(contributor or "").strip(),
            authors=tuple(str(a).strip() for a in (r.get("authors") or []) if str(a).strip()),
            contents=str(r.get("contents") or "").strip(),
            click_count=int(r.get("click-count") or 0),
        ))
    if dropped_private:
        logger.info("corpus_graph: excluded %d non-public element(s)", dropped_private)
    return sorted(out, key=lambda e: e.id)


# --------------------------------------------------------------------------- edge layers

def curated_edges(details_path: str | Path, known: Iterable[str]) -> List[Edge]:
    """``related-elements`` from the platform's detail endpoint — the listing endpoint omits it.

    Self-links and targets outside the element universe are dropped rather than materialised as
    phantom nodes.
    """
    known = set(known)
    p = Path(details_path)
    lines = p.read_text(encoding="utf-8").splitlines()
    records = ([json.loads(x) for x in lines if x.strip()] if p.suffix == ".jsonl"
               else json.loads(p.read_text(encoding="utf-8")))
    out: Dict[Tuple[str, str], Edge] = {}
    for r in records:
        src = str(r.get("id") or "")
        if src not in known:
            continue
        for entry in (r.get("related-elements") or []):
            dst = str((entry or {}).get("id") or "") if isinstance(entry, dict) else str(entry)
            if not dst or dst == src or dst not in known:
                continue
            e = Edge(src, dst, "RELATED_TO", LAYER_WEIGHTS["RELATED_TO"],
                     "platform related-elements")
            out.setdefault(e.key(), e)
    return sorted(out.values(), key=lambda e: e.key())


def collection_edges(details_path: str | Path, known: Iterable[str]) -> List[Edge]:
    """Co-membership of a curated collection, **degree-normalised**.

    A collection of n members is a clique of n(n-1)/2 pairs. Measured: the largest collection
    (29 members) alone is 406 of 667 collection pairs — 61%. Left raw it would dominate the
    partition with one curator's grouping, so each pair carries 1/(n-1) of the layer weight and
    a member's total collection weight stays ~constant regardless of clique size.
    """
    known = set(known)
    p = Path(details_path)
    lines = p.read_text(encoding="utf-8").splitlines()
    records = ([json.loads(x) for x in lines if x.strip()] if p.suffix == ".jsonl"
               else json.loads(p.read_text(encoding="utf-8")))
    members: Dict[str, List[str]] = defaultdict(list)
    for r in records:
        eid = str(r.get("id") or "")
        if eid not in known:
            continue
        for c in (r.get("collections") or []):
            name = str((c or {}).get("id") or (c or {}).get("title") or c) if isinstance(c, dict) \
                else str(c)
            if name:
                members[name].append(eid)
    out: List[Edge] = []
    for name, ids in sorted(members.items()):
        ids = sorted(set(ids))
        if len(ids) < 2:
            continue
        w = LAYER_WEIGHTS["IN_COLLECTION"] * 2.0 / (len(ids) - 1)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                out.append(Edge(ids[i], ids[j], "IN_COLLECTION", w,
                                f"collection {name} ({len(ids)} members)"))
    return out


def knn_edges(elements: Sequence[Element], *, k: int = 10,
              model_name: str = "all-MiniLM-L6-v2",
              vectors=None) -> List[Edge]:
    """Semantic k-nearest-neighbour edges over element text.

    Vectors are L2-normalised before the dot product. The vectors already indexed in OpenSearch
    are NOT normalised (``dense_embedding_server.py:28`` mean-pools ``last_hidden_state`` with no
    attention-mask weighting), which makes their norms track text length — so neighbours drawn
    from that field partly rank by document length. Normalising here makes the similarity an
    actual cosine.

    A mutual neighbour (each in the other's top-k) is weighted full; a one-way neighbour is
    discounted, because "A is in B's top 10" is a much weaker claim when B is not in A's.
    """
    if vectors is None:
        from sentence_transformers import SentenceTransformer  # imported lazily: heavy
        model = SentenceTransformer(model_name)
        vectors = model.encode([e.text for e in elements], batch_size=32,
                               show_progress_bar=False, convert_to_numpy=True)
    import numpy as np

    v = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    v = v / norms
    sim = v @ v.T
    np.fill_diagonal(sim, -1.0)

    k = max(1, min(k, len(elements) - 1))
    topk = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    neighbours = {i: set(topk[i].tolist()) for i in range(len(elements))}

    out: Dict[Tuple[str, str], Edge] = {}
    for i, nbrs in neighbours.items():
        for j in nbrs:
            mutual = i in neighbours[j]
            cos = float(sim[i, j])
            w = LAYER_WEIGHTS["SIMILAR_TO"] * max(0.0, cos) * (1.0 if mutual else 0.6)
            e = Edge(elements[i].id, elements[j].id, "SIMILAR_TO", w,
                     f"cosine {cos:.3f}{' (mutual)' if mutual else ''}")
            prev = out.get(e.key())
            if prev is None or e.weight > prev.weight:
                out[e.key()] = e
    return sorted(out.values(), key=lambda e: e.key())


def citation_edges(corpus_dir: str | Path, id_by_prefix: Dict[str, str],
                   known: Iterable[str]) -> List[Edge]:
    """``CITES``/``USES`` parsed from notebook text by ``extractors.analysis.citations``.

    Measured to add zero pairs the other layers do not already connect. Kept anyway: they are
    the only edges here a human wrote *as a link to a specific element*, so they are the right
    evidence to surface when explaining why two elements are connected, even when they are not
    what connects them.
    """
    from extractors.analysis.citations import platform_citations

    known = set(known)
    out: Dict[Tuple[str, str], Edge] = {}
    for path in sorted(Path(corpus_dir).glob("*.ipynb")):
        src = id_by_prefix.get(path.name.split("__")[0])
        if not src or src not in known:
            continue
        try:
            nb = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        for cell in (nb.get("cells") or []):
            s = cell.get("source")
            text = "".join(s) if isinstance(s, list) else str(s or "")
            for c in platform_citations(text):
                if c.element_id == src or c.element_id not in known:
                    continue
                e = Edge(src, c.element_id, c.rel, LAYER_WEIGHTS.get(c.rel, 1.0),
                         f"platform URL in notebook ({c.host})")
                out.setdefault(e.key(), e)
    return sorted(out.values(), key=lambda e: e.key())


# --------------------------------------------------------------------------- fuse + partition

def fuse(*layers: Sequence[Edge]) -> List[Edge]:
    """Combine layers, summing weight per undirected pair and recording every supporting layer.

    Support count matters more than the summed weight when reading a result: measured on the
    750-element graph, **81.7% of edges rest on a single layer** and only 207 have three or more
    agreeing. A fused graph is a union, not a consensus, and the ``rel`` string keeps that
    visible instead of averaging it away.
    """
    merged: Dict[Tuple[str, str], Dict] = {}
    for layer in layers:
        for e in layer:
            slot = merged.setdefault(e.key(), {"w": 0.0, "rels": set(), "ev": []})
            slot["w"] += e.weight
            slot["rels"].add(e.rel)
            if e.evidence:
                slot["ev"].append(f"{e.rel}: {e.evidence}")
    out = []
    for (a, b), slot in merged.items():
        out.append(Edge(a, b, "+".join(sorted(slot["rels"])), round(slot["w"], 6),
                        " | ".join(slot["ev"][:4])))
    return sorted(out, key=lambda e: e.key())


def build_graph(elements: Sequence[Element], edges: Sequence[Edge]):
    """Build the graph with CANONICAL insertion order.

    This is the whole reason the function exists. Louvain visits nodes in insertion order, and
    ``seed=`` does not pin that. Measured: 9 insertion orders of one identical graph produce 9
    distinct partitions, at every size tested up to 750 nodes / 9,112 edges. Sorting here is
    what makes two runs over an unchanged corpus return the same community ids — without it,
    cached community summaries and any ``community_id`` written onto an element document drift
    for no reason and it looks like the corpus changed.
    """
    import networkx as nx

    g = nx.Graph()
    for e in sorted(elements, key=lambda x: x.id):
        g.add_node(e.id, title=e.title, resource_type=e.resource_type, tags=list(e.tags),
                   contributor=e.contributor, click_count=e.click_count)
    for e in sorted(edges, key=lambda x: (x.key(), x.rel)):
        a, b = e.key()
        g.add_edge(a, b, weight=float(e.weight), rel=e.rel, evidence=e.evidence,
                   support=len(e.rel.split("+")))
    return g


def partition(graph, *, resolution: float = 1.0, seed: int = 17) -> Dict[str, int]:
    """Louvain communities as ``{element_id: community_index}``, ids assigned deterministically.

    Community indices are assigned by descending size then by the lexicographically smallest
    member, so the same corpus yields the same ids across runs — a raw enumeration index would
    reshuffle whenever two communities tie on size.
    """
    from networkx.algorithms import community as nx_community

    parts = nx_community.louvain_communities(graph, weight="weight", resolution=resolution,
                                             seed=seed)
    ordered = sorted(parts, key=lambda p: (-len(p), min(p)))
    return {node: idx for idx, part in enumerate(ordered) for node in sorted(part)}


def community_profiles(graph, membership: Dict[str, int], *,
                       top_tags: int = 6) -> List[Dict]:
    """Per-community descriptive profile: size, type mix, distinctive tags, central members.

    Tags are ranked by a TF-IDF-style lift (share inside the community over share in the corpus)
    rather than raw frequency. Measured reason: ``flood risk`` is on 162 of 750 elements and
    ``flood map`` on 158, from a single contributor, so a raw-frequency label reports the
    corpus's largest upload batch as the theme of any community that touches it.
    """
    members: Dict[int, List[str]] = defaultdict(list)
    for node, cid in membership.items():
        members[cid].append(node)

    corpus_tag_share: Counter = Counter()
    for _, data in graph.nodes(data=True):
        for t in set(data.get("tags") or []):
            corpus_tag_share[t.lower()] += 1
    n_total = max(1, graph.number_of_nodes())

    profiles: List[Dict] = []
    for cid in sorted(members):
        ids = sorted(members[cid])
        types: Counter = Counter()
        tags: Counter = Counter()
        contributors: Counter = Counter()
        for nid in ids:
            d = graph.nodes[nid]
            types[d.get("resource_type", "unknown")] += 1
            contributors[d.get("contributor") or "unknown"] += 1
            for t in set(d.get("tags") or []):
                tags[t.lower()] += 1

        lift = []
        for t, c in tags.items():
            inside = c / len(ids)
            overall = corpus_tag_share[t] / n_total
            if c >= 2 and overall > 0:
                lift.append((inside * math.log(1.0 + inside / overall), t, c))
        lift.sort(reverse=True)

        sub = graph.subgraph(ids)
        degree = sorted(sub.degree(weight="weight"), key=lambda kv: (-kv[1], kv[0]))
        top_contrib, top_contrib_n = (contributors.most_common(1) or [("", 0)])[0]

        profiles.append({
            "community": cid,
            "size": len(ids),
            "types": dict(types.most_common()),
            "dominant_type": types.most_common(1)[0][0] if types else "unknown",
            "type_purity": round(types.most_common(1)[0][1] / len(ids), 3) if types else 0.0,
            "distinctive_tags": [{"tag": t, "count": c} for _, t, c in lift[:top_tags]],
            "top_contributor": top_contrib,
            # A community that is one lab's upload is a provenance artifact, not a research
            # theme. Corpus-wide contributor NMI against the partition measured 0.58, so this
            # is reported per community rather than assumed away.
            "contributor_concentration": round(top_contrib_n / len(ids), 3),
            "central_members": [
                {"id": nid, "title": graph.nodes[nid].get("title", ""),
                 "resource_type": graph.nodes[nid].get("resource_type", ""),
                 "strength": round(float(w), 3)}
                for nid, w in degree[:5]
            ],
            "internal_edges": sub.number_of_edges(),
            "members": ids,
        })
    return profiles


@dataclass
class CorpusGraph:
    graph: object
    elements: List[Element]
    edges: List[Edge]
    membership: Dict[str, int]
    profiles: List[Dict] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)


def build_corpus_graph(details_path: str | Path, *, corpus_dir: Optional[str | Path] = None,
                       id_by_prefix: Optional[Dict[str, str]] = None,
                       k: int = 10, resolution: float = 1.0, seed: int = 17,
                       vectors=None) -> CorpusGraph:
    """End-to-end build. Every layer is optional except the elements themselves."""
    import networkx as nx

    elements = load_elements(details_path)
    ids = [e.id for e in elements]

    curated = curated_edges(details_path, ids)
    collections = collection_edges(details_path, ids)
    knn = knn_edges(elements, k=k, vectors=vectors)
    cites = (citation_edges(corpus_dir, id_by_prefix or {}, ids)
             if corpus_dir and id_by_prefix else [])

    edges = fuse(curated, collections, knn, cites)
    graph = build_graph(elements, edges)
    membership = partition(graph, resolution=resolution, seed=seed)
    profiles = community_profiles(graph, membership)

    layer_pairs = {name: len({e.key() for e in layer}) for name, layer in
                   (("RELATED_TO", curated), ("IN_COLLECTION", collections),
                    ("SIMILAR_TO", knn), ("CITES/USES", cites))}
    isolated = [n for n, d in graph.degree() if d == 0]
    stats = {
        "elements": len(elements),
        "edges": graph.number_of_edges(),
        "layer_pairs": layer_pairs,
        "components": nx.number_connected_components(graph),
        "isolated": len(isolated),
        # Stated because it is the honest caveat: kNN gives every node k neighbours by
        # construction, so "0 isolated" describes the layer, not the corpus. Without kNN,
        # 103 of 750 are isolated; with only human-curated evidence, 312 of 750 are.
        "isolated_without_knn": len(
            [n for n, d in build_graph(elements, fuse(curated, collections, cites)).degree()
             if d == 0]),
        "communities": len(profiles),
        "modularity": None,
        "single_layer_edge_fraction": round(
            sum(1 for _, _, d in graph.edges(data=True) if d.get("support", 1) == 1)
            / max(1, graph.number_of_edges()), 4),
        "resolution": resolution,
        "seed": seed,
        "k": k,
    }
    try:
        from networkx.algorithms import community as nx_community

        groups = defaultdict(set)
        for node, cid in membership.items():
            groups[cid].add(node)
        stats["modularity"] = round(
            nx_community.modularity(graph, [groups[c] for c in sorted(groups)],
                                    weight="weight"), 4)
    except Exception as exc:  # pragma: no cover - diagnostic only
        logger.warning("modularity unavailable: %s", exc)

    return CorpusGraph(graph=graph, elements=elements, edges=edges, membership=membership,
                       profiles=profiles, stats=stats)
