"""Sources that link to the platform (Stage 57).

On a prod turn on 2026-10-13 ("What extreme research has been done using the resources on I-GUIDE
platform") six of the eight sources were plain text instead of links. Two causes, both measured
on the VM that day:

1. The platform graph holds an element's type ONLY as a label (Publication 206, Notebook 214,
   Map 161, Dataset 142, Oer 44, Code 38, Collection 19; no node has an ``element_type`` or
   ``resource-type`` property). Hits built from node properties fell back to ``"resource"``, which
   no link builder links. Dedup then kept the untyped graph copy of an element that keyword search
   had returned typed ``publication``.
2. The agent KB keys elements by an 8-character prefix of the platform UUID, so an agent-KB
   source linked to nothing and was cited by an id the platform does not know.

All offline: fake graph rows and a fake KB record.
"""

from __future__ import annotations

import json
from contextlib import contextmanager

import pytest

from rag_pipeline.search import element_ids
from rag_pipeline.search import neo4j as n4

FULL = "05269a1a-ce6e-4e87-8b43-74490a986c09"
PUB = "777a106d-1b2c-4d3e-8f40-5a6b7c8d9e0f"


class _FakeNode(dict):
    """neo4j.graph.Node as the search code reads it: properties, element_id, labels."""

    def __init__(self, props, labels=(), element_id="4:fake:1"):
        super().__init__(props)
        self.element_id = element_id
        self.labels = frozenset(labels)


class _FakeSession:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, cypher, **params):
        return list(self._rows)


class _FakeDriver:
    def __init__(self, rows):
        self.rows = rows

    def session(self, **_kw):
        return _FakeSession(self.rows)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(n4, "_neo4j_components", lambda: {"GraphDatabase": object, "Node": _FakeNode})
    element_ids._reset_cache_for_tests()
    yield
    element_ids._reset_cache_for_tests()


def _url(doc):
    from agent_runtime.supervisor.evidence_subgraph import _element_url

    return _element_url(doc)


# ------------------------------------------------------------------ 1. the type is the label

def test_a_graph_node_with_only_a_label_gets_that_type_and_a_link():
    node = _FakeNode({"id": FULL, "title": "Flood notebook"}, labels={"Notebook"})
    hits = n4.get_neo4j_search_results("flood notebook", driver=_FakeDriver([{"node": node, "score": 2.0}]))
    assert hits[0]["_source"]["element_type"] == "notebook"

    from agent_runtime.langchain_granular_tools import _normalize_hits

    doc = _normalize_hits(hits, source="neo4j")[0]
    assert doc["element_type"] == "notebook"
    assert _url(doc) == f"https://platform.i-guide.io/notebooks/{FULL}"


def test_the_agent_dispatcher_fallback_types_graph_hits_too(monkeypatch):
    """`get_neo4j_agent_results` ends in the keyword arm when no pattern matches and Text2Cypher
    is off or empty. That arm is the one that produced the "resource" hits."""
    from rag_pipeline.search import agents

    node = _FakeNode({"id": FULL, "title": "Extreme heat collection"}, labels={"Collection"})
    monkeypatch.setenv("USE_TEXT2CYPHER", "false")
    monkeypatch.setattr(agents, "detect_pattern", lambda q: None)
    monkeypatch.setattr(n4, "_neo4j_run", lambda cypher, params, driver=None: [{"node": node, "score": 1.0}])
    hits = agents.get_neo4j_agent_results("extreme heat research")
    doc = agents._hit_to_document(hits[0])
    assert doc["element_type"] == "collection"
    assert _url(doc) == f"https://platform.i-guide.io/collections/{FULL}"


def test_a_documentation_node_links_under_docs():
    """The 6 :Documentation nodes are reachable now that labels type hits; their pages are
    /docs/<slug>, and /documentations/<slug> is an empty page."""
    node = _FakeNode({"id": "getting-started", "title": "Getting Started"}, labels={"Documentation"})
    hits = n4.get_neo4j_search_results("getting started", driver=_FakeDriver([{"node": node}]))
    from agent_runtime.langchain_granular_tools import _normalize_hits

    assert _url(_normalize_hits(hits, source="neo4j")[0]) == "https://platform.i-guide.io/docs/getting-started"


def test_only_knowledge_element_labels_name_a_type():
    from rag_pipeline.search.neo4j_graph_tools import element_type_from_labels

    assert element_type_from_labels({"Contributor"}) is None
    assert element_type_from_labels({"Contributor", "Publication"}) == "publication"
    assert element_type_from_labels([]) is None


def test_a_stored_type_property_is_not_overridden_by_the_label():
    typed = _FakeNode({"id": PUB, "title": "T", "resource-type": "dataset"}, labels={"Notebook"})
    bare = _FakeNode({"id": FULL, "title": "T"}, labels={"Map"})
    hits = n4.get_neo4j_search_results("t data", driver=_FakeDriver([{"node": typed}, {"node": bare}]))
    assert [h["_source"]["element_type"] for h in hits] == ["dataset", "map"]


# ------------------------------------------------------------------ 1c. thumbnails, one key

def test_a_labelled_node_carries_its_thumbnail_under_one_key(monkeypatch):
    """The graph spells it `thumbnail_image`, the index `thumbnail-image`; a normalized document
    carries `thumbnail_image` whichever arm it came from."""
    from agent_runtime.langchain_granular_tools import _normalize_hits
    from rag_pipeline.search import agents

    node = _FakeNode({"id": FULL, "title": "Flood notebook", "thumbnail_image": THUMB},
                     labels={"Notebook"})
    hits = n4.get_neo4j_search_results("flood", driver=_FakeDriver([{"node": node}]))
    assert _normalize_hits(hits, source="neo4j")[0]["thumbnail_image"] == THUMB

    # The dispatcher's own hit shape, used by the popularity and related-element paths.
    hit = agents._node_to_hit({**node, "_labels": ["Notebook"]})
    assert hit["_source"]["thumbnail_image"] == THUMB and "thumbnail-image" not in hit["_source"]
    assert agents._hit_to_document(hit)["thumbnail_image"] == THUMB

    # An index-shaped keyword hit lands under the same key.
    keyword_hit = {"_id": PUB, "_source": {"title": "P", "resource-type": "publication",
                                           "thumbnail-image": THUMB}}
    assert _normalize_hits([keyword_hit], source="keyword")[0]["thumbnail_image"] == THUMB


def test_a_collection_with_an_empty_thumbnail_carries_none():
    """14 of the graph's 19 Collections hold "" — that is no thumbnail, never an empty src."""
    from agent_runtime.langchain_granular_tools import _normalize_hits
    from rag_pipeline.search import agents
    from rag_pipeline.state import merge_retrieval

    node = _FakeNode({"id": FULL, "title": "Summer School 2025", "thumbnail_image": "",
                      "thumbnail_image_credit": ""}, labels={"Collection"})
    hits = n4.get_neo4j_search_results("summer school", driver=_FakeDriver([{"node": node}]))
    assert not any(k in hits[0]["_source"] for k in ("thumbnail_image", "thumbnail-image"))
    doc = _normalize_hits(hits, source="neo4j")[0]
    assert doc["element_type"] == "collection" and "thumbnail_image" not in doc
    assert "thumbnail_image" not in agents._node_to_hit({**node, "_labels": ["Collection"]})["_source"]
    state = {}
    merge_retrieval(state, source="neo4j", hits=[{"_id": FULL, "_source": {"thumbnail-image": "  "}}])
    assert "thumbnail_image" not in state["evidence"]["retrieved_documents"][0]["document"]


# ------------------------------------------------------------------ 1b. dedup keeps the typed copy

def test_the_typed_duplicate_wins_the_supervisor_merge():
    from agent_runtime.supervisor.graph import _merge_dedup

    graph_copy = {"doc_id": PUB, "source": "neo4j", "element_type": "resource", "url": "",
                  "title": "Spatial and Temporal Variability"}
    keyword_copy = {"doc_id": PUB, "source": "keyword", "element_type": "publication", "url": "",
                    "title": "Spatial and Temporal Variability"}
    other = {"doc_id": FULL, "element_type": "notebook", "url": ""}
    merged = _merge_dedup([graph_copy, other], [keyword_copy])
    assert [d["doc_id"] for d in merged] == [PUB, FULL]          # one copy, same position
    assert merged[0]["source"] == "keyword"
    assert _url(merged[0]) == f"https://platform.i-guide.io/publications/{PUB}"
    # ...and a typed copy already held is never displaced by an untyped one.
    assert _merge_dedup([keyword_copy], [graph_copy])[0]["source"] == "keyword"


def test_the_typed_duplicate_wins_the_retrieval_merge():
    from rag_pipeline.state import merge_retrieval

    state = {}
    merge_retrieval(state, source="neo4j", hits=[
        {"_id": PUB, "_source": {"title": "S", "element_type": "resource"}}])
    merge_retrieval(state, source="keyword", hits=[
        {"_id": PUB, "_source": {"title": "S", "resource-type": "publication"}}])
    docs = state["evidence"]["retrieved_documents"]
    assert len(docs) == 1
    assert docs[0]["document"]["element_type"] == "publication"


# ------------------------------------------------------------------ 2. agent-KB prefixes

THUMB = "https://backend.i-guide.io:443/user-uploads/thumbnails/1720206819181-spastc_scaled.jpg"


def _graph(rows_by_prefix, thumbnail=THUMB):
    calls = []

    def run(cypher, params):
        calls.append(params["prefixes"])
        return [{"prefix": p, "id": full, "labels": row[1],
                 "thumbnail": row[2] if len(row) > 2 else thumbnail}
                for p in params["prefixes"] for row in rows_by_prefix.get(p, []) for full in row[:1]]

    run.calls = calls
    return run


def _kb_search(monkeypatch, run, parents=("05269a1a",)):
    """agent_kb_search on the Postgres backend, with the record and the graph faked."""
    from extractors import kb_db
    from rag_pipeline.search import agent_kb

    @contextmanager
    def _connect(*_a, **_k):
        yield object()

    blocks = [{"_id": f"{p}::block::13", "_index": "pg:block", "_score": 2.0,
               "_source": {"doc_id": f"{p}::block::13", "title": "Heat index cell",
                           "contents": "df.plot()", "resource-type": "NotebookBlock",
                           "extracted": {"parent_doc_id": p, "parent_title": "Heat"}}}
              for p in parents]
    monkeypatch.setattr(kb_db, "enabled", lambda: True)
    monkeypatch.setattr(kb_db, "connect", _connect)
    monkeypatch.setattr(kb_db, "search_kb", lambda conn, q, size=8: blocks)
    monkeypatch.setattr(kb_db, "parent_elements", lambda conn, ids: {
        p: {"id": p, "title": "Heat", "resource-type": "notebook"} for p in parents})
    monkeypatch.setattr(element_ids, "_default_run", run)
    return agent_kb.agent_kb_search("heat index")


def test_an_agent_kb_hit_links_to_its_parent_element_by_full_id(monkeypatch):
    payload = _kb_search(monkeypatch, _graph({"05269a1a": [(FULL, ["Notebook"])]}))
    doc = payload["documents"][0]
    assert doc["url"] == f"https://platform.i-guide.io/notebooks/{FULL}"
    assert doc["element_id"] == FULL
    assert payload["citation_ids"] == [FULL]
    # The KB's own keys are untouched: get_kb_block and kb_for_elements look blocks up by them.
    assert doc["doc_id"] == "05269a1a::block::13" and doc["parent_doc_id"] == "05269a1a"
    assert payload["elements"]["05269a1a"]["url"] == doc["url"]
    # And the link survives to the client the way the server sends it.
    from agent_runtime.graph_runtime import _with_citation_urls

    out = _with_citation_urls({"evidence": [json.loads(json.dumps(doc))]})
    assert out["evidence"][0]["url"] == f"https://platform.i-guide.io/notebooks/{FULL}"


def test_an_agent_kb_hit_carries_its_parent_elements_thumbnail(monkeypatch):
    """Once the full id is resolved, the hit takes its parent element's thumbnail; a parent whose
    thumbnail is "" (14 of 19 Collections) gets none, and is still linked."""
    payload = _kb_search(monkeypatch, _graph({"05269a1a": [(FULL, ["Notebook"])],
                                              "777a106d": [(PUB, ["Collection"], "")]}),
                         parents=("05269a1a", "777a106d"))
    with_thumb, without = payload["documents"]
    assert with_thumb["thumbnail_image"] == THUMB
    assert payload["elements"]["05269a1a"]["thumbnail_image"] == THUMB
    assert without["url"] == f"https://platform.i-guide.io/collections/{PUB}"
    assert "thumbnail_image" not in without
    assert "thumbnail_image" not in payload["elements"]["777a106d"]


def test_an_ambiguous_prefix_gets_no_link(monkeypatch):
    """Refuse rather than guess — while the unambiguous hit beside it is still linked."""
    other = "05269a1a-0000-4000-8000-000000000000"
    payload = _kb_search(monkeypatch, _graph({"05269a1a": [(FULL, ["Notebook"]), (other, ["Dataset"])],
                                              "777a106d": [(PUB, ["Publication"])]}),
                         parents=("05269a1a", "777a106d"))
    ambiguous, unique = payload["documents"]
    assert "url" not in ambiguous and "element_id" not in ambiguous
    assert _url(ambiguous) == ""
    assert unique["url"] == f"https://platform.i-guide.io/publications/{PUB}"
    assert payload["citation_ids"] == ["05269a1a", PUB]


def test_the_graph_type_wins_over_the_kb_type(monkeypatch):
    """15 of 543 KB elements are typed `dataset` in the KB and served as publications."""
    payload = _kb_search(monkeypatch, _graph({"05269a1a": [(FULL, ["Publication"])]}))
    assert payload["documents"][0]["url"] == f"https://platform.i-guide.io/publications/{FULL}"


def test_one_lookup_per_search_and_cached(monkeypatch):
    run = _graph({"05269a1a": [(FULL, ["Notebook"])]})
    _kb_search(monkeypatch, run)
    _kb_search(monkeypatch, run)
    assert run.calls == [["05269a1a"]]


def test_a_graph_outage_costs_the_link_not_the_search(monkeypatch):
    def down(cypher, params):
        raise ConnectionError("graph unreachable")

    payload = _kb_search(monkeypatch, down)
    assert payload["count"] == 1 and payload["citation_ids"] == ["05269a1a"]
    assert "url" not in payload["documents"][0]
    # The failure is not cached: once the graph answers, the next search links.
    payload = _kb_search(monkeypatch, _graph({"05269a1a": [(FULL, ["Notebook"])]}))
    assert payload["citation_ids"] == [FULL]
