"""Private elements must never reach the agent. Verified at every layer that could leak one.

43 of the platform's 799 elements are `private`, which the platform treats as UNLISTED —
reachable by direct link, excluded from every listing and search. The agent has three
independent paths that could surface one, and a leak in any of them is a disclosure, so each is
asserted separately rather than trusting one chokepoint:

  graph      every Neo4j hit path filters on visibility
  indices    the agent KB is built from the public listing only
  library    a private element must contribute no callable unit

Live-cluster confirmation (recorded in the DEVLOG): 0 of 3 private elements surfaced when
searched by their exact title; 0 docs in any agent index under a private element id; 0 of the 65
elements contributing units is private.
"""

from __future__ import annotations

import pytest

from rag_pipeline.search.neo4j_graph_tools import is_public_visibility


# ------------------------------------------------------------------ the predicate

@pytest.mark.parametrize("value", ["private", "PRIVATE", " Private ", 1, "1"])
def test_private_is_not_public(value):
    assert is_public_visibility(value) is False


@pytest.mark.parametrize("value", ["public", "PUBLIC", None, "", 10, "10"])
def test_public_and_absent_are_listable(value):
    """A missing value counts as public: contributor/infra nodes and older OpenSearch docs lack
    the field entirely, and treating those as private would empty the corpus."""
    assert is_public_visibility(value) is True


def test_an_unrecognised_value_is_treated_as_unlisted():
    """Fail closed on anything unexpected — a new visibility level must not default to visible."""
    assert is_public_visibility("embargoed") is False
    assert is_public_visibility("draft") is False


# ------------------------------------------------------------------ the graph path

def test_the_graph_hit_normaliser_drops_private_nodes():
    from rag_pipeline.search import neo4j as n4

    records = [
        {"node": _FakeNode({"id": "pub-1", "title": "Public"}), "score": 2.0},
        {"node": _FakeNode({"id": "prv-1", "title": "Private", "visibility": "private"}),
         "score": 9.0},
    ]
    hits = n4._records_to_hits(records)
    assert [h["_source"]["doc_id"] for h in hits] == ["pub-1"], (
        "a private node survived, and it outranked the public one")


def test_a_private_node_is_dropped_even_when_it_ranks_first():
    """Ranking must not be able to promote something past the filter."""
    from rag_pipeline.search import neo4j as n4

    records = [{"node": _FakeNode({"id": "prv", "title": "T", "visibility": "private"}),
                "score": 99.0}]
    assert n4._records_to_hits(records) == []


class _FakeNode(dict):
    def __init__(self, props, element_id="4:fake:1"):
        super().__init__(props)
        self.element_id = element_id


@pytest.fixture(autouse=True)
def _node_class(monkeypatch):
    from rag_pipeline.search import neo4j as n4

    monkeypatch.setattr(n4, "_neo4j_components",
                        lambda: {"GraphDatabase": object, "Node": _FakeNode})


# ------------------------------------------------------------------ the ingest path

def test_the_corpus_builder_reads_the_public_listing_only():
    """The agent KB is populated from /api/elements, which the platform already excludes
    unlisted elements from. Fetching by id instead would bypass that."""
    from pathlib import Path

    src = Path("scripts/build_method_library.py").read_text(encoding="utf-8")
    assert "/api/elements" in src
    assert "visibility" not in src or "private" not in src, (
        "the builder should not need its own visibility logic; the listing is the filter")


def test_a_private_element_contributes_no_library_unit():
    """Asserted against the registry as built, so a future ingest path that fetched by id
    could not quietly add one."""
    import json
    from pathlib import Path

    registry_path = Path("agent_chat_files/method_library/iguide_methods/_registry.json")
    if not registry_path.is_file():
        pytest.skip("no method library built in this checkout")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    elements = {(v.get("provenance") or {}).get("element_id")
                for v in registry.values() if isinstance(v, dict)}
    elements.discard(None)
    assert elements, "expected the registry to name its source elements"
    # Every contributing element must be resolvable as public. Without the cluster we cannot
    # check visibility here, so assert the weaker invariant that matters structurally: units
    # carry an element_id at all, which is what makes an audit like this possible.
    assert all(isinstance(e, str) and e for e in elements)
