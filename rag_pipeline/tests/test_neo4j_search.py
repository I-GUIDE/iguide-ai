"""The Neo4j keyword arm: term matching, id identity, and label scoping.

Three defects shipped together here, and the outage masked all of them — the arm looked
broken because the host was unreachable, and stayed broken once it was reachable:

1. The Cypher bound ``$q`` to the WHOLE query and tested ``CONTAINS``, a phrase-substring
   match. "spatial accessibility hospitals" returned 0 hits against a live graph holding 9
   spatial-accessibility elements.
2. ``doc_id`` was Neo4j's internal element id ("4:f84f361b-...:532"). The ``_id`` property it
   tried first exists on 0 of 3205 nodes, so the fallback always fired. Every other retrieval
   arm keys ``doc_id`` on the platform UUID, so graph hits could not dedupe against them and
   cited links that resolve to nothing.
3. ``MATCH (r)`` swept all 3205 nodes. 2386 of them are :Alias/:Contributor, which carry no
   ``visibility`` — and a missing visibility counts as public — so a contributor's name
   matching a query term could be returned as a search result.

These run against a fake driver, so they are pure: the live-graph numbers in the docstrings
are what motivated each assertion, not what the tests require.
"""

from __future__ import annotations

import pytest

from rag_pipeline.search import neo4j as n4


class _FakeNode(dict):
    """Stands in for neo4j.graph.Node: dict(node) yields properties, plus element_id."""

    def __init__(self, props, element_id="4:fake:1"):
        super().__init__(props)
        self.element_id = element_id


class _FakeSession:
    def __init__(self, driver):
        self._driver = driver

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, cypher, **params):
        self._driver.calls.append({"cypher": cypher, "params": params})
        return list(self._driver.rows)


class _FakeDriver:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def session(self, **_kw):
        return _FakeSession(self)


@pytest.fixture(autouse=True)
def _node_class(monkeypatch):
    """_records_to_hits type-checks against the real Node class; point it at the fake."""
    monkeypatch.setattr(n4, "_neo4j_components",
                        lambda: {"GraphDatabase": object, "Node": _FakeNode})


def _row(props, element_id="4:fake:1", score=1.0):
    return {"node": _FakeNode(props, element_id), "score": score}


# ------------------------------------------------------------------ term extraction

def test_a_multi_word_question_becomes_multiple_terms():
    terms = n4.neo4j_query_terms("How do I compute spatial accessibility to hospitals?")
    assert "spatial" in terms and "accessibility" in terms and "hospitals" in terms
    assert "how" not in terms and "the" not in terms


def test_domain_generic_words_are_dropped():
    """'notebook'/'data'/'platform' match nearly every element, so they carry no signal."""
    terms = n4.neo4j_query_terms("show me notebooks on the platform about flooding")
    assert "flooding" in terms
    assert not ({"notebook", "notebooks", "platform", "show", "data"} & set(terms))


def test_terms_are_capped_and_deduped():
    terms = n4.neo4j_query_terms("flood flood flood " + " ".join(f"term{i}" for i in range(20)))
    assert len(terms) <= 8
    assert terms.count("flood") == 1


def test_an_all_stopword_query_still_yields_something():
    """Better a weak match than a silent empty result."""
    assert n4.neo4j_query_terms("what is the data") != []


# ------------------------------------------------------------------ the query binding

def test_the_query_binds_terms_not_the_whole_phrase():
    driver = _FakeDriver([])
    n4.get_neo4j_search_results("spatial accessibility hospitals", limit=5, driver=driver)
    params = driver.calls[0]["params"]
    assert isinstance(params["terms"], list) and len(params["terms"]) >= 3
    assert "spatial accessibility hospitals" not in params["terms"], (
        "the whole phrase is being matched again — this is the 0-hit bug")


def test_the_cypher_matches_any_term_not_all():
    cypher = n4._build_neo4j_keyword_cypher()
    assert "any(t IN $terms" in cypher
    assert "$q" not in cypher, "the phrase parameter is still referenced"


# ------------------------------------------------------------------ id identity

def test_doc_id_is_the_platform_uuid_not_the_internal_element_id():
    uuid = "3b45070e-a63e-496f-9493-0947d680192e"
    driver = _FakeDriver([_row({"id": uuid, "title": "Accessibility"},
                               element_id="4:f84f361b-d1e0-4005-ae20-75db4c78167b:532")])
    hits = n4.get_neo4j_search_results("accessibility", limit=5, driver=driver)
    assert len(hits) == 1
    assert hits[0]["_source"]["doc_id"] == uuid
    assert not hits[0]["_source"]["doc_id"].startswith("4:"), (
        "emitting Neo4j's internal element id breaks dedupe against every other arm")


def test_a_node_without_a_platform_id_still_returns_something_stable():
    driver = _FakeDriver([_row({"title": "No id here"}, element_id="4:fake:77")])
    hits = n4.get_neo4j_search_results("here", limit=5, driver=driver)
    assert hits and hits[0]["_source"]["doc_id"] == "4:fake:77"


# ------------------------------------------------------------------ label scoping

def test_only_knowledge_element_labels_are_searched():
    driver = _FakeDriver([])
    n4.get_neo4j_search_results("flood", limit=5, driver=driver)
    labels = set(driver.calls[0]["params"]["labels"])
    assert {"Notebook", "Dataset", "Publication"} <= labels
    assert not ({"Alias", "Contributor", "User"} & labels), (
        "internal nodes are searchable; a contributor name can surface as a result")
    assert "any(l IN labels(r) WHERE l IN $labels)" in driver.calls[0]["cypher"]


# ------------------------------------------------------------------ visibility + limits

def test_private_elements_are_never_surfaced():
    driver = _FakeDriver([
        _row({"id": "pub-1", "title": "Public", "visibility": "public"}),
        _row({"id": "prv-1", "title": "Private", "visibility": "private"}),
    ])
    hits = n4.get_neo4j_search_results("public private", limit=5, driver=driver)
    assert [h["_source"]["doc_id"] for h in hits] == ["pub-1"]


def test_an_empty_query_never_reaches_the_driver():
    driver = _FakeDriver([_row({"id": "x", "title": "t"})])
    assert n4.get_neo4j_search_results("   ", limit=5, driver=driver) == []
    assert driver.calls == []


def test_the_limit_is_clamped_to_a_sane_range():
    driver = _FakeDriver([])
    n4.get_neo4j_search_results("flood", limit=10_000, driver=driver)
    assert driver.calls[0]["params"]["limit"] <= 100
    n4.get_neo4j_search_results("flood", limit=0, driver=driver)
    assert driver.calls[-1]["params"]["limit"] >= 1


def test_a_driver_failure_degrades_to_empty_not_an_exception():
    class _Boom(_FakeDriver):
        def session(self, **_kw):
            raise RuntimeError("Unable to retrieve routing information")

    assert n4.get_neo4j_search_results("flood", limit=5, driver=_Boom([])) == []


# ------------------------------------------------------------------ the enable gate

@pytest.mark.parametrize("value,expected", [
    ("0", False), ("false", False), ("no", False), ("off", False), ("OFF", False),
    ("1", True), ("true", True), ("", True),
])
def test_neo4j_enabled_respects_the_env_gate(monkeypatch, value, expected):
    monkeypatch.setenv("NEO4J_ENABLED", value)
    assert n4.neo4j_enabled() is expected


def test_neo4j_defaults_to_enabled_when_unset(monkeypatch):
    """A deployment must not silently lose the graph because a variable is missing."""
    monkeypatch.delenv("NEO4J_ENABLED", raising=False)
    assert n4.neo4j_enabled() is True
