"""Linking a publication to the element that implements it.

The symbol-matching route produces one edge across the whole corpus, and its ceiling is
structural: `tools_referenced` records what SOFTWARE a paper used — "OSMnx", "Python", "Census
Bureau API" — while a method library is indexed by FUNCTION. Papers name libraries; libraries
contain functions.

Drawn at the element level instead, from two kinds of evidence a person left behind. The tests
here pin the parts that were wrong in the first version and would be easy to get wrong again.
"""

from __future__ import annotations

import pytest

from extractors.analysis import spec_links as sl


# ------------------------------------------------------------------ normalisation

@pytest.mark.parametrize("raw, expected", [
    ("Shaowen Wang", "shaowen wang"),
    # The platform's author lists carry the conjunction into the last entry, so the same person
    # appears as two different strings across two records.
    ("and Shaowen Wang", "shaowen wang"),
    ("  Jeon-Young  Kang ", "jeon-young kang"),
    ("Vincent L. Freeman", "vincent l. freeman"),
])
def test_one_person_has_one_key(raw, expected):
    assert sl.author_key(raw) == expected


@pytest.mark.parametrize("raw", [
    "10.1080/13658816.2024.2326445",
    "https://doi.org/10.1080/13658816.2024.2326445",
    "http://dx.doi.org/10.1080/13658816.2024.2326445",
    "10.1080/13658816.2024.2326445.",
])
def test_a_doi_is_comparable_however_it_was_written(raw):
    assert sl.normalize_doi(raw) == "10.1080/13658816.2024.2326445"


def test_dois_are_found_in_running_text():
    text = "We follow the method of (doi: 10.1016/j.healthplace.2009.06.002) and extend it."
    assert "10.1016/j.healthplace.2009.06.002" in sl.dois_in(text)


# ------------------------------------------------------------------ citation evidence

def test_a_cited_doi_links_without_corroboration():
    """A DOI in a notebook is not a coincidence of vocabulary — a person wrote down which paper
    this implements. It is the only evidence here that needs no second signal."""
    links = sl.links_from_citation(
        "pub1", "https://doi.org/10.1080/13658816.2024.2326445",
        {"nb1": "see 10.1080/13658816.2024.2326445 for the method",
         "nb2": "unrelated"})
    assert [l.target_element for l in links] == ["nb1"]
    assert links[0].confidence == "high"
    assert links[0].evidence == "cited_doi"


def test_a_publication_does_not_cite_itself():
    links = sl.links_from_citation("pub1", "10.1080/13658816.2024.2326445",
                                  {"pub1": "10.1080/13658816.2024.2326445"})
    assert links == []


def test_no_doi_means_no_citation_link():
    assert sl.links_from_citation("pub1", "", {"nb1": "10.1080/13658816.2024.2326445"}) == []


# ------------------------------------------------------------------ author + topic

def _authors(mapping):
    return {sl.author_key(k): v for k, v in mapping.items()}


def test_the_top_candidate_links_when_it_stands_apart():
    """The E2SFCA shape: 0.7364 against 0.443 for the runner-up."""
    links = sl.links_from_authors(
        "pub1", ["Alexander Michels"],
        _authors({"Alexander Michels": ["nbA", "nbB"]}),
        {"nbA": 0.7364, "nbB": 0.443}.get)
    assert [l.target_element for l in links] == ["nbA"]
    assert links[0].confidence == "medium"
    assert links[0].detail["runner_up_score"] == 0.443
    assert links[0].detail["shared_authors"] == ["alexander michels"]


def test_clustered_scores_link_to_nothing():
    """The failure the margin exists for. `a182e493` scored 0.7011 / 0.6971 / 0.6254 — three
    candidates within 11%, which is the shape of "several of this author's notebooks are about
    this topic", not of one implementing the paper. A relative floor admitted all three."""
    links = sl.links_from_authors(
        "pub1", ["A Person"],
        _authors({"A Person": ["nbA", "nbB", "nbC"]}),
        {"nbA": 0.7011, "nbB": 0.6971, "nbC": 0.6254}.get)
    assert links == []


def test_a_lone_candidate_still_has_to_be_relevant():
    """Otherwise a spec whose only co-author wrote something unrelated links to it for lack of
    competition."""
    assert sl.links_from_authors("pub1", ["A"], _authors({"A": ["nbA"]}),
                                 {"nbA": 0.001}.get) == []
    assert sl.links_from_authors("pub1", ["A"], _authors({"A": ["nbA"]}),
                                 {"nbA": 0.9}.get)[0].target_element == "nbA"


def test_at_most_one_implementation_is_claimed():
    """A publication describes one method. A second "implementation" is a claim this evidence
    cannot support, and a wrong IMPLEMENTED_BY edge is fabricated provenance, not a gap."""
    links = sl.links_from_authors(
        "pub1", ["A"], _authors({"A": ["nbA", "nbB", "nbC"]}),
        {"nbA": 0.9, "nbB": 0.2, "nbC": 0.1}.get)
    assert len(links) == 1


def test_author_overlap_alone_proposes_nothing():
    """Author overlap by itself gave one spec SEVENTEEN candidates including "CyberGIS-Compute
    Core" — a co-authorship graph, not an implementation link. Without a scorer that separates
    them, nothing is emitted."""
    links = sl.links_from_authors(
        "pub1", ["A"], _authors({"A": [f"nb{i}" for i in range(17)]}),
        lambda e: 0.5)
    assert links == []


def test_a_spec_with_no_shared_author_proposes_nothing():
    assert sl.links_from_authors("pub1", ["A"], _authors({"B": ["nbA"]}), {"nbA": 0.9}.get) == []


# ------------------------------------------------------------------ merging

def test_a_pair_found_twice_keeps_its_strongest_evidence():
    """An edge carrying its weakest justification understates what is known about it."""
    weak = sl.SpecLink("pub1", "nb1", "medium", "shared_author_and_topic")
    strong = sl.SpecLink("pub1", "nb1", "high", "cited_doi")
    merged = sl.merge([weak], [strong])
    assert len(merged) == 1
    assert merged[0].confidence == "high" and merged[0].evidence == "cited_doi"


def test_an_edge_carries_the_vocabulary_the_repo_already_uses():
    edge = sl.SpecLink("pub1", "nb1", "high", "cited_doi",
                      {"doi": "10.1080/x"}).as_edge()
    assert edge["rel"] == "IMPLEMENTED_BY"
    assert edge["src"] == "pub1::methodspec" and edge["dst"] == "nb1"
    assert edge["detail"]["confidence"] == "high" and edge["detail"]["by"] == "cited_doi"


# ------------------------------------------------------------------ against the record

@pytest.mark.integration
def test_the_corpus_links_the_paper_to_its_implementation():
    from extractors import kb_db

    try:
        ctx = kb_db.connect()
        conn = ctx.__enter__()
    except Exception as exc:
        pytest.skip(f"no database: {type(exc).__name__}")
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM spec_link")
            if not cur.fetchone()[0]:
                pytest.skip("no links; run scripts/build_spec_links.py first")
            cur.execute("""SELECT target_element, confidence FROM spec_link
                            WHERE spec_element = '355786a5'""")
            rows = cur.fetchall()
        assert [r[0] for r in rows] == ["3b45070e"], rows
    finally:
        ctx.__exit__(None, None, None)
