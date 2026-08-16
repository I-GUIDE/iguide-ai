"""The one cross-element edge extraction can actually see.

Every other edge the four extractors emit is parent-to-child within a single element. Measured
over the 174 cached corpus notebooks: 4,212 edges, of which 4,071 (96.7%) satisfy
``dst.startswith(src + "::")`` — a pure function of the id, recoverable by the shipped
``doc_ids.parent_doc_id`` — and ``cross_element`` is **0**. So the extraction graph was a forest
of 419 stars, and no partition, traversal or community over it could say anything.

Authors do link elements to each other; they just do it in prose and in download lines. Measured:
179 i-guide.io URLs across the corpus, 100 carrying an element UUID, 30 source notebooks, 40
distinct targets, **41 of 41 resolving to a live platform element**, giving 99 edges over 56
nodes with a largest component of 28.

The tests below fix the three decisions that make that number trustworthy rather than merely
large: what is excluded, that hosts mean different things, and that a self-link is not an edge.
"""

from __future__ import annotations

import pytest

from extractors.analysis.citations import Citation, platform_citations

NB = "6f1f5a52-8c9e-4a3f-9d2e-1b0c7e4a55d1"
DS = "0a2b4c6d-1e3f-4a5b-8c7d-9e0f1a2b3c4d"


# ------------------------------------------------------------------ what counts as a citation

def test_a_platform_element_link_is_a_citation():
    (c,) = platform_citations(f"See https://platform.i-guide.io/notebooks/{NB} for the method.")
    assert c == Citation(element_id=NB, rel="CITES", path_hint="notebooks", host="platform")


def test_a_storage_download_is_USES_not_CITES():
    """The distinction is the point: linking to an element's page is a reference, but pulling a
    file out of its bucket means the notebook consumes it. Collapsing the two would report a
    data dependency as a reading suggestion."""
    (c,) = platform_citations(
        f"!wget https://storage.i-guide.io/datasets/{DS}/mii_brazil_analysis_ready.gpkg.zip")
    assert c.rel == "USES" and c.element_id == DS


def test_the_uses_edge_points_at_a_real_node():
    """Contrast with publication_extractor.py:401, whose USES dst is `str(ds)` — an LLM-named
    dataset string that nothing in the repo resolves to an element id. This dst is a UUID."""
    (c,) = platform_citations(f"https://storage.i-guide.io/datasets/{DS}/x.zip")
    assert c.element_id == DS
    assert "-" in c.element_id and len(c.element_id) == 36


@pytest.mark.parametrize("url", [
    "https://i-guide.io/projects/vulnerability-analysis-for-aging-dam-infrastructure/",
    "https://i-guide.io/spatial-ai-challenge-2024/accepted-abstracts/",
    "https://backend.i-guide.io/user-uploads/thumbnails/1720206819181-spastc_scaled.jpg",
    "https://platform.i-guide.io/notebooks/",
    "https://platform.i-guide.io/oers",
])
def test_urls_without_an_element_uuid_are_not_citations(url):
    """All five shapes are in the real corpus (13, 2, 2, 1 and 1 hits respectively). Requiring a
    full 8-4-4-4-12 UUID is what excludes them — no host denylist needed. The thumbnail case is
    the sharp one: `1720206819181-spastc_scaled` is a long digit run that a looser hex pattern
    would happily accept."""
    assert platform_citations(url) == []


def test_a_url_building_template_is_not_a_citation():
    """`platform.i-guide.io/{element_type` appears once in the corpus — code that CONSTRUCTS a
    URL. It names no element, so it must not produce an edge."""
    assert platform_citations('f"https://platform.i-guide.io/{element_type}/{eid}"') == []


# ------------------------------------------------------------------ robustness on real text

@pytest.mark.parametrize("trailing", ["</td", "`", ".", ")", ",", "]"])
def test_trailing_junk_does_not_corrupt_the_id(trailing):
    """The corpus contains `…zip</td`, a trailing backtick and a trailing period on otherwise
    valid URLs. Matching stops at the UUID, so nothing needs stripping."""
    (c,) = platform_citations(f"https://platform.i-guide.io/notebooks/{NB}{trailing}")
    assert c.element_id == NB


def test_uppercase_uuids_and_hosts_normalise():
    (c,) = platform_citations(f"HTTPS://PLATFORM.I-GUIDE.IO/Notebooks/{NB.upper()}")
    assert c.element_id == NB and c.path_hint == "notebooks"


def test_http_and_https_both_match():
    assert platform_citations(f"http://platform.i-guide.io/notebooks/{NB}")[0].element_id == NB


def test_the_same_element_cited_four_times_is_one_citation():
    text = " ".join([f"https://platform.i-guide.io/notebooks/{NB}"] * 4)
    assert len(platform_citations(text)) == 1


def test_page_link_and_download_of_the_same_element_are_two_citations():
    """De-duplication is keyed on (element, rel). One element both referenced and consumed is two
    distinct facts, and collapsing them would silently drop the data dependency."""
    text = (f"https://platform.i-guide.io/datasets/{DS} and "
            f"https://storage.i-guide.io/datasets/{DS}/f.zip")
    assert sorted(c.rel for c in platform_citations(text)) == ["CITES", "USES"]


def test_order_of_appearance_is_preserved():
    text = f"https://platform.i-guide.io/datasets/{DS} then https://platform.i-guide.io/notebooks/{NB}"
    assert [c.element_id for c in platform_citations(text)] == [DS, NB]


def test_empty_and_none_are_safe():
    assert platform_citations("") == []
    assert platform_citations(None) == []


def test_the_path_hint_is_recorded_but_never_asserted_as_a_type():
    """A URL under /notebooks/ can point at any element; the segment is a routing artefact.
    Recording it as `path_hint` rather than `resource_type` keeps a corpus-level validator
    honest about what extraction actually knew."""
    (c,) = platform_citations(f"https://platform.i-guide.io/notebooks/{DS}")
    assert c.path_hint == "notebooks"
    assert not hasattr(c, "resource_type")


# ------------------------------------------------------------------ through the extractor

def _nb(cells):
    import nbformat

    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"language": "python", "name": "python3"}
    nb.cells = [nbformat.v4.new_markdown_cell(s) if k == "md"
                else nbformat.v4.new_code_cell(s) for k, s in cells]
    return nb


def _extract(tmp_path, cells, element_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"):
    import nbformat

    from extractors.base import ExtractContext
    from extractors.notebook_extractor import NotebookExtractor

    p = tmp_path / "n.ipynb"
    nbformat.write(_nb(cells), str(p))
    return NotebookExtractor().extract(str(p), ctx=ExtractContext(
        element_id=element_id, element_type="notebook"))


def test_a_citation_in_a_markdown_cell_becomes_an_edge(tmp_path):
    """Markdown is where citations live, and the extractor's cell loop buffers markdown into
    `md_context` for the NEXT code cell — so a scan that ran only over code cells would find
    nothing here."""
    res = _extract(tmp_path, [("md", f"Builds on https://platform.i-guide.io/notebooks/{NB}"),
                              ("code", "x = 1")])
    cites = [e for e in res.edges if e.rel == "CITES"]
    assert len(cites) == 1 and cites[0].dst == NB


def test_a_trailing_markdown_cell_is_not_dropped(tmp_path):
    """A markdown cell with no code cell after it never becomes anyone's md_context. Before the
    scan moved above the loop's `continue`, this citation was invisible."""
    res = _extract(tmp_path, [("code", "x = 1"),
                              ("md", f"See https://platform.i-guide.io/notebooks/{NB}")])
    assert [e.dst for e in res.edges if e.rel == "CITES"] == [NB]


def test_a_notebook_citing_itself_emits_no_edge(tmp_path):
    """One of the 174 corpus notebooks links to its own element page. That is a real sentence and
    a self-loop; a graph does not want it."""
    me = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    res = _extract(tmp_path, [("md", f"https://platform.i-guide.io/notebooks/{me}")],
                   element_id=me)
    assert [e for e in res.edges if e.rel in ("CITES", "USES")] == []


def test_citation_edges_are_the_only_ones_reaching_another_element(tmp_path):
    """The characterising assertion, and it needs three buckets rather than two.

    An edge whose dst merely fails to start with the element id is not thereby cross-element.
    HAS_WORKFLOW's dst is `workflow_id_for(asset_id)` — a sha1 handle that is the doc_id of
    nothing at all (measured across the corpus: 141 such edges, 141 distinct dangling dsts, 0 of
    them an element id or an emitted doc_id). Lumping those in with citations would let a
    dangling edge masquerade as cross-element structure, which is the exact overcount this whole
    line of work exists to avoid.
    """
    import re

    res = _extract(tmp_path, [("md", f"https://platform.i-guide.io/notebooks/{NB}"),
                              ("code", "def f():\n    return 1\n")])
    me = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    uuid = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

    within = [e for e in res.edges if e.dst.startswith(me + "::")]
    reaching = [e for e in res.edges if uuid.match(e.dst) and e.dst != me]
    dangling = [e for e in res.edges if e not in within and e not in reaching]

    assert {e.rel for e in reaching} == {"CITES"}, "only a citation reaches another element"
    assert {e.rel for e in within} <= {"INCLUDES", "DEFINES"}
    # Documented, not asserted away: HAS_WORKFLOW is the known dangling case.
    assert all(e.rel == "HAS_WORKFLOW" for e in dangling)


def test_the_edge_records_how_it_was_found(tmp_path):
    """Provenance on the edge itself, so a later corpus pass can tell a parsed URL from a
    name-match heuristic without re-deriving it."""
    res = _extract(tmp_path, [("md", f"https://storage.i-guide.io/datasets/{DS}/a.zip")])
    (e,) = [e for e in res.edges if e.rel == "USES"]
    assert e.detail["by"] == "platform_url"
    assert e.detail["host"] == "storage"
    assert e.detail["confidence"] == "high"


def test_a_notebook_with_no_citations_emits_none(tmp_path):
    res = _extract(tmp_path, [("code", "import pandas as pd\ndf = pd.DataFrame()\n")])
    assert [e for e in res.edges if e.rel in ("CITES", "USES")] == []
