"""Publication extraction: an explicit status, and the first IMPLEMENTED_BY edges.

**Status.** Extraction outcome used to be a free-text ``note``, so the one thing a reader needs
— was a method actually extracted, or is this an empty shell? — required string matching on a
message that also carried exception class names. And the two cases that matter most are
indistinguishable by shape: an empty ``steps`` list from a paper that genuinely describes no
reproducible method, and an empty ``steps`` list because the LLM was unreachable. Downstream,
both rendered as "this publication describes no method" — a claim the extractor is not entitled
to make.

**IMPLEMENTED_BY.** Declared in ``base.py`` from the start and written by nothing, so a paper's
method spec and the code that realises it had no connection at all — which is most of the point
of extracting both. Matching is by symbol name, and every edge says so: ``confidence: low``,
``by: symbol_match``. Asserting that a paper's method IS this function on the strength of a
shared name would be fabricated provenance, and provenance is the one thing here that has to be
trustworthy.
"""

from __future__ import annotations

import pytest

from extractors.publication_extractor import (DEGRADED_STATUSES, STATUS_EXTRACTED,
                                              STATUS_NO_TEXT, STATUS_UNAVAILABLE,
                                              STATUS_UNPARSEABLE, extract_method,
                                              implemented_by_edges)


# ------------------------------------------------------------------ status

def test_no_text_is_its_own_status():
    spec = extract_method("")
    assert spec["status"] == STATUS_NO_TEXT and spec["degraded"] is True


def test_an_unreachable_llm_is_reported_as_unavailable(monkeypatch):
    """NOT as "no method found". The paper may describe an excellent method."""
    import rag_pipeline.llm_utils as llm

    monkeypatch.setattr(llm, "call_llm",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("connection refused")))
    spec = extract_method("A long methods section describing a real workflow.")
    assert spec["status"] == STATUS_UNAVAILABLE
    assert spec["degraded"] is True
    assert "RuntimeError" in spec["error"]
    assert spec["steps"] == []


def test_unparseable_model_output_is_distinguished_from_an_outage(monkeypatch):
    import rag_pipeline.llm_utils as llm

    monkeypatch.setattr(llm, "call_llm", lambda *a, **k: "I could not do that, sorry.")
    spec = extract_method("Methods: we did things.")
    assert spec["status"] == STATUS_UNPARSEABLE and spec["degraded"] is True


def test_a_successful_extraction_is_not_degraded(monkeypatch):
    import rag_pipeline.llm_utils as llm

    monkeypatch.setattr(llm, "call_llm", lambda *a, **k: (
        '{"summary": "2SFCA accessibility", "steps": ["build catchments", "compute ratios"],'
        ' "datasets_referenced": ["hospitals"], "tools_referenced": ["pysal"], "params": {}}'))
    spec = extract_method("Methods section.")
    assert spec["status"] == STATUS_EXTRACTED
    assert spec["degraded"] is False
    assert len(spec["steps"]) == 2


def test_every_degraded_status_is_declared_as_such():
    assert DEGRADED_STATUSES == {STATUS_UNPARSEABLE, STATUS_UNAVAILABLE, STATUS_NO_TEXT}
    assert STATUS_EXTRACTED not in DEGRADED_STATUSES


# ------------------------------------------------------------------ the asset

def _asset(monkeypatch, tmp_path, llm_reply=None, raise_exc=False):
    import rag_pipeline.llm_utils as llm

    from extractors.base import ExtractContext
    from extractors.publication_extractor import PublicationExtractor

    if raise_exc:
        monkeypatch.setattr(llm, "call_llm",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    elif llm_reply is not None:
        monkeypatch.setattr(llm, "call_llm", lambda *a, **k: llm_reply)

    path = tmp_path / "paper.txt"
    path.write_text("Methods. We computed spatial accessibility.", encoding="utf-8")
    ctx = ExtractContext(element_id="pub01", element_type="publication",
                         fields={"title": "A Paper"})
    return PublicationExtractor().extract(str(path), ctx=ctx)


def test_a_degraded_spec_says_so_at_the_START_of_contents(monkeypatch, tmp_path):
    """Prefixed, not appended: the evidence view truncates, so a caveat that appears after
    4000 characters is a caveat nobody reads."""
    result = _asset(monkeypatch, tmp_path, raise_exc=True)
    contents = result.assets[0].contents
    assert contents.startswith("[METHOD SPEC UNAVAILABLE")
    assert "NOT evidence" in contents


def test_a_degraded_spec_is_flagged_in_the_structured_payload(monkeypatch, tmp_path):
    extracted = _asset(monkeypatch, tmp_path, raise_exc=True).assets[0].extracted
    assert extracted["degraded"] is True
    assert extracted["is_method_spec"] is False
    assert extracted["status"] == STATUS_UNAVAILABLE


def test_a_good_spec_carries_no_caveat_and_is_a_method_spec(monkeypatch, tmp_path):
    reply = ('{"summary": "s", "steps": ["one"], "datasets_referenced": [],'
             ' "tools_referenced": [], "params": {}}')
    result = _asset(monkeypatch, tmp_path, llm_reply=reply)
    assert not result.assets[0].contents.startswith("[METHOD SPEC")
    assert result.assets[0].extracted["is_method_spec"] is True
    assert result.warnings == []


def test_a_degraded_spec_produces_a_warning(monkeypatch, tmp_path):
    result = _asset(monkeypatch, tmp_path, raise_exc=True)
    assert any(STATUS_UNAVAILABLE in w for w in result.warnings)


# ------------------------------------------------------------------ IMPLEMENTED_BY

@pytest.fixture()
def registry(monkeypatch):
    import agent_runtime.method_library as ml

    fake = {
        "ke_a.plot_choropleth_map": {"library_symbol": "plot_choropleth_map",
                                     "module": "iguide_methods.ke_a.v_1"},
        "ke_b.spatial_join_and_count": {"library_symbol": "spatial_join_and_count",
                                        "module": "iguide_methods.ke_b.v_2"},
        "ke_c.load_data": {"library_symbol": "load_data", "module": "iguide_methods.ke_c.v_3"},
        "plot_choropleth_map": {"library_symbol": "plot_choropleth_map",
                                "alias_for": "ke_a.plot_choropleth_map"},
        "get_url": {"ambiguous": True, "library_symbol": "get_url", "candidates": ["x", "y"]},
    }
    monkeypatch.setattr(ml, "load_registry", lambda: fake)
    return fake


def test_a_named_method_produces_an_edge(registry):
    edges = implemented_by_edges("pub::spec", ["plot_choropleth_map"])
    assert len(edges) == 1
    assert edges[0].rel == "IMPLEMENTED_BY"
    assert edges[0].dst == "ke_a.plot_choropleth_map"


def test_every_edge_admits_it_is_only_a_name_match(registry):
    edge = implemented_by_edges("pub::spec", ["spatial_join_and_count"])[0]
    assert edge.detail["confidence"] == "low"
    assert edge.detail["by"] == "symbol_match"
    assert edge.detail["matched_name"] == "spatial_join_and_count"


def test_a_dotted_reference_matches_the_bare_symbol(registry):
    """A paper writes "we used geopandas.sjoin"; the unit is named `sjoin`."""
    edges = implemented_by_edges("pub::spec", ["mypkg.plot_choropleth_map"])
    assert len(edges) == 1


def test_a_call_style_reference_matches(registry):
    assert len(implemented_by_edges("pub::spec", ["plot_choropleth_map(gdf)"])) == 1


@pytest.mark.parametrize("name", ["run", "data", "get", "load", "model", "plot", "map", "abc"])
def test_generic_names_produce_no_edges(registry, name):
    """These would otherwise link a paper to half the library."""
    assert implemented_by_edges("pub::spec", [name]) == []


def test_a_partial_name_does_not_match(registry):
    """Full symbol only — "join" must not link to spatial_join_and_count."""
    assert implemented_by_edges("pub::spec", ["join"]) == []


def test_aliases_and_ambiguous_entries_are_skipped(registry):
    """An alias would duplicate its qualified entry; an ambiguous stub has no module."""
    edges = implemented_by_edges("pub::spec", ["plot_choropleth_map", "get_url"])
    assert [e.dst for e in edges] == ["ke_a.plot_choropleth_map"]


def test_no_library_means_no_edges(monkeypatch):
    """Guessing at symbols with no registry would invent provenance outright."""
    import agent_runtime.method_library as ml

    monkeypatch.setattr(ml, "load_registry", lambda: {})
    assert implemented_by_edges("pub::spec", ["plot_choropleth_map"]) == []


def test_the_extractor_emits_the_edges(monkeypatch, tmp_path, registry):
    reply = ('{"summary": "s", "steps": ["one"], "datasets_referenced": [],'
             ' "tools_referenced": ["plot_choropleth_map"], "params": {}}')
    result = _asset(monkeypatch, tmp_path, llm_reply=reply)
    rels = [e.rel for e in result.edges]
    assert "IMPLEMENTED_BY" in rels
    assert "DESCRIBES_METHOD" in rels
