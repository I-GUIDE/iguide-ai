"""Joining the agent KB to search results by element id.

The KB was reachable only by TEXT: ``agent_kb_search`` matches a query against
``title``/``contents``/``extracted.embed_text``. That leaves a hole with a sharp edge — a hit found
by **spatial** search (a bounding box) or by **graph** search (a relation) can never text-match its
own extracted content, so its units, schema and method spec stayed invisible however good they
were. Both sides carry the same element id and nothing joined on it.

Two consequences, and the second matters as much as the first:

* an element found by any method now carries what extraction produced from it;
* standalone KB rows whose parent is already in the result set are folded away, because they were
  competing with their own element for an evidence slot. Deduplicating evidence by parent was an
  explicit exit criterion the text-only union could not meet.

The cluster is not required: the join is exercised through an injected client, so these run
anywhere.
"""

from __future__ import annotations

import pytest

from rag_pipeline.search.agent_kb import (MAX_BLOCKS_PER_ELEMENT, MAX_UNITS_PER_ELEMENT,
                                          attach_kb_to_documents, kb_for_elements)

CONTRACT = {
    "library_symbol": "calculate_buffers",
    "library_module": "iguide_methods.ke_b1fa548b.v_abc123",
    "slice_sha": "abc123",
    "signature": "def calculate_buffers(gdf: gpd.GeoDataFrame, buffer: Number)",
    "doc_summary": "Replace geometry with buffers.",
    "params": [{"name": "gdf", "annotation": "gpd.GeoDataFrame", "declared_unit": "metres",
                "crs_expectation": "projected", "required": True}],
    "invariants": [{"check": "projected_crs", "target": "gdf"}],
    "requirements": {"pip": ["geopandas"]},
    "callability": {"verdict": "callable", "reason": ""},
    "import_line": "from iguide_methods.ke_b1fa548b.v_abc123 import calculate_buffers",
}


def _doc(doc_id, parent, extracted, index="iguide_agent_method_units"):
    return {"_index": index, "_id": doc_id,
            "_source": {"doc_id": doc_id, "title": doc_id,
                        "extracted": {"parent_doc_id": parent, **extracted}}}


class FakeClient:
    """Answers the terms lookup the join issues, honouring the parent filter."""

    def __init__(self, docs, missing=()):
        self._docs = list(docs)
        self._missing = set(missing)
        outer = self

        class _Indices:
            def exists(self, index):
                return index not in outer._missing

        self.indices = _Indices()
        self.searches = []

    def search(self, index, body):
        self.searches.append(index)
        shoulds = (((body or {}).get("query") or {}).get("bool") or {}).get("should") or []
        wanted = set()
        for clause in shoulds:
            for values in (clause.get("terms") or {}).values():
                wanted.update(values)
        hits = [d for d in self._docs
                if d["_index"] == index
                and ((d["_source"].get("extracted") or {}).get("parent_doc_id") in wanted
                     or d["_source"].get("doc_id") in wanted)]
        return {"hits": {"hits": hits}}


# ------------------------------------------------------------------ lookup by id

def test_an_element_found_by_geometry_still_gets_its_extracted_content():
    """The case text search structurally cannot reach: nothing in the query matched the
    sub-documents, because the match was a bounding box."""
    client = FakeClient([_doc("b1fa548b::unit::calculate_buffers", "b1fa548b",
                              {"kind": "method_unit", "unit": CONTRACT})])
    summaries = kb_for_elements(["b1fa548b"], client=client)
    assert "b1fa548b" in summaries
    unit = summaries["b1fa548b"]["units"][0]
    assert unit["symbol"] == "calculate_buffers"
    assert unit["import_line"].endswith("import calculate_buffers")


def test_an_element_with_nothing_extracted_is_absent_not_an_empty_stub():
    """A caller should be able to test membership. A stub would make "we have nothing for this"
    indistinguishable from "we have an empty something"."""
    client = FakeClient([])
    assert kb_for_elements(["ffffffff"], client=client) == {}


def test_no_ids_means_no_queries():
    client = FakeClient([])
    assert kb_for_elements([], client=client) == {}
    assert kb_for_elements([None, "", "  "], client=client) == {}
    assert client.searches == [], "an empty id list still hit the cluster"


def test_an_unreachable_index_costs_only_its_own_contribution():
    """Enrichment is additive. A turn must still answer from what it already has."""
    client = FakeClient([_doc("e1::unit::f", "e1", {"kind": "method_unit", "unit": CONTRACT})],
                        missing={"iguide_agent_notebook_blocks"})
    assert "e1" in kb_for_elements(["e1"], client=client)


def test_a_dataset_element_is_its_own_document():
    """Dataset and publication records ARE the element, so they are found by doc_id rather than by
    parent — the lookup has to ask both ways."""
    client = FakeClient([{
        "_index": "iguide_agent_datasets", "_id": "265e6957",
        "_source": {"doc_id": "265e6957", "title": "Crime",
                    "extracted": {"kind": "dataset", "parent_doc_id": "265e6957",
                                  "format": "CSV", "crs": "EPSG:4326", "row_count": 20000,
                                  "schema": ["ID", "Date", "Latitude", "Longitude"]}}}])
    summary = kb_for_elements(["265e6957"], client=client)["265e6957"]
    assert summary["dataset"]["crs"] == "EPSG:4326"
    assert summary["dataset"]["row_count"] == 20000


def test_a_publication_element_carries_its_method_spec():
    client = FakeClient([{
        "_index": "iguide_agent_publication_methodspecs", "_id": "31fd4fc6",
        "_source": {"doc_id": "31fd4fc6", "title": "Corridors",
                    "extracted": {"kind": "publication", "parent_doc_id": "31fd4fc6",
                                  "status": "llm_extracted", "is_method_spec": True,
                                  "steps": [f"step {i}" for i in range(30)],
                                  "tools_referenced": ["pNISE"]}}}])
    spec = kb_for_elements(["31fd4fc6"], client=client)["31fd4fc6"]["publication"]
    assert spec["tools_referenced"] == ["pNISE"]
    assert len(spec["steps"]) <= 12, "a 30-step paper must not eat the evidence budget"


# ------------------------------------------------------------------ budget and ordering

def test_a_large_notebook_does_not_consume_the_whole_evidence_budget():
    blocks = [_doc(f"nb1::block::{i}", "nb1",
                   {"kind": "notebook_block", "order": i,
                    "block": {"markdown_context": f"cell {i}"}},
                   index="iguide_agent_notebook_blocks") for i in range(40)]
    summary = kb_for_elements(["nb1"], client=FakeClient(blocks))["nb1"]
    assert len(summary["blocks"]) == MAX_BLOCKS_PER_ELEMENT
    assert summary["block_count"] == 40, "the true count must survive the cap"
    assert [b["order"] for b in summary["blocks"]] == [0, 1, 2, 3], "blocks lost their order"


def test_importable_units_outrank_bare_ones_when_the_cap_bites():
    """The cap decides what survives, so the ordering decides what the agent sees. A unit it can
    import is worth more of the budget than a name it cannot."""
    bare = {**{k: v for k, v in CONTRACT.items() if k != "import_line"},
            "library_symbol": "zzz_bare"}
    docs = [_doc(f"e1::unit::bare{i}", "e1", {"kind": "method_unit",
                                              "unit": {**bare, "library_symbol": f"bare{i}"}})
            for i in range(MAX_UNITS_PER_ELEMENT)]
    docs.append(_doc("e1::unit::calculate_buffers", "e1",
                     {"kind": "method_unit", "unit": CONTRACT}))
    summary = kb_for_elements(["e1"], client=FakeClient(docs))["e1"]
    assert summary["units"][0]["symbol"] == "calculate_buffers"
    assert summary["unit_count"] == MAX_UNITS_PER_ELEMENT + 1


# ------------------------------------------------------------------ the join

def _sweep_docs(parent="b1fa548b"):
    return [
        {"doc_id": parent, "source": "spatial", "title": "found by bounding box"},
        {"doc_id": "no-such-element", "source": "keyword", "title": "nothing extracted"},
        {"doc_id": f"{parent}::block::3", "source": "agent_kb", "parent_doc_id": parent,
         "title": "a block of the element above"},
    ]


def _client(parent="b1fa548b"):
    return FakeClient([_doc(f"{parent}::unit::calculate_buffers", parent,
                            {"kind": "method_unit", "unit": CONTRACT})])


def test_the_join_enriches_the_element_and_folds_its_duplicate_row():
    result = attach_kb_to_documents(_sweep_docs(), client=_client())
    assert result["attached"] == 1
    assert result["folded"] == 1
    ids = [d["doc_id"] for d in result["documents"]]
    assert "b1fa548b::block::3" not in ids, "a block competed with its own element for a slot"
    enriched = next(d for d in result["documents"] if d["doc_id"] == "b1fa548b")
    assert enriched["extracted"]["units"][0]["symbol"] == "calculate_buffers"


def test_a_document_with_nothing_extracted_is_left_exactly_as_it_was():
    result = attach_kb_to_documents(_sweep_docs(), client=_client())
    plain = next(d for d in result["documents"] if d["doc_id"] == "no-such-element")
    assert "extracted" not in plain


def test_a_kb_row_whose_parent_is_absent_is_kept():
    """Folding is for duplicates. A KB hit for an element no other method found is the only
    evidence there is for it, and dropping it would lose a result."""
    docs = [{"doc_id": "other::block::1", "source": "agent_kb", "parent_doc_id": "other",
             "title": "an orphan block"}]
    result = attach_kb_to_documents(docs, client=_client())
    assert [d["doc_id"] for d in result["documents"]] == ["other::block::1"]
    assert result["folded"] == 0


def test_the_actionable_list_is_what_the_agent_can_run():
    """References and actionable items are different things. An import line is the second: it goes
    straight into `execute_code`, where the library is mounted read-only."""
    result = attach_kb_to_documents(_sweep_docs(), client=_client())
    assert len(result["actionable"]) == 1
    item = result["actionable"][0]
    assert item["element"] == "b1fa548b"
    assert item["import_line"].startswith("from iguide_methods.")
    assert item["requirements"] == ["geopandas"]


def test_a_unit_with_no_import_line_is_not_offered_as_actionable():
    """It is still a reference — it says the method exists. Offering it as runnable would hand the
    agent a line that fails inside the sandbox, where it cannot recover."""
    contract = {k: v for k, v in CONTRACT.items() if k != "import_line"}
    client = FakeClient([_doc("b1fa548b::unit::calculate_buffers", "b1fa548b",
                              {"kind": "method_unit", "unit": contract})])
    result = attach_kb_to_documents(_sweep_docs(), client=client)
    assert result["actionable"] == []
    assert result["documents"][0]["extracted"]["units"][0]["signature"]


def test_an_empty_result_set_is_handled_without_touching_the_cluster():
    client = FakeClient([])
    assert attach_kb_to_documents([], client=client)["documents"] == []
    assert client.searches == []


def test_non_dict_entries_do_not_break_the_join():
    result = attach_kb_to_documents([None, "junk", {"doc_id": "b1fa548b", "source": "keyword"}],
                                    client=_client())
    assert len(result["documents"]) == 1
    assert result["attached"] == 1


# ------------------------------------------------------------------ what the model sees

def _rendered(doc):
    from agent_runtime.supervisor.evidence_subgraph import _format_documents

    return _format_documents([doc])


def test_the_enrichment_reaches_the_evidence_the_model_reads():
    """The join enriched a document the model never saw the enrichment of: `_doc_block` rendered
    title, url and contents only. An attached contract that is not rendered is not wired up."""
    result = attach_kb_to_documents(_sweep_docs(), client=_client())
    element = next(d for d in result["documents"] if d["doc_id"] == "b1fa548b")
    text = _rendered(element)
    assert "from iguide_methods." in text, "the import line never reached the prompt"
    assert "calculate_buffers" in text
    assert "geopandas" in text, "the dependency the caller has to install is missing"


def test_runnable_and_reference_are_labelled_differently():
    """An import line runs; a method spec informs. Rendering both as undifferentiated prose is how
    the agent kept saying "adapt this notebook" while an importable function sat one field away."""
    result = attach_kb_to_documents(_sweep_docs(), client=_client())
    element = next(d for d in result["documents"] if d["doc_id"] == "b1fa548b")
    assert "RUNNABLE METHODS" in _rendered(element)


def test_a_dataset_schema_reaches_the_prompt():
    client = FakeClient([{
        "_index": "iguide_agent_datasets", "_id": "265e6957",
        "_source": {"doc_id": "265e6957", "extracted": {
            "kind": "dataset", "parent_doc_id": "265e6957", "format": "CSV",
            "crs": "EPSG:4326", "row_count": 20000,
            "schema": ["ID", "Date", "Primary Type", "Latitude", "Longitude"]}}}])
    docs = [{"doc_id": "265e6957", "source": "spatial", "title": "Chicago crime"}]
    element = attach_kb_to_documents(docs, client=client)["documents"][0]
    text = _rendered(element)
    assert "DATASET STRUCTURE" in text
    assert "Primary Type" in text and "EPSG:4326" in text


def test_a_publication_method_reaches_the_prompt():
    client = FakeClient([{
        "_index": "iguide_agent_publication_methodspecs", "_id": "31fd4fc6",
        "_source": {"doc_id": "31fd4fc6", "extracted": {
            "kind": "publication", "parent_doc_id": "31fd4fc6", "status": "llm_extracted",
            "steps": ["Acquire EISPC raster data", "Derive two cost surfaces"]}}}])
    docs = [{"doc_id": "31fd4fc6", "source": "keyword", "title": "Corridors"}]
    text = _rendered(attach_kb_to_documents(docs, client=client)["documents"][0])
    assert "METHOD THIS PAPER DESCRIBES" in text
    assert "Derive two cost surfaces" in text


def test_a_document_with_nothing_attached_renders_exactly_as_before():
    """Enrichment is additive. A result with no extracted content must not gain a stray header."""
    plain = {"doc_id": "x", "title": "Plain", "contents": "Just a description."}
    text = _rendered(plain)
    assert text.strip().endswith("Just a description.")
    assert "RUNNABLE" not in text and "DATASET STRUCTURE" not in text


def test_the_rendered_block_is_bounded():
    """It rides on top of the document's own contents, so an element with 200 blocks must not
    crowd out every other result."""
    from agent_runtime.supervisor.evidence_subgraph import (EXTRACTED_MAX_CHARS,
                                                            _render_extracted)

    huge = {"units": [{"symbol": f"f{i}", "signature": "def f()" * 40,
                       "import_line": "from x import y" * 20} for i in range(50)],
            "unit_count": 50}
    assert len(_render_extracted(huge)) <= EXTRACTED_MAX_CHARS


def test_a_capped_unit_list_says_how_many_more_there_are():
    from agent_runtime.supervisor.evidence_subgraph import _render_extracted

    text = _render_extracted({"units": [{"symbol": "a", "import_line": "from x import a"}],
                              "unit_count": 9})
    assert "9 methods in total" in text and "kb_method_search" in text


@pytest.mark.parametrize("value", [None, {}, "not a dict", 42, []])
def test_a_malformed_extracted_payload_renders_nothing_rather_than_raising(value):
    from agent_runtime.supervisor.evidence_subgraph import _render_extracted

    assert _render_extracted(value) == ""


def test_a_loader_says_where_its_staged_path_comes_from():
    """A loader advertises `load_x(staged_path)` and nothing said where a staged_path comes from.

    Measured on a live turn: the agent read this signature, called the loader without staging, got
    "block not found", and fell back to a different tool over a different subset -- reporting THEFT
    as 9,993 records where the file says 27,824. `stage_element` was available and appeared zero
    times in the whole run state. A generic prompt rule did not survive the moment of choice, so
    the parameter carries its own instruction at the point of use.
    """
    from agent_runtime.supervisor.evidence_subgraph import _render_extracted

    text = _render_extracted(
        {"units": [{"symbol": "load_chicago_crime_data_2026",
                    "signature": "def load_chicago_crime_data_2026(staged_path)",
                    "import_line": "from iguide_methods.ke__265e6957.v_abc import load_x"}]},
        "265e6957")
    assert "stage_element" in text
    assert "265e6957" in text
    assert "not in the sandbox until you do" in text


def test_a_unit_that_takes_no_staged_path_gets_no_staging_instruction():
    """Noise on every other unit would train the model to skim past it."""
    from agent_runtime.supervisor.evidence_subgraph import _render_extracted

    text = _render_extracted(
        {"units": [{"symbol": "calculate_buffers",
                    "signature": "def calculate_buffers(gdf, buffer)",
                    "import_line": "from iguide_methods.x.v_a import calculate_buffers"}]},
        "b1fa548b")
    assert "stage_element" not in text
