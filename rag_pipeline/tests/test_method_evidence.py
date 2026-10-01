"""A method the code peer found is evidence the answer can cite (S3), and the KB tool set is defined
once (S1, consolidated).

The 2026-10-01 integration found the code peers' KB tool names hardcoded in three places that had
drifted: the code peer's binding dropped the method tools, and the evidence allowlist never had
them. So a method a peer found could shape the code but never be cited, and the answer could not
say which platform element its numbers came from.
"""
from __future__ import annotations

import json

from agent_runtime import capability_registry
from agent_runtime.supervisor import graph as g


def test_the_kb_tool_set_is_defined_once():
    assert g._CODE_PEER_KB_TOOLS == set(capability_registry.KB_CODE_PEER_TOOLS)


def test_every_kb_tool_counts_as_evidence():
    missing = set(capability_registry.KB_CODE_PEER_TOOLS) - set(g._RETRIEVAL_TOOLS)
    assert not missing, f"tools a code peer holds whose results can never be cited: {missing}"


def _row(name, payload):
    return {"name": name, "content": json.dumps(payload)}


HIT = {"symbol": "e2sfca", "signature": "def e2sfca(catchments, write_to, ...)",
       "doc_summary": "Calculates Enhanced Two-Step Floating Catchment Area (E2SFCA).",
       "import_line": "from iguide_methods.ke__3b45070e_x.v_028fc3432827 import e2sfca",
       "element_id": "3b45070e", "requirements": ["geopandas", "pandas", "tqdm"]}


def test_a_method_search_result_becomes_a_citable_document_not_a_raw_row():
    docs = g._evidence_from_artifacts({"tool_results": [
        _row("kb_method_search", {"source": "method_library", "count": 1, "results": [HIT]})]})
    assert len(docs) == 1
    doc = docs[0]
    assert doc["title"] == "e2sfca — callable method"
    assert "import: from iguide_methods.ke__3b45070e_x.v_028fc3432827 import e2sfca" in doc["contents"]
    assert doc["citation_ids"] == ["3b45070e"], "a unit is evidence about the element it came from"


def test_a_contract_result_is_rendered_the_same_way():
    contract = {**HIT, "library_symbol": "e2sfca", "symbol": None,
                "requirements": {"pip": ["geopandas", "pandas", "tqdm"], "inferred": []}}
    docs = g._evidence_from_artifacts({"tool_results": [_row("get_method_contract", contract)]})
    assert len(docs) == 1
    assert "requires: geopandas, pandas, tqdm" in docs[0]["contents"]


def test_a_refused_lookup_is_not_cited_as_a_method():
    """An ambiguous or unknown name comes back with an `error`; citing it would present a refusal
    as a finding."""
    refused = {"symbol": "spatial_join_and_count", "error": "defined by two elements"}
    docs = g._evidence_from_artifacts({"tool_results": [_row("get_method_contract", refused)]})
    assert docs == []


def test_unrelated_tools_are_still_not_harvested():
    """The allowlist's original purpose: a geocoder returning {"results": [...]} is not evidence."""
    docs = g._evidence_from_artifacts({"tool_results": [
        _row("geocode_places", {"results": [{"title": "Chicago"}]})]})
    assert docs == []
