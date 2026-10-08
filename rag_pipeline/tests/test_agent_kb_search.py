"""What ``agent_kb_search`` hands the agent — the last hop of the contract chain.

The contract that makes a library unit usable had to survive four transfers, and it was being
dropped at three of them:

1. the analyzer computes it — fine;
2. ``opensearch_emitter._build_source`` mirrored ``block``/``runnable``/``spatial`` and never
   learned about ``unit``, so 349 indexed documents held no contract (fixed in M8.7);
3. ``_fan_out`` ran the index emitter before the library emitter that assigns ``library_module``,
   so no document could carry an import line (fixed in M8.8);
4. **``normalize_hit``** — this module — built ``title``/``contents``/``runnable_tool`` and nothing
   about the unit, so even a fully populated document arrived at the agent as a name.

Any one of those left intact makes the other three pointless, and none of them raises: the agent
just sees a function it cannot call and re-implements it instead. That failure is invisible in
logs and shows up only as a worse answer.

This file was committed empty in ``de8c0e4`` — an empty test file is worse than a missing one,
because it reads as coverage.
"""

from __future__ import annotations

import pytest

from rag_pipeline.search.agent_kb import (_method_payload, build_keyword_query, group_by_parent,
                                          normalize_hit, normalize_hits)

CONTRACT = {
    "qualified_name": "calculate_buffers",
    "library_symbol": "calculate_buffers",
    "library_module": "iguide_methods.ke_b1fa548b_spastc.v_c43f0727bb2a",
    "slice_sha": "c43f0727bb2a",
    "unit_kind": "function",
    "signature": "def calculate_buffers(gdf: gpd.GeoDataFrame, buffer: Number) -> gpd.GeoDataFrame",
    "doc_summary": "Makes a deepcopy with geography replaced with buffers.",
    "returns": "gpd.GeoDataFrame",
    "params": [
        {"name": "gdf", "annotation": "gpd.GeoDataFrame", "inferred_type": "geodataframe",
         "declared_unit": "metres", "crs_expectation": "projected", "required": True,
         "evidence": "annotation 'gpd.GeoDataFrame' | body performs a metric operation",
         "schema": [], "default": ""},
        {"name": "buffer", "annotation": "Number", "inferred_type": "number", "required": True,
         "declared_unit": "", "crs_expectation": "", "evidence": "parameter name 'buffer'"},
    ],
    "invariants": [{"check": "projected_crs", "target": "gdf", "args": {"unit": "metres"}},
                   {"check": "reject_all_nan", "target": "gdf"}],
    "requirements": {"pip": ["geopandas"], "inferred": []},
    "callability": {"verdict": "callable", "reason": ""},
    "import_line": "from iguide_methods.ke_b1fa548b_spastc.v_c43f0727bb2a import calculate_buffers",
}


def _hit(extracted, *, doc_id="b1fa548b::unit::calculate_buffers", rtype="MethodUnit"):
    return {"_id": doc_id, "_index": "iguide_agent_method_units", "_score": 3.5,
            "_source": {"doc_id": doc_id, "title": "calculate_buffers — SPASTC",
                        "contents": "def calculate_buffers(gdf, buffer)",
                        "resource-type": rtype, "extracted": extracted}}


# ---------------------------------------------------------------- the method payload

def test_a_unit_hit_carries_what_the_agent_needs_to_call_it():
    method = normalize_hit(_hit({"kind": "method_unit", "unit": CONTRACT}), "keyword")["method"]
    assert method["signature"].startswith("def calculate_buffers")
    assert method["import_line"].endswith("import calculate_buffers")
    assert method["requirements"] == ["geopandas"]
    assert method["invariants"] == ["projected_crs", "reject_all_nan"]
    assert method["callable"] is True


def test_the_declared_unit_and_crs_expectation_reach_the_agent():
    """These two facts are the difference between a correct answer and a 21.5-km buffer reported
    as 25 km. The agent cannot honour a contract it was not shown."""
    method = normalize_hit(_hit({"unit": CONTRACT}), "keyword")["method"]
    params = method["params_with_preconditions"]
    gdf = next(p for p in params if p["name"] == "gdf")
    assert gdf["declared_unit"] == "metres"
    assert gdf["crs_expectation"] == "projected"
    # Only the constrained parameters are listed, so the count of the rest has to survive
    # separately — otherwise a three-parameter method looks like a one-parameter method.
    assert method["param_count"] == len(CONTRACT["params"])


def test_the_import_line_is_pinned_to_the_slice():
    method = normalize_hit(_hit({"unit": CONTRACT}), "keyword")["method"]
    assert method["slice_sha"] in method["import_line"]


def test_per_parameter_inference_evidence_is_left_for_the_contract_tool():
    """This payload goes into a token-limited evidence view. Full evidence belongs to
    ``get_method_contract``, which is called for one promising hit rather than for all eight."""
    params = normalize_hit(_hit({"unit": CONTRACT}),
                           "keyword")["method"]["params_with_preconditions"]
    assert all("evidence" not in p for p in params)
    assert all("schema" not in p for p in params)
    # Stricter now, and this is the point: `annotation` and `inferred_type` restate what
    # `signature` already carries, so shipping them for every parameter of every hit was ~1,200
    # tokens per search spent to say a thing twice. A parameter appears here only when it
    # carries a precondition the signature CANNOT express.
    assert all("annotation" not in p for p in params)
    assert all("inferred_type" not in p for p in params)
    assert all(p.get("declared_unit") or p.get("crs_expectation") for p in params)


def test_empty_contract_fields_are_dropped_rather_than_sent_as_blanks():
    thin = {"library_symbol": "f", "signature": "def f(x)", "params": [], "invariants": [],
            "requirements": {}, "doc_summary": ""}
    method = normalize_hit(_hit({"unit": thin}), "keyword")["method"]
    assert set(method) == {"symbol", "signature"}, method


def test_an_unanalyzed_unit_is_not_reported_as_not_callable():
    """Absent verdict and negative verdict are different facts. Flattening callability to a bool
    made "nothing analyzed this" indistinguishable from "this cannot be imported", and the first
    reported as False would push the agent off a usable unit — the same fail-versus-
    cannot-determine distinction the invariant gate makes."""
    thin = {"library_symbol": "f", "signature": "def f(x)"}
    assert "callable" not in normalize_hit(_hit({"unit": thin}), "keyword")["method"]


def test_a_unit_not_shipped_to_a_library_advertises_no_import():
    contract = {k: v for k, v in CONTRACT.items() if k != "import_line"}
    method = normalize_hit(_hit({"unit": contract}), "keyword")["method"]
    assert "import_line" not in method
    assert method["signature"]


def test_a_not_independently_callable_unit_says_so_and_says_why():
    """``needs_globals`` reported as a bare False looks like a broken unit. The reason — "reads the
    module-level global PARAMS" — is what tells the agent to pass it as an argument instead."""
    contract = {**CONTRACT, "callability": {"verdict": "needs_globals",
                                            "reason": "reads the module-level global PARAMS"}}
    method = normalize_hit(_hit({"unit": contract}), "keyword")["method"]
    assert method["callable"] is False
    assert method["not_callable"] == "needs_globals"
    assert "PARAMS" in method["not_callable_reason"]


def test_a_callable_unit_carries_no_negative_fields():
    method = normalize_hit(_hit({"unit": CONTRACT}), "keyword")["method"]
    assert method["callable"] is True
    assert "not_callable" not in method and "not_callable_reason" not in method


# ---------------------------------------------------------------- everything else is unchanged

def test_a_notebook_block_carries_no_method_key_at_all():
    """An empty ``method: {}`` on every block would cost tokens on every hit and mean nothing."""
    hit = _hit({"kind": "notebook_block",
                "block": {"resolved_tools": ["geopandas"], "markdown_context": "Load"}},
               doc_id="b1fa548b::block::6", rtype="NotebookBlock")
    row = normalize_hit(hit, "keyword")
    assert "method" not in row
    assert row["resolved_tools"] == ["geopandas"]


@pytest.mark.parametrize("extracted", [{}, {"unit": None}, {"unit": {}}, {"unit": "nonsense"}])
def test_a_malformed_or_absent_unit_does_not_raise(extracted):
    assert _method_payload(extracted) is None


def test_the_existing_fields_still_arrive():
    row = normalize_hit(_hit({"unit": CONTRACT}), "semantic")
    assert row["doc_id"] == "b1fa548b::unit::calculate_buffers"
    assert row["parent_doc_id"] == "b1fa548b"
    assert row["resource_type"] == "MethodUnit"
    assert row["matched"] == "semantic"
    assert row["score"] == 3.5


def test_a_parent_is_derived_when_the_document_does_not_declare_one():
    row = normalize_hit(_hit({"unit": CONTRACT}), "keyword")
    assert row["parent_doc_id"] == "b1fa548b"


# ---------------------------------------------------------------- merge behaviour

def test_a_document_found_both_ways_is_reported_once_and_marked():
    hit = _hit({"unit": CONTRACT})
    merged = normalize_hits([hit], [hit], size=8)
    assert len(merged) == 1
    assert merged[0]["matched"] == "keyword+semantic"
    assert merged[0]["method"]["import_line"] == CONTRACT["import_line"]


def test_merging_preserves_the_method_payload_of_a_semantic_only_hit():
    other = _hit({"unit": CONTRACT}, doc_id="x::unit::f")
    merged = normalize_hits([], [other], size=8)
    assert merged[0]["method"]["signature"]


def test_grouping_by_parent_still_works_for_units_and_blocks():
    unit = _hit({"unit": CONTRACT})
    block = _hit({"kind": "notebook_block"}, doc_id="b1fa548b::block::6", rtype="NotebookBlock")
    rows = normalize_hits([unit, block], [], size=8)
    assert group_by_parent(rows) == {
        "b1fa548b": ["b1fa548b::unit::calculate_buffers", "b1fa548b::block::6"]}


def test_the_keyword_query_searches_the_embed_text_the_emitter_writes():
    """``extracted.embed_text`` is where the emitter puts a unit's inferred types and declared
    units. Querying only title and contents would leave that unreachable."""
    fields = build_keyword_query("buffer in metres", 8)["query"]["multi_match"]["fields"]
    assert "extracted.embed_text" in fields
    assert "title^2" in fields
