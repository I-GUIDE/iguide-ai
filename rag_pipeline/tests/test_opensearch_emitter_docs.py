"""What an index document must carry, and what it must never be able to lose.

The emitter is where extraction meets retrieval, and every defect it has looks like success from
both sides. Measured on the live cluster before these tests existed:

* **349 indexed ``method_unit`` documents carried no contract.** ``_build_source`` mirrored
  ``block``/``runnable``/``spatial`` into ``extracted`` and never learned about ``unit``, so
  signature, parameter types, declared units, CRS expectations, invariants, pip requirements,
  module and slice_sha were all dropped. Nothing crashed: ``kb_method_search`` reads the on-disk
  registry and kept working, so the loss showed up only on the other path — a unit surfaced by
  ``agent_kb_search`` told the agent a function exists while withholding everything needed to
  call it.
* **Platform form fields were applied AFTER the canonical keys**, so a submission field named
  ``contents`` or ``doc_id`` overwrote the document's identity.
* **Dataset and publication documents carried no ``extracted.parent_doc_id``** — the one field
  orphan reconciliation keys on.
* **``block``/``runnable``/``spatial`` were written as explicit nulls** on all 349.

Two of these are silent-success bugs of the kind this codebase keeps producing: the reconciler
reports ``deleted_orphans: 0`` when it cannot see the documents at all, and a term query against a
dynamically-mapped ``text`` field returns nothing in a way indistinguishable from "nothing
matched". So the mapping is asserted here too.
"""

from __future__ import annotations

import pytest

from extractors.base import (AssetRecord, EMIT_OPENSEARCH, KIND_DATASET, KIND_METHOD_UNIT,
                             KIND_NOTEBOOK_BLOCK, ProvenanceEdge)
from extractors.emitters.opensearch_emitter import (RESERVED_KEYS, build_docs, import_line_for,
                                                    index_mapping, mapping_drift)
from extractors.manifest import UnifiedManifest


CONTRACT = {
    "qualified_name": "calculate_buffers",
    "unit_kind": "function",
    "signature": "def calculate_buffers(gdf: gpd.GeoDataFrame, buffer: Number) -> gpd.GeoDataFrame",
    "params": [{"name": "gdf", "annotation": "gpd.GeoDataFrame", "inferred_type": "geodataframe",
                "declared_unit": "metres", "crs_expectation": "projected", "required": True,
                "evidence": "annotation 'gpd.GeoDataFrame' | body performs a metric operation"}],
    "returns": "gpd.GeoDataFrame",
    "docstring": "Makes a deepcopy with geography replaced with buffers.\n\nArgs:\n    gdf: ...",
    "doc_summary": "Makes a deepcopy with geography replaced with buffers.",
    "callability": {"verdict": "callable", "reason": "", "analyzer_version": 1},
    "invariants": [{"check": "projected_crs", "target": "gdf", "args": {"unit": "metres"}}],
    "requirements": {"pip": ["geopandas"], "inferred": []},
    "library_module": "iguide_methods.ke_b1fa548b_spastc.v_c43f0727bb2a",
    "library_symbol": "calculate_buffers",
    "slice_sha": "c43f0727bb2a",
}


def _manifest(*assets, edges=()):
    man = UnifiedManifest(element_id="b1fa548b", element_type="notebook")
    man.assets.extend(assets)
    man.provenance_edges.extend(edges)
    return man


def _unit_asset(**over):
    kw = dict(asset_id="b1fa548b::unit::calculate_buffers", kind=KIND_METHOD_UNIT,
              resource_type="MethodUnit", doc_id="b1fa548b::unit::calculate_buffers",
              emit_targets=[EMIT_OPENSEARCH], title="calculate_buffers — SPASTC",
              contents="def calculate_buffers(gdf, buffer)", unit=dict(CONTRACT),
              extracted={"parent_doc_id": "b1fa548b", "callable": True})
    kw.update(over)
    return AssetRecord(**kw)


def _only(manifest):
    docs = build_docs(manifest)
    assert len(docs) == 1, f"expected one doc, got {len(docs)}"
    return docs[0][2]


# ------------------------------------------------------- the contract reaches the index

def test_a_unit_document_carries_the_whole_contract():
    unit = _only(_manifest(_unit_asset()))["extracted"]["unit"]
    for field in ("signature", "params", "returns", "invariants", "requirements",
                  "library_module", "library_symbol", "slice_sha", "unit_kind", "callability"):
        assert unit.get(field), f"{field} missing from the indexed unit payload"


def test_a_unit_document_carries_a_ready_import_line():
    """The whole promise of the library is that a hit is immediately usable. Without this the
    agent has a symbol name and has to guess the module — and it did guess, wrongly."""
    unit = _only(_manifest(_unit_asset()))["extracted"]["unit"]
    assert unit["import_line"] == (
        "from iguide_methods.ke_b1fa548b_spastc.v_c43f0727bb2a import calculate_buffers")


def test_the_import_line_is_pinned_to_the_slice_not_the_element_package():
    """A re-ingest mints a new ``v_<sha>`` module and leaves the old one importable. An artifact
    that recorded the element package would silently re-run against different code."""
    unit = _only(_manifest(_unit_asset()))["extracted"]["unit"]
    assert unit["slice_sha"] in unit["import_line"]


def test_a_unit_with_no_library_module_gets_no_import_line():
    """A unit indexed but not emitted to a library has nothing to import. Offering a constructed
    line would be worse than offering none: it would fail at runtime, inside the sandbox."""
    contract = {k: v for k, v in CONTRACT.items() if k != "library_module"}
    unit = _only(_manifest(_unit_asset(unit=contract)))["extracted"]["unit"]
    assert "import_line" not in unit
    assert import_line_for(contract) == ""


def test_the_unbounded_docstring_is_left_out_but_its_summary_is_not():
    unit = _only(_manifest(_unit_asset()))["extracted"]["unit"]
    assert "docstring" not in unit
    assert unit["doc_summary"] == "Makes a deepcopy with geography replaced with buffers."


def test_per_parameter_evidence_survives():
    """The declared unit and CRS expectation are INFERRED. Keeping the evidence is what lets a
    reader check the inference rather than trust it."""
    param = _only(_manifest(_unit_asset()))["extracted"]["unit"]["params"][0]
    assert "metric operation" in param["evidence"]


# ------------------------------------------------------- the embedded text

def test_a_unit_embeds_its_types_units_and_dependencies():
    """A unit asset used to fall through to raw ``contents``, so the signature was embedded by
    accident and the parameter types, declared units and deps were embedded nowhere. "buffer a
    GeoDataFrame in metres" cannot match words that are not in the embedded text."""
    text = _only(_manifest(_unit_asset()))["extracted"]["embed_text"]
    for token in ("calculate_buffers", "geodataframe", "metres", "projected", "geopandas"):
        assert token in text.lower(), f"{token!r} absent from the embedded text"


def test_a_block_still_embeds_prose_not_code():
    asset = AssetRecord(
        asset_id="b1fa548b::block::3", kind=KIND_NOTEBOOK_BLOCK, resource_type="NotebookBlock",
        doc_id="b1fa548b::block::3", emit_targets=[EMIT_OPENSEARCH], title="SPASTC — cell 3",
        contents="gdf = gpd.read_file('x.shp')",
        block={"markdown_context": "Load the county boundaries", "resolved_tools": ["geopandas"],
               "imports": ["geopandas"], "code": "gdf = gpd.read_file('x.shp')"})
    text = _only(_manifest(asset))["extracted"]["embed_text"]
    assert "county boundaries" in text
    assert "read_file" not in text


# ------------------------------------------------------- identity cannot be overwritten

@pytest.mark.parametrize("reserved", ["doc_id", "contents", "title", "resource-type",
                                      "element_type", "extracted"])
def test_a_platform_field_cannot_overwrite_a_canonical_key(reserved):
    """Form fields were applied after the canonical keys. A doc_id that disagrees with its own
    _id is unreachable by every reader in this module, including the reconciler."""
    src = _only(_manifest(_unit_asset(source_fields={reserved: "HIJACKED", "tags": ["ok"]})))
    assert src[reserved] != "HIJACKED"
    assert src["doc_id"] == "b1fa548b::unit::calculate_buffers"
    assert src["tags"] == ["ok"], "a non-reserved form field must still be inherited"


def test_a_dropped_form_field_is_recorded_rather_than_vanishing():
    """Silently discarding submitted data is its own failure. Naming it in the document is how
    someone finds out their `contents` field went nowhere."""
    src = _only(_manifest(_unit_asset(source_fields={"contents": "x", "tags": []})))
    assert src["extracted"]["source_fields_dropped"] == ["contents"]


def test_a_clean_asset_records_no_dropped_fields():
    assert "source_fields_dropped" not in _only(_manifest(_unit_asset()))["extracted"]


def test_every_reserved_key_is_one_the_emitter_actually_writes():
    src = _only(_manifest(_unit_asset()))
    for key in RESERVED_KEYS:
        if key == "contents-embedding":
            continue          # written by the embed pass, not by _build_source
        assert key in src, f"{key} is reserved but never written — the guard is stale"


# ------------------------------------------------------- reconciliation can find the doc

def test_a_dataset_document_carries_its_own_parent_doc_id():
    """Dataset and publication documents ARE their element, so no extractor set parent_doc_id.
    Reconciliation keys on it, so those documents could never appear in the diff that decides
    what to delete — and an orphan set that is empty because the query found nothing reads
    exactly like an index with no orphans."""
    asset = AssetRecord(asset_id="265e6957", kind=KIND_DATASET, resource_type="Dataset",
                        doc_id="265e6957", emit_targets=[EMIT_OPENSEARCH], title="Crime export",
                        contents="format=CSV", extracted={"parent_type": "Dataset"})
    assert _only(_manifest(asset))["extracted"]["parent_doc_id"] == "265e6957"


def test_a_derived_document_keeps_the_parent_the_extractor_declared():
    src = _only(_manifest(_unit_asset(extracted={"parent_doc_id": "explicit", "callable": True})))
    assert src["extracted"]["parent_doc_id"] == "explicit"


def test_a_derived_document_with_no_declared_parent_falls_back_to_the_id_rule():
    src = _only(_manifest(_unit_asset(extracted={"callable": True})))
    assert src["extracted"]["parent_doc_id"] == "b1fa548b"


# ------------------------------------------------------- no nulls, no duplicates

def test_absent_subpayloads_are_omitted_not_written_as_null():
    extracted = _only(_manifest(_unit_asset()))["extracted"]
    for name in ("block", "runnable", "spatial"):
        assert name not in extracted, f"{name} written as an explicit null on every unit doc"


def test_the_bounding_box_is_written_once_at_the_top_level():
    """The envelope needs the top-level geo_shape mapping to be searchable. The copy inside
    `extracted` got a second, dynamically-mapped home and no second reader."""
    envelope = {"type": "envelope", "coordinates": [[-87.9, 42.0], [-87.5, 41.6]]}
    asset = AssetRecord(asset_id="d1", kind=KIND_DATASET, resource_type="Dataset", doc_id="d1",
                        emit_targets=[EMIT_OPENSEARCH], title="d",
                        spatial={"crs": "EPSG:4326", "spatial-bounding-box-geojson": envelope})
    src = _only(_manifest(asset))
    assert src["spatial-bounding-box-geojson"] == envelope
    assert "spatial-bounding-box-geojson" not in src["extracted"]["spatial"]
    assert src["extracted"]["spatial"]["crs"] == "EPSG:4326"


def test_provenance_edges_touching_the_doc_are_attached():
    edge = ProvenanceEdge(src="b1fa548b", rel="DEFINES",
                          dst="b1fa548b::unit::calculate_buffers", detail={"verdict": "callable"})
    other = ProvenanceEdge(src="zzz", rel="DEFINES", dst="zzz::unit::other")
    src = _only(_manifest(_unit_asset(), edges=[edge, other]))
    rels = [e["rel"] for e in src["extracted"]["provenance"]]
    assert rels == ["DEFINES"]


# ------------------------------------------------------- the mapping backs the queries

@pytest.mark.parametrize("path", ["extracted.parent_doc_id", "extracted.kind",
                                  "extracted.unit.library_symbol", "extracted.unit.slice_sha",
                                  "extracted.unit.import_line", "extracted.unit_name",
                                  "extracted.status"])
def test_every_term_queried_field_is_mapped_keyword(path):
    """Left dynamic, a string becomes text + .keyword and a term query matches only when the
    value survives the standard analyzer whole. The corpus's 8-char hex element ids do; a full
    platform UUID would be split on its hyphens and match nothing."""
    node = index_mapping()["mappings"]["properties"]
    for part in path.split("."):
        node = node["properties"][part] if "properties" in node else node[part]
    assert node["type"] == "keyword", f"{path} is {node['type']}, must be keyword"


def test_the_filter_kb_method_search_uses_is_a_boolean():
    props = index_mapping()["mappings"]["properties"]["extracted"]["properties"]
    assert props["callable"]["type"] == "boolean"


def test_the_contract_object_stays_dynamic_below_the_declared_paths():
    """The analyzer grows new contract fields. A strict mapping would reject the document instead
    of storing them, which trades a silent gap for a silent loss."""
    unit = index_mapping()["mappings"]["properties"]["extracted"]["properties"]["unit"]
    assert unit.get("dynamic") not in ("strict", False)
    assert "params" not in unit["properties"], "params is nested and free-form; leave it dynamic"


class _FakeIndices:
    def __init__(self, live):
        self._live = live

    def exists(self, index):
        return index in self._live

    def get_mapping(self, index):
        return {index: {"mappings": {"properties": self._live[index]}}}


class _FakeClient:
    def __init__(self, live):
        self.indices = _FakeIndices(live)


def test_drift_is_reported_when_a_live_field_has_the_wrong_type():
    """This is what actually happened: four indices were created before the mapping existed, and
    `ensure_index` returns early when the index is present — so a schema change lands in the code
    and never reaches the cluster."""
    live = {"iguide_agent_method_units": {
        "doc_id": {"type": "keyword"},
        "extracted": {"properties": {"parent_doc_id": {
            "type": "text", "fields": {"keyword": {"type": "keyword"}}}}}}}
    report = mapping_drift(_FakeClient(live), "iguide_agent_method_units")
    assert report["drift"]["extracted.parent_doc_id"] == {"want": "keyword", "live": "text"}


def test_a_matching_index_reports_no_drift():
    from extractors.emitters.opensearch_emitter import _flatten_mapping

    want = index_mapping()["mappings"]["properties"]
    report = mapping_drift(_FakeClient({"i": want}), "i")
    assert report["drift"] == {}
    assert _flatten_mapping(want)["extracted.unit.slice_sha"] == "keyword"


def test_a_missing_index_is_not_drift():
    assert mapping_drift(_FakeClient({}), "absent") == {
        "index": "absent", "exists": False, "drift": {}}
