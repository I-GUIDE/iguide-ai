"""Shared extractor interfaces and result types.

Every extractor (notebook, code, dataset, publication) implements the
``Extractor`` protocol and returns an ``ExtractionResult``. The ingest harness
merges results into a ``manifest.UnifiedManifest`` and fans out to the emitters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

# Emit targets an asset may be routed to.
EMIT_OPENSEARCH = "opensearch"
EMIT_MCP = "mcp"
EMIT_SKILL = "skill"
# Callable units emitted as an importable method library (see extractors/analysis/slices.py).
EMIT_LIBRARY = "library"
VALID_TARGETS = (EMIT_OPENSEARCH, EMIT_MCP, EMIT_SKILL, EMIT_LIBRARY)

# Asset kinds produced by the four extractors.
KIND_NOTEBOOK_BLOCK = "notebook_block"
KIND_CODE_BLOCK = "code_block"          # function/class API surface
KIND_DATASET = "dataset"
KIND_PUBLICATION = "publication"        # method-spec / provenance
KIND_METHOD_UNIT = "method_unit"        # ONE callable function + its contract


@dataclass
class AssetRecord:
    """One extracted, addressable asset (a block, a dataset, a method-spec).

    ``block`` / ``spatial`` / ``runnable`` are kind-specific sub-payloads; see
    EXTRACTOR_DESIGN.md §1 for the field contract. ``runnable`` is present only
    when the unit passed the validated-runnable gate (notebook workflow / code
    entry point) and was promoted to an MCP manifest.
    """

    asset_id: str
    kind: str
    resource_type: str                       # OpenSearch resource-type (NotebookBlock, CodeAsset, Dataset, PublicationMethodSpec)
    doc_id: str
    emit_targets: List[str] = field(default_factory=lambda: [EMIT_OPENSEARCH])
    source_rel_path: str = ""
    title: str = ""
    contents: str = ""                       # searchable text (carries the [runnable: ...] pointer when promoted)
    block: Optional[Dict[str, Any]] = None   # {code, markdown_context, constructs, resolved_tools, file_io, imports}
    spatial: Optional[Dict[str, Any]] = None # {crs, bounds, resolution, schema, spatial-bounding-box-geojson, ...}
    runnable: Optional[Dict[str, Any]] = None  # {workflow_id, mode, entrypoint, entrypoint_parameters, source_path, manifest_path, runnable_tool}
    # A serialized contracts.UnitContract. Mirrors how block/runnable/spatial already ride
    # along, so opensearch_emitter needs no structural change to carry it.
    unit: Optional[Dict[str, Any]] = None
    # Emitted slice source for EMIT_LIBRARY units. Deliberately a separate field rather than
    # part of `unit`: `unit` is mirrored into the OpenSearch document, and putting a whole
    # module's source there would bloat every index doc for no retrieval benefit.
    slice_source: str = ""
    extracted: Dict[str, Any] = field(default_factory=dict)  # additive metadata mirrored into the OpenSearch `extracted` object
    source_fields: Dict[str, Any] = field(default_factory=dict)  # platform form fields inherited into _source (authors, tags, contributor, abstract, ...)


@dataclass
class ProvenanceEdge:
    """A cross-link between assets (Neo4j edge + OpenSearch provenance entry)."""

    src: str                                 # source doc_id / asset_id
    # INCLUDES | DEFINES | IMPLEMENTED_BY | USES | HAS_WORKFLOW | DESCRIBES_METHOD | CITES
    #
    # All but the last two are parent-to-child WITHIN one element. Measured over the 174 cached
    # corpus notebooks: 4,071 of 4,212 emitted edges satisfy `dst.startswith(src + "::")`, and
    # `cross_element` is 0 — so this vocabulary described a forest of stars, not a graph.
    # CITES (and the storage-host form of USES) are the exception: their dst is another
    # element's platform id, which is not derivable from src by any rule.
    rel: str
    dst: str                                 # target doc_id / workflow_id / platform element id
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillSpec:
    """The overall pipeline distilled into a SKILL.md bundle (one per notebook)."""

    name: str                                # slug, ^[a-z0-9][a-z0-9-]{0,63}$
    description: str
    allowed_tools: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    ordered_steps: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """What a single extractor returns for one input (file or repo)."""

    assets: List[AssetRecord] = field(default_factory=list)
    edges: List[ProvenanceEdge] = field(default_factory=list)
    skill: Optional[SkillSpec] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class ExtractContext:
    """Shared context threaded into every extractor call."""

    element_id: str = ""                     # platform-assigned knowledge-element id (the doc_id anchor)
    element_type: str = ""                   # notebook | code | dataset | publication
    source_url: str = ""                     # repo URL (github path) or file source
    repo_id: str = ""                        # fallback anchor when element_id is absent
    commit_sha: str = ""
    fields: Dict[str, Any] = field(default_factory=dict)  # platform form metadata (title, authors, tags, contributor, ...)
    targets: Sequence[str] = VALID_TARGETS
    reingest: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def anchor(self) -> str:
        """The parent id all derived doc_ids hang off — the platform element id."""
        return self.element_id or self.repo_id


@runtime_checkable
class Extractor(Protocol):
    """Protocol implemented by every extractor."""

    name: str

    def extract(self, path: str, *, ctx: ExtractContext) -> ExtractionResult:
        """Extract assets from ``path`` (a file, dir, or repo checkout)."""
        ...


__all__ = [
    "EMIT_OPENSEARCH", "EMIT_MCP", "EMIT_SKILL", "VALID_TARGETS",
    "KIND_NOTEBOOK_BLOCK", "KIND_CODE_BLOCK", "KIND_DATASET", "KIND_PUBLICATION",
    "AssetRecord", "ProvenanceEdge", "SkillSpec", "ExtractionResult",
    "ExtractContext", "Extractor",
]
