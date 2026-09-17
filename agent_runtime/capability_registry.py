"""What each peer can actually do, in one place, so the supervisor's prompt cannot drift from it.

The supervisor chooses a capability — search, analyze, code, done — from a description of what
those peers can do. That description used to be a hand-written paragraph with nothing deriving
it from the peers' real toolsets, and it drifted silently. Three toolsets were bound to a peer
without ever being mentioned to the supervisor: terrain, administrative boundaries, and
geocoding. The cost was not cosmetic. Asked for a DEM, the supervisor searched the knowledge
base for "digital elevation model", because as far as it had been told, `analyze` did overlays
and embeddings and nothing else. That was a correct decision from a stale description.

So the inventory lives here and the prompt is generated from it. What stays hand-written in the
prompt is the REASONING guidance — when to stop, how to read the evidence summary, that model
names are arguments rather than datasets — because that is judgement, not inventory, and
generating it would lose the nuance that makes it useful.

Keeping this honest is one test: ``rag_pipeline/tests/test_supervisor_knows_its_peers.py``
asserts that the factories named here are exactly the factories the peer builders call. Bind a
toolset without describing it and that test fails with its name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Toolset:
    """One bound toolset, and the clause that lets the supervisor route work to it."""

    factory: str
    #: A clause, not a sentence — these are joined into a list the model reads at speed.
    summary: str


# Both peers bind nearly the same spatial toolkit; ``peers`` records which ones actually get it,
# so a capability offered by only one is never described as if both had it.
_SHARED: Tuple[Toolset, ...] = (
    Toolset("make_langchain_geocode_tools",
            "geocoding a place name or address to coordinates"),
    Toolset("make_admin_boundary_tools",
            "fetching an administrative boundary (city, county, state, census tract) as real "
            "geometry"),
    Toolset("make_terrain_tools",
            "elevation and terrain: a DEM for a bounding box or shape, slope and aspect, zonal "
            "statistics over a raster, inundation at a water level"),
    Toolset("make_overlay_tools",
            "overlay, buffer, clip, dissolve and intersection"),
    Toolset("make_aggregate_tools",
            "aggregating and summarising features by area or attribute"),
    Toolset("make_temporal_tools",
            "temporal analysis and change over time"),
    Toolset("make_spatial_stats_tools",
            "spatial statistics, autocorrelation and clustering"),
    Toolset("make_langchain_geo_tools",
            "vector inspection, plotting, reprojection and GeoJSON handling"),
    Toolset("make_langchain_qgis_tools",
            "QGIS/PyQGIS processing algorithms"),
    Toolset("make_rs_embed_tools",
            "remote-sensing foundation-model embeddings for a map region"),
    Toolset("make_rs_embed_zonal_tools",
            "segmenting a region into look-alike zones from those embeddings"),
    Toolset("make_langchain_granular_tools",
            "retrieving datasets, publications and notebooks"),
    Toolset("make_conversation_file_tools",
            "listing the files this conversation has produced"),
    Toolset("make_code_execution_tools",
            "running code in a sandbox"),
)

_ANALYZE_ONLY: Tuple[Toolset, ...] = (
    Toolset("make_geo_analysis_tools", "general GIS and geospatial analysis"),
    Toolset("make_langchain_file_tools", "reading and writing uploaded files"),
    Toolset("make_langchain_mcp_tools", "external MCP tools, including a live QGIS instance"),
)

_CODE_ONLY: Tuple[Toolset, ...] = (
    Toolset("make_skill_tools", "packaged skills and saved workflows"),
)

CAPABILITIES: Dict[str, Tuple[Toolset, ...]] = {
    "analyze": _SHARED + _ANALYZE_ONLY,
    "code": _SHARED + _CODE_ONLY,
}


def factories() -> set:
    """Every factory this registry claims a peer binds. Compared against reality by a test."""
    return {t.factory for caps in CAPABILITIES.values() for t in caps}


def describe(capability: str) -> str:
    """The inventory clause list for a capability, deduplicated and in declared order."""
    seen, out = set(), []
    for tool in CAPABILITIES.get(capability, ()):  # declared order is the reading order
        if tool.summary in seen:
            continue
        seen.add(tool.summary)
        out.append(tool.summary)
    return "; ".join(out)
