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
asserts, per peer, that the factories named here for a peer are exactly the factories that peer
binds, wherever the binding is made: its own builder in ``graph.py``, or ``build_agent_executor``
and ``collect_tools``, which the test reaches by running the peer's real assembly code. Bind a
toolset without describing it and that test fails with its name and the place it is bound.

The ``code`` capability is the one whose peer is swapped per request (``code_peer``, falling back
to ``AGENT_CODE_PEER``), so its description is per BACKEND — ``describe_code_peer``. The LangChain
peer binds ``CAPABILITIES["code"]``; a CLI peer binds none of it and is described as itself. The
decider's ``code`` line stayed hand-written when the ``analyze`` line was generated, and it
drifted both ways: a code-only entry here never reached it, and it promised every backend the
LangChain toolkit, plus "saved workflows", which the code peer cannot run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Toolset:
    """One bound toolset, and the clause that lets the supervisor route work to it."""

    factory: str
    #: A clause, not a sentence — these are joined into a list the model reads at speed.
    summary: str
    #: Appended to `summary` only while the extraction bundle is on (agent_runtime/extraction_flag.py).
    extraction_summary: str = ""
    #: The whole toolset is part of the extraction bundle: not described while it is off, because
    #: its factory then binds nothing and the supervisor must not route work to it.
    requires_extraction: bool = False
    #: The factory binds nothing unless skill discovery finds a skill (``make_skill_tools`` returns
    #: ``[]`` for an empty registry), so the clause is true only then. The deployed image shipped
    #: no skill roots at all while the decider was told the code peer had skills.
    requires_skills: bool = False
    #: The factory binds nothing unless this environment variable is on, so the clause is true
    #: only then. A gated capability the decider is told about while it is off sends work to a
    #: tool that is not there.
    requires_flag: str = ""


# The knowledge-base tools the code-WRITING peers hold, defined ONCE. The peer binding
# (`supervisor/graph.py:_CODE_PEER_KB_TOOLS`) and the evidence allowlist (`_RETRIEVAL_TOOLS`) both
# read this tuple. The 2026-10-01 integration found the same names hardcoded in three places that
# had drifted apart: the code peer's binding dropped the method tools, and the evidence allowlist
# never had them, so a method the peer found could not be cited.
KB_CODE_PEER_TOOLS: Tuple[str, ...] = (
    "agent_kb_search", "get_kb_block", "kb_method_search", "get_method_contract",
)

# What each EVIDENCE CONSUMER can do with an extracted method, declared rather than inferred from a
# peer's name, so the evidence a consumer reads describes only what it can act on
# (evidence_subgraph._render_extracted and _doc_block). A consumer that gains a capability gets
# the view that uses it by changing its row here; rag_pipeline/tests/test_evidence_consumers.py
# holds each peer's row to the tools its builder actually binds.
RUN_LIBRARY = "run_library"          # imports `iguide_methods`: execute_code, library mounted
STAGE_INPUTS = "stage_inputs"        # calls stage_element / stage_url
SEARCH_METHODS = "search_methods"    # calls kb_method_search / get_method_contract
OFFER_LIBRARY = "offer_library"      # tells a HUMAN which library methods this agent can run

EVIDENCE_CONSUMERS: Dict[str, FrozenSet[str]] = {
    # The answerer and the evidence subgraph write for a HUMAN, who can neither import the
    # library nor call the agent's tools. So methods arrive as references this agent can run,
    # not as import lines to paste. The map UI, 2026-10-01: an answer told the user to write
    # `from iguide_methods ...`, which works only inside the agent's sandbox. This is a
    # rendering change, not an evidence cut (B3's question is untouched).
    "answer": frozenset({OFFER_LIBRARY}),
    # default_code_fn: execute_code (network-none sandbox, library mounted), staging, method tools.
    "code_peer": frozenset({RUN_LIBRARY, STAGE_INPUTS, SEARCH_METHODS}),
    # default_analyze_fn: execute_code and the method tools, but NOT the staging tools.
    "analyze_peer": frozenset({RUN_LIBRARY, SEARCH_METHODS}),
    # claude_peer / opencode_peer: none, deliberately. Their container keeps network access and the
    # model credential, a trust tier below the execute_code sandbox, so submitter-authored library
    # code does not run there (extraction review, D1). They hold none of the agent's tools either.
    "cli_peer": frozenset(),
}


def consumer_capabilities(consumer: str = "answer") -> FrozenSet[str]:
    """What *consumer* can do with an extracted method right now: its row, or nothing while the
    extraction bundle is off. An unknown consumer gets nothing: a view that under-describes costs a
    reuse, while one that over-describes costs a failed import."""
    from agent_runtime.extraction_flag import extraction_enabled

    if not extraction_enabled():
        return frozenset()
    return EVIDENCE_CONSUMERS.get(consumer, frozenset())


# Both peers bind nearly the same spatial toolkit; the split below records which ones actually get
# it, so a capability offered by only one is never described as if both had it, and one both have
# is never described as one peer's.
_SHARED: Tuple[Toolset, ...] = (
    Toolset("make_langchain_geocode_tools",
            "geocoding a place name or address to coordinates"),
    Toolset("make_langchain_osm_tools",
            "finding real-world features of a kind inside an area from live OpenStreetMap "
            "(schools, hospitals, parks, rivers, roads) as a file to measure against"),
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
            "retrieving datasets, publications and notebooks",
            extraction_summary=("the extracted method library — callable functions from platform "
                                "elements with their contracts and pinned import lines")),
    Toolset("make_conversation_file_tools",
            "listing the files this conversation has produced"),
    Toolset("make_code_execution_tools",
            "running code in a sandbox"),
    Toolset("make_public_data_tools",
            "downloading public data (Census boundaries and tables, USGS elevation, "
            "OpenStreetMap) from approved sources as files for that code",
            requires_flag="AGENT_PUBLIC_FETCH"),
    # Shared, not code-only: both peers bind the same two loaders over the same skill roots. The
    # code peer calls make_skill_tools itself, and since 6ba1bd3 build_agent_executor adds it for
    # every peer that hands over a preloaded tool list, analyze included. Described as the code
    # peer's alone, it made matching a skill look like a reason to choose code, and with a CLI
    # code peer, which loads none, the prompt mentioned skills nowhere. The clause promises
    # loading, which both can do. Following a skill takes the tools it names, and the rest of
    # each line says which ones that peer has.
    #
    # This clause used to read "packaged skills and saved workflows", and the decider repeated it
    # as a capability. The code peer cannot run a saved workflow: a notebook workflow the
    # extractor packages as a skill runs through an MCP tool (the server's
    # ``run_notebook_workflow``), and MCP tools are bound in the analyze and search peers behind
    # include_mcp_tools, never in the code peer. Running one is that MCP toolset's capability, not
    # this one's.
    Toolset("make_skill_tools",
            "loading packaged skills: step-by-step instructions for particular analyses",
            requires_skills=True),
)

_ANALYZE_ONLY: Tuple[Toolset, ...] = (
    Toolset("make_geo_analysis_tools", "general GIS and geospatial analysis"),
    Toolset("make_langchain_file_tools", "reading and writing uploaded files"),
    Toolset("make_langchain_mcp_tools", "external MCP tools, including a live QGIS instance"),
)

# The one code-only toolset: only default_code_fn binds the staging tools. It reaches the decider
# through describe_code_peer, and only while the extraction bundle is on.
_CODE_ONLY: Tuple[Toolset, ...] = (
    Toolset("make_langchain_staging_tools",
            "staging a platform dataset into the code sandbox (by element id or URL) so extracted "
            "loaders and methods can read it", requires_extraction=True),
)

CAPABILITIES: Dict[str, Tuple[Toolset, ...]] = {
    "analyze": _SHARED + _ANALYZE_ONLY,
    "code": _SHARED + _CODE_ONLY,
}


def factories() -> set:
    """Every factory this registry claims some peer binds. The drift test compares per peer."""
    return {t.factory for caps in CAPABILITIES.values() for t in caps}


def skills_available(skill_roots: Optional[Sequence[str]] = None) -> bool:
    """Whether ``make_skill_tools`` would bind anything: the same discovery over the same roots.

    Never raises. An unreadable root reads as "no skills", which under-describes the peer —
    the cheaper error, since over-describing it sends work to tools that are not there.
    """
    try:
        from agent_runtime.skills import SkillRegistry

        return bool(SkillRegistry.discover(skill_roots))
    except Exception:  # noqa: BLE001 - a prompt must still be produced
        return False


def flag_on(name: str) -> bool:
    """The same reading the gated factories make of their switch."""
    import os

    return (os.getenv(name) or "").strip().lower() in ("1", "true", "yes", "on")


def describe(capability: str, *, skill_roots: Optional[Sequence[str]] = None) -> str:
    """The inventory clause list for a capability, deduplicated and in declared order.

    ``skill_roots`` are the roots the peer's skill tools are built from. A ``requires_skills``
    toolset is described only when discovery there finds at least one skill, and a
    ``requires_flag`` toolset only when its switch is on.
    """
    from agent_runtime.extraction_flag import extraction_enabled

    on = extraction_enabled()
    seen, out = set(), []
    skills: Optional[bool] = None  # discovered once, and only if a toolset needs the answer
    for tool in CAPABILITIES.get(capability, ()):  # declared order is the reading order
        if tool.requires_extraction and not on:
            continue
        if tool.requires_flag and not flag_on(tool.requires_flag):
            continue
        if tool.requires_skills:
            if skills is None:
                skills = skills_available(skill_roots)
            if not skills:
                continue
        clause = (f"{tool.summary}, and {tool.extraction_summary}"
                  if (on and tool.extraction_summary) else tool.summary)
        if clause in seen:
            continue
        seen.add(clause)
        out.append(clause)
    return "; ".join(out)


# --- The code peer is swappable, so "what code can do" depends on which one runs ----------
#
# ``code_peer`` on the request, falling back to ``AGENT_CODE_PEER``, selects the LangChain peer
# (which binds ``CAPABILITIES["code"]``) or an agentic CLI (``agent_runtime/claude_peer.py``,
# ``agent_runtime/opencode_peer.py``) that binds NONE of it. Describing the toolkit while a CLI
# runs tells the supervisor about tools that are not there.

#: Which description row each backend reads, keyed by the backend name the code node resolves.
#: ``code_peer`` and ``cli_peer`` are the names the per-consumer capability table on the
#: extraction work gives these same two peers, so when that table lands its rows are read here
#: by the same key instead of being kept in a parallel structure.
CODE_PEER_CONSUMER: Dict[str, str] = {
    "langchain": "code_peer",
    "claude": "cli_peer",
    "opencode": "cli_peer",
}

#: How the decider names each CLI backend.
CLI_PEER_NAMES: Dict[str, str] = {"claude": "Claude Code", "opencode": "opencode"}

#: What a CLI peer has INSTEAD of the toolkit. Each clause is a fact of the two CLI modules: a
#: fresh container per run that keeps network access (``build_docker_argv``, held by a test),
#: uploads staged into ``/work``, evidence and analysis inlined into the brief and capped
#: (``_build_peer_prompt``; files an earlier step produced are NOT staged), and any GeoJSON it
#: writes turned into a layer (``build_map_layers``).
CLI_PEER: Tuple[str, ...] = (
    "writing, running and debugging its own code, with network access to install packages and "
    "fetch data",
    "reading the conversation's uploaded files, staged into its working directory",
    "working from the evidence and analysis results gathered before it starts, abridged into "
    "its brief as text",
    "putting any GeoJSON file it writes on the map",
)


def is_cli_code_peer(backend: str) -> bool:
    """Whether ``backend`` is a CLI peer: none of the toolkit, and no ``request_capability``."""
    return CODE_PEER_CONSUMER.get(backend) == "cli_peer"


def describe_code_peer(backend: str, *, skill_roots: Optional[Sequence[str]] = None) -> str:
    """The inventory clause list for the code peer ``backend`` names.

    ``backend`` is what ``supervisor.graph._code_peer_backend`` resolved for this request:
    ``langchain``, ``claude`` or ``opencode``. An unrecognised name reads as the LangChain peer,
    because that is what the code node runs for it.
    """
    if is_cli_code_peer(backend):
        return "; ".join(CLI_PEER)
    return describe("code", skill_roots=skill_roots)
