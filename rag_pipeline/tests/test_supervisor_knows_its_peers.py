"""The supervisor's description of its peers must not drift behind the tools they actually bind.

The decider chooses a capability from a hand-written paragraph describing what each peer can do.
Nothing derives that paragraph from the peers' real toolsets, so it drifts — and silently. When
four terrain tools were added and bound to BOTH peers, the paragraph was never updated, so the
supervisor did not know its analysis peer could compute a DEM. Asked for one, it searched the
knowledge base for "digital elevation model" instead. That was not a weak model guessing badly;
it was a correct decision from a stale description.

The inventory is now GENERATED from ``agent_runtime.capability_registry``, so the prompt cannot
fall behind the registry. What can still fall behind is the registry itself, and that is what
these tests hold: the factories the registry claims must be exactly the factories the peer
builders call. Bind a new toolset without describing it and this fails with its name.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

GRAPH = Path(__file__).resolve().parents[2] / "agent_runtime" / "supervisor" / "graph.py"
SOURCE = GRAPH.read_text(encoding="utf-8")

# What the supervisor must SAY for each toolset its peers bind. One of the listed terms has to
# appear in the decider prompt — they are the words a model would need to see to connect a
# request to that capability.
#
# Adding a toolset means adding a line here AND words to the prompt. That is the whole point:
# the cost of binding a capability the supervisor cannot describe is one failing test, paid
# immediately, instead of a class of misrouted turns discovered months later.
REQUIRED_TERMS = {
    "make_terrain_tools": ("dem", "elevation", "terrain", "slope"),
    "make_overlay_tools": ("overlay", "buffer", "clip", "dissolve"),
    "make_aggregate_tools": ("aggregat", "summaris", "summariz"),
    "make_temporal_tools": ("temporal", "time series", "over time"),
    "make_spatial_stats_tools": ("statistic", "autocorrelat", "cluster"),
    "make_geo_analysis_tools": ("gis", "geospatial", "spatial"),
    "make_admin_boundary_tools": ("boundary", "boundaries", "administrative"),
    "make_rs_embed_tools": ("embedding", "embed"),
    "make_rs_embed_zonal_tools": ("zone", "zonal", "segment"),
    "make_langchain_qgis_tools": ("qgis", "pyqgis"),
    "make_langchain_geocode_tools": ("geocod", "place name", "address"),
    "make_langchain_geo_tools": ("reproject", "vector", "geojson"),
    "make_langchain_granular_tools": ("search", "retriev"),
    "make_langchain_file_tools": ("file", "upload"),
    "make_conversation_file_tools": ("file", "upload"),
    "make_code_execution_tools": ("code", "execut"),
    "make_skill_tools": ("skill", "workflow"),
    "make_langchain_mcp_tools": ("mcp", "external tool", "qgis"),
}


def _bound_toolsets() -> set:
    """Every `make_*_tools(` factory the supervisor graph actually calls."""
    return set(re.findall(r"\b(make_[a-z0-9_]+_tools)\s*\(", SOURCE))


def _registry_factories() -> set:
    from agent_runtime.capability_registry import factories
    return factories()


def _decider_prompt() -> str:
    """What the decider is ACTUALLY shown: the hand-written framing plus the generated
    inventory. Reading only the source would now miss the inventory entirely — it is composed
    at runtime — and every keyword check below would be testing leftover prose."""
    from agent_runtime.capability_registry import describe
    start = SOURCE.index("You are the orchestration supervisor")
    end = SOURCE.index("Respond ONLY with JSON", start)
    framing = SOURCE[start:end]
    return f"{framing}\n{describe('analyze')}\n{describe('code')}".lower()


def test_the_registry_describes_everything_the_peers_bind():
    """The drift that mattered: bound to a peer, invisible to the supervisor."""
    undescribed = _bound_toolsets() - _registry_factories()
    assert not undescribed, (
        f"these toolsets are bound to a peer but missing from capability_registry: "
        f"{sorted(undescribed)}. The supervisor cannot route work to a capability it has not "
        "been told about — that is how a DEM request became a knowledge-base search.")


def test_the_registry_does_not_claim_tools_the_peers_do_not_bind():
    """Drift in the other direction: promising a capability that is not there sends the
    supervisor to a peer that cannot deliver, and the turn fails further downstream where the
    cause is much harder to see."""
    phantom = _registry_factories() - _bound_toolsets()
    assert not phantom, (
        f"capability_registry claims these, but no peer binds them: {sorted(phantom)}")


def test_every_bound_toolset_has_a_stated_expectation():
    """A toolset nobody listed here is a toolset nobody thought about describing."""
    unlisted = _bound_toolsets() - set(REQUIRED_TERMS)
    assert not unlisted, (
        f"these toolsets are bound to a peer but absent from REQUIRED_TERMS: {sorted(unlisted)}. "
        "Add each one here with the words the supervisor should use for it, and add those words "
        "to the registry — otherwise the supervisor cannot route work to it.")


def test_the_inventory_actually_reaches_the_prompt():
    """The generator is not decorative: if it silently returned the fallback, every keyword
    test below would still pass against leftover hand-written prose."""
    from agent_runtime.capability_registry import describe
    from agent_runtime.supervisor.graph import _capability_inventory
    assert _capability_inventory("analyze") == describe("analyze")
    assert "elevation and terrain" in _capability_inventory("analyze")


@pytest.mark.parametrize("factory", sorted(REQUIRED_TERMS))
def test_the_supervisor_can_describe_what_its_peers_bind(factory):
    if factory not in _bound_toolsets():
        pytest.skip(f"{factory} is not bound in this build")
    prompt = _decider_prompt()
    terms = REQUIRED_TERMS[factory]
    assert any(term in prompt for term in terms), (
        f"{factory} is bound to a peer, but the decider prompt never says any of {terms}. "
        "The supervisor cannot route a request to a capability it has not been told about — "
        "this is exactly how a DEM request became a knowledge-base search.")


def test_the_regression_that_prompted_this_guard():
    """Terrain shipped bound-but-undescribed. Named explicitly so it cannot quietly return."""
    assert "make_terrain_tools" in _bound_toolsets()
    prompt = _decider_prompt()
    assert any(t in prompt for t in ("dem", "elevation", "terrain")), (
        "the terrain toolset is bound but the supervisor's description of `analyze` does not "
        "mention elevation work")
