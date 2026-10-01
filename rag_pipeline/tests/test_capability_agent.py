"""A capability question is answered by reasoning over live introspection, not a fixed prompt.

The old route was a one-shot composer that received the inventory and NO query, under an
instruction to "cover everything in the inventory". So every capability question got the same
grouped catalogue of every area, written by a model that had never seen what was asked: "what
can you do with satellite imagery?" returned five generic headings and never mentioned
embeddings, though the inventory carried eight tools for exactly that. From the user's side
that is indistinguishable from a hardcoded answer.

It also truncated: json.dumps(inventory, indent=1)[:12000] against a 27,397-char blob dropped
56% of the surface — always the registries added last, and sliced mid-object, so the model
received malformed JSON.
"""
from __future__ import annotations

import json

import pytest

from agent_runtime.capabilities import describe_capabilities, make_capability_tools


@pytest.fixture()
def introspect():
    tools = make_capability_tools(include_mcp_tools=False)
    assert tools, "the introspection tool must exist"
    return tools[0]


# --- introspection is live, filterable, and cannot mislead --------------------------------

def test_the_agent_can_look_up_its_own_surface(introspect):
    """The overview is now the AREA MAP — every tool name, grouped, no descriptions."""
    out = json.loads(introspect.invoke({}))
    assert out["total_tools"] > 40
    every = [n for a in out["areas"] for n in a["tool_names"]]
    assert "embed_zones" in every


def test_a_topic_narrows_the_result(introspect):
    out = json.loads(introspect.invoke({"topic": "cluster"}))
    assert 0 < out["matched"] < out["total_tools"]
    assert any("cluster" in t["name"] or "cluster" in (t["description"] or "").lower()
               for t in out["detail"])
    # every match says which area it came from, so the agent can open the neighbours
    assert all(t.get("area") for t in out["detail"])


def test_a_weak_topic_match_cannot_read_as_the_whole_answer(introspect):
    """The vocabulary gap is real: the embedding tools say "remote-sensing foundation model",
    so "satellite imagery" matches almost nothing by substring. The area map coming back every
    time is what stops that reading as the whole answer — and unlike a flat list of 70 names it
    tells the agent WHERE to look next."""
    out = json.loads(introspect.invoke({"topic": "satellite imagery"}))
    every = [n for a in out["areas"] for n in a["tool_names"]]
    for want in ("embed_region", "embed_zones", "align_embedding_colors", "fit_zone_model",
                 "predict_from_package", "list_embedding_models"):
        assert want in every, f"{want} must stay discoverable on a weak match"
    assert "hint" in out


def test_an_unsupported_topic_says_so_rather_than_guessing(introspect):
    out = json.loads(introspect.invoke({"topic": "quantum computing"}))
    assert out["matched"] == 0
    assert out["areas"], "and still shows what IS available"


def test_the_payload_stays_small_enough_to_reason_over(introspect):
    """The point of a filterable tool: no 12,000-char truncation of a 27,000-char blob."""
    assert len(introspect.invoke({"topic": "satellite imagery"})) < 8000


# --- the loop itself ----------------------------------------------------------------------

class _ScriptedAgent:
    """Stands in for the ReAct executor: records the question, returns a canned answer."""

    def __init__(self, answer="Satellite imagery: embeddings for a rectangle or per polygon."):
        self.answer = answer
        self.seen = []


def test_the_question_reaches_the_agent(monkeypatch):
    captured = {}

    import agent_runtime.executor_factory as ef

    def _fake_build(**kwargs):
        captured["prompt"] = kwargs.get("system_prompt_override")
        captured["tools"] = [str(getattr(t, "name", "")) for t in (kwargs.get("preloaded_tools") or [])]
        return object()

    def _fake_invoke(executor, *, query, chat_history=None, config=None):
        captured["query"] = query
        return {"messages": []}

    monkeypatch.setattr(ef, "build_agent_executor", _fake_build)
    monkeypatch.setattr(ef, "invoke_agent_with_payload_fallback", _fake_invoke)
    monkeypatch.setattr(ef, "build_default_llm", lambda: object())

    describe_capabilities(query="what can you do with satellite imagery?",
                          include_mcp_tools=False)

    assert captured["query"] == "what can you do with satellite imagery?"
    assert "list_my_capabilities" in captured["tools"], "the agent must be able to introspect"
    assert "INTROSPECT" in (captured["prompt"] or "")
    assert "ANSWER THE QUESTION ASKED" in (captured["prompt"] or "")


def test_the_real_model_listers_are_available_to_it(monkeypatch):
    """"which models can you use?" should be answerable with the actual names."""
    captured = {}
    import agent_runtime.executor_factory as ef
    monkeypatch.setattr(ef, "build_agent_executor",
                        lambda **kw: captured.setdefault(
                            "tools", [str(getattr(t, "name", "")) for t in kw.get("preloaded_tools") or []]) and object() or object())
    monkeypatch.setattr(ef, "invoke_agent_with_payload_fallback",
                        lambda *a, **k: {"messages": []})
    monkeypatch.setattr(ef, "build_default_llm", lambda: object())

    describe_capabilities(query="which embedding models do you have?", include_mcp_tools=False)
    assert "list_embedding_models" in captured["tools"]


def test_it_falls_back_to_a_mechanical_listing_when_no_model_answers(monkeypatch):
    import agent_runtime.executor_factory as ef

    def _boom(**kwargs):
        raise RuntimeError("no llm")

    monkeypatch.setattr(ef, "build_agent_executor", _boom)
    monkeypatch.setattr(ef, "build_default_llm", lambda: object())

    out = describe_capabilities(query="what can you do?", include_mcp_tools=False)
    assert "Available capabilities:" in out
    assert "embed_zones" in out, "the fallback is still derived from the live registries"


def test_the_route_passes_the_users_question(monkeypatch):
    """The node used to call describe_capabilities without a query at all."""
    import inspect

    from agent_runtime import orchestrator_graph

    src = inspect.getsource(orchestrator_graph)
    assert 'query=state.get("query")' in src


# --- hierarchical discovery ----------------------------------------------------------------
#
# The tree is the registry structure the code already has, so a new module becomes a new area
# with no branch defined anywhere. Nothing is authored per tool or per area.

def test_no_arguments_returns_a_navigable_map_not_the_detail(introspect):
    import json as _j

    out = _j.loads(introspect.invoke({}))
    assert len(out["areas"]) > 10
    assert all(set(a) == {"area", "tools", "tool_names"} for a in out["areas"])
    # a map, not a dump: no descriptions at this level
    assert "description" not in _j.dumps(out["areas"])
    assert len(_j.dumps(out)) < 6000, "the overview must stay cheap to read"


def test_the_total_is_distinct_not_the_sum_of_areas(introspect):
    """A few tools are registered by two registries, so summing per-area counts overstates."""
    import json as _j

    from agent_runtime.capabilities import collect_capability_inventory

    out = _j.loads(introspect.invoke({}))
    assert out["total_tools"] == len(collect_capability_inventory(include_mcp_tools=False)["tools"])
    assert out["listings"] >= out["total_tools"]


def test_an_area_can_be_opened_for_detail(introspect):
    import json as _j

    out = _j.loads(introspect.invoke({"area": "rs_embed_zonal"}))
    names = [t["name"] for d in out["detail"] for t in d["tools"]]
    assert "embed_zones" in names
    assert all(t.get("description") for d in out["detail"] for t in d["tools"])


def test_an_unknown_area_says_so_and_still_shows_the_map(introspect):
    import json as _j

    out = _j.loads(introspect.invoke({"area": "quantum"}))
    assert out["detail"] == [] and "hint" in out
    assert out["areas"], "the map must always come back so the agent can navigate"


def test_a_weak_topic_match_leaves_the_agent_somewhere_to_go(introspect):
    """The real recovery path: "satellite imagery" matches almost nothing by substring, but
    rs_embed and rs_embed_zonal are visible in the map and can be opened."""
    import json as _j

    out = _j.loads(introspect.invoke({"topic": "satellite imagery"}))
    area_ids = [a["area"] for a in out["areas"]]
    assert any("embed" in a for a in area_ids)
    assert "hint" in out and "literal" in out["hint"]


def test_area_ids_are_derived_from_the_factory_name():
    """So a new registry names its own area — nothing to add to a table."""
    from agent_runtime.capabilities import _area_id

    assert _area_id("make_rs_embed_zonal_tools") == "rs_embed_zonal"
    assert _area_id("make_langchain_granular_tools") == "granular"
    assert _area_id("make_admin_boundary_tools") == "admin_boundary"


def test_a_new_registry_becomes_a_new_area(monkeypatch):
    """The property that matters when the tool set changes."""
    import json as _j

    import agent_runtime.capabilities as cap
    from langchain_core.tools import tool

    @tool
    def brand_new_thing(x: str) -> str:
        """Does something nobody anticipated."""
        return x

    real = cap._discover_registry_factories
    monkeypatch.setattr(cap, "_discover_registry_factories",
                        lambda: [*real(), ("make_brand_new_tools", lambda: [brand_new_thing])])

    out = _j.loads(cap.make_capability_tools(include_mcp_tools=False)[0].invoke({}))
    assert "brand_new" in [a["area"] for a in out["areas"]]
    detail = _j.loads(cap.make_capability_tools(include_mcp_tools=False)[0]
                      .invoke({"area": "brand_new"}))
    assert "brand_new_thing" in [t["name"] for d in detail["detail"] for t in d["tools"]]
