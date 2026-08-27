"""Capability/meta questions ("what tools do you have") answered from the live registries.

No LLM in the supervisor pipeline sees the tool registry, so these were previously answered
with irrelevant KB retrieval or the no-grounding reply.
"""

from __future__ import annotations

from agent_runtime.capabilities import describe_capabilities, is_capability_query


def test_capability_detection_positive():
    for q in (
        "what tools do you have",
        "What tools can you use?",
        "Which tools do you have access to",
        "what can you do",
        "What are your capabilities?",
        "list your tools",
        "show me your available tools",
        "what skills do you have",
    ):
        assert is_capability_query(q), q


def test_capability_detection_does_not_hijack_domain_questions():
    for q in (
        "what tools are available for flood mapping",     # domain, not about the assistant
        "what tools do researchers use for accessibility analysis",
        "show tools for hydrology analysis",
        "explain the QGIS processing tools in this notebook",
        "find datasets about surveying tools",
    ):
        assert not is_capability_query(q), q


def test_inventory_is_read_from_the_live_registries():
    """The answer's source of truth is the real tool registry — nothing hardcoded here — so a
    tool added/removed anywhere shows up without editing this module."""
    from agent_runtime.capabilities import collect_capability_inventory
    inv = collect_capability_inventory()
    names = {t["name"] for t in inv["tools"]}
    # representative tools from several independent registries
    assert {"keyword_search", "semantic_search", "neo4j_search"} <= names      # granular search
    assert "geocode_places" in names                                          # geocode registry
    assert {"inspect_vector", "render_map_image"} <= names                         # geo registry
    assert all(t["description"] for t in inv["tools"] if t["name"] == "keyword_search")
    assert inv["code_execution"]["enabled"] in (True, False)
    assert isinstance(inv["skills"], list)


def test_new_tool_appears_without_touching_this_module(monkeypatch):
    """Data-driven proof: a tool that exists only in the registry reaches the LLM prompt."""
    from types import SimpleNamespace
    import agent_runtime.capabilities as cap

    monkeypatch.setattr(cap, "collect_capability_inventory", lambda **k: {
        "tools": [{"name": "brand_new_tool", "description": "Does a brand new thing."}],
        "code_execution": {"enabled": True, "sandbox_backend": "docker"}, "skills": [],
    })
    seen = {}

    def fake_llm(prompt):
        seen["prompt"] = prompt
        return "Composed answer."
    assert cap.describe_capabilities(llm=fake_llm) == "Composed answer."
    assert "brand_new_tool" in seen["prompt"]           # inventory reached the model
    assert "Does a brand new thing." in seen["prompt"]


def test_answer_is_llm_composed_from_the_inventory():
    """describe_capabilities delegates the wording to the LLM (no authored capability prose)."""
    from agent_runtime.capabilities import describe_capabilities
    captured = {}

    class _LLM:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return SimpleNamespaceContent("**Finding things**\nI can search the platform.")

    class SimpleNamespaceContent:
        def __init__(self, content):
            self.content = content

    out = describe_capabilities(llm=_LLM())
    assert out == "**Finding things**\nI can search the platform."
    p = captured["prompt"]
    assert "Tool inventory" in p and "keyword_search" in p        # live inventory in the prompt
    assert "Do NOT name internal tools" in p                      # style constraint enforced


def test_mechanical_fallback_when_no_model_answers():
    """If the model is unreachable the answer still comes from the registry, not a hand-written
    blurb."""
    from agent_runtime.capabilities import describe_capabilities

    def broken_llm(prompt):
        raise RuntimeError("llm down")
    text = describe_capabilities(llm=broken_llm)
    assert "Available capabilities:" in text
    assert "keyword_search" in text          # mechanical listing derived from the inventory


def test_graph_routes_capability_question_deterministically(monkeypatch):
    """'what tools do you have' must be answered by the capabilities node — the orchestrate
    strategy (LLM pipeline) must not run at all."""
    import agent_runtime.orchestrator_graph as og
    import agent_runtime.strategy as strat

    def explode(*a, **k):
        raise AssertionError("orchestrate strategy must not run for a capability question")
    monkeypatch.setattr(strat, "get_orchestration_strategy", explode)

    class _LLM:
        def invoke(self, prompt):
            assert "Tool inventory" in prompt      # composed from the live inventory
            class R:
                content = "I can search the knowledge base and run analyses."
            return R()

    graph = og.build_orchestrator_graph(llm=_LLM())
    state = graph.invoke({"query": "what tools do you have", "chat_history": [], "thread_id": None})
    assert state.get("final_answer") == "I can search the knowledge base and run analyses."
