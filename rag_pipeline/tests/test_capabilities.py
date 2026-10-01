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
    """Data-driven proof: a tool that exists only in the registry reaches the MODEL.

    It now arrives through the introspection tool rather than being pasted into a prompt —
    which is what removed the 12,000-char truncation that used to drop 56% of the surface.
    """
    import agent_runtime.capabilities as cap

    from langchain_core.tools import tool

    @tool
    def brand_new_tool(x: str) -> str:
        """Does a brand new thing."""
        return x

    # Stub at the DISCOVERY seam, which is the real contract now: a registry appears and its
    # tools come with it, without this module (or the tool) naming either.
    monkeypatch.setattr(cap, "_discover_registry_factories",
                        lambda: [("make_brand_new_tools", lambda: [brand_new_tool])])

    introspect = cap.make_capability_tools(include_mcp_tools=False)[0]
    assert "brand_new_tool" in introspect.invoke({})                      # in the area map
    assert "Does a brand new thing." in introspect.invoke({"area": "brand_new"})


def test_answer_is_llm_composed_not_authored(monkeypatch):
    """The wording is still the model's — there is no authored capability prose anywhere.

    Previously this asserted the prompt contained "Tool inventory"; the inventory now reaches
    the model as a tool result, so the contract is about WHERE the answer comes from.
    """
    from agent_runtime.capabilities import describe_capabilities
    import agent_runtime.executor_factory as ef

    monkeypatch.setattr(ef, "build_default_llm", lambda: object())
    monkeypatch.setattr(ef, "build_agent_executor", lambda **kw: object())
    monkeypatch.setattr(ef, "invoke_agent_with_payload_fallback",
                        lambda *a, **k: {"messages": [_Msg("**Finding things**\nI can search.")]})

    assert describe_capabilities(query="what can you do?") == \
        "**Finding things**\nI can search."


class _Msg:
    """An AI message shaped the way extract_final_answer reads one."""

    def __init__(self, content):
        self.content = content
        self.tool_calls = []


def test_the_agent_is_given_the_means_to_introspect(monkeypatch):
    """A fixed prompt can only answer the question it was written for; the loop can look."""
    from agent_runtime.capabilities import describe_capabilities
    import agent_runtime.executor_factory as ef

    captured = {}

    def _build(**kw):
        captured["tools"] = [str(getattr(t, "name", "")) for t in (kw.get("preloaded_tools") or [])]
        captured["prompt"] = kw.get("system_prompt_override") or ""
        return object()

    monkeypatch.setattr(ef, "build_default_llm", lambda: object())
    monkeypatch.setattr(ef, "build_agent_executor", _build)
    monkeypatch.setattr(ef, "invoke_agent_with_payload_fallback",
                        lambda *a, **k: {"messages": [_Msg("ok")]})

    describe_capabilities(query="can you handle GeoTIFF?")
    assert "list_my_capabilities" in captured["tools"]
    assert "INTROSPECT" in captured["prompt"]


def test_mechanical_fallback_when_no_model_answers():
    """If the model is unreachable the answer still comes from the registry, not a hand-written
    blurb."""
    from agent_runtime.capabilities import describe_capabilities

    import agent_runtime.executor_factory as ef

    def _boom(**kwargs):
        raise RuntimeError("llm down")

    ef_build = ef.build_agent_executor
    ef.build_agent_executor = _boom
    try:
        text = describe_capabilities(query="what can you do?")
    finally:
        ef.build_agent_executor = ef_build
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

    import agent_runtime.executor_factory as ef
    monkeypatch.setattr(ef, "build_agent_executor", lambda **kw: object())
    monkeypatch.setattr(ef, "invoke_agent_with_payload_fallback", lambda *a, **k: {
        "messages": [_Msg("I can search the knowledge base and run analyses.")]})

    graph = og.build_orchestrator_graph(llm=object())
    state = graph.invoke({"query": "what tools do you have", "chat_history": [], "thread_id": None})
    assert state.get("final_answer") == "I can search the knowledge base and run analyses."
