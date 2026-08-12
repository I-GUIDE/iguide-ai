"""Finding and reading extracted methods: kb_method_search / get_method_contract.

The library emitter writes callable slices and the sandbox mounts them, but until these tools
existed the agent had no way to learn a unit was there — 40 extracted, 0 discoverable.

Two properties carry the weight:

* an import line returned here must be the one that works inside the sandbox, and
* an ambiguous bare name must refuse rather than resolve. The registry admits collisions
  (three symbols on the real corpus are defined by two elements each); a tool that quietly
  returned one of them would hand the agent a confidently wrong import.

Also asserted: both names are in ``RAG_COMPONENT_TOOL_NAMES``. That set is a hard filter in
``tool_policy``, so a tool missing from it is registered, documented, and unreachable for
every intent — a failure that no test of the tool function itself can catch.
"""

from __future__ import annotations

import json

import pytest

from agent_runtime import method_library as ml


@pytest.fixture()
def registry():
    return {
        "ke_crime.load_crime_points": {
            "module": "iguide_methods.ke_crime.v_ab12cd34",
            "library_symbol": "load_crime_points",
            "qualified_name": "load_crime_points",
            "element_package": "ke_crime",
            "signature": "def load_crime_points(path, crs='EPSG:4326')",
            "doc_summary": "Read Chicago crime CSV into a GeoDataFrame of points.",
            "params": [{"name": "path", "kind": "positional"}],
            "returns": "GeoDataFrame",
            "invariants": [{"name": "requires_projected_crs"}],
            "requirements": {"pip": ["geopandas"]},
            "slice_sha": "ab12cd34",
            "provenance": {"element_id": "cca9b545", "element_title": "Chicago Crime Heatmap"},
        },
        "load_crime_points": {
            "module": "iguide_methods.ke_crime.v_ab12cd34",
            "library_symbol": "load_crime_points",
            "alias_for": "ke_crime.load_crime_points",
            "signature": "def load_crime_points(path, crs='EPSG:4326')",
            "doc_summary": "Read Chicago crime CSV into a GeoDataFrame of points.",
            "requirements": {"pip": ["geopandas"]},
            "slice_sha": "ab12cd34",
            "provenance": {"element_id": "cca9b545"},
        },
        "ke_flood.compute_accessibility": {
            "module": "iguide_methods.ke_flood.v_ff00ff00",
            "library_symbol": "compute_accessibility",
            "element_package": "ke_flood",
            "signature": "def compute_accessibility(demand, supply)",
            "doc_summary": "Two-step floating catchment accessibility.",
            "requirements": {"pip": []},
            "slice_sha": "ff00ff00",
            "provenance": {"element_id": "aaa11122", "element_title": "Spatial Accessibility"},
        },
        "get_url": {
            "ambiguous": True,
            "library_symbol": "get_url",
            "candidates": ["ke_alpha.get_url", "ke_beta.get_url"],
            "doc_summary": "'get_url' is defined by more than one element; "
                           "import it by its qualified name.",
        },
        "ke_alpha.get_url": {
            "module": "iguide_methods.ke_alpha.v_1", "library_symbol": "get_url",
            "element_package": "ke_alpha", "signature": "def get_url(x)",
            "doc_summary": "alpha url", "slice_sha": "1", "provenance": {"element_id": "alpha"},
        },
        "ke_beta.get_url": {
            "module": "iguide_methods.ke_beta.v_2", "library_symbol": "get_url",
            "element_package": "ke_beta", "signature": "def get_url(y)",
            "doc_summary": "beta url", "slice_sha": "2", "provenance": {"element_id": "beta"},
        },
    }


# ------------------------------------------------------------------ search

def test_a_topic_word_finds_the_method(registry):
    hits = ml.search_methods("chicago crime points", registry=registry)
    assert hits and hits[0]["symbol"] == "ke_crime.load_crime_points"


def test_a_snake_case_symbol_is_findable_by_its_parts(registry):
    """'load_crime_points' matched as one opaque token would make the obvious query fail."""
    assert ml.search_methods("crime", registry=registry)[0]["symbol"] == "ke_crime.load_crime_points"
    assert ml.search_methods("load points", registry=registry)[0]["symbol"] == "ke_crime.load_crime_points"


def test_search_returns_a_usable_import_line(registry):
    hit = ml.search_methods("accessibility", registry=registry)[0]
    assert hit["import_line"] == (
        "from iguide_methods.ke_flood.v_ff00ff00 import compute_accessibility")


def test_the_import_line_is_pinned_to_the_slice_sha(registry):
    """A re-ingest must not silently change the code a recorded artifact re-imports."""
    hit = ml.search_methods("accessibility", registry=registry)[0]
    assert "v_ff00ff00" in hit["import_line"]


def test_bare_aliases_do_not_duplicate_their_qualified_entry(registry):
    hits = ml.search_methods("crime", registry=registry)
    assert [h["symbol"] for h in hits].count("ke_crime.load_crime_points") == 1
    assert "load_crime_points" not in [h["symbol"] for h in hits]


def test_an_ambiguous_name_surfaces_as_ambiguous_not_as_one_winner(registry):
    hits = ml.search_methods("get_url", registry=registry)
    amb = [h for h in hits if h.get("ambiguous")]
    assert amb and sorted(amb[0]["candidates"]) == ["ke_alpha.get_url", "ke_beta.get_url"]


def test_no_match_returns_empty_rather_than_a_weak_guess(registry):
    assert ml.search_methods("quantum chromodynamics", registry=registry) == []


def test_search_respects_its_limit(registry):
    assert len(ml.search_methods("url crime accessibility", registry=registry, limit=1)) == 1


# ------------------------------------------------------------------ contracts

def test_a_contract_carries_invariants_and_requirements(registry):
    c = ml.get_contract("ke_crime.load_crime_points", registry=registry)
    assert c["signature"].startswith("def load_crime_points")
    assert c["invariants"] == [{"name": "requires_projected_crs"}]
    assert c["requirements"]["pip"] == ["geopandas"]
    assert c["provenance"]["element_id"] == "cca9b545"


def test_a_bare_name_resolves_through_its_alias(registry):
    c = ml.get_contract("load_crime_points", registry=registry)
    assert c["import_line"] == "from iguide_methods.ke_crime.v_ab12cd34 import load_crime_points"
    assert c["invariants"] == [{"name": "requires_projected_crs"}]


def test_an_ambiguous_contract_refuses_and_lists_candidates(registry):
    c = ml.get_contract("get_url", registry=registry)
    assert c.get("ambiguous") is True
    assert c["candidates"] == ["ke_alpha.get_url", "ke_beta.get_url"]
    assert "import_line" not in c, "an ambiguous name must not produce an import line"


def test_an_unqualified_name_with_one_owner_still_resolves(registry):
    c = ml.get_contract("compute_accessibility", registry=registry)
    assert c["module"] == "iguide_methods.ke_flood.v_ff00ff00"


def test_an_unknown_symbol_reports_what_is_available(registry):
    c = ml.get_contract("no_such_method", registry=registry)
    assert "error" in c and c["available"] >= 3


def test_an_empty_symbol_is_an_error_not_a_lookup(registry):
    assert "error" in ml.get_contract("", registry=registry)


# ------------------------------------------------------------------ empty library

def test_an_absent_library_reads_as_empty_not_a_crash(monkeypatch):
    monkeypatch.setattr(ml, "registry_path", lambda: None)
    assert ml.load_registry() == {}
    assert ml.search_methods("anything") == []


def test_an_empty_library_says_so_instead_of_implying_none_exists(monkeypatch):
    """"No results" and "nothing ingested" are different claims; the model cannot tell them
    apart from an empty list, and reports the stronger one."""
    monkeypatch.setattr(ml, "load_registry", lambda: {})
    monkeypatch.setattr(ml, "library_root", lambda: None)
    from agent_runtime.langchain_granular_tools import kb_method_search_tool

    payload = json.loads(kb_method_search_tool("crime"))
    assert payload["results"] == []
    assert "not evidence" in payload["note"]


def test_a_populated_library_with_no_match_says_how_many_units_exist(monkeypatch, registry):
    monkeypatch.setattr(ml, "load_registry", lambda: registry)
    monkeypatch.setattr(ml, "library_root", lambda: "/tmp/lib")
    from agent_runtime.langchain_granular_tools import kb_method_search_tool

    payload = json.loads(kb_method_search_tool("quantum chromodynamics"))
    assert payload["results"] == []
    assert "units" in payload["note"] and payload["library"]["units"] >= 3


# ------------------------------------------------------------------ wiring

def test_both_tools_are_in_the_component_filter():
    """tool_policy filters against this set; a name absent from it is stripped for EVERY intent."""
    from agent_runtime.graph_state import RAG_COMPONENT_TOOL_NAMES

    assert {"kb_method_search", "get_method_contract"} <= RAG_COMPONENT_TOOL_NAMES


def test_both_tools_are_registered_with_the_agent():
    from agent_runtime.langchain_granular_tools import make_langchain_granular_tools

    names = {getattr(t, "name", "") for t in make_langchain_granular_tools()}
    assert {"kb_method_search", "get_method_contract"} <= names


@pytest.mark.parametrize("intent", ["code_task", "general_discovery", "hybrid"])
def test_the_tools_survive_the_policy_filter(intent):
    """code_task is the one that matters — that peer writes the analysis code."""
    from agent_runtime.langchain_granular_tools import make_langchain_granular_tools
    from agent_runtime.tool_policy import select_allowed_tools

    names = [getattr(t, "name", "") for t in make_langchain_granular_tools()]
    kept = set(select_allowed_tools(intent, names))
    assert {"kb_method_search", "get_method_contract"} <= kept


def test_analysis_intent_keeps_retrieval_tools():
    """This test previously PINNED the opposite behaviour, with a note to revisit it
    deliberately rather than discover it in a run. The reachability audit revisited it.

    ``analysis_task`` selected ANALYSIS_TOOL_NAMES alone — 5 crime-MCP + 5 QGIS tools — and on
    the legacy path that intent is applied to the SearchAgent itself, so an analysis question
    built the evidence-gathering agent with ZERO retrieval tools and was answered ungrounded.
    The empty-selection fallback never fired because file/skill/quality tools had already been
    appended, so the stripping was silent.
    """
    from agent_runtime.langchain_granular_tools import make_langchain_granular_tools
    from agent_runtime.tool_policy import select_allowed_tools

    names = [getattr(t, "name", "") for t in make_langchain_granular_tools()]
    kept = set(select_allowed_tools("analysis_task", names))
    assert "keyword_search" in kept, "an analysis agent cannot ground itself"
    assert "agent_kb_search" in kept
    assert {"kb_method_search", "get_method_contract"} <= kept

def test_the_tool_returns_json_the_model_can_parse(monkeypatch, registry):
    monkeypatch.setattr(ml, "load_registry", lambda: registry)
    monkeypatch.setattr(ml, "library_root", lambda: "/tmp/lib")
    from agent_runtime.langchain_granular_tools import (get_method_contract_tool,
                                                        kb_method_search_tool)

    hits = json.loads(kb_method_search_tool("crime", limit=3))["results"]
    assert hits[0]["symbol"] == "ke_crime.load_crime_points"
    contract = json.loads(get_method_contract_tool(hits[0]["symbol"]))
    assert contract["import_line"] == hits[0]["import_line"]


# ------------------------------------------------------------------ the second filter

def _names(enabled):
    from agent_runtime.langchain_granular_tools import make_langchain_granular_tools
    return {getattr(t, "name", "")
            for t in make_langchain_granular_tools(enabled_search_methods=enabled)}


def test_enabling_a_search_tool_brings_its_reader(): 
    """A reader is the second half of its search tool, never an independent method.

    Both kb_method_search and agent_kb_search return truncated hits; enabling one without its
    reader leaves the agent able to FIND a method but unable to read its contract.
    """
    assert {"kb_method_search", "get_method_contract"} <= _names(["kb_method_search"])
    assert {"agent_kb_search", "get_kb_block"} <= _names(["agent_kb_search"])
    assert {"web_search", "web_fetch"} <= _names(["web_search"])


def test_a_reader_is_not_offered_without_its_search_tool():
    kept = _names(["keyword_search"])
    assert "get_method_contract" not in kept
    assert "get_kb_block" not in kept


def test_the_prototype_default_reaches_the_method_library():
    """The prototype's enabled_search_methods list is a HARD filter, and it omitted both
    agent_kb_search and kb_method_search — so the KB and the method library were registered,
    allowed by tool_policy, and still never offered to the search peer."""
    import re
    from pathlib import Path

    html = Path("examples/iguide_chat_prototype.html").read_text(encoding="utf-8")
    match = re.search(r"enabled_search_methods:\s*\[(.*?)\]", html, re.DOTALL)
    assert match, "the prototype no longer declares enabled_search_methods"
    configured = set(re.findall(r'"([a-z_]+)"', match.group(1)))
    assert {"agent_kb_search", "kb_method_search"} <= configured
    assert {"kb_method_search", "get_method_contract",
            "agent_kb_search", "get_kb_block"} <= _names(sorted(configured))


# ------------------------------------------------------------------ the third filter: the prompt

def test_the_search_persona_names_the_method_library():
    """The tool being ALLOWED is not enough — the peer follows the enumerated playbook.

    Observed with the tool registered, allowed by tool_policy and enabled in the request: the
    peer called web_search, keyword_search, semantic_search and neo4j_get_element_by_id, then
    answered "adapt this notebook" while `plot_choropleth_map` sat in the library with a
    working import line. The COVERAGE rule lists tools by name and nothing named the library.
    """
    from agent_runtime.prompts import SEARCH_AGENT_PROMPT as P

    assert "kb_method_search" in P
    assert "agent_kb_search" in P, "sub-document evidence is absent from the coverage fan-out"
    assert "get_method_contract" in P


def test_the_code_persona_checks_for_an_existing_method_first():
    from agent_runtime.prompts import CODE_AGENT_PROMPT as P

    assert "kb_method_search" in P


# ------------------------------------------------------------------ deterministic union

def test_an_off_topic_question_matches_no_method(registry):
    """Symbol names are full of connective words — determine_number_of_cluster contains "of" —
    so without stopword filtering "what is the capital of France" scored a clustering method
    6.5 on that word alone and it reached the evidence set."""
    assert ml.search_methods("what is the capital of France", registry=registry) == []


def test_a_query_of_only_stopwords_returns_nothing(registry):
    assert ml.search_methods("what is the of and to", registry=registry) == []


def test_stopword_filtering_does_not_break_short_symbol_queries(registry):
    """'get url' must still find get_url; the filter applies to the QUERY, not the symbol.

    Note "get" IS a query stopword, so the match rides on "url" and the joined token "get_url".
    """
    hits = ml.search_methods("get_url", registry=registry)
    assert hits, "a bare symbol name became unfindable"
    assert any("get_url" in h["symbol"] for h in hits)


def test_the_sweep_unions_methods_without_the_model_choosing(monkeypatch, registry):
    """The peer would not call the tool: across three runs of the same question, with it
    registered, policy-allowed, request-enabled and named in two persona rules, it was called
    zero times. graph.py:1445 already states the principle — do not rely on tool choice."""
    monkeypatch.setattr(ml, "load_registry", lambda: registry)
    from agent_runtime.supervisor.graph import _method_units_as_documents

    docs = _method_units_as_documents("load chicago crime points", 8)
    assert docs, "the sweep surfaced no method for a query that clearly matches one"
    assert docs[0]["source"] == "method_library"
    assert "from iguide_methods" in docs[0]["contents"], (
        "evidence without the import line is barely better than naming a notebook")


def test_swept_methods_cite_their_source_element_not_a_synthetic_id(monkeypatch, registry):
    monkeypatch.setattr(ml, "load_registry", lambda: registry)
    from agent_runtime.supervisor.graph import _method_units_as_documents

    doc = _method_units_as_documents("load chicago crime points", 8)[0]
    assert doc["citation_ids"] == ["cca9b545"]
    assert not doc["citation_ids"][0].startswith("method::")


def test_the_sweep_applies_a_relevance_floor(monkeypatch, registry):
    """The sweep spends evidence slots unasked, so it takes only clear matches."""
    monkeypatch.setattr(ml, "load_registry", lambda: registry)
    from agent_runtime.supervisor.graph import _method_units_as_documents

    assert len(_method_units_as_documents("load chicago crime points", 8)) <= 4


def test_the_sweep_respects_the_enabled_methods_allowlist(monkeypatch, registry):
    monkeypatch.setattr(ml, "load_registry", lambda: registry)
    from agent_runtime.supervisor.graph import _direct_search_sweep

    docs = _direct_search_sweep("load chicago crime points", ["keyword_search"], k=5)
    assert not [d for d in docs if d.get("source") == "method_library"]


def test_every_swept_document_carries_an_import_line(monkeypatch, registry):
    """The ambiguous STUB has no import line and is a dead end as evidence; its qualified
    candidates do have one and are legitimate evidence. The invariant is the import line, not
    the name."""
    monkeypatch.setattr(ml, "load_registry", lambda: registry)
    from agent_runtime.supervisor.graph import _method_units_as_documents

    docs = _method_units_as_documents("get_url", 8)
    assert docs, "both qualified get_url units should be reachable"
    for d in docs:
        assert d.get("import_line"), f"{d['title']} entered evidence with no way to import it"
        assert "from iguide_methods" in d["contents"]


# ------------------------------------------------------------------ the fourth filter

def test_the_api_accepts_the_method_tools_as_search_methods():
    """The fourth independent gate on the same name.

    kb_method_search had to be added to FOUR separate lists before it could be reached:
    RAG_COMPONENT_TOOL_NAMES (tool_policy), the enabled_search_methods filter in
    make_langchain_granular_tools, the prototype's CONFIG, and this request-validation
    allowlist. Missing here, the whole request 400s — observed as
    "unknown search method(s): 'kb_method_search'" with the run never starting.
    """
    from agent_runtime.search_methods import KNOWN_SEARCH_METHODS, normalize_search_methods

    assert {"kb_method_search", "get_method_contract"} <= set(KNOWN_SEARCH_METHODS)
    assert normalize_search_methods(["kb_method_search"]) == ["kb_method_search"]


def test_the_prototype_payload_validates_end_to_end():
    """Every method the shipped prototype sends must survive request validation."""
    import re
    from pathlib import Path

    from agent_runtime.search_methods import normalize_search_methods

    html = Path("examples/iguide_chat_prototype.html").read_text(encoding="utf-8")
    configured = re.findall(r'"([a-z_]+)"',
                            re.search(r"enabled_search_methods:\s*\[(.*?)\]", html, re.DOTALL).group(1))
    assert set(normalize_search_methods(configured)) == set(configured)


# ------------------------------------------------------------------ the fifth filter

def test_both_code_running_peers_can_reach_the_method_library():
    """Both peers that can RUN code must be able to FIND a method.

    default_code_fn built its KB toolset from a hardcoded allowlist that listed only
    agent_kb_search/get_kb_block, so the peer whose sandbox mounts the library could not see
    it; told by its own prompt to call kb_method_search and not having it, the model guessed
    `from method_library import ...` and got ModuleNotFoundError. default_analyze_fn holds
    execute_code too and had the same gap.
    """
    import inspect

    from agent_runtime.supervisor import graph

    for fn in (graph.default_code_fn, graph.default_analyze_fn):
        src = inspect.getsource(fn)
        assert "_CODE_PEER_KB_TOOLS" in src, f"{fn.__name__} cannot reach the method library"
    assert {"kb_method_search", "get_method_contract"} <= graph._CODE_PEER_KB_TOOLS


def test_execute_code_tells_the_model_the_package_name():
    """A peer that skips the tool must still not be able to guess wrong."""
    from agent_runtime.langchain_exec_tools import make_code_execution_tools

    desc = make_code_execution_tools()[0].description
    assert "iguide_methods" in desc
    assert "kb_method_search" in desc


def test_the_tool_description_survives_the_cli_shim_intact():
    """The shim's old 600-char cut landed inside execute_code's description at
    "...they are installed wi", removing how to pass dependencies."""
    from agent_runtime.chat_claude_cli import _tool_schema
    from agent_runtime.langchain_exec_tools import make_code_execution_tools

    schema = _tool_schema(make_code_execution_tools()[0])
    assert "dependencies" in schema["description"]
    assert "iguide_methods" in schema["description"]
    assert not schema["description"].endswith("installed wi")


# ------------------------------------------------------------------ agent_kb in the sweep

def test_agent_kb_documents_enter_the_sweep(monkeypatch):
    """The first version of this arm read payload["results"]; agent_kb_search returns
    "documents". No exception, no log — a permanently empty arm that the surrounding
    `except` could never have caught, because nothing raised."""
    from rag_pipeline.search import agent_kb
    from agent_runtime.supervisor import graph

    monkeypatch.setattr(agent_kb, "agent_kb_search", lambda q, size=None, **kw: {
        "source": "agent_kb", "count": 1,
        "documents": [{"doc_id": "elem1::block::3", "parent_doc_id": "elem1",
                       "title": "load step", "contents": "gpd.read_file(...)",
                       "score": 2.0, "matched": "keyword"}],
        "citation_ids": ["elem1"], "elements": {}})

    docs = graph._direct_search_sweep("chicago crime", ["agent_kb_search"], k=5)
    kb = [d for d in docs if d.get("source") == "agent_kb"]
    assert kb, "agent_kb documents never reached the evidence set"
    assert kb[0]["contents"] == "gpd.read_file(...)"


def test_swept_agent_kb_docs_cite_the_parent_element(monkeypatch):
    """A block id is not something a reader can open."""
    from rag_pipeline.search import agent_kb
    from agent_runtime.supervisor import graph

    monkeypatch.setattr(agent_kb, "agent_kb_search", lambda q, size=None, **kw: {
        "documents": [{"doc_id": "elem1::block::3", "parent_doc_id": "elem1",
                       "title": "t", "contents": "c"}]})
    doc = [d for d in graph._direct_search_sweep("q", ["agent_kb_search"], k=5)
           if d.get("source") == "agent_kb"][0]
    assert doc["citation_ids"] == ["elem1"]


def test_agent_kb_is_skipped_when_not_enabled(monkeypatch):
    from rag_pipeline.search import agent_kb
    from agent_runtime.supervisor import graph

    called = []
    monkeypatch.setattr(agent_kb, "agent_kb_search",
                        lambda q, size=None, **kw: called.append(q) or {"documents": []})
    graph._direct_search_sweep("q", ["keyword_search"], k=5)
    assert called == []


# ------------------------------------------------------------------ shipped clients + prompt

def _html(name):
    from pathlib import Path
    return Path("examples") / name


def test_both_shipped_clients_can_request_the_kb_and_method_library():
    """The prototype fix left the same defect live in the second shipped page, which
    api/server.py serves at /agent/dashboard."""
    for name in ("iguide_chat_prototype.html", "agent_chat_stream_demo.html"):
        text = _html(name).read_text(encoding="utf-8")
        assert "agent_kb_search" in text, f"{name} cannot request sub-document evidence"
        assert "kb_method_search" in text, f"{name} cannot request the method library"


def test_the_prototype_logs_method_library_calls():
    """A tool name absent from RETRIEVAL_TOOLS is dropped from the reasoning log entirely, so
    the user cannot tell whether the agent consulted the library or ignored it."""
    import re

    text = _html("iguide_chat_prototype.html").read_text(encoding="utf-8")
    block = re.search(r"const RETRIEVAL_TOOLS=new Set\(\[(.*?)\]\)", text, re.DOTALL).group(1)
    listed = set(re.findall(r'"([a-z0-9_]+)"', block))
    assert {"kb_method_search", "get_method_contract"} <= listed


def test_the_search_prompt_names_only_tools_that_can_exist():
    """rule 8 named `fetch_element_source`; the live MCP server exposes it as
    `mcp_fetch_element_source`, and MCP tools are always prefixed."""
    from agent_runtime.prompts import SEARCH_AGENT_PROMPT as P

    assert "`fetch_element_source`" not in P
    assert "mcp_fetch_element_source" in P


def test_the_element_source_tool_survives_the_policy_filter():
    """It was in no name set, so it was stripped for every intent except analysis_task —
    where it survived only because the empty-selection fallback returns everything."""
    from agent_runtime.tool_policy import select_allowed_tools

    for intent in ("general_discovery", "code_task", "hybrid"):
        kept = select_allowed_tools(intent, ["keyword_search", "mcp_fetch_element_source"])
        assert "mcp_fetch_element_source" in kept, f"stripped for {intent}"
