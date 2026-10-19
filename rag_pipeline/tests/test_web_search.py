"""Open-web search: metadata-only results, per-turn budget, and URL provenance.

The failure modes this guards are the ones that make a naive web-search tool worse than none:

* page bodies leaking into the discovery stage (cost explodes, and the model answers from chrome);
* an unbounded number of network calls per turn;
* a refusal (disabled / budget spent / provider down) reading to the model as "the web has
  nothing", which it then reports as a finding;
* fabricated-but-plausible URLs in the answer — already observed twice in this service.
"""

from __future__ import annotations

import json

import pytest

import rag_pipeline.search.web as W
import rag_pipeline.search.web_utils as WU


@pytest.fixture(autouse=True)
def _fresh_budget(monkeypatch):
    """Each test gets its own ledger and the documented default caps."""
    for var in (
        "AGENT_WEB_ENABLED",
        "AGENT_WEB_MAX_SEARCHES_PER_TURN",
        "AGENT_WEB_MAX_FETCHES_PER_TURN",
        "AGENT_WEB_SNIPPET_CHARS",
        "AGENT_WEB_PROVIDER",
    ):
        monkeypatch.delenv(var, raising=False)
    WU.begin_turn()


def _raw(n=3, host="example.org"):
    return [
        {"title": f"Dam safety report {i}", "href": f"https://{host}/dams/{i}",
         "body": f"Inspection findings for dam {i} in Illinois."}
        for i in range(1, n + 1)
    ]


def _stub_provider(monkeypatch, raw):
    calls = {}

    def fake(query, *, limit=6, recency_days=None):
        calls["query"] = query
        calls["limit"] = limit
        calls["recency_days"] = recency_days
        return [
            WU.WebHit(url=r["href"], title=r["title"], snippet=r["body"],
                      provider="stub", rank=i)
            for i, r in enumerate(raw, start=1)
        ]

    monkeypatch.setitem(W._PROVIDERS, "duckduckgo", fake)
    return calls


# --- query hygiene -------------------------------------------------------------


def test_web_query_strips_only_the_conversational_framing():
    assert W.web_query("Can you search the web for dam failures in Illinois") == "dam failures in Illinois"
    assert W.web_query("look up NHDPlus documentation") == "NHDPlus documentation"
    assert W.web_query("Please help me find dam inspection standards") == "find dam inspection standards"
    # A product name that happens to be a search engine is part of the subject, not framing.
    assert W.web_query("Google Earth Engine NDVI tutorial") == "Google Earth Engine NDVI tutorial"
    # Unlike the catalog path's focus_query, "open data" survives: on the web those are real,
    # discriminating tokens, not filler.
    assert W.web_query("open data portal Illinois") == "open data portal Illinois"
    assert W.web_query("   ") == ""


# --- canonicalization + dedupe -------------------------------------------------


def test_canonical_url_drops_fragment_trackers_and_default_port():
    assert WU.canonical_url("HTTPS://Example.ORG:443/a/b?x=1&utm_source=news#frag") == \
        "https://example.org/a/b?x=1"
    assert WU.canonical_url("https://example.org/a?gclid=123") == "https://example.org/a"
    assert WU.canonical_url("https://example.org") == "https://example.org/"
    assert WU.canonical_url("") == ""
    # A non-default port is significant and must be kept.
    assert WU.canonical_url("http://example.org:8080/a") == "http://example.org:8080/a"


def test_dedupe_collapses_the_same_document_across_engines():
    hits = [
        WU.WebHit(url="https://example.org/a?utm_source=x", title="A", snippet="", provider="p1"),
        WU.WebHit(url="https://example.org/a#section", title="A dup", snippet="", provider="p2"),
        WU.WebHit(url="https://example.org/b", title="B", snippet="", provider="p1"),
    ]
    kept = W.dedupe(hits)
    assert [h.title for h in kept] == ["A", "B"]


def test_doc_id_is_stable_across_url_variants():
    a = WU.WebHit(url="https://example.org/a?utm_source=x", title="", snippet="", provider="p")
    b = WU.WebHit(url="https://example.org/a", title="", snippet="", provider="p")
    assert a.doc_id() == b.doc_id()


# --- relevance: gentle, and never empties the set ------------------------------


def test_filter_drops_zero_evidence_hits():
    hits = [
        WU.WebHit(url="https://x/1", title="Illinois Dam Safety", snippet="", provider="p"),
        WU.WebHit(url="https://x/2", title="Cheap flights to Rome", snippet="book now", provider="p"),
    ]
    kept = W.filter_hits(hits, ["dam", "illinois"])
    assert [h.url for h in kept] == ["https://x/1"]


def test_filter_falls_back_to_engine_ranking_rather_than_returning_nothing():
    """A search engine has already ranked for relevance and web snippets are short, so a query
    whose terms are simply absent from the snippets must not wipe out every result."""
    hits = [WU.WebHit(url="https://x/1", title="USGS water resources", snippet="gauges", provider="p")]
    assert W.filter_hits(hits, ["sedimentation"]) == hits
    assert W.filter_hits(hits, []) == hits


# --- metadata only ------------------------------------------------------------


def test_search_results_carry_no_page_body(monkeypatch):
    _stub_provider(monkeypatch, _raw(2))
    result = W.run_web_search("dam safety Illinois")
    assert result["count"] == 2
    for item in result["results"]:
        assert set(item) == {"doc_id", "title", "url", "snippet", "provider", "published", "rank"}
        assert "content" not in item and "body" not in item


def test_snippets_are_capped_far_below_catalog_abstracts(monkeypatch):
    long_body = "sediment " * 500
    monkeypatch.setitem(
        W._PROVIDERS, "duckduckgo",
        lambda q, *, limit=6, recency_days=None: [
            WU.WebHit(url="https://x/1", title="Sedimentation", provider="p", rank=1,
                      snippet=long_body[: WU.search_snippet_chars()])
        ],
    )
    result = W.run_web_search("sedimentation")
    assert len(result["results"][0]["snippet"]) <= WU.search_snippet_chars() == 300


def test_recency_days_maps_onto_engine_buckets(monkeypatch):
    calls = _stub_provider(monkeypatch, _raw(1))
    W.run_web_search("news", recency_days=7)
    assert calls["recency_days"] == 7
    assert W._timelimit(1) == "d" and W._timelimit(7) == "w"
    assert W._timelimit(30) == "m" and W._timelimit(200) == "y"
    assert W._timelimit(None) is None and W._timelimit(5000) is None


# --- budget -------------------------------------------------------------------


def test_budget_exhaustion_returns_an_error_not_an_empty_result(monkeypatch):
    _stub_provider(monkeypatch, _raw(1))
    for _ in range(WU.max_searches_per_turn()):
        assert "error" not in W.run_web_search("dams")
    spent = W.run_web_search("dams")
    assert spent["count"] == 0
    # Crucially distinguishable from "found nothing": the model must not report a spent budget as
    # an absence of sources.
    assert "budget exhausted" in spent["error"]


def test_a_spent_budget_says_it_is_final_for_the_turn(monkeypatch):
    """Stage 82: the refusal carries a code and says no retry this turn can change it, so the
    turn log can stop the next call instead of matching the message."""
    monkeypatch.setenv("AGENT_WEB_MAX_SEARCHES_PER_TURN", "0")
    spent = W.run_web_search("dams")
    assert spent["reason"] == "budget_exhausted"
    assert spent["retryable"] is False and spent["refused_for"] == "turn"
    monkeypatch.setenv("AGENT_WEB_ENABLED", "false")
    off = W.run_web_search("dams")
    assert off["reason"] == "web_disabled" and off["refused_for"] == "turn"


def test_the_tool_passes_the_refusal_fields_to_the_model(monkeypatch):
    from agent_runtime.langchain_granular_tools import web_search_tool

    monkeypatch.setenv("AGENT_WEB_MAX_SEARCHES_PER_TURN", "0")
    payload = json.loads(web_search_tool("dams"))
    assert payload["reason"] == "budget_exhausted"
    assert payload["retryable"] is False and payload["refused_for"] == "turn"


def test_begin_turn_resets_the_budget(monkeypatch):
    _stub_provider(monkeypatch, _raw(1))
    for _ in range(WU.max_searches_per_turn()):
        W.run_web_search("dams")
    assert "error" in W.run_web_search("dams")
    WU.begin_turn()
    assert "error" not in W.run_web_search("dams")


def test_kill_switch_blocks_the_network(monkeypatch):
    called = {"n": 0}

    def boom(q, *, limit=6, recency_days=None):
        called["n"] += 1
        return []

    monkeypatch.setitem(W._PROVIDERS, "duckduckgo", boom)
    monkeypatch.setenv("AGENT_WEB_ENABLED", "false")
    result = W.run_web_search("dams")
    assert "disabled" in result["error"]
    assert called["n"] == 0          # refused before any request went out


def test_provider_outage_degrades_to_an_error_payload(monkeypatch):
    def boom(q, *, limit=6, recency_days=None):
        raise RuntimeError("engine unreachable")

    monkeypatch.setitem(W._PROVIDERS, "duckduckgo", boom)
    result = W.run_web_search("dams")
    assert result["count"] == 0 and "unreachable" in result["error"]


def test_limit_is_capped_regardless_of_what_the_model_asks_for(monkeypatch):
    calls = _stub_provider(monkeypatch, _raw(1))
    W.run_web_search("dams", limit=500)
    assert calls["limit"] == 10


# --- provenance ---------------------------------------------------------------


def test_surfaced_urls_become_citable_in_both_raw_and_canonical_form(monkeypatch):
    monkeypatch.setitem(
        W._PROVIDERS, "duckduckgo",
        lambda q, *, limit=6, recency_days=None: [
            WU.WebHit(url="https://example.org/a?utm_source=x", title="A", snippet="dams",
                      provider="p", rank=1)
        ],
    )
    W.run_web_search("dams")
    allowed = WU.allowed_urls()
    assert "https://example.org/a?utm_source=x" in allowed   # as the model sees it
    assert "https://example.org/a" in allowed                # as we key it


def test_a_fabricated_download_link_is_stripped_while_a_real_one_survives(monkeypatch):
    from agent_runtime.runtime_utils import sanitize_answer_links

    monkeypatch.setitem(
        W._PROVIDERS, "duckduckgo",
        lambda q, *, limit=6, recency_days=None: [
            WU.WebHit(url="https://data.example.gov/dams.csv", title="Dams", snippet="dams",
                      provider="p", rank=1)
        ],
    )
    W.run_web_search("dams")
    answer = (
        "See [Download the dam inventory](https://data.example.gov/dams.csv) and "
        "[Download the summary](https://totally-invented.example.net/summary.csv)."
    )
    cleaned = sanitize_answer_links(answer, allowed_file_ids=[], allowed_urls=WU.allowed_urls())
    assert "https://data.example.gov/dams.csv" in cleaned
    assert "totally-invented" not in cleaned


# --- the internal hit envelope ------------------------------------------------


def test_hits_ride_the_existing_envelope_so_citations_and_links_work(monkeypatch):
    from agent_runtime.langchain_granular_tools import _landing_url, _normalize_hits

    _stub_provider(monkeypatch, _raw(2))
    hits = W.results_to_hits(W.run_web_search("dams"))
    docs = _normalize_hits(hits, source="web")
    assert len(docs) == 2
    assert docs[0]["element_type"] == "web"
    assert docs[0]["url"] == "https://example.org/dams/1"
    assert _landing_url(hits[0]["_source"]) == "https://example.org/dams/1"


def test_web_documents_link_to_the_page_not_a_fabricated_platform_route():
    """element_type 'web' must not be pluralized into {FRONTEND_DOMAIN}/webs/<id>."""
    from agent_runtime.supervisor.evidence_subgraph import _element_url

    doc = {"doc_id": "web-abc123", "element_type": "web", "title": "T",
           "url": "https://example.org/page"}
    assert _element_url(doc) == "https://example.org/page"


def test_tool_payload_reports_a_refusal_instead_of_a_silent_zero(monkeypatch):
    from agent_runtime.langchain_granular_tools import web_search_tool

    _stub_provider(monkeypatch, _raw(1))
    payload = json.loads(web_search_tool("dams"))
    assert payload["source"] == "web" and payload["count"] == 1
    assert payload["citation_ids"] == [payload["documents"][0]["doc_id"]]
    assert payload["budget"]["searches"] == 1

    monkeypatch.setenv("AGENT_WEB_ENABLED", "false")
    refused = json.loads(web_search_tool("dams"))
    assert refused["count"] == 0 and "disabled" in refused["error"]


def test_web_search_is_gateable_per_request():
    from agent_runtime.search_methods import normalize_search_methods

    assert normalize_search_methods(["web"]) == ["web_search"]
    assert normalize_search_methods("internet, openweb") == ["web_search"]
    assert normalize_search_methods(["keyword_search"]) == ["keyword_search"]


def test_web_search_stays_out_of_the_every_turn_direct_sweep(monkeypatch):
    """The sweep unions cheap in-house methods on EVERY turn; the open web must not join it."""
    from agent_runtime.supervisor import graph as G

    def boom(*a, **k):
        raise AssertionError("the direct sweep must never touch the open web")

    monkeypatch.setattr(W, "run_web_search", boom)
    monkeypatch.setattr(W, "get_web_search_results", boom)
    G._direct_search_sweep("find open datasets about dams in Illinois", ["keyword_search"])
