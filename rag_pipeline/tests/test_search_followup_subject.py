"""Stage 86: a method-only follow-up must not become the relevance query.

The trace this pins (2026-10-06, deepseek-v4-flash): turn 1 asked for Chicago crime data; turn 2
said "Use I-GUIDE keyword search only". The search peer called keyword_search('Chicago crime
data') and got the eight Chicago documents, then the deterministic sweep searched the literal
sentence and the rerank judged everything against it, so the eight kept were the search-tutorial
pages and the answer said there were no crime datasets.

Offline: the peer, the OpenSearch backend and the rerank model are all stubbed. The stub reranker
scores by word overlap with the query it is GIVEN, so it reproduces the failure exactly when it
is handed the literal message.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

from agent_runtime.supervisor import graph as G
from agent_runtime.supervisor_graph import run_supervisor

TOPIC = "Find Chicago crime data"
FOLLOW_UP = "Use I-GUIDE keyword search only"


def _hit(doc_id, title, contents):
    return {"_id": doc_id, "_score": 1.0, "_source": {"title": title, "contents": contents}}


CHICAGO = [_hit(f"chi{i}", f"Chicago crime incidents {2018 + i}",
                f"Reported crime incidents in the city of Chicago, year {2018 + i}.")
           for i in range(8)]
TUTORIALS = [_hit(f"tut{i}", f"How to use I-GUIDE keyword search, part {i}",
                  "A guide to keyword search only mode and other search options on I-GUIDE.")
             for i in range(8)]


SWEPT = []


def _backend(query, size=None, **_):
    """The keyword index: the literal sentence finds the tutorials, the subject finds Chicago."""
    SWEPT.append(str(query))
    return list(TUTORIALS if "keyword" in str(query).lower() else CHICAGO)


def _overlap_reranker(prompt: str) -> str:
    """A relevance judge that ranks by word overlap with the query it is handed."""
    if "relevance judge" not in prompt.lower():
        return "x"
    query = prompt.split("User query:\n", 1)[1].split("\n\nCandidate documents:", 1)[0]
    terms = {t for t in re.findall(r"[a-z]+", query.lower()) if len(t) > 3}
    ranking = []
    for block in prompt.split("Candidate documents:\n", 1)[1].split("\n\n"):
        m = re.search(r"doc_id=(\S+)\ntitle=(.*)", block)
        if m:
            words = set(re.findall(r"[a-z]+", m.group(2).lower()))
            ranking.append({"doc_id": m.group(1), "score": len(terms & words) / 10})
    ranking.sort(key=lambda r: -r["score"])
    return json.dumps({"ranking": ranking})


class _PeerSession:
    """The search peer: one keyword_search call on the subject it resolved."""

    def __init__(self, peer_query):
        self.peer_query = peer_query
        self.prompts = []

    def run(self, prompt, **_):
        from agent_runtime.langchain_granular_tools import _build_payload

        self.prompts.append(prompt)
        call = {"name": "keyword_search", "args": {"query": self.peer_query}, "id": "c1"}
        result = {"name": "keyword_search", "tool_call_id": "c1",
                  "content": _build_payload(_backend(self.peer_query), source="keyword")}
        return SimpleNamespace(artifacts={"tool_calls": [call], "tool_results": [result]},
                               answer="", resp={})


@pytest.fixture
def stubbed(monkeypatch):
    import agent_runtime.executor_factory as ef
    import rag_pipeline.search.keyword as kw

    monkeypatch.setenv("AGENT_EXTRACTION", "0")
    SWEPT.clear()
    monkeypatch.setattr(kw, "get_keyword_search_results", _backend)
    monkeypatch.setattr(ef, "build_search_agent_executor", lambda **_: object())
    sessions = []

    def use_peer(peer_query):
        session = _PeerSession(peer_query)
        sessions.append(session)
        monkeypatch.setattr(ef, "open_peer_session", lambda *a, **k: session)
        return session

    return use_peer


def _turn(query, history):
    search = G.default_search_fn(llm=_overlap_reranker, enabled_search_methods=["keyword_search"])
    return run_supervisor(
        query, chat_history=history, llm=_overlap_reranker,
        decide_fn=lambda s, d: "done" if s.get("search_attempts") else "search",
        search_fn=search, synthesize_fn=lambda *a, **k: "answer",
        enabled_search_methods=["keyword_search"], do_audit=False, top_k=8,
    )


def test_method_only_follow_up_keeps_the_peers_hits(stubbed):
    stubbed("Chicago crime data")
    first = _turn(TOPIC, [])
    assert {d["doc_id"] for d in first["evidence"]} == {f"chi{i}" for i in range(8)}

    history = [{"role": "user", "content": TOPIC},
               {"role": "assistant", "content": "Here are eight Chicago crime datasets."}]
    stubbed("Chicago crime data")
    SWEPT.clear()
    second = _turn(FOLLOW_UP, history)
    kept = {d["doc_id"] for d in second["evidence"]}
    # Before Stage 86: the sweep searched the sentence, the rerank judged against it, and the
    # eight kept were tut0..tut7.
    assert kept == {f"chi{i}" for i in range(8)}
    # The peer's own call plus the sweep; none of them on the instruction.
    assert SWEPT and not any("keyword search only" in q.lower() for q in SWEPT)


def test_follow_up_resolves_from_history_when_the_peer_searched_the_sentence(stubbed):
    """A peer as literal as the old sweep must not set the subject either."""
    stubbed(FOLLOW_UP)
    history = [{"role": "user", "content": TOPIC},
               {"role": "assistant", "content": "Here are eight Chicago crime datasets."}]
    second = _turn(FOLLOW_UP, history)
    assert {d["doc_id"] for d in second["evidence"]} >= {f"chi{i}" for i in range(4)}
    assert not any(d["doc_id"].startswith("tut") for d in second["evidence"][:4])


# --- the resolver itself ---------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Use I-GUIDE keyword search only", "only datasets", "try semantic search instead",
    "just notebooks please", "search again",
])
def test_method_only_messages(text):
    assert G._is_method_only(text)


@pytest.mark.parametrize("text", [
    "Chicago crime data", "use keyword search for flood maps in Iowa", "notebooks about wildfire",
])
def test_messages_that_name_a_subject(text):
    assert not G._is_method_only(text)


def test_resolver_skips_earlier_meta_turns_and_assistant_text():
    history = [{"role": "user", "content": "flood risk in Houston"},
               {"role": "assistant", "content": "Chicago crime results"},
               {"role": "user", "content": "only datasets"}]
    assert G._resolve_search_subject(FOLLOW_UP, history) == "flood risk in Houston"
    assert G._resolve_search_subject("wildfire smoke", history) == "wildfire smoke"
    assert G._resolve_search_subject(FOLLOW_UP, []) is None


def test_peer_queries_are_distinct_subjects_in_call_order():
    arts = {"tool_calls": [
        {"name": "keyword_search", "args": {"query": "Chicago crime data"}},
        {"name": "semantic_search", "args": {"query": "chicago crime DATA"}},
        {"name": "web_fetch", "args": {"query": "https://example.org"}},
        {"name": "keyword_search", "args": {"query": FOLLOW_UP}},
        {"name": "spatial_search", "args": {"query": "crime near Chicago"}},
    ]}
    assert G._peer_search_queries(arts) == ["Chicago crime data", "crime near Chicago"]


def test_search_subjects_prefers_what_was_searched_for():
    history = [{"role": "user", "content": TOPIC}]
    sweep, rq = G._search_subjects(FOLLOW_UP, history, ["Chicago crime data"])
    assert sweep == ["Chicago crime data", TOPIC]
    assert rq.startswith(TOPIC) and "Chicago crime data" in rq and "keyword search" not in rq
    # nothing better known -> exactly the old behaviour
    assert G._search_subjects(FOLLOW_UP, [], []) == ([FOLLOW_UP], FOLLOW_UP)
    # a content-bearing message is its own subject
    assert G._search_subjects("dams in Illinois", [], []) == (["dams in Illinois"],
                                                              "dams in Illinois")
