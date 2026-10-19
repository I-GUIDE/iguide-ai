"""A refusal no retry can change is given once, and a retry cannot get past a budget (stage 82).

Live, 2026-10-16 (agent.i-guide.io, sess-da4493c7, Lumen deepseek-v4-flash, "What direction
would you travel from BPHC site BPS-H80-031470 to BPHC site BPS-H80-017740"): the trace read
"retrying web_search after: web search budget exhausted for this turn (3/3)" again and again,
and "retrying overpass_search after: no_location". Nothing retried those calls but the model:
each new query was a new call key, so the turn log's identical-call memo never matched, and an
identical failure was run once more on the chance it was an outage. The tools now say, in
fields, when a refusal is final (`retryable: false`) and when it holds for every call of the
tool this turn (`refused_for: "turn"`), and the turn log answers those calls without running them.

These drive the REAL create_agent with the real middleware stack and the real web/Overpass
refusals, inside a supervisor turn (`turn_log.bind`). No network is touched: a spent budget
and a missing location are both refused before any request.
"""
from __future__ import annotations

import json

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

import rag_pipeline.search.web_fetch as WF
import rag_pipeline.search.web_utils as WU
from agent_runtime import executor_factory as ef
from agent_runtime import turn_log
from agent_runtime.langchain_granular_tools import web_fetch_tool, web_search_tool


class ToolAwareFake(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
        return self


RAN = {"web_search": 0, "web_fetch": 0, "overpass_search": 0, "flaky": 0}


@tool("web_search")
def web_search(query: str) -> str:
    """Search the open web."""
    RAN["web_search"] += 1
    return web_search_tool(query)


@tool("web_fetch")
def web_fetch(url: str) -> str:
    """Read one web page."""
    RAN["web_fetch"] += 1
    return web_fetch_tool(url)


@tool("overpass_search")
def overpass(feature: str, place: str = "", bbox: str = "") -> str:
    """Find OSM features."""
    from rag_pipeline.search.overpass import overpass_search

    RAN["overpass_search"] += 1
    return json.dumps(overpass_search(feature, place=place or None, bbox=bbox or None))


@tool("flaky")
def flaky(feature: str) -> str:
    """Fails like an outage the first time."""
    RAN["flaky"] += 1
    if RAN["flaky"] == 1:
        return '{"error": "overpass_failed", "message": "504 Gateway Timeout", "count": 0}'
    return '{"count": 56}'


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    for var in ("AGENT_WEB_ENABLED", "AGENT_WEB_MAX_SEARCHES_PER_TURN",
                "AGENT_WEB_MAX_FETCHES_PER_TURN"):
        monkeypatch.delenv(var, raising=False)
    for k in RAN:
        RAN[k] = 0
    WF._CACHE.clear()
    WU.begin_turn()


def _call(name, args, i):
    return {"name": name, "args": args, "id": f"call_{i}", "type": "tool_call"}


def _run(calls, tools=(web_search, web_fetch, overpass, flaky), log=None):
    log = log or turn_log.new_log()
    agent = create_agent(
        model=ToolAwareFake(messages=iter(
            [AIMessage(content="", tool_calls=[c]) for c in calls]
            + [AIMessage(content="done")])),
        tools=list(tools), system_prompt="t", middleware=ef._default_middleware())
    with turn_log.bind(log, "search"):
        out = agent.invoke({"messages": [HumanMessage(content="q")]})
    return [m for m in out["messages"] if isinstance(m, ToolMessage)], log


def test_a_budget_refused_search_is_not_run_again_with_a_new_query(monkeypatch):
    """The incident's shape: the budget is spent, and the model tries query after query."""
    charged = {"n": 0}
    real = WU.charge_search

    def counting():
        charged["n"] += 1
        return real()

    monkeypatch.setattr(WU, "charge_search", counting)
    monkeypatch.setenv("AGENT_WEB_MAX_SEARCHES_PER_TURN", "0")
    queries = ["BPS-H80-031470", "BPHC site BPS-H80-031470 address",
               "HRSA health center BPS-H80-017740", "BPS-H80-017740 location"]
    msgs, log = _run([_call("web_search", {"query": q}, i) for i, q in enumerate(queries, 1)])

    assert RAN["web_search"] == 1 and charged["n"] == 1
    first, *rest = msgs
    assert json.loads(first.content)["reason"] == "budget_exhausted"
    assert rest, "the later calls still get an answer"
    for m in rest:
        assert "refused for the rest of this turn" in m.content
        assert m.additional_kwargs.get("repeat_of") == "call_1"
    # Answered from the log, so they add nothing: the progress guard ends the run instead of
    # letting it spend a step per query.
    assert len(rest) == turn_log.PROGRESS_LIMIT


def test_a_retried_fetch_cannot_read_past_the_budget(monkeypatch):
    """The second incident line: a page another turn had read came back from the cache after
    this turn's budget was spent. Through the agent loop, the page is refused and the calls
    after it are not run."""
    requested = []

    class _Resp:
        status_code = 200
        headers = {"Content-Type": "text/html"}

        def iter_content(self, size):
            return iter([b"<html><body><p>BPHC site</p></body></html>"])

        def close(self):
            pass

    class _S:
        def get(self, url, **kwargs):
            requested.append(url)
            return _Resp()

    monkeypatch.setattr(WF, "fetch_session", lambda: _S())
    monkeypatch.setattr(WF, "validate_url", lambda url: None)     # no DNS in this test

    WF.fetch_and_extract("https://data.example.gov/site")          # an earlier turn read it
    WU.begin_turn()
    limit = WU.max_fetches_per_turn()
    urls = [f"https://data.example.gov/p{i}" for i in range(limit)]
    urls += ["https://data.example.gov/site", "https://data.example.gov/site?x=1",
             "https://data.example.gov/other"]
    msgs, _ = _run([_call("web_fetch", {"url": u}, i) for i, u in enumerate(urls, 1)])

    assert WU.budget().fetches == limit
    assert all("error" not in json.loads(m.content) for m in msgs[:limit])
    refused = json.loads(msgs[limit].content)
    assert refused["reason"] == "budget_exhausted" and "text" not in refused
    assert msgs[limit + 1:], "the later calls still get an answer"
    assert all("refused for the rest of this turn" in m.content for m in msgs[limit + 1:])
    assert RAN["web_fetch"] == limit + 1
    assert len(requested) == 1 + limit                         # nothing fetched past the budget


def test_an_identical_no_location_call_is_not_retried_even_once():
    """`no_location` is final for its arguments (a geocoding miss is cached for the process),
    so the one retry an outage gets is not given to it."""
    same = {"feature": "school", "place": "", "bbox": ""}
    msgs, _ = _run([_call("overpass_search", same, i) for i in range(1, 4)])
    assert RAN["overpass_search"] == 1
    assert json.loads(msgs[0].content)["error"] == "no_location"
    for m in msgs[1:]:
        assert "Same call" in m.content and "final for these arguments" in m.content


def test_a_no_location_call_with_a_location_added_still_runs(monkeypatch):
    """`no_location` closes the CALL, not the tool: the model fixing its arguments is the way
    out, and it must be allowed."""
    from rag_pipeline.search import overpass as ov

    monkeypatch.setattr(ov, "overpass_search",
                        lambda feature, place=None, bbox=None, **k: (
                            {"error": "no_location", "count": 0, "features": [], "retryable": False}
                            if not (place or bbox) else {"count": 0, "features": []}))
    msgs, _ = _run([_call("overpass_search", {"feature": "school"}, 1),
                    _call("overpass_search", {"feature": "school", "bbox": "-88,40,-87,41"}, 2)])
    assert RAN["overpass_search"] == 2
    assert "Same call" not in msgs[1].content


def test_a_retryable_failure_still_gets_its_one_retry():
    """An outage is not a refusal: the identical call runs once more, as before."""
    msgs, _ = _run([_call("flaky", {"feature": "school"}, 1),
                    _call("flaky", {"feature": "school"}, 2)])
    assert RAN["flaky"] == 2 and msgs[-1].content == '{"count": 56}'


def test_refusal_fields_are_read_from_fields_not_prose():
    budget = turn_log.parse_result('{"error": "web search budget exhausted for this turn (3/3)"}')
    assert budget["retryable"] is True and budget["refused_for"] is None
    marked = turn_log.parse_result(json.dumps(WU.Refusal("anything", "budget_exhausted").fields()))
    assert marked["retryable"] is False and marked["refused_for"] == "turn"
    # A success is never final, whatever it carries.
    ok = turn_log.parse_result('{"ok": true, "retryable": false, "refused_for": "turn"}')
    assert ok["retryable"] is True and ok["refused_for"] is None


def test_a_refusal_closes_only_its_own_tool():
    log = turn_log.new_log()
    log.record_result(peer="search", run=None, name="web_search", args={"query": "a"},
                      call_id="c1", content=json.dumps(WU.Refusal("spent", "budget_exhausted").fields()))
    assert log.refusal_for_turn("web_search")["call_id"] == "c1"
    assert log.refusal_for_turn("web_fetch") is None
