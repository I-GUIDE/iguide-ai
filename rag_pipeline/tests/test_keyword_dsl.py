"""Keyword search matches TITLE as well as contents (Stage 86).

"Chicago Crime data 2026" is found by its title on the platform's own UI, but its contents never
say Chicago, crime or data, so a `match` on `contents` alone could not find it. Offline: the
OpenSearch client is a stub that records the body it was sent.
"""

from __future__ import annotations

from rag_pipeline.search import keyword as kw


class _Client:
    def __init__(self, hits=None):
        self.bodies = []
        self.hits = hits or []

    def search(self, index, body):
        self.bodies.append(body)
        return {"hits": {"hits": self.hits}}


def test_dsl_matches_title_boosted_and_contents():
    clause = kw.keyword_query_dsl("Chicago crime data")
    mm = clause["multi_match"]
    assert mm["query"] == "Chicago crime data"
    assert "title^3" in mm["fields"] and "contents" in mm["fields"]
    assert "match" not in clause


def test_search_sends_the_multi_match_body():
    client = _Client()
    kw.get_keyword_search_results("Chicago crime data", size=5, client=client, index="idx")
    (body,) = client.bodies
    assert body["size"] == 5
    assert body["query"] == kw.keyword_query_dsl("Chicago crime data")


def test_title_only_hit_is_returned_normalised():
    hit = {"_id": "e1", "_score": 7.5,
           "_source": {"title": "Chicago Crime data 2026", "contents": "Incident points, CSV."}}
    out = kw.get_keyword_search_results("Chicago crime", client=_Client([hit]), index="idx")
    assert [h["_id"] for h in out] == ["e1"]
    assert out[0]["_source"]["title"] == "Chicago Crime data 2026"


def test_blank_query_sends_nothing():
    client = _Client()
    assert kw.get_keyword_search_results("   ", client=client, index="idx") == []
    assert client.bodies == []
