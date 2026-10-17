"""The agent can see why a tool answered as it did, and ask Overpass its own question.

Live, 2026-10-14 (MultiGlobeQA, "Is Basco larger in area than 1st District of Palawan?"): the
boundary query returned 158 features and no error, all but one of them unnamed pieces of
outlines. The agent could neither see that nor check it another way, so it routed around the
tool to Wikipedia. Two things change:

* ``overpass_search`` always returns ``diagnostics`` (what the result is made of, and warnings),
  and on ``debug=True`` the Overpass QL it sent, every mirror's answer and the raw response.
* The code bridge gains ``overpass_query(ql)``: the code's own read-only Overpass query, sent by
  the agent to the configured mirrors with JSON output and capped cost, so a hypothesis about
  the tool can be tested against the service directly.

Offline: ``requests.post`` is patched; nothing leaves the process.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_runtime import tool_bridge
from rag_pipeline.search import overpass as ov


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for name in ("AGENT_CODE_BRIDGE_MAX_CALLS", "AGENT_CODE_BRIDGE_MAX_RAW_QUERIES",
                 "AGENT_CODE_BRIDGE_RAW_MAX_MB"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "store"))
    monkeypatch.setattr(ov, "OVERPASS_ENDPOINTS", ["https://a/api", "https://b/api"])


class _Resp:
    def __init__(self, payload, status=200):
        self.status_code, self._payload = status, payload
        self.text = json.dumps(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"{self.status_code} Server Error")

    def json(self):
        return self._payload


def _patch(monkeypatch, responses):
    sent = []

    def fake_post(url, data=None, headers=None, timeout=None):
        sent.append({"url": url, "query": data["data"], "timeout": timeout})
        return responses.pop(0)

    monkeypatch.setattr(ov.requests, "post", fake_post)
    return sent


def _piece(i):
    return {"type": "way", "id": i, "tags": {"boundary": "administrative"},
            "geometry": [{"lon": 0.0, "lat": 0.0}, {"lon": 0.1, "lat": 0.1}]}


# --------------------------------------------------------------------------- diagnostics

def test_unnamed_pieces_are_flagged_in_every_result(monkeypatch):
    _patch(monkeypatch, [_Resp({"elements": [_piece(i) for i in range(12)]})])
    res = ov.overpass_search("boundary=administrative", bbox="0,0,1,1")
    d = res["diagnostics"]
    assert d["features"] == 12 and d["named"] == 0
    assert d["elements_by_type"] == {"way": 12} and d["geometry_types"] == {"LineString": 12}
    assert "none of the 12 features has a name" in d["warnings"][0]
    assert "debug" not in res                                   # only when asked


def test_a_name_that_matches_nothing_says_so(monkeypatch):
    _patch(monkeypatch, [_Resp({"elements": []}), _Resp({"elements": []})])
    res = ov.overpass_search("Bascoo", bbox="0,0,1,1")
    assert res["count"] == 0 and "matching 'Bascoo'" in res["diagnostics"]["warnings"][0]


# --------------------------------------------------------------------------- debug

def test_debug_shows_the_query_every_mirror_and_the_raw_answer(monkeypatch):
    timeout = {"remark": 'runtime error: Query timed out in "query" at line 1', "elements": []}
    _patch(monkeypatch, [_Resp(timeout), _Resp({"elements": [_piece(1)]})])
    res = ov.overpass_search("boundary=administrative", bbox="0,0,1,1", debug=True)
    (ex,) = res["debug"]["exchanges"]
    assert "relation[boundary=administrative]" in ex["query"]
    first, second = ex["attempts"]
    assert first["endpoint"] == "https://a/api" and "timed out" in first["remark"]
    assert second["endpoint"] == "https://b/api" and "error" not in second
    assert ex["endpoint"] == "https://b/api" and '"elements"' in ex["raw_excerpt"]


def test_debug_on_a_failure_keeps_the_evidence(monkeypatch):
    _patch(monkeypatch, [_Resp({}, 504), _Resp({}, 429)])
    res = ov.overpass_search("boundary", bbox="0,0,1,1", debug=True)
    assert res["error"] == "overpass_failed"
    statuses = [a["status"] for a in res["debug"]["exchanges"][0]["attempts"]]
    assert statuses == [504, 429]


def test_the_model_tool_passes_debug_and_always_shows_diagnostics(monkeypatch):
    from agent_runtime import langchain_granular_tools as G

    _patch(monkeypatch, [_Resp({"elements": [_piece(1)]}), _Resp({"elements": [_piece(1)]})])
    plain = json.loads(G.overpass_search_tool("boundary", bbox="0,0,1,1"))
    assert "diagnostics" in plain and "debug" not in plain
    dbg = json.loads(G.overpass_search_tool("boundary", bbox="0,0,1,1", debug=True))
    assert dbg["debug"]["exchanges"][0]["attempts"][0]["endpoint"] == "https://a/api"


# --------------------------------------------------------------------------- raw query rules

def test_a_raw_query_is_json_with_capped_cost_and_settings_first():
    q = ov.prepare_raw_query('[out:json][timeout:600][maxsize:99999999999];\nrelation(1);out geom;')
    assert q.startswith(f"[out:json][timeout:{ov.RAW_TIMEOUT_MAX_S}][maxsize:{ov.RAW_MAXSIZE_MAX}];")
    assert q.count("[timeout:") == 1 and q.endswith("relation(1);out geom;")
    bare = ov.prepare_raw_query('relation["name"="Basco"](20,121,21,122); out tags;')
    assert bare.startswith("[out:json][timeout:") and 'relation["name"="Basco"]' in bare


@pytest.mark.parametrize("ql, why", [
    ("[out:xml];node(1);out;", "[out:xml] is not allowed"),
    ("", "empty"),
    ("node(1);out;" + " " * 10 + "x" * 5000, "characters"),
    ("[timeout:soon];node(1);out;", "whole numbers"),
])
def test_a_raw_query_that_breaks_a_rule_is_refused(ql, why):
    with pytest.raises(ov.RawQueryRefused, match=why.replace("[", r"\[").replace("]", r"\]")):
        ov.prepare_raw_query(ql)


def test_the_http_wait_covers_the_server_timeout(monkeypatch):
    sent = _patch(monkeypatch, [_Resp({"elements": []})])
    ov.raw_query("[timeout:80];node(1);out;")
    assert sent[0]["timeout"] == 90


def test_an_oversized_answer_is_a_mirror_failure(monkeypatch):
    big = {"elements": [_piece(i) for i in range(200)]}
    _patch(monkeypatch, [_Resp(big), _Resp(big)])
    ex = ov.raw_query("way(1);out;", max_bytes=1000)
    assert ex["data"] is None and "cap" in ex["error"]


# --------------------------------------------------------------------------- the bridge

def _server(tmp_path, **kw):
    srv = tool_bridge.BridgeServer(tmp_path / "bridge", container_root="/bridge", record=False,
                                   **kw)
    srv.prepare()
    return srv


def _request(srv, name, rid="a" * 32, **args):
    (srv.inbox / f"{rid}.req").write_text(json.dumps({"id": rid, "name": name, "args": args}))
    srv.serve_pending()
    return json.loads((srv.out / "resp" / f"{rid}.json").read_text())


def test_overpass_query_through_the_bridge_hands_back_the_raw_answer(tmp_path, monkeypatch):
    rel = {"type": "relation", "id": 1513433, "tags": {"name": "Basco"}}
    _patch(monkeypatch, [_Resp({}, 504), _Resp({"elements": [rel, _piece(2)]})])
    srv = _server(tmp_path)
    ans = _request(srv, "overpass_query", query='relation["name"="Basco"](20,121,21,122);out tags;')
    assert ans["ok"], ans
    r = ans["result"]
    assert r["elements"] == 2 and r["elements_by_type"] == {"relation": 1, "way": 1}
    assert r["endpoint"] == "https://b/api" and r["attempts"][0]["status"] == 504
    assert r["query"].startswith("[out:json]")
    local = srv.out / Path(r["path"]).relative_to("/bridge/out")
    assert json.loads(local.read_text())["elements"][0]["id"] == 1513433


def test_the_bridge_refuses_a_raw_query_that_breaks_a_rule(tmp_path):
    ans = _request(_server(tmp_path), "overpass_query", query="[out:csv(name)];node(1);out;")
    assert ans["ok"] is False and "[out:csv(name)] is not allowed" in ans["error"]


def test_raw_queries_have_their_own_per_run_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_CODE_BRIDGE_MAX_RAW_QUERIES", "1")
    _patch(monkeypatch, [_Resp({"elements": []}), _Resp({"elements": []})])
    srv = _server(tmp_path)
    assert _request(srv, "overpass_query", query="node(1);out;")["ok"]
    assert "at most 1 raw Overpass" in _request(srv, "overpass_query", rid="b" * 32,
                                                query="node(2);out;")["error"]


def test_a_raw_query_no_mirror_answers_is_refused_with_the_attempts(tmp_path, monkeypatch):
    _patch(monkeypatch, [_Resp({}, 504), _Resp({}, 500)])
    ans = _request(_server(tmp_path), "overpass_query", query="node(1);out;")
    assert ans["ok"] is False and "504" in ans["error"] and "https://b/api" in ans["error"]


def test_bridge_overpass_search_passes_debug_and_reports_diagnostics(monkeypatch):
    seen = []

    def fake(feature, place=None, bbox=None, limit=500, max_limit=500, debug=False):
        seen.append(debug)
        return {"query": {"osm_filter": "boundary"}, "count": 1,
                "features": [{"osm_type": "way", "osm_id": 1, "name": "(unnamed)",
                              "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]}}],
                "debug": {"exchanges": [{"query": "Q", "attempts": []}]}}

    monkeypatch.setattr(ov, "overpass_search", fake)
    out = tool_bridge._overpass_handler("boundary", bbox=[0, 0, 1, 1], debug=True)
    assert seen == [True]
    assert out["debug"]["tiles"][0]["exchanges"][0]["query"] == "Q"
    assert out["diagnostics"]["named"] == 0 and out["diagnostics"]["warnings"]


def test_the_client_offers_overpass_query_and_the_note_says_when_to_use_it(monkeypatch):
    assert "def overpass_query(query)" in tool_bridge.CLIENT_SOURCE
    assert "overpass_query" in tool_bridge.DEFAULT_HANDLERS
    note = tool_bridge.model_note()
    assert "overpass_query(ql)" in note and "compare the two" in note
