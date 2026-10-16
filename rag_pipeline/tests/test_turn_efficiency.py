"""Four causes of one slow turn (Chicago demo, the SKATER step, 2026-10-15).

Analyze drew the regions at 14:37:38 and finished at 14:38:42. The supervisor then routed to
search, which read 45 of the conversation's own files over 3 min 16 s (prompt 8k → 56k tokens)
and added nothing. Along the way every failed execute_code showed a library warning instead of
its error, and choropleth_image failed on GeoJSON.

1. In the search peer, the conversation's own outputs are not read; uploads are, at most
   AGENT_SEARCH_FILE_READS distinct files per run, each once.
2. A search after the map is delivered, on a request that asks for nothing retrieval supplies,
   is vetoed; and the visible trace line carries the decider's reason.
3. A failure's trace line names the exception, not the warning printed before it.
4. choropleth_image reads GeoJSON.
"""
from __future__ import annotations

import io
import json

import pytest
from werkzeug.datastructures import FileStorage

from agent_runtime import file_store, identity, turn_log
from agent_runtime import langchain_file_tools as lft
from agent_runtime.streaming_trace import _outcome
from agent_runtime.supervisor import graph as sg


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("AGENT_PUBLIC_BASE_URL", "")
    monkeypatch.delenv("UPLOAD_FOLDER", raising=False)
    lft._SEARCH_READS.clear()
    return tmp_path


@pytest.fixture()
def conversation():
    ut = identity.set_user(identity.User(id="u1", role=4))
    st = file_store.set_session("sess-1")
    yield
    file_store.reset_session(st)
    identity.reset_user(ut)


def _output(name, body="a,b\n1,2\n"):
    return file_store.create_output_file(name, body)


def _upload(name, body=b"a,b\n1,2\n"):
    return file_store.save_uploaded_file(FileStorage(stream=io.BytesIO(body), filename=name))


def _read(ref):
    return json.loads(lft.read_text_file_tool(ref))


# --- 1. the search peer does not re-read the turn's own work ------------------------------

def test_outside_the_search_peer_an_output_is_read(conversation):
    rec = _output("rates.csv")
    assert "content" in _read(rec["file_id"])


def test_the_search_peer_does_not_read_the_conversations_own_output(conversation):
    rec = _output("rates.csv")
    with turn_log.bind(turn_log.new_log(), "search"):
        got = _read(rec["file_id"])
        inspected = json.loads(lft.inspect_file_for_analysis_tool(rec["file_id"]))
    assert got["ok"] is False and "own analysis" in got["error"]
    assert inspected["ok"] is False


def test_the_search_peer_reads_an_upload_once_and_at_most_the_budget(conversation, monkeypatch):
    recs = [_upload(f"doc{i}.csv") for i in range(lft.SEARCH_READ_BUDGET + 1)]
    with turn_log.bind(turn_log.new_log(), "search"):
        first = _read(recs[0]["file_id"])
        again = _read(recs[0]["file_id"])
        rest = [_read(r["file_id"]) for r in recs[1:]]
    assert "content" in first
    assert again["ok"] is False and "already read" in again["error"]
    assert all("content" in r for r in rest[:-1])
    assert rest[-1]["ok"] is False and "at most" in rest[-1]["error"]


def test_the_budget_is_per_run_not_per_conversation(conversation):
    rec = _upload("doc.csv")
    for _ in range(2):          # two turns: each search run has its own budget
        with turn_log.bind(turn_log.new_log(), "search"):
            assert "content" in _read(rec["file_id"])


def test_the_analyze_peer_is_not_limited(conversation):
    rec = _output("rates.csv")
    with turn_log.bind(turn_log.new_log(), "analyze"):
        for _ in range(lft.SEARCH_READ_BUDGET + 2):
            assert "content" in _read(rec["file_id"])


# --- 2. a delivered map ends the turn unless retrieval was asked for ----------------------

def _delivered():
    return {"summary": "Regions drawn.", "tool_calls": [{"name": "add_map_layer"}],
            "tool_results": [{"name": "add_map_layer", "ok": True, "content": {
                "ok": True, "map_layer": {"kind": "map_layer", "id": "r1", "name": "Regions",
                                          "render": "categorical", "bounds": [0, 0, 1, 1],
                                          "url": "https://x/agent/files/f1/download"}}}]}


SKATER = ("Group the 77 community areas into 6 contiguous regions with similar violent crime "
          "rates using SKATER. Map the regions and give each one's community areas, population "
          "and mean rate.")


def test_the_skater_step_is_done_once_its_map_is_delivered():
    assert sg._work_delivered_without_retrieval({"query": SKATER,
                                                 "analysis_results": _delivered()})


def test_retrieval_language_keeps_search_open():
    q = SKATER + " Cite the literature on SKATER."
    assert not sg._work_delivered_without_retrieval({"query": q, "analysis_results": _delivered()})


def test_no_map_no_veto():
    assert not sg._work_delivered_without_retrieval({"query": SKATER, "analysis_results": {
        "summary": "", "tool_results": [{"name": "x", "ok": False, "content": {"ok": False}}]}})
    assert not sg._work_delivered_without_retrieval({"query": SKATER})


def _run_turn(query, monkeypatch, reason="check the literature for regionalization"):
    """A whole supervisor turn whose analysis already delivered the map and whose decider
    then asks for search, as the live decider did. Returns (searched?, trace events)."""
    events, searched = [], []

    def decide(_state, _payload):
        if _state.get("actions"):
            return "done"
        d = sg.Decision("search")
        d.reason = reason
        return d

    monkeypatch.setattr(sg, "emit_trace_event",
                        lambda kind, payload, **kw: events.append((kind, payload)))
    graph = sg.build_supervisor_graph(
        search_fn=lambda q, s: searched.append(q) or [], synthesize_fn=lambda *a: "ans",
        decide_fn=decide, do_rerank=False, do_audit=False)
    graph.invoke({"query": query, "thread_id": f"t-eff-{len(query)}", "actions": [],
                  "evidence": [], "analysis_results": _delivered()})
    return bool(searched), events


def test_the_supervisor_vetoes_search_and_says_why(monkeypatch):
    searched, events = _run_turn(SKATER, monkeypatch)
    assert not searched
    assert any("the map is drawn" in str(p.get("message")) for _, p in events)


def test_the_visible_route_line_carries_the_deciders_reason(monkeypatch):
    searched, events = _run_turn(SKATER + " Cite the literature.", monkeypatch)
    assert searched
    lines = [p["message"] for k, p in events if k == "node_completed"
             and p.get("stage") == "supervisor"]
    assert "Next: search — check the literature for regionalization" in lines


def test_the_route_line_shows_the_whole_reason(monkeypatch):
    """User, 2026-10-15: the line stopped mid-sentence at 200 characters ("…plus citywide and
    highest-"). The reason is the decider's whole account of its choice; show all of it."""
    why = ("The task requires inspecting three uploaded files, joining crime counts to community "
           "area geometries and ACS population data, computing rates, and producing a choropleth "
           "map plus citywide and highest-rate figures, and the literature on SKATER is asked for.")
    assert len(why) > 200
    _, events = _run_turn(SKATER + " Cite the literature.", monkeypatch, reason=why)
    lines = [p["message"] for k, p in events if k == "node_completed"
             and p.get("stage") == "supervisor"]
    assert f"Next: search — {why}" in lines


# --- 3. a failure names its exception -----------------------------------------------------

TRACE = ("/opt/lib/libpysal/weights/contiguity.py:347: FutureWarning: `use_index` defaults to "
         "False but will default to True in future.\n  warnings.warn(\n"
         "Traceback (most recent call last):\n  File \"script.py\", line 12, in <module>\n"
         "    w = Queen.from_dataframe(gdf)\nKeyError: 'community'\n")


def test_the_exception_line_wins_over_a_leading_warning():
    out = _outcome(json.dumps({"ok": False, "exit_code": 1, "stderr": TRACE}))
    assert out.startswith("failed — KeyError: 'community' · ")
    assert "Queen.from_dataframe" in out, "the trace stays behind it, for the expanded row"


def test_a_dotted_exception_name_is_found():
    err = "Traceback (most recent call last):\n  File \"x\"\npyarrow.lib.ArrowInvalid: bad footer"
    assert _outcome(json.dumps({"ok": False, "stderr": err})).startswith(
        "failed — pyarrow.lib.ArrowInvalid: bad footer · ")


def test_a_one_line_reason_is_unchanged():
    assert _outcome(json.dumps({"ok": False, "error": "HTTP 403"})) == "failed — HTTP 403"


# --- 4. choropleth_image reads GeoJSON ----------------------------------------------------

def test_choropleth_image_reads_geojson(conversation):
    pytest.importorskip("geopandas")
    pytest.importorskip("matplotlib")
    from extractors.geo_handles import choropleth_image

    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": n, "rate": r}, "geometry": {
            "type": "Polygon", "coordinates": [[[x, 0], [x + 1, 0], [x + 1, 1], [x, 1], [x, 0]]]}}
        for x, (n, r) in enumerate([("A", 1.0), ("B", 5.0), ("C", 9.0)])]}
    rec = _output("areas.geojson", json.dumps(fc))
    out = json.loads(choropleth_image(rec["file_id"], "rate", title="Rate"))
    assert out.get("ok") is True, out
    assert out["features"] == 3 and str(out.get("file_id", "")).startswith("file_")
    missing = json.loads(choropleth_image(rec["file_id"], "nope"))
    assert "rate" in missing["error"]
