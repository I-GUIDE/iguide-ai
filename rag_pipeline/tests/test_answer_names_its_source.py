"""A list of features says where it came from, because where it came from decides what it covers.

Live, 2026-10-08 19:42 UTC on agent.i-guide.io: asked which schools lie within 1 mile of a drawn
box, the agent answered "18 schools" with distances that were right to within 4 m. The 18 came
from the Chicago Public Schools locations file on the Chicago Data Portal, and the answer never
said so. OpenStreetMap has 31 schools in the same buffer. Two of the missing ones sit inside the
box itself (Sahs Elementary, of the Central Stickney district, and Saint Camillus, a Catholic
school), and eight Catholic and three other private schools are missing as well. Read without its
source, "18 schools" is a claim about all schools; with it, it is a correct statement about one
district's file.

This is a CHECK on the composed answer, not an instruction to the answerer: the sources are read
from the tool record. Stage 38 added the line only when the answer listed features and named
none; stage 45 (agent_runtime/source_catalog.py) renders it for every answer that used data,
because whether an answer named its source depended on the model (33 of 50 runs on
deepseek-v4-flash, 18 of 51 on gpt-5.6-luna in the stage 41 baseline).
"""
from __future__ import annotations

import json

import pytest

from agent_runtime.supervisor import graph as g

CPS_URL = "https://data.cityofchicago.org/api/views/abcd-1234/rows.csv?accessType=DOWNLOAD"
CPS_TITLE = "Chicago Public Schools - School Locations SY2425 | City of Chicago | Data Portal"

LIST = """Schools within 1 mile:

| School | Distance (m) |
|---|---|
| TWAIN | 0.0 |
| PASTEUR | 343.5 |
| GLOBAL CITIZENSHIP | 452.8 |
| HEARST | 519.0 |
"""


def _code_result_from_the_portal():
    return {"answer": "x", "executed": True,
            "tool_calls": [
                {"name": "web_search", "args": {"query": "Chicago public schools locations"},
                 "id": "c1"},
                {"name": "stage_url", "args": {"url": CPS_URL}, "id": "c2"},
                {"name": "execute_code", "args": {"code": "..."}, "id": "c3"},
            ],
            "tool_results": [
                {"name": "web_search", "tool_call_id": "c1", "content": json.dumps({
                    "documents": [{"title": CPS_TITLE, "url": CPS_URL.split("?")[0]},
                                {"title": "Something else", "url": "https://example.org/x"}]})},
                {"name": "stage_url", "tool_call_id": "c2",
                 "content": json.dumps({"staged_path": "/work/inputs/rows.csv", "origin": CPS_URL})},
                {"name": "execute_code", "tool_call_id": "c3", "content": "{}"},
            ]}


def _osm_analysis_result():
    return {"summary": "x", "tool_calls": [
        {"name": "overpass_search", "args": {"feature": "school", "bbox": "..."}, "id": "a1"}],
        "tool_results": [{"name": "overpass_search", "tool_call_id": "a1", "content": json.dumps({
            "count": 31, "features": [],
            "source_statement": "OpenStreetMap features tagged amenity=school in bbox [...], as "
                                "mapped by OSM contributors: public, private and religious ones "
                                "alike wherever someone has mapped them, and not an official "
                                "register."})}]}


def _state(**slots):
    return slots


def test_a_staged_dataset_is_named_by_the_title_it_was_found_under():
    sources = g._answer_sources(_state(code_result=_code_result_from_the_portal()))
    statements = [s for s, _ in sources]
    assert any("Chicago Public Schools - School Locations SY2425" in s for s in statements), statements
    assert any("data.cityofchicago.org" in s or "City of Chicago" in s for s in statements)


def test_a_portal_download_is_matched_to_the_page_it_was_found_on():
    """Socrata: the CSV lives under /api/views/<id>/, the page search found under /<name>/<id>."""
    result = _code_result_from_the_portal()
    result["tool_results"][0]["content"] = json.dumps({"documents": [
        {"title": CPS_TITLE,
         "url": "https://data.cityofchicago.org/Education/Chicago-Public-Schools-School-Locations-SY2425/abcd-1234"}]})
    statements = [s for s, _ in g._answer_sources(_state(code_result=result))]
    assert any("Chicago Public Schools - School Locations SY2425" in s for s in statements), statements


def test_a_catalogued_source_says_what_it_leaves_out():
    statements = [s for s, _ in g._answer_sources(_state(analysis_results=_osm_analysis_result()))]
    assert statements and "OpenStreetMap" in statements[0]
    assert "leaves out whatever volunteers have not mapped" in statements[0]
    assert "ODbL" in statements[0]


def test_every_answer_that_used_data_names_it():
    out = g._with_sources(LIST, _state(code_result=_code_result_from_the_portal()))
    assert out.startswith(LIST.rstrip())
    assert "**Sources:**" in out and "Chicago Public Schools - School Locations SY2425" in out


def test_the_line_does_not_depend_on_what_the_answer_says():
    """An answer that names its source in prose gets the line too: the prose is the model's,
    the line is the record's."""
    named = LIST + "\nFrom OpenStreetMap."
    assert "**Sources:**" in g._with_sources(named, _state(analysis_results=_osm_analysis_result()))
    prose = "The nearest school is TWAIN, 0 m from the site."
    assert "**Sources:**" in g._with_sources(prose, _state(code_result=_code_result_from_the_portal()))


def test_no_source_in_the_record_adds_nothing():
    assert g._with_sources(LIST, _state(code_result={"answer": "x", "tool_calls": [],
                                                     "tool_results": []})) == LIST


def test_the_turn_answer_carries_the_source(monkeypatch):
    """Through the graph: the code peer staged the portal file and the answerer did not say so."""
    box = {"i": 0}

    def decide(_s, _d):
        box["i"] += 1
        return "code" if box["i"] == 1 else "done"

    out = g.run_supervisor("which schools are within 1 mile?", decide_fn=decide,
                           search_fn=lambda q, s: [], do_rerank=False,
                           code_fn=lambda q, ev, st: _code_result_from_the_portal(),
                           synthesize_fn=lambda *a, **k: LIST)
    assert "Chicago Public Schools - School Locations SY2425" in out["final_answer"]


def test_the_sources_line_sits_with_the_answer_not_after_the_verdict(monkeypatch):
    box = {"i": 0}

    def decide(_s, _d):
        box["i"] += 1
        return {1: "analyze", 2: "code"}.get(box["i"], "done")

    def analyze(q, ev, st):
        raise RuntimeError("the analysis peer died")

    out = g.run_supervisor("which schools are within 1 mile?", decide_fn=decide,
                           search_fn=lambda q, s: [], do_rerank=False, analyze_fn=analyze,
                           code_fn=lambda q, ev, st: _code_result_from_the_portal(),
                           synthesize_fn=lambda *a, **k: LIST)
    final = out["final_answer"]
    assert "**Sources:**" in final and "\n\n---\n\n" in final, final
    assert final.index("**Sources:**") < final.index("\n\n---\n\n")


def test_one_source_however_many_calls_and_never_from_a_repeat_note():
    """Seen in the final stage 38 replay: Overpass failed, the peer asked again with other boxes,
    and the line listed three OSM 'sources', one built from the repeat guard's note."""
    res = _osm_analysis_result()
    first = res["tool_results"][0]
    second = {**first, "tool_call_id": "a2"}
    note = {"name": "overpass_search", "tool_call_id": "a3",
            "content": "[Same call, same arguments as a1 ...]\n{\"error\": \"x\"}"}
    res["tool_calls"] += [{"name": "overpass_search", "args": {}, "id": "a2"},
                          {"name": "overpass_search", "args": {}, "id": "a3"}]
    res["tool_results"] += [second, note]
    sources = g._answer_sources(_state(analysis_results=res))
    assert len(sources) == 1


def test_two_sources_read_as_two():
    state = _state(analysis_results=_osm_analysis_result(),
                   code_result=_code_result_from_the_portal())
    line = g._with_sources(LIST, state).split("**Sources:**")[1]
    assert "OpenStreetMap" in line and "Chicago Public Schools" in line


@pytest.mark.parametrize("record,expected", [
    ({}, ["slope.tif (your upload)"]),                                   # the run's own name
    ({"filename": "slope_deg.tif"}, ["slope_deg.tif (your upload)"]),    # the store's name
    ({"filename": "x.tif", "kind": "output"}, []),                       # the agent's own file
])
def test_an_upload_read_only_by_code_is_named(monkeypatch, record, expected):
    """Stage 45's harness run: T11 and U02 read their uploads only through `execute_code`, whose
    arguments reach the graph as a string, and the answer named no source."""
    import agent_runtime.file_store as file_store

    monkeypatch.setattr(file_store, "get_file_record", lambda fid: record)
    content = json.dumps({"ok": True, "stdout": "slope ok\n",
                          "input_files": [{"ref": "file_25b1b37c732f",
                                           "file_id": "file_25b1b37c732f",
                                           "filename": "slope.tif"}]})
    result = {"answer": "x", "tool_calls": [
        {"name": "execute_code", "id": "k1",
         "args": "{'code': 'import rasterio', 'input_files': ['file_25b1b37c732f']}"}],
        "tool_results": [{"name": "execute_code", "tool_call_id": "k1", "content": content}]}
    statements = [s for s, _ in g._answer_sources(_state(code_result=result))]
    assert statements == expected, statements
