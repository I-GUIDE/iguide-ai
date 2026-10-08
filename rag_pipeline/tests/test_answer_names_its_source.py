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
from the tool record, and a line naming them is added only when the answer lists features and
names none of them.
"""
from __future__ import annotations

import json

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


def test_a_staged_dataset_is_named_by_the_title_it_was_found_under():
    sources = g._feature_sources({"code_result": _code_result_from_the_portal()})
    assert len(sources) == 1
    assert "Chicago Public Schools - School Locations SY2425" in sources[0]
    assert "data.cityofchicago.org" in sources[0]


def test_a_portal_download_is_matched_to_the_page_it_was_found_on():
    """Socrata: the CSV lives under /api/views/<id>/, the page search found under /<name>/<id>."""
    result = _code_result_from_the_portal()
    result["tool_results"][0]["content"] = json.dumps({"documents": [
        {"title": CPS_TITLE,
         "url": "https://data.cityofchicago.org/Education/Chicago-Public-Schools-School-Locations-SY2425/abcd-1234"}]})
    sources = g._feature_sources({"code_result": result})
    assert "Chicago Public Schools - School Locations SY2425" in sources[0]


def test_an_osm_list_carries_its_own_statement():
    sources = g._feature_sources({"analysis_results": _osm_analysis_result()})
    assert sources and "OpenStreetMap" in sources[0] and "not an official" in sources[0]


def test_a_list_that_names_no_source_gets_one():
    out = g._with_feature_source(LIST, {"code_result": _code_result_from_the_portal()})
    assert out.startswith(LIST.rstrip())
    tail = out[len(LIST.rstrip()):]
    assert "Chicago Public Schools - School Locations SY2425" in tail
    assert "only" in tail          # it says the list covers that source and no more


def test_a_list_that_already_names_its_source_is_left_alone():
    named = LIST + "\nSource: the Chicago Public Schools locations file (data.cityofchicago.org)."
    assert g._with_feature_source(named, {"code_result": _code_result_from_the_portal()}) == named
    osm = LIST + "\nFrom OpenStreetMap."
    assert g._with_feature_source(osm, {"analysis_results": _osm_analysis_result()}) == osm


def test_prose_without_a_list_is_left_alone():
    prose = "The nearest school is TWAIN, 0 m from the site."
    assert g._with_feature_source(prose, {"code_result": _code_result_from_the_portal()}) == prose


def test_no_source_in_the_record_adds_nothing():
    assert g._with_feature_source(LIST, {"code_result": {"answer": "x", "tool_calls": [],
                                                         "tool_results": []}}) == LIST


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


def test_the_source_sits_with_the_list_not_after_the_caveats():
    """Seen in the local replay: the line landed under the invariant gate's caveat, and the OSM
    statement's own full stop doubled ("still listed.. The list")."""
    answer = LIST + "\n⚠️ A deterministic invariant check COULD NOT VERIFY this run.\n\n- detail"
    out = g._with_feature_source(answer, {"analysis_results": _osm_analysis_result()})
    assert out.index("**Source:**") < out.index("⚠️")
    assert ".. The list" not in out and "register. The list" in out
    assert out.endswith("- detail")


def test_one_osm_source_however_many_calls_and_never_from_a_repeat_note():
    """Seen in the final replay: Overpass failed, the peer asked again with other boxes, and the
    line listed three OSM 'sources', one of them built from the repeat guard's note (not JSON)
    as 'features tagged as requested'. One source is one source; the last good call describes it."""
    res = _osm_analysis_result()
    first = res["tool_results"][0]
    second = {**first, "tool_call_id": "a2", "content": first["content"].replace(
        "bbox [...]", "bbox [-87.78, 41.77, -87.72, 41.82]")}
    note = {"name": "overpass_search", "tool_call_id": "a3",
            "content": "[This is the same call, with the same arguments, as a1 ...]\n{\"error\": \"x\"}"}
    res["tool_calls"] += [{"name": "overpass_search", "args": {}, "id": "a2"},
                          {"name": "overpass_search", "args": {}, "id": "a3"}]
    res["tool_results"] += [second, note]
    sources = g._feature_sources({"analysis_results": res})
    assert len(sources) == 1 and "-87.78, 41.77" in sources[0]
    out = g._with_feature_source(LIST, {"analysis_results": res})
    assert "as requested" not in out and "that source holds" in out


def test_two_sources_read_as_two():
    state = {"analysis_results": _osm_analysis_result(), "code_result": _code_result_from_the_portal()}
    out = g._with_feature_source(LIST, state)
    assert "these sources hold." in out
