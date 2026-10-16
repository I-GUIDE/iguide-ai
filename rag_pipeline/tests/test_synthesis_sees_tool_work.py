"""A turn whose analysis ran tools, and ran no code, must not be told "the code did not run".

Live, 2026-10-15 ~22:54 CDT (local agent, lumen deepseek-v4-flash, thread
sess-7c3ef000-…): "Count Chicago's serious crashes from 2020 to 2024 in each 500 m hexagon and
map the counts." The ANALYZE peer ran inspect_file_for_analysis, inspect_vector,
count_points_in_areas (ok, 3,028 hexagons, a choropleth delivered), summary_statistics,
select_by_attribute (1 feature, a second layer delivered) and read_text_file — every call ok,
none of them execute_code. The ledger recorded all six. The answer said: "the execution record
shows the code did not run, so there are no computed crash counts, no map layers, and no output
files to report."

Cause: `_apply_execution_honesty` wrote ``executed=False`` on every analyze result that had no
execute_code record, and synthesis turned that flag into "EXECUTION: the code in this result
did NOT run … do not describe its output as a result." A turn that wrote no code was told its
output was not a result.
"""
from __future__ import annotations

import json

from rag_pipeline.tests.test_supervisor_graph import _DELIVERED, _resp

# The six calls of the live turn. count_points_in_areas and select_by_attribute delivered map
# layers; read_text_file returns the file's text and carries no `ok` key, as the real tool does.
_TOOLS = ["inspect_file_for_analysis", "inspect_vector", "count_points_in_areas",
          "summary_statistics", "select_by_attribute", "read_text_file"]
_RESULTS = {
    "inspect_file_for_analysis": {"ok": True, "file_id": "file_crashes", "rows": 10412},
    "inspect_vector": {"ok": True, "feature_count": 3028, "crs": "EPSG:4326"},
    "count_points_in_areas": {**_DELIVERED, "feature_count": 3028, "total_points": 10008,
                              "map_layer": {**_DELIVERED["map_layer"],
                                            "label": "crashes per hex"}},
    "summary_statistics": {"ok": True, "field": "count", "max": 35, "sum": 10008},
    "select_by_attribute": {**_DELIVERED, "feature_count": 1,
                            "map_layer": {**_DELIVERED["map_layer"], "label": "top hexagon",
                                          "url": "/agent/files/file_layer2/download"}},
    "read_text_file": {"file_id": "file_top", "text": "hex_id,count\nH1852,35\n"},
}
_SUMMARY = ("Counted 10,008 serious crashes inside the 500 m grid (3,028 hexagons) and mapped "
            "the counts; the busiest hexagon is H1852 with 35 crashes.")


def _analyze_peer(monkeypatch, summary=_SUMMARY, tools=_TOOLS, results=_RESULTS):
    """default_analyze_fn with a scripted executor and code execution AVAILABLE, as in prod."""
    import agent_runtime.executor_factory as ef
    from agent_runtime.supervisor.graph import default_analyze_fn

    seen = []

    def fake_invoke(executor, query=None, chat_history=None, config=None, **kw):
        seen.append(query)
        return _resp(summary, tool_names=tools, results=results)

    monkeypatch.setattr(ef, "build_agent_executor", lambda **kw: object())
    monkeypatch.setattr(ef, "invoke_agent_with_payload_fallback", fake_invoke)
    monkeypatch.setattr("agent_runtime.code_execution.is_code_exec_enabled", lambda: True)
    fn = default_analyze_fn(include_mcp_tools=False, code_exec=True,
                            input_file_ids=["file_crashes", "file_hex"])
    return fn, seen


def test_a_tool_only_analysis_makes_no_claim_about_code(monkeypatch):
    """No execute_code call and no code in the summary: there is no code to have run or not."""
    from agent_runtime.supervisor.graph import _execution_note

    fn, seen = _analyze_peer(monkeypatch)
    out = fn("count crashes per hexagon and map them", [], {"thread_id": "t"})
    assert len(seen) == 1, "nothing to challenge: no code was written"
    assert out["on_map"] is True
    assert "executed" not in out, out.get("executed")
    assert _execution_note(out) == ""


def test_synthesis_is_not_told_the_turn_did_nothing(monkeypatch):
    """The live turn end to end: analyze, then done, then the default writer."""
    from agent_runtime.supervisor.graph import run_supervisor

    analyze, _ = _analyze_peer(monkeypatch)
    prompts = []

    def llm(prompt: str) -> str:
        prompts.append(prompt)
        return "There are 10,008 crashes in the grid; H1852 has the most (35)."

    box = {"i": 0}

    def decide(s, d):
        box["i"] += 1
        return "analyze" if box["i"] == 1 else "done"

    state = run_supervisor(
        "Count Chicago's serious crashes from 2020 to 2024 in each 500 m hexagon and map the "
        "counts. How many crashes fall in the grid, and which hexagon has the most?",
        llm=llm, thread_id="sess-synthesis-sees-work", decide_fn=decide, analyze_fn=analyze,
        search_fn=lambda q, s: [], do_rerank=False, do_audit=False)

    writer = [p for p in prompts if "Analysis results:" in p]
    assert writer, "the default writer never ran"
    sent = writer[-1]
    assert "did NOT run" not in sent, sent[sent.find("EXECUTION"):][:300]
    assert "10,008" in sent and "H1852" in sent
    assert "10,008" in state["final_answer"]


def test_a_failed_code_run_does_not_disown_the_tools_that_ran():
    """execute_code failed, the counting tool succeeded: the note is about the code only."""
    from agent_runtime.supervisor.graph import _execution_note

    result = {
        "summary": "Counted 10,008 crashes; the script that styled the map failed.",
        "executed": False, "execution_error": "NameError: name 'gpd' is not defined",
        "tool_calls": [{"name": "count_points_in_areas", "args": {}, "id": "a"},
                       {"name": "execute_code", "args": {}, "id": "b"}],
        "tool_results": [
            {"name": "count_points_in_areas", "tool_call_id": "a",
             "content": json.dumps(_RESULTS["count_points_in_areas"])},
            {"name": "execute_code", "tool_call_id": "b",
             "content": json.dumps({"ok": False, "exit_code": 1,
                                    "error": "NameError: name 'gpd' is not defined"})},
        ],
    }
    note = _execution_note(result)
    assert "did NOT run" in note and "NameError" in note
    assert "count_points_in_areas" in note
    assert "do not describe its output as a result" not in note.lower()


def test_unrun_code_is_still_flagged(monkeypatch):
    """The honesty check this flag exists for: a code block handed back without a run."""
    from agent_runtime.supervisor.graph import _execution_note

    fn, seen = _analyze_peer(
        monkeypatch, summary="Here is the script:\n```python\nprint(10008)\n```",
        tools=["inspect_vector"], results={"inspect_vector": _RESULTS["inspect_vector"]})
    out = fn("count crashes per hexagon", [], {"thread_id": "t"})
    assert len(seen) == 2, "the unrun code is challenged once"
    assert out["executed"] is False
    assert "did NOT run" in _execution_note(out)
