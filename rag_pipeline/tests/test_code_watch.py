"""Stage 78: a code run that can be watched, stopped and reviewed while it executes.

Production T16 (2026-10-17) lost three of six runs to a 60 s limit on correct but slow code, and
the model was told only "execution timed out". These pin the pieces that replace that:

* the rule check before a run (code_watch.precheck) — the two slow constructs of T16 and the
  stated-parameter check, plus what must NOT be flagged;
* where a timed-out run was (stack_report), read from a real run's stack;
* the concurrent review: a critical verdict with evidence stops the container; one whose quote
  is not in the code, or that calls a line slow while the stack is elsewhere, does not;
* Stop: turn_control cancels a run in about a second and raises at the next model/tool step;
* the time limit: 180 s by default, never more than the cap.

The run tests use the local backend, so they need no Docker; the docker backend shares the
same supervision loop (code_execution._supervise) and differs only in how it signals and kills.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime import code_watch, turn_control  # noqa: E402
from agent_runtime import code_execution  # noqa: E402
from agent_runtime.code_execution import LocalSubprocessExecutor  # noqa: E402


@pytest.fixture(autouse=True)
def _work_root(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_CODE_EXEC_WORK_ROOT", str(tmp_path / "work"))
    (tmp_path / "work").mkdir()
    monkeypatch.setenv("AGENT_INVARIANT_GATE", "0")
    monkeypatch.setenv("AGENT_CODE_EXEC_PREINSTALLED", "numpy,pandas")


SLOW_NESTED = """import math
pts = [(i * 0.1, i * 0.2) for i in range(30000)]
best = []
print("starting", flush=True)
for a in pts:
    for b in pts:
        d = math.hypot(a[0] - b[0], a[1] - b[1])
    best.append(d)
"""


def _rules(code: str, brief: str = "") -> list:
    return [f["rule"] for f in code_watch.precheck(code, brief=brief)]


# --------------------------------------------------------------------------- rules

def test_unprepared_predicate_against_one_geometry_is_flagged():
    code = ("import geopandas as gpd\nb = gpd.read_file('b.geojson')\n"
            "slr = gpd.read_file('slr.geojson').geometry.iloc[0]\nhit = b[b.intersects(slr)]\n")
    assert "unprepared_predicate" in _rules(code)


def test_a_prepared_geometry_is_not_flagged():
    code = ("import geopandas as gpd, shapely\nb = gpd.read_file('b.geojson')\n"
            "slr = gpd.read_file('slr.geojson').geometry.iloc[0]\nshapely.prepare(slr)\n"
            "hit = b[b.intersects(slr)]\n")
    assert "unprepared_predicate" not in _rules(code)


def test_nested_row_loops_are_flagged_and_name_the_outer_loop():
    found = code_watch.precheck(SLOW_NESTED)
    nested = [f for f in found if f["rule"] == "nested_loop"]
    assert nested and nested[0]["line"] == 6 and "line 5" in nested[0]["message"]


def test_a_loop_over_a_few_literal_values_is_not_a_row_loop():
    code = ("for year in (2019, 2020, 2021):\n    for row in rows:\n        pass\n"
            "for i in range(3):\n    for r in rows:\n        pass\n")
    assert "nested_loop" not in _rules(code)


def test_iterrows_union_and_read_inside_a_loop():
    code = ("import geopandas as gpd\nfor name in names:\n"
            "    g = gpd.read_file(name)\n    u = g.union_all()\n"
            "for i, r in df.iterrows():\n    pass\n")
    rules = _rules(code)
    assert {"read_in_loop", "union_in_loop", "iterrows"} <= set(rules)


def test_rowwise_geometry_apply():
    code = "df['d'] = df.geometry.apply(lambda g: g.distance(pt))\n"
    assert "rowwise_geometry" in _rules(code)


def test_a_stated_buffer_the_code_never_uses_is_flagged():
    """The open issue of 2026-09: told 'buffer 2500 m', the agent used 1000."""
    code = "zone = sites.buffer(1000)\n"
    found = code_watch.precheck(code, brief="Task: buffer the sites by 2500 m and count homes")
    assert [f for f in found if f["rule"] == "stated_parameter" and "2500 m" in f["message"]]


@pytest.mark.parametrize("code", ["zone = sites.buffer(2500)\n", "R_KM = 2.5\nz = s.buffer(R_KM*1000)\n",
                                  "zone = sites.buffer(2500.0)\n"])
def test_the_stated_value_in_any_unit_is_not_flagged(code):
    assert "stated_parameter" not in _rules(code, "buffer the sites by 2,500 m")


def test_a_figure_with_no_parameter_word_is_not_a_parameter():
    assert "stated_parameter" not in _rules("x = 1\n", "the river is 300 km long")


def test_syntax_errors_and_empty_code_yield_nothing():
    assert code_watch.precheck("def (:\n") == []
    assert code_watch.precheck("") == []


# --------------------------------------------------------------------------- stack parsing

DUMP = """Current thread 0x0000000102 (most recent call first):
  File "/usr/lib/python3.11/site-packages/shapely/predicates.py", line 812 in intersects
  File "/work/script.py", line 47 in overlay
  File "/work/script.py", line 60 in <module>
"""


def test_parse_stack_maps_lines_through_the_gate_offset_and_keeps_the_library_call():
    frames = code_watch.parse_stack(DUMP, script="script.py", line_offset=40, code_lines=30)
    assert [f["line"] for f in frames] == [7, 20]
    assert frames[0]["inside"].startswith("shapely/predicates.py:812")


def test_parse_stack_reads_only_the_last_dump():
    two = DUMP + DUMP.replace("line 47", "line 49")
    assert code_watch.parse_stack(two, script="script.py")[0]["line"] == 49


def test_frames_in_the_gate_prologue_are_dropped():
    frames = code_watch.parse_stack(DUMP, script="script.py", line_offset=50, code_lines=5)
    assert frames == []


# --------------------------------------------------------------------------- verdict guard

def _verdict(**kw):
    base = {"verdict": "critical", "category": "performance", "line": 5,
            "quote": "for a in pts:", "problem": "quadratic", "fix": "use a KD-tree"}
    base.update(kw)
    return base


def test_a_performance_verdict_on_the_loop_the_program_is_in_is_accepted():
    ok, why = code_watch.accept_verdict(_verdict(), code=SLOW_NESTED, brief="",
                                        frames=[{"line": 7, "function": "<module>"}],
                                        elapsed=10, timeout=180)
    assert ok, why


def test_a_quote_that_is_not_on_the_cited_line_is_refused():
    ok, why = code_watch.accept_verdict(_verdict(quote="gdf.sjoin(x)"), code=SLOW_NESTED,
                                        brief="", frames=[{"line": 7}], elapsed=10, timeout=180)
    assert not ok and "quote" in why


def test_calling_a_line_slow_while_the_program_is_elsewhere_is_refused():
    ok, why = code_watch.accept_verdict(_verdict(line=2, quote="pts = [(i * 0.1"),
                                        code=SLOW_NESTED, brief="",
                                        frames=[{"line": 7}], elapsed=10, timeout=180)
    assert not ok and "line 7" in why


def test_without_a_stack_a_performance_stop_waits_for_half_the_limit():
    args = dict(code=SLOW_NESTED, brief="", frames=[], timeout=180)
    assert not code_watch.accept_verdict(_verdict(), elapsed=30, **args)[0]
    assert code_watch.accept_verdict(_verdict(), elapsed=95, **args)[0]


def test_a_parameter_verdict_must_quote_the_task():
    code = "zone = sites.buffer(1000)\n"
    v = _verdict(category="parameter", line=1, quote="sites.buffer(1000)",
                 task_quote="buffer of 2500 m")
    assert code_watch.accept_verdict(v, code=code, brief="use a buffer of 2500 m", frames=[],
                                     elapsed=6, timeout=180)[0]
    assert not code_watch.accept_verdict({**v, "task_quote": "3 km"}, code=code,
                                         brief="use a buffer of 2500 m", frames=[],
                                         elapsed=6, timeout=180)[0]


@pytest.mark.parametrize("bad", [{"verdict": "ok"}, _verdict(line=99), _verdict(fix=""),
                                 _verdict(category="style"), "not a dict"])
def test_anything_less_than_a_complete_critical_verdict_stops_nothing(bad):
    assert not code_watch.accept_verdict(bad, code=SLOW_NESTED, brief="", frames=[{"line": 7}],
                                         elapsed=100, timeout=180)[0]


def test_parse_verdict_reads_fenced_json():
    v = code_watch.parse_verdict('Here:\n```json\n{"verdict": "ok"}\n```')
    assert v == {"verdict": "ok"}


# --------------------------------------------------------------------------- real runs

def test_a_timed_out_run_says_which_line_it_was_on_and_why():
    r = LocalSubprocessExecutor().execute(SLOW_NESTED, timeout=3).to_dict()
    assert r["timed_out"] and r["error"] == "execution timed out after 3 s"
    assert r["stuck_at"]["line"] in (6, 7)
    assert r["stuck_at"]["explained_by"]["rule"] == "nested_loop"
    assert r["stderr"].startswith("[timed out at line")
    assert "starting" in r["stdout"]
    assert ".iguide_rt" not in json.dumps(r["artifacts"])


def _watch(reviewer, events, **kw):
    return code_watch.RunWatch(code="", timeout=0, label="t", brief=kw.pop("brief", "nearest"),
                               reviewer=reviewer, review_after=kw.pop("review_after", 0.5),
                               perf_floor=kw.pop("perf_floor", 0.0),
                               hints_after=kw.pop("hints_after", 0.5),
                               heartbeat_every=0.5, emit=lambda e, d: events.append((e, d)), **kw)


def test_a_critical_review_stops_the_run_and_returns_the_fix():
    events: list = []

    def reviewer(prompt):
        assert "LIVE STACK (where" in prompt and "line 7" in prompt
        return json.dumps(_verdict(fix="scipy.spatial.cKDTree(pts).query(pts, k=2)"))

    started = time.monotonic()
    r = LocalSubprocessExecutor().execute(SLOW_NESTED, timeout=60,
                                          watch=_watch(reviewer, events)).to_dict()
    assert time.monotonic() - started < 20
    assert r["interrupted"] == "review" and not r["ok"]
    assert r["review"]["stopped_run"] and r["review"]["line"] == 5
    assert "cKDTree" in r["stderr"] and r["stderr"].startswith("[code review, line 5")
    kinds = [(e, d.get("stage")) for e, d in events]
    assert ("code_review", "rules") in kinds and ("code_progress", "finished") in kinds
    assert any(e == "code_review" and d.get("stopped") for e, d in events)


def test_a_performance_stop_waits_for_the_floor_and_rechecks_the_stack():
    """Live 2026-10-17: an sjoin was stopped at 7.7 s as too slow, and the same join had just
    finished in 12 s. A performance verdict now waits for the floor, then needs the program to
    still be in the cited code."""
    events: list = []
    reviewer = lambda prompt: json.dumps(_verdict())   # noqa: E731
    r = LocalSubprocessExecutor().execute(SLOW_NESTED, timeout=60,
                                          watch=_watch(reviewer, events, perf_floor=4.0)).to_dict()
    assert r["interrupted"] == "review" and r["review"]["confirmed_at_s"] >= 4.0
    assert r["ran_seconds"] >= 4.0
    assert any(e == "code_review" and d.get("stage") == "held" for e, d in events)


def test_a_held_performance_stop_is_dropped_when_the_program_moved_on():
    code = ("import time\nfor i in range(3):\n    time.sleep(1)\n"
            "x = 0\nwhile x < 10**9:\n    x += 1\n")
    v = {"verdict": "critical", "category": "performance", "line": 2,
         "quote": "for i in range(3):", "problem": "slow", "fix": "do less"}
    r = LocalSubprocessExecutor().execute(
        code, timeout=8, watch=_watch(lambda p: json.dumps(v), [], perf_floor=4.5)).to_dict()
    assert "interrupted" not in r and r["timed_out"]
    assert r["review"]["not_acted_on"].startswith("re-checked at")


def test_a_held_performance_stop_never_fires_on_a_run_that_finishes():
    v = {"verdict": "critical", "category": "performance", "line": 2,
         "quote": "time.sleep(2)", "problem": "slow", "fix": "do less"}
    r = LocalSubprocessExecutor().execute(
        "import time\ntime.sleep(2)\n", timeout=60,
        watch=_watch(lambda p: json.dumps(v), [], perf_floor=30)).to_dict()
    assert r["ok"] and "interrupted" not in r and r["review"]["held_until_s"] == 30


def test_an_unsupported_verdict_is_reported_and_the_run_continues():
    events: list = []
    reviewer = lambda prompt: json.dumps(_verdict(quote="nothing like this"))   # noqa: E731
    r = LocalSubprocessExecutor().execute(SLOW_NESTED, timeout=4,
                                          watch=_watch(reviewer, events)).to_dict()
    assert r["timed_out"] and "interrupted" not in r
    assert r["review"]["not_acted_on"].startswith("the quote is not")


def test_observe_mode_never_stops_a_run():
    events: list = []
    reviewer = lambda prompt: json.dumps(_verdict())   # noqa: E731
    r = LocalSubprocessExecutor().execute(SLOW_NESTED, timeout=4,
                                          watch=_watch(reviewer, events, mode="observe")).to_dict()
    assert r["timed_out"] and r["review"]["verdict"] == "critical"
    assert not r["review"].get("stopped_run")


def test_a_quick_run_never_waits_for_its_review():
    events: list = []
    slow_reviewer = lambda prompt: (time.sleep(30), "{}")[1]   # noqa: E731
    started = time.monotonic()
    r = LocalSubprocessExecutor().execute("import time\ntime.sleep(1.5)\nprint('ok')\n",
                                          timeout=60,
                                          watch=_watch(slow_reviewer, events)).to_dict()
    assert r["ok"] and time.monotonic() - started < 12
    assert r["review"]["state"] == "dropped"


def test_review_off_makes_no_model_call():
    calls = []
    r = LocalSubprocessExecutor().execute(
        "import time\ntime.sleep(1.5)\n", timeout=60,
        watch=_watch(lambda p: calls.append(p) or "{}", [], mode="off")).to_dict()
    assert r["ok"] and calls == [] and "review" not in r


def test_stop_kills_a_running_run_within_seconds():
    turn = turn_control.begin("alice")
    result = {}

    def run():
        with turn_control.bound(turn):
            result["r"] = LocalSubprocessExecutor().execute(SLOW_NESTED, timeout=120).to_dict()

    t = threading.Thread(target=run)
    t.start()
    time.sleep(2.0)
    started = time.monotonic()
    assert turn_control.request_cancel(turn.id, "alice") == "cancelled"
    t.join(timeout=15)
    assert not t.is_alive() and time.monotonic() - started < 6
    assert result["r"]["interrupted"] == "cancelled" and not result["r"]["ok"]
    turn_control.end(turn)


def test_the_time_limit_is_capped(monkeypatch):
    monkeypatch.setattr(code_execution, "MAX_TIMEOUT", 2)
    r = LocalSubprocessExecutor().execute("import time\ntime.sleep(10)\n", timeout=900).to_dict()
    assert r["timed_out"] and r["time_limit_s"] == 2
    assert "900 s was asked for and the cap is 2 s" in r["stderr"]


def test_the_default_limit_is_three_minutes():
    assert code_execution.DEFAULT_TIMEOUT == 180 and code_execution.MAX_TIMEOUT == 600


# --------------------------------------------------------------------------- turn control

def test_only_the_owner_can_cancel():
    turn = turn_control.begin("alice")
    try:
        assert turn_control.request_cancel(turn.id, "bob") == "not_yours"
        assert not turn.cancelled
        assert turn_control.request_cancel("nope", "alice") == "not_found"
        assert turn_control.request_cancel(turn.id, "alice") == "cancelled" and turn.cancelled
    finally:
        turn_control.end(turn)


def test_a_cancelled_turn_raises_at_the_next_model_or_tool_step():
    """BaseException, so neither the tool node nor a containment wrapper turns it into an
    observation the agent carries on from."""
    from agent_runtime.streaming_trace import StreamingTraceCallbackHandler

    assert not issubclass(turn_control.TurnCancelled, Exception)
    turn = turn_control.begin(None)
    handler = StreamingTraceCallbackHandler()
    with turn_control.bound(turn):
        handler.on_tool_start({"name": "x"}, "{}")          # not cancelled: no effect
        turn_control.request_cancel(turn.id, None)
        with pytest.raises(turn_control.TurnCancelled):
            handler.on_tool_start({"name": "x"}, "{}")
        with pytest.raises(turn_control.TurnCancelled):
            handler.on_chat_model_start({"name": "m"}, [[]])
    turn_control.end(turn)


def test_closing_the_stream_cancels_the_turn(monkeypatch):
    """A reader that goes away (closed tab, aborted fetch) stops the turn instead of leaving it
    to compute into a closed socket."""
    from agent_runtime import graph_runtime

    seen = {}

    class FakeGraph:
        def invoke(self, state):
            seen["turn"] = turn_control.current()
            for _ in range(200):
                time.sleep(0.05)
                turn_control.raise_if_cancelled()
            return {}

    monkeypatch.setattr(graph_runtime, "build_orchestrator_graph", lambda **kw: FakeGraph())
    monkeypatch.setenv("AGENT_STREAM_HEARTBEAT_SECONDS", "1")
    gen = graph_runtime.stream_agent_query_events("q", checkpointer=None)
    first = next(gen)
    assert first["event"] == "turn" and first["data"]["turn_id"]
    # Past the status events to the first keepalive: the worker is running by then.
    while next(gen)["event"] != "keepalive":
        pass
    gen.close()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not (seen.get("turn") and seen["turn"].cancelled):
        time.sleep(0.05)
    assert seen["turn"].cancelled and seen["turn"].reason == "the client disconnected"


def test_cancel_route_answers_404_for_an_unknown_turn(monkeypatch):
    import api.server as server

    monkeypatch.setenv("AGENT_CHAT_API_KEY", "k")
    with server.app.test_client() as client:
        res = client.post("/agent/chat/cancel", json={"turnId": "nope"},
                          headers={"X-API-KEY": "k"})
        assert res.status_code == 404 and res.get_json()["reason"] == "no_such_turn"


def test_cancel_route_stops_a_registered_turn(monkeypatch):
    import api.server as server

    monkeypatch.setenv("AGENT_CHAT_API_KEY", "k")
    turn = turn_control.begin(None)
    try:
        with server.app.test_client() as client:
            res = client.post("/agent/chat/cancel", json={"turnId": turn.id},
                              headers={"X-API-KEY": "k"})
        assert res.status_code == 200 and turn.cancelled
    finally:
        turn_control.end(turn)


# --------------------------------------------------------------------------- input sizes

def test_geometry_size_reports_vertices_not_just_features(tmp_path):
    import geopandas as gpd
    from shapely.geometry import Point

    ring = Point(0, 0).buffer(1, quad_segs=4000)   # one feature, ~16,000 vertices
    path = tmp_path / "big.geojson"
    gpd.GeoDataFrame(geometry=[ring], crs="EPSG:4326").to_file(path, driver="GeoJSON")
    size = code_watch.geometry_size(str(path))
    assert size["features"] == 1 and size["vertices_max"] > 10_000
    assert "shapely.prepare" in size["note"]


# --------------------------------------------------------------------------- quiet hints (S78.1)

QUICK_NESTED = ("rows = list(range(40))\ntotal = 0\nfor a in rows:\n    for b in rows:\n"
                "        total += a * b\nprint(total)\n")


def test_a_fast_run_shows_and_returns_no_performance_hints():
    """Production T16, 2026-10-17: 'nested loop' and 'rowwise geometry' were shown on runs that
    took 1.8 s. A hint on every run is noise."""
    events: list = []
    w = code_watch.RunWatch(code="", timeout=0, label="q", mode="off",
                            emit=lambda e, d: events.append((e, d)))
    r = LocalSubprocessExecutor().execute(QUICK_NESTED, timeout=60, watch=w).to_dict()
    assert r["ok"] and "performance_hints" not in r
    assert not [d for e, d in events if e == "code_review"]


def test_a_slow_run_shows_its_hints_once_and_returns_them():
    events: list = []
    w = code_watch.RunWatch(code="", timeout=0, label="s", mode="off", hints_after=1.0,
                            emit=lambda e, d: events.append((e, d)))
    r = LocalSubprocessExecutor().execute(SLOW_NESTED, timeout=4, watch=w).to_dict()
    shown = [d for e, d in events if e == "code_review" and d.get("stage") == "rules"]
    assert len(shown) == 1 and shown[0]["message"].startswith("Still running after 1 s")
    assert [h["rule"] for h in r["performance_hints"]] == ["nested_loop"]


def test_a_run_past_the_result_threshold_returns_hints(monkeypatch):
    monkeypatch.setenv("AGENT_CODE_HINTS_RESULT_AFTER_S", "1")
    r = LocalSubprocessExecutor().execute(
        "import time\nrows = range(3000)\nfor a in rows:\n    for b in rows:\n        pass\n"
        "time.sleep(1.2)\n", timeout=60).to_dict()
    assert r["ok"] and r["performance_hints"][0]["rule"] == "nested_loop"


def test_a_parameter_check_is_shown_at_once_and_returned_even_on_a_fast_run():
    events: list = []
    w = code_watch.RunWatch(code="", timeout=0, label="p", mode="off",
                            brief="Task: buffer the sites by 2500 m",
                            emit=lambda e, d: events.append((e, d)))
    r = LocalSubprocessExecutor().execute("radius = 1000\nprint(radius)\n", timeout=60,
                                          watch=w).to_dict()
    assert r["ok"] and r["parameter_checks"][0]["rule"] == "stated_parameter"
    shown = [d for e, d in events if e == "code_review"]
    assert shown and shown[0]["message"].startswith("Parameter check: the task states")


def test_hints_for_result_splits_by_kind_and_speed():
    found = [{"rule": "nested_loop", "line": 3}, {"rule": "stated_parameter", "line": 0}]
    assert code_watch.hints_for_result(found, ran_s=1.0, slow=False) == (
        [], [{"rule": "stated_parameter", "line": 0}])
    assert code_watch.hints_for_result(found, ran_s=1.0, slow=True)[0] == [
        {"rule": "nested_loop", "line": 3}]


# --------------------------------------------------------------------------- step narration (S78.1)

def _llm_end(handler, content, tool_calls):
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    msg = AIMessage(content=content, tool_calls=tool_calls)
    handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]))


def test_the_sentence_beside_tool_calls_is_emitted_as_the_step_plan():
    """deepseek writes what it is about to do next to its tool calls (6 of 6 probes); the trace
    used to drop it with the duplicate calls, leaving bare function calls with no reason."""
    from agent_runtime.streaming_trace import active_callback_handler, trace_context

    events = []
    with trace_context(events.append, agent_dev=True):
        handler = active_callback_handler()
        _llm_end(handler, "\n\nI'll inspect both files to understand their structure.",
                 [{"name": "inspect_vector", "args": {"file_id": "f1"}, "id": "c1"},
                  {"name": "inspect_vector", "args": {"file_id": "f2"}, "id": "c2"}])
        _llm_end(handler, "", [{"name": "execute_code", "args": {"code": "x"}, "id": "c3"}])
    plans = [e["data"] for e in events if e["event"] == "tool_plan"]
    assert len(plans) == 1
    assert plans[0]["message"] == "I'll inspect both files to understand their structure."
    assert plans[0]["tools"] == ["inspect_vector", "inspect_vector"]
    # the calls themselves are still not duplicated as messages
    assert not [e for e in events if e["event"] == "llm_interaction"]


def test_both_peers_are_asked_to_narrate_their_steps():
    from agent_runtime.supervisor.prompts import ANALYSIS_WORKFLOW_PROMPT, CODE_PEER_PROMPT

    for prompt in (ANALYSIS_WORKFLOW_PROMPT, CODE_PEER_PROMPT):
        assert "what you are about to do and why" in prompt


def test_tool_call_events_carry_the_arguments_as_a_dict():
    """The client turns a call into words from its arguments; a Python repr string gave it
    nothing to read ("Inspecting a file")."""
    from agent_runtime.streaming_trace import active_callback_handler, trace_context

    events = []
    with trace_context(events.append, agent_dev=True):
        h = active_callback_handler()
        h.on_tool_start({"name": "inspect_vector"}, "{'file_id': 'file_885af7d38fd5'}",
                        inputs={"file_id": "file_885af7d38fd5"})
        h.on_tool_start({"name": "inspect_vector"}, "{'file_id': 'file_2437c7f294d7'}")
    args = [e["data"]["args"] for e in events if e["event"] == "tool_call"]
    assert args == [{"file_id": "file_885af7d38fd5"}, {"file_id": "file_2437c7f294d7"}]


def test_the_code_step_says_what_it_did():
    """"Code ready" read as code waiting to run, after the step had run it many times."""
    import json as _json

    from agent_runtime.supervisor.graph import _code_step_summary

    ok = {"name": "execute_code", "content": _json.dumps({"ok": True, "exit_code": 0})}
    bad = {"name": "execute_code", "content": _json.dumps({"ok": False, "exit_code": 1,
                                                           "error": "boom"})}
    other = {"name": "inspect_vector", "content": "{}"}
    assert _code_step_summary({"tool_results": [ok, bad, ok, other]}) == \
        "Code step finished: 3 runs, 1 failed"
    assert _code_step_summary({"tool_results": [ok]}) == "Code step finished: 1 run"
    assert _code_step_summary({"tool_results": [other]}) == "Code step finished"
    assert _code_step_summary("not a dict") == "Code step finished"
