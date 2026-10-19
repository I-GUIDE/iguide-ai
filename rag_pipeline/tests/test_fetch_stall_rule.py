"""Stage 83: a run that keeps fetching without producing anything ends, and says what it tried.

The live turn behind this (production, 2026-10-16 20:41 UTC, Lumen deepseek-v4-flash): asked
for the bearing between two HRSA health-center sites, the code peer ran 27+ steps in ten
minutes: about 40 `stage_url` calls against one ArcGIS REST host (five layers, each queried
several ways) and its download pages, interleaved with `execute_code` reads of the staged
files. The files held empty query results or service metadata, never the two sites. Stage 42's
rule ends a run after two steps that add nothing new, but every fetch returned fresh bytes and
every read printed fresh text, so it never fired. Stage 83 counts the steps that fetch from a
URL since the run last PRODUCED something: a new output file, map layer or evidence document.
A fetched file is input. Steps that fetch nothing are not counted.

Like test_repeat_tool_calls, these drive the REAL create_agent with the real middleware stack
inside a supervisor turn (`turn_log.bind`). Offline: every tool is a local stub.
"""
from __future__ import annotations

import itertools
import json

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from agent_runtime import executor_factory as ef
from agent_runtime import turn_log
from agent_runtime.supervisor import graph as g

ARCGIS = "https://gisportal.example.gov/arcgis/rest/services/Health/MapServer"
SEQ = itertools.count(1)


class ToolAwareFake(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
        return self


def _hex(n: int) -> str:
    return f"{n:012x}"


@tool
def stage_url(url: str, filename: str = "") -> str:
    """Stage a public file into the workspace."""
    n = next(SEQ)
    return json.dumps({"ok": True, "staged_path": f"/work/inputs/q{n}.json", "origin": url,
                       "bytes": 300 + n, "sha256": _hex(n) * 5, "fetched_at": f"t{n}"})


@tool
def fetch_public_data(url: str, filename: str = "") -> str:
    """Fetch a public file and keep it as a conversation file."""
    n = next(SEQ)
    return json.dumps({"ok": True, "file_id": f"file_{_hex(n)}", "url": url, "bytes": 200})


@tool
def execute_code(code: str) -> str:
    """Run code. Saves its own source on every run, as the real tool does."""
    n = next(SEQ)
    return json.dumps({"ok": True, "exit_code": 0,
                       "stdout": f"features: 0\nfields: {n}\n",
                       "artifacts": [{"file_id": f"file_{_hex(n)}", "filename": f"read_{n}.py",
                                      "kind": "source"}],
                       "input_files": [{"file_id": "file_00000000abcd", "filename": "q.json"}]})


@tool
def process_tile(tile: int) -> str:
    """Process one tile and write its output."""
    n = next(SEQ)
    return json.dumps({"ok": True, "file_id": f"file_{_hex(n)}", "count": 120 + tile})


@tool
def keyword_search(query: str) -> str:
    """Search the knowledge base."""
    n = next(SEQ)
    return json.dumps({"source": "keyword", "count": 1,
                       "documents": [{"doc_id": f"doc-{n}", "title": query}]})


TOOLS = [stage_url, fetch_public_data, execute_code, process_tile, keyword_search]


def _call(name, args):
    return {"name": name, "args": args, "id": f"call_{next(SEQ)}", "type": "tool_call"}


def _agent(steps, final="done"):
    replies = [AIMessage(content="", tool_calls=calls) for calls in steps]
    return create_agent(model=ToolAwareFake(messages=iter(replies + [AIMessage(content=final)])),
                        tools=TOOLS, system_prompt="t", middleware=ef._default_middleware())


def _invoke(agent, log, peer="code"):
    with turn_log.bind(log, peer) as b:
        out = agent.invoke({"messages": [HumanMessage(content="q")]},
                           config={"recursion_limit": 400})
    return out, b.run


def _tools_run(out):
    return [m for m in out["messages"] if isinstance(m, ToolMessage)]


def _probe_steps(n):
    """The incident's shape: a query against another layer, then a read of what came back."""
    steps = []
    for i in range(n):
        if i % 2 == 0:
            layer = 16 + (i // 2) % 6
            steps.append([_call("stage_url", {"url": f"{ARCGIS}/{layer}/query?where=1%3D1&f=json"
                                                     f"&n={i}"})])
        else:
            steps.append([_call("execute_code", {"code": f"print(open('q{i}.json').read()) #{i}"})])
    return steps


@pytest.fixture(autouse=True)
def _default_limit(monkeypatch):
    monkeypatch.delenv("AGENT_FETCH_STALL_STEPS", raising=False)


def test_probing_that_keeps_succeeding_ends_and_says_what_it_tried():
    log = turn_log.new_log()
    out, run = _invoke(_agent(_probe_steps(30)), log)

    limit = turn_log.FETCH_STALL_LIMIT
    assert len(_tools_run(out)) == 2 * limit - 1           # fetch, read, ..., fetch; not 30
    final = out["messages"][-1].content
    assert final.startswith("[Stopped:") and "without producing anything" in final
    assert f"stage_url x{limit}" in final and f"execute_code x{limit - 1}" in final
    assert f"gisportal.example.gov x{limit}" in final
    assert log.run_stalled(run)
    # Stage 42's rule did not fire: every one of those steps was new.
    steps = [e for e in log.events if e["kind"] == "step" and e["run"] == run]
    assert all(e["productive"] for e in steps)
    ended = [e for e in log.events if e["kind"] == "run_ended" and e["run"] == run][0]
    assert ended["stalled"] and ended["stopped"]


def test_files_fetched_from_a_url_are_input_not_output():
    """fetch_public_data keeps what it fetched as a conversation file, with a new file id each
    time. A fetch is how the incident's run escaped a count of new results; a file id must not
    let the same probing escape this rule."""
    log = turn_log.new_log()
    steps = [[_call("fetch_public_data", {"url": f"{ARCGIS}/{i}/query?f=geojson"})]
             for i in range(20)]
    out, run = _invoke(_agent(steps), log)
    assert len(_tools_run(out)) == turn_log.FETCH_STALL_LIMIT
    assert log.run_stalled(run)


def test_a_long_run_that_produces_outputs_is_not_cut():
    """Tiled fetches, many files: 24 steps, each writing a new output, run to the end."""
    log = turn_log.new_log()
    steps = [[_call("process_tile", {"tile": i})] for i in range(24)]
    out, run = _invoke(_agent(steps, final="all tiles done"), log)
    assert len(_tools_run(out)) == 24
    assert out["messages"][-1].content == "all tiles done"
    assert not log.run_stalled(run)


def test_fetches_followed_by_outputs_keep_the_run_going():
    """Stage a file, inspect it, compute from it, write the result: the output resets the count,
    so three fetch-inspect rounds with an output each take as long as they need."""
    log = turn_log.new_log()
    steps = []
    for i in range(8):
        steps += [[_call("stage_url", {"url": f"https://data.example.org/part{i}.csv"})],
                  [_call("execute_code", {"code": f"inspect {i}"})],
                  [_call("process_tile", {"tile": i})]]
    out, run = _invoke(_agent(steps), log)
    assert len(_tools_run(out)) == 24
    assert not log.run_stalled(run)


def test_computing_without_fetching_is_not_counted():
    """The shape that ruled out counting every step: correctly answered archived runs that
    computed on a DEM for 18 steps, printing figures and writing no file. They fetch nothing."""
    log = turn_log.new_log()
    steps = [[_call("execute_code", {"code": f"print(slope_stat({i}))"})] for i in range(20)]
    out, run = _invoke(_agent(steps, final="mean slope 4.2"), log)
    assert len(_tools_run(out)) == 20
    assert not log.run_stalled(run)


def test_debugging_is_judged_by_the_progress_rule_not_this_one():
    """Edit, run, fail with a new error, eight times over, then a run that writes the output."""
    kinds = ["NameError", "KeyError", "TypeError", "ValueError", "IndexError", "OSError",
             "ZeroDivisionError", "AttributeError"]
    errors = iter(kinds + ["ok"])

    @tool
    def run_script(path: str) -> str:
        """Run a script."""
        e = next(errors)
        if e == "ok":
            return json.dumps({"ok": True, "artifacts": [{"file_id": "file_0000000beef1",
                                                          "kind": "output"}]})
        return json.dumps({"ok": False, "error": f"{e}: something else broke"})

    @tool
    def edit_file(path: str, text: str) -> str:
        """Edit a workspace file."""
        return json.dumps({"ok": True, "path": path, "chars": len(text)})

    steps = []
    for i in range(8):
        steps += [[_call("edit_file", {"path": "a.py", "text": f"v{i}"})],
                  [_call("run_script", {"path": f"a.py#{i}"})]]
    steps.append([_call("run_script", {"path": "a.py#final"})])
    agent = create_agent(model=ToolAwareFake(messages=iter(
        [AIMessage(content="", tool_calls=c) for c in steps] + [AIMessage(content="fixed")])),
        tools=[run_script, edit_file], system_prompt="t", middleware=ef._default_middleware())
    log = turn_log.new_log()
    with turn_log.bind(log, "code") as b:
        out = agent.invoke({"messages": [HumanMessage(content="q")]},
                           config={"recursion_limit": 400})
    assert out["messages"][-1].content == "fixed"
    assert not log.run_stalled(b.run)


def test_new_evidence_documents_count_as_produced():
    """A search run reading pages between searches: the documents each search returns reset
    the count, so twelve page fetches over the run are fine."""
    log = turn_log.new_log()
    steps = []
    for i in range(12):
        steps += [[_call("stage_url", {"url": f"https://docs.example.org/p{i}.html"})],
                  [_call("keyword_search", {"query": f"q{i}"})]]
    out, run = _invoke(_agent(steps, final="found"), log, peer="search")
    assert out["messages"][-1].content == "found"
    assert not log.run_stalled(run)


@pytest.mark.parametrize("value, expected_calls", [("3", 5), ("0", 30)])
def test_the_limit_can_be_set_and_turned_off(monkeypatch, value, expected_calls):
    monkeypatch.setenv("AGENT_FETCH_STALL_STEPS", value)
    log = turn_log.new_log()
    out, _ = _invoke(_agent(_probe_steps(30)), log)
    assert len(_tools_run(out)) == expected_calls


def test_a_peer_whose_run_stalled_is_not_run_again():
    """The incident's turn bounced between peers before the code peer probed. A run the stall
    rule ended counts as one that added nothing, so the supervisor does not send the same need
    back to the same peer to probe the same sources again."""
    log = turn_log.new_log()
    _, run = _invoke(_agent(_probe_steps(30)), log)
    assert log.run_was_productive(run)               # it did record new results...
    state = {"turn_log_id": log.id}
    assert g._last_run_added_nothing("code", state)  # ...but went nowhere

    fresh = turn_log.new_log()
    _invoke(_agent([[_call("process_tile", {"tile": 1})]]), fresh)
    assert not g._last_run_added_nothing("code", {"turn_log_id": fresh.id})


# --------------------------------------------------------------------------- produced()

def test_what_counts_as_produced():
    src = {"ok": True, "artifacts": [{"file_id": "file_aaaaaaaaaaaa", "kind": "source"},
                                     {"file_id": "file_bbbbbbbbbbbb", "kind": "output"}],
           "input_files": [{"file_id": "file_cccccccccccc"}],
           "stdout": "read file_cccccccccccc"}
    assert turn_log.produced(json.dumps(src)) == ["file_bbbbbbbbbbbb"]
    cut = json.dumps(src)[:-40]                      # not whole JSON any more
    assert "file_aaaaaaaaaaaa" not in turn_log.produced(cut)
    assert turn_log.produced('{"ok": true, "layer_id": "layer_0123456789ab"}') == \
        ["layer_0123456789ab"]
    assert turn_log.produced('{"documents": [{"doc_id": "d1"}, {"doc_id": "d1"}]}') == ["doc:d1"]
    fetched = '{"ok": true, "file_id": "file_dddddddddddd"}'
    assert turn_log.produced(fetched, {"url": "https://x.org/a.csv"}) == []
    assert turn_log.produced(fetched, {"urls": ["http://x.org/a", "b"]}) == []
    assert turn_log.produced(fetched, {"path": "a.csv"}) == ["file_dddddddddddd"]


def test_fetched_hosts():
    assert turn_log.fetched_hosts({"url": " HTTPS://Data.Example.org/x?y=1", "n": 3}) == \
        ["data.example.org"]
    assert turn_log.fetched_hosts({"code": "requests.get('https://a.org')"}) == []
    assert turn_log.fetched_hosts(None) == []
