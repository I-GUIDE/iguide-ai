"""A call is not a run, a dead end gets an intervention, and a dead peer is not a dead turn.

`result["executed"]` is read downstream as "the code ran", and it was derived from whether
execute_code was CALLED — so a non-zero exit was reported as a success and synthesis described
a failed run as a working one. The sandbox reports failure as DATA (`ok` is computed from
exit_code/timeout/error), so the outcome has to be read out of the payload.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime.supervisor import graph as g


def _exec_result(ok, error="", exit_code=0):
    return {"name": "execute_code", "tool_call_id": "c1",
            "content": json.dumps({"ok": ok, "exit_code": exit_code, "stdout": "",
                                   "stderr": error, "error": error or None})}


# --- 1. a call is not a run ----------------------------------------------------------------

def test_a_failed_run_is_not_reported_as_executed():
    ran, err = g._execution_outcome({"tool_results": [_exec_result(False, "ModuleNotFoundError: pysal", 1)]})
    assert ran is False
    assert "pysal" in err


def test_a_successful_run_is():
    ran, err = g._execution_outcome({"tool_results": [_exec_result(True)]})
    assert ran is True and err == ""


def test_a_failure_followed_by_a_success_counts_as_run():
    """The run/read/fix loop is the intended path — fixing it is success, not failure."""
    ran, _err = g._execution_outcome({"tool_results": [
        _exec_result(False, "NameError", 1), _exec_result(True)]})
    assert ran is True


def test_the_call_check_still_answers_its_own_question():
    """_has_execution_record gates the did-you-run-it retry, which is about a peer that never
    tried. A failed run DID try, so telling it "you did not run it" would be false."""
    arts = {"tool_calls": [{"name": "execute_code"}],
            "tool_results": [_exec_result(False, "boom", 1)]}
    assert g._has_execution_record(arts) is True
    assert g._execution_outcome(arts)[0] is False


def test_an_unparseable_result_is_not_a_success():
    ran, _ = g._execution_outcome({"tool_results": [
        {"name": "execute_code", "content": "<not json>"}]})
    assert ran is False


# --- 2. the dead-end detector reads what execute_code emits ---------------------------------

def test_two_identical_sandbox_failures_are_a_dead_end():
    """The detector already parsed the `ok` key execute_code carries; it was simply never
    wired into this peer."""
    stuck = g._repeatedly_failed_tools({"tool_results": [
        _exec_result(False, "ModuleNotFoundError: pysal", 1),
        _exec_result(False, "ModuleNotFoundError: pysal", 1)]})
    assert "execute_code" in stuck
    assert "pysal" in stuck["execute_code"]


def test_one_failure_is_not_a_dead_end():
    stuck = g._repeatedly_failed_tools({"tool_results": [_exec_result(False, "boom", 1)]})
    assert stuck == {}


# --- 3. a dead peer is not a dead turn ------------------------------------------------------


def test_a_spiralling_code_peer_does_not_kill_the_turn():
    """The real graph, with a peer that raises the way the recursion limit does. code_node had
    no handler, so this propagated out of the supervisor and the user got an SSE error and no
    answer at all — even when the other peers had produced something worth saying."""
    from langgraph.errors import GraphRecursionError

    from agent_runtime.supervisor.graph import run_supervisor
    from rag_pipeline.tests.test_supervisor_graph import _fake_llm, _scripted

    def _spiral(q, ev, st):
        raise GraphRecursionError("Recursion limit of 60 reached")

    state = run_supervisor(
        "write code", llm=_fake_llm,
        decide_fn=_scripted(["code", "done"]),
        search_fn=lambda q, s: [],
        code_fn=_spiral,
        synthesize_fn=lambda q, ev, ar, cr, ch, pa=None: f"final:{(cr or {}).get('error', '')}",
        do_audit=False,
    )
    # The contract is that the TURN survives and says so — not that any particular synthesis
    # path runs. With no evidence and an empty code answer the supervisor legitimately falls to
    # its general-answer route, which is still an answer where there used to be an SSE error.
    assert isinstance(state.get("final_answer"), str) and state["final_answer"]
    assert "GraphRecursionError" in state["code_result"]["error"]
    assert state["code_result"]["executed"] is False


def test_a_working_code_peer_is_unaffected():
    """The guard must not swallow a normal run."""
    from agent_runtime.supervisor.graph import run_supervisor
    from rag_pipeline.tests.test_supervisor_graph import _fake_llm, _scripted

    state = run_supervisor(
        "write code", llm=_fake_llm,
        decide_fn=_scripted(["code", "done"]),
        search_fn=lambda q, s: [],
        code_fn=lambda q, ev, st: {"answer": "code-answer", "executed": True},
        synthesize_fn=lambda q, ev, ar, cr, ch, pa=None: f"final:{(cr or {}).get('answer', '')}",
        do_audit=False,
    )
    assert state["final_answer"] == "final:code-answer"
    assert "error" not in state["code_result"]


# --- the guard follows the TOOL, not the peer ---------------------------------------------
#
# execute_code is bound to the analyze peer as well, and the router sends most code-shaped
# work there — measured live, 7 of 7 turns. A benchmark run returned a Socrata loader as "the
# code you actually ran" with no execute_code record anywhere in the turn: the code peer's
# guard existed, and analyze never consulted it.

class _FakeSession:
    """Enough of PeerSession for the honesty helper."""

    def __init__(self, reply="ran it", tool_results=None):
        self._reply, self._results = reply, tool_results or []
        self.runs = []
        self.turn_artifacts = {"tool_calls": [], "tool_results": []}

    def run(self, text):
        self.runs.append(text)
        self.turn_artifacts = {"tool_calls": [{"name": "execute_code"}],
                               "tool_results": self._results}

        class _R:
            resp = {"messages": []}
        return _R()


def _apply(result, session, prose_key="answer", caps=(), exec_available=True):
    return g._apply_execution_honesty(
        session, result, prose_key=prose_key, exec_available=exec_available,
        caps=list(caps), node="test")


def test_a_shipped_code_block_with_no_run_is_retried(monkeypatch):
    monkeypatch.setattr(g, "extract_final_answer", lambda *a, **k: "ran it", raising=False)
    session = _FakeSession(tool_results=[_exec_result(True)])
    result = {"answer": "Here:\n```python\nprint(1)\n```", "tool_calls": [], "tool_results": []}
    assert _apply(result, session) is True
    assert session.runs and "never run" in session.runs[0]
    assert result["executed"] is True


def test_the_analyze_peer_gets_the_same_guard(monkeypatch):
    """The whole point: same helper, different prose key."""
    monkeypatch.setattr(g, "extract_final_answer", lambda *a, **k: "ran it", raising=False)
    session = _FakeSession(tool_results=[_exec_result(True)])
    result = {"summary": "Here:\n```python\nprint(1)\n```", "tool_calls": [], "tool_results": []}
    assert _apply(result, session, prose_key="summary") is True
    assert result["executed"] is True


def test_an_answer_with_no_code_block_is_left_alone():
    session = _FakeSession()
    result = {"summary": "Twelve counties border Champaign County.",
              "tool_calls": [], "tool_results": []}
    assert _apply(result, session, prose_key="summary") is False
    assert session.runs == []
    assert result["executed"] is False


def test_a_blocked_peer_is_still_challenged_on_shipped_code(monkeypatch):
    """Being blocked is a reason a peer cannot RUN code. It is not a reason to present unrun
    code as executed — a live turn requested a capability and still returned a network loader
    as "the code you actually ran"."""
    monkeypatch.setattr(g, "extract_final_answer", lambda *a, **k: "noted", raising=False)
    session = _FakeSession(tool_results=[])
    result = {"answer": "```python\nprint(1)\n```", "tool_calls": [], "tool_results": []}
    assert _apply(result, session, caps=["search"]) is True
    assert session.runs, "the challenge must fire even when a capability was requested"
    assert result["executed"] is False


def test_the_blocked_challenge_asks_for_a_label_not_a_run(monkeypatch):
    """Demanding a run from a peer that cannot run is the wrong instruction; the remedy forks
    while the challenge does not."""
    monkeypatch.setattr(g, "extract_final_answer", lambda *a, **k: "noted", raising=False)
    blocked, free = _FakeSession(), _FakeSession()
    _apply({"answer": "```python\nx\n```", "tool_calls": [], "tool_results": []},
           blocked, caps=["search"])
    _apply({"answer": "```python\nx\n```", "tool_calls": [], "tool_results": []}, free)
    assert "UNRUN" in blocked.runs[0] and "keep the request" in blocked.runs[0]
    assert "re-run until it works" in free.runs[0]
    assert "UNRUN" not in free.runs[0]


def test_a_failed_run_is_reported_with_its_error():
    session = _FakeSession()
    result = {"answer": "done", "tool_calls": [{"name": "execute_code"}],
              "tool_results": [_exec_result(False, "ModuleNotFoundError: pysal", 1)]}
    _apply(result, session)
    assert result["executed"] is False
    assert "pysal" in result["execution_error"]


def test_a_later_success_clears_a_stale_error():
    session = _FakeSession()
    result = {"answer": "done", "tool_calls": [{"name": "execute_code"}],
              "tool_results": [_exec_result(False, "NameError", 1), _exec_result(True)],
              "execution_error": "NameError"}
    _apply(result, session)
    assert result["executed"] is True
    assert "execution_error" not in result


def test_no_sandbox_means_no_retry():
    session = _FakeSession()
    result = {"answer": "```python\nprint(1)\n```", "tool_calls": [], "tool_results": []}
    assert _apply(result, session, exec_available=False) is False
    assert session.runs == []


# --- 4. driving the real default_code_fn through the dead end ------------------------------
#
# Everything above unit-tests the predicates. Nothing drove the peer, so the dead-end
# intervention and the honesty call site could both be deleted outright without a single test
# noticing — verified by mutation. These close that, and pin the half of the outcome contract
# that was missing: a re-run that SUCCEEDS must remove the earlier run's error, or a success
# ships beside the failure it just fixed and the auditor reads a working run as a broken one.

class _ScriptedRun:
    """Matches executor_factory.PeerRun: resp, artifacts, answer."""

    def __init__(self, answer, artifacts):
        self.resp = {"answer": answer}
        self.artifacts = artifacts
        self.answer = answer


class _ScriptedSession:
    """Faithful to PeerSession: each run returns only ITS artifacts, turn_artifacts accrues."""

    def __init__(self, script):
        self.script = list(script)
        self.prompts = []
        self.turn_artifacts = {"tool_calls": [], "tool_results": []}

    def run(self, query, chat_history=None):
        self.prompts.append(query)
        answer, results = self.script.pop(0) if self.script else ("done", [])
        calls = [{"name": "execute_code", "id": "c"} for _ in results]
        self.turn_artifacts["tool_calls"].extend(calls)
        self.turn_artifacts["tool_results"].extend(results)
        return _ScriptedRun(answer, {"tool_calls": calls, "tool_results": list(results)})


def _drive_code_peer(monkeypatch, script):
    """Run the real default_code_fn with only the executor boundary stubbed."""
    import agent_runtime.executor_factory as ef
    import agent_runtime.runtime_utils as ru

    session = _ScriptedSession(script)
    monkeypatch.setattr(ef, "open_peer_session", lambda *a, **k: session)
    monkeypatch.setattr(ef, "build_agent_executor", lambda *a, **k: object())
    monkeypatch.setattr(ef, "agent_config", lambda *a, **k: {})
    monkeypatch.setattr(ef, "child_thread_id", lambda *a, **k: "t")
    monkeypatch.setattr(ru, "extract_final_answer", lambda resp: (resp or {}).get("answer", ""))
    monkeypatch.setattr(ru, "extract_search_artifacts",
                        lambda *a, **k: {"tool_calls": [], "tool_results": []})

    fn = g.default_code_fn(llm=object(), code_exec=True)
    return fn("analyse it", [], {"thread_id": "th"}), session


FAILED_RUN = _exec_result(False, "ModuleNotFoundError: pysal", 1)


def test_the_dead_end_intervention_actually_fires(monkeypatch):
    """Two failures of one tool and no capability request: the peer gets one more run."""
    _out, session = _drive_code_peer(monkeypatch, [
        ("no code fence here", [FAILED_RUN, FAILED_RUN]),
        ("fixed it", [_exec_result(True)]),
    ])
    assert len(session.prompts) == 2, "the peer was never handed the dead-end observation"
    assert "failed repeatedly" in session.prompts[1] or "pysal" in session.prompts[1]


def test_a_successful_re_run_clears_the_earlier_error(monkeypatch):
    """The half that was missing. execution_error is set from the first two failures, and the
    dead-end branch re-derived the outcome without removing it."""
    out, _session = _drive_code_peer(monkeypatch, [
        ("no code fence here", [FAILED_RUN, FAILED_RUN]),
        ("fixed it", [_exec_result(True)]),
    ])
    assert out["executed"] is True
    assert "execution_error" not in out, f"stale error survived: {out.get('execution_error')!r}"


def test_a_re_run_that_also_fails_still_reports_the_failure(monkeypatch):
    """Clearing must not become unconditional."""
    out, _session = _drive_code_peer(monkeypatch, [
        ("no code fence here", [FAILED_RUN, FAILED_RUN]),
        ("still broken", [_exec_result(False, "pysal again", 1)]),
    ])
    assert out["executed"] is False
    assert "pysal" in out["execution_error"]


def test_one_extra_run_only(monkeypatch):
    """5f828f4 caps the intervention at one extra run per turn."""
    _out, session = _drive_code_peer(monkeypatch, [
        ("no fence", [FAILED_RUN, FAILED_RUN]),
        ("no fence", [FAILED_RUN, FAILED_RUN]),
        ("no fence", [FAILED_RUN, FAILED_RUN]),
    ])
    assert len(session.prompts) == 2


def test_the_outcome_helper_is_the_only_writer():
    """Both call sites must go through it, or the clear drifts out of one of them again."""
    import inspect
    src = inspect.getsource(g.default_code_fn) + inspect.getsource(g._apply_execution_honesty)
    assert 'result["executed"]' not in src, "executed is being set outside the helper"


# --- 5. a fence is not always code --------------------------------------------------------
#
# The predicate is a bare-fence match, which is a fair proxy on the code peer, whose
# deliverable IS code. Pointed at the analyze peer — whose deliverable is prose plus map
# layers — an ordinary table of counts or a stdout excerpt read as shipped code, spending a
# whole extra model run challenging the peer about code it never wrote.

ANALYZE_PROSE_WITH_A_TABLE = (
    "Counts per community area:\n\n```\nAustin 1204\nLoop 873\n```\n\n"
    "The choropleth is on your interactive map."
)
ANALYZE_PROSE_WITH_TEXT_FENCE = "Result:\n```text\nGEOID  n\n17019  801\n```"
REAL_PYTHON = "Here is the loader:\n\n```python\nimport requests\nr = requests.get(url)\n```"


def test_an_untagged_table_is_not_shipped_code_for_a_prose_peer():
    assert g._ships_unrun_code(ANALYZE_PROSE_WITH_A_TABLE, any_fence=False) is False
    assert g._ships_unrun_code(ANALYZE_PROSE_WITH_TEXT_FENCE, any_fence=False) is False


def test_real_code_is_still_caught_for_a_prose_peer():
    assert g._ships_unrun_code(REAL_PYTHON, any_fence=False) is True


def test_the_code_peer_reading_is_unchanged():
    """Its deliverable is code, so an untagged fence is still a fair proxy there."""
    for text in (ANALYZE_PROSE_WITH_A_TABLE, ANALYZE_PROSE_WITH_TEXT_FENCE, REAL_PYTHON):
        assert g._ships_unrun_code(text) is True


def test_the_analyze_call_site_asks_for_the_stricter_reading():
    """Otherwise the default silently reinstates the false positive."""
    import inspect
    src = inspect.getsource(g.default_analyze_fn)
    assert "any_fence=False" in src


# --- 6. the flag has to REACH the answerer -------------------------------------------------
#
# `executed` exists so synthesis cannot describe a failed run as a working one, and it was left
# to survive a serialization instead of being stated. It did not: in analysis_results it sits
# after tool_calls/tool_results and json.dumps(...)[:2000] cuts it, and for the code peer the
# normal branch sends only `answer` and never dumps the dict at all.

def _synthesis_prompt(analysis=None, code=None):
    seen = {}
    fn = g.default_synthesize_fn(llm=lambda prompt: seen.setdefault("p", prompt) or "answer")
    fn("q", [], analysis, code, [])
    return seen["p"]


def test_a_failed_analysis_run_is_stated_not_buried():
    """The flag itself is genuinely truncated away — the note is what has to arrive."""
    big = {"summary": "s", "tool_calls": [{"x": "y" * 400}],
           "tool_results": [{"c": "z" * 1600}],
           "executed": False, "execution_error": "ModuleNotFoundError: pysal"}
    prompt = _synthesis_prompt(analysis=big)
    assert "did NOT run" in prompt
    assert "pysal" in prompt


def test_a_failed_code_run_is_stated_even_though_only_the_answer_is_sent():
    prompt = _synthesis_prompt(code={
        "answer": "Here is the loader:\n```python\nimport requests\n```",
        "executed": False, "execution_error": "no network in the sandbox"})
    assert "did NOT run" in prompt
    assert "no network in the sandbox" in prompt


def test_a_successful_run_says_so():
    prompt = _synthesis_prompt(code={"answer": "counts are 1204 and 873", "executed": True})
    assert "RAN" in prompt
    assert "did NOT run" not in prompt


def test_a_result_with_no_execution_claim_gets_no_note():
    """Search-only turns must not gain a sentence about code."""
    prompt = _synthesis_prompt(analysis={"summary": "no code here"})
    assert "EXECUTION:" not in prompt


# --- 7. two failures are not automatically the SAME failure --------------------------------

def test_the_latest_error_is_reported_not_the_first():
    """Handing back an error the peer already fixed sends it to re-fix a solved problem."""
    arts = {"tool_results": [
        _exec_result(False, "ModuleNotFoundError: pysal", 1),
        _exec_result(False, "KeyError: 'GEOID'", 1)]}
    err = g._repeatedly_failed_tools(arts)["execute_code"]
    assert "GEOID" in err, err
    assert "2 different errors" in err, "the peer is not told the failures differed"


def test_identical_failures_are_reported_plainly():
    arts = {"tool_results": [
        _exec_result(False, "ModuleNotFoundError: pysal", 1),
        _exec_result(False, "ModuleNotFoundError: pysal", 1)]}
    assert g._repeatedly_failed_tools(arts)["execute_code"] == "ModuleNotFoundError: pysal"
