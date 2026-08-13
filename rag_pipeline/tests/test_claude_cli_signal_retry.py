"""When the CLI subprocess is killed by a signal rather than answering.

Observed live, twice in one server log::

    RuntimeError: claude CLI exited -11 (model=sonnet):      <- empty stderr
    During task with name 'model' ... 'search' ... 'orchestrate'

A negative ``returncode`` means the process was killed by a signal (-11 is SIGSEGV). That is a
crash of the tool, not an answer about the request, and it is transient by nature. The exception
propagated out of the langgraph node, through ``graph_runtime``'s ``raise worker_error[0]``, and
out of the SSE stream — discarding a turn that had already completed two search sweeps, resolved
a bounding box, selected a library method and written an evidence summary.

The asymmetry is the whole design here. Retrying a signal death costs one subprocess. Retrying an
auth failure, a budget refusal or an API error costs the user's quota to reach the same
conclusion slower, so those are deliberately NOT retried — which is why every mode is asserted
below rather than just the happy path.

Pure: ``subprocess.run`` is replaced, so nothing is executed and no quota is spent.
"""

from __future__ import annotations

import subprocess
import types

import pytest

from rag_pipeline import llm_claude_cli


class _Proc:
    """Just enough of CompletedProcess for the code under test."""

    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


OK = _Proc(0, '{"type":"result","result":"ANSWER"}')
SEGV = _Proc(-11, "", "")
SIGKILL = _Proc(-9, "", "")
AUTH = _Proc(1, '{"is_error":true,"result":"Failed to authenticate. '
                'API Error: 401 OAuth access token has expired."}')
API_ERROR = _Proc(1, '{"is_error":true,"subtype":"error","result":"API Error: 500"}')
PLAIN_NONZERO = _Proc(2, "", "budget exceeded")
NO_OUTPUT = _Proc(0, "")


@pytest.fixture()
def scripted(monkeypatch):
    """Feed a sequence of fake subprocess results; report how many were consumed."""
    monkeypatch.setattr(llm_claude_cli.shutil, "which", lambda _name: "/usr/bin/claude")
    monkeypatch.setattr(llm_claude_cli, "_load_token_from_env_file", lambda: None)
    monkeypatch.setattr(llm_claude_cli, "time", types.SimpleNamespace(sleep=lambda _s: None))
    calls = {"n": 0}

    def install(*results):
        seq = list(results)

        def run(*_a, **_k):
            calls["n"] += 1
            return seq[min(calls["n"] - 1, len(seq) - 1)]

        monkeypatch.setattr(llm_claude_cli, "subprocess",
                            types.SimpleNamespace(run=run,
                                                  TimeoutExpired=subprocess.TimeoutExpired))
        return calls

    return install


# ------------------------------------------------------------------ a signal death is retried

def test_a_segfault_followed_by_success_returns_the_answer(scripted):
    calls = scripted(SEGV, OK)
    assert llm_claude_cli.call("hi") == "ANSWER"
    assert calls["n"] == 2, "the crash must not reach the caller when a retry succeeds"


def test_two_segfaults_still_recover(scripted):
    calls = scripted(SEGV, SEGV, OK)
    assert llm_claude_cli.call("hi") == "ANSWER"
    assert calls["n"] == 3


def test_a_persistent_crash_gives_up_and_says_how_many_times_it_tried(scripted):
    """A blip and a reproducible crash need different responses from whoever reads the log."""
    calls = scripted(SEGV)
    with pytest.raises(RuntimeError) as err:
        llm_claude_cli.call("hi")
    assert calls["n"] == 3, "1 initial attempt + 2 retries"
    assert "signal 11" in str(err.value)
    assert "3 attempt" in str(err.value)


def test_sigkill_is_treated_the_same_as_sigsegv(scripted):
    """Any signal death is a crash. -9 is what an OOM killer leaves behind."""
    calls = scripted(SIGKILL, OK)
    assert llm_claude_cli.call("hi") == "ANSWER"
    assert calls["n"] == 2


def test_the_message_says_signal_not_exit_status(scripted):
    """`exited -11` is wrong and unreadable: -11 is not an exit status. The returncode's sign is
    the only place that information exists, and it is gone by the time anyone reads the message."""
    calls = scripted(SEGV)
    with pytest.raises(RuntimeError) as err:
        llm_claude_cli.call("hi")
    assert "killed by signal 11" in str(err.value)
    assert "no diagnostic output" in str(err.value), (
        "an empty stderr is itself informative and must not read as a missing message")
    assert calls["n"] == 3


# ------------------------------------------------------------------ everything else is NOT

def test_a_clean_run_starts_exactly_one_subprocess(scripted):
    calls = scripted(OK)
    assert llm_claude_cli.call("hi") == "ANSWER"
    assert calls["n"] == 1


def test_an_auth_failure_is_not_retried(scripted):
    """Reproducible by definition. Retrying spends the user's quota to fail identically, and it
    delays the one message that tells them what to do."""
    calls = scripted(AUTH)
    with pytest.raises(llm_claude_cli.ClaudeCliUnavailable):
        llm_claude_cli.call("hi")
    assert calls["n"] == 1


def test_a_structured_api_error_is_not_retried(scripted):
    calls = scripted(API_ERROR)
    with pytest.raises(RuntimeError):
        llm_claude_cli.call("hi")
    assert calls["n"] == 1


def test_a_plain_nonzero_exit_is_not_retried(scripted):
    calls = scripted(PLAIN_NONZERO)
    with pytest.raises(RuntimeError):
        llm_claude_cli.call("hi")
    assert calls["n"] == 1


def test_empty_output_is_not_retried(scripted):
    """A successful exit with no output is a contract violation, not a crash."""
    calls = scripted(NO_OUTPUT)
    with pytest.raises(RuntimeError):
        llm_claude_cli.call("hi")
    assert calls["n"] == 1


def test_a_timeout_is_not_retried(scripted, monkeypatch):
    """A retry would burn another full timeout and land in the same place."""
    monkeypatch.setattr(llm_claude_cli.shutil, "which", lambda _n: "/usr/bin/claude")

    def run(*_a, **_k):
        raise subprocess.TimeoutExpired(cmd="claude", timeout=1)

    monkeypatch.setattr(llm_claude_cli, "subprocess",
                        types.SimpleNamespace(run=run,
                                              TimeoutExpired=subprocess.TimeoutExpired))
    with pytest.raises(RuntimeError, match="timed out"):
        llm_claude_cli.call("hi")


# ------------------------------------------------------------------ the knob

def test_retries_can_be_switched_off(scripted, monkeypatch):
    """0 restores the previous behaviour exactly, so the change is reversible in an incident
    without a deploy."""
    monkeypatch.setenv("CLAUDE_CLI_SIGNAL_RETRIES", "0")
    calls = scripted(SEGV)
    with pytest.raises(RuntimeError):
        llm_claude_cli.call("hi")
    assert calls["n"] == 1


@pytest.mark.parametrize("raw,expected", [("0", 0), ("1", 1), ("5", 5),
                                          ("99", 5), ("-3", 0), ("banana", 2), ("", 2)])
def test_the_retry_count_is_clamped_and_never_raises(raw, expected, monkeypatch):
    """A typo in an env var must not crash the LLM path, and an unbounded value must not turn a
    reproducible crash into a very slow reproducible crash."""
    monkeypatch.setenv("CLAUDE_CLI_SIGNAL_RETRIES", raw)
    assert llm_claude_cli._signal_retries() == expected


# ------------------------------------------------------------------ a crash AFTER the answer

SEGV_AFTER_ANSWER = _Proc(-11, '{"type":"result","is_error":false,"result":"REAL ANSWER"}')
SEGV_PARTIAL = _Proc(-11, '{"type":"result","is_error":true,"subtype":"crash","result":"partial"}')


def test_a_completed_answer_is_never_discarded_over_an_exit_status(scripted):
    """The CLI writes its result object when the turn FINISHES. If the process is then killed
    during teardown, the work is done and paid for.

    The first version of this retry did not help here at all, because the payload branch is
    evaluated before the returncode and raised:

        RuntimeError: claude CLI error (exit=-11): REAL ANSWER

    — putting the answer inside the text of the exception complaining about it. Found by an
    adversarial audit of the commit that added the retry, not by the tests that came with it.
    """
    calls = scripted(SEGV_AFTER_ANSWER)
    assert llm_claude_cli.call("hi") == "REAL ANSWER"
    assert calls["n"] == 1, "a finished answer must not be re-requested either"


def test_a_signal_death_with_partial_output_is_still_retried(scripted):
    """Checking the signal only in the no-payload branch made the retry unreachable whenever the
    dying process had already flushed some JSON — which is the common case, since it writes
    progress as it goes."""
    calls = scripted(SEGV_PARTIAL, OK)
    assert llm_claude_cli.call("hi") == "ANSWER"
    assert calls["n"] == 2


# ------------------------------------------------------------------ transient upstream statuses

def _api(status: int, message: str = "upstream said no") -> _Proc:
    import json as _json

    return _Proc(1, _json.dumps({"type": "result", "is_error": True,
                                 "api_error_status": status, "result": message}))


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504, 529])
def test_a_transient_upstream_status_is_retried(scripted, status):
    """429 is a rate limit and 529 is Anthropic's overloaded signal; both are explicitly
    retryable, and 5xx is a server fault rather than a property of the prompt. Classified from the
    structured `api_error_status` field, not by matching the wording of an error string."""
    calls = scripted(_api(status), OK)
    assert llm_claude_cli.call("hi") == "ANSWER"
    assert calls["n"] == 2


@pytest.mark.parametrize("status", [400, 403, 404, 422])
def test_a_permanent_upstream_status_is_not_retried(scripted, status):
    """Retrying a malformed request reaches the same answer slower while spending quota."""
    calls = scripted(_api(status))
    with pytest.raises(RuntimeError):
        llm_claude_cli.call("hi")
    assert calls["n"] == 1


def test_a_401_is_an_auth_failure_not_a_transient_status(scripted):
    """It has a fix the user must perform, so it must reach them immediately rather than after
    three attempts — and it must carry the re-authentication hint."""
    calls = scripted(_api(401, "Failed to authenticate. API Error: 401 OAuth access token "
                               "has expired. Re-authenticate to continue."))
    with pytest.raises(llm_claude_cli.ClaudeCliUnavailable):
        llm_claude_cli.call("hi")
    assert calls["n"] == 1


def test_a_persistent_rate_limit_gives_up_rather_than_hammering(scripted):
    calls = scripted(_api(429))
    with pytest.raises(RuntimeError, match="429"):
        llm_claude_cli.call("hi")
    assert calls["n"] == 3
