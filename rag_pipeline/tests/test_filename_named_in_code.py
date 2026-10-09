"""A program that opens a conversation file by its FILENAME finds it, and a run that writes under
an input's name keeps what it wrote.

PR #93 (S30.7) staged a file_id the code names. The other half of the same failure stayed open:
`gpd.read_file('Champaign_County.geojson')` with no `input_files`, after admin_boundary had made
that file in the conversation. Seen in the local replay of 2026-10-08 on prototype 40356bd and in
PR #85's replays: the first execute_code failed with "No such file", and the next run listed it.

A filename is not unique the way an id is, so the scan is narrower than #93's: only files THIS
conversation made for THIS owner. And it could not land before the second fix here. Staged names
were left out of the outputs and the copy back to the workspace, so once a filename the code
WRITES could be staged, `gdf.to_file('schools.geojson')` over an earlier schools.geojson would
have produced nothing at all.
"""

from __future__ import annotations

import json

import pytest

from agent_runtime import artifacts, file_store, identity
from agent_runtime import langchain_exec_tools as exec_tools
from agent_runtime.code_execution import ExecResult, LocalSubprocessExecutor, session_workspace_dir

GEOJSON = json.dumps({"type": "FeatureCollection", "features": []})


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "store"))
    monkeypatch.setenv("AGENT_CODE_EXEC_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("AGENT_PUBLIC_BASE_URL", "")
    monkeypatch.delenv("UPLOAD_FOLDER", raising=False)
    return tmp_path


def as_caller(user, session, fn):
    user_token = identity.set_user(identity.User(id=user, role=4) if user else None)
    session_token = file_store.set_session(session)
    try:
        return fn()
    finally:
        file_store.reset_session(session_token)
        identity.reset_user(user_token)


def made(filename="Champaign_County.geojson", content=GEOJSON, *, user="alice",
         session="conv-alice"):
    """What a tool such as admin_boundary leaves behind: a stored output of the conversation."""
    return as_caller(user, session,
                     lambda: file_store.create_output_file(filename, content))


class _Capture:
    def __init__(self):
        self.input_files = None

    def execute(self, code, language="python", timeout=None, dependencies=None,
                input_files=None, **kwargs):
        self.input_files = input_files
        return ExecResult(exit_code=0, stdout="", stderr="", backend="stub")


def run(tool_args, *, user="alice", session="conv-alice", executor=None):
    tools = exec_tools.make_code_execution_tools(executor=executor or LocalSubprocessExecutor(),
                                                 session_id=session)
    return as_caller(user, session, lambda: json.loads(tools[0].invoke(tool_args)))


# --- the live failure ------------------------------------------------------------------------

def test_code_that_opens_a_conversation_file_by_its_filename_runs():
    """The replay's first run in shape: the filename in the code, nothing in input_files."""
    fid = made()["file_id"]

    out = run({"code": "print(open('Champaign_County.geojson').read())"})

    assert out["ok"] is True, out
    assert "FeatureCollection" in out["stdout"]
    staged = out["input_files"][0]
    assert staged["file_id"] == fid
    assert staged["available_as"] == [fid, "Champaign_County.geojson"]
    assert staged["staged_because"] == "named in the code"


def test_the_newest_file_of_that_name_in_the_conversation_is_the_one_staged():
    """As find_files and resolve_file_ref order them, so the name means what a lookup returns."""
    import os
    import time

    old = made(content='{"v": "old"}')
    new = made(content='{"v": "new"}')
    now = time.time()
    for rec, t in ((old, now - 60), (new, now)):
        path = file_store.resolve_file_id(rec["file_id"])
        os.utime(path, (t, t))

    out = run({"code": "print(open('Champaign_County.geojson').read())"})

    assert '"new"' in out["stdout"], out
    assert [s["file_id"] for s in out["input_files"]] == [new["file_id"]]


def test_a_path_in_front_of_the_name_still_counts_and_a_longer_name_does_not():
    made("data.csv", "a\n1\n")
    ex = _Capture()

    run({"code": "open('./data.csv')"}, executor=ex)
    assert {s["dest"] for s in ex.input_files} >= {"data.csv"}

    for code in ("open('my_data.csv')", "open('data.csv.bak')", "open('data.csvx')"):
        ex = _Capture()
        run({"code": code}, executor=ex)
        assert ex.input_files == [], code


# --- scope: this conversation, this owner, never a path --------------------------------------

def test_another_conversations_file_of_that_name_is_not_staged():
    """The same user, another conversation: a name means a file only inside its conversation."""
    made(session="conv-alice-older")
    ex = _Capture()

    out = run({"code": "open('Champaign_County.geojson')"}, executor=ex)

    assert ex.input_files == []
    assert "input_files" not in out


def test_another_users_file_of_that_name_is_not_staged_even_under_the_same_conversation_id():
    """A thread id is client input; a borrowed one must not reach another user's file."""
    made(user="bob", session="conv-alice")
    ex = _Capture()

    run({"code": "open('Champaign_County.geojson')"}, executor=ex)

    assert ex.input_files == []


def test_another_users_file_does_not_reach_the_sandbox_end_to_end():
    made(user="bob", session="conv-bob")

    out = run({"code": "print(open('Champaign_County.geojson').read())"})

    assert out["ok"] is False
    assert "FeatureCollection" not in out["stdout"]


def test_the_unowned_legacy_pool_is_not_staged_by_name():
    """find_files offers session-less records for reuse; a bare name in code is not a reuse."""
    made(user=None, session=None)
    ex = _Capture()

    run({"code": "open('Champaign_County.geojson')"}, executor=ex)

    assert ex.input_files == []


def test_without_a_conversation_nothing_is_staged_by_name():
    made(user=None, session=None)
    ex = _Capture()

    tools = exec_tools.make_code_execution_tools(executor=ex)
    as_caller(None, None, lambda: tools[0].invoke({"code": "open('Champaign_County.geojson')"}))

    assert ex.input_files == []


# --- the scan only fills a gap ---------------------------------------------------------------

def test_a_name_the_workspace_already_holds_is_left_to_the_workspace():
    """A file there is one an earlier run wrote, and the program means that one."""
    made()
    workspace = session_workspace_dir("conv-alice")
    (workspace / "Champaign_County.geojson").write_text('{"mine": true}', encoding="utf-8")

    out = run({"code": "print(open('Champaign_County.geojson').read())"})

    assert out["stdout"].strip() == '{"mine": true}', out
    assert "input_files" not in out


def test_a_name_an_attached_input_already_has_keeps_its_owner():
    """An explicitly listed file named data.csv owns that name; the scan adds no rival."""
    mine = made("data.csv", "listed\n")
    made("data.csv", "other\n")
    ex = _Capture()

    run({"code": "open('data.csv')", "input_files": [mine["file_id"]]}, executor=ex)

    assert [s["source"] for s in ex.input_files if s["dest"] == "data.csv"] == [
        str(file_store.resolve_file_id(mine["file_id"]))]
    assert len({s["source"] for s in ex.input_files}) == 1


# --- the hazard: writing under an input's name -----------------------------------------------

def test_a_run_that_writes_under_a_staged_name_keeps_its_output():
    """Fails on 23cfd02: the output was excluded as a staged input and existed nowhere."""
    rec = made("schools.geojson", '{"v": "input"}')

    out = run({"code": "open('schools.geojson', 'w').write('{\"v\": \"output\"}')",
               "input_files": [rec["file_id"]]})

    assert out["ok"] is True, out
    written = [a for a in out["artifacts"] if a["filename"] == "schools.geojson"]
    assert len(written) == 1, out["artifacts"]
    path = as_caller("alice", "conv-alice",
                     lambda: file_store.resolve_file_id(written[0]["file_id"]))
    assert path.read_text(encoding="utf-8") == '{"v": "output"}'
    # The input itself is untouched in the store.
    assert file_store.resolve_file_id(rec["file_id"]).read_text(encoding="utf-8") == \
        '{"v": "input"}'
    # …and the next run reads what this one wrote, from the workspace.
    ws = session_workspace_dir("conv-alice")
    assert (ws / "schools.geojson").read_text(encoding="utf-8") == '{"v": "output"}'


def test_writing_a_name_staged_because_the_code_named_it_keeps_the_output():
    """The case the filename scan creates: the code names schools.geojson only to write it."""
    made("schools.geojson", '{"v": "earlier"}')

    out = run({"code": "open('schools.geojson', 'w').write('{\"v\": \"now\"}')"})

    assert out["ok"] is True, out
    assert [a["filename"] for a in out["artifacts"]].count("schools.geojson") == 1
    ws = session_workspace_dir("conv-alice")
    assert (ws / "schools.geojson").read_text(encoding="utf-8") == '{"v": "now"}'


def test_an_input_only_read_is_still_not_an_output():
    rec = made("in.csv", "a\n1\n")

    out = run({"code": "print(open('in.csv').read())", "input_files": [rec["file_id"]]})

    names = [a["filename"] for a in out["artifacts"]]
    assert "in.csv" not in names and rec["file_id"] not in names
    assert not (session_workspace_dir("conv-alice") / "in.csv").exists()


def test_rewriting_identical_bytes_is_not_an_output():
    rec = made("in.csv", "a\n1\n")

    out = run({"code": "d = open('in.csv').read(); open('in.csv', 'w').write(d)",
               "input_files": [rec["file_id"]]})

    assert "in.csv" not in [a["filename"] for a in out["artifacts"]]


def test_the_manifest_hashes_the_input_the_run_read_not_what_it_wrote(monkeypatch):
    monkeypatch.setenv("AGENT_ARTIFACT_EMIT", "1")
    if not artifacts.artifacts_enabled():
        pytest.skip("artifact emission is not switchable by AGENT_ARTIFACT_EMIT here")
    import hashlib

    rec = made("schools.geojson", '{"v": "input"}')
    run({"code": "open('schools.geojson', 'w').write('{\"v\": \"output\"}')",
         "input_files": [rec["file_id"]]})

    rows = [json.loads(line) for line in
            (session_workspace_dir("conv-alice") / artifacts.INPUTS_FILENAME)
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    row = next(r for r in rows if r["name"] == "schools.geojson")
    assert row["sha256"] == hashlib.sha256(b'{"v": "input"}').hexdigest()
