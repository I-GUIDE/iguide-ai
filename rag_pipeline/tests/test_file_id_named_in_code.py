"""A program that opens an input by the file_id a tool handed it finds the file.

Every file-making tool (admin_boundary, overpass_search, add_map_layer, an upload) returns a
`file_id`, and staging puts each input under that id as well as its filename. But only the inputs
`input_files` listed were staged. In all three live turns of 2026-10-08 the first execute_code
was `gpd.read_file("file_2272c8426ec9")` with no `input_files` (threads sess-1e8e3edd 19:12 UTC,
sess-07bc717f 21:34 UTC, and the PR #85 replays). The read failed, the next run listed an empty
directory, and the third run passed `input_files`. One or two sandbox runs per turn, spent on an
interface that took the id as a name in one place and not in the other.

The negative tests matter as much: a named id goes through the same owner check as a listed one,
so naming another user's id in code stages nothing.
"""

from __future__ import annotations

import io
import json

import pytest
from werkzeug.datastructures import FileStorage

from agent_runtime import file_store, identity
from agent_runtime import langchain_exec_tools as exec_tools
from agent_runtime.code_execution import ExecResult, LocalSubprocessExecutor


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


def boundary(user="alice", session="conv-alice"):
    """What admin_boundary leaves behind: a stored output with a minted id."""
    return as_caller(user, session, lambda: file_store.create_output_file(
        "Champaign_County.geojson",
        json.dumps({"type": "FeatureCollection", "features": []})))


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

def test_code_that_opens_an_input_by_its_file_id_runs():
    """The 19:12 UTC call, verbatim in shape: the id in the code, nothing in input_files."""
    fid = boundary()["file_id"]

    out = run({"code": f"print(open('{fid}').read())"})

    assert out["ok"] is True, out
    assert "FeatureCollection" in out["stdout"]
    staged = out["input_files"][0]
    assert staged["file_id"] == fid
    assert staged["available_as"] == [fid, "Champaign_County.geojson"]
    assert staged["staged_because"] == "named in the code"


def test_the_filename_of_a_named_id_is_reachable_too():
    """Staged through the same allocator, so the file has both its names, as a listed one does."""
    fid = boundary()["file_id"]

    out = run({"code": f"# reads {fid}\nprint(open('Champaign_County.geojson').read())"})

    assert out["ok"] is True, out
    assert "FeatureCollection" in out["stdout"]


def test_an_entrypoint_file_that_names_an_id_gets_it_staged():
    """An entrypoint run has no inline code; the file it runs is what names the input."""
    fid = boundary()["file_id"]
    tools = exec_tools.make_code_execution_tools(executor=LocalSubprocessExecutor(),
                                                 session_id="conv-alice")
    write = next(t for t in tools if t.name == "write_workspace_file")
    as_caller("alice", "conv-alice", lambda: write.invoke(
        {"path": "main.py", "content": f"print(open('{fid}').read())\n"}))

    out = run({"entrypoint": "main.py"})

    assert out["ok"] is True, out
    assert "FeatureCollection" in out["stdout"]


# --- what must not change --------------------------------------------------------------------

def test_listing_the_id_in_input_files_is_unchanged():
    fid = boundary()["file_id"]

    out = run({"code": "print(open('Champaign_County.geojson').read())", "input_files": [fid]})

    assert out["ok"] is True, out
    staged = out["input_files"]
    assert len(staged) == 1 and staged[0]["available_as"] == [fid, "Champaign_County.geojson"]
    assert "staged_because" not in staged[0]


def test_an_id_both_listed_and_named_is_staged_once():
    fid = boundary()["file_id"]
    ex = _Capture()

    out = run({"code": f"open('{fid}')", "input_files": [fid]}, executor=ex)

    assert sorted(s["dest"] for s in ex.input_files) == sorted([fid, "Champaign_County.geojson"])
    assert len(out["input_files"]) == 1
    assert "input_file_errors" not in out


def test_code_naming_no_id_stages_nothing():
    ex = _Capture()

    out = run({"code": "print('file_2272c8426ec9x', 'myfile_2272c8426ec9', 'file_demo')"},
              executor=ex)

    assert ex.input_files == []
    assert "input_files" not in out and "input_file_errors" not in out


# --- ownership (Stage 30): naming an id is no wider than listing it ---------------------------

def test_another_users_id_named_in_code_is_not_staged():
    fid = boundary(user="alice", session="conv-alice")["file_id"]
    ex = _Capture()

    out = run({"code": f"print(open('{fid}').read())"}, user="bob", session="conv-bob",
              executor=ex)

    assert ex.input_files == []
    assert "input_files" not in out
    assert out["input_file_errors"] == [
        {"ref": fid, "error": "named in the code, but no file you can read has this file_id"}]


def test_another_users_id_does_not_reach_the_sandbox_end_to_end():
    fid = boundary(user="alice", session="conv-alice")["file_id"]

    out = run({"code": f"print(open('{fid}').read())"}, user="bob", session="conv-bob")

    assert out["ok"] is False
    assert "FeatureCollection" not in out["stdout"]


def test_an_id_never_minted_is_refused_like_another_users():
    """The message is the same either way, so it does not tell bob that alice's id exists."""
    ex = _Capture()

    out = run({"code": "open('file_000000000000')"}, user="bob", session="conv-bob", executor=ex)

    assert ex.input_files == []
    assert out["input_file_errors"] == [
        {"ref": "file_000000000000",
         "error": "named in the code, but no file you can read has this file_id"}]


def test_a_named_id_of_an_upload_from_the_same_user_is_staged():
    """Uploads carry no conversation (the store binds none at /agent/files/upload), so the check
    is the owner's, as for a listed id."""
    rec = as_caller("alice", None, lambda: file_store.save_uploaded_file(
        FileStorage(stream=io.BytesIO(b"a,b\n1,2\n"), filename="data.csv")))
    ex = _Capture()

    run({"code": f"open('{rec['file_id']}')"}, executor=ex)

    assert {s["dest"] for s in ex.input_files} == {rec["file_id"], "data.csv"}
