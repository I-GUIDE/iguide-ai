"""A file named by its bare filename was read from ANY conversation, by ANY user.

`read_text_file` and `inspect_file_for_analysis` turned a bare filename into a path with their own
scan of `uploads/` and `outputs/`: the newest file whose on-disk name ended in `__<filename>`,
whoever wrote it. The store's own lookup, `find_files`, is scoped to the conversation (S7.9) and
to the owner (S9.3), and it was never consulted. Measured on prototype 2026-10-01: alice wrote
`summary.md` in her conversation, and bob read it back by name from his, while
`find_files("summary.md")` returned `[]` for him. `execute_code(input_files=[...])` resolves
through the same helper, so the same name put alice's file in bob's sandbox.

The negative tests are the ones that matter. The positive ones pin what scoping must not break:
a conversation still reads its own file by name, and the unowned legacy pool stays reusable,
which is the answer `find_files` already gives.
"""

from __future__ import annotations

import io
import json
import os

import pytest
from werkzeug.datastructures import FileStorage

from agent_runtime import file_store, identity
from agent_runtime import langchain_exec_tools as exec_tools
from agent_runtime.langchain_file_tools import (inspect_file_for_analysis_tool,
                                                read_text_file_tool, write_text_file_tool)


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("AGENT_PUBLIC_BASE_URL", "")
    monkeypatch.delenv("UPLOAD_FOLDER", raising=False)
    return tmp_path


def as_caller(user, session, fn):
    """Run ``fn`` as ``user`` in conversation ``session``. A user of None is dev or demo mode,
    where nobody is identified; a session of None is a write from before sessions existed."""
    user_token = identity.set_user(identity.User(id=user, role=4) if user else None)
    session_token = file_store.set_session(session)
    try:
        return fn()
    finally:
        file_store.reset_session(session_token)
        identity.reset_user(user_token)


def write(user, session, name, body):
    return as_caller(user, session, lambda: file_store.create_output_file(name, body))


def upload(user, name, body):
    """As `/agent/files/upload` stores a file: the user is bound, and no conversation is."""
    return as_caller(user, None, lambda: file_store.save_uploaded_file(
        FileStorage(stream=io.BytesIO(body), filename=name)))


def backdate(record, seconds=60):
    """Make ``record`` older, so "the newest" is decided by the test and not by the clock."""
    path = file_store.resolve_file_id(record["file_id"])
    st = path.stat()
    os.utime(path, (st.st_atime - seconds, st.st_mtime - seconds))


def _read(name):
    return json.loads(read_text_file_tool(name))["content"]


def _inspect(name):
    content = json.loads(inspect_file_for_analysis_tool(name))["analysis_ready_content"]
    return content.get("content", content)   # a CSV comes back as a header and rows instead


TOOLS = pytest.mark.parametrize("tool", [_read, _inspect],
                                ids=["read_text_file", "inspect_file_for_analysis"])


# --- another user's file is not reachable by name -----------------------------------------

@TOOLS
def test_another_users_file_is_not_read_by_name(tool):
    """The reported case: a different user, in a different conversation."""
    write("alice", "conv-alice", "summary.md", "ALICE PRIVATE")

    assert as_caller("bob", "conv-bob", lambda: file_store.find_files("summary.md")) == [], \
        "precondition: the store's own lookup already hides it"
    with pytest.raises(ValueError):
        as_caller("bob", "conv-bob", lambda: tool("summary.md"))


@TOOLS
def test_a_borrowed_conversation_id_does_not_reach_another_users_file(tool):
    """The conversation id comes from the client, so it cannot be the only gate. Bob sending
    alice's thread id is refused on ownership alone."""
    write("alice", "conv-alice", "summary.md", "ALICE PRIVATE")

    with pytest.raises(ValueError):
        as_caller("bob", "conv-alice", lambda: tool("summary.md"))


@TOOLS
def test_another_users_upload_is_not_read_by_name(tool):
    """The file toolset attaches on an upload turn, and uploads are where common names live."""
    upload("alice", "results.csv", b"ALICE PRIVATE")

    with pytest.raises(ValueError):
        as_caller("bob", "conv-bob", lambda: tool("results.csv"))


def test_a_refusal_does_not_confirm_the_file_exists():
    """The download endpoint answers 404, not 403, for the same reason. An error that differs for
    "exists, but is someone else's" would make the tool an oracle for other users' filenames."""
    write("alice", "conv-alice", "summary.md", "ALICE PRIVATE")

    def refusal(name):
        with pytest.raises(ValueError) as exc:
            as_caller("bob", "conv-bob", lambda: read_text_file_tool(name))
        return str(exc.value).replace(name, "<name>")

    assert refusal("summary.md") == refusal("nothing-by-this-name.md")


# --- another conversation's file is not reachable by name either --------------------------

@TOOLS
def test_the_same_users_other_conversation_is_not_read_by_name(tool):
    write("alice", "conv-alice-1", "summary.md", "FROM CONVERSATION 1")

    with pytest.raises(ValueError):
        as_caller("alice", "conv-alice-2", lambda: tool("summary.md"))


@TOOLS
def test_in_dev_mode_another_conversation_is_not_read_by_name(tool):
    """Nobody is identified in dev or demo mode, so the conversation is the only scope there."""
    write(None, "conv-1", "summary.md", "FROM CONVERSATION 1")

    with pytest.raises(ValueError):
        as_caller(None, "conv-2", lambda: tool("summary.md"))


def test_a_newer_file_elsewhere_does_not_shadow_this_conversations_own():
    """On prototype bob did not even get his OWN summary.md back once alice had written a newer
    one, because the scan took the newest of everyone's."""
    mine = write("bob", "conv-bob", "summary.md", "BOB'S OWN")
    write("alice", "conv-alice", "summary.md", "ALICE PRIVATE")
    backdate(mine)

    assert as_caller("bob", "conv-bob", lambda: _read("summary.md")) == "BOB'S OWN"


# --- what scoping must not break ----------------------------------------------------------

@TOOLS
def test_a_conversation_reads_its_own_file_by_name(tool):
    write("alice", "conv-alice", "summary.md", "MINE")

    assert as_caller("alice", "conv-alice", lambda: tool("summary.md")) == "MINE"


@TOOLS
def test_in_dev_mode_a_conversation_reads_its_own_file_by_name(tool):
    write(None, "conv-1", "summary.md", "MINE")

    assert as_caller(None, "conv-1", lambda: tool("summary.md")) == "MINE"


@TOOLS
def test_the_unowned_legacy_pool_is_still_read_by_name(tool):
    """A record written before sessions and owners existed belongs to nobody. find_files keeps
    offering it for reuse, and a read by name gives the same answer instead of a second policy."""
    write(None, None, "legacy_notes.md", "LEGACY")

    assert as_caller("alice", "conv-alice", lambda: tool("legacy_notes.md")) == "LEGACY"


def test_a_read_by_name_reports_the_file_it_resolved():
    """The scan found a path and no record, so a read by name came back with file_id null.
    Resolving through the store returns the record: the model learns which file it read, and
    gets its link."""
    rec = write("alice", "conv-alice", "summary.md", "MINE")

    out = as_caller("alice", "conv-alice", lambda: json.loads(read_text_file_tool("summary.md")))

    assert out["file_id"] == rec["file_id"]
    assert out["download_url"] == rec["download_url"]
    assert out["filename"] == "summary.md"


# --- which file a name means --------------------------------------------------------------

def test_a_name_is_matched_exactly():
    """find_files matches a substring, and the old scan matched an on-disk suffix after `__`,
    which a filename may itself contain. Neither is the file that was named."""
    write("alice", "conv-alice", "final__summary.md", "SUFFIX MATCH")    # the old scan's hit
    write("alice", "conv-alice", "old_summary.md", "SUBSTRING MATCH")    # find_files' hit

    with pytest.raises(ValueError):
        as_caller("alice", "conv-alice", lambda: read_text_file_tool("summary.md"))


def test_newer_near_misses_do_not_push_the_exact_name_off_the_page():
    """find_files sorts every substring match newest first and only then cuts its page (20 by
    default), so a run of newer `*summary.md` files would hide the one actually named."""
    exact = write("alice", "conv-alice", "summary.md", "EXACT")
    backdate(exact)
    for i in range(25):
        write("alice", "conv-alice", f"draft{i}_summary.md", "NEAR MISS")

    assert as_caller("alice", "conv-alice", lambda: _read("summary.md")) == "EXACT"


def test_of_several_files_with_the_name_the_newest_is_read():
    first = write("alice", "conv-alice", "summary.md", "FIRST")
    write("alice", "conv-alice", "summary.md", "SECOND")
    backdate(first)

    assert as_caller("alice", "conv-alice", lambda: _read("summary.md")) == "SECOND"


def test_a_newer_upload_outranks_an_older_output_of_the_same_name():
    """Why recency decides, and not "this conversation's own files first". An upload is never
    stamped with a conversation (the upload route binds the user, not the thread), so own-first
    would rank an earlier turn's output above the corrected file the user has just uploaded under
    the same name."""
    earlier = write("alice", "conv-alice", "data.csv", "a\n1\n")
    upload("alice", "data.csv", b"a\n2\n")
    backdate(earlier)

    assert as_caller("alice", "conv-alice", lambda: _read("data.csv")) == "a\n2\n"


# --- execute_code stages through the same lookup ------------------------------------------

def test_execute_code_does_not_stage_another_users_file_by_name():
    write("alice", "conv-alice", "summary.md", "ALICE PRIVATE")

    staging, _staged, errors, _skipped = as_caller(
        "bob", "conv-bob", lambda: exec_tools._build_staging(["summary.md"]))

    assert staging == []
    assert [e["ref"] for e in errors] == ["summary.md"]


def test_execute_code_stages_its_own_file_under_the_name_it_was_asked_for():
    """With no record, the scan's path was staged under its on-disk name, `file_<id>__summary.md`,
    so `open("summary.md")` in the sandbox found nothing. The record gives it both names."""
    rec = write("alice", "conv-alice", "summary.md", "MINE")

    _staging, staged, errors, _skipped = as_caller(
        "alice", "conv-alice", lambda: exec_tools._build_staging(["summary.md"]))

    assert errors == []
    assert staged[0]["available_as"] == [rec["file_id"], "summary.md"]


# --- the write tool's path branch ran the same scan ---------------------------------------

def test_write_text_file_does_not_reach_another_users_file_through_the_scan():
    """write_text_file sends a bare filename to create_output_file, but one starting with "."
    takes the path branch, which ran the scan. The scan matched an on-disk SUFFIX, so ".md"
    found alice's `notes__.md` (stored as `file_<id>__notes__.md`) and overwrote it."""
    rec = write("alice", "conv-alice", "notes__.md", "ALICE NOTES")

    as_caller("bob", "conv-bob", lambda: write_text_file_tool(".md", "BOB WAS HERE", overwrite=True))

    assert file_store.resolve_file_id(rec["file_id"]).read_text() == "ALICE NOTES"
