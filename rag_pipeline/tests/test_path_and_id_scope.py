"""A stored file named by its file_id or by a path into the store reached ANY user's file.

Stage 21 scoped a bare filename. The two other ways a tool names a stored file were left open:

* a file_id went straight to the record. Outside the store, `may_read` was checked only by the
  download endpoint, so every tool that takes an id (the file tools, `execute_code`'s
  input_files, the geo, QGIS, terrain and rs-embed tools) read another user's file;
* a path. The storage root is one of the file tools' allowed roots, so a relative or absolute
  path into `uploads/`, `outputs/` or `metadata/` was read, and written, with no check at all.

Measured on prototype 447b961, 2026-10-03: bob read alice's `summary.md` by its id, by
`outputs/<id>__summary.md`, and by its absolute path; staged it into his sandbox both ways; read
her record in `metadata/`; and overwrote her file through `write_text_file`.

The policy these tests pin: an id and a path get the same owner check, in any of the owner's
conversations, which is what the download link already does. The unowned legacy pool stays
readable, as `find_files` offers it. `metadata/` is never reachable by path, and a raw-path
write may replace only an output this conversation wrote, the rule `overwrite=True` follows.
"""

from __future__ import annotations

import io
import json
import shutil
import time
from pathlib import Path

import jwt
import pytest
from werkzeug.datastructures import FileStorage

from agent_runtime import file_store, identity
from agent_runtime import langchain_exec_tools as exec_tools
from agent_runtime import langchain_file_tools as file_tools
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


def upload(user, name, body, session=None):
    """As `/agent/files/upload` stores a file: the user is bound, and the conversation when the
    client sends one (Stage 23)."""
    return as_caller(user, session, lambda: file_store.save_uploaded_file(
        FileStorage(stream=io.BytesIO(body), filename=name)))


def stored(record):
    """The bytes on disk, read without going through any check under test."""
    return (file_store.storage_root() / record["relative_path"]).read_text()


# The three ways a model names a stored file it was shown. write_output_file reports the
# relative path, read_text_file the absolute one, and every tool the id.
def by_id(record):
    return record["file_id"]


def by_relative_path(record):
    return record["relative_path"]


def by_absolute_path(record):
    return str((file_store.storage_root() / record["relative_path"]).resolve())


REFS = pytest.mark.parametrize("name_it", [by_id, by_relative_path, by_absolute_path],
                               ids=["file_id", "relative_path", "absolute_path"])
PATHS = pytest.mark.parametrize("name_it", [by_relative_path, by_absolute_path],
                                ids=["relative_path", "absolute_path"])


def _read(ref):
    return json.loads(read_text_file_tool(ref))["content"]


def _inspect(ref):
    content = json.loads(inspect_file_for_analysis_tool(ref))["analysis_ready_content"]
    return content.get("content", content)   # a CSV comes back as a header and rows instead


TOOLS = pytest.mark.parametrize("tool", [_read, _inspect],
                                ids=["read_text_file", "inspect_file_for_analysis"])


# --- another user's file, by id or by path ------------------------------------------------

@TOOLS
@REFS
def test_another_users_file_is_not_read(tool, name_it):
    rec = write("alice", "conv-alice", "summary.md", "ALICE PRIVATE")

    with pytest.raises(ValueError):
        as_caller("bob", "conv-bob", lambda: tool(name_it(rec)))


@TOOLS
@REFS
def test_another_users_upload_is_not_read(tool, name_it):
    rec = upload("alice", "results.csv", b"ALICE PRIVATE")

    with pytest.raises(ValueError):
        as_caller("bob", "conv-bob", lambda: tool(name_it(rec)))


@REFS
def test_a_borrowed_conversation_id_does_not_reach_another_users_file(name_it):
    """The conversation id comes from the client, so ownership alone decides."""
    rec = write("alice", "conv-alice", "summary.md", "ALICE PRIVATE")

    with pytest.raises(ValueError):
        as_caller("bob", "conv-alice", lambda: _read(name_it(rec)))


@REFS
def test_a_refusal_does_not_confirm_the_file_exists(name_it):
    """The download endpoint answers 404 rather than 403 for this reason. A refusal that differed
    from a missing file's error would tell bob which of alice's ids and paths are real."""
    rec = write("alice", "conv-alice", "summary.md", "ALICE PRIVATE")
    never_written = {"file_id": "file_0123456789ab",
                     "relative_path": "outputs/file_0123456789ab__summary.md"}

    def refusal(ref):
        with pytest.raises(ValueError) as exc:
            as_caller("bob", "conv-bob", lambda: read_text_file_tool(ref))
        return str(exc.value).replace(ref, "<ref>")

    assert refusal(name_it(rec)) == refusal(name_it(never_written))


# --- what the owner keeps -----------------------------------------------------------------

@TOOLS
@REFS
def test_the_owner_reads_their_own_file(tool, name_it):
    rec = write("alice", "conv-alice", "summary.md", "MINE")

    assert as_caller("alice", "conv-alice", lambda: tool(name_it(rec))) == "MINE"


@REFS
def test_the_owner_reads_it_from_another_of_their_conversations(name_it):
    """An id or a path names exactly one file, so the conversation scope, which exists to tell
    files with the same NAME apart, has nothing to decide. The download link already works from
    every one of the owner's conversations, and the tools now give the same answer."""
    rec = write("alice", "conv-alice-1", "summary.md", "MINE")

    assert as_caller("alice", "conv-alice-2", lambda: _read(name_it(rec))) == "MINE"


@REFS
def test_a_read_reports_the_record_it_resolved(name_it):
    """A path into the store came back with file_id and download_url null. It now resolves to the
    record, the way an id does and a bare filename has since Stage 21."""
    rec = write("alice", "conv-alice", "summary.md", "MINE")

    out = as_caller("alice", "conv-alice", lambda: json.loads(read_text_file_tool(name_it(rec))))

    assert out["file_id"] == rec["file_id"]
    assert out["download_url"] == rec["download_url"]
    assert out["filename"] == "summary.md"


@REFS
def test_the_unowned_legacy_pool_is_still_read(name_it, monkeypatch):
    """Written before owners and conversations existed. find_files offers it to every caller for
    reuse, so an id and a path do too, in strict token mode as well. Only the browser download
    refuses an unowned file once strict (S9.3), and that endpoint keeps its own rule."""
    monkeypatch.setenv("AGENT_TOKEN_STRICT", "1")
    rec = write(None, None, "legacy_notes.md", "LEGACY")

    assert as_caller("alice", "conv-alice", lambda: _read(name_it(rec))) == "LEGACY"


@REFS
def test_in_dev_mode_an_id_or_a_path_reaches_any_conversations_file(name_it):
    """Nobody is identified in dev or demo mode, and an id is not scoped by conversation, so an id
    or a path reaches the file from any conversation. The download link does the same there."""
    rec = write(None, "conv-1", "summary.md", "FROM CONVERSATION 1")

    assert as_caller(None, "conv-2", lambda: _read(name_it(rec))) == "FROM CONVERSATION 1"


# --- the store's own directories ----------------------------------------------------------

def test_a_record_is_not_read_by_path():
    """metadata/ holds the records themselves, which say whose each file is and which
    conversation made it. A tool has no use for them, the owner included."""
    rec = write("alice", "conv-alice", "summary.md", "MINE")

    for ref in (f"metadata/{rec['file_id']}.json",
                str(file_store.storage_root() / "metadata" / f"{rec['file_id']}.json")):
        with pytest.raises(ValueError):
            as_caller("alice", "conv-alice", lambda: _read(ref))


def test_a_file_no_record_names_is_not_read_by_path(store):
    """No record means no owner to check, so it is refused, as its id would be."""
    (store / "outputs").mkdir(exist_ok=True)
    stray = store / "outputs" / "file_0123456789ab__stray.md"
    stray.write_text("STRAY")

    with pytest.raises(ValueError):
        as_caller("alice", "conv-alice", lambda: _read(str(stray)))


def test_a_path_is_its_records_file_or_nothing(store):
    """The on-disk name names a record, and that record has to name the same file back. A file
    beside it that carries the same id prefix is not that record's file."""
    rec = write("bob", "conv-bob", "notes.md", "BOB'S NOTES")
    beside = store / "outputs" / f"{rec['file_id']}__other.md"
    beside.write_text("NOT THE RECORD'S FILE")

    with pytest.raises(ValueError):
        as_caller("bob", "conv-bob", lambda: _read(str(beside)))


def test_the_store_is_checked_however_a_path_reaches_it(tmp_path, monkeypatch):
    """The default store, agent_chat_files/, sits inside the repo, which is itself an allowed
    root, and .env.example points UPLOAD_FOLDER at the store's uploads/. Either admits a path
    into the store, and neither may skip the store's checks."""
    repo = tmp_path / "repo"
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(repo / "agent_chat_files"))
    monkeypatch.setenv("UPLOAD_FOLDER", str(repo / "agent_chat_files" / "uploads"))
    monkeypatch.setattr(file_tools, "_repo_root", lambda: repo)
    rec = write("alice", "conv-alice", "summary.md", "ALICE PRIVATE")
    up = upload("alice", "results.csv", b"ALICE PRIVATE")

    for ref in (f"agent_chat_files/{rec['relative_path']}", by_absolute_path(up),
                f"agent_chat_files/{up['relative_path']}"):
        with pytest.raises(ValueError):
            as_caller("bob", "conv-bob", lambda: _read(ref))
    assert as_caller("alice", "conv-alice",
                     lambda: _read(f"agent_chat_files/{rec['relative_path']}")) == "ALICE PRIVATE"


def test_a_differently_cased_path_is_still_the_store(store):
    """Path.resolve() does not canonicalise case, so on a case-insensitive filesystem (macOS, a
    Windows bind mount) OUTPUTS/ is the store's outputs/ while being a different string."""
    (store / "case_probe").write_text("x")
    if not (store / "CASE_PROBE").exists():
        pytest.skip("case-sensitive filesystem")
    rec = write("alice", "conv-alice", "summary.md", "ALICE PRIVATE")

    with pytest.raises(ValueError):
        as_caller("bob", "conv-bob",
                  lambda: _read(str(store / rec["relative_path"].replace("outputs", "OUTPUTS", 1))))


# --- execute_code stages through the same lookup ------------------------------------------

@REFS
def test_execute_code_does_not_stage_another_users_file(name_it):
    rec = write("alice", "conv-alice", "summary.md", "ALICE PRIVATE")

    staging, _staged, errors, _skipped = as_caller(
        "bob", "conv-bob", lambda: exec_tools._build_staging([name_it(rec)]))

    assert staging == []
    assert [e["ref"] for e in errors] == [name_it(rec)]


@REFS
def test_execute_code_stages_its_own_file_under_its_filename(name_it):
    """Staged by path it went in under its on-disk name, `file_<id>__summary.md`, so
    `open("summary.md")` in the sandbox found nothing. The record gives it both names."""
    rec = write("alice", "conv-alice", "summary.md", "MINE")

    _staging, staged, errors, _skipped = as_caller(
        "alice", "conv-alice", lambda: exec_tools._build_staging([name_it(rec)]))

    assert errors == []
    assert staged[0]["available_as"] == [rec["file_id"], "summary.md"]


# --- writes by path -----------------------------------------------------------------------

@PATHS
def test_another_users_file_is_not_overwritten(name_it):
    rec = write("alice", "conv-alice", "summary.md", "ALICE PRIVATE")

    with pytest.raises(ValueError):
        as_caller("bob", "conv-bob",
                  lambda: write_text_file_tool(name_it(rec), "BOB WAS HERE", overwrite=True))
    assert stored(rec) == "ALICE PRIVATE"


def test_a_record_is_not_written_by_path(store):
    """A record says whose its file is. Writing one by path would rewrite the ownership the
    checks read, so no caller writes there, the owner included."""
    rec = write("alice", "conv-alice", "summary.md", "MINE")
    record_path = store / "metadata" / f"{rec['file_id']}.json"
    before = record_path.read_text()

    for user, session in (("bob", "conv-bob"), ("alice", "conv-alice")):
        for ref in (f"metadata/{rec['file_id']}.json", str(record_path)):
            with pytest.raises(ValueError):
                as_caller(user, session, lambda: write_text_file_tool(ref, "{}", overwrite=True))
    assert record_path.read_text() == before


def test_a_new_file_is_not_created_in_the_store_by_path(store):
    """A data file no record names has no link, is never listed, and could not be read back by
    path. write_output_file is how a file enters the store."""
    with pytest.raises(ValueError):
        as_caller("alice", "conv-alice", lambda: write_text_file_tool("outputs/report.md", "x"))
    assert not (store / "outputs" / "report.md").exists()


@PATHS
def test_a_conversation_overwrites_its_own_output_by_path(name_it):
    rec = write("alice", "conv-alice", "summary.md", "FIRST")

    out = json.loads(as_caller("alice", "conv-alice", lambda: write_text_file_tool(
        name_it(rec), "SECOND", overwrite=True)))

    assert stored(rec) == "SECOND"
    assert out["file_id"] == rec["file_id"]


def test_the_same_users_other_conversation_does_not_overwrite_by_path():
    """The rule overwrite=True follows (Stage 22): a file is replaced only in the conversation
    that made it, because an answer there links to it and shows what it held."""
    rec = write("alice", "conv-alice-1", "summary.md", "FIRST")

    with pytest.raises(ValueError):
        as_caller("alice", "conv-alice-2", lambda: write_text_file_tool(
            rec["relative_path"], "SECOND", overwrite=True))
    assert stored(rec) == "FIRST"


def test_in_dev_mode_another_conversation_does_not_overwrite_by_path():
    rec = write(None, "conv-1", "summary.md", "FIRST")

    with pytest.raises(ValueError):
        as_caller(None, "conv-2", lambda: write_text_file_tool(
            rec["relative_path"], "SECOND", overwrite=True))
    assert stored(rec) == "FIRST"


def test_an_upload_is_not_overwritten_by_path():
    """An upload is the user's original. A corrected copy is a new output."""
    rec = upload("alice", "data.csv", b"a\n1\n", session="conv-alice")

    with pytest.raises(ValueError):
        as_caller("alice", "conv-alice", lambda: write_text_file_tool(
            rec["relative_path"], "a\n2\n", overwrite=True))
    assert stored(rec) == "a\n1\n"


def test_the_legacy_pool_is_read_but_not_overwritten_by_path():
    """Every conversation reads the pool, so a write there would change what all of them read."""
    rec = write(None, None, "legacy_notes.md", "LEGACY")

    with pytest.raises(ValueError):
        as_caller("alice", "conv-alice", lambda: write_text_file_tool(
            rec["relative_path"], "MINE NOW", overwrite=True))
    assert stored(rec) == "LEGACY"


# --- the other tools take ids through the same lookup -------------------------------------

def test_resolve_file_ref_does_not_resolve_another_users_id():
    """predict_from_package and the rest of rs-embed name a package by id or by filename."""
    rec = write("alice", "conv-alice", "vectors.npz", "ALICE PRIVATE")

    with pytest.raises(ValueError):
        as_caller("bob", "conv-bob", lambda: file_store.resolve_file_ref(rec["file_id"]))
    path, record, _also = as_caller("alice", "conv-alice",
                                    lambda: file_store.resolve_file_ref(rec["file_id"]))
    assert record["file_id"] == rec["file_id"]


def test_the_geo_tools_do_not_open_another_users_file():
    from agent_runtime import langchain_geo_tools as geo

    rec = write("alice", "conv-alice", "zones.geojson", '{"type": "FeatureCollection", "features": []}')

    for name_it in (by_id, by_absolute_path):
        with pytest.raises(ValueError):
            as_caller("bob", "conv-bob", lambda: geo._resolve(name_it(rec)))
    _path, record = as_caller("alice", "conv-alice", lambda: geo._resolve(by_absolute_path(rec)))
    assert record["file_id"] == rec["file_id"]


def test_qgis_does_not_resolve_another_users_id():
    """An id QGIS cannot resolve is passed on as the string it was, as a typo would be."""
    from rag_pipeline import qgis_headless_tools as qgis

    rec = write("alice", "conv-alice", "zones.geojson", "{}")

    assert as_caller("bob", "conv-bob", lambda: qgis._resolve_layer_ref(rec["file_id"])) \
        == rec["file_id"]
    assert as_caller("alice", "conv-alice", lambda: qgis._resolve_layer_ref(rec["file_id"])) \
        == by_absolute_path(rec)


def _assembled_parts(user, session, shp):
    from rag_pipeline import qgis_headless_tools as qgis

    staged = Path(as_caller(user, session, lambda: qgis._resolve_layer_ref(shp["file_id"])))
    assert staged.parent.name.startswith("qgis_shp_"), staged   # assembled, not the store itself
    try:
        return sorted((p.name, p.read_bytes()) for p in staged.parent.iterdir())
    finally:
        shutil.rmtree(staged.parent, ignore_errors=True)


def test_qgis_assembles_a_shapefile_only_from_parts_this_caller_may_read():
    """A shapefile's parts are found by their shared stem, because the store names each upload
    `<file_id>__<name>` and they do not sit side by side. That scan read every upload in the
    store, so bob's parcels.shp was opened with alice's parcels.dbf as its attributes."""
    shp = upload("bob", "parcels.shp", b"BOB SHP", session="conv-bob")
    upload("bob", "parcels.shx", b"BOB SHX", session="conv-bob")
    upload("alice", "parcels.dbf", b"ALICE DBF", session="conv-alice")

    assert _assembled_parts("bob", "conv-bob", shp) == [
        ("parcels.shp", b"BOB SHP"), ("parcels.shx", b"BOB SHX")]


def test_qgis_assembles_a_shapefile_from_its_own_conversations_parts():
    """Its parts are found by NAME, so they follow the name rule of Stage 21: the parts uploaded
    in one conversation make one shapefile, and another conversation's parcels.dbf is not one of
    them, in dev mode too."""
    shp = upload(None, "parcels.shp", b"SHP 2", session="conv-2")
    upload(None, "parcels.shx", b"SHX 2", session="conv-2")
    upload(None, "parcels.dbf", b"DBF 1", session="conv-1")

    assert _assembled_parts(None, "conv-2", shp) == [("parcels.shp", b"SHP 2"),
                                                     ("parcels.shx", b"SHX 2")]


def test_the_prompt_does_not_name_another_users_file():
    """The chat request carries file_ids from the client, and their filenames go into the prompt."""
    from agent_runtime.agent_chat_service import _augment_user_input_with_file_ids

    rec = upload("alice", "salaries_2026.csv", b"ALICE PRIVATE")

    text = as_caller("bob", "conv-bob",
                     lambda: _augment_user_input_with_file_ids("hello", [rec["file_id"]]))

    assert "salaries_2026.csv" not in text


def test_only_a_single_token_is_looked_up_as_an_id(store):
    """An id names its record's file, metadata/<id>.json. Every id the store has minted is
    `file_` and twelve hex digits, so anything with a separator or a dot is not one, whatever
    sits at the place it would name."""
    rec = write("alice", "conv-alice", "summary.md", "MINE")
    (store / "outputs" / "record_like.json").write_text(json.dumps(
        {"file_id": "record_like", "filename": "summary.md", "relative_path": rec["relative_path"]}))

    for ref in ("../outputs/record_like", "..\\outputs\\record_like", f"{rec['file_id']}.json",
                f"./{rec['file_id']}", f"metadata/{rec['file_id']}"):
        assert file_store.get_file_record(ref) is None, ref
    assert file_store.get_file_record(rec["file_id"])["file_id"] == rec["file_id"]


# --- the download endpoint, the other reader of ids ---------------------------------------

SECRET = "s" * 64
COOKIE = "jwt-access-token-dev"


def test_a_refused_download_reads_exactly_as_an_unknown_id(monkeypatch):
    """S9.3 made a refusal a 404, not a 403, so that it would not confirm the id. Its body said
    "No file found for id" where an id that was never minted gets "unknown file_id", which told
    the caller the same thing in other words."""
    import api.server as server

    monkeypatch.setenv("AGENT_MODE", "token")
    monkeypatch.setenv("JWT_ACCESS_TOKEN_SECRET", SECRET)
    monkeypatch.setenv("JWT_ACCESS_TOKEN_NAME", COOKIE)
    monkeypatch.delenv("AGENT_TOKEN_STRICT", raising=False)
    monkeypatch.delenv("AGENT_CHAT_API_KEY", raising=False)
    monkeypatch.delenv("DEMO_MODE", raising=False)
    alices = write("alice", "conv-alice", "mine.txt", "ALICE PRIVATE")
    nobodys = write(None, None, "legacy.txt", "LEGACY")          # refused once strict
    bob = jwt.encode({"id": "bob", "role": 4, "exp": int(time.time()) + 3600},
                     SECRET, algorithm="HS256")

    def body(file_id):
        with server.app.test_client() as client:
            client.set_cookie(COOKIE, bob, domain="localhost")
            res = client.get(f"/agent/files/{file_id}/download")
        return res.status_code, res.get_data(as_text=True).replace(file_id, "<id>")

    assert body(alices["file_id"]) == body("file_0123456789ab")
    assert body(nobodys["file_id"]) == body("file_0123456789ab")
