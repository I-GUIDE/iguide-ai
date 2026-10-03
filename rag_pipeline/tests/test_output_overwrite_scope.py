"""An output's file_id names one write, and no other conversation's write lands on it.

`qgis_metric_buffer` registered its result with `create_output_file_from_path(...,
overwrite=True)`. With `overwrite`, the store scanned EVERY record for an output of the same
filename, reused that record's file_id, copied the new bytes over the old file and rewrote the
record with the CURRENT session and owner. The tool's default name is "buffer.geojson", so the
second person to buffer anything took over the first person's file. The first person's download
link then served the second person's buffer, and their record said it belonged to someone else:
hidden from their conversation by session scoping, and a 404 on their own link in token mode.

Observed 2026-10-01 inside ONE conversation: file_7e8178fd7165
("Champaign_city_2km_buffer.geojson") was written at 15:36:02 and silently rewritten at 15:38:36
by a re-grounding pass.

Two rules are pinned here:

* The QGIS tools never overwrite. A buffer or a rendered map is a result, and its file_id names
  that result. A re-run under the same name, in this conversation or any other, is a new file.
* `overwrite=True`, which the model can still pass to write_output_file and write_text_file,
  replaces only a file THIS conversation wrote for THIS caller. It never replaces another
  conversation's file or another user's, and replaces nothing when no conversation is bound.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import pytest

from agent_runtime import file_store, identity as idm
from agent_runtime.langchain_file_tools import write_output_file_tool, write_text_file_tool
from rag_pipeline import qgis_headless_tools
from rag_pipeline.qgis_headless_tools import pyqgis_render_map_tool, qgis_metric_buffer_tool

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "store"))
    monkeypatch.delenv("AGENT_PUBLIC_BASE_URL", raising=False)
    monkeypatch.delenv("QGIS_JOB_ROOT", raising=False)
    return tmp_path / "store"


@contextmanager
def caller(session, user=None):
    """Bind the conversation and the signed-in user (None: nobody) as the API edge does. Both
    are always bound, so nothing a previous test left in either ContextVar leaks in."""
    session_token = file_store.set_session(session)
    user_token = idm.set_user(idm.User(id=user, role=4) if user else None)
    try:
        yield
    finally:
        idm.reset_user(user_token)
        file_store.reset_session(session_token)


def _read(record):
    return file_store.resolve_file_id(record["file_id"]).read_text(encoding="utf-8")


def _stamp(record):
    """Whose the record says it is NOW, read back from the store."""
    now = file_store.get_file_record(record["file_id"])
    return now["session"], now["owner_id"]


def _buffer(monkeypatch, tag, **kwargs):
    """qgis_metric_buffer with a stand-in qgis_process. Every step writes a GeoJSON carrying
    `tag` to its OUTPUT, so the registered file says which call produced it."""
    body = json.dumps({"type": "FeatureCollection", "features": [], "tag": tag})

    def run(command, **_):
        for arg in command:
            if arg.startswith("OUTPUT="):
                Path(arg.removeprefix("OUTPUT=")).write_text(body, encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

    monkeypatch.setattr(qgis_headless_tools.subprocess, "run", run)
    out = json.loads(qgis_metric_buffer_tool(f"/data/{tag}.geojson", distance_meters=2000, **kwargs))
    assert out["ok"] is True, out
    return out["managed_output"]


def _render(monkeypatch, tag):
    """pyqgis_render_map with a stand-in worker that draws `tag` into the default map.png."""
    def run(command, **_):
        if len(command) < 3 or not str(command[1]).endswith("qgis_pyqgis_worker.py"):
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="")  # a probe
        spec = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        png = Path(spec["job_dir"]) / spec["output_filename"]
        png.write_bytes(b"\x89PNG\r\n\x1a\n" + tag.encode("utf-8"))
        Path(spec["result_path"]).write_text(json.dumps({"ok": True, "output_path": str(png)}),
                                             encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(qgis_headless_tools.subprocess, "run", run)
    out = json.loads(pyqgis_render_map_tool(
        json.dumps([{"path": "/data/points.geojson", "provider": "ogr"}])))
    assert out["ok"] is True, out
    return out["managed_output"]


# --- the defect, through the tool that had it -------------------------------------------

def test_another_conversations_buffer_does_not_take_over_mine(monkeypatch):
    with caller("conv-alice", "alice"):
        mine = _buffer(monkeypatch, "alice")          # the default name: buffer.geojson
    with caller("conv-bob", "bob"):
        theirs = _buffer(monkeypatch, "bob")

    assert mine["filename"] == theirs["filename"] == "buffer.geojson"
    assert theirs["file_id"] != mine["file_id"], "bob's buffer was written over alice's file"
    assert json.loads(_read(mine))["tag"] == "alice", "alice's link now serves bob's buffer"
    assert _stamp(mine) == ("conv-alice", "alice"), "alice's record was re-stamped as bob's"
    with caller("conv-alice", "alice"):
        # The download endpoint's own predicate: False is the 404 alice would get on her link.
        assert file_store.may_read(file_store.get_file_record(mine["file_id"]),
                                   allow_unowned=False)
        assert [r["file_id"] for r in file_store.find_files("buffer.geojson")] == [mine["file_id"]]


def test_a_rerun_in_the_same_conversation_is_a_new_file(monkeypatch):
    """The 2026-10-01 case: a re-ground re-ran the buffer under the name it had used before.
    A re-run can differ (a corrected distance, here), so the first link must keep its bytes."""
    name = "Champaign_city_2km_buffer.geojson"
    with caller("conv-alice", "alice"):
        first = _buffer(monkeypatch, "2000 m", output_filename=name)
        second = _buffer(monkeypatch, "2500 m", output_filename=name)

    assert second["file_id"] != first["file_id"]
    assert json.loads(_read(first))["tag"] == "2000 m"
    assert json.loads(_read(second))["tag"] == "2500 m"


def test_another_conversations_map_does_not_take_over_mine(monkeypatch):
    """pyqgis_render_map had the same `overwrite=True`, under the default name map.png."""
    with caller("conv-alice", "alice"):
        mine = _render(monkeypatch, "alice's map")
    with caller("conv-bob", "bob"):
        theirs = _render(monkeypatch, "bob's map")

    assert mine["filename"] == theirs["filename"] == "map.png"
    assert theirs["file_id"] != mine["file_id"]
    assert file_store.resolve_file_id(mine["file_id"]).read_bytes().endswith(b"alice's map")
    assert _stamp(mine) == ("conv-alice", "alice")


# --- overwrite=True, which the model can still ask for ----------------------------------

def _text(name, body, **kwargs):
    return file_store.create_output_file(name, body, **kwargs)


def _copied(name, body, **kwargs):
    src = file_store.storage_root().parent / "sources" / uuid4().hex / name
    src.parent.mkdir(parents=True)
    src.write_text(body, encoding="utf-8")
    return file_store.create_output_file_from_path(src, filename=name, **kwargs)


WRITERS = pytest.mark.parametrize("write", [_text, _copied], ids=["text", "from_path"])


@WRITERS
def test_overwrite_never_reaches_another_users_file(write):
    with caller("conv-alice", "alice"):
        mine = write("notes.md", "alice's notes")
    with caller("conv-bob", "bob"):
        theirs = write("notes.md", "bob's notes", overwrite=True)

    assert theirs["file_id"] != mine["file_id"]
    assert _read(mine) == "alice's notes"
    assert _stamp(mine) == ("conv-alice", "alice")


@WRITERS
def test_overwrite_never_reaches_another_conversation_of_the_same_user(write):
    """The earlier conversation's answer links to this file. Rewriting it from a later one would
    change what that link serves, and re-stamp it out of the conversation that made it."""
    with caller("conv-1", "alice"):
        mine = write("notes.md", "first conversation")
    with caller("conv-2", "alice"):
        later = write("notes.md", "second conversation", overwrite=True)

    assert later["file_id"] != mine["file_id"]
    assert _read(mine) == "first conversation"
    assert _stamp(mine) == ("conv-1", "alice")


@WRITERS
def test_a_borrowed_conversation_id_does_not_let_another_user_overwrite(write):
    """The conversation id arrives from the client. Matching on it alone would let anyone who
    learned it replace that conversation's files, so the owner has to match as well."""
    with caller("conv-alice", "alice"):
        mine = write("notes.md", "alice's notes")
    with caller("conv-alice", "bob"):
        theirs = write("notes.md", "bob's notes", overwrite=True)

    assert theirs["file_id"] != mine["file_id"]
    assert _read(mine) == "alice's notes"
    assert _stamp(mine) == ("conv-alice", "alice")


@WRITERS
def test_with_no_conversation_bound_overwrite_writes_a_new_file(write):
    """Without a conversation there is no 'this conversation's file' to replace, and the records
    that match are the unstamped legacy pool that every conversation reuses."""
    with caller(None):
        legacy = write("buffer.geojson", "legacy")
        again = write("buffer.geojson", "new", overwrite=True)

    assert again["file_id"] != legacy["file_id"]
    assert _read(legacy) == "legacy"


@WRITERS
@pytest.mark.parametrize("user", [None, "alice"], ids=["no-identity", "signed-in"])
def test_overwrite_still_replaces_this_conversations_own_file(write, user):
    """What overwrite is FOR: the model updating a file it saved earlier in this conversation,
    under the same link. Both with nobody identified (dev, demo) and signed in."""
    with caller("conv-alice", user):
        first = write("report.md", "draft")
        second = write("report.md", "final", overwrite=True)

    assert second["file_id"] == first["file_id"]
    assert _read(first) == "final"
    assert _stamp(first) == ("conv-alice", user)


def test_overwrite_replaces_the_newest_of_this_conversations_files_with_that_name():
    """The scan took the first match in directory order, which is no order at all. The newest is
    the one find_files and resolve_file_ref name first, and the link the model gave last."""
    with caller("conv-alice"):
        old = _text("report.md", "v1")
        new = _text("report.md", "v2")
        now = file_store.resolve_file_id(new["file_id"]).stat().st_mtime
        os.utime(file_store.resolve_file_id(old["file_id"]), (now - 100, now - 100))
        replaced = _text("report.md", "v3", overwrite=True)

    assert replaced["file_id"] == new["file_id"]
    assert _read(old) == "v1"
    assert _read(new) == "v3"


@pytest.mark.parametrize("tool", [write_output_file_tool, write_text_file_tool],
                         ids=["write_output_file", "write_text_file"])
def test_the_models_write_tools_cannot_overwrite_another_users_file(tool):
    """Both model-facing write tools pass the model's `overwrite` straight to the store."""
    with caller("conv-alice", "alice"):
        mine = json.loads(tool("summary.md", "alice's summary"))
    with caller("conv-bob", "bob"):
        theirs = json.loads(tool("summary.md", "bob's summary", overwrite=True))

    assert theirs["file_id"] != mine["file_id"]
    assert _read(mine) == "alice's summary"


# --- the class -----------------------------------------------------------------------------

def test_no_producer_hard_codes_overwrite():
    """A tool's result is a new file. `overwrite=True` written into a producer turns its output
    name into a slot that every later run of that tool rewrites; the two QGIS tools had it,
    under the default names buffer.geojson and map.png. A pass-through of the caller's own
    `overwrite=...` is fine, and is what the model-facing write tools do."""
    offenders = []
    for package in ("agent_runtime", "rag_pipeline", "MCP_server", "extractors", "api"):
        for path in sorted((REPO / package).rglob("*.py")):
            if "tests" in path.relative_to(REPO).parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if name not in {"create_output_file", "create_output_file_from_path"}:
                    continue
                for kw in node.keywords:
                    if (kw.arg == "overwrite" and isinstance(kw.value, ast.Constant)
                            and kw.value.value is True):
                        offenders.append(f"{path.relative_to(REPO)}:{node.lineno}")
    assert offenders == []
