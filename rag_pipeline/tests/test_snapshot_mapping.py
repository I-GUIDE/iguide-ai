"""A conversation record must never be refused for the shape of what a tool sent.

`chat_memory` has no explicit mapping, and until M8.77 the client's record went in as an object,
so OpenSearch typed every field of it from the first conversation that carried it. The record
holds each answer's `agent_result`, whose tool arguments and results take whatever shape the tool
and the model chose that turn. On 2026-10-08 21:36 UTC `overpass_search` sent `bbox` as the string
"-87.93,41.87,..." where an earlier turn had sent a float array, the PUT came back 500 with
`mapper_parsing_exception`, and the turn never reached History.

The fakes the other tests use accept anything, which is why this passed CI. The store here
enforces types the way OpenSearch does, starting from the live `chat_memory` mapping (or from
none): the first value seen decides a field's type, and a document any field cannot hold is
refused whole. Its rules are `test_trace_mapping`'s, which were checked verdict for verdict
against OpenSearch 2.14.0.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import api.server as server  # noqa: E402
from rag_pipeline import memory_module as mm  # noqa: E402
from rag_pipeline.tests.chat_memory_mapping import chat_memory_mapping  # noqa: E402
from rag_pipeline.tests.test_memory_ownership import FakeOpenSearch, as_user  # noqa: E402
from rag_pipeline.tests.test_trace_mapping import MappingError, _check  # noqa: E402


class TypedFakeOpenSearch(FakeOpenSearch):
    """FakeOpenSearch whose fields keep the type they were first given, as a real index does."""

    def __init__(self, mapping=None):
        super().__init__()
        self.props = copy.deepcopy(mapping or {})
        self.born: set = set()  # every field a document introduced, across the whole test

    def check(self, body):
        trial, born = copy.deepcopy(self.props), set()
        for key, value in body.items():
            _check(value, key, trial, key, born)
        return trial, born

    def _accept(self, body):
        self.props, born = self.check(body)  # a refused document changes nothing
        self.born |= born

    def index(self, *, index, id, body, refresh=None):
        self._accept(body)
        super().index(index=index, id=id, body=json.loads(json.dumps(body)), refresh=refresh)

    def update(self, *, index, id, body, refresh=None):
        if id in self.docs:
            self._accept(body["doc"])
        super().update(index=index, id=id, body=json.loads(json.dumps(body)), refresh=refresh)


@pytest.fixture(params=["live mapping", "empty index"])
def store(request, monkeypatch):
    fake = TypedFakeOpenSearch(chat_memory_mapping() if request.param == "live mapping" else None)
    monkeypatch.setattr(mm, "_get_opensearch_client", lambda: fake)
    monkeypatch.delenv("AGENT_TOKEN_STRICT", raising=False)
    monkeypatch.delenv("AGENT_SESSION_SNAPSHOT_MAX_BYTES", raising=False)
    return fake


def a_turn(args, *, result=None):
    """A record whose one answer carries a tool call, in the shape the 21:36 turn had."""
    call = {"name": "overpass_search", "args": args}
    return {
        "id": "sess-07bc717f", "title": "Hospitals near the Loop", "threadId": "thread-1",
        "createdAt": 1791495399000, "updatedAt": 1791495405000,
        "messages": [
            {"role": "user", "text": "hospitals near the Loop"},
            {"role": "agent", "text": "Found 12.", "response": {
                "answer": "Found 12.", "message_id": "m-1",
                "agent_result": {"orchestration_result": {"analysis_results": {
                    "tool_calls": [call],
                    "tool_results": [result if result is not None else {"count": 12}]}}}}},
        ],
        "layers": [], "fileIds": [], "model": "deepseek-v4-flash", "provider": "lumen",
    }


AS_FLOATS = a_turn({"bbox": [-87.93, 41.87, -87.6, 42.0], "tags": {"amenity": "hospital"}})
AS_STRING = a_turn({"bbox": "-87.93,41.87,-87.6,42.0", "tags": "amenity=hospital"})


def _save(mid, record):
    return as_user("alice", lambda: mm.save_session_snapshot(mid, record))


def _read(mid):
    return as_user("alice", lambda: mm.get_session_snapshot(mid))


# --- the fake is strict enough to have caught it ---------------------------------------

def test_the_live_mapping_refuses_the_record_the_cluster_refused():
    with pytest.raises(MappingError, match=r"analysis_results\.tool_calls\.args\.bbox"):
        TypedFakeOpenSearch(chat_memory_mapping()).check({"session_snapshot": AS_STRING})


def test_stored_as_an_object_the_second_shape_is_refused():
    """The old writer against an empty index: the first record's types bind the second."""
    fake = TypedFakeOpenSearch()
    fake.index(index="chat_memory", id="a", body={"session_snapshot": AS_FLOATS})
    with pytest.raises(MappingError, match=r"tool_calls\.args\.bbox"):
        fake.index(index="chat_memory", id="b", body={"session_snapshot": AS_STRING})


# --- the fix --------------------------------------------------------------------------

def test_two_saves_whose_tool_args_differ_in_type_both_succeed(store):
    first = as_user("alice", lambda: mm.create_memory("one"))
    second = as_user("alice", lambda: mm.create_memory("two"))
    _save(first, AS_FLOATS)
    _save(second, AS_STRING)
    _save(first, AS_STRING)  # and the same conversation changing shape between turns
    assert _read(first)["messages"][1]["response"]["agent_result"] == \
        AS_STRING["messages"][1]["response"]["agent_result"]
    assert _read(second)["messages"][1]["response"]["agent_result"] == \
        AS_STRING["messages"][1]["response"]["agent_result"]


def test_the_lost_turn_is_stored_and_listed(store):
    mid = as_user("alice", lambda: mm.create_memory("x"))
    _save(mid, AS_STRING)
    listed = as_user("alice", lambda: mm.list_memories())
    assert [c["memoryId"] for c in listed] == [mid]
    assert listed[0]["messageCount"] == 2


def test_no_record_adds_a_field_to_the_mapping(store):
    """Every argument name used to become a field; past 1,000 every save would fail."""
    mid = as_user("alice", lambda: mm.create_memory("x"))
    _save(mid, AS_FLOATS)  # the document's own fields, which an empty index meets here
    store.born.clear()
    for args in ({"bbox": [1.5, 2.5]}, {"bbox": "1,2"}, {"limit": 6}, {"limit": "all"},
                 {"when": "2026-10-08"}, {"when": "yesterday"}, {"geom": {"type": "Point"}},
                 {"geom": "POINT (1 2)"}, {f"arg_{n}": n for n in range(50)}):
        _save(mid, a_turn(args, result={"features": [{"properties": args}]}))
    assert store.born == set(), sorted(store.born)


# --- reading back is unchanged ----------------------------------------------------------

AWKWARD = {
    "id": "sess-x", "title": "Ünïcode — 芝加哥", "threadId": "t", "createdAt": 1, "updatedAt": 2,
    "messages": [{"role": "agent", "text": "", "streaming": False, "response": {
        "agent_result": {"ints": [1, 2, 3], "floats": [1.0, 0.1, -87.93, 1e-12, 12345678901.5],
                         "big": 2 ** 53 + 1, "none": None, "empty": {}, "nested": [[[]]],
                         "bools": [True, False], "mixed": [1, "1", 1.0, None, {"a": [1]}],
                         "date_like": "2026-10-08", "quote": "he said \"hi\"\n\t\\"}}}],
    "layers": [{"kind": "geojson", "id": "l", "bounds": [-88.3, 40.0, -88.1, 40.2],
                "style": {"fill": "#ff0000"},
                "data": {"type": "FeatureCollection", "features": [
                    {"type": "Feature", "properties": {"name": 7, "pop": "n/a"},
                     "geometry": {"type": "GeometryCollection", "geometries": []}}]}}],
    "fileIds": ["file_abc"], "region": {"bbox": "-88.3,40.0,-88.1,40.2"}, "model": "m",
}


def test_a_round_trip_returns_the_identical_record(store):
    mid = as_user("alice", lambda: mm.create_memory("x"))
    _save(mid, AWKWARD)
    got = _read(mid)
    # The four keys the server has always set on the way out, from the document's own fields.
    server_owned = {"memoryId", "title", "createdAt", "updatedAt"}
    assert got["memoryId"] == mid and got["title"] == AWKWARD["title"]
    sent = {k: v for k, v in AWKWARD.items() if k not in server_owned}
    back = {k: v for k, v in got.items() if k not in server_owned}
    # Compared as JSON so 1 and 1.0, or True and 1, cannot pass for each other.
    assert json.dumps(back, sort_keys=True) == json.dumps(sent, sort_keys=True)


def test_the_endpoint_serves_the_same_shape_as_before(store, monkeypatch):
    """GET /agent/conversations/<id> for a record stored now and the same record stored as an
    object before M8.77: the client cannot tell them apart."""
    monkeypatch.delenv("AGENT_PERSISTENT_MEMORY", raising=False)
    monkeypatch.setattr(server, "assert_memory_owner", lambda memory_id: None)
    client = server.app.test_client()
    assert client.put("/agent/conversations/new", json=AWKWARD).status_code == 200
    stamp = store.docs["new"]
    # The same record as the old writer left it: an object under `session_snapshot`.
    store.docs["old"] = {k: v for k, v in stamp.items() if k != mm.SNAPSHOT_TEXT_FIELD}
    store.docs["old"]["session_snapshot"] = json.loads(json.dumps(AWKWARD))
    new, old = (client.get(f"/agent/conversations/{i}").get_json() for i in ("new", "old"))
    assert new.pop("memoryId") == "new" and old.pop("memoryId") == "old"
    assert json.dumps(new, sort_keys=True) == json.dumps(old, sort_keys=True)


# --- documents written before the change -----------------------------------------------

def _legacy(store, mid, record):
    """A conversation as the old writer left it, bypassing the type check: it is already in."""
    store.docs[mid] = {"conversationName": record["title"], "owner_id": "alice",
                       "chat_history": [], "createdAt": "2026-10-01T00:00:00+00:00",
                       "updatedAt": "2026-10-01T00:00:00+00:00",
                       "session_snapshot": json.loads(json.dumps(record))}
    store.visible.add(mid)


def test_an_old_document_still_loads_and_lists(store):
    _legacy(store, "old", AS_FLOATS)
    assert _read("old")["messages"] == AS_FLOATS["messages"]
    assert [c["memoryId"] for c in as_user("alice", lambda: mm.list_memories())] == ["old"]


def test_saving_an_old_document_keeps_one_copy(store):
    _legacy(store, "old", AS_FLOATS)
    _save("old", AS_STRING)
    assert store.docs["old"]["session_snapshot"] is None
    assert _read("old")["messages"] == AS_STRING["messages"]


def test_an_unreadable_text_record_falls_back_rather_than_failing(store):
    _legacy(store, "old", AS_FLOATS)
    store.docs["old"][mm.SNAPSHOT_TEXT_FIELD] = "{not json"
    assert _read("old")["messages"] == AS_FLOATS["messages"]
