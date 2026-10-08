"""A trace must fit the `chat_traces` mapping, or OpenSearch refuses the whole turn.

The fake store in `test_turn_traces` accepts anything, which is why this was never caught: the
real index typed `events.data.args` as `text` from early documents, and a tool called with no
arguments sends `{}`. 30 live turns were refused with `mapper_parsing_exception` between 10-01
and 10-08, and `save_turn_trace` could only log a warning.

The store here enforces field types the way OpenSearch does: the snapshot mapping for known
fields, dynamic typing from first sight for new ones, and a 400 for any value its field cannot
hold. Its rules are written out independently of `trace_mapping.leaf_accepts`, so the writer is
checked against a second reading of the rules rather than against itself.
"""
from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime import identity as idm  # noqa: E402
from agent_runtime.streaming_trace import (StreamingTraceCallbackHandler,  # noqa: E402
                                           emit_trace_event, trace_context)
from rag_pipeline import memory_module as mm  # noqa: E402
from rag_pipeline.trace_mapping import TRACE_MAPPING, fit_trace_document  # noqa: E402


class MappingError(Exception):
    """Stands in for opensearchpy's RequestError(400, 'mapper_parsing_exception', ...)."""


_DETECTED_DATE = re.compile(r"-?\d{4}(-\d{2}(-\d{2}(T[\d:.]+(Z|[+-]\d{2}:?\d{2})?)?)?|/\d{2}/\d{2})")


def _dynamic_type(value):
    # Measured on OpenSearch 2.14.0: these shapes are date-detected on first sight.
    if isinstance(value, str) and _DETECTED_DATE.fullmatch(value):
        return "date"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "long"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "text"
    raise AssertionError(f"unexpected scalar {value!r}")


def _check(value, path, props, key, born):
    """Raise MappingError where OpenSearch would. Grows ``props`` like dynamic mapping does.

    ``born`` holds the fields this document is introducing. A field that already exists takes
    any value its type can hold, but two dynamic types for a NEW field in one document conflict
    (`cannot be changed from type [text] to [long]`), which is stricter than across documents.
    """
    items = value if isinstance(value, list) else [value]
    for item in items:
        if isinstance(item, list):
            _check(item, path, props, key, born)
            continue
        if item is None:
            continue
        spec = props.get(key)
        if spec is None:  # first sight: the value decides the type, for every later document
            spec = {"properties": {}} if isinstance(item, dict) else _dynamic_type(item)
            props[key] = spec
            born.add(path)
        elif path in born and not isinstance(item, dict) and not isinstance(spec, dict) \
                and _dynamic_type(item) != spec:
            raise MappingError(f"mapper [{path}] cannot be changed from type [{spec}] "
                               f"to [{_dynamic_type(item)}]")
        if isinstance(spec, dict):
            if not isinstance(item, dict):
                raise MappingError(f"object field [{path}] given {item!r}")
            for k, v in item.items():
                _check(v, f"{path}.{k}", spec["properties"], k, born)
        elif isinstance(item, dict):
            raise MappingError(f"failed to parse field [{path}] of type [{spec}]: {item!r}")
        elif spec in ("long", "float"):
            if isinstance(item, bool):
                raise MappingError(f"[{path}] of type [{spec}] given a boolean")
            try:
                float(item)
            except (TypeError, ValueError):
                raise MappingError(f"[{path}] of type [{spec}] given {item!r}") from None
        elif spec == "date" and not (isinstance(item, (int, float)) or _dynamic_type(item) == "date"):
            raise MappingError(f"failed to parse field [{path}] of type [date]: {item!r}")
        elif spec == "boolean" and not (isinstance(item, bool) or item in ("true", "false")):
            raise MappingError(f"[{path}] of type [boolean] given {item!r}")


class MappedIndex:
    """One `chat_traces`, starting from the live mapping snapshot, kept for the whole test."""

    def __init__(self):
        self.props = copy.deepcopy(TRACE_MAPPING)
        self.docs: dict[str, dict] = {}

    def check(self, body):
        trial, born = copy.deepcopy(self.props), set()
        for key, value in body.items():
            _check(value, key, trial, key, born)
        return trial

    def index(self, *, index, id, body, **_kw):
        self.props = self.check(body)  # a refused document changes nothing
        self.docs[id] = json.loads(json.dumps(body))

    def get(self, *, index, id):
        return {"_source": copy.deepcopy(self.docs[id])}


@pytest.fixture
def live(monkeypatch):
    store = MappedIndex()
    monkeypatch.setattr(mm, "_get_opensearch_client", lambda: store)
    monkeypatch.delenv("AGENT_TRACE_MAX_BYTES", raising=False)
    return store


def _save(events, memory_id="mem-1"):
    token = idm.set_user(idm.User(id="alice", role=4))
    try:
        return mm.save_turn_trace(memory_id, thread_id="t", query="q", events=events,
                                  answer="a", model="m", provider="p")
    finally:
        idm.reset_user(token)


def _record_tool_calls(calls):
    """Run tool starts through the REAL callback handler, the way a LangChain turn does."""
    recorded = []
    handler = StreamingTraceCallbackHandler()
    with trace_context(lambda _e: None, agent_dev=False, recorder=recorded.append):
        for name, input_str in calls:
            handler.on_tool_start({"name": name}, input_str, run_id=f"run-{name}")
    return recorded


# --- the turn that was lost ---------------------------------------------------------

def test_a_tool_called_with_no_arguments_is_stored(live):
    """The 10-08 19:55 turn: `list_conversation_files` and `list_available_skills`, no args.

    LangChain passes `str(tool_input)`. With arguments that is a Python repr, not JSON, so `args`
    stays a string; with none it is "{}", which parses to a dict and hits the `text` field.
    """
    events = _record_tool_calls([("list_conversation_files", "{}"),
                                 ("list_available_skills", "{}"),
                                 ("search_knowledge", "{'query': 'flood', 'limit': 6}")])
    assert events[0]["data"]["args"] == {}, "the shape that reached the index"

    with pytest.raises(MappingError, match=r"events\.data\.args"):
        MappedIndex().check({"events": events})  # unfitted: refused, as on the cluster

    res = _save(events)
    assert res["stored"] is True, res
    doc = live.docs[res["traceId"]]
    assert [e["data"]["args"] for e in doc["events"]] == [
        "{}", "{}", "{'query': 'flood', 'limit': 6}"]
    assert doc["events"][0]["data"]["tool_calls"] == [
        {"name": "list_conversation_files", "args": "{}"}]


def test_arguments_with_keys_are_stored_as_json_text(live):
    """The other refusals: `{query=..., limit=6}` from emitters that send real dicts."""
    args = {"query": "¿Qué conjuntos de datos hay sobre sequías en México?", "limit": 6,
            "bbox": [-118.4, 14.5, -86.7, 32.7], "nested": {"crs": "EPSG:4326", "keep": None}}
    with trace_context(lambda _e: None, agent_dev=False, recorder=(events := []).append):
        emit_trace_event("tool_call", {"name": "search_knowledge", "args": args,
                                       "tool_calls": [{"name": "search_knowledge", "args": args}]})
    res = _save(events)
    assert res["stored"] is True, res
    data = live.docs[res["traceId"]]["events"][0]["data"]
    assert json.loads(data["args"]) == args, "the full arguments survive, losslessly"
    assert json.loads(data["tool_calls"][0]["args"]) == args


# --- every other field that could disagree with its mapping -------------------------

@pytest.mark.parametrize("field,value,stored_as", [
    ("content", [{"type": "text", "text": "hi"}], "content"),      # text <- content blocks
    ("tools", [{"area": "terrain", "tools": 3}], "tools"),          # text <- inventory rows
    ("message", {"error": "boom"}, "message"),                      # text <- an error object
    ("bounds", {"west": -88.7, "east": -87.9}, "bounds_text"),      # float <- a named bbox
    ("count", "all", "count_text"),                                 # long <- a word
    ("duration_s", True, "duration_s_text"),                        # float <- a boolean
    ("flagged", "maybe", "flagged_text"),                           # boolean <- a word
    ("issues", ["unsupported claim"], "issues_text"),               # object <- strings
    ("embedding", "rs-embed:v2", "embedding_text"),                 # object <- a string
])
def test_a_value_its_field_cannot_hold_is_kept_beside_it(live, field, value, stored_as):
    res = _save([{"event": "x", "data": {field: value, "name": "t"}}])
    assert res["stored"] is True, res
    data = live.docs[res["traceId"]]["events"][0]["data"]
    assert json.loads(data[stored_as]) == value, "kept, as JSON text"
    assert data["name"] == "t", "and the rest of the event is untouched"


def test_a_mismatch_inside_an_object_field_moves_only_that_leaf(live):
    legend = [{"label": "high", "color": [215, 48, 39, 255]}, {"label": "low", "color": "#4575b4"}]
    res = _save([{"event": "map_layer", "data": {"legend": legend}}])
    assert res["stored"] is True, res
    kept = live.docs[res["traceId"]]["events"][0]["data"]["legend"]
    assert kept[0] == {"label": "high", "color": [215, 48, 39, 255]}
    assert kept[1] == {"label": "low", "color_text": '"#4575b4"'}


def test_values_that_already_fit_are_not_touched(live):
    data = {"name": "dem_for_region", "duration_s": 0.4, "count": 3, "flagged": False,
            "bounds": [-88.7, 39.8, -87.9, 40.4], "tools": ["a", "b"],
            "issues": [{"claim": "c", "reason": "r"}], "outcome": "1 layer", "args": "{}"}
    res = _save([{"event": "tool_result", "data": dict(data)}])
    assert live.docs[res["traceId"]]["events"][0]["data"] == data


def test_a_new_structure_cannot_lock_the_shape_for_later_turns(live):
    """Unmapped and structured -> text. Otherwise the first turn's shape types the field."""
    first = _save([{"event": "x", "data": {"plan": {"steps": 3}}}], "mem-1")
    second = _save([{"event": "x", "data": {"plan": "just answer"}}], "mem-2")
    third = _save([{"event": "x", "data": {"plan": [{"step": "a"}, {"step": 2}]}}], "mem-3")
    with pytest.raises(MappingError, match="cannot be changed"):  # as OpenSearch 2.14.0 does
        MappedIndex().check({"events": [{"data": {"plan": [{"step": "a"}, {"step": 2}]}}]})
    assert all(r["stored"] for r in (first, second, third)), (first, second, third)
    assert live.props["events"]["properties"]["data"]["properties"]["plan"] == "text"


def test_a_new_scalar_field_cannot_lock_its_type_either(live):
    """A number is no safer than a dict: seen first as `3`, a field maps `long` and refuses
    `"all"` from the next turn; seen as text and a number in one document, it is refused at once.
    """
    first = _save([{"event": "x", "data": {"limit": 6}}], "mem-1")
    second = _save([{"event": "x", "data": {"limit": "all"}}], "mem-2")
    mixed = _save([{"event": "x", "data": {"step": "a"}}, {"event": "x", "data": {"step": 2}}],
                  "mem-3")
    assert all(r["stored"] for r in (first, second, mixed)), (first, second, mixed)
    data = live.props["events"]["properties"]["data"]["properties"]
    assert data["limit"] == "text" and data["step"] == "text"
    stored = live.docs[mixed["traceId"]]["events"]
    assert [e["data"]["step"] for e in stored] == ["a", "2"]


def test_a_date_like_string_cannot_make_a_new_field_a_date(live):
    """Dynamic date detection: first seen as `2026-10-08`, a field refuses every later word."""
    first = _save([{"event": "x", "data": {"window": "2026-10-08"}}], "mem-1")
    second = _save([{"event": "x", "data": {"window": "last week"}}], "mem-2")
    assert first["stored"] and second["stored"], (first, second)
    assert live.props["events"]["properties"]["data"]["properties"]["window"] == "text"
    assert json.loads(live.docs[first["traceId"]]["events"][0]["data"]["window"]) == "2026-10-08"


def test_the_whole_document_fits_including_the_top_level():
    body = {"memory_id": "m", "thread_id": None, "owner_id": "alice", "query": "q",
            "answer": None, "model": "m", "provider": "p", "event_count": 1,
            "dropped_count": 0, "createdAt": "2026-10-08T19:55:34Z",
            "events": [{"event": "tool_call", "node": "agent", "agent_role": "agent",
                        "data": {"args": {}, "tool_calls": [{"name": "x", "args": {}}]}}]}
    MappedIndex().check(fit_trace_document(body))
