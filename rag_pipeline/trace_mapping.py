"""What the `chat_traces` index will accept, and the writer-side fit that keeps every turn inside it.

`chat_traces` has no explicit mapping. OpenSearch typed each field from the first document that
carried it, and once a field has a type a document disagreeing with it is rejected WHOLE, with a
400 that `save_turn_trace` can only log. The trace is then silently lost.

That is how `events.data.args` became a trap. LangChain hands `on_tool_start` a Python repr of
the arguments (`"{'query': 'x'}"`), which is not JSON, so `args` stayed a string and the field was
mapped `text`. A tool called with NO arguments arrives as `"{}"`, which IS JSON, parses to a
dict, and every turn that called one was refused: 30 turns between 10-01 and 10-08, all on
`events.data.args`. Other emitters (`_normalize_tool_call`, the supervisor's automatic calls) send
dicts with real keys, and those turns were refused the same way.

The fix is on the writer because the index is on the shared prod cluster and a field's type
cannot be changed in place. Three rules:

* A tool's arguments are always stored as JSON text. That is the type the index already has,
  and it is the only type they COULD share: each tool has its own argument names and types
  (`limit: 6` here, `limit: "all"` there), so as an object they would collide with each other
  and grow the mapping by one field per argument name ever used.
* A mapped field gets a value of its mapped type. A structure sent to a `text` field becomes
  JSON text in place. A value a numeric, boolean or object field cannot hold moves to a sibling
  `<field>_text`, so it is kept and searchable rather than costing the whole turn.
* An unmapped field is stored as text: strings as they are, anything else as JSON. Letting it
  map dynamically would let one turn's shape decide every later turn's, which is the original
  bug again, and a number is no safer than a dict: a new field seen as `3` maps `long` and then
  refuses `"all"`, and one seen as text then as a number in the SAME document is refused
  outright (`cannot be changed from type [text] to [long]`, measured on OpenSearch 2.14.0).
  A date-like string is quoted too: a new field first seen as `"2026-10-08"` maps `date` and
  then refuses `"yesterday"`. The cost is that a new numeric or date field is searchable only
  as text until someone maps it.

`TRACE_MAPPING` is a snapshot of the live mapping, read with `GET chat_traces/_mapping` on
2026-10-08. It is a Python module and not a JSON file because `.dockerignore` excludes `*.json`,
and a snapshot missing from the image would fit nothing. When the live mapping gains a field,
add it here; until then that field is handled by the unmapped rules above.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

# Keys whose value is a tool's arguments, wherever they appear in an event.
ARGUMENT_KEYS = frozenset({"args", "arguments", "tool_args"})

_EVENT_DATA = {
    "agent": "text",
    "args": "text",
    "attempt": "long",
    "attempts": "long",
    "blocked_on_capability": "boolean",
    "bounds": "float",
    "content": "text",
    "count": "long",
    "duration_s": "float",
    "embedding": {"properties": {
        "file_id": "text",
        "filename": "text",
        "model": "text",
        "models_in_package": "text",
        "months": "text",
        "recoloured_on_shared_basis": "boolean",
    }},
    "flagged": "boolean",
    "hallucination_detected": "boolean",
    "id": "text",
    "issues": {"properties": {
        "claim": "text",
        "reason": "text",
        "source": "text",
        "status": "text",
    }},
    "kind": "text",
    "label": "text",
    "legend": {"properties": {"color": "long", "label": "text"}},
    "message": "text",
    "model": "text",
    "name": "text",
    "next": "text",
    "opacity": "float",
    "outcome": "text",
    "outline": "boolean",
    "reason": "text",
    "render": "text",
    "route": "text",
    "rows": "long",
    "sampled": "boolean",
    "sequence": "long",
    "severity": "text",
    "source": "text",
    "stage": "text",
    "style_by": "text",
    "tool_calls": {"properties": {"args": "text", "name": "text"}},
    "tool_name": "text",
    "tools": "text",
    "total": "long",
    "url": "text",
    "why": "text",
}

# Leaf -> type name; object -> {"properties": {...}}. Every text field also carries a
# `keyword` sub-field (ignore_above 256), which does not affect what a document may hold.
TRACE_MAPPING: Dict[str, Any] = {
    "answer": "text",
    "createdAt": "date",
    "dropped_count": "long",
    "event_count": "long",
    "events": {"properties": {
        "agent_role": "text",
        "data": {"properties": _EVENT_DATA},
        "event": "text",
        "node": "text",
    }},
    "memory_id": "text",
    "owner_id": "text",
    "query": "text",
    "thread_id": "text",
}

_TEXT = frozenset({"text", "keyword"})
# What dynamic date detection catches, generously: OpenSearch 2.14.0 mapped `2026-10`,
# `2026-10-08`, `2026-10-08T19:55:34Z`, `2026/10/08` and `-2026-10-08` as `date`, and left
# `2026`, `20261008`, `10:30` and `2026-10-08 19:55:34` as text. Over-matching costs only quotes.
_DATE_LIKE = re.compile(r"^[+-]?\d{4,}[-/]\d")
_NUMERIC = frozenset({"long", "integer", "short", "byte", "float", "double", "half_float",
                      "scaled_float"})


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _flatten(value: Any):
    """OpenSearch indexes an array of values, at any depth, as that many values of the field."""
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten(item)
    else:
        yield value


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value)  # numeric strings are coerced by the index
            return True
        except ValueError:
            return False
    return False


def leaf_accepts(field_type: str, value: Any) -> bool:
    """Whether a leaf field of ``field_type`` will take ``value`` without a mapping error."""
    for item in _flatten(value):
        if item is None:
            continue
        if isinstance(item, dict):
            return False
        if field_type in _TEXT:
            if not isinstance(item, (str, int, float, bool)):
                return False
        elif field_type in _NUMERIC:
            if not _is_number(item):
                return False
        elif field_type == "boolean":
            if not (isinstance(item, bool) or item in ("true", "false", "")):
                return False
        elif not isinstance(item, (str, int, float)):  # date and anything rarer
            return False
    return True


def _plain_text(value: Any) -> bool:
    """Strings that a field first seen now will map as `text`, and keep it that way."""
    return all(item is None or (isinstance(item, str) and not _DATE_LIKE.match(item))
               for item in _flatten(value))


def _fit_object(obj: Dict[str, Any], props: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in obj.items():
        key = str(key)
        if key in ARGUMENT_KEYS and value is not None and not isinstance(value, str):
            value = _json_text(value)
        spec = (props or {}).get(key)
        if spec is None:
            out[key] = value if _plain_text(value) else _json_text(value)
        elif isinstance(spec, dict):
            fitted = _fit_object_field(value, spec["properties"])
            if fitted is _REJECT:
                out[f"{key}_text"] = _json_text(value)
            else:
                out[key] = fitted
        elif leaf_accepts(spec, value):
            out[key] = value
        elif spec in _TEXT:
            out[key] = _json_text(value)
        else:
            out[f"{key}_text"] = _json_text(value)
    return out


_REJECT = object()


def _fit_object_field(value: Any, props: Dict[str, Any]) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return _fit_object(value, props)
    if isinstance(value, (list, tuple)) and all(isinstance(v, dict) or v is None for v in value):
        return [None if v is None else _fit_object(v, props) for v in value]
    return _REJECT


def fit_trace_document(body: Dict[str, Any]) -> Dict[str, Any]:
    """A copy of ``body`` that the `chat_traces` mapping will accept. Never raises."""
    return _fit_object(dict(body), TRACE_MAPPING)
