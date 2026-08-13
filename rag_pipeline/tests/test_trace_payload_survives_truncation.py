"""What the trace tells a developer, when the payload it describes has been cut short.

The reasoning trace is truncated for display, and the client salvages what it can from the
fragment by regex — it cannot parse half a JSON document. So a scalar summary placed AFTER a
long array is simply gone by the time anyone reads it, and the line renders from whatever the
salvage found: nothing.

Observed live in the prototype: `kb_method_search → no results (log truncated)` for a call that
returned TEN methods. That is not cosmetic. A developer reading that trace concludes the method
library is empty and goes to fix the extraction — I did exactly that, and spent the detour
proving the library was fine.

Two independent causes, both fixed and both asserted here: the count came after the bulk, and a
method-unit row carries no `title`/`name`/`doc_id`/`url` for the client's row builder to find.
"""

from __future__ import annotations

import json
import re

import pytest


def _payload(query="buffer geometry spatial", limit=10):
    from agent_runtime.langchain_granular_tools import kb_method_search_tool

    return kb_method_search_tool(query, limit)


@pytest.fixture(autouse=True)
def _library(monkeypatch):
    import os

    monkeypatch.setenv("AGENT_METHOD_LIBRARY_DIR",
                       os.path.abspath("agent_chat_files/method_library"))


def test_the_count_is_emitted_at_all():
    """The client cannot derive it: no row has a title, so `items.length` is 0 for real hits."""
    assert "count" in json.loads(_payload())


def test_the_count_precedes_the_results_array():
    """Ordering is load-bearing, not style. `json.dumps` preserves dict order, and the salvage
    regexes a truncated prefix — so `count` after `results` is unreachable exactly when it is
    needed."""
    raw = _payload()
    assert raw.index('"count"') < raw.index('"results"')


@pytest.mark.parametrize("cut", [80, 150, 400, 2000])
def test_the_count_survives_an_arbitrary_cut(cut):
    """Mirrors the client's salvage: regex the prefix, do not parse it."""
    raw = _payload()
    if len(raw) <= cut:
        pytest.skip("payload shorter than the cut")
    found = re.search(r'"count"\s*:\s*(-?\d+)', raw[:cut])
    assert found, f"count is unrecoverable from the first {cut} chars"
    assert int(found.group(1)) == json.loads(raw)["count"]


def test_a_method_row_is_identifiable_by_symbol():
    """The client builds a row from title|name|doc_id|id|symbol. A method unit has only the
    last, so without it every row was skipped and the count fell through to zero."""
    results = json.loads(_payload())["results"]
    assert results, "the corpus library should match a geometry query"
    assert all(r.get("symbol") for r in results)
    assert not any(r.get("title") or r.get("name") or r.get("doc_id") for r in results), (
        "if a method row ever gains a title, this test is stale — but the client must not "
        "depend on one appearing")


def test_the_prototype_row_builder_accepts_a_symbol():
    """Asserted against the shipped page, because the fix has to be in the file the browser
    loads, not only in the server."""
    from pathlib import Path

    src = Path("examples/iguide_chat_prototype.html").read_text(encoding="utf-8")
    assert "d.symbol" in src, "payloadItems must recognise a method-unit row"
    assert "title|symbol" in src, "looseJson must salvage a symbol from a truncated payload"


def test_an_empty_library_and_an_unmatched_query_stay_distinguishable():
    """Pre-existing behaviour worth pinning: "nothing ingested" and "nothing matched" are
    different situations and the model cannot tell them apart from an empty list."""
    payload = json.loads(_payload("zzz_no_such_method_anywhere_zzz"))
    assert payload["count"] == 0
    assert "note" in payload and "units from" in payload["note"]
