"""How the knowledge base OFFERS what it holds, as distinct from what it holds.

Retrieval quality here is not a ranking problem. Cells and units answer different questions —
"how did someone do this" and "what can I call" — and a single text-similarity ranking lets cells
win, because a cell indexes its whole markdown and code while a unit indexes a signature and one
summary line. Audited over six queries: 24 of 48 hits actionable, and a plainly method-shaped
question ("count how many points fall in each polygon") returned four cells and ONE distinct
callable method.

Two things this file records that are easy to get wrong twice:

* **Length normalisation is not the fix.** Tested: `ts_rank` flags 2 and 8 promote 22-character
  cells ("view point", "view polygon") and push units out of the top six entirely.
* **A quota is not the fix either.** The first attempt took exactly `size // 2` units and filled
  the rest with cells, which capped units as well as guaranteeing them — a query matching eight
  methods lost four — and pinned the metric it was meant to move at exactly 50% by construction.
"""

from __future__ import annotations

import pytest

from extractors import kb_db


def _conn():
    try:
        import psycopg  # noqa: F401
    except ImportError:
        pytest.skip("psycopg is not installed")
    try:
        ctx = kb_db.connect()
        return ctx, ctx.__enter__()
    except Exception as exc:
        pytest.skip(f"no database at {kb_db.dsn()}: {type(exc).__name__}")


@pytest.fixture(scope="module")
def conn():
    ctx, connection = _conn()
    if not kb_db.table_counts(connection).get("unit"):
        ctx.__exit__(None, None, None)
        pytest.skip("no rows; run scripts/backfill_kb_db.py first")
    yield connection
    ctx.__exit__(None, None, None)


# ----------------------------------------------------------------- pure, no database

def test_the_offering_uses_a_floor_and_never_a_cap():
    """Read from the source, because the difference is one line and it is the whole point."""
    import inspect

    src = inspect.getsource(kb_db.search_kb)
    assert "floor = max(1, size // 2)" in src
    assert "deduped[:half]" not in src, "units are being capped again"


def test_a_refused_unit_may_not_displace_a_callable_one():
    import inspect

    src = inspect.getsource(kb_db.search_kb)
    assert "(deduped or refused_units)" in src, (
        "refused units must be a fallback, not a peer of callable ones")


# --------------------------------------------------------------- against the record

@pytest.mark.integration
def test_a_method_shaped_question_returns_methods(conn):
    """The defect this revision exists for."""
    for query in ("count how many points fall in each polygon",
                  "reproject a geodataframe before measuring distance",
                  "compute NDVI from sentinel-2"):
        hits = kb_db.search_kb(conn, query, size=8)
        units = [h for h in hits if h["_index"] == "pg:unit"]
        assert len(units) >= 4, f"{query!r} returned only {len(units)} callable methods"


@pytest.mark.integration
def test_a_query_that_matches_many_methods_is_not_cut_back_to_the_floor(conn):
    """The regression the quota introduced: eight matching methods became four.

    Asserted against the FLOOR, not against a fixed eight. The first version of this test
    demanded exactly eight and failed at seven — which was dedup correctly collapsing a
    duplicate symbol, not the floor capping anything. A test that cannot tell those apart would
    have been "fixed" by deleting the dedup.
    """
    hits = kb_db.search_kb(conn, "aggregate points into hexagon grid cells", size=8)
    units = [h for h in hits if h["_index"] == "pg:unit"]
    assert len(units) > 8 // 2, f"units were cut back to the floor: {len(units)}"


@pytest.mark.integration
def test_the_same_symbol_is_not_offered_twice(conn):
    """`spatial_join_and_count` is defined by two elements. That ambiguity is real and the
    caller must resolve it, but two identical rows spend two slots to say one thing."""
    hits = kb_db.search_kb(conn, "spatial join count points in polygons", size=8)
    seen = []
    for hit in hits:
        unit = (hit["_source"].get("extracted") or {}).get("unit") or {}
        if unit:
            seen.append((unit.get("library_symbol"), unit.get("signature")))
    assert len(seen) == len(set(seen)), seen


@pytest.mark.integration
def test_a_callable_unit_is_preferred_over_a_refused_one(conn):
    """Both doors into the KB must agree. `search_units` filtered on verdict and `search_kb`
    did not, so a `needs_instance` unit could be offered through `agent_kb_search` as if it
    were callable — reached, in the audit, by querying with its own summary."""
    with conn.cursor() as cur:
        cur.execute("""SELECT doc_summary FROM unit WHERE verdict <> 'callable'
                        AND doc_summary <> '' ORDER BY length(doc_summary) DESC LIMIT 1""")
        row = cur.fetchone()
    if not row:
        pytest.skip("no refused units loaded")
    hits = kb_db.search_kb(conn, row[0], size=8)
    units = [((h["_source"].get("extracted") or {}).get("unit") or {}) for h in hits]
    units = [u for u in units if u]
    verdicts = [(u.get("callability") or {}).get("verdict") for u in units]
    if not verdicts:
        pytest.skip("the probe query matched no units")

    # DISPLACE, not outrank. The first version of this test demanded that every callable unit
    # rank above every refused one, and that is the wrong rule: this probe queries with a
    # refused unit's own summary, so it legitimately matches best, and burying the most
    # relevant hit under weaker ones would be worse than showing it labelled. What must never
    # happen is a refused unit taking a slot a callable one would have had.
    assert any(v == "callable" for v in verdicts), verdicts
    assert all(v is not None for v in verdicts), (
        "a unit reached the agent with no verdict at all — the affirmative fact was implicit "
        "again")


@pytest.mark.integration
def test_first_contact_is_an_excerpt_and_names_its_reader(conn):
    """Eight cells at 4,000 characters was most of a ~4,000-token search payload, spent before
    the agent had chosen a cell. The body is one `get_kb_block(doc_id)` away."""
    from rag_pipeline.search.agent_kb import normalize_hits

    hits = kb_db.search_kb(conn, "reproject to a projected crs", size=8)
    docs = normalize_hits(hits, [], 8)
    assert docs
    for doc in docs:
        assert len(doc["contents"]) <= 900, len(doc["contents"])
        if "more chars" in doc["contents"]:
            assert "get_kb_block" in doc["contents"]
            assert doc["doc_id"], "the reader was named but not the id it needs"


# ------------------------------------------------- publications: the third channel

@pytest.mark.integration
def test_extracted_method_specs_are_reachable(conn):
    """203 publication specs sat behind a GIN index nothing queried.

    `search_kb` searched `block` and `unit`. Each of those specs cost an LLM call to produce and
    none of them could reach the agent. Measured on the case that exposed it: asked for a
    paper's catchment bands and decay weights, the agent answered "its abstract (the only text
    retrieved) does not state the specific minute-based catchment bands" — it had found the
    paper AND the callable implementation, and the spec between them was the missing half.
    """
    hits = kb_db.search_kb(conn, "enhanced two step floating catchment area accessibility",
                           size=8)
    specs = [h for h in hits if h["_index"] == "pg:publication"]
    assert specs, "no method spec reachable for a query that names the method"


@pytest.mark.integration
def test_a_spec_is_carried_whole_including_its_parameters(conn):
    """The parameters a signature cannot express live in the middle of the step list.

    A first budget of 2,600 characters cut the E2SFCA spec at step 12, one step before "Apply
    distance-decay weights (1, 0.68, 0.22)" — truncating away the exact numbers that were the
    reason to retrieve it. The budget is now set from the corpus (63 specs, median 3,060, max
    5,615), not guessed.
    """
    hits = kb_db.search_kb(conn, "enhanced two step floating catchment area distance decay "
                                 "weights", size=8)
    spec = next((h for h in hits if h["_index"] == "pg:publication"), None)
    if spec is None:
        pytest.skip("the E2SFCA spec is not in this record")
    contents = spec["_source"]["contents"]
    assert "0.68" in contents and "0.22" in contents, "the decay weights were truncated away"
    assert spec["_source"]["extracted"]["spec"]["step_count"] > 20


@pytest.mark.integration
def test_no_spec_in_the_corpus_is_truncated(conn):
    """The budget must clear the real distribution, not the median."""
    with conn.cursor() as cur:
        cur.execute("""SELECT max(length(summary) + coalesce((
                         SELECT sum(length(v)) + 4 * count(*)
                           FROM jsonb_array_elements_text(p.steps) v), 0))
                         FROM publication p WHERE jsonb_array_length(p.steps) > 0""")
        longest = cur.fetchone()[0] or 0
    assert longest <= kb_db._SPEC_CHARS, (
        f"the longest spec is {longest} chars and the budget is {kb_db._SPEC_CHARS}")


# ------------------------------------------- collapsing a duplicate must not erase the fact

@pytest.mark.integration
def test_a_collapsed_duplicate_still_reports_the_ambiguity(conn):
    """Dedup that hides a real ambiguity is worse than the duplication it removes.

    `spatial_join_and_count` is defined by two elements with byte-identical signatures and
    different slice shas, so the bare name does not identify code. Asked for "the exact import
    line", the agent returned ONE and did not mention the other — it had been shown both,
    adjacent and distinguishable only by a hash inside a module path, and took the first.
    `get_contract` refuses a bare ambiguous name for exactly this reason and was never
    consulted, because the search result already looked like an answer.
    """
    hits = kb_db.search_units(conn, "spatial join count points in polygons", limit=6)
    shared = [h for h in hits if h.get("ambiguous")]
    if not shared:
        pytest.skip("no ambiguous symbol matched this query in this record")
    for hit in shared:
        assert hit.get("also_defined_by"), hit["symbol"]
        assert all(s.get("element_id") for s in hit["also_defined_by"])
        assert hit.get("disambiguate_with"), "named the problem without naming the resolution"


@pytest.mark.integration
def test_collapsing_does_not_shrink_the_result_set(conn):
    """The collapse happens after an over-fetch, so asking for six still returns six."""
    hits = kb_db.search_units(conn, "load data", limit=6)
    assert len(hits) == 6, len(hits)
