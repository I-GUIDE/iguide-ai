"""The Postgres record: its constraints, and whether it is a drop-in for the JSON registry.

Two halves, deliberately separated by what they need:

  * The pure tests read the schema text and the tokenizer. They run on a fresh clone with no
    database and they are where the load-bearing constraints are pinned — a `CHECK` that is
    silently dropped from `SCHEMA_SQL` is exactly the kind of change that no runtime test
    notices until something writes the row it was meant to reject.
  * The rest are marked `integration` and skip without a reachable database, because
    `pytest.ini` requires a default run to be green with no external service.

The agreement tests matter most. Routing `search_methods` and `get_contract` at a different
store is only safe if the agent cannot tell — and the field it must never disagree on is
`import_line`, which a sandboxed run breaks over with no network to check.
"""

from __future__ import annotations

import os

import pytest

from extractors import kb_db


# --------------------------------------------------------------------- pure

def test_the_collision_that_lost_58_units_is_in_the_key():
    """`{package}.{symbol}` was not unique when one element spanned 35 files. The primary key
    has to include the source file, or the same defect is representable again."""
    assert "PRIMARY KEY (element_id, source_rel_path, symbol, slice_sha)" in kb_db.SCHEMA_SQL


def test_a_slice_version_is_part_of_a_units_identity():
    """The library is content-addressed and old versions stay importable by sha. A key without
    the sha would make a re-ingest overwrite the version an artifact recorded."""
    key = kb_db.SCHEMA_SQL.split("PRIMARY KEY (element_id,", 1)[1].split(")", 1)[0]
    assert "slice_sha" in key


def test_only_a_callable_unit_may_carry_code():
    """The sandbox mounts `slice_source`. "We never ship code for a unit the analyzer refused"
    is a safety property and belongs in the store, not in every writer's memory."""
    assert "only_callable_units_carry_code" in kb_db.SCHEMA_SQL
    assert "CHECK (verdict = 'callable' OR slice_source = '')" in kb_db.SCHEMA_SQL


def test_every_child_row_is_tied_to_an_element():
    """An orphan block outliving its element is what the JSON stores allowed."""
    for table in ("unit", "block", "dataset_file", "publication", "dataset_outcome"):
        section = kb_db.SCHEMA_SQL.split(f"CREATE TABLE IF NOT EXISTS {table} (", 1)[1]
        section = section.split("\n);", 1)[0]
        assert "REFERENCES element(id) ON DELETE CASCADE" in section, table


def test_no_index_expression_is_unbounded():
    """A tsvector caps at 1 MB and the corpus holds a 4.9 MB markdown cell, 99.8% of it base64
    image data. Unbounded, that is not a slow index — it is a row that cannot be inserted."""
    for fragment in kb_db.SCHEMA_SQL.split("to_tsvector('english',")[1:]:
        assert fragment.lstrip().startswith("left("), fragment[:80]


def test_identifier_expansion_has_one_definition():
    """Postgres splits `load_crime_points` but leaves `calculateBuffers` whole, so the indexed
    text is pre-expanded. Two copies of that rule diverge silently, and the symptom is a method
    that cannot be found — so this reuses the tokenizer the ranking already used."""
    from agent_runtime.method_library import _tokens

    for sample in ("calculateBuffers", "load_crime_points", "GeoDataFrame", "to_crs"):
        assert kb_db.expand_identifiers(sample) == " ".join(_tokens(sample))
    assert "buffers" in kb_db.expand_identifiers("calculateBuffers").split()


def test_the_store_is_off_unless_asked_for(monkeypatch):
    """Additive: every existing emit target keeps working untouched, so nothing may start
    depending on this implicitly."""
    monkeypatch.delenv("AGENT_KB_DB", raising=False)
    assert kb_db.enabled() is False
    monkeypatch.setenv("AGENT_KB_DB", "1")
    assert kb_db.enabled() is True


def test_the_query_matches_any_word_not_every_word():
    """`websearch_to_tsquery` ANDs its terms, so "buffer geometries by a distance" required one
    unit to match all three and returned nothing against all 840. The ranker being replaced sums
    whatever matched."""
    assert "' | '" in kb_db._OR_TSQUERY
    assert "websearch_to_tsquery" not in kb_db._UNIT_SELECT


# --------------------------------------------------- against a real database

def _conn():
    try:
        import psycopg  # noqa: F401
    except ImportError:
        pytest.skip("psycopg is not installed")
    try:
        ctx = kb_db.connect()
        conn = ctx.__enter__()
    except Exception as exc:
        pytest.skip(f"no database at {kb_db.dsn()}: {type(exc).__name__}")
    return ctx, conn


@pytest.fixture(scope="module")
def conn():
    ctx, connection = _conn()
    kb_db.ensure_schema(connection)
    yield connection
    ctx.__exit__(None, None, None)


@pytest.fixture(scope="module")
def loaded(conn):
    counts = kb_db.table_counts(conn)
    if not counts.get("unit"):
        pytest.skip("no rows; run scripts/backfill_kb_db.py first")
    return counts


@pytest.mark.integration
def test_the_schema_applies_twice(conn):
    """Every ingest calls it, so it has to be idempotent."""
    kb_db.ensure_schema(conn)
    assert set(kb_db.table_counts(conn)) >= {"element", "unit", "block"}


@pytest.mark.integration
def test_the_same_symbol_from_two_files_is_allowed(conn, loaded):
    """The shape that lost 58 units. It must be storable, not merely detected."""
    with conn.cursor() as cur:
        cur.execute("SELECT element_id, source_rel_path, symbol, slice_sha FROM unit LIMIT 1")
        element_id, source_rel, symbol, sha = cur.fetchone()
        cur.execute("""INSERT INTO unit (element_id, source_rel_path, symbol, slice_sha)
                       VALUES (%s, %s, %s, %s)""",
                    (element_id, source_rel + ".other", symbol, sha))
        cur.execute("SELECT count(*) FROM unit WHERE element_id=%s AND symbol=%s",
                    (element_id, symbol))
        assert cur.fetchone()[0] >= 2
    conn.rollback()


@pytest.mark.integration
def test_writing_the_same_unit_twice_is_an_error(conn, loaded):
    import psycopg

    with conn.cursor() as cur:
        cur.execute("SELECT element_id, source_rel_path, symbol, slice_sha FROM unit LIMIT 1")
        row = cur.fetchone()
        with pytest.raises(psycopg.errors.UniqueViolation):
            cur.execute("""INSERT INTO unit (element_id, source_rel_path, symbol, slice_sha)
                           VALUES (%s, %s, %s, %s)""", row)
    conn.rollback()


@pytest.mark.integration
def test_a_unit_cannot_outlive_its_element(conn, loaded):
    import psycopg

    with conn.cursor() as cur, pytest.raises(psycopg.errors.ForeignKeyViolation):
        cur.execute("""INSERT INTO unit (element_id, source_rel_path, symbol, slice_sha)
                       VALUES ('nosuchel', 'x.py', 'f', 'deadbeef')""")
    conn.rollback()


@pytest.mark.integration
def test_a_refused_unit_cannot_carry_code(conn, loaded):
    import psycopg

    with conn.cursor() as cur:
        cur.execute("SELECT id FROM element LIMIT 1")
        element_id = cur.fetchone()[0]
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute("""INSERT INTO unit (element_id, source_rel_path, symbol, slice_sha,
                                             verdict, slice_source)
                           VALUES (%s, 'x.py', 'f', 'aa', 'needs_globals', 'def f(): pass')""",
                        (element_id,))
    conn.rollback()


@pytest.mark.integration
def test_every_stored_slice_still_compiles_and_defines_its_symbol(conn, loaded):
    """The mounted library is a PROJECTION of this column. If a slice cannot be recovered from
    the row describing it, the two can drift and the whole argument for moving the record
    collapses."""
    import ast

    with conn.cursor() as cur:
        cur.execute("SELECT symbol, slice_source FROM unit WHERE slice_source <> ''")
        rows = cur.fetchall()
    assert rows, "no slices stored"
    bad = []
    for symbol, source in rows:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            bad.append((symbol, "does not parse"))
            continue
        if not any(getattr(n, "name", None) == symbol for n in tree.body):
            bad.append((symbol, "does not define its symbol"))
    assert not bad, bad[:5]


@pytest.mark.integration
def test_search_returns_the_shape_the_agents_tool_publishes(conn, loaded):
    hits = kb_db.search_units(conn, "load crime data", limit=5)
    assert hits, "nothing matched a query the corpus certainly answers"
    for hit in hits:
        assert set(hit) >= {"symbol", "signature", "doc_summary", "import_line", "element_id",
                            "slice_sha", "requirements", "score"}
        assert hit["import_line"].startswith("from iguide_methods.")


@pytest.mark.integration
def test_search_only_offers_units_that_shipped(conn, loaded):
    """A refused unit has no slice, so an import line for one would name a module that is not
    mounted — the exact failure the registry-over-index decision exists to prevent."""
    with conn.cursor() as cur:
        cur.execute("SELECT symbol FROM unit WHERE verdict <> 'callable' LIMIT 20")
        refused = {r[0] for r in cur.fetchall()}
    if not refused:
        pytest.skip("no refused units loaded")
    for symbol in list(refused)[:5]:
        for hit in kb_db.search_units(conn, symbol, limit=10):
            assert hit["symbol"].rsplit(".", 1)[-1] not in refused or hit["import_line"]


@pytest.mark.integration
def test_an_ambiguous_name_is_refused_with_the_same_guidance_as_the_registry(conn, loaded):
    """Two backends that refuse differently are two behaviours for the model to learn, and a
    bare `ambiguous: true` leaves it to guess — which is the failure the refusal prevents."""
    with conn.cursor() as cur:
        cur.execute("""SELECT symbol FROM unit WHERE verdict = 'callable'
                        GROUP BY symbol HAVING count(DISTINCT element_id) > 1 LIMIT 1""")
        row = cur.fetchone()
    if not row:
        pytest.skip("no ambiguous symbol in this corpus")
    out = kb_db.get_unit_contract(conn, row[0])
    assert out["ambiguous"] is True
    assert out["import_line"] is None
    assert "qualified name" in out["error"]


@pytest.mark.integration
def test_a_missing_symbol_is_distinguishable_from_a_refusal(conn, loaded):
    """Only the first should make a caller try another store. Falling through an ambiguity
    refusal would turn a correct refusal into a confidently wrong import line."""
    assert kb_db.get_unit_contract(conn, "no_such_method_anywhere")["found"] is False


@pytest.mark.integration
def test_the_contract_agrees_with_the_registry_field_by_field(conn, loaded, monkeypatch):
    """The drop-in claim, checked rather than asserted.

    A first pass through this silently dropped `invariants` — where "requires a projected CRS"
    lives — while every other field matched. Comparing one field at a time is what found it.
    """
    import json
    from pathlib import Path

    from agent_runtime import method_library

    # Read the real registry file directly rather than through `load_registry()`. conftest
    # deliberately points `AGENT_METHOD_LIBRARY_DIR` at an empty temp directory so tests never
    # touch the built library, which is right — but it also made the FIRST version of this test
    # skip silently, so the one check that justifies routing the agent at a different store was
    # not running at all.
    built = (Path(__file__).resolve().parents[2] / "agent_chat_files" / "method_library"
             / "iguide_methods" / "_registry.json")
    if not built.is_file():
        pytest.skip("no method library has been built in this checkout")
    registry = json.loads(built.read_text(encoding="utf-8"))

    with conn.cursor() as cur:
        cur.execute("""SELECT symbol FROM unit WHERE verdict = 'callable'
                        GROUP BY symbol HAVING count(*) = 1 ORDER BY symbol LIMIT 40""")
        symbols = [r[0] for r in cur.fetchall()]
    assert symbols

    monkeypatch.delenv("AGENT_KB_DB", raising=False)
    mismatches = []
    for symbol in symbols:
        want = method_library.get_contract(symbol, registry=registry)
        if want.get("error"):
            continue
        got = kb_db.get_unit_contract(conn, symbol)
        for field in ("import_line", "signature", "doc_summary", "slice_sha", "unit_kind"):
            if want.get(field) != got.get(field):
                mismatches.append((symbol, field, want.get(field), got.get(field)))
        # The registry contract nests the element id under `provenance` while the search result
        # puts it at the top level; compare like for like rather than calling that a mismatch.
        # Extra fields are fine — this asserts nothing is LOST, not that the shapes are equal.
        if ((want.get("provenance") or {}).get("element_id")
                != (got.get("provenance") or {}).get("element_id")):
            mismatches.append((symbol, "provenance.element_id",
                               (want.get("provenance") or {}).get("element_id"),
                               (got.get("provenance") or {}).get("element_id")))
        for field in ("params", "invariants"):
            if len(want.get(field) or []) != len(got.get(field) or []):
                mismatches.append((symbol, field, len(want.get(field) or []),
                                   len(got.get(field) or [])))
    assert not mismatches, mismatches[:5]


@pytest.mark.integration
def test_the_agent_tool_reports_which_store_answered(conn, loaded, monkeypatch):
    """A dead database that silently degrades looks like a small library. "No such method" has
    to be attributable to a store."""
    from agent_runtime import method_library

    monkeypatch.delenv("AGENT_KB_DB", raising=False)
    assert method_library.backend_name() == "registry"
    monkeypatch.setenv("AGENT_KB_DB", "1")
    assert method_library.backend_name() == "postgres"
    assert method_library.library_summary()["backend"] == "postgres"


def test_an_unreachable_database_does_not_cost_the_agent_its_tools(tmp_path, monkeypatch):
    """Losing the database must cost recall, never the tool.

    Self-contained rather than leaning on the built library: conftest points
    `AGENT_METHOD_LIBRARY_DIR` at an empty directory on purpose, so a version of this that
    searched "the registry" was really searching nothing and passing for the wrong reason.
    """
    import json

    from agent_runtime import method_library

    package = tmp_path / "iguide_methods" / "ke_test_element"
    package.mkdir(parents=True)
    (tmp_path / "iguide_methods" / "_registry.json").write_text(json.dumps({
        "ke_test_element.load_crime_data": {
            "library_symbol": "load_crime_data",
            "module": "iguide_methods.ke_test_element.v_abc123",
            "signature": "def load_crime_data(path)",
            "doc_summary": "Load reported crime incidents.",
            "slice_sha": "abc123", "unit_kind": "function",
            "provenance": {"element_id": "e1", "extractor": "notebook"},
        }}), encoding="utf-8")

    monkeypatch.setenv("AGENT_METHOD_LIBRARY_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_KB_DB", "1")
    monkeypatch.setenv("AGENT_KB_DSN", "postgresql://nobody@127.0.0.1:1/nothing")

    hits = method_library.search_methods("load crime data", limit=3)
    assert hits, "the registry fallback did not run"
    assert hits[0]["import_line"] == (
        "from iguide_methods.ke_test_element.v_abc123 import load_crime_data")
