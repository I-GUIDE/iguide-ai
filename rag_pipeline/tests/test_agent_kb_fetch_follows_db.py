"""Every id a search returns can be fetched from the store that returned it.

``agent_kb_search`` reads the Postgres record first when ``AGENT_KB_DB`` is set. ``get_kb_block``
and ``kb_for_elements`` chose their store from ``AGENT_KB_BACKEND`` alone, so on the VM (both flags
set) a search answered from Postgres and the fetch of its own results went to the cluster.
Measured 2026-10-10 inside agent-api: 6 of 6 NotebookBlock ids found (the cluster holds the same
blocks), 0 of 6 MethodUnit ids (``<el>::unit::<symbol>::<sha>``), 0 of 5 PublicationMethodSpec ids
(``<el>::methodspec``) — the two id kinds the record MINTS rather than stores.

The offline half runs the real ``kb_db`` readers against an in-memory connection that answers
their SQL from three small tables. The integration half (skips without a database) checks the
round trip the bug broke: fetch every id ``search_kb`` returns and get the same document back.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from extractors import kb_db
from rag_pipeline.search import agent_kb

EL = "b1fa548b"
SPEC_EL = "5e7c0001"
DATA_EL = "265e6957"

BLOCKS = [
    # _BLOCK_COLS order: doc_id, element_id, title, markdown, code, tools, ord, imports,
    #                    element_title, element_type
    (f"{EL}::block::1", EL, "", "Load the crimes.", "df = pd.read_csv('crimes.csv')",
     ["pandas"], 1, ["pandas"], "Crime buffers", "notebook"),
    (f"{EL}::block::0", EL, "", "", "import pandas as pd", [], 0, ["pandas"],
     "Crime buffers", "notebook"),
]
UNITS = [
    # _UNIT_COLS order
    (EL, "calculate_buffers", "crime.calculate_buffers", "iguide_methods.ke_b1fa548b.v_abc123",
     "abc123", "def calculate_buffers(gdf, buffer)", "Replace geometry with buffers.",
     "function", "callable", {"pip": ["geopandas"]},
     [{"check": "projected_crs", "target": "gdf"}],
     [{"name": "buffer", "declared_unit": "metres"}], "GeoDataFrame", {}, "Crime buffers"),
]
SPECS = [
    # _SPEC_COLS order: element_id, title, summary, steps, tools, datasets, doi, implemented_by
    (SPEC_EL, "E2SFCA accessibility", "Two-step floating catchment.",
     ["Build catchments", "Apply distance-decay weights (1, 0.68, 0.22)"], ["networkx"], [],
     "10.1/x", [{"element_id": EL, "title": "Crime buffers", "confidence": "high",
                 "evidence": "cited doi"}]),
]
ELEMENTS = {EL: ("notebook", "Crime buffers"), SPEC_EL: ("publication", "E2SFCA accessibility"),
            DATA_EL: ("dataset", "Crime points")}
DATASETS = [(DATA_EL, "crimes.csv", {"format": "CSV", "crs": "EPSG:4326", "row_count": 20000,
                                     "schema": ["ID", "Latitude", "Longitude"]})]


class FakeCursor:
    """Answers the readers' queries by the tables and keys they name. Not a SQL engine — a
    query this does not recognise fails loudly rather than returning nothing."""

    def __init__(self, conn):
        self.conn = conn
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=()):
        self.conn.queries.append(sql)
        flat = " ".join(sql.split())
        if "FROM unit u" in flat and "u.slice_sha = %s" in flat:
            el, sym, sha = params
            rows = [u for u in UNITS if (u[0], u[1], u[4]) == (el, sym, sha)]
        elif "FROM unit u" in flat and "u.element_id = ANY" in flat:
            rows = [u for u in UNITS if u[0] in params[0]]
        elif "FROM publication p" in flat and "p.element_id = %s" in flat:
            rows = [s for s in SPECS if s[0] == params[0]]
        elif "FROM publication WHERE element_id = ANY" in flat:
            rows = [(s[0], "ok", s[3], s[5], s[4]) for s in SPECS if s[0] in params[0]]
        elif "FROM block b" in flat and "b.doc_id = %s" in flat:
            rows = [b for b in BLOCKS if b[0] == params[0]]
        elif "FROM block b" in flat and "b.element_id = %s" in flat:
            rows = sorted((b for b in BLOCKS if b[1] == params[0]), key=lambda b: b[6])
        elif "FROM block b" in flat and "b.element_id = ANY" in flat:
            rows = sorted((b for b in BLOCKS if b[1] in params[0]), key=lambda b: (b[1], b[6]))
        elif "FROM dataset_file f" in flat:
            rows = [d for d in DATASETS if d[0] in params[0]]
        elif "FROM element WHERE id = ANY" in flat:
            rows = [(i, ELEMENTS[i][1], ELEMENTS[i][0], "", "") for i in params[0]
                    if i in ELEMENTS]
        else:                                                        # pragma: no cover
            raise AssertionError(f"FakeCursor does not know this query: {flat[:120]}")
        self._rows = list(rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


class FakeConn:
    def __init__(self):
        self.queries = []

    def cursor(self):
        return FakeCursor(self)


@pytest.fixture
def db(monkeypatch):
    """AGENT_KB_DB on, the connection faked, and the cluster made unusable: any read that went
    to it instead of the record fails the test rather than quietly answering."""
    conn = FakeConn()

    @contextmanager
    def _connect(*_a, **_kw):
        yield conn

    monkeypatch.setenv("AGENT_KB_DB", "1")
    monkeypatch.setenv("AGENT_KB_BACKEND", "opensearch")
    monkeypatch.setattr(kb_db, "connect", _connect)

    def _no_cluster():
        raise AssertionError("read the cluster while AGENT_KB_DB is set")

    monkeypatch.setattr(agent_kb, "_os_client", _no_cluster)
    return conn


# ------------------------------------------------------------------ the three id kinds

def test_a_block_id_is_fetched_from_the_record(db):
    out = agent_kb.get_kb_block(f"{EL}::block::1")
    assert out["found"] is True and out["backend"] == "postgres"
    assert out["index"] == "pg:block"
    assert out["source"]["extracted"]["block"]["code"] == "df = pd.read_csv('crimes.csv')"


def test_a_unit_id_is_parsed_back_into_the_key_it_was_minted_from(db, monkeypatch):
    """`<el>::unit::<symbol>::<sha>` is not stored anywhere; 0 of 6 were found by a lookup."""
    monkeypatch.setenv("AGENT_EXTRACTION", "1")
    out = agent_kb.get_kb_block(f"{EL}::unit::calculate_buffers::abc123")
    assert out["found"] is True and out["index"] == "pg:unit"
    unit = out["source"]["extracted"]["unit"]
    assert unit["library_symbol"] == "calculate_buffers"
    assert unit["import_line"] == "from iguide_methods.ke_b1fa548b.v_abc123 import calculate_buffers"
    assert unit["callability"]["verdict"] == "callable"


def test_a_unit_id_with_the_wrong_sha_is_not_found(db):
    """The sha is part of the identity: another version is not this one."""
    assert agent_kb.get_kb_block(f"{EL}::unit::calculate_buffers::ffffff")["found"] is False


def test_a_methodspec_id_is_fetched_whole(db):
    out = agent_kb.get_kb_block(f"{SPEC_EL}::methodspec")
    assert out["found"] is True and out["index"] == "pg:publication"
    src = out["source"]
    assert src["resource-type"] == "PublicationMethodSpec"
    assert "2. Apply distance-decay weights (1, 0.68, 0.22)" in src["contents"]
    assert "IMPLEMENTED BY element b1fa548b" in src["contents"]


def test_a_bare_element_id_bundles_its_blocks_in_notebook_order(db):
    out = agent_kb.get_kb_block(EL)
    assert out["found"] is True and out["is_element"] is True
    assert out["backend"] == "postgres"
    assert out["block_ids"] == [f"{EL}::block::0", f"{EL}::block::1"]
    code = out["source"]["extracted"]["block"]["code"]
    assert code.index("import pandas") < code.index("read_csv")


def test_an_element_with_no_blocks_says_it_is_in_the_record(db):
    """"not found" for a publication element must not read as "not in the corpus"."""
    out = agent_kb.get_kb_block(SPEC_EL)
    assert out["found"] is False and "in the record" in out["note"]
    assert "note" not in agent_kb.get_kb_block("00000000")


def test_the_fetched_doc_is_the_searched_doc(db):
    """Search and fetch build the doc with the same function, so they cannot disagree."""
    unit_hit = kb_db._unit_hit(UNITS[0], 0.5)
    fetched = agent_kb.get_kb_block(unit_hit["_id"])
    assert fetched["source"] == agent_kb._reference_only(unit_hit["_source"])


# ------------------------------------------------------------------ gating survives the move

def test_extraction_off_strips_the_import_line_from_a_db_unit(db, monkeypatch):
    """M8.67: no library is mounted, so nothing may be handed over as importable."""
    monkeypatch.delenv("AGENT_EXTRACTION", raising=False)
    unit = agent_kb.get_kb_block(f"{EL}::unit::calculate_buffers::abc123")["source"][
        "extracted"]["unit"]
    assert "import_line" not in unit and "slice_sha" not in unit and "callability" not in unit
    assert unit["signature"].startswith("def calculate_buffers")


def test_extraction_off_strips_the_import_line_from_the_join(db, monkeypatch):
    monkeypatch.delenv("AGENT_EXTRACTION", raising=False)
    unit = agent_kb.kb_for_elements([EL])[EL]["units"][0]
    assert "import_line" not in unit and unit["symbol"] == "calculate_buffers"


# ------------------------------------------------------------------ the join by element

def test_the_join_reads_the_record(db, monkeypatch):
    monkeypatch.setenv("AGENT_EXTRACTION", "1")
    out = agent_kb.kb_for_elements([EL, SPEC_EL, DATA_EL, "00000000"])
    assert set(out) == {EL, SPEC_EL, DATA_EL}, "an element with nothing extracted is absent"
    assert out[EL]["unit_count"] == 1 and out[EL]["block_count"] == 2
    assert out[EL]["units"][0]["import_line"].endswith("import calculate_buffers")
    assert [b["order"] for b in out[EL]["blocks"]] == [0, 1]
    assert out[EL]["blocks"][1]["context"] == "Load the crimes."
    assert out[SPEC_EL]["publication"]["is_method_spec"] is True
    assert len(out[SPEC_EL]["publication"]["steps"]) == 2
    assert out[DATA_EL]["dataset"] == {"format": "CSV", "crs": "EPSG:4326", "row_count": 20000,
                                       "schema": ["ID", "Latitude", "Longitude"]}


def test_attach_folds_rows_on_the_record_too(db):
    res = agent_kb.attach_kb_to_documents([
        {"doc_id": EL, "source": "keyword"},
        {"doc_id": f"{EL}::block::1", "source": "agent_kb", "parent_doc_id": EL}])
    assert res["attached"] == 1 and res["folded"] == 1
    assert res["documents"][0]["extracted"]["block_count"] == 2


# ------------------------------------------------------------------ a database that fails

@pytest.fixture
def dead_db(monkeypatch):
    @contextmanager
    def _connect(*_a, **_kw):
        raise OSError("connection refused")
        yield  # pragma: no cover

    monkeypatch.setenv("AGENT_KB_DB", "1")
    monkeypatch.setenv("AGENT_KB_BACKEND", "opensearch")
    monkeypatch.setattr(kb_db, "connect", _connect)

    def _no_cluster():
        raise AssertionError("fell back to the cluster")

    monkeypatch.setattr(agent_kb, "_os_client", _no_cluster)


def test_a_failed_database_is_reported_by_all_three_readers_and_none_falls_back(dead_db, caplog):
    import logging

    search = agent_kb.agent_kb_search("buffers")
    assert search["count"] == 0 and search["backend"] == "postgres"
    assert "agent KB database failed" in search["note"]

    fetch = agent_kb.get_kb_block(f"{EL}::block::1")
    assert fetch["found"] is False and fetch["backend"] == "postgres"
    assert "agent KB database failed" in fetch["note"] and "connection refused" in fetch["note"]

    with caplog.at_level(logging.WARNING):
        assert agent_kb.kb_for_elements([EL]) == {}
    assert any("agent KB database failed" in r.getMessage() for r in caplog.records)


# ------------------------------------------------------------------ flag off: unchanged

def test_flag_off_fetch_uses_the_backend_flag(monkeypatch):
    """AGENT_KB_DB unset: the cluster, as before — the record is never opened."""
    def _no_db(*_a, **_kw):
        raise AssertionError("opened the record with AGENT_KB_DB unset")

    monkeypatch.delenv("AGENT_KB_DB", raising=False)
    monkeypatch.setenv("AGENT_KB_BACKEND", "opensearch")
    monkeypatch.setattr(kb_db, "connect", _no_db)

    class Client:
        def get(self, index, id):
            return {"found": True, "_source": {"doc_id": id, "title": "from the cluster"}}

    monkeypatch.setattr(agent_kb, "_os_client", lambda: Client())
    out = agent_kb.get_kb_block(f"{EL}::block::1")
    assert out["found"] is True and out["source"]["title"] == "from the cluster"
    assert "backend" not in out


def test_flag_off_join_uses_the_local_store(monkeypatch):
    from extractors import kb_store

    def _no_db(*_a, **_kw):
        raise AssertionError("opened the record with AGENT_KB_DB unset")

    monkeypatch.delenv("AGENT_KB_DB", raising=False)
    monkeypatch.delenv("AGENT_KB_BACKEND", raising=False)
    monkeypatch.setattr(kb_db, "connect", _no_db)
    monkeypatch.setattr(kb_store, "local_blocks_for_parent", lambda el, _i: [
        (f"{el}::block::0", {"doc_id": f"{el}::block::0", "extracted": {
            "kind": "notebook_block", "parent_doc_id": el, "order": 0, "block": {}}})])
    monkeypatch.setattr(kb_store, "local_get", lambda _el, _i: (None, None))
    assert agent_kb.kb_for_elements([EL])[EL]["block_count"] == 1


def test_an_injected_client_wins_over_the_flag(db):
    """The same precedence search uses: a caller that names a cluster gets that cluster."""
    class Client:
        def get(self, index, id):
            return {"found": True, "_source": {"doc_id": id, "title": "injected"}}

    out = agent_kb.get_kb_block(f"{EL}::block::1", client=Client())
    assert out["source"]["title"] == "injected" and "backend" not in out
    assert db.queries == []


# ------------------------------------------------------------------ against a real record

@pytest.fixture
def real_conn():
    if not kb_db.enabled():
        pytest.skip("AGENT_KB_DB not set")
    try:
        ctx = kb_db.connect()
        conn = ctx.__enter__()
    except Exception as exc:
        pytest.skip(f"no database at {kb_db.dsn()}: {type(exc).__name__}")
    yield conn
    ctx.__exit__(None, None, None)


@pytest.mark.integration
@pytest.mark.parametrize("query", ["buffer geometries by a distance",
                                   "enhanced two step floating catchment area accessibility",
                                   "spatial join count points in polygons"])
def test_every_id_search_returns_can_be_fetched(real_conn, query):
    """The round trip the bug broke, over whatever id kinds the record returns."""
    hits = kb_db.search_kb(real_conn, query, size=8)
    if not hits:
        pytest.skip("no hits for this query in this record")
    for hit in hits:
        got = kb_db.get_doc(real_conn, hit["_id"])
        assert got is not None, hit["_id"]
        # Search annotates a unit it collapsed with its same-signature siblings; that is a fact
        # about the result set, not about the stored unit.
        want = hit["_source"]
        unit = (want.get("extracted") or {}).get("unit")
        if unit:
            unit = {k: v for k, v in unit.items() if k not in ("also_defined_by", "shared_symbol")}
            want = {**want, "extracted": {**want["extracted"], "unit": unit}}
        assert got["_source"] == want, hit["_id"]
