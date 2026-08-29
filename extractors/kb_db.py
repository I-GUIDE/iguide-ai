"""PostgreSQL system of record for extracted content.

**Why a fourth store, when OpenSearch and Neo4j already exist.** Those two belong to the
website. The agent writes its extracted content into separate ``iguide_agent_*`` indices on the
platform's cluster, which means re-indexing the agent's own knowledge base needs write access to
production website infrastructure, and a mapping change is a re-crawl of the corpus — network
fetches plus an LLM call per publication — because nothing durable holds what extraction found.
The two file-backed stores that grew alongside them (``_registry.json`` and ``outputs/*.json``)
have no constraints at all.

So this is not another index. It is the record the indices are built FROM:

  * **Constraints the JSON registry could not have.** ``UNIQUE (element_id, source_rel_path,
    symbol, slice_sha)`` is the collision that silently dropped 58 of 362 callable units when the
    key was ``{package}.{symbol}`` and one element spanned 35 files. In a 2.5 MB file rewritten
    whole there is nowhere to put that rule; here it is a write error.
  * **Transactional re-ingest.** Delete-orphans, upsert and record-the-run commit together, so a
    half-written element is not representable.
  * **``slice_source`` lives here.** ``agent_runtime/method_library`` deliberately reads the
    on-disk registry rather than an index, because "an index doc and the mounted library drift
    independently, and the failure mode of that drift is the worst kind — the agent is told to
    import something that does not exist, inside a container with no network to check." Storing
    the slice itself keeps that guarantee while moving the record: the mounted library becomes a
    PROJECTION of this table rather than an independent writer, so there is nothing to drift.

Scale is not the argument and should not be used as one — the whole corpus is ~10k rows and
~100 MB. This buys correctness and joins, not throughput.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

DEFAULT_DSN = "postgresql://iguide:iguide_dev@127.0.0.1:5544/iguide_agent"

# Bumped when the schema below changes in a way that makes existing rows unreadable. Recorded on
# every ingest run so a backfill written against an older shape is identifiable rather than merely
# wrong.
SCHEMA_VERSION = 1


def dsn() -> str:
    """Connection string. Local-only default: the dev database is bound to 127.0.0.1."""
    return (os.getenv("AGENT_KB_DSN") or "").strip() or DEFAULT_DSN


def enabled() -> bool:
    """Whether the Postgres record is configured for this process.

    Off by default. The store is additive — every existing emit target keeps working untouched —
    so nothing should start depending on it implicitly.
    """
    return (os.getenv("AGENT_KB_DB") or "").strip().lower() in {"1", "true", "yes", "on"}


@contextmanager
def connect(dsn_override: Optional[str] = None) -> Iterator[Any]:
    """A connection with autocommit OFF, so callers get a transaction by default.

    Re-ingest is the reason: an element's rows are deleted and rewritten together, and a crash
    between those two must leave the previous version intact rather than an empty element.
    """
    import psycopg

    conn = psycopg.connect(dsn_override or dsn())
    try:
        yield conn
    finally:
        conn.close()


# --------------------------------------------------------------------------- schema

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS element (
    id            text PRIMARY KEY,
    title         text NOT NULL DEFAULT '',
    element_type  text NOT NULL DEFAULT '',
    tags          jsonb NOT NULL DEFAULT '[]'::jsonb,
    authors       jsonb NOT NULL DEFAULT '[]'::jsonb,
    source_url    text NOT NULL DEFAULT '',
    doi           text NOT NULL DEFAULT '',
    fields        jsonb NOT NULL DEFAULT '{}'::jsonb,
    updated_at    timestamptz NOT NULL DEFAULT now()
);

-- One row per (element, file, symbol, VERSION). The slice_sha is in the key on purpose: the
-- library is content-addressed, an edited function mints a new module, and old versions stay
-- importable by sha. Making the sha part of the identity is what lets both exist.
CREATE TABLE IF NOT EXISTS unit (
    element_id       text NOT NULL REFERENCES element(id) ON DELETE CASCADE,
    source_rel_path  text NOT NULL,
    symbol           text NOT NULL,
    slice_sha        text NOT NULL,
    qualified_name   text NOT NULL DEFAULT '',
    library_module   text NOT NULL DEFAULT '',
    signature        text NOT NULL DEFAULT '',
    returns          text NOT NULL DEFAULT '',
    return_kind      text NOT NULL DEFAULT '',
    unit_kind        text NOT NULL DEFAULT '',
    doc_summary      text NOT NULL DEFAULT '',
    docstring        text NOT NULL DEFAULT '',
    verdict          text NOT NULL DEFAULT '',
    extractor        text NOT NULL DEFAULT '',
    fast_path        boolean NOT NULL DEFAULT false,
    is_current       boolean NOT NULL DEFAULT true,
    callability      jsonb NOT NULL DEFAULT '{}'::jsonb,
    params           jsonb NOT NULL DEFAULT '[]'::jsonb,
    invariants       jsonb NOT NULL DEFAULT '[]'::jsonb,
    requirements     jsonb NOT NULL DEFAULT '{}'::jsonb,
    provenance       jsonb NOT NULL DEFAULT '{}'::jsonb,
    -- The slice itself. This is what makes the mounted library a projection rather than a
    -- second writer that can drift from the contract describing it.
    slice_source     text NOT NULL DEFAULT '',
    -- Identifier text pre-expanded by the caller: Postgres splits `load_crime_points` into
    -- load/crime/point but leaves `calculateBuffers` whole, and "buffer" has to find it.
    symbol_text      text NOT NULL DEFAULT '',
    element_text     text NOT NULL DEFAULT '',
    updated_at       timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (element_id, source_rel_path, symbol, slice_sha),
    -- A refused unit has no slice, and an empty `slice_sha` is how it says so. The check makes
    -- "we never ship code for a unit the analyzer refused" a property of the store rather than
    -- a convention every writer has to remember — the sandbox mounts what is in this column.
    CONSTRAINT only_callable_units_carry_code
        CHECK (verdict = 'callable' OR slice_source = ''),
    -- Every input is bounded. A tsvector caps at 1 MB and the corpus contains a markdown cell
    -- of 4.9 MB, 99.8% of it base64 image data, so an unbounded index expression is not a
    -- tuning question but a row that cannot be inserted at all. The COLUMN keeps everything;
    -- only what is searchable is bounded.
    search tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', left(coalesce(symbol_text, ''), 100000)), 'A') ||
        setweight(to_tsvector('english', left(coalesce(doc_summary, ''), 100000)), 'B') ||
        setweight(to_tsvector('english', left(coalesce(element_text, ''), 100000)), 'C') ||
        setweight(to_tsvector('english', left(coalesce(signature, ''), 100000)), 'D')
    ) STORED
);
CREATE INDEX IF NOT EXISTS unit_search_idx  ON unit USING gin (search);
CREATE INDEX IF NOT EXISTS unit_element_idx ON unit (element_id);
CREATE INDEX IF NOT EXISTS unit_verdict_idx ON unit (verdict) WHERE is_current;
CREATE INDEX IF NOT EXISTS unit_symbol_idx  ON unit (symbol);

CREATE TABLE IF NOT EXISTS block (
    doc_id      text PRIMARY KEY,
    element_id  text NOT NULL REFERENCES element(id) ON DELETE CASCADE,
    ord         integer,
    title       text NOT NULL DEFAULT '',
    code        text NOT NULL DEFAULT '',
    markdown    text NOT NULL DEFAULT '',
    parse_ok    boolean,
    tools       jsonb NOT NULL DEFAULT '[]'::jsonb,
    imports     jsonb NOT NULL DEFAULT '[]'::jsonb,
    file_refs   jsonb NOT NULL DEFAULT '[]'::jsonb,
    constructs  jsonb NOT NULL DEFAULT '[]'::jsonb,
    updated_at  timestamptz NOT NULL DEFAULT now(),
    search tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', left(coalesce(title, ''), 100000)), 'A') ||
        setweight(to_tsvector('english', left(coalesce(markdown, ''), 100000)), 'B') ||
        setweight(to_tsvector('english', left(coalesce(code, ''), 100000)), 'C')
    ) STORED
);
CREATE INDEX IF NOT EXISTS block_search_idx  ON block USING gin (search);
CREATE INDEX IF NOT EXISTS block_element_idx ON block (element_id, ord);

CREATE TABLE IF NOT EXISTS dataset_file (
    element_id     text NOT NULL REFERENCES element(id) ON DELETE CASCADE,
    file           text NOT NULL,
    bytes          bigint,
    format         text NOT NULL DEFAULT '',
    family         text NOT NULL DEFAULT '',
    row_count      bigint,
    crs            text NOT NULL DEFAULT '',
    geometry_type  text NOT NULL DEFAULT '',
    bounds         jsonb NOT NULL DEFAULT '[]'::jsonb,
    columns        jsonb NOT NULL DEFAULT '[]'::jsonb,
    variables      jsonb NOT NULL DEFAULT '[]'::jsonb,
    extracted      jsonb NOT NULL DEFAULT '{}'::jsonb,
    envelope       jsonb,
    loader         jsonb,
    stage          text NOT NULL DEFAULT '',
    error          text NOT NULL DEFAULT '',
    updated_at     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (element_id, file),
    search tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', left(coalesce(file, ''), 100000)), 'A') ||
        setweight(to_tsvector('english',
            left(coalesce(format || ' ' || family || ' ' || crs, ''), 100000)), 'C')
    ) STORED
);
CREATE INDEX IF NOT EXISTS dataset_search_idx ON dataset_file USING gin (search);
CREATE INDEX IF NOT EXISTS dataset_crs_idx    ON dataset_file (crs);

-- Per-element dataset result, separate from `dataset_file` because most dataset elements never
-- produce a local file: 102 of 145 resolve to a portal page, a login wall or a listing. "Why
-- there is no file" is the extraction finding for those, and dropping it would leave the record
-- claiming the corpus has 43 datasets when it has 145 with 43 readable.
CREATE TABLE IF NOT EXISTS dataset_outcome (
    element_id    text PRIMARY KEY REFERENCES element(id) ON DELETE CASCADE,
    stage         text NOT NULL DEFAULT '',
    error         text NOT NULL DEFAULT '',
    link_kind     text NOT NULL DEFAULT '',
    note          text NOT NULL DEFAULT '',
    primary_file  text NOT NULL DEFAULT '',
    files_listed  integer,
    members       integer,
    listed_bytes  bigint,
    detail        jsonb NOT NULL DEFAULT '{}'::jsonb,
    updated_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS dataset_outcome_stage_idx ON dataset_outcome (stage);

CREATE TABLE IF NOT EXISTS publication (
    element_id           text PRIMARY KEY REFERENCES element(id) ON DELETE CASCADE,
    doi                  text NOT NULL DEFAULT '',
    licence              text NOT NULL DEFAULT '',
    outcome              text NOT NULL DEFAULT '',
    reason               text NOT NULL DEFAULT '',
    status               text NOT NULL DEFAULT '',
    summary              text NOT NULL DEFAULT '',
    steps                jsonb NOT NULL DEFAULT '[]'::jsonb,
    datasets_referenced  jsonb NOT NULL DEFAULT '[]'::jsonb,
    tools_referenced     jsonb NOT NULL DEFAULT '[]'::jsonb,
    declared_params      jsonb NOT NULL DEFAULT '{}'::jsonb,
    chars                bigint,
    sha256               text NOT NULL DEFAULT '',
    updated_at           timestamptz NOT NULL DEFAULT now(),
    search tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', left(coalesce(summary, ''), 100000)), 'B') ||
        setweight(to_tsvector('english', left(coalesce(steps::text, ''), 100000)), 'C')
    ) STORED
);
CREATE INDEX IF NOT EXISTS publication_search_idx ON publication USING gin (search);

-- Skip-if-unchanged, moved off the OpenSearch `ingest_runs` index so the decision to re-extract
-- commits in the same transaction as the rows it describes.
CREATE TABLE IF NOT EXISTS ingest_run (
    element_id      text PRIMARY KEY,
    fingerprint     text NOT NULL,
    schema_version  integer NOT NULL,
    doc_count       integer,
    summary         jsonb NOT NULL DEFAULT '{}'::jsonb,
    at              timestamptz NOT NULL DEFAULT now()
);
"""


def ensure_schema(conn) -> None:
    """Create every table and index if absent. Idempotent; safe to call on each ingest."""
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    conn.commit()


def expand_identifiers(text: str) -> str:
    """Identifier text with camelCase and snake_case boundaries made into separate words.

    Postgres's text-search parser splits ``load_crime_points`` into load/crime/point but leaves
    ``calculateBuffers`` as one token, so "buffer" would not find it. The split rule has exactly
    one definition in this repo — ``method_library._tokens``, which documents why it exists — and
    this reuses it rather than restating it, because two copies of a tokenizer diverge silently
    and the symptom is a method that cannot be found.
    """
    from agent_runtime.method_library import _tokens

    return " ".join(_tokens(text or ""))


def table_counts(conn) -> Dict[str, int]:
    """Row counts per table — the cheapest honest answer to "did the backfill land"."""
    out: Dict[str, int] = {}
    with conn.cursor() as cur:
        for table in ("element", "unit", "block", "dataset_file", "dataset_outcome",
                      "publication", "ingest_run"):
            cur.execute(f"SELECT count(*) FROM {table}")
            out[table] = int(cur.fetchone()[0])
    return out


# --------------------------------------------------------------------------- reads

# Mirrors `method_library._FIELD_WEIGHTS` (symbol 4.0, qualified/summary 2.0, element 1.0,
# signature 0.5) onto the four tsvector labels, normalised the way ts_rank expects them:
# {D, C, B, A}. Keeping the same relative ordering is what makes the two rankings comparable
# rather than merely different.
TS_WEIGHTS = "{0.125, 0.25, 0.5, 1.0}"

# ANY of the query's words, not all of them.
#
# `websearch_to_tsquery` and `plainto_tsquery` both AND their terms, so "buffer geometries by a
# distance" required one unit to match buffer AND geometry AND distance and returned NOTHING
# against all 840. The ranker being replaced sums the weights of whatever matched, so OR is both
# the useful behaviour and the one that makes the two rankings comparable. Building the query
# from `tsvector_to_array` keeps the same stemming and stop-word list as the indexed side —
# an analyser mismatch between index time and query time is the classic way a search quietly
# stops finding things.
_OR_TSQUERY = """
    to_tsquery('english', nullif(array_to_string(
        tsvector_to_array(to_tsvector('english', %s)), ' | '), ''))
"""

_UNIT_SELECT = """
    SELECT u.element_id, u.symbol, u.qualified_name, u.library_module, u.slice_sha,
           u.signature, u.doc_summary, u.unit_kind, u.verdict, u.requirements, u.invariants,
           e.title AS element_title,
           ts_rank(%s::float4[], u.search, q) AS score
      FROM unit u
      JOIN element e ON e.id = u.element_id,
           LATERAL (SELECT """ + _OR_TSQUERY.strip() + """ AS q) t
     WHERE u.search @@ q
"""


def _unit_row(row) -> Dict[str, Any]:
    """The exact shape `method_library._summarize` returns, so this is a drop-in.

    Including `import_line` pinned to the `v_<sha>` module: an agent that gets a different shape
    from a different backend has to learn two contracts, and the import line is the one field a
    wrong answer breaks a sandboxed run over.
    """
    (element_id, symbol, qualified, module, sha, signature, summary, unit_kind, verdict,
     requirements, invariants, element_title, score) = row
    checks = [f"{i.get('check')}({i.get('target')})"
              for i in (invariants or []) if isinstance(i, dict)]
    return {
        "symbol": qualified or symbol,
        "unit_kind": unit_kind,
        "signature": signature,
        "doc_summary": summary,
        "import_line": f"from {module} import {symbol}" if module and symbol else None,
        "element_id": element_id,
        "element_title": element_title,
        "slice_sha": sha,
        "requirements": (requirements or {}).get("pip") or [],
        "requires": checks or None,
        "score": round(float(score), 4),
    }


def search_units(conn, query: str, *, limit: int = 8,
                 callable_only: bool = True) -> List[Dict[str, Any]]:
    """Rank units against a natural-language query using Postgres full text search.

    `websearch_to_tsquery` rather than `plainto_tsquery`: it accepts quoted phrases and `-word`
    negation, which is what a person types, and it never raises on punctuation the way
    `to_tsquery` does.

    The query text is expanded the same way the indexed identifiers were, so "calculateBuffers"
    typed as one word still matches — an asymmetry between index-time and query-time analysis is
    the classic way a search silently stops finding things.
    """
    sql = _UNIT_SELECT
    params: List[Any] = [TS_WEIGHTS, f"{query} {expand_identifiers(query)}"]
    if callable_only:
        sql += " AND u.verdict = 'callable'"
    # Over-fetch, because collapsing same-symbol duplicates below can otherwise return fewer
    # results than asked for.
    sql += " ORDER BY score DESC, u.symbol LIMIT %s"
    params.append(max(1, int(limit)) * 3)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = [_unit_row(r) for r in cur.fetchall()]

    # One entry per (symbol, signature), carrying the elements that share it.
    #
    # Two identical rows spend two slots to say one thing — but simply dropping the second
    # ERASES a real ambiguity, and that is worse. Asked for "the exact import line" for
    # `spatial_join_and_count`, which two elements define identically, the agent returned one
    # and did not mention the other; it had been shown both, adjacent and distinguishable only
    # by a hash inside a module path, and took the first. `get_contract` refuses a bare
    # ambiguous name for exactly this reason and was never consulted, because the search result
    # already looked like an answer.
    collapsed: Dict[tuple, Dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("symbol") or "").rsplit(".", 1)[-1], row.get("signature"))
        first = collapsed.get(key)
        if first is None:
            collapsed[key] = row
            continue
        siblings = first.setdefault("also_defined_by", [])
        rows_seen = first.setdefault("also_defined_by_rows", [dict(first)])
        entry = {"element_id": row.get("element_id"),
                 "element_title": row.get("element_title"),
                 "import_line": row.get("import_line")}
        if entry not in siblings:
            siblings.append(entry)
            rows_seen.append(row)
        # The import line is the field a wrong choice breaks a run over, so an ambiguous row
        # must not present one as if it were THE answer.
        # NOT `ambiguous`. That key already means "this is an ambiguity stub with no
        # signature and no import line", and `_method_units_as_documents` filters on it — so
        # setting it here deleted a real, callable, importable method from the evidence and the
        # agent reported having no import line for it at all.
        first["shared_symbol"] = True
        # The SAME refusal `get_unit_contract` makes, applied at first contact instead of only
        # at the second tool call.
        #
        # Carrying the fact was not enough. Given `shared_symbol` and `also_defined_by` in the
        # payload, the agent still answered "here's the exact import line" for one element and
        # never mentioned the other — it had the fact and did not act on it. `get_contract`
        # refuses a bare ambiguous name for exactly this case and was never consulted, because
        # the search result already looked like an answer. So the search result stops looking
        # like one: no bare import line for a name that does not identify code, and the
        # candidates plus the resolution in its place. This repo's own rule — a capability the
        # model must elect is a capability you do not have; add a check, not an instruction.
        first["import_line"] = None
        first["import_line_candidates"] = [
            {"qualified_name": row.get("symbol"), "element_id": row.get("element_id"),
             "element_title": row.get("element_title"),
             "import_line": row.get("import_line")}
            for row in (first.get("also_defined_by_rows") or [])]
        first["disambiguate_with"] = "get_method_contract(<qualified_name>)"
        # The same `error` key and the same sentence `get_unit_contract` uses, because an
        # ABSENCE carries no meaning. Withholding the import line stopped the agent handing
        # over the wrong one — the safety property held — but it then reported "the module
        # path isn't present in what was retrieved", reading a deliberate refusal as missing
        # data. That is the third time in this codebase that an absent field has been read as
        # a negative: a missing callability verdict meant "not callable", a missing bbox meant
        # "no geometry", and now a missing import line means "we don't have it".
        first["error"] = (f"{first.get('symbol', '').rsplit('.', 1)[-1]!r} is defined by more "
                          f"than one element; ask again with a qualified name.")
    for row in collapsed.values():
        row.pop("also_defined_by_rows", None)          # scaffolding, not payload
    return list(collapsed.values())[:max(1, int(limit))]


def get_unit_contract(conn, symbol: str) -> Dict[str, Any]:
    """Full contract for one symbol, by bare or qualified name.

    An ambiguous bare name returns the candidates and NO import line — the same refusal the
    registry reader makes, for the same reason: guessing which of two same-named units the
    caller meant produces a run that imports the wrong code and reports success.
    """
    bare = symbol.rsplit(".", 1)[-1]
    with conn.cursor() as cur:
        cur.execute("""
            SELECT u.element_id, u.symbol, u.qualified_name, u.library_module, u.slice_sha,
                   u.signature, u.doc_summary, u.unit_kind, u.verdict, u.requirements,
                   u.invariants, e.title, u.params, u.returns, u.docstring, u.callability,
                   u.source_rel_path, u.extractor
              FROM unit u JOIN element e ON e.id = u.element_id
             WHERE u.is_current AND (u.symbol = %s OR u.qualified_name = %s)
             ORDER BY u.symbol
        """, (bare, symbol))
        rows = cur.fetchall()
    if not rows:
        # `found: False` distinguishes "this store has no such symbol" from "this store refuses
        # to guess between two". Only the first should make a caller try another backend; the
        # second is an answer, and falling through it would turn a correct refusal into a
        # confidently wrong import line.
        return {"symbol": symbol, "found": False,
                "error": f"no method named {symbol!r} in the library"}
    if len(rows) > 1 and symbol == bare:
        # The same refusal text the registry reader gives, not merely the same refusal. The
        # model has to be told what to do next — ask again with a qualified name — and a bare
        # `ambiguous: true` leaves it to guess, which is the failure this branch exists to
        # prevent. Two backends that refuse differently are two behaviours to learn.
        return {"symbol": symbol, "ambiguous": True,
                "candidates": [{"qualified_name": r[2], "element_id": r[0],
                                "source_rel_path": r[16], "signature": r[5]} for r in rows],
                "import_line": None,
                "error": f"{symbol!r} is defined by more than one element; "
                         f"ask again with a qualified name."}
    r = rows[0]
    out = _unit_row(tuple(r[:12]) + (0.0,))
    # `invariants` and the full `requirements` dict, not only the `requires` summary line that
    # `_unit_row` derives. The registry contract carries both, and the difference is not
    # cosmetic: invariants are where "requires a projected CRS" lives, and a contract that
    # silently drops them tells the caller nothing about the metric discipline it must keep.
    # Caught by diffing the two backends field by field rather than eyeballing one.
    out.update({"params": r[12], "returns": r[13], "docstring": r[14], "callability": r[15],
                "source_rel_path": r[16], "verdict": r[8],
                "invariants": r[10] or [], "requirements": r[9] or {},
                "module": r[3], "provenance": {"element_id": r[0], "source_rel_path": r[16],
                                               "extractor": r[17]}})
    out.pop("score", None)
    return out


def slice_source(conn, element_id: str, symbol: str, slice_sha: str) -> Optional[str]:
    """The exact slice a contract describes.

    This is why the mounted library can be a projection: the bytes the sandbox imports are
    reproducible from the row that describes them, so the two cannot disagree.
    """
    with conn.cursor() as cur:
        cur.execute("""SELECT slice_source FROM unit
                        WHERE element_id = %s AND symbol = %s AND slice_sha = %s""",
                    (element_id, symbol, slice_sha))
        row = cur.fetchone()
    return row[0] if row else None


_BLOCK_TSQUERY = _OR_TSQUERY

# A spec is already the distilled form — 31 numbered steps, not 31 cells of code — so it is
# carried WHOLE, and the budget is set from the corpus rather than guessed: 63 specs, median
# 3,060 characters, maximum 5,615. A first guess of 2,600 truncated the E2SFCA spec at step 12,
# one step before "Apply distance-decay weights (1, 0.68, 0.22)" — losing the exact parameters
# that were the reason to retrieve it. Truncating a compression discards information that has
# no cheaper representation; 6,000 clears every spec in the corpus and still bounds a runaway.
_SPEC_CHARS = 6000


def search_kb(conn, query: str, *, size: int = 8) -> List[Dict[str, Any]]:
    """Blocks and units, ranked together, shaped as OpenSearch hits.

    Returning the cluster's hit shape rather than a shape of its own is deliberate:
    `rag_pipeline.search.agent_kb.normalize_hits` already turns a hit into the evidence
    document the agent reads, including the method payload and the parent-element link. A
    second normaliser here would be a second place for those to drift, and the symptom of that
    drift is a contract field quietly missing from the evidence view — which has already
    happened once in this chain, at every link.

    Blocks and units compete in one ranking because the agent's question ("how do I do X") is
    answered by either: a cell that did it, or a unit that can be called to do it.
    """
    if not (query or "").strip():
        return []
    expanded = f"{query} {expand_identifiers(query)}"
    hits: List[Dict[str, Any]] = []
    with conn.cursor() as cur:
        cur.execute("""
            SELECT b.doc_id, b.element_id, b.title, b.markdown, b.code, b.tools, b.ord,
                   e.title AS element_title, e.element_type,
                   ts_rank('{0.1,0.2,0.4,1.0}'::float4[], b.search, q) AS score
              FROM block b JOIN element e ON e.id = b.element_id,
                   LATERAL (SELECT """ + _OR_TSQUERY.strip() + """ AS q) t
             WHERE b.search @@ q
             ORDER BY score DESC, b.doc_id
             LIMIT %s
        """, (expanded, size))
        for (doc_id, element_id, title, markdown, code, tools, ord_, el_title,
             el_type, score) in cur.fetchall():
            contents = f"{markdown}\n\n{code}".strip() if markdown else (code or "")
            hits.append({
                "_id": doc_id, "_index": "pg:block", "_score": float(score),
                "_source": {
                    "doc_id": doc_id, "title": title or el_title or doc_id,
                    "contents": contents, "resource-type": "NotebookBlock",
                    "extracted": {"parent_doc_id": element_id, "parent_title": el_title,
                                  "parent_type": el_type, "order": ord_,
                                  "block": {"resolved_tools": tools or [], "code": code}},
                }})

        cur.execute("""
            SELECT u.element_id, u.symbol, u.qualified_name, u.library_module, u.slice_sha,
                   u.signature, u.doc_summary, u.unit_kind, u.verdict, u.requirements,
                   u.invariants, u.params, u.returns, u.callability, e.title,
                   ts_rank('{0.125,0.25,0.5,1.0}'::float4[], u.search, q) AS score
              FROM unit u JOIN element e ON e.id = u.element_id,
                   LATERAL (SELECT """ + _OR_TSQUERY.strip() + """ AS q) t
             WHERE u.search @@ q
             ORDER BY score DESC, u.symbol
             LIMIT %s
        """, (expanded, size))
        for (element_id, symbol, qualified, module, sha, signature, summary, unit_kind,
             verdict, requirements, invariants, params, returns, callability, el_title,
             score) in cur.fetchall():
            hits.append({
                "_id": f"{element_id}::unit::{symbol}::{sha}", "_index": "pg:unit",
                "_score": float(score),
                "_source": {
                    "doc_id": f"{element_id}::unit::{symbol}::{sha}",
                    "title": symbol, "contents": f"{signature}\n{summary}".strip(),
                    "resource-type": "MethodUnit",
                    "extracted": {"parent_doc_id": element_id, "parent_title": el_title,
                                  "unit": {
                                      "library_symbol": symbol, "qualified_name": qualified,
                                      "signature": signature, "doc_summary": summary,
                                      "unit_kind": unit_kind, "params": params or [],
                                      "returns": returns, "invariants": invariants or [],
                                      "requirements": requirements or {},
                                      # From the COLUMN, always. The registry never persisted
                                      # callability for the units it ships — "it is in the
                                      # library, therefore callable" — so all 840 callable units
                                      # carry `{}` here while all 149 refused ones carry a full
                                      # dict. A consumer reading the JSON saw the negative fact
                                      # and never the positive one.
                                      "callability": {**(callability or {}),
                                                      "verdict": verdict},
                                      "slice_sha": sha,
                                      # Pinned to the v_<sha> module, same as everywhere else:
                                      # an evidence view that names a method without saying how
                                      # to import it makes the agent guess the one field a
                                      # sandboxed run cannot recover from.
                                      "import_line": (f"from {module} import {symbol}"
                                                      if module and symbol else None),
                                  }},
                }})
        # --- publications: the method as the literature states it ------------------------
        #
        # A third question, not a variant of the other two. A cell says how someone did it, a
        # unit says what can be called, and a spec says what the method IS — including the
        # parameters a caller has to supply and that no signature can carry: the paper's own
        # catchment bands and decay weights are what turn `e2sfca(..., distances, weights)`
        # from a signature into a runnable call.
        cur.execute("""
            SELECT p.element_id, e.title, p.summary, p.steps, p.tools_referenced,
                   p.datasets_referenced, p.doi,
                   ts_rank('{0.1,0.2,0.4,1.0}'::float4[], p.search, q) AS score
              FROM publication p JOIN element e ON e.id = p.element_id,
                   LATERAL (SELECT """ + _OR_TSQUERY.strip() + """ AS q) t
             WHERE p.search @@ q AND jsonb_array_length(p.steps) > 0
             ORDER BY score DESC, p.element_id
             LIMIT %s
        """, (expanded, size))
        for (element_id, title, summary, steps, tools, datasets, doi, score) in cur.fetchall():
            # The steps ARE the payload and they are already the distilled form, so they get a
            # larger budget than a raw cell excerpt. Numbered, because the agent has to be able
            # to say which step it is following.
            body = [summary.strip()] if summary else []
            body += [f"{i}. {st}" for i, st in enumerate(steps or [], start=1)]
            contents = "\n".join(body)
            hits.append({
                "_id": f"{element_id}::methodspec", "_index": "pg:publication",
                "_score": float(score),
                "_source": {
                    "doc_id": f"{element_id}::methodspec", "title": title or element_id,
                    "contents": contents[:_SPEC_CHARS],
                    "resource-type": "PublicationMethodSpec",
                    "extracted": {"parent_doc_id": element_id, "parent_title": title,
                                  "parent_type": "publication", "doi": doi,
                                  "spec": {"step_count": len(steps or []),
                                           "tools_referenced": tools or [],
                                           "datasets_referenced": datasets or []}},
                }})

    # Cells and units answer DIFFERENT questions — "how did someone do this" and "what can I
    # call" — and ranking them on one text-similarity axis lets cells win, because a cell indexes
    # its whole markdown and code while a unit indexes a signature and one summary line.
    # Measured over six queries: 24 of 48 hits actionable, and "count how many points fall in
    # each polygon" returned four cells and ONE distinct callable method.
    #
    # Length normalisation is not the fix — tested, and it makes it worse: ts_rank flags 2 and 8
    # promote 22-character cells ("view point", "view polygon") and push units out entirely. The
    # fix is a quota, because the two channels are not competing for the same slot.
    units = [h for h in hits if h["_index"] == "pg:unit"]
    blocks = [h for h in hits if h["_index"] not in ("pg:unit", "pg:publication")]
    specs = [h for h in hits if h["_index"] == "pg:publication"]

    # A REFUSED unit cannot be called, so it must never displace one that can. It is still
    # offered when nothing callable matched, because "there is a method but it reads a
    # module-level frame" is a better answer than "no method" — the payload labels it
    # `not_callable` with the reason.
    def _verdict(hit):
        # Reads the verdict the row was built with, which is now always present because
        # `search_kb` writes the column into the payload. Before that it read a key that was
        # absent for every callable unit, so this classified all 840 of them as refused.
        unit = ((hit["_source"].get("extracted") or {}).get("unit") or {})
        return (unit.get("callability") or {}).get("verdict") or ""

    callable_units = [h for h in units if _verdict(h) == "callable"]
    refused_units = [h for h in units if _verdict(h) != "callable"]

    # Same symbol from two elements is a real ambiguity the caller must resolve, but two
    # identical rows spend two slots to say one thing. Keep the best-scoring of each signature.
    seen: Dict[tuple, Dict[str, Any]] = {}
    deduped = []
    for hit in callable_units:
        unit = (hit["_source"].get("extracted") or {}).get("unit") or {}
        key = (unit.get("library_symbol"), unit.get("signature"))
        if key in seen:
            # Collapse the row, keep the fact. See `search_units` for what erasing it cost.
            first = seen[key]
            first.setdefault("also_defined_by", []).append(
                {"element_id": (hit["_source"].get("extracted") or {}).get("parent_doc_id"),
                 "import_line": unit.get("import_line")})
            first["shared_symbol"] = True
            continue
        seen[key] = unit
        deduped.append(hit)

    # A FLOOR, not a quota. The first version took exactly `size // 2` units and filled the rest
    # with cells, which capped units as well as guaranteeing them: a query legitimately matching
    # eight methods lost four of them to cells. It also pinned the metric it was meant to move —
    # "actionable fraction" is exactly 50% for every query, by construction.
    #
    # So: rank normally, then promote units until they hold at least half the slots. Units are
    # never capped, and a query with nothing callable is unchanged.
    ranked = sorted(deduped + refused_units + specs + blocks,
                    key=lambda h: -h["_score"])[:size]
    floor = max(1, size // 2)
    if sum(1 for h in ranked if h["_index"] == "pg:unit") < floor:
        promoted = (deduped or refused_units)[:floor]
        rest = [h for h in ranked if h not in promoted]
        ranked = (promoted + rest)[:size]

    # ONE spec, when one matched. A floor of one and never a cap, for the same reason units get
    # one: a spec answers a question the other two channels cannot, and it lost every slot on
    # raw score — "enhanced two step floating catchment area accessibility" returned four units
    # and two cells from the implementing notebook while the paper that DEFINES the method sat
    # ninth. One slot, because a spec is dense (up to 6,000 characters) and a second adds little.
    if specs and not any(h["_index"] == "pg:publication" for h in ranked):
        rest = [h for h in ranked if h is not ranked[-1]]
        ranked = [specs[0]] + rest
        ranked.sort(key=lambda h: -h["_score"])
    ranked.sort(key=lambda h: -h["_score"])
    return ranked


def parent_elements(conn, element_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Title/type/tags for the elements a result set cites, for the evidence view's header."""
    ids = sorted({str(e) for e in element_ids if e})
    if not ids:
        return {}
    with conn.cursor() as cur:
        cur.execute("""SELECT id, title, element_type, source_url, doi
                         FROM element WHERE id = ANY(%s)""", (ids,))
        return {r[0]: {"id": r[0], "title": r[1], "resource-type": r[2],
                       "source_url": r[3], "doi": r[4]} for r in cur.fetchall()}


__all__ = ["DEFAULT_DSN", "SCHEMA_VERSION", "SCHEMA_SQL", "TS_WEIGHTS", "dsn", "enabled",
           "connect", "ensure_schema", "expand_identifiers", "table_counts",
           "search_units", "get_unit_contract", "slice_source", "search_kb",
           "parent_elements"]
