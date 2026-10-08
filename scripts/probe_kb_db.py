#!/usr/bin/env python3
"""What the record can answer that the files it was loaded from cannot.

    python scripts/probe_kb_db.py

Ranking is not the interesting difference — `compare_method_search.py` measures that and finds it
close. These are the three properties a system of record has and a pile of JSON does not:

  1. **Fidelity.** Does a slice come back out byte-identical, and does it still compile? If not,
     the mounted library cannot be a projection of this table and the whole argument collapses.
  2. **Constraints.** The collision that silently dropped 58 of 362 callable units — is it now a
     write error rather than a discovery?
  3. **Joins.** Questions that span extraction types. Each of these is a single statement here
     and a bespoke script over four JSON files otherwise, which is why none of them has ever
     been asked.

Read-only except for one deliberately-failing INSERT inside a rolled-back savepoint.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from extractors import kb_db  # noqa: E402

LIBRARY = REPO / "agent_chat_files" / "method_library" / "iguide_methods"


def check_fidelity(conn) -> None:
    print("1. FIDELITY — is the slice recoverable from the row that describes it?")
    with conn.cursor() as cur:
        cur.execute("""SELECT element_id, symbol, slice_sha, library_module, slice_source
                         FROM unit WHERE slice_source <> '' ORDER BY element_id, symbol""")
        rows = cur.fetchall()
    identical = compiles = defines = 0
    missing_on_disk = 0
    for element_id, symbol, sha, module, source in rows:
        package = module.split(".")[1] if module.count(".") >= 1 else ""
        on_disk = LIBRARY / package / f"v_{sha}.py"
        try:
            if on_disk.read_text(encoding="utf-8") == source:
                identical += 1
        except OSError:
            missing_on_disk += 1
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        compiles += 1
        if any(getattr(n, "name", None) == symbol for n in tree.body):
            defines += 1
    total = len(rows)
    print(f"   slices stored                      {total}")
    print(f"   byte-identical to the mounted file {identical}/{total}")
    print(f"   parse as Python                    {compiles}/{total}")
    print(f"   define the symbol they claim       {defines}/{total}")
    if missing_on_disk:
        print(f"   not on disk to compare against     {missing_on_disk}")
    print("   -> the library can be rebuilt from this table, so the two cannot drift\n"
          if identical == total and defines == total else
          "   -> NOT a faithful projection; the drift argument does not hold\n")


def check_constraint(conn) -> None:
    print("2. CONSTRAINTS — is the 58-unit collision now a write error?")
    with conn.cursor() as cur:
        cur.execute("""SELECT element_id, source_rel_path, symbol, slice_sha
                         FROM unit ORDER BY element_id LIMIT 1""")
        element_id, source_rel, symbol, sha = cur.fetchone()

        # Two units of the same name from DIFFERENT files in one element: the exact shape that
        # the flat `{package}.{symbol}` key could not represent. It must be accepted.
        cur.execute("SAVEPOINT probe")
        cur.execute("""INSERT INTO unit (element_id, source_rel_path, symbol, slice_sha)
                       VALUES (%s, %s, %s, %s)""",
                    (element_id, source_rel + ".other", symbol, sha))
        cur.execute("SELECT count(*) FROM unit WHERE element_id = %s AND symbol = %s",
                    (element_id, symbol))
        both = cur.fetchone()[0]
        cur.execute("ROLLBACK TO SAVEPOINT probe")
        print(f"   same symbol, two source files      accepted ({both} rows coexist)")

        # The same unit twice is the write that must fail.
        cur.execute("SAVEPOINT probe2")
        try:
            cur.execute("""INSERT INTO unit (element_id, source_rel_path, symbol, slice_sha)
                           VALUES (%s, %s, %s, %s)""", (element_id, source_rel, symbol, sha))
            print("   duplicate of an existing unit      ACCEPTED — the constraint is missing")
        except Exception as exc:
            print(f"   duplicate of an existing unit      rejected: "
                  f"{type(exc).__name__}")
        cur.execute("ROLLBACK TO SAVEPOINT probe2")

        # A unit whose element does not exist is the orphan the JSON stores allowed.
        cur.execute("SAVEPOINT probe3")
        try:
            cur.execute("""INSERT INTO unit (element_id, source_rel_path, symbol, slice_sha)
                           VALUES ('nosuchel', 'x.py', 'f', 'deadbeef')""")
            print("   unit with no parent element        ACCEPTED — no referential integrity")
        except Exception as exc:
            print(f"   unit with no parent element        rejected: {type(exc).__name__}")
        cur.execute("ROLLBACK TO SAVEPOINT probe3")
    conn.rollback()
    print()


QUERIES = [
    ("Callable methods, by the extractor that produced them", """
        SELECT extractor, count(*) FROM unit WHERE verdict = 'callable'
         GROUP BY extractor ORDER BY 2 DESC"""),
    ("Why units were refused — the extraction limit to lift next", """
        SELECT verdict, count(*) FROM unit GROUP BY verdict ORDER BY 2 DESC"""),
    ("Elements that ship a method AND a readable dataset (the runnable pairs)", """
        SELECT e.id, left(e.title, 46), count(DISTINCT u.symbol) AS methods,
               count(DISTINCT d.file) AS files
          FROM element e
          JOIN unit u ON u.element_id = e.id AND u.verdict = 'callable'
          JOIN dataset_file d ON d.element_id = e.id AND d.crs <> ''
         GROUP BY e.id, e.title ORDER BY methods DESC LIMIT 5"""),
    ("Publications describing a method, whose element also ships callable code", """
        SELECT p.element_id, left(e.title, 40), jsonb_array_length(p.steps) AS steps,
               count(u.symbol) AS units
          FROM publication p JOIN element e ON e.id = p.element_id
          LEFT JOIN unit u ON u.element_id = p.element_id AND u.verdict = 'callable'
         WHERE jsonb_array_length(p.steps) > 0
         GROUP BY p.element_id, e.title, p.steps ORDER BY units DESC, steps DESC LIMIT 5"""),
    ("Most-required pip packages across every callable unit", """
        SELECT pkg, count(*) FROM unit,
               LATERAL jsonb_array_elements_text(coalesce(requirements->'pip', '[]'::jsonb)) pkg
         WHERE verdict = 'callable' GROUP BY pkg ORDER BY 2 DESC LIMIT 8"""),
    ("Units that declare a CRS invariant — the metric-discipline surface", """
        SELECT count(*) FILTER (WHERE inv->>'check' LIKE '%%crs%%') AS crs_checks,
               count(DISTINCT element_id) AS elements
          FROM unit, LATERAL jsonb_array_elements(invariants) inv"""),
    ("Datasets that never produced a local file, by reason", """
        SELECT stage, count(*) FROM dataset_outcome GROUP BY stage ORDER BY 2 DESC LIMIT 6"""),
    ("Notebook cells mentioning a CRS reprojection, and their elements", """
        SELECT count(*) AS cells, count(DISTINCT element_id) AS notebooks
          FROM block WHERE code ILIKE '%%to_crs%%'"""),
    ("Coverage: elements holding each combination of extracted content", """
        SELECT (EXISTS (SELECT 1 FROM block b WHERE b.element_id = e.id)) AS has_cells,
               (EXISTS (SELECT 1 FROM unit u WHERE u.element_id = e.id
                        AND u.verdict = 'callable')) AS has_methods,
               (EXISTS (SELECT 1 FROM publication p WHERE p.element_id = e.id)) AS has_paper,
               count(*)
          FROM element e GROUP BY 1, 2, 3 ORDER BY 4 DESC"""),
]


def run_queries(conn) -> None:
    print("3. JOINS — questions that span extraction types\n")
    for title, sql in QUERIES:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
        print(f"   {title}")
        for row in rows:
            cells = ["-" if v is None else str(v) for v in row]
            print("      " + "   ".join(f"{c[:48]:<{min(len(c), 48)}}" for c in cells))
        print()


def main() -> int:
    with kb_db.connect() as conn:
        counts = kb_db.table_counts(conn)
        print("rows: " + ", ".join(f"{v} {k}" for k, v in counts.items()) + "\n")
        check_fidelity(conn)
        check_constraint(conn)
        run_queries(conn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
