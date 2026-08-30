#!/usr/bin/env python3
"""Draw IMPLEMENTED_BY links from a publication's method spec to the element implementing it.

    python scripts/build_spec_links.py
    python scripts/build_spec_links.py --dry-run

The symbol-matching route in `publication_extractor` produces ONE edge across the whole corpus,
and its ceiling is structural: `tools_referenced` records what software a paper used — "OSMnx",
"Python", "Census Bureau API" — while a method library is indexed by function. Papers name
libraries; libraries contain functions.

So the link is drawn at the element level, from two kinds of evidence a human left behind:

  cited_doi                 the element's own text names the paper's DOI. Somebody wrote that
                            down deliberately, so it needs no corroboration.
  shared_author_and_topic   the paper and the element share a person AND the element's text
                            scores against the spec. Neither half works alone — author overlap
                            by itself gave one spec seventeen candidates including
                            "CyberGIS-Compute Core", which is a co-authorship graph, not an
                            implementation link.

The topical score is Postgres full-text rank over the candidate's own cells and units, using the
same OR-tsquery the agent's KB search uses, so retrieval and linking agree about what "matches"
means. Author lists come from the public element listing in one request; nothing here writes to
the platform.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Dict

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from extractors import kb_db  # noqa: E402
from extractors.analysis import spec_links  # noqa: E402

BACKEND = "https://backend.i-guide.io"


def platform_authors() -> Dict[str, Dict]:
    """Short element id -> title, type and author keys, from the public listing.

    `size`, not `limit`: `limit` is accepted and ignored, returning the default page of ten.
    """
    import requests

    resp = requests.get(f"{BACKEND}/api/elements", params={"size": 2000}, timeout=90)
    resp.raise_for_status()
    out = {}
    for element in (resp.json() or {}).get("elements") or []:
        short = str(element.get("id") or "")[:8]
        if not short:
            continue
        out[short] = {
            "title": element.get("title") or "",
            "type": element.get("resource-type") or "",
            "authors": [spec_links.author_key(a) for a in (element.get("authors") or [])
                        if spec_links.author_key(a)],
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(REPO / "outputs" / "spec_links.json"))
    args = ap.parse_args()

    meta = platform_authors()
    print(f"platform elements: {len(meta)}   with authors: "
          f"{sum(1 for m in meta.values() if m['authors'])}")

    with kb_db.connect(args.dsn) as conn:
        kb_db.ensure_schema(conn)
        with conn.cursor() as cur:
            # Every publication with a DOI, not only those with extracted steps. Both
            # DOI-citing pairs in the corpus point at publications whose step extraction came
            # back empty, so gating the citation route behind a spec produced nothing from the
            # strongest evidence available. A human writing a paper's DOI into a notebook is
            # saying "this implements that" whether or not an LLM got a step list out of the PDF.
            cur.execute("""SELECT element_id, doi, summary, steps FROM publication
                            WHERE jsonb_array_length(steps) > 0 OR doi <> ''""")
            specs = cur.fetchall()
            cur.execute("SELECT DISTINCT element_id FROM unit WHERE verdict = 'callable'")
            implementers = {r[0] for r in cur.fetchall()}
            # One pass for the citation route: every element's own text, once.
            cur.execute("""SELECT element_id, string_agg(markdown || ' ' || code, ' ')
                             FROM block GROUP BY element_id""")
            texts = dict(cur.fetchall())
        print(f"specs with steps: {len(specs)}   elements shipping callable code: "
              f"{len(implementers)}")

        by_author = collections.defaultdict(set)
        for short in implementers:
            for key in (meta.get(short) or {}).get("authors", []):
                by_author[key].add(short)

        def make_scorer(query: str):
            def score(element: str) -> float:
                with conn.cursor() as cur:
                    cur.execute("""
                        WITH q AS (SELECT to_tsquery('english', nullif(array_to_string(
                                     tsvector_to_array(to_tsvector('english', %s)),
                                     ' | '), '')) AS tq)
                        SELECT coalesce((SELECT sum(ts_rank(b.search, q.tq)) FROM block b, q
                                          WHERE b.element_id = %s AND b.search @@ q.tq), 0)
                             + coalesce((SELECT sum(ts_rank(u.search, q.tq)) FROM unit u, q
                                          WHERE u.element_id = %s AND u.search @@ q.tq), 0)
                    """, (query, element, element))
                    return float(cur.fetchone()[0] or 0.0)
            return score

        all_links = []
        for spec_element, doi, summary, steps in specs:
            query = " ".join([summary or ""] + [str(s) for s in (steps or [])])
            cited = spec_links.links_from_citation(spec_element, doi, texts)
            authored = spec_links.links_from_authors(
                spec_element,
                (meta.get(spec_element) or {}).get("authors", []),
                by_author,
                make_scorer(query))
            all_links.extend(spec_links.merge(cited, authored))

        links = spec_links.merge(all_links)
        by_conf = collections.Counter(l.confidence for l in links)
        print(f"\nlinks: {len(links)}  " + ", ".join(f"{v} {k}" for k, v in by_conf.items()))

        with conn.cursor() as cur:
            for link in links:
                cur.execute("""
                    INSERT INTO spec_link (spec_element, target_element, confidence, evidence,
                                           detail)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (spec_element, target_element) DO UPDATE SET
                        confidence = excluded.confidence, evidence = excluded.evidence,
                        detail = excluded.detail, updated_at = now()
                """, (link.spec_element, link.target_element, link.confidence,
                      link.evidence, json.dumps(link.detail)))
        if args.dry_run:
            conn.rollback()
            print("-- dry run, rolled back --")
        else:
            conn.commit()

    print()
    for link in links:
        spec_title = str((meta.get(link.spec_element) or {}).get("title"))[:38]
        target_title = str((meta.get(link.target_element) or {}).get("title"))[:38]
        print(f"  {link.confidence:<7}{link.evidence:<26}"
              f"{link.spec_element} {spec_title:<40} -> {link.target_element} {target_title}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"links": [l.as_edge() for l in links]}, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
