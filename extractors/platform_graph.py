"""Element records read from the platform's Neo4j graph. READ ONLY.

Why this exists alongside the REST API. The public API rate-limits a corpus sweep — a full pass
issues one request per element, 750 of them, and returns 429 partway through. The failure then
arrives as ``metadata_error`` on *every* element, which reads as "the corpus is unreachable"
rather than "we asked too fast". The graph holds the same records, answers a whole label in one
query, and is the store the API is a view of.

**This module never writes.** Every query is ``MATCH … RETURN``. The graph carries real user
data — bookmarks, edit permissions, contributor links — and extraction has no business changing
any of it. There is deliberately no session method here that could.

Property names differ between the two sources: the graph uses ``github_repo_link`` where the API
uses ``github-repo-link``. Callers (the extractors) speak the API's dialect, so records are
translated on the way out rather than every consumer learning both.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

# Neo4j label -> the element_type the extractors dispatch on.
LABEL_FOR_TYPE = {
    "notebook": "Notebook",
    "code": "Code",
    "dataset": "Dataset",
    "publication": "Publication",
    "map": "Map",
    "oer": "Oer",
}

# The fetchable-source property per type, measured on the live graph. Recorded here because the
# numbers are the answer to "why does this type yield nothing":
#   notebook  notebook_repo 200/200        — fully fetchable
#   code      github_repo_link 38/38       — fully fetchable
#   dataset   direct_download_link 73/141  — half have only external_link, a landing page
#   publication external_link 198/200      — a DOI/landing page, not a PDF
#   map       external_iframe_link 4/161   — essentially nothing to fetch
#   oer       oer_elink_urls 14/44         — mostly links out
SOURCE_PROPERTY = {
    "notebook": ("notebook_repo", "notebook_url", "notebook_file"),
    "code": ("github_repo_link",),
    "dataset": ("direct_download_link", "external_link"),
    "publication": ("external_link",),
    "map": ("external_iframe_link",),
    "oer": ("oer_elink_urls",),
}


def is_enabled() -> bool:
    return bool(os.getenv("NEO4J_CONNECTION_STRING") or os.getenv("NEO4J_URI"))


def _driver():
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_CONNECTION_STRING") or os.getenv("NEO4J_URI") or ""
    user = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME") or "neo4j"
    return GraphDatabase.driver(uri, auth=(user, os.getenv("NEO4J_PASSWORD") or ""))


def _api_dialect(props: Dict[str, Any]) -> Dict[str, Any]:
    """Graph property names -> the hyphenated names the extractors already read.

    Both spellings are kept. A caller that knows the graph can use either, and nothing downstream
    has to be taught that two names mean one thing.
    """
    out: Dict[str, Any] = dict(props)
    for key, value in list(props.items()):
        hyphen = key.replace("_", "-")
        if hyphen != key and hyphen not in out:
            out[hyphen] = value
    return out


def elements(element_type: str, *, limit: int = 2000,
             include_private: bool = False) -> List[Dict[str, Any]]:
    """Every element of one type, as API-shaped records.

    Private elements are excluded by default and the filter is **fail-closed**: anything whose
    ``visibility`` is not exactly ``public`` is dropped, so a value nobody anticipated keeps data
    out rather than letting it through. 43 of 793 elements are private on the live graph.
    """
    label = LABEL_FOR_TYPE.get(element_type)
    if not label:
        raise ValueError(f"no graph label for element type {element_type!r}; "
                         f"known: {sorted(LABEL_FOR_TYPE)}")

    # The visibility filter belongs in the QUERY, not after it. Applied afterwards, LIMIT counts
    # private nodes too: asking for 200 notebooks returned 203 rows minus 23 private = 177
    # public, silently three short of the real 180, and the shortfall looked like three elements
    # that simply failed to extract.
    where = "" if include_private else " WHERE n.visibility = 'public'"
    driver = _driver()
    try:
        with driver.session(database=os.getenv("NEO4J_DB") or None) as session:
            rows = session.run(
                f"MATCH (n:{label}){where} RETURN n LIMIT $limit", limit=int(limit)).data()
    finally:
        driver.close()

    out: List[Dict[str, Any]] = []
    for row in rows:
        props = dict(row["n"])
        if not include_private:
            # Belt and braces: the query already filtered, and this re-checks fail-closed so a
            # future edit to the Cypher cannot quietly start leaking private elements.
            if str(props.get("visibility") or "").strip().lower() != "public":
                continue
        record = _api_dialect(props)
        record.setdefault("resource-type", element_type)
        out.append(record)
    return out


def source_url(record: Dict[str, Any], element_type: str) -> Optional[str]:
    """The first fetchable source on a record, or None when the element has no artifact.

    None is a real answer, not a failure: 157 of 161 maps genuinely have nothing to fetch. Their
    value is the spatial metadata already on the record, not a file.
    """
    for prop in SOURCE_PROPERTY.get(element_type, ()):
        value = record.get(prop) or record.get(prop.replace("_", "-"))
        if isinstance(value, (list, tuple)):
            value = next((v for v in value if str(v).startswith("http")), None)
        text = str(value or "").strip()
        if text.startswith(("http://", "https://")):
            return text
    return None


def counts(include_private: bool = False) -> Dict[str, int]:
    """Per-type element counts, for a coverage denominator that is not guessed."""
    driver = _driver()
    out: Dict[str, int] = {}
    try:
        with driver.session(database=os.getenv("NEO4J_DB") or None) as session:
            for etype, label in LABEL_FOR_TYPE.items():
                clause = "" if include_private else " WHERE n.visibility = 'public'"
                out[etype] = session.run(
                    f"MATCH (n:{label}){clause} RETURN count(n) AS c").single()["c"]
    finally:
        driver.close()
    return out


__all__ = ["elements", "counts", "source_url", "is_enabled",
           "LABEL_FOR_TYPE", "SOURCE_PROPERTY"]
