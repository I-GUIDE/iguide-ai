"""
neo4j_graph_tools.py
--------------------
Prewritten, parameterized Cypher tools for common relational query patterns
in the I-GUIDE knowledge graph.

These are deterministic, schema-aware queries that cover high-frequency
graph traversal patterns. They run BEFORE Text2Cypher and are much faster,
cheaper, and more reliable for known query shapes.

Hierarchy (called in order by GraphQueryDispatcher):
  1. Pattern match  → one of the tools below (deterministic, no LLM)
  2. Text2Cypher    → LLM-generated Cypher (flexible, catches everything else)
  3. Basic fallback → search_neo4j.get_neo4j_search_results (keyword-only)

Each tool returns List[Dict] in the same raw-hit shape expected by
_records_to_hits() in search_neo4j.py.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("neo4j_graph_tools")


# ---------------------------------------------------------------------------
# Schema constants — update these when the Neo4j schema changes
# ---------------------------------------------------------------------------
# Last verified: 2026-04 against Neo4j instance at 149.165.155.135
#
# To add a new resource type: add its label to _RESOURCE_LABELS
# To add a new internal/infra node: add its label to _INTERNAL_LABELS
# These can also be overridden at runtime via env vars without a code change:
#   NEO4J_RESOURCE_LABELS=Notebook,Dataset,Publication,...
#   NEO4J_INTERNAL_LABELS=Contributor,User,Alias,...

_DEFAULT_RESOURCE_LABELS = {
    "Notebook", "Dataset", "Publication", "Oer",
    "Documentation", "Map", "Code", "Collection",
}

_DEFAULT_INTERNAL_LABELS = {
    "Contributor", "User", "Alias", "Temp", "Notification",
}

_PUBLIC_VISIBILITY = "public"
# The platform stores node visibility as the STRING 'public' on newer nodes and the legacy
# NUMERIC 10 on older ones (utils.parseVisibility in the platform backend treats both as
# PUBLIC). Compare via toString(...) IN list so legacy nodes aren't silently filtered out.
_PUBLIC_VISIBILITIES = [_PUBLIC_VISIBILITY, "10"]


def is_public_visibility(value) -> bool:
    """True when a node/doc visibility marks it publicly LISTABLE.

    The platform treats visibility 'private' (legacy numeric 1) as UNLISTED: reachable by
    direct link but excluded from every listing/search. A missing value counts as public
    (contributor/infra nodes and OpenSearch docs may lack the field); any other value marks
    the element unlisted and it must NOT be surfaced by search.
    """
    if value is None or value == "":
        return True
    return str(value).strip().lower() in {v.lower() for v in _PUBLIC_VISIBILITIES}
_MAX_RELATED_DEPTH = 3
_DEFAULT_RELATED_DEPTH = 2


def _get_resource_labels() -> set:
    env_val = os.getenv("NEO4J_RESOURCE_LABELS", "").strip()
    if env_val:
        return {l.strip() for l in env_val.split(",") if l.strip()}
    return _DEFAULT_RESOURCE_LABELS


def element_type_from_labels(labels: Any) -> Optional[str]:
    """The element type a graph node's LABEL names, lower-cased, or None.

    The platform graph stores an element's type ONLY as its label. Measured on the VM on
    2026-10-13: Publication 206, Notebook 214, Map 161, Dataset 142, Oer 44, Code 38,
    Collection 19, and no node has an ``element_type`` or ``resource-type`` property. Hits built
    from node properties alone therefore fell back to ``element_type: "resource"``, which no link
    builder can turn into a platform page, so six of eight sources in a prod answer that day were
    plain text.

    Only knowledge-element labels count (``_get_resource_labels``): a node that is also a
    :Contributor must not become a "contributor" element. Sorted, so a node with two element
    labels gets the same type on every call rather than whichever a set iterates first.
    """
    if not labels:
        return None
    known = _get_resource_labels()
    matches = sorted(str(label) for label in labels if str(label) in known)
    return matches[0].lower() if matches else None


def _get_internal_labels() -> set:
    env_val = os.getenv("NEO4J_INTERNAL_LABELS", "").strip()
    if env_val:
        return {l.strip() for l in env_val.split(",") if l.strip()}
    return _DEFAULT_INTERNAL_LABELS


# ---------------------------------------------------------------------------
# Cypher constants
# ---------------------------------------------------------------------------

# Shared score expression reused across queries for consistency.
# Uses coalesce throughout so nodes without click_count still score correctly.
_SCORE_EXPR = (
    "coalesce(log10(toFloat(coalesce(r.click_count, 0)) + 1), 0) * 0.3 "
    "+ coalesce(toFloat(count { (r)--() }), 0) * 0.05"
)

_CYPHER_BY_AUTHOR = """
MATCH (c:Contributor)-[:CONTRIBUTED]->(r)
WHERE toLower(coalesce(c.display_first_name, '')) CONTAINS toLower($first)
  AND toLower(coalesce(c.display_last_name, ''))  CONTAINS toLower($last)
  AND toString(r.visibility) IN $public_visibilities
WITH r, c,
     {SCORE_EXPR} AS score
RETURN r AS node, score, c.display_first_name + ' ' + c.display_last_name AS matched_author
ORDER BY score DESC
LIMIT $limit
""".replace("{SCORE_EXPR}", _SCORE_EXPR)

# Secondary: search User nodes (same split-name approach)
_CYPHER_BY_USER_AUTHOR = """
MATCH (u:User)-[:CONTRIBUTED]->(r)
WHERE toLower(coalesce(u.display_first_name, '')) CONTAINS toLower($first)
  AND toLower(coalesce(u.display_last_name, ''))  CONTAINS toLower($last)
  AND toString(r.visibility) IN $public_visibilities
WITH r, u,
     {SCORE_EXPR} AS score
RETURN r AS node, score, u.display_first_name + ' ' + u.display_last_name AS matched_author
ORDER BY score DESC
LIMIT $limit
""".replace("{SCORE_EXPR}", _SCORE_EXPR)

_CYPHER_BY_ORGANIZATION = """
MATCH (c:Contributor)-[:CONTRIBUTED]->(r)
WHERE (toLower(coalesce(c.organization, '')) CONTAINS toLower($org)
   OR toLower(coalesce(c.affiliation, ''))  CONTAINS toLower($org)
   OR toLower(coalesce(c.institution, ''))  CONTAINS toLower($org))
  AND toString(r.visibility) IN $public_visibilities
WITH r, c,
     {SCORE_EXPR} AS score
RETURN r AS node, score, c.organization AS matched_org
ORDER BY score DESC
LIMIT $limit
""".replace("{SCORE_EXPR}", _SCORE_EXPR)

_CYPHER_BY_TAG = """
MATCH (r)
WHERE any(tag IN coalesce(r.tags, []) WHERE toLower(tag) CONTAINS toLower($tag))
  AND toString(r.visibility) IN $public_visibilities
WITH r,
     1.5 + {SCORE_EXPR} AS score
RETURN r AS node, score
ORDER BY score DESC
LIMIT $limit
""".replace("{SCORE_EXPR}", _SCORE_EXPR)

# Matches against node labels directly — automatically picks up new resource
# types without a code change as long as _RESOURCE_LABELS is kept current.
_CYPHER_BY_RESOURCE_TYPE = """
MATCH (r)
WHERE any(label IN labels(r) WHERE toLower(label) CONTAINS toLower($rtype))
  AND NOT any(label IN labels(r) WHERE label IN $internal_labels)
  AND toString(r.visibility) IN $public_visibilities
WITH r,
     {SCORE_EXPR} AS score
RETURN r AS node, score
ORDER BY score DESC
LIMIT $limit
""".replace("{SCORE_EXPR}", _SCORE_EXPR)

_CYPHER_RELATED_TO = """
MATCH (seed)-[:RELATED]-(r)
WHERE toLower(coalesce(seed.title, '')) CONTAINS toLower($title)
  AND toString(r.visibility) IN $public_visibilities
WITH r, seed,
     {SCORE_EXPR} AS score
RETURN r AS node, score, seed.title AS seed_title
ORDER BY score DESC
LIMIT $limit
""".replace("{SCORE_EXPR}", _SCORE_EXPR)

_CYPHER_IN_COLLECTION = """
MATCH (r)-[:BELONGS_TO]->(col:Collection)
WHERE toLower(coalesce(col.title, col.name, '')) CONTAINS toLower($collection)
  AND toString(r.visibility) IN $public_visibilities
WITH r, col,
     {SCORE_EXPR} AS score
RETURN r AS node, score, col.title AS collection_name
ORDER BY score DESC
LIMIT $limit
""".replace("{SCORE_EXPR}", _SCORE_EXPR)

# Sorts purely by click_count DESC. Optionally filtered by resource type label.
# $rtype = '' means no type filter (return all resource types).
_CYPHER_BY_POPULARITY = """
MATCH (r)
WHERE ($rtype = '' OR any(label IN labels(r) WHERE toLower(label) CONTAINS toLower($rtype)))
  AND NOT any(label IN labels(r) WHERE label IN $internal_labels)
  AND toString(r.visibility) IN $public_visibilities
WITH r, coalesce(toFloat(r.click_count), 0) AS clicks
WHERE clicks > 0
RETURN r AS node, clicks AS score
ORDER BY score DESC
LIMIT $limit
"""

# Sorts purely by click_count DESC. Optionally filtered by resource type label.
# $rtype = '' means no type filter (return all resource types).
_CYPHER_BY_POPULARITY = """
MATCH (r)
WHERE ($rtype = '' OR any(label IN labels(r) WHERE toLower(label) CONTAINS toLower($rtype)))
  AND NOT any(label IN labels(r) WHERE label IN $internal_labels)
  AND toString(r.visibility) IN $public_visibilities
WITH r, coalesce(toFloat(r.click_count), 0) AS clicks
WHERE clicks > 0
RETURN r AS node, clicks AS score
ORDER BY score DESC
LIMIT $limit
"""

_CYPHER_GET_ELEMENT_BY_ID = """
MATCH (n {id: $element_id})
WHERE toString(n.visibility) IN $public_visibilities
OPTIONAL MATCH (c)-[:CONTRIBUTED]-(n)
RETURN n AS node,
       1.0 AS score,
       CASE
         WHEN c IS NULL THEN null
         ELSE trim(coalesce(c.display_first_name, '') + ' ' + coalesce(c.display_last_name, ''))
       END AS matched_author
LIMIT 1
"""
_CYPHER_GET_ELEMENT_BY_ID = """
MATCH (n {id: $element_id})
WHERE toString(n.visibility) IN $public_visibilities
OPTIONAL MATCH (c)-[:CONTRIBUTED]-(n)
RETURN n AS node,
       1.0 AS score,
       CASE
         WHEN c IS NULL THEN null
         ELSE trim(coalesce(c.display_first_name, '') + ' ' + coalesce(c.display_last_name, ''))
       END AS matched_author
LIMIT 1
"""



# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------

# Each entry: (pattern_name, compiled_regex, list_of_capture_group_names)
#
# ORDER MATTERS — first match wins. Rules:
#   1. by_popularity    — must be first; "by click count" would otherwise match by_author
#   2. by_author        — must come before by_organization (both can match "by X")
#   3. in_collection    — must come before by_resource_type ("resources in X collection"
#                         would otherwise match "resources" as a resource type)
#   4. related_to       — must come before by_resource_type for the same reason
#   5. by_organization  — before by_resource_type (org names can look like type words)
#   6. by_tag           — explicit trigger words, safe anywhere
#   7. by_resource_type — last resort; only fires on explicit verb + type word
_ID_TOKEN = r"([A-Za-z0-9][A-Za-z0-9_.:/-]{1,120})"

_PATTERNS: List[Tuple[str, re.Pattern, List[str]]] = [
    # 0. Exact ID lookup / neighborhood exploration. These are deterministic
    # graph operations and should not fall through to Text2Cypher.
    (
        "explore_related_by_id",
        re.compile(
            r"\b(?:neighbors?|related(?:\s+knowledge)?(?:\s+(?:nodes?|elements?|resources?))?|"
            r"explore(?:\s+(?:related|neighbors?|graph))?|graph(?:\s+around)?|"
            r"connections?)\s+(?:for|of|from|around|to)?\s*"
            r"(?:knowledge\s+element|element|resource)?\s*"
            r"(?:id[:#]?\s*)?"
            + _ID_TOKEN,
            re.I,
        ),
        ["element_id"],
    ),
    (
        "element_by_id",
        re.compile(
            r"\b(?:(?:knowledge\s+element|element|resource)\s+(?:id[:#]?\s*)?|"
            r"id[:#]\s*)"
            + _ID_TOKEN,
            re.I,
        ),
        ["element_id"],
    ),
    # 1. Popularity — must be first so "by click count" isn't parsed as an author name
    (
        "by_popularity",
        re.compile(
            r"\b(?:most\s+popular|most\s+clicked|most\s+viewed|most\s+visited|"
            r"most\s+accessed|top\s+clicked|top\s+viewed|top\s+rated|"
            r"highest\s+clicks?|highest\s+click\s+count|trending|popular)\b"
            r"(?:\s+(datasets?|notebooks?|publications?|maps?|code|oers?|"
            r"documentation|collections?))?",
            re.I,
        ),
        ["rtype"],
    ),
    # 2. Author / person — matches "Firstname Lastname" or "Firstname Middle Lastname"
    #    Excludes institutional trigger words so "University of Illinois" routes to
    #    by_organization instead.
    (
        "by_author",
        re.compile(
            r"\b(?:by|from|authored\s+by|written\s+by|published\s+by|"
            r"publications?\s+(?:by|from)|notebooks?\s+(?:by|from)|"
            r"resources?\s+(?:by|from)|contributions?\s+(?:by|from)|"
            r"what\s+has\s+|work\s+(?:by|from))\s+"
            r"(?!(?:university|institute|lab|laboratory|center|centre|"
            r"agency|department|dept|college|school|nasa|noaa|usgs|epa)\b)"
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b"  # 2-3 words, all Title Case
            r"(?!\s+(?:university|institute|lab|of\s+))",
            re.I,
        ),
        ["name"],
    ),
    # 2. Collection membership — "in/within the X collection"
    (
        "in_collection",
        re.compile(
            r"\b(?:in|within|part\s+of|inside|belonging\s+to)\s+"
            r"(?:the\s+)?(?:collection\s+)?"
            r"[\"']?([a-zA-Z0-9 _\-]{3,60})[\"']?"
            r"(?:\s+collection)?",
            re.I,
        ),
        ["collection"],
    ),
    # 3. Related-to — "related to / similar to / associated with X"
    (
        "related_to",
        re.compile(
            r"\b(?:related\s+to|similar\s+to|associated\s+with)\s+"
            r"[\"']?([^\"'\n]{3,60})[\"']?",
            re.I,
        ),
        ["title"],
    ),
    # 4. Organization — requires capitalized org name to reduce false matches
    (
        "by_organization",
        re.compile(
            r"\b(?:from|affiliated\s+with|contributed\s+by|submitted\s+by|at|by)\s+"
            r"((?:[A-Z][a-zA-Z&,.\-]+\s*){1,6})"
            r"(?:\s+(?:university|institute|lab|laboratory|center|centre|"
            r"agency|department|dept|college|school|program|project))?",
            re.I,
        ),
        ["org"],
    ),
    # 5. Tag — explicit trigger words only
    (
        "by_tag",
        re.compile(
            r"\b(?:tagged\s+(?:with\s+)?|tag[:\s]+|"
            r"labeled\s+(?:with\s+)?|category[:\s]+|topic[:\s]+)"
            r"[\"']?([a-zA-Z0-9 _\-]+)[\"']?",
            re.I,
        ),
        ["tag"],
    ),
    # 7. Resource type — requires an explicit action verb before the type word
    #    so bare mentions like "resources" or "datasets from NASA" don't trigger it
    (
        "by_resource_type",
        re.compile(
            r"\b(?:show\s+(?:me\s+)?(?:all\s+)?|find\s+(?:all\s+)?|"
            r"list\s+(?:all\s+)?|get\s+(?:all\s+)?|browse\s+(?:all\s+)?|"
            r"all\s+)"
            r"(datasets?|notebooks?|publications?|maps?|code|oers?|"
            r"documentation|collections?)\b",
            re.I,
        ),
        ["rtype"],
    ),
]


def detect_pattern(query: str) -> Optional[Tuple[str, Dict[str, str]]]:
    """
    Return (pattern_name, captured_params) for the first matching pattern,
    or None if no pattern matches.

    >>> detect_pattern("publications by John Smith")
    ('by_author', {'name': 'John Smith'})
    >>> detect_pattern("datasets from NASA")
    ('by_organization', {'org': 'NASA'})
    >>> detect_pattern("what is climate change")
    None
    """
    for pattern_name, regex, group_names in _PATTERNS:
        m = regex.search(query)
        if m:
            # Map positional groups to named keys
            captured = {}
            for i, key in enumerate(group_names, 1):
                try:
                    val = m.group(i)
                except IndexError:
                    val = ""
                captured[key] = (val or "").strip()

            # Skip if the captured value is empty or too short.
            # by_popularity allows an empty rtype (means "all resource types").
            main_val = captured.get(group_names[0], "")
            if len(main_val) < 2 and pattern_name != "by_popularity":
                continue
            if pattern_name in {"element_by_id", "explore_related_by_id"}:
                if not _looks_like_element_id(main_val):
                    continue

            log.debug("Pattern '%s' matched query '%s' → %s", pattern_name, query, captured)
            return pattern_name, captured

    return None


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

_TOOL_MAP: Dict[str, str] = {
    "by_author": _CYPHER_BY_AUTHOR,
    "by_organization": _CYPHER_BY_ORGANIZATION,
    "by_tag": _CYPHER_BY_TAG,
    "by_resource_type": _CYPHER_BY_RESOURCE_TYPE,
    "by_popularity": _CYPHER_BY_POPULARITY,
    "related_to": _CYPHER_RELATED_TO,
    "in_collection": _CYPHER_IN_COLLECTION,
}


def _split_name(full_name: str) -> Tuple[str, str]:
    """
    Split a name string into (first, last) for independent matching.
    Handles middle names: "Smit Bharat Vasani" → first="Smit", last="Vasani"
    Single word: "Vasani" → first="", last="Vasani"
    """
    parts = full_name.strip().split()
    if len(parts) == 0:
        return "", ""
    if len(parts) == 1:
        return "", parts[0]          # treat single word as last name
    return parts[0], parts[-1]       # first word = first name, last word = last name


def _looks_like_element_id(value: str) -> bool:
    value = (value or "").strip()
    if len(value) < 2:
        return False
    # Element ids on this platform are UUID-shaped (and otherwise always carry a digit or id
    # punctuation). A purely alphabetic token — e.g. "knowledge", "elements", "resources" — is
    # an English word swept up by a noun phrase, NOT an id; treating it as a seed produced a
    # bogus empty traversal. Require a digit or id punctuation so plain words are rejected.
    return bool(re.search(r"[\d_:/.-]", value))


def _normalize_element_id(element_id: str) -> str:
    normalized = str(element_id or "").strip()
    if not normalized:
        raise ValueError("element_id must not be empty")
    return normalized


def _clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def build_element_by_id_query(element_id: str) -> Tuple[str, Dict[str, Any]]:
    """
    Return a deterministic public-only lookup query for a knowledge element ID.
    """
    return _CYPHER_GET_ELEMENT_BY_ID, {
        "element_id": _normalize_element_id(element_id),
        "public_visibilities": _PUBLIC_VISIBILITIES,
    }


def build_explore_related_nodes_query(
    element_id: str,
    depth: int = _DEFAULT_RELATED_DEPTH,
    limit: int = 50,
) -> Tuple[str, Dict[str, Any]]:
    """
    Return a deterministic public-only RELATED traversal query.

    The relationship-depth bound is sanitized before interpolation because
    Neo4j variable-length relationship bounds cannot be parameterized reliably.
    """
    safe_depth = _clamp_int(depth, _DEFAULT_RELATED_DEPTH, 1, _MAX_RELATED_DEPTH)
    safe_limit = _clamp_int(limit, 50, 1, 100)
    cypher = f"""
MATCH (seed {{id: $element_id}})
WHERE toString(seed.visibility) IN $public_visibilities
CALL {{
  WITH seed
  MATCH path = (seed)-[:RELATED*1..{safe_depth}]-(related)
  WHERE all(path_node IN nodes(path) WHERE toString(path_node.visibility) IN $public_visibilities)
  WITH related, relationships(path) AS rels, length(path) AS path_depth
  ORDER BY path_depth ASC
  LIMIT $limit
  WITH collect(DISTINCT related) AS nodes, collect(rels) AS rel_lists
  WITH nodes, reduce(all_rels = [], rel_list IN rel_lists | all_rels + rel_list) AS flattened_rels
  RETURN nodes,
         [rel IN flattened_rels | {{src: startNode(rel).id, dst: endNode(rel).id, type: type(rel)}}] AS edges
}}
RETURN seed, nodes, edges
"""
    return cypher, {
        "element_id": _normalize_element_id(element_id),
        "public_visibilities": _PUBLIC_VISIBILITIES,
        "limit": safe_limit,
    }


def build_tool_query(
    pattern_name: str,
    captured: Dict[str, str],
    limit: int,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Return (cypher, params) for a matched pattern, or None if unsupported.
    Author queries search Contributor nodes first; User nodes are the fallback
    handled by run_user_author_fallback.
    """
    if pattern_name == "element_by_id":
        return build_element_by_id_query(captured.get("element_id", ""))

    cypher = _TOOL_MAP.get(pattern_name)
    if not cypher:
        return None

    params: Dict[str, Any] = {"limit": max(1, min(limit, 100)),
                              "public_visibilities": _PUBLIC_VISIBILITIES}

    if pattern_name == "by_author":
        first, last = _split_name(captured.get("name", ""))
        params["first"] = first
        params["last"] = last
    elif pattern_name == "by_organization":
        params["org"] = captured.get("org", "")
    elif pattern_name == "by_tag":
        params["tag"] = captured.get("tag", "")
    elif pattern_name == "by_resource_type":
        rtype = captured.get("rtype", "")
        # Labels are singular (Notebook, Dataset) but queries use plurals (notebooks, datasets)
        # Strip trailing 's' so "notebooks" matches label "Notebook"
        if rtype.endswith("s") and not rtype.endswith("ss"):
            rtype = rtype[:-1]
        params["rtype"] = rtype
        params["internal_labels"] = list(_get_internal_labels())
    elif pattern_name == "related_to":
        params["title"] = captured.get("title", "")
    elif pattern_name == "in_collection":
        params["collection"] = captured.get("collection", "")
    elif pattern_name == "by_popularity":
        rtype = captured.get("rtype") or ""
        if rtype.endswith("s") and not rtype.endswith("ss"):
            rtype = rtype[:-1]
        params["rtype"] = rtype.lower()
        params["internal_labels"] = list(_get_internal_labels())

    return cypher, params


def run_user_author_fallback(
    captured: Dict[str, str],
    limit: int,
    run_fn,
) -> List[Dict[str, Any]]:
    """
    Secondary author lookup via User nodes when Contributor lookup returns nothing.
    Uses the same first/last split so middle names don't break matching.
    """
    first, last = _split_name(captured.get("name", ""))
    params = {"first": first, "last": last, "limit": max(1, min(limit, 100)),
              "public_visibilities": _PUBLIC_VISIBILITIES}
    try:
        return run_fn(_CYPHER_BY_USER_AUTHOR, params)
    except Exception as exc:
        log.warning("User author fallback failed: %s", exc)
        return []
