"""Validation and normalization of the ``enabledSearchMethods`` request field.

The allowlist is a pure name-equality filter over the retrieval tool registry, so a name that
does not match EXACTLY used to be dropped in silence: ``"Keyword_Search"``, ``"neo4j"`` or a typo
produced an agent with NO retrieval tools, which then answered "I couldn't find any supporting
evidence" with HTTP 200 and no hint that the request was at fault.

This module makes that impossible:

* case/format differences and the obvious short forms are NORMALIZED to the real tool name;
* a genuinely unknown name is REJECTED with a message naming it and listing the valid methods
  (the API layer turns that into a 400);
* an empty list is treated as "unspecified" (all methods) rather than "no retrieval at all",
  because a client asking for zero search tools is almost always a bug, not an intent.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

logger = logging.getLogger(__name__)

# The retrieval tools the allowlist can name (mirrors the registry in
# agent_runtime.langchain_granular_tools.make_langchain_granular_tools).
KNOWN_SEARCH_METHODS: tuple = (
    "keyword_search",
    "semantic_search",
    "neo4j_search",
    "neo4j_get_element_by_id",
    "neo4j_explore_related_nodes",
    "spatial_search",
    "opengeodata_search",
    "overpass_search",
    "web_search",
    "web_fetch",
    "agent_kb_search",
    "get_kb_block",
    "kb_method_search",
    "get_method_contract",
)

# Forgiving spellings that unambiguously mean one of the above.
_ALIASES = {
    "keyword": "keyword_search",
    "keywordsearch": "keyword_search",
    "bm25": "keyword_search",
    "semantic": "semantic_search",
    "semanticsearch": "semantic_search",
    "vector": "semantic_search",
    "vectorsearch": "semantic_search",
    "neo4j": "neo4j_search",
    "neo4jsearch": "neo4j_search",
    "graph": "neo4j_search",
    "graphsearch": "neo4j_search",
    "spatial": "spatial_search",
    "spatialsearch": "spatial_search",
    "geo": "spatial_search",
    "opengeodata": "opengeodata_search",
    "opengeodatasearch": "opengeodata_search",
    "open_geo_data": "opengeodata_search",
    "external": "opengeodata_search",
    "overpass": "overpass_search",
    "osm": "overpass_search",
    "openstreetmap": "overpass_search",
    "web": "web_search",
    "websearch": "web_search",
    "internet": "web_search",
    "openweb": "web_search",
    "webfetch": "web_fetch",
    "fetch": "web_fetch",
    "webpage": "web_fetch",
    "agentkb": "agent_kb_search",
    "agent_kb": "agent_kb_search",
    "agentkbsearch": "agent_kb_search",
    "kbblock": "get_kb_block",
    "getkbblock": "get_kb_block",
    "kbmethod": "kb_method_search",
    "kbmethodsearch": "kb_method_search",
    "methodsearch": "kb_method_search",
    "method_library": "kb_method_search",
    "methodlibrary": "kb_method_search",
    "methods": "kb_method_search",
    "methodcontract": "get_method_contract",
    "getmethodcontract": "get_method_contract",
}


def _canonical(name: str) -> Optional[str]:
    """The registry name for *name*, or None when it matches nothing."""
    raw = str(name or "").strip()
    if not raw:
        return None
    lowered = raw.lower()
    if lowered in KNOWN_SEARCH_METHODS:
        return lowered
    squashed = lowered.replace("-", "").replace("_", "").replace(" ", "")
    if squashed in _ALIASES:
        return _ALIASES[squashed]
    for known in KNOWN_SEARCH_METHODS:
        if squashed == known.replace("_", ""):
            return known
    return None


def normalize_search_methods(value: Any) -> Optional[List[str]]:
    """Validate/normalize an ``enabledSearchMethods`` value.

    Accepts None, a list of names, or a comma-separated string. Returns the canonical tool names
    (order preserved, de-duplicated) or None for "unspecified" (= every method available).

    Raises ValueError for a non-list/str value or for any unrecognized method name; the API layer
    maps that to HTTP 400 so a client learns immediately instead of silently getting an agent with
    no retrieval tools.
    """
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        items: Sequence[Any] = str(value).split(",")
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        raise ValueError(
            "enabled_search_methods must be a list of strings or a comma-separated string"
        )

    canonical: List[str] = []
    unknown: List[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        name = _canonical(text)
        if name is None:
            unknown.append(text)
        elif name not in canonical:
            canonical.append(name)

    if unknown:
        raise ValueError(
            "unknown search method(s): " + ", ".join(repr(u) for u in unknown)
            + ". Valid methods: " + ", ".join(KNOWN_SEARCH_METHODS)
        )
    if not canonical:
        # An empty/blank list would otherwise strip EVERY retrieval tool and yield a confident
        # "no evidence" answer. Treat it as unspecified instead.
        logger.warning("enabled_search_methods was empty after normalization; using all methods")
        return None
    return canonical


__all__ = ["KNOWN_SEARCH_METHODS", "normalize_search_methods"]
