"""Open-web search: the CHEAP discovery stage.

This stage returns metadata only — title, URL, engine snippet. No page bodies. That split is the
main reason the efficient agents stay cheap: the model reads ~200-character snippets, decides which
one or two of ten results are worth opening, and only then pays for a page. Nothing is "filtered
out" downstream; seven of ten documents are simply never retrieved.

Contrast with the shape this replaces (``MCP_server/tools/search_tools.py``): hard-coded query
templates, raw engine dicts handed straight to the model, no cap, no dedupe, no relevance test.

Provider is pluggable but defaults to DuckDuckGo via ``ddgs`` — already a dependency, and keyless,
so this needs no API-key procurement to run in every environment.
"""

from __future__ import annotations

import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import web_utils as WU
from .opengeodata_new import _term_hit as term_hit
from .opengeodata_new import meaningful_terms
from .utils import get_logger

logger = get_logger("web_search")

# Conversational scaffolding to drop before handing the query to an engine. Deliberately far
# narrower than the catalog path's focus_query(): that one also strips "data", "open", "dataset",
# which are real, discriminating tokens on the open web ("open data portal Illinois"). Only the
# addressed-to-an-assistant framing goes.
# Every alternative must be able to match a NON-empty string: an all-optional branch matches the
# empty string, and a zero-width match ends the enclosing `+` repetition, so only the first prefix
# would ever be stripped. "google" is deliberately absent — stripping it would maul the many real
# queries that begin with the product name ("Google Earth Engine", "Google Colab").
_QUERY_PREAMBLE_RE = re.compile(
    r"^\s*(?:"
    r"(?:can|could|would|will)\s+you\s+(?:please\s+)?|"
    r"(?:please\s+)?help\s+me\s+(?:to\s+)?|"
    r"please\s+|"
    r"i\s+(?:want|need|would\s+like)\s+(?:to\s+)?(?:know|find|see|get)?\s*|"
    r"(?:tell|show|give)\s+me\s+(?:about\s+)?|"
    r"(?:do\s+a\s+|run\s+a\s+|perform\s+a\s+)?web\s+search\s+(?:for|about|on)\s+|"
    r"search\s+(?:the\s+web\s+)?(?:for|about|on)\s+|"
    r"look\s+up\s+"
    r")+",
    re.IGNORECASE,
)

_RECENCY_BUCKETS = ((1, "d"), (7, "w"), (31, "m"), (366, "y"))


def web_query(query: Optional[str]) -> str:
    """The query as an engine should see it: conversational framing removed, subject intact."""
    text = str(query or "").strip()
    cleaned = _QUERY_PREAMBLE_RE.sub("", text).strip(" ?.!,:;\"'")
    return cleaned or text


def _timelimit(recency_days: Optional[int]) -> Optional[str]:
    """Map a day count onto the engine's coarse recency buckets."""
    if not recency_days or recency_days <= 0:
        return None
    for days, code in _RECENCY_BUCKETS:
        if recency_days <= days:
            return code
    return None


# --- providers ------------------------------------------------------------------


def search_duckduckgo(query: str, *, limit: int = 6, recency_days: Optional[int] = None) -> List[WU.WebHit]:
    """DuckDuckGo text search. Keyless; returns metadata only."""
    try:
        from ddgs import DDGS
    except Exception as exc:  # pragma: no cover - dependency is pinned
        raise RuntimeError("ddgs is not installed; open-web search is unavailable") from exc

    kwargs: Dict[str, Any] = {
        "max_results": max(1, limit),
        "region": "us-en",
        # Measured on this deployment, result quality varies a lot by backend and by query, in both
        # directions: "auto" (multi-engine, deduped) padded a technical query with dictionary and
        # encyclopedia pages, while the single "duckduckgo" backend returned four hits from an
        # unrelated diet site for "NHDPlus HR documentation". "auto" is the default because
        # aggregating several engines survives any one of them rate-limiting us; override per
        # deployment with AGENT_WEB_BACKEND. A keyed provider is the upgrade path if the free
        # engines' variance proves unacceptable.
        "backend": str(os.getenv("AGENT_WEB_BACKEND", "auto")).strip() or "auto",
    }
    limit_code = _timelimit(recency_days)
    if limit_code:
        kwargs["timelimit"] = limit_code
    raw = DDGS(timeout=WU.web_timeout()).text(query, **kwargs) or []

    hits: List[WU.WebHit] = []
    for rank, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        url = str(item.get("href") or item.get("url") or "").strip()
        if not url.startswith(("http://", "https://")):
            continue
        hits.append(
            WU.WebHit(
                url=url,
                title=str(item.get("title") or url).strip(),
                snippet=str(item.get("body") or "").strip()[: WU.search_snippet_chars()],
                provider="duckduckgo",
                rank=rank,
            )
        )
    return hits


_PROVIDERS: Dict[str, Callable[..., List[WU.WebHit]]] = {"duckduckgo": search_duckduckgo}


def default_provider() -> str:
    name = str(os.getenv("AGENT_WEB_PROVIDER", "duckduckgo")).strip().lower()
    return name if name in _PROVIDERS else "duckduckgo"


# --- dedupe + relevance ---------------------------------------------------------


def dedupe(hits: Sequence[WU.WebHit]) -> List[WU.WebHit]:
    """Drop repeats of the same document, comparing canonical URLs (trackers/fragments removed)."""
    seen: set = set()
    kept: List[WU.WebHit] = []
    for hit in hits:
        key = WU.canonical_url(hit.url)
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(hit)
    return kept


def filter_hits(hits: Sequence[WU.WebHit], terms: Sequence[str]) -> List[WU.WebHit]:
    """Drop results with no subject-term evidence in title or snippet.

    Much gentler than the catalog gate, on purpose. A catalog connector returns whatever its
    fielded query matched, so relevance has to be established here. A search engine has ALREADY
    ranked for relevance, so aggressive filtering mostly destroys good results — and web snippets
    are short enough that a real match can genuinely be absent from them.

    So this only removes hits with zero evidence, and never empties the set: if nothing shows term
    evidence, the engine's own ranking is better than nothing.
    """
    if not terms:
        return list(hits)
    kept = [
        hit
        for hit in hits
        if any(term_hit(term, f"{hit.title} {hit.snippet}".lower()) for term in terms)
    ]
    return kept or list(hits)


# --- entry point ----------------------------------------------------------------


def run_web_search(
    query: str,
    *,
    limit: int = 6,
    recency_days: Optional[int] = None,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Search the open web and return metadata-only results plus observability fields.

    Never raises for the caller: a disabled feature, an exhausted budget, or a provider outage
    comes back as ``{"error": ...}`` so the agent reads it as a tool result and moves on instead of
    failing the turn.
    """
    denied = WU.charge_search()
    if denied:
        logger.info("web search refused: %s", denied)
        return {"query": query, **denied.fields(), "count": 0, "results": []}

    engine_query = web_query(query)
    if not engine_query:
        return {"query": query, "error": "empty query", "count": 0, "results": []}

    name = (provider or default_provider()).lower()
    search = _PROVIDERS.get(name, search_duckduckgo)
    capped = max(1, min(int(limit or 6), 10))

    try:
        raw_hits = search(engine_query, limit=capped, recency_days=recency_days)
    except Exception as exc:
        logger.warning("web search provider %s failed: %s", name, exc)
        return {
            "query": query,
            "search_query": engine_query,
            "provider": name,
            "error": f"web search provider unavailable: {exc}",
            "count": 0,
            "results": [],
        }

    found = len(raw_hits)
    unique = dedupe(raw_hits)
    terms = meaningful_terms(query)
    kept = filter_hits(unique, terms)[:capped]

    # Record before returning: a URL is citable only because a search surfaced it.
    WU.record_urls([hit.url for hit in kept])

    return {
        "query": query,
        "search_query": engine_query,
        "provider": name,
        "terms": terms,
        "candidates_found": found,
        "duplicates_dropped": found - len(unique),
        "filtered_out": len(unique) - len(kept),
        "count": len(kept),
        "results": [
            {
                "doc_id": hit.doc_id(),
                "title": hit.title,
                "url": hit.url,
                "snippet": hit.snippet,
                "provider": hit.provider,
                "published": hit.published,
                "rank": hit.rank,
            }
            for hit in kept
        ],
        "budget": WU.budget_snapshot(),
    }


def results_to_hits(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a ``run_web_search`` payload into the service's internal hit shape.

    Riding the existing envelope means ``_normalize_hits`` produces the citation ids and
    ``_landing_url`` produces the hyperlink with no downstream change at all. ``visibility`` is
    left unset, which the platform reads as public.
    """
    hits: List[Dict[str, Any]] = []
    for item in (payload or {}).get("results") or []:
        hits.append(
            {
                "_id": item["doc_id"],
                "_score": max(0.0, 1.0 - 0.01 * int(item.get("rank") or 0)),
                "_source": {
                    "doc_id": item["doc_id"],
                    "title": item["title"],
                    "contents": item["snippet"],
                    "element_type": "web",
                    "url": item["url"],
                    "provider": item.get("provider"),
                    "published": item.get("published"),
                },
            }
        )
    return hits


def get_web_search_results(query: str, size: int = 6, **_: Any) -> List[Dict[str, Any]]:
    """Alias matching the other search modules' ``get_*_results(query, size=...)`` convention."""
    return results_to_hits(run_web_search(query, limit=size))


__all__ = [
    "dedupe",
    "filter_hits",
    "get_web_search_results",
    "results_to_hits",
    "run_web_search",
    "search_duckduckgo",
    "web_query",
]
