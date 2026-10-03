from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, Optional

# Configure module-wide logging once so search modules share formatting.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger configured with the shared log level."""
    return logging.getLogger(name)


def getenv(name: str, required: bool = True, default: Optional[str] = None) -> str:
    """Read a setting, preferring this tier's own value.

    ``PLATFORM_TIER=dev`` makes ``FOO_DEV`` win over ``FOO``. Every search read comes through
    here, which is how one switch moves the index names too: dev's knowledge base is
    ``iguide-platform-embeddings-dev`` while the bare ``OPENSEARCH_INDEX`` still says
    ``new-opensearch-index``, and querying the wrong one fails silently — the cluster answers,
    the query succeeds, and there are simply no results.
    """
    value = None
    try:
        from agent_runtime import platform_endpoints
        # SEARCH_TIER, not PLATFORM_TIER: which knowledge base to query is a different question
        # from which platform mints the tokens, and pointing one at dev and the other at prod is
        # an ordinary thing to want.
        value = platform_endpoints.tiered_env(name, default,
                                              tier=platform_endpoints.search_tier())
    except Exception:  # noqa: BLE001 - search predates the tier table; never hard-depend on it
        value = os.getenv(name, default)
    if required and (value is None or value == ""):
        raise RuntimeError(f"Missing required environment variable: {name}")
    if value and len(value) >= 2 and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
        value = value[1:-1]
    return value or ""


def safe_score(val: Any, default: float = 1.0) -> float:
    try:
        score = float(val)
        return score if math.isfinite(score) else default
    except Exception:
        return default


def normalize_source_fields(source: Dict[str, Any], fallback_id: str) -> Dict[str, Any]:
    if not isinstance(source, dict):
        source = {}
    source = dict(source)

    source.setdefault("doc_id", fallback_id)
    source.setdefault("title", source.get("name") or "No Title")
    source.setdefault("contents", source.get("abstract") or source.get("description") or "No Content")
    if "element_type" not in source and "resource-type" in source:
        source["element_type"] = source["resource-type"]

    return source


__all__ = [
    "get_logger",
    "getenv",
    "normalize_source_fields",
    "safe_score",
]


def snippet_chars(default: int = 4000) -> int:
    """Max characters kept per document snippet in normalized search hits.

    The old hard-coded 800 truncated most real abstracts (OpenGeoData descriptions routinely run
    1-3k characters), so citations and the structured results shown to users were cut mid-sentence.
    Prompt cost stays bounded downstream, where the synthesizer caps each evidence block anyway.
    Tune with AGENT_SEARCH_SNIPPET_CHARS.
    """
    try:
        return max(200, int(os.getenv("AGENT_SEARCH_SNIPPET_CHARS", str(default))))
    except (TypeError, ValueError):
        return default
