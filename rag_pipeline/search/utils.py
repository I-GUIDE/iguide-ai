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
    "default_top_k",
]


def default_top_k(default: int = 20) -> int:
    """Retrieval window: candidates each retrieval method returns per call.

    Single source of truth for a number that was previously the literal ``8`` in six
    independent places — ``keyword.py``, ``semantic.py``, four tool signatures in
    ``agent_runtime/langchain_granular_tools.py``, and ``_direct_search_sweep`` — none of
    which the agent ever overrode.

    Measured against the GeoPathfinder benchmark's full expected sets (37 ids, BM25, each
    task's verbatim prompt), recall by window:

        k=8  -> 22/37 (59%)      k=20 -> 29/37 (78%)
        k=50 -> 33/37 (89%)      k=100 -> 34/37 (92%)

    Hence the default of 20: it recovers 7 of the 15 elements the old window dropped, for
    the cost of one integer. Only 3 of the 37 are unreachable at any k and are genuine
    indexing gaps (afbee4bd, 643aaea1, de05a428).

    This is NOT the same knob as ``AGENT_SUPERVISOR_TOP_K`` (``supervisor/graph.py:71``),
    which caps how much evidence survives the rerank into the answer prompt. Retrieve wide,
    rerank, then show few — raising both would scale answer-prompt tokens with k for no
    recall gain. Tune with ``AGENT_SEARCH_TOP_K``.

    IMPORTANT for callers: resolve this at CALL time, never as a function default argument.
    A default is bound at import, so `limit: int = default_top_k()` would freeze the value
    and silently ignore the environment.
    """
    try:
        return max(1, min(int(os.getenv("AGENT_SEARCH_TOP_K", str(default))), 100))
    except (TypeError, ValueError):
        return default


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
