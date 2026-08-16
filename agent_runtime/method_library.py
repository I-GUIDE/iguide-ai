"""Agent-side reader for the generated method library.

The extractor writes callable slices to ``storage_root()/method_library/iguide_methods/`` and
``code_execution`` mounts that directory read-only into the sandbox. Everything in between —
"which methods exist, what do they take, how do I import one" — is this module.

**Why the registry and not OpenSearch.** The plan called for ``kb_method_search`` to query a
``MethodUnit`` index. It reads ``_registry.json`` instead, because the registry is written by
the same emit that writes the modules: an import line taken from it is guaranteed to resolve
inside the sandbox. An index doc and the mounted library drift independently, and the failure
mode of that drift is the worst kind — the agent is told to import something that does not
exist, inside a container with no network to check. Search quality is the tradeoff; see
``search_methods`` for where that stops being acceptable.

Nothing here imports the units. Reading a contract must never execute third-party slice code
in the agent process — that is the whole point of the sandbox.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

PACKAGE_NAME = "iguide_methods"
_REGISTRY_NAME = "_registry.json"

# Kept in sync with code_execution.METHOD_LIBRARY_MOUNT. Imported lazily so this module stays
# usable (and testable) without the docker-facing code path.
_DEFAULT_MOUNT = "/opt/iguide_methods"


def library_root() -> Optional[Path]:
    """Host directory containing the ``iguide_methods`` package, or None if nothing is ingested."""
    try:
        from agent_runtime.code_execution import method_library_dir
    except Exception:  # pragma: no cover - only when the exec path is unavailable
        override = (os.getenv("AGENT_METHOD_LIBRARY_DIR") or "").strip()
        return Path(override) if override and Path(override).is_dir() else None
    return method_library_dir()


def registry_path() -> Optional[Path]:
    root = library_root()
    if root is None:
        return None
    path = Path(root) / PACKAGE_NAME / _REGISTRY_NAME
    return path if path.is_file() else None


def load_registry() -> Dict[str, Any]:
    """The symbol -> contract map, or ``{}`` when no library has been built.

    Read fresh on each call rather than cached at import: ingest runs in a different process
    and a long-lived agent server would otherwise serve a registry from before the last
    extraction, reporting "no such method" for units that exist on disk.
    """
    path = registry_path()
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# Dropped from the QUERY before scoring. Symbol names are full of these as connective parts —
# `determine_number_of_cluster`, `load_chicago_community_areas` — so without this filter
# "what is the capital of France" scores a clustering method 6.5 on the word "of" alone.
# Filtering the query (not the symbol) keeps `get_url` findable by "get url".
_QUERY_STOPWORDS = frozenset("""
a an and any are as at be by can do does for from get has have how i if in into is it its me
my of on or that the their there these this to use using want was what when where which who
will with would you your please already instead scratch code data platform notebook notebooks
""".split())


def _tokens(text: str) -> List[str]:
    """Split on non-alphanumerics AND on camelCase/snake_case word boundaries.

    ``load_crime_points`` has to be findable by "crime" and by "load points"; matching the
    symbol as one opaque token would make the most obvious query for a method fail.
    """
    out: List[str] = []
    for raw in _TOKEN_RE.findall(text or ""):
        parts = re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|\d+", raw) or [raw]
        out.extend(p.lower() for p in parts)
        if len(parts) > 1:
            out.append(raw.lower())
    return out


def _entry_text(key: str, entry: Dict[str, Any]) -> Dict[str, List[str]]:
    prov = entry.get("provenance") or {}
    return {
        "symbol": _tokens(str(entry.get("library_symbol") or key)),
        "qualified": _tokens(key),
        "summary": _tokens(str(entry.get("doc_summary") or "")),
        "signature": _tokens(str(entry.get("signature") or "")),
        "element": _tokens(str(prov.get("element_title") or prov.get("element_id") or "")),
    }


# A symbol-name match is the strongest signal a user can give ("call load_crime_points"), and
# a signature match the weakest — parameter names repeat across unrelated methods.
_FIELD_WEIGHTS = {"symbol": 4.0, "qualified": 2.0, "summary": 2.0, "element": 1.0, "signature": 0.5}


def _stem(token: str) -> str:
    """Crude suffix stripping, enough to make a singular and a plural the same token.

    Measured consequence of not having it: the query "buffer geometries by a distance" scored
    ZERO against ``calculate_buffers``, because the symbol tokenises to ``buffers`` while the
    query says ``buffer``. It still ranked first while the library was small — nothing else
    matched either — and fell out of the top ten the moment a corpus containing a literal
    ``MemoryBuffer`` was added. The ranking had been decided by absence of competition rather
    than by relevance, and that is invisible until competition arrives.

    Deliberately not a real stemmer: no new dependency, and over-stemming (``gis`` -> ``gi``)
    would be worse here than under-stemming.
    """
    low = token.lower()
    for suffix in ("ing", "ies", "es", "s"):
        if len(low) > len(suffix) + 3 and low.endswith(suffix):
            return low[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return low


def _idf(registry: Dict[str, Any]) -> Dict[str, float]:
    """Inverse document frequency over the registry's own symbols and summaries.

    Without it every matched token is worth the same, so a unit matching "data" scores like one
    matching "choropleth". At 227 units that was tolerable; at 1,900 the common tokens swamp the
    discriminating ones.
    """
    import math

    df: Dict[str, int] = {}
    total = 0
    for key, entry in registry.items():
        if not isinstance(entry, dict) or entry.get("alias_for"):
            continue
        total += 1
        for token in {_stem(t) for t in _tokens(f"{key} {entry.get('doc_summary') or ''}")}:
            df[token] = df.get(token, 0) + 1
    if not total:
        return {}
    return {t: math.log((total + 1) / (n + 1)) + 1.0 for t, n in df.items()}


def _score(query_tokens: List[str], fields: Dict[str, List[str]],
           idf: Optional[Dict[str, float]] = None) -> float:
    if not query_tokens:
        return 0.0
    total = 0.0
    wanted = {_stem(t) for t in query_tokens}
    for field, weight in _FIELD_WEIGHTS.items():
        present = {_stem(t) for t in (fields.get(field) or ())}
        if not present:
            continue
        for token in wanted & present:
            total += weight * ((idf or {}).get(token, 1.0))
    return total


def search_methods(query: str, *, limit: int = 8,
                   registry: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Rank library methods against a natural-language query.

    Token overlap with field weighting — deliberately simple, and honest about its ceiling:
    it has no IDF and no semantic matching, so at a few hundred units a query whose words do
    not literally appear in a symbol, summary or element title will not find it. That is
    adequate for the current corpus (tens of units) and is the point at which to index
    MethodUnit docs properly and rank there instead. It is *not* a reason to guess: a method
    that cannot be found is a coverage problem, while a wrong import line is a broken run.
    """
    reg = load_registry() if registry is None else registry
    qt = [t for t in _tokens(query) if t not in _QUERY_STOPWORDS]
    if not qt:
        return []
    idf = _idf(reg)
    scored: List[tuple] = []
    for key, entry in reg.items():
        if not isinstance(entry, dict):
            continue
        # Bare aliases duplicate their qualified entry; rank the qualified one so the import
        # line is unambiguous, and let ambiguous stubs through so a collision is visible
        # rather than silently absent.
        if entry.get("alias_for"):
            continue
        if entry.get("ambiguous"):
            score = _score(qt, {"symbol": _tokens(str(entry.get("library_symbol") or key)),
                                "qualified": _tokens(key), "summary": [], "signature": [],
                                "element": []}, idf)
            if score > 0:
                scored.append((score, key, entry))
            continue
        score = _score(qt, _entry_text(key, entry), idf)
        if score > 0:
            scored.append((score, key, entry))

    scored.sort(key=lambda row: (-row[0], row[1]))
    return [_summarize(key, entry, score) for score, key, entry in scored[:max(1, int(limit))]]


def _summarize(key: str, entry: Dict[str, Any], score: float) -> Dict[str, Any]:
    if entry.get("ambiguous"):
        return {"symbol": key, "ambiguous": True,
                "candidates": entry.get("candidates") or [],
                "doc_summary": entry.get("doc_summary"), "score": round(score, 2)}
    prov = entry.get("provenance") or {}
    # Invariants are included in the SEARCH result, not only in get_method_contract. Observed:
    # an agent found `calculate_buffers`, never called get_method_contract, and so never saw
    # that the method requires a projected CRS — it happened to reproject anyway and then
    # described that as satisfying the invariant, which is the right action for the wrong
    # reason. A requirement the caller must satisfy belongs where the caller first sees the
    # method; a second tool call it may not make is not a reliable channel.
    invariants = [i for i in (entry.get("invariants") or []) if isinstance(i, dict)]
    return {
        "symbol": key,
        # Surfaced explicitly, not left implicit in the signature text: constructing a class and
        # calling a function are different actions, and the caller decides which to write.
        "unit_kind": entry.get("unit_kind"),
        "signature": entry.get("signature"),
        "doc_summary": entry.get("doc_summary"),
        "import_line": import_line(entry),
        "element_id": prov.get("element_id"),
        "slice_sha": entry.get("slice_sha"),
        "requirements": (entry.get("requirements") or {}).get("pip") or [],
        "requires": [f"{i.get('check')}({i.get('target')})" for i in invariants] or None,
        "score": round(score, 2),
    }


def import_line(entry: Dict[str, Any]) -> Optional[str]:
    """The exact, version-pinned import for a unit.

    Pinned to the ``v_<slice_sha>`` module rather than the element package's re-export, so a
    run that is re-executed after a re-ingest imports the *same* code it was verified against.
    The element package alias keeps working; it just is not what an artifact should record.
    """
    module, symbol = entry.get("module"), entry.get("library_symbol")
    if not module or not symbol:
        return None
    return f"from {module} import {symbol}"


def get_contract(symbol: str, *, registry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Full contract for one symbol, by bare or qualified name.

    An ambiguous bare name returns the candidate list and NO import line — the same refusal
    ``iguide_methods.get()`` makes at runtime. Resolving it here to "whichever was ingested
    last" would hand the agent a confidently wrong answer.
    """
    reg = load_registry() if registry is None else registry
    name = (symbol or "").strip()
    if not name:
        return {"error": "empty symbol"}

    entry = reg.get(name)
    if entry is None:
        suffix = f".{name}"
        matches = sorted(k for k in reg if k.endswith(suffix))
        if len(matches) == 1:
            name, entry = matches[0], reg[matches[0]]
        elif matches:
            return {"symbol": name, "ambiguous": True, "candidates": matches,
                    "error": f"{name!r} is defined by more than one element; "
                             f"ask again with a qualified name."}
        else:
            return {"symbol": name, "error": f"no method named {name!r} in the library",
                    "available": len([k for k, v in reg.items()
                                      if isinstance(v, dict) and not v.get("alias_for")])}

    if not isinstance(entry, dict):
        return {"symbol": name, "error": "malformed registry entry"}
    if entry.get("ambiguous"):
        return {"symbol": name, "ambiguous": True,
                "candidates": entry.get("candidates") or [],
                "error": f"{name!r} is defined by more than one element; "
                         f"ask again with a qualified name."}

    resolved = entry if not entry.get("alias_for") else reg.get(entry["alias_for"], entry)
    prov = resolved.get("provenance") or {}
    return {
        "symbol": entry.get("alias_for") or name,
        "unit_kind": resolved.get("unit_kind"),
        "signature": resolved.get("signature"),
        "doc_summary": resolved.get("doc_summary"),
        "params": resolved.get("params") or [],
        "returns": resolved.get("returns"),
        "invariants": resolved.get("invariants") or [],
        "requirements": resolved.get("requirements") or {},
        "import_line": import_line(resolved),
        "module": resolved.get("module"),
        "slice_sha": resolved.get("slice_sha"),
        "provenance": prov,
        "mounted_at": _DEFAULT_MOUNT,
    }


def library_summary() -> Dict[str, Any]:
    """Counts for observability — what the agent can actually reach right now."""
    reg = load_registry()
    units = [v for k, v in reg.items()
             if isinstance(v, dict) and not v.get("alias_for") and not v.get("ambiguous")]
    ambiguous = [k for k, v in reg.items() if isinstance(v, dict) and v.get("ambiguous")]
    elements = {(v.get("provenance") or {}).get("element_id") for v in units}
    kinds: Dict[str, int] = {}
    for v in units:
        kinds[str(v.get("unit_kind") or "unknown")] = \
            kinds.get(str(v.get("unit_kind") or "unknown"), 0) + 1
    return {"units": len(units), "elements": len(elements - {None}), "kinds": kinds,
            "ambiguous_names": sorted(ambiguous), "registry_entries": len(reg),
            "root": str(library_root() or "")}


__all__ = ["library_root", "registry_path", "load_registry", "search_methods",
           "get_contract", "import_line", "library_summary", "PACKAGE_NAME"]
