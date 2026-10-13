"""Resolve the agent KB's short element ids to platform element ids.

The agent KB keys every element by the first 8 characters of its platform UUID (``05269a1a``
for ``05269a1a-ce6e-4e87-8b43-74490a986c09``). Measured on the VM's Postgres record on
2026-10-13: all 543 ``element.id`` values are 8 characters long, and no column holds the full
id (``source_url`` and ``fields`` are empty on every row). The prefix is enough to find a KB
block, but not to link the element: ``platform.i-guide.io/notebooks/05269a1a`` is not a page,
so every agent-KB source reached "Sources used" as plain text and was cited by an id the
platform does not know.

The platform graph has the full id on every knowledge element, so the prefix is resolved there:
one batched ``UNWIND … STARTS WITH`` per search, cached per process. Measured the same day: all
543 prefixes resolved, none matched two elements, in about half a second for the whole set.
The OpenSearch index was the other candidate and is worse: ``_id`` refuses prefix queries, and
the ``id`` keyword field is missing on 131 of its 763 documents, ``05269a1a…`` among them.

A prefix that matches more than one element resolves to NOTHING: a guessed link to the wrong
element is worse than no link. The element's type comes from the graph label as well, because
the KB's own type disagreed with the platform's on 15 of the 543 elements (the KB says
``dataset``, the platform serves them as publications).
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

from .utils import get_logger, thumbnail_of

log = get_logger("element_ids")

_FULL_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_PREFIX = re.compile(r"^[0-9a-f]{8}$", re.I)

# Element ids do not change, but an element can be added after a miss was cached; an hour bounds
# how long a newly published element stays unlinked.
_CACHE_TTL_S = 3600.0
_CACHE: Dict[str, tuple] = {}     # prefix -> (stored_at, resolution or None)

_CYPHER = """
UNWIND $prefixes AS p
MATCH (r)
WHERE any(l IN labels(r) WHERE l IN $labels) AND r.id STARTS WITH p
RETURN p AS prefix, r.id AS id, labels(r) AS labels, r.thumbnail_image AS thumbnail
"""


def is_full_element_id(value: Any) -> bool:
    return bool(_FULL_ID.match(str(value or "").strip()))


def _default_run(cypher: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    from .neo4j import _neo4j_run, neo4j_enabled

    if not neo4j_enabled():
        return []
    return _neo4j_run(cypher, params)


def resolve_element_prefixes(
    prefixes: Iterable[str],
    *,
    run: Optional[Callable[[str, Dict[str, Any]], List[Dict[str, Any]]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """``{prefix: {"id": full_uuid, "element_type": "notebook"}}`` for each prefix that names
    exactly one platform element. Ambiguous and unknown prefixes are absent from the result.

    Never raises: a graph outage means no links this turn, not a failed search.
    """
    from .neo4j_graph_tools import _get_resource_labels, element_type_from_labels

    wanted = sorted({str(p).strip().lower() for p in prefixes if _PREFIX.match(str(p or "").strip())})
    now = time.monotonic()
    out: Dict[str, Dict[str, Any]] = {}
    missing: List[str] = []
    for p in wanted:
        cached = _CACHE.get(p)
        if cached and now - cached[0] < _CACHE_TTL_S:
            if cached[1]:
                out[p] = cached[1]
        else:
            missing.append(p)
    if not missing:
        return out
    try:
        rows = (run or _default_run)(_CYPHER, {"prefixes": missing,
                                              "labels": sorted(_get_resource_labels())})
    except Exception as exc:  # noqa: BLE001 - a link is never worth failing a search over
        log.warning("element id resolution failed: %s", exc)
        return out
    found: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows or []:
        prefix, full = str(row.get("prefix") or ""), str(row.get("id") or "")
        if not full.lower().startswith(prefix) or not is_full_element_id(full):
            continue
        resolution = {"id": full, "element_type": element_type_from_labels(row.get("labels"))}
        thumb = thumbnail_of({"thumbnail_image": row.get("thumbnail")})
        if thumb:
            resolution["thumbnail_image"] = thumb
        found.setdefault(prefix, {})[full] = resolution
    for p in missing:
        matches = list((found.get(p) or {}).values())
        if len(matches) > 1:
            log.warning("element id prefix %s matches %d elements; not linking it", p, len(matches))
        resolution = matches[0] if len(matches) == 1 and matches[0]["element_type"] else None
        _CACHE[p] = (now, resolution)
        if resolution:
            out[p] = resolution
    return out


def platform_url(element_type: str, element_id: str) -> str:
    """The platform page of an element, by the same rule as ``_element_url``."""
    from agent_runtime.supervisor.evidence_subgraph import _element_url

    return _element_url({"element_type": element_type, "doc_id": element_id})


def link_kb_documents(docs: List[Dict[str, Any]], elements: Dict[str, Dict[str, Any]], *,
                      run: Optional[Callable[[str, Dict[str, Any]], List[Dict[str, Any]]]] = None,
                      ) -> List[str]:
    """Give each agent-KB hit its parent element's platform id and link, in place.

    Sets ``element_id`` (the full UUID), ``url`` and, when the element has one,
    ``thumbnail_image`` on each hit whose ``parent_doc_id`` resolves, and the same on the
    matching ``elements`` entry (as ``platform_id``/``url``/``thumbnail_image``). ``parent_doc_id``
    and ``doc_id`` are left as the KB's own keys, because ``get_kb_block`` and
    ``kb_for_elements`` look blocks up by them. Returns the citation ids: the full id where one
    was resolved, else the KB's id unchanged.
    """
    parents = [str(d.get("parent_doc_id") or "") for d in docs]
    resolved = resolve_element_prefixes([p for p in parents if not is_full_element_id(p)], run=run)
    citations: List[str] = []
    for d, parent in zip(docs, parents):
        hit = resolved.get(parent.lower())
        element = d.get("element")
        if not hit and is_full_element_id(parent) and isinstance(element, dict):
            # Already a platform id (an index built with full ids): only the type is needed.
            etype = str(element.get("resource-type") or "").strip().lower()
            hit = {"id": parent, "element_type": etype} if etype else None
        if hit:
            url = platform_url(hit["element_type"], hit["id"])
            d["element_id"] = hit["id"]
            if url:
                d["url"] = url
            if hit.get("thumbnail_image"):
                d["thumbnail_image"] = hit["thumbnail_image"]
            if isinstance(element, dict):
                element["platform_id"] = hit["id"]
                element["platform_type"] = hit["element_type"]
                if url:
                    element["url"] = url
                if hit.get("thumbnail_image"):
                    element["thumbnail_image"] = hit["thumbnail_image"]
            citations.append(hit["id"])
        else:
            citations.append(parent)
    return citations


def _reset_cache_for_tests() -> None:
    _CACHE.clear()


__all__ = ["is_full_element_id", "link_kb_documents", "platform_url", "resolve_element_prefixes"]
