"""Platform element references written into a notebook's own text.

Authors cite other knowledge elements by URL — in a markdown cell explaining where the input
came from, or in the `wget`/`requests.get` line that actually downloads it. Measured across the
174 cached corpus notebooks: **179 i-guide.io URLs, 100 of them carrying an element UUID, from
30 notebooks to 41 distinct targets, and 41 of 41 targets resolve to a live platform element.**

That matters because it is the only *cross-element* signal extraction can see. Every other edge
the extractors emit is parent-to-child within one element: measured over the same 174 notebooks,
4,071 of 4,212 edges satisfy ``dst.startswith(src + "::")`` and are therefore a pure function of
the id, and ``cross_element`` is **0**. A citation is not derivable from any id — it exists only
because a human wrote it down.

Two hosts appear, and they do NOT mean the same thing, so they do not produce the same edge:

``platform.i-guide.io/<type>/<uuid>``
    A link to the element's page. The author is pointing at it — background, prior work, "this
    notebook builds on that one". That is ``CITES``.

``storage.i-guide.io/<type>/<uuid>/<filename>``
    The object store. A URL of this shape in a notebook is a *download*, so the notebook does
    not merely reference that element, it consumes it. That is ``USES``, and it is the first
    ``USES`` edge in this codebase whose ``dst`` is a real node: the existing one at
    ``publication_extractor.py:401`` sets ``dst=str(ds)`` from an LLM-named dataset string, which
    nothing anywhere resolves to an element id.

Deliberately NOT captured, because a UUID is required and none of these carry one:
``i-guide.io/projects/…`` and ``/spatial-ai-challenge-2024/`` (13 + 3 hits, marketing pages),
``backend.i-guide.io/user-uploads/thumbnails/…`` (10 hits, images), and the f-string template
``platform.i-guide.io/{element_type`` (1 hit, code that BUILDS a URL rather than naming one).

The path segment (``notebooks``, ``datasets``, ``oers``) is recorded as a hint and is never
asserted to be the element's type. Resolving a UUID to its real ``resource-type`` needs the
platform API, and this module is pure — it runs inside extraction, which is per-element and
offline. The corpus-level pass is what validates.
"""

from __future__ import annotations

import re
from typing import List, NamedTuple

__all__ = ["Citation", "REL_BY_HOST", "platform_citations"]

# The full 8-4-4-4-12 form only. A shorter hex run would match commit shas, object keys and the
# `1720206819181-spastc_scaled.jpg` in thumbnail URLs; requiring the full shape is what keeps
# the marketing and image URLs out without needing a host denylist.
_UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"

_CITATION_RE = re.compile(
    r"https?://(?P<host>platform|storage)\.i-guide\.io/(?P<hint>[A-Za-z][A-Za-z0-9_-]*)/"
    r"(?P<uuid>" + _UUID + r")",
    re.IGNORECASE,
)

# Which relationship each host implies. `rel` values are from the vocabulary already declared in
# base.py:66 — this module adds CITES, and reuses USES rather than inventing a synonym.
REL_BY_HOST = {"platform": "CITES", "storage": "USES"}


class Citation(NamedTuple):
    """One reference from a notebook to a platform element.

    A NamedTuple so callers can drop them in a set: the same element cited from four cells is
    one edge, and de-duplication should not need a key function.
    """

    element_id: str      # the cited element's platform UUID, lowercased
    rel: str             # CITES | USES
    path_hint: str       # the URL's own path segment ('notebooks'), a HINT, not the type
    host: str            # platform | storage


def platform_citations(text: str) -> List[Citation]:
    """Every platform element reference in ``text``, in order of appearance, de-duplicated.

    Order is preserved rather than returning a set, because the first mention of an element in a
    notebook is usually the one with the explanatory prose around it, and a caller that wants to
    quote evidence wants that one.

    Matching is anchored on the URL scheme and stops at the UUID, so surrounding junk needs no
    stripping — the corpus contains ``…zip</td``, a trailing backtick and a trailing period on
    otherwise valid URLs, and all three end up outside the match.
    """
    if not text:
        return []
    out: List[Citation] = []
    seen: set = set()
    for m in _CITATION_RE.finditer(text):
        cite = Citation(
            element_id=m.group("uuid").lower(),
            rel=REL_BY_HOST[m.group("host").lower()],
            path_hint=m.group("hint").lower(),
            host=m.group("host").lower(),
        )
        key = (cite.element_id, cite.rel)
        if key in seen:
            continue
        seen.add(key)
        out.append(cite)
    return out
