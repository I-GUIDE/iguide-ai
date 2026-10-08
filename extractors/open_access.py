"""Resolve a DOI to an open-access PDF, so a publication's METHOD can be read.

Publications were the largest element type (203) contributing zero method specs, and the reason
was never the extractor: ``external_link`` is a DOI or a publisher landing page for 198 of 200,
not a PDF, so there was no text to extract from. 143 of those resolve through ``doi.org``.

Unpaywall is the route. It is a free, no-key API over the same DOIs, run by OurResearch, and it
returns the *legally open* copy when one exists — publisher OA, a repository deposit, or an
author manuscript. Using it means the corpus only ever ingests text the publisher or author put
in the open, which matters because the platform's own licence fields are empty or null for most
elements and "we could fetch it" is not the same as "we may use it".

Roughly half of any given corpus has no open copy. That is reported as a typed outcome rather
than a failure, because "this paper is paywalled" is a fact about the world and a re-run will
not change it — quite unlike "the network was down".
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

UNPAYWALL = "https://api.unpaywall.org/v2/{doi}"

_DOI_RE = re.compile(r"(10\.\d{4,9}/[^\s\"'<>&]+)", re.I)


@dataclass
class OpenAccessResult:
    """What was found for one DOI. ``pdf_url`` is None unless an open copy exists."""

    doi: str
    pdf_url: Optional[str] = None
    landing_url: Optional[str] = None
    is_oa: bool = False
    licence: Optional[str] = None
    version: Optional[str] = None
    host_type: Optional[str] = None
    reason: str = ""
    # Which of Unpaywall's locations the pdf_url came from: "best" or "alternate". A repository
    # copy is a different artifact from the published version and provenance should say so.
    pdf_from: str = ""


def extract_doi(text: Any) -> Optional[str]:
    """The DOI inside a link or citation string, or None.

    Handles ``https://doi.org/10.x/y``, a bare ``10.x/y``, and a publisher URL that embeds one
    (``iopscience.iop.org/article/10.1088/...``) — 34 of the corpus's publication links are the
    third shape, so matching only ``doi.org`` would leave those unresolved.
    """
    match = _DOI_RE.search(str(text or ""))
    if not match:
        return None
    doi = match.group(1).rstrip(").,;")
    # A trailing .pdf or .html belongs to the URL, not the DOI.
    for suffix in (".pdf", ".html", ".xml", ".full"):
        if doi.lower().endswith(suffix):
            doi = doi[: -len(suffix)]
    return doi


def _email() -> str:
    """Unpaywall requires a contact address; it is a courtesy identifier, not a credential."""
    return (os.getenv("UNPAYWALL_EMAIL") or os.getenv("CONTACT_EMAIL")
            or "iguide-agent@illinois.edu").strip()


def resolve(doi: str, *, timeout: int = 30, session: Any = None) -> OpenAccessResult:
    """Ask Unpaywall for an open copy of *doi*."""
    import requests

    getter = session or requests
    result = OpenAccessResult(doi=doi)
    try:
        resp = getter.get(UNPAYWALL.format(doi=doi), params={"email": _email()},
                          timeout=timeout)
    except Exception as exc:
        result.reason = f"{type(exc).__name__}: {exc}"[:120]
        return result

    if resp.status_code == 404:
        result.reason = "not in unpaywall (not a registered crossref DOI)"
        return result
    if resp.status_code != 200:
        result.reason = f"unpaywall HTTP {resp.status_code}"
        return result
    try:
        data: Dict[str, Any] = resp.json()
    except ValueError:
        result.reason = "unpaywall returned non-JSON"
        return result

    result.is_oa = bool(data.get("is_oa"))
    best = data.get("best_oa_location") or {}
    if not result.is_oa or not best:
        result.reason = "no open-access copy exists"
        return result

    result.pdf_url = best.get("url_for_pdf") or None
    result.landing_url = best.get("url_for_landing_page") or best.get("url")
    result.licence = best.get("license")
    result.version = best.get("version")
    result.host_type = best.get("host_type")
    if result.pdf_url:
        result.pdf_from = "best"
        return result

    # `best_oa_location` is Unpaywall's own ranking, which prefers the PUBLISHED version -- and a
    # published version behind a landing page often has no direct PDF while a repository deposit
    # of the same paper does. `oa_locations` holds them all, so look before giving up.
    #
    # Measured over the corpus's 177 DOIs: 39 came back "open access, but no direct PDF url" from
    # `best_oa_location` alone. This is the cheapest recovery available on the type, because it
    # costs no extra request -- the alternates are in the response already.
    for location in (data.get("oa_locations") or []):
        if not isinstance(location, dict):
            continue
        candidate = location.get("url_for_pdf")
        if candidate:
            result.pdf_url = candidate
            result.pdf_from = "alternate"
            result.licence = location.get("license") or result.licence
            result.version = location.get("version") or result.version
            result.host_type = location.get("host_type") or result.host_type
            result.landing_url = (location.get("url_for_landing_page")
                                  or result.landing_url)
            return result

    # Still nothing directly downloadable. The landing URL remains, and the reader added in M8.10
    # routes on content rather than extension, so a full-text HTML article is now readable. Not a
    # fetch failure -- a different shape of the same open copy.
    result.reason = ("open access as a landing page only; no PDF at any of "
                     f"{len(data.get('oa_locations') or [])} open location(s)")
    return result


def resolve_many(dois, *, pause: float = 0.15, timeout: int = 30) -> Dict[str, OpenAccessResult]:
    """Resolve a batch, politely. Unpaywall asks for ~100k/day and no burst hammering."""
    import requests

    out: Dict[str, OpenAccessResult] = {}
    with requests.Session() as session:
        for doi in dois:
            if not doi or doi in out:
                continue
            out[doi] = resolve(doi, timeout=timeout, session=session)
            time.sleep(pause)
    return out


__all__ = ["extract_doi", "resolve", "resolve_many", "OpenAccessResult"]
