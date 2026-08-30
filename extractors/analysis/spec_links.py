"""Linking a publication's method spec to the element that implements it.

The existing ``IMPLEMENTED_BY`` matches a spec's ``tools_referenced`` against library symbol
names. Measured over the corpus it produces **one** edge, and the ceiling is structural rather
than a tuning problem: ``tools_referenced`` records what SOFTWARE a paper used — "OSMnx",
"Python", "Jupyter Notebook", "Census Bureau API" — while a method library is indexed by
FUNCTION. Papers name libraries; libraries contain functions, and the two vocabularies barely
intersect.

So the link is drawn at the ELEMENT level, from evidence a human left behind:

**Citation** — the notebook names the paper's DOI in its own text. Somebody wrote that down on
purpose, so it is the strongest signal available and it needs no corroboration. Measured: 2
pairs across the corpus. Rare, and right when present.

**Author + topic** — the paper and the element share a person, AND the element's own text scores
against the spec. Neither half works alone. Author overlap by itself gave one spec **seventeen**
candidates, including "CyberGIS-Compute Core" and "Data Collection": a prolific author links a
paper to everything they ever touched, which is a co-authorship graph, not an implementation
link. Topic by itself would link any two papers about accessibility. Together they are
discriminating — on the E2SFCA case the real implementation scores 0.736 against 0.443 for the
next candidate and 0.010 for the co-authored bystander.

Nothing here does I/O. The caller supplies the author index and a scorer, which is what makes
the ranking testable without a corpus, a network or a database.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

# The MARGIN between the best candidate and the runner-up, which is what distinguishes "this
# one" from "one of these". Measured on the corpus: `355786a5` scores 0.7364 then 0.443 — the
# top candidate stands apart, and it is the right one. `a182e493` scores 0.7011, 0.6971 and
# 0.6254 — three candidates within 11%, where the author signal is not discriminating and the
# topical score cannot break the tie either. A relative floor admitted all three; a margin
# admits neither the second nor the first.
TOPIC_MARGIN_RATIO = 0.80
# Even a lone candidate has to clear something, or a spec whose only co-author wrote something
# unrelated links to it for lack of competition.
TOPIC_MIN = 0.05
# One. A publication describes one method; a second "implementation" is a claim the evidence
# here cannot support, and the cost of a wrong IMPLEMENTED_BY edge is a fabricated provenance
# claim rather than a missing one.
MAX_LINKS_PER_SPEC = 1

_AND_PREFIX = re.compile(r"^\s*(and|&)\s+", re.I)
_PUNCT = re.compile(r"[^\w\s.-]")
_DOI = re.compile(r"10\.\d{4,9}/[^\s\"'<>)\]}]+", re.I)
_DOI_URL = re.compile(r"^https?://(?:dx\.)?doi\.org/", re.I)


def author_key(name: Any) -> str:
    """One person's name, comparable across two records that spell it differently.

    "and Shaowen Wang" and "Shaowen Wang" are the same person — the platform's author lists
    carry the conjunction into the last entry. Case, punctuation and repeated spaces go too.
    """
    text = _AND_PREFIX.sub("", str(name or "").strip())
    text = _PUNCT.sub("", text).strip().lower()
    return re.sub(r"\s+", " ", text)


def normalize_doi(value: Any) -> str:
    """A DOI comparable whether it arrived bare or as a doi.org URL."""
    text = str(value or "").strip().lower().rstrip(".,);]")
    return _DOI_URL.sub("", text)


def dois_in(text: Any) -> List[str]:
    """Every DOI written into a body of text, normalised."""
    return [normalize_doi(m) for m in _DOI.findall(str(text or ""))]


@dataclass
class SpecLink:
    """One proposed implementation link, carrying WHY it was drawn."""

    spec_element: str
    target_element: str
    confidence: str                    # high | medium
    evidence: str                      # cited_doi | shared_author_and_topic
    detail: Dict[str, Any] = field(default_factory=dict)

    def as_edge(self) -> Dict[str, Any]:
        """The ``ProvenanceEdge`` shape, so this joins the existing vocabulary rather than
        inventing a second one."""
        return {"src": f"{self.spec_element}::methodspec", "rel": "IMPLEMENTED_BY",
                "dst": self.target_element,
                "detail": {"confidence": self.confidence, "by": self.evidence, **self.detail}}


def links_from_citation(spec_element: str, spec_doi: str,
                        element_texts: Dict[str, str]) -> List[SpecLink]:
    """Elements whose own text names this paper's DOI.

    No corroboration required, and deliberately so: a DOI in a notebook is not a coincidence of
    vocabulary, it is a person writing down which paper this implements.
    """
    doi = normalize_doi(spec_doi)
    if not doi:
        return []
    out = []
    for element, text in (element_texts or {}).items():
        if element == spec_element:
            continue
        if doi in dois_in(text):
            out.append(SpecLink(spec_element, element, "high", "cited_doi", {"doi": doi}))
    return out


def links_from_authors(spec_element: str,
                       spec_authors: Sequence[str],
                       candidates_by_author: Dict[str, Iterable[str]],
                       score: Callable[[str], float],
                       *,
                       margin_ratio: float = TOPIC_MARGIN_RATIO,
                       minimum: float = TOPIC_MIN,
                       limit: int = MAX_LINKS_PER_SPEC) -> List[SpecLink]:
    """Shared-author candidates, ranked by topic and cut at a relative floor.

    ``score`` takes a candidate element id and returns how well that element's own text matches
    the spec. It is injected rather than computed here so this module stays pure — the real
    scorer is a Postgres full-text rank, and a test can pass a dictionary.
    """
    keys = {author_key(a) for a in (spec_authors or []) if author_key(a)}
    # Which of the spec's authors led to each candidate — recorded, because an edge claiming
    # "shared author" without naming the person is not checkable by whoever reads it later.
    via: Dict[str, set] = {}
    for key in keys:
        for candidate in (candidates_by_author.get(key) or ()):
            if candidate and candidate != spec_element:
                via.setdefault(candidate, set()).add(key)
    candidates = set(via)
    if not candidates:
        return []

    scored = sorted(((float(score(c) or 0.0), c) for c in candidates), reverse=True)
    best = scored[0][0]
    if best < minimum:
        # Every candidate is irrelevant. Emitting the least irrelevant one is the failure this
        # floor exists to prevent.
        return []

    # The runner-up decides whether the leader means anything. Clustered scores are the shape of
    # "several of this author's notebooks are about this topic", which is not an implementation.
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    if runner_up > best * margin_ratio:
        return []

    out = []
    for value, element in scored[:limit]:
        out.append(SpecLink(
            spec_element, element, "medium", "shared_author_and_topic",
            {"topic_score": round(value, 4), "topic_rank": len(out) + 1,
             "best_score": round(best, 4), "runner_up_score": round(runner_up, 4),
             "shared_authors": sorted(via.get(element, ()))[:6],
             "candidates_considered": len(candidates)}))
    return out


def merge(*groups: Iterable[SpecLink]) -> List[SpecLink]:
    """One link per (spec, target), keeping the strongest evidence.

    A pair found by BOTH a cited DOI and author overlap is one link, not two, and it is the
    citation that should be reported — an edge carrying its weakest justification understates
    what is known about it.
    """
    rank = {"high": 2, "medium": 1}
    best: Dict[tuple, SpecLink] = {}
    for link in (l for group in groups for l in group):
        key = (link.spec_element, link.target_element)
        current = best.get(key)
        if current is None or rank.get(link.confidence, 0) > rank.get(current.confidence, 0):
            best[key] = link
    return sorted(best.values(),
                  key=lambda l: (-rank.get(l.confidence, 0), l.spec_element, l.target_element))


__all__ = ["SpecLink", "author_key", "normalize_doi", "dois_in", "links_from_citation",
           "links_from_authors", "merge", "TOPIC_MARGIN_RATIO", "TOPIC_MIN",
           "MAX_LINKS_PER_SPEC"]
