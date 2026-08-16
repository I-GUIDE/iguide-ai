"""Is this element's code worth putting in the method library?

Callability answers "does this run standing alone". It says nothing about whether the code is
reusable *domain* knowledge, and the two came apart badly when the code corpus was first swept:
36 elements produced 1,922 callable units, of which 17% looked geospatial and 3% carried a
contract. The top contributors were ML/RAG experiment repos — ``run_demo.py`` gave 222 units,
``infer.py`` 204 — whose exports are internal plumbing: ``pad_to_match(up, skip)``,
``arg_dict_to_str(args)``, ``collect_evidence(evidence, subtrees)``.

The measured cost of admitting them: ``calculate_buffers`` fell from rank 1 to outside the top
ten for "buffer geometries by a distance", beaten by a ``MemoryBuffer`` that merely contains the
token. A library that answers a nonsense query with ten confident results is worse than a
smaller one that answers it with none.

**The signal is the README**, not the function names. A repo's README is a human explaining what
the repo is for, written before anyone thought about extraction — so it is both the most honest
description available and the one the authors of noise never wrote about geospatial analysis.
Scoring the README rejects a whole repository in one decision, which is the right granularity:
these repos are uniformly irrelevant, not a mix.

Every score records its evidence. A rejected element must be auditable, or the library's
coverage becomes a matter of opinion.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

# Domain vocabulary, weighted by how strongly each term implies geospatial ANALYSIS rather than
# an incidental mention. "geopandas" in a README is near-proof; "data" is worthless.
_STRONG = (
    "geospatial", "geopandas", "rasterio", "shapely", "postgis", "arcgis", "qgis", "gdal",
    "cartopy", "xarray", "netcdf", "shapefile", "geotiff", "choropleth", "geocod",
    "coordinate reference", "spatial analysis", "spatial statistic", "remote sensing",
    "watershed", "hydrolog", "floodplain", "land cover", "land use", "digital elevation",
    "accessibility", "isochrone", "catchment", "census tract", "openstreetmap", "osmnx",
    "spatial join", "map projection", "epsg", "latitude", "longitude",
)
_MEDIUM = (
    "spatial", "geographic", "raster", "vector data", "basemap", "cartograph", "gis",
    "satellite", "lidar", "elevation", "terrain", "climate", "weather", "hydro", "flood",
    "urban", "mobility", "trajectory", "region", "boundary", "parcel", "zoning",
    "population density", "epidemiolog", "environment",
)
# Terms that mark a repo as something else entirely. Not disqualifying on their own — a
# geospatial deep-learning repo is legitimate — but they cancel a weak positive.
_OFF_DOMAIN = (
    "language model", "llm", "retrieval-augmented", "rag ", "chatbot", "prompt engineering",
    "transformer architecture", "fine-tun", "benchmark suite", "leaderboard",
    "reinforcement learning", "recommendation system", "sentiment analysis",
)

STRONG_WEIGHT, MEDIUM_WEIGHT, OFF_WEIGHT = 3.0, 1.0, -2.0
DEFAULT_THRESHOLD = 3.0


def _hits(text: str, terms: Tuple[str, ...]) -> List[str]:
    return sorted({t for t in terms if t in text})


def score_element(record: Dict[str, Any]) -> Dict[str, Any]:
    """Domain-relevance score for one element record, with the evidence that produced it.

    Reads the README first and falls back to the description. Both are prose written for humans;
    neither is the function names, which is the point — the names are what misled the raw sweep.
    """
    readme = str(record.get("github_repo_readme") or record.get("github-repo-readme") or "")
    parts = [readme,
             str(record.get("title") or ""),
             str(record.get("contents") or ""),
             " ".join(str(t) for t in (record.get("tags") or []))]
    text = re.sub(r"\s+", " ", " ".join(parts)).lower()

    strong = _hits(text, _STRONG)
    medium = _hits(text, _MEDIUM)
    off = _hits(text, _OFF_DOMAIN)
    score = (STRONG_WEIGHT * len(strong) + MEDIUM_WEIGHT * len(medium)
             + OFF_WEIGHT * len(off))
    return {
        "score": round(score, 2),
        "strong": strong[:6],
        "medium": medium[:6],
        "off_domain": off[:4],
        "has_readme": bool(readme.strip()),
        "chars": len(text),
    }


def is_relevant(record: Dict[str, Any], threshold: float = DEFAULT_THRESHOLD) -> bool:
    """Whether to extract library units from this element.

    An element with NO readable prose at all is admitted rather than rejected: the absence of
    evidence is not evidence of irrelevance, and rejecting on it would silently drop elements
    for having thin metadata. The unit-level analyzer still gates what those elements produce.
    """
    verdict = score_element(record)
    if not verdict["has_readme"] and verdict["chars"] < 120:
        return True
    return verdict["score"] >= threshold


__all__ = ["score_element", "is_relevant", "DEFAULT_THRESHOLD"]
