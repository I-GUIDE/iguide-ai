"""Turn a dataset repository's LANDING page into the file URLs behind it.

64 of the corpus's 130 datasets resolve to HTML rather than data. They are not one problem but
two, and conflating them is why they all read as "unfetchable":

**A deposit behind a landing page** (~36). figshare, Zenodo, HydroShare, Dataverse, Hugging Face
and GitHub all serve a human page at the URL the platform records, and expose the actual files
through a documented API. Resolvable.

**A portal's front door** (~28). ``fao.org/faostat/en/#data``, ``who.int/data/#data``,
``worldbank.org/indicator``, ``mrlc.gov/data`` — the element is a pointer to a *website*, not to
a deposited dataset. There is no file, and there never was. Reporting these as failed downloads
invites someone to go looking for a bug; ``classify_link`` names them instead.

Every resolver returns a list of ``{name, url, bytes}`` and an empty list on anything unexpected.
None of them guesses a URL pattern: each reads a documented listing endpoint, because a
constructed download link that 404s is indistinguishable from a dataset that has been withdrawn.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# Hosts whose landing page hides a deposit reachable by API.
DEPOSIT_HOSTS = ("figshare.com", "zenodo.org", "hydroshare.org", "dataverse.harvard.edu",
                 "databank.illinois.edu", "huggingface.co", "github.com", "data.msdlive.org")

# Hosts that are a data PORTAL rather than a deposit: the link is a front door, and the useful
# content is behind a search UI, a login, or a per-country picker.
PORTAL_HOSTS = ("fao.org", "who.int", "worldbank.org", "iea.org", "mrlc.gov", "mtbs.gov",
                "wri.org", "iucnredlist.org", "chc.ucsb.edu", "jrc.ec.europa.eu",
                "climateaction.unfccc.int", "disasters.geoplatform.gov", "wildfirerisk.org",
                "routesdashboard.org", "wildlifetradeportal.org", "ucdp.uu.se",
                "nid.sec.usace.army.mil", "developers.google.com", "mygeohub.org")

# Hosts with no stable public listing API. Distinguished from a portal: there IS a file, we just
# cannot enumerate it without a browser session.
OPAQUE_HOSTS = ("drive.google.com", "dropbox.com", "sedac.ciesin.columbia.edu")


def classify_link(url: str) -> str:
    """``deposit`` | ``portal`` | ``opaque`` | ``direct`` — what kind of thing this URL is.

    The distinction earns its keep at index time: a portal element should be searchable as a
    pointer to a data source, and a deposit element should be searchable by its files' schema.
    Treating both as "download failed" loses the first entirely.
    """
    lowered = (url or "").lower()
    if not lowered.startswith(("http://", "https://")):
        return "direct"
    host = re.sub(r"^https?://", "", lowered).split("/")[0]
    for group, label in ((PORTAL_HOSTS, "portal"), (OPAQUE_HOSTS, "opaque"),
                         (DEPOSIT_HOSTS, "deposit")):
        if any(host == h or host.endswith("." + h) for h in group):
            return label
    return "direct"


def _json(url: str, session: Any, timeout: int = 45) -> Optional[Any]:
    try:
        resp = session.get(url, timeout=timeout, allow_redirects=True,
                           headers={"Accept": "application/json"})
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


def _figshare(url: str, session: Any) -> List[Dict[str, Any]]:
    """figshare: an article id in the path, then /v2/articles/{id}/files."""
    match = re.search(r"figshare\.com/(?:articles/[^/]+/[^/]+/(\d+)|ndownloader/files/(\d+))", url)
    if not match:
        match = re.search(r"figshare\.com/.*?/(\d{6,})", url)
    if not match:
        return []
    article = next(g for g in match.groups() if g)
    data = _json(f"https://api.figshare.com/v2/articles/{article}/files", session)
    if not isinstance(data, list):
        return []
    return [{"name": f.get("name"), "url": f.get("download_url"), "bytes": f.get("size")}
            for f in data if f.get("download_url")]


def _zenodo(url: str, session: Any) -> List[Dict[str, Any]]:
    match = re.search(r"zenodo\.org/(?:records?|record)/(\d+)", url) or \
        re.search(r"zenodo\.(\d+)", url)
    if not match:
        return []
    data = _json(f"https://zenodo.org/api/records/{match.group(1)}", session)
    if not isinstance(data, dict):
        return []
    out = []
    for f in data.get("files") or []:
        link = (f.get("links") or {}).get("self") or f.get("link")
        if link:
            out.append({"name": f.get("key") or f.get("filename"), "url": link,
                        "bytes": f.get("size")})
    return out


def _hydroshare(url: str, session: Any) -> List[Dict[str, Any]]:
    match = re.search(r"hydroshare\.org/resource/([0-9a-f]{32})", url)
    if not match:
        return []
    data = _json(f"https://www.hydroshare.org/hsapi/resource/{match.group(1)}/files/", session)
    if not isinstance(data, dict):
        return []
    return [{"name": f.get("file_name"), "url": f.get("url"), "bytes": f.get("size")}
            for f in (data.get("results") or []) if f.get("url")]


def _huggingface(url: str, session: Any) -> List[Dict[str, Any]]:
    match = re.search(r"huggingface\.co/datasets/([^/?#]+/[^/?#]+|[^/?#]+)", url)
    if not match:
        return []
    repo = match.group(1)
    data = _json(f"https://huggingface.co/api/datasets/{repo}", session)
    if not isinstance(data, dict):
        return []
    out = []
    for sibling in (data.get("siblings") or []):
        name = sibling.get("rfilename")
        if not name:
            continue
        out.append({"name": name,
                    "url": f"https://huggingface.co/datasets/{repo}/resolve/main/{name}",
                    "bytes": None})
    return out


def _dataverse(url: str, session: Any) -> List[Dict[str, Any]]:
    """Dataverse and the Illinois Data Bank, which speaks the same API."""
    host = re.sub(r"^https?://", "", url).split("/")[0]
    doi = re.search(r"(10\.\d{4,}/[^\s&?#]+)", url)
    if doi:
        data = _json(f"https://{host}/api/datasets/:persistentId?persistentId=doi:"
                     f"{doi.group(1)}", session)
    else:
        ident = re.search(r"/datasets/([A-Za-z0-9\-]+)", url)
        if not ident:
            return []
        data = _json(f"https://{host}/api/datasets/{ident.group(1)}", session)
    if not isinstance(data, dict):
        return []
    latest = ((data.get("data") or {}).get("latestVersion") or {})
    out = []
    for entry in latest.get("files") or []:
        df = entry.get("dataFile") or {}
        fid = df.get("id")
        if fid:
            out.append({"name": df.get("filename"),
                        "url": f"https://{host}/api/access/datafile/{fid}",
                        "bytes": df.get("filesize")})
    return out


def _github_tree(url: str, session: Any) -> List[Dict[str, Any]]:
    """A GitHub directory link: list the tree and return raw URLs for data-looking files."""
    match = re.search(r"github\.com/([^/]+)/([^/]+)(?:/tree/([^/]+)(?:/(.*))?)?", url)
    if not match:
        return []
    owner, repo, ref, path = match.group(1), match.group(2).replace(".git", ""), \
        match.group(3) or "HEAD", (match.group(4) or "").strip("/")
    data = _json(f"https://api.github.com/repos/{owner}/{repo}/git/trees/{ref}?recursive=1",
                 session)
    if not isinstance(data, dict):
        return []
    keep = (".csv", ".geojson", ".json", ".tif", ".tiff", ".shp", ".dbf", ".prj", ".zip",
            ".nc", ".parquet", ".gpkg", ".xlsx", ".txt")
    out = []
    for node in (data.get("tree") or []):
        p = str(node.get("path") or "")
        if node.get("type") != "blob" or not p.lower().endswith(keep):
            continue
        if path and not p.startswith(path):
            continue
        out.append({"name": p.rsplit("/", 1)[-1],
                    "url": f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{p}",
                    "bytes": node.get("size")})
    return out


_RESOLVERS: Tuple[Tuple[str, Any], ...] = (
    ("figshare.com", _figshare),
    ("zenodo.org", _zenodo),
    ("hydroshare.org", _hydroshare),
    ("huggingface.co", _huggingface),
    ("dataverse.harvard.edu", _dataverse),
    ("databank.illinois.edu", _dataverse),
    ("github.com", _github_tree),
)


def resolve_files(url: str, *, session: Any = None, follow_doi: bool = True) -> Dict[str, Any]:
    """``{kind, files, note}`` for a landing-page URL.

    A ``doi.org`` link is followed once, because 6 of the corpus's landing pages are a DOI that
    lands on Zenodo or Dataverse — resolving the DOI is the difference between "no API for
    doi.org" and a file listing.
    """
    import requests

    own = session is None
    session = session or requests.Session()
    try:
        kind = classify_link(url)
        if follow_doi and re.search(r"^https?://(dx\.)?doi\.org/", url or ""):
            try:
                resolved = session.head(url, timeout=30, allow_redirects=True)
                target = resolved.url or url
                if target != url:
                    inner = resolve_files(target, session=session, follow_doi=False)
                    inner["note"] = f"followed DOI to {target[:80]}; " + inner.get("note", "")
                    return inner
            except Exception:
                pass
        if kind == "portal":
            return {"kind": "portal", "files": [],
                    "note": "a data portal's front door, not a deposited file; the element is a "
                            "pointer to a data source"}
        if kind == "opaque":
            return {"kind": "opaque", "files": [],
                    "note": "a file exists but the host offers no public listing API"}
        for host, resolver in _RESOLVERS:
            if host in (url or "").lower():
                files = resolver(url, session)
                return {"kind": "deposit", "files": files,
                        "note": (f"{len(files)} file(s) from the {host} API" if files
                                 else f"{host} API returned no files")}
        return {"kind": kind, "files": [], "note": "no resolver for this host"}
    finally:
        if own:
            session.close()


__all__ = ["resolve_files", "classify_link", "DEPOSIT_HOSTS", "PORTAL_HOSTS", "OPAQUE_HOSTS"]
