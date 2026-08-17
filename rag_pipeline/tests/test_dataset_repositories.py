"""Turning a dataset repository's landing page into the files behind it.

64 of the corpus's 130 datasets resolve to HTML rather than data, and they are TWO problems that
were being reported as one:

* **a deposit behind a landing page** (36) — figshare, Zenodo, HydroShare, Dataverse, Hugging
  Face and GitHub serve a human page at the recorded URL and expose the files through a
  documented API. 24 of these now resolve, recovering among other things 341 heat-stress rasters
  (20 GB) and a 23-part US boundary shapefile.
* **a data portal's front door** (20) — ``fao.org/faostat/#data``, ``who.int/data/#data``,
  ``worldbank.org/indicator``. The element is a pointer to a *website*; there is no file and
  there never was. Calling it a failed download invites someone to hunt a bug that does not
  exist, and loses the fact that the element is still a useful pointer.

The classification is the load-bearing part. No resolver constructs a download URL from a
pattern — each reads a documented listing endpoint, because a guessed link that 404s is
indistinguishable from a dataset that was withdrawn.

Network calls are faked; the URL parsing and the response shapes are what these exercise.
"""

from __future__ import annotations

import json

import pytest

from extractors.dataset_repositories import classify_link, resolve_files


class _Response:
    def __init__(self, payload=None, status=200, url=""):
        self._payload, self.status_code, self.url = payload, status, url

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class _Session:
    """Answers a fixed {url_fragment: payload} map, and records what was asked for."""

    def __init__(self, routes=None, head_to=None):
        self.routes, self.head_to = routes or {}, head_to
        self.asked = []

    def get(self, url, **_kw):
        self.asked.append(url)
        for fragment, payload in self.routes.items():
            if fragment in url:
                return _Response(payload)
        return _Response(status=404)

    def head(self, url, **_kw):
        return _Response(url=self.head_to or url)

    def close(self):
        pass


# ------------------------------------------------------------------ classification

@pytest.mark.parametrize("url,expected", [
    ("https://figshare.com/ndownloader/files/43249983", "deposit"),
    ("https://zenodo.org/records/13822382", "deposit"),
    ("https://www.hydroshare.org/resource/" + "a" * 32 + "/", "deposit"),
    ("https://huggingface.co/datasets/siacus/flourishing", "deposit"),
    ("https://github.com/I-GUIDE/GeoWISE/tree/main/data", "deposit"),
    ("https://www.fao.org/faostat/en/#data", "portal"),
    ("https://www.who.int/data/#data", "portal"),
    ("https://data.worldbank.org/indicator?tab=featured", "portal"),
    ("https://www.mrlc.gov/data", "portal"),
    ("https://drive.google.com/file/d/abc/edit", "opaque"),
    ("https://www.dropbox.com/scl/fo/xyz", "opaque"),
    ("https://storage.i-guide.io/bucket/file.zip", "direct"),
])
def test_a_link_is_classified_by_what_it_actually_is(url, expected):
    assert classify_link(url) == expected


def test_a_portal_is_not_reported_as_a_failure():
    """20 of the corpus's 130 datasets are a portal front door. "Download failed" is the wrong
    description: nothing failed, and the element is still a usable pointer to a data source."""
    result = resolve_files("https://www.fao.org/faostat/en/#data", session=_Session())
    assert result["kind"] == "portal"
    assert result["files"] == []
    assert "pointer to a data source" in result["note"]


def test_an_opaque_host_is_distinguished_from_a_portal():
    """A Google Drive link HAS a file — it just cannot be enumerated without a browser session.
    That is a different fact from "this is a website", and a different fix."""
    result = resolve_files("https://drive.google.com/file/d/abc/edit", session=_Session())
    assert result["kind"] == "opaque"
    assert "no public listing API" in result["note"]


# ------------------------------------------------------------------ the resolvers

def test_figshare_files_come_from_the_articles_api():
    session = _Session({"api.figshare.com/v2/articles/43249983/files": [
        {"name": "sites.csv", "download_url": "https://ndownloader/1", "size": 2048},
        {"name": "readme.txt", "download_url": "https://ndownloader/2", "size": 100}]})
    result = resolve_files("https://figshare.com/articles/dataset/x/43249983", session=session)
    assert result["kind"] == "deposit"
    assert [f["name"] for f in result["files"]] == ["sites.csv", "readme.txt"]
    assert result["files"][0]["url"] == "https://ndownloader/1"


def test_zenodo_files_come_from_the_records_api():
    session = _Session({"zenodo.org/api/records/13822382": {
        "files": [{"key": "data.zip", "size": 5_000_000,
                   "links": {"self": "https://zenodo.org/api/files/x/data.zip"}}]}})
    result = resolve_files("https://zenodo.org/records/13822382", session=session)
    assert result["files"][0]["name"] == "data.zip"
    assert result["files"][0]["bytes"] == 5_000_000


def test_hydroshare_files_come_from_the_hsapi():
    rid = "08745ae940ab4433b6f73de6b79088f7"
    session = _Session({f"hsapi/resource/{rid}/files/": {
        "results": [{"file_name": "flow.csv", "url": "https://hs/x/flow.csv", "size": 12}]}})
    result = resolve_files(f"https://www.hydroshare.org/resource/{rid}/", session=session)
    assert result["files"][0]["name"] == "flow.csv"


def test_a_github_directory_lists_only_data_looking_files():
    """A repo tree holds code, licences and notebooks too. Returning all of them would make the
    'primary member' a README."""
    session = _Session({"git/trees/main": {"tree": [
        {"path": "data/sites.geojson", "type": "blob", "size": 900},
        {"path": "data/notes.md", "type": "blob", "size": 10},
        {"path": "src/run.py", "type": "blob", "size": 50},
        {"path": "data", "type": "tree"}]}})
    result = resolve_files("https://github.com/o/r/tree/main/data", session=session)
    names = [f["name"] for f in result["files"]]
    assert "sites.geojson" in names
    assert "run.py" not in names and "notes.md" not in names


def test_a_doi_is_followed_once_to_whichever_host_it_lands_on():
    """6 of the corpus's landing pages are a DOI that resolves to Zenodo or Dataverse. Without
    following it, the answer is "no resolver for doi.org" and the files are lost."""
    session = _Session(
        {"zenodo.org/api/records/5559791": {
            "files": [{"key": "x.nc", "links": {"self": "https://z/x.nc"}, "size": 7}]}},
        head_to="https://zenodo.org/records/5559791")
    result = resolve_files("https://doi.org/10.5281/zenodo.5559791", session=session)
    assert result["files"][0]["name"] == "x.nc"
    assert "followed DOI" in result["note"]


# ------------------------------------------------------------------ failing honestly

def test_an_api_that_returns_nothing_says_so_rather_than_looking_empty():
    result = resolve_files("https://zenodo.org/records/999", session=_Session())
    assert result["files"] == []
    assert "returned no files" in result["note"]


def test_an_unknown_deposit_host_is_not_guessed_at():
    result = resolve_files("https://data.msdlive.org/records/abc", session=_Session())
    assert result["files"] == [] and "no resolver" in result["note"]


def test_a_malformed_api_response_does_not_raise():
    session = _Session({"api.figshare.com": {"unexpected": "shape"}})
    assert resolve_files("https://figshare.com/articles/dataset/x/123",
                         session=session)["files"] == []
