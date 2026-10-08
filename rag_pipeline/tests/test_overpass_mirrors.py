"""overpass_search falls back across mirrors, and the default list reaches one that answered.

On 2026-10-08 overpass-api.de, overpass.kumi.systems and lz4.overpass-api.de failed together
(504/500/504) twice in 20 minutes, while overpass.private.coffee answered the same query. Since
stage 38 binds overpass_search in the analyze and code peers, that outage fails ordinary GIS
questions. Offline: ``requests.post`` is patched, nothing leaves the process.
"""
from __future__ import annotations

import pytest
import requests

from rag_pipeline.search import overpass as ov

PRIVATE_COFFEE = "https://overpass.private.coffee/api/interpreter"
MAIL_RU = "https://maps.mail.ru/osm/tools/overpass/api/interpreter"
# The list before stage 40; all three failed together on 2026-10-08.
OLD_THREE = ("https://overpass-api.de/api/interpreter",
             "https://overpass.kumi.systems/api/interpreter",
             "https://lz4.overpass-api.de/api/interpreter")
BBOX = "-87.75,41.76,-87.69,41.805"          # SW Chicago, ~5 km


class _Resp:
    def __init__(self, status: int, payload=None):
        self.status_code = status
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Server Error")

    def json(self):
        return self._payload


_SCHOOL = {"type": "node", "id": 1, "lat": 41.78, "lon": -87.72,
           "tags": {"amenity": "school", "name": "Test School"}}


def _patch_post(monkeypatch, answering: str):
    calls = []

    def fake_post(url, data=None, headers=None, timeout=None):
        calls.append(url)
        if url == answering:
            return _Resp(200, {"elements": [_SCHOOL]})
        return _Resp(504)

    monkeypatch.setattr(ov.requests, "post", fake_post)
    return calls


def test_the_defaults_reach_beyond_the_three_that_failed_together():
    added = set(ov._DEFAULT_ENDPOINTS) - set(OLD_THREE)
    assert {PRIVATE_COFFEE, MAIL_RU} <= added


@pytest.mark.parametrize("answering", [PRIVATE_COFFEE, MAIL_RU])
def test_when_every_other_mirror_fails_the_fallback_reaches_the_new_one(monkeypatch, answering):
    monkeypatch.setattr(ov, "OVERPASS_ENDPOINTS", list(ov._DEFAULT_ENDPOINTS))
    calls = _patch_post(monkeypatch, answering)

    out = ov.overpass_search("school", bbox=BBOX)

    assert "error" not in out, out
    assert out["count"] == 1 and out["features"][0]["name"] == "Test School"
    assert calls[-1] == answering
    assert calls == list(ov._DEFAULT_ENDPOINTS[:len(calls)]), "mirrors are tried in list order"


def test_the_first_mirror_that_answers_ends_the_search(monkeypatch):
    first = ov._DEFAULT_ENDPOINTS[0]
    monkeypatch.setattr(ov, "OVERPASS_ENDPOINTS", list(ov._DEFAULT_ENDPOINTS))
    calls = _patch_post(monkeypatch, first)

    assert ov.overpass_search("school", bbox=BBOX)["count"] == 1
    assert calls == [first]


def test_all_mirrors_failing_is_an_error_payload_not_a_raise(monkeypatch):
    monkeypatch.setattr(ov, "OVERPASS_ENDPOINTS", list(ov._DEFAULT_ENDPOINTS))
    calls = _patch_post(monkeypatch, "https://nowhere.invalid/")

    out = ov.overpass_search("school", bbox=BBOX)

    assert out["error"] == "overpass_failed" and out["count"] == 0
    assert calls == list(ov._DEFAULT_ENDPOINTS)
