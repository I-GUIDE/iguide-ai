"""Overpass keeps its last good answer and serves it, marked, when every mirror fails (stage 70).

2026-10-15 22:55 to 10-16 07:45 CDT all six public mirrors failed for hours: overpass-api.de 502
"at capacity", z. and lz4. timeouts, kumi.systems and private.coffee 500, maps.mail.ru 504/400,
and 429 when the model fired three calls at once. One live Loop road-network turn got through
after 148 s on the fifth mirror. The team demos live; a failed fetch on stage is the risk.

Offline: ``requests.post`` is patched; nothing reaches Overpass. The conftest gives every test
its own OVERPASS_CACHE_DIR.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import pytest

from rag_pipeline.search import overpass as ov

LOOP_BBOX = "-87.645,41.866,-87.618,41.889"


class _Resp:
    def __init__(self, payload, status=200):
        self.status_code, self._payload = status, payload
        self.text = json.dumps(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"{self.status_code}")

    def json(self):
        return self._payload


def _school(i):
    return {"type": "node", "id": i, "lat": 41.87 + i * 1e-4, "lon": -87.63,
            "tags": {"amenity": "school", "name": f"School {i}"}}


def _grid(n=4, x0=-87.645, y0=41.866, step=0.001):
    els, wid = [], 1000
    for i in range(n):
        for j in range(n - 1):
            for horizontal in (True, False):
                a = (x0 + j * step, y0 + i * step) if horizontal else (x0 + i * step, y0 + j * step)
                b = (a[0] + step, a[1]) if horizontal else (a[0], a[1] + step)
                els.append({"type": "way", "id": wid, "tags": {"highway": "residential",
                                                              "name": f"Street {i}"},
                            "geometry": [{"lon": a[0], "lat": a[1]},
                                         {"lon": b[0], "lat": b[1]}]})
                wid += 1
    return els


def _patch(monkeypatch, responses, endpoints=("https://a/api", "https://b/api")):
    """``requests.post`` answers from *responses* in order; an Exception item is raised."""
    sent = []

    def fake_post(url, data=None, headers=None, timeout=None):
        sent.append((url, data["data"]))
        nxt = responses.pop(0) if responses else RuntimeError("502 at capacity")
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    monkeypatch.setattr(ov.requests, "post", fake_post)
    monkeypatch.setattr(ov, "OVERPASS_ENDPOINTS", list(endpoints))
    return sent


def _down(n=20):
    return [RuntimeError("502 This Overpass API server is currently at capacity")] * n


@pytest.fixture
def store(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "store"))
    return tmp_path


# --------------------------------------------------------------------------- the cache itself

def test_a_good_answer_is_kept_keyed_by_the_query_text(monkeypatch):
    sent = _patch(monkeypatch, [_Resp({"elements": [_school(1)]})])
    out = ov.overpass_search("school", bbox=LOOP_BBOX)
    assert out["count"] == 1 and "cached" not in out
    query = sent[0][1]
    path = Path(os.environ["OVERPASS_CACHE_DIR"]) / f"{hashlib.sha256(query.encode()).hexdigest()}.json.gz"
    rec = json.loads(gzip.decompress(path.read_bytes()))
    assert rec["query"] == query and rec["endpoint"] == "https://a/api"
    assert rec["data"]["elements"][0]["id"] == 1 and rec["cached_at"].endswith("+00:00")


def test_every_mirror_down_serves_the_kept_copy_and_says_so(monkeypatch):
    _patch(monkeypatch, [_Resp({"elements": [_school(1), _school(2)]})])
    ov.overpass_search("school", bbox=LOOP_BBOX)

    _patch(monkeypatch, _down())
    out = ov.overpass_search("school", bbox=LOOP_BBOX, debug=True)
    assert "error" not in out and out["count"] == 2
    assert out["cached"] is True and out["cache_reason"] == "mirrors_failed"
    assert out["cached_at"].endswith("+00:00") and out["cache_age_s"] >= 0
    note = out["diagnostics"]["warnings"][0]
    assert note == out["cache_note"]
    assert "CACHED COPY" in note and "every Overpass mirror failed" in note and "UTC" in note
    attempts = out["debug"]["exchanges"][-1]["attempts"]
    assert [a.get("error", "")[:3] for a in attempts[:2]] == ["502", "502"]
    assert attempts[-1]["cache"] == "hit"


def test_no_kept_copy_is_the_same_failure_as_before(monkeypatch):
    _patch(monkeypatch, _down())
    out = ov.overpass_search("school", bbox=LOOP_BBOX)
    assert out["error"] == "overpass_failed" and out["count"] == 0 and "cached" not in out


def test_the_key_leaves_the_mirror_out(monkeypatch):
    _patch(monkeypatch, [_Resp({"elements": [_school(1)]})], endpoints=("https://a/api",))
    ov.overpass_search("school", bbox=LOOP_BBOX)
    _patch(monkeypatch, _down(), endpoints=("https://c/api", "https://d/api"))
    out = ov.overpass_search("school", bbox=LOOP_BBOX)
    assert out["cached"] is True and out["count"] == 1


def test_a_different_query_is_not_served_anothers_answer(monkeypatch):
    _patch(monkeypatch, [_Resp({"elements": [_school(1)]})])
    ov.overpass_search("school", bbox=LOOP_BBOX)
    _patch(monkeypatch, _down())
    assert ov.overpass_search("school", bbox="-87.646,41.866,-87.618,41.889")["error"] == \
        "overpass_failed"
    assert ov.overpass_search("hospital", bbox=LOOP_BBOX)["error"] == "overpass_failed"


def test_by_default_the_mirrors_are_asked_even_with_a_kept_copy(monkeypatch):
    _patch(monkeypatch, [_Resp({"elements": [_school(1)]})])
    ov.overpass_search("school", bbox=LOOP_BBOX)
    sent = _patch(monkeypatch, [_Resp({"elements": [_school(1), _school(3)]})])
    out = ov.overpass_search("school", bbox=LOOP_BBOX)
    assert len(sent) == 1 and out["count"] == 2 and "cached" not in out


def test_a_ttl_serves_a_recent_copy_without_asking_and_marks_it(monkeypatch):
    _patch(monkeypatch, [_Resp({"elements": [_school(1)]})])
    ov.overpass_search("school", bbox=LOOP_BBOX)
    monkeypatch.setenv("OVERPASS_CACHE_TTL_S", "3600")
    sent = _patch(monkeypatch, [_Resp({"elements": []})])
    out = ov.overpass_search("school", bbox=LOOP_BBOX)
    assert sent == [] and out["count"] == 1
    assert out["cached"] is True and out["cache_reason"] == "fresh"
    assert "CACHED COPY" in out["diagnostics"]["warnings"][0]


def _age_entry(seconds):
    from datetime import datetime, timedelta, timezone

    for path in Path(os.environ["OVERPASS_CACHE_DIR"]).glob("*.json.gz"):
        rec = json.loads(gzip.decompress(path.read_bytes()))
        rec["cached_at"] = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat(
            timespec="seconds")
        path.write_bytes(gzip.compress(json.dumps(rec).encode()))


def test_a_copy_past_the_ttl_is_not_used_instead_of_asking(monkeypatch):
    _patch(monkeypatch, [_Resp({"elements": [_school(1)]})])
    ov.overpass_search("school", bbox=LOOP_BBOX)
    _age_entry(7200)
    monkeypatch.setenv("OVERPASS_CACHE_TTL_S", "3600")
    sent = _patch(monkeypatch, [_Resp({"elements": [_school(1), _school(2)]})])
    out = ov.overpass_search("school", bbox=LOOP_BBOX)
    assert len(sent) == 1 and "cached" not in out and out["count"] == 2


def test_the_age_is_stated_and_a_max_age_refuses_older_copies(monkeypatch):
    _patch(monkeypatch, [_Resp({"elements": [_school(1)]})])
    ov.overpass_search("school", bbox=LOOP_BBOX)
    _age_entry(3 * 3600)
    _patch(monkeypatch, _down())
    out = ov.overpass_search("school", bbox=LOOP_BBOX)
    assert out["cached"] is True and "3.0 h old" in out["cache_note"]
    monkeypatch.setenv("OVERPASS_CACHE_MAX_AGE_S", "3600")
    _patch(monkeypatch, _down())
    assert ov.overpass_search("school", bbox=LOOP_BBOX)["error"] == "overpass_failed"


def test_the_cache_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("OVERPASS_CACHE", "0")
    _patch(monkeypatch, [_Resp({"elements": [_school(1)]})])
    ov.overpass_search("school", bbox=LOOP_BBOX)
    assert list(Path(os.environ["OVERPASS_CACHE_DIR"]).glob("*")) == []
    _patch(monkeypatch, _down())
    assert ov.overpass_search("school", bbox=LOOP_BBOX)["error"] == "overpass_failed"


def test_a_damaged_entry_is_no_entry(monkeypatch):
    _patch(monkeypatch, [_Resp({"elements": [_school(1)]})])
    ov.overpass_search("school", bbox=LOOP_BBOX)
    for path in Path(os.environ["OVERPASS_CACHE_DIR"]).glob("*.json.gz"):
        path.write_bytes(b"not gzip")
    _patch(monkeypatch, _down())
    assert ov.overpass_search("school", bbox=LOOP_BBOX)["error"] == "overpass_failed"


def test_the_folder_is_pruned_oldest_first(monkeypatch):
    folder = Path(os.environ["OVERPASS_CACHE_DIR"])
    boxes = ("-87.6,41.8,-87.5,41.9", "-87.7,41.8,-87.6,41.9", "-87.8,41.8,-87.7,41.9")
    _patch(monkeypatch, [_Resp({"elements": [_school(0)]})])
    ov.overpass_search("school", bbox=boxes[0])
    (first,) = folder.glob("*.json.gz")
    # Room for two entries of this size, not three.
    monkeypatch.setenv("OVERPASS_CACHE_MAX_MB", str(first.stat().st_size * 2.5 / 1024 / 1024))
    os.utime(first, (time.time() - 60, time.time() - 60))
    for i, bbox in enumerate(boxes[1:], 1):
        _patch(monkeypatch, [_Resp({"elements": [_school(i)]})])
        ov.overpass_search("school", bbox=bbox)
    left = set(folder.glob("*.json.gz"))
    assert len(left) == 2 and first not in left


# --------------------------------------------------------------------------- where it lives

def test_where_the_cache_lives(monkeypatch, tmp_path):
    monkeypatch.setenv("OVERPASS_CACHE_DIR", str(tmp_path / "explicit"))
    assert ov.cache_dir() == (tmp_path / "explicit").resolve()

    monkeypatch.delenv("OVERPASS_CACHE_DIR")
    monkeypatch.setattr(ov.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "data"))
    assert ov.cache_dir() == (tmp_path / "data").resolve() / "overpass_cache"

    home = tmp_path / "home" / ".cache" / "iguide-agent" / "overpass"
    # A relative root resolves into the repo; so does an absolute one inside this checkout.
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", "./agent_chat_files")
    assert ov.cache_dir() == home
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(ov._REPO_ROOT / "agent_chat_files"))
    if (ov._REPO_ROOT / ".git").exists():
        assert ov.cache_dir() == home
    monkeypatch.delenv("AGENT_FILE_STORAGE_ROOT")
    assert ov.cache_dir() == home

    monkeypatch.setenv("OVERPASS_CACHE", "off")
    assert ov.cache_dir() is None


# --------------------------------------------------------------------------- concurrency

def test_an_identical_query_in_flight_is_sent_once(monkeypatch):
    gate = threading.Event()
    calls = []

    def slow_post(url, data=None, headers=None, timeout=None):
        calls.append(url)
        gate.wait(5)
        return _Resp({"elements": [_school(1)]})

    monkeypatch.setattr(ov.requests, "post", slow_post)
    monkeypatch.setattr(ov, "OVERPASS_ENDPOINTS", ["https://a/api"])
    results = []
    threads = [threading.Thread(target=lambda: results.append(ov._exchange("same query")))
               for _ in range(3)]
    for t in threads:
        t.start()
    time.sleep(0.2)
    gate.set()
    for t in threads:
        t.join(5)
    assert len(calls) == 1
    assert len(results) == 3 and all(r["data"]["elements"][0]["id"] == 1 for r in results)
    assert sum(1 for r in results if r.get("shared")) == 2


def test_at_most_two_requests_are_in_flight(monkeypatch):
    monkeypatch.setenv("OVERPASS_MAX_CONCURRENT", "2")
    lock, live, peak = threading.Lock(), [0], [0]

    def post(url, data=None, headers=None, timeout=None):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.1)
        with lock:
            live[0] -= 1
        return _Resp({"elements": []})

    monkeypatch.setattr(ov.requests, "post", post)
    monkeypatch.setattr(ov, "OVERPASS_ENDPOINTS", ["https://a/api"])
    threads = [threading.Thread(target=ov._exchange, args=(f"query {i}",)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert peak[0] == 2


# --------------------------------------------------------------------------- every path

def test_a_raw_query_is_served_from_the_cache_too(monkeypatch, store):
    from agent_runtime import tool_bridge

    ql = "[out:json];node[amenity=school](41.86,-87.65,41.89,-87.61);out;"
    _patch(monkeypatch, [_Resp({"elements": [_school(1)]})])
    live = tool_bridge._overpass_query_handler(ql)
    assert live["elements"] == 1 and "cached" not in live
    _patch(monkeypatch, _down())
    out = tool_bridge._overpass_query_handler(ql)
    assert out["ok"] is True and out["elements"] == 1
    assert out["cached"] is True and "CACHED COPY" in out["cache_note"]
    assert out["attempts"][-1]["cache"] == "hit"


def _tools():
    from agent_runtime.langchain_granular_tools import (
        make_langchain_granular_tools, make_langchain_osm_tools)

    (peer,) = make_langchain_osm_tools()
    (search,) = [t for t in make_langchain_granular_tools() if t.name == "overpass_search"]
    return {"analyze/code": peer, "search": search}


@pytest.mark.parametrize("which", ["analyze/code", "search"])
def test_a_road_network_survives_an_outage_and_says_it_is_cached(monkeypatch, store, which):
    from agent_runtime import tool_bridge
    from agent_runtime.file_store import resolve_file_id

    monkeypatch.setattr(tool_bridge, "_OVERPASS_RETRY_S", 0)
    _patch(monkeypatch, [_Resp({"elements": _grid()})])
    tool = _tools()[which]
    live = json.loads(tool.invoke({"feature": "road network", "bbox": LOOP_BBOX}))
    assert live["count"] == 24 and "cached" not in live

    _patch(monkeypatch, _down(40))
    out = json.loads(tool.invoke({"feature": "road network", "bbox": LOOP_BBOX}))
    assert "error" not in out and out["count"] == 24 and out["file_id"]
    assert out["cached"] is True and out["cached_tiles"] == 1 and out["cached_at"]
    assert "CACHED COPY" in out["summary"] and "CACHED COPY" in out["documents"][0]["contents"]
    assert "cached" in out["map_layer"]["label"]
    assert out["diagnostics"]["warnings"][0] == out["cache_note"]
    fc = json.loads(Path(resolve_file_id(out["file_id"])).read_text(encoding="utf-8"))
    assert len(fc["features"]) == 24
    if which == "analyze/code":
        assert "Cached copy from" in out["source_statement"]


def test_a_feature_list_says_it_is_cached_in_its_evidence(monkeypatch, store):
    _patch(monkeypatch, [_Resp({"elements": [_school(1)]})])
    _tools()["analyze/code"].invoke({"feature": "school", "bbox": LOOP_BBOX})
    _patch(monkeypatch, _down())
    out = json.loads(_tools()["analyze/code"].invoke({"feature": "school", "bbox": LOOP_BBOX}))
    assert out["cached"] is True and out["count"] == 1
    assert "Cached copy from" in out["documents"][0]["contents"]
    assert "Cached copy from" in out["source_statement"]


def test_the_bridge_feature_fetch_carries_the_mark(monkeypatch, store):
    from agent_runtime import tool_bridge

    monkeypatch.setattr(tool_bridge, "_OVERPASS_RETRY_S", 0)
    bbox = [-87.645, 41.866, -87.618, 41.889]
    _patch(monkeypatch, [_Resp({"elements": [_school(1), _school(2)]})])
    tool_bridge._overpass_handler("school", bbox=bbox)
    _patch(monkeypatch, _down())
    out = tool_bridge._overpass_handler("school", bbox=bbox)
    assert out["count"] == 2 and out["cached"] is True
    assert "CACHED COPY" in out["diagnostics"]["warnings"][0]


def test_the_sources_line_says_cached_whatever_the_answer_says():
    from agent_runtime import source_catalog

    payload = {"count": 2, "cached": True, "cached_at": "2026-10-16T03:12:00+00:00",
               "cache_reason": "mirrors_failed"}
    ((statement, names),) = source_catalog.source_of("overpass_search", {}, json.dumps(payload))
    assert statement.startswith("OpenStreetMap (via Overpass)")
    assert "a cached copy from 2026-10-16 03:12 UTC, not a live fetch" in statement
    ((live, _),) = source_catalog.source_of("overpass_search", {}, json.dumps({"count": 2}))
    assert "cached" not in live
    # A code run that fetched through the bridge names it the same way.
    run = {"bridge_calls": [{"ok": True, "name": "overpass_query", "args": {},
                             "result": payload}]}
    assert any("cached copy" in s for s, _ in source_catalog.source_of("execute_code", {}, run))
