"""The harness saves each turn's map layers while the server is up, named so the replay finds them.

The replay (map-ui-prototype/scripts/replay-server.ts) serves layer NN of a turn from
layers/<turn>__NN__*, NN counting every map_layer event in stream order. So the numbering here
must count a layer it could not fetch, and an inline one, exactly as the page counts them;
`npm run check:replay` holds the page's side of the same contract.
"""
from __future__ import annotations

import json

from gis_harness import screenshots as s


class _Resp:
    def __init__(self, body: bytes, ctype: str, status: int = 200):
        self.content, self.headers, self.status = body, {"content-type": ctype}, status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(self.status)


def _events(tmp_path):
    ev = [
        {"event": "status", "data": {"status": "Started"}},
        {"event": "map_layer", "data": {"id": "a", "label": "OSM: school",
                                        "geojson": {"type": "FeatureCollection", "features": []}}},
        {"event": "map_layer", "data": {"id": "b", "label": "Gone", "url": "/agent/files/file_b/download"}},
        {"event": "agent_trace", "data": {"type": "map_layer",
                                          "detail": {"id": "c", "label": "Slope / dem", "render": "raster",
                                                     "url": "/agent/files/file_c/download"}}},
        {"event": "agent_trace", "data": {"type": "tool_call", "detail": {"name": "x"}}},
    ]
    p = tmp_path / "T06.t1.events.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in ev) + "\nnot json\n")
    return p


def test_map_layer_events_follow_the_client(tmp_path):
    got = s.map_layer_events(_events(tmp_path))
    assert [d["id"] for d in got] == ["a", "b", "c"]


def test_capture_layers_names_by_stream_position(tmp_path, monkeypatch):
    def fake_get(url, timeout):
        assert url.startswith("http://srv:1/agent/files/")
        if "file_b" in url:
            return _Resp(b"", "application/json", 404)
        return _Resp(b"\x89PNG", "image/png")

    monkeypatch.setattr(s.requests, "get", fake_get)
    counts = s.capture_layers("http://srv:1/", _events(tmp_path))
    assert counts == {"saved": 1, "inline": 1, "failed": 1}
    names = sorted(p.name for p in (tmp_path / "layers").iterdir())
    # Layer 2 failed and has no file, but layer 3 is still layer 3.
    assert names == ["T06.t1__01__OSM_school.geojson", "T06.t1__03__Slope_dem.png"]
